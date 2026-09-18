import numpy as np

from config import (
    CLOUD_PERCENTILE,
    CLOUD_NDVI_MAX,
    WHOLE_DATE_CLOUD_FRACTION,
    INDEX_NAMES,
    SCENARIOS,
)
from dataset import S2_BANDS


def _safe_percentile(x, q, axis=None):
    return np.nanpercentile(x, q, axis=axis)


def detect_cloud_like_pixels(image):
    """
    Heuristic only: there is no cloud mask in this subset.

    Uses:
      - high visible brightness (B2/B3/B4)
      - low NDVI
      - temporal anomaly of visible brightness

    Returns:
      pixel_mask: (T,H,W), True = suspicious/cloud-like
      date_fraction: (T,)
    """
    x = image.astype(np.float32, copy=False)
    B2, B3, B4, B8 = x[:, 0], x[:, 1], x[:, 2], x[:, 6]

    # Normalize visible brightness per date to its spatial distribution.
    vis = (B2 + B3 + B4) / 3.0
    date_thr = np.nanpercentile(vis, CLOUD_PERCENTILE, axis=(1, 2), keepdims=True)
    bright_spatial = vis >= date_thr

    # Per-pixel temporal upper percentile.
    temporal_thr = np.nanpercentile(vis, CLOUD_PERCENTILE, axis=0, keepdims=True)
    bright_temporal = vis >= temporal_thr

    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)

    # Require brightness + low vegetation signal.
    mask = bright_spatial & bright_temporal & (ndvi < CLOUD_NDVI_MAX)

    # Very small isolated detections are less concerning; keep the mask
    # conservative. No morphology dependency is required.
    date_fraction = mask.reshape(mask.shape[0], -1).mean(axis=1)

    return mask, date_fraction


def temporal_interpolate(image, invalid_mask):
    """
    Replace invalid pixels by linear interpolation over valid dates.
    If a pixel is invalid at the beginning/end, use nearest valid date.
    If a pixel has no valid date, use the temporal median.
    """
    T, C, H, W = image.shape
    out = image.astype(np.float32, copy=True)

    # Vectorized over H x W. We only loop over dates, not over pixels.
    valid = ~invalid_mask
    date_ids = np.arange(T, dtype=np.int32)[:, None, None]

    prev_idx = np.maximum.accumulate(
        np.where(valid, date_ids, -1), axis=0
    )
    next_idx = np.minimum.accumulate(
        np.where(valid, date_ids, T), axis=0
    )[::-1][::-1]

    # The reverse cumulative minimum above is easiest to express explicitly.
    next_idx = np.minimum.accumulate(
        np.where(valid, date_ids, T)[::-1], axis=0
    )[::-1]

    rows, cols = np.indices((H, W))

    for t in range(T):
        bad = invalid_mask[t]
        if not bad.any():
            continue

        p = prev_idx[t]
        n = next_idx[t]

        only_next = bad & (p < 0) & (n < T)
        only_prev = bad & (p >= 0) & (n >= T)
        both = bad & (p >= 0) & (n < T)
        neither = bad & (p < 0) & (n >= T)

        for c in range(C):
            arr = out[:, c]
            vals = arr[t]

            if only_next.any():
                rr, cc = np.where(only_next)
                vals[rr, cc] = arr[n[rr, cc], rr, cc]

            if only_prev.any():
                rr, cc = np.where(only_prev)
                vals[rr, cc] = arr[p[rr, cc], rr, cc]

            if both.any():
                rr, cc = np.where(both)
                pp = p[rr, cc]
                nn = n[rr, cc]
                alpha = (t - pp) / np.maximum(nn - pp, 1)
                vals[rr, cc] = (
                    arr[pp, rr, cc] * (1.0 - alpha)
                    + arr[nn, rr, cc] * alpha
                )

            if neither.any():
                median = np.nanmedian(arr[:, neither], axis=0)
                rr, cc = np.where(neither)
                vals[rr, cc] = median

            out[t, c] = vals

    return out


def calculate_indices(image, names=None):
    """image: (T,C,H,W), C must contain the 10 Sentinel-2 bands."""
    if names is None:
        names = INDEX_NAMES

    B2, B3, B4 = image[:, 0], image[:, 1], image[:, 2]
    B5 = image[:, 3]
    B6, B7, B8 = image[:, 4], image[:, 5], image[:, 6]
    B11, B12 = image[:, 8], image[:, 9]

    out = []
    for name in names:
        if name == "NDVI":
            v = (B8 - B4) / (B8 + B4 + 1e-6)
        elif name == "EVI":
            denominator = B8 + 6 * B4 - 7.5 * B2 + 1.0

            v = np.divide(
                2.5 * (B8 - B4),
                denominator,
                out=np.zeros_like(B8, dtype=np.float32),
                where=np.abs(denominator) > 1e-10
            )
        elif name == "LSWI":
            v = (B8 - B11) / (B8 + B11 + 1e-6)
        elif name == "NDRE":
            v = (B8 - B5) / (B8 + B5 + 1e-6)
        elif name == "NDWI":
            v = (B3 - B8) / (B3 + B8 + 1e-6)
        elif name == "SAVI":
            v = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)
        else:
            raise ValueError(f"Unknown index: {name}")

        out.append(v.astype(np.float32))

    return np.stack(out, axis=1)


def preprocess_image(image, scenario="baseline", selected_features=None):
    """
    Returns:
        processed: (T,C,H,W)
        cloud_mask: (T,H,W)

    Cloud-like pixels are temporally interpolated rather than deleting
    dates, preserving the fixed 46-date temporal sequence.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    x = image.astype(np.float32, copy=False)
    cloud_mask, date_fraction = detect_cloud_like_pixels(x)

    # If a whole date is strongly contaminated, mark every pixel invalid.
    severe_dates = date_fraction >= WHOLE_DATE_CLOUD_FRACTION
    if severe_dates.any():
        cloud_mask[severe_dates] = True

    x_clean = temporal_interpolate(x, cloud_mask)

    if SCENARIOS[scenario]["use_indices"]:
        idx = calculate_indices(x_clean, INDEX_NAMES)
        features = np.concatenate([x_clean, idx], axis=1)
        names = S2_BANDS + INDEX_NAMES
    else:
        features = x_clean
        names = S2_BANDS

    if selected_features is not None:
        name_to_idx = {n: i for i, n in enumerate(names)}
        missing = [n for n in selected_features if n not in name_to_idx]
        if missing:
            raise ValueError(f"Selected features not available: {missing}")
        features = features[:, [name_to_idx[n] for n in selected_features]]
        names = selected_features

    return features.astype(np.float32), cloud_mask


def inspect_cloud_statistics(image):
    mask, fractions = detect_cloud_like_pixels(image)
    return {
        "mean_cloud_like_fraction": float(fractions.mean()),
        "max_cloud_like_fraction": float(fractions.max()),
        "dates_above_10pct": int((fractions > 0.10).sum()),
        "date_fractions": fractions,
    }
