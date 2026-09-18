from pathlib import Path
import json
import numpy as np

from config import (
    FEATURE_DIR,
    CORRELATION_THRESHOLD,
    MAX_SELECTED_FEATURES,
    INDEX_NAMES,
)
from dataset import NUM_CLASSES, IGNORE_INDEX, remap_target, S2_BANDS
from preprocessing import preprocess_image


def _feature_matrix_for_sample(image, target, scenario):
    features, _ = preprocess_image(image, scenario=scenario)

    features = np.nanmean(features, axis=0)
    X = features.transpose(1, 2, 0).reshape(-1, features.shape[0])
    y = remap_target(target[0]).reshape(-1)
    valid = y != IGNORE_INDEX
    valid &= np.all(np.isfinite(X), axis=1)

    X = X[valid]
    y = y[valid]

    return X, y


def fisher_scores(X, y, n_classes=NUM_CLASSES):
    scores = np.zeros(X.shape[1], dtype=np.float64)
    global_mean = np.mean(X, axis=0)

    for c in range(n_classes):
        m = y == c
        if m.sum() < 2:
            continue
        xc = X[m]
        mean_c = xc.mean(axis=0)
        var_c = xc.var(axis=0) + 1e-8
        scores += m.sum() * (mean_c - global_mean) ** 2 / var_c

    return scores / max(len(y), 1)


def select_features(
    pairs,
    train_indices,
    scenario,
    fold,
    output_dir=FEATURE_DIR,
):
    """
    Feature selection uses TRAINING FOLD ONLY.
    For baseline scenario this function is not needed.
    """
    if scenario not in ("utae_selected", "utae_indices"):
        return S2_BANDS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_y = []

    for idx in train_indices:
        sample = pairs[idx]
        image = np.load(sample["s2_path"])
        target = np.load(sample["target_path"])
        X, y = _feature_matrix_for_sample(image, target, scenario)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    rng = np.random.default_rng(42)
    max_pixels = min(len(y), 500_000)
    if len(y) > max_pixels:
        ids = rng.choice(len(y), size=max_pixels, replace=False)
        X, y = X[ids], y[ids]

    names = S2_BANDS + (INDEX_NAMES if scenario == "utae_indices" else [])
    scores = fisher_scores(X, y)
    ranking = np.argsort(scores)[::-1]
    corr = np.corrcoef(X, rowvar=False)
    selected = []

    for i in ranking:
        if not np.isfinite(scores[i]):
            continue

        if all(abs(corr[i, j]) < CORRELATION_THRESHOLD for j in selected):
            selected.append(int(i))

        max_features = MAX_SELECTED_FEATURES[scenario]
        if max_features is not None and len(selected) >= max_features:
            break

    selected_names = [names[i] for i in selected]
    ranking_rows = [
        {
            "feature": names[int(i)],
            "fisher_score": float(scores[int(i)]),
            "selected": int(i) in selected,
        }
        for i in ranking
    ]

    out_file = output_dir / f"{scenario}_fold_{fold}_feature_selection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario,
                "selected_features": selected_names,
                "ranking": ranking_rows,
                "correlation_threshold": CORRELATION_THRESHOLD,
                "max_selected_features": MAX_SELECTED_FEATURES[scenario],
                "n_training_pixels_used": int(len(y)),
            },
            f,
            indent=2,
        )

    print("\nFEATURE SELECTION")
    print("-" * 60)
    for row in ranking_rows:
        print(
            f"{row['feature']:>6}  "
            f"Fisher={row['fisher_score']:.6f}  "
            f"{'SELECTED' if row['selected'] else ''}"
        )

    print("\nSelected features:")
    print(selected_names)

    return selected_names


def load_selected_features(scenario, fold, output_dir=FEATURE_DIR):
    path = Path(output_dir) / f"{scenario}_fold_{fold}_feature_selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run feature selection for the training fold first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["selected_features"]
