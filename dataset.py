from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    N_DATES, N_BANDS, HEIGHT, WIDTH,
    S2_DIR, TARGET_DIR, GEOJSON,
)

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Original classes 0-18 are retained. Original 19 is Void.
CLASS_MAPPING = {i: i for i in range(19)}
NUM_CLASSES = 19
IGNORE_INDEX = 19

EXPECTED_S2_SHAPE = (N_DATES, N_BANDS, HEIGHT, WIDTH)
EXPECTED_TARGET_SHAPE = (1, HEIGHT, WIDTH)


def load_metadata(geojson_path=GEOJSON):
    with open(Path(geojson_path), "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson["type"] != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection.")

    metadata = {}
    for feature in geojson["features"]:
        p = feature["properties"]
        sample_id = str(p["id"])
        metadata[sample_id] = {
            "id": sample_id,
            "Fold": int(p["Fold"]),
            "N_Parcel": int(p["N_Parcel"]),
            "dates-S2": p["dates-S2"],
            "ID_PATCH": str(p["ID_PATCH"]),
            "TILE": p.get("TILE"),
            "Parcel_Cover": p.get("Parcel_Cover"),
        }
    return metadata


def build_sample_pairs(s2_dir=S2_DIR, target_dir=TARGET_DIR, metadata=None):
    if metadata is None:
        metadata = load_metadata()

    s2_dir, target_dir = Path(s2_dir), Path(target_dir)
    pairs = []

    for sample_id in sorted(metadata.keys()):
        s2_path = s2_dir / f"S2_{sample_id}.npy"
        target_path = target_dir / f"TARGET_{sample_id}.npy"

        if not s2_path.exists():
            raise FileNotFoundError(s2_path)
        if not target_path.exists():
            raise FileNotFoundError(target_path)

        pairs.append({
            "id": sample_id,
            "s2_path": s2_path,
            "target_path": target_path,
            "Fold": metadata[sample_id]["Fold"],
            "N_Parcel": metadata[sample_id]["N_Parcel"],
            "dates-S2": metadata[sample_id]["dates-S2"],
        })
    return pairs


def get_fold_indices(pairs, validation_fold):
    train_indices, val_indices = [], []
    for idx, sample in enumerate(pairs):
        if sample["Fold"] == validation_fold:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    return train_indices, val_indices


def remap_target(target):
    out = np.full(target.shape, IGNORE_INDEX, dtype=np.uint8)
    for original_id, new_id in CLASS_MAPPING.items():
        out[target == original_id] = new_id
    return out


class PASTISDataset(Dataset):
    def __init__(
        self,
        pairs,
        indices,
        mean,
        std,
        scenario="baseline",
        feature_names=None,
    ):
        self.pairs = pairs
        self.indices = list(indices)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.scenario = scenario
        self.feature_names = feature_names

        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("mean/std must be one-dimensional.")
        if len(self.mean) != len(self.std):
            raise ValueError("mean/std length mismatch.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.pairs[actual_idx]

        image = np.load(sample["s2_path"])
        target = np.load(sample["target_path"])

        if image.shape != EXPECTED_S2_SHAPE:
            raise ValueError(
                f"Unexpected S2 shape {image.shape}; expected {EXPECTED_S2_SHAPE}"
            )
        if target.shape != EXPECTED_TARGET_SHAPE:
            raise ValueError(
                f"Unexpected target shape {target.shape}; expected {EXPECTED_TARGET_SHAPE}"
            )

        from preprocessing import preprocess_image
        image, _ = preprocess_image(
            image,
            scenario=self.scenario,
            selected_features=self.feature_names,
        )

        if image.shape[1] != len(self.mean):
            raise ValueError(
                f"Feature count mismatch: image has {image.shape[1]} "
                f"channels but mean/std have {len(self.mean)}."
            )

        mean = self.mean.reshape(1, -1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1)
        image = (image - mean) / std

        target = remap_target(target[0])

        return (
            torch.from_numpy(image.astype(np.float32)),
            torch.from_numpy(target.astype(np.int64)),
        )


def compute_statistics(
    pairs,
    indices,
    scenario="baseline",
    selected_features=None,
):
    first = True
    sum_x = None
    sum_x2 = None
    count = 0

    for idx in indices:
        sample = pairs[idx]
        image = np.load(sample["s2_path"])
        image, _ = __import__("preprocessing").preprocess_image(
            image,
            scenario=scenario,
            selected_features=selected_features,
        )
        x = image.transpose(1, 0, 2, 3).reshape(image.shape[1], -1).astype(np.float64)

        if first:
            sum_x = np.zeros(x.shape[0], dtype=np.float64)
            sum_x2 = np.zeros(x.shape[0], dtype=np.float64)
            first = False

        sum_x += x.sum(axis=1)
        sum_x2 += np.square(x).sum(axis=1)
        count += x.shape[1]

    mean = sum_x / count
    var = np.maximum(sum_x2 / count - mean ** 2, 0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)
