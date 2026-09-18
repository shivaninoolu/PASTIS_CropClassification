import argparse
from pathlib import Path

import numpy as np

from config import CHECKPOINT_DIR
from dataset import (
    load_metadata,
    build_sample_pairs,
    get_fold_indices,
    NUM_CLASSES,
    IGNORE_INDEX,
)

WEIGHT_CAP = 5.0

USE_SQRT_INVERSE = True

WEIGHT_DIR = Path(CHECKPOINT_DIR).parent / "class_weights"

def count_training_pixels(pairs, train_idx):

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    print("\nCounting training pixels...")

    for n, idx in enumerate(train_idx, start=1):

        pair = pairs[idx]
        target_path = pair["target_path"]
        target = np.load(target_path)
        if target.ndim == 3 and target.shape[0] == 1:
            target = target[0]

        if target.shape != (128, 128):
            raise ValueError(
                f"Unexpected target shape for {target_path}: "
                f"{target.shape}. Expected (1, 128, 128) or (128, 128)."
            )

        target = target.astype(np.int64, copy=False)
        valid = target != IGNORE_INDEX

        if np.any(valid):
            values = target[valid]
            invalid = (values < 0) | (values >= NUM_CLASSES)

            if np.any(invalid):
                bad_values = np.unique(values[invalid])
                raise ValueError(
                    f"Invalid class IDs found in {target_path}: "
                    f"{bad_values.tolist()}"
                )

            fold_counts = np.bincount(
                values,
                minlength=NUM_CLASSES
            )

            counts += fold_counts

        if n % 10 == 0 or n == len(train_idx):
            print(
                f"  Processed {n}/{len(train_idx)} training samples"
            )

    return counts


# ============================================================
# CALCULATE WEIGHTS
# ============================================================

def calculate_weights(counts):

    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    present = counts > 0
    if not np.any(present):
        raise ValueError(
            "No valid training pixels were found."
        )

    frequencies = counts[present].astype(np.float64)
    
    if USE_SQRT_INVERSE:
        raw_weights = 1.0 / np.sqrt(frequencies)
    else:
        raw_weights = 1.0 / frequencies
    raw_weights /= raw_weights.mean()
    raw_weights = np.minimum(
        raw_weights,
        WEIGHT_CAP
    )
    raw_weights /= raw_weights.mean()

    weights[present] = raw_weights.astype(np.float32)

    return weights


def print_weight_table(counts, weights):
    total = counts.sum()

    print("\n" + "=" * 72)
    print("CLASS WEIGHTS")
    print("=" * 72)

    print(
        f"{'Class':>8} "
        f"{'Pixels':>15} "
        f"{'Percentage':>12} "
        f"{'Weight':>12}"
    )

    print("-" * 72)

    for c in range(NUM_CLASSES):

        pixels = counts[c]

        if total > 0:
            percentage = 100.0 * pixels / total
        else:
            percentage = 0.0

        print(
            f"{c:>8} "
            f"{pixels:>15,} "
            f"{percentage:>11.4f}% "
            f"{weights[c]:>12.4f}"
        )

    print("-" * 72)

    print(
        f"{'TOTAL':>8} "
        f"{total:>15,}"
    )

    print("=" * 72)

def process_fold(pairs, fold):
    print("\n" + "=" * 72)
    print(f"FOLD {fold}")
    print("=" * 72)

    train_idx, val_idx = get_fold_indices(
        pairs,
        fold
    )

    print(f"Training samples   : {len(train_idx)}")
    print(f"Validation samples : {len(val_idx)}")

    counts = count_training_pixels(
        pairs,
        train_idx
    )

    weights = calculate_weights(
        counts
    )

    print_weight_table(
        counts,
        weights
    )


    WEIGHT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        WEIGHT_DIR /
        f"class_weights_fold_{fold}.npy"
    )

    np.save(
        output_path,
        weights
    )

    print(
        f"\nSaved class weights to:\n"
        f"  {output_path}"
    )

    return counts, weights


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate fold-specific class weights for PASTIS."
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold to process (1-5). "
            "If omitted, all five folds are processed."
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print("PASTIS FOLD-SPECIFIC CLASS WEIGHTS")
    print("=" * 72)

    metadata = load_metadata()

    pairs = build_sample_pairs(
        metadata=metadata
    )

    print(f"\nTotal samples: {len(pairs)}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Ignore index: {IGNORE_INDEX}")
    print(f"Weight cap: {WEIGHT_CAP}")
    print(
        "Weighting: square-root inverse frequency"
        if USE_SQRT_INVERSE
        else "Weighting: inverse frequency"
    )

    if args.fold is not None:

        if args.fold < 1 or args.fold > 5:
            raise ValueError(
                "--fold must be between 1 and 5."
            )

        process_fold(
            pairs,
            args.fold
        )

    else:

        for fold in range(1, 6):
            process_fold(
                pairs,
                fold
            )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"\nClass-weight files are stored in:\n"
        f"  {WEIGHT_DIR}"
    )


if __name__ == "__main__":
    main()