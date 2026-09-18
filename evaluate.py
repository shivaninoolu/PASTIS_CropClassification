import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, EVALUATION_DIR, CHECKPOINT_DIR, SCENARIOS
from dataset import (
    load_metadata, build_sample_pairs, get_fold_indices,
    PASTISDataset, NUM_CLASSES, IGNORE_INDEX,
)
from model import build_model

CLASS_NAMES = [
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits vegetables flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
]


def confusion_matrix_update(cm, pred, target):
    valid = target != IGNORE_INDEX
    p = pred[valid].reshape(-1)
    y = target[valid].reshape(-1)

    for yy, pp in zip(y.tolist(), p.tolist()):
        if 0 <= yy < NUM_CLASSES and 0 <= pp < NUM_CLASSES:
            cm[yy, pp] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="baseline")
    parser.add_argument(
        "--save_probs",
        action="store_true",
        default=True,
        help="Save per-class softmax probability maps alongside predictions (default: on).",
    )
    parser.add_argument(
        "--no_save_probs",
        dest="save_probs",
        action="store_false",
        help="Disable saving probability maps (predictions/metrics are still saved).",
    )
    parser.add_argument(
        "--prob_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for saved probability maps. float16 halves disk usage.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = CHECKPOINT_DIR / f"best_{args.scenario}_fold_{args.fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py for this scenario/fold first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    metadata = load_metadata()
    pairs = build_sample_pairs(metadata=metadata)
    _, val_idx = get_fold_indices(pairs, args.fold)

    selected_features = checkpoint.get("selected_features")
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    n_bands = checkpoint["n_bands"]

    ds = PASTISDataset(
        pairs, val_idx, mean, std,
        scenario=args.scenario,
        feature_names=selected_features,
    )
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model_name = SCENARIOS[args.scenario]["model"]
    model = build_model(
        model_name, n_bands=n_bands,
        n_classes=NUM_CLASSES,
        temporal_features=32,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    prediction_dir = EVALUATION_DIR / args.scenario / f"fold_{args.fold}" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    probability_dir = None
    prob_dtype = np.float16 if args.prob_dtype == "float16" else np.float32
    if args.save_probs:
        probability_dir = EVALUATION_DIR / args.scenario / f"fold_{args.fold}" / "probabilities"
        probability_dir.mkdir(parents=True, exist_ok=True)

    total_correct = 0
    total_valid = 0

    with torch.no_grad():
        offset = 0
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            probs_np = probs.cpu().numpy()
            targets_np = targets.numpy()

            for b in range(len(preds)):
                confusion_matrix_update(cm, preds[b], targets_np[b])

                valid = targets_np[b] != IGNORE_INDEX
                total_correct += int((preds[b][valid] == targets_np[b][valid]).sum())
                total_valid += int(valid.sum())

                sample_id = pairs[val_idx[offset]]["id"]
                np.save(
                    prediction_dir / f"prediction_{sample_id}.npy",
                    preds[b].astype(np.uint8),
                )

                if probability_dir is not None:
                    np.save(
                        probability_dir / f"prob_{sample_id}.npy",
                        probs_np[b].astype(prob_dtype),
                    )

                offset += 1

    ious = []
    per_class_iou = []
    per_class_accuracy = []
    
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
    
        union = tp + fp + fn
        actual_pixels = tp + fn
    
        # IoU
        iou = np.nan if union == 0 else tp / union
    
        # Class accuracy / recall
        class_accuracy = (
            np.nan if actual_pixels == 0
            else tp / actual_pixels
        )
    
        per_class_iou.append(iou)
        per_class_accuracy.append(class_accuracy)
    
        if not np.isnan(iou):
            ious.append(iou)
    
    accuracy = total_correct / max(total_valid, 1)
    
    miou = float(np.mean(ious)) if ious else 0.0
    
    out_dir = EVALUATION_DIR / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = out_dir / f"fold_{args.fold}_metrics.csv"
    
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
    
        writer.writerow([
            "scenario",
            "fold",
            "overall_accuracy",
            "miou",
            "checkpoint_epoch"
        ])
    
        writer.writerow([
            args.scenario,
            args.fold,
            accuracy,
            miou,
            checkpoint["epoch"]
        ])
    
    per_class_path = out_dir / f"fold_{args.fold}_per_class_metrics.csv"
    
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
    
        writer.writerow([
            "scenario",
            "fold",
            "class_id",
            "class_name",
            "iou",
            "class_accuracy"
        ])
    
        for c in range(NUM_CLASSES):
    
            iou = per_class_iou[c]
            class_accuracy = per_class_accuracy[c]
    
            writer.writerow([
                args.scenario,
                args.fold,
                c,
                CLASS_NAMES[c],
                "" if np.isnan(iou) else iou,
                "" if np.isnan(class_accuracy) else class_accuracy
            ])
    
    np.savetxt(
        out_dir / f"fold_{args.fold}_confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )
    
    print("=" * 70)
    print(f"EVALUATION — {args.scenario} — FOLD {args.fold}")
    print("=" * 70)
    
    print(f"Overall Accuracy : {accuracy:.4f}")
    print(f"mIoU             : {miou:.4f}")
    print(f"Checkpoint epoch : {checkpoint['epoch']}")
    print(f"Predictions      : {prediction_dir}")
    print(f"Metrics saved    : {metrics_path}")
    print(f"Per-class saved  : {per_class_path}")


if __name__ == "__main__":
    main()
