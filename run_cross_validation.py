import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from config import SCENARIOS, EVALUATION_DIR, CHECKPOINT_DIR


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent

    results = []

    for fold in range(1, 6):
        print("\n" + "=" * 80)
        print(f"TRAINING SCENARIO={args.scenario}, FOLD={fold}")
        print("=" * 80)

        ckpt_path = CHECKPOINT_DIR / f"best_baseline_fold_{fold}.pth"

        if ckpt_path.exists():
            print(f"Checkpoint already exists:")
            print(ckpt_path)
            print(f"Skipping training for Fold {fold}.")
        
        else:
            train_cmd = [
                sys.executable, str(src_dir / "train.py"),
                "--fold", str(fold),
                "--scenario", args.scenario,
            ]
        
            if args.epochs is not None:
                train_cmd += ["--epochs", str(args.epochs)]
            if args.batch_size is not None:
                train_cmd += ["--batch_size", str(args.batch_size)]
            if args.lr is not None:
                train_cmd += ["--lr", str(args.lr)]
        
            subprocess.run(train_cmd, check=True)

        print("\n" + "=" * 80)
        print(f"EVALUATING SCENARIO={args.scenario}, FOLD={fold}")
        print("=" * 80)

        eval_cmd = [
            sys.executable,
            str(src_dir / "evaluate.py"),
            "--fold", str(fold),
            "--scenario", args.scenario,
        ]

        subprocess.run(eval_cmd, check=True)

        metrics_file = (
            EVALUATION_DIR
            / args.scenario
            / f"fold_{fold}_metrics.csv"
        )

        with open(metrics_file, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))

            fold_result = {
                "fold": fold,
                "accuracy": float(row["overall_accuracy"]),
                "miou": float(row["miou"]),
                "checkpoint_epoch": int(row["checkpoint_epoch"]),
            }
        per_class_file = (
            EVALUATION_DIR
            / args.scenario
            / f"fold_{fold}_per_class_metrics.csv"
        )

        with open(per_class_file, newline="", encoding="utf-8") as f:
            per_class_rows = list(csv.DictReader(f))

        for row in per_class_rows:
            class_id = int(row["class_id"])
            class_name = row["class_name"]

            column_name = (
                f"class_{class_id}_{class_name}"
                .replace(" ", "_")
                .replace("/", "_")
            )

            if row["class_accuracy"] == "":
                fold_result[column_name] = np.nan
            else:
                fold_result[column_name] = float(row["class_accuracy"])

        results.append(fold_result)

    out_dir = EVALUATION_DIR / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_results_path = out_dir / "cross_validation_results.csv"

    accuracies = [r["accuracy"] for r in results]
    mious = [r["miou"] for r in results]
    epochs = [r["checkpoint_epoch"] for r in results]

    per_class_columns = []

    for r in results:
        for key in r.keys():
            if key.startswith("class_") and key not in per_class_columns:
                per_class_columns.append(key)

    per_class_columns.sort(
        key=lambda x: int(x.split("_")[1])
    )

    with open(
        cv_results_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "scenario",
            "fold",
            "overall_accuracy",
            "miou",
            "checkpoint_epoch",
        ] + per_class_columns)

        for r in results:

            writer.writerow([
                args.scenario,
                r["fold"],
                r["accuracy"],
                r["miou"],
                r["checkpoint_epoch"],
            ] + [
                r.get(class_col, "")
                for class_col in per_class_columns
            ])

        writer.writerow([
            args.scenario,
            "MEAN",
            np.mean(accuracies),
            np.mean(mious),
            np.mean(epochs),
        ] + [
            np.nanmean([
                r.get(class_col, np.nan)
                for r in results
            ])
            for class_col in per_class_columns
        ])

        writer.writerow([
            args.scenario,
            "STD",
            np.std(accuracies),
            np.std(mious),
            np.std(epochs),
        ] + [
            np.nanstd([
                r.get(class_col, np.nan)
                for r in results
            ])
            for class_col in per_class_columns
        ])

    print("\n" + "=" * 80)
    print(f"5-FOLD CROSS-VALIDATION COMPLETE: {args.scenario}")
    print("=" * 80)

    for r in results:
        print(
            f"Fold {r['fold']}: "
            f"Overall Accuracy={r['accuracy']:.4f}, "
            f"mIoU={r['miou']:.4f}, "
            f"best epoch={r['checkpoint_epoch']}"
        )

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_miou = np.mean(mious)
    std_miou = np.std(mious)

    print(
        f"\nMean Overall Accuracy = "
        f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
    )

    print(
        f"Mean mIoU             = "
        f"{mean_miou:.4f} ± {std_miou:.4f}"
    )

    print(f"\nResults saved to: {cv_results_path}")

if __name__ == "__main__":
    run()
