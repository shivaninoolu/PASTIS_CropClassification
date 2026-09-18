import argparse
import csv

import numpy as np

from config import ROOT1
from dataset import load_metadata, build_sample_pairs
from preprocessing import inspect_cloud_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()

    metadata = load_metadata()
    pairs = build_sample_pairs(metadata=metadata)

    if args.sample_id:
        pairs = [p for p in pairs if p["id"] == str(args.sample_id)]
        if not pairs:
            raise ValueError(f"Sample {args.sample_id} not found.")
    else:
        pairs = pairs[:args.max_samples]

    rows = []

    for sample in pairs:
        image = np.load(sample["s2_path"])
        stats = inspect_cloud_statistics(image)

        print("\nSample:", sample["id"])
        print("Mean cloud-like fraction:", stats["mean_cloud_like_fraction"])
        print("Max date fraction:", stats["max_cloud_like_fraction"])
        print("Dates >10%:", stats["dates_above_10pct"])
        print("Per-date fractions:")
        print(np.round(stats["date_fractions"], 4))

        for date_idx, frac in enumerate(stats["date_fractions"]):
            rows.append([sample["id"], date_idx, float(frac)])

    out = ROOT1 / "cloud_diagnostic.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "date_index", "cloud_like_fraction"])
        writer.writerows(rows)

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
