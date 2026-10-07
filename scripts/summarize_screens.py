#!/usr/bin/env python3
"""Rank completed proxy runs by best official PQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/screens")
    args = parser.parse_args()
    rows = []
    for path in Path(args.root).glob("*/metrics/full_pq_epoch*.json"):
        report = json.loads(path.read_text())
        rows.append({
            "candidate": path.parent.parent.name,
            "epoch": int(report["epoch"]),
            "pq": float(report["official_pq"]),
            "tp": int(report["hit_pairs"]),
            "fp": int(report["false_positives"]),
            "fn": int(report["false_negatives"]),
            "seconds_per_image": float(report["seconds_per_physical_image"]),
            "gpu_peak_mb": float(report["inference_gpu_peak_mb"]),
        })
    best = {}
    for row in rows:
        current = best.get(row["candidate"])
        if current is None or row["pq"] > current["pq"]:
            best[row["candidate"]] = row
    for row in sorted(best.values(), key=lambda item: item["pq"], reverse=True):
        print(
            f"{row['candidate']:18s} PQ={row['pq']:.6f} epoch={row['epoch']} "
            f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']} "
            f"sec/image={row['seconds_per_image']:.3f} GPU={row['gpu_peak_mb']:.0f}MiB"
        )
    if not best:
        raise SystemExit("no completed screen reports found")


if __name__ == "__main__":
    main()
