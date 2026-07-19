#!/usr/bin/env python
"""Compute normalized DTW for residual-flow samples and word priors."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from flow.smplx_features import (
    COMPACT_EXPRESSION,
    COMPACT_JAW,
    COMPACT_LEFT_HAND,
    COMPACT_RIGHT_HAND,
    COMPACT_UPPER_BODY,
)


PARTS = {
    "whole": slice(0, 133),
    "upper_body": COMPACT_UPPER_BODY,
    "left_hand": COMPACT_LEFT_HAND,
    "right_hand": COMPACT_RIGHT_HAND,
    "hands": slice(COMPACT_LEFT_HAND.start, COMPACT_RIGHT_HAND.stop),
    "face": slice(COMPACT_JAW.start, COMPACT_EXPRESSION.stop),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute DTW and normalized DTW distances between GT motion, flow output, "
            "and residual word-concatenation prior saved by flow.sample_text_conditional."
        )
    )
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None, help="Dataset directory containing meta/mean.npy and meta/std.npy.")
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--sample_glob", default="sample_*.npz")
    parser.add_argument("--sample_key", default="motion")
    parser.add_argument("--gt_key", default="motion")
    parser.add_argument("--prior_key", default="coarse_motion")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--no_normalize", action="store_true", help="Use raw compact SMPL-X features instead of dataset-standardized features.")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["whole", "upper_body", "left_hand", "right_hand", "hands", "face"],
        choices=sorted(PARTS),
    )
    return parser.parse_args()


def load_motion(path, key):
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path}: missing key {key!r}; available keys: {data.files}")
        motion = data[key].astype(np.float32)
    if motion.ndim != 2:
        raise ValueError(f"{path}: {key!r} must be [T, D], got {motion.shape}")
    return motion


def load_stats(data_dir, dim):
    if data_dir is None:
        return None, None
    mean_path = data_dir / "meta" / "mean.npy"
    std_path = data_dir / "meta" / "std.npy"
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"Missing mean/std under {data_dir / 'meta'}")
    mean = np.load(mean_path).astype(np.float32).reshape(1, -1)
    std = np.load(std_path).astype(np.float32).reshape(1, -1)
    if mean.shape[1] != dim or std.shape[1] != dim:
        raise ValueError(f"Stats dim mismatch: mean={mean.shape}, std={std.shape}, motion_dim={dim}")
    return mean, np.maximum(std, 1e-6)


def standardize(motion, mean, std):
    if mean is None or std is None:
        return motion.astype(np.float32, copy=False)
    return ((motion - mean) / std).astype(np.float32, copy=False)


def frame_distance_matrix(a, b):
    # Root-mean-square feature distance per aligned frame pair.
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=-1, dtype=np.float64), dtype=np.float64)


def dtw_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return {
            "dtw": math.inf,
            "path_len": 0,
            "ndtw": math.inf,
            "ndtw_ref": math.inf,
        }

    dist = frame_distance_matrix(a, b)
    n, m = dist.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    plen = np.zeros((n + 1, m + 1), dtype=np.int32)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = (
                (acc[i - 1, j], plen[i - 1, j]),
                (acc[i, j - 1], plen[i, j - 1]),
                (acc[i - 1, j - 1], plen[i - 1, j - 1]),
            )
            best_acc, best_len = min(candidates, key=lambda item: item[0])
            acc[i, j] = dist[i - 1, j - 1] + best_acc
            plen[i, j] = best_len + 1

    total = float(acc[n, m])
    path_len = int(plen[n, m])
    return {
        "dtw": total,
        "path_len": path_len,
        "ndtw": total / max(path_len, 1),
        "ndtw_ref": total / max(n, 1),
    }


def pair_files(samples_dir, sample_glob):
    sample_files = sorted(samples_dir.glob(sample_glob))
    pairs = []
    for sample_path in sample_files:
        suffix = sample_path.stem.replace("sample_", "", 1)
        gt_path = samples_dir / f"gt_{suffix}.npz"
        if gt_path.is_file():
            pairs.append((suffix, gt_path, sample_path))
    return pairs


def summarize(rows):
    summary = {}
    keys = sorted({(row["comparison"], row["part"]) for row in rows})
    for comparison, part in keys:
        subset = [row for row in rows if row["comparison"] == comparison and row["part"] == part]
        ndtw = np.asarray([row["ndtw"] for row in subset], dtype=np.float64)
        ndtw_ref = np.asarray([row["ndtw_ref"] for row in subset], dtype=np.float64)
        dtw = np.asarray([row["dtw"] for row in subset], dtype=np.float64)
        summary[f"{comparison}/{part}"] = {
            "count": int(len(subset)),
            "dtw_mean": float(dtw.mean()),
            "dtw_std": float(dtw.std()),
            "ndtw_mean": float(ndtw.mean()),
            "ndtw_std": float(ndtw.std()),
            "ndtw_median": float(np.median(ndtw)),
            "ndtw_ref_mean": float(ndtw_ref.mean()),
            "ndtw_ref_std": float(ndtw_ref.std()),
            "ndtw_ref_median": float(np.median(ndtw_ref)),
        }
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "comparison",
        "part",
        "gt_len",
        "pred_len",
        "dtw",
        "path_len",
        "ndtw",
        "ndtw_ref",
        "sample",
        "gt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    pairs = pair_files(args.samples_dir, args.sample_glob)
    if args.max_items > 0:
        pairs = pairs[: args.max_items]
    if not pairs:
        raise RuntimeError(f"No sample/gt pairs found under {args.samples_dir}")

    first_gt = load_motion(pairs[0][1], args.gt_key)
    mean, std = (None, None) if args.no_normalize else load_stats(args.data_dir, first_gt.shape[1])

    rows = []
    skipped_prior = 0
    for index, gt_path, sample_path in pairs:
        gt = standardize(load_motion(gt_path, args.gt_key), mean, std)
        sample = standardize(load_motion(sample_path, args.sample_key), mean, std)
        try:
            prior = standardize(load_motion(sample_path, args.prior_key), mean, std)
        except KeyError:
            prior = None
            skipped_prior += 1

        comparisons = [("flow", sample)]
        if prior is not None:
            comparisons.append(("word_prior", prior))

        for comparison, pred in comparisons:
            for part in args.parts:
                sl = PARTS[part]
                values = dtw_distance(gt[:, sl], pred[:, sl])
                rows.append(
                    {
                        "index": index,
                        "comparison": comparison,
                        "part": part,
                        "gt_len": int(len(gt)),
                        "pred_len": int(len(pred)),
                        "dtw": values["dtw"],
                        "path_len": values["path_len"],
                        "ndtw": values["ndtw"],
                        "ndtw_ref": values["ndtw_ref"],
                        "sample": str(sample_path),
                        "gt": str(gt_path),
                    }
                )

    payload = {
        "samples_dir": str(args.samples_dir),
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "normalized": not args.no_normalize,
        "definition": (
            "dtw uses RMS distance per frame pair over the selected compact-SMPL-X feature slice; "
            "ndtw = dtw / optimal_path_length; ndtw_ref = dtw / len(gt). Lower is better."
        ),
        "num_pairs": len(pairs),
        "skipped_prior": skipped_prior,
        "parts": args.parts,
        "summary": summarize(rows),
        "rows": rows,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)

    print(json.dumps({k: payload[k] for k in ["num_pairs", "skipped_prior", "normalized", "summary"]}, indent=2))


if __name__ == "__main__":
    main()
