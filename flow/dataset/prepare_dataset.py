#!/usr/bin/env python
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from flow.residual_prior import split_word_variant_name
from flow.smplx_features import extract_compact_from_pickle, resample_by_fps


DEFAULT_PKL_DIR = Path("/media/cvpr/haomian/data/how2sign_pkls_cropTrue_shapeFalse")
DEFAULT_SOKE_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare compact upper-body SMPL-X features and Flow JSONL manifests."
    )
    parser.add_argument("--pkl_dir", type=Path, default=DEFAULT_PKL_DIR)
    parser.add_argument("--soke_root", type=Path, default=DEFAULT_SOKE_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--target_fps", type=float, default=20.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--min_raw_frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split. 0 means all rows.")
    parser.add_argument(
        "--parse_word_variant_names",
        action="store_true",
        help=(
            "If sample names or output filenames follow WORD-0001 style, add "
            "lexicon_key=WORD, variant_id=0001, and word=WORD to the manifest. "
            "This is useful for word/gloss prior datasets with multiple clips per word."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_csv_path(soke_root, split):
    return soke_root / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"


def row_duration(row):
    if "DURATION" in row and not pd.isna(row["DURATION"]):
        return float(row["DURATION"])
    if "START_REALIGNED" in row and "END_REALIGNED" in row:
        return float(row["END_REALIGNED"]) - float(row["START_REALIGNED"])
    return None


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_split(args, split, running_stats):
    csv_path = split_csv_path(args.soke_root, split)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing annotation CSV: {csv_path}")

    split_out = args.out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    manifest = []
    csv = pd.read_csv(csv_path)

    iterator = tqdm(csv.iterrows(), total=len(csv), desc=f"prepare {split}")
    for _, row in iterator:
        if args.limit > 0 and len(manifest) >= args.limit:
            break
        name = str(row["SENTENCE_NAME"])
        text = str(row["SENTENCE"])
        duration = row_duration(row)
        if duration is not None and duration > args.max_duration:
            continue

        pkl_path = args.pkl_dir / f"{name}.pkl"
        if not pkl_path.is_file():
            warnings.warn(f"Skipping {name}: missing pickle {pkl_path}", RuntimeWarning)
            continue

        out_path = split_out / f"{name}.npz"
        if out_path.is_file() and not args.overwrite:
            with np.load(out_path) as data:
                num_frames = int(data["motion"].shape[0])
        else:
            try:
                motion, left_valid, right_valid = extract_compact_from_pickle(pkl_path)
            except Exception as exc:
                warnings.warn(f"Skipping {name}: failed to load {pkl_path}: {exc}", RuntimeWarning)
                continue
            if len(motion) < args.min_raw_frames:
                warnings.warn(f"Skipping {name}: only {len(motion)} frames", RuntimeWarning)
                continue

            fps = float(row["fps"]) if "fps" in row and not pd.isna(row["fps"]) else None
            motion, left_valid, right_valid = resample_by_fps(
                motion, left_valid, right_valid, source_fps=fps, target_fps=args.target_fps
            )
            np.savez_compressed(
                out_path,
                motion=motion.astype(np.float32),
                left_valid=left_valid.astype(np.float32),
                right_valid=right_valid.astype(np.float32),
            )
            num_frames = int(motion.shape[0])

        rel_path = out_path.relative_to(args.out_dir).as_posix()
        item = {
            "name": name,
            "motion_path": rel_path,
            "text": text,
            "fps": float(args.target_fps),
            "num_frames": num_frames,
            "duration": float(num_frames / args.target_fps),
        }
        if args.parse_word_variant_names:
            lexicon_key, variant_id = split_word_variant_name(name)
            if variant_id is None:
                lexicon_key, variant_id = split_word_variant_name(out_path.stem)
            if variant_id is not None:
                item["lexicon_key"] = lexicon_key
                item["variant_id"] = variant_id
                item["word"] = lexicon_key
        manifest.append(item)

        if split == "train":
            with np.load(out_path) as data:
                motion = data["motion"].astype(np.float64)
            running_stats["count"] += motion.shape[0]
            running_stats["sum"] += motion.sum(axis=0)
            running_stats["sumsq"] += np.square(motion).sum(axis=0)

    manifest_path = args.out_dir / "meta" / f"manifest_{split}.jsonl"
    write_jsonl(manifest_path, manifest)
    return manifest


def save_stats(args, stats):
    if stats["count"] <= 0:
        raise RuntimeError("No training frames were processed; cannot compute mean/std.")
    mean = stats["sum"] / stats["count"]
    var = np.maximum(stats["sumsq"] / stats["count"] - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)
    meta = args.out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    np.save(meta / "mean.npy", mean.astype(np.float32))
    np.save(meta / "std.npy", std.astype(np.float32))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "count": 0,
        "sum": np.zeros(133, dtype=np.float64),
        "sumsq": np.zeros(133, dtype=np.float64),
    }
    for split in args.splits:
        manifest = process_split(args, split, stats)
        print(f"{split}: wrote {len(manifest)} samples")
    if "train" in args.splits:
        save_stats(args, stats)
        print(f"Saved mean/std under {args.out_dir / 'meta'}")


if __name__ == "__main__":
    main()
