#!/usr/bin/env python
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split an existing prepared upper-SMPL-X dataset into train/val/test."
    )
    parser.add_argument(
        "--src_dir",
        type=Path,
        default=Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_split_80_10_10"),
    )
    parser.add_argument("--src_split", default="train")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--link_mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize(src_path, dst_path, mode, overwrite):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        if not overwrite:
            return
        dst_path.unlink()

    if mode == "hardlink":
        os.link(src_path, dst_path)
    elif mode == "symlink":
        dst_path.symlink_to(src_path)
    else:
        shutil.copy2(src_path, dst_path)


def split_rows(rows, val_ratio, test_ratio, seed):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0, 1).")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val_ratio + --test_ratio must be smaller than 1.")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_val = round(n_total * val_ratio)
    n_test = round(n_total * test_ratio)
    val_rows = shuffled[:n_val]
    test_rows = shuffled[n_val : n_val + n_test]
    train_rows = shuffled[n_val + n_test :]
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def rewrite_and_link_rows(rows, split, src_dir, out_dir, link_mode, overwrite):
    rewritten = []
    for row in tqdm(rows, desc=f"link {split}"):
        src_path = src_dir / row["motion_path"]
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source motion file: {src_path}")
        dst_rel = Path(split) / src_path.name
        dst_path = out_dir / dst_rel
        materialize(src_path, dst_path, link_mode, overwrite)

        item = dict(row)
        item["motion_path"] = dst_rel.as_posix()
        rewritten.append(item)
    return rewritten


def save_train_stats(out_dir, train_rows):
    count = 0
    total = np.zeros(133, dtype=np.float64)
    total_sq = np.zeros(133, dtype=np.float64)

    for row in tqdm(train_rows, desc="train stats"):
        with np.load(out_dir / row["motion_path"]) as data:
            motion = data["motion"].astype(np.float64)
        count += motion.shape[0]
        total += motion.sum(axis=0)
        total_sq += np.square(motion).sum(axis=0)

    if count <= 0:
        raise RuntimeError("No train frames found; cannot compute mean/std.")

    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(meta_dir / "mean.npy", mean.astype(np.float32))
    np.save(meta_dir / "std.npy", std.astype(np.float32))


def main():
    args = parse_args()
    src_manifest = args.src_dir / "meta" / f"manifest_{args.src_split}.jsonl"
    rows = read_jsonl(src_manifest)
    if not rows:
        raise RuntimeError(f"No rows found in {src_manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = split_rows(rows, args.val_ratio, args.test_ratio, args.seed)

    rewritten_splits = {}
    for split, split_rows_ in splits.items():
        rewritten = rewrite_and_link_rows(
            split_rows_, split, args.src_dir, args.out_dir, args.link_mode, args.overwrite
        )
        write_jsonl(args.out_dir / "meta" / f"manifest_{split}.jsonl", rewritten)
        rewritten_splits[split] = rewritten
        print(f"{split}: wrote {len(rewritten)} samples")

    save_train_stats(args.out_dir, rewritten_splits["train"])
    print(f"Saved split dataset under {args.out_dir}")


if __name__ == "__main__":
    main()
