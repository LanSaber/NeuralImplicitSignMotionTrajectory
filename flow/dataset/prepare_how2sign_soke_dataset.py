#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_HOW2SIGN_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx")

POSE_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]

FRAME_RE = re.compile(r"_(\d+)_3D\.pkl$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SOKE How2Sign per-frame SMPL-X poses in SOKE_FLOW format."
    )
    parser.add_argument("--how2sign_root", type=Path, default=DEFAULT_HOW2SIGN_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--target_fps", type=float, default=20.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--min_raw_frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split. 0 means all rows.")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel workers. 0 chooses a conservative default.")
    parser.add_argument("--compress", action="store_true", help="Use np.savez_compressed instead of faster uncompressed npz.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_csv_path(root, split):
    return root / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"


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


def resample_array(array, target_frames, nearest=False):
    array = np.asarray(array)
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if len(array) == target_frames:
        return array.copy()
    if len(array) == 1:
        return np.repeat(array, target_frames, axis=0)

    src = np.linspace(0.0, 1.0, num=len(array), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    if nearest:
        index = np.clip(np.rint(dst * (len(array) - 1)).astype(np.int64), 0, len(array) - 1)
        return array[index].copy()

    flat = array.reshape(len(array), -1)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(dst, src, flat[:, dim])
    return out.reshape((target_frames,) + array.shape[1:]).astype(np.float32, copy=False)


def resample_by_fps(motion, left_valid, right_valid, source_fps, target_fps):
    if source_fps is None or target_fps is None or target_fps <= 0:
        return motion, left_valid, right_valid
    source_fps = float(source_fps)
    if source_fps <= 0:
        return motion, left_valid, right_valid
    target_frames = max(1, int(round(len(motion) * float(target_fps) / source_fps)))
    return (
        resample_array(motion, target_frames, nearest=False),
        resample_array(left_valid, target_frames, nearest=True).astype(np.float32),
        resample_array(right_valid, target_frames, nearest=True).astype(np.float32),
    )


def frame_index(path):
    match = FRAME_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def frame_files_for(pose_dir):
    frame_paths = []
    for path in pose_dir.iterdir():
        if path.suffix != ".pkl":
            continue
        index = frame_index(path)
        if index is not None:
            frame_paths.append((index, path))
    return [path for _, path in sorted(frame_paths, key=lambda item: item[0])]


def load_compact_sequence(frame_paths):
    full179 = []
    for frame_path in frame_paths:
        with frame_path.open("rb") as handle:
            frame = pickle.load(handle)

        missing_keys = [key for key in POSE_KEYS if key not in frame]
        if missing_keys:
            raise KeyError(f"{frame_path} is missing keys: {missing_keys}")

        full179.append(np.concatenate([np.asarray(frame[key], dtype=np.float32) for key in POSE_KEYS], axis=0))

    full179 = np.stack(full179, axis=0).astype(np.float32, copy=False)
    return np.concatenate(
        [
            full179[:, 36:66],
            full179[:, 66:111],
            full179[:, 111:156],
            full179[:, 156:159],
            full179[:, 169:179],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def process_one(task):
    name = task["name"]
    pose_dir = Path(task["pose_dir"])
    out_path = Path(task["out_path"])
    source_fps = task["source_fps"]
    target_fps = task["target_fps"]
    effective_fps = target_fps if target_fps and target_fps > 0 else source_fps

    if not pose_dir.is_dir():
        return {"status": "skip", "name": name, "reason": f"missing pose directory {pose_dir}"}

    frame_paths = frame_files_for(pose_dir)
    raw_frame_count = len(frame_paths)
    if raw_frame_count < task["min_raw_frames"]:
        return {"status": "skip", "name": name, "reason": f"only {raw_frame_count} pose frames"}

    try:
        if out_path.is_file() and not task["overwrite"]:
            with np.load(out_path) as data:
                motion = data["motion"].astype(np.float32)
                num_frames = int(motion.shape[0])
        else:
            motion = load_compact_sequence(frame_paths)
            left_valid = np.ones(len(motion), dtype=np.float32)
            right_valid = np.ones(len(motion), dtype=np.float32)
            motion, left_valid, right_valid = resample_by_fps(
                motion,
                left_valid,
                right_valid,
                source_fps=source_fps,
                target_fps=target_fps,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_npz = np.savez_compressed if task["compress"] else np.savez
            save_npz(
                out_path,
                motion=motion.astype(np.float32),
                left_valid=left_valid.astype(np.float32),
                right_valid=right_valid.astype(np.float32),
            )
            num_frames = int(motion.shape[0])
    except Exception as exc:
        return {"status": "skip", "name": name, "reason": f"failed to load/write pose frames: {exc}"}

    item = {
        "name": name,
        "motion_path": task["rel_path"],
        "text": task["text"],
        "gloss": "",
        "fps": float(effective_fps),
        "num_frames": num_frames,
        "duration": float(num_frames / effective_fps) if effective_fps > 0 else 0.0,
        "dataset": "how2sign",
        "source_name": name,
        "source_split": task["split"],
        "source_fps": float(source_fps),
        "source_pose_frames": int(raw_frame_count),
        "video_id": task["video_id"],
        "video_name": task["video_name"],
        "sentence_id": task["sentence_id"],
        "start_realigned": task["start_realigned"],
        "end_realigned": task["end_realigned"],
    }
    result = {"status": "ok", "item": item}
    if task["stats"]:
        motion64 = motion.astype(np.float64)
        result["stat_count"] = int(motion64.shape[0])
        result["stat_sum"] = motion64.sum(axis=0)
        result["stat_sumsq"] = np.square(motion64).sum(axis=0)
    return result


def worker_count(args):
    if args.num_workers > 0:
        return args.num_workers
    return max(1, min(8, os.cpu_count() or 1))


def finite_float(value, default=0.0):
    if value is None or pd.isna(value):
        return default
    value = float(value)
    if math.isfinite(value):
        return value
    return default


def process_split(args, split, running_stats):
    csv_path = split_csv_path(args.how2sign_root, split)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing How2Sign annotation CSV: {csv_path}")

    csv = pd.read_csv(csv_path)
    split_out = args.out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    manifest = []
    tasks = []
    skipped_duration = 0

    for _, row in csv.iterrows():
        if args.limit > 0 and len(tasks) >= args.limit:
            break

        duration = row_duration(row)
        if duration is not None and duration > args.max_duration:
            skipped_duration += 1
            continue

        name = str(row["SENTENCE_NAME"])
        out_path = split_out / f"{name}.npz"
        rel_path = out_path.relative_to(args.out_dir).as_posix()
        tasks.append(
            {
                "name": name,
                "pose_dir": args.how2sign_root / split / "poses" / name,
                "out_path": out_path,
                "rel_path": rel_path,
                "text": str(row["SENTENCE"]),
                "split": split,
                "source_fps": finite_float(row.get("fps"), default=args.target_fps),
                "target_fps": float(args.target_fps),
                "min_raw_frames": int(args.min_raw_frames),
                "overwrite": bool(args.overwrite),
                "compress": bool(args.compress),
                "stats": split == "train",
                "video_id": str(row.get("VIDEO_ID", "")),
                "video_name": str(row.get("VIDEO_NAME", "")),
                "sentence_id": str(row.get("SENTENCE_ID", "")),
                "start_realigned": finite_float(row.get("START_REALIGNED"), default=0.0),
                "end_realigned": finite_float(row.get("END_REALIGNED"), default=0.0),
            }
        )

    workers = worker_count(args)
    print(
        f"{split}: processing {len(tasks)} annotations with {workers} workers "
        f"({skipped_duration} skipped by max_duration)",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(process_one, tasks), start=1):
            if result["status"] != "ok":
                warnings.warn(f"Skipping {result['name']}: {result['reason']}", RuntimeWarning)
            else:
                manifest.append(result["item"])
                if split == "train":
                    running_stats["count"] += result["stat_count"]
                    running_stats["sum"] += result["stat_sum"]
                    running_stats["sumsq"] += result["stat_sumsq"]

            if index % 500 == 0 or index == len(tasks):
                print(f"{split}: processed {index}/{len(tasks)} annotations", flush=True)

    write_jsonl(args.out_dir / "meta" / f"manifest_{split}.jsonl", manifest)
    return manifest, skipped_duration


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
        manifest, skipped_duration = process_split(args, split, stats)
        print(f"{split}: wrote {len(manifest)} samples; skipped_duration={skipped_duration}")
    if "train" in args.splits:
        save_stats(args, stats)
        print(f"Saved mean/std under {args.out_dir / 'meta'}")


if __name__ == "__main__":
    main()
