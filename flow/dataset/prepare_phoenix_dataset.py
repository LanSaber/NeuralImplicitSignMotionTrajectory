#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np


DEFAULT_PHOENIX_ROOT = Path("/media/cvpr/haomian/data/SOKE/Phoenix_2014T")
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx")

POSE_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]

SPLIT_MAP = {
    "train": "train",
    "dev": "val",
    "test": "test",
}


class FakeStorage:
    pass


class FakeTensor:
    def __init__(self, size=None, stride=None):
        self.shape = tuple(size or ())
        self.stride = tuple(stride or ())


def fake_load_from_bytes(_payload):
    return FakeStorage()


def fake_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    return FakeTensor(size, stride)


class TorchlessUnpickler(pickle.Unpickler):
    """Load Phoenix annotation pickles without requiring torch."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return fake_load_from_bytes
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return fake_rebuild_tensor_v2
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare RWTH-PHOENIX-Weather 2014T SMPL-X poses in SOKE_FLOW format."
    )
    parser.add_argument("--phoenix_root", type=Path, default=DEFAULT_PHOENIX_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--source_fps", type=float, default=25.0)
    parser.add_argument("--target_fps", type=float, default=20.0)
    parser.add_argument("--min_raw_frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per source split. 0 means all rows.")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel workers. 0 chooses a conservative default.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_annotations(path):
    with gzip.open(path, "rb") as handle:
        return TorchlessUnpickler(handle).load()


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


def frame_files_for(pose_dir):
    return sorted(path for path in pose_dir.iterdir() if path.suffix == ".pkl")


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


def safe_stem(source_name):
    return Path(source_name).name


def process_one(task):
    source_name = task["source_name"]
    pose_dir = Path(task["pose_dir"])
    out_path = Path(task["out_path"])
    rel_path = task["rel_path"]
    source_fps = task["source_fps"]
    target_fps = task["target_fps"]
    effective_fps = target_fps if target_fps and target_fps > 0 else source_fps

    if not pose_dir.is_dir():
        return {"status": "skip", "name": source_name, "reason": f"missing pose directory {pose_dir}"}

    frame_paths = frame_files_for(pose_dir)
    raw_frame_count = len(frame_paths)
    if raw_frame_count < task["min_raw_frames"]:
        return {"status": "skip", "name": source_name, "reason": f"only {raw_frame_count} pose frames"}

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
            np.savez_compressed(
                out_path,
                motion=motion.astype(np.float32),
                left_valid=left_valid.astype(np.float32),
                right_valid=right_valid.astype(np.float32),
            )
            num_frames = int(motion.shape[0])
    except Exception as exc:
        return {"status": "skip", "name": source_name, "reason": f"failed to load/write pose frames: {exc}"}

    item = {
        "name": task["stem"],
        "motion_path": rel_path,
        "text": task["text"],
        "gloss": task["gloss"],
        "signer": task["signer"],
        "fps": float(effective_fps),
        "num_frames": num_frames,
        "duration": float(num_frames / effective_fps),
        "dataset": "phoenix14t",
        "source_name": source_name,
        "source_split": task["source_split"],
        "source_fps": float(source_fps),
        "source_num_frames": int(task["source_num_frames"] if task["source_num_frames"] is not None else raw_frame_count),
        "source_pose_frames": int(raw_frame_count),
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


def process_split(args, source_split, running_stats):
    out_split = SPLIT_MAP.get(source_split, source_split)
    ann_path = args.phoenix_root / f"phoenix14t.{source_split}"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing Phoenix annotation file: {ann_path}")

    annotations = load_annotations(ann_path)
    split_out = args.out_dir / out_split
    split_out.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen_paths = set()

    tasks = []
    for ann in annotations:
        if args.limit > 0 and len(tasks) >= args.limit:
            break

        source_name = str(ann["name"])
        stem = safe_stem(source_name)
        out_path = split_out / f"{stem}.npz"
        rel_path = out_path.relative_to(args.out_dir).as_posix()
        if rel_path in seen_paths:
            raise RuntimeError(f"Duplicate output path in split {out_split}: {rel_path}")
        seen_paths.add(rel_path)

        tasks.append(
            {
                "source_name": source_name,
                "stem": stem,
                "pose_dir": args.phoenix_root / source_name,
                "out_path": out_path,
                "rel_path": rel_path,
                "text": str(ann.get("text", "")),
                "gloss": str(ann.get("gloss", "")),
                "signer": str(ann.get("signer", "")),
                "source_split": source_split,
                "source_fps": float(args.source_fps),
                "target_fps": float(args.target_fps),
                "source_num_frames": ann.get("num_frames", None),
                "min_raw_frames": int(args.min_raw_frames),
                "overwrite": bool(args.overwrite),
                "stats": out_split == "train",
            }
        )

    workers = worker_count(args)
    print(f"{source_split}: processing {len(tasks)} annotations with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(process_one, tasks), start=1):
            if result["status"] != "ok":
                warnings.warn(f"Skipping {result['name']}: {result['reason']}", RuntimeWarning)
            else:
                manifest.append(result["item"])
                if out_split == "train":
                    running_stats["count"] += result["stat_count"]
                    running_stats["sum"] += result["stat_sum"]
                    running_stats["sumsq"] += result["stat_sumsq"]

            if index % 500 == 0 or index == len(tasks):
                print(f"{source_split}: processed {index}/{len(tasks)} annotations", flush=True)

    manifest_path = args.out_dir / "meta" / f"manifest_{out_split}.jsonl"
    write_jsonl(manifest_path, manifest)
    return out_split, manifest


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
    for source_split in args.splits:
        out_split, manifest = process_split(args, source_split, stats)
        print(f"{source_split} -> {out_split}: wrote {len(manifest)} samples")
    if "train" in args.splits:
        save_stats(args, stats)
        print(f"Saved mean/std under {args.out_dir / 'meta'}")


if __name__ == "__main__":
    main()
