from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from flow.smplx_features import (
    compact_axis_angle_to_rot6d,
    smplx182_from_compact,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw per-frame smpl_samples pickles to SOKE compact SMPL-X NPZs.")
    parser.add_argument("--src_dir", type=Path, default=Path("/media/cvpr/haomian/data/SOKE_FLOW/smpl_samples"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/media/cvpr/haomian/data/SOKE_FLOW/smpl_samples_soke_upper_smplx"),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--expression_source", default="smplx", choices=["smplx", "flame"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def frame_to_compact(frame, expression_source="smplx"):
    smplx = frame["smplx_coeffs"]
    flame = frame["flame_coeffs"]

    body_pose = np.asarray(smplx["body_pose"], dtype=np.float32).reshape(21, 3)
    upper_body = body_pose[11:21].reshape(-1)
    left_hand = np.asarray(smplx["left_hand_pose"], dtype=np.float32).reshape(-1)
    right_hand = np.asarray(smplx["right_hand_pose"], dtype=np.float32).reshape(-1)
    jaw = np.asarray(flame["jaw_params"], dtype=np.float32).reshape(3)
    if expression_source == "flame":
        expression = np.asarray(flame["expression_params"], dtype=np.float32).reshape(-1)[:10]
    else:
        expression = np.asarray(smplx["exp"], dtype=np.float32).reshape(-1)[:10]

    compact = np.concatenate([upper_body, left_hand, right_hand, jaw, expression]).astype(np.float32, copy=False)
    if compact.shape != (133,):
        raise ValueError(f"Expected compact frame shape (133,), got {compact.shape}")
    if not np.isfinite(compact).all():
        raise ValueError("Non-finite values found in compact SMPL-X frame.")
    return compact


def convert_file(path, out_motion_path, fps, expression_source):
    data = load_pickle(path)
    frame_keys = sorted(key for key in data.keys() if str(key).startswith("frame_"))
    if not frame_keys:
        raise ValueError(f"No frame_XXXX records found in {path}")

    motion = np.stack([frame_to_compact(data[key], expression_source=expression_source) for key in frame_keys])
    left_valid = np.ones(len(motion), dtype=np.float32)
    right_valid = np.ones(len(motion), dtype=np.float32)
    smplx = smplx182_from_compact(motion)

    out_motion_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_motion_path,
        motion=motion.astype(np.float32),
        smplx=smplx.astype(np.float32),
        left_valid=left_valid,
        right_valid=right_valid,
        fps=np.asarray(float(fps), dtype=np.float32),
        source_pickle=np.asarray(str(path)),
        expression_source=np.asarray(str(expression_source)),
    )
    return {
        "name": path.stem,
        "motion_path": str(out_motion_path.relative_to(out_motion_path.parents[1])),
        "text": "",
        "gloss": "",
        "num_frames": int(len(motion)),
        "fps": float(fps),
        "duration": float(len(motion) / max(float(fps), 1e-6)),
        "source_pickle": str(path),
    }, motion


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_stats(out_dir, motions):
    all_axis = np.concatenate(motions, axis=0).astype(np.float32)
    mean = all_axis.mean(axis=0).astype(np.float32)
    std = np.maximum(all_axis.std(axis=0), 1e-4).astype(np.float32)
    np.save(out_dir / "meta" / "mean.npy", mean)
    np.save(out_dir / "meta" / "std.npy", std)

    all_rot6d = compact_axis_angle_to_rot6d(all_axis).astype(np.float32)
    mean_rot6d = all_rot6d.mean(axis=0).astype(np.float32)
    std_rot6d = np.maximum(all_rot6d.std(axis=0), 1e-4).astype(np.float32)
    np.save(out_dir / "meta" / "mean_rot6d.npy", mean_rot6d)
    np.save(out_dir / "meta" / "std_rot6d.npy", std_rot6d)


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {src_dir}")

    files = sorted(src_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {src_dir}")
    if out_dir.exists() and any(out_dir.glob("motions/*.npz")) and not args.overwrite:
        raise FileExistsError(f"{out_dir}/motions already contains NPZs. Pass --overwrite to regenerate.")

    rows = []
    motions = []
    for path in tqdm(files, desc="convert smpl_samples"):
        out_motion_path = out_dir / "motions" / f"{path.stem}.npz"
        row, motion = convert_file(path, out_motion_path, fps=args.fps, expression_source=args.expression_source)
        rows.append(row)
        motions.append(motion)

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(meta_dir / f"manifest_{args.split}.jsonl", rows)
    write_jsonl(meta_dir / f"manifest_{args.val_split}.jsonl", rows)
    save_stats(out_dir, motions)

    summary = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "num_sequences": len(rows),
        "num_frames": int(sum(len(motion) for motion in motions)),
        "min_frames": int(min(len(motion) for motion in motions)),
        "max_frames": int(max(len(motion) for motion in motions)),
        "fps": float(args.fps),
        "expression_source": str(args.expression_source),
    }
    (meta_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
