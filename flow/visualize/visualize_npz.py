#!/usr/bin/env python
"""Render generated flow SMPL-X NPZ files."""
import argparse
from pathlib import Path

import numpy as np

from flow.render import smplx182_to_vertices, write_vertices_video
from flow.smplx_features import COMPACT_DIM, FULL_SMPLX_DIM, smplx182_from_compact


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render generated flow .npz SMPL-X samples to MP4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more .npz files, or directories containing sample_*.npz files.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("visualize/flow/flow_npz"))
    parser.add_argument("--motion_key", default="motion")
    parser.add_argument("--smplx_key", default="smplx")
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 renders the full sequence.")
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
        help="Use none for canonical zero-root generated samples; use how2sign_front for raw How2Sign pickles.",
    )
    return parser.parse_args()


def collect_inputs(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("sample_*.npz")))
            files.extend(sorted(path.glob("*.npz")) if not files else [])
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    deduped = []
    seen = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(file_path)
    return deduped


def load_smplx_sequence(path, motion_key, smplx_key, max_frames):
    with np.load(path) as data:
        if smplx_key in data.files:
            smplx = data[smplx_key].astype(np.float32)
            if smplx.ndim != 2 or smplx.shape[1] != FULL_SMPLX_DIM:
                raise ValueError(f"{path}: key {smplx_key!r} must have shape [T, 182], got {smplx.shape}")
        elif motion_key in data.files:
            motion = data[motion_key].astype(np.float32)
            if motion.ndim != 2 or motion.shape[1] != COMPACT_DIM:
                raise ValueError(f"{path}: key {motion_key!r} must have shape [T, 133], got {motion.shape}")
            smplx = smplx182_from_compact(motion)
        else:
            raise KeyError(f"{path}: expected key {smplx_key!r} or {motion_key!r}; found {data.files}")

    if max_frames > 0:
        smplx = smplx[:max_frames]
    return smplx


def main():
    args = parse_args()
    files = collect_inputs(args.input)
    if not files:
        raise RuntimeError("No .npz files found to render.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        print(f"Rendering {path}")
        smplx = load_smplx_sequence(path, args.motion_key, args.smplx_key, args.max_frames)
        vertices, faces = smplx182_to_vertices(
            smplx,
            model_dir=args.model_dir,
            device=args.device,
            batch_size=args.smplx_batch_size,
        )
        out_path = args.out_dir / f"{path.stem}.mp4"
        write_vertices_video(
            vertices,
            faces,
            out_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            face_stride=args.software_face_stride,
            label=path.stem,
            view_transform=args.view_transform,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
