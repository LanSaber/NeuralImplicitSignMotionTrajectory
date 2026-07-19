#!/usr/bin/env python
"""Render ground-truth and generated flow SMPL-X sequences side by side."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from flow.render import (
    SoftwareMeshRenderer,
    apply_view_transform,
    normalize_vertices,
    smplx182_to_vertices,
)
from flow.visualize.visualize_npz import collect_inputs, load_smplx_sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GT compact/full SMPL-X and generated flow samples side by side."
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth .npz file.")
    parser.add_argument(
        "--pred",
        type=Path,
        nargs="+",
        required=True,
        help="Generated .npz file(s), or directories containing sample_*.npz.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("visualize/flow/flow_compare"))
    parser.add_argument("--motion_key", default="motion")
    parser.add_argument("--smplx_key", default="smplx")
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 renders the full longer sequence.")
    parser.add_argument("--end_mode", default="blank", choices=["blank", "hold"])
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
    )
    parser.add_argument("--gt_label", default="Ground truth")
    parser.add_argument("--pred_label", default="Flow prediction")
    return parser.parse_args()


def prepare_vertices(smplx, args):
    vertices, faces = smplx182_to_vertices(
        smplx,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.smplx_batch_size,
    )
    vertices = normalize_vertices(apply_view_transform(vertices, args.view_transform))
    return vertices, faces


def render_or_empty(renderer, vertices, frame_idx, end_mode, color):
    if frame_idx < len(vertices):
        return renderer.render(vertices[frame_idx], color=color)
    if end_mode == "hold" and len(vertices) > 0:
        return renderer.render(vertices[-1], color=color)
    return np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)


def draw_panel(gt_frame, pred_frame, gt_label, pred_label, frame_idx, total_frames):
    header_h = 52
    h, w = gt_frame.shape[:2]
    canvas = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)
    canvas[:header_h] = np.array([18, 20, 24], dtype=np.uint8)
    canvas[header_h:, :w] = gt_frame
    canvas[header_h:, w:] = pred_frame

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 8), gt_label[:80], fill=(238, 238, 238), font=font)
    draw.text((w + 12, 8), pred_label[:80], fill=(238, 238, 238), font=font)
    draw.text((12, 30), f"frame {frame_idx + 1}/{total_frames}", fill=(185, 190, 198), font=font)
    return np.asarray(image)


def write_pair_video(args, gt_vertices, pred_vertices, pred_path, faces):
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    total_frames = max(len(gt_vertices), len(pred_vertices))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    out_path = args.out_dir / f"{pred_path.stem}_vs_gt.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_idx in tqdm(range(total_frames), desc=f"render {out_path.name}"):
            gt_frame = render_or_empty(
                renderer,
                gt_vertices,
                frame_idx,
                args.end_mode,
                color=(1.0, 0.92, 0.72, 1.0),
            )
            pred_frame = render_or_empty(
                renderer,
                pred_vertices,
                frame_idx,
                args.end_mode,
                color=(0.48, 0.78, 1.0, 1.0),
            )
            writer.append_data(
                draw_panel(
                    gt_frame,
                    pred_frame,
                    args.gt_label,
                    f"{args.pred_label}: {pred_path.stem}",
                    frame_idx,
                    total_frames,
                )
            )
    finally:
        writer.close()
    return out_path


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_files = collect_inputs(args.pred)
    if not pred_files:
        raise RuntimeError("No prediction .npz files found.")

    print(f"Loading GT: {args.gt}")
    gt_smplx = load_smplx_sequence(args.gt, args.motion_key, args.smplx_key, max_frames=0)
    gt_vertices, faces = prepare_vertices(gt_smplx, args)
    print(f"GT frames: {len(gt_vertices)}")

    for pred_path in pred_files:
        print(f"Loading prediction: {pred_path}")
        pred_smplx = load_smplx_sequence(pred_path, args.motion_key, args.smplx_key, max_frames=0)
        pred_vertices, _ = prepare_vertices(pred_smplx, args)
        print(f"Prediction frames: {len(pred_vertices)}")
        out_path = write_pair_video(args, gt_vertices, pred_vertices, pred_path, faces)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
