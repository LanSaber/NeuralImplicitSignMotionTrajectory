#!/usr/bin/env python
"""Render GT, flow output, and word-prior SMPL-X sequences side by side."""
import argparse
import json
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


UPPER_BODY_PARTS = {
    "head",
    "neck",
    "spine",
    "spine1",
    "spine2",
    "hips",
    "leftShoulder",
    "rightShoulder",
    "leftArm",
    "rightArm",
    "leftForeArm",
    "rightForeArm",
    "leftHand",
    "rightHand",
    "leftHandIndex1",
    "rightHandIndex1",
    "leftEye",
    "rightEye",
    "eyeballs",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render GT, flow prediction, and residual word-prior SMPL-X sequences "
            "side by side."
        )
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth .npz file.")
    parser.add_argument(
        "--pred",
        type=Path,
        nargs="+",
        required=True,
        help="Generated .npz file(s), or directories containing sample_*.npz.",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=None,
        help="Optional word-prior .npz. Defaults to each --pred file.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("visualize/flow/flow_compare_three"))
    parser.add_argument("--gt_motion_key", default="motion")
    parser.add_argument("--gt_smplx_key", default="smplx")
    parser.add_argument("--pred_motion_key", default="motion")
    parser.add_argument("--pred_smplx_key", default="smplx")
    parser.add_argument("--prior_motion_key", default="coarse_motion")
    parser.add_argument("--prior_smplx_key", default="coarse_smplx")
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 renders the full longest sequence.")
    parser.add_argument("--end_mode", default="blank", choices=["blank", "hold"])
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
    )
    parser.add_argument("--gt_label", default="Ground truth")
    parser.add_argument("--pred_label", default="Flow matching")
    parser.add_argument("--prior_label", default="Word concatenation")
    parser.add_argument(
        "--upper_body_only",
        action="store_true",
        help="Render only upper-body SMPL-X mesh faces using smplx_vert_segmentation.json.",
    )
    return parser.parse_args()


def load_upper_body_faces(faces, model_dir):
    seg_path = Path(model_dir) / "smplx_vert_segmentation.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"Missing SMPL-X segmentation file: {seg_path}")
    with seg_path.open("r", encoding="utf-8") as handle:
        segmentation = json.load(handle)
    indices = set()
    for part in UPPER_BODY_PARTS:
        indices.update(int(idx) for idx in segmentation.get(part, []))
    if not indices:
        raise RuntimeError(f"No upper-body vertex indices found in {seg_path}")
    allowed = np.zeros(max(int(faces.max()) + 1, max(indices) + 1), dtype=bool)
    allowed[list(indices)] = True
    face_mask = np.all(allowed[faces], axis=1)
    return faces[face_mask], np.asarray(sorted(indices), dtype=np.int64)


def normalize_vertices_by_indices(vertices, vertex_indices, target_height=2.0):
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    selected = vertices[:, vertex_indices, :].reshape(-1, 3)
    center = (selected.min(axis=0) + selected.max(axis=0)) * 0.5
    vertices -= center
    selected = vertices[:, vertex_indices, :].reshape(-1, 3)
    extent = float(selected[:, 1].max() - selected[:, 1].min())
    if extent <= 1e-6:
        extent = float(np.max(selected.max(axis=0) - selected.min(axis=0)))
    if target_height > 0 and extent > 1e-6:
        vertices *= float(target_height) / extent
    return vertices


def prepare_vertices(smplx, args):
    vertices, faces = smplx182_to_vertices(
        smplx,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.smplx_batch_size,
    )
    vertices = apply_view_transform(vertices, args.view_transform)
    if args.upper_body_only:
        faces, upper_indices = load_upper_body_faces(faces, args.model_dir)
        vertices = normalize_vertices_by_indices(vertices, upper_indices)
    else:
        vertices = normalize_vertices(vertices)
    return vertices, faces


def render_or_empty(renderer, vertices, frame_idx, end_mode, color):
    if frame_idx < len(vertices):
        return renderer.render(vertices[frame_idx], color=color)
    if end_mode == "hold" and len(vertices) > 0:
        return renderer.render(vertices[-1], color=color)
    return np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)


def draw_panel(frames, labels, frame_idx, total_frames):
    header_h = 56
    h, w = frames[0].shape[:2]
    canvas = np.zeros((h + header_h, w * len(frames), 3), dtype=np.uint8)
    canvas[:header_h] = np.array([18, 20, 24], dtype=np.uint8)
    for idx, frame in enumerate(frames):
        canvas[header_h:, idx * w : (idx + 1) * w] = frame

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, label in enumerate(labels):
        draw.text((idx * w + 12, 8), str(label)[:92], fill=(238, 238, 238), font=font)
    draw.text((12, 32), f"frame {frame_idx + 1}/{total_frames}", fill=(185, 190, 198), font=font)
    return np.asarray(image)


def write_three_way_video(args, gt_vertices, pred_vertices, prior_vertices, pred_path, faces):
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    total_frames = max(len(gt_vertices), len(pred_vertices), len(prior_vertices))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    out_path = args.out_dir / f"{pred_path.stem}_gt_flow_word.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    labels = [
        args.gt_label,
        f"{args.pred_label}: {pred_path.stem}",
        args.prior_label,
    ]
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
            prior_frame = render_or_empty(
                renderer,
                prior_vertices,
                frame_idx,
                args.end_mode,
                color=(0.58, 0.95, 0.70, 1.0),
            )
            writer.append_data(
                draw_panel(
                    [gt_frame, pred_frame, prior_frame],
                    labels,
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
    if args.prior is not None and len(pred_files) != 1:
        raise ValueError("--prior can only be used when rendering one --pred file.")

    print(f"Loading GT: {args.gt}")
    gt_smplx = load_smplx_sequence(args.gt, args.gt_motion_key, args.gt_smplx_key, max_frames=0)
    gt_vertices, faces = prepare_vertices(gt_smplx, args)
    print(f"GT frames: {len(gt_vertices)}")

    for pred_path in pred_files:
        prior_path = args.prior if args.prior is not None else pred_path
        print(f"Loading flow prediction: {pred_path}")
        pred_smplx = load_smplx_sequence(
            pred_path,
            args.pred_motion_key,
            args.pred_smplx_key,
            max_frames=0,
        )
        pred_vertices, _ = prepare_vertices(pred_smplx, args)
        print(f"Flow frames: {len(pred_vertices)}")

        print(f"Loading word prior: {prior_path}")
        prior_smplx = load_smplx_sequence(
            prior_path,
            args.prior_motion_key,
            args.prior_smplx_key,
            max_frames=0,
        )
        prior_vertices, _ = prepare_vertices(prior_smplx, args)
        print(f"Word-prior frames: {len(prior_vertices)}")

        out_path = write_three_way_video(args, gt_vertices, pred_vertices, prior_vertices, pred_path, faces)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
