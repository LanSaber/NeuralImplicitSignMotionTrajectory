#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from flow.render import SoftwareMeshRenderer, apply_view_transform, normalize_vertices, smplx182_to_vertices
from flow.visualize.visualize_npz import load_smplx_sequence


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
    parser = argparse.ArgumentParser(description="Render GT, scaffold, meta-prior, and adapted meta-implicit outputs.")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True, help="Combined sample_XXXX.npz from export_meta_implicit_samples.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--end_mode", default="blank", choices=["blank", "hold"])
    parser.add_argument(
        "--normalization_scope",
        default="gt",
        choices=["gt", "all", "independent"],
        help="Use one camera normalization from GT, from all panels, or independent per panel.",
    )
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
    )
    parser.add_argument("--upper_body_only", action="store_true")
    parser.add_argument("--gt_label", default="Ground truth")
    parser.add_argument("--scaffold_label", default="Scaffold")
    parser.add_argument("--prior_label", default="Meta prior")
    parser.add_argument("--adapted_label", default="Meta adapted")
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


def normalization_params(vertices_list, vertex_indices=None, target_height=2.0):
    selected_chunks = []
    for vertices in vertices_list:
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertex_indices is None:
            selected_chunks.append(vertices.reshape(-1, 3))
        else:
            selected_chunks.append(vertices[:, vertex_indices, :].reshape(-1, 3))
    selected = np.concatenate(selected_chunks, axis=0)
    center = (selected.min(axis=0) + selected.max(axis=0)) * 0.5
    centered = selected - center
    extent = float(centered[:, 1].max() - centered[:, 1].min())
    if extent <= 1e-6:
        extent = float(np.max(centered.max(axis=0) - centered.min(axis=0)))
    scale = 1.0
    if target_height > 0 and extent > 1e-6:
        scale = float(target_height) / extent
    return center.astype(np.float32), float(scale)


def apply_normalization(vertices, center, scale):
    return (np.asarray(vertices, dtype=np.float32) - center.reshape(1, 1, 3)) * float(scale)


def prepare_vertices_raw(smplx, args):
    vertices, faces = smplx182_to_vertices(
        smplx,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.smplx_batch_size,
    )
    vertices = apply_view_transform(vertices, args.view_transform)
    return vertices, faces


def prepare_vertices(smplx, args):
    vertices, faces = prepare_vertices_raw(smplx, args)
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


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_smplx_sequence(args.gt, "motion", "smplx", max_frames=0)
    scaffold = load_smplx_sequence(args.sample, "coarse_motion", "coarse_smplx", max_frames=0)
    prior = load_smplx_sequence(args.sample, "meta_prior_motion", "meta_prior_smplx", max_frames=0)
    adapted = load_smplx_sequence(args.sample, "meta_adapted_motion", "meta_adapted_smplx", max_frames=0)
    gt_vertices, faces = prepare_vertices_raw(gt, args)
    scaffold_vertices, _ = prepare_vertices_raw(scaffold, args)
    prior_vertices, _ = prepare_vertices_raw(prior, args)
    adapted_vertices, _ = prepare_vertices_raw(adapted, args)

    norm_indices = None
    if args.upper_body_only:
        faces, norm_indices = load_upper_body_faces(faces, args.model_dir)
    if args.normalization_scope == "independent":
        if args.upper_body_only:
            gt_vertices = normalize_vertices_by_indices(gt_vertices, norm_indices)
            scaffold_vertices = normalize_vertices_by_indices(scaffold_vertices, norm_indices)
            prior_vertices = normalize_vertices_by_indices(prior_vertices, norm_indices)
            adapted_vertices = normalize_vertices_by_indices(adapted_vertices, norm_indices)
        else:
            gt_vertices = normalize_vertices(gt_vertices)
            scaffold_vertices = normalize_vertices(scaffold_vertices)
            prior_vertices = normalize_vertices(prior_vertices)
            adapted_vertices = normalize_vertices(adapted_vertices)
    else:
        reference = [gt_vertices] if args.normalization_scope == "gt" else [
            gt_vertices,
            scaffold_vertices,
            prior_vertices,
            adapted_vertices,
        ]
        center, scale = normalization_params(reference, vertex_indices=norm_indices)
        gt_vertices = apply_normalization(gt_vertices, center, scale)
        scaffold_vertices = apply_normalization(scaffold_vertices, center, scale)
        prior_vertices = apply_normalization(prior_vertices, center, scale)
        adapted_vertices = apply_normalization(adapted_vertices, center, scale)

    total_frames = max(len(gt_vertices), len(scaffold_vertices), len(prior_vertices), len(adapted_vertices))
    if args.max_frames > 0:
        total_frames = min(total_frames, int(args.max_frames))
    renderer = SoftwareMeshRenderer(faces, width=args.width, height=args.height, face_stride=args.software_face_stride)
    out_path = args.out_dir / f"{args.sample.stem}_meta_compare.mp4"
    writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    labels = [args.gt_label, args.scaffold_label, args.prior_label, args.adapted_label]
    colors = [
        (1.0, 0.92, 0.72, 1.0),
        (0.58, 0.95, 0.70, 1.0),
        (0.64, 0.64, 1.0, 1.0),
        (0.48, 0.78, 1.0, 1.0),
    ]
    try:
        for frame_idx in tqdm(range(total_frames), desc=f"render {out_path.name}"):
            frames = [
                render_or_empty(renderer, gt_vertices, frame_idx, args.end_mode, colors[0]),
                render_or_empty(renderer, scaffold_vertices, frame_idx, args.end_mode, colors[1]),
                render_or_empty(renderer, prior_vertices, frame_idx, args.end_mode, colors[2]),
                render_or_empty(renderer, adapted_vertices, frame_idx, args.end_mode, colors[3]),
            ]
            writer.append_data(draw_panel(frames, labels, frame_idx, total_frames))
    finally:
        writer.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
