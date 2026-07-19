#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from flow.render import SoftwareMeshRenderer, normalize_vertices
from flow.visualize.visualize_npz import load_smplx_sequence
from NIAF.continuous_sign_field.visualization.visualize_meta_implicit_compare import (
    apply_normalization,
    draw_panel,
    load_upper_body_faces,
    normalization_params,
    normalize_vertices_by_indices,
    prepare_vertices_raw,
    render_or_empty,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GT, adapter context, continuous prior, and final trajectory."
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
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
    )
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=[
            "none",
            "how2sign_front",
            "rot_x_180",
            "rot_y_180",
            "rot_z_180",
            "flip_y",
            "flip_z",
        ],
    )
    parser.add_argument("--upper_body_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_smplx_sequence(args.gt, "motion", "smplx", max_frames=0)
    adapter = load_smplx_sequence(
        args.sample,
        "adapter_context_motion",
        "adapter_context_smplx",
        max_frames=0,
    )
    prior = load_smplx_sequence(
        args.sample,
        "continuous_prior_motion",
        "continuous_prior_smplx",
        max_frames=0,
    )
    final = load_smplx_sequence(args.sample, "motion", "smplx", max_frames=0)
    sequences = [gt, adapter, prior, final]
    vertices = []
    faces = None
    for sequence in sequences:
        sequence_vertices, sequence_faces = prepare_vertices_raw(sequence, args)
        vertices.append(sequence_vertices)
        if faces is None:
            faces = sequence_faces

    normalization_indices = None
    if args.upper_body_only:
        faces, normalization_indices = load_upper_body_faces(faces, args.model_dir)
    if args.normalization_scope == "independent":
        for index, sequence_vertices in enumerate(vertices):
            if args.upper_body_only:
                vertices[index] = normalize_vertices_by_indices(
                    sequence_vertices, normalization_indices
                )
            else:
                vertices[index] = normalize_vertices(sequence_vertices)
    else:
        reference = vertices[:1] if args.normalization_scope == "gt" else vertices
        center, scale = normalization_params(
            reference, vertex_indices=normalization_indices
        )
        vertices = [apply_normalization(value, center, scale) for value in vertices]

    total_frames = max(len(value) for value in vertices)
    if args.max_frames > 0:
        total_frames = min(total_frames, int(args.max_frames))
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    out_path = args.out_dir / f"{args.sample.stem}_continuous_compare.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    labels = [
        "Ground truth",
        "Adapter context",
        "Continuous prior",
        "Continuous final",
    ]
    colors = [
        (1.0, 0.92, 0.72, 1.0),
        (0.58, 0.95, 0.70, 1.0),
        (0.64, 0.64, 1.0, 1.0),
        (0.48, 0.78, 1.0, 1.0),
    ]
    try:
        for frame_index in tqdm(range(total_frames), desc=f"render {out_path.name}"):
            panels = [
                render_or_empty(
                    renderer,
                    sequence_vertices,
                    frame_index,
                    args.end_mode,
                    color,
                )
                for sequence_vertices, color in zip(vertices, colors)
            ]
            writer.append_data(draw_panel(panels, labels, frame_index, total_frames))
    finally:
        writer.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
