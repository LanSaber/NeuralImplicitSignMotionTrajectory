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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a frame-aligned GT, scaffold, Stage 1, and Stage 2 "
            "continuous-trajectory comparison."
        )
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--stage1_sample", type=Path, required=True)
    parser.add_argument("--stage2_sample", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
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
    parser.add_argument(
        "--scaffold_tolerance",
        type=float,
        default=1e-5,
        help="Maximum allowed Stage 1/2 scaffold parameter difference.",
    )
    parser.add_argument("--gt_label", default="Original SMPL-X pose")
    parser.add_argument("--scaffold_label", default="Scaffold SMPL-X pose")
    parser.add_argument("--stage1_label", default="Stage 1 trajectory")
    parser.add_argument("--stage2_label", default="Stage 2 trajectory")
    return parser.parse_args()


def _load_scalar(npz_path: Path, key: str):
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data:
            return None
        value = np.asarray(data[key])
        return value.item() if value.ndim == 0 else value.tolist()


def validate_alignment(
    gt: np.ndarray,
    stage1_scaffold: np.ndarray,
    stage2_scaffold: np.ndarray,
    stage1: np.ndarray,
    stage2: np.ndarray,
    scaffold_tolerance: float,
):
    lengths = {
        "ground truth": len(gt),
        "Stage 1 scaffold": len(stage1_scaffold),
        "Stage 2 scaffold": len(stage2_scaffold),
        "Stage 1 trajectory": len(stage1),
        "Stage 2 trajectory": len(stage2),
    }
    if len(set(lengths.values())) != 1:
        formatted = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            "The four-panel renderer requires exact frame alignment; " + formatted
        )
    scaffold_delta = float(
        np.max(np.abs(stage1_scaffold.astype(np.float64) - stage2_scaffold))
    )
    if scaffold_delta > float(scaffold_tolerance):
        raise ValueError(
            "Stage 1 and Stage 2 exports do not contain the same scaffold: "
            f"max_abs_delta={scaffold_delta:.6g}, "
            f"tolerance={float(scaffold_tolerance):.6g}"
        )
    return lengths["ground truth"], scaffold_delta


def _normalize_panel_vertices(vertices, faces, args):
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
    return vertices, faces


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_smplx_sequence(args.gt, "motion", "smplx", max_frames=0)
    stage1_scaffold = load_smplx_sequence(
        args.stage1_sample,
        "adapter_context_motion",
        "adapter_context_smplx",
        max_frames=0,
    )
    stage2_scaffold = load_smplx_sequence(
        args.stage2_sample,
        "adapter_context_motion",
        "adapter_context_smplx",
        max_frames=0,
    )
    stage1 = load_smplx_sequence(args.stage1_sample, "motion", "smplx", max_frames=0)
    stage2 = load_smplx_sequence(args.stage2_sample, "motion", "smplx", max_frames=0)

    total_frames, scaffold_delta = validate_alignment(
        gt,
        stage1_scaffold,
        stage2_scaffold,
        stage1,
        stage2,
        args.scaffold_tolerance,
    )
    stage1_name = _load_scalar(args.stage1_sample, "name")
    stage2_name = _load_scalar(args.stage2_sample, "name")
    if stage1_name != stage2_name:
        raise ValueError(
            f"Stage export names differ: Stage 1={stage1_name!r}, "
            f"Stage 2={stage2_name!r}"
        )
    for label, sample in (("Stage 1", args.stage1_sample), ("Stage 2", args.stage2_sample)):
        length_mode = _load_scalar(sample, "length_mode")
        if length_mode != "ground_truth":
            raise ValueError(
                f"{label} sample must use length_mode='ground_truth', got "
                f"{length_mode!r}"
            )

    sequences = [gt, stage1_scaffold, stage1, stage2]
    vertices = []
    faces = None
    for sequence in sequences:
        sequence_vertices, sequence_faces = prepare_vertices_raw(sequence, args)
        vertices.append(sequence_vertices)
        if faces is None:
            faces = sequence_faces
    vertices, faces = _normalize_panel_vertices(vertices, faces, args)

    if args.max_frames > 0:
        total_frames = min(total_frames, int(args.max_frames))
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    out_path = args.out_dir / (
        f"{args.stage1_sample.stem}_gt_scaffold_stage1_stage2.mp4"
    )
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    labels = [
        args.gt_label,
        args.scaffold_label,
        args.stage1_label,
        args.stage2_label,
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
                renderer.render(sequence_vertices[frame_index], color=color)
                for sequence_vertices, color in zip(vertices, colors)
            ]
            writer.append_data(draw_panel(panels, labels, frame_index, total_frames))
    finally:
        writer.close()

    print(
        f"Saved: {out_path}\n"
        f"Sequence: {stage1_name}\n"
        f"Frames: {total_frames}\n"
        f"Stage scaffold max_abs_delta: {scaffold_delta:.6g}"
    )


if __name__ == "__main__":
    main()
