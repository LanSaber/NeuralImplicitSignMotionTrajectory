#!/usr/bin/env python
"""Render sentence-level upper-body SMPL-X videos from posfix_contact word pickles."""
import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from flow.prepare_tracked_guard_dataset import extract_sequence
from flow.render import (
    SoftwareMeshRenderer,
    apply_view_transform,
    normalize_vertices,
    smplx182_to_vertices,
)
from flow.smplx_features import smplx182_from_compact


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/posfix_contact")
DEFAULT_OUT_DIR = Path("visualize/posfix_contact_sentence_concat")
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
        description="Concatenate posfix_contact word poses into sentence SMPL-X videos."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--coverage", type=Path, default=None)
    parser.add_argument("--sentence_ids", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking_file", default="optim_tracking_ehm.pkl")
    parser.add_argument("--expression_source", choices=["smplx", "flame"], default="smplx")
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 renders the full concatenated sentence.")
    parser.add_argument("--gap_frames", type=int, default=0, help="Optional neutral pause inserted between words.")
    parser.add_argument(
        "--compare_smoothing",
        action="store_true",
        help="Render boundary-trimmed direct concatenation beside trim/interpolation-smoothed concatenation.",
    )
    parser.add_argument(
        "--left_trim_frames",
        type=int,
        default=25,
        help="Frames trimmed from each direct-concat word boundary on the left comparison panel.",
    )
    parser.add_argument("--trim_frames", type=int, default=30)
    parser.add_argument("--transition_frames", type=int, default=10)
    parser.add_argument("--end_mode", default="blank", choices=["blank", "hold"])
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
    )
    parser.add_argument("--upper_body_only", dest="upper_body_only", action="store_true", default=True)
    parser.add_argument("--full_body", dest="upper_body_only", action="store_false")
    parser.add_argument("--no_render", action="store_true")
    return parser.parse_args()


def parse_coverage(path):
    text = Path(path).read_text(encoding="utf-8")
    sentences = {}
    for block in re.split(r"--- Sentence\s+", text)[1:]:
        header, body = block.split("---", 1)
        sentence_id = int(header.strip())
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        sentence_text = lines[0]
        with_line = next((line for line in lines if line.startswith("WITH video")), "")
        words = [Path(match).stem for match in re.findall(r"\[([^\]]+\.mp4)\]", with_line)]
        if not words:
            raise ValueError(f"Could not parse word list for sentence {sentence_id} in {path}")
        sentences[sentence_id] = {"text": sentence_text, "words": words}
    return sentences


def load_word_motion(data_dir, word, tracking_file, expression_source):
    tracking_path = data_dir / word / tracking_file
    if not tracking_path.is_file():
        raise FileNotFoundError(f"Missing tracking file for {word!r}: {tracking_path}")
    motion, left_valid, right_valid = extract_sequence(tracking_path, expression_source)
    return motion, left_valid, right_valid, tracking_path


def load_sentence_clips(data_dir, sentence, tracking_file, expression_source):
    clips = []
    for word in sentence["words"]:
        motion, left_valid, right_valid, tracking_path = load_word_motion(
            data_dir,
            word,
            tracking_file,
            expression_source,
        )
        clips.append(
            {
                "word": word,
                "motion": motion,
                "left_valid": left_valid,
                "right_valid": right_valid,
                "source": str(tracking_path),
            }
        )
    return clips


def neutral_gap(frame_dim, frames):
    if frames <= 0:
        return None
    return np.zeros((frames, frame_dim), dtype=np.float32)


def make_sentence_item(sentence_id, sentence, motion, left_valid, right_valid, boundaries, sources, stem, label):
    return {
        "sentence_id": sentence_id,
        "text": sentence["text"],
        "words": list(sentence["words"]),
        "motion": motion.astype(np.float32, copy=False),
        "left_valid": left_valid.astype(np.float32, copy=False),
        "right_valid": right_valid.astype(np.float32, copy=False),
        "boundaries": boundaries,
        "sources": sources,
        "stem": stem,
        "label": label,
    }


def build_sentence_from_clips(sentence_id, sentence, clips, gap_frames):
    motions = []
    left_valids = []
    right_valids = []
    boundaries = []
    cursor = 0
    gap = int(max(0, gap_frames))

    for word_idx, clip in enumerate(clips):
        motion = clip["motion"]
        start = cursor
        end = start + len(motion)
        boundaries.append(
            {
                "word": clip["word"],
                "start": start,
                "end": end,
                "index": word_idx,
                "kind": "word",
            }
        )
        motions.append(motion)
        left_valids.append(clip["left_valid"])
        right_valids.append(clip["right_valid"])
        cursor = end

        if gap > 0 and word_idx < len(clips) - 1:
            pause = neutral_gap(motion.shape[1], gap)
            motions.append(pause)
            left_valids.append(np.zeros(gap, dtype=np.float32))
            right_valids.append(np.zeros(gap, dtype=np.float32))
            cursor += gap

    return make_sentence_item(
        sentence_id,
        sentence,
        np.concatenate(motions, axis=0),
        np.concatenate(left_valids, axis=0),
        np.concatenate(right_valids, axis=0),
        boundaries,
        [clip["source"] for clip in clips],
        f"sentence_{sentence_id:02d}_concat",
        "Direct concatenation",
    )


def build_sentence(data_dir, sentence_id, sentence, tracking_file, expression_source, gap_frames):
    clips = load_sentence_clips(data_dir, sentence, tracking_file, expression_source)
    return build_sentence_from_clips(sentence_id, sentence, clips, gap_frames)


def build_trimmed_direct_sentence_from_clips(sentence_id, sentence, clips, trim_frames):
    trim = int(max(0, trim_frames))
    motions = []
    left_valids = []
    right_valids = []
    boundaries = []
    cursor = 0

    for word_idx, clip in enumerate(clips):
        motion = clip["motion"]
        start_trim = 0 if word_idx == 0 else trim
        end_trim = 0 if word_idx == len(clips) - 1 else trim
        if start_trim + end_trim >= len(motion):
            raise ValueError(
                f"Cannot trim {start_trim}+{end_trim} frames from {clip['word']!r} "
                f"with only {len(motion)} frames."
            )
        kept_motion = motion[start_trim : len(motion) - end_trim]
        kept_left = clip["left_valid"][start_trim : len(motion) - end_trim]
        kept_right = clip["right_valid"][start_trim : len(motion) - end_trim]
        start = cursor
        end = start + len(kept_motion)
        boundaries.append(
            {
                "word": clip["word"],
                "start": start,
                "end": end,
                "index": word_idx,
                "kind": "word",
            }
        )
        motions.append(kept_motion)
        left_valids.append(kept_left)
        right_valids.append(kept_right)
        cursor = end

    return make_sentence_item(
        sentence_id,
        sentence,
        np.concatenate(motions, axis=0),
        np.concatenate(left_valids, axis=0),
        np.concatenate(right_valids, axis=0),
        boundaries,
        [clip["source"] for clip in clips],
        f"sentence_{sentence_id:02d}_directtrim{trim}",
        f"Direct trim {trim}",
    )


def interpolate_rows(left, right, frames):
    frames = int(frames)
    if frames <= 0:
        return np.zeros((0, left.shape[-1]), dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, frames + 2, dtype=np.float32)[1:-1, None]
    return ((1.0 - alpha) * left[None] + alpha * right[None]).astype(np.float32)


def build_smoothed_sentence_from_clips(sentence_id, sentence, clips, trim_frames, transition_frames):
    trim = int(max(0, trim_frames))
    transition = int(max(0, transition_frames))
    motions = []
    left_valids = []
    right_valids = []
    boundaries = []
    cursor = 0

    for word_idx, clip in enumerate(clips):
        motion = clip["motion"]
        start_trim = 0 if word_idx == 0 else trim
        end_trim = 0 if word_idx == len(clips) - 1 else trim
        if start_trim + end_trim >= len(motion):
            raise ValueError(
                f"Cannot trim {start_trim}+{end_trim} frames from {clip['word']!r} "
                f"with only {len(motion)} frames."
            )
        kept_motion = motion[start_trim : len(motion) - end_trim]
        kept_left = clip["left_valid"][start_trim : len(motion) - end_trim]
        kept_right = clip["right_valid"][start_trim : len(motion) - end_trim]
        start = cursor
        end = start + len(kept_motion)
        boundaries.append(
            {
                "word": clip["word"],
                "start": start,
                "end": end,
                "index": word_idx,
                "kind": "word",
            }
        )
        motions.append(kept_motion)
        left_valids.append(kept_left)
        right_valids.append(kept_right)
        cursor = end

        if word_idx < len(clips) - 1 and transition > 0:
            next_clip = clips[word_idx + 1]
            next_start_trim = trim
            if next_start_trim >= len(next_clip["motion"]):
                raise ValueError(f"Cannot trim {next_start_trim} frames from {next_clip['word']!r}.")
            interp = interpolate_rows(kept_motion[-1], next_clip["motion"][next_start_trim], transition)
            start = cursor
            end = start + len(interp)
            boundaries.append(
                {
                    "word": f"{clip['word']}->{next_clip['word']}",
                    "start": start,
                    "end": end,
                    "index": word_idx,
                    "kind": "transition",
                }
            )
            motions.append(interp)
            left_valids.append(
                interpolate_rows(
                    kept_left[-1:].astype(np.float32),
                    next_clip["left_valid"][next_start_trim : next_start_trim + 1].astype(np.float32),
                    transition,
                ).reshape(-1)
            )
            right_valids.append(
                interpolate_rows(
                    kept_right[-1:].astype(np.float32),
                    next_clip["right_valid"][next_start_trim : next_start_trim + 1].astype(np.float32),
                    transition,
                ).reshape(-1)
            )
            cursor = end

    return make_sentence_item(
        sentence_id,
        sentence,
        np.concatenate(motions, axis=0),
        np.concatenate(left_valids, axis=0),
        np.concatenate(right_valids, axis=0),
        boundaries,
        [clip["source"] for clip in clips],
        f"sentence_{sentence_id:02d}_trim{trim}_interp{transition}",
        f"Trim {trim} + interpolate {transition}",
    )


def save_sentence_npz(out_dir, item, fps):
    smplx = smplx182_from_compact(item["motion"])
    stem = item.get("stem") or f"sentence_{item['sentence_id']:02d}_concat"
    path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        path,
        motion=item["motion"],
        smplx=smplx,
        left_valid=item["left_valid"],
        right_valid=item["right_valid"],
        text=item["text"],
        words=np.asarray(item["words"]),
        boundaries=np.asarray(
            [
                (entry["word"], entry["start"], entry["end"], entry.get("kind", "word"))
                for entry in item["boundaries"]
            ],
            dtype=object,
        ),
        source_paths=np.asarray(item["sources"]),
        label=item.get("label", ""),
        fps=np.asarray(fps, dtype=np.int32),
    )
    return path, smplx


def load_upper_body_faces(faces, model_dir):
    seg_path = Path(model_dir) / "smplx_vert_segmentation.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"Missing SMPL-X segmentation file: {seg_path}")

    import json

    with seg_path.open("r", encoding="utf-8") as handle:
        segments = json.load(handle)

    indices = set()
    for part in UPPER_BODY_PARTS:
        indices.update(int(index) for index in segments.get(part, []))
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
    if extent > 1e-6:
        vertices *= float(target_height) / extent
    return vertices


def word_for_frame(boundaries, frame_idx):
    for entry in boundaries:
        if entry["start"] <= frame_idx < entry["end"]:
            return entry
    return None


def frame_label(item, frame_idx):
    current = word_for_frame(item["boundaries"], frame_idx)
    if current is None:
        return "pause"
    if current.get("kind") == "transition":
        return f"transition: {current['word']}"
    return f"word {current['index'] + 1}/{len(item['words'])}: {current['word']}"


def draw_panel(frame, item, frame_idx, total_frames):
    header_h = 56
    h, w = frame.shape[:2]
    canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    canvas[:header_h] = np.array([18, 20, 24], dtype=np.uint8)
    canvas[header_h:] = frame

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    word_label = frame_label(item, frame_idx)
    draw.text(
        (12, 8),
        f"Sentence {item['sentence_id']:02d} {item.get('label', 'concatenated word poses')} | {word_label}"[:92],
        fill=(238, 238, 238),
        font=font,
    )
    draw.text(
        (12, 32),
        f"frame {frame_idx + 1}/{total_frames}",
        fill=(185, 190, 198),
        font=font,
    )
    return np.asarray(image)


def prepare_vertices(args, smplx_sequences):
    lengths = [len(smplx) if args.max_frames <= 0 else min(len(smplx), args.max_frames) for smplx in smplx_sequences]
    clipped = [smplx[:length] for smplx, length in zip(smplx_sequences, lengths)]
    combined = np.concatenate(clipped, axis=0)
    vertices, faces = smplx182_to_vertices(
        combined,
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

    out = []
    cursor = 0
    for length in lengths:
        out.append(vertices[cursor : cursor + length])
        cursor += length
    return out, faces


def render_or_empty(renderer, vertices, frame_idx, end_mode, color):
    if frame_idx < len(vertices):
        return renderer.render(vertices[frame_idx], color=color)
    if end_mode == "hold" and len(vertices) > 0:
        return renderer.render(vertices[-1], color=color)
    return np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)


def draw_compare_panel(left_frame, right_frame, left_item, right_item, frame_idx, total_frames):
    header_h = 56
    h, w = left_frame.shape[:2]
    canvas = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)
    canvas[:header_h] = np.array([18, 20, 24], dtype=np.uint8)
    canvas[header_h:, :w] = left_frame
    canvas[header_h:, w:] = right_frame

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left_title = f"Left: {left_item.get('label', 'Direct')} | {frame_label(left_item, frame_idx)}"
    right_title = f"Right: {right_item.get('label', 'Smoothed')} | {frame_label(right_item, frame_idx)}"
    draw.text((12, 8), left_title[:92], fill=(238, 238, 238), font=font)
    draw.text((w + 12, 8), right_title[:92], fill=(238, 238, 238), font=font)
    draw.text((12, 32), f"frame {frame_idx + 1}/{total_frames}", fill=(185, 190, 198), font=font)
    draw.text((w + 12, 32), f"frame {frame_idx + 1}/{total_frames}", fill=(185, 190, 198), font=font)
    return np.asarray(image)


def render_compare_video(args, left_item, left_smplx, right_item, right_smplx):
    (left_vertices, right_vertices), faces = prepare_vertices(args, [left_smplx, right_smplx])
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    total_frames = max(len(left_vertices), len(right_vertices))
    left_desc = f"directtrim{args.left_trim_frames}" if args.left_trim_frames > 0 else "concat"
    stem = (
        f"sentence_{left_item['sentence_id']:02d}_{left_desc}_vs_"
        f"trim{args.trim_frames}_interp{args.transition_frames}_upper"
    )
    out_path = args.out_dir / f"{stem}.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_idx in tqdm(range(total_frames), desc=f"render {out_path.name}"):
            left_frame = render_or_empty(
                renderer,
                left_vertices,
                frame_idx,
                args.end_mode,
                color=(0.48, 0.78, 1.0, 1.0),
            )
            right_frame = render_or_empty(
                renderer,
                right_vertices,
                frame_idx,
                args.end_mode,
                color=(0.58, 0.95, 0.70, 1.0),
            )
            writer.append_data(
                draw_compare_panel(left_frame, right_frame, left_item, right_item, frame_idx, total_frames)
            )
    finally:
        writer.close()
    return out_path


def render_sentence_video(args, item, smplx):
    if args.max_frames > 0:
        smplx = smplx[: args.max_frames]
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

    total_frames = len(vertices)
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    stem = item.get("stem") or f"sentence_{item['sentence_id']:02d}_concat"
    out_path = args.out_dir / f"{stem}_upper.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_idx in tqdm(range(total_frames), desc=f"render {out_path.name}"):
            frame = renderer.render(vertices[frame_idx], color=(0.48, 0.78, 1.0, 1.0))
            writer.append_data(draw_panel(frame, item, frame_idx, total_frames))
    finally:
        writer.close()
    return out_path


def main():
    args = parse_args()
    args.coverage = args.coverage or args.data_dir / "SENTENCES_coverage.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sentences = parse_coverage(args.coverage)

    for sentence_id in args.sentence_ids:
        if sentence_id not in sentences:
            raise KeyError(f"Sentence {sentence_id} not found in {args.coverage}")
        sentence = sentences[sentence_id]
        clips = load_sentence_clips(args.data_dir, sentence, args.tracking_file, args.expression_source)
        if args.compare_smoothing:
            item = build_trimmed_direct_sentence_from_clips(
                sentence_id,
                sentence,
                clips,
                args.left_trim_frames,
            )
            npz_path, smplx = save_sentence_npz(args.out_dir, item, args.fps)
            print(
                f"Saved {npz_path}: {len(item['words'])} words, "
                f"{len(item['motion'])} frames, {len(item['motion']) / args.fps:.2f}s"
            )
            smooth_item = build_smoothed_sentence_from_clips(
                sentence_id,
                sentence,
                clips,
                args.trim_frames,
                args.transition_frames,
            )
            smooth_npz_path, smooth_smplx = save_sentence_npz(args.out_dir, smooth_item, args.fps)
            print(
                f"Saved {smooth_npz_path}: {len(smooth_item['words'])} words, "
                f"{len(smooth_item['motion'])} frames, {len(smooth_item['motion']) / args.fps:.2f}s"
            )
            if len(item["motion"]) != len(smooth_item["motion"]):
                print(
                    "Warning: comparison sequences have different frame counts: "
                    f"left={len(item['motion'])}, right={len(smooth_item['motion'])}"
                )
            if not args.no_render:
                out_path = render_compare_video(args, item, smplx, smooth_item, smooth_smplx)
                print(f"Saved {out_path}")
        else:
            item = build_sentence_from_clips(sentence_id, sentence, clips, args.gap_frames)
            npz_path, smplx = save_sentence_npz(args.out_dir, item, args.fps)
            print(
                f"Saved {npz_path}: {len(item['words'])} words, "
                f"{len(item['motion'])} frames, {len(item['motion']) / args.fps:.2f}s"
            )
            if not args.no_render:
                out_path = render_sentence_video(args, item, smplx)
                print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
