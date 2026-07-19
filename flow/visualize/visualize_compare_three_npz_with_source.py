#!/usr/bin/env python
"""Render source video, GT, flow output, and word-prior SMPL-X sequences.

The PHOENIX `spa_feat_p14t` directory stores per-frame visual features
(`T x 2048` float arrays), not RGB frames. This visualizer can inspect that
kind of feature root and explain why it cannot be shown as original video, and
it will add a source panel when `--source_root` points to real RGB images or a
video file.
"""
import argparse
import json
import pickle
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SOURCE_NAME_KEYS = ("name", "source_name", "sequence_name", "video_id", "id")
PANEL_NAMES = ("source", "gt", "prior", "soke", "pred")
NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "dark": (12, 14, 18),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render GT, flow prediction, and residual word-prior SMPL-X sequences "
            "side by side, optionally with an original RGB/video source panel."
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
    parser.add_argument("--out_dir", type=Path, default=Path("visualize/flow/flow_compare_three_source"))
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
    parser.add_argument("--pred_label", default="SoftArrangerFlow")
    parser.add_argument("--prior_label", default="Word Prior")
    parser.add_argument("--soke_label", default="Original SOKE")
    parser.add_argument("--source_label", default="Original video")
    parser.add_argument(
        "--append_file_names",
        action="store_true",
        help="Append source sequence and sample file names to panel labels.",
    )
    parser.add_argument(
        "--source_fit",
        default="contain",
        choices=["contain", "cover"],
        help="Resize original frames to fit inside the panel or crop-cover the full panel.",
    )
    parser.add_argument(
        "--resample_to_source_frames",
        action="store_true",
        help=(
            "Use the original source frame count as the output timeline and linearly "
            "resample GT, word-prior, SOKE, and flow mesh sequences to that length."
        ),
    )
    parser.add_argument(
        "--background_color",
        default="12,14,18",
        help="Panel background color: named color like white, or RGB as R,G,B.",
    )
    parser.add_argument(
        "--panel_order",
        nargs="+",
        choices=PANEL_NAMES,
        default=None,
        help="Optional explicit panel order, e.g. source gt prior soke pred.",
    )
    parser.add_argument(
        "--upper_body_only",
        action="store_true",
        help="Render only upper-body SMPL-X mesh faces using smplx_vert_segmentation.json.",
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=None,
        help=(
            "Optional root containing RGB frames or videos. Expected examples: "
            "root/split/name/*.jpg, root/split/name.mp4, root/name/*.png, or an exact video file."
        ),
    )
    parser.add_argument(
        "--source_split",
        default="",
        help="Optional split folder under --source_root, for example train, dev, or test.",
    )
    parser.add_argument(
        "--source_name",
        default="",
        help="Optional source sequence name override. Defaults to the `name` field in the .npz.",
    )
    parser.add_argument(
        "--source_position",
        default="left",
        choices=["left", "right"],
        help="Where to place the source panel when source RGB frames are found.",
    )
    parser.add_argument(
        "--allow_missing_source",
        action="store_true",
        help="Continue with the three SMPL-X panels when --source_root has no RGB frames/video.",
    )
    parser.add_argument(
        "--soke_vertices",
        type=Path,
        default=None,
        help=(
            "Optional SOKE generated-vertices .npz file, or a directory containing "
            "*_soke_vertices.npz files from test/visualize_soke_test.py --save_npz."
        ),
    )
    parser.add_argument(
        "--soke_vertices_key",
        default="vertices_soke",
        help="Array key to read from --soke_vertices.",
    )
    parser.add_argument(
        "--meta_filename",
        default="render_meta.json",
        help="Metadata JSON filename written under --out_dir. Use --no_meta to disable.",
    )
    parser.add_argument(
        "--no_meta",
        action="store_true",
        help="Do not write or update render metadata in --out_dir.",
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


def parse_rgb_color(value):
    value = str(value).strip().lower()
    if value in NAMED_COLORS:
        return NAMED_COLORS[value]
    parts = value.replace(";", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"Color must be a name or R,G,B triplet, got: {value}")
    rgb = tuple(int(part.strip()) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"RGB color channels must be in [0, 255], got: {value}")
    return rgb


def text_colors_for_background(rgb):
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    if luminance > 150:
        return (28, 32, 38), (92, 98, 110), (220, 224, 230)
    return (238, 238, 238), (185, 190, 198), (42, 46, 54)


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


def prepare_precomputed_vertices(vertices, args, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices with shape [T, V, 3], got {vertices.shape}")
    vertices = apply_view_transform(vertices, args.view_transform)
    if args.upper_body_only:
        _, upper_indices = load_upper_body_faces(faces, args.model_dir)
        vertices = normalize_vertices_by_indices(vertices, upper_indices)
    else:
        vertices = normalize_vertices(vertices)
    return vertices


def read_npz_string(data, key):
    if key not in data.files:
        return ""
    value = npz_scalar_to_string(data[key])
    return value or ""


def resolve_soke_vertices_path(path, source_name):
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"SOKE vertices path does not exist: {path}")

    npz_files = sorted(path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in SOKE vertices directory: {path}")

    safe_source = source_name.replace("/", "_").replace("\\", "_")
    matches = [candidate for candidate in npz_files if source_name in candidate.stem or safe_source in candidate.stem]
    if matches:
        return matches[0]
    if len(npz_files) == 1:
        return npz_files[0]
    raise FileNotFoundError(
        f"Could not find a SOKE vertices file matching source name '{source_name}' in {path}"
    )


def load_soke_vertices(args, faces, source_name):
    if args.soke_vertices is None:
        return None, None

    soke_path = resolve_soke_vertices_path(args.soke_vertices, source_name)
    with np.load(soke_path, allow_pickle=True) as data:
        key = args.soke_vertices_key
        if key not in data.files:
            for fallback in ("vertices_soke", "vertices_rst", "vertices_pred"):
                if fallback in data.files:
                    key = fallback
                    break
        if key not in data.files:
            raise KeyError(f"{soke_path} has no SOKE vertices key '{args.soke_vertices_key}'")
        vertices = np.asarray(data[key], dtype=np.float32)
        if "length_soke" in data.files:
            vertices = vertices[: int(np.asarray(data["length_soke"]).item())]
        soke_name = read_npz_string(data, "name")

    if soke_name and source_name and soke_name != source_name and Path(soke_name).name != source_name:
        print(
            "Warning: SOKE vertices name does not match flow sample: "
            f"SOKE='{soke_name}', flow='{source_name}'"
        )
    print(f"Loading SOKE vertices: {soke_path} ({len(vertices)} frames)")
    return prepare_precomputed_vertices(vertices, args, faces), soke_path


def render_or_empty(renderer, vertices, frame_idx, end_mode, color, background_rgb):
    if frame_idx < len(vertices):
        return renderer.render(vertices[frame_idx], color=color)
    if end_mode == "hold" and len(vertices) > 0:
        return renderer.render(vertices[-1], color=color)
    frame = np.empty((renderer.height, renderer.width, 3), dtype=np.uint8)
    frame[:, :] = np.asarray(background_rgb, dtype=np.uint8)
    return frame


def npz_scalar_to_string(value):
    array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
    elif array.size == 1:
        scalar = array.reshape(-1)[0]
    else:
        return None
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    return str(scalar)


def read_source_name(npz_path, fallback=""):
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            for key in SOURCE_NAME_KEYS:
                if key in data.files:
                    value = npz_scalar_to_string(data[key])
                    if value:
                        return value
    except Exception as exc:
            print(f"Warning: could not read source name from {npz_path}: {exc}")
    return fallback


def npz_scalar_value(data, key):
    if key not in data.files:
        return None
    array = np.asarray(data[key])
    if array.shape == ():
        value = array.item()
    elif array.size == 1:
        value = array.reshape(-1)[0]
    else:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_npz_metadata(npz_path):
    if npz_path is None:
        return {}
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        return {}
    keys = (
        "name",
        "text",
        "raw_text",
        "gloss",
        "label_word",
        "condition_field",
        "source_name",
        "source_split",
        "src",
        "dataset",
        "signer",
        "motion_path",
        "length",
        "fps",
        "duration",
        "num_frames",
        "source_fps",
        "source_num_frames",
        "source_pose_frames",
        "length_soke",
    )
    metadata = {}
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            for key in keys:
                value = npz_scalar_value(data, key)
                if value is not None and value != "":
                    metadata[key] = value
    except Exception as exc:
        metadata["read_error"] = str(exc)
    return metadata


def first_metadata_value(*metas, key):
    for meta in metas:
        value = meta.get(key)
        if value is not None and value != "":
            return value
    return ""


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def update_render_metadata(args, render_meta):
    if args.no_meta:
        return None
    meta_path = args.out_dir / args.meta_filename
    records = []
    if meta_path.is_file():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("renders"), list):
                records = existing["renders"]
            elif isinstance(existing, list):
                records = existing
        except Exception as exc:
            print(f"Warning: could not read existing metadata {meta_path}: {exc}")

    render_meta = json_safe(render_meta)
    output_video = render_meta.get("output_video", "")
    records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("output_video") != output_video
    ]
    records.append(render_meta)
    payload = {"renders": records}
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def unique_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def source_candidates(source_root, source_split, source_name):
    root = Path(source_root)
    if root.is_file():
        return [root]

    names = [source_name]
    if "/" in source_name:
        names.append(source_name.split("/")[-1])
    if "\\" in source_name:
        names.append(source_name.split("\\")[-1])

    bases = []
    if source_split:
        bases.append(root / source_split)
    bases.append(root)

    candidates = []
    for base in bases:
        for name in names:
            if not name:
                continue
            candidates.append(base / name)
            candidates.append(base / f"{name}.npy")
            for suffix in sorted(VIDEO_SUFFIXES):
                candidates.append(base / f"{name}{suffix}")
    return unique_paths(candidates)


def sorted_media_files(directory, suffixes):
    files = [path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    return sorted(files, key=lambda path: path.name)


def image_resample_filter():
    return getattr(Image, "Resampling", Image).LANCZOS


def array_to_panel_frame(frame, width, height, fit="contain", background_rgb=(0, 0, 0)):
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected an image-like array, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)

    image = Image.fromarray(array)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    resample = image_resample_filter()
    if fit == "cover":
        scale = max(width / image.width, height / image.height)
        resized_size = (
            max(width, int(image.width * scale + 0.5)),
            max(height, int(image.height * scale + 0.5)),
        )
        image = image.resize(resized_size, resample)
        left = max((image.width - width) // 2, 0)
        top = max((image.height - height) // 2, 0)
        image = image.crop((left, top, left + width, top + height))
        if image.mode == "RGBA":
            canvas = Image.new("RGB", (width, height), background_rgb)
            canvas.paste(image, (0, 0), image)
            image = canvas
        else:
            image = image.convert("RGB")
        return np.asarray(image)

    image.thumbnail((width, height), resample)
    canvas = Image.new("RGB", (width, height), background_rgb)
    offset = ((width - image.width) // 2, (height - image.height) // 2)
    if image.mode == "RGBA":
        canvas.paste(image, offset, image)
    else:
        canvas.paste(image, offset)
    return np.asarray(canvas)


def sample_indices(source_len, target_len):
    if source_len <= 0 or target_len <= 0:
        return []
    if target_len == 1:
        return [0]
    return np.linspace(0, source_len - 1, target_len).round().astype(np.int64).tolist()


def resample_sequence_linear(sequence, target_len):
    sequence = np.asarray(sequence, dtype=np.float32)
    source_len = len(sequence)
    if target_len <= 0:
        return sequence[:0]
    if source_len == target_len:
        return sequence
    if source_len <= 0:
        raise ValueError("Cannot resample an empty sequence.")
    if source_len == 1:
        return np.repeat(sequence, target_len, axis=0)

    positions = np.linspace(0, source_len - 1, target_len, dtype=np.float32)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, source_len - 1)
    alpha_shape = (target_len,) + (1,) * (sequence.ndim - 1)
    alpha = (positions - left).reshape(alpha_shape).astype(np.float32)
    return sequence[left] * (1.0 - alpha) + sequence[right] * alpha


def count_video_frames(video_path):
    reader = imageio.get_reader(str(video_path))
    try:
        try:
            count = reader.count_frames()
            if count > 0:
                return int(count)
        except Exception:
            pass
        return sum(1 for _ in reader)
    finally:
        reader.close()


def source_candidate_frame_count(candidate):
    suffix = candidate.suffix.lower()
    if candidate.is_dir():
        image_files = sorted_media_files(candidate, IMAGE_SUFFIXES)
        if image_files:
            return len(image_files)
        video_files = sorted_media_files(candidate, VIDEO_SUFFIXES)
        if video_files:
            return count_video_frames(video_files[0])
        return None
    if suffix in IMAGE_SUFFIXES:
        return 1
    if suffix in VIDEO_SUFFIXES:
        return count_video_frames(candidate)
    if suffix == ".npy":
        array = np.load(candidate, mmap_mode="r")
        if array.ndim == 4 and array.shape[-1] in (3, 4):
            return int(array.shape[0])
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            return 1
    return None


def source_frame_count(args, source_name):
    reasons = []
    for candidate in source_candidates(args.source_root, args.source_split, source_name):
        if not candidate.exists():
            continue
        try:
            count = source_candidate_frame_count(candidate)
        except Exception as exc:
            reasons.append(f"{candidate}: {exc}")
            continue
        if count is not None and count > 0:
            print(f"Using source frame count from {candidate}: {count} frames")
            return count
        reasons.append(inspect_non_rgb_candidate(candidate))

    message = (
        f"No RGB source frames/video found for source name '{source_name}' under {args.source_root}."
    )
    if reasons:
        message += "\n" + "\n".join(f"  - {reason}" for reason in reasons[:6])
    raise FileNotFoundError(message)


def load_image_sequence_panel(image_files, total_frames, width, height, fit="contain", background_rgb=(0, 0, 0)):
    indices = sample_indices(len(image_files), total_frames)
    frames = []
    for index in indices:
        with Image.open(image_files[index]) as image:
            frames.append(array_to_panel_frame(np.asarray(image), width, height, fit, background_rgb))
    return frames


def load_video_panel(video_path, total_frames, width, height, fit="contain", background_rgb=(0, 0, 0)):
    reader = imageio.get_reader(str(video_path))
    try:
        raw_frames = [frame for frame in reader]
    finally:
        reader.close()
    indices = sample_indices(len(raw_frames), total_frames)
    return [array_to_panel_frame(raw_frames[index], width, height, fit, background_rgb) for index in indices]


def load_npy_image_panel(npy_path, total_frames, width, height, fit="contain", background_rgb=(0, 0, 0)):
    array = np.load(npy_path, mmap_mode="r")
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return [array_to_panel_frame(array, width, height, fit, background_rgb)] * total_frames
    if array.ndim == 4 and array.shape[-1] in (3, 4):
        indices = sample_indices(array.shape[0], total_frames)
        return [array_to_panel_frame(array[index], width, height, fit, background_rgb) for index in indices]
    raise ValueError(
        f"{npy_path} is not RGB video data; shape={array.shape}, dtype={array.dtype}. "
        "PHOENIX spa_feat_p14t files are usually T x 2048 feature arrays."
    )


def describe_pickle(path):
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        return f"{path} could not be read as pickle: {exc}"
    if isinstance(value, dict):
        keys = ", ".join(list(value.keys())[:12])
        return f"{path} is a pickle dict with keys [{keys}], not an RGB image."
    array = np.asarray(value)
    return f"{path} pickle payload has shape {array.shape} and type {type(value).__name__}."


def inspect_non_rgb_candidate(path):
    if path.is_dir():
        pkl_files = sorted(path.glob("*.pkl"))
        if pkl_files:
            return describe_pickle(pkl_files[0])
        return f"{path} has no supported image or video files."
    if path.suffix.lower() == ".npy":
        try:
            array = np.load(path, mmap_mode="r")
            return f"{path} is a .npy array with shape {array.shape}, dtype={array.dtype}, not RGB frames."
        except Exception as exc:
            return f"{path} could not be inspected as .npy: {exc}"
    if path.suffix.lower() == ".pkl":
        return describe_pickle(path)
    return f"{path} exists but is not a supported RGB image/video source."


def load_source_panel(args, source_name, total_frames):
    if args.source_root is None:
        return None

    reasons = []
    for candidate in source_candidates(args.source_root, args.source_split, source_name):
        if not candidate.exists():
            continue
        suffix = candidate.suffix.lower()
        try:
            if candidate.is_dir():
                image_files = sorted_media_files(candidate, IMAGE_SUFFIXES)
                if image_files:
                    print(f"Loading source image sequence: {candidate} ({len(image_files)} frames)")
                    return load_image_sequence_panel(
                        image_files,
                        total_frames,
                        args.width,
                        args.height,
                        fit=args.source_fit,
                        background_rgb=args.background_rgb,
                    )
                video_files = sorted_media_files(candidate, VIDEO_SUFFIXES)
                if video_files:
                    print(f"Loading source video: {video_files[0]}")
                    return load_video_panel(
                        video_files[0],
                        total_frames,
                        args.width,
                        args.height,
                        fit=args.source_fit,
                        background_rgb=args.background_rgb,
                    )
                reasons.append(inspect_non_rgb_candidate(candidate))
            elif suffix in IMAGE_SUFFIXES:
                print(f"Loading source image: {candidate}")
                return load_image_sequence_panel(
                    [candidate],
                    total_frames,
                    args.width,
                    args.height,
                    fit=args.source_fit,
                    background_rgb=args.background_rgb,
                )
            elif suffix in VIDEO_SUFFIXES:
                print(f"Loading source video: {candidate}")
                return load_video_panel(
                    candidate,
                    total_frames,
                    args.width,
                    args.height,
                    fit=args.source_fit,
                    background_rgb=args.background_rgb,
                )
            elif suffix == ".npy":
                print(f"Inspecting source .npy: {candidate}")
                return load_npy_image_panel(
                    candidate,
                    total_frames,
                    args.width,
                    args.height,
                    fit=args.source_fit,
                    background_rgb=args.background_rgb,
                )
            else:
                reasons.append(inspect_non_rgb_candidate(candidate))
        except ValueError as exc:
            reasons.append(str(exc))

    message = (
        f"No RGB source frames/video found for source name '{source_name}' under {args.source_root}."
    )
    if reasons:
        message += "\n" + "\n".join(f"  - {reason}" for reason in reasons[:6])
    if args.allow_missing_source:
        print(f"Warning: {message}")
        return None
    raise FileNotFoundError(message)


def draw_panel(frames, labels, frame_idx, total_frames, background_rgb, text_rgb, subtext_rgb, separator_rgb):
    header_h = 56
    h, w = frames[0].shape[:2]
    canvas = np.empty((h + header_h, w * len(frames), 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(background_rgb, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        canvas[header_h:, idx * w : (idx + 1) * w] = frame

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, label in enumerate(labels):
        x0 = idx * w
        if idx > 0:
            draw.line([(x0, 0), (x0, h + header_h)], fill=separator_rgb, width=1)
        for line_idx, line in enumerate(str(label).splitlines()[:2]):
            fill = text_rgb if line_idx == 0 else subtext_rgb
            draw.text((x0 + 12, 8 + line_idx * 16), line[:92], fill=fill, font=font)
    draw.line([(0, header_h - 1), (w * len(frames), header_h - 1)], fill=separator_rgb, width=1)
    frame_text_y = 42 if any("\n" in str(label) for label in labels) else 32
    draw.text((12, frame_text_y), f"frame {frame_idx + 1}/{total_frames}", fill=subtext_rgb, font=font)
    return np.asarray(image)


def write_video(args, gt_vertices, pred_vertices, prior_vertices, pred_path, prior_path, faces):
    renderer = SoftwareMeshRenderer(
        faces,
        width=args.width,
        height=args.height,
        face_stride=args.software_face_stride,
    )
    renderer.background = args.background_rgb
    source_name = args.source_name or read_source_name(pred_path, fallback=pred_path.stem)
    soke_vertices, soke_path = load_soke_vertices(args, faces, source_name)
    gt_meta = read_npz_metadata(args.gt)
    pred_meta = read_npz_metadata(pred_path)
    prior_meta = read_npz_metadata(prior_path)
    soke_meta = read_npz_metadata(soke_path)
    source_frame_count_value = None
    original_lengths = {
        "gt": len(gt_vertices),
        "flow": len(pred_vertices),
        "word": len(prior_vertices),
    }
    if soke_vertices is not None:
        original_lengths["soke"] = len(soke_vertices)

    if args.resample_to_source_frames:
        if args.source_root is None:
            raise ValueError("--resample_to_source_frames requires --source_root.")
        source_frame_count_value = source_frame_count(args, source_name)
        total_frames = source_frame_count_value
    else:
        sequence_lengths = [len(gt_vertices), len(pred_vertices), len(prior_vertices)]
        if soke_vertices is not None:
            sequence_lengths.append(len(soke_vertices))
        total_frames = max(sequence_lengths)
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    if args.resample_to_source_frames:
        gt_vertices = resample_sequence_linear(gt_vertices, total_frames)
        pred_vertices = resample_sequence_linear(pred_vertices, total_frames)
        prior_vertices = resample_sequence_linear(prior_vertices, total_frames)
        if soke_vertices is not None:
            soke_vertices = resample_sequence_linear(soke_vertices, total_frames)
        print(
            "Resampled mesh timelines to source length "
            f"{total_frames}: "
            + ", ".join(f"{name} {length}->{total_frames}" for name, length in original_lengths.items())
        )

    source_frames = load_source_panel(args, source_name, total_frames)
    has_source = source_frames is not None
    has_soke = soke_vertices is not None

    source_label = args.source_label
    if args.append_file_names:
        source_label = f"{source_label}: {source_name}"
    labels_by_panel = {
        "gt": args.gt_label,
        "pred": f"{args.pred_label}: {pred_path.stem}" if args.append_file_names else args.pred_label,
        "prior": args.prior_label,
    }
    if has_soke:
        labels_by_panel["soke"] = (
            f"{args.soke_label}: {soke_path.stem}"
            if args.append_file_names and soke_path is not None
            else args.soke_label
        )
    if has_source:
        labels_by_panel["source"] = source_label

    if args.panel_order:
        ordered_panels = list(args.panel_order)
        missing = [panel for panel in ordered_panels if panel not in labels_by_panel]
        if missing:
            raise RuntimeError(f"Requested unavailable panel(s): {', '.join(missing)}")
    else:
        ordered_panels = ["gt", "pred", "prior"]
        if has_soke:
            ordered_panels.append("soke")
        if args.source_position == "left":
            ordered_panels = ["source"] + ordered_panels if has_source else ordered_panels
        else:
            ordered_panels = ordered_panels + ["source"] if has_source else ordered_panels

    suffix_map = {"source": "source", "gt": "gt", "pred": "flow", "prior": "word", "soke": "soke"}
    out_suffix = "_".join(suffix_map[panel] for panel in ordered_panels)
    out_path = args.out_dir / f"{pred_path.stem}_{out_suffix}.mp4"
    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    labels = [labels_by_panel[panel] for panel in ordered_panels]

    try:
        for frame_idx in tqdm(range(total_frames), desc=f"render {out_path.name}"):
            gt_frame = render_or_empty(
                renderer,
                gt_vertices,
                frame_idx,
                args.end_mode,
                color=(1.0, 0.92, 0.72, 1.0),
                background_rgb=args.background_rgb,
            )
            pred_frame = render_or_empty(
                renderer,
                pred_vertices,
                frame_idx,
                args.end_mode,
                color=(0.48, 0.78, 1.0, 1.0),
                background_rgb=args.background_rgb,
            )
            prior_frame = render_or_empty(
                renderer,
                prior_vertices,
                frame_idx,
                args.end_mode,
                color=(0.58, 0.95, 0.70, 1.0),
                background_rgb=args.background_rgb,
            )
            if has_soke:
                soke_frame = render_or_empty(
                    renderer,
                    soke_vertices,
                    frame_idx,
                    args.end_mode,
                    color=(0.95, 0.70, 0.95, 1.0),
                    background_rgb=args.background_rgb,
                )
            frames_by_panel = {
                "gt": gt_frame,
                "pred": pred_frame,
                "prior": prior_frame,
            }
            if has_source:
                frames_by_panel["source"] = source_frames[frame_idx]
            if has_soke:
                frames_by_panel["soke"] = soke_frame
            frames = [frames_by_panel[panel] for panel in ordered_panels]
            writer.append_data(
                draw_panel(
                    frames,
                    labels,
                    frame_idx,
                    total_frames,
                    args.background_rgb,
                    args.text_rgb,
                    args.subtext_rgb,
                    args.separator_rgb,
                )
            )
    finally:
        writer.close()
    render_meta = {
        "output_video": str(out_path),
        "sequence": {
            "name": source_name,
            "text": first_metadata_value(pred_meta, gt_meta, soke_meta, key="text"),
            "raw_text": first_metadata_value(pred_meta, gt_meta, soke_meta, key="raw_text"),
            "gloss": first_metadata_value(pred_meta, gt_meta, soke_meta, key="gloss"),
            "label_word": first_metadata_value(pred_meta, gt_meta, soke_meta, key="label_word"),
            "condition_field": first_metadata_value(pred_meta, gt_meta, soke_meta, key="condition_field"),
            "dataset": first_metadata_value(pred_meta, gt_meta, soke_meta, key="dataset"),
            "signer": first_metadata_value(pred_meta, gt_meta, soke_meta, key="signer"),
        },
        "inputs": {
            "gt": str(args.gt),
            "pred": str(pred_path),
            "prior": str(prior_path),
            "soke_vertices": str(soke_path) if soke_path is not None else "",
            "source_root": str(args.source_root) if args.source_root is not None else "",
            "source_split": args.source_split,
        },
        "npz_metadata": {
            "gt": gt_meta,
            "pred": pred_meta,
            "prior": prior_meta,
            "soke": soke_meta,
        },
        "layout": {
            "panel_order": ordered_panels,
            "labels": {panel: labels_by_panel[panel] for panel in ordered_panels},
            "background_color": list(args.background_rgb),
            "source_fit": args.source_fit,
            "upper_body_only": bool(args.upper_body_only),
            "append_file_names": bool(args.append_file_names),
        },
        "frames": {
            "gt_original": int(original_lengths.get("gt", len(gt_vertices))),
            "pred_original": int(original_lengths.get("flow", len(pred_vertices))),
            "prior_original": int(original_lengths.get("word", len(prior_vertices))),
            "soke_original": int(original_lengths["soke"]) if "soke" in original_lengths else None,
            "source_original": int(source_frame_count_value) if source_frame_count_value is not None else None,
            "output": int(total_frames),
            "resample_to_source_frames": bool(args.resample_to_source_frames),
            "end_mode": args.end_mode,
        },
        "render": {
            "fps": int(args.fps),
            "panel_width": int(args.width),
            "panel_height": int(args.height),
            "software_face_stride": int(args.software_face_stride),
            "view_transform": args.view_transform,
            "device": args.device,
        },
    }
    meta_path = update_render_metadata(args, render_meta)
    if meta_path is not None:
        print(f"Updated metadata: {meta_path}")
    return out_path


def main():
    args = parse_args()
    args.background_rgb = parse_rgb_color(args.background_color)
    args.text_rgb, args.subtext_rgb, args.separator_rgb = text_colors_for_background(args.background_rgb)
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

        out_path = write_video(args, gt_vertices, pred_vertices, prior_vertices, pred_path, prior_path, faces)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
