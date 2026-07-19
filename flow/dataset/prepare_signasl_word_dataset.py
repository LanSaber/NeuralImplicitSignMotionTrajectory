#!/usr/bin/env python
import argparse
import json
import re
import shutil
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from flow.smplx_features import COMPACT_DIM, load_pickle_cpu, resample_by_fps, to_numpy


DEFAULT_SENTENCE_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx")
DEFAULT_PKL_DIR = Path("/media/cvpr/haomian/data/SignASL/smpl_pkl")
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_word_signasl")

SOURCE_NAME_RE = re.compile(r"^(.+)_([0-9]+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a shared word-prior dataset from the SignASL SMPL-X "
            "dictionary pickles."
        )
    )
    parser.add_argument("--sentence_dir", type=Path, default=DEFAULT_SENTENCE_DIR)
    parser.add_argument("--pkl_dir", type=Path, default=DEFAULT_PKL_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset_name", default="signasl_word")
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS metadata written to the manifest, and resample target when --source_fps is set.",
    )
    parser.add_argument(
        "--source_fps",
        type=float,
        default=0.0,
        help="Optional source FPS. Values <= 0 keep dictionary clip frame counts unchanged.",
    )
    parser.add_argument("--min_frames", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all pickles.")
    parser.add_argument("--compress", action="store_true", help="Use np.savez_compressed for clips.")
    parser.add_argument(
        "--write_split_aliases",
        dest="write_split_aliases",
        action="store_true",
        default=True,
        help="Also write train/val/test manifests pointing to the shared all/ clips.",
    )
    parser.add_argument(
        "--no_split_aliases",
        dest="write_split_aliases",
        action="store_false",
        help="Only write manifest_all.jsonl.",
    )
    parser.add_argument(
        "--summary_filename",
        default="prepare_signasl_word_dataset_summary.json",
        help="Summary JSON filename under OUT_DIR/meta.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_source_name(path):
    match = SOURCE_NAME_RE.match(Path(path).stem)
    if not match:
        raise ValueError(f"Expected <label>_<variant>.pkl, got {Path(path).name}")
    return match.group(1), match.group(2)


def sanitize_lexicon_key(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "WORD"


def display_text(value):
    text = re.sub(r"[-_]+", " ", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def source_sort_key(path):
    try:
        label, variant = parse_source_name(path)
    except ValueError:
        return ("ZZZ", Path(path).stem, 0, Path(path).name)
    return (sanitize_lexicon_key(label), label, int(variant), Path(path).name)


def require_pose_array(data, key, width):
    if key not in data:
        raise KeyError(f"missing {key}")
    array = to_numpy(data[key], dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{key} has shape {array.shape}, expected [T, {width}]")
    return array


def compact_from_signasl_pickle(data):
    body = require_pose_array(data, "smplx_body_pose", 63)
    left_hand = require_pose_array(data, "smplx_lhand_pose", 45)
    right_hand = require_pose_array(data, "smplx_rhand_pose", 45)
    jaw = require_pose_array(data, "smplx_jaw_pose", 3)
    expression = require_pose_array(data, "smplx_expr", 10)

    lengths = {
        "smplx_body_pose": len(body),
        "smplx_lhand_pose": len(left_hand),
        "smplx_rhand_pose": len(right_hand),
        "smplx_jaw_pose": len(jaw),
        "smplx_expr": len(expression),
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"SMPL-X component lengths differ: {lengths}")
    if not unique_lengths or next(iter(unique_lengths)) <= 0:
        raise ValueError("empty SMPL-X clip")

    compact = np.concatenate(
        [
            body[:, 33:63],
            left_hand,
            right_hand,
            jaw,
            expression,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    if compact.shape[1] != COMPACT_DIM:
        raise ValueError(f"compact motion has shape {compact.shape}, expected [T, {COMPACT_DIM}]")
    if not np.isfinite(compact).all():
        raise ValueError("compact motion contains NaN or Inf")

    valid = np.ones(len(compact), dtype=np.float32)
    return compact, valid.copy(), valid.copy()


def load_motion(path, args):
    data = load_pickle_cpu(path)
    motion, left_valid, right_valid = compact_from_signasl_pickle(data)
    source_pose_frames = int(motion.shape[0])
    source_video_frames = data.get("num_frames", None)
    if source_video_frames is not None:
        source_video_frames = int(source_video_frames)

    source_fps = float(args.source_fps) if args.source_fps and args.source_fps > 0 else None
    if source_fps is not None:
        motion, left_valid, right_valid = resample_by_fps(
            motion,
            left_valid,
            right_valid,
            source_fps=source_fps,
            target_fps=args.fps,
        )
    return motion, left_valid, right_valid, source_pose_frames, source_video_frames


def reset_output(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = args.out_dir / "meta"
    all_dir = args.out_dir / "all"
    existing = []
    if all_dir.exists():
        existing.append(all_dir)
    for split in ["all", "train", "val", "test"]:
        manifest = meta / f"manifest_{split}.jsonl"
        if manifest.exists():
            existing.append(manifest)
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{args.out_dir} already contains prepared files; pass --overwrite to replace them."
        )

    if args.overwrite and all_dir.exists():
        shutil.rmtree(all_dir)
    all_dir.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for split in ["all", "train", "val", "test"]:
            manifest = meta / f"manifest_{split}.jsonl"
            if manifest.exists():
                manifest.unlink()


def add_stats(stats, motion):
    motion64 = motion.astype(np.float64)
    stats["count"] += motion64.shape[0]
    stats["sum"] += motion64.sum(axis=0)
    stats["sumsq"] += np.square(motion64).sum(axis=0)


def save_stats(out_dir, stats):
    if stats["count"] <= 0:
        raise RuntimeError("No word frames were processed; cannot compute mean/std.")
    mean = stats["sum"] / stats["count"]
    var = np.maximum(stats["sumsq"] / stats["count"] - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)
    meta = out_dir / "meta"
    np.save(meta / "mean.npy", mean.astype(np.float32))
    np.save(meta / "std.npy", std.astype(np.float32))


def process_dictionary(args):
    pkl_paths = sorted(args.pkl_dir.glob("*.pkl"), key=source_sort_key)
    if args.limit > 0:
        pkl_paths = pkl_paths[: args.limit]

    stats = {
        "count": 0,
        "sum": np.zeros(COMPACT_DIM, dtype=np.float64),
        "sumsq": np.zeros(COMPACT_DIM, dtype=np.float64),
    }
    variant_counts = defaultdict(int)
    lexicon_counts = Counter()
    rows = []
    skipped = []
    save_npz = np.savez_compressed if args.compress else np.savez
    source_fps = float(args.source_fps) if args.source_fps and args.source_fps > 0 else None

    iterator = tqdm(pkl_paths, desc="prepare SignASL word prior")
    for pkl_path in iterator:
        try:
            source_label, source_variant_id = parse_source_name(pkl_path)
            lexicon_key = sanitize_lexicon_key(source_label)
        except Exception as exc:
            skipped.append({"source_path": str(pkl_path), "reason": str(exc)})
            continue

        try:
            motion, left_valid, right_valid, source_pose_frames, source_video_frames = load_motion(
                pkl_path, args
            )
        except Exception as exc:
            skipped.append({"source_path": str(pkl_path), "reason": str(exc)})
            continue

        if len(motion) < args.min_frames:
            skipped.append(
                {
                    "source_path": str(pkl_path),
                    "source_label": source_label,
                    "reason": f"only {len(motion)} frames",
                }
            )
            continue

        variant_counts[lexicon_key] += 1
        lexicon_counts[lexicon_key] += 1
        variant_id = f"{variant_counts[lexicon_key]:04d}"
        name = f"{lexicon_key}-{variant_id}"
        rel_path = Path("all") / f"{name}.npz"
        out_path = args.out_dir / rel_path

        save_npz(
            out_path,
            motion=motion.astype(np.float32, copy=False),
            left_valid=left_valid.astype(np.float32, copy=False),
            right_valid=right_valid.astype(np.float32, copy=False),
        )
        add_stats(stats, motion)

        label_text = display_text(source_label)
        row = {
            "name": name,
            "motion_path": rel_path.as_posix(),
            "text": label_text,
            "gloss": lexicon_key,
            "lexicon_key": lexicon_key,
            "variant_id": variant_id,
            "word": label_text,
            "fps": float(args.fps),
            "num_frames": int(motion.shape[0]),
            "duration": float(motion.shape[0] / float(args.fps)),
            "dataset": args.dataset_name,
            "source_dataset": "signasl_smpl_pkl",
            "source_data_dir": str(args.pkl_dir),
            "source_name": pkl_path.stem,
            "source_motion_path": pkl_path.name,
            "source_label": source_label,
            "source_variant_id": str(source_variant_id),
            "source_pose_frames": int(source_pose_frames),
            "source_video_frames": source_video_frames,
            "source_fps": source_fps,
            "source_sentence_data_dir": str(args.sentence_dir),
            "split_method": "dictionary_clip",
            "word_format_version": "word_variant_v1",
            "signer": "",
        }
        rows.append(row)

    return rows, stats, lexicon_counts, skipped


def write_manifests(args, rows):
    meta = args.out_dir / "meta"
    write_jsonl(meta / "manifest_all.jsonl", rows)
    aliases = ["all"]
    if args.write_split_aliases:
        for split in ["train", "val", "test"]:
            write_jsonl(meta / f"manifest_{split}.jsonl", rows)
            aliases.append(split)
    return aliases


def write_summary(args, rows, stats, lexicon_counts, skipped, manifest_aliases):
    summary = {
        "dataset_name": args.dataset_name,
        "source_dictionary_dir": str(args.pkl_dir),
        "source_sentence_data_dir": str(args.sentence_dir),
        "out_dir": str(args.out_dir),
        "word_variant_format": {
            "version": "word_variant_v1",
            "applied": True,
            "filename_pattern": "<lexicon_key>-<variant_id>.npz",
            "fields": ["lexicon_key", "variant_id", "word", "gloss"],
            "lexicon_key_mode": "ascii",
        },
        "pruning": {
            "applied": False,
            "reason": "SignASL dictionary clips were already pruned before this conversion.",
        },
        "clip_count": int(len(rows)),
        "lexicon_count": int(len(lexicon_counts)),
        "frame_count": int(stats["count"]),
        "fps": float(args.fps),
        "source_fps": float(args.source_fps) if args.source_fps and args.source_fps > 0 else None,
        "resampled": bool(args.source_fps and args.source_fps > 0),
        "min_frames": int(args.min_frames),
        "compressed_npz": bool(args.compress),
        "manifest_aliases": manifest_aliases,
        "top_lexicon_counts": [
            {"lexicon_key": key, "count": int(count)}
            for key, count in lexicon_counts.most_common(20)
        ],
        "skipped_count": int(len(skipped)),
        "skipped": skipped[:200],
    }
    path = args.out_dir / "meta" / args.summary_filename
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"numpy\.core\.numeric.*")
    args = parse_args()
    if not args.sentence_dir.is_dir():
        raise FileNotFoundError(f"Missing sentence dataset: {args.sentence_dir}")
    if not args.pkl_dir.is_dir():
        raise FileNotFoundError(f"Missing SignASL pickle directory: {args.pkl_dir}")

    reset_output(args)
    rows, stats, lexicon_counts, skipped = process_dictionary(args)
    save_stats(args.out_dir, stats)
    aliases = write_manifests(args, rows)
    write_summary(args, rows, stats, lexicon_counts, skipped, aliases)

    print(f"Saved SignASL word prior dataset under {args.out_dir}")
    print(f"clips={len(rows)} lexicon_keys={len(lexicon_counts)} skipped={len(skipped)}")
    print(f"manifests={', '.join(aliases)}")


if __name__ == "__main__":
    main()
