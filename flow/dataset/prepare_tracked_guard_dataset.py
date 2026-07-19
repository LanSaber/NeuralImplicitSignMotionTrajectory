#!/usr/bin/env python
import argparse
import csv
import json
import random
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from flow.smplx_features import COMPACT_DIM, load_pickle_cpu, to_numpy


DEFAULT_TRACKED_DIR = Path("/media/cvpr/haomian/data/tracked_v6arm_guard_sent")
DEFAULT_TEXT_MAP = DEFAULT_TRACKED_DIR / "sentences.csv"
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert tracked_v6arm_guard_sent pickles to the 133D SOKE flow dataset format."
    )
    parser.add_argument("--tracked_dir", type=Path, default=DEFAULT_TRACKED_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking_file", default="optim_tracking_ehm.pkl")
    parser.add_argument(
        "--dir_prefix",
        default="sentence_",
        help="Only scan subdirectories with this prefix. Ignored when --all_dirs is set.",
    )
    parser.add_argument(
        "--all_dirs",
        action="store_true",
        help="Scan every immediate subdirectory under --tracked_dir, useful for word/gloss folders.",
    )
    parser.add_argument(
        "--expression_source",
        choices=["smplx", "flame"],
        default="smplx",
        help="Use smplx_coeffs['exp'] or flame_coeffs['expression_params'] for the 10D expression slice.",
    )
    parser.add_argument("--text_map", type=Path, default=DEFAULT_TEXT_MAP, help="Optional JSON/JSONL/CSV mapping sentence names to text.")
    parser.add_argument("--missing_text", choices=["empty", "name"], default="empty")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--same_splits",
        action="store_true",
        default=True,
        help="Put every valid sequence into train, val, and test instead of making disjoint splits.",
    )
    parser.add_argument(
        "--disjoint_splits",
        dest="same_splits",
        action="store_false",
        help="Use train/val/test ratios to make disjoint splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--word_variant_format",
        action="store_true",
        help=(
            "Write word/gloss dictionaries as <lexicon_key>-<variant_id>.npz and add "
            "lexicon_key, variant_id, word, and gloss manifest fields."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_text_map(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing text map: {path}")

    suffix = path.suffix.lower()
    rows = []
    if suffix == ".jsonl":
        rows = read_jsonl(path)
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {str(key): str(value) for key, value in obj.items()}
        if isinstance(obj, list):
            rows = obj
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported text map suffix {suffix}; use JSON, JSONL, or CSV.")

    text_map = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (
            row.get("name")
            or row.get("filename")
            or row.get("sentence")
            or row.get("sentence_id")
            or row.get("SENTENCE_NAME")
        )
        text = row.get("text") or row.get("sentence_text") or row.get("SENTENCE")
        if name is not None and text is not None:
            name = str(name)
            text_map[name] = str(text)
            text_map[Path(name).stem] = str(text)
    return text_map


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_flat(value):
    return to_numpy(value, dtype=np.float32).reshape(-1)


def as_pose(value, joints, name):
    pose = to_numpy(value, dtype=np.float32).reshape(-1, 3)
    if pose.shape[0] < joints:
        raise ValueError(f"{name} has {pose.shape[0]} joints, expected at least {joints}")
    return pose[:joints]


def scalar_or_default(value, default=1.0):
    if value is None:
        return float(default)
    array = to_numpy(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return float(default)
    return float(array[0])


def frame_compact(frame, expression_source):
    smplx = frame["smplx_coeffs"]
    flame = frame.get("flame_coeffs", {})

    body_pose = as_pose(smplx["body_pose"], 21, "body_pose")
    upper_body = body_pose[11:21].reshape(-1)
    if upper_body.shape[0] != 30:
        raise ValueError(f"upper_body has {upper_body.shape[0]} dims, expected 30")

    left_hand = as_pose(smplx["left_hand_pose"], 15, "left_hand_pose").reshape(-1)
    right_hand = as_pose(smplx["right_hand_pose"], 15, "right_hand_pose").reshape(-1)

    if "jaw_params" in flame:
        jaw = as_flat(flame["jaw_params"])[:3]
    elif "jaw_pose" in smplx:
        jaw = as_flat(smplx["jaw_pose"])[:3]
    else:
        jaw = np.zeros(3, dtype=np.float32)
    if jaw.shape[0] != 3:
        raise ValueError(f"jaw has {jaw.shape[0]} dims, expected 3")

    if expression_source == "flame" and "expression_params" in flame:
        expression = as_flat(flame["expression_params"])[:10]
    else:
        expression = as_flat(smplx.get("exp", flame.get("expression_params", np.zeros(10, dtype=np.float32))))[:10]
    if expression.shape[0] < 10:
        expression = np.pad(expression, (0, 10 - expression.shape[0]))

    compact = np.concatenate([upper_body, left_hand, right_hand, jaw, expression], axis=0)
    if compact.shape[0] != COMPACT_DIM:
        raise ValueError(f"compact feature has {compact.shape[0]} dims, expected {COMPACT_DIM}")
    return compact.astype(np.float32, copy=False)


def frame_hand_valid(frame):
    visibility = frame.get("hand_visibility", {})
    left = scalar_or_default(visibility.get("left_hand_visible"), 1.0)
    right = scalar_or_default(visibility.get("right_hand_visible"), 1.0)
    return left, right


def extract_sequence(path, expression_source):
    data = load_pickle_cpu(path)
    frame_keys = sorted(key for key in data.keys() if isinstance(key, str) and key.startswith("frame_"))
    if not frame_keys:
        raise ValueError(f"{path} does not contain frame_* entries")

    motions = []
    left_valid = []
    right_valid = []
    for key in frame_keys:
        frame = data[key]
        motions.append(frame_compact(frame, expression_source))
        left, right = frame_hand_valid(frame)
        left_valid.append(left)
        right_valid.append(right)

    return (
        np.stack(motions).astype(np.float32, copy=False),
        np.asarray(left_valid, dtype=np.float32),
        np.asarray(right_valid, dtype=np.float32),
    )


def sequence_dirs(root, prefix):
    return sorted(
        path
        for path in Path(root).iterdir()
        if path.is_dir() and (not prefix or path.name.startswith(prefix))
    )


def split_items(items, args):
    items = list(items)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.limit > 0:
        items = items[: args.limit]
    if args.same_splits:
        return {
            "train": list(items),
            "val": list(items),
            "test": list(items),
        }

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive.")
    train_ratio = args.train_ratio / total_ratio
    val_ratio = args.val_ratio / total_ratio

    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def manifest_text(name, text_map, missing_text):
    if name in text_map:
        return text_map[name]
    return name if missing_text == "name" else ""


def sanitize_lexicon_key(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "WORD"


def build_word_variant_map(items):
    by_key = {}
    for item in items:
        source_name = item.name
        key = sanitize_lexicon_key(source_name)
        by_key.setdefault(key, []).append(source_name)

    variant_map = {}
    for key, names in by_key.items():
        for index, source_name in enumerate(names, start=1):
            variant_id = f"{index:04d}"
            variant_map[source_name] = {
                "lexicon_key": key,
                "variant_id": variant_id,
                "variant_name": f"{key}-{variant_id}",
                "word": source_name,
            }
    return variant_map


def save_stats(out_dir, stats):
    if stats["count"] <= 0:
        raise RuntimeError("No training frames were processed; cannot compute mean/std.")
    mean = stats["sum"] / stats["count"]
    var = np.maximum(stats["sumsq"] / stats["count"] - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)
    meta = out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    np.save(meta / "mean.npy", mean.astype(np.float32))
    np.save(meta / "std.npy", std.astype(np.float32))


def write_config(args, counts, skipped):
    config = {
        "dataset_name": args.out_dir.name,
        "out_dir": str(args.out_dir),
        "tracked_dir": str(args.tracked_dir),
        "text_map": str(args.text_map) if args.text_map is not None else None,
        "tracking_file": args.tracking_file,
        "dir_prefix": args.dir_prefix,
        "all_dirs": bool(args.all_dirs),
        "expression_source": args.expression_source,
        "fps": args.fps,
        "seed": args.seed,
        "same_splits": bool(args.same_splits),
        "word_variant_format": (
            {
                "version": "word_variant_v1",
                "applied": True,
                "filename_pattern": "<lexicon_key>-<variant_id>.npz",
                "fields": ["lexicon_key", "variant_id", "word", "gloss"],
            }
            if args.word_variant_format
            else False
        ),
        "split_counts": counts,
        "skipped": skipped,
        "notes": "motion is compact 133D: upper body, hands, jaw, first 10 expression coefficients",
    }
    meta = args.out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "prepare_tracked_guard_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if not args.tracked_dir.is_dir():
        raise FileNotFoundError(f"Missing tracked dataset directory: {args.tracked_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text_map = load_text_map(args.text_map)
    candidates = []
    skipped = []
    dir_prefix = "" if args.all_dirs else args.dir_prefix
    for sent_dir in sequence_dirs(args.tracked_dir, dir_prefix):
        tracking_path = sent_dir / args.tracking_file
        if tracking_path.is_file():
            candidates.append(sent_dir)
        else:
            skipped.append({"name": sent_dir.name, "reason": f"missing {args.tracking_file}"})

    splits = split_items(candidates, args)
    word_variant_map = build_word_variant_map(candidates) if args.word_variant_format else {}
    stats = {
        "count": 0,
        "sum": np.zeros(COMPACT_DIM, dtype=np.float64),
        "sumsq": np.zeros(COMPACT_DIM, dtype=np.float64),
    }
    counts = {}

    for split, items in splits.items():
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for sent_dir in tqdm(items, desc=f"prepare {split}"):
            source_name = sent_dir.name
            word_info = word_variant_map.get(source_name)
            name = word_info["variant_name"] if word_info is not None else source_name
            tracking_path = sent_dir / args.tracking_file
            out_path = split_dir / f"{name}.npz"
            try:
                if out_path.is_file() and not args.overwrite:
                    with np.load(out_path) as data:
                        motion = data["motion"].astype(np.float32)
                        left_valid = data["left_valid"].astype(np.float32)
                        right_valid = data["right_valid"].astype(np.float32)
                else:
                    motion, left_valid, right_valid = extract_sequence(tracking_path, args.expression_source)
                    if len(motion) < args.min_frames:
                        raise ValueError(f"only {len(motion)} frames")
                    np.savez_compressed(
                        out_path,
                        motion=motion.astype(np.float32),
                        left_valid=left_valid.astype(np.float32),
                        right_valid=right_valid.astype(np.float32),
                    )
            except Exception as exc:
                warnings.warn(f"Skipping {name}: {exc}", RuntimeWarning)
                skipped.append({"name": name, "reason": str(exc)})
                continue

            rel_path = out_path.relative_to(args.out_dir).as_posix()
            item = {
                "name": name,
                "motion_path": rel_path,
                "text": manifest_text(source_name, text_map, args.missing_text),
                "fps": float(args.fps),
                "num_frames": int(motion.shape[0]),
                "duration": float(motion.shape[0] / args.fps) if args.fps > 0 else 0.0,
                "source_path": str(tracking_path),
                "expression_source": args.expression_source,
            }
            if word_info is not None:
                item.update(
                    {
                        "lexicon_key": word_info["lexicon_key"],
                        "variant_id": word_info["variant_id"],
                        "word": word_info["word"],
                        "gloss": word_info["word"],
                        "source_name": source_name,
                        "word_format_version": "word_variant_v1",
                    }
                )
            manifest.append(item)

            if split == "train":
                motion64 = motion.astype(np.float64)
                stats["count"] += motion64.shape[0]
                stats["sum"] += motion64.sum(axis=0)
                stats["sumsq"] += np.square(motion64).sum(axis=0)

        write_jsonl(args.out_dir / "meta" / f"manifest_{split}.jsonl", manifest)
        counts[split] = len(manifest)
        print(f"{split}: wrote {len(manifest)} samples")

    save_stats(args.out_dir, stats)
    write_config(args, counts, skipped)
    print(f"Saved mean/std/manifests under {args.out_dir / 'meta'}")
    if skipped:
        print(f"Skipped {len(skipped)} entries; see prepare_tracked_guard_config.json")


if __name__ == "__main__":
    main()
