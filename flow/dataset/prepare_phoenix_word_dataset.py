#!/usr/bin/env python
import argparse
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx")
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word")
DEFAULT_CTC_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Phoenix gloss word-prior dataset by splitting each prepared "
            "sentence motion over its gloss tokens."
        )
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--dataset_name",
        default="phoenix14t_word",
        help="Dataset label written into output manifest rows.",
    )
    parser.add_argument(
        "--source_dataset_name",
        default="",
        help="Fallback source dataset label when source rows do not provide dataset.",
    )
    parser.add_argument(
        "--lexicon_key_mode",
        choices=["ascii", "unicode"],
        default="ascii",
        help="Use ascii for Phoenix-style filenames or unicode for non-ASCII gloss keys such as CSL-Daily.",
    )
    parser.add_argument(
        "--summary_filename",
        default="prepare_phoenix_word_dataset_summary.json",
        help="Summary JSON filename under OUT_DIR/meta.",
    )
    parser.add_argument(
        "--split_method",
        choices=["even", "ctc"],
        default="even",
        help="Use the original equal slicing or precomputed CTC forced-alignment spans.",
    )
    parser.add_argument(
        "--alignment_dir",
        type=Path,
        default=None,
        help="Directory containing {split}.jsonl CTC alignments. Defaults to SOURCE_DIR/meta/ctc_alignments.",
    )
    parser.add_argument(
        "--boundary_policy",
        choices=["core", "span", "midpoint"],
        default="midpoint",
        help="Alignment policy used by the CTC span files; stored in the output summary.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
        help="For --split_method ctc, fall back to even slicing when a token confidence is below this value.",
    )
    parser.add_argument(
        "--min_segment_frames",
        type=int,
        default=None,
        help="Skip gloss segments shorter than this many frames. Defaults to 1 for even, 4 for ctc.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed for each word clip. Smaller but much slower for many tiny files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_segment_frames is None:
        args.min_segment_frames = 4 if args.split_method == "ctc" else 1
    if args.split_method == "ctc":
        if args.alignment_dir is None:
            args.alignment_dir = args.source_dir / "meta" / "ctc_alignments"
        if args.out_dir == DEFAULT_OUT_DIR:
            args.out_dir = DEFAULT_CTC_OUT_DIR
    return args


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_lexicon_key(value, mode="ascii"):
    if mode == "ascii":
        text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
        text = re.sub(r"_+", "_", text)
        return text or "GLOSS"

    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "GLOSS"


def even_segments(length, count):
    if count <= 0:
        return []
    base = int(length) // int(count)
    remainder = int(length) % int(count)
    segments = []
    start = 0
    for index in range(int(count)):
        seg_len = base + (1 if index < remainder else 0)
        end = start + seg_len
        segments.append((start, end))
        start = end
    return segments


def load_alignment_map(alignment_dir, split):
    path = Path(alignment_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CTC alignment file: {path}")
    rows = read_jsonl(path)
    alignments = {}
    duplicates = []
    for row in rows:
        name = row.get("name", "")
        if not name:
            continue
        if name in alignments:
            duplicates.append(name)
        alignments[name] = row
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate alignment names in {path}: {sample}")
    return alignments


def _segment_record(
    start,
    end,
    split_method,
    confidence=None,
    core_frames=None,
    peak_frame=None,
    fallback=False,
    fallback_reason="",
    boundary_policy="",
):
    return {
        "start": int(start),
        "end": int(end),
        "split_method": split_method,
        "alignment_confidence": None if confidence is None else float(confidence),
        "core_frames": None if core_frames is None else int(core_frames),
        "peak_frame": None if peak_frame is None else int(peak_frame),
        "alignment_fallback": bool(fallback),
        "alignment_fallback_reason": str(fallback_reason or ""),
        "alignment_boundary_policy": boundary_policy,
    }


def _even_segment_records(
    length,
    gloss_count,
    split_method="even_by_gloss_count",
    reason="",
    boundary_policy="",
):
    return [
        _segment_record(
            start,
            end,
            split_method,
            confidence=0.0 if split_method.startswith("even_fallback") else None,
            core_frames=end - start,
            fallback=split_method.startswith("even_fallback"),
            fallback_reason=reason,
            boundary_policy=boundary_policy,
        )
        for start, end in even_segments(length, gloss_count)
    ]


def _normalize_ctc_segment(segment, length):
    if not isinstance(segment, (list, tuple)) or len(segment) != 2:
        return None
    start = max(0, min(int(segment[0]), int(length)))
    end = max(start, min(int(segment[1]), int(length)))
    return start, end


def select_segment_records(args, row, gloss_tokens, motion_length, alignment_map):
    gloss_count = len(gloss_tokens)
    even_records = _even_segment_records(motion_length, gloss_count)
    if args.split_method == "even":
        return even_records

    alignment = alignment_map.get(row.get("name", ""))
    if alignment is None:
        return _even_segment_records(
            motion_length,
            gloss_count,
            split_method="even_fallback",
            reason="missing_ctc_alignment",
            boundary_policy=args.boundary_policy,
        )

    segments = alignment.get("segments", [])
    if len(segments) != gloss_count:
        return _even_segment_records(
            motion_length,
            gloss_count,
            split_method="even_fallback",
            reason=f"alignment_segment_count_{len(segments)}_!=_gloss_count_{gloss_count}",
            boundary_policy=args.boundary_policy,
        )

    if alignment.get("split_method") == "even_fallback":
        return _even_segment_records(
            motion_length,
            gloss_count,
            split_method="even_fallback",
            reason=alignment.get("row_fallback_reason", "ctc_row_fallback"),
            boundary_policy=args.boundary_policy,
        )

    confidence = alignment.get("confidence", [])
    core_frames = alignment.get("core_frames", [])
    peak_frames = alignment.get("peak_frames", [])
    segment_fallback = alignment.get("segment_fallback", [])
    fallback_reason = alignment.get("fallback_reason", [])
    boundary_policy = alignment.get("boundary_policy", args.boundary_policy)
    records = []

    for index, segment in enumerate(segments):
        normalized = _normalize_ctc_segment(segment, motion_length)
        conf = confidence[index] if index < len(confidence) else None
        core = core_frames[index] if index < len(core_frames) else None
        peak = peak_frames[index] if index < len(peak_frames) else None
        token_fallback = bool(segment_fallback[index]) if index < len(segment_fallback) else False
        reason = fallback_reason[index] if index < len(fallback_reason) else ""

        fallback_needed = normalized is None
        if not fallback_needed:
            start, end = normalized
            fallback_needed = end - start < args.min_segment_frames
            if fallback_needed and not reason:
                reason = f"ctc_segment_has_{end - start}_frames"

        if (
            not fallback_needed
            and conf is not None
            and args.confidence_threshold > 0
            and float(conf) < args.confidence_threshold
        ):
            fallback_needed = True
            reason = f"alignment_confidence_{float(conf):.4f}_below_{args.confidence_threshold:.4f}"

        if fallback_needed:
            start, end = even_segments(motion_length, gloss_count)[index]
            records.append(
                _segment_record(
                    start,
                    end,
                    "even_fallback",
                    confidence=conf,
                    core_frames=core,
                    peak_frame=peak,
                    fallback=True,
                    fallback_reason=reason or "invalid_ctc_segment",
                    boundary_policy=boundary_policy,
                )
            )
            continue

        records.append(
            _segment_record(
                start,
                end,
                "even_fallback" if token_fallback else "ctc_forced_align",
                confidence=conf,
                core_frames=core,
                peak_frame=peak,
                fallback=token_fallback,
                fallback_reason=reason,
                boundary_policy=boundary_policy,
            )
        )

    return records


def reset_output(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = args.out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing = [
            args.out_dir / split
            for split in args.splits
            if (args.out_dir / split).exists()
        ]
        existing += [
            meta / f"manifest_{split}.jsonl"
            for split in args.splits
            if (meta / f"manifest_{split}.jsonl").exists()
        ]
        if existing:
            raise FileExistsError(
                f"{args.out_dir} already contains prepared files; pass --overwrite to replace them."
            )
    for split in args.splits:
        split_dir = args.out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
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
        raise RuntimeError("No training word frames were processed; cannot compute mean/std.")
    mean = stats["sum"] / stats["count"]
    var = np.maximum(stats["sumsq"] / stats["count"] - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)
    meta = out_dir / "meta"
    np.save(meta / "mean.npy", mean.astype(np.float32))
    np.save(meta / "std.npy", std.astype(np.float32))


def process_split(args, split, stats):
    manifest_path = args.source_dir / "meta" / f"manifest_{split}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    rows = read_jsonl(manifest_path)
    alignment_map = load_alignment_map(args.alignment_dir, split) if args.split_method == "ctc" else {}
    variant_counts = defaultdict(int)
    out_rows = []
    skipped = []

    for row in rows:
        gloss_tokens = str(row.get("gloss", "")).split()
        if not gloss_tokens:
            skipped.append({"name": row.get("name", ""), "reason": "empty gloss"})
            continue

        source_path = args.source_dir / row["motion_path"]
        with np.load(source_path) as data:
            motion = data["motion"].astype(np.float32)
            left_valid = data["left_valid"].astype(np.float32)
            right_valid = data["right_valid"].astype(np.float32)

        if motion.ndim != 2 or motion.shape[1] != 133:
            skipped.append({"name": row.get("name", ""), "reason": f"bad motion shape {motion.shape}"})
            continue

        segment_records = select_segment_records(args, row, gloss_tokens, len(motion), alignment_map)
        for gloss_index, (gloss, segment) in enumerate(zip(gloss_tokens, segment_records)):
            start = segment["start"]
            end = segment["end"]
            if end - start < args.min_segment_frames:
                skipped.append(
                    {
                        "name": row.get("name", ""),
                        "gloss": gloss,
                        "gloss_index": gloss_index,
                        "reason": f"segment has {end - start} frames",
                    }
                )
                continue

            lexicon_key = sanitize_lexicon_key(gloss, args.lexicon_key_mode)
            variant_counts[lexicon_key] += 1
            variant_id = f"{variant_counts[lexicon_key]:04d}"
            name = f"{lexicon_key}-{variant_id}"
            rel_path = Path(split) / f"{name}.npz"
            out_path = args.out_dir / rel_path

            segment_motion = motion[start:end]
            segment_left = left_valid[start:end]
            segment_right = right_valid[start:end]
            save_npz = np.savez_compressed if args.compress else np.savez
            save_npz(
                out_path,
                motion=segment_motion.astype(np.float32, copy=False),
                left_valid=segment_left.astype(np.float32, copy=False),
                right_valid=segment_right.astype(np.float32, copy=False),
            )

            if split == "train":
                add_stats(stats, segment_motion)

            manifest_row = {
                "name": name,
                "motion_path": rel_path.as_posix(),
                "text": gloss,
                "gloss": gloss,
                "lexicon_key": lexicon_key,
                "variant_id": variant_id,
                "word": gloss,
                "fps": float(row.get("fps", 20.0)),
                "num_frames": int(segment_motion.shape[0]),
                "duration": float(segment_motion.shape[0] / float(row.get("fps", 20.0))),
                "dataset": args.dataset_name,
                "source_dataset": row.get("dataset", args.source_dataset_name or args.source_dir.name),
                "source_data_dir": str(args.source_dir),
                "source_name": row.get("name", ""),
                "source_motion_path": row["motion_path"],
                "source_split": row.get("source_split", split),
                "source_text": row.get("text", ""),
                "source_gloss": row.get("gloss", ""),
                "source_gloss_index": int(gloss_index),
                "source_gloss_count": int(len(gloss_tokens)),
                "segment_start_frame": int(start),
                "segment_end_frame": int(end),
                "split_method": segment["split_method"],
                "word_format_version": "word_variant_v1",
                "signer": row.get("signer", ""),
            }
            if args.split_method == "ctc":
                manifest_row.update(
                    {
                        "alignment_confidence": segment["alignment_confidence"],
                        "core_frames": segment["core_frames"],
                        "peak_frame": segment["peak_frame"],
                        "alignment_fallback": segment["alignment_fallback"],
                        "alignment_fallback_reason": segment["alignment_fallback_reason"],
                        "alignment_boundary_policy": segment["alignment_boundary_policy"],
                    }
                )
            out_rows.append(manifest_row)
            if len(out_rows) % 5000 == 0:
                print(f"{split}: wrote {len(out_rows)} word clips...", flush=True)

    write_jsonl(args.out_dir / "meta" / f"manifest_{split}.jsonl", out_rows)
    return out_rows, skipped


def write_summary(args, counts, skipped):
    summary = {
        "dataset_name": args.out_dir.name,
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "word_variant_format": {
            "version": "word_variant_v1",
            "applied": True,
            "filename_pattern": "<lexicon_key>-<variant_id>.npz",
            "fields": ["lexicon_key", "variant_id", "word", "gloss"],
            "lexicon_key_mode": args.lexicon_key_mode,
        },
        "split_method": "ctc_forced_align" if args.split_method == "ctc" else "even_by_gloss_count",
        "split_method_arg": args.split_method,
        "alignment_dir": str(args.alignment_dir) if args.alignment_dir else "",
        "alignment_boundary_policy": args.boundary_policy,
        "alignment_confidence_threshold": float(args.confidence_threshold),
        "min_segment_frames": int(args.min_segment_frames),
        "compressed_npz": bool(args.compress),
        "split_counts": counts,
        "skipped": skipped,
    }
    path = args.out_dir / "meta" / args.summary_filename
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Missing source dataset: {args.source_dir}")
    reset_output(args)

    stats = {
        "count": 0,
        "sum": np.zeros(133, dtype=np.float64),
        "sumsq": np.zeros(133, dtype=np.float64),
    }
    counts = {}
    all_skipped = []
    for split in args.splits:
        rows, skipped = process_split(args, split, stats)
        counts[split] = len(rows)
        all_skipped.extend({"split": split, **item} for item in skipped)
        print(f"{split}: wrote {len(rows)} word clips, skipped {len(skipped)}")

    if "train" in args.splits:
        save_stats(args.out_dir, stats)
    write_summary(args, counts, all_skipped)
    print(f"Saved word dataset under {args.out_dir}")


if __name__ == "__main__":
    main()
