#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np

from flow.align.dataset import (
    gloss_tokens,
    load_gloss_vocab,
    load_motion_arrays,
    prepare_motion_features,
    read_jsonl,
    tokens_to_ids,
)


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx")
NEG_INF = -1.0e30


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def even_segments(length, count):
    if count <= 0:
        return []
    length = int(length)
    count = int(count)
    base = length // count
    remainder = length % count
    start = 0
    segments = []
    for index in range(count):
        seg_len = base + (1 if index < remainder else 0)
        end = start + seg_len
        segments.append([int(start), int(end)])
        start = end
    return segments


def ctc_viterbi_align(log_probs, labels, blank_id=0):
    log_probs = np.asarray(log_probs, dtype=np.float64)
    labels = [int(label) for label in labels]
    if log_probs.ndim != 2:
        raise ValueError(f"Expected log_probs [T,V], got {log_probs.shape}")
    if not labels:
        raise ValueError("Cannot align an empty label sequence.")

    frames, vocab_size = log_probs.shape
    if frames <= 0:
        raise ValueError("Cannot align an empty frame sequence.")
    max_label = max(labels + [blank_id])
    if max_label >= vocab_size:
        raise ValueError(f"Label id {max_label} is outside vocab size {vocab_size}")

    extended = np.full(2 * len(labels) + 1, int(blank_id), dtype=np.int64)
    extended[1::2] = np.asarray(labels, dtype=np.int64)
    states = len(extended)

    dp = np.full((frames, states), NEG_INF, dtype=np.float64)
    back = np.full((frames, states), -1, dtype=np.int32)
    dp[0, 0] = log_probs[0, extended[0]]
    if states > 1:
        dp[0, 1] = log_probs[0, extended[1]]

    for t in range(1, frames):
        for s in range(states):
            best_score = dp[t - 1, s]
            best_state = s

            if s - 1 >= 0 and dp[t - 1, s - 1] > best_score:
                best_score = dp[t - 1, s - 1]
                best_state = s - 1

            if (
                s - 2 >= 0
                and extended[s] != blank_id
                and extended[s] != extended[s - 2]
                and dp[t - 1, s - 2] > best_score
            ):
                best_score = dp[t - 1, s - 2]
                best_state = s - 2

            if best_score <= NEG_INF / 2:
                continue
            dp[t, s] = best_score + log_probs[t, extended[s]]
            back[t, s] = best_state

    final_candidates = [states - 1]
    if states >= 2:
        final_candidates.append(states - 2)
    final_state = max(final_candidates, key=lambda s: dp[frames - 1, s])
    final_score = float(dp[frames - 1, final_state])
    if final_score <= NEG_INF / 2:
        raise ValueError("No valid CTC path found for this frame/label sequence.")

    state_path = np.zeros(frames, dtype=np.int32)
    state = int(final_state)
    for t in range(frames - 1, -1, -1):
        state_path[t] = state
        if t > 0:
            state = int(back[t, state])
            if state < 0:
                raise ValueError("Invalid CTC backtrace.")

    return {
        "state_path": state_path,
        "extended_labels": extended,
        "score": final_score,
    }


def _expand_to_minimum(start, end, lower, upper, min_frames):
    start = int(start)
    end = int(end)
    lower = int(lower)
    upper = int(upper)
    min_frames = int(min_frames)
    while end - start < min_frames and (start > lower or end < upper):
        if start > lower:
            start -= 1
        if end - start >= min_frames:
            break
        if end < upper:
            end += 1
    return start, end


def _midpoint_boundary(left_peak, right_peak):
    # Frame centers are at t + 0.5; floor keeps the boundary monotonic and integer.
    return int(np.floor((int(left_peak) + int(right_peak) + 1) / 2.0))


def extract_segments(
    state_path,
    labels,
    log_probs,
    blank_id=0,
    min_segment_frames=4,
    boundary_policy="core",
):
    state_path = np.asarray(state_path, dtype=np.int32)
    log_probs = np.asarray(log_probs, dtype=np.float64)
    labels = [int(label) for label in labels]
    frames = int(len(state_path))
    fallback_segments = even_segments(frames, len(labels))

    if boundary_policy not in {"core", "span", "midpoint"}:
        raise ValueError(f"Unknown boundary policy: {boundary_policy}")

    core_bounds = []
    core_positions = []
    peak_frames = []
    for index in range(len(labels)):
        token_state = 2 * index + 1
        positions = np.flatnonzero(state_path == token_state)
        core_positions.append(positions)
        if len(positions) == 0:
            core_bounds.append(None)
            peak_frames.append(None)
        else:
            core_bounds.append((int(positions[0]), int(positions[-1]) + 1))
            label_scores = log_probs[positions, labels[index]]
            peak_frames.append(int(positions[int(np.argmax(label_scores))]))

    midpoint_boundaries = None
    if boundary_policy == "midpoint":
        midpoint_boundaries = [0]
        for index in range(len(labels) - 1):
            if peak_frames[index] is not None and peak_frames[index + 1] is not None:
                boundary = _midpoint_boundary(peak_frames[index], peak_frames[index + 1])
            else:
                boundary = fallback_segments[index][1]
            boundary = max(midpoint_boundaries[-1], min(int(boundary), frames))
            midpoint_boundaries.append(boundary)
        midpoint_boundaries.append(frames)

    segments = []
    confidence = []
    core_frames = []
    segment_fallback = []
    fallback_reason = []

    for index, label in enumerate(labels):
        positions = core_positions[index]
        core_len = int(len(positions))
        core_frames.append(core_len)
        if core_len > 0:
            posterior = np.exp(log_probs[positions, label])
            conf = float(np.clip(np.mean(posterior), 0.0, 1.0))
        else:
            conf = 0.0
        confidence.append(conf)

        if core_bounds[index] is None or peak_frames[index] is None:
            segments.append(fallback_segments[index])
            segment_fallback.append(True)
            fallback_reason.append("missing_core_frames")
            continue

        if boundary_policy == "midpoint":
            start = midpoint_boundaries[index]
            end = midpoint_boundaries[index + 1]
            lower = start
            upper = end
        elif boundary_policy == "span":
            start = core_bounds[index][0]
            if index + 1 < len(labels) and core_bounds[index + 1] is not None:
                end = core_bounds[index + 1][0]
            else:
                end = frames
            lower = core_bounds[index - 1][1] if index > 0 and core_bounds[index - 1] else 0
            upper = core_bounds[index + 1][0] if index + 1 < len(labels) and core_bounds[index + 1] else frames
        else:
            start, end = core_bounds[index]
            lower = core_bounds[index - 1][1] if index > 0 and core_bounds[index - 1] else 0
            upper = core_bounds[index + 1][0] if index + 1 < len(labels) and core_bounds[index + 1] else frames

        start = max(lower, min(start, upper))
        end = max(start, min(end, upper))
        start, end = _expand_to_minimum(start, end, lower, upper, min_segment_frames)

        if end - start < min_segment_frames:
            segments.append(fallback_segments[index])
            segment_fallback.append(True)
            fallback_reason.append(f"segment_has_{end - start}_frames")
        else:
            segments.append([int(start), int(end)])
            segment_fallback.append(False)
            fallback_reason.append("")

    return {
        "segments": segments,
        "confidence": confidence,
        "core_frames": core_frames,
        "peak_frames": peak_frames,
        "segment_fallback": segment_fallback,
        "fallback_reason": fallback_reason,
    }


def make_even_fallback_row(row, num_frames, tokens, reason, split, boundary_policy):
    segments = even_segments(num_frames, len(tokens))
    return {
        "name": row.get("name", ""),
        "motion_path": row.get("motion_path", ""),
        "split": split,
        "num_frames": int(num_frames),
        "glosses": tokens,
        "segments": segments,
        "confidence": [0.0 for _ in tokens],
        "core_frames": [max(0, end - start) for start, end in segments],
        "peak_frames": [None for _ in tokens],
        "segment_fallback": [True for _ in tokens],
        "fallback_reason": [reason for _ in tokens],
        "fallback": True,
        "row_fallback_reason": reason,
        "split_method": "even_fallback",
        "boundary_policy": boundary_policy,
        "alignment_score": None,
    }


def make_alignment_row(
    row,
    split,
    tokens,
    labels,
    log_probs,
    blank_id,
    min_segment_frames,
    boundary_policy,
):
    alignment = ctc_viterbi_align(log_probs, labels, blank_id=blank_id)
    spans = extract_segments(
        alignment["state_path"],
        labels,
        log_probs,
        blank_id=blank_id,
        min_segment_frames=min_segment_frames,
        boundary_policy=boundary_policy,
    )
    return {
        "name": row.get("name", ""),
        "motion_path": row.get("motion_path", ""),
        "split": split,
        "num_frames": int(log_probs.shape[0]),
        "glosses": tokens,
        "label_ids": [int(label) for label in labels],
        "segments": spans["segments"],
        "confidence": spans["confidence"],
        "core_frames": spans["core_frames"],
        "peak_frames": spans["peak_frames"],
        "segment_fallback": spans["segment_fallback"],
        "fallback_reason": spans["fallback_reason"],
        "fallback": bool(any(spans["segment_fallback"])),
        "row_fallback_reason": "",
        "split_method": "ctc_forced_align",
        "boundary_policy": boundary_policy,
        "alignment_score": float(alignment["score"]),
    }


def load_checkpoint_model(checkpoint_path, device):
    import torch

    from flow.align.ctc_recognizer import GlossCTCRecognizer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model")
    if model_config is None:
        raise KeyError("Checkpoint is missing config.model.")
    model = GlossCTCRecognizer(**model_config).to(device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint is missing model_state_dict.")
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def resolve_device(device):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infer_feature_config(args, checkpoint):
    dataset_config = checkpoint.get("config", {}).get("dataset", {})
    features = args.features or dataset_config.get("features", "motion")
    append_valid = bool(args.append_valid or dataset_config.get("append_valid", False))
    gate_hands = bool(args.gate_hands or dataset_config.get("gate_hands", False))
    return features, append_valid, gate_hands


def batch_valid_items(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_split(args, split, model, checkpoint, vocab, device):
    import torch

    manifest_path = args.data_dir / "meta" / f"manifest_{split}.jsonl"
    rows = read_jsonl(manifest_path)
    mean = np.load(args.data_dir / "meta" / "mean.npy").astype(np.float32)
    std = np.load(args.data_dir / "meta" / "std.npy").astype(np.float32)
    features_name, append_valid, gate_hands = infer_feature_config(args, checkpoint)

    output_rows = []
    pending = []
    gloss_to_id = vocab["gloss_to_id"]
    blank_id = int(vocab["blank_id"])

    def flush_pending():
        if not pending:
            return
        max_len = max(item["features"].shape[0] for item in pending)
        dim = pending[0]["features"].shape[-1]
        motion = torch.zeros(len(pending), max_len, dim, dtype=torch.float32, device=device)
        lengths = torch.zeros(len(pending), dtype=torch.long, device=device)
        for index, item in enumerate(pending):
            frames = item["features"].shape[0]
            motion[index, :frames] = torch.from_numpy(item["features"]).to(device)
            lengths[index] = frames
        with torch.no_grad():
            log_probs = model(motion, lengths).detach().cpu().numpy()

        for index, item in enumerate(pending):
            row = item["row"]
            tokens = item["tokens"]
            labels = item["labels"]
            frames = int(lengths[index].item())
            try:
                output_rows.append(
                    make_alignment_row(
                        row,
                        split,
                        tokens,
                        labels,
                        log_probs[index, :frames],
                        blank_id=blank_id,
                        min_segment_frames=args.min_segment_frames,
                        boundary_policy=args.boundary_policy,
                    )
                )
            except ValueError as exc:
                output_rows.append(
                    make_even_fallback_row(
                        row,
                        frames,
                        tokens,
                        reason=f"ctc_alignment_failed:{exc}",
                        split=split,
                        boundary_policy=args.boundary_policy,
                    )
                )
        pending.clear()

    for row_index, row in enumerate(rows, start=1):
        tokens = gloss_tokens(row.get("gloss", ""))
        if not tokens:
            output_rows.append(
                make_even_fallback_row(row, int(row.get("num_frames", 0)), tokens, "empty_gloss", split, args.boundary_policy)
            )
            continue

        labels, oov = tokens_to_ids(tokens, gloss_to_id)
        motion, left_valid, right_valid = load_motion_arrays(args.data_dir, row)
        frames = int(motion.shape[0])
        if oov:
            output_rows.append(
                make_even_fallback_row(
                    row,
                    frames,
                    tokens,
                    reason="oov_gloss:" + ",".join(oov),
                    split=split,
                    boundary_policy=args.boundary_policy,
                )
            )
            continue
        if frames <= 0 or frames < len(labels):
            output_rows.append(
                make_even_fallback_row(
                    row,
                    frames,
                    tokens,
                    reason=f"frames_lt_glosses:{frames}<{len(labels)}",
                    split=split,
                    boundary_policy=args.boundary_policy,
                )
            )
            continue

        sample_features = prepare_motion_features(
            motion,
            left_valid,
            right_valid,
            mean,
            std,
            features=features_name,
            append_valid=append_valid,
            gate_hands=gate_hands,
        )
        pending.append(
            {
                "row": row,
                "tokens": tokens,
                "labels": labels,
                "features": sample_features,
            }
        )
        if len(pending) >= args.batch_size:
            flush_pending()

        if row_index % 500 == 0:
            print(f"{split}: queued/aligned {row_index}/{len(rows)} sentences", flush=True)

    flush_pending()
    out_path = args.out_dir / f"{split}.jsonl"
    write_jsonl(out_path, output_rows)
    print(f"{split}: wrote {len(output_rows)} alignment rows to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump Phoenix CTC forced alignments as per-gloss frame spans."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--features", choices=["motion", "motion_velocity"], default=None)
    parser.add_argument("--append_valid", action="store_true")
    parser.add_argument("--gate_hands", action="store_true")
    parser.add_argument("--boundary_policy", choices=["core", "span", "midpoint"], default="midpoint")
    parser.add_argument("--min_segment_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir = args.out_dir or args.data_dir / "meta" / "ctc_alignments"
    vocab_path = args.vocab_path or args.data_dir / "meta" / "gloss_vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing vocab: {vocab_path}. Run flow.align.build_gloss_vocab first.")

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    vocab = load_gloss_vocab(vocab_path)
    for split in args.splits:
        run_split(args, split, model, checkpoint, vocab, device)


if __name__ == "__main__":
    main()
