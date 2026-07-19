#!/usr/bin/env python
"""Pose-only word retrieval from a sentence motion sequence.

This evaluator intentionally does not use sentence text for retrieval. Text is
used only after retrieval to score the predicted word sequence.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.content_style_adapter import build_adapter_from_config
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.latent_codec import LatentMotionCodec
from flow.residual_prior import text_tokens
from flow.smplx_features import normalize_rotation_rep, rotation_rep_stats_paths


DEFAULT_ADAPTER_CHECKPOINT = Path(
    "experiments/flow/adapter/chatsign175_adapter_jointvae_b16_online/checkpoints/best.pt"
)
DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175")
DEFAULT_WORD_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word")
DEFAULT_OUT_DIR = Path("visualize/pose_word_retrieval")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve ordered word-pose dictionary clips from sentence pose only, "
            "using latent-space segment matching and dynamic programming."
        )
    )
    parser.add_argument("--adapter_checkpoint", "--adapter-checkpoint", type=Path, default=DEFAULT_ADAPTER_CHECKPOINT)
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=DEFAULT_WORD_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--word_split", "--word-split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--limit_words", "--limit-words", type=int, default=0)
    parser.add_argument("--feature_space", "--feature-space", default="adapter_content", choices=["vae_latent", "adapter_content"])
    parser.add_argument("--out_dir", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--min_frames", "--min-frames", type=int, default=0)
    parser.add_argument("--max_frames", "--max-frames", type=int, default=0)
    parser.add_argument("--length_multiple", "--length-multiple", type=int, default=0)
    parser.add_argument("--word_min_frames", "--word-min-frames", type=int, default=0)
    parser.add_argument("--word_max_frames", "--word-max-frames", type=int, default=0)
    parser.add_argument("--word_length_multiple", "--word-length-multiple", type=int, default=0)

    parser.add_argument("--min_segment_latents", "--min-segment-latents", type=int, default=4)
    parser.add_argument("--max_segment_latents", "--max-segment-latents", type=int, default=24)
    parser.add_argument("--duration_weight", "--duration-weight", type=float, default=0.1)
    parser.add_argument("--step_penalty", "--step-penalty", type=float, default=0.03)
    parser.add_argument(
        "--length_prior",
        "--length-prior",
        default="word",
        choices=["none", "word"],
        help="When 'word', segment durations are softly compared to dictionary word latent lengths.",
    )
    return parser.parse_args()


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def path_from_config(value, default=None):
    if value is None or value == "":
        return default
    return Path(value)


def resolve_runtime(args, checkpoint):
    data_cfg = checkpoint.get("data_config", {})
    latent_cfg = checkpoint.get("latent_config", {})
    train_args = checkpoint.get("args", {})

    rotation_rep = normalize_rotation_rep(
        data_cfg.get("rotation_rep")
        or latent_cfg.get("rotation_rep")
        or train_args.get("rotation_rep")
        or "axis_angle"
    )
    mean_path = path_from_config(data_cfg.get("mean_path"))
    std_path = path_from_config(data_cfg.get("std_path"))
    if mean_path is None or std_path is None or not mean_path.is_file() or not std_path.is_file():
        stats_data_dir = path_from_config(data_cfg.get("stats_data_dir"), args.data_dir)
        mean_path, std_path = rotation_rep_stats_paths(stats_data_dir, rotation_rep)
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {mean_path} and {std_path}")

    vae_checkpoint = args.vae_checkpoint or path_from_config(latent_cfg.get("vae_checkpoint"))
    if vae_checkpoint is None:
        raise ValueError("Could not resolve VAE checkpoint from adapter checkpoint; pass --vae_checkpoint.")

    latent_stats = latent_cfg.get("stats")
    if not latent_stats or "mean" not in latent_stats or "std" not in latent_stats:
        raise ValueError("Adapter checkpoint does not contain latent_config.stats.")

    max_frames = args.max_frames if args.max_frames > 0 else int(data_cfg.get("max_frames", 400))
    min_frames = args.min_frames if args.min_frames > 0 else int(data_cfg.get("min_frames", 40))
    length_multiple = args.length_multiple if args.length_multiple > 0 else int(data_cfg.get("length_multiple", 4))
    word_min_frames = args.word_min_frames if args.word_min_frames > 0 else min_frames
    word_max_frames = args.word_max_frames if args.word_max_frames > 0 else max_frames
    word_length_multiple = args.word_length_multiple if args.word_length_multiple > 0 else length_multiple

    return {
        "rotation_rep": rotation_rep,
        "mean_path": Path(mean_path),
        "std_path": Path(std_path),
        "vae_checkpoint": Path(vae_checkpoint),
        "latent_stats": latent_stats,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "length_multiple": length_multiple,
        "word_min_frames": word_min_frames,
        "word_max_frames": word_max_frames,
        "word_length_multiple": word_length_multiple,
    }


def latent_stats_tensors(stats, device, dtype=torch.float32):
    return {
        "mean": torch.as_tensor(stats["mean"], dtype=dtype, device=device),
        "std": torch.as_tensor(stats["std"], dtype=dtype, device=device),
    }


@torch.no_grad()
def encode_feature_batch(batch, codec, adapter, latent_stats, feature_space, device):
    motion = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    z_raw, latent_mask = codec.encode(motion, mask=mask)
    z = codec.normalize_latent(z_raw, latent_stats)
    if feature_space == "vae_latent":
        features = z
    elif feature_space == "adapter_content":
        if adapter is None:
            raise ValueError("feature_space=adapter_content requires --adapter_checkpoint.")
        features = adapter(z, mask=latent_mask)["content_tokens"]
    else:
        raise ValueError(f"Unsupported feature_space: {feature_space}")
    return features.detach().cpu(), latent_mask.detach().cpu(), z.detach().cpu()


def load_word_features(args, runtime, codec, adapter, latent_stats, device):
    dataset = UpperSMPLXFlowDataset(
        args.word_data_dir,
        split=args.word_split,
        mean_path=runtime["mean_path"],
        std_path=runtime["std_path"],
        min_frames=runtime["word_min_frames"],
        max_frames=runtime["word_max_frames"],
        length_multiple=runtime["word_length_multiple"],
        random_crop=False,
        limit=args.limit_words,
        rotation_rep=runtime["rotation_rep"],
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_upper_smplx)
    words = []
    for batch in tqdm(loader, desc="encode words"):
        features, latent_mask, _z = encode_feature_batch(batch, codec, adapter, latent_stats, args.feature_space, device)
        for idx, name in enumerate(batch["name"]):
            valid = latent_mask[idx].bool()
            item_feature = features[idx, valid].float().numpy()
            if len(item_feature) == 0:
                continue
            words.append(
                {
                    "index": len(words),
                    "name": str(name),
                    "tokens": text_tokens(str(name)),
                    "text": str(batch["text"][idx]),
                    "feature": item_feature.astype(np.float32, copy=False),
                    "latent_length": int(valid.sum().item()),
                    "frame_length": int(batch["length"][idx].item()),
                }
            )
    if not words:
        raise RuntimeError(f"No word features loaded from {args.word_data_dir}")
    return words


def resample_feature(feature, target_len):
    feature = np.asarray(feature, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if len(feature) == target_len:
        return feature.copy()
    if len(feature) == 1:
        return np.repeat(feature, target_len, axis=0)
    src = np.linspace(0.0, 1.0, num=len(feature), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, feature.shape[-1]), dtype=np.float32)
    for dim in range(feature.shape[-1]):
        out[:, dim] = np.interp(dst, src, feature[:, dim])
    return out


def build_resampled_word_cache(words, min_len, max_len, device):
    cache = {}
    word_lengths = torch.tensor([word["latent_length"] for word in words], dtype=torch.float32, device=device)
    for length in range(int(min_len), int(max_len) + 1):
        tensors = [resample_feature(word["feature"], length) for word in words]
        cache[length] = torch.from_numpy(np.stack(tensors, axis=0)).to(device=device)
    return cache, word_lengths


def best_word_for_segment(segment, seg_len, word_cache, word_lengths, args, device):
    segment_tensor = torch.from_numpy(segment.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)
    word_tensor = word_cache[int(seg_len)]
    costs = (word_tensor - segment_tensor).abs().mean(dim=(1, 2))
    if args.length_prior == "word" and args.duration_weight > 0:
        seg_length = torch.full_like(word_lengths, float(seg_len))
        denom = torch.maximum(word_lengths, seg_length).clamp_min(1.0)
        costs = costs + float(args.duration_weight) * (word_lengths - seg_length).abs() / denom
    best_cost, best_idx = torch.min(costs, dim=0)
    return int(best_idx.item()), float(best_cost.item())


def segment_lengths_for_position(pos, total_len, min_seg, max_seg):
    remaining = total_len - pos
    lengths = [length for length in range(min_seg, max_seg + 1) if length <= remaining]
    if remaining > 0 and remaining < min_seg:
        lengths.append(remaining)
    return lengths


def retrieve_path(sentence_feature, words, word_cache, word_lengths, args, device):
    total_len = int(len(sentence_feature))
    min_seg = max(1, int(args.min_segment_latents))
    max_seg = max(min_seg, int(args.max_segment_latents))
    max_seg = min(max_seg, total_len)

    dp = np.full(total_len + 1, np.inf, dtype=np.float64)
    back = [None for _ in range(total_len + 1)]
    dp[0] = 0.0
    for start in range(total_len):
        if not np.isfinite(dp[start]):
            continue
        for seg_len in segment_lengths_for_position(start, total_len, min_seg, max_seg):
            end = start + seg_len
            segment = sentence_feature[start:end]
            word_idx, word_cost = best_word_for_segment(segment, seg_len, word_cache, word_lengths, args, device)
            score = float(dp[start]) + word_cost + float(args.step_penalty)
            if score < dp[end]:
                dp[end] = score
                back[end] = {
                    "start": start,
                    "end": end,
                    "word_index": word_idx,
                    "segment_cost": word_cost,
                    "step_score": score,
                }

    if not np.isfinite(dp[total_len]):
        raise RuntimeError("Dynamic programming failed to cover the sentence latent sequence.")

    path = []
    pos = total_len
    while pos > 0:
        item = back[pos]
        if item is None:
            raise RuntimeError(f"Broken DP backpointer at latent position {pos}.")
        word = words[item["word_index"]]
        path.append(
            {
                "start_latent": int(item["start"]),
                "end_latent": int(item["end"]),
                "word_index": int(item["word_index"]),
                "word": word["name"],
                "word_tokens": word["tokens"],
                "word_latent_length": int(word["latent_length"]),
                "word_frame_length": int(word["frame_length"]),
                "segment_cost": float(item["segment_cost"]),
            }
        )
        pos = int(item["start"])
    path.reverse()
    return path, float(dp[total_len])


def multiset_f1(pred_tokens, label_tokens):
    pred_counter = Counter(pred_tokens)
    label_counter = Counter(label_tokens)
    overlap = sum((pred_counter & label_counter).values())
    precision = overlap / max(sum(pred_counter.values()), 1)
    recall = overlap / max(sum(label_counter.values()), 1)
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1, overlap


def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for idx_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[idx_b] = prev[idx_b - 1] + 1
            else:
                cur[idx_b] = max(cur[idx_b - 1], prev[idx_b])
        prev = cur
    return prev[-1]


def score_retrieval(path, label_text):
    pred_tokens = []
    for item in path:
        pred_tokens.extend(item["word_tokens"])
    label_tokens = text_tokens(label_text)
    precision, recall, f1, overlap = multiset_f1(pred_tokens, label_tokens)
    lcs = lcs_length(pred_tokens, label_tokens)
    return {
        "label_tokens": label_tokens,
        "pred_tokens": pred_tokens,
        "token_overlap": int(overlap),
        "token_precision": float(precision),
        "token_recall": float(recall),
        "token_f1": float(f1),
        "ordered_lcs": int(lcs),
        "ordered_lcs_recall": float(lcs / max(len(label_tokens), 1)),
    }


def summarize(samples):
    keys = [
        "path_score",
        "path_length",
        "token_precision",
        "token_recall",
        "token_f1",
        "ordered_lcs_recall",
        "avg_segment_cost",
    ]
    out = {}
    for key in keys:
        values = [float(item[key]) for item in samples if key in item and np.isfinite(float(item[key]))]
        if values:
            out[key] = float(np.mean(values))
    return out


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.adapter_checkpoint, map_location="cpu")
    runtime = resolve_runtime(args, checkpoint)
    latent_stats = latent_stats_tensors(runtime["latent_stats"], device)

    codec = LatentMotionCodec(runtime["vae_checkpoint"], device=device)
    if normalize_rotation_rep(codec.rotation_rep) != runtime["rotation_rep"]:
        raise ValueError(f"VAE rotation_rep={codec.rotation_rep} does not match {runtime['rotation_rep']}")

    adapter = None
    if args.feature_space == "adapter_content":
        adapter = build_adapter_from_config(checkpoint["model_config"]).to(device).eval()
        adapter.load_state_dict(checkpoint["model"], strict=True)

    sentence_dataset = UpperSMPLXFlowDataset(
        args.data_dir,
        split=args.split,
        mean_path=runtime["mean_path"],
        std_path=runtime["std_path"],
        min_frames=runtime["min_frames"],
        max_frames=runtime["max_frames"],
        length_multiple=runtime["length_multiple"],
        random_crop=False,
        rotation_rep=runtime["rotation_rep"],
    )
    if args.index < 0 or args.index >= len(sentence_dataset):
        raise IndexError(f"--index {args.index} is outside split length {len(sentence_dataset)}")
    end = min(args.index + max(args.num_samples, 1), len(sentence_dataset))
    sentence_items = [sentence_dataset[idx] for idx in range(args.index, end)]

    print(f"Adapter checkpoint: {args.adapter_checkpoint}")
    print(f"VAE checkpoint: {runtime['vae_checkpoint']}")
    print(f"Feature space: {args.feature_space}")
    print(f"Sentence data: {args.data_dir} split={args.split} samples={len(sentence_dataset)}")
    print(f"Word data: {args.word_data_dir} split={args.word_split}")
    print("Text is used only for evaluation labels, not for retrieval.")

    words = load_word_features(args, runtime, codec, adapter, latent_stats, device)
    min_seg = max(1, int(args.min_segment_latents))
    max_seg = max(min_seg, int(args.max_segment_latents))
    # Cache from 1 because the final DP tail can be shorter than min_seg.
    word_cache, word_lengths = build_resampled_word_cache(words, 1, max_seg, device)
    print(f"Loaded word clips: {len(words)}")
    print(f"Segment latent lengths: min={min_seg} max={max_seg}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for local_idx, item in enumerate(tqdm(sentence_items, desc="retrieve sentences")):
        batch = collate_upper_smplx([item])
        features, latent_mask, _z = encode_feature_batch(batch, codec, adapter, latent_stats, args.feature_space, device)
        valid = latent_mask[0].bool()
        sentence_feature = features[0, valid].float().numpy()
        path, path_score = retrieve_path(sentence_feature, words, word_cache, word_lengths, args, device)
        for node in path:
            node["start_frame_approx"] = int(node["start_latent"] * codec.downsample_factor)
            node["end_frame_approx"] = int(min(node["end_latent"] * codec.downsample_factor, item["length"]))
        eval_scores = score_retrieval(path, item["text"])
        avg_segment_cost = float(np.mean([node["segment_cost"] for node in path])) if path else float("nan")
        sample = {
            "index": int(args.index + local_idx),
            "name": str(item["name"]),
            "text": str(item["text"]),
            "frame_length": int(item["length"]),
            "latent_length": int(valid.sum().item()),
            "path_score": float(path_score),
            "path_length": int(len(path)),
            "avg_segment_cost": avg_segment_cost,
            "retrieved_words": [node["word"] for node in path],
            "path": path,
        }
        sample.update(eval_scores)
        samples.append(sample)
        print(
            f"[{sample['index']:03d}] {sample['name']} "
            f"path_len={sample['path_length']} f1={sample['token_f1']:.3f} "
            f"lcs_recall={sample['ordered_lcs_recall']:.3f} "
            f"words={sample['retrieved_words'][:8]}"
        )

    metrics = {
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "vae_checkpoint": str(runtime["vae_checkpoint"]),
        "data_dir": str(args.data_dir),
        "word_data_dir": str(args.word_data_dir),
        "split": args.split,
        "word_split": args.word_split,
        "feature_space": args.feature_space,
        "text_used_for_retrieval": False,
        "num_samples": len(samples),
        "num_word_clips": len(words),
        "config": {
            "min_segment_latents": args.min_segment_latents,
            "max_segment_latents": args.max_segment_latents,
            "duration_weight": args.duration_weight,
            "step_penalty": args.step_penalty,
            "length_prior": args.length_prior,
        },
        "mean": summarize(samples),
        "samples": samples,
    }
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
