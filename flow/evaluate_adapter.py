#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flow.content_style_adapter import build_adapter_from_config
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.latent_codec import LatentMotionCodec
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import (
    compact_from_rotation_representation,
    normalize_rotation_rep,
    rotation_rep_stats_paths,
    smplx182_from_compact,
)
from flow.temporal_word_attention import WordCandidateBuilder, build_arranger_from_config
from flow.text_encoder import FrozenT5TextEncoder
from flow.train_adapter import (
    candidate_group_usage,
    encode_word_candidates,
    encode_word_text_features,
    per_sample_group_entropy_peakiness,
)


DEFAULT_OUT_DIR = Path("visualize/flow/flow_adapter_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a word-prior content-style adapter checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=None)
    parser.add_argument("--word_split", "--word-split", default="")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--text_model_path", "--text-model-path", type=Path, default=None)
    parser.add_argument("--max_text_tokens", "--max-text-tokens", type=int, default=0)
    parser.add_argument(
        "--stats_data_dir",
        "--stats-data-dir",
        type=Path,
        default=None,
        help="Override dataset directory that owns normalization mean/std.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--shuffle_word_candidates", action="store_true")
    parser.add_argument("--candidate_seed", type=int, default=123)
    parser.add_argument("--min_frames", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--length_multiple", type=int, default=0)
    parser.add_argument(
        "--lazy_word_motions",
        action="store_true",
        help="Load word-prior motion arrays on demand instead of preloading the full word dataset.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def path_from_config(value, default=None):
    if value is None or value == "":
        return default
    return Path(value)


def bool_from_config(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_runtime(args, checkpoint):
    data_cfg = checkpoint.get("data_config", {})
    latent_cfg = checkpoint.get("latent_config", {})
    train_args = checkpoint.get("args", {})
    text_cfg = checkpoint.get("text_config", {})
    candidate_cfg = checkpoint.get("candidate_config", {})
    arranger_cfg = checkpoint.get("arranger_config", {})
    loss_cfg = checkpoint.get("loss_config", {})

    data_dir = args.data_dir or path_from_config(data_cfg.get("data_dir"))
    if data_dir is None:
        raise ValueError("Could not resolve data_dir from checkpoint; pass --data_dir.")

    word_data_dir = args.word_data_dir or path_from_config(data_cfg.get("word_data_dir"))
    if word_data_dir is None:
        raise ValueError("Could not resolve word_data_dir from checkpoint; pass --word_data_dir.")

    word_split = args.word_split or str(data_cfg.get("word_split") or train_args.get("word_split") or "train")
    rotation_rep = normalize_rotation_rep(
        data_cfg.get("rotation_rep")
        or latent_cfg.get("rotation_rep")
        or train_args.get("rotation_rep")
        or "axis_angle"
    )

    if args.stats_data_dir is not None:
        mean_path, std_path = rotation_rep_stats_paths(args.stats_data_dir, rotation_rep)
    else:
        mean_path = path_from_config(data_cfg.get("mean_path"))
        std_path = path_from_config(data_cfg.get("std_path"))
        if mean_path is None or std_path is None or not mean_path.is_file() or not std_path.is_file():
            stats_data_dir = path_from_config(data_cfg.get("stats_data_dir"), data_dir)
            mean_path, std_path = rotation_rep_stats_paths(stats_data_dir, rotation_rep)

    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {mean_path} and {std_path}")

    vae_checkpoint = args.vae_checkpoint or path_from_config(latent_cfg.get("vae_checkpoint"))
    if vae_checkpoint is None:
        raise ValueError("Could not resolve VAE checkpoint from adapter checkpoint; pass --vae_checkpoint.")

    max_frames = args.max_frames if args.max_frames > 0 else int(data_cfg.get("max_frames", 400))
    min_frames = args.min_frames if args.min_frames > 0 else int(data_cfg.get("min_frames", 40))
    length_multiple = (
        args.length_multiple if args.length_multiple > 0 else int(data_cfg.get("length_multiple", 4))
    )

    latent_stats = latent_cfg.get("stats")
    if not latent_stats or "mean" not in latent_stats or "std" not in latent_stats:
        raise ValueError("Adapter checkpoint does not contain latent_config.stats.")

    prior_mode = str(checkpoint.get("prior_mode") or train_args.get("prior_mode", "concat"))
    adapter_enabled = bool_from_config(
        checkpoint.get("adapter_enabled"),
        default=not bool_from_config(train_args.get("disable_adapter"), default=False),
    )
    text_model_path = args.text_model_path or path_from_config(text_cfg.get("text_model_path"))
    max_text_tokens = args.max_text_tokens if args.max_text_tokens > 0 else int(text_cfg.get("max_text_tokens", 64))

    return {
        "data_dir": Path(data_dir),
        "word_data_dir": Path(word_data_dir),
        "word_split": word_split,
        "rotation_rep": rotation_rep,
        "mean_path": Path(mean_path),
        "std_path": Path(std_path),
        "vae_checkpoint": Path(vae_checkpoint),
        "max_frames": max_frames,
        "min_frames": min_frames,
        "length_multiple": length_multiple,
        "latent_stats": latent_stats,
        "prior_mode": prior_mode,
        "adapter_enabled": adapter_enabled,
        "text_model_path": Path(text_model_path) if text_model_path is not None else None,
        "max_text_tokens": max_text_tokens,
        "num_word_candidates": int(candidate_cfg.get("num_word_candidates", train_args.get("num_word_candidates", 32))),
        "num_negative_candidates": int(candidate_cfg.get("num_negative_candidates", train_args.get("num_negative_candidates", 16))),
        "candidate_selection": str(candidate_cfg.get("candidate_selection", train_args.get("candidate_selection", "flat"))),
        "max_positive_variants_per_key": int(
            candidate_cfg.get(
                "max_positive_variants_per_key",
                train_args.get("max_positive_variants_per_key", 0),
            )
        ),
        "max_word_latent_frames": int(
            arranger_cfg.get("max_word_latent_frames", train_args.get("max_word_latent_frames", 64))
        ),
        "group_coverage_mass": float(
            loss_cfg.get("group_coverage_mass", train_args.get("group_coverage_mass", 0.5))
        ),
        "group_entropy_peak_target": float(
            loss_cfg.get("group_entropy_peak_target", train_args.get("group_entropy_peak_target", 0.6931471805599453))
        ),
        "lazy_word_motions": bool(args.lazy_word_motions),
    }


def denormalize(norm_motion, dataset):
    return norm_motion * dataset.std + dataset.mean


def masked_l1_torch(a, b, mask):
    weights = mask.to(device=a.device, dtype=a.dtype).unsqueeze(-1).expand_as(a)
    return ((a - b).abs() * weights).sum() / weights.sum().clamp_min(1.0)


def diff_array(x, order):
    out = np.asarray(x, dtype=np.float32)
    for _ in range(order):
        out = np.diff(out, axis=0)
    return out


def l1_array(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def motion_metrics(prefix, motion, gt_motion):
    metrics = {f"{prefix}_pose_l1": l1_array(motion, gt_motion)}
    for order, name in [(1, "vel"), (2, "accel"), (3, "jerk")]:
        if len(motion) > order and len(gt_motion) > order:
            metrics[f"{prefix}_{name}_l1"] = l1_array(diff_array(motion, order), diff_array(gt_motion, order))
        else:
            metrics[f"{prefix}_{name}_l1"] = float("nan")
    return metrics


def per_sample_attention_diagnostics(attention, target_mask):
    if attention.shape[1] <= 1:
        batch = attention.shape[0]
        zero = torch.zeros(batch, dtype=attention.dtype, device=attention.device)
        return zero, zero
    flat = attention.reshape(attention.shape[0], attention.shape[1], -1)
    valid = (target_mask[:, 1:] & target_mask[:, :-1]).to(device=attention.device, dtype=attention.dtype)
    frame_l1 = (flat[:, 1:] - flat[:, :-1]).abs().sum(dim=-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    temporal_l1 = (frame_l1 * valid).sum(dim=1) / denom
    frame_cos = torch.nn.functional.cosine_similarity(flat[:, 1:], flat[:, :-1], dim=-1, eps=1e-8)
    frame_cos = (frame_cos * valid).sum(dim=1) / denom
    has_pair = valid.sum(dim=1) > 0
    return temporal_l1 * has_pair.to(dtype=attention.dtype), frame_cos * has_pair.to(dtype=attention.dtype)


def per_sample_delta_rms(x, mask):
    if x.shape[1] <= 1:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    diff = x[:, 1:] - x[:, :-1]
    valid = (mask[:, 1:] & mask[:, :-1]).to(device=x.device, dtype=x.dtype).unsqueeze(-1).expand_as(diff)
    denom = valid.sum(dim=(1, 2)).clamp_min(1.0)
    return ((diff.pow(2) * valid).sum(dim=(1, 2)) / denom).sqrt()


def per_sample_temporal_std(x, mask, eps=1e-6):
    weights = mask.to(device=x.device, dtype=x.dtype)
    counts = weights.sum(dim=1)
    denom = counts.clamp_min(1.0).view(-1, 1)
    mean = (x * weights.unsqueeze(-1)).sum(dim=1) / denom
    var = ((x - mean[:, None, :]).pow(2) * weights.unsqueeze(-1)).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(float(eps))).mean(dim=-1)
    return std * (counts > 1).to(dtype=x.dtype)


def summarize_metrics(samples):
    numeric_keys = sorted(
        {
            key
            for item in samples
            for key, value in item.items()
            if isinstance(value, (int, float)) and key not in {"index", "length"}
        }
    )
    summary = {}
    for key in numeric_keys:
        values = [float(item[key]) for item in samples if key in item and np.isfinite(float(item[key]))]
        if values:
            summary[key] = float(np.mean(values))

    for metric in ["latent_l1", "pose_l1", "vel_l1", "accel_l1", "jerk_l1"]:
        for base_name, compare_name in [("word", "adapted"), ("prior", "adapted"), ("word", "prior")]:
            base_key = f"{base_name}_{metric}"
            compare_key = f"{compare_name}_{metric}"
            if base_key not in numeric_keys or compare_key not in numeric_keys:
                continue
            comparable = [
                item
                for item in samples
                if base_key in item
                and compare_key in item
                and np.isfinite(float(item[base_key]))
                and np.isfinite(float(item[compare_key]))
            ]
            if comparable:
                summary[f"{metric}_{compare_name}_better_than_{base_name}_rate"] = float(
                    np.mean([float(item[compare_key]) < float(item[base_key]) for item in comparable])
                )
    return summary


def save_motion_npz(path, motion, representation, rotation_rep, item, length, source_index, **extra):
    payload = {
        "motion": motion.astype(np.float32),
        "representation": representation.astype(np.float32),
        "rotation_rep": rotation_rep,
        "smplx": smplx182_from_compact(motion),
        "name": str(item["name"]),
        "text": str(item.get("text", "")),
        "length": int(length),
        "source_index": int(source_index),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)


@torch.no_grad()
def evaluate_batch(
    adapter,
    latent_codec,
    batch,
    prior_builder,
    dataset,
    runtime,
    device,
    arranger=None,
    candidate_builder=None,
    text_encoder=None,
    shuffle_candidates=False,
):
    raw_x = batch["motion"].to(device)
    raw_mask = batch["mask"].to(device)
    lengths = batch["length"].detach().cpu().tolist()

    latent_stats = {
        "mean": torch.as_tensor(runtime["latent_stats"]["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(runtime["latent_stats"]["std"], dtype=torch.float32, device=device),
    }
    z_sent_raw, latent_mask = latent_codec.encode(raw_x, mask=raw_mask)
    z_sent = latent_codec.normalize_latent(z_sent_raw, latent_stats)

    prior_raw, prior_stats = prior_builder.batch(
        batch["text"],
        lengths,
        max_len=raw_x.shape[1],
        device=device,
        dtype=raw_x.dtype,
    )
    prior_raw = prior_raw * raw_mask.to(device=device, dtype=prior_raw.dtype).unsqueeze(-1)
    z_word_raw, _ = latent_codec.encode(prior_raw, mask=raw_mask)
    z_word = latent_codec.normalize_latent(z_word_raw, latent_stats)
    z_prior = z_word
    prior_aligned_raw = prior_raw
    arranger_out = None
    candidate_batch = None
    if runtime["prior_mode"] == "soft_arranger":
        if arranger is None or candidate_builder is None or text_encoder is None:
            raise RuntimeError("Soft-arranger evaluation requires arranger, candidate_builder, and text_encoder.")
        max_word_latent_frames = int(runtime["max_word_latent_frames"])
        max_motion_frames = min(
            int(latent_codec.max_frames),
            max_word_latent_frames * int(latent_codec.downsample_factor),
        )
        candidate_batch = candidate_builder.batch(
            batch["text"],
            device=device,
            dtype=raw_x.dtype,
            shuffle=shuffle_candidates,
            max_motion_frames=max_motion_frames,
        )
        word_latents, word_latent_mask = encode_word_candidates(
            candidate_batch,
            latent_codec,
            latent_stats,
            device,
            max_word_latent_frames=max_word_latent_frames,
        )
        sentence_text = text_encoder.encode(batch["text"]).to(device=device, dtype=raw_x.dtype)
        word_text = encode_word_text_features(text_encoder, candidate_batch.texts, device, raw_x.dtype)
        arranger_out = arranger(
            sentence_text,
            word_text,
            word_latents,
            word_latent_mask,
            candidate_batch.candidate_mask,
            latent_mask,
        )
        z_prior = arranger_out["z_prior_aligned"]
        prior_aligned_raw = latent_codec.decode(
            latent_codec.denormalize_latent(z_prior, latent_stats),
            target_length=raw_x.shape[1],
            mask=raw_mask,
            latent_mask=latent_mask,
        )

    if runtime["adapter_enabled"]:
        if adapter is None:
            raise RuntimeError("Adapter is enabled in the checkpoint but no adapter module was loaded.")
        z_adapt = adapter(z_prior, mask=latent_mask)["z_adapt"]
    else:
        z_adapt = z_prior * latent_mask.to(device=device, dtype=z_prior.dtype).unsqueeze(-1)
    z_adapt_raw = latent_codec.denormalize_latent(z_adapt, latent_stats)
    x_adapt = latent_codec.decode(
        z_adapt_raw,
        target_length=raw_x.shape[1],
        mask=raw_mask,
        latent_mask=latent_mask,
    )

    latent_word_l1 = masked_l1_torch(z_word, z_sent, latent_mask).detach().cpu().item()
    latent_prior_l1 = masked_l1_torch(z_prior, z_sent, latent_mask).detach().cpu().item()
    latent_adapt_l1 = masked_l1_torch(z_adapt, z_sent, latent_mask).detach().cpu().item()

    out = {
        "raw_x": raw_x.detach().cpu().numpy(),
        "prior_raw": prior_raw.detach().cpu().numpy(),
        "prior_aligned_raw": prior_aligned_raw.detach().cpu().numpy(),
        "x_adapt": x_adapt.detach().cpu().numpy(),
        "z_sent": z_sent.detach().cpu().numpy(),
        "z_word": z_word.detach().cpu().numpy(),
        "z_prior": z_prior.detach().cpu().numpy(),
        "z_adapt": z_adapt.detach().cpu().numpy(),
        "latent_mask": latent_mask.detach().cpu().numpy(),
        "latent_word_l1_batch": latent_word_l1,
        "latent_prior_l1_batch": latent_prior_l1,
        "latent_adapt_l1_batch": latent_adapt_l1,
        "prior_stats": prior_stats,
    }
    if arranger_out is not None:
        out.update(
            {
                "attention": arranger_out["attention"].detach().cpu().numpy(),
                "word_gate_probs": arranger_out["word_gate_probs"].detach().cpu().numpy(),
                "word_usage": arranger_out["word_usage"].detach().cpu().numpy(),
                "null_usage": arranger_out["null_usage"].detach().cpu().numpy(),
                "candidate_mask": candidate_batch.candidate_mask.detach().cpu().numpy(),
                "candidate_labels": candidate_batch.labels.detach().cpu().numpy(),
                "candidate_group_ids": candidate_batch.group_ids.detach().cpu().numpy(),
                "group_mask": candidate_batch.group_mask.detach().cpu().numpy(),
                "candidate_names": candidate_batch.names,
                "group_texts": candidate_batch.group_texts,
                "candidate_stats": candidate_batch.stats,
            }
        )
        group_usage = candidate_group_usage(
            arranger_out["attention"],
            candidate_batch.group_ids,
            candidate_batch.group_mask,
            latent_mask,
        )
        group_mask = candidate_batch.group_mask.to(device=device, dtype=torch.bool)
        valid_groups = group_mask.to(dtype=raw_x.dtype)
        if group_usage.shape[1] > 0:
            group_counts = valid_groups.sum(dim=1, keepdim=True).clamp_min(1.0)
            target_per_group = float(runtime["group_coverage_mass"]) / group_counts
            group_coverage = ((group_usage >= target_per_group) & group_mask).to(dtype=raw_x.dtype).sum(dim=1)
            group_coverage = group_coverage / valid_groups.sum(dim=1).clamp_min(1.0)
            group_usage_total = group_usage.sum(dim=1)
            has_group = group_mask.any(dim=1)
            group_usage_max = group_usage.masked_fill(~group_mask, -1.0).max(dim=1).values
            group_usage_max = group_usage_max * has_group.to(dtype=raw_x.dtype)
        else:
            group_coverage = raw_x.new_zeros(raw_x.shape[0])
            group_usage_total = raw_x.new_zeros(raw_x.shape[0])
            group_usage_max = raw_x.new_zeros(raw_x.shape[0])
        group_entropy, group_peak_prob = per_sample_group_entropy_peakiness(
            arranger_out["attention"],
            candidate_batch.group_ids,
            candidate_batch.group_mask,
            latent_mask,
        )
        neg_mask = candidate_batch.candidate_mask & (candidate_batch.labels <= 0.5)
        pos_mask = candidate_batch.candidate_mask & (candidate_batch.labels > 0.5)
        gate_probs = arranger_out["word_gate_probs"]
        pos_count = pos_mask.to(dtype=raw_x.dtype).sum(dim=1)
        neg_count_raw = neg_mask.to(dtype=raw_x.dtype).sum(dim=1)
        gate_pos = (gate_probs * pos_mask.to(dtype=raw_x.dtype)).sum(dim=1) / pos_count.clamp_min(1.0)
        gate_neg = (gate_probs * neg_mask.to(dtype=raw_x.dtype)).sum(dim=1) / neg_count_raw.clamp_min(1.0)
        nan_value = raw_x.new_full((raw_x.shape[0],), float("nan"))
        gate_pos = torch.where(pos_count > 0, gate_pos, nan_value)
        gate_neg = torch.where(neg_count_raw > 0, gate_neg, nan_value)
        gate_gap = gate_pos - gate_neg
        neg_count = neg_mask.to(dtype=raw_x.dtype).sum(dim=1).clamp_min(1.0)
        negative_usage = (arranger_out["word_usage"] * neg_mask.to(dtype=raw_x.dtype)).sum(dim=1) / neg_count
        attention_temporal_l1, attention_frame_cosine = per_sample_attention_diagnostics(
            arranger_out["attention"],
            latent_mask,
        )
        out.update(
            {
                "group_usage": group_usage.detach().cpu().numpy(),
                "group_coverage": group_coverage.detach().cpu().numpy(),
                "group_usage_total": group_usage_total.detach().cpu().numpy(),
                "group_usage_max": group_usage_max.detach().cpu().numpy(),
                "group_entropy": group_entropy.detach().cpu().numpy(),
                "group_peak_prob": group_peak_prob.detach().cpu().numpy(),
                "gate_pos": gate_pos.detach().cpu().numpy(),
                "gate_neg": gate_neg.detach().cpu().numpy(),
                "gate_gap": gate_gap.detach().cpu().numpy(),
                "negative_usage": negative_usage.detach().cpu().numpy(),
                "null_usage_pct": (arranger_out["null_usage"] * 100.0).detach().cpu().numpy(),
                "attention_temporal_l1": attention_temporal_l1.detach().cpu().numpy(),
                "attention_frame_cosine": attention_frame_cosine.detach().cpu().numpy(),
                "prior_latent_std": per_sample_temporal_std(z_prior, latent_mask).detach().cpu().numpy(),
                "gt_latent_std": per_sample_temporal_std(z_sent, latent_mask).detach().cpu().numpy(),
                "prior_delta_rms": per_sample_delta_rms(z_prior, latent_mask).detach().cpu().numpy(),
                "gt_delta_rms": per_sample_delta_rms(z_sent, latent_mask).detach().cpu().numpy(),
            }
        )
    return out


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    runtime = resolve_runtime(args, checkpoint)

    dataset = UpperSMPLXFlowDataset(
        runtime["data_dir"],
        split=args.split,
        mean_path=runtime["mean_path"],
        std_path=runtime["std_path"],
        min_frames=runtime["min_frames"],
        max_frames=runtime["max_frames"],
        length_multiple=runtime["length_multiple"],
        random_crop=False,
        rotation_rep=runtime["rotation_rep"],
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {runtime['data_dir']} split={args.split}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index {args.index} is outside split length {len(dataset)}")

    end = min(args.index + max(args.num_samples, 1), len(dataset))
    items = [dataset[idx] for idx in range(args.index, end)]
    batch = collate_upper_smplx(items)

    latent_codec = LatentMotionCodec(runtime["vae_checkpoint"], device=device)
    if normalize_rotation_rep(latent_codec.rotation_rep) != runtime["rotation_rep"]:
        raise ValueError(
            f"VAE rotation_rep={latent_codec.rotation_rep} does not match adapter checkpoint "
            f"rotation_rep={runtime['rotation_rep']}"
        )
    adapter = None
    if runtime["adapter_enabled"]:
        adapter = build_adapter_from_config(checkpoint["model_config"]).to(device).eval()
        adapter.load_state_dict(checkpoint["model"], strict=True)
    arranger = None
    text_encoder = None

    prior_builder = WordMotionPrior(
        runtime["word_data_dir"],
        split=runtime["word_split"],
        target_mean=dataset.mean,
        target_std=dataset.std,
        rotation_rep=runtime["rotation_rep"],
        lazy_motions=runtime["lazy_word_motions"],
    )
    candidate_builder = None
    if runtime["prior_mode"] == "soft_arranger":
        if "arranger_config" not in checkpoint or "arranger_model" not in checkpoint:
            raise RuntimeError("Soft-arranger checkpoint is missing arranger_config or arranger_model.")
        if runtime["text_model_path"] is None:
            raise RuntimeError("Soft-arranger checkpoint has no text model path; pass --text_model_path.")
        arranger = build_arranger_from_config(checkpoint["arranger_config"]).to(device).eval()
        arranger.load_state_dict(checkpoint["arranger_model"], strict=True)
        candidate_builder = WordCandidateBuilder(
            prior_builder,
            num_word_candidates=runtime["num_word_candidates"],
            num_negative_candidates=runtime["num_negative_candidates"],
            candidate_selection=runtime["candidate_selection"],
            max_positive_variants_per_key=runtime["max_positive_variants_per_key"],
            seed=args.candidate_seed,
        )
        text_encoder = FrozenT5TextEncoder(
            runtime["text_model_path"],
            device=device,
            max_length=runtime["max_text_tokens"],
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"VAE: {runtime['vae_checkpoint']}")
    print(f"Data: {runtime['data_dir']} split={args.split} samples={len(dataset)}")
    print(
        f"Word prior: {runtime['word_data_dir']} split={runtime['word_split']} "
        f"entries={len(prior_builder.entries)} lazy={runtime['lazy_word_motions']}"
    )
    print(f"Prior mode: {runtime['prior_mode']}")
    print(f"Adapter enabled: {runtime['adapter_enabled']}")
    if runtime["prior_mode"] == "soft_arranger":
        print(
            f"Text encoder: {runtime['text_model_path']} max_tokens={runtime['max_text_tokens']}"
        )
        print(
            f"Candidates: K={runtime['num_word_candidates']} negatives={runtime['num_negative_candidates']} "
            f"selection={runtime['candidate_selection']} "
            f"max_pos_variants_per_key={runtime['max_positive_variants_per_key']} "
            f"max_word_latent_frames={runtime['max_word_latent_frames']} "
            f"shuffle={args.shuffle_word_candidates}"
        )
    print(f"Rotation representation: {runtime['rotation_rep']}")
    print(f"Output: {args.out_dir}")

    result = evaluate_batch(
        adapter,
        latent_codec,
        batch,
        prior_builder,
        dataset,
        runtime,
        device,
        arranger=arranger,
        candidate_builder=candidate_builder,
        text_encoder=text_encoder,
        shuffle_candidates=args.shuffle_word_candidates,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for local_idx, item in enumerate(items):
        length = int(item["length"])
        source_index = args.index + local_idx

        gt_rep = denormalize(result["raw_x"][local_idx, :length], dataset)
        word_rep = denormalize(result["prior_raw"][local_idx, :length], dataset)
        prior_rep = denormalize(result["prior_aligned_raw"][local_idx, :length], dataset)
        adapted_rep = denormalize(result["x_adapt"][local_idx, :length], dataset)

        gt_motion = compact_from_rotation_representation(gt_rep, runtime["rotation_rep"])
        word_motion = compact_from_rotation_representation(word_rep, runtime["rotation_rep"])
        prior_motion = compact_from_rotation_representation(prior_rep, runtime["rotation_rep"])
        adapted_motion = compact_from_rotation_representation(adapted_rep, runtime["rotation_rep"])

        gt_path = args.out_dir / f"gt_{local_idx:02d}.npz"
        word_stem = "raw_concat_prior" if runtime["prior_mode"] == "soft_arranger" else "word_prior"
        word_path = args.out_dir / f"{word_stem}_{local_idx:02d}.npz"
        prior_path = args.out_dir / f"soft_arranger_prior_{local_idx:02d}.npz"
        adapted_path = args.out_dir / f"adapted_{local_idx:02d}.npz"
        attention_path = args.out_dir / f"attention_{local_idx:02d}.npz"

        save_motion_npz(
            gt_path,
            gt_motion,
            gt_rep,
            runtime["rotation_rep"],
            item,
            length,
            source_index,
        )
        save_motion_npz(
            word_path,
            word_motion,
            word_rep,
            runtime["rotation_rep"],
            item,
            length,
            source_index,
            prior_stats=json.dumps(result["prior_stats"][local_idx]),
        )
        if runtime["prior_mode"] == "soft_arranger":
            save_motion_npz(
                prior_path,
                prior_motion,
                prior_rep,
                runtime["rotation_rep"],
                item,
                length,
                source_index,
                candidate_stats=json.dumps(result["candidate_stats"][local_idx]),
            )
        save_motion_npz(
            adapted_path,
            adapted_motion,
            adapted_rep,
            runtime["rotation_rep"],
            item,
            length,
            source_index,
            coarse_motion=prior_motion.astype(np.float32),
            coarse_smplx=smplx182_from_compact(prior_motion),
            z_sent=result["z_sent"][local_idx].astype(np.float32),
            z_word=result["z_word"][local_idx].astype(np.float32),
            z_prior=result["z_prior"][local_idx].astype(np.float32),
            z_adapt=result["z_adapt"][local_idx].astype(np.float32),
            latent_mask=result["latent_mask"][local_idx].astype(np.bool_),
            prior_stats=json.dumps(result["prior_stats"][local_idx]),
            checkpoint=str(args.checkpoint),
            checkpoint_epoch=int(checkpoint.get("epoch", -1)),
            checkpoint_global_step=int(checkpoint.get("global_step", -1)),
        )
        if runtime["prior_mode"] == "soft_arranger":
            np.savez_compressed(
                attention_path,
                attention=result["attention"][local_idx].astype(np.float32),
                word_gate_probs=result["word_gate_probs"][local_idx].astype(np.float32),
                word_usage=result["word_usage"][local_idx].astype(np.float32),
                null_usage=np.asarray(result["null_usage"][local_idx], dtype=np.float32),
                candidate_mask=result["candidate_mask"][local_idx].astype(np.bool_),
                candidate_labels=result["candidate_labels"][local_idx].astype(np.float32),
                candidate_group_ids=result["candidate_group_ids"][local_idx].astype(np.int64),
                group_mask=result["group_mask"][local_idx].astype(np.bool_),
                group_usage=result["group_usage"][local_idx].astype(np.float32),
                group_coverage=np.asarray(result["group_coverage"][local_idx], dtype=np.float32),
                group_usage_total=np.asarray(result["group_usage_total"][local_idx], dtype=np.float32),
                group_usage_max=np.asarray(result["group_usage_max"][local_idx], dtype=np.float32),
                group_entropy=np.asarray(result["group_entropy"][local_idx], dtype=np.float32),
                group_peak_prob=np.asarray(result["group_peak_prob"][local_idx], dtype=np.float32),
                gate_pos=np.asarray(result["gate_pos"][local_idx], dtype=np.float32),
                gate_neg=np.asarray(result["gate_neg"][local_idx], dtype=np.float32),
                gate_gap=np.asarray(result["gate_gap"][local_idx], dtype=np.float32),
                negative_usage=np.asarray(result["negative_usage"][local_idx], dtype=np.float32),
                null_usage_pct=np.asarray(result["null_usage_pct"][local_idx], dtype=np.float32),
                attention_temporal_l1=np.asarray(result["attention_temporal_l1"][local_idx], dtype=np.float32),
                attention_frame_cosine=np.asarray(result["attention_frame_cosine"][local_idx], dtype=np.float32),
                prior_latent_std=np.asarray(result["prior_latent_std"][local_idx], dtype=np.float32),
                gt_latent_std=np.asarray(result["gt_latent_std"][local_idx], dtype=np.float32),
                prior_delta_rms=np.asarray(result["prior_delta_rms"][local_idx], dtype=np.float32),
                gt_delta_rms=np.asarray(result["gt_delta_rms"][local_idx], dtype=np.float32),
                candidate_names=np.asarray(result["candidate_names"][local_idx], dtype=str),
                group_texts=np.asarray(result["group_texts"][local_idx], dtype=str),
                candidate_stats=json.dumps(result["candidate_stats"][local_idx]),
                name=str(item["name"]),
                text=str(item.get("text", "")),
                source_index=source_index,
            )

        latent_valid = result["latent_mask"][local_idx].astype(bool)
        z_sent = result["z_sent"][local_idx, latent_valid]
        z_word = result["z_word"][local_idx, latent_valid]
        z_prior = result["z_prior"][local_idx, latent_valid]
        z_adapt = result["z_adapt"][local_idx, latent_valid]
        sample_metrics = {
            "index": source_index,
            "name": str(item["name"]),
            "text": str(item.get("text", "")),
            "length": length,
            "gt_path": str(gt_path),
            "word_prior_path": str(word_path),
            "prior_path": str(prior_path) if runtime["prior_mode"] == "soft_arranger" else str(word_path),
            "adapted_path": str(adapted_path),
            "attention_path": str(attention_path) if runtime["prior_mode"] == "soft_arranger" else "",
            "word_latent_l1": l1_array(z_word, z_sent),
            "prior_latent_l1": l1_array(z_prior, z_sent),
            "adapted_latent_l1": l1_array(z_adapt, z_sent),
            "word_prior_coverage": float(result["prior_stats"][local_idx]["coverage"]),
            "word_prior_matches": int(result["prior_stats"][local_idx]["matched_count"]),
        }
        if runtime["prior_mode"] == "soft_arranger":
            sample_metrics.update(
                {
                    "attention_temporal_l1": float(result["attention_temporal_l1"][local_idx]),
                    "attention_frame_cosine": float(result["attention_frame_cosine"][local_idx]),
                    "group_coverage": float(result["group_coverage"][local_idx]),
                    "group_usage_total": float(result["group_usage_total"][local_idx]),
                    "group_usage_max": float(result["group_usage_max"][local_idx]),
                    "group_entropy": float(result["group_entropy"][local_idx]),
                    "group_peak_prob": float(result["group_peak_prob"][local_idx]),
                    "gate_pos": float(result["gate_pos"][local_idx]),
                    "gate_neg": float(result["gate_neg"][local_idx]),
                    "gate_gap": float(result["gate_gap"][local_idx]),
                    "negative_usage": float(result["negative_usage"][local_idx]),
                    "null_usage_pct": float(result["null_usage_pct"][local_idx]),
                    "prior_latent_std": float(result["prior_latent_std"][local_idx]),
                    "gt_latent_std": float(result["gt_latent_std"][local_idx]),
                    "prior_delta_rms": float(result["prior_delta_rms"][local_idx]),
                    "gt_delta_rms": float(result["gt_delta_rms"][local_idx]),
                }
            )
        sample_metrics.update(motion_metrics("word", word_motion, gt_motion))
        sample_metrics.update(motion_metrics("prior", prior_motion, gt_motion))
        sample_metrics.update(motion_metrics("adapted", adapted_motion, gt_motion))
        samples.append(sample_metrics)

        print(
            f"[{local_idx:02d}] index={source_index} length={length} "
            f"latent word={sample_metrics['word_latent_l1']:.5f} "
            f"prior={sample_metrics['prior_latent_l1']:.5f} "
            f"adapted={sample_metrics['adapted_latent_l1']:.5f} "
            f"pose word={sample_metrics['word_pose_l1']:.5f} "
            f"prior={sample_metrics['prior_pose_l1']:.5f} "
            f"adapted={sample_metrics['adapted_pose_l1']:.5f}"
        )
        print(f"  saved: {gt_path}")
        print(f"  saved: {word_path}")
        if runtime["prior_mode"] == "soft_arranger":
            print(f"  saved: {prior_path}")
            print(f"  saved: {attention_path}")
        print(f"  saved: {adapted_path}")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "index": args.index,
        "num_samples": len(samples),
        "rotation_rep": runtime["rotation_rep"],
        "prior_mode": runtime["prior_mode"],
        "adapter_enabled": runtime["adapter_enabled"],
        "data_dir": str(runtime["data_dir"]),
        "word_data_dir": str(runtime["word_data_dir"]),
        "vae_checkpoint": str(runtime["vae_checkpoint"]),
        "samples": samples,
        "mean": summarize_metrics(samples),
    }
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
