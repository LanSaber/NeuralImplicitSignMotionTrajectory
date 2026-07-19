#!/usr/bin/env python
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from flow.checkpointing import load_top_k_checkpoints, maybe_save_top_k_checkpoint
from flow.content_style_adapter import ContentStyleAdapter
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.distributed import (
    add_distributed_args,
    barrier,
    cleanup_distributed,
    distributed_mean_scalars,
    rank_zero_print,
    resolve_device as resolve_distributed_device,
    setup_distributed,
    unwrap_model,
    wrap_model,
)
from flow.latent_codec import LatentMotionCodec, compute_latent_stats, serializable_latent_stats
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import (
    feature_weight_vector,
    normalize_rotation_rep,
    rotation_rep_slices,
    rotation_rep_stats_paths,
)
from flow.temporal_word_attention import SoftWordArranger, WordCandidateBuilder
from flow.text_encoder import FrozenT5TextEncoder


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175")
DEFAULT_WORD_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word")
DEFAULT_VAE_CHECKPOINT = Path("experiments/flow/VAE/chatsign175_sentence_word_joint_rot6d_vae_b32/checkpoints/best.pt")
DEFAULT_OUT_DIR = Path("experiments/flow/adapter/chatsign175_adapter_jointvae_v1")
DEFAULT_TEXT_MODEL_PATH = Path("deps/flan-t5-base")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a paired word-concat to sentence latent adapter.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=DEFAULT_WORD_DATA_DIR)
    parser.add_argument("--word_split", "--word-split", default="train")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--prior_mode", "--prior-mode", default="concat", choices=["concat", "soft_arranger"])
    parser.add_argument(
        "--disable_softarranger",
        "--disable-softarranger",
        action="store_true",
        help="Ablation alias for --prior_mode concat. Uses deterministic first-variant word concatenation.",
    )
    parser.add_argument(
        "--disable_adapter",
        "--disable-adapter",
        action="store_true",
        help="Bypass the content-style adapter and decode/evaluate the source prior directly.",
    )
    parser.add_argument("--enable_adapter", "--enable-adapter", dest="disable_adapter", action="store_false")
    parser.set_defaults(disable_adapter=False)
    parser.add_argument("--text_model_path", "--text-model-path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument("--max_text_tokens", "--max-text-tokens", type=int, default=64)
    parser.add_argument(
        "--condition_field",
        "--condition-field",
        default="text",
        choices=["text", "gloss", "text_gloss", "label_word"],
        help="Manifest field used for word matching and sentence T5 features.",
    )
    parser.add_argument(
        "--stats_data_dir",
        "--stats-data-dir",
        type=Path,
        default=None,
        help="Dataset directory that owns VAE-normalization mean/std. Defaults to the VAE checkpoint data_config.data_dir.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--length_multiple", type=int, default=4)
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--no_random_crop", dest="random_crop", action="store_false")
    parser.set_defaults(random_crop=False)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--content_dim", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--arranger_hidden_dim", type=int, default=512)
    parser.add_argument("--arranger_num_heads", type=int, default=8)
    parser.add_argument("--arranger_dropout", type=float, default=0.0)
    parser.add_argument("--max_word_latent_frames", type=int, default=64)
    parser.add_argument(
        "--disable_arranger_candidate_gates",
        "--disable-arranger-candidate-gates",
        action="store_true",
        help="Inner SoftWordArranger ablation: remove candidate gate attention bias.",
    )
    parser.add_argument(
        "--disable_arranger_null_memory",
        "--disable-arranger-null-memory",
        action="store_true",
        help="Inner SoftWordArranger ablation: remove the learned NULL memory token.",
    )
    parser.add_argument(
        "--disable_arranger_word_text_features",
        "--disable-arranger-word-text-features",
        action="store_true",
        help="Inner SoftWordArranger ablation: zero candidate word-text features e_k.",
    )
    parser.add_argument(
        "--disable_arranger_word_motion_latents",
        "--disable-arranger-word-motion-latents",
        action="store_true",
        help="Inner SoftWordArranger ablation: zero candidate motion-latent token features u_k.",
    )
    parser.add_argument("--num_word_candidates", type=int, default=32)
    parser.add_argument("--num_negative_candidates", type=int, default=16)
    parser.add_argument(
        "--candidate_selection",
        "--candidate-selection",
        default="flat",
        choices=["flat", "round_robin"],
        help="How to select positive word variants before adding negatives.",
    )
    parser.add_argument(
        "--max_positive_variants_per_key",
        "--max-positive-variants-per-key",
        type=int,
        default=0,
        help="Cap positive motion variants per matched lexicon key/span. 0 disables the cap.",
    )
    parser.add_argument("--shuffle_word_candidates", action="store_true")
    parser.add_argument("--no_shuffle_word_candidates", dest="shuffle_word_candidates", action="store_false")
    parser.set_defaults(shuffle_word_candidates=True)
    parser.add_argument(
        "--lazy_word_motions",
        "--lazy-word-motions",
        action="store_true",
        help="Defer loading word-clip motions until matched (only parse the manifest at "
        "startup). Recommended for large word dictionaries or slow/compressed storage "
        "(e.g. a squashfs overlay), where eagerly np.load-ing every clip is the startup "
        "bottleneck. Requires num_frames in the word manifest.",
    )

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stats_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hand_weight", type=float, default=5.0)
    parser.add_argument("--jaw_weight", type=float, default=2.0)
    parser.add_argument("--expression_weight", type=float, default=2.0)
    parser.add_argument("--hand_valid_floor", type=float, default=0.2)
    parser.add_argument("--latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--pose_loss_weight", type=float, default=0.5)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.5)
    parser.add_argument("--accel_loss_weight", type=float, default=0.25)
    parser.add_argument("--jerk_loss_weight", type=float, default=0.1)
    parser.add_argument("--style_loss_weight", type=float, default=0.1)
    parser.add_argument("--content_pair_loss_weight", type=float, default=0.1)
    parser.add_argument("--delta_loss_weight", type=float, default=0.001)
    parser.add_argument("--orth_loss_weight", type=float, default=0.0)
    parser.add_argument("--content_domain_confusion_loss_weight", type=float, default=0.0)
    parser.add_argument("--gradient_reversal_lambda", type=float, default=1.0)
    parser.add_argument("--arranger_prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--gate_bce_loss_weight", type=float, default=0.1)
    parser.add_argument("--gate_sparsity_loss_weight", type=float, default=0.01)
    parser.add_argument("--attention_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--null_usage_loss_weight", type=float, default=0.01)
    parser.add_argument("--group_coverage_loss_weight", type=float, default=0.02)
    parser.add_argument("--group_coverage_mass", type=float, default=0.5)
    parser.add_argument("--group_entropy_peak_loss_weight", "--group-entropy-peak-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--group_entropy_peak_target",
        "--group-entropy-peak-target",
        type=float,
        default=0.6931471805599453,
        help="Entropy hinge target for per-frame matched occurrence-group attention. log(2) allows roughly two active groups.",
    )
    parser.add_argument("--attention_variation_loss_weight", type=float, default=0.01)
    parser.add_argument("--attention_variation_target", type=float, default=0.05)
    parser.add_argument("--prior_velocity_loss_weight", type=float, default=0.25)
    parser.add_argument("--prior_accel_loss_weight", type=float, default=0.05)
    parser.add_argument("--prior_variance_floor_loss_weight", type=float, default=0.05)
    parser.add_argument("--prior_variance_floor_ratio", type=float, default=0.5)
    parser.add_argument("--negative_usage_loss_weight", type=float, default=0.02)

    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_last_every", type=int, default=10)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", "--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--resume_without_optimizer", "--resume-without-optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-flow-adapter")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    add_distributed_args(parser)
    args = parser.parse_args()
    if args.disable_softarranger:
        args.prior_mode = "concat"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def adapter_enabled(args):
    return not bool(getattr(args, "disable_adapter", False))


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def resolve_stats_paths(args, latent_codec):
    checkpoint_data_dir = latent_codec.checkpoint.get("data_config", {}).get("data_dir")
    stats_data_dir = args.stats_data_dir or (Path(checkpoint_data_dir) if checkpoint_data_dir else None)
    if stats_data_dir is None:
        stats_data_dir = args.data_dir
    stats_data_dir = Path(stats_data_dir)
    mean_path, std_path = rotation_rep_stats_paths(stats_data_dir, latent_codec.rotation_rep)
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(
            "Missing VAE normalization stats. Expected files: "
            f"{mean_path} and {std_path}. Pass --stats_data_dir if the VAE was trained with another dataset directory."
        )
    return stats_data_dir, mean_path, std_path


def make_dataset(args, split, limit, shuffle, mean_path, std_path, rotation_rep):
    return UpperSMPLXFlowDataset(
        args.data_dir,
        split=split,
        mean_path=mean_path,
        std_path=std_path,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        length_multiple=args.length_multiple,
        random_crop=shuffle and args.random_crop,
        limit=limit,
        rotation_rep=rotation_rep,
    )


# Set in main() to the active WordCandidateBuilder so DataLoader workers can
# re-seed its RNG per worker/epoch (negatives must differ across epochs).
_ACTIVE_CANDIDATE_BUILDER = None


def _candidate_worker_init(worker_id):
    import random as _random

    if _ACTIVE_CANDIDATE_BUILDER is not None:
        # torch.initial_seed() is assigned per worker by the DataLoader and
        # changes each epoch, so this yields fresh negatives every epoch.
        _ACTIVE_CANDIDATE_BUILDER.rng = _random.Random(int(torch.initial_seed()) % (2**31))


def make_prior_collate(args, prior_builder, candidate_builder, max_motion_frames, shuffle_candidates):
    """Build the concat prior (and, for soft_arranger, the candidate batch) inside
    the DataLoader worker so it parallelizes and overlaps GPU compute instead of
    blocking the main training loop. Tensors are built on CPU and moved to the
    device in compute_adapter_losses."""

    def _collate(items):
        batch = collate_upper_smplx(items)
        texts = condition_texts_from_batch(batch, args)
        lengths = batch["length"].tolist()
        max_len = int(batch["motion"].shape[1])
        prior_raw, prior_stats = prior_builder.batch(
            texts, lengths, max_len=max_len, device=None, dtype=torch.float32
        )
        batch["prior_raw"] = prior_raw
        batch["prior_stats"] = prior_stats
        if candidate_builder is not None:
            batch["candidate_batch"] = candidate_builder.batch(
                texts,
                device=None,
                dtype=torch.float32,
                shuffle=shuffle_candidates,
                max_motion_frames=max_motion_frames,
            )
        return batch

    return _collate


def make_loader(
    args,
    split,
    limit,
    shuffle,
    mean_path,
    std_path,
    rotation_rep,
    pin_memory=False,
    distributed=False,
    world_size=1,
    collate_fn=None,
    dataset=None,
    worker_init_fn=None,
):
    if dataset is None:
        dataset = make_dataset(args, split, limit, shuffle, mean_path, std_path, rotation_rep)
    if collate_fn is None:
        collate_fn = collate_upper_smplx
    enough_global_samples = len(dataset) >= max(int(world_size), 1) * args.batch_size
    drop_last = shuffle and len(dataset) >= args.batch_size and (not distributed or enough_global_samples)
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
    )
    return dataset, loader, sampler


def build_feature_weights(mask, left_valid, right_valid, args, device, rotation_rep):
    dim = mask.shape[1]
    slices = rotation_rep_slices(rotation_rep)
    weights = feature_weight_vector(args.hand_weight, device=device, rotation_rep=rotation_rep).view(1, 1, -1)
    weights = weights.expand(mask.shape[0], dim, weights.shape[-1]).clone()
    weights[:, :, slices["jaw"]] *= float(args.jaw_weight)
    weights[:, :, slices["expression"]] *= float(args.expression_weight)
    valid_floor = float(args.hand_valid_floor)
    left_scale = valid_floor + (1.0 - valid_floor) * left_valid.to(device).unsqueeze(-1)
    right_scale = valid_floor + (1.0 - valid_floor) * right_valid.to(device).unsqueeze(-1)
    weights[:, :, slices["left_hand"]] *= left_scale
    weights[:, :, slices["right_hand"]] *= right_scale
    weights *= mask.to(device=device, dtype=weights.dtype).unsqueeze(-1)
    return weights


def weighted_smooth_l1(pred, target, weights):
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def weighted_mse(pred, target, weights):
    loss = (pred - target).pow(2) * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def diff_weights(weights, order=1):
    out = weights
    for _ in range(order):
        out = torch.minimum(out[:, 1:], out[:, :-1])
    return out


def third_difference(x):
    return x[:, 3:] - 3.0 * x[:, 2:-1] + 3.0 * x[:, 1:-2] - x[:, :-3]


def coverage_metrics(prior_stats, device):
    if not prior_stats:
        return {
            "prior_cov": torch.tensor(0.0, device=device),
            "prior_matches": torch.tensor(0.0, device=device),
        }
    coverage = sum(float(item["coverage"]) for item in prior_stats) / len(prior_stats)
    matched = sum(float(item["matched_count"]) for item in prior_stats) / len(prior_stats)
    return {
        "prior_cov": torch.tensor(coverage, device=device),
        "prior_matches": torch.tensor(matched, device=device),
    }


def cross_covariance_orth_loss(content, style, eps=1e-6):
    """Penalize shared batch covariance between content and style embeddings."""

    if content.shape[0] < 2:
        return content.new_tensor(0.0)
    content = content - content.mean(dim=0, keepdim=True)
    style = style - style.mean(dim=0, keepdim=True)
    content = content / content.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))
    style = style / style.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))
    cross_cov = content.transpose(0, 1).matmul(style) / float(content.shape[0])
    return cross_cov.pow(2).mean()


def masked_bce_with_logits(logits, labels, mask):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = mask.to(device=logits.device, dtype=logits.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mean_value(values, mask):
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def attention_smoothness_loss(attention, target_mask):
    if attention.shape[1] <= 1:
        return attention.new_tensor(0.0)
    diff = (attention[:, 1:] - attention[:, :-1]).abs().sum(dim=(-1, -2))
    valid = (target_mask[:, 1:] & target_mask[:, :-1]).to(device=attention.device, dtype=attention.dtype)
    return (diff * valid).sum() / valid.sum().clamp_min(1.0)


def _empty_group_usage(attention):
    batch = int(attention.shape[0])
    return attention.new_zeros((batch, 0))


def candidate_group_frame_attention(attention, group_ids, group_mask):
    if group_mask is None or group_mask.numel() == 0:
        batch, target_len = int(attention.shape[0]), int(attention.shape[1])
        return attention.new_zeros((batch, target_len, 0))
    group_mask = group_mask.to(device=attention.device, dtype=torch.bool)
    group_ids = group_ids.to(device=attention.device, dtype=torch.long)
    group_count = int(group_mask.shape[1])
    if group_count <= 0:
        batch, target_len = int(attention.shape[0]), int(attention.shape[1])
        return attention.new_zeros((batch, target_len, 0))

    batch, target_len, _num_candidates, _word_len = attention.shape
    candidate_mass = attention.sum(dim=-1)
    valid_candidate_group = group_ids >= 0
    safe_group_ids = group_ids.clamp_min(0).clamp_max(group_count - 1)
    scatter_ids = safe_group_ids[:, None, :].expand(batch, target_len, -1)
    scatter_values = candidate_mass * valid_candidate_group[:, None, :].to(dtype=attention.dtype)
    group_attention = attention.new_zeros((batch, target_len, group_count))
    group_attention.scatter_add_(dim=2, index=scatter_ids, src=scatter_values)
    return group_attention * group_mask[:, None, :].to(dtype=attention.dtype)


def candidate_group_usage(attention, group_ids, group_mask, target_mask):
    group_attention = candidate_group_frame_attention(attention, group_ids, group_mask)
    if group_attention.shape[-1] <= 0:
        return _empty_group_usage(attention)
    valid_target = target_mask.to(device=attention.device, dtype=attention.dtype)
    denom = valid_target.sum(dim=1, keepdim=True).clamp_min(1.0)
    usage = (group_attention * valid_target[:, :, None]).sum(dim=1) / denom
    group_mask = group_mask.to(device=attention.device, dtype=torch.bool)
    return usage * group_mask.to(dtype=attention.dtype)


def group_coverage_loss(attention, group_ids, group_mask, target_mask, min_total_mass=0.5):
    usage = candidate_group_usage(attention, group_ids, group_mask, target_mask)
    if usage.numel() == 0:
        zero = attention.new_tensor(0.0)
        return zero, zero, zero, zero
    group_mask = group_mask.to(device=attention.device, dtype=torch.bool)
    valid_groups = group_mask.to(dtype=attention.dtype)
    group_counts = valid_groups.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_per_group = float(min_total_mass) / group_counts
    missing = F.relu(target_per_group - usage).pow(2) * valid_groups
    denom = valid_groups.sum().clamp_min(1.0)
    loss = missing.sum() / denom
    coverage_rate = ((usage >= target_per_group) & group_mask).to(dtype=attention.dtype).sum() / denom
    group_total = usage.sum(dim=1)
    has_group = group_mask.any(dim=1)
    if has_group.any():
        group_max = usage.masked_fill(~group_mask, -1.0).max(dim=1).values
        group_total_mean = group_total[has_group].mean()
        group_max_mean = group_max[has_group].mean()
    else:
        group_total_mean = attention.new_tensor(0.0)
        group_max_mean = attention.new_tensor(0.0)
    return loss, coverage_rate, group_total_mean, group_max_mean


def per_sample_group_entropy_peakiness(attention, group_ids, group_mask, target_mask, eps=1e-8):
    group_attention = candidate_group_frame_attention(attention, group_ids, group_mask)
    batch = int(attention.shape[0])
    if group_attention.shape[-1] <= 0:
        zero = attention.new_zeros(batch)
        return zero, zero

    group_mass = group_attention.sum(dim=-1, keepdim=True)
    probs = group_attention / group_mass.clamp_min(float(eps))
    probs = probs * (group_mass > float(eps)).to(dtype=attention.dtype)
    entropy = -(probs * probs.clamp_min(float(eps)).log()).sum(dim=-1)
    peak_prob = probs.max(dim=-1).values

    has_positive_mass = group_mass.squeeze(-1) > float(eps)
    valid_frame = target_mask.to(device=attention.device, dtype=torch.bool) & has_positive_mass
    valid_weight = valid_frame.to(dtype=attention.dtype)
    denom = valid_weight.sum(dim=1).clamp_min(1.0)
    has_valid = valid_frame.any(dim=1).to(dtype=attention.dtype)
    sample_entropy = (entropy * valid_weight).sum(dim=1) / denom
    sample_peak = (peak_prob * valid_weight).sum(dim=1) / denom
    return sample_entropy * has_valid, sample_peak * has_valid


def group_entropy_peakiness_loss(attention, group_ids, group_mask, target_mask, target_entropy=0.6931471805599453, eps=1e-8):
    group_attention = candidate_group_frame_attention(attention, group_ids, group_mask)
    if group_attention.shape[-1] <= 0:
        zero = attention.new_tensor(0.0)
        return zero, zero, zero

    group_mass = group_attention.sum(dim=-1, keepdim=True)
    probs = group_attention / group_mass.clamp_min(float(eps))
    probs = probs * (group_mass > float(eps)).to(dtype=attention.dtype)
    entropy = -(probs * probs.clamp_min(float(eps)).log()).sum(dim=-1)
    peak_prob = probs.max(dim=-1).values
    valid_frame = target_mask.to(device=attention.device, dtype=torch.bool) & (group_mass.squeeze(-1) > float(eps))
    if valid_frame.sum() <= 0:
        zero = attention.new_tensor(0.0)
        return zero, zero, zero

    valid_weight = valid_frame.to(dtype=attention.dtype)
    denom = valid_weight.sum().clamp_min(1.0)
    loss = (F.relu(entropy - float(target_entropy)) * valid_weight).sum() / denom
    entropy_mean = (entropy * valid_weight).sum() / denom
    peak_mean = (peak_prob * valid_weight).sum() / denom
    return loss, entropy_mean, peak_mean


def attention_variation_loss(attention, target_mask, target_variation=0.05):
    if attention.shape[1] <= 1:
        zero = attention.new_tensor(0.0)
        return zero, zero, zero
    flat = attention.reshape(attention.shape[0], attention.shape[1], -1)
    valid = (target_mask[:, 1:] & target_mask[:, :-1]).to(device=attention.device, dtype=attention.dtype)
    if valid.sum() <= 0:
        zero = attention.new_tensor(0.0)
        return zero, zero, zero
    frame_l1 = (flat[:, 1:] - flat[:, :-1]).abs().sum(dim=-1)
    loss = (F.relu(float(target_variation) - frame_l1).pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
    temporal_l1 = (frame_l1 * valid).sum() / valid.sum().clamp_min(1.0)
    frame_cos = F.cosine_similarity(flat[:, 1:], flat[:, :-1], dim=-1, eps=1e-8)
    frame_cos = (frame_cos * valid).sum() / valid.sum().clamp_min(1.0)
    return loss, temporal_l1, frame_cos


def latent_difference(x, order):
    out = x
    for _ in range(int(order)):
        out = out[:, 1:] - out[:, :-1]
    return out


def latent_diff_mask(mask, order):
    out = mask
    for _ in range(int(order)):
        out = out[:, 1:] & out[:, :-1]
    return out


def latent_dynamics_loss(pred, target, mask, order):
    if pred.shape[1] <= int(order):
        return pred.new_tensor(0.0)
    pred_diff = latent_difference(pred, order)
    target_diff = latent_difference(target, order)
    valid = latent_diff_mask(mask, order).to(device=pred.device, dtype=pred.dtype)
    weights = valid.unsqueeze(-1).expand_as(pred_diff)
    return weighted_smooth_l1(pred_diff, target_diff, weights)


def masked_delta_rms(x, mask, order=1):
    if x.shape[1] <= int(order):
        return x.new_tensor(0.0)
    diff = latent_difference(x, order)
    valid = latent_diff_mask(mask, order).to(device=x.device, dtype=x.dtype).unsqueeze(-1).expand_as(diff)
    denom = valid.sum().clamp_min(1.0)
    return ((diff.pow(2) * valid).sum() / denom).sqrt()


def masked_temporal_std(x, mask, eps=1e-6):
    weights = mask.to(device=x.device, dtype=x.dtype)
    counts = weights.sum(dim=1)
    valid = counts > 1
    denom = counts.clamp_min(1.0).view(-1, 1)
    mean = (x * weights.unsqueeze(-1)).sum(dim=1) / denom
    var = ((x - mean[:, None, :]).pow(2) * weights.unsqueeze(-1)).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(float(eps)))
    sample_std = std.mean(dim=-1)
    return sample_std, valid


def prior_variance_floor_loss(prior, target, mask, floor_ratio=0.5):
    prior_std, valid = masked_temporal_std(prior, mask)
    target_std, _target_valid = masked_temporal_std(target, mask)
    if not valid.any():
        zero = prior.new_tensor(0.0)
        return zero, zero, zero
    loss = F.relu(float(floor_ratio) * target_std.detach() - prior_std).pow(2)
    loss = loss[valid].mean()
    return loss, prior_std[valid].mean(), target_std[valid].mean()


def encode_word_candidates(candidate_batch, latent_codec, latent_stats, device, max_word_latent_frames=None):
    word_motion = candidate_batch.motion
    word_mask = candidate_batch.frame_mask
    if max_word_latent_frames is not None:
        max_motion_frames = min(
            int(latent_codec.max_frames),
            int(max_word_latent_frames) * int(latent_codec.downsample_factor),
        )
    else:
        max_motion_frames = int(latent_codec.max_frames)
    if word_motion.shape[2] > max_motion_frames:
        word_motion = word_motion[:, :, :max_motion_frames]
        word_mask = word_mask[:, :, :max_motion_frames]
    batch_size, num_candidates, word_frames, dim = word_motion.shape
    flat_motion = word_motion.reshape(batch_size * num_candidates, word_frames, dim)
    flat_mask = word_mask.reshape(batch_size * num_candidates, word_frames)
    word_z_raw, word_z_mask = latent_codec.encode(flat_motion, mask=flat_mask)
    word_z = latent_codec.normalize_latent(word_z_raw, latent_stats)
    word_z = word_z.reshape(batch_size, num_candidates, word_z.shape[1], word_z.shape[2])
    word_z_mask = word_z_mask.reshape(batch_size, num_candidates, word_z_mask.shape[1])
    valid_candidates = candidate_batch.candidate_mask.to(device=device, dtype=torch.bool)
    word_z = torch.where(valid_candidates[:, :, None, None], word_z, torch.zeros_like(word_z))
    word_z_mask = word_z_mask & valid_candidates[:, :, None]
    return word_z, word_z_mask


def encode_word_text_features(text_encoder, candidate_texts, device, dtype):
    flat_texts = [text for row in candidate_texts for text in row]
    word_text = text_encoder.encode(flat_texts).to(device=device, dtype=dtype)
    batch_size = len(candidate_texts)
    num_candidates = len(candidate_texts[0]) if candidate_texts else 0
    return word_text.reshape(batch_size, num_candidates, -1)


def condition_texts_from_batch(batch, args):
    if args.condition_field == "gloss":
        return [gloss if gloss else text for text, gloss in zip(batch["text"], batch["gloss"])]
    if args.condition_field == "text_gloss":
        return [
            f"{text} {gloss}".strip() if gloss else text
            for text, gloss in zip(batch["text"], batch["gloss"])
        ]
    if args.condition_field == "label_word":
        label_words = batch.get("label_word", batch["text"])
        return [lw if lw else text for text, lw in zip(batch["text"], label_words)]
    return batch["text"]


def compute_adapter_losses(
    adapter,
    batch,
    args,
    device,
    prior_builder,
    latent_codec,
    latent_stats,
    arranger=None,
    candidate_builder=None,
    text_encoder=None,
):
    use_adapter = adapter_enabled(args)
    raw_x = batch["motion"].to(device)
    raw_mask = batch["mask"].to(device)
    left_valid = batch["left_valid"].to(device)
    right_valid = batch["right_valid"].to(device)
    lengths = batch["length"].detach().cpu().tolist()
    condition_texts = condition_texts_from_batch(batch, args)

    with torch.no_grad():
        z_sent_raw, latent_mask = latent_codec.encode(raw_x, mask=raw_mask)
        z_sent = latent_codec.normalize_latent(z_sent_raw, latent_stats)
        if "prior_raw" in batch:
            prior_raw = batch["prior_raw"].to(device=device, dtype=raw_x.dtype)
            prior_stats = batch["prior_stats"]
        else:
            prior_raw, prior_stats = prior_builder.batch(
                condition_texts,
                lengths,
                max_len=raw_x.shape[1],
                device=device,
                dtype=raw_x.dtype,
            )
        prior_raw = prior_raw * raw_mask.to(device=device, dtype=prior_raw.dtype).unsqueeze(-1)
        z_word_raw, _ = latent_codec.encode(prior_raw, mask=raw_mask)
        z_word = latent_codec.normalize_latent(z_word_raw, latent_stats)

    arranger_out = None
    candidate_batch = None
    z_source = z_word
    prior_latent_loss = weighted_smooth_l1(z_source, z_sent, latent_mask.to(device=device, dtype=z_sent.dtype).unsqueeze(-1).expand_as(z_sent))
    gate_bce_loss = raw_x.new_tensor(0.0)
    gate_sparsity_loss = raw_x.new_tensor(0.0)
    attn_smooth_loss = raw_x.new_tensor(0.0)
    group_cov_loss = raw_x.new_tensor(0.0)
    group_entropy_peak_loss = raw_x.new_tensor(0.0)
    attn_var_loss = raw_x.new_tensor(0.0)
    prior_vel_loss = raw_x.new_tensor(0.0)
    prior_accel_latent_loss = raw_x.new_tensor(0.0)
    prior_var_floor_loss = raw_x.new_tensor(0.0)
    negative_usage_loss = raw_x.new_tensor(0.0)
    null_usage_loss = raw_x.new_tensor(0.0)
    gate_pos = raw_x.new_tensor(0.0)
    gate_neg = raw_x.new_tensor(0.0)
    word_usage_pos = raw_x.new_tensor(0.0)
    word_usage_neg = raw_x.new_tensor(0.0)
    group_coverage = raw_x.new_tensor(0.0)
    group_usage_total = raw_x.new_tensor(0.0)
    group_usage_max = raw_x.new_tensor(0.0)
    group_entropy = raw_x.new_tensor(0.0)
    group_peak_prob = raw_x.new_tensor(0.0)
    attention_temporal_l1 = raw_x.new_tensor(0.0)
    attention_frame_cosine = raw_x.new_tensor(0.0)
    prior_latent_std = raw_x.new_tensor(0.0)
    gt_latent_std = raw_x.new_tensor(0.0)
    prior_delta_rms = raw_x.new_tensor(0.0)
    gt_delta_rms = raw_x.new_tensor(0.0)

    if args.prior_mode == "soft_arranger":
        if arranger is None or candidate_builder is None or text_encoder is None:
            raise RuntimeError("prior_mode=soft_arranger requires arranger, candidate_builder, and text_encoder.")
        if "candidate_batch" in batch:
            candidate_batch = batch["candidate_batch"].to(device)
        else:
            candidate_batch = candidate_builder.batch(
                condition_texts,
                device=device,
                dtype=raw_x.dtype,
                shuffle=bool(
                    args.shuffle_word_candidates
                    and (arranger.training if arranger is not None else adapter.training)
                ),
                max_motion_frames=min(
                    int(latent_codec.max_frames),
                    int(args.max_word_latent_frames) * int(latent_codec.downsample_factor),
                ),
            )
        with torch.no_grad():
            word_latents, word_latent_mask = encode_word_candidates(
                candidate_batch,
                latent_codec,
                latent_stats,
                device,
                max_word_latent_frames=args.max_word_latent_frames,
            )
            sentence_text = text_encoder.encode(condition_texts).to(device=device, dtype=raw_x.dtype)
            word_text = encode_word_text_features(text_encoder, candidate_batch.texts, device, raw_x.dtype)
        arranger_out = arranger(
            sentence_text,
            word_text,
            word_latents,
            word_latent_mask,
            candidate_batch.candidate_mask,
            latent_mask,
        )
        z_source = arranger_out["z_prior_aligned"]
        prior_latent_loss = weighted_smooth_l1(z_source, z_sent, latent_mask.to(device=device, dtype=z_sent.dtype).unsqueeze(-1).expand_as(z_sent))
        prior_vel_loss = latent_dynamics_loss(z_source, z_sent, latent_mask, order=1)
        prior_accel_latent_loss = latent_dynamics_loss(z_source, z_sent, latent_mask, order=2)
        prior_var_floor_loss, prior_latent_std, gt_latent_std = prior_variance_floor_loss(
            z_source,
            z_sent,
            latent_mask,
            floor_ratio=args.prior_variance_floor_ratio,
        )
        prior_delta_rms = masked_delta_rms(z_source, latent_mask, order=1)
        gt_delta_rms = masked_delta_rms(z_sent, latent_mask, order=1)
        gate_bce_loss = masked_bce_with_logits(
            arranger_out["word_gate_logits"],
            candidate_batch.labels,
            candidate_batch.candidate_mask,
        )
        gate_sparsity_loss = masked_mean_value(
            arranger_out["word_gate_probs"],
            candidate_batch.candidate_mask,
        )
        attn_smooth_loss = attention_smoothness_loss(arranger_out["attention"], latent_mask)
        group_cov_loss, group_coverage, group_usage_total, group_usage_max = group_coverage_loss(
            arranger_out["attention"],
            candidate_batch.group_ids,
            candidate_batch.group_mask,
            latent_mask,
            min_total_mass=args.group_coverage_mass,
        )
        group_entropy_peak_loss, group_entropy, group_peak_prob = group_entropy_peakiness_loss(
            arranger_out["attention"],
            candidate_batch.group_ids,
            candidate_batch.group_mask,
            latent_mask,
            target_entropy=args.group_entropy_peak_target,
        )
        attn_var_loss, attention_temporal_l1, attention_frame_cosine = attention_variation_loss(
            arranger_out["attention"],
            latent_mask,
            target_variation=args.attention_variation_target,
        )
        null_usage_loss = arranger_out["null_usage"].mean()
        pos_mask = candidate_batch.candidate_mask & (candidate_batch.labels > 0.5)
        neg_mask = candidate_batch.candidate_mask & (candidate_batch.labels <= 0.5)
        if pos_mask.any():
            gate_pos = masked_mean_value(arranger_out["word_gate_probs"], pos_mask)
            word_usage_pos = masked_mean_value(arranger_out["word_usage"], pos_mask)
        if neg_mask.any():
            gate_neg = masked_mean_value(arranger_out["word_gate_probs"], neg_mask)
            word_usage_neg = masked_mean_value(arranger_out["word_usage"], neg_mask)
            negative_usage_loss = word_usage_neg

    latent_weights = latent_mask.to(device=device, dtype=z_sent.dtype).unsqueeze(-1).expand_as(z_sent)
    content_pair_loss = raw_x.new_tensor(0.0)
    style_loss = raw_x.new_tensor(0.0)
    style_acc = raw_x.new_tensor(0.0)
    orth_loss = raw_x.new_tensor(0.0)
    content_domain_loss = raw_x.new_tensor(0.0)
    content_domain_acc = raw_x.new_tensor(0.0)
    delta_loss = raw_x.new_tensor(0.0)

    if use_adapter:
        word_out = adapter(z_source, mask=latent_mask)
        sent_out = adapter(z_sent, mask=latent_mask)
        z_adapt = word_out["z_adapt"]

        content_pair_loss = F.smooth_l1_loss(word_out["content_pooled"], sent_out["content_pooled"])

        style_logits = torch.cat([word_out["style_logits"], sent_out["style_logits"]], dim=0)
        style_labels = torch.cat(
            [
                torch.zeros(z_word.shape[0], dtype=torch.long, device=device),
                torch.ones(z_sent.shape[0], dtype=torch.long, device=device),
            ],
            dim=0,
        )
        style_loss = F.cross_entropy(style_logits, style_labels)
        style_acc = (style_logits.argmax(dim=-1) == style_labels).float().mean()
        content = torch.cat([word_out["content_pooled"], sent_out["content_pooled"]], dim=0)
        style = torch.cat([word_out["style_pooled"], sent_out["style_pooled"]], dim=0)
        orth_loss = cross_covariance_orth_loss(content, style)
        if "content_domain_logits" in word_out and "content_domain_logits" in sent_out:
            content_domain_logits = torch.cat(
                [word_out["content_domain_logits"], sent_out["content_domain_logits"]],
                dim=0,
            )
            content_domain_loss = F.cross_entropy(content_domain_logits, style_labels)
            content_domain_acc = (content_domain_logits.argmax(dim=-1) == style_labels).float().mean()
        elif args.content_domain_confusion_loss_weight > 0:
            raise RuntimeError(
                "content_domain_confusion_loss_weight is positive, but the adapter was built without "
                "a content-domain classifier."
            )
        delta_loss = weighted_mse(word_out["delta"], torch.zeros_like(word_out["delta"]), latent_weights)
    else:
        z_adapt = z_source * latent_mask.to(device=device, dtype=z_source.dtype).unsqueeze(-1)

    latent_loss = weighted_smooth_l1(z_adapt, z_sent, latent_weights)
    latent_word_loss = weighted_smooth_l1(z_word, z_sent, latent_weights)

    z_adapt_raw = latent_codec.denormalize_latent(z_adapt, latent_stats)
    x_adapt = latent_codec.decode(
        z_adapt_raw,
        target_length=raw_x.shape[1],
        mask=raw_mask,
        latent_mask=latent_mask,
    )

    pose_weights = build_feature_weights(raw_mask, left_valid, right_valid, args, device, args.rotation_rep)
    pose_loss = weighted_smooth_l1(x_adapt, raw_x, pose_weights)
    pose_word_loss = weighted_smooth_l1(prior_raw, raw_x, pose_weights)

    vel_loss = raw_x.new_tensor(0.0)
    vel_word_loss = raw_x.new_tensor(0.0)
    if raw_x.shape[1] > 1:
        vel_weights = diff_weights(pose_weights, order=1)
        vel_loss = weighted_smooth_l1(x_adapt[:, 1:] - x_adapt[:, :-1], raw_x[:, 1:] - raw_x[:, :-1], vel_weights)
        vel_word_loss = weighted_smooth_l1(
            prior_raw[:, 1:] - prior_raw[:, :-1],
            raw_x[:, 1:] - raw_x[:, :-1],
            vel_weights,
        )

    accel_loss = raw_x.new_tensor(0.0)
    accel_word_loss = raw_x.new_tensor(0.0)
    if raw_x.shape[1] > 2:
        accel_weights = diff_weights(pose_weights, order=2)
        pred_accel = x_adapt[:, 2:] - 2.0 * x_adapt[:, 1:-1] + x_adapt[:, :-2]
        true_accel = raw_x[:, 2:] - 2.0 * raw_x[:, 1:-1] + raw_x[:, :-2]
        word_accel = prior_raw[:, 2:] - 2.0 * prior_raw[:, 1:-1] + prior_raw[:, :-2]
        accel_loss = weighted_smooth_l1(pred_accel, true_accel, accel_weights)
        accel_word_loss = weighted_smooth_l1(word_accel, true_accel, accel_weights)

    jerk_loss = raw_x.new_tensor(0.0)
    jerk_word_loss = raw_x.new_tensor(0.0)
    if raw_x.shape[1] > 3:
        jerk_weights = diff_weights(pose_weights, order=3)
        jerk_loss = weighted_smooth_l1(third_difference(x_adapt), third_difference(raw_x), jerk_weights)
        jerk_word_loss = weighted_smooth_l1(third_difference(prior_raw), third_difference(raw_x), jerk_weights)

    arranger_loss_scale = 1.0 if args.prior_mode == "soft_arranger" else 0.0
    adapter_loss_scale = 1.0 if use_adapter else 0.0
    total = (
        args.latent_loss_weight * latent_loss
        + args.pose_loss_weight * pose_loss
        + args.velocity_loss_weight * vel_loss
        + args.accel_loss_weight * accel_loss
        + args.jerk_loss_weight * jerk_loss
        + adapter_loss_scale * args.style_loss_weight * style_loss
        + adapter_loss_scale * args.content_pair_loss_weight * content_pair_loss
        + adapter_loss_scale * args.delta_loss_weight * delta_loss
        + adapter_loss_scale * args.orth_loss_weight * orth_loss
        + adapter_loss_scale * args.content_domain_confusion_loss_weight * content_domain_loss
        + arranger_loss_scale * args.arranger_prior_loss_weight * prior_latent_loss
        + arranger_loss_scale * args.gate_bce_loss_weight * gate_bce_loss
        + arranger_loss_scale * args.gate_sparsity_loss_weight * gate_sparsity_loss
        + arranger_loss_scale * args.attention_smoothness_weight * attn_smooth_loss
        + arranger_loss_scale * args.null_usage_loss_weight * null_usage_loss
        + arranger_loss_scale * args.group_coverage_loss_weight * group_cov_loss
        + arranger_loss_scale * args.group_entropy_peak_loss_weight * group_entropy_peak_loss
        + arranger_loss_scale * args.attention_variation_loss_weight * attn_var_loss
        + arranger_loss_scale * args.prior_velocity_loss_weight * prior_vel_loss
        + arranger_loss_scale * args.prior_accel_loss_weight * prior_accel_latent_loss
        + arranger_loss_scale * args.prior_variance_floor_loss_weight * prior_var_floor_loss
        + arranger_loss_scale * args.negative_usage_loss_weight * negative_usage_loss
    )

    out = {
        "loss": total,
        "latent": latent_loss.detach(),
        "latent_word": latent_word_loss.detach(),
        "latent_gain": (latent_word_loss - latent_loss).detach(),
        "latent_prior": prior_latent_loss.detach(),
        "latent_prior_gain": (latent_word_loss - prior_latent_loss).detach(),
        "latent_adapt_gain": (prior_latent_loss - latent_loss).detach(),
        "pose": pose_loss.detach(),
        "pose_word": pose_word_loss.detach(),
        "pose_gain": (pose_word_loss - pose_loss).detach(),
        "vel": vel_loss.detach(),
        "vel_word": vel_word_loss.detach(),
        "accel": accel_loss.detach(),
        "accel_word": accel_word_loss.detach(),
        "jerk": jerk_loss.detach(),
        "jerk_word": jerk_word_loss.detach(),
        "style": style_loss.detach(),
        "style_acc": style_acc.detach(),
        "orth": orth_loss.detach(),
        "content_domain": content_domain_loss.detach(),
        "content_domain_acc": content_domain_acc.detach(),
        "content_pair": content_pair_loss.detach(),
        "delta": delta_loss.detach(),
        "gate_bce": gate_bce_loss.detach(),
        "gate_sparsity": gate_sparsity_loss.detach(),
        "attention_smooth": attn_smooth_loss.detach(),
        "group_coverage_loss": group_cov_loss.detach(),
        "group_entropy_peak_loss": group_entropy_peak_loss.detach(),
        "attention_variation_loss": attn_var_loss.detach(),
        "prior_velocity_latent": prior_vel_loss.detach(),
        "prior_accel_latent": prior_accel_latent_loss.detach(),
        "prior_variance_floor": prior_var_floor_loss.detach(),
        "negative_usage_loss": negative_usage_loss.detach(),
        "null_usage": null_usage_loss.detach(),
        "gate_pos": gate_pos.detach(),
        "gate_neg": gate_neg.detach(),
        "word_usage_pos": word_usage_pos.detach(),
        "word_usage_neg": word_usage_neg.detach(),
        "group_coverage": group_coverage.detach(),
        "group_usage_total": group_usage_total.detach(),
        "group_usage_max": group_usage_max.detach(),
        "group_entropy": group_entropy.detach(),
        "group_peak_prob": group_peak_prob.detach(),
        "attention_temporal_l1": attention_temporal_l1.detach(),
        "attention_frame_cosine": attention_frame_cosine.detach(),
        "prior_latent_std": prior_latent_std.detach(),
        "gt_latent_std": gt_latent_std.detach(),
        "prior_delta_rms": prior_delta_rms.detach(),
        "gt_delta_rms": gt_delta_rms.detach(),
    }
    out.update(coverage_metrics(prior_stats, device))
    return out


def average_losses(items):
    if not items:
        return {}
    keys = sorted(key for key in items[0] if key != "loss")
    return {key: float(torch.stack([item[key].detach().cpu() for item in items]).mean()) for key in keys}


@torch.no_grad()
def validate(adapter, loader, args, device, prior_builder, latent_codec, latent_stats, arranger=None, candidate_builder=None, text_encoder=None):
    adapter.eval()
    if arranger is not None:
        arranger.eval()
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        losses.append(
            compute_adapter_losses(
                adapter,
                batch,
                args,
                device,
                prior_builder,
                latent_codec,
                latent_stats,
                arranger=arranger,
                candidate_builder=candidate_builder,
                text_encoder=text_encoder,
            )
        )
    return average_losses(losses)


def split_model_bundle(model):
    if isinstance(model, dict):
        return model["adapter"], model.get("arranger")
    return model, None


def checkpoint_payload(model, optimizer, args, epoch, global_step):
    adapter, arranger = split_model_bundle(model)
    payload = {
        "model": unwrap_model(adapter).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "prior_mode": args.prior_mode,
        "adapter_enabled": adapter_enabled(args),
        "ablation_config": {
            "softarranger_enabled": args.prior_mode == "soft_arranger",
            "adapter_enabled": adapter_enabled(args),
            "disable_softarranger": bool(getattr(args, "disable_softarranger", False)),
            "disable_adapter": bool(getattr(args, "disable_adapter", False)),
            "disable_arranger_candidate_gates": bool(getattr(args, "disable_arranger_candidate_gates", False)),
            "disable_arranger_null_memory": bool(getattr(args, "disable_arranger_null_memory", False)),
            "disable_arranger_word_text_features": bool(getattr(args, "disable_arranger_word_text_features", False)),
            "disable_arranger_word_motion_latents": bool(getattr(args, "disable_arranger_word_motion_latents", False)),
        },
    }
    if arranger is not None:
        payload["arranger_model"] = unwrap_model(arranger).state_dict()
    payload.update(
        {
            "model_config": {
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "content_dim": args.content_dim,
                "style_dim": args.style_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "max_frames": args.flow_max_frames,
                "num_domains": 2,
                "use_content_domain_classifier": args.use_content_domain_classifier,
                "gradient_reversal_lambda": args.gradient_reversal_lambda,
            },
            "data_config": {
                "data_dir": str(args.data_dir),
                "word_data_dir": str(args.word_data_dir),
                "word_split": args.word_split,
                "stats_data_dir": str(args.stats_data_dir),
                "mean_path": str(args.mean_path),
                "std_path": str(args.std_path),
                "train_split": args.train_split,
                "val_split": args.val_split,
                "condition_field": args.condition_field,
                "min_frames": args.min_frames,
                "max_frames": args.max_frames,
                "length_multiple": args.length_multiple,
                "rotation_rep": args.rotation_rep,
                "normalized": True,
            },
            "latent_config": args.latent_config,
            "loss_config": {
                "hand_weight": args.hand_weight,
                "jaw_weight": args.jaw_weight,
                "expression_weight": args.expression_weight,
                "latent_loss_weight": args.latent_loss_weight,
                "pose_loss_weight": args.pose_loss_weight,
                "velocity_loss_weight": args.velocity_loss_weight,
                "accel_loss_weight": args.accel_loss_weight,
                "jerk_loss_weight": args.jerk_loss_weight,
                "style_loss_weight": args.style_loss_weight,
                "content_pair_loss_weight": args.content_pair_loss_weight,
                "delta_loss_weight": args.delta_loss_weight,
                "orth_loss_weight": args.orth_loss_weight,
                "content_domain_confusion_loss_weight": args.content_domain_confusion_loss_weight,
                "gradient_reversal_lambda": args.gradient_reversal_lambda,
                "arranger_prior_loss_weight": args.arranger_prior_loss_weight,
                "gate_bce_loss_weight": args.gate_bce_loss_weight,
                "gate_sparsity_loss_weight": args.gate_sparsity_loss_weight,
                "attention_smoothness_weight": args.attention_smoothness_weight,
                "null_usage_loss_weight": args.null_usage_loss_weight,
                "group_coverage_loss_weight": args.group_coverage_loss_weight,
                "group_coverage_mass": args.group_coverage_mass,
                "group_entropy_peak_loss_weight": args.group_entropy_peak_loss_weight,
                "group_entropy_peak_target": args.group_entropy_peak_target,
                "attention_variation_loss_weight": args.attention_variation_loss_weight,
                "attention_variation_target": args.attention_variation_target,
                "prior_velocity_loss_weight": args.prior_velocity_loss_weight,
                "prior_accel_loss_weight": args.prior_accel_loss_weight,
                "prior_variance_floor_loss_weight": args.prior_variance_floor_loss_weight,
                "prior_variance_floor_ratio": args.prior_variance_floor_ratio,
                "negative_usage_loss_weight": args.negative_usage_loss_weight,
            },
            "args": serializable_args(args),
        }
    )
    if args.prior_mode == "soft_arranger":
        payload["arranger_config"] = {
            "latent_dim": args.latent_dim,
            "text_dim": args.text_dim,
            "hidden_dim": args.arranger_hidden_dim,
            "num_heads": args.arranger_num_heads,
            "dropout": args.arranger_dropout,
            "max_frames": args.flow_max_frames,
            "max_word_latent_frames": args.max_word_latent_frames,
            "use_candidate_gates": not args.disable_arranger_candidate_gates,
            "use_null_memory": not args.disable_arranger_null_memory,
            "use_word_text_features": not args.disable_arranger_word_text_features,
            "use_word_motion_latents": not args.disable_arranger_word_motion_latents,
        }
        payload["text_config"] = {
            "text_model_path": str(args.text_model_path),
            "text_dim": args.text_dim,
            "max_text_tokens": args.max_text_tokens,
            "pooling": "mean",
            "frozen": True,
        }
        payload["candidate_config"] = {
            "num_word_candidates": args.num_word_candidates,
            "num_negative_candidates": args.num_negative_candidates,
            "candidate_selection": args.candidate_selection,
            "max_positive_variants_per_key": args.max_positive_variants_per_key,
            "shuffle_word_candidates": args.shuffle_word_candidates,
        }
    return payload


def save_checkpoint(path, model, optimizer, args, epoch, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, args, epoch, global_step), path)


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resume_training_state(path, adapter, optimizer, device, load_optimizer=True, arranger=None):
    ckpt = load_checkpoint(path, map_location="cpu")
    unwrap_model(adapter).load_state_dict(ckpt["model"], strict=True)
    if arranger is not None:
        if "arranger_model" not in ckpt:
            raise RuntimeError(f"Resume checkpoint {path} does not contain arranger_model.")
        unwrap_model(arranger).load_state_dict(ckpt["arranger_model"], strict=True)
    if load_optimizer and optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def main():
    args = parse_args()
    args.adapter_enabled = adapter_enabled(args)
    dist_info = setup_distributed(args)
    if dist_info["is_main"]:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + int(dist_info["rank"]))
    device = resolve_distributed_device(args.device, dist_info)
    pin_memory = device.type == "cuda"

    rank_zero_print(dist_info, f"Loading frozen VAE: {args.vae_checkpoint}")
    latent_codec = LatentMotionCodec(args.vae_checkpoint, device=device)
    args.rotation_rep = normalize_rotation_rep(latent_codec.rotation_rep)
    args.latent_dim = int(latent_codec.latent_dim)
    args.flow_max_frames = int(latent_codec.max_latent_frames)
    args.use_content_domain_classifier = bool(args.content_domain_confusion_loss_weight > 0)
    args.text_dim = 0
    stats_data_dir, mean_path, std_path = resolve_stats_paths(args, latent_codec)
    args.stats_data_dir = stats_data_dir
    args.mean_path = mean_path
    args.std_path = std_path

    candidate_builder = None
    text_encoder = None
    arranger = None
    if args.prior_mode == "soft_arranger":
        rank_zero_print(dist_info, f"Loading frozen text encoder: {args.text_model_path}")
        text_encoder = FrozenT5TextEncoder(
            args.text_model_path,
            device=device,
            max_length=args.max_text_tokens,
        )
        args.text_dim = int(text_encoder.text_dim)

    adapter = ContentStyleAdapter(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        content_dim=args.content_dim,
        style_dim=args.style_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_frames=args.flow_max_frames,
        num_domains=2,
        use_content_domain_classifier=args.use_content_domain_classifier,
        gradient_reversal_lambda=args.gradient_reversal_lambda,
    ).to(device)
    if not args.adapter_enabled:
        adapter = freeze_module(adapter)
    if args.prior_mode == "soft_arranger":
        arranger = SoftWordArranger(
            latent_dim=args.latent_dim,
            text_dim=args.text_dim,
            hidden_dim=args.arranger_hidden_dim,
            num_heads=args.arranger_num_heads,
            dropout=args.arranger_dropout,
            max_frames=args.flow_max_frames,
            max_word_latent_frames=args.max_word_latent_frames,
            use_candidate_gates=not args.disable_arranger_candidate_gates,
            use_null_memory=not args.disable_arranger_null_memory,
            use_word_text_features=not args.disable_arranger_word_text_features,
            use_word_motion_latents=not args.disable_arranger_word_motion_latents,
        ).to(device)

    rank_zero_print(dist_info, "Wrapping trainable models for distributed training...")
    if args.adapter_enabled:
        adapter = wrap_model(adapter, dist_info, device)
    if arranger is not None:
        arranger = wrap_model(arranger, dist_info, device)
    trainable_parameters = list(adapter.parameters()) if args.adapter_enabled else []
    if arranger is not None:
        trainable_parameters.extend(arranger.parameters())
    model_bundle = {"adapter": adapter, "arranger": arranger} if arranger is not None else adapter
    optimizer = (
        torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
        if trainable_parameters
        else None
    )
    if optimizer is None:
        rank_zero_print(
            dist_info,
            "Ablation has no trainable parameters; this run will report deterministic prior metrics only.",
        )

    resume_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_from_checkpoint}")
        resume_epoch, global_step = resume_training_state(
            args.resume_from_checkpoint,
            adapter,
            optimizer,
            device,
            load_optimizer=not args.resume_without_optimizer,
            arranger=arranger,
        )

    # Datasets are created first (we need their normalization stats to build the
    # word prior); the loaders are constructed afterwards so their collate_fn can
    # build the prior/candidates in the workers.
    rank_zero_print(dist_info, f"Loading train dataset split={args.train_split}")
    train_dataset = make_dataset(
        args, args.train_split, args.limit_train, True, mean_path, std_path, args.rotation_rep
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found under {args.data_dir}")

    val_dataset = None
    val_manifest = args.data_dir / "meta" / f"manifest_{args.val_split}.jsonl"
    want_val = dist_info["is_main"] and args.val_every > 0 and val_manifest.is_file()
    if want_val:
        val_dataset = make_dataset(
            args, args.val_split, args.limit_val, False, mean_path, std_path, args.rotation_rep
        )

    rank_zero_print(
        dist_info,
        f"Loading word prior: {args.word_data_dir} split={args.word_split}",
    )
    prior_builder = WordMotionPrior(
        args.word_data_dir,
        split=args.word_split,
        target_mean=train_dataset.mean,
        target_std=train_dataset.std,
        rotation_rep=args.rotation_rep,
        lazy_motions=args.lazy_word_motions,
    )
    if args.prior_mode == "soft_arranger":
        candidate_builder = WordCandidateBuilder(
            prior_builder,
            num_word_candidates=args.num_word_candidates,
            num_negative_candidates=args.num_negative_candidates,
            candidate_selection=args.candidate_selection,
            max_positive_variants_per_key=args.max_positive_variants_per_key,
            seed=args.seed,
        )

    global _ACTIVE_CANDIDATE_BUILDER
    _ACTIVE_CANDIDATE_BUILDER = candidate_builder
    max_word_motion_frames = min(
        int(latent_codec.max_frames),
        int(args.max_word_latent_frames) * int(latent_codec.downsample_factor),
    )
    train_collate = make_prior_collate(
        args, prior_builder, candidate_builder, max_word_motion_frames,
        shuffle_candidates=bool(args.shuffle_word_candidates),
    )
    _, train_loader, train_sampler = make_loader(
        args,
        args.train_split,
        args.limit_train,
        shuffle=True,
        mean_path=mean_path,
        std_path=std_path,
        rotation_rep=args.rotation_rep,
        pin_memory=pin_memory,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
        collate_fn=train_collate,
        dataset=train_dataset,
        worker_init_fn=_candidate_worker_init,
    )

    val_loader = None
    if want_val:
        val_collate = make_prior_collate(
            args, prior_builder, candidate_builder, max_word_motion_frames,
            shuffle_candidates=False,
        )
        _, val_loader, _ = make_loader(
            args,
            args.val_split,
            args.limit_val,
            shuffle=False,
            mean_path=mean_path,
            std_path=std_path,
            rotation_rep=args.rotation_rep,
            pin_memory=pin_memory,
            collate_fn=val_collate,
            dataset=val_dataset,
            worker_init_fn=_candidate_worker_init,
        )

    rank_zero_print(dist_info, "Computing train-split latent normalization stats...")
    stats_dataset = make_dataset(
        args,
        args.train_split,
        args.limit_train,
        shuffle=False,
        mean_path=mean_path,
        std_path=std_path,
        rotation_rep=args.rotation_rep,
    )
    latent_stats = compute_latent_stats(
        stats_dataset,
        latent_codec,
        batch_size=args.stats_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    args.latent_config = latent_codec.checkpoint_config()
    args.latent_config["stats"] = serializable_latent_stats(latent_stats)

    cfg = serializable_args(args)
    if dist_info["is_main"]:
        with (args.out_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2, default=str)

    wandb_run = None
    if dist_info["is_main"] and args.wandb:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY", "")
        wandb_mode = os.environ.get("WANDB_MODE", "").lower()
        if wandb_api_key and wandb_mode not in {"disabled", "dryrun", "offline"}:
            wandb.login(key=wandb_api_key, relogin=True)
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": cfg,
        }
        if args.wandb_id:
            wandb_kwargs["id"] = args.wandb_id
        if args.wandb_resume:
            wandb_kwargs["resume"] = args.wandb_resume
        wandb_run = wandb.init(**wandb_kwargs)

    rank_zero_print(dist_info, f"Device: {device}")
    rank_zero_print(
        dist_info,
        f"Distributed: enabled={dist_info['enabled']} backend={dist_info['backend']} "
        f"world_size={dist_info['world_size']}",
    )
    rank_zero_print(dist_info, f"Data: {args.data_dir}")
    rank_zero_print(
        dist_info,
        f"Word data: {args.word_data_dir} split={args.word_split} entries={len(prior_builder.entries)}",
    )
    rank_zero_print(dist_info, f"Condition field: {args.condition_field}")
    rank_zero_print(dist_info, f"Prior mode: {args.prior_mode}")
    rank_zero_print(
        dist_info,
        f"Ablation: softarranger_enabled={args.prior_mode == 'soft_arranger'} "
        f"adapter_enabled={args.adapter_enabled}",
    )
    if args.prior_mode == "soft_arranger":
        rank_zero_print(
            dist_info,
            f"Text encoder: {args.text_model_path} dim={args.text_dim} max_tokens={args.max_text_tokens}"
        )
        rank_zero_print(
            dist_info,
            "Candidates: "
            f"K={args.num_word_candidates} negatives={args.num_negative_candidates} "
            f"selection={args.candidate_selection} "
            f"max_pos_variants_per_key={args.max_positive_variants_per_key} "
            f"shuffle={args.shuffle_word_candidates}"
        )
        rank_zero_print(
            dist_info,
            "Arranger anti-collapse: "
            f"group_cov={args.group_coverage_loss_weight}@{args.group_coverage_mass} "
            f"group_entropy_peak={args.group_entropy_peak_loss_weight}@{args.group_entropy_peak_target} "
            f"attn_var={args.attention_variation_loss_weight}@{args.attention_variation_target} "
            f"prior_vel={args.prior_velocity_loss_weight} "
            f"prior_accel={args.prior_accel_loss_weight} "
            f"var_floor={args.prior_variance_floor_loss_weight}@{args.prior_variance_floor_ratio} "
            f"neg_usage={args.negative_usage_loss_weight} "
            f"attn_smooth={args.attention_smoothness_weight}"
        )
    rank_zero_print(dist_info, f"Stats data: {args.stats_data_dir}")
    rank_zero_print(
        dist_info,
        f"VAE: {args.vae_checkpoint} rotation_rep={args.rotation_rep} latent_dim={args.latent_dim}",
    )
    rank_zero_print(dist_info, f"Train samples: {len(train_dataset)}")
    if val_loader is not None:
        rank_zero_print(dist_info, f"Val samples: {len(val_loader.dataset)}")
    rank_zero_print(
        dist_info,
        "Adapter: "
        f"hidden={args.hidden_dim} content={args.content_dim} style={args.style_dim} "
        f"layers={args.num_layers} heads={args.num_heads} latent_frames={args.flow_max_frames} "
        f"content_domain_classifier={args.use_content_domain_classifier}"
    )
    if arranger is not None:
        rank_zero_print(
            dist_info,
            "Arranger: "
            f"hidden={args.arranger_hidden_dim} heads={args.arranger_num_heads} "
            f"dropout={args.arranger_dropout} max_word_latent_frames={args.max_word_latent_frames}"
        )
        rank_zero_print(
            dist_info,
            "Arranger components: "
            f"candidate_gates={not args.disable_arranger_candidate_gates} "
            f"null_memory={not args.disable_arranger_null_memory} "
            f"word_text_features={not args.disable_arranger_word_text_features} "
            f"word_motion_latents={not args.disable_arranger_word_motion_latents}"
        )
    if args.resume_from_checkpoint is not None:
        rank_zero_print(
            dist_info,
            f"Resumed from {args.resume_from_checkpoint} "
            f"(epoch={resume_epoch}, global_step={global_step}, "
            f"optimizer={'no' if args.resume_without_optimizer else 'yes'})"
        )
    rank_zero_print(dist_info, f"Output: {args.out_dir}")

    top_checkpoints = (
        load_top_k_checkpoints(args.out_dir / "checkpoints", args.save_top_k, metric_name="latent")
        if dist_info["is_main"] and args.resume_from_checkpoint is not None
        else []
    )
    start_epoch = resume_epoch + 1
    last_completed_epoch = resume_epoch
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        adapter.train()
        if not args.adapter_enabled:
            adapter.eval()
        if arranger is not None:
            arranger.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not dist_info["is_main"])
        for batch in progress:
            losses = compute_adapter_losses(
                adapter,
                batch,
                args,
                device,
                prior_builder,
                latent_codec,
                latent_stats,
                arranger=arranger,
                candidate_builder=candidate_builder,
                text_encoder=text_encoder,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
                optimizer.step()
            global_step += 1
            epoch_losses.append({key: value.detach() for key, value in losses.items()})
            progress.set_postfix(
                loss=f"{float(losses['loss'].detach().cpu()):.4f}",
                latent=f"{float(losses['latent'].detach().cpu()):.4f}",
                gain=f"{float(losses['latent_gain'].detach().cpu()):.4f}",
            )

        train_metrics = average_losses(epoch_losses)
        if "loss" in epoch_losses[0]:
            train_metrics["loss"] = float(torch.stack([item["loss"].detach().cpu() for item in epoch_losses]).mean())
        train_metrics = distributed_mean_scalars(train_metrics, device, dist_info)
        log_metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        log_metrics["epoch"] = epoch
        log_metrics["global_step"] = global_step

        if dist_info["is_main"] and val_loader is not None and epoch % args.val_every == 0:
            val_metrics = validate(
                unwrap_model(adapter),
                val_loader,
                args,
                device,
                prior_builder,
                latent_codec,
                latent_stats,
                arranger=unwrap_model(arranger) if arranger is not None else None,
                candidate_builder=candidate_builder,
                text_encoder=text_encoder,
            )
            log_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            score = float(val_metrics.get("latent", float("inf")))
            top_checkpoints, did_save = maybe_save_top_k_checkpoint(
                args.out_dir / "checkpoints",
                model_bundle,
                optimizer,
                args,
                epoch,
                global_step,
                score,
                args.save_top_k,
                top_checkpoints,
                save_checkpoint,
                metric_name="latent",
            )
            if did_save:
                rank_zero_print(
                    dist_info,
                    f"Saved validation top-{args.save_top_k} checkpoint at epoch {epoch} with latent={score:.6f}",
                )

        if wandb_run is not None:
            wandb_run.log(log_metrics, step=global_step)

        summary = " ".join(
            f"{key}={value:.5f}"
            for key, value in log_metrics.items()
            if key.startswith("train/") or key.startswith("val/")
        )
        rank_zero_print(dist_info, f"epoch={epoch} step={global_step} {summary}")

        if dist_info["is_main"] and epoch % args.save_every == 0:
            save_checkpoint(args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", model_bundle, optimizer, args, epoch, global_step)
        if dist_info["is_main"] and epoch % args.save_last_every == 0:
            save_checkpoint(args.out_dir / "checkpoints" / "last.pt", model_bundle, optimizer, args, epoch, global_step)
        last_completed_epoch = epoch
        barrier(dist_info)

    if dist_info["is_main"]:
        save_checkpoint(args.out_dir / "checkpoints" / "last.pt", model_bundle, optimizer, args, last_completed_epoch, global_step)
    elapsed = time.time() - start_time
    rank_zero_print(
        dist_info,
        f"Training complete: epoch={last_completed_epoch} global_step={global_step} elapsed_sec={elapsed:.1f}",
    )
    if wandb_run is not None:
        wandb_run.finish()
    barrier(dist_info)
    cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
