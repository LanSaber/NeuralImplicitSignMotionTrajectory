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

from flow.adapter_prior import FrozenAdapterPrior
from flow.checkpointing import load_top_k_checkpoints, maybe_save_top_k_checkpoint
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
from flow.model import (
    apply_model_size_preset,
    build_text_conditioned_model_from_args,
    call_text_model,
    count_parameters,
    make_sequence_noise,
    sample_euler_text,
    sample_heun_text,
)
from flow.latent_codec import (
    LatentMotionCodec,
    compute_latent_stats,
    serializable_latent_stats,
)
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import (
    COMPACT_DIM,
    compact_from_rotation_representation,
    feature_weight_vector,
    normalize_rotation_rep,
    rotation_rep_dim,
    rotation_rep_slices,
)
from flow.text_encoder import FrozenT5TextEncoder


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_smoke")
DEFAULT_OUT_DIR = Path("experiments/flow/text_cond_32")
DEFAULT_TEXT_MODEL_PATH = Path("deps/flan-t5-base")
DEFAULT_VAE_CHECKPOINT = Path("experiments/flow/VAE/chatsign175_temporal_vae_b32/checkpoints/best.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train text-conditioned rectified flow on 133D SMPL-X.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--text_model_path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument("--max_text_tokens", type=int, default=64)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--limit_train", type=int, default=32)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--length_multiple", type=int, default=4)
    parser.add_argument(
        "--rotation_rep",
        "--rotation-rep",
        default="axis_angle",
        choices=["axis_angle", "rot6d"],
        help="Rotation representation consumed by the flow network or frozen VAE.",
    )
    parser.add_argument(
        "--motion_space",
        "--motion-space",
        default="smplx",
        choices=["smplx", "latent"],
        help="smplx trains flow in compact 133D space; latent trains in frozen VAE latent space.",
    )
    parser.add_argument(
        "--vae_checkpoint",
        "--vae-checkpoint",
        type=Path,
        default=DEFAULT_VAE_CHECKPOINT,
        help="Frozen VAE checkpoint used when --motion_space latent.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume trainable model, optimizer, epoch, and global step from this checkpoint.",
    )
    parser.add_argument(
        "--resume_without_optimizer",
        "--resume-without-optimizer",
        action="store_true",
        help="Load only model weights when resuming.",
    )
    parser.add_argument(
        "--source_mode",
        "--source-mode",
        default="noise",
        choices=["noise", "residual", "adapter_residual"],
        help=(
            "noise keeps the original pure-noise flow source; residual starts from a "
            "dictionary-composed word prior plus noise; adapter_residual starts from a "
            "frozen soft-arranger/content-style adapter latent prior plus noise."
        ),
    )
    parser.add_argument(
        "--adapter_checkpoint",
        "--adapter-checkpoint",
        type=Path,
        default=None,
        help="Frozen adapter checkpoint used when --source_mode adapter_residual.",
    )
    parser.add_argument(
        "--word_data_dir",
        "--word-data-dir",
        type=Path,
        default=None,
        help="Flow-format word/gloss dataset used when --source_mode residual.",
    )
    parser.add_argument("--word_split", "--word-split", default="train")
    parser.add_argument(
        "--condition_field",
        "--condition-field",
        default="text",
        choices=["text", "gloss", "text_gloss", "label_word"],
        help=(
            "Manifest field used as the conditioning text for both the flow model and the "
            "frozen adapter prior. Use label_word for unsegmented-text languages (e.g. CSL) "
            "so the word-prior matcher sees whitespace-segmented words."
        ),
    )
    parser.add_argument(
        "--residual_noise_scale",
        "--residual-noise-scale",
        type=float,
        default=0.25,
        help="Scale of smooth Gaussian noise added to the composed word prior for residual flow.",
    )

    parser.add_argument(
        "--text_conditioning",
        "--text-conditioning",
        default="pooled",
        choices=["pooled", "token_prefix"],
        help="pooled keeps the original mean-pooled sentence condition; token_prefix uses T5 token embeddings as Transformer prefix tokens.",
    )
    parser.add_argument(
        "--model_size",
        "--model-size",
        type=lambda value: value.lower(),
        default="custom",
        choices=["custom", "small", "base", "large", "xl"],
        help="Preset model size. custom keeps --hidden_dim/--num_layers/--num_heads.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hand_weight", type=float, default=3.0)
    parser.add_argument("--hand_valid_floor", type=float, default=0.2)
    parser.add_argument("--latent_loss_weight", "--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose_loss_weight", type=float, default=2.0)
    parser.add_argument("--velocity_loss_weight", type=float, default=2.0)
    parser.add_argument("--accel_loss_weight", type=float, default=1.0)

    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--sample_length", type=int, default=0, help="0 uses each sampled train clip length.")
    parser.add_argument("--sampler", default="heun", choices=["euler", "heun"])
    parser.add_argument("--noise_samples", type=int, default=8, help="Noise/time draws per batch item.")
    parser.add_argument("--noise_smoothing", type=int, default=9, help="Temporal Gaussian kernel for source noise.")
    parser.add_argument("--no_random_crop", dest="no_random_crop", action="store_true", default=True)
    parser.add_argument("--random_crop", dest="no_random_crop", action="store_false")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_last_every", type=int, default=50)
    parser.add_argument("--save_top_k", "--save-top-k", dest="save_top_k", type=int, default=3)
    parser.add_argument("--val_every", type=int, default=0)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-flow")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    add_distributed_args(parser)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_weights(mask, left_valid, right_valid, args, device):
    rotation_rep = normalize_rotation_rep(getattr(args, "rotation_rep", "axis_angle"))
    dim = rotation_rep_dim(rotation_rep)
    slices = rotation_rep_slices(rotation_rep)
    weights = feature_weight_vector(args.hand_weight, device=device, rotation_rep=rotation_rep).view(1, 1, dim)
    weights = weights.expand(mask.shape[0], mask.shape[1], dim).clone()
    valid_floor = float(args.hand_valid_floor)
    left_scale = valid_floor + (1.0 - valid_floor) * left_valid.to(device).unsqueeze(-1)
    right_scale = valid_floor + (1.0 - valid_floor) * right_valid.to(device).unsqueeze(-1)
    weights[:, :, slices["left_hand"]] *= left_scale
    weights[:, :, slices["right_hand"]] *= right_scale
    weights *= mask.to(device).unsqueeze(-1)
    return weights


def weighted_mse(pred, target, weights):
    loss = (pred - target).pow(2) * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def weighted_smooth_l1(pred, target, weights):
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def diff_weights(weights, order=1):
    out = weights
    for _ in range(order):
        out = torch.minimum(out[:, 1:], out[:, :-1])
    return out


def select_condition_texts(field, texts, label_words=None, glosses=None):
    """Pick the conditioning text per item from the chosen manifest field, falling
    back to raw text when the field is empty/missing."""
    if field == "label_word" and label_words is not None:
        return [w if w else t for t, w in zip(texts, label_words)]
    if field == "gloss" and glosses is not None:
        return [g if g else t for t, g in zip(texts, glosses)]
    if field == "text_gloss" and glosses is not None:
        return [f"{t} {g}".strip() if g else t for t, g in zip(texts, glosses)]
    return list(texts)


def batch_condition_texts(batch, args):
    return select_condition_texts(
        getattr(args, "condition_field", "text"),
        batch["text"],
        batch.get("label_word"),
        batch.get("gloss"),
    )


def items_condition_texts(items, args):
    return select_condition_texts(
        getattr(args, "condition_field", "text"),
        [item["text"] for item in items],
        [item.get("label_word", "") for item in items],
        [item.get("gloss", "") for item in items],
    )


def encode_text_condition(text_encoder, texts, args, device, dtype):
    if getattr(args, "text_conditioning", "pooled") == "token_prefix":
        text_tokens, text_mask = text_encoder.encode_tokens(texts)
        return (
            text_tokens.to(device=device, dtype=dtype),
            text_mask.to(device=device, dtype=torch.bool),
        )
    return text_encoder.encode(texts).to(device=device, dtype=dtype)


def repeat_text_condition(text_condition, repeats):
    if repeats <= 1:
        return text_condition
    if isinstance(text_condition, (tuple, list)):
        text_tokens, text_mask = text_condition
        return (
            text_tokens.repeat_interleave(repeats, dim=0),
            text_mask.repeat_interleave(repeats, dim=0),
        )
    return text_condition.repeat_interleave(repeats, dim=0)


def make_flow_source(x1, mask, batch, args, device, prior_builder=None):
    source_noise = make_sequence_noise(
        x1.shape,
        device=device,
        mask=mask,
        smoothing=args.noise_smoothing,
        dtype=x1.dtype,
    )
    if args.source_mode != "residual":
        return source_noise, None
    if prior_builder is None:
        raise RuntimeError("--source_mode residual requires a WordMotionPrior.")
    prior, prior_stats = prior_builder.batch(
        batch_condition_texts(batch, args),
        batch["length"].detach().cpu().tolist(),
        max_len=x1.shape[1],
        device=device,
        dtype=x1.dtype,
    )
    x0 = prior + float(args.residual_noise_scale) * source_noise
    x0 = x0 * mask.to(device=device, dtype=x0.dtype).unsqueeze(-1)
    return x0, prior_stats


def make_latent_flow_source(
    z1,
    latent_mask,
    raw_x,
    raw_mask,
    batch,
    args,
    device,
    latent_codec,
    latent_stats,
    prior_builder=None,
    adapter_prior=None,
):
    source_noise = make_sequence_noise(
        z1.shape,
        device=device,
        mask=latent_mask,
        smoothing=args.noise_smoothing,
        dtype=z1.dtype,
    )
    if args.source_mode == "adapter_residual":
        if adapter_prior is None:
            raise RuntimeError("--source_mode adapter_residual requires a FrozenAdapterPrior.")
        prior_out = adapter_prior.build_prior(
            batch_condition_texts(batch, args),
            raw_x,
            raw_mask,
            batch["length"].detach().cpu().tolist(),
        )
        x0 = prior_out["z_prior"] + float(args.residual_noise_scale) * source_noise
        x0 = x0 * latent_mask.to(device=device, dtype=x0.dtype).unsqueeze(-1)
        return x0, prior_out["stats"], prior_out["prior_raw"]

    if args.source_mode != "residual":
        return source_noise, None, None
    if prior_builder is None:
        raise RuntimeError("--source_mode residual requires a WordMotionPrior.")

    prior_raw, prior_stats = prior_builder.batch(
        batch_condition_texts(batch, args),
        batch["length"].detach().cpu().tolist(),
        max_len=raw_x.shape[1],
        device=device,
        dtype=raw_x.dtype,
    )
    prior_raw = prior_raw * raw_mask.to(device=device, dtype=prior_raw.dtype).unsqueeze(-1)
    prior_z, _prior_latent_mask = latent_codec.encode(prior_raw, mask=raw_mask)
    prior_z = latent_codec.normalize_latent(prior_z, latent_stats)
    x0 = prior_z + float(args.residual_noise_scale) * source_noise
    x0 = x0 * latent_mask.to(device=device, dtype=x0.dtype).unsqueeze(-1)
    return x0, prior_stats, prior_raw


def compute_smplx_losses(model, text_encoder, batch, args, device, prior_builder=None):
    x1 = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    left_valid = batch["left_valid"].to(device)
    right_valid = batch["right_valid"].to(device)
    text_condition = encode_text_condition(text_encoder, batch_condition_texts(batch, args), args, device, x1.dtype)
    x0, prior_stats = make_flow_source(x1, mask, batch, args, device, prior_builder=prior_builder)

    noise_samples = max(int(args.noise_samples), 1)
    if noise_samples > 1:
        x1 = x1.repeat_interleave(noise_samples, dim=0)
        mask = mask.repeat_interleave(noise_samples, dim=0)
        left_valid = left_valid.repeat_interleave(noise_samples, dim=0)
        right_valid = right_valid.repeat_interleave(noise_samples, dim=0)
        text_condition = repeat_text_condition(text_condition, noise_samples)
        x0 = x0.repeat_interleave(noise_samples, dim=0)

    t = torch.rand(x1.shape[0], device=device)
    t_view = t.view(-1, 1, 1)
    xt = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = call_text_model(model, xt, t, text_condition, mask=mask)

    weights = loss_weights(mask, left_valid, right_valid, args, device)
    flow_loss = weighted_mse(v_pred, v_target, weights)

    x1_pred = xt + (1.0 - t_view) * v_pred
    pose_loss = weighted_smooth_l1(x1_pred, x1, weights)

    vel_loss = x1.new_tensor(0.0)
    if x1.shape[1] > 1:
        vel_weights = diff_weights(weights, order=1)
        vel_loss = weighted_smooth_l1(x1_pred[:, 1:] - x1_pred[:, :-1], x1[:, 1:] - x1[:, :-1], vel_weights)

    accel_loss = x1.new_tensor(0.0)
    if x1.shape[1] > 2:
        accel_weights = diff_weights(weights, order=2)
        pred_accel = x1_pred[:, 2:] - 2.0 * x1_pred[:, 1:-1] + x1_pred[:, :-2]
        true_accel = x1[:, 2:] - 2.0 * x1[:, 1:-1] + x1[:, :-2]
        accel_loss = weighted_smooth_l1(pred_accel, true_accel, accel_weights)

    total = (
        flow_loss
        + args.pose_loss_weight * pose_loss
        + args.velocity_loss_weight * vel_loss
        + args.accel_loss_weight * accel_loss
    )
    out = {
        "loss": total,
        "flow": flow_loss.detach(),
        "pose": pose_loss.detach(),
        "vel": vel_loss.detach(),
        "accel": accel_loss.detach(),
    }
    if prior_stats is not None:
        coverage = sum(item["coverage"] for item in prior_stats) / max(len(prior_stats), 1)
        matched = sum(item["matched_count"] for item in prior_stats) / max(len(prior_stats), 1)
        out["prior_cov"] = x1.new_tensor(float(coverage))
        out["prior_matches"] = x1.new_tensor(float(matched))
    return out


def compute_latent_losses(
    model,
    text_encoder,
    batch,
    args,
    device,
    prior_builder=None,
    adapter_prior=None,
    latent_codec=None,
    latent_stats=None,
):
    if latent_codec is None or latent_stats is None:
        raise RuntimeError("--motion_space latent requires a loaded VAE codec and latent stats.")

    raw_x = batch["motion"].to(device)
    raw_mask = batch["mask"].to(device)
    left_valid = batch["left_valid"].to(device)
    right_valid = batch["right_valid"].to(device)

    z1, latent_mask = latent_codec.encode(raw_x, mask=raw_mask)
    z1 = latent_codec.normalize_latent(z1, latent_stats)
    text_condition = encode_text_condition(text_encoder, batch_condition_texts(batch, args), args, device, z1.dtype)
    z0, prior_stats, _prior_raw = make_latent_flow_source(
        z1,
        latent_mask,
        raw_x,
        raw_mask,
        batch,
        args,
        device,
        latent_codec,
        latent_stats,
        prior_builder=prior_builder,
        adapter_prior=adapter_prior,
    )

    noise_samples = max(int(args.noise_samples), 1)
    if noise_samples > 1:
        z1 = z1.repeat_interleave(noise_samples, dim=0)
        latent_mask = latent_mask.repeat_interleave(noise_samples, dim=0)
        raw_x = raw_x.repeat_interleave(noise_samples, dim=0)
        raw_mask = raw_mask.repeat_interleave(noise_samples, dim=0)
        left_valid = left_valid.repeat_interleave(noise_samples, dim=0)
        right_valid = right_valid.repeat_interleave(noise_samples, dim=0)
        text_condition = repeat_text_condition(text_condition, noise_samples)
        z0 = z0.repeat_interleave(noise_samples, dim=0)

    t = torch.rand(z1.shape[0], device=device)
    t_view = t.view(-1, 1, 1)
    zt = (1.0 - t_view) * z0 + t_view * z1
    v_target = z1 - z0
    v_pred = call_text_model(model, zt, t, text_condition, mask=latent_mask)

    latent_weights = latent_mask.to(device=device, dtype=z1.dtype).unsqueeze(-1).expand_as(z1)
    flow_loss = weighted_mse(v_pred, v_target, latent_weights)

    z1_pred_norm = zt + (1.0 - t_view) * v_pred
    latent_recon_loss = weighted_smooth_l1(z1_pred_norm, z1, latent_weights)

    z1_pred = latent_codec.denormalize_latent(z1_pred_norm, latent_stats)
    x1_pred = latent_codec.decode(
        z1_pred,
        target_length=raw_x.shape[1],
        mask=raw_mask,
        latent_mask=latent_mask,
    )

    weights = loss_weights(raw_mask, left_valid, right_valid, args, device)
    pose_loss = weighted_smooth_l1(x1_pred, raw_x, weights)

    vel_loss = raw_x.new_tensor(0.0)
    if raw_x.shape[1] > 1:
        vel_weights = diff_weights(weights, order=1)
        vel_loss = weighted_smooth_l1(x1_pred[:, 1:] - x1_pred[:, :-1], raw_x[:, 1:] - raw_x[:, :-1], vel_weights)

    accel_loss = raw_x.new_tensor(0.0)
    if raw_x.shape[1] > 2:
        accel_weights = diff_weights(weights, order=2)
        pred_accel = x1_pred[:, 2:] - 2.0 * x1_pred[:, 1:-1] + x1_pred[:, :-2]
        true_accel = raw_x[:, 2:] - 2.0 * raw_x[:, 1:-1] + raw_x[:, :-2]
        accel_loss = weighted_smooth_l1(pred_accel, true_accel, accel_weights)

    total = (
        args.latent_loss_weight * (flow_loss + latent_recon_loss)
        + args.pose_loss_weight * pose_loss
        + args.velocity_loss_weight * vel_loss
        + args.accel_loss_weight * accel_loss
    )
    out = {
        "loss": total,
        "flow": flow_loss.detach(),
        "latent_recon": latent_recon_loss.detach(),
        "pose": pose_loss.detach(),
        "vel": vel_loss.detach(),
        "accel": accel_loss.detach(),
    }
    if prior_stats is not None:
        coverage = sum(item["coverage"] for item in prior_stats) / max(len(prior_stats), 1)
        matched = sum(item["matched_count"] for item in prior_stats) / max(len(prior_stats), 1)
        out["prior_cov"] = raw_x.new_tensor(float(coverage))
        out["prior_matches"] = raw_x.new_tensor(float(matched))
    return out


def compute_losses(
    model,
    text_encoder,
    batch,
    args,
    device,
    prior_builder=None,
    adapter_prior=None,
    latent_codec=None,
    latent_stats=None,
):
    if getattr(args, "motion_space", "smplx") == "latent":
        return compute_latent_losses(
            model,
            text_encoder,
            batch,
            args,
            device,
            prior_builder=prior_builder,
            adapter_prior=adapter_prior,
            latent_codec=latent_codec,
            latent_stats=latent_stats,
        )
    return compute_smplx_losses(model, text_encoder, batch, args, device, prior_builder=prior_builder)


def make_loader(args, split, limit, shuffle, distributed=False, world_size=1):
    dataset = UpperSMPLXFlowDataset(
        args.data_dir,
        split=split,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        length_multiple=args.length_multiple,
        random_crop=shuffle and not args.no_random_crop,
        limit=limit,
        rotation_rep=args.rotation_rep,
    )
    enough_global_samples = len(dataset) >= max(int(world_size), 1) * args.batch_size
    drop_last = shuffle and len(dataset) >= args.batch_size and (not distributed or enough_global_samples)
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_upper_smplx,
        drop_last=drop_last,
    )
    return dataset, loader, sampler


def average_dict(dicts):
    if not dicts:
        return {}
    keys = [key for key in dicts[0].keys() if key != "loss"]
    return {key: float(torch.stack([item[key].cpu() for item in dicts]).mean()) for key in keys}


@torch.no_grad()
def validate(model, text_encoder, loader, args, device, prior_builder=None, adapter_prior=None, latent_codec=None, latent_stats=None):
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        losses.append(
            compute_losses(
                model,
                text_encoder,
                batch,
                args,
                device,
                prior_builder=prior_builder,
                adapter_prior=adapter_prior,
                latent_codec=latent_codec,
                latent_stats=latent_stats,
            )
        )
    return average_dict(losses)


@torch.no_grad()
def save_samples(
    model,
    text_encoder,
    dataset,
    args,
    device,
    epoch,
    prior_builder=None,
    adapter_prior=None,
    latent_codec=None,
    latent_stats=None,
):
    if len(dataset) == 0:
        return
    sample_dir = args.out_dir / "samples" / f"epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.from_numpy(dataset.mean).to(device).view(1, 1, -1)
    std = torch.from_numpy(dataset.std).to(device).view(1, 1, -1)

    items = [dataset[index] for index in range(min(4, len(dataset)))]
    if args.sample_length > 0:
        lengths = [min(args.max_frames, int(args.sample_length)) for _ in items]
    else:
        lengths = [min(args.max_frames, int(item["length"])) for item in items]
    max_len = max(lengths)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool, device=device)
    for idx, length in enumerate(lengths):
        mask[idx, :length] = True

    text_condition = encode_text_condition(
        text_encoder,
        items_condition_texts(items, args),
        args,
        device,
        dtype=mean.dtype,
    )
    sampler = sample_heun_text if args.sampler == "heun" else sample_euler_text
    prior = None
    latent_prior = None
    coarse_motion = None

    if args.source_mode == "adapter_residual":
        if getattr(args, "motion_space", "smplx") != "latent":
            raise RuntimeError("--source_mode adapter_residual is only supported with --motion_space latent.")
        if adapter_prior is None:
            raise RuntimeError("--source_mode adapter_residual requires a FrozenAdapterPrior.")
        raw_x = torch.zeros(len(items), max_len, mean.shape[-1], dtype=mean.dtype, device=device)
        for idx, item in enumerate(items):
            item_len = min(int(item["length"]), max_len)
            raw_x[idx, :item_len] = item["motion"][:item_len].to(device=device, dtype=mean.dtype)
        prior_out = adapter_prior.build_prior(
            items_condition_texts(items, args),
            raw_x,
            mask,
            lengths,
        )
        latent_prior = prior_out["z_prior"]
        coarse_representation = (prior_out["prior_raw"] * std + mean).detach().cpu().numpy().astype(np.float32)
        coarse_motion = compact_from_rotation_representation(coarse_representation, args.rotation_rep)
    elif args.source_mode == "residual":
        if prior_builder is None:
            raise RuntimeError("--source_mode residual requires a WordMotionPrior.")
        prior, _ = prior_builder.batch(
            items_condition_texts(items, args),
            lengths,
            max_len=max_len,
            device=device,
            dtype=mean.dtype,
        )
        coarse_representation = (prior * std + mean).detach().cpu().numpy().astype(np.float32)
        coarse_motion = compact_from_rotation_representation(coarse_representation, args.rotation_rep)

    if getattr(args, "motion_space", "smplx") == "latent":
        if latent_codec is None or latent_stats is None:
            raise RuntimeError("--motion_space latent requires a loaded VAE codec and latent stats.")
        latent_mask = latent_codec.latent_mask(mask)
        if latent_prior is None and prior is not None:
            prior_z, _ = latent_codec.encode(prior, mask=mask)
            latent_prior = latent_codec.normalize_latent(prior_z, latent_stats)
        latent_samples = sampler(
            model,
            text_condition,
            (len(items), latent_mask.shape[1], latent_codec.latent_dim),
            steps=args.sample_steps,
            device=device,
            mask=latent_mask,
            noise_smoothing=args.noise_smoothing,
            x0=latent_prior,
            source_noise_scale=args.residual_noise_scale if latent_prior is not None else 1.0,
        )
        latent_samples = latent_codec.denormalize_latent(latent_samples, latent_stats)
        samples = latent_codec.decode(
            latent_samples,
            target_length=max_len,
            mask=mask,
            latent_mask=latent_mask,
        )
        samples = samples * std + mean
    else:
        samples = sampler(
            model,
            text_condition,
            (len(items), max_len, args.input_dim),
            steps=args.sample_steps,
            device=device,
            mask=mask,
            noise_smoothing=args.noise_smoothing,
            x0=prior,
            source_noise_scale=args.residual_noise_scale if prior is not None else 1.0,
        )
        samples = samples * std + mean

    for idx, item in enumerate(items):
        length = lengths[idx]
        representation = samples[idx, :length].detach().cpu().numpy().astype(np.float32)
        motion = compact_from_rotation_representation(representation, args.rotation_rep)
        target_representation = item["motion"][: item["length"]].numpy().astype(np.float32)
        target_representation = target_representation * dataset.std[None] + dataset.mean[None]
        target = compact_from_rotation_representation(target_representation, args.rotation_rep)
        payload = {
            "motion": motion,
            "representation": representation,
            "rotation_rep": args.rotation_rep,
            "text": item["text"],
            "name": item["name"],
            "target_motion": target.astype(np.float32),
            "target_representation": target_representation.astype(np.float32),
        }
        if coarse_motion is not None:
            payload["coarse_motion"] = coarse_motion[idx, :length]
        np.savez_compressed(sample_dir / f"sample_{idx:02d}.npz", **payload)


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def checkpoint_payload(model, optimizer, args, epoch, global_step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": {
            "text_conditioning": args.text_conditioning,
            "motion_space": args.motion_space,
            "rotation_rep": args.rotation_rep,
            "model_size": args.model_size,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "max_frames": args.flow_max_frames,
            "raw_max_frames": args.max_frames,
            "text_dim": args.text_dim,
        },
        "text_config": {
            "text_model_path": str(args.text_model_path),
            "text_dim": args.text_dim,
            "max_text_tokens": args.max_text_tokens,
            "pooling": "mean" if args.text_conditioning == "pooled" else "tokens",
            "conditioning": args.text_conditioning,
            "condition_field": args.condition_field,
            "frozen": True,
        },
        "data_config": {
            "data_dir": str(args.data_dir),
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "length_multiple": args.length_multiple,
            "rotation_rep": args.rotation_rep,
        },
        "source_config": {
            "source_mode": args.source_mode,
            "word_data_dir": str(args.word_data_dir) if args.word_data_dir is not None else None,
            "word_split": args.word_split,
            "condition_field": args.condition_field,
            "residual_noise_scale": args.residual_noise_scale,
            "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint is not None else None,
            "adapter_prior_config": getattr(args, "adapter_prior_config", None),
        },
        "latent_config": getattr(args, "latent_config", None),
        "args": serializable_args(args),
    }


def save_checkpoint(path, model, optimizer, args, epoch, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(unwrap_model(model), optimizer, args, epoch, global_step), path)


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


def resume_training_state(path, model, optimizer, device, load_optimizer=True):
    ckpt = load_checkpoint(path, map_location="cpu")
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    if load_optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def main():
    args = parse_args()
    apply_model_size_preset(args)
    args.rotation_rep = normalize_rotation_rep(args.rotation_rep)
    args.adapter_prior_config = None
    if args.source_mode == "adapter_residual":
        if args.motion_space != "latent":
            raise ValueError("--source_mode adapter_residual requires --motion_space latent.")
        if args.adapter_checkpoint is None:
            raise ValueError("--source_mode adapter_residual requires --adapter_checkpoint.")
    dist_info = setup_distributed(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + int(dist_info["rank"]))
    device = resolve_distributed_device(args.device, dist_info)

    latent_codec = None
    latent_stats = None
    if args.motion_space == "latent":
        latent_codec = LatentMotionCodec(args.vae_checkpoint, device=device)
        args.rotation_rep = latent_codec.rotation_rep

    train_dataset, train_loader, train_sampler = make_loader(
        args,
        args.train_split,
        args.limit_train,
        shuffle=True,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found under {args.data_dir}")

    prior_builder = None
    adapter_prior = None
    if args.source_mode == "residual":
        if args.word_data_dir is None:
            raise ValueError("--source_mode residual requires --word_data_dir")
        prior_builder = WordMotionPrior(
            args.word_data_dir,
            split=args.word_split,
            target_mean=train_dataset.mean,
            target_std=train_dataset.std,
            rotation_rep=args.rotation_rep,
        )
    elif args.source_mode == "adapter_residual":
        adapter_prior = FrozenAdapterPrior.from_checkpoint(
            args.adapter_checkpoint,
            device=device,
            latent_codec=latent_codec,
            target_mean=train_dataset.mean,
            target_std=train_dataset.std,
            word_data_dir=args.word_data_dir,
            word_split=args.word_split,
            text_model_path=args.text_model_path,
            max_text_tokens=args.max_text_tokens,
            candidate_seed=args.seed,
        )
        args.adapter_prior_config = adapter_prior.checkpoint_config()

    val_loader = None
    val_manifest = args.data_dir / "meta" / f"manifest_{args.val_split}.jsonl"
    if dist_info["is_main"] and args.val_every > 0 and val_manifest.is_file():
        val_dataset, maybe_val_loader, _ = make_loader(args, args.val_split, args.limit_val, shuffle=False)
        if len(val_dataset) > 0:
            val_loader = maybe_val_loader

    args.input_dim = rotation_rep_dim(args.rotation_rep)
    args.flow_max_frames = args.max_frames
    args.latent_config = None
    if args.motion_space == "latent":
        if args.max_frames > latent_codec.max_frames:
            raise ValueError(
                f"--max_frames {args.max_frames} exceeds VAE max_frames={latent_codec.max_frames}; "
                "train a larger VAE or lower --max_frames."
            )
        args.input_dim = latent_codec.latent_dim
        args.flow_max_frames = latent_codec.latent_length(args.max_frames)
        if adapter_prior is not None:
            latent_stats = adapter_prior.latent_stats
        else:
            stats_dataset = UpperSMPLXFlowDataset(
                args.data_dir,
                split=args.train_split,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                length_multiple=args.length_multiple,
                random_crop=False,
                limit=args.limit_train,
                rotation_rep=args.rotation_rep,
            )
            latent_stats = compute_latent_stats(
                stats_dataset,
                latent_codec,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        args.latent_config = latent_codec.checkpoint_config()
        args.latent_config["stats"] = serializable_latent_stats(latent_stats)

    text_encoder = FrozenT5TextEncoder(
        args.text_model_path,
        device=device,
        max_length=args.max_text_tokens,
        local_files_only=True,
        cache=True,
    )
    args.text_dim = text_encoder.text_dim
    model = build_text_conditioned_model_from_args(args).to(device)
    model = wrap_model(model, dist_info, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_from_checkpoint}")
        resume_epoch, global_step = resume_training_state(
            args.resume_from_checkpoint,
            model,
            optimizer,
            device,
            load_optimizer=not args.resume_without_optimizer,
        )

    cfg = serializable_args(args)
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

    if dist_info["is_main"]:
        with (args.out_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2, default=str)

    rank_zero_print(dist_info, f"Device: {device}")
    rank_zero_print(
        dist_info,
        f"Distributed: enabled={dist_info['enabled']} backend={dist_info['backend']} "
        f"world_size={dist_info['world_size']}",
    )
    rank_zero_print(dist_info, f"Train samples: {len(train_dataset)}")
    rank_zero_print(dist_info, f"Rotation representation: {args.rotation_rep} input_dim={args.input_dim}")
    if prior_builder is not None:
        rank_zero_print(
            dist_info,
            f"Residual source: word_data_dir={args.word_data_dir} split={args.word_split} "
            f"entries={len(prior_builder.entries)} noise_scale={args.residual_noise_scale}",
        )
    if adapter_prior is not None:
        rank_zero_print(
            dist_info,
            f"Adapter residual source: adapter={args.adapter_checkpoint} "
            f"prior_mode={adapter_prior.prior_mode} word_data_dir={adapter_prior.prior_builder.data_dir} "
            f"split={adapter_prior.prior_builder.split} entries={len(adapter_prior.prior_builder.entries)} "
            f"noise_scale={args.residual_noise_scale}",
        )
    rank_zero_print(dist_info, f"Text model: {args.text_model_path} (dim={args.text_dim}, frozen=True)")
    if latent_codec is not None:
        rank_zero_print(
            dist_info,
            f"Latent VAE: {args.vae_checkpoint} latent_dim={latent_codec.latent_dim} "
            f"downsample={latent_codec.downsample_factor} flow_max_frames={args.flow_max_frames}",
        )
    rank_zero_print(
        dist_info,
        f"Flow model: space={args.motion_space} conditioning={args.text_conditioning} size={args.model_size} "
        f"rotation_rep={args.rotation_rep} input_dim={args.input_dim} max_frames={args.flow_max_frames} "
        f"hidden={args.hidden_dim} layers={args.num_layers} heads={args.num_heads} "
        f"params={count_parameters(unwrap_model(model)) / 1e6:.2f}M",
    )
    if args.resume_from_checkpoint is not None:
        rank_zero_print(
            dist_info,
            f"Resumed from: {args.resume_from_checkpoint} "
            f"(epoch={resume_epoch}, global_step={global_step}, "
            f"optimizer={'no' if args.resume_without_optimizer else 'yes'})",
        )
    rank_zero_print(dist_info, f"Output: {args.out_dir}")

    best_val = float("inf")
    top_checkpoints = (
        load_top_k_checkpoints(args.out_dir / "checkpoints", args.save_top_k, metric_name="flow")
        if dist_info["is_main"] and args.resume_from_checkpoint is not None
        else []
    )
    if top_checkpoints:
        best_val = top_checkpoints[0]["score"]
        rank_zero_print(
            dist_info,
            f"Recovered validation top-{args.save_top_k} metadata from existing checkpoints "
            f"(best_flow={best_val:.5f})",
        )
    start = time.time()
    start_epoch = resume_epoch + 1
    last_completed_epoch = resume_epoch
    if start_epoch > args.epochs:
        rank_zero_print(
            dist_info,
            f"Checkpoint epoch {resume_epoch} is already >= requested epochs {args.epochs}; no training steps run.",
        )
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not dist_info["is_main"])
        for batch in pbar:
            losses = compute_losses(
                model,
                text_encoder,
                batch,
                args,
                device,
                prior_builder=prior_builder,
                adapter_prior=adapter_prior,
                latent_codec=latent_codec,
                latent_stats=latent_stats,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            train_losses.append({key: value.detach() for key, value in losses.items()})
            pbar.set_postfix(loss=float(losses["loss"].detach().cpu()))

        train_avg = average_dict(train_losses)
        train_avg["loss"] = float(torch.stack([item["loss"].cpu() for item in train_losses]).mean())
        train_avg = distributed_mean_scalars(train_avg, device, dist_info)
        log = {f"train/{key}": value for key, value in train_avg.items()}

        val_avg = None
        val_due = val_loader is not None and epoch % args.val_every == 0
        if val_due and dist_info["is_main"]:
            val_avg = validate(
                unwrap_model(model),
                text_encoder,
                val_loader,
                args,
                device,
                prior_builder=prior_builder,
                adapter_prior=adapter_prior,
                latent_codec=latent_codec,
                latent_stats=latent_stats,
            )
            log.update({f"val/{key}": value for key, value in val_avg.items()})
            top_checkpoints, did_save_top = maybe_save_top_k_checkpoint(
                args.out_dir / "checkpoints",
                model,
                optimizer,
                args,
                epoch,
                global_step,
                val_avg.get("flow", float("inf")),
                args.save_top_k,
                top_checkpoints,
                save_checkpoint,
                metric_name="flow",
            )
            if top_checkpoints:
                best_val = top_checkpoints[0]["score"]
            if did_save_top:
                rank_zero_print(
                    dist_info,
                    f"saved validation top-{args.save_top_k} checkpoint candidate "
                    f"epoch={epoch} val_flow={val_avg.get('flow', float('inf')):.5f} best_flow={best_val:.5f}",
                )

        if dist_info["is_main"] and args.save_last_every > 0 and epoch % args.save_last_every == 0:
            save_checkpoint(args.out_dir / "checkpoints" / "last.pt", model, optimizer, args, epoch, global_step)
        if dist_info["is_main"] and args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                args,
                epoch,
                global_step,
            )
        if dist_info["is_main"] and args.sample_every > 0 and epoch % args.sample_every == 0:
            save_samples(
                unwrap_model(model),
                text_encoder,
                train_dataset,
                args,
                device,
                epoch,
                prior_builder=prior_builder,
                adapter_prior=adapter_prior,
                latent_codec=latent_codec,
                latent_stats=latent_stats,
            )
        barrier(dist_info)

        elapsed = (time.time() - start) / 60.0
        msg = f"epoch={epoch} elapsed_min={elapsed:.1f} train_loss={train_avg['loss']:.5f}"
        msg += " " + " ".join(
            f"train_{key}={value:.5f}" for key, value in train_avg.items() if key != "loss"
        )
        if val_avg:
            msg += " " + " ".join(f"val_{k}={v:.5f}" for k, v in val_avg.items())
        rank_zero_print(dist_info, msg)
        if wandb_run is not None:
            import wandb

            wandb.log(log, step=global_step)
        last_completed_epoch = epoch

    if dist_info["is_main"]:
        save_checkpoint(
            args.out_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            args,
            last_completed_epoch,
            global_step,
        )
        if wandb_run is not None:
            wandb_run.finish()
    barrier(dist_info)
    cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
