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

from flow.VAE.model import TemporalSMPLXVAE, count_parameters
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
from flow.smplx_features import (
    COMPACT_DIM,
    compact_from_rotation_representation,
    compact_rot6d_to_axis_angle_torch,
    feature_weight_vector,
    normalize_rotation_rep,
    ROTATION_REP_AXIS_ANGLE,
    ROTATION_REP_ROT6D,
    rotation_rep_dim,
    rotation_rep_slices,
    rotation_rep_stats_paths,
)


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175")
DEFAULT_OUT_DIR = Path("experiments/flow/VAE/chatsign175_vae")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal VAE for compact 133D SMPL-X signing motion.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--length_multiple", type=int, default=4)
    parser.add_argument(
        "--rotation_rep",
        "--rotation-rep",
        default="axis_angle",
        choices=["axis_angle", "rot6d"],
        help="Rotation representation consumed by the VAE.",
    )
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--no_random_crop", dest="random_crop", action="store_false")
    parser.set_defaults(random_crop=False)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--downsample_factor", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hand_weight", type=float, default=5.0)
    parser.add_argument("--jaw_weight", type=float, default=2.0)
    parser.add_argument("--expression_weight", type=float, default=2.0)
    parser.add_argument("--hand_valid_floor", type=float, default=0.2)
    parser.add_argument("--pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--velocity_loss_weight", type=float, default=1.0)
    parser.add_argument("--accel_loss_weight", type=float, default=0.5)
    parser.add_argument("--jerk_loss_weight", "--jerk-loss-weight", type=float, default=0.25)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--kl_start_epoch", type=int, default=200)
    parser.add_argument("--kl_warmup_epochs", type=int, default=300)

    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_last_every", type=int, default=10)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", "--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--resume_without_optimizer", "--resume-without-optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-flow-vae")
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


def make_loader(args, split, limit, shuffle, pin_memory=False, distributed=False, world_size=1):
    dataset = UpperSMPLXFlowDataset(
        args.data_dir,
        split=split,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        length_multiple=args.length_multiple,
        random_crop=shuffle and args.random_crop,
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
        pin_memory=pin_memory,
        collate_fn=collate_upper_smplx,
        drop_last=drop_last,
    )
    return dataset, loader, sampler


def feature_weights_for_rep(mask, left_valid, right_valid, args, device, rotation_rep):
    rotation_rep = normalize_rotation_rep(rotation_rep)
    dim = rotation_rep_dim(rotation_rep)
    slices = rotation_rep_slices(rotation_rep)
    weights = feature_weight_vector(args.hand_weight, device=device, rotation_rep=rotation_rep).view(1, 1, dim)
    weights = weights.expand(mask.shape[0], mask.shape[1], dim).clone()
    weights[:, :, slices["jaw"]] *= float(args.jaw_weight)
    weights[:, :, slices["expression"]] *= float(args.expression_weight)

    valid_floor = float(args.hand_valid_floor)
    left_scale = valid_floor + (1.0 - valid_floor) * left_valid.to(device).unsqueeze(-1)
    right_scale = valid_floor + (1.0 - valid_floor) * right_valid.to(device).unsqueeze(-1)
    weights[:, :, slices["left_hand"]] *= left_scale
    weights[:, :, slices["right_hand"]] *= right_scale
    weights *= mask.to(device).unsqueeze(-1)
    return weights


def feature_weights(mask, left_valid, right_valid, args, device):
    return feature_weights_for_rep(mask, left_valid, right_valid, args, device, args.rotation_rep)


def weighted_smooth_l1(pred, target, weights):
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def diff_weights(weights, order=1):
    out = weights
    for _ in range(order):
        out = torch.minimum(out[:, 1:], out[:, :-1])
    return out


def third_difference(x):
    return x[:, 3:] - 3.0 * x[:, 2:-1] + 3.0 * x[:, 1:-2] - x[:, :-3]


def kl_weight_for_epoch(args, epoch):
    if args.kl_weight <= 0:
        return 0.0
    if epoch < args.kl_start_epoch:
        return 0.0
    warmup = max(int(args.kl_warmup_epochs), 1)
    progress = min(max((epoch - args.kl_start_epoch + 1) / warmup, 0.0), 1.0)
    return float(args.kl_weight) * progress


def kl_loss(mu, logvar, latent_mask):
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if latent_mask is not None:
        kl = kl * latent_mask.unsqueeze(-1).to(kl.dtype)
        denom = latent_mask.sum().clamp_min(1).to(kl.dtype) * kl.shape[-1]
    else:
        denom = torch.tensor(kl.numel(), device=kl.device, dtype=kl.dtype)
    return kl.sum() / denom


def build_axis_angle_metric_context(args, dataset, device):
    """Prepare stats for validation metrics in normalized axis-angle space."""

    if args.rotation_rep == ROTATION_REP_AXIS_ANGLE:
        return None

    if args.rotation_rep != ROTATION_REP_ROT6D:
        raise ValueError(f"Unsupported rotation representation for axis-angle metric: {args.rotation_rep}")

    axis_mean_path, axis_std_path = rotation_rep_stats_paths(args.data_dir, ROTATION_REP_AXIS_ANGLE)
    if not axis_mean_path.is_file() or not axis_std_path.is_file():
        raise FileNotFoundError(
            "Axis-angle mean/std are required for comparable rot6d validation metrics: "
            f"{axis_mean_path}, {axis_std_path}"
        )

    rep_mean = torch.as_tensor(dataset.mean, dtype=torch.float32, device=device).view(1, 1, -1)
    rep_std = torch.as_tensor(dataset.std, dtype=torch.float32, device=device).view(1, 1, -1)
    axis_mean = torch.as_tensor(np.load(axis_mean_path).astype(np.float32), dtype=torch.float32, device=device).view(1, 1, -1)
    axis_std = torch.as_tensor(np.load(axis_std_path).astype(np.float32), dtype=torch.float32, device=device).view(1, 1, -1)
    if axis_mean.shape[-1] != COMPACT_DIM or axis_std.shape[-1] != COMPACT_DIM:
        raise ValueError(
            f"Axis-angle stats must have dim {COMPACT_DIM}, got mean={axis_mean.shape[-1]} std={axis_std.shape[-1]}"
        )
    return {
        "rep_mean": rep_mean,
        "rep_std": rep_std,
        "axis_mean": axis_mean,
        "axis_std": axis_std,
    }


def axis_angle_validation_pose_loss(recon, target, mask, left_valid, right_valid, args, device, context):
    if args.rotation_rep == ROTATION_REP_AXIS_ANGLE:
        recon_axis_norm = recon
        target_axis_norm = target
    else:
        if context is None:
            raise RuntimeError("rot6d validation requires an axis-angle metric context.")
        recon_rep = recon * context["rep_std"].to(dtype=recon.dtype) + context["rep_mean"].to(dtype=recon.dtype)
        target_rep = target * context["rep_std"].to(dtype=target.dtype) + context["rep_mean"].to(dtype=target.dtype)
        recon_axis = compact_rot6d_to_axis_angle_torch(recon_rep)
        target_axis = compact_rot6d_to_axis_angle_torch(target_rep)
        axis_mean = context["axis_mean"].to(dtype=recon_axis.dtype)
        axis_std = context["axis_std"].to(dtype=recon_axis.dtype)
        recon_axis_norm = (recon_axis - axis_mean) / axis_std.clamp_min(1e-8)
        target_axis_norm = (target_axis - axis_mean) / axis_std.clamp_min(1e-8)

    weights = feature_weights_for_rep(mask, left_valid, right_valid, args, device, ROTATION_REP_AXIS_ANGLE)
    return weighted_smooth_l1(recon_axis_norm, target_axis_norm, weights)


def compute_losses(model, batch, args, device, epoch, sample=True, axis_angle_metric_context=None):
    x = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    left_valid = batch["left_valid"].to(device)
    right_valid = batch["right_valid"].to(device)
    out = model(x, mask=mask, sample=sample)
    recon = out["recon"]

    weights = feature_weights(mask, left_valid, right_valid, args, device)
    pose = weighted_smooth_l1(recon, x, weights)

    vel = x.new_tensor(0.0)
    if x.shape[1] > 1:
        vel_weights = diff_weights(weights, order=1)
        vel = weighted_smooth_l1(recon[:, 1:] - recon[:, :-1], x[:, 1:] - x[:, :-1], vel_weights)

    accel = x.new_tensor(0.0)
    if x.shape[1] > 2:
        accel_weights = diff_weights(weights, order=2)
        recon_accel = recon[:, 2:] - 2.0 * recon[:, 1:-1] + recon[:, :-2]
        target_accel = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
        accel = weighted_smooth_l1(recon_accel, target_accel, accel_weights)

    jerk = x.new_tensor(0.0)
    if x.shape[1] > 3:
        jerk_weights = diff_weights(weights, order=3)
        jerk = weighted_smooth_l1(third_difference(recon), third_difference(x), jerk_weights)

    kl = kl_loss(out["mu"], out["logvar"], out["latent_mask"])
    beta = x.new_tensor(kl_weight_for_epoch(args, epoch))
    total = (
        args.pose_loss_weight * pose
        + args.velocity_loss_weight * vel
        + args.accel_loss_weight * accel
        + args.jerk_loss_weight * jerk
        + beta * kl
    )
    metrics = {
        "loss": total,
        "recon": pose.detach(),
        "vel": vel.detach(),
        "accel": accel.detach(),
        "jerk": jerk.detach(),
        "kl": kl.detach(),
        "kl_beta": beta.detach(),
    }
    if axis_angle_metric_context is not None or args.rotation_rep == ROTATION_REP_AXIS_ANGLE:
        metrics["recon_axis_angle"] = axis_angle_validation_pose_loss(
            recon,
            x,
            mask,
            left_valid,
            right_valid,
            args,
            device,
            axis_angle_metric_context,
        ).detach()
    return metrics


def average_losses(items):
    if not items:
        return {}
    keys = sorted(key for key in items[0] if key != "loss")
    return {key: float(torch.stack([item[key].detach().cpu() for item in items]).mean()) for key in keys}


@torch.no_grad()
def validate(model, loader, args, device, epoch, axis_angle_metric_context=None):
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        losses.append(
            compute_losses(
                model,
                batch,
                args,
                device,
                epoch,
                sample=False,
                axis_angle_metric_context=axis_angle_metric_context,
            )
        )
    return average_losses(losses)


@torch.no_grad()
def save_recon_samples(model, dataset, args, device, epoch):
    if len(dataset) == 0:
        return
    sample_dir = args.out_dir / "samples" / f"epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    items = [dataset[index] for index in range(min(4, len(dataset)))]
    batch = collate_upper_smplx(items)
    x = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    out = model(x, mask=mask, sample=False)
    recon = out["recon"].detach().cpu().numpy()
    target = x.detach().cpu().numpy()
    mean = dataset.mean[None, None]
    std = dataset.std[None, None]
    recon = recon * std + mean
    target = target * std + mean

    for idx, item in enumerate(items):
        length = int(item["length"])
        recon_rep = recon[idx, :length].astype(np.float32)
        target_rep = target[idx, :length].astype(np.float32)
        recon_motion = compact_from_rotation_representation(recon_rep, args.rotation_rep)
        target_motion = compact_from_rotation_representation(target_rep, args.rotation_rep)
        np.savez_compressed(
            sample_dir / f"recon_{idx:02d}.npz",
            motion=recon_motion.astype(np.float32),
            target_motion=target_motion.astype(np.float32),
            representation=recon_rep,
            target_representation=target_rep,
            rotation_rep=args.rotation_rep,
            text=item["text"],
            name=item["name"],
            length=length,
        )


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def checkpoint_payload(model, optimizer, args, epoch, global_step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_config": {
            "input_dim": rotation_rep_dim(args.rotation_rep),
            "rotation_rep": args.rotation_rep,
            "representation_dim": rotation_rep_dim(args.rotation_rep),
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "max_frames": args.max_frames,
            "downsample_factor": args.downsample_factor,
        },
        "data_config": {
            "data_dir": str(args.data_dir),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "length_multiple": args.length_multiple,
            "normalized": True,
            "rotation_rep": args.rotation_rep,
        },
        "loss_config": {
            "hand_weight": args.hand_weight,
            "jaw_weight": args.jaw_weight,
            "expression_weight": args.expression_weight,
            "pose_loss_weight": args.pose_loss_weight,
            "velocity_loss_weight": args.velocity_loss_weight,
            "accel_loss_weight": args.accel_loss_weight,
            "jerk_loss_weight": args.jerk_loss_weight,
            "kl_weight": args.kl_weight,
            "kl_start_epoch": args.kl_start_epoch,
            "kl_warmup_epochs": args.kl_warmup_epochs,
        },
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
    args.rotation_rep = normalize_rotation_rep(args.rotation_rep)
    args.validation_score_metric = "recon_axis_angle" if args.rotation_rep == ROTATION_REP_ROT6D else "recon"
    dist_info = setup_distributed(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_distributed_device(args.device, dist_info)
    set_seed(args.seed + int(dist_info["rank"]))
    pin_memory = device.type == "cuda"

    train_dataset, train_loader, train_sampler = make_loader(
        args,
        args.train_split,
        args.limit_train,
        shuffle=True,
        pin_memory=pin_memory,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found under {args.data_dir}")
    val_dataset = None
    val_loader = None
    val_manifest = args.data_dir / "meta" / f"manifest_{args.val_split}.jsonl"
    if dist_info["is_main"] and args.val_every > 0 and val_manifest.is_file():
        val_dataset, val_loader, _ = make_loader(
            args,
            args.val_split,
            args.limit_val,
            shuffle=False,
            pin_memory=pin_memory,
        )
    axis_angle_metric_context = build_axis_angle_metric_context(args, train_dataset, device)

    model = TemporalSMPLXVAE(
        input_dim=rotation_rep_dim(args.rotation_rep),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_frames=args.max_frames,
        downsample_factor=args.downsample_factor,
    ).to(device)
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
    rank_zero_print(dist_info, f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        rank_zero_print(dist_info, f"Val samples: {len(val_dataset)}")
    rank_zero_print(
        dist_info,
        "VAE: "
        f"hidden={args.hidden_dim} latent={args.latent_dim} layers={args.num_layers} "
        f"heads={args.num_heads} downsample={args.downsample_factor} "
        f"params={count_parameters(unwrap_model(model)) / 1e6:.2f}M"
    )
    rank_zero_print(dist_info, f"Rotation rep: {args.rotation_rep}")
    rank_zero_print(dist_info, f"Validation checkpoint metric: {args.validation_score_metric}")
    if args.resume_from_checkpoint is not None:
        rank_zero_print(
            dist_info,
            f"Resumed from {args.resume_from_checkpoint} "
            f"(epoch={resume_epoch}, global_step={global_step}, "
            f"optimizer={'no' if args.resume_without_optimizer else 'yes'})"
        )
    rank_zero_print(dist_info, f"Output: {args.out_dir}")

    top_checkpoints = (
        load_top_k_checkpoints(args.out_dir / "checkpoints", args.save_top_k, metric_name=args.validation_score_metric)
        if dist_info["is_main"] and args.resume_from_checkpoint is not None
        else []
    )
    start_epoch = resume_epoch + 1
    last_completed_epoch = resume_epoch
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not dist_info["is_main"])
        for batch in progress:
            losses = compute_losses(model, batch, args, device, epoch)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            epoch_losses.append({key: value.detach() for key, value in losses.items()})
            progress.set_postfix(
                loss=f"{float(losses['loss'].detach().cpu()):.4f}",
                recon=f"{float(losses['recon'].detach().cpu()):.4f}",
                kl_beta=f"{float(losses['kl_beta'].detach().cpu()):.2e}",
            )

        train_metrics = average_losses(epoch_losses)
        if epoch_losses:
            train_metrics["loss"] = float(torch.stack([item["loss"].cpu() for item in epoch_losses]).mean())
        train_metrics = distributed_mean_scalars(train_metrics, device, dist_info)
        log_metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        log_metrics["epoch"] = epoch
        log_metrics["global_step"] = global_step

        if dist_info["is_main"] and val_loader is not None and epoch % args.val_every == 0:
            val_metrics = validate(
                unwrap_model(model),
                val_loader,
                args,
                device,
                epoch,
                axis_angle_metric_context=axis_angle_metric_context,
            )
            log_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            score = float(val_metrics.get(args.validation_score_metric, val_metrics.get("recon", float("inf"))))
            top_checkpoints, did_save = maybe_save_top_k_checkpoint(
                args.out_dir / "checkpoints",
                model,
                optimizer,
                args,
                epoch,
                global_step,
                score,
                args.save_top_k,
                top_checkpoints,
                save_checkpoint,
                metric_name=args.validation_score_metric,
            )
            if did_save:
                rank_zero_print(
                    dist_info,
                    f"Saved validation top-{args.save_top_k} checkpoint at epoch {epoch} "
                    f"with {args.validation_score_metric}={score:.6f}"
                )

        if dist_info["is_main"] and wandb_run is not None:
            wandb_run.log(log_metrics, step=global_step)

        if dist_info["is_main"]:
            summary = " ".join(
                f"{key}={value:.5f}"
                for key, value in log_metrics.items()
                if key.startswith("train/") or key.startswith("val/")
            )
            print(f"epoch={epoch} step={global_step} {summary}")

        if dist_info["is_main"] and epoch % args.sample_every == 0:
            save_recon_samples(unwrap_model(model), val_dataset or train_dataset, args, device, epoch)
        if dist_info["is_main"] and epoch % args.save_every == 0:
            save_checkpoint(args.out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", model, optimizer, args, epoch, global_step)
        if dist_info["is_main"] and epoch % args.save_last_every == 0:
            save_checkpoint(args.out_dir / "checkpoints" / "last.pt", model, optimizer, args, epoch, global_step)
        barrier(dist_info)
        last_completed_epoch = epoch

    if dist_info["is_main"]:
        save_checkpoint(args.out_dir / "checkpoints" / "last.pt", model, optimizer, args, last_completed_epoch, global_step)
    elapsed = time.time() - start_time
    rank_zero_print(dist_info, f"Training complete: epoch={last_completed_epoch} global_step={global_step} elapsed_sec={elapsed:.1f}")
    if dist_info["is_main"] and wandb_run is not None:
        wandb_run.finish()
    barrier(dist_info)
    cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
