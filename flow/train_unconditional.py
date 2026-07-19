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

from flow.checkpointing import maybe_save_top_k_checkpoint
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
from flow.model import build_model_from_args, make_sequence_noise, sample_euler, sample_heun
from flow.smplx_features import (
    COMPACT_DIM,
    COMPACT_LEFT_HAND,
    COMPACT_RIGHT_HAND,
    feature_weight_vector,
)


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx")
DEFAULT_OUT_DIR = Path("experiments/flow/upper_smplx_uncond")


def parse_args():
    parser = argparse.ArgumentParser(description="Train unconditional rectified flow on 133D SMPL-X.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=64)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--length_multiple", type=int, default=4)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hand_weight", type=float, default=3.0)
    parser.add_argument("--hand_valid_floor", type=float, default=0.2)
    parser.add_argument("--pose_loss_weight", type=float, default=0.5)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.1)
    parser.add_argument("--accel_loss_weight", type=float, default=0.05)

    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--sample_length", type=int, default=0, help="0 uses the first train sample length.")
    parser.add_argument("--sampler", default="euler", choices=["euler", "heun"])
    parser.add_argument("--noise_samples", type=int, default=1, help="Noise/time draws per batch item.")
    parser.add_argument("--noise_smoothing", type=int, default=0, help="Temporal Gaussian kernel for source noise.")
    parser.add_argument("--no_random_crop", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--save_last_every", type=int, default=1)
    parser.add_argument("--save_top_k", "--save-top-k", dest="save_top_k", type=int, default=3)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-flow")
    parser.add_argument("--wandb_run_name", default=None)
    add_distributed_args(parser)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_weights(mask, left_valid, right_valid, args, device):
    weights = feature_weight_vector(args.hand_weight, device=device).view(1, 1, COMPACT_DIM)
    weights = weights.expand(mask.shape[0], mask.shape[1], COMPACT_DIM).clone()
    valid_floor = float(args.hand_valid_floor)
    left_scale = valid_floor + (1.0 - valid_floor) * left_valid.to(device).unsqueeze(-1)
    right_scale = valid_floor + (1.0 - valid_floor) * right_valid.to(device).unsqueeze(-1)
    weights[:, :, COMPACT_LEFT_HAND] *= left_scale
    weights[:, :, COMPACT_RIGHT_HAND] *= right_scale
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


def compute_losses(model, batch, args, device):
    x1 = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    left_valid = batch["left_valid"].to(device)
    right_valid = batch["right_valid"].to(device)

    noise_samples = max(int(args.noise_samples), 1)
    if noise_samples > 1:
        x1 = x1.repeat_interleave(noise_samples, dim=0)
        mask = mask.repeat_interleave(noise_samples, dim=0)
        left_valid = left_valid.repeat_interleave(noise_samples, dim=0)
        right_valid = right_valid.repeat_interleave(noise_samples, dim=0)

    x0 = make_sequence_noise(
        x1.shape,
        device=device,
        mask=mask,
        smoothing=args.noise_smoothing,
        dtype=x1.dtype,
    )
    t = torch.rand(x1.shape[0], device=device)
    t_view = t.view(-1, 1, 1)
    xt = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = model(xt, t, mask=mask)

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
    return {
        "loss": total,
        "flow": flow_loss.detach(),
        "pose": pose_loss.detach(),
        "vel": vel_loss.detach(),
        "accel": accel_loss.detach(),
    }


def make_loader(args, split, limit, shuffle, distributed=False, world_size=1):
    dataset = UpperSMPLXFlowDataset(
        args.data_dir,
        split=split,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        length_multiple=args.length_multiple,
        random_crop=shuffle and not args.no_random_crop,
        limit=limit,
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
def validate(model, loader, args, device):
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        losses.append(compute_losses(model, batch, args, device))
    return average_dict(losses)


@torch.no_grad()
def save_samples(model, dataset, args, device, epoch):
    sample_dir = args.out_dir / "samples" / f"epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.from_numpy(dataset.mean).to(device).view(1, 1, -1)
    std = torch.from_numpy(dataset.std).to(device).view(1, 1, -1)
    if args.sample_length > 0:
        length = min(args.max_frames, args.sample_length)
    else:
        length = min(args.max_frames, int(dataset[0]["length"]))
    mask = torch.ones(4, length, dtype=torch.bool, device=device)
    sampler = sample_heun if args.sampler == "heun" else sample_euler
    samples = sampler(
        model,
        (4, length, COMPACT_DIM),
        steps=args.sample_steps,
        device=device,
        mask=mask,
        noise_smoothing=args.noise_smoothing,
    )
    samples = samples * std + mean
    for idx in range(samples.shape[0]):
            np.savez_compressed(
                sample_dir / f"sample_{idx:02d}.npz",
                motion=samples[idx].detach().cpu().numpy().astype(np.float32),
            )


def checkpoint_payload(model, optimizer, args, epoch, global_step):
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "max_frames": args.max_frames,
        },
        "data_config": {
            "data_dir": str(args.data_dir),
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "length_multiple": args.length_multiple,
        },
        "args": serializable_args,
    }


def save_checkpoint(path, model, optimizer, args, epoch, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(unwrap_model(model), optimizer, args, epoch, global_step), path)


def main():
    args = parse_args()
    dist_info = setup_distributed(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + int(dist_info["rank"]))
    device = resolve_distributed_device(args.device, dist_info)

    train_dataset, train_loader, train_sampler = make_loader(
        args,
        args.train_split,
        args.limit_train,
        shuffle=True,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    val_loader = None
    if (
        dist_info["is_main"]
        and args.val_every > 0
        and (args.data_dir / "meta" / f"manifest_{args.val_split}.jsonl").is_file()
    ):
        _, val_loader, _ = make_loader(args, args.val_split, args.limit_val, shuffle=False)

    model = build_model_from_args(args).to(device)
    model = wrap_model(model, dist_info, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb_run = None
    if dist_info["is_main"] and args.wandb:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY", "")
        wandb_mode = os.environ.get("WANDB_MODE", "").lower()
        if wandb_api_key and wandb_mode not in {"disabled", "dryrun", "offline"}:
            wandb.login(key=wandb_api_key, relogin=True)

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    if dist_info["is_main"]:
        with (args.out_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    rank_zero_print(dist_info, f"Device: {device}")
    rank_zero_print(
        dist_info,
        f"Distributed: enabled={dist_info['enabled']} backend={dist_info['backend']} "
        f"world_size={dist_info['world_size']}",
    )
    rank_zero_print(dist_info, f"Train samples: {len(train_dataset)}")
    rank_zero_print(dist_info, f"Output: {args.out_dir}")

    global_step = 0
    best_val = float("inf")
    top_checkpoints = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not dist_info["is_main"])
        for batch in pbar:
            losses = compute_losses(model, batch, args, device)
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
            val_avg = validate(unwrap_model(model), val_loader, args, device)
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
            save_samples(unwrap_model(model), train_dataset, args, device, epoch)
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
            wandb.log(log, step=global_step)

    if dist_info["is_main"]:
        save_checkpoint(args.out_dir / "checkpoints" / "last.pt", model, optimizer, args, args.epochs, global_step)
        if wandb_run is not None:
            wandb_run.finish()
    barrier(dist_info)
    cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
