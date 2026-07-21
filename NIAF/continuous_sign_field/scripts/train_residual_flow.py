from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

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
from flow.smplx_features import COMPACT6D_DIM
from flow.text_encoder import FrozenT5TextEncoder
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import (
    ContinuousSignDataset,
    LengthBucketDistributedSampler,
    collate_continuous_sign,
    denormalize_motion,
)
from NIAF.continuous_sign_field.flow_matching import (
    endpoint_from_velocity,
    heun_integrate,
    sample_bridge,
)
from NIAF.continuous_sign_field.losses import endpoint_losses, masked_feature_mse
from NIAF.continuous_sign_field.metrics import ScalarAverager, append_jsonl, tensor_dict_to_float
from NIAF.continuous_sign_field.models import ResidualFlowTransformer
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.oracle_smplx_field.geometry.smplx_fk import DifferentiableSMPLXForward


def parse_args():
    parser = argparse.ArgumentParser(description="Train continuous SMPL-X residual flow with GT-anchor scaffold.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-niaf-continuous-sign-field")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    add_distributed_args(parser)
    return parser.parse_args()


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def move_batch_to_device(batch, device):
    out = dict(batch)
    for key in ("motion", "mask", "length", "left_valid", "right_valid", "fps", "duration", "index"):
        out[key] = batch[key].to(device)
    if "target_parts" in batch:
        out["target_parts"] = {key: value.to(device) for key, value in batch["target_parts"].items()}
    return out


def conditioning_texts(batch, field):
    field = str(field)
    if field == "gloss":
        return batch["gloss"]
    if field == "text_gloss":
        return [f"{text} {gloss}".strip() for text, gloss in zip(batch["text"], batch["gloss"])]
    return batch["text"]


def make_loader(
    cfg,
    split,
    limit,
    shuffle,
    distributed=False,
    world_size=1,
    num_workers=None,
):
    data_cfg = cfg["data"]
    dataset = ContinuousSignDataset(
        cfg,
        split=split,
        limit=limit,
        random_crop=bool(data_cfg.get("random_crop", False)) and split == data_cfg.get("train_split", "train"),
        require_fk_cache=True,
    )
    train_cfg = cfg.get("train", {})
    batch_size_key = "batch_size" if shuffle else "eval_batch_size"
    batch_size = int(
        train_cfg.get(batch_size_key, train_cfg.get("batch_size", 2))
    )
    drop_last = bool(shuffle and train_cfg.get("drop_last", False))
    if bool(train_cfg.get("length_bucketed_batches", False)):
        replicas = int(world_size) if distributed else 1
        rank = dist.get_rank() if distributed and dist.is_initialized() else 0
        sampler = LengthBucketDistributedSampler(
            dataset.estimated_lengths,
            batch_size=batch_size,
            num_replicas=replicas,
            rank=rank,
            shuffle=shuffle,
            seed=int(cfg.get("seed", 1234)),
            drop_last=drop_last,
            pad_to_full_batch=bool(shuffle and not drop_last),
        )
    else:
        sampler = (
            DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            if distributed
            else None
        )
    if num_workers is None:
        num_workers = int(train_cfg.get("num_workers", 0))
    else:
        num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            train_cfg.get("persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(
            train_cfg.get("prefetch_factor", 2)
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle) and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        collate_fn=collate_continuous_sign,
        drop_last=drop_last,
        **loader_kwargs,
    )
    return dataset, loader, sampler


def build_text_encoder(cfg, device):
    text_cfg = cfg.get("text", {})
    return FrozenT5TextEncoder(
        text_cfg.get("model_path", "deps/flan-t5-base"),
        device=device,
        max_length=int(text_cfg.get("max_tokens", 64)),
        local_files_only=bool(text_cfg.get("local_files_only", True)),
        cache=bool(text_cfg.get("cache", True)),
    )


def build_model(cfg, text_dim):
    model_cfg = cfg.get("model", {})
    return ResidualFlowTransformer(
        pose_dim=COMPACT6D_DIM,
        text_dim=int(text_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        time_fourier_bands=int(model_cfg.get("time_fourier_bands", 8)),
        use_text_cross_attention=bool(model_cfg.get("use_text_cross_attention", True)),
    )


def build_fk(cfg, device):
    metric_cfg = cfg.get("metrics", {})
    fk = DifferentiableSMPLXForward(
        model_dir=metric_cfg.get("model_dir", "deps/smpl_models"),
        gender=metric_cfg.get("gender", "NEUTRAL"),
        device=device,
        betas_mode=metric_cfg.get("betas_mode", "h2s_fixed"),
    )
    fk.eval()
    for param in fk.parameters():
        param.requires_grad_(False)
    return fk


def encode_batch_text(text_encoder, batch, cfg, model_device):
    texts = conditioning_texts(batch, cfg.get("text", {}).get("condition_field", "text"))
    tokens, token_mask = text_encoder.encode_tokens(texts)
    return tokens.to(model_device), token_mask.to(model_device)


def prepare_motion(batch, dataset, device):
    mean = torch.from_numpy(dataset.mean).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(dataset.std).to(device=device, dtype=torch.float32)
    return denormalize_motion(batch["motion"].to(dtype=torch.float32), mean, std)


def run_train_step(model, fk, text_encoder, scaffold_provider, optimizer, batch, dataset, cfg, device):
    train_cfg = cfg.get("train", {})
    flow_cfg = cfg.get("flow", {})
    loss_cfg = cfg.get("loss", {})

    x = prepare_motion(batch, dataset, device)
    mask = batch["mask"]
    lengths = batch["length"]
    target_parts = batch["target_parts"]
    scaffold, _anchor_mask = scaffold_provider.build(batch, x=x)
    tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
    target_residual = (x - scaffold) * mask.unsqueeze(-1).to(x.dtype)
    residual_t, target_velocity, _source_residual, flow_t = sample_bridge(
        target_residual,
        mask,
        noise_sigma=float(flow_cfg.get("source_noise_sigma", 0.05)),
        kernel_size=int(flow_cfg.get("source_noise_kernel", 9)),
        smooth_sigma=float(flow_cfg.get("source_noise_smooth_sigma", 2.0)),
    )
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)

    pred_velocity = model(
        residual_t,
        scaffold,
        tau,
        flow_t,
        mask=mask,
        text_tokens=text_tokens,
        text_mask=text_mask,
    )
    loss_fm = masked_feature_mse(
        pred_velocity,
        target_velocity,
        mask,
        hand_weight=float(loss_cfg.get("hand_weight", 5.0)),
    )
    endpoint_residual = endpoint_from_velocity(residual_t, pred_velocity, flow_t)
    pred_endpoint = scaffold + endpoint_residual
    loss_endpoint, endpoint_loss_dict = endpoint_losses(
        pred_endpoint,
        x,
        mask,
        lengths,
        target_parts,
        fk=fk,
        weights=loss_cfg,
        hand_weight=float(loss_cfg.get("hand_weight", 5.0)),
        fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
    )
    residual_loss = (endpoint_residual.square() * mask.unsqueeze(-1).to(endpoint_residual.dtype)).mean()
    total = (
        float(loss_cfg.get("lambda_fm", 1.0)) * loss_fm
        + loss_endpoint
        + float(loss_cfg.get("lambda_res", 0.0)) * residual_loss
    )

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
    optimizer.step()

    losses = tensor_dict_to_float(endpoint_loss_dict)
    losses["loss_fm"] = float(loss_fm.detach().cpu().item())
    losses["loss_res"] = float(residual_loss.detach().cpu().item())
    losses["loss_total"] = float(total.detach().cpu().item())
    return losses


@torch.no_grad()
def evaluate(model, fk, text_encoder, scaffold_provider, loader, dataset, cfg, device, max_batches=0, show_progress=True):
    model.eval()
    loss_cfg = cfg.get("loss", {})
    eval_cfg = cfg.get("eval", {})
    avg = ScalarAverager()
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False, disable=not show_progress)):
        if max_batches and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        x = prepare_motion(batch, dataset, device)
        mask = batch["mask"]
        lengths = batch["length"]
        target_parts = batch["target_parts"]
        scaffold, _anchor_mask = scaffold_provider.build(batch, x=x)
        tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
        text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
        pred_residual = heun_integrate(
            model,
            scaffold,
            tau,
            mask,
            text_tokens=text_tokens,
            text_mask=text_mask,
            steps=int(eval_cfg.get("solver_steps", 4)),
        )
        pred = scaffold + pred_residual
        _pred_total, pred_losses = endpoint_losses(
            pred,
            x,
            mask,
            lengths,
            target_parts,
            fk=fk,
            weights=loss_cfg,
            hand_weight=float(loss_cfg.get("hand_weight", 5.0)),
            fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
        )
        _scaffold_total, scaffold_losses = endpoint_losses(
            scaffold,
            x,
            mask,
            lengths,
            target_parts,
            fk=fk,
            weights=loss_cfg,
            hand_weight=float(loss_cfg.get("hand_weight", 5.0)),
            fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
        )
        avg.update(tensor_dict_to_float(pred_losses), n=len(batch["name"]), prefix="pred")
        avg.update(tensor_dict_to_float(scaffold_losses), n=len(batch["name"]), prefix="scaffold")
    return avg.mean()


def save_checkpoint(path, model, optimizer, epoch, global_step, cfg, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def init_wandb(args, cfg, dist_info, out_dir):
    if not dist_info["is_main"] or not args.wandb:
        return None
    import wandb

    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    if wandb_api_key and wandb_mode not in {"disabled", "dryrun", "offline"}:
        wandb.login(key=wandb_api_key, relogin=True)
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": {
            "experiment": cfg.get("experiment_name", "continuous_sign_field"),
            "config": cfg,
            "world_size": dist_info.get("world_size", 1),
            "output_dir": str(out_dir),
        },
        "dir": str(Path(os.environ.get("WANDB_DIR", out_dir))),
    }
    if args.wandb_id:
        wandb_kwargs["id"] = args.wandb_id
    if args.wandb_resume:
        wandb_kwargs["resume"] = args.wandb_resume
    return wandb.init(**wandb_kwargs)


def apply_overrides(cfg, args):
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.max_train_batches is not None:
        cfg.setdefault("train", {})["max_train_batches"] = int(args.max_train_batches)
    if args.max_val_batches is not None:
        cfg.setdefault("eval", {})["max_batches"] = int(args.max_val_batches)
    if args.limit_train is not None:
        cfg.setdefault("data", {})["limit_train"] = int(args.limit_train)
    if args.limit_val is not None:
        cfg.setdefault("data", {})["limit_val"] = int(args.limit_val)
    if args.device is not None:
        cfg["device"] = args.device
    if args.text_device is not None:
        cfg.setdefault("text", {})["device"] = args.text_device
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    dist_info = setup_distributed(args)
    set_seed(int(cfg.get("seed", 1234)) + int(dist_info.get("rank", 0)))
    device = resolve_distributed_device(cfg.get("device", "auto"), dist_info)
    text_device = torch.device(cfg.get("text", {}).get("device", "cpu"))
    out_dir = Path(cfg["output"]["out_dir"])
    if dist_info["is_main"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    train_split = cfg.get("data", {}).get("train_split", "train")
    val_split = cfg.get("data", {}).get("val_split", "val")
    train_dataset, train_loader, train_sampler = make_loader(
        cfg,
        train_split,
        limit=cfg.get("data", {}).get("limit_train", 0),
        shuffle=True,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    val_dataset, val_loader, val_sampler = make_loader(
        cfg,
        val_split,
        limit=cfg.get("data", {}).get("limit_val", 0),
        shuffle=False,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    rank_zero_print(
        dist_info,
        f"Loaded datasets: train={len(train_dataset)} val={len(val_dataset)} "
        f"world_size={dist_info['world_size']} batch_per_rank={cfg.get('train', {}).get('batch_size', 2)}",
    )
    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, train_dataset, device)
    model = build_model(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})
    start_epoch = 1
    global_step = 0
    optimizer_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer_state = checkpoint.get("optimizer")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    model = wrap_model(model, dist_info, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    best_metric = float("inf")
    epochs = int(train_cfg.get("epochs", 20))
    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    val_every = int(train_cfg.get("val_every", 1))
    save_every = int(train_cfg.get("save_every", 1))
    start_time = time.time()
    wandb_run = init_wandb(args, cfg, dist_info, out_dir)

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        model.train()
        avg = ScalarAverager()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", disable=not dist_info["is_main"])
        for batch_idx, batch in enumerate(pbar):
            if max_train_batches and batch_idx >= max_train_batches:
                break
            batch = move_batch_to_device(batch, device)
            losses = run_train_step(
                model,
                fk,
                text_encoder,
                scaffold_provider,
                optimizer,
                batch,
                train_dataset,
                cfg,
                device,
            )
            global_step += 1
            avg.update(losses, n=len(batch["name"]), prefix="train")
            if dist_info["is_main"]:
                pbar.set_postfix(loss=f"{losses['loss_total']:.4f}", fm=f"{losses['loss_fm']:.4f}")

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
        }
        train_metrics = distributed_mean_scalars(avg.mean(), device, dist_info)
        row.update(train_metrics)

        if epoch % val_every == 0:
            val_metrics = evaluate(
                unwrap_model(model),
                fk,
                text_encoder,
                scaffold_provider,
                val_loader,
                val_dataset,
                cfg,
                device,
                max_batches=int(cfg.get("eval", {}).get("max_batches", 0)),
                show_progress=dist_info["is_main"],
            )
            val_metrics = distributed_mean_scalars(val_metrics, device, dist_info)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = float(row.get("val_pred_loss_endpoint", row.get("val_pred_loss_rot6d", float("inf"))))
            if dist_info["is_main"] and score < best_metric:
                best_metric = score
                save_checkpoint(
                    out_dir / "checkpoints" / "best.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                )

        if dist_info["is_main"]:
            if epoch % save_every == 0:
                save_checkpoint(
                    out_dir / "checkpoints" / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                )
            append_jsonl(out_dir / "metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True))
            if wandb_run is not None:
                wandb_run.log(dict(row), step=global_step)
        barrier(dist_info)

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
