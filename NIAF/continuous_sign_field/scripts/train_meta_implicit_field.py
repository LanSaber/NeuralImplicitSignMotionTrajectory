from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.losses import (
    endpoint_losses,
    fk_temporal_dynamics_losses,
    fk_temporal_regularization_losses,
    masked_feature_l1,
    parts_from_rot6d_chunks,
)
from NIAF.continuous_sign_field.meta_learning import (
    adapt_code,
    build_support_query_masks,
    highpass_residual_loss,
    masked_residual_loss,
    support_adaptation_loss,
)
from NIAF.continuous_sign_field.metrics import ScalarAverager, append_jsonl, tensor_dict_to_float
from NIAF.continuous_sign_field.models import MetaImplicitResidualField
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_fk,
    build_text_encoder,
    encode_batch_text,
    make_loader,
    move_batch_to_device,
    prepare_motion,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAML-style meta-implicit NIAF trajectory fields.")
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
    parser.add_argument("--wandb_project", default="soke-niaf-meta-implicit")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


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


def build_meta_model(cfg, text_dim):
    model_cfg = cfg.get("model", {})
    return MetaImplicitResidualField(
        pose_dim=COMPACT6D_DIM,
        text_dim=int(text_dim),
        code_dim=int(model_cfg.get("code_dim", cfg.get("meta", {}).get("code_dim", 128))),
        context_hidden_dim=int(model_cfg.get("context_hidden_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        depth=int(model_cfg.get("depth", 4)),
        time_fourier_bands=int(model_cfg.get("time_fourier_bands", 10)),
        omega0_first=float(model_cfg.get("omega0_first", 20.0)),
        omega0_hidden=float(model_cfg.get("omega0_hidden", 1.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        residual_scale_learnable=bool(model_cfg.get("residual_scale_learnable", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        condition_dim=int(model_cfg.get("condition_dim", 0)),
    )


def endpoint_pose_weights(loss_cfg):
    keep = {"lambda_rot6d", "lambda_geo", "lambda_joint", "lambda_hand", "lambda_expr"}
    return {key: value for key, value in loss_cfg.items() if key in keep}


def fk_dynamics_weights(meta_loss_cfg):
    keep = {"lambda_fk_vel", "lambda_fk_acc", "lambda_fk_jerk", "fk_temporal_include_hand_parts"}
    return {key: value for key, value in meta_loss_cfg.items() if key in keep}


def fk_regularization_weights(meta_loss_cfg):
    keep = {"lambda_fk_vel_reg", "lambda_fk_acc_reg", "lambda_fk_jerk_reg", "fk_temporal_include_hand_parts"}
    return {key: value for key, value in meta_loss_cfg.items() if key in keep}


def hand_path_ratio(parts, target_parts, mask):
    ratios = []
    for key in ("lhand", "rhand"):
        for idx in range(mask.shape[0]):
            valid = mask[idx]
            if int(valid.sum().item()) <= 1:
                continue
            pred = parts[key][idx, valid]
            target = target_parts[key][idx, valid].to(device=pred.device, dtype=pred.dtype)
            pred_path = torch.linalg.norm(pred[1:] - pred[:-1], dim=-1).sum()
            target_path = torch.linalg.norm(target[1:] - target[:-1], dim=-1).sum()
            ratios.append(pred_path / target_path.clamp_min(1e-6))
    if not ratios:
        return mask.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(ratios).mean()


def path_ratio_metric(fk, pred, target_parts, mask, chunk_size=128):
    valid_pred = pred[mask]
    if valid_pred.numel() == 0:
        return pred.new_tensor(0.0)
    flat_parts = parts_from_rot6d_chunks(fk, valid_pred, chunk_size=chunk_size)
    padded = {}
    for key, target in target_parts.items():
        value = target.new_zeros(target.shape, device=pred.device, dtype=pred.dtype)
        value[mask] = flat_parts[key].to(device=pred.device, dtype=pred.dtype)
        padded[key] = value
    target = {key: value.to(device=pred.device, dtype=pred.dtype) for key, value in target_parts.items()}
    return hand_path_ratio(padded, target, mask)


def prepare_meta_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device):
    x = prepare_motion(batch, dataset, device)
    mask = batch["mask"]
    lengths = batch["length"]
    target_parts = batch["target_parts"]
    scaffold, anchor_mask = scaffold_provider.build(batch, x=x)
    tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    z0 = model.initial_code(scaffold, mask, lengths, text_tokens=text_tokens, text_mask=text_mask)
    target_residual = (x - scaffold) * mask.unsqueeze(-1).to(x.dtype)
    meta_cfg = cfg.get("meta", {})
    support_mask, query_mask = build_support_query_masks(
        mask,
        anchor_mask=anchor_mask,
        support_mode=meta_cfg.get("support_mode", "stride"),
        support_stride=int(meta_cfg.get("support_stride", 8)),
    )
    return {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "target_parts": target_parts,
        "scaffold": scaffold,
        "anchor_mask": anchor_mask,
        "support_mask": support_mask,
        "query_mask": query_mask,
        "tau": tau,
        "z0": z0,
        "target_residual": target_residual,
    }


def meta_losses(model, fk, prepared, cfg, prefix=""):
    loss_cfg = cfg.get("loss", {})
    meta_loss_cfg = cfg.get("meta_loss", {})
    hand_weight = float(loss_cfg.get("hand_weight", meta_loss_cfg.get("hand_weight", 5.0)))
    pose_weights = endpoint_pose_weights(loss_cfg)
    x = prepared["x"]
    mask = prepared["mask"]
    query_mask = prepared["query_mask"]
    reconstruction_mask = query_mask
    if str(meta_loss_cfg.get("reconstruction_mask", "query")).lower() in {"all", "valid", "full"}:
        reconstruction_mask = mask
    support_mask = prepared["support_mask"]
    target_parts = prepared["target_parts"]
    scaffold = prepared["scaffold"]
    tau = prepared["tau"]
    z0 = prepared["z0"]
    target_residual = prepared["target_residual"]

    prior_residual = model(tau, scaffold, z0, mask=mask)
    prior = scaffold + prior_residual
    adapted_code, inner_losses = adapt_code(
        model,
        z0,
        tau,
        scaffold,
        x,
        target_residual,
        support_mask,
        cfg,
    )
    adapted_residual = model(tau, scaffold, adapted_code, mask=mask)
    adapted = scaffold + adapted_residual

    query_endpoint, query_endpoint_dict = endpoint_losses(
        adapted,
        x,
        reconstruction_mask,
        prepared["lengths"],
        target_parts,
        fk=fk,
        weights=pose_weights,
        hand_weight=hand_weight,
        fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
    )
    query_residual = masked_residual_loss(
        adapted_residual,
        target_residual,
        reconstruction_mask,
        hand_weight=hand_weight,
        loss_type=str(meta_loss_cfg.get("query_residual_loss", "l1")),
    )
    prior_query_residual = masked_residual_loss(
        prior_residual,
        target_residual,
        reconstruction_mask,
        hand_weight=hand_weight,
        loss_type=str(meta_loss_cfg.get("query_residual_loss", "l1")),
    )
    support_consistency, support_losses = support_adaptation_loss(
        model,
        adapted_code,
        tau,
        scaffold,
        x,
        target_residual,
        support_mask,
        cfg,
    )
    residual_vel = highpass_residual_loss(
        adapted_residual,
        target_residual,
        mask,
        order=1,
        hand_weight=hand_weight,
    )
    residual_acc = highpass_residual_loss(
        adapted_residual,
        target_residual,
        mask,
        order=2,
        hand_weight=hand_weight,
    )
    fk_dynamics, fk_dynamics_dict = fk_temporal_dynamics_losses(
        adapted,
        mask,
        prepared["lengths"],
        target_parts,
        fk,
        weights=fk_dynamics_weights(meta_loss_cfg),
        hand_weight=hand_weight,
        fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
    )
    fk_regularization, fk_regularization_dict = fk_temporal_regularization_losses(
        adapted,
        mask,
        prepared["lengths"],
        target_parts,
        fk,
        weights=fk_regularization_weights(meta_loss_cfg),
        hand_weight=hand_weight,
        fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
    )
    residual_rms = torch.sqrt(((adapted_residual ** 2) * mask.unsqueeze(-1).to(adapted_residual.dtype)).mean())

    total = (
        float(meta_loss_cfg.get("lambda_query_endpoint", 1.0)) * query_endpoint
        + float(meta_loss_cfg.get("lambda_query_residual", 1.0)) * query_residual
        + float(meta_loss_cfg.get("lambda_prior_query_residual", 0.25)) * prior_query_residual
        + float(meta_loss_cfg.get("lambda_support_consistency", 0.25)) * support_consistency
        + float(meta_loss_cfg.get("lambda_residual_vel", 0.5)) * residual_vel
        + float(meta_loss_cfg.get("lambda_residual_acc", 0.25)) * residual_acc
        + fk_dynamics
        + fk_regularization
    )

    losses = {f"{prefix}loss_total": total}
    losses.update({f"{prefix}{key}": value for key, value in query_endpoint_dict.items()})
    losses.update({f"{prefix}{key}": value for key, value in inner_losses.items()})
    losses.update({f"{prefix}loss_query_residual": query_residual})
    losses.update({f"{prefix}loss_prior_query_residual": prior_query_residual})
    losses.update({f"{prefix}loss_support_consistency": support_consistency})
    losses.update({f"{prefix}loss_residual_vel": residual_vel})
    losses.update({f"{prefix}loss_residual_acc": residual_acc})
    losses.update({f"{prefix}{key}": value for key, value in fk_dynamics_dict.items()})
    losses.update({f"{prefix}loss_fk_dynamics": fk_dynamics})
    losses.update({f"{prefix}{key}": value for key, value in fk_regularization_dict.items()})
    losses.update({f"{prefix}loss_fk_regularization": fk_regularization})
    losses.update({f"{prefix}adapted_residual_rms": residual_rms})
    losses.update({f"{prefix}support_fraction": support_mask.float().mean()})
    losses.update({f"{prefix}query_fraction": query_mask.float().mean()})
    losses.update({f"{prefix}reconstruction_fraction": reconstruction_mask.float().mean()})
    return total, losses, {
        "prior": prior,
        "adapted": adapted,
        "scaffold": scaffold,
        "adapted_code": adapted_code,
        "prior_residual": prior_residual,
        "adapted_residual": adapted_residual,
    }


def run_train_step(model, fk, text_encoder, scaffold_provider, optimizer, batch, dataset, cfg, device):
    train_cfg = cfg.get("train", {})
    prepared = prepare_meta_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device)
    total, losses, _outputs = meta_losses(model, fk, prepared, cfg)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
    optimizer.step()
    return tensor_dict_to_float(losses)


def evaluate(model, fk, text_encoder, scaffold_provider, loader, dataset, cfg, device, max_batches=0, show_progress=True):
    model.eval()
    loss_cfg = cfg.get("loss", {})
    pose_weights = endpoint_pose_weights(loss_cfg)
    hand_weight = float(loss_cfg.get("hand_weight", 5.0))
    avg = ScalarAverager()
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False, disable=not show_progress)):
        if max_batches and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        prepared = prepare_meta_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device)
        _total, losses, outputs = meta_losses(model, fk, prepared, cfg)
        x = prepared["x"]
        mask = prepared["mask"]
        lengths = prepared["lengths"]
        target_parts = prepared["target_parts"]
        for name, pred in (
            ("scaffold", outputs["scaffold"]),
            ("prior", outputs["prior"]),
            ("adapted", outputs["adapted"]),
        ):
            _score, pred_losses = endpoint_losses(
                pred,
                x,
                mask,
                lengths,
                target_parts,
                fk=fk,
                weights=pose_weights,
                hand_weight=hand_weight,
                fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
            )
            pred_losses = tensor_dict_to_float(pred_losses)
            _dyn_total, dynamics_losses = fk_temporal_dynamics_losses(
                pred,
                mask,
                lengths,
                target_parts,
                fk,
                weights={
                    "lambda_fk_vel": 1.0,
                    "lambda_fk_acc": 1.0,
                    "lambda_fk_jerk": 1.0,
                    "fk_temporal_include_hand_parts": True,
                },
                hand_weight=hand_weight,
                fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
            )
            pred_losses.update(tensor_dict_to_float(dynamics_losses))
            pred_losses["hand_path_ratio"] = float(
                path_ratio_metric(
                    fk,
                    pred,
                    target_parts,
                    mask,
                    chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
                )
                .detach()
                .cpu()
                .item()
            )
            avg.update(pred_losses, n=len(batch["name"]), prefix=name)
        avg.update(tensor_dict_to_float(losses), n=len(batch["name"]), prefix="meta")
    model.train()
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
            "model_type": "meta_implicit_residual_field",
        },
        path,
    )


def init_wandb(args, cfg, out_dir):
    if not args.wandb:
        return None
    import wandb

    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    if wandb_api_key and wandb_mode not in {"disabled", "dryrun", "offline"}:
        wandb.login(key=wandb_api_key, relogin=True)
    kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": {
            "experiment": cfg.get("experiment_name", "meta_implicit_continuous_sign_field"),
            "config": cfg,
            "output_dir": str(out_dir),
        },
        "dir": str(Path(os.environ.get("WANDB_DIR", out_dir))),
    }
    if args.wandb_id:
        kwargs["id"] = args.wandb_id
    if args.wandb_resume:
        kwargs["resume"] = args.wandb_resume
    return wandb.init(**kwargs)


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 1234)))
    device = resolve_device(cfg.get("device", "auto"))
    text_device = resolve_device(cfg.get("text", {}).get("device", "cpu"))
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "val")
    train_dataset, train_loader, train_sampler = make_loader(
        cfg,
        train_split,
        limit=data_cfg.get("limit_train", 0),
        shuffle=True,
        distributed=False,
    )
    val_dataset, val_loader, _val_sampler = make_loader(
        cfg,
        val_split,
        limit=data_cfg.get("limit_val", 0),
        shuffle=False,
        distributed=False,
    )
    print(
        f"Loaded datasets: train={len(train_dataset)} val={len(val_dataset)} "
        f"batch={cfg.get('train', {}).get('batch_size', 2)}"
    )

    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, train_dataset, device)
    model = build_meta_model(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    epochs = int(train_cfg.get("epochs", 20))
    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    val_every = int(train_cfg.get("val_every", 1))
    save_every = int(train_cfg.get("save_every", 1))
    best_metric = float("inf")
    start_time = time.time()
    wandb_run = init_wandb(args, cfg, out_dir)

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        avg = ScalarAverager()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            if max_train_batches and batch_idx >= max_train_batches:
                break
            batch = move_batch_to_device(batch, device)
            losses = run_train_step(model, fk, text_encoder, scaffold_provider, optimizer, batch, train_dataset, cfg, device)
            global_step += 1
            avg.update(losses, n=len(batch["name"]), prefix="train")
            pbar.set_postfix(
                loss=f"{losses['loss_total']:.4f}",
                qres=f"{losses['loss_query_residual']:.4f}",
                rms=f"{losses['adapted_residual_rms']:.4f}",
            )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
        }
        row.update(avg.mean())
        if epoch % val_every == 0:
            val_metrics = evaluate(
                model,
                fk,
                text_encoder,
                scaffold_provider,
                val_loader,
                val_dataset,
                cfg,
                device,
                max_batches=int(cfg.get("eval", {}).get("max_batches", 0)),
                show_progress=True,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = float(row.get("val_adapted_loss_rot6d", row.get("val_meta_loss_query_residual", float("inf"))))
            if score < best_metric:
                best_metric = score
                save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, global_step, cfg, row)

        if epoch % save_every == 0:
            save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, global_step, cfg, row)
        append_jsonl(out_dir / "metrics.jsonl", row)
        print(json.dumps(row, sort_keys=True))
        if wandb_run is not None:
            wandb_run.log(dict(row), step=global_step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
