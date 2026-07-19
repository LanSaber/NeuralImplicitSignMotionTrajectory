from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.losses import (
    endpoint_losses,
    fk_temporal_dynamics_losses,
    fk_temporal_regularization_losses,
)
from NIAF.continuous_sign_field.meta_learning import highpass_residual_loss, masked_residual_loss
from NIAF.continuous_sign_field.metrics import ScalarAverager, append_jsonl, tensor_dict_to_float
from NIAF.continuous_sign_field.models import LocalAmortizedImplicitResidualField
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.train_meta_implicit_field import path_ratio_metric
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_fk,
    build_text_encoder,
    encode_batch_text,
    make_loader,
    move_batch_to_device,
    prepare_motion,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-amortized local implicit SMPL-X residual field.")
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
    parser.add_argument("--wandb_project", default="soke-niaf-local-implicit")
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


def build_local_model(cfg, text_dim):
    model_cfg = cfg.get("model", {})
    duration_cfg = cfg.get("duration", {})
    return LocalAmortizedImplicitResidualField(
        pose_dim=COMPACT6D_DIM,
        text_dim=int(text_dim),
        code_dim=int(model_cfg.get("code_dim", 128)),
        context_hidden_dim=int(model_cfg.get("context_hidden_dim", 256)),
        context_layers=int(model_cfg.get("context_layers", 2)),
        context_heads=int(model_cfg.get("context_heads", 8)),
        local_stride=int(model_cfg.get("local_stride", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        depth=int(model_cfg.get("depth", 4)),
        time_fourier_bands=int(model_cfg.get("time_fourier_bands", 10)),
        context_time_fourier_bands=int(model_cfg.get("context_time_fourier_bands", 6)),
        omega0_first=float(model_cfg.get("omega0_first", 20.0)),
        omega0_hidden=float(model_cfg.get("omega0_hidden", 1.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        residual_scale_learnable=bool(model_cfg.get("residual_scale_learnable", True)),
        body_gate_bias=float(model_cfg.get("body_gate_bias", -3.0)),
        hand_gate_bias=float(model_cfg.get("hand_gate_bias", -2.0)),
        face_gate_bias=float(model_cfg.get("face_gate_bias", -3.0)),
        duration_hidden_dim=int(duration_cfg.get("hidden_dim", 256)),
        duration_initial_frames=float(duration_cfg.get("initial_frames", 80.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def fk_regularization_weights(local_loss_cfg):
    keep = {"lambda_fk_vel_reg", "lambda_fk_acc_reg", "lambda_fk_jerk_reg", "fk_temporal_include_hand_parts"}
    return {key: value for key, value in local_loss_cfg.items() if key in keep}


def local_code_smoothness(codes, mask):
    if codes.shape[1] <= 1:
        return codes.new_tensor(0.0)
    valid = mask[:, 1:] & mask[:, :-1]
    diff = torch.abs(codes[:, 1:] - codes[:, :-1])
    weight = valid.unsqueeze(-1).to(diff.dtype)
    return (diff * weight).sum() / weight.expand_as(diff).sum().clamp_min(1.0)


def masked_gate_means(gates, mask):
    weight = mask.unsqueeze(-1).to(gates.dtype)
    values = (gates * weight).sum(dim=(0, 1)) / weight.sum().clamp_min(1.0)
    return {
        "gate_body": values[0],
        "gate_lhand": values[1],
        "gate_rhand": values[2],
        "gate_face": values[3],
    }


def prepare_local_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device):
    x = prepare_motion(batch, dataset, device)
    mask = batch["mask"]
    lengths = batch["length"]
    scaffold, _anchor_mask = scaffold_provider.build(batch, x=None)
    tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    outputs = model(tau, scaffold, mask, text_tokens, text_mask=text_mask)
    return {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "target_parts": batch["target_parts"],
        "scaffold": scaffold,
        "tau": tau,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "outputs": outputs,
    }


def local_losses(model, fk, prepared, cfg):
    loss_cfg = cfg.get("loss", {})
    local_cfg = cfg.get("local_loss", {})
    hand_weight = float(loss_cfg.get("hand_weight", 5.0))
    x = prepared["x"]
    mask = prepared["mask"]
    lengths = prepared["lengths"]
    target_parts = prepared["target_parts"]
    scaffold = prepared["scaffold"]
    outputs = prepared["outputs"]
    pred = outputs["prediction"]
    residual = outputs["residual"]
    target_residual = (x - scaffold) * mask.unsqueeze(-1).to(x.dtype)

    endpoint, endpoint_dict = endpoint_losses(
        pred,
        x,
        mask,
        lengths,
        target_parts,
        fk=fk,
        weights=loss_cfg,
        hand_weight=hand_weight,
        fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
    )
    residual_loss = masked_residual_loss(
        residual,
        target_residual,
        mask,
        hand_weight=hand_weight,
        loss_type=str(local_cfg.get("residual_loss", "l1")),
    )
    residual_vel = highpass_residual_loss(residual, target_residual, mask, order=1, hand_weight=hand_weight)
    residual_acc = highpass_residual_loss(residual, target_residual, mask, order=2, hand_weight=hand_weight)
    fk_regularization, fk_regularization_dict = fk_temporal_regularization_losses(
        pred,
        mask,
        lengths,
        target_parts,
        fk,
        weights=fk_regularization_weights(local_cfg),
        hand_weight=hand_weight,
        fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
    )
    code_smooth = local_code_smoothness(outputs["local_codes"], outputs["local_code_mask"])
    target_log_frames = torch.log(lengths.to(dtype=outputs["pred_log_frames"].dtype).clamp_min(1.0))
    duration_loss = F.smooth_l1_loss(outputs["pred_log_frames"], target_log_frames)
    valid = mask.unsqueeze(-1).to(residual.dtype)
    residual_rms = torch.sqrt((residual.square() * valid).sum() / valid.expand_as(residual).sum().clamp_min(1.0))
    gate_penalty = (outputs["gates"] * mask.unsqueeze(-1).to(outputs["gates"].dtype)).sum()
    gate_penalty = gate_penalty / mask.sum().clamp_min(1).to(gate_penalty.dtype) / outputs["gates"].shape[-1]

    total = (
        float(local_cfg.get("lambda_endpoint", 1.0)) * endpoint
        + float(local_cfg.get("lambda_residual", 0.5)) * residual_loss
        + float(local_cfg.get("lambda_residual_vel", 0.25)) * residual_vel
        + float(local_cfg.get("lambda_residual_acc", 0.1)) * residual_acc
        + float(local_cfg.get("lambda_code_smooth", 0.01)) * code_smooth
        + float(local_cfg.get("lambda_duration", 0.25)) * duration_loss
        + float(local_cfg.get("lambda_gate", 0.0)) * gate_penalty
        + fk_regularization
    )

    predicted_frames = torch.exp(outputs["pred_log_frames"])
    duration_mae = torch.abs(predicted_frames - lengths.to(predicted_frames.dtype)).mean()
    duration_relative = (
        torch.abs(predicted_frames - lengths.to(predicted_frames.dtype))
        / lengths.to(predicted_frames.dtype).clamp_min(1.0)
    ).mean()
    losses = {
        "loss_total": total,
        **endpoint_dict,
        "loss_residual": residual_loss,
        "loss_residual_vel": residual_vel,
        "loss_residual_acc": residual_acc,
        "loss_code_smooth": code_smooth,
        "loss_duration": duration_loss,
        "loss_gate": gate_penalty,
        "loss_fk_regularization": fk_regularization,
        "residual_rms": residual_rms,
        "duration_mae_frames": duration_mae,
        "duration_relative_error": duration_relative,
    }
    losses.update(fk_regularization_dict)
    losses.update(masked_gate_means(outputs["gates"], mask))
    return total, losses


def run_train_step(model, fk, text_encoder, scaffold_provider, optimizer, batch, dataset, cfg, device):
    prepared = prepare_local_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device)
    total, losses = local_losses(model, fk, prepared, cfg)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0)))
    optimizer.step()
    return tensor_dict_to_float(losses)


@torch.no_grad()
def evaluate(model, fk, text_encoder, scaffold_provider, loader, dataset, cfg, device, max_batches=0):
    model.eval()
    loss_cfg = cfg.get("loss", {})
    hand_weight = float(loss_cfg.get("hand_weight", 5.0))
    chunk_size = int(cfg.get("metrics", {}).get("fk_batch_size", 128))
    avg = ScalarAverager()
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        prepared = prepare_local_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device)
        _total, losses = local_losses(model, fk, prepared, cfg)
        for name, pred in (("scaffold", prepared["scaffold"]), ("pred", prepared["outputs"]["prediction"])):
            _endpoint, pred_losses = endpoint_losses(
                pred,
                prepared["x"],
                prepared["mask"],
                prepared["lengths"],
                prepared["target_parts"],
                fk=fk,
                weights=loss_cfg,
                hand_weight=hand_weight,
                fk_chunk_size=chunk_size,
            )
            _dynamics, dynamics_losses = fk_temporal_dynamics_losses(
                pred,
                prepared["mask"],
                prepared["lengths"],
                prepared["target_parts"],
                fk,
                weights={
                    "lambda_fk_vel": 1.0,
                    "lambda_fk_acc": 1.0,
                    "lambda_fk_jerk": 1.0,
                    "fk_temporal_include_hand_parts": True,
                },
                hand_weight=hand_weight,
                fk_chunk_size=chunk_size,
            )
            pred_metrics = tensor_dict_to_float(pred_losses)
            pred_metrics.update(tensor_dict_to_float(dynamics_losses))
            pred_metrics["hand_path_ratio"] = float(
                path_ratio_metric(
                    fk,
                    pred,
                    prepared["target_parts"],
                    prepared["mask"],
                    chunk_size=chunk_size,
                ).item()
            )
            avg.update(pred_metrics, n=len(batch["name"]), prefix=name)
        avg.update(tensor_dict_to_float(losses), n=len(batch["name"]), prefix="local")
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
            "model_type": "local_amortized_implicit_residual_field",
        },
        path,
    )


def init_wandb(args, cfg, out_dir):
    if not args.wandb:
        return None
    import wandb

    api_key = os.environ.get("WANDB_API_KEY", "")
    mode = os.environ.get("WANDB_MODE", "").lower()
    if api_key and mode not in {"disabled", "dryrun", "offline"}:
        wandb.login(key=api_key, relogin=True)
    kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": {
            "experiment": cfg.get("experiment_name", "local_amortized_implicit_field"),
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
    train_dataset, train_loader, _train_sampler = make_loader(
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
    print(f"Loaded datasets: train={len(train_dataset)} val={len(val_dataset)}")

    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, train_dataset, device)
    model = build_local_model(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("metrics", {}).get("val_pred_loss_endpoint", float("inf")))

    epochs = int(train_cfg.get("epochs", 30))
    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    val_every = int(train_cfg.get("val_every", 1))
    save_every = int(train_cfg.get("save_every", 1))
    start_time = time.time()
    wandb_run = init_wandb(args, cfg, out_dir)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        avg = ScalarAverager()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
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
            pbar.set_postfix(
                loss=f"{losses['loss_total']:.4f}",
                res=f"{losses['loss_residual']:.4f}",
                dur=f"{losses['duration_mae_frames']:.1f}",
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
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = float(row.get("val_pred_loss_endpoint", row.get("val_local_loss_total", float("inf"))))
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
