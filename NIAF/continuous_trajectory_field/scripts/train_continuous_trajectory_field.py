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
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.losses import (
    endpoint_losses,
    fk_temporal_dynamics_losses,
)
from NIAF.continuous_sign_field.metrics import (
    ScalarAverager,
    append_jsonl,
    tensor_dict_to_float,
)
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
from NIAF.continuous_trajectory_field.losses import (
    analytic_fk_dynamics_losses,
    duration_regression_loss,
    local_field_regularization,
    prior_and_residual_losses,
)
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    validate_train_only_retrieval_bank,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an amortized continuous SMPL-X trajectory field."
    )
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
    parser.add_argument(
        "--warm_start",
        type=Path,
        default=None,
        help="Load model weights only, reset epoch/optimizer state, and start a new run.",
    )
    parser.add_argument(
        "--reset_local_branch",
        action="store_true",
        help="Reinitialize density/local-context/local-head parameters after warm start.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-niaf-continuous-trajectory")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    add_distributed_args(parser)
    return parser.parse_args()


def apply_overrides(cfg, args):
    mappings = (
        ("epochs", "train", "epochs"),
        ("batch_size", "train", "batch_size"),
        ("max_train_batches", "train", "max_train_batches"),
        ("max_val_batches", "eval", "max_batches"),
        ("limit_train", "data", "limit_train"),
        ("limit_val", "data", "limit_val"),
    )
    for argument, section, key in mappings:
        value = getattr(args, argument)
        if value is not None:
            cfg.setdefault(section, {})[key] = int(value)
    if args.device is not None:
        cfg["device"] = args.device
    if args.text_device is not None:
        cfg.setdefault("text", {})["device"] = args.text_device
    if args.out_dir is not None:
        cfg.setdefault("output", {})["out_dir"] = str(args.out_dir)
    if args.warm_start is not None:
        cfg.setdefault("train", {})["warm_start_checkpoint"] = str(args.warm_start)
    if args.reset_local_branch:
        cfg.setdefault("train", {})["reset_local_branch_on_warm_start"] = True
    return cfg


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def native_query_times(lengths, max_len, device, dtype):
    unit = normalized_time_grid(lengths, max_len=max_len, device=device, dtype=dtype)
    return 2.0 * unit.squeeze(-1) - 1.0


def _masked_rms(values, mask):
    selected = values[mask]
    if selected.numel() == 0:
        return values.new_tensor(0.0)
    return torch.sqrt(selected.square().mean().clamp_min(1e-12))


def _masked_pearson(left, right, mask):
    left = left[mask]
    right = right[mask]
    if left.numel() < 2:
        return left.new_tensor(0.0)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1e-12)
    return (left * right).sum() / denominator


def prepare_field_batch(model, text_encoder, provider, batch, dataset, cfg, device):
    target = prepare_motion(batch, dataset, device)
    adapter_context, _anchors, metadata = provider.build_with_metadata(
        batch,
        x=None,
        use_cache=True,
    )
    retrieval = metadata["retrieval_features"]
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    tau = native_query_times(
        batch["length"],
        max_len=target.shape[1],
        device=device,
        dtype=target.dtype,
    )
    outputs = model(
        text_tokens=text_tokens,
        adapter_context=adapter_context,
        context_mask=batch["mask"],
        retrieval_evidence=retrieval,
        query_times=tau,
        text_mask=text_mask,
        time_domain="normalized",
        query_mask=batch["mask"],
    )
    return {
        "target": target,
        "adapter_context": adapter_context,
        "retrieval": retrieval,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "tau": tau,
        "outputs": outputs,
    }


def _scheduled_analytic_weights(cfg, epoch):
    weights = dict(cfg.get("analytic_dynamics", {}))
    start_epoch = int(weights.get("start_epoch", 1))
    jerk_start = int(weights.get("jerk_start_epoch", start_epoch))
    jerk_ramp = max(int(weights.get("jerk_ramp_epochs", 1)), 1)
    if int(epoch) < start_epoch:
        for key in (
            "lambda_analytic_fk_vel",
            "lambda_analytic_fk_acc",
            "lambda_analytic_fk_jerk",
            "lambda_analytic_fk_jerk_reg",
        ):
            weights[key] = 0.0
        return weights
    jerk_scale = min(max((int(epoch) - jerk_start + 1) / jerk_ramp, 0.0), 1.0)
    for key in ("lambda_analytic_fk_jerk", "lambda_analytic_fk_jerk_reg"):
        weights[key] = float(weights.get(key, 0.0)) * jerk_scale
    return weights


def compute_batch_losses(
    model,
    fk,
    text_encoder,
    provider,
    batch,
    dataset,
    cfg,
    device,
    epoch=1,
    training=True,
):
    prepared = prepare_field_batch(
        model,
        text_encoder,
        provider,
        batch,
        dataset,
        cfg,
        device,
    )
    target = prepared["target"]
    adapter_context = prepared["adapter_context"]
    outputs = prepared["outputs"]
    trajectory = outputs["trajectory"]
    mask = batch["mask"]
    lengths = batch["length"]
    target_parts = batch.get("target_parts")
    loss_cfg = cfg.get("loss", {})
    objective_cfg = cfg.get("objective", {})
    temporal_cfg = cfg.get("temporal_loss", {})
    hand_weight = float(loss_cfg.get("hand_weight", 5.0))
    fk_chunk_size = int(cfg.get("metrics", {}).get("fk_batch_size", 128))

    endpoint_total, endpoint = endpoint_losses(
        outputs["prediction"],
        target,
        mask,
        lengths,
        target_parts,
        fk=fk,
        weights=loss_cfg,
        hand_weight=hand_weight,
        fk_chunk_size=fk_chunk_size,
    )
    auxiliary = prior_and_residual_losses(
        outputs,
        adapter_context,
        target,
        mask,
        hand_weight=hand_weight,
    )
    duration = duration_regression_loss(
        trajectory.log_duration_seconds,
        batch["duration"],
        beta=float(cfg.get("duration", {}).get("huber_beta", 0.15)),
    )
    local = local_field_regularization(trajectory)
    finite_total, finite_losses = fk_temporal_dynamics_losses(
        outputs["prediction"],
        mask,
        lengths,
        target_parts,
        fk,
        weights=temporal_cfg,
        hand_weight=hand_weight,
        fk_chunk_size=fk_chunk_size,
    )

    analytic_weights = _scheduled_analytic_weights(cfg, epoch)
    derivative_duration = (
        batch["duration"]
        if bool(analytic_weights.get("duration_teacher_forcing", True))
        else trajectory.duration_seconds
    )
    analytic_total, analytic = analytic_fk_dynamics_losses(
        unwrap_model(model),
        trajectory,
        fk,
        target_parts,
        lengths,
        derivative_duration,
        weights=analytic_weights,
        query_count=int(analytic_weights.get("query_count", 8)),
        hand_weight=hand_weight,
        smooth_kernel=int(analytic_weights.get("smooth_kernel", 7)),
        smooth_sigma=float(analytic_weights.get("smooth_sigma", 1.5)),
        jerk_target_ratio=float(analytic_weights.get("jerk_target_ratio", 0.75)),
        randomize_queries=bool(training and analytic_weights.get("randomize_queries", True)),
    )

    total = float(objective_cfg.get("lambda_endpoint", 1.0)) * endpoint_total
    total = total + float(objective_cfg.get("lambda_prior", 0.25)) * auxiliary["loss_prior"]
    total = total + float(objective_cfg.get("lambda_residual", 0.5)) * auxiliary["loss_residual"]
    total = total + float(objective_cfg.get("lambda_duration", 0.25)) * duration
    total = total + float(objective_cfg.get("lambda_local_modulation", 1e-5)) * local[
        "loss_local_modulation"
    ]
    total = total + float(objective_cfg.get("lambda_local_width", 0.0)) * local[
        "loss_local_width"
    ]
    total = total + finite_total + analytic_total

    losses = {}
    losses.update(endpoint)
    losses.update(auxiliary)
    losses.update(local)
    losses.update(finite_losses)
    losses.update(analytic)
    losses["loss_duration"] = duration
    losses["duration_pred_seconds"] = trajectory.duration_seconds.mean()
    losses["duration_target_seconds"] = batch["duration"].mean()
    losses["residual_rms"] = torch.sqrt(
        outputs["correction_axis"][mask].square().mean().clamp_min(1e-12)
    )
    local_axis = outputs["local_correction_axis"]
    global_axis = outputs["global_correction_axis"]
    local_parts = {
        "body": slice(0, 30),
        "left_hand": slice(30, 75),
        "right_hand": slice(75, 120),
        "face": slice(120, 133),
    }
    losses["local_residual_rms"] = _masked_rms(local_axis, mask)
    losses["global_residual_rms"] = _masked_rms(global_axis, mask)
    for part_name, part_slice in local_parts.items():
        losses[f"local_residual_rms_{part_name}"] = _masked_rms(
            local_axis[..., part_slice], mask
        )
    valid_count = mask.to(outputs["local_coverage"].dtype).sum().clamp_min(1.0)
    losses["local_window_coverage"] = (
        outputs["local_coverage"] * mask.to(outputs["local_coverage"].dtype)
    ).sum() / valid_count
    overlap = (outputs["local_weights"] > 0.05).sum(dim=-1) > 1
    losses["local_window_overlap_fraction"] = (
        overlap & mask
    ).to(local_axis.dtype).sum() / valid_count
    losses["active_local_fields"] = trajectory.local_mask.sum(dim=1).float().mean()
    if trajectory.num_local_fields and bool(trajectory.local_mask.any()):
        active = trajectory.local_mask.to(trajectory.local_uncertainty.dtype)
        losses["local_uncertainty_mean"] = (
            trajectory.local_uncertainty * active
        ).sum() / active.sum().clamp_min(1.0)
        losses["center_density_uncertainty_correlation"] = _masked_pearson(
            trajectory.context_density,
            1.0 - prepared["retrieval"][..., 0].clamp(0.0, 1.0),
            mask,
        )
        if trajectory.local_part_gates is not None:
            gate_mask = active.unsqueeze(-1)
            gate_count = gate_mask.sum().clamp_min(1.0)
            part_names = ("body", "left_hand", "right_hand", "face")
            for part_index, part_name in enumerate(part_names):
                part_gates = trajectory.local_part_gates[..., part_index]
                mean_gate = (part_gates * active).sum() / gate_count
                losses[f"local_gate_mean_{part_name}"] = mean_gate
                losses[f"local_gate_std_{part_name}"] = torch.sqrt(
                    (
                        (part_gates - mean_gate).square()
                        * active
                    ).sum()
                    / gate_count
                )
    else:
        losses["local_uncertainty_mean"] = local_axis.new_tensor(0.0)
        losses["center_density_uncertainty_correlation"] = local_axis.new_tensor(0.0)
    losses["loss_total"] = total
    return total, losses, prepared


def optimizer_step(total, model, optimizer, cfg, loss_divisor=1.0):
    (total / float(loss_divisor)).backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0))
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


_LOCAL_PARAMETER_PREFIXES = (
    "hypernetwork.density_head.",
    "hypernetwork.local_context.",
    "hypernetwork.local_head.",
    "hypernetwork.local_gate_head.",
    "part_local_fields.",
)


_STAGE2_WARM_START_PREFIXES = (
    "hypernetwork.local_gate_head.",
    "part_local_fields.",
)


def load_warm_start_state(model, state_dict):
    """Load a global/Stage 1 checkpoint into an optional Stage 2 model."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not any(name.startswith(prefix) for prefix in _STAGE2_WARM_START_PREFIXES)
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Warm-start checkpoint is incompatible: "
            f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
        )
    return incompatible


def build_optimizer(model, cfg):
    train_cfg = cfg.get("train", {})
    base_lr = float(train_cfg.get("lr", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    use_groups = any(
        key in train_cfg
        for key in (
            "local_warmup_epochs",
            "warmup_local_lr",
            "joint_local_lr",
            "joint_global_lr",
        )
    )
    if not use_groups:
        return torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    local_parameters = []
    global_parameters = []
    for name, parameter in unwrap_model(model).named_parameters():
        if any(name.startswith(prefix) for prefix in _LOCAL_PARAMETER_PREFIXES):
            local_parameters.append(parameter)
        else:
            global_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": global_parameters, "lr": base_lr, "group_name": "global"},
            {"params": local_parameters, "lr": base_lr, "group_name": "local"},
        ],
        weight_decay=weight_decay,
    )


def configure_optimizer_epoch(optimizer, cfg, epoch):
    train_cfg = cfg.get("train", {})
    base_lr = float(train_cfg.get("lr", 2e-4))
    warmup_epochs = max(int(train_cfg.get("local_warmup_epochs", 0)), 0)
    in_warmup = warmup_epochs > 0 and int(epoch) <= warmup_epochs
    assigned = {}
    for group in optimizer.param_groups:
        name = group.get("group_name")
        if name == "global":
            key = "warmup_global_lr" if in_warmup else "joint_global_lr"
            default = 0.0 if in_warmup else base_lr
            group["lr"] = float(train_cfg.get(key, default))
            assigned[name] = group["lr"]
        elif name == "local":
            key = "warmup_local_lr" if in_warmup else "joint_local_lr"
            group["lr"] = float(train_cfg.get(key, base_lr))
            assigned[name] = group["lr"]
        else:
            assigned["all"] = float(group["lr"])
    return assigned


def slice_batch(batch, start, end):
    """Slice a collated CPU batch without moving unused samples to the GPU."""

    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:end]
        elif isinstance(value, dict):
            sliced[key] = {
                nested_key: nested_value[start:end]
                for nested_key, nested_value in value.items()
            }
        elif isinstance(value, tuple):
            sliced[key] = value[start:end]
        elif isinstance(value, list):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _bounded_microbatch_size(batch, sample_budget=0, frame_budget=0):
    logical_size = len(batch["name"])
    microbatch_size = logical_size
    if int(sample_budget) > 0:
        microbatch_size = min(microbatch_size, int(sample_budget))
    if int(frame_budget) > 0:
        padded_frames = int(batch["motion"].shape[1])
        microbatch_size = min(
            microbatch_size,
            max(int(frame_budget) // max(padded_frames, 1), 1),
        )
    return microbatch_size


def memory_microbatch_size(batch, cfg):
    """Choose a GPU training microbatch within sample and frame budgets."""

    train_cfg = cfg.get("train", {})
    return _bounded_microbatch_size(
        batch,
        sample_budget=train_cfg.get("max_samples_per_memory_batch", 0),
        frame_budget=train_cfg.get("max_frames_per_memory_batch", 0),
    )


def validation_microbatch_size(batch, cfg):
    """Choose a validation microbatch; default to one sample for dense JVPs."""

    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    return _bounded_microbatch_size(
        batch,
        sample_budget=eval_cfg.get("max_samples_per_memory_batch", 1),
        frame_budget=eval_cfg.get(
            "max_frames_per_memory_batch",
            train_cfg.get("max_frames_per_memory_batch", 0),
        ),
    )


WANDB_BATCH_METRICS = (
    "loss_total",
    "loss_endpoint",
    "loss_joint",
    "loss_hand_relative",
    "loss_path",
    "loss_duration",
    "loss_analytic_fk_jerk",
    "analytic_fk_jerk_ratio",
    "residual_rms",
    "global_residual_rms",
    "local_residual_rms",
)


def wandb_train_batch_payload(
    metrics,
    *,
    epoch,
    optimizer_step,
    logical_batch,
    logical_batches,
):
    payload = {
        "train/optimizer_step": int(optimizer_step),
        "train/batch/epoch": int(epoch),
        "train/batch/logical_batch": int(logical_batch),
        "train/batch/logical_batches": int(logical_batches),
    }
    for name in WANDB_BATCH_METRICS:
        if name in metrics:
            payload[f"train/batch/{name}"] = float(metrics[name])
    return payload


def wandb_train_epoch_payload(row):
    payload = {
        "train/epoch_step": int(row["epoch"]),
        "train/epoch/global_step": int(row["global_step"]),
        "train/epoch/elapsed_sec": float(row["elapsed_sec"]),
    }
    for name, value in row.items():
        if name.startswith("train_"):
            payload[f"train/epoch/{name.removeprefix('train_')}"] = value
        elif name.startswith("lr_"):
            payload[f"optimizer/{name.removeprefix('lr_')}_lr"] = value
    return payload


def wandb_validation_pending_payload(epoch, global_step):
    return {
        "validation/epoch_step": int(epoch),
        "validation/global_step": int(global_step),
        "validation/pending": 1.0,
    }


def wandb_validation_payload(row):
    payload = {
        "validation/epoch_step": int(row["epoch"]),
        "validation/global_step": int(row["global_step"]),
        "validation/pending": float(row.get("validation_pending", 0.0)),
    }
    for name, value in row.items():
        if name.startswith("val_"):
            payload[f"validation/{name.removeprefix('val_')}"] = value
        elif name.startswith("selection_"):
            payload[f"validation/selection/{name.removeprefix('selection_')}"] = value
    return payload


def configure_wandb_metrics(wandb_run):
    definitions = (
        ("train/optimizer_step", None),
        ("train/batch/*", "train/optimizer_step"),
        ("train/epoch_step", None),
        ("train/epoch/*", "train/epoch_step"),
        ("optimizer/*", "train/epoch_step"),
        ("validation/epoch_step", None),
        ("validation/*", "validation/epoch_step"),
    )
    for name, step_metric in definitions:
        kwargs = {"step_metric": step_metric} if step_metric else {}
        wandb_run.define_metric(name, **kwargs)


def run_train_epoch(
    model,
    fk,
    text_encoder,
    provider,
    loader,
    dataset,
    optimizer,
    cfg,
    device,
    epoch,
    dist_info,
    wandb_run=None,
    global_step_start=0,
):
    model.train()
    average = ScalarAverager()
    max_batches = int(cfg.get("train", {}).get("max_train_batches", 0))
    accumulation = max(int(cfg.get("train", {}).get("accumulation_steps", 1)), 1)
    log_every = max(int(cfg.get("train", {}).get("log_every_batches", 10)), 0)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}",
        leave=False,
        disable=not dist_info["is_main"],
    )
    steps = 0
    pending = 0
    step_average = ScalarAverager()
    last_batch_index = 0
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= max_batches:
            break
        logical_size = len(batch["name"])
        microbatch_size = memory_microbatch_size(batch, cfg)
        batch_losses = {}
        for start in range(0, logical_size, microbatch_size):
            end = min(start + microbatch_size, logical_size)
            microbatch = move_batch_to_device(slice_batch(batch, start, end), device)
            total, losses, _prepared = compute_batch_losses(
                model,
                fk,
                text_encoder,
                provider,
                microbatch,
                dataset,
                cfg,
                device,
                epoch=epoch,
                training=True,
            )
            microbatch_count = end - start
            gradient_weight = microbatch_count / logical_size
            (total * gradient_weight / accumulation).backward()
            float_losses = tensor_dict_to_float(losses)
            average.update(float_losses, n=microbatch_count, prefix="train")
            for name, value in float_losses.items():
                batch_losses[name] = (
                    batch_losses.get(name, 0.0) + value * gradient_weight
                )
        pending += 1
        last_batch_index = batch_index + 1
        step_average.update(batch_losses, n=logical_size)
        if pending == accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0))
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
            if wandb_run is not None:
                wandb_run.log(
                    wandb_train_batch_payload(
                        step_average.mean(),
                        epoch=epoch,
                        optimizer_step=global_step_start + steps,
                        logical_batch=last_batch_index,
                        logical_batches=len(loader),
                    )
                )
            step_average = ScalarAverager()
        if dist_info["is_main"]:
            progress.set_postfix(
                loss=f"{batch_losses['loss_total']:.4f}",
                residual=f"{batch_losses['residual_rms']:.4f}",
            )
            if log_every and (batch_index + 1) % log_every == 0:
                rank_zero_print(
                    dist_info,
                    f"epoch={epoch} logical_batch={batch_index + 1}/{len(loader)} "
                    f"logical_size={logical_size} memory_microbatch={microbatch_size} "
                    f"max_frames={int(batch['motion'].shape[1])} "
                    f"loss={batch_losses['loss_total']:.6f} "
                    f"residual_rms={batch_losses['residual_rms']:.6f}",
                )
    if pending:
        correction = accumulation / pending
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0))
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        if wandb_run is not None:
            wandb_run.log(
                wandb_train_batch_payload(
                    step_average.mean(),
                    epoch=epoch,
                    optimizer_step=global_step_start + steps,
                    logical_batch=last_batch_index,
                    logical_batches=len(loader),
                )
            )
    return average.mean(), steps


def deterministic_validation_dynamics(
    model,
    fk,
    trajectory,
    target_parts,
    lengths,
    duration_seconds,
    cfg,
):
    validation_cfg = cfg.get("validation_dynamics", {})
    if not bool(validation_cfg.get("enabled", False)):
        return {}

    # Validation differentiates only with respect to query time. Freezing model
    # parameters avoids retaining a large, unused parameter-gradient graph.
    parameters = list(model.parameters()) + list(fk.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.enable_grad():
            _total, metrics = analytic_fk_dynamics_losses(
                model,
                trajectory.detach(),
                fk,
                target_parts,
                lengths,
                duration_seconds,
                weights={
                    "lambda_analytic_fk_vel": 1.0,
                    "lambda_analytic_fk_acc": 1.0,
                    "lambda_analytic_fk_jerk": 1.0,
                    "lambda_analytic_fk_jerk_reg": 1.0,
                },
                query_count=int(validation_cfg.get("query_count", 32)),
                hand_weight=float(cfg.get("loss", {}).get("hand_weight", 5.0)),
                smooth_kernel=int(validation_cfg.get("smooth_kernel", 7)),
                smooth_sigma=float(validation_cfg.get("smooth_sigma", 1.5)),
                jerk_target_ratio=float(validation_cfg.get("jerk_target_ratio", 0.75)),
                randomize_queries=False,
            )
        return metrics
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


@torch.no_grad()
def evaluate_microbatch(
    model,
    fk,
    text_encoder,
    provider,
    batch,
    dataset,
    cfg,
    device,
    epoch=1,
):
    batch = move_batch_to_device(batch, device)
    metrics = {}

    def add_metrics(prefix, values):
        for name, value in tensor_dict_to_float(values).items():
            metrics[f"{prefix}_{name}"] = value

    # Dense analytic dynamics are evaluated separately below. Keeping them out
    # of the endpoint pass avoids constructing the same high-order graph twice.
    analytic_cfg = cfg.get("analytic_dynamics", {})
    disabled = dict(analytic_cfg)
    for key in (
        "lambda_analytic_fk_vel",
        "lambda_analytic_fk_acc",
        "lambda_analytic_fk_jerk",
        "lambda_analytic_fk_jerk_reg",
    ):
        disabled[key] = 0.0
    cfg["analytic_dynamics"] = disabled
    try:
        _total, losses, prepared = compute_batch_losses(
            model,
            fk,
            text_encoder,
            provider,
            batch,
            dataset,
            cfg,
            device,
            epoch=epoch,
            training=False,
        )
    finally:
        cfg["analytic_dynamics"] = analytic_cfg
    add_metrics("pred", losses)

    validation_cfg = cfg.get("validation_dynamics", {})
    derivative_duration = (
        batch["duration"]
        if bool(validation_cfg.get("duration_teacher_forcing", True))
        else prepared["outputs"]["trajectory"].duration_seconds
    )
    dense_dynamics = deterministic_validation_dynamics(
        model,
        fk,
        prepared["outputs"]["trajectory"],
        batch.get("target_parts"),
        batch["length"],
        derivative_duration,
        cfg,
    )
    add_metrics("pred_dense", dense_dynamics)

    target = prepared["target"]
    for label, values in (
        ("prior", prepared["outputs"]["prior"]),
        ("scaffold", prepared["adapter_context"]),
    ):
        _baseline_total, baseline_losses = endpoint_losses(
            values,
            target,
            batch["mask"],
            batch["length"],
            batch.get("target_parts"),
            fk=fk,
            weights=cfg.get("loss", {}),
            hand_weight=float(cfg.get("loss", {}).get("hand_weight", 5.0)),
            fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
        )
        add_metrics(label, baseline_losses)
    if bool(cfg.get("eval", {}).get("residual_branch_ablation", False)):
        trajectory = prepared["outputs"]["trajectory"]
        for label, query_options in (
            ("global_only", {"include_local_residual": False}),
            ("local_only", {"include_global_residual": False}),
        ):
            values = model.query_trajectory(
                trajectory,
                prepared["tau"],
                query_mask=batch["mask"],
                **query_options,
            )
            _ablation_total, ablation_losses = endpoint_losses(
                values,
                target,
                batch["mask"],
                batch["length"],
                batch.get("target_parts"),
                fk=fk,
                weights=cfg.get("loss", {}),
                hand_weight=float(cfg.get("loss", {}).get("hand_weight", 5.0)),
                fk_chunk_size=int(cfg.get("metrics", {}).get("fk_batch_size", 128)),
            )
            add_metrics(label, ablation_losses)
    return metrics


@torch.no_grad()
def evaluate(
    model,
    fk,
    text_encoder,
    provider,
    loader,
    dataset,
    cfg,
    device,
    epoch=1,
    max_batches=0,
    show_progress=True,
):
    model.eval()
    average = ScalarAverager()
    progress = tqdm(loader, desc="val", leave=False, disable=not show_progress)
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= int(max_batches):
            break
        logical_size = len(batch["name"])
        microbatch_size = validation_microbatch_size(batch, cfg)
        microbatch_count = (logical_size + microbatch_size - 1) // microbatch_size
        for microbatch_index, start in enumerate(
            range(0, logical_size, microbatch_size), start=1
        ):
            end = min(start + microbatch_size, logical_size)
            metrics = evaluate_microbatch(
                model,
                fk,
                text_encoder,
                provider,
                slice_batch(batch, start, end),
                dataset,
                cfg,
                device,
                epoch=epoch,
            )
            average.update(metrics, n=end - start)
            if show_progress:
                progress.set_postfix(
                    logical=f"{batch_index + 1}/{len(loader)}",
                    micro=f"{microbatch_index}/{microbatch_count}",
                )
        if (
            bool(cfg.get("eval", {}).get("empty_cache_between_batches", True))
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.empty_cache()
    return average.mean()


def _selection_value(metrics, name, specification=None):
    specification = dict(specification or {})
    value = float(metrics.get(name, float("inf")))
    baseline_name = specification.get("relative_to")
    baseline = None
    if baseline_name:
        baseline = float(metrics.get(str(baseline_name), float("inf")))
        value = value - baseline
    return value, baseline_name, baseline


def selection_diagnostics(metrics, cfg, return_details=False):
    configured = cfg.get("selection", {}).get("weights", {})
    contributions = {}
    if not configured:
        score = float(metrics.get("pred_loss_endpoint", float("inf")))
        contributions["pred_loss_endpoint"] = score
    else:
        score = 0.0
        for name, weight_specification in configured.items():
            if isinstance(weight_specification, dict):
                weight_specification = dict(weight_specification)
                weight = float(weight_specification.get("weight", 1.0))
                value, _baseline_name, _baseline = _selection_value(
                    metrics, name, weight_specification
                )
            else:
                weight = float(weight_specification)
                value = float(metrics.get(name, 0.0))
            contribution = weight * value
            contributions[name] = contribution
            score += contribution

    selection_cfg = cfg.get("selection", {})
    total_violation = 0.0
    constraint_rows = []
    rejection_reasons = []
    for name, specification in selection_cfg.get("constraints", {}).items():
        specification = dict(specification)
        value, baseline_name, baseline = _selection_value(metrics, name, specification)
        if not math.isfinite(value):
            total_violation = float("inf")
            rejection_reasons.append(f"{name} is missing or non-finite")
            break
        default_scale = max(
            abs(float(specification.get("max", specification.get("min", 1.0)))),
            1e-8,
        )
        scale = max(float(specification.get("scale", default_scale)), 1e-8)
        violation = 0.0
        if "max" in specification:
            violation += max(value - float(specification["max"]), 0.0) / scale
        if "min" in specification:
            violation += max(float(specification["min"]) - value, 0.0) / scale
        total_violation += violation
        if violation > 0:
            relation = f" relative to {baseline_name}" if baseline_name else ""
            rejection_reasons.append(
                f"{name}{relation} value={value:.6g} violates "
                f"min={specification.get('min')} max={specification.get('max')}"
            )
        constraint_rows.append(
            {
                "metric": name,
                "value": value,
                "baseline_metric": baseline_name,
                "baseline_value": baseline,
                "violation": violation,
            }
        )

    score += float(selection_cfg.get("constraint_penalty", 0.0)) * total_violation
    feasible = math.isfinite(total_violation) and total_violation <= 1e-12
    result = (score, total_violation, feasible)
    if not return_details:
        return result
    return (*result, {
        "contributions": contributions,
        "constraints": constraint_rows,
        "rejection_reasons": rejection_reasons,
    })


def selection_score(metrics, cfg):
    return selection_diagnostics(metrics, cfg)[0]


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
            "model_type": "continuous_trajectory_field",
            "trajectory_contract_version": 1,
        },
        path,
    )


def init_wandb(args, cfg, dist_info, out_dir, start_epoch=1, initial_global_step=0):
    if not args.wandb or not dist_info["is_main"]:
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
            "experiment": cfg.get("experiment_name", "continuous_trajectory_field"),
            "config": cfg,
            "world_size": dist_info.get("world_size", 1),
            "output_dir": str(out_dir),
            "launch": {
                "resume_checkpoint": str(args.resume) if args.resume else None,
                "warm_start_checkpoint": (
                    str(args.warm_start) if args.warm_start else None
                ),
                "start_epoch": int(start_epoch),
                "initial_global_step": int(initial_global_step),
            },
        },
        "dir": str(Path(os.environ.get("WANDB_DIR", out_dir))),
    }
    if args.wandb_id:
        kwargs["id"] = args.wandb_id
    if args.wandb_resume:
        kwargs["resume"] = args.wandb_resume
    run = wandb.init(**kwargs)
    configure_wandb_metrics(run)
    return run


def main():
    args = parse_args()
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("--resume and --warm_start are mutually exclusive")
    cfg = apply_overrides(load_config(args.config), args)
    dist_info = setup_distributed(args)
    set_seed(int(cfg.get("seed", 1234)) + int(dist_info.get("rank", 0)))
    device = resolve_distributed_device(cfg.get("device", "auto"), dist_info)
    text_device = torch.device(cfg.get("text", {}).get("device", "cpu"))
    out_dir = Path(cfg["output"]["out_dir"])
    if dist_info["is_main"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )

    data_cfg = cfg.get("data", {})
    train_dataset, train_loader, train_sampler = make_loader(
        cfg,
        data_cfg.get("train_split", "train"),
        limit=data_cfg.get("limit_train", 0),
        shuffle=True,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    val_dataset, val_loader, val_sampler = make_loader(
        cfg,
        data_cfg.get("val_split", "val"),
        limit=data_cfg.get("limit_val", 0),
        shuffle=False,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
        # Validation first iterates this loader after CUDA models and the
        # online scaffold stack have been initialized. Forking workers at that
        # point can inherit locked CUDA/transformer threads and deadlock before
        # the first batch, so use a synchronous loader by default.
        num_workers=int(cfg.get("eval", {}).get("num_workers", 0)),
    )
    rank_zero_print(
        dist_info,
        f"Loaded continuous trajectory datasets: train={len(train_dataset)} "
        f"val={len(val_dataset)} world_size={dist_info['world_size']} "
        f"batch_per_rank={cfg.get('train', {}).get('batch_size', 1)} "
        f"train_sampler={type(train_sampler).__name__ if train_sampler is not None else 'none'}",
    )
    text_encoder = build_text_encoder(cfg, text_device)
    provider = ScaffoldProvider(cfg, train_dataset, device)
    retrieval_bank = validate_train_only_retrieval_bank(cfg, provider)
    rank_zero_print(dist_info, f"Retrieval bank: {json.dumps(retrieval_bank, sort_keys=True)}")
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})
    start_epoch = 1
    global_step = 0
    optimizer_state = None
    best_score = float("inf")
    best_infeasible_score = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer_state = checkpoint.get("optimizer")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        checkpoint_metrics = checkpoint.get("metrics", {})
        checkpoint_score = float(checkpoint_metrics.get("selection_score", float("inf")))
        if bool(checkpoint_metrics.get("selection_feasible", False)):
            best_score = checkpoint_score
        else:
            best_infeasible_score = checkpoint_score
    elif args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu")
        incompatible = load_warm_start_state(model, checkpoint["model"])
        reset_local = bool(
            args.reset_local_branch
            or train_cfg.get("reset_local_branch_on_warm_start", False)
        )
        if reset_local:
            model.reset_local_branch()
        rank_zero_print(
            dist_info,
            f"Warm-started model weights from {args.warm_start} "
            f"(checkpoint_epoch={checkpoint.get('epoch', 0)}, "
            f"reset_local_branch={reset_local}, "
            f"new_parameters={len(incompatible.missing_keys)}); "
            "optimizer and epoch reset.",
        )

    model = wrap_model(model, dist_info, device)
    optimizer = build_optimizer(model, cfg)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    epochs = int(train_cfg.get("epochs", 20))
    val_every = max(int(train_cfg.get("val_every", 1)), 1)
    save_every = max(int(train_cfg.get("save_every", 1)), 1)
    start_time = time.time()
    wandb_run = init_wandb(
        args,
        cfg,
        dist_info,
        out_dir,
        start_epoch=start_epoch,
        initial_global_step=global_step,
    )
    if wandb_run is not None:
        rank_zero_print(
            dist_info,
            f"W&B run: id={wandb_run.id} url={wandb_run.url}",
        )
    for epoch in range(start_epoch, epochs + 1):
        optimizer_lrs = configure_optimizer_epoch(optimizer, cfg, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        train_metrics, optimizer_steps = run_train_epoch(
            model,
            fk,
            text_encoder,
            provider,
            train_loader,
            train_dataset,
            optimizer,
            cfg,
            device,
            epoch,
            dist_info,
            wandb_run=wandb_run,
            global_step_start=global_step,
        )
        global_step += optimizer_steps
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
            **{f"lr_{name}": value for name, value in optimizer_lrs.items()},
        }
        row.update(distributed_mean_scalars(train_metrics, device, dist_info))
        if dist_info["is_main"] and wandb_run is not None:
            wandb_run.log(wandb_train_epoch_payload(row))

        validation_due = epoch % val_every == 0
        if validation_due:
            # Persist the completed training epoch before entering the much
            # heavier dense-JVP validation path. Successful validation
            # overwrites this recovery snapshot with complete metrics below.
            row["validation_pending"] = 1.0
            if dist_info["is_main"]:
                save_checkpoint(
                    out_dir / "checkpoints" / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                )
            barrier(dist_info)
            if dist_info["is_main"] and wandb_run is not None:
                wandb_run.log(wandb_validation_pending_payload(epoch, global_step))
            val_metrics = evaluate(
                unwrap_model(model),
                fk,
                text_encoder,
                provider,
                val_loader,
                val_dataset,
                cfg,
                device,
                epoch=epoch,
                max_batches=int(cfg.get("eval", {}).get("max_batches", 0)),
                show_progress=dist_info["is_main"],
            )
            val_metrics = distributed_mean_scalars(val_metrics, device, dist_info)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["validation_pending"] = 0.0
            (
                score,
                constraint_violation,
                selection_feasible,
                selection_details,
            ) = selection_diagnostics(val_metrics, cfg, return_details=True)
            row["selection_score"] = score
            row["selection_constraint_violation"] = constraint_violation
            row["selection_feasible"] = float(selection_feasible)
            row["selection_rejection_reasons"] = "; ".join(
                selection_details["rejection_reasons"]
            )
            if dist_info["is_main"] and wandb_run is not None:
                wandb_run.log(wandb_validation_payload(row))
            if dist_info["is_main"] and bool(
                cfg.get("selection", {}).get("keep_validation_checkpoints", False)
            ):
                save_checkpoint(
                    out_dir / "checkpoints" / f"epoch{epoch:04d}.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                )
            if selection_feasible and score < best_score:
                best_score = score
                if dist_info["is_main"]:
                    save_checkpoint(
                        out_dir / "checkpoints" / "best.pt",
                        unwrap_model(model),
                        optimizer,
                        epoch,
                        global_step,
                        cfg,
                        row,
                    )
            elif not selection_feasible and score < best_infeasible_score:
                best_infeasible_score = score
                if dist_info["is_main"]:
                    save_checkpoint(
                        out_dir / "checkpoints" / "best_infeasible.pt",
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
        barrier(dist_info)

    has_feasible_checkpoint = math.isfinite(best_score)
    if dist_info["is_main"]:
        selection_summary = {
            "has_feasible_checkpoint": has_feasible_checkpoint,
            "best_feasible_score": best_score if has_feasible_checkpoint else None,
            "best_infeasible_score": (
                best_infeasible_score
                if math.isfinite(best_infeasible_score)
                else None
            ),
            "required": bool(cfg.get("selection", {}).get("require_feasible", False)),
        }
        (out_dir / "selection_summary.json").write_text(
            json.dumps(selection_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    barrier(dist_info)
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(dist_info)
    if bool(cfg.get("selection", {}).get("require_feasible", False)) and not has_feasible_checkpoint:
        raise RuntimeError(
            "No validation checkpoint satisfied the configured selection constraints; "
            "see selection_summary.json and best_infeasible.pt."
        )


if __name__ == "__main__":
    main()
