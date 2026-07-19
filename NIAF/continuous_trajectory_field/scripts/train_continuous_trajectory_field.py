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
    return cfg


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def native_query_times(lengths, max_len, device, dtype):
    unit = normalized_time_grid(lengths, max_len=max_len, device=device, dtype=dtype)
    return 2.0 * unit.squeeze(-1) - 1.0


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
    losses["active_local_fields"] = trajectory.local_mask.sum(dim=1).float().mean()
    losses["loss_total"] = total
    return total, losses, prepared


def optimizer_step(total, model, optimizer, cfg, loss_divisor=1.0):
    (total / float(loss_divisor)).backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0))
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


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
):
    model.train()
    average = ScalarAverager()
    max_batches = int(cfg.get("train", {}).get("max_train_batches", 0))
    accumulation = max(int(cfg.get("train", {}).get("accumulation_steps", 1)), 1)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}",
        leave=False,
        disable=not dist_info["is_main"],
    )
    steps = 0
    pending = 0
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        total, losses, _prepared = compute_batch_losses(
            model,
            fk,
            text_encoder,
            provider,
            batch,
            dataset,
            cfg,
            device,
            epoch=epoch,
            training=True,
        )
        (total / accumulation).backward()
        pending += 1
        if pending == accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0))
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
        float_losses = tensor_dict_to_float(losses)
        average.update(float_losses, n=len(batch["name"]), prefix="train")
        if dist_info["is_main"]:
            progress.set_postfix(
                loss=f"{float_losses['loss_total']:.4f}",
                residual=f"{float_losses['residual_rms']:.4f}",
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
    for batch_index, batch in enumerate(
        tqdm(loader, desc="val", leave=False, disable=not show_progress)
    ):
        if max_batches and batch_index >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        # Analytic JVP losses require autograd and are exercised in training and
        # dedicated evaluation; endpoint validation stays inexpensive.
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
        average.update(tensor_dict_to_float(losses), n=len(batch["name"]), prefix="pred")

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
        average.update(
            tensor_dict_to_float(dense_dynamics),
            n=len(batch["name"]),
            prefix="pred_dense",
        )

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
            average.update(
                tensor_dict_to_float(baseline_losses),
                n=len(batch["name"]),
                prefix=label,
            )
    return average.mean()


def selection_diagnostics(metrics, cfg):
    configured = cfg.get("selection", {}).get("weights", {})
    if not configured:
        score = float(metrics.get("pred_loss_endpoint", float("inf")))
    else:
        score = 0.0
        for name, weight in configured.items():
            score += float(weight) * float(metrics.get(name, 0.0))

    selection_cfg = cfg.get("selection", {})
    total_violation = 0.0
    for name, specification in selection_cfg.get("constraints", {}).items():
        specification = dict(specification)
        value = float(metrics.get(name, float("inf")))
        if not math.isfinite(value):
            total_violation = float("inf")
            break
        default_scale = max(
            abs(float(specification.get("max", specification.get("min", 1.0)))),
            1e-8,
        )
        scale = max(float(specification.get("scale", default_scale)), 1e-8)
        if "max" in specification:
            total_violation += max(value - float(specification["max"]), 0.0) / scale
        if "min" in specification:
            total_violation += max(float(specification["min"]) - value, 0.0) / scale

    score += float(selection_cfg.get("constraint_penalty", 0.0)) * total_violation
    feasible = math.isfinite(total_violation) and total_violation <= 1e-12
    return score, total_violation, feasible


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


def init_wandb(args, cfg, dist_info, out_dir):
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
    )
    rank_zero_print(
        dist_info,
        f"Loaded continuous trajectory datasets: train={len(train_dataset)} "
        f"val={len(val_dataset)} world_size={dist_info['world_size']} "
        f"batch_per_rank={cfg.get('train', {}).get('batch_size', 1)}",
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
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer_state = checkpoint.get("optimizer")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_score = float(checkpoint.get("metrics", {}).get("selection_score", best_score))

    model = wrap_model(model, dist_info, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    epochs = int(train_cfg.get("epochs", 20))
    val_every = max(int(train_cfg.get("val_every", 1)), 1)
    save_every = max(int(train_cfg.get("save_every", 1)), 1)
    start_time = time.time()
    wandb_run = init_wandb(args, cfg, dist_info, out_dir)
    for epoch in range(start_epoch, epochs + 1):
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
        )
        global_step += optimizer_steps
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
        }
        row.update(distributed_mean_scalars(train_metrics, device, dist_info))

        if epoch % val_every == 0:
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
            score, constraint_violation, selection_feasible = selection_diagnostics(
                val_metrics, cfg
            )
            row["selection_score"] = score
            row["selection_constraint_violation"] = constraint_violation
            row["selection_feasible"] = float(selection_feasible)
            if dist_info["is_main"] and score < best_score:
                best_score = score
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
