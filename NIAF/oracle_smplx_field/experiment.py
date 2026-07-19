from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from flow.smplx_features import COMPACT6D_DIM
from NIAF.oracle_smplx_field.data import (
    build_dataset,
    compute_joint_parts,
    load_oracle_sample,
    save_motion_npz,
    select_pilot_indices,
)
from NIAF.oracle_smplx_field.geometry.smplx_fk import DifferentiableSMPLXForward
from NIAF.oracle_smplx_field.losses import LOSS_SCHEDULES, dense_physical_loss, pose_field_loss
from NIAF.oracle_smplx_field.metrics import evaluate_pose_prediction, summarize_rows, write_jsonl
from NIAF.oracle_smplx_field.models import DirectSirenPoseField, ResidualSirenPoseField, build_baseline
from NIAF.oracle_smplx_field.time_parameterization import make_time_grid


MODEL_PRESETS = {
    "linear_rot6d": {"type": "linear_rot6d"},
    "slerp": {"type": "slerp"},
    "cubic": {"type": "cubic_rot6d"},
    "direct_siren": {"type": "direct_siren", "hidden": 256, "depth": 3},
    "residual_siren": {"type": "residual_siren", "hidden": 256, "depth": 3, "scaffold": "linear_rot6d"},
    "residual_siren_linear": {"type": "residual_siren", "hidden": 256, "depth": 3, "scaffold": "linear_rot6d"},
    "residual_siren_slerp": {"type": "residual_siren", "hidden": 256, "depth": 3, "scaffold": "slerp"},
}


def set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def is_baseline(model_spec):
    return model_spec["type"] in {"linear_rot6d", "slerp", "cubic_rot6d"}


def train_eval_indices(length, fit_mode):
    length = int(length)
    if fit_mode == "fit_all":
        idx = torch.arange(length, dtype=torch.long)
        return idx, idx
    if fit_mode == "even_odd":
        train = torch.arange(0, length, 2, dtype=torch.long)
        test = torch.arange(1, length, 2, dtype=torch.long)
        return train, test
    if fit_mode == "odd_even":
        train = torch.arange(1, length, 2, dtype=torch.long)
        if len(train) == 0:
            train = torch.arange(length, dtype=torch.long)
        test = torch.arange(0, length, 2, dtype=torch.long)
        return train, test
    if fit_mode.startswith("stride"):
        stride = 4
        try:
            stride = int(str(fit_mode).rsplit("_", 1)[1])
        except Exception:
            stride = 4
        train = list(range(0, length, stride))
        if train[-1] != length - 1:
            train.append(length - 1)
        train = torch.tensor(sorted(set(train)), dtype=torch.long)
        mask = torch.ones(length, dtype=torch.bool)
        mask[train] = False
        test = torch.arange(length, dtype=torch.long)[mask]
        return train, test
    if fit_mode.startswith("block_middle"):
        pct = 25
        try:
            pct = int(str(fit_mode).rsplit("_", 1)[1])
        except Exception:
            pct = 25
        block = max(1, int(round(length * max(min(pct, 90), 1) / 100.0)))
        start = max(1, (length - block) // 2)
        end = min(length - 1, start + block)
        if end <= start:
            end = min(length - 1, start + 1)
        test = torch.arange(start, end, dtype=torch.long)
        mask = torch.ones(length, dtype=torch.bool)
        mask[test] = False
        train = torch.arange(length, dtype=torch.long)[mask]
        return train, test
    if fit_mode.startswith("keyframe_sparse"):
        count = 8
        try:
            count = int(str(fit_mode).rsplit("_", 1)[1])
        except Exception:
            count = 8
        count = max(2, min(count, length))
        anchors = torch.linspace(0, length - 1, count).round().long().unique(sorted=True)
        if anchors[0].item() != 0:
            anchors = torch.cat([torch.tensor([0], dtype=torch.long), anchors])
        if anchors[-1].item() != length - 1:
            anchors = torch.cat([anchors, torch.tensor([length - 1], dtype=torch.long)])
        mask = torch.ones(length, dtype=torch.bool)
        mask[anchors] = False
        test = torch.arange(length, dtype=torch.long)[mask]
        return anchors, test
    raise ValueError(f"Unsupported fit_mode={fit_mode!r}")


def build_model(model_spec, tau_train, x_train, cfg, tau_all=None, x_all=None, train_idx=None):
    model_type = model_spec["type"]
    model_cfg = cfg.get("model", {})
    if model_type == "direct_siren":
        return DirectSirenPoseField(
            output_dim=COMPACT6D_DIM,
            hidden=int(model_spec.get("hidden", model_cfg.get("hidden", 256))),
            depth=int(model_spec.get("depth", model_cfg.get("depth", 3))),
            omega0=float(model_spec.get("omega0", model_cfg.get("omega0_first", 20.0))),
            omega=float(model_spec.get("omega", model_cfg.get("omega0_hidden", 1.0))),
        )
    if model_type == "residual_siren":
        scaffold = str(model_spec.get("scaffold", model_cfg.get("anchor_scaffold", "linear_rot6d"))).lower()
        knot_tau = tau_train
        knot_x = x_train
        if scaffold in {"slerp", "rotation_slerp"}:
            if tau_all is None or x_all is None or train_idx is None:
                raise ValueError("Slerp residual scaffold requires tau_all, x_all, and train_idx")
            baseline = build_baseline("slerp")
            baseline.fit(tau_all[train_idx], x_all[train_idx])
            knot_tau = tau_all.detach().cpu()
            knot_x = baseline.predict(knot_tau, device=torch.device("cpu"), dtype=torch.float32).detach().cpu()
        elif scaffold not in {"linear", "linear_rot6d", "linear_interp"}:
            raise ValueError(f"Unsupported residual scaffold={scaffold!r}")
        return ResidualSirenPoseField(
            knot_tau,
            knot_x,
            output_dim=COMPACT6D_DIM,
            hidden=int(model_spec.get("hidden", model_cfg.get("hidden", 256))),
            depth=int(model_spec.get("depth", model_cfg.get("depth", 3))),
            omega0=float(model_spec.get("omega0", model_cfg.get("omega0_first", 20.0))),
            omega=float(model_spec.get("omega", model_cfg.get("omega0_hidden", 1.0))),
            residual_scale=float(model_cfg.get("residual_scale_init", 0.1)),
            learnable_scale=bool(model_cfg.get("residual_scale_learnable", True)),
        )
    raise ValueError(f"Unsupported neural model type={model_type!r}")


def fit_baseline(model_spec, tau_all, x_all, train_idx, device):
    baseline = build_baseline(model_spec["type"])
    baseline.fit(tau_all[train_idx], x_all[train_idx])
    pred = baseline.predict(tau_all, device=device, dtype=torch.float32)
    train_loss = torch.abs(pred[train_idx.to(device)] - x_all[train_idx.to(device)]).mean()
    return pred, {"train_loss": float(train_loss.detach().cpu().item())}, baseline


def index_parts(parts, idx, device):
    idx = idx.to(device)
    return {key: value[idx] for key, value in parts.items()}


def predict_parts_in_chunks(fk, pred_x, chunk_size=128):
    chunks = []
    for start in range(0, len(pred_x), chunk_size):
        end = min(start + chunk_size, len(pred_x))
        chunks.append(fk.parts_from_rot6d(pred_x[start:end]))
    out = {}
    for key in chunks[0].keys():
        out[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
    return out


def fit_neural(model, sample, tau_all, train_idx, schedule, cfg, fk, target_parts, device):
    fit_cfg = cfg["fit"]
    model = model.to(device)
    tau_all = tau_all.to(device=device, dtype=torch.float32)
    x_all = sample.x_rot6d.to(device=device, dtype=torch.float32)
    train_idx = train_idx.to(device)
    steps = int(fit_cfg.get("steps", 1000))
    batch_points = int(fit_cfg.get("batch_points", 128))
    hand_weight = float(cfg["loss"].get("hand_weight", 3.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fit_cfg.get("lr", 1e-4)),
        weight_decay=float(fit_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    grad_clip = float(fit_cfg.get("grad_clip", 1.0))
    rng = torch.Generator(device=device)
    rng.manual_seed(int(cfg.get("seed", 1234)) + int(sample.index))
    dense_points = int(fit_cfg.get("dense_points", 0))
    if dense_points <= 0:
        dense_points = int(fit_cfg.get("dense_multiplier", 0)) * int(sample.length)
    tau_dense = None
    if dense_points > 0 and (
        schedule.get("lambda_dense_joint_acc", 0.0) > 0 or schedule.get("lambda_dense_joint_jerk", 0.0) > 0
    ):
        dense_points = max(4, int(dense_points))
        tau_dense = torch.linspace(
            float(tau_all.min().detach().cpu().item()),
            float(tau_all.max().detach().cpu().item()),
            dense_points,
            device=device,
            dtype=torch.float32,
        )
    last = {}
    for _step in range(steps):
        if batch_points > 0 and batch_points < len(train_idx):
            local = torch.randperm(len(train_idx), generator=rng, device=device)[:batch_points]
            idx = train_idx[local]
        else:
            idx = train_idx
        optimizer.zero_grad(set_to_none=True)
        pred = model(tau_all[idx])
        target = x_all[idx]
        target_batch_parts = index_parts(target_parts, idx, device)
        loss, losses = pose_field_loss(
            pred,
            target,
            fk=fk,
            target_parts=target_batch_parts,
            schedule=schedule,
            hand_weight=hand_weight,
        )
        if hasattr(model, "residual_magnitude") and schedule.get("lambda_res", 0.0) > 0:
            res_loss = model.residual_magnitude(tau_all[idx])
            loss = loss + float(schedule["lambda_res"]) * res_loss
            losses["loss_res"] = res_loss
        if tau_dense is not None:
            dense_pred = model(tau_dense)
            physical_loss, physical_losses = dense_physical_loss(
                dense_pred,
                fk=fk,
                schedule=schedule,
                hand_weight=hand_weight,
            )
            loss = loss + physical_loss
            losses.update(physical_losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        last = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
        last["train_loss"] = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        pred_all = model(tau_all).detach()
    return pred_all, last, model


def expand_grid(cfg):
    grid = cfg["grid"]
    models = grid.get("models", ["residual_siren"])
    times = grid.get("time_modes", ["uniform"])
    fit_modes = grid.get("fit_modes", ["fit_all"])
    schedules = grid.get("loss_schedules", ["S1"])
    runs = []
    for model_id in models:
        model_spec = dict(MODEL_PRESETS.get(model_id, {"type": model_id}))
        schedule_iter = ["baseline"] if is_baseline(model_spec) else schedules
        for time_mode, fit_mode, schedule_id in itertools.product(times, fit_modes, schedule_iter):
            runs.append(
                {
                    "model_id": model_id,
                    "model_spec": model_spec,
                    "time_mode": time_mode,
                    "fit_mode": fit_mode,
                    "loss_schedule": schedule_id,
                    "loss_weights": {} if schedule_id == "baseline" else dict(LOSS_SCHEDULES[schedule_id]),
                }
            )
    max_runs = int(grid.get("max_runs", 0))
    return runs[:max_runs] if max_runs > 0 else runs


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(pred_x, sample, train_idx, eval_idx, fk, target_parts, cfg):
    device = pred_x.device
    pred_parts = predict_parts_in_chunks(fk, pred_x, chunk_size=int(cfg["metrics"].get("smplx_batch_size", 128)))
    row = {}
    x_all = sample.x_rot6d.to(device)
    row.update(evaluate_pose_prediction(pred_x, x_all, pred_parts, target_parts, prefix="all"))
    row.update(
        evaluate_pose_prediction(
            pred_x[train_idx.to(device)],
            x_all[train_idx.to(device)],
            index_parts(pred_parts, train_idx, device),
            index_parts(target_parts, train_idx, device),
            prefix="train",
        )
    )
    if len(eval_idx) > 0:
        row.update(
            evaluate_pose_prediction(
                pred_x[eval_idx.to(device)],
                x_all[eval_idx.to(device)],
                index_parts(pred_parts, eval_idx, device),
                index_parts(target_parts, eval_idx, device),
                prefix="heldout",
            )
        )
    return row


def run_experiment(cfg):
    set_seed(cfg.get("seed", 1234))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(cfg)
    indices = cfg.get("_indices") or select_pilot_indices(dataset, cfg)
    runs = expand_grid(cfg)
    (out_dir / "grid.json").write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")

    metric_cfg = cfg.get("metrics", {})
    fk = DifferentiableSMPLXForward(
        model_dir=Path(metric_cfg.get("model_dir", "deps/smpl_models")),
        gender=metric_cfg.get("gender", "NEUTRAL"),
        device=device,
        betas_mode=metric_cfg.get("betas_mode", "h2s_fixed"),
    )
    fk.eval()
    for param in fk.parameters():
        param.requires_grad_(False)

    rows = []
    start_time = time.time()
    joint_cache = {}
    for sample_ordinal, index in enumerate(indices):
        sample = load_oracle_sample(dataset, index, device)
        with torch.no_grad():
            target_parts = fk.parts_from_rot6d(sample.x_rot6d.to(device))
        cpu_joint_parts = {key: value.detach().cpu().numpy() for key, value in target_parts.items()}
        joint_cache[sample.index] = cpu_joint_parts
        for run in runs:
            row = {
                "index": int(sample.index),
                "sample_ordinal": int(sample_ordinal),
                "sample_id": sample.sample_id,
                "text": sample.text,
                "T": int(sample.length),
                "model": run["model_id"],
                "model_type": run["model_spec"]["type"],
                "scaffold": run["model_spec"].get("scaffold", ""),
                "time_mode": run["time_mode"],
                "fit_mode": run["fit_mode"],
                "loss_schedule": run["loss_schedule"],
            }
            try:
                tau_all = make_time_grid(
                    run["time_mode"],
                    sample.length,
                    joint_parts=cpu_joint_parts,
                    field_range=cfg["fit"].get("field_range", "-1_1"),
                    hand_weight=float(cfg["loss"].get("hand_weight", 3.0)),
                )
                train_idx, eval_idx = train_eval_indices(sample.length, run["fit_mode"])
                if is_baseline(run["model_spec"]):
                    pred_x, train_log, _model_obj = fit_baseline(run["model_spec"], tau_all, sample.x_rot6d, train_idx, device)
                else:
                    model = build_model(
                        run["model_spec"],
                        tau_all[train_idx],
                        sample.x_rot6d[train_idx].detach().cpu(),
                        cfg,
                        tau_all=tau_all,
                        x_all=sample.x_rot6d.detach().cpu(),
                        train_idx=train_idx,
                    )
                    pred_x, train_log, _model_obj = fit_neural(
                        model,
                        sample,
                        tau_all,
                        train_idx,
                        run["loss_weights"],
                        cfg,
                        fk,
                        target_parts,
                        device,
                    )
                row.update(train_log)
                row.update(evaluate_run(pred_x, sample, train_idx, eval_idx, fk, target_parts, cfg))
                if cfg["output"].get("save_npz", False):
                    stem = f"{sample_ordinal:04d}_{run['model_id']}_{run['time_mode']}_{run['loss_schedule']}_{run['fit_mode']}"
                    npz_dir = out_dir / "npz" / stem
                    save_motion_npz(npz_dir / "gt_000.npz", sample.x_rot6d.cpu(), sample, "gt")
                    save_motion_npz(npz_dir / "sample_000.npz", pred_x.cpu(), sample, "oracle_smplx_field")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if cfg["output"].get("flush_every_run", True):
                write_jsonl(out_dir / "metrics_rows.jsonl", rows)

    write_jsonl(out_dir / "metrics_rows.jsonl", rows)
    save_csv(out_dir / "metrics_rows.csv", rows)
    summary = {
        "elapsed_sec": time.time() - start_time,
        "num_samples": len(indices),
        "num_grid_runs": len(runs),
        "summary": summarize_rows(rows, ["model", "time_mode", "loss_schedule", "fit_mode"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
