from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from NIAF.oracle_latent_field.config import load_latent_stats_from_config
from NIAF.oracle_latent_field.data import (
    build_codec,
    build_dataset,
    compute_gt_pose_points,
    dtw_decoded_metrics,
    load_oracle_sample,
    save_motion_npz,
)
from NIAF.oracle_latent_field.losses import (
    directional_derivatives,
    finite_difference_acceleration,
    finite_difference_velocity,
    masked_smooth_l1,
)
from NIAF.oracle_latent_field.metrics import (
    adjacent_random_ratio,
    feature_sequence_metrics,
    latent_metrics,
    summarize_rows,
    write_jsonl,
)
from NIAF.oracle_latent_field.models import FourierFeatureMLP, ReLUMlp, ResidualLatentField, SirenLatentField, build_baseline
from NIAF.oracle_latent_field.plots import save_latent_plots
from NIAF.oracle_latent_field.time_parameterization import make_time_grid


MODEL_PRESETS = {
    "A1": {"type": "siren", "hidden": 128, "depth": 3},
    "A2": {"type": "siren", "hidden": 256, "depth": 3},
    "A3": {"type": "siren", "hidden": 512, "depth": 3},
    "A4": {"type": "siren", "hidden": 256, "depth": 4},
    "A5": {"type": "residual_siren", "hidden": 256, "depth": 3},
    "A6": {"type": "fourier_mlp", "hidden": 256, "depth": 3, "num_frequencies": 16},
    "A7": {"type": "relu_mlp", "hidden": 256, "depth": 3},
    "linear": {"type": "linear"},
    "cubic": {"type": "cubic_spline"},
    "bspline": {"type": "bspline", "control_points": 16},
    "dct": {"type": "dct", "components": 32},
}


LOSS_PRESETS = {
    "L1": {"lambda_z": 1.0, "lambda_vel": 0.0, "lambda_acc": 0.0, "lambda_jerk": 0.0, "lambda_dec": 0.0},
    "L2": {"lambda_z": 1.0, "lambda_vel": 0.05, "lambda_acc": 0.0, "lambda_jerk": 0.0, "lambda_dec": 0.0},
    "L3": {"lambda_z": 1.0, "lambda_vel": 0.05, "lambda_acc": 0.01, "lambda_jerk": 0.0, "lambda_dec": 0.0},
    "L4": {"lambda_z": 1.0, "lambda_vel": 0.05, "lambda_acc": 0.0, "lambda_jerk": 1.0e-6, "lambda_dec": 0.0},
    "L5": {"lambda_z": 1.0, "lambda_vel": 0.05, "lambda_acc": 0.0, "lambda_jerk": 1.0e-6, "lambda_dec": 0.1},
}


def set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def is_baseline(model_spec):
    return model_spec["type"] in {"linear", "cubic_spline", "bspline", "dct"}


def build_model(model_id, model_spec, dz, train_s, train_z, cfg):
    model_type = model_spec["type"]
    if model_type == "siren":
        return SirenLatentField(
            dz=dz,
            hidden=int(model_spec.get("hidden", cfg["model"].get("hidden", 256))),
            depth=int(model_spec.get("depth", cfg["model"].get("depth", 3))),
            omega0=float(model_spec.get("omega0", cfg["model"].get("omega0", 30.0))),
            omega=float(model_spec.get("omega", cfg["model"].get("omega", 1.0))),
        )
    if model_type == "residual_siren":
        return ResidualLatentField(
            train_s,
            train_z,
            dz=dz,
            hidden=int(model_spec.get("hidden", cfg["model"].get("hidden", 256))),
            depth=int(model_spec.get("depth", cfg["model"].get("depth", 3))),
            omega0=float(model_spec.get("omega0", cfg["model"].get("omega0", 30.0))),
            omega=float(model_spec.get("omega", cfg["model"].get("omega", 1.0))),
        )
    if model_type == "fourier_mlp":
        return FourierFeatureMLP(
            dz=dz,
            hidden=int(model_spec.get("hidden", 256)),
            depth=int(model_spec.get("depth", 3)),
            num_frequencies=int(model_spec.get("num_frequencies", 16)),
        )
    if model_type == "relu_mlp":
        return ReLUMlp(
            dz=dz,
            hidden=int(model_spec.get("hidden", 256)),
            depth=int(model_spec.get("depth", 3)),
        )
    raise ValueError(f"{model_id}: unsupported neural model type={model_type!r}")


def train_eval_indices(length, fit_mode):
    if fit_mode == "fit_all":
        idx = torch.arange(length, dtype=torch.long)
        return idx, idx
    if fit_mode == "even_odd":
        train = torch.arange(1, length, 2, dtype=torch.long)
        test = torch.tensor([i for i in range(length) if i % 2 == 0], dtype=torch.long)
        if len(train) == 0:
            train = torch.arange(length, dtype=torch.long)
        return train, test
    if fit_mode.startswith("sparse"):
        stride = 4
        if "stride" in fit_mode:
            try:
                stride = int(fit_mode.rsplit("stride", 1)[1].strip("_"))
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
    raise ValueError(f"Unsupported fit_mode={fit_mode!r}")


def decode_latents(z_norm, sample, codec, latent_stats):
    z_raw = codec.denormalize_latent(z_norm.unsqueeze(0), latent_stats)
    latent_mask = torch.ones(1, z_raw.shape[1], dtype=torch.bool, device=z_raw.device)
    frame_mask = torch.ones(1, sample.length, dtype=torch.bool, device=z_raw.device)
    return codec.decode(z_raw, target_length=sample.length, mask=frame_mask, latent_mask=latent_mask)[0, : sample.length]


def decode_dense_latents(z_norm, target_length, codec, latent_stats):
    z_raw = codec.denormalize_latent(z_norm.unsqueeze(0), latent_stats)
    latent_mask = torch.ones(1, z_raw.shape[1], dtype=torch.bool, device=z_raw.device)
    frame_mask = torch.ones(1, target_length, dtype=torch.bool, device=z_raw.device)
    return codec.decode(z_raw, target_length=target_length, mask=frame_mask, latent_mask=latent_mask)[0, :target_length]


def fit_neural_field(model, sample, s_all, train_idx, loss_weights, cfg, codec, latent_stats, device):
    fit_cfg = cfg["fit"]
    model = model.to(device)
    s_all = s_all.to(device=device, dtype=torch.float32)
    z_all = sample.z_norm.to(device=device, dtype=torch.float32)
    train_idx = train_idx.to(device)
    train_s = s_all[train_idx]
    train_z = z_all[train_idx]
    vel_target = finite_difference_velocity(train_z, train_s).detach()
    acc_target = finite_difference_acceleration(train_z, train_s).detach()

    steps = int(fit_cfg.get("steps", 5000))
    warmup_steps = min(int(fit_cfg.get("warmup_steps", 500)), steps)
    batch_points = int(fit_cfg.get("batch_points", 64))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fit_cfg.get("lr", 5e-4)),
        weight_decay=float(fit_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    grad_clip = float(fit_cfg.get("grad_clip", 1.0))
    dec_start = int(cfg["loss"].get("dec_start", max(warmup_steps, steps // 2)))
    dec_every = max(int(cfg["loss"].get("dec_every", 20)), 1)
    allow_holdout_dec = bool(cfg["loss"].get("allow_decoded_loss_in_holdout", False))

    last = {}
    rng = torch.Generator(device=device)
    rng.manual_seed(int(cfg.get("seed", 1234)) + int(sample.index))
    for step in range(steps):
        if batch_points > 0 and batch_points < len(train_idx):
            local = torch.randperm(len(train_idx), generator=rng, device=device)[:batch_points]
        else:
            local = torch.arange(len(train_idx), device=device)
        s_b = train_s[local]
        z_b = train_z[local]

        active = dict(loss_weights)
        if step < warmup_steps and active.get("lambda_z", 1.0) > 0:
            active.update({"lambda_vel": 0.0, "lambda_acc": 0.0, "lambda_jerk": 0.0, "lambda_dec": 0.0})

        max_order = 0
        if active.get("lambda_vel", 0.0) > 0:
            max_order = max(max_order, 1)
        if active.get("lambda_acc", 0.0) > 0:
            max_order = max(max_order, 2)
        if active.get("lambda_jerk", 0.0) > 0:
            max_order = max(max_order, 3)

        optimizer.zero_grad(set_to_none=True)
        if max_order > 0:
            pred, dz, ddz, dddz = directional_derivatives(model, s_b, order=max_order)
        else:
            pred = model(s_b)
            dz = ddz = dddz = torch.zeros_like(pred)

        loss_z = F.smooth_l1_loss(pred, z_b)
        loss = active.get("lambda_z", 1.0) * loss_z
        loss_vel = pred.new_tensor(0.0)
        loss_acc = pred.new_tensor(0.0)
        loss_jerk = pred.new_tensor(0.0)
        loss_dec = pred.new_tensor(0.0)
        if active.get("lambda_vel", 0.0) > 0:
            loss_vel = F.smooth_l1_loss(dz, vel_target[local])
            loss = loss + active["lambda_vel"] * loss_vel
        if active.get("lambda_acc", 0.0) > 0:
            loss_acc = F.smooth_l1_loss(ddz, acc_target[local])
            loss = loss + active["lambda_acc"] * loss_acc
        if active.get("lambda_jerk", 0.0) > 0:
            loss_jerk = dddz.pow(2).sum(dim=-1).mean()
            loss = loss + active["lambda_jerk"] * loss_jerk
        if (
            active.get("lambda_dec", 0.0) > 0
            and step >= dec_start
            and step % dec_every == 0
            and (allow_holdout_dec or torch.equal(train_idx.cpu(), torch.arange(sample.latent_length)))
        ):
            z_full = model(s_all)
            x_hat = decode_latents(z_full, sample, codec, latent_stats)
            loss_dec = masked_smooth_l1(x_hat, sample.x_norm.to(device), sample.frame_mask.to(device))
            loss = loss + active["lambda_dec"] * loss_dec

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        last = {
            "train_loss": float(loss.detach().cpu().item()),
            "train_loss_z": float(loss_z.detach().cpu().item()),
            "train_loss_vel": float(loss_vel.detach().cpu().item()),
            "train_loss_acc": float(loss_acc.detach().cpu().item()),
            "train_loss_jerk": float(loss_jerk.detach().cpu().item()),
            "train_loss_dec": float(loss_dec.detach().cpu().item()),
        }

    model.eval()
    with torch.no_grad():
        pred_all = model(s_all).detach()
    return pred_all, last, model


def fit_baseline(model_spec, sample, s_all, train_idx, device):
    baseline = build_baseline(model_spec["type"], **model_spec)
    train_s = s_all[train_idx]
    train_z = sample.z_norm[train_idx]
    baseline.fit(train_s, train_z)
    pred = baseline.predict(s_all, device=device, dtype=torch.float32)
    return pred, {"train_loss": float(F.smooth_l1_loss(pred[train_idx].cpu(), train_z.cpu()).item())}, baseline


def dense_evaluate(field_or_baseline, is_neural, sample, s_all, cfg, codec, latent_stats, device):
    out = {}
    factors = cfg["output"].get("dense_query_factors", [2, 4])
    max_latent = int(codec.max_latent_frames)
    for factor in factors:
        factor = int(factor)
        dense_len = int(sample.latent_length * factor)
        key = f"dense{factor}x"
        if dense_len > max_latent:
            out[f"{key}_skipped"] = 1.0
            continue
        dense_s = torch.linspace(
            float(s_all.min().item()),
            float(s_all.max().item()),
            dense_len,
            dtype=torch.float32,
            device=device,
        ).view(-1, 1)
        with torch.no_grad():
            if is_neural:
                z_dense = field_or_baseline(dense_s)
            else:
                z_dense = field_or_baseline.predict(dense_s.cpu(), device=device)
            target_len = int(sample.length * factor)
            x_dense = decode_dense_latents(z_dense, target_len, codec, latent_stats)
        out[f"{key}_latent_len"] = float(dense_len)
        out.update(feature_sequence_metrics(x_dense, x_dense, prefix=key))
    return out


def peak_midpoint_metrics(field_or_baseline, is_neural, sample, s_all, cfg, device):
    z_gt = sample.z_norm.to(device)
    s_device = s_all.to(device=device, dtype=torch.float32)
    if z_gt.shape[0] < 2:
        return {}
    distances = torch.linalg.norm(z_gt[1:] - z_gt[:-1], dim=-1)
    top_k = min(int(cfg["output"].get("peak_top_k", 5)), len(distances))
    values, indices = torch.topk(distances, k=top_k)
    mid_s = 0.5 * (s_device[indices] + s_device[indices + 1])
    with torch.no_grad():
        if is_neural:
            z_mid = field_or_baseline(mid_s)
        else:
            z_mid = field_or_baseline.predict(mid_s.cpu(), device=device)
    z_linear = 0.5 * (z_gt[indices] + z_gt[indices + 1])
    midpoint_delta = torch.linalg.norm(z_mid - z_linear, dim=-1)
    return {
        "peak_adjacent_l2_mean": float(values.float().mean().detach().cpu().item()),
        "peak_adjacent_l2_max": float(values.float().max().detach().cpu().item()),
        "peak_midpoint_vs_linear_l2_mean": float(midpoint_delta.float().mean().detach().cpu().item()),
        "peak_midpoint_vs_linear_l2_max": float(midpoint_delta.float().max().detach().cpu().item()),
    }


def evaluate_prediction(
    pred_z,
    sample,
    s_all,
    train_idx,
    eval_idx,
    model_obj,
    is_neural,
    cfg,
    codec,
    latent_stats,
    device,
    dataset=None,
):
    row = {}
    z_gt = sample.z_norm.to(device)
    pred_z = pred_z.to(device)
    row.update(latent_metrics(pred_z, z_gt, prefix="all"))
    row.update(latent_metrics(pred_z[train_idx.to(device)], z_gt[train_idx.to(device)], prefix="train"))
    if len(eval_idx) > 0:
        row.update(latent_metrics(pred_z[eval_idx.to(device)], z_gt[eval_idx.to(device)], prefix="heldout"))
    row.update(adjacent_random_ratio(z_gt, seed=int(cfg.get("seed", 1234)) + sample.index))
    with torch.no_grad():
        x_field = decode_latents(pred_z, sample, codec, latent_stats)
    x_gt = sample.x_norm.to(device)
    x_vae = sample.x_vae_norm.to(device)
    row.update(feature_sequence_metrics(x_vae, x_gt, prefix="vae"))
    row.update(feature_sequence_metrics(x_field, x_gt, prefix="field"))
    if "vae_feature_mae" in row and "field_feature_mae" in row:
        row["field_to_vae_feature_mae_gap"] = row["field_feature_mae"] - row["vae_feature_mae"]
    row.update(dense_evaluate(model_obj, is_neural, sample, s_all, cfg, codec, latent_stats, device))
    row.update(peak_midpoint_metrics(model_obj, is_neural, sample, s_all, cfg, device))
    if dataset is not None and cfg["metrics"].get("compute_dtw", False):
        metric_cfg = cfg["metrics"]
        for alignment_mode in metric_cfg.get("alignment_modes", ["default"]):
            row.update(
                dtw_decoded_metrics(
                    x_field,
                    sample,
                    dataset,
                    sample.rotation_rep,
                    model_dir=Path(metric_cfg.get("model_dir", "deps/smpl_models")),
                    gender=metric_cfg.get("gender", "NEUTRAL"),
                    device=metric_cfg.get("metric_device", "cpu"),
                    batch_size=int(metric_cfg.get("smplx_batch_size", 128)),
                    betas_mode=metric_cfg.get("betas_mode", "h2s_fixed"),
                    parts=tuple(metric_cfg.get("parts", ["body", "lhand", "rhand", "wholebody"])),
                    alignment_mode=alignment_mode,
                )
            )
    return row, x_field.detach()


def expand_grid(cfg):
    grid = cfg["grid"]
    models = grid.get("models", ["A2"])
    times = grid.get("time_modes", ["uniform"])
    losses = grid.get("losses", ["L1"])
    fit_modes = grid.get("fit_modes", ["fit_all"])
    runs = []
    for model_id in models:
        model_spec = dict(MODEL_PRESETS.get(model_id, {"type": model_id}))
        loss_iter = ["baseline"] if is_baseline(model_spec) else losses
        for time_mode, loss_id, fit_mode in itertools.product(times, loss_iter, fit_modes):
            loss_weights = dict(LOSS_PRESETS.get(loss_id, LOSS_PRESETS["L1"]))
            loss_id_effective = loss_id
            if is_baseline(model_spec):
                loss_id_effective = "baseline"
                loss_weights = dict(LOSS_PRESETS["L1"])
            runs.append(
                {
                    "model_id": model_id,
                    "model_spec": model_spec,
                    "time_mode": time_mode,
                    "loss_id": loss_id_effective,
                    "fit_mode": fit_mode,
                    "loss_weights": loss_weights,
                }
            )
    return runs


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_latent_trajectory_npz(path, sample, pred_z, run):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_gt = sample.z_norm.detach().cpu().float().numpy()
    z_fit = pred_z.detach().cpu().float().numpy()
    np.savez_compressed(
        path,
        z_sent=z_gt.astype(np.float32),
        z_fit=z_fit.astype(np.float32),
        latent_mask=np.ones(len(z_gt), dtype=np.bool_),
        name=str(sample.sample_id),
        text=str(sample.text),
        length=int(sample.length),
        source_index=int(sample.index),
        model=str(run["model_id"]),
        model_type=str(run["model_spec"]["type"]),
        time_mode=str(run["time_mode"]),
        loss=str(run["loss_id"]),
        fit_mode=str(run["fit_mode"]),
    )


def run_experiment(cfg, overrides=None):
    if overrides:
        cfg = {**cfg, **overrides}
    set_seed(cfg.get("seed", 1234))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    codec = build_codec(cfg, device)
    latent_stats = load_latent_stats_from_config(cfg["vae"]["latent_stats_config"])
    indices = cfg.get("_indices")
    if indices is None:
        from NIAF.oracle_latent_field.data import select_pilot_indices

        indices = select_pilot_indices(dataset, cfg, downsample_factor=codec.downsample_factor)
    runs = expand_grid(cfg)
    max_runs = int(cfg["grid"].get("max_runs", 0))
    if max_runs > 0:
        runs = runs[:max_runs]

    (out_dir / "grid.json").write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    rows = []
    start_time = time.time()
    pose_cache = {}
    for sample_ordinal, index in enumerate(indices):
        sample = load_oracle_sample(dataset, index, codec, latent_stats, device)
        for run_ordinal, run in enumerate(runs):
            model_spec = run["model_spec"]
            row = {
                "index": int(sample.index),
                "sample_ordinal": int(sample_ordinal),
                "sample_id": sample.sample_id,
                "text": sample.text,
                "T": int(sample.length),
                "L": int(sample.latent_length),
                "model": run["model_id"],
                "model_type": model_spec["type"],
                "time_mode": run["time_mode"],
                "loss": run["loss_id"],
                "fit_mode": run["fit_mode"],
            }
            try:
                pose_points = None
                if run["time_mode"] in {"pose_arclength", "pose_arc"}:
                    if sample.index not in pose_cache:
                        pose_cache[sample.index] = compute_gt_pose_points(
                            sample,
                            dataset,
                            sample.rotation_rep,
                            model_dir=Path(cfg["metrics"].get("model_dir", "deps/smpl_models")),
                            device=cfg["metrics"].get("metric_device", "cpu"),
                            batch_size=int(cfg["metrics"].get("smplx_batch_size", 128)),
                            betas_mode=cfg["metrics"].get("betas_mode", "h2s_fixed"),
                        )
                    pose_points = pose_cache[sample.index]
                s_all = make_time_grid(
                    run["time_mode"],
                    sample.z_norm.cpu(),
                    pose_points=pose_points,
                    downsample_factor=codec.downsample_factor,
                    field_range=cfg["fit"].get("field_range", "-1_1"),
                )
                train_idx, eval_idx = train_eval_indices(sample.latent_length, run["fit_mode"])
                if is_baseline(model_spec):
                    pred_z, train_log, model_obj = fit_baseline(model_spec, sample, s_all, train_idx, device)
                    is_neural_model = False
                else:
                    model = build_model(
                        run["model_id"],
                        model_spec,
                        sample.z_norm.shape[-1],
                        s_all[train_idx],
                        sample.z_norm[train_idx],
                        cfg,
                    )
                    pred_z, train_log, model_obj = fit_neural_field(
                        model,
                        sample,
                        s_all,
                        train_idx,
                        run["loss_weights"],
                        cfg,
                        codec,
                        latent_stats,
                        device,
                    )
                    is_neural_model = True
                    if cfg["output"].get("save_fitted_params", False):
                        param_dir = out_dir / "fitted_params" / run["model_id"]
                        param_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "model_id": run["model_id"],
                                "model_spec": model_spec,
                                "state_dict": model_obj.state_dict(),
                                "sample_id": sample.sample_id,
                                "sample_index": sample.index,
                                "time_mode": run["time_mode"],
                                "fit_mode": run["fit_mode"],
                                "loss": run["loss_id"],
                            },
                            param_dir / f"{sample_ordinal:04d}_{run['time_mode']}_{run['fit_mode']}_{run['loss_id']}.pt",
                        )
                metrics, x_field = evaluate_prediction(
                    pred_z,
                    sample,
                    s_all,
                    train_idx,
                    eval_idx,
                    model_obj,
                    is_neural_model,
                    cfg,
                    codec,
                    latent_stats,
                    device,
                    dataset=dataset,
                )
                row.update(train_log)
                row.update(metrics)
                if cfg["output"].get("save_plots", False):
                    stem = f"{sample_ordinal:04d}_{run['model_id']}_{run['time_mode']}_{run['loss_id']}_{run['fit_mode']}"
                    save_latent_plots(out_dir / "plots", stem, sample.z_norm.cpu(), pred_z.detach().cpu())
                if cfg["output"].get("save_latent_npz", False):
                    stem = f"{sample_ordinal:04d}_{run['model_id']}_{run['time_mode']}_{run['loss_id']}_{run['fit_mode']}"
                    save_latent_trajectory_npz(
                        out_dir / "latent_npz" / f"{stem}.npz",
                        sample,
                        pred_z,
                        run,
                    )
                if cfg["output"].get("save_npz", False):
                    stem = f"{sample_ordinal:04d}_{run['model_id']}_{run['time_mode']}_{run['loss_id']}_{run['fit_mode']}"
                    npz_dir = out_dir / "npz" / stem
                    save_motion_npz(npz_dir / "gt_000.npz", sample.x_norm.cpu(), dataset, sample.rotation_rep, sample, "gt")
                    save_motion_npz(npz_dir / "vae_000.npz", sample.x_vae_norm.cpu(), dataset, sample.rotation_rep, sample, "vae")
                    save_motion_npz(npz_dir / "sample_000.npz", x_field.cpu(), dataset, sample.rotation_rep, sample, "oracle_field")
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
        "summary": summarize_rows(rows, ["model", "time_mode", "loss", "fit_mode"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
