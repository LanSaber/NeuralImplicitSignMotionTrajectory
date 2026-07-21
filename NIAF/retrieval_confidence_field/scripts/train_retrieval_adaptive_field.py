from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
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
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.losses import (
    endpoint_losses,
    fk_temporal_dynamics_losses,
    fk_temporal_regularization_losses,
    prediction_parts_from_rot6d,
)
from NIAF.continuous_sign_field.meta_learning import highpass_residual_loss
from NIAF.continuous_sign_field.metrics import ScalarAverager, append_jsonl, tensor_dict_to_float
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import (
    ARTICULATOR_NAMES,
    RetrievalConfidenceAdaptiveField,
    articulator_scaffold_error,
    confidence_target_from_error,
    target_tangent_correction,
)
from NIAF.retrieval_confidence_field.models.uncertainty_adaptive import (
    DEFAULT_ARTICULATOR_STRIDES,
    RetrievalUncertaintyAdaptiveKnotField,
    adaptive_knot_density_target,
    correction_need_target_from_error,
)
from NIAF.retrieval_confidence_field.models.segmental import (
    RetrievalUncertaintySegmentalField,
    boundary_temporal_matching_loss,
)
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


__all__ = [
    "apply_overrides",
    "build_retrieval_adaptive_model",
    "confidence_metrics",
    "confidence_temperatures",
    "evaluate",
    "init_wandb",
    "main",
    "masked_group_means",
    "multi_scale_code_smoothness",
    "parse_args",
    "prepare_retrieval_batch",
    "resolve_device",
    "retrieval_adaptive_losses",
    "run_train_step",
    "save_checkpoint",
    "scale_metrics",
    "set_seed",
    "tangent_correction_loss",
    "validate_train_only_retrieval_bank",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a retrieval-confidence adaptive neural sign field."
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
    parser.add_argument("--wandb_project", default="soke-niaf-retrieval-adaptive")
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
    if args.out_dir is not None:
        cfg.setdefault("output", {})["out_dir"] = str(args.out_dir)
    return cfg


def build_retrieval_adaptive_model(cfg, text_dim):
    model_cfg = cfg.get("model", {})
    duration_cfg = cfg.get("duration", {})
    model_type = str(model_cfg.get("type", "retrieval_confidence_adaptive_field"))
    if model_type in {
        "retrieval_uncertainty_adaptive_knot_field",
        "retrieval_uncertainty_segmental_field",
    }:
        configured_strides = model_cfg.get(
            "articulator_code_strides", DEFAULT_ARTICULATOR_STRIDES
        )
        articulator_strides = {
            name: tuple(configured_strides.get(name, DEFAULT_ARTICULATOR_STRIDES[name]))
            for name in ARTICULATOR_NAMES
        }
        model_class = (
            RetrievalUncertaintySegmentalField
            if model_type == "retrieval_uncertainty_segmental_field"
            else RetrievalUncertaintyAdaptiveKnotField
        )
        model_kwargs = dict(
            pose_dim=COMPACT6D_DIM,
            text_dim=int(text_dim),
            retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
            code_dim=int(model_cfg.get("code_dim", 64)),
            context_hidden_dim=int(model_cfg.get("context_hidden_dim", 256)),
            context_layers=int(model_cfg.get("context_layers", 1)),
            context_heads=int(model_cfg.get("context_heads", 8)),
            articulator_code_strides=articulator_strides,
            knot_kernel_width=float(model_cfg.get("knot_kernel_width", 0.75)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            depth=int(model_cfg.get("depth", 4)),
            time_fourier_bands=int(model_cfg.get("time_fourier_bands", 10)),
            context_time_fourier_bands=int(
                model_cfg.get("context_time_fourier_bands", 6)
            ),
            omega0_first=float(model_cfg.get("omega0_first", 20.0)),
            omega0_hidden=float(model_cfg.get("omega0_hidden", 1.0)),
            residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
            residual_scale_learnable=bool(
                model_cfg.get("residual_scale_learnable", True)
            ),
            body_gate_bias=float(model_cfg.get("body_gate_bias", -2.0)),
            hand_gate_bias=float(model_cfg.get("hand_gate_bias", -0.5)),
            face_gate_bias=float(model_cfg.get("face_gate_bias", -2.0)),
            calibrator_hidden_dim=int(model_cfg.get("calibrator_hidden_dim", 128)),
            calibrator_text_dim=int(model_cfg.get("calibrator_text_dim", 64)),
            calibrator_time_fourier_bands=int(
                model_cfg.get("calibrator_time_fourier_bands", 4)
            ),
            trust_initial=float(model_cfg.get("trust_initial", 0.5)),
            correction_need_initial=float(
                model_cfg.get("correction_need_initial", 0.5)
            ),
            knot_density_initial=float(model_cfg.get("knot_density_initial", 0.5)),
            correction_need_floor=float(model_cfg.get("correction_need_floor", 0.25)),
            allocation_scale_bias=float(model_cfg.get("allocation_scale_bias", 1.5)),
            evidence_density_bias=float(model_cfg.get("evidence_density_bias", 1.0)),
            density_floor=float(model_cfg.get("density_floor", 0.05)),
            duration_hidden_dim=int(duration_cfg.get("hidden_dim", 256)),
            duration_initial_frames=float(duration_cfg.get("initial_frames", 80.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        if model_class is RetrievalUncertaintySegmentalField:
            model_kwargs.update(
                segment_rollout_layers=int(
                    model_cfg.get("segment_rollout_layers", 2)
                ),
                segment_window_multiplier=float(
                    model_cfg.get("segment_window_multiplier", 2.0)
                ),
                minimum_segment_frames=int(
                    model_cfg.get("minimum_segment_frames", 8)
                ),
                maximum_segment_frames=int(
                    model_cfg.get("maximum_segment_frames", 32)
                ),
                segment_text_window_radius=float(
                    model_cfg.get("segment_text_window_radius", 0.2)
                ),
                segment_boundary_stride=int(
                    model_cfg.get("segment_boundary_stride", 8)
                ),
            )
        return model_class(**model_kwargs)
    if model_type != "retrieval_confidence_adaptive_field":
        raise ValueError(f"Unsupported retrieval field model.type={model_type!r}")
    return RetrievalConfidenceAdaptiveField(
        pose_dim=COMPACT6D_DIM,
        text_dim=int(text_dim),
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        code_dim=int(model_cfg.get("code_dim", 64)),
        context_hidden_dim=int(model_cfg.get("context_hidden_dim", 256)),
        context_layers=int(model_cfg.get("context_layers", 1)),
        context_heads=int(model_cfg.get("context_heads", 8)),
        code_strides=tuple(model_cfg.get("code_strides", (4, 8, 16))),
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
        confidence_hidden_dim=int(model_cfg.get("confidence_hidden_dim", 128)),
        confidence_text_dim=int(model_cfg.get("confidence_text_dim", 64)),
        confidence_time_fourier_bands=int(
            model_cfg.get("confidence_time_fourier_bands", 4)
        ),
        confidence_initial=float(model_cfg.get("confidence_initial", 0.5)),
        confidence_residual_floor=float(model_cfg.get("confidence_residual_floor", 0.1)),
        confidence_scale_bias=float(model_cfg.get("confidence_scale_bias", 1.0)),
        duration_hidden_dim=int(duration_cfg.get("hidden_dim", 256)),
        duration_initial_frames=float(duration_cfg.get("initial_frames", 80.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def validate_train_only_retrieval_bank(cfg, provider):
    retrieval_cfg = cfg.get("retrieval", {})
    require_train_only = bool(retrieval_cfg.get("require_train_only_bank", True))
    expected_split = retrieval_cfg.get("expected_word_split")
    if expected_split is None:
        if not require_train_only:
            return None
        expected_split = "train.balanced"
    expected_split = str(expected_split)
    expected = (
        Path(cfg["adapter"]["word_data_dir"])
        / "meta"
        / f"manifest_{expected_split}.jsonl"
    ).resolve()
    prior_builder = provider.adapter_prior.prior_builder if provider.adapter_prior is not None else None
    if prior_builder is not None:
        actual = Path(prior_builder.manifest_path).resolve()
        entries = len(prior_builder.entries)
        lexicon_keys = len(prior_builder.entries_by_key)
    else:
        word_split = str(cfg.get("adapter", {}).get("word_split", ""))
        actual = (
            Path(cfg["adapter"]["word_data_dir"]) / "meta" / f"manifest_{word_split}.jsonl"
        ).resolve()
        entries = None
        lexicon_keys = None
    if actual != expected:
        mode = "training-only" if require_train_only else "configured"
        raise ValueError(
            f"Strict retrieval mode expected the {mode} word bank at {expected}, "
            f"but loaded {actual}. Set adapter.word_split={expected_split}."
        )

    cache_summary = None
    if prior_builder is None and bool(getattr(provider, "cache_only", False)):
        cache_dir = getattr(provider, "scaffold_cache_dir", None)
        summary_path = Path(cache_dir) / "cache_summary.json" if cache_dir is not None else None
        if summary_path is None or not summary_path.is_file():
            raise FileNotFoundError(
                "Strict cache-only training requires cache_summary.json. Run the retrieval "
                "confidence scaffold cache job before training."
            )
        cache_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(cache_summary.get("word_split")) != expected_split:
            raise ValueError(
                f"Scaffold cache was built with word_split={cache_summary.get('word_split')!r}, "
                f"not {expected_split}."
            )
        cached_manifest = cache_summary.get("word_manifest")
        if cached_manifest and Path(cached_manifest).resolve() != expected:
            raise ValueError(
                f"Scaffold cache was built from {cached_manifest}, not {expected}."
            )
        cached_checkpoint = cache_summary.get("adapter_checkpoint")
        configured_checkpoint = cfg.get("adapter", {}).get("checkpoint")
        if (
            cached_checkpoint
            and configured_checkpoint
            and Path(cached_checkpoint).resolve() != Path(configured_checkpoint).resolve()
        ):
            raise ValueError(
                f"Scaffold cache was built with adapter {cached_checkpoint}, "
                f"not {configured_checkpoint}."
            )
        if not bool(cache_summary.get("require_retrieval_features", False)):
            raise ValueError("Scaffold cache does not declare retrieval confidence features.")
        cached_features = tuple(cache_summary.get("retrieval_feature_names") or ())
        if cached_features and cached_features != RETRIEVAL_FEATURE_NAMES:
            raise ValueError(f"Unexpected cached retrieval feature layout: {cached_features}")

    if entries is None:
        entries = 0
        keys = set()
        with actual.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entries += 1
                key = row.get("lexicon_key") or row.get("word") or row.get("gloss")
                if key:
                    keys.add(str(key))
        lexicon_keys = len(keys)
    return {
        "manifest": str(actual),
        "entries": entries,
        "lexicon_keys": lexicon_keys,
        "cache_summary": str(summary_path) if cache_summary is not None else None,
    }


def _masked_mean(values, mask):
    weight = mask.to(device=values.device, dtype=values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum() / weight.expand_as(values).sum().clamp_min(1.0)


def multi_scale_code_smoothness(scale_codes, scale_masks):
    losses = []
    for codes, code_mask in zip(scale_codes, scale_masks):
        if codes.shape[1] <= 1:
            continue
        valid = code_mask[:, 1:] & code_mask[:, :-1]
        losses.append(_masked_mean(torch.abs(codes[:, 1:] - codes[:, :-1]), valid))
    if not losses:
        return scale_codes[0].new_tensor(0.0)
    return torch.stack(losses).mean()


def tangent_correction_loss(pred, target, mask, hand_weight=5.0):
    weights = pred.new_ones(pred.shape[-1])
    weights[30:120] = float(hand_weight)
    return _masked_mean(torch.abs(pred - target) * weights.view(1, 1, -1), mask)


def confidence_temperatures(cfg):
    configured = cfg.get("confidence", {}).get("error_temperatures", {})
    defaults = {"body": 0.30, "lhand": 0.45, "rhand": 0.45, "face": 0.35}
    if isinstance(configured, (list, tuple)):
        if len(configured) != len(ARTICULATOR_NAMES):
            raise ValueError("confidence.error_temperatures must have four values.")
        return [float(value) for value in configured]
    return [float(configured.get(name, defaults[name])) for name in ARTICULATOR_NAMES]


def correction_need_temperatures(cfg):
    configured = cfg.get("adaptation", {}).get("correction_need_temperatures", {})
    defaults = {"body": 0.25, "lhand": 0.30, "rhand": 0.30, "face": 0.30}
    if isinstance(configured, (list, tuple)):
        if len(configured) != len(ARTICULATOR_NAMES):
            raise ValueError("adaptation.correction_need_temperatures must have four values.")
        return [float(value) for value in configured]
    return [float(configured.get(name, defaults[name])) for name in ARTICULATOR_NAMES]


def masked_group_means(values, mask, prefix):
    return {
        f"{prefix}_{name}": _masked_mean(values[:, :, index], mask)
        for index, name in enumerate(ARTICULATOR_NAMES)
    }


def confidence_metrics(confidence, target, errors, mask):
    metrics = {}
    for index, name in enumerate(ARTICULATOR_NAMES):
        valid = mask.bool()
        pred_values = confidence[:, :, index][valid]
        target_values = target[:, :, index][valid]
        error_values = errors[:, :, index][valid]
        metrics[f"confidence_mae_{name}"] = torch.abs(pred_values - target_values).mean()
        pred_centered = pred_values - pred_values.mean()
        target_centered = target_values - target_values.mean()
        denominator = torch.sqrt(
            pred_centered.square().sum() * target_centered.square().sum()
        ).clamp_min(1e-8)
        metrics[f"confidence_corr_{name}"] = (
            pred_centered * target_centered
        ).sum() / denominator
        metrics[f"scaffold_error_{name}"] = error_values.mean()
    return metrics


def scale_metrics(scale_weights, mask, strides):
    weight = mask[:, :, None, None].to(scale_weights.dtype)
    means = (scale_weights * weight).sum(dim=(0, 1)) / weight.sum().clamp_min(1.0)
    return {
        f"scale_{name}_stride{stride}": means[group_index, scale_index]
        for group_index, name in enumerate(ARTICULATOR_NAMES)
        for scale_index, stride in enumerate(strides)
    }


def articulator_scale_metrics(scale_weights, mask, articulator_strides):
    weight = mask[:, :, None, None].to(scale_weights.dtype)
    means = (scale_weights * weight).sum(dim=(0, 1)) / weight.sum().clamp_min(1.0)
    return {
        f"scale_{name}_stride{stride}": means[group_index, scale_index]
        for group_index, name in enumerate(ARTICULATOR_NAMES)
        for scale_index, stride in enumerate(articulator_strides[name])
    }


def prepare_retrieval_batch(model, text_encoder, scaffold_provider, batch, dataset, cfg, device):
    x = prepare_motion(batch, dataset, device)
    mask = batch["mask"]
    lengths = batch["length"]
    scaffold, _anchor_mask, metadata = scaffold_provider.build_with_metadata(batch, x=None)
    retrieval_features = metadata["retrieval_features"]
    tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    outputs = model(
        tau,
        scaffold,
        mask,
        text_tokens,
        retrieval_features,
        text_mask=text_mask,
    )
    scaffold_errors = articulator_scaffold_error(
        scaffold,
        x,
        mask,
        expression_weight=float(cfg.get("confidence", {}).get("expression_weight", 0.1)),
    )
    confidence_target = confidence_target_from_error(
        scaffold_errors,
        confidence_temperatures(cfg),
        mask,
    )
    prepared = {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "target_parts": batch["target_parts"],
        "scaffold": scaffold,
        "retrieval_features": retrieval_features,
        "tau": tau,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "outputs": outputs,
        "scaffold_errors": scaffold_errors,
        "confidence_target": confidence_target,
    }
    base_model = unwrap_model(model)
    if isinstance(base_model, RetrievalUncertaintyAdaptiveKnotField):
        target_correction = target_tangent_correction(scaffold, x, mask)
        correction_need_target = correction_need_target_from_error(
            scaffold_errors,
            correction_need_temperatures(cfg),
            mask,
        )
        density_target = adaptive_knot_density_target(
            target_correction,
            correction_need_target,
            mask,
            transition_weight=float(
                cfg.get("adaptation", {}).get("density_transition_weight", 0.5)
            ),
        )
        prepared.update(
            {
                "target_correction": target_correction,
                "trust_target": confidence_target,
                "correction_need_target": correction_need_target,
                "knot_density_target": density_target,
            }
        )
    return prepared


def retrieval_adaptive_losses(model, fk, prepared, cfg):
    endpoint_cfg = cfg.get("loss", {})
    field_cfg = cfg.get("field_loss", {})
    hand_weight = float(endpoint_cfg.get("hand_weight", 5.0))
    chunk_size = int(cfg.get("metrics", {}).get("fk_batch_size", 128))
    x = prepared["x"]
    mask = prepared["mask"]
    lengths = prepared["lengths"]
    scaffold = prepared["scaffold"]
    outputs = prepared["outputs"]
    pred = outputs["prediction"]
    base_model = unwrap_model(model)
    uncertainty_adaptive = "correction_need" in outputs

    endpoint_fk_keys = (
        "lambda_joint",
        "lambda_hand",
        "lambda_hand_relative",
        "lambda_vel",
        "lambda_vel_hand",
        "lambda_acc",
        "lambda_path",
    )
    field_fk_keys = (
        "lambda_fk_vel",
        "lambda_fk_acc",
        "lambda_fk_jerk",
        "lambda_fk_vel_reg",
        "lambda_fk_acc_reg",
        "lambda_fk_jerk_reg",
    )
    pred_parts = None
    if any(float(endpoint_cfg.get(key, 0.0)) > 0 for key in endpoint_fk_keys) or any(
        float(field_cfg.get(key, 0.0)) > 0 for key in field_fk_keys
    ):
        pred_parts = prediction_parts_from_rot6d(
            pred,
            mask,
            prepared["target_parts"],
            fk,
            fk_chunk_size=chunk_size,
        )

    endpoint, endpoint_dict = endpoint_losses(
        pred,
        x,
        mask,
        lengths,
        prepared["target_parts"],
        fk=fk,
        weights=endpoint_cfg,
        hand_weight=hand_weight,
        fk_chunk_size=chunk_size,
        pred_parts=pred_parts,
    )
    target_correction = prepared.get("target_correction")
    if target_correction is None:
        target_correction = target_tangent_correction(scaffold, x, mask)
    correction = tangent_correction_loss(
        outputs["correction_axis"],
        target_correction,
        mask,
        hand_weight=hand_weight,
    )
    target_residual = (x - scaffold) * mask.unsqueeze(-1).to(x.dtype)
    residual_vel = highpass_residual_loss(
        outputs["residual"], target_residual, mask, order=1, hand_weight=hand_weight
    )
    residual_acc = highpass_residual_loss(
        outputs["residual"], target_residual, mask, order=2, hand_weight=hand_weight
    )
    residual_jerk = highpass_residual_loss(
        outputs["residual"], target_residual, mask, order=3, hand_weight=hand_weight
    )

    dynamics_weights = {
        key: field_cfg.get(key, 0.0)
        for key in (
            "lambda_fk_vel",
            "lambda_fk_acc",
            "lambda_fk_jerk",
            "fk_temporal_include_hand_parts",
        )
    }
    dynamics, dynamics_dict = fk_temporal_dynamics_losses(
        pred,
        mask,
        lengths,
        prepared["target_parts"],
        fk,
        weights=dynamics_weights,
        hand_weight=hand_weight,
        fk_chunk_size=chunk_size,
        pred_parts=pred_parts,
    )
    regularization_weights = {
        key: field_cfg.get(key, 0.0)
        for key in (
            "lambda_fk_vel_reg",
            "lambda_fk_acc_reg",
            "lambda_fk_jerk_reg",
            "fk_temporal_include_hand_parts",
        )
    }
    regularization, regularization_dict = fk_temporal_regularization_losses(
        pred,
        mask,
        lengths,
        prepared["target_parts"],
        fk,
        weights=regularization_weights,
        hand_weight=hand_weight,
        fk_chunk_size=chunk_size,
        pred_parts=pred_parts,
    )

    confidence_target = prepared["confidence_target"]
    confidence_loss = _masked_mean(
        F.smooth_l1_loss(outputs["confidence"], confidence_target, reduction="none"),
        mask,
    )
    if uncertainty_adaptive:
        correction_need_target = prepared["correction_need_target"]
        knot_density_target = prepared["knot_density_target"]
        correction_need_loss = _masked_mean(
            F.smooth_l1_loss(
                outputs["correction_need"], correction_need_target, reduction="none"
            ),
            mask,
        )
        knot_density_loss = _masked_mean(
            F.smooth_l1_loss(
                outputs["knot_density"], knot_density_target, reduction="none"
            ),
            mask,
        )
        scale_strength = float(
            cfg.get("adaptation", {}).get("scale_target_strength", 2.0)
        )
        allocation_signal = confidence_target - correction_need_target
    else:
        correction_need_target = None
        knot_density_target = None
        correction_need_loss = pred.new_tensor(0.0)
        knot_density_loss = pred.new_tensor(0.0)
        scale_strength = float(
            cfg.get("confidence", {}).get("scale_target_strength", 2.0)
        )
        allocation_signal = 2.0 * confidence_target - 1.0
    scale_logits = (
        scale_strength
        * allocation_signal.unsqueeze(-1)
        * base_model.scale_rank.to(dtype=x.dtype).view(1, 1, 1, -1)
    )
    target_scale_weights = torch.softmax(scale_logits, dim=-1)
    scale_loss = _masked_mean(
        F.smooth_l1_loss(outputs["scale_weights"], target_scale_weights, reduction="none"),
        mask,
    )
    confident_gate = _masked_mean(confidence_target * outputs["gates"], mask)
    gate_penalty = _masked_mean(outputs["gates"], mask)
    if uncertainty_adaptive:
        gate_target_loss = _masked_mean(
            F.smooth_l1_loss(
                outputs["gates"], correction_need_target, reduction="none"
            ),
            mask,
        )
        preserve_gate = _masked_mean(
            confidence_target * (1.0 - correction_need_target) * outputs["gates"],
            mask,
        )
    else:
        gate_target_loss = pred.new_tensor(0.0)
        preserve_gate = confident_gate
    code_smooth = multi_scale_code_smoothness(
        outputs["scale_codes"], outputs["scale_code_masks"]
    )
    segment_pose = pred.new_tensor(0.0)
    segment_vel = pred.new_tensor(0.0)
    segment_acc = pred.new_tensor(0.0)
    if "segment_boundary_mask" in outputs:
        boundaries = outputs["segment_boundary_mask"] & mask
        segment_pose = boundary_temporal_matching_loss(
            pred,
            x,
            mask,
            boundaries,
            order=0,
            hand_weight=hand_weight,
        )
        segment_vel = boundary_temporal_matching_loss(
            pred,
            x,
            mask,
            boundaries,
            order=1,
            hand_weight=hand_weight,
        )
        segment_acc = boundary_temporal_matching_loss(
            pred,
            x,
            mask,
            boundaries,
            order=2,
            hand_weight=hand_weight,
        )

    target_log_frames = torch.log(lengths.to(outputs["pred_log_frames"].dtype).clamp_min(1.0))
    duration_loss = F.smooth_l1_loss(
        outputs["pred_log_frames"],
        target_log_frames,
        beta=float(cfg.get("duration", {}).get("huber_beta", 1.0)),
    )
    total = (
        float(field_cfg.get("lambda_endpoint", 1.0)) * endpoint
        + float(field_cfg.get("lambda_correction", 0.5)) * correction
        + float(field_cfg.get("lambda_residual_vel", 0.25)) * residual_vel
        + float(field_cfg.get("lambda_residual_acc", 0.1)) * residual_acc
        + float(field_cfg.get("lambda_residual_jerk", 0.0)) * residual_jerk
        + float(field_cfg.get("lambda_scale_prior", 0.05)) * scale_loss
        + float(field_cfg.get("lambda_gate", 0.0)) * gate_penalty
        + float(field_cfg.get("lambda_code_smooth", 0.01)) * code_smooth
        + float(field_cfg.get("lambda_segment_pose", 0.0)) * segment_pose
        + float(field_cfg.get("lambda_segment_vel", 0.0)) * segment_vel
        + float(field_cfg.get("lambda_segment_acc", 0.0)) * segment_acc
        + float(field_cfg.get("lambda_duration", 0.25)) * duration_loss
        + dynamics
        + regularization
    )
    if uncertainty_adaptive:
        total = (
            total
            + float(field_cfg.get("lambda_trust", 0.5)) * confidence_loss
            + float(field_cfg.get("lambda_correction_need", 0.75))
            * correction_need_loss
            + float(field_cfg.get("lambda_knot_density", 0.25))
            * knot_density_loss
            + float(field_cfg.get("lambda_gate_target", 0.1)) * gate_target_loss
            + float(field_cfg.get("lambda_preserve_gate", 0.02)) * preserve_gate
        )
    else:
        total = (
            total
            + float(field_cfg.get("lambda_confidence", 0.5)) * confidence_loss
            + float(field_cfg.get("lambda_confident_gate", 0.05)) * confident_gate
        )

    valid = mask.unsqueeze(-1).to(outputs["residual"].dtype)
    residual_rms = torch.sqrt(
        (outputs["residual"].square() * valid).sum()
        / valid.expand_as(outputs["residual"]).sum().clamp_min(1.0)
    )
    predicted_frames = torch.exp(outputs["pred_log_frames"])
    losses = {
        "loss_total": total,
        **endpoint_dict,
        **dynamics_dict,
        **regularization_dict,
        "loss_correction": correction,
        "loss_residual_vel": residual_vel,
        "loss_residual_acc": residual_acc,
        "loss_residual_jerk": residual_jerk,
        "loss_confidence": confidence_loss,
        "loss_trust": confidence_loss,
        "loss_correction_need": correction_need_loss,
        "loss_knot_density": knot_density_loss,
        "loss_scale_prior": scale_loss,
        "loss_confident_gate": confident_gate,
        "loss_gate_target": gate_target_loss,
        "loss_preserve_gate": preserve_gate,
        "loss_gate": gate_penalty,
        "loss_code_smooth": code_smooth,
        "loss_segment_pose": segment_pose,
        "loss_segment_vel": segment_vel,
        "loss_segment_acc": segment_acc,
        "loss_duration": duration_loss,
        "loss_fk_dynamics": dynamics,
        "loss_fk_regularization": regularization,
        "residual_rms": residual_rms,
        "duration_mae_frames": torch.abs(
            predicted_frames - lengths.to(predicted_frames.dtype)
        ).mean(),
        "duration_relative_error": (
            torch.abs(predicted_frames - lengths.to(predicted_frames.dtype))
            / lengths.to(predicted_frames.dtype).clamp_min(1.0)
        ).mean(),
    }
    losses.update(masked_group_means(outputs["confidence"], mask, "confidence"))
    losses.update(masked_group_means(confidence_target, mask, "confidence_target"))
    losses.update(masked_group_means(outputs["gates"], mask, "gate"))
    losses.update(
        confidence_metrics(
            outputs["confidence"],
            confidence_target,
            prepared["scaffold_errors"],
            mask,
        )
    )
    if uncertainty_adaptive:
        losses.update(
            masked_group_means(
                outputs["correction_need"], mask, "correction_need"
            )
        )
        losses.update(
            masked_group_means(
                correction_need_target, mask, "correction_need_target"
            )
        )
        losses.update(masked_group_means(outputs["knot_density"], mask, "knot_density"))
        losses.update(
            masked_group_means(knot_density_target, mask, "knot_density_target")
        )
        losses["retrieval_uncertainty"] = _masked_mean(
            outputs["retrieval_uncertainty"], mask
        )
        losses.update(
            articulator_scale_metrics(
                outputs["scale_weights"],
                mask,
                base_model.articulator_code_strides,
            )
        )
        for group_index, name in enumerate(ARTICULATOR_NAMES):
            for scale_index, stride in enumerate(
                base_model.articulator_code_strides[name]
            ):
                losses[f"knots_{name}_stride{stride}"] = outputs["knot_counts"][
                    :, group_index, scale_index
                ].to(dtype=x.dtype).mean()
    else:
        losses.update(
            scale_metrics(outputs["scale_weights"], mask, base_model.code_strides)
        )
    return total, losses


def run_train_step(model, fk, text_encoder, scaffold_provider, optimizer, batch, dataset, cfg, device):
    prepared = prepare_retrieval_batch(
        model, text_encoder, scaffold_provider, batch, dataset, cfg, device
    )
    total, losses = retrieval_adaptive_losses(model, fk, prepared, cfg)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg.get("train", {}).get("grad_clip", 1.0))
    )
    optimizer.step()
    return tensor_dict_to_float(losses)


@torch.no_grad()
def evaluate(
    model,
    fk,
    text_encoder,
    scaffold_provider,
    loader,
    dataset,
    cfg,
    device,
    max_batches=0,
    show_progress=True,
):
    model.eval()
    endpoint_cfg = cfg.get("loss", {})
    hand_weight = float(endpoint_cfg.get("hand_weight", 5.0))
    chunk_size = int(cfg.get("metrics", {}).get("fk_batch_size", 128))
    avg = ScalarAverager()
    for batch_index, batch in enumerate(
        tqdm(loader, desc="val", leave=False, disable=not show_progress)
    ):
        if max_batches and batch_index >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        prepared = prepare_retrieval_batch(
            model, text_encoder, scaffold_provider, batch, dataset, cfg, device
        )
        _total, losses = retrieval_adaptive_losses(model, fk, prepared, cfg)
        for name, prediction in (
            ("scaffold", prepared["scaffold"]),
            ("pred", prepared["outputs"]["prediction"]),
        ):
            pred_parts = prediction_parts_from_rot6d(
                prediction,
                prepared["mask"],
                prepared["target_parts"],
                fk,
                fk_chunk_size=chunk_size,
            )
            _endpoint, endpoint_dict = endpoint_losses(
                prediction,
                prepared["x"],
                prepared["mask"],
                prepared["lengths"],
                prepared["target_parts"],
                fk=fk,
                weights=endpoint_cfg,
                hand_weight=hand_weight,
                fk_chunk_size=chunk_size,
                pred_parts=pred_parts,
            )
            _dynamics, dynamics_dict = fk_temporal_dynamics_losses(
                prediction,
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
                pred_parts=pred_parts,
            )
            metrics = tensor_dict_to_float(endpoint_dict)
            metrics.update(tensor_dict_to_float(dynamics_dict))
            metrics["hand_path_ratio"] = float(
                path_ratio_metric(
                    fk,
                    prediction,
                    prepared["target_parts"],
                    prepared["mask"],
                    chunk_size=chunk_size,
                ).item()
            )
            avg.update(metrics, n=len(batch["name"]), prefix=name)
        avg.update(tensor_dict_to_float(losses), n=len(batch["name"]), prefix="adaptive")
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
            "model_type": str(
                cfg.get("model", {}).get(
                    "type", "retrieval_confidence_adaptive_field"
                )
            ),
            "retrieval_feature_names": RETRIEVAL_FEATURE_NAMES,
        },
        path,
    )


def init_wandb(args, cfg, out_dir, dist_info=None):
    if not args.wandb or not (dist_info or {}).get("is_main", True):
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
            "experiment": cfg.get("experiment_name", "retrieval_confidence_adaptive_field"),
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


def validation_selection_score(row, cfg):
    weights = cfg.get("selection", {}).get("weights", {})
    if weights:
        score = 0.0
        for name, weight in weights.items():
            key = name if str(name).startswith("val_") else f"val_{name}"
            if key not in row:
                raise KeyError(f"Selection metric {key!r} was not produced by validation.")
            score += float(weight) * float(row[key])
        return score
    return float(
        row.get(
            "val_pred_loss_endpoint",
            row.get("val_adaptive_loss_total", float("inf")),
        )
    )


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    print(
        "DDP startup: "
        f"host={os.uname().nodename} "
        f"slurm_rank={os.environ.get('SLURM_PROCID', 'none')} "
        f"world_size={os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1'))}",
        flush=True,
    )
    dist_info = setup_distributed(args)
    print(
        "DDP ready: "
        f"host={os.uname().nodename} rank={dist_info['rank']} "
        f"world_size={dist_info['world_size']} backend={dist_info['backend']}",
        flush=True,
    )
    set_seed(int(cfg.get("seed", 1234)) + int(dist_info.get("rank", 0)))
    device = resolve_distributed_device(cfg.get("device", "auto"), dist_info)
    text_device = torch.device(cfg.get("text", {}).get("device", "cpu"))
    out_dir = Path(cfg["output"]["out_dir"])
    if dist_info["is_main"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
    barrier(dist_info)

    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "val")
    train_dataset, train_loader, train_sampler = make_loader(
        cfg,
        train_split,
        limit=data_cfg.get("limit_train", 0),
        shuffle=True,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    val_dataset, val_loader, val_sampler = make_loader(
        cfg,
        val_split,
        limit=data_cfg.get("limit_val", 0),
        shuffle=False,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
    )
    rank_zero_print(
        dist_info,
        f"Loaded datasets: train={len(train_dataset)} val={len(val_dataset)} "
        f"world_size={dist_info['world_size']} "
        f"batch_per_rank={cfg.get('train', {}).get('batch_size', 2)}",
    )

    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, train_dataset, device)
    retrieval_bank = validate_train_only_retrieval_bank(cfg, scaffold_provider)
    rank_zero_print(
        dist_info, f"Retrieval bank: {json.dumps(retrieval_bank, sort_keys=True)}"
    )
    model = build_retrieval_adaptive_model(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})

    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    optimizer_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer_state = checkpoint.get("optimizer")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(
            checkpoint.get("metrics", {}).get(
                "selection_score",
                checkpoint.get("metrics", {}).get(
                    "val_pred_loss_endpoint", float("inf")
                ),
            )
        )
    model = wrap_model(model, dist_info, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    epochs = int(train_cfg.get("epochs", 40))
    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    val_every = int(train_cfg.get("val_every", 1))
    save_every = int(train_cfg.get("save_every", 1))
    start_time = time.time()
    wandb_run = init_wandb(args, cfg, out_dir, dist_info=dist_info)

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        model.train()
        average = ScalarAverager()
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            disable=not dist_info["is_main"],
        )
        for batch_index, batch in enumerate(progress):
            if max_train_batches and batch_index >= max_train_batches:
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
            average.update(losses, n=len(batch["name"]), prefix="train")
            if dist_info["is_main"]:
                progress.set_postfix(
                    loss=f"{losses['loss_total']:.4f}",
                    trust=f"{losses['loss_trust']:.4f}",
                    res=f"{losses['residual_rms']:.4f}",
                )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
        }
        row.update(distributed_mean_scalars(average.mean(), device, dist_info))
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
            score = validation_selection_score(row, cfg)
            row["selection_score"] = score
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
