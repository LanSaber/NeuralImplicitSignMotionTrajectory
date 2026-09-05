from __future__ import annotations

import argparse
import copy
import hashlib
import json
import inspect
import math
import os
import random
import time
from pathlib import Path
from dataclasses import fields, is_dataclass, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    masked_feature_l1,
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
    coarse_and_residual_losses,
    duration_regression_loss,
    local_field_regularization,
    prior_and_residual_losses,
)
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    validate_train_only_retrieval_bank,
)


DUAL_MODE_MODEL_TYPE = "dual_mode_continuous_trajectory_field"
SENTENCE_MEMORY_MODEL_TYPE = "sentence_memory_continuous_trajectory_field"
TRAJECTORY_CONTRACT_VERSIONS = {
    "continuous_trajectory_field": 1,
    DUAL_MODE_MODEL_TYPE: 2,
    SENTENCE_MEMORY_MODEL_TYPE: 3,
}
WORD_PRIOR_MODES = {"dropout", "off", "on"}
SENTENCE_MEMORY_TRAIN_MODES = {"dropout", "off", "on"}
SENTENCE_MEMORY_EVAL_MODES = {"off", "on", "shuffled"}
WORD_PRIOR_PART_NAMES = ("body", "left_hand", "right_hand", "face")


def configured_model_type(cfg):
    return str(
        cfg.get("model", {}).get("type", "continuous_trajectory_field")
    ).lower()


def is_dual_mode(cfg):
    return configured_model_type(cfg) in {
        DUAL_MODE_MODEL_TYPE,
        SENTENCE_MEMORY_MODEL_TYPE,
    }


def is_word_prior_model(cfg):
    return configured_model_type(cfg) in {
        DUAL_MODE_MODEL_TYPE,
        SENTENCE_MEMORY_MODEL_TYPE,
    }


def is_sentence_memory_model(cfg):
    return configured_model_type(cfg) == SENTENCE_MEMORY_MODEL_TYPE


def sentence_memory_enabled(cfg):
    """Return the global v3 retrieval switch, independent of requested modes."""

    return bool(cfg.get("sentence_memory", {}).get("enabled", True))


def configured_word_prior_train_mode(cfg):
    mode = str(
        cfg.get("conditioning", {}).get("word_prior_train_mode", "dropout")
    ).lower()
    if mode not in WORD_PRIOR_MODES:
        raise ValueError(
            "conditioning.word_prior_train_mode must be one of "
            f"{sorted(WORD_PRIOR_MODES)}, got {mode!r}"
        )
    return mode


def configured_word_prior_eval_modes(cfg):
    configured = cfg.get("eval", {}).get("word_prior_modes")
    if configured is None:
        return ("off", "on")
    if isinstance(configured, str):
        configured = [configured]
    modes = tuple(dict.fromkeys(str(mode).lower() for mode in configured))
    if not modes:
        raise ValueError("eval.word_prior_modes must contain at least one mode")
    invalid = sorted(set(modes) - {"off", "on"})
    if invalid:
        raise ValueError(
            "eval.word_prior_modes supports only deterministic 'off' and 'on' "
            f"modes, got {invalid}"
        )
    return modes


def requires_scaffold_provider(cfg):
    if not is_dual_mode(cfg):
        return True
    if configured_word_prior_train_mode(cfg) != "off":
        return True
    if is_sentence_memory_model(cfg):
        mode = str(
            cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off")
        ).lower()
        if mode not in {"off", "on"}:
            raise ValueError(
                "eval.sentence_memory_word_prior_mode must be 'off' or 'on'"
            )
        return mode == "on"
    return any(mode != "off" for mode in configured_word_prior_eval_modes(cfg))


def configured_sentence_memory_train_mode(cfg):
    if not sentence_memory_enabled(cfg):
        return "off"
    mode = str(
        cfg.get("conditioning", {}).get(
            "sentence_memory_train_mode", "dropout"
        )
    ).lower()
    if mode not in SENTENCE_MEMORY_TRAIN_MODES:
        raise ValueError(
            "conditioning.sentence_memory_train_mode must be one of "
            f"{sorted(SENTENCE_MEMORY_TRAIN_MODES)}, got {mode!r}"
        )
    return mode


def configured_sentence_memory_eval_modes(cfg):
    if not sentence_memory_enabled(cfg):
        return ("off",)
    configured = cfg.get("eval", {}).get("sentence_memory_modes")
    if configured is None:
        return ("off", "on")
    if isinstance(configured, str):
        configured = [configured]
    modes = tuple(dict.fromkeys(str(mode).lower() for mode in configured))
    if not modes:
        raise ValueError("eval.sentence_memory_modes must contain at least one mode")
    invalid = sorted(set(modes) - SENTENCE_MEMORY_EVAL_MODES)
    if invalid:
        raise ValueError(
            "eval.sentence_memory_modes supports only deterministic "
            f"{sorted(SENTENCE_MEMORY_EVAL_MODES)} modes, got {invalid}"
        )
    return modes


def sentence_memory_provider_required(modes):
    """Return whether resolved modes include retrieval-backed inference."""

    if isinstance(modes, str):
        modes = (modes,)
    return any(str(mode).lower() in {"on", "shuffled"} for mode in modes)


def requires_sentence_memory_provider(cfg):
    if not is_sentence_memory_model(cfg) or not sentence_memory_enabled(cfg):
        return False
    return (
        configured_sentence_memory_train_mode(cfg) != "off"
        or sentence_memory_provider_required(
            configured_sentence_memory_eval_modes(cfg)
        )
    )


def build_sentence_memory_provider(cfg, text_encoder, dataset=None):
    # Keep the optional v3 dependency out of the v1/v2 import path.
    from NIAF.continuous_trajectory_field.sentence_memory import (
        SentenceMemoryProvider,
    )

    identity = (
        text_encoder.checkpoint_identity()
        if hasattr(text_encoder, "checkpoint_identity")
        else {
            "model_path": configured_text_encoder_identity(cfg),
            "text_dim": int(text_encoder.text_dim),
        }
    )
    constructor_parameters = inspect.signature(
        SentenceMemoryProvider
    ).parameters
    kwargs = {"text_encoder_identity": identity}
    if "dataset" in constructor_parameters:
        kwargs["dataset"] = dataset
    return SentenceMemoryProvider(cfg, **kwargs)


def configured_text_encoder_identity(cfg):
    return str(
        cfg.get("text", {}).get("model_path", "deps/flan-t5-base")
    ).replace("\\", "/")


def checkpoint_contract(cfg):
    model_type = configured_model_type(cfg)
    try:
        version = TRAJECTORY_CONTRACT_VERSIONS[model_type]
    except KeyError as error:
        raise ValueError(f"Unsupported model type {model_type!r}") from error
    return model_type, version


def sentence_memory_behavior_identity(cfg):
    """Return the strict, bank-independent v3 behavior contract."""

    memory_cfg = dict(cfg.get("sentence_memory", {}))
    model_cfg = dict(cfg.get("model", {}))
    conditioning_cfg = dict(cfg.get("conditioning", {}))
    duration_cfg = dict(cfg.get("duration", {}))
    filter_cfg = dict(memory_cfg.get("filter") or {})
    k = max(int(memory_cfg.get("k", 8)), 1)
    top_m = max(int(memory_cfg.get("top_m", max(k, 64))), k)
    key_dim = memory_cfg.get("key_dim")
    payload = {
        "schema_version": 3,
        "trajectory": {
            "context_hidden_dim": int(model_cfg.get("context_hidden_dim", 256)),
            "context_layers": int(model_cfg.get("context_layers", 3)),
            "field_hidden_dim": int(model_cfg.get("field_hidden_dim", 256)),
            "field_depth": int(model_cfg.get("field_depth", 4)),
            "max_local_fields": int(model_cfg.get("max_local_fields", 24)),
            "frames_per_local_field": int(
                model_cfg.get("frames_per_local_field", 20)
            ),
            "minimum_local_width": float(
                model_cfg.get("minimum_local_width", 0.06)
            ),
            "maximum_local_width": float(
                model_cfg.get("maximum_local_width", 0.50)
            ),
            "quantile_temperature": float(
                model_cfg.get("quantile_temperature", 0.02)
            ),
            # The dual-mode factory resolves these two settings explicitly for
            # both v2 and v3, regardless of inherited config values.
            "local_center_mode": "uniform",
            "use_retrieval_guidance": False,
            "local_window_epsilon": float(
                model_cfg.get("local_window_epsilon", 1e-4)
            ),
            "part_specific_local_experts": bool(
                model_cfg.get("part_specific_local_experts", False)
            ),
            "time_dependent_local_gates": bool(
                model_cfg.get("time_dependent_local_gates", False)
            ),
            "local_body_omega0_first": float(
                model_cfg.get("local_body_omega0_first", 15.0)
            ),
            "local_hand_omega0_first": float(
                model_cfg.get("local_hand_omega0_first", 30.0)
            ),
            "local_face_omega0_first": float(
                model_cfg.get("local_face_omega0_first", 20.0)
            ),
            "omega0_first": float(model_cfg.get("omega0_first", 20.0)),
            "omega0_hidden": float(model_cfg.get("omega0_hidden", 1.0)),
            "residual_amplitude": float(
                model_cfg.get("residual_amplitude", 0.10)
            ),
            "residual_amplitude_learnable": bool(
                model_cfg.get("residual_amplitude_learnable", True)
            ),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "temporal_slot_count": int(
                conditioning_cfg.get("temporal_slot_count", 16)
            ),
            "temporal_slot_layers": int(
                conditioning_cfg.get("temporal_slot_layers", 2)
            ),
            "temporal_slot_heads": int(
                conditioning_cfg.get("temporal_slot_heads", 8)
            ),
            "context_fps": float(conditioning_cfg.get("context_fps", 20.0)),
            "initial_duration_seconds": float(
                duration_cfg.get("initial_seconds", 4.0)
            ),
            "minimum_duration_seconds": float(
                duration_cfg.get("min_seconds", 0.8)
            ),
            "maximum_duration_seconds": float(
                duration_cfg.get("max_seconds", 20.0)
            ),
        },
        "model": {
            "motion_dim": int(memory_cfg.get("motion_dim", 256)),
            "key_dim": int(key_dim) if key_dim is not None else "text_encoder_dim",
            "attention_layers": int(memory_cfg.get("attention_layers", 2)),
            "attention_heads": int(memory_cfg.get("attention_heads", 8)),
            "score_temperature": max(
                float(memory_cfg.get("score_temperature", 0.10)), 1e-4
            ),
            "duration_weight": max(
                float(memory_cfg.get("duration_weight", 0.10)), 0.0
            ),
            "retrieval_prior_scale": float(
                memory_cfg.get("retrieval_prior_scale", 1.0)
            ),
            "gate_initial_bias": float(
                memory_cfg.get("gate_initial_bias", -2.2)
            ),
        },
        "retrieval": {
            "enabled": bool(memory_cfg.get("enabled", True)),
            "k": k,
            "top_m": top_m,
            "duration_weight": max(
                float(memory_cfg.get("duration_weight", 0.10)), 0.0
            ),
            "sampling": str(
                memory_cfg.get(
                    "sampling", memory_cfg.get("train_sampling", "weighted")
                )
            ).lower(),
            "sampling_temperature": max(
                float(memory_cfg.get("sampling_temperature", 0.07)), 1e-6
            ),
            "candidate_dropout_probability": float(
                memory_cfg.get("candidate_dropout_probability", 0.10)
            ),
            "seed": int(cfg.get("seed", memory_cfg.get("seed", 1234))),
            "filter_policy": {
                "exclude_self": bool(filter_cfg.get("exclude_self", True)),
                "exclude_same_source": bool(
                    filter_cfg.get("exclude_same_source", True)
                ),
                "exclude_same_group": bool(
                    filter_cfg.get("exclude_same_group", True)
                ),
                "exclude_exact_text_train": bool(
                    filter_cfg.get("exclude_exact_text_train", True)
                ),
                "exclude_exact_text_eval": bool(
                    filter_cfg.get("exclude_exact_text_eval", False)
                ),
                "group_fields": list(
                    filter_cfg.get("group_fields")
                    or ("sentence_id", "source_name", "name")
                ),
            },
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "digest": digest}


def sentence_memory_resume_identity(cfg):
    """Hash every behavior-affecting setting required for exact v3 resume.

    Output naming, bank placement, and the requested terminal epoch are
    operational and may change. Bank payloads, neighbor tables, the text
    frontend, RNG state, and Phase-B gate evidence have their own persisted
    identities. Everything else remains part of the exact-continuation
    contract, including data/batching, losses, trainability, optimizer,
    validation, and checkpoint-selection policy.
    """

    payload = copy.deepcopy(dict(cfg))
    payload.pop("experiment_name", None)
    payload.pop("output", None)

    train_cfg = payload.get("train")
    if isinstance(train_cfg, dict):
        # Extending the terminal epoch does not alter any epoch-indexed update.
        train_cfg.pop("epochs", None)
        # This is a launch control, and v3 rejects the true value separately.
        train_cfg.pop("reset_optimizer_on_resume", None)
        # These inputs affect only a fresh initialization. Their provenance is
        # already persisted in the checkpoint/parity/gate records and they are
        # deliberately absent from a later --resume invocation.
        train_cfg.pop("base_checkpoint", None)
        train_cfg.pop("warm_start_checkpoint", None)
        train_cfg.pop("reset_local_branch_on_warm_start", None)

    memory_cfg = payload.get("sentence_memory")
    if isinstance(memory_cfg, dict):
        for key in (
            "bank_dir",
            "local_bank_env",
            "resolved_identity",
            "resolved_behavior_identity",
        ):
            memory_cfg.pop(key, None)

    safety_cfg = payload.get("sentence_memory_safety")
    if isinstance(safety_cfg, dict):
        safety_cfg.pop("v2_to_v3_text_only_parity", None)
        phase_b_cfg = safety_cfg.get("phase_b")
        if isinstance(phase_b_cfg, dict):
            # The accepted report is validated through its own canonical digest.
            phase_b_cfg.pop("gate_report", None)
            phase_b_cfg.pop("resolved_scientific_gate", None)

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "payload": payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def validate_sentence_memory_resume_identity(checkpoint, cfg, source="checkpoint"):
    """Reject a v3 resume whose active training behavior differs."""

    actual = checkpoint.get("sentence_memory_resume_identity")
    if not isinstance(actual, dict) or int(actual.get("schema_version", -1)) != 1:
        raise RuntimeError(f"{source} has no exact v3 resume-behavior identity")
    actual_payload = actual.get("payload")
    if not isinstance(actual_payload, dict):
        raise RuntimeError(f"{source} has a malformed v3 resume-behavior payload")
    actual_digest = hashlib.sha256(
        json.dumps(actual_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if actual.get("digest") != actual_digest:
        raise RuntimeError(f"{source} has an invalid v3 resume-behavior digest")
    expected = sentence_memory_resume_identity(cfg)
    if actual_digest != expected["digest"]:
        changed_sections = sorted(
            key
            for key in set(actual_payload) | set(expected["payload"])
            if actual_payload.get(key) != expected["payload"].get(key)
        )
        raise RuntimeError(
            f"{source} cannot be resumed exactly with the active config; "
            f"changed behavior sections={changed_sections}"
        )
    return expected


def configured_phase_b_gate_report(cfg):
    value = (
        cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("gate_report")
    )
    return Path(value) if value else None


def validate_phase_b_scientific_gate_settings(settings, source="gate report"):
    """Require the canonical gate and at least the documented statistical rigor."""

    if not isinstance(settings, dict):
        raise RuntimeError(f"{source} has no explicit scientific-gate settings")
    required = {
        "comparison",
        "primary_subset",
        "gate_metric",
        "bootstrap_samples",
        "bootstrap_seed",
        "confidence",
        "minimum_pairs",
        "duration_tolerance_seconds",
        "hand_path_max_relative_degradation",
        "parity_tolerance",
    }
    missing = sorted(required - set(settings))
    extra = sorted(set(settings) - required)
    if missing or extra:
        raise RuntimeError(
            f"{source} scientific-gate settings are not canonical: "
            f"missing={missing}, unsupported={extra}"
        )
    if settings["comparison"] != "flow":
        raise RuntimeError(f"{source} must use comparison='flow'")
    if settings["primary_subset"] != "novel_text":
        raise RuntimeError(f"{source} must use primary_subset='novel_text'")
    if settings["gate_metric"] != "ndtw":
        raise RuntimeError(f"{source} must use gate_metric='ndtw'")

    def integer(name):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"{source} setting {name} must be an integer")
        parsed = int(value)
        if float(value) != float(parsed):
            raise RuntimeError(f"{source} setting {name} must be an integer")
        return parsed

    def finite_float(name):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"{source} setting {name} must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise RuntimeError(f"{source} setting {name} must be finite")
        return parsed

    bootstrap_samples = integer("bootstrap_samples")
    bootstrap_seed = integer("bootstrap_seed")
    confidence = finite_float("confidence")
    minimum_pairs = integer("minimum_pairs")
    duration_tolerance = finite_float("duration_tolerance_seconds")
    hand_tolerance = finite_float("hand_path_max_relative_degradation")
    parity_tolerance = finite_float("parity_tolerance")
    if bootstrap_samples < 10_000:
        raise RuntimeError(f"{source} must use at least 10000 bootstrap samples")
    if bootstrap_seed != 1234:
        raise RuntimeError(f"{source} must use canonical bootstrap_seed=1234")
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{source} must use 95% confidence")
    if minimum_pairs < 2:
        raise RuntimeError(f"{source} must require at least two paired samples")
    for name, value, maximum in (
        ("duration_tolerance_seconds", duration_tolerance, 1e-7),
        ("hand_path_max_relative_degradation", hand_tolerance, 0.02),
        ("parity_tolerance", parity_tolerance, 1e-7),
    ):
        if value < 0.0 or value > maximum:
            raise RuntimeError(
                f"{source} setting {name}={value} weakens the canonical "
                f"maximum {maximum}"
            )
    return {
        "comparison": "flow",
        "primary_subset": "novel_text",
        "gate_metric": "ndtw",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
        "minimum_pairs": minimum_pairs,
        "duration_tolerance_seconds": duration_tolerance,
        "hand_path_max_relative_degradation": hand_tolerance,
        "parity_tolerance": parity_tolerance,
    }


def validate_phase_b_launch(
    cfg, *, warm_start=None, resume=None, phase_b_gate_report=None
):
    sentence_memory_model = is_sentence_memory_model(cfg)
    phase_b_enabled = bool(
        sentence_memory_model
        and cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("enabled", False)
    )
    if sentence_memory_model and warm_start is not None and not phase_b_enabled:
        raise ValueError(
            "v3 --warm_start is reserved for a model-only Phase-B run. "
            "Initialize Phase A with --base_checkpoint or continue it with --resume."
        )
    if (
        sentence_memory_model
        and resume is not None
        and bool(cfg.get("train", {}).get("reset_optimizer_on_resume", False))
    ):
        raise ValueError(
            "Exact v3 --resume cannot use train.reset_optimizer_on_resume=true; "
            "use the Phase-B --warm_start workflow for a fresh optimizer."
        )
    if phase_b_enabled and warm_start is None and resume is None:
        raise ValueError(
            "Sentence-memory Phase B must start with --warm_start <phase-a-v3.pt> "
            "or continue an existing Phase-B run with --resume."
        )
    gate_report = (
        Path(phase_b_gate_report)
        if phase_b_gate_report is not None
        else configured_phase_b_gate_report(cfg)
    )
    if phase_b_enabled and warm_start is not None and gate_report is None:
        raise ValueError(
            "Sentence-memory Phase B requires --phase_b_gate_report pointing "
            "to an accepted scientific gate artifact generated from the selected "
            "Phase-A checkpoint."
        )


def validate_phase_b_resume_checkpoint(cfg, checkpoint, source="checkpoint"):
    phase_b_enabled = bool(
        is_sentence_memory_model(cfg)
        and cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("enabled", False)
    )
    if not phase_b_enabled:
        return
    checkpoint_cfg = checkpoint.get("config") or {}
    checkpoint_phase_b = bool(
        checkpoint_cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("enabled", False)
    )
    if not checkpoint_phase_b:
        raise RuntimeError(
            f"{source} is not a Phase-B checkpoint; initialize Phase B from "
            "Phase A with --warm_start, not --resume"
        )
    gate = checkpoint.get("phase_b_scientific_gate")
    if not isinstance(gate, dict) or not bool(gate.get("accepted", False)):
        raise RuntimeError(
            f"{source} has no accepted Phase-B scientific-gate provenance"
        )
    validate_phase_b_scientific_gate_settings(
        gate.get("settings"), source=f"{source} Phase-B gate provenance"
    )


def validate_phase_b_warm_start_checkpoint(cfg, checkpoint, source="checkpoint"):
    """Fail closed unless a Phase-B warm start comes from frozen-base Phase A."""

    phase_b_enabled = bool(
        is_sentence_memory_model(cfg)
        and cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("enabled", False)
    )
    if not phase_b_enabled:
        return
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise RuntimeError(
            f"{source} has no saved config, so it cannot be proven to be a "
            "Phase-A v3 checkpoint"
        )
    checkpoint_phase_b_cfg = (
        checkpoint_cfg.get("sentence_memory_safety", {}).get("phase_b", {})
    )
    if "enabled" not in checkpoint_phase_b_cfg:
        raise RuntimeError(
            f"{source} has no explicit Phase-A marker "
            "(sentence_memory_safety.phase_b.enabled=false)"
        )
    if bool(checkpoint_phase_b_cfg["enabled"]):
        raise RuntimeError(
            f"{source} is a Phase-B checkpoint; Phase B must warm-start from "
            "a Phase-A checkpoint"
        )
    checkpoint_train_cfg = checkpoint_cfg.get("train", {})
    if not bool(checkpoint_train_cfg.get("freeze_base", False)):
        raise RuntimeError(
            f"{source} is not a frozen-base Phase-A checkpoint "
            "(train.freeze_base must be true)"
        )
    if checkpoint_train_cfg.get("unfreeze_base_prefixes"):
        raise RuntimeError(
            f"{source} is not a frozen-base Phase-A checkpoint "
            "(train.unfreeze_base_prefixes must be empty)"
        )
    if not checkpoint_train_cfg.get("base_checkpoint"):
        raise RuntimeError(
            f"{source} does not record the v2 base checkpoint used to initialize "
            "Phase A"
        )
    teacher_checkpoint = (
        cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("teacher_checkpoint")
    )
    if not teacher_checkpoint:
        raise RuntimeError(
            "Phase-B config has no teacher_checkpoint to bind to the Phase-A base"
        )
    phase_a_base = Path(checkpoint_train_cfg["base_checkpoint"]).resolve()
    active_teacher = Path(teacher_checkpoint).resolve()
    if phase_a_base != active_teacher:
        raise RuntimeError(
            f"{source} was initialized from {phase_a_base}, but Phase B uses "
            f"teacher {active_teacher}"
        )
    parity = checkpoint.get("v2_to_v3_text_only_parity")
    if not isinstance(parity, dict) or not bool(parity.get("passed", False)):
        raise RuntimeError(
            f"{source} has no passing stored v2-to-v3 text-only parity proof"
        )


def _phase_b_gate_source_paths(report):
    source_files = report.get("provenance", {}).get("source_files", {})
    required = {
        f"{mode}_{kind}": source_files.get(f"{mode}_{kind}", {}).get("path")
        for mode in (
            "text_only",
            "sentence_memory",
            "shuffled_sentence_memory",
        )
        for kind in ("export_summary", "default_dtw", "pa_dtw")
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise RuntimeError(
            "Phase-B scientific-gate report is missing source paths: "
            f"{missing}"
        )
    return required


def validate_phase_b_scientific_gate(
    cfg, checkpoint, checkpoint_path, report_path
):
    """Recompute and bind the Phase-B gate to its exact Phase-A evidence."""

    from NIAF.continuous_trajectory_field.scripts.analyze_sentence_memory_phase_b_gate import (
        SCHEMA_NAME,
        SCHEMA_VERSION,
        analyze_phase_b_gate,
    )

    report_path = Path(report_path).resolve()
    if not report_path.is_file():
        raise RuntimeError(
            f"Phase-B scientific-gate report does not exist: {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read Phase-B scientific-gate report {report_path}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise RuntimeError("Phase-B scientific-gate report must be a JSON object")
    if (
        report.get("schema_name") != SCHEMA_NAME
        or int(report.get("schema_version", -1)) != SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Phase-B scientific-gate report has an unsupported schema: "
            f"{report.get('schema_name')!r}/v{report.get('schema_version')!r}"
        )
    if not bool(report.get("accepted", False)):
        raise RuntimeError(
            f"Phase-B scientific gate rejected the run: {report_path}"
        )

    source_paths = _phase_b_gate_source_paths(report)
    settings = validate_phase_b_scientific_gate_settings(
        report.get("settings"), source=str(report_path)
    )
    recomputed = analyze_phase_b_gate(
        export_paths={
            mode: source_paths[f"{mode}_export_summary"]
            for mode in (
                "text_only",
                "sentence_memory",
                "shuffled_sentence_memory",
            )
        },
        dtw_paths={
            mode: {
                "default": source_paths[f"{mode}_default_dtw"],
                "pa": source_paths[f"{mode}_pa_dtw"],
            }
            for mode in (
                "text_only",
                "sentence_memory",
                "shuffled_sentence_memory",
            )
        },
        phase_a_checkpoint=Path(checkpoint_path).resolve(),
        comparison=settings.get("comparison", "flow"),
        gate_metric=settings.get("gate_metric", "ndtw"),
        bootstrap_samples=int(settings.get("bootstrap_samples", 10_000)),
        bootstrap_seed=int(settings.get("bootstrap_seed", 1234)),
        confidence=float(settings.get("confidence", 0.95)),
        min_pairs=int(settings.get("minimum_pairs", 2)),
        duration_tolerance=float(
            settings.get("duration_tolerance_seconds", 1e-7)
        ),
        hand_path_max_relative_degradation=float(
            settings.get("hand_path_max_relative_degradation", 0.02)
        ),
        parity_tolerance=float(settings.get("parity_tolerance", 1e-7)),
    )
    if not bool(recomputed.get("accepted", False)):
        failed = sorted(
            name
            for name, check in recomputed.get("checks", {}).items()
            if bool(check.get("evaluated", False))
            and not bool(check.get("passed", False))
        )
        raise RuntimeError(
            "Phase-B scientific gate no longer passes when recomputed; "
            f"failed checks: {failed}"
        )
    recomputed_settings = validate_phase_b_scientific_gate_settings(
        recomputed.get("settings"), source="recomputed Phase-B gate"
    )
    if recomputed_settings != settings:
        raise RuntimeError(
            "Phase-B scientific-gate settings differ after recomputation"
        )
    if report.get("gate_identity") != recomputed.get("gate_identity"):
        raise RuntimeError(
            "Phase-B scientific-gate identity does not match recomputed evidence"
        )
    expected_bank_id = (
        cfg.get("sentence_memory", {}).get("resolved_identity") or {}
    ).get("bank_id")
    actual_bank_id = recomputed.get("provenance", {}).get("bank_id")
    if not expected_bank_id or str(actual_bank_id) != str(expected_bank_id):
        raise RuntimeError(
            "Phase-B gate bank identity differs from the active bank: "
            f"gate={actual_bank_id!r}, active={expected_bank_id!r}"
        )
    checkpoint_parity = checkpoint.get("v2_to_v3_text_only_parity")
    if checkpoint_parity != (
        recomputed.get("checks", {}).get("stored_v2_parity")
    ):
        # The analyzer normalizes the stored proof into a check, so compare the
        # numerical contract rather than accepting an unrelated checkpoint.
        parity_check = recomputed.get("checks", {}).get("stored_v2_parity", {})
        if not (
            isinstance(checkpoint_parity, dict)
            and bool(parity_check.get("passed", False))
            and float(checkpoint_parity.get("prediction_max_abs", math.inf))
            == float(parity_check.get("prediction_max_abs", math.inf))
            and float(checkpoint_parity.get("duration_max_abs", math.inf))
            == float(parity_check.get("duration_max_abs", math.inf))
        ):
            raise RuntimeError(
                "Phase-B gate parity evidence does not match the warm-start checkpoint"
            )
    resolved = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "accepted": True,
        "report_path": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "gate_identity": recomputed["gate_identity"],
        "settings": recomputed_settings,
        "phase_a_checkpoint": recomputed["provenance"]["checkpoint"],
        "bank_id": actual_bank_id,
    }
    cfg.setdefault("sentence_memory_safety", {}).setdefault("phase_b", {})[
        "resolved_scientific_gate"
    ] = resolved
    return resolved


def set_sentence_memory_provider_epoch_from_checkpoint(provider, checkpoint):
    """Use the saved training epoch for deterministic standalone retrieval."""

    epoch = int(checkpoint.get("epoch", 0))
    if provider is not None:
        provider.set_epoch(epoch)
    return epoch


def evaluated_loader_sample_count(loader, max_batches=0):
    """Return the exact number of local rows consumed by an evaluation pass."""

    sampler = getattr(loader, "sampler", None)
    total = len(sampler) if sampler is not None else len(loader.dataset)
    max_batches = max(int(max_batches), 0)
    if max_batches:
        batch_size = max(int(getattr(loader, "batch_size", 1) or 1), 1)
        total = min(total, max_batches * batch_size)
    return int(total)


def distributed_sample_weighted_mean_scalars(
    values, local_sample_count, device, dist_info
):
    """Reduce rank-local sample means without assuming equal shard sizes."""

    if not dist_info.get("enabled", False):
        return values
    world_size = int(dist_info.get("world_size") or dist.get_world_size())
    rank_keys = [None] * world_size
    dist.all_gather_object(rank_keys, sorted(values))
    keys = sorted({key for values_on_rank in rank_keys for key in values_on_rank})
    if not keys:
        return {}
    local_count = float(local_sample_count)
    numerators = [
        float(values[key]) * local_count if key in values else 0.0 for key in keys
    ]
    denominators = [local_count if key in values else 0.0 for key in keys]
    tensor = torch.tensor(
        numerators + denominators,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = len(keys)
    total_counts = tensor[count:]
    if bool((total_counts <= 0).any()):
        raise RuntimeError("Distributed evaluation consumed no samples for a metric")
    means = tensor[:count] / total_counts
    return {key: float(value) for key, value in zip(keys, means.cpu().tolist())}


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
    parser.add_argument(
        "--sentence_memory_k",
        type=int,
        default=None,
        help="Override sentence_memory.k for a fresh K-ablation run.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize a v3 sentence-memory model from a compatible v2 "
            "text checkpoint, while resetting epoch and optimizer state."
        ),
    )
    parser.add_argument(
        "--warm_start",
        type=Path,
        default=None,
        help=(
            "Load same-contract model weights only and reset epoch/optimizer. "
            "For v3 this is the Phase-A to Phase-B transition."
        ),
    )
    parser.add_argument(
        "--phase_b_gate_report",
        type=Path,
        default=None,
        help=(
            "Accepted Phase-B scientific-gate JSON generated from the exact "
            "Phase-A checkpoint supplied to --warm_start."
        ),
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
    if args.sentence_memory_k is not None:
        if int(args.sentence_memory_k) < 1:
            raise ValueError("--sentence_memory_k must be positive")
        cfg.setdefault("sentence_memory", {})["k"] = int(
            args.sentence_memory_k
        )
    if args.out_dir is not None:
        cfg.setdefault("output", {})["out_dir"] = str(args.out_dir)
    if args.base_checkpoint is not None:
        cfg.setdefault("train", {})["base_checkpoint"] = str(
            args.base_checkpoint
        )
    if args.warm_start is not None:
        cfg.setdefault("train", {})["warm_start_checkpoint"] = str(args.warm_start)
    if args.phase_b_gate_report is not None:
        cfg.setdefault("sentence_memory_safety", {}).setdefault(
            "phase_b", {}
        )["gate_report"] = str(args.phase_b_gate_report)
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


def _per_sample_compact_l1(prediction, target, mask, hand_weight):
    return torch.stack(
        [
            masked_feature_l1(
                prediction[index : index + 1],
                target[index : index + 1],
                mask[index : index + 1],
                hand_weight=hand_weight,
            )
            for index in range(prediction.shape[0])
        ]
    )


def _per_sample_part_compact_l1(prediction, target, mask):
    """Return independent body/LH/RH/face errors for each batch item."""

    part_slices = (
        slice(0, 60),
        slice(60, 150),
        slice(150, 240),
        slice(240, 256),
    )
    per_part = []
    for part_slice in part_slices:
        difference = (prediction[..., part_slice] - target[..., part_slice]).abs()
        numerator = (difference * mask[..., None].to(difference.dtype)).sum(
            dim=(1, 2)
        )
        denominator = (
            mask.sum(dim=1).to(difference.dtype)
            * max(part_slice.stop - part_slice.start, 1)
        ).clamp_min(1.0)
        per_part.append(numerator / denominator)
    return torch.stack(per_part, dim=-1)


def _prepared_word_forward_kwargs(prepared, batch, cfg):
    if not is_word_prior_model(cfg):
        return {}
    adapter_context = prepared["adapter_context"]
    return {
        "word_prior_context": adapter_context,
        "word_prior_mask": batch["mask"] if adapter_context is not None else None,
        "word_prior_features": prepared["retrieval"],
        "word_prior_available": prepared["word_prior_available"],
    }


def phase_b_off_distillation(
    sentence_off_teacher,
    prepared,
    batch,
    phase_b_cfg,
    device,
):
    """Distill the differentiable v3 text-only path from the frozen v2 teacher.

    ``prepare_field_batch`` produces the student text-only rows alongside the
    primary rows in one call through the DDP wrapper.  This helper only runs the
    frozen teacher and forms the loss, avoiding a second trainable forward
    before backward.
    """

    if sentence_off_teacher is None:
        raise RuntimeError(
            "Phase-B off distillation is enabled without a frozen teacher"
        )
    teacher_kwargs = {
        "text_tokens": prepared["text_tokens"],
        "query_times": prepared["tau"],
        "text_mask": prepared["text_mask"],
        "time_domain": "normalized",
        "query_mask": batch["mask"],
        "word_prior_available": torch.zeros(
            prepared["text_tokens"].shape[0],
            dtype=torch.bool,
            device=device,
        ),
    }
    with torch.no_grad():
        teacher_prediction = sentence_off_teacher(**teacher_kwargs)[
            "prediction"
        ].detach()
    student_prediction = prepared.get("phase_b_student_off_prediction")
    if student_prediction is None:
        raise RuntimeError(
            "Phase-B batch has no differentiable text-only student prediction"
        )
    valid_frames = batch["mask"].bool()
    if bool(valid_frames.any()):
        loss = F.smooth_l1_loss(
            student_prediction[valid_frames],
            teacher_prediction[valid_frames],
            beta=float(phase_b_cfg.get("huber_beta", 0.1)),
        )
    else:
        loss = student_prediction.sum() * 0.0
    return teacher_prediction, student_prediction, loss


def sentence_memory_off_baseline(
    model,
    prepared,
    batch,
    cfg,
    device,
    *,
    frozen_teacher_prediction=None,
):
    """Return the fixed safety baseline, preferring the Phase-B teacher."""

    if frozen_teacher_prediction is not None:
        return frozen_teacher_prediction.detach()
    word_kwargs = _prepared_word_forward_kwargs(prepared, batch, cfg)
    with torch.no_grad():
        return unwrap_model(model)(
            text_tokens=prepared["text_tokens"],
            query_times=prepared["tau"],
            text_mask=prepared["text_mask"],
            time_domain="normalized",
            query_mask=batch["mask"],
            **word_kwargs,
            sentence_memory_available=torch.zeros(
                prepared["text_tokens"].shape[0],
                dtype=torch.bool,
                device=device,
            ),
        )["prediction"].detach()


def _word_prior_availability(batch_size, cfg, device, mode):
    mode = str(mode).lower()
    if mode not in WORD_PRIOR_MODES:
        raise ValueError(
            f"word_prior_mode must be one of {sorted(WORD_PRIOR_MODES)}, got {mode!r}"
        )
    if mode == "off":
        return torch.zeros(int(batch_size), dtype=torch.bool, device=device)
    if mode == "on":
        return torch.ones(int(batch_size), dtype=torch.bool, device=device)
    dropout_probability = float(
        cfg.get("conditioning", {}).get("word_prior_dropout_probability", 0.5)
    )
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("word_prior_dropout_probability must be in [0, 1]")
    return torch.rand(int(batch_size), device=device) >= dropout_probability


def _sentence_memory_availability(
    batch_size,
    cfg,
    device,
    mode,
    *,
    batch=None,
    epoch=0,
):
    if not sentence_memory_enabled(cfg):
        return torch.zeros(int(batch_size), dtype=torch.bool, device=device)
    mode = str(mode).lower()
    if mode == "shuffled":
        return torch.ones(int(batch_size), dtype=torch.bool, device=device)
    if mode not in SENTENCE_MEMORY_TRAIN_MODES:
        allowed = sorted(SENTENCE_MEMORY_TRAIN_MODES | {"shuffled"})
        raise ValueError(
            f"sentence_memory_mode must be one of {allowed}, got {mode!r}"
        )
    if mode == "off":
        return torch.zeros(int(batch_size), dtype=torch.bool, device=device)
    if mode == "on":
        return torch.ones(int(batch_size), dtype=torch.bool, device=device)
    dropout_probability = float(
        cfg.get("conditioning", {}).get(
            "sentence_memory_dropout_probability", 0.1
        )
    )
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError(
            "sentence_memory_dropout_probability must be in [0, 1]"
        )
    batch = batch or {}
    names = list(batch.get("name", ()))
    paths = list(batch.get("motion_path", ()))
    indices = batch.get("index")
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().view(-1).tolist()
    elif indices is None:
        indices = ()
    else:
        indices = list(indices)
    seed = int(cfg.get("seed", 1234))
    retained = []
    for index in range(int(batch_size)):
        identity = "|".join(
            (
                str(names[index]) if index < len(names) else "",
                str(paths[index]) if index < len(paths) else "",
                str(indices[index]) if index < len(indices) else str(index),
            )
        )
        token = f"{seed}|{int(epoch)}|sentence-dropout|{identity}"
        draw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        retained.append(draw / float(16**16 - 1) >= dropout_probability)
    return torch.tensor(retained, dtype=torch.bool, device=device)


def sentence_memory_query_key(text_tokens, text_mask):
    mask = text_mask.to(dtype=text_tokens.dtype).unsqueeze(-1)
    pooled = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled.float(), dim=-1)


def _sentence_memory_field(memory_batch, name, *aliases, default=None):
    for candidate in (name, *aliases):
        if hasattr(memory_batch, candidate):
            return getattr(memory_batch, candidate)
        if isinstance(memory_batch, dict) and candidate in memory_batch:
            return memory_batch[candidate]
    return default


def sentence_memory_forward_kwargs(memory_batch):
    available = _sentence_memory_field(memory_batch, "available")
    if available is None:
        raise ValueError("Sentence-memory retrieval result is missing 'available'")
    return {
        "sentence_motion_tokens": _sentence_memory_field(memory_batch, "tokens"),
        "sentence_motion_mask": _sentence_memory_field(memory_batch, "token_mask"),
        "sentence_motion_tau": _sentence_memory_field(memory_batch, "token_tau"),
        "sentence_text_keys": _sentence_memory_field(
            memory_batch, "candidate_keys", "text_keys"
        ),
        "sentence_scores": _sentence_memory_field(memory_batch, "scores"),
        "sentence_durations": _sentence_memory_field(memory_batch, "durations"),
        "sentence_part_validity": _sentence_memory_field(
            memory_batch, "part_validity", "hand_valid"
        ),
        "sentence_candidate_mask": _sentence_memory_field(
            memory_batch, "candidate_mask"
        ),
        "sentence_memory_available": available,
    }


def retrieve_sentence_memory(
    provider,
    *,
    dataset,
    batch,
    text_tokens,
    text_mask,
    predicted_duration,
    available,
    training,
    device,
    mode,
):
    """Call either the key-based or token-based provider contract centrally."""

    parameters = inspect.signature(provider.retrieve).parameters
    candidates = {
        "dataset": dataset,
        "batch": batch,
        "query_key": sentence_memory_query_key(text_tokens, text_mask),
        "query_tokens": text_tokens,
        "query_mask": text_mask,
        "predicted_duration": predicted_duration,
        "available": available,
        "training": bool(training),
        "device": device,
        "mode": mode,
    }
    kwargs = {
        name: value for name, value in candidates.items() if name in parameters
    }
    return provider.retrieve(**kwargs)


def deterministic_sentence_shuffle_mask(batch, available, cfg, epoch):
    probability = float(
        cfg.get("sentence_memory_safety", {}).get(
            "shuffle_probability", 0.25
        )
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "sentence_memory_safety.shuffle_probability must be in [0, 1]"
        )
    names = list(batch.get("name", ()))
    paths = list(batch.get("motion_path", ()))
    indices = batch.get("index")
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().view(-1).tolist()
    elif indices is None:
        indices = ()
    else:
        indices = list(indices)
    seed = int(cfg.get("seed", 1234))
    selected = []
    for index in range(len(available)):
        identity = "|".join(
            (
                str(names[index]) if index < len(names) else "",
                str(paths[index]) if index < len(paths) else "",
                str(indices[index]) if index < len(indices) else str(index),
            )
        )
        token = f"{seed}|{int(epoch)}|sentence-shuffle|{identity}"
        draw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        selected.append(draw / float(16**16 - 1) < probability)
    return torch.tensor(selected, dtype=torch.bool, device=available.device) & available.bool()


def merge_sentence_memory_batches(normal, shuffled, shuffled_mask):
    """Replace selected batch rows while preserving tensor/provenance alignment."""

    if type(normal) is not type(shuffled):
        raise TypeError("Normal and shuffled sentence-memory batches must share a type")
    updates = {}
    field_names = (
        [field.name for field in fields(normal)]
        if is_dataclass(normal)
        else list(normal.keys())
    )
    for name in field_names:
        if name == "provenance":
            continue
        normal_value = _sentence_memory_field(normal, name)
        shuffled_value = _sentence_memory_field(shuffled, name)
        if torch.is_tensor(normal_value) and torch.is_tensor(shuffled_value):
            if name in {
                "tokens",
                "token_mask",
                "token_tau",
                "part_validity",
            }:
                common_length = max(normal_value.shape[2], shuffled_value.shape[2])

                def pad_token_length(value):
                    if value.shape[2] == common_length:
                        return value
                    shape = list(value.shape)
                    shape[2] = common_length
                    padded = value.new_zeros(shape)
                    slices = [slice(None)] * value.ndim
                    slices[2] = slice(0, value.shape[2])
                    padded[tuple(slices)] = value
                    return padded

                normal_value = pad_token_length(normal_value)
                shuffled_value = pad_token_length(shuffled_value)
            elif normal_value.shape != shuffled_value.shape:
                raise ValueError(
                    f"Sentence-memory field {name!r} has incompatible shapes "
                    f"{tuple(normal_value.shape)} and {tuple(shuffled_value.shape)}"
                )
            selector = shuffled_mask
            while selector.ndim < normal_value.ndim:
                selector = selector.unsqueeze(-1)
            updates[name] = torch.where(selector, shuffled_value, normal_value)
    normal_provenance = copy.deepcopy(
        _sentence_memory_field(normal, "provenance", default={})
    )
    normal_provenance.update(
        {
            "sentence_memory_shuffled_mask": shuffled_mask.detach().cpu().tolist(),
            "shuffled_source": copy.deepcopy(
                _sentence_memory_field(shuffled, "provenance", default={})
            ),
        }
    )
    updates["provenance"] = normal_provenance
    if is_dataclass(normal):
        return replace(normal, **updates)
    merged = dict(normal)
    merged.update(updates)
    return merged


def _duplicate_optional_batch_kwargs(kwargs, batch_size):
    duplicated = {}
    for name, value in kwargs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            duplicated[name] = torch.cat((value, value), dim=0)
        else:
            duplicated[name] = value
    return duplicated


def _slice_forward_output_batch(outputs, batch_slice, combined_batch_size):
    sliced = {}
    for name, value in outputs.items():
        if torch.is_tensor(value):
            sliced[name] = (
                value[batch_slice]
                if value.ndim > 0 and value.shape[0] == combined_batch_size
                else value
            )
        elif is_dataclass(value):
            value_batch_size = getattr(value, "batch_size", None)
            updates = {}
            for field in fields(value):
                field_value = getattr(value, field.name)
                updates[field.name] = (
                    field_value[batch_slice]
                    if torch.is_tensor(field_value)
                    and field_value.ndim > 0
                    and value_batch_size == combined_batch_size
                    and field_value.shape[0] == combined_batch_size
                    else field_value
                )
            sliced[name] = replace(value, **updates)
        else:
            sliced[name] = value
    return sliced


def phase_b_combined_model_forward(
    model,
    *,
    text_tokens,
    query_times,
    text_mask,
    query_mask,
    word_kwargs,
    sentence_kwargs,
):
    """Run primary-memory and text-only student rows in one DDP forward graph."""

    batch_size = int(text_tokens.shape[0])
    device = text_tokens.device
    combined_word = _duplicate_optional_batch_kwargs(word_kwargs, batch_size)
    primary_word_available = word_kwargs.get("word_prior_available")
    if primary_word_available is None:
        primary_word_available = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
    combined_word["word_prior_available"] = torch.cat(
        (
            primary_word_available.bool(),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        ),
        dim=0,
    )
    combined_sentence = _duplicate_optional_batch_kwargs(
        sentence_kwargs, batch_size
    )
    primary_sentence_available = sentence_kwargs.get(
        "sentence_memory_available"
    )
    if primary_sentence_available is None:
        primary_sentence_available = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
    combined_sentence["sentence_memory_available"] = torch.cat(
        (
            primary_sentence_available.bool(),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        ),
        dim=0,
    )
    combined_outputs = model(
        text_tokens=torch.cat((text_tokens, text_tokens), dim=0),
        query_times=torch.cat((query_times, query_times), dim=0),
        text_mask=torch.cat((text_mask, text_mask), dim=0),
        time_domain="normalized",
        query_mask=torch.cat((query_mask, query_mask), dim=0),
        **combined_word,
        **combined_sentence,
    )
    combined_batch_size = 2 * batch_size
    primary_outputs = _slice_forward_output_batch(
        combined_outputs, slice(0, batch_size), combined_batch_size
    )
    student_off_prediction = combined_outputs["prediction"][batch_size:]
    return primary_outputs, student_off_prediction


def prepare_field_batch(
    model,
    text_encoder,
    provider,
    batch,
    dataset,
    cfg,
    device,
    word_prior_mode=None,
    sentence_memory_provider=None,
    sentence_memory_mode=None,
    training=True,
    epoch=0,
):
    target = prepare_motion(batch, dataset, device)
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    tau = native_query_times(
        batch["length"],
        max_len=target.shape[1],
        device=device,
        dtype=target.dtype,
    )
    adapter_context = None
    retrieval = None
    word_availability = None
    sentence_availability = None
    sentence_memory_batch = None
    sentence_memory_shuffled_mask = torch.zeros(
        target.shape[0], dtype=torch.bool, device=device
    )
    word_kwargs = {}
    sentence_kwargs = {}
    phase_b_student_off_prediction = None
    if is_word_prior_model(cfg):
        resolved_word_mode = (
            configured_word_prior_train_mode(cfg)
            if word_prior_mode is None
            else str(word_prior_mode).lower()
        )
        word_availability = _word_prior_availability(
            target.shape[0], cfg, device, resolved_word_mode
        )
        if bool(word_availability.any()):
            if provider is None:
                raise ValueError("word-prior mode requires a ScaffoldProvider")
            adapter_context, _anchors, metadata = provider.build_with_metadata(
                batch,
                x=None,
                use_cache=True,
            )
            retrieval = metadata["retrieval_features"]
        word_kwargs = {
            "word_prior_context": adapter_context,
            "word_prior_mask": (
                batch["mask"] if adapter_context is not None else None
            ),
            "word_prior_features": retrieval,
            "word_prior_available": word_availability,
        }

    if is_sentence_memory_model(cfg):
        resolved_sentence_mode = (
            configured_sentence_memory_train_mode(cfg)
            if sentence_memory_mode is None
            else str(sentence_memory_mode).lower()
        )
        sentence_availability = _sentence_memory_availability(
            target.shape[0],
            cfg,
            device,
            resolved_sentence_mode,
            batch=batch,
            epoch=epoch,
        )
        sentence_kwargs = {
            "sentence_memory_available": sentence_availability
        }
        if bool(sentence_availability.any()):
            if sentence_memory_provider is None:
                raise ValueError(
                    "sentence-memory mode requires a SentenceMemoryProvider"
                )
            _log_duration, predicted_duration = unwrap_model(model).predict_duration(
                text_tokens,
                text_mask=text_mask,
            )
            sentence_memory_batch = retrieve_sentence_memory(
                sentence_memory_provider,
                dataset=dataset,
                batch=batch,
                text_tokens=text_tokens,
                text_mask=text_mask,
                predicted_duration=predicted_duration.detach(),
                available=sentence_availability,
                training=training,
                device=device,
                mode=(
                    "shuffled"
                    if resolved_sentence_mode == "shuffled"
                    else "on"
                ),
            )
            if bool(
                training
                and cfg.get("sentence_memory_safety", {}).get("enabled", False)
            ):
                effective_available = _sentence_memory_field(
                    sentence_memory_batch, "available"
                ).bool()
                sentence_memory_shuffled_mask = deterministic_sentence_shuffle_mask(
                    batch, effective_available, cfg, epoch
                )
                if bool(sentence_memory_shuffled_mask.any()):
                    shuffled_batch = retrieve_sentence_memory(
                        sentence_memory_provider,
                        dataset=dataset,
                        batch=batch,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                        predicted_duration=predicted_duration.detach(),
                        # Only the rows selected for the corruption control
                        # need a second lookup/payload gather.  Keeping other
                        # rows unavailable avoids doubling sentence-bank I/O
                        # for every training microbatch.
                        available=sentence_memory_shuffled_mask,
                        training=True,
                        device=device,
                        mode="shuffled",
                    )
                    sentence_memory_batch = merge_sentence_memory_batches(
                        sentence_memory_batch,
                        shuffled_batch,
                        sentence_memory_shuffled_mask,
                    )
            sentence_kwargs = sentence_memory_forward_kwargs(
                sentence_memory_batch
            )
            sentence_availability = sentence_kwargs[
                "sentence_memory_available"
            ]
            sentence_memory_shuffled_mask = (
                sentence_memory_shuffled_mask & sentence_availability.bool()
            )

    if is_dual_mode(cfg):
        phase_b_combined_forward = bool(
            training
            and is_sentence_memory_model(cfg)
            and cfg.get("sentence_memory_safety", {})
            .get("phase_b", {})
            .get("enabled", False)
        )
        if phase_b_combined_forward:
            outputs, phase_b_student_off_prediction = (
                phase_b_combined_model_forward(
                    model,
                    text_tokens=text_tokens,
                    query_times=tau,
                    text_mask=text_mask,
                    query_mask=batch["mask"],
                    word_kwargs=word_kwargs,
                    sentence_kwargs=sentence_kwargs,
                )
            )
        else:
            outputs = model(
                text_tokens=text_tokens,
                query_times=tau,
                text_mask=text_mask,
                time_domain="normalized",
                query_mask=batch["mask"],
                **word_kwargs,
                **sentence_kwargs,
            )
    else:
        if provider is None:
            raise ValueError("v1 continuous trajectory training requires a ScaffoldProvider")
        adapter_context, _anchors, metadata = provider.build_with_metadata(
            batch,
            x=None,
            use_cache=True,
        )
        retrieval = metadata["retrieval_features"]
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
        "word_prior_available": word_availability,
        "sentence_memory_available": sentence_availability,
        "sentence_memory_batch": sentence_memory_batch,
        "sentence_memory_shuffled_mask": sentence_memory_shuffled_mask,
        "phase_b_student_off_prediction": phase_b_student_off_prediction,
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
    word_prior_mode=None,
    sentence_memory_provider=None,
    sentence_memory_mode=None,
    sentence_off_teacher=None,
):
    prepared = prepare_field_batch(
        model,
        text_encoder,
        provider,
        batch,
        dataset,
        cfg,
        device,
        word_prior_mode=word_prior_mode,
        sentence_memory_provider=sentence_memory_provider,
        sentence_memory_mode=sentence_memory_mode,
        training=training,
        epoch=epoch,
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
    if is_dual_mode(cfg):
        auxiliary = coarse_and_residual_losses(
            outputs,
            target,
            mask,
            hand_weight=hand_weight,
        )
    else:
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

    sentence_safe = target.new_tensor(0.0)
    sentence_shuffle = target.new_tensor(0.0)
    sentence_gate_sparsity = target.new_tensor(0.0)
    sentence_delta_sparsity = target.new_tensor(0.0)
    sentence_off_distill = target.new_tensor(0.0)
    sentence_off_compact = target.new_tensor(0.0)
    sentence_off_sample_count = target.new_tensor(0.0)
    sentence_safety_cfg = cfg.get("sentence_memory_safety", {})
    phase_b_cfg = sentence_safety_cfg.get("phase_b", {})
    phase_b_enabled = bool(training and phase_b_cfg.get("enabled", False))
    phase_b_teacher_prediction = None
    if phase_b_enabled:
        (
            phase_b_teacher_prediction,
            _phase_b_student_off_prediction,
            sentence_off_distill,
        ) = phase_b_off_distillation(
            sentence_off_teacher,
            prepared,
            batch,
            phase_b_cfg,
            device,
        )
    sentence_safety_enabled = bool(
        training
        and is_sentence_memory_model(cfg)
        and sentence_safety_cfg.get("enabled", False)
        and prepared["sentence_memory_available"].bool().any()
    )
    if sentence_safety_enabled:
        off_prediction = sentence_memory_off_baseline(
            model,
            prepared,
            batch,
            cfg,
            device,
            frozen_teacher_prediction=phase_b_teacher_prediction,
        )
        available = prepared["sentence_memory_available"].bool()
        memory_error = _per_sample_part_compact_l1(
            outputs["prediction"], target, mask
        )
        off_error = _per_sample_part_compact_l1(off_prediction, target, mask)
        sentence_off_compact = off_error.mean()
        sentence_off_sample_count = off_error.new_tensor(float(len(off_error)))
        safe_margin = float(sentence_safety_cfg.get("nonregression_margin", 0.0))
        shuffled_mask = prepared["sentence_memory_shuffled_mask"].bool()
        normal_available = available & ~shuffled_mask
        if bool(normal_available.any()):
            sentence_safe = F.relu(
                memory_error[normal_available]
                - off_error[normal_available]
                + safe_margin
            ).mean()

        memory_gates = getattr(trajectory, "sentence_memory_gates", None)
        if memory_gates is not None:
            sentence_gate_sparsity = memory_gates[available].abs().mean()
        delta = outputs["prediction"] - off_prediction
        available_mask = mask & available[:, None]
        if bool(available_mask.any()):
            sentence_delta_sparsity = delta[available_mask].abs().mean()
        if bool(shuffled_mask.any()):
            shuffled_frames = mask & shuffled_mask[:, None]
            sentence_shuffle = F.smooth_l1_loss(
                outputs["prediction"][shuffled_frames],
                off_prediction[shuffled_frames],
                beta=float(sentence_safety_cfg.get("shuffle_huber_beta", 0.1)),
            )

    total = float(objective_cfg.get("lambda_endpoint", 1.0)) * endpoint_total
    if is_dual_mode(cfg):
        total = total + float(objective_cfg.get("lambda_coarse", 0.25)) * auxiliary[
            "loss_coarse"
        ]
    else:
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
    total = total + float(objective_cfg.get("lambda_sentence_safe", 1.0)) * sentence_safe
    total = total + float(objective_cfg.get("lambda_sentence_shuffle", 1.0)) * sentence_shuffle
    sentence_sparsity_weight = float(
        objective_cfg.get("lambda_sentence_sparsity", 1e-4)
    )
    total = total + sentence_sparsity_weight * (
        sentence_gate_sparsity + sentence_delta_sparsity
    )
    total = total + float(
        objective_cfg.get("lambda_sentence_off_distill", 1.0)
    ) * sentence_off_distill

    losses = {}
    losses.update(endpoint)
    losses.update(auxiliary)
    losses.update(local)
    losses.update(finite_losses)
    losses.update(analytic)
    losses["loss_duration"] = duration
    if is_sentence_memory_model(cfg):
        losses["loss_sentence_safe"] = sentence_safe
        losses["loss_sentence_shuffle"] = sentence_shuffle
        losses["loss_sentence_gate_sparsity"] = sentence_gate_sparsity
        losses["loss_sentence_delta_sparsity"] = sentence_delta_sparsity
        losses["loss_sentence_off_distill"] = sentence_off_distill
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
        if not is_dual_mode(cfg):
            losses["center_density_uncertainty_correlation"] = _masked_pearson(
                trajectory.context_density,
                1.0 - prepared["retrieval"][..., 0].clamp(0.0, 1.0),
                mask,
            )
        else:
            losses["center_density_uncertainty_correlation"] = local_axis.new_tensor(
                0.0
            )
        if trajectory.local_part_gates is not None:
            gate_mask = active.unsqueeze(-1)
            gate_count = gate_mask.sum().clamp_min(1.0)
            for part_index, part_name in enumerate(WORD_PRIOR_PART_NAMES):
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
    if is_word_prior_model(cfg):
        availability = prepared["word_prior_available"].bool()
        losses["word_prior_available_fraction"] = availability.float().mean()
        if not is_sentence_memory_model(cfg):
            for label, sample_selector in (
                ("text_only", ~availability),
                ("word_prior", availability),
            ):
                mode_mask = mask & sample_selector[:, None]
                losses[f"mode_{label}_sample_count"] = sample_selector.float().sum()
                losses[f"mode_{label}_compact_l1"] = masked_feature_l1(
                    outputs["prediction"],
                    target,
                    mode_mask,
                    hand_weight=hand_weight,
                )
        word_gates = trajectory.word_prior_gates
        if word_gates is None:
            raise RuntimeError("Dual-mode trajectory is missing word_prior_gates")
        available_slots = availability[:, None].to(word_gates.dtype)
        available_count = (
            available_slots.sum() * word_gates.shape[1]
        ).clamp_min(1.0)
        for part_index, part_name in enumerate(WORD_PRIOR_PART_NAMES):
            losses[f"word_prior_gate_mean_{part_name}"] = (
                word_gates[..., part_index] * available_slots
            ).sum() / available_count
    if is_sentence_memory_model(cfg):
        memory_availability = prepared["sentence_memory_available"].bool()
        losses["sentence_memory_available_fraction"] = (
            memory_availability.float().mean()
        )
        losses["sentence_memory_shuffled_fraction"] = prepared[
            "sentence_memory_shuffled_mask"
        ].float().mean()
        for label, sample_selector in (("sentence_memory", memory_availability),):
            mode_mask = mask & sample_selector[:, None]
            losses[f"mode_{label}_sample_count"] = sample_selector.float().sum()
            losses[f"mode_{label}_compact_l1"] = masked_feature_l1(
                outputs["prediction"],
                target,
                mode_mask,
                hand_weight=hand_weight,
            )
        if sentence_safety_enabled:
            losses["mode_text_only_sample_count"] = sentence_off_sample_count
            losses["mode_text_only_compact_l1"] = sentence_off_compact
        else:
            text_only = ~memory_availability
            losses["mode_text_only_sample_count"] = text_only.float().sum()
            losses["mode_text_only_compact_l1"] = masked_feature_l1(
                outputs["prediction"],
                target,
                mask & text_only[:, None],
                hand_weight=hand_weight,
            )
        memory_gates = getattr(trajectory, "sentence_memory_gates", None)
        if memory_gates is not None:
            available_slots = memory_availability[:, None].to(memory_gates.dtype)
            available_count = (
                available_slots.sum() * memory_gates.shape[1]
            ).clamp_min(1.0)
            for part_index, part_name in enumerate(WORD_PRIOR_PART_NAMES):
                losses[f"sentence_memory_gate_mean_{part_name}"] = (
                    memory_gates[..., part_index] * available_slots
                ).sum() / available_count
        for diagnostic_name in (
            "sentence_memory_null_mass",
            "sentence_memory_candidate_mass",
        ):
            diagnostic = getattr(trajectory, diagnostic_name, None)
            if diagnostic is not None:
                if diagnostic_name == "sentence_memory_candidate_mass":
                    diagnostic = diagnostic.sum(dim=-1)
                selector = memory_availability
                while selector.ndim < diagnostic.ndim:
                    selector = selector.unsqueeze(-1)
                selector = selector.expand_as(diagnostic)
                selected = diagnostic[selector]
                losses[diagnostic_name] = (
                    selected.mean()
                    if selected.numel()
                    else diagnostic.new_tensor(0.0)
                )
        memory_batch = prepared["sentence_memory_batch"]
        if memory_batch is not None:
            candidate_mask = _sentence_memory_field(
                memory_batch, "candidate_mask"
            )
            scores = _sentence_memory_field(memory_batch, "scores")
            duration_gap = _sentence_memory_field(
                memory_batch,
                "duration_log_gap",
                "duration_ratio",
            )
            if candidate_mask is not None and scores is not None:
                selected = scores[candidate_mask.bool()]
                losses["sentence_memory_retrieval_score"] = (
                    selected.mean()
                    if selected.numel()
                    else scores.new_tensor(0.0)
                )
            if candidate_mask is not None and duration_gap is not None:
                selected = duration_gap[candidate_mask.bool()]
                losses["sentence_memory_duration_gap"] = (
                    selected.abs().mean()
                    if selected.numel()
                    else duration_gap.new_tensor(0.0)
                )
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

_SENTENCE_MEMORY_PARAMETER_PREFIXES = (
    "hypernetwork.sentence_memory_",
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


def is_sentence_memory_parameter(name):
    return any(
        str(name).startswith(prefix)
        for prefix in _SENTENCE_MEMORY_PARAMETER_PREFIXES
    )


def validate_sentence_memory_base_checkpoint(checkpoint, cfg, source="checkpoint"):
    actual_type = checkpoint.get("model_type")
    actual_version = checkpoint.get("trajectory_contract_version")
    if actual_type != DUAL_MODE_MODEL_TYPE or int(actual_version or -1) != 2:
        raise RuntimeError(
            f"{source} must be a v2 {DUAL_MODE_MODEL_TYPE!r} checkpoint; "
            f"got model_type={actual_type!r}, "
            f"trajectory_contract_version={actual_version!r}"
        )
    expected_text_encoder = configured_text_encoder_identity(cfg)
    actual_text_encoder = checkpoint.get("text_encoder_identity")
    if actual_text_encoder is None and isinstance(checkpoint.get("config"), dict):
        actual_text_encoder = configured_text_encoder_identity(checkpoint["config"])
    if (
        actual_text_encoder is not None
        and str(actual_text_encoder).replace("\\", "/") != expected_text_encoder
    ):
        raise RuntimeError(
            f"{source} uses text encoder {actual_text_encoder!r}; expected "
            f"{expected_text_encoder!r}"
        )


def load_sentence_memory_base_state(model, state_dict):
    """Strictly migrate v2 weights, permitting only the new v3 memory modules."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not is_sentence_memory_parameter(name)
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Sentence-memory base checkpoint is incompatible: "
            f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
        )
    return incompatible


@torch.no_grad()
def validate_v2_to_v3_text_only_parity(
    model,
    checkpoint,
    *,
    text_dim,
    device,
    tolerance=1e-7,
):
    """Prove that migrated v3 memory-off inference equals its v2 source."""

    source_cfg = copy.deepcopy(checkpoint.get("config") or {})
    if configured_model_type(source_cfg) != DUAL_MODE_MODEL_TYPE:
        raise RuntimeError("Text-only parity requires a v2 source checkpoint")
    source_model = build_continuous_trajectory_field(
        source_cfg, text_dim=int(text_dim)
    ).to(device)
    source_model.load_state_dict(checkpoint["model"], strict=True)
    source_model.eval()
    target_model = unwrap_model(model)
    target_was_training = target_model.training
    target_model.eval()

    length = 5
    base = torch.arange(
        2 * length * int(text_dim), device=device, dtype=torch.float32
    ).reshape(2, length, int(text_dim))
    text_tokens = torch.sin(base * 0.013)
    text_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]],
        dtype=torch.bool,
        device=device,
    )
    query_times = torch.linspace(-1.0, 1.0, 7, device=device).expand(2, -1)
    unavailable = torch.zeros(2, dtype=torch.bool, device=device)
    common = {
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "query_times": query_times,
        "time_domain": "normalized",
        "word_prior_available": unavailable,
    }
    source_output = source_model(**common)
    target_output = target_model(
        **common,
        sentence_memory_available=unavailable,
    )
    maximum_error = float(
        (source_output["prediction"] - target_output["prediction"])
        .abs()
        .max()
        .item()
    )
    duration_error = float(
        (
            source_output["trajectory"].duration_seconds
            - target_output["trajectory"].duration_seconds
        )
        .abs()
        .max()
        .item()
    )
    if target_was_training:
        target_model.train()
    del source_model
    if maximum_error > float(tolerance) or duration_error > float(tolerance):
        raise RuntimeError(
            "v2-to-v3 text-only parity failed: "
            f"prediction_max_abs={maximum_error:.9g}, "
            f"duration_max_abs={duration_error:.9g}, "
            f"tolerance={float(tolerance):.9g}"
        )
    return {
        "prediction_max_abs": maximum_error,
        "duration_max_abs": duration_error,
        "tolerance": float(tolerance),
        "passed": True,
    }


def configure_sentence_memory_trainable_parameters(model, cfg):
    train_cfg = cfg.get("train", {})
    freeze_base = bool(train_cfg.get("freeze_base", False))
    unfreeze_prefixes = tuple(
        str(prefix) for prefix in train_cfg.get("unfreeze_base_prefixes", ())
    )
    if not is_sentence_memory_model(cfg):
        return {"freeze_base": False, "trainable": None}
    trainable = []
    for name, parameter in unwrap_model(model).named_parameters():
        enabled = (
            not freeze_base
            or is_sentence_memory_parameter(name)
            or any(name.startswith(prefix) for prefix in unfreeze_prefixes)
        )
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if freeze_base and not trainable:
        raise RuntimeError(
            "train.freeze_base=true but no v3 sentence-memory parameters matched "
            f"prefixes {_SENTENCE_MEMORY_PARAMETER_PREFIXES}"
        )
    return {
        "freeze_base": freeze_base,
        "unfreeze_base_prefixes": unfreeze_prefixes,
        "trainable": trainable,
    }


def build_sentence_off_teacher(cfg, text_dim, device):
    """Build the frozen v2 teacher used by optional Phase-B off distillation."""

    phase_b = cfg.get("sentence_memory_safety", {}).get("phase_b", {})
    if not bool(phase_b.get("enabled", False)):
        return None
    checkpoint_path = phase_b.get("teacher_checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "sentence_memory_safety.phase_b.teacher_checkpoint is required "
            "when Phase-B off distillation is enabled"
        )
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    validate_sentence_memory_base_checkpoint(
        checkpoint, cfg, source=str(checkpoint_path)
    )
    teacher_cfg = copy.deepcopy(checkpoint.get("config", {}))
    if configured_model_type(teacher_cfg) != DUAL_MODE_MODEL_TYPE:
        raise RuntimeError("Phase-B teacher config is not a v2 dual-mode model")
    teacher = build_continuous_trajectory_field(
        teacher_cfg, text_dim=text_dim
    ).to(device)
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # Bind the Phase-B validation guard to the actual source-v2 validation
    # result.  This avoids re-running a fourth validation forward every epoch
    # while still making the 0.5% no-regression threshold checkpoint-specific.
    validation_prefix = "val_text_only/"
    teacher_validation = {
        name.removeprefix(validation_prefix): value
        for name, value in (checkpoint.get("metrics") or {}).items()
        if name.startswith(validation_prefix)
    }
    if not teacher_validation:
        raise RuntimeError(
            "Phase-B teacher checkpoint has no val_text_only/* metrics; "
            "it cannot establish the required text-only validation guard"
        )
    teacher_score = float(selection_diagnostics(teacher_validation, cfg)[0])
    if not math.isfinite(teacher_score):
        raise RuntimeError(
            "Phase-B teacher has a non-finite text-only validation score"
        )
    phase_b["teacher_validation_selection_score"] = teacher_score
    return teacher


def build_optimizer(model, cfg):
    train_cfg = cfg.get("train", {})
    base_lr = float(train_cfg.get("lr", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    use_groups = is_sentence_memory_model(cfg) or any(
        key in train_cfg
        for key in (
            "local_warmup_epochs",
            "warmup_local_lr",
            "joint_local_lr",
            "joint_global_lr",
        )
    )
    if not use_groups:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable parameters remain for the optimizer")
        return torch.optim.AdamW(parameters, lr=base_lr, weight_decay=weight_decay)

    local_parameters = []
    global_parameters = []
    sentence_memory_parameters = []
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if is_sentence_memory_parameter(name):
            sentence_memory_parameters.append(parameter)
            continue
        if any(name.startswith(prefix) for prefix in _LOCAL_PARAMETER_PREFIXES):
            local_parameters.append(parameter)
        else:
            global_parameters.append(parameter)
    groups = []
    if global_parameters:
        groups.append(
            {"params": global_parameters, "lr": base_lr, "group_name": "global"}
        )
    if local_parameters:
        groups.append(
            {"params": local_parameters, "lr": base_lr, "group_name": "local"}
        )
    if sentence_memory_parameters:
        groups.append(
            {
                "params": sentence_memory_parameters,
                "lr": float(train_cfg.get("sentence_memory_lr", base_lr)),
                "group_name": "sentence_memory",
            }
        )
    if not groups:
        raise RuntimeError("No trainable parameters remain for the optimizer")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


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
        elif name == "sentence_memory":
            group["lr"] = float(train_cfg.get("sentence_memory_lr", base_lr))
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


def synchronized_memory_microbatch_size(batch, cfg, dist_info, device):
    """Choose one memory-safe microbatch size shared by every DDP rank.

    A rank's padded sequence length can differ from the other ranks, so its
    locally safe frame-budget size can differ as well. DDP requires every rank
    to execute the same sequence of backward collectives. Gathering the local
    sizes and using their minimum preserves each rank's memory bound while
    guaranteeing an identical number of microbatches for equal logical batch
    sizes.
    """

    logical_size = len(batch["name"])
    local_microbatch_size = memory_microbatch_size(batch, cfg)
    if not dist_info.get("enabled", False):
        return local_microbatch_size
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed microbatch synchronization requires an initialized "
            "torch.distributed process group"
        )

    backend = str(dist_info.get("backend") or dist.get_backend()).lower()
    if backend == "nccl":
        if device.type != "cuda":
            raise RuntimeError(
                "NCCL microbatch synchronization requires a CUDA device"
            )
        collective_device = device
    else:
        collective_device = torch.device("cpu")

    local_schedule = torch.tensor(
        [logical_size, local_microbatch_size],
        dtype=torch.int64,
        device=collective_device,
    )
    gathered_schedules = [
        torch.empty_like(local_schedule) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_schedules, local_schedule)
    rank_schedules = [
        tuple(int(value) for value in schedule.detach().cpu().tolist())
        for schedule in gathered_schedules
    ]

    logical_sizes = {schedule[0] for schedule in rank_schedules}
    if len(logical_sizes) != 1:
        raise RuntimeError(
            "DDP ranks received different logical batch sizes; an identical "
            f"backward schedule cannot be formed: {rank_schedules}"
        )
    shared_microbatch_size = min(schedule[1] for schedule in rank_schedules)
    if shared_microbatch_size <= 0:
        raise RuntimeError(
            f"Invalid distributed memory-microbatch schedule: {rank_schedules}"
        )
    return shared_microbatch_size


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
    "word_prior_available_fraction",
    "sentence_memory_available_fraction",
    "sentence_memory_shuffled_fraction",
    "sentence_memory_null_mass",
    "sentence_memory_candidate_mass",
    "sentence_memory_retrieval_score",
    "sentence_memory_duration_gap",
    "loss_sentence_safe",
    "loss_sentence_shuffle",
    "loss_sentence_gate_sparsity",
    "loss_sentence_delta_sparsity",
    "loss_sentence_off_distill",
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
    for name, value in metrics.items():
        if name.startswith("mode_text_only_"):
            key = name.removeprefix("mode_text_only_")
            payload[f"train/text_only/batch/{key}"] = float(value)
        elif name.startswith("mode_word_prior_"):
            key = name.removeprefix("mode_word_prior_")
            payload[f"train/word_prior/batch/{key}"] = float(value)
        elif name.startswith("word_prior_gate_"):
            payload[f"train/word_prior/gates/{name.removeprefix('word_prior_gate_')}"] = float(value)
        elif name.startswith("mode_sentence_memory_"):
            key = name.removeprefix("mode_sentence_memory_")
            payload[f"train/sentence_memory/batch/{key}"] = float(value)
        elif name.startswith("sentence_memory_gate_"):
            key = name.removeprefix("sentence_memory_gate_")
            payload[f"train/sentence_memory/gates/{key}"] = float(value)
        elif name.startswith("sentence_memory_retrieval_"):
            key = name.removeprefix("sentence_memory_retrieval_")
            payload[f"train/retrieval/batch/{key}"] = float(value)
        elif name == "sentence_memory_duration_gap":
            payload["train/retrieval/batch/duration_gap"] = float(value)
    return payload


def wandb_train_epoch_payload(row):
    payload = {
        "train/epoch_step": int(row["epoch"]),
        "train/epoch/global_step": int(row["global_step"]),
        "train/epoch/elapsed_sec": float(row["elapsed_sec"]),
    }
    for name, value in row.items():
        if name.startswith("train_"):
            key = name.removeprefix("train_")
            if key.startswith("mode_text_only_"):
                payload[f"train/text_only/epoch/{key.removeprefix('mode_text_only_')}"] = value
            elif key.startswith("mode_word_prior_"):
                payload[f"train/word_prior/epoch/{key.removeprefix('mode_word_prior_')}"] = value
            elif key.startswith("word_prior_gate_"):
                payload[f"train/word_prior/epoch/gates/{key.removeprefix('word_prior_gate_')}"] = value
            elif key.startswith("mode_sentence_memory_"):
                payload[f"train/sentence_memory/epoch/{key.removeprefix('mode_sentence_memory_')}"] = value
            elif key.startswith("sentence_memory_gate_"):
                payload[f"train/sentence_memory/epoch/gates/{key.removeprefix('sentence_memory_gate_')}"] = value
            elif key.startswith("sentence_memory_retrieval_"):
                payload[f"train/retrieval/epoch/{key.removeprefix('sentence_memory_retrieval_')}"] = value
            elif key == "sentence_memory_duration_gap":
                payload["train/retrieval/epoch/duration_gap"] = value
            else:
                payload[f"train/epoch/{key}"] = value
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
        ("train/text_only/batch/*", "train/optimizer_step"),
        ("train/word_prior/batch/*", "train/optimizer_step"),
        ("train/word_prior/gates/*", "train/optimizer_step"),
        ("train/sentence_memory/batch/*", "train/optimizer_step"),
        ("train/sentence_memory/gates/*", "train/optimizer_step"),
        ("train/retrieval/batch/*", "train/optimizer_step"),
        ("train/epoch_step", None),
        ("train/epoch/*", "train/epoch_step"),
        ("train/text_only/epoch/*", "train/epoch_step"),
        ("train/word_prior/epoch/*", "train/epoch_step"),
        ("train/sentence_memory/epoch/*", "train/epoch_step"),
        ("train/retrieval/epoch/*", "train/epoch_step"),
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
    sentence_memory_provider=None,
    sentence_off_teacher=None,
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
        microbatch_size = synchronized_memory_microbatch_size(
            batch,
            cfg,
            dist_info,
            device,
        )
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
                sentence_memory_provider=sentence_memory_provider,
                sentence_off_teacher=sentence_off_teacher,
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
    word_prior_mode=None,
    sentence_memory_provider=None,
    sentence_memory_mode=None,
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
            word_prior_mode=word_prior_mode,
            sentence_memory_provider=sentence_memory_provider,
            sentence_memory_mode=sentence_memory_mode,
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
    if is_dual_mode(cfg):
        baselines = [("coarse", prepared["outputs"]["coarse"])]
        if prepared["adapter_context"] is not None:
            baselines.append(("scaffold", prepared["adapter_context"]))
    else:
        baselines = [
            ("prior", prepared["outputs"]["prior"]),
            ("scaffold", prepared["adapter_context"]),
        ]
    for label, values in baselines:
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
    word_prior_mode=None,
    sentence_memory_provider=None,
    sentence_memory_mode=None,
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
                word_prior_mode=word_prior_mode,
                sentence_memory_provider=sentence_memory_provider,
                sentence_memory_mode=sentence_memory_mode,
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


def evaluate_configured_modes(
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
    sentence_memory_provider=None,
):
    """Evaluate v2 deterministically both without and with its optional prior."""

    if not is_dual_mode(cfg):
        return evaluate(
            model,
            fk,
            text_encoder,
            provider,
            loader,
            dataset,
            cfg,
            device,
            epoch=epoch,
            max_batches=max_batches,
            show_progress=show_progress,
        )
    if is_sentence_memory_model(cfg):
        combined = {}
        namespaces = {
            "off": "text_only",
            "on": "sentence_memory",
            "shuffled": "shuffled_sentence_memory",
        }
        # v3 keeps the v2 word-prior branch. Phase A fixes it off, while this
        # explicit setting leaves future word+sentence experiments possible.
        fixed_word_mode = str(
            cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off")
        ).lower()
        if fixed_word_mode not in {"off", "on"}:
            raise ValueError(
                "eval.sentence_memory_word_prior_mode must be 'off' or 'on'"
            )
        for mode in configured_sentence_memory_eval_modes(cfg):
            metrics = evaluate(
                model,
                fk,
                text_encoder,
                provider,
                loader,
                dataset,
                cfg,
                device,
                epoch=epoch,
                max_batches=max_batches,
                show_progress=show_progress,
                word_prior_mode=fixed_word_mode,
                sentence_memory_provider=sentence_memory_provider,
                sentence_memory_mode=mode,
            )
            namespace = namespaces[mode]
            combined.update(
                {f"{namespace}/{name}": value for name, value in metrics.items()}
            )
        return combined
    combined = {}
    namespaces = {"off": "text_only", "on": "word_prior"}
    for mode in configured_word_prior_eval_modes(cfg):
        namespace = namespaces[mode]
        metrics = evaluate(
            model,
            fk,
            text_encoder,
            provider,
            loader,
            dataset,
            cfg,
            device,
            epoch=epoch,
            max_batches=max_batches,
            show_progress=show_progress,
            word_prior_mode=mode,
        )
        combined.update({f"{namespace}/{name}": value for name, value in metrics.items()})
    return combined


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
    for name, specification in (selection_cfg.get("constraints") or {}).items():
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


def _metrics_in_namespace(metrics, namespace):
    prefix = f"{namespace}/"
    return {
        name.removeprefix(prefix): value
        for name, value in metrics.items()
        if name.startswith(prefix)
    }


def checkpoint_selection_diagnostics(metrics, cfg, return_details=False):
    """Apply contract-specific checkpoint selection and ablation gates."""

    if not is_dual_mode(cfg):
        return selection_diagnostics(metrics, cfg, return_details=return_details)

    if is_sentence_memory_model(cfg):
        text_metrics = _metrics_in_namespace(metrics, "text_only")
        if not sentence_memory_enabled(cfg):
            if not text_metrics:
                raise ValueError(
                    "Disabled sentence memory did not produce text_only metrics"
                )
            score, violation, feasible, details = selection_diagnostics(
                text_metrics, cfg, return_details=True
            )
            details["dual_mode"] = {
                "selection_source": "text_only",
                "sentence_memory_enabled": False,
            }
            result = (score, violation, feasible)
            return (*result, details) if return_details else result
        memory_metrics = _metrics_in_namespace(metrics, "sentence_memory")
        shuffled_metrics = _metrics_in_namespace(
            metrics, "shuffled_sentence_memory"
        )
        selected_namespace = "sentence_memory" if memory_metrics else "text_only"
        selected_metrics = memory_metrics or text_metrics
        if not selected_metrics:
            raise ValueError(
                "Sentence-memory validation did not produce text_only or "
                "sentence_memory metrics"
            )
        score, violation, feasible, details = selection_diagnostics(
            selected_metrics, cfg, return_details=True
        )
        memory_details = {"selection_source": selected_namespace}
        require_improvement = bool(
            cfg.get("selection", {}).get(
                "require_sentence_memory_improvement", True
            )
        )
        if text_metrics and memory_metrics:
            text_score = float(selection_diagnostics(text_metrics, cfg)[0])
            memory_score = float(score)
            minimum_improvement = float(
                cfg.get("selection", {}).get(
                    "sentence_memory_min_relative_improvement", 0.001
                )
            )
            if minimum_improvement < 0:
                raise ValueError(
                    "sentence_memory_min_relative_improvement must be non-negative"
                )
            scale = max(abs(text_score), 1e-8)
            allowed_memory_score = text_score - minimum_improvement * scale
            comparison_violation = (
                max(memory_score - allowed_memory_score, 0.0) / scale
                if math.isfinite(memory_score) and math.isfinite(text_score)
                else float("inf")
            )
            if require_improvement:
                violation = float(violation) + comparison_violation
                feasible = bool(feasible and comparison_violation <= 1e-12)
                if comparison_violation > 0:
                    details["rejection_reasons"].append(
                        "sentence-memory score "
                        f"{memory_score:.6g} did not improve over text-only "
                        f"score {text_score:.6g} by the required "
                        f"{100.0 * minimum_improvement:.2f}%"
                    )
            memory_details.update(
                {
                    "text_only_score": text_score,
                    "sentence_memory_score": memory_score,
                    "sentence_memory_allowed_score": allowed_memory_score,
                    "sentence_memory_relative_improvement": (
                        (text_score - memory_score) / scale
                    ),
                    "sentence_memory_min_relative_improvement": minimum_improvement,
                    "sentence_memory_improvement_violation": comparison_violation,
                    "require_sentence_memory_improvement": float(
                        require_improvement
                    ),
                }
            )
            require_hand_path = bool(
                cfg.get("selection", {}).get(
                    "require_hand_path_nonregression", False
                )
            )
            if require_hand_path:
                hand_tolerance = float(
                    cfg.get("selection", {}).get(
                        "hand_path_max_relative_degradation", 0.02
                    )
                )
                if hand_tolerance < 0:
                    raise ValueError(
                        "hand_path_max_relative_degradation must be non-negative"
                    )
                hand_rows = {}
                for hand_name in ("lhand", "rhand"):
                    metric_name = f"pred_loss_path_{hand_name}"
                    baseline_value = text_metrics.get(metric_name)
                    memory_value = memory_metrics.get(metric_name)
                    if baseline_value is None or memory_value is None:
                        hand_violation = float("inf")
                        details["rejection_reasons"].append(
                            f"required hand-path metric {metric_name!r} is missing"
                        )
                        hand_rows[hand_name] = {
                            "text_only": baseline_value,
                            "sentence_memory": memory_value,
                            "allowed": None,
                            "violation": hand_violation,
                        }
                    else:
                        baseline_value = float(baseline_value)
                        memory_value = float(memory_value)
                        hand_scale = max(abs(baseline_value), 1e-8)
                        allowed_value = (
                            baseline_value + hand_tolerance * hand_scale
                        )
                        hand_violation = max(
                            memory_value - allowed_value, 0.0
                        ) / hand_scale
                        if hand_violation > 0:
                            details["rejection_reasons"].append(
                                f"sentence-memory {hand_name} path error "
                                f"{memory_value:.6g} exceeds the text-only "
                                f"{baseline_value:.6g} by more than "
                                f"{100.0 * hand_tolerance:.2f}%"
                            )
                        hand_rows[hand_name] = {
                            "text_only": baseline_value,
                            "sentence_memory": memory_value,
                            "allowed": allowed_value,
                            "violation": hand_violation,
                        }
                    violation = float(violation) + hand_violation
                    feasible = bool(feasible and hand_violation <= 1e-12)
                memory_details["hand_path_nonregression"] = hand_rows

        phase_b_cfg = (
            cfg.get("sentence_memory_safety", {}).get("phase_b", {})
        )
        if bool(phase_b_cfg.get("enabled", False)):
            teacher_score = phase_b_cfg.get("teacher_validation_selection_score")
            if teacher_score is None:
                raise ValueError(
                    "Phase-B selection requires the validation score bound to "
                    "its frozen v2 teacher"
                )
            tolerance = float(
                phase_b_cfg.get("text_only_max_relative_degradation", 0.005)
            )
            if tolerance < 0:
                raise ValueError(
                    "phase_b.text_only_max_relative_degradation must be non-negative"
                )
            current_text_score = (
                float(selection_diagnostics(text_metrics, cfg)[0])
                if text_metrics
                else float("inf")
            )
            teacher_score = float(teacher_score)
            comparison_scale = max(abs(teacher_score), 1e-8)
            allowed_text_score = teacher_score + tolerance * comparison_scale
            teacher_violation = (
                max(current_text_score - allowed_text_score, 0.0)
                / comparison_scale
                if math.isfinite(current_text_score)
                and math.isfinite(teacher_score)
                else float("inf")
            )
            violation = float(violation) + teacher_violation
            feasible = bool(feasible and teacher_violation <= 1e-12)
            if teacher_violation > 0:
                details["rejection_reasons"].append(
                    "Phase-B text-only score "
                    f"{current_text_score:.6g} is more than "
                    f"{100.0 * tolerance:.2f}% worse than frozen-v2 teacher "
                    f"score {teacher_score:.6g}"
                )
            memory_details["phase_b_text_only_guard"] = {
                "text_only_score": current_text_score,
                "teacher_score": teacher_score,
                "allowed_text_only_score": allowed_text_score,
                "max_relative_degradation": tolerance,
                "violation": teacher_violation,
            }
        require_shuffled_improvement = bool(
            cfg.get("selection", {}).get(
                "require_sentence_memory_outperform_shuffled", True
            )
        )
        if memory_metrics and shuffled_metrics:
            memory_score = float(selection_diagnostics(memory_metrics, cfg)[0])
            shuffled_score = float(
                selection_diagnostics(shuffled_metrics, cfg)[0]
            )
            shuffled_minimum_improvement = float(
                cfg.get("selection", {}).get(
                    "sentence_memory_min_relative_improvement_over_shuffled",
                    0.001,
                )
            )
            if shuffled_minimum_improvement < 0:
                raise ValueError(
                    "sentence_memory_min_relative_improvement_over_shuffled "
                    "must be non-negative"
                )
            shuffled_scale = max(abs(shuffled_score), 1e-8)
            allowed_memory_score = (
                shuffled_score
                - shuffled_minimum_improvement * shuffled_scale
            )
            shuffled_violation = (
                max(memory_score - allowed_memory_score, 0.0)
                / shuffled_scale
                if math.isfinite(memory_score)
                and math.isfinite(shuffled_score)
                else float("inf")
            )
            if require_shuffled_improvement:
                violation = float(violation) + shuffled_violation
                feasible = bool(feasible and shuffled_violation <= 1e-12)
                if shuffled_violation > 0:
                    details["rejection_reasons"].append(
                        "sentence-memory score "
                        f"{memory_score:.6g} did not outperform shuffled-memory "
                        f"score {shuffled_score:.6g} by the required "
                        f"{100.0 * shuffled_minimum_improvement:.2f}%"
                    )
            memory_details.update(
                {
                    "shuffled_sentence_memory_score": shuffled_score,
                    "sentence_memory_allowed_score_vs_shuffled": (
                        allowed_memory_score
                    ),
                    "sentence_memory_relative_improvement_over_shuffled": (
                        (shuffled_score - memory_score) / shuffled_scale
                    ),
                    "sentence_memory_min_relative_improvement_over_shuffled": (
                        shuffled_minimum_improvement
                    ),
                    "sentence_memory_shuffled_improvement_violation": (
                        shuffled_violation
                    ),
                    "require_sentence_memory_outperform_shuffled": float(
                        require_shuffled_improvement
                    ),
                }
            )
        elif require_shuffled_improvement:
            violation = float("inf")
            feasible = False
            details["rejection_reasons"].append(
                "required sentence_memory and shuffled_sentence_memory "
                "validation metrics were not both produced"
            )
        details["dual_mode"] = memory_details
        result = (float(score), float(violation), bool(feasible))
        return (*result, details) if return_details else result

    text_metrics = _metrics_in_namespace(metrics, "text_only")
    word_metrics = _metrics_in_namespace(metrics, "word_prior")
    if not text_metrics or not word_metrics:
        namespace, selected_metrics = (
            ("text_only", text_metrics) if text_metrics else ("word_prior", word_metrics)
        )
        if not selected_metrics:
            raise ValueError("Dual-mode validation did not produce a configured mode")
        score, violation, feasible, details = selection_diagnostics(
            selected_metrics, cfg, return_details=True
        )
        details["dual_mode"] = {"selection_source": namespace}
        result = (score, violation, feasible)
        return (*result, details) if return_details else result
    (
        text_score,
        text_violation,
        text_feasible,
        details,
    ) = selection_diagnostics(text_metrics, cfg, return_details=True)
    word_score, _word_selection_violation, _word_feasible = selection_diagnostics(
        word_metrics, cfg
    )
    tolerance = float(
        cfg.get("selection", {}).get(
            "word_prior_max_relative_degradation", 0.02
        )
    )
    if tolerance < 0:
        raise ValueError("word_prior_max_relative_degradation must be non-negative")
    comparison_scale = max(abs(float(text_score)), 1e-8)
    allowed_word_score = float(text_score) + tolerance * comparison_scale
    word_finite = math.isfinite(float(word_score))
    word_violation = (
        max(float(word_score) - allowed_word_score, 0.0) / comparison_scale
        if word_finite and math.isfinite(float(text_score))
        else float("inf")
    )
    total_violation = float(text_violation) + word_violation
    feasible = bool(text_feasible and word_finite and word_violation <= 1e-12)
    relative_degradation = (
        (float(word_score) - float(text_score)) / comparison_scale
        if word_finite and math.isfinite(float(text_score))
        else float("inf")
    )
    if word_violation > 0:
        details["rejection_reasons"].append(
            "word-prior selection score "
            f"{word_score:.6g} is more than {100.0 * tolerance:.2f}% worse "
            f"than text-only score {text_score:.6g}"
        )
    details["dual_mode"] = {
        "selection_source": "text_only",
        "text_only_score": float(text_score),
        "word_prior_score": float(word_score),
        "word_prior_allowed_score": float(allowed_word_score),
        "word_prior_relative_degradation": float(relative_degradation),
        "word_prior_max_relative_degradation": tolerance,
        "word_prior_violation": float(word_violation),
    }
    result = (float(text_score), total_violation, feasible)
    return (*result, details) if return_details else result


def validate_checkpoint_contract(checkpoint, cfg, source="checkpoint"):
    expected_type, expected_version = checkpoint_contract(cfg)
    actual_type = checkpoint.get("model_type")
    actual_version = checkpoint.get("trajectory_contract_version")
    # Historical v1 checkpoints predate explicit identity metadata.
    if actual_type is None and expected_type == "continuous_trajectory_field":
        actual_type = "continuous_trajectory_field"
    if actual_version is None and expected_version == 1:
        actual_version = 1
    if actual_type != expected_type or int(actual_version or -1) != expected_version:
        raise RuntimeError(
            f"{source} has model_type={actual_type!r}, "
            f"trajectory_contract_version={actual_version!r}; expected "
            f"model_type={expected_type!r}, "
            f"trajectory_contract_version={expected_version}. "
            "Resume/evaluation requires an exact model contract match; use the "
            "explicit v2-to-v3 base-checkpoint path only for migration. A v2 "
            "contract cannot load v1 weights implicitly."
        )
    expected_text_encoder = configured_text_encoder_identity(cfg)
    actual_text_encoder = checkpoint.get("text_encoder_identity")
    if actual_text_encoder is None and isinstance(checkpoint.get("config"), dict):
        actual_text_encoder = configured_text_encoder_identity(checkpoint["config"])
    if (
        actual_text_encoder is not None
        and str(actual_text_encoder).replace("\\", "/") != expected_text_encoder
    ):
        raise RuntimeError(
            f"{source} uses text encoder {actual_text_encoder!r}; expected "
            f"{expected_text_encoder!r}. Text encoders with the same hidden size "
            "are still semantically incompatible."
        )


def validate_sentence_memory_checkpoint_identity(
    checkpoint,
    provider,
    source="checkpoint",
    cfg=None,
    text_encoder_identity=None,
):
    if cfg is not None and is_sentence_memory_model(cfg):
        actual_behavior = checkpoint.get("sentence_memory_behavior_identity")
        if actual_behavior is None and isinstance(checkpoint.get("config"), dict):
            actual_behavior = checkpoint["config"].get("sentence_memory", {}).get(
                "resolved_behavior_identity"
            )
        if actual_behavior is None:
            raise RuntimeError(
                f"{source} has no sentence-memory behavior identity"
            )
        actual_payload = {
            key: value
            for key, value in dict(actual_behavior).items()
            if key != "digest"
        }
        actual_digest = hashlib.sha256(
            json.dumps(
                actual_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if actual_behavior.get("digest") != actual_digest:
            raise RuntimeError(
                f"{source} has an invalid sentence-memory behavior identity digest"
            )
        expected_behavior = sentence_memory_behavior_identity(cfg)
        if actual_digest != expected_behavior["digest"]:
            raise RuntimeError(
                f"{source} sentence-memory behavior differs from the active config: "
                f"checkpoint={actual_digest}, active={expected_behavior['digest']}"
            )
    if provider is None:
        if text_encoder_identity is not None and is_sentence_memory_model(cfg or {}):
            from NIAF.continuous_trajectory_field.sentence_memory import (
                compare_text_encoder_identity,
                complete_text_encoder_identity,
            )

            checkpoint_memory = checkpoint.get("sentence_memory_identity") or {}
            expected_text = checkpoint_memory.get("text_encoder")
            if not isinstance(expected_text, dict):
                raise RuntimeError(
                    f"{source} has no persisted sentence-memory text-encoder identity"
                )
            active_text = complete_text_encoder_identity(text_encoder_identity)
            compare_text_encoder_identity(expected_text, active_text)
        return
    actual = checkpoint.get("sentence_memory_identity")
    if actual is None and isinstance(checkpoint.get("config"), dict):
        actual = checkpoint["config"].get("sentence_memory", {}).get(
            "resolved_identity"
        )
    if actual is None:
        raise RuntimeError(
            f"{source} has no sentence-memory bank identity"
        )
    expected = provider.identity
    if str(actual.get("bank_id")) != str(expected.get("bank_id")):
        raise RuntimeError(
            f"{source} uses sentence-memory bank {actual.get('bank_id')!r}; "
            f"active bank is {expected.get('bank_id')!r}"
        )
    actual_neighbors = actual.get("neighbor_tables")
    expected_neighbors = expected.get("neighbor_tables")
    if not isinstance(actual_neighbors, dict):
        raise RuntimeError(
            f"{source} has no persisted sentence-neighbor table identity"
        )
    if actual_neighbors != expected_neighbors:
        raise RuntimeError(
            f"{source} sentence-neighbor tables differ from the active provider"
        )


def checkpoint_selection_state(best_score, best_infeasible_score):
    """Serialize historical checkpoint-selection minima without infinities."""

    return {
        "schema_version": 1,
        "best_feasible_score": (
            float(best_score) if math.isfinite(float(best_score)) else None
        ),
        "best_infeasible_score": (
            float(best_infeasible_score)
            if math.isfinite(float(best_infeasible_score))
            else None
        ),
    }


def restore_checkpoint_selection_scores(checkpoint, *, require=False, source="checkpoint"):
    state = checkpoint.get("selection_state")
    if not isinstance(state, dict) or int(state.get("schema_version", -1)) != 1:
        if require:
            raise RuntimeError(
                f"{source} has no exact historical selection_state"
            )
        metrics = checkpoint.get("metrics") or {}
        score = float(metrics.get("selection_score", float("inf")))
        if bool(metrics.get("selection_feasible", False)):
            return score, float("inf")
        return float("inf"), score

    def parse(name):
        value = state.get(name)
        if value is None:
            return float("inf")
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"{source} selection_state.{name} is non-finite")
        return value

    return parse("best_feasible_score"), parse("best_infeasible_score")


def capture_local_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        ),
    }


def distributed_checkpoint_rng_state(dist_info):
    local = {
        "rank": int(dist_info.get("rank", 0)),
        "state": capture_local_rng_state(),
    }
    if dist_info.get("enabled", False):
        gathered = [None for _ in range(int(dist_info["world_size"]))]
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    return {
        "schema_version": 1,
        "world_size": int(dist_info.get("world_size", 1)),
        "rank_states": gathered,
    }


def restore_checkpoint_rng_state(checkpoint, dist_info, *, require=False, source="checkpoint"):
    payload = checkpoint.get("rng_state")
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        if require:
            raise RuntimeError(f"{source} has no exact RNG continuation state")
        return False
    expected_world_size = int(dist_info.get("world_size", 1))
    if int(payload.get("world_size", -1)) != expected_world_size:
        raise RuntimeError(
            f"{source} RNG state was saved for world_size="
            f"{payload.get('world_size')}, active world_size={expected_world_size}"
        )
    rank = int(dist_info.get("rank", 0))
    matches = [
        item.get("state")
        for item in payload.get("rank_states", ())
        if isinstance(item, dict) and int(item.get("rank", -1)) == rank
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError(f"{source} has no unique RNG state for rank {rank}")
    state = matches[0]
    numpy_state = state.get("numpy") or {}
    try:
        random.setstate(state["python"])
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("saved CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state(cuda_state.cpu(), torch.cuda.current_device())
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{source} contains malformed RNG state: {error}") from error
    return True


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    cfg,
    metrics,
    *,
    selection_state=None,
    rng_state=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type, contract_version = checkpoint_contract(cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": cfg,
            "metrics": metrics,
            "selection_state": selection_state,
            "rng_state": rng_state,
            "model_type": model_type,
            "trajectory_contract_version": contract_version,
            "text_encoder_identity": configured_text_encoder_identity(cfg),
            "sentence_memory_identity": cfg.get("sentence_memory", {}).get(
                "resolved_identity"
            ),
            "sentence_memory_behavior_identity": (
                sentence_memory_behavior_identity(cfg)
                if is_sentence_memory_model(cfg)
                else None
            ),
            "sentence_memory_resume_identity": (
                sentence_memory_resume_identity(cfg)
                if is_sentence_memory_model(cfg)
                else None
            ),
            "v2_to_v3_text_only_parity": cfg.get(
                "sentence_memory_safety", {}
            ).get("v2_to_v3_text_only_parity"),
            "phase_b_scientific_gate": cfg.get(
                "sentence_memory_safety", {}
            ).get("phase_b", {}).get("resolved_scientific_gate"),
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
                "base_checkpoint": cfg.get("train", {}).get("base_checkpoint"),
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


def validate_fresh_output_directory(out_dir, *, resume=None):
    """Prevent stale selection artifacts from masquerading as a fresh run."""

    out_dir = Path(out_dir)
    if resume is not None or not out_dir.exists():
        return
    try:
        first_entry = next(out_dir.iterdir())
    except StopIteration:
        return
    raise RuntimeError(
        f"Refusing to start a fresh run in nonempty output directory {out_dir}; "
        f"found {first_entry.name!r}. Use --resume for an exact continuation "
        "or choose a new --out_dir."
    )


def main():
    args = parse_args()
    if sum(
        value is not None
        for value in (args.resume, args.warm_start, args.base_checkpoint)
    ) > 1:
        raise ValueError(
            "--resume, --warm_start, and --base_checkpoint are mutually exclusive"
        )
    cfg = apply_overrides(load_config(args.config), args)
    validate_phase_b_launch(
        cfg,
        warm_start=args.warm_start,
        resume=args.resume,
        phase_b_gate_report=args.phase_b_gate_report,
    )
    if is_sentence_memory_model(cfg):
        cfg.setdefault("sentence_memory", {})[
            "resolved_behavior_identity"
        ] = sentence_memory_behavior_identity(cfg)
    if args.warm_start is not None and configured_model_type(cfg) == DUAL_MODE_MODEL_TYPE:
        raise ValueError(
            "Dual-mode v2 must be trained from scratch and does not accept "
            "--warm_start. Use --resume for an exact v2 continuation."
        )
    out_dir = Path(cfg["output"]["out_dir"])
    validate_fresh_output_directory(out_dir, resume=args.resume)
    dist_info = setup_distributed(args)
    set_seed(int(cfg.get("seed", 1234)) + int(dist_info.get("rank", 0)))
    device = resolve_distributed_device(cfg.get("device", "auto"), dist_info)
    text_device = torch.device(cfg.get("text", {}).get("device", "cpu"))
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
    provider = (
        ScaffoldProvider(cfg, train_dataset, device)
        if requires_scaffold_provider(cfg)
        else None
    )
    sentence_memory_provider = (
        build_sentence_memory_provider(cfg, text_encoder, dataset=train_dataset)
        if requires_sentence_memory_provider(cfg)
        else None
    )
    if sentence_memory_provider is not None:
        cfg.setdefault("sentence_memory", {})[
            "resolved_identity"
        ] = sentence_memory_provider.identity
        sentence_memory_provider.validate_query_dataset(
            train_dataset,
            require_neighbors=(
                configured_sentence_memory_train_mode(cfg) != "off"
            ),
        )
        sentence_memory_provider.validate_query_dataset(
            val_dataset,
            require_neighbors=any(
                mode != "off"
                for mode in configured_sentence_memory_eval_modes(cfg)
            ),
        )
        rank_zero_print(
            dist_info,
            "Sentence memory: "
            + json.dumps(
                getattr(sentence_memory_provider, "config_summary", {}),
                sort_keys=True,
            ),
        )
        if dist_info["is_main"]:
            (out_dir / "config.resolved.json").write_text(
                json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
            )
    elif is_sentence_memory_model(cfg):
        # A globally disabled/all-off v3 run has no bank provider, but its text
        # frontend remains part of the checkpoint contract.
        from NIAF.continuous_trajectory_field.sentence_memory import (
            complete_text_encoder_identity,
        )

        cfg.setdefault("sentence_memory", {})["resolved_identity"] = {
            "bank_id": None,
            "retrieval_active": False,
            "text_encoder": complete_text_encoder_identity(
                text_encoder.checkpoint_identity()
            ),
            "neighbor_tables": {},
        }
    retrieval_bank = (
        validate_train_only_retrieval_bank(cfg, provider)
        if provider is not None
        else {"enabled": False, "reason": "word_prior_disabled"}
    )
    rank_zero_print(dist_info, f"Retrieval bank: {json.dumps(retrieval_bank, sort_keys=True)}")
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})
    base_checkpoint_path = None
    if args.resume is None:
        configured_base = train_cfg.get("base_checkpoint")
        if configured_base:
            base_checkpoint_path = Path(configured_base)
    if base_checkpoint_path is not None and not is_sentence_memory_model(cfg):
        raise ValueError(
            "train.base_checkpoint/--base_checkpoint is supported only by the "
            "v3 sentence-memory model"
        )
    if (
        is_sentence_memory_model(cfg)
        and bool(train_cfg.get("freeze_base", False))
        and args.resume is None
        and args.warm_start is None
        and base_checkpoint_path is None
    ):
        raise ValueError(
            "A frozen-base sentence-memory run requires train.base_checkpoint "
            "or --base_checkpoint; refusing to freeze a randomly initialized v3 base."
        )
    start_epoch = 1
    global_step = 0
    optimizer_state = None
    resume_rng_checkpoint = None
    best_score = float("inf")
    best_infeasible_score = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_phase_b_resume_checkpoint(
            cfg, checkpoint, source=str(args.resume)
        )
        validate_checkpoint_contract(checkpoint, cfg, source=str(args.resume))
        if is_sentence_memory_model(cfg):
            validate_sentence_memory_resume_identity(
                checkpoint, cfg, source=str(args.resume)
            )
        validate_sentence_memory_checkpoint_identity(
            checkpoint,
            sentence_memory_provider,
            source=str(args.resume),
            cfg=cfg,
            text_encoder_identity=(
                text_encoder.checkpoint_identity()
                if hasattr(text_encoder, "checkpoint_identity")
                else None
            ),
        )
        if is_sentence_memory_model(cfg):
            parity = checkpoint.get("v2_to_v3_text_only_parity")
            if isinstance(parity, dict):
                cfg.setdefault("sentence_memory_safety", {})[
                    "v2_to_v3_text_only_parity"
                ] = parity
            gate = checkpoint.get("phase_b_scientific_gate")
            if isinstance(gate, dict):
                cfg.setdefault("sentence_memory_safety", {}).setdefault(
                    "phase_b", {}
                )["resolved_scientific_gate"] = gate
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer_state = checkpoint.get("optimizer")
        if is_sentence_memory_model(cfg) and optimizer_state is None:
            raise RuntimeError(
                f"{args.resume} has no optimizer state for exact v3 continuation"
            )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_score, best_infeasible_score = restore_checkpoint_selection_scores(
            checkpoint,
            require=is_sentence_memory_model(cfg),
            source=str(args.resume),
        )
        if is_sentence_memory_model(cfg):
            # Restore only after model/teacher/DDP/W&B initialization has
            # finished, so those setup steps cannot consume continuation RNG.
            resume_rng_checkpoint = checkpoint
    elif args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu")
        validate_checkpoint_contract(checkpoint, cfg, source=str(args.warm_start))
        if is_sentence_memory_model(cfg):
            validate_phase_b_warm_start_checkpoint(
                cfg, checkpoint, source=str(args.warm_start)
            )
            validate_sentence_memory_checkpoint_identity(
                checkpoint,
                sentence_memory_provider,
                source=str(args.warm_start),
                cfg=cfg,
                text_encoder_identity=(
                    text_encoder.checkpoint_identity()
                    if hasattr(text_encoder, "checkpoint_identity")
                    else None
                ),
            )
            parity = checkpoint.get("v2_to_v3_text_only_parity")
            cfg.setdefault("sentence_memory_safety", {})[
                "v2_to_v3_text_only_parity"
            ] = parity
            gate_report = configured_phase_b_gate_report(cfg)
            gate = validate_phase_b_scientific_gate(
                cfg,
                checkpoint,
                args.warm_start,
                gate_report,
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            rank_zero_print(
                dist_info,
                f"Warm-started v3 model weights from {args.warm_start} "
                f"(checkpoint_epoch={checkpoint.get('epoch', 0)}); optimizer, "
                "epoch, and selection state reset for a new phase; "
                f"scientific_gate={gate['gate_identity']['digest']}.",
            )
        else:
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
    elif base_checkpoint_path is not None:
        checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
        validate_sentence_memory_base_checkpoint(
            checkpoint, cfg, source=str(base_checkpoint_path)
        )
        incompatible = load_sentence_memory_base_state(
            model, checkpoint["model"]
        )
        parity = validate_v2_to_v3_text_only_parity(
            model,
            checkpoint,
            text_dim=text_encoder.text_dim,
            device=device,
            tolerance=float(
                train_cfg.get("base_checkpoint_parity_tolerance", 1e-7)
            ),
        )
        cfg.setdefault("sentence_memory_safety", {})[
            "v2_to_v3_text_only_parity"
        ] = parity
        rank_zero_print(
            dist_info,
            f"Initialized v3 base from {base_checkpoint_path} "
            f"(checkpoint_epoch={checkpoint.get('epoch', 0)}, "
            f"new_sentence_memory_parameters={len(incompatible.missing_keys)}, "
            f"text_only_max_abs={parity['prediction_max_abs']:.3g}); "
            "optimizer and epoch reset.",
        )

    trainable_summary = configure_sentence_memory_trainable_parameters(model, cfg)
    sentence_off_teacher = build_sentence_off_teacher(
        cfg, text_encoder.text_dim, device
    )
    if is_sentence_memory_model(cfg):
        if dist_info["is_main"]:
            (out_dir / "config.resolved.json").write_text(
                json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
            )
        rank_zero_print(
            dist_info,
            "Sentence-memory trainability: "
            f"freeze_base={trainable_summary['freeze_base']} "
            f"trainable_tensors={len(trainable_summary['trainable'])}",
        )

    model = wrap_model(
        model,
        dist_info,
        device,
        find_unused_parameters=is_dual_mode(cfg),
    )
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
    if resume_rng_checkpoint is not None:
        restore_checkpoint_rng_state(
            resume_rng_checkpoint,
            dist_info,
            require=True,
            source=str(args.resume),
        )
    for epoch in range(start_epoch, epochs + 1):
        optimizer_lrs = configure_optimizer_epoch(optimizer, cfg, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        if sentence_memory_provider is not None:
            sentence_memory_provider.set_epoch(epoch)
            sentence_memory_provider.validate_query_dataset(
                train_dataset,
                require_neighbors=(
                    configured_sentence_memory_train_mode(cfg) != "off"
                ),
            )
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
            sentence_memory_provider=sentence_memory_provider,
            sentence_off_teacher=sentence_off_teacher,
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
        epoch_checkpoint_rng = None
        if validation_due:
            # Persist the completed training epoch before entering the much
            # heavier dense-JVP validation path. Successful validation
            # overwrites this recovery snapshot with complete metrics below.
            row["validation_pending"] = 1.0
            pending_rng = distributed_checkpoint_rng_state(dist_info)
            if dist_info["is_main"]:
                save_checkpoint(
                    out_dir / "checkpoints" / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                    selection_state=checkpoint_selection_state(
                        best_score, best_infeasible_score
                    ),
                    rng_state=pending_rng,
                )
            barrier(dist_info)
            if dist_info["is_main"] and wandb_run is not None:
                wandb_run.log(wandb_validation_pending_payload(epoch, global_step))
            if sentence_memory_provider is not None:
                sentence_memory_provider.validate_query_dataset(
                    val_dataset,
                    require_neighbors=any(
                        mode != "off"
                        for mode in configured_sentence_memory_eval_modes(cfg)
                    ),
                )
            val_metrics = evaluate_configured_modes(
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
                sentence_memory_provider=sentence_memory_provider,
            )
            val_metrics = distributed_sample_weighted_mean_scalars(
                val_metrics,
                evaluated_loader_sample_count(
                    val_loader, int(cfg.get("eval", {}).get("max_batches", 0))
                ),
                device,
                dist_info,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["validation_pending"] = 0.0
            (
                score,
                constraint_violation,
                selection_feasible,
                selection_details,
            ) = checkpoint_selection_diagnostics(val_metrics, cfg, return_details=True)
            row["selection_score"] = score
            row["selection_constraint_violation"] = constraint_violation
            row["selection_feasible"] = float(selection_feasible)
            row["selection_rejection_reasons"] = "; ".join(
                selection_details["rejection_reasons"]
            )
            for name, value in selection_details.get("dual_mode", {}).items():
                if isinstance(value, (int, float)):
                    row[f"selection_{name}"] = value
            if dist_info["is_main"] and wandb_run is not None:
                wandb_run.log(wandb_validation_payload(row))
            keep_validation_checkpoint = bool(
                cfg.get("selection", {}).get("keep_validation_checkpoints", False)
            )
            improved_feasible = bool(selection_feasible and score < best_score)
            improved_infeasible = bool(
                not selection_feasible and score < best_infeasible_score
            )
            if selection_feasible and score < best_score:
                best_score = score
            elif not selection_feasible and score < best_infeasible_score:
                best_infeasible_score = score
            if (
                keep_validation_checkpoint
                or improved_feasible
                or improved_infeasible
                or epoch % save_every == 0
            ):
                epoch_checkpoint_rng = distributed_checkpoint_rng_state(dist_info)
            current_selection_state = checkpoint_selection_state(
                best_score, best_infeasible_score
            )
            if dist_info["is_main"] and keep_validation_checkpoint:
                save_checkpoint(
                    out_dir / "checkpoints" / f"epoch{epoch:04d}.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                    selection_state=current_selection_state,
                    rng_state=epoch_checkpoint_rng,
                )
            if dist_info["is_main"] and improved_feasible:
                save_checkpoint(
                    out_dir / "checkpoints" / "best.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                    selection_state=current_selection_state,
                    rng_state=epoch_checkpoint_rng,
                )
            elif dist_info["is_main"] and improved_infeasible:
                save_checkpoint(
                    out_dir / "checkpoints" / "best_infeasible.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                    selection_state=current_selection_state,
                    rng_state=epoch_checkpoint_rng,
                )

        if epoch % save_every == 0:
            if epoch_checkpoint_rng is None:
                epoch_checkpoint_rng = distributed_checkpoint_rng_state(dist_info)
            if dist_info["is_main"]:
                save_checkpoint(
                    out_dir / "checkpoints" / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    row,
                    selection_state=checkpoint_selection_state(
                        best_score, best_infeasible_score
                    ),
                    rng_state=epoch_checkpoint_rng,
                )
        if dist_info["is_main"]:
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
