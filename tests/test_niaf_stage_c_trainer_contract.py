from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.scripts import (
    train_continuous_trajectory_field as trainer,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    FIXED_EVIDENCE_CONTROLS_MODE,
    STAGE_C_GENERATOR_PREFIXES,
    STAGE_C_GENERATOR_PARAMETER_COUNT,
    STAGE_C_GENERATOR_PARAMETER_NAMES,
    build_optimizer,
    configure_sentence_memory_trainable_parameters,
    configured_sentence_off_distillation,
    requires_isolated_development_validation,
    selected_checkpoint_artifact_name,
    sentence_memory_architecture_identity,
    sentence_memory_behavior_identity,
    sentence_memory_evaluation_control_identity,
    sentence_memory_objective_identity,
    sentence_memory_selection_aggregation_identity,
    stage_c_trainability_contract,
    stage_c_optimizer_parameter_mapping_contract,
    validate_stage_c_calibration_transition,
    validate_stage_c_distributed_runtime,
    validate_stage_c_optimizer_parameter_mapping,
    validate_stage_c_resume_checkpoint,
    validate_stage_c_selection_comparability,
    validate_stage_c_training_contract,
    validate_stage_c_warm_start_checkpoint,
)


def _identity(**payload):
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**payload, "digest": digest}


def _calibration(identity):
    return _identity(
        schema_name="signtrajfield_sentence_memory_relevance_calibration",
        schema_version=1,
        artifact_identity=identity,
        calibration_sha256=("1" if identity == "source" else "2") * 64,
        map_sha256="3" * 64,
        map_content_digest="5" * 64,
        heldout_auroc=0.98,
        heldout_probability_gap=0.79,
        minimum_heldout_auroc=0.75,
        minimum_heldout_probability_gap=0.20,
        coefficients={
            "a": 2.0,
            "b": 0.25,
            "formula": "sigmoid(a*(adjusted_score-b))",
            "intercept": -0.5,
            "intercept_derivation": "float64(-a*b)",
            "slope": 2.0,
        },
    )


def _model_cfg(calibration):
    return {
        "model": {
            "type": "sentence_memory_continuous_trajectory_field",
            "dropout": 0.0,
        },
        "sentence_memory": {
            "enabled": True,
            "key_value_mode": "factorized_metadata_motion_v1",
            "candidate_value_mode": "centered_candidate_covariance_v1",
            "relevance_gate_mode": "frozen_absolute_adjusted_score_v1",
            "association_mode": "absolute_text_motion_v1",
            "association_dim": 128,
            "association_temperature": 0.10,
            "temporal_prior_mode": "none",
            "duration_weight": 0.05,
            "relevance_slope": 2.0,
            "relevance_intercept": -0.5,
            "resolved_relevance_calibration_identity": calibration,
        },
        "conditioning": {
            "word_prior_train_mode": "off",
            "sentence_memory_train_mode": "dropout",
        },
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_c_cfg(tmp_path, *, arm="memory"):
    active_calibration = _calibration("active")
    cfg = _model_cfg(active_calibration)
    cfg["seed"] = 1234
    source_config_path = tmp_path / "source_stage_b.yaml"
    source_config_path.write_text("source-stage-b\n", encoding="utf-8")
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"frozen-v2")
    calibration_dir = tmp_path / "fresh_calibration"
    cfg["sentence_memory"]["relevance_calibration"] = {
        "artifact_dir": str(calibration_dir)
    }
    cfg["conditioning"]["sentence_memory_train_mode"] = (
        "dropout" if arm == "memory" else "off"
    )
    cfg["conditioning"]["sentence_memory_dropout_probability"] = (
        0.25 if arm == "memory" else 1.0
    )
    cfg["eval"] = {
        "max_samples_per_memory_batch": 1,
        "max_frames_per_memory_batch": 512,
        "sentence_memory_modes": [
            "off",
            "on",
            "motion_shuffled_n0",
            "motion_shuffled_n1",
            "motion_shuffled_n2",
            "cross_query_motion",
            "full_replacement",
            "joint_tuple_permuted",
            "uniform_final_mass",
            "analytic_prior",
            "association_disabled",
        ],
        "evaluation_corruption": {
            "mode": FIXED_EVIDENCE_CONTROLS_MODE,
            "seed": 1234,
        },
    }
    cfg["train"] = {
        "base_checkpoint": None,
        "freeze_base": True,
        "freeze_sentence_memory": True,
        "unfreeze_base_prefixes": list(STAGE_C_GENERATOR_PREFIXES),
        "epochs": 1,
        "batch_size": 64,
        "accumulation_steps": 2,
        "early_stopping_patience": 0,
        "early_stopping_min_epochs": 1,
        "local_warmup_epochs": 0,
        "val_every": 1,
        "save_every": 1,
        "joint_global_lr": 1e-5,
        "joint_local_lr": 1e-5,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "length_bucketed_batches": True,
        "drop_last": False,
        "max_samples_per_memory_batch": 8,
        "max_frames_per_memory_batch": 2048,
    }
    cfg["objective"] = {
        "lambda_sentence_safe": 1.0,
        "lambda_sentence_shuffle": 1.0,
        "lambda_sentence_sparsity": 1e-4,
        "lambda_sentence_off_distill": 1.0,
        "lambda_sentence_association_bce": 0.10,
        "lambda_sentence_association_infonce": 0.05,
    }
    cfg["data"] = {"train_split": "train", "val_split": "val"}
    cfg["selection"] = {
        "require_feasible": False,
        "aggregation": "normalized_text_cluster_equal_v1",
    }
    cfg["validation_text_partition"] = {
        "enabled": True,
        "split": "val",
        "partition_digest": "8" * 64,
        "expected_partition_digest": "8" * 64,
        "expected_development_manifest_sha256": "9" * 64,
        "expected_development_rows": 347,
        "development_text_count": 256,
        "evaluation_corruption_map_identity": _identity(
            schema_name="synthetic_stage_c_corruption_map",
            schema_version=1,
        ),
    }
    cfg["sentence_memory_safety"] = {
        "enabled": True,
        "nonregression_margin": 0.0,
        "shuffle_probability": 0.25,
        "shuffle_huber_beta": 0.10,
        "paired_corruption": {"enabled": False},
        "phase_b": {"enabled": False},
        "stage_c": {
            "schema_name": "signtrajfield_centered_stage_c_generator_adaptation",
            "schema_version": 1,
            "enabled": True,
            "development_only": True,
            "non_authorizing": True,
            "promotion_eligible": False,
            "arm": arm,
            "huber_beta": 0.10,
            "text_only_max_relative_degradation": 0.005,
            "source_checkpoint": {
                "path": str(tmp_path / "best_infeasible.pt"),
                "sha256": "0" * 64,
                "selection_status": "best_infeasible",
                "epoch": 5,
                "global_step": 360,
            },
            "source_terminal_decision": {
                "path": str(tmp_path / "decision.json"),
                "sha256": "0" * 64,
                "decision_identity": "6" * 64,
                "status": "valid_infeasible",
                "authorized_purpose": None,
            },
            "frozen_v2_teacher": {
                "path": str(teacher_path),
                "sha256": _sha(teacher_path),
            },
            "source_stage_b": {
                "config_path": str(source_config_path),
                "config_sha256": _sha(source_config_path),
                "architecture_identity": "0" * 64,
            },
            "active_stage_c": {
                "calibration_artifact_dir": str(calibration_dir),
                "calibration_schema_name": (
                    "signtrajfield_sentence_memory_relevance_calibration"
                ),
                "calibration_schema_version": 1,
            },
        },
    }
    return cfg


def _source_checkpoint(cfg):
    source_cfg = _model_cfg(_calibration("source"))
    source_cfg["selection"] = copy.deepcopy(cfg["selection"])
    source_cfg["selection"]["require_feasible"] = True
    source_cfg["eval"] = copy.deepcopy(cfg["eval"])
    source_cfg["validation_text_partition"] = copy.deepcopy(
        cfg["validation_text_partition"]
    )
    source_cfg["sentence_memory_safety"] = {
        "phase_b": {"enabled": False},
        "paired_corruption": {"enabled": True},
    }
    architecture = sentence_memory_architecture_identity(source_cfg)
    behavior = sentence_memory_behavior_identity(source_cfg)
    cfg["sentence_memory_safety"]["stage_c"]["source_stage_b"][
        "architecture_identity"
    ] = architecture["digest"]
    return {
        "epoch": 5,
        "global_step": 360,
        "config": source_cfg,
        "selection_state": {
            "schema_version": 3,
            "best_feasible_score": None,
            "best_infeasible_score": 1.0,
            "best_infeasible_key": [2.0, 1.0],
        },
        "metrics": {
            "selection_feasible": 0.0,
            "val_text_only/pred_loss_endpoint": 1.0,
        },
        "sentence_memory_architecture_identity": architecture,
        "sentence_memory_behavior_identity": behavior,
        "sentence_memory_relevance_calibration_identity": (
            source_cfg["sentence_memory"][
                "resolved_relevance_calibration_identity"
            ]
        ),
        "sentence_memory_selection_aggregation_identity": (
            sentence_memory_selection_aggregation_identity(cfg)
        ),
        "sentence_memory_evaluation_control_identity": (
            sentence_memory_evaluation_control_identity(cfg)
        ),
        "sentence_memory_validation_corruption_map_identity": copy.deepcopy(
            cfg["validation_text_partition"][
                "evaluation_corruption_map_identity"
            ]
        ),
        "sentence_memory_identity": {
            "bank_id": "bank",
            "neighbor_tables": {"train": {"sha256": "7" * 64}},
        },
        "v2_to_v3_text_only_parity": {
            "passed": True,
            "prediction_max_abs": 0.0,
            "duration_max_abs": 0.0,
        },
    }


def _synthetic_stage_c_pins(cfg):
    stage_c = cfg["sentence_memory_safety"]["stage_c"]
    return {
        name: copy.deepcopy(stage_c[name])
        for name in trainer.STAGE_C_PINNED_SOURCE
    }


def _validate_synthetic_contract(cfg, **kwargs):
    with patch.object(
        trainer, "STAGE_C_PINNED_SOURCE", _synthetic_stage_c_pins(cfg)
    ):
        return validate_stage_c_training_contract(cfg, **kwargs)


@pytest.mark.parametrize(
    ("config_name", "arm", "mode", "dropout_probability"),
    (
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_"
            "generator_adaptation_memory_pilot.yaml",
            "memory",
            "dropout",
            0.25,
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_"
            "generator_adaptation_matched_off_pilot.yaml",
            "matched_off",
            "off",
            1.0,
        ),
    ),
)
def test_checked_in_stage_c_configs_match_core_contract(
    config_name, arm, mode, dropout_probability
):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "NIAF"
        / "continuous_trajectory_field"
        / "configs"
        / config_name
    )
    cfg = load_config(config_path)
    stage_c = cfg["sentence_memory_safety"]["stage_c"]
    validated = validate_stage_c_training_contract(
        cfg, stage_c_warm_start=Path(stage_c["source_checkpoint"]["path"])
    )
    objective = sentence_memory_objective_identity(cfg)
    assert validated["arm"] == arm
    assert objective["sentence_memory_train_mode"] == mode
    assert objective["sentence_memory_dropout_probability"] == dropout_probability


def test_stage_c_contract_is_isolated_and_arm_locked(tmp_path):
    cfg = _stage_c_cfg(tmp_path)
    warm_start = Path(cfg["sentence_memory_safety"]["stage_c"]["source_checkpoint"]["path"])
    assert _validate_synthetic_contract(
        cfg, stage_c_warm_start=warm_start
    )["arm"] == "memory"
    assert requires_isolated_development_validation(cfg)
    phase, settings = configured_sentence_off_distillation(cfg)
    assert phase == "stage_c"
    assert settings["frozen_v2_teacher"]["sha256"]
    objective = sentence_memory_objective_identity(cfg)
    assert objective["schema_version"] == 4
    assert objective["mode"] == "centered_stage_c_generator_adaptation_v1"
    assert objective["sentence_memory_dropout_probability"] == 0.25
    assert objective["trainability"]["freeze_sentence_memory"] is True
    assert objective["optimizer"] == {
        "joint_global_lr": 1e-5,
        "joint_local_lr": 1e-5,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "local_warmup_epochs": 0,
    }
    assert objective["bounded_memory_batches"]["seed"] == 1234
    assert selected_checkpoint_artifact_name(cfg, feasible=True) == (
        "best_exploratory.pt"
    )
    assert selected_checkpoint_artifact_name(cfg, feasible=False) == (
        "best_exploratory_infeasible.pt"
    )

    mismatched = copy.deepcopy(cfg)
    mismatched["conditioning"]["sentence_memory_train_mode"] = "off"
    with pytest.raises(ValueError, match="arm 'memory'.*'dropout'"):
        _validate_synthetic_contract(
            mismatched, stage_c_warm_start=warm_start
        )
    matched_off = copy.deepcopy(cfg)
    matched_off["sentence_memory_safety"]["stage_c"]["arm"] = "matched_off"
    matched_off["conditioning"]["sentence_memory_train_mode"] = "off"
    matched_off["conditioning"]["sentence_memory_dropout_probability"] = 1.0
    assert _validate_synthetic_contract(
        matched_off, stage_c_warm_start=warm_start
    )["arm"] == "matched_off"
    matched_off_objective = sentence_memory_objective_identity(matched_off)
    assert matched_off_objective["sentence_memory_dropout_probability"] == 1.0
    assert matched_off_objective["digest"] != objective["digest"]
    for dropout_probability in (0.5, float("nan")):
        changed = copy.deepcopy(cfg)
        changed["conditioning"][
            "sentence_memory_dropout_probability"
        ] = dropout_probability
        with pytest.raises(
            ValueError, match="sentence_memory_dropout_probability=0.25"
        ):
            _validate_synthetic_contract(
                changed, stage_c_warm_start=warm_start
            )
    for name, value in (
        ("epochs", 2),
        ("batch_size", 32),
        ("accumulation_steps", 1),
        ("early_stopping_patience", 1),
        ("early_stopping_min_epochs", 2),
        ("local_warmup_epochs", 1),
        ("val_every", 2),
        ("save_every", 2),
    ):
        changed = copy.deepcopy(cfg)
        changed["train"][name] = value
        with pytest.raises(ValueError, match=rf"train\.{name}"):
            _validate_synthetic_contract(
                changed, stage_c_warm_start=warm_start
            )

    truncated = copy.deepcopy(cfg)
    truncated["data"]["limit_train"] = 128
    with pytest.raises(ValueError, match="forbids data.limit_train"):
        _validate_synthetic_contract(
            truncated, stage_c_warm_start=warm_start
        )

    for section, name, value, message in (
        ("objective", "lambda_sentence_safe", 0.5, "objective.lambda_sentence_safe"),
        (
            "objective",
            "lambda_sentence_shuffle",
            0.5,
            "objective.lambda_sentence_shuffle",
        ),
        (
            "objective",
            "lambda_sentence_sparsity",
            0.0,
            "objective.lambda_sentence_sparsity",
        ),
        (
            "objective",
            "lambda_sentence_off_distill",
            0.5,
            "objective.lambda_sentence_off_distill",
        ),
        (
            "sentence_memory_safety",
            "nonregression_margin",
            0.01,
            "sentence_memory_safety.nonregression_margin",
        ),
        (
            "sentence_memory_safety",
            "shuffle_probability",
            0.5,
            "sentence_memory_safety.shuffle_probability",
        ),
        (
            "sentence_memory_safety",
            "shuffle_huber_beta",
            0.2,
            "sentence_memory_safety.shuffle_huber_beta",
        ),
        ("train", "joint_global_lr", 2e-5, "train.joint_global_lr"),
        ("train", "joint_local_lr", 2e-5, "train.joint_local_lr"),
        ("train", "weight_decay", 0.0, "train.weight_decay"),
        ("train", "grad_clip", 0.5, "train.grad_clip"),
        (
            "train",
            "max_samples_per_memory_batch",
            4,
            "train.max_samples_per_memory_batch",
        ),
        (
            "train",
            "max_frames_per_memory_batch",
            1024,
            "train.max_frames_per_memory_batch",
        ),
        (
            "eval",
            "max_samples_per_memory_batch",
            2,
            "eval.max_samples_per_memory_batch",
        ),
        (
            "eval",
            "max_frames_per_memory_batch",
            1024,
            "eval.max_frames_per_memory_batch",
        ),
    ):
        changed = copy.deepcopy(cfg)
        changed[section][name] = value
        with pytest.raises(ValueError, match=message):
            _validate_synthetic_contract(
                changed, stage_c_warm_start=warm_start
            )
    changed = copy.deepcopy(cfg)
    changed["seed"] = 4321
    with pytest.raises(ValueError, match="config.seed"):
        _validate_synthetic_contract(
            changed, stage_c_warm_start=warm_start
        )
    for name, value in (
        ("length_bucketed_batches", False),
        ("drop_last", True),
    ):
        changed = copy.deepcopy(cfg)
        changed["train"][name] = value
        with pytest.raises(ValueError, match=rf"train\.{name}"):
            _validate_synthetic_contract(
                changed, stage_c_warm_start=warm_start
            )

    pinned = _synthetic_stage_c_pins(cfg)
    changed = copy.deepcopy(cfg)
    changed["sentence_memory_safety"]["stage_c"]["source_checkpoint"][
        "sha256"
    ] = "f" * 64
    with patch.object(trainer, "STAGE_C_PINNED_SOURCE", pinned), pytest.raises(
        ValueError, match="exact approved source_checkpoint"
    ):
        validate_stage_c_training_contract(
            changed, stage_c_warm_start=warm_start
        )


def test_stage_c_runtime_requires_two_one_gpu_nccl_ranks(tmp_path):
    cfg = _stage_c_cfg(tmp_path)
    runtime = {
        "enabled": True,
        "world_size": 2,
        "backend": "nccl",
        "local_rank": 0,
    }
    with patch("torch.cuda.device_count", return_value=1):
        identity = validate_stage_c_distributed_runtime(
            cfg, runtime, torch.device("cuda:0")
        )
    assert identity["effective_global_batch"] == 256
    assert identity["world_size"] == 2
    with patch("torch.cuda.device_count", return_value=1), pytest.raises(
        RuntimeError, match="world_size=2"
    ):
        validate_stage_c_distributed_runtime(
            cfg, {**runtime, "world_size": 1}, torch.device("cuda:0")
        )


class _Hypernetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_context = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, 256),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.coarse_head = nn.Linear(128, 2304)
        self.residual_head = nn.Linear(128, 2181)
        self.gate_head = nn.Linear(128, 4)
        self.local_context = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 128),
            nn.SiLU(),
            nn.Identity(),
        )
        self.local_head = nn.Linear(128, 2182)
        self.local_gate_head = nn.Linear(128, 4)
        self.sentence_memory_fusion = nn.Linear(2, 2)
        self.unapproved = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hypernetwork = _Hypernetwork()


def test_stage_c_trainability_freezes_memory_and_uses_exact_allowlist(tmp_path):
    cfg = _stage_c_cfg(tmp_path)
    model = _Model()
    summary = configure_sentence_memory_trainable_parameters(model, cfg)
    assert summary["freeze_sentence_memory"] is True
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        name
        for name, _parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in STAGE_C_GENERATOR_PREFIXES)
    }
    assert not any("sentence_memory" in name for name in trainable)
    assert not any("unapproved" in name for name in trainable)
    assert tuple(sorted(trainable)) == STAGE_C_GENERATOR_PARAMETER_NAMES
    assert summary["trainable_parameter_count"] == (
        STAGE_C_GENERATOR_PARAMETER_COUNT
    )
    assert summary["trainability_contract"] == stage_c_trainability_contract()
    optimizer = build_optimizer(model, cfg)
    assert validate_stage_c_optimizer_parameter_mapping(
        model, optimizer, cfg
    ) == stage_c_optimizer_parameter_mapping_contract()

    drifted = _Model()
    drifted.hypernetwork.global_context.add_module("6", nn.Linear(1, 1))
    with pytest.raises(RuntimeError, match="exactly the 20 pinned"):
        configure_sentence_memory_trainable_parameters(drifted, cfg)


def test_stage_c_calibration_transition_allows_only_provenance_change(tmp_path):
    cfg = _stage_c_cfg(tmp_path)
    checkpoint = _source_checkpoint(cfg)
    transition = validate_stage_c_calibration_transition(cfg, checkpoint)
    assert transition["source_architecture_identity"] != transition[
        "active_architecture_identity"
    ]
    assert transition["semantic_equivalence"]["coefficients_exact"]

    changed = copy.deepcopy(cfg)
    changed["sentence_memory"]["resolved_relevance_calibration_identity"] = (
        _calibration("active")
    )
    changed["sentence_memory"]["resolved_relevance_calibration_identity"][
        "coefficients"
    ]["slope"] = 3.0
    changed["sentence_memory"]["resolved_relevance_calibration_identity"] = _identity(
        **{
            key: value
            for key, value in changed["sentence_memory"][
                "resolved_relevance_calibration_identity"
            ].items()
            if key != "digest"
        }
    )
    with pytest.raises(RuntimeError, match="coefficients"):
        validate_stage_c_calibration_transition(changed, checkpoint)

    for field, value in (
        ("map_sha256", "4" * 64),
        ("map_content_digest", "a" * 64),
        ("heldout_auroc", 0.97),
        ("heldout_probability_gap", 0.78),
        ("minimum_heldout_auroc", 0.76),
        ("minimum_heldout_probability_gap", 0.21),
    ):
        changed = copy.deepcopy(cfg)
        calibration = copy.deepcopy(
            changed["sentence_memory"][
                "resolved_relevance_calibration_identity"
            ]
        )
        calibration[field] = value
        changed["sentence_memory"][
            "resolved_relevance_calibration_identity"
        ] = _identity(
            **{key: item for key, item in calibration.items() if key != "digest"}
        )
        with pytest.raises(RuntimeError, match=field):
            validate_stage_c_calibration_transition(changed, checkpoint)


def test_stage_c_selection_reference_uses_exact_source_score_definition(tmp_path):
    cfg = _stage_c_cfg(tmp_path)
    checkpoint = _source_checkpoint(cfg)
    comparability = validate_stage_c_selection_comparability(cfg, checkpoint)
    assert comparability["evaluation_modes"] == cfg["eval"][
        "sentence_memory_modes"
    ]
    assert comparability["active_require_feasible"] is False
    assert comparability["source_require_feasible"] is True

    changed = copy.deepcopy(cfg)
    changed["selection"]["weights"] = {"pred_loss_endpoint": 2.0}
    with pytest.raises(RuntimeError, match="changed fields=.*weights"):
        validate_stage_c_selection_comparability(changed, checkpoint)

    changed = copy.deepcopy(cfg)
    changed["eval"]["sentence_memory_modes"] = list(
        reversed(changed["eval"]["sentence_memory_modes"])
    )
    with pytest.raises(ValueError, match="evaluation modes must exactly match"):
        validate_stage_c_selection_comparability(changed, checkpoint)

    changed = copy.deepcopy(cfg)
    changed["eval"]["evaluation_corruption"]["seed"] = 99
    with pytest.raises(ValueError, match="seed 1234"):
        validate_stage_c_selection_comparability(changed, checkpoint)


def test_stage_c_warm_start_binds_terminal_decision_and_resume(tmp_path):
    cfg = _stage_c_cfg(tmp_path)
    checkpoint = _source_checkpoint(cfg)
    checkpoint_path = Path(
        cfg["sentence_memory_safety"]["stage_c"]["source_checkpoint"]["path"]
    )
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = _sha(checkpoint_path)
    stage_c = cfg["sentence_memory_safety"]["stage_c"]
    stage_c["source_checkpoint"]["sha256"] = checkpoint_sha
    decision = {
        "schema_name": "signtrajfield_centered_memory_ordered_decision",
        "schema_version": 1,
        "decision_identity": stage_c["source_terminal_decision"][
            "decision_identity"
        ],
        "status": "valid_infeasible",
        "stage": "stage2",
        "integrity_valid": True,
        "development_feasible": False,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "epoch": 5,
            "global_step": 360,
            "has_feasible_checkpoint": False,
            "config": {
                "sha256": stage_c["source_stage_b"]["config_sha256"]
            },
            "identities": {
                "architecture": {
                    "digest": stage_c["source_stage_b"][
                        "architecture_identity"
                    ]
                }
            },
        },
        "development_integrity": {
            "v2_checkpoint_sha256": stage_c["frozen_v2_teacher"]["sha256"]
        },
    }
    decision_path = Path(stage_c["source_terminal_decision"]["path"])
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    stage_c["source_terminal_decision"]["sha256"] = _sha(decision_path)
    provider = SimpleNamespace(
        identity={
            "bank_id": "bank",
            "neighbor_tables": {"train": {"sha256": "7" * 64}},
        }
    )
    with patch("torch.cuda.device_count", return_value=1):
        distribution = validate_stage_c_distributed_runtime(
            cfg,
            {
                "enabled": True,
                "world_size": 2,
                "backend": "nccl",
                "local_rank": 0,
            },
            torch.device("cuda:0"),
        )
    _validate_synthetic_contract(
        cfg, stage_c_warm_start=checkpoint_path
    )
    provenance = validate_stage_c_warm_start_checkpoint(
        cfg, checkpoint, checkpoint_path, provider=provider
    )
    assert provenance["scope"]["non_authorizing"]
    assert provenance["state_reset"]["optimizer"] == "fresh"
    assert provenance["memory_semantic_equivalence"]["neighbor_tables_exact"]

    resumed = {
        "config": copy.deepcopy(cfg),
        "stage_c_provenance": provenance,
        "stage_c_distribution_contract": distribution,
        "stage_c_trainability_contract": stage_c_trainability_contract(),
        "stage_c_optimizer_parameter_mapping": (
            stage_c_optimizer_parameter_mapping_contract()
        ),
    }
    restored = copy.deepcopy(cfg)
    restored["sentence_memory_safety"]["stage_c"].pop(
        "resolved_source_provenance"
    )
    restored["sentence_memory_safety"]["stage_c"][
        "resolved_distribution_contract"
    ] = copy.deepcopy(distribution)
    assert validate_stage_c_resume_checkpoint(restored, resumed)["digest"] == (
        provenance["digest"]
    )
