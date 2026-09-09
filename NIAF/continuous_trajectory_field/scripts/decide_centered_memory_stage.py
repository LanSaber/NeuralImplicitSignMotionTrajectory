"""Fail-closed ordered operations for centered/relevance Phase-A'''.

This is a separate protocol adapter: importing or executing it never changes
the on-disk Phase-A'' helper.  It reuses its mature partition/export/parity
audits inside this process while installing centered-only identities, config
validation, six-epoch replay, and relevance-calibration binding.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.relevance_calibration import (
    sha256_file,
    validate_relevance_calibration_artifact,
    validate_relevance_calibration_source,
)
from NIAF.continuous_trajectory_field.scripts import (
    decide_factorized_memory_stage as legacy,
)


SCHEMA_VERSION = 1
SCHEMA_NAME = "signtrajfield_centered_memory_ordered_decision"
AUTHORIZATION_SCHEMA_NAME = "signtrajfield_centered_memory_authorization"
HOLDOUT_SPEND_SCHEMA_NAME = "signtrajfield_centered_memory_confirmation_spend"
STAGE2_INPUT_SCHEMA_NAME = "signtrajfield_centered_memory_stage2_input"
LAUNCH_INPUT_SCHEMA_NAME = "signtrajfield_centered_memory_run_launch"
RESUME_ATTEMPT_SCHEMA_NAME = "signtrajfield_centered_memory_resume_attempt"
EXECUTION_LEASE_SCHEMA_NAME = "signtrajfield_centered_memory_execution_lease"
EXECUTION_LEASE_ATTESTATION_SCHEMA_NAME = (
    "signtrajfield_centered_memory_execution_lease_attestation"
)
CALIBRATION_COMPLETION_SCHEMA_NAME = (
    "signtrajfield_sentence_memory_relevance_calibration_completion"
)
DIAGNOSTIC_SCHEMA_NAME = "signtrajfield_centered_locked_development_diagnostic"
DIAGNOSTIC_READY_SCHEMA_NAME = (
    "signtrajfield_centered_locked_development_diagnostic_ready"
)
CALIBRATION_STAGE = "calibration"
STAGE1 = "stage1"
STAGE2 = "stage2"
STAGES = (STAGE1, STAGE2)
LEASE_STAGES = (CALIBRATION_STAGE, *STAGES)
EXPERIMENTS = {
    STAGE1: (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
        "relevance_motion_contrast_v1"
    ),
    STAGE2: (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
        "absolute_binding_motion_contrast_v1"
    ),
}
LEASE_EXPERIMENTS = {
    CALIBRATION_STAGE: "csl_daily_sentence_memory_relevance_calibration_v1_retry3",
    **EXPERIMENTS,
}
STAGE1_EVAL_MODES = (
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
)
STAGE2_EVAL_MODES = (*STAGE1_EVAL_MODES, "association_disabled")
EVAL_MODES = {STAGE1: STAGE1_EVAL_MODES, STAGE2: STAGE2_EVAL_MODES}
EVALUATION_CORRUPTION = {
    "mode": "fixed_evidence_controls_v1",
    "seed": 1234,
    "nonce": None,
    "nonces": {
        "motion_shuffled_n0": "csl_daily_pair_derangement_v1_n0",
        "motion_shuffled_n1": "csl_daily_pair_derangement_v1_n1",
        "motion_shuffled_n2": "csl_daily_pair_derangement_v1_n2",
        "cross_query_motion": "csl_daily_cross_query_motion_v1",
        "full_replacement": "csl_daily_full_candidate_replacement_v1",
        "joint_tuple_permuted": "csl_daily_joint_tuple_permutation_v1",
    },
}
MINIMUM_EPOCHS = 3
MAXIMUM_EPOCHS = 6
PATIENCE = 2
EXPECTED_BANK_ID = legacy.EXPECTED_BANK_ID
EXPECTED_TRAIN_NEIGHBOR_SHA256 = legacy.EXPECTED_NEIGHBOR_SHA256["train"]
EXPECTED_TRAIN_ROWS = 18_399
GLOBAL_HOLDOUT_SPEND_RELATIVE = Path(
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/"
    "confirmation_holdout_spent.json"
)
SOURCE_ROOT = Path(__file__).resolve().parents[3]
OrderedDecisionError = legacy.OrderedDecisionError
_legacy_resolved_config_projection = legacy._resolved_config_projection
_ACTIVE_INTEGRITY_AUDIT: Path | None = None


def _calibration_directory(cfg: Mapping[str, Any]) -> Path:
    configured = dict(
        dict(cfg.get("sentence_memory", {}) or {}).get("relevance_calibration", {})
        or {}
    )
    directory = Path(str(configured.get("artifact_dir", "")))
    if not str(directory):
        raise OrderedDecisionError("Centered config has no calibration artifact_dir")
    if not directory.is_absolute():
        directory = SOURCE_ROOT / directory
    expected = SOURCE_ROOT / (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_sentence_memory_relevance_calibration_v1_retry3"
    )
    if directory.resolve() != expected.resolve():
        raise OrderedDecisionError(
            "Centered config uses an unapproved calibration path"
        )
    return directory.resolve()


def _validate_and_resolve_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    memory = dict(cfg.get("sentence_memory", {}) or {})
    configured = dict(memory.get("relevance_calibration", {}) or {})
    expected_keys = {
        "artifact_dir",
        "artifact_identity",
        "schema_name",
        "schema_version",
        "minimum_heldout_auroc",
        "minimum_heldout_probability_gap",
    }
    if set(configured) - expected_keys:
        raise OrderedDecisionError("Calibration config contains unsupported fields")
    if (
        configured.get("schema_name")
        != "signtrajfield_sentence_memory_relevance_calibration"
        or int(configured.get("schema_version", -1)) != 1
        or float(configured.get("minimum_heldout_auroc", math.nan)) != 0.75
        or float(configured.get("minimum_heldout_probability_gap", math.nan)) != 0.20
    ):
        raise OrderedDecisionError("Calibration schema or gates changed")
    try:
        calibration = validate_relevance_calibration_artifact(
            _calibration_directory(cfg),
            expected_identity=configured.get("artifact_identity"),
            minimum_auroc=0.75,
            minimum_probability_gap=0.20,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise OrderedDecisionError("Sealed relevance calibration is invalid") from error
    inputs = dict(calibration.get("inputs", {}) or {})
    bank_input = dict(inputs.get("bank", {}) or {})
    neighbor_input = dict(inputs.get("train_neighbor_table", {}) or {})
    query_split = dict(calibration.get("query_split", {}) or {})
    if (
        bank_input.get("bank_id") != EXPECTED_BANK_ID
        or int(bank_input.get("item_count", -1)) != EXPECTED_TRAIN_ROWS
        or neighbor_input.get("sha256") != EXPECTED_TRAIN_NEIGHBOR_SHA256
        or int(neighbor_input.get("query_count", -1)) != EXPECTED_TRAIN_ROWS
        or int(query_split.get("fit_queries", -1)) != 16_559
        or int(query_split.get("holdout_queries", -1)) != 1_840
    ):
        raise OrderedDecisionError(
            "Calibration is not bound to the approved train bank"
        )
    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    # Trainer and every public construction path use the same resolver.  It
    # reopens the three sealed files before copying coefficients into cfg.
    try:
        resolved = trainer.resolve_sentence_memory_relevance_calibration(cfg)
    except (OSError, RuntimeError, ValueError) as error:
        raise OrderedDecisionError("Calibration runtime resolution failed") from error
    if not isinstance(resolved, Mapping):
        raise OrderedDecisionError("Calibration runtime identity is absent")
    if resolved.get("artifact_identity") != calibration.get("identity") or resolved.get(
        "map_content_digest"
    ) != dict(calibration.get("map", {}) or {}).get("content_digest"):
        raise OrderedDecisionError("Runtime calibration identity differs from seal")
    return dict(resolved)


def _validate_calibration_source_binding(
    cfg: Mapping[str, Any],
    *,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str | None = None,
    source_remote_head: str | None = None,
) -> dict[str, Any]:
    try:
        calibration = validate_relevance_calibration_artifact(
            _calibration_directory(cfg),
            expected_identity=dict(
                dict(cfg.get("sentence_memory", {}) or {}).get(
                    "resolved_relevance_calibration_identity", {}
                )
                or {}
            ).get("artifact_identity"),
        )
        return validate_relevance_calibration_source(
            calibration,
            source_root=source_root,
            expected_git_head=source_git_head,
            expected_remote_ref=source_remote_ref,
            expected_remote_head=source_remote_head,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise OrderedDecisionError(
            "Relevance calibration source binding is invalid"
        ) from error


def _expected_raw_config(stage: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    base_path = SOURCE_ROOT / (
        "NIAF/continuous_trajectory_field/configs/"
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1.yaml"
    )
    expected = load_config(base_path)
    expected["experiment_name"] = EXPERIMENTS[STAGE1]
    expected.setdefault("sentence_memory", {}).update(
        {
            "key_value_mode": "factorized_metadata_motion_v1",
            "candidate_value_mode": "centered_candidate_covariance_v1",
            "relevance_gate_mode": "frozen_absolute_adjusted_score_v1",
            "association_mode": "none",
            "temporal_prior_mode": "none",
            "temporal_prior_sigma": 0.25,
            "temporal_prior_scale": 1.0,
            "relevance_calibration": copy.deepcopy(
                dict(cfg.get("sentence_memory", {}) or {}).get(
                    "relevance_calibration", {}
                )
            ),
        }
    )
    expected.setdefault("eval", {}).update(
        {
            "sentence_memory_modes": list(STAGE1_EVAL_MODES),
            "export_sentence_memory_mode": "on",
            "evaluation_corruption": {
                "mode": "fixed_evidence_controls_v1",
                "seed": 1234,
                "nonce": None,
            },
        }
    )
    expected.setdefault("train", {}).update(
        {
            "epochs": MAXIMUM_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "early_stopping_min_epochs": MINIMUM_EPOCHS,
        }
    )
    expected.setdefault("output", {})["out_dir"] = (
        "experiments/NIAF/continuous_trajectory_field/" + EXPERIMENTS[STAGE1]
    )
    if stage == STAGE2:
        expected["experiment_name"] = EXPERIMENTS[STAGE2]
        expected["sentence_memory"]["association_mode"] = "absolute_text_motion_v1"
        expected["eval"]["sentence_memory_modes"] = list(STAGE2_EVAL_MODES)
        expected.setdefault("objective", {}).update(
            {
                "lambda_sentence_association_bce": 0.10,
                "lambda_sentence_association_infonce": 0.05,
                "association_temperature": 0.10,
            }
        )
        expected["output"]["out_dir"] = (
            "experiments/NIAF/continuous_trajectory_field/" + EXPERIMENTS[STAGE2]
        )
    return expected


def validate_config(stage: str, config_path: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise OrderedDecisionError(f"Unknown centered stage: {stage}")
    config_path = config_path.resolve()
    if config_path.name != f"{EXPERIMENTS[stage]}.yaml":
        raise OrderedDecisionError("Config filename is not the approved full config")
    cfg = load_config(config_path)
    if cfg != _expected_raw_config(stage, cfg):
        raise OrderedDecisionError(
            "Centered full config differs outside the predeclared protocol"
        )
    memory = dict(cfg.get("sentence_memory", {}) or {})
    if (
        memory.get("candidate_value_mode") != "centered_candidate_covariance_v1"
        or memory.get("relevance_gate_mode") != "frozen_absolute_adjusted_score_v1"
        or memory.get("association_mode")
        != ("none" if stage == STAGE1 else "absolute_text_motion_v1")
        or memory.get("temporal_prior_mode") != "none"
    ):
        raise OrderedDecisionError("Centered architecture assignment changed")
    objective = dict(cfg.get("objective", {}) or {})
    association_fields = {
        "lambda_sentence_association_bce",
        "lambda_sentence_association_infonce",
        "association_temperature",
    }
    if stage == STAGE1 and association_fields.intersection(objective):
        raise OrderedDecisionError("Stage A must not carry dormant association weights")
    if stage == STAGE2 and {
        name: float(objective.get(name, math.nan)) for name in association_fields
    } != {
        "lambda_sentence_association_bce": 0.10,
        "lambda_sentence_association_infonce": 0.05,
        "association_temperature": 0.10,
    }:
        raise OrderedDecisionError("Stage B association objective changed")
    if (
        tuple(dict(cfg.get("eval", {}) or {}).get("sentence_memory_modes", ()))
        != (EVAL_MODES[stage])
    ):
        raise OrderedDecisionError("Centered evaluation mode order changed")
    resolved_cfg = copy.deepcopy(cfg)
    _validate_and_resolve_calibration(resolved_cfg)
    return resolved_cfg


def _resolved_config_projection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    payload = _legacy_resolved_config_projection(cfg)
    memory = payload.get("sentence_memory")
    if isinstance(memory, dict):
        # The approved config is resolved with the same external artifact, so
        # these remain in the projection and must compare exactly.
        required = {
            "relevance_slope",
            "relevance_intercept",
            "resolved_relevance_calibration_identity",
        }
        if bool(required.intersection(memory)) and not required.issubset(memory):
            raise OrderedDecisionError("Partial runtime calibration resolution")
    return payload


def _validated_runtime_v2_parity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact runtime parity proof required for selection replay."""

    safety = cfg.get("sentence_memory_safety")
    if not isinstance(safety, Mapping):
        raise OrderedDecisionError("Resolved config lacks sentence-memory safety")
    value = safety.get("v2_to_v3_text_only_parity")
    if not isinstance(value, Mapping) or set(value) != {
        "prediction_max_abs",
        "duration_max_abs",
        "tolerance",
        "passed",
    }:
        raise OrderedDecisionError("Resolved config lacks an exact v2 parity proof")
    if value.get("passed") is not True:
        raise OrderedDecisionError("Resolved v2 parity proof did not pass")

    numeric: dict[str, float] = {}
    for name in ("prediction_max_abs", "duration_max_abs", "tolerance"):
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise OrderedDecisionError(f"Resolved v2 parity {name} is not numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise OrderedDecisionError(f"Resolved v2 parity {name} is non-finite")
        numeric[name] = number
    if numeric["tolerance"] != 1e-7:
        raise OrderedDecisionError("Resolved v2 parity tolerance changed")
    if any(
        numeric[name] < 0.0 or numeric[name] > numeric["tolerance"]
        for name in ("prediction_max_abs", "duration_max_abs")
    ):
        raise OrderedDecisionError("Resolved v2 parity exceeds its strict tolerance")
    return {
        "prediction_max_abs": numeric["prediction_max_abs"],
        "duration_max_abs": numeric["duration_max_abs"],
        "tolerance": numeric["tolerance"],
        "passed": True,
    }


def _validated_nonnegative_max_abs(value: Any, *, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrderedDecisionError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= float(maximum):
        raise OrderedDecisionError(
            f"{name} is outside the nonnegative bound [0, {maximum}]"
        )
    return number


def _selection_replay_config(
    approved: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    """Add only the independently validated runtime proof to the leaf config."""

    replay = copy.deepcopy(dict(approved))
    safety = replay.get("sentence_memory_safety")
    if not isinstance(safety, dict):
        raise OrderedDecisionError("Approved config lacks sentence-memory safety")
    if "v2_to_v3_text_only_parity" in safety:
        raise OrderedDecisionError("Approved config embeds a runtime v2 parity proof")
    safety["v2_to_v3_text_only_parity"] = _validated_runtime_v2_parity(resolved)
    return replay


def _validated_checkpoint_v2_parity(
    checkpoint: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    parity = _validated_runtime_v2_parity(
        {
            "sentence_memory_safety": {
                "v2_to_v3_text_only_parity": checkpoint.get("v2_to_v3_text_only_parity")
            }
        }
    )
    if parity != _validated_runtime_v2_parity(resolved):
        raise OrderedDecisionError(
            "Checkpoint v2 parity differs from the resolved runtime proof"
        )
    return parity


def _validate_resolved_config(
    *, stage: str, resolved: Mapping[str, Any], approved: Mapping[str, Any]
) -> None:
    if _resolved_config_projection(resolved) != _resolved_config_projection(approved):
        raise OrderedDecisionError(
            "Resolved centered config differs from the approved full config"
        )
    if str(resolved.get("device", "")).lower() != "cuda":
        raise OrderedDecisionError("Full centered stage did not run on CUDA")
    if (
        Path(str(dict(resolved.get("output", {}) or {}).get("out_dir", ""))).name
        != EXPERIMENTS[stage]
    ):
        raise OrderedDecisionError("Resolved centered output path changed")
    partition = dict(resolved.get("validation_text_partition", {}) or {})
    required = {
        "partition_digest": legacy.EXPECTED_PARTITION_DIGEST,
        "partition_artifact_identity": legacy.EXPECTED_PARTITION_ARTIFACT_IDENTITY,
        "expected_development_manifest_sha256": (
            legacy.EXPECTED_DEVELOPMENT_MANIFEST_SHA256
        ),
        "development_row_count": legacy.EXPECTED_DEVELOPMENT_ROWS,
        "development_text_count": legacy.EXPECTED_DEVELOPMENT_TEXTS,
        "confirmation_evaluated_during_training": False,
        "exact_seen_evaluated_during_training": False,
        "retrieval_query_mode": "exact_name_indexed_full_val_table_subset_v1",
    }
    if any(partition.get(name) != value for name, value in required.items()):
        raise OrderedDecisionError("Resolved sealed-development controls changed")
    forbidden = {
        "resolved_artifact",
        "assignments",
        "confirmation_manifest",
        "confirmation_normalized_texts",
        "development_normalized_texts",
        "partition_dir",
    }
    if forbidden.intersection(partition):
        raise OrderedDecisionError("Resolved config embeds sealed holdout membership")
    _validated_runtime_v2_parity(resolved)


def _checkpoint_identity_evidence(
    checkpoint: Mapping[str, Any],
    *,
    stage: str,
    approved_cfg: Mapping[str, Any],
    resolved_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    checkpoint_cfg = dict(checkpoint.get("config", {}) or {})
    if checkpoint_cfg != dict(resolved_cfg):
        raise OrderedDecisionError("Checkpoint config differs from terminal config")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch not in range(1, MAXIMUM_EPOCHS + 1):
        raise OrderedDecisionError("Checkpoint epoch is outside [1, 6]")
    memory = dict(checkpoint_cfg.get("sentence_memory", {}) or {})
    expected_calibration = dict(
        dict(approved_cfg.get("sentence_memory", {}) or {}).get(
            "resolved_relevance_calibration_identity", {}
        )
        or {}
    )
    if (
        memory.get("candidate_value_mode") != "centered_candidate_covariance_v1"
        or memory.get("association_mode")
        != ("none" if stage == STAGE1 else "absolute_text_motion_v1")
        or dict(memory.get("resolved_relevance_calibration_identity", {}) or {})
        != expected_calibration
    ):
        raise OrderedDecisionError("Checkpoint architecture/calibration changed")
    parity = _validated_checkpoint_v2_parity(checkpoint, resolved_cfg)
    partition = dict(checkpoint_cfg.get("validation_text_partition", {}) or {})
    if (
        partition.get("partition_digest") != legacy.EXPECTED_PARTITION_DIGEST
        or partition.get("confirmation_evaluated_during_training") is not False
    ):
        raise OrderedDecisionError("Checkpoint is not bound to sealed development only")
    rng = dict(checkpoint.get("rng_state", {}) or {})
    rank_states = list(rng.get("rank_states", []) or [])
    if int(rng.get("world_size", -1)) != 4 or sorted(
        int(row.get("rank", -1)) for row in rank_states if isinstance(row, Mapping)
    ) != [0, 1, 2, 3]:
        raise OrderedDecisionError("Checkpoint lacks exact four-rank RNG state")

    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    identities: dict[str, Any] = {}
    for name in legacy.IDENTITY_NAMES:
        value = checkpoint.get(f"sentence_memory_{name}_identity")
        identities[name] = {
            "digest": legacy._identity_digest(value, name),
            "value": value,
        }
    expected_functions = {
        "architecture": trainer.sentence_memory_architecture_identity,
        "behavior": trainer.sentence_memory_behavior_identity,
        "objective": trainer.sentence_memory_objective_identity,
        "evaluation_control": trainer.sentence_memory_evaluation_control_identity,
        "selection_aggregation": trainer.sentence_memory_selection_aggregation_identity,
    }
    for name, function in expected_functions.items():
        if identities[name]["value"] != function(approved_cfg):
            raise OrderedDecisionError(f"Checkpoint {name} identity changed")
    if identities["resume"]["value"] != trainer.sentence_memory_resume_identity(
        resolved_cfg
    ):
        raise OrderedDecisionError("Checkpoint resume identity changed")
    expected_map = trainer.sentence_memory_validation_corruption_map_identity(
        approved_cfg,
        partition_digest=legacy.EXPECTED_PARTITION_DIGEST,
        bank_id=EXPECTED_BANK_ID,
    )
    if identities["validation_corruption_map"]["value"] != expected_map:
        raise OrderedDecisionError("Checkpoint fixed-control map identity changed")
    memory_identity = dict(checkpoint.get("sentence_memory_identity", {}) or {})
    tables = dict(memory_identity.get("neighbor_tables", {}) or {})
    hashes = {
        split: str(dict(tables.get(split, {}) or {}).get("sha256", ""))
        for split in ("train", "val")
    }
    if (
        memory_identity.get("bank_id") != EXPECTED_BANK_ID
        or set(tables) != {"train", "val"}
        or hashes != legacy.EXPECTED_NEIGHBOR_SHA256
    ):
        raise OrderedDecisionError("Checkpoint bank/train-val identity changed")
    for name, tensor in dict(checkpoint.get("model", {}) or {}).items():
        if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
            raise OrderedDecisionError(f"Checkpoint tensor is non-finite: {name}")
    return {
        "epoch": epoch,
        "global_step": int(checkpoint.get("global_step", -1)),
        "parity": parity,
        "identities": identities,
        "bank_id": EXPECTED_BANK_ID,
        "neighbor_table_sha256": hashes,
        "relevance_calibration": expected_calibration,
    }


def _replay_selection(
    rows: list[dict[str, Any]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    # Selection consumes the fresh-v2 parity proof injected only after model
    # construction.  Refuse raw leaf configs so a replay cannot silently turn
    # the two parity gates into NaN violations.
    _validated_runtime_v2_parity(cfg)
    early = trainer.initial_early_stopping_state()
    best_feasible_score = math.inf
    best_feasible_row = None
    best_infeasible_score = math.inf
    best_infeasible_key = None
    best_infeasible_row = None
    selection_states: dict[int, dict[str, Any]] = {}
    details_by_epoch: dict[int, Any] = {}
    for index, row in enumerate(rows):
        epoch = int(row.get("epoch", -1))
        validation = legacy._selection_inputs_from_persisted_row(row)
        score, violation, feasible, details = trainer.checkpoint_selection_diagnostics(
            validation, cfg, return_details=True
        )
        if (
            float(row.get("selection_score", math.nan)) != float(score)
            or float(row.get("selection_constraint_violation", math.nan))
            != float(violation)
            or bool(float(row.get("selection_feasible", 0.0))) != bool(feasible)
        ):
            raise OrderedDecisionError("Persisted selection diagnostics changed")
        early, improved, stopped = trainer.update_early_stopping_state(
            early,
            feasible=feasible,
            normalized_constraint_violation=violation,
            selection_score_value=score,
            epoch=epoch,
            patience=PATIENCE,
            minimum_epoch=MINIMUM_EPOCHS,
        )
        expected_row_state = {
            "early_stopping_improved": float(improved),
            "early_stopping_bad_validation_count": float(early["bad_validation_count"]),
            "early_stopping_validation_count": float(early["validation_count"]),
            "early_stopping_requested": float(stopped),
        }
        if any(
            float(row.get(name, math.nan)) != value
            for name, value in expected_row_state.items()
        ):
            raise OrderedDecisionError("Early-stopping state differs from replay")
        if feasible and score < best_feasible_score:
            best_feasible_score = float(score)
            best_feasible_row = row
        elif not feasible:
            key = (float(violation), float(score))
            if best_infeasible_key is None or key < best_infeasible_key:
                best_infeasible_key = key
                best_infeasible_score = float(score)
                best_infeasible_row = row
        selection_states[epoch] = trainer.checkpoint_selection_state(
            best_feasible_score,
            best_infeasible_score,
            early_stopping_state=early,
            best_infeasible_key=best_infeasible_key,
        )
        details_by_epoch[epoch] = details
        if stopped and index != len(rows) - 1:
            raise OrderedDecisionError("Metrics continue after replayed stop")
    return {
        "best_feasible_row": best_feasible_row,
        "best_infeasible_row": best_infeasible_row,
        "best_feasible_score": (
            None if not math.isfinite(best_feasible_score) else best_feasible_score
        ),
        "best_infeasible_score": (
            None if not math.isfinite(best_infeasible_score) else best_infeasible_score
        ),
        "best_infeasible_key": (
            None if best_infeasible_key is None else list(best_infeasible_key)
        ),
        "early_stopping": early,
        "selection_states": selection_states,
        "details_by_epoch": details_by_epoch,
    }


def _validate_resume_history(
    run_dir: Path,
    launch: Mapping[str, Any],
    *,
    stage: str,
    config_path: Path,
    stage1_authorization: Path | None,
) -> list[dict[str, Any]]:
    root = Path(f"{run_dir.resolve()}.prerequisites") / "resume_attempts"
    if not root.exists():
        return []
    evidence = []
    for path in sorted(root.glob("*.json")):
        value = legacy._json(path)
        payload = {
            key: child for key, child in value.items() if key != "resume_identity"
        }
        original = dict(value.get("original_run_launch", {}) or {})
        source = dict(value.get("source", {}) or {})
        config = dict(value.get("config", {}) or {})
        checkpoint = dict(value.get("last_checkpoint", {}) or {})
        progress = dict(value.get("resume_progress", {}) or {})
        reconciliation = dict(value.get("metrics_reconciliation", {}) or {})
        if (
            value.get("schema_name") != RESUME_ATTEMPT_SCHEMA_NAME
            or value.get("stage") != stage
            or value.get("experiment_name") != EXPERIMENTS[stage]
            or value.get("recorded_before_resume") is not True
            or value.get("wandb") != "disabled"
            or value.get("staged_neighbor_splits") != ["train", "val"]
            or value.get("test_data_accessed") is not False
            or value.get("confirmation_manifest_opened") is not False
            or legacy._digest_json(payload) != value.get("resume_identity")
        ):
            raise OrderedDecisionError("Resume-attempt evidence is invalid")
        launch_path = Path(f"{run_dir.resolve()}.prerequisites") / (
            "run_launch_identity.json"
        )
        if (
            Path(str(original.get("path", ""))).resolve() != launch_path.resolve()
            or original.get("sha256") != sha256_file(launch_path)
            or original.get("launch_identity") != launch.get("launch_identity")
            or source != dict(launch.get("source", {}) or {})
            or dict(value.get("resume_source_check", {}) or {}) != source
            or Path(str(config.get("path", ""))).resolve() != config_path.resolve()
            or config.get("sha256") != sha256_file(config_path)
            or Path(str(checkpoint.get("path", ""))).resolve()
            != (run_dir / "checkpoints" / "last.pt").resolve()
            or int(checkpoint.get("epoch", -1)) not in range(1, MAXIMUM_EPOCHS + 1)
            or checkpoint.get("optimizer_state_present") is not True
            or checkpoint.get("optimizer_reset") is not False
            or progress.get("mode")
            not in {"validation_pending", "epoch_complete", "terminal_complete"}
            or int(progress.get("checkpoint_epoch", -1))
            != int(checkpoint.get("epoch", -2))
            or progress.get("checkpoint_metrics_sha256")
            != legacy._digest_json(dict(progress.get("checkpoint_metrics", {}) or {}))
            or progress.get("selection_state_sha256")
            != legacy._digest_json(dict(progress.get("selection_state", {}) or {}))
            or reconciliation.get("performed") not in {True, False}
            or value.get("relevance_calibration")
            != checkpoint.get("identity_evidence", {}).get("relevance_calibration")
        ):
            raise OrderedDecisionError("Resume attempt changed its exact run state")
        terminal = value.get("terminal_finalization")
        if progress.get("mode") == "terminal_complete":
            if not isinstance(terminal, Mapping):
                raise OrderedDecisionError("Terminal resume lacks finalization")
            summary_path = Path(str(terminal.get("selection_summary_path", "")))
            summary = dict(terminal.get("selection_summary_payload", {}) or {})
            selected = dict(terminal.get("selected_checkpoint", {}) or {})
            if (
                terminal.get("required") is not True
                or terminal.get("trainer_reentry") is not False
                or summary_path.resolve()
                != (run_dir / "selection_summary.json").resolve()
                or summary_path.is_file() is not True
                or legacy._json(summary_path) != summary
                or legacy._digest_json(summary)
                != terminal.get("selection_summary_payload_sha256")
                or sha256_file(Path(str(selected.get("path", ""))))
                != selected.get("sha256")
            ):
                raise OrderedDecisionError("Terminal resume finalization changed")
        elif terminal is not None:
            raise OrderedDecisionError("Nonterminal resume has terminal evidence")
        rng = dict(progress.get("rng_state", {}) or {})
        if int(rng.get("world_size", -1)) != 4 or [
            int(row.get("rank", -1)) for row in list(rng.get("rank_states", []) or [])
        ] != [0, 1, 2, 3]:
            raise OrderedDecisionError("Resume attempt lost four-rank RNG evidence")
        for script in value.get("resume_launchers", []):
            script = dict(script)
            if sha256_file(Path(str(script.get("path", "")))) != script.get("sha256"):
                raise OrderedDecisionError("A centered resume launcher changed")
        predecessor = value.get("stage1_authorization")
        if stage == STAGE2:
            if stage1_authorization is None or not isinstance(predecessor, Mapping):
                raise OrderedDecisionError("Stage-B resume lost Stage-A authorization")
            authorization = verify_authorization(
                stage1_authorization, purpose="stage2", stage=STAGE1
            )
            if (
                Path(str(predecessor.get("path", ""))).resolve()
                != stage1_authorization.resolve()
                or predecessor.get("sha256") != sha256_file(stage1_authorization)
                or predecessor.get("authorization_identity")
                != authorization.get("authorization_identity")
            ):
                raise OrderedDecisionError("Stage-B resume predecessor changed")
        elif predecessor is not None:
            raise OrderedDecisionError("Stage-A resume has predecessor evidence")
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "resume_identity": value["resume_identity"],
                "job_id": str(dict(value.get("slurm", {}) or {}).get("job_id", "")),
                "checkpoint_epoch_before_resume": int(checkpoint["epoch"]),
                "checkpoint_sha256_before_resume": checkpoint["sha256"],
            }
        )
    return evidence


_EXACT_ZERO_AUDITS = (
    "uniform_final_vs_off_prediction_max_abs",
    "uniform_final_vs_off_duration_max_abs",
    "broadcast_complete_vs_off_prediction_max_abs",
    "broadcast_complete_vs_off_duration_max_abs",
    "all_null_vs_off_prediction_max_abs",
    "all_null_vs_off_duration_max_abs",
    "all_null_gate_max_abs",
    "all_null_candidate_mass_max_abs",
    "all_null_one_minus_null_mass_max_abs",
)


def _validate_selected_integrity_audit(
    path: Path,
    *,
    stage: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    cfg: Mapping[str, Any],
    launch_path: Path,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a standalone complete-dev replay of the selected checkpoint."""

    path = path.resolve()
    result = legacy._json(path)
    legacy._finite_tree(result, path="selected_integrity_audit")
    if (
        result.get("split") != "val"
        or int(result.get("dataset_examples", -1)) != legacy.EXPECTED_DEVELOPMENT_ROWS
        or int(result.get("max_batches_per_rank", -1)) != 0
        or result.get("sentence_memory_mode") != "auto"
        or int(result.get("checkpoint_epoch", -1)) != int(checkpoint.get("epoch", -2))
        or int(result.get("checkpoint_global_step", -1))
        != int(checkpoint.get("global_step", -2))
    ):
        raise OrderedDecisionError(
            "Selected-checkpoint audit is not a complete sealed-dev auto replay"
        )
    metrics = dict(result.get("metrics", {}) or {})
    namespaces = {
        "off": "text_only",
        "on": "sentence_memory",
        "motion_shuffled_n0": "motion_shuffled_n0_sentence_memory",
        "motion_shuffled_n1": "motion_shuffled_n1_sentence_memory",
        "motion_shuffled_n2": "motion_shuffled_n2_sentence_memory",
        "cross_query_motion": "cross_query_motion_sentence_memory",
        "full_replacement": "full_replacement_sentence_memory",
        "joint_tuple_permuted": "joint_tuple_permuted_sentence_memory",
        "uniform_final_mass": "uniform_final_mass_sentence_memory",
        "analytic_prior": "analytic_prior_sentence_memory",
        "association_disabled": "association_disabled_sentence_memory",
    }
    for mode in EVAL_MODES[stage]:
        prefix = f"{namespaces[mode]}/"
        if not any(str(name).startswith(prefix) for name in metrics):
            raise OrderedDecisionError(
                f"Selected-checkpoint audit lacks mode namespace {mode}"
            )
    for name in _EXACT_ZERO_AUDITS:
        value = float(metrics.get(f"paired_sentence_memory/{name}", math.nan))
        if value != 0.0:
            raise OrderedDecisionError(
                f"Selected-checkpoint exact integrity audit failed: {name}={value}"
            )
    joint = {
        name: _validated_nonnegative_max_abs(
            metrics.get(f"paired_sentence_memory/{name}"),
            name=f"selected_integrity_audit.{name}",
            maximum=1e-7,
        )
        for name in (
            "joint_tuple_prediction_max_abs",
            "joint_tuple_duration_max_abs",
        )
    }

    provenance = dict(result.get("centered_evaluation_provenance", {}) or {})
    without_identity = {
        name: value for name, value in provenance.items() if name != "identity"
    }
    if (
        provenance.get("schema_name")
        != "signtrajfield_centered_public_evaluation_provenance"
        or int(provenance.get("schema_version", -1)) != 1
        or legacy._digest_json(without_identity) != provenance.get("identity")
        or provenance.get("test_data_accessed") is not False
        or provenance.get("confirmation_manifest_opened") is not False
    ):
        raise OrderedDecisionError(
            "Selected-checkpoint evaluator provenance is invalid"
        )
    source = dict(provenance.get("source", {}) or {})
    launch_source = dict(launch.get("source", {}) or {})
    if source != {
        "git_head": str(launch_source.get("git_head", "")).lower(),
        "run_launch_identity": launch.get("launch_identity"),
        "run_launch_path": str(launch_path.resolve()),
        "run_launch_sha256": sha256_file(launch_path),
    }:
        raise OrderedDecisionError("Selected-checkpoint audit source is detached")
    config_path = config_path.resolve()
    if dict(provenance.get("config", {}) or {}) != {
        "path": str(config_path.resolve()),
        "sha256": sha256_file(config_path),
    }:
        raise OrderedDecisionError("Selected-checkpoint audit config is detached")
    if dict(provenance.get("checkpoint", {}) or {}) != {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
    }:
        raise OrderedDecisionError("Selected-checkpoint audit checkpoint is detached")

    runtime = dict(provenance.get("development_validation_runtime", {}) or {})
    runtime_without_identity = {
        name: value for name, value in runtime.items() if name != "runtime_identity"
    }
    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    if (
        runtime.get("schema_name") != "signtrajfield_development_validation_runtime"
        or runtime.get("validation_only") is not True
        or runtime.get("runtime_identity")
        != legacy._digest_json(runtime_without_identity)
        or runtime.get("runtime_identity")
        != dict(partition_cfg.get("development_runtime_identity", {}) or {}).get(
            "digest"
        )
        or dict(runtime.get("development_manifest", {}) or {}).get("sha256")
        != legacy.EXPECTED_DEVELOPMENT_MANIFEST_SHA256
        or int(dict(runtime.get("development_manifest", {}) or {}).get("row_count", -1))
        != legacy.EXPECTED_DEVELOPMENT_ROWS
    ):
        raise OrderedDecisionError("Selected-checkpoint audit dev runtime changed")
    query = dict(provenance.get("development_validation_query_binding", {}) or {})
    if (
        query.get("split") != "val"
        or query.get("manifest_sha256") != legacy.EXPECTED_DEVELOPMENT_MANIFEST_SHA256
        or query.get("neighbor_lookup_mode") != "exact_name_indexed_parent_subset_v1"
        or query.get("parent_query_manifest_sha256")
        != legacy.EXPECTED_VALIDATION_MANIFEST_SHA256
        or int(query.get("parent_query_count", -1)) != 1_077
    ):
        raise OrderedDecisionError("Selected-checkpoint audit query binding changed")

    expected_calibration = dict(
        dict(cfg.get("sentence_memory", {}) or {}).get(
            "resolved_relevance_calibration_identity", {}
        )
        or {}
    )
    if (
        result.get("sentence_memory_relevance_calibration_identity")
        != expected_calibration
        or provenance.get("relevance_calibration_identity") != expected_calibration
    ):
        raise OrderedDecisionError("Selected-checkpoint audit calibration changed")
    expected_identities = {
        name: checkpoint.get(f"sentence_memory_{checkpoint_name}_identity")
        for name, checkpoint_name in {
            "objective": "objective",
            "architecture": "architecture",
            "evaluation_control": "evaluation_control",
            "selection_aggregation": "selection_aggregation",
            "validation_control_map": "validation_corruption_map",
        }.items()
    }
    if dict(provenance.get("identities", {}) or {}) != expected_identities:
        raise OrderedDecisionError("Selected-checkpoint audit identities changed")
    parity = dict(
        dict(result.get("selection_details", {}) or {})
        .get("dual_mode", {})
        .get("v2_text_only_parity", {})
        or {}
    )
    checkpoint_parity = dict(checkpoint.get("v2_to_v3_text_only_parity", {}) or {})
    for name in ("prediction_max_abs", "duration_max_abs"):
        audit_value = _validated_nonnegative_max_abs(
            parity.get(name),
            name=f"selected_integrity_audit.v2_{name}",
            maximum=1e-7,
        )
        if audit_value != float(checkpoint_parity.get(name, math.inf)):
            raise OrderedDecisionError("Selected-checkpoint audit v2 parity changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "provenance_identity": provenance["identity"],
        "exact_zero": {name: 0.0 for name in _EXACT_ZERO_AUDITS},
        "joint_tuple": joint,
        "v2_text_only_parity": parity,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }


def _validate_run(
    *, stage: str, run_dir: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = run_dir.resolve()
    if run_dir.name != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Run directory has the wrong centered stage name")
    cfg = validate_config(stage, config_path)
    resolved_path = run_dir / "config.resolved.json"
    resolved = legacy._json(resolved_path)
    _validate_resolved_config(stage=stage, resolved=resolved, approved=cfg)
    metrics_path = run_dir / "metrics.jsonl"
    rows = legacy._read_metrics(metrics_path)
    completed = [
        row
        for row in rows
        if float(row.get("validation_pending", 1.0)) == 0.0
        and "selection_feasible" in row
    ]
    if (
        len(completed) != len(rows)
        or not MINIMUM_EPOCHS <= len(completed) <= MAXIMUM_EPOCHS
    ):
        raise OrderedDecisionError("Run lacks three-to-six complete validations")
    epochs = [int(row.get("epoch", -1)) for row in completed]
    if epochs != list(range(1, max(epochs) + 1)):
        raise OrderedDecisionError("Run epoch history is incomplete")
    namespaces = {
        "off": "text_only",
        "on": "sentence_memory",
        "motion_shuffled_n0": "motion_shuffled_n0_sentence_memory",
        "motion_shuffled_n1": "motion_shuffled_n1_sentence_memory",
        "motion_shuffled_n2": "motion_shuffled_n2_sentence_memory",
        "cross_query_motion": "cross_query_motion_sentence_memory",
        "full_replacement": "full_replacement_sentence_memory",
        "joint_tuple_permuted": "joint_tuple_permuted_sentence_memory",
        "uniform_final_mass": "uniform_final_mass_sentence_memory",
        "analytic_prior": "analytic_prior_sentence_memory",
        "association_disabled": "association_disabled_sentence_memory",
    }
    required_prefixes = tuple(f"val_{namespaces[mode]}/" for mode in EVAL_MODES[stage])
    for row in completed:
        legacy._finite_tree(row, path=f"metrics.epoch_{row.get('epoch')}")
        if any(
            not any(str(key).startswith(prefix) for key in row)
            for prefix in required_prefixes
        ):
            raise OrderedDecisionError("Validation epoch lacks a centered control mode")
    replay_cfg = _selection_replay_config(cfg, resolved)
    replay = _replay_selection(completed, replay_cfg)
    summary = legacy._json(run_dir / "selection_summary.json")
    has_feasible = replay["best_feasible_row"] is not None
    expected_summary = {
        "has_feasible_checkpoint": has_feasible,
        "best_feasible_score": replay["best_feasible_score"],
        "best_infeasible_score": replay["best_infeasible_score"],
        "required": True,
        "best_infeasible_key": replay["best_infeasible_key"],
        "early_stopping": replay["early_stopping"],
    }
    if summary != expected_summary:
        raise OrderedDecisionError("Selection summary differs from replay")
    stopped = bool(replay["early_stopping"].get("stopped", False))
    if not stopped and max(epochs) != MAXIMUM_EPOCHS:
        raise OrderedDecisionError("Run neither early-stopped nor reached epoch six")
    checkpoint_name = "best.pt" if has_feasible else "best_infeasible.pt"
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    checkpoint = legacy._load_checkpoint(checkpoint_path)
    evidence = _checkpoint_identity_evidence(
        checkpoint,
        stage=stage,
        approved_cfg=cfg,
        resolved_cfg=resolved,
    )
    selected_row = (
        replay["best_feasible_row"] if has_feasible else replay["best_infeasible_row"]
    )
    if (
        not isinstance(selected_row, Mapping)
        or dict(checkpoint.get("metrics", {}) or {}) != dict(selected_row)
        or dict(checkpoint.get("selection_state", {}) or {})
        != replay["selection_states"][int(selected_row["epoch"])]
    ):
        raise OrderedDecisionError("Selected checkpoint is not replayed winner")
    last = legacy._load_checkpoint(run_dir / "checkpoints" / "last.pt")
    if (
        int(last.get("epoch", -1)) != max(epochs)
        or dict(last.get("metrics", {}) or {}) != completed[-1]
        or dict(last.get("selection_state", {}) or {})
        != replay["selection_states"][max(epochs)]
    ):
        raise OrderedDecisionError("Terminal last.pt differs from replay")
    launch_path = Path(f"{run_dir}.prerequisites") / "run_launch_identity.json"
    launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=None,
        defer_stage2_predecessor=True,
    )
    if _ACTIVE_INTEGRITY_AUDIT is None:
        raise OrderedDecisionError(
            "Centered decision requires a standalone selected-checkpoint audit"
        )
    integrity_audit = _validate_selected_integrity_audit(
        _ACTIVE_INTEGRITY_AUDIT,
        stage=stage,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        cfg=resolved,
        launch_path=launch_path,
        launch=launch,
    )
    stage1_authorization = None
    if stage == STAGE2:
        predecessor = dict(launch.get("stage1_authorization", {}) or {})
        stage1_authorization = Path(str(predecessor.get("path", ""))).resolve()
    evidence.update(
        {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "metrics_jsonl_sha256": sha256_file(metrics_path),
            "selection_summary_sha256": sha256_file(run_dir / "selection_summary.json"),
            "resolved_config_sha256": sha256_file(resolved_path),
            "terminal_epoch": max(epochs),
            "validation_events": len(completed),
            "scientific_gate": replay["details_by_epoch"][int(selected_row["epoch"])],
            "has_feasible_checkpoint": has_feasible,
            "resume_attempts": _validate_resume_history(
                run_dir,
                launch,
                stage=stage,
                config_path=config_path,
                stage1_authorization=stage1_authorization,
            ),
            "selected_checkpoint_integrity_audit": integrity_audit,
        }
    )
    return checkpoint, evidence, dict(selected_row)


def _load_dev_export(*args: Any, **kwargs: Any) -> dict[str, Any]:
    directory = Path(args[0] if args else kwargs["directory"]).resolve()
    expected_mode = str(kwargs["expected_mode"])
    checkpoint_path = Path(kwargs["checkpoint_path"]).resolve()
    config_path = Path(kwargs["config_path"]).resolve()
    partition = kwargs["partition"]
    require_control = bool(kwargs.get("require_factorized_control", True))
    summary_path = directory / "export_summary.json"
    summary = legacy._json(summary_path)
    if (
        summary.get("split") != "val"
        or summary.get("length_mode") != "predicted"
        or summary.get("word_prior_mode") != "off"
        or summary.get("sentence_memory_mode") != expected_mode
    ):
        raise OrderedDecisionError("Invalid centered development export controls")
    if require_control:
        if dict(summary.get("sentence_memory_evaluation_corruption", {}) or {}) != (
            EVALUATION_CORRUPTION
        ):
            raise OrderedDecisionError("Centered export control identity changed")
        legacy.validate_factorized_export_query_binding(
            summary.get("sentence_memory_query_binding"),
            expected_authority="config_pinned_development_manifest_v1",
            expected_manifest_file="manifest_development.jsonl",
            expected_manifest_sha256=partition["development_manifest_sha256"],
            expected_query_rows=legacy.EXPECTED_DEVELOPMENT_ROWS,
        )
        stage = STAGE1 if config_path.stem == EXPERIMENTS[STAGE1] else STAGE2
        cfg = validate_config(stage, config_path)
        _validate_centered_export_calibration_identity(
            summary,
            expected=cfg["sentence_memory"][
                "resolved_relevance_calibration_identity"
            ],
        )
    if (
        legacy._resolve_existing(summary.get("checkpoint"), directory)
        != checkpoint_path
    ):
        raise OrderedDecisionError("Development export used another checkpoint")
    if legacy._resolve_existing(summary.get("config"), directory) != config_path:
        raise OrderedDecisionError("Development export used another config")
    manifest = dict(summary.get("manifest", {}) or {})
    authority = (
        "config_pinned_development_manifest_v1"
        if require_control
        else "explicit_isolated_development_environment_v1"
    )
    if (
        manifest.get("canonical_source_manifest") is not None
        or manifest.get("canonical_sample_count") is not None
        or manifest.get("is_complete_canonical_manifest") is not None
        or manifest.get("is_canonical_order") is not None
        or manifest.get("canonical_manifest_inspection")
        != "forbidden_isolated_explicit_manifest_v1"
        or manifest.get("query_manifest_authority") != authority
        or manifest.get("dataset_manifest_sha256")
        != partition["development_manifest_sha256"]
        or manifest.get("sealed_partition_artifact_identity")
        != legacy.EXPECTED_PARTITION_ARTIFACT_IDENTITY
        or legacy._resolve_existing(manifest.get("dataset_manifest"), directory)
        != Path(partition["development_manifest"]).resolve()
    ):
        raise OrderedDecisionError("Development export manifest evidence changed")
    output_manifest = legacy._resolve_existing(
        manifest.get("output_manifest"), directory
    )
    if sha256_file(output_manifest) != partition["development_manifest_sha256"]:
        raise OrderedDecisionError("Development export opened the wrong rows")
    rows = list(summary.get("rows", []) or [])
    if len(rows) != legacy.EXPECTED_DEVELOPMENT_ROWS:
        raise OrderedDecisionError("Development export is incomplete")
    identities = [
        (str(row.get("name", "")), legacy.normalize_sentence_text(row.get("text", "")))
        for row in rows
    ]
    expected_identities = [
        (str(row.get("name", "")), legacy.normalize_sentence_text(row.get("text", "")))
        for row in partition["rows"]
    ]
    if identities != expected_identities:
        raise OrderedDecisionError("Development export row order changed")
    durations = np.asarray(
        [row.get("predicted_duration_seconds") for row in rows], dtype=np.float64
    )
    if not bool(np.isfinite(durations).all()):
        raise OrderedDecisionError("Development export durations are non-finite")
    return {
        "dir": directory,
        "summary": summary,
        "summary_path": summary_path,
        "rows": rows,
        "durations": durations,
    }


def _validate_centered_export_calibration_identity(
    summary: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require both centered export identity fields to match exactly."""

    expected = dict(expected)
    top_level = summary.get("sentence_memory_relevance_calibration_identity")
    checkpoint_identities = summary.get("sentence_memory_checkpoint_identities")
    nested = (
        checkpoint_identities.get("relevance_calibration")
        if isinstance(checkpoint_identities, Mapping)
        else None
    )
    if top_level != expected or nested != expected or top_level != nested:
        raise OrderedDecisionError("Centered export calibration identity changed")
    return expected


def _augment_calibration_identity(
    payload: dict[str, Any], identity_key: str
) -> dict[str, Any]:
    config = dict(payload.get("config", {}) or payload.get("stage2_config", {}) or {})
    cfg_path = Path(str(config.get("path", "")))
    stage = STAGE1 if cfg_path.stem == EXPERIMENTS[STAGE1] else STAGE2
    resolved = validate_config(stage, cfg_path)
    calibration = dict(
        resolved["sentence_memory"]["resolved_relevance_calibration_identity"]
    )
    launch_source = dict(payload.get("source", {}) or {})
    _validate_calibration_source_binding(
        resolved,
        source_root=Path(str(launch_source.get("repository_root", ""))),
        source_git_head=str(launch_source.get("git_head", "")),
        source_remote_ref=str(launch_source.get("remote_ref", "")),
        source_remote_head=str(launch_source.get("remote_head", "")),
    )
    payload["relevance_calibration"] = calibration
    without_identity = {
        key: value for key, value in payload.items() if key != identity_key
    }
    payload[identity_key] = legacy._digest_json(without_identity)
    return payload


def record_run_launch(**kwargs: Any) -> dict[str, Any]:
    stage = str(kwargs["stage"])
    config_path = Path(kwargs["config_path"]).resolve()
    v2_checkpoint = Path(kwargs["v2_checkpoint_path"]).resolve()
    source_root = Path(kwargs["source_root"]).resolve()
    source_head = str(kwargs["source_git_head"]).lower()
    remote_ref = str(kwargs["source_remote_ref"])
    remote_head = str(kwargs["source_remote_head"]).lower()
    cfg = validate_config(stage, config_path)
    if sha256_file(v2_checkpoint) != legacy.EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Launch baseline is not pinned v2")
    source = legacy._validate_shared_source_checkout(
        source_root,
        expected_head=source_head,
        expected_remote_ref=remote_ref,
        expected_remote_head=remote_head,
    )
    predecessor = None
    stage1_authorization = kwargs.get("stage1_authorization")
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage-B launch lacks Stage-A authorization")
        authorization = verify_authorization(
            Path(stage1_authorization), purpose="stage2", stage=STAGE1
        )
        legacy._require_stage2_source_match(authorization, source)
        predecessor = {
            "path": str(Path(stage1_authorization).resolve()),
            "sha256": sha256_file(stage1_authorization),
            "authorization_identity": authorization["authorization_identity"],
            "decision_identity": authorization["decision_identity"],
        }
    elif stage1_authorization is not None:
        raise OrderedDecisionError("Stage-A launch cannot have a predecessor")
    launchers = [
        {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for path in kwargs["launcher_paths"]
    ]
    without = {
        "schema_name": LAUNCH_INPUT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "recorded_before_training": True,
        "source": {
            "git_head": source_head,
            "remote_ref": remote_ref,
            "remote_head": remote_head,
            "remote_ref_exact_match_checked": True,
            "worktree_clean_checked": True,
            "repository_root": source["repository_root"],
            "git_directory": source["git_directory"],
            "durable_experiments_root": source["durable_experiments_root"],
            "frozen_text_model_root": source["frozen_text_model_root"],
            "standalone_shared_clone_checked": True,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "fresh_v2_checkpoint": {
            "path": str(v2_checkpoint),
            "sha256": legacy.EXPECTED_V2_SHA256,
        },
        "launchers": launchers,
        "slurm": {
            "job_id": str(os.environ.get("SLURM_JOB_ID", "")),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
            "tasks": int(os.environ.get("SLURM_NTASKS", "0")),
        },
        "stage1_authorization": predecessor,
        "wandb": "disabled",
        "staged_neighbor_splits": ["train", "val"],
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    if not without["slurm"]["job_id"] or without["slurm"]["nodes"] != 4:
        raise OrderedDecisionError("Centered launch requires four Slurm nodes")
    value = _augment_calibration_identity(without, "launch_identity")
    _validate_calibration_source_binding(
        cfg,
        source_root=source_root,
        source_git_head=source_head,
        source_remote_ref=remote_ref,
        source_remote_head=remote_head,
    )
    return legacy._atomic_json_record(Path(kwargs["out_file"]).resolve(), value)


def record_stage2_input(**kwargs: Any) -> dict[str, Any]:
    authorization_path = Path(kwargs["authorization_path"]).resolve()
    config_path = Path(kwargs["stage2_config_path"]).resolve()
    v2_checkpoint = Path(kwargs["v2_checkpoint_path"]).resolve()
    source_root = Path(kwargs["source_root"]).resolve()
    source_head = str(kwargs["source_git_head"]).lower()
    remote_ref = str(kwargs["source_remote_ref"])
    remote_head = str(kwargs["source_remote_head"]).lower()
    authorization = verify_authorization(
        authorization_path, purpose="stage2", stage=STAGE1
    )
    cfg = validate_config(STAGE2, config_path)
    if sha256_file(v2_checkpoint) != legacy.EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Stage-B input is not pinned to v2")
    source = legacy._validate_shared_source_checkout(
        source_root,
        expected_head=source_head,
        expected_remote_ref=remote_ref,
        expected_remote_head=remote_head,
    )
    legacy._require_stage2_source_match(authorization, source)
    without = {
        "schema_name": STAGE2_INPUT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE2,
        "experiment_name": EXPERIMENTS[STAGE2],
        "recorded_before_training": True,
        "stage1_authorization": {
            "path": str(authorization_path),
            "sha256": sha256_file(authorization_path),
            "authorization_identity": authorization["authorization_identity"],
            "decision_identity": authorization["decision_identity"],
        },
        "stage2_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "fresh_v2_checkpoint": {
            "path": str(v2_checkpoint),
            "sha256": legacy.EXPECTED_V2_SHA256,
        },
        "source": source,
        "resume": False,
        "warm_start": False,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    value = _augment_calibration_identity(without, "input_identity")
    _validate_calibration_source_binding(
        cfg,
        source_root=source_root,
        source_git_head=source_head,
        source_remote_ref=remote_ref,
        source_remote_head=remote_head,
    )
    return legacy._atomic_json_record(Path(kwargs["out_file"]).resolve(), value)


def _validate_run_launch(
    path: Path,
    *,
    stage: str,
    config_path: Path,
    stage1_authorization: Path | None,
    defer_stage2_predecessor: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    value = legacy._json(path)
    without = {
        name: child for name, child in value.items() if name != "launch_identity"
    }
    if (
        value.get("schema_name") != LAUNCH_INPUT_SCHEMA_NAME
        or int(value.get("schema_version", -1)) != SCHEMA_VERSION
        or value.get("stage") != stage
        or value.get("experiment_name") != EXPERIMENTS[stage]
        or value.get("recorded_before_training") is not True
        or value.get("wandb") != "disabled"
        or value.get("staged_neighbor_splits") != ["train", "val"]
        or value.get("test_data_accessed") is not False
        or value.get("confirmation_manifest_opened") is not False
        or legacy._digest_json(without) != value.get("launch_identity")
    ):
        raise OrderedDecisionError("Centered run-launch identity is invalid")
    source = dict(value.get("source", {}) or {})
    if (
        source.get("worktree_clean_checked") is not True
        or source.get("remote_ref_exact_match_checked") is not True
        or source.get("standalone_shared_clone_checked") is not True
        or str(source.get("remote_head", "")).lower()
        != str(source.get("git_head", "")).lower()
    ):
        raise OrderedDecisionError("Centered run-launch source proof is invalid")
    config = dict(value.get("config", {}) or {})
    if Path(
        str(config.get("path", ""))
    ).resolve() != config_path.resolve() or config.get("sha256") != sha256_file(
        config_path
    ):
        raise OrderedDecisionError("Centered run-launch config changed")
    baseline = dict(value.get("fresh_v2_checkpoint", {}) or {})
    if (
        baseline.get("sha256") != legacy.EXPECTED_V2_SHA256
        or sha256_file(Path(str(baseline.get("path", "")))) != legacy.EXPECTED_V2_SHA256
    ):
        raise OrderedDecisionError("Centered run-launch v2 changed")
    if int(dict(value.get("slurm", {}) or {}).get("nodes", -1)) != 4:
        raise OrderedDecisionError("Centered run-launch was not four-node")
    for script in value.get("launchers", []):
        script = dict(script)
        if sha256_file(Path(str(script.get("path", "")))) != script.get("sha256"):
            raise OrderedDecisionError("A centered launch script changed")
    predecessor_value = value.get("stage1_authorization")
    predecessor = stage1_authorization
    if stage == STAGE2 and defer_stage2_predecessor:
        predecessor = Path(str(dict(predecessor_value or {}).get("path", "")))
    if stage == STAGE2:
        if predecessor is None or not isinstance(predecessor_value, Mapping):
            raise OrderedDecisionError("Stage-B run-launch lacks predecessor")
        authorization = verify_authorization(
            predecessor, purpose="stage2", stage=STAGE1
        )
        if (
            Path(str(predecessor_value.get("path", ""))).resolve()
            != predecessor.resolve()
            or predecessor_value.get("sha256") != sha256_file(predecessor)
            or predecessor_value.get("authorization_identity")
            != authorization.get("authorization_identity")
        ):
            raise OrderedDecisionError("Stage-B run-launch predecessor changed")
        legacy._require_stage2_source_match(authorization, source)
    elif predecessor_value is not None:
        raise OrderedDecisionError("Stage-A run-launch has a predecessor")
    resolved_config = validate_config(stage, config_path)
    expected = resolved_config["sentence_memory"][
        "resolved_relevance_calibration_identity"
    ]
    if value.get("relevance_calibration") != expected:
        raise OrderedDecisionError("Run launch calibration identity changed")
    return value


def _validate_stage2_input(
    path: Path,
    *,
    stage1_authorization: Path,
    config_path: Path,
) -> dict[str, Any]:
    path = path.resolve()
    value = legacy._json(path)
    without = {name: child for name, child in value.items() if name != "input_identity"}
    if (
        value.get("schema_name") != STAGE2_INPUT_SCHEMA_NAME
        or value.get("stage") != STAGE2
        or value.get("experiment_name") != EXPERIMENTS[STAGE2]
        or value.get("recorded_before_training") is not True
        or value.get("resume") is not False
        or value.get("warm_start") is not False
        or legacy._digest_json(without) != value.get("input_identity")
    ):
        raise OrderedDecisionError("Stage-B input identity is invalid")
    predecessor = dict(value.get("stage1_authorization", {}) or {})
    if Path(
        str(predecessor.get("path", ""))
    ).resolve() != stage1_authorization.resolve() or predecessor.get(
        "sha256"
    ) != sha256_file(stage1_authorization):
        raise OrderedDecisionError("Stage-B input predecessor changed")
    config = dict(value.get("stage2_config", {}) or {})
    if Path(
        str(config.get("path", ""))
    ).resolve() != config_path.resolve() or config.get("sha256") != sha256_file(
        config_path
    ):
        raise OrderedDecisionError("Stage-B input config changed")
    baseline = dict(value.get("fresh_v2_checkpoint", {}) or {})
    if (
        baseline.get("sha256") != legacy.EXPECTED_V2_SHA256
        or sha256_file(Path(str(baseline.get("path", "")))) != legacy.EXPECTED_V2_SHA256
    ):
        raise OrderedDecisionError("Stage-B input is not fresh pinned v2")
    authorization = verify_authorization(
        stage1_authorization, purpose="stage2", stage=STAGE1
    )
    legacy._require_stage2_source_match(
        authorization, dict(value.get("source", {}) or {})
    )
    expected = validate_config(STAGE2, config_path)["sentence_memory"][
        "resolved_relevance_calibration_identity"
    ]
    if value.get("relevance_calibration") != expected:
        raise OrderedDecisionError("Stage-B launch input calibration identity changed")
    return value


def _centered_resume_progress_evidence(
    checkpoint: Mapping[str, Any],
    metrics: list[dict[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    epoch = int(checkpoint.get("epoch", -1))
    if epoch not in range(1, MAXIMUM_EPOCHS + 1):
        raise OrderedDecisionError("Exact resume requires last.pt from epoch 1..6")
    checkpoint_row = dict(checkpoint.get("metrics", {}) or {})
    legacy._finite_tree(checkpoint_row, path="checkpoint.metrics")
    if int(checkpoint_row.get("epoch", -1)) != epoch:
        raise OrderedDecisionError(
            "last.pt metric epoch does not match checkpoint epoch"
        )
    try:
        pending = float(checkpoint_row.get("validation_pending", math.nan))
    except (TypeError, ValueError) as error:
        raise OrderedDecisionError("last.pt validation_pending is malformed") from error
    if pending not in {0.0, 1.0}:
        raise OrderedDecisionError("last.pt validation_pending must be zero or one")
    completed_count = epoch - 1 if pending == 1.0 else epoch
    expected_epochs = list(range(1, completed_count + 1))
    if [int(row.get("epoch", -1)) for row in metrics] != expected_epochs or any(
        float(row.get("validation_pending", 1.0)) != 0.0 for row in metrics
    ):
        raise OrderedDecisionError("Resume metrics do not match completed validations")
    selection = dict(checkpoint.get("selection_state", {}) or {})
    early = dict(selection.get("early_stopping", {}) or {})
    if int(early.get("validation_count", -1)) != completed_count or early.get(
        "last_validation_epoch"
    ) != (completed_count if completed_count else None):
        raise OrderedDecisionError("Resume patience state is not epoch-aligned")
    if metrics:
        replay = _replay_selection(metrics, cfg)
        expected_selection = replay["selection_states"][completed_count]
    else:
        expected_selection = {
            "schema_version": 2,
            "best_feasible_score": None,
            "best_infeasible_score": None,
            "early_stopping": {
                "schema_version": 1,
                "best_key": None,
                "last_key": None,
                "bad_validation_count": 0,
                "validation_count": 0,
                "last_validation_epoch": None,
                "stopped": False,
                "stop_epoch": None,
            },
        }
    if selection != expected_selection:
        raise OrderedDecisionError("Resume selection state differs from exact replay")
    if pending == 1.0:
        if early.get("stopped") is not False or early.get("stop_epoch") is not None:
            raise OrderedDecisionError("Pending validation follows a terminal state")
        if any(
            name in checkpoint_row
            for name in (
                "selection_feasible",
                "selection_score",
                "selection_constraint_violation",
            )
        ):
            raise OrderedDecisionError("Pending row already has selection results")
        mode = "validation_pending"
    else:
        if not metrics or checkpoint_row != metrics[-1]:
            raise OrderedDecisionError("Completed last.pt differs from terminal metric")
        if any(
            name not in checkpoint_row
            for name in (
                "selection_feasible",
                "selection_score",
                "selection_constraint_violation",
            )
        ):
            raise OrderedDecisionError("Completed last.pt lacks selection results")
        stopped = bool(early.get("stopped"))
        if stopped != (early.get("stop_epoch") is not None):
            raise OrderedDecisionError("Completed last.pt has inconsistent stop state")
        mode = (
            "terminal_complete"
            if stopped or epoch == MAXIMUM_EPOCHS
            else "epoch_complete"
        )
    return {
        "mode": mode,
        "checkpoint_epoch": epoch,
        "completed_validation_epochs": expected_epochs,
        "checkpoint_metrics": checkpoint_row,
        "checkpoint_metrics_sha256": legacy._digest_json(checkpoint_row),
        "selection_state": selection,
        "selection_state_sha256": legacy._digest_json(selection),
        "rng_state": legacy._rng_resume_evidence(checkpoint),
    }


def record_run_resume(
    *,
    stage: str,
    run_dir: Path,
    out_file: Path,
    config_path: Path,
    partition_dir: Path,
    last_checkpoint_path: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    launcher_paths: list[Path],
    stage1_authorization: Path | None,
) -> dict[str, Any]:
    """Validate, reconcile, and attest an exact six-epoch continuation."""

    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    last_checkpoint_path = last_checkpoint_path.resolve()
    if run_dir.name != EXPERIMENTS[stage] or last_checkpoint_path != (
        run_dir / "checkpoints" / "last.pt"
    ):
        raise OrderedDecisionError("Resume checkpoint is not this stage's last.pt")
    if (run_dir / "evaluation" / "ordered_development_decision").exists():
        raise OrderedDecisionError("A decided centered stage cannot resume")
    approved = validate_config(stage, config_path)
    resolved_path = run_dir / "config.resolved.json"
    resolved = legacy._json(resolved_path)
    _validate_resolved_config(stage=stage, resolved=resolved, approved=approved)
    launch_path = Path(f"{run_dir}.prerequisites") / "run_launch_identity.json"
    launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=stage1_authorization,
    )
    launch_source = dict(launch.get("source", {}) or {})
    source_check = legacy._validate_shared_source_checkout(
        source_root,
        expected_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    if source_check != launch_source:
        raise OrderedDecisionError("Resume source differs from initial launch")
    _validate_calibration_source_binding(
        approved,
        source_root=source_root,
        source_git_head=source_git_head,
        source_remote_ref=source_remote_ref,
        source_remote_head=source_remote_head,
    )
    partition = legacy._validate_partition(partition_dir)
    before_checkpoint_sha = sha256_file(last_checkpoint_path)
    checkpoint = legacy._load_checkpoint(last_checkpoint_path)
    epoch = int(checkpoint.get("epoch", -1))
    if epoch not in range(1, MAXIMUM_EPOCHS + 1):
        raise OrderedDecisionError("Exact resume requires last.pt from epoch 1..6")
    if not isinstance(checkpoint.get("optimizer"), Mapping):
        raise OrderedDecisionError("Resume checkpoint lacks optimizer state")
    identity = _checkpoint_identity_evidence(
        checkpoint, stage=stage, approved_cfg=approved, resolved_cfg=resolved
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_existed = metrics_path.is_file()
    metrics_before_sha = sha256_file(metrics_path) if metrics_existed else None
    checkpoint_row = dict(checkpoint.get("metrics", {}) or {})
    try:
        pending = float(checkpoint_row.get("validation_pending", math.nan))
    except (TypeError, ValueError) as error:
        raise OrderedDecisionError("last.pt validation_pending is malformed") from error
    reconciliation: dict[str, Any] = {"performed": False}
    try:
        metrics = legacy._read_metrics(metrics_path) if metrics_existed else []
    except OrderedDecisionError:
        if pending != 0.0 or not metrics_existed:
            raise
        metrics, reconciliation = legacy._recover_truncated_terminal_metric(
            metrics_path, checkpoint_row, epoch
        )
    if pending == 0.0 and [int(row.get("epoch", -1)) for row in metrics] == list(
        range(1, epoch)
    ):
        reconciliation = legacy._atomic_append_reconciled_metric(
            metrics_path, checkpoint_row
        ) | {"kind": "missing_complete_row"}
        metrics = legacy._read_metrics(metrics_path)
    replay_cfg = _selection_replay_config(approved, resolved)
    progress = _centered_resume_progress_evidence(checkpoint, metrics, replay_cfg)
    terminal_summary = None
    terminal_selected = None
    if progress["mode"] == "terminal_complete":
        terminal_summary = legacy._terminal_selection_summary_payload(
            dict(checkpoint.get("selection_state", {}) or {})
        )
        replay = _replay_selection(metrics, replay_cfg)
        selected_name = (
            "best.pt"
            if terminal_summary["has_feasible_checkpoint"]
            else "best_infeasible.pt"
        )
        selected_path = run_dir / "checkpoints" / selected_name
        selected_checkpoint = legacy._load_checkpoint(selected_path)
        legacy._validate_replayed_winner(
            selected_checkpoint,
            replay,
            has_feasible=bool(terminal_summary["has_feasible_checkpoint"]),
        )
        terminal_selected = {
            "path": str(selected_path.resolve()),
            "sha256": sha256_file(selected_path),
            "epoch": int(selected_checkpoint.get("epoch", -1)),
        }
    predecessor_evidence = None
    stage2_input_evidence = None
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage-B resume lacks Stage-A authorization")
        stage2_path = Path(f"{run_dir}.prerequisites") / "ordered_stage_input.json"
        stage2_value = _validate_stage2_input(
            stage2_path,
            stage1_authorization=stage1_authorization,
            config_path=config_path,
        )
        authorization = verify_authorization(
            stage1_authorization, purpose="stage2", stage=STAGE1
        )
        predecessor_evidence = {
            "path": str(stage1_authorization.resolve()),
            "sha256": sha256_file(stage1_authorization),
            "authorization_identity": authorization["authorization_identity"],
            "decision_identity": authorization["decision_identity"],
        }
        stage2_input_evidence = {
            "path": str(stage2_path.resolve()),
            "sha256": sha256_file(stage2_path),
            "input_identity": stage2_value["input_identity"],
        }
    scripts = [
        {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for path in launcher_paths
    ]
    slurm = {
        "job_id": str(os.environ.get("SLURM_JOB_ID", "")),
        "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        "tasks": int(os.environ.get("SLURM_NTASKS", "0")),
    }
    if not slurm["job_id"] or slurm["nodes"] != 4:
        raise OrderedDecisionError("Exact resume requires a four-node Slurm job")
    without = {
        "schema_name": RESUME_ATTEMPT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "recorded_before_resume": True,
        "original_run_launch": {
            "path": str(launch_path.resolve()),
            "sha256": sha256_file(launch_path),
            "launch_identity": launch["launch_identity"],
        },
        "source": launch_source,
        "resume_source_check": source_check,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved_path": str(resolved_path.resolve()),
            "resolved_sha256": sha256_file(resolved_path),
        },
        "partition": {
            "path": str(Path(partition["dir"])),
            "partition_digest": legacy.EXPECTED_PARTITION_DIGEST,
        },
        "last_checkpoint": {
            "path": str(last_checkpoint_path),
            "sha256": before_checkpoint_sha,
            "epoch": epoch,
            "global_step": int(checkpoint.get("global_step", -1)),
            "identity_evidence": identity,
            "optimizer_state_present": True,
            "optimizer_reset": False,
        },
        "metrics_before_resume": {
            "path": str(metrics_path.resolve()),
            "existed": metrics_existed,
            "sha256": metrics_before_sha,
        },
        "metrics_reconciliation": reconciliation,
        "metrics_jsonl_present_before_trainer_reentry": metrics_path.is_file(),
        "metrics_jsonl_sha256": (
            sha256_file(metrics_path)
            if metrics_path.is_file()
            else hashlib.sha256(b"").hexdigest()
        ),
        "resume_progress": progress,
        "terminal_finalization": (
            None
            if terminal_summary is None
            else {
                "required": True,
                "selection_summary_path": str(
                    (run_dir / "selection_summary.json").resolve()
                ),
                "selection_summary_payload": terminal_summary,
                "selection_summary_payload_sha256": legacy._digest_json(
                    terminal_summary
                ),
                "selected_checkpoint": terminal_selected,
                "trainer_reentry": False,
            }
        ),
        "stage1_authorization": predecessor_evidence,
        "stage2_launch_input": stage2_input_evidence,
        "resume_launchers": scripts,
        "slurm": slurm,
        "wandb": "disabled",
        "staged_neighbor_splits": ["train", "val"],
        "relevance_calibration": identity["relevance_calibration"],
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    if sha256_file(last_checkpoint_path) != before_checkpoint_sha:
        raise OrderedDecisionError("last.pt changed during resume preflight")
    terminal_summary_path = run_dir / "selection_summary.json"
    if (
        terminal_summary is not None
        and terminal_summary_path.exists()
        and legacy._json(terminal_summary_path) != terminal_summary
    ):
        raise OrderedDecisionError(
            "Existing terminal selection summary differs from replay"
        )
    value = {**without, "resume_identity": legacy._digest_json(without)}
    legacy._atomic_json_record(out_file.resolve(), value)
    if terminal_summary is not None and not terminal_summary_path.exists():
        legacy._atomic_write_terminal_selection_summary(
            terminal_summary_path, terminal_summary
        )
    return value


def _validate_execution_lease_attestation(
    attestation_path: Path, *, lease_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate that an attestation names the currently active centered claim."""

    attestation = legacy._json(attestation_path.resolve())
    without_identity = {
        name: value
        for name, value in attestation.items()
        if name != "attestation_identity"
    }
    claim = dict(attestation.get("claim", {}) or {})
    if (
        attestation.get("schema_name") != EXECUTION_LEASE_ATTESTATION_SCHEMA_NAME
        or int(attestation.get("schema_version", -1)) != SCHEMA_VERSION
        or legacy._digest_json(without_identity)
        != attestation.get("attestation_identity")
        or Path(str(attestation.get("lease_path", ""))).resolve()
        != lease_path.resolve()
        or not lease_path.exists()
        or _lease_owner(lease_path.resolve()) != claim
    ):
        raise OrderedDecisionError(
            "Centered execution-lease attestation is not the active claim"
        )
    return attestation, claim


def _precheckpoint_output_evidence(run_dir: Path, *, stage: str) -> dict[str, Any]:
    """Prove an interrupted output tree contains no persisted training progress."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise OrderedDecisionError("Pre-checkpoint output is not a real directory")
    allowed = {
        "config.resolved.json",
        "validation_text_partition.json",
        "validation_text_partition.json.tmp",
        "checkpoints",
    }
    entries = sorted(run_dir.iterdir(), key=lambda child: child.name)
    unexpected = [child.name for child in entries if child.name not in allowed]
    if unexpected:
        raise OrderedDecisionError(
            "Pre-checkpoint output contains material progress or unknown artifacts: "
            f"{unexpected}"
        )
    checkpoint_temporaries: list[dict[str, Any]] = []
    checkpoints = run_dir / "checkpoints"
    if checkpoints.exists():
        if not checkpoints.is_dir() or checkpoints.is_symlink():
            raise OrderedDecisionError("Pre-checkpoint checkpoint path is unsafe")
        for path in sorted(checkpoints.iterdir(), key=lambda child: child.name):
            if (
                not path.is_file()
                or path.is_symlink()
                or re.fullmatch(r"\.last\.pt\.[A-Za-z0-9_-]{6,64}\.tmp", path.name)
                is None
            ):
                raise OrderedDecisionError(
                    "Pre-checkpoint recovery refuses a published or unknown "
                    f"checkpoint artifact: {path.name}"
                )
            checkpoint_temporaries.append(
                {
                    "file": f"checkpoints/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "published": False,
                }
            )
    artifacts: dict[str, Any] = {}
    for filename in (
        "config.resolved.json",
        "validation_text_partition.json",
        "validation_text_partition.json.tmp",
    ):
        path = run_dir / filename
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 5_000_000:
            raise OrderedDecisionError(f"Unsafe initialization artifact: {path}")
        row: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "json_complete": False,
        }
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        if value is not None:
            if not isinstance(value, dict):
                raise OrderedDecisionError(
                    f"Initialization JSON is not an object: {path}"
                )
            row["json_complete"] = True
            if filename == "config.resolved.json":
                if value.get("experiment_name") != EXPERIMENTS[stage]:
                    raise OrderedDecisionError(
                        "Initialization config belongs to another experiment"
                    )
                partition = dict(value.get("validation_text_partition", {}) or {})
                forbidden = {
                    "resolved_artifact",
                    "assignments",
                    "confirmation_manifest",
                    "confirmation_normalized_texts",
                    "development_normalized_texts",
                    "partition_dir",
                }
                if forbidden.intersection(partition):
                    raise OrderedDecisionError(
                        "Initialization config persisted sealed membership"
                    )
            elif (
                value.get("schema_name")
                != "signtrajfield_development_validation_runtime"
                or value.get("validation_only") is not True
                or "assignments" in value
            ):
                raise OrderedDecisionError(
                    "Initialization partition artifact is not development-only"
                )
        artifacts[filename] = row
    return {
        "entries": [child.name for child in entries],
        "artifacts": artifacts,
        "unpublished_checkpoint_temporaries": checkpoint_temporaries,
        "published_checkpoints_absent": True,
        "metrics_absent": not (run_dir / "metrics.jsonl").exists(),
        "selection_summary_absent": not (run_dir / "selection_summary.json").exists(),
        "material_progress_preserved": True,
    }


def record_precheckpoint_retry(
    *,
    stage: str,
    run_dir: Path,
    out_file: Path,
    config_path: Path,
    partition_dir: Path,
    lease_path: Path,
    lease_attestation_path: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    """Quarantine an attested pre-first-checkpoint attempt for a fresh retry."""

    if stage not in STAGES:
        raise OrderedDecisionError("Unknown centered pre-checkpoint stage")
    validate_config(stage, config_path)
    partition = legacy._validate_partition(partition_dir)
    run_dir = run_dir.resolve()
    launch_path = Path(f"{run_dir}.prerequisites") / "run_launch_identity.json"
    launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=stage1_authorization,
    )
    source = legacy._validate_shared_source_checkout(
        source_root,
        expected_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    if source != dict(launch.get("source", {}) or {}):
        raise OrderedDecisionError("Pre-checkpoint source differs from run launch")
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage-B retry lacks Stage-A authorization")
        _validate_stage2_input(
            Path(f"{run_dir}.prerequisites") / "ordered_stage_input.json",
            stage1_authorization=stage1_authorization,
            config_path=config_path,
        )
    elif stage1_authorization is not None:
        raise OrderedDecisionError("Stage-A retry cannot have a predecessor")

    _attestation, claim = _validate_execution_lease_attestation(
        lease_attestation_path, lease_path=lease_path.resolve()
    )
    if (
        claim.get("purpose") != "training"
        or claim.get("stage") != stage
        or claim.get("source_git_head") != str(source_git_head).lower()
        or claim.get("binding_identity") != sha256_file(config_path)
    ):
        raise OrderedDecisionError("Pre-checkpoint retry has the wrong active lease")
    replaced = claim.get("replaced_stale_owner")
    original_job = str(dict(launch.get("slurm", {}) or {}).get("job_id", ""))
    same_requeued_job = (
        replaced is None
        and claim.get("slurm_job_id") == original_job
        and int(os.environ.get("SLURM_RESTART_COUNT", "0") or "0") > 0
    )
    if replaced is None and not same_requeued_job:
        raise OrderedDecisionError(
            "Pre-checkpoint retry lacks scheduler-proven stale-owner evidence"
        )
    if isinstance(replaced, Mapping) and dict(
        replaced.get("scheduler_evidence", {}) or {}
    ).get("classification") not in {"terminal", "not_found"}:
        raise OrderedDecisionError(
            "Pre-checkpoint predecessor was not scheduler-proven terminal"
        )

    quarantine_root = Path(f"{run_dir}.prerequisites") / "precheckpoint_outputs"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    claimed: set[Path] = set()
    if out_file.parent.is_dir():
        for record_path in out_file.parent.glob("*.json"):
            try:
                record = legacy._json(record_path)
            except OrderedDecisionError:
                continue
            if record.get("schema_name") != (
                "signtrajfield_centered_memory_precheckpoint_retry"
            ):
                continue
            for row in record.get("quarantined_outputs", []):
                if isinstance(row, Mapping):
                    claimed.add(Path(str(row.get("path", ""))).resolve())
    if run_dir.exists():
        _precheckpoint_output_evidence(run_dir, stage=stage)
        quarantine = quarantine_root / (
            f"{claim['slurm_job_id']}.{claim['claim_identity']}.output"
        )
        if quarantine.exists():
            raise OrderedDecisionError("Pre-checkpoint quarantine target exists")
        os.rename(run_dir, quarantine)
    candidates = sorted(path for path in quarantine_root.iterdir() if path.is_dir())
    if len([path for path in candidates if path.resolve() not in claimed]) > 1:
        raise OrderedDecisionError(
            "Pre-checkpoint history has multiple unattested output trees"
        )
    quarantined = [
        {
            "path": str(path.resolve()),
            "previously_attested": path.resolve() in claimed,
            "evidence": _precheckpoint_output_evidence(path, stage=stage),
        }
        for path in candidates
    ]
    without_identity = {
        "schema_name": "signtrajfield_centered_memory_precheckpoint_retry",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "fresh_epoch_restart": 1,
        "run_dir": str(run_dir),
        "quarantined_outputs": quarantined,
        "run_launch": {
            "path": str(launch_path.resolve()),
            "sha256": sha256_file(launch_path),
            "launch_identity": launch["launch_identity"],
        },
        "development_manifest": {
            "path": str(partition["development_manifest"]),
            "sha256": partition["development_manifest_sha256"],
        },
        "active_lease_claim_identity": claim["claim_identity"],
        "stale_owner_evidence": replaced,
        "same_slurm_job_requeue": same_requeued_job,
        "source": source,
        "config_sha256": sha256_file(config_path),
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    payload = {
        **without_identity,
        "retry_identity": legacy._digest_json(without_identity),
    }
    return legacy._atomic_json_record(
        out_file.resolve(), payload, allow_matching_existing=True
    )


def _lease_owner(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise OrderedDecisionError("Execution lease is not a real directory")
    owner = legacy._json(path / "owner.json")
    ready = legacy._json(path / "READY")
    without = {key: value for key, value in owner.items() if key != "claim_identity"}
    if (
        owner.get("schema_name") != EXECUTION_LEASE_SCHEMA_NAME
        or int(owner.get("schema_version", -1)) != SCHEMA_VERSION
        or owner.get("purpose")
        not in {"calibration", "training", "decision", "diagnostic", "confirmation"}
        or owner.get("stage") not in LEASE_STAGES
        or owner.get("experiment_name") != LEASE_EXPERIMENTS[owner["stage"]]
        or not re.fullmatch(r"[0-9]+", str(owner.get("slurm_job_id", "")))
        or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            str(owner.get("source_git_head", "")).lower(),
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(owner.get("binding_identity", "")))
        or not isinstance(owner.get("claim_nonce"), str)
        or legacy._digest_json(without) != owner.get("claim_identity")
        or ready
        != {
            "schema_name": EXECUTION_LEASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "claim_identity": owner.get("claim_identity"),
        }
    ):
        raise OrderedDecisionError("Execution lease is malformed")
    return owner


def _publish_lease(path: Path, owner: Mapping[str, Any]) -> bool:
    building = path.with_name(
        f".{path.name}.claiming.{owner['slurm_job_id']}.{uuid.uuid4().hex}"
    )
    building.mkdir(parents=False, exist_ok=False)
    try:
        legacy._atomic_json_record(building / "owner.json", owner)
        legacy._atomic_json_record(
            building / "READY",
            {
                "schema_name": EXECUTION_LEASE_SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "claim_identity": owner["claim_identity"],
            },
        )
        try:
            os.rename(building, path)
        except (FileExistsError, OSError):
            return False
        return True
    finally:
        if building.exists():
            shutil.rmtree(building)


def acquire_execution_lease(
    *,
    lease_path: Path,
    attestation_path: Path,
    purpose: str,
    stage: str,
    source_git_head: str,
    slurm_job_id: str,
    binding_identity: str,
) -> dict[str, Any]:
    allowed = {
        "calibration": {CALIBRATION_STAGE},
        "training": set(STAGES),
        "decision": set(STAGES),
        "diagnostic": set(STAGES),
        "confirmation": set(STAGES),
    }
    if purpose not in allowed or stage not in allowed[purpose]:
        raise OrderedDecisionError("Invalid purpose/stage lease pairing")
    source_git_head = str(source_git_head).lower()
    slurm_job_id = str(slurm_job_id)
    binding_identity = str(binding_identity).lower()
    if (
        not re.fullmatch(r"[0-9]+", slurm_job_id)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_git_head)
        or not re.fullmatch(r"[0-9a-f]{64}", binding_identity)
    ):
        raise OrderedDecisionError("Execution lease identities are malformed")
    lease_path = lease_path.resolve()
    attestation_path = attestation_path.resolve()
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    replaced = None
    for _attempt in range(3):
        if lease_path.exists():
            existing = _lease_owner(lease_path)
            same = all(
                existing.get(name) == value
                for name, value in {
                    "purpose": purpose,
                    "stage": stage,
                    "source_git_head": source_git_head,
                    "slurm_job_id": slurm_job_id,
                    "binding_identity": binding_identity,
                }.items()
            )
            if same:
                owner = existing
                break
            scheduler = legacy._slurm_job_terminal_evidence(existing["slurm_job_id"])
            history = lease_path.with_name(f"{lease_path.name}.history")
            history.mkdir(parents=True, exist_ok=True)
            stale = history / (
                f"{existing['slurm_job_id']}.{existing['claim_identity']}.stale"
            )
            try:
                os.rename(lease_path, stale)
            except (FileNotFoundError, FileExistsError, OSError):
                continue
            replaced = {
                "claim_identity": existing["claim_identity"],
                "slurm_job_id": existing["slurm_job_id"],
                "scheduler_evidence": scheduler,
            }
        without = {
            "schema_name": EXECUTION_LEASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "purpose": purpose,
            "stage": stage,
            "experiment_name": LEASE_EXPERIMENTS[stage],
            "slurm_job_id": slurm_job_id,
            "slurm_restart_count": int(
                os.environ.get("SLURM_RESTART_COUNT", "0") or "0"
            ),
            "source_git_head": source_git_head,
            "binding_identity": binding_identity,
            "claim_nonce": uuid.uuid4().hex,
            "created_unix_ns": time.time_ns(),
            "replaced_stale_owner": replaced,
        }
        candidate = {**without, "claim_identity": legacy._digest_json(without)}
        if _publish_lease(lease_path, candidate):
            owner = candidate
            break
    else:
        raise OrderedDecisionError("Could not acquire execution lease")
    without_attestation = {
        "schema_name": EXECUTION_LEASE_ATTESTATION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "lease_path": str(lease_path),
        "claim": owner,
    }
    attestation = {
        **without_attestation,
        "attestation_identity": legacy._digest_json(without_attestation),
    }
    return legacy._atomic_json_record(
        attestation_path, attestation, allow_matching_existing=True
    )


def release_execution_lease(
    *, lease_path: Path, attestation_path: Path
) -> dict[str, Any]:
    lease_path = lease_path.resolve()
    attestation = legacy._json(attestation_path.resolve())
    without = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_identity"
    }
    owner = _lease_owner(lease_path)
    if (
        legacy._digest_json(without) != attestation.get("attestation_identity")
        or attestation.get("lease_path") != str(lease_path)
        or attestation.get("claim") != owner
    ):
        raise OrderedDecisionError("Lease attestation is not the active claim")
    history = lease_path.with_name(f"{lease_path.name}.history")
    history.mkdir(parents=True, exist_ok=True)
    released = history / f"{owner['claim_identity']}.released"
    try:
        os.rename(lease_path, released)
    except (FileNotFoundError, FileExistsError, OSError) as error:
        raise OrderedDecisionError("Execution lease changed during release") from error
    return {"released": True, "claim_identity": owner["claim_identity"]}


def _write_centered_decision(
    out_dir: Path,
    decision: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
) -> None:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if legacy._json(out_dir / "decision.json") == dict(decision):
            return
        raise OrderedDecisionError("Refusing to replace centered decision")
    building = out_dir.with_name(f".{out_dir.name}.building.{uuid.uuid4().hex}")
    building.mkdir(parents=True, exist_ok=False)
    try:
        legacy._atomic_json_record(building / "decision.json", decision)
        if authorization is not None:
            legacy._atomic_json_record(
                building / f"authorize_{authorization['purpose']}.json",
                authorization,
            )
        legacy._atomic_json_record(
            building / "READY",
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "decision_identity": decision["decision_identity"],
                "status": decision["status"],
                "authorized_purpose": (
                    authorization.get("purpose") if authorization else None
                ),
            },
        )
        os.rename(building, out_dir)
    finally:
        if building.exists():
            shutil.rmtree(building)


def _write_centered_integrity_failure(
    out_dir: Path, decision: Mapping[str, Any]
) -> None:
    if out_dir.exists():
        raise OrderedDecisionError("A terminal centered decision already exists")
    history = out_dir.resolve().parent / (f"{out_dir.name}.integrity_invalid_attempts")
    history.mkdir(parents=True, exist_ok=True)
    target = history / str(decision["decision_identity"])
    if target.exists():
        if legacy._json(target / "decision.json") == dict(decision):
            return
        raise OrderedDecisionError("Centered integrity identity collision")
    building = history / f".{target.name}.building.{uuid.uuid4().hex}"
    building.mkdir()
    try:
        legacy._atomic_json_record(building / "decision.json", decision)
        legacy._atomic_json_record(
            building / "READY",
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "decision_identity": decision["decision_identity"],
                "status": legacy.INTEGRITY_INVALID,
                "authorized_purpose": None,
                "terminal": False,
            },
        )
        os.rename(building, target)
    finally:
        if building.exists():
            shutil.rmtree(building)


def record_calibration_completion(
    *,
    artifact_dir: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    lease_path: Path,
    lease_attestation_path: Path,
    out_file: Path,
    publication_existing: bool,
) -> dict[str, Any]:
    """Attest publication recovery before releasing the calibration lease."""

    calibration = validate_relevance_calibration_artifact(artifact_dir)
    source = validate_relevance_calibration_source(
        calibration,
        source_root=source_root,
        expected_git_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    owner = _lease_owner(lease_path.resolve())
    attestation = legacy._json(lease_attestation_path.resolve())
    attestation_without_identity = {
        name: value
        for name, value in attestation.items()
        if name != "attestation_identity"
    }
    if (
        legacy._digest_json(attestation_without_identity)
        != attestation.get("attestation_identity")
        or attestation.get("lease_path") != str(lease_path.resolve())
        or attestation.get("claim") != owner
        or owner.get("purpose") != "calibration"
        or owner.get("stage") != CALIBRATION_STAGE
        or owner.get("source_git_head") != str(source_git_head).lower()
    ):
        raise OrderedDecisionError("Calibration completion has the wrong active lease")
    artifact_dir = artifact_dir.resolve()
    without_identity = {
        "schema_name": CALIBRATION_COMPLETION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "artifact": {
            "path": str(artifact_dir),
            "identity": calibration["identity"],
            "calibration_sha256": sha256_file(artifact_dir / "calibration.json"),
            "map_sha256": sha256_file(artifact_dir / "calibration_map.npz"),
            "ready_sha256": sha256_file(artifact_dir / "READY"),
        },
        "source": source,
        "lease": {
            "path": str(lease_path.resolve()),
            "claim_identity": owner["claim_identity"],
            "attestation_path": str(lease_attestation_path.resolve()),
            "attestation_sha256": sha256_file(lease_attestation_path),
            "replaced_stale_owner": owner.get("replaced_stale_owner"),
        },
        "publication_existing_at_attempt_start": bool(publication_existing),
        "artifact_mutated_during_reconciliation": False,
        "ready_revalidated": True,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    value = {
        **without_identity,
        "completion_identity": legacy._digest_json(without_identity),
    }
    return legacy._atomic_json_record(out_file.resolve(), value)


def verify_authorization(path: Path, *, purpose: str, stage: str) -> dict[str, Any]:
    if stage not in STAGES or purpose not in {
        "stage2",
        "diagnostic",
        "confirmation",
    }:
        raise OrderedDecisionError("Unknown centered authorization purpose/stage")
    path = path.resolve()
    value = legacy._json(path)
    without = {
        name: child for name, child in value.items() if name != "authorization_identity"
    }
    if (
        value.get("schema_name") != AUTHORIZATION_SCHEMA_NAME
        or int(value.get("schema_version", -1)) != SCHEMA_VERSION
        or value.get("purpose") != purpose
        or value.get("stage") != stage
        or value.get("experiment_name") != EXPERIMENTS[stage]
        or legacy._digest_json(without) != value.get("authorization_identity")
    ):
        raise OrderedDecisionError("Centered authorization type/digest changed")
    decision = legacy._json(path.parent / "decision.json")
    ready = legacy._json(path.parent / "READY")
    decision_without_identity = {
        name: child for name, child in decision.items() if name != "decision_identity"
    }
    expected_status = (
        legacy.VALID_INFEASIBLE if purpose == "stage2" else legacy.DEVELOPMENT_FEASIBLE
    )
    if (
        decision.get("schema_name") != SCHEMA_NAME
        or int(decision.get("schema_version", -1)) != SCHEMA_VERSION
        or decision.get("stage") != stage
        or decision.get("experiment_name") != EXPERIMENTS[stage]
        or decision.get("status") != expected_status
        or decision.get("integrity_valid") is not True
        or decision.get("development_feasible")
        != (expected_status == legacy.DEVELOPMENT_FEASIBLE)
        or legacy._digest_json(decision_without_identity)
        != decision.get("decision_identity")
        or decision.get("decision_identity") != value.get("decision_identity")
        or ready.get("decision_identity") != value.get("decision_identity")
        or ready.get("schema_name") != SCHEMA_NAME
        or int(ready.get("schema_version", -1)) != SCHEMA_VERSION
        or ready.get("status") != expected_status
        or ready.get("authorized_purpose")
        != ("stage2" if purpose == "stage2" else "diagnostic")
        or value.get("checkpoint") != decision.get("checkpoint")
        or value.get("partition") != decision.get("partition")
        or value.get("run_launch_identity") != decision.get("run_launch_identity")
        or value.get("predecessor_authorization")
        != decision.get("predecessor_authorization")
    ):
        raise OrderedDecisionError("Centered authorization is detached from decision")
    checkpoint = dict(value.get("checkpoint", {}) or {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise OrderedDecisionError("Authorized centered checkpoint changed")
    for filename, key in (
        ("metrics.jsonl", "metrics_jsonl_sha256"),
        ("selection_summary.json", "selection_summary_sha256"),
        ("config.resolved.json", "resolved_config_sha256"),
    ):
        artifact = checkpoint_path.parents[1] / filename
        if sha256_file(artifact) != checkpoint.get(key):
            raise OrderedDecisionError(f"Authorized run artifact changed: {filename}")
    audit = dict(checkpoint.get("selected_checkpoint_integrity_audit", {}) or {})
    legacy._finite_tree(audit, path="authorization.selected_checkpoint_integrity_audit")
    audit_path = Path(str(audit.get("path", "")))
    exact_zero = dict(audit.get("exact_zero", {}) or {})
    joint = dict(audit.get("joint_tuple", {}) or {})
    v2_parity = dict(audit.get("v2_text_only_parity", {}) or {})
    joint_names = {
        "joint_tuple_prediction_max_abs",
        "joint_tuple_duration_max_abs",
    }
    if set(joint) != joint_names:
        raise OrderedDecisionError("Authorized selected-checkpoint joint audit changed")
    for name in joint_names:
        _validated_nonnegative_max_abs(
            joint[name], name=f"authorization.audit.{name}", maximum=1e-7
        )
    if set(v2_parity) != {"prediction_max_abs", "duration_max_abs"}:
        raise OrderedDecisionError("Authorized selected-checkpoint v2 audit changed")
    resolved_artifact = legacy._json(
        checkpoint_path.parents[1] / "config.resolved.json"
    )
    checkpoint_parity = _validated_checkpoint_v2_parity(
        {"v2_to_v3_text_only_parity": checkpoint.get("parity")},
        resolved_artifact,
    )
    for name in ("prediction_max_abs", "duration_max_abs"):
        audit_value = _validated_nonnegative_max_abs(
            v2_parity[name],
            name=f"authorization.audit.v2_{name}",
            maximum=1e-7,
        )
        if audit_value != checkpoint_parity[name]:
            raise OrderedDecisionError(
                "Authorized evaluator parity differs from checkpoint parity"
            )
    if (
        not audit_path.is_file()
        or sha256_file(audit_path) != audit.get("sha256")
        or exact_zero != {name: 0.0 for name in _EXACT_ZERO_AUDITS}
        or audit.get("test_data_accessed") is not False
        or audit.get("confirmation_manifest_opened") is not False
    ):
        raise OrderedDecisionError(
            "Authorized selected-checkpoint integrity replay changed"
        )
    calibration = dict(checkpoint.get("relevance_calibration", {}) or {})
    config = dict(checkpoint.get("config", {}) or {})
    config_path = Path(str(config.get("path", "")))
    if sha256_file(config_path) != config.get("sha256"):
        raise OrderedDecisionError("Authorized centered config changed")
    resolved_config = validate_config(stage, config_path)
    expected = resolved_config["sentence_memory"][
        "resolved_relevance_calibration_identity"
    ]
    if calibration != expected:
        raise OrderedDecisionError(
            "Authorization calibration differs from the live sealed artifact"
        )
    launch_source = dict(
        dict(value.get("run_launch_identity", {}) or {}).get("source", {}) or {}
    )
    executing_source = legacy._validate_shared_source_checkout(
        SOURCE_ROOT,
        expected_head=str(launch_source.get("git_head", "")),
        expected_remote_ref=str(launch_source.get("remote_ref", "")),
        expected_remote_head=str(launch_source.get("remote_head", "")),
    )
    if executing_source != launch_source:
        raise OrderedDecisionError(
            "Authorization is being consumed from another source checkout"
        )
    _validate_calibration_source_binding(
        resolved_config,
        source_root=SOURCE_ROOT,
        source_git_head=str(launch_source.get("git_head", "")),
        source_remote_ref=str(launch_source.get("remote_ref", "")),
        source_remote_head=str(launch_source.get("remote_head", "")),
    )
    if purpose == "confirmation":
        diagnostic = dict(value.get("development_diagnostic", {}) or {})
        _validate_diagnostic_ready(
            Path(str(diagnostic.get("directory", ""))),
            diagnostic_authorization_path=Path(
                str(diagnostic.get("authorization_path", ""))
            ),
            expected_stage=stage,
            expected_checkpoint=checkpoint,
            expected_calibration=expected,
            expected_evidence=diagnostic,
        )
    return value


def _validate_diagnostic_summary_contract(
    summary: Mapping[str, Any], *, stage: str
) -> None:
    """Reject incomplete explanatory diagnostics before holdout authorization."""

    expected_export_modes = (*EVAL_MODES[stage], "broadcast_complete", "all_null")
    if (
        summary.get("stage") != stage
        or summary.get("experiment_name") != EXPERIMENTS[stage]
        or summary.get("scientific_split") != "val_development_novel_text_only"
        or summary.get("partition_digest") != legacy.EXPECTED_PARTITION_DIGEST
        or summary.get("development_manifest_sha256")
        != legacy.EXPECTED_DEVELOPMENT_MANIFEST_SHA256
        or int(summary.get("development_rows", -1)) != legacy.EXPECTED_DEVELOPMENT_ROWS
        or int(summary.get("development_text_clusters", -1))
        != legacy.EXPECTED_DEVELOPMENT_TEXTS
        or dict(summary.get("bootstrap", {}) or {}) != {"samples": 10_000, "seed": 1234}
        or summary.get("explanatory_only") is not True
        or summary.get("scientific_gate") is not False
        or summary.get("changes_checkpoint_selection") is not False
    ):
        raise OrderedDecisionError("Locked diagnostic scope/completeness changed")

    query_bindings = dict(summary.get("query_bindings", {}) or {})
    export_hashes = dict(summary.get("input_export_summary_sha256", {}) or {})
    if (
        tuple(query_bindings) != expected_export_modes
        or tuple(export_hashes) != expected_export_modes
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in export_hashes.values()
        )
    ):
        raise OrderedDecisionError(
            "Locked diagnostic does not bind the complete ordered export set"
        )
    for mode, binding in query_bindings.items():
        try:
            legacy.validate_factorized_export_query_binding(
                binding,
                expected_authority="config_pinned_development_manifest_v1",
                expected_manifest_file="manifest_development.jsonl",
                expected_manifest_sha256=(legacy.EXPECTED_DEVELOPMENT_MANIFEST_SHA256),
                expected_query_rows=legacy.EXPECTED_DEVELOPMENT_ROWS,
            )
        except OrderedDecisionError as error:
            raise OrderedDecisionError(
                f"Locked diagnostic query binding changed for {mode}"
            ) from error

    expected_controls = tuple(
        mode for mode in EVAL_MODES[stage] if mode not in {"off", "on"}
    ) + ("broadcast_complete",)
    controls = dict(summary.get("control_provenance", {}) or {})
    if set(controls) != set(expected_controls) or len(controls) != len(
        expected_controls
    ):
        raise OrderedDecisionError("Locked diagnostic control set is incomplete")
    legacy._finite_tree(controls, path="diagnostic.control_provenance")
    if any(
        not isinstance(value, Mapping) or value.get("passed") is not True
        for value in controls.values()
    ):
        raise OrderedDecisionError("A locked diagnostic control audit did not pass")

    invariants = dict(summary.get("exact_invariants", {}) or {})
    expected_invariants = (
        "joint_tuple_vs_correct",
        "uniform_final_vs_off",
        "broadcast_complete_vs_off",
        "all_null_vs_off",
        "all_null",
    )
    if tuple(invariants) != expected_invariants:
        raise OrderedDecisionError("Locked diagnostic invariant set is incomplete")
    legacy._finite_tree(invariants, path="diagnostic.exact_invariants")
    joint = dict(invariants.get("joint_tuple_vs_correct", {}) or {})
    if joint.get("passed") is not True:
        raise OrderedDecisionError("Joint-tuple selected-checkpoint audit failed")
    for name in ("prediction_max_abs", "duration_max_abs"):
        _validated_nonnegative_max_abs(
            joint.get(name),
            name=f"diagnostic.joint_tuple_vs_correct.{name}",
            maximum=1e-7,
        )
    for name in (
        "uniform_final_vs_off",
        "broadcast_complete_vs_off",
        "all_null_vs_off",
    ):
        row = dict(invariants.get(name, {}) or {})
        if (
            row.get("passed") is not True
            or row.get("requires_bitwise_array_equality") is not True
            or row.get("prediction_arrays_equal") is not True
            or row.get("duration_values_equal") is not True
            or float(row.get("prediction_max_abs", math.inf)) != 0.0
            or float(row.get("duration_max_abs", math.inf)) != 0.0
        ):
            raise OrderedDecisionError(
                f"Locked diagnostic exact equality failed: {name}"
            )
    all_null = dict(invariants.get("all_null", {}) or {})
    if (
        all_null.get("passed") is not True
        or int(all_null.get("payload_reads", -1)) != 0
        or float(all_null.get("gate_max_abs", math.inf)) != 0.0
        or float(all_null.get("candidate_mass_max_abs", math.inf)) != 0.0
        or float(all_null.get("null_mass_max_abs_error", math.inf)) != 0.0
    ):
        raise OrderedDecisionError("Locked diagnostic all-null audit is not exact")


def _validate_diagnostic_layerwise_csv(
    directory: Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    relative = str(summary.get("layerwise_metrics_file", ""))
    if relative != "layerwise_metrics.csv":
        raise OrderedDecisionError("Locked diagnostic layerwise filename changed")
    path = directory / relative
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise OrderedDecisionError("Locked diagnostic layerwise CSV is absent")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "mode",
            "layer",
            "part",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
        ]:
            raise OrderedDecisionError("Locked diagnostic layerwise columns changed")
        rows = list(reader)
    if not rows:
        raise OrderedDecisionError("Locked diagnostic layerwise CSV is empty")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        try:
            normalized_row = {
                "mode": str(row["mode"]),
                "layer": int(row["layer"]),
                "part": str(row["part"]),
                "metric": str(row["metric"]),
                "mean": float(row["mean"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise OrderedDecisionError(
                "Locked diagnostic layerwise row is malformed"
            ) from error
        legacy._finite_tree(normalized_row, path="diagnostic.layerwise_row")
        normalized.append(normalized_row)
    if legacy._digest_json(normalized) != summary.get("layerwise_rows_identity"):
        raise OrderedDecisionError("Locked diagnostic layerwise rows changed")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(normalized),
        "finite": True,
        "rows_identity": summary["layerwise_rows_identity"],
    }


def _validate_diagnostic_ready(
    directory: Path,
    *,
    diagnostic_authorization_path: Path,
    expected_stage: str,
    expected_checkpoint: Mapping[str, Any],
    expected_calibration: Mapping[str, Any],
    expected_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    summary_path = directory / "summary.json"
    ready_path = directory / "READY"
    summary = legacy._json(summary_path)
    ready = legacy._json(ready_path)
    summary_payload = {
        name: value for name, value in summary.items() if name != "identity"
    }
    ready_payload = {
        name: value for name, value in ready.items() if name != "ready_identity"
    }
    authorization = verify_authorization(
        diagnostic_authorization_path.resolve(),
        purpose="diagnostic",
        stage=expected_stage,
    )
    auth_ref = dict(summary.get("authorization", {}) or {})
    checkpoint_ref = dict(summary.get("checkpoint", {}) or {})
    _validate_diagnostic_summary_contract(summary, stage=expected_stage)
    layerwise = _validate_diagnostic_layerwise_csv(directory, summary)
    if (
        summary.get("schema_name") != DIAGNOSTIC_SCHEMA_NAME
        or int(summary.get("schema_version", -1)) != 1
        or summary.get("completed") is not True
        or summary.get("development_validation_only") is not True
        or summary.get("test_data_accessed") is not False
        or summary.get("confirmation_manifest_opened") is not False
        or summary.get("identity") != legacy._digest_json(summary_payload)
        or auth_ref
        != {
            "path": str(diagnostic_authorization_path.resolve()),
            "sha256": sha256_file(diagnostic_authorization_path),
            "authorization_identity": authorization["authorization_identity"],
        }
        or checkpoint_ref.get("path") != expected_checkpoint.get("path")
        or checkpoint_ref.get("sha256") != expected_checkpoint.get("sha256")
        or summary.get("relevance_calibration") != dict(expected_calibration)
        or ready_payload
        != {
            "schema_name": DIAGNOSTIC_READY_SCHEMA_NAME,
            "schema_version": 1,
            "diagnostic_identity": summary.get("identity"),
            "summary_sha256": sha256_file(summary_path),
            "authorization_identity": authorization["authorization_identity"],
        }
        or ready.get("ready_identity") != legacy._digest_json(ready_payload)
    ):
        raise OrderedDecisionError("Locked development diagnostic is invalid")
    evidence = {
        "directory": str(directory),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": sha256_file(ready_path),
        "diagnostic_identity": summary["identity"],
        "authorization_path": str(diagnostic_authorization_path.resolve()),
        "authorization_sha256": sha256_file(diagnostic_authorization_path),
        "authorization_identity": authorization["authorization_identity"],
        "layerwise_csv": layerwise,
    }
    if expected_evidence is not None and dict(expected_evidence) != evidence:
        raise OrderedDecisionError("Confirmation authorization diagnostic changed")
    return evidence


def finalize_confirmation_authorization(
    *,
    stage: str,
    diagnostic_authorization_path: Path,
    diagnostic_dir: Path,
    out_file: Path,
) -> dict[str, Any]:
    diagnostic_authorization = verify_authorization(
        diagnostic_authorization_path, purpose="diagnostic", stage=stage
    )
    decision_path = diagnostic_authorization_path.resolve().parent / "decision.json"
    decision = legacy._json(decision_path)
    if (
        decision.get("status") != legacy.DEVELOPMENT_FEASIBLE
        or decision.get("development_feasible") is not True
        or decision.get("integrity_valid") is not True
    ):
        raise OrderedDecisionError("Only a feasible centered decision may confirm")
    checkpoint = dict(diagnostic_authorization.get("checkpoint", {}) or {})
    calibration = dict(checkpoint.get("relevance_calibration", {}) or {})
    diagnostic = _validate_diagnostic_ready(
        diagnostic_dir,
        diagnostic_authorization_path=diagnostic_authorization_path,
        expected_stage=stage,
        expected_checkpoint=checkpoint,
        expected_calibration=calibration,
    )
    if (SOURCE_ROOT / GLOBAL_HOLDOUT_SPEND_RELATIVE).exists():
        raise OrderedDecisionError("Global confirmation holdout is already spent")
    authorization = _centered_authorization_payload(
        purpose="confirmation", stage=stage, decision=decision
    )
    authorization["development_diagnostic"] = diagnostic
    without_identity = {
        name: value
        for name, value in authorization.items()
        if name != "authorization_identity"
    }
    authorization["authorization_identity"] = legacy._digest_json(without_identity)
    expected_out = diagnostic_authorization_path.resolve().parent / (
        "authorize_confirmation.json"
    )
    if out_file.resolve() != expected_out:
        raise OrderedDecisionError("Confirmation authorization output is noncanonical")
    return legacy._atomic_json_record(out_file.resolve(), authorization)


def spend_confirmation(
    *,
    authorization_path: Path,
    stage: str,
    marker_path: Path,
    allow_matching_existing: bool,
) -> dict[str, Any]:
    expected = (SOURCE_ROOT / GLOBAL_HOLDOUT_SPEND_RELATIVE).resolve()
    if marker_path.resolve() != expected:
        raise OrderedDecisionError(
            "Confirmation must use the single legacy-compatible global spend path"
        )
    authorization = verify_authorization(
        authorization_path, purpose="confirmation", stage=stage
    )
    without_identity = {
        "schema_name": HOLDOUT_SPEND_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "authorization_path": str(authorization_path.resolve()),
        "authorization_identity": authorization["authorization_identity"],
        "checkpoint_sha256": authorization["checkpoint"]["sha256"],
        "partition_digest": authorization["partition"]["partition_digest"],
        "confirmation_holdout_spent": True,
        "test_data_accessed": False,
    }
    payload = {
        **without_identity,
        "spend_identity": legacy._digest_json(without_identity),
    }
    try:
        return legacy._atomic_json_record(
            marker_path.resolve(),
            payload,
            allow_matching_existing=allow_matching_existing,
        )
    except OrderedDecisionError as error:
        if marker_path.exists():
            raise OrderedDecisionError(
                "Global confirmation spend marker belongs to another "
                "authorization or was modified"
            ) from error
        raise


def _centered_authorization_payload(
    *, purpose: str, stage: str, decision: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_name": AUTHORIZATION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "purpose": purpose,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "decision_identity": decision["decision_identity"],
        "checkpoint": decision["checkpoint"],
        "partition": decision["partition"],
        "run_launch_identity": decision["run_launch_identity"],
        "predecessor_authorization": decision.get("predecessor_authorization"),
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    return {**payload, "authorization_identity": legacy._digest_json(payload)}


def _decide_stage_validated(
    *,
    stage: str,
    run_dir: Path,
    config_path: Path,
    partition_dir: Path,
    memory_off_dir: Path,
    all_null_dir: Path,
    v2_memory_off_dir: Path,
    v2_config_path: Path,
    v2_checkpoint_path: Path,
    out_dir: Path,
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    predecessor_evidence = None
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage B requires Stage-A authorization")
        predecessor = verify_authorization(
            stage1_authorization, purpose="stage2", stage=STAGE1
        )
        predecessor_evidence = {
            "path": str(stage1_authorization.resolve()),
            "sha256": sha256_file(stage1_authorization),
            "authorization_identity": predecessor["authorization_identity"],
            "decision_identity": predecessor["decision_identity"],
        }
    elif stage1_authorization is not None:
        raise OrderedDecisionError("Stage A cannot consume predecessor authorization")

    partition = legacy._validate_partition(partition_dir)
    checkpoint, checkpoint_evidence, _selected_metrics = _validate_run(
        stage=stage, run_dir=run_dir, config_path=config_path
    )
    checkpoint_path = Path(str(checkpoint_evidence["path"])).resolve()
    launch_path = Path(f"{Path(run_dir).resolve()}.prerequisites") / (
        "run_launch_identity.json"
    )
    launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=stage1_authorization,
    )
    launch_source = dict(launch.get("source", {}) or {})
    execution_source = legacy._validate_shared_source_checkout(
        SOURCE_ROOT,
        expected_head=str(launch_source.get("git_head", "")),
        expected_remote_ref=str(launch_source.get("remote_ref", "")),
        expected_remote_head=str(launch_source.get("remote_head", "")),
    )
    if execution_source != launch_source:
        raise OrderedDecisionError(
            "Centered decision is not executing the immutable run source"
        )
    run_launch_evidence = {
        "path": str(launch_path.resolve()),
        "sha256": sha256_file(launch_path),
        "launch_identity": launch["launch_identity"],
        "source": launch_source,
        "slurm": launch["slurm"],
        "resume_attempts": checkpoint_evidence["resume_attempts"],
    }
    stage2_input_evidence = None
    if stage == STAGE2:
        assert stage1_authorization is not None
        stage2_input_path = Path(f"{Path(run_dir).resolve()}.prerequisites") / (
            "ordered_stage_input.json"
        )
        stage2_input = _validate_stage2_input(
            stage2_input_path,
            stage1_authorization=stage1_authorization,
            config_path=config_path,
        )
        stage2_input_evidence = {
            "path": str(stage2_input_path.resolve()),
            "sha256": sha256_file(stage2_input_path),
            "input_identity": stage2_input["input_identity"],
        }

    memory_off = _load_dev_export(
        memory_off_dir,
        expected_mode="off",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        partition=partition,
    )
    all_null = _load_dev_export(
        all_null_dir,
        expected_mode="all_null",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        partition=partition,
    )
    if sha256_file(v2_checkpoint_path) != legacy.EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Development parity used another v2 checkpoint")
    v2_memory_off = _load_dev_export(
        v2_memory_off_dir,
        expected_mode="not_applicable",
        checkpoint_path=v2_checkpoint_path,
        config_path=v2_config_path,
        partition=partition,
        require_factorized_control=False,
    )
    all_null_evidence = legacy._validate_development_all_null(memory_off, all_null)
    selected_v2_parity = legacy._validate_selected_v2_parity(memory_off, v2_memory_off)
    has_feasible = bool(checkpoint_evidence["has_feasible_checkpoint"])
    status = legacy.DEVELOPMENT_FEASIBLE if has_feasible else legacy.VALID_INFEASIBLE
    without_identity = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "status": status,
        "integrity_valid": True,
        "development_feasible": has_feasible,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
        "checkpoint": checkpoint_evidence,
        "partition": {
            "dir": str(partition["dir"]),
            "partition_digest": legacy.EXPECTED_PARTITION_DIGEST,
            "development_manifest": str(partition["development_manifest"]),
            "development_manifest_sha256": partition["development_manifest_sha256"],
            "development_rows": legacy.EXPECTED_DEVELOPMENT_ROWS,
            "development_unique_texts": legacy.EXPECTED_DEVELOPMENT_TEXTS,
        },
        "development_integrity": {
            "selected_checkpoint_replay": checkpoint_evidence[
                "selected_checkpoint_integrity_audit"
            ],
            "all_null_vs_memory_off": all_null_evidence,
            "selected_memory_off_vs_v2": selected_v2_parity,
            "memory_off_export_summary_sha256": sha256_file(memory_off["summary_path"]),
            "all_null_export_summary_sha256": sha256_file(all_null["summary_path"]),
            "v2_memory_off_export_summary_sha256": sha256_file(
                v2_memory_off["summary_path"]
            ),
            "v2_checkpoint_sha256": legacy.EXPECTED_V2_SHA256,
            "v2_config": {
                "path": str(v2_config_path.resolve()),
                "sha256": sha256_file(v2_config_path),
            },
        },
        "predecessor_authorization": predecessor_evidence,
        "stage2_launch_input": stage2_input_evidence,
        "run_launch_identity": run_launch_evidence,
    }
    decision = {
        **without_identity,
        "decision_identity": legacy._digest_json(without_identity),
    }
    purpose = "diagnostic" if has_feasible else "stage2"
    if stage == STAGE2 and not has_feasible:
        purpose = "stop"
    authorization = (
        _centered_authorization_payload(purpose=purpose, stage=stage, decision=decision)
        if purpose != "stop"
        else None
    )
    _write_centered_decision(out_dir, decision, authorization)
    return decision


def decide_stage(
    *,
    stage: str,
    run_dir: Path,
    config_path: Path,
    partition_dir: Path,
    memory_off_dir: Path,
    all_null_dir: Path,
    v2_memory_off_dir: Path,
    v2_config_path: Path,
    v2_checkpoint_path: Path,
    out_dir: Path,
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    """Make one ordered decision while preserving failed audits separately."""

    try:
        return _decide_stage_validated(
            stage=stage,
            run_dir=run_dir,
            config_path=config_path,
            partition_dir=partition_dir,
            memory_off_dir=memory_off_dir,
            all_null_dir=all_null_dir,
            v2_memory_off_dir=v2_memory_off_dir,
            v2_config_path=v2_config_path,
            v2_checkpoint_path=v2_checkpoint_path,
            out_dir=out_dir,
            stage1_authorization=stage1_authorization,
        )
    except OrderedDecisionError as error:
        without_identity = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "experiment_name": EXPERIMENTS.get(stage),
            "status": legacy.INTEGRITY_INVALID,
            "integrity_valid": False,
            "development_feasible": False,
            "error": str(error),
            "test_data_accessed": False,
            "confirmation_manifest_opened": False,
        }
        decision = {
            **without_identity,
            "decision_identity": legacy._digest_json(without_identity),
        }
        _write_centered_integrity_failure(out_dir, decision)
        return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--stage", choices=STAGES, required=True)
    decide.add_argument("--run_dir", type=Path, required=True)
    decide.add_argument("--config", type=Path, required=True)
    decide.add_argument("--partition_dir", type=Path, required=True)
    decide.add_argument("--memory_off_dir", type=Path, required=True)
    decide.add_argument("--all_null_dir", type=Path, required=True)
    decide.add_argument("--v2_memory_off_dir", type=Path, required=True)
    decide.add_argument("--v2_config", type=Path, required=True)
    decide.add_argument("--v2_checkpoint", type=Path, required=True)
    decide.add_argument("--out_dir", type=Path, required=True)
    decide.add_argument("--integrity_audit", type=Path, required=True)
    decide.add_argument("--stage1_authorization", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument(
        "--purpose", choices=("stage2", "diagnostic", "confirmation"), required=True
    )
    verify.add_argument("--stage", choices=STAGES, required=True)
    verify.add_argument("--authorization", type=Path, required=True)
    spend = sub.add_parser("spend-confirmation")
    spend.add_argument("--stage", choices=STAGES, required=True)
    spend.add_argument("--authorization", type=Path, required=True)
    spend.add_argument("--marker", type=Path, required=True)
    spend.add_argument("--allow_matching_existing", action="store_true")
    stage2 = sub.add_parser("record-stage2-input")
    stage2.add_argument("--authorization", type=Path, required=True)
    stage2.add_argument("--out_file", type=Path, required=True)
    stage2.add_argument("--config", type=Path, required=True)
    stage2.add_argument("--v2_checkpoint", type=Path, required=True)
    stage2.add_argument("--source_root", type=Path, required=True)
    stage2.add_argument("--source_git_head", required=True)
    stage2.add_argument("--source_remote_ref", required=True)
    stage2.add_argument("--source_remote_head", required=True)
    launch = sub.add_parser("record-run-launch")
    launch.add_argument("--stage", choices=STAGES, required=True)
    launch.add_argument("--out_file", type=Path, required=True)
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--v2_checkpoint", type=Path, required=True)
    launch.add_argument("--source_root", type=Path, required=True)
    launch.add_argument("--source_git_head", required=True)
    launch.add_argument("--source_remote_ref", required=True)
    launch.add_argument("--source_remote_head", required=True)
    launch.add_argument("--launcher", type=Path, action="append", required=True)
    launch.add_argument("--stage1_authorization", type=Path)
    resume = sub.add_parser("record-run-resume")
    resume.add_argument("--stage", choices=STAGES, required=True)
    resume.add_argument("--run_dir", type=Path, required=True)
    resume.add_argument("--out_file", type=Path, required=True)
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--partition_dir", type=Path, required=True)
    resume.add_argument("--last_checkpoint", type=Path, required=True)
    resume.add_argument("--source_root", type=Path, required=True)
    resume.add_argument("--source_git_head", required=True)
    resume.add_argument("--source_remote_ref", required=True)
    resume.add_argument("--source_remote_head", required=True)
    resume.add_argument("--launcher", type=Path, action="append", required=True)
    resume.add_argument("--stage1_authorization", type=Path)
    precheckpoint = sub.add_parser("record-precheckpoint-retry")
    precheckpoint.add_argument("--stage", choices=STAGES, required=True)
    precheckpoint.add_argument("--run_dir", type=Path, required=True)
    precheckpoint.add_argument("--out_file", type=Path, required=True)
    precheckpoint.add_argument("--config", type=Path, required=True)
    precheckpoint.add_argument("--partition_dir", type=Path, required=True)
    precheckpoint.add_argument("--lease", type=Path, required=True)
    precheckpoint.add_argument("--lease_attestation", type=Path, required=True)
    precheckpoint.add_argument("--source_root", type=Path, required=True)
    precheckpoint.add_argument("--source_git_head", required=True)
    precheckpoint.add_argument("--source_remote_ref", required=True)
    precheckpoint.add_argument("--source_remote_head", required=True)
    precheckpoint.add_argument("--stage1_authorization", type=Path)
    acquire = sub.add_parser("acquire-execution-lease")
    acquire.add_argument(
        "--purpose",
        choices=("calibration", "training", "decision", "diagnostic", "confirmation"),
        required=True,
    )
    acquire.add_argument("--stage", choices=LEASE_STAGES, required=True)
    acquire.add_argument("--lease", type=Path, required=True)
    acquire.add_argument("--out_file", type=Path, required=True)
    acquire.add_argument("--source_git_head", required=True)
    acquire.add_argument("--slurm_job_id", required=True)
    acquire.add_argument("--binding_identity", required=True)
    release = sub.add_parser("release-execution-lease")
    release.add_argument("--lease", type=Path, required=True)
    release.add_argument("--attestation", type=Path, required=True)
    completion = sub.add_parser("record-calibration-completion")
    completion.add_argument("--artifact_dir", type=Path, required=True)
    completion.add_argument("--source_root", type=Path, required=True)
    completion.add_argument("--source_git_head", required=True)
    completion.add_argument("--source_remote_ref", required=True)
    completion.add_argument("--source_remote_head", required=True)
    completion.add_argument("--lease", type=Path, required=True)
    completion.add_argument("--lease_attestation", type=Path, required=True)
    completion.add_argument("--out_file", type=Path, required=True)
    completion.add_argument("--publication_existing", action="store_true")
    finalize = sub.add_parser("finalize-confirmation-authorization")
    finalize.add_argument("--stage", choices=STAGES, required=True)
    finalize.add_argument("--diagnostic_authorization", type=Path, required=True)
    finalize.add_argument("--diagnostic_dir", type=Path, required=True)
    finalize.add_argument("--out_file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    global _ACTIVE_INTEGRITY_AUDIT
    args = parse_args()
    try:
        if args.command == "decide":
            _ACTIVE_INTEGRITY_AUDIT = args.integrity_audit.resolve()
            result = decide_stage(
                stage=args.stage,
                run_dir=args.run_dir,
                config_path=args.config,
                partition_dir=args.partition_dir,
                memory_off_dir=args.memory_off_dir,
                all_null_dir=args.all_null_dir,
                v2_memory_off_dir=args.v2_memory_off_dir,
                v2_config_path=args.v2_config,
                v2_checkpoint_path=args.v2_checkpoint,
                out_dir=args.out_dir,
                stage1_authorization=args.stage1_authorization,
            )
        elif args.command == "verify":
            result = verify_authorization(
                args.authorization, purpose=args.purpose, stage=args.stage
            )
        elif args.command == "spend-confirmation":
            result = spend_confirmation(
                authorization_path=args.authorization,
                stage=args.stage,
                marker_path=args.marker,
                allow_matching_existing=args.allow_matching_existing,
            )
        elif args.command == "record-stage2-input":
            result = record_stage2_input(
                authorization_path=args.authorization,
                out_file=args.out_file,
                stage2_config_path=args.config,
                v2_checkpoint_path=args.v2_checkpoint,
                source_root=args.source_root,
                source_git_head=args.source_git_head,
                source_remote_ref=args.source_remote_ref,
                source_remote_head=args.source_remote_head,
            )
        elif args.command == "record-run-launch":
            result = record_run_launch(
                stage=args.stage,
                out_file=args.out_file,
                config_path=args.config,
                v2_checkpoint_path=args.v2_checkpoint,
                source_root=args.source_root,
                source_git_head=args.source_git_head,
                source_remote_ref=args.source_remote_ref,
                source_remote_head=args.source_remote_head,
                launcher_paths=args.launcher,
                stage1_authorization=args.stage1_authorization,
            )
        elif args.command == "record-run-resume":
            result = record_run_resume(
                stage=args.stage,
                run_dir=args.run_dir,
                out_file=args.out_file,
                config_path=args.config,
                partition_dir=args.partition_dir,
                last_checkpoint_path=args.last_checkpoint,
                source_root=args.source_root,
                source_git_head=args.source_git_head,
                source_remote_ref=args.source_remote_ref,
                source_remote_head=args.source_remote_head,
                launcher_paths=args.launcher,
                stage1_authorization=args.stage1_authorization,
            )
        elif args.command == "record-precheckpoint-retry":
            result = record_precheckpoint_retry(
                stage=args.stage,
                run_dir=args.run_dir,
                out_file=args.out_file,
                config_path=args.config,
                partition_dir=args.partition_dir,
                lease_path=args.lease,
                lease_attestation_path=args.lease_attestation,
                source_root=args.source_root,
                source_git_head=args.source_git_head,
                source_remote_ref=args.source_remote_ref,
                source_remote_head=args.source_remote_head,
                stage1_authorization=args.stage1_authorization,
            )
        elif args.command == "acquire-execution-lease":
            result = acquire_execution_lease(
                lease_path=args.lease,
                attestation_path=args.out_file,
                purpose=args.purpose,
                stage=args.stage,
                source_git_head=args.source_git_head,
                slurm_job_id=args.slurm_job_id,
                binding_identity=args.binding_identity,
            )
        elif args.command == "release-execution-lease":
            result = release_execution_lease(
                lease_path=args.lease, attestation_path=args.attestation
            )
        elif args.command == "record-calibration-completion":
            result = record_calibration_completion(
                artifact_dir=args.artifact_dir,
                source_root=args.source_root,
                source_git_head=args.source_git_head,
                source_remote_ref=args.source_remote_ref,
                source_remote_head=args.source_remote_head,
                lease_path=args.lease,
                lease_attestation_path=args.lease_attestation,
                out_file=args.out_file,
                publication_existing=args.publication_existing,
            )
        elif args.command == "finalize-confirmation-authorization":
            result = finalize_confirmation_authorization(
                stage=args.stage,
                diagnostic_authorization_path=args.diagnostic_authorization,
                diagnostic_dir=args.diagnostic_dir,
                out_file=args.out_file,
            )
        else:  # pragma: no cover
            raise OrderedDecisionError(f"Unsupported command: {args.command}")
    finally:
        _ACTIVE_INTEGRITY_AUDIT = None
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "decide" and result.get("status") == legacy.INTEGRITY_INVALID:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
