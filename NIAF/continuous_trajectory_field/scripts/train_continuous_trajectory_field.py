from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import inspect
import math
import os
import random
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
from NIAF.continuous_sign_field.data import ExactDistributedEvalSampler
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
SENTENCE_MEMORY_EVAL_MODES = {
    "off",
    "on",
    "shuffled",
    "motion_shuffled",
    "analytic_prior",
    "motion_shuffled_n0",
    "motion_shuffled_n1",
    "motion_shuffled_n2",
    "cross_query_motion",
    "full_replacement",
    "joint_tuple_permuted",
    "uniform_final_mass",
    "association_disabled",
}
LEGACY_EVALUATION_CORRUPTION_MODE = "checkpoint_epoch_v1"
FIXED_EVALUATION_CORRUPTION_MODE = "fixed_query_condition_v1"
FIXED_EVIDENCE_CONTROLS_MODE = "fixed_evidence_controls_v1"
FIXED_EVIDENCE_CONTROL_NONCES = {
    "motion_shuffled_n0": "csl_daily_pair_derangement_v1_n0",
    "motion_shuffled_n1": "csl_daily_pair_derangement_v1_n1",
    "motion_shuffled_n2": "csl_daily_pair_derangement_v1_n2",
    "cross_query_motion": "csl_daily_cross_query_motion_v1",
    "full_replacement": "csl_daily_full_candidate_replacement_v1",
    "joint_tuple_permuted": "csl_daily_joint_tuple_permutation_v1",
}
CENTERED_CANDIDATE_VALUE_MODE = "centered_candidate_covariance_v1"
ABSOLUTE_ASSOCIATION_MODE = "absolute_text_motion_v1"
LEGACY_SELECTION_AGGREGATION = "sample_weighted_rows_v1"
CLUSTER_EQUAL_SELECTION_AGGREGATION = "normalized_text_cluster_equal_v1"
WORD_PRIOR_PART_NAMES = ("body", "left_hand", "right_hand", "face")
FACTORIZED_VALIDATION_PARTITION_ENV = (
    "SIGNTRAJ_VALIDATION_TEXT_PARTITION_DIR"
)
DEVELOPMENT_VALIDATION_MANIFEST = "manifest_development.jsonl"
DEVELOPMENT_VALIDATION_RUNTIME_SCHEMA = (
    "signtrajfield_development_validation_runtime"
)
STAGE_C_SCHEMA_NAME = "signtrajfield_centered_stage_c_generator_adaptation"
STAGE_C_SCHEMA_VERSION = 1
STAGE_C_ARM_TRAIN_MODES = {
    "memory": "dropout",
    "matched_off": "off",
}
STAGE_C_ARM_DROPOUT_PROBABILITIES = {
    "memory": 0.25,
    "matched_off": 1.0,
}
STAGE_C_EVALUATION_MODES = (
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
)
STAGE_C_GENERATOR_PREFIXES = (
    "hypernetwork.global_context.",
    "hypernetwork.coarse_head.",
    "hypernetwork.residual_head.",
    "hypernetwork.gate_head.",
    "hypernetwork.local_context.",
    "hypernetwork.local_head.",
    "hypernetwork.local_gate_head.",
)
STAGE_C_GENERATOR_PARAMETER_NAMES = (
    "hypernetwork.coarse_head.bias",
    "hypernetwork.coarse_head.weight",
    "hypernetwork.gate_head.bias",
    "hypernetwork.gate_head.weight",
    "hypernetwork.global_context.0.bias",
    "hypernetwork.global_context.0.weight",
    "hypernetwork.global_context.1.bias",
    "hypernetwork.global_context.1.weight",
    "hypernetwork.global_context.4.bias",
    "hypernetwork.global_context.4.weight",
    "hypernetwork.local_context.0.bias",
    "hypernetwork.local_context.0.weight",
    "hypernetwork.local_context.1.bias",
    "hypernetwork.local_context.1.weight",
    "hypernetwork.local_gate_head.bias",
    "hypernetwork.local_gate_head.weight",
    "hypernetwork.local_head.bias",
    "hypernetwork.local_head.weight",
    "hypernetwork.residual_head.bias",
    "hypernetwork.residual_head.weight",
)
STAGE_C_GENERATOR_PARAMETER_COUNT = 1_109_395
STAGE_C_OPTIMIZER_PARAMETER_GROUPS = (
    (
        "global",
        (
            "hypernetwork.global_context.0.weight",
            "hypernetwork.global_context.0.bias",
            "hypernetwork.global_context.1.weight",
            "hypernetwork.global_context.1.bias",
            "hypernetwork.global_context.4.weight",
            "hypernetwork.global_context.4.bias",
            "hypernetwork.coarse_head.weight",
            "hypernetwork.coarse_head.bias",
            "hypernetwork.residual_head.weight",
            "hypernetwork.residual_head.bias",
            "hypernetwork.gate_head.weight",
            "hypernetwork.gate_head.bias",
        ),
    ),
    (
        "local",
        (
            "hypernetwork.local_context.0.weight",
            "hypernetwork.local_context.0.bias",
            "hypernetwork.local_context.1.weight",
            "hypernetwork.local_context.1.bias",
            "hypernetwork.local_head.weight",
            "hypernetwork.local_head.bias",
            "hypernetwork.local_gate_head.weight",
            "hypernetwork.local_gate_head.bias",
        ),
    ),
)
STAGE_C_PINNED_SOURCE = {
    "source_checkpoint": {
        "path": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
            "absolute_binding_motion_contrast_v1/checkpoints/best_infeasible.pt"
        ),
        "sha256": "b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202",
        "selection_status": "best_infeasible",
        "epoch": 5,
        "global_step": 360,
    },
    "source_terminal_decision": {
        "path": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
            "absolute_binding_motion_contrast_v1/evaluation/"
            "ordered_development_decision/decision.json"
        ),
        "sha256": "8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5",
        "decision_identity": "7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69",
        "status": "valid_infeasible",
        "authorized_purpose": None,
    },
    "frozen_v2_teacher": {
        "path": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
        ),
        "sha256": "06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54",
    },
    "source_stage_b": {
        "config_path": (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
            "absolute_binding_motion_contrast_v1.yaml"
        ),
        "config_sha256": "7741da46d37a4b77f481663f25fa580281f6d29de6c2616baebcc30dac59b85e",
        "architecture_identity": "bc69fd35ac58e10bc894460c35175f13356b79416df40e8c57156b1929614236",
    },
}


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


def centered_sentence_memory_enabled(cfg):
    """Return whether the opt-in candidate-centered A''' path is active."""

    return str(
        cfg.get("sentence_memory", {}).get("candidate_value_mode", "")
    ).lower() == CENTERED_CANDIDATE_VALUE_MODE


def configured_sentence_memory_association(cfg):
    """Resolve the optional absolute text-motion association objective."""

    memory = dict(cfg.get("sentence_memory", {}) or {})
    configured = memory.get("association", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError("sentence_memory.association must be a mapping")
    mode = str(
        memory.get("association_mode", configured.get("mode", "none"))
    ).lower()
    if mode not in {"none", ABSOLUTE_ASSOCIATION_MODE}:
        raise ValueError(
            "sentence_memory.association_mode must be 'none' or "
            f"{ABSOLUTE_ASSOCIATION_MODE!r}"
        )
    # In Stage A the association branch and objective do not exist. Dormant
    # association hyperparameters therefore must neither be validated nor enter
    # its objective/architecture/resume identity.
    if mode == "none":
        return {"enabled": False, "mode": "none"}
    temperature = float(
        cfg.get("objective", {}).get(
            "association_temperature",
            memory.get(
                "association_temperature", configured.get("temperature", 0.10)
            ),
        )
    )
    descriptor_dim = int(
        memory.get("association_dim", configured.get("descriptor_dim", 128))
    )
    threshold_initial = float(
        memory.get(
            "association_threshold_initial",
            configured.get("threshold_initial", 0.0),
        )
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("sentence_memory.association.temperature must be positive")
    if descriptor_dim < 1:
        raise ValueError(
            "sentence_memory.association.descriptor_dim must be positive"
        )
    if not math.isfinite(threshold_initial) or not -1.0 < threshold_initial < 1.0:
        raise ValueError(
            "sentence_memory.association.threshold_initial must be finite and "
            "strictly between -1 and 1"
        )
    return {
        "enabled": True,
        "mode": mode,
        "temperature": temperature,
        "descriptor_dim": descriptor_dim,
        "threshold_initial": threshold_initial,
    }


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
    centered = centered_sentence_memory_enabled(cfg)
    if (
        not centered
        and not paired_sentence_memory_corruption_config(cfg)["enabled"]
    ):
        legacy_modes = {
            "off",
            "on",
            "shuffled",
            "motion_shuffled",
            "analytic_prior",
        }
        unsupported = sorted(set(modes) - legacy_modes)
        if unsupported:
            raise ValueError(
                "Non-centered sentence-memory evaluation without paired "
                "corruption supports only legacy modes "
                f"{sorted(legacy_modes)}, got unsupported modes {unsupported}"
            )
    if centered:
        expected = (
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
        if configured_sentence_memory_association(cfg)["enabled"]:
            expected = (*expected, "association_disabled")
        if modes != expected:
            raise ValueError(
                "Centered sentence-memory evaluation modes must exactly match "
                f"the ordered protocol list {list(expected)}, got {list(modes)}"
            )
    return modes


def sentence_memory_provider_required(modes):
    """Return whether resolved modes include retrieval-backed inference."""

    if isinstance(modes, str):
        modes = (modes,)
    return any(
        str(mode).lower()
        in (
            {
                "on",
                "shuffled",
                "motion_shuffled",
                "analytic_prior",
            }
            | (SENTENCE_MEMORY_EVAL_MODES - {"off"})
        )
        for mode in modes
    )


def configured_evaluation_corruption(cfg):
    """Resolve validation corruption without changing legacy configurations."""

    configured = cfg.get("eval", {}).get("evaluation_corruption")
    if configured is None:
        return {
            "mode": LEGACY_EVALUATION_CORRUPTION_MODE,
            "seed": int(cfg.get("seed", 1234)),
            "nonce": None,
        }
    if not isinstance(configured, dict):
        raise ValueError("eval.evaluation_corruption must be a mapping")
    unsupported = sorted(set(configured) - {"mode", "seed", "nonce", "nonces"})
    if unsupported:
        raise ValueError(
            "eval.evaluation_corruption has unsupported fields: "
            f"{unsupported}"
        )
    mode = str(configured.get("mode", "")).lower()
    if mode not in {
        LEGACY_EVALUATION_CORRUPTION_MODE,
        FIXED_EVALUATION_CORRUPTION_MODE,
        FIXED_EVIDENCE_CONTROLS_MODE,
    }:
        raise ValueError(
            "eval.evaluation_corruption.mode must be one of "
            f"{[LEGACY_EVALUATION_CORRUPTION_MODE, FIXED_EVALUATION_CORRUPTION_MODE, FIXED_EVIDENCE_CONTROLS_MODE]}, "
            f"got {mode!r}"
        )
    seed = int(configured.get("seed", cfg.get("seed", 1234)))
    nonce = configured.get("nonce")
    if mode == FIXED_EVIDENCE_CONTROLS_MODE:
        if seed != 1234:
            raise ValueError("fixed evidence controls require seed 1234")
        if nonce is not None:
            raise ValueError(
                "fixed evidence controls use named condition nonces; "
                "eval.evaluation_corruption.nonce must be absent"
            )
        configured_nonces = configured.get("nonces")
        if configured_nonces is not None and dict(configured_nonces) != (
            FIXED_EVIDENCE_CONTROL_NONCES
        ):
            raise ValueError(
                "fixed evidence-control nonces are protocol constants and may "
                "not be changed"
            )
        return {
            "mode": mode,
            "seed": seed,
            "nonce": None,
            "nonces": dict(FIXED_EVIDENCE_CONTROL_NONCES),
        }
    if configured.get("nonces") is not None:
        raise ValueError(
            "eval.evaluation_corruption.nonces is valid only for "
            "fixed_evidence_controls_v1"
        )
    if mode == FIXED_EVALUATION_CORRUPTION_MODE:
        if seed != 1234:
            raise ValueError(
                "fixed validation corruption requires seed 1234"
            )
        if nonce is None or not str(nonce):
            raise ValueError(
                "fixed validation corruption requires a non-empty versioned nonce"
            )
        nonce = str(nonce)
    elif nonce is not None:
        raise ValueError(
            "legacy checkpoint-epoch corruption must not configure a nonce"
        )
    return {"mode": mode, "seed": seed, "nonce": nonce}


def configured_selection_aggregation(cfg):
    configured = cfg.get("selection", {}).get("aggregation")
    if configured is None:
        return LEGACY_SELECTION_AGGREGATION
    mode = str(configured).lower()
    if mode not in {
        LEGACY_SELECTION_AGGREGATION,
        CLUSTER_EQUAL_SELECTION_AGGREGATION,
    }:
        raise ValueError(
            "selection.aggregation must be one of "
            f"{[LEGACY_SELECTION_AGGREGATION, CLUSTER_EQUAL_SELECTION_AGGREGATION]}, "
            f"got {mode!r}"
        )
    return mode


def _digest_named_identity(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def stage_c_trainability_contract():
    """Return the immutable schema-v1 generator payload contract."""

    return _digest_named_identity(
        {
            "schema_name": "signtrajfield_stage_c_trainability_contract",
            "schema_version": 1,
            "trainable_parameter_names": list(
                STAGE_C_GENERATOR_PARAMETER_NAMES
            ),
            "trainable_tensor_count": len(STAGE_C_GENERATOR_PARAMETER_NAMES),
            "trainable_parameter_count": STAGE_C_GENERATOR_PARAMETER_COUNT,
            "sentence_memory_frozen": True,
        }
    )


def stage_c_optimizer_parameter_mapping_contract():
    """Bind serialized optimizer parameter IDs to Stage-C tensor names."""

    return _digest_named_identity(
        {
            "schema_name": "signtrajfield_stage_c_optimizer_parameter_mapping",
            "schema_version": 1,
            "groups": [
                {
                    "group_index": index,
                    "group_name": group_name,
                    "parameter_names": list(parameter_names),
                }
                for index, (group_name, parameter_names) in enumerate(
                    STAGE_C_OPTIMIZER_PARAMETER_GROUPS
                )
            ],
            "mapping_rule": (
                "optimizer.param_groups[group_index].params[i] maps to "
                "groups[group_index].parameter_names[i]"
            ),
        }
    )


def sentence_memory_evaluation_control_identity(cfg):
    control = configured_evaluation_corruption(cfg)
    if control["mode"] == FIXED_EVIDENCE_CONTROLS_MODE:
        return _digest_named_identity(
            {
                "schema_name": "sentence_memory_evaluation_control",
                "schema_version": 2,
                "mode": control["mode"],
                "seed": int(control["seed"]),
                "condition_nonces": dict(control["nonces"]),
                "query_identity": "source_name_motion_path_or_index_v1",
                "condition_separation": "explicit_condition_token_v1",
                "motion_pair_controls": (
                    "three_fixed_valid_rank_cyclic_derangements_v1"
                ),
                "cross_query_motion": (
                    "normal_keys_plus_fixed_alternative_query_motion_v1"
                ),
                "full_replacement": "fixed_alternative_candidate_bundle_v1",
                "joint_tuple_permutation": (
                    "complete_valid_candidate_tuple_cyclic_permutation_v1"
                ),
                "joint_tuple_invariant": (
                    "global_max_prediction_and_duration_abs_at_most_1e-7"
                ),
                "uniform_final_mass": (
                    "same_structural_u_used_by_centering_v1"
                ),
                "broadcast_complete_motion_audit": {
                    "nonce": "csl_daily_broadcast_motion_payload_audit_v1",
                    "source_rule": (
                        "sha256_query_seed_nonce_select_one_valid_rank_then_"
                        "broadcast_tokens_mask_tau_part_validity_v1"
                    ),
                    "provider_reads": 0,
                    "expected": "prediction_and_duration_exact_memory_off",
                },
                "all_null_audit": {
                    "construction": (
                        "arbitrary_retrieved_payload_with_candidate_token_part_"
                        "masks_zero_and_candidate_ids_minus_one_v1"
                    ),
                    "provider_reads": 0,
                    "expected": (
                        "prediction_duration_gate_candidate_mass_exact_off_and_"
                        "null_mass_exact_one"
                    ),
                },
                "training_corruption": "epoch_dependent_unchanged",
            }
        )
    return _digest_named_identity(
        {
            "schema_name": "sentence_memory_evaluation_control",
            "schema_version": 1,
            "mode": control["mode"],
            "seed": int(control["seed"]),
            "nonce": control["nonce"],
            "motion_query_identity": "source_name_motion_path_or_index_v1",
            "full_query_identity": (
                "name_motion_path_semantic_source_and_source_group_v1"
            ),
            "condition_separation": "explicit_condition_token_v1",
            "motion_shuffle": "valid_rank_cyclic_derangement_v1",
            "full_shuffle": "stable_query_alternative_groups_v1",
            "training_corruption": "epoch_dependent_unchanged",
        }
    )


def sentence_memory_validation_corruption_map_identity(
    cfg, *, partition_digest, bank_id
):
    """Bind a deterministic validation map to its queries and motion bank."""

    control = sentence_memory_evaluation_control_identity(cfg)
    if control["mode"] == FIXED_EVIDENCE_CONTROLS_MODE:
        payload = {
            "schema_name": "sentence_memory_validation_corruption_map",
            "schema_version": 2,
            "evaluation_control_digest": control["digest"],
            "validation_partition_digest": str(partition_digest),
            "bank_id": str(bank_id),
            "conditions": list(FIXED_EVIDENCE_CONTROL_NONCES),
            "condition_nonces": dict(FIXED_EVIDENCE_CONTROL_NONCES),
            "query_identity": control["query_identity"],
        }
    else:
        payload = {
            "schema_name": "sentence_memory_validation_corruption_map",
            "schema_version": 1,
            "evaluation_control_digest": control["digest"],
            "validation_partition_digest": str(partition_digest),
            "bank_id": str(bank_id),
            "conditions": ["motion_shuffled", "shuffled"],
            "motion_query_identity": control["motion_query_identity"],
            "full_query_identity": control["full_query_identity"],
        }
    return _digest_named_identity(payload)


def sentence_memory_selection_aggregation_identity(cfg):
    mode = configured_selection_aggregation(cfg)
    if centered_sentence_memory_enabled(cfg):
        return _digest_named_identity(
            {
                "schema_name": "sentence_memory_selection_aggregation",
                "schema_version": 2,
                "mode": mode,
                "definition": {
                    "normalization": "NFKC_casefold_whitespace_collapse",
                    "metric_reduction": (
                        "mean_rows_within_text_then_equal_mean_texts"
                    ),
                    "distributed_partition": (
                        "whole_cluster_round_robin_no_padding_v1"
                    ),
                    "Rpair": (
                        "sqrt(mean_three_nonces(equal_text_mean(row_element_mse_"
                        "correct_vs_pair_nonce))/equal_text_mean(row_element_mse_"
                        "correct_vs_off))"
                    ),
                    "identity_utility_fraction": (
                        "(mean_nonce(pair_composite)-correct_composite)/(text_only_"
                        "composite-correct_composite); denominator>1e-8; invalid_"
                        "denominator_emits_value_0_and_valid_false"
                    ),
                    "association_matching": (
                        "within_correct_row_symmetric_K_to_V_and_V_to_K; equal_"
                        "semantic_groups_or_source_sign_variants_or_duplicate_"
                        "items_positive; all_excluded_from_negatives; anchors_"
                        "require_true_negative"
                    ),
                },
                "development_gates": {
                    "correct_relative_gain_over_off": 0.001,
                    "correct_relative_gain_over_mean_pair": 0.0025,
                    "correct_relative_gain_over_cross_query_motion": 0.0025,
                    "correct_relative_gain_over_full_replacement": 0.0025,
                    "correct_strictly_beats_each_pair_nonce": True,
                    "Rpair_minimum": 0.10,
                    "per_nonce_Rpair_minimum": 0.05,
                    "identity_utility_fraction_minimum": 0.25,
                    "identity_utility_denominator_minimum": 1e-8,
                    "hand_path_max_relative_degradation_vs_off": 0.02,
                    "correct_hand_path_no_worse_than_each_corruption": True,
                    "joint_tuple_prediction_max_abs": 1e-7,
                    "joint_tuple_duration_max_abs": 1e-7,
                    "uniform_final_prediction_and_duration_exact_off": True,
                    "broadcast_complete_prediction_and_duration_exact_off": True,
                    "all_null_prediction_duration_gate_mass_exact": True,
                    "stored_v2_prediction_and_duration_max_abs": 1e-7,
                    "stage_b_matching_top1_minimum": 0.25,
                    "stage_b_true_minus_best_negative_margin_strictly_positive": True,
                },
            }
        )
    if mode == CLUSTER_EQUAL_SELECTION_AGGREGATION:
        definition = {
            "normalization": "NFKC_casefold_whitespace_collapse",
            "metric_reduction": "mean_rows_within_text_then_equal_mean_texts",
            "distributed_partition": "whole_cluster_round_robin_no_padding_v1",
            "rmotion": (
                "sqrt(equal_text_mean(row_element_mse_correct_vs_motion) / "
                "equal_text_mean(row_element_mse_correct_vs_off))"
            ),
        }
    else:
        definition = {
            "metric_reduction": "sample_weighted_rows",
            "rmotion": "pooled_valid_elements",
        }
    return _digest_named_identity(
        {
            "schema_name": "sentence_memory_selection_aggregation",
            "schema_version": 1,
            "mode": mode,
            "definition": definition,
        }
    )


def sentence_memory_architecture_identity(cfg):
    """Describe the sentence-memory Q/K/V and temporal-prior contract."""

    memory = dict(cfg.get("sentence_memory", {}) or {})
    mode = str(memory.get("key_value_mode", "legacy_mixed_v1")).lower()
    if mode not in {"legacy_mixed_v1", "factorized_metadata_motion_v1"}:
        raise ValueError(f"Unsupported sentence_memory.key_value_mode {mode!r}")
    temporal_mode = str(memory.get("temporal_prior_mode", "none")).lower()
    if temporal_mode not in {"none", "gaussian"}:
        raise ValueError(
            "sentence_memory.temporal_prior_mode must be 'none' or 'gaussian'"
        )
    sigma = float(memory.get("temporal_prior_sigma", 0.25))
    scale = float(memory.get("temporal_prior_scale", 1.0))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sentence_memory.temporal_prior_sigma must be positive")
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "sentence_memory.temporal_prior_scale must be non-negative"
        )
    if mode == "legacy_mixed_v1":
        return _digest_named_identity(
            {
                "schema_name": "sentence_memory_architecture",
                "schema_version": 1,
                "key_value_mode": mode,
                "implementation": "joint_motion_metadata_memory_v1",
            }
        )

    duration_weight = max(float(memory.get("duration_weight", 0.10)), 0.0)
    temperature = max(float(memory.get("score_temperature", 0.10)), 1e-4)
    retrieval_scale = float(memory.get("retrieval_prior_scale", 1.0))
    token_prior = (
        "uniform over tokens satisfying motion_mask and part_validity>0, "
        "normalized separately for each batch-slot-part-candidate"
    )
    if temporal_mode == "gaussian":
        token_prior = (
            "log(part_validity) + temporal_prior_scale * "
            "(-0.5*((slot_tau-motion_tau)/temporal_prior_sigma)^2), "
            "normalized over motion tokens separately for each "
            "batch-slot-part-candidate"
        )
    payload = {
            "schema_name": "sentence_memory_architecture",
            "schema_version": 1,
            "key_value_mode": mode,
            "query": {
                "sources": ["frozen_target_text_slots", "trainable_part_embedding"],
                "construction": (
                    "bias_free_part_query_projection(text_slot)+part_embedding; "
                    "per-layer affine query layernorm/projection"
                ),
                "later_query_uses_state": False,
            },
            "key": {
                "sources": [
                    "candidate_mt5_mean_key",
                    "cosine_score",
                    "analytic_retrieval_probability",
                    "normalized_adjusted_rank",
                    "duration_log_gap",
                    "adjusted_top2_margin",
                ],
                "motion_content_enters_key": False,
                "adjusted_score": (
                    "cosine_score-duration_weight*abs(log("
                    "query_duration/candidate_duration))"
                ),
                "retrieval_probability": (
                    "masked_softmax(adjusted_score/score_temperature)"
                ),
                "rank": (
                    "count(valid adjusted scores greater)/max(valid_count-1,1)"
                ),
                "margin": "top1_minus_top2_adjusted_or_zero",
                "construction": (
                    "key_projection(mt5_key)+feature_projection([score,"
                    "retrieval_probability,rank,duration_gap,margin]); "
                    "per-layer key layernorm/projection"
                ),
            },
            "value": {
                "sources": ["vae_motion_mu"],
                "metadata_text_tau_or_validity_enters_value": False,
                "construction": (
                    "non_affine_layernorm_then_bias_free_linear; per-layer "
                    "non_affine_layernorm_then_bias_free_linear"
                ),
            },
            "attention": {
                "real_logit": (
                    "QK/sqrt(head_dim)+retrieval_prior_scale*log("
                    "retrieval_probability)+normalized_token_log_prior"
                ),
                "score_temperature": temperature,
                "duration_weight": duration_weight,
                "retrieval_prior_scale": retrieval_scale,
                "part_validity": (
                    "structurally valid iff motion_mask and candidate_mask and "
                    "part_validity>0; invalid logits are negative infinity"
                ),
                "temporal_prior_mode": temporal_mode,
                "temporal_prior_formula": token_prior,
                "temporal_prior_sigma": sigma,
                "temporal_prior_scale": scale,
                "all_invalid": (
                    "only null logit finite; null_mass=1 and state/gate/residual=0"
                ),
            },
            "zero_preserving_path": [
                "encoder_motion_layernorm_non_affine",
                "encoder_motion_projection_bias_free",
                "layer_value_layernorm_non_affine",
                "layer_value_and_output_projections_bias_free",
                "state_layernorm_non_affine_and_feedforward_bias_free",
                "sentence_part_projections_and_fusion_bias_free",
            ],
            "analytic_prior_mode": (
                "replace only real learned QK logits with exact zeros; retain "
                "retrieval/token priors, motion values, learned null QK/bias/"
                "confidence, attention dropout, state feed-forward, gates, "
                "part fusion, and frozen generator"
            ),
        }
    if centered_sentence_memory_enabled(cfg):
        if temporal_mode != "none":
            raise ValueError(
                "Centered candidate covariance requires temporal_prior_mode='none'"
            )
        association = configured_sentence_memory_association(cfg)
        if association["enabled"] and association["descriptor_dim"] != 128:
            raise ValueError(
                "Centered absolute association requires descriptor_dim=128"
            )
        relevance_mode = str(memory.get("relevance_gate_mode", "none")).lower()
        if relevance_mode != "frozen_absolute_adjusted_score_v1":
            raise ValueError(
                "Centered candidate covariance requires the frozen absolute "
                "adjusted-score relevance gate"
            )
        calibration_identity = memory.get(
            "resolved_relevance_calibration_identity"
        )
        if not isinstance(calibration_identity, dict):
            raise RuntimeError(
                "Centered sentence memory has no resolved relevance-calibration "
                "identity"
            )
        payload.update(
            {
                "schema_version": 3,
                "candidate_value_mode": CENTERED_CANDIDATE_VALUE_MODE,
                "candidate_value": {
                    "centering": (
                        "effective_candidate_summary_minus_exact_structural_"
                        "uniform_u_mean_after_last_nonlinearity_v1"
                    ),
                    "candidate_ids_required": True,
                    "candidate_ids_enter_learned_scores": False,
                    "structural_token_distribution": {
                        "support": (
                            "post_candidate_dropout_candidate_mask AND motion_mask "
                            "AND part_validity>0, separately per batch-slot-part-"
                            "candidate"
                        ),
                        "q": (
                            "q[s,p,k,u]=1/count_u(support[s,p,k,:]) on support; "
                            "otherwise 0"
                        ),
                    },
                    "per_layer_candidate_summary": (
                        "x[k,u]=P_motion(LN_0_motion(vae_motion_mu[k,u])); "
                        "v[l,h,k,u]=P_value[l,h](LN_0_value[l](x[k,u])); "
                        "m[l,h,s,p,k]=sum_u(q[s,p,k,u]*v[l,h,k,u])"
                    ),
                    "supported_candidate_set": (
                        "A[s,p]={k:any_u support[s,p,k,u]}; N[s,p]=|A[s,p]|; "
                        "u[s,p,k]=1/N[s,p] for k in A[s,p], else 0"
                    ),
                    "base_candidate_mass": (
                        "b[l,h,s,p,k]=sum_u(base_null_plus_real_softmax_mass"
                        "[l,h,s,p,k,u])"
                    ),
                    "evidence_application": {
                        "relevance": (
                            "g_rel[k]=sigmoid(relevance_slope*adjusted_score[k]+"
                            "relevance_intercept)"
                        ),
                        "association": (
                            "g_assoc[k]=sigmoid(absolute_association_logit[k]) "
                            "when enabled, else 1 on valid candidates"
                        ),
                        "final_mass": (
                            "a[l,h,s,p,k]=b[l,h,s,p,k]*g_rel[k]*g_assoc[k]"
                        ),
                        "placement": (
                            "multiply after base null+real softmax without "
                            "renormalizing surviving candidates"
                        ),
                        "rejection": "all rejected real mass is routed to null",
                    },
                    "rho": "rho[l,h,s,p]=sum_k(a[l,h,s,p,k])",
                    "reference": (
                        "m_ref[l,h,s,p]=m of the supported candidate with "
                        "minimum stable item ID"
                    ),
                    "residual": (
                        "r[l,h,s,p]=sum_{k in A[s,p]}((a[l,h,s,p,k]-"
                        "rho[l,h,s,p]*u[s,p,k])*(m[l,h,s,p,k]-"
                        "m_ref[l,h,s,p]))"
                    ),
                    "zero_or_one_supported_candidate": (
                        "N[s,p]<=1 gives null_mass=1, candidate_mass=0, "
                        "state=0, gate=0, residual=0 exactly"
                    ),
                    "uniform_control": (
                        "set final a=rho*u using the same exact structural u "
                        "tensor and set centered coefficients to exact zero"
                    ),
                    "broadcast_control": (
                        "identical complete motion payload gives identical m and "
                        "exact zero centered residual"
                    ),
                },
                "relevance_gate": {
                    "mode": relevance_mode,
                    "feature": "cosine-0.05*duration_log_gap",
                    "slope": float(memory["relevance_slope"]),
                    "intercept": float(memory["relevance_intercept"]),
                    "frozen": True,
                    "calibration_identity": calibration_identity,
                },
                "association": (
                    {
                        **association,
                        "descriptors": "l2_normalized_text_key_and_motion_v1",
                        "descriptor_formula": {
                            "eK": (
                                "l2_normalize(P_K(candidate_mt5_mean_key)); "
                                "P_K_is_bias_free"
                            ),
                            "eV": (
                                "l2_normalize(P_V(mask_normalized_mean_u("
                                "LN_0(vae_motion_mu[k,u])))); LN_0_is_non_affine; "
                                "P_V_is_bias_free"
                            ),
                            "motion_pool_mask": (
                                "binary_sentence_motion_mask_after_candidate_mask"
                            ),
                            "descriptor_dim": 128,
                            "projections_bias_free": True,
                            "excluded_from_descriptors": [
                                "target_text_slots",
                                "query_duration",
                                "retrieval_scores",
                                "candidate_durations",
                                "motion_tau",
                                "part_validity_magnitude",
                                "candidate_ids",
                            ],
                        },
                        "absolute_logit": (
                            "(cosine-tanh(trainable_threshold))/temperature"
                        ),
                        "threshold_parameterization": (
                            "tanh(trainable_scalar), strictly bounded to [-1,1]"
                        ),
                        "attention_gate": "sigmoid(absolute_logit)",
                    }
                    if association["enabled"]
                    else {"enabled": False, "mode": "none"}
                ),
                "attention_controls": {
                    "association_disabled": (
                        "remove_only_association_gate_keep_relevance_v1"
                    ),
                    "uniform_final_candidate_mass": (
                        "replace_all_learned_retrieval_evidence_redistribution_"
                        "with_exact_structural_u_v1"
                    ),
                },
            }
        )
    return _digest_named_identity(payload)


def validate_sentence_memory_architecture_identity(
    checkpoint, cfg, source="checkpoint"
):
    expected = sentence_memory_architecture_identity(cfg)
    return _validate_named_identity(
        checkpoint.get("sentence_memory_architecture_identity"),
        expected,
        source=source,
        label="sentence-memory architecture",
        allow_missing=(expected["key_value_mode"] == "legacy_mixed_v1"),
    )


def _requires_explicit_sentence_memory_control_identities(cfg):
    memory_mode = str(
        cfg.get("sentence_memory", {}).get(
            "key_value_mode", "legacy_mixed_v1"
        )
    ).lower()
    return bool(
        cfg.get("eval", {}).get("evaluation_corruption") is not None
        or cfg.get("selection", {}).get("aggregation") is not None
        or memory_mode == "factorized_metadata_motion_v1"
    )


def _validate_named_identity(actual, expected, *, source, label, allow_missing):
    if actual is None:
        if allow_missing:
            return expected
        raise RuntimeError(f"{source} has no persisted {label} identity")
    if not isinstance(actual, dict):
        raise RuntimeError(f"{source} has a malformed {label} identity")
    actual_payload = {key: value for key, value in actual.items() if key != "digest"}
    actual_digest = hashlib.sha256(
        json.dumps(actual_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if actual.get("digest") != actual_digest:
        raise RuntimeError(f"{source} has an invalid {label} identity digest")
    if actual_digest != expected["digest"]:
        raise RuntimeError(
            f"{source} {label} differs from the active config: "
            f"checkpoint={actual_digest}, active={expected['digest']}"
        )
    return expected


def validate_sentence_memory_evaluation_control_identity(
    checkpoint, cfg, source="checkpoint"
):
    expected = sentence_memory_evaluation_control_identity(cfg)
    return _validate_named_identity(
        checkpoint.get("sentence_memory_evaluation_control_identity"),
        expected,
        source=source,
        label="sentence-memory evaluation-control",
        allow_missing=(
            not _requires_explicit_sentence_memory_control_identities(cfg)
            and expected["mode"] == LEGACY_EVALUATION_CORRUPTION_MODE
        ),
    )


def validate_sentence_memory_selection_aggregation_identity(
    checkpoint, cfg, source="checkpoint"
):
    expected = sentence_memory_selection_aggregation_identity(cfg)
    return _validate_named_identity(
        checkpoint.get("sentence_memory_selection_aggregation_identity"),
        expected,
        source=source,
        label="sentence-memory selection-aggregation",
        allow_missing=(
            not _requires_explicit_sentence_memory_control_identities(cfg)
            and expected["mode"] == LEGACY_SELECTION_AGGREGATION
        ),
    )


def validate_sentence_memory_validation_corruption_map_identity(
    checkpoint, cfg, source="checkpoint"
):
    expected = cfg.get("validation_text_partition", {}).get(
        "evaluation_corruption_map_identity"
    )
    if expected is None:
        if _requires_explicit_sentence_memory_control_identities(cfg):
            raise RuntimeError(
                "Active factorized run has no resolved validation corruption-map "
                "identity"
            )
        return None
    return _validate_named_identity(
        checkpoint.get("sentence_memory_validation_corruption_map_identity"),
        expected,
        source=source,
        label="sentence-memory validation corruption-map",
        allow_missing=False,
    )


def sentence_memory_evaluation_corruption_kwargs(
    cfg, *, training, condition
):
    """Return fixed-control kwargs only for validation corruptions."""

    control = configured_evaluation_corruption(cfg)
    condition = str(condition).lower()
    if (
        not bool(training)
        and control["mode"] == FIXED_EVIDENCE_CONTROLS_MODE
        and condition in control["nonces"]
    ):
        return {
            "corruption_nonce": control["nonces"][condition],
            "corruption_seed": int(control["seed"]),
            "corruption_condition": condition,
        }
    if (
        bool(training)
        or condition not in {"shuffled", "motion_shuffled"}
        or control["mode"] != FIXED_EVALUATION_CORRUPTION_MODE
    ):
        return {}
    return {
        "corruption_nonce": control["nonce"],
        "corruption_seed": int(control["seed"]),
        "corruption_condition": condition,
    }


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


def resolve_sentence_memory_relevance_calibration(cfg):
    """Validate and bind the frozen absolute-relevance calibration.

    This runs before model construction.  Only plain scalar coefficients and a
    content identity are copied into the runtime config; the model never reads
    or discovers calibration artifacts itself.
    """

    memory = cfg.get("sentence_memory", {})
    mode = str(memory.get("relevance_gate_mode", "none")).lower()
    if mode in {"", "none"}:
        return None
    if mode != "frozen_absolute_adjusted_score_v1":
        raise ValueError(f"Unsupported sentence_memory.relevance_gate_mode {mode!r}")
    if not centered_sentence_memory_enabled(cfg):
        raise ValueError(
            "Frozen absolute relevance requires "
            "candidate_value_mode=centered_candidate_covariance_v1"
        )
    configured = memory.get("relevance_calibration")
    if not isinstance(configured, dict):
        raise ValueError(
            "sentence_memory.relevance_calibration must be a mapping"
        )
    expected_fields = {
        "artifact_dir",
        "artifact_identity",
        "schema_name",
        "schema_version",
        "minimum_heldout_auroc",
        "minimum_heldout_probability_gap",
    }
    unsupported = sorted(set(configured) - expected_fields)
    if unsupported:
        raise ValueError(
            "sentence_memory.relevance_calibration has unsupported fields: "
            f"{unsupported}"
        )
    if configured.get("schema_name") != (
        "signtrajfield_sentence_memory_relevance_calibration"
    ) or int(configured.get("schema_version", -1)) != 1:
        raise ValueError("Relevance-calibration schema differs from version 1")
    artifact_dir = configured.get("artifact_dir")
    if not artifact_dir:
        raise ValueError(
            "sentence_memory.relevance_calibration.artifact_dir is required"
        )
    minimum_auroc = float(configured.get("minimum_heldout_auroc", 0.75))
    minimum_gap = float(
        configured.get("minimum_heldout_probability_gap", 0.20)
    )
    if minimum_auroc < 0.75 or minimum_gap < 0.20:
        raise ValueError(
            "Relevance calibration may not weaken held-out AUROC 0.75 or "
            "probability-gap 0.20 gates"
        )
    from NIAF.continuous_trajectory_field.relevance_calibration import (
        canonical_json,
        sha256_file,
        validate_relevance_calibration_artifact,
    )

    calibration = validate_relevance_calibration_artifact(
        artifact_dir,
        expected_identity=configured.get("artifact_identity"),
        minimum_auroc=minimum_auroc,
        minimum_probability_gap=minimum_gap,
    )
    feature = dict(calibration.get("feature", {}) or {})
    duration_weight = float(memory.get("duration_weight", 0.10))
    if (
        feature.get("mode") != "absolute_adjusted_score_v1"
        or float(feature.get("duration_weight", math.nan)) != duration_weight
    ):
        raise RuntimeError(
            "Relevance calibration feature/duration contract differs from "
            "sentence-memory retrieval"
        )
    directory = Path(artifact_dir)
    ready = json.loads((directory / "READY").read_text(encoding="utf-8"))
    coefficients = dict(calibration.get("coefficients", {}) or {})
    try:
        parameter_a = float(coefficients["a"])
        parameter_b = float(coefficients["b"])
        slope = float(coefficients["slope"])
        intercept = float(coefficients["intercept"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Sealed relevance calibration lacks finite a/b and "
            "slope/intercept coefficients"
        ) from error
    formula = coefficients.get("formula")
    intercept_derivation = coefficients.get("intercept_derivation")
    if (
        not all(
            math.isfinite(value)
            for value in (parameter_a, parameter_b, slope, intercept)
        )
        or parameter_a <= 0.0
        or slope <= 0.0
        or parameter_a != slope
        or intercept != float(-parameter_a * parameter_b)
        or formula != "sigmoid(a*(adjusted_score-b))"
        or intercept_derivation != "float64(-a*b)"
    ):
        raise RuntimeError(
            "Sealed relevance-calibration coefficient parameterizations "
            "are not exactly equivalent"
        )
    resolved_identity_payload = {
        "schema_name": calibration["schema_name"],
        "schema_version": int(calibration["schema_version"]),
        "artifact_identity": calibration["identity"],
        "calibration_sha256": sha256_file(directory / "calibration.json"),
        "map_sha256": str(ready["map_sha256"]),
        "map_content_digest": str(calibration["map"]["content_digest"]),
        "heldout_auroc": float(calibration["holdout"]["auroc"]),
        "heldout_probability_gap": float(
            calibration["holdout"]["probability_gap"]
        ),
        "minimum_heldout_auroc": minimum_auroc,
        "minimum_heldout_probability_gap": minimum_gap,
        "coefficients": {
            "a": parameter_a,
            "b": parameter_b,
            "formula": formula,
            "intercept": intercept,
            "intercept_derivation": intercept_derivation,
            "slope": slope,
        },
    }
    resolved_identity = {
        **resolved_identity_payload,
        "digest": hashlib.sha256(
            canonical_json(resolved_identity_payload).encode("utf-8")
        ).hexdigest(),
    }
    for name, resolved in (
        ("relevance_slope", slope),
        ("relevance_intercept", intercept),
    ):
        existing = memory.get(name)
        if existing is not None and float(existing) != resolved:
            raise RuntimeError(
                f"Configured sentence_memory.{name} differs from sealed calibration"
            )
        memory[name] = resolved
    existing_identity = memory.get("resolved_relevance_calibration_identity")
    if existing_identity is not None and existing_identity != resolved_identity:
        raise RuntimeError(
            "Resolved relevance-calibration identity differs from sealed artifact"
        )
    memory["resolved_relevance_calibration_identity"] = resolved_identity
    return resolved_identity


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
    if str(memory_cfg.get("key_value_mode", "legacy_mixed_v1")).lower() == (
        "factorized_metadata_motion_v1"
    ):
        payload["sentence_memory_architecture_digest"] = (
            sentence_memory_architecture_identity(cfg)["digest"]
        )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "digest": digest}


_PAIRED_SENTENCE_OBJECTIVE_WEIGHTS = (
    "lambda_sentence_benefit",
    "lambda_sentence_motion_rank",
    "lambda_sentence_motion_fallback",
    "lambda_sentence_full_shuffle_rank",
    "lambda_sentence_full_shuffle_fallback",
)


def paired_sentence_memory_corruption_config(cfg):
    """Resolve and validate the opt-in paired Phase-A objective contract."""

    safety_cfg = cfg.get("sentence_memory_safety", {})
    paired_cfg = safety_cfg.get("paired_corruption", {})
    if paired_cfg is None:
        paired_cfg = {}
    if not isinstance(paired_cfg, dict):
        raise ValueError("sentence_memory_safety.paired_corruption must be a mapping")
    enabled = bool(paired_cfg.get("enabled", False))
    resolved = {
        "enabled": enabled,
        "full_shuffle_probability": float(
            paired_cfg.get("full_shuffle_probability", 0.10)
        ),
        "benefit_margin_relative": float(
            paired_cfg.get("benefit_margin_relative", 0.001)
        ),
        "ranking_margin_relative": float(
            paired_cfg.get("ranking_margin_relative", 0.005)
        ),
        "detach_corrupt_ranking": bool(
            paired_cfg.get("detach_corrupt_ranking", True)
        ),
        "fallback_huber_beta": float(
            paired_cfg.get("fallback_huber_beta", 0.10)
        ),
    }
    for name in ("benefit_margin_relative", "ranking_margin_relative"):
        if not math.isfinite(resolved[name]) or resolved[name] < 0.0:
            raise ValueError(
                f"sentence_memory_safety.paired_corruption.{name} must be "
                "finite and non-negative"
            )
    probability = resolved["full_shuffle_probability"]
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "sentence_memory_safety.paired_corruption."
            "full_shuffle_probability must be in [0, 1]"
        )
    if (
        not math.isfinite(resolved["fallback_huber_beta"])
        or resolved["fallback_huber_beta"] <= 0.0
    ):
        raise ValueError(
            "sentence_memory_safety.paired_corruption.fallback_huber_beta "
            "must be finite and positive"
        )
    if enabled and not resolved["detach_corrupt_ranking"]:
        raise ValueError(
            "The paired sentence-memory objective requires "
            "detach_corrupt_ranking=true so ranking cannot reward corrupt-output "
            "degradation"
        )
    return resolved


def validate_paired_sentence_memory_training_contract(cfg):
    """Fail closed unless paired corruption is a frozen, text-only Phase-A run."""

    paired_cfg = paired_sentence_memory_corruption_config(cfg)
    if not paired_cfg["enabled"]:
        return paired_cfg
    if not is_sentence_memory_model(cfg) or not sentence_memory_enabled(cfg):
        raise ValueError("Paired corruption requires an enabled v3 sentence-memory model")
    if not bool(cfg.get("sentence_memory_safety", {}).get("enabled", False)):
        raise ValueError("Paired corruption requires sentence_memory_safety.enabled=true")
    if configured_sentence_memory_train_mode(cfg) != "on":
        raise ValueError(
            "Paired corruption requires conditioning.sentence_memory_train_mode='on' "
            "so every usable row has a correct retrieval"
        )
    if configured_word_prior_train_mode(cfg) != "off":
        raise ValueError(
            "Paired corruption requires conditioning.word_prior_train_mode='off' "
            "for a strict text-only teacher"
        )
    train_cfg = cfg.get("train", {})
    if not bool(train_cfg.get("freeze_base", False)):
        raise ValueError("Paired corruption Phase A requires train.freeze_base=true")
    if train_cfg.get("unfreeze_base_prefixes"):
        raise ValueError(
            "Paired corruption Phase A requires train.unfreeze_base_prefixes=[]"
        )
    if bool(
        cfg.get("sentence_memory_safety", {})
        .get("phase_b", {})
        .get("enabled", False)
    ):
        raise ValueError("Paired corruption is a Phase-A-only training objective")
    if float(cfg.get("model", {}).get("dropout", 0.0)) != 0.0:
        raise ValueError(
            "Paired corruption requires model.dropout=0 for a deterministic "
            "memory-off teacher"
        )
    epochs = int(train_cfg.get("epochs", 0))
    maximum_epochs = 6 if centered_sentence_memory_enabled(cfg) else 4
    if epochs < 1 or epochs > maximum_epochs:
        raise ValueError(
            "Paired corruption requires an explicit train.epochs in "
            f"[1, {maximum_epochs}]; this mechanism experiment must never "
            f"train after epoch {maximum_epochs}"
        )
    objective_cfg = cfg.get("objective", {})
    for name in _PAIRED_SENTENCE_OBJECTIVE_WEIGHTS:
        value = float(objective_cfg.get(name, 1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"objective.{name} must be finite and non-negative")
    sparsity_weight = float(
        objective_cfg.get("lambda_sentence_sparsity", 1e-4)
    )
    if not math.isfinite(sparsity_weight) or sparsity_weight < 0.0:
        raise ValueError(
            "objective.lambda_sentence_sparsity must be finite and non-negative"
        )
    if centered_sentence_memory_enabled(cfg):
        if configured_selection_aggregation(cfg) != (
            CLUSTER_EQUAL_SELECTION_AGGREGATION
        ):
            raise ValueError(
                "Centered sentence-memory experiments require normalized-text "
                "cluster-equal checkpoint selection"
            )
        if configured_evaluation_corruption(cfg)["mode"] != (
            FIXED_EVIDENCE_CONTROLS_MODE
        ):
            raise ValueError(
                "Centered sentence-memory experiments require fixed evidence "
                "evaluation controls"
            )
        configured_sentence_memory_eval_modes(cfg)
        if bool(cfg.get("selection", {}).get("require_feasible", False)):
            if int(train_cfg.get("early_stopping_patience", -1)) != 2 or int(
                train_cfg.get(
                    "early_stopping_min_epochs",
                    train_cfg.get("early_stopping_min_epoch", -1),
                )
            ) != 3:
                raise ValueError(
                    "Centered production runs require patience two and minimum "
                    "epoch three"
                )
        memory_cfg = cfg.get("sentence_memory", {})
        if str(memory_cfg.get("key_value_mode", "")).lower() != (
            "factorized_metadata_motion_v1"
        ):
            raise ValueError(
                "Centered candidate covariance requires factorized metadata/motion K/V"
            )
        if str(memory_cfg.get("temporal_prior_mode", "none")).lower() != "none":
            raise ValueError(
                "Centered candidate covariance does not permit a temporal prior"
            )
        association = configured_sentence_memory_association(cfg)
        if association["enabled"]:
            bce_weight = float(
                objective_cfg.get("lambda_sentence_association_bce", math.nan)
            )
            infonce_weight = float(
                objective_cfg.get(
                    "lambda_sentence_association_infonce", math.nan
                )
            )
            if (
                bce_weight != 0.10
                or infonce_weight != 0.05
                or association["temperature"] != 0.10
            ):
                raise ValueError(
                    "Absolute association requires BCE weight 0.10, InfoNCE "
                    "weight 0.05, and temperature 0.10"
                )
    return paired_cfg


def sentence_memory_objective_identity(cfg):
    """Describe the training-only sentence-memory objective independently."""

    paired_cfg = paired_sentence_memory_corruption_config(cfg)
    association = configured_sentence_memory_association(cfg)
    if stage_c_enabled(cfg):
        stage_c = configured_stage_c(cfg)
        objective_cfg = cfg.get("objective", {})
        safety_cfg = cfg.get("sentence_memory_safety", {})
        train_cfg = cfg.get("train", {})
        provenance = stage_c.get("resolved_source_provenance") or {}
        teacher_reference = provenance.get("teacher_validation_reference")
        payload = {
            "schema_version": 4,
            "mode": "centered_stage_c_generator_adaptation_v1",
            "arm": stage_c.get("arm"),
            "sentence_memory_train_mode": configured_sentence_memory_train_mode(
                cfg
            ),
            "sentence_memory_dropout_probability": float(
                cfg.get("conditioning", {}).get(
                    "sentence_memory_dropout_probability", math.nan
                )
            ),
            "paired_corruption": {"enabled": False},
            "weights": {
                "lambda_sentence_safe": float(
                    objective_cfg.get("lambda_sentence_safe", 1.0)
                ),
                "lambda_sentence_shuffle": float(
                    objective_cfg.get("lambda_sentence_shuffle", 1.0)
                ),
                "lambda_sentence_sparsity": float(
                    objective_cfg.get("lambda_sentence_sparsity", 1e-4)
                ),
                "lambda_sentence_off_distill": float(
                    objective_cfg.get("lambda_sentence_off_distill", 1.0)
                ),
            },
            "safety": {
                "nonregression_margin": float(
                    safety_cfg.get("nonregression_margin", 0.0)
                ),
                "shuffle_probability": float(
                    safety_cfg.get("shuffle_probability", 0.25)
                ),
                "shuffle_huber_beta": float(
                    safety_cfg.get("shuffle_huber_beta", 0.10)
                ),
            },
            "off_distillation": {
                "teacher_checkpoint_sha256": stage_c.get(
                    "frozen_v2_teacher", {}
                ).get("sha256"),
                "huber_beta": float(stage_c.get("huber_beta", 0.10)),
                "student_path": "differentiable_memory_off_same_ddp_forward_v1",
                "teacher_path": "frozen_v2_no_grad_v1",
                "validation_guard": (
                    "live_memory_off_vs_exact_parity_stage_b_source_on_same_"
                    "sealed_cluster_equal_development_v1"
                ),
                "teacher_validation_reference": copy.deepcopy(
                    teacher_reference
                ),
                "text_only_max_relative_degradation": float(
                    stage_c.get("text_only_max_relative_degradation", 0.005)
                ),
            },
            "trainability": {
                "freeze_base": bool(train_cfg.get("freeze_base", False)),
                "freeze_sentence_memory": bool(
                    train_cfg.get("freeze_sentence_memory", False)
                ),
                "generator_prefixes": list(
                    train_cfg.get("unfreeze_base_prefixes", ())
                ),
                "tensor_contract": stage_c_trainability_contract(),
                "optimizer_parameter_mapping": (
                    stage_c_optimizer_parameter_mapping_contract()
                ),
            },
            "optimizer": {
                "joint_global_lr": float(
                    train_cfg.get("joint_global_lr", math.nan)
                ),
                "joint_local_lr": float(
                    train_cfg.get("joint_local_lr", math.nan)
                ),
                "weight_decay": float(
                    train_cfg.get("weight_decay", math.nan)
                ),
                "grad_clip": float(train_cfg.get("grad_clip", math.nan)),
                "local_warmup_epochs": int(
                    train_cfg.get("local_warmup_epochs", -1)
                ),
            },
            "bounded_memory_batches": {
                "seed": int(cfg.get("seed", -1)),
                "train_max_samples": int(
                    train_cfg.get("max_samples_per_memory_batch", -1)
                ),
                "train_max_frames": int(
                    train_cfg.get("max_frames_per_memory_batch", -1)
                ),
                "validation_max_samples": int(
                    cfg.get("eval", {}).get(
                        "max_samples_per_memory_batch", -1
                    )
                ),
                "validation_max_frames": int(
                    cfg.get("eval", {}).get(
                        "max_frames_per_memory_batch", -1
                    )
                ),
                "length_bucketed_batches": bool(
                    train_cfg.get("length_bucketed_batches", False)
                ),
                "drop_last": bool(train_cfg.get("drop_last", True)),
            },
            "validation_checkpoint_schedule": {
                "val_every": int(train_cfg.get("val_every", -1)),
                "save_every": int(train_cfg.get("save_every", -1)),
            },
            "distribution_contract": copy.deepcopy(
                stage_c.get("resolved_distribution_contract")
            ),
            "authorization": {
                "development_only": True,
                "non_authorizing": True,
                "promotion_eligible": False,
            },
        }
    elif (
        paired_cfg["enabled"]
        and centered_sentence_memory_enabled(cfg)
        and association["enabled"]
    ):
        objective_cfg = cfg.get("objective", {})
        payload = {
            "schema_version": 3,
            "mode": "paired_centered_absolute_association_v1",
            "paired_corruption": paired_cfg,
            "weights": {
                **{
                    name: float(objective_cfg.get(name, 1.0))
                    for name in _PAIRED_SENTENCE_OBJECTIVE_WEIGHTS
                },
                "lambda_sentence_sparsity": float(
                    objective_cfg.get("lambda_sentence_sparsity", 1e-4)
                ),
                "lambda_sentence_association_bce": float(
                    objective_cfg.get("lambda_sentence_association_bce", 0.10)
                ),
                "lambda_sentence_association_infonce": float(
                    objective_cfg.get(
                        "lambda_sentence_association_infonce", 0.05
                    )
                ),
            },
            "association": association,
            "formula": {
                "ordinary_and_pair_losses": "paired_correct_motion_or_full_v1",
                "absolute_bce": (
                    "class_balanced_global_means; correct_diagonal_positive; "
                    "motion_deranged_diagonal_negative; full_internal_diagonal_"
                    "positive; original_key_plus_full_motion_negative"
                ),
                "infonce": (
                    "symmetric_K_to_V_and_V_to_K_masked_multi_positive_v1"
                ),
                "infonce_pool": (
                    "all_valid_correct_candidates_on_each_DDP_rank; equal_"
                    "semantic_group_positive; different_group_negative"
                ),
                "identity_masking": (
                    "semantic_group_index_from_bank_item_group_ids_is_multi_"
                    "positive; source_sign_variant_index_is_sorted_unique_"
                    "canonical_source_group_id_and_is_multi_positive; duplicate_"
                    "concrete_bank_item_id_is_multi_positive; all_are_excluded_"
                    "from_negatives"
                ),
                "infonce_anchor": "requires_positive_and_true_negative",
                "distributed_reduction": (
                    "global_exact_differentiable_numerator_and_valid_count_v1"
                ),
                "training_corruption_schedule": (
                    "query_identity_epoch_seed_v1; evaluation controls excluded"
                ),
            },
        }
    elif paired_cfg["enabled"]:
        objective_cfg = cfg.get("objective", {})
        payload = {
            "schema_version": 2,
            "mode": "paired_correct_motion_or_full_v1",
            "paired_corruption": paired_cfg,
            "weights": {
                **{
                    name: float(objective_cfg.get(name, 1.0))
                    for name in _PAIRED_SENTENCE_OBJECTIVE_WEIGHTS
                },
                "lambda_sentence_sparsity": float(
                    objective_cfg.get("lambda_sentence_sparsity", 1e-4)
                ),
            },
            "formula": {
                "part_error": "masked_compact_l1_body_left_hand_right_hand_face",
                "denominator": "stop_gradient(text_off_error).clamp_min(1e-6)",
                "benefit": "relu((correct-text_off)/denominator+margin)",
                "ranking": "relu((correct-stop_gradient(corrupt))/denominator+margin)",
                "fallback": "partwise_masked_smooth_l1(corrupt,text_off)",
                "reduction": "masked_sum_divided_by_batch_times_four_parts",
                "ordinary_losses": "correct_only",
                "motion_shuffle_query_identity": "name_and_motion_path_index_fallback_v1",
                "training_corruption_schedule": (
                    "query_identity_epoch_seed_v1; evaluation controls excluded"
                ),
            },
        }
    else:
        payload = {
            "schema_version": 1,
            "mode": "legacy_replacement_v1",
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def validate_sentence_memory_objective_identity(checkpoint, cfg, source="checkpoint"):
    """Bind exact resume to the implemented training-objective semantics."""

    expected = sentence_memory_objective_identity(cfg)
    actual = checkpoint.get("sentence_memory_objective_identity")
    if actual is None and int(expected.get("schema_version", 0)) >= 2:
        raise RuntimeError(
            f"{source} has no persisted sentence-memory objective identity; "
            "protected sentence-memory resume requires the named schema "
            "produced by the same trainer implementation"
        )
    if actual is None and isinstance(checkpoint.get("config"), dict):
        # Pre-identity legacy checkpoints can be reconstructed without weakening
        # a paired resume: reconstruction is permitted only for the original,
        # paired-disabled objective.
        actual = sentence_memory_objective_identity(checkpoint["config"])
    if not isinstance(actual, dict):
        raise RuntimeError(f"{source} has no sentence-memory objective identity")
    actual_payload = {key: value for key, value in actual.items() if key != "digest"}
    actual_digest = hashlib.sha256(
        json.dumps(actual_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if actual.get("digest") != actual_digest:
        raise RuntimeError(f"{source} has an invalid sentence-memory objective digest")
    if actual_digest != expected["digest"]:
        raise RuntimeError(
            f"{source} sentence-memory objective differs from the active config: "
            f"checkpoint={actual_digest}, active={expected['digest']}"
        )
    return expected


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

    if (
        centered_sentence_memory_enabled(cfg)
        and not configured_sentence_memory_association(cfg)["enabled"]
    ):
        dormant_objective = payload.get("objective")
        if isinstance(dormant_objective, dict):
            for name in (
                "lambda_sentence_association_bce",
                "lambda_sentence_association_infonce",
                "association_temperature",
            ):
                dormant_objective.pop(name, None)

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
        train_cfg.pop("stage_c_warm_start_checkpoint", None)
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
        relevance_cfg = memory_cfg.get("relevance_calibration")
        if isinstance(relevance_cfg, dict):
            # Placement is operational; the resolved sealed content identity
            # and copied coefficients remain in the exact-resume digest.
            relevance_cfg.pop("artifact_dir", None)

    safety_cfg = payload.get("sentence_memory_safety")
    if isinstance(safety_cfg, dict):
        safety_cfg.pop("v2_to_v3_text_only_parity", None)
        phase_b_cfg = safety_cfg.get("phase_b")
        if isinstance(phase_b_cfg, dict):
            # The accepted report is validated through its own canonical digest.
            phase_b_cfg.pop("gate_report", None)
            phase_b_cfg.pop("resolved_scientific_gate", None)
        stage_c_cfg = safety_cfg.get("stage_c")
        if isinstance(stage_c_cfg, dict):
            # Runtime provenance is validated independently and contains file
            # placement plus the teacher's measured validation score.
            stage_c_cfg.pop("resolved_source_provenance", None)
            stage_c_cfg.pop("teacher_validation_selection_score", None)
            stage_c_cfg.pop("resolved_distribution_contract", None)

    if str(
        cfg.get("sentence_memory", {}).get(
            "key_value_mode", "legacy_mixed_v1"
        )
    ).lower() == "factorized_metadata_motion_v1":
        payload["_sentence_memory_architecture_digest"] = (
            sentence_memory_architecture_identity(cfg)["digest"]
        )
        payload["_sentence_memory_evaluation_control_digest"] = (
            sentence_memory_evaluation_control_identity(cfg)["digest"]
        )
        payload["_sentence_memory_selection_aggregation_digest"] = (
            sentence_memory_selection_aggregation_identity(cfg)["digest"]
        )

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


def configured_stage_c(cfg):
    """Return the explicit Stage-C contract without enabling it implicitly."""

    configured = cfg.get("sentence_memory_safety", {}).get("stage_c", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError("sentence_memory_safety.stage_c must be a mapping")
    return configured


def stage_c_enabled(cfg):
    return bool(
        is_sentence_memory_model(cfg)
        and configured_stage_c(cfg).get("enabled", False)
    )


def _stage_c_declared_contract(stage_c):
    """Select the immutable, user-declared portion of a Stage-C contract."""

    names = (
        "schema_name",
        "schema_version",
        "development_only",
        "non_authorizing",
        "promotion_eligible",
        "arm",
        "source_checkpoint",
        "source_terminal_decision",
        "frozen_v2_teacher",
        "source_stage_b",
        "active_stage_c",
        "huber_beta",
        "text_only_max_relative_degradation",
    )
    return {name: copy.deepcopy(stage_c.get(name)) for name in names}


def _require_exact_mapping(value, *, label, fields):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    missing = sorted(set(fields) - set(value))
    unsupported = sorted(set(value) - set(fields))
    if missing or unsupported:
        raise ValueError(
            f"{label} must have exactly fields {sorted(fields)}; "
            f"missing={missing}, unsupported={unsupported}"
        )
    return value


def _require_exact_number(mapping, name, expected, *, label):
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Stage-C schema v1 requires {label}.{name}={expected}"
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed != float(expected):
        raise ValueError(
            f"Stage-C schema v1 requires {label}.{name}={expected}"
        )
    return parsed


def configured_sentence_off_distillation(cfg):
    """Resolve the one training phase allowed to use the frozen-v2 teacher."""

    safety = cfg.get("sentence_memory_safety", {})
    phase_b = safety.get("phase_b", {})
    phase_b_active = bool(
        isinstance(phase_b, dict) and phase_b.get("enabled", False)
    )
    stage_c = configured_stage_c(cfg)
    stage_c_active = bool(stage_c.get("enabled", False))
    if phase_b_active and stage_c_active:
        raise ValueError("Phase B and Stage C may not be enabled together")
    if phase_b_active:
        return "phase_b", phase_b
    if stage_c_active:
        return "stage_c", stage_c
    return None, None


def validate_stage_c_training_contract(
    cfg, *, stage_c_warm_start=None, resume=None
):
    """Validate the development-only generator-adaptation contract.

    This contract is deliberately separate from canonical Phase B. It cannot
    authorize confirmation/test access or promotion, and its two arms differ
    only in whether sentence memory is sampled during training.
    """

    stage_c = configured_stage_c(cfg)
    enabled = bool(stage_c.get("enabled", False))
    if not enabled:
        if stage_c_warm_start is not None:
            raise ValueError(
                "--stage_c_warm_start requires sentence_memory_safety."
                "stage_c.enabled=true"
            )
        return None
    supported = {
        "enabled",
        "schema_name",
        "schema_version",
        "development_only",
        "non_authorizing",
        "promotion_eligible",
        "arm",
        "source_checkpoint",
        "source_terminal_decision",
        "frozen_v2_teacher",
        "source_stage_b",
        "active_stage_c",
        "huber_beta",
        "text_only_max_relative_degradation",
        # Runtime-only fields are persisted in resolved configs/checkpoints and
        # omitted from the exact resume identity below.
        "teacher_validation_selection_score",
        "resolved_source_provenance",
        "resolved_distribution_contract",
    }
    unsupported = sorted(set(stage_c) - supported)
    if unsupported:
        raise ValueError(
            "sentence_memory_safety.stage_c has unsupported fields: "
            f"{unsupported}"
        )
    required = supported - {
        "teacher_validation_selection_score",
        "resolved_source_provenance",
        "resolved_distribution_contract",
    }
    missing = sorted(required - set(stage_c))
    if missing:
        raise ValueError(
            "sentence_memory_safety.stage_c is missing required fields: "
            f"{missing}"
        )
    if stage_c.get("schema_name") != STAGE_C_SCHEMA_NAME or int(
        stage_c.get("schema_version", -1)
    ) != STAGE_C_SCHEMA_VERSION:
        raise ValueError(
            "Stage C requires schema "
            f"{STAGE_C_SCHEMA_NAME!r}/v{STAGE_C_SCHEMA_VERSION}"
        )
    for name, expected in (
        ("development_only", True),
        ("non_authorizing", True),
        ("promotion_eligible", False),
    ):
        if stage_c.get(name) is not expected:
            raise ValueError(f"Stage C requires {name}={expected!r}")
    arm = str(stage_c.get("arm", ""))
    if arm not in STAGE_C_ARM_TRAIN_MODES:
        raise ValueError(
            f"Stage C arm must be one of {sorted(STAGE_C_ARM_TRAIN_MODES)}"
        )
    if stage_c_warm_start is None and resume is None:
        raise ValueError(
            "Stage C must start with --stage_c_warm_start or continue with "
            "--resume"
        )
    if stage_c_warm_start is not None and resume is not None:
        raise ValueError("Stage-C warm start and exact resume are mutually exclusive")

    source_checkpoint = _require_exact_mapping(
        stage_c.get("source_checkpoint"),
        label="sentence_memory_safety.stage_c.source_checkpoint",
        fields={"path", "sha256", "selection_status", "epoch", "global_step"},
    )
    if not str(source_checkpoint.get("path", "")):
        raise ValueError("Stage-C source checkpoint path must be non-empty")
    _require_sha256(
        source_checkpoint.get("sha256"), label="Stage-C source checkpoint sha256"
    )
    if source_checkpoint.get("selection_status") != "best_infeasible":
        raise ValueError(
            "Stage C must declare source_checkpoint.selection_status="
            "'best_infeasible'"
        )
    if int(source_checkpoint.get("epoch", -1)) != 5 or int(
        source_checkpoint.get("global_step", -1)
    ) != 360:
        raise ValueError("Stage C requires the selected epoch-5/step-360 source")

    source_decision = _require_exact_mapping(
        stage_c.get("source_terminal_decision"),
        label="sentence_memory_safety.stage_c.source_terminal_decision",
        fields={
            "path",
            "sha256",
            "decision_identity",
            "status",
            "authorized_purpose",
        },
    )
    if not str(source_decision.get("path", "")):
        raise ValueError("Stage-C terminal-decision path must be non-empty")
    _require_sha256(
        source_decision.get("sha256"), label="Stage-C terminal-decision sha256"
    )
    _require_sha256(
        source_decision.get("decision_identity"),
        label="Stage-C terminal decision identity",
    )
    if source_decision.get("status") != "valid_infeasible":
        raise ValueError("Stage C requires terminal status 'valid_infeasible'")
    if source_decision.get("authorized_purpose") is not None:
        raise ValueError("Stage C requires authorized_purpose=null")

    teacher = _require_exact_mapping(
        stage_c.get("frozen_v2_teacher"),
        label="sentence_memory_safety.stage_c.frozen_v2_teacher",
        fields={"path", "sha256"},
    )
    if not str(teacher.get("path", "")):
        raise ValueError("Stage-C frozen-v2 teacher path must be non-empty")
    _require_sha256(teacher.get("sha256"), label="Stage-C frozen-v2 teacher sha256")
    source_stage_b = _require_exact_mapping(
        stage_c.get("source_stage_b"),
        label="sentence_memory_safety.stage_c.source_stage_b",
        fields={"config_path", "config_sha256", "architecture_identity"},
    )
    if not str(source_stage_b.get("config_path", "")):
        raise ValueError("Stage-C source Stage-B config path must be non-empty")
    _require_sha256(
        source_stage_b.get("config_sha256"), label="Stage-C Stage-B config sha256"
    )
    _require_sha256(
        source_stage_b.get("architecture_identity"),
        label="Stage-C Stage-B architecture identity",
    )
    for section_name, pinned in STAGE_C_PINNED_SOURCE.items():
        configured = stage_c.get(section_name)
        if configured != pinned:
            changed = sorted(
                name
                for name in set(configured or {}) | set(pinned)
                if (configured or {}).get(name) != pinned.get(name)
            )
            raise ValueError(
                "Stage-C schema v1 requires the exact approved "
                f"{section_name} source binding; changed fields={changed}"
            )
    active_stage_c = _require_exact_mapping(
        stage_c.get("active_stage_c"),
        label="sentence_memory_safety.stage_c.active_stage_c",
        fields={
            "calibration_artifact_dir",
            "calibration_schema_name",
            "calibration_schema_version",
        },
    )
    if active_stage_c.get("calibration_schema_name") != (
        "signtrajfield_sentence_memory_relevance_calibration"
    ) or int(active_stage_c.get("calibration_schema_version", -1)) != 1:
        raise ValueError("Stage C requires relevance-calibration schema v1")
    active_calibration_dir = Path(
        active_stage_c.get("calibration_artifact_dir", "")
    ).resolve()
    configured_calibration_dir = Path(
        cfg.get("sentence_memory", {})
        .get("relevance_calibration", {})
        .get("artifact_dir", "")
    ).resolve()
    if not str(active_stage_c.get("calibration_artifact_dir", "")) or (
        active_calibration_dir != configured_calibration_dir
    ):
        raise ValueError(
            "Stage-C active calibration directory must exactly match "
            "sentence_memory.relevance_calibration.artifact_dir"
        )

    if not is_sentence_memory_model(cfg) or not sentence_memory_enabled(cfg):
        raise ValueError("Stage C requires an enabled v3 sentence-memory model")
    if not centered_sentence_memory_enabled(cfg):
        raise ValueError("Stage C requires centered candidate covariance")
    if not configured_sentence_memory_association(cfg)["enabled"]:
        raise ValueError("Stage C requires the centered Stage-B association model")
    safety = cfg.get("sentence_memory_safety", {})
    if safety.get("enabled") is not True:
        raise ValueError("Stage C requires sentence_memory_safety.enabled=true")
    phase_b = safety.get("phase_b")
    if not isinstance(phase_b, dict) or phase_b.get("enabled") is not False:
        raise ValueError("Stage C requires explicit phase_b.enabled=false")
    paired = safety.get("paired_corruption")
    if not isinstance(paired, dict) or paired.get("enabled") is not False:
        raise ValueError("Stage C requires explicit paired_corruption.enabled=false")

    train_cfg = cfg.get("train", {})
    if train_cfg.get("freeze_base") is not True:
        raise ValueError("Stage C requires train.freeze_base=true")
    if train_cfg.get("freeze_sentence_memory") is not True:
        raise ValueError("Stage C requires train.freeze_sentence_memory=true")
    prefixes = tuple(str(value) for value in train_cfg.get("unfreeze_base_prefixes", ()))
    if prefixes != STAGE_C_GENERATOR_PREFIXES:
        raise ValueError(
            "Stage C requires exactly the seven approved generator prefixes "
            f"in protocol order: {list(STAGE_C_GENERATOR_PREFIXES)}"
        )
    if train_cfg.get("base_checkpoint") is not None:
        raise ValueError("Stage C forbids train.base_checkpoint; use its distinct warm start")
    if bool(train_cfg.get("reset_local_branch_on_warm_start", False)):
        raise ValueError("Stage C forbids resetting the warm-started local branch")
    exact_training = {
        "epochs": 1,
        "batch_size": 64,
        "accumulation_steps": 2,
        "early_stopping_patience": 0,
        "early_stopping_min_epochs": 1,
        "local_warmup_epochs": 0,
        "val_every": 1,
        "save_every": 1,
    }
    for name, expected in exact_training.items():
        _require_exact_number(train_cfg, name, expected, label="train")
    for name, expected in (
        ("joint_global_lr", 1e-5),
        ("joint_local_lr", 1e-5),
        ("weight_decay", 1e-4),
        ("grad_clip", 1.0),
        ("max_samples_per_memory_batch", 8),
        ("max_frames_per_memory_batch", 2048),
    ):
        _require_exact_number(train_cfg, name, expected, label="train")
    if train_cfg.get("length_bucketed_batches") is not True:
        raise ValueError(
            "Stage-C schema v1 requires train.length_bucketed_batches=true"
        )
    if train_cfg.get("drop_last") is not False:
        raise ValueError("Stage-C schema v1 requires train.drop_last=false")
    _require_exact_number(cfg, "seed", 1234, label="config")
    expected_mode = STAGE_C_ARM_TRAIN_MODES[arm]
    actual_mode = configured_sentence_memory_train_mode(cfg)
    if actual_mode != expected_mode:
        raise ValueError(
            f"Stage-C arm {arm!r} requires sentence-memory train mode "
            f"{expected_mode!r}, got {actual_mode!r}"
        )
    expected_dropout_probability = STAGE_C_ARM_DROPOUT_PROBABILITIES[arm]
    _require_exact_number(
        cfg.get("conditioning", {}),
        "sentence_memory_dropout_probability",
        expected_dropout_probability,
        label="conditioning",
    )
    if configured_word_prior_train_mode(cfg) != "off":
        raise ValueError("Stage C requires word_prior_train_mode='off'")
    if float(cfg.get("model", {}).get("dropout", 0.0)) != 0.0:
        raise ValueError("Stage C requires model.dropout=0")

    for name, expected in (
        ("huber_beta", 0.10),
        ("text_only_max_relative_degradation", 0.005),
    ):
        _require_exact_number(stage_c, name, expected, label="stage_c")
    for name, expected in (
        ("nonregression_margin", 0.0),
        ("shuffle_probability", 0.25),
        ("shuffle_huber_beta", 0.10),
    ):
        _require_exact_number(
            safety,
            name,
            expected,
            label="sentence_memory_safety",
        )
    objective = cfg.get("objective", {})
    for name, expected in (
        ("lambda_sentence_safe", 1.0),
        ("lambda_sentence_shuffle", 1.0),
        ("lambda_sentence_sparsity", 1e-4),
        ("lambda_sentence_off_distill", 1.0),
    ):
        _require_exact_number(objective, name, expected, label="objective")
    eval_cfg = cfg.get("eval", {})
    for name, expected in (
        ("max_samples_per_memory_batch", 1),
        ("max_frames_per_memory_batch", 512),
    ):
        _require_exact_number(eval_cfg, name, expected, label="eval")
    partition = cfg.get("validation_text_partition")
    if not isinstance(partition, dict) or partition.get("enabled") is not True:
        raise ValueError("Stage C requires the sealed development validation partition")
    data_cfg = cfg.get("data", {})
    if str(data_cfg.get("train_split", "train")) != "train" or str(
        data_cfg.get("val_split", "val")
    ) != "val":
        raise ValueError("Stage C permits only train and val data splits")
    if data_cfg.get("limit_train") not in (None, 0):
        raise ValueError("Stage C forbids data.limit_train")
    if data_cfg.get("limit_val") not in (None, 0):
        raise ValueError("Stage C forbids data.limit_val")
    if cfg.get("selection", {}).get("require_feasible") is not False:
        raise ValueError(
            "Stage C requires selection.require_feasible=false because it is "
            "development-only and non-authorizing"
        )
    configured_sentence_memory_eval_modes(cfg)
    return stage_c


def validate_stage_c_distributed_runtime(cfg, dist_info, device):
    """Bind schema-v1 Stage C to exactly two one-GPU NCCL ranks."""

    if not stage_c_enabled(cfg):
        return None
    visible_devices = int(torch.cuda.device_count())
    checks = {
        "ddp_enabled": bool(dist_info.get("enabled", False)),
        "world_size": int(dist_info.get("world_size", -1)),
        "backend": str(dist_info.get("backend")),
        "local_rank": int(dist_info.get("local_rank", -1)),
        "device_type": str(torch.device(device).type),
        "device_index": torch.device(device).index,
        "visible_cuda_devices_per_rank": visible_devices,
    }
    if not checks["ddp_enabled"] or checks["world_size"] != 2:
        raise RuntimeError("Stage-C schema v1 requires DDP world_size=2")
    if checks["backend"] != "nccl":
        raise RuntimeError("Stage-C schema v1 requires the NCCL backend")
    if checks["device_type"] != "cuda" or visible_devices != 1:
        raise RuntimeError(
            "Stage-C schema v1 requires exactly one visible CUDA GPU per rank"
        )
    if checks["local_rank"] != 0 or checks["device_index"] != 0:
        raise RuntimeError(
            "Stage-C paired-node contract requires one rank per node on cuda:0"
        )
    train_cfg = cfg["train"]
    resolved = _digest_named_identity(
        {
            "schema_name": "signtrajfield_stage_c_distribution_contract",
            "schema_version": 1,
            "world_size": 2,
            "ranks_per_node": 1,
            "visible_cuda_devices_per_rank": 1,
            "backend": "nccl",
            "micro_batch_per_rank": int(train_cfg["batch_size"]),
            "accumulation_steps": int(train_cfg["accumulation_steps"]),
            "effective_global_batch": (
                2
                * int(train_cfg["batch_size"])
                * int(train_cfg["accumulation_steps"])
            ),
            "epochs": int(train_cfg["epochs"]),
        }
    )
    configured_stage_c(cfg)["resolved_distribution_contract"] = resolved
    return resolved


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


def _sha256_file_stream(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pinned_stage_c_file(specification, *, label, path_key="path"):
    path = Path(specification[path_key]).resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    actual = _sha256_file_stream(path)
    expected = _require_sha256(
        specification["sha256" if path_key == "path" else "config_sha256"],
        label=f"{label} SHA256",
    )
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: actual={actual}, expected={expected}"
        )
    return path, actual


def _validated_named_identity(value, *, label):
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is missing or malformed")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("digest") != digest:
        raise RuntimeError(f"{label} has an invalid digest")
    return payload, digest


def validate_stage_c_selection_comparability(
    cfg, checkpoint, *, source="checkpoint"
):
    """Bind the Stage-B off reference to the active scoring definition."""

    source_cfg = checkpoint.get("config")
    if not isinstance(source_cfg, dict):
        raise RuntimeError(f"{source} has no saved Stage-B config")
    active_selection = copy.deepcopy(cfg.get("selection"))
    source_selection = copy.deepcopy(source_cfg.get("selection"))
    if not isinstance(active_selection, dict) or not isinstance(
        source_selection, dict
    ):
        raise RuntimeError(
            "Stage C requires explicit source and active selection definitions"
        )
    if active_selection.get("require_feasible") is not False:
        raise RuntimeError("Stage C requires active selection.require_feasible=false")
    if source_selection.get("require_feasible") is not True:
        raise RuntimeError(
            f"{source} does not record the canonical feasible Stage-B selection"
        )
    active_selection.pop("require_feasible")
    source_selection.pop("require_feasible")
    if active_selection != source_selection:
        changed = sorted(
            name
            for name in set(active_selection) | set(source_selection)
            if active_selection.get(name) != source_selection.get(name)
        )
        raise RuntimeError(
            "Stage-C selection score definition differs from Stage B beyond "
            f"the authorized require_feasible switch; changed fields={changed}"
        )

    active_modes = configured_sentence_memory_eval_modes(cfg)
    source_modes = configured_sentence_memory_eval_modes(source_cfg)
    if active_modes != STAGE_C_EVALUATION_MODES or source_modes != active_modes:
        raise RuntimeError(
            "Stage C requires the exact canonical 11-mode Stage-B evaluation order"
        )
    active_control = sentence_memory_evaluation_control_identity(cfg)
    source_control = sentence_memory_evaluation_control_identity(source_cfg)
    if (
        active_control != source_control
        or active_control.get("mode") != FIXED_EVIDENCE_CONTROLS_MODE
    ):
        raise RuntimeError(
            "Stage-C evaluation-control identity differs from canonical Stage B"
        )
    validate_sentence_memory_evaluation_control_identity(
        checkpoint, cfg, source=source
    )
    active_aggregation = sentence_memory_selection_aggregation_identity(cfg)
    source_aggregation = sentence_memory_selection_aggregation_identity(
        source_cfg
    )
    if (
        active_aggregation != source_aggregation
        or active_aggregation.get("mode") != CLUSTER_EQUAL_SELECTION_AGGREGATION
    ):
        raise RuntimeError(
            "Stage-C selection aggregation differs from canonical Stage B"
        )
    validate_sentence_memory_selection_aggregation_identity(
        checkpoint, cfg, source=source
    )
    return _digest_named_identity(
        {
            "schema_name": "signtrajfield_stage_c_selection_comparability",
            "schema_version": 1,
            "selection_definition_except_require_feasible": active_selection,
            "active_require_feasible": False,
            "source_require_feasible": True,
            "evaluation_modes": list(active_modes),
            "evaluation_control_identity": active_control["digest"],
            "selection_aggregation_identity": active_aggregation["digest"],
        }
    )


def validate_stage_c_calibration_transition(cfg, checkpoint, *, source="checkpoint"):
    """Validate the explicit old-source to fresh-active calibration transition.

    General checkpoint validators remain exact. Stage C alone may replace the
    calibration *provenance*, after independently proving that the coefficients
    and calibration-map content are unchanged and every other architecture and
    behavior field is identical.
    """

    expected_source_architecture = configured_stage_c(cfg)["source_stage_b"][
        "architecture_identity"
    ]
    source_architecture_payload, source_architecture_digest = (
        _validated_named_identity(
            checkpoint.get("sentence_memory_architecture_identity"),
            label=f"{source} source architecture identity",
        )
    )
    if source_architecture_digest != expected_source_architecture:
        raise RuntimeError(
            "Stage-C source architecture is not the pinned Stage-B identity: "
            f"actual={source_architecture_digest}, "
            f"expected={expected_source_architecture}"
        )
    active_architecture = sentence_memory_architecture_identity(cfg)
    active_architecture_payload, active_architecture_digest = (
        _validated_named_identity(
            active_architecture, label="active Stage-C architecture identity"
        )
    )

    source_calibration = (
        source_architecture_payload.get("relevance_gate", {}).get(
            "calibration_identity"
        )
    )
    active_calibration = (
        active_architecture_payload.get("relevance_gate", {}).get(
            "calibration_identity"
        )
    )
    source_calibration_payload, source_calibration_digest = (
        _validated_named_identity(
            source_calibration, label=f"{source} source calibration identity"
        )
    )
    active_calibration_payload, active_calibration_digest = (
        _validated_named_identity(
            active_calibration, label="active Stage-C calibration identity"
        )
    )
    checkpoint_calibration = checkpoint.get(
        "sentence_memory_relevance_calibration_identity"
    )
    if checkpoint_calibration != source_calibration:
        raise RuntimeError(
            "Stage-C source checkpoint calibration metadata differs from its "
            "source architecture identity"
        )
    # Only repository/file-bound provenance is permitted to change. Everything
    # that can affect or qualify the calibration result remains exact: map
    # bytes/content, held-out statistics, minimum gates, coefficient formula,
    # and all schema-defined deterministic semantics.
    provenance_only_fields = {"artifact_identity", "calibration_sha256"}
    source_calibration_semantics = {
        name: value
        for name, value in source_calibration_payload.items()
        if name not in provenance_only_fields
    }
    active_calibration_semantics = {
        name: value
        for name, value in active_calibration_payload.items()
        if name not in provenance_only_fields
    }
    if source_calibration_semantics != active_calibration_semantics:
        changed = sorted(
            name
            for name in set(source_calibration_semantics)
            | set(active_calibration_semantics)
            if source_calibration_semantics.get(name)
            != active_calibration_semantics.get(name)
        )
        raise RuntimeError(
            "Stage-C fresh calibration is not semantically equivalent to the "
            f"source calibration; changed fields={changed}"
        )
    if active_calibration_digest == source_calibration_digest:
        raise RuntimeError(
            "Stage C requires a freshly attested calibration identity, not the "
            "source repository-bound identity"
        )

    source_structural = copy.deepcopy(source_architecture_payload)
    active_structural = copy.deepcopy(active_architecture_payload)
    source_structural["relevance_gate"].pop("calibration_identity", None)
    active_structural["relevance_gate"].pop("calibration_identity", None)
    if source_structural != active_structural:
        raise RuntimeError(
            "Stage-C active architecture differs from Stage B beyond the "
            "explicit calibration-provenance transition"
        )

    source_behavior_payload, _source_behavior_digest = _validated_named_identity(
        checkpoint.get("sentence_memory_behavior_identity"),
        label=f"{source} source behavior identity",
    )
    active_behavior_payload, active_behavior_digest = _validated_named_identity(
        sentence_memory_behavior_identity(cfg),
        label="active Stage-C behavior identity",
    )
    source_behavior_payload.pop("sentence_memory_architecture_digest", None)
    active_behavior_payload.pop("sentence_memory_architecture_digest", None)
    if source_behavior_payload != active_behavior_payload:
        raise RuntimeError(
            "Stage-C active sentence-memory behavior differs from Stage B "
            "beyond calibration provenance"
        )
    return {
        "schema_name": "signtrajfield_stage_c_calibration_transition",
        "schema_version": 1,
        "source_calibration_identity": source_calibration_digest,
        "active_calibration_identity": active_calibration_digest,
        "source_architecture_identity": source_architecture_digest,
        "active_architecture_identity": active_architecture_digest,
        "active_behavior_identity": active_behavior_digest,
        "semantic_equivalence": {
            "coefficients_exact": True,
            "map_sha256_exact": True,
            "map_content_digest_exact": True,
            "heldout_metrics_exact": True,
            "minimum_gates_exact": True,
            "all_non_provenance_calibration_fields_exact": True,
            "structural_architecture_except_calibration_provenance_exact": True,
            "behavior_except_architecture_digest_exact": True,
        },
    }


def validate_stage_c_warm_start_checkpoint(
    cfg, checkpoint, checkpoint_path, *, provider, source="checkpoint"
):
    """Bind Stage C to the exact non-authorizing Stage-B terminal state."""

    if not stage_c_enabled(cfg):
        return None
    if provider is None:
        raise RuntimeError("Stage C requires an active sentence-memory provider")
    stage_c = configured_stage_c(cfg)
    declared = _stage_c_declared_contract(stage_c)
    source_checkpoint = stage_c["source_checkpoint"]
    expected_checkpoint_path = Path(source_checkpoint["path"]).resolve()
    actual_checkpoint_path = Path(checkpoint_path).resolve()
    if actual_checkpoint_path != expected_checkpoint_path:
        raise RuntimeError(
            "--stage_c_warm_start does not match the pinned source checkpoint: "
            f"actual={actual_checkpoint_path}, expected={expected_checkpoint_path}"
        )
    if actual_checkpoint_path.name != "best_infeasible.pt":
        raise RuntimeError("Stage C must warm-start from a best_infeasible.pt artifact")
    actual_checkpoint_sha256 = _sha256_file_stream(actual_checkpoint_path)
    if actual_checkpoint_sha256 != source_checkpoint["sha256"]:
        raise RuntimeError(
            "Stage-C source checkpoint SHA256 mismatch: "
            f"actual={actual_checkpoint_sha256}, "
            f"expected={source_checkpoint['sha256']}"
        )
    expected_epoch = int(source_checkpoint["epoch"])
    expected_step = int(source_checkpoint["global_step"])
    if int(checkpoint.get("epoch", -1)) != expected_epoch or int(
        checkpoint.get("global_step", -1)
    ) != expected_step:
        raise RuntimeError(
            "Stage-C source checkpoint is not the pinned epoch/global step: "
            f"actual={checkpoint.get('epoch')}/{checkpoint.get('global_step')}, "
            f"expected={expected_epoch}/{expected_step}"
        )
    source_cfg = checkpoint.get("config")
    if not isinstance(source_cfg, dict):
        raise RuntimeError(f"{source} has no saved Stage-B config")
    if not centered_sentence_memory_enabled(source_cfg) or not (
        configured_sentence_memory_association(source_cfg)["enabled"]
    ):
        raise RuntimeError(
            f"{source} is not a centered absolute-association Stage-B checkpoint"
        )
    source_phase_b = source_cfg.get("sentence_memory_safety", {}).get(
        "phase_b", {}
    )
    if not isinstance(source_phase_b, dict) or source_phase_b.get("enabled") is not False:
        raise RuntimeError(f"{source} lacks the explicit canonical phase_b=false marker")
    source_paired = source_cfg.get("sentence_memory_safety", {}).get(
        "paired_corruption", {}
    )
    if not isinstance(source_paired, dict) or source_paired.get("enabled") is not True:
        raise RuntimeError(f"{source} is not the paired centered Stage-B source")
    selection_state = checkpoint.get("selection_state")
    if not isinstance(selection_state, dict):
        raise RuntimeError(f"{source} has no selection state")
    if selection_state.get("best_feasible_score") is not None:
        raise RuntimeError(f"{source} records a feasible checkpoint")
    if selection_state.get("best_infeasible_score") is None or (
        selection_state.get("best_infeasible_key") is None
    ):
        raise RuntimeError(f"{source} has no best-infeasible selection provenance")
    metrics = checkpoint.get("metrics") or {}
    if bool(metrics.get("selection_feasible", True)):
        raise RuntimeError(f"{source} is not marked selection-infeasible")
    initialization_parity = checkpoint.get("v2_to_v3_text_only_parity")
    if not isinstance(initialization_parity, dict) or not bool(
        initialization_parity.get("passed", False)
    ):
        raise RuntimeError(
            f"{source} has no passing source initialization-parity proof"
        )
    selection_comparability = validate_stage_c_selection_comparability(
        cfg, checkpoint, source=source
    )
    validate_sentence_memory_validation_corruption_map_identity(
        checkpoint, cfg, source=source
    )
    source_partition = source_cfg.get("validation_text_partition") or {}
    active_partition = cfg.get("validation_text_partition") or {}
    partition_fields = (
        "partition_digest",
        "expected_partition_digest",
        "expected_development_manifest_sha256",
        "expected_development_rows",
        "development_text_count",
    )
    partition_binding = {
        name: copy.deepcopy(active_partition.get(name)) for name in partition_fields
    }
    if any(value is None for value in partition_binding.values()) or any(
        source_partition.get(name) != active_partition.get(name)
        for name in partition_fields
    ):
        raise RuntimeError(
            "Stage-C source and active sealed development partitions differ"
        )
    source_text_metrics = {
        name.removeprefix("val_text_only/"): value
        for name, value in metrics.items()
        if name.startswith("val_text_only/")
    }
    if not source_text_metrics:
        raise RuntimeError(
            f"{source} has no sealed-development val_text_only metrics"
        )
    teacher_validation_score = float(
        selection_diagnostics(source_text_metrics, source_cfg)[0]
    )
    if not math.isfinite(teacher_validation_score):
        raise RuntimeError(
            f"{source} has a non-finite sealed-development text-only score"
        )
    stage_c["teacher_validation_selection_score"] = teacher_validation_score

    expected_architecture = stage_c["source_stage_b"]["architecture_identity"]
    calibration_transition = validate_stage_c_calibration_transition(
        cfg, checkpoint, source=source
    )
    if calibration_transition["source_architecture_identity"] != expected_architecture:
        raise RuntimeError("Stage-C source architecture identity is not the pinned value")

    source_stage_b = stage_c["source_stage_b"]
    config_path, config_sha256 = _validate_pinned_stage_c_file(
        source_stage_b,
        label="Stage-C source Stage-B config",
        path_key="config_path",
    )
    decision_spec = stage_c["source_terminal_decision"]
    decision_path, decision_sha256 = _validate_pinned_stage_c_file(
        decision_spec, label="Stage-C source terminal decision"
    )
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read Stage-C source terminal decision {decision_path}: {error}"
        ) from error
    if not isinstance(decision, dict):
        raise RuntimeError("Stage-C source terminal decision must be a JSON object")
    if decision.get("schema_name") != "signtrajfield_centered_memory_ordered_decision" or int(
        decision.get("schema_version", -1)
    ) != 1:
        raise RuntimeError("Stage-C source terminal decision has an unsupported schema")
    for name, expected in (
        ("decision_identity", decision_spec["decision_identity"]),
        ("status", decision_spec["status"]),
        ("authorized_purpose", decision_spec["authorized_purpose"]),
    ):
        if decision.get(name) != expected:
            raise RuntimeError(
                f"Stage-C terminal decision {name} mismatch: "
                f"actual={decision.get(name)!r}, expected={expected!r}"
            )
    if decision.get("stage") != "stage2":
        raise RuntimeError("Stage-C source decision is not the Stage-B/stage2 decision")
    if decision.get("integrity_valid") is not True or decision.get(
        "development_feasible"
    ) is not False:
        raise RuntimeError(
            "Stage C requires an integrity-valid, development-infeasible source decision"
        )
    if decision.get("confirmation_manifest_opened") is not False or decision.get(
        "test_data_accessed"
    ) is not False:
        raise RuntimeError(
            "Stage-C source decision must attest no confirmation or test access"
        )
    decision_checkpoint = decision.get("checkpoint")
    if not isinstance(decision_checkpoint, dict):
        raise RuntimeError("Stage-C terminal decision has no checkpoint provenance")
    try:
        decision_checkpoint_path = Path(decision_checkpoint["path"]).resolve()
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "Stage-C terminal decision has no valid checkpoint path"
        ) from error
    if decision_checkpoint_path != actual_checkpoint_path:
        raise RuntimeError(
            "Stage-C terminal decision names a different source checkpoint"
        )
    for name, expected in (
        ("sha256", actual_checkpoint_sha256),
        ("epoch", expected_epoch),
        ("global_step", expected_step),
        ("has_feasible_checkpoint", False),
    ):
        if decision_checkpoint.get(name) != expected:
            raise RuntimeError(
                f"Stage-C terminal decision checkpoint {name} mismatch"
            )
    decision_config = decision_checkpoint.get("config") or {}
    if decision_config.get("sha256") != config_sha256:
        raise RuntimeError(
            "Stage-C terminal decision does not bind the pinned Stage-B config"
        )
    decision_architecture = (
        decision_checkpoint.get("identities", {}).get("architecture", {}).get("digest")
    )
    if decision_architecture != expected_architecture:
        raise RuntimeError(
            "Stage-C terminal decision does not bind the pinned architecture"
        )

    teacher_path, teacher_sha256 = _validate_pinned_stage_c_file(
        stage_c["frozen_v2_teacher"], label="Stage-C frozen-v2 teacher"
    )
    decision_teacher_sha256 = (
        decision.get("development_integrity", {}).get("v2_checkpoint_sha256")
    )
    if decision_teacher_sha256 != teacher_sha256:
        raise RuntimeError(
            "Stage-C terminal decision uses a different frozen-v2 checkpoint"
        )

    checkpoint_memory = checkpoint.get("sentence_memory_identity")
    active_memory = getattr(provider, "identity", None)
    if not isinstance(checkpoint_memory, dict) or not isinstance(active_memory, dict):
        raise RuntimeError("Stage C requires source and active memory identities")
    if checkpoint_memory.get("bank_id") != active_memory.get("bank_id"):
        raise RuntimeError("Stage-C source and active memory bank IDs differ")
    checkpoint_neighbors = checkpoint_memory.get("neighbor_tables")
    active_neighbors = active_memory.get("neighbor_tables")
    if not isinstance(checkpoint_neighbors, dict) or checkpoint_neighbors != active_neighbors:
        raise RuntimeError(
            "Stage-C source and active sentence-neighbor table identities differ"
        )
    if not isinstance(active_neighbors.get("train"), dict):
        raise RuntimeError("Stage C has no identity-bound training neighbor table")
    distribution_contract = stage_c.get("resolved_distribution_contract")
    _validated_named_identity(
        distribution_contract, label="active Stage-C distribution contract"
    )

    provenance = _digest_named_identity(
        {
            "schema_name": "signtrajfield_stage_c_warm_start_provenance",
            "schema_version": 1,
            "declared_contract": declared,
            "source_checkpoint": {
                "path": str(actual_checkpoint_path),
                "sha256": actual_checkpoint_sha256,
                "selection_status": "best_infeasible",
                "epoch": expected_epoch,
                "global_step": expected_step,
            },
            "source_terminal_decision": {
                "path": str(decision_path),
                "sha256": decision_sha256,
                "decision_identity": decision["decision_identity"],
                "status": decision["status"],
                "authorized_purpose": decision.get("authorized_purpose"),
            },
            "source_stage_b": {
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "architecture_identity": expected_architecture,
            },
            "frozen_v2_teacher": {
                "path": str(teacher_path),
                "sha256": teacher_sha256,
            },
            "calibration_transition": calibration_transition,
            "memory_semantic_equivalence": {
                "bank_id": active_memory["bank_id"],
                "bank_id_exact": True,
                "neighbor_tables_exact": True,
                "train_neighbor_table_identity": copy.deepcopy(
                    active_neighbors["train"]
                ),
            },
            # This is explicitly source-time evidence. It is not copied into
            # Stage-C's live v2_to_v3_text_only_parity checkpoint field because
            # generator updates invalidate that equality after the first step.
            "source_initialization_v2_text_only_parity": copy.deepcopy(
                initialization_parity
            ),
            "teacher_validation_reference": {
                "source": "stage_b_memory_off_exact_v2_parity",
                "selection_score": teacher_validation_score,
                "selection_comparability_identity": (
                    selection_comparability["digest"]
                ),
                "selection_comparability": copy.deepcopy(
                    selection_comparability
                ),
                "selection_aggregation_identity": (
                    checkpoint[
                        "sentence_memory_selection_aggregation_identity"
                    ]["digest"]
                ),
                "validation_corruption_map_identity": (
                    checkpoint[
                        "sentence_memory_validation_corruption_map_identity"
                    ]["digest"]
                ),
                "sealed_development_partition": partition_binding,
            },
            "state_reset": {
                "model_weights_only": True,
                "epoch": 1,
                "global_step": 0,
                "optimizer": "fresh",
                "selection": "fresh",
            },
            "distribution_contract": copy.deepcopy(
                distribution_contract
            ),
            "trainability_contract": stage_c_trainability_contract(),
            "optimizer_parameter_mapping": (
                stage_c_optimizer_parameter_mapping_contract()
            ),
            "scope": {
                "development_only": True,
                "non_authorizing": True,
                "promotion_eligible": False,
                "confirmation_manifest_opened": False,
                "test_data_accessed": False,
            },
        }
    )
    stage_c["resolved_source_provenance"] = provenance
    return provenance


def validate_stage_c_resume_checkpoint(cfg, checkpoint, source="checkpoint"):
    """Require exact Stage-C provenance before resuming an adaptation run."""

    if not stage_c_enabled(cfg):
        return None
    checkpoint_cfg = checkpoint.get("config") or {}
    checkpoint_stage_c = configured_stage_c(checkpoint_cfg)
    if not bool(checkpoint_stage_c.get("enabled", False)):
        raise RuntimeError(f"{source} is not a Stage-C checkpoint")
    if _stage_c_declared_contract(checkpoint_stage_c) != _stage_c_declared_contract(
        configured_stage_c(cfg)
    ):
        raise RuntimeError(f"{source} has a different Stage-C declared contract")
    provenance = checkpoint.get("stage_c_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{source} has no Stage-C warm-start provenance")
    payload = {key: value for key, value in provenance.items() if key != "digest"}
    actual_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if provenance.get("digest") != actual_digest:
        raise RuntimeError(f"{source} has an invalid Stage-C provenance digest")
    if provenance.get("declared_contract") != _stage_c_declared_contract(
        configured_stage_c(cfg)
    ):
        raise RuntimeError(f"{source} Stage-C provenance does not match active config")
    saved_distribution = checkpoint.get("stage_c_distribution_contract")
    active_distribution = configured_stage_c(cfg).get(
        "resolved_distribution_contract"
    )
    if not isinstance(saved_distribution, dict) or (
        saved_distribution != active_distribution
    ):
        raise RuntimeError(
            f"{source} Stage-C distribution contract differs from this launch"
        )
    _validated_named_identity(
        saved_distribution, label=f"{source} Stage-C distribution contract"
    )
    if provenance.get("distribution_contract") != active_distribution:
        raise RuntimeError(
            f"{source} warm-start provenance has a different distribution contract"
        )
    expected_trainability = stage_c_trainability_contract()
    saved_trainability = checkpoint.get("stage_c_trainability_contract")
    if saved_trainability != expected_trainability:
        raise RuntimeError(
            f"{source} Stage-C trainability contract differs from schema v1"
        )
    _validated_named_identity(
        saved_trainability, label=f"{source} Stage-C trainability contract"
    )
    if provenance.get("trainability_contract") != expected_trainability:
        raise RuntimeError(
            f"{source} warm-start provenance has a different trainability contract"
        )
    expected_optimizer_mapping = (
        stage_c_optimizer_parameter_mapping_contract()
    )
    saved_optimizer_mapping = checkpoint.get(
        "stage_c_optimizer_parameter_mapping"
    )
    if saved_optimizer_mapping != expected_optimizer_mapping:
        raise RuntimeError(
            f"{source} Stage-C optimizer parameter mapping differs from schema v1"
        )
    _validated_named_identity(
        saved_optimizer_mapping,
        label=f"{source} Stage-C optimizer parameter mapping",
    )
    if provenance.get("optimizer_parameter_mapping") != (
        expected_optimizer_mapping
    ):
        raise RuntimeError(
            f"{source} warm-start provenance has a different optimizer mapping"
        )
    if provenance.get("state_reset") != {
        "model_weights_only": True,
        "epoch": 1,
        "global_step": 0,
        "optimizer": "fresh",
        "selection": "fresh",
    }:
        raise RuntimeError(f"{source} lacks exact Stage-C state-reset provenance")
    scope = provenance.get("scope") or {}
    if scope != {
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }:
        raise RuntimeError(f"{source} has invalid Stage-C authorization scope")
    configured_stage_c(cfg)["resolved_source_provenance"] = copy.deepcopy(
        provenance
    )
    return provenance


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
    """Return the exact number of local aggregation units consumed."""

    explicit_units = getattr(loader, "evaluation_unit_count", None)
    if explicit_units is not None:
        total = int(explicit_units)
    else:
        sampler = getattr(loader, "sampler", None)
        total = len(sampler) if sampler is not None else len(loader.dataset)
    max_batches = max(int(max_batches), 0)
    if max_batches:
        if explicit_units is not None:
            total = min(total, max_batches)
        else:
            batch_size = max(int(getattr(loader, "batch_size", 1) or 1), 1)
            total = min(total, max_batches * batch_size)
    return int(total)


@dataclass(frozen=True)
class DevelopmentValidationRuntime:
    """Development-only view of a sealed validation partition.

    This object deliberately contains no confirmation row, normalized text, or
    assignment.  The training process can therefore validate and instantiate
    the development manifest without parsing ``partition.json`` or opening the
    confirmation manifest.
    """

    partition_dir: Path
    manifest_path: Path
    normalized_texts: tuple[str, ...]
    development_row_count: int
    development_text_count: int
    confirmation_row_count: int
    confirmation_text_count: int
    partition_digest: str
    artifact_payload: dict


def requires_isolated_development_validation(cfg):
    """Return whether a protected factorized run needs its sealed dev view."""

    memory_mode = str(
        cfg.get("sentence_memory", {}).get(
            "key_value_mode", "legacy_mixed_v1"
        )
    ).lower()
    partition_cfg = cfg.get("validation_text_partition")
    return bool(
        memory_mode == "factorized_metadata_motion_v1"
        and (
            paired_sentence_memory_corruption_config(cfg)["enabled"]
            or stage_c_enabled(cfg)
        )
        and isinstance(partition_cfg, dict)
        and partition_cfg.get("enabled", False)
    )


def requires_development_only_selection(cfg):
    """Keep protected training phases off confirmation rows by construction."""

    return bool(
        paired_sentence_memory_corruption_config(cfg)["enabled"]
        or stage_c_enabled(cfg)
    )


def _require_sha256(value, *, label):
    value = str(value or "").lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def load_isolated_development_validation_runtime(cfg, *, partition_dir=None):
    """Attest and read only the prebuilt development validation manifest.

    The full validation manifest, ``partition.json``, exact-seen manifest, and
    confirmation manifest are intentionally outside this function's access
    surface.  Membership is pinned by the development-manifest SHA256 in the
    experiment config, while the full partition digest/counts remain pinned as
    numeric/digest metadata for the later post-lock verifier.
    """

    if not requires_isolated_development_validation(cfg):
        return None
    partition_cfg = cfg.get("validation_text_partition", {})
    if str(partition_cfg.get("split", "val")) != "val":
        raise ValueError("Isolated development validation requires split='val'")

    requested_dir = partition_dir
    if requested_dir is None:
        requested_dir = os.environ.get(FACTORIZED_VALIDATION_PARTITION_ENV)
    if not requested_dir:
        raise ValueError(
            f"Factorized training requires explicit {FACTORIZED_VALIDATION_PARTITION_ENV} "
            "pointing to the pre-existing sealed validation partition"
        )
    directory = Path(requested_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Sealed validation partition directory does not exist: {directory}"
        )

    # READY is the only partition-level file opened by training. Keep the
    # accepted envelope small so it cannot become a covert copy of holdout
    # assignments or sentence text.
    ready_path = directory / "READY"
    if not ready_path.is_file():
        raise FileNotFoundError(f"Validation partition is not READY: {directory}")
    if ready_path.stat().st_size > 4096:
        raise ValueError("Validation partition READY metadata is unexpectedly large")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if set(ready) != {"schema_name", "schema_version", "artifact_identity"}:
        raise ValueError("Validation partition READY metadata has an unsafe schema")
    if ready.get("schema_name") != "signtrajfield_validation_text_cluster_partition":
        raise ValueError("Validation partition READY schema name is invalid")
    if int(ready.get("schema_version", -1)) != 1:
        raise ValueError("Validation partition READY schema version is invalid")
    artifact_identity = _require_sha256(
        ready.get("artifact_identity"), label="READY artifact_identity"
    )
    expected_artifact_identity = _require_sha256(
        partition_cfg.get("expected_partition_artifact_identity"),
        label=(
            "validation_text_partition.expected_partition_artifact_identity"
        ),
    )
    if artifact_identity != expected_artifact_identity:
        raise ValueError(
            "Validation partition READY artifact identity mismatch: "
            f"actual={artifact_identity}, expected={expected_artifact_identity}"
        )

    partition_digest = _require_sha256(
        partition_cfg.get("expected_partition_digest"),
        label="validation_text_partition.expected_partition_digest",
    )
    source_manifest_sha256 = _require_sha256(
        partition_cfg.get("expected_validation_manifest_sha256"),
        label="validation_text_partition.expected_validation_manifest_sha256",
    )
    development_manifest_sha256 = _require_sha256(
        partition_cfg.get("expected_development_manifest_sha256"),
        label=(
            "validation_text_partition.expected_development_manifest_sha256"
        ),
    )
    expected_bank_id = _require_sha256(
        partition_cfg.get("expected_bank_id"),
        label="validation_text_partition.expected_bank_id",
    )

    def required_nonnegative_integer(name):
        if name not in partition_cfg:
            raise ValueError(f"validation_text_partition.{name} is required")
        value = int(partition_cfg[name])
        if value < 0:
            raise ValueError(
                f"validation_text_partition.{name} must be non-negative"
            )
        return value

    expected_validation_rows = required_nonnegative_integer(
        "expected_validation_rows"
    )
    expected_development_rows = required_nonnegative_integer(
        "expected_development_rows"
    )
    expected_confirmation_rows = required_nonnegative_integer(
        "expected_confirmation_rows"
    )
    expected_novel_text_count = required_nonnegative_integer(
        "expected_novel_text_count"
    )
    development_text_count = required_nonnegative_integer(
        "development_text_count"
    )
    if expected_development_rows + expected_confirmation_rows > expected_validation_rows:
        raise ValueError(
            "Development and confirmation rows exceed the pinned validation row count"
        )
    if development_text_count > expected_novel_text_count:
        raise ValueError(
            "Development text count exceeds the pinned novel-text population"
        )
    confirmation_text_count = expected_novel_text_count - development_text_count

    manifest_path = directory / DEVELOPMENT_VALIDATION_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Sealed partition is missing {DEVELOPMENT_VALIDATION_MANIFEST}"
        )
    digest = hashlib.sha256()
    rows = []
    with manifest_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid development manifest row {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Development manifest row {line_number} is not an object"
                )
            rows.append(row)
    actual_development_sha256 = digest.hexdigest()
    if actual_development_sha256 != development_manifest_sha256:
        raise ValueError(
            "Development manifest SHA256 mismatch: "
            f"actual={actual_development_sha256}, "
            f"expected={development_manifest_sha256}"
        )
    if len(rows) != expected_development_rows:
        raise ValueError(
            "Development manifest row count mismatch: "
            f"actual={len(rows)}, expected={expected_development_rows}"
        )
    wrong_split = [
        index
        for index, row in enumerate(rows)
        if str(row.get("source_split", "val")) != "val"
    ]
    if wrong_split:
        raise ValueError(
            "Development manifest contains a non-validation source row: "
            f"index={wrong_split[0]}"
        )
    from NIAF.continuous_trajectory_field.sentence_memory import (
        normalize_sentence_text,
    )

    normalized_texts = tuple(
        normalize_sentence_text(row.get("text", "")) for row in rows
    )
    if any(not text for text in normalized_texts):
        raise ValueError("Development manifest contains an empty normalized text")
    if len(set(normalized_texts)) != development_text_count:
        raise ValueError(
            "Development manifest unique-text count mismatch: "
            f"actual={len(set(normalized_texts))}, expected={development_text_count}"
        )

    payload_without_identity = {
        "schema_name": DEVELOPMENT_VALIDATION_RUNTIME_SCHEMA,
        "schema_version": 1,
        "validation_only": True,
        "partition": {
            "digest": partition_digest,
            "artifact_identity": artifact_identity,
        },
        "source_validation_manifest": {
            "sha256": source_manifest_sha256,
            "row_count": expected_validation_rows,
        },
        "development_manifest": {
            "file": DEVELOPMENT_VALIDATION_MANIFEST,
            "sha256": actual_development_sha256,
            "row_count": expected_development_rows,
            "unique_text_count": development_text_count,
        },
        "confirmation_counts": {
            "row_count": expected_confirmation_rows,
            "unique_text_count": confirmation_text_count,
        },
        "bank_id": expected_bank_id,
        "retrieval_query_mode": "exact_name_indexed_full_val_table_subset_v1",
        "holdout_access": "development_manifest_only_v1",
    }
    runtime_identity = hashlib.sha256(
        json.dumps(
            payload_without_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    safe_payload = {
        **payload_without_identity,
        "runtime_identity": runtime_identity,
    }
    # These fields are the complete allowed partition evidence persisted into
    # config/checkpoints. In particular there is no resolved full artifact,
    # assignment list, confirmation text, or operational filesystem path.
    partition_cfg["partition_digest"] = partition_digest
    partition_cfg["partition_artifact_identity"] = artifact_identity
    partition_cfg["development_row_count"] = expected_development_rows
    partition_cfg["confirmation_row_count"] = expected_confirmation_rows
    partition_cfg["confirmation_text_count"] = confirmation_text_count
    partition_cfg["development_manifest_identity"] = dict(
        safe_payload["development_manifest"]
    )
    partition_cfg["development_runtime_identity"] = {
        "schema_name": DEVELOPMENT_VALIDATION_RUNTIME_SCHEMA,
        "schema_version": 1,
        "digest": runtime_identity,
    }
    partition_cfg["retrieval_query_mode"] = safe_payload[
        "retrieval_query_mode"
    ]
    partition_cfg["exact_seen_evaluated_during_training"] = False
    partition_cfg["confirmation_evaluated_during_training"] = False
    return DevelopmentValidationRuntime(
        partition_dir=directory,
        manifest_path=manifest_path,
        normalized_texts=normalized_texts,
        development_row_count=expected_development_rows,
        development_text_count=development_text_count,
        confirmation_row_count=expected_confirmation_rows,
        confirmation_text_count=confirmation_text_count,
        partition_digest=partition_digest,
        artifact_payload=safe_payload,
    )


def isolated_development_validation_config(cfg, runtime):
    """Return a loader-only config that points ``val`` at the attested dev file."""

    if runtime is None:
        return cfg
    data_cfg = cfg.get("data", {})
    split = str(data_cfg.get("val_split", "val"))
    if split != "val":
        raise ValueError("Isolated development validation requires data.val_split='val'")
    if int(data_cfg.get("limit_val", 0) or 0) != 0:
        raise ValueError(
            "Isolated development validation forbids data.limit_val; use "
            "eval.max_batches only for an explicit engineering smoke"
        )
    loader_cfg = copy.deepcopy(cfg)
    loader_cfg.setdefault("data", {})["val_manifest_path"] = str(
        runtime.manifest_path
    )
    return loader_cfg


def bind_sentence_memory_validation_dataset(
    cfg, sentence_memory_provider, dataset, runtime
):
    """Bind validation retrieval, using the canonical table by exact dev name."""

    require_memory = any(
        mode != "off" for mode in configured_sentence_memory_eval_modes(cfg)
    )
    if runtime is None:
        return sentence_memory_provider.validate_query_dataset(
            dataset, require_neighbors=require_memory
        )
    if not require_memory:
        raise RuntimeError(
            "Factorized paired validation unexpectedly has no active memory mode"
        )
    partition_cfg = cfg.get("validation_text_partition", {})
    expected_bank_id = str(partition_cfg.get("expected_bank_id", ""))
    actual_bank_id = str(sentence_memory_provider.identity.get("bank_id", ""))
    if actual_bank_id != expected_bank_id:
        raise RuntimeError(
            "Development validation bank mismatch: "
            f"actual={actual_bank_id!r}, expected={expected_bank_id!r}"
        )
    full_table_identity = (
        sentence_memory_provider.identity.get("neighbor_tables", {}).get(
            "val"
        )
    )
    if not isinstance(full_table_identity, dict):
        raise RuntimeError(
            "Development validation requires the identity-bound canonical "
            "neighbors_val table"
        )
    expected_full_manifest = str(
        partition_cfg.get("expected_validation_manifest_sha256", "")
    )
    if str(full_table_identity.get("query_manifest_sha256", "")) != (
        expected_full_manifest
    ):
        raise RuntimeError(
            "Canonical validation neighbor table manifest mismatch: "
            f"actual={full_table_identity.get('query_manifest_sha256')!r}, "
            f"expected={expected_full_manifest!r}"
        )
    expected_full_rows = int(
        partition_cfg.get("expected_validation_rows", -1)
    )
    if int(full_table_identity.get("query_count", -1)) != expected_full_rows:
        raise RuntimeError(
            "Canonical validation neighbor table row count mismatch: "
            f"actual={full_table_identity.get('query_count')!r}, "
            f"expected={expected_full_rows}"
        )
    sentence_memory_provider.set_dataset_with_name_indexed_neighbor_subset(
        dataset,
        parent_manifest_sha256=expected_full_manifest,
        expected_subset_manifest_sha256=(
            runtime.artifact_payload["development_manifest"]["sha256"]
        ),
    )
    table = sentence_memory_provider.neighbor_table
    if not isinstance(table, dict) or table.get("lookup_mode") != (
        "exact_name_indexed_parent_subset_v1"
    ):
        raise RuntimeError(
            "Development validation did not bind an exact name-indexed neighbor subset"
        )
    return {
        "split": str(getattr(dataset, "split", "")),
        "manifest_sha256": sentence_memory_provider._dataset_manifest_sha256,
        "neighbor_table": table.get("path"),
        "neighbor_lookup_mode": table["lookup_mode"],
        "parent_query_manifest_sha256": table[
            "parent_query_manifest_sha256"
        ],
        "parent_query_count": int(table["parent_query_count"]),
    }


def build_isolated_development_validation_loader(
    cfg,
    dataset,
    loader,
    sentence_memory_provider,
    dist_info,
    runtime,
):
    """Build cluster-equal validation directly over the attested dev manifest."""

    if runtime is None:
        raise ValueError("An isolated development runtime is required")
    if len(dataset) != runtime.development_row_count:
        raise RuntimeError(
            "Loaded development dataset row count differs from its attestation: "
            f"actual={len(dataset)}, expected={runtime.development_row_count}"
        )
    rows = list(dataset.base.items)
    if len(rows) != runtime.development_row_count:
        raise RuntimeError(
            "Development dataset manifest row count differs from its attestation"
        )
    from NIAF.continuous_trajectory_field.sentence_memory import (
        normalize_sentence_text,
    )

    normalized_texts = tuple(
        normalize_sentence_text(row.get("text", "")) for row in rows
    )
    if normalized_texts != runtime.normalized_texts:
        raise RuntimeError(
            "Loaded development dataset differs from the attested manifest order"
        )
    seen_indices = [
        index
        for index, text in enumerate(normalized_texts)
        if sentence_memory_provider.is_seen_text(text)
    ]
    if seen_indices:
        raise RuntimeError(
            "Development manifest contains an exact train-bank text: "
            f"index={seen_indices[0]}"
        )
    table = sentence_memory_provider.neighbor_table
    if not isinstance(table, dict) or table.get("lookup_mode") != (
        "exact_name_indexed_parent_subset_v1"
    ):
        raise RuntimeError(
            "Development loader requires the exact name-indexed canonical "
            "neighbor subset"
        )

    num_workers = int(cfg.get("eval", {}).get("num_workers", 0))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.get("train", {}).get("persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(
            cfg.get("train", {}).get("prefetch_factor", 2)
        )
    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("train", {}).get("pin_memory", True)),
        "collate_fn": loader.collate_fn,
        **loader_kwargs,
    }
    aggregation = configured_selection_aggregation(cfg)
    from NIAF.continuous_trajectory_field.validation_text_partitions import (
        NormalizedTextClusterBatchSampler,
    )

    if aggregation == CLUSTER_EQUAL_SELECTION_AGGREGATION:
        sampler = NormalizedTextClusterBatchSampler(
            normalized_texts,
            num_replicas=int(dist_info.get("world_size", 1)),
            rank=int(dist_info.get("rank", 0)),
        )
        development_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            **common_loader_kwargs,
        )
        development_loader.evaluation_unit = "normalized_text_cluster"
        development_loader.evaluation_unit_count = len(sampler)
        development_loader.selection_aggregation = aggregation
    else:
        sampler = None
        if dist_info.get("enabled", False):
            sampler = ExactDistributedEvalSampler(
                dataset.estimated_lengths,
                num_replicas=int(dist_info.get("world_size", 1)),
                rank=int(dist_info.get("rank", 0)),
                sort_by_length=bool(
                    cfg.get("train", {}).get("length_bucketed_batches", False)
                ),
            )
        development_loader = DataLoader(
            dataset,
            batch_size=int(getattr(loader, "batch_size", 1) or 1),
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            **common_loader_kwargs,
        )
    partition_cfg = cfg["validation_text_partition"]
    partition_cfg["selection_aggregation_identity"] = (
        sentence_memory_selection_aggregation_identity(cfg)
    )
    partition_cfg["evaluation_corruption_map_identity"] = (
        sentence_memory_validation_corruption_map_identity(
            cfg,
            partition_digest=runtime.partition_digest,
            bank_id=sentence_memory_provider.identity["bank_id"],
        )
    )
    return development_loader, sampler, runtime


class DatasetIndexView(Dataset):
    """Load selected rows while retaining the canonical source manifest."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        self.split = source.split
        # Provider validation must continue to see the complete canonical
        # manifest, never a synthesized development-only manifest.
        self.base = source.base
        self.estimated_lengths = tuple(
            source.estimated_lengths[index] for index in self.indices
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[int(index)]]


def build_development_validation_loader(
    cfg,
    dataset,
    loader,
    sentence_memory_provider,
    dist_info,
):
    """Restrict protected checkpoint selection to fixed novel dev texts."""

    if not requires_development_only_selection(cfg):
        return loader, getattr(loader, "sampler", None), None
    partition_cfg = cfg.get("validation_text_partition")
    if not isinstance(partition_cfg, dict) or not bool(
        partition_cfg.get("enabled", False)
    ):
        raise ValueError(
            "Paired corruption requires validation_text_partition.enabled=true"
        )
    if sentence_memory_provider is None:
        raise ValueError("Development validation partition requires a memory provider")
    rows = list(dataset.base.items)
    expected_rows = partition_cfg.get("expected_validation_rows")
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise RuntimeError(
            "Validation row count differs from validation_text_partition: "
            f"actual={len(rows)}, expected={int(expected_rows)}"
        )
    expected_bank_id = partition_cfg.get("expected_bank_id")
    actual_bank_id = sentence_memory_provider.identity.get("bank_id")
    if expected_bank_id is not None and str(actual_bank_id) != str(expected_bank_id):
        raise RuntimeError(
            "Validation partition bank mismatch: "
            f"actual={actual_bank_id!r}, expected={expected_bank_id!r}"
        )
    val_table = (
        sentence_memory_provider.identity.get("neighbor_tables", {}).get("val", {})
    )
    expected_manifest = partition_cfg.get("expected_validation_manifest_sha256")
    actual_manifest = val_table.get("query_manifest_sha256")
    if expected_manifest is not None and str(actual_manifest) != str(expected_manifest):
        raise RuntimeError(
            "Validation partition manifest mismatch: "
            f"actual={actual_manifest!r}, expected={expected_manifest!r}"
        )

    from NIAF.continuous_trajectory_field.validation_text_partitions import (
        DEVELOPMENT,
        EXACT_SEEN,
        NormalizedTextClusterBatchSampler,
        partition_validation_text_clusters,
    )

    texts = [str(row.get("text", "")) for row in rows]
    partition = partition_validation_text_clusters(
        texts,
        exact_seen=[sentence_memory_provider.is_seen_text(text) for text in texts],
        seed=int(partition_cfg.get("seed", 1234)),
        development_text_count=int(
            partition_cfg.get("development_text_count", 256)
        ),
        expected_novel_text_count=(
            int(partition_cfg["expected_novel_text_count"])
            if partition_cfg.get("expected_novel_text_count") is not None
            else None
        ),
    )
    development_indices = [
        index for index, label in enumerate(partition.labels) if label == DEVELOPMENT
    ]
    if not development_indices:
        raise RuntimeError("Validation development partition contains no rows")
    resolved_counts = partition.artifact_payload["counts"]
    for configured_name, resolved_name in (
        ("expected_development_rows", "development_rows"),
        ("expected_confirmation_rows", "confirmation_rows"),
    ):
        expected = partition_cfg.get(configured_name)
        actual = int(resolved_counts[resolved_name])
        if expected is not None and actual != int(expected):
            raise RuntimeError(
                f"Validation partition {resolved_name} mismatch: "
                f"actual={actual}, expected={int(expected)}"
            )
    expected_digest = partition_cfg.get("expected_partition_digest")
    if expected_digest is not None and str(expected_digest) != str(
        partition.partition_digest
    ):
        raise RuntimeError(
            "Validation partition digest mismatch: "
            f"actual={partition.partition_digest}, expected={expected_digest}"
        )
    subset = DatasetIndexView(dataset, development_indices)
    lengths = [dataset.estimated_lengths[index] for index in development_indices]
    num_workers = int(cfg.get("eval", {}).get("num_workers", 0))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.get("train", {}).get("persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(
            cfg.get("train", {}).get("prefetch_factor", 2)
        )
    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("train", {}).get("pin_memory", True)),
        "collate_fn": loader.collate_fn,
        **loader_kwargs,
    }
    aggregation = configured_selection_aggregation(cfg)
    if aggregation == CLUSTER_EQUAL_SELECTION_AGGREGATION:
        sampler = NormalizedTextClusterBatchSampler(
            [partition.normalized_texts[index] for index in development_indices],
            num_replicas=int(dist_info.get("world_size", 1)),
            rank=int(dist_info.get("rank", 0)),
        )
        development_loader = DataLoader(
            subset,
            batch_sampler=sampler,
            **common_loader_kwargs,
        )
        # These explicit units keep distributed averaging and max_batches from
        # accidentally falling back to signer-row counts.
        development_loader.evaluation_unit = "normalized_text_cluster"
        development_loader.evaluation_unit_count = len(sampler)
        development_loader.selection_aggregation = aggregation
    else:
        sampler = None
        if dist_info.get("enabled", False):
            sampler = ExactDistributedEvalSampler(
                lengths,
                num_replicas=int(dist_info.get("world_size", 1)),
                rank=int(dist_info.get("rank", 0)),
                sort_by_length=bool(
                    cfg.get("train", {}).get("length_bucketed_batches", False)
                ),
            )
        development_loader = DataLoader(
            subset,
            batch_size=int(getattr(loader, "batch_size", 1) or 1),
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            **common_loader_kwargs,
        )
    partition_cfg["partition_digest"] = partition.partition_digest
    partition_cfg["resolved_artifact"] = partition.artifact_payload
    partition_cfg["development_row_count"] = len(development_indices)
    partition_cfg["exact_seen_row_count"] = sum(
        label == EXACT_SEEN for label in partition.labels
    )
    partition_cfg["exact_seen_text_count"] = len(partition.exact_seen_texts)
    # Exact-seen rows are descriptive rather than development evidence.  They
    # are intentionally deferred to the standalone confirmation report so the
    # per-epoch trainer never lets them affect selection or patience.
    partition_cfg["exact_seen_evaluated_during_training"] = False
    partition_cfg["confirmation_evaluated_during_training"] = False
    partition_cfg["selection_aggregation_identity"] = (
        sentence_memory_selection_aggregation_identity(cfg)
    )
    partition_cfg["evaluation_corruption_map_identity"] = (
        sentence_memory_validation_corruption_map_identity(
            cfg,
            partition_digest=partition.partition_digest,
            bank_id=actual_bank_id,
        )
    )
    return development_loader, sampler, partition


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
        "--stage_c_warm_start",
        type=Path,
        default=None,
        help=(
            "Load the exact pinned centered Stage-B best_infeasible model into "
            "a development-only, non-authorizing Stage-C adaptation run while "
            "resetting epoch, optimizer, and checkpoint selection state."
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
    if args.stage_c_warm_start is not None:
        cfg.setdefault("train", {})["stage_c_warm_start_checkpoint"] = str(
            args.stage_c_warm_start
        )
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


def _per_sample_part_compact_smooth_l1(prediction, target, mask, beta):
    """Return masked Smooth-L1 values with the same four-part decomposition."""

    part_slices = (
        slice(0, 60),
        slice(60, 150),
        slice(150, 240),
        slice(240, 256),
    )
    per_part = []
    for part_slice in part_slices:
        difference = F.smooth_l1_loss(
            prediction[..., part_slice],
            target[..., part_slice],
            beta=float(beta),
            reduction="none",
        )
        numerator = (difference * mask[..., None].to(difference.dtype)).sum(
            dim=(1, 2)
        )
        denominator = (
            mask.sum(dim=1).to(difference.dtype)
            * max(part_slice.stop - part_slice.start, 1)
        ).clamp_min(1.0)
        per_part.append(numerator / denominator)
    return torch.stack(per_part, dim=-1)


def _fixed_batch_part_reduction(values, row_mask):
    """Reduce Bx4 values without rank-dependent selected-row normalization."""

    if values.ndim != 2 or values.shape[1] != len(WORD_PRIOR_PART_NAMES):
        raise ValueError("Paired sentence-memory values must have shape [B,4]")
    row_mask = row_mask.to(device=values.device).bool()
    if row_mask.shape != (values.shape[0],):
        raise ValueError("Paired sentence-memory row mask must have shape [B]")
    selected = values * row_mask[:, None].to(values.dtype)
    denominator = max(int(values.shape[0]) * int(values.shape[1]), 1)
    return selected.sum() / denominator


def paired_sentence_memory_losses(
    *,
    correct_prediction,
    corrupt_prediction,
    off_prediction,
    target,
    frame_mask,
    correct_available,
    motion_mask,
    full_shuffle_mask,
    cfg,
):
    """Compute the provenance-bound paired benefit/rank/fallback objective."""

    paired_cfg = paired_sentence_memory_corruption_config(cfg)
    if not paired_cfg["enabled"]:
        raise ValueError("paired_sentence_memory_losses requires paired corruption")
    correct_error = _per_sample_part_compact_l1(
        correct_prediction, target, frame_mask
    )
    corrupt_error = _per_sample_part_compact_l1(
        corrupt_prediction, target, frame_mask
    )
    off_error = _per_sample_part_compact_l1(off_prediction, target, frame_mask)
    denominator = off_error.detach().clamp_min(1e-6)
    benefit_values = F.relu(
        (correct_error - off_error.detach()) / denominator
        + paired_cfg["benefit_margin_relative"]
    )
    corrupt_comparator = (
        corrupt_error.detach()
        if paired_cfg["detach_corrupt_ranking"]
        else corrupt_error
    )
    rank_values = F.relu(
        (correct_error - corrupt_comparator) / denominator
        + paired_cfg["ranking_margin_relative"]
    )
    fallback_values = _per_sample_part_compact_smooth_l1(
        corrupt_prediction,
        off_prediction.detach(),
        frame_mask,
        beta=paired_cfg["fallback_huber_beta"],
    )
    losses = {
        "loss_sentence_benefit": _fixed_batch_part_reduction(
            benefit_values, correct_available
        ),
        "loss_sentence_motion_rank": _fixed_batch_part_reduction(
            rank_values, motion_mask
        ),
        "loss_sentence_motion_fallback": _fixed_batch_part_reduction(
            fallback_values, motion_mask
        ),
        "loss_sentence_full_shuffle_rank": _fixed_batch_part_reduction(
            rank_values, full_shuffle_mask
        ),
        "loss_sentence_full_shuffle_fallback": _fixed_batch_part_reduction(
            fallback_values, full_shuffle_mask
        ),
    }
    diagnostics = {
        "correct_error": correct_error,
        "corrupt_error": corrupt_error,
        "off_error": off_error,
        "benefit_values": benefit_values,
        "rank_values": rank_values,
        "fallback_values": fallback_values,
    }
    return losses, diagnostics


def _globally_synchronized_loss_mean(local_numerator, local_count):
    """Return one global mean with the correct gradient under DDP averaging.

    The detached value is identical on every rank.  Its surrogate gradient is
    scaled by world size because DDP averages parameter gradients after each
    rank contributes its local numerator.
    """

    if local_numerator.ndim != 0:
        raise ValueError("A synchronized loss numerator must be scalar")
    count = torch.as_tensor(
        local_count,
        dtype=local_numerator.dtype,
        device=local_numerator.device,
    ).reshape(())
    if bool(dist.is_available() and dist.is_initialized()):
        world_size = dist.get_world_size()
        global_count = count.detach().clone()
        global_value = local_numerator.detach().clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_value, op=dist.ReduceOp.SUM)
        if float(global_count) <= 0.0:
            return local_numerator * 0.0, global_count, global_value
        loss = global_value / global_count + (
            float(world_size)
            * (local_numerator - local_numerator.detach())
            / global_count
        )
        return loss, global_count, global_value
    if float(count) <= 0.0:
        return local_numerator * 0.0, count, local_numerator.detach()
    return local_numerator / count, count, local_numerator.detach()


def balanced_absolute_association_bce(
    positive_logits,
    positive_mask,
    negative_logits,
    negative_mask,
):
    """Class-balanced absolute match BCE with global valid-pair reductions."""

    positive_mask = positive_mask.to(device=positive_logits.device).bool()
    negative_mask = negative_mask.to(device=negative_logits.device).bool()
    if positive_mask.shape != positive_logits.shape:
        raise ValueError("Positive association mask/logits shapes differ")
    if negative_mask.shape != negative_logits.shape:
        raise ValueError("Negative association mask/logits shapes differ")
    positive_sum = F.softplus(-positive_logits[positive_mask]).sum()
    negative_sum = F.softplus(negative_logits[negative_mask]).sum()
    positive_mean, positive_count, _ = _globally_synchronized_loss_mean(
        positive_sum, positive_mask.sum()
    )
    negative_mean, negative_count, _ = _globally_synchronized_loss_mean(
        negative_sum, negative_mask.sum()
    )
    # Both classes carry exactly one half of the objective regardless of the
    # 90/10 corruption mixture or unequal per-rank valid counts.
    loss = 0.5 * (positive_mean + negative_mean)
    return loss, {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_mean": positive_mean.detach(),
        "negative_mean": negative_mean.detach(),
    }


def _masked_multi_positive_direction(logits, positives, negatives):
    candidate_set = positives | negatives
    valid_anchor = positives.any(dim=1) & negatives.any(dim=1)
    if bool(valid_anchor.any()):
        negative_infinity = torch.finfo(logits.dtype).min
        positive_lse = torch.logsumexp(
            logits.masked_fill(~positives, negative_infinity), dim=1
        )
        candidate_lse = torch.logsumexp(
            logits.masked_fill(~candidate_set, negative_infinity), dim=1
        )
        values = candidate_lse - positive_lse
        numerator = values[valid_anchor].sum()
    else:
        numerator = logits.sum() * 0.0
    mean, count, _ = _globally_synchronized_loss_mean(
        numerator, valid_anchor.sum()
    )
    return mean, count, valid_anchor


def symmetric_masked_multi_positive_infonce(
    key_descriptor,
    motion_descriptor,
    candidate_mask,
    semantic_group_ids,
    candidate_ids,
    *,
    temperature,
    source_group_ids=None,
):
    """Contrast all valid correct candidates in the local physical B*K pool."""

    if key_descriptor.shape != motion_descriptor.shape or key_descriptor.ndim != 3:
        raise ValueError("Association descriptors must share shape [B,K,D]")
    if candidate_mask.shape != key_descriptor.shape[:2]:
        raise ValueError("Association candidate mask must have shape [B,K]")
    for name, value in (
        ("semantic_group_ids", semantic_group_ids),
        ("candidate_ids", candidate_ids),
    ):
        if value is None or value.shape != candidate_mask.shape:
            raise ValueError(f"Association {name} must have shape [B,K]")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Association InfoNCE temperature must be positive")
    keys = key_descriptor.reshape(-1, key_descriptor.shape[-1])
    motions = motion_descriptor.reshape(-1, motion_descriptor.shape[-1])
    valid = candidate_mask.reshape(-1).bool()
    groups = semantic_group_ids.to(device=keys.device).reshape(-1)
    items = candidate_ids.to(device=keys.device).reshape(-1)
    if source_group_ids is None:
        source_groups = torch.full_like(items, -1)
    else:
        if source_group_ids.shape != candidate_mask.shape:
            raise ValueError(
                "Association source_group_ids must have shape [B,K]"
            )
        source_groups = source_group_ids.to(device=keys.device).reshape(-1)
    pair_valid = valid[:, None] & valid[None, :]
    same_semantics = (groups[:, None] == groups[None, :]) & (
        groups[:, None] >= 0
    )
    same_source_variant = (
        (source_groups[:, None] == source_groups[None, :])
        & (source_groups[:, None] >= 0)
    )
    same_item = (items[:, None] == items[None, :]) & (items[:, None] >= 0)
    # All known equivalent realizations are multi-positives. The complement
    # among valid pairs is the true-negative pool, so none can silently become
    # a false negative.
    positives = pair_valid & (same_semantics | same_source_variant | same_item)
    negatives = pair_valid & ~(
        same_semantics | same_source_variant | same_item
    )
    logits = (keys @ motions.transpose(0, 1)) / temperature
    key_loss, key_count, key_valid = _masked_multi_positive_direction(
        logits, positives, negatives
    )
    motion_loss, motion_count, motion_valid = _masked_multi_positive_direction(
        logits.transpose(0, 1), positives.transpose(0, 1), negatives.transpose(0, 1)
    )
    return 0.5 * (key_loss + motion_loss), {
        "key_anchor_count": key_count,
        "motion_anchor_count": motion_count,
        "key_valid_anchor": key_valid,
        "motion_valid_anchor": motion_valid,
        "positive_pair_count": positives.sum().detach(),
        "negative_pair_count": negatives.sum().detach(),
    }


def _association_tensor(outputs, suffix):
    name = f"sentence_memory_association_{suffix}"
    value = outputs.get(name)
    if value is None:
        raise RuntimeError(
            f"Absolute-association model output is missing {name!r}"
        )
    return value


def sentence_memory_association_losses(
    *,
    correct_outputs,
    corrupt_outputs,
    correct_memory,
    corrupt_memory,
    motion_row_mask,
    full_row_mask,
    cfg,
):
    """Build the Stage-B absolute BCE and correct-pool InfoNCE losses."""

    association = configured_sentence_memory_association(cfg)
    if not association["enabled"]:
        raise ValueError("Absolute-association losses require Stage B")
    correct_logits = _association_tensor(correct_outputs, "logit")
    corrupt_logits = _association_tensor(corrupt_outputs, "logit")
    correct_mask = _association_tensor(correct_outputs, "mask").bool()
    corrupt_mask = _association_tensor(corrupt_outputs, "mask").bool()
    if correct_logits.shape != correct_mask.shape or (
        corrupt_logits.shape != corrupt_mask.shape
    ):
        raise RuntimeError("Association aligned logits/masks have different shapes")
    motion_rows = motion_row_mask.to(device=correct_logits.device).bool()
    full_rows = full_row_mask.to(device=correct_logits.device).bool()
    if motion_rows.shape != correct_logits.shape[:1] or (
        full_rows.shape != correct_logits.shape[:1]
    ):
        raise ValueError("Association corruption masks must have shape [B]")

    correct_ids = _sentence_memory_field(correct_memory, "ids")
    corrupt_ids = _sentence_memory_field(corrupt_memory, "ids")
    correct_groups = _sentence_memory_field(correct_memory, "group_ids")
    corrupt_groups = _sentence_memory_field(corrupt_memory, "group_ids")
    correct_source_groups = _sentence_memory_field(
        correct_memory, "source_group_ids"
    )
    corrupt_source_groups = _sentence_memory_field(
        corrupt_memory, "source_group_ids"
    )
    corrupt_motion_ids = _sentence_memory_field(
        corrupt_memory, "motion_source_ids"
    )
    corrupt_motion_groups = _sentence_memory_field(
        corrupt_memory, "motion_source_group_ids"
    )
    corrupt_motion_source_groups = _sentence_memory_field(
        corrupt_memory, "motion_source_source_group_ids"
    )
    if (
        correct_ids is None
        or corrupt_ids is None
        or correct_groups is None
        or correct_source_groups is None
        or corrupt_source_groups is None
        or corrupt_motion_ids is None
        or corrupt_motion_groups is None
        or corrupt_motion_source_groups is None
    ):
        raise RuntimeError(
            "Absolute association requires candidate item, semantic-group, and "
            "source/sign-variant IDs"
        )
    correct_ids = correct_ids.to(correct_logits.device)
    corrupt_ids = corrupt_ids.to(correct_logits.device)
    correct_groups = correct_groups.to(correct_logits.device)
    corrupt_groups = corrupt_groups.to(correct_logits.device)
    correct_source_groups = correct_source_groups.to(correct_logits.device)
    corrupt_source_groups = corrupt_source_groups.to(correct_logits.device)
    corrupt_motion_ids = corrupt_motion_ids.to(correct_logits.device)
    corrupt_motion_groups = corrupt_motion_groups.to(correct_logits.device)
    corrupt_motion_source_groups = corrupt_motion_source_groups.to(
        correct_logits.device
    )

    equivalent_to_destination = (
        (
            (correct_groups == corrupt_motion_groups)
            & (correct_groups >= 0)
        )
        | (
            (correct_source_groups == corrupt_motion_source_groups)
            & (correct_source_groups >= 0)
        )
        | ((correct_ids == corrupt_motion_ids) & (correct_ids >= 0))
    )
    internally_equivalent = (
        ((corrupt_groups == corrupt_motion_groups) & (corrupt_groups >= 0))
        | (
            (corrupt_source_groups == corrupt_motion_source_groups)
            & (corrupt_source_groups >= 0)
        )
        | ((corrupt_ids == corrupt_motion_ids) & (corrupt_ids >= 0))
    )
    malformed_full = (
        corrupt_mask & full_rows[:, None] & ~internally_equivalent
    )
    if bool(malformed_full.any()):
        raise RuntimeError(
            "A full-replacement tuple does not match its recorded motion source"
        )

    # Correct candidate tuples and internally matched full-replacement tuples
    # are positives.  A motion-only derangement breaks each destination
    # diagonal, and original keys paired with full-replacement motion provide
    # the absolute cross-query negatives.
    positive_logits = torch.cat((correct_logits, corrupt_logits), dim=1)
    positive_mask = torch.cat(
        (
            correct_mask,
            corrupt_mask
            & (
                full_rows[:, None]
                | (motion_rows[:, None] & equivalent_to_destination)
            ),
        ),
        dim=1,
    )
    motion_negative_mask = (
        corrupt_mask
        & motion_rows[:, None]
        & ~equivalent_to_destination
    )
    correct_key = _association_tensor(correct_outputs, "key_descriptor")
    corrupt_motion = _association_tensor(corrupt_outputs, "motion_descriptor")
    threshold = _association_tensor(correct_outputs, "threshold")
    cross_cosine = (correct_key * corrupt_motion).sum(dim=-1)
    cross_logits = (
        cross_cosine - threshold.to(cross_cosine.device)
    ) / association["temperature"]
    cross_negative_mask = (
        correct_mask
        & corrupt_mask
        & full_rows[:, None]
        & ~equivalent_to_destination
    )
    negative_logits = torch.cat((corrupt_logits, cross_logits), dim=1)
    negative_mask = torch.cat(
        (motion_negative_mask, cross_negative_mask), dim=1
    )
    bce, bce_diagnostics = balanced_absolute_association_bce(
        positive_logits,
        positive_mask,
        negative_logits,
        negative_mask,
    )
    infonce, infonce_diagnostics = symmetric_masked_multi_positive_infonce(
        correct_key,
        _association_tensor(correct_outputs, "motion_descriptor"),
        correct_mask,
        correct_groups,
        correct_ids,
        temperature=association["temperature"],
        source_group_ids=correct_source_groups,
    )
    return {
        "loss_sentence_association_bce": bce,
        "loss_sentence_association_infonce": infonce,
    }, {
        "bce": bce_diagnostics,
        "infonce": infonce_diagnostics,
        "cross_full_cosine": cross_cosine,
        "cross_full_mask": cross_negative_mask,
    }


def sentence_memory_association_matching_metrics(outputs, memory_batch):
    """Measure within-row symmetric correct-at-1 and positive/negative margin."""

    keys = _association_tensor(outputs, "key_descriptor")
    motions = _association_tensor(outputs, "motion_descriptor")
    support = _association_tensor(outputs, "mask").bool()
    groups = _sentence_memory_field(memory_batch, "group_ids")
    items = _sentence_memory_field(memory_batch, "ids")
    source_groups = _sentence_memory_field(memory_batch, "source_group_ids")
    if groups is None or items is None or source_groups is None:
        raise RuntimeError(
            "Association matching requires semantic-group, source-group, and "
            "item identities"
        )
    groups = groups.to(keys.device)
    items = items.to(keys.device)
    source_groups = source_groups.to(keys.device)
    cosine = torch.einsum("bkd,bjd->bkj", keys, motions)
    pair_valid = support[:, :, None] & support[:, None, :]
    same_semantics = (groups[:, :, None] == groups[:, None, :]) & (
        groups[:, :, None] >= 0
    )
    same_source_variant = (
        (source_groups[:, :, None] == source_groups[:, None, :])
        & (source_groups[:, :, None] >= 0)
    )
    same_item = (items[:, :, None] == items[:, None, :]) & (
        items[:, :, None] >= 0
    )
    positives = pair_valid & (same_semantics | same_source_variant | same_item)
    negatives = pair_valid & ~(same_semantics | same_source_variant | same_item)

    def direction(values, positive_mask, negative_mask):
        valid_anchor = positive_mask.any(dim=-1) & negative_mask.any(dim=-1)
        negative_infinity = torch.finfo(values.dtype).min
        best_positive = values.masked_fill(
            ~positive_mask, negative_infinity
        ).max(dim=-1).values
        best_negative = values.masked_fill(
            ~negative_mask, negative_infinity
        ).max(dim=-1).values
        permitted = positive_mask | negative_mask
        best_index = values.masked_fill(~permitted, negative_infinity).argmax(dim=-1)
        at_one = torch.gather(positive_mask, -1, best_index[..., None]).squeeze(-1)
        return at_one.float(), best_positive - best_negative, valid_anchor

    key_top1, key_margin, key_valid = direction(cosine, positives, negatives)
    motion_top1, motion_margin, motion_valid = direction(
        cosine.transpose(1, 2),
        positives.transpose(1, 2),
        negatives.transpose(1, 2),
    )
    top1_sum = (
        key_top1[key_valid].sum() + motion_top1[motion_valid].sum()
    )
    margin_sum = (
        key_margin[key_valid].sum() + motion_margin[motion_valid].sum()
    )
    count = key_valid.sum() + motion_valid.sum()
    denominator = count.to(dtype=cosine.dtype).clamp_min(1.0)
    return {
        "association_matching_top1": top1_sum / denominator,
        "association_matching_margin": margin_sum / denominator,
        "association_matching_valid_anchor_count": count.to(cosine.dtype),
    }


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
    before backward. The historical function name is retained for callers;
    both Phase B and the isolated Stage-C contract use this exact mechanism.
    """

    if sentence_off_teacher is None:
        raise RuntimeError(
            "Sentence-off distillation is enabled without a frozen teacher"
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
            "Distillation batch has no differentiable text-only student prediction"
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
    if mode in SENTENCE_MEMORY_EVAL_MODES - {"off", "on"}:
        return torch.ones(int(batch_size), dtype=torch.bool, device=device)
    if mode not in SENTENCE_MEMORY_TRAIN_MODES:
        allowed = sorted(
            SENTENCE_MEMORY_TRAIN_MODES
            | (SENTENCE_MEMORY_EVAL_MODES - {"off", "on"})
        )
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
    kwargs = {
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
    candidate_ids = _sentence_memory_field(memory_batch, "ids")
    if candidate_ids is not None:
        kwargs["sentence_candidate_ids"] = candidate_ids
    return kwargs


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
    corruption_nonce=None,
    corruption_seed=None,
    corruption_condition=None,
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
        "corruption_nonce": corruption_nonce,
        "corruption_seed": corruption_seed,
        "corruption_condition": corruption_condition,
    }
    kwargs = {
        name: value for name, value in candidates.items() if name in parameters
    }
    return provider.retrieve(**kwargs)


def deterministic_sentence_shuffle_mask(batch, available, cfg, epoch):
    paired_cfg = paired_sentence_memory_corruption_config(cfg)
    probability = (
        paired_cfg["full_shuffle_probability"]
        if paired_cfg["enabled"]
        else float(
            cfg.get("sentence_memory_safety", {}).get(
                "shuffle_probability", 0.25
            )
        )
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "sentence-memory full-shuffle probability must be in [0, 1]"
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
    if "motion_source_ids" in normal_provenance:
        selected_rows = shuffled_mask.detach().cpu().bool().tolist()
        item_sources = copy.deepcopy(normal_provenance["motion_source_ids"])
        shuffled_ids = _sentence_memory_field(shuffled, "ids").detach().cpu().tolist()
        group_sources = copy.deepcopy(
            normal_provenance.get("motion_source_group_ids")
        )
        variant_sources = copy.deepcopy(
            normal_provenance.get("motion_source_variant_ids")
        )
        shuffled_groups = _sentence_memory_field(shuffled, "group_ids")
        shuffled_groups = (
            shuffled_groups.detach().cpu().tolist()
            if shuffled_groups is not None
            else _sentence_memory_field(shuffled, "provenance", default={}).get(
                "candidate_group_ids"
            )
        )
        shuffled_variants = _sentence_memory_field(
            shuffled, "source_group_ids"
        )
        shuffled_variants = (
            shuffled_variants.detach().cpu().tolist()
            if shuffled_variants is not None
            else None
        )
        for row, selected in enumerate(selected_rows):
            if selected:
                item_sources[row] = shuffled_ids[row]
                if group_sources is not None and shuffled_groups is not None:
                    group_sources[row] = shuffled_groups[row]
                if variant_sources is not None and shuffled_variants is not None:
                    variant_sources[row] = shuffled_variants[row]
        normal_provenance["motion_source_ids"] = item_sources
        normal_provenance["motion_source_group_ids"] = group_sources
        normal_provenance["motion_source_variant_ids"] = variant_sources
    updates["provenance"] = normal_provenance
    if is_dataclass(normal):
        return replace(normal, **updates)
    merged = dict(normal)
    merged.update(updates)
    return merged


def sentence_memory_query_ids(batch):
    """Return stable per-example IDs independent of batch order and DDP rank."""

    names = list(batch.get("name", ()))
    paths = list(batch.get("motion_path", ()))
    indices = batch.get("index")
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().view(-1).tolist()
    elif indices is None:
        indices = ()
    else:
        indices = list(indices)
    batch_size = max(len(names), len(paths), len(indices))
    identifiers = []
    for index in range(batch_size):
        name = str(names[index]) if index < len(names) else ""
        path = str(paths[index]) if index < len(paths) else ""
        if name or path:
            # Dataset-local row indices change under a subset/reordered
            # manifest. Name/path are immutable source identity; use an index
            # only when neither is available.
            identifiers.append(
                json.dumps(
                    ["source_v1", name, path],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        else:
            stable_index = indices[index] if index < len(indices) else index
            identifiers.append(f"index_v1:{stable_index}")
    return identifiers


_SENTENCE_TOKEN_FIELDS = {
    "sentence_motion_tokens",
    "sentence_motion_mask",
    "sentence_motion_tau",
    "sentence_part_validity",
}


def _pad_sentence_token_length(value, token_length):
    if int(value.shape[2]) == int(token_length):
        return value
    shape = list(value.shape)
    shape[2] = int(token_length)
    padded = value.new_zeros(shape)
    slices = [slice(None)] * value.ndim
    slices[2] = slice(0, value.shape[2])
    padded[tuple(slices)] = value
    return padded


def _concatenate_sentence_forward_kwargs(*branches):
    if len(branches) < 1:
        raise ValueError("At least one sentence-memory branch is required")
    keys = set(branches[0])
    if any(set(branch) != keys for branch in branches[1:]):
        raise ValueError("Sentence-memory branches have different forward fields")
    combined = {}
    for name in sorted(keys):
        values = [branch[name] for branch in branches]
        if not all(torch.is_tensor(value) for value in values):
            if any(value is not values[0] for value in values[1:]):
                raise ValueError(
                    f"Non-tensor sentence-memory field {name!r} differs by branch"
                )
            combined[name] = values[0]
            continue
        if name in _SENTENCE_TOKEN_FIELDS:
            token_length = max(int(value.shape[2]) for value in values)
            values = [
                _pad_sentence_token_length(value, token_length) for value in values
            ]
        reference = tuple(values[0].shape[1:])
        if any(tuple(value.shape[1:]) != reference for value in values[1:]):
            raise ValueError(
                f"Sentence-memory field {name!r} has incompatible branch shapes"
            )
        combined[name] = torch.cat(values, dim=0)
    return combined


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


def paired_sentence_memory_model_forward(
    model,
    *,
    text_tokens,
    query_times,
    text_mask,
    query_mask,
    word_kwargs,
    correct_sentence_kwargs,
    corrupt_sentence_kwargs,
):
    """Run paired correct and corrupt retrieval through one DDP forward graph."""

    batch_size = int(text_tokens.shape[0])
    combined_word = _duplicate_optional_batch_kwargs(word_kwargs, batch_size)
    combined_sentence = _concatenate_sentence_forward_kwargs(
        correct_sentence_kwargs,
        corrupt_sentence_kwargs,
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
    correct_outputs = _slice_forward_output_batch(
        combined_outputs, slice(0, batch_size), combined_batch_size
    )
    corrupt_outputs = _slice_forward_output_batch(
        combined_outputs, slice(batch_size, combined_batch_size), combined_batch_size
    )
    return correct_outputs, corrupt_outputs


def sentence_memory_integrity_audit_forwards(
    model,
    *,
    prepared,
    batch,
    cfg,
    broadcast_memory,
):
    """Replay broadcast and arbitrary-payload all-null controls in-process."""

    common = {
        "text_tokens": prepared["text_tokens"],
        "query_times": prepared["tau"],
        "text_mask": prepared["text_mask"],
        "time_domain": "normalized",
        "query_mask": batch["mask"],
        **_prepared_word_forward_kwargs(prepared, batch, cfg),
    }
    broadcast_outputs = model(
        **common,
        **sentence_memory_forward_kwargs(broadcast_memory),
    )
    null_kwargs = sentence_memory_forward_kwargs(
        prepared["sentence_memory_batch"]
    )
    # A true all-null structural input: keep arbitrary payload values in memory
    # but mask every candidate/token/part and invalidate its stable ID.  This
    # exercises the null path without another provider read.
    null_kwargs["sentence_motion_mask"] = torch.zeros_like(
        null_kwargs["sentence_motion_mask"], dtype=torch.bool
    )
    null_kwargs["sentence_part_validity"] = torch.zeros_like(
        null_kwargs["sentence_part_validity"]
    )
    null_kwargs["sentence_candidate_mask"] = torch.zeros_like(
        null_kwargs["sentence_candidate_mask"], dtype=torch.bool
    )
    null_kwargs["sentence_candidate_ids"] = torch.full_like(
        null_kwargs["sentence_candidate_ids"], -1
    )
    all_null_outputs = model(**common, **null_kwargs)
    return broadcast_outputs, all_null_outputs


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
    sentence_memory_corrupt_batch = None
    sentence_memory_corrupt_outputs = None
    sentence_memory_motion_permutation = None
    sentence_memory_control_informative = torch.zeros(
        target.shape[0], dtype=torch.bool, device=device
    )
    sentence_memory_motion_mask = torch.zeros(
        target.shape[0], dtype=torch.bool, device=device
    )
    sentence_memory_full_shuffle_mask = torch.zeros(
        target.shape[0], dtype=torch.bool, device=device
    )
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
                    if resolved_sentence_mode in {"shuffled", "full_replacement"}
                    else "on"
                ),
                **sentence_memory_evaluation_corruption_kwargs(
                    cfg,
                    training=training,
                    condition=resolved_sentence_mode,
                ),
            )
            if not training and resolved_sentence_mode in {
                "shuffled",
                "full_replacement",
            }:
                # Evaluation retrieves a fully shuffled candidate bundle for
                # every available row.  Keep the diagnostic masks faithful to
                # that control even though no training corruption branch runs.
                sentence_memory_full_shuffle_mask = _sentence_memory_field(
                    sentence_memory_batch, "available"
                ).bool()
                sentence_memory_shuffled_mask = (
                    sentence_memory_full_shuffle_mask.clone()
                )
                sentence_memory_control_informative = (
                    sentence_memory_full_shuffle_mask.clone()
                )
            if resolved_sentence_mode in {
                "motion_shuffled",
                "motion_shuffled_n0",
                "motion_shuffled_n1",
                "motion_shuffled_n2",
            }:
                from NIAF.continuous_trajectory_field.sentence_memory import (
                    motion_only_shuffle_sentence_memory_batch,
                )

                (
                    sentence_memory_batch,
                    sentence_memory_motion_permutation,
                    motion_informative,
                ) = motion_only_shuffle_sentence_memory_batch(
                    sentence_memory_batch,
                    query_ids=sentence_memory_query_ids(batch),
                    epoch=int(epoch),
                    seed=int(
                        configured_evaluation_corruption(cfg)["seed"]
                        if not training
                        else cfg.get("seed", 1234)
                    ),
                    **{
                        key: value
                        for key, value in sentence_memory_evaluation_corruption_kwargs(
                            cfg,
                            training=training,
                            condition=resolved_sentence_mode,
                        ).items()
                        if key != "corruption_seed"
                    },
                )
                sentence_memory_motion_mask = (
                    _sentence_memory_field(sentence_memory_batch, "available").bool()
                    & motion_informative.bool()
                )
                sentence_memory_control_informative = (
                    sentence_memory_motion_mask.clone()
                )
            if not training and resolved_sentence_mode == "cross_query_motion":
                from NIAF.continuous_trajectory_field.sentence_memory import (
                    replace_sentence_memory_motion_payload,
                )

                cross_kwargs = sentence_memory_evaluation_corruption_kwargs(
                    cfg, training=False, condition=resolved_sentence_mode
                )
                source_batch = retrieve_sentence_memory(
                    sentence_memory_provider,
                    dataset=dataset,
                    batch=batch,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    predicted_duration=predicted_duration.detach(),
                    available=sentence_availability,
                    training=False,
                    device=device,
                    mode="shuffled",
                    **cross_kwargs,
                )
                sentence_memory_batch, cross_informative = (
                    replace_sentence_memory_motion_payload(
                        sentence_memory_batch,
                        source_batch,
                        corruption_nonce=cross_kwargs["corruption_nonce"],
                        corruption_condition=resolved_sentence_mode,
                    )
                )
                sentence_memory_control_informative = cross_informative.bool()
            if not training and resolved_sentence_mode == "joint_tuple_permuted":
                from NIAF.continuous_trajectory_field.sentence_memory import (
                    joint_tuple_permute_sentence_memory_batch,
                )

                joint_kwargs = sentence_memory_evaluation_corruption_kwargs(
                    cfg, training=False, condition=resolved_sentence_mode
                )
                (
                    sentence_memory_batch,
                    sentence_memory_motion_permutation,
                    joint_informative,
                ) = joint_tuple_permute_sentence_memory_batch(
                    sentence_memory_batch,
                    query_ids=sentence_memory_query_ids(batch),
                    seed=int(joint_kwargs["corruption_seed"]),
                    corruption_nonce=joint_kwargs["corruption_nonce"],
                    corruption_condition=resolved_sentence_mode,
                )
                sentence_memory_control_informative = joint_informative.bool()
            if bool(
                training
                and cfg.get("sentence_memory_safety", {}).get("enabled", False)
            ):
                effective_available = _sentence_memory_field(
                    sentence_memory_batch, "available"
                ).bool()
                requested_full_shuffle = deterministic_sentence_shuffle_mask(
                    batch, effective_available, cfg, epoch
                )
                if paired_sentence_memory_corruption_config(cfg)["enabled"]:
                    from NIAF.continuous_trajectory_field.sentence_memory import (
                        motion_only_shuffle_sentence_memory_batch,
                    )

                    (
                        motion_corrupt_batch,
                        sentence_memory_motion_permutation,
                        motion_informative,
                    ) = motion_only_shuffle_sentence_memory_batch(
                        sentence_memory_batch,
                        query_ids=sentence_memory_query_ids(batch),
                        epoch=int(epoch),
                        seed=int(cfg.get("seed", 1234)),
                    )
                    sentence_memory_corrupt_batch = motion_corrupt_batch
                    if bool(requested_full_shuffle.any()):
                        shuffled_batch = retrieve_sentence_memory(
                            sentence_memory_provider,
                            dataset=dataset,
                            batch=batch,
                            text_tokens=text_tokens,
                            text_mask=text_mask,
                            predicted_duration=predicted_duration.detach(),
                            # Full corruption retains its historical sparse
                            # payload gather; motion-only corruption is in-memory.
                            available=requested_full_shuffle,
                            training=True,
                            device=device,
                            mode="shuffled",
                        )
                        sentence_memory_full_shuffle_mask = (
                            requested_full_shuffle
                            & _sentence_memory_field(
                                shuffled_batch, "available"
                            ).bool()
                        )
                        sentence_memory_corrupt_batch = merge_sentence_memory_batches(
                            sentence_memory_corrupt_batch,
                            shuffled_batch,
                            sentence_memory_full_shuffle_mask,
                        )
                    sentence_memory_motion_mask = (
                        effective_available
                        & motion_informative.bool()
                        & ~sentence_memory_full_shuffle_mask
                    )
                    sentence_memory_shuffled_mask = (
                        sentence_memory_full_shuffle_mask
                    )
                else:
                    sentence_memory_shuffled_mask = requested_full_shuffle
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
            attention_modes = {
                "analytic_prior": "analytic_prior",
                "uniform_final_mass": "uniform_final_candidate_mass",
                "association_disabled": "association_disabled",
            }
            if resolved_sentence_mode in attention_modes:
                sentence_kwargs["sentence_memory_attention_mode"] = attention_modes[
                    resolved_sentence_mode
                ]
            sentence_availability = sentence_kwargs[
                "sentence_memory_available"
            ]
            sentence_memory_shuffled_mask = (
                sentence_memory_shuffled_mask & sentence_availability.bool()
            )

    if is_dual_mode(cfg):
        paired_forward = bool(
            training
            and is_sentence_memory_model(cfg)
            and paired_sentence_memory_corruption_config(cfg)["enabled"]
            and sentence_memory_corrupt_batch is not None
        )
        _off_distillation_phase, off_distillation_cfg = (
            configured_sentence_off_distillation(cfg)
        )
        phase_b_combined_forward = bool(
            training
            and is_sentence_memory_model(cfg)
            and off_distillation_cfg is not None
        )
        if paired_forward:
            outputs, sentence_memory_corrupt_outputs = (
                paired_sentence_memory_model_forward(
                    model,
                    text_tokens=text_tokens,
                    query_times=tau,
                    text_mask=text_mask,
                    query_mask=batch["mask"],
                    word_kwargs=word_kwargs,
                    correct_sentence_kwargs=sentence_kwargs,
                    corrupt_sentence_kwargs=sentence_memory_forward_kwargs(
                        sentence_memory_corrupt_batch
                    ),
                )
            )
        elif phase_b_combined_forward:
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
        "sentence_memory_corrupt_batch": sentence_memory_corrupt_batch,
        "sentence_memory_corrupt_outputs": sentence_memory_corrupt_outputs,
        "sentence_memory_motion_permutation": sentence_memory_motion_permutation,
        "sentence_memory_control_informative": sentence_memory_control_informative,
        "sentence_memory_motion_mask": sentence_memory_motion_mask,
        "sentence_memory_full_shuffle_mask": sentence_memory_full_shuffle_mask,
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
    paired_loss_values = {
        name: target.new_tensor(0.0)
        for name in (
            "loss_sentence_benefit",
            "loss_sentence_motion_rank",
            "loss_sentence_motion_fallback",
            "loss_sentence_full_shuffle_rank",
            "loss_sentence_full_shuffle_fallback",
        )
    }
    paired_diagnostics = None
    association_loss_values = {
        "loss_sentence_association_bce": target.new_tensor(0.0),
        "loss_sentence_association_infonce": target.new_tensor(0.0),
    }
    association_diagnostics = None
    sentence_safety_cfg = cfg.get("sentence_memory_safety", {})
    paired_cfg = paired_sentence_memory_corruption_config(cfg)
    _off_distillation_phase, phase_b_cfg = configured_sentence_off_distillation(
        cfg
    )
    phase_b_enabled = bool(training and phase_b_cfg is not None)
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
        off_error = _per_sample_part_compact_l1(off_prediction, target, mask)
        sentence_off_compact = off_error.mean()
        sentence_off_sample_count = off_error.new_tensor(float(len(off_error)))
        if paired_cfg["enabled"]:
            corrupt_outputs = prepared["sentence_memory_corrupt_outputs"]
            if corrupt_outputs is None:
                raise RuntimeError(
                    "Paired sentence-memory training produced no corrupt output"
                )
            paired_loss_values, paired_diagnostics = paired_sentence_memory_losses(
                correct_prediction=outputs["prediction"],
                corrupt_prediction=corrupt_outputs["prediction"],
                off_prediction=off_prediction,
                target=target,
                frame_mask=mask,
                correct_available=available,
                motion_mask=prepared["sentence_memory_motion_mask"].bool(),
                full_shuffle_mask=prepared[
                    "sentence_memory_full_shuffle_mask"
                ].bool(),
                cfg=cfg,
            )
            if configured_sentence_memory_association(cfg)["enabled"]:
                (
                    association_loss_values,
                    association_diagnostics,
                ) = sentence_memory_association_losses(
                    correct_outputs=outputs,
                    corrupt_outputs=corrupt_outputs,
                    correct_memory=prepared["sentence_memory_batch"],
                    corrupt_memory=prepared["sentence_memory_corrupt_batch"],
                    motion_row_mask=prepared[
                        "sentence_memory_motion_mask"
                    ].bool(),
                    full_row_mask=prepared[
                        "sentence_memory_full_shuffle_mask"
                    ].bool(),
                    cfg=cfg,
                )
        else:
            memory_error = _per_sample_part_compact_l1(
                outputs["prediction"], target, mask
            )
            safe_margin = float(
                sentence_safety_cfg.get("nonregression_margin", 0.0)
            )
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
        if not paired_cfg["enabled"]:
            shuffled_mask = prepared["sentence_memory_shuffled_mask"].bool()
            if bool(shuffled_mask.any()):
                shuffled_frames = mask & shuffled_mask[:, None]
                sentence_shuffle = F.smooth_l1_loss(
                    outputs["prediction"][shuffled_frames],
                    off_prediction[shuffled_frames],
                    beta=float(
                        sentence_safety_cfg.get("shuffle_huber_beta", 0.1)
                    ),
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
    if paired_cfg["enabled"] and training:
        for loss_name, weight_name in (
            ("loss_sentence_benefit", "lambda_sentence_benefit"),
            ("loss_sentence_motion_rank", "lambda_sentence_motion_rank"),
            (
                "loss_sentence_motion_fallback",
                "lambda_sentence_motion_fallback",
            ),
            (
                "loss_sentence_full_shuffle_rank",
                "lambda_sentence_full_shuffle_rank",
            ),
            (
                "loss_sentence_full_shuffle_fallback",
                "lambda_sentence_full_shuffle_fallback",
            ),
        ):
            total = total + float(objective_cfg.get(weight_name, 1.0)) * (
                paired_loss_values[loss_name]
            )
        if configured_sentence_memory_association(cfg)["enabled"]:
            total = total + float(
                objective_cfg["lambda_sentence_association_bce"]
            ) * association_loss_values["loss_sentence_association_bce"]
            total = total + float(
                objective_cfg["lambda_sentence_association_infonce"]
            ) * association_loss_values[
                "loss_sentence_association_infonce"
            ]
    else:
        total = total + float(
            objective_cfg.get("lambda_sentence_safe", 1.0)
        ) * sentence_safe
        total = total + float(
            objective_cfg.get("lambda_sentence_shuffle", 1.0)
        ) * sentence_shuffle
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
        losses.update(paired_loss_values)
        if centered_sentence_memory_enabled(cfg):
            losses.update(association_loss_values)
        losses["loss_sentence_gate_sparsity"] = sentence_gate_sparsity
        losses["loss_sentence_delta_sparsity"] = sentence_delta_sparsity
        losses["loss_sentence_off_distill"] = sentence_off_distill
        if paired_diagnostics is not None:
            motion_mask = prepared["sentence_memory_motion_mask"].bool()
            full_mask = prepared["sentence_memory_full_shuffle_mask"].bool()
            for part_index, part_name in enumerate(WORD_PRIOR_PART_NAMES):
                for label, values in (
                    ("correct", paired_diagnostics["correct_error"]),
                    ("corrupt", paired_diagnostics["corrupt_error"]),
                    ("text_off", paired_diagnostics["off_error"]),
                ):
                    losses[
                        f"sentence_memory_{label}_compact_l1_{part_name}"
                    ] = values[:, part_index].mean()
                for label, row_selector in (
                    ("motion", motion_mask),
                    ("full_shuffle", full_mask),
                ):
                    selected_count = row_selector.float().sum().clamp_min(1.0)
                    losses[
                        f"sentence_memory_{label}_rank_violation_{part_name}"
                    ] = (
                        (
                            paired_diagnostics["rank_values"][:, part_index] > 0
                        ).to(target.dtype)
                        * row_selector.to(target.dtype)
                    ).sum() / selected_count
        if association_diagnostics is not None:
            for name, value in association_diagnostics["bce"].items():
                losses[f"sentence_memory_association_bce_{name}"] = value
            for name, value in association_diagnostics["infonce"].items():
                if torch.is_tensor(value) and value.ndim == 0:
                    losses[f"sentence_memory_association_infonce_{name}"] = value
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
        losses["sentence_memory_motion_corrupt_fraction"] = prepared[
            "sentence_memory_motion_mask"
        ].float().mean()
        losses["sentence_memory_full_shuffle_fraction"] = prepared[
            "sentence_memory_full_shuffle_mask"
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
        if centered_sentence_memory_enabled(cfg):
            association_mask = outputs.get("sentence_memory_association_mask")
            for name in (
                "sentence_memory_relevance_logit",
                "sentence_memory_relevance_gate",
                "sentence_memory_association_cosine",
                "sentence_memory_association_logit",
                "sentence_memory_association_gate",
            ):
                value = outputs.get(name)
                if value is None:
                    continue
                selector = (
                    association_mask.bool()
                    if association_mask is not None
                    else torch.ones_like(value, dtype=torch.bool)
                )
                selected = value[selector]
                losses[name] = (
                    selected.mean() if selected.numel() else value.sum() * 0.0
                )
            threshold = outputs.get("sentence_memory_association_threshold")
            if threshold is not None:
                losses["sentence_memory_association_threshold"] = threshold
            final_mass = outputs.get(
                "sentence_memory_final_part_candidate_mass"
            )
            final_null = outputs.get("sentence_memory_final_part_null_mass")
            if final_mass is not None and final_null is not None:
                candidate_total = final_mass.sum(dim=-1)
                normalized_mass = final_mass / candidate_total[..., None].clamp_min(
                    1e-12
                )
                entropy = -(
                    normalized_mass
                    * normalized_mass.clamp_min(1e-12).log()
                ).sum(dim=-1)
                active = candidate_total > 0
                losses["sentence_memory_final_candidate_mass"] = (
                    candidate_total.mean()
                )
                losses["sentence_memory_final_null_mass"] = final_null.mean()
                losses["sentence_memory_final_candidate_entropy"] = (
                    entropy[active].mean()
                    if bool(active.any())
                    else entropy.sum() * 0.0
                )
                losses["sentence_memory_final_candidate_effective_k"] = (
                    entropy[active].exp().mean()
                    if bool(active.any())
                    else entropy.sum() * 0.0
                )
                losses["sentence_memory_final_candidate_max_share"] = (
                    normalized_mass.max(dim=-1).values[active].mean()
                    if bool(active.any())
                    else normalized_mass.sum() * 0.0
                )
            centered_update = outputs.get("sentence_memory_centered_update")
            if centered_update is not None:
                losses["sentence_memory_centered_update_rms"] = torch.sqrt(
                    centered_update.float().square().mean().clamp_min(1e-24)
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
    if stage_c_enabled(cfg):
        if not freeze_base or train_cfg.get("freeze_sentence_memory") is not True:
            raise RuntimeError(
                "Stage C requires explicit base and sentence-memory freezing"
            )
        if unfreeze_prefixes != STAGE_C_GENERATOR_PREFIXES:
            raise RuntimeError(
                "Stage C trainability differs from the approved generator allowlist"
            )
        named_parameters = list(unwrap_model(model).named_parameters())
        trainable = []
        matched_prefixes = set()
        for name, parameter in named_parameters:
            matched = tuple(
                prefix for prefix in STAGE_C_GENERATOR_PREFIXES if name.startswith(prefix)
            )
            enabled = bool(matched) and not is_sentence_memory_parameter(name)
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(name)
                matched_prefixes.update(matched)
        missing_prefixes = sorted(set(STAGE_C_GENERATOR_PREFIXES) - matched_prefixes)
        if not trainable or missing_prefixes:
            raise RuntimeError(
                "Stage C generator allowlist did not match the complete model; "
                f"missing_prefixes={missing_prefixes}"
            )
        memory_trainable = [
            name
            for name, parameter in named_parameters
            if is_sentence_memory_parameter(name) and parameter.requires_grad
        ]
        if memory_trainable:
            raise RuntimeError(
                "Stage C must freeze every sentence-memory tensor; unexpectedly "
                f"trainable={memory_trainable}"
            )
        actual_trainable_names = tuple(sorted(trainable))
        if actual_trainable_names != STAGE_C_GENERATOR_PARAMETER_NAMES:
            missing = sorted(
                set(STAGE_C_GENERATOR_PARAMETER_NAMES) - set(actual_trainable_names)
            )
            unexpected = sorted(
                set(actual_trainable_names) - set(STAGE_C_GENERATOR_PARAMETER_NAMES)
            )
            raise RuntimeError(
                "Stage C must train exactly the 20 pinned generator tensors; "
                f"missing={missing}, unexpected={unexpected}"
            )
        trainable_parameter_count = sum(
            int(parameter.numel())
            for name, parameter in named_parameters
            if name in STAGE_C_GENERATOR_PARAMETER_NAMES
        )
        if trainable_parameter_count != STAGE_C_GENERATOR_PARAMETER_COUNT:
            raise RuntimeError(
                "Stage-C generator parameter count differs from the pinned "
                f"1,109,395-parameter payload: actual={trainable_parameter_count}"
            )
        tensor_contract = stage_c_trainability_contract()
        return {
            "freeze_base": True,
            "freeze_sentence_memory": True,
            "unfreeze_base_prefixes": unfreeze_prefixes,
            "trainable": list(actual_trainable_names),
            "trainable_parameter_count": trainable_parameter_count,
            "trainability_contract": tensor_contract,
            "stage_c_arm": configured_stage_c(cfg)["arm"],
        }
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
    """Build the frozen v2 teacher used by protected off distillation."""

    phase_name, phase_b = configured_sentence_off_distillation(cfg)
    if phase_b is None:
        return None
    if phase_name == "stage_c":
        teacher_spec = phase_b.get("frozen_v2_teacher") or {}
        checkpoint_path = teacher_spec.get("path")
        if not checkpoint_path:
            raise ValueError(
                "sentence_memory_safety.stage_c.frozen_v2_teacher.path is required"
            )
        teacher_path = Path(checkpoint_path).resolve()
        if not teacher_path.is_file():
            raise RuntimeError(f"Stage-C frozen-v2 teacher does not exist: {teacher_path}")
        actual_sha256 = _sha256_file_stream(teacher_path)
        expected_sha256 = _require_sha256(
            teacher_spec.get("sha256"), label="Stage-C frozen-v2 teacher sha256"
        )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Stage-C frozen-v2 teacher SHA256 mismatch: "
                f"actual={actual_sha256}, expected={expected_sha256}"
            )
        checkpoint_path = teacher_path
    else:
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

    if phase_name == "stage_c":
        # The v2 checkpoint's stored validation is the legacy 1,077-row,
        # row-weighted split and is not comparable to Stage C's sealed
        # 347-row/256-text cluster-equal development view. Warm-start
        # validation has already bound the exact-parity Stage-B memory-off
        # score on that identical development partition instead.
        teacher_score = phase_b.get("teacher_validation_selection_score")
        if teacher_score is None or not math.isfinite(float(teacher_score)):
            raise RuntimeError(
                "Stage C has no like-for-like sealed-development frozen-v2 "
                "validation reference"
            )
    else:
        # Preserve canonical Phase-B behavior exactly.
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


def validate_stage_c_optimizer_parameter_mapping(model, optimizer, cfg):
    """Prove the persisted optimizer-ID/name mapping for Stage C."""

    if not stage_c_enabled(cfg):
        return None
    names_by_parameter_id = {
        id(parameter): name
        for name, parameter in unwrap_model(model).named_parameters()
    }
    actual_groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameter_names = []
        for parameter in group.get("params", ()):
            name = names_by_parameter_id.get(id(parameter))
            if name is None:
                raise RuntimeError(
                    "Stage-C optimizer contains a parameter absent from the model"
                )
            parameter_names.append(name)
        actual_groups.append(
            {
                "group_index": group_index,
                "group_name": group.get("group_name"),
                "parameter_names": parameter_names,
            }
        )
    expected = stage_c_optimizer_parameter_mapping_contract()
    if actual_groups != expected["groups"]:
        raise RuntimeError(
            "Stage-C optimizer parameter ordering/grouping differs from the "
            f"pinned mapping: actual={actual_groups}"
        )
    return expected


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
    "loss_sentence_benefit",
    "loss_sentence_motion_rank",
    "loss_sentence_motion_fallback",
    "loss_sentence_full_shuffle_rank",
    "loss_sentence_full_shuffle_fallback",
    "loss_sentence_gate_sparsity",
    "loss_sentence_delta_sparsity",
    "loss_sentence_off_distill",
    "sentence_memory_motion_corrupt_fraction",
    "sentence_memory_full_shuffle_fraction",
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


def _distributed_finite_flag(local_finite, device, dist_info):
    if not dist_info.get("enabled", False):
        return bool(local_finite)
    backend = str(dist_info.get("backend") or dist.get_backend()).lower()
    flag_device = device if backend == "nccl" else torch.device("cpu")
    flag = torch.tensor(
        int(bool(local_finite)), dtype=torch.int32, device=flag_device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def require_finite_training_losses(total, losses, device, dist_info):
    """Fail all ranks together before backward when any diagnostic is non-finite."""

    nonfinite = []
    for name, value in {"loss_total": total, **losses}.items():
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value.detach()).all())
        else:
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
        if not finite:
            nonfinite.append(name)
    globally_finite = _distributed_finite_flag(
        not nonfinite, device, dist_info
    )
    if not globally_finite:
        raise FloatingPointError(
            "Non-finite training loss/diagnostic detected before backward; "
            f"local_nonfinite={sorted(nonfinite)}"
        )


def require_finite_training_gradients(model, device, dist_info):
    """Fail all ranks together before an optimizer can apply non-finite gradients."""

    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad.detach()).all())
    ]
    globally_finite = _distributed_finite_flag(
        not nonfinite, device, dist_info
    )
    if not globally_finite:
        raise FloatingPointError(
            "Non-finite training gradient detected before optimizer step; "
            f"local_nonfinite={sorted(nonfinite)}"
        )


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
            require_finite_training_losses(total, losses, device, dist_info)
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
            require_finite_training_gradients(model, device, dist_info)
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
        require_finite_training_gradients(model, device, dist_info)
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
    return_prepared=False,
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
    if (
        is_sentence_memory_model(cfg)
        and configured_sentence_memory_association(cfg)["enabled"]
        and str(sentence_memory_mode or "").lower() == "on"
        and prepared.get("sentence_memory_batch") is not None
    ):
        metrics.update(
            tensor_dict_to_float(
                sentence_memory_association_matching_metrics(
                    prepared["outputs"], prepared["sentence_memory_batch"]
                )
            )
        )

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
    if return_prepared:
        return metrics, prepared
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
    cluster_equal = (
        getattr(loader, "evaluation_unit", None) == "normalized_text_cluster"
    )
    progress = tqdm(loader, desc="val", leave=False, disable=not show_progress)
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= int(max_batches):
            break
        logical_size = len(batch["name"])
        if cluster_equal:
            from NIAF.continuous_trajectory_field.sentence_memory import (
                normalize_sentence_text,
            )

            normalized = {
                normalize_sentence_text(value) for value in batch.get("text", ())
            }
            if len(normalized) != 1:
                raise RuntimeError(
                    "Cluster-equal loader batch must contain exactly one "
                    "normalized validation text"
                )
        cluster_average = ScalarAverager() if cluster_equal else None
        # A metric must first be formed independently for each signer row.
        microbatch_size = (
            1 if cluster_equal else validation_microbatch_size(batch, cfg)
        )
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
            if cluster_equal:
                cluster_average.update(metrics, n=1)
            else:
                average.update(metrics, n=end - start)
            if show_progress:
                progress.set_postfix(
                    logical=f"{batch_index + 1}/{len(loader)}",
                    micro=f"{microbatch_index}/{microbatch_count}",
                )
        if cluster_equal:
            average.update(cluster_average.mean(), n=1)
        if (
            bool(cfg.get("eval", {}).get("empty_cache_between_batches", True))
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.empty_cache()
    return average.mean()


_PAIRED_USAGE_RAW_PREFIX = "_paired_usage_raw/"
_PAIRED_CONTROL_MAX_RAW_PREFIX = "_paired_control_max_raw/"
_PAIRED_USAGE_PART_SLICES = {
    "all": slice(0, 256),
    "body": slice(0, 60),
    "left_hand": slice(60, 150),
    "right_hand": slice(150, 240),
    "face": slice(240, 256),
}


def paired_sentence_memory_usage_moments(
    correct_prediction,
    motion_prediction,
    off_prediction,
    frame_mask,
):
    """Return additive moments for exact cross-rank Rmotion reduction."""

    if not (
        correct_prediction.shape
        == motion_prediction.shape
        == off_prediction.shape
    ):
        raise ValueError("Rmotion predictions must have identical shapes")
    if frame_mask.shape != correct_prediction.shape[:2]:
        raise ValueError("Rmotion frame mask has an incompatible shape")
    moments = {}
    for part_name, part_slice in _PAIRED_USAGE_PART_SLICES.items():
        expanded_mask = frame_mask[..., None].expand(
            -1, -1, part_slice.stop - part_slice.start
        )
        for label, comparison in (
            ("motion", motion_prediction),
            ("off", off_prediction),
        ):
            difference = (
                correct_prediction[..., part_slice] - comparison[..., part_slice]
            ).double()
            moments[f"{part_name}/{label}_square_sum"] = float(
                difference[expanded_mask].square().sum().cpu()
            )
            moments[f"{part_name}/{label}_element_count"] = float(
                expanded_mask.sum().cpu()
            )
    return moments


def cluster_mean_paired_usage_moments(row_moments):
    """Average row-level per-element MSEs within one normalized text."""

    rows = list(row_moments)
    if not rows:
        return {}
    output = {}
    for part_name in _PAIRED_USAGE_PART_SLICES:
        for label in ("motion", "off"):
            sum_name = f"{part_name}/{label}_square_sum"
            count_name = f"{part_name}/{label}_element_count"
            values = []
            for row in rows:
                count = float(row.get(count_name, 0.0))
                if count > 0.0:
                    values.append(float(row[sum_name]) / count)
            if values:
                # Downstream distributed reduction treats each normalized text
                # as one additive observation.  The historical field names are
                # retained to keep the external metric schema compatible.
                output[sum_name] = float(sum(values) / len(values))
                output[count_name] = 1.0
    return output


_CENTERED_PAIR_MODES = (
    "motion_shuffled_n0",
    "motion_shuffled_n1",
    "motion_shuffled_n2",
)


def centered_pair_usage_moments(
    correct_prediction,
    pair_predictions,
    off_prediction,
    frame_masks,
):
    """Return additive per-part moments for all three fixed pair controls."""

    if tuple(pair_predictions) != _CENTERED_PAIR_MODES:
        raise ValueError("Rpair requires the three ordered fixed pair controls")
    output = {}
    common_mask = None
    for mode in _CENTERED_PAIR_MODES:
        frame_mask = frame_masks[mode].bool()
        common_mask = frame_mask if common_mask is None else common_mask & frame_mask
    for part_name, part_slice in _PAIRED_USAGE_PART_SLICES.items():
        expanded_mask = common_mask[..., None].expand(
            -1, -1, part_slice.stop - part_slice.start
        )
        off_difference = (
            correct_prediction[..., part_slice] - off_prediction[..., part_slice]
        ).double()
        output[f"{part_name}/pair_off_square_sum"] = float(
            off_difference[expanded_mask].square().sum().cpu()
        )
        output[f"{part_name}/pair_off_element_count"] = float(
            expanded_mask.sum().cpu()
        )
        for nonce_index, mode in enumerate(_CENTERED_PAIR_MODES):
            difference = (
                correct_prediction[..., part_slice]
                - pair_predictions[mode][..., part_slice]
            ).double()
            output[f"{part_name}/pair_n{nonce_index}_square_sum"] = float(
                difference[expanded_mask].square().sum().cpu()
            )
            output[f"{part_name}/pair_n{nonce_index}_element_count"] = float(
                expanded_mask.sum().cpu()
            )
    return output


def cluster_mean_centered_pair_usage_moments(row_moments):
    """Average row MSE within a text before equal-cluster Rpair reduction."""

    rows = list(row_moments)
    if not rows:
        return {}
    output = {}
    labels = ("pair_off", "pair_n0", "pair_n1", "pair_n2")
    for part_name in _PAIRED_USAGE_PART_SLICES:
        for label in labels:
            sum_name = f"{part_name}/{label}_square_sum"
            count_name = f"{part_name}/{label}_element_count"
            values = [
                float(row[sum_name]) / float(row[count_name])
                for row in rows
                if float(row.get(count_name, 0.0)) > 0.0
            ]
            if values:
                output[sum_name] = float(sum(values) / len(values))
                output[count_name] = 1.0
    return output


@torch.no_grad()
def evaluate_paired_sentence_memory_modes(
    model,
    fk,
    text_encoder,
    provider,
    loader,
    dataset,
    cfg,
    device,
    *,
    epoch,
    max_batches,
    show_progress,
    sentence_memory_provider,
):
    """Evaluate all controls together so Rmotion uses exactly paired rows."""

    modes = configured_sentence_memory_eval_modes(cfg)
    centered_protocol = centered_sentence_memory_enabled(cfg)
    required = (
        {"off", "on", *_CENTERED_PAIR_MODES, "cross_query_motion", "full_replacement"}
        if centered_protocol
        else {"off", "on", "motion_shuffled", "shuffled"}
    )
    missing = sorted(required - set(modes))
    if missing:
        raise ValueError(
            "Paired corruption validation requires sentence-memory modes "
            f"{sorted(required)}; missing={missing}"
        )
    namespaces = {
        "off": "text_only",
        "on": "sentence_memory",
        "shuffled": "shuffled_sentence_memory",
        "motion_shuffled": "motion_shuffled_sentence_memory",
        "analytic_prior": "analytic_prior_sentence_memory",
        "motion_shuffled_n0": "motion_shuffled_n0_sentence_memory",
        "motion_shuffled_n1": "motion_shuffled_n1_sentence_memory",
        "motion_shuffled_n2": "motion_shuffled_n2_sentence_memory",
        "cross_query_motion": "cross_query_motion_sentence_memory",
        "full_replacement": "full_replacement_sentence_memory",
        "joint_tuple_permuted": "joint_tuple_permuted_sentence_memory",
        "uniform_final_mass": "uniform_final_mass_sentence_memory",
        "association_disabled": "association_disabled_sentence_memory",
    }
    fixed_word_mode = str(
        cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off")
    ).lower()
    if fixed_word_mode != "off":
        raise ValueError("Paired corruption validation requires the word prior off")
    model.eval()
    averages = {mode: ScalarAverager() for mode in modes}
    cluster_equal = (
        getattr(loader, "evaluation_unit", None) == "normalized_text_cluster"
    )
    configured_cluster_equal = (
        configured_selection_aggregation(cfg)
        == CLUSTER_EQUAL_SELECTION_AGGREGATION
    )
    if cluster_equal != configured_cluster_equal:
        raise RuntimeError(
            "Paired validation loader does not match selection.aggregation: "
            f"loader_cluster_equal={cluster_equal}, "
            f"configured_cluster_equal={configured_cluster_equal}"
        )
    raw_moments = {}
    progress = tqdm(loader, desc="val", leave=False, disable=not show_progress)
    for batch_index, batch in enumerate(progress):
        if max_batches and batch_index >= int(max_batches):
            break
        logical_size = len(batch["name"])
        if cluster_equal:
            from NIAF.continuous_trajectory_field.sentence_memory import (
                normalize_sentence_text,
            )

            normalized = {
                normalize_sentence_text(value) for value in batch.get("text", ())
            }
            if len(normalized) != 1:
                raise RuntimeError(
                    "Cluster-equal loader batch must contain exactly one "
                    "normalized validation text"
                )
        cluster_averages = (
            {mode: ScalarAverager() for mode in modes}
            if cluster_equal
            else None
        )
        cluster_row_moments = []
        microbatch_size = (
            1 if cluster_equal else validation_microbatch_size(batch, cfg)
        )
        microbatch_count = (logical_size + microbatch_size - 1) // microbatch_size
        for microbatch_index, start in enumerate(
            range(0, logical_size, microbatch_size), start=1
        ):
            end = min(start + microbatch_size, logical_size)
            microbatch = slice_batch(batch, start, end)
            predictions = {}
            control_predictions = {}
            control_durations = {}
            correct_prepared = None
            moved_mask = None
            pair_masks = {}
            for mode in modes:
                metrics, prepared = evaluate_microbatch(
                    model,
                    fk,
                    text_encoder,
                    provider,
                    microbatch,
                    dataset,
                    cfg,
                    device,
                    epoch=epoch,
                    word_prior_mode=fixed_word_mode,
                    sentence_memory_provider=sentence_memory_provider,
                    sentence_memory_mode=mode,
                    return_prepared=True,
                )
                if cluster_equal:
                    cluster_averages[mode].update(metrics, n=1)
                else:
                    averages[mode].update(metrics, n=end - start)
                paired_prediction_modes = (
                    {"off", "on", *_CENTERED_PAIR_MODES}
                    if centered_protocol
                    else {"off", "on", "motion_shuffled"}
                )
                if mode in paired_prediction_modes:
                    predictions[mode] = prepared["outputs"]["prediction"].detach()
                    if mode == "motion_shuffled":
                        # Both sides of Rmotion must use exactly the rows on
                        # which a nontrivial motion permutation was possible.
                        informative = prepared[
                            "sentence_memory_motion_mask"
                        ].bool()
                        moved_mask = (
                            microbatch["mask"].to(device).bool()
                            & informative[:, None]
                        )
                    elif mode in _CENTERED_PAIR_MODES:
                        informative = prepared[
                            "sentence_memory_control_informative"
                        ].bool()
                        pair_masks[mode] = (
                            microbatch["mask"].to(device).bool()
                            & informative[:, None]
                        )
                if centered_protocol and mode in {
                    "off",
                    "on",
                    "joint_tuple_permuted",
                    "uniform_final_mass",
                }:
                    control_predictions[mode] = prepared["outputs"][
                        "prediction"
                    ].detach()
                    control_durations[mode] = prepared["outputs"][
                        "trajectory"
                    ].duration_seconds.detach()
                    if mode == "on":
                        correct_prepared = prepared
            if centered_protocol:
                moments = centered_pair_usage_moments(
                    predictions["on"],
                    {mode: predictions[mode] for mode in _CENTERED_PAIR_MODES},
                    predictions["off"],
                    pair_masks,
                )
            else:
                moments = paired_sentence_memory_usage_moments(
                    predictions["on"],
                    predictions["motion_shuffled"],
                    predictions["off"],
                    moved_mask,
                )
            if cluster_equal:
                cluster_row_moments.append(moments)
            else:
                for name, value in moments.items():
                    raw_moments[name] = raw_moments.get(name, 0.0) + value
            if centered_protocol:
                from NIAF.continuous_trajectory_field.sentence_memory import (
                    broadcast_sentence_memory_motion_payload,
                )

                broadcast_memory, _broadcast_source, _broadcast_informative = (
                    broadcast_sentence_memory_motion_payload(
                        correct_prepared["sentence_memory_batch"],
                        query_ids=sentence_memory_query_ids(microbatch),
                        seed=1234,
                    )
                )
                broadcast_outputs, all_null_outputs = (
                    sentence_memory_integrity_audit_forwards(
                        model,
                        prepared=correct_prepared,
                        batch=microbatch,
                        cfg=cfg,
                        broadcast_memory=broadcast_memory,
                    )
                )

                def tensor_max_abs(value):
                    return float(value.detach().abs().max().cpu()) if value.numel() else 0.0

                all_null_trajectory = all_null_outputs["trajectory"]
                all_null_gates = getattr(
                    all_null_trajectory, "sentence_memory_gates", None
                )
                all_null_candidate_mass = all_null_outputs.get(
                    "sentence_memory_final_part_candidate_mass"
                )
                all_null_null_mass = all_null_outputs.get(
                    "sentence_memory_final_part_null_mass"
                )
                control_maxima = {
                    "joint_tuple_prediction_max_abs": float(
                        (
                            control_predictions["on"]
                            - control_predictions["joint_tuple_permuted"]
                        ).abs().max().cpu()
                    ),
                    "joint_tuple_duration_max_abs": tensor_max_abs(
                        control_durations["on"]
                        - control_durations["joint_tuple_permuted"]
                    ),
                    "uniform_final_vs_off_prediction_max_abs": float(
                        (
                            control_predictions["off"]
                            - control_predictions["uniform_final_mass"]
                        ).abs().max().cpu()
                    ),
                    "uniform_final_vs_off_duration_max_abs": tensor_max_abs(
                        control_durations["uniform_final_mass"]
                        - control_durations["off"]
                    ),
                    "broadcast_complete_vs_off_prediction_max_abs": tensor_max_abs(
                        broadcast_outputs["prediction"]
                        - control_predictions["off"]
                    ),
                    "broadcast_complete_vs_off_duration_max_abs": tensor_max_abs(
                        broadcast_outputs["trajectory"].duration_seconds
                        - control_durations["off"]
                    ),
                    "all_null_vs_off_prediction_max_abs": tensor_max_abs(
                        all_null_outputs["prediction"]
                        - control_predictions["off"]
                    ),
                    "all_null_vs_off_duration_max_abs": tensor_max_abs(
                        all_null_trajectory.duration_seconds
                        - control_durations["off"]
                    ),
                    "all_null_gate_max_abs": (
                        tensor_max_abs(all_null_gates)
                        if all_null_gates is not None
                        else float("inf")
                    ),
                    "all_null_candidate_mass_max_abs": (
                        tensor_max_abs(all_null_candidate_mass)
                        if all_null_candidate_mass is not None
                        else float("inf")
                    ),
                    "all_null_one_minus_null_mass_max_abs": (
                        tensor_max_abs(1.0 - all_null_null_mass)
                        if all_null_null_mass is not None
                        else float("inf")
                    ),
                }
                for name, value in control_maxima.items():
                    raw_name = f"{_PAIRED_CONTROL_MAX_RAW_PREFIX}{name}"
                    raw_moments[raw_name] = max(
                        raw_moments.get(raw_name, 0.0), value
                    )
            if show_progress:
                progress.set_postfix(
                    logical=f"{batch_index + 1}/{len(loader)}",
                    micro=f"{microbatch_index}/{microbatch_count}",
                )
        if cluster_equal:
            for mode in modes:
                averages[mode].update(cluster_averages[mode].mean(), n=1)
            cluster_moments = (
                cluster_mean_centered_pair_usage_moments(cluster_row_moments)
                if centered_protocol
                else cluster_mean_paired_usage_moments(cluster_row_moments)
            )
            for name, value in cluster_moments.items():
                raw_moments[name] = raw_moments.get(name, 0.0) + value
        if (
            bool(cfg.get("eval", {}).get("empty_cache_between_batches", True))
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.empty_cache()
    combined = {}
    for mode, average in averages.items():
        namespace = namespaces[mode]
        combined.update(
            {f"{namespace}/{name}": value for name, value in average.mean().items()}
        )
    for name, value in raw_moments.items():
        if name.startswith(_PAIRED_CONTROL_MAX_RAW_PREFIX):
            combined[name] = value
        else:
            combined[f"{_PAIRED_USAGE_RAW_PREFIX}{name}"] = value
    return combined


def distributed_validation_metrics(values, local_sample_count, device, dist_info):
    """Reduce ordinary means and additive paired-usage moments correctly."""

    raw = {
        name.removeprefix(_PAIRED_USAGE_RAW_PREFIX): float(value)
        for name, value in values.items()
        if name.startswith(_PAIRED_USAGE_RAW_PREFIX)
    }
    control_maxima = {
        name.removeprefix(_PAIRED_CONTROL_MAX_RAW_PREFIX): float(value)
        for name, value in values.items()
        if name.startswith(_PAIRED_CONTROL_MAX_RAW_PREFIX)
    }
    ordinary = {
        name: value
        for name, value in values.items()
        if not name.startswith(_PAIRED_USAGE_RAW_PREFIX)
        and not name.startswith(_PAIRED_CONTROL_MAX_RAW_PREFIX)
    }
    ordinary = distributed_sample_weighted_mean_scalars(
        ordinary, local_sample_count, device, dist_info
    )
    if control_maxima:
        control_names = sorted(control_maxima)
        maxima = torch.tensor(
            [control_maxima[name] for name in control_names],
            dtype=torch.float64,
            device=device,
        )
        if dist_info.get("enabled", False):
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        ordinary.update(
            {
                f"paired_sentence_memory/{name}": float(value)
                for name, value in zip(
                    control_names, maxima.detach().cpu().tolist()
                )
            }
        )
    if dist_info.get("enabled", False):
        world_size = int(dist_info.get("world_size") or dist.get_world_size())
        rank_raw_names = [None] * world_size
        dist.all_gather_object(rank_raw_names, sorted(raw))
        raw_names = sorted(
            {name for names_on_rank in rank_raw_names for name in names_on_rank}
        )
    else:
        raw_names = sorted(raw)
    if not raw_names:
        return ordinary
    additive = torch.tensor(
        [raw.get(name, 0.0) for name in raw_names],
        dtype=torch.float64,
        device=device,
    )
    if dist_info.get("enabled", False):
        dist.all_reduce(additive, op=dist.ReduceOp.SUM)
    reduced_raw = {
        name: float(value)
        for name, value in zip(raw_names, additive.detach().cpu().tolist())
    }
    if any(name.endswith("/pair_n0_square_sum") for name in reduced_raw):
        for part_name in _PAIRED_USAGE_PART_SLICES:
            off_sum = reduced_raw[f"{part_name}/pair_off_square_sum"]
            off_count = reduced_raw[f"{part_name}/pair_off_element_count"]
            if off_count <= 0.0:
                raise RuntimeError(f"Rpair {part_name} has no valid off elements")
            off_mse = off_sum / off_count
            nonce_mse = []
            prefix = f"paired_sentence_memory/{part_name}"
            ordinary[f"{prefix}_pair_off_rms"] = math.sqrt(off_mse)
            for nonce_index in range(3):
                label = f"pair_n{nonce_index}"
                square_sum = reduced_raw[f"{part_name}/{label}_square_sum"]
                element_count = reduced_raw[
                    f"{part_name}/{label}_element_count"
                ]
                if element_count <= 0.0:
                    raise RuntimeError(
                        f"Rpair {part_name} nonce {nonce_index} has no valid elements"
                    )
                mse = square_sum / element_count
                nonce_mse.append(mse)
                ordinary[f"{prefix}_{label}_square_sum"] = square_sum
                ordinary[f"{prefix}_{label}_element_count"] = element_count
                ordinary[f"{prefix}_{label}_rms"] = math.sqrt(mse)
                ordinary[f"{prefix}_Rpair_n{nonce_index}"] = math.sqrt(
                    mse / max(off_mse, 1e-24)
                )
            ordinary[f"{prefix}_Rpair"] = math.sqrt(
                (sum(nonce_mse) / len(nonce_mse)) / max(off_mse, 1e-24)
            )
        ordinary["paired_sentence_memory/Rpair"] = ordinary[
            "paired_sentence_memory/all_Rpair"
        ]
        for nonce_index in range(3):
            ordinary[f"paired_sentence_memory/Rpair_n{nonce_index}"] = ordinary[
                f"paired_sentence_memory/all_Rpair_n{nonce_index}"
            ]
        return ordinary
    for part_name in _PAIRED_USAGE_PART_SLICES:
        motion_sum = reduced_raw[f"{part_name}/motion_square_sum"]
        motion_count = reduced_raw[f"{part_name}/motion_element_count"]
        off_sum = reduced_raw[f"{part_name}/off_square_sum"]
        off_count = reduced_raw[f"{part_name}/off_element_count"]
        if motion_count <= 0.0 or off_count <= 0.0:
            raise RuntimeError(f"Rmotion {part_name} has no valid elements")
        motion_rms = math.sqrt(motion_sum / motion_count)
        off_rms = math.sqrt(off_sum / off_count)
        prefix = f"paired_sentence_memory/{part_name}"
        ordinary[f"{prefix}_motion_square_sum"] = motion_sum
        ordinary[f"{prefix}_motion_element_count"] = motion_count
        ordinary[f"{prefix}_off_square_sum"] = off_sum
        ordinary[f"{prefix}_off_element_count"] = off_count
        ordinary[f"{prefix}_motion_rms"] = motion_rms
        ordinary[f"{prefix}_off_rms"] = off_rms
        ordinary[f"{prefix}_Rmotion"] = motion_rms / max(off_rms, 1e-12)
    ordinary["paired_sentence_memory/Rmotion"] = ordinary[
        "paired_sentence_memory/all_Rmotion"
    ]
    return ordinary


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
        # Stage C deliberately disables the paired *training* objective, but
        # retains the centered joint validation protocol (including Rpair and
        # the fixed evidence controls).  Dispatch by the validation protocol,
        # not only by the training-objective flag.
        if (
            paired_sentence_memory_corruption_config(cfg)["enabled"]
            or centered_sentence_memory_enabled(cfg)
        ):
            return evaluate_paired_sentence_memory_modes(
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
                sentence_memory_provider=sentence_memory_provider,
            )
        combined = {}
        namespaces = {
            "off": "text_only",
            "on": "sentence_memory",
            "shuffled": "shuffled_sentence_memory",
            "motion_shuffled": "motion_shuffled_sentence_memory",
            "analytic_prior": "analytic_prior_sentence_memory",
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


def _centered_sentence_memory_selection_diagnostics(
    metrics, cfg, *, return_details=False
):
    """Apply the fixed A''' development gates to cluster-equal metrics."""

    namespaces = {
        "off": "text_only",
        "correct": "sentence_memory",
        "pair_n0": "motion_shuffled_n0_sentence_memory",
        "pair_n1": "motion_shuffled_n1_sentence_memory",
        "pair_n2": "motion_shuffled_n2_sentence_memory",
        "cross": "cross_query_motion_sentence_memory",
        "full": "full_replacement_sentence_memory",
    }
    mode_metrics = {
        label: _metrics_in_namespace(metrics, namespace)
        for label, namespace in namespaces.items()
    }
    missing = [label for label, values in mode_metrics.items() if not values]
    if missing:
        raise ValueError(
            "Centered sentence-memory selection is missing modes: "
            + ", ".join(missing)
        )
    score, violation, feasible, details = selection_diagnostics(
        mode_metrics["correct"], cfg, return_details=True
    )
    violation = float(violation)
    reasons = details["rejection_reasons"]
    scores = {
        label: float(selection_diagnostics(values, cfg)[0])
        for label, values in mode_metrics.items()
    }
    diagnostic = {
        "selection_source": "sentence_memory",
        **{f"{label}_score": value for label, value in scores.items()},
    }

    def add_minimum(name, value, minimum, scale=None, reason=None):
        nonlocal violation, feasible
        scale = max(float(scale if scale is not None else minimum), 1e-8)
        amount = (
            max(float(minimum) - float(value), 0.0) / scale
            if math.isfinite(float(value))
            else 1.0
        )
        violation += amount
        feasible = bool(feasible and amount <= 1e-12)
        diagnostic[name] = float(value)
        diagnostic[f"{name}_minimum"] = float(minimum)
        diagnostic[f"{name}_violation"] = amount
        if amount > 0.0:
            reasons.append(reason or f"{name}={value:.6g} is below {minimum:.6g}")

    def add_relative_win(name, comparator_score, minimum):
        comparator_score = float(comparator_score)
        scale = max(abs(comparator_score), 1e-8)
        gain = (comparator_score - scores["correct"]) / scale
        add_minimum(
            name,
            gain,
            minimum,
            scale=max(minimum, 1e-8),
            reason=(
                f"correct score {scores['correct']:.6g} does not beat {name} "
                f"comparator {comparator_score:.6g} by {100*minimum:.2f}%"
            ),
        )

    def add_maximum(name, value, maximum, scale=None, reason=None):
        nonlocal violation, feasible
        scale = max(float(scale if scale is not None else maximum), 1e-12)
        amount = (
            max(float(value) - float(maximum), 0.0) / scale
            if math.isfinite(float(value))
            else 1.0
        )
        violation += amount
        feasible = bool(feasible and amount <= 1e-12)
        diagnostic[name] = float(value)
        diagnostic[f"{name}_maximum"] = float(maximum)
        diagnostic[f"{name}_violation"] = amount
        if amount > 0.0:
            reasons.append(reason or f"{name}={value:.6g} exceeds {maximum:.6g}")

    add_relative_win("relative_gain_over_off", scores["off"], 0.001)
    pair_scores = [scores[f"pair_n{index}"] for index in range(3)]
    mean_pair_score = sum(pair_scores) / 3.0
    diagnostic["mean_pair_score"] = mean_pair_score
    add_relative_win("relative_gain_over_mean_pair", mean_pair_score, 0.0025)
    add_relative_win("relative_gain_over_cross", scores["cross"], 0.0025)
    add_relative_win("relative_gain_over_full", scores["full"], 0.0025)
    for nonce_index, pair_score in enumerate(pair_scores):
        # Strict per-nonce ordering is represented by one machine-epsilon-sized
        # violation at equality, without imposing an undeclared effect margin.
        scale = max(abs(pair_score), 1e-8)
        strict_gain = (pair_score - scores["correct"]) / scale
        add_minimum(
            f"relative_gain_over_pair_n{nonce_index}",
            strict_gain,
            2e-12,
            scale=1.0,
            reason=(
                f"correct score does not strictly beat pair nonce {nonce_index}"
            ),
        )

    utility_denominator_raw = scores["off"] - scores["correct"]
    utility_numerator = mean_pair_score - scores["correct"]
    utility_denominator_valid = bool(
        math.isfinite(utility_denominator_raw)
        and utility_denominator_raw > 1e-8
        and math.isfinite(utility_numerator)
    )
    utility = (
        utility_numerator / utility_denominator_raw
        if utility_denominator_valid
        else 0.0
    )
    diagnostic["identity_utility_denominator"] = (
        utility_denominator_raw
        if math.isfinite(utility_denominator_raw)
        else 0.0
    )
    diagnostic["identity_utility_denominator_valid"] = utility_denominator_valid
    add_minimum(
        "identity_utility_fraction",
        utility,
        0.25,
        scale=0.25,
        reason=(
            "identity utility is below 0.25 or correct-vs-off denominator is "
            "not finite and greater than 1e-8"
        ),
    )

    add_minimum(
        "Rpair",
        float(metrics.get("paired_sentence_memory/Rpair", math.nan)),
        0.10,
        scale=0.10,
    )
    for nonce_index in range(3):
        add_minimum(
            f"Rpair_n{nonce_index}",
            float(
                metrics.get(
                    f"paired_sentence_memory/Rpair_n{nonce_index}", math.nan
                )
            ),
            0.05,
            scale=0.05,
        )
    add_maximum(
        "joint_tuple_prediction_max_abs",
        float(
            metrics.get(
                "paired_sentence_memory/joint_tuple_prediction_max_abs",
                math.nan,
            )
        ),
        1e-7,
        scale=1e-7,
        reason="joint tuple permutation is not prediction-equivariant within 1e-7",
    )
    add_maximum(
        "joint_tuple_duration_max_abs",
        float(
            metrics.get(
                "paired_sentence_memory/joint_tuple_duration_max_abs",
                math.nan,
            )
        ),
        1e-7,
        scale=1e-7,
        reason="joint tuple permutation is not duration-equivariant within 1e-7",
    )
    add_maximum(
        "uniform_final_vs_off_prediction_max_abs",
        float(
            metrics.get(
                "paired_sentence_memory/uniform_final_vs_off_prediction_max_abs",
                math.nan,
            )
        ),
        0.0,
        scale=1e-12,
        reason="uniform final candidate mass is not exactly memory-off",
    )
    for audit_name in (
        "uniform_final_vs_off_duration_max_abs",
        "broadcast_complete_vs_off_prediction_max_abs",
        "broadcast_complete_vs_off_duration_max_abs",
        "all_null_vs_off_prediction_max_abs",
        "all_null_vs_off_duration_max_abs",
        "all_null_gate_max_abs",
        "all_null_candidate_mass_max_abs",
        "all_null_one_minus_null_mass_max_abs",
    ):
        add_maximum(
            audit_name,
            float(
                metrics.get(
                    f"paired_sentence_memory/{audit_name}", math.nan
                )
            ),
            0.0,
            scale=1e-12,
            reason=f"integrity audit {audit_name} is not exactly zero",
        )
    parity = cfg.get("sentence_memory_safety", {}).get(
        "v2_to_v3_text_only_parity"
    )
    if stage_c_enabled(cfg):
        # Generator adaptation intentionally invalidates source-time equality.
        # Do not present the inherited zero-valued proof as a live metric.
        provenance = configured_stage_c(cfg).get("resolved_source_provenance") or {}
        diagnostic["v2_text_only_parity"] = {
            "applicable": False,
            "reason": "source_initialization_only_generator_adapted",
            "source_initialization": copy.deepcopy(
                provenance.get("source_initialization_v2_text_only_parity")
            ),
        }
    else:
        parity_details = {}
        for parity_name in ("prediction_max_abs", "duration_max_abs"):
            parity_value = (
                float(parity.get(parity_name, math.nan))
                if isinstance(parity, dict) and bool(parity.get("passed", False))
                else math.nan
            )
            add_maximum(
                f"v2_{parity_name}",
                parity_value,
                1e-7,
                scale=1e-7,
                reason=f"stored v2 {parity_name} parity proof exceeds 1e-7",
            )
            parity_details[parity_name] = parity_value
        diagnostic["v2_text_only_parity"] = parity_details

    off_phase_name, off_phase_cfg = configured_sentence_off_distillation(cfg)
    if off_phase_cfg is not None:
        teacher_score = off_phase_cfg.get("teacher_validation_selection_score")
        if teacher_score is None:
            raise ValueError(
                f"{off_phase_name} selection requires the validation score "
                "bound to its frozen v2 teacher"
            )
        teacher_score = float(teacher_score)
        tolerance = float(
            off_phase_cfg.get("text_only_max_relative_degradation", 0.005)
        )
        if tolerance < 0.0:
            raise ValueError(
                f"{off_phase_name}.text_only_max_relative_degradation must be "
                "non-negative"
            )
        scale = max(abs(teacher_score), 1e-8)
        allowed = teacher_score + tolerance * scale
        teacher_violation = (
            max(scores["off"] - allowed, 0.0) / scale
            if math.isfinite(scores["off"]) and math.isfinite(teacher_score)
            else float("inf")
        )
        violation += teacher_violation
        feasible = bool(feasible and teacher_violation <= 1e-12)
        if teacher_violation > 0.0:
            reasons.append(
                f"{off_phase_name} live text-only score {scores['off']:.6g} "
                f"is more than {100.0 * tolerance:.2f}% worse than frozen-v2 "
                f"teacher score {teacher_score:.6g}"
            )
        diagnostic[f"{off_phase_name}_text_only_guard"] = {
            "live_text_only_score": scores["off"],
            "teacher_score": teacher_score,
            "allowed_text_only_score": allowed,
            "max_relative_degradation": tolerance,
            "violation": teacher_violation,
        }

    hand_details = {}
    for hand in ("lhand", "rhand"):
        metric_name = f"pred_loss_path_{hand}"
        correct_value = float(mode_metrics["correct"].get(metric_name, math.nan))
        off_value = float(mode_metrics["off"].get(metric_name, math.nan))
        hand_scale = max(abs(off_value), 1e-8) if math.isfinite(off_value) else 1.0
        degradation = (
            (correct_value - off_value) / hand_scale
            if math.isfinite(correct_value) and math.isfinite(off_value)
            else math.nan
        )
        # Express an upper bound as a lower-bound gate on its negation.
        add_minimum(
            f"{hand}_negative_degradation_vs_off",
            -degradation,
            -0.02,
            scale=0.02,
            reason=f"correct {hand} path degrades by more than 2% versus off",
        )
        corruption_rows = {}
        for label in ("pair_n0", "pair_n1", "pair_n2", "cross", "full"):
            corrupt_value = float(mode_metrics[label].get(metric_name, math.nan))
            gain = (
                (corrupt_value - correct_value) / max(abs(corrupt_value), 1e-8)
                if math.isfinite(correct_value) and math.isfinite(corrupt_value)
                else math.nan
            )
            add_minimum(
                f"{hand}_gain_over_{label}",
                gain,
                0.0,
                scale=1.0,
                reason=f"correct {hand} path error exceeds {label}",
            )
            corruption_rows[label] = corrupt_value
        hand_details[hand] = {
            "correct": correct_value,
            "off": off_value,
            "relative_degradation_vs_off": degradation,
            "corruptions": corruption_rows,
        }
    diagnostic["hand_path_gates"] = hand_details

    if configured_sentence_memory_association(cfg)["enabled"]:
        add_minimum(
            "association_matching_top1",
            float(
                mode_metrics["correct"].get(
                    "association_matching_top1", math.nan
                )
            ),
            0.25,
            scale=0.25,
        )
        add_minimum(
            "association_matching_margin",
            float(
                mode_metrics["correct"].get(
                    "association_matching_margin", math.nan
                )
            ),
            2e-12,
            scale=1.0,
            reason="mean true-minus-best-negative association margin is not positive",
        )

    nonfinite = sorted(
        name for name, value in metrics.items() if not math.isfinite(float(value))
    )
    if nonfinite:
        violation += float(len(nonfinite))
        feasible = False
        reasons.append("non-finite validation metrics: " + ", ".join(nonfinite))
        diagnostic["nonfinite_metric_count"] = len(nonfinite)
    details["dual_mode"] = diagnostic
    result = (float(score), float(violation), bool(feasible))
    return (*result, details) if return_details else result


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
        if centered_sentence_memory_enabled(cfg):
            return _centered_sentence_memory_selection_diagnostics(
                metrics, cfg, return_details=return_details
            )
        memory_metrics = _metrics_in_namespace(metrics, "sentence_memory")
        shuffled_metrics = _metrics_in_namespace(
            metrics, "shuffled_sentence_memory"
        )
        motion_shuffled_metrics = _metrics_in_namespace(
            metrics, "motion_shuffled_sentence_memory"
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

        off_phase_name, phase_b_cfg = configured_sentence_off_distillation(cfg)
        if phase_b_cfg is not None:
            teacher_score = phase_b_cfg.get("teacher_validation_selection_score")
            if teacher_score is None:
                raise ValueError(
                    f"{off_phase_name} selection requires the validation score "
                    "bound to its frozen v2 teacher"
                )
            tolerance = float(
                phase_b_cfg.get("text_only_max_relative_degradation", 0.005)
            )
            if tolerance < 0:
                raise ValueError(
                    f"{off_phase_name}.text_only_max_relative_degradation must "
                    "be non-negative"
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
                    f"{off_phase_name} text-only score "
                    f"{current_text_score:.6g} is more than "
                    f"{100.0 * tolerance:.2f}% worse than frozen-v2 teacher "
                    f"score {teacher_score:.6g}"
                )
            memory_details[f"{off_phase_name}_text_only_guard"] = {
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
        require_motion_improvement = bool(
            cfg.get("selection", {}).get(
                "require_sentence_memory_outperform_motion_shuffled",
                paired_sentence_memory_corruption_config(cfg)["enabled"],
            )
        )
        if memory_metrics and motion_shuffled_metrics:
            memory_score = float(selection_diagnostics(memory_metrics, cfg)[0])
            motion_score = float(
                selection_diagnostics(motion_shuffled_metrics, cfg)[0]
            )
            motion_minimum_improvement = float(
                cfg.get("selection", {}).get(
                    "sentence_memory_min_relative_improvement_over_motion_shuffled",
                    0.001,
                )
            )
            if motion_minimum_improvement < 0:
                raise ValueError(
                    "sentence_memory_min_relative_improvement_over_motion_shuffled "
                    "must be non-negative"
                )
            motion_scale = max(abs(motion_score), 1e-8)
            allowed_memory_score = (
                motion_score - motion_minimum_improvement * motion_scale
            )
            motion_violation = (
                max(memory_score - allowed_memory_score, 0.0) / motion_scale
                if math.isfinite(memory_score) and math.isfinite(motion_score)
                else float("inf")
            )
            if require_motion_improvement:
                violation = float(violation) + motion_violation
                feasible = bool(feasible and motion_violation <= 1e-12)
                if motion_violation > 0:
                    details["rejection_reasons"].append(
                        "sentence-memory score "
                        f"{memory_score:.6g} did not outperform motion-shuffled "
                        f"memory score {motion_score:.6g} by the required "
                        f"{100.0 * motion_minimum_improvement:.2f}%"
                    )
            memory_details.update(
                {
                    "motion_shuffled_sentence_memory_score": motion_score,
                    "sentence_memory_allowed_score_vs_motion_shuffled": (
                        allowed_memory_score
                    ),
                    "sentence_memory_relative_improvement_over_motion_shuffled": (
                        (motion_score - memory_score) / motion_scale
                    ),
                    "sentence_memory_min_relative_improvement_over_motion_shuffled": (
                        motion_minimum_improvement
                    ),
                    "sentence_memory_motion_shuffled_improvement_violation": (
                        motion_violation
                    ),
                    "require_sentence_memory_outperform_motion_shuffled": float(
                        require_motion_improvement
                    ),
                }
            )
        elif require_motion_improvement:
            violation = float("inf")
            feasible = False
            details["rejection_reasons"].append(
                "required sentence_memory and motion_shuffled_sentence_memory "
                "validation metrics were not both produced"
            )

        paired_objective = paired_sentence_memory_corruption_config(cfg)["enabled"]
        if paired_objective:
            minimum_rmotion = float(
                cfg.get("selection", {}).get("sentence_memory_min_Rmotion", 0.05)
            )
            if not math.isfinite(minimum_rmotion) or minimum_rmotion < 0.0:
                raise ValueError(
                    "selection.sentence_memory_min_Rmotion must be finite and "
                    "non-negative"
                )
            rmotion = float(
                metrics.get("paired_sentence_memory/Rmotion", float("nan"))
            )
            rmotion_violation = (
                max(minimum_rmotion - rmotion, 0.0) / max(minimum_rmotion, 1e-8)
                if math.isfinite(rmotion)
                else float("inf")
            )
            violation = float(violation) + rmotion_violation
            feasible = bool(feasible and rmotion_violation <= 1e-12)
            if rmotion_violation > 0:
                details["rejection_reasons"].append(
                    f"paired compact-output Rmotion={rmotion:.6g} is below the "
                    f"required {minimum_rmotion:.6g}"
                )
            memory_details.update(
                {
                    "Rmotion": rmotion,
                    "minimum_Rmotion": minimum_rmotion,
                    "Rmotion_violation": rmotion_violation,
                }
            )
            for part_name in ("body", "left_hand", "right_hand", "face"):
                memory_details[f"Rmotion_{part_name}"] = float(
                    metrics.get(
                        f"paired_sentence_memory/{part_name}_Rmotion",
                        float("nan"),
                    )
                )
            nonfinite_metrics = sorted(
                name
                for name, value in metrics.items()
                if not math.isfinite(float(value))
            )
            if nonfinite_metrics:
                # A diagnostic that is not part of the scalar selection score
                # must still make the checkpoint scientifically infeasible.
                # Use a finite normalized penalty so lexicographic patience and
                # exact checkpoint state remain serializable.
                if not math.isfinite(float(violation)):
                    violation = 0.0
                violation = float(violation) + float(len(nonfinite_metrics))
                if not math.isfinite(float(score)):
                    score = float(np.finfo(np.float64).max)
                feasible = False
                details["rejection_reasons"].append(
                    "non-finite validation losses/diagnostics: "
                    + ", ".join(nonfinite_metrics)
                )
                memory_details["nonfinite_metric_count"] = len(
                    nonfinite_metrics
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


def checkpoint_selection_state(
    best_score,
    best_infeasible_score,
    *,
    early_stopping_state=None,
    best_infeasible_key=None,
):
    """Serialize historical checkpoint-selection minima without infinities."""

    serialized_infeasible_key = None
    if best_infeasible_key is not None:
        if not isinstance(best_infeasible_key, (list, tuple)) or len(
            best_infeasible_key
        ) != 2:
            raise RuntimeError(
                "best_infeasible_key must contain constraint violation and score"
            )
        serialized_infeasible_key = [
            float(best_infeasible_key[0]),
            float(best_infeasible_key[1]),
        ]
        if not all(math.isfinite(value) for value in serialized_infeasible_key):
            raise RuntimeError("best_infeasible_key must contain finite values")
    state = {
        "schema_version": (
            3
            if best_infeasible_key is not None
            else (2 if early_stopping_state is not None else 1)
        ),
        "best_feasible_score": (
            float(best_score) if math.isfinite(float(best_score)) else None
        ),
        "best_infeasible_score": (
            float(best_infeasible_score)
            if math.isfinite(float(best_infeasible_score))
            else None
        ),
    }
    if early_stopping_state is not None:
        state["early_stopping"] = copy.deepcopy(early_stopping_state)
    if serialized_infeasible_key is not None:
        state["best_infeasible_key"] = serialized_infeasible_key
    return state


def selected_checkpoint_artifact_name(cfg, *, feasible):
    """Return a promotion-safe filename for a selected validation artifact."""

    if stage_c_enabled(cfg):
        return (
            "best_exploratory.pt"
            if bool(feasible)
            else "best_exploratory_infeasible.pt"
        )
    return "best.pt" if bool(feasible) else "best_infeasible.pt"


def restore_checkpoint_selection_scores(checkpoint, *, require=False, source="checkpoint"):
    state = checkpoint.get("selection_state")
    if not isinstance(state, dict) or int(state.get("schema_version", -1)) not in {
        1,
        2,
        3,
    }:
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


def restore_best_infeasible_selection_key(
    checkpoint,
    *,
    require=False,
    source="checkpoint",
):
    """Restore the lexicographic infeasible (violation, score) artifact key."""

    state = checkpoint.get("selection_state")
    value = state.get("best_infeasible_key") if isinstance(state, dict) else None
    best_score = (
        state.get("best_infeasible_score") if isinstance(state, dict) else None
    )
    if value is None:
        if require and best_score is not None:
            raise RuntimeError(
                f"{source} has no exact best-infeasible lexicographic key"
            )
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"{source} has a malformed best_infeasible_key")
    restored = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in restored):
        raise RuntimeError(f"{source} has a non-finite best_infeasible_key")
    if best_score is None or float(best_score) != restored[1]:
        raise RuntimeError(
            f"{source} best_infeasible_key disagrees with best_infeasible_score"
        )
    return restored


def initial_early_stopping_state():
    return {
        "schema_version": 1,
        "best_key": None,
        "last_key": None,
        "bad_validation_count": 0,
        "validation_count": 0,
        "last_validation_epoch": None,
        "stopped": False,
        "stop_epoch": None,
    }


def _validate_early_stopping_key(value, source):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RuntimeError(f"{source} must contain a three-value lexicographic key")
    feasibility_rank = int(value[0])
    if feasibility_rank not in {0, 1} or float(value[0]) != feasibility_rank:
        raise RuntimeError(f"{source} has an invalid feasibility rank")
    violation = float(value[1])
    score = float(value[2])
    if not math.isfinite(violation) or not math.isfinite(score):
        raise RuntimeError(f"{source} contains a non-finite value")
    return (feasibility_rank, violation, score)


def restore_early_stopping_state(
    checkpoint,
    *,
    require=False,
    source="checkpoint",
):
    selection_state = checkpoint.get("selection_state")
    payload = (
        selection_state.get("early_stopping")
        if isinstance(selection_state, dict)
        else None
    )
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        if require:
            raise RuntimeError(f"{source} has no exact early-stopping state")
        return initial_early_stopping_state()
    restored = copy.deepcopy(payload)
    for name in ("bad_validation_count", "validation_count"):
        value = restored.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{source} early_stopping.{name} is invalid")
    for name in ("best_key", "last_key"):
        value = restored.get(name)
        if value is not None:
            restored[name] = list(
                _validate_early_stopping_key(
                    value, f"{source} early_stopping.{name}"
                )
            )
    for name in ("last_validation_epoch", "stop_epoch"):
        value = restored.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise RuntimeError(f"{source} early_stopping.{name} is invalid")
    restored["stopped"] = bool(restored.get("stopped", False))
    if restored["stopped"] != (restored.get("stop_epoch") is not None):
        raise RuntimeError(f"{source} has inconsistent early-stopping terminal state")
    return restored


def update_early_stopping_state(
    state,
    *,
    feasible,
    normalized_constraint_violation,
    selection_score_value,
    epoch,
    patience,
    minimum_epoch=2,
):
    """Update strict lexicographic validation patience without hidden state."""

    state = copy.deepcopy(state)
    if bool(state.get("stopped", False)):
        raise RuntimeError("Cannot update an early-stopping state after it stopped")
    key = (
        0 if bool(feasible) else 1,
        float(normalized_constraint_violation),
        float(selection_score_value),
    )
    key = _validate_early_stopping_key(key, "active early-stopping key")
    previous = state.get("best_key")
    previous_key = (
        _validate_early_stopping_key(previous, "early-stopping best_key")
        if previous is not None
        else None
    )
    improved = previous_key is None or key < previous_key
    state["validation_count"] = int(state.get("validation_count", 0)) + 1
    state["last_validation_epoch"] = int(epoch)
    state["last_key"] = list(key)
    if improved:
        state["best_key"] = list(key)
        state["bad_validation_count"] = 0
    else:
        state["bad_validation_count"] = int(
            state.get("bad_validation_count", 0)
        ) + 1
    patience = int(patience)
    minimum_epoch = int(minimum_epoch)
    should_stop = bool(
        patience > 0
        and int(epoch) >= minimum_epoch
        and int(state["bad_validation_count"]) >= patience
    )
    state["stopped"] = should_stop
    state["stop_epoch"] = int(epoch) if should_stop else None
    return state, improved, should_stop


def pending_validation_resume_state(checkpoint, *, paired_objective_enabled):
    """Resolve an interrupted post-train/pre-validation epoch exactly."""

    metrics = checkpoint.get("metrics") or {}
    try:
        validation_pending = float(metrics.get("validation_pending", 0.0)) == 1.0
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "checkpoint metrics.validation_pending is malformed"
        ) from error
    if not validation_pending:
        return None, None
    if not paired_objective_enabled:
        raise RuntimeError(
            "checkpoint ended before validation; exact recovery of pending "
            "validation is implemented only for paired Phase A"
        )
    epoch = int(checkpoint.get("epoch", 0))
    if epoch < 1:
        raise RuntimeError("checkpoint has an invalid pending-validation epoch")
    if int(metrics.get("epoch", epoch)) != epoch:
        raise RuntimeError(
            "checkpoint pending-validation metric epoch does not match its epoch"
        )
    return epoch, copy.deepcopy(metrics)


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
    payload = {
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
            "sentence_memory_objective_identity": (
                sentence_memory_objective_identity(cfg)
                if is_sentence_memory_model(cfg)
                else None
            ),
            "sentence_memory_architecture_identity": (
                sentence_memory_architecture_identity(cfg)
                if is_sentence_memory_model(cfg)
                else None
            ),
            "sentence_memory_relevance_calibration_identity": (
                cfg.get("sentence_memory", {}).get(
                    "resolved_relevance_calibration_identity"
                )
                if is_sentence_memory_model(cfg)
                else None
            ),
            "sentence_memory_evaluation_control_identity": (
                sentence_memory_evaluation_control_identity(cfg)
                if is_sentence_memory_model(cfg)
                else None
            ),
            "sentence_memory_selection_aggregation_identity": (
                sentence_memory_selection_aggregation_identity(cfg)
                if is_sentence_memory_model(cfg)
                else None
            ),
            "sentence_memory_validation_corruption_map_identity": (
                cfg.get("validation_text_partition", {}).get(
                    "evaluation_corruption_map_identity"
                )
                if is_sentence_memory_model(cfg)
                else None
            ),
            "v2_to_v3_text_only_parity": cfg.get(
                "sentence_memory_safety", {}
            ).get("v2_to_v3_text_only_parity"),
            "phase_b_scientific_gate": cfg.get(
                "sentence_memory_safety", {}
            ).get("phase_b", {}).get("resolved_scientific_gate"),
            "stage_c_provenance": (
                copy.deepcopy(
                    configured_stage_c(cfg).get("resolved_source_provenance")
                )
                if stage_c_enabled(cfg)
                else None
            ),
            "stage_c_distribution_contract": (
                copy.deepcopy(
                    configured_stage_c(cfg).get(
                        "resolved_distribution_contract"
                    )
                )
                if stage_c_enabled(cfg)
                else None
            ),
            "stage_c_trainability_contract": (
                stage_c_trainability_contract()
                if stage_c_enabled(cfg)
                else None
            ),
            "stage_c_optimizer_parameter_mapping": (
                validate_stage_c_optimizer_parameter_mapping(
                    model, optimizer, cfg
                )
                if stage_c_enabled(cfg)
                else None
            ),
            "stage_c_authorization_scope": (
                {
                    "schema_name": STAGE_C_SCHEMA_NAME,
                    "schema_version": STAGE_C_SCHEMA_VERSION,
                    "arm": configured_stage_c(cfg).get("arm"),
                    "development_only": True,
                    "non_authorizing": True,
                    "promotion_eligible": False,
                    "confirmation_manifest_opened": False,
                    "test_data_accessed": False,
                }
                if stage_c_enabled(cfg)
                else None
            ),
        }

    # ``last.pt`` is the sole exact-continuation boundary for an interrupted
    # epoch. Never truncate the previous valid checkpoint in place: serialize
    # beside it, durably flush when the filesystem supports that operation, and
    # publish with one atomic replacement.
    temporary_path = None
    descriptor = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            torch.save(payload, handle)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as error:
                unsupported = {
                    errno.EINVAL,
                    getattr(errno, "ENOTSUP", errno.EINVAL),
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }
                if error.errno not in unsupported:
                    raise
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                try:
                    os.fsync(directory_descriptor)
                except OSError as error:
                    unsupported = {
                        errno.EINVAL,
                        getattr(errno, "ENOTSUP", errno.EINVAL),
                        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                    }
                    if error.errno not in unsupported:
                        raise
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json_atomic(path, payload):
    """Durably publish JSON without truncating an existing valid target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_path = None
    descriptor = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as error:
                unsupported = {
                    errno.EINVAL,
                    getattr(errno, "ENOTSUP", errno.EINVAL),
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }
                if error.errno not in unsupported:
                    raise
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                try:
                    os.fsync(directory_descriptor)
                except OSError as error:
                    unsupported = {
                        errno.EINVAL,
                        getattr(errno, "ENOTSUP", errno.EINVAL),
                        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                    }
                    if error.errno not in unsupported:
                        raise
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
        for value in (
            args.resume,
            args.warm_start,
            args.stage_c_warm_start,
            args.base_checkpoint,
        )
    ) > 1:
        raise ValueError(
            "--resume, --warm_start, --stage_c_warm_start, and "
            "--base_checkpoint are mutually exclusive"
        )
    cfg = apply_overrides(load_config(args.config), args)
    if centered_sentence_memory_enabled(cfg):
        resolve_sentence_memory_relevance_calibration(cfg)
    validate_paired_sentence_memory_training_contract(cfg)
    validate_phase_b_launch(
        cfg,
        warm_start=args.warm_start,
        resume=args.resume,
        phase_b_gate_report=args.phase_b_gate_report,
    )
    validate_stage_c_training_contract(
        cfg,
        stage_c_warm_start=args.stage_c_warm_start,
        resume=args.resume,
    )
    if is_sentence_memory_model(cfg):
        cfg.setdefault("sentence_memory", {})[
            "resolved_behavior_identity"
        ] = sentence_memory_behavior_identity(cfg)
    isolated_validation_runtime = load_isolated_development_validation_runtime(
        cfg
    )
    if (
        args.warm_start is not None or args.stage_c_warm_start is not None
    ) and configured_model_type(cfg) == DUAL_MODE_MODEL_TYPE:
        raise ValueError(
            "Dual-mode v2 must be trained from scratch and does not accept "
            "--warm_start. Use --resume for an exact v2 continuation."
        )
    out_dir = Path(cfg["output"]["out_dir"])
    validate_fresh_output_directory(out_dir, resume=args.resume)
    dist_info = setup_distributed(args)
    set_seed(int(cfg.get("seed", 1234)) + int(dist_info.get("rank", 0)))
    device = resolve_distributed_device(cfg.get("device", "auto"), dist_info)
    validate_stage_c_distributed_runtime(cfg, dist_info, device)
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
    validation_loader_cfg = isolated_development_validation_config(
        cfg, isolated_validation_runtime
    )
    val_dataset, val_loader, val_sampler = make_loader(
        validation_loader_cfg,
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
        bind_sentence_memory_validation_dataset(
            cfg,
            sentence_memory_provider,
            val_dataset,
            isolated_validation_runtime,
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
    validation_text_partition = None
    if requires_development_only_selection(cfg):
        if isolated_validation_runtime is not None:
            val_loader, val_sampler, validation_text_partition = (
                build_isolated_development_validation_loader(
                    cfg,
                    val_dataset,
                    val_loader,
                    sentence_memory_provider,
                    dist_info,
                    isolated_validation_runtime,
                )
            )
        else:
            val_loader, val_sampler, validation_text_partition = (
                build_development_validation_loader(
                    cfg,
                    val_dataset,
                    val_loader,
                    sentence_memory_provider,
                    dist_info,
                )
            )
        if isinstance(validation_text_partition, DevelopmentValidationRuntime):
            development_text_count = (
                validation_text_partition.development_text_count
            )
            confirmation_text_count = (
                validation_text_partition.confirmation_text_count
            )
        else:
            development_text_count = len(
                validation_text_partition.development_texts
            )
            confirmation_text_count = len(
                validation_text_partition.confirmation_texts
            )
        rank_zero_print(
            dist_info,
            "Validation text partition: "
            f"digest={validation_text_partition.partition_digest} "
            f"development_texts={development_text_count} "
            f"confirmation_texts={confirmation_text_count} "
            "training_evaluates=development_only",
        )
        if dist_info["is_main"]:
            partition_path = out_dir / "validation_text_partition.json"
            partial_path = partition_path.with_suffix(".json.tmp")
            partial_path.write_text(
                json.dumps(
                    validation_text_partition.artifact_payload,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(partial_path, partition_path)
            (out_dir / "config.resolved.json").write_text(
                json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
            )
    retrieval_bank = (
        validate_train_only_retrieval_bank(cfg, provider)
        if provider is not None
        else {"enabled": False, "reason": "word_prior_disabled"}
    )
    rank_zero_print(dist_info, f"Retrieval bank: {json.dumps(retrieval_bank, sort_keys=True)}")
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(device)
    fk = build_fk(cfg, device)
    train_cfg = cfg.get("train", {})
    paired_objective_enabled = paired_sentence_memory_corruption_config(cfg)[
        "enabled"
    ]
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_epochs = int(
        train_cfg.get(
            "early_stopping_min_epochs",
            train_cfg.get("early_stopping_min_epoch", 2),
        )
    )
    if early_stopping_patience < 0:
        raise ValueError("train.early_stopping_patience must be non-negative")
    if early_stopping_min_epochs < 1:
        raise ValueError("train.early_stopping_min_epochs must be at least one")
    early_stopping_state = (
        initial_early_stopping_state()
        if early_stopping_patience > 0
        else None
    )
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
        and args.stage_c_warm_start is None
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
    resume_pending_validation_epoch = None
    resume_pending_validation_row = None
    best_score = float("inf")
    best_infeasible_score = float("inf")
    best_infeasible_key = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_phase_b_resume_checkpoint(
            cfg, checkpoint, source=str(args.resume)
        )
        validate_stage_c_resume_checkpoint(
            cfg, checkpoint, source=str(args.resume)
        )
        validate_checkpoint_contract(checkpoint, cfg, source=str(args.resume))
        if is_sentence_memory_model(cfg):
            validate_sentence_memory_resume_identity(
                checkpoint, cfg, source=str(args.resume)
            )
            validate_sentence_memory_objective_identity(
                checkpoint, cfg, source=str(args.resume)
            )
            validate_sentence_memory_architecture_identity(
                checkpoint, cfg, source=str(args.resume)
            )
            validate_sentence_memory_evaluation_control_identity(
                checkpoint, cfg, source=str(args.resume)
            )
            validate_sentence_memory_selection_aggregation_identity(
                checkpoint, cfg, source=str(args.resume)
            )
            validate_sentence_memory_validation_corruption_map_identity(
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
            if isinstance(parity, dict) and not stage_c_enabled(cfg):
                cfg.setdefault("sentence_memory_safety", {})[
                    "v2_to_v3_text_only_parity"
                ] = parity
            elif stage_c_enabled(cfg):
                cfg.setdefault("sentence_memory_safety", {}).pop(
                    "v2_to_v3_text_only_parity", None
                )
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
        if paired_objective_enabled or stage_c_enabled(cfg):
            best_infeasible_key = restore_best_infeasible_selection_key(
                checkpoint,
                require=True,
                source=str(args.resume),
            )
        if early_stopping_state is not None:
            early_stopping_state = restore_early_stopping_state(
                checkpoint,
                require=True,
                source=str(args.resume),
            )
            if early_stopping_state["stopped"]:
                raise RuntimeError(
                    f"{args.resume} already reached its configured early-stopping "
                    f"condition at epoch {early_stopping_state['stop_epoch']}"
                )
        (
            resume_pending_validation_epoch,
            resume_pending_validation_row,
        ) = pending_validation_resume_state(
            checkpoint,
            paired_objective_enabled=(
                paired_objective_enabled or stage_c_enabled(cfg)
            ),
        )
        if resume_pending_validation_epoch is not None:
            start_epoch = resume_pending_validation_epoch
        if is_sentence_memory_model(cfg):
            # Restore only after model/teacher/DDP/W&B initialization has
            # finished, so those setup steps cannot consume continuation RNG.
            resume_rng_checkpoint = checkpoint
    elif args.stage_c_warm_start:
        checkpoint = torch.load(args.stage_c_warm_start, map_location="cpu")
        validate_checkpoint_contract(
            checkpoint, cfg, source=str(args.stage_c_warm_start)
        )
        stage_c_provenance = validate_stage_c_warm_start_checkpoint(
            cfg,
            checkpoint,
            args.stage_c_warm_start,
            provider=sentence_memory_provider,
            source=str(args.stage_c_warm_start),
        )
        validate_sentence_memory_checkpoint_identity(
            checkpoint,
            sentence_memory_provider,
            source=str(args.stage_c_warm_start),
            # Stage C has an explicit, stricter cross-source calibration
            # transition above. Passing the active cfg to the general validator
            # would falsely claim the old checkpoint used the fresh identity.
            cfg=None,
            text_encoder_identity=(
                text_encoder.checkpoint_identity()
                if hasattr(text_encoder, "checkpoint_identity")
                else None
            ),
        )
        # The source parity proof describes only the frozen Stage-B
        # initialization. Generator updates make it stale, so Stage-C
        # checkpoints expose it only inside their explicit provenance record.
        cfg.setdefault("sentence_memory_safety", {}).pop(
            "v2_to_v3_text_only_parity", None
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        rank_zero_print(
            dist_info,
            f"Warm-started Stage-C model weights from {args.stage_c_warm_start} "
            f"(source_epoch={checkpoint.get('epoch')}, "
            f"source_global_step={checkpoint.get('global_step')}); optimizer, "
            "epoch/global-step, and selection state reset; "
            f"arm={configured_stage_c(cfg)['arm']} "
            f"provenance={stage_c_provenance['digest']} "
            "development_only=true non_authorizing=true "
            "promotion_eligible=false.",
        )
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
    validate_stage_c_optimizer_parameter_mapping(model, optimizer, cfg)
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
        resume_validation_only = bool(
            resume_pending_validation_epoch is not None
            and int(resume_pending_validation_epoch) == int(epoch)
        )
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        if sentence_memory_provider is not None:
            sentence_memory_provider.set_epoch(epoch)
        if resume_validation_only:
            if resume_pending_validation_row is None:
                raise RuntimeError("Pending-validation resume has no saved epoch row")
            row = copy.deepcopy(resume_pending_validation_row)
            row["validation_pending"] = 1.0
            row["resumed_pending_validation"] = 1.0
            rank_zero_print(
                dist_info,
                f"Resuming pending validation for completed epoch {epoch}; "
                "the training epoch will not be replayed.",
            )
        else:
            optimizer_lrs = configure_optimizer_epoch(optimizer, cfg, epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if sentence_memory_provider is not None:
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
                **{
                    f"lr_{name}": value
                    for name, value in optimizer_lrs.items()
                },
            }
            row.update(distributed_mean_scalars(train_metrics, device, dist_info))
            if dist_info["is_main"] and wandb_run is not None:
                wandb_run.log(wandb_train_epoch_payload(row))

        validation_due = resume_validation_only or epoch % val_every == 0
        epoch_checkpoint_rng = None
        if validation_due:
            # Persist the completed training epoch before entering the much
            # heavier dense-JVP validation path. Successful validation
            # overwrites this recovery snapshot with complete metrics below.
            if not resume_validation_only:
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
                            best_score,
                            best_infeasible_score,
                            early_stopping_state=early_stopping_state,
                            best_infeasible_key=best_infeasible_key,
                        ),
                        rng_state=pending_rng,
                    )
                barrier(dist_info)
                if dist_info["is_main"] and wandb_run is not None:
                    wandb_run.log(
                        wandb_validation_pending_payload(epoch, global_step)
                    )
            if sentence_memory_provider is not None:
                bind_sentence_memory_validation_dataset(
                    cfg,
                    sentence_memory_provider,
                    val_dataset,
                    isolated_validation_runtime,
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
            val_metrics = distributed_validation_metrics(
                val_metrics,
                evaluated_loader_sample_count(
                    val_loader, int(cfg.get("eval", {}).get("max_batches", 0))
                ),
                device,
                dist_info,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            if is_sentence_memory_model(cfg):
                row["sentence_memory_evaluation_control_digest"] = (
                    sentence_memory_evaluation_control_identity(cfg)["digest"]
                )
                row["sentence_memory_selection_aggregation_digest"] = (
                    sentence_memory_selection_aggregation_identity(cfg)["digest"]
                )
                row["sentence_memory_architecture_digest"] = (
                    sentence_memory_architecture_identity(cfg)["digest"]
                )
                corruption_map = (
                    cfg.get("validation_text_partition", {}).get(
                        "evaluation_corruption_map_identity"
                    )
                )
                if isinstance(corruption_map, dict):
                    row["validation_corruption_map_digest"] = corruption_map[
                        "digest"
                    ]
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
            if stage_c_enabled(cfg):
                row.update(
                    {
                        "stage_c_development_only": 1.0,
                        "stage_c_non_authorizing": 1.0,
                        "stage_c_promotion_eligible": 0.0,
                    }
                )
            row["selection_rejection_reasons"] = "; ".join(
                selection_details["rejection_reasons"]
            )
            for name, value in selection_details.get("dual_mode", {}).items():
                if isinstance(value, (int, float)):
                    row[f"selection_{name}"] = value
            if stage_c_enabled(cfg):
                stage_c_guard = selection_details.get("dual_mode", {}).get(
                    "stage_c_text_only_guard"
                )
                if not isinstance(stage_c_guard, dict):
                    raise RuntimeError(
                        "Stage-C selection did not expose its live memory-off guard"
                    )
                for name in (
                    "live_text_only_score",
                    "teacher_score",
                    "allowed_text_only_score",
                    "max_relative_degradation",
                    "violation",
                ):
                    value = float(stage_c_guard.get(name, math.nan))
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"Stage-C text-only guard field {name!r} is non-finite"
                        )
                    row[f"selection_stage_c_text_only_guard_{name}"] = value
                row["selection_stage_c_text_only_guard_passed"] = float(
                    stage_c_guard["violation"] <= 1e-12
                )
            early_stop_requested = False
            if early_stopping_state is not None:
                (
                    early_stopping_state,
                    early_stopping_improved,
                    early_stop_requested,
                ) = update_early_stopping_state(
                    early_stopping_state,
                    feasible=selection_feasible,
                    normalized_constraint_violation=constraint_violation,
                    selection_score_value=score,
                    epoch=epoch,
                    patience=early_stopping_patience,
                    minimum_epoch=early_stopping_min_epochs,
                )
                row["early_stopping_improved"] = float(
                    early_stopping_improved
                )
                row["early_stopping_bad_validation_count"] = float(
                    early_stopping_state["bad_validation_count"]
                )
                row["early_stopping_validation_count"] = float(
                    early_stopping_state["validation_count"]
                )
                row["early_stopping_requested"] = float(
                    early_stop_requested
                )
            if dist_info["is_main"] and wandb_run is not None:
                wandb_run.log(wandb_validation_payload(row))
            keep_validation_checkpoint = bool(
                cfg.get("selection", {}).get("keep_validation_checkpoints", False)
            )
            improved_feasible = bool(selection_feasible and score < best_score)
            current_infeasible_key = (
                (float(constraint_violation), float(score))
                if not selection_feasible
                else None
            )
            improved_infeasible = bool(
                current_infeasible_key is not None
                and (
                    best_infeasible_key is None
                    or current_infeasible_key < best_infeasible_key
                )
            )
            if selection_feasible and score < best_score:
                best_score = score
            elif improved_infeasible:
                best_infeasible_score = score
                best_infeasible_key = current_infeasible_key
            if (
                keep_validation_checkpoint
                or improved_feasible
                or improved_infeasible
                or epoch % save_every == 0
            ):
                epoch_checkpoint_rng = distributed_checkpoint_rng_state(dist_info)
            current_selection_state = checkpoint_selection_state(
                best_score,
                best_infeasible_score,
                early_stopping_state=early_stopping_state,
                best_infeasible_key=best_infeasible_key,
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
                    out_dir
                    / "checkpoints"
                    / selected_checkpoint_artifact_name(cfg, feasible=True),
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
                    out_dir
                    / "checkpoints"
                    / selected_checkpoint_artifact_name(cfg, feasible=False),
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
                        best_score,
                        best_infeasible_score,
                        early_stopping_state=early_stopping_state,
                        best_infeasible_key=best_infeasible_key,
                    ),
                    rng_state=epoch_checkpoint_rng,
                )
        if dist_info["is_main"]:
            append_jsonl(out_dir / "metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True))
        barrier(dist_info)
        if resume_validation_only:
            resume_pending_validation_epoch = None
            resume_pending_validation_row = None
        if validation_due and early_stopping_state is not None:
            if bool(early_stopping_state["stopped"]):
                break

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
            "best_infeasible_key": (
                list(best_infeasible_key)
                if best_infeasible_key is not None
                else None
            ),
            "early_stopping": copy.deepcopy(early_stopping_state),
        }
        if stage_c_enabled(cfg):
            selection_summary["stage_c"] = {
                "schema_name": STAGE_C_SCHEMA_NAME,
                "schema_version": STAGE_C_SCHEMA_VERSION,
                "arm": configured_stage_c(cfg)["arm"],
                "development_only": True,
                "non_authorizing": True,
                "promotion_eligible": False,
                "canonical_best_checkpoint_published": False,
                "best_feasible_artifact": (
                    "checkpoints/best_exploratory.pt"
                    if has_feasible_checkpoint
                    else None
                ),
                "best_infeasible_artifact": (
                    "checkpoints/best_exploratory_infeasible.pt"
                    if math.isfinite(best_infeasible_score)
                    else None
                ),
            }
        write_json_atomic(out_dir / "selection_summary.json", selection_summary)
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
