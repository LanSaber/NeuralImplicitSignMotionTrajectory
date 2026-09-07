"""Analyze the one-shot confirmation for the ordered Phase-A'' experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from NIAF.continuous_trajectory_field.phase_a_motion_contrast import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    MAX_DURATION_ABS_DIFFERENCE_SECONDS,
    MAX_HAND_PATH_RELATIVE_DEGRADATION,
    MIN_RELATIVE_PA_NDTW_IMPROVEMENT,
    MIN_RMOTION_LOWER_CI,
    MIN_RMOTION_POINT,
    cluster_bootstrap_indices,
    confidence_interval,
    holm_adjust,
    paired_lower_is_better,
    rmotion_summary,
    sign_flip_lower_tail_pvalue,
    stable_subseed,
)
from NIAF.continuous_trajectory_field.sentence_memory import sha256_file
from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    ALIGNMENTS,
    DTW_FILE,
    PARTS,
    ConfirmationInputError,
    _cluster_index_groups,
    _cluster_mean,
    _prediction_parity,
    _row_motion_mse,
    _v2_checkpoint_evidence,
    _validate_all_null_export,
    _validate_motion_shuffle_pair,
    load_confirmation_partition,
    load_integrity_export,
    load_mode_export,
)
from NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage import (
    DEVELOPMENT_FEASIBLE,
    EXPERIMENTS,
    EXPECTED_CONFIRMATION_TEXTS,
    EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
    EXPECTED_EVALUATION_CORRUPTION,
    EXPECTED_NEIGHBOR_SHA256,
    EXPECTED_PARTITION_DIGEST,
    EXPECTED_VALIDATION_MANIFEST_SHA256,
    HOLDOUT_SPEND_SCHEMA_NAME,
    OrderedDecisionError,
    SCHEMA_VERSION as DECISION_SCHEMA_VERSION,
    STAGES,
    validate_factorized_export_query_binding,
    verify_authorization,
)


SCHEMA_NAME = "signtrajfield_factorized_memory_confirmation"
SCHEMA_VERSION = 1
PRIMARY_MODE_MEMORY = {
    "text_only": "off",
    "sentence_memory": "on",
    "motion_shuffled_sentence_memory": "motion_shuffled",
    "shuffled_sentence_memory": "shuffled",
}
ANALYTIC_MODE = "analytic_prior_sentence_memory"
ATTENTION_PARTS = ("body", "lhand", "rhand", "face")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfirmationInputError(f"Required file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationInputError(f"Expected JSON object: {path}")
    return value


def _validate_spend_marker(
    marker_path: Path,
    *,
    authorization_path: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _json(marker_path)
    if (
        marker.get("schema_name") != HOLDOUT_SPEND_SCHEMA_NAME
        or int(marker.get("schema_version", -1)) != DECISION_SCHEMA_VERSION
        or marker.get("confirmation_holdout_spent") is not True
        or marker.get("authorization_path") != str(authorization_path.resolve())
        or marker.get("authorization_identity")
        != authorization.get("authorization_identity")
        or marker.get("checkpoint_sha256")
        != dict(authorization.get("checkpoint", {}) or {}).get("sha256")
        or marker.get("partition_digest") != EXPECTED_PARTITION_DIGEST
    ):
        raise ConfirmationInputError("Confirmation spend marker is invalid")
    payload = {key: value for key, value in marker.items() if key != "spend_identity"}
    if _digest_json(payload) != marker.get("spend_identity"):
        raise ConfirmationInputError("Confirmation spend identity is invalid")
    return marker


def _resolve_sample(row: Mapping[str, Any], directory: Path) -> Path:
    path = Path(str(row.get("sample", "")))
    candidates = (path,) if path.is_absolute() else (Path.cwd() / path, directory / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfirmationInputError(f"Cannot resolve export sample: {path}")


def _validate_analytic_pair(
    correct: Mapping[str, Any], analytic: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove analytic mode changes only attention logits, not retrieval inputs."""

    if len(correct["rows"]) != len(analytic["rows"]):
        raise ConfirmationInputError("Analytic-prior export row count changed")
    metadata_fields = (
        "sentence_memory_ids",
        "sentence_memory_scores",
        "sentence_memory_durations",
        "sentence_memory_duration_log_gap",
        "sentence_memory_candidate_mask",
        "sentence_memory_group_ids",
        "sentence_memory_exact_text",
        "sentence_memory_query_seen_text",
    )
    for correct_row, analytic_row in zip(correct["rows"], analytic["rows"]):
        with np.load(
            _resolve_sample(correct_row, Path(correct["dir"])), allow_pickle=False
        ) as correct_sample, np.load(
            _resolve_sample(analytic_row, Path(analytic["dir"])), allow_pickle=False
        ) as analytic_sample:
            for field in metadata_fields:
                if field not in correct_sample.files or field not in analytic_sample.files:
                    raise ConfirmationInputError(
                        f"Analytic/correct export lacks retrieval field {field}"
                    )
                if not np.array_equal(correct_sample[field], analytic_sample[field]):
                    raise ConfirmationInputError(
                        f"Analytic prior changed retrieval metadata {field}"
                    )
            if str(np.asarray(analytic_sample["sentence_memory_attention_mode"]).item()) != (
                "analytic_prior"
            ):
                raise ConfirmationInputError("Analytic export did not bypass learned QK")
            if str(np.asarray(correct_sample["sentence_memory_attention_mode"]).item()) != (
                "learned"
            ):
                raise ConfirmationInputError("Correct export did not use learned attention")
    return {
        "same_candidate_ids_and_metadata": True,
        "correct_attention_mode": "learned",
        "analytic_attention_mode": "analytic_prior",
        "promotion_comparator": False,
        "passed": True,
    }


def _authorization_evidence(
    authorization_path: Path,
    *,
    checkpoint_path: Path,
    config_path: Path,
) -> tuple[str, dict[str, Any]]:
    raw = _json(authorization_path)
    stage = str(raw.get("stage", ""))
    if stage not in STAGES:
        raise ConfirmationInputError("Authorization has an unknown stage")
    try:
        authorization = verify_authorization(
            authorization_path, purpose="confirmation", stage=stage
        )
    except RuntimeError as error:
        raise ConfirmationInputError(str(error)) from error
    checkpoint = dict(authorization.get("checkpoint", {}) or {})
    if checkpoint_path.resolve() != Path(str(checkpoint.get("path", ""))).resolve():
        raise ConfirmationInputError("Requested checkpoint differs from authorization")
    if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise ConfirmationInputError("Authorized checkpoint hash changed")
    expected_config = dict(checkpoint.get("config", {}) or {})
    if config_path.resolve() != Path(str(expected_config.get("path", ""))).resolve():
        raise ConfirmationInputError("Requested config differs from authorization")
    if sha256_file(config_path) != expected_config.get("sha256"):
        raise ConfirmationInputError("Authorized config hash changed")
    launch_evidence = dict(authorization.get("run_launch_identity", {}) or {})
    launch_path = Path(str(launch_evidence.get("path", "")))
    if (
        not launch_path.is_file()
        or sha256_file(launch_path) != launch_evidence.get("sha256")
    ):
        raise ConfirmationInputError("Authorized run-launch evidence changed")
    launch = _json(launch_path)
    launch_payload = {
        key: value for key, value in launch.items() if key != "launch_identity"
    }
    if (
        _digest_json(launch_payload) != launch.get("launch_identity")
        or launch.get("launch_identity") != launch_evidence.get("launch_identity")
        or dict(launch.get("source", {}) or {})
        != dict(launch_evidence.get("source", {}) or {})
    ):
        raise ConfirmationInputError("Authorized run-launch identity is invalid")
    source = dict(launch.get("source", {}) or {})
    if (
        source.get("worktree_clean_checked") is not True
        or source.get("remote_ref_exact_match_checked") is not True
        or str(source.get("remote_head", "")).lower()
        != str(source.get("git_head", "")).lower()
    ):
        raise ConfirmationInputError("Authorized source lacks exact remote proof")
    decision = _json(authorization_path.parent / "decision.json")
    if (
        decision.get("status") != DEVELOPMENT_FEASIBLE
        or decision.get("integrity_valid") is not True
        or decision.get("stage") != stage
    ):
        raise ConfirmationInputError("Authorization is not from a feasible stage")
    if stage == "stage2":
        predecessor = dict(decision.get("predecessor_authorization", {}) or {})
        predecessor_path = Path(str(predecessor.get("path", "")))
        if not predecessor_path.is_file() or sha256_file(predecessor_path) != predecessor.get(
            "sha256"
        ):
            raise ConfirmationInputError("Stage-2 predecessor authorization changed")
        try:
            prior = verify_authorization(
                predecessor_path, purpose="stage2", stage="stage1"
            )
        except RuntimeError as error:
            raise ConfirmationInputError(str(error)) from error
        if prior.get("authorization_identity") != predecessor.get(
            "authorization_identity"
        ):
            raise ConfirmationInputError("Stage-2 predecessor identity mismatch")
    return stage, authorization


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    midpoint = 0.5 * (first + second)

    def divergence(value: np.ndarray) -> float:
        positive = value > 0.0
        return float(
            np.sum(value[positive] * np.log(value[positive] / midpoint[positive]))
        )

    return 0.5 * (divergence(first) + divergence(second))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _attention_row_metrics(export: Mapping[str, Any]) -> dict[str, np.ndarray]:
    artifact = dict(
        export.get("summary", {}).get("factorized_attention_artifacts", {}) or {}
    )
    if (
        artifact.get("schema") != "final_layer_part_attention_mass_v1"
        or artifact.get("layer") != "final"
        or artifact.get("head_reduction") != "mean"
        or artifact.get("part_order") != list(ATTENTION_PARTS)
    ):
        raise ConfirmationInputError(
            "Export lacks the declared final-layer, head-mean attention contract"
        )
    rows: list[dict[str, float]] = []
    for row in export["rows"]:
        sample_path = _resolve_sample(row, Path(export["dir"]))
        with np.load(sample_path, allow_pickle=False) as sample:
            required = (
                "sentence_memory_part_null_mass",
                "sentence_memory_part_candidate_mass",
                "sentence_memory_part_token_mass",
                "sentence_memory_token_tau",
                "sentence_memory_token_mask",
                "sentence_memory_part_validity",
                "trajectory_sentence_memory_gates",
            )
            missing = [name for name in required if name not in sample.files]
            if missing:
                raise ConfirmationInputError(
                    f"Factorized attention export lacks fields: {missing}"
                )
            null = np.asarray(
                sample["sentence_memory_part_null_mass"], dtype=np.float64
            )
            candidate = np.asarray(
                sample["sentence_memory_part_candidate_mass"], dtype=np.float64
            )
            token = np.asarray(
                sample["sentence_memory_part_token_mass"], dtype=np.float64
            )
            token_tau = np.asarray(sample["sentence_memory_token_tau"], dtype=np.float64)
            token_mask = np.asarray(sample["sentence_memory_token_mask"], dtype=bool)
            validity = np.asarray(sample["sentence_memory_part_validity"], dtype=np.float64)
            gates = np.asarray(
                sample["trajectory_sentence_memory_gates"], dtype=np.float64
            )
        if (
            null.ndim != 2
            or candidate.ndim != 3
            or token.ndim != 4
            or null.shape != candidate.shape[:2]
            or candidate.shape != token.shape[:3]
            or token_tau.shape != token.shape[2:]
            or token_mask.shape != token.shape[2:]
            or validity.shape != (*token.shape[2:], len(ATTENTION_PARTS))
            or gates.shape != null.shape
            or null.shape[1] != len(ATTENTION_PARTS)
        ):
            raise ConfirmationInputError("Malformed factorized attention artifact shapes")
        if not all(
            np.isfinite(value).all()
            for value in (null, candidate, token, token_tau, validity, gates)
        ):
            raise ConfirmationInputError("Non-finite factorized attention artifact")
        if np.max(np.abs(candidate - token.sum(axis=-1)), initial=0.0) > 2e-5:
            raise ConfirmationInputError("Candidate/token attention mass is not conserved")
        if np.max(np.abs(candidate.sum(axis=-1) + null - 1.0), initial=0.0) > 2e-5:
            raise ConfirmationInputError("Real/null attention mass is not conserved")
        structural_valid = np.transpose(
            token_mask[:, :, None] & (validity > 0.0), (2, 0, 1)
        )[None, ...]
        structural_valid = np.broadcast_to(structural_valid, token.shape)
        if np.max(np.abs(token[~structural_valid]), initial=0.0) > 1e-7:
            raise ConfirmationInputError("Attention assigned mass to an invalid token")

        real_mass = candidate.sum(axis=-1, keepdims=True)
        normalized_candidate = np.divide(
            candidate,
            real_mass,
            out=np.zeros_like(candidate),
            where=real_mass > 1e-12,
        )
        normalized_token = np.divide(
            token,
            real_mass[..., None],
            out=np.zeros_like(token),
            where=real_mass[..., None] > 1e-12,
        )
        slot_tau = np.linspace(-1.0, 1.0, null.shape[0], dtype=np.float64)
        row_metrics: dict[str, float] = {}
        for part_index, part in enumerate(ATTENTION_PARTS):
            probabilities = normalized_candidate[:, part_index]
            entropy = -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=-1
            )
            expected_source = np.sum(
                normalized_token[:, part_index] * token_tau[None, :, :],
                axis=(-1, -2),
            )
            adjacent_jsd = [
                _jensen_shannon(probabilities[index], probabilities[index + 1])
                for index in range(len(probabilities) - 1)
            ]
            row_metrics[f"{part}/effective_k"] = float(np.exp(entropy).mean())
            row_metrics[f"{part}/maximum_candidate_share"] = float(
                probabilities.max(axis=-1).mean()
            )
            row_metrics[f"{part}/adjacent_slot_jsd"] = float(
                np.mean(adjacent_jsd) if adjacent_jsd else 0.0
            )
            row_metrics[f"{part}/source_time_correlation"] = _correlation(
                slot_tau, expected_source
            )
            row_metrics[f"{part}/source_time_mae"] = float(
                np.mean(np.abs(slot_tau - expected_source))
            )
            row_metrics[f"{part}/gate_mean"] = float(gates[:, part_index].mean())
            row_metrics[f"{part}/null_mass_mean"] = float(null[:, part_index].mean())
        part_jsd = []
        for slot in range(null.shape[0]):
            for first in range(len(ATTENTION_PARTS)):
                for second in range(first + 1, len(ATTENTION_PARTS)):
                    part_jsd.append(
                        _jensen_shannon(
                            normalized_candidate[slot, first],
                            normalized_candidate[slot, second],
                        )
                    )
        row_metrics["all/part_jsd"] = float(np.mean(part_jsd))
        rows.append(row_metrics)
    names = sorted(rows[0])
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in names
    }


def _aggregate_attention_diagnostics(
    modes: Mapping[str, Mapping[str, Any]],
    *,
    groups: list[np.ndarray],
    bootstrap: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    aggregate: dict[str, Any] = {}
    values_by_mode: dict[str, dict[str, np.ndarray]] = {}
    for mode_name, export in modes.items():
        row_metrics = _attention_row_metrics(export)
        cluster_metrics = {
            name: _cluster_mean(values, groups)
            for name, values in row_metrics.items()
        }
        values_by_mode[mode_name] = cluster_metrics
        aggregate[mode_name] = {
            name: {
                "mean": float(values.mean()),
                "ci95": confidence_interval(values[bootstrap].mean(axis=1)),
                "cluster_count": len(values),
            }
            for name, values in cluster_metrics.items()
        }
    return aggregate, values_by_mode


def _analytic_learned_differences(
    *,
    modes: Mapping[str, Mapping[str, Any]],
    groups: list[np.ndarray],
    bootstrap: np.ndarray,
    attention_values: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    analytic_mse = _cluster_mean(
        _row_motion_mse(modes["sentence_memory"], modes[ANALYTIC_MODE]), groups
    )
    off_mse = _cluster_mean(
        _row_motion_mse(modes["sentence_memory"], modes["text_only"]), groups
    )
    output_ratio = rmotion_summary(analytic_mse, off_mse, bootstrap)
    correct_attention = attention_values["sentence_memory"]
    analytic_attention = attention_values[ANALYTIC_MODE]
    attention_differences = {}
    for name in sorted(correct_attention):
        differences = analytic_attention[name] - correct_attention[name]
        attention_differences[name] = {
            "analytic_minus_learned_mean": float(differences.mean()),
            "ci95": confidence_interval(differences[bootstrap].mean(axis=1)),
        }
    direct_attention_rows = _paired_attention_distance_rows(
        modes["sentence_memory"], modes[ANALYTIC_MODE]
    )
    direct_attention = {}
    for name, row_values in direct_attention_rows.items():
        cluster_values = _cluster_mean(row_values, groups)
        direct_attention[name] = {
            "mean": float(cluster_values.mean()),
            "ci95": confidence_interval(
                cluster_values[bootstrap].mean(axis=1)
            ),
            "cluster_count": len(cluster_values),
        }
    return {
        "output_Ranalytic_vs_memory_off": output_ratio,
        "direct_attention_distances": direct_attention,
        "attention_metric_differences": attention_differences,
        "explanatory_only": True,
        "promotion_comparator": False,
    }


def _paired_attention_distance_rows(
    learned: Mapping[str, Any], analytic: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Return final-layer, head-mean learned/analytic attention distances."""

    if len(learned["rows"]) != len(analytic["rows"]):
        raise ConfirmationInputError("Attention exports have different row counts")
    rows: list[dict[str, float]] = []
    for learned_row, analytic_row in zip(learned["rows"], analytic["rows"]):
        with np.load(
            _resolve_sample(learned_row, Path(learned["dir"])),
            allow_pickle=False,
        ) as learned_sample, np.load(
            _resolve_sample(analytic_row, Path(analytic["dir"])),
            allow_pickle=False,
        ) as analytic_sample:
            candidate_learned = np.asarray(
                learned_sample["sentence_memory_part_candidate_mass"],
                dtype=np.float64,
            )
            candidate_analytic = np.asarray(
                analytic_sample["sentence_memory_part_candidate_mass"],
                dtype=np.float64,
            )
            token_learned = np.asarray(
                learned_sample["sentence_memory_part_token_mass"],
                dtype=np.float64,
            )
            token_analytic = np.asarray(
                analytic_sample["sentence_memory_part_token_mass"],
                dtype=np.float64,
            )
            null_learned = np.asarray(
                learned_sample["sentence_memory_part_null_mass"],
                dtype=np.float64,
            )
            null_analytic = np.asarray(
                analytic_sample["sentence_memory_part_null_mass"],
                dtype=np.float64,
            )
            gate_learned = np.asarray(
                learned_sample["trajectory_sentence_memory_gates"],
                dtype=np.float64,
            )
            gate_analytic = np.asarray(
                analytic_sample["trajectory_sentence_memory_gates"],
                dtype=np.float64,
            )
        if (
            candidate_learned.shape != candidate_analytic.shape
            or token_learned.shape != token_analytic.shape
            or null_learned.shape != null_analytic.shape
            or gate_learned.shape != gate_analytic.shape
            or candidate_learned.shape != token_learned.shape[:-1]
            or null_learned.shape != candidate_learned.shape[:2]
            or gate_learned.shape != null_learned.shape
        ):
            raise ConfirmationInputError(
                "Learned/analytic attention artifact shapes differ"
            )
        real_learned = candidate_learned.sum(axis=-1, keepdims=True)
        real_analytic = candidate_analytic.sum(axis=-1, keepdims=True)
        conditional_candidate_learned = np.divide(
            candidate_learned,
            real_learned,
            out=np.zeros_like(candidate_learned),
            where=real_learned > 1e-12,
        )
        conditional_candidate_analytic = np.divide(
            candidate_analytic,
            real_analytic,
            out=np.zeros_like(candidate_analytic),
            where=real_analytic > 1e-12,
        )
        conditional_token_learned = np.divide(
            token_learned,
            real_learned[..., None],
            out=np.zeros_like(token_learned),
            where=real_learned[..., None] > 1e-12,
        )
        conditional_token_analytic = np.divide(
            token_analytic,
            real_analytic[..., None],
            out=np.zeros_like(token_analytic),
            where=real_analytic[..., None] > 1e-12,
        )
        rows.append(
            {
                "conditional_candidate_total_variation": float(
                    0.5
                    * np.abs(
                        conditional_candidate_analytic
                        - conditional_candidate_learned
                    ).sum(axis=-1).mean()
                ),
                "conditional_token_total_variation": float(
                    0.5
                    * np.abs(
                        conditional_token_analytic - conditional_token_learned
                    ).sum(axis=(-1, -2)).mean()
                ),
                "null_mass_mean_absolute_difference": float(
                    np.abs(null_analytic - null_learned).mean()
                ),
                "gate_mean_absolute_difference": float(
                    np.abs(gate_analytic - gate_learned).mean()
                ),
            }
        )
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in sorted(rows[0])
    }


def _load_development_diagnostic(
    directory: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config_path: Path,
    authorization_path: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    directory = directory.resolve()
    ready_path = directory / "READY"
    provenance_path = directory / "provenance.json"
    summary_path = directory / "summary.json"
    ready = _json(ready_path)
    provenance = _json(provenance_path)
    summary = _json(summary_path)
    if ready.get("schema_name") != "signtrajfield_temporal_slot_diagnostics":
        raise ConfirmationInputError("Development diagnostic has the wrong schema")
    if ready.get("provenance_sha256") != sha256_file(provenance_path):
        raise ConfirmationInputError("Development diagnostic READY is detached")
    artifact_manifest_path = directory / "artifact_manifest.json"
    artifact_manifest = _json(artifact_manifest_path)
    if (
        provenance.get("artifact_manifest_sha256")
        != sha256_file(artifact_manifest_path)
        or artifact_manifest.get("schema")
        != "signtrajfield_temporal_slot_artifacts"
        or int(artifact_manifest.get("version", -1)) != 1
    ):
        raise ConfirmationInputError(
            "Development diagnostic artifact manifest is invalid"
        )
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or "summary.json" not in artifacts:
        raise ConfirmationInputError(
            "Development diagnostic artifact manifest is incomplete"
        )
    for relative, metadata in artifacts.items():
        relative_path = Path(str(relative))
        artifact_path = (directory / relative_path).resolve()
        evidence = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
        if (
            relative_path.is_absolute()
            or not artifact_path.is_relative_to(directory)
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != int(evidence.get("bytes", -1))
            or sha256_file(artifact_path) != evidence.get("sha256")
        ):
            raise ConfirmationInputError(
                f"Development diagnostic artifact changed: {relative}"
            )
    if (
        Path(str(provenance.get("checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or provenance.get("checkpoint_sha256") != checkpoint_sha256
        or Path(str(provenance.get("config", ""))).resolve()
        != config_path.resolve()
        or provenance.get("config_sha256") != sha256_file(config_path)
    ):
        raise ConfirmationInputError(
            "Development diagnostic used another checkpoint or config"
        )
    launch_source = dict(
        dict(authorization.get("run_launch_identity", {}) or {}).get("source", {})
        or {}
    )
    source_head = str(launch_source.get("git_head", "")).lower()
    locked = dict(provenance.get("locked_run_evidence", {}) or {})
    provenance_git = dict(provenance.get("git", {}) or {})
    source_checkout = dict(locked.get("source_checkout", {}) or {})
    if (
        not source_head
        or str(launch_source.get("remote_head", "")).lower() != source_head
        or launch_source.get("standalone_shared_clone_checked") is not True
        or launch_source.get("worktree_clean_checked") is not True
        or launch_source.get("remote_ref_exact_match_checked") is not True
        or str(provenance.get("git_head", "")).lower() != source_head
        or str(provenance_git.get("commit", "")).lower() != source_head
        or provenance_git.get("tracked_worktree_clean") is not True
        or str(locked.get("source_git_head", "")).lower() != source_head
        or dict(locked.get("run_launch_source", {}) or {}) != launch_source
        or str(source_checkout.get("git_head", "")).lower() != source_head
        or source_checkout.get("standalone_shared_clone_checked") is not True
        or source_checkout.get("worktree_clean_checked") is not True
        or any(
            not str(launch_source.get(name, ""))
            for name in (
                "repository_root",
                "git_directory",
                "durable_experiments_root",
                "frozen_text_model_root",
            )
        )
        or any(
            str(source_checkout.get(name, ""))
            != str(launch_source.get(name, ""))
            for name in (
                "repository_root",
                "git_directory",
                "durable_experiments_root",
                "frozen_text_model_root",
            )
        )
        or Path(str(locked.get("authorization_path", ""))).resolve()
        != authorization_path.resolve()
        or locked.get("authorization_sha256") != sha256_file(authorization_path)
        or locked.get("authorization_identity")
        != authorization.get("authorization_identity")
        or locked.get("decision_identity") != authorization.get("decision_identity")
    ):
        raise ConfirmationInputError(
            "Development diagnostic source is detached from its authorization"
        )
    from NIAF.continuous_trajectory_field.scripts.diagnose_temporal_slots import (
        _diagnostic_source_hashes,
    )

    if provenance.get("source_sha256") != _diagnostic_source_hashes():
        raise ConfirmationInputError(
            "Development diagnostic executable/dependency source hashes changed"
        )
    retrieval_lookup = dict(
        provenance.get("retrieval_lookup_evidence", {}) or {}
    )
    canonical_lookup = dict(retrieval_lookup.get("canonical_table", {}) or {})
    development_lookup = dict(
        retrieval_lookup.get("development_subset", {}) or {}
    )
    authorization_partition = dict(authorization.get("partition", {}) or {})
    authorization_checkpoint = dict(authorization.get("checkpoint", {}) or {})
    authorized_tables = dict(
        authorization_checkpoint.get("neighbor_table_sha256", {}) or {}
    )
    sha_fields = (
        canonical_lookup.get("parent_query_order_sha256"),
        development_lookup.get("query_order_sha256"),
        development_lookup.get("parent_query_rows_sha256"),
    )
    if (
        retrieval_lookup.get("schema_name")
        != "signtrajfield_factorized_development_neighbor_lookup"
        or int(retrieval_lookup.get("schema_version", -1)) != 1
        or retrieval_lookup.get("lookup_mode")
        != "exact_name_indexed_parent_subset_v1"
        or retrieval_lookup.get("online_retrieval_fallback") != "forbidden"
        or canonical_lookup.get("split") != "val"
        or canonical_lookup.get("sha256") != EXPECTED_NEIGHBOR_SHA256["val"]
        or canonical_lookup.get("sha256") != authorized_tables.get("val")
        or canonical_lookup.get("parent_query_manifest_sha256")
        != EXPECTED_VALIDATION_MANIFEST_SHA256
        or int(canonical_lookup.get("parent_query_count", -1)) != 1_077
        or development_lookup.get("query_manifest_sha256")
        != EXPECTED_DEVELOPMENT_MANIFEST_SHA256
        or development_lookup.get("query_manifest_sha256")
        != authorization_partition.get("development_manifest_sha256")
        or int(development_lookup.get("query_count", -1)) != 347
        or development_lookup.get("parent_query_rows_unique") is not True
        or int(development_lookup.get("parent_query_row_min", -1)) < 0
        or int(development_lookup.get("parent_query_row_max", 1_077)) >= 1_077
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in sha_fields
        )
    ):
        raise ConfirmationInputError(
            "Development diagnostic did not use the authorized exact-name "
            "canonical validation-neighbor subset"
        )
    counts = dict(provenance.get("query_counts", {}) or {})
    if int(counts.get("passive_rows", -1)) != 347 or int(
        counts.get("causal_rows", -1)
    ) != 128:
        raise ConfirmationInputError(
            "Locked diagnostic must cover 347 development rows and 128 causal texts"
        )
    if summary.get("stage") != "full" or summary.get("validation_only") is not True:
        raise ConfirmationInputError("Development diagnostic is not a full val-only audit")
    return {
        "directory": str(directory),
        "READY_sha256": sha256_file(ready_path),
        "provenance_sha256": sha256_file(provenance_path),
        "summary_sha256": sha256_file(summary_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "query_counts": counts,
        "passive_attention_and_slots": summary.get("passive"),
        "causal_locality": summary.get("causal"),
        "integrity_checks": summary.get("checks"),
        "retrieval_lookup_evidence": retrieval_lookup,
        "diagnostic_only": True,
        "changed_checkpoint_selection": False,
    }


def analyze_confirmation(
    *,
    authorization_path: Path,
    holdout_spend_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    v2_checkpoint_path: Path,
    v2_config_path: Path,
    partition_dir: Path,
    mode_dirs: Mapping[str, Path],
    analytic_prior_dir: Path,
    development_diagnostic_dir: Path,
    all_null_dir: Path,
    v2_text_only_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if int(bootstrap_samples) != BOOTSTRAP_SAMPLES or int(seed) != BOOTSTRAP_SEED:
        raise ConfirmationInputError("Confirmation requires 10,000 resamples, seed 1234")
    if set(mode_dirs) != set(PRIMARY_MODE_MEMORY):
        raise ConfirmationInputError("Exactly four promotion modes are required")
    stage, authorization = _authorization_evidence(
        authorization_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    spend = _validate_spend_marker(
        holdout_spend_path,
        authorization_path=authorization_path,
        authorization=authorization,
    )
    partition = load_confirmation_partition(partition_dir)
    if partition["payload"].get("partition_digest") != EXPECTED_PARTITION_DIGEST:
        raise ConfirmationInputError("Confirmation partition digest changed")
    checkpoint_evidence = dict(authorization["checkpoint"])
    checkpoint_epoch = int(checkpoint_evidence["epoch"])
    development_diagnostic = _load_development_diagnostic(
        development_diagnostic_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(checkpoint_evidence["sha256"]),
        config_path=config_path,
        authorization_path=authorization_path,
        authorization=authorization,
    )
    modes = {
        name: load_mode_export(
            name,
            Path(directory),
            partition=partition,
            checkpoint_path=checkpoint_path.resolve(),
            config_path=config_path.resolve(),
            checkpoint_epoch=checkpoint_epoch,
            motion_shuffle_seed=int(seed),
            expected_evaluation_corruption=EXPECTED_EVALUATION_CORRUPTION,
        )
        for name, directory in mode_dirs.items()
    }
    analytic = load_mode_export(
        ANALYTIC_MODE,
        analytic_prior_dir,
        partition=partition,
        checkpoint_path=checkpoint_path.resolve(),
        config_path=config_path.resolve(),
        checkpoint_epoch=checkpoint_epoch,
        motion_shuffle_seed=int(seed),
        expected_memory_mode="analytic_prior",
        expected_evaluation_corruption=EXPECTED_EVALUATION_CORRUPTION,
    )
    all_modes = {**modes, ANALYTIC_MODE: analytic}
    try:
        query_bindings = {
            name: validate_factorized_export_query_binding(
                value["summary"].get("sentence_memory_query_binding"),
                expected_authority="post_spend_environment_manifest_v1",
                expected_manifest_file="manifest_confirmation.jsonl",
                expected_manifest_sha256=partition["manifest_sha256"],
                expected_query_rows=len(partition["manifest_rows"]),
            )
            for name, value in all_modes.items()
        }
    except OrderedDecisionError as error:
        raise ConfirmationInputError(str(error)) from error
    analytic_integrity = _validate_analytic_pair(modes["sentence_memory"], analytic)
    _validate_motion_shuffle_pair(
        modes["sentence_memory"], modes["motion_shuffled_sentence_memory"]
    )
    epochs = {
        int(value["summary"].get("checkpoint_epoch", -1))
        for value in all_modes.values()
    }
    if epochs != {checkpoint_epoch}:
        raise ConfirmationInputError("Five-mode exports do not share the locked epoch")

    all_null = load_integrity_export(
        "all_null",
        all_null_dir,
        partition=partition,
        checkpoint_path=checkpoint_path.resolve(),
        config_path=config_path.resolve(),
    )
    try:
        all_null_query_binding = validate_factorized_export_query_binding(
            all_null["summary"].get("sentence_memory_query_binding"),
            expected_authority="post_spend_environment_manifest_v1",
            expected_manifest_file="manifest_confirmation.jsonl",
            expected_manifest_sha256=partition["manifest_sha256"],
            expected_query_rows=len(partition["manifest_rows"]),
        )
    except OrderedDecisionError as error:
        raise ConfirmationInputError(str(error)) from error
    v2_text_only = load_integrity_export(
        "v2_text_only",
        v2_text_only_dir,
        partition=partition,
        checkpoint_path=v2_checkpoint_path.resolve(),
        config_path=v2_config_path.resolve(),
    )
    v2_checkpoint = _v2_checkpoint_evidence(v2_checkpoint_path.resolve())
    all_null_diagnostics = _validate_all_null_export(all_null)
    all_null_vs_off = _prediction_parity(
        all_null, modes["text_only"], require_exact=True
    )
    selected_off_vs_v2 = _prediction_parity(modes["text_only"], v2_text_only)

    reference_duration = modes["text_only"]["durations"]
    duration_max_abs = max(
        float(np.max(np.abs(value["durations"] - reference_duration), initial=0.0))
        for value in all_modes.values()
    )
    cluster_names, groups = _cluster_index_groups(partition["normalized_texts"])
    if len(cluster_names) != EXPECTED_CONFIRMATION_TEXTS:
        raise ConfirmationInputError("Confirmation statistical unit count is not 540")
    bootstrap = cluster_bootstrap_indices(
        len(cluster_names), samples=int(bootstrap_samples), seed=int(seed)
    )
    attention_modes = {
        name: value for name, value in all_modes.items() if name != "text_only"
    }
    attention_diagnostics, attention_values = _aggregate_attention_diagnostics(
        attention_modes, groups=groups, bootstrap=bootstrap
    )
    analytic_vs_learned = _analytic_learned_differences(
        modes=all_modes,
        groups=groups,
        bootstrap=bootstrap,
        attention_values=attention_values,
    )

    aggregate_metrics: dict[str, Any] = {}
    cluster_values: dict[tuple[str, str, str, str], np.ndarray] = {}
    for alignment in ALIGNMENTS:
        aggregate_metrics[alignment] = {}
        for part in PARTS:
            aggregate_metrics[alignment][part] = {}
            for metric in ("dtw", "ndtw", "ndtw_ref"):
                aggregate_metrics[alignment][part][metric] = {}
                for mode_name, mode in all_modes.items():
                    values = _cluster_mean(mode["metrics"][alignment][part][metric], groups)
                    cluster_values[(alignment, part, metric, mode_name)] = values
                    aggregate_metrics[alignment][part][metric][mode_name] = {
                        "mean": float(values.mean()),
                        "ci95": confidence_interval(values[bootstrap].mean(axis=1)),
                        "cluster_count": len(values),
                        "promotion_comparator": mode_name != ANALYTIC_MODE,
                    }

    hand_values: dict[tuple[str, str, str], np.ndarray] = {}
    aggregate_hands: dict[str, Any] = {}
    for hand in ("lhand", "rhand"):
        aggregate_hands[hand] = {}
        for metric in ("motion_path_error", "jerk_magnitude_ratio"):
            aggregate_hands[hand][metric] = {}
            for mode_name, mode in all_modes.items():
                values = _cluster_mean(mode["metrics"]["default"][hand][metric], groups)
                hand_values[(hand, metric, mode_name)] = values
                aggregate_hands[hand][metric][mode_name] = {
                    "mean": float(values.mean()),
                    "ci95": confidence_interval(values[bootstrap].mean(axis=1)),
                    "cluster_count": len(values),
                    "promotion_comparator": mode_name != ANALYTIC_MODE,
                }

    comparators = (
        "text_only",
        "motion_shuffled_sentence_memory",
        "shuffled_sentence_memory",
    )
    correct_pa = cluster_values[("pa", "wholebody", "ndtw", "sentence_memory")]
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for comparator in comparators:
        reference = cluster_values[("pa", "wholebody", "ndtw", comparator)]
        effect = paired_lower_is_better(correct_pa, reference, bootstrap)
        differences = np.asarray(effect.pop("cluster_differences"))
        p_value = sign_flip_lower_tail_pvalue(
            differences,
            samples=int(bootstrap_samples),
            seed=stable_subseed(seed, f"pa-ndtw-{comparator}"),
        )
        effect["one_sided_randomization_p"] = p_value
        comparisons[comparator] = effect
        raw_p[comparator] = p_value
    corruption_names = (
        "motion_shuffled_sentence_memory",
        "shuffled_sentence_memory",
    )
    adjusted = holm_adjust({name: raw_p[name] for name in corruption_names})
    for comparator, effect in comparisons.items():
        effect["holm_family"] = (
            "motion_shuffle_and_full_shuffle" if comparator in adjusted else None
        )
        effect["holm_adjusted_p"] = adjusted.get(comparator)
        effect["passed"] = bool(
            float(effect["relative_improvement"])
            >= MIN_RELATIVE_PA_NDTW_IMPROVEMENT
            and float(effect["absolute_difference_ci95"][1]) < 0.0
            and (
                float(effect["holm_adjusted_p"]) < 0.05
                if effect["holm_adjusted_p"] is not None
                else True
            )
        )

    motion_rows = _row_motion_mse(
        modes["sentence_memory"], modes["motion_shuffled_sentence_memory"]
    )
    off_rows = _row_motion_mse(modes["sentence_memory"], modes["text_only"])
    motion_mse = _cluster_mean(motion_rows, groups)
    off_mse = _cluster_mean(off_rows, groups)
    rmotion = rmotion_summary(motion_mse, off_mse, bootstrap)
    rmotion["passed"] = bool(
        float(rmotion["value"]) >= MIN_RMOTION_POINT
        and float(rmotion["ci95"][0]) >= MIN_RMOTION_LOWER_CI
    )

    hand_path: dict[str, Any] = {}
    for hand in ("lhand", "rhand"):
        correct = hand_values[(hand, "motion_path_error", "sentence_memory")]
        reference = hand_values[(hand, "motion_path_error", "text_only")]
        correct_boot = correct[bootstrap].mean(axis=1)
        reference_boot = reference[bootstrap].mean(axis=1)
        degradation_boot = (correct_boot - reference_boot) / np.maximum(
            np.abs(reference_boot), 1e-12
        )
        degradation = float(
            (correct.mean() - reference.mean()) / max(abs(reference.mean()), 1e-12)
        )
        hand_path[hand] = {
            "sentence_memory_mean": float(correct.mean()),
            "text_only_mean": float(reference.mean()),
            "relative_degradation": degradation,
            "relative_degradation_ci95": confidence_interval(degradation_boot),
            "noninferiority_margin": MAX_HAND_PATH_RELATIVE_DEGRADATION,
            "passed": bool(
                float(confidence_interval(degradation_boot)[1])
                < MAX_HAND_PATH_RELATIVE_DEGRADATION
            ),
        }

    duration = {
        "maximum_abs_difference_seconds": duration_max_abs,
        "tolerance_seconds": MAX_DURATION_ABS_DIFFERENCE_SECONDS,
        "passed": duration_max_abs <= MAX_DURATION_ABS_DIFFERENCE_SECONDS,
    }
    integrity = {
        "all_null_diagnostics": all_null_diagnostics,
        "all_null_vs_memory_off": all_null_vs_off,
        "memory_off_vs_original_v2": selected_off_vs_v2,
        "stored_initialization_parity": checkpoint_evidence["parity"],
        "analytic_prior_same_retrieval_inputs": analytic_integrity,
    }
    integrity["passed"] = bool(
        all_null_diagnostics["passed"]
        and all_null_vs_off["passed"]
        and selected_off_vs_v2["passed"]
        and analytic_integrity["passed"]
    )
    gates = {
        "pa_ndtw_wholebody": {
            "minimum_relative_improvement": MIN_RELATIVE_PA_NDTW_IMPROVEMENT,
            "comparisons": comparisons,
            "passed": all(value["passed"] for value in comparisons.values()),
        },
        "rmotion": rmotion,
        "hand_path": {
            "maximum_relative_degradation": MAX_HAND_PATH_RELATIVE_DEGRADATION,
            "hands": hand_path,
            "passed": all(value["passed"] for value in hand_path.values()),
        },
        "duration_invariance": duration,
        "selected_checkpoint_integrity": integrity,
    }
    promoted = all(bool(value["passed"]) for value in gates.values())

    cluster_rows = []
    for index, text in enumerate(cluster_names):
        row: dict[str, Any] = {
            "cluster_index": index,
            "normalized_text": text,
            "signer_rows": int(len(groups[index])),
            "motion_shuffled_mse": float(motion_mse[index]),
            "memory_off_mse": float(off_mse[index]),
        }
        for mode in all_modes:
            row[f"pa_ndtw_wholebody_{mode}"] = float(
                cluster_values[("pa", "wholebody", "ndtw", mode)][index]
            )
            for hand in ("lhand", "rhand"):
                row[f"motion_path_error_{hand}_{mode}"] = float(
                    hand_values[(hand, "motion_path_error", mode)][index]
                )
        cluster_rows.append(row)

    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "scientific_split": "val_confirmation_novel_text_only",
        "test_data_accessed": False,
        "confirmation_holdout_spent": True,
        "holdout_spend": spend,
        "statistical_unit": "normalized sentence text; signer rows averaged first",
        "cluster_count": len(cluster_names),
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "confidence": 0.95,
            "multiple_testing": (
                "Holm over only motion-shuffle and full-shuffle PA-nDTW comparisons"
            ),
        },
        "authorization": authorization,
        "original_v2_checkpoint": v2_checkpoint,
        "partition": {
            "path": str(partition["dir"]),
            "partition_digest": partition["payload"]["partition_digest"],
            "confirmation_manifest_sha256": partition["manifest_sha256"],
            "confirmation_rows": len(partition["manifest_rows"]),
        },
        "inputs": {
            name: {
                "directory": str(value["dir"]),
                "export_summary_sha256": sha256_file(value["summary_path"]),
                "dtw_artifacts": {
                    alignment: {
                        "json_sha256": sha256_file(
                            Path(value["dir"]) / DTW_FILE[alignment]
                        ),
                        "csv_sha256": sha256_file(
                            (Path(value["dir"]) / DTW_FILE[alignment]).with_suffix(
                                ".csv"
                            )
                        ),
                    }
                    for alignment in ALIGNMENTS
                },
                "sentence_memory_mode": (
                    "analytic_prior"
                    if name == ANALYTIC_MODE
                    else PRIMARY_MODE_MEMORY[name]
                ),
                "promotion_comparator": name != ANALYTIC_MODE,
                "sentence_memory_query_binding": query_bindings[name],
            }
            for name, value in all_modes.items()
        },
        "aggregate_metrics": aggregate_metrics,
        "aggregate_hand_diagnostics": aggregate_hands,
        "confirmation_attention_contract": {
            "layer": "final",
            "head_reduction": "mean",
            "part_specific": True,
            "token_specific": True,
            "per_head_available": False,
            "richer_per_head_and_causal_evidence_source": (
                "locked_development_temporal_slot_diagnostic"
            ),
        },
        "confirmation_attention_diagnostics": attention_diagnostics,
        "analytic_prior_vs_learned": analytic_vs_learned,
        "locked_development_temporal_slot_diagnostic": development_diagnostic,
        "all_null_sentence_memory_query_binding": all_null_query_binding,
        "promotion_gates": gates,
        "promoted_factorized_memory_checkpoint": promoted,
        "analytic_prior_interpretation_only": True,
        "stage2_after_confirmation_permitted": False,
        "cluster_rows_identity": _digest_json(cluster_rows),
        "interpretation": (
            "If any confirmation gate fails, the holdout remains spent and no "
            "adaptive Stage 2, threshold change, checkpoint promotion, test access, "
            "or Phase B is permitted."
        ),
    }
    return summary, cluster_rows


def write_outputs(
    out_dir: Path, summary: Mapping[str, Any], cluster_rows: list[Mapping[str, Any]]
) -> None:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        _verify_existing_outputs(out_dir, summary=summary, cluster_rows=cluster_rows)
        return
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    building = out_dir.with_name(f".{out_dir.name}.building.{attempt}")
    if building.exists():
        raise ConfirmationInputError(f"Incomplete confirmation build exists: {building}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        (building / "confirmation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (building / "confirmation_clusters.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
            writer.writeheader()
            writer.writerows(cluster_rows)
        gates = summary["promotion_gates"]
        report = [
            "# Phase-A'' factorized-memory confirmation",
            "",
            f"- Stage: `{summary['stage']}`",
            "- Decision: "
            f"**{'PASS' if summary['promoted_factorized_memory_checkpoint'] else 'FAIL'}**",
            f"- Population: {summary['cluster_count']} novel validation text clusters",
            f"- Whole-body PA-nDTW gate: {gates['pa_ndtw_wholebody']['passed']}",
            f"- Rmotion: {gates['rmotion']['value']:.6f} "
            f"(95% CI {gates['rmotion']['ci95']})",
            f"- Hand-path gate: {gates['hand_path']['passed']}",
            f"- Duration parity: {gates['duration_invariance']['passed']}",
            f"- Integrity gate: {gates['selected_checkpoint_integrity']['passed']}",
            "- Analytic-prior mode is explanatory only and never a comparator.",
            "",
            "The confirmation holdout is spent regardless of this decision. A failure "
            "cannot authorize an adaptive Stage 2 or Phase B.",
            "",
        ]
        (building / "report.md").write_text("\n".join(report), encoding="utf-8")
        files = {}
        for name in ("confirmation_summary.json", "confirmation_clusters.csv", "report.md"):
            path = building / name
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        manifest = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "summary_identity": _digest_json(summary),
            "files": files,
        }
        (building / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (building / "READY").write_text(
            json.dumps(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "summary_identity": manifest["summary_identity"],
                    "promoted_factorized_memory_checkpoint": summary[
                        "promoted_factorized_memory_checkpoint"
                    ],
                    "confirmation_holdout_spent": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(building, out_dir)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise


def _verify_existing_outputs(
    out_dir: Path,
    *,
    summary: Mapping[str, Any],
    cluster_rows: list[Mapping[str, Any]],
) -> None:
    """Accept only an exact, READY-complete continuation of this analysis."""

    existing_summary = _json(out_dir / "confirmation_summary.json")
    if existing_summary != dict(summary):
        raise ConfirmationInputError(
            "Existing confirmation analysis belongs to different inputs/results"
        )
    if existing_summary.get("cluster_rows_identity") != _digest_json(cluster_rows):
        raise ConfirmationInputError("Existing confirmation cluster identity changed")
    manifest = _json(out_dir / "artifact_manifest.json")
    ready = _json(out_dir / "READY")
    summary_identity = _digest_json(summary)
    if (
        manifest.get("schema_name") != SCHEMA_NAME
        or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or manifest.get("summary_identity") != summary_identity
        or ready.get("schema_name") != SCHEMA_NAME
        or int(ready.get("schema_version", -1)) != SCHEMA_VERSION
        or ready.get("summary_identity") != summary_identity
        or ready.get("confirmation_holdout_spent") is not True
        or ready.get("promoted_factorized_memory_checkpoint")
        is not bool(summary["promoted_factorized_memory_checkpoint"])
    ):
        raise ConfirmationInputError("Existing confirmation READY identity is invalid")
    files = dict(manifest.get("files", {}) or {})
    for name in (
        "confirmation_summary.json",
        "confirmation_clusters.csv",
        "report.md",
    ):
        evidence = dict(files.get(name, {}) or {})
        path = out_dir / name
        if (
            not path.is_file()
            or sha256_file(path) != evidence.get("sha256")
            or path.stat().st_size != int(evidence.get("bytes", -1))
        ):
            raise ConfirmationInputError(
                f"Existing confirmation artifact changed: {name}"
            )
    with (out_dir / "confirmation_clusters.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        if len(list(csv.DictReader(handle))) != len(cluster_rows):
            raise ConfirmationInputError(
                "Existing confirmation cluster CSV is incomplete"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--holdout_spend", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition_dir", type=Path, required=True)
    parser.add_argument("--text_only_dir", type=Path, required=True)
    parser.add_argument("--sentence_memory_dir", type=Path, required=True)
    parser.add_argument("--motion_shuffled_sentence_memory_dir", type=Path, required=True)
    parser.add_argument("--shuffled_sentence_memory_dir", type=Path, required=True)
    parser.add_argument("--analytic_prior_sentence_memory_dir", type=Path, required=True)
    parser.add_argument("--development_diagnostic_dir", type=Path, required=True)
    parser.add_argument("--all_null_dir", type=Path, required=True)
    parser.add_argument("--v2_text_only_dir", type=Path, required=True)
    parser.add_argument("--v2_checkpoint", type=Path, required=True)
    parser.add_argument("--v2_config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, rows = analyze_confirmation(
        authorization_path=args.authorization,
        holdout_spend_path=args.holdout_spend,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        v2_checkpoint_path=args.v2_checkpoint,
        v2_config_path=args.v2_config,
        partition_dir=args.partition_dir,
        mode_dirs={
            "text_only": args.text_only_dir,
            "sentence_memory": args.sentence_memory_dir,
            "motion_shuffled_sentence_memory": args.motion_shuffled_sentence_memory_dir,
            "shuffled_sentence_memory": args.shuffled_sentence_memory_dir,
        },
        analytic_prior_dir=args.analytic_prior_sentence_memory_dir,
        development_diagnostic_dir=args.development_diagnostic_dir,
        all_null_dir=args.all_null_dir,
        v2_text_only_dir=args.v2_text_only_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_outputs(args.out_dir, summary, rows)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "stage": summary["stage"],
                "cluster_count": summary["cluster_count"],
                "promoted": summary["promoted_factorized_memory_checkpoint"],
            },
            indent=2,
        )
    )
    if not summary["promoted_factorized_memory_checkpoint"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
