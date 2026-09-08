"""Analyze the one-shot Phase-A''' centered-memory confirmation.

The analyzer is deliberately separate from Phase-A'': old exports and gates
retain their byte-for-byte protocol.  This module accepts only the locked
centered Stage A/B controls, reduces signer realizations within normalized text
before inference, and treats the shared confirmation spend marker as
irreversible regardless of promotion outcome.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from NIAF.continuous_trajectory_field.phase_a_motion_contrast import (
    cluster_bootstrap_indices,
    confidence_interval,
    holm_adjust,
    paired_lower_is_better,
    rmotion_summary,
    sign_flip_lower_tail_pvalue,
    stable_subseed,
)
from NIAF.continuous_trajectory_field.sentence_memory import sha256_file
from NIAF.continuous_trajectory_field.scripts.analyze_factorized_memory_confirmation import (
    _aggregate_attention_diagnostics,
    _validate_analytic_pair,
)
from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    ALIGNMENTS,
    PARTS,
    ConfirmationInputError,
    _cluster_index_groups,
    _cluster_mean,
    _prediction_parity,
    _row_motion_mse,
    _validate_all_null_export,
    _v2_checkpoint_evidence,
    load_confirmation_partition,
    load_integrity_export,
    load_mode_export,
)
from NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage import (
    DIAGNOSTIC_READY_SCHEMA_NAME,
    DIAGNOSTIC_SCHEMA_NAME,
    EVALUATION_CORRUPTION,
    EXPERIMENTS,
    GLOBAL_HOLDOUT_SPEND_RELATIVE,
    HOLDOUT_SPEND_SCHEMA_NAME,
    SCHEMA_VERSION as DECISION_SCHEMA_VERSION,
    SOURCE_ROOT,
    STAGE1,
    STAGE1_EVAL_MODES,
    STAGE2,
    STAGE2_EVAL_MODES,
    verify_authorization,
)
from NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage import (
    DEVELOPMENT_FEASIBLE,
    EXPECTED_CONFIRMATION_TEXTS,
    EXPECTED_PARTITION_DIGEST,
    OrderedDecisionError,
    validate_factorized_export_query_binding,
)


SCHEMA_NAME = "signtrajfield_centered_memory_confirmation"
SCHEMA_VERSION = 1
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 1234
MIN_RELATIVE_IMPROVEMENT = 0.005
MIN_RPAIR_POINT = 0.10
MIN_RPAIR_LOWER = 0.05
MIN_IDENTITY_UTILITY = 0.25
MAX_HAND_DEGRADATION = 0.02
PARITY_TOLERANCE = 1e-7
PAIR_MODES = (
    "motion_shuffled_n0",
    "motion_shuffled_n1",
    "motion_shuffled_n2",
)
PROMOTION_COMPARATORS = (
    "text_only",
    "mean_motion_derangement",
    "cross_query_motion",
    "full_replacement",
)
HOLM_FAMILY = (
    "mean_motion_derangement",
    "cross_query_motion",
    "full_replacement",
)
MODE_LABELS = {
    "off": "text_only",
    "on": "sentence_memory",
    "motion_shuffled_n0": "motion_shuffled_n0",
    "motion_shuffled_n1": "motion_shuffled_n1",
    "motion_shuffled_n2": "motion_shuffled_n2",
    "cross_query_motion": "cross_query_motion",
    "full_replacement": "full_replacement",
    "joint_tuple_permuted": "joint_tuple_permuted",
    "uniform_final_mass": "uniform_final_mass",
    "analytic_prior": "analytic_prior",
    "association_disabled": "association_disabled",
    "broadcast_complete": "broadcast_complete",
}
SAME_METADATA_MODES = {
    *PAIR_MODES,
    "cross_query_motion",
    "uniform_final_mass",
    "analytic_prior",
    "association_disabled",
    "broadcast_complete",
}
METADATA_FIELDS = (
    "sentence_memory_ids",
    "sentence_memory_group_ids",
    "sentence_memory_source_group_ids",
    "sentence_memory_scores",
    "sentence_memory_durations",
    "sentence_memory_duration_log_gap",
    "sentence_memory_candidate_mask",
)


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
        raise ConfirmationInputError(f"Required JSON does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationInputError(f"Expected a JSON object: {path}")
    return value


def _canonical_csv_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ConfirmationInputError("Cannot publish an empty post-selection table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _durable_write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Publish a complete directory atomically without replacing evidence."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ConfirmationInputError(
            "Atomic post-selection publication requires renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ConfirmationInputError(
                f"Post-selection output already exists: {destination}"
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _validate_exact_completed_directory(
    directory: Path,
    *,
    expected_files: set[str],
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ConfirmationInputError(
            f"Post-selection output is not a real directory: {directory}"
        )
    actual = {child.name for child in directory.iterdir()}
    if actual != expected_files or any(
        not child.is_file() or child.is_symlink() for child in directory.iterdir()
    ):
        raise ConfirmationInputError(
            f"Completed post-selection directory has unexpected contents: {directory}"
        )


def _sample_path(export: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("sample", "")))
    candidates = (path,) if path.is_absolute() else (
        Path(export["dir"]) / path,
        Path(export["dir"]) / path.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfirmationInputError(f"Cannot resolve export sample {path}")


def _authorization_evidence(
    authorization_path: Path,
    *,
    checkpoint_path: Path,
    config_path: Path,
    purpose: str = "confirmation",
) -> tuple[str, dict[str, Any]]:
    raw = _json(authorization_path)
    stage = str(raw.get("stage", ""))
    if stage not in {STAGE1, STAGE2}:
        raise ConfirmationInputError("Centered authorization has an unknown stage")
    try:
        authorization = verify_authorization(
            authorization_path, purpose=purpose, stage=stage
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfirmationInputError(str(error)) from error
    checkpoint = dict(authorization.get("checkpoint", {}) or {})
    config = dict(checkpoint.get("config", {}) or {})
    if (
        Path(str(checkpoint.get("path", ""))).resolve() != checkpoint_path.resolve()
        or sha256_file(checkpoint_path) != checkpoint.get("sha256")
        or Path(str(config.get("path", ""))).resolve() != config_path.resolve()
        or sha256_file(config_path) != config.get("sha256")
    ):
        raise ConfirmationInputError("Requested checkpoint/config is not authorized")
    decision = _json(authorization_path.parent / "decision.json")
    if (
        decision.get("status") != DEVELOPMENT_FEASIBLE
        or decision.get("integrity_valid") is not True
        or decision.get("stage") != stage
    ):
        raise ConfirmationInputError("Authorization is not from a feasible stage")
    return stage, authorization


def _validate_spend_marker(
    marker_path: Path,
    *,
    authorization_path: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected_path = (SOURCE_ROOT / GLOBAL_HOLDOUT_SPEND_RELATIVE).resolve()
    if marker_path.resolve() != expected_path:
        raise ConfirmationInputError("Confirmation used a non-global spend marker")
    marker = _json(marker_path)
    checkpoint = dict(authorization.get("checkpoint", {}) or {})
    required = {
        "schema_name": HOLDOUT_SPEND_SCHEMA_NAME,
        "schema_version": DECISION_SCHEMA_VERSION,
        "confirmation_holdout_spent": True,
        "authorization_path": str(authorization_path.resolve()),
        "authorization_identity": authorization.get("authorization_identity"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "partition_digest": EXPECTED_PARTITION_DIGEST,
    }
    if any(marker.get(name) != value for name, value in required.items()):
        raise ConfirmationInputError("Confirmation spend marker is invalid")
    payload = {key: value for key, value in marker.items() if key != "spend_identity"}
    if marker.get("spend_identity") != _digest_json(payload):
        raise ConfirmationInputError("Confirmation spend identity is invalid")
    return marker


def _expected_identities(authorization: Mapping[str, Any]) -> dict[str, Any]:
    identities = dict(
        dict(authorization.get("checkpoint", {}) or {}).get("identities", {}) or {}
    )
    result = {
        name: dict(value or {}).get("value") for name, value in identities.items()
    }
    result["relevance_calibration"] = dict(
        dict(authorization.get("checkpoint", {}) or {}).get(
            "relevance_calibration", {}
        )
        or {}
    )
    if any(value is None for value in result.values()) or not result:
        raise ConfirmationInputError("Authorization lacks complete centered identities")
    return result


def _validate_export_identities(
    exports: Mapping[str, Mapping[str, Any]],
    *,
    expected: Mapping[str, Any],
) -> None:
    for label, export in exports.items():
        actual = dict(
            export.get("summary", {}).get(
                "sentence_memory_checkpoint_identities", {}
            )
            or {}
        )
        if actual != dict(expected):
            raise ConfirmationInputError(
                f"{label} export has detached architecture/control identities"
            )


def _load_sample(export: Mapping[str, Any], index: int):
    return np.load(
        _sample_path(export, export["rows"][index]), allow_pickle=False
    )


def _require_array_equal(first, second, field: str, label: str) -> None:
    if field not in first.files or field not in second.files:
        raise ConfirmationInputError(f"{label} lacks metadata field {field}")
    if not np.array_equal(first[field], second[field]):
        raise ConfirmationInputError(f"{label} changed metadata field {field}")


def validate_centered_control_pair(
    correct: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Validate the causal intervention and its fixed nonce row by row."""

    if len(correct["rows"]) != len(control["rows"]):
        raise ConfirmationInputError(f"{mode} row count changed")
    expected_nonce = EVALUATION_CORRUPTION["nonces"].get(mode)
    informative_rows = 0
    for index in range(len(correct["rows"])):
        with _load_sample(correct, index) as normal, _load_sample(control, index) as changed:
            payload_field = "sentence_memory_motion_payload_digest"
            if payload_field not in normal.files or payload_field not in changed.files:
                raise ConfirmationInputError(
                    f"{mode} lacks complete motion-package digests"
                )
            normal_payload = np.asarray(normal[payload_field])
            changed_payload = np.asarray(changed[payload_field])
            if mode in SAME_METADATA_MODES:
                for field in METADATA_FIELDS:
                    _require_array_equal(normal, changed, field, mode)
            if mode in {
                "uniform_final_mass",
                "analytic_prior",
                "association_disabled",
            } and not np.array_equal(normal_payload, changed_payload):
                raise ConfirmationInputError(f"{mode} changed the motion payload")
            if expected_nonce is not None:
                for field, expected in (
                    ("sentence_memory_evaluation_corruption_mode", "fixed_evidence_controls_v1"),
                    ("sentence_memory_evaluation_corruption_nonce", expected_nonce),
                    ("sentence_memory_evaluation_corruption_condition", mode),
                ):
                    if field not in changed.files or str(
                        np.asarray(changed[field]).reshape(-1)[0]
                    ) != str(expected):
                        raise ConfirmationInputError(
                            f"{mode} has the wrong fixed-control provenance"
                        )
            if mode in PAIR_MODES:
                fields = (
                    "sentence_memory_motion_candidate_permutation",
                    "sentence_memory_motion_source_ids",
                    "sentence_memory_motion_shuffle_informative",
                )
                if any(field not in changed.files for field in fields):
                    raise ConfirmationInputError(f"{mode} lacks derangement provenance")
                ids = np.asarray(normal["sentence_memory_ids"], dtype=np.int64)
                mask = np.asarray(normal["sentence_memory_candidate_mask"], dtype=bool)
                permutation = np.asarray(
                    changed["sentence_memory_motion_candidate_permutation"],
                    dtype=np.int64,
                )
                if not bool(changed["sentence_memory_motion_shuffle_informative"]):
                    raise ConfirmationInputError(f"{mode} is not informative")
                valid = np.flatnonzero(mask)
                if (
                    permutation.shape != ids.shape
                    or len(valid) < 2
                    or np.any(permutation[valid] == valid)
                    or not np.array_equal(
                        np.asarray(changed["sentence_memory_motion_source_ids"]),
                        ids[permutation],
                    )
                    or not np.array_equal(
                        changed_payload,
                        normal_payload[permutation],
                    )
                ):
                    raise ConfirmationInputError(f"{mode} is not a valid derangement")
                informative_rows += 1
            elif mode == "cross_query_motion":
                if (
                    "sentence_memory_cross_query_motion_informative" not in changed.files
                    or "sentence_memory_motion_source_ids" not in changed.files
                    or not bool(changed["sentence_memory_cross_query_motion_informative"])
                ):
                    raise ConfirmationInputError("Cross-query motion is not informative")
                if np.array_equal(
                    np.asarray(changed["sentence_memory_motion_source_ids"]),
                    np.asarray(normal["sentence_memory_ids"]),
                ):
                    raise ConfirmationInputError("Cross-query motion kept every source ID")
                valid = np.asarray(normal["sentence_memory_candidate_mask"], dtype=bool)
                if np.array_equal(changed_payload[valid], normal_payload[valid]):
                    raise ConfirmationInputError("Cross-query motion kept every payload")
                informative_rows += 1
            elif mode == "full_replacement":
                normal_ids = np.asarray(normal["sentence_memory_ids"])
                changed_ids = np.asarray(changed["sentence_memory_ids"])
                valid = np.asarray(changed["sentence_memory_candidate_mask"], dtype=bool)
                if not valid.any() or np.array_equal(normal_ids[valid], changed_ids[valid]):
                    raise ConfirmationInputError("Full replacement kept the original tuple")
                informative_rows += 1
            elif mode == "joint_tuple_permuted":
                permutation_field = "sentence_memory_joint_tuple_candidate_permutation"
                if permutation_field not in changed.files:
                    raise ConfirmationInputError("Joint-tuple export lacks permutation")
                permutation = np.asarray(changed[permutation_field], dtype=np.int64)
                informative_field = "sentence_memory_joint_tuple_informative"
                if (
                    informative_field not in changed.files
                    or not bool(np.asarray(changed[informative_field]).reshape(-1)[0])
                    or permutation.shape != normal_payload.shape
                    or not np.array_equal(
                        changed_payload, normal_payload[permutation]
                    )
                ):
                    raise ConfirmationInputError(
                        "Joint tuple did not permute the complete motion package"
                    )
                for field in METADATA_FIELDS:
                    if field not in normal.files or field not in changed.files:
                        raise ConfirmationInputError(f"Joint tuple lacks {field}")
                    source = np.asarray(normal[field])
                    if source.ndim == 0 or not np.array_equal(
                        changed[field], source[permutation]
                    ):
                        raise ConfirmationInputError(
                            f"Joint tuple did not permute {field} consistently"
                        )
            elif mode == "broadcast_complete":
                rank_field = "sentence_memory_broadcast_motion_source_rank"
                source_field = "sentence_memory_motion_source_ids"
                nonce_field = "sentence_memory_broadcast_motion_audit_nonce"
                informative_field = "sentence_memory_broadcast_motion_informative"
                if (
                    rank_field not in changed.files
                    or source_field not in changed.files
                    or nonce_field not in changed.files
                    or informative_field not in changed.files
                ):
                    raise ConfirmationInputError("Broadcast export lacks source provenance")
                rank = int(np.asarray(changed[rank_field]).reshape(-1)[0])
                ids = np.asarray(normal["sentence_memory_ids"], dtype=np.int64)
                source_ids = np.asarray(changed[source_field], dtype=np.int64)
                if rank < 0 or rank >= len(ids) or not np.all(source_ids == ids[rank]):
                    raise ConfirmationInputError("Broadcast source package is inconsistent")
                supported = np.asarray(
                    normal["sentence_memory_candidate_mask"], dtype=bool
                )
                if (
                    not bool(np.asarray(changed[informative_field]).reshape(-1)[0])
                    or supported.sum() < 2
                    or not np.all(changed_payload[supported] == normal_payload[rank])
                ):
                    raise ConfirmationInputError(
                        "Broadcast did not copy one complete package to every candidate"
                    )
                if str(np.asarray(changed[nonce_field]).item()) != (
                    "csl_daily_broadcast_motion_payload_audit_v1"
                ):
                    raise ConfirmationInputError("Broadcast audit nonce changed")
            elif mode == "uniform_final_mass":
                mass_field = "sentence_memory_final_part_candidate_mass"
                support_field = "sentence_memory_candidate_support"
                if mass_field not in changed.files or support_field not in changed.files:
                    raise ConfirmationInputError("Uniform control lacks final mass/support")
                mass = np.asarray(changed[mass_field])
                support = np.asarray(changed[support_field], dtype=bool)
                if mass.shape != support.shape or np.any(mass[~support] != 0.0):
                    raise ConfirmationInputError("Uniform control used unsupported mass")
                for query_mass, query_support in zip(
                    mass.reshape(-1, mass.shape[-1]),
                    support.reshape(-1, support.shape[-1]),
                ):
                    values = query_mass[query_support]
                    if len(values) > 1 and not np.all(values == values[0]):
                        raise ConfirmationInputError(
                            "Forced final candidate mass is not exactly uniform"
                        )
            attention_expected = {
                "uniform_final_mass": "uniform_final_candidate_mass",
                "analytic_prior": "analytic_prior",
                "association_disabled": "association_disabled",
            }.get(mode)
            if attention_expected is not None:
                if str(np.asarray(changed["sentence_memory_attention_mode"]).item()) != attention_expected:
                    raise ConfirmationInputError(f"{mode} used the wrong attention mode")
    return {
        "mode": mode,
        "rows": len(correct["rows"]),
        "informative_rows": informative_rows,
        "fixed_nonce": expected_nonce,
        "passed": True,
    }


def compute_centered_confirmation_statistics(
    *,
    pa_ndtw: Mapping[str, np.ndarray],
    pair_mse: Mapping[str, np.ndarray],
    off_mse: np.ndarray,
    hand_path: Mapping[str, Mapping[str, np.ndarray]],
    bootstrap: np.ndarray,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute all causal promotion statistics from text-cluster values."""

    correct = np.asarray(pa_ndtw["sentence_memory"], dtype=np.float64)
    mean_pair = np.mean(
        np.stack([np.asarray(pa_ndtw[name]) for name in PAIR_MODES], axis=0),
        axis=0,
    )
    references = {
        "text_only": np.asarray(pa_ndtw["text_only"], dtype=np.float64),
        "mean_motion_derangement": mean_pair,
        "cross_query_motion": np.asarray(pa_ndtw["cross_query_motion"], dtype=np.float64),
        "full_replacement": np.asarray(pa_ndtw["full_replacement"], dtype=np.float64),
    }
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for name, reference in references.items():
        effect = paired_lower_is_better(correct, reference, bootstrap)
        differences = np.asarray(effect.pop("cluster_differences"))
        p_value = sign_flip_lower_tail_pvalue(
            differences,
            samples=int(bootstrap.shape[0]),
            seed=stable_subseed(seed, f"centered-pa-ndtw-{name}"),
        )
        effect["one_sided_randomization_p"] = p_value
        comparisons[name] = effect
        raw_p[name] = p_value
    adjusted = holm_adjust({name: raw_p[name] for name in HOLM_FAMILY})
    for name, effect in comparisons.items():
        effect["holm_adjusted_p"] = adjusted.get(name)
        effect["passed"] = bool(
            float(effect["relative_improvement"]) >= MIN_RELATIVE_IMPROVEMENT
            and float(effect["absolute_difference_ci95"][1]) < 0.0
            and (
                float(effect["holm_adjusted_p"]) < 0.05
                if name in HOLM_FAMILY
                else True
            )
        )

    pair_numerator = np.mean(
        np.stack([np.asarray(pair_mse[name]) for name in PAIR_MODES], axis=0),
        axis=0,
    )
    rpair = rmotion_summary(pair_numerator, np.asarray(off_mse), bootstrap)
    rpair["definition"] = (
        "sqrt(mean-over-three-nonces cluster-MSE(correct,deranged) / "
        "cluster-MSE(correct,memory-off))"
    )
    rpair["passed"] = bool(
        float(rpair["value"]) >= MIN_RPAIR_POINT
        and float(rpair["ci95"][0]) >= MIN_RPAIR_LOWER
    )

    off = references["text_only"]
    denominator = float(off.mean() - correct.mean())
    numerator = float(mean_pair.mean() - correct.mean())
    off_boot = off[bootstrap].mean(axis=1)
    correct_boot = correct[bootstrap].mean(axis=1)
    pair_boot = mean_pair[bootstrap].mean(axis=1)
    utility_boot = (pair_boot - correct_boot) / np.maximum(
        off_boot - correct_boot, 1e-12
    )
    identity_utility = {
        "definition": "(mean-derangement PA-nDTW - correct)/(off - correct)",
        "value": numerator / max(denominator, 1e-12),
        "ci95": confidence_interval(utility_boot),
        "minimum": MIN_IDENTITY_UTILITY,
        "positive_denominator": denominator > 0.0,
    }
    identity_utility["passed"] = bool(
        identity_utility["positive_denominator"]
        and float(identity_utility["value"]) >= MIN_IDENTITY_UTILITY
    )

    hands = {}
    for hand in ("lhand", "rhand"):
        correct_hand = np.asarray(hand_path[hand]["sentence_memory"], dtype=np.float64)
        off_hand = np.asarray(hand_path[hand]["text_only"], dtype=np.float64)
        degradation = (correct_hand.mean() - off_hand.mean()) / max(
            abs(float(off_hand.mean())), 1e-12
        )
        degradation_boot = (
            correct_hand[bootstrap].mean(axis=1)
            - off_hand[bootstrap].mean(axis=1)
        ) / np.maximum(np.abs(off_hand[bootstrap].mean(axis=1)), 1e-12)
        interval = confidence_interval(degradation_boot)
        hands[hand] = {
            "relative_degradation": float(degradation),
            "relative_degradation_ci95": interval,
            "upper_bound_maximum": MAX_HAND_DEGRADATION,
            "passed": float(interval[1]) < MAX_HAND_DEGRADATION,
        }
    return {
        "pa_ndtw_wholebody": {
            "comparisons": comparisons,
            "passed": all(value["passed"] for value in comparisons.values()),
        },
        "Rpair": rpair,
        "identity_utility": identity_utility,
        "hand_path": {
            "hands": hands,
            "passed": all(value["passed"] for value in hands.values()),
        },
    }


def _association_matching_diagnostics(
    export: Mapping[str, Any],
    *,
    groups: list[np.ndarray],
    bootstrap: np.ndarray,
) -> dict[str, Any] | None:
    row_accuracy = []
    row_margin = []
    row_gate = []
    for row in export["rows"]:
        with np.load(_sample_path(export, row), allow_pickle=False) as sample:
            if "sentence_memory_association_key_descriptor" not in sample.files:
                return None
            required = (
                "sentence_memory_association_motion_descriptor",
                "sentence_memory_association_gate",
                "sentence_memory_group_ids",
                "sentence_memory_candidate_mask",
            )
            if any(name not in sample.files for name in required):
                raise ConfirmationInputError("Association diagnostic is incomplete")
            key = np.asarray(
                sample["sentence_memory_association_key_descriptor"], dtype=np.float64
            )
            motion = np.asarray(
                sample["sentence_memory_association_motion_descriptor"],
                dtype=np.float64,
            )
            gate = np.asarray(
                sample["sentence_memory_association_gate"], dtype=np.float64
            )
            group = np.asarray(sample["sentence_memory_group_ids"], dtype=np.int64)
            valid = np.asarray(
                sample["sentence_memory_candidate_mask"], dtype=bool
            )
        if key.shape != motion.shape or key.ndim != 2 or key.shape[0] != len(valid):
            raise ConfirmationInputError("Association descriptor shape changed")
        if (
            gate.shape != valid.shape
            or group.shape != valid.shape
            or not np.isfinite(key).all()
            or not np.isfinite(motion).all()
            or not np.isfinite(gate).all()
            or np.any(gate < 0.0)
            or np.any(gate > 1.0)
        ):
            raise ConfirmationInputError("Association diagnostic is malformed")
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            raise ConfirmationInputError("Association confirmation row has <2 candidates")
        key_norm = np.linalg.norm(key[indices], axis=-1)
        motion_norm = np.linalg.norm(motion[indices], axis=-1)
        if (
            np.max(np.abs(key_norm - 1.0), initial=0.0) > 1e-5
            or np.max(np.abs(motion_norm - 1.0), initial=0.0) > 1e-5
        ):
            raise ConfirmationInputError("Association descriptors are not normalized")
        similarity = key[indices] @ motion[indices].T
        group_valid = group[indices]
        correct = []
        margins = []
        for anchor in range(len(indices)):
            positive = group_valid == group_valid[anchor]
            negative = ~positive
            if not negative.any():
                continue
            predicted = int(np.argmax(similarity[anchor]))
            correct.append(bool(positive[predicted]))
            margins.append(
                float(
                    similarity[anchor, positive].max()
                    - similarity[anchor, negative].max()
                )
            )
        if not correct:
            raise ConfirmationInputError("Association row has no unambiguous negatives")
        row_accuracy.append(float(np.mean(correct)))
        row_margin.append(float(np.mean(margins)))
        row_gate.append(float(gate[valid].mean()))
    output = {}
    for name, values in (
        ("top1_accuracy", row_accuracy),
        ("true_minus_best_negative_margin", row_margin),
        ("association_gate_mean", row_gate),
    ):
        cluster = _cluster_mean(np.asarray(values), groups)
        output[name] = {
            "mean": float(cluster.mean()),
            "ci95": confidence_interval(cluster[bootstrap].mean(axis=1)),
        }
    output["explanatory_only"] = True
    return output


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
    broadcast_dir: Path,
    all_null_dir: Path,
    v2_text_only_dir: Path,
    development_diagnostic_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if int(bootstrap_samples) != BOOTSTRAP_SAMPLES or int(seed) != BOOTSTRAP_SEED:
        raise ConfirmationInputError("Centered confirmation requires 10,000/1234 bootstrap")
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
    expected_modes = STAGE1_EVAL_MODES if stage == STAGE1 else STAGE2_EVAL_MODES
    if tuple(mode_dirs) != tuple(expected_modes):
        raise ConfirmationInputError("Confirmation mode order differs from protocol")
    partition = load_confirmation_partition(partition_dir)
    if partition["payload"].get("partition_digest") != EXPECTED_PARTITION_DIGEST:
        raise ConfirmationInputError("Confirmation partition digest changed")
    checkpoint_epoch = int(dict(authorization["checkpoint"])["epoch"])
    exports = {
        mode: load_mode_export(
            MODE_LABELS[mode],
            Path(mode_dirs[mode]),
            partition=partition,
            checkpoint_path=checkpoint_path.resolve(),
            config_path=config_path.resolve(),
            checkpoint_epoch=checkpoint_epoch,
            motion_shuffle_seed=seed,
            expected_memory_mode=mode,
            expected_evaluation_corruption=EVALUATION_CORRUPTION,
        )
        for mode in expected_modes
    }
    broadcast = load_mode_export(
        MODE_LABELS["broadcast_complete"],
        broadcast_dir,
        partition=partition,
        checkpoint_path=checkpoint_path.resolve(),
        config_path=config_path.resolve(),
        checkpoint_epoch=checkpoint_epoch,
        motion_shuffle_seed=seed,
        expected_memory_mode="broadcast_complete",
        expected_evaluation_corruption=EVALUATION_CORRUPTION,
    )
    all_null = load_integrity_export(
        "all_null",
        all_null_dir,
        partition=partition,
        checkpoint_path=checkpoint_path.resolve(),
        config_path=config_path.resolve(),
    )
    v2 = load_integrity_export(
        "v2_text_only",
        v2_text_only_dir,
        partition=partition,
        checkpoint_path=v2_checkpoint_path.resolve(),
        config_path=v2_config_path.resolve(),
    )
    expected_identities = _expected_identities(authorization)
    _validate_export_identities(
        {
            **exports,
            "broadcast_complete": broadcast,
            "all_null": all_null,
        },
        expected=expected_identities,
    )

    query_bindings = {}
    for mode, export in {**exports, "broadcast_complete": broadcast}.items():
        try:
            query_bindings[mode] = validate_factorized_export_query_binding(
                export["summary"].get("sentence_memory_query_binding"),
                expected_authority="post_spend_environment_manifest_v1",
                expected_manifest_file="manifest_confirmation.jsonl",
                expected_manifest_sha256=partition["manifest_sha256"],
                expected_query_rows=len(partition["manifest_rows"]),
            )
        except OrderedDecisionError as error:
            raise ConfirmationInputError(str(error)) from error
    try:
        query_bindings["all_null"] = validate_factorized_export_query_binding(
            all_null["summary"].get("sentence_memory_query_binding"),
            expected_authority="post_spend_environment_manifest_v1",
            expected_manifest_file="manifest_confirmation.jsonl",
            expected_manifest_sha256=partition["manifest_sha256"],
            expected_query_rows=len(partition["manifest_rows"]),
        )
    except OrderedDecisionError as error:
        raise ConfirmationInputError(str(error)) from error

    controls = {}
    for mode in expected_modes:
        if mode in {"off", "on", "full_replacement"}:
            continue
        controls[mode] = validate_centered_control_pair(
            exports["on"], exports[mode], mode=mode
        )
    controls["full_replacement"] = validate_centered_control_pair(
        exports["on"], exports["full_replacement"], mode="full_replacement"
    )
    controls["broadcast_complete"] = validate_centered_control_pair(
        exports["on"], broadcast, mode="broadcast_complete"
    )
    _validate_analytic_pair(exports["on"], exports["analytic_prior"])

    parity = {
        "joint_tuple_vs_correct": _prediction_parity(
            exports["joint_tuple_permuted"], exports["on"]
        ),
        "uniform_final_vs_off": _prediction_parity(
            exports["uniform_final_mass"], exports["off"], require_exact=True
        ),
        "broadcast_complete_vs_off": _prediction_parity(
            broadcast, exports["off"], require_exact=True
        ),
        "all_null_vs_off": _prediction_parity(
            all_null, exports["off"], require_exact=True
        ),
        "memory_off_vs_v2": _prediction_parity(exports["off"], v2),
    }
    parity["joint_tuple_vs_correct"]["passed"] = bool(
        parity["joint_tuple_vs_correct"]["prediction_max_abs"] <= PARITY_TOLERANCE
        and parity["joint_tuple_vs_correct"]["duration_max_abs"] <= PARITY_TOLERANCE
    )
    all_null_diagnostics = _validate_all_null_export(all_null)
    all_null_diagnostics["passed"] = bool(
        all_null_diagnostics["gate_max_abs"] == 0.0
        and all_null_diagnostics["candidate_mass_max_abs"] == 0.0
        and all_null_diagnostics["null_mass_max_abs_error"] == 0.0
    )
    all_null_diagnostics["requires_exact_mass_and_gate_equality"] = True
    duration_reference = exports["off"]["durations"]
    duration_max_abs = max(
        float(np.max(np.abs(export["durations"] - duration_reference), initial=0.0))
        for export in [*exports.values(), broadcast, all_null]
    )

    cluster_names, groups = _cluster_index_groups(partition["normalized_texts"])
    if len(cluster_names) != EXPECTED_CONFIRMATION_TEXTS:
        raise ConfirmationInputError("Confirmation unit count is not 540 texts")
    bootstrap = cluster_bootstrap_indices(
        len(cluster_names), samples=bootstrap_samples, seed=seed
    )
    cluster_metrics: dict[tuple[str, str, str, str], np.ndarray] = {}
    aggregate_metrics: dict[str, Any] = {}
    for alignment in ALIGNMENTS:
        aggregate_metrics[alignment] = {}
        for part in PARTS:
            aggregate_metrics[alignment][part] = {}
            for metric in ("dtw", "ndtw", "ndtw_ref"):
                aggregate_metrics[alignment][part][metric] = {}
                for mode, export in exports.items():
                    values = _cluster_mean(export["metrics"][alignment][part][metric], groups)
                    cluster_metrics[(alignment, part, metric, mode)] = values
                    aggregate_metrics[alignment][part][metric][mode] = {
                        "mean": float(values.mean()),
                        "ci95": confidence_interval(values[bootstrap].mean(axis=1)),
                    }

    pa = {
        mode: cluster_metrics[("pa", "wholebody", "ndtw", mode)]
        for mode in expected_modes
    }
    pair_mse = {
        mode: _cluster_mean(_row_motion_mse(exports["on"], exports[mode]), groups)
        for mode in PAIR_MODES
    }
    off_mse = _cluster_mean(_row_motion_mse(exports["on"], exports["off"]), groups)
    hands = {
        hand: {
            mode: _cluster_mean(
                exports[mode]["metrics"]["default"][hand]["motion_path_error"],
                groups,
            )
            for mode in expected_modes
        }
        for hand in ("lhand", "rhand")
    }
    aggregate_hands = {
        hand: {
            metric: {
                mode: {
                    "mean": float(values.mean()),
                    "ci95": confidence_interval(
                        values[bootstrap].mean(axis=1)
                    ),
                }
                for mode in expected_modes
                for values in [
                    _cluster_mean(
                        exports[mode]["metrics"]["default"][hand][metric],
                        groups,
                    )
                ]
            }
            for metric in ("motion_path_error", "jerk_magnitude_ratio")
        }
        for hand in ("lhand", "rhand")
    }
    scientific = compute_centered_confirmation_statistics(
        pa_ndtw=pa,
        pair_mse=pair_mse,
        off_mse=off_mse,
        hand_path=hands,
        bootstrap=bootstrap,
        seed=seed,
    )
    integrity = {
        "prediction_and_duration_parity": parity,
        "all_null_diagnostics": all_null_diagnostics,
        "duration_max_abs": duration_max_abs,
        "duration_tolerance": PARITY_TOLERANCE,
        "original_v2_checkpoint": _v2_checkpoint_evidence(
            v2_checkpoint_path.resolve()
        ),
        "stored_v2_parity": dict(authorization["checkpoint"]).get("parity"),
        "passed": bool(
            all(value["passed"] for value in parity.values())
            and all_null_diagnostics["passed"]
            and duration_max_abs <= PARITY_TOLERANCE
        ),
    }
    attention_exports = {
        mode: exports[mode]
        for mode in expected_modes
        if mode not in {"off", "uniform_final_mass"}
    }
    attention, _attention_values = _aggregate_attention_diagnostics(
        attention_exports, groups=groups, bootstrap=bootstrap
    )
    association_diagnostics = _association_matching_diagnostics(
        exports["on"], groups=groups, bootstrap=bootstrap
    )
    if stage == STAGE2 and association_diagnostics is None:
        raise ConfirmationInputError(
            "Stage B confirmation lacks association diagnostics"
        )
    if stage == STAGE1 and association_diagnostics is not None:
        raise ConfirmationInputError(
            "Stage A unexpectedly exposed association diagnostics"
        )
    finite = all(
        np.isfinite(value).all()
        for value in [*pa.values(), *pair_mse.values(), off_mse]
    )
    promotion_gates = {**scientific, "integrity": integrity, "finite": finite}
    promoted = bool(
        finite
        and integrity["passed"]
        and all(scientific[name]["passed"] for name in scientific)
    )

    development_ready = _json(development_diagnostic_dir / "READY")
    development_summary_path = development_diagnostic_dir / "summary.json"
    development_summary = _json(development_summary_path)
    if (
        development_ready.get("schema_name")
        != DIAGNOSTIC_READY_SCHEMA_NAME
        or development_summary.get("schema_name") != DIAGNOSTIC_SCHEMA_NAME
        or dict(development_summary.get("checkpoint", {}) or {}).get("sha256")
        != dict(authorization["checkpoint"])["sha256"]
        or development_ready.get("summary_sha256")
        != sha256_file(development_summary_path)
        or development_ready.get("diagnostic_identity")
        != development_summary.get("identity")
        or dict(authorization.get("development_diagnostic", {}) or {}).get(
            "diagnostic_identity"
        )
        != development_summary.get("identity")
    ):
        raise ConfirmationInputError("Development diagnostic is missing or detached")

    rows = []
    for index, text in enumerate(cluster_names):
        row = {
            "cluster_index": index,
            "normalized_text": text,
            "signer_rows": int(len(groups[index])),
            "off_mse": float(off_mse[index]),
        }
        for mode in expected_modes:
            row[f"pa_ndtw_wholebody_{mode}"] = float(pa[mode][index])
        for mode in PAIR_MODES:
            row[f"pair_mse_{mode}"] = float(pair_mse[mode][index])
        rows.append(row)

    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "scientific_split": "val_confirmation_novel_text_only",
        "test_data_accessed": False,
        "confirmation_holdout_spent": True,
        "stage2_after_confirmation_permitted": False,
        "authorization": authorization,
        "holdout_spend": spend,
        "partition_digest": EXPECTED_PARTITION_DIGEST,
        "confirmation_manifest_sha256": partition["manifest_sha256"],
        "confirmation_rows": len(partition["manifest_rows"]),
        "cluster_count": len(cluster_names),
        "statistical_unit": "normalized text; signer realizations averaged first",
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence": 0.95,
            "holm_family": list(HOLM_FAMILY),
        },
        "inputs": {
            mode: {
                "directory": str(export["dir"]),
                "export_summary_sha256": sha256_file(export["summary_path"]),
                "query_binding": query_bindings[mode],
            }
            for mode, export in {**exports, "broadcast_complete": broadcast}.items()
        },
        "all_null_query_binding": query_bindings["all_null"],
        "control_provenance": controls,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_hand_metrics": aggregate_hands,
        "attention_diagnostics_explanatory_only": attention,
        "association_matching_explanatory_only": association_diagnostics,
        "analytic_prior_promotion_comparator": False,
        "association_disabled_promotion_comparator": False,
        "development_layerwise_diagnostic": development_summary,
        "promotion_gates": promotion_gates,
        "promoted_centered_memory_checkpoint": promoted,
        "cluster_rows_identity": _digest_json(rows),
        "interpretation": (
            "A failure spends the holdout and forbids adaptive Stage B, threshold "
            "changes, test access, checkpoint promotion, and Phase B."
        ),
    }
    return summary, rows


def write_outputs(
    out_dir: Path,
    summary: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> None:
    out_dir = out_dir.resolve()
    summary_bytes = (
        json.dumps(
            summary, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    csv_bytes = _canonical_csv_bytes(rows)
    ready_payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "confirmation_holdout_spent": True,
        "promoted_centered_memory_checkpoint": bool(
            summary["promoted_centered_memory_checkpoint"]
        ),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "cluster_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "cluster_rows_identity": _digest_json(rows),
    }
    ready_payload["ready_identity"] = _digest_json(ready_payload)
    ready_bytes = (
        json.dumps(ready_payload, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if out_dir.exists():
        _validate_exact_completed_directory(
            out_dir,
            expected_files={
                "confirmation_summary.json",
                "confirmation_clusters.csv",
                "READY",
            },
        )
        ready = _json(out_dir / "READY")
        existing = _json(out_dir / "confirmation_summary.json")
        if (
            existing != dict(summary)
            or ready != ready_payload
            or (out_dir / "confirmation_summary.json").read_bytes()
            != summary_bytes
            or (out_dir / "confirmation_clusters.csv").read_bytes() != csv_bytes
            or (out_dir / "READY").read_bytes() != ready_bytes
        ):
            raise ConfirmationInputError("Existing confirmation output is detached")
        return
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    building = out_dir.with_name(f".{out_dir.name}.building.{attempt}")
    if building.exists():
        raise ConfirmationInputError(f"Incomplete analysis build exists: {building}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    # Preserve any interrupted build for forensic inspection; a restart may
    # reuse only the exact fully published destination above.
    _durable_write_bytes(building / "confirmation_summary.json", summary_bytes)
    _durable_write_bytes(building / "confirmation_clusters.csv", csv_bytes)
    _durable_write_bytes(building / "READY", ready_bytes)
    _fsync_directory(building)
    _rename_directory_no_replace(building, out_dir)
    _fsync_directory(out_dir.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--holdout_spend", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition_dir", type=Path, required=True)
    parser.add_argument("--development_diagnostic_dir", type=Path, required=True)
    parser.add_argument("--broadcast_complete_dir", type=Path, required=True)
    parser.add_argument("--all_null_dir", type=Path, required=True)
    parser.add_argument("--v2_text_only_dir", type=Path, required=True)
    parser.add_argument("--v2_checkpoint", type=Path, required=True)
    parser.add_argument("--v2_config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    for mode in STAGE2_EVAL_MODES:
        parser.add_argument(f"--{mode}_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = _json(args.authorization)
    stage = str(raw.get("stage", ""))
    expected_modes = STAGE1_EVAL_MODES if stage == STAGE1 else STAGE2_EVAL_MODES
    mode_dirs = {}
    for mode in expected_modes:
        value = getattr(args, f"{mode}_dir")
        if value is None:
            raise ConfirmationInputError(f"Missing --{mode}_dir")
        mode_dirs[mode] = value
    if stage == STAGE1 and args.association_disabled_dir is not None:
        raise ConfirmationInputError("Stage A cannot inspect association-disabled mode")
    summary, rows = analyze_confirmation(
        authorization_path=args.authorization,
        holdout_spend_path=args.holdout_spend,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        v2_checkpoint_path=args.v2_checkpoint,
        v2_config_path=args.v2_config,
        partition_dir=args.partition_dir,
        mode_dirs=mode_dirs,
        broadcast_dir=args.broadcast_complete_dir,
        all_null_dir=args.all_null_dir,
        v2_text_only_dir=args.v2_text_only_dir,
        development_diagnostic_dir=args.development_diagnostic_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_outputs(args.out_dir, summary, rows)
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "cluster_count": summary["cluster_count"],
                "promoted": summary["promoted_centered_memory_checkpoint"],
            },
            indent=2,
        )
    )
    if not summary["promoted_centered_memory_checkpoint"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
