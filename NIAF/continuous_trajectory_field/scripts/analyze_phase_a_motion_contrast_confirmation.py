from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.phase_a_motion_contrast import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    MAX_DURATION_ABS_DIFFERENCE_SECONDS,
    MAX_HAND_PATH_RELATIVE_DEGRADATION,
    MAX_INTEGRITY_OUTPUT_ABS_DIFFERENCE,
    MIN_RELATIVE_PA_NDTW_IMPROVEMENT,
    MIN_RMOTION_LOWER_CI,
    MIN_RMOTION_POINT,
    V2_TEXT_ONLY_CHECKPOINT_SHA256,
    cluster_bootstrap_indices,
    confidence_interval,
    holm_adjust,
    paired_lower_is_better,
    rmotion_summary,
    sign_flip_lower_tail_pvalue,
    stable_subseed,
)
from NIAF.continuous_trajectory_field.sentence_memory import (
    normalize_sentence_text,
    read_jsonl,
    sha256_file,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    sentence_memory_behavior_identity,
    sentence_memory_objective_identity,
    sentence_memory_resume_identity,
)
from NIAF.continuous_trajectory_field.validation_text_partitions import (
    CONFIRMATION,
    SCHEMA_NAME as PARTITION_SCHEMA_NAME,
)


SCHEMA_NAME = "signtrajfield_phase_a_motion_contrast_confirmation"
SCHEMA_VERSION = 1
EXPECTED_CONFIRMATION_TEXTS = 540
EXPECTED_NOVEL_TEXTS = 796
EXPECTED_DEVELOPMENT_ROWS = 347
EXPECTED_EXPERIMENT_NAME = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1"
)
PARTS = ("body", "lhand", "rhand", "face", "wholebody")
ALIGNMENTS = ("default", "pa")
MODE_SENTENCE_MEMORY = {
    "text_only": "off",
    "sentence_memory": "on",
    "shuffled_sentence_memory": "shuffled",
    "motion_shuffled_sentence_memory": "motion_shuffled",
}
DTW_FILE = {
    "default": "dtw_mpjpe_t2m_default_h2s_betas.json",
    "pa": "dtw_mpjpe_t2m_pa_h2s_betas.json",
}


class ConfirmationInputError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-shot, validation-only Phase-A motion-contrast "
            "confirmation analysis. Test inputs are rejected."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition_dir", type=Path, required=True)
    parser.add_argument("--text_only_dir", type=Path, required=True)
    parser.add_argument("--sentence_memory_dir", type=Path, required=True)
    parser.add_argument("--shuffled_sentence_memory_dir", type=Path, required=True)
    parser.add_argument(
        "--motion_shuffled_sentence_memory_dir", type=Path, required=True
    )
    parser.add_argument("--all_null_dir", type=Path, required=True)
    parser.add_argument("--v2_text_only_dir", type=Path, required=True)
    parser.add_argument("--v2_checkpoint", type=Path, required=True)
    parser.add_argument("--v2_config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


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
        raise ConfirmationInputError(f"Expected a JSON object: {path}")
    return value


def _resolve_existing(path_value: Any, *, fallback_dir: Path) -> Path:
    path = Path(str(path_value))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((Path.cwd() / path, fallback_dir / path.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfirmationInputError(f"Referenced file does not exist: {path_value}")


def load_confirmation_partition(partition_dir: Path) -> dict[str, Any]:
    partition_dir = partition_dir.resolve()
    payload = _json(partition_dir / "partition.json")
    ready = _json(partition_dir / "READY")
    if payload.get("schema_name") != PARTITION_SCHEMA_NAME:
        raise ConfirmationInputError("Unexpected validation-partition schema")
    if payload.get("split") != "val" or not payload.get("validation_only"):
        raise ConfirmationInputError("Partition is not explicitly validation-only")
    if ready.get("artifact_identity") != payload.get("artifact_identity"):
        raise ConfirmationInputError("Partition READY identity mismatch")
    identity_payload = {
        key: value for key, value in payload.items() if key != "artifact_identity"
    }
    if _digest_json(identity_payload) != payload.get("artifact_identity"):
        raise ConfirmationInputError("Partition artifact identity is invalid")
    counts = dict(payload.get("counts", {}) or {})
    if int(counts.get("novel_unique_texts", -1)) != EXPECTED_NOVEL_TEXTS:
        raise ConfirmationInputError("Partition does not contain 796 novel texts")
    if int(counts.get("confirmation_unique_texts", -1)) != EXPECTED_CONFIRMATION_TEXTS:
        raise ConfirmationInputError("Partition does not contain 540 confirmation texts")
    artifact = dict(payload.get("manifest_artifacts", {}) or {}).get(CONFIRMATION)
    if not isinstance(artifact, dict):
        raise ConfirmationInputError("Partition has no confirmation manifest identity")
    manifest_path = partition_dir / str(artifact.get("file", ""))
    if sha256_file(manifest_path) != artifact.get("sha256"):
        raise ConfirmationInputError("Confirmation-manifest hash mismatch")
    manifest_rows = read_jsonl(manifest_path)
    if len(manifest_rows) != int(artifact.get("row_count", -1)):
        raise ConfirmationInputError("Confirmation-manifest row count mismatch")
    assignments = {
        str(row["normalized_text"]): str(row["label"])
        for row in payload.get("assignments", [])
    }
    normalized = [normalize_sentence_text(row.get("text", "")) for row in manifest_rows]
    if any(assignments.get(text) != CONFIRMATION for text in normalized):
        raise ConfirmationInputError(
            "Confirmation manifest contains development, seen, or unknown text"
        )
    if len(set(normalized)) != EXPECTED_CONFIRMATION_TEXTS:
        raise ConfirmationInputError(
            "Confirmation manifest does not cover all 540 unique text clusters"
        )
    return {
        "dir": partition_dir,
        "payload": payload,
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": str(artifact["sha256"]),
        "manifest_rows": manifest_rows,
        "normalized_texts": normalized,
    }


def _expected_row_identity(rows: list[Mapping[str, Any]]) -> list[tuple[str, str]]:
    return [
        (str(row.get("name", "")), normalize_sentence_text(row.get("text", "")))
        for row in rows
    ]


def load_mode_export(
    mode: str,
    directory: Path,
    *,
    partition: Mapping[str, Any],
    checkpoint_path: Path,
    config_path: Path | None = None,
    checkpoint_epoch: int | None = None,
    motion_shuffle_seed: int | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    summary_path = directory / "export_summary.json"
    summary = _json(summary_path)
    expected_memory_mode = MODE_SENTENCE_MEMORY[mode]
    if str(summary.get("split")) != "val":
        raise ConfirmationInputError(f"{mode} export is not validation-only")
    if str(summary.get("length_mode")) != "predicted":
        raise ConfirmationInputError(f"{mode} did not use predicted-duration evaluation")
    if str(summary.get("word_prior_mode")) != "off":
        raise ConfirmationInputError(f"{mode} did not keep the word prior off")
    if str(summary.get("sentence_memory_mode")) != expected_memory_mode:
        raise ConfirmationInputError(
            f"{mode} has sentence-memory mode {summary.get('sentence_memory_mode')!r}"
        )
    if config_path is not None and _resolve_existing(
        summary.get("config"), fallback_dir=directory
    ) != config_path.resolve():
        raise ConfirmationInputError(f"{mode} used a different inference config")
    exported_checkpoint = _resolve_existing(
        summary.get("checkpoint"), fallback_dir=directory
    )
    if exported_checkpoint != checkpoint_path:
        raise ConfirmationInputError(f"{mode} used a different checkpoint")

    manifest = dict(summary.get("manifest", {}) or {})
    if int(manifest.get("sample_count", -1)) != len(partition["manifest_rows"]):
        raise ConfirmationInputError(f"{mode} export has an incomplete confirmation set")
    output_manifest = _resolve_existing(
        manifest.get("output_manifest"), fallback_dir=directory
    )
    if sha256_file(output_manifest) != partition["manifest_sha256"]:
        raise ConfirmationInputError(f"{mode} export used the wrong validation rows")
    rows = list(summary.get("rows", []))
    if len(rows) != len(partition["manifest_rows"]):
        raise ConfirmationInputError(f"{mode} summary row count is incomplete")
    if _expected_row_identity(rows) != _expected_row_identity(
        partition["manifest_rows"]
    ):
        raise ConfirmationInputError(f"{mode} export row order/identity mismatch")
    durations = np.asarray(
        [row.get("predicted_duration_seconds") for row in rows], dtype=np.float64
    )
    if not np.isfinite(durations).all():
        raise ConfirmationInputError(f"{mode} has non-finite predicted durations")
    for row in rows:
        subset = str(row.get("sentence_memory_text_subset", ""))
        if mode != "text_only" and subset != "novel_text":
            raise ConfirmationInputError(
                f"{mode} contains a non-novel sentence-memory query"
            )

    metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    expected_indices = {f"{index:04d}" for index in range(len(rows))}
    for alignment in ALIGNMENTS:
        dtw_path = directory / DTW_FILE[alignment]
        payload = _json(dtw_path)
        if str(payload.get("alignment_mode")) != alignment:
            raise ConfirmationInputError(f"{mode}/{alignment} alignment mismatch")
        if int(payload.get("num_pairs", -1)) != len(rows):
            raise ConfirmationInputError(f"{mode}/{alignment} has incomplete pairs")
        by_part: dict[str, dict[str, np.ndarray]] = {}
        for part in PARTS:
            selected = [
                row
                for row in payload.get("rows", [])
                if row.get("comparison") == "flow" and row.get("part") == part
            ]
            if {str(row.get("index")) for row in selected} != expected_indices:
                raise ConfirmationInputError(
                    f"{mode}/{alignment}/{part} row identity mismatch"
                )
            selected.sort(key=lambda row: int(row["index"]))
            fields = ("dtw", "ndtw", "ndtw_ref")
            values = {
                field: np.asarray([row[field] for row in selected], dtype=np.float64)
                for field in fields
            }
            if part in {"lhand", "rhand"} and alignment == "default":
                for field in ("motion_path_error", "jerk_magnitude_ratio"):
                    if not all(field in row for row in selected):
                        raise ConfirmationInputError(
                            f"{mode}/{part} is missing required {field} diagnostics"
                        )
                    values[field] = np.asarray(
                        [row[field] for row in selected], dtype=np.float64
                    )
            if not all(np.isfinite(value).all() for value in values.values()):
                raise ConfirmationInputError(
                    f"{mode}/{alignment}/{part} contains non-finite metrics"
                )
            by_part[part] = values
        metrics[alignment] = by_part

    if mode == "motion_shuffled_sentence_memory":
        if checkpoint_epoch is None or motion_shuffle_seed is None:
            raise ConfirmationInputError(
                "Motion-shuffled validation requires its checkpoint-epoch/seed lock"
            )
        if int(summary.get("sentence_memory_motion_shuffle_epoch", -1)) != int(
            checkpoint_epoch
        ):
            raise ConfirmationInputError(
                "Motion-shuffled export did not use the selected checkpoint epoch"
            )
        if int(summary.get("sentence_memory_motion_shuffle_seed", -1)) != int(
            motion_shuffle_seed
        ):
            raise ConfirmationInputError(
                "Motion-shuffled export did not use the locked global seed"
            )
        for row in rows:
            sample_path = _resolve_existing(row["sample"], fallback_dir=directory)
            with np.load(sample_path, allow_pickle=False) as sample:
                if "sentence_memory_motion_shuffle_informative" not in sample.files:
                    raise ConfirmationInputError(
                        "Motion-shuffled export lacks corruption provenance"
                    )
                if not bool(sample["sentence_memory_motion_shuffle_informative"]):
                    raise ConfirmationInputError(
                        "Motion-shuffled confirmation row was not informative"
                    )
                for field, expected in (
                    ("sentence_memory_motion_shuffle_epoch", checkpoint_epoch),
                    ("sentence_memory_motion_shuffle_seed", motion_shuffle_seed),
                ):
                    if field not in sample.files:
                        raise ConfirmationInputError(
                            f"Motion-shuffled export lacks {field} provenance"
                        )
                    actual = np.asarray(sample[field]).reshape(-1)
                    if len(actual) != 1 or int(actual[0]) != int(expected):
                        raise ConfirmationInputError(
                            f"Motion-shuffled export has the wrong {field}"
                        )

    return {
        "dir": directory,
        "summary_path": summary_path.resolve(),
        "summary": summary,
        "rows": rows,
        "durations": durations,
        "metrics": metrics,
    }


def load_integrity_export(
    label: str,
    directory: Path,
    *,
    partition: Mapping[str, Any],
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Load a prediction-only integrity control over the locked val rows."""

    expected = {
        "all_null": (
            "all_null",
            "sentence_memory_continuous_trajectory_field",
        ),
        "v2_text_only": (
            "not_applicable",
            "dual_mode_continuous_trajectory_field",
        ),
    }
    if label not in expected:
        raise ValueError(f"Unknown integrity export: {label}")
    expected_memory_mode, expected_model_type = expected[label]
    directory = directory.resolve()
    summary_path = directory / "export_summary.json"
    summary = _json(summary_path)
    if str(summary.get("split")) != "val":
        raise ConfirmationInputError(f"{label} export is not validation-only")
    if str(summary.get("length_mode")) != "predicted":
        raise ConfirmationInputError(f"{label} did not use predicted duration")
    if str(summary.get("word_prior_mode")) != "off":
        raise ConfirmationInputError(f"{label} did not keep the word prior off")
    if str(summary.get("sentence_memory_mode")) != expected_memory_mode:
        raise ConfirmationInputError(f"{label} has the wrong sentence-memory mode")
    if str(summary.get("model_type")) != expected_model_type:
        raise ConfirmationInputError(f"{label} has the wrong model contract")
    if config_path is not None and _resolve_existing(
        summary.get("config"), fallback_dir=directory
    ) != config_path.resolve():
        raise ConfirmationInputError(f"{label} used a different inference config")
    exported_checkpoint = _resolve_existing(
        summary.get("checkpoint"), fallback_dir=directory
    )
    if exported_checkpoint != checkpoint_path:
        raise ConfirmationInputError(f"{label} used a different checkpoint")

    manifest = dict(summary.get("manifest", {}) or {})
    if int(manifest.get("sample_count", -1)) != len(partition["manifest_rows"]):
        raise ConfirmationInputError(f"{label} export is incomplete")
    output_manifest = _resolve_existing(
        manifest.get("output_manifest"), fallback_dir=directory
    )
    if sha256_file(output_manifest) != partition["manifest_sha256"]:
        raise ConfirmationInputError(f"{label} export used the wrong validation rows")
    rows = list(summary.get("rows", []))
    if len(rows) != len(partition["manifest_rows"]):
        raise ConfirmationInputError(f"{label} export row count is incomplete")
    if _expected_row_identity(rows) != _expected_row_identity(
        partition["manifest_rows"]
    ):
        raise ConfirmationInputError(f"{label} export row order/identity mismatch")
    durations = np.asarray(
        [row.get("predicted_duration_seconds") for row in rows], dtype=np.float64
    )
    if not np.isfinite(durations).all():
        raise ConfirmationInputError(f"{label} has non-finite predicted durations")
    for row in rows:
        if label == "all_null" and str(
            row.get("sentence_memory_text_subset", "")
        ) != "novel_text":
            raise ConfirmationInputError("All-null export contains a non-novel query")
        _resolve_existing(row.get("sample"), fallback_dir=directory)
    return {
        "dir": directory,
        "summary_path": summary_path.resolve(),
        "summary": summary,
        "rows": rows,
        "durations": durations,
    }


def _load_rot6d(row: Mapping[str, Any], directory: Path) -> np.ndarray:
    path = _resolve_existing(row.get("sample"), fallback_dir=directory)
    with np.load(path, allow_pickle=False) as sample:
        if "rot6d" not in sample.files:
            raise ConfirmationInputError(f"Export sample has no rot6d payload: {path}")
        values = np.asarray(sample["rot6d"], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ConfirmationInputError(f"Invalid rot6d payload: {path}")
    return values


def _prediction_parity(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    require_exact: bool = False,
) -> dict[str, Any]:
    """Measure strict paired pose/duration parity for two locked exports."""

    if len(first["rows"]) != len(second["rows"]):
        raise ConfirmationInputError("Integrity exports have different row counts")
    output_max_abs = 0.0
    prediction_arrays_equal = True
    for row_a, row_b in zip(first["rows"], second["rows"]):
        if int(row_a.get("sample_lengths", {}).get("fps20", -1)) != int(
            row_b.get("sample_lengths", {}).get("fps20", -1)
        ):
            raise ConfirmationInputError(
                "Integrity exports have different predicted output lengths"
            )
        motion_a = _load_rot6d(row_a, first["dir"])
        motion_b = _load_rot6d(row_b, second["dir"])
        if motion_a.shape != motion_b.shape:
            raise ConfirmationInputError(
                "Integrity exports have different predicted trajectory shapes"
            )
        prediction_arrays_equal = bool(
            prediction_arrays_equal and np.array_equal(motion_a, motion_b)
        )
        if motion_a.size:
            output_max_abs = max(
                output_max_abs, float(np.max(np.abs(motion_a - motion_b)))
            )
    duration_max_abs = float(
        np.max(np.abs(first["durations"] - second["durations"]), initial=0.0)
    )
    duration_values_equal = bool(
        np.array_equal(first["durations"], second["durations"])
    )
    prediction_tolerance = (
        0.0 if require_exact else MAX_INTEGRITY_OUTPUT_ABS_DIFFERENCE
    )
    duration_tolerance = (
        0.0 if require_exact else MAX_DURATION_ABS_DIFFERENCE_SECONDS
    )
    return {
        "prediction_max_abs": output_max_abs,
        "duration_max_abs": duration_max_abs,
        "prediction_arrays_equal": prediction_arrays_equal,
        "duration_values_equal": duration_values_equal,
        "requires_bitwise_array_equality": bool(require_exact),
        "prediction_tolerance": prediction_tolerance,
        "duration_tolerance": duration_tolerance,
        "passed": bool(
            (
                prediction_arrays_equal
                and duration_values_equal
                if require_exact
                else output_max_abs <= prediction_tolerance
                and duration_max_abs <= duration_tolerance
            )
        ),
    }


def _validate_all_null_export(export: Mapping[str, Any]) -> dict[str, Any]:
    """Prove all-null conditioning has null mass one and a zero residual gate."""

    gate_max_abs = 0.0
    candidate_mass_max_abs = 0.0
    null_mass_max_abs_error = 0.0
    for row in export["rows"]:
        sample_path = _resolve_existing(row["sample"], fallback_dir=export["dir"])
        with np.load(sample_path, allow_pickle=False) as sample:
            required = (
                "sentence_memory_ids",
                "sentence_memory_candidate_mask",
                "sentence_memory_payload_reads",
                "trajectory_sentence_memory_available",
                "trajectory_sentence_memory_gates",
                "trajectory_sentence_memory_null_mass",
                "trajectory_sentence_memory_candidate_mass",
            )
            missing = [field for field in required if field not in sample.files]
            if missing:
                raise ConfirmationInputError(
                    f"All-null export lacks integrity fields: {missing}"
                )
            if np.asarray(sample["sentence_memory_candidate_mask"]).any():
                raise ConfirmationInputError("All-null candidate mask is not empty")
            if not np.all(np.asarray(sample["sentence_memory_ids"]) == -1):
                raise ConfirmationInputError("All-null candidate IDs are not -1")
            payload_reads = np.asarray(
                sample["sentence_memory_payload_reads"]
            ).reshape(-1)
            if len(payload_reads) != 1 or int(payload_reads[0]) != 0:
                raise ConfirmationInputError("All-null export performed payload reads")
            if np.asarray(sample["trajectory_sentence_memory_available"]).any():
                raise ConfirmationInputError(
                    "All-null candidates unexpectedly remained available"
                )
            gates = np.asarray(
                sample["trajectory_sentence_memory_gates"], dtype=np.float64
            )
            null_mass = np.asarray(
                sample["trajectory_sentence_memory_null_mass"], dtype=np.float64
            )
            candidate_mass = np.asarray(
                sample["trajectory_sentence_memory_candidate_mass"], dtype=np.float64
            )
            if not all(
                np.isfinite(value).all()
                for value in (gates, null_mass, candidate_mass)
            ):
                raise ConfirmationInputError("All-null diagnostics are non-finite")
            gate_max_abs = max(gate_max_abs, float(np.max(np.abs(gates), initial=0.0)))
            candidate_mass_max_abs = max(
                candidate_mass_max_abs,
                float(np.max(np.abs(candidate_mass), initial=0.0)),
            )
            null_mass_max_abs_error = max(
                null_mass_max_abs_error,
                float(np.max(np.abs(null_mass - 1.0), initial=0.0)),
            )
    tolerance = MAX_INTEGRITY_OUTPUT_ABS_DIFFERENCE
    return {
        "payload_reads": 0,
        "gate_max_abs": gate_max_abs,
        "candidate_mass_max_abs": candidate_mass_max_abs,
        "null_mass_max_abs_error": null_mass_max_abs_error,
        "tolerance": tolerance,
        "passed": bool(
            gate_max_abs <= tolerance
            and candidate_mass_max_abs <= tolerance
            and null_mass_max_abs_error <= tolerance
        ),
    }


def _validate_motion_shuffle_pair(
    correct: Mapping[str, Any], motion_shuffled: Mapping[str, Any]
) -> None:
    metadata_fields = (
        "sentence_memory_ids",
        "sentence_memory_scores",
        "sentence_memory_durations",
        "sentence_memory_duration_log_gap",
        "sentence_memory_candidate_mask",
    )
    for correct_row, shuffled_row in zip(correct["rows"], motion_shuffled["rows"]):
        correct_path = _resolve_existing(correct_row["sample"], fallback_dir=correct["dir"])
        shuffled_path = _resolve_existing(
            shuffled_row["sample"], fallback_dir=motion_shuffled["dir"]
        )
        with np.load(correct_path, allow_pickle=False) as correct_sample, np.load(
            shuffled_path, allow_pickle=False
        ) as shuffled_sample:
            for field in metadata_fields:
                if field not in correct_sample.files or field not in shuffled_sample.files:
                    raise ConfirmationInputError(
                        f"Motion-control pair is missing candidate metadata {field}"
                    )
                if not np.array_equal(correct_sample[field], shuffled_sample[field]):
                    raise ConfirmationInputError(
                        f"Motion-only shuffle changed candidate metadata {field}"
                    )
            required = (
                "sentence_memory_motion_candidate_permutation",
                "sentence_memory_motion_source_ids",
                "sentence_memory_motion_shuffle_informative",
            )
            if any(field not in shuffled_sample.files for field in required):
                raise ConfirmationInputError(
                    "Motion-shuffled export lacks permutation/source provenance"
                )
            informative = bool(
                np.asarray(
                    shuffled_sample["sentence_memory_motion_shuffle_informative"]
                ).reshape(-1)[0]
            )
            if not informative:
                raise ConfirmationInputError(
                    "Motion-shuffled confirmation row was not informative"
                )
            ids = np.asarray(correct_sample["sentence_memory_ids"], dtype=np.int64)
            mask = np.asarray(
                correct_sample["sentence_memory_candidate_mask"], dtype=np.bool_
            )
            permutation = np.asarray(
                shuffled_sample["sentence_memory_motion_candidate_permutation"],
                dtype=np.int64,
            )
            source_ids = np.asarray(
                shuffled_sample["sentence_memory_motion_source_ids"], dtype=np.int64
            )
            if permutation.shape != ids.shape or source_ids.shape != ids.shape:
                raise ConfirmationInputError("Motion-shuffle provenance has wrong shape")
            if (permutation < 0).any() or (permutation >= len(ids)).any():
                raise ConfirmationInputError("Motion-shuffle permutation is out of bounds")
            valid = np.flatnonzero(mask & (ids >= 0))
            if len(valid) < 2 or np.any(permutation[valid] == valid):
                raise ConfirmationInputError(
                    "Motion-shuffle permutation did not derange all valid candidates"
                )
            if not np.array_equal(source_ids, ids[permutation]):
                raise ConfirmationInputError(
                    "Motion-shuffle source IDs do not match the recorded permutation"
                )


def _row_motion_mse(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> np.ndarray:
    values = []
    for row_a, row_b in zip(first["rows"], second["rows"]):
        motion_a = _load_rot6d(row_a, first["dir"])
        motion_b = _load_rot6d(row_b, second["dir"])
        if motion_a.shape != motion_b.shape:
            raise ConfirmationInputError(
                "Paired exports have different predicted trajectory shapes"
            )
        values.append(float(np.square(motion_a - motion_b).mean()))
    return np.asarray(values, dtype=np.float64)


def _cluster_index_groups(normalized_rows: list[str]) -> tuple[list[str], list[np.ndarray]]:
    cluster_names = sorted(set(normalized_rows))
    groups = [
        np.flatnonzero(np.asarray(normalized_rows, dtype=object) == name)
        for name in cluster_names
    ]
    return cluster_names, groups


def _cluster_mean(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.asarray([values[group].mean() for group in groups], dtype=np.float64)


_DYNAMIC_PARTITION_FIELDS = (
    "partition_digest",
    "resolved_artifact",
    "development_row_count",
    "exact_seen_row_count",
    "exact_seen_text_count",
    "exact_seen_evaluated_during_training",
    "confirmation_evaluated_during_training",
)


def _static_phase_a_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only attested runtime fields before exact config comparison."""

    payload = copy.deepcopy(dict(cfg))
    memory_cfg = payload.get("sentence_memory")
    if isinstance(memory_cfg, dict):
        memory_cfg.pop("resolved_identity", None)
        memory_cfg.pop("resolved_behavior_identity", None)
    safety_cfg = payload.get("sentence_memory_safety")
    if isinstance(safety_cfg, dict):
        safety_cfg.pop("v2_to_v3_text_only_parity", None)
    partition_cfg = payload.get("validation_text_partition")
    if isinstance(partition_cfg, dict):
        for field in _DYNAMIC_PARTITION_FIELDS:
            partition_cfg.pop(field, None)
    out_dir = str(dict(payload.get("output", {}) or {}).get("out_dir", ""))
    if Path(out_dir).name != EXPECTED_EXPERIMENT_NAME:
        raise ConfirmationInputError(
            "Checkpoint/config output directory is not the full Phase-A' run"
        )
    payload.setdefault("output", {})["out_dir"] = EXPECTED_EXPERIMENT_NAME
    return payload


def _validate_phase_a_prime_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_cfg: Mapping[str, Any],
    partition_digest: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(checkpoint.get("config", {}) or {}))
    if cfg.get("experiment_name") != EXPECTED_EXPERIMENT_NAME:
        raise ConfirmationInputError(
            "Checkpoint is not the exact Phase-A motion-contrast experiment"
        )
    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 1 or epoch > 4:
        raise ConfirmationInputError(
            "Confirmation requires a full-run checkpoint from epoch 1 through 4"
        )
    if str(cfg.get("device")) != "cuda" or str(
        cfg.get("text", {}).get("device")
    ) != "cpu":
        raise ConfirmationInputError(
            "Confirmation requires GPU trajectory/CPU mT5 training placement"
        )

    train_cfg = dict(cfg.get("train", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    eval_cfg = dict(cfg.get("eval", {}) or {})
    selection_cfg = dict(cfg.get("selection", {}) or {})
    objective_cfg = dict(cfg.get("objective", {}) or {})
    memory_cfg = dict(cfg.get("sentence_memory", {}) or {})
    if int(train_cfg.get("epochs", -1)) != 4:
        raise ConfirmationInputError("Confirmation requires the four-epoch full config")
    if int(train_cfg.get("early_stopping_patience", -1)) != 2 or int(
        train_cfg.get("early_stopping_min_epochs", -1)
    ) != 2:
        raise ConfirmationInputError("Checkpoint has the wrong early-stop contract")
    if any(
        int(value or 0) != 0
        for value in (
            data_cfg.get("limit_train", 0),
            data_cfg.get("limit_val", 0),
            train_cfg.get("max_train_batches", 0),
            eval_cfg.get("max_batches", 0),
        )
    ):
        raise ConfirmationInputError(
            "Smoke, subset, or max-batch checkpoints cannot enter confirmation"
        )
    if not bool(selection_cfg.get("require_feasible", False)):
        raise ConfirmationInputError("Checkpoint did not require feasible selection")
    if not bool(train_cfg.get("freeze_base", False)) or train_cfg.get(
        "unfreeze_base_prefixes"
    ):
        raise ConfirmationInputError("Checkpoint is not frozen-base Phase A")
    expected_scalars = {
        "k": 8,
        "top_m": 64,
        "duration_weight": 0.05,
        "score_temperature": 0.10,
        "candidate_dropout_probability": 0.10,
    }
    for field, expected in expected_scalars.items():
        actual = memory_cfg.get(field)
        if actual is None or float(actual) != float(expected):
            raise ConfirmationInputError(
                f"Checkpoint has the wrong sentence_memory.{field}"
            )
    if int(cfg.get("seed", -1)) != 1234:
        raise ConfirmationInputError("Checkpoint has the wrong global seed")
    if float(objective_cfg.get("lambda_sentence_sparsity", math.nan)) != 1e-4:
        raise ConfirmationInputError("Checkpoint has the wrong sparsity objective")

    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    if partition_cfg.get("partition_digest") != partition_digest:
        raise ConfirmationInputError(
            "Checkpoint is not bound to the locked validation-text partition"
        )
    if int(partition_cfg.get("development_row_count", -1)) != (
        EXPECTED_DEVELOPMENT_ROWS
    ):
        raise ConfirmationInputError(
            "Checkpoint does not attest the complete 347-row development set"
        )
    if partition_cfg.get("confirmation_evaluated_during_training") is not False:
        raise ConfirmationInputError(
            "Checkpoint does not attest that confirmation stayed locked"
        )
    resolved_partition = partition_cfg.get("resolved_artifact")
    if not isinstance(resolved_partition, dict):
        raise ConfirmationInputError("Checkpoint lacks resolved partition provenance")
    resolved_without_digest = {
        key: value
        for key, value in resolved_partition.items()
        if key != "partition_digest"
    }
    if _digest_json(resolved_without_digest) != resolved_partition.get(
        "partition_digest"
    ):
        raise ConfirmationInputError("Checkpoint partition provenance digest is invalid")
    if resolved_partition.get("partition_digest") != partition_digest:
        raise ConfirmationInputError(
            "Checkpoint resolved partition does not match the locked digest"
        )
    counts = dict(resolved_partition.get("counts", {}) or {})
    expected_counts = {
        "rows": 1077,
        "novel_unique_texts": EXPECTED_NOVEL_TEXTS,
        "development_unique_texts": 256,
        "confirmation_unique_texts": EXPECTED_CONFIRMATION_TEXTS,
        "development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "confirmation_rows": 728,
    }
    if any(int(counts.get(name, -1)) != value for name, value in expected_counts.items()):
        raise ConfirmationInputError("Checkpoint partition counts are incomplete")

    expected_runtime_cfg = copy.deepcopy(dict(expected_cfg))
    expected_runtime_cfg["device"] = "cuda"
    expected_partition_cfg = expected_runtime_cfg.setdefault(
        "validation_text_partition", {}
    )
    for field in _DYNAMIC_PARTITION_FIELDS:
        if field in partition_cfg:
            expected_partition_cfg[field] = copy.deepcopy(partition_cfg[field])
    if _static_phase_a_config(cfg) != _static_phase_a_config(expected_runtime_cfg):
        raise ConfirmationInputError(
            "Checkpoint config differs from the exact full Phase-A' config"
        )

    actual_objective = checkpoint.get("sentence_memory_objective_identity")
    recomputed_objective = sentence_memory_objective_identity(cfg)
    expected_objective = sentence_memory_objective_identity(expected_runtime_cfg)
    if actual_objective != recomputed_objective or actual_objective != expected_objective:
        raise ConfirmationInputError(
            "Checkpoint objective identity is missing, invalid, or unexpected"
        )
    actual_behavior = checkpoint.get("sentence_memory_behavior_identity")
    recomputed_behavior = sentence_memory_behavior_identity(cfg)
    expected_behavior = sentence_memory_behavior_identity(expected_runtime_cfg)
    if actual_behavior != recomputed_behavior or actual_behavior != expected_behavior:
        raise ConfirmationInputError(
            "Checkpoint sentence-memory behavior identity is invalid"
        )
    actual_resume = checkpoint.get("sentence_memory_resume_identity")
    recomputed_resume = sentence_memory_resume_identity(cfg)
    expected_resume = sentence_memory_resume_identity(expected_runtime_cfg)
    if actual_resume != recomputed_resume or actual_resume != expected_resume:
        raise ConfirmationInputError(
            "Checkpoint exact-resume identity is missing, invalid, or unexpected"
        )

    metrics = dict(checkpoint.get("metrics", {}) or {})
    required_namespaces = (
        "val_text_only/",
        "val_sentence_memory/",
        "val_shuffled_sentence_memory/",
        "val_motion_shuffled_sentence_memory/",
    )
    if any(not any(key.startswith(prefix) for key in metrics) for prefix in required_namespaces):
        raise ConfirmationInputError("Checkpoint lacks all four development modes")
    selection_state = dict(checkpoint.get("selection_state", {}) or {})
    if selection_state.get("best_feasible_score") is None:
        raise ConfirmationInputError("Checkpoint lacks feasible selection history")
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict) or int(rng_state.get("world_size", -1)) != 4:
        raise ConfirmationInputError(
            "Confirmation requires a checkpoint produced by the four-rank full run"
        )
    rank_states = list(rng_state.get("rank_states", []))
    if sorted(
        int(row.get("rank", -1)) for row in rank_states if isinstance(row, dict)
    ) != [0, 1, 2, 3]:
        raise ConfirmationInputError(
            "Checkpoint lacks exact RNG state for all four training ranks"
        )
    return {
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "eligible_checkpoint_epoch_range": [1, 4],
        "full_train_and_validation": True,
        "training_world_size": 4,
        "development_row_count": EXPECTED_DEVELOPMENT_ROWS,
        "confirmation_evaluated_during_training": False,
        "objective_identity": actual_objective,
        "behavior_identity": actual_behavior,
        "resume_identity_digest": actual_resume["digest"],
    }


def _checkpoint_evidence(
    checkpoint_path: Path,
    partition_digest: str,
    config_path: Path,
) -> dict[str, Any]:
    if checkpoint_path.name == "best_infeasible.pt":
        raise ConfirmationInputError("best_infeasible.pt cannot be confirmed/promoted")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ConfirmationInputError("Checkpoint payload is not a mapping")
    cfg = dict(checkpoint.get("config", {}) or {})
    config_path = config_path.resolve()
    if (
        not config_path.is_file()
        or config_path.name != f"{EXPECTED_EXPERIMENT_NAME}.yaml"
    ):
        raise ConfirmationInputError("Expected full Phase-A' config is unavailable")
    expected_cfg = load_config(config_path)
    phase_a_prime_contract = _validate_phase_a_prime_checkpoint_contract(
        checkpoint,
        expected_cfg=expected_cfg,
        partition_digest=partition_digest,
    )
    if cfg.get("model", {}).get("type") != "sentence_memory_continuous_trajectory_field":
        raise ConfirmationInputError("Checkpoint is not a v3 sentence-memory model")
    if not bool(cfg.get("train", {}).get("freeze_base", False)):
        raise ConfirmationInputError("Confirmation requires a frozen-base Phase-A model")
    if cfg.get("train", {}).get("unfreeze_base_prefixes"):
        raise ConfirmationInputError("Confirmation rejects partially unfrozen models")
    if bool(cfg.get("sentence_memory_safety", {}).get("phase_b", {}).get("enabled")):
        raise ConfirmationInputError("Confirmation rejects Phase-B checkpoints")
    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    if partition_cfg.get("partition_digest") != partition_digest:
        raise ConfirmationInputError(
            "Checkpoint is not bound to the locked validation-text partition"
        )
    metrics = dict(checkpoint.get("metrics", {}) or {})
    if float(metrics.get("selection_feasible", 0.0)) != 1.0:
        raise ConfirmationInputError("Checkpoint did not pass development selection")
    parity = dict(checkpoint.get("v2_to_v3_text_only_parity", {}) or {})
    if (
        not parity.get("passed")
        or float(parity.get("prediction_max_abs", math.inf)) > 1e-7
        or float(parity.get("duration_max_abs", math.inf)) > 1e-7
    ):
        raise ConfirmationInputError("Checkpoint lacks strict v2 text-only parity")
    neighbor_tables = dict(
        dict(checkpoint.get("sentence_memory_identity", {}) or {}).get(
            "neighbor_tables", {}
        )
        or {}
    )
    if "test" in neighbor_tables:
        raise ConfirmationInputError(
            "Checkpoint identity contains a test neighbor table; no-test confirmation "
            "requires train/val-only staging"
        )
    if not {"train", "val"}.issubset(neighbor_tables):
        raise ConfirmationInputError("Checkpoint lacks train/val neighbor identities")
    return {
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "development_selection_score": float(metrics["selection_score"]),
        "development_selection_feasible": True,
        "v2_to_v3_text_only_parity": parity,
        "sentence_memory_identity": checkpoint.get("sentence_memory_identity"),
        "sentence_memory_behavior_identity": checkpoint.get(
            "sentence_memory_behavior_identity"
        ),
        "sentence_memory_objective_identity": checkpoint.get(
            "sentence_memory_objective_identity"
        ),
        "phase_a_prime_contract": phase_a_prime_contract,
        "expected_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
    }


def _run_completion_evidence(
    run_dir: Path,
    *,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Require terminal full-run evidence independently of best-checkpoint epoch."""

    run_dir = run_dir.resolve()
    if run_dir.name != EXPECTED_EXPERIMENT_NAME:
        raise ConfirmationInputError("Confirmation run directory has the wrong name")
    expected_checkpoint = (run_dir / "checkpoints" / "best.pt").resolve()
    if checkpoint_path != expected_checkpoint:
        raise ConfirmationInputError(
            "Confirmation accepts only RUN_DIR/checkpoints/best.pt"
        )
    summary_path = run_dir / "selection_summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    resolved_config_path = run_dir / "config.resolved.json"
    summary = _json(summary_path)
    if (
        summary.get("has_feasible_checkpoint") is not True
        or summary.get("required") is not True
    ):
        raise ConfirmationInputError(
            "Run did not terminate with a required feasible checkpoint"
        )
    best_score = float(summary.get("best_feasible_score", math.nan))
    if not math.isfinite(best_score):
        raise ConfirmationInputError("Run has no finite feasible score")
    early_stopping = summary.get("early_stopping")
    if not isinstance(early_stopping, dict) or int(
        early_stopping.get("validation_count", -1)
    ) < 2:
        raise ConfirmationInputError(
            "Full run did not complete at least two validation events"
        )
    rows = read_jsonl(metrics_path)
    if len(rows) < 2:
        raise ConfirmationInputError("Run metrics contain fewer than two epochs")
    complete_rows = [
        row
        for row in rows
        if float(row.get("validation_pending", 1.0)) == 0.0
        and "selection_feasible" in row
    ]
    if len(complete_rows) < 2 or len(complete_rows) != len(rows):
        raise ConfirmationInputError(
            "Run has incomplete or pending development validation evidence"
        )
    epochs = [int(row.get("epoch", -1)) for row in complete_rows]
    if epochs != list(range(1, max(epochs) + 1)) or max(epochs) < 2:
        raise ConfirmationInputError("Run epoch history is incomplete or noncanonical")
    required_namespaces = (
        "val_text_only/",
        "val_sentence_memory/",
        "val_shuffled_sentence_memory/",
        "val_motion_shuffled_sentence_memory/",
    )
    for row in complete_rows:
        if any(not any(key.startswith(prefix) for key in row) for prefix in required_namespaces):
            raise ConfirmationInputError(
                "A completed epoch lacks one of the four development modes"
            )
    feasible_rows = [
        row for row in complete_rows if float(row.get("selection_feasible", 0.0)) == 1.0
    ]
    if not feasible_rows:
        raise ConfirmationInputError("Run metrics contain no feasible epoch")
    metric_best = min(float(row["selection_score"]) for row in feasible_rows)
    if best_score != metric_best or best_score != float(
        checkpoint["development_selection_score"]
    ):
        raise ConfirmationInputError(
            "best.pt score does not match metrics.jsonl/selection_summary.json"
        )
    selected_rows = [
        row
        for row in feasible_rows
        if int(row["epoch"]) == int(checkpoint["epoch"])
        and float(row["selection_score"]) == best_score
    ]
    if len(selected_rows) != 1:
        raise ConfirmationInputError("best.pt epoch is not the unique best metric row")
    resolved_cfg = _json(resolved_config_path)
    if resolved_cfg.get("experiment_name") != EXPECTED_EXPERIMENT_NAME:
        raise ConfirmationInputError("Resolved run config has the wrong experiment")
    return {
        "run_dir": str(run_dir),
        "metrics_jsonl": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
            "complete_validation_events": len(complete_rows),
            "terminal_epoch": max(epochs),
        },
        "selection_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
            "best_feasible_score": best_score,
            "early_stopping": early_stopping,
        },
        "resolved_config": {
            "path": str(resolved_config_path.resolve()),
            "sha256": sha256_file(resolved_config_path),
        },
        "selected_checkpoint_sha256": checkpoint["sha256"],
    }


def _v2_checkpoint_evidence(checkpoint_path: Path) -> dict[str, Any]:
    digest = sha256_file(checkpoint_path)
    if digest != V2_TEXT_ONLY_CHECKPOINT_SHA256:
        raise ConfirmationInputError(
            "The original v2 text-only checkpoint SHA256 does not match the lock"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ConfirmationInputError("Original v2 checkpoint is not a mapping")
    if str(checkpoint.get("model_type")) != "dual_mode_continuous_trajectory_field":
        raise ConfirmationInputError("Original baseline checkpoint is not v2")
    cfg = dict(checkpoint.get("config", {}) or {})
    if str(cfg.get("text", {}).get("model_path", "")).replace("\\", "/") != (
        "deps/mt5-base"
    ):
        raise ConfirmationInputError("Original v2 checkpoint does not use mT5")
    return {
        "path": str(checkpoint_path),
        "sha256": digest,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "model_type": str(checkpoint.get("model_type")),
        "trajectory_contract_version": int(
            checkpoint.get("trajectory_contract_version", -1)
        ),
    }


def analyze_confirmation(
    *,
    checkpoint_path: Path,
    run_dir: Path,
    config_path: Path,
    v2_checkpoint_path: Path,
    v2_config_path: Path,
    partition_dir: Path,
    mode_dirs: Mapping[str, Path],
    all_null_dir: Path,
    v2_text_only_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if int(bootstrap_samples) != BOOTSTRAP_SAMPLES or int(seed) != BOOTSTRAP_SEED:
        raise ConfirmationInputError(
            "The locked confirmation protocol requires 10,000 resamples, seed 1234"
        )
    if set(mode_dirs) != set(MODE_SENTENCE_MEMORY):
        raise ConfirmationInputError("Exactly four predeclared inference modes are required")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise ConfirmationInputError(f"Checkpoint does not exist: {checkpoint_path}")
    v2_checkpoint_path = v2_checkpoint_path.resolve()
    if not v2_checkpoint_path.is_file():
        raise ConfirmationInputError(
            f"Original v2 checkpoint does not exist: {v2_checkpoint_path}"
        )
    config_path = config_path.resolve()
    v2_config_path = v2_config_path.resolve()
    if (
        not v2_config_path.is_file()
        or v2_config_path.name != "csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
    ):
        raise ConfirmationInputError("Pinned v2 inference config is unavailable")
    partition = load_confirmation_partition(partition_dir)
    checkpoint = _checkpoint_evidence(
        checkpoint_path,
        str(partition["payload"]["partition_digest"]),
        config_path,
    )
    run_completion = _run_completion_evidence(
        Path(run_dir), checkpoint_path=checkpoint_path, checkpoint=checkpoint
    )
    v2_checkpoint = _v2_checkpoint_evidence(v2_checkpoint_path)
    modes = {
        mode: load_mode_export(
            mode,
            Path(mode_dirs[mode]),
            partition=partition,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            checkpoint_epoch=checkpoint["epoch"],
            motion_shuffle_seed=int(seed),
        )
        for mode in MODE_SENTENCE_MEMORY
    }
    _validate_motion_shuffle_pair(
        modes["sentence_memory"], modes["motion_shuffled_sentence_memory"]
    )
    epochs = {int(mode["summary"].get("checkpoint_epoch", -1)) for mode in modes.values()}
    if epochs != {checkpoint["epoch"]}:
        raise ConfirmationInputError("Export checkpoint epochs do not match the lock")
    all_null = load_integrity_export(
        "all_null",
        Path(all_null_dir),
        partition=partition,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    v2_text_only = load_integrity_export(
        "v2_text_only",
        Path(v2_text_only_dir),
        partition=partition,
        checkpoint_path=v2_checkpoint_path,
        config_path=v2_config_path,
    )
    if int(all_null["summary"].get("checkpoint_epoch", -1)) != checkpoint["epoch"]:
        raise ConfirmationInputError("All-null export checkpoint epoch mismatch")
    if int(v2_text_only["summary"].get("checkpoint_epoch", -1)) != v2_checkpoint[
        "epoch"
    ]:
        raise ConfirmationInputError("v2 text-only export checkpoint epoch mismatch")
    all_null_diagnostics = _validate_all_null_export(all_null)
    all_null_vs_off = _prediction_parity(
        all_null, modes["text_only"], require_exact=True
    )
    selected_off_vs_v2 = _prediction_parity(modes["text_only"], v2_text_only)

    reference_duration = modes["text_only"]["durations"]
    duration_max_abs = max(
        float(np.max(np.abs(mode["durations"] - reference_duration), initial=0.0))
        for mode in modes.values()
    )
    cluster_names, groups = _cluster_index_groups(partition["normalized_texts"])
    if len(cluster_names) != EXPECTED_CONFIRMATION_TEXTS:
        raise ConfirmationInputError("Confirmation statistical unit count is not 540")
    bootstrap = cluster_bootstrap_indices(
        len(cluster_names), samples=int(bootstrap_samples), seed=int(seed)
    )

    aggregate_metrics: dict[str, Any] = {}
    cluster_metric_values: dict[tuple[str, str, str, str], np.ndarray] = {}
    for alignment in ALIGNMENTS:
        aggregate_metrics[alignment] = {}
        for part in PARTS:
            aggregate_metrics[alignment][part] = {}
            for metric in ("dtw", "ndtw", "ndtw_ref"):
                aggregate_metrics[alignment][part][metric] = {}
                for mode_name, mode in modes.items():
                    values = _cluster_mean(
                        mode["metrics"][alignment][part][metric], groups
                    )
                    cluster_metric_values[(alignment, part, metric, mode_name)] = values
                    aggregate_metrics[alignment][part][metric][mode_name] = {
                        "mean": float(values.mean()),
                        "ci95": confidence_interval(values[bootstrap].mean(axis=1)),
                        "cluster_count": len(values),
                    }

    aggregate_hand_diagnostics: dict[str, Any] = {}
    hand_metric_values: dict[tuple[str, str, str], np.ndarray] = {}
    for hand in ("lhand", "rhand"):
        aggregate_hand_diagnostics[hand] = {}
        for metric in ("motion_path_error", "jerk_magnitude_ratio"):
            aggregate_hand_diagnostics[hand][metric] = {}
            for mode_name, mode in modes.items():
                values = _cluster_mean(
                    mode["metrics"]["default"][hand][metric], groups
                )
                hand_metric_values[(hand, metric, mode_name)] = values
                aggregate_hand_diagnostics[hand][metric][mode_name] = {
                    "mean": float(values.mean()),
                    "ci95": confidence_interval(values[bootstrap].mean(axis=1)),
                    "cluster_count": len(values),
                }

    primary_comparators = (
        "text_only",
        "shuffled_sentence_memory",
        "motion_shuffled_sentence_memory",
    )
    correct_pa = cluster_metric_values[("pa", "wholebody", "ndtw", "sentence_memory")]
    comparisons: dict[str, Any] = {}
    raw_p = {}
    for comparator in primary_comparators:
        reference = cluster_metric_values[("pa", "wholebody", "ndtw", comparator)]
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
    # The predeclared family contains exactly the two corruption controls.
    corruption_p = {
        name: raw_p[name]
        for name in (
            "shuffled_sentence_memory",
            "motion_shuffled_sentence_memory",
        )
    }
    adjusted_p = holm_adjust(corruption_p)
    for comparator, effect in comparisons.items():
        effect["holm_family"] = (
            "full_shuffle_and_motion_shuffle"
            if comparator in adjusted_p
            else None
        )
        effect["holm_adjusted_p"] = adjusted_p.get(comparator)
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

    correct_motion_mse_rows = _row_motion_mse(
        modes["sentence_memory"], modes["motion_shuffled_sentence_memory"]
    )
    correct_off_mse_rows = _row_motion_mse(
        modes["sentence_memory"], modes["text_only"]
    )
    motion_mse = _cluster_mean(correct_motion_mse_rows, groups)
    off_mse = _cluster_mean(correct_off_mse_rows, groups)
    rmotion = rmotion_summary(motion_mse, off_mse, bootstrap)
    rmotion["passed"] = bool(
        float(rmotion["value"]) >= MIN_RMOTION_POINT
        and float(rmotion["ci95"][0]) >= MIN_RMOTION_LOWER_CI
    )

    hand_path: dict[str, Any] = {}
    for hand in ("lhand", "rhand"):
        correct = hand_metric_values[(hand, "motion_path_error", "sentence_memory")]
        reference = hand_metric_values[(hand, "motion_path_error", "text_only")]
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
            "inference": "cluster-bootstrap CI; not part of the Holm family",
        }
    for hand, result in hand_path.items():
        result["passed"] = bool(
            float(result["relative_degradation_ci95"][1])
            < MAX_HAND_PATH_RELATIVE_DEGRADATION
        )

    duration = {
        "maximum_abs_difference_seconds": duration_max_abs,
        "tolerance_seconds": MAX_DURATION_ABS_DIFFERENCE_SECONDS,
        "passed": duration_max_abs <= MAX_DURATION_ABS_DIFFERENCE_SECONDS,
    }
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
        "selected_checkpoint_integrity": {
            "all_null_diagnostics": all_null_diagnostics,
            "all_null_vs_memory_off": all_null_vs_off,
            "memory_off_vs_original_v2": selected_off_vs_v2,
            "stored_initialization_parity": checkpoint[
                "v2_to_v3_text_only_parity"
            ],
            "passed": bool(
                all_null_diagnostics["passed"]
                and all_null_vs_off["passed"]
                and selected_off_vs_v2["passed"]
            ),
        },
    }
    promoted = all(bool(value["passed"]) for value in gates.values())

    cluster_rows = []
    for cluster_index, text in enumerate(cluster_names):
        row: dict[str, Any] = {
            "cluster_index": cluster_index,
            "normalized_text": text,
            "signer_rows": int(len(groups[cluster_index])),
            "motion_shuffled_mse": float(motion_mse[cluster_index]),
            "memory_off_mse": float(off_mse[cluster_index]),
        }
        for mode in MODE_SENTENCE_MEMORY:
            row[f"pa_ndtw_wholebody_{mode}"] = float(
                cluster_metric_values[("pa", "wholebody", "ndtw", mode)][
                    cluster_index
                ]
            )
            for hand in ("lhand", "rhand"):
                row[f"motion_path_error_{hand}_{mode}"] = float(
                    hand_metric_values[(hand, "motion_path_error", mode)][
                        cluster_index
                    ]
                )
        cluster_rows.append(row)

    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scientific_split": "val_confirmation_novel_text_only",
        "test_data_accessed": False,
        "statistical_unit": "normalized sentence text; signer rows averaged first",
        "cluster_count": len(cluster_names),
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "confidence": 0.95,
            "multiple_testing": (
                "Holm correction over exactly the full-shuffle and "
                "motion-shuffle PA-nDTW comparisons"
            ),
        },
        "checkpoint": {"path": str(checkpoint_path), **checkpoint},
        "run_completion": run_completion,
        "original_v2_checkpoint": v2_checkpoint,
        "original_v2_config": {
            "path": str(v2_config_path),
            "sha256": sha256_file(v2_config_path),
        },
        "partition": {
            "path": str(partition["dir"]),
            "partition_digest": partition["payload"]["partition_digest"],
            "artifact_identity": partition["payload"]["artifact_identity"],
            "confirmation_manifest_sha256": partition["manifest_sha256"],
            "confirmation_rows": len(partition["manifest_rows"]),
        },
        "inputs": {
            mode: {
                "directory": str(value["dir"]),
                "export_summary_sha256": sha256_file(value["summary_path"]),
                "sentence_memory_mode": MODE_SENTENCE_MEMORY[mode],
            }
            for mode, value in modes.items()
        },
        "integrity_inputs": {
            "all_null": {
                "directory": str(all_null["dir"]),
                "export_summary_sha256": sha256_file(all_null["summary_path"]),
                "sentence_memory_mode": "all_null",
            },
            "v2_text_only": {
                "directory": str(v2_text_only["dir"]),
                "export_summary_sha256": sha256_file(
                    v2_text_only["summary_path"]
                ),
                "sentence_memory_mode": "not_applicable",
            },
        },
        "aggregate_metrics": aggregate_metrics,
        "aggregate_hand_diagnostics": aggregate_hand_diagnostics,
        "promotion_gates": gates,
        "promoted_for_phase_b": promoted,
        "interpretation": (
            "Phase B is justified only when every predeclared confirmation gate "
            "passes; otherwise keep the inherited generator frozen."
        ),
    }
    return summary, cluster_rows


def _write_outputs(
    out_dir: Path, summary: dict[str, Any], cluster_rows: list[dict[str, Any]]
) -> None:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise ConfirmationInputError(f"Refusing to overwrite confirmation: {out_dir}")
    building = out_dir.with_name(f".{out_dir.name}.building")
    if building.exists():
        raise ConfirmationInputError(f"Incomplete confirmation build exists: {building}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        summary_path = building / "confirmation_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        csv_path = building / "confirmation_clusters.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
            writer.writeheader()
            writer.writerows(cluster_rows)
        gates = summary["promotion_gates"]
        report = [
            "# Phase-A motion-contrast confirmation",
            "",
            f"- Decision: **{'PASS' if summary['promoted_for_phase_b'] else 'FAIL'}**",
            f"- Population: {summary['cluster_count']} novel validation text clusters",
            f"- PA-nDTW gate: {gates['pa_ndtw_wholebody']['passed']}",
            f"- Rmotion: {gates['rmotion']['value']:.6f} "
            f"(95% CI {gates['rmotion']['ci95']})",
            f"- Hand-path gate: {gates['hand_path']['passed']}",
            f"- Duration parity: {gates['duration_invariance']['passed']}",
            "- Selected-checkpoint all-null/v2 integrity: "
            f"{gates['selected_checkpoint_integrity']['passed']}",
            "",
            "Phase B is allowed only when every gate passes.",
            "",
        ]
        (building / "report.md").write_text("\n".join(report), encoding="utf-8")
        files = {}
        for name in (
            "confirmation_summary.json",
            "confirmation_clusters.csv",
            "report.md",
        ):
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
                    "promoted_for_phase_b": summary["promoted_for_phase_b"],
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


def main() -> None:
    args = parse_args()
    mode_dirs = {
        "text_only": args.text_only_dir,
        "sentence_memory": args.sentence_memory_dir,
        "shuffled_sentence_memory": args.shuffled_sentence_memory_dir,
        "motion_shuffled_sentence_memory": args.motion_shuffled_sentence_memory_dir,
    }
    summary, cluster_rows = analyze_confirmation(
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
        config_path=args.config,
        v2_checkpoint_path=args.v2_checkpoint,
        v2_config_path=args.v2_config,
        partition_dir=args.partition_dir,
        mode_dirs=mode_dirs,
        all_null_dir=args.all_null_dir,
        v2_text_only_dir=args.v2_text_only_dir,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    _write_outputs(args.out_dir, summary, cluster_rows)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "cluster_count": summary["cluster_count"],
                "promoted_for_phase_b": summary["promoted_for_phase_b"],
            },
            indent=2,
        )
    )
    if not summary["promoted_for_phase_b"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
