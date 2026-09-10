"""Train-only relevance calibration for sentence-motion memory.

The calibration intentionally works from explicit file names.  In particular,
it never enumerates neighbor tables, so a caller can prove that only the train
table was made visible in its node-local staging directory.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CALIBRATION_SCHEMA_NAME = "signtrajfield_sentence_memory_relevance_calibration"
CALIBRATION_SCHEMA_VERSION = 1
READY_SCHEMA_NAME = "signtrajfield_sentence_memory_relevance_calibration_ready"
MAP_SCHEMA_NAME = "signtrajfield_sentence_memory_relevance_calibration_map"
MAP_SCHEMA_VERSION = 1
METHOD_NAME = "balanced_logistic_float64_v1"
SPLIT_MODE = "sha256_ranked_query_identity_90_10_v1"
ALTERNATIVE_MAP_MODE = "train_item_full_replacement_coprime_stride_v1"
FEATURE_MODE = "absolute_adjusted_score_v1"
DEFAULT_SEED = 1234
SPLIT_NONCE = "csl_daily_relevance_calibration_split_v1"
ALTERNATIVE_NONCE = "csl_daily_relevance_calibration_full_replacement_v1"
REQUIRED_SOURCE_FILES = frozenset(
    {
        "NIAF/continuous_trajectory_field/relevance_calibration.py",
        "NIAF/continuous_trajectory_field/scripts/calibrate_sentence_memory_relevance.py",
        "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py",
        "scripts/NIAF/calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh",
        "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh",
    }
)
LEGACY_SOURCE_FILE_PROFILE = "sentence_memory_relevance_v1"
STAGE_C_SOURCE_FILE_PROFILE = "stage_c_generator_adaptation_v1"
STAGE_C_RETRY2_SOURCE_FILE_PROFILE = "stage_c_generator_adaptation_retry2_v1"
STAGE_C_PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE = (
    "stage_c_generator_adaptation_protocol_v2_run_r3"
)
STAGE_C_REQUIRED_SOURCE_FILES = frozenset(
    {
        "NIAF/continuous_trajectory_field/relevance_calibration.py",
        "NIAF/continuous_trajectory_field/scripts/calibrate_sentence_memory_relevance.py",
        "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py",
        "scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh",
        "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh",
    }
)
STAGE_C_PROTOCOL_V2_RUN_R3_REQUIRED_SOURCE_FILES = frozenset(
    {
        "NIAF/continuous_trajectory_field/relevance_calibration.py",
        "NIAF/continuous_trajectory_field/scripts/calibrate_sentence_memory_relevance.py",
        "NIAF/continuous_trajectory_field/scripts/decide_centered_memory_stage.py",
        (
            "scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_"
            "protocol_v2_run_r3_sbatch.sh"
        ),
        "scripts/NIAF/stage_sentence_memory_train_val_only_node.sh",
    }
)
SOURCE_FILE_PROFILES = {
    LEGACY_SOURCE_FILE_PROFILE: REQUIRED_SOURCE_FILES,
    STAGE_C_SOURCE_FILE_PROFILE: STAGE_C_REQUIRED_SOURCE_FILES,
    STAGE_C_RETRY2_SOURCE_FILE_PROFILE: STAGE_C_REQUIRED_SOURCE_FILES,
    STAGE_C_PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE: (
        STAGE_C_PROTOCOL_V2_RUN_R3_REQUIRED_SOURCE_FILES
    ),
}


class RelevanceCalibrationError(RuntimeError):
    """Raised when calibration inputs or a sealed artifact are invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RelevanceCalibrationError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RelevanceCalibrationError(f"Expected a JSON object: {path}")
    return value


def _identity_payload(value: Mapping[str, Any], identity_key: str) -> dict[str, Any]:
    payload = {key: item for key, item in value.items() if key != identity_key}
    if value.get(identity_key) != digest_json(payload):
        raise RelevanceCalibrationError(
            f"Invalid {identity_key.replace('_', ' ')} digest"
        )
    return payload


def calibrated_probability(
    adjusted_score: np.ndarray | Sequence[float] | float,
    *,
    intercept: float,
    slope: float,
) -> np.ndarray:
    """Apply a numerically stable scalar logistic calibration in float64."""

    score = np.asarray(adjusted_score, dtype=np.float64)
    logits = float(intercept) + float(slope) * score
    output = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    output[~positive] = exp_logits / (1.0 + exp_logits)
    return output


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0 or positives + negatives != len(labels):
        raise RelevanceCalibrationError("Logistic labels must contain both 0 and 1")
    weights = np.empty(len(labels), dtype=np.float64)
    weights[labels == 1] = 0.5 / positives
    weights[labels == 0] = 0.5 / negatives
    return weights


def fit_balanced_logistic(
    scores: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Fit an unregularized, class-balanced scalar logistic model in float64."""

    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(x) != len(y) or len(x) < 2 or not bool(np.isfinite(x).all()):
        raise RelevanceCalibrationError("Calibration scores/labels are malformed")
    weights = _balanced_weights(y)
    mean = float(np.sum(weights * x))
    variance = float(np.sum(weights * np.square(x - mean)))
    scale = math.sqrt(max(variance, 1e-24))
    if scale <= 1e-12:
        raise RelevanceCalibrationError("Calibration scores have zero variance")
    normalized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x), dtype=np.float64), normalized))
    beta = np.zeros(2, dtype=np.float64)

    def objective(value: np.ndarray) -> float:
        logits = design @ value
        # log(1 + exp(z)) - y*z, stable for large magnitude logits.
        loss = np.logaddexp(0.0, logits) - y.astype(np.float64) * logits
        return float(np.sum(weights * loss))

    converged = False
    iteration = 0
    for iteration in range(1, int(maximum_iterations) + 1):
        logits = design @ beta
        probability = calibrated_probability(logits, intercept=0.0, slope=1.0)
        gradient = design.T @ (weights * (probability - y))
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (curvature[:, None] * design)
        # This is numerical damping, not statistical regularization.  It only
        # resolves the finite-precision singularity of a two-parameter solve.
        hessian += np.eye(2, dtype=np.float64) * 1e-15
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise RelevanceCalibrationError(
                "Balanced logistic Hessian is singular"
            ) from error
        if float(np.max(np.abs(gradient))) <= float(tolerance):
            converged = True
            break
        current = objective(beta)
        multiplier = 1.0
        accepted = False
        while multiplier >= 2.0**-30:
            proposal = beta - multiplier * step
            if objective(proposal) <= current:
                beta = proposal
                accepted = True
                break
            multiplier *= 0.5
        if not accepted:
            raise RelevanceCalibrationError("Balanced logistic line search failed")
        if float(np.max(np.abs(multiplier * step))) <= float(tolerance):
            converged = True
            break
    if not converged:
        raise RelevanceCalibrationError(
            f"Balanced logistic did not converge in {maximum_iterations} iterations"
        )

    slope = float(beta[1] / scale)
    raw_intercept = float(beta[0] - beta[1] * mean / scale)
    # Seal both equivalent parameterizations.  The requested public form is
    # sigmoid(a * (score - b)); derive the consumed intercept from the stored
    # float64 a/b pair so the relationship can be checked exactly on reload.
    threshold = float(-raw_intercept / slope)
    intercept = float(-slope * threshold)
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0.0:
        raise RelevanceCalibrationError(
            "Calibrated relevance must have a finite positive score slope"
        )
    return {
        "a": slope,
        "b": threshold,
        "formula": "sigmoid(a*(adjusted_score-b))",
        "intercept": intercept,
        "intercept_derivation": "float64(-a*b)",
        "slope": slope,
        "iterations": iteration,
        "converged": True,
        "fit_score_mean": mean,
        "fit_score_scale": scale,
        "class_weight_positive_sum": float(weights[y == 1].sum()),
        "class_weight_negative_sum": float(weights[y == 0].sum()),
    }


def binary_auroc(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Compute AUROC with average ranks for ties."""

    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(x) or not bool(np.isfinite(x).all()):
        raise RelevanceCalibrationError("AUROC inputs are malformed")
    positive_count = int(np.count_nonzero(y == 1))
    negative_count = int(np.count_nonzero(y == 0))
    if positive_count == 0 or negative_count == 0:
        raise RelevanceCalibrationError("AUROC requires both classes")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def query_identity(row: Mapping[str, Any], item_id: int) -> dict[str, Any]:
    return {
        "bank_item_id": int(item_id),
        "motion_path": str(row.get("motion_path", "")),
        "name": str(row.get("name", "")),
        "semantic_group_id": str(row.get("semantic_group_id", "")),
        "source_group_id": str(row.get("source_group_id", "")),
        "source_id": str(row.get("source_id", "")),
    }


def sha_query_partition(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = 0.10,
    nonce: str = SPLIT_NONCE,
) -> tuple[np.ndarray, list[str]]:
    """Return an exact 90/10 split by sorted SHA256 query identities."""

    if not 0.0 < float(validation_fraction) < 1.0 or not rows:
        raise RelevanceCalibrationError(
            "Query split requires rows and a fraction in (0,1)"
        )
    digests = [
        hashlib.sha256(
            canonical_json(
                {
                    "mode": SPLIT_MODE,
                    "nonce": str(nonce),
                    "query": query_identity(row, index),
                    "seed": int(seed),
                }
            ).encode("utf-8")
        ).hexdigest()
        for index, row in enumerate(rows)
    ]
    validation_count = max(1, int(round(len(rows) * float(validation_fraction))))
    validation_count = min(validation_count, len(rows) - 1)
    order = sorted(range(len(rows)), key=lambda index: (digests[index], index))
    split = np.zeros(len(rows), dtype=np.uint8)
    split[np.asarray(order[:validation_count], dtype=np.int64)] = 1
    return split, digests


def deterministic_alternative_item_ids(
    *,
    query_item_id: int,
    query_row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    positive_item_ids: Sequence[int],
    positive_group_ids: Sequence[int],
    item_group_ids: np.ndarray,
    count: int = 8,
    seed: int = DEFAULT_SEED,
    nonce: str = ALTERNATIVE_NONCE,
) -> list[int]:
    """Choose exact negative train items using a stable full-bank permutation."""

    item_count = len(rows)
    if item_count < 2 or len(item_group_ids) != item_count:
        raise RelevanceCalibrationError("Alternative map has malformed train items")
    excluded_items = {int(query_item_id), *map(int, positive_item_ids)}
    excluded_groups = {
        int(item_group_ids[int(query_item_id)]),
        *map(int, positive_group_ids),
    }
    token = canonical_json(
        {
            "mode": ALTERNATIVE_MAP_MODE,
            "nonce": str(nonce),
            "query": query_identity(query_row, query_item_id),
            "seed": int(seed),
        }
    )
    hashed = hashlib.sha256(token.encode("utf-8")).digest()
    start = int.from_bytes(hashed[:8], "big") % item_count
    step = (
        1
        if item_count == 1
        else 1 + int.from_bytes(hashed[8:16], "big") % (item_count - 1)
    )
    while math.gcd(step, item_count) != 1:
        step += 1
        if step >= item_count:
            step = 1
    query_semantic = str(query_row.get("semantic_group_id", ""))
    query_source = str(query_row.get("source_id", ""))
    query_source_group = str(query_row.get("source_group_id", ""))
    selected: list[int] = []
    selected_groups: set[int] = set()
    for probe in range(item_count):
        item_id = int((start + probe * step) % item_count)
        candidate = rows[item_id]
        group_id = int(item_group_ids[item_id])
        if (
            item_id in excluded_items
            or group_id in excluded_groups
            or group_id in selected_groups
        ):
            continue
        if (
            query_semantic
            and str(candidate.get("semantic_group_id", "")) == query_semantic
        ):
            continue
        if query_source and str(candidate.get("source_id", "")) == query_source:
            continue
        if (
            query_source_group
            and str(candidate.get("source_group_id", "")) == query_source_group
        ):
            continue
        selected.append(item_id)
        selected_groups.add(group_id)
        if len(selected) == int(count):
            return selected
    raise RelevanceCalibrationError(
        f"Query {query_item_id} has only {len(selected)} deterministic alternatives"
    )


def calibration_map_content_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash array names, exact dtypes/shapes, and C-order contents."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        descriptor = canonical_json(
            {"dtype": array.dtype.str, "name": name, "shape": list(array.shape)}
        )
        digest.update(descriptor.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _candidate_allowed(
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_id: int,
    query_id: int,
) -> bool:
    if int(candidate_id) == int(query_id):
        return False
    for key in ("semantic_group_id", "source_id", "source_group_id"):
        left = str(query.get(key, ""))
        right = str(candidate.get(key, ""))
        if left and right and left == right:
            return False
    return True


def deterministic_positive_item_id(
    *,
    query_item_id: int,
    group_id: int,
    rows: Sequence[Mapping[str, Any]],
    group_offsets: np.ndarray,
    group_item_ids: np.ndarray,
    durations: np.ndarray,
    motion_offsets: np.ndarray,
    hand_valid: np.ndarray,
) -> int:
    """Mirror the provider's deterministic evaluation realization selection."""

    start = int(group_offsets[int(group_id)])
    end = int(group_offsets[int(group_id) + 1])
    query = rows[int(query_item_id)]
    query_duration = max(float(durations[int(query_item_id)]), 1e-6)
    ranking: list[tuple[float, float, int]] = []
    for raw_id in np.asarray(group_item_ids[start:end]).tolist():
        item_id = int(raw_id)
        if not _candidate_allowed(query, rows[item_id], item_id, query_item_id):
            continue
        duration_gap = abs(
            math.log(max(float(durations[item_id]), 1e-6)) - math.log(query_duration)
        )
        token_start = int(motion_offsets[item_id])
        token_end = int(motion_offsets[item_id + 1])
        hand_quality = float(
            np.asarray(hand_valid[token_start:token_end], dtype=np.float64).mean()
        )
        ranking.append((duration_gap, -hand_quality, item_id))
    if not ranking:
        raise RelevanceCalibrationError(
            f"No allowed realization for query={query_item_id}, group={group_id}"
        )
    return int(min(ranking)[2])


def _load_npy(path: Path, expected: Mapping[str, Any]) -> np.ndarray:
    if sha256_file(path) != str(expected.get("sha256", "")):
        raise RelevanceCalibrationError(f"Bank artifact SHA256 mismatch: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != list(expected.get("shape", [])) or str(value.dtype) != str(
        expected.get("dtype", "")
    ):
        raise RelevanceCalibrationError(f"Bank artifact shape/dtype mismatch: {path}")
    return value


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != str(expected_sha256):
        raise RelevanceCalibrationError(f"Bank JSONL SHA256 mismatch: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RelevanceCalibrationError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise RelevanceCalibrationError(
                    f"Expected object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _npz_scalar(data: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise RelevanceCalibrationError(f"Neighbor scalar is malformed: {name}")
    return value.item()


def validate_bank_index_order(rows: Sequence[Mapping[str, Any]]) -> None:
    """Prove metadata row position is the exact concrete bank item ID."""

    for row_index, row in enumerate(rows):
        try:
            bank_index = int(row.get("bank_index", -1))
        except (TypeError, ValueError) as error:
            raise RelevanceCalibrationError(
                f"Bank metadata row {row_index} has an invalid bank_index"
            ) from error
        if bank_index != row_index:
            raise RelevanceCalibrationError(
                "Bank metadata order is not identity-preserving: "
                f"row={row_index}, bank_index={bank_index}"
            )


def load_train_inputs(
    bank_dir: str | Path,
    train_neighbor_path: str | Path,
    *,
    expected_bank_id: str,
    expected_neighbor_sha256: str,
) -> dict[str, Any]:
    """Load only explicit train-bank artifacts and the explicit train table."""

    bank_dir = Path(bank_dir)
    neighbor_path = Path(train_neighbor_path)
    # Deliberately do not list/glob ``bank_dir`` or synthesize another split.
    manifest_path = bank_dir / "bank.json"
    ready_path = bank_dir / "READY"
    build_summary_path = bank_dir / "build_summary.json"
    manifest = _json(manifest_path)
    ready = _json(ready_path)
    if (
        manifest.get("schema_name") != "signtrajfield_sentence_motion_memory"
        or int(manifest.get("schema_version", -1)) != 1
        or ready.get("schema_name") != "signtrajfield_sentence_motion_memory"
        or int(ready.get("schema_version", -1)) != 1
        or manifest.get("bank_id") != expected_bank_id
        or ready.get("bank_id") != expected_bank_id
        or ready.get("build_summary_sha256") != sha256_file(build_summary_path)
    ):
        raise RelevanceCalibrationError("Sentence-memory bank identity changed")
    if neighbor_path.name != "neighbors_train.npz":
        raise RelevanceCalibrationError("Calibration accepts only neighbors_train.npz")
    neighbor_sha256 = sha256_file(neighbor_path)
    if neighbor_sha256 != expected_neighbor_sha256:
        raise RelevanceCalibrationError("Train-neighbor table SHA256 changed")
    artifacts = dict(manifest.get("artifacts", {}) or {})

    def artifact(name: str) -> tuple[Path, Mapping[str, Any]]:
        identity = dict(artifacts.get(name, {}) or {})
        filename = str(identity.get("file", ""))
        if not filename or Path(filename).name != filename:
            raise RelevanceCalibrationError(f"Unsafe bank artifact path: {name}")
        return bank_dir / filename, identity

    metadata_path, metadata_identity = artifact("metadata")
    rows = _load_jsonl(metadata_path, str(metadata_identity.get("sha256", "")))
    if not rows or any(str(row.get("source_split", "")) != "train" for row in rows):
        raise RelevanceCalibrationError("Calibration bank is not train-only")
    validate_bank_index_order(rows)
    arrays = {}
    for name in (
        "group_keys",
        "group_offsets",
        "group_item_ids",
        "item_group_ids",
        "durations",
        "offsets",
        "hand_valid",
    ):
        path, identity = artifact(name)
        arrays[name] = _load_npy(path, identity)

    with np.load(neighbor_path, allow_pickle=False) as data:
        required = {
            "schema_name",
            "schema_version",
            "bank_id",
            "query_split",
            "query_manifest_sha256",
            "query_order_sha256",
            "query_names",
            "top_m",
            "ids",
            "scores",
        }
        if not required.issubset(data.files):
            raise RelevanceCalibrationError("Train-neighbor table is incomplete")
        if (
            str(_npz_scalar(data, "schema_name")) != "signtrajfield_sentence_neighbors"
            or int(_npz_scalar(data, "schema_version")) != 1
            or str(_npz_scalar(data, "bank_id")) != expected_bank_id
            or str(_npz_scalar(data, "query_split")) != "train"
        ):
            raise RelevanceCalibrationError("Train-neighbor table identity changed")
        top_m = int(_npz_scalar(data, "top_m"))
        ids = np.asarray(data["ids"], dtype=np.int64)
        scores = np.asarray(data["scores"], dtype=np.float64)
        query_names = [str(value) for value in np.asarray(data["query_names"]).tolist()]
        neighbor_identity = {
            "bytes": neighbor_path.stat().st_size,
            "query_manifest_sha256": str(_npz_scalar(data, "query_manifest_sha256")),
            "query_order_sha256": str(_npz_scalar(data, "query_order_sha256")),
            "query_count": len(query_names),
            "sha256": neighbor_sha256,
            "top_m": top_m,
        }
    if ids.shape != scores.shape or ids.shape[0] != len(rows) or top_m < 8:
        raise RelevanceCalibrationError("Train-neighbor arrays have the wrong shape")
    if query_names != [str(row.get("name", "")) for row in rows]:
        raise RelevanceCalibrationError("Train-neighbor query order changed")
    if (
        bool(np.any(ids[:, :8] < 0))
        or bool(np.any(ids[:, :8] >= int(arrays["group_keys"].shape[0])))
        or not bool(np.isfinite(scores).all())
    ):
        raise RelevanceCalibrationError(
            "Top-8 train neighbors are incomplete/non-finite"
        )
    if len(arrays["item_group_ids"]) != len(rows):
        raise RelevanceCalibrationError("Bank item metadata/array count differs")
    return {
        "arrays": arrays,
        "bank_identity": {
            "bank_id": expected_bank_id,
            "bank_manifest_sha256": sha256_file(manifest_path),
            "bank_ready_sha256": sha256_file(ready_path),
            "item_count": len(rows),
            "semantic_group_count": int(arrays["group_keys"].shape[0]),
        },
        "neighbor_identity": neighbor_identity,
        "neighbor_ids": ids,
        "neighbor_scores": scores,
        "rows": rows,
    }


def build_calibration_map(
    inputs: Mapping[str, Any],
    *,
    duration_weight: float = 0.05,
    candidate_count: int = 8,
    seed: int = DEFAULT_SEED,
    split_nonce: str = SPLIT_NONCE,
    alternative_nonce: str = ALTERNATIVE_NONCE,
) -> dict[str, np.ndarray]:
    rows = list(inputs["rows"])
    source = dict(inputs["arrays"])
    group_keys = np.asarray(source["group_keys"], dtype=np.float64)
    item_group_ids = np.asarray(source["item_group_ids"], dtype=np.int64)
    durations = np.asarray(source["durations"], dtype=np.float64)
    neighbor_ids = np.asarray(inputs["neighbor_ids"], dtype=np.int64)
    neighbor_scores = np.asarray(inputs["neighbor_scores"], dtype=np.float64)
    query_count = len(rows)
    width = int(candidate_count)
    if width != 8:
        raise RelevanceCalibrationError("Calibration requires exactly eight candidates")
    split, query_digests = sha_query_partition(
        rows, seed=seed, validation_fraction=0.10, nonce=split_nonce
    )
    positive_groups = np.asarray(neighbor_ids[:, :width], dtype=np.int32).copy()
    positive_items = np.full((query_count, width), -1, dtype=np.int32)
    negative_groups = np.full((query_count, width), -1, dtype=np.int32)
    negative_items = np.full((query_count, width), -1, dtype=np.int32)
    positive_cosine = np.asarray(neighbor_scores[:, :width], dtype=np.float64).copy()
    negative_cosine = np.zeros((query_count, width), dtype=np.float64)
    positive_gap = np.zeros((query_count, width), dtype=np.float64)
    negative_gap = np.zeros((query_count, width), dtype=np.float64)
    for query_id, query in enumerate(rows):
        for rank, group_id in enumerate(positive_groups[query_id]):
            positive_items[query_id, rank] = deterministic_positive_item_id(
                query_item_id=query_id,
                group_id=int(group_id),
                rows=rows,
                group_offsets=np.asarray(source["group_offsets"]),
                group_item_ids=np.asarray(source["group_item_ids"]),
                durations=durations,
                motion_offsets=np.asarray(source["offsets"]),
                hand_valid=np.asarray(source["hand_valid"]),
            )
        alternatives = deterministic_alternative_item_ids(
            query_item_id=query_id,
            query_row=query,
            rows=rows,
            positive_item_ids=positive_items[query_id],
            positive_group_ids=positive_groups[query_id],
            item_group_ids=item_group_ids,
            count=width,
            seed=seed,
            nonce=alternative_nonce,
        )
        negative_items[query_id] = np.asarray(alternatives, dtype=np.int32)
        negative_groups[query_id] = item_group_ids[negative_items[query_id]]
        query_key = group_keys[int(item_group_ids[query_id])]
        negative_cosine[query_id] = group_keys[negative_groups[query_id]] @ query_key
        query_duration = max(float(durations[query_id]), 1e-6)
        positive_gap[query_id] = np.abs(
            np.log(query_duration)
            - np.log(np.clip(durations[positive_items[query_id]], 1e-6, None))
        )
        negative_gap[query_id] = np.abs(
            np.log(query_duration)
            - np.log(np.clip(durations[negative_items[query_id]], 1e-6, None))
        )
    positive_adjusted = positive_cosine - float(duration_weight) * positive_gap
    negative_adjusted = negative_cosine - float(duration_weight) * negative_gap
    max_name = max(1, max(len(str(row.get("name", ""))) for row in rows))
    return {
        "negative_adjusted_scores": negative_adjusted,
        "negative_cosine_scores": negative_cosine,
        "negative_duration_log_gaps": negative_gap,
        "negative_group_ids": negative_groups,
        "negative_item_ids": negative_items,
        "positive_adjusted_scores": positive_adjusted,
        "positive_cosine_scores": positive_cosine,
        "positive_duration_log_gaps": positive_gap,
        "positive_group_ids": positive_groups,
        "positive_item_ids": positive_items,
        "query_item_ids": np.arange(query_count, dtype=np.int32),
        "query_names": np.asarray(
            [str(row.get("name", "")) for row in rows], dtype=f"<U{max_name}"
        ),
        "query_partition": split,
        "query_sha256": np.asarray(query_digests, dtype="<U64"),
    }


def calibrate_from_map(
    arrays: Mapping[str, np.ndarray],
    *,
    minimum_auroc: float = 0.75,
    minimum_probability_gap: float = 0.20,
) -> dict[str, Any]:
    split = np.asarray(arrays["query_partition"], dtype=np.uint8)
    positive = np.asarray(arrays["positive_adjusted_scores"], dtype=np.float64)
    negative = np.asarray(arrays["negative_adjusted_scores"], dtype=np.float64)
    if positive.shape != negative.shape or positive.shape != (len(split), 8):
        raise RelevanceCalibrationError("Calibration map has the wrong dimensions")
    fit = split == 0
    holdout = split == 1
    if not bool(fit.any()) or not bool(holdout.any()):
        raise RelevanceCalibrationError("Calibration map lacks fit or holdout queries")
    fit_scores = np.concatenate((positive[fit].reshape(-1), negative[fit].reshape(-1)))
    fit_labels = np.concatenate(
        (
            np.ones(int(fit.sum()) * 8, dtype=np.int8),
            np.zeros(int(fit.sum()) * 8, dtype=np.int8),
        )
    )
    coefficients = fit_balanced_logistic(fit_scores, fit_labels)
    positive_probability = calibrated_probability(
        positive[holdout],
        intercept=coefficients["intercept"],
        slope=coefficients["slope"],
    ).reshape(-1)
    negative_probability = calibrated_probability(
        negative[holdout],
        intercept=coefficients["intercept"],
        slope=coefficients["slope"],
    ).reshape(-1)
    holdout_labels = np.concatenate(
        (
            np.ones(len(positive_probability), dtype=np.int8),
            np.zeros(len(negative_probability), dtype=np.int8),
        )
    )
    holdout_probability = np.concatenate((positive_probability, negative_probability))
    auroc = binary_auroc(holdout_labels, holdout_probability)
    positive_mean = float(positive_probability.mean())
    negative_mean = float(negative_probability.mean())
    gap = positive_mean - negative_mean
    accepted = bool(
        auroc >= float(minimum_auroc) and gap >= float(minimum_probability_gap)
    )
    return {
        "accepted": accepted,
        "coefficients": coefficients,
        "fit": {
            "negative_samples": int(fit.sum()) * 8,
            "positive_samples": int(fit.sum()) * 8,
            "queries": int(fit.sum()),
        },
        "holdout": {
            "auroc": float(auroc),
            "negative_mean_probability": negative_mean,
            "negative_samples": len(negative_probability),
            "positive_mean_probability": positive_mean,
            "positive_samples": len(positive_probability),
            "probability_gap": gap,
            "queries": int(holdout.sum()),
        },
        "thresholds": {
            "minimum_auroc": float(minimum_auroc),
            "minimum_probability_gap": float(minimum_probability_gap),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing path.

    Phase-A''' runs on Linux hosts.  ``renameat2(RENAME_NOREPLACE)`` closes the
    existence-check race that a preceding ``Path.exists`` plus ``os.rename``
    would leave open.  Failing closed on hosts without this primitive is safer
    than silently weakening artifact immutability.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RelevanceCalibrationError(
            "Atomic no-replace artifact publication requires renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise RelevanceCalibrationError(
                f"Calibration output already exists: {destination}"
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _make_read_only_if_supported(path: Path, mode: int) -> bool:
    try:
        path.chmod(mode)
    except OSError as error:
        # The durable CIFS experiments mount rejects chmod.  Immutability there
        # is enforced by no-replace publication plus launchers that reject any
        # existing accepted/rejected path.  Local POSIX filesystems also gain
        # read-only mode bits as defense in depth.
        if error.errno not in {
            errno.EACCES,
            errno.EPERM,
            errno.ENOTSUP,
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        }:
            raise
        return False
    return True


def publish_calibration(
    *,
    output_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    result: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    seed: int,
    duration_weight: float,
    split_nonce: str,
    alternative_nonce: str,
) -> dict[str, Any]:
    """Atomically publish a calibration; READY exists only when gates pass."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise RelevanceCalibrationError(f"Calibration output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        map_path = temporary / "calibration_map.npz"
        np.savez_compressed(
            map_path,
            schema_name=np.asarray(MAP_SCHEMA_NAME),
            schema_version=np.asarray(MAP_SCHEMA_VERSION, dtype=np.int32),
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
        _fsync_file(map_path)
        content_digest = calibration_map_content_digest(arrays)
        split = np.asarray(arrays["query_partition"], dtype=np.uint8)
        payload = {
            "accepted": bool(result["accepted"]),
            "coefficients": dict(result["coefficients"]),
            "feature": {
                "duration_weight": float(duration_weight),
                "formula": "cosine-0.05*abs(log(query_duration/candidate_duration))",
                "mode": FEATURE_MODE,
            },
            "fit": dict(result["fit"]),
            "holdout": dict(result["holdout"]),
            "inputs": dict(input_identity),
            "map": {
                "alternative_count_per_query": 8,
                "alternative_map_mode": ALTERNATIVE_MAP_MODE,
                "alternative_nonce": str(alternative_nonce),
                "content_digest": content_digest,
                "file": "calibration_map.npz",
                "positive_count_per_query": 8,
            },
            "method": {
                "class_balance": "equal_total_weight_per_class",
                "dtype": "float64",
                "name": METHOD_NAME,
                "regularization": "none",
            },
            "publication": {
                "atomic_directory_rename": True,
                "existing_destination_policy": "reject_no_replace",
                "logical_immutability": True,
            },
            "query_split": {
                "fit_queries": int(np.count_nonzero(split == 0)),
                "holdout_queries": int(np.count_nonzero(split == 1)),
                "mode": SPLIT_MODE,
                "nonce": str(split_nonce),
                "partition_digest": digest_json(split.astype(int).tolist()),
            },
            "schema_name": CALIBRATION_SCHEMA_NAME,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "seed": int(seed),
            "source": dict(source_identity),
            "thresholds": dict(result["thresholds"]),
        }
        calibration = {**payload, "identity": digest_json(payload)}
        calibration_path = temporary / "calibration.json"
        _write_json(calibration_path, calibration)
        if bool(result["accepted"]):
            ready_payload = {
                "artifact_identity": calibration["identity"],
                "calibration_sha256": sha256_file(calibration_path),
                "map_content_digest": content_digest,
                "map_sha256": sha256_file(map_path),
                "schema_name": READY_SCHEMA_NAME,
                "schema_version": 1,
            }
            _write_json(
                temporary / "READY",
                {**ready_payload, "ready_identity": digest_json(ready_payload)},
            )
        else:
            _write_json(
                temporary / "REJECTED",
                {
                    "artifact_identity": calibration["identity"],
                    "reason": "held-out calibration acceptance gates failed",
                    "schema_name": READY_SCHEMA_NAME,
                    "schema_version": 1,
                },
            )
        _fsync_directory(temporary)
        for artifact_path in temporary.iterdir():
            _make_read_only_if_supported(artifact_path, 0o444)
        _make_read_only_if_supported(temporary, 0o555)
        _rename_directory_no_replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        if temporary.exists():
            _make_read_only_if_supported(temporary, 0o755)
            for path in temporary.iterdir():
                _make_read_only_if_supported(path, 0o644)
                path.unlink()
            temporary.rmdir()
        raise
    return calibration


def _load_map_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if (
            str(_npz_scalar(data, "schema_name")) != MAP_SCHEMA_NAME
            or int(_npz_scalar(data, "schema_version")) != MAP_SCHEMA_VERSION
        ):
            raise RelevanceCalibrationError("Calibration map schema changed")
        return {
            name: np.asarray(data[name])
            for name in data.files
            if name not in {"schema_name", "schema_version"}
        }


def validate_relevance_calibration_artifact(
    artifact_dir: str | Path,
    *,
    expected_identity: str | None = None,
    minimum_auroc: float = 0.75,
    minimum_probability_gap: float = 0.20,
) -> dict[str, Any]:
    """Validate a sealed accepted artifact without discovering other files."""

    directory = Path(artifact_dir)
    ready_path = directory / "READY"
    calibration_path = directory / "calibration.json"
    map_path = directory / "calibration_map.npz"
    ready = _json(ready_path)
    _identity_payload(ready, "ready_identity")
    if (
        ready.get("schema_name") != READY_SCHEMA_NAME
        or int(ready.get("schema_version", -1)) != 1
        or ready.get("calibration_sha256") != sha256_file(calibration_path)
        or ready.get("map_sha256") != sha256_file(map_path)
    ):
        raise RelevanceCalibrationError("Calibration READY envelope changed")
    calibration = _json(calibration_path)
    _identity_payload(calibration, "identity")
    if (
        calibration.get("schema_name") != CALIBRATION_SCHEMA_NAME
        or int(calibration.get("schema_version", -1)) != CALIBRATION_SCHEMA_VERSION
        or calibration.get("accepted") is not True
        or ready.get("artifact_identity") != calibration.get("identity")
    ):
        raise RelevanceCalibrationError("Calibration artifact is not accepted/sealed")
    if expected_identity is not None and calibration.get("identity") != str(
        expected_identity
    ):
        raise RelevanceCalibrationError("Calibration identity differs from config")
    arrays = _load_map_arrays(map_path)
    content_digest = calibration_map_content_digest(arrays)
    if content_digest != dict(calibration.get("map", {}) or {}).get(
        "content_digest"
    ) or content_digest != ready.get("map_content_digest"):
        raise RelevanceCalibrationError("Calibration map content digest changed")
    feature = dict(calibration.get("feature", {}) or {})
    method = dict(calibration.get("method", {}) or {})
    map_identity = dict(calibration.get("map", {}) or {})
    publication = dict(calibration.get("publication", {}) or {})
    query_split = dict(calibration.get("query_split", {}) or {})
    thresholds = dict(calibration.get("thresholds", {}) or {})
    partition = np.asarray(arrays.get("query_partition"), dtype=np.uint8)
    if (
        feature.get("mode") != FEATURE_MODE
        or float(feature.get("duration_weight", math.nan)) != 0.05
        or method
        != {
            "class_balance": "equal_total_weight_per_class",
            "dtype": "float64",
            "name": METHOD_NAME,
            "regularization": "none",
        }
        or publication
        != {
            "atomic_directory_rename": True,
            "existing_destination_policy": "reject_no_replace",
            "logical_immutability": True,
        }
        or map_identity.get("alternative_map_mode") != ALTERNATIVE_MAP_MODE
        or map_identity.get("alternative_nonce") != ALTERNATIVE_NONCE
        or map_identity.get("file") != "calibration_map.npz"
        or int(map_identity.get("alternative_count_per_query", -1)) != 8
        or int(map_identity.get("positive_count_per_query", -1)) != 8
        or int(calibration.get("seed", -1)) != DEFAULT_SEED
        or query_split.get("mode") != SPLIT_MODE
        or query_split.get("nonce") != SPLIT_NONCE
        or query_split.get("partition_digest")
        != digest_json(partition.astype(int).tolist())
        or int(query_split.get("fit_queries", -1))
        != int(np.count_nonzero(partition == 0))
        or int(query_split.get("holdout_queries", -1))
        != int(np.count_nonzero(partition == 1))
        or float(thresholds.get("minimum_auroc", math.nan)) != float(minimum_auroc)
        or float(thresholds.get("minimum_probability_gap", math.nan))
        != float(minimum_probability_gap)
    ):
        raise RelevanceCalibrationError("Calibration method/map contract changed")
    holdout = dict(calibration.get("holdout", {}) or {})
    if float(holdout.get("auroc", -math.inf)) < float(minimum_auroc) or float(
        holdout.get("probability_gap", -math.inf)
    ) < float(minimum_probability_gap):
        raise RelevanceCalibrationError("Calibration acceptance metrics are too low")
    coefficients = dict(calibration.get("coefficients", {}) or {})
    slope = float(coefficients.get("slope", math.nan))
    intercept = float(coefficients.get("intercept", math.nan))
    parameter_a = float(coefficients.get("a", math.nan))
    parameter_b = float(coefficients.get("b", math.nan))
    if (
        not all(
            math.isfinite(value)
            for value in (intercept, slope, parameter_a, parameter_b)
        )
        or slope <= 0.0
        or parameter_a <= 0.0
        or parameter_a != slope
        or intercept != float(-parameter_a * parameter_b)
        or coefficients.get("formula") != "sigmoid(a*(adjusted_score-b))"
        or coefficients.get("intercept_derivation") != "float64(-a*b)"
    ):
        raise RelevanceCalibrationError("Calibration coefficients are invalid")
    return calibration


def validate_relevance_calibration_source(
    calibration: Mapping[str, Any],
    *,
    source_root: str | Path,
    expected_git_head: str,
    expected_remote_ref: str | None = None,
    expected_remote_head: str | None = None,
    expected_source_file_profile: str | None = None,
) -> dict[str, Any]:
    """Bind a sealed calibration to the immutable checkout consuming it.

    This deliberately checks only paths explicitly sealed in ``source_files``;
    it never discovers files from the repository or sentence-memory bank.
    The launcher separately proves the checkout is clean and pushed.
    """

    root = Path(source_root).resolve()
    source = dict(calibration.get("source", {}) or {})
    git_head = str(expected_git_head).lower()
    if (
        Path(str(source.get("repository_root", ""))).resolve() != root
        or str(source.get("git_head", "")).lower() != git_head
    ):
        raise RelevanceCalibrationError(
            "Calibration source root/commit differs from the consuming checkout"
        )
    if expected_remote_ref is not None and source.get("remote_ref") != str(
        expected_remote_ref
    ):
        raise RelevanceCalibrationError("Calibration remote ref changed")
    if expected_remote_head is not None and (
        str(source.get("remote_head", "")).lower() != str(expected_remote_head).lower()
        or str(expected_remote_head).lower() != git_head
    ):
        raise RelevanceCalibrationError("Calibration lacks exact pushed-source proof")
    raw_profile = source.get("source_file_profile")
    source_file_profile = (
        LEGACY_SOURCE_FILE_PROFILE if raw_profile is None else str(raw_profile)
    )
    if source_file_profile not in SOURCE_FILE_PROFILES:
        raise RelevanceCalibrationError("Unknown calibration source-file profile")
    if (
        expected_source_file_profile is not None
        and source_file_profile != str(expected_source_file_profile)
    ):
        raise RelevanceCalibrationError("Calibration source-file profile changed")
    files = dict(source.get("source_files", {}) or {})
    if set(files) != SOURCE_FILE_PROFILES[source_file_profile]:
        raise RelevanceCalibrationError(
            "Calibration source-file map differs from its exact profile"
        )
    validated: dict[str, Any] = {}
    for relative, recorded in sorted(files.items()):
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RelevanceCalibrationError(
                "Calibration source file escapes its repository root"
            ) from error
        identity = dict(recorded or {})
        if (
            not path.is_file()
            or path.is_symlink()
            or int(identity.get("bytes", -1)) != path.stat().st_size
            or identity.get("sha256") != sha256_file(path)
        ):
            raise RelevanceCalibrationError(
                f"Calibration source file changed: {relative}"
            )
        validated[str(relative)] = identity
    return {
        "git_head": git_head,
        "remote_ref": source.get("remote_ref"),
        "remote_head": str(source.get("remote_head", "")).lower(),
        "repository_root": str(root),
        "source_file_profile": source_file_profile,
        "source_files": validated,
    }
