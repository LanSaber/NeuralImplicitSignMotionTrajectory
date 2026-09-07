from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


BOOTSTRAP_SEED = 1234
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

MIN_RELATIVE_PA_NDTW_IMPROVEMENT = 0.005
MIN_RMOTION_POINT = 0.10
MIN_RMOTION_LOWER_CI = 0.05
MAX_HAND_PATH_RELATIVE_DEGRADATION = 0.02
MAX_DURATION_ABS_DIFFERENCE_SECONDS = 1e-7
MAX_INTEGRITY_OUTPUT_ABS_DIFFERENCE = 1e-7
V2_TEXT_ONLY_CHECKPOINT_SHA256 = (
    "06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
)


def stable_subseed(seed: int, label: str) -> int:
    token = hashlib.sha256(f"{int(seed)}|{label}".encode("utf-8")).digest()
    return int.from_bytes(token[:8], byteorder="big", signed=False)


def cluster_bootstrap_indices(
    cluster_count: int,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if int(cluster_count) <= 0:
        raise ValueError("cluster_count must be positive")
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0,
        int(cluster_count),
        size=(int(samples), int(cluster_count)),
        dtype=np.int32,
    )


def confidence_interval(
    values: np.ndarray,
    *,
    confidence: float = CONFIDENCE_LEVEL,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Confidence-interval values must be finite and non-empty")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    tail = 0.5 * (1.0 - float(confidence))
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return [float(lower), float(upper)]


def _paired_arrays(
    correct: np.ndarray,
    reference: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    indices = np.asarray(bootstrap_indices)
    if correct.shape != reference.shape or not len(correct):
        raise ValueError("Paired arrays must be non-empty with identical shapes")
    if not np.isfinite(correct).all() or not np.isfinite(reference).all():
        raise ValueError("Paired arrays contain non-finite values")
    if indices.ndim != 2 or indices.shape[1] != len(correct):
        raise ValueError("bootstrap_indices has the wrong cluster dimension")
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= len(correct):
        raise ValueError("bootstrap_indices contains an out-of-bounds cluster")
    correct_means = correct[indices].mean(axis=1)
    reference_means = reference[indices].mean(axis=1)
    return correct, reference, np.stack((correct_means, reference_means), axis=0)


def paired_lower_is_better(
    correct: np.ndarray,
    reference: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    correct, reference, boot = _paired_arrays(
        correct, reference, bootstrap_indices
    )
    correct_mean = float(correct.mean())
    reference_mean = float(reference.mean())
    denominator = max(abs(reference_mean), 1e-12)
    bootstrap_difference = boot[0] - boot[1]
    bootstrap_relative = (boot[1] - boot[0]) / np.maximum(
        np.abs(boot[1]), 1e-12
    )
    return {
        "correct_mean": correct_mean,
        "reference_mean": reference_mean,
        "absolute_difference": correct_mean - reference_mean,
        "absolute_difference_ci95": confidence_interval(bootstrap_difference),
        "relative_improvement": (reference_mean - correct_mean) / denominator,
        "relative_improvement_ci95": confidence_interval(bootstrap_relative),
        "cluster_differences": correct - reference,
    }


def sign_flip_lower_tail_pvalue(
    values: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """Paired randomization p-value for the alternative mean(values) < 0."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Sign-flip values must be finite and non-empty")
    observed = float(values.mean())
    rng = np.random.default_rng(int(seed))
    extreme = 0
    completed = 0
    chunk = 512
    while completed < int(samples):
        count = min(chunk, int(samples) - completed)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = (signs * values[None, :]).mean(axis=1)
        extreme += int(np.count_nonzero(permuted <= observed))
        completed += count
    return float((extreme + 1) / (int(samples) + 1))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down family-wise adjusted p-values."""

    if not p_values:
        return {}
    rows = []
    for name, raw in p_values.items():
        value = float(raw)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for {name!r}: {raw}")
        rows.append((str(name), value))
    rows.sort(key=lambda row: (row[1], row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(rows)
    for rank, (name, raw) in enumerate(rows):
        running = max(running, (family_size - rank) * raw)
        adjusted[name] = min(float(running), 1.0)
    return adjusted


def rmotion_summary(
    motion_shuffled_mse: np.ndarray,
    memory_off_mse: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    numerator = np.asarray(motion_shuffled_mse, dtype=np.float64).reshape(-1)
    denominator = np.asarray(memory_off_mse, dtype=np.float64).reshape(-1)
    indices = np.asarray(bootstrap_indices)
    if numerator.shape != denominator.shape or not len(numerator):
        raise ValueError("Rmotion arrays must be non-empty with identical shapes")
    if (
        not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
        or (numerator < 0.0).any()
        or (denominator < 0.0).any()
    ):
        raise ValueError("Rmotion squared-error arrays must be finite and non-negative")
    if indices.ndim != 2 or indices.shape[1] != len(numerator):
        raise ValueError("bootstrap_indices has the wrong Rmotion cluster dimension")
    denominator_mean = float(denominator.mean())
    point = float(
        np.sqrt(float(numerator.mean()) / max(denominator_mean, 1e-24))
    )
    bootstrap = np.sqrt(
        numerator[indices].mean(axis=1)
        / np.maximum(denominator[indices].mean(axis=1), 1e-24)
    )
    return {
        "definition": (
            "sqrt(cluster-mean MSE(correct,motion-shuffled) / "
            "cluster-mean MSE(correct,memory-off)) in exported rot6d"
        ),
        "value": point,
        "ci95": confidence_interval(bootstrap),
        "motion_shuffled_mse_mean": float(numerator.mean()),
        "memory_off_mse_mean": denominator_mean,
    }
