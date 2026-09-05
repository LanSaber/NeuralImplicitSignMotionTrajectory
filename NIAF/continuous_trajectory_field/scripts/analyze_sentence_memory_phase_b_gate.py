from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_NAME = "signtrajfield_phase_b_scientific_gate"
SCHEMA_VERSION = 2
MODES = ("text_only", "sentence_memory", "shuffled_sentence_memory")
ALIGNMENTS = ("default", "pa")
METRICS = ("dtw", "ndtw", "ndtw_ref")
HAND_DIAGNOSTICS = ("motion_path_error", "motion_path_ratio", "jerk_magnitude_ratio")
MANDATORY_PARTS = ("body", "lhand", "rhand", "wholebody")
OPTIONAL_PARTS = ("face",)
SUBSETS = ("overall", "novel_text", "exact_seen_text")
PRIMARY_SUBSET = "novel_text"
REPORT_PURPOSES = ("phase_b_gate", "test_report")
MEMORY_PARTS = ("body", "lhand", "rhand", "face")


class PhaseBGateInputError(RuntimeError):
    """Raised when gate evidence is incomplete or cannot be paired exactly."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired predicted-length DTW exports and emit a reproducible "
            "Phase-B sentence-memory gate artifact."
        )
    )
    parser.add_argument("--text_export_summary", type=Path, required=True)
    parser.add_argument("--memory_export_summary", type=Path, required=True)
    parser.add_argument("--shuffled_export_summary", type=Path, required=True)
    for mode in MODES:
        parser.add_argument(f"--{mode}_default_dtw", type=Path, required=True)
        parser.add_argument(f"--{mode}_pa_dtw", type=Path, required=True)
    parser.add_argument("--phase_a_checkpoint", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--comparison", default="flow")
    parser.add_argument(
        "--purpose",
        default="phase_b_gate",
        choices=REPORT_PURPOSES,
        help=(
            "phase_b_gate requires validation evidence and may accept Phase B; "
            "test_report requires test evidence and is always non-authorizing."
        ),
    )
    parser.add_argument(
        "--gate_metric",
        default="ndtw",
        choices=METRICS,
        help="PA whole-body metric used for the accept/reject decision.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=2,
        help="Minimum paired novel-text validation samples required by the gate.",
    )
    parser.add_argument("--duration_tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--hand_path_max_relative_degradation", type=float, default=0.02
    )
    parser.add_argument("--parity_tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--fail_on_reject",
        action="store_true",
        help="Exit with status 2 after writing the artifact when the gate rejects.",
    )
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


def _sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseBGateInputError(
            f"Could not read JSON evidence {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PhaseBGateInputError(f"Expected a JSON object in {path}")
    return value


def _finite_float(value: Any, *, source: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PhaseBGateInputError(f"{source} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise PhaseBGateInputError(f"{source} is not finite: {result!r}")
    return result


def _index(value: Any, *, source: str) -> str:
    if value is None:
        raise PhaseBGateInputError(f"{source} has no sample index")
    result = str(value)
    if not result:
        raise PhaseBGateInputError(f"{source} has an empty sample index")
    return result


def _load_export_summary(
    path: str | Path,
    *,
    expected_mode: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PhaseBGateInputError(f"{path} has no non-empty rows list")
    split = str(payload.get("split", ""))
    if split not in {"val", "test"}:
        raise PhaseBGateInputError(
            f"{path} is split={payload.get('split')!r}; expected 'val' or 'test'"
        )
    if str(payload.get("length_mode", "")) != "predicted":
        raise PhaseBGateInputError(
            f"{path} is not a predicted-length export: "
            f"length_mode={payload.get('length_mode')!r}"
        )
    actual_mode = str(payload.get("sentence_memory_mode", ""))
    if actual_mode != expected_mode:
        raise PhaseBGateInputError(
            f"{path} has sentence_memory_mode={actual_mode!r}, expected {expected_mode!r}"
        )

    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PhaseBGateInputError(f"{path}: rows[{row_number}] is not an object")
        sample_index = _index(row.get("index"), source=f"{path}: rows[{row_number}]")
        if sample_index in indexed:
            raise PhaseBGateInputError(f"{path} repeats sample index {sample_index!r}")
        duration = _finite_float(
            row.get("predicted_duration_seconds"),
            source=f"{path}: sample {sample_index} predicted_duration_seconds",
        )
        subset = str(row.get("sentence_memory_text_subset", ""))
        allowed_subsets = {"novel_text", "exact_seen_text"}
        if expected_mode == "off":
            # A text-only v3 export deliberately avoids opening the external
            # sentence bank. The retrieval-on export is the authoritative
            # source for the paired query's seen/novel stratum.
            allowed_subsets.add("not_available")
        if subset not in allowed_subsets:
            raise PhaseBGateInputError(
                f"{path}: sample {sample_index} has invalid sentence-memory subset {subset!r}"
            )
        memory_diagnostics = _optional_memory_diagnostics(
            row.get("sentence_memory_diagnostics"),
            source=f"{path}: sample {sample_index} sentence_memory_diagnostics",
        )
        indexed[sample_index] = {
            **row,
            "index": sample_index,
            "predicted_duration_seconds": duration,
            "sentence_memory_text_subset": subset,
            "sentence_memory_diagnostics": memory_diagnostics,
        }
    if int(payload.get("num_exported", len(rows))) != len(rows):
        raise PhaseBGateInputError(
            f"{path}: num_exported does not equal the number of rows"
        )
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise PhaseBGateInputError(
            f"{path}: export has no canonical-manifest coverage evidence"
        )
    if manifest.get("is_complete_canonical_manifest") is not True:
        raise PhaseBGateInputError(
            f"{path}: export does not cover the complete configured {split} manifest"
        )
    canonical_count = int(manifest.get("canonical_sample_count", -1))
    if canonical_count != len(rows):
        raise PhaseBGateInputError(
            f"{path}: canonical manifest has {canonical_count} rows but export has "
            f"{len(rows)}"
        )
    return payload, indexed


def _optional_memory_diagnostics(value: Any, *, source: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PhaseBGateInputError(f"{source} must be an object when present")
    available = value.get("available")
    if not isinstance(available, bool):
        raise PhaseBGateInputError(f"{source}.available must be boolean")
    gate_values = value.get("gate_mean_by_part")
    if not isinstance(gate_values, Mapping):
        raise PhaseBGateInputError(f"{source}.gate_mean_by_part must be an object")
    gates = {
        part: _finite_float(
            gate_values.get(part), source=f"{source}.gate_mean_by_part.{part}"
        )
        for part in MEMORY_PARTS
    }
    null_mass = _finite_float(
        value.get("null_mass_mean"), source=f"{source}.null_mass_mean"
    )
    for label, diagnostic_value in (("null_mass_mean", null_mass), *gates.items()):
        if not -1e-6 <= diagnostic_value <= 1.0 + 1e-6:
            raise PhaseBGateInputError(
                f"{source}.{label} must lie in [0,1], got {diagnostic_value}"
            )
    candidate_mass_value = value.get("candidate_mass_mean")
    candidate_mass = (
        None
        if candidate_mass_value is None
        else _finite_float(candidate_mass_value, source=f"{source}.candidate_mass_mean")
    )
    if candidate_mass is not None and not -1e-6 <= candidate_mass <= 1.0 + 1e-6:
        raise PhaseBGateInputError(
            f"{source}.candidate_mass_mean must lie in [0,1], got {candidate_mass}"
        )
    return {
        "available": available,
        "gate_mean_by_part": gates,
        "null_mass_mean": null_mass,
        "candidate_mass_mean": candidate_mass,
    }


def _load_dtw_rows(
    path: str | Path,
    *,
    expected_alignment: str,
    comparison: str,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]], set[str]]:
    payload = _load_json(path)
    actual_alignment = str(payload.get("alignment_mode", ""))
    if actual_alignment != expected_alignment:
        raise PhaseBGateInputError(
            f"{path} has alignment_mode={actual_alignment!r}, "
            f"expected {expected_alignment!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PhaseBGateInputError(f"{path} has no non-empty rows list")

    indexed: dict[str, dict[str, dict[str, float]]] = {}
    parts: set[str] = set()
    selected_count = 0
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict) or str(row.get("comparison", "")) != comparison:
            continue
        selected_count += 1
        sample_index = _index(row.get("index"), source=f"{path}: rows[{row_number}]")
        part = str(row.get("part", "")).lower()
        if part not in set(MANDATORY_PARTS + OPTIONAL_PARTS):
            raise PhaseBGateInputError(
                f"{path}: unsupported part {part!r} for sample {sample_index}"
            )
        if part in indexed.setdefault(sample_index, {}):
            raise PhaseBGateInputError(
                f"{path} repeats comparison={comparison!r}, sample={sample_index!r}, "
                f"part={part!r}"
            )
        indexed[sample_index][part] = {
            metric: _finite_float(
                row.get(metric),
                source=f"{path}: sample {sample_index} {part}/{metric}",
            )
            for metric in METRICS
        }
        if part in {"lhand", "rhand"}:
            indexed[sample_index][part].update(
                {
                    metric: _finite_float(
                        row.get(metric),
                        source=f"{path}: sample {sample_index} {part}/{metric}",
                    )
                    for metric in HAND_DIAGNOSTICS
                }
            )
        parts.add(part)
    if selected_count == 0:
        raise PhaseBGateInputError(f"{path} has no rows for comparison={comparison!r}")
    missing_parts = sorted(set(MANDATORY_PARTS) - parts)
    if missing_parts:
        raise PhaseBGateInputError(f"{path} is missing mandatory parts {missing_parts}")
    expected_parts = parts
    for sample_index, part_rows in indexed.items():
        if set(part_rows) != expected_parts:
            raise PhaseBGateInputError(
                f"{path}: sample {sample_index!r} has parts {sorted(part_rows)}, "
                f"expected {sorted(expected_parts)}"
            )
    return payload, indexed, parts


def _same_indices(named_rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    names = list(named_rows)
    if not names:
        raise PhaseBGateInputError("No paired evidence was supplied")
    expected = set(named_rows[names[0]])
    if not expected:
        raise PhaseBGateInputError("Paired evidence contains zero samples")
    for name in names[1:]:
        actual = set(named_rows[name])
        if actual != expected:
            missing = sorted(expected - actual)[:10]
            extra = sorted(actual - expected)[:10]
            raise PhaseBGateInputError(
                f"Sample indices do not pair exactly for {name}: "
                f"missing={missing}, extra={extra}"
            )
    return sorted(expected)


def _stable_seed(base_seed: int, *labels: str) -> int:
    token = "|".join((str(int(base_seed)), *map(str, labels)))
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)


def paired_bootstrap_interval(
    differences: Sequence[float] | np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return {
            "count": 0,
            "mean_difference": None,
            "confidence": float(confidence),
            "lower": None,
            "upper": None,
            "bootstrap_samples": int(samples),
            "seed": int(seed),
        }
    if not bool(np.isfinite(values).all()):
        raise PhaseBGateInputError("Bootstrap differences contain non-finite values")
    if int(samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    chunk_size = min(512, int(samples))
    for start in range(0, int(samples), chunk_size):
        end = min(start + chunk_size, int(samples))
        selections = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[selections].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "count": int(len(values)),
        "mean_difference": float(values.mean()),
        "confidence": float(confidence),
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def _mode_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "median": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
    }


def _memory_diagnostics_report(
    export_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    subset_indices: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    missing_by_mode: dict[str, int] = {}
    for mode in MODES:
        rows = export_rows[mode]
        missing = sum(
            rows[index].get("sentence_memory_diagnostics") is None
            for index in subset_indices["overall"]
        )
        missing_by_mode[mode] = int(missing)
        modes[mode] = {}
        for subset in SUBSETS:
            selected = [
                rows[index]["sentence_memory_diagnostics"]
                for index in subset_indices[subset]
                if rows[index].get("sentence_memory_diagnostics") is not None
            ]
            total = len(subset_indices[subset])
            modes[mode][subset] = {
                "rows": int(total),
                "rows_with_diagnostics": int(len(selected)),
                "coverage_fraction": float(len(selected) / max(total, 1)),
                "available_fraction": (
                    float(np.mean([row["available"] for row in selected]))
                    if selected
                    else None
                ),
                "gate_mean_by_part": {
                    part: _mode_summary(
                        np.asarray(
                            [row["gate_mean_by_part"][part] for row in selected],
                            dtype=np.float64,
                        )
                    )
                    for part in MEMORY_PARTS
                },
                "null_mass_mean": _mode_summary(
                    np.asarray(
                        [row["null_mass_mean"] for row in selected],
                        dtype=np.float64,
                    )
                ),
                "candidate_mass_mean": _mode_summary(
                    np.asarray(
                        [
                            row["candidate_mass_mean"]
                            for row in selected
                            if row.get("candidate_mass_mean") is not None
                        ],
                        dtype=np.float64,
                    )
                ),
            }
    complete = all(count == 0 for count in missing_by_mode.values())
    return {
        "available": bool(complete),
        "missing_rows_by_mode": missing_by_mode,
        "warning": (
            None
            if complete
            else (
                "One or more export summaries lack sentence_memory_diagnostics; "
                "gate/null aggregates are partial or unavailable. Re-export with "
                "the current exporter to populate them."
            )
        ),
        "modes": modes,
    }


def _metric_report(
    values: Mapping[str, np.ndarray],
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
    labels: Sequence[str],
) -> dict[str, Any]:
    memory_minus_text = values["sentence_memory"] - values["text_only"]
    memory_minus_shuffled = (
        values["sentence_memory"] - values["shuffled_sentence_memory"]
    )
    text_seed = _stable_seed(bootstrap_seed, *labels, "memory_minus_text")
    shuffled_seed = _stable_seed(bootstrap_seed, *labels, "memory_minus_shuffled")
    return {
        "modes": {name: _mode_summary(values[name]) for name in MODES},
        "paired_differences": {
            "memory_minus_text": paired_bootstrap_interval(
                memory_minus_text,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=text_seed,
            ),
            "memory_minus_shuffled": paired_bootstrap_interval(
                memory_minus_shuffled,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=shuffled_seed,
            ),
        },
    }


def _extract_bank_id(summary: Mapping[str, Any], *, source: str) -> str:
    memory = summary.get("sentence_memory")
    bank_id = memory.get("bank_id") if isinstance(memory, Mapping) else None
    if not bank_id:
        raise PhaseBGateInputError(f"{source} has no sentence-memory bank_id")
    return str(bank_id)


def _check_duration_invariance(
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    indices: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    if float(tolerance) < 0.0:
        raise ValueError("duration_tolerance must be non-negative")
    text = rows["text_only"]
    comparisons = {}
    all_passed = True
    for mode in ("sentence_memory", "shuffled_sentence_memory"):
        differences = np.asarray(
            [
                abs(
                    float(rows[mode][index]["predicted_duration_seconds"])
                    - float(text[index]["predicted_duration_seconds"])
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        maximum = float(differences.max(initial=0.0))
        passed = bool(maximum <= float(tolerance))
        all_passed = all_passed and passed
        comparisons[f"{mode}_vs_text_only"] = {
            "passed": passed,
            "maximum_absolute_difference_seconds": maximum,
            "mean_absolute_difference_seconds": float(differences.mean()),
        }
    return {
        "evaluated": True,
        "passed": bool(all_passed),
        "tolerance_seconds": float(tolerance),
        "comparisons": comparisons,
    }


def _first_metric(metrics: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if name in metrics and metrics[name] is not None:
            return _finite_float(metrics[name], source=f"checkpoint metric {name}")
    return None


def _extract_parity(checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    for name in (
        "v2_to_v3_text_only_parity",
        "base_checkpoint_parity",
        "text_only_parity",
    ):
        value = checkpoint.get(name)
        if isinstance(value, Mapping):
            return dict(value)
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        safety = config.get("sentence_memory_safety")
        if isinstance(safety, Mapping):
            for name in ("v2_to_v3_text_only_parity", "base_checkpoint_parity"):
                value = safety.get(name)
                if isinstance(value, Mapping):
                    return dict(value)
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    prediction = _first_metric(
        metrics,
        (
            "v2_to_v3_text_only_parity/prediction_max_abs",
            "base_checkpoint_parity/prediction_max_abs",
            "text_only_parity/prediction_max_abs",
            "v2_to_v3_text_only_prediction_max_abs",
        ),
    )
    duration = _first_metric(
        metrics,
        (
            "v2_to_v3_text_only_parity/duration_max_abs",
            "base_checkpoint_parity/duration_max_abs",
            "text_only_parity/duration_max_abs",
            "v2_to_v3_text_only_duration_max_abs",
        ),
    )
    if prediction is None:
        return None
    return {"prediction_max_abs": prediction, "duration_max_abs": duration}


def _checkpoint_evidence(
    path: Path | None,
    *,
    expected_path: str,
    expected_bank_id: str,
    hand_tolerance: float,
    parity_tolerance: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if path is None:
        skipped_required = {
            "evaluated": False,
            "passed": None,
            "required_for_acceptance": True,
            "reason": "--phase_a_checkpoint was not supplied",
        }
        skipped_diagnostic = {
            **skipped_required,
            "required_for_acceptance": False,
        }
        return {"supplied": False}, {
            "phase_a_checkpoint_contract": dict(skipped_required),
            "checkpoint_overall_hand_path_diagnostic": dict(skipped_diagnostic),
            "stored_v2_parity": dict(skipped_required),
        }
    if hand_tolerance < 0.0:
        raise ValueError("hand_path_max_relative_degradation must be non-negative")
    if parity_tolerance < 0.0:
        raise ValueError("parity_tolerance must be non-negative")
    path = Path(path).resolve()
    if not path.is_file():
        raise PhaseBGateInputError(f"Phase-A checkpoint does not exist: {path}")
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        raise PhaseBGateInputError(
            f"Could not load Phase-A checkpoint {path}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise PhaseBGateInputError(f"Phase-A checkpoint {path} is not a mapping")

    summary_checkpoint_matches = False
    try:
        summary_checkpoint_matches = Path(expected_path).resolve() == path
    except (OSError, RuntimeError):
        summary_checkpoint_matches = str(path) == str(expected_path)
    checkpoint_cfg = checkpoint.get("config")
    checkpoint_cfg = checkpoint_cfg if isinstance(checkpoint_cfg, Mapping) else {}
    phase_b_cfg = (
        checkpoint_cfg.get("sentence_memory_safety", {}).get("phase_b", {})
        if isinstance(checkpoint_cfg.get("sentence_memory_safety", {}), Mapping)
        else {}
    )
    train_cfg = checkpoint_cfg.get("train", {})
    train_cfg = train_cfg if isinstance(train_cfg, Mapping) else {}
    identity = checkpoint.get("sentence_memory_identity")
    checkpoint_bank_id = (
        str(identity.get("bank_id"))
        if isinstance(identity, Mapping) and identity.get("bank_id")
        else None
    )
    contract_passed = bool(
        summary_checkpoint_matches
        and checkpoint.get("model_type")
        == "sentence_memory_continuous_trajectory_field"
        and int(checkpoint.get("trajectory_contract_version", -1)) == 3
        and phase_b_cfg.get("enabled") is False
        and bool(train_cfg.get("freeze_base", False))
        and not train_cfg.get("unfreeze_base_prefixes")
        and checkpoint_bank_id == expected_bank_id
    )
    contract_check = {
        "evaluated": True,
        "passed": contract_passed,
        "required_for_acceptance": True,
        "summary_checkpoint_matches": summary_checkpoint_matches,
        "model_type": checkpoint.get("model_type"),
        "trajectory_contract_version": checkpoint.get("trajectory_contract_version"),
        "phase_b_enabled": phase_b_cfg.get("enabled"),
        "freeze_base": train_cfg.get("freeze_base"),
        "unfreeze_base_prefixes": train_cfg.get("unfreeze_base_prefixes"),
        "checkpoint_bank_id": checkpoint_bank_id,
        "expected_bank_id": expected_bank_id,
    }

    checkpoint_metrics = checkpoint.get("metrics")
    checkpoint_metrics = (
        checkpoint_metrics if isinstance(checkpoint_metrics, Mapping) else {}
    )
    hand_rows: dict[str, Any] = {}
    hand_passed = True
    for hand in ("lhand", "rhand"):
        text_value = _first_metric(
            checkpoint_metrics,
            (
                f"val_text_only/pred_loss_path_{hand}",
                f"text_only/pred_loss_path_{hand}",
            ),
        )
        memory_value = _first_metric(
            checkpoint_metrics,
            (
                f"val_sentence_memory/pred_loss_path_{hand}",
                f"sentence_memory/pred_loss_path_{hand}",
            ),
        )
        if text_value is None or memory_value is None:
            passed = False
            allowed = None
            relative = None
        else:
            scale = max(abs(text_value), 1e-12)
            allowed = text_value + float(hand_tolerance) * scale
            relative = (memory_value - text_value) / scale
            passed = bool(memory_value <= allowed)
        hand_passed = hand_passed and passed
        hand_rows[hand] = {
            "passed": passed,
            "text_only": text_value,
            "sentence_memory": memory_value,
            "allowed_sentence_memory": allowed,
            "relative_degradation": relative,
        }
    hand_check = {
        "evaluated": True,
        "passed": bool(hand_passed),
        "required_for_acceptance": False,
        "subset": "overall_checkpoint_aggregate",
        "reason": (
            "Checkpoint validation losses are aggregate-only and cannot establish "
            "the novel-text success criterion. The paired export check is authoritative."
        ),
        "maximum_relative_degradation": float(hand_tolerance),
        "hands": hand_rows,
    }

    parity = _extract_parity(checkpoint)
    if parity is None:
        parity_check = {
            "evaluated": True,
            "passed": False,
            "required_for_acceptance": True,
            "reason": "checkpoint has no stored v2-to-v3 text-only parity evidence",
            "tolerance": float(parity_tolerance),
        }
    else:
        prediction_error = _finite_float(
            parity.get("prediction_max_abs"),
            source="checkpoint parity prediction_max_abs",
        )
        duration_value = parity.get("duration_max_abs")
        duration_error = (
            None
            if duration_value is None
            else _finite_float(
                duration_value, source="checkpoint parity duration_max_abs"
            )
        )
        recorded_passed = parity.get("passed")
        passed = bool(
            prediction_error <= float(parity_tolerance)
            and (duration_error is None or duration_error <= float(parity_tolerance))
            and recorded_passed is not False
        )
        parity_check = {
            "evaluated": True,
            "passed": passed,
            "required_for_acceptance": True,
            "prediction_max_abs": prediction_error,
            "duration_max_abs": duration_error,
            "recorded_passed": recorded_passed,
            "tolerance": float(parity_tolerance),
        }

    checkpoint_provenance = {
        "supplied": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "bank_id": checkpoint_bank_id,
    }
    return checkpoint_provenance, {
        "phase_a_checkpoint_contract": contract_check,
        "checkpoint_overall_hand_path_diagnostic": hand_check,
        "stored_v2_parity": parity_check,
    }


def analyze_phase_b_gate(
    *,
    export_paths: Mapping[str, str | Path],
    dtw_paths: Mapping[str, Mapping[str, str | Path]],
    phase_a_checkpoint: str | Path | None = None,
    purpose: str = "phase_b_gate",
    comparison: str = "flow",
    gate_metric: str = "ndtw",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 1234,
    confidence: float = 0.95,
    min_pairs: int = 2,
    duration_tolerance: float = 1e-7,
    hand_path_max_relative_degradation: float = 0.02,
    parity_tolerance: float = 1e-7,
) -> dict[str, Any]:
    if purpose not in REPORT_PURPOSES:
        raise ValueError(f"purpose must be one of {REPORT_PURPOSES}")
    if gate_metric not in METRICS:
        raise ValueError(f"gate_metric must be one of {METRICS}")
    if int(min_pairs) < 1:
        raise ValueError("min_pairs must be positive")

    expected_export_modes = {
        "text_only": "off",
        "sentence_memory": "on",
        "shuffled_sentence_memory": "shuffled",
    }
    export_payloads: dict[str, dict[str, Any]] = {}
    export_rows: dict[str, dict[str, dict[str, Any]]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        if mode not in export_paths:
            raise PhaseBGateInputError(f"Missing export summary for {mode}")
        path = Path(export_paths[mode])
        payload, rows = _load_export_summary(
            path, expected_mode=expected_export_modes[mode]
        )
        export_payloads[mode] = payload
        export_rows[mode] = rows
        source_files[f"{mode}_export_summary"] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }

    indices = _same_indices(export_rows)
    reference_rows = export_rows["sentence_memory"]
    for index in indices:
        reference_name = str(reference_rows[index].get("name", ""))
        reference_text = str(reference_rows[index].get("text", ""))
        reference_subset = reference_rows[index]["sentence_memory_text_subset"]
        for mode in MODES:
            row = export_rows[mode][index]
            if (
                str(row.get("name", "")) != reference_name
                or str(row.get("text", "")) != reference_text
                or row["sentence_memory_text_subset"]
                not in {reference_subset, "not_available"}
            ):
                raise PhaseBGateInputError(
                    f"Export summaries disagree on identity/subset for sample {index!r}"
                )

    split_values = {str(payload.get("split")) for payload in export_payloads.values()}
    checkpoint_values = {
        str(payload.get("checkpoint", "")) for payload in export_payloads.values()
    }
    bank_ids = {
        _extract_bank_id(payload, source=f"{mode} export summary")
        for mode, payload in export_payloads.items()
    }
    if len(split_values) != 1:
        raise PhaseBGateInputError(
            f"Export summaries have inconsistent splits: {split_values}"
        )
    evidence_split = next(iter(split_values))
    required_split = "val" if purpose == "phase_b_gate" else "test"
    if evidence_split != required_split:
        raise PhaseBGateInputError(
            f"purpose={purpose!r} requires split={required_split!r}, "
            f"got {evidence_split!r}"
        )
    if len(checkpoint_values) != 1 or "" in checkpoint_values:
        raise PhaseBGateInputError(
            f"Export summaries have inconsistent checkpoints: {checkpoint_values}"
        )
    if len(bank_ids) != 1:
        raise PhaseBGateInputError(
            f"Export summaries have inconsistent sentence-memory bank IDs: {bank_ids}"
        )
    checkpoint_path_from_export = next(iter(checkpoint_values))
    bank_id = next(iter(bank_ids))

    dtw_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    dtw_rows: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    common_parts: set[str] | None = None
    all_indexed_evidence: dict[str, Mapping[str, Any]] = dict(export_rows)
    for mode in MODES:
        if mode not in dtw_paths:
            raise PhaseBGateInputError(f"Missing DTW evidence for {mode}")
        dtw_payloads[mode] = {}
        dtw_rows[mode] = {}
        for alignment in ALIGNMENTS:
            if alignment not in dtw_paths[mode]:
                raise PhaseBGateInputError(
                    f"Missing {alignment} DTW evidence for {mode}"
                )
            path = Path(dtw_paths[mode][alignment])
            payload, rows, parts = _load_dtw_rows(
                path,
                expected_alignment=alignment,
                comparison=comparison,
            )
            dtw_payloads[mode][alignment] = payload
            dtw_rows[mode][alignment] = rows
            all_indexed_evidence[f"{mode}_{alignment}_dtw"] = rows
            source_files[f"{mode}_{alignment}_dtw"] = {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "metric_preset": payload.get("metric_preset"),
                "betas_mode": payload.get("betas_mode"),
            }
            if common_parts is None:
                common_parts = set(parts)
            elif set(parts) != common_parts:
                raise PhaseBGateInputError(
                    "DTW files do not report the same parts: "
                    f"expected {sorted(common_parts)}, got {sorted(parts)} in {path}"
                )
    _same_indices(all_indexed_evidence)
    assert common_parts is not None

    for alignment in ALIGNMENTS:
        presets = {
            str(dtw_payloads[mode][alignment].get("metric_preset", ""))
            for mode in MODES
        }
        betas_modes = {
            str(dtw_payloads[mode][alignment].get("betas_mode", "")) for mode in MODES
        }
        if len(presets) != 1 or "" in presets:
            raise PhaseBGateInputError(
                f"{alignment} DTW files use inconsistent metric presets: {presets}"
            )
        if len(betas_modes) != 1 or "" in betas_modes:
            raise PhaseBGateInputError(
                f"{alignment} DTW files use inconsistent betas modes: {betas_modes}"
            )
    for mode in MODES:
        for index in indices:
            for hand in ("lhand", "rhand"):
                for metric in HAND_DIAGNOSTICS:
                    default_value = dtw_rows[mode]["default"][index][hand][metric]
                    pa_value = dtw_rows[mode]["pa"][index][hand][metric]
                    if not math.isclose(
                        default_value,
                        pa_value,
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ):
                        raise PhaseBGateInputError(
                            "Unwarped hand diagnostic differs between default and "
                            f"PA files: mode={mode}, sample={index}, hand={hand}, "
                            f"metric={metric}"
                        )

    subset_indices = {
        "overall": list(indices),
        "novel_text": [
            index
            for index in indices
            if reference_rows[index]["sentence_memory_text_subset"] == "novel_text"
        ],
        "exact_seen_text": [
            index
            for index in indices
            if reference_rows[index]["sentence_memory_text_subset"] == "exact_seen_text"
        ],
    }
    memory_diagnostics = _memory_diagnostics_report(export_rows, subset_indices)
    part_order = [part for part in MANDATORY_PARTS if part in common_parts]
    if "face" in common_parts:
        part_order.insert(-1, "face")

    metrics_report: dict[str, Any] = {}
    for subset in SUBSETS:
        metrics_report[subset] = {}
        selected_indices = subset_indices[subset]
        for alignment in ALIGNMENTS:
            metrics_report[subset][alignment] = {}
            for part in part_order:
                metrics_report[subset][alignment][part] = {}
                for metric in METRICS:
                    arrays = {
                        mode: np.asarray(
                            [
                                dtw_rows[mode][alignment][index][part][metric]
                                for index in selected_indices
                            ],
                            dtype=np.float64,
                        )
                        for mode in MODES
                    }
                    metrics_report[subset][alignment][part][metric] = _metric_report(
                        arrays,
                        bootstrap_samples=int(bootstrap_samples),
                        confidence=float(confidence),
                        bootstrap_seed=int(bootstrap_seed),
                        labels=(subset, alignment, part, metric),
                    )
        metrics_report[subset]["unwarped"] = {}
        for hand in ("lhand", "rhand"):
            metrics_report[subset]["unwarped"][hand] = {}
            for metric in HAND_DIAGNOSTICS:
                arrays = {
                    mode: np.asarray(
                        [
                            dtw_rows[mode]["default"][index][hand][metric]
                            for index in selected_indices
                        ],
                        dtype=np.float64,
                    )
                    for mode in MODES
                }
                metrics_report[subset]["unwarped"][hand][metric] = _metric_report(
                    arrays,
                    bootstrap_samples=int(bootstrap_samples),
                    confidence=float(confidence),
                    bootstrap_seed=int(bootstrap_seed),
                    labels=(subset, "unwarped", hand, metric),
                )

    primary_indices = subset_indices[PRIMARY_SUBSET]
    primary = metrics_report[PRIMARY_SUBSET]["pa"]["wholebody"][gate_metric]
    primary_text = primary["paired_differences"]["memory_minus_text"]
    primary_shuffled = primary["paired_differences"]["memory_minus_shuffled"]
    enough_pairs = len(primary_indices) >= int(min_pairs)
    predicted_hand_rows = {}
    predicted_hand_passed = True
    for hand in ("lhand", "rhand"):
        report = metrics_report[PRIMARY_SUBSET]["unwarped"][hand]["motion_path_error"]
        text_value = report["modes"]["text_only"]["mean"]
        memory_value = report["modes"]["sentence_memory"]["mean"]
        if text_value is None or memory_value is None:
            scale = None
            allowed = None
            relative = None
            passed = False
        else:
            scale = max(abs(float(text_value)), 1e-12)
            allowed = (
                float(text_value) + float(hand_path_max_relative_degradation) * scale
            )
            relative = (float(memory_value) - float(text_value)) / scale
            passed = bool(float(memory_value) <= allowed)
        predicted_hand_passed = predicted_hand_passed and passed
        predicted_hand_rows[hand] = {
            "passed": passed,
            "text_only": text_value,
            "sentence_memory": memory_value,
            "allowed_sentence_memory": allowed,
            "relative_degradation": relative,
        }
    checks: dict[str, dict[str, Any]] = {
        "minimum_novel_text_paired_samples": {
            "evaluated": True,
            "passed": bool(enough_pairs),
            "required_for_acceptance": True,
            "subset": PRIMARY_SUBSET,
            "count": len(primary_indices),
            "minimum": int(min_pairs),
        },
        "novel_text_pa_wholebody_memory_better_than_text": {
            "evaluated": True,
            "passed": bool(
                primary_text["mean_difference"] is not None
                and primary_text["mean_difference"] < 0.0
            ),
            "required_for_acceptance": True,
            "subset": PRIMARY_SUBSET,
            "metric": gate_metric,
            **primary_text,
        },
        "novel_text_pa_wholebody_memory_vs_text_ci_below_zero": {
            "evaluated": True,
            "passed": bool(
                primary_text["upper"] is not None and primary_text["upper"] < 0.0
            ),
            "required_for_acceptance": True,
            "subset": PRIMARY_SUBSET,
            "metric": gate_metric,
            **primary_text,
        },
        "novel_text_pa_wholebody_memory_better_than_shuffled": {
            "evaluated": True,
            "passed": bool(
                primary_shuffled["mean_difference"] is not None
                and primary_shuffled["mean_difference"] < 0.0
            ),
            "required_for_acceptance": True,
            "subset": PRIMARY_SUBSET,
            "metric": gate_metric,
            **primary_shuffled,
        },
        "duration_invariance": _check_duration_invariance(
            export_rows, indices, float(duration_tolerance)
        )
        | {
            "required_for_acceptance": True,
            "subset": "overall",
        },
        "novel_text_predicted_hand_path_nonregression": {
            "evaluated": True,
            "passed": bool(predicted_hand_passed),
            "required_for_acceptance": True,
            "subset": PRIMARY_SUBSET,
            "count": len(primary_indices),
            "maximum_relative_degradation": float(hand_path_max_relative_degradation),
            "hands": predicted_hand_rows,
        },
        "export_gate_null_diagnostics_available": {
            "evaluated": True,
            "passed": bool(memory_diagnostics["available"]),
            "required_for_acceptance": False,
            "missing_rows_by_mode": memory_diagnostics["missing_rows_by_mode"],
            "warning": memory_diagnostics["warning"],
        },
    }

    checkpoint_provenance, checkpoint_checks = _checkpoint_evidence(
        Path(phase_a_checkpoint) if phase_a_checkpoint is not None else None,
        expected_path=checkpoint_path_from_export,
        expected_bank_id=bank_id,
        hand_tolerance=float(hand_path_max_relative_degradation),
        parity_tolerance=float(parity_tolerance),
    )
    checks.update(checkpoint_checks)
    required_checks = [
        value
        for value in checks.values()
        if bool(value.get("evaluated", False))
        and bool(value.get("required_for_acceptance", True))
    ]
    scientific_criteria_passed = bool(
        required_checks and all(bool(value.get("passed")) for value in required_checks)
    )
    accepted = bool(purpose == "phase_b_gate" and scientific_criteria_passed)

    settings = {
        "comparison": str(comparison),
        "primary_subset": PRIMARY_SUBSET,
        "gate_metric": gate_metric,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "confidence": float(confidence),
        "minimum_pairs": int(min_pairs),
        "duration_tolerance_seconds": float(duration_tolerance),
        "hand_path_max_relative_degradation": float(hand_path_max_relative_degradation),
        "parity_tolerance": float(parity_tolerance),
    }
    decision_identity_payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "purpose": str(purpose),
        "evidence_split": evidence_split,
        "settings": settings,
        "sample_index_sha256": _digest_json(indices),
        "primary_sample_index_sha256": _digest_json(primary_indices),
        "source_sha256": {
            name: value["sha256"] for name, value in sorted(source_files.items())
        },
        "checkpoint_sha256": checkpoint_provenance.get("sha256"),
        "checkpoint_path": checkpoint_path_from_export,
        "bank_id": bank_id,
        "checks": checks,
    }
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "scientific_criteria_passed": scientific_criteria_passed,
        "decision": {
            "purpose": str(purpose),
            "primary_subset": PRIMARY_SUBSET,
            "phase_b_authorizing": bool(purpose == "phase_b_gate"),
            "reason": (
                "validation scientific gate"
                if purpose == "phase_b_gate"
                else "test evidence is report-only and can never authorize Phase B"
            ),
            "required_checks": [
                name
                for name, check in checks.items()
                if bool(check.get("required_for_acceptance", True))
            ],
            "diagnostic_checks": [
                name
                for name, check in checks.items()
                if not bool(check.get("required_for_acceptance", True))
            ],
        },
        "settings": settings,
        "checks": checks,
        "subset_counts": {name: len(values) for name, values in subset_indices.items()},
        "parts": part_order,
        "metrics": metrics_report,
        "sentence_memory_diagnostics": memory_diagnostics,
        "provenance": {
            "source_files": source_files,
            "sample_indices": indices,
            "sample_index_sha256": _digest_json(indices),
            "primary_sample_indices": primary_indices,
            "primary_sample_index_sha256": _digest_json(primary_indices),
            "split": evidence_split,
            "length_mode": "predicted",
            "checkpoint_path_from_exports": checkpoint_path_from_export,
            "checkpoint": checkpoint_provenance,
            "bank_id": bank_id,
        },
        "gate_identity": {
            "algorithm": "sha256-canonical-json",
            "digest": _digest_json(decision_identity_payload),
        },
    }


def main():
    args = parse_args()
    export_paths = {
        "text_only": args.text_export_summary,
        "sentence_memory": args.memory_export_summary,
        "shuffled_sentence_memory": args.shuffled_export_summary,
    }
    dtw_paths = {
        mode: {
            alignment: getattr(args, f"{mode}_{alignment}_dtw")
            for alignment in ALIGNMENTS
        }
        for mode in MODES
    }
    result = analyze_phase_b_gate(
        export_paths=export_paths,
        dtw_paths=dtw_paths,
        phase_a_checkpoint=args.phase_a_checkpoint,
        purpose=args.purpose,
        comparison=args.comparison,
        gate_metric=args.gate_metric,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        min_pairs=args.min_pairs,
        duration_tolerance=args.duration_tolerance,
        hand_path_max_relative_degradation=(args.hand_path_max_relative_degradation),
        parity_tolerance=args.parity_tolerance,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": result["accepted"],
                "scientific_criteria_passed": result["scientific_criteria_passed"],
                "purpose": result["decision"]["purpose"],
                "gate_identity": result["gate_identity"],
                "subset_counts": result["subset_counts"],
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    if (
        args.fail_on_reject
        and args.purpose == "phase_b_gate"
        and not result["accepted"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
