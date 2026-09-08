"""Development-only layerwise diagnostics for locked Phase-A''' memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from NIAF.continuous_trajectory_field.phase_a_motion_contrast import (
    cluster_bootstrap_indices,
    confidence_interval,
)
from NIAF.continuous_trajectory_field.sentence_memory import (
    normalize_sentence_text,
    read_jsonl,
    sha256_file,
)
from NIAF.continuous_trajectory_field.scripts.analyze_centered_memory_confirmation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _authorization_evidence,
    _canonical_csv_bytes,
    _digest_json,
    _durable_write_bytes,
    _fsync_directory,
    _prediction_parity,
    _rename_directory_no_replace,
    _sample_path,
    _validate_all_null_export,
    _validate_exact_completed_directory,
    _validate_export_identities,
    _expected_identities,
    validate_centered_control_pair,
)
from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    ConfirmationInputError,
    _cluster_index_groups,
    _cluster_mean,
)
from NIAF.continuous_trajectory_field.scripts.decide_centered_memory_stage import (
    DIAGNOSTIC_READY_SCHEMA_NAME,
    DIAGNOSTIC_SCHEMA_NAME,
    EXPERIMENTS,
    STAGE1,
    STAGE1_EVAL_MODES,
    STAGE2_EVAL_MODES,
)
from NIAF.continuous_trajectory_field.scripts.decide_factorized_memory_stage import (
    EXPECTED_DEVELOPMENT_TEXTS,
    EXPECTED_PARTITION_DIGEST,
    OrderedDecisionError,
    validate_factorized_export_query_binding,
)


SCHEMA_NAME = DIAGNOSTIC_SCHEMA_NAME
SCHEMA_VERSION = 1
PART_NAMES = ("body", "lhand", "rhand", "face")


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfirmationInputError(f"Required JSON does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationInputError(f"Expected a JSON object: {path}")
    return value


def _load_development_manifest(
    partition_dir: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Open only the already-used development manifest, never confirmation."""

    partition = dict(authorization.get("partition", {}) or {})
    if partition.get("partition_digest") != EXPECTED_PARTITION_DIGEST:
        raise ConfirmationInputError("Development partition digest changed")
    path = partition_dir.resolve() / "manifest_development.jsonl"
    authorized_path = Path(str(partition.get("development_manifest", ""))).resolve()
    if (
        path != authorized_path
        or sha256_file(path) != partition.get("development_manifest_sha256")
    ):
        raise ConfirmationInputError("Development manifest differs from authorization")
    rows = read_jsonl(path)
    expected_rows = int(partition.get("development_rows", -1))
    texts = [normalize_sentence_text(row.get("text", "")) for row in rows]
    if (
        len(rows) != expected_rows
        or len(set(texts)) != EXPECTED_DEVELOPMENT_TEXTS
        or any(not value for value in texts)
    ):
        raise ConfirmationInputError("Development manifest counts changed")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "rows": rows,
        "normalized_texts": texts,
    }


def _row_identity(rows: list[Mapping[str, Any]]) -> list[tuple[str, str]]:
    return [
        (str(row.get("name", "")), normalize_sentence_text(row.get("text", "")))
        for row in rows
    ]


def _load_export(
    mode: str,
    directory: Path,
    *,
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    directory = directory.resolve()
    summary_path = directory / "export_summary.json"
    summary = _json(summary_path)
    if (
        summary.get("split") != "val"
        or summary.get("length_mode") != "predicted"
        or summary.get("word_prior_mode") != "off"
        or summary.get("sentence_memory_mode") != mode
        or Path(str(summary.get("checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or Path(str(summary.get("config", ""))).resolve() != config_path.resolve()
    ):
        raise ConfirmationInputError(f"Development export {mode} has wrong inputs")
    rows = list(summary.get("rows", []) or [])
    if len(rows) != len(manifest["rows"]) or _row_identity(rows) != _row_identity(
        manifest["rows"]
    ):
        raise ConfirmationInputError(f"Development export {mode} row identity changed")
    manifest_summary = dict(summary.get("manifest", {}) or {})
    if (
        int(manifest_summary.get("sample_count", -1)) != len(rows)
        or manifest_summary.get("dataset_manifest_sha256") != manifest["sha256"]
        or summary.get("num_exported") != len(rows)
    ):
        raise ConfirmationInputError(f"Development export {mode} is incomplete")
    for row in rows:
        _sample_path({"dir": directory}, row)
    try:
        binding = validate_factorized_export_query_binding(
            summary.get("sentence_memory_query_binding"),
            expected_authority="config_pinned_development_manifest_v1",
            expected_manifest_file="manifest_development.jsonl",
            expected_manifest_sha256=manifest["sha256"],
            expected_query_rows=len(rows),
        )
    except OrderedDecisionError as error:
        raise ConfirmationInputError(str(error)) from error
    durations = np.asarray(
        [row.get("predicted_duration_seconds") for row in rows], dtype=np.float64
    )
    if not np.isfinite(durations).all():
        raise ConfirmationInputError(f"Development export {mode} has non-finite duration")
    return {
        "dir": directory,
        "summary": summary,
        "summary_path": summary_path,
        "rows": rows,
        "durations": durations,
        "query_binding": binding,
    }


def _load_rot6d(export: Mapping[str, Any], index: int) -> np.ndarray:
    with np.load(
        _sample_path(export, export["rows"][index]), allow_pickle=False
    ) as sample:
        value = np.asarray(sample["rot6d"], dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ConfirmationInputError("Development prediction is malformed")
    return value


def _jsd(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    midpoint = 0.5 * (first + second)
    value = 0.0
    for distribution in (first, second):
        positive = distribution > 0.0
        value += 0.5 * float(
            np.sum(
                distribution[positive]
                * np.log(distribution[positive] / midpoint[positive])
            )
        )
    return value


def _layer_rows(export: Mapping[str, Any], mode: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row_index, export_row in enumerate(export["rows"]):
        with np.load(_sample_path(export, export_row), allow_pickle=False) as sample:
            required = (
                "sentence_memory_layer_part_candidate_mass",
                "sentence_memory_layer_part_null_mass",
                "sentence_memory_layer_part_token_mass",
                "sentence_memory_centered_update",
                "sentence_memory_relevance_gate",
                "sentence_memory_candidate_support",
                "sentence_memory_token_tau",
                "trajectory_sentence_memory_gates",
            )
            missing = [name for name in required if name not in sample.files]
            if missing:
                raise ConfirmationInputError(
                    f"Centered layer diagnostic lacks fields {missing}"
                )
            candidate = np.asarray(
                sample["sentence_memory_layer_part_candidate_mass"],
                dtype=np.float64,
            )
            null = np.asarray(
                sample["sentence_memory_layer_part_null_mass"], dtype=np.float64
            )
            token = np.asarray(
                sample["sentence_memory_layer_part_token_mass"], dtype=np.float64
            )
            update = np.asarray(
                sample["sentence_memory_centered_update"], dtype=np.float64
            )
            relevance = np.asarray(
                sample["sentence_memory_relevance_gate"], dtype=np.float64
            )
            support = np.asarray(
                sample["sentence_memory_candidate_support"], dtype=bool
            )
            tau = np.asarray(sample["sentence_memory_token_tau"], dtype=np.float64)
            gates = np.asarray(
                sample["trajectory_sentence_memory_gates"], dtype=np.float64
            )
            association = (
                np.asarray(sample["sentence_memory_association_gate"], dtype=np.float64)
                if "sentence_memory_association_gate" in sample.files
                else None
            )
        if (
            candidate.ndim != 4
            or null.shape != candidate.shape[:-1]
            or token.shape[:4] != candidate.shape
            or update.shape[:3] != candidate.shape[:3]
            or relevance.ndim != 1
            or relevance.shape[0] != candidate.shape[-1]
            or support.shape != candidate.shape[1:]
            or tau.shape != token.shape[-2:]
            or gates.shape != candidate.shape[1:3]
            or (
                association is not None
                and association.shape != relevance.shape
            )
        ):
            raise ConfirmationInputError("Centered layer diagnostic shapes changed")
        if not all(
            np.isfinite(value).all()
            for value in (
                candidate,
                null,
                token,
                update,
                relevance,
                tau,
                gates,
                *(() if association is None else (association,)),
            )
        ):
            raise ConfirmationInputError("Centered layer diagnostic is non-finite")
        if (
            np.any(relevance < 0.0)
            or np.any(relevance > 1.0)
            or (
                association is not None
                and (np.any(association < 0.0) or np.any(association > 1.0))
            )
        ):
            raise ConfirmationInputError("Centered evidence gate is outside [0,1]")
        for layer in range(candidate.shape[0]):
            for slot in range(candidate.shape[1]):
                for part_index, part in enumerate(PART_NAMES):
                    mass = candidate[layer, slot, part_index]
                    real = float(mass.sum())
                    conditional = mass / real if real > 1e-12 else np.zeros_like(mass)
                    entropy = -float(
                        np.sum(
                            conditional
                            * np.log(np.clip(conditional, 1e-12, None))
                        )
                    )
                    token_conditional = (
                        token[layer, slot, part_index] / real
                        if real > 1e-12
                        else np.zeros_like(token[layer, slot, part_index])
                    )
                    expected_tau = float(np.sum(token_conditional * tau))
                    adjacent_jsd = (
                        _jsd(
                            candidate[layer, slot - 1, part_index]
                            / max(
                                float(
                                    candidate[layer, slot - 1, part_index].sum()
                                ),
                                1e-12,
                            ),
                            conditional,
                        )
                        if slot > 0
                        else 0.0
                    )
                    part_jsd_values = []
                    for other_part in range(len(PART_NAMES)):
                        if other_part == part_index:
                            continue
                        other_mass = candidate[layer, slot, other_part]
                        other_conditional = other_mass / max(
                            float(other_mass.sum()), 1e-12
                        )
                        part_jsd_values.append(_jsd(conditional, other_conditional))
                    supported = support[slot, part_index]
                    if association is not None:
                        if np.any(association[supported] <= 0.0):
                            raise ConfirmationInputError(
                                "Supported association gate is not strictly positive"
                            )
                        post_relevance_mass = np.zeros_like(mass)
                        np.divide(
                            mass,
                            association,
                            out=post_relevance_mass,
                            where=supported,
                        )
                        absolute_relevance_mass = float(
                            post_relevance_mass.sum()
                        )
                    else:
                        absolute_relevance_mass = real
                    rows.append(
                        {
                            "row_index": float(row_index),
                            "layer": float(layer),
                            "slot": float(slot),
                            "part_index": float(part_index),
                            "real_candidate_mass": real,
                            "null_mass": float(null[layer, slot, part_index]),
                            "candidate_entropy": entropy,
                            "effective_k": float(np.exp(entropy)),
                            "maximum_candidate_share": float(
                                conditional.max(initial=0.0)
                            ),
                            "expected_memory_tau": expected_tau,
                            "temporal_alignment_abs_error": abs(
                                expected_tau
                                - float(
                                    np.linspace(-1.0, 1.0, candidate.shape[1])[slot]
                                )
                            ),
                            "adjacent_slot_jsd": adjacent_jsd,
                            "part_jsd": float(np.mean(part_jsd_values)),
                            "centered_update_rms": float(
                                np.sqrt(np.square(update[layer, slot, part_index]).mean())
                            ),
                            "fusion_gate": float(gates[slot, part_index]),
                            "relevance_gate_mean": float(
                                relevance[supported].mean() if supported.any() else 0.0
                            ),
                            "absolute_relevance_mass": float(
                                absolute_relevance_mass
                            ),
                            "association_gate_mean": float(
                                association[supported].mean()
                                if association is not None and supported.any()
                                else 1.0
                            ),
                            "absolute_association_mass": float(
                                real
                            ),
                        }
                    )
    return rows


def _aggregate_layer_rows(
    rows: list[Mapping[str, float]],
    *,
    mode: str,
    groups: list[np.ndarray],
    bootstrap: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    metric_names = [
        name
        for name in rows[0]
        if name not in {"row_index", "layer", "slot", "part_index"}
    ]
    layers = sorted({int(row["layer"]) for row in rows})
    for layer in layers:
        for part_index, part in enumerate(PART_NAMES):
            selected = [
                row
                for row in rows
                if int(row["layer"]) == layer
                and int(row["part_index"]) == part_index
            ]
            for metric in metric_names:
                row_values = np.zeros(sum(len(group) for group in groups), dtype=np.float64)
                counts = np.zeros_like(row_values)
                for row in selected:
                    index = int(row["row_index"])
                    row_values[index] += float(row[metric])
                    counts[index] += 1.0
                if np.any(counts == 0):
                    raise ConfirmationInputError("Layer diagnostic lost a query row")
                cluster = _cluster_mean(row_values / counts, groups)
                output.append(
                    {
                        "mode": mode,
                        "layer": layer,
                        "part": part,
                        "metric": metric,
                        "mean": float(cluster.mean()),
                        "ci95_low": float(
                            confidence_interval(cluster[bootstrap].mean(axis=1))[0]
                        ),
                        "ci95_high": float(
                            confidence_interval(cluster[bootstrap].mean(axis=1))[1]
                        ),
                    }
                )
    return output


def _centered_update_sensitivity(
    correct: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    groups: list[np.ndarray],
    bootstrap: np.ndarray,
) -> dict[str, Any]:
    per_row = []
    for index in range(len(correct["rows"])):
        with np.load(
            _sample_path(correct, correct["rows"][index]), allow_pickle=False
        ) as first, np.load(
            _sample_path(control, control["rows"][index]), allow_pickle=False
        ) as second:
            name = "sentence_memory_centered_update"
            if name not in first.files or name not in second.files:
                raise ConfirmationInputError("Centered-update diagnostic is absent")
            first_update = np.asarray(first[name], dtype=np.float64)
            second_update = np.asarray(second[name], dtype=np.float64)
        if first_update.shape != second_update.shape or first_update.ndim != 4:
            raise ConfirmationInputError("Centered-update control shapes changed")
        per_row.append(np.square(first_update - second_update).mean(axis=(1, 2, 3)))
    values = np.asarray(per_row, dtype=np.float64)
    result = {}
    for layer in range(values.shape[1]):
        cluster = _cluster_mean(values[:, layer], groups)
        root_bootstrap = np.sqrt(cluster[bootstrap].mean(axis=1))
        result[str(layer)] = {
            "rms": float(np.sqrt(cluster.mean())),
            "ci95": confidence_interval(root_bootstrap),
        }
    return result


def analyze_diagnostic(
    *,
    authorization_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    partition_dir: Path,
    mode_dirs: Mapping[str, Path],
    broadcast_dir: Path,
    all_null_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if bootstrap_samples != BOOTSTRAP_SAMPLES or seed != BOOTSTRAP_SEED:
        raise ConfirmationInputError("Diagnostic requires 10,000/1234 bootstrap")
    stage, authorization = _authorization_evidence(
        authorization_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        purpose="diagnostic",
    )
    if (
        authorization.get("confirmation_manifest_opened") is not False
        or authorization.get("test_data_accessed") is not False
    ):
        raise ConfirmationInputError("Diagnostic authorization is not pre-confirmation")
    expected_modes = STAGE1_EVAL_MODES if stage == STAGE1 else STAGE2_EVAL_MODES
    if tuple(mode_dirs) != tuple(expected_modes):
        raise ConfirmationInputError("Diagnostic mode order changed")
    manifest = _load_development_manifest(partition_dir, authorization)
    exports = {
        mode: _load_export(
            mode,
            mode_dirs[mode],
            manifest=manifest,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        for mode in expected_modes
    }
    broadcast = _load_export(
        "broadcast_complete",
        broadcast_dir,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    all_null = _load_export(
        "all_null",
        all_null_dir,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    _validate_export_identities(
        {**exports, "broadcast_complete": broadcast, "all_null": all_null},
        expected=_expected_identities(authorization),
    )
    controls = {
        mode: validate_centered_control_pair(exports["on"], exports[mode], mode=mode)
        for mode in expected_modes
        if mode not in {"off", "on", "full_replacement"}
    }
    controls["full_replacement"] = validate_centered_control_pair(
        exports["on"], exports["full_replacement"], mode="full_replacement"
    )
    controls["broadcast_complete"] = validate_centered_control_pair(
        exports["on"], broadcast, mode="broadcast_complete"
    )
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
    }
    parity["joint_tuple_vs_correct"]["passed"] = bool(
        parity["joint_tuple_vs_correct"]["prediction_max_abs"] <= 1e-7
        and parity["joint_tuple_vs_correct"]["duration_max_abs"] <= 1e-7
    )
    all_null_diagnostics = _validate_all_null_export(all_null)
    all_null_diagnostics["passed"] = bool(
        all_null_diagnostics["gate_max_abs"] == 0.0
        and all_null_diagnostics["candidate_mass_max_abs"] == 0.0
        and all_null_diagnostics["null_mass_max_abs_error"] == 0.0
    )
    all_null_diagnostics["requires_exact_mass_and_gate_equality"] = True
    if not all(value["passed"] for value in parity.values()) or not all_null_diagnostics[
        "passed"
    ]:
        raise ConfirmationInputError("Development integrity invariant failed")

    cluster_names, groups = _cluster_index_groups(manifest["normalized_texts"])
    bootstrap = cluster_bootstrap_indices(
        len(cluster_names), samples=bootstrap_samples, seed=seed
    )
    layer_rows = []
    raw_by_mode = {}
    for mode, export in exports.items():
        if mode == "off":
            continue
        raw = _layer_rows(export, mode)
        raw_by_mode[mode] = raw
        layer_rows.extend(
            _aggregate_layer_rows(raw, mode=mode, groups=groups, bootstrap=bootstrap)
        )

    output_sensitivity = {}
    layer_sensitivity = {}
    for mode in expected_modes:
        if mode in {"off", "on"}:
            continue
        row_mse = []
        for index in range(len(exports["on"]["rows"])):
            correct = _load_rot6d(exports["on"], index)
            changed = _load_rot6d(exports[mode], index)
            if correct.shape != changed.shape:
                raise ConfirmationInputError(f"{mode} output length changed")
            row_mse.append(float(np.square(correct - changed).mean()))
        cluster = _cluster_mean(np.asarray(row_mse), groups)
        output_sensitivity[mode] = {
            "rot6d_mse_mean": float(cluster.mean()),
            "ci95": confidence_interval(cluster[bootstrap].mean(axis=1)),
        }
        layer_sensitivity[mode] = _centered_update_sensitivity(
            exports["on"], exports[mode], groups=groups, bootstrap=bootstrap
        )

    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "scientific_split": "val_development_novel_text_only",
        "completed": True,
        "development_validation_only": True,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(dict(authorization["checkpoint"])["epoch"]),
        },
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
            "authorization_identity": authorization["authorization_identity"],
        },
        "relevance_calibration": dict(authorization["checkpoint"]).get(
            "relevance_calibration"
        ),
        "partition_digest": EXPECTED_PARTITION_DIGEST,
        "development_manifest_sha256": manifest["sha256"],
        "development_rows": len(manifest["rows"]),
        "development_text_clusters": len(cluster_names),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
        "control_provenance": controls,
        "exact_invariants": {**parity, "all_null": all_null_diagnostics},
        "output_corruption_sensitivity": output_sensitivity,
        "layer_centered_update_sensitivity": layer_sensitivity,
        "layerwise_metrics_file": "layerwise_metrics.csv",
        "explanatory_only": True,
        "scientific_gate": False,
        "changes_checkpoint_selection": False,
        "query_bindings": {
            mode: export["query_binding"]
            for mode, export in {
                **exports,
                "broadcast_complete": broadcast,
                "all_null": all_null,
            }.items()
        },
        "input_export_summary_sha256": {
            mode: sha256_file(export["summary_path"])
            for mode, export in {
                **exports,
                "broadcast_complete": broadcast,
                "all_null": all_null,
            }.items()
        },
        "layerwise_rows_identity": _digest_json(layer_rows),
    }
    summary["identity"] = _digest_json(summary)
    return summary, layer_rows


def write_outputs(
    out_dir: Path,
    summary: Mapping[str, Any],
    layer_rows: list[Mapping[str, Any]],
) -> None:
    out_dir = out_dir.resolve()
    summary_bytes = (
        json.dumps(
            summary, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    csv_bytes = _canonical_csv_bytes(layer_rows)
    ready_payload = {
        "schema_name": DIAGNOSTIC_READY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "diagnostic_identity": summary["identity"],
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "authorization_identity": dict(summary["authorization"])[
            "authorization_identity"
        ],
    }
    ready_payload["ready_identity"] = _digest_json(ready_payload)
    ready_bytes = (
        json.dumps(ready_payload, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if out_dir.exists():
        _validate_exact_completed_directory(
            out_dir,
            expected_files={"summary.json", "layerwise_metrics.csv", "READY"},
        )
        existing = _json(out_dir / "summary.json")
        ready = _json(out_dir / "READY")
        if (
            existing != dict(summary)
            or ready != ready_payload
            or (out_dir / "summary.json").read_bytes() != summary_bytes
            or (out_dir / "layerwise_metrics.csv").read_bytes() != csv_bytes
            or (out_dir / "READY").read_bytes() != ready_bytes
        ):
            raise ConfirmationInputError("Existing diagnostic output is detached")
        return
    building = out_dir.with_name(
        f".{out_dir.name}.building.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    if building.exists():
        raise ConfirmationInputError(f"Incomplete diagnostic build exists: {building}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    _durable_write_bytes(building / "summary.json", summary_bytes)
    _durable_write_bytes(building / "layerwise_metrics.csv", csv_bytes)
    _durable_write_bytes(building / "READY", ready_bytes)
    _fsync_directory(building)
    _rename_directory_no_replace(building, out_dir)
    _fsync_directory(out_dir.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition_dir", type=Path, required=True)
    parser.add_argument("--broadcast_complete_dir", type=Path, required=True)
    parser.add_argument("--all_null_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    for mode in STAGE2_EVAL_MODES:
        parser.add_argument(f"--{mode}_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = str(_json(args.authorization).get("stage", ""))
    expected_modes = STAGE1_EVAL_MODES if stage == STAGE1 else STAGE2_EVAL_MODES
    mode_dirs = {}
    for mode in expected_modes:
        value = getattr(args, f"{mode}_dir")
        if value is None:
            raise ConfirmationInputError(f"Missing --{mode}_dir")
        mode_dirs[mode] = value
    summary, rows = analyze_diagnostic(
        authorization_path=args.authorization,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        partition_dir=args.partition_dir,
        mode_dirs=mode_dirs,
        broadcast_dir=args.broadcast_complete_dir,
        all_null_dir=args.all_null_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_outputs(args.out_dir, summary, rows)
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "development_rows": summary["development_rows"],
                "development_text_clusters": summary[
                    "development_text_clusters"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
