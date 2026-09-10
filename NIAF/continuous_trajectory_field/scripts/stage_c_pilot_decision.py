"""Make the immutable, development-only decision for a paired Stage-C run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    publish_bytes_no_replace,
)


POLICY_SCHEMA = "signtrajfield_stage_c_generator_adaptation_decision_policy"
DECISION_SCHEMA = "signtrajfield_stage_c_generator_adaptation_decision"
SCHEMA_VERSION = 1
SOURCE_SHA256 = "b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202"
ARMS = ("memory", "matched_off")
EXPECTED_STEPS = {"smoke": 1, "pilot": 72}
PROTOCOL_V2_RUN_R3_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
    "decision_policy_v1.json"
)
PROTOCOL_V3_RUN_R4_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
    "decision_policy_v1.json"
)
PROTOCOL_V4_RUN_R5_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_"
    "decision_policy_v1.json"
)
SMOKE_PREREQUISITE_POLICY_NAMES = frozenset(
    {
        PROTOCOL_V2_RUN_R3_POLICY_NAME,
        PROTOCOL_V3_RUN_R4_POLICY_NAME,
        PROTOCOL_V4_RUN_R5_POLICY_NAME,
    }
)
INVARIANT_KEYS = (
    "selection_joint_tuple_prediction_max_abs",
    "selection_joint_tuple_duration_max_abs",
    "selection_uniform_final_vs_off_prediction_max_abs",
    "selection_uniform_final_vs_off_duration_max_abs",
    "selection_broadcast_complete_vs_off_prediction_max_abs",
    "selection_broadcast_complete_vs_off_duration_max_abs",
    "selection_all_null_vs_off_prediction_max_abs",
    "selection_all_null_vs_off_duration_max_abs",
    "selection_all_null_gate_max_abs",
    "selection_all_null_candidate_mass_max_abs",
    "selection_all_null_one_minus_null_mass_max_abs",
)
METRIC_KEYS = {
    "correct_score": "selection_correct_score",
    "correct_vs_off_relative_gain": "selection_relative_gain_over_off",
    "lhand_negative_degradation_vs_off": (
        "selection_lhand_negative_degradation_vs_off"
    ),
    "live_off_allowed_score": (
        "selection_stage_c_text_only_guard_allowed_text_only_score"
    ),
    "live_off_guard_max_relative_degradation": (
        "selection_stage_c_text_only_guard_max_relative_degradation"
    ),
    "live_off_guard_passed": "selection_stage_c_text_only_guard_passed",
    "live_off_score": (
        "selection_stage_c_text_only_guard_live_text_only_score"
    ),
    "live_off_teacher_score": (
        "selection_stage_c_text_only_guard_teacher_score"
    ),
    "off_score": "selection_off_score",
    "rpair": "selection_Rpair",
    "rpair_n0": "selection_Rpair_n0",
    "rpair_n1": "selection_Rpair_n1",
    "rpair_n2": "selection_Rpair_n2",
    "rhand_negative_degradation_vs_off": (
        "selection_rhand_negative_degradation_vs_off"
    ),
    "selection_score": "selection_score",
}
THRESHOLDS = {
    "correct_score_minimum_relative_improvement_over_matched_off": 0.001,
    "correct_score_minimum_relative_improvement_over_source": 0.001,
    "correct_vs_off_minimum_relative_gain": 0.001,
    "invariant_exact_value": 0.0,
    "lhand_negative_degradation_vs_off_minimum": -0.02,
    "live_off_maximum_relative_degradation_over_source": 0.005,
    "rpair_minimum": 0.10,
    "rpair_nonce_minimum": 0.05,
    "rhand_negative_degradation_vs_off_minimum": -0.02,
}


class StageCDecisionError(RuntimeError):
    """The paired execution cannot support a deterministic Stage-C decision."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise StageCDecisionError(f"{label} is not a regular file: {path}")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageCDecisionError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise StageCDecisionError(f"{label} must be a JSON mapping")
    return value


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    import torch

    _regular(path, label)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise StageCDecisionError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise StageCDecisionError(f"{label} is not a checkpoint mapping")
    return value


def _validate_source_binding_generation(
    *,
    bound_files: Any,
    decision_policy: Any,
    mode: str | None = None,
    prior_one_update_smoke: Any = None,
) -> int:
    if not isinstance(bound_files, dict) or not isinstance(decision_policy, dict):
        raise StageCDecisionError("source binding generation is malformed")
    if set(decision_policy) != {"path", "sha256"}:
        raise StageCDecisionError("source binding decision policy is malformed")
    policy_path = Path(str(decision_policy["path"])).resolve()
    try:
        normalized_files = {
            str(Path(path).resolve()): value for path, value in bound_files.items()
        }
    except TypeError as error:
        raise StageCDecisionError("source binding file path is malformed") from error
    if len(normalized_files) != len(bound_files):
        raise StageCDecisionError("source binding contains duplicate resolved paths")
    if normalized_files.get(str(policy_path)) != decision_policy["sha256"]:
        raise StageCDecisionError("decision policy is absent from the source binding")
    policy_name = policy_path.name
    names = {Path(path).name for path in bound_files}
    protocol_v2_exclusive_markers = {
        (
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "decision_policy_v1.json"
        ),
        (
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "recovery_evidence_v1.json"
        ),
    }
    protocol_v3_exclusive_markers = {
        (
            "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
            "decision_policy_v1.json"
        ),
        (
            "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
            "recovery_evidence_v1.json"
        ),
    }
    protocol_v4_exclusive_markers = {
        (
            "csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_"
            "decision_policy_v1.json"
        ),
        (
            "csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_"
            "recovery_evidence_v1.json"
        ),
    }
    protocol_v2_required_markers = protocol_v2_exclusive_markers | {
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_pilot_run_r2.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_pilot_run_r2.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3.yaml"
        ),
        "run_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh",
        (
            "calibrate_csl_daily_stage_c_generator_adaptation_"
            "protocol_v2_run_r3_sbatch.sh"
        ),
    }
    if policy_name == PROTOCOL_V2_RUN_R3_POLICY_NAME:
        if mode not in EXPECTED_STEPS:
            raise StageCDecisionError(
                "protocol-v2 source binding execution mode is not exact"
            )
        expected_count = 30 if mode == "pilot" else 29
        if (
            len(bound_files) != expected_count
            or not protocol_v2_required_markers.issubset(names)
        ):
            raise StageCDecisionError("protocol-v2 source binding file set is not exact")
        if mode == "pilot":
            if not isinstance(prior_one_update_smoke, dict):
                raise StageCDecisionError(
                    "pilot source binding lacks the one-update smoke prerequisite"
                )
            ready_path = str(
                Path(str(prior_one_update_smoke.get("ready_path", ""))).resolve()
            )
            if normalized_files.get(ready_path) != prior_one_update_smoke.get(
                "ready_sha256"
            ):
                raise StageCDecisionError(
                    "pilot source binding does not hash-bind smoke READY"
                )
        elif prior_one_update_smoke is not None:
            raise StageCDecisionError(
                "smoke source binding must not contain a self-prerequisite"
            )
        return expected_count
    protocol_v3_required_markers = protocol_v3_exclusive_markers | {
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v3_run_r4.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v3_run_r4.yaml"
        ),
        "run_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh",
        (
            "calibrate_csl_daily_stage_c_generator_adaptation_"
            "protocol_v3_run_r4_sbatch.sh"
        ),
        "ARCHIVE.json",
    }
    if policy_name == PROTOCOL_V3_RUN_R4_POLICY_NAME:
        if mode not in EXPECTED_STEPS:
            raise StageCDecisionError(
                "protocol-v3 source binding execution mode is not exact"
            )
        expected_count = 31 if mode == "pilot" else 30
        if (
            len(bound_files) != expected_count
            or not protocol_v3_required_markers.issubset(names)
        ):
            raise StageCDecisionError("protocol-v3 source binding file set is not exact")
        if mode == "pilot":
            if not isinstance(prior_one_update_smoke, dict):
                raise StageCDecisionError(
                    "pilot source binding lacks the one-update smoke prerequisite"
                )
            ready_path = str(
                Path(str(prior_one_update_smoke.get("ready_path", ""))).resolve()
            )
            if normalized_files.get(ready_path) != prior_one_update_smoke.get(
                "ready_sha256"
            ):
                raise StageCDecisionError(
                    "pilot source binding does not hash-bind smoke READY"
                )
        elif prior_one_update_smoke is not None:
            raise StageCDecisionError(
                "smoke source binding must not contain a self-prerequisite"
            )
        return expected_count
    protocol_v4_required_markers = protocol_v4_exclusive_markers | {
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v4_run_r5.yaml"
        ),
        (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v4_run_r5.yaml"
        ),
        "run_csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5.sh",
        (
            "calibrate_csl_daily_stage_c_generator_adaptation_"
            "protocol_v4_run_r5_sbatch.sh"
        ),
        "ARCHIVE.json",
    }
    if policy_name == PROTOCOL_V4_RUN_R5_POLICY_NAME:
        if mode not in EXPECTED_STEPS:
            raise StageCDecisionError(
                "protocol-v4 source binding execution mode is not exact"
            )
        expected_count = 31 if mode == "pilot" else 30
        if (
            len(bound_files) != expected_count
            or not protocol_v4_required_markers.issubset(names)
        ):
            raise StageCDecisionError("protocol-v4 source binding file set is not exact")
        if mode == "pilot":
            if not isinstance(prior_one_update_smoke, dict):
                raise StageCDecisionError(
                    "pilot source binding lacks the one-update smoke prerequisite"
                )
            ready_path = str(
                Path(str(prior_one_update_smoke.get("ready_path", ""))).resolve()
            )
            if normalized_files.get(ready_path) != prior_one_update_smoke.get(
                "ready_sha256"
            ):
                raise StageCDecisionError(
                    "pilot source binding does not hash-bind smoke READY"
                )
        elif prior_one_update_smoke is not None:
            raise StageCDecisionError(
                "smoke source binding must not contain a self-prerequisite"
            )
        return expected_count
    if policy_name in {
        "csl_daily_stage_c_generator_adaptation_decision_policy_v1.json",
        "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json",
    }:
        if len(bound_files) != 27 or names.intersection(
            protocol_v2_exclusive_markers
            | protocol_v3_exclusive_markers
            | protocol_v4_exclusive_markers
        ):
            raise StageCDecisionError("protocol-v1 source binding file set is not exact")
        return 27
    raise StageCDecisionError("source binding decision policy is not registered")


def _validate_pilot_smoke_prerequisite(
    *,
    value: Any,
    complete: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen and compare the exact one-update smoke prerequisite."""

    if not isinstance(value, dict):
        raise StageCDecisionError(
            "pilot lacks the exact one-update smoke prerequisite"
        )
    policy_binding = complete.get("decision_policy")
    if not isinstance(policy_binding, dict):
        raise StageCDecisionError("pilot decision-policy binding is malformed")
    policy_path = Path(str(policy_binding.get("path", ""))).resolve()
    if policy_path.name not in SMOKE_PREREQUISITE_POLICY_NAMES:
        raise StageCDecisionError(
            "one-update smoke prerequisite is restricted to registered protocols"
        )
    try:
        project_root = policy_path.parents[3]
    except IndexError as error:
        raise StageCDecisionError(
            "pilot decision-policy path cannot identify the source root"
        ) from error
    head = str(complete.get("source_git_head", ""))
    protocol_root = {
        PROTOCOL_V2_RUN_R3_POLICY_NAME: "protocol_v2_run_r3",
        PROTOCOL_V3_RUN_R4_POLICY_NAME: "protocol_v3_run_r4",
        PROTOCOL_V4_RUN_R5_POLICY_NAME: "protocol_v4_run_r5",
    }[policy_path.name]
    expected_smoke_root = project_root / (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
        f"adaptation_{protocol_root}_smoke"
    )
    expected_ready = project_root / (
        "experiments/NIAF/continuous_trajectory_field/"
        f"csl_daily_stage_c_generator_adaptation_{protocol_root}_"
        f"prerequisites/source_{head}/smoke/PUBLICATION/READY"
    )
    if (
        Path(str(value.get("smoke_root_path", ""))).resolve()
        != expected_smoke_root.resolve()
        or Path(str(value.get("ready_path", ""))).resolve()
        != expected_ready.resolve()
    ):
        raise StageCDecisionError(
            "pilot one-update smoke prerequisite path is not canonical"
        )
    from NIAF.continuous_trajectory_field.scripts import (
        stage_c_pilot_prerequisites as prerequisites,
    )

    try:
        recomputed = prerequisites.validate_smoke(
            smoke_ready=expected_ready,
            smoke_root=expected_smoke_root,
            source_git_head=head,
            pair_constraint=str(complete.get("pair_constraint", "")),
        )
    except prerequisites.PrerequisiteError as error:
        raise StageCDecisionError(
            f"pilot one-update smoke prerequisite changed: {error}"
        ) from error
    launch_artifacts = launch.get("artifacts")
    if not isinstance(launch_artifacts, dict):
        raise StageCDecisionError("pilot launch artifact binding is malformed")
    expected_current = {
        "memory_config_sha256": (launch_artifacts.get("memory_config") or {}).get(
            "sha256"
        ),
        "matched_off_config_sha256": (
            launch_artifacts.get("matched_off_config") or {}
        ).get("sha256"),
        "cpu_gate_sha256": (launch_artifacts.get("cpu_gate") or {}).get("sha256"),
        "calibration_completion_sha256": (
            launch_artifacts.get("calibration_completion") or {}
        ).get("sha256"),
    }
    if (
        value != recomputed
        or value.get("source_git_head") != head
        or value.get("source_remote_ref") != complete.get("source_remote_ref")
        or value.get("source_remote_head") != complete.get("source_remote_head")
        or value.get("pair_constraint") != complete.get("pair_constraint")
        or value.get("decision_policy_sha256") != policy_binding.get("sha256")
        or any(value.get(key) != expected for key, expected in expected_current.items())
    ):
        raise StageCDecisionError(
            "pilot one-update smoke prerequisite differs from active science"
        )
    return recomputed


def _validate_network_evidence(
    complete: Mapping[str, Any], execution_complete: Path, mode: str
) -> dict[str, Any]:
    """Reopen every hash-bound RoCE audit before creating a decision."""

    from NIAF.continuous_trajectory_field.scripts import stage_c_paired_roce as roce

    attempt = execution_complete.parent.resolve()
    network = attempt / "network_preflight"
    audit_key = (
        "smoke_checkpoint_audit" if mode == "smoke" else "pilot_checkpoint_audit"
    )
    audit_name = (
        "SMOKE_CHECKPOINT_AUDIT.json"
        if mode == "smoke"
        else "PILOT_CHECKPOINT_AUDIT.json"
    )
    relative_paths = {
        "launch": "LAUNCH.json",
        "source_binding": "network_preflight/SOURCE_BINDING.json",
        "pair": "network_preflight/PAIR.json",
        "counter_health": "network_preflight/COUNTER_HEALTH.json",
        "dual_vs_single": "network_preflight/NCCL_DUAL_VS_SINGLE.json",
        "train_memory_logs": "network_preflight/NCCL_LOGS_train_memory.json",
        "train_matched_off_logs": (
            "network_preflight/NCCL_LOGS_train_matched_off.json"
        ),
        audit_key: f"network_preflight/{audit_name}",
    }
    for profile in roce.NCCL_PROFILES:
        relative_paths[f"benchmark_{profile}"] = (
            f"network_preflight/NCCL_BENCHMARK_{profile}.json"
        )
        relative_paths[f"benchmark_logs_{profile}"] = (
            f"network_preflight/NCCL_LOGS_{profile}.json"
        )
    artifacts = complete.get("execution_artifacts")
    if not isinstance(artifacts, dict):
        raise StageCDecisionError("paired execution RoCE artifact set is not exact")
    pair_path = _regular(attempt / relative_paths["pair"], "RoCE artifact pair")
    if artifacts.get("pair") != sha256_file(pair_path):
        raise StageCDecisionError("RoCE artifact hash changed: pair")
    pair = _json(pair_path, "RoCE artifact pair")
    nodes = pair.get("nodes")
    if (
        not isinstance(nodes, list)
        or len(nodes) != 2
        or len(set(nodes)) != 2
        or not all(
            isinstance(node, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node) is not None
            for node in nodes
        )
    ):
        raise StageCDecisionError("PAIR evidence is malformed")
    for index, node in enumerate(nodes):
        relative_paths[f"node_audit_{index}"] = f"network_preflight/{node}.node.json"
        relative_paths[f"connectivity_audit_{index}"] = (
            f"network_preflight/{node}.connectivity.json"
        )
        for phase in ("pre", "post"):
            relative_paths[f"counter_{phase}_{index}"] = (
                f"network_preflight/{node}.counters_{phase}.json"
            )
    if set(artifacts) != set(relative_paths):
        raise StageCDecisionError("paired execution RoCE artifact set is not exact")
    values: dict[str, dict[str, Any]] = {"pair": pair}
    for key, relative in relative_paths.items():
        if key == "pair":
            continue
        path = _regular(attempt / relative, f"RoCE artifact {key}")
        if artifacts[key] != sha256_file(path):
            raise StageCDecisionError(f"RoCE artifact hash changed: {key}")
        if key != audit_key:
            values[key] = _json(path, f"RoCE artifact {key}")

    source_binding = values["source_binding"]
    bound_files = source_binding.get("files")
    prior_one_update_smoke = source_binding.get("prior_one_update_smoke")
    if isinstance(bound_files, dict):
        _validate_source_binding_generation(
            bound_files=bound_files,
            decision_policy=complete.get("decision_policy"),
            mode=mode,
            prior_one_update_smoke=prior_one_update_smoke,
        )
    source_binding_fields = {
        "schema_name",
        "schema_version",
        "source_git_head",
        "source_remote_ref",
        "source_remote_head",
        "files",
        "development_only",
        "non_authorizing",
    }
    decision_policy_path = Path(
        str((complete.get("decision_policy") or {}).get("path", ""))
    )
    protocol_v2 = decision_policy_path.name in SMOKE_PREREQUISITE_POLICY_NAMES
    if protocol_v2:
        source_binding_fields.add("execution_mode")
        if mode == "pilot":
            source_binding_fields.add("prior_one_update_smoke")
    if (
        set(source_binding) != source_binding_fields
        or (protocol_v2 and source_binding.get("execution_mode") != mode)
        or source_binding.get("schema_name")
        != "signtrajfield_stage_c_source_binding"
        or source_binding.get("schema_version") != 1
        or source_binding.get("source_git_head") != complete.get("source_git_head")
        or source_binding.get("source_remote_ref") != complete.get("source_remote_ref")
        or source_binding.get("source_remote_head")
        != complete.get("source_remote_head")
        or source_binding.get("development_only") is not True
        or source_binding.get("non_authorizing") is not True
        or not isinstance(bound_files, dict)
    ):
        raise StageCDecisionError("source binding manifest is malformed")
    for path_text, expected_sha in bound_files.items():
        path = _regular(Path(path_text), "source-bound Stage-C file")
        if sha256_file(path) != expected_sha:
            raise StageCDecisionError(f"source-bound file changed: {path}")

    pair_constraint = complete.get("pair_constraint")
    try:
        recomputed_pair = roce.validate_pair_evidence(
            network, nodes, str(pair_constraint)
        )
        roce.validate_peer_evidence(network, recomputed_pair)
        recomputed_counters = roce.validate_counter_snapshots(network, nodes)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, roce.PreflightError) as error:
        raise StageCDecisionError(
            f"raw RoCE link/peer/counter evidence is invalid: {error}"
        ) from error
    if pair != recomputed_pair:
        raise StageCDecisionError("PAIR evidence is not reproducible from node audits")
    if values["counter_health"] != recomputed_counters:
        raise StageCDecisionError(
            "counter health is not reproducible from four raw snapshots"
        )

    try:
        recomputed_comparison = roce.validate_benchmarks(network, nodes)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, roce.PreflightError) as error:
        raise StageCDecisionError(f"NCCL benchmark evidence is invalid: {error}") from error
    if values["dual_vs_single"] != recomputed_comparison:
        raise StageCDecisionError("dual-vs-single selection is not reproducible")
    selected_profile = recomputed_comparison["training_profile"]

    def validate_log_audit(key: str, expected_profile: str) -> None:
        value = values[key]
        intended = [
            item.split(":", 1)[0]
            for item in roce.NCCL_PROFILES[expected_profile].split(",")
        ]
        log_paths = value.get("logs")
        per_log = value.get("per_log_evidence")
        if (
            set(value)
            != {
                "schema_name",
                "schema_version",
                "profile",
                "intended_hcas",
                "logs",
                "hosts",
                "net_ib_line_count",
                "socket_fallback_detected",
                "per_log_evidence",
            }
            or
            value.get("schema_name") != roce.SCHEMA
            or value.get("schema_version") != roce.SCHEMA_VERSION
            or value.get("profile") != expected_profile
            or value.get("intended_hcas") != intended
            or value.get("socket_fallback_detected") is not False
            or value.get("hosts") != sorted(nodes)
            or not isinstance(value.get("net_ib_line_count"), int)
            or value["net_ib_line_count"] < 2
            or not isinstance(log_paths, list)
            or len(log_paths) != 2
            or not isinstance(per_log, list)
            or len(per_log) != 2
        ):
            raise StageCDecisionError(f"NCCL log audit is malformed: {key}")
        reopened = []
        recomputed_hosts = []
        recomputed_line_count = 0
        expected_prefix = (
            f"nccl.train.memory.{expected_profile}."
            if key == "train_memory_logs"
            else (
                f"nccl.train.matched_off.{expected_profile}."
                if key == "train_matched_off_logs"
                else f"nccl.{expected_profile}."
            )
        )
        for path_text, row in zip(log_paths, per_log, strict=True):
            path = _regular(Path(path_text), f"raw NCCL rank log {key}")
            if path.resolve().parent != network:
                raise StageCDecisionError(f"NCCL log escaped attempt directory: {key}")
            contents = path.read_text(encoding="utf-8", errors="replace")
            ib_lines = [line for line in contents.splitlines() if "NET/IB" in line]
            suffix = path.name.removeprefix(expected_prefix).removesuffix(".log")
            host, separator, process_id = suffix.rpartition(".")
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "path",
                    "sha256",
                    "host",
                    "net_ib_line_count",
                    "intended_hcas_present",
                    "socket_fallback_detected",
                }
                or row.get("path") != str(path.resolve())
                or row.get("sha256") != sha256_file(path)
                or row.get("host") != host
                or row.get("net_ib_line_count") != len(ib_lines)
                or row.get("intended_hcas_present") != intended
                or row.get("socket_fallback_detected") is not False
                or not ib_lines
                or "NET/Socket" in contents
                or "Using network Socket" in contents
                or not path.name.startswith(expected_prefix)
                or not path.name.endswith(".log")
                or not separator
                or re.fullmatch(r"[0-9]+", process_id) is None
                or host not in nodes
                or any(not any(hca in line for line in ib_lines) for hca in intended)
            ):
                raise StageCDecisionError(
                    f"raw NCCL rank log lacks exact NET/IB/HCA proof: {key}"
                )
            reopened.append(str(path.resolve()))
            recomputed_hosts.append(host)
            recomputed_line_count += len(ib_lines)
        if (
            len(set(reopened)) != 2
            or sorted(recomputed_hosts) != sorted(nodes)
            or len(set(recomputed_hosts)) != 2
            or value["net_ib_line_count"] != recomputed_line_count
        ):
            raise StageCDecisionError(
                f"NCCL log audit does not represent both exact pair ranks: {key}"
            )

    for profile in roce.NCCL_PROFILES:
        validate_log_audit(f"benchmark_logs_{profile}", profile)
    validate_log_audit("train_memory_logs", selected_profile)
    validate_log_audit("train_matched_off_logs", selected_profile)

    launch = values["launch"]
    launch_artifacts = launch.get("artifacts")
    if (
        launch.get("schema_name")
        != "signtrajfield_centered_stage_c_paired_pilot_launch"
        or launch.get("schema_version") != 1
        or launch.get("execution_mode") != mode
        or launch.get("source_git_head") != complete.get("source_git_head")
        or launch.get("source_remote_ref") != complete.get("source_remote_ref")
        or launch.get("source_remote_head") != complete.get("source_remote_head")
        or launch.get("pair_constraint") != pair_constraint
        or launch.get("world_size") != 2
        or launch.get("batch_per_rank") != 64
        or launch.get("accumulation_steps") != 2
        or launch.get("effective_global_batch") != 256
        or launch.get("training_network_profile") != selected_profile
        or launch.get("training_nccl_ib_hca")
        != recomputed_comparison["training_hcas"]
        or launch.get("development_only") is not True
        or launch.get("non_authorizing") is not True
        or launch.get("promotion_eligible") is not False
        or launch.get("confirmation_or_test_access_permitted") is not False
        or (launch.get("artifacts", {}).get("source_binding_manifest") or {}).get(
            "sha256"
        )
        != artifacts["source_binding"]
    ):
        raise StageCDecisionError("launch evidence disagrees with RoCE decision")
    if protocol_v2:
        if not isinstance(launch_artifacts, dict) or (
            "prior_one_update_smoke" in launch_artifacts
        ):
            raise StageCDecisionError(
                "protocol-v2 launch smoke-prerequisite placement is malformed"
            )
        if mode == "pilot":
            if (
                complete.get("prior_one_update_smoke")
                != prior_one_update_smoke
                or launch.get("prior_one_update_smoke")
                != prior_one_update_smoke
            ):
                raise StageCDecisionError(
                    "pilot smoke prerequisite differs across bound evidence"
                )
            _validate_pilot_smoke_prerequisite(
                value=prior_one_update_smoke,
                complete=complete,
                launch=launch,
            )
        elif any(
            "prior_one_update_smoke" in value
            for value in (source_binding, launch, complete)
        ):
            raise StageCDecisionError(
                "smoke execution contains a forbidden self-prerequisite"
            )
    return {
        "pair_constraint": pair_constraint,
        "nodes": nodes,
        "training_profile": selected_profile,
        "training_hcas": recomputed_comparison["training_hcas"],
        "harmful_counter_increment_count": 0,
        "all_three_benchmark_profiles_valid": True,
        "all_five_log_audits_and_ten_rank_logs_valid": True,
    }


def _finite_numbers(value: Any, label: str) -> None:
    import torch

    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise StageCDecisionError(f"{label} contains a non-finite tensor")
    elif isinstance(value, Mapping):
        for name, item in value.items():
            _finite_numbers(item, f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise StageCDecisionError(f"{label} contains a non-finite number")


def _float(metrics: Mapping[str, Any], key: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise StageCDecisionError(f"required decision metric is absent: {key}") from error
    if not math.isfinite(value):
        raise StageCDecisionError(f"required decision metric is non-finite: {key}")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def validate_policy(path: Path) -> dict[str, Any]:
    policy = _json(path, "Stage-C decision policy")
    base_fields = {
        "schema_name",
        "schema_version",
        "authorization",
        "execution",
        "invariant_metric_keys",
        "metric_keys",
        "pilot_thresholds",
        "retry_policy",
        "source_checkpoint",
        "status_values",
    }
    allowed_field_sets = {
        frozenset(base_fields),
        frozenset(base_fields | {"recovery_contract"}),
    }
    if frozenset(policy) not in allowed_field_sets:
        raise StageCDecisionError("Stage-C decision policy fields are not exact")
    recovery_contract = policy.get("recovery_contract")
    if "recovery_contract" in policy:
        if path.name == (
            "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
        ):
            expected_recovery_contract = {
                "evidence_manifest": {
                    "path": (
                        "NIAF/continuous_trajectory_field/configs/"
                        "csl_daily_stage_c_generator_adaptation_retry2_"
                        "recovery_evidence_v1.json"
                    ),
                    "sha256": (
                        "ccf47b72390c4be28775b207c67f7372d3cbc599fcddb3545df8295acc88b39b"
                    ),
                },
                "failed_smoke_job_id": "143525",
                "failure_class": (
                    "pre_science_source_terminal_decision_schema_compatibility"
                ),
                "prior_evidence_must_be_preserved": True,
                "prior_execution_lease_created": False,
                "prior_scientific_output_observed": False,
                "retry_authorization_reason": (
                    "attested_pre_claim_pre_science_implementation_failure"
                ),
                "retry_authorized": True,
                "run_generation": "run_r2",
                "supersedes_run_generation": "run_r1",
            }
            expected_retry_policy = {
                "malformed_completion": "stop_no_replace",
                "partial_scientific_output": "stop_no_retry",
                "unique_complete_execution": "reuse_across_all_job_ids",
                "zero_scientific_output": (
                    "new_attempt_permitted_after_terminal_lease"
                ),
            }
        elif path.name == (
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "decision_policy_v1.json"
        ):
            expected_recovery_contract = {
                "allowed_operational_change": (
                    "centered_evaluator_dispatch_when_centered_evaluation_enabled"
                ),
                "evidence_manifest": {
                    "path": (
                        "NIAF/continuous_trajectory_field/configs/"
                        "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                        "recovery_evidence_v1.json"
                    ),
                    "sha256": (
                        "f955dbc7ce03f5f6a400d8ae2fa21c7ca5a280fec7f7a37da674d35a027e9209"
                    ),
                },
                "incident_archive": {
                    "path": (
                        "experiments/NIAF/continuous_trajectory_field/"
                        "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                        "adaptation_smoke.invalid_attempts/"
                        "source_f12b993b5de361423df3b8cbfb4e873f4a95ad1e_"
                        "smoke143541_pilot143542/ARCHIVE.json"
                    ),
                    "sha256": (
                        "82ce35cf3d7337f218bda189080a45b9e3faedec200750310d0958296f9d1855"
                    ),
                },
                "matched_off_arm_started": False,
                "new_protocol_generation_authorized": True,
                "prior_execution_lease_created": True,
                "prior_memory_optimizer_updates_observed": 1,
                "prior_scientific_output_observed": True,
                "protocol_generation": "protocol_v2",
                "resume_authorized": False,
                "run_generation": "run_r3",
                "same_protocol_retry_authorized": False,
                "supersedes_protocol": "stage_c_generator_adaptation_retry2_v1",
                "supersedes_run_generation": "run_r2",
            }
            expected_retry_policy = {
                "malformed_completion": "stop_no_replace",
                "partial_scientific_output": "stop_no_retry",
                "unique_complete_execution": "reuse_only_for_publication_recovery",
                "zero_scientific_output": "stop_no_retry_within_protocol_generation",
            }
        elif path.name == PROTOCOL_V3_RUN_R4_POLICY_NAME:
            expected_recovery_contract = {
                "allowed_operational_change": (
                    "increase_only_the_bounded_archived_clone_local_git_metadata_timeout"
                ),
                "downstream_proof_delegation": (
                    "validation_preserving_execution_safety: CPU READY authenticates "
                    "the one-time historical proof; downstream reopens only bound "
                    "hashes and current source/config identities."
                ),
                "evidence_manifest": {
                    "path": (
                        "NIAF/continuous_trajectory_field/configs/"
                        "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
                        "recovery_evidence_v1.json"
                    ),
                    "sha256": (
                        "90383a1a5be787e946a4a1fba3e31e9673e07edc76ec77a1a7573e2800e5913c"
                    ),
                },
                "incident_archive": {
                    "path": (
                        "experiments/NIAF/continuous_trajectory_field/"
                        "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                        "cpu.invalid_attempts/"
                        "source_90cad3d3a7a09279b4a882e8c28a13b2a320ec1a_"
                        "cpu143574_dependents143575_143577/ARCHIVE.json"
                    ),
                    "sha256": (
                        "66368adf6a2310732a4a2bcca87625d45c8dd51c0c4912b16ceba9b4c12b8dc2"
                    ),
                },
                "new_protocol_generation_authorized": True,
                "prior_execution_lease_created": False,
                "prior_scientific_output_observed": False,
                "protocol_generation": "protocol_v3",
                "resume_authorized": False,
                "run_generation": "run_r4",
                "same_protocol_retry_authorized": False,
                "supersedes_protocol": "protocol_v2",
                "supersedes_run_generation": "run_r3",
                "zero_science_archive_required": True,
            }
            expected_retry_policy = {
                "malformed_completion": "stop_no_replace",
                "partial_scientific_output": "stop_no_retry",
                "unique_complete_execution": "reuse_only_for_publication_recovery",
                "zero_scientific_output": "stop_no_retry_within_protocol_generation",
            }
        elif path.name == PROTOCOL_V4_RUN_R5_POLICY_NAME:
            expected_recovery_contract = {
                "allowed_operational_change": (
                    "correct_only_the_standalone_clone_cpu_test_fixture_evidence_root_binding"
                ),
                "downstream_proof_delegation": (
                    "validation_preserving_execution_safety: CPU READY authenticates "
                    "the one-time historical proof; downstream reopens only bound "
                    "hashes and current source/config identities."
                ),
                "evidence_manifest": {
                    "path": (
                        "NIAF/continuous_trajectory_field/configs/"
                        "csl_daily_stage_c_generator_adaptation_protocol_v4_run_r5_"
                        "recovery_evidence_v1.json"
                    ),
                    "sha256": "c6ab0d7626622c8163ac1917906ba606b2cc582d09cc8b44fe592008ee727d24",
                },
                "incident_archive": {
                    "path": (
                        "experiments/NIAF/continuous_trajectory_field/"
                        "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
                        "cpu.invalid_attempts/"
                        "source_740400412386d60a8d6f56b0f871bc16dee755b0_"
                        "cpu143593_dependents143594_143596/ARCHIVE.json"
                    ),
                    "sha256": "a23682b584bcc05ccee67ccec7526d33efc924104a0ea34209740bb4131e5f71",
                },
                "new_protocol_generation_authorized": True,
                "prior_execution_lease_created": False,
                "prior_scientific_output_observed": False,
                "protocol_generation": "protocol_v4",
                "resume_authorized": False,
                "run_generation": "run_r5",
                "same_protocol_retry_authorized": False,
                "supersedes_protocol": "protocol_v3",
                "supersedes_run_generation": "run_r4",
                "zero_science_archive_required": True,
            }
            expected_retry_policy = {
                "malformed_completion": "stop_no_replace",
                "partial_scientific_output": "stop_no_retry",
                "unique_complete_execution": "reuse_only_for_publication_recovery",
                "zero_scientific_output": "stop_no_retry_within_protocol_generation",
            }
        else:
            raise StageCDecisionError("Stage-C recovery policy path changed")
        if recovery_contract != expected_recovery_contract:
            raise StageCDecisionError("Stage-C retry-2 recovery contract changed")
    else:
        if path.name != "csl_daily_stage_c_generator_adaptation_decision_policy_v1.json":
            raise StageCDecisionError("Stage-C legacy decision policy path changed")
        expected_retry_policy = {
            "malformed_completion": "stop_no_replace",
            "partial_scientific_output": "stop_no_retry",
            "unique_complete_execution": "reuse_across_all_job_ids",
            "zero_scientific_output": "new_attempt_permitted_after_terminal_lease",
        }
    expected_authorization = {
        "authorized_purpose": None,
        "confirmation_manifest_opened": False,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "test_data_accessed": False,
    }
    expected_execution = {
        "arms": list(ARMS),
        "accumulation_steps": 2,
        "batch_per_rank": 64,
        "effective_global_batch": 256,
        "expected_epoch": 1,
        "expected_global_step": EXPECTED_STEPS,
        "full_pilot_data_limit_train": 0,
        "full_pilot_data_limit_val": 0,
        "full_pilot_drop_last": False,
        "full_pilot_eval_max_batches": 0,
        "full_pilot_length_bucketed_batches": True,
        "full_pilot_train_max_batches": 0,
        "smoke_eval_max_batches": 1,
        "smoke_train_max_batches": 2,
        "world_size": 2,
    }
    if path.name in SMOKE_PREREQUISITE_POLICY_NAMES:
        expected_execution.update(
            {
                "pilot_requires_prior_one_update_smoke_ready": True,
                "smoke_requires_prior_one_update_smoke_ready": False,
            }
        )
    if (
        policy["schema_name"] != POLICY_SCHEMA
        or policy["schema_version"] != SCHEMA_VERSION
        or policy["authorization"] != expected_authorization
        or policy["execution"] != expected_execution
        or policy["invariant_metric_keys"] != list(INVARIANT_KEYS)
        or policy["metric_keys"] != METRIC_KEYS
        or policy["pilot_thresholds"] != THRESHOLDS
        or policy["retry_policy"] != expected_retry_policy
        or policy["source_checkpoint"]
        != {
            "epoch": 5,
            "global_step": 360,
            "selection_status": "best_infeasible",
            "sha256": SOURCE_SHA256,
        }
        or policy["status_values"]
        != {
            "failure": {"next_permitted_action": "none", "status": "stop"},
            "pilot_success": {
                "next_permitted_action": (
                    "none_requires_fresh_preregistration_without_pilot_outcome_access"
                ),
                "status": "pilot_complete_development_signal",
            },
            "smoke_success": {
                "next_permitted_action": "run_one_epoch_development_pilot",
                "status": "smoke_ready",
            },
        }
    ):
        raise StageCDecisionError("Stage-C decision policy content changed")
    return policy


def _named_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or "digest" not in value:
        raise StageCDecisionError(f"{label} is missing its named identity")
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}
    if value["digest"] != digest_json(unsigned):
        raise StageCDecisionError(f"{label} named-identity digest changed")
    return value


def _normalize_config(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("experiment_name", None)
    result.get("output", {}).pop("out_dir", None)
    result.get("conditioning", {}).pop("sentence_memory_train_mode", None)
    result.get("conditioning", {}).pop("sentence_memory_dropout_probability", None)
    result.get("sentence_memory_safety", {}).get("stage_c", {}).pop("arm", None)
    return result


def _normalize_named(value: dict[str, Any], *fields: str) -> dict[str, Any]:
    result = {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}
    for field in fields:
        result.pop(field, None)
    return result


def _audit_arm_checkpoint(
    checkpoint: dict[str, Any], *, arm: str, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_step = EXPECTED_STEPS[mode]
    if checkpoint.get("epoch") != 1 or checkpoint.get("global_step") != expected_step:
        raise StageCDecisionError(
            f"{arm} checkpoint is not epoch=1/global_step={expected_step}"
        )
    metrics = checkpoint.get("metrics")
    config = checkpoint.get("config")
    if not isinstance(metrics, dict) or not isinstance(config, dict):
        raise StageCDecisionError(f"{arm} checkpoint lacks metrics/config mappings")
    _finite_numbers(checkpoint.get("model"), f"{arm}.model")
    _finite_numbers(checkpoint.get("optimizer"), f"{arm}.optimizer")
    _finite_numbers(metrics, f"{arm}.metrics")
    stage = config.get("sentence_memory_safety", {}).get("stage_c", {})
    expected_train_mode, expected_dropout = {
        "memory": ("dropout", 0.25),
        "matched_off": ("off", 1.0),
    }[arm]
    if (
        stage.get("arm") != arm
        or stage.get("enabled") is not True
        or stage.get("development_only") is not True
        or stage.get("non_authorizing") is not True
        or stage.get("promotion_eligible") is not False
        or stage.get("source_checkpoint", {}).get("sha256") != SOURCE_SHA256
        or config.get("train", {}).get("epochs") != 1
        or config.get("train", {}).get("batch_size") != 64
        or config.get("train", {}).get("accumulation_steps") != 2
        or config.get("train", {}).get("length_bucketed_batches") is not True
        or config.get("train", {}).get("drop_last") is not False
        or config.get("data", {}).get("limit_train", 0) != 0
        or config.get("data", {}).get("limit_val", 0) != 0
        or config.get("conditioning", {}).get("sentence_memory_train_mode")
        != expected_train_mode
        or float(
            config.get("conditioning", {}).get(
                "sentence_memory_dropout_probability", float("nan")
            )
        )
        != expected_dropout
    ):
        raise StageCDecisionError(f"{arm} checkpoint lost its Stage-C execution contract")
    expected_caps = (2, 1) if mode == "smoke" else (0, 0)
    observed_caps = (
        config.get("train", {}).get("max_train_batches", 0),
        config.get("eval", {}).get("max_batches", 0),
    )
    if observed_caps != expected_caps:
        raise StageCDecisionError(
            f"{arm} checkpoint has wrong {mode} batch caps: {observed_caps}"
        )
    scope = checkpoint.get("stage_c_authorization_scope")
    if scope != {
        "schema_name": "signtrajfield_centered_stage_c_generator_adaptation",
        "schema_version": 1,
        "arm": arm,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }:
        raise StageCDecisionError(f"{arm} checkpoint authorization scope changed")
    identities = {
        name: _named_identity(checkpoint.get(key), f"{arm}.{name}")
        for name, key in {
            "provenance": "stage_c_provenance",
            "calibration": "sentence_memory_relevance_calibration_identity",
            "distribution": "stage_c_distribution_contract",
            "architecture": "sentence_memory_architecture_identity",
            "objective": "sentence_memory_objective_identity",
            "trainability": "stage_c_trainability_contract",
            "optimizer_mapping": "stage_c_optimizer_parameter_mapping",
        }.items()
    }
    objective = identities["objective"]
    if (
        objective.get("arm") != arm
        or objective.get("sentence_memory_train_mode") != expected_train_mode
        or float(
            objective.get("sentence_memory_dropout_probability", float("nan"))
        )
        != expected_dropout
    ):
        raise StageCDecisionError(
            f"{arm} objective identity lost its exact arm memory controls"
        )
    return metrics, {"config": config, **identities}


def _relative_improvement(candidate: float, baseline: float) -> float:
    return (float(baseline) - float(candidate)) / max(abs(float(baseline)), 1e-8)


def _verify_selection_recomputation(
    metrics: Mapping[str, Any], config: Mapping[str, Any], trainer: Any, arm: str
) -> None:
    validation = {
        name.removeprefix("val_"): value
        for name, value in metrics.items()
        if name.startswith("val_")
    }
    if not validation:
        raise StageCDecisionError(f"{arm} checkpoint has no embedded validation metrics")
    score, violation, feasible, details = trainer.checkpoint_selection_diagnostics(
        validation, config, return_details=True
    )
    expected_top = {
        "selection_score": float(score),
        "selection_constraint_violation": float(violation),
        "selection_feasible": float(feasible),
    }
    for key, expected in expected_top.items():
        if not _close(_float(metrics, key), expected):
            raise StageCDecisionError(
                f"{arm} stored {key} differs from deterministic recomputation"
            )
    dual = details.get("dual_mode")
    if not isinstance(dual, dict):
        raise StageCDecisionError(f"{arm} recomputation lacks dual-mode details")
    for name, value in dual.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            key = f"selection_{name}"
            if not _close(_float(metrics, key), float(value)):
                raise StageCDecisionError(
                    f"{arm} stored {key} differs from deterministic recomputation"
                )
    guard = dual.get("stage_c_text_only_guard")
    if not isinstance(guard, dict):
        raise StageCDecisionError(f"{arm} recomputation lacks live text-only guard")
    for name in (
        "live_text_only_score",
        "teacher_score",
        "allowed_text_only_score",
        "max_relative_degradation",
        "violation",
    ):
        key = f"selection_stage_c_text_only_guard_{name}"
        if not _close(_float(metrics, key), float(guard[name])):
            raise StageCDecisionError(
                f"{arm} stored {key} differs from deterministic recomputation"
            )
    expected_passed = float(float(guard["violation"]) <= 1e-12)
    if _float(metrics, "selection_stage_c_text_only_guard_passed") != expected_passed:
        raise StageCDecisionError(f"{arm} stored live guard pass flag changed")


def _check(name: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def _pilot_metric_checks(
    metrics: Mapping[str, Mapping[str, Any]], *, source_correct: float, source_off: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    allowed_off = source_off + THRESHOLDS[
        "live_off_maximum_relative_degradation_over_source"
    ] * max(abs(source_off), 1e-8)
    for arm in ARMS:
        row = metrics[arm]
        live = _float(row, METRIC_KEYS["live_off_score"])
        teacher = _float(row, METRIC_KEYS["live_off_teacher_score"])
        allowed = _float(row, METRIC_KEYS["live_off_allowed_score"])
        maximum = _float(
            row, METRIC_KEYS["live_off_guard_max_relative_degradation"]
        )
        passed = _float(row, METRIC_KEYS["live_off_guard_passed"])
        off = _float(row, METRIC_KEYS["off_score"])
        checks.extend(
            (
                _check(f"{arm}.live_off_matches_off", live, off, _close(live, off)),
                _check(
                    f"{arm}.live_off_teacher_is_pinned_source",
                    teacher,
                    source_off,
                    _close(teacher, source_off),
                ),
                _check(
                    f"{arm}.live_off_allowed_is_exact",
                    allowed,
                    allowed_off,
                    _close(allowed, allowed_off),
                ),
                _check(
                    f"{arm}.live_off_relative_degradation",
                    maximum,
                    0.005,
                    _close(maximum, 0.005),
                ),
                _check(
                    f"{arm}.live_off_ceiling",
                    live,
                    {"maximum": allowed_off},
                    live <= allowed_off and passed == 1.0,
                ),
            )
        )
        for invariant in INVARIANT_KEYS:
            observed = _float(row, invariant)
            checks.append(
                _check(f"{arm}.{invariant}", observed, {"exact": 0.0}, observed == 0.0)
            )

    memory_score = _float(metrics["memory"], METRIC_KEYS["selection_score"])
    memory_correct = _float(metrics["memory"], METRIC_KEYS["correct_score"])
    matched_score = _float(metrics["matched_off"], METRIC_KEYS["selection_score"])
    checks.extend(
        (
            _check(
                "memory.selection_score_is_correct_score",
                memory_score,
                memory_correct,
                _close(memory_score, memory_correct),
            ),
            _check(
                "memory.correct_improvement_over_matched_off",
                _relative_improvement(memory_score, matched_score),
                {"minimum": 0.001},
                _relative_improvement(memory_score, matched_score) >= 0.001,
            ),
            _check(
                "memory.correct_improvement_over_stage_b_source",
                _relative_improvement(memory_score, source_correct),
                {"minimum": 0.001},
                _relative_improvement(memory_score, source_correct) >= 0.001,
            ),
            _check(
                "memory.correct_vs_off_relative_gain",
                _float(metrics["memory"], METRIC_KEYS["correct_vs_off_relative_gain"]),
                {"minimum": 0.001},
                _float(
                    metrics["memory"], METRIC_KEYS["correct_vs_off_relative_gain"]
                )
                >= 0.001,
            ),
            _check(
                "memory.Rpair",
                _float(metrics["memory"], METRIC_KEYS["rpair"]),
                {"minimum": 0.10},
                _float(metrics["memory"], METRIC_KEYS["rpair"]) >= 0.10,
            ),
        )
    )
    for nonce in range(3):
        observed = _float(metrics["memory"], METRIC_KEYS[f"rpair_n{nonce}"])
        checks.append(
            _check(f"memory.Rpair_n{nonce}", observed, {"minimum": 0.05}, observed >= 0.05)
        )
    for hand in ("lhand", "rhand"):
        observed = _float(
            metrics["memory"],
            METRIC_KEYS[f"{hand}_negative_degradation_vs_off"],
        )
        checks.append(
            _check(
                f"memory.{hand}_negative_degradation_vs_off",
                observed,
                {"minimum": -0.02},
                observed >= -0.02,
            )
        )
    return checks


def _validate_current_execution_scope(
    *,
    complete: Mapping[str, Any],
    execution_complete: Path,
    active_lease_attestation: Path,
    expected_source_git_head: str,
    expected_source_remote_ref: str,
    expected_source_remote_head: str,
    expected_pair_constraint: str,
) -> None:
    """Reject reuse from any other source, foundation, config, or pair."""

    from NIAF.continuous_trajectory_field.scripts import stage_c_execution_control

    attestation = _json(active_lease_attestation, "active execution lease attestation")
    lease_path = Path(str(attestation.get("lease_path", "")))
    try:
        _validated_attestation, claim = stage_c_execution_control.validate_attestation(
            active_lease_attestation, lease_path=lease_path
        )
    except stage_c_execution_control.StageCExecutionControlError as error:
        if lease_path.exists():
            raise StageCDecisionError(
                "active execution lease attestation changed"
            ) from error
        try:
            _validated_attestation, claim = (
                stage_c_execution_control.validate_finalized_attestation(
                    active_lease_attestation, lease_path=lease_path
                )
            )
        except stage_c_execution_control.StageCExecutionControlError as final_error:
            raise StageCDecisionError(
                "execution lease attestation has no exact active or released claim"
            ) from final_error
    binding = claim["execution_binding"]
    head = expected_source_git_head.lower()
    remote_head = expected_source_remote_head.lower()
    execution_claim = complete.get("active_lease_claim_identity")
    replaced = claim.get("replaced_stale_owner") or {}
    if (
        binding.get("source_git_head") != head
        or binding.get("source_remote_ref") != expected_source_remote_ref
        or binding.get("source_remote_head") != remote_head
        or binding.get("pair_constraint") != expected_pair_constraint
        or complete.get("source_git_head") != head
        or complete.get("source_remote_ref") != expected_source_remote_ref
        or complete.get("source_remote_head") != remote_head
        or complete.get("pair_constraint") != expected_pair_constraint
        or execution_claim
        not in {
            claim["claim_identity"],
            replaced.get("claim_identity"),
            *claim.get("prior_claim_identities", []),
        }
    ):
        raise StageCDecisionError(
            "reusable execution differs from the active source/pair lease"
        )
    launch = _json(execution_complete.parent / "LAUNCH.json", "execution launch")
    launch_artifacts = launch.get("artifacts") or {}
    for artifact_name, binding_name in (
        ("memory_config", "memory_config_sha256"),
        ("matched_off_config", "matched_off_config_sha256"),
        ("cpu_gate", "cpu_gate_sha256"),
        ("calibration_completion", "calibration_completion_sha256"),
    ):
        if (launch_artifacts.get(artifact_name) or {}).get("sha256") != binding.get(
            binding_name
        ):
            raise StageCDecisionError(
                f"reusable execution changed active {artifact_name} binding"
            )
    policy_path = Path(
        str((complete.get("decision_policy") or {}).get("path", ""))
    )
    if policy_path.name in SMOKE_PREREQUISITE_POLICY_NAMES:
        mode = complete.get("execution_mode")
        if mode == "pilot":
            prior_smoke = complete.get("prior_one_update_smoke")
            if launch.get("prior_one_update_smoke") != prior_smoke:
                raise StageCDecisionError(
                    "reusable pilot changed its one-update smoke prerequisite"
                )
            _validate_pilot_smoke_prerequisite(
                value=prior_smoke,
                complete=complete,
                launch=launch,
            )
        elif mode == "smoke":
            if any(
                "prior_one_update_smoke" in value for value in (complete, launch)
            ):
                raise StageCDecisionError(
                    "reusable smoke contains a forbidden self-prerequisite"
                )
        else:
            raise StageCDecisionError(
                "reusable protocol-v2 execution mode is not exact"
            )


def make_decision(
    *,
    mode: str,
    execution_complete: Path,
    source_checkpoint: Path,
    memory_checkpoint: Path,
    matched_off_checkpoint: Path,
    policy_path: Path,
    active_lease_attestation: Path,
    expected_source_git_head: str,
    expected_source_remote_ref: str,
    expected_source_remote_head: str,
    expected_pair_constraint: str,
) -> dict[str, Any]:
    if mode not in EXPECTED_STEPS:
        raise StageCDecisionError("Stage-C decision mode must be smoke or pilot")
    policy = validate_policy(policy_path)
    complete = _json(execution_complete, "paired execution COMPLETE")
    if (
        complete.get("schema_name")
        != "signtrajfield_centered_stage_c_paired_execution_complete"
        or complete.get("schema_version") != 1
        or complete.get("execution_mode") != mode
        or complete.get("development_only") is not True
        or complete.get("non_authorizing") is not True
        or complete.get("promotion_eligible") is not False
        or complete.get("confirmation_manifest_opened") is not False
        or complete.get("test_data_accessed") is not False
        or re.fullmatch(r"[0-9a-f]{64}", str(complete.get("active_lease_claim_identity", "")))
        is None
        or complete.get("decision_policy", {}).get("sha256")
        != sha256_file(policy_path)
    ):
        raise StageCDecisionError("paired execution COMPLETE scope/binding is invalid")
    _validate_current_execution_scope(
        complete=complete,
        execution_complete=execution_complete,
        active_lease_attestation=active_lease_attestation,
        expected_source_git_head=expected_source_git_head,
        expected_source_remote_ref=expected_source_remote_ref,
        expected_source_remote_head=expected_source_remote_head,
        expected_pair_constraint=expected_pair_constraint,
    )
    network_summary = _validate_network_evidence(complete, execution_complete, mode)
    expected_step = EXPECTED_STEPS[mode]
    summaries = complete.get("arm_execution_summaries")
    if summaries != {
        arm: {
            "epoch": 1,
            "global_step": expected_step,
            "data_limit_train": 0,
            "data_limit_val": 0,
            "train_max_batches": 2 if mode == "smoke" else 0,
            "eval_max_batches": 1 if mode == "smoke" else 0,
            "length_bucketed_batches": True,
            "drop_last": False,
        }
        for arm in ARMS
    }:
        raise StageCDecisionError("paired execution COMPLETE has wrong arm semantics")
    source_sha = sha256_file(_regular(source_checkpoint, "Stage-B source checkpoint"))
    if source_sha != policy["source_checkpoint"]["sha256"]:
        raise StageCDecisionError("Stage-B source checkpoint SHA256 changed")
    arm_paths = {"memory": memory_checkpoint, "matched_off": matched_off_checkpoint}
    arms = complete.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise StageCDecisionError("paired execution COMPLETE arm set is not exact")
    checkpoints: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        path = arm_paths[arm]
        if arms[arm].get("checkpoints/last.pt") != sha256_file(path):
            raise StageCDecisionError(f"{arm} checkpoint hash differs from COMPLETE")
        checkpoints[arm] = _load_checkpoint(path, f"{arm} checkpoint")
        metrics[arm], identities[arm] = _audit_arm_checkpoint(
            checkpoints[arm], arm=arm, mode=mode
        )
    if _normalize_config(identities["memory"]["config"]) != _normalize_config(
        identities["matched_off"]["config"]
    ):
        raise StageCDecisionError("Stage-C arm configs differ beyond the approved switch")
    for name in (
        "calibration",
        "distribution",
        "architecture",
        "trainability",
        "optimizer_mapping",
    ):
        if identities["memory"][name] != identities["matched_off"][name]:
            raise StageCDecisionError(f"Stage-C arms bind different {name} identities")
    for arm in ARMS:
        provenance = identities[arm]["provenance"]
        if provenance.get("source_checkpoint", {}).get("sha256") != SOURCE_SHA256:
            raise StageCDecisionError(f"{arm} provenance lost the pinned source")
    memory_provenance = copy.deepcopy(identities["memory"]["provenance"])
    matched_provenance = copy.deepcopy(identities["matched_off"]["provenance"])
    for value in (memory_provenance, matched_provenance):
        value.pop("digest", None)
        value.get("declared_contract", {}).pop("arm", None)
    if memory_provenance != matched_provenance:
        raise StageCDecisionError("Stage-C arm provenance differs beyond arm")
    if _normalize_named(
        identities["memory"]["objective"],
        "arm",
        "sentence_memory_train_mode",
        "sentence_memory_dropout_probability",
    ) != _normalize_named(
        identities["matched_off"]["objective"],
        "arm",
        "sentence_memory_train_mode",
        "sentence_memory_dropout_probability",
    ):
        raise StageCDecisionError("Stage-C arm objectives differ beyond availability")

    source = _load_checkpoint(source_checkpoint, "Stage-B source checkpoint")
    _finite_numbers(source.get("model"), "source.model")
    if source.get("epoch") != 5 or source.get("global_step") != 360:
        raise StageCDecisionError("Stage-B source checkpoint epoch/step changed")
    source_metrics = source.get("metrics")
    source_config = source.get("config")
    if not isinstance(source_metrics, dict) or not isinstance(source_config, dict):
        raise StageCDecisionError("Stage-B source lacks metrics/config")
    source_correct = _float(source_metrics, "selection_score")
    raw_validation_metrics = {
        name.removeprefix("val_"): value
        for name, value in source_metrics.items()
        if name.startswith("val_")
    }
    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    recomputed, _violation, _feasible, details = trainer.checkpoint_selection_diagnostics(
        raw_validation_metrics, source_config, return_details=True
    )
    source_off = float(details.get("dual_mode", {}).get("off_score", math.nan))
    if not math.isfinite(source_off) or not _close(recomputed, source_correct):
        raise StageCDecisionError("Stage-B source selection/off score cannot be reproduced")
    for arm in ARMS:
        _verify_selection_recomputation(
            metrics[arm], identities[arm]["config"], trainer, arm
        )

    checks = [
        _check("integrity.scope_and_config", True, True, True),
        _check("integrity.all_serialized_numbers_finite", True, True, True),
        _check("integrity.roce_artifact_set_and_semantics", True, True, True),
        _check("execution.exact_epoch", 1, 1, True),
        _check("execution.exact_global_step", expected_step, expected_step, True),
    ]
    audit_key = (
        "smoke_checkpoint_audit" if mode == "smoke" else "pilot_checkpoint_audit"
    )
    audit_name = (
        "SMOKE_CHECKPOINT_AUDIT.json"
        if mode == "smoke"
        else "PILOT_CHECKPOINT_AUDIT.json"
    )
    audit_hash = complete.get("execution_artifacts", {}).get(audit_key)
    audit_path = execution_complete.parent / "network_preflight" / audit_name
    audit = _json(audit_path, f"strict {mode} checkpoint audit")
    expected_audit_schema = (
        "signtrajfield_stage_c_one_update_checkpoint_audit"
        if mode == "smoke"
        else "signtrajfield_stage_c_full_epoch_checkpoint_audit"
    )
    audit_passed = (
        audit_hash == sha256_file(audit_path)
        and audit.get("schema_name") == expected_audit_schema
        and audit.get("execution_mode") == mode
        and audit.get("expected_global_step") == expected_step
        and audit.get("schema_version") == 1
        and audit.get("exact_model_tensor_count") == 233
        and audit.get("exact_trainable_tensor_count") == 20
        and audit.get("exact_changed_trainable_tensor_count") == 20
        and audit.get("exact_trainable_parameter_count") == 1_109_395
        and audit.get("exact_frozen_model_tensor_count") == 213
        and audit.get("exact_frozen_nonmemory_base_tensor_count") == 151
        and audit.get("exact_frozen_sentence_memory_tensor_count") == 62
        and audit.get("all_model_and_optimizer_tensors_finite") is True
        and audit.get(
            "all_sentence_memory_and_nonapproved_tensors_bitwise_frozen"
        )
        is True
        and audit.get("all_trainables_have_nonzero_finite_optimizer_moments")
        is True
        and audit.get("canonical_best_checkpoint_published") is False
        and audit.get("development_only") is True
        and audit.get("non_authorizing") is True
    )
    if mode == "smoke":
        audit_passed = bool(
            audit_passed
            and audit.get("all_trainables_have_nonzero_finite_one_step_moments")
            is True
        )
    checks.append(
        _check(
            f"{mode}.strict_checkpoint_audit_bound",
            audit_hash,
            "sha256",
            audit_passed,
        )
    )
    if mode == "pilot":
        checks.extend(
            _pilot_metric_checks(
                metrics, source_correct=source_correct, source_off=source_off
            )
        )
    passed = all(row["passed"] for row in checks)
    if passed and mode == "smoke":
        status = "smoke_ready"
        next_permitted_action = "run_one_epoch_development_pilot"
    elif passed:
        status = "pilot_complete_development_signal"
        next_permitted_action = (
            "none_requires_fresh_preregistration_without_pilot_outcome_access"
        )
    else:
        status = "stop"
        next_permitted_action = "none"
    payload = {
        "schema_name": DECISION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "execution_mode": mode,
        "status": status,
        "next_permitted_action": next_permitted_action,
        "source_git_head": complete["source_git_head"],
        "pair_constraint": complete["pair_constraint"],
        "active_lease_claim_identity": complete["active_lease_claim_identity"],
        "execution_complete": {
            "path": str(execution_complete.resolve()),
            "sha256": sha256_file(execution_complete),
        },
        "policy": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
        },
        "source_checkpoint": {
            "path": str(source_checkpoint.resolve()),
            "sha256": source_sha,
            "selection_score": source_correct,
            "off_score": source_off,
        },
        "roce_evidence": network_summary,
        "checks": checks,
        "failed_checks": [row["name"] for row in checks if not row["passed"]],
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "authorized_purpose": None,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    if policy_path.name in SMOKE_PREREQUISITE_POLICY_NAMES and mode == "pilot":
        payload["prior_one_update_smoke"] = complete["prior_one_update_smoke"]
    return {**payload, "decision_identity": digest_json(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--mode", choices=tuple(EXPECTED_STEPS), required=True)
        command.add_argument("--execution_complete", type=Path, required=True)
        command.add_argument("--source_checkpoint", type=Path, required=True)
        command.add_argument("--memory_checkpoint", type=Path, required=True)
        command.add_argument("--matched_off_checkpoint", type=Path, required=True)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--active_lease_attestation", type=Path, required=True)
        command.add_argument("--expected_source_git_head", required=True)
        command.add_argument("--expected_source_remote_ref", required=True)
        command.add_argument("--expected_source_remote_head", required=True)
        command.add_argument("--expected_pair_constraint", required=True)

    create = commands.add_parser("create")
    add_inputs(create)
    create.add_argument("--out_file", type=Path, required=True)
    verify = commands.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--decision", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = make_decision(
        mode=args.mode,
        execution_complete=args.execution_complete,
        source_checkpoint=args.source_checkpoint,
        memory_checkpoint=args.memory_checkpoint,
        matched_off_checkpoint=args.matched_off_checkpoint,
        policy_path=args.policy,
        active_lease_attestation=args.active_lease_attestation,
        expected_source_git_head=args.expected_source_git_head,
        expected_source_remote_ref=args.expected_source_remote_ref,
        expected_source_remote_head=args.expected_source_remote_head,
        expected_pair_constraint=args.expected_pair_constraint,
    )
    if args.command == "verify":
        observed = _json(args.decision, "Stage-C immutable decision")
        if observed != result:
            raise StageCDecisionError(
                "stored Stage-C decision differs from deterministic recomputation"
            )
    else:
        encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        publish_bytes_no_replace(args.out_file, encoded)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
