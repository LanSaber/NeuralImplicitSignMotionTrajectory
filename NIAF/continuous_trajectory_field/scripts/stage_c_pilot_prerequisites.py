"""Exact prerequisite-chain validation for the non-authorizing Stage-C pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from NIAF.continuous_trajectory_field.relevance_calibration import (
    validate_relevance_calibration_artifact,
    validate_relevance_calibration_source,
)
from NIAF.continuous_trajectory_field.scripts.stage_c_pilot_decision import (
    digest_json,
    validate_policy,
)
from NIAF.continuous_trajectory_field.scripts.stage_c_calibration_transition import (
    build_transition_audit,
)
from NIAF.continuous_trajectory_field.scripts.stage_c_calibration_control import (
    CalibrationLeaseError,
    _validate_binding as validate_calibration_lease_binding,
)
from NIAF.continuous_trajectory_field.scripts.stage_c_execution_control import (
    StageCExecutionControlError,
    _validate_binding as validate_execution_lease_binding,
)


class PrerequisiteError(RuntimeError):
    """A Stage-C prerequisite is missing, stale, mutable, or out of scope."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise PrerequisiteError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _exact_json(path: Path, *, label: str, fields: set[str]) -> dict:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrerequisiteError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict) or set(value) != fields:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PrerequisiteError(
            f"{label} fields differ: expected={sorted(fields)}, observed={observed}"
        )
    return value


def validate_cpu_gate(
    *,
    cpu_gate: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
) -> dict:
    """Validate the exact immutable-source CPU gate before downstream writes."""

    head = source_git_head.lower()
    remote_head = source_remote_head.lower()
    cpu = _exact_json(
        cpu_gate,
        label="Stage-C CPU gate",
        fields={
            "schema_name",
            "schema_version",
            "source_git_head",
            "source_remote_ref",
            "source_remote_head",
            "slurm_job_id",
            "compileall",
            "ruff_version",
            "pytest_version",
            "complete_repository_test_glob",
            "development_only",
            "confirmation_manifest_opened",
            "test_data_accessed",
            "passed",
        },
    )
    if cpu != {
        **cpu,
        "schema_name": "signtrajfield_stage_c_cpu_gate",
        "schema_version": 1,
        "source_git_head": head,
        "source_remote_ref": source_remote_ref,
        "source_remote_head": remote_head,
        "compileall": True,
        "ruff_version": "0.12.0",
        "pytest_version": "8.4.2",
        "complete_repository_test_glob": "tests/test_*.py",
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "passed": True,
    } or not str(cpu["slurm_job_id"]).isdigit():
        raise PrerequisiteError("Stage-C CPU gate content is not exact/current")
    return cpu


def validate_foundation(
    *,
    cpu_gate: Path,
    calibration_completion: Path,
    calibration_dir: Path,
    calibration_launcher: Path,
    stage_c_config: Path,
    source_terminal_decision: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
) -> dict:
    head = source_git_head.lower()
    remote_head = source_remote_head.lower()
    validate_cpu_gate(
        cpu_gate=cpu_gate,
        source_git_head=head,
        source_remote_ref=source_remote_ref,
        source_remote_head=remote_head,
    )

    completion = _exact_json(
        calibration_completion,
        label="Stage-C calibration completion",
        fields={
            "schema_name",
            "schema_version",
            "source_git_head",
            "source_remote_ref",
            "source_remote_head",
            "artifact_path",
            "artifact_identity",
            "calibration_sha256",
            "map_sha256",
            "ready_sha256",
            "cpu_gate_sha256",
            "launcher_sha256",
            "calibration_transition_path",
            "calibration_transition_sha256",
            "calibration_transition_identity",
            "active_lease_claim_identity",
            "lease_attestation_path",
            "lease_attestation_sha256",
            "lease_attestation_identity",
            "train_only",
            "development_only",
            "confirmation_manifest_opened",
            "test_data_accessed",
        },
    )
    calibration_dir = calibration_dir.resolve()
    _regular_file(calibration_launcher, "Stage-C calibration launcher")
    artifact = validate_relevance_calibration_artifact(calibration_dir)
    validate_relevance_calibration_source(
        artifact,
        source_root=source_root,
        expected_git_head=head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=remote_head,
        expected_source_file_profile="stage_c_generator_adaptation_v1",
    )
    exact_completion = {
        "schema_name": "signtrajfield_stage_c_calibration_completion",
        "schema_version": 1,
        "source_git_head": head,
        "source_remote_ref": source_remote_ref,
        "source_remote_head": remote_head,
        "artifact_path": str(calibration_dir),
        "artifact_identity": artifact["identity"],
        "calibration_sha256": sha256_file(calibration_dir / "calibration.json"),
        "map_sha256": sha256_file(calibration_dir / "calibration_map.npz"),
        "ready_sha256": sha256_file(calibration_dir / "READY"),
        "cpu_gate_sha256": sha256_file(cpu_gate),
        "launcher_sha256": sha256_file(calibration_launcher),
        "calibration_transition_path": completion["calibration_transition_path"],
        "calibration_transition_sha256": completion[
            "calibration_transition_sha256"
        ],
        "calibration_transition_identity": completion[
            "calibration_transition_identity"
        ],
        "active_lease_claim_identity": completion["active_lease_claim_identity"],
        "lease_attestation_path": completion["lease_attestation_path"],
        "lease_attestation_sha256": completion["lease_attestation_sha256"],
        "lease_attestation_identity": completion["lease_attestation_identity"],
        "train_only": True,
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    if completion != exact_completion:
        raise PrerequisiteError("Stage-C calibration completion binding is not exact")
    transition_path = _regular_file(
        Path(completion["calibration_transition_path"]),
        "Stage-C calibration transition audit",
    )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    recomputed_transition = build_transition_audit(
        source_terminal_decision=source_terminal_decision,
        stage_c_config=stage_c_config,
    )
    if (
        transition != recomputed_transition
        or completion["calibration_transition_sha256"]
        != sha256_file(transition_path)
        or completion["calibration_transition_identity"]
        != transition.get("audit_identity")
    ):
        raise PrerequisiteError("Stage-C calibration transition is not exact")
    attestation_path = _regular_file(
        Path(completion["lease_attestation_path"]),
        "Stage-C calibration lease attestation",
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    unsigned_attestation = {
        key: value for key, value in attestation.items() if key != "attestation_identity"
    }
    claim = attestation.get("claim") or {}
    unsigned_claim = {
        key: value for key, value in claim.items() if key != "claim_identity"
    }
    try:
        validate_calibration_lease_binding(claim.get("binding"))
    except CalibrationLeaseError as error:
        raise PrerequisiteError(
            "Stage-C calibration claim binding changed"
        ) from error
    if (
        set(attestation)
        != {
            "schema_name",
            "schema_version",
            "lease_path",
            "current_slurm_job_id",
            "current_slurm_restart_count",
            "claim",
            "restart_reconciliation",
            "attestation_identity",
        }
        or set(claim)
        != {
            "schema_name",
            "schema_version",
            "slurm_job_id",
            "slurm_restart_count",
            "binding",
            "claim_nonce",
            "created_unix_ns",
            "claim_identity",
        }
        or attestation.get("schema_name")
        != "signtrajfield_stage_c_calibration_lease_attestation"
        or attestation.get("schema_version") != 1
        or claim.get("schema_name") != "signtrajfield_stage_c_calibration_lease"
        or claim.get("schema_version") != 1
        or attestation.get("lease_path")
        != str((calibration_completion.parent / "active_execution_lease").resolve())
        or completion["lease_attestation_sha256"] != sha256_file(attestation_path)
        or completion["lease_attestation_identity"]
        != attestation.get("attestation_identity")
        or attestation.get("attestation_identity") != digest_json(unsigned_attestation)
        or claim.get("claim_identity") != digest_json(unsigned_claim)
        or claim.get("claim_identity") != completion["active_lease_claim_identity"]
        or claim.get("binding", {}).get("source_git_head") != head
        or claim.get("binding", {}).get("cpu_gate_sha256") != sha256_file(cpu_gate)
        or claim.get("binding", {}).get("launcher_sha256")
        != sha256_file(calibration_launcher)
    ):
        raise PrerequisiteError("Stage-C calibration lease attestation changed")
    history = calibration_completion.parent / "active_execution_lease.history"
    terminal_dirs = [
        path
        for suffix in ("released", "recovered_released")
        for path in history.glob(f"{claim['claim_identity']}.{suffix}")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(terminal_dirs) != 1:
        raise PrerequisiteError("Stage-C calibration lease lacks terminal archive")
    terminal = json.loads(
        (terminal_dirs[0] / "TERMINAL.json").read_text(encoding="utf-8")
    )
    unsigned_terminal = {
        key: value for key, value in terminal.items() if key != "terminal_identity"
    }
    if (
        terminal.get("terminal_identity") != digest_json(unsigned_terminal)
        or terminal.get("claim_identity") != claim["claim_identity"]
        or terminal.get("outcome") not in {"released", "recovered_released"}
        or terminal.get("completion_path") != str(calibration_completion.resolve())
        or terminal.get("completion_sha256") != sha256_file(calibration_completion)
    ):
        raise PrerequisiteError("Stage-C calibration terminal archive changed")
    return {
        "cpu_gate_sha256": sha256_file(cpu_gate),
        "calibration_completion_sha256": sha256_file(calibration_completion),
        "calibration_identity": artifact["identity"],
    }


def _require_hashes(directory: Path, observed: dict, expected: dict[str, str]) -> None:
    if set(observed) != set(expected):
        raise PrerequisiteError("Stage-C completed artifact set is not exact")
    for name, relative in expected.items():
        path = _regular_file(directory / relative, f"Stage-C smoke artifact {name}")
        if observed[name] != sha256_file(path):
            raise PrerequisiteError(f"Stage-C smoke artifact hash changed: {name}")


def validate_smoke(
    *,
    smoke_ready: Path,
    smoke_root: Path,
    source_git_head: str,
    pair_constraint: str,
) -> dict:
    head = source_git_head.lower()
    ready = _exact_json(
        smoke_ready,
        label="Stage-C smoke READY",
        fields={
            "schema_name",
            "schema_version",
            "execution_mode",
            "source_git_head",
            "pair_constraint",
            "active_lease_claim_identity",
            "execution_lease_claim_identity",
            "complete_path",
            "complete_sha256",
            "decision_path",
            "decision_sha256",
            "decision_identity",
            "decision_status",
            "expected_epoch",
            "expected_global_step_per_arm",
            "one_optimizer_update_per_arm",
            "one_full_train_epoch_per_arm",
            "development_only",
            "non_authorizing",
            "promotion_eligible",
            "authorized_purpose",
            "confirmation_manifest_opened",
            "test_data_accessed",
        },
    )
    if (
        ready["schema_name"] != "signtrajfield_stage_c_mode_ready"
        or ready["schema_version"] != 1
        or ready["execution_mode"] != "smoke"
        or ready["source_git_head"] != head
        or ready["pair_constraint"] != pair_constraint
        or ready["decision_status"] != "smoke_ready"
        or ready["expected_epoch"] != 1
        or ready["expected_global_step_per_arm"] != 1
        or ready["one_optimizer_update_per_arm"] is not True
        or ready["one_full_train_epoch_per_arm"] is not False
        or ready["development_only"] is not True
        or ready["non_authorizing"] is not True
        or ready["promotion_eligible"] is not False
        or ready["authorized_purpose"] is not None
        or ready["confirmation_manifest_opened"] is not False
        or ready["test_data_accessed"] is not False
    ):
        raise PrerequisiteError("Stage-C smoke READY content is not exact/current")

    mode_complete_path = Path(ready["complete_path"])
    mode_complete = _exact_json(
        mode_complete_path,
        label="Stage-C smoke mode COMPLETE",
        fields={
            "schema_name",
            "schema_version",
            "execution_mode",
            "source_git_head",
            "pair_constraint",
            "slurm_job_id",
            "slurm_restart_count",
            "active_lease_claim_identity",
            "execution_lease_claim_identity",
            "lease_attestation_path",
            "lease_attestation_sha256",
            "lease_attestation_identity",
            "execution_complete_path",
            "execution_complete_sha256",
            "decision_path",
            "decision_sha256",
            "decision_identity",
            "decision_status",
            "decision_policy_path",
            "decision_policy_sha256",
            "expected_epoch",
            "expected_global_step_per_arm",
            "world_size",
            "batch_per_rank",
            "accumulation_steps",
            "effective_global_batch",
            "development_only",
            "non_authorizing",
            "promotion_eligible",
            "authorized_purpose",
            "confirmation_manifest_opened",
            "test_data_accessed",
        },
    )
    if (
        mode_complete_path.resolve()
        != (smoke_ready.parent / "COMPLETE.json").resolve()
        or ready["complete_sha256"] != sha256_file(mode_complete_path)
        or mode_complete["schema_name"] != "signtrajfield_stage_c_mode_complete"
        or mode_complete["schema_version"] != 1
        or mode_complete["execution_mode"] != "smoke"
        or mode_complete["source_git_head"] != head
        or mode_complete["pair_constraint"] != pair_constraint
        or mode_complete["active_lease_claim_identity"]
        != ready["active_lease_claim_identity"]
        or mode_complete["execution_lease_claim_identity"]
        != ready["execution_lease_claim_identity"]
        or mode_complete["decision_path"] != ready["decision_path"]
        or mode_complete["decision_sha256"] != ready["decision_sha256"]
        or mode_complete["decision_identity"] != ready["decision_identity"]
        or mode_complete["decision_status"] != "smoke_ready"
        or mode_complete["expected_epoch"] != 1
        or mode_complete["expected_global_step_per_arm"] != 1
        or mode_complete["world_size"] != 2
        or mode_complete["batch_per_rank"] != 64
        or mode_complete["accumulation_steps"] != 2
        or mode_complete["effective_global_batch"] != 256
        or mode_complete["development_only"] is not True
        or mode_complete["non_authorizing"] is not True
        or mode_complete["promotion_eligible"] is not False
        or mode_complete["authorized_purpose"] is not None
        or mode_complete["confirmation_manifest_opened"] is not False
        or mode_complete["test_data_accessed"] is not False
    ):
        raise PrerequisiteError("Stage-C smoke mode publication is not exact")

    decision_path = _regular_file(
        Path(mode_complete["decision_path"]), "Stage-C smoke decision"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision_unsigned = {
        key: value for key, value in decision.items() if key != "decision_identity"
    }
    if (
        mode_complete["decision_sha256"] != sha256_file(decision_path)
        or decision.get("decision_identity") != digest_json(decision_unsigned)
        or decision.get("decision_identity") != mode_complete["decision_identity"]
        or decision.get("execution_mode") != "smoke"
        or decision.get("status") != "smoke_ready"
        or decision.get("next_permitted_action")
        != "run_one_epoch_development_pilot"
        or decision.get("active_lease_claim_identity")
        != mode_complete["execution_lease_claim_identity"]
        or decision.get("development_only") is not True
        or decision.get("non_authorizing") is not True
        or decision.get("promotion_eligible") is not False
        or decision.get("authorized_purpose") is not None
        or decision.get("confirmation_manifest_opened") is not False
        or decision.get("test_data_accessed") is not False
    ):
        raise PrerequisiteError("Stage-C smoke decision is not exact/non-authorizing")
    policy_path = _regular_file(
        Path(mode_complete["decision_policy_path"]), "Stage-C decision policy"
    )
    validate_policy(policy_path)
    if mode_complete["decision_policy_sha256"] != sha256_file(policy_path):
        raise PrerequisiteError("Stage-C decision policy hash changed")

    attestation_path = _regular_file(
        Path(mode_complete["lease_attestation_path"]),
        "Stage-C smoke lease attestation",
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation_unsigned = {
        key: value for key, value in attestation.items() if key != "attestation_identity"
    }
    claim = attestation.get("claim") or {}
    claim_unsigned = {
        key: value for key, value in claim.items() if key != "claim_identity"
    }
    try:
        validate_execution_lease_binding(claim.get("execution_binding"))
    except StageCExecutionControlError as error:
        raise PrerequisiteError("Stage-C smoke execution binding changed") from error
    if (
        set(attestation)
        != {
            "schema_name",
            "schema_version",
            "lease_path",
            "current_slurm_job_id",
            "current_slurm_restart_count",
            "claim",
            "restart_reconciliation",
            "attestation_identity",
        }
        or set(claim)
        != {
            "schema_name",
            "schema_version",
            "mode",
            "slurm_job_id",
            "slurm_restart_count",
            "execution_binding",
            "claim_nonce",
            "created_unix_ns",
            "replaced_stale_owner",
            "prior_claim_identities",
            "claim_identity",
        }
        or attestation.get("schema_name")
        != "signtrajfield_stage_c_execution_lease_attestation"
        or attestation.get("schema_version") != 1
        or claim.get("schema_name") != "signtrajfield_stage_c_execution_lease"
        or claim.get("schema_version") != 1
        or claim.get("mode") != "smoke"
        or attestation.get("lease_path")
        != str((smoke_ready.parent.parent / "active_execution_lease").resolve())
        or attestation.get("current_slurm_job_id")
        != mode_complete["slurm_job_id"]
        or attestation.get("current_slurm_restart_count")
        != mode_complete["slurm_restart_count"]
        or mode_complete["lease_attestation_sha256"] != sha256_file(attestation_path)
        or attestation.get("attestation_identity") != digest_json(attestation_unsigned)
        or attestation.get("attestation_identity")
        != mode_complete["lease_attestation_identity"]
        or claim.get("claim_identity") != digest_json(claim_unsigned)
        or claim.get("claim_identity") != mode_complete["active_lease_claim_identity"]
        or claim.get("mode") != "smoke"
        or claim.get("execution_binding", {}).get("source_git_head") != head
        or claim.get("execution_binding", {}).get("pair_constraint") != pair_constraint
    ):
        raise PrerequisiteError("Stage-C smoke lease attestation is not exact")

    complete_path = Path(mode_complete["execution_complete_path"])
    _regular_file(complete_path, "Stage-C paired smoke execution COMPLETE")
    attempt = complete_path.parent
    if (
        mode_complete["execution_complete_sha256"] != sha256_file(complete_path)
        or attempt.parent.name != "executions"
        or not attempt.name.startswith("restart_")
        or attempt.parent.parent.parent.name != "attempts"
        or attempt.parents[3].name != f"source_{head}"
        or attempt.parents[4].name != "sources"
        or attempt.parents[5].resolve() != smoke_root.resolve()
    ):
        raise PrerequisiteError("Stage-C smoke execution lies outside its result root")
    complete = _exact_json(
        complete_path,
        label="Stage-C paired smoke execution COMPLETE",
        fields={
            "schema_name",
            "schema_version",
            "development_only",
            "non_authorizing",
            "promotion_eligible",
            "execution_mode",
            "source_git_head",
            "source_remote_ref",
            "source_remote_head",
            "pair_constraint",
            "world_size",
            "batch_per_rank",
            "accumulation_steps",
            "effective_global_batch",
            "confirmation_manifest_opened",
            "test_data_accessed",
            "active_lease_claim_identity",
            "lease_attestation_path",
            "lease_attestation_sha256",
            "lease_attestation_identity",
            "decision_policy",
            "arms",
            "arm_execution_summaries",
            "execution_artifacts",
        },
    )
    if {
        "schema_name": complete["schema_name"],
        "schema_version": complete["schema_version"],
        "development_only": complete["development_only"],
        "non_authorizing": complete["non_authorizing"],
        "promotion_eligible": complete["promotion_eligible"],
        "execution_mode": complete["execution_mode"],
        "source_git_head": complete["source_git_head"],
        "pair_constraint": complete["pair_constraint"],
        "confirmation_manifest_opened": complete["confirmation_manifest_opened"],
        "test_data_accessed": complete["test_data_accessed"],
        "world_size": complete["world_size"],
        "batch_per_rank": complete["batch_per_rank"],
        "accumulation_steps": complete["accumulation_steps"],
        "effective_global_batch": complete["effective_global_batch"],
    } != {
        "schema_name": "signtrajfield_centered_stage_c_paired_execution_complete",
        "schema_version": 1,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "execution_mode": "smoke",
        "source_git_head": head,
        "pair_constraint": pair_constraint,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
    } or (
        complete["source_remote_head"] != head
        or not str(complete["source_remote_ref"]).startswith("origin/")
        or complete["lease_attestation_path"]
        != mode_complete["lease_attestation_path"]
        or complete["lease_attestation_sha256"]
        != mode_complete["lease_attestation_sha256"]
        or complete["lease_attestation_identity"]
        != mode_complete["lease_attestation_identity"]
    ):
        raise PrerequisiteError("Stage-C smoke COMPLETE scope is not exact")
    expected_summaries = {
        arm: {
            "epoch": 1,
            "global_step": 1,
            "data_limit_train": 0,
            "data_limit_val": 0,
            "train_max_batches": 2,
            "eval_max_batches": 1,
            "length_bucketed_batches": True,
            "drop_last": False,
        }
        for arm in ("memory", "matched_off")
    }
    if (
        complete["active_lease_claim_identity"]
        != mode_complete["execution_lease_claim_identity"]
        or complete["arm_execution_summaries"] != expected_summaries
        or complete["decision_policy"]
        != {"path": str(policy_path.resolve()), "sha256": sha256_file(policy_path)}
    ):
        raise PrerequisiteError("Stage-C smoke execution binding/step semantics changed")
    arm_files = {
        "config.resolved.json": "config.resolved.json",
        "metrics.jsonl": "metrics.jsonl",
        "selection_summary.json": "selection_summary.json",
        "checkpoints/last.pt": "checkpoints/last.pt",
        "checkpoints/epoch0001.pt": "checkpoints/epoch0001.pt",
    }
    if set(complete["arms"]) != {"memory", "matched_off"}:
        raise PrerequisiteError("Stage-C smoke does not contain both exact arms")
    for arm in ("memory", "matched_off"):
        _require_hashes(attempt / arm, complete["arms"][arm], arm_files)

    network_files = {
        "launch": "LAUNCH.json",
        "source_binding": "network_preflight/SOURCE_BINDING.json",
        "pair": "network_preflight/PAIR.json",
        "counter_health": "network_preflight/COUNTER_HEALTH.json",
        "dual_vs_single": "network_preflight/NCCL_DUAL_VS_SINGLE.json",
        "train_memory_logs": "network_preflight/NCCL_LOGS_train_memory.json",
        "train_matched_off_logs": "network_preflight/NCCL_LOGS_train_matched_off.json",
    }
    pair_evidence = json.loads(
        (attempt / "network_preflight/PAIR.json").read_text(encoding="utf-8")
    )
    pair_nodes = pair_evidence.get("nodes")
    if (
        not isinstance(pair_nodes, list)
        or len(pair_nodes) != 2
        or len(set(pair_nodes)) != 2
        or any(
            not isinstance(node, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node) is None
            for node in pair_nodes
        )
    ):
        raise PrerequisiteError("Stage-C smoke pair node set is not exact")
    for index, node in enumerate(pair_nodes):
        network_files[f"node_audit_{index}"] = (
            f"network_preflight/{node}.node.json"
        )
        network_files[f"connectivity_audit_{index}"] = (
            f"network_preflight/{node}.connectivity.json"
        )
        for phase in ("pre", "post"):
            network_files[f"counter_{phase}_{index}"] = (
                f"network_preflight/{node}.counters_{phase}.json"
            )
    for profile in ("single_primary", "single_secondary", "dual"):
        network_files[f"benchmark_{profile}"] = (
            f"network_preflight/NCCL_BENCHMARK_{profile}.json"
        )
        network_files[f"benchmark_logs_{profile}"] = (
            f"network_preflight/NCCL_LOGS_{profile}.json"
        )
    network_files["smoke_checkpoint_audit"] = (
        "network_preflight/SMOKE_CHECKPOINT_AUDIT.json"
    )
    _require_hashes(attempt, complete["execution_artifacts"], network_files)
    from NIAF.continuous_trajectory_field.scripts.stage_c_pilot_decision import (
        StageCDecisionError,
        _validate_network_evidence,
    )

    try:
        _validate_network_evidence(complete, complete_path, "smoke")
    except StageCDecisionError as error:
        raise PrerequisiteError(
            f"Stage-C smoke raw RoCE evidence changed: {error}"
        ) from error

    checkpoint_audit = json.loads(
        (attempt / network_files["smoke_checkpoint_audit"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        checkpoint_audit.get("schema_name")
        != "signtrajfield_stage_c_one_update_checkpoint_audit"
        or checkpoint_audit.get("schema_version") != 1
        or checkpoint_audit.get("execution_mode") != "smoke"
        or checkpoint_audit.get("expected_global_step") != 1
        or checkpoint_audit.get("exact_trainable_tensor_count") != 20
        or checkpoint_audit.get("exact_trainable_parameter_count") != 1_109_395
        or checkpoint_audit.get("exact_model_tensor_count") != 233
        or checkpoint_audit.get("exact_changed_trainable_tensor_count") != 20
        or checkpoint_audit.get("exact_frozen_model_tensor_count") != 213
        or checkpoint_audit.get("exact_frozen_nonmemory_base_tensor_count") != 151
        or checkpoint_audit.get("exact_frozen_sentence_memory_tensor_count") != 62
        or checkpoint_audit.get("all_model_and_optimizer_tensors_finite") is not True
        or checkpoint_audit.get(
            "all_sentence_memory_and_nonapproved_tensors_bitwise_frozen"
        )
        is not True
        or checkpoint_audit.get(
            "all_trainables_have_nonzero_finite_one_step_moments"
        )
        is not True
        or checkpoint_audit.get(
            "all_trainables_have_nonzero_finite_optimizer_moments"
        )
        is not True
        or checkpoint_audit.get("same_source_provenance_except_arm") is not True
        or checkpoint_audit.get(
            "same_calibration_distribution_architecture_memory"
        )
        is not True
        or checkpoint_audit.get(
            "same_objective_except_training_memory_availability"
        )
        is not True
        or checkpoint_audit.get("canonical_best_checkpoint_published") is not False
        or checkpoint_audit.get("development_only") is not True
        or checkpoint_audit.get("non_authorizing") is not True
    ):
        raise PrerequisiteError("Stage-C smoke checkpoint audit is incomplete")

    launch = json.loads((attempt / "LAUNCH.json").read_text(encoding="utf-8"))
    binding = claim["execution_binding"]
    launch_artifacts = launch.get("artifacts") or {}
    if (
        launch.get("schema_name") != "signtrajfield_centered_stage_c_paired_pilot_launch"
        or launch.get("schema_version") != 1
        or launch.get("source_git_head") != head
        or launch.get("pair_constraint") != pair_constraint
        or launch.get("execution_mode") != "smoke"
        or launch.get("development_only") is not True
        or launch.get("non_authorizing") is not True
        or launch.get("promotion_eligible") is not False
        or launch.get("confirmation_or_test_access_permitted") is not False
        or launch.get("active_lease_claim_identity")
        != complete["active_lease_claim_identity"]
        or (launch.get("lease_attestation") or {}).get("sha256")
        != mode_complete["lease_attestation_sha256"]
        or (launch_artifacts.get("memory_config") or {}).get("sha256")
        != binding["memory_config_sha256"]
        or (launch_artifacts.get("matched_off_config") or {}).get("sha256")
        != binding["matched_off_config_sha256"]
        or (launch_artifacts.get("cpu_gate") or {}).get("sha256")
        != binding["cpu_gate_sha256"]
        or (launch_artifacts.get("calibration_completion") or {}).get("sha256")
        != binding["calibration_completion_sha256"]
    ):
        raise PrerequisiteError("Stage-C smoke launch evidence has the wrong scope")
    counters = json.loads(
        (attempt / "network_preflight/COUNTER_HEALTH.json").read_text(encoding="utf-8")
    )
    if counters.get("harmful_counter_increment_count") != 0:
        raise PrerequisiteError("Stage-C smoke contains harmful rail-counter increments")
    comparison = json.loads(
        (attempt / "network_preflight/NCCL_DUAL_VS_SINGLE.json").read_text(
            encoding="utf-8"
        )
    )
    if launch.get("training_network_profile") != comparison.get("training_profile"):
        raise PrerequisiteError("Stage-C smoke training profile differs from benchmark gate")
    for name in (
        "NCCL_LOGS_train_memory.json",
        "NCCL_LOGS_train_matched_off.json",
        "NCCL_LOGS_single_primary.json",
        "NCCL_LOGS_single_secondary.json",
        "NCCL_LOGS_dual.json",
    ):
        value = json.loads((attempt / "network_preflight" / name).read_text(encoding="utf-8"))
        if value.get("socket_fallback_detected") is not False or int(
            value.get("net_ib_line_count", 0)
        ) < 2 or not isinstance(value.get("per_log_evidence"), list) or len(
            value["per_log_evidence"]
        ) != 2 or value.get("hosts") != sorted(pair_nodes) or any(
            row.get("socket_fallback_detected") is not False
            or int(row.get("net_ib_line_count", 0)) < 1
            or row.get("host") not in pair_nodes
            for row in value["per_log_evidence"]
        ):
            raise PrerequisiteError(f"Stage-C smoke lacks positive NET/IB proof: {name}")
    control_dir = smoke_ready.parent.parent
    history = control_dir / "active_execution_lease.history"
    released = [
        path
        for suffix in ("released", "recovered_released")
        for path in history.glob(f"{ready['active_lease_claim_identity']}.{suffix}")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(released) != 1:
        raise PrerequisiteError("Stage-C smoke active lease lacks unique terminal archive")
    terminal = json.loads((released[0] / "TERMINAL.json").read_text(encoding="utf-8"))
    terminal_unsigned = {
        key: value for key, value in terminal.items() if key != "terminal_identity"
    }
    if (
        terminal.get("terminal_identity") != digest_json(terminal_unsigned)
        or terminal.get("claim_identity") != ready["active_lease_claim_identity"]
        or terminal.get("outcome") not in {"released", "recovered_released"}
        or terminal.get("completion_path") != str(mode_complete_path.resolve())
        or terminal.get("completion_sha256") != sha256_file(mode_complete_path)
        or terminal.get("ready_path") != str(smoke_ready.resolve())
        or terminal.get("ready_sha256") != sha256_file(smoke_ready)
    ):
        raise PrerequisiteError("Stage-C smoke terminal lease archive changed")
    return {
        "smoke_ready_sha256": sha256_file(smoke_ready),
        "smoke_complete_sha256": sha256_file(complete_path),
        "smoke_decision_sha256": sha256_file(decision_path),
        "smoke_decision_identity": decision["decision_identity"],
        "smoke_decision_status": decision["status"],
        "training_network_profile": comparison["training_profile"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("validate-cpu")
    cpu.add_argument("--cpu_gate", type=Path, required=True)
    cpu.add_argument("--source_git_head", required=True)
    cpu.add_argument("--source_remote_ref", required=True)
    cpu.add_argument("--source_remote_head", required=True)
    foundation = commands.add_parser("validate-foundation")
    foundation.add_argument("--cpu_gate", type=Path, required=True)
    foundation.add_argument("--calibration_completion", type=Path, required=True)
    foundation.add_argument("--calibration_dir", type=Path, required=True)
    foundation.add_argument("--calibration_launcher", type=Path, required=True)
    foundation.add_argument("--stage_c_config", type=Path, required=True)
    foundation.add_argument("--source_terminal_decision", type=Path, required=True)
    foundation.add_argument("--source_root", type=Path, required=True)
    foundation.add_argument("--source_git_head", required=True)
    foundation.add_argument("--source_remote_ref", required=True)
    foundation.add_argument("--source_remote_head", required=True)
    smoke = commands.add_parser("validate-smoke")
    smoke.add_argument("--smoke_ready", type=Path, required=True)
    smoke.add_argument("--smoke_root", type=Path, required=True)
    smoke.add_argument("--source_git_head", required=True)
    smoke.add_argument("--pair_constraint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = vars(args)
    command = kwargs.pop("command")
    if command == "validate-cpu":
        value = validate_cpu_gate(**kwargs)
    elif command == "validate-foundation":
        value = validate_foundation(**kwargs)
    else:
        value = validate_smoke(**kwargs)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
