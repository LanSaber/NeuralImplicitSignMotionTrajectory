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


SOURCE_TERMINAL_DECISION_FIELDS = {
    "checkpoint",
    "confirmation_manifest_opened",
    "decision_identity",
    "development_feasible",
    "development_integrity",
    "experiment_name",
    "integrity_valid",
    "partition",
    "predecessor_authorization",
    "run_launch_identity",
    "schema_name",
    "schema_version",
    "stage",
    "stage2_launch_input",
    "status",
    "test_data_accessed",
}
SOURCE_TERMINAL_DECISION_SCHEMA = "signtrajfield_centered_memory_ordered_decision"
SOURCE_TERMINAL_DECISION_EXPERIMENT = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "absolute_binding_motion_contrast_v1"
)
SOURCE_CHECKPOINT_SHA256 = (
    "b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202"
)
SOURCE_TERMINAL_DECISION_SHA256 = (
    "8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5"
)
SOURCE_TERMINAL_DECISION_IDENTITY = (
    "7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69"
)
RETRY2_RECOVERY_SCHEMA = "signtrajfield_stage_c_retry2_recovery_evidence"
RUN_R1_SOURCE_GIT_HEAD = "a883baf4a0a95b4ebb2007564837c4d9b4f817cc"
RUN_R1_SOURCE_REMOTE_REF = (
    "origin/codex/csl-daily-centered-generator-stage-c-v1-run-r1"
)


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


def validate_source_terminal_decision(
    *,
    decision_path: Path,
    expected_sha256: str,
    expected_identity: str,
) -> dict:
    """Validate the exact pinned legacy Stage-B decision schema.

    Schema v1 intentionally has no top-level ``authorized_purpose`` field.
    Its absence, together with the explicit no-confirmation/no-test fields and
    infeasible status, is the pinned non-authorizing state.  Adding the field,
    even as null, is a schema change and is rejected here.
    """

    decision = _exact_json(
        decision_path,
        label="Stage-B source terminal decision",
        fields=SOURCE_TERMINAL_DECISION_FIELDS,
    )
    checkpoint = decision.get("checkpoint")
    unsigned = {
        key: value for key, value in decision.items() if key != "decision_identity"
    }
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_identity)
        or sha256_file(decision_path) != expected_sha256
        or decision["decision_identity"] != expected_identity
        or decision["decision_identity"] != digest_json(unsigned)
        or decision["schema_name"] != SOURCE_TERMINAL_DECISION_SCHEMA
        or decision["schema_version"] != 1
        or decision["stage"] != "stage2"
        or decision["status"] != "valid_infeasible"
        or decision["integrity_valid"] is not True
        or decision["development_feasible"] is not False
        or decision["confirmation_manifest_opened"] is not False
        or decision["test_data_accessed"] is not False
        or decision["experiment_name"] != SOURCE_TERMINAL_DECISION_EXPERIMENT
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != SOURCE_CHECKPOINT_SHA256
        or checkpoint.get("epoch") != 5
        or checkpoint.get("global_step") != 360
        or checkpoint.get("has_feasible_checkpoint") is not False
    ):
        raise PrerequisiteError(
            "Stage-B source terminal decision is not the exact valid-infeasible "
            "legacy decision"
        )
    return decision


def validate_retry2_recovery_evidence(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
) -> dict:
    """Reopen the immutable run-r1 incident evidence before retry-2 work."""

    policy = validate_policy(policy_path)
    recovery = policy.get("recovery_contract")
    if not isinstance(recovery, dict):
        raise PrerequisiteError("Stage-C retry-2 policy lacks recovery evidence")
    manifest_binding = recovery.get("evidence_manifest")
    if not isinstance(manifest_binding, dict) or set(manifest_binding) != {
        "path",
        "sha256",
    }:
        raise PrerequisiteError("Stage-C retry-2 evidence binding is malformed")
    expected_manifest_path = (source_root / manifest_binding["path"]).resolve()
    if recovery_manifest.resolve() != expected_manifest_path:
        raise PrerequisiteError("Stage-C retry-2 evidence path changed")
    _regular_file(recovery_manifest, "Stage-C retry-2 recovery evidence manifest")
    if sha256_file(recovery_manifest) != manifest_binding["sha256"]:
        raise PrerequisiteError("Stage-C retry-2 recovery evidence hash changed")

    manifest = _exact_json(
        recovery_manifest,
        label="Stage-C retry-2 recovery evidence manifest",
        fields={
            "schema_name",
            "schema_version",
            "run_generation",
            "incident",
            "canonical_evidence_root",
            "required_files",
            "calibration_artifact",
            "required_absent_paths",
            "confirmation_spend_marker",
            "authorization",
        },
    )
    expected_incident = {
        "calibration_job_id": "143524",
        "cancelled_dependency_job_id": "143526",
        "cpu_gate_job_id": "143523",
        "failed_smoke_job_id": "143525",
        "failure_class": (
            "pre_science_source_terminal_decision_schema_compatibility"
        ),
        "prior_scientific_output_observed": False,
        "source_git_head": RUN_R1_SOURCE_GIT_HEAD,
        "source_remote_ref": RUN_R1_SOURCE_REMOTE_REF,
    }
    expected_authorization = {
        "confirmation_manifest_opened": False,
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "test_data_accessed": False,
    }
    if (
        manifest["schema_name"] != RETRY2_RECOVERY_SCHEMA
        or manifest["schema_version"] != 1
        or manifest["run_generation"] != "run_r2"
        or manifest["incident"] != expected_incident
        or manifest["authorization"] != expected_authorization
        or evidence_root.resolve()
        != Path(manifest["canonical_evidence_root"]).resolve()
    ):
        raise PrerequisiteError("Stage-C retry-2 recovery evidence scope changed")

    required_files = manifest["required_files"]
    expected_file_names = {
        "calibration_complete",
        "cpu_ready",
        "failed_smoke_stderr",
        "failed_smoke_stdout",
    }
    if not isinstance(required_files, dict) or set(required_files) != expected_file_names:
        raise PrerequisiteError("Stage-C retry-2 incident file set changed")
    reopened: dict[str, Path] = {}
    for name, spec in required_files.items():
        if (
            not isinstance(spec, dict)
            or set(spec) != {"bytes", "path", "sha256"}
            or not isinstance(spec["bytes"], int)
            or spec["bytes"] < 0
            or not isinstance(spec["path"], str)
            or Path(spec["path"]).is_absolute()
            or ".." in Path(spec["path"]).parts
            or re.fullmatch(r"[0-9a-f]{64}", str(spec["sha256"])) is None
        ):
            raise PrerequisiteError(f"Stage-C retry-2 incident file is malformed: {name}")
        path = _regular_file(
            evidence_root / spec["path"], f"Stage-C retry-2 incident file {name}"
        )
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise PrerequisiteError(
                f"Stage-C retry-2 incident file changed: {name}"
            )
        reopened[name] = path

    cpu = validate_cpu_gate(
        cpu_gate=reopened["cpu_ready"],
        source_git_head=RUN_R1_SOURCE_GIT_HEAD,
        source_remote_ref=RUN_R1_SOURCE_REMOTE_REF,
        source_remote_head=RUN_R1_SOURCE_GIT_HEAD,
    )
    completion = _exact_json(
        reopened["calibration_complete"],
        label="Stage-C run-r1 calibration completion",
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
    calibration_spec = manifest["calibration_artifact"]
    if not isinstance(calibration_spec, dict) or set(calibration_spec) != {
        "path",
        "artifact_identity",
        "calibration_sha256",
        "map_sha256",
        "ready_sha256",
    }:
        raise PrerequisiteError("Stage-C run-r1 calibration specification changed")
    calibration_dir = evidence_root / calibration_spec["path"]
    if not calibration_dir.is_dir() or calibration_dir.is_symlink():
        raise PrerequisiteError("Stage-C run-r1 calibration directory changed")
    artifact = validate_relevance_calibration_artifact(calibration_dir)
    expected_completion = {
        "source_git_head": RUN_R1_SOURCE_GIT_HEAD,
        "source_remote_ref": RUN_R1_SOURCE_REMOTE_REF,
        "source_remote_head": RUN_R1_SOURCE_GIT_HEAD,
        "artifact_path": str(calibration_dir.resolve()),
        "artifact_identity": calibration_spec["artifact_identity"],
        "calibration_sha256": calibration_spec["calibration_sha256"],
        "map_sha256": calibration_spec["map_sha256"],
        "ready_sha256": calibration_spec["ready_sha256"],
        "cpu_gate_sha256": required_files["cpu_ready"]["sha256"],
        "train_only": True,
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    if (
        completion.get("schema_name")
        != "signtrajfield_stage_c_calibration_completion"
        or completion.get("schema_version") != 1
        or any(completion.get(key) != value for key, value in expected_completion.items())
        or artifact.get("identity") != calibration_spec["artifact_identity"]
        or sha256_file(calibration_dir / "calibration.json")
        != calibration_spec["calibration_sha256"]
        or sha256_file(calibration_dir / "calibration_map.npz")
        != calibration_spec["map_sha256"]
        or sha256_file(calibration_dir / "READY")
        != calibration_spec["ready_sha256"]
        or cpu["slurm_job_id"] != "143523"
    ):
        raise PrerequisiteError("Stage-C run-r1 calibration evidence changed")

    absent_paths = manifest["required_absent_paths"]
    if (
        not isinstance(absent_paths, list)
        or len(absent_paths) != 6
        or len(set(absent_paths)) != 6
    ):
        raise PrerequisiteError("Stage-C retry-2 absent-path contract changed")
    for relative in absent_paths:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise PrerequisiteError("Stage-C retry-2 absent path is malformed")
        target = evidence_root / path
        if target.exists() or target.is_symlink():
            raise PrerequisiteError(
                f"Stage-C run-r1 scientific output unexpectedly exists: {relative}"
            )
    spend_relative = Path(manifest["confirmation_spend_marker"])
    if spend_relative.is_absolute() or ".." in spend_relative.parts:
        raise PrerequisiteError("Stage-C confirmation-spend path is malformed")
    spend = evidence_root / spend_relative
    if spend.exists() or spend.is_symlink():
        raise PrerequisiteError("Stage-C confirmation holdout is already spent")

    audit = {
        "schema_name": "signtrajfield_stage_c_retry2_recovery_audit",
        "schema_version": 1,
        "recovery_manifest_sha256": manifest_binding["sha256"],
        "run_r1_source_git_head": RUN_R1_SOURCE_GIT_HEAD,
        "failed_smoke_job_id": "143525",
        "prior_execution_lease_created": False,
        "prior_scientific_output_observed": False,
        "retry_authorization_reason": (
            "attested_pre_claim_pre_science_implementation_failure"
        ),
        "retry_authorized": True,
        "calibration_artifact_identity": artifact["identity"],
        "required_absent_paths": absent_paths,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "development_only": True,
        "non_authorizing": True,
    }
    return {**audit, "audit_identity": digest_json(audit)}


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
    recovery_policy: Path,
    recovery_manifest: Path,
    recovery_evidence_root: Path,
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
    recovery_audit = validate_retry2_recovery_evidence(
        policy_path=recovery_policy,
        recovery_manifest=recovery_manifest,
        source_root=source_root,
        evidence_root=recovery_evidence_root,
    )
    validate_source_terminal_decision(
        decision_path=source_terminal_decision,
        expected_sha256=SOURCE_TERMINAL_DECISION_SHA256,
        expected_identity=SOURCE_TERMINAL_DECISION_IDENTITY,
    )
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
        expected_source_file_profile="stage_c_generator_adaptation_retry2_v1",
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
        or claim.get("binding", {}).get("source_file_profile")
        != "stage_c_generator_adaptation_retry2_v1"
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
        "recovery_evidence_audit_identity": recovery_audit["audit_identity"],
        "recovery_evidence_manifest_sha256": recovery_audit[
            "recovery_manifest_sha256"
        ],
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
    recovery = commands.add_parser("validate-recovery")
    recovery.add_argument("--policy_path", type=Path, required=True)
    recovery.add_argument("--recovery_manifest", type=Path, required=True)
    recovery.add_argument("--source_root", type=Path, required=True)
    recovery.add_argument("--evidence_root", type=Path, required=True)
    foundation = commands.add_parser("validate-foundation")
    foundation.add_argument("--recovery_policy", type=Path, required=True)
    foundation.add_argument("--recovery_manifest", type=Path, required=True)
    foundation.add_argument("--recovery_evidence_root", type=Path, required=True)
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
    elif command == "validate-recovery":
        value = validate_retry2_recovery_evidence(**kwargs)
    elif command == "validate-foundation":
        value = validate_foundation(**kwargs)
    else:
        value = validate_smoke(**kwargs)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
