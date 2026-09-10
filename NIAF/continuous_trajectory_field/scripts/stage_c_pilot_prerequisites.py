"""Exact prerequisite-chain validation for the non-authorizing Stage-C pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from NIAF.continuous_sign_field.config import load_config

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
    _validate_claim as validate_execution_lease_claim,
)


class PrerequisiteError(RuntimeError):
    """A Stage-C prerequisite is missing, stale, mutable, or out of scope."""


def validate_unspent_confirmation_holdout(*, spend_marker: Path) -> dict[str, Any]:
    """Fail closed when the confirmation-holdout spend marker exists in any form."""

    marker = spend_marker.resolve(strict=False)
    if spend_marker.exists() or spend_marker.is_symlink():
        raise PrerequisiteError(
            f"confirmation-holdout spend marker exists: {spend_marker}"
        )
    return {
        "schema_name": "signtrajfield_stage_c_confirmation_holdout_absence",
        "schema_version": 1,
        "spend_marker_path": str(marker),
        "spend_marker_absent": True,
        "development_only": True,
        "non_authorizing": True,
    }


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
RUN_R2_SOURCE_GIT_HEAD = "f12b993b5de361423df3b8cbfb4e873f4a95ad1e"
RUN_R2_SOURCE_REMOTE_REF = (
    "origin/codex/csl-daily-centered-generator-stage-c-v1-run-r2"
)
PROTOCOL_V2_RUN_R3_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
    "decision_policy_v1.json"
)
RETRY2_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
)
PROTOCOL_V2_RUN_R3_RECOVERY_SCHEMA = (
    "signtrajfield_stage_c_protocol_v2_run_r3_recovery_evidence"
)
PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE = (
    "stage_c_generator_adaptation_protocol_v2_run_r3"
)
PROTOCOL_V3_RUN_R4_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
    "decision_policy_v1.json"
)
PROTOCOL_V3_RUN_R4_RECOVERY_SCHEMA = (
    "signtrajfield_stage_c_protocol_v3_run_r4_recovery_evidence"
)
PROTOCOL_V3_RUN_R4_SOURCE_FILE_PROFILE = (
    "stage_c_generator_adaptation_protocol_v3_run_r4"
)
PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_PATH = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_cpu.invalid_attempts/"
    "source_90cad3d3a7a09279b4a882e8c28a13b2a320ec1a_"
    "cpu143574_dependents143575_143577/ARCHIVE.json"
)
PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_SHA256 = (
    "66368adf6a2310732a4a2bcca87625d45c8dd51c0c4912b16ceba9b4c12b8dc2"
)
PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_BYTES = 11_442
PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_HEAD = (
    "90cad3d3a7a09279b4a882e8c28a13b2a320ec1a"
)
PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_REMOTE_REF = (
    "origin/codex/csl-daily-centered-generator-stage-c-v2-run-r3"
)
PROTOCOL_V3_RUN_R4_LOCAL_GIT_TIMEOUT_SECONDS = 600
PROTOCOL_V3_RUN_R4_REMOTE_REF_TIMEOUT_SECONDS = 120
RETRY2_SOURCE_FILE_PROFILE = "stage_c_generator_adaptation_retry2_v1"
PROTOCOL_V2_RUN_R3_RECOVERY_SHA256 = (
    "f955dbc7ce03f5f6a400d8ae2fa21c7ca5a280fec7f7a37da674d35a027e9209"
)
RUN_R2_POLICY_SHA256 = (
    "8eacff17c161127573ad784a77542a089ebc7eca468b7f6bc4ac9263c6a66edc"
)
RUN_R2_RECOVERY_SHA256 = (
    "ccf47b72390c4be28775b207c67f7372d3cbc599fcddb3545df8295acc88b39b"
)
RUN_R2_INCIDENT_ARCHIVE_SHA256 = (
    "82ce35cf3d7337f218bda189080a45b9e3faedec200750310d0958296f9d1855"
)
RUN_R2_INCIDENT_ARCHIVE_BYTES = 30_003
TREE_DIGEST_ALGORITHM = (
    "sha256 of sorted NUL-safe sha256sum inventory with paths relative to root "
    "prefixed by ./"
)
SOURCE_CHECKPOINT_PATH = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "absolute_binding_motion_contrast_v1/checkpoints/best_infeasible.pt"
)
SOURCE_DECISION_PATH = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "absolute_binding_motion_contrast_v1/evaluation/"
    "ordered_development_decision/decision.json"
)
FROZEN_V2_TEACHER_PATH = (
    "experiments/NIAF/continuous_trajectory_field/"
    "csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt"
)
FROZEN_V2_TEACHER_SHA256 = (
    "06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
)
SOURCE_STAGE_B_CONFIG_PATH = (
    "NIAF/continuous_trajectory_field/configs/"
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_"
    "absolute_binding_motion_contrast_v1.yaml"
)
SOURCE_STAGE_B_CONFIG_SHA256 = (
    "7741da46d37a4b77f481663f25fa580281f6d29de6c2616baebcc30dac59b85e"
)
SOURCE_STAGE_B_ARCHITECTURE_IDENTITY = (
    "bc69fd35ac58e10bc894460c35175f13356b79416df40e8c57156b1929614236"
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


def _safe_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PrerequisiteError(f"{label} path is malformed")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PrerequisiteError(f"{label} path is malformed")
    return path


def _reopen_bound_file(
    *,
    root: Path,
    binding: Any,
    label: str,
    allow_absolute: bool = False,
    expected_bytes: int | None = None,
) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise PrerequisiteError(f"{label} binding is malformed")
    raw_path = binding["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise PrerequisiteError(f"{label} path is malformed")
    relative = Path(raw_path)
    if relative.is_absolute():
        if not allow_absolute:
            raise PrerequisiteError(f"{label} path must be relative")
        path = relative
    else:
        if ".." in relative.parts:
            raise PrerequisiteError(f"{label} path is malformed")
        path = root / relative
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
            raise PrerequisiteError(f"{label} path escapes its evidence root")
    path = _regular_file(path, label)
    expected_sha256 = binding["sha256"]
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is None
        or sha256_file(path) != expected_sha256
        or (expected_bytes is not None and path.stat().st_size != expected_bytes)
    ):
        raise PrerequisiteError(f"{label} hash/size changed")
    return path


def _reopen_file_specs(
    *, root: Path, specs: Any, label: str, allow_absolute: bool = False
) -> dict[str, Path]:
    if not isinstance(specs, Mapping):
        raise PrerequisiteError(f"{label} inventory is malformed")
    reopened: dict[str, Path] = {}
    for name, spec in specs.items():
        if (
            not isinstance(name, str)
            or not isinstance(spec, Mapping)
            or set(spec) != {"bytes", "sha256"}
            or not isinstance(spec["bytes"], int)
            or spec["bytes"] < 0
        ):
            raise PrerequisiteError(f"{label} inventory entry is malformed: {name}")
        relative = Path(name)
        if relative.is_absolute():
            if not allow_absolute:
                raise PrerequisiteError(f"{label} inventory path is absolute: {name}")
            path = relative
        else:
            if ".." in relative.parts:
                raise PrerequisiteError(f"{label} inventory path is malformed: {name}")
            path = root / relative
            resolved_root = root.resolve()
            resolved_path = path.resolve()
            if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
                raise PrerequisiteError(f"{label} inventory path escapes its root: {name}")
        path = _regular_file(path, f"{label} file {name}")
        if (
            path.stat().st_size != spec["bytes"]
            or re.fullmatch(r"[0-9a-f]{64}", str(spec["sha256"])) is None
            or sha256_file(path) != spec["sha256"]
        ):
            raise PrerequisiteError(f"{label} file changed: {name}")
        reopened[name] = path
    return reopened


def _tree_digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(files, key=lambda value: value.encode("utf-8")):
        digest.update(f"{files[relative]['sha256']}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _reopen_tree_inventory(
    *, evidence_root: Path, inventory: Any, label: str, expected_fields: set[str]
) -> Path:
    if not isinstance(inventory, Mapping) or set(inventory) != expected_fields:
        raise PrerequisiteError(f"{label} tree inventory fields changed")
    relative_root = _safe_relative_path(inventory["root"], f"{label} root")
    root = evidence_root / relative_root
    resolved_evidence_root = evidence_root.resolve()
    resolved_root = root.resolve()
    if (
        resolved_root != resolved_evidence_root
        and resolved_evidence_root not in resolved_root.parents
    ):
        raise PrerequisiteError(f"{label} root escapes the canonical evidence root")
    if not root.is_dir() or root.is_symlink():
        raise PrerequisiteError(f"{label} root is not a regular directory")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise PrerequisiteError(f"{label} tree contains a symlink")
    if any(not path.is_dir() and not path.is_file() for path in entries):
        raise PrerequisiteError(f"{label} tree contains a special filesystem node")
    files = inventory["files"]
    reopened = _reopen_file_specs(root=root, specs=files, label=label)
    observed_files = {
        path.relative_to(root).as_posix() for path in entries if path.is_file()
    }
    if observed_files != set(files):
        raise PrerequisiteError(f"{label} tree file set changed")
    if (
        inventory["tree_digest_algorithm"] != TREE_DIGEST_ALGORITHM
        or inventory["file_count"] != len(files)
        or inventory["file_count"] != len(reopened)
        or inventory["total_bytes"]
        != sum(int(spec["bytes"]) for spec in files.values())
        or inventory["tree_digest"] != _tree_digest(files)
    ):
        raise PrerequisiteError(f"{label} tree inventory totals changed")
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_standalone_source_clone(
    *,
    clone: Path,
    expected_head: str,
    expected_remote_ref: str,
    local_metadata_timeout_seconds: int = 30,
    remote_ref_timeout_seconds: int = 30,
) -> None:
    """Authenticate one archived standalone clone with explicitly bounded I/O.

    Older generations retain their historical 30-second contract.  The
    protocol-v3/run-r4 recovery passes its separately preregistered 600-second
    local-metadata and 120-second remote-ref bounds exactly once in its CPU
    gate; runtime paths never call this helper for that historical clone.
    """
    if not clone.is_dir() or clone.is_symlink() or not (clone / ".git").is_dir():
        raise PrerequisiteError("Stage-C archived source clone is not standalone")
    if (
        not isinstance(local_metadata_timeout_seconds, int)
        or not isinstance(remote_ref_timeout_seconds, int)
        or local_metadata_timeout_seconds <= 0
        or remote_ref_timeout_seconds <= 0
    ):
        raise PrerequisiteError("Stage-C archived source clone timeout is malformed")

    def git(*arguments: str, timeout_seconds: int) -> str:
        try:
            result = subprocess.run(
                (
                    "git",
                    "-c",
                    f"safe.directory={clone}",
                    "-C",
                    str(clone),
                    *arguments,
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PrerequisiteError(
                "cannot revalidate Stage-C archived source clone"
            ) from error
        if result.returncode:
            raise PrerequisiteError(
                "cannot revalidate Stage-C archived source clone: "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    top_level = Path(
        git("rev-parse", "--show-toplevel", timeout_seconds=local_metadata_timeout_seconds)
    ).resolve()
    observed_head = git(
        "rev-parse", "HEAD", timeout_seconds=local_metadata_timeout_seconds
    ).lower()
    status = git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        timeout_seconds=local_metadata_timeout_seconds,
    )
    if (
        top_level != clone.resolve()
        or observed_head != expected_head
        or status
        or not expected_remote_ref.startswith("origin/")
    ):
        raise PrerequisiteError("Stage-C archived source clone identity changed")
    branch = expected_remote_ref.removeprefix("origin/")
    remote_row = git(
        "ls-remote",
        "--heads",
        "origin",
        f"refs/heads/{branch}",
        timeout_seconds=remote_ref_timeout_seconds,
    )
    if remote_row.split() != [expected_head, f"refs/heads/{branch}"]:
        raise PrerequisiteError("Stage-C archived remote ref/head changed")


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


def _validate_protocol_v2_archive_semantics(
    *, archive: Mapping[str, Any], evidence_root: Path
) -> dict[str, Any]:
    expected_fields = {
        "schema_name",
        "schema_version",
        "created_at",
        "reason",
        "immutable_no_replace_contract",
        "protocol_generation",
        "source_science",
        "jobs",
        "observed_execution_boundary",
        "failure",
        "retained_evidence",
        "required_absent_paths_revalidated_at",
        "required_absent_paths",
        "authorization",
        "recovery",
    }
    immutable_contract = {
        "logical_immutability": True,
        "existing_destination_policy": "reject_no_replace",
        "evidence_moved": False,
        "evidence_deleted": False,
        "evidence_overwritten": False,
        "archive_is_a_manifest_only": True,
        "retained_evidence_must_remain_at_original_paths": True,
    }
    protocol = archive.get("protocol_generation")
    source_science = archive.get("source_science")
    jobs = archive.get("jobs")
    boundary = archive.get("observed_execution_boundary")
    failure = archive.get("failure")
    authorization = archive.get("authorization")
    recovery = archive.get("recovery")
    if (
        set(archive) != expected_fields
        or archive.get("schema_name")
        != "signtrajfield_stage_c_generator_adaptation_terminal_incident_archive"
        or archive.get("schema_version") != 2
        or archive.get("reason")
        != "post_update_mid_validation_centered_evaluator_dispatch_failure"
        or archive.get("immutable_no_replace_contract") != immutable_contract
        or not isinstance(protocol, Mapping)
        or not isinstance(source_science, Mapping)
        or not isinstance(jobs, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(failure, Mapping)
        or not isinstance(authorization, Mapping)
        or not isinstance(recovery, Mapping)
    ):
        raise PrerequisiteError("Stage-C run-r2 incident archive scope changed")

    expected_protocol = {
        "protocol": RETRY2_SOURCE_FILE_PROFILE,
        "run": "run-r2",
        "source_git_head": RUN_R2_SOURCE_GIT_HEAD,
        "source_remote_head": RUN_R2_SOURCE_GIT_HEAD,
        "source_remote_ref": RUN_R2_SOURCE_REMOTE_REF,
        "source_clone": (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2"
        ),
        "source_clone_clean_when_independently_rechecked": True,
        "runbook": {
            "path": (
                "docs/NIAF/continuous_trajectory_field/"
                "stage_c_generator_adaptation_retry2_v1.md"
            ),
            "sha256": (
                "0ab73312aa0c8d5e684d70a0246b3ac993e63f2b1cbf55a8ab9bfcbd250e8b50"
            ),
        },
        "predecessor_runbook": {
            "path": (
                "docs/NIAF/continuous_trajectory_field/"
                "stage_c_generator_adaptation_v1.md"
            ),
            "sha256": (
                "d73af22a40e60791f857d10837ef3b482ed23f2925ab98928c4f654dd779ec7a"
            ),
        },
        "decision_policy": {
            "path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
            ),
            "sha256": RUN_R2_POLICY_SHA256,
        },
        "recovery_evidence": {
            "path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_retry2_"
                "recovery_evidence_v1.json"
            ),
            "sha256": RUN_R2_RECOVERY_SHA256,
        },
    }
    if protocol != expected_protocol:
        raise PrerequisiteError("Stage-C run-r2 protocol generation changed")
    source_clone = Path(protocol["source_clone"])
    _validate_standalone_source_clone(
        clone=source_clone,
        expected_head=RUN_R2_SOURCE_GIT_HEAD,
        expected_remote_ref=RUN_R2_SOURCE_REMOTE_REF,
    )
    for name in ("runbook", "predecessor_runbook", "decision_policy", "recovery_evidence"):
        _reopen_bound_file(
            root=source_clone,
            binding=protocol[name],
            label=f"Stage-C run-r2 {name.replace('_', ' ')}",
        )

    expected_stage_b = {
        "path": SOURCE_CHECKPOINT_PATH,
        "sha256": SOURCE_CHECKPOINT_SHA256,
        "selection_status": "best_infeasible",
        "epoch": 5,
        "global_step": 360,
    }
    expected_decision = {
        "path": SOURCE_DECISION_PATH,
        "sha256": SOURCE_TERMINAL_DECISION_SHA256,
        "decision_identity": SOURCE_TERMINAL_DECISION_IDENTITY,
        "stage": "stage2",
        "status": "valid_infeasible",
        "integrity_valid": True,
        "development_feasible": False,
        "top_level_authorized_purpose_present": False,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    expected_teacher = {
        "path": FROZEN_V2_TEACHER_PATH,
        "sha256": FROZEN_V2_TEACHER_SHA256,
    }
    if source_science != {
        "stage_b_checkpoint": expected_stage_b,
        "stage_b_terminal_decision": expected_decision,
        "frozen_v2_teacher": expected_teacher,
    }:
        raise PrerequisiteError("Stage-C run-r2 source science changed")
    _reopen_bound_file(
        root=evidence_root,
        binding={key: expected_stage_b[key] for key in ("path", "sha256")},
        label="Stage-B source checkpoint",
    )
    decision_path = _reopen_bound_file(
        root=evidence_root,
        binding={key: expected_decision[key] for key in ("path", "sha256")},
        label="Stage-B source terminal decision",
    )
    validate_source_terminal_decision(
        decision_path=decision_path,
        expected_sha256=SOURCE_TERMINAL_DECISION_SHA256,
        expected_identity=SOURCE_TERMINAL_DECISION_IDENTITY,
    )
    _reopen_bound_file(
        root=evidence_root,
        binding=expected_teacher,
        label="frozen v2 teacher checkpoint",
    )

    if (
        set(jobs) != {"provenance", "cpu_gate", "calibration", "smoke", "pilot", "current_scheduler_availability"}
        or jobs.get("cpu_gate", {}).get("job_id") != "143539"
        or jobs.get("cpu_gate", {}).get("state") != "COMPLETED"
        or jobs.get("calibration", {}).get("job_id") != "143540"
        or jobs.get("calibration", {}).get("state") != "COMPLETED"
        or jobs.get("smoke", {}).get("job_id") != "143541"
        or jobs.get("smoke", {}).get("state") != "FAILED"
        or jobs.get("smoke", {}).get("exit_code") != "143:0"
        or jobs.get("pilot", {}).get("job_id") != "143542"
        or jobs.get("pilot", {}).get("post_cancel_state") != "CANCELLED"
        or jobs.get("pilot", {}).get("post_cancel_runtime") != "00:00:00"
        or jobs.get("pilot", {}).get("post_cancel_node_list") is not None
        or jobs.get("pilot", {}).get("post_cancel_alloc_tres") is not None
    ):
        raise PrerequisiteError("Stage-C run-r2 incident jobs changed")

    expected_boundary_values = {
        "requested_arm_order": ["memory", "matched_off"],
        "arms_started": ["memory"],
        "arms_not_started": ["matched_off"],
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
        "memory_arm_logical_batches_completed": 2,
        "memory_arm_optimizer_updates_completed": 1,
        "memory_arm_checkpoint_epoch": 1,
        "memory_arm_checkpoint_global_step": 1,
        "memory_arm_checkpoint_validation_pending": True,
        "validation_computation_reached": [
            "off_first_batch",
            "on_first_batch",
            "motion_shuffled_n0_first_batch",
        ],
        "validation_values_published": False,
        "metrics_jsonl_published": False,
        "selection_summary_published": False,
        "completed_checkpoint_published": False,
        "smoke_checkpoint_audit_published": False,
        "smoke_decision_published": False,
        "smoke_ready_published": False,
        "pilot_started": False,
    }
    if any(boundary.get(key) != value for key, value in expected_boundary_values.items()):
        raise PrerequisiteError("Stage-C run-r2 scientific boundary changed")
    if (
        boundary.get("declared_trainability", {}).get(
            "post_update_changed_and_frozen_bitwise_audit_completed"
        )
        is not False
        or boundary.get("data_scope", {}).get("test_neighbor_table_staged") is not False
        or boundary.get("data_scope", {}).get("confirmation_manifest_opened") is not False
        or boundary.get("data_scope", {}).get("test_data_accessed") is not False
        or boundary.get("network", {}).get(
            "post_training_counter_delta_audit_completed"
        )
        is not False
    ):
        raise PrerequisiteError("Stage-C run-r2 partial-science scope changed")

    expected_source_trainer = {
        "path": (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2/"
            "NIAF/continuous_trajectory_field/scripts/"
            "train_continuous_trajectory_field.py"
        ),
        "sha256": (
            "8a284a491c5bd65894f8cba44ad83e3d699a3ee301212b8be4974632502ff254"
        ),
    }
    if (
        set(failure) != {
            "primary_exception",
            "primary_rank",
            "same_key_error_observed_on_rank_1",
            "rank_1_tcp_store_error_is_consequential",
            "source_trainer",
            "root_cause",
            "operational_fix_boundary",
        }
        or failure.get("primary_exception") != "KeyError: 'motion_shuffled_n0'"
        or failure.get("primary_rank") != 0
        or failure.get("same_key_error_observed_on_rank_1") is not True
        or failure.get("rank_1_tcp_store_error_is_consequential") is not True
        or failure.get("source_trainer") != expected_source_trainer
        or failure.get("operational_fix_boundary")
        != (
            "Dispatch to the paired centered evaluator when paired-corruption "
            "training is enabled OR centered sentence-memory evaluation is enabled. "
            "Do not enable the paired training objective, change the evaluation suite, "
            "extend only the legacy namespace map, or change scientific hyperparameters "
            "or thresholds."
        )
    ):
        raise PrerequisiteError("Stage-C run-r2 failure classification changed")
    _reopen_bound_file(
        root=source_clone,
        binding=expected_source_trainer,
        label="Stage-C run-r2 source trainer",
        allow_absolute=True,
    )

    if authorization != {
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "checkpoint_promotion_authorized": False,
        "confirmation_authorized": False,
        "test_authorized": False,
        "longer_run_authorized": False,
        "additional_retry_under_protocol_v1_authorized": False,
        "r2_last_checkpoint_resume_authorized": False,
        "r2_calibration_reuse_authorized": False,
        "r2_smoke_or_pilot_output_reuse_authorized": False,
        "confirmation_holdout_spent": False,
    }:
        raise PrerequisiteError("Stage-C run-r2 incident authorization changed")

    expected_recovery = {
        "run_r2_disposition": "terminal_stop_partial_scientific_output",
        "current_retry2_policy_allows_rerun": False,
        "new_protocol_generation_required": True,
        "new_protocol_must_be_preregistered_without_pilot_outcome_access": True,
        "new_protocol_basis": (
            "operational evaluator-dispatch correction only; no pilot decision or "
            "validation value was published and no scientific hyperparameter, arm, "
            "threshold, data split, or duration may be adapted from run-r2"
        ),
        "required_archive_path": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "smoke.invalid_attempts/"
            "source_f12b993b5de361423df3b8cbfb4e873f4a95ad1e_"
            "smoke143541_pilot143542/ARCHIVE.json"
        ),
        "new_runbook": (
            "docs/NIAF/continuous_trajectory_field/"
            "stage_c_generator_adaptation_protocol_v2_run_r3.md"
        ),
        "new_branch": "codex/csl-daily-centered-generator-stage-c-v2-run-r3",
        "new_source_clone": (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_v2_run_source_r3"
        ),
        "new_decision_policy": (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "decision_policy_v1.json"
        ),
        "new_recovery_evidence": (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "recovery_evidence_v1.json"
        ),
        "new_calibration_root": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_sentence_memory_relevance_calibration_stage_c_generator_"
            "adaptation_protocol_v2_run_r3"
        ),
        "new_prerequisite_root": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_prerequisites"
        ),
        "new_smoke_root": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "protocol_v2_run_r3_smoke"
        ),
        "new_pilot_root": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "protocol_v2_run_r3_pilot"
        ),
        "new_logs_prefix": "logs/sbatch/csl_stage_c_protocol_v2_run_r3_",
        "mandatory_order": [
            "fresh_complete_cpu_gate",
            "fresh_source_bound_train_only_calibration",
            "fresh_two_node_forced_ib_network_and_one_update_smoke",
            "one_epoch_pilot_only_if_smoke_ready",
        ],
        "new_source_head_and_all_file_hashes_must_be_bound_after_immutable_commit": True,
        "new_smoke_and_pilot_must_use_fresh_no_replace_namespaces": True,
        "retained_run_r2_evidence_must_not_be_deleted_moved_or_modified": True,
    }
    if recovery != expected_recovery:
        raise PrerequisiteError("Stage-C protocol-v2 recovery contract changed")
    return {
        "source_clone": source_clone,
        "fresh_namespace_roots": {
            name: evidence_root / _safe_relative_path(recovery[name], name)
            for name in (
                "new_calibration_root",
                "new_prerequisite_root",
                "new_smoke_root",
                "new_pilot_root",
            )
        },
    }


def validate_protocol_v2_run_r3_recovery_evidence(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
) -> dict:
    """Reopen the terminal run-r2 incident before protocol-v2/run-r3 work."""

    policy = validate_policy(policy_path)
    if policy_path.name != PROTOCOL_V2_RUN_R3_POLICY_NAME:
        raise PrerequisiteError("Stage-C protocol-v2 recovery policy path changed")
    recovery_contract = policy.get("recovery_contract")
    if not isinstance(recovery_contract, Mapping):
        raise PrerequisiteError("Stage-C protocol-v2 policy lacks recovery evidence")
    manifest_binding = recovery_contract.get("evidence_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise PrerequisiteError("Stage-C protocol-v2 manifest binding is malformed")
    expected_manifest = (source_root / str(manifest_binding.get("path", ""))).resolve()
    if recovery_manifest.resolve() != expected_manifest:
        raise PrerequisiteError("Stage-C protocol-v2 recovery evidence path changed")
    _reopen_bound_file(
        root=source_root,
        binding=manifest_binding,
        label="Stage-C protocol-v2 recovery evidence manifest",
    )
    manifest = _exact_json(
        recovery_manifest,
        label="Stage-C protocol-v2 recovery evidence manifest",
        fields={
            "schema_name",
            "schema_version",
            "authorization",
            "canonical_evidence_root",
            "incident_archive",
            "new_protocol",
            "predecessor_chain",
            "run_r2_scientific_boundary",
            "source_checkpoint",
        },
    )
    expected_authorization = {
        "confirmation_manifest_opened": False,
        "development_only": True,
        "longer_run_authorized": False,
        "non_authorizing": True,
        "promotion_eligible": False,
        "test_data_accessed": False,
    }
    expected_new_protocol = {
        "allowed_operational_change": (
            "centered_evaluator_dispatch_when_centered_evaluation_enabled"
        ),
        "calibration_source_file_profile": PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE,
        "protocol_generation": "protocol_v2",
        "run_generation": "run_r3",
        "same_protocol_resume_authorized": False,
        "same_protocol_retry_authorized": False,
        "scientific_settings_must_equal_run_r2": True,
        "source_branch": "codex/csl-daily-centered-generator-stage-c-v2-run-r3",
        "source_clone": (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_v2_run_source_r3"
        ),
    }
    expected_predecessors = {
        "run_r1": {
            "decision_policy_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_decision_policy_v1.json"
            ),
            "decision_policy_sha256": (
                "728813f4a504b9673eac8f35fc77ed4b19e3e862467bf26cdff8427c9ef0a896"
            ),
            "run_generation": "run_r1",
            "source_git_head": RUN_R1_SOURCE_GIT_HEAD,
            "source_remote_ref": RUN_R1_SOURCE_REMOTE_REF,
        },
        "run_r2": {
            "decision_policy_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
            ),
            "decision_policy_sha256": RUN_R2_POLICY_SHA256,
            "recovery_evidence_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_retry2_"
                "recovery_evidence_v1.json"
            ),
            "recovery_evidence_sha256": RUN_R2_RECOVERY_SHA256,
            "run_generation": "run_r2",
            "source_git_head": RUN_R2_SOURCE_GIT_HEAD,
            "source_remote_ref": RUN_R2_SOURCE_REMOTE_REF,
        },
    }
    expected_incident = {
        "attempt_file_count": 27,
        "calibration_file_count": 3,
        "job_log_file_count": 6,
        "path": recovery_contract["incident_archive"]["path"],
        "prerequisite_file_count": 11,
        "required_absent_path_count": 27,
        "schema_name": (
            "signtrajfield_stage_c_generator_adaptation_terminal_incident_archive"
        ),
        "schema_version": 2,
        "sha256": RUN_R2_INCIDENT_ARCHIVE_SHA256,
    }
    expected_source_checkpoint = {
        "epoch": 5,
        "global_step": 360,
        "path": SOURCE_CHECKPOINT_PATH,
        "selection_status": "best_infeasible",
        "sha256": SOURCE_CHECKPOINT_SHA256,
    }
    if (
        manifest["schema_name"] != PROTOCOL_V2_RUN_R3_RECOVERY_SCHEMA
        or manifest["schema_version"] != 1
        or manifest["authorization"] != expected_authorization
        or evidence_root.resolve()
        != Path(str(manifest["canonical_evidence_root"])).resolve()
        or manifest["new_protocol"] != expected_new_protocol
        or manifest["predecessor_chain"] != expected_predecessors
        or manifest["incident_archive"] != expected_incident
        or manifest["source_checkpoint"] != expected_source_checkpoint
    ):
        raise PrerequisiteError("Stage-C protocol-v2 recovery manifest scope changed")

    incident_binding = recovery_contract.get("incident_archive")
    if not isinstance(incident_binding, Mapping):
        raise PrerequisiteError("Stage-C protocol-v2 incident binding is malformed")
    archive_path = _reopen_bound_file(
        root=evidence_root,
        binding=incident_binding,
        label="Stage-C run-r2 incident archive",
        expected_bytes=RUN_R2_INCIDENT_ARCHIVE_BYTES,
    )
    archive = _exact_json(
        archive_path,
        label="Stage-C run-r2 incident archive",
        fields={
            "schema_name",
            "schema_version",
            "created_at",
            "reason",
            "immutable_no_replace_contract",
            "protocol_generation",
            "source_science",
            "jobs",
            "observed_execution_boundary",
            "failure",
            "retained_evidence",
            "required_absent_paths_revalidated_at",
            "required_absent_paths",
            "authorization",
            "recovery",
        },
    )
    archive_scope = _validate_protocol_v2_archive_semantics(
        archive=archive, evidence_root=evidence_root
    )
    source_clone = archive_scope["source_clone"]

    r1 = expected_predecessors["run_r1"]
    r1_policy = _reopen_bound_file(
        root=source_clone,
        binding={
            "path": r1["decision_policy_path"],
            "sha256": r1["decision_policy_sha256"],
        },
        label="Stage-C run-r1 decision policy",
    )
    validate_policy(r1_policy)
    r2 = expected_predecessors["run_r2"]
    r2_policy = _reopen_bound_file(
        root=source_clone,
        binding={
            "path": r2["decision_policy_path"],
            "sha256": r2["decision_policy_sha256"],
        },
        label="Stage-C run-r2 decision policy",
    )
    r2_manifest = _reopen_bound_file(
        root=source_clone,
        binding={
            "path": r2["recovery_evidence_path"],
            "sha256": r2["recovery_evidence_sha256"],
        },
        label="Stage-C run-r2 recovery evidence",
    )
    retry2_audit = validate_retry2_recovery_evidence(
        policy_path=r2_policy,
        recovery_manifest=r2_manifest,
        source_root=source_clone,
        evidence_root=evidence_root,
    )

    retained = archive["retained_evidence"]
    if not isinstance(retained, Mapping) or set(retained) != {
        "attempt",
        "prerequisites",
        "calibration",
        "job_logs",
        "run_source_configs",
    }:
        raise PrerequisiteError("Stage-C run-r2 retained-evidence set changed")
    attempt_root = _reopen_tree_inventory(
        evidence_root=evidence_root,
        inventory=retained["attempt"],
        label="Stage-C run-r2 attempt",
        expected_fields={
            "root",
            "file_count",
            "total_bytes",
            "tree_digest_algorithm",
            "tree_digest",
            "files",
        },
    )
    prerequisite_root = _reopen_tree_inventory(
        evidence_root=evidence_root,
        inventory=retained["prerequisites"],
        label="Stage-C run-r2 prerequisites",
        expected_fields={
            "root",
            "file_count",
            "total_bytes",
            "tree_digest_algorithm",
            "tree_digest",
            "files",
            "smoke_lease",
        },
    )
    calibration_root = _reopen_tree_inventory(
        evidence_root=evidence_root,
        inventory=retained["calibration"],
        label="Stage-C run-r2 calibration",
        expected_fields={
            "root",
            "file_count",
            "total_bytes",
            "tree_digest_algorithm",
            "tree_digest",
            "files",
            "artifact_identity",
            "calibration_identity",
            "map_content_digest",
            "transition_identity",
            "heldout_auroc",
            "heldout_probability_gap",
            "coefficient_a",
            "coefficient_b",
            "intercept",
            "train_only",
            "confirmation_manifest_opened",
            "test_data_accessed",
        },
    )
    if (
        retained["attempt"]["file_count"] != manifest["incident_archive"]["attempt_file_count"]
        or retained["prerequisites"]["file_count"]
        != manifest["incident_archive"]["prerequisite_file_count"]
        or retained["calibration"]["file_count"]
        != manifest["incident_archive"]["calibration_file_count"]
        or retained["prerequisites"].get("smoke_lease", {}).get("terminal_outcome")
        != "failed"
        or retained["prerequisites"].get("smoke_lease", {}).get(
            "self_reported_exit_code"
        )
        != 143
        or retained["calibration"].get("train_only") is not True
        or retained["calibration"].get("confirmation_manifest_opened") is not False
        or retained["calibration"].get("test_data_accessed") is not False
    ):
        raise PrerequisiteError("Stage-C run-r2 retained evidence semantics changed")

    smoke_lease = retained["prerequisites"]["smoke_lease"]
    expected_smoke_lease = {
        "claim_identity": (
            "306531a6e845d8ba5c7f589f37dfb2f188285ff0ec5ab95fe582102ac95f2692"
        ),
        "execution_binding_digest": (
            "47e534c79aed7a8eb2691eb369ff602177475f88d721411cdcdee31a0be2eaf9"
        ),
        "terminal_identity": (
            "ac6e8f80d4665a2b05d4d4e3bd85f7d32852e231295b35e30bffd15b9fd3b551"
        ),
        "terminal_outcome": "failed",
        "self_reported_exit_code": 143,
        "active_lease_released_to_failed_history": True,
    }
    if smoke_lease != expected_smoke_lease:
        raise PrerequisiteError("Stage-C run-r2 smoke lease summary changed")
    lease_directory = (
        prerequisite_root
        / "smoke/active_execution_lease.history"
        / f"{smoke_lease['claim_identity']}.failed"
    )
    attestation_path = prerequisite_root / "smoke/lease_attestations/143541.0.json"
    terminal_path = lease_directory / "TERMINAL.json"
    try:
        claim = validate_execution_lease_claim(lease_directory)
    except StageCExecutionControlError as error:
        raise PrerequisiteError("Stage-C run-r2 smoke lease claim changed") from error
    attestation = _exact_json(
        attestation_path,
        label="Stage-C run-r2 smoke lease attestation",
        fields={
            "schema_name",
            "schema_version",
            "lease_path",
            "current_slurm_job_id",
            "current_slurm_restart_count",
            "claim",
            "restart_reconciliation",
            "attestation_identity",
        },
    )
    terminal = _exact_json(
        terminal_path,
        label="Stage-C run-r2 smoke lease terminal",
        fields={
            "schema_name",
            "schema_version",
            "outcome",
            "claim_identity",
            "current_slurm_job_id",
            "current_slurm_restart_count",
            "attestation_sha256",
            "self_reported_exit_code",
            "terminal_identity",
        },
    )
    unsigned_attestation = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_identity"
    }
    unsigned_terminal = {
        key: value for key, value in terminal.items() if key != "terminal_identity"
    }
    if (
        claim.get("claim_identity") != smoke_lease["claim_identity"]
        or claim.get("slurm_job_id") != "143541"
        or claim.get("slurm_restart_count") != 0
        or claim.get("mode") != "smoke"
        or claim.get("execution_binding", {}).get("digest")
        != smoke_lease["execution_binding_digest"]
        or claim.get("execution_binding", {}).get("source_git_head")
        != RUN_R2_SOURCE_GIT_HEAD
        or claim.get("execution_binding", {}).get("source_remote_ref")
        != RUN_R2_SOURCE_REMOTE_REF
        or attestation.get("claim") != claim
        or attestation.get("current_slurm_job_id") != "143541"
        or attestation.get("current_slurm_restart_count") != 0
        or attestation.get("restart_reconciliation") is not None
        or attestation.get("attestation_identity")
        != digest_json(unsigned_attestation)
        or terminal.get("schema_name")
        != "signtrajfield_stage_c_execution_lease_terminal"
        or terminal.get("schema_version") != 1
        or terminal.get("outcome") != smoke_lease["terminal_outcome"]
        or terminal.get("claim_identity") != smoke_lease["claim_identity"]
        or terminal.get("current_slurm_job_id") != "143541"
        or terminal.get("current_slurm_restart_count") != 0
        or terminal.get("attestation_sha256") != sha256_file(attestation_path)
        or terminal.get("self_reported_exit_code")
        != smoke_lease["self_reported_exit_code"]
        or terminal.get("terminal_identity") != smoke_lease["terminal_identity"]
        or terminal.get("terminal_identity") != digest_json(unsigned_terminal)
    ):
        raise PrerequisiteError("Stage-C run-r2 smoke lease evidence changed")
    job_logs = _reopen_file_specs(
        root=evidence_root,
        specs=retained["job_logs"],
        label="Stage-C run-r2 job log",
    )
    if len(job_logs) != manifest["incident_archive"]["job_log_file_count"]:
        raise PrerequisiteError("Stage-C run-r2 job-log count changed")

    run_source_configs = retained["run_source_configs"]
    if not isinstance(run_source_configs, Mapping) or len(run_source_configs) != 2:
        raise PrerequisiteError("Stage-C run-r2 source-config inventory changed")
    expected_config_hashes = {
        (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2/"
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_pilot_run_r2.yaml"
        ): "62a6b955594580377638ee1d7a2f47ae4e62ae1ba7258f1cbedd474cf998612d",
        (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2/"
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_pilot_run_r2.yaml"
        ): "dc3722d838c45cb4480552f1db224b9c63f1fdc04cc363219f4d01f625f3b7a4",
    }
    if run_source_configs != expected_config_hashes:
        raise PrerequisiteError("Stage-C run-r2 source-config bindings changed")
    for path_value, expected_sha256 in run_source_configs.items():
        _reopen_bound_file(
            root=source_clone,
            binding={"path": path_value, "sha256": expected_sha256},
            label="Stage-C run-r2 source config",
            allow_absolute=True,
        )

    absent_paths = archive["required_absent_paths"]
    if (
        not isinstance(absent_paths, list)
        or len(absent_paths) != manifest["incident_archive"]["required_absent_path_count"]
        or len(set(absent_paths)) != len(absent_paths)
    ):
        raise PrerequisiteError("Stage-C run-r2 absent-path inventory changed")
    old_paths = {attempt_root, prerequisite_root, calibration_root}
    for relative_value in absent_paths:
        relative = _safe_relative_path(relative_value, "Stage-C run-r2 absent evidence")
        target = evidence_root / relative
        old_paths.add(target)
        if target.exists() or target.is_symlink():
            raise PrerequisiteError(
                f"Stage-C run-r2 required-absent output now exists: {relative_value}"
            )

    fresh_roots = archive_scope["fresh_namespace_roots"]
    fresh_values = list(fresh_roots.values())
    if len(set(fresh_values)) != len(fresh_values) or any(
        _paths_overlap(fresh, old) for fresh in fresh_values for old in old_paths
    ):
        raise PrerequisiteError("Stage-C protocol-v2 namespaces are not fresh/distinct")

    boundary = archive["observed_execution_boundary"]
    manifest_boundary = manifest["run_r2_scientific_boundary"]
    expected_manifest_boundary = {
        "completed_checkpoint_published": boundary["completed_checkpoint_published"],
        "matched_off_arm_started": "matched_off" in boundary["arms_started"],
        "memory_arm_logical_batches_completed": boundary[
            "memory_arm_logical_batches_completed"
        ],
        "memory_arm_optimizer_updates_completed": boundary[
            "memory_arm_optimizer_updates_completed"
        ],
        "metrics_jsonl_published": boundary["metrics_jsonl_published"],
        "pilot_started": boundary["pilot_started"],
        "prior_execution_lease_created": True,
        "prior_scientific_output_observed": True,
        "smoke_ready_published": boundary["smoke_ready_published"],
        "validation_computation_reached": boundary["validation_computation_reached"],
        "validation_values_published": boundary["validation_values_published"],
    }
    if manifest_boundary != expected_manifest_boundary:
        raise PrerequisiteError("Stage-C protocol-v2 scientific boundary changed")

    audit = {
        "schema_name": "signtrajfield_stage_c_protocol_v2_run_r3_recovery_audit",
        "schema_version": 1,
        "recovery_manifest_sha256": PROTOCOL_V2_RUN_R3_RECOVERY_SHA256,
        "incident_archive_path": str(archive_path.resolve()),
        "incident_archive_sha256": RUN_R2_INCIDENT_ARCHIVE_SHA256,
        "run_r2_source_git_head": RUN_R2_SOURCE_GIT_HEAD,
        "failed_smoke_job_id": "143541",
        "prior_execution_lease_created": True,
        "prior_scientific_output_observed": True,
        "prior_memory_optimizer_updates_observed": 1,
        "matched_off_arm_started": False,
        "same_protocol_retry_authorized": False,
        "resume_authorized": False,
        "calibration_source_file_profile": PROTOCOL_V2_RUN_R3_SOURCE_FILE_PROFILE,
        "predecessor_recovery_audit_identity": retry2_audit["audit_identity"],
        "required_absent_paths": absent_paths,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "development_only": True,
        "non_authorizing": True,
    }
    return {**audit, "audit_identity": digest_json(audit)}


def _protocol_v3_run_r4_archive_absences() -> dict[str, Any]:
    """Return the immutable zero-science absence contract from the v3 manifest."""

    return {
        "arm_roots": [
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_memory_protocol_v2_run_r3"
            ),
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_matched_off_protocol_v2_run_r3"
            ),
        ],
        "prerequisite_calibration_smoke_pilot_roots": [
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                "prerequisites/source_90cad3d3a7a09279b4a882e8c28a13b2a320ec1a"
            ),
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_sentence_memory_relevance_calibration_stage_c_generator_"
                "adaptation_protocol_v2_run_r3"
            ),
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_protocol_v2_run_r3_smoke"
            ),
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_protocol_v2_run_r3_pilot"
            ),
        ],
        "six_runtime_logs": [
            "logs/sbatch/csl_stage_c_protocol_v2_run_r3_calibration_143575.out",
            "logs/sbatch/csl_stage_c_protocol_v2_run_r3_calibration_143575.err",
            "logs/sbatch/csl_stage_c_protocol_v2_run_r3_smoke_143576.out",
            "logs/sbatch/csl_stage_c_protocol_v2_run_r3_smoke_143576.err",
            "logs/sbatch/csl_stage_c_protocol_v2_run_r3_pilot_143577.out",
            "logs/sbatch/csl_stage_c_protocol_v2_run_r3_pilot_143577.err",
        ],
        "confirmation_holdout_spend_marker": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_"
            "ordered_v1_control/confirmation_holdout_spent.json"
        ),
    }


def _validate_protocol_v3_run_r4_recovery(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
    validate_historical_clone_metadata: bool,
) -> dict[str, Any]:
    """Authenticate protocol-v3/run-r4 recovery without creating any artifact.

    The archive's only allowed change class delegates the costly historical
    proof to the CPU gate. Callers after that gate reopen only the archive,
    policy, manifest, and current-source bindings by hash; they never run
    Git status or another historical-clone scan.
    """

    policy = validate_policy(policy_path)
    if policy_path.name != PROTOCOL_V3_RUN_R4_POLICY_NAME:
        raise PrerequisiteError("Stage-C protocol-v3 recovery policy path changed")
    recovery_contract = policy.get("recovery_contract")
    if not isinstance(recovery_contract, Mapping):
        raise PrerequisiteError("Stage-C protocol-v3 policy lacks recovery evidence")
    manifest_binding = recovery_contract.get("evidence_manifest")
    incident_binding = recovery_contract.get("incident_archive")
    if not isinstance(manifest_binding, Mapping) or not isinstance(
        incident_binding, Mapping
    ):
        raise PrerequisiteError("Stage-C protocol-v3 recovery binding is malformed")
    expected_manifest = (source_root / str(manifest_binding.get("path", ""))).resolve()
    if recovery_manifest.resolve() != expected_manifest:
        raise PrerequisiteError("Stage-C protocol-v3 recovery evidence path changed")
    _reopen_bound_file(
        root=source_root,
        binding=manifest_binding,
        label="Stage-C protocol-v3 recovery evidence manifest",
    )
    manifest = _exact_json(
        recovery_manifest,
        label="Stage-C protocol-v3 recovery evidence manifest",
        fields={
            "schema_name",
            "schema_version",
            "authorization",
            "canonical_evidence_root",
            "incident_archive",
            "new_protocol",
            "predecessor_chain",
            "source_checkpoint",
            "zero_science_boundary",
            "zero_science_runtime_absences",
        },
    )
    expected_authorization = {
        "confirmation_manifest_opened": False,
        "development_only": True,
        "longer_run_authorized": False,
        "non_authorizing": True,
        "promotion_eligible": False,
        "test_data_accessed": False,
    }
    expected_new_protocol = {
        "allowed_operational_change": (
            "increase_only_the_bounded_archived_clone_local_git_metadata_timeout"
        ),
        "calibration_source_file_profile": PROTOCOL_V3_RUN_R4_SOURCE_FILE_PROFILE,
        "downstream_proof_delegation": (
            "validation_preserving_execution_safety: CPU READY authenticates the "
            "one-time historical proof; downstream reopens only bound hashes and "
            "current source/config identities."
        ),
        "historical_archived_clone_local_git_metadata_timeout_seconds": (
            PROTOCOL_V3_RUN_R4_LOCAL_GIT_TIMEOUT_SECONDS
        ),
        "historical_remote_ref_timeout_seconds": (
            PROTOCOL_V3_RUN_R4_REMOTE_REF_TIMEOUT_SECONDS
        ),
        "protocol_generation": "protocol_v3",
        "run_generation": "run_r4",
        "same_protocol_resume_authorized": False,
        "same_protocol_retry_authorized": False,
        "scientific_settings_must_equal_protocol_v2_run_r3": True,
        "source_branch": "codex/csl-daily-centered-generator-stage-c-v3-run-r4",
        "source_clone": (
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_v3_run_source_r4"
        ),
    }
    expected_incident = {
        "path": PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_PATH,
        "sha256": PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_SHA256,
        "bytes": PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_BYTES,
        "schema_name": (
            "signtrajfield_stage_c_generator_adaptation_terminal_incident_archive"
        ),
        "schema_version": 3,
        "required_absent_path_count": 13,
    }
    expected_boundary = {
        "calibration_started": False,
        "compileall_started": False,
        "cpu_gate_ready_published": False,
        "data_or_neighbor_table_opened": False,
        "execution_lease_created": False,
        "gpu_allocated": False,
        "optimizer_updates": 0,
        "pytest_started": False,
        "ruff_started": False,
        "scientific_output_observed": False,
        "trainer_started": False,
        "validation_batches": 0,
    }
    if (
        manifest["schema_name"] != PROTOCOL_V3_RUN_R4_RECOVERY_SCHEMA
        or manifest["schema_version"] != 1
        or manifest["authorization"] != expected_authorization
        or evidence_root.resolve()
        != Path(str(manifest["canonical_evidence_root"])).resolve()
        or manifest["new_protocol"] != expected_new_protocol
        or manifest["incident_archive"] != expected_incident
        or manifest["source_checkpoint"]
        != {
            "epoch": 5,
            "global_step": 360,
            "path": SOURCE_CHECKPOINT_PATH,
            "selection_status": "best_infeasible",
            "sha256": SOURCE_CHECKPOINT_SHA256,
        }
        or manifest["zero_science_boundary"] != expected_boundary
        or manifest["zero_science_runtime_absences"]
        != _protocol_v3_run_r4_archive_absences()
    ):
        raise PrerequisiteError("Stage-C protocol-v3 recovery manifest scope changed")
    if incident_binding != {
        "path": PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_PATH,
        "sha256": PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_SHA256,
    }:
        raise PrerequisiteError("Stage-C protocol-v3 incident binding changed")
    archive_path = _reopen_bound_file(
        root=evidence_root,
        binding=incident_binding,
        label="Stage-C protocol-v2/run-r3 zero-science archive",
        expected_bytes=PROTOCOL_V3_RUN_R4_INCIDENT_ARCHIVE_BYTES,
    )
    archive = _exact_json(
        archive_path,
        label="Stage-C protocol-v2/run-r3 zero-science archive",
        fields={
            "authorization",
            "created_at",
            "failure",
            "immutable_no_replace_contract",
            "jobs",
            "observed_execution_boundary",
            "protocol_generation",
            "reason",
            "recovery",
            "required_absent_paths",
            "required_absent_paths_revalidated_at",
            "retained_evidence",
            "schema_name",
            "schema_version",
            "source_science",
        },
    )
    required_absences = _protocol_v3_run_r4_archive_absences()
    flattened_absences = (
        required_absences["prerequisite_calibration_smoke_pilot_roots"]
        + required_absences["arm_roots"]
        + required_absences["six_runtime_logs"]
        + [required_absences["confirmation_holdout_spend_marker"]]
    )
    expected_failure = {
        "command": [
            "git",
            "-c",
            "safe.directory=/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2",
            "-C",
            "/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        "exception": "subprocess.TimeoutExpired",
        "message": "cannot revalidate Stage-C archived source clone",
        "phase": "initial_validate_recovery_before_cpu_gate_work",
        "prerequisite_helper_line": 347,
        "root_cause": (
            "the fixed 30-second subprocess timeout was shorter than transient "
            "CIFS metadata latency on the allocated CPU node"
        ),
        "timeout_seconds": 30,
        "wrapper_line": 74,
    }
    expected_archive_protocol = {
        "protocol": "protocol_v2",
        "run": "run_r3",
        "status": "terminal_zero_science_preflight_failure",
    }
    if (
        archive.get("schema_name")
        != "signtrajfield_stage_c_generator_adaptation_terminal_incident_archive"
        or archive.get("schema_version") != 3
        or archive.get("protocol_generation") != expected_archive_protocol
        or archive.get("failure") != expected_failure
        or archive.get("observed_execution_boundary") != expected_boundary
        or archive.get("required_absent_paths") != flattened_absences
        or archive.get("recovery")
        != {
            "allowed_change_class": (
                "increase_only_the_bounded_archived_clone_local_git_metadata_timeout"
            ),
            "new_protocol_generation_required": True,
            "resume_authorized": False,
            "same_protocol_retry_authorized": False,
            "scientific_settings_may_change": False,
        }
        or archive.get("source_science")
        != {
            "checkpoint_epoch": 5,
            "checkpoint_global_step": 360,
            "checkpoint_selection_status": "best_infeasible",
            "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "scientific_configuration_was_not_executed": True,
        }
        or archive.get("authorization") != {
            "authorized_purpose": None,
            "confirmation_manifest_opened": False,
            "development_only": True,
            "non_authorizing": True,
            "promotion_eligible": False,
            "test_data_accessed": False,
        }
    ):
        raise PrerequisiteError("Stage-C protocol-v2/run-r3 zero-science archive changed")

    retained = archive.get("retained_evidence")
    if not isinstance(retained, Mapping):
        raise PrerequisiteError("Stage-C protocol-v2/run-r3 retained evidence changed")
    prelaunch = retained.get("prelaunch_validation")
    run_source = retained.get("run_source")
    source_files = retained.get("source_files")
    if (
        prelaunch
        != {
            "config_audit_identity": (
                "5c1534ffb887ff1eb05f229adda0b9b8be8a4845df9311456fb15faf86257f8d"
            ),
            "incident_archive_sha256": RUN_R2_INCIDENT_ARCHIVE_SHA256,
            "recovery_audit_identity": (
                "74e5ff1f1a6211ccb746eecd82b384c0e152efc5d98486a27548bc895574a2ea"
            ),
        }
        or not isinstance(run_source, Mapping)
        or not isinstance(source_files, Mapping)
        or run_source.get("branch")
        != "codex/csl-daily-centered-generator-stage-c-v2-run-r3"
        or run_source.get("clone")
        != "/media/cvpr/haomian/SignTrajField_centered_stage_c_v2_run_source_r3"
        or run_source.get("git_head") != PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_HEAD
        or run_source.get("remote_head") != PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_HEAD
        or run_source.get("remote_ref")
        != PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_REMOTE_REF
        or run_source.get("status_clean") is not True
    ):
        raise PrerequisiteError("Stage-C protocol-v2/run-r3 retained provenance changed")

    expected_r3_files = {
        (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3.yaml"
        ): "77e61c12a68f5f5bd60cc23355ca935fa5bf0a1ed777a9da440203417e3750cc",
        (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3.yaml"
        ): "c36f5617d20d3ddc7386f8ec8d4811f58417fe716d86a5dc249c38a71258f880",
        (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "decision_policy_v1.json"
        ): "7cc5a9443d0b7b65280b192c7ec64475103590bf2ae7da32dccc3eea667052fe",
        (
            "NIAF/continuous_trajectory_field/configs/"
            "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
            "recovery_evidence_v1.json"
        ): PROTOCOL_V2_RUN_R3_RECOVERY_SHA256,
    }
    for relative, expected_sha256 in expected_r3_files.items():
        record = source_files.get(relative)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != expected_sha256
            or not isinstance(record.get("bytes"), int)
        ):
            raise PrerequisiteError("Stage-C protocol-v2/run-r3 source-file archive changed")
        _reopen_bound_file(
            root=source_root,
            binding={"path": relative, "sha256": expected_sha256},
            label=f"Stage-C retained source file {relative}",
        )

    expected_predecessor_chain = {
        "run_r1": {
            "decision_policy_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_decision_policy_v1.json"
            ),
            "decision_policy_sha256": (
                "728813f4a504b9673eac8f35fc77ed4b19e3e862467bf26cdff8427c9ef0a896"
            ),
            "run_generation": "run_r1",
            "source_git_head": RUN_R1_SOURCE_GIT_HEAD,
            "source_remote_ref": RUN_R1_SOURCE_REMOTE_REF,
        },
        "run_r2": {
            "decision_policy_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json"
            ),
            "decision_policy_sha256": RUN_R2_POLICY_SHA256,
            "recovery_evidence_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_retry2_"
                "recovery_evidence_v1.json"
            ),
            "recovery_evidence_sha256": RUN_R2_RECOVERY_SHA256,
            "run_generation": "run_r2",
            "source_git_head": RUN_R2_SOURCE_GIT_HEAD,
            "source_remote_ref": RUN_R2_SOURCE_REMOTE_REF,
        },
        "run_r3": {
            "decision_policy_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                "decision_policy_v1.json"
            ),
            "decision_policy_sha256": expected_r3_files[
                (
                    "NIAF/continuous_trajectory_field/configs/"
                    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                    "decision_policy_v1.json"
                )
            ],
            "recovery_evidence_path": (
                "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
                "recovery_evidence_v1.json"
            ),
            "recovery_evidence_sha256": PROTOCOL_V2_RUN_R3_RECOVERY_SHA256,
            "run_generation": "run_r3",
            "source_git_head": PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_HEAD,
            "source_remote_ref": PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_REMOTE_REF,
        },
    }
    if manifest["predecessor_chain"] != expected_predecessor_chain:
        raise PrerequisiteError("Stage-C protocol-v3 predecessor chain changed")

    # The current immutable source must still carry every link in the chain.
    # These are pure hash re-opens, so runtime callers authenticate provenance
    # without touching the archived r3 clone's Git metadata again.
    for generation in ("run_r1", "run_r2", "run_r3"):
        predecessor = expected_predecessor_chain[generation]
        _reopen_bound_file(
            root=source_root,
            binding={
                "path": predecessor["decision_policy_path"],
                "sha256": predecessor["decision_policy_sha256"],
            },
            label=f"Stage-C protocol-v3 predecessor {generation} decision policy",
        )
        if generation != "run_r1":
            _reopen_bound_file(
                root=source_root,
                binding={
                    "path": predecessor["recovery_evidence_path"],
                    "sha256": predecessor["recovery_evidence_sha256"],
                },
                label=(
                    f"Stage-C protocol-v3 predecessor {generation} recovery evidence"
                ),
            )

    if validate_historical_clone_metadata:
        _validate_standalone_source_clone(
            clone=Path(str(run_source["clone"])),
            expected_head=PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_HEAD,
            expected_remote_ref=PROTOCOL_V3_RUN_R4_ARCHIVED_SOURCE_REMOTE_REF,
            local_metadata_timeout_seconds=PROTOCOL_V3_RUN_R4_LOCAL_GIT_TIMEOUT_SECONDS,
            remote_ref_timeout_seconds=PROTOCOL_V3_RUN_R4_REMOTE_REF_TIMEOUT_SECONDS,
        )
    audit = {
        "schema_name": "signtrajfield_stage_c_protocol_v3_run_r4_recovery_audit",
        "schema_version": 1,
        "recovery_manifest_sha256": sha256_file(recovery_manifest),
        "decision_policy_sha256": sha256_file(policy_path),
        "incident_archive_path": str(archive_path.resolve()),
        "incident_archive_sha256": sha256_file(archive_path),
        "zero_science_archive_bound": True,
        "six_runtime_logs_absent_before_recovery": True,
        "six_runtime_roots_absent_before_recovery": True,
        "confirmation_holdout_spend_marker_absent": True,
        "historical_clone_metadata_checked_in_cpu_gate": bool(
            validate_historical_clone_metadata
        ),
        "historical_clone_local_git_metadata_timeout_seconds": (
            PROTOCOL_V3_RUN_R4_LOCAL_GIT_TIMEOUT_SECONDS
        ),
        "historical_remote_ref_timeout_seconds": (
            PROTOCOL_V3_RUN_R4_REMOTE_REF_TIMEOUT_SECONDS
        ),
        "calibration_source_file_profile": PROTOCOL_V3_RUN_R4_SOURCE_FILE_PROFILE,
        "development_only": True,
        "non_authorizing": True,
    }
    # The identity intentionally excludes the execution-location boolean: a
    # later runtime re-open must authenticate the CPU proof without rerunning it.
    identity_payload = dict(audit)
    identity_payload.pop("historical_clone_metadata_checked_in_cpu_gate")
    audit["audit_identity"] = digest_json(identity_payload)
    return audit


def validate_protocol_v3_run_r4_recovery_evidence(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Run the one permitted historical validation before CPU-gate work."""

    return _validate_protocol_v3_run_r4_recovery(
        policy_path=policy_path,
        recovery_manifest=recovery_manifest,
        source_root=source_root,
        evidence_root=evidence_root,
        validate_historical_clone_metadata=True,
    )


def validate_protocol_v3_run_r4_runtime_bindings(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Reopen immutable recovery artifacts without historical Git operations."""

    return _validate_protocol_v3_run_r4_recovery(
        policy_path=policy_path,
        recovery_manifest=recovery_manifest,
        source_root=source_root,
        evidence_root=evidence_root,
        validate_historical_clone_metadata=False,
    )


def validate_recovery_evidence(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
) -> dict:
    """Dispatch exact recovery validation by immutable policy filename."""

    if policy_path.name == RETRY2_POLICY_NAME:
        return validate_retry2_recovery_evidence(
            policy_path=policy_path,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=evidence_root,
        )
    if policy_path.name == PROTOCOL_V2_RUN_R3_POLICY_NAME:
        return validate_protocol_v2_run_r3_recovery_evidence(
            policy_path=policy_path,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=evidence_root,
        )
    if policy_path.name == PROTOCOL_V3_RUN_R4_POLICY_NAME:
        return validate_protocol_v3_run_r4_recovery_evidence(
            policy_path=policy_path,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=evidence_root,
        )
    raise PrerequisiteError("Stage-C recovery policy filename is not registered")


_CROSS_GENERATION_CONFIG_FIELDS = (
    ("experiment_name",),
    ("output", "out_dir"),
    ("sentence_memory", "relevance_calibration", "artifact_dir"),
    (
        "sentence_memory_safety",
        "stage_c",
        "active_stage_c",
        "calibration_artifact_dir",
    ),
)
_ARM_CONFIG_FIELDS = _CROSS_GENERATION_CONFIG_FIELDS + (
    ("conditioning", "sentence_memory_train_mode"),
    ("conditioning", "sentence_memory_dropout_probability"),
    ("sentence_memory_safety", "stage_c", "arm"),
)


def _normalize_config_fields(
    value: Mapping[str, Any], fields: tuple[tuple[str, ...], ...], label: str
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    for field_path in fields:
        parent: Any = normalized
        for name in field_path[:-1]:
            if not isinstance(parent, dict) or name not in parent:
                raise PrerequisiteError(
                    f"{label} lacks normalization field {'.'.join(field_path)}"
                )
            parent = parent[name]
        leaf = field_path[-1]
        if not isinstance(parent, dict) or leaf not in parent:
            raise PrerequisiteError(
                f"{label} lacks normalization field {'.'.join(field_path)}"
            )
        parent.pop(leaf)
    return normalized


def _validate_stage_c_arm_config(
    *, config: Mapping[str, Any], arm: str, generation: str
) -> None:
    expected_mode = "dropout" if arm == "memory" else "off"
    expected_dropout = 0.25 if arm == "memory" else 1.0
    stage = config.get("sentence_memory_safety", {}).get("stage_c", {})
    expected_checkpoint = {
        "path": SOURCE_CHECKPOINT_PATH,
        "sha256": SOURCE_CHECKPOINT_SHA256,
        "selection_status": "best_infeasible",
        "epoch": 5,
        "global_step": 360,
    }
    expected_source_decision = {
        "path": SOURCE_DECISION_PATH,
        "sha256": SOURCE_TERMINAL_DECISION_SHA256,
        "decision_identity": SOURCE_TERMINAL_DECISION_IDENTITY,
        "status": "valid_infeasible",
        "authorized_purpose": None,
    }
    expected_source_stage_b = {
        "config_path": SOURCE_STAGE_B_CONFIG_PATH,
        "config_sha256": SOURCE_STAGE_B_CONFIG_SHA256,
        "architecture_identity": SOURCE_STAGE_B_ARCHITECTURE_IDENTITY,
    }
    expected_teacher = {
        "path": FROZEN_V2_TEACHER_PATH,
        "sha256": FROZEN_V2_TEACHER_SHA256,
    }
    conditioning = config.get("conditioning", {})
    train = config.get("train", {})
    if (
        not isinstance(stage, Mapping)
        or stage.get("arm") != arm
        or stage.get("source_checkpoint") != expected_checkpoint
        or stage.get("source_terminal_decision") != expected_source_decision
        or stage.get("source_stage_b") != expected_source_stage_b
        or stage.get("frozen_v2_teacher") != expected_teacher
        or conditioning.get("sentence_memory_train_mode") != expected_mode
        or conditioning.get("sentence_memory_dropout_probability") != expected_dropout
        or train.get("base_checkpoint") is not None
    ):
        raise PrerequisiteError(
            f"Stage-C {generation} {arm} config changed its arm or Stage-B warm start"
        )


def validate_protocol_v2_run_r3_configs(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
    memory_config: Path,
    matched_off_config: Path,
) -> dict[str, Any]:
    """Prove run-r3 settings equal immutable run-r2 outside fresh identities."""

    recovery_audit = validate_protocol_v2_run_r3_recovery_evidence(
        policy_path=policy_path,
        recovery_manifest=recovery_manifest,
        source_root=source_root,
        evidence_root=evidence_root,
    )
    expected_names = {
        "memory": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "memory_protocol_v2_run_r3.yaml"
        ),
        "matched_off": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_"
            "matched_off_protocol_v2_run_r3.yaml"
        ),
    }
    for arm, path in (("memory", memory_config), ("matched_off", matched_off_config)):
        _regular_file(path, f"Stage-C protocol-v2 {arm} config")
        expected_path = (
            source_root
            / "NIAF/continuous_trajectory_field/configs"
            / expected_names[arm]
        ).resolve()
        if path.resolve() != expected_path:
            raise PrerequisiteError(f"Stage-C protocol-v2 {arm} config path changed")

    archive_path = Path(recovery_audit["incident_archive_path"])
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    run_r2_bindings = archive["retained_evidence"]["run_source_configs"]
    run_r2_paths: dict[str, Path] = {}
    for raw_path, expected_sha256 in run_r2_bindings.items():
        path = _regular_file(Path(raw_path), "Stage-C run-r2 source config")
        if sha256_file(path) != expected_sha256:
            raise PrerequisiteError("Stage-C run-r2 source config changed")
        if "matched_off_pilot_run_r2" in path.name:
            arm = "matched_off"
        elif "memory_pilot_run_r2" in path.name:
            arm = "memory"
        else:
            raise PrerequisiteError("Stage-C run-r2 source config name changed")
        if arm in run_r2_paths:
            raise PrerequisiteError("Stage-C run-r2 source config arms are not unique")
        run_r2_paths[arm] = path
    if set(run_r2_paths) != {"memory", "matched_off"}:
        raise PrerequisiteError("Stage-C run-r2 source config pair changed")

    configs = {
        "run_r2": {
            arm: load_config(path) for arm, path in run_r2_paths.items()
        },
        "run_r3": {
            "memory": load_config(memory_config),
            "matched_off": load_config(matched_off_config),
        },
    }
    for generation, generation_configs in configs.items():
        for arm, config in generation_configs.items():
            if not isinstance(config, Mapping):
                raise PrerequisiteError(f"Stage-C {generation} {arm} config is malformed")
            _validate_stage_c_arm_config(
                config=config, arm=arm, generation=generation
            )
        memory_normalized = _normalize_config_fields(
            generation_configs["memory"], _ARM_CONFIG_FIELDS, f"{generation} memory"
        )
        off_normalized = _normalize_config_fields(
            generation_configs["matched_off"],
            _ARM_CONFIG_FIELDS,
            f"{generation} matched-off",
        )
        if memory_normalized != off_normalized:
            raise PrerequisiteError(
                f"Stage-C {generation} arms differ outside the approved memory switch"
            )

    for arm in ("memory", "matched_off"):
        run_r2_normalized = _normalize_config_fields(
            configs["run_r2"][arm],
            _CROSS_GENERATION_CONFIG_FIELDS,
            f"run_r2 {arm}",
        )
        run_r3_normalized = _normalize_config_fields(
            configs["run_r3"][arm],
            _CROSS_GENERATION_CONFIG_FIELDS,
            f"run_r3 {arm}",
        )
        if run_r2_normalized != run_r3_normalized:
            raise PrerequisiteError(
                f"Stage-C run-r3 {arm} scientific settings differ from run-r2"
            )

    calibration_root = (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_sentence_memory_relevance_calibration_stage_c_generator_"
        "adaptation_protocol_v2_run_r3"
    )
    expected_outputs = {
        "memory": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3"
        ),
        "matched_off": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3"
        ),
    }
    for arm, config in configs["run_r3"].items():
        stage = config["sentence_memory_safety"]["stage_c"]
        if (
            config.get("experiment_name") != Path(expected_outputs[arm]).name
            or config.get("output", {}).get("out_dir") != expected_outputs[arm]
            or config.get("sentence_memory", {})
            .get("relevance_calibration", {})
            .get("artifact_dir")
            != calibration_root
            or stage.get("active_stage_c", {}).get("calibration_artifact_dir")
            != calibration_root
        ):
            raise PrerequisiteError(
                f"Stage-C protocol-v2 {arm} fresh namespace changed"
            )
    r3_outputs = [
        evidence_root / _safe_relative_path(value, "Stage-C protocol-v2 arm output")
        for value in expected_outputs.values()
    ]
    r2_outputs = [
        evidence_root
        / _safe_relative_path(
            config["output"]["out_dir"], "Stage-C run-r2 arm output"
        )
        for config in configs["run_r2"].values()
    ]
    archive_recovery = archive["recovery"]
    protocol_roots = r3_outputs + [
        evidence_root
        / _safe_relative_path(
            archive_recovery[name], f"Stage-C protocol-v2 {name}"
        )
        for name in (
            "new_calibration_root",
            "new_prerequisite_root",
            "new_smoke_root",
            "new_pilot_root",
        )
    ]
    if (
        len(set(protocol_roots)) != len(protocol_roots)
        or any(
            _paths_overlap(left, right)
            for index, left in enumerate(protocol_roots)
            for right in protocol_roots[index + 1 :]
        )
        or any(
            _paths_overlap(new, old) for new in protocol_roots for old in r2_outputs
        )
    ):
        raise PrerequisiteError("Stage-C protocol-v2 arm namespaces are not distinct")

    audit = {
        "schema_name": "signtrajfield_stage_c_protocol_v2_run_r3_config_audit",
        "schema_version": 1,
        "recovery_audit_identity": recovery_audit["audit_identity"],
        "memory_config_sha256": sha256_file(memory_config),
        "matched_off_config_sha256": sha256_file(matched_off_config),
        "run_r2_memory_config_sha256": sha256_file(run_r2_paths["memory"]),
        "run_r2_matched_off_config_sha256": sha256_file(
            run_r2_paths["matched_off"]
        ),
        "scientific_settings_equal_run_r2": True,
        "same_settings_except_arm_switch": True,
        "original_stage_b_warm_start_pinned": True,
        "development_only": True,
        "non_authorizing": True,
    }
    return {**audit, "audit_identity": digest_json(audit)}


def _validate_protocol_v3_run_r4_configs(
    *,
    recovery_audit: Mapping[str, Any],
    source_root: Path,
    evidence_root: Path,
    memory_config: Path,
    matched_off_config: Path,
) -> dict[str, Any]:
    """Prove v3/r4 is a fresh namespace with the exact r3 scientific config."""

    expected_names = {
        "memory": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v3_run_r4.yaml"
        ),
        "matched_off": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v3_run_r4.yaml"
        ),
    }
    r3_names = {
        "memory": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v2_run_r3.yaml"
        ),
        "matched_off": (
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v2_run_r3.yaml"
        ),
    }
    current = {"memory": memory_config, "matched_off": matched_off_config}
    configs: dict[str, dict[str, Mapping[str, Any]]] = {"run_r3": {}, "run_r4": {}}
    for arm, path in current.items():
        _regular_file(path, f"Stage-C protocol-v3 {arm} config")
        expected_path = (
            source_root / "NIAF/continuous_trajectory_field/configs" / expected_names[arm]
        ).resolve()
        if path.resolve() != expected_path:
            raise PrerequisiteError(f"Stage-C protocol-v3 {arm} config path changed")
        r3_path = (
            source_root / "NIAF/continuous_trajectory_field/configs" / r3_names[arm]
        )
        _regular_file(r3_path, f"Stage-C protocol-v2/run-r3 {arm} config")
        configs["run_r3"][arm] = load_config(r3_path)
        configs["run_r4"][arm] = load_config(path)
    for generation, generation_configs in configs.items():
        for arm, config in generation_configs.items():
            if not isinstance(config, Mapping):
                raise PrerequisiteError(f"Stage-C {generation} {arm} config is malformed")
            _validate_stage_c_arm_config(
                config=config,
                arm=arm,
                generation=generation,
            )
        if _normalize_config_fields(
            generation_configs["memory"], _ARM_CONFIG_FIELDS, f"{generation} memory"
        ) != _normalize_config_fields(
            generation_configs["matched_off"],
            _ARM_CONFIG_FIELDS,
            f"{generation} matched-off",
        ):
            raise PrerequisiteError(
                f"Stage-C {generation} arms differ outside the approved memory switch"
            )
    for arm in ("memory", "matched_off"):
        if _normalize_config_fields(
            configs["run_r3"][arm],
            _CROSS_GENERATION_CONFIG_FIELDS,
            f"run_r3 {arm}",
        ) != _normalize_config_fields(
            configs["run_r4"][arm],
            _CROSS_GENERATION_CONFIG_FIELDS,
            f"run_r4 {arm}",
        ):
            raise PrerequisiteError(
                f"Stage-C run-r4 {arm} scientific settings differ from run-r3"
            )
    calibration_root = (
        "experiments/NIAF/continuous_trajectory_field/"
        "csl_daily_sentence_memory_relevance_calibration_stage_c_generator_"
        "adaptation_protocol_v3_run_r4"
    )
    outputs = {
        "memory": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_memory_protocol_v3_run_r4"
        ),
        "matched_off": (
            "experiments/NIAF/continuous_trajectory_field/"
            "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
            "adaptation_matched_off_protocol_v3_run_r4"
        ),
    }
    fresh_roots = [
        evidence_root / _safe_relative_path(value, "Stage-C protocol-v3 arm output")
        for value in outputs.values()
    ]
    fresh_roots.extend(
        evidence_root
        / _safe_relative_path(value, "Stage-C protocol-v3 fresh namespace")
        for value in (
            calibration_root,
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_"
                "prerequisites"
            ),
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_protocol_v3_run_r4_smoke"
            ),
            (
                "experiments/NIAF/continuous_trajectory_field/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_protocol_v3_run_r4_pilot"
            ),
        )
    )
    if len(set(fresh_roots)) != len(fresh_roots) or any(
        _paths_overlap(left, right)
        for index, left in enumerate(fresh_roots)
        for right in fresh_roots[index + 1 :]
    ):
        raise PrerequisiteError("Stage-C protocol-v3 result namespaces overlap")
    for arm, config in configs["run_r4"].items():
        stage = config["sentence_memory_safety"]["stage_c"]
        if (
            config.get("experiment_name") != Path(outputs[arm]).name
            or config.get("output", {}).get("out_dir") != outputs[arm]
            or config.get("sentence_memory", {})
            .get("relevance_calibration", {})
            .get("artifact_dir")
            != calibration_root
            or stage.get("active_stage_c", {}).get("calibration_artifact_dir")
            != calibration_root
        ):
            raise PrerequisiteError(
                f"Stage-C protocol-v3 {arm} fresh namespace changed"
            )
    audit = {
        "schema_name": "signtrajfield_stage_c_protocol_v3_run_r4_config_audit",
        "schema_version": 1,
        "recovery_audit_identity": recovery_audit["audit_identity"],
        "memory_config_sha256": sha256_file(memory_config),
        "matched_off_config_sha256": sha256_file(matched_off_config),
        "run_r3_memory_config_sha256": sha256_file(
            source_root / "NIAF/continuous_trajectory_field/configs" / r3_names["memory"]
        ),
        "run_r3_matched_off_config_sha256": sha256_file(
            source_root
            / "NIAF/continuous_trajectory_field/configs"
            / r3_names["matched_off"]
        ),
        "scientific_settings_equal_protocol_v2_run_r3": True,
        "same_settings_except_arm_switch": True,
        "original_stage_b_warm_start_pinned": True,
        "development_only": True,
        "non_authorizing": True,
    }
    return {**audit, "audit_identity": digest_json(audit)}


def validate_protocol_v3_run_r4_configs(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
    memory_config: Path,
    matched_off_config: Path,
) -> dict[str, Any]:
    """Run the v3/r4 config audit in the CPU-only historical gate."""

    return _validate_protocol_v3_run_r4_configs(
        recovery_audit=validate_protocol_v3_run_r4_recovery_evidence(
            policy_path=policy_path,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=evidence_root,
        ),
        source_root=source_root,
        evidence_root=evidence_root,
        memory_config=memory_config,
        matched_off_config=matched_off_config,
    )


def validate_protocol_v3_run_r4_runtime_configs(
    *,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
    memory_config: Path,
    matched_off_config: Path,
) -> dict[str, Any]:
    """Recompute source/config hashes without historical archived-clone Git I/O."""

    return _validate_protocol_v3_run_r4_configs(
        recovery_audit=validate_protocol_v3_run_r4_runtime_bindings(
            policy_path=policy_path,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=evidence_root,
        ),
        source_root=source_root,
        evidence_root=evidence_root,
        memory_config=memory_config,
        matched_off_config=matched_off_config,
    )


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


def validate_protocol_v3_run_r4_cpu_gate(
    *,
    cpu_gate: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    policy_path: Path,
    recovery_manifest: Path,
    source_root: Path,
    evidence_root: Path,
    memory_config: Path,
    matched_off_config: Path,
) -> dict[str, Any]:
    """Authenticate schema-v2 CPU READY without rerunning historical Git scans."""

    head = source_git_head.lower()
    remote_head = source_remote_head.lower()
    recovery_audit = validate_protocol_v3_run_r4_runtime_bindings(
        policy_path=policy_path,
        recovery_manifest=recovery_manifest,
        source_root=source_root,
        evidence_root=evidence_root,
    )
    config_audit = validate_protocol_v3_run_r4_runtime_configs(
        policy_path=policy_path,
        recovery_manifest=recovery_manifest,
        source_root=source_root,
        evidence_root=evidence_root,
        memory_config=memory_config,
        matched_off_config=matched_off_config,
    )
    cpu = _exact_json(
        cpu_gate,
        label="Stage-C protocol-v3 CPU gate",
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
            "recovery_audit_identity",
            "config_audit_identity",
            "incident_archive_sha256",
            "decision_policy_sha256",
            "recovery_manifest_sha256",
            "memory_config_sha256",
            "matched_off_config_sha256",
            "historical_clone_metadata_checked_in_cpu_gate",
            "historical_clone_local_git_metadata_timeout_seconds",
            "historical_remote_ref_timeout_seconds",
        },
    )
    exact = {
        "schema_name": "signtrajfield_stage_c_cpu_gate",
        "schema_version": 2,
        "source_git_head": head,
        "source_remote_ref": source_remote_ref,
        "source_remote_head": remote_head,
        "slurm_job_id": cpu.get("slurm_job_id"),
        "compileall": True,
        "ruff_version": "0.12.0",
        "pytest_version": "8.4.2",
        "complete_repository_test_glob": "tests/test_*.py",
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
        "passed": True,
        "recovery_audit_identity": recovery_audit["audit_identity"],
        "config_audit_identity": config_audit["audit_identity"],
        "incident_archive_sha256": recovery_audit["incident_archive_sha256"],
        "decision_policy_sha256": recovery_audit["decision_policy_sha256"],
        "recovery_manifest_sha256": recovery_audit["recovery_manifest_sha256"],
        "memory_config_sha256": config_audit["memory_config_sha256"],
        "matched_off_config_sha256": config_audit["matched_off_config_sha256"],
        "historical_clone_metadata_checked_in_cpu_gate": True,
        "historical_clone_local_git_metadata_timeout_seconds": (
            PROTOCOL_V3_RUN_R4_LOCAL_GIT_TIMEOUT_SECONDS
        ),
        "historical_remote_ref_timeout_seconds": (
            PROTOCOL_V3_RUN_R4_REMOTE_REF_TIMEOUT_SECONDS
        ),
    }
    if cpu != exact or not str(cpu["slurm_job_id"]).isdigit():
        raise PrerequisiteError("Stage-C protocol-v3 CPU gate content is not exact")
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
    if recovery_policy.name == PROTOCOL_V3_RUN_R4_POLICY_NAME:
        recovery_audit = validate_protocol_v3_run_r4_runtime_bindings(
            policy_path=recovery_policy,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=recovery_evidence_root,
        )
        validate_protocol_v3_run_r4_cpu_gate(
            cpu_gate=cpu_gate,
            source_git_head=head,
            source_remote_ref=source_remote_ref,
            source_remote_head=remote_head,
            policy_path=recovery_policy,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=recovery_evidence_root,
            memory_config=stage_c_config,
            matched_off_config=(
                source_root
                / "NIAF/continuous_trajectory_field/configs/"
                "csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_"
                "adaptation_matched_off_protocol_v3_run_r4.yaml"
            ),
        )
    else:
        recovery_audit = validate_recovery_evidence(
            policy_path=recovery_policy,
            recovery_manifest=recovery_manifest,
            source_root=source_root,
            evidence_root=recovery_evidence_root,
        )
    expected_source_file_profile = recovery_audit.get(
        "calibration_source_file_profile", RETRY2_SOURCE_FILE_PROFILE
    )
    validate_source_terminal_decision(
        decision_path=source_terminal_decision,
        expected_sha256=SOURCE_TERMINAL_DECISION_SHA256,
        expected_identity=SOURCE_TERMINAL_DECISION_IDENTITY,
    )
    if recovery_policy.name != PROTOCOL_V3_RUN_R4_POLICY_NAME:
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
        expected_source_file_profile=expected_source_file_profile,
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
        != expected_source_file_profile
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
        "schema_name": (
            "signtrajfield_stage_c_prior_one_update_smoke_prerequisite"
        ),
        "schema_version": 1,
        "smoke_root_path": str(smoke_root.resolve()),
        "ready_path": str(smoke_ready.resolve()),
        "ready_sha256": sha256_file(smoke_ready),
        "mode_complete_path": str(mode_complete_path.resolve()),
        "mode_complete_sha256": sha256_file(mode_complete_path),
        "execution_complete_path": str(complete_path.resolve()),
        "execution_complete_sha256": sha256_file(complete_path),
        "decision_path": str(decision_path.resolve()),
        "decision_sha256": sha256_file(decision_path),
        "decision_identity": decision["decision_identity"],
        "decision_status": decision["status"],
        "decision_policy_path": str(policy_path.resolve()),
        "decision_policy_sha256": sha256_file(policy_path),
        "source_git_head": head,
        "source_remote_ref": complete["source_remote_ref"],
        "source_remote_head": complete["source_remote_head"],
        "pair_constraint": pair_constraint,
        "memory_config_sha256": binding["memory_config_sha256"],
        "matched_off_config_sha256": binding["matched_off_config_sha256"],
        "cpu_gate_sha256": binding["cpu_gate_sha256"],
        "calibration_completion_sha256": binding[
            "calibration_completion_sha256"
        ],
        "source_binding_sha256": complete["execution_artifacts"][
            "source_binding"
        ],
        "active_lease_claim_identity": ready["active_lease_claim_identity"],
        "execution_lease_claim_identity": ready[
            "execution_lease_claim_identity"
        ],
        "expected_epoch": 1,
        "expected_global_step_per_arm": 1,
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
        "training_network_profile": comparison["training_profile"],
        "development_only": True,
        "non_authorizing": True,
        "promotion_eligible": False,
        "authorized_purpose": None,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
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
    configs = commands.add_parser("validate-protocol-v2-configs")
    configs.add_argument("--policy_path", type=Path, required=True)
    configs.add_argument("--recovery_manifest", type=Path, required=True)
    configs.add_argument("--source_root", type=Path, required=True)
    configs.add_argument("--evidence_root", type=Path, required=True)
    configs.add_argument("--memory_config", type=Path, required=True)
    configs.add_argument("--matched_off_config", type=Path, required=True)
    v3_configs = commands.add_parser("validate-protocol-v3-configs")
    v3_configs.add_argument("--policy_path", type=Path, required=True)
    v3_configs.add_argument("--recovery_manifest", type=Path, required=True)
    v3_configs.add_argument("--source_root", type=Path, required=True)
    v3_configs.add_argument("--evidence_root", type=Path, required=True)
    v3_configs.add_argument("--memory_config", type=Path, required=True)
    v3_configs.add_argument("--matched_off_config", type=Path, required=True)
    runtime = commands.add_parser("validate-protocol-v3-runtime")
    runtime.add_argument("--policy_path", type=Path, required=True)
    runtime.add_argument("--recovery_manifest", type=Path, required=True)
    runtime.add_argument("--source_root", type=Path, required=True)
    runtime.add_argument("--evidence_root", type=Path, required=True)
    v3_cpu = commands.add_parser("validate-protocol-v3-cpu")
    v3_cpu.add_argument("--cpu_gate", type=Path, required=True)
    v3_cpu.add_argument("--source_git_head", required=True)
    v3_cpu.add_argument("--source_remote_ref", required=True)
    v3_cpu.add_argument("--source_remote_head", required=True)
    v3_cpu.add_argument("--policy_path", type=Path, required=True)
    v3_cpu.add_argument("--recovery_manifest", type=Path, required=True)
    v3_cpu.add_argument("--source_root", type=Path, required=True)
    v3_cpu.add_argument("--evidence_root", type=Path, required=True)
    v3_cpu.add_argument("--memory_config", type=Path, required=True)
    v3_cpu.add_argument("--matched_off_config", type=Path, required=True)
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
    unspent = commands.add_parser("validate-unspent")
    unspent.add_argument("--spend_marker", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = vars(args)
    command = kwargs.pop("command")
    if command == "validate-cpu":
        value = validate_cpu_gate(**kwargs)
    elif command == "validate-recovery":
        value = validate_recovery_evidence(**kwargs)
    elif command == "validate-protocol-v2-configs":
        value = validate_protocol_v2_run_r3_configs(**kwargs)
    elif command == "validate-protocol-v3-configs":
        value = validate_protocol_v3_run_r4_configs(**kwargs)
    elif command == "validate-protocol-v3-runtime":
        value = validate_protocol_v3_run_r4_runtime_bindings(**kwargs)
    elif command == "validate-protocol-v3-cpu":
        value = validate_protocol_v3_run_r4_cpu_gate(**kwargs)
    elif command == "validate-foundation":
        value = validate_foundation(**kwargs)
    elif command == "validate-smoke":
        value = validate_smoke(**kwargs)
    else:
        value = validate_unspent_confirmation_holdout(**kwargs)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
