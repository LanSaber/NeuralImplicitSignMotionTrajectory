"""Authenticated, recoverable singleton lease for Stage-C calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    AtomicPublishError,
    publish_bytes_no_replace,
    rename_no_replace,
)


LEASE_SCHEMA = "signtrajfield_stage_c_calibration_lease"
ATTESTATION_SCHEMA = "signtrajfield_stage_c_calibration_lease_attestation"
TERMINAL_SCHEMA = "signtrajfield_stage_c_calibration_lease_terminal"
SCHEMA_VERSION = 1
SOURCE_FILE_PROFILES = {
    "stage_c_generator_adaptation_v1",
    "stage_c_generator_adaptation_retry2_v1",
}
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


class CalibrationLeaseError(RuntimeError):
    """The calibration singleton lease is invalid or cannot be reconciled."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CalibrationLeaseError(f"lease record is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationLeaseError(f"invalid lease record: {path}") from error
    if not isinstance(value, dict):
        raise CalibrationLeaseError(f"lease record is not a mapping: {path}")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        publish_bytes_no_replace(path, encoded)
    except AtomicPublishError as error:
        raise CalibrationLeaseError(
            f"refusing to replace immutable calibration record: {path}"
        ) from error


def calibration_binding(
    *,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    launcher: Path,
    cpu_gate: Path,
    audit_config: Path,
    bank_manifest: Path,
    bank_ready: Path,
    train_neighbors: Path,
) -> dict[str, Any]:
    head = source_git_head.lower()
    files = {
        "launcher_sha256": launcher,
        "cpu_gate_sha256": cpu_gate,
        "audit_config_sha256": audit_config,
        "bank_manifest_sha256": bank_manifest,
        "bank_ready_sha256": bank_ready,
        "train_neighbors_sha256": train_neighbors,
    }
    if (
        re.fullmatch(r"[0-9a-f]{40}", head) is None
        or source_remote_head.lower() != head
        or re.fullmatch(r"origin/[A-Za-z0-9._/-]+", source_remote_ref) is None
        or any(not path.is_file() or path.is_symlink() for path in files.values())
    ):
        raise CalibrationLeaseError("calibration lease binding is malformed")
    payload = {
        "schema_name": "signtrajfield_stage_c_calibration_binding",
        "schema_version": SCHEMA_VERSION,
        "source_git_head": head,
        "source_remote_ref": source_remote_ref,
        "source_remote_head": head,
        **{name: sha256_file(path) for name, path in files.items()},
        "source_file_profile": "stage_c_generator_adaptation_retry2_v1",
        "seed": 1234,
        "duration_weight": 0.05,
        "minimum_auroc": 0.75,
        "minimum_probability_gap": 0.20,
        "train_only": True,
        "development_only": True,
        "confirmation_manifest_opened": False,
        "test_data_accessed": False,
    }
    return {**payload, "digest": digest_json(payload)}


def _validate_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationLeaseError("calibration binding is not a mapping")
    unsigned = {key: item for key, item in value.items() if key != "digest"}
    sha_fields = [name for name in value if name.endswith("_sha256")]
    if (
        set(value)
        != {
            "schema_name",
            "schema_version",
            "source_git_head",
            "source_remote_ref",
            "source_remote_head",
            "launcher_sha256",
            "cpu_gate_sha256",
            "audit_config_sha256",
            "bank_manifest_sha256",
            "bank_ready_sha256",
            "train_neighbors_sha256",
            "source_file_profile",
            "seed",
            "duration_weight",
            "minimum_auroc",
            "minimum_probability_gap",
            "train_only",
            "development_only",
            "confirmation_manifest_opened",
            "test_data_accessed",
            "digest",
        }
        or value.get("schema_name") != "signtrajfield_stage_c_calibration_binding"
        or value.get("schema_version") != SCHEMA_VERSION
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_git_head", ""))) is None
        or value.get("source_remote_head") != value.get("source_git_head")
        or re.fullmatch(
            r"origin/[A-Za-z0-9._/-]+", str(value.get("source_remote_ref", ""))
        )
        is None
        or value.get("source_file_profile") not in SOURCE_FILE_PROFILES
        or value.get("seed") != 1234
        or value.get("duration_weight") != 0.05
        or value.get("minimum_auroc") != 0.75
        or value.get("minimum_probability_gap") != 0.20
        or value.get("train_only") is not True
        or value.get("development_only") is not True
        or value.get("confirmation_manifest_opened") is not False
        or value.get("test_data_accessed") is not False
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value[name])) is None for name in sha_fields)
        or value.get("digest") != digest_json(unsigned)
    ):
        raise CalibrationLeaseError("calibration binding content is not exact")
    return value


def _validate_owner(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise CalibrationLeaseError("calibration active lease is not a directory")
    owner = _json(path / "owner.json")
    ready = _json(path / "READY")
    unsigned = {key: value for key, value in owner.items() if key != "claim_identity"}
    _validate_binding(owner.get("binding"))
    if (
        set(owner)
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
        or owner.get("schema_name") != LEASE_SCHEMA
        or owner.get("schema_version") != SCHEMA_VERSION
        or re.fullmatch(r"[0-9]+", str(owner.get("slurm_job_id", ""))) is None
        or not isinstance(owner.get("slurm_restart_count"), int)
        or owner.get("slurm_restart_count") < 0
        or re.fullmatch(r"[0-9a-f]{32}", str(owner.get("claim_nonce", ""))) is None
        or not isinstance(owner.get("created_unix_ns"), int)
        or owner.get("claim_identity") != digest_json(unsigned)
        or ready
        != {
            "schema_name": LEASE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "claim_identity": owner.get("claim_identity"),
        }
    ):
        raise CalibrationLeaseError("calibration active lease is malformed")
    return owner


def _scheduler(job_id: str) -> tuple[str, int]:
    try:
        result = subprocess.run(
            ("scontrol", "show", "job", "-o", job_id),
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CalibrationLeaseError("cannot query calibration lease owner") from error
    state = re.search(r"(?:^|\s)JobState=([A-Z_]+)", result.stdout)
    restarts = re.search(r"(?:^|\s)Restarts=([0-9]+)", result.stdout)
    if result.returncode or state is None or restarts is None:
        raise CalibrationLeaseError("scheduler did not prove calibration owner state")
    return state.group(1), int(restarts.group(1))


def _terminal(job_id: str) -> dict[str, Any]:
    state, restarts = _scheduler(job_id)
    if state not in TERMINAL_STATES:
        raise CalibrationLeaseError(f"calibration lease owner {job_id} is live ({state})")
    return {"job_id": job_id, "state": state, "restarts": restarts}


def _publish(path: Path, owner: dict[str, Any]) -> bool:
    building = path.with_name(f".{path.name}.claiming.{uuid.uuid4().hex}")
    building.mkdir(parents=False, exist_ok=False)
    try:
        _exclusive_json(building / "owner.json", owner)
        _exclusive_json(
            building / "READY",
            {
                "schema_name": LEASE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "claim_identity": owner["claim_identity"],
            },
        )
        try:
            rename_no_replace(building, path)
        except (AtomicPublishError, OSError):
            return False
        return True
    finally:
        if building.exists():
            shutil.rmtree(building)


def acquire(
    *,
    lease: Path,
    attestation: Path,
    job_id: str,
    restart: int,
    binding: dict[str, Any],
) -> dict[str, Any]:
    _validate_binding(binding)
    if re.fullmatch(r"[0-9]+", job_id) is None or restart < 0:
        raise CalibrationLeaseError("numeric Slurm job/restart required")
    lease = lease.resolve()
    lease.parent.mkdir(parents=True, exist_ok=True)
    owner = None
    reconciliation = None
    for _attempt in range(3):
        if lease.exists():
            existing = _validate_owner(lease)
            same_logical_job = (
                existing["slurm_job_id"] == job_id
                and existing["binding"] == binding
            )
            interrupted_terminal_path = lease / "TERMINAL.json"
            if interrupted_terminal_path.exists():
                interrupted_terminal = _json(interrupted_terminal_path)
                interrupted_unsigned = {
                    key: value
                    for key, value in interrupted_terminal.items()
                    if key != "terminal_identity"
                }
                if (
                    interrupted_terminal.get("terminal_identity")
                    != digest_json(interrupted_unsigned)
                    or interrupted_terminal.get("claim_identity")
                    != existing["claim_identity"]
                    or interrupted_terminal.get("outcome") != "failed"
                ):
                    raise CalibrationLeaseError(
                        "released calibration lease requires completion reconciliation"
                    )
                if same_logical_job:
                    state, scheduler_restart = _scheduler(job_id)
                    if (
                        restart <= existing["slurm_restart_count"]
                        or scheduler_restart < restart
                        or state in TERMINAL_STATES
                    ):
                        raise CalibrationLeaseError(
                            "scheduler did not prove failed-lease requeue"
                        )
                else:
                    _terminal(existing["slurm_job_id"])
                history = lease.with_name(f"{lease.name}.history")
                history.mkdir(parents=True, exist_ok=True)
                rename_no_replace(
                    lease,
                    history / f"{existing['claim_identity']}.failed",
                )
                continue
            if same_logical_job:
                state, scheduler_restart = _scheduler(job_id)
                if (
                    restart <= existing["slurm_restart_count"]
                    or scheduler_restart < restart
                    or state in TERMINAL_STATES
                ):
                    raise CalibrationLeaseError("scheduler did not prove a newer requeue")
                owner = existing
                reconciliation = {
                    "kind": "same_job_requeue",
                    "state": state,
                    "prior_restart": existing["slurm_restart_count"],
                    "current_restart": restart,
                    "scheduler_restart": scheduler_restart,
                }
                break
            scheduler = _terminal(existing["slurm_job_id"])
            history = lease.with_name(f"{lease.name}.history")
            history.mkdir(parents=True, exist_ok=True)
            suffix = "stale"
            terminal_path = lease / "TERMINAL.json"
            if terminal_path.exists():
                terminal = _json(terminal_path)
                unsigned = {
                    key: value
                    for key, value in terminal.items()
                    if key != "terminal_identity"
                }
                if (
                    terminal.get("claim_identity") != existing["claim_identity"]
                    or terminal.get("terminal_identity") != digest_json(unsigned)
                ):
                    raise CalibrationLeaseError("existing terminal record changed")
                suffix = str(terminal.get("outcome"))
            destination = history / f"{existing['claim_identity']}.{suffix}"
            had_terminal = terminal_path.exists()
            rename_no_replace(lease, destination)
            if not had_terminal and suffix == "stale":
                stale_payload = {
                    "schema_name": TERMINAL_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "outcome": "stale",
                    "claim_identity": existing["claim_identity"],
                    "scheduler_evidence": scheduler,
                }
                _exclusive_json(
                    destination / "STALE_TERMINAL.json",
                    {**stale_payload, "terminal_identity": digest_json(stale_payload)},
                )
        unsigned = {
            "schema_name": LEASE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "slurm_job_id": job_id,
            "slurm_restart_count": restart,
            "binding": binding,
            "claim_nonce": uuid.uuid4().hex,
            "created_unix_ns": time.time_ns(),
        }
        candidate = {**unsigned, "claim_identity": digest_json(unsigned)}
        if _publish(lease, candidate):
            owner = candidate
            break
    if owner is None:
        raise CalibrationLeaseError("cannot acquire calibration lease atomically")
    unsigned_attestation = {
        "schema_name": ATTESTATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "lease_path": str(lease),
        "current_slurm_job_id": job_id,
        "current_slurm_restart_count": restart,
        "claim": owner,
        "restart_reconciliation": reconciliation,
    }
    value = {
        **unsigned_attestation,
        "attestation_identity": digest_json(unsigned_attestation),
    }
    _exclusive_json(attestation.resolve(), value)
    return value


def _validate_attestation(lease: Path, attestation: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _json(attestation.resolve())
    unsigned = {key: item for key, item in value.items() if key != "attestation_identity"}
    owner = _validate_owner(lease.resolve())
    if (
        value.get("attestation_identity") != digest_json(unsigned)
        or Path(str(value.get("lease_path", ""))).resolve() != lease.resolve()
        or value.get("claim") != owner
    ):
        raise CalibrationLeaseError("calibration attestation is not the active claim")
    return value, owner


def _archive(
    *, lease: Path, attestation: Path, outcome: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    value, owner = _validate_attestation(lease, attestation)
    unsigned = {
        "schema_name": TERMINAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "claim_identity": owner["claim_identity"],
        "current_slurm_job_id": value["current_slurm_job_id"],
        "current_slurm_restart_count": value["current_slurm_restart_count"],
        "attestation_sha256": sha256_file(attestation),
        **dict(evidence),
    }
    terminal = {**unsigned, "terminal_identity": digest_json(unsigned)}
    terminal_path = lease / "TERMINAL.json"
    archive_outcome = outcome
    if terminal_path.exists():
        existing = _json(terminal_path)
        existing_unsigned = {
            key: item for key, item in existing.items() if key != "terminal_identity"
        }
        compatible_recovery = (
            outcome == "recovered_released"
            and existing.get("outcome") == "released"
        )
        if (
            existing.get("terminal_identity") != digest_json(existing_unsigned)
            or existing.get("claim_identity") != owner["claim_identity"]
            or (
                not compatible_recovery
                and existing.get("outcome") != outcome
            )
            or any(
                existing.get(key) != item
                for key, item in evidence.items()
                if key != "scheduler_evidence" or not compatible_recovery
            )
        ):
            raise CalibrationLeaseError("existing calibration terminal differs")
        terminal = existing
        archive_outcome = str(existing["outcome"])
    else:
        _exclusive_json(terminal_path, terminal)
    history = lease.with_name(f"{lease.name}.history")
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"{owner['claim_identity']}.{archive_outcome}"
    rename_no_replace(lease, destination)
    return {"history_path": str(destination), "terminal": terminal}


def release(*, lease: Path, attestation: Path, completion: Path) -> dict[str, Any]:
    _value, owner = _validate_attestation(lease, attestation)
    completed = _json(completion)
    publication_attestation = Path(
        str(completed.get("lease_attestation_path", ""))
    )
    publication_value, publication_owner = _validate_attestation(
        lease, publication_attestation
    )
    if (
        completed.get("active_lease_claim_identity") != owner["claim_identity"]
        or publication_owner != owner
        or completed.get("lease_attestation_sha256")
        != sha256_file(publication_attestation)
        or completed.get("lease_attestation_identity")
        != publication_value.get("attestation_identity")
        or completed.get("source_git_head") != owner["binding"]["source_git_head"]
    ):
        raise CalibrationLeaseError("calibration completion does not bind active lease")
    return _archive(
        lease=lease,
        attestation=attestation,
        outcome="released",
        evidence={
            "completion_path": str(completion.resolve()),
            "completion_sha256": sha256_file(completion),
        },
    )


def inspect_completed_recovery(
    *, lease: Path, completion: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Authenticate a published completion that still has its active lease."""

    owner = _validate_owner(lease.resolve())
    completed = _json(completion)
    publication_attestation = Path(
        str(completed.get("lease_attestation_path", ""))
    )
    publication_value, publication_owner = _validate_attestation(
        lease, publication_attestation
    )
    if (
        completed.get("schema_name")
        != "signtrajfield_stage_c_calibration_completion"
        or completed.get("schema_version") != SCHEMA_VERSION
        or completed.get("active_lease_claim_identity")
        != owner["claim_identity"]
        or publication_owner != owner
        or completed.get("lease_attestation_sha256")
        != sha256_file(publication_attestation)
        or completed.get("lease_attestation_identity")
        != publication_value.get("attestation_identity")
        or completed.get("source_git_head")
        != owner["binding"]["source_git_head"]
        or completed.get("source_remote_ref")
        != owner["binding"]["source_remote_ref"]
        or completed.get("source_remote_head")
        != owner["binding"]["source_remote_head"]
        or completed.get("cpu_gate_sha256")
        != owner["binding"]["cpu_gate_sha256"]
        or completed.get("launcher_sha256")
        != owner["binding"]["launcher_sha256"]
        or completed.get("train_only") is not True
        or completed.get("development_only") is not True
        or completed.get("confirmation_manifest_opened") is not False
        or completed.get("test_data_accessed") is not False
    ):
        raise CalibrationLeaseError(
            "calibration completion is not the authenticated active claim"
        )
    return owner, completed, publication_attestation


def reconcile_completed(
    *,
    lease: Path,
    completion: Path,
    current_job_id: str,
    current_restart: int,
) -> dict[str, Any]:
    owner, completed, attestation = inspect_completed_recovery(
        lease=lease, completion=completion
    )
    current_job = str(current_job_id)
    restart = int(current_restart)
    if re.fullmatch(r"[0-9]+", current_job) is None or restart < 0:
        raise CalibrationLeaseError(
            "completion reconciliation requires numeric current job/restart"
        )
    if current_job == owner["slurm_job_id"]:
        state, scheduler_restart = _scheduler(current_job)
        if (
            restart <= owner["slurm_restart_count"]
            or scheduler_restart < restart
            or state in TERMINAL_STATES
        ):
            raise CalibrationLeaseError(
                "scheduler did not prove a newer calibration requeue"
            )
        scheduler = {
            "classification": "scheduler_requeue_restart",
            "job_id": current_job,
            "state": state,
            "prior_restart_count": owner["slurm_restart_count"],
            "current_restart_count": restart,
            "scheduler_restart_count": scheduler_restart,
        }
    else:
        scheduler = {
            "classification": "scheduler_terminal",
            **_terminal(owner["slurm_job_id"]),
        }
    return _archive(
        lease=lease,
        attestation=attestation,
        outcome="recovered_released",
        evidence={
            "scheduler_evidence": scheduler,
            "completion_path": str(completion.resolve()),
            "completion_sha256": sha256_file(completion),
        },
    )


def fail(*, lease: Path, attestation: Path, exit_code: int) -> dict[str, Any]:
    return _archive(
        lease=lease,
        attestation=attestation,
        outcome="failed",
        evidence={"self_reported_exit_code": int(exit_code)},
    )


def _binding_args(args: argparse.Namespace) -> dict[str, Any]:
    return calibration_binding(
        source_git_head=args.source_git_head,
        source_remote_ref=args.source_remote_ref,
        source_remote_head=args.source_remote_head,
        launcher=args.launcher,
        cpu_gate=args.cpu_gate,
        audit_config=args.audit_config,
        bank_manifest=args.bank_manifest,
        bank_ready=args.bank_ready,
        train_neighbors=args.train_neighbors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument("--lease", type=Path, required=True)
    acquire_parser.add_argument("--attestation", type=Path, required=True)
    acquire_parser.add_argument("--slurm_job_id", required=True)
    acquire_parser.add_argument("--slurm_restart_count", type=int, required=True)
    acquire_parser.add_argument("--source_git_head", required=True)
    acquire_parser.add_argument("--source_remote_ref", required=True)
    acquire_parser.add_argument("--source_remote_head", required=True)
    for name in (
        "launcher",
        "cpu_gate",
        "audit_config",
        "bank_manifest",
        "bank_ready",
        "train_neighbors",
    ):
        acquire_parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("release", "fail"):
        command = commands.add_parser(name)
        command.add_argument("--lease", type=Path, required=True)
        command.add_argument("--attestation", type=Path, required=True)
        if name == "release":
            command.add_argument("--completion", type=Path, required=True)
        else:
            command.add_argument("--exit_code", type=int, required=True)
    reconcile = commands.add_parser("reconcile-completed")
    reconcile.add_argument("--lease", type=Path, required=True)
    reconcile.add_argument("--completion", type=Path, required=True)
    reconcile.add_argument("--current_job_id", required=True)
    reconcile.add_argument("--current_restart", type=int, required=True)
    inspect = commands.add_parser("inspect-completed")
    inspect.add_argument("--lease", type=Path, required=True)
    inspect.add_argument("--completion", type=Path, required=True)
    inspect.add_argument("--expected_source_git_head", required=True)
    inspect.add_argument("--expected_source_remote_ref", required=True)
    inspect.add_argument("--expected_source_remote_head", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "acquire":
        result = acquire(
            lease=args.lease,
            attestation=args.attestation,
            job_id=args.slurm_job_id,
            restart=args.slurm_restart_count,
            binding=_binding_args(args),
        )
    elif args.command == "release":
        result = release(
            lease=args.lease,
            attestation=args.attestation,
            completion=args.completion,
        )
    elif args.command == "fail":
        result = fail(
            lease=args.lease,
            attestation=args.attestation,
            exit_code=args.exit_code,
        )
    elif args.command == "reconcile-completed":
        result = reconcile_completed(
            lease=args.lease,
            completion=args.completion,
            current_job_id=args.current_job_id,
            current_restart=args.current_restart,
        )
    else:
        owner, completed, attestation = inspect_completed_recovery(
            lease=args.lease,
            completion=args.completion,
        )
        binding = owner["binding"]
        if (
            binding["source_git_head"] != args.expected_source_git_head.lower()
            or binding["source_remote_ref"] != args.expected_source_remote_ref
            or binding["source_remote_head"]
            != args.expected_source_remote_head.lower()
        ):
            raise CalibrationLeaseError(
                "calibration recovery belongs to another immutable source"
            )
        result = {
            "claim_identity": owner["claim_identity"],
            "completion_sha256": sha256_file(args.completion),
            "attestation_path": str(attestation.resolve()),
            "artifact_identity": completed.get("artifact_identity"),
            "recovery_needed": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
