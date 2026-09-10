"""Authenticated singleton execution leases for paired Stage-C experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


LEASE_SCHEMA_NAME = "signtrajfield_stage_c_execution_lease"
ATTESTATION_SCHEMA_NAME = "signtrajfield_stage_c_execution_lease_attestation"
TERMINAL_SCHEMA_NAME = "signtrajfield_stage_c_execution_lease_terminal"
SCHEMA_VERSION = 1
MODES = {"smoke", "pilot"}
PROTOCOL_V2_RUN_R3_POLICY_NAME = (
    "csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_"
    "decision_policy_v1.json"
)
PAIR_RE = re.compile(r"pair(0[1-9]|1[0-5])")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
BINDING_FIELDS = {
    "schema_name",
    "schema_version",
    "mode",
    "source_git_head",
    "source_remote_ref",
    "source_remote_head",
    "pair_constraint",
    "world_size",
    "batch_per_rank",
    "accumulation_steps",
    "effective_global_batch",
    "memory_config_sha256",
    "matched_off_config_sha256",
    "cpu_gate_sha256",
    "calibration_completion_sha256",
    "development_only",
    "non_authorizing",
    "digest",
}
TERMINAL_SLURM_STATES = {
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


class StageCExecutionControlError(RuntimeError):
    """A singleton Stage-C claim is invalid, live, or cannot be reconciled."""


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


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StageCExecutionControlError(f"record is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageCExecutionControlError(f"invalid JSON record {path}") from error
    if not isinstance(value, dict):
        raise StageCExecutionControlError(f"record is not a JSON mapping: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        publish_bytes_no_replace(path, encoded)
    except AtomicPublishError as error:
        raise StageCExecutionControlError(
            f"refusing to replace immutable record: {path}"
        ) from error
    return dict(value)


def execution_binding(
    *,
    mode: str,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    pair_constraint: str,
    memory_config: Path,
    matched_off_config: Path,
    cpu_gate: Path,
    calibration_completion: Path,
) -> dict[str, Any]:
    head = str(source_git_head).lower()
    remote_head = str(source_remote_head).lower()
    if (
        mode not in MODES
        or PAIR_RE.fullmatch(pair_constraint) is None
        or re.fullmatch(r"[0-9a-f]{40}", head) is None
        or remote_head != head
        or not source_remote_ref.startswith("origin/")
    ):
        raise StageCExecutionControlError("Stage-C execution binding is malformed")
    paths = {
        "memory_config_sha256": memory_config,
        "matched_off_config_sha256": matched_off_config,
        "cpu_gate_sha256": cpu_gate,
        "calibration_completion_sha256": calibration_completion,
    }
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise StageCExecutionControlError(f"binding input is not regular: {label}")
    payload = {
        "schema_name": "signtrajfield_stage_c_execution_binding",
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "source_git_head": head,
        "source_remote_ref": source_remote_ref,
        "source_remote_head": remote_head,
        "pair_constraint": pair_constraint,
        "world_size": 2,
        "batch_per_rank": 64,
        "accumulation_steps": 2,
        "effective_global_batch": 256,
        **{name: sha256_file(path) for name, path in paths.items()},
        "development_only": True,
        "non_authorizing": True,
    }
    return {**payload, "digest": digest_json(payload)}


def _validate_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != BINDING_FIELDS:
        raise StageCExecutionControlError("execution binding fields are not exact")
    unsigned = {key: item for key, item in value.items() if key != "digest"}
    digest_fields = (
        "memory_config_sha256",
        "matched_off_config_sha256",
        "cpu_gate_sha256",
        "calibration_completion_sha256",
    )
    if (
        value.get("schema_name") != "signtrajfield_stage_c_execution_binding"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("mode") not in MODES
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_git_head", "")))
        is None
        or value.get("source_remote_head") != value.get("source_git_head")
        or re.fullmatch(r"origin/[A-Za-z0-9._/-]+", str(value.get("source_remote_ref", "")))
        is None
        or PAIR_RE.fullmatch(str(value.get("pair_constraint", ""))) is None
        or value.get("world_size") != 2
        or value.get("batch_per_rank") != 64
        or value.get("accumulation_steps") != 2
        or value.get("effective_global_batch") != 256
        or any(SHA256_RE.fullmatch(str(value.get(name, ""))) is None for name in digest_fields)
        or value.get("development_only") is not True
        or value.get("non_authorizing") is not True
        or value.get("digest") != digest_json(unsigned)
    ):
        raise StageCExecutionControlError("execution binding content is malformed")
    return value


def _validate_protocol_v2_smoke_prerequisite(
    *,
    mode: str,
    completion: Mapping[str, Any],
    ready: Mapping[str, Any],
    execution_complete: Mapping[str, Any],
    decision: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> None:
    records = (completion, ready, execution_complete, decision, launch)
    if mode == "smoke":
        if any("prior_one_update_smoke" in value for value in records):
            raise StageCExecutionControlError(
                "smoke publication contains a forbidden self-prerequisite"
            )
        return
    if mode != "pilot":
        raise StageCExecutionControlError(
            "protocol-v2 publication mode is not exact"
        )
    prior_smoke = completion.get("prior_one_update_smoke")
    if not isinstance(prior_smoke, dict) or any(
        value.get("prior_one_update_smoke") != prior_smoke
        for value in records[1:]
    ):
        raise StageCExecutionControlError(
            "pilot smoke prerequisite differs across publication evidence"
        )
    from NIAF.continuous_trajectory_field.scripts.stage_c_pilot_decision import (
        StageCDecisionError,
        _validate_pilot_smoke_prerequisite,
    )

    try:
        _validate_pilot_smoke_prerequisite(
            value=prior_smoke,
            complete=execution_complete,
            launch=launch,
        )
    except StageCDecisionError as error:
        raise StageCExecutionControlError(
            f"pilot smoke prerequisite is invalid: {error}"
        ) from error


def _validate_claim(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise StageCExecutionControlError(f"active lease is not a real directory: {path}")
    owner = _json(path / "owner.json")
    ready = _json(path / "READY")
    expected_fields = {
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
    payload = {key: value for key, value in owner.items() if key != "claim_identity"}
    binding = owner.get("execution_binding")
    try:
        _validate_binding(binding)
    except StageCExecutionControlError as error:
        raise StageCExecutionControlError(
            f"active execution lease has an invalid binding: {path}"
        ) from error
    replaced = owner.get("replaced_stale_owner")
    prior = owner.get("prior_claim_identities")
    replaced_valid = replaced is None or (
        isinstance(replaced, dict)
        and set(replaced)
        == {
            "claim_identity",
            "slurm_job_id",
            "slurm_restart_count",
            "scheduler_evidence",
        }
        and SHA256_RE.fullmatch(str(replaced.get("claim_identity", ""))) is not None
        and re.fullmatch(r"[0-9]+", str(replaced.get("slurm_job_id", ""))) is not None
        and isinstance(replaced.get("slurm_restart_count"), int)
        and isinstance(replaced.get("scheduler_evidence"), dict)
    )
    if (
        set(owner) != expected_fields
        or owner.get("schema_name") != LEASE_SCHEMA_NAME
        or owner.get("schema_version") != SCHEMA_VERSION
        or owner.get("mode") not in MODES
        or re.fullmatch(r"[0-9]+", str(owner.get("slurm_job_id", ""))) is None
        or not isinstance(owner.get("slurm_restart_count"), int)
        or int(owner["slurm_restart_count"]) < 0
        or not replaced_valid
        or not isinstance(prior, list)
        or len(prior) != len(set(prior))
        or any(SHA256_RE.fullmatch(str(value)) is None for value in prior)
        or owner.get("claim_identity") in prior
        or binding.get("mode") != owner.get("mode")
        or re.fullmatch(r"[0-9a-f]{32}", str(owner.get("claim_nonce", ""))) is None
        or not isinstance(owner.get("created_unix_ns"), int)
        or owner.get("claim_identity") != digest_json(payload)
        or ready
        != {
            "schema_name": LEASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "claim_identity": owner.get("claim_identity"),
        }
    ):
        raise StageCExecutionControlError(f"active execution lease is malformed: {path}")
    return owner


def _publish_lease(path: Path, owner: Mapping[str, Any]) -> bool:
    building = path.with_name(
        f".{path.name}.claiming.{owner['slurm_job_id']}.{uuid.uuid4().hex}"
    )
    building.mkdir(parents=False, exist_ok=False)
    try:
        _atomic_json(building / "owner.json", owner)
        _atomic_json(
            building / "READY",
            {
                "schema_name": LEASE_SCHEMA_NAME,
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


def _scheduler_job(job_id: str) -> tuple[str, int, str]:
    try:
        result = subprocess.run(
            ("scontrol", "show", "job", "-o", str(job_id)),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StageCExecutionControlError(
            f"cannot query scheduler for lease owner {job_id}"
        ) from error
    combined = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        raise StageCExecutionControlError(
            f"scheduler did not prove lease owner {job_id} terminal: {combined}"
        )
    state_match = re.search(r"(?:^|\s)JobState=([A-Z_]+)", result.stdout)
    restart_match = re.search(r"(?:^|\s)Restarts=([0-9]+)", result.stdout)
    if state_match is None or restart_match is None:
        raise StageCExecutionControlError(
            f"scheduler response lacks state/restart evidence for {job_id}"
        )
    return state_match.group(1), int(restart_match.group(1)), result.stdout.strip()


def _terminal_evidence(job_id: str) -> dict[str, Any]:
    state, restarts, _raw = _scheduler_job(job_id)
    if state not in TERMINAL_SLURM_STATES:
        raise StageCExecutionControlError(
            f"execution lease is owned by live Slurm job {job_id} ({state})"
        )
    return {
        "classification": "scheduler_terminal",
        "job_id": str(job_id),
        "state": state,
        "restarts": restarts,
    }


def _restart_evidence(job_id: str, old_restart: int, current_restart: int) -> dict[str, Any]:
    state, scheduler_restarts, _raw = _scheduler_job(job_id)
    if (
        current_restart <= old_restart
        or scheduler_restarts < current_restart
        or state in TERMINAL_SLURM_STATES
    ):
        raise StageCExecutionControlError(
            "scheduler did not prove a newer live generation of the same Slurm job"
        )
    return {
        "classification": "scheduler_requeue_restart",
        "job_id": str(job_id),
        "state": state,
        "prior_restart_count": old_restart,
        "current_restart_count": current_restart,
        "scheduler_restart_count": scheduler_restarts,
    }


def _validate_terminal_record(path: Path, *, claim_identity: str) -> dict[str, Any]:
    terminal = _json(path)
    unsigned = {
        key: value for key, value in terminal.items() if key != "terminal_identity"
    }
    if (
        terminal.get("schema_name") != TERMINAL_SCHEMA_NAME
        or terminal.get("schema_version") != SCHEMA_VERSION
        or terminal.get("claim_identity") != claim_identity
        or terminal.get("outcome")
        not in {"failed", "released", "recovered_released"}
        or terminal.get("terminal_identity") != digest_json(unsigned)
    ):
        raise StageCExecutionControlError("existing lease terminal record is malformed")
    return terminal


def acquire_execution_lease(
    *,
    lease_path: Path,
    attestation_path: Path,
    slurm_job_id: str,
    slurm_restart_count: int,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Acquire one mode/pair lease, or attest a scheduler-proven requeue."""

    job_id = str(slurm_job_id)
    restart = int(slurm_restart_count)
    if re.fullmatch(r"[0-9]+", job_id) is None or restart < 0:
        raise StageCExecutionControlError("lease requires numeric Slurm job/restart IDs")
    _validate_binding(binding)
    lease_path = lease_path.resolve()
    attestation_path = attestation_path.resolve()
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    replaced = None
    prior_claim_identities: set[str] = set()
    restart_reconciliation = None
    owner = None
    history = lease_path.with_name(f"{lease_path.name}.history")
    if history.is_dir() and not lease_path.exists():
        for archived in history.iterdir():
            if not archived.is_dir() or archived.is_symlink():
                continue
            try:
                archived_owner = _validate_claim(archived)
            except StageCExecutionControlError:
                continue
            if archived_owner.get("execution_binding") != binding:
                continue
            if not (
                (archived / "TERMINAL.json").is_file()
                or (archived / "STALE_TERMINAL.json").is_file()
            ):
                continue
            prior_claim_identities.add(archived_owner["claim_identity"])
            prior_claim_identities.update(
                archived_owner.get("prior_claim_identities", [])
            )
    for _attempt in range(3):
        if lease_path.exists():
            existing = _validate_claim(lease_path)
            same_logical_job = (
                existing["slurm_job_id"] == job_id
                and existing["mode"] == binding["mode"]
                and existing["execution_binding"] == binding
            )
            interrupted_terminal_path = lease_path / "TERMINAL.json"
            if interrupted_terminal_path.exists():
                interrupted_terminal = _validate_terminal_record(
                    interrupted_terminal_path,
                    claim_identity=existing["claim_identity"],
                )
                if interrupted_terminal["outcome"] != "failed":
                    raise StageCExecutionControlError(
                        "a released active lease requires exact publication reconciliation"
                    )
                if same_logical_job:
                    _restart_evidence(
                        job_id, existing["slurm_restart_count"], restart
                    )
                else:
                    _terminal_evidence(str(existing["slurm_job_id"]))
                history.mkdir(parents=True, exist_ok=True)
                destination = history / f"{existing['claim_identity']}.failed"
                rename_no_replace(lease_path, destination)
                prior_claim_identities.add(existing["claim_identity"])
                prior_claim_identities.update(
                    existing.get("prior_claim_identities", [])
                )
                continue
            if same_logical_job:
                if restart == existing["slurm_restart_count"]:
                    raise StageCExecutionControlError(
                        "the exact Slurm job/restart already owns the active lease"
                    )
                restart_reconciliation = _restart_evidence(
                    job_id, existing["slurm_restart_count"], restart
                )
                owner = existing
                break
            scheduler = _terminal_evidence(str(existing["slurm_job_id"]))
            history = lease_path.with_name(f"{lease_path.name}.history")
            history.mkdir(parents=True, exist_ok=True)
            existing_terminal_path = lease_path / "TERMINAL.json"
            existing_terminal = None
            if existing_terminal_path.exists():
                existing_terminal = _validate_terminal_record(
                    existing_terminal_path,
                    claim_identity=existing["claim_identity"],
                )
            suffix = existing_terminal["outcome"] if existing_terminal else "stale"
            stale = history / f"{existing['claim_identity']}.{suffix}"
            try:
                rename_no_replace(lease_path, stale)
            except (AtomicPublishError, FileNotFoundError, OSError):
                continue
            if existing_terminal is None:
                stale_terminal_payload = {
                    "schema_name": TERMINAL_SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "outcome": "stale_scheduler_terminal",
                    "claim_identity": existing["claim_identity"],
                    "scheduler_evidence": scheduler,
                    "replacement_slurm_job_id": job_id,
                    "replacement_slurm_restart_count": restart,
                }
                _atomic_json(
                    stale / "STALE_TERMINAL.json",
                    {
                        **stale_terminal_payload,
                        "terminal_identity": digest_json(stale_terminal_payload),
                    },
                )
            replaced = {
                "claim_identity": existing["claim_identity"],
                "slurm_job_id": existing["slurm_job_id"],
                "slurm_restart_count": existing["slurm_restart_count"],
                "scheduler_evidence": scheduler,
            }
            prior_claim_identities.add(existing["claim_identity"])
            prior_claim_identities.update(existing.get("prior_claim_identities", []))

        payload = {
            "schema_name": LEASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "mode": binding["mode"],
            "slurm_job_id": job_id,
            "slurm_restart_count": restart,
            "execution_binding": binding,
            "claim_nonce": uuid.uuid4().hex,
            "created_unix_ns": time.time_ns(),
            "replaced_stale_owner": replaced,
            "prior_claim_identities": sorted(prior_claim_identities),
        }
        candidate = {**payload, "claim_identity": digest_json(payload)}
        if _publish_lease(lease_path, candidate):
            owner = candidate
            break
    if owner is None:
        raise StageCExecutionControlError("could not atomically acquire Stage-C lease")

    attestation_payload = {
        "schema_name": ATTESTATION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "lease_path": str(lease_path),
        "current_slurm_job_id": job_id,
        "current_slurm_restart_count": restart,
        "claim": owner,
        "restart_reconciliation": restart_reconciliation,
    }
    attestation = {
        **attestation_payload,
        "attestation_identity": digest_json(attestation_payload),
    }
    return _atomic_json(attestation_path, attestation)


def validate_attestation(
    attestation_path: Path, *, lease_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    lease_path = lease_path.resolve()
    attestation = _json(attestation_path.resolve())
    claim = _validate_claim(lease_path)
    _validate_attestation_record(
        attestation,
        expected_lease_path=lease_path,
        expected_claim=claim,
    )
    return attestation, claim


def _validate_attestation_record(
    attestation: Mapping[str, Any],
    *,
    expected_lease_path: Path,
    expected_claim: Mapping[str, Any],
) -> None:
    payload = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_identity"
    }
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
        or not isinstance(attestation.get("claim"), dict)
        or attestation.get("schema_name") != ATTESTATION_SCHEMA_NAME
        or attestation.get("schema_version") != SCHEMA_VERSION
        or Path(str(attestation.get("lease_path", ""))).resolve()
        != expected_lease_path.resolve()
        or attestation.get("attestation_identity") != digest_json(payload)
        or attestation.get("claim") != expected_claim
    ):
        raise StageCExecutionControlError(
            "lease attestation does not authenticate the expected claim"
        )


def _archive_active(
    *,
    lease_path: Path,
    attestation_path: Path,
    outcome: str,
    terminal_payload: Mapping[str, Any],
) -> dict[str, Any]:
    attestation, claim = validate_attestation(
        attestation_path, lease_path=lease_path
    )
    terminal_without_identity = {
        "schema_name": TERMINAL_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "claim_identity": claim["claim_identity"],
        "current_slurm_job_id": attestation["current_slurm_job_id"],
        "current_slurm_restart_count": attestation[
            "current_slurm_restart_count"
        ],
        "attestation_sha256": sha256_file(attestation_path),
        **dict(terminal_payload),
    }
    terminal = {
        **terminal_without_identity,
        "terminal_identity": digest_json(terminal_without_identity),
    }
    terminal_path = lease_path / "TERMINAL.json"
    archive_outcome = outcome
    if terminal_path.exists():
        existing = _validate_terminal_record(
            terminal_path, claim_identity=claim["claim_identity"]
        )
        compatible_recovery = (
            outcome == "recovered_released"
            and existing.get("outcome") == "released"
        )
        comparable_fields = set(terminal_payload) | {"claim_identity"}
        if (
            not compatible_recovery
            and existing.get("outcome") != outcome
        ) or any(
            existing.get(name) != terminal.get(name)
            for name in comparable_fields
            if name != "scheduler_evidence" or not compatible_recovery
        ):
            raise StageCExecutionControlError(
                "existing terminal record differs from requested exact archival"
            )
        terminal = existing
        archive_outcome = str(existing["outcome"])
    else:
        _atomic_json(terminal_path, terminal)
    history = lease_path.with_name(f"{lease_path.name}.history")
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"{claim['claim_identity']}.{archive_outcome}"
    try:
        rename_no_replace(lease_path, destination)
    except (AtomicPublishError, FileNotFoundError, OSError) as error:
        raise StageCExecutionControlError(
            "active lease changed during terminal archival"
        ) from error
    return {
        "outcome": archive_outcome,
        "claim_identity": claim["claim_identity"],
        "history_path": str(destination),
        "terminal_identity": terminal["terminal_identity"],
    }


def _validate_publication(
    *,
    lease_path: Path,
    completion_path: Path,
    ready_path: Path,
    attestation_lease_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    claim = _validate_claim(lease_path.resolve())
    completion = _json(completion_path.resolve())
    ready = _json(ready_path.resolve())
    completion_fields = {
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
    }
    ready_fields = {
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
    }
    publication_attestation_path = Path(
        str(completion.get("lease_attestation_path", ""))
    )
    publication_attestation = _json(publication_attestation_path.resolve())
    _validate_attestation_record(
        publication_attestation,
        expected_lease_path=attestation_lease_path or lease_path,
        expected_claim=claim,
    )
    publication_claim = publication_attestation["claim"]
    mode = claim["mode"]
    expected_step = 1 if mode == "smoke" else 72
    expected_success_status = (
        "smoke_ready" if mode == "smoke" else "pilot_complete_development_signal"
    )
    execution_claim_identity = completion.get("execution_lease_claim_identity")
    replaced = claim.get("replaced_stale_owner") or {}
    decision_path = Path(str(completion.get("decision_path", "")))
    policy_path = Path(str(completion.get("decision_policy_path", "")))
    execution_path = Path(str(completion.get("execution_complete_path", "")))
    protocol_v2 = policy_path.name == PROTOCOL_V2_RUN_R3_POLICY_NAME
    requires_smoke_prerequisite = protocol_v2 and mode == "pilot"
    if requires_smoke_prerequisite:
        completion_fields.add("prior_one_update_smoke")
        ready_fields.add("prior_one_update_smoke")
    if (
        set(completion) != completion_fields
        or set(ready) != ready_fields
        or completion.get("schema_name") != "signtrajfield_stage_c_mode_complete"
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("execution_mode") != mode
        or completion.get("source_git_head")
        != claim["execution_binding"]["source_git_head"]
        or completion.get("pair_constraint")
        != claim["execution_binding"]["pair_constraint"]
        or completion.get("active_lease_claim_identity") != claim["claim_identity"]
        or completion.get("lease_attestation_sha256")
        != sha256_file(publication_attestation_path)
        or completion.get("lease_attestation_identity")
        != publication_attestation.get("attestation_identity")
        or publication_claim != claim
        or not execution_path.is_file()
        or execution_path.is_symlink()
        or completion.get("execution_complete_sha256") != sha256_file(execution_path)
        or not decision_path.is_file()
        or decision_path.is_symlink()
        or completion.get("decision_sha256") != sha256_file(decision_path)
        or SHA256_RE.fullmatch(str(completion.get("decision_identity", ""))) is None
        or execution_claim_identity
        not in {
            claim["claim_identity"],
            replaced.get("claim_identity"),
            *claim.get("prior_claim_identities", []),
        }
        or completion.get("decision_status") not in {expected_success_status, "stop"}
        or not policy_path.is_file()
        or policy_path.is_symlink()
        or completion.get("decision_policy_sha256") != sha256_file(policy_path)
        or completion.get("expected_epoch") != 1
        or completion.get("expected_global_step_per_arm") != expected_step
        or completion.get("world_size") != 2
        or completion.get("batch_per_rank") != 64
        or completion.get("accumulation_steps") != 2
        or completion.get("effective_global_batch") != 256
        or completion.get("development_only") is not True
        or completion.get("non_authorizing") is not True
        or completion.get("promotion_eligible") is not False
        or completion.get("authorized_purpose") is not None
        or completion.get("confirmation_manifest_opened") is not False
        or completion.get("test_data_accessed") is not False
        or ready
        != {
            "schema_name": "signtrajfield_stage_c_mode_ready",
            "schema_version": SCHEMA_VERSION,
            "execution_mode": mode,
            "source_git_head": claim["execution_binding"]["source_git_head"],
            "pair_constraint": claim["execution_binding"]["pair_constraint"],
            "active_lease_claim_identity": claim["claim_identity"],
            "execution_lease_claim_identity": execution_claim_identity,
            "complete_path": str(completion_path.resolve()),
            "complete_sha256": sha256_file(completion_path),
            "decision_path": str(decision_path.resolve()),
            "decision_sha256": sha256_file(decision_path),
            "decision_identity": completion["decision_identity"],
            "decision_status": completion["decision_status"],
            "expected_epoch": 1,
            "expected_global_step_per_arm": expected_step,
            "one_optimizer_update_per_arm": mode == "smoke",
            "one_full_train_epoch_per_arm": mode == "pilot",
            "development_only": True,
            "non_authorizing": True,
            "promotion_eligible": False,
            "authorized_purpose": None,
            "confirmation_manifest_opened": False,
            "test_data_accessed": False,
            **(
                {
                    "prior_one_update_smoke": completion.get(
                        "prior_one_update_smoke"
                    )
                }
                if requires_smoke_prerequisite
                else {}
            ),
        }
    ):
        raise StageCExecutionControlError(
            "immutable Stage-C COMPLETE/READY publication is malformed"
        )
    decision = _json(decision_path)
    unsigned_decision = {
        key: value for key, value in decision.items() if key != "decision_identity"
    }
    expected_next_action = (
        "run_one_epoch_development_pilot"
        if completion["decision_status"] == "smoke_ready"
        else (
            "none_requires_fresh_preregistration_without_pilot_outcome_access"
            if completion["decision_status"]
            == "pilot_complete_development_signal"
            else "none"
        )
    )
    if (
        decision.get("decision_identity") != digest_json(unsigned_decision)
        or decision.get("decision_identity") != completion["decision_identity"]
        or decision.get("status") != completion["decision_status"]
        or decision.get("execution_mode") != mode
        or decision.get("source_git_head")
        != claim["execution_binding"]["source_git_head"]
        or decision.get("pair_constraint")
        != claim["execution_binding"]["pair_constraint"]
        or decision.get("next_permitted_action") != expected_next_action
        or decision.get("active_lease_claim_identity")
        != execution_claim_identity
        or decision.get("development_only") is not True
        or decision.get("non_authorizing") is not True
        or decision.get("promotion_eligible") is not False
        or decision.get("authorized_purpose") is not None
    ):
        raise StageCExecutionControlError("immutable Stage-C decision is malformed")
    if protocol_v2:
        execution_complete = _json(execution_path)
        launch = _json(execution_path.parent / "LAUNCH.json")
        _validate_protocol_v2_smoke_prerequisite(
            mode=mode,
            completion=completion,
            ready=ready,
            execution_complete=execution_complete,
            decision=decision,
            launch=launch,
        )
    return claim, completion, ready, publication_attestation


def validate_finalized_publication(
    *,
    lease_path: Path,
    completion_path: Path,
    ready_path: Path,
    expected_source_git_head: str,
    expected_pair_constraint: str,
) -> dict[str, Any]:
    """Validate a released immutable mode publication without an active lease."""

    active = lease_path.resolve()
    if active.exists():
        raise StageCExecutionControlError(
            "finalized-publication validation requires no active lease"
        )
    completion_hint = _json(completion_path.resolve())
    claim_identity = str(completion_hint.get("active_lease_claim_identity", ""))
    history = active.with_name(f"{active.name}.history")
    matches = [
        path
        for suffix in ("released", "recovered_released")
        for path in history.glob(f"{claim_identity}.{suffix}")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise StageCExecutionControlError(
            "finalized publication lacks one exact released lease archive"
        )
    archive = matches[0]
    claim, completion, ready, _attestation = _validate_publication(
        lease_path=archive,
        completion_path=completion_path,
        ready_path=ready_path,
        attestation_lease_path=active,
    )
    terminal = _validate_terminal_record(
        archive / "TERMINAL.json", claim_identity=claim["claim_identity"]
    )
    if (
        terminal.get("outcome") not in {"released", "recovered_released"}
        or terminal.get("completion_path") != str(completion_path.resolve())
        or terminal.get("completion_sha256") != sha256_file(completion_path)
        or terminal.get("ready_path") != str(ready_path.resolve())
        or terminal.get("ready_sha256") != sha256_file(ready_path)
        or claim["execution_binding"]["source_git_head"]
        != expected_source_git_head.lower()
        or claim["execution_binding"]["pair_constraint"]
        != expected_pair_constraint
    ):
        raise StageCExecutionControlError(
            "finalized publication terminal/source/pair binding changed"
        )
    return {
        "claim_identity": claim["claim_identity"],
        "execution_mode": claim["mode"],
        "decision_status": completion["decision_status"],
        "ready_decision_status": ready["decision_status"],
        "history_path": str(archive),
        "terminal_identity": terminal["terminal_identity"],
    }


def validate_finalized_attestation(
    attestation_path: Path, *, lease_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate an attestation against its unique released archive."""

    active = lease_path.resolve()
    if active.exists():
        raise StageCExecutionControlError(
            "finalized attestation validation requires no active lease"
        )
    attestation = _json(attestation_path.resolve())
    claim = attestation.get("claim")
    if not isinstance(claim, dict):
        raise StageCExecutionControlError("finalized attestation claim is malformed")
    claim_identity = str(claim.get("claim_identity", ""))
    history = active.with_name(f"{active.name}.history")
    matches = [
        path
        for suffix in ("released", "recovered_released")
        for path in history.glob(f"{claim_identity}.{suffix}")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(matches) != 1 or _validate_claim(matches[0]) != claim:
        raise StageCExecutionControlError(
            "finalized attestation lacks one exact released claim archive"
        )
    terminal = _validate_terminal_record(
        matches[0] / "TERMINAL.json", claim_identity=claim_identity
    )
    if terminal.get("outcome") not in {"released", "recovered_released"}:
        raise StageCExecutionControlError("finalized attestation archive is not released")
    _validate_attestation_record(
        attestation,
        expected_lease_path=active,
        expected_claim=claim,
    )
    return attestation, claim


def release_execution_lease(
    *, lease_path: Path, attestation_path: Path, completion_path: Path, ready_path: Path
) -> dict[str, Any]:
    attestation, claim = validate_attestation(
        attestation_path, lease_path=lease_path
    )
    publication_claim, completion, ready, _publication_attestation = (
        _validate_publication(
            lease_path=lease_path,
            completion_path=completion_path,
            ready_path=ready_path,
        )
    )
    if (
        completion.get("active_lease_claim_identity") != claim["claim_identity"]
        or publication_claim != claim
        or ready.get("active_lease_claim_identity") != claim["claim_identity"]
        or ready.get("complete_path") != str(completion_path.resolve())
        or ready.get("complete_sha256") != sha256_file(completion_path)
        or ready.get("source_git_head")
        != claim["execution_binding"]["source_git_head"]
        or ready.get("pair_constraint")
        != claim["execution_binding"]["pair_constraint"]
        or ready.get("execution_mode") != claim["mode"]
        or attestation["current_slurm_job_id"]
        != str(os.environ.get("SLURM_JOB_ID", ""))
        or attestation["current_slurm_restart_count"]
        != int(os.environ.get("SLURM_RESTART_COUNT", "0") or "0")
    ):
        raise StageCExecutionControlError(
            "immutable completion/READY do not bind the exact active lease claim"
        )
    return _archive_active(
        lease_path=lease_path,
        attestation_path=attestation_path,
        outcome="released",
        terminal_payload={
            "completion_path": str(completion_path.resolve()),
            "completion_sha256": sha256_file(completion_path),
            "ready_path": str(ready_path.resolve()),
            "ready_sha256": sha256_file(ready_path),
        },
    )


def reconcile_published_execution(
    *,
    lease_path: Path,
    completion_path: Path,
    ready_path: Path,
    current_slurm_job_id: str,
    current_slurm_restart_count: int,
) -> dict[str, Any]:
    """Archive a publish-before-release crash after exact scheduler proof."""

    claim, _completion, _ready, publication_attestation = _validate_publication(
        lease_path=lease_path,
        completion_path=completion_path,
        ready_path=ready_path,
    )
    current_job = str(current_slurm_job_id)
    current_restart = int(current_slurm_restart_count)
    if re.fullmatch(r"[0-9]+", current_job) is None or current_restart < 0:
        raise StageCExecutionControlError(
            "publication reconciliation requires numeric current job/restart"
        )
    if current_job == claim["slurm_job_id"]:
        scheduler = _restart_evidence(
            current_job,
            int(claim["slurm_restart_count"]),
            current_restart,
        )
    else:
        scheduler = _terminal_evidence(str(claim["slurm_job_id"]))
    publication_attestation_path = Path(
        str(_completion["lease_attestation_path"])
    )
    if publication_attestation.get("claim") != claim:
        raise StageCExecutionControlError(
            "publication attestation does not own the stale active claim"
        )
    return _archive_active(
        lease_path=lease_path,
        attestation_path=publication_attestation_path,
        outcome="recovered_released",
        terminal_payload={
            "scheduler_evidence": scheduler,
            "completion_path": str(completion_path.resolve()),
            "completion_sha256": sha256_file(completion_path),
            "ready_path": str(ready_path.resolve()),
            "ready_sha256": sha256_file(ready_path),
        },
    )


def fail_execution_lease(
    *, lease_path: Path, attestation_path: Path, exit_code: int
) -> dict[str, Any]:
    return _archive_active(
        lease_path=lease_path,
        attestation_path=attestation_path,
        outcome="failed",
        terminal_payload={"self_reported_exit_code": int(exit_code)},
    )


def _binding_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return execution_binding(
        mode=args.mode,
        source_git_head=args.source_git_head,
        source_remote_ref=args.source_remote_ref,
        source_remote_head=args.source_remote_head,
        pair_constraint=args.pair_constraint,
        memory_config=args.memory_config,
        matched_off_config=args.matched_off_config,
        cpu_gate=args.cpu_gate,
        calibration_completion=args.calibration_completion,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--lease", type=Path, required=True)
    acquire.add_argument("--attestation", type=Path, required=True)
    acquire.add_argument("--slurm_job_id", required=True)
    acquire.add_argument("--slurm_restart_count", type=int, required=True)
    acquire.add_argument("--mode", choices=tuple(sorted(MODES)), required=True)
    acquire.add_argument("--source_git_head", required=True)
    acquire.add_argument("--source_remote_ref", required=True)
    acquire.add_argument("--source_remote_head", required=True)
    acquire.add_argument("--pair_constraint", required=True)
    acquire.add_argument("--memory_config", type=Path, required=True)
    acquire.add_argument("--matched_off_config", type=Path, required=True)
    acquire.add_argument("--cpu_gate", type=Path, required=True)
    acquire.add_argument("--calibration_completion", type=Path, required=True)
    release = commands.add_parser("release")
    release.add_argument("--lease", type=Path, required=True)
    release.add_argument("--attestation", type=Path, required=True)
    release.add_argument("--completion", type=Path, required=True)
    release.add_argument("--ready", type=Path, required=True)
    failed = commands.add_parser("fail")
    failed.add_argument("--lease", type=Path, required=True)
    failed.add_argument("--attestation", type=Path, required=True)
    failed.add_argument("--exit_code", type=int, required=True)
    reconcile = commands.add_parser("reconcile-published")
    reconcile.add_argument("--lease", type=Path, required=True)
    reconcile.add_argument("--completion", type=Path, required=True)
    reconcile.add_argument("--ready", type=Path, required=True)
    reconcile.add_argument("--current_slurm_job_id", required=True)
    reconcile.add_argument("--current_slurm_restart_count", type=int, required=True)
    inspect = commands.add_parser("inspect-published")
    inspect.add_argument("--lease", type=Path, required=True)
    inspect.add_argument("--completion", type=Path, required=True)
    inspect.add_argument("--ready", type=Path, required=True)
    inspect.add_argument("--expected_source_git_head", required=True)
    inspect.add_argument("--expected_pair_constraint", required=True)
    finalized = commands.add_parser("validate-finalized")
    finalized.add_argument("--lease", type=Path, required=True)
    finalized.add_argument("--completion", type=Path, required=True)
    finalized.add_argument("--ready", type=Path, required=True)
    finalized.add_argument("--expected_source_git_head", required=True)
    finalized.add_argument("--expected_pair_constraint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "acquire":
        result = acquire_execution_lease(
            lease_path=args.lease,
            attestation_path=args.attestation,
            slurm_job_id=args.slurm_job_id,
            slurm_restart_count=args.slurm_restart_count,
            binding=_binding_from_args(args),
        )
    elif args.command == "release":
        result = release_execution_lease(
            lease_path=args.lease,
            attestation_path=args.attestation,
            completion_path=args.completion,
            ready_path=args.ready,
        )
    elif args.command == "fail":
        result = fail_execution_lease(
            lease_path=args.lease,
            attestation_path=args.attestation,
            exit_code=args.exit_code,
        )
    elif args.command == "reconcile-published":
        result = reconcile_published_execution(
            lease_path=args.lease,
            completion_path=args.completion,
            ready_path=args.ready,
            current_slurm_job_id=args.current_slurm_job_id,
            current_slurm_restart_count=args.current_slurm_restart_count,
        )
    elif args.command == "inspect-published":
        claim, completion, ready, _attestation = _validate_publication(
            lease_path=args.lease,
            completion_path=args.completion,
            ready_path=args.ready,
        )
        if (
            claim["execution_binding"]["source_git_head"]
            != args.expected_source_git_head.lower()
            or claim["execution_binding"]["pair_constraint"]
            != args.expected_pair_constraint
        ):
            raise StageCExecutionControlError(
                "published recovery belongs to another source or pair"
            )
        result = {
            "claim_identity": claim["claim_identity"],
            "execution_mode": claim["mode"],
            "completion_sha256": sha256_file(args.completion),
            "ready_sha256": sha256_file(args.ready),
            "decision_status": completion["decision_status"],
            "ready_decision_status": ready["decision_status"],
            "recovery_needed": True,
        }
    else:
        result = validate_finalized_publication(
            lease_path=args.lease,
            completion_path=args.completion,
            ready_path=args.ready,
            expected_source_git_head=args.expected_source_git_head,
            expected_pair_constraint=args.expected_pair_constraint,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
