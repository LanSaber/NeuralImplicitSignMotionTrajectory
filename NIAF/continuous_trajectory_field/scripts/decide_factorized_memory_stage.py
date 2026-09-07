"""Fail-closed ordering gate for the CSL-Daily Phase-A'' experiment.

The command never reads a test artifact or the confirmation manifest.  A
stage decision is made only after development-only memory-off and all-null
exports prove exact zero-memory behavior.  The resulting authorization is
content-addressed and can be consumed exactly as follows:

* Stage 1 ``valid_infeasible`` -> Stage 2 may start fresh from v2.
* The first stage with ``development_feasible`` -> confirmation may be opened.
* ``integrity_invalid`` -> neither action is authorized.

Confirmation access is recorded in a common exclusive marker before its
manifest is opened.  This deliberately spends the holdout even if the later
scientific analysis fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import (
    normalize_sentence_text,
    read_jsonl,
    sha256_file,
)
from NIAF.continuous_trajectory_field.validation_text_partitions import (
    SCHEMA_NAME as PARTITION_SCHEMA_NAME,
)


SCHEMA_NAME = "signtrajfield_factorized_memory_ordered_decision"
SCHEMA_VERSION = 1
AUTHORIZATION_SCHEMA_NAME = "signtrajfield_factorized_memory_authorization"
HOLDOUT_SPEND_SCHEMA_NAME = "signtrajfield_factorized_memory_confirmation_spend"
STAGE2_INPUT_SCHEMA_NAME = "signtrajfield_factorized_memory_stage2_input"
LAUNCH_INPUT_SCHEMA_NAME = "signtrajfield_factorized_memory_run_launch"
RESUME_ATTEMPT_SCHEMA_NAME = "signtrajfield_factorized_memory_resume_attempt"
EXECUTION_LEASE_SCHEMA_NAME = "signtrajfield_factorized_memory_execution_lease"
EXECUTION_LEASE_ATTESTATION_SCHEMA_NAME = (
    "signtrajfield_factorized_memory_execution_lease_attestation"
)

STAGE1 = "stage1"
STAGE2 = "stage2"
STAGES = (STAGE1, STAGE2)
DEVELOPMENT_FEASIBLE = "development_feasible"
VALID_INFEASIBLE = "valid_infeasible"
INTEGRITY_INVALID = "integrity_invalid"

EXPERIMENTS = {
    STAGE1: "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1",
    STAGE2: "csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1",
}
TEMPORAL_PRIOR_MODES = {STAGE1: "none", STAGE2: "gaussian"}
EXPECTED_PARTITION_DIGEST = (
    "80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e"
)
EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f"
)
EXPECTED_PARTITION_ARTIFACT_IDENTITY = (
    "2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "2d443adf2cd489709e19d3e55b74128e7f4470b3413692e587e9dd2ed521dabe"
)
EXPECTED_V2_SHA256 = (
    "06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54"
)
EXPECTED_BANK_ID = (
    "a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd"
)
EXPECTED_NEIGHBOR_SHA256 = {
    "train": "d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852",
    "val": "b5f5d0914e55ba9e9c79963bacb955b97d62f99f16f117381187e72c345eb199",
}
EXPECTED_EVAL_MODES = (
    "off",
    "on",
    "motion_shuffled",
    "shuffled",
    "analytic_prior",
)
EXPECTED_EVALUATION_CORRUPTION = {
    "mode": "fixed_query_condition_v1",
    "seed": 1234,
    "nonce": "csl_daily_validation_corruption_v1",
}
EXPECTED_SELECTION_AGGREGATION = "normalized_text_cluster_equal_v1"
EXPECTED_DEVELOPMENT_ROWS = 347
EXPECTED_DEVELOPMENT_TEXTS = 256
EXPECTED_CONFIRMATION_TEXTS = 540
EXPECTED_TRAIN_ROWS = 18_399
IDENTITY_NAMES = (
    "architecture",
    "behavior",
    "objective",
    "resume",
    "evaluation_control",
    "selection_aggregation",
    "validation_corruption_map",
)
SOURCE_ROOT = Path(__file__).resolve().parents[3]


class OrderedDecisionError(RuntimeError):
    """Raised when a stage or authorization fails a predeclared integrity rule."""


_TERMINAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


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


def _validate_shared_source_checkout(
    source_root: Path,
    *,
    expected_head: str,
    expected_remote_ref: str,
    expected_remote_head: str,
) -> dict[str, Any]:
    """Prove that a batch job is executing an exact clean shared clone.

    Codex worktrees in this workspace use a ``.git`` file that points into a
    node-local directory under ``/home``.  Such a checkout cannot be resolved
    reliably on a scheduler-selected node.  Full experiment jobs therefore
    require a standalone clone on the shared ``/media/cvpr`` filesystem.
    """

    source_root = source_root.resolve()
    shared_root = Path("/media/cvpr").resolve()
    dot_git = source_root / ".git"
    if (
        not source_root.is_relative_to(shared_root)
        or not dot_git.is_dir()
        or dot_git.is_symlink()
    ):
        raise OrderedDecisionError(
            "Source must be a standalone shared clone with a real .git directory"
        )
    durable_paths = {
        "durable_experiments_root": source_root / "experiments",
        "frozen_text_model_root": source_root / "deps" / "mt5-base",
    }
    resolved_durable_paths: dict[str, str] = {}
    for name, path in durable_paths.items():
        if not path.is_dir() or not path.resolve().is_relative_to(shared_root):
            raise OrderedDecisionError(
                f"Shared source checkout lacks durable {name}"
            )
        resolved_durable_paths[name] = str(path.resolve())

    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ("git", "-C", str(source_root), *arguments),
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise OrderedDecisionError(
                f"Cannot validate shared source checkout: git {' '.join(arguments)}"
            ) from error

    common_dir = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    if common_dir.resolve() != dot_git.resolve():
        raise OrderedDecisionError(
            "Source checkout uses external or node-local git metadata"
        )
    actual_head = git("rev-parse", "HEAD").lower()
    expected_head = str(expected_head).lower()
    expected_remote_head = str(expected_remote_head).lower()
    if actual_head != expected_head or expected_remote_head != expected_head:
        raise OrderedDecisionError("Source HEAD differs from its approved remote head")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise OrderedDecisionError("Source checkout is not clean")
    remote_ref = str(expected_remote_ref)
    if not remote_ref.startswith("origin/") or remote_ref == "origin/":
        raise OrderedDecisionError("Source remote ref is not an origin branch")
    branch = remote_ref.removeprefix("origin/")
    live_rows = [
        line.split()
        for line in git("ls-remote", "--heads", "origin", f"refs/heads/{branch}").splitlines()
        if line.strip()
    ]
    if len(live_rows) != 1 or live_rows[0][0].lower() != expected_head:
        raise OrderedDecisionError("Live origin branch differs from the approved source")
    return {
        "repository_root": str(source_root),
        "git_directory": str(dot_git.resolve()),
        "git_head": actual_head,
        "remote_ref": remote_ref,
        "remote_head": live_rows[0][0].lower(),
        "standalone_shared_clone_checked": True,
        "worktree_clean_checked": True,
        "remote_ref_exact_match_checked": True,
    } | resolved_durable_paths


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OrderedDecisionError(f"Required file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OrderedDecisionError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise OrderedDecisionError(f"Expected a JSON object: {path}")
    return value


def _atomic_json_record(
    path: Path, payload: Mapping[str, Any], *, allow_matching_existing: bool = False
) -> dict[str, Any]:
    """Publish one durable JSON record without silently replacing evidence."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hard link publishes the fully fsynced inode with
            # no-replace semantics.  This avoids path.exists()+rename races on
            # the shared experiment filesystem.
            os.link(temporary, path)
        except FileExistsError as error:
            if allow_matching_existing and _json(path) == dict(payload):
                pass
            else:
                raise OrderedDecisionError(
                    f"Refusing to replace immutable record: {path}"
                ) from error
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    if temporary.exists():
        temporary.unlink()
    return dict(payload)


def _validated_execution_lease(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise OrderedDecisionError(f"Active execution lease is not a directory: {path}")
    payload = _json(path / "owner.json")
    ready = _json(path / "READY")
    identity = payload.get("claim_identity")
    without_identity = {
        key: value for key, value in payload.items() if key != "claim_identity"
    }
    if (
        payload.get("schema_name") != EXECUTION_LEASE_SCHEMA_NAME
        or int(payload.get("schema_version", -1)) != SCHEMA_VERSION
        or payload.get("purpose")
        not in {"training", "confirmation", "decision", "diagnostic"}
        or payload.get("stage") not in STAGES
        or payload.get("experiment_name") != EXPERIMENTS[payload["stage"]]
        or not re.fullmatch(r"[0-9]+", str(payload.get("slurm_job_id", "")))
        or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            str(payload.get("source_git_head", "")).lower(),
        )
        or not isinstance(payload.get("claim_nonce"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("binding_identity", ""))
        )
        or _digest_json(without_identity) != identity
        or ready
        != {
            "schema_name": EXECUTION_LEASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "claim_identity": identity,
        }
    ):
        raise OrderedDecisionError(f"Active execution lease is malformed: {path}")
    return payload


def _slurm_job_terminal_evidence(job_id: str) -> dict[str, Any]:
    """Return scheduler proof that an earlier lease owner cannot still write."""

    try:
        result = subprocess.run(
            ("scontrol", "show", "job", "-o", str(job_id)),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OrderedDecisionError(
            f"Cannot query Slurm state for lease owner {job_id}"
        ) from error
    combined = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        if "invalid job id" in combined.lower():
            return {
                "job_id": str(job_id),
                "classification": "not_found",
                "state": "NOT_FOUND",
                "scontrol_returncode": int(result.returncode),
            }
        raise OrderedDecisionError(
            f"Slurm did not prove lease owner {job_id} terminal: {combined}"
        )
    match = re.search(r"(?:^|\s)JobState=([A-Z_]+)", result.stdout)
    if match is None:
        raise OrderedDecisionError(
            f"Slurm returned no JobState for lease owner {job_id}"
        )
    state = match.group(1)
    if state not in _TERMINAL_SLURM_STATES:
        raise OrderedDecisionError(
            f"Execution lease is owned by live Slurm job {job_id} ({state})"
        )
    return {
        "job_id": str(job_id),
        "classification": "terminal",
        "state": state,
        "scontrol_returncode": 0,
    }


def _publish_execution_lease_directory(
    lease_path: Path, owner: Mapping[str, Any]
) -> bool:
    """Atomically publish a complete, non-empty lease directory.

    POSIX rename cannot replace a non-empty directory.  Consequently two
    compute nodes racing this operation cannot clobber or nest either claim;
    exactly one rename succeeds.  The owner and READY records are fully written
    in the private sibling directory before publication.
    """

    building = lease_path.with_name(
        f".{lease_path.name}.claiming.{owner['slurm_job_id']}.{uuid.uuid4().hex}"
    )
    building.mkdir(parents=False, exist_ok=False)
    try:
        _atomic_json_record(building / "owner.json", owner)
        _atomic_json_record(
            building / "READY",
            {
                "schema_name": EXECUTION_LEASE_SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "claim_identity": owner["claim_identity"],
            },
        )
        try:
            os.rename(building, lease_path)
        except (FileExistsError, OSError):
            # A concurrent winner leaves a complete non-empty destination.
            # Never use shutil.move here: it would nest this claim.
            return False
        return True
    finally:
        if building.exists():
            shutil.rmtree(building)


def acquire_execution_lease(
    *,
    lease_path: Path,
    attestation_path: Path,
    purpose: str,
    stage: str,
    source_git_head: str,
    slurm_job_id: str,
    binding_identity: str,
) -> dict[str, Any]:
    """Acquire a singleton lease, replacing only a scheduler-proven stale owner."""

    if purpose not in {"training", "confirmation", "decision", "diagnostic"}:
        raise OrderedDecisionError("Unknown execution-lease purpose")
    if stage not in STAGES:
        raise OrderedDecisionError("Unknown execution-lease stage")
    source_git_head = str(source_git_head).lower()
    slurm_job_id = str(slurm_job_id)
    binding_identity = str(binding_identity).lower()
    if not re.fullmatch(r"[0-9]+", slurm_job_id):
        raise OrderedDecisionError("Execution lease requires a numeric Slurm job ID")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_git_head):
        raise OrderedDecisionError("Execution lease source identity is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", binding_identity):
        raise OrderedDecisionError("Execution lease binding identity is malformed")
    lease_path = lease_path.resolve()
    attestation_path = attestation_path.resolve()
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    replaced: dict[str, Any] | None = None
    for _attempt in range(3):
        if lease_path.exists():
            existing = _validated_execution_lease(lease_path)
            same_owner = (
                existing.get("purpose") == purpose
                and existing.get("stage") == stage
                and existing.get("source_git_head") == source_git_head
                and existing.get("slurm_job_id") == slurm_job_id
                and existing.get("binding_identity") == binding_identity
            )
            if same_owner:
                owner = existing
                break
            scheduler_evidence = _slurm_job_terminal_evidence(
                str(existing["slurm_job_id"])
            )
            history = lease_path.with_name(f"{lease_path.name}.history")
            history.mkdir(parents=True, exist_ok=True)
            stale_path = history / (
                f"{existing['slurm_job_id']}.{existing['claim_identity']}.stale"
            )
            try:
                # The destination is claim-specific and non-empty after the
                # first winner.  A delayed contender therefore cannot rename a
                # newly published lease into the old claim's history path.
                os.rename(lease_path, stale_path)
            except (FileNotFoundError, FileExistsError, OSError):
                continue
            replaced = {
                "claim_identity": existing["claim_identity"],
                "slurm_job_id": existing["slurm_job_id"],
                "scheduler_evidence": scheduler_evidence,
            }

        without_identity = {
            "schema_name": EXECUTION_LEASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "purpose": purpose,
            "stage": stage,
            "experiment_name": EXPERIMENTS[stage],
            "slurm_job_id": slurm_job_id,
            "slurm_restart_count": int(
                os.environ.get("SLURM_RESTART_COUNT", "0") or "0"
            ),
            "source_git_head": source_git_head,
            "binding_identity": binding_identity,
            "claim_nonce": uuid.uuid4().hex,
            "created_unix_ns": time.time_ns(),
            "replaced_stale_owner": replaced,
        }
        candidate = {
            **without_identity,
            "claim_identity": _digest_json(without_identity),
        }
        if _publish_execution_lease_directory(lease_path, candidate):
            owner = candidate
            break
    else:
        raise OrderedDecisionError(
            f"Could not acquire execution lease without a concurrent race: {lease_path}"
        )

    attestation_without_identity = {
        "schema_name": EXECUTION_LEASE_ATTESTATION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "lease_path": str(lease_path),
        "claim": owner,
    }
    attestation = {
        **attestation_without_identity,
        "attestation_identity": _digest_json(attestation_without_identity),
    }
    return _atomic_json_record(
        attestation_path, attestation, allow_matching_existing=True
    )


def release_execution_lease(
    *, lease_path: Path, attestation_path: Path
) -> dict[str, Any]:
    """Release only the exact claim recorded by this successful attempt."""

    lease_path = lease_path.resolve()
    _attestation, claim = _validate_execution_lease_attestation(
        attestation_path, lease_path=lease_path
    )
    active = claim
    history = lease_path.with_name(f"{lease_path.name}.history")
    history.mkdir(parents=True, exist_ok=True)
    released = history / f"{active['claim_identity']}.released"
    if released.exists():
        raise OrderedDecisionError("Execution lease was already released")
    try:
        os.rename(lease_path, released)
    except (FileNotFoundError, FileExistsError, OSError) as error:
        raise OrderedDecisionError(
            "Execution lease changed concurrently during release"
        ) from error
    return {
        "released": True,
        "claim_identity": claim["claim_identity"],
        "lease_path": str(lease_path),
    }


def _validate_execution_lease_attestation(
    attestation_path: Path, *, lease_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    attestation = _json(attestation_path.resolve())
    without_identity = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_identity"
    }
    claim = dict(attestation.get("claim", {}) or {})
    if (
        attestation.get("schema_name")
        != EXECUTION_LEASE_ATTESTATION_SCHEMA_NAME
        or _digest_json(without_identity) != attestation.get("attestation_identity")
        or Path(str(attestation.get("lease_path", ""))).resolve() != lease_path
    ):
        raise OrderedDecisionError("Execution-lease attestation is invalid")
    if not lease_path.exists() or _validated_execution_lease(lease_path) != claim:
        raise OrderedDecisionError(
            "Execution-lease attestation is not the active attempt"
        )
    return attestation, claim


def _finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise OrderedDecisionError(f"Non-finite numeric value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, path=f"{path}[{index}]")


def _resolve_existing(value: Any, fallback_dir: Path) -> Path:
    path = Path(str(value))
    candidates = (path,) if path.is_absolute() else (Path.cwd() / path, fallback_dir / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise OrderedDecisionError(f"Referenced file does not exist: {value}")


def _validate_partition(partition_dir: Path) -> dict[str, Any]:
    """Validate only the sealed development view of the partition.

    In particular, this gate must not read ``partition.json`` because that
    full artifact includes confirmation assignments.  Full-partition checks
    belong to the irreversibly spent confirmation job.
    """

    partition_dir = partition_dir.resolve()
    ready = _json(partition_dir / "READY")
    if ready != {
        "schema_name": PARTITION_SCHEMA_NAME,
        "schema_version": 1,
        "artifact_identity": EXPECTED_PARTITION_ARTIFACT_IDENTITY,
    }:
        raise OrderedDecisionError("Sealed partition READY identity changed")
    development_manifest = partition_dir / "manifest_development.jsonl"
    if sha256_file(development_manifest) != EXPECTED_DEVELOPMENT_MANIFEST_SHA256:
        raise OrderedDecisionError("Development manifest hash mismatch")
    rows = read_jsonl(development_manifest)
    if len(rows) != EXPECTED_DEVELOPMENT_ROWS:
        raise OrderedDecisionError("Development manifest row count changed")
    texts = [normalize_sentence_text(row.get("text", "")) for row in rows]
    if len(set(texts)) != EXPECTED_DEVELOPMENT_TEXTS:
        raise OrderedDecisionError("Development manifest text-cluster count changed")
    if any(str(row.get("source_split", "")) != "val" for row in rows):
        raise OrderedDecisionError("Development manifest contains a non-validation row")
    # Intentionally do not open manifest_confirmation.jsonl here.
    return {
        "dir": partition_dir,
        "development_manifest": development_manifest.resolve(),
        "development_manifest_sha256": EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "rows": rows,
    }


def _validate_config(stage: str, config_path: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise OrderedDecisionError(f"Unknown stage: {stage}")
    config_path = config_path.resolve()
    if config_path.name != f"{EXPERIMENTS[stage]}.yaml":
        raise OrderedDecisionError("Stage config filename is not the approved full config")
    cfg = load_config(config_path)
    if cfg.get("experiment_name") != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Stage config has the wrong experiment name")
    memory = dict(cfg.get("sentence_memory", {}) or {})
    checks = {
        "key_value_mode": "factorized_metadata_motion_v1",
        "temporal_prior_mode": TEMPORAL_PRIOR_MODES[stage],
    }
    for name, expected in checks.items():
        if memory.get(name) != expected:
            raise OrderedDecisionError(f"sentence_memory.{name} is not {expected!r}")
    scalars = {
        "k": 8,
        "top_m": 64,
        "duration_weight": 0.05,
        "score_temperature": 0.10,
        "candidate_dropout_probability": 0.10,
        "temporal_prior_sigma": 0.25,
        "temporal_prior_scale": 1.0,
    }
    for name, expected in scalars.items():
        if float(memory.get(name, math.nan)) != float(expected):
            raise OrderedDecisionError(f"sentence_memory.{name} changed")
    train = dict(cfg.get("train", {}) or {})
    if (
        int(train.get("epochs", -1)) != 4
        or int(train.get("early_stopping_patience", -1)) != 2
        or int(train.get("early_stopping_min_epochs", -1)) != 2
        or int(train.get("batch_size", -1)) != 32
        or int(train.get("accumulation_steps", -1)) != 2
        or int(train.get("max_samples_per_memory_batch", -1)) != 8
        or int(train.get("max_frames_per_memory_batch", -1)) != 2048
        or int(train.get("max_train_batches", -1)) != 0
    ):
        raise OrderedDecisionError("Stage config does not match the approved training budget")
    if train.get("freeze_base") is not True or train.get("unfreeze_base_prefixes"):
        raise OrderedDecisionError("Only frozen-base Phase A is permitted")
    if int(cfg.get("seed", -1)) != 1234:
        raise OrderedDecisionError("Stage config seed changed")
    if list(dict(cfg.get("eval", {}) or {}).get("sentence_memory_modes", [])) != list(
        EXPECTED_EVAL_MODES
    ):
        raise OrderedDecisionError("Stage config must contain the fixed five-mode order")
    if int(dict(cfg.get("eval", {}) or {}).get("max_batches", -1)) != 0:
        raise OrderedDecisionError("Full development validation cannot be batch-capped")
    if dict(dict(cfg.get("eval", {}) or {}).get("evaluation_corruption", {}) or {}) != (
        EXPECTED_EVALUATION_CORRUPTION
    ):
        raise OrderedDecisionError("Fixed validation-corruption control changed")
    if dict(cfg.get("selection", {}) or {}).get("aggregation") != (
        EXPECTED_SELECTION_AGGREGATION
    ):
        raise OrderedDecisionError("Cluster-equal selection aggregation changed")
    if bool(dict(cfg.get("sentence_memory_safety", {}) or {}).get("phase_b", {}).get("enabled")):
        raise OrderedDecisionError("Phase B is forbidden")
    paired = dict(
        dict(cfg.get("sentence_memory_safety", {}) or {}).get(
            "paired_corruption", {}
        )
        or {}
    )
    expected_paired = {
        "enabled": True,
        "full_shuffle_probability": 0.10,
        "benefit_margin_relative": 0.001,
        "ranking_margin_relative": 0.005,
        "detach_corrupt_ranking": True,
        "fallback_huber_beta": 0.10,
    }
    if paired != expected_paired:
        raise OrderedDecisionError("Paired corruption objective controls changed")
    objective = dict(cfg.get("objective", {}) or {})
    expected_objective_weights = {
        "lambda_sentence_benefit": 1.0,
        "lambda_sentence_motion_rank": 1.0,
        "lambda_sentence_motion_fallback": 1.0,
        "lambda_sentence_full_shuffle_rank": 1.0,
        "lambda_sentence_full_shuffle_fallback": 1.0,
        "lambda_sentence_sparsity": 1e-4,
    }
    if any(
        float(objective.get(name, math.nan)) != expected
        for name, expected in expected_objective_weights.items()
    ):
        raise OrderedDecisionError("Motion-contrast objective weights changed")
    expected_output = Path(str(dict(cfg.get("output", {}) or {}).get("out_dir", ""))).name
    if expected_output != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Stage output directory is not the full-run directory")
    base_path = config_path.parent / (
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml"
    )
    expected_cfg = load_config(base_path)
    expected_cfg["experiment_name"] = EXPERIMENTS[stage]
    expected_cfg.setdefault("sentence_memory", {}).update(
        {
            "key_value_mode": "factorized_metadata_motion_v1",
            "temporal_prior_mode": TEMPORAL_PRIOR_MODES[stage],
            "temporal_prior_sigma": 0.25,
            "temporal_prior_scale": 1.0,
        }
    )
    expected_cfg.setdefault("eval", {}).update(
        {
            "sentence_memory_word_prior_mode": "off",
            "sentence_memory_modes": list(EXPECTED_EVAL_MODES),
            "export_sentence_memory_mode": "on",
            "evaluation_corruption": dict(EXPECTED_EVALUATION_CORRUPTION),
        }
    )
    expected_cfg.setdefault("selection", {})[
        "aggregation"
    ] = EXPECTED_SELECTION_AGGREGATION
    expected_cfg.setdefault("validation_text_partition", {})[
        "expected_development_manifest_sha256"
    ] = EXPECTED_DEVELOPMENT_MANIFEST_SHA256
    expected_cfg["validation_text_partition"][
        "expected_partition_artifact_identity"
    ] = EXPECTED_PARTITION_ARTIFACT_IDENTITY
    expected_cfg.setdefault("output", {})["out_dir"] = (
        "experiments/NIAF/continuous_trajectory_field/" + EXPERIMENTS[stage]
    )
    if cfg != expected_cfg:
        changed_sections = sorted(
            key
            for key in set(cfg) | set(expected_cfg)
            if cfg.get(key) != expected_cfg.get(key)
        )
        raise OrderedDecisionError(
            "Full-stage leaf config differs outside the approved architecture "
            f"and controls; changed sections={changed_sections}"
        )
    return cfg


def _resolved_config_projection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Strip only trainer-added provenance and normalized operational paths."""

    payload = json.loads(json.dumps(cfg))
    payload.pop("device", None)
    payload.pop("validation_text_partition", None)
    output = payload.get("output")
    if isinstance(output, dict):
        output["out_dir"] = Path(str(output.get("out_dir", ""))).name
    memory = payload.get("sentence_memory")
    if isinstance(memory, dict):
        memory.pop("resolved_identity", None)
        memory.pop("resolved_behavior_identity", None)
    safety = payload.get("sentence_memory_safety")
    if isinstance(safety, dict):
        safety.pop("v2_to_v3_text_only_parity", None)
    return payload


def _validate_resolved_config(
    *, stage: str, resolved: Mapping[str, Any], approved: Mapping[str, Any]
) -> None:
    """Reject any behavior-affecting CLI/runtime departure from the leaf config."""

    if _resolved_config_projection(resolved) != _resolved_config_projection(approved):
        runtime = _resolved_config_projection(resolved)
        expected = _resolved_config_projection(approved)
        changed_sections = sorted(
            key
            for key in set(runtime) | set(expected)
            if runtime.get(key) != expected.get(key)
        )
        raise OrderedDecisionError(
            "Resolved runtime config differs from the approved full-stage config; "
            f"changed sections={changed_sections}"
        )
    if str(resolved.get("device", "")).lower() != "cuda":
        raise OrderedDecisionError("Full-stage resolved device is not CUDA")
    if Path(
        str(dict(resolved.get("output", {}) or {}).get("out_dir", ""))
    ).name != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Resolved output directory has the wrong stage")
    partition = dict(resolved.get("validation_text_partition", {}) or {})
    expected_development_manifest_identity = {
        "file": "manifest_development.jsonl",
        "sha256": EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "row_count": EXPECTED_DEVELOPMENT_ROWS,
        "unique_text_count": EXPECTED_DEVELOPMENT_TEXTS,
    }
    development_runtime_payload = {
        "schema_name": "signtrajfield_development_validation_runtime",
        "schema_version": 1,
        "validation_only": True,
        "partition": {
            "digest": EXPECTED_PARTITION_DIGEST,
            "artifact_identity": EXPECTED_PARTITION_ARTIFACT_IDENTITY,
        },
        "source_validation_manifest": {
            "sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "row_count": 1_077,
        },
        "development_manifest": expected_development_manifest_identity,
        "confirmation_counts": {"row_count": 728, "unique_text_count": 540},
        "bank_id": EXPECTED_BANK_ID,
        "retrieval_query_mode": "exact_name_indexed_full_val_table_subset_v1",
        "holdout_access": "development_manifest_only_v1",
    }
    required_partition = {
        "partition_digest": EXPECTED_PARTITION_DIGEST,
        "partition_artifact_identity": EXPECTED_PARTITION_ARTIFACT_IDENTITY,
        "expected_partition_digest": EXPECTED_PARTITION_DIGEST,
        "expected_partition_artifact_identity": EXPECTED_PARTITION_ARTIFACT_IDENTITY,
        "expected_validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
        "expected_development_manifest_sha256": EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "expected_validation_rows": 1_077,
        "expected_development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "expected_confirmation_rows": 728,
        "expected_novel_text_count": 796,
        "development_text_count": EXPECTED_DEVELOPMENT_TEXTS,
        "expected_bank_id": EXPECTED_BANK_ID,
        "development_row_count": EXPECTED_DEVELOPMENT_ROWS,
        "confirmation_row_count": 728,
        "confirmation_text_count": 540,
        "development_manifest_identity": expected_development_manifest_identity,
        "development_runtime_identity": {
            "schema_name": "signtrajfield_development_validation_runtime",
            "schema_version": 1,
            "digest": _digest_json(development_runtime_payload),
        },
        "retrieval_query_mode": "exact_name_indexed_full_val_table_subset_v1",
        "confirmation_evaluated_during_training": False,
        "exact_seen_evaluated_during_training": False,
    }
    if any(partition.get(key) != value for key, value in required_partition.items()):
        raise OrderedDecisionError("Resolved validation-partition controls changed")
    forbidden_partition_fields = {
        "resolved_artifact",
        "assignments",
        "confirmation_manifest",
        "confirmation_normalized_texts",
        "development_normalized_texts",
        "partition_dir",
    }
    if forbidden_partition_fields.intersection(partition):
        raise OrderedDecisionError(
            "Resolved training config embeds validation membership or holdout paths"
        )

    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    expected_selection = trainer.sentence_memory_selection_aggregation_identity(
        approved
    )
    expected_map = trainer.sentence_memory_validation_corruption_map_identity(
        approved,
        partition_digest=EXPECTED_PARTITION_DIGEST,
        bank_id=EXPECTED_BANK_ID,
    )
    if partition.get("selection_aggregation_identity") != expected_selection:
        raise OrderedDecisionError(
            "Resolved partition has the wrong selection-aggregation identity"
        )
    if partition.get("evaluation_corruption_map_identity") != expected_map:
        raise OrderedDecisionError(
            "Resolved partition has the wrong validation-corruption map identity"
        )
    memory = dict(resolved.get("sentence_memory", {}) or {})
    if memory.get("resolved_behavior_identity") != (
        trainer.sentence_memory_behavior_identity(approved)
    ):
        raise OrderedDecisionError("Resolved behavior identity changed")
    resolved_memory = dict(memory.get("resolved_identity", {}) or {})
    if resolved_memory.get("bank_id") != EXPECTED_BANK_ID:
        raise OrderedDecisionError("Resolved memory bank identity changed")
    parity = dict(
        dict(resolved.get("sentence_memory_safety", {}) or {}).get(
            "v2_to_v3_text_only_parity", {}
        )
        or {}
    )
    if (
        parity.get("passed") is not True
        or float(parity.get("prediction_max_abs", math.inf)) > 1e-7
        or float(parity.get("duration_max_abs", math.inf)) > 1e-7
    ):
        raise OrderedDecisionError("Resolved config lacks strict v2 parity")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise OrderedDecisionError(
                f"Invalid metrics JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise OrderedDecisionError(f"Metrics line {line_number} is not an object")
        _finite_tree(row, path=f"metrics[{line_number}]")
        rows.append(row)
    return rows


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise OrderedDecisionError("Selected checkpoint has no model state dictionary")
    for name, tensor in checkpoint["model"].items():
        if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
            raise OrderedDecisionError(f"Checkpoint tensor is non-finite: {name}")
    return checkpoint


def _identity_digest(value: Any, name: str) -> str:
    if not isinstance(value, Mapping):
        raise OrderedDecisionError(f"Checkpoint lacks {name} identity")
    digest = value.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise OrderedDecisionError(f"Checkpoint has malformed {name} identity")
    return digest


def _checkpoint_identity_evidence(
    checkpoint: Mapping[str, Any],
    *,
    stage: str,
    approved_cfg: Mapping[str, Any],
    resolved_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_cfg = dict(checkpoint.get("config", {}) or {})
    if checkpoint_cfg != dict(resolved_cfg):
        raise OrderedDecisionError(
            "Checkpoint config is not exactly the terminal resolved config"
        )
    if checkpoint_cfg.get("experiment_name") != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Selected checkpoint came from another experiment")
    memory = dict(checkpoint_cfg.get("sentence_memory", {}) or {})
    if memory.get("key_value_mode") != "factorized_metadata_motion_v1" or memory.get(
        "temporal_prior_mode"
    ) != TEMPORAL_PRIOR_MODES[stage]:
        raise OrderedDecisionError("Checkpoint architecture differs from its stage")
    if int(checkpoint.get("epoch", -1)) not in range(1, 5):
        raise OrderedDecisionError("Selected checkpoint epoch is outside [1, 4]")
    parity = dict(checkpoint.get("v2_to_v3_text_only_parity", {}) or {})
    if (
        parity.get("passed") is not True
        or float(parity.get("prediction_max_abs", math.inf)) > 1e-7
        or float(parity.get("duration_max_abs", math.inf)) > 1e-7
    ):
        raise OrderedDecisionError("Checkpoint lacks strict v2 memory-off parity")
    partition_cfg = dict(checkpoint_cfg.get("validation_text_partition", {}) or {})
    if partition_cfg.get("partition_digest") != EXPECTED_PARTITION_DIGEST:
        raise OrderedDecisionError("Checkpoint is not bound to the locked partition")
    if partition_cfg.get("confirmation_evaluated_during_training") is not False:
        raise OrderedDecisionError("Checkpoint does not attest confirmation isolation")
    rng = dict(checkpoint.get("rng_state", {}) or {})
    rank_states = list(rng.get("rank_states", []) or [])
    if int(rng.get("world_size", -1)) != 4 or sorted(
        int(row.get("rank", -1)) for row in rank_states if isinstance(row, Mapping)
    ) != [0, 1, 2, 3]:
        raise OrderedDecisionError("Checkpoint lacks exact four-rank RNG state")

    identities: dict[str, Any] = {}
    for name in IDENTITY_NAMES:
        key = f"sentence_memory_{name}_identity"
        value = checkpoint.get(key)
        identities[name] = {"digest": _identity_digest(value, name), "value": value}

    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    approved_identity_functions = {
        "architecture": trainer.sentence_memory_architecture_identity,
        "behavior": trainer.sentence_memory_behavior_identity,
        "objective": trainer.sentence_memory_objective_identity,
        "evaluation_control": trainer.sentence_memory_evaluation_control_identity,
        "selection_aggregation": trainer.sentence_memory_selection_aggregation_identity,
    }
    for name, function in approved_identity_functions.items():
        expected = function(approved_cfg)
        if identities[name]["value"] != expected:
            raise OrderedDecisionError(
                f"Checkpoint {name} identity differs from the approved config"
            )
    expected_resume = trainer.sentence_memory_resume_identity(resolved_cfg)
    if identities["resume"]["value"] != expected_resume:
        raise OrderedDecisionError(
            "Checkpoint resume identity differs from its exact resolved config"
        )
    expected_map = trainer.sentence_memory_validation_corruption_map_identity(
        approved_cfg,
        partition_digest=EXPECTED_PARTITION_DIGEST,
        bank_id=EXPECTED_BANK_ID,
    )
    if identities["validation_corruption_map"]["value"] != expected_map:
        raise OrderedDecisionError(
            "Checkpoint validation-corruption map identity does not recompute"
        )

    memory_identity = dict(checkpoint.get("sentence_memory_identity", {}) or {})
    if memory_identity.get("bank_id") != EXPECTED_BANK_ID:
        raise OrderedDecisionError("Checkpoint bank identity changed")
    tables = dict(memory_identity.get("neighbor_tables", {}) or {})
    if set(tables) != {"train", "val"}:
        raise OrderedDecisionError("Checkpoint must contain only train/val neighbors")
    table_hashes = {
        split: str(dict(tables.get(split, {}) or {}).get("sha256", ""))
        for split in ("train", "val")
    }
    if table_hashes != EXPECTED_NEIGHBOR_SHA256:
        raise OrderedDecisionError("Checkpoint neighbor-table identity changed")
    return {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint.get("global_step", -1)),
        "parity": parity,
        "identities": identities,
        "bank_id": EXPECTED_BANK_ID,
        "neighbor_table_sha256": table_hashes,
    }


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = float(metrics.get(name, math.nan))
    if not math.isfinite(value):
        raise OrderedDecisionError(f"Required selection metric is missing: {name}")
    return value


def independently_feasible(metrics: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    correct = _metric(metrics, "selection_sentence_memory_score")
    baselines = {
        "text_only": _metric(metrics, "selection_text_only_score"),
        "motion_shuffled": _metric(
            metrics, "selection_motion_shuffled_sentence_memory_score"
        ),
        "full_shuffled": _metric(metrics, "selection_shuffled_sentence_memory_score"),
    }
    improvements = {
        name: (baseline - correct) / max(abs(baseline), 1e-12)
        for name, baseline in baselines.items()
    }
    rmotion = _metric(metrics, "selection_Rmotion")
    hand_degradation = {}
    for hand in ("lhand", "rhand"):
        off = _metric(metrics, f"val_text_only/pred_loss_path_{hand}")
        on = _metric(metrics, f"val_sentence_memory/pred_loss_path_{hand}")
        hand_degradation[hand] = (on - off) / max(abs(off), 1e-12)
    feasible = bool(
        all(value >= 0.001 for value in improvements.values())
        and rmotion >= 0.05
        and all(value <= 0.02 for value in hand_degradation.values())
    )
    return feasible, {
        "correct_score": correct,
        "comparator_scores": baselines,
        "relative_improvements": improvements,
        "Rmotion": rmotion,
        "hand_path_relative_degradation": hand_degradation,
        "passed": feasible,
    }


def _replay_selection_history(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay the trainer's strict lexicographic selection and patience state."""

    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    early = trainer.initial_early_stopping_state()
    best_feasible_score = math.inf
    best_feasible_row: dict[str, Any] | None = None
    best_infeasible_score = math.inf
    best_infeasible_key: tuple[float, float] | None = None
    best_infeasible_row: dict[str, Any] | None = None
    selection_states: dict[int, dict[str, Any]] = {}
    gates: list[tuple[bool, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if bool(early.get("stopped", False)):
            raise OrderedDecisionError("Metrics continue after replayed early stopping")
        epoch = int(row.get("epoch", -1))
        feasible, details = independently_feasible(row)
        gates.append((feasible, details))
        score = _metric(row, "selection_score")
        violation = _metric(row, "selection_constraint_violation")
        early, improved, stopped = trainer.update_early_stopping_state(
            early,
            feasible=feasible,
            normalized_constraint_violation=violation,
            selection_score_value=score,
            epoch=epoch,
            patience=2,
            minimum_epoch=2,
        )
        expected_row_state = {
            "early_stopping_improved": float(improved),
            "early_stopping_bad_validation_count": float(
                early["bad_validation_count"]
            ),
            "early_stopping_validation_count": float(early["validation_count"]),
            "early_stopping_requested": float(stopped),
        }
        if any(float(row.get(name, math.nan)) != value for name, value in expected_row_state.items()):
            raise OrderedDecisionError(
                f"Epoch {epoch} early-stop diagnostics disagree with exact replay"
            )
        if feasible and score < best_feasible_score:
            best_feasible_score = score
            best_feasible_row = row
        elif not feasible:
            key = (violation, score)
            if best_infeasible_key is None or key < best_infeasible_key:
                best_infeasible_key = key
                best_infeasible_score = score
                best_infeasible_row = row
        selection_states[epoch] = trainer.checkpoint_selection_state(
            best_feasible_score,
            best_infeasible_score,
            early_stopping_state=early,
            best_infeasible_key=best_infeasible_key,
        )
        if stopped and index != len(rows) - 1:
            raise OrderedDecisionError("Metrics continue after replayed early stopping")
    return {
        "gates": gates,
        "best_feasible_row": best_feasible_row,
        "best_infeasible_row": best_infeasible_row,
        "best_feasible_score": (
            None if not math.isfinite(best_feasible_score) else best_feasible_score
        ),
        "best_infeasible_score": (
            None if not math.isfinite(best_infeasible_score) else best_infeasible_score
        ),
        "best_infeasible_key": (
            None if best_infeasible_key is None else list(best_infeasible_key)
        ),
        "early_stopping": early,
        "selection_states": selection_states,
    }


def _validate_replayed_winner(
    checkpoint: Mapping[str, Any], replay: Mapping[str, Any], *, has_feasible: bool
) -> dict[str, Any]:
    selected_row = (
        replay.get("best_feasible_row")
        if has_feasible
        else replay.get("best_infeasible_row")
    )
    if not isinstance(selected_row, Mapping) or dict(
        checkpoint.get("metrics", {}) or {}
    ) != dict(selected_row):
        raise OrderedDecisionError(
            "Selected checkpoint is not the exact replayed lexicographic winner"
        )
    selected_epoch = int(selected_row.get("epoch", -1))
    selection_states = dict(replay.get("selection_states", {}) or {})
    if int(checkpoint.get("epoch", -1)) != selected_epoch or dict(
        checkpoint.get("selection_state", {}) or {}
    ) != selection_states.get(selected_epoch):
        raise OrderedDecisionError(
            "Selected checkpoint epoch/state differs from the replayed winner"
        )
    return dict(selected_row)


def _selection_inputs_from_persisted_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore the unprefixed validation namespace consumed by selection."""

    payload = {
        str(name)[len("val_") :]: value
        for name, value in row.items()
        if str(name).startswith("val_")
    }
    if not payload or any(name.startswith("val_") for name in payload):
        raise OrderedDecisionError(
            "Persisted epoch row lacks canonical val_-prefixed selection inputs"
        )
    return payload


def _validate_run(
    *, stage: str, run_dir: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = run_dir.resolve()
    if run_dir.name != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Run directory has the wrong stage name")
    cfg = _validate_config(stage, config_path)
    resolved_path = run_dir / "config.resolved.json"
    resolved = _json(resolved_path)
    if resolved.get("experiment_name") != EXPERIMENTS[stage]:
        raise OrderedDecisionError("Resolved run config has the wrong experiment")
    _validate_resolved_config(stage=stage, resolved=resolved, approved=cfg)
    summary = _json(run_dir / "selection_summary.json")
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise OrderedDecisionError("Run has no terminal metrics.jsonl")
    rows = _read_metrics(metrics_path)
    completed = [
        row
        for row in rows
        if float(row.get("validation_pending", 1.0)) == 0.0
        and "selection_feasible" in row
    ]
    if len(completed) != len(rows) or not 2 <= len(completed) <= 4:
        raise OrderedDecisionError("Run lacks two-to-four complete validation events")
    epochs = [int(row.get("epoch", -1)) for row in completed]
    if epochs != list(range(1, max(epochs) + 1)):
        raise OrderedDecisionError("Run epoch history is incomplete or noncanonical")
    namespaces = (
        "val_text_only/",
        "val_sentence_memory/",
        "val_motion_shuffled_sentence_memory/",
        "val_shuffled_sentence_memory/",
        "val_analytic_prior_sentence_memory/",
    )
    for row in completed:
        if any(not any(str(key).startswith(prefix) for key in row) for prefix in namespaces):
            raise OrderedDecisionError("A validation epoch lacks one of five modes")
    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    for row in completed:
        validation_metrics = _selection_inputs_from_persisted_row(row)
        score, violation, feasible, _details = trainer.checkpoint_selection_diagnostics(
            validation_metrics, cfg, return_details=True
        )
        if (
            float(row.get("selection_score", math.nan)) != float(score)
            or float(row.get("selection_constraint_violation", math.nan))
            != float(violation)
            or bool(float(row.get("selection_feasible", 0.0))) != bool(feasible)
        ):
            raise OrderedDecisionError(
                "Epoch selection score/violation differs from exact recomputation"
            )
    replay = _replay_selection_history(completed)
    recomputed = replay["gates"]
    for row, (feasible, _details) in zip(completed, recomputed):
        if bool(float(row.get("selection_feasible", 0.0))) != feasible:
            raise OrderedDecisionError("Trainer feasibility disagrees with independent gate")
    has_feasible = bool(summary.get("has_feasible_checkpoint", False))
    if has_feasible != any(result[0] for result in recomputed):
        raise OrderedDecisionError("Selection summary feasibility is inconsistent")
    expected_summary = {
        "has_feasible_checkpoint": has_feasible,
        "best_feasible_score": replay["best_feasible_score"],
        "best_infeasible_score": replay["best_infeasible_score"],
        "required": True,
        "best_infeasible_key": replay["best_infeasible_key"],
        "early_stopping": replay["early_stopping"],
    }
    if summary != expected_summary:
        raise OrderedDecisionError(
            "Selection summary disagrees with exact lexicographic replay"
        )
    if not bool(replay["early_stopping"].get("stopped", False)) and max(epochs) != 4:
        raise OrderedDecisionError("Run neither early-stopped nor reached epoch four")
    checkpoint_name = "best.pt" if has_feasible else "best_infeasible.pt"
    checkpoint_path = (run_dir / "checkpoints" / checkpoint_name).resolve()
    if not checkpoint_path.is_file():
        raise OrderedDecisionError(f"Selected checkpoint is missing: {checkpoint_name}")
    if has_feasible and (run_dir / "checkpoints" / "best_infeasible.pt").resolve() == checkpoint_path:
        raise OrderedDecisionError("best_infeasible.pt can never be promoted")
    checkpoint = _load_checkpoint(checkpoint_path)
    evidence = _checkpoint_identity_evidence(
        checkpoint,
        stage=stage,
        approved_cfg=cfg,
        resolved_cfg=resolved,
    )
    metric_identity_names = {
        "sentence_memory_architecture_digest": "architecture",
        "sentence_memory_evaluation_control_digest": "evaluation_control",
        "sentence_memory_selection_aggregation_digest": "selection_aggregation",
        "validation_corruption_map_digest": "validation_corruption_map",
    }
    for row in completed:
        for metric_name, identity_name in metric_identity_names.items():
            if row.get(metric_name) != evidence["identities"][identity_name]["digest"]:
                raise OrderedDecisionError(
                    f"Epoch metrics have the wrong {metric_name}"
                )
    selected_metrics = dict(checkpoint.get("metrics", {}) or {})
    _validate_replayed_winner(checkpoint, replay, has_feasible=has_feasible)
    selected_feasible, selected_gate = independently_feasible(selected_metrics)
    if selected_feasible != has_feasible:
        raise OrderedDecisionError("Selected checkpoint has the wrong feasibility class")
    expected_score = (
        summary.get("best_feasible_score")
        if has_feasible
        else summary.get("best_infeasible_score")
    )
    if float(selected_metrics.get("selection_score", math.nan)) != float(expected_score):
        raise OrderedDecisionError("Selected checkpoint score disagrees with summary")
    last_checkpoint_path = run_dir / "checkpoints" / "last.pt"
    last_checkpoint = _load_checkpoint(last_checkpoint_path)
    if (
        int(last_checkpoint.get("epoch", -1)) != max(epochs)
        or dict(last_checkpoint.get("metrics", {}) or {}) != completed[-1]
        or dict(last_checkpoint.get("selection_state", {}) or {})
        != replay["selection_states"][max(epochs)]
    ):
        raise OrderedDecisionError(
            "Terminal last.pt disagrees with replayed metrics and patience state"
        )
    evidence.update(
        {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
            "metrics_jsonl_sha256": sha256_file(metrics_path),
            "selection_summary_sha256": sha256_file(run_dir / "selection_summary.json"),
            "resolved_config_sha256": sha256_file(resolved_path),
            "terminal_epoch": max(epochs),
            "validation_events": len(completed),
            "scientific_gate": selected_gate,
            "has_feasible_checkpoint": has_feasible,
        }
    )
    return checkpoint, evidence, selected_metrics


def _load_dev_export(
    directory: Path,
    *,
    expected_mode: str,
    checkpoint_path: Path,
    config_path: Path,
    partition: Mapping[str, Any],
    require_factorized_control: bool = True,
) -> dict[str, Any]:
    directory = directory.resolve()
    summary_path = directory / "export_summary.json"
    summary = _json(summary_path)
    if (
        summary.get("split") != "val"
        or summary.get("length_mode") != "predicted"
        or summary.get("word_prior_mode") != "off"
        or summary.get("sentence_memory_mode") != expected_mode
    ):
        raise OrderedDecisionError(f"Invalid development {expected_mode} export controls")
    if require_factorized_control and dict(
        summary.get("sentence_memory_evaluation_corruption", {}) or {}
    ) != EXPECTED_EVALUATION_CORRUPTION:
        raise OrderedDecisionError("Development export control identity changed")
    if require_factorized_control:
        validate_factorized_export_query_binding(
            summary.get("sentence_memory_query_binding"),
            expected_authority="config_pinned_development_manifest_v1",
            expected_manifest_file="manifest_development.jsonl",
            expected_manifest_sha256=partition[
                "development_manifest_sha256"
            ],
            expected_query_rows=EXPECTED_DEVELOPMENT_ROWS,
        )
    if _resolve_existing(summary.get("checkpoint"), directory) != checkpoint_path.resolve():
        raise OrderedDecisionError("Development export used another checkpoint")
    if _resolve_existing(summary.get("config"), directory) != config_path.resolve():
        raise OrderedDecisionError("Development export used another config")
    manifest = dict(summary.get("manifest", {}) or {})
    expected_manifest_authority = (
        "config_pinned_development_manifest_v1"
        if require_factorized_control
        else "explicit_isolated_development_environment_v1"
    )
    dataset_manifest = _resolve_existing(
        manifest.get("dataset_manifest"), directory
    )
    if (
        manifest.get("canonical_source_manifest") is not None
        or manifest.get("canonical_sample_count") is not None
        or manifest.get("is_complete_canonical_manifest") is not None
        or manifest.get("is_canonical_order") is not None
        or manifest.get("canonical_manifest_inspection")
        != "forbidden_isolated_explicit_manifest_v1"
        or manifest.get("query_manifest_authority")
        != expected_manifest_authority
        or manifest.get("dataset_manifest_sha256")
        != partition["development_manifest_sha256"]
        or manifest.get("sealed_partition_artifact_identity")
        != EXPECTED_PARTITION_ARTIFACT_IDENTITY
        or dataset_manifest != Path(partition["development_manifest"]).resolve()
    ):
        raise OrderedDecisionError(
            "Development export lacks the sealed no-holdout-read manifest evidence"
        )
    output_manifest = _resolve_existing(manifest.get("output_manifest"), directory)
    if sha256_file(output_manifest) != partition["development_manifest_sha256"]:
        raise OrderedDecisionError("Development export opened the wrong rows")
    rows = list(summary.get("rows", []) or [])
    if len(rows) != EXPECTED_DEVELOPMENT_ROWS:
        raise OrderedDecisionError("Development integrity export is incomplete")
    expected_rows = partition["rows"]
    identities = [
        (str(row.get("name", "")), normalize_sentence_text(row.get("text", "")))
        for row in rows
    ]
    expected_identities = [
        (str(row.get("name", "")), normalize_sentence_text(row.get("text", "")))
        for row in expected_rows
    ]
    if identities != expected_identities:
        raise OrderedDecisionError("Development export row identity/order changed")
    durations = np.asarray(
        [row.get("predicted_duration_seconds") for row in rows], dtype=np.float64
    )
    if not np.isfinite(durations).all():
        raise OrderedDecisionError("Development export has non-finite durations")
    return {
        "dir": directory,
        "summary": summary,
        "summary_path": summary_path.resolve(),
        "rows": rows,
        "durations": durations,
    }


def validate_factorized_export_query_binding(
    value: Any,
    *,
    expected_authority: str,
    expected_manifest_file: str,
    expected_manifest_sha256: str,
    expected_query_rows: int,
) -> dict[str, Any]:
    """Validate the no-fallback canonical-table projection in an export."""

    if not isinstance(value, Mapping):
        raise OrderedDecisionError(
            "Factorized export lacks its exact query-table binding"
        )
    binding = dict(value)
    expected_keys = {
        "schema_name",
        "schema_version",
        "authority",
        "partition_artifact_identity",
        "query_manifest",
        "canonical_neighbor_table",
        "bank_id",
        "lookup_mode",
        "online_fallback_allowed",
        "digest",
    }
    query_manifest = dict(binding.get("query_manifest", {}) or {})
    canonical_table = dict(binding.get("canonical_neighbor_table", {}) or {})
    canonical_order_sha256 = str(canonical_table.get("query_order_sha256", ""))
    without_digest = {
        key: item for key, item in binding.items() if key != "digest"
    }
    if (
        set(binding) != expected_keys
        or binding.get("schema_name")
        != "factorized_sentence_memory_export_query_binding"
        or int(binding.get("schema_version", -1)) != 1
        or binding.get("authority") != expected_authority
        or binding.get("partition_artifact_identity")
        != EXPECTED_PARTITION_ARTIFACT_IDENTITY
        or query_manifest
        != {
            "file": expected_manifest_file,
            "sha256": expected_manifest_sha256,
            "row_count": int(expected_query_rows),
        }
        or set(canonical_table)
        != {
            "sha256",
            "query_manifest_sha256",
            "query_order_sha256",
            "query_count",
        }
        or canonical_table.get("sha256") != EXPECTED_NEIGHBOR_SHA256["val"]
        or canonical_table.get("query_manifest_sha256")
        != EXPECTED_VALIDATION_MANIFEST_SHA256
        or int(canonical_table.get("query_count", -1)) != 1_077
        or len(canonical_order_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in canonical_order_sha256
        )
        or binding.get("bank_id") != EXPECTED_BANK_ID
        or binding.get("lookup_mode")
        != "exact_name_indexed_parent_subset_v1"
        or binding.get("online_fallback_allowed") is not False
        or binding.get("digest") != _digest_json(without_digest)
    ):
        raise OrderedDecisionError(
            "Factorized export query-table binding is invalid or detached"
        )
    return binding


def _load_rot6d(export: Mapping[str, Any], row: Mapping[str, Any]) -> np.ndarray:
    sample_path = _resolve_existing(row.get("sample"), Path(export["dir"]))
    with np.load(sample_path, allow_pickle=False) as sample:
        if "rot6d" not in sample.files:
            raise OrderedDecisionError(f"Export sample has no rot6d: {sample_path}")
        values = np.asarray(sample["rot6d"])
    if not np.isfinite(values).all():
        raise OrderedDecisionError(f"Export sample has non-finite rot6d: {sample_path}")
    return values


def _validate_development_all_null(
    memory_off: Mapping[str, Any], all_null: Mapping[str, Any]
) -> dict[str, Any]:
    if not np.array_equal(memory_off["durations"], all_null["durations"]):
        raise OrderedDecisionError("All-null changed development durations")
    max_abs = 0.0
    for off_row, null_row in zip(memory_off["rows"], all_null["rows"]):
        off = _load_rot6d(memory_off, off_row)
        null = _load_rot6d(all_null, null_row)
        if off.shape != null.shape or not np.array_equal(off, null):
            difference = math.inf if off.shape != null.shape else float(
                np.max(np.abs(off.astype(np.float64) - null.astype(np.float64)), initial=0.0)
            )
            raise OrderedDecisionError(
                f"All-null is not exactly memory-off; max_abs={difference}"
            )
        max_abs = max(
            max_abs,
            float(np.max(np.abs(off.astype(np.float64) - null.astype(np.float64)), initial=0.0)),
        )
        sample_path = _resolve_existing(null_row.get("sample"), Path(all_null["dir"]))
        with np.load(sample_path, allow_pickle=False) as sample:
            required = (
                "sentence_memory_candidate_mask",
                "sentence_memory_payload_reads",
                "trajectory_sentence_memory_gates",
                "trajectory_sentence_memory_null_mass",
                "trajectory_sentence_memory_candidate_mass",
            )
            missing = [name for name in required if name not in sample.files]
            if missing:
                raise OrderedDecisionError(f"All-null sample lacks fields: {missing}")
            if np.asarray(sample["sentence_memory_candidate_mask"]).any():
                raise OrderedDecisionError("All-null candidate mask is not empty")
            if np.asarray(sample["sentence_memory_payload_reads"]).reshape(-1).tolist() != [0]:
                raise OrderedDecisionError("All-null export read a motion payload")
            if not np.array_equal(
                np.asarray(sample["trajectory_sentence_memory_gates"]),
                np.zeros_like(sample["trajectory_sentence_memory_gates"]),
            ):
                raise OrderedDecisionError("All-null gate is not exactly zero")
            if not np.array_equal(
                np.asarray(sample["trajectory_sentence_memory_candidate_mass"]),
                np.zeros_like(sample["trajectory_sentence_memory_candidate_mass"]),
            ):
                raise OrderedDecisionError("All-null candidate mass is not exactly zero")
            if not np.array_equal(
                np.asarray(sample["trajectory_sentence_memory_null_mass"]),
                np.ones_like(sample["trajectory_sentence_memory_null_mass"]),
            ):
                raise OrderedDecisionError("All-null mass is not exactly one")
    return {
        "development_rows": len(memory_off["rows"]),
        "prediction_arrays_equal": True,
        "duration_values_equal": True,
        "prediction_max_abs": max_abs,
        "all_null_gate_exact_zero": True,
        "all_null_candidate_mass_exact_zero": True,
        "all_null_mass_exact_one": True,
        "payload_reads": 0,
        "passed": True,
    }


def _validate_selected_v2_parity(
    memory_off: Mapping[str, Any], v2_text_only: Mapping[str, Any]
) -> dict[str, Any]:
    duration_max_abs = float(
        np.max(
            np.abs(memory_off["durations"] - v2_text_only["durations"]),
            initial=0.0,
        )
    )
    prediction_max_abs = 0.0
    for selected_row, v2_row in zip(memory_off["rows"], v2_text_only["rows"]):
        selected = _load_rot6d(memory_off, selected_row)
        baseline = _load_rot6d(v2_text_only, v2_row)
        if selected.shape != baseline.shape:
            raise OrderedDecisionError("Selected memory-off/v2 trajectory shapes differ")
        prediction_max_abs = max(
            prediction_max_abs,
            float(
                np.max(
                    np.abs(selected.astype(np.float64) - baseline.astype(np.float64)),
                    initial=0.0,
                )
            ),
        )
    passed = prediction_max_abs <= 1e-7 and duration_max_abs <= 1e-7
    if not passed:
        raise OrderedDecisionError(
            "Selected checkpoint memory-off differs from pinned v2; "
            f"prediction_max_abs={prediction_max_abs}, "
            f"duration_max_abs={duration_max_abs}"
        )
    return {
        "prediction_max_abs": prediction_max_abs,
        "duration_max_abs": duration_max_abs,
        "tolerance": 1e-7,
        "development_rows": len(memory_off["rows"]),
        "passed": True,
    }


def _authorization_payload(
    *, purpose: str, stage: str, decision: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_name": AUTHORIZATION_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "purpose": purpose,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "decision_identity": decision["decision_identity"],
        "checkpoint": decision["checkpoint"],
        "partition": decision["partition"],
        "run_launch_identity": decision["run_launch_identity"],
        "predecessor_authorization": decision.get("predecessor_authorization"),
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    return {**payload, "authorization_identity": _digest_json(payload)}


def _require_stage2_source_match(
    authorization: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep both predeclared arms on the exact Stage-1 source checkout."""

    authorized_source = dict(
        dict(authorization.get("run_launch_identity", {}) or {}).get("source", {})
        or {}
    )
    if not authorized_source or dict(source) != authorized_source:
        raise OrderedDecisionError(
            "Stage 2 source differs from the predeclared Stage 1 source"
        )
    return authorized_source


def record_stage2_input(
    *,
    authorization_path: Path,
    out_file: Path,
    stage2_config_path: Path,
    v2_checkpoint_path: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
) -> dict[str, Any]:
    """Record Stage-1 authorization before Stage-2 initializes data or weights."""

    authorization = verify_authorization(
        authorization_path, purpose="stage2", stage=STAGE1
    )
    _validate_config(STAGE2, stage2_config_path)
    if sha256_file(v2_checkpoint_path) != EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Stage 2 is not pinned to the approved v2 checkpoint")
    source_checkout = _validate_shared_source_checkout(
        source_root,
        expected_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    _require_stage2_source_match(authorization, source_checkout)
    without_identity = {
        "schema_name": STAGE2_INPUT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE2,
        "experiment_name": EXPERIMENTS[STAGE2],
        "recorded_before_training": True,
        "stage1_authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
            "authorization_identity": authorization["authorization_identity"],
            "decision_identity": authorization["decision_identity"],
        },
        "stage2_config": {
            "path": str(stage2_config_path.resolve()),
            "sha256": sha256_file(stage2_config_path),
        },
        "fresh_v2_checkpoint": {
            "path": str(v2_checkpoint_path.resolve()),
            "sha256": EXPECTED_V2_SHA256,
        },
        "source": source_checkout,
        "resume": False,
        "warm_start": False,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    payload = {**without_identity, "input_identity": _digest_json(without_identity)}
    return _atomic_json_record(out_file, payload)


def record_run_launch(
    *,
    stage: str,
    out_file: Path,
    config_path: Path,
    v2_checkpoint_path: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    launcher_paths: list[Path],
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    """Atomically attest the clean pushed source selected before training."""

    _validate_config(stage, config_path)
    if sha256_file(v2_checkpoint_path) != EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Launch baseline is not the pinned v2 checkpoint")
    source_git_head = str(source_git_head).lower()
    if len(source_git_head) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in source_git_head
    ):
        raise OrderedDecisionError("Launch source git identity is malformed")
    if not str(source_remote_ref).startswith("origin/"):
        raise OrderedDecisionError("Launch source must be an exact origin remote ref")
    source_remote_head = str(source_remote_head).lower()
    if source_remote_head != source_git_head:
        raise OrderedDecisionError(
            "Live origin branch head does not equal the selected source commit"
        )
    source_checkout = _validate_shared_source_checkout(
        source_root,
        expected_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    scripts = []
    for path in launcher_paths:
        resolved = path.resolve()
        scripts.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    predecessor = None
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage-2 launch lacks Stage-1 authorization")
        authorization = verify_authorization(
            stage1_authorization, purpose="stage2", stage=STAGE1
        )
        _require_stage2_source_match(authorization, source_checkout)
        predecessor = {
            "path": str(stage1_authorization.resolve()),
            "sha256": sha256_file(stage1_authorization),
            "authorization_identity": authorization["authorization_identity"],
            "decision_identity": authorization["decision_identity"],
        }
    elif stage1_authorization is not None:
        raise OrderedDecisionError("Stage-1 launch cannot have a predecessor")
    without_identity = {
        "schema_name": LAUNCH_INPUT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "recorded_before_training": True,
        "source": {
            "git_head": source_git_head,
            "remote_ref": str(source_remote_ref),
            "remote_head": source_remote_head,
            "remote_ref_exact_match_checked": True,
            "worktree_clean_checked": True,
            "repository_root": source_checkout["repository_root"],
            "git_directory": source_checkout["git_directory"],
            "durable_experiments_root": source_checkout[
                "durable_experiments_root"
            ],
            "frozen_text_model_root": source_checkout["frozen_text_model_root"],
            "standalone_shared_clone_checked": True,
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "fresh_v2_checkpoint": {
            "path": str(v2_checkpoint_path.resolve()),
            "sha256": EXPECTED_V2_SHA256,
        },
        "launchers": scripts,
        "slurm": {
            "job_id": str(os.environ.get("SLURM_JOB_ID", "")),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
            "tasks": int(os.environ.get("SLURM_NTASKS", "0")),
        },
        "stage1_authorization": predecessor,
        "wandb": "disabled",
        "staged_neighbor_splits": ["train", "val"],
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    if without_identity["slurm"]["job_id"] == "" or without_identity["slurm"][
        "nodes"
    ] != 4:
        raise OrderedDecisionError("Full launch identity requires a four-node Slurm job")
    payload = {**without_identity, "launch_identity": _digest_json(without_identity)}
    return _atomic_json_record(out_file, payload)


def _validate_run_launch(
    path: Path,
    *,
    stage: str,
    config_path: Path,
    stage1_authorization: Path | None,
) -> dict[str, Any]:
    value = _json(path)
    without_identity = {key: child for key, child in value.items() if key != "launch_identity"}
    if (
        value.get("schema_name") != LAUNCH_INPUT_SCHEMA_NAME
        or value.get("stage") != stage
        or value.get("experiment_name") != EXPERIMENTS[stage]
        or value.get("recorded_before_training") is not True
        or value.get("wandb") != "disabled"
        or value.get("staged_neighbor_splits") != ["train", "val"]
        or _digest_json(without_identity) != value.get("launch_identity")
    ):
        raise OrderedDecisionError("Run-launch identity is invalid")
    source = dict(value.get("source", {}) or {})
    if (
        source.get("worktree_clean_checked") is not True
        or source.get("remote_ref_exact_match_checked") is not True
        or source.get("standalone_shared_clone_checked") is not True
        or not str(source.get("remote_ref", "")).startswith("origin/")
        or str(source.get("remote_head", "")).lower()
        != str(source.get("git_head", "")).lower()
        or not Path(str(source.get("repository_root", ""))).is_relative_to(
            Path("/media/cvpr")
        )
        or not Path(
            str(source.get("durable_experiments_root", ""))
        ).is_relative_to(Path("/media/cvpr"))
        or not Path(
            str(source.get("frozen_text_model_root", ""))
        ).is_relative_to(Path("/media/cvpr"))
    ):
        raise OrderedDecisionError("Run-launch source is not clean and pushed")
    config = dict(value.get("config", {}) or {})
    if (
        Path(str(config.get("path", ""))).resolve() != config_path.resolve()
        or sha256_file(config_path) != config.get("sha256")
    ):
        raise OrderedDecisionError("Run-launch config identity changed")
    baseline = dict(value.get("fresh_v2_checkpoint", {}) or {})
    if baseline.get("sha256") != EXPECTED_V2_SHA256 or sha256_file(
        Path(str(baseline.get("path", "")))
    ) != EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Run-launch v2 identity changed")
    if int(dict(value.get("slurm", {}) or {}).get("nodes", -1)) != 4:
        raise OrderedDecisionError("Run-launch allocation was not four nodes")
    for script in value.get("launchers", []):
        script = dict(script)
        if sha256_file(Path(str(script.get("path", "")))) != script.get("sha256"):
            raise OrderedDecisionError("A launch script changed after training")
    predecessor = value.get("stage1_authorization")
    if stage == STAGE2:
        if stage1_authorization is None or not isinstance(predecessor, Mapping):
            raise OrderedDecisionError("Stage-2 launch lacks predecessor evidence")
        if (
            Path(str(predecessor.get("path", ""))).resolve()
            != stage1_authorization.resolve()
            or sha256_file(stage1_authorization) != predecessor.get("sha256")
        ):
            raise OrderedDecisionError("Stage-2 launch predecessor changed")
        authorization = verify_authorization(
            stage1_authorization, purpose="stage2", stage=STAGE1
        )
        _require_stage2_source_match(authorization, source)
    elif predecessor is not None:
        raise OrderedDecisionError("Stage-1 launch unexpectedly has a predecessor")
    return value


def _validate_stage2_input(
    path: Path,
    *,
    stage1_authorization: Path,
    config_path: Path,
) -> dict[str, Any]:
    value = _json(path)
    without_identity = {key: child for key, child in value.items() if key != "input_identity"}
    if (
        value.get("schema_name") != STAGE2_INPUT_SCHEMA_NAME
        or value.get("stage") != STAGE2
        or value.get("experiment_name") != EXPERIMENTS[STAGE2]
        or value.get("recorded_before_training") is not True
        or value.get("resume") is not False
        or value.get("warm_start") is not False
        or _digest_json(without_identity) != value.get("input_identity")
    ):
        raise OrderedDecisionError("Stage-2 launch-input evidence is invalid")
    predecessor = dict(value.get("stage1_authorization", {}) or {})
    if (
        Path(str(predecessor.get("path", ""))).resolve()
        != stage1_authorization.resolve()
        or sha256_file(stage1_authorization) != predecessor.get("sha256")
    ):
        raise OrderedDecisionError("Stage-2 launch used another Stage-1 authorization")
    config = dict(value.get("stage2_config", {}) or {})
    if (
        Path(str(config.get("path", ""))).resolve() != config_path.resolve()
        or sha256_file(config_path) != config.get("sha256")
    ):
        raise OrderedDecisionError("Stage-2 launch config evidence changed")
    baseline = dict(value.get("fresh_v2_checkpoint", {}) or {})
    if baseline.get("sha256") != EXPECTED_V2_SHA256 or sha256_file(
        Path(str(baseline.get("path", "")))
    ) != EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Stage-2 launch did not start from the pinned v2")
    authorization = verify_authorization(
        stage1_authorization, purpose="stage2", stage=STAGE1
    )
    _require_stage2_source_match(
        authorization, dict(value.get("source", {}) or {})
    )
    return value


def _precheckpoint_output_evidence(
    run_dir: Path, *, stage: str
) -> dict[str, Any]:
    """Prove an output tree contains no persisted optimizer/training progress."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise OrderedDecisionError("Pre-checkpoint recovery output is not a directory")
    allowed = {
        "config.resolved.json",
        "validation_text_partition.json",
        "validation_text_partition.json.tmp",
        "checkpoints",
    }
    entries = sorted(run_dir.iterdir(), key=lambda path: path.name)
    unexpected = [path.name for path in entries if path.name not in allowed]
    if unexpected:
        raise OrderedDecisionError(
            f"Pre-checkpoint output contains non-initialization artifacts: {unexpected}"
        )
    checkpoints = run_dir / "checkpoints"
    checkpoint_temporaries: list[dict[str, Any]] = []
    if checkpoints.exists():
        if not checkpoints.is_dir() or checkpoints.is_symlink():
            raise OrderedDecisionError(
                "Pre-checkpoint recovery refuses any checkpoint artifact"
            )
        for path in sorted(checkpoints.iterdir(), key=lambda child: child.name):
            if (
                not path.is_file()
                or path.is_symlink()
                or re.fullmatch(
                    r"\.last\.pt\.[A-Za-z0-9_-]{6,64}\.tmp", path.name
                )
                is None
            ):
                raise OrderedDecisionError(
                    "Pre-checkpoint recovery refuses published or unknown "
                    f"checkpoint artifact: {path.name}"
                )
            checkpoint_temporaries.append(
                {
                    "file": f"checkpoints/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "published": False,
                }
            )
    artifacts: dict[str, Any] = {}
    for filename in (
        "config.resolved.json",
        "validation_text_partition.json",
        "validation_text_partition.json.tmp",
    ):
        path = run_dir / filename
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 5_000_000:
            raise OrderedDecisionError(
                f"Initialization artifact is unsafe for recovery: {path}"
            )
        row = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "json_complete": False,
        }
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Direct config writes can be interrupted. The immutable source
            # config and run-launch identity, not this incomplete copy, are
            # authoritative for the fresh retry.
            value = None
        if value is not None:
            if not isinstance(value, dict):
                raise OrderedDecisionError(
                    f"Initialization JSON is not an object: {path}"
                )
            row["json_complete"] = True
            if filename == "config.resolved.json":
                if value.get("experiment_name") != EXPERIMENTS[stage]:
                    raise OrderedDecisionError(
                        "Initialization config belongs to another experiment"
                    )
                partition = dict(value.get("validation_text_partition", {}) or {})
                forbidden = {
                    "resolved_artifact",
                    "assignments",
                    "confirmation_manifest",
                    "confirmation_normalized_texts",
                    "development_normalized_texts",
                    "partition_dir",
                }
                if forbidden.intersection(partition):
                    raise OrderedDecisionError(
                        "Initialization config persisted holdout membership"
                    )
            elif filename.startswith("validation_text_partition.json"):
                if (
                    value.get("schema_name")
                    != "signtrajfield_development_validation_runtime"
                    or value.get("validation_only") is not True
                    or "assignments" in value
                ):
                    raise OrderedDecisionError(
                        "Initialization partition artifact is not development-only"
                    )
        artifacts[filename] = row
    return {
        "entries": [path.name for path in entries],
        "artifacts": artifacts,
        "unpublished_checkpoint_temporaries": checkpoint_temporaries,
        "published_checkpoints_absent": True,
        "metrics_absent": not (run_dir / "metrics.jsonl").exists(),
        "selection_summary_absent": not (run_dir / "selection_summary.json").exists(),
        "material_progress_preserved": True,
    }


def record_precheckpoint_retry(
    *,
    stage: str,
    run_dir: Path,
    out_file: Path,
    config_path: Path,
    partition_dir: Path,
    lease_path: Path,
    lease_attestation_path: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    """Quarantine only initialization output so epoch 1 can restart safely."""

    if stage not in STAGES:
        raise OrderedDecisionError("Unknown pre-checkpoint recovery stage")
    _validate_config(stage, config_path)
    partition = _validate_partition(partition_dir)
    launch_path = Path(f"{run_dir.resolve()}.prerequisites") / (
        "run_launch_identity.json"
    )
    launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=stage1_authorization,
    )
    source = _validate_shared_source_checkout(
        source_root,
        expected_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    if source != dict(launch.get("source", {}) or {}):
        raise OrderedDecisionError("Pre-checkpoint retry source differs from launch")
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage-2 retry lacks Stage-1 authorization")
        _validate_stage2_input(
            Path(f"{run_dir.resolve()}.prerequisites")
            / "ordered_stage_input.json",
            stage1_authorization=stage1_authorization,
            config_path=config_path,
        )
    elif stage1_authorization is not None:
        raise OrderedDecisionError("Stage-1 retry cannot have a predecessor")

    _lease_attestation, claim = _validate_execution_lease_attestation(
        lease_attestation_path, lease_path=lease_path.resolve()
    )
    if (
        claim.get("purpose") != "training"
        or claim.get("stage") != stage
        or claim.get("source_git_head") != str(source_git_head).lower()
        or claim.get("binding_identity") != sha256_file(config_path)
    ):
        raise OrderedDecisionError("Pre-checkpoint retry has the wrong active lease")
    replaced = claim.get("replaced_stale_owner")
    original_job = str(dict(launch.get("slurm", {}) or {}).get("job_id", ""))
    same_requeued_job = (
        replaced is None
        and claim.get("slurm_job_id") == original_job
        and int(os.environ.get("SLURM_RESTART_COUNT", "0") or "0") > 0
    )
    if replaced is None and not same_requeued_job:
        raise OrderedDecisionError(
            "Pre-checkpoint retry lacks scheduler-proven stale-owner evidence"
        )
    if isinstance(replaced, Mapping):
        scheduler = dict(replaced.get("scheduler_evidence", {}) or {})
        if scheduler.get("classification") not in {"terminal", "not_found"}:
            raise OrderedDecisionError(
                "Pre-checkpoint retry predecessor was not proven terminal"
            )

    run_dir = run_dir.resolve()
    quarantine_root = Path(f"{run_dir}.prerequisites") / "precheckpoint_outputs"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    claimed_quarantines: set[Path] = set()
    if out_file.parent.is_dir():
        for record_path in out_file.parent.glob("*.json"):
            try:
                record = _json(record_path)
            except OrderedDecisionError:
                continue
            if (
                record.get("schema_name")
                != "signtrajfield_factorized_memory_precheckpoint_retry"
            ):
                continue
            for row in record.get("quarantined_outputs", []):
                if isinstance(row, Mapping):
                    claimed_quarantines.add(
                        Path(str(row.get("path", ""))).resolve()
                    )
    if run_dir.exists():
        # Validate before moving: an unsafe tree must remain untouched for
        # manual inspection rather than becoming a failed "safe" quarantine.
        _precheckpoint_output_evidence(run_dir, stage=stage)
        quarantine = quarantine_root / (
            f"{claim['slurm_job_id']}.{claim['claim_identity']}.output"
        )
        if quarantine.exists():
            raise OrderedDecisionError("Pre-checkpoint quarantine target exists")
        os.rename(run_dir, quarantine)
    candidates = sorted(path for path in quarantine_root.iterdir() if path.is_dir())
    unclaimed = [
        path for path in candidates if path.resolve() not in claimed_quarantines
    ]
    if len(unclaimed) > 1:
        raise OrderedDecisionError(
            "Pre-checkpoint history contains multiple unattested output trees"
        )
    quarantined_outputs = [
        {
            "path": str(path.resolve()),
            "previously_attested": path.resolve() in claimed_quarantines,
            "evidence": _precheckpoint_output_evidence(path, stage=stage),
        }
        for path in candidates
    ]
    without_identity = {
        "schema_name": "signtrajfield_factorized_memory_precheckpoint_retry",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "fresh_epoch_restart": 1,
        "run_dir": str(run_dir),
        "quarantined_outputs": quarantined_outputs,
        "run_launch": {
            "path": str(launch_path.resolve()),
            "sha256": sha256_file(launch_path),
            "launch_identity": launch["launch_identity"],
        },
        "development_manifest": {
            "path": str(partition["development_manifest"]),
            "sha256": partition["development_manifest_sha256"],
        },
        "active_lease_claim_identity": claim["claim_identity"],
        "stale_owner_evidence": replaced,
        "same_slurm_job_requeue": same_requeued_job,
        "source": source,
        "config_sha256": sha256_file(config_path),
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    payload = {
        **without_identity,
        "retry_identity": _digest_json(without_identity),
    }
    return _atomic_json_record(out_file, payload, allow_matching_existing=True)


def _rng_resume_evidence(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the four-rank RNG payload needed for an exact continuation."""

    rng = dict(checkpoint.get("rng_state", {}) or {})
    states = list(rng.get("rank_states", []) or [])
    if int(rng.get("world_size", -1)) != 4 or len(states) != 4:
        raise OrderedDecisionError("Resume checkpoint lacks four-rank RNG state")
    evidence: list[dict[str, Any]] = []
    ranks: list[int] = []
    for row in states:
        if not isinstance(row, Mapping) or not isinstance(row.get("state"), Mapping):
            raise OrderedDecisionError("Resume checkpoint has malformed rank RNG state")
        rank = int(row.get("rank", -1))
        state = dict(row["state"])
        numpy_state = dict(state.get("numpy", {}) or {})
        cpu_state = state.get("torch_cpu")
        cuda_state = state.get("torch_cuda")
        if (
            not isinstance(state.get("python"), (list, tuple))
            or not numpy_state.get("bit_generator")
            or not isinstance(numpy_state.get("state"), (list, tuple))
            or not torch.is_tensor(cpu_state)
            or cpu_state.numel() < 1
            or not torch.is_tensor(cuda_state)
            or cuda_state.numel() < 1
        ):
            raise OrderedDecisionError(
                f"Resume checkpoint rank {rank} lacks exact CPU/CUDA RNG state"
            )

        def tensor_sha256(value: torch.Tensor) -> str:
            array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
            return hashlib.sha256(array.tobytes()).hexdigest()

        ranks.append(rank)
        evidence.append(
            {
                "rank": rank,
                "python_state_present": True,
                "numpy_state_sha256": _digest_json(numpy_state),
                "torch_cpu_sha256": tensor_sha256(cpu_state),
                "torch_cuda_sha256": tensor_sha256(cuda_state),
            }
        )
    if sorted(ranks) != [0, 1, 2, 3]:
        raise OrderedDecisionError("Resume checkpoint RNG ranks are not exactly 0..3")
    return {"world_size": 4, "rank_states": sorted(evidence, key=lambda row: row["rank"])}


def _resume_progress_evidence(
    checkpoint: Mapping[str, Any], metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Accept only complete-epoch or post-train/pending-validation snapshots."""

    epoch = int(checkpoint.get("epoch", -1))
    if epoch not in range(1, 5):
        raise OrderedDecisionError("Exact resume requires last.pt from epoch 1..4")
    checkpoint_row = dict(checkpoint.get("metrics", {}) or {})
    _finite_tree(checkpoint_row, path="checkpoint.metrics")
    if int(checkpoint_row.get("epoch", -1)) != epoch:
        raise OrderedDecisionError("last.pt metric epoch does not match checkpoint epoch")
    try:
        pending_value = float(checkpoint_row.get("validation_pending", math.nan))
    except (TypeError, ValueError) as error:
        raise OrderedDecisionError("last.pt validation_pending is malformed") from error
    if pending_value not in {0.0, 1.0}:
        raise OrderedDecisionError("last.pt validation_pending must be exactly zero or one")

    selection = dict(checkpoint.get("selection_state", {}) or {})
    early = dict(selection.get("early_stopping", {}) or {})
    selection_schema = int(selection.get("schema_version", -1))
    infeasible_score = selection.get("best_infeasible_score")
    infeasible_key = selection.get("best_infeasible_key")
    if (
        selection_schema not in {2, 3}
        or int(early.get("schema_version", -1)) != 1
        or (selection_schema == 2 and (infeasible_score is not None or infeasible_key is not None))
        or (selection_schema == 3 and (infeasible_score is None or infeasible_key is None))
    ):
        raise OrderedDecisionError("Resume checkpoint lacks exact selection state")
    if selection_schema == 3:
        try:
            normalized_infeasible_key = [float(value) for value in infeasible_key]
        except (TypeError, ValueError) as error:
            raise OrderedDecisionError("Resume checkpoint has malformed infeasible key") from error
        if (
            len(normalized_infeasible_key) != 2
            or not all(math.isfinite(value) for value in normalized_infeasible_key)
            or normalized_infeasible_key[1] != float(infeasible_score)
        ):
            raise OrderedDecisionError("Resume checkpoint infeasible minima disagree")
    if early.get("stopped") not in {True, False}:
        raise OrderedDecisionError("Resume checkpoint has malformed early-stop state")
    completed_count = epoch - 1 if pending_value == 1.0 else epoch
    expected_epochs = list(range(1, completed_count + 1))
    if [int(row.get("epoch", -1)) for row in metrics] != expected_epochs:
        raise OrderedDecisionError(
            "Resume metrics history does not match completed validations"
        )
    if any(float(row.get("validation_pending", 1.0)) != 0.0 for row in metrics):
        raise OrderedDecisionError("Resume metrics JSONL contains an incomplete row")
    last_validation_epoch = early.get("last_validation_epoch")
    expected_last = completed_count if completed_count else None
    if (
        int(early.get("validation_count", -1)) != completed_count
        or last_validation_epoch != expected_last
    ):
        raise OrderedDecisionError(
            "Resume early-stop state is not aligned with completed validations"
        )
    if metrics:
        replay = _replay_selection_history(metrics)
        expected_selection = replay["selection_states"][completed_count]
    else:
        from NIAF.continuous_trajectory_field.scripts import (
            train_continuous_trajectory_field as trainer,
        )

        expected_selection = trainer.checkpoint_selection_state(
            math.inf,
            math.inf,
            early_stopping_state=trainer.initial_early_stopping_state(),
            best_infeasible_key=None,
        )
    if selection != expected_selection:
        raise OrderedDecisionError(
            "Resume selection state differs from exact metrics replay"
        )

    if pending_value == 1.0:
        if early.get("stopped") is not False or early.get("stop_epoch") is not None:
            raise OrderedDecisionError("Pending validation cannot follow a terminal state")
        if any(
            name in checkpoint_row
            for name in ("selection_feasible", "selection_score", "selection_constraint_violation")
        ):
            raise OrderedDecisionError("Pending-validation row already contains selection results")
        mode = "validation_pending"
    else:
        if not metrics or checkpoint_row != metrics[-1]:
            raise OrderedDecisionError("Completed last.pt does not match terminal metrics row")
        for name in (
            "selection_feasible",
            "selection_score",
            "selection_constraint_violation",
        ):
            if name not in checkpoint_row:
                raise OrderedDecisionError(f"Completed last.pt lacks {name}")
        stopped = bool(early.get("stopped"))
        if stopped != (early.get("stop_epoch") is not None):
            raise OrderedDecisionError("Completed last.pt has inconsistent stop state")
        mode = "terminal_complete" if stopped or epoch == 4 else "epoch_complete"

    return {
        "mode": mode,
        "checkpoint_epoch": epoch,
        "completed_validation_epochs": expected_epochs,
        "checkpoint_metrics": checkpoint_row,
        "checkpoint_metrics_sha256": _digest_json(checkpoint_row),
        "selection_state": selection,
        "selection_state_sha256": _digest_json(selection),
        "rng_state": _rng_resume_evidence(checkpoint),
    }


def _atomic_append_reconciled_metric(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically repair the save-before-JSONL crash window with checkpoint data."""

    previous = path.read_bytes() if path.is_file() else b""
    previous_sha256 = hashlib.sha256(previous).hexdigest()
    if previous and not previous.endswith(b"\n"):
        raise OrderedDecisionError("Cannot reconcile a metrics JSONL without final newline")
    encoded = (json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.reconcile.{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(previous)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "performed": True,
        "source": "last_checkpoint.metrics",
        "previous_metrics_sha256": previous_sha256,
        "reconciled_row_sha256": _digest_json(dict(row)),
        "result_metrics_sha256": sha256_file(path),
    }


def _recover_truncated_terminal_metric(
    path: Path, checkpoint_row: Mapping[str, Any], checkpoint_epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover only an interrupted final append that prefixes checkpoint.metrics."""

    raw = path.read_bytes()
    boundary = raw.rfind(b"\n")
    prefix = raw[: boundary + 1] if boundary >= 0 else b""
    fragment = raw[boundary + 1 :]
    try:
        fragment_text = fragment.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OrderedDecisionError("Metrics JSONL ends in non-UTF8 corruption") from error
    canonical_row = json.dumps(dict(checkpoint_row), sort_keys=True, allow_nan=False)
    if not fragment_text or not canonical_row.startswith(fragment_text):
        raise OrderedDecisionError(
            "Malformed metrics JSONL is not a truncated checkpoint row"
        )
    prefix_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(prefix.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise OrderedDecisionError(
                f"Metrics JSONL has malformed interior line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise OrderedDecisionError("Metrics JSONL prefix contains a non-object")
        _finite_tree(row, path=f"metrics[{line_number}]")
        prefix_rows.append(row)
    if [int(row.get("epoch", -1)) for row in prefix_rows] != list(
        range(1, checkpoint_epoch)
    ):
        raise OrderedDecisionError(
            "Truncated metrics recovery prefix is not complete through epoch N-1"
        )
    previous_sha256 = hashlib.sha256(raw).hexdigest()
    temporary = path.with_name(f".{path.name}.truncated-repair.{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(prefix)
            handle.write(canonical_row.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    rows = _read_metrics(path)
    return rows, {
        "performed": True,
        "kind": "truncated_final_append",
        "source": "last_checkpoint.metrics",
        "previous_metrics_sha256": previous_sha256,
        "truncated_fragment_sha256": hashlib.sha256(fragment).hexdigest(),
        "reconciled_row_sha256": _digest_json(dict(checkpoint_row)),
        "result_metrics_sha256": sha256_file(path),
    }


def _terminal_selection_summary_payload(
    selection_state: Mapping[str, Any],
) -> dict[str, Any]:
    best_score = selection_state.get("best_feasible_score")
    best_infeasible_score = selection_state.get("best_infeasible_score")
    has_feasible = best_score is not None and math.isfinite(float(best_score))
    return {
        "has_feasible_checkpoint": has_feasible,
        "best_feasible_score": float(best_score) if has_feasible else None,
        "best_infeasible_score": (
            float(best_infeasible_score)
            if best_infeasible_score is not None
            and math.isfinite(float(best_infeasible_score))
            else None
        ),
        "required": True,
        "best_infeasible_key": selection_state.get("best_infeasible_key"),
        "early_stopping": selection_state.get("early_stopping"),
    }


def _atomic_write_terminal_selection_summary(
    path: Path, summary: Mapping[str, Any]
) -> None:
    """Reconstruct the trainer's terminal summary after a post-epoch crash."""

    temporary = path.with_name(f".{path.name}.finalize.{os.getpid()}")
    encoded = (
        json.dumps(dict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def record_run_resume(
    *,
    stage: str,
    run_dir: Path,
    out_file: Path,
    config_path: Path,
    partition_dir: Path,
    last_checkpoint_path: Path,
    source_root: Path,
    source_git_head: str,
    source_remote_ref: str,
    source_remote_head: str,
    launcher_paths: list[Path],
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    """Validate and attest an exact four-rank continuation before resuming."""

    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    last_checkpoint_path = last_checkpoint_path.resolve()
    if run_dir.name != EXPERIMENTS[stage] or last_checkpoint_path != (
        run_dir / "checkpoints" / "last.pt"
    ):
        raise OrderedDecisionError("Resume checkpoint is not this stage's last.pt")
    if (run_dir / "evaluation" / "ordered_development_decision").exists():
        raise OrderedDecisionError("A decided stage cannot resume training")
    approved = _validate_config(stage, config_path)
    resolved_path = run_dir / "config.resolved.json"
    resolved = _json(resolved_path)
    _validate_resolved_config(stage=stage, resolved=resolved, approved=approved)
    launch_path = Path(f"{run_dir}.prerequisites") / "run_launch_identity.json"
    launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=stage1_authorization,
    )
    source_git_head = str(source_git_head).lower()
    launch_source = dict(launch.get("source", {}) or {})
    if source_git_head != str(launch_source.get("git_head", "")).lower():
        raise OrderedDecisionError("Resume source differs from the original launch")
    if str(launch_source.get("remote_head", "")).lower() != source_git_head:
        raise OrderedDecisionError("Original launch lacks exact remote source proof")
    source_check = _validate_shared_source_checkout(
        source_root,
        expected_head=source_git_head,
        expected_remote_ref=source_remote_ref,
        expected_remote_head=source_remote_head,
    )
    if (
        source_check["remote_ref"] != launch_source.get("remote_ref")
        or source_check["repository_root"] != launch_source.get("repository_root")
    ):
        raise OrderedDecisionError("Resume did not use the original shared source clone")

    partition = _validate_partition(partition_dir)
    before_sha256 = sha256_file(last_checkpoint_path)
    checkpoint = _load_checkpoint(last_checkpoint_path)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch not in range(1, 5):
        raise OrderedDecisionError("Exact resume requires last.pt from epoch 1..4")
    if not isinstance(checkpoint.get("optimizer"), Mapping):
        raise OrderedDecisionError("Resume checkpoint lacks optimizer state")
    identity_evidence = _checkpoint_identity_evidence(
        checkpoint,
        stage=stage,
        approved_cfg=approved,
        resolved_cfg=resolved,
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_existed = metrics_path.is_file()
    metrics_before_sha256 = sha256_file(metrics_path) if metrics_existed else None
    checkpoint_row = dict(checkpoint.get("metrics", {}) or {})
    try:
        validation_pending = float(checkpoint_row.get("validation_pending", math.nan))
    except (TypeError, ValueError) as error:
        raise OrderedDecisionError("last.pt validation_pending is malformed") from error
    reconciliation: dict[str, Any] = {"performed": False}
    try:
        metrics = _read_metrics(metrics_path) if metrics_existed else []
    except OrderedDecisionError:
        if validation_pending != 0.0 or not metrics_existed:
            raise
        metrics, reconciliation = _recover_truncated_terminal_metric(
            metrics_path, checkpoint_row, checkpoint_epoch
        )
    # The trainer intentionally saves a complete last.pt before appending its
    # JSONL row.  If interrupted in that narrow window, the checkpoint row is
    # the sole authoritative byte-for-byte repair source.
    if (
        validation_pending == 0.0
        and [int(row.get("epoch", -1)) for row in metrics]
        == list(range(1, checkpoint_epoch))
    ):
        reconciliation = _atomic_append_reconciled_metric(metrics_path, checkpoint_row) | {
            "kind": "missing_complete_row"
        }
        metrics = _read_metrics(metrics_path)
    progress = _resume_progress_evidence(checkpoint, metrics)
    terminal_summary = None
    terminal_selected_checkpoint = None
    if progress["mode"] == "terminal_complete":
        terminal_summary = _terminal_selection_summary_payload(
            dict(checkpoint.get("selection_state", {}) or {})
        )
        terminal_replay = _replay_selection_history(metrics)
        selected_name = (
            "best.pt"
            if terminal_summary["has_feasible_checkpoint"]
            else "best_infeasible.pt"
        )
        selected_path = run_dir / "checkpoints" / selected_name
        selected_checkpoint = _load_checkpoint(selected_path)
        _validate_replayed_winner(
            selected_checkpoint,
            terminal_replay,
            has_feasible=bool(terminal_summary["has_feasible_checkpoint"]),
        )
        terminal_selected_checkpoint = {
            "path": str(selected_path.resolve()),
            "sha256": sha256_file(selected_path),
            "epoch": int(selected_checkpoint.get("epoch", -1)),
        }
    stage2_input_evidence = None
    predecessor_evidence = None
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage-2 resume lacks Stage-1 authorization")
        stage2_input_path = Path(f"{run_dir}.prerequisites") / (
            "ordered_stage_input.json"
        )
        stage2_input = _validate_stage2_input(
            stage2_input_path,
            stage1_authorization=stage1_authorization,
            config_path=config_path,
        )
        stage2_input_evidence = {
            "path": str(stage2_input_path.resolve()),
            "sha256": sha256_file(stage2_input_path),
            "input_identity": stage2_input["input_identity"],
        }
        authorization = verify_authorization(
            stage1_authorization, purpose="stage2", stage=STAGE1
        )
        predecessor_evidence = {
            "path": str(stage1_authorization.resolve()),
            "sha256": sha256_file(stage1_authorization),
            "authorization_identity": authorization["authorization_identity"],
            "decision_identity": authorization["decision_identity"],
        }
    scripts = [
        {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for path in launcher_paths
    ]
    slurm = {
        "job_id": str(os.environ.get("SLURM_JOB_ID", "")),
        "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        "tasks": int(os.environ.get("SLURM_NTASKS", "0")),
    }
    if not slurm["job_id"] or slurm["nodes"] != 4:
        raise OrderedDecisionError("Exact resume requires a four-node Slurm job")
    without_identity = {
        "schema_name": RESUME_ATTEMPT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "recorded_before_resume": True,
        "original_run_launch": {
            "path": str(launch_path.resolve()),
            "sha256": sha256_file(launch_path),
            "launch_identity": launch["launch_identity"],
        },
        "source": launch_source,
        "resume_source_check": source_check,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved_path": str(resolved_path.resolve()),
            "resolved_sha256": sha256_file(resolved_path),
        },
        "partition": {
            "path": str(Path(partition["dir"])),
            "partition_digest": EXPECTED_PARTITION_DIGEST,
        },
        "last_checkpoint": {
            "path": str(last_checkpoint_path),
            "sha256": before_sha256,
            "epoch": checkpoint_epoch,
            "global_step": int(checkpoint.get("global_step", -1)),
            "identity_evidence": identity_evidence,
            "optimizer_state_present": True,
            "optimizer_reset": False,
        },
        "metrics_before_resume": {
            "path": str(metrics_path.resolve()),
            "existed": metrics_existed,
            "sha256": metrics_before_sha256,
        },
        "metrics_reconciliation": reconciliation,
        "metrics_jsonl_present_before_trainer_reentry": metrics_path.is_file(),
        "metrics_jsonl_sha256": (
            sha256_file(metrics_path)
            if metrics_path.is_file()
            else hashlib.sha256(b"").hexdigest()
        ),
        "resume_progress": progress,
        "terminal_finalization": (
            None
            if terminal_summary is None
            else {
                "required": True,
                "selection_summary_path": str(
                    (run_dir / "selection_summary.json").resolve()
                ),
                "selection_summary_payload": terminal_summary,
                "selection_summary_payload_sha256": _digest_json(terminal_summary),
                "selected_checkpoint": terminal_selected_checkpoint,
                "trainer_reentry": False,
            }
        ),
        "stage1_authorization": predecessor_evidence,
        "stage2_launch_input": stage2_input_evidence,
        "resume_launchers": scripts,
        "slurm": slurm,
        "wandb": "disabled",
        "staged_neighbor_splits": ["train", "val"],
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
    }
    if sha256_file(last_checkpoint_path) != before_sha256:
        raise OrderedDecisionError("last.pt changed during resume preflight")
    payload = {**without_identity, "resume_identity": _digest_json(without_identity)}
    _atomic_json_record(out_file, payload)
    if terminal_summary is not None:
        _atomic_write_terminal_selection_summary(
            run_dir / "selection_summary.json", terminal_summary
        )
    return payload


def _validate_resume_attempt_history(
    *,
    run_dir: Path,
    stage: str,
    launch_path: Path,
    launch: Mapping[str, Any],
    config_path: Path,
    stage1_authorization: Path | None,
) -> list[dict[str, Any]]:
    root = Path(f"{run_dir.resolve()}.prerequisites") / "resume_attempts"
    if not root.exists():
        return []
    if not root.is_dir():
        raise OrderedDecisionError("Resume-attempt path is not a directory")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    if any(not path.is_file() or path.suffix != ".json" for path in entries):
        raise OrderedDecisionError("Resume-attempt directory contains an unknown entry")
    evidence_rows: list[dict[str, Any]] = []
    for path in entries:
        value = _json(path)
        payload = {
            key: child for key, child in value.items() if key != "resume_identity"
        }
        if (
            value.get("schema_name") != RESUME_ATTEMPT_SCHEMA_NAME
            or value.get("stage") != stage
            or value.get("experiment_name") != EXPERIMENTS[stage]
            or value.get("recorded_before_resume") is not True
            or value.get("wandb") != "disabled"
            or value.get("staged_neighbor_splits") != ["train", "val"]
            or value.get("test_data_accessed") is not False
            or value.get("confirmation_manifest_opened") is not False
            or _digest_json(payload) != value.get("resume_identity")
        ):
            raise OrderedDecisionError(f"Invalid resume-attempt evidence: {path}")
        original = dict(value.get("original_run_launch", {}) or {})
        if (
            Path(str(original.get("path", ""))).resolve() != launch_path.resolve()
            or sha256_file(launch_path) != original.get("sha256")
            or launch.get("launch_identity") != original.get("launch_identity")
        ):
            raise OrderedDecisionError("Resume attempt changed the original launch")
        if dict(value.get("source", {}) or {}) != dict(
            launch.get("source", {}) or {}
        ):
            raise OrderedDecisionError("Resume attempt used another source")
        if dict(value.get("resume_source_check", {}) or {}) != dict(
            launch.get("source", {}) or {}
        ):
            raise OrderedDecisionError("Resume attempt lacks live shared-source proof")
        config = dict(value.get("config", {}) or {})
        resolved_path = run_dir / "config.resolved.json"
        if (
            Path(str(config.get("path", ""))).resolve() != config_path.resolve()
            or sha256_file(config_path) != config.get("sha256")
            or Path(str(config.get("resolved_path", ""))).resolve()
            != resolved_path.resolve()
            or sha256_file(resolved_path) != config.get("resolved_sha256")
        ):
            raise OrderedDecisionError("Resume attempt config identity changed")
        checkpoint = dict(value.get("last_checkpoint", {}) or {})
        if (
            Path(str(checkpoint.get("path", ""))).resolve()
            != (run_dir / "checkpoints" / "last.pt").resolve()
            or int(checkpoint.get("epoch", -1)) not in range(1, 5)
            or int(checkpoint.get("global_step", -1)) <= 0
            or checkpoint.get("optimizer_state_present") is not True
            or checkpoint.get("optimizer_reset") is not False
        ):
            raise OrderedDecisionError("Resume attempt did not preserve last.pt state")
        progress = dict(value.get("resume_progress", {}) or {})
        if (
            progress.get("mode")
            not in {"validation_pending", "epoch_complete", "terminal_complete"}
            or int(progress.get("checkpoint_epoch", -1))
            != int(checkpoint.get("epoch", -2))
            or progress.get("checkpoint_metrics_sha256")
            != _digest_json(dict(progress.get("checkpoint_metrics", {}) or {}))
            or progress.get("selection_state_sha256")
            != _digest_json(dict(progress.get("selection_state", {}) or {}))
        ):
            raise OrderedDecisionError("Resume attempt has invalid progress evidence")
        rng_evidence = dict(progress.get("rng_state", {}) or {})
        rank_evidence = list(rng_evidence.get("rank_states", []) or [])
        if int(rng_evidence.get("world_size", -1)) != 4 or [
            int(row.get("rank", -1)) for row in rank_evidence
        ] != [0, 1, 2, 3]:
            raise OrderedDecisionError("Resume attempt lost four-rank RNG evidence")
        reconciliation = dict(value.get("metrics_reconciliation", {}) or {})
        if reconciliation.get("performed") not in {True, False}:
            raise OrderedDecisionError("Resume metrics reconciliation is malformed")
        metrics_present = value.get("metrics_jsonl_present_before_trainer_reentry")
        if not isinstance(metrics_present, bool) or (
            not metrics_present
            and value.get("metrics_jsonl_sha256") != hashlib.sha256(b"").hexdigest()
        ):
            raise OrderedDecisionError("Resume metrics presence evidence is malformed")
        for digest_name in (
            "metrics_jsonl_sha256",
            "resume_identity",
        ):
            digest = value.get(digest_name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise OrderedDecisionError(f"Resume attempt has malformed {digest_name}")
        terminal = value.get("terminal_finalization")
        if progress.get("mode") == "terminal_complete":
            if not isinstance(terminal, Mapping) or terminal.get("required") is not True:
                raise OrderedDecisionError("Terminal resume lacks finalization evidence")
            summary_path = Path(str(terminal.get("selection_summary_path", "")))
            summary_payload = dict(terminal.get("selection_summary_payload", {}) or {})
            selected_checkpoint = dict(terminal.get("selected_checkpoint", {}) or {})
            selected_path = Path(str(selected_checkpoint.get("path", "")))
            if (
                summary_path.resolve() != (run_dir / "selection_summary.json").resolve()
                or _digest_json(summary_payload)
                != terminal.get("selection_summary_payload_sha256")
                or _json(summary_path) != summary_payload
                or selected_path.name
                != (
                    "best.pt"
                    if summary_payload.get("has_feasible_checkpoint") is True
                    else "best_infeasible.pt"
                )
                or sha256_file(selected_path) != selected_checkpoint.get("sha256")
                or terminal.get("trainer_reentry") is not False
            ):
                raise OrderedDecisionError("Terminal resume finalization changed")
        elif terminal is not None:
            raise OrderedDecisionError("Nonterminal resume has terminal evidence")
        slurm = dict(value.get("slurm", {}) or {})
        if int(slurm.get("nodes", -1)) != 4 or not str(slurm.get("job_id", "")):
            raise OrderedDecisionError("Resume attempt did not use four Slurm nodes")
        for script in value.get("resume_launchers", []):
            script = dict(script)
            if sha256_file(Path(str(script.get("path", "")))) != script.get(
                "sha256"
            ):
                raise OrderedDecisionError("A resume launcher changed")
        predecessor = value.get("stage1_authorization")
        if stage == STAGE2:
            if stage1_authorization is None or not isinstance(predecessor, Mapping):
                raise OrderedDecisionError("Stage-2 resume lost predecessor evidence")
            authorization = verify_authorization(
                stage1_authorization, purpose="stage2", stage=STAGE1
            )
            if (
                Path(str(predecessor.get("path", ""))).resolve()
                != stage1_authorization.resolve()
                or sha256_file(stage1_authorization) != predecessor.get("sha256")
                or authorization.get("authorization_identity")
                != predecessor.get("authorization_identity")
            ):
                raise OrderedDecisionError("Stage-2 resume predecessor changed")
            stage2_input = dict(value.get("stage2_launch_input", {}) or {})
            stage2_input_path = Path(f"{run_dir.resolve()}.prerequisites") / (
                "ordered_stage_input.json"
            )
            if (
                Path(str(stage2_input.get("path", ""))).resolve()
                != stage2_input_path.resolve()
                or sha256_file(stage2_input_path) != stage2_input.get("sha256")
                or _json(stage2_input_path).get("input_identity")
                != stage2_input.get("input_identity")
            ):
                raise OrderedDecisionError("Stage-2 resume launch-input changed")
        elif predecessor is not None:
            raise OrderedDecisionError("Stage-1 resume has predecessor evidence")
        evidence_rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "resume_identity": value["resume_identity"],
                "job_id": str(slurm["job_id"]),
                "checkpoint_epoch_before_resume": int(checkpoint["epoch"]),
                "checkpoint_sha256_before_resume": checkpoint["sha256"],
            }
        )
    return evidence_rows


def _decide_stage_validated(
    *,
    stage: str,
    run_dir: Path,
    config_path: Path,
    partition_dir: Path,
    memory_off_dir: Path,
    all_null_dir: Path,
    v2_memory_off_dir: Path,
    v2_config_path: Path,
    v2_checkpoint_path: Path,
    out_dir: Path,
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    predecessor_authorization = None
    if stage == STAGE2:
        if stage1_authorization is None:
            raise OrderedDecisionError("Stage 2 requires Stage-1 authorization")
        predecessor = verify_authorization(
            stage1_authorization, purpose="stage2", stage=STAGE1
        )
        predecessor_authorization = {
            "path": str(stage1_authorization.resolve()),
            "sha256": sha256_file(stage1_authorization),
            "authorization_identity": predecessor["authorization_identity"],
            "decision_identity": predecessor["decision_identity"],
        }
    elif stage1_authorization is not None:
        raise OrderedDecisionError("Stage 1 cannot consume another-stage authorization")
    partition = _validate_partition(partition_dir)
    checkpoint, checkpoint_evidence, _metrics = _validate_run(
        stage=stage, run_dir=run_dir, config_path=config_path
    )
    launch_path = Path(f"{Path(run_dir).resolve()}.prerequisites") / (
        "run_launch_identity.json"
    )
    run_launch = _validate_run_launch(
        launch_path,
        stage=stage,
        config_path=config_path,
        stage1_authorization=stage1_authorization,
    )
    run_source = dict(run_launch.get("source", {}) or {})
    executed_source = _validate_shared_source_checkout(
        SOURCE_ROOT,
        expected_head=str(run_source.get("git_head", "")),
        expected_remote_ref=str(run_source.get("remote_ref", "")),
        expected_remote_head=str(run_source.get("remote_head", "")),
    )
    if executed_source != run_source:
        raise OrderedDecisionError(
            "Ordered decision is not executing the authorized training source"
        )
    run_launch_evidence = {
        "path": str(launch_path),
        "sha256": sha256_file(launch_path),
        "launch_identity": run_launch["launch_identity"],
        "source": run_launch["source"],
        "slurm": run_launch["slurm"],
        "resume_attempts": _validate_resume_attempt_history(
            run_dir=run_dir,
            stage=stage,
            launch_path=launch_path,
            launch=run_launch,
            config_path=config_path,
            stage1_authorization=stage1_authorization,
        ),
    }
    stage2_launch_input = None
    if stage == STAGE2:
        assert stage1_authorization is not None
        stage2_input_path = Path(
            f"{Path(run_dir).resolve()}.prerequisites"
        ) / "ordered_stage_input.json"
        stage2_input = _validate_stage2_input(
            stage2_input_path,
            stage1_authorization=stage1_authorization,
            config_path=config_path,
        )
        stage2_launch_input = {
            "path": str(stage2_input_path),
            "sha256": sha256_file(stage2_input_path),
            "input_identity": stage2_input["input_identity"],
        }
    _ = checkpoint
    memory_off = _load_dev_export(
        memory_off_dir,
        expected_mode="off",
        checkpoint_path=Path(checkpoint_evidence["path"]),
        config_path=config_path,
        partition=partition,
    )
    all_null = _load_dev_export(
        all_null_dir,
        expected_mode="all_null",
        checkpoint_path=Path(checkpoint_evidence["path"]),
        config_path=config_path,
        partition=partition,
    )
    if sha256_file(v2_checkpoint_path) != EXPECTED_V2_SHA256:
        raise OrderedDecisionError("Development parity used another v2 checkpoint")
    v2_memory_off = _load_dev_export(
        v2_memory_off_dir,
        expected_mode="not_applicable",
        checkpoint_path=v2_checkpoint_path,
        config_path=v2_config_path,
        partition=partition,
        require_factorized_control=False,
    )
    all_null_evidence = _validate_development_all_null(memory_off, all_null)
    selected_v2_parity = _validate_selected_v2_parity(memory_off, v2_memory_off)
    status = (
        DEVELOPMENT_FEASIBLE
        if checkpoint_evidence["has_feasible_checkpoint"]
        else VALID_INFEASIBLE
    )
    without_identity = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "status": status,
        "integrity_valid": True,
        "development_feasible": status == DEVELOPMENT_FEASIBLE,
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
        "checkpoint": checkpoint_evidence,
        "partition": {
            "dir": str(partition["dir"]),
            "partition_digest": EXPECTED_PARTITION_DIGEST,
            "development_manifest": str(partition["development_manifest"]),
            "development_manifest_sha256": partition["development_manifest_sha256"],
            "development_rows": EXPECTED_DEVELOPMENT_ROWS,
            "development_unique_texts": EXPECTED_DEVELOPMENT_TEXTS,
        },
        "development_integrity": {
            "all_null_vs_memory_off": all_null_evidence,
            "selected_memory_off_vs_v2": selected_v2_parity,
            "memory_off_export_summary_sha256": sha256_file(memory_off["summary_path"]),
            "all_null_export_summary_sha256": sha256_file(all_null["summary_path"]),
            "v2_memory_off_export_summary_sha256": sha256_file(
                v2_memory_off["summary_path"]
            ),
            "v2_checkpoint_sha256": EXPECTED_V2_SHA256,
            "v2_config": {
                "path": str(v2_config_path.resolve()),
                "sha256": sha256_file(v2_config_path),
            },
        },
        "predecessor_authorization": predecessor_authorization,
        "stage2_launch_input": stage2_launch_input,
        "run_launch_identity": run_launch_evidence,
    }
    decision = {**without_identity, "decision_identity": _digest_json(without_identity)}
    purpose = "confirmation" if status == DEVELOPMENT_FEASIBLE else "stage2"
    if stage == STAGE2 and status == VALID_INFEASIBLE:
        purpose = "stop"
    authorization = (
        _authorization_payload(purpose=purpose, stage=stage, decision=decision)
        if purpose != "stop"
        else None
    )
    _write_decision(out_dir, decision, authorization)
    return decision


def decide_stage(
    *,
    stage: str,
    run_dir: Path,
    config_path: Path,
    partition_dir: Path,
    memory_off_dir: Path,
    all_null_dir: Path,
    v2_memory_off_dir: Path,
    v2_config_path: Path,
    v2_checkpoint_path: Path,
    out_dir: Path,
    stage1_authorization: Path | None = None,
) -> dict[str, Any]:
    """Make one terminal decision, persisting integrity failures without auth."""

    try:
        return _decide_stage_validated(
            stage=stage,
            run_dir=run_dir,
            config_path=config_path,
            partition_dir=partition_dir,
            memory_off_dir=memory_off_dir,
            all_null_dir=all_null_dir,
            v2_memory_off_dir=v2_memory_off_dir,
            v2_config_path=v2_config_path,
            v2_checkpoint_path=v2_checkpoint_path,
            out_dir=out_dir,
            stage1_authorization=stage1_authorization,
        )
    except OrderedDecisionError as error:
        without_identity = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "experiment_name": EXPERIMENTS.get(stage),
            "status": INTEGRITY_INVALID,
            "integrity_valid": False,
            "development_feasible": False,
            "error": str(error),
            "test_data_accessed": False,
            "confirmation_manifest_opened": False,
        }
        decision = {
            **without_identity,
            "decision_identity": _digest_json(without_identity),
        }
        _write_integrity_attempt(out_dir, decision)
        return decision


def _write_integrity_attempt(
    canonical_out_dir: Path, decision: Mapping[str, Any]
) -> None:
    """Persist a nonterminal failed audit without occupying the decision path."""

    canonical_out_dir = canonical_out_dir.resolve()
    if canonical_out_dir.exists():
        raise OrderedDecisionError(
            "A terminal ordered decision already exists; refusing another audit"
        )
    identity = str(decision.get("decision_identity", ""))
    payload = {
        key: value for key, value in decision.items() if key != "decision_identity"
    }
    if (
        decision.get("status") != INTEGRITY_INVALID
        or decision.get("integrity_valid") is not False
        or _digest_json(payload) != identity
    ):
        raise OrderedDecisionError("Malformed integrity-invalid audit payload")
    history_root = canonical_out_dir.parent / (
        f"{canonical_out_dir.name}.integrity_invalid_attempts"
    )
    target = history_root / identity
    if target.exists():
        if (
            _json(target / "decision.json") == dict(decision)
            and _json(target / "READY").get("decision_identity") == identity
        ):
            return
        raise OrderedDecisionError("Integrity-invalid audit identity collision")
    history_root.mkdir(parents=True, exist_ok=True)
    building = history_root / f".{identity}.building.{os.getpid()}"
    if building.exists():
        raise OrderedDecisionError(f"Integrity-audit build exists: {building}")
    building.mkdir()
    try:
        (building / "decision.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (building / "READY").write_text(
            json.dumps(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "decision_identity": identity,
                    "status": INTEGRITY_INVALID,
                    "authorized_purpose": None,
                    "terminal": False,
                    "canonical_decision_path_occupied": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(building, target)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise


def _write_decision(
    out_dir: Path,
    decision: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
) -> None:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        existing = _json(out_dir / "decision.json")
        if existing == decision and _json(out_dir / "READY").get(
            "decision_identity"
        ) == decision.get("decision_identity"):
            return
        raise OrderedDecisionError(f"Refusing to overwrite decision directory: {out_dir}")
    building = out_dir.with_name(f".{out_dir.name}.building")
    if building.exists():
        raise OrderedDecisionError(f"Incomplete decision build exists: {building}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        (building / "decision.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if authorization is not None:
            filename = f"authorize_{authorization['purpose']}.json"
            (building / filename).write_text(
                json.dumps(authorization, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        (building / "READY").write_text(
            json.dumps(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "decision_identity": decision["decision_identity"],
                    "status": decision["status"],
                    "authorized_purpose": (
                        authorization.get("purpose") if authorization else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(building, out_dir)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise


def verify_authorization(path: Path, *, purpose: str, stage: str) -> dict[str, Any]:
    path = path.resolve()
    value = _json(path)
    if (
        value.get("schema_name") != AUTHORIZATION_SCHEMA_NAME
        or int(value.get("schema_version", -1)) != SCHEMA_VERSION
        or value.get("purpose") != purpose
        or value.get("stage") != stage
        or value.get("experiment_name") != EXPERIMENTS[stage]
    ):
        raise OrderedDecisionError("Authorization type/stage does not match request")
    payload = {key: child for key, child in value.items() if key != "authorization_identity"}
    if _digest_json(payload) != value.get("authorization_identity"):
        raise OrderedDecisionError("Authorization identity is invalid")
    decision_path = path.parent / "decision.json"
    ready_path = path.parent / "READY"
    decision = _json(decision_path)
    ready = _json(ready_path)
    if (
        decision.get("decision_identity") != value.get("decision_identity")
        or ready.get("decision_identity") != value.get("decision_identity")
    ):
        raise OrderedDecisionError("Authorization is detached from its decision")
    checkpoint = dict(value.get("checkpoint", {}) or {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise OrderedDecisionError("Authorized checkpoint changed after decision")
    for filename, key in (
        ("metrics.jsonl", "metrics_jsonl_sha256"),
        ("selection_summary.json", "selection_summary_sha256"),
        ("config.resolved.json", "resolved_config_sha256"),
    ):
        artifact = checkpoint_path.parents[1] / filename
        if sha256_file(artifact) != checkpoint.get(key):
            raise OrderedDecisionError(f"Authorized run artifact changed: {filename}")
    return value


def spend_confirmation(
    *,
    authorization_path: Path,
    stage: str,
    marker_path: Path,
    allow_matching_existing: bool = False,
) -> dict[str, Any]:
    """Atomically spend the holdout, or verify an exact continuation marker.

    ``allow_matching_existing`` is deliberately narrower than a generic resume:
    it accepts only the byte-equivalent semantic payload derived from the same
    live authorization.  It never replaces or updates an existing marker.
    """

    authorization = verify_authorization(
        authorization_path, purpose="confirmation", stage=stage
    )
    marker_path = marker_path.resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload_without_identity = {
        "schema_name": HOLDOUT_SPEND_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "experiment_name": EXPERIMENTS[stage],
        "authorization_path": str(authorization_path.resolve()),
        "authorization_identity": authorization["authorization_identity"],
        "checkpoint_sha256": authorization["checkpoint"]["sha256"],
        "partition_digest": authorization["partition"]["partition_digest"],
        "confirmation_holdout_spent": True,
        "test_data_accessed": False,
    }
    payload = {
        **payload_without_identity,
        "spend_identity": _digest_json(payload_without_identity),
    }
    try:
        return _atomic_json_record(
            marker_path,
            payload,
            allow_matching_existing=allow_matching_existing,
        )
    except OrderedDecisionError as error:
        if marker_path.exists():
            if allow_matching_existing:
                raise OrderedDecisionError(
                    "Existing confirmation-spend marker belongs to another "
                    "authorization or has been modified"
                ) from error
            raise OrderedDecisionError(
                f"Confirmation holdout was already spent: {marker_path}"
            ) from error
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    decide = subparsers.add_parser("decide")
    decide.add_argument("--stage", choices=STAGES, required=True)
    decide.add_argument("--run_dir", type=Path, required=True)
    decide.add_argument("--config", type=Path, required=True)
    decide.add_argument("--partition_dir", type=Path, required=True)
    decide.add_argument("--memory_off_dir", type=Path, required=True)
    decide.add_argument("--all_null_dir", type=Path, required=True)
    decide.add_argument("--v2_memory_off_dir", type=Path, required=True)
    decide.add_argument("--v2_config", type=Path, required=True)
    decide.add_argument("--v2_checkpoint", type=Path, required=True)
    decide.add_argument("--out_dir", type=Path, required=True)
    decide.add_argument("--stage1_authorization", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--purpose", choices=("stage2", "confirmation"), required=True)
    verify.add_argument("--stage", choices=STAGES, required=True)
    verify.add_argument("--authorization", type=Path, required=True)

    spend = subparsers.add_parser("spend-confirmation")
    spend.add_argument("--stage", choices=STAGES, required=True)
    spend.add_argument("--authorization", type=Path, required=True)
    spend.add_argument("--marker", type=Path, required=True)
    spend.add_argument(
        "--allow_matching_existing",
        action="store_true",
        help=(
            "permit an exact continuation only when the existing immutable "
            "marker matches this authorization"
        ),
    )
    record = subparsers.add_parser("record-stage2-input")
    record.add_argument("--authorization", type=Path, required=True)
    record.add_argument("--out_file", type=Path, required=True)
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--v2_checkpoint", type=Path, required=True)
    record.add_argument("--source_root", type=Path, required=True)
    record.add_argument("--source_git_head", required=True)
    record.add_argument("--source_remote_ref", required=True)
    record.add_argument("--source_remote_head", required=True)
    launch = subparsers.add_parser("record-run-launch")
    launch.add_argument("--stage", choices=STAGES, required=True)
    launch.add_argument("--out_file", type=Path, required=True)
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--v2_checkpoint", type=Path, required=True)
    launch.add_argument("--source_root", type=Path, required=True)
    launch.add_argument("--source_git_head", required=True)
    launch.add_argument("--source_remote_ref", required=True)
    launch.add_argument("--source_remote_head", required=True)
    launch.add_argument("--launcher", type=Path, action="append", required=True)
    launch.add_argument("--stage1_authorization", type=Path)
    resume = subparsers.add_parser("record-run-resume")
    resume.add_argument("--stage", choices=STAGES, required=True)
    resume.add_argument("--run_dir", type=Path, required=True)
    resume.add_argument("--out_file", type=Path, required=True)
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--partition_dir", type=Path, required=True)
    resume.add_argument("--last_checkpoint", type=Path, required=True)
    resume.add_argument("--source_root", type=Path, required=True)
    resume.add_argument("--source_git_head", required=True)
    resume.add_argument("--source_remote_ref", required=True)
    resume.add_argument("--source_remote_head", required=True)
    resume.add_argument("--launcher", type=Path, action="append", required=True)
    resume.add_argument("--stage1_authorization", type=Path)
    precheckpoint = subparsers.add_parser("record-precheckpoint-retry")
    precheckpoint.add_argument("--stage", choices=STAGES, required=True)
    precheckpoint.add_argument("--run_dir", type=Path, required=True)
    precheckpoint.add_argument("--out_file", type=Path, required=True)
    precheckpoint.add_argument("--config", type=Path, required=True)
    precheckpoint.add_argument("--partition_dir", type=Path, required=True)
    precheckpoint.add_argument("--lease", type=Path, required=True)
    precheckpoint.add_argument("--lease_attestation", type=Path, required=True)
    precheckpoint.add_argument("--source_root", type=Path, required=True)
    precheckpoint.add_argument("--source_git_head", required=True)
    precheckpoint.add_argument("--source_remote_ref", required=True)
    precheckpoint.add_argument("--source_remote_head", required=True)
    precheckpoint.add_argument("--stage1_authorization", type=Path)
    acquire = subparsers.add_parser("acquire-execution-lease")
    acquire.add_argument(
        "--purpose",
        choices=("training", "confirmation", "decision", "diagnostic"),
        required=True,
    )
    acquire.add_argument("--stage", choices=STAGES, required=True)
    acquire.add_argument("--lease", type=Path, required=True)
    acquire.add_argument("--out_file", type=Path, required=True)
    acquire.add_argument("--source_git_head", required=True)
    acquire.add_argument("--slurm_job_id", required=True)
    acquire.add_argument("--binding_identity", required=True)
    release = subparsers.add_parser("release-execution-lease")
    release.add_argument("--lease", type=Path, required=True)
    release.add_argument("--attestation", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "decide":
        result = decide_stage(
            stage=args.stage,
            run_dir=args.run_dir,
            config_path=args.config,
            partition_dir=args.partition_dir,
            memory_off_dir=args.memory_off_dir,
            all_null_dir=args.all_null_dir,
            v2_memory_off_dir=args.v2_memory_off_dir,
            v2_config_path=args.v2_config,
            v2_checkpoint_path=args.v2_checkpoint,
            out_dir=args.out_dir,
            stage1_authorization=args.stage1_authorization,
        )
    elif args.command == "verify":
        result = verify_authorization(
            args.authorization, purpose=args.purpose, stage=args.stage
        )
    elif args.command == "spend-confirmation":
        result = spend_confirmation(
            authorization_path=args.authorization,
            stage=args.stage,
            marker_path=args.marker,
            allow_matching_existing=args.allow_matching_existing,
        )
    elif args.command == "record-stage2-input":
        result = record_stage2_input(
            authorization_path=args.authorization,
            out_file=args.out_file,
            stage2_config_path=args.config,
            v2_checkpoint_path=args.v2_checkpoint,
            source_root=args.source_root,
            source_git_head=args.source_git_head,
            source_remote_ref=args.source_remote_ref,
            source_remote_head=args.source_remote_head,
        )
    elif args.command == "record-run-launch":
        result = record_run_launch(
            stage=args.stage,
            out_file=args.out_file,
            config_path=args.config,
            v2_checkpoint_path=args.v2_checkpoint,
            source_root=args.source_root,
            source_git_head=args.source_git_head,
            source_remote_ref=args.source_remote_ref,
            source_remote_head=args.source_remote_head,
            launcher_paths=args.launcher,
            stage1_authorization=args.stage1_authorization,
        )
    elif args.command == "record-run-resume":
        result = record_run_resume(
            stage=args.stage,
            run_dir=args.run_dir,
            out_file=args.out_file,
            config_path=args.config,
            partition_dir=args.partition_dir,
            last_checkpoint_path=args.last_checkpoint,
            source_root=args.source_root,
            source_git_head=args.source_git_head,
            source_remote_ref=args.source_remote_ref,
            source_remote_head=args.source_remote_head,
            launcher_paths=args.launcher,
            stage1_authorization=args.stage1_authorization,
        )
    elif args.command == "record-precheckpoint-retry":
        result = record_precheckpoint_retry(
            stage=args.stage,
            run_dir=args.run_dir,
            out_file=args.out_file,
            config_path=args.config,
            partition_dir=args.partition_dir,
            lease_path=args.lease,
            lease_attestation_path=args.lease_attestation,
            source_root=args.source_root,
            source_git_head=args.source_git_head,
            source_remote_ref=args.source_remote_ref,
            source_remote_head=args.source_remote_head,
            stage1_authorization=args.stage1_authorization,
        )
    elif args.command == "acquire-execution-lease":
        result = acquire_execution_lease(
            lease_path=args.lease,
            attestation_path=args.out_file,
            purpose=args.purpose,
            stage=args.stage,
            source_git_head=args.source_git_head,
            slurm_job_id=args.slurm_job_id,
            binding_identity=args.binding_identity,
        )
    else:
        result = release_execution_lease(
            lease_path=args.lease,
            attestation_path=args.attestation,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "decide" and result.get("status") == INTEGRITY_INVALID:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
