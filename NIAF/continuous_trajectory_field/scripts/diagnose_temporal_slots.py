"""Diagnose temporal-slot organization and sentence-memory use on CSL-Daily.

This entry point exposes two explicit, provenance-checked profiles: the
completed epoch-2 duration-weight-0.05 Phase-A experiment, and the locked
``best.pt`` selected by the full Phase-A' motion-contrast run.  It is a
read-only validation audit: it never updates weights, never reads the test
split, and never initializes W&B.

The command has two independently promoted stages.  ``smoke`` exercises the
full pipeline on eight validation examples; ``full`` audits all validation rows
and runs the causal intervention sweep on a deterministic 128-text subset.
Interrupted passive batches and causal queries resume from identity-bound NPY
memmaps and atomic progress shards.  A ``READY`` marker is written only after
inference, causal probes, artifact validation, plots, and the report all
complete successfully.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from flow.evaluate.dtw_mpjpe_t2m_default import FACE_LANDMARK_COUNT
from flow.latent_codec import LatentMotionCodec
from flow.smplx_features import (
    COMPACT6D_EXPRESSION,
    COMPACT6D_JAW,
    COMPACT6D_LEFT_HAND,
    COMPACT6D_RIGHT_HAND,
    COMPACT6D_UPPER_BODY,
)
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import (
    ContinuousSignDataset,
    collate_continuous_sign,
)
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_fk,
    build_text_encoder,
    encode_batch_text,
)
from NIAF.continuous_trajectory_field.models import (
    build_continuous_trajectory_field,
)
from NIAF.continuous_trajectory_field.models.hierarchical_field import (
    LOCAL_PART_SLICES,
)
from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryBatch,
    motion_only_shuffle_sentence_memory_batch,
    normalize_sentence_text,
    sentence_text_hash,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    build_sentence_memory_provider,
    retrieve_sentence_memory,
    sentence_memory_behavior_identity,
    sentence_memory_forward_kwargs,
    sentence_memory_objective_identity,
    sentence_memory_query_ids,
    sentence_memory_resume_identity,
    set_seed,
    set_sentence_memory_provider_epoch_from_checkpoint,
    validate_checkpoint_contract,
    validate_sentence_memory_checkpoint_identity,
)
from NIAF.continuous_trajectory_field.scripts.diagnose_sentence_retrieval import (
    bank_latent_statistics,
    normalized_time_latent_rmse,
)
from NIAF.continuous_trajectory_field.temporal_slot_diagnostics import (
    DiagnosticCapture,
    ModuleOutputIntervention,
    analytic_retrieval_prior,
    cluster_bootstrap_interval,
    compute_locality_metrics,
    compute_slot_metrics,
    compute_text_attention_metrics,
    conditional_candidate_metrics,
    deterministic_permutations,
    deterministic_rademacher,
    holm_adjust,
    jensen_shannon_divergence,
    normalized_text_attention_position_ranks,
    rank_correlation_last_dim,
    select_stratified_queries,
    stable_int_seed,
)
from NIAF.oracle_smplx_field.geometry.smplx_fk import (
    default_joint_parts_torch,
)


SCHEMA_NAME = "signtrajfield_temporal_slot_diagnostics"
SCHEMA_VERSION = 1
RESUME_SCHEMA_NAME = "signtrajfield_temporal_slot_diagnostics_progress"
RESUME_SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "f47176153726e1d1890b2860e0cfbbd73677d4adab7124ab14ae95434b286be7"
)
EXPECTED_EPOCH = 2
EXPECTED_VALIDATION_ROWS = 1_077
EXPECTED_DURATION_WEIGHT = 0.05
EXPECTED_SCORE_TEMPERATURE = 0.10
EXPECTED_K = 8
EXPECTED_DATA_DIR = Path(
    "/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx"
)
EXPECTED_VALIDATION_MANIFEST = EXPECTED_DATA_DIR / "meta/manifest_val.jsonl"
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "2d443adf2cd489709e19d3e55b74128e7f4470b3413692e587e9dd2ed521dabe"
)
PART_NAMES = ("body", "left_hand", "right_hand", "face")
MEMORY_CONDITIONS = ("correct", "shuffled", "motion_only_shuffle")
INTERVENTION_STAGES = ("planner", "fused")
BRANCH_NAMES = ("coarse", "global", "local")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_CONFIG = (
    PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/configs"
    / "csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2.yaml"
)
EXPECTED_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments/NIAF/continuous_trajectory_field"
    / EXPECTED_EXPERIMENT
    / "checkpoints/epoch0002.pt"
)
EXPECTED_OUTPUT = (
    PROJECT_ROOT
    / "experiments/NIAF/continuous_trajectory_field"
    / EXPECTED_EXPERIMENT
    / "evaluation/epoch0002_validation_slot_diagnostics"
)
LEGACY_PROFILE_NAME = "dw005_epoch2"
PHASE_A_PRIME_PROFILE_NAME = "phase_a_motion_contrast_v1"
PHASE_A_PRIME_EXPERIMENT = (
    "csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1"
)
PHASE_A_PRIME_CONFIG = (
    PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/configs"
    / f"{PHASE_A_PRIME_EXPERIMENT}.yaml"
)
PHASE_A_PRIME_RUN_DIR = (
    PROJECT_ROOT
    / "experiments/NIAF/continuous_trajectory_field"
    / PHASE_A_PRIME_EXPERIMENT
)
PHASE_A_PRIME_CHECKPOINT = PHASE_A_PRIME_RUN_DIR / "checkpoints/best.pt"
PHASE_A_PRIME_OUTPUT = (
    PHASE_A_PRIME_RUN_DIR / "evaluation/locked_validation_slot_diagnostics"
)


@dataclasses.dataclass(frozen=True)
class DiagnosticProfile:
    name: str
    experiment_name: str
    config: Path
    checkpoint: Path
    output: Path
    fixed_checkpoint_sha256: str | None = None
    fixed_epoch: int | None = None
    require_locked_full_run: bool = False


DIAGNOSTIC_PROFILES = {
    LEGACY_PROFILE_NAME: DiagnosticProfile(
        name=LEGACY_PROFILE_NAME,
        experiment_name=EXPECTED_EXPERIMENT,
        config=EXPECTED_CONFIG,
        checkpoint=EXPECTED_CHECKPOINT,
        output=EXPECTED_OUTPUT,
        fixed_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        fixed_epoch=EXPECTED_EPOCH,
    ),
    PHASE_A_PRIME_PROFILE_NAME: DiagnosticProfile(
        name=PHASE_A_PRIME_PROFILE_NAME,
        experiment_name=PHASE_A_PRIME_EXPERIMENT,
        config=PHASE_A_PRIME_CONFIG,
        checkpoint=PHASE_A_PRIME_CHECKPOINT,
        output=PHASE_A_PRIME_OUTPUT,
        require_locked_full_run=True,
    ),
}
DIAGNOSTIC_RUNNER_SOURCE = Path(__file__).resolve()
DIAGNOSTIC_CORE_SOURCE = (
    PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/temporal_slot_diagnostics.py"
)
DIAGNOSTIC_DEPENDENCY_SOURCES = {
    "sentence_memory.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/sentence_memory.py",
    "sentence_memory_trajectory_hypernetwork.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/sentence_memory_trajectory_hypernetwork.py",
    "dual_mode_trajectory_hypernetwork.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/dual_mode_trajectory_hypernetwork.py",
    "trajectory_hypernetwork.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/trajectory_hypernetwork.py",
    "hierarchical_field.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/hierarchical_field.py",
    "trajectory_instance.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/trajectory_instance.py",
    "modulated_siren.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/modulated_siren.py",
    "trajectory_models_init.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/models/__init__.py",
    "trajectory_trainer.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/scripts/train_continuous_trajectory_field.py",
    "retrieval_diagnostic_helpers.py": PROJECT_ROOT
    / "NIAF/continuous_trajectory_field/scripts/diagnose_sentence_retrieval.py",
    "residual_flow_helpers.py": PROJECT_ROOT
    / "NIAF/continuous_sign_field/scripts/train_residual_flow.py",
    "continuous_sign_data.py": PROJECT_ROOT / "NIAF/continuous_sign_field/data.py",
    "continuous_sign_config.py": PROJECT_ROOT / "NIAF/continuous_sign_field/config.py",
    "continuous_sign_meta_implicit.py": PROJECT_ROOT
    / "NIAF/continuous_sign_field/models/meta_implicit.py",
    "smplx_fk.py": PROJECT_ROOT / "NIAF/oracle_smplx_field/geometry/smplx_fk.py",
    "adapter_prior.py": PROJECT_ROOT / "flow/adapter_prior.py",
    "flow_dataset_init.py": PROJECT_ROOT / "flow/dataset/__init__.py",
    "upper_smplx_dataset.py": PROJECT_ROOT / "flow/dataset/upper_smplx.py",
    "dtw_keypoint_layout.py": PROJECT_ROOT
    / "flow/evaluate/dtw_mpjpe_t2m_default.py",
    "ndtw_smplx_keypoints.py": PROJECT_ROOT
    / "flow/evaluate/ndtw_smplx_keypoints.py",
    "flow_render.py": PROJECT_ROOT / "flow/render.py",
    "text_encoder.py": PROJECT_ROOT / "flow/text_encoder.py",
    "latent_codec.py": PROJECT_ROOT / "flow/latent_codec.py",
    "temporal_vae.py": PROJECT_ROOT / "flow/VAE/model.py",
    "smplx_features.py": PROJECT_ROOT / "flow/smplx_features.py",
}
READY_REQUIRED_FILES = (
    "summary.json",
    "provenance.json",
    "report.md",
    "query_metrics.csv",
    "slot_metrics.csv",
    "text_attention_metrics.csv",
    "part_attention_metrics.csv",
    "candidate_metrics.csv",
    "causal_metrics.csv",
    "causal_selection.csv",
    "representative_examples.json",
    "arrays/manifest.json",
    "plots/slot_cosine.png",
    "plots/text_attention.png",
    "plots/candidate_attention.png",
    "plots/gate_null_timelines.png",
    "plots/causal_influence.png",
)
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure temporal-slot collapse, slot-to-token organization, "
            "sentence-memory selectivity, and causal slot locality for the "
            "provenance-locked CSL-Daily duration-weight-0.05 checkpoint."
        )
    )
    parser.add_argument(
        "--profile",
        choices=tuple(DIAGNOSTIC_PROFILES),
        default=LEGACY_PROFILE_NAME,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out_dir", "--out-dir", type=Path)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument(
        "--perturb_batch_size", "--perturb-batch-size", type=int, default=128
    )
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--text_device", "--text-device", choices=("cpu",), default="cpu")
    parser.add_argument("--query_points", "--query-points", type=int, default=128)
    parser.add_argument("--causal_size", "--causal-size", type=int, default=128)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--permutation_samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fk_batch_size", "--fk-batch-size", type=int, default=512)
    parser.add_argument("--verify_hashes", "--verify-hashes", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an identity-matching partial passive/causal sweep, or return "
            "successfully when an identity-matching READY result exists."
        ),
    )
    args = parser.parse_args()
    profile = DIAGNOSTIC_PROFILES[args.profile]
    args.config = args.config or profile.config
    args.checkpoint = args.checkpoint or profile.checkpoint
    args.out_dir = args.out_dir or profile.output
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_file_hashes(sources: Mapping[str, Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, path in sorted(sources.items()):
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
            raise RuntimeError(f"Behavior dependency escapes the project: {resolved}")
        output[str(name)] = _sha256(resolved)
    return output


def _diagnostic_source_hashes() -> dict[str, str]:
    """Bind every explicit module that can alter diagnostic inference."""

    return _source_file_hashes(
        {
            "diagnose_temporal_slots.py": DIAGNOSTIC_RUNNER_SOURCE,
            "temporal_slot_diagnostics.py": DIAGNOSTIC_CORE_SOURCE,
            **DIAGNOSTIC_DEPENDENCY_SOURCES,
        }
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON control file without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    # Some CIFS servers do not support directory fsync.  The
                    # file itself was fsynced before the atomic replacement.
                    pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _git_identity() -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                arguments,
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = command("git", "status", "--porcelain")
    return {
        # Compute nodes cannot resolve this Codex worktree's node-local gitdir.
        # `_git_head()` therefore accepts the submit-host identity exported by
        # the launcher, while the explicit source-file hashes bind the bytes
        # that actually define this diagnostic.
        "commit": _git_head(),
        "branch": command("git", "branch", "--show-current"),
        "tracked_worktree_clean": status == "" if status is not None else None,
    }


def _git_head() -> str:
    value = str(os.environ.get("SIGNTRAJ_SOURCE_GIT_HEAD", "")).strip()
    if not value:
        try:
            value = subprocess.check_output(
                ("git", "rev-parse", "HEAD"),
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Cannot bind diagnostic identity to git HEAD; submit through "
                "the launcher with SOURCE_GIT_HEAD exported from the submit host"
            ) from error
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise RuntimeError(f"Unexpected git HEAD identity: {value!r}")
    return value.lower()


def _selected_profile(args: argparse.Namespace) -> DiagnosticProfile:
    name = str(getattr(args, "profile", LEGACY_PROFILE_NAME))
    try:
        return DIAGNOSTIC_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown diagnostic profile: {name!r}") from error


def _validate_arguments(
    args: argparse.Namespace, profile: DiagnosticProfile | None = None
) -> None:
    profile = profile or _selected_profile(args)
    for name in (
        "batch_size",
        "perturb_batch_size",
        "num_workers",
        "query_points",
        "causal_size",
        "directions",
        "bootstrap_samples",
        "permutation_samples",
        "fk_batch_size",
    ):
        value = int(getattr(args, name))
        if name == "num_workers":
            if value < 0:
                raise ValueError("num_workers must be non-negative")
        elif value < 1:
            raise ValueError(f"{name} must be positive")
    if args.query_points < 2:
        raise ValueError("query_points must be at least two")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isclose(float(args.epsilon), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The approved diagnostic fixes epsilon at 0.10 (and 0.05)")
    if args.stage == "full" and int(args.causal_size) != 128:
        raise ValueError("The approved full audit requires causal_size=128")
    if args.stage == "full" and int(args.directions) != 4:
        raise ValueError("The approved full audit requires four directions")
    if args.stage == "full" and int(args.query_points) != 128:
        raise ValueError("The approved full audit requires 128 query points")

    online_values = {"online", "run"}
    if str(os.environ.get("WANDB_MODE", "")).strip().lower() in online_values:
        raise RuntimeError("Temporal-slot diagnostics refuse online W&B mode")
    if str(os.environ.get("WANDB_DISABLED", "")).strip().lower() in {"0", "false"}:
        raise RuntimeError("Temporal-slot diagnostics require W&B to be disabled")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    config = args.config.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if config != profile.config.resolve(strict=True):
        raise ValueError(
            f"Profile {profile.name!r} is pinned to config {profile.config}"
        )
    if checkpoint != profile.checkpoint.resolve(strict=True):
        raise ValueError(
            f"Profile {profile.name!r} is pinned to checkpoint "
            f"{profile.checkpoint}"
        )
    checkpoint_sha = _sha256(checkpoint)
    if (
        profile.fixed_checkpoint_sha256 is not None
        and checkpoint_sha != profile.fixed_checkpoint_sha256
    ):
        raise RuntimeError(
            "Pinned checkpoint hash mismatch: "
            f"expected={profile.fixed_checkpoint_sha256}, actual={checkpoint_sha}"
        )
    if args.stage == "full" and args.out_dir.resolve() != profile.output.resolve():
        raise ValueError(f"Full output must be written to {profile.output}")


def _validate_config(
    cfg: Mapping[str, Any], profile: DiagnosticProfile | None = None
) -> None:
    profile = profile or DIAGNOSTIC_PROFILES[LEGACY_PROFILE_NAME]
    if str(cfg.get("experiment_name")) != profile.experiment_name:
        raise ValueError("Unexpected experiment_name in pinned configuration")
    if str(cfg.get("data", {}).get("train_split")) != "train":
        raise ValueError("The sentence bank must remain train-only")
    if str(cfg.get("data", {}).get("val_split", "val")) != "val":
        raise ValueError("This diagnostic permits only the validation split")
    configured_data_dir = Path(str(cfg.get("data", {}).get("data_dir", "")))
    if configured_data_dir.resolve() != EXPECTED_DATA_DIR:
        raise ValueError(
            f"Diagnostic data_dir must be canonical CSL-Daily: {EXPECTED_DATA_DIR}"
        )
    configured_manifest = cfg.get("data", {}).get("val_manifest_path")
    manifest_path = (
        Path(str(configured_manifest))
        if configured_manifest
        else configured_data_dir / "meta/manifest_val.jsonl"
    )
    if manifest_path.resolve() != EXPECTED_VALIDATION_MANIFEST:
        raise ValueError(
            "Diagnostic validation manifest must be the canonical CSL-Daily "
            f"manifest: {EXPECTED_VALIDATION_MANIFEST}"
        )
    if str(cfg.get("model", {}).get("type")) != (
        "sentence_memory_continuous_trajectory_field"
    ):
        raise ValueError("The pinned config is not a v3 sentence-memory model")
    memory = cfg.get("sentence_memory", {})
    if memory.get("neighbor_files"):
        raise ValueError(
            "Validation-only diagnostic refuses sentence_memory.neighbor_files "
            "overrides; train/val tables must come from the split-filtered local bank"
        )
    checks = {
        "duration_weight": (float(memory.get("duration_weight", -1)), EXPECTED_DURATION_WEIGHT),
        "score_temperature": (
            float(memory.get("score_temperature", -1)),
            EXPECTED_SCORE_TEMPERATURE,
        ),
        "k": (int(memory.get("k", -1)), EXPECTED_K),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"sentence_memory.{name}={actual!r}; expected {expected!r}")
    if str(cfg.get("conditioning", {}).get("word_prior_train_mode")) != "off":
        raise ValueError("Word prior must be disabled for this attribution audit")


def _load_checkpoint(
    path: Path, *, expected_epoch: int | None = EXPECTED_EPOCH
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("Checkpoint does not contain a model state dictionary")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch < 1:
        raise RuntimeError(f"Checkpoint has invalid epoch={checkpoint_epoch!r}")
    if expected_epoch is not None and checkpoint_epoch != expected_epoch:
        raise RuntimeError(
            f"Checkpoint epoch={checkpoint_epoch!r}; expected {expected_epoch}"
        )
    return checkpoint


_PHASE_A_PRIME_DYNAMIC_PARTITION_FIELDS = (
    "partition_digest",
    "resolved_artifact",
    "development_row_count",
    "exact_seen_row_count",
    "exact_seen_text_count",
    "exact_seen_evaluated_during_training",
    "confirmation_evaluated_during_training",
)


def _static_phase_a_prime_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only attested runtime fields before exact config comparison."""

    payload = copy.deepcopy(dict(cfg))
    memory_cfg = payload.get("sentence_memory")
    if isinstance(memory_cfg, dict):
        memory_cfg.pop("resolved_identity", None)
        memory_cfg.pop("resolved_behavior_identity", None)
    safety_cfg = payload.get("sentence_memory_safety")
    if isinstance(safety_cfg, dict):
        safety_cfg.pop("v2_to_v3_text_only_parity", None)
    partition_cfg = payload.get("validation_text_partition")
    if isinstance(partition_cfg, dict):
        for field in _PHASE_A_PRIME_DYNAMIC_PARTITION_FIELDS:
            partition_cfg.pop(field, None)
    out_dir = str(dict(payload.get("output", {}) or {}).get("out_dir", ""))
    if Path(out_dir).name != PHASE_A_PRIME_EXPERIMENT:
        raise RuntimeError("Checkpoint output is not the full Phase-A' run")
    payload.setdefault("output", {})["out_dir"] = PHASE_A_PRIME_EXPERIMENT
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read locked-run artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Locked-run artifact is not an object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"Cannot read locked-run metrics {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON in {path} at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"Non-object metrics row in {path} at line {line_number}")
        rows.append(row)
    return rows


def _validate_phase_a_prime_locked_run(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    config_path: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Prove that ``best.pt`` came from the completed four-rank full run.

    This deliberately consumes only checkpoint and run-control artifacts.  It
    does not require confirmation results and never opens a test manifest or a
    test neighbor table.
    """

    run_dir = (run_dir or PHASE_A_PRIME_RUN_DIR).resolve()
    if run_dir.name != PHASE_A_PRIME_EXPERIMENT:
        raise RuntimeError("Phase-A' run directory has the wrong experiment name")
    resolved_checkpoint = checkpoint_path.resolve(strict=True)
    if resolved_checkpoint != (run_dir / "checkpoints/best.pt").resolve():
        raise RuntimeError(
            "Phase-A' diagnostics accept only the locked RUN_DIR/checkpoints/best.pt"
        )
    if config_path.resolve(strict=True) != PHASE_A_PRIME_CONFIG.resolve(strict=True):
        raise RuntimeError("Phase-A' diagnostics require the exact full-run config")

    cfg = copy.deepcopy(dict(checkpoint.get("config", {}) or {}))
    expected_cfg = load_config(config_path)
    expected_cfg["device"] = "cuda"
    checkpoint_partition_cfg = dict(
        cfg.get("validation_text_partition", {}) or {}
    )
    expected_partition_cfg = expected_cfg.setdefault(
        "validation_text_partition", {}
    )
    for field in _PHASE_A_PRIME_DYNAMIC_PARTITION_FIELDS:
        if field in checkpoint_partition_cfg:
            expected_partition_cfg[field] = copy.deepcopy(
                checkpoint_partition_cfg[field]
            )
    if _static_phase_a_prime_config(cfg) != _static_phase_a_prime_config(expected_cfg):
        raise RuntimeError("Selected checkpoint config differs from the full-run config")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 1 or epoch > 4:
        raise RuntimeError("Selected Phase-A' epoch must be in [1, 4]")

    train_cfg = dict(cfg.get("train", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    eval_cfg = dict(cfg.get("eval", {}) or {})
    if (
        int(train_cfg.get("epochs", -1)) != 4
        or int(train_cfg.get("early_stopping_patience", -1)) != 2
        or int(train_cfg.get("early_stopping_min_epochs", -1)) != 2
    ):
        raise RuntimeError("Selected checkpoint lacks the approved early-stop contract")
    if any(
        int(value or 0) != 0
        for value in (
            data_cfg.get("limit_train", 0),
            data_cfg.get("limit_val", 0),
            train_cfg.get("max_train_batches", 0),
            eval_cfg.get("max_batches", 0),
        )
    ):
        raise RuntimeError("Smoke or subset checkpoints cannot enter full diagnostics")
    if not bool(train_cfg.get("freeze_base", False)) or train_cfg.get(
        "unfreeze_base_prefixes"
    ):
        raise RuntimeError("Selected checkpoint is not frozen-base Phase A")
    if bool(cfg.get("sentence_memory_safety", {}).get("phase_b", {}).get("enabled")):
        raise RuntimeError("Phase-B checkpoints cannot enter Phase-A' diagnostics")

    metrics = dict(checkpoint.get("metrics", {}) or {})
    if float(metrics.get("selection_feasible", 0.0)) != 1.0:
        raise RuntimeError("Selected checkpoint did not pass development selection")
    required_namespaces = (
        "val_text_only/",
        "val_sentence_memory/",
        "val_shuffled_sentence_memory/",
        "val_motion_shuffled_sentence_memory/",
    )
    if any(not any(str(key).startswith(prefix) for key in metrics) for prefix in required_namespaces):
        raise RuntimeError("Selected checkpoint lacks all four development modes")
    parity = dict(checkpoint.get("v2_to_v3_text_only_parity", {}) or {})
    if (
        parity.get("passed") is not True
        or float(parity.get("prediction_max_abs", math.inf)) > 1e-7
        or float(parity.get("duration_max_abs", math.inf)) > 1e-7
    ):
        raise RuntimeError("Selected checkpoint lacks strict v2 text-only parity")

    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    partition_digest = partition_cfg.get("partition_digest")
    resolved_partition = partition_cfg.get("resolved_artifact")
    if not isinstance(partition_digest, str) or len(partition_digest) != 64:
        raise RuntimeError("Selected checkpoint lacks a locked partition digest")
    if not isinstance(resolved_partition, dict):
        raise RuntimeError("Selected checkpoint lacks resolved partition provenance")
    resolved_without_digest = {
        key: value for key, value in resolved_partition.items() if key != "partition_digest"
    }
    if _digest_json(resolved_without_digest) != resolved_partition.get(
        "partition_digest"
    ) or resolved_partition.get("partition_digest") != partition_digest:
        raise RuntimeError("Selected checkpoint partition provenance is invalid")
    counts = dict(resolved_partition.get("counts", {}) or {})
    expected_counts = {
        "rows": 1_077,
        "novel_unique_texts": 796,
        "development_unique_texts": 256,
        "confirmation_unique_texts": 540,
        "development_rows": 347,
        "confirmation_rows": 728,
    }
    if any(int(counts.get(name, -1)) != value for name, value in expected_counts.items()):
        raise RuntimeError("Selected checkpoint partition counts are incomplete")
    if partition_cfg.get("confirmation_evaluated_during_training") is not False:
        raise RuntimeError("Checkpoint does not attest that confirmation stayed locked")

    for name, compute in (
        ("objective", sentence_memory_objective_identity),
        ("behavior", sentence_memory_behavior_identity),
        ("resume", sentence_memory_resume_identity),
    ):
        actual = checkpoint.get(f"sentence_memory_{name}_identity")
        if actual != compute(cfg) or actual != compute(expected_cfg):
            raise RuntimeError(f"Selected checkpoint has an invalid {name} identity")
    neighbor_tables = dict(
        dict(checkpoint.get("sentence_memory_identity", {}) or {}).get(
            "neighbor_tables", {}
        )
        or {}
    )
    if set(neighbor_tables) != {"train", "val"}:
        raise RuntimeError(
            "Selected checkpoint must contain exactly train/val neighbor identities"
        )
    rng_state = dict(checkpoint.get("rng_state", {}) or {})
    rank_states = list(rng_state.get("rank_states", []) or [])
    if int(rng_state.get("world_size", -1)) != 4 or sorted(
        int(row.get("rank", -1)) for row in rank_states if isinstance(row, dict)
    ) != [0, 1, 2, 3]:
        raise RuntimeError("Selected checkpoint lacks four-rank full-run RNG state")

    summary_path = run_dir / "selection_summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    resolved_config_path = run_dir / "config.resolved.json"
    summary = _read_json_object(summary_path)
    rows = _read_jsonl_objects(metrics_path)
    resolved_cfg = _read_json_object(resolved_config_path)
    if _static_phase_a_prime_config(resolved_cfg) != _static_phase_a_prime_config(
        expected_cfg
    ):
        raise RuntimeError("Resolved run config differs from the approved full config")
    if summary.get("has_feasible_checkpoint") is not True or summary.get(
        "required"
    ) is not True:
        raise RuntimeError("Full run did not terminate with a required feasible checkpoint")
    early_state = summary.get("early_stopping")
    if not isinstance(early_state, dict) or int(
        early_state.get("validation_count", -1)
    ) < 2:
        raise RuntimeError("Full run has fewer than two development validations")
    completed = [
        row
        for row in rows
        if float(row.get("validation_pending", 1.0)) == 0.0
        and "selection_feasible" in row
    ]
    if len(completed) < 2 or len(completed) != len(rows):
        raise RuntimeError("Full run has incomplete development-validation evidence")
    epochs = [int(row.get("epoch", -1)) for row in completed]
    if epochs != list(range(1, max(epochs) + 1)) or max(epochs) < 2:
        raise RuntimeError("Full-run epoch history is incomplete or noncanonical")
    if not bool(early_state.get("stopped", False)) and max(epochs) != 4:
        raise RuntimeError("Run has neither early-stop nor four-epoch completion evidence")
    for row in completed:
        if any(not any(str(key).startswith(prefix) for key in row) for prefix in required_namespaces):
            raise RuntimeError("A completed epoch lacks a development evaluation mode")
    feasible_rows = [
        row for row in completed if float(row.get("selection_feasible", 0.0)) == 1.0
    ]
    if not feasible_rows:
        raise RuntimeError("Full-run metrics contain no feasible checkpoint")
    best_score = float(summary.get("best_feasible_score", math.nan))
    checkpoint_score = float(metrics.get("selection_score", math.nan))
    metric_best = min(float(row["selection_score"]) for row in feasible_rows)
    if not all(math.isfinite(value) for value in (best_score, checkpoint_score, metric_best)):
        raise RuntimeError("Selected checkpoint score evidence is non-finite")
    if best_score != metric_best or checkpoint_score != best_score:
        raise RuntimeError("Selected checkpoint score disagrees with full-run evidence")
    checkpoint_selection = dict(checkpoint.get("selection_state", {}) or {})
    if float(checkpoint_selection.get("best_feasible_score", math.nan)) != best_score:
        raise RuntimeError("Selected checkpoint lacks matching feasible-selection state")
    selected_rows = [
        row
        for row in feasible_rows
        if int(row.get("epoch", -1)) == epoch
        and float(row.get("selection_score", math.nan)) == best_score
    ]
    if len(selected_rows) != 1:
        raise RuntimeError("best.pt is not the unique selected metrics row")

    return {
        "schema_name": "phase_a_motion_contrast_locked_run",
        "schema_version": 1,
        "run_dir": str(run_dir),
        "checkpoint_epoch": epoch,
        "partition_digest": partition_digest,
        "complete_validation_events": len(completed),
        "terminal_epoch": max(epochs),
        "best_feasible_score": best_score,
        "selection_summary_sha256": _sha256(summary_path),
        "metrics_jsonl_sha256": _sha256(metrics_path),
        "resolved_config_sha256": _sha256(resolved_config_path),
        "objective_identity": checkpoint["sentence_memory_objective_identity"],
        "behavior_identity": checkpoint["sentence_memory_behavior_identity"],
        "resume_identity": checkpoint["sentence_memory_resume_identity"],
    }


def _validate_checkpoint_identity_without_test(
    checkpoint: Mapping[str, Any],
    provider,
    *,
    cfg: Mapping[str, Any],
    text_encoder_identity: Mapping[str, Any],
    source: str,
) -> None:
    """Validate the active val table without ever opening the test table.

    Training checkpoints persist identities for all tables that were staged at
    training time.  Requiring exact dictionary equality here would force the
    provider to open and hash ``neighbors_test.npz``, contrary to this audit's
    validation-only contract.  The shared validator still checks the complete
    behavior and text-encoder contracts; this diagnostic then checks the bank
    ID and every table physically present in its split-filtered local bank.
    """

    validate_sentence_memory_checkpoint_identity(
        checkpoint,
        None,
        source=source,
        cfg=cfg,
        text_encoder_identity=text_encoder_identity,
    )
    persisted = checkpoint.get("sentence_memory_identity")
    if persisted is None and isinstance(checkpoint.get("config"), Mapping):
        persisted = checkpoint["config"].get("sentence_memory", {}).get(
            "resolved_identity"
        )
    if not isinstance(persisted, Mapping):
        raise RuntimeError(f"{source} has no sentence-memory bank identity")
    active = provider.identity
    if str(persisted.get("bank_id")) != str(active.get("bank_id")):
        raise RuntimeError(
            f"{source} sentence bank differs from the active validation bank"
        )
    active_tables = active.get("neighbor_tables")
    persisted_tables = persisted.get("neighbor_tables")
    if not isinstance(active_tables, Mapping) or set(active_tables) != {"train", "val"}:
        raise RuntimeError(
            "Validation diagnostic requires exactly the staged train and val "
            "neighbor tables"
        )
    if not isinstance(persisted_tables, Mapping):
        raise RuntimeError(f"{source} has no persisted sentence-neighbor identities")
    for split in ("train", "val"):
        if persisted_tables.get(split) != active_tables.get(split):
            raise RuntimeError(
                f"{source} {split}-neighbor identity differs from the staged table"
            )


class _ResumeWorkspace:
    """Identity-bound, append-only progress for an interrupted diagnostic.

    Raw tensors live in reopenable NPY memmaps.  Variable-length tabular rows
    are committed as one atomic JSON shard per passive batch or causal query.
    ``progress.json`` advances only after the memmaps are flushed and the shard
    hash is known, so an interrupted unit is simply recomputed and overwritten.
    """

    def __init__(
        self,
        root: Path,
        *,
        expected_identity: Mapping[str, Any],
        passive_total: int,
        causal_total: int,
        resume: bool,
    ) -> None:
        self.root = root
        self.control_root = root / ".resume"
        self.identity_path = self.control_root / "identity.json"
        self.progress_path = self.control_root / "progress.json"
        self.identity_digest = _digest_json(expected_identity)
        self.expected_identity = _jsonable(expected_identity)
        self.passive_total = int(passive_total)
        self.causal_total = int(causal_total)

        if root.exists():
            if not resume:
                raise FileExistsError(
                    f"Incomplete output exists: {root}; pass --resume to continue it"
                )
            self._open_existing()
        else:
            root.mkdir(parents=False)
            self.control_root.mkdir(parents=False)
            identity = {
                "schema_name": RESUME_SCHEMA_NAME,
                "schema_version": RESUME_SCHEMA_VERSION,
                "expected_ready_identity": self.expected_identity,
                "expected_ready_identity_sha256": self.identity_digest,
                "created_at": _utc_now(),
            }
            _atomic_write_json(self.identity_path, identity)
            self.progress = {
                "schema_name": RESUME_SCHEMA_NAME,
                "schema_version": RESUME_SCHEMA_VERSION,
                "expected_ready_identity_sha256": self.identity_digest,
                "passive_total": self.passive_total,
                "causal_total": self.causal_total,
                "passive": {"next_position": 0, "shards": []},
                "causal": {
                    "next_position": 0,
                    "selection_hash": None,
                    "shards": [],
                },
                "attempts": [],
                "status": "running",
                "updated_at": _utc_now(),
            }
            self._record_attempt()

    def _read_json_mapping(self, path: Path, *, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot read {label} at {path}: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} must contain a JSON object: {path}")
        return value

    def _open_existing(self) -> None:
        identity = self._read_json_mapping(
            self.identity_path, label="diagnostic resume identity"
        )
        stored_expected = identity.get("expected_ready_identity")
        stored_digest = identity.get("expected_ready_identity_sha256")
        if (
            identity.get("schema_name") != RESUME_SCHEMA_NAME
            or int(identity.get("schema_version", -1)) != RESUME_SCHEMA_VERSION
            or stored_digest != _digest_json(stored_expected)
            or stored_digest != self.identity_digest
            or stored_expected != self.expected_identity
        ):
            raise RuntimeError(
                "Refusing stale partial diagnostic: resume identity does not "
                "match the active checkpoint/config/bank/manifest/source/settings"
            )
        self.progress = self._read_json_mapping(
            self.progress_path, label="diagnostic resume progress"
        )
        self._validate_progress()
        self._record_attempt()

    def _record_attempt(self) -> None:
        updated = json.loads(json.dumps(self.progress))
        attempts = list(updated.get("attempts", []))
        attempts.append(
            {
                "opened_at": _utc_now(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "pid": os.getpid(),
            }
        )
        updated["attempts"] = attempts
        updated["status"] = "running"
        updated["updated_at"] = _utc_now()
        _atomic_write_json(self.progress_path, updated)
        self.progress = updated

    def _resolve_shard(self, relative: str, *, kind: str) -> Path:
        relative_path = Path(str(relative))
        if relative_path.is_absolute():
            raise RuntimeError("Resume shard paths must be relative")
        path = (self.root / relative_path).resolve()
        required_root = (self.control_root / kind).resolve()
        if not path.is_relative_to(required_root):
            raise RuntimeError(f"Resume {kind} shard escapes its control directory")
        return path

    def _validate_progress(self) -> None:
        value = self.progress
        if (
            value.get("schema_name") != RESUME_SCHEMA_NAME
            or int(value.get("schema_version", -1)) != RESUME_SCHEMA_VERSION
            or value.get("expected_ready_identity_sha256") != self.identity_digest
            or int(value.get("passive_total", -1)) != self.passive_total
            or int(value.get("causal_total", -1)) != self.causal_total
        ):
            raise RuntimeError("Diagnostic resume progress identity/count mismatch")
        for kind, total in (
            ("passive", self.passive_total),
            ("causal", self.causal_total),
        ):
            state = value.get(kind)
            if not isinstance(state, Mapping):
                raise RuntimeError(f"Diagnostic resume progress lacks {kind} state")
            cursor = 0
            shards = state.get("shards")
            if not isinstance(shards, list):
                raise RuntimeError(f"Diagnostic resume {kind} shards must be a list")
            for entry in shards:
                if not isinstance(entry, Mapping):
                    raise RuntimeError(f"Malformed diagnostic resume {kind} shard")
                start = int(entry.get("start", -1))
                end = int(entry.get("end", -1))
                if start != cursor or end <= start or end > total:
                    raise RuntimeError(
                        f"Non-contiguous diagnostic resume {kind} progress at "
                        f"[{start}, {end}); expected start {cursor}"
                    )
                path = self._resolve_shard(str(entry.get("path", "")), kind=kind)
                if not path.is_file() or path.stat().st_size != int(
                    entry.get("bytes", -1)
                ):
                    raise RuntimeError(f"Missing or truncated resume shard: {path}")
                if _sha256(path) != entry.get("sha256"):
                    raise RuntimeError(f"Resume shard hash mismatch: {path}")
                cursor = end
            if cursor != int(state.get("next_position", -1)):
                raise RuntimeError(
                    f"Diagnostic resume {kind} cursor differs from committed shards"
                )
        if self.causal_next and self.passive_next != self.passive_total:
            raise RuntimeError("Causal progress exists before passive completion")
        if self.causal_next and not self.progress["causal"].get("selection_hash"):
            raise RuntimeError("Causal progress exists without a selection identity")

    @property
    def passive_next(self) -> int:
        return int(self.progress["passive"]["next_position"])

    @property
    def causal_next(self) -> int:
        return int(self.progress["causal"]["next_position"])

    @property
    def has_committed_work(self) -> bool:
        return bool(self.passive_next or self.causal_next)

    def _write_shard(
        self,
        *,
        kind: str,
        start: int,
        end: int,
        payload: Mapping[str, Any],
        store: "ArrayStore",
    ) -> None:
        state = self.progress[kind]
        if int(state["next_position"]) != int(start):
            raise RuntimeError(
                f"Cannot commit {kind} [{start}, {end}); next position is "
                f"{state['next_position']}"
            )
        total = self.passive_total if kind == "passive" else self.causal_total
        if end <= start or end > total:
            raise RuntimeError(f"Invalid {kind} progress interval [{start}, {end})")
        if kind == "passive" and len(payload.get("queries", [])) != end - start:
            raise RuntimeError("Passive resume shard query count mismatch")
        if kind == "causal" and len(payload.get("selection", [])) != end - start:
            raise RuntimeError("Causal resume shard selection count mismatch")

        # Flushing precedes the commit record.  A kill before progress advances
        # leaves this unit uncommitted, so the next run overwrites the same span.
        store.flush()
        array_spans = store.span_manifest(kind, start, end)
        relative = Path(".resume") / kind / f"{start:06d}_{end:06d}.json"
        path = self.root / relative
        shard = {
            "schema_name": RESUME_SCHEMA_NAME,
            "schema_version": RESUME_SCHEMA_VERSION,
            "expected_ready_identity_sha256": self.identity_digest,
            "kind": kind,
            "start": int(start),
            "end": int(end),
            "selection_hash": (
                self.progress["causal"].get("selection_hash")
                if kind == "causal"
                else None
            ),
            "array_spans": array_spans,
            "payload": payload,
        }
        _atomic_write_json(path, shard)
        entry = {
            "path": str(relative),
            "start": int(start),
            "end": int(end),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        updated = json.loads(json.dumps(self.progress))
        updated[kind]["shards"].append(entry)
        updated[kind]["next_position"] = int(end)
        updated["updated_at"] = _utc_now()
        _atomic_write_json(self.progress_path, updated)
        self.progress = updated

    def commit_passive(
        self,
        start: int,
        end: int,
        payload: Mapping[str, Any],
        store: "ArrayStore",
    ) -> None:
        self._write_shard(
            kind="passive", start=start, end=end, payload=payload, store=store
        )

    def bind_selection(self, selection_hash: str) -> None:
        if self.passive_next != self.passive_total:
            raise RuntimeError("Cannot bind causal selection before passive completion")
        stored = self.progress["causal"].get("selection_hash")
        if stored is not None and stored != str(selection_hash):
            raise RuntimeError(
                "Refusing causal resume because the deterministic selection changed"
            )
        if stored is None:
            updated = json.loads(json.dumps(self.progress))
            updated["causal"]["selection_hash"] = str(selection_hash)
            updated["updated_at"] = _utc_now()
            _atomic_write_json(self.progress_path, updated)
            self.progress = updated

    def commit_causal(
        self,
        start: int,
        end: int,
        payload: Mapping[str, Any],
        store: "ArrayStore",
    ) -> None:
        if self.progress["causal"].get("selection_hash") is None:
            raise RuntimeError("Causal selection identity was not bound")
        self._write_shard(
            kind="causal", start=start, end=end, payload=payload, store=store
        )

    def _read_committed_shard(
        self, kind: str, entry: Mapping[str, Any]
    ) -> dict[str, Any]:
        path = self._resolve_shard(str(entry["path"]), kind=kind)
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("bytes", -1))
            or _sha256(path) != entry.get("sha256")
        ):
            raise RuntimeError(f"Committed resume shard hash mismatch: {path}")
        shard = self._read_json_mapping(path, label=f"diagnostic {kind} shard")
        if (
            shard.get("schema_name") != RESUME_SCHEMA_NAME
            or int(shard.get("schema_version", -1)) != RESUME_SCHEMA_VERSION
            or shard.get("expected_ready_identity_sha256") != self.identity_digest
            or shard.get("kind") != kind
            or int(shard.get("start", -1)) != int(entry["start"])
            or int(shard.get("end", -1)) != int(entry["end"])
            or (
                kind == "causal"
                and shard.get("selection_hash")
                != self.progress["causal"].get("selection_hash")
            )
            or not isinstance(shard.get("array_spans"), Mapping)
            or not isinstance(shard.get("payload"), Mapping)
        ):
            raise RuntimeError(f"Resume shard content mismatch: {path}")
        return shard

    def _load_shard_payloads(self, kind: str) -> list[dict[str, Any]]:
        return [
            dict(self._read_committed_shard(kind, entry)["payload"])
            for entry in self.progress[kind]["shards"]
        ]

    def validate_array_checkpoints(self, store: "ArrayStore") -> None:
        """Reject any committed NPY span that differs from its atomic shard."""

        for kind in ("passive", "causal"):
            for entry in self.progress[kind]["shards"]:
                path = self._resolve_shard(str(entry["path"]), kind=kind)
                shard = self._read_committed_shard(kind, entry)
                expected = shard.get("array_spans")
                if not isinstance(expected, Mapping):
                    raise RuntimeError(f"Resume shard lacks array span hashes: {path}")
                actual = store.span_manifest(
                    kind, int(entry["start"]), int(entry["end"])
                )
                if actual != expected:
                    raise RuntimeError(
                        f"Committed diagnostic array span hash mismatch: {path}"
                    )

    def load_passive(self) -> dict[str, Any]:
        combined = {
            "queries": [],
            "slots": [],
            "text_attention": [],
            "part_attention": [],
            "candidates": [],
            "checks": {
                "text_attention_replay_max_abs": 0.0,
                "instrumented_prediction_max_abs": 0.0,
                "all_null_vs_memory_off_max_abs": 0.0,
                "duration_memory_on_off_max_abs": 0.0,
            },
        }
        for payload in self._load_shard_payloads("passive"):
            for key in (
                "queries",
                "slots",
                "text_attention",
                "part_attention",
                "candidates",
            ):
                rows = payload.get(key)
                if not isinstance(rows, list):
                    raise RuntimeError(f"Passive shard lacks list field {key}")
                combined[key].extend(rows)
            checks = payload.get("checks")
            if not isinstance(checks, Mapping):
                raise RuntimeError("Passive shard lacks integrity checks")
            for key in combined["checks"]:
                combined["checks"][key] = max(
                    float(combined["checks"][key]), float(checks.get(key, 0.0))
                )
        if len(combined["queries"]) != self.passive_next:
            raise RuntimeError("Committed passive query rows do not match progress")
        return combined

    def load_causal(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        selection: list[dict[str, Any]] = []
        for payload in self._load_shard_payloads("causal"):
            current_rows = payload.get("rows")
            current_selection = payload.get("selection")
            if not isinstance(current_rows, list) or not isinstance(
                current_selection, list
            ):
                raise RuntimeError("Causal shard lacks row/selection lists")
            rows.extend(current_rows)
            selection.extend(current_selection)
        if len(selection) != self.causal_next:
            raise RuntimeError("Committed causal selections do not match progress")
        return rows, selection

    def mark_finalizing(self) -> None:
        if self.passive_next != self.passive_total or self.causal_next != self.causal_total:
            raise RuntimeError("Cannot finalize an incomplete diagnostic")
        updated = json.loads(json.dumps(self.progress))
        updated["status"] = "finalizing"
        updated["updated_at"] = _utc_now()
        _atomic_write_json(self.progress_path, updated)
        self.progress = updated


def _write_artifact_manifest(root: Path) -> str:
    """Hash every non-circular READY artifact and return the manifest hash."""

    targets = tuple(
        relative for relative in READY_REQUIRED_FILES if relative != "provenance.json"
    )
    output_root = root.resolve()
    artifacts: dict[str, Any] = {}
    for relative in targets:
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(output_root) or not artifact.is_file():
            raise RuntimeError(f"Cannot manifest missing READY artifact: {relative}")
        artifacts[relative] = {
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }
    manifest = {
        "schema": "signtrajfield_temporal_slot_artifacts",
        "version": 1,
        "artifacts": artifacts,
    }
    path = root / ARTIFACT_MANIFEST_FILE
    _atomic_write_json(path, manifest)
    return _sha256(path)


def _validate_artifact_manifest(root: Path, provenance: Mapping[str, Any]) -> bool:
    path = root / ARTIFACT_MANIFEST_FILE
    if not path.is_file() or provenance.get("artifact_manifest_sha256") != _sha256(path):
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        targets = {
            relative
            for relative in READY_REQUIRED_FILES
            if relative != "provenance.json"
        }
        if (
            manifest.get("schema") != "signtrajfield_temporal_slot_artifacts"
            or int(manifest.get("version", -1)) != 1
            or not isinstance(manifest.get("artifacts"), Mapping)
            or set(manifest["artifacts"]) != targets
        ):
            return False
        output_root = root.resolve()
        for relative, metadata in manifest["artifacts"].items():
            if not isinstance(metadata, Mapping):
                return False
            relative_path = Path(relative)
            if relative_path.is_absolute():
                return False
            artifact = (root / relative_path).resolve()
            if not artifact.is_relative_to(output_root) or not artifact.is_file():
                return False
            if artifact.stat().st_size != int(metadata.get("bytes", -1)):
                return False
            if _sha256(artifact) != metadata.get("sha256"):
                return False
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _ready_identity(
    out_dir: Path, expected: Mapping[str, Any]
) -> dict[str, Any] | None:
    if expected.get("source_sha256") != _diagnostic_source_hashes():
        return None
    if expected.get("git_head") != _git_head():
        return None
    try:
        profile = DIAGNOSTIC_PROFILES[str(expected.get("profile"))]
        config_path = Path(str(expected.get("config_path"))).resolve(strict=True)
        checkpoint_path = Path(str(expected.get("checkpoint_path"))).resolve(
            strict=True
        )
    except (KeyError, OSError):
        return None
    if config_path != profile.config.resolve() or checkpoint_path != profile.checkpoint.resolve():
        return None
    if _sha256(config_path) != expected.get("config_sha256"):
        return None
    if _sha256(checkpoint_path) != expected.get("checkpoint_sha256"):
        return None
    current_cfg = load_config(config_path)
    _validate_config(current_cfg, profile)
    current_cfg.setdefault("data", {})["random_crop"] = False
    current_cfg.setdefault("text", {})["device"] = "cpu"
    current_cfg["device"] = "cuda"
    if _digest_json(current_cfg) != expected.get("resolved_config_sha256"):
        return None
    if _sha256(EXPECTED_VALIDATION_MANIFEST) != (
        expected.get("validation_manifest") or {}
    ).get("sha256"):
        return None
    ready = out_dir / "READY"
    identity_path = out_dir / "provenance.json"
    if not ready.is_file() or not identity_path.is_file():
        return None
    try:
        marker = json.loads(ready.read_text(encoding="utf-8"))
        value = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        marker.get("schema_name") != SCHEMA_NAME
        or int(marker.get("schema_version", -1)) != SCHEMA_VERSION
        or marker.get("provenance_sha256") != _sha256(identity_path)
        or marker.get("profile") != expected.get("profile")
        or marker.get("checkpoint_sha256") != expected.get("checkpoint_sha256")
        or int(marker.get("checkpoint_epoch", -1))
        != int(expected.get("checkpoint_epoch", -2))
    ):
        return None
    if any(not (out_dir / relative).is_file() for relative in READY_REQUIRED_FILES):
        return None
    if not _validate_artifact_manifest(out_dir, value):
        return None
    if value.get("schema_name") != SCHEMA_NAME or int(
        value.get("schema_version", -1)
    ) != SCHEMA_VERSION:
        return None
    actual_bank = (value.get("bank_identity") or {}).get("bank_id")
    comparisons = {
        "profile": value.get("profile"),
        "stage": value.get("stage"),
        "split": value.get("split"),
        "checkpoint_path": value.get("checkpoint"),
        "checkpoint_sha256": value.get("checkpoint_sha256"),
        "checkpoint_epoch": value.get("checkpoint_epoch"),
        "config_path": value.get("config"),
        "config_sha256": value.get("config_sha256"),
        "resolved_config_sha256": value.get("resolved_config_sha256"),
        "bank_id": actual_bank,
        "settings": value.get("settings"),
        "query_counts": value.get("query_counts"),
        "source_sha256": value.get("source_sha256"),
        "git_head": value.get("git_head"),
        "neighbor_table_sha256": value.get("neighbor_table_sha256"),
        "validation_manifest": value.get("validation_manifest"),
        "locked_run_evidence": value.get("locked_run_evidence"),
    }
    if comparisons != dict(expected):
        return None
    manifest_path = out_dir / "arrays/manifest.json"
    if value.get("array_manifest_sha256") != _sha256(manifest_path):
        return None
    try:
        arrays = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_hashes = bool(expected.get("settings", {}).get("verify_hashes"))
        output_root = out_dir.resolve()
        for metadata in arrays.values():
            relative = Path(str(metadata["path"]))
            if relative.is_absolute():
                return None
            artifact = (out_dir / relative).resolve()
            if not artifact.is_relative_to(output_root) or not artifact.is_file():
                return None
            if artifact.stat().st_size != int(metadata["bytes"]):
                return None
            if verify_hashes:
                expected_hash = metadata.get("sha256")
                if not expected_hash or _sha256(artifact) != expected_hash:
                    return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return value


@contextmanager
def _atomic_output(
    out_dir: Path,
    *,
    resume: bool,
    expected_identity: Mapping[str, Any],
    passive_rows: int,
    causal_rows: int,
):
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = out_dir.with_name(f".{out_dir.name}.partial")
    lock_path = out_dir.with_name(f".{out_dir.name}.resume.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another diagnostic process holds the resume lock: {lock_path}"
            ) from error

        if out_dir.exists():
            identity = _ready_identity(out_dir, expected_identity)
            if resume and identity is not None:
                yield None
                return
            if identity is not None:
                raise FileExistsError(
                    f"Completed output already exists: {out_dir}; pass --resume to reuse it"
                )
            raise FileExistsError(
                f"Incomplete output exists: {out_dir}; inspect or move it before retrying"
            )

        # A process can be killed after READY is written but before the atomic
        # directory promotion.  Reuse that fully validated tree without
        # repeating inference.
        if partial.exists() and (partial / "READY").is_file():
            identity = _ready_identity(partial, expected_identity)
            if not resume or identity is None:
                raise RuntimeError(
                    f"Unpromoted partial READY output failed reuse policy: {partial}"
                )
            os.replace(partial, out_dir)
            yield None
            return

        workspace = _ResumeWorkspace(
            partial,
            expected_identity=expected_identity,
            passive_total=int(passive_rows),
            causal_total=int(causal_rows),
            resume=resume,
        )
        try:
            yield workspace
            workspace.mark_finalizing()
            if (partial / "READY").exists():
                raise RuntimeError("READY must be created only by the atomic finalizer")
            provenance_path = partial / "provenance.json"
            if not provenance_path.is_file():
                raise RuntimeError("Cannot finalize diagnostic without provenance.json")
            ready_path = partial / "READY"
            _atomic_write_json(
                ready_path,
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "profile": expected_identity.get("profile"),
                    "checkpoint_sha256": expected_identity.get(
                        "checkpoint_sha256"
                    ),
                    "checkpoint_epoch": expected_identity.get(
                        "checkpoint_epoch"
                    ),
                    "provenance_sha256": _sha256(provenance_path),
                },
            )
            if _ready_identity(partial, expected_identity) is None:
                ready_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Produced artifacts failed final provenance/integrity validation"
                )
            try:
                os.replace(partial, out_dir)
            except BaseException:
                # A retained partial tree is incomplete if promotion fails.
                # Never leave a misleading READY marker in that tree.
                ready_path.unlink(missing_ok=True)
                raise
        except BaseException:
            if partial.exists():
                (partial / "READY").unlink(missing_ok=True)
            print(
                f"Incomplete resumable diagnostic retained at {partial}",
                file=sys.stderr,
            )
            raise


def _tokenize_for_audit(text_encoder, texts: Sequence[str], token_mask: torch.Tensor):
    encoded = text_encoder.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(text_encoder.max_length),
        return_tensors="pt",
    )
    ids = encoded["input_ids"].long()
    mask = encoded["attention_mask"].bool()
    expected = token_mask.detach().cpu().bool()
    if mask.shape != expected.shape or not torch.equal(mask, expected):
        raise RuntimeError("Retokenized attention mask does not match encoded token mask")
    eos_id = getattr(text_encoder.tokenizer, "eos_token_id", None)
    eos_mask = torch.zeros_like(mask) if eos_id is None else (ids == int(eos_id)) & mask
    return ids, mask, eos_mask


def _stack_sentence_layers(snapshot):
    if not snapshot.sentence_layers:
        raise RuntimeError("Expected sentence-memory attention captures")
    candidate = torch.stack(
        [layer.part_candidate_mass for layer in snapshot.sentence_layers], dim=1
    )
    null = torch.stack(
        [layer.part_null_mass for layer in snapshot.sentence_layers], dim=1
    )
    token = torch.stack([layer.token_mass for layer in snapshot.sentence_layers], dim=1)
    return candidate, null, token


def _replace_memory(memory: SentenceMemoryBatch, **updates: Any) -> SentenceMemoryBatch:
    return dataclasses.replace(memory, **updates)


def _motion_only_shuffle(
    memory: SentenceMemoryBatch,
    query_ids: Sequence[str],
    *,
    epoch: int,
    seed: int,
) -> tuple[SentenceMemoryBatch, torch.Tensor]:
    shuffled, permutations, _informative = (
        motion_only_shuffle_sentence_memory_batch(
            memory,
            query_ids=query_ids,
            epoch=epoch,
            seed=seed,
        )
    )
    return shuffled, permutations


def _all_null(memory: SentenceMemoryBatch) -> SentenceMemoryBatch:
    return _replace_memory(
        memory,
        token_mask=torch.zeros_like(memory.token_mask),
        candidate_mask=torch.zeros_like(memory.candidate_mask),
        available=torch.ones_like(memory.available),
        provenance={**memory.provenance, "mode": "all_null"},
    )


def _query_ids(batch: Mapping[str, Any]) -> list[str]:
    return sentence_memory_query_ids(batch)


def _first_unique_novel_indices(dataset, provider, count: int) -> list[int]:
    indices = []
    seen: set[str] = set()
    for index, row in enumerate(dataset.base.items):
        text = str(row.get("text", ""))
        normalized = normalize_sentence_text(text)
        if not normalized or normalized in seen or provider.is_seen_text(text):
            continue
        seen.add(normalized)
        indices.append(index)
        if len(indices) == int(count):
            break
    if len(indices) != int(count):
        raise RuntimeError(
            f"Could not find {count} unique novel validation rows for smoke"
        )
    return indices


def _run_model(
    model,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    query_tau: torch.Tensor,
    memory: SentenceMemoryBatch | None,
):
    kwargs = {} if memory is None else sentence_memory_forward_kwargs(memory)
    return model(
        text_tokens=text_tokens,
        text_mask=text_mask,
        query_times=query_tau,
        query_mask=None,
        time_domain="normalized",
        word_prior_available=torch.zeros(
            text_tokens.shape[0], dtype=torch.bool, device=text_tokens.device
        ),
        **kwargs,
    )


def _mean_tensor(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def _average_ranks_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        # Mirror temporal_slot_diagnostics._average_ranks exactly: sorted,
        # adjacent near-tie groups using atol=1e-8 and rtol=1e-6.  Adjacent
        # chaining is intentional and keeps the permutation-null statistic on
        # precisely the same estimand as the primary text Spearman metric.
        while stop < len(values):
            left = float(values[order[stop - 1]])
            right = float(values[order[stop]])
            tolerance = 1e-8 + 1e-6 * max(abs(left), abs(right))
            if abs(right - left) > tolerance:
                break
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman_arrays(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _average_ranks_array(first)
    second_rank = _average_ranks_array(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = float(
        np.sqrt(np.square(first_rank).sum() * np.square(second_rank).sum())
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_rank, second_rank) / denominator)


def _scalar_slot_summary(metrics: Mapping[str, torch.Tensor], row: int) -> dict[str, float]:
    keys = (
        "off_diagonal_mean",
        "off_diagonal_median",
        "off_diagonal_p90",
        "off_diagonal_p95",
        "adjacent_cosine_mean",
        "far_cosine_mean",
        "first_last_cosine",
        "across_slot_variance",
        "temporal_variance_ratio",
        "temporal_total_variation",
        "centered_effective_rank",
    )
    output = {key: float(metrics[key][row].float().item()) for key in keys}
    cosine = metrics["cosine_matrix"][row]
    diagonal = torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    output["off_diagonal_fraction_above_0p95"] = float(
        (cosine[~diagonal] > 0.95).float().mean().item()
    )
    return output


class ArrayStore:
    """Fixed-shape, memory-mappable raw diagnostics."""

    PASSIVE_ARRAYS = (
        "query_dataset_index",
        "token_ids",
        "text_mask",
        "planner_slots",
        "fused_slots",
        "sentence_delta",
        "text_attention",
        "text_expected_token_position_rank",
        "candidate_mass",
        "token_mass",
        "part_null_mass",
        "sentence_gates",
        "analytic_prior",
        "candidate_latent_ntrmse",
        "candidate_ids",
        "candidate_scores",
        "duration_log_gap",
        "candidate_mask",
        "motion_source_permutation",
    )
    CAUSAL_ARRAYS = (
        "causal_fk_response",
        "causal_branch_response",
        "memory_on_off_fk_difference",
        "all_null_off_fk_parity",
        "causal_dataset_index",
        "rademacher_directions",
    )

    def __init__(
        self,
        root: Path,
        *,
        rows: int,
        slots: int,
        hidden: int,
        text_layers: int,
        text_heads: int,
        max_text_tokens: int,
        sentence_layers: int,
        candidates: int,
        max_motion_tokens: int,
        causal_rows: int,
        directions: int,
        query_points: int,
        reopen: bool = False,
    ):
        self.root = root / "arrays"
        self.reopen = bool(reopen)
        if self.reopen:
            if not self.root.is_dir():
                raise RuntimeError(f"Cannot reopen missing array store: {self.root}")
        else:
            self.root.mkdir()
        self.specs: dict[str, tuple[tuple[int, ...], str]] = {}
        self.arrays: dict[str, np.memmap] = {}

        self._create("query_dataset_index", (rows,), "int64", -1)
        self._create("token_ids", (rows, max_text_tokens), "int32", -1)
        self._create("text_mask", (rows, max_text_tokens), "uint8", 0)
        self._create("planner_slots", (rows, slots, hidden), "float16", 0)
        self._create(
            "fused_slots",
            (rows, len(MEMORY_CONDITIONS), slots, hidden),
            "float16",
            0,
        )
        self._create(
            "sentence_delta",
            (rows, len(MEMORY_CONDITIONS), slots, hidden),
            "float16",
            0,
        )
        self._create(
            "text_attention",
            (rows, text_layers, text_heads, slots, max_text_tokens),
            # Keep full replay precision: rank-based permutation statistics are
            # sensitive to artificial FP16 ties.
            "float32",
            0,
        )
        self._create(
            "text_expected_token_position_rank",
            (rows, text_layers, text_heads, slots),
            "float32",
            0,
        )
        attention_prefix = (
            rows,
            len(MEMORY_CONDITIONS),
            sentence_layers,
            slots,
            len(PART_NAMES),
            candidates,
        )
        self._create("candidate_mass", attention_prefix, "float16", 0)
        self._create(
            "token_mass", attention_prefix + (max_motion_tokens,), "float16", 0
        )
        self._create(
            "part_null_mass", attention_prefix[:-1], "float16", 1
        )
        self._create(
            "sentence_gates",
            (rows, len(MEMORY_CONDITIONS), slots, len(PART_NAMES)),
            "float16",
            0,
        )
        self._create("analytic_prior", (rows, candidates), "float32", 0)
        self._create("candidate_latent_ntrmse", (rows, candidates), "float32", 0)
        self._create("candidate_ids", (rows, 2, candidates), "int32", -1)
        self._create("candidate_scores", (rows, 2, candidates), "float32", 0)
        self._create("duration_log_gap", (rows, 2, candidates), "float32", 0)
        self._create("candidate_mask", (rows, 2, candidates), "uint8", 0)
        self._create("motion_source_permutation", (rows, candidates), "int16", -1)
        self._create(
            "causal_fk_response",
            (
                causal_rows,
                len(INTERVENTION_STAGES),
                2,
                directions,
                slots,
                len(PART_NAMES),
                query_points,
            ),
            "float16",
            0,
        )
        self._create(
            "causal_branch_response",
            (
                causal_rows,
                len(INTERVENTION_STAGES),
                2,
                directions,
                len(BRANCH_NAMES),
                slots,
                len(PART_NAMES),
                query_points,
            ),
            "float16",
            0,
        )
        self._create(
            "memory_on_off_fk_difference",
            (causal_rows, len(PART_NAMES), query_points),
            "float32",
            0,
        )
        self._create(
            "all_null_off_fk_parity",
            (causal_rows, len(PART_NAMES), query_points),
            "float32",
            0,
        )
        self._create("causal_dataset_index", (causal_rows,), "int64", -1)
        self._create(
            "rademacher_directions",
            (causal_rows, len(INTERVENTION_STAGES), slots, directions, hidden),
            "float16",
            0,
        )

        specification = {
            name: {"shape": list(shape), "dtype": dtype}
            for name, (shape, dtype) in self.specs.items()
        }
        marker_path = self.root / ".initialized.json"
        if self.reopen:
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Cannot validate reopenable array store {marker_path}: {error}"
                ) from error
            if marker != {
                "schema": "temporal_slot_diagnostic_arrays",
                "version": 1,
                "specification": specification,
                "specification_sha256": _digest_json(specification),
            }:
                raise RuntimeError("Reopenable diagnostic array specification changed")
        else:
            self.flush()
            _atomic_write_json(
                marker_path,
                {
                    "schema": "temporal_slot_diagnostic_arrays",
                    "version": 1,
                    "specification": specification,
                    "specification_sha256": _digest_json(specification),
                },
            )

    def _create(self, name: str, shape: tuple[int, ...], dtype: str, fill: Any):
        path = self.root / f"{name}.npy"
        expected_dtype = np.dtype(dtype)
        if self.reopen:
            try:
                array = np.load(path, mmap_mode="r+", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Cannot reopen diagnostic array {path}: {error}") from error
            if tuple(array.shape) != tuple(shape) or array.dtype != expected_dtype:
                raise RuntimeError(
                    f"Diagnostic array specification mismatch for {path}: "
                    f"found shape={array.shape}, dtype={array.dtype}; "
                    f"expected shape={shape}, dtype={expected_dtype}"
                )
        else:
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=expected_dtype, shape=shape
            )
            array[...] = fill
        self.arrays[name] = array
        self.specs[name] = (shape, str(expected_dtype))

    def __getitem__(self, name: str) -> np.memmap:
        return self.arrays[name]

    def flush(self):
        for value in self.arrays.values():
            value.flush()

    def span_manifest(self, kind: str, start: int, end: int) -> dict[str, Any]:
        names = self.PASSIVE_ARRAYS if kind == "passive" else self.CAUSAL_ARRAYS
        if kind not in {"passive", "causal"}:
            raise ValueError(f"Unknown diagnostic array span kind: {kind}")
        output: dict[str, Any] = {}
        for name in names:
            value = np.asarray(self.arrays[name][int(start) : int(end)])
            if not value.flags.c_contiguous:
                raise RuntimeError(f"Diagnostic span is not contiguous: {name}")
            digest = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
            output[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(value.nbytes),
                "sha256": digest,
            }
        return output

    def manifest(self, *, verify_hashes: bool) -> dict[str, Any]:
        self.flush()
        output = {}
        for name, (shape, dtype) in self.specs.items():
            path = self.root / f"{name}.npy"
            output[name] = {
                "path": str(path.relative_to(self.root.parent)),
                "shape": list(shape),
                "dtype": dtype,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path) if verify_hashes else None,
            }
        return output


def _copy_padded(
    destination: np.ndarray,
    source: torch.Tensor | np.ndarray,
    *,
    leading_slice: slice,
) -> None:
    value = (
        source.detach().cpu().numpy() if torch.is_tensor(source) else np.asarray(source)
    )
    target = destination[leading_slice]
    if value.ndim != target.ndim:
        raise RuntimeError(
            f"Cannot pad diagnostic tensor {value.shape} into {target.shape}"
        )
    slices = tuple(slice(0, size) for size in value.shape)
    target[slices] = value.astype(target.dtype, copy=False)


def _candidate_margin(memory: SentenceMemoryBatch) -> torch.Tensor:
    adjusted = memory.scores - EXPECTED_DURATION_WEIGHT * memory.duration_log_gap.abs()
    adjusted = adjusted.masked_fill(~memory.candidate_mask, -torch.inf)
    top = torch.topk(adjusted, k=min(2, adjusted.shape[-1]), dim=-1).values
    if top.shape[-1] == 1:
        return torch.zeros_like(top[:, 0])
    result = top[:, 0] - top[:, 1]
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def _condition_capture(
    model,
    text_tokens,
    text_mask,
    query_tau,
    memory,
    *,
    replay_text_attention: bool,
):
    with DiagnosticCapture(
        model,
        replay_text_attention=replay_text_attention,
        replay_tolerance=1e-6,
    ) as capture:
        outputs = _run_model(model, text_tokens, text_mask, query_tau, memory)
        snapshot = capture.snapshot(clear=True, cpu=True)
    return outputs, snapshot


def _metric_row_mean(value: torch.Tensor, row: int) -> float:
    return float(value[row].detach().float().mean().cpu().item())


def _candidate_condition_summary(
    metrics: Mapping[str, torch.Tensor], row: int
) -> dict[str, float]:
    """Summarize conditional selectivity without counting null pairs as ties."""

    argmax = metrics["argmax_candidate"][row, -1]  # [S,P], -1 is null-only
    valid = argmax >= 0

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        selected = values.float()[mask]
        return float(selected.mean().item()) if selected.numel() else 0.0

    effective = metrics["effective_candidate_count"][row, -1]
    maximum = metrics["maximum_candidate_share"][row, -1]
    adjacent_valid = valid[1:] & valid[:-1]
    adjacent_switch = metrics["adjacent_candidate_switch"][row, -1]
    adjacent_jsd = metrics["adjacent_slot_jsd"][row, -1].permute(1, 0)

    part_upper = torch.triu_indices(valid.shape[-1], valid.shape[-1], 1)
    part_valid = valid[:, part_upper[0]] & valid[:, part_upper[1]]
    part_switch = metrics["part_candidate_switch"][
        row, -1, :, part_upper[0], part_upper[1]
    ]
    part_jsd = metrics["part_jsd_matrix"][
        row, -1, :, part_upper[0], part_upper[1]
    ]
    return {
        "effective_k": masked_mean(effective, valid),
        "max_share": masked_mean(maximum, valid),
        "null_mass": float(metrics["null_mass"][row, -1].float().mean().item()),
        "slot_jsd": masked_mean(adjacent_jsd, adjacent_valid),
        "part_jsd": masked_mean(part_jsd, part_valid),
        "adjacent_switch": masked_mean(adjacent_switch, adjacent_valid),
        "part_switch": masked_mean(part_switch, part_valid),
        "adjacent_switch_coverage": float(adjacent_valid.float().mean().item()),
        "part_switch_coverage": float(part_valid.float().mean().item()),
    }


def _append_attention_rows(
    rows: list[dict[str, Any]],
    query_ids: Sequence[str],
    metrics: Mapping[str, torch.Tensor],
):
    correlation = metrics["slot_token_position_spearman"]
    for sample, query_id in enumerate(query_ids):
        for layer in range(correlation.shape[1]):
            for head in range(correlation.shape[2]):
                rows.append(
                    {
                        "query_id": query_id,
                        "layer": layer,
                        "head": head,
                        "normalized_entropy": float(
                            metrics["normalized_entropy"][sample, layer, head]
                            .float()
                            .mean()
                            .item()
                        ),
                        "effective_token_count": float(
                            metrics["effective_token_count"][sample, layer, head]
                            .float()
                            .mean()
                            .item()
                        ),
                        "maximum_token_mass": float(
                            metrics["maximum_token_mass"][sample, layer, head]
                            .float()
                            .mean()
                            .item()
                        ),
                        "eos_mass": float(
                            metrics["eos_mass"][sample, layer, head].float().mean().item()
                        ),
                        "slot_token_position_spearman": float(
                            correlation[sample, layer, head].float().item()
                        ),
                        "inversion_rate": float(
                            metrics["inversion_rate"][sample, layer, head]
                            .float()
                            .item()
                        ),
                        "attention_span": float(
                            metrics["attention_span"][sample, layer, head]
                            .float()
                            .item()
                        ),
                        "adjacent_slot_jsd": float(
                            metrics["adjacent_slot_jsd"][sample, layer, head]
                            .float()
                            .mean()
                            .item()
                        ),
                    }
                )


def _append_part_attention_rows(
    rows: list[dict[str, Any]],
    query_ids: Sequence[str],
    condition: str,
    metrics: Mapping[str, torch.Tensor],
):
    effective = metrics["effective_candidate_count"]
    part_count = effective.shape[-1]
    for sample, query_id in enumerate(query_ids):
        for layer in range(effective.shape[1]):
            for slot in range(effective.shape[2]):
                for part, part_name in enumerate(PART_NAMES):
                    part_indices = torch.arange(part_count)
                    other = part_indices != part
                    argmax = metrics["argmax_candidate"][sample, layer, slot]
                    valid_part_pairs = other & (argmax[part] >= 0) & (argmax >= 0)
                    part_switch_values = metrics["part_candidate_switch"][
                        sample, layer, slot, part
                    ][valid_part_pairs]
                    rows.append(
                        {
                            "query_id": query_id,
                            "condition": condition,
                            "layer": layer,
                            "slot": slot,
                            "part": part_name,
                            "effective_candidate_count": float(
                                effective[sample, layer, slot, part].float().item()
                            ),
                            "maximum_candidate_share": float(
                                metrics["maximum_candidate_share"][
                                    sample, layer, slot, part
                                ]
                                .float()
                                .item()
                            ),
                            "real_candidate_mass": float(
                                metrics["real_candidate_mass"][sample, layer, slot, part]
                                .float()
                                .item()
                            ),
                            "null_mass": float(
                                metrics["null_mass"][sample, layer, slot, part]
                                .float()
                                .item()
                            ),
                            "argmax_candidate": int(
                                metrics["argmax_candidate"][sample, layer, slot, part]
                                .long()
                                .item()
                            ),
                            "adjacent_slot_jsd": (
                                float(
                                    metrics["adjacent_slot_jsd"][
                                        sample, layer, part, slot
                                    ]
                                    .float()
                                    .item()
                                )
                                if slot + 1 < effective.shape[2]
                                else None
                            ),
                            "adjacent_candidate_switch": (
                                bool(
                                    metrics["adjacent_candidate_switch"][
                                        sample, layer, slot, part
                                    ].item()
                                )
                                if slot + 1 < effective.shape[2]
                                and int(
                                    metrics["argmax_candidate"][
                                        sample, layer, slot, part
                                    ].item()
                                )
                                >= 0
                                and int(
                                    metrics["argmax_candidate"][
                                        sample, layer, slot + 1, part
                                    ].item()
                                )
                                >= 0
                                else None
                            ),
                            "part_candidate_switch_fraction": float(
                                part_switch_values.float().mean().item()
                            )
                            if part_switch_values.numel()
                            else None,
                            "part_candidate_switch_valid_comparisons": int(
                                part_switch_values.numel()
                            ),
                            "expected_memory_tau": float(
                                metrics["expected_memory_tau"][sample, layer, slot, part]
                                .float()
                                .item()
                            ),
                            "memory_time_absolute_error": float(
                                metrics["memory_time_absolute_error"][
                                    sample, layer, slot, part
                                ]
                                .float()
                                .item()
                            ),
                        }
                    )


def _append_candidate_rows(
    rows: list[dict[str, Any]],
    query_ids: Sequence[str],
    condition: str,
    memory: SentenceMemoryBatch,
    analytic_prior: torch.Tensor | None,
    latent_ntrmse: np.ndarray | None = None,
):
    provenance = memory.provenance
    names = provenance.get("candidate_names", [[] for _ in query_ids])
    groups = provenance.get("candidate_groups", [[] for _ in query_ids])
    exact_text = provenance.get("candidate_exact_text", [[] for _ in query_ids])
    for sample, query_id in enumerate(query_ids):
        for rank in range(memory.ids.shape[1]):
            rows.append(
                {
                    "query_id": query_id,
                    "condition": condition,
                    "rank": rank,
                    "item_id": int(memory.ids[sample, rank].detach().cpu().item()),
                    "candidate_name": names[sample][rank] if sample < len(names) else None,
                    "candidate_group": (
                        groups[sample][rank] if sample < len(groups) else None
                    ),
                    "candidate_exact_text": (
                        bool(exact_text[sample][rank])
                        if sample < len(exact_text)
                        and rank < len(exact_text[sample])
                        else False
                    ),
                    "valid": bool(
                        memory.candidate_mask[sample, rank].detach().cpu().item()
                    ),
                    "score": float(
                        memory.scores[sample, rank].detach().float().cpu().item()
                    ),
                    "duration_seconds": float(
                        memory.durations[sample, rank].detach().float().cpu().item()
                    ),
                    "duration_log_gap": float(
                        memory.duration_log_gap[sample, rank]
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    ),
                    "analytic_prior": (
                        float(analytic_prior[sample, rank].detach().float().cpu().item())
                        if analytic_prior is not None
                        else None
                    ),
                    "latent_ntrmse": (
                        float(latent_ntrmse[sample, rank])
                        if latent_ntrmse is not None
                        else None
                    ),
                }
            )


def _passive_sweep(
    *,
    model,
    text_encoder,
    provider,
    codec,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    dataset,
    indices: Sequence[int],
    cfg,
    device: torch.device,
    checkpoint_epoch: int,
    args,
    store: ArrayStore,
    workspace: _ResumeWorkspace,
):
    start_position = int(workspace.passive_next)
    loader = DataLoader(
        Subset(dataset, list(indices)[start_position:]),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_continuous_sign,
    )
    query_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    text_attention_rows: list[dict[str, Any]] = []
    part_attention_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    replay_max = 0.0
    instrumented_prediction_max_abs = 0.0
    all_null_off_max_abs = 0.0
    duration_parity_max_abs = 0.0
    written = start_position
    query_grid_1d = torch.linspace(
        -1.0, 1.0, int(args.query_points), device=device, dtype=torch.float32
    )

    with torch.inference_mode():
        for batch in tqdm(loader, desc="passive validation", unit="batch"):
            row_offsets = {
                "queries": len(query_rows),
                "slots": len(slot_rows),
                "text_attention": len(text_attention_rows),
                "part_attention": len(part_attention_rows),
                "candidates": len(candidate_rows),
            }
            batch_checks = {
                "text_attention_replay_max_abs": 0.0,
                "instrumented_prediction_max_abs": 0.0,
                "all_null_vs_memory_off_max_abs": 0.0,
                "duration_memory_on_off_max_abs": 0.0,
            }
            batch_size = len(batch["name"])
            destination = slice(written, written + batch_size)
            query_ids = _query_ids(batch)
            text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
            token_ids, audited_mask, eos_mask = _tokenize_for_audit(
                text_encoder, batch["text"], text_mask
            )
            _log_duration, predicted_duration = model.predict_duration(
                text_tokens, text_mask=text_mask
            )
            available = torch.ones(batch_size, dtype=torch.bool, device=device)
            correct = retrieve_sentence_memory(
                provider,
                dataset=dataset,
                batch=batch,
                text_tokens=text_tokens,
                text_mask=text_mask,
                predicted_duration=predicted_duration.detach(),
                available=available,
                training=False,
                device=device,
                mode="on",
            )
            shuffled = retrieve_sentence_memory(
                provider,
                dataset=dataset,
                batch=batch,
                text_tokens=text_tokens,
                text_mask=text_mask,
                predicted_duration=predicted_duration.detach(),
                available=available,
                training=False,
                device=device,
                mode="shuffled",
            )
            motion_only, motion_permutation = _motion_only_shuffle(
                correct,
                query_ids,
                epoch=checkpoint_epoch,
                seed=args.seed,
            )
            store["motion_source_permutation"][destination] = (
                motion_permutation.detach().cpu().numpy().astype(np.int16)
            )
            memories = {
                "correct": correct,
                "shuffled": shuffled,
                "motion_only_shuffle": motion_only,
            }
            query_tau = query_grid_1d[None].expand(batch_size, -1)
            query_mu, query_mu_mask = codec.encode(
                batch["motion"].to(device, non_blocking=True),
                mask=batch["mask"].to(device, non_blocking=True),
            )

            # The uninstrumented reference is checked once. Attention replay is
            # observational and must not change any prediction.
            uninstrumented = None
            if written == 0:
                uninstrumented = _run_model(
                    model, text_tokens, text_mask, query_tau, correct
                )

            condition_outputs = {}
            condition_candidate_metrics = {}
            condition_candidate_summary = {}
            condition_gate_summary = {}
            condition_delta_summary = {}
            for condition_index, condition in enumerate(MEMORY_CONDITIONS):
                memory = memories[condition]
                outputs, snapshot = _condition_capture(
                    model,
                    text_tokens,
                    text_mask,
                    query_tau,
                    memory,
                    replay_text_attention=condition == "correct",
                )
                condition_outputs[condition] = outputs
                if uninstrumented is not None and condition == "correct":
                    current_instrumented_max = float(
                        (outputs["prediction"] - uninstrumented["prediction"])
                        .abs()
                        .amax()
                        .item()
                    )
                    instrumented_prediction_max_abs = max(
                        instrumented_prediction_max_abs, current_instrumented_max
                    )
                    batch_checks["instrumented_prediction_max_abs"] = max(
                        batch_checks["instrumented_prediction_max_abs"],
                        current_instrumented_max,
                    )
                current_replay_max = (
                    float(snapshot.text_attention_replay_max_abs.max().item())
                    if snapshot.text_attention_replay_max_abs.numel()
                    else 0.0
                )
                replay_max = max(replay_max, current_replay_max)
                batch_checks["text_attention_replay_max_abs"] = max(
                    batch_checks["text_attention_replay_max_abs"],
                    current_replay_max,
                )
                if snapshot.sentence_delta is None:
                    raise RuntimeError("Memory-on capture is missing sentence_delta")
                candidate_mass, null_mass, token_mass = _stack_sentence_layers(snapshot)
                candidate_metrics = conditional_candidate_metrics(
                    candidate_mass,
                    null_mass,
                    candidate_mask=memory.candidate_mask.detach().cpu(),
                    token_mass=token_mass,
                    token_tau=memory.token_tau.detach().cpu(),
                    token_mask=memory.token_mask.detach().cpu(),
                    slot_tau=snapshot.slot_tau,
                )
                condition_candidate_metrics[condition] = candidate_metrics
                condition_candidate_summary[condition] = [
                    _candidate_condition_summary(candidate_metrics, row)
                    for row in range(batch_size)
                ]
                invariant_values = {
                    "candidate_null_conservation": candidate_metrics[
                        "mass_conservation_error"
                    ].max(),
                    "invalid_candidate_mass": candidate_metrics[
                        "invalid_candidate_mass_max"
                    ].max(),
                    "candidate_to_token_mass": candidate_metrics[
                        "token_candidate_mass_error"
                    ].max(),
                    "invalid_token_mass": candidate_metrics[
                        "invalid_token_mass_max"
                    ].max(),
                }
                limits = {
                    "candidate_null_conservation": 1e-5,
                    "invalid_candidate_mass": 1e-7,
                    "candidate_to_token_mass": 1e-6,
                    "invalid_token_mass": 1e-7,
                }
                for invariant, value in invariant_values.items():
                    if float(value.item()) > limits[invariant]:
                        raise RuntimeError(
                            f"{condition} {invariant} invariant failed: "
                            f"value={float(value.item()):.9g}, "
                            f"limit={limits[invariant]:.9g}"
                        )
                _append_part_attention_rows(
                    part_attention_rows, query_ids, condition, candidate_metrics
                )

                store["fused_slots"][destination, condition_index] = (
                    snapshot.fused_slots.numpy().astype(np.float16)
                )
                store["sentence_delta"][destination, condition_index] = (
                    snapshot.sentence_delta.numpy().astype(np.float16)
                )
                _copy_padded(
                    store["candidate_mass"][:, condition_index],
                    candidate_mass,
                    leading_slice=destination,
                )
                _copy_padded(
                    store["token_mass"][:, condition_index],
                    token_mass,
                    leading_slice=destination,
                )
                _copy_padded(
                    store["part_null_mass"][:, condition_index],
                    null_mass,
                    leading_slice=destination,
                )
                gates = outputs["trajectory"].sentence_memory_gates
                if gates is None:
                    raise RuntimeError("Memory-on trajectory is missing sentence gates")
                store["sentence_gates"][destination, condition_index] = (
                    gates.detach().cpu().numpy().astype(np.float16)
                )
                condition_gate_summary[condition] = [
                    float(gates[row].float().mean().item())
                    for row in range(batch_size)
                ]
                condition_delta_summary[condition] = [
                    float(snapshot.sentence_delta[row].float().norm(dim=-1).mean().item())
                    for row in range(batch_size)
                ]

                fused_metrics = compute_slot_metrics(snapshot.fused_slots)
                for row, query_id in enumerate(query_ids):
                    for slot in range(snapshot.fused_slots.shape[1]):
                        slot_rows.append(
                            {
                                "query_id": query_id,
                                "representation": "fused",
                                "condition": condition,
                                "slot": slot,
                                "tau": float(snapshot.slot_tau[slot].item()),
                                "norm": float(
                                    snapshot.fused_slots[row, slot].float().norm().item()
                                ),
                                "delta_norm": float(
                                    snapshot.sentence_delta[row, slot]
                                    .float()
                                    .norm()
                                    .item()
                                ),
                                "cosine_to_next": (
                                    float(
                                        fused_metrics["cosine_matrix"][
                                            row, slot, slot + 1
                                        ].item()
                                    )
                                    if slot + 1 < snapshot.fused_slots.shape[1]
                                    else None
                                ),
                            }
                        )

                if condition == "correct":
                    raw_metrics = compute_slot_metrics(snapshot.planner_slots)
                    text_metrics = compute_text_attention_metrics(
                        snapshot.text_attention,
                        audited_mask,
                        eos_mask=eos_mask,
                    )
                    if float(text_metrics["invalid_token_mass_max"].max().item()) > 1e-7:
                        raise RuntimeError("Text attention assigned mass to padding")
                    if float(text_metrics["attention_mass_error"].max().item()) > 1e-6:
                        raise RuntimeError("Text attention mass does not sum to one")
                    _append_attention_rows(
                        text_attention_rows, query_ids, text_metrics
                    )
                    store["planner_slots"][destination] = (
                        snapshot.planner_slots.numpy().astype(np.float16)
                    )
                    _copy_padded(
                        store["text_attention"],
                        text_metrics["normalized_attention"],
                        leading_slice=destination,
                    )
                    store["text_expected_token_position_rank"][destination] = (
                        text_metrics["expected_token_position_rank"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    for row, query_id in enumerate(query_ids):
                        for slot in range(snapshot.planner_slots.shape[1]):
                            slot_rows.append(
                                {
                                    "query_id": query_id,
                                    "representation": "planner",
                                    "condition": "text",
                                    "slot": slot,
                                    "tau": float(snapshot.slot_tau[slot].item()),
                                    "norm": float(
                                        snapshot.planner_slots[row, slot]
                                        .float()
                                        .norm()
                                        .item()
                                    ),
                                    "delta_norm": 0.0,
                                    "cosine_to_next": (
                                        float(
                                            raw_metrics["cosine_matrix"][
                                                row, slot, slot + 1
                                            ].item()
                                        )
                                        if slot + 1 < snapshot.planner_slots.shape[1]
                                        else None
                                    ),
                                }
                            )
                    raw_summary = [
                        _scalar_slot_summary(raw_metrics, row)
                        for row in range(batch_size)
                    ]
                    text_summary = [
                        {
                            "text_spearman": _metric_row_mean(
                                text_metrics["slot_token_position_spearman"], row
                            ),
                            "text_inversion_rate": _metric_row_mean(
                                text_metrics["inversion_rate"], row
                            ),
                            "text_normalized_entropy": _metric_row_mean(
                                text_metrics["normalized_entropy"], row
                            ),
                            "text_adjacent_jsd": _metric_row_mean(
                                text_metrics["adjacent_slot_jsd"], row
                            ),
                            "text_eos_mass": _metric_row_mean(
                                text_metrics["eos_mass"], row
                            ),
                        }
                        for row in range(batch_size)
                    ]
                fused_summary = [
                    _scalar_slot_summary(fused_metrics, row)
                    for row in range(batch_size)
                ]
                condition_outputs[condition + "_fused_summary"] = fused_summary

            prior = analytic_retrieval_prior(
                correct.scores,
                correct.duration_log_gap,
                correct.candidate_mask,
                duration_weight=EXPECTED_DURATION_WEIGHT,
                temperature=EXPECTED_SCORE_TEMPERATURE,
            )
            store["analytic_prior"][destination] = prior.detach().cpu().numpy()
            candidate_distance = np.zeros(
                tuple(correct.candidate_mask.shape), dtype=np.float32
            )
            correct_tokens = correct.tokens.detach().cpu().float().numpy()
            correct_token_mask = correct.token_mask.detach().cpu().bool().numpy()
            query_mu_cpu = query_mu.detach().cpu().float().numpy()
            query_mu_mask_cpu = query_mu_mask.detach().cpu().bool().numpy()
            for row in range(batch_size):
                query_latent = query_mu_cpu[row, query_mu_mask_cpu[row]]
                for rank in range(correct.candidate_mask.shape[1]):
                    if not bool(correct.candidate_mask[row, rank].item()):
                        continue
                    candidate_latent = correct_tokens[
                        row, rank, correct_token_mask[row, rank]
                    ]
                    candidate_distance[row, rank] = normalized_time_latent_rmse(
                        query_latent,
                        candidate_latent,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        points=64,
                    )
            store["candidate_latent_ntrmse"][destination] = candidate_distance
            _append_candidate_rows(
                candidate_rows,
                query_ids,
                "correct",
                correct,
                prior,
                candidate_distance,
            )
            _append_candidate_rows(candidate_rows, query_ids, "shuffled", shuffled, None)

            for memory_index, memory in enumerate((correct, shuffled)):
                store["candidate_ids"][destination, memory_index] = (
                    memory.ids.detach().cpu().numpy().astype(np.int32)
                )
                store["candidate_scores"][destination, memory_index] = (
                    memory.scores.detach().cpu().numpy()
                )
                store["duration_log_gap"][destination, memory_index] = (
                    memory.duration_log_gap.detach().cpu().numpy()
                )
                store["candidate_mask"][destination, memory_index] = (
                    memory.candidate_mask.detach().cpu().numpy().astype(np.uint8)
                )

            off_outputs = _run_model(model, text_tokens, text_mask, query_tau, None)
            null_outputs, null_snapshot = _condition_capture(
                model,
                text_tokens,
                text_mask,
                query_tau,
                _all_null(correct),
                replay_text_attention=False,
            )
            null_candidate, null_part, _null_token = _stack_sentence_layers(null_snapshot)
            current_null_off_max = float(
                (null_outputs["prediction"] - off_outputs["prediction"])
                .abs()
                .amax()
                .item()
            )
            all_null_off_max_abs = max(all_null_off_max_abs, current_null_off_max)
            batch_checks["all_null_vs_memory_off_max_abs"] = max(
                batch_checks["all_null_vs_memory_off_max_abs"], current_null_off_max
            )
            current_duration_max = float(
                (
                    condition_outputs["correct"]["trajectory"].duration_seconds
                    - off_outputs["trajectory"].duration_seconds
                )
                .abs()
                .amax()
                .item()
            )
            duration_parity_max_abs = max(duration_parity_max_abs, current_duration_max)
            batch_checks["duration_memory_on_off_max_abs"] = max(
                batch_checks["duration_memory_on_off_max_abs"], current_duration_max
            )
            if not bool((null_part == 1).all()) or not bool((null_candidate == 0).all()):
                raise RuntimeError("All-null control violated exact null-attention contract")
            null_gates = null_outputs["trajectory"].sentence_memory_gates
            if null_gates is None or not bool((null_gates == 0).all()):
                raise RuntimeError("All-null control produced nonzero sentence gates")

            margins = _candidate_margin(correct).detach().cpu()
            correct_candidate_metrics = condition_candidate_metrics["correct"]
            learned_distribution = correct_candidate_metrics[
                "conditional_candidate_mass"
            ][:, -1].mean(dim=(1, 2))
            learned_distribution = learned_distribution / learned_distribution.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            learned_prior_jsd = jensen_shannon_divergence(
                learned_distribution, prior.detach().cpu(), dim=-1
            )
            correct_conditional = correct_candidate_metrics[
                "conditional_candidate_mass"
            ][:, -1]
            correct_shuffled_attention_jsd = jensen_shannon_divergence(
                correct_conditional,
                condition_candidate_metrics["shuffled"][
                    "conditional_candidate_mass"
                ][:, -1],
                dim=-1,
            ).mean(dim=(1, 2))
            correct_motion_attention_jsd = jensen_shannon_divergence(
                correct_conditional,
                condition_candidate_metrics["motion_only_shuffle"][
                    "conditional_candidate_mass"
                ][:, -1],
                dim=-1,
            ).mean(dim=(1, 2))

            ids_array = batch["index"].detach().cpu().long().numpy()
            store["query_dataset_index"][destination] = ids_array
            store["token_ids"][destination, : token_ids.shape[1]] = (
                token_ids.numpy().astype(np.int32)
            )
            store["text_mask"][destination, : audited_mask.shape[1]] = (
                audited_mask.numpy().astype(np.uint8)
            )
            for row, query_id in enumerate(query_ids):
                exact_seen = provider.is_seen_text(batch["text"][row])
                correct_prediction = condition_outputs["correct"]["prediction"][row]
                shuffled_prediction = condition_outputs["shuffled"]["prediction"][row]
                motion_prediction = condition_outputs["motion_only_shuffle"][
                    "prediction"
                ][row]
                off_prediction = off_outputs["prediction"][row]
                query_rows.append(
                    {
                        "row_position": written + row,
                        "dataset_index": int(ids_array[row]),
                        "query_id": query_id,
                        "name": batch["name"][row],
                        "motion_path": batch["motion_path"][row],
                        "text": batch["text"][row],
                        "normalized_text": normalize_sentence_text(batch["text"][row]),
                        "text_hash": sentence_text_hash(batch["text"][row]),
                        "novel_text": not exact_seen,
                        "predicted_duration": float(predicted_duration[row].item()),
                        "retrieval_margin": float(margins[row].item()),
                        "candidate_effective_k": condition_candidate_summary[
                            "correct"
                        ][row]["effective_k"],
                        "candidate_max_share": condition_candidate_summary[
                            "correct"
                        ][row]["max_share"],
                        "candidate_null_mass": condition_candidate_summary[
                            "correct"
                        ][row]["null_mass"],
                        "candidate_adjacent_switch_rate": condition_candidate_summary[
                            "correct"
                        ][row]["adjacent_switch"],
                        "candidate_part_switch_rate": condition_candidate_summary[
                            "correct"
                        ][row]["part_switch"],
                        "candidate_adjacent_switch_coverage": condition_candidate_summary[
                            "correct"
                        ][row]["adjacent_switch_coverage"],
                        "candidate_part_switch_coverage": condition_candidate_summary[
                            "correct"
                        ][row]["part_switch_coverage"],
                        "correct_shuffled_attention_jsd": float(
                            correct_shuffled_attention_jsd[row].item()
                        ),
                        "correct_motion_shuffle_attention_jsd": float(
                            correct_motion_attention_jsd[row].item()
                        ),
                        "learned_analytic_prior_jsd": float(
                            learned_prior_jsd[row].item()
                        ),
                        "learned_analytic_prior_spearman": _spearman_arrays(
                            learned_distribution[row].numpy(),
                            prior[row].detach().cpu().numpy(),
                        ),
                        "learned_candidate_quality_spearman": _spearman_arrays(
                            learned_distribution[row][
                                correct.candidate_mask[row].detach().cpu()
                            ].numpy(),
                            -candidate_distance[row][
                                correct.candidate_mask[row].detach().cpu().numpy()
                            ],
                        ),
                        "analytic_prior_candidate_quality_spearman": _spearman_arrays(
                            prior[row][correct.candidate_mask[row]]
                            .detach()
                            .cpu()
                            .numpy(),
                            -candidate_distance[row][
                                correct.candidate_mask[row].detach().cpu().numpy()
                            ],
                        ),
                        "correct_vs_off_rms": float(
                            (correct_prediction - off_prediction)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "correct_vs_shuffled_rms": float(
                            (correct_prediction - shuffled_prediction)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "correct_vs_motion_shuffle_rms": float(
                            (correct_prediction - motion_prediction)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        **{
                            f"candidate_{condition}_{key}": value
                            for condition in MEMORY_CONDITIONS
                            for key, value in condition_candidate_summary[condition][
                                row
                            ].items()
                        },
                        **{
                            f"gate_{condition}_mean": condition_gate_summary[
                                condition
                            ][row]
                            for condition in MEMORY_CONDITIONS
                        },
                        **{
                            f"sentence_delta_{condition}_mean_norm": (
                                condition_delta_summary[condition][row]
                            )
                            for condition in MEMORY_CONDITIONS
                        },
                        **{f"planner_{key}": value for key, value in raw_summary[row].items()},
                        **text_summary[row],
                        **{
                            f"fused_correct_{key}": value
                            for key, value in condition_outputs[
                                "correct_fused_summary"
                            ][row].items()
                        },
                    }
                )
            end_position = written + batch_size
            workspace.commit_passive(
                written,
                end_position,
                {
                    "queries": query_rows[row_offsets["queries"] :],
                    "slots": slot_rows[row_offsets["slots"] :],
                    "text_attention": text_attention_rows[
                        row_offsets["text_attention"] :
                    ],
                    "part_attention": part_attention_rows[
                        row_offsets["part_attention"] :
                    ],
                    "candidates": candidate_rows[row_offsets["candidates"] :],
                    "checks": batch_checks,
                },
                store,
            )
            written = end_position
            del (
                condition_outputs,
                off_outputs,
                null_outputs,
                condition_candidate_metrics,
                condition_candidate_summary,
                condition_gate_summary,
                condition_delta_summary,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if written != len(indices):
        raise RuntimeError(f"Passive sweep wrote {written} rows; expected {len(indices)}")
    return {
        "queries": query_rows,
        "slots": slot_rows,
        "text_attention": text_attention_rows,
        "part_attention": part_attention_rows,
        "candidates": candidate_rows,
        "checks": {
            "text_attention_replay_max_abs": replay_max,
            "instrumented_prediction_max_abs": instrumented_prediction_max_abs,
            "all_null_vs_memory_off_max_abs": all_null_off_max_abs,
            "duration_memory_on_off_max_abs": duration_parity_max_abs,
        },
    }


def _repeat_memory(memory: SentenceMemoryBatch, count: int) -> SentenceMemoryBatch:
    count = int(count)
    if memory.tokens.shape[0] != 1:
        raise ValueError("Causal memory expansion expects a single query")

    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value.expand(count, *value.shape[1:])

    return _replace_memory(
        memory,
        tokens=repeat(memory.tokens),
        token_mask=repeat(memory.token_mask),
        token_tau=repeat(memory.token_tau),
        candidate_mask=repeat(memory.candidate_mask),
        candidate_keys=repeat(memory.candidate_keys),
        scores=repeat(memory.scores),
        durations=repeat(memory.durations),
        duration_log_gap=repeat(memory.duration_log_gap),
        part_validity=repeat(memory.part_validity),
        ids=repeat(memory.ids),
        available=repeat(memory.available),
    )


def _intervention_transform(
    *,
    stage: str,
    slots: torch.Tensor,
    directions: torch.Tensor,
    signed_amplitude: torch.Tensor,
):
    slots = slots.long()

    def transform(_module, _args, _kwargs, output):
        if stage == "planner":
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Planner intervention expected (slots, tau)")
            values, tau = output
        else:
            if not torch.is_tensor(output):
                raise RuntimeError("Fused-slot intervention expected a tensor")
            values, tau = output, None
        if values.shape[0] != len(slots):
            raise RuntimeError("Intervention metadata and forward batch disagree")
        changed = values.clone()
        row = torch.arange(len(slots), device=values.device)
        changed[row, slots.to(values.device)] += (
            directions.to(device=values.device, dtype=values.dtype)
            * signed_amplitude.to(device=values.device, dtype=values.dtype)[:, None]
        )
        return (changed, tau) if stage == "planner" else changed

    return transform


def _causal_variant_outputs(
    *,
    model,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    memory: SentenceMemoryBatch,
    query_tau_1d: torch.Tensor,
    stage: str,
    directions: torch.Tensor,
    epsilon: float,
    microbatch: int,
) -> dict[str, torch.Tensor]:
    slot_count, direction_count, hidden = directions.shape
    if hidden != model.hypernetwork.context_hidden_dim:
        raise ValueError("Rademacher probe hidden dimension does not match model")
    slot_index = torch.arange(slot_count).repeat_interleave(direction_count * 2)
    direction_index = torch.arange(direction_count).repeat_interleave(2).repeat(
        slot_count
    )
    signs = torch.tensor([-1.0, 1.0]).repeat(slot_count * direction_count)
    probe = directions[slot_index, direction_index]
    amplitudes = signs * float(epsilon)
    module = (
        model.hypernetwork.text_planner
        if stage == "planner"
        else model.hypernetwork.fusion_norm
    )
    gathered: dict[str, list[torch.Tensor]] = defaultdict(list)
    total = len(slot_index)
    with torch.inference_mode():
        for start in range(0, total, int(microbatch)):
            stop = min(start + int(microbatch), total)
            count = stop - start
            transform = _intervention_transform(
                stage=stage,
                slots=slot_index[start:stop],
                directions=probe[start:stop],
                signed_amplitude=amplitudes[start:stop],
            )
            with ModuleOutputIntervention(module, transform):
                output = _run_model(
                    model,
                    text_tokens.expand(count, -1, -1),
                    text_mask.expand(count, -1),
                    query_tau_1d[None].expand(count, -1),
                    _repeat_memory(memory, count),
                )
            for output_name, model_name in (
                ("prediction", "prediction"),
                ("coarse", "coarse"),
                ("global", "global_correction_axis"),
                ("local", "local_correction_axis"),
            ):
                gathered[output_name].append(output[model_name].detach().cpu())
    result = {}
    for name, pieces in gathered.items():
        value = torch.cat(pieces, dim=0)
        result[name] = value.reshape(
            slot_count,
            direction_count,
            2,
            len(query_tau_1d),
            value.shape[-1],
        )
    return result


def _normalized_fk_parts(joints: torch.Tensor, vertices: torch.Tensor):
    parts = default_joint_parts_torch(joints, vertices)
    # `default_joint_parts_torch` exposes articulation-normalized hands as well
    # as a root-relative whole body.  Causal influence must retain wrist/path
    # displacement, so slice both hands from the root-relative wholebody tensor
    # instead of using the wrist-relative lhand/rhand entries.
    body_count = parts["body"].shape[1]
    left_count = parts["lhand"].shape[1]
    right_count = parts["rhand"].shape[1]
    # `wholebody` is normalized to the first concatenated upper-body point
    # (joint 12), while body is pelvis-relative.  Adding the pelvis-relative
    # location of joint 12 reconstructs pelvis-relative hands exactly.
    wholebody = parts["wholebody"] + parts["body"][:, 0:1, :]
    left_start = body_count
    right_start = left_start + left_count
    if wholebody.shape[1] != body_count + left_count + right_count:
        raise RuntimeError("Unexpected SMPL-X wholebody part layout")
    face = joints[:, -FACE_LANDMARK_COUNT:, :] - joints[:, 0:1, :]
    return (
        parts["body"],
        wholebody[:, left_start:right_start],
        wholebody[:, right_start:],
        face,
    )


def _fk_pair_response(
    plus: torch.Tensor,
    minus: torch.Tensor,
    *,
    fk,
    device: torch.device,
    chunk_size: int,
    divisor: float,
) -> torch.Tensor:
    """Return mean per-keypoint distances as ``[...,P]``."""

    if plus.shape != minus.shape or plus.shape[-1] != 256:
        raise ValueError("FK response expects matching compact-rot6d tensors")
    original = plus.shape[:-1]
    plus_flat = plus.reshape(-1, plus.shape[-1])
    minus_flat = minus.reshape_as(plus_flat)
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(plus_flat), int(chunk_size)):
            stop = min(start + int(chunk_size), len(plus_flat))
            count = stop - start
            values = torch.cat(
                [plus_flat[start:stop], minus_flat[start:stop]], dim=0
            ).to(device=device, dtype=torch.float32)
            joints, vertices = fk.forward_rot6d(values)
            plus_parts = _normalized_fk_parts(joints[:count], vertices[:count])
            minus_parts = _normalized_fk_parts(joints[count:], vertices[count:])
            response = torch.stack(
                [
                    (positive - negative)
                    .norm(dim=-1)
                    .mean(dim=-1)
                    / float(divisor)
                    for positive, negative in zip(plus_parts, minus_parts)
                ],
                dim=-1,
            )
            pieces.append(response.detach().cpu())
            del joints, vertices, values
    return torch.cat(pieces, dim=0).reshape(*original, len(PART_NAMES))


def _compact_pair_response(
    plus: torch.Tensor,
    minus: torch.Tensor,
    *,
    slices: Sequence[slice],
    divisor: float,
) -> torch.Tensor:
    values = []
    for part_slice in slices:
        difference = plus[..., part_slice] - minus[..., part_slice]
        values.append(
            difference.float().square().mean(dim=-1).sqrt() / float(divisor)
        )
    return torch.stack(values, dim=-1)


def _responses_from_variants(
    variants: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
    fk,
    device: torch.device,
    fk_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [S,R,2,T,D] -> [S,R,P,T]
    prediction = variants["prediction"]
    fk_response = _fk_pair_response(
        prediction[:, :, 1],
        prediction[:, :, 0],
        fk=fk,
        device=device,
        chunk_size=fk_batch_size,
        divisor=2.0 * float(epsilon),
    ).permute(0, 1, 3, 2)

    compact_slices = (
        COMPACT6D_UPPER_BODY,
        COMPACT6D_LEFT_HAND,
        COMPACT6D_RIGHT_HAND,
        slice(COMPACT6D_JAW.start, COMPACT6D_EXPRESSION.stop),
    )
    residual_slices = tuple(LOCAL_PART_SLICES[name] for name in PART_NAMES)
    branches = []
    for name, slices in (
        ("coarse", compact_slices),
        ("global", residual_slices),
        ("local", residual_slices),
    ):
        value = variants[name]
        response = _compact_pair_response(
            value[:, :, 1],
            value[:, :, 0],
            slices=slices,
            divisor=2.0 * float(epsilon),
        )
        # [S,R,T,P] -> [S,R,P,T]
        branches.append(response.permute(0, 1, 3, 2))
    branch_response = torch.stack(branches, dim=2)  # [S,R,B,P,T]
    return fk_response, branch_response


def _causal_fk_baselines(
    *,
    model,
    text_tokens,
    text_mask,
    memory,
    query_tau,
    fk,
    device,
    fk_batch_size,
):
    with torch.inference_mode():
        on = _run_model(
            model, text_tokens, text_mask, query_tau[None], memory
        )["prediction"][0].detach().cpu()
        off = _run_model(
            model, text_tokens, text_mask, query_tau[None], None
        )["prediction"][0].detach().cpu()
        null = _run_model(
            model, text_tokens, text_mask, query_tau[None], _all_null(memory)
        )["prediction"][0].detach().cpu()
    memory_effect = _fk_pair_response(
        on,
        off,
        fk=fk,
        device=device,
        chunk_size=fk_batch_size,
        divisor=1.0,
    ).transpose(0, 1)
    parity_effect = _fk_pair_response(
        null,
        off,
        fk=fk,
        device=device,
        chunk_size=fk_batch_size,
        divisor=1.0,
    ).transpose(0, 1)
    return memory_effect, parity_effect


def _append_locality_rows(
    rows: list[dict[str, Any]],
    *,
    query_id: str,
    stage: str,
    epsilon: float,
    direction_response: torch.Tensor,
    slot_tau: torch.Tensor,
    query_tau: torch.Tensor,
    memory_effect: torch.Tensor,
    parity_noise: torch.Tensor,
):
    # Treat directions as independent probes. Per-slot spatial measurements and
    # the across-slot temporal correlation have different estimands, so export
    # them as separate row scopes rather than duplicating one rho 16 times.
    response = direction_response.permute(1, 0, 2, 3)  # [R,S,P,T]
    metrics = compute_locality_metrics(
        response,
        slot_tau,
        query_tau,
        local_window=0.25,
    )
    direction_count, slot_count, part_count, _times = response.shape
    response_mean = response.float().mean(dim=-1)
    memory_mean = memory_effect.float().mean(dim=-1)
    numerical_noise = parity_noise.float().amax(dim=-1).clamp_min(1e-12)
    informative_mask = (
        float(epsilon) * response_mean > 100.0 * numerical_noise[None, None, :]
    ) & (
        float(epsilon) * response_mean >= 0.01 * memory_mean[None, None, :]
    )
    for direction in range(direction_count):
        for part in range(part_count):
            usable = informative_mask[direction, :, part]
            usable_indices = usable.nonzero(as_tuple=False).flatten()
            rho = None
            if usable_indices.numel() >= 2:
                rho = _spearman_arrays(
                    slot_tau[usable_indices].detach().cpu().numpy(),
                    metrics["response_center_tau"][
                        direction, usable_indices, part
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                )
            rows.append(
                {
                    "query_id": query_id,
                    "stage": stage,
                    "epsilon": float(epsilon),
                    "direction": direction,
                    "slot": None,
                    "part": PART_NAMES[part],
                    "metric_scope": "direction",
                    "slot_response_time_spearman": rho,
                    "informative_slot_count": int(usable_indices.numel()),
                }
            )
            for slot in range(slot_count):
                row_response_mean = float(response_mean[direction, slot, part].item())
                rows.append(
                    {
                        "query_id": query_id,
                        "stage": stage,
                        "epsilon": float(epsilon),
                        "direction": direction,
                        "slot": slot,
                        "part": PART_NAMES[part],
                        "metric_scope": "slot",
                        "response_mean": row_response_mean,
                        "response_total": float(
                            metrics["response_total"][direction, slot, part].item()
                        ),
                        "response_center_tau": float(
                            metrics["response_center_tau"][direction, slot, part].item()
                        ),
                        "response_center_absolute_error": float(
                            metrics["response_center_absolute_error"][
                                direction, slot, part
                            ].item()
                        ),
                        "local_response_fraction": float(
                            metrics["local_response_fraction"][direction, slot, part]
                            .item()
                        ),
                        "far_leakage_fraction": float(
                            metrics["far_leakage_fraction"][direction, slot, part]
                            .item()
                        ),
                        "radius_50": float(
                            metrics["radius_50"][direction, slot, part].item()
                        ),
                        "radius_80": float(
                            metrics["radius_80"][direction, slot, part].item()
                        ),
                        "memory_on_off_mean": float(memory_mean[part].item()),
                        "numerical_noise": float(numerical_noise[part].item()),
                        "informative": bool(usable[slot].item()),
                    }
                )


def _causal_sweep(
    *,
    model,
    fk,
    text_encoder,
    provider,
    dataset,
    selection,
    passive_queries: Sequence[Mapping[str, Any]],
    cfg,
    device: torch.device,
    args,
    store: ArrayStore,
    workspace: _ResumeWorkspace,
):
    query_tau = torch.linspace(
        -1.0, 1.0, int(args.query_points), device=device, dtype=torch.float32
    )
    slot_tau = model.hypernetwork.text_planner.slot_tau(device="cpu", dtype=torch.float32)
    causal_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    epsilons = (float(args.epsilon), float(args.epsilon) / 2.0)

    start_position = int(workspace.causal_next)
    selected = list(
        zip(selection.indices, selection.duration_bins, selection.margin_bins)
    )
    for causal_position, (passive_position, duration_bin, margin_bin) in enumerate(
        tqdm(
            selected[start_position:],
            total=len(selected) - start_position,
            desc="causal validation",
            unit="query",
        ),
        start=start_position,
    ):
        causal_row_start = len(causal_rows)
        selection_row_start = len(selection_rows)
        passive = passive_queries[int(passive_position)]
        dataset_index = int(passive["dataset_index"])
        batch = collate_continuous_sign([dataset[dataset_index]])
        query_id = _query_ids(batch)[0]
        if query_id != passive["query_id"]:
            raise RuntimeError("Causal query identity changed after passive selection")
        text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
        with torch.inference_mode():
            _log_duration, predicted_duration = model.predict_duration(
                text_tokens, text_mask=text_mask
            )
            memory = retrieve_sentence_memory(
                provider,
                dataset=dataset,
                batch=batch,
                text_tokens=text_tokens,
                text_mask=text_mask,
                predicted_duration=predicted_duration.detach(),
                available=torch.ones(1, dtype=torch.bool, device=device),
                training=False,
                device=device,
                mode="on",
            )
        memory_effect, parity_noise = _causal_fk_baselines(
            model=model,
            text_tokens=text_tokens,
            text_mask=text_mask,
            memory=memory,
            query_tau=query_tau,
            fk=fk,
            device=device,
            fk_batch_size=args.fk_batch_size,
        )
        store["memory_on_off_fk_difference"][causal_position] = (
            memory_effect.numpy()
        )
        store["all_null_off_fk_parity"][causal_position] = parity_noise.numpy()
        store["causal_dataset_index"][causal_position] = dataset_index
        selection_rows.append(
            {
                "causal_position": causal_position,
                "passive_position": int(passive_position),
                "dataset_index": dataset_index,
                "query_id": query_id,
                "name": passive["name"],
                "text": passive["text"],
                "duration_bin": int(duration_bin),
                "margin_bin": int(margin_bin),
            }
        )

        for stage_index, stage in enumerate(INTERVENTION_STAGES):
            directions = deterministic_rademacher(
                (
                    model.hypernetwork.temporal_slot_count,
                    int(args.directions),
                    model.hypernetwork.context_hidden_dim,
                ),
                # Use identical directions for planner and fused interventions
                # so their Jacobian scales/locality remain directly comparable.
                seed=stable_int_seed("slot-probe", query_id, base_seed=args.seed),
                device="cpu",
            )
            store["rademacher_directions"][causal_position, stage_index] = (
                directions.numpy().astype(np.float16)
            )
            stage_responses = []
            for epsilon_index, epsilon in enumerate(epsilons):
                variants = _causal_variant_outputs(
                    model=model,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    memory=memory,
                    query_tau_1d=query_tau,
                    stage=stage,
                    directions=directions,
                    epsilon=epsilon,
                    microbatch=args.perturb_batch_size,
                )
                fk_response, branch_response = _responses_from_variants(
                    variants,
                    epsilon=epsilon,
                    fk=fk,
                    device=device,
                    fk_batch_size=args.fk_batch_size,
                )
                store["causal_fk_response"][
                    causal_position, stage_index, epsilon_index
                ] = fk_response.permute(1, 0, 2, 3).numpy().astype(np.float16)
                store["causal_branch_response"][
                    causal_position, stage_index, epsilon_index
                ] = branch_response.permute(1, 2, 0, 3, 4).numpy().astype(
                    np.float16
                )
                _append_locality_rows(
                    causal_rows,
                    query_id=query_id,
                    stage=stage,
                    epsilon=epsilon,
                    direction_response=fk_response,
                    slot_tau=slot_tau,
                    query_tau=query_tau.detach().cpu(),
                    memory_effect=memory_effect,
                    parity_noise=parity_noise,
                )
                stage_responses.append(fk_response)
                del variants, fk_response, branch_response
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # [S,R,P,T] -> [S,R,P].  Epsilon consistency is meaningful only
            # for the same partwise probes that exceeded the FK-unit
            # detectability thresholds at the main epsilon.
            large_total = stage_responses[0].float().mean(dim=-1)
            small_total = stage_responses[1].float().mean(dim=-1)
            ratio = small_total / large_total.clamp_min(1e-12)
            for slot in range(ratio.shape[0]):
                for direction in range(ratio.shape[1]):
                    for part, part_name in enumerate(PART_NAMES):
                        parity_floor = max(
                            1e-12,
                            float(parity_noise[part].float().max().item()),
                        )
                        large_effect = float(args.epsilon) * float(
                            large_total[slot, direction, part].item()
                        )
                        small_effect = (float(args.epsilon) / 2.0) * float(
                            small_total[slot, direction, part].item()
                        )
                        memory_mean = float(memory_effect[part].float().mean().item())
                        informative = (
                            large_effect > 100.0 * parity_floor
                            and large_effect >= 0.01 * memory_mean
                            and small_effect > 100.0 * parity_floor
                            and small_effect >= 0.01 * memory_mean
                        )
                        ratio_value = float(ratio[slot, direction, part].item())
                        causal_rows.append(
                            {
                                "query_id": query_id,
                                "stage": stage,
                                "epsilon": "consistency",
                                "direction": direction,
                                "slot": slot,
                                "part": part_name,
                                "informative": informative,
                                "jacobian_scale_ratio_half_over_full": ratio_value,
                                "consistent_0p8_to_1p25": bool(
                                    informative and 0.8 <= ratio_value <= 1.25
                                ),
                            }
                        )
        workspace.commit_causal(
            causal_position,
            causal_position + 1,
            {
                "rows": causal_rows[causal_row_start:],
                "selection": selection_rows[selection_row_start:],
            },
            store,
        )
    return causal_rows, selection_rows


def _bootstrap_summary(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    novel = [row for row in rows if bool(row.get("novel_text", True))]
    clusters = [str(row.get("normalized_text") or row.get("query_id")) for row in novel]
    result = {}
    for field in fields:
        selected = [row for row in novel if row.get(field) is not None]
        if len(selected) != len(novel):
            continue
        result[field] = cluster_bootstrap_interval(
            [float(row[field]) for row in novel],
            clusters,
            samples=int(samples),
            seed=stable_int_seed("bootstrap", field, base_seed=seed),
        )
    return result


def _causal_aggregate(
    causal_rows: Sequence[Mapping[str, Any]],
    query_lookup: Mapping[str, Mapping[str, Any]],
    *,
    epsilon: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fields = (
        "response_mean",
        "response_center_absolute_error",
        "local_response_fraction",
        "far_leakage_fraction",
        "radius_50",
        "radius_80",
    )
    for stage in INTERVENTION_STAGES:
        output[stage] = {}
        for part in PART_NAMES:
            selected = [
                row
                for row in causal_rows
                if row.get("stage") == stage
                and row.get("part") == part
                and row.get("metric_scope") == "slot"
                and isinstance(row.get("epsilon"), float)
                and math.isclose(float(row["epsilon"]), float(epsilon))
            ]
            by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in selected:
                by_query[str(row["query_id"])].append(row)
            query_metrics = []
            for query_id in sorted(by_query):
                entries = by_query[query_id]
                query_metrics.append(
                    {
                        "query_id": query_id,
                        "normalized_text": query_lookup[query_id]["normalized_text"],
                        **{
                            field: float(np.mean([float(row[field]) for row in entries]))
                            for field in fields
                        },
                        "informative_fraction": float(
                            np.mean([bool(row["informative"]) for row in entries])
                        ),
                    }
                )
            clusters = [row["normalized_text"] for row in query_metrics]
            summary = {}
            for field in (*fields, "informative_fraction"):
                summary[field] = cluster_bootstrap_interval(
                    [row[field] for row in query_metrics],
                    clusters,
                    samples=int(samples),
                    seed=stable_int_seed(
                        "causal-bootstrap", stage, part, field, base_seed=seed
                    ),
                )
            informative_query_metrics = []
            for query_id in sorted(by_query):
                entries = [row for row in by_query[query_id] if bool(row["informative"])]
                if not entries:
                    continue
                informative_query_metrics.append(
                    {
                        "query_id": query_id,
                        "normalized_text": query_lookup[query_id]["normalized_text"],
                        **{
                            field: float(np.mean([float(row[field]) for row in entries]))
                            for field in fields
                        },
                    }
                )
            summary["informative_query_count"] = len(informative_query_metrics)
            if informative_query_metrics:
                informative_clusters = [
                    row["normalized_text"] for row in informative_query_metrics
                ]
                summary["informative_only"] = {
                    field: cluster_bootstrap_interval(
                        [row[field] for row in informative_query_metrics],
                        informative_clusters,
                        samples=int(samples),
                        seed=stable_int_seed(
                            "causal-informative-bootstrap",
                            stage,
                            part,
                            field,
                            base_seed=seed,
                        ),
                    )
                    for field in fields
                }
            correlation_rows = [
                row
                for row in causal_rows
                if row.get("stage") == stage
                and row.get("part") == part
                and row.get("metric_scope") == "direction"
                and isinstance(row.get("epsilon"), float)
                and math.isclose(float(row["epsilon"]), float(epsilon))
                and row.get("slot_response_time_spearman") is not None
            ]
            correlation_by_query: dict[str, list[float]] = defaultdict(list)
            for row in correlation_rows:
                correlation_by_query[str(row["query_id"])].append(
                    float(row["slot_response_time_spearman"])
                )
            summary["temporal_correlation_query_count"] = len(correlation_by_query)
            if correlation_by_query:
                correlation_ids = sorted(correlation_by_query)
                correlation_interval = cluster_bootstrap_interval(
                    [
                        float(np.mean(correlation_by_query[key]))
                        for key in correlation_ids
                    ],
                    [query_lookup[key]["normalized_text"] for key in correlation_ids],
                    samples=int(samples),
                    seed=stable_int_seed(
                        "causal-informative-bootstrap",
                        stage,
                        part,
                        "slot_response_time_spearman",
                        base_seed=seed,
                    ),
                )
                summary.setdefault("informative_only", {})[
                    "slot_response_time_spearman"
                ] = correlation_interval
            consistency_all = [
                row
                for row in causal_rows
                if row.get("stage") == stage
                and row.get("part") == part
                and row.get("epsilon") == "consistency"
                and row.get("jacobian_scale_ratio_half_over_full") is not None
            ]
            consistency = [row for row in consistency_all if bool(row.get("informative"))]
            consistency_all_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(
                list
            )
            for row in consistency_all:
                consistency_all_by_query[str(row["query_id"])].append(row)
            if consistency_all_by_query:
                coverage_ids = sorted(consistency_all_by_query)
                summary["epsilon_scale_informative_fraction"] = (
                    cluster_bootstrap_interval(
                        [
                            float(
                                np.mean(
                                    [
                                        bool(row["informative"])
                                        for row in consistency_all_by_query[key]
                                    ]
                                )
                            )
                            for key in coverage_ids
                        ],
                        [query_lookup[key]["normalized_text"] for key in coverage_ids],
                        samples=int(samples),
                        seed=stable_int_seed(
                            "epsilon-scale-coverage", stage, part, base_seed=seed
                        ),
                    )
                )
            consistency_by_query: dict[str, list[float]] = defaultdict(list)
            for row in consistency:
                consistency_by_query[str(row["query_id"])].append(
                    float(row["jacobian_scale_ratio_half_over_full"])
                )
            summary["epsilon_scale_informative_query_count"] = len(
                consistency_by_query
            )
            if consistency_by_query:
                consistency_ids = sorted(consistency_by_query)
                summary["epsilon_scale_ratio"] = cluster_bootstrap_interval(
                    [
                        float(np.mean(consistency_by_query[key]))
                        for key in consistency_ids
                    ],
                    [query_lookup[key]["normalized_text"] for key in consistency_ids],
                    samples=int(samples),
                    seed=stable_int_seed(
                        "epsilon-scale-bootstrap", stage, part, base_seed=seed
                    ),
                )
                summary["epsilon_scale_consistent_fraction"] = (
                    cluster_bootstrap_interval(
                        [
                            float(
                                np.mean(
                                    [
                                        0.8 <= value <= 1.25
                                        for value in consistency_by_query[key]
                                    ]
                                )
                            )
                            for key in consistency_ids
                        ],
                        [
                            query_lookup[key]["normalized_text"]
                            for key in consistency_ids
                        ],
                        samples=int(samples),
                        seed=stable_int_seed(
                            "epsilon-scale-consistency", stage, part, base_seed=seed
                        ),
                    )
                )
            output[stage][part] = summary
    return output


def _causal_permutation_tests(
    store: ArrayStore,
    *,
    causal_count: int,
    query_points: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test temporal locality against a deterministic slot-label null."""

    slot_count = store["causal_fk_response"].shape[4]
    slot_tau = torch.linspace(-1.0, 1.0, slot_count)
    query_tau = torch.linspace(-1.0, 1.0, int(query_points))
    permutation = deterministic_permutations(
        int(permutations), slot_count, seed=stable_int_seed("locality-null", base_seed=seed)
    )
    output: dict[str, Any] = {}
    raw_p_values: dict[str, list[float]] = {}
    for stage_index, stage in enumerate(INTERVENTION_STAGES):
        output[stage] = {}
        raw_p_values[stage] = []
        # Main epsilon (index zero), average deterministic directions.
        response = torch.from_numpy(
            np.asarray(
                store["causal_fk_response"][:causal_count, stage_index, 0],
                dtype=np.float32,
            )
        ).mean(dim=1)  # [N,S,P,T]
        observed = compute_locality_metrics(response, slot_tau, query_tau)
        for part, part_name in enumerate(PART_NAMES[:3]):
            observed_value = float(
                observed["local_response_fraction"][:, :, part].mean().item()
            )
            null_values = []
            for start in range(0, len(permutation), 32):
                current = permutation[start : start + 32]
                # Permuting response rows breaks their association with slot tau.
                expanded = response[:, None].expand(-1, len(current), -1, -1, -1)
                gather = current[None, :, :, None, None].expand(
                    response.shape[0], -1, -1, response.shape[2], response.shape[3]
                )
                permuted = torch.gather(expanded, 2, gather).reshape(
                    -1, slot_count, response.shape[2], response.shape[3]
                )
                metrics = compute_locality_metrics(permuted, slot_tau, query_tau)
                values = metrics["local_response_fraction"][..., part].reshape(
                    response.shape[0], len(current), slot_count
                )
                null_values.extend(values.mean(dim=(0, 2)).tolist())
            null_array = np.asarray(null_values, dtype=np.float64)
            p_value = float(
                (1 + np.count_nonzero(null_array >= observed_value))
                / (len(null_array) + 1)
            )
            raw_p_values[stage].append(p_value)
            output[stage][part_name] = {
                "observed_local_response_fraction": observed_value,
                "permutation_null_mean": float(null_array.mean()),
                "raw_one_sided_p": p_value,
            }
        adjusted = holm_adjust(raw_p_values[stage])
        for part_name, value in zip(PART_NAMES[:3], adjusted):
            output[stage][part_name]["holm_adjusted_p"] = float(value)
    return output


def _text_attention_permutation_test(
    store: ArrayStore,
    queries: Sequence[Mapping[str, Any]],
    *,
    rows: int,
    batch_size: int,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    if int(rows) < 1 or int(batch_size) < 1:
        raise ValueError("rows and batch_size must be positive")
    attention_array = store["text_attention"]
    mask_array = store["text_mask"]
    persisted_array = store["text_expected_token_position_rank"]
    if (
        attention_array.ndim != 5
        or mask_array.ndim != 2
        or persisted_array.ndim != 4
        or attention_array.shape[0] < rows
        or mask_array.shape[0] < rows
        or persisted_array.shape[0] < rows
        or attention_array.shape[1:4] != persisted_array.shape[1:]
        or attention_array.shape[-1] != mask_array.shape[-1]
    ):
        raise RuntimeError(
            "Stored text attention, mask, and expected-position ranks have "
            "incompatible shapes"
        )

    # Raw persisted attention remains authoritative. Reconstruct the exact
    # primary rank for every row before novel-text filtering. Each chunk and
    # token extent mirror the identity-bound passive DataLoader batch because
    # floating-point reduction geometry can affect a threshold near-tie.
    persisted_all = np.asarray(persisted_array[:rows], dtype=np.float32)
    reconstructed = torch.empty(tuple(persisted_all.shape), dtype=torch.float32)
    for start in range(0, int(rows), int(batch_size)):
        stop = min(start + int(batch_size), int(rows))
        batch_mask_array = np.asarray(mask_array[start:stop], dtype=np.bool_)
        token_counts = batch_mask_array.sum(axis=-1, dtype=np.int64)
        if bool((token_counts < 1).any()):
            raise RuntimeError("Stored text mask contains an empty query")
        contiguous_mask = (
            np.arange(batch_mask_array.shape[-1])[None, :]
            < token_counts[:, None]
        )
        if not bool(np.array_equal(batch_mask_array, contiguous_mask)):
            raise RuntimeError("Stored text mask is not contiguous from token zero")
        token_extent = int(token_counts.max())
        if bool(
            np.count_nonzero(
                np.asarray(
                    attention_array[start:stop, ..., token_extent:],
                    dtype=np.float32,
                )
            )
        ):
            raise RuntimeError(
                "Stored text attention has nonzero mass beyond its passive "
                f"batch token extent in rows [{start},{stop})"
            )
        probability = torch.from_numpy(
            np.array(
                attention_array[start:stop, ..., :token_extent],
                dtype=np.float32,
                copy=True,
            )
        )
        batch_mask = torch.from_numpy(
            np.array(batch_mask_array[:, :token_extent], copy=True)
        )
        try:
            _expected_position, batch_rank = (
                normalized_text_attention_position_ranks(
                    probability,
                    batch_mask,
                )
            )
        except ValueError as error:
            raise RuntimeError(
                "Stored normalized text attention failed source validation in "
                f"passive batch [{start},{stop}): {error}"
            ) from error
        reconstructed[start:stop] = batch_rank
    persisted_tensor = torch.from_numpy(np.array(persisted_all, copy=True))
    if not torch.equal(reconstructed, persisted_tensor):
        mismatch = reconstructed != persisted_tensor
        first = tuple(int(value) for value in torch.nonzero(mismatch)[0].tolist())
        max_abs = float(
            (reconstructed - persisted_tensor).abs().max().item()
        )
        raise RuntimeError(
            "Persisted text expected-position ranks do not match their source "
            f"attention: first_index={first}, max_abs={max_abs:.9g}"
        )

    novel_indices = [
        index for index, row in enumerate(queries[:rows]) if bool(row["novel_text"])
    ]
    if not novel_indices:
        raise RuntimeError("Text-attention permutation test has no novel-text rows")
    persisted_rank = persisted_all[novel_indices]
    if persisted_rank.ndim != 4 or persisted_rank.shape[-1] < 1:
        raise RuntimeError(
            "Persisted text expected-position ranks must have shape [N,D,H,S]"
        )
    if not bool(np.isfinite(persisted_rank).all()):
        raise RuntimeError("Persisted text expected-position ranks are non-finite")
    slot_count = int(persisted_rank.shape[-1])
    expected_rank_sum = 0.5 * slot_count * (slot_count - 1)
    rank_sum_error = float(
        np.max(
            np.abs(
                persisted_rank.sum(axis=-1, dtype=np.float64)
                - expected_rank_sum
            ),
            initial=0.0,
        )
    )
    if (
        float(persisted_rank.min(initial=0.0)) < 0.0
        or float(persisted_rank.max(initial=0.0)) > slot_count - 1
        or rank_sum_error > 1e-6
    ):
        raise RuntimeError(
            "Persisted text expected-position ranks violate average-rank bounds "
            f"or conservation: max_sum_error={rank_sum_error:.9g}"
        )
    by_text: dict[str, list[int]] = defaultdict(list)
    for local_index, source_index in enumerate(novel_indices):
        by_text[str(queries[source_index]["normalized_text"])].append(local_index)

    # Reproduce the float32 rank correlation used by the primary metric.  The
    # rank tensor is persisted at passive-inference time because recomputing
    # expected token positions through NumPy changes the reduction precision;
    # values at the near-tie threshold can then receive different ranks.
    rank_tensor = torch.from_numpy(np.array(persisted_rank, copy=True))
    slot_rank_tensor = torch.arange(slot_count, dtype=rank_tensor.dtype)
    observed_per_head_tensor = rank_correlation_last_dim(
        slot_rank_tensor.view(1, 1, 1, -1),
        rank_tensor,
    )
    observed_per_row = (
        observed_per_head_tensor.mean(dim=(1, 2)).cpu().numpy().astype(np.float64)
    )

    ranks = persisted_rank.reshape(-1, slot_count).astype(np.float64)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    ranks /= np.maximum(np.linalg.norm(ranks, axis=-1, keepdims=True), 1e-12)
    slot = np.arange(slot_count, dtype=np.float64)
    slot -= slot.mean()
    slot /= np.linalg.norm(slot)
    primary_per_row = np.asarray(
        [float(queries[index]["text_spearman"]) for index in novel_indices],
        dtype=np.float64,
    )
    replay_primary_max_abs = float(
        np.max(np.abs(observed_per_row - primary_per_row), initial=0.0)
    )
    if replay_primary_max_abs != 0.0:
        raise RuntimeError(
            "Persisted text-attention ranks differ from the live "
            f"primary metric: max_abs={replay_primary_max_abs:.9g}"
        )
    cluster_weight = np.zeros(len(novel_indices), dtype=np.float64)
    for source_rows in by_text.values():
        cluster_weight[source_rows] = 1.0 / (
            len(by_text) * max(len(source_rows), 1)
        )
    observed = float(np.dot(observed_per_row, cluster_weight))
    order = deterministic_permutations(
        int(permutations),
        slot_count,
        seed=stable_int_seed("text-slot-null", base_seed=seed),
    ).numpy()
    null = []
    for start in range(0, len(order), 32):
        current = order[start : start + 32]
        permuted = ranks[:, current]
        per_head = np.einsum("mcs,s->mc", permuted, slot).reshape(
            *persisted_rank.shape[:-1], len(current)
        )
        per_row = per_head.mean(axis=(1, 2))
        null.extend(np.einsum("nc,n->c", per_row, cluster_weight))
    null_array = np.asarray(null, dtype=np.float64)
    return {
        "novel_text_clusters": len(by_text),
        "stored_vs_primary_max_abs": replay_primary_max_abs,
        "observed_mean_spearman": observed,
        "permutation_null_mean": float(null_array.mean()),
        "permutation_null_std": float(null_array.std()),
        "one_sided_p": float(
            (1 + np.count_nonzero(null_array >= observed)) / (len(null_array) + 1)
        ),
    }


def _artifact_checks(store: ArrayStore, *, rows: int, causal_rows: int) -> None:
    if bool((store["query_dataset_index"][:rows] < 0).any()):
        raise RuntimeError("Not every passive query row was written")
    if bool((store["causal_dataset_index"][:causal_rows] < 0).any()):
        raise RuntimeError("Not every causal query row was written")
    for name, array in store.arrays.items():
        if np.issubdtype(array.dtype, np.floating):
            flat = array.reshape(-1)
            for start in range(0, len(flat), 4_194_304):
                if not bool(np.isfinite(flat[start : start + 4_194_304]).all()):
                    raise RuntimeError(f"Non-finite values found in arrays/{name}.npy")


def _plot_heatmap(
    axis,
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    image = axis.imshow(values, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return image


def _write_plots(
    out_dir: Path,
    store: ArrayStore,
    queries: Sequence[Mapping[str, Any]],
    causal_rows: Sequence[Mapping[str, Any]],
    *,
    rows: int,
    causal_count: int,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for diagnostic figures") from error

    plot_dir = out_dir / "plots"
    plot_dir.mkdir()
    planner = np.asarray(store["planner_slots"][:rows], dtype=np.float32)
    fused = np.asarray(store["fused_slots"][:rows, 0], dtype=np.float32)

    def cosine(values):
        values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
        return np.einsum("nsh,nth->nst", values, values).mean(axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (cosine(planner), cosine(fused)),
        ("Planner slots", "Fused slots: correct retrieval"),
    ):
        image = _plot_heatmap(
            axis, values, title=title, xlabel="slot", ylabel="slot", vmin=-1, vmax=1
        )
        figure.colorbar(image, ax=axis)
    figure.savefig(plot_dir / "slot_cosine.png", dpi=180)
    plt.close(figure)

    planner_cosine = cosine(planner)
    fused_cosine = cosine(fused)
    lag = np.arange(1, planner_cosine.shape[0])
    planner_lag = np.asarray(
        [np.diagonal(planner_cosine, offset=int(value)).mean() for value in lag]
    )
    fused_lag = np.asarray(
        [np.diagonal(fused_cosine, offset=int(value)).mean() for value in lag]
    )
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(lag, planner_lag, marker="o", label="planner")
    axis.plot(lag, fused_lag, marker="o", label="fused/correct")
    axis.set_xlabel("slot lag")
    axis.set_ylabel("mean cosine")
    axis.set_title("Temporal-slot similarity by lag")
    axis.legend()
    figure.savefig(plot_dir / "slot_lag.png", dpi=180)
    plt.close(figure)

    attention = np.asarray(store["text_attention"][:rows], dtype=np.float32).mean(
        axis=(0, 1, 2)
    )
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    image = _plot_heatmap(
        axis,
        attention,
        title="Mean planner cross-attention",
        xlabel="mT5 token position",
        ylabel="temporal slot",
    )
    figure.colorbar(image, ax=axis)
    figure.savefig(plot_dir / "text_attention.png", dpi=180)
    plt.close(figure)

    candidate = np.asarray(store["candidate_mass"][:rows, 0, -1], dtype=np.float32)
    real_sum = candidate.sum(axis=-1, keepdims=True)
    candidate = np.divide(candidate, np.maximum(real_sum, 1e-8)).mean(axis=(0, 2))
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = _plot_heatmap(
        axis,
        candidate,
        title="Conditional candidate attention (last layer)",
        xlabel="retrieved candidate rank",
        ylabel="temporal slot",
    )
    figure.colorbar(image, ax=axis)
    figure.savefig(plot_dir / "candidate_attention.png", dpi=180)
    plt.close(figure)

    gates = np.asarray(store["sentence_gates"][:rows], dtype=np.float32).mean(axis=0)
    null = np.asarray(store["part_null_mass"][:rows, :, -1], dtype=np.float32).mean(
        axis=0
    )
    figure, axes = plt.subplots(
        2,
        len(MEMORY_CONDITIONS),
        figsize=(15, 7),
        constrained_layout=True,
    )
    for condition_index, condition in enumerate(MEMORY_CONDITIONS):
        gate_image = _plot_heatmap(
            axes[0, condition_index],
            gates[condition_index].T,
            title=f"{condition}: gate",
            xlabel="slot",
            ylabel="part",
            vmin=0,
            vmax=1,
        )
        null_image = _plot_heatmap(
            axes[1, condition_index],
            null[condition_index].T,
            title=f"{condition}: null mass",
            xlabel="slot",
            ylabel="part",
            vmin=0,
            vmax=1,
        )
        figure.colorbar(gate_image, ax=axes[0, condition_index])
        figure.colorbar(null_image, ax=axes[1, condition_index])
    figure.savefig(plot_dir / "gate_null_timelines.png", dpi=180)
    plt.close(figure)

    response = np.asarray(
        store["causal_fk_response"][:causal_count, :, 0], dtype=np.float32
    ).mean(axis=(0, 2))  # [stage,S,P,T]
    figure, axes = plt.subplots(
        len(INTERVENTION_STAGES),
        len(PART_NAMES),
        figsize=(16, 7),
        constrained_layout=True,
    )
    for stage_index, stage in enumerate(INTERVENTION_STAGES):
        for part, part_name in enumerate(PART_NAMES):
            image = _plot_heatmap(
                axes[stage_index, part],
                response[stage_index, :, part],
                title=f"{stage}: {part_name}",
                xlabel="trajectory time",
                ylabel="perturbed slot",
            )
            figure.colorbar(image, ax=axes[stage_index, part])
    figure.savefig(plot_dir / "causal_influence.png", dpi=180)
    plt.close(figure)

    # Representative categories are selected by fixed scalar rules, never by
    # test results or qualitative inspection.
    category_indices = {
        "most_collapsed": max(
            range(rows), key=lambda index: queries[index]["planner_off_diagonal_mean"]
        ),
        "most_diverse": min(
            range(rows), key=lambda index: queries[index]["planner_off_diagonal_mean"]
        ),
        "most_uniform_candidate_attention": max(
            range(rows), key=lambda index: queries[index]["candidate_effective_k"]
        ),
        "most_selective_candidate_attention": min(
            range(rows), key=lambda index: queries[index]["candidate_effective_k"]
        ),
    }
    locality_by_query: dict[str, list[float]] = defaultdict(list)
    for row in causal_rows:
        if (
            row.get("stage") == "fused"
            and row.get("part") in PART_NAMES[:3]
            and row.get("metric_scope") == "slot"
            and isinstance(row.get("epsilon"), float)
            and math.isclose(float(row["epsilon"]), 0.10)
            and bool(row.get("informative"))
        ):
            locality_by_query[str(row["query_id"])].append(
                float(row["local_response_fraction"])
            )
    if locality_by_query:
        best_id = max(locality_by_query, key=lambda key: np.mean(locality_by_query[key]))
        worst_id = min(locality_by_query, key=lambda key: np.mean(locality_by_query[key]))
        by_id = {str(row["query_id"]): index for index, row in enumerate(queries)}
        category_indices["best_temporal_locality"] = by_id[best_id]
        category_indices["worst_temporal_locality"] = by_id[worst_id]
    representatives = []
    for category, index in category_indices.items():
        representatives.append(
            {
                "category": category,
                "row_position": int(index),
                "query_id": queries[index]["query_id"],
                "text": queries[index]["text"],
            }
        )
        figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        token_image = _plot_heatmap(
            axes[0],
            np.asarray(store["text_attention"][index], dtype=np.float32).mean(axis=(0, 1)),
            title=f"{category}: text attention",
            xlabel="token",
            ylabel="slot",
        )
        figure.colorbar(token_image, ax=axes[0])
        candidate_value = np.asarray(
            store["candidate_mass"][index, 0, -1], dtype=np.float32
        ).mean(axis=1)
        candidate_value /= np.maximum(candidate_value.sum(axis=-1, keepdims=True), 1e-8)
        candidate_image = _plot_heatmap(
            axes[1],
            candidate_value,
            title="candidate attention",
            xlabel="candidate",
            ylabel="slot",
        )
        figure.colorbar(candidate_image, ax=axes[1])
        figure.savefig(plot_dir / f"representative_{category}.png", dpi=180)
        plt.close(figure)
    _write_json(out_dir / "representative_examples.json", representatives)
    return representatives


def _interval_value(summary: Mapping[str, Any], *path: str, field: str = "estimate"):
    value: Any = summary
    for key in path:
        value = value[key]
    if dataclasses.is_dataclass(value):
        value = getattr(value, field)
    elif isinstance(value, Mapping):
        value = value[field]
    return float(np.asarray(value).mean())


def _interpretation(passive_summary, causal_summary) -> list[str]:
    raw_high_fraction = _interval_value(
        passive_summary, "planner_off_diagonal_fraction_above_0p95"
    )
    raw_rank = _interval_value(passive_summary, "planner_centered_effective_rank")
    fused_high_fraction = _interval_value(
        passive_summary, "fused_correct_off_diagonal_fraction_above_0p95"
    )
    fused_rank = _interval_value(passive_summary, "fused_correct_centered_effective_rank")
    text_rho_lower = _interval_value(
        passive_summary, "text_spearman", field="lower"
    )
    effective_k = _interval_value(passive_summary, "candidate_effective_k")
    max_share = _interval_value(passive_summary, "candidate_max_share")
    motion_change = _interval_value(
        passive_summary, "correct_vs_motion_shuffle_rms"
    )
    informative_fraction = _interval_value(
        causal_summary, "fused", "body", "informative_fraction"
    )
    informative_fraction_lower = _interval_value(
        causal_summary, "fused", "body", "informative_fraction", field="lower"
    )
    informative_summary = causal_summary["fused"]["body"].get("informative_only")
    if informative_summary:
        fused_body_local = _interval_value(
            informative_summary, "local_response_fraction"
        )
        fused_body_lower = _interval_value(
            informative_summary, "local_response_fraction", field="lower"
        )
        fused_center_error = _interval_value(
            informative_summary, "response_center_absolute_error"
        )
        fused_rho_lower = (
            _interval_value(
                informative_summary, "slot_response_time_spearman", field="lower"
            )
            if "slot_response_time_spearman" in informative_summary
            else -math.inf
        )
    else:
        fused_body_local = 0.0
        fused_body_lower = 0.0
        fused_center_error = math.inf
        fused_rho_lower = -math.inf
    epsilon_scale = causal_summary["fused"]["body"].get("epsilon_scale_ratio")
    epsilon_ratio = (
        _interval_value(epsilon_scale) if epsilon_scale is not None else math.nan
    )
    findings = []
    if raw_high_fraction >= 0.95 or raw_rank < 2.0:
        findings.append("Raw planner slots meet the predeclared severe-collapse rule.")
    else:
        findings.append("Raw planner slots do not meet the severe-collapse rule.")
    if fused_high_fraction >= 0.95 or fused_rank < 2.0:
        findings.append("Correct-retrieval fused slots meet the severe-collapse rule.")
    else:
        findings.append("Correct-retrieval fused slots retain more than rank-two structure.")
    if text_rho_lower > 0:
        findings.append("Slot-to-token position correlation is positive with a 95% CI above zero.")
    else:
        findings.append("Slot-to-token position correlation is not reliably positive.")
    if effective_k > 7.5 and max_share < 0.16:
        findings.append("Candidate attention meets the predeclared nearly-uniform rule.")
    else:
        findings.append("Candidate attention is detectably selective by the predeclared rule.")
    if motion_change <= 1e-7:
        findings.append("Motion-only candidate shuffling has no numerically meaningful output effect.")
    locality_coverage = informative_fraction >= 0.50 and informative_fraction_lower > 0.0
    locality_passed = (
        locality_coverage
        and fused_body_local >= 0.50
        and fused_body_lower >= 0.40
        and fused_center_error <= 0.30
        and fused_rho_lower > 0.0
        and math.isfinite(epsilon_ratio)
        and 0.8 <= epsilon_ratio <= 1.25
    )
    if not locality_coverage:
        findings.append(
            "Fused-slot body locality is uninterpretable because no probe exceeded "
            "the predeclared detectability thresholds with adequate coverage."
        )
    elif locality_passed:
        findings.append("Fused-slot body response meets the temporal-locality criterion.")
    else:
        findings.append(
            "Fused-slot body locality is inconclusive: at least one of local mass, "
            "center error, positive temporal correlation, or epsilon-scale stability "
            "does not meet its predeclared criterion."
        )
    return findings


def _write_report(
    path: Path,
    *,
    stage: str,
    row_count: int,
    novel_rows: int,
    unique_novel: int,
    checks: Mapping[str, Any],
    passive_summary: Mapping[str, Any],
    causal_summary: Mapping[str, Any],
    permutation_tests: Mapping[str, Any],
    selection,
    findings: Sequence[str],
):
    lines = [
        "# CSL-Daily epoch-2 temporal-slot and sentence-memory audit",
        "",
        f"Stage: `{stage}`. Validation rows: **{row_count}**; novel rows: "
        f"**{novel_rows}**; unique novel texts: **{unique_novel}**.",
        "",
        "This is a validation-only diagnostic of the fixed duration-weight-0.05 "
        "Phase-A checkpoint. It is not a promoted model evaluation.",
        "",
        "## Integrity checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Retrieval-condition summary",
            "",
            "| condition | effective K | max share | null mass | slot JSD | "
            "part JSD | adjacent switches | adjacent coverage | part switches | "
            "part coverage | gate | residual norm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in MEMORY_CONDITIONS:
        values = [
            _interval_value(passive_summary, f"candidate_{condition}_{metric}")
            for metric in (
                "effective_k",
                "max_share",
                "null_mass",
                "slot_jsd",
                "part_jsd",
                "adjacent_switch",
                "adjacent_switch_coverage",
                "part_switch",
                "part_switch_coverage",
            )
        ]
        gate = _interval_value(passive_summary, f"gate_{condition}_mean")
        residual = _interval_value(
            passive_summary, f"sentence_delta_{condition}_mean_norm"
        )
        lines.append(
            f"| {condition} | "
            + " | ".join(f"{value:.5g}" for value in (*values, gate, residual))
            + " |"
        )
    lines.extend(
        [
            "",
            "Correct-vs-shuffled and correct-vs-motion-only attention/output "
            "deltas, plus learned-vs-analytic-prior and audit-only latent-quality "
            "correlations, are reported with cluster-bootstrap intervals in "
            "`summary.json`.",
        ]
    )
    lines.extend(["", "## Predeclared interpretation", ""])
    lines.extend(f"- {finding}" for finding in findings)
    text_null = permutation_tests["text_attention"]
    lines.extend(
        [
            "",
            "## Permutation controls",
            "",
            "- Novel-text, cluster-averaged text-attention slot-position "
            f"correlation: `{text_null['observed_mean_spearman']:.6g}`; "
            f"one-sided permutation p=`{text_null['one_sided_p']:.6g}`.",
            "- Partwise causal-locality permutation tests and Holm-adjusted "
            "p-values are recorded in `summary.json`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Causal selection",
            "",
            f"Selected {len(selection.indices)} unique novel texts; selection hash "
            f"`{selection.selection_hash}`.",
            "",
            "Planner and fused slots were perturbed with four deterministic unit-"
            "Rademacher directions at epsilon 0.10 and 0.05. Trajectories were "
            "queried on 128 normalized times and converted to partwise SMPL-X "
            "root-relative keypoint responses; hand responses retain wrist-path "
            "motion rather than normalizing each hand at its wrist.",
            "",
            "## Machine-readable results",
            "",
            "See `summary.json`, the CSV tables, `arrays/manifest.json`, and "
            "`representative_examples.json`. Confidence intervals use equal-weight "
            "normalized-text clusters. Locality p-values use the deterministic "
            "slot-label permutation null and Holm correction over body and hands.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    profile = _selected_profile(args)
    _validate_arguments(args, profile)
    cfg = load_config(args.config)
    _validate_config(cfg, profile)
    cfg.setdefault("data", {})["random_crop"] = False
    cfg.setdefault("text", {})["device"] = "cpu"
    cfg["device"] = "cuda"
    resolved_config_sha256 = _digest_json(cfg)
    checkpoint_sha256 = _sha256(args.checkpoint)
    checkpoint = _load_checkpoint(
        args.checkpoint, expected_epoch=profile.fixed_epoch
    )
    checkpoint_epoch = int(checkpoint["epoch"])
    locked_run_evidence: dict[str, Any] | None = None
    if profile.require_locked_full_run:
        locked_run_evidence = _validate_phase_a_prime_locked_run(
            checkpoint,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
        )
    validation_manifest_sha256 = _sha256(EXPECTED_VALIDATION_MANIFEST)
    if validation_manifest_sha256 != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise RuntimeError(
            "Canonical CSL-Daily validation manifest hash mismatch: "
            f"expected={EXPECTED_VALIDATION_MANIFEST_SHA256}, "
            f"actual={validation_manifest_sha256}"
        )
    validation_manifest_identity = {
        "path": str(EXPECTED_VALIDATION_MANIFEST),
        "sha256": validation_manifest_sha256,
    }
    set_seed(int(args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("The diagnostic requires a Slurm-allocated CUDA GPU")
    device = torch.device("cuda")
    stage_rows = 8 if args.stage == "smoke" else EXPECTED_VALIDATION_ROWS
    stage_causal_rows = 8 if args.stage == "smoke" else int(args.causal_size)
    bank_override = os.environ.get(
        str(cfg["sentence_memory"].get("local_bank_env", "SIGNTRAJ_SENTENCE_MEMORY_DIR"))
    )
    bank_dir = Path(bank_override or cfg["sentence_memory"]["bank_dir"])
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError(
            "Temporal-slot diagnostics require a Slurm allocation and a "
            "split-filtered node-local bank"
        )
    expected_local_bank = Path(f"/tmp/signtraj_sentence_memory_{slurm_job_id}")
    if bank_dir.resolve() != expected_local_bank:
        raise RuntimeError(
            "Validation-only diagnostic requires the exact job-local bank path "
            f"{expected_local_bank}; got {bank_dir.resolve()}"
        )
    if (bank_dir / "neighbors_test.npz").exists():
        raise RuntimeError(
            "Validation-only diagnostic refuses a bank directory containing "
            "neighbors_test.npz; stage only the validation neighbor table"
        )
    neighbor_paths = {
        split: bank_dir / f"neighbors_{split}.npz" for split in ("train", "val")
    }
    missing_neighbors = [
        str(path) for path in neighbor_paths.values() if not path.is_file()
    ]
    if missing_neighbors:
        raise FileNotFoundError(
            f"Required train/validation neighbor tables are missing: {missing_neighbors}"
        )
    bank_metadata = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    source_sha256 = _diagnostic_source_hashes()
    git_head = _git_head()
    expected_settings = {
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "perturb_batch_size": int(args.perturb_batch_size),
        "fk_batch_size": int(args.fk_batch_size),
        "num_workers": int(args.num_workers),
        "query_points": int(args.query_points),
        "causal_size": stage_causal_rows,
        "directions": int(args.directions),
        "epsilons": [float(args.epsilon), float(args.epsilon) / 2.0],
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutation_samples": int(args.permutation_samples),
        "duration_weight": EXPECTED_DURATION_WEIGHT,
        "score_temperature": EXPECTED_SCORE_TEMPERATURE,
        "k": EXPECTED_K,
        "motion_shuffle_epoch": checkpoint_epoch,
        "verify_hashes": bool(args.verify_hashes),
    }
    expected_ready_identity = {
        "profile": profile.name,
        "stage": args.stage,
        "split": "val",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "resolved_config_sha256": resolved_config_sha256,
        "bank_id": str(bank_metadata.get("bank_id")),
        "neighbor_table_sha256": {
            split: _sha256(path) for split, path in neighbor_paths.items()
        },
        "validation_manifest": validation_manifest_identity,
        "settings": expected_settings,
        "query_counts": {
            "passive_rows": stage_rows,
            "causal_rows": stage_causal_rows,
        },
        "source_sha256": source_sha256,
        "git_head": git_head,
        "locked_run_evidence": locked_run_evidence,
    }

    with _atomic_output(
        args.out_dir,
        resume=bool(args.resume),
        expected_identity=expected_ready_identity,
        passive_rows=stage_rows,
        causal_rows=stage_causal_rows,
    ) as workspace:
        if workspace is None:
            print(f"READY diagnostic already exists: {args.out_dir}")
            return args.out_dir
        building = workspace.root
        validate_checkpoint_contract(checkpoint, cfg, source=str(args.checkpoint))

        dataset = ContinuousSignDataset(
            cfg,
            split="val",
            limit=0,
            random_crop=False,
            require_fk_cache=False,
        )
        if dataset.split != "val" or len(dataset) != EXPECTED_VALIDATION_ROWS:
            raise RuntimeError(
                f"Expected canonical val with {EXPECTED_VALIDATION_ROWS} rows; "
                f"found split={dataset.split!r}, rows={len(dataset)}"
            )
        text_encoder = build_text_encoder(cfg, torch.device("cpu"))
        provider = build_sentence_memory_provider(cfg, text_encoder, dataset=dataset)
        provider.validate_query_dataset(dataset, require_neighbors=True)
        provider_epoch = set_sentence_memory_provider_epoch_from_checkpoint(
            provider, checkpoint
        )
        if provider_epoch != checkpoint_epoch:
            raise RuntimeError("Provider epoch differs from the pinned checkpoint")
        _validate_checkpoint_identity_without_test(
            checkpoint,
            provider,
            source=str(args.checkpoint),
            cfg=cfg,
            text_encoder_identity=text_encoder.checkpoint_identity(),
        )
        model = build_continuous_trajectory_field(
            cfg, text_dim=text_encoder.text_dim
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        model.requires_grad_(False)
        del checkpoint["model"]

        passive_indices = (
            _first_unique_novel_indices(dataset, provider, 8)
            if args.stage == "smoke"
            else list(range(len(dataset)))
        )
        causal_count = stage_causal_rows
        slots = int(model.hypernetwork.temporal_slot_count)
        hidden = int(model.hypernetwork.context_hidden_dim)
        text_layers = len(model.hypernetwork.text_planner.decoder.layers)
        text_heads = int(model.hypernetwork.text_planner.decoder.layers[0].multihead_attn.num_heads)
        sentence_layers = len(model.hypernetwork.sentence_memory_encoder.layers)
        max_motion_tokens = int(np.asarray(provider.item_motion_lengths).max())
        arrays_root = building / "arrays"
        arrays_marker = arrays_root / ".initialized.json"
        if arrays_root.exists() and not arrays_marker.is_file():
            if workspace.has_committed_work:
                raise RuntimeError(
                    "Committed progress exists but the reopenable array store is "
                    "not fully initialized"
                )
            # No cursor has advanced, so an interrupted one-time allocation has
            # no scientific data. Rebuild only this fixed partial arrays path.
            shutil.rmtree(arrays_root)
        if not arrays_root.exists() and workspace.has_committed_work:
            raise RuntimeError("Committed progress exists without its array store")
        reopen_arrays = arrays_root.exists()
        store = ArrayStore(
            building,
            rows=len(passive_indices),
            slots=slots,
            hidden=hidden,
            text_layers=text_layers,
            text_heads=text_heads,
            max_text_tokens=int(text_encoder.max_length),
            sentence_layers=sentence_layers,
            candidates=int(provider.k),
            max_motion_tokens=max_motion_tokens,
            causal_rows=causal_count,
            directions=int(args.directions),
            query_points=int(args.query_points),
            reopen=reopen_arrays,
        )
        if reopen_arrays:
            # Validate committed data before trusting either progress cursor.
            workspace.validate_array_checkpoints(store)
        if workspace.passive_next < len(passive_indices):
            codec_path = Path(cfg["sentence_memory"]["codec_checkpoint"])
            if not codec_path.is_absolute():
                codec_path = PROJECT_ROOT / codec_path
            codec = LatentMotionCodec(codec_path, device=device)
            if codec.rotation_rep != "rot6d" or int(codec.latent_dim) != int(
                provider.latent_dim
            ):
                raise RuntimeError(
                    "Active VAE codec is incompatible with the sentence bank"
                )
            latent_mean, latent_std = bank_latent_statistics(provider.motion)
            _passive_sweep(
                model=model,
                text_encoder=text_encoder,
                provider=provider,
                codec=codec,
                latent_mean=latent_mean,
                latent_std=latent_std,
                dataset=dataset,
                indices=passive_indices,
                cfg=cfg,
                device=device,
                checkpoint_epoch=checkpoint_epoch,
                args=args,
                store=store,
                workspace=workspace,
            )
            del codec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        passive = workspace.load_passive()
        checks = passive["checks"]
        check_limits = {
            "text_attention_replay_max_abs": 1e-6,
            "instrumented_prediction_max_abs": 1e-7,
            "all_null_vs_memory_off_max_abs": 1e-7,
            "duration_memory_on_off_max_abs": 1e-7,
        }
        for name, limit in check_limits.items():
            if float(checks[name]) > limit:
                raise RuntimeError(
                    f"Fail-closed integrity check {name}={checks[name]:.9g} > {limit:.9g}"
                )

        queries = passive["queries"]
        novel_mask = [bool(row["novel_text"]) for row in queries]
        if args.stage == "full":
            novel_rows = sum(novel_mask)
            exact_rows = len(queries) - novel_rows
            unique_novel = len(
                {row["normalized_text"] for row in queries if row["novel_text"]}
            )
            if (novel_rows, exact_rows, unique_novel) != (1_075, 2, 796):
                raise RuntimeError(
                    "Canonical validation novelty counts changed: "
                    f"novel={novel_rows}, exact={exact_rows}, unique_novel={unique_novel}"
                )
        selection = select_stratified_queries(
            [row["query_id"] for row in queries],
            [row["normalized_text"] for row in queries],
            [row["predicted_duration"] for row in queries],
            [row["retrieval_margin"] for row in queries],
            novel_mask,
            sample_count=causal_count,
            bins=4,
            seed=int(args.seed),
        )
        workspace.bind_selection(selection.selection_hash)
        if workspace.causal_next < causal_count:
            fk = build_fk(cfg, device)
            _causal_sweep(
                model=model,
                fk=fk,
                text_encoder=text_encoder,
                provider=provider,
                dataset=dataset,
                selection=selection,
                passive_queries=queries,
                cfg=cfg,
                device=device,
                args=args,
                store=store,
                workspace=workspace,
            )
        causal_rows, selection_rows = workspace.load_causal()
        # Detect storage corruption even when this attempt did not reopen: the
        # final manifest must never bless bytes that differ from a committed
        # passive-batch or causal-query checkpoint.
        workspace.validate_array_checkpoints(store)
        _artifact_checks(
            store, rows=len(passive_indices), causal_rows=causal_count
        )

        _write_csv(building / "query_metrics.csv", queries)
        _write_csv(building / "slot_metrics.csv", passive["slots"])
        _write_csv(building / "text_attention_metrics.csv", passive["text_attention"])
        _write_csv(building / "part_attention_metrics.csv", passive["part_attention"])
        _write_csv(building / "candidate_metrics.csv", passive["candidates"])
        _write_csv(building / "causal_metrics.csv", causal_rows)
        _write_csv(building / "causal_selection.csv", selection_rows)

        passive_fields = (
            "planner_off_diagonal_mean",
            "planner_off_diagonal_p95",
            "planner_off_diagonal_fraction_above_0p95",
            "planner_centered_effective_rank",
            "planner_temporal_variance_ratio",
            "fused_correct_off_diagonal_mean",
            "fused_correct_off_diagonal_p95",
            "fused_correct_off_diagonal_fraction_above_0p95",
            "fused_correct_centered_effective_rank",
            "fused_correct_temporal_variance_ratio",
            "text_spearman",
            "text_inversion_rate",
            "text_normalized_entropy",
            "text_adjacent_jsd",
            "candidate_effective_k",
            "candidate_max_share",
            "candidate_null_mass",
            "candidate_adjacent_switch_rate",
            "candidate_part_switch_rate",
            "candidate_adjacent_switch_coverage",
            "candidate_part_switch_coverage",
            "learned_analytic_prior_jsd",
            "learned_analytic_prior_spearman",
            "learned_candidate_quality_spearman",
            "analytic_prior_candidate_quality_spearman",
            "correct_shuffled_attention_jsd",
            "correct_motion_shuffle_attention_jsd",
            "correct_vs_off_rms",
            "correct_vs_shuffled_rms",
            "correct_vs_motion_shuffle_rms",
        ) + tuple(
            f"candidate_{condition}_{metric}"
            for condition in MEMORY_CONDITIONS
            for metric in (
                "effective_k",
                "max_share",
                "null_mass",
                "slot_jsd",
                "part_jsd",
                "adjacent_switch",
                "part_switch",
                "adjacent_switch_coverage",
                "part_switch_coverage",
            )
        ) + tuple(
            f"gate_{condition}_mean" for condition in MEMORY_CONDITIONS
        ) + tuple(
            f"sentence_delta_{condition}_mean_norm"
            for condition in MEMORY_CONDITIONS
        )
        passive_summary = _bootstrap_summary(
            queries,
            passive_fields,
            samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
        query_lookup = {str(row["query_id"]): row for row in queries}
        causal_summary = _causal_aggregate(
            causal_rows,
            query_lookup,
            epsilon=float(args.epsilon),
            samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
        permutation_tests = {
            "text_attention": _text_attention_permutation_test(
                store,
                queries,
                rows=len(passive_indices),
                batch_size=int(args.batch_size),
                permutations=int(args.permutation_samples),
                seed=int(args.seed),
            ),
            "causal_locality": _causal_permutation_tests(
                store,
                causal_count=causal_count,
                query_points=int(args.query_points),
                permutations=int(args.permutation_samples),
                seed=int(args.seed),
            ),
        }
        representatives = _write_plots(
            building,
            store,
            queries,
            causal_rows,
            rows=len(passive_indices),
            causal_count=causal_count,
        )
        findings = _interpretation(passive_summary, causal_summary)
        summary = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "profile": profile.name,
            "stage": args.stage,
            "validation_only": True,
            "diagnostic_only": True,
            "counts": {
                "validation_rows": len(queries),
                "novel_rows": sum(novel_mask),
                "exact_seen_rows": len(queries) - sum(novel_mask),
                "unique_novel_texts": len(
                    {row["normalized_text"] for row in queries if row["novel_text"]}
                ),
                "causal_queries": causal_count,
            },
            "checks": checks,
            "selection": selection,
            "passive": passive_summary,
            "causal": causal_summary,
            "permutation_tests": permutation_tests,
            "interpretation": findings,
            "representatives": representatives,
        }
        _write_json(building / "summary.json", summary)

        store.flush()
        array_manifest = store.manifest(verify_hashes=bool(args.verify_hashes))
        _write_json(building / "arrays/manifest.json", array_manifest)
        _write_report(
            building / "report.md",
            stage=args.stage,
            row_count=len(queries),
            novel_rows=sum(novel_mask),
            unique_novel=len(
                {row["normalized_text"] for row in queries if row["novel_text"]}
            ),
            checks=checks,
            passive_summary=passive_summary,
            causal_summary=causal_summary,
            permutation_tests=permutation_tests,
            selection=selection,
            findings=findings,
        )
        artifact_manifest_sha256 = _write_artifact_manifest(building)
        if _sha256(args.checkpoint) != checkpoint_sha256:
            raise RuntimeError("Selected checkpoint changed during the diagnostic")
        if profile.require_locked_full_run:
            final_locked_evidence = _validate_phase_a_prime_locked_run(
                checkpoint,
                checkpoint_path=args.checkpoint,
                config_path=args.config,
            )
            if final_locked_evidence != locked_run_evidence:
                raise RuntimeError("Locked full-run provenance changed during the audit")
        provenance = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "profile": profile.name,
            "stage": args.stage,
            "validation_only": True,
            "split": "val",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint_epoch,
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "resolved_config_sha256": resolved_config_sha256,
            "source_sha256": source_sha256,
            "git_head": git_head,
            "neighbor_table_sha256": expected_ready_identity[
                "neighbor_table_sha256"
            ],
            "validation_manifest": validation_manifest_identity,
            "locked_run_evidence": locked_run_evidence,
            "bank_identity": provider.identity,
            "git": _git_identity(),
            "runtime": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(device),
                "wandb_mode": os.environ.get("WANDB_MODE"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "settings": expected_settings,
            "query_counts": expected_ready_identity["query_counts"],
            "selection_hash": selection.selection_hash,
            "array_manifest_sha256": _sha256(building / "arrays/manifest.json"),
            "artifact_manifest_sha256": artifact_manifest_sha256,
        }
        _write_json(building / "provenance.json", provenance)
        required = (
            "summary.json",
            "provenance.json",
            "report.md",
            "query_metrics.csv",
            "causal_metrics.csv",
            "arrays/manifest.json",
            ARTIFACT_MANIFEST_FILE,
        )
        missing = [name for name in required if not (building / name).is_file()]
        if missing:
            raise RuntimeError(f"Diagnostic finalization is missing {missing}")
    print(f"Temporal-slot diagnostics complete: {args.out_dir}")
    return args.out_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
