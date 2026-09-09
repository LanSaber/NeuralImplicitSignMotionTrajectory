"""Audit the exact one-update Stage-C checkpoints before publishing smoke READY."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from NIAF.continuous_trajectory_field.scripts.stage_c_atomic import (
    publish_bytes_no_replace,
)


SCHEMA_NAME = "signtrajfield_stage_c_one_update_checkpoint_audit"
SCHEMA_VERSION = 1
EXPECTED_SOURCE_SHA256 = (
    "b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202"
)
EXPECTED_SOURCE_EPOCH = 5
EXPECTED_SOURCE_GLOBAL_STEP = 360
EXPECTED_TRAINABLE_TENSORS = 20
EXPECTED_TRAINABLE_PARAMETERS = 1_109_395
EXPECTED_MODEL_TENSORS = 233
EXPECTED_FROZEN_MODEL_TENSORS = 213
EXPECTED_FROZEN_NONMEMORY_BASE_TENSORS = 151
EXPECTED_FROZEN_SENTENCE_MEMORY_TENSORS = 62
SENTENCE_MEMORY_PREFIX = "hypernetwork.sentence_memory_"
REQUIRED_CHECKPOINTS = {"last.pt", "epoch0001.pt"}
EXPLORATORY_SELECTION_CHECKPOINTS = {
    "best_exploratory.pt",
    "best_exploratory_infeasible.pt",
}


class SmokeCheckpointAuditError(RuntimeError):
    """A smoke checkpoint cannot support the Stage-C readiness claim."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise SmokeCheckpointAuditError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path


def _canonical_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _named_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or "digest" not in value:
        raise SmokeCheckpointAuditError(f"{label} is missing or malformed")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}
    if value.get("digest") != _canonical_digest(payload):
        raise SmokeCheckpointAuditError(f"{label} has an invalid digest")
    return value


def _require_finite_tensors(value: Any, *, label: str) -> int:
    import torch

    count = 0
    if torch.is_tensor(value):
        count += 1
        if not bool(torch.isfinite(value).all().item()):
            raise SmokeCheckpointAuditError(f"{label} contains a non-finite tensor")
    elif isinstance(value, dict):
        for key, item in value.items():
            count += _require_finite_tensors(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            count += _require_finite_tensors(item, label=f"{label}[{index}]")
    return count


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    _regular_file(path, "Stage-C checkpoint")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SmokeCheckpointAuditError(f"cannot load checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise SmokeCheckpointAuditError(f"checkpoint is not a mapping: {path}")
    return value


def _audit_optimizer(
    checkpoint: dict[str, Any],
    model: dict[str, Any],
    expected_mapping: dict[str, Any],
    expected_step: int,
) -> dict[str, Any]:
    import torch

    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise SmokeCheckpointAuditError("Stage-C optimizer serialization is not exact")
    states = optimizer["state"]
    groups = optimizer["param_groups"]
    expected_groups = expected_mapping["groups"]
    if not isinstance(states, dict) or not isinstance(groups, list) or len(groups) != 2:
        raise SmokeCheckpointAuditError("Stage-C optimizer lacks its two exact groups")

    parameter_ids: list[Any] = []
    parameter_names: list[str] = []
    nonzero_moment_names: list[str] = []
    for group_index, (expected, observed) in enumerate(
        zip(expected_groups, groups, strict=True)
    ):
        if (
            observed.get("group_name") != expected["group_name"]
            or expected["group_index"] != group_index
            or float(observed.get("lr", float("nan"))) != 1e-5
            or float(observed.get("weight_decay", float("nan"))) != 1e-4
            or tuple(observed.get("betas", ())) != (0.9, 0.999)
            or float(observed.get("eps", float("nan"))) != 1e-8
            or observed.get("amsgrad") is not False
        ):
            raise SmokeCheckpointAuditError(
                "Stage-C optimizer group identity/hyperparameters changed"
            )
        ids = observed.get("params")
        names = expected["parameter_names"]
        if not isinstance(ids, list) or len(ids) != len(names):
            raise SmokeCheckpointAuditError("Stage-C optimizer ID/name mapping changed")
        for parameter_id, name in zip(ids, names, strict=True):
            if name not in model or parameter_id not in states:
                raise SmokeCheckpointAuditError(
                    f"Stage-C optimizer does not represent trainable tensor {name}"
                )
            state = states[parameter_id]
            if not isinstance(state, dict) or set(state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise SmokeCheckpointAuditError(
                    f"Stage-C optimizer state is incomplete for {name}"
                )
            step = state["step"]
            first = state["exp_avg"]
            second = state["exp_avg_sq"]
            if (
                not torch.is_tensor(step)
                or step.numel() != 1
                or float(step.item()) != float(expected_step)
                or not torch.is_tensor(first)
                or not torch.is_tensor(second)
                or first.shape != model[name].shape
                or second.shape != model[name].shape
                or not bool(torch.isfinite(first).all().item())
                or not bool(torch.isfinite(second).all().item())
                or int(torch.count_nonzero(first).item()) == 0
                or int(torch.count_nonzero(second).item()) == 0
            ):
                raise SmokeCheckpointAuditError(
                    f"Stage-C tensor {name} lacks finite nonzero AdamW moments "
                    f"at step {expected_step}"
                )
            parameter_ids.append(parameter_id)
            parameter_names.append(name)
            nonzero_moment_names.append(name)
    if (
        len(parameter_ids) != EXPECTED_TRAINABLE_TENSORS
        or len(set(parameter_ids)) != EXPECTED_TRAINABLE_TENSORS
        or set(states) != set(parameter_ids)
        or len(parameter_names) != EXPECTED_TRAINABLE_TENSORS
        or len(set(parameter_names)) != EXPECTED_TRAINABLE_TENSORS
    ):
        raise SmokeCheckpointAuditError(
            "Stage-C optimizer contains missing, duplicate, or extra parameters"
        )
    return {
        "optimizer_state_tensor_count": _require_finite_tensors(
            optimizer, label="optimizer"
        ),
        "represented_trainable_tensor_count": len(parameter_names),
        "nonzero_finite_moment_tensor_count": len(nonzero_moment_names),
    }


def _audit_checkpoint(
    path: Path,
    *,
    arm: str,
    source: dict[str, Any],
    expected_names: tuple[str, ...],
    expected_trainability: dict[str, Any],
    expected_mapping: dict[str, Any],
    expected_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    checkpoint = _load_checkpoint(path)
    if checkpoint.get("epoch") != 1 or checkpoint.get("global_step") != expected_step:
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} did not produce epoch=1/global_step={expected_step}"
        )
    model = checkpoint.get("model")
    source_model = source.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(source_model, dict)
        or set(model) != set(source_model)
        or any(not torch.is_tensor(value) for value in model.values())
        or any(not torch.is_tensor(value) for value in source_model.values())
    ):
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} model state differs structurally from Stage B"
        )
    _require_finite_tensors(model, label=f"{arm}.{path.name}.model")

    approved = set(expected_names)
    frozen = set(model) - approved
    memory = {name for name in model if name.startswith(SENTENCE_MEMORY_PREFIX)}
    frozen_nonmemory = frozen - memory
    if (
        len(model) != EXPECTED_MODEL_TENSORS
        or len(approved) != EXPECTED_TRAINABLE_TENSORS
        or not approved.issubset(model)
        or len(frozen) != EXPECTED_FROZEN_MODEL_TENSORS
        or len(memory) != EXPECTED_FROZEN_SENTENCE_MEMORY_TENSORS
        or not memory.issubset(frozen)
        or len(frozen_nonmemory) != EXPECTED_FROZEN_NONMEMORY_BASE_TENSORS
    ):
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} does not have the exact "
            "20 trainable/213 frozen (151 base + 62 memory) tensor split"
        )
    changed_trainable = []
    for name in sorted(model):
        if name in frozen and not torch.equal(model[name], source_model[name]):
            raise SmokeCheckpointAuditError(
                f"{arm}/{path.name} changed frozen model tensor {name}"
            )
        if name in approved and not torch.equal(model[name], source_model[name]):
            changed_trainable.append(name)
    if len(changed_trainable) != EXPECTED_TRAINABLE_TENSORS:
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} changed {len(changed_trainable)}/"
            f"{EXPECTED_TRAINABLE_TENSORS} approved generator tensors"
        )
    trainable_parameters = sum(int(model[name].numel()) for name in approved)
    if trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} trainable parameter count changed"
        )

    if checkpoint.get("stage_c_trainability_contract") != expected_trainability:
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} trainability contract changed"
        )
    if checkpoint.get("stage_c_optimizer_parameter_mapping") != expected_mapping:
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} optimizer mapping contract changed"
        )
    provenance = _named_identity(
        checkpoint.get("stage_c_provenance"), label=f"{arm} provenance"
    )
    state_reset = provenance.get("state_reset")
    if state_reset != {
        "model_weights_only": True,
        "epoch": 1,
        "global_step": 0,
        "optimizer": "fresh",
        "selection": "fresh",
    }:
        raise SmokeCheckpointAuditError(f"{arm}/{path.name} lacks exact reset proof")
    source_spec = provenance.get("source_checkpoint") or {}
    if (
        source_spec.get("sha256") != EXPECTED_SOURCE_SHA256
        or source_spec.get("epoch") != EXPECTED_SOURCE_EPOCH
        or source_spec.get("global_step") != EXPECTED_SOURCE_GLOBAL_STEP
        or source_spec.get("selection_status") != "best_infeasible"
    ):
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} provenance does not bind the exact Stage-B source"
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
        raise SmokeCheckpointAuditError(
            f"{arm}/{path.name} authorization scope changed"
        )
    calibration = _named_identity(
        checkpoint.get("sentence_memory_relevance_calibration_identity"),
        label=f"{arm} calibration",
    )
    distribution = _named_identity(
        checkpoint.get("stage_c_distribution_contract"),
        label=f"{arm} distribution",
    )
    objective = _named_identity(
        checkpoint.get("sentence_memory_objective_identity"),
        label=f"{arm} objective",
    )
    if (
        objective.get("arm") != arm
        or objective.get("sentence_memory_train_mode")
        != {"memory": "dropout", "matched_off": "off"}[arm]
        or float(
            objective.get("sentence_memory_dropout_probability", float("nan"))
        )
        != {"memory": 0.25, "matched_off": 1.0}[arm]
    ):
        raise SmokeCheckpointAuditError(f"{arm}/{path.name} objective arm changed")
    optimizer_summary = _audit_optimizer(
        checkpoint, model, expected_mapping, expected_step
    )
    summary = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": 1,
        "global_step": expected_step,
        "model_tensor_count": len(model),
        "model_tensors_finite": True,
        "trainable_tensor_count": len(approved),
        "trainable_parameter_count": trainable_parameters,
        "changed_trainable_tensor_count": len(changed_trainable),
        "frozen_model_tensor_count": len(frozen),
        "frozen_model_tensors_bitwise_exact": True,
        "sentence_memory_tensor_count": len(memory),
        "sentence_memory_tensors_bitwise_exact": True,
        **optimizer_summary,
    }
    identities = {
        "provenance": provenance,
        "calibration": calibration,
        "distribution": distribution,
        "objective": objective,
        "architecture": _named_identity(
            checkpoint.get("sentence_memory_architecture_identity"),
            label=f"{arm} architecture",
        ),
        "memory": checkpoint.get("sentence_memory_identity"),
    }
    if not isinstance(identities["memory"], dict):
        raise SmokeCheckpointAuditError(f"{arm}/{path.name} memory identity is absent")
    return summary, identities


def _without_digest(value: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "digest"}


def _normalized_provenance(value: dict[str, Any]) -> dict[str, Any]:
    payload = _without_digest(value)
    declared = payload.get("declared_contract")
    if not isinstance(declared, dict):
        raise SmokeCheckpointAuditError("Stage-C provenance lacks declared contract")
    declared.pop("arm", None)
    return payload


def _normalized_objective(value: dict[str, Any]) -> dict[str, Any]:
    payload = _without_digest(value)
    payload.pop("arm", None)
    payload.pop("sentence_memory_train_mode", None)
    payload.pop("sentence_memory_dropout_probability", None)
    return payload


def audit_stage_c_checkpoints(
    *,
    source_checkpoint: Path,
    memory_dir: Path,
    matched_off_dir: Path,
    execution_mode: str,
) -> dict[str, Any]:
    from NIAF.continuous_trajectory_field.scripts import (
        train_continuous_trajectory_field as trainer,
    )

    if execution_mode not in {"smoke", "pilot"}:
        raise SmokeCheckpointAuditError("execution mode must be smoke or pilot")
    expected_step = 1 if execution_mode == "smoke" else 72
    if sha256_file(_regular_file(source_checkpoint, "Stage-B source checkpoint")) != (
        EXPECTED_SOURCE_SHA256
    ):
        raise SmokeCheckpointAuditError("Stage-B source checkpoint SHA256 changed")
    source = _load_checkpoint(source_checkpoint)
    if (
        source.get("epoch") != EXPECTED_SOURCE_EPOCH
        or source.get("global_step") != EXPECTED_SOURCE_GLOBAL_STEP
    ):
        raise SmokeCheckpointAuditError("Stage-B source epoch/global step changed")
    _require_finite_tensors(source.get("model"), label="source.model")
    expected_trainability = trainer.stage_c_trainability_contract()
    expected_mapping = trainer.stage_c_optimizer_parameter_mapping_contract()
    expected_names = tuple(expected_trainability["trainable_parameter_names"])

    arm_summaries = {}
    arm_identities = {}
    for arm, directory in (
        ("memory", memory_dir),
        ("matched_off", matched_off_dir),
    ):
        checkpoint_dir = directory / "checkpoints"
        if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
            raise SmokeCheckpointAuditError(
                f"{arm} has no regular checkpoint directory"
            )
        paths = sorted(checkpoint_dir.iterdir())
        if any(not path.is_file() or path.is_symlink() for path in paths):
            raise SmokeCheckpointAuditError(f"{arm} checkpoint directory is not regular")
        names = {path.name for path in paths}
        selected = names & EXPLORATORY_SELECTION_CHECKPOINTS
        if (
            not REQUIRED_CHECKPOINTS.issubset(names)
            or len(selected) != 1
            or names != REQUIRED_CHECKPOINTS | selected
        ):
            raise SmokeCheckpointAuditError(
                f"{arm} checkpoint names are not exact exploratory artifacts: {sorted(names)}"
            )
        summaries = {}
        identities = None
        for path in paths:
            summary, current_identities = _audit_checkpoint(
                path,
                arm=arm,
                source=source,
                expected_names=expected_names,
                expected_trainability=expected_trainability,
                expected_mapping=expected_mapping,
                expected_step=expected_step,
            )
            summaries[path.name] = summary
            if identities is None:
                identities = current_identities
            elif current_identities != identities:
                raise SmokeCheckpointAuditError(
                    f"{arm} checkpoint identity bindings differ within the smoke"
                )
        arm_summaries[arm] = summaries
        arm_identities[arm] = identities

    memory = arm_identities["memory"]
    matched = arm_identities["matched_off"]
    for name in ("calibration", "distribution", "architecture", "memory"):
        if memory[name] != matched[name]:
            raise SmokeCheckpointAuditError(
                f"Stage-C smoke arms bind different {name} identities"
            )
    if _normalized_provenance(memory["provenance"]) != _normalized_provenance(
        matched["provenance"]
    ):
        raise SmokeCheckpointAuditError(
            "Stage-C smoke arms bind different source provenance beyond arm"
        )
    if _normalized_objective(memory["objective"]) != _normalized_objective(
        matched["objective"]
    ):
        raise SmokeCheckpointAuditError(
            "Stage-C smoke objectives differ beyond training-memory availability"
        )
    return {
        "schema_name": (
            SCHEMA_NAME
            if execution_mode == "smoke"
            else "signtrajfield_stage_c_full_epoch_checkpoint_audit"
        ),
        "schema_version": SCHEMA_VERSION,
        "execution_mode": execution_mode,
        "expected_global_step": expected_step,
        "source_checkpoint": {
            "path": str(source_checkpoint.resolve()),
            "sha256": EXPECTED_SOURCE_SHA256,
            "epoch": EXPECTED_SOURCE_EPOCH,
            "global_step": EXPECTED_SOURCE_GLOBAL_STEP,
        },
        "arms": arm_summaries,
        "exact_trainable_tensor_count": EXPECTED_TRAINABLE_TENSORS,
        "exact_trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "exact_frozen_model_tensor_count": EXPECTED_FROZEN_MODEL_TENSORS,
        "exact_model_tensor_count": EXPECTED_MODEL_TENSORS,
        "exact_frozen_nonmemory_base_tensor_count": (
            EXPECTED_FROZEN_NONMEMORY_BASE_TENSORS
        ),
        "exact_frozen_sentence_memory_tensor_count": (
            EXPECTED_FROZEN_SENTENCE_MEMORY_TENSORS
        ),
        "exact_changed_trainable_tensor_count": EXPECTED_TRAINABLE_TENSORS,
        "all_model_and_optimizer_tensors_finite": True,
        "all_sentence_memory_and_nonapproved_tensors_bitwise_frozen": True,
        "all_trainables_have_nonzero_finite_one_step_moments": True,
        "all_trainables_have_nonzero_finite_optimizer_moments": True,
        "same_source_provenance_except_arm": True,
        "same_calibration_distribution_architecture_memory": True,
        "same_objective_except_training_memory_availability": True,
        "canonical_best_checkpoint_published": False,
        "development_only": True,
        "non_authorizing": True,
    }


def audit_smoke_checkpoints(
    *, source_checkpoint: Path, memory_dir: Path, matched_off_dir: Path
) -> dict[str, Any]:
    return audit_stage_c_checkpoints(
        source_checkpoint=source_checkpoint,
        memory_dir=memory_dir,
        matched_off_dir=matched_off_dir,
        execution_mode="smoke",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--memory_dir", type=Path, required=True)
    parser.add_argument("--matched_off_dir", type=Path, required=True)
    parser.add_argument(
        "--execution_mode", choices=("smoke", "pilot"), default="smoke"
    )
    parser.add_argument("--out_file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_stage_c_checkpoints(
        source_checkpoint=args.source_checkpoint,
        memory_dir=args.memory_dir,
        matched_off_dir=args.matched_off_dir,
        execution_mode=args.execution_mode,
    )
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    publish_bytes_no_replace(args.out_file, encoded)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
