from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import (
    ContinuousSignDataset,
    collate_continuous_sign,
)
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_eval_samples import (
    read_jsonl,
    rot6d_to_axis_and_smplx,
    save_eval_npz,
    select_manifest,
)
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_text_encoder,
    encode_batch_text,
    move_batch_to_device,
    prepare_motion,
)
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryBatch,
    broadcast_sentence_memory_motion_payload,
    joint_tuple_permute_sentence_memory_batch,
    motion_only_shuffle_sentence_memory_batch,
    replace_sentence_memory_motion_payload,
    sha256_file,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    _sentence_memory_field,
    build_sentence_memory_provider,
    centered_sentence_memory_enabled,
    checkpoint_contract,
    configured_evaluation_corruption,
    is_dual_mode,
    is_sentence_memory_model,
    retrieve_sentence_memory,
    resolve_sentence_memory_relevance_calibration,
    sentence_memory_enabled,
    sentence_memory_evaluation_corruption_kwargs,
    sentence_memory_forward_kwargs,
    sentence_memory_provider_required,
    sentence_memory_query_ids,
    set_sentence_memory_provider_epoch_from_checkpoint,
    validate_checkpoint_contract,
    validate_sentence_memory_architecture_identity,
    validate_sentence_memory_checkpoint_identity,
    validate_sentence_memory_evaluation_control_identity,
    validate_sentence_memory_selection_aggregation_identity,
    validate_sentence_memory_validation_corruption_map_identity,
)
from NIAF.retrieval_confidence_field.scripts.export_retrieval_adaptive_samples import (
    generation_batch,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    validate_train_only_retrieval_bank,
)


FACTORIZED_EXPECTED_QUERY_MANIFEST_ENV = (
    "SIGNTRAJ_EXPECTED_QUERY_MANIFEST_SHA256"
)
ISOLATED_DEVELOPMENT_MANIFEST_ENV = (
    "SIGNTRAJ_ISOLATED_DEVELOPMENT_MANIFEST_SHA256"
)
ISOLATED_DEVELOPMENT_ROWS_ENV = "SIGNTRAJ_ISOLATED_DEVELOPMENT_ROWS"
ISOLATED_DEVELOPMENT_ARTIFACT_ENV = (
    "SIGNTRAJ_ISOLATED_DEVELOPMENT_PARTITION_ARTIFACT_IDENTITY"
)

CENTERED_EXPORT_MODES = (
    "motion_shuffled_n0",
    "motion_shuffled_n1",
    "motion_shuffled_n2",
    "cross_query_motion",
    "full_replacement",
    "joint_tuple_permuted",
    "uniform_final_mass",
    "association_disabled",
    "broadcast_complete",
)


def _validated_sha256(value, *, label):
    value = str(value or "").lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def is_factorized_sentence_memory_config(cfg):
    return bool(
        is_sentence_memory_model(cfg)
        and sentence_memory_enabled(cfg)
        and str(
            cfg.get("sentence_memory", {}).get(
                "key_value_mode", "legacy_mixed_v1"
            )
        ).lower()
        == "factorized_metadata_motion_v1"
    )


def authorize_factorized_export_manifest(cfg, manifest_path):
    """Authorize one sealed dev or explicitly post-spend confirmation manifest."""

    if not is_factorized_sentence_memory_config(cfg):
        return None
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Factorized export query manifest does not exist: {manifest_path}"
        )
    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    ready_path = manifest_path.parent / "READY"
    if not ready_path.is_file() or ready_path.stat().st_size > 4096:
        raise RuntimeError(
            "Factorized export manifest is not inside a sealed partition"
        )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    expected_artifact_identity = _validated_sha256(
        partition_cfg.get("expected_partition_artifact_identity"),
        label=(
            "validation_text_partition.expected_partition_artifact_identity"
        ),
    )
    if ready != {
        "schema_name": "signtrajfield_validation_text_cluster_partition",
        "schema_version": 1,
        "artifact_identity": expected_artifact_identity,
    }:
        raise RuntimeError("Factorized export sealed partition identity changed")

    selected_sha256 = sha256_file(manifest_path)
    development_sha256 = _validated_sha256(
        partition_cfg.get("expected_development_manifest_sha256"),
        label=(
            "validation_text_partition.expected_development_manifest_sha256"
        ),
    )
    post_spend_sha256 = os.environ.get(
        FACTORIZED_EXPECTED_QUERY_MANIFEST_ENV
    )
    if selected_sha256 == development_sha256:
        if manifest_path.name != "manifest_development.jsonl":
            raise RuntimeError(
                "Pinned development manifest has an unexpected filename"
            )
        authority = "config_pinned_development_manifest_v1"
        expected_rows = int(
            partition_cfg.get("expected_development_rows", -1)
        )
    else:
        post_spend_sha256 = _validated_sha256(
            post_spend_sha256,
            label=FACTORIZED_EXPECTED_QUERY_MANIFEST_ENV,
        )
        if selected_sha256 != post_spend_sha256:
            raise RuntimeError(
                "Factorized export query manifest is neither the pinned "
                "development set nor the explicitly authorized post-spend set"
            )
        if manifest_path.name != "manifest_confirmation.jsonl":
            raise RuntimeError(
                "Post-spend factorized export requires manifest_confirmation.jsonl"
            )
        authority = "post_spend_environment_manifest_v1"
        expected_rows = int(
            partition_cfg.get("expected_confirmation_rows", -1)
        )
    if expected_rows < 1:
        raise RuntimeError("Factorized export has no pinned query row count")
    return {
        "authority": authority,
        "manifest_path": manifest_path,
        "manifest_sha256": selected_sha256,
        "expected_rows": expected_rows,
        "partition_artifact_identity": expected_artifact_identity,
    }


def authorize_isolated_development_export_manifest(manifest_path):
    """Authorize a pre-confirmation dev manifest for nonfactorized parity export."""

    configured = {
        "manifest_sha256": os.environ.get(ISOLATED_DEVELOPMENT_MANIFEST_ENV),
        "row_count": os.environ.get(ISOLATED_DEVELOPMENT_ROWS_ENV),
        "artifact_identity": os.environ.get(ISOLATED_DEVELOPMENT_ARTIFACT_ENV),
    }
    if not any(value is not None for value in configured.values()):
        return None
    if any(value is None for value in configured.values()):
        raise RuntimeError(
            "Isolated development export requires all three explicit "
            "manifest/count/partition environment controls"
        )
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.name != "manifest_development.jsonl":
        raise RuntimeError(
            "Isolated development export requires manifest_development.jsonl"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Isolated development manifest does not exist: {manifest_path}"
        )
    expected_manifest_sha256 = _validated_sha256(
        configured["manifest_sha256"], label=ISOLATED_DEVELOPMENT_MANIFEST_ENV
    )
    expected_artifact_identity = _validated_sha256(
        configured["artifact_identity"],
        label=ISOLATED_DEVELOPMENT_ARTIFACT_ENV,
    )
    try:
        expected_rows = int(configured["row_count"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{ISOLATED_DEVELOPMENT_ROWS_ENV} must be a positive integer"
        ) from error
    if expected_rows < 1 or str(expected_rows) != str(configured["row_count"]):
        raise ValueError(
            f"{ISOLATED_DEVELOPMENT_ROWS_ENV} must be a canonical positive integer"
        )
    ready_path = manifest_path.parent / "READY"
    if not ready_path.is_file() or ready_path.stat().st_size > 4096:
        raise RuntimeError(
            "Isolated development manifest is not inside a sealed partition"
        )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready != {
        "schema_name": "signtrajfield_validation_text_cluster_partition",
        "schema_version": 1,
        "artifact_identity": expected_artifact_identity,
    }:
        raise RuntimeError("Isolated development partition identity changed")
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_manifest_sha256:
        raise RuntimeError("Isolated development manifest identity changed")
    return {
        "authority": "explicit_isolated_development_environment_v1",
        "manifest_path": manifest_path,
        "manifest_sha256": actual_sha256,
        "expected_rows": expected_rows,
        "partition_artifact_identity": expected_artifact_identity,
    }


def prepare_export_manifest(
    cfg,
    *,
    split,
    out_dir,
    num_samples,
    seed,
    manifest,
    selection_mode,
):
    """Select export rows without inspecting canonical val for factorized subsets."""

    factorized = is_factorized_sentence_memory_config(cfg)
    isolated_development = None
    isolated_controls_present = any(
        os.environ.get(name) is not None
        for name in (
            ISOLATED_DEVELOPMENT_MANIFEST_ENV,
            ISOLATED_DEVELOPMENT_ROWS_ENV,
            ISOLATED_DEVELOPMENT_ARTIFACT_ENV,
        )
    )
    if not factorized and isolated_controls_present:
        if manifest is None:
            raise RuntimeError(
                "Isolated development export requires an explicit manifest"
            )
        isolated_development = authorize_isolated_development_export_manifest(
            manifest
        )
    if factorized or isolated_development is not None:
        if str(split) != "val":
            raise RuntimeError("Isolated experiment exports are validation-only")
        if manifest is None:
            raise RuntimeError(
                "Isolated export requires an explicit sealed manifest"
            )
        authorization = (
            authorize_factorized_export_manifest(cfg, manifest)
            if factorized
            else isolated_development
        )
        selected_manifest, selected_rows, summary = select_manifest(
            cfg,
            split,
            out_dir,
            num_samples,
            seed,
            manifest=manifest,
            selection_mode=selection_mode,
        )
        if len(selected_rows) != int(authorization["expected_rows"]):
            raise RuntimeError(
                "Isolated export query row count differs from its "
                f"authorization: actual={len(selected_rows)}, "
                f"expected={authorization['expected_rows']}"
            )
        # The dataset reads the sealed source directly. Re-serializing rows to
        # the output copy would change the byte-level manifest identity.
        dataset_manifest = Path(manifest).resolve()
        selected_manifest_copy = Path(selected_manifest).resolve()
        summary.update(
            {
                "canonical_source_manifest": None,
                "canonical_sample_count": None,
                "is_complete_canonical_manifest": None,
                "is_canonical_order": None,
                "canonical_manifest_inspection": (
                    "forbidden_isolated_explicit_manifest_v1"
                ),
                "selected_manifest_copy": str(selected_manifest_copy),
                "selected_manifest_copy_sha256": sha256_file(
                    selected_manifest_copy
                ),
                "output_manifest": str(dataset_manifest),
                "dataset_manifest": str(dataset_manifest),
                "dataset_manifest_sha256": authorization["manifest_sha256"],
                "query_manifest_authority": authorization["authority"],
                "sealed_partition_artifact_identity": authorization[
                    "partition_artifact_identity"
                ],
            }
        )
        # Keep the standalone selection summary aligned with the manifest
        # evidence embedded in export_summary.json.
        (Path(out_dir) / "sample_manifest_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return dataset_manifest, selected_rows, summary

    configured_manifest = cfg.get("data", {}).get(f"{split}_manifest_path")
    canonical_manifest = (
        Path(configured_manifest)
        if configured_manifest
        else Path(cfg["data"]["data_dir"])
        / "meta"
        / f"manifest_{split}.jsonl"
    )
    canonical_rows = read_jsonl(canonical_manifest)
    selected_manifest, selected_rows, summary = select_manifest(
        cfg,
        split,
        out_dir,
        num_samples,
        seed,
        manifest=manifest,
        selection_mode=selection_mode,
    )
    complete_canonical_manifest = (
        len(selected_rows) == len(canonical_rows)
        and sorted(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in selected_rows
        )
        == sorted(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in canonical_rows
        )
    )
    canonical_order = selected_rows == canonical_rows
    summary.update(
        {
            "canonical_source_manifest": str(canonical_manifest),
            "canonical_sample_count": len(canonical_rows),
            "is_complete_canonical_manifest": complete_canonical_manifest,
            "is_canonical_order": canonical_order,
        }
    )
    if manifest is not None:
        # Explicit manifests are already the query authority. Preserve their
        # byte identity instead of routing the dataset and provenance through
        # select_manifest's semantically equivalent JSONL reserialization.
        dataset_manifest = Path(manifest).resolve()
        selected_manifest_copy = Path(selected_manifest).resolve()
        summary.update(
            {
                "selected_manifest_copy": str(selected_manifest_copy),
                "selected_manifest_copy_sha256": sha256_file(
                    selected_manifest_copy
                ),
                "output_manifest": str(dataset_manifest),
                "dataset_manifest": str(dataset_manifest),
                "dataset_manifest_sha256": sha256_file(dataset_manifest),
            }
        )
        (Path(out_dir) / "sample_manifest_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        dataset_manifest = (
            canonical_manifest if canonical_order else selected_manifest
        )
    return dataset_manifest, selected_rows, summary


def bind_factorized_export_neighbor_subset(cfg, dataset, provider):
    """Bind an authorized val subset to exact canonical neighbor-table rows."""

    if not is_factorized_sentence_memory_config(cfg):
        return None
    if provider is None:
        raise RuntimeError(
            "Factorized export requires a sentence-memory provider even for "
            "memory-off/all-null provenance"
        )
    if str(getattr(dataset, "split", "")) != "val":
        raise RuntimeError("Factorized experiment exports are validation-only")
    base = getattr(dataset, "base", dataset)
    manifest_path = Path(getattr(base, "manifest_path", ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Factorized export query manifest does not exist: {manifest_path}"
        )
    authorization = authorize_factorized_export_manifest(cfg, manifest_path)
    selected_sha256 = authorization["manifest_sha256"]
    partition_cfg = dict(cfg.get("validation_text_partition", {}) or {})
    authority = authorization["authority"]
    expected_rows = int(authorization["expected_rows"])
    if expected_rows < 1 or len(dataset) != expected_rows:
        raise RuntimeError(
            "Factorized export query row count differs from its authorization: "
            f"actual={len(dataset)}, expected={expected_rows}"
        )

    parent_manifest_sha256 = _validated_sha256(
        partition_cfg.get("expected_validation_manifest_sha256"),
        label="validation_text_partition.expected_validation_manifest_sha256",
    )
    expected_parent_rows = int(
        partition_cfg.get("expected_validation_rows", -1)
    )
    expected_bank_id = str(partition_cfg.get("expected_bank_id", ""))
    provider_identity = provider.identity
    if str(provider_identity.get("bank_id", "")) != expected_bank_id:
        raise RuntimeError("Factorized export sentence bank identity changed")
    full_table_identity = (
        provider_identity.get("neighbor_tables", {}).get("val")
    )
    if not isinstance(full_table_identity, dict):
        raise RuntimeError(
            "Factorized export requires the canonical validation neighbor table"
        )
    if str(full_table_identity.get("query_manifest_sha256", "")) != (
        parent_manifest_sha256
    ):
        raise RuntimeError(
            "Factorized export canonical neighbor manifest identity changed"
        )
    if int(full_table_identity.get("query_count", -1)) != expected_parent_rows:
        raise RuntimeError(
            "Factorized export canonical neighbor query count changed"
        )
    provider.set_dataset_with_name_indexed_neighbor_subset(
        dataset,
        parent_manifest_sha256=parent_manifest_sha256,
        expected_subset_manifest_sha256=selected_sha256,
    )
    subset = provider.neighbor_table
    if not isinstance(subset, dict) or subset.get("lookup_mode") != (
        "exact_name_indexed_parent_subset_v1"
    ):
        raise RuntimeError(
            "Factorized export did not bind an exact name-indexed neighbor subset"
        )
    if (
        subset.get("parent_query_manifest_sha256")
        != parent_manifest_sha256
        or int(subset.get("parent_query_count", -1)) != expected_parent_rows
    ):
        raise RuntimeError(
            "Factorized export neighbor subset lost its canonical parent identity"
        )
    payload = {
        "schema_name": "factorized_sentence_memory_export_query_binding",
        "schema_version": 1,
        "authority": authority,
        "partition_artifact_identity": authorization[
            "partition_artifact_identity"
        ],
        "query_manifest": {
            "file": manifest_path.name,
            "sha256": selected_sha256,
            "row_count": len(dataset),
        },
        "canonical_neighbor_table": {
            "sha256": full_table_identity.get("sha256"),
            "query_manifest_sha256": parent_manifest_sha256,
            "query_order_sha256": full_table_identity.get(
                "query_order_sha256"
            ),
            "query_count": expected_parent_rows,
        },
        "bank_id": expected_bank_id,
        "lookup_mode": subset["lookup_mode"],
        "online_fallback_allowed": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export arbitrarily sampled continuous trajectory instances."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of rows to export; 0 exports the complete split (default).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--selection_mode",
        default="random",
        choices=("random", "first"),
        help="Choose random manifest rows or the first rows used by limit_train.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--length_mode",
        default="predicted",
        choices=["predicted", "ground_truth_sampling", "ground_truth"],
        help=(
            "predicted uses predicted context and output lengths; "
            "ground_truth_sampling keeps the predicted context but queries the "
            "trajectory at the GT frame count; ground_truth also rebuilds the "
            "context at the GT length"
        ),
    )
    parser.add_argument(
        "--word_prior",
        default="auto",
        choices=("auto", "off", "on"),
        help=(
            "For v2, export text-only (off) or prior-enabled (on). auto uses "
            "text-only for v2 and the required scaffold path for v1."
        ),
    )
    parser.add_argument(
        "--sentence_memory",
        default="auto",
        choices=(
            "auto",
            "off",
            "on",
            "shuffled",
            "motion_shuffled",
            "motion_shuffled_n0",
            "motion_shuffled_n1",
            "motion_shuffled_n2",
            "cross_query_motion",
            "full_replacement",
            "joint_tuple_permuted",
            "uniform_final_mass",
            "analytic_prior",
            "association_disabled",
            "broadcast_complete",
            "all_null",
        ),
        help=(
            "For v3, export with memory off, retrieved memory on, or the "
            "configured deterministic corruption/ablation controls. The "
            "broadcast-complete and all-null modes are integrity controls. "
            "auto uses the configured export mode (on by default)."
        ),
    )
    parser.add_argument("--context_fps", type=float, default=20.0)
    parser.add_argument(
        "--sample_fps", type=float, nargs="+", default=[20.0, 40.0, 80.0]
    )
    return parser.parse_args()


def sampled_lengths(duration_seconds, fps, min_frames=2, max_frames=4000):
    return (
        torch.round(duration_seconds * float(fps))
        .long()
        .clamp(int(min_frames), int(max_frames))
    )


def resampled_frame_counts(
    reference_lengths,
    reference_fps,
    target_fps,
    min_frames=2,
    max_frames=4000,
):
    intervals = (reference_lengths.long() - 1).clamp_min(1)
    lengths = (
        torch.round(
            intervals.to(torch.float32) * float(target_fps) / float(reference_fps)
        ).long()
        + 1
    )
    return lengths.clamp(int(min_frames), int(max_frames))


def padded_normalized_grid(lengths, device, dtype):
    max_length = int(lengths.max().item())
    mask = torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    denominator = (lengths - 1).clamp_min(1).to(dtype).unsqueeze(1)
    coordinate = torch.arange(max_length, device=device, dtype=dtype).unsqueeze(0)
    tau = -1.0 + 2.0 * coordinate / denominator
    tau = torch.where(lengths.unsqueeze(1) > 1, tau, torch.zeros_like(tau))
    return tau.clamp(-1.0, 1.0), mask


def _fps_key(fps):
    value = float(fps)
    return (
        f"fps{int(value)}" if value.is_integer() else f"fps{value:g}".replace(".", "p")
    )


def _trajectory_numpy(instance, index):
    output = {}
    for key, value in instance.select(index).detach().tensor_dict().items():
        output[key] = value.squeeze(0).cpu().numpy()
    return output


def _memory_row_numpy(memory_batch, field, index, *aliases):
    value = _sentence_memory_field(memory_batch, field, *aliases)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[index].detach().cpu().numpy()
    row = value[index]
    if torch.is_tensor(row):
        return row.detach().cpu().numpy()
    return np.asarray(row)


def _sentence_memory_motion_payload_digests(memory_batch, index):
    """Hash each complete candidate motion package for causal-control audits."""

    fields = ("tokens", "token_mask", "token_tau", "part_validity")
    values = {
        name: np.asarray(_memory_row_numpy(memory_batch, name, index))
        for name in fields
    }
    candidate_count = int(values["tokens"].shape[0])
    if candidate_count < 1 or any(
        value.ndim < 1 or int(value.shape[0]) != candidate_count
        for value in values.values()
    ):
        raise ValueError("Malformed centered candidate motion package")
    digests = []
    for candidate in range(candidate_count):
        hasher = hashlib.sha256()
        for name in fields:
            value = np.ascontiguousarray(values[name][candidate])
            hasher.update(name.encode("utf-8"))
            hasher.update(value.dtype.str.encode("ascii"))
            hasher.update(json.dumps(value.shape).encode("ascii"))
            hasher.update(value.tobytes(order="C"))
        digests.append(hasher.hexdigest())
    return np.asarray(digests, dtype="<U64")


def sentence_memory_diagnostics_row(instance, index):
    """Return compact per-sample gate/null summaries for paired reports."""

    gates = getattr(instance, "sentence_memory_gates", None)
    null_mass = getattr(instance, "sentence_memory_null_mass", None)
    if gates is None or null_mass is None:
        return None
    row_gates = gates[int(index)].detach().float()
    row_null = null_mass[int(index)].detach().float()
    if row_gates.ndim != 2 or row_gates.shape[-1] != 4 or row_null.ndim != 1:
        raise ValueError("Malformed sentence-memory trajectory diagnostics")
    part_names = ("body", "lhand", "rhand", "face")
    available = getattr(instance, "sentence_memory_available", None)
    candidate_mass = getattr(instance, "sentence_memory_candidate_mass", None)
    candidate_mass_mean = None
    if candidate_mass is not None:
        row_candidate_mass = candidate_mass[int(index)].detach().float()
        if row_candidate_mass.ndim != 2:
            raise ValueError("Malformed sentence-memory candidate attention mass")
        candidate_mass_mean = float(row_candidate_mass.sum(dim=-1).mean().item())
    return {
        "available": bool(
            available is not None and available[int(index)].detach().bool().item()
        ),
        "gate_mean_by_part": {
            name: float(row_gates[:, part_index].mean().item())
            for part_index, name in enumerate(part_names)
        },
        "null_mass_mean": float(row_null.mean().item()),
        "candidate_mass_mean": candidate_mass_mean,
    }


class FactorizedAttentionCapture:
    """Retain attention and centered-evidence diagnostics for one export batch.

    Historical factorized exports keep their final-layer fields unchanged.  A
    centered configuration additionally exposes every layer plus the frozen
    relevance and optional absolute-association diagnostics.  The latter are
    read only after the complete encoder forward, so hooks never alter model
    inputs or outputs.
    """

    def __init__(self, model, cfg):
        self.value = None
        self.handle = None
        self.handles = []
        self.layer_values = {}
        self.encoder = None
        self.centered = False
        memory_cfg = dict(cfg.get("sentence_memory", {}) or {})
        if memory_cfg.get("key_value_mode") != "factorized_metadata_motion_v1":
            return
        self.encoder = model.hypernetwork.sentence_memory_encoder
        self.centered = (
            memory_cfg.get("candidate_value_mode")
            == "centered_candidate_covariance_v1"
        )
        if not self.encoder.layers:
            raise ValueError("Factorized sentence memory has no attention layer")
        self.slot_count = int(model.hypernetwork.temporal_slot_count)
        self.part_count = 4
        layer_indices = (
            range(len(self.encoder.layers))
            if self.centered
            else (len(self.encoder.layers) - 1,)
        )
        for layer_index in layer_indices:
            layer = self.encoder.layers[layer_index]
            handle = layer.register_forward_hook(
                self._layer_hook(layer_index)
            )
            self.handles.append(handle)
        # Preserve the public legacy attribute used by existing callers/tests.
        self.handle = self.handles[-1]

    def _layer_hook(self, layer_index):
        def hook(module, inputs, output):
            self._capture_layer(layer_index, module, inputs, output)

        return hook

    def _capture_layer(self, layer_index, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 4:
            raise ValueError("Malformed factorized attention-layer output")
        _state, null_mass, candidate_mass, token_mass = output
        batch = int(null_mass.shape[0])
        expected_queries = self.slot_count * self.part_count
        if null_mass.shape[1] != expected_queries:
            raise ValueError("Factorized attention query count changed")
        self.layer_values[int(layer_index)] = {
            "null_mass": null_mass.detach().reshape(
                batch, self.slot_count, self.part_count
            ).cpu(),
            "candidate_mass": candidate_mass.detach().reshape(
                batch,
                self.slot_count,
                self.part_count,
                candidate_mass.shape[-1],
            ).cpu(),
            "token_mass": token_mass.detach().reshape(
                batch,
                self.slot_count,
                self.part_count,
                token_mass.shape[-2],
                token_mass.shape[-1],
            ).cpu(),
        }
        self.value = self.layer_values[int(layer_index)]

    def _hook(self, _module, _inputs, output):
        # Backward-compatible direct hook used by existing unit tests.
        self._capture_layer(0, _module, _inputs, output)

    def clear(self):
        self.value = None
        self.layer_values = {}

    def row(self, index):
        if self.handle is None:
            return None
        if self.value is None:
            raise RuntimeError("Factorized attention hook did not observe a forward")
        result = {
            name: value[int(index)].numpy() for name, value in self.value.items()
        }
        if self.centered:
            if len(self.layer_values) != len(self.encoder.layers):
                raise RuntimeError("Centered export did not capture every memory layer")
            result.update(
                {
                    "layer_null_mass": torch.stack(
                        [self.layer_values[layer]["null_mass"] for layer in sorted(self.layer_values)],
                        dim=1,
                    )[int(index)].numpy(),
                    "layer_candidate_mass": torch.stack(
                        [self.layer_values[layer]["candidate_mass"] for layer in sorted(self.layer_values)],
                        dim=1,
                    )[int(index)].numpy(),
                    "layer_token_mass": torch.stack(
                        [self.layer_values[layer]["token_mass"] for layer in sorted(self.layer_values)],
                        dim=1,
                    )[int(index)].numpy(),
                }
            )
            debug = dict(getattr(self.encoder, "_last_debug", {}) or {})
            for name, value in debug.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > int(index):
                    result[name] = value[int(index)].detach().cpu().numpy()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.handle = None


def sentence_memory_text_subset(memory_batch, provider, index, text):
    """Classify a query only when the train-bank vocabulary is available."""

    provenance = (
        getattr(memory_batch, "provenance", {}) if memory_batch is not None else {}
    )
    seen_rows = (provenance or {}).get("query_seen_text", [])
    if index < len(seen_rows):
        return "exact_seen_text" if bool(seen_rows[index]) else "novel_text"
    if provider is not None:
        return "exact_seen_text" if provider.is_seen_text(text) else "novel_text"
    return "not_available"


def all_null_sentence_memory_batch(batch, cfg, device):
    """Create an enabled, candidate-free memory input without payload reads."""

    batch_size = len(batch.get("name", ()))
    candidate_count = int(cfg.get("sentence_memory", {}).get("k", 8))
    motion_dim = int(cfg.get("sentence_memory", {}).get("motion_dim", 256))
    key_dim = int(cfg.get("sentence_memory", {}).get("key_dim", 768))
    if batch_size < 1 or candidate_count < 1 or motion_dim < 1 or key_dim < 1:
        raise ValueError("Invalid all-null sentence-memory shape")
    return SentenceMemoryBatch(
        tokens=torch.zeros(batch_size, candidate_count, 1, motion_dim, device=device),
        token_mask=torch.zeros(
            batch_size, candidate_count, 1, dtype=torch.bool, device=device
        ),
        token_tau=torch.zeros(batch_size, candidate_count, 1, device=device),
        candidate_mask=torch.zeros(
            batch_size, candidate_count, dtype=torch.bool, device=device
        ),
        candidate_keys=torch.zeros(
            batch_size, candidate_count, key_dim, device=device
        ),
        scores=torch.zeros(batch_size, candidate_count, device=device),
        durations=torch.zeros(batch_size, candidate_count, device=device),
        duration_log_gap=torch.zeros(batch_size, candidate_count, device=device),
        part_validity=torch.zeros(
            batch_size, candidate_count, 1, 4, device=device
        ),
        ids=torch.full(
            (batch_size, candidate_count), -1, dtype=torch.int64, device=device
        ),
        group_ids=torch.full(
            (batch_size, candidate_count), -1, dtype=torch.int64, device=device
        ),
        source_group_ids=torch.full(
            (batch_size, candidate_count), -1, dtype=torch.int64, device=device
        ),
        available=torch.ones(batch_size, dtype=torch.bool, device=device),
        provenance={
            "mode": "all_null",
            "payload_reads": 0,
            "query_seen_text": [False] * batch_size,
            "candidate_group_ids": [[-1] * candidate_count for _ in range(batch_size)],
            "candidate_exact_text": [
                [False] * candidate_count for _ in range(batch_size)
            ],
        },
    )


@torch.no_grad()
def prepare_inference_batch(
    model,
    text_encoder,
    provider,
    batch,
    dataset,
    cfg,
    device,
    context_fps,
    length_mode,
    word_prior_mode="auto",
    sentence_memory_provider=None,
    sentence_memory_mode="auto",
    sentence_memory_epoch=0,
):
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    predicted_log_duration, predicted_duration = model.predict_duration(
        text_tokens, text_mask=text_mask
    )
    duration_cfg = cfg.get("duration", {})
    min_frames = int(
        duration_cfg.get("min_frames", cfg.get("data", {}).get("min_frames", 40))
    )
    max_frames = int(
        duration_cfg.get("max_frames", cfg.get("data", {}).get("max_frames", 400))
    )
    multiple = int(
        duration_cfg.get(
            "length_multiple", cfg.get("data", {}).get("length_multiple", 4)
        )
    )
    predicted_context_lengths = model.predict_lengths(
        text_tokens,
        text_mask=text_mask,
        fps=float(context_fps),
        min_frames=min_frames,
        max_frames=max_frames,
        multiple=multiple,
    )
    context_lengths = (
        batch["length"] if length_mode == "ground_truth" else predicted_context_lengths
    )
    dual_mode = is_dual_mode(cfg)
    sentence_memory_model = is_sentence_memory_model(cfg)
    if not dual_mode and word_prior_mode != "auto":
        raise ValueError("word_prior_mode is available only for dual-mode v2")
    resolved_mode = (
        str(cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off"))
        if sentence_memory_model and word_prior_mode == "auto"
        else ("off" if dual_mode and word_prior_mode == "auto" else word_prior_mode)
    )
    if dual_mode and resolved_mode not in {"off", "on"}:
        raise ValueError("word_prior_mode must be 'off' or 'on'")
    if not sentence_memory_model and sentence_memory_mode != "auto":
        raise ValueError("sentence_memory_mode is available only for v3")
    resolved_sentence_mode = (
        "off"
        if sentence_memory_model and not sentence_memory_enabled(cfg)
        else (
            str(cfg.get("eval", {}).get("export_sentence_memory_mode", "on"))
            if sentence_memory_model and sentence_memory_mode == "auto"
            else sentence_memory_mode
        )
    )
    if sentence_memory_model and resolved_sentence_mode not in {
        "off",
        "on",
        "shuffled",
        "motion_shuffled",
        "analytic_prior",
        "all_null",
        *CENTERED_EXPORT_MODES,
    }:
        raise ValueError(
            "Unsupported v3 sentence_memory_mode "
            f"{resolved_sentence_mode!r}"
        )
    if (
        resolved_sentence_mode in CENTERED_EXPORT_MODES
        and not centered_sentence_memory_enabled(cfg)
    ):
        raise ValueError(
            f"sentence_memory_mode={resolved_sentence_mode!r} requires the "
            "centered-candidate architecture"
        )

    adapter_context = None
    retrieval_features = None
    context_mask = None
    word_kwargs = {}
    if dual_mode and resolved_mode == "off":
        context_lengths = torch.zeros_like(batch["length"])
        word_kwargs = {
            "word_prior_available": torch.zeros(
                text_tokens.shape[0], dtype=torch.bool, device=device
            )
        }
    else:
        if provider is None:
            raise ValueError("Prior-enabled inference requires a ScaffoldProvider")
        generated = generation_batch(batch, context_lengths, device)
        adapter_context, _anchors, metadata = provider.build_with_metadata(
            generated,
            x=None,
            use_cache=False,
        )
        retrieval_features = metadata["retrieval_features"]
        context_mask = generated["mask"]
        if dual_mode:
            word_kwargs = {
                "word_prior_context": adapter_context,
                "word_prior_mask": context_mask,
                "word_prior_features": retrieval_features,
                "word_prior_available": torch.ones(
                    text_tokens.shape[0], dtype=torch.bool, device=device
                ),
            }
        else:
            trajectory = model.encode_trajectory(
                text_tokens=text_tokens,
                adapter_context=adapter_context,
                context_mask=context_mask,
                retrieval_evidence=retrieval_features,
                text_mask=text_mask,
            )
    sentence_memory_batch = None
    sentence_kwargs = {}
    evaluation_corruption = configured_evaluation_corruption(cfg)
    if sentence_memory_model:
        sentence_available = torch.full(
            (text_tokens.shape[0],),
            resolved_sentence_mode != "off",
            dtype=torch.bool,
            device=device,
        )
        sentence_kwargs = {"sentence_memory_available": sentence_available}
        if bool(sentence_available.any()):
            if resolved_sentence_mode == "all_null":
                sentence_memory_batch = all_null_sentence_memory_batch(
                    batch, cfg, device
                )
            else:
                if sentence_memory_provider is None:
                    raise ValueError(
                        "Retrieval-backed inference requires a "
                        "SentenceMemoryProvider"
                    )
                retrieval_mode = (
                    "shuffled"
                    if resolved_sentence_mode == "full_replacement"
                    else "on"
                    if resolved_sentence_mode
                    in {
                        "motion_shuffled",
                        "motion_shuffled_n0",
                        "motion_shuffled_n1",
                        "motion_shuffled_n2",
                        "cross_query_motion",
                        "joint_tuple_permuted",
                        "uniform_final_mass",
                        "analytic_prior",
                        "association_disabled",
                        "broadcast_complete",
                    }
                    else resolved_sentence_mode
                )
                sentence_memory_batch = retrieve_sentence_memory(
                    sentence_memory_provider,
                    dataset=dataset,
                    batch=batch,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    predicted_duration=predicted_duration.detach(),
                    available=sentence_available,
                    training=False,
                    device=device,
                    mode=retrieval_mode,
                    **sentence_memory_evaluation_corruption_kwargs(
                        cfg,
                        training=False,
                        condition=resolved_sentence_mode,
                    ),
                )
            if resolved_sentence_mode in {
                "motion_shuffled",
                "motion_shuffled_n0",
                "motion_shuffled_n1",
                "motion_shuffled_n2",
            }:
                corruption_kwargs = sentence_memory_evaluation_corruption_kwargs(
                    cfg, training=False, condition=resolved_sentence_mode
                )
                sentence_memory_batch, _permutation, _informative = (
                    motion_only_shuffle_sentence_memory_batch(
                        sentence_memory_batch,
                        query_ids=sentence_memory_query_ids(batch),
                        epoch=int(sentence_memory_epoch),
                        seed=int(
                            corruption_kwargs.get(
                                "corruption_seed", cfg.get("seed", 1234)
                            )
                        ),
                        corruption_nonce=corruption_kwargs.get(
                            "corruption_nonce"
                        ),
                        corruption_condition=str(
                            corruption_kwargs.get(
                                "corruption_condition", "motion_shuffled"
                            )
                        ),
                    )
                )
            if resolved_sentence_mode == "cross_query_motion":
                corruption_kwargs = sentence_memory_evaluation_corruption_kwargs(
                    cfg, training=False, condition=resolved_sentence_mode
                )
                source_batch = retrieve_sentence_memory(
                    sentence_memory_provider,
                    dataset=dataset,
                    batch=batch,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    predicted_duration=predicted_duration.detach(),
                    available=sentence_available,
                    training=False,
                    device=device,
                    mode="shuffled",
                    **corruption_kwargs,
                )
                sentence_memory_batch, _informative = (
                    replace_sentence_memory_motion_payload(
                        sentence_memory_batch,
                        source_batch,
                        corruption_nonce=corruption_kwargs["corruption_nonce"],
                        corruption_condition=resolved_sentence_mode,
                    )
                )
            if resolved_sentence_mode == "joint_tuple_permuted":
                corruption_kwargs = sentence_memory_evaluation_corruption_kwargs(
                    cfg, training=False, condition=resolved_sentence_mode
                )
                sentence_memory_batch, _permutation, _informative = (
                    joint_tuple_permute_sentence_memory_batch(
                        sentence_memory_batch,
                        query_ids=sentence_memory_query_ids(batch),
                        seed=int(corruption_kwargs["corruption_seed"]),
                        corruption_nonce=corruption_kwargs["corruption_nonce"],
                        corruption_condition=resolved_sentence_mode,
                    )
                )
            if resolved_sentence_mode == "broadcast_complete":
                sentence_memory_batch, _source_rank, _informative = (
                    broadcast_sentence_memory_motion_payload(
                        sentence_memory_batch,
                        query_ids=sentence_memory_query_ids(batch),
                        seed=int(cfg.get("seed", 1234)),
                    )
                )
            sentence_kwargs = sentence_memory_forward_kwargs(sentence_memory_batch)
            attention_modes = {
                "analytic_prior": "analytic_prior",
                "uniform_final_mass": "uniform_final_candidate_mass",
                "association_disabled": "association_disabled",
            }
            if resolved_sentence_mode in attention_modes:
                sentence_kwargs["sentence_memory_attention_mode"] = attention_modes[
                    resolved_sentence_mode
                ]
    if dual_mode:
        trajectory = model.encode_trajectory(
            text_tokens=text_tokens,
            text_mask=text_mask,
            **word_kwargs,
            **sentence_kwargs,
        )
    output_duration = (
        trajectory.duration_seconds if length_mode == "predicted" else batch["duration"]
    )
    return {
        "trajectory": trajectory,
        "adapter_context": adapter_context,
        "retrieval_features": retrieval_features,
        "context_mask": context_mask,
        "context_lengths": context_lengths,
        "predicted_context_lengths": predicted_context_lengths,
        "predicted_duration": predicted_duration,
        "predicted_log_duration": predicted_log_duration,
        "output_duration": output_duration,
        "word_prior_mode": resolved_mode if dual_mode else "v1_required",
        "sentence_memory_mode": (
            resolved_sentence_mode if sentence_memory_model else "not_applicable"
        ),
        "sentence_memory_batch": sentence_memory_batch,
        "sentence_memory_attention_mode": (
            {
                "analytic_prior": "analytic_prior",
                "uniform_final_mass": "uniform_final_candidate_mass",
                "association_disabled": "association_disabled",
            }.get(resolved_sentence_mode, "learned")
        ),
        "sentence_memory_motion_shuffle_epoch": (
            int(sentence_memory_epoch)
            if resolved_sentence_mode == "motion_shuffled"
            and evaluation_corruption["mode"]
            != "fixed_query_condition_v1"
            else None
        ),
        "sentence_memory_motion_shuffle_seed": (
            int(evaluation_corruption["seed"])
            if resolved_sentence_mode
            in {
                "motion_shuffled",
                "motion_shuffled_n0",
                "motion_shuffled_n1",
                "motion_shuffled_n2",
            }
            else None
        ),
        "sentence_memory_evaluation_corruption": evaluation_corruption,
    }


@torch.no_grad()
def sample_trajectory_fps(
    model,
    trajectory,
    durations,
    fps_values,
    reference_lengths=None,
    reference_fps=None,
):
    samples = {}
    for fps in fps_values:
        if reference_lengths is None:
            lengths = sampled_lengths(durations, fps)
        else:
            if reference_fps is None:
                raise ValueError("reference_fps is required with reference_lengths")
            lengths = resampled_frame_counts(reference_lengths, reference_fps, fps)
        tau, mask = padded_normalized_grid(
            lengths,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        outputs = model.query_trajectory(
            trajectory,
            tau,
            time_domain="normalized",
            query_mask=mask,
            return_details=True,
        )
        samples[float(fps)] = {
            "lengths": lengths,
            "mask": mask,
            "tau": tau,
            "outputs": outputs,
        }
    return samples


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if centered_sentence_memory_enabled(cfg):
        # Re-open and hash the sealed train-only calibration before model
        # construction.  Export never trusts coefficients copied into a
        # checkpoint or an unsealed config.
        resolve_sentence_memory_relevance_calibration(cfg)
    cfg.setdefault("scaffold", {})["cache_only"] = False
    cfg.setdefault("scaffold", {})["prefer_cache"] = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest, _selected_rows, manifest_summary = prepare_export_manifest(
        cfg,
        split=args.split,
        out_dir=out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        manifest=args.manifest,
        selection_mode=args.selection_mode,
    )
    cfg.setdefault("data", {})[f"{args.split}_manifest_path"] = str(dataset_manifest)
    cfg.setdefault("data", {})[f"limit_{args.split}"] = 0
    device = resolve_device(args.device)
    text_device = resolve_device(args.text_device)
    dataset = ContinuousSignDataset(
        cfg,
        split=args.split,
        limit=0,
        random_crop=False,
        require_fk_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_continuous_sign,
    )
    text_encoder = build_text_encoder(cfg, text_device)
    dual_mode = is_dual_mode(cfg)
    sentence_memory_model = is_sentence_memory_model(cfg)
    if not dual_mode and args.word_prior != "auto":
        raise ValueError("--word_prior is available only for dual-mode v2")
    resolved_word_prior_mode = (
        str(cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off"))
        if sentence_memory_model and args.word_prior == "auto"
        else ("off" if dual_mode and args.word_prior == "auto" else args.word_prior)
    )
    if not sentence_memory_model and args.sentence_memory != "auto":
        raise ValueError("--sentence_memory is available only for v3")
    resolved_sentence_memory_mode = (
        "off"
        if sentence_memory_model and not sentence_memory_enabled(cfg)
        else (
            str(cfg.get("eval", {}).get("export_sentence_memory_mode", "on"))
            if sentence_memory_model and args.sentence_memory == "auto"
            else args.sentence_memory
        )
    )
    needs_provider = not dual_mode or resolved_word_prior_mode == "on"
    provider = ScaffoldProvider(cfg, dataset, device) if needs_provider else None
    factorized_export = is_factorized_sentence_memory_config(cfg)
    sentence_memory_provider = (
        build_sentence_memory_provider(cfg, text_encoder, dataset=dataset)
        if sentence_memory_model
        and (
            factorized_export
            or sentence_memory_provider_required(resolved_sentence_memory_mode)
        )
        else None
    )
    sentence_memory_query_binding = None
    if sentence_memory_provider is not None:
        if factorized_export:
            sentence_memory_query_binding = (
                bind_factorized_export_neighbor_subset(
                    cfg, dataset, sentence_memory_provider
                )
            )
        else:
            sentence_memory_provider.validate_query_dataset(
                # Legacy exports retain their audited online fallback for
                # arbitrary debug subsets.
                dataset,
                require_neighbors=False,
            )
    retrieval_bank = (
        validate_train_only_retrieval_bank(cfg, provider)
        if provider is not None
        else None
    )
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(
        device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if centered_sentence_memory_enabled(cfg):
        parity = checkpoint.get("v2_to_v3_text_only_parity")
        if not isinstance(parity, dict) or not bool(parity.get("passed", False)):
            raise RuntimeError(
                "Centered sentence-memory export requires passing stored v2 parity"
            )
        cfg.setdefault("sentence_memory_safety", {})[
            "v2_to_v3_text_only_parity"
        ] = parity
    validate_checkpoint_contract(checkpoint, cfg, source=str(args.checkpoint))
    validate_sentence_memory_checkpoint_identity(
        checkpoint,
        sentence_memory_provider,
        source=str(args.checkpoint),
        cfg=cfg,
        text_encoder_identity=(
            text_encoder.checkpoint_identity()
            if hasattr(text_encoder, "checkpoint_identity")
            else None
        ),
    )
    if centered_sentence_memory_enabled(cfg):
        validate_sentence_memory_architecture_identity(
            checkpoint, cfg, source=str(args.checkpoint)
        )
        validate_sentence_memory_evaluation_control_identity(
            checkpoint, cfg, source=str(args.checkpoint)
        )
        validate_sentence_memory_selection_aggregation_identity(
            checkpoint, cfg, source=str(args.checkpoint)
        )
        if cfg.get("validation_text_partition", {}).get(
            "evaluation_corruption_map_identity"
        ) is not None:
            validate_sentence_memory_validation_corruption_map_identity(
                checkpoint, cfg, source=str(args.checkpoint)
            )
    checkpoint_epoch = set_sentence_memory_provider_epoch_from_checkpoint(
        sentence_memory_provider, checkpoint
    )
    model_type, contract_version = checkpoint_contract(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    factorized_attention = FactorizedAttentionCapture(model, cfg)

    fps_values = tuple(dict.fromkeys(float(value) for value in args.sample_fps))
    rows = []
    sample_counter = 0
    for batch in tqdm(loader, desc="export continuous trajectories"):
        factorized_attention.clear()
        batch = move_batch_to_device(batch, device)
        target = prepare_motion(batch, dataset, device)
        inference = prepare_inference_batch(
            model,
            text_encoder,
            provider,
            batch,
            dataset,
            cfg,
            device,
            context_fps=args.context_fps,
            length_mode=args.length_mode,
            word_prior_mode=resolved_word_prior_mode,
            sentence_memory_provider=sentence_memory_provider,
            sentence_memory_mode=resolved_sentence_memory_mode,
            sentence_memory_epoch=checkpoint_epoch,
        )
        sampled = sample_trajectory_fps(
            model,
            inference["trajectory"],
            inference["output_duration"],
            fps_values,
            reference_lengths=(
                batch["length"] if args.length_mode != "predicted" else None
            ),
            reference_fps=args.context_fps,
        )
        main_fps = fps_values[0]
        main_sample = sampled[main_fps]
        branch_samples = {}
        if bool(cfg.get("eval", {}).get("residual_branch_ablation", False)):
            branch_samples["global_only"] = model.query_trajectory(
                inference["trajectory"],
                main_sample["tau"],
                time_domain="normalized",
                query_mask=main_sample["mask"],
                include_local_residual=False,
            )
            branch_samples["local_only"] = model.query_trajectory(
                inference["trajectory"],
                main_sample["tau"],
                time_domain="normalized",
                query_mask=main_sample["mask"],
                include_global_residual=False,
            )

        for local_index in range(len(batch["name"])):
            suffix = f"{sample_counter:04d}"
            gt_length = int(batch["length"][local_index].item())
            output_length = int(main_sample["lengths"][local_index].item())
            context_length = int(inference["context_lengths"][local_index].item())
            meta = {
                "name": batch["name"][local_index],
                "text": batch["text"][local_index],
                "gloss": batch["gloss"][local_index],
                "split": args.split,
                "source_index": sample_counter,
            }
            gt_rot6d, gt_axis, gt_smplx = rot6d_to_axis_and_smplx(
                target[local_index, :gt_length]
            )
            prediction = main_sample["outputs"]["prediction"][
                local_index, :output_length
            ]
            coarse_name = "coarse" if dual_mode else "prior"
            coarse = main_sample["outputs"][coarse_name][local_index, :output_length]
            pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(prediction)
            coarse_rot6d, coarse_axis, coarse_smplx = rot6d_to_axis_and_smplx(coarse)
            gt_path = out_dir / f"gt_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_eval_npz(
                gt_path, gt_axis, gt_smplx, gt_rot6d, meta, label="ground_truth"
            )
            extra = {
                f"continuous_{coarse_name}_motion": coarse_axis.astype(np.float32),
                f"continuous_{coarse_name}_smplx": coarse_smplx.astype(np.float32),
                f"continuous_{coarse_name}_rot6d": coarse_rot6d.astype(np.float32),
                "checkpoint": np.asarray(str(args.checkpoint)),
                "checkpoint_epoch": np.asarray(
                    int(checkpoint.get("epoch", -1)), dtype=np.int32
                ),
                "model_type": np.asarray(model_type),
                "trajectory_contract_version": np.asarray(
                    contract_version, dtype=np.int32
                ),
                "word_prior_mode": np.asarray(inference["word_prior_mode"]),
                "sentence_memory_mode": np.asarray(inference["sentence_memory_mode"]),
                "sentence_memory_attention_mode": np.asarray(
                    inference["sentence_memory_attention_mode"]
                ),
                "length_mode": np.asarray(args.length_mode),
                "context_fps": np.asarray(float(args.context_fps), dtype=np.float32),
                "sample_fps": np.asarray(float(main_fps), dtype=np.float32),
                "ground_truth_length": np.asarray(gt_length, dtype=np.int32),
                "context_length": np.asarray(context_length, dtype=np.int32),
                "output_length": np.asarray(output_length, dtype=np.int32),
                "predicted_duration_seconds": np.asarray(
                    float(inference["trajectory"].duration_seconds[local_index].item()),
                    dtype=np.float32,
                ),
            }
            if inference["adapter_context"] is not None:
                context_rot6d, context_axis, context_smplx = rot6d_to_axis_and_smplx(
                    inference["adapter_context"][local_index, :context_length]
                )
                extra.update(
                    {
                        "adapter_context_motion": context_axis.astype(np.float32),
                        "adapter_context_smplx": context_smplx.astype(np.float32),
                        "adapter_context_rot6d": context_rot6d.astype(np.float32),
                        "retrieval_features": inference["retrieval_features"][
                            local_index, :context_length
                        ]
                        .cpu()
                        .float()
                        .numpy(),
                    }
                )
            memory_batch = inference["sentence_memory_batch"]
            if memory_batch is not None:
                for output_name, field_name, aliases in (
                    ("sentence_memory_ids", "ids", ()),
                    ("sentence_memory_scores", "scores", ()),
                    ("sentence_memory_durations", "durations", ()),
                    (
                        "sentence_memory_duration_log_gap",
                        "duration_log_gap",
                        ("duration_ratio",),
                    ),
                    ("sentence_memory_candidate_mask", "candidate_mask", ()),
                    ("sentence_memory_token_tau", "token_tau", ()),
                    ("sentence_memory_token_mask", "token_mask", ()),
                    ("sentence_memory_part_validity", "part_validity", ()),
                ):
                    value = _memory_row_numpy(
                        memory_batch,
                        field_name,
                        local_index,
                        *aliases,
                    )
                    if value is not None:
                        extra[output_name] = value
                if centered_sentence_memory_enabled(cfg):
                    for output_name, field_name in (
                        ("sentence_memory_group_ids", "group_ids"),
                        (
                            "sentence_memory_source_group_ids",
                            "source_group_ids",
                        ),
                    ):
                        value = _memory_row_numpy(
                            memory_batch, field_name, local_index
                        )
                        if value is not None:
                            extra[output_name] = value
                    extra["sentence_memory_motion_payload_digest"] = (
                        _sentence_memory_motion_payload_digests(
                            memory_batch, local_index
                        )
                    )
                memory_provenance = getattr(memory_batch, "provenance", {}) or {}
                candidate_group_ids = memory_provenance.get("candidate_group_ids", [])
                candidate_exact_text = memory_provenance.get("candidate_exact_text", [])
                query_seen_text = memory_provenance.get("query_seen_text", [])
                if local_index < len(candidate_group_ids):
                    extra["sentence_memory_group_ids"] = np.asarray(
                        candidate_group_ids[local_index], dtype=np.int64
                    )
                if local_index < len(candidate_exact_text):
                    extra["sentence_memory_exact_text"] = np.asarray(
                        candidate_exact_text[local_index], dtype=np.bool_
                    )
                if local_index < len(query_seen_text):
                    extra["sentence_memory_query_seen_text"] = np.asarray(
                        bool(query_seen_text[local_index]), dtype=np.bool_
                    )
                bank_id = memory_provenance.get("bank_id")
                if bank_id:
                    extra["sentence_memory_bank_id"] = np.asarray(str(bank_id))
                for provenance_name, output_name in (
                    (
                        "motion_candidate_permutation",
                        "sentence_memory_motion_candidate_permutation",
                    ),
                    ("motion_source_ids", "sentence_memory_motion_source_ids"),
                    (
                        "motion_only_shuffle_informative",
                        "sentence_memory_motion_shuffle_informative",
                    ),
                    (
                        "joint_tuple_candidate_permutation",
                        "sentence_memory_joint_tuple_candidate_permutation",
                    ),
                    (
                        "joint_tuple_informative",
                        "sentence_memory_joint_tuple_informative",
                    ),
                    (
                        "cross_query_motion_informative",
                        "sentence_memory_cross_query_motion_informative",
                    ),
                    (
                        "broadcast_motion_source_rank",
                        "sentence_memory_broadcast_motion_source_rank",
                    ),
                    (
                        "broadcast_motion_informative",
                        "sentence_memory_broadcast_motion_informative",
                    ),
                ):
                    provenance_rows = memory_provenance.get(provenance_name, [])
                    if local_index < len(provenance_rows):
                        extra[output_name] = np.asarray(
                            provenance_rows[local_index]
                        )
                for provenance_name, output_name in (
                    (
                        "motion_only_shuffle_epoch",
                        "sentence_memory_motion_shuffle_epoch",
                    ),
                    (
                        "motion_only_shuffle_seed",
                        "sentence_memory_motion_shuffle_seed",
                    ),
                    ("payload_reads", "sentence_memory_payload_reads"),
                ):
                    if provenance_name in memory_provenance:
                        provenance_value = memory_provenance[provenance_name]
                        if provenance_value is not None:
                            extra[output_name] = np.asarray(provenance_value)
                for provenance_name, output_name in (
                    (
                        "evaluation_corruption_mode",
                        "sentence_memory_evaluation_corruption_mode",
                    ),
                    (
                        "evaluation_corruption_nonce",
                        "sentence_memory_evaluation_corruption_nonce",
                    ),
                    (
                        "evaluation_corruption_condition",
                        "sentence_memory_evaluation_corruption_condition",
                    ),
                    (
                        "broadcast_motion_audit_nonce",
                        "sentence_memory_broadcast_motion_audit_nonce",
                    ),
                ):
                    provenance_value = memory_provenance.get(provenance_name)
                    if provenance_value is not None:
                        extra[output_name] = np.asarray(str(provenance_value))
            attention_row = (
                factorized_attention.row(local_index)
                if memory_batch is not None
                else None
            )
            if attention_row is not None:
                attention_fields = {
                    "null_mass": "sentence_memory_part_null_mass",
                    "candidate_mass": "sentence_memory_part_candidate_mass",
                    "token_mass": "sentence_memory_part_token_mass",
                    "layer_null_mass": "sentence_memory_layer_part_null_mass",
                    "layer_candidate_mass": (
                        "sentence_memory_layer_part_candidate_mass"
                    ),
                    "layer_token_mass": "sentence_memory_layer_part_token_mass",
                }
                for source_name, output_name in attention_fields.items():
                    if source_name in attention_row:
                        extra[output_name] = np.asarray(
                            attention_row[source_name], dtype=np.float32
                        )
                for source_name, value in attention_row.items():
                    if not source_name.startswith("sentence_memory_"):
                        continue
                    extra[source_name] = np.asarray(value, dtype=np.float32)
            extra.update(_trajectory_numpy(inference["trajectory"], local_index))
            for branch_name, branch_prediction in branch_samples.items():
                branch_rot6d, branch_axis, branch_smplx = rot6d_to_axis_and_smplx(
                    branch_prediction[local_index, :output_length]
                )
                extra[f"{branch_name}_rot6d"] = branch_rot6d.astype(np.float32)
                extra[f"{branch_name}_motion"] = branch_axis.astype(np.float32)
                extra[f"{branch_name}_smplx"] = branch_smplx.astype(np.float32)
            for fps, fps_sample in sampled.items():
                fps_length = int(fps_sample["lengths"][local_index].item())
                fps_prediction = fps_sample["outputs"]["prediction"][
                    local_index, :fps_length
                ]
                fps_rot6d, fps_axis, fps_smplx = rot6d_to_axis_and_smplx(fps_prediction)
                key = _fps_key(fps)
                extra[f"continuous_{key}_rot6d"] = fps_rot6d.astype(np.float32)
                extra[f"continuous_{key}_motion"] = fps_axis.astype(np.float32)
                extra[f"continuous_{key}_smplx"] = fps_smplx.astype(np.float32)
                extra[f"continuous_{key}_tau"] = (
                    fps_sample["tau"][local_index, :fps_length].cpu().float().numpy()
                )
            save_eval_npz(
                sample_path,
                pred_axis,
                pred_smplx,
                pred_rot6d,
                meta,
                label=(
                    "signtrajfield_rag_v3"
                    if sentence_memory_model
                    else (
                        "signtrajfield_v2"
                        if dual_mode
                        else "niaf_continuous_trajectory_field"
                    )
                ),
                extra=extra,
            )
            text_subset = sentence_memory_text_subset(
                inference["sentence_memory_batch"],
                sentence_memory_provider,
                local_index,
                meta["text"],
            )
            rows.append(
                {
                    "index": suffix,
                    "name": meta["name"],
                    "text": meta["text"],
                    "ground_truth_length": gt_length,
                    "ground_truth_duration_seconds": float(
                        batch["duration"][local_index].item()
                    ),
                    "context_length": context_length,
                    "word_prior_mode": inference["word_prior_mode"],
                    "sentence_memory_mode": inference["sentence_memory_mode"],
                    "sentence_memory_text_subset": (text_subset)
                    if sentence_memory_model
                    else "not_applicable",
                    "sentence_memory_diagnostics": (
                        sentence_memory_diagnostics_row(
                            inference["trajectory"], local_index
                        )
                        if sentence_memory_model
                        else None
                    ),
                    "predicted_duration_seconds": float(
                        inference["trajectory"].duration_seconds[local_index].item()
                    ),
                    "sample_lengths": {
                        _fps_key(fps): int(value["lengths"][local_index].item())
                        for fps, value in sampled.items()
                    },
                    "gt": str(gt_path),
                    "sample": str(sample_path),
                }
            )
            sample_counter += 1

    duration_errors = [
        abs(row["predicted_duration_seconds"] - row["ground_truth_duration_seconds"])
        for row in rows
    ]
    factorized_attention_artifacts = None
    if factorized_attention.handle is not None:
        factorized_attention_artifacts = {
            "schema": "final_layer_part_attention_mass_v1",
            "part_order": ["body", "lhand", "rhand", "face"],
            "slot_count": int(model.hypernetwork.temporal_slot_count),
            "layer": "final",
            "head_reduction": "mean",
            "null_mass_field": "sentence_memory_part_null_mass",
            "candidate_mass_field": "sentence_memory_part_candidate_mass",
            "token_mass_field": "sentence_memory_part_token_mass",
        }
        if centered_sentence_memory_enabled(cfg):
            factorized_attention_artifacts.update(
                {
                    "centered_layerwise_schema": "centered_evidence_layers_v1",
                    "layer_count": len(
                        model.hypernetwork.sentence_memory_encoder.layers
                    ),
                }
            )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "length_mode": args.length_mode,
        "model_type": model_type,
        "trajectory_contract_version": contract_version,
        "word_prior_mode": resolved_word_prior_mode if dual_mode else "v1_required",
        "sentence_memory_mode": (
            resolved_sentence_memory_mode if sentence_memory_model else "not_applicable"
        ),
        "sentence_memory_attention_mode": (
            {
                "analytic_prior": "analytic_prior",
                "uniform_final_mass": "uniform_final_candidate_mass",
                "association_disabled": "association_disabled",
            }.get(resolved_sentence_memory_mode, "learned")
            if sentence_memory_model
            else "not_applicable"
        ),
        "sentence_memory_motion_shuffle_epoch": (
            int(checkpoint_epoch)
            if sentence_memory_model
            and resolved_sentence_memory_mode == "motion_shuffled"
            and configured_evaluation_corruption(cfg)["mode"]
            != "fixed_query_condition_v1"
            else None
        ),
        "sentence_memory_motion_shuffle_seed": (
            int(configured_evaluation_corruption(cfg)["seed"])
            if sentence_memory_model
            and resolved_sentence_memory_mode
            in {
                "motion_shuffled",
                "motion_shuffled_n0",
                "motion_shuffled_n1",
                "motion_shuffled_n2",
            }
            else None
        ),
        "sentence_memory_evaluation_corruption": (
            configured_evaluation_corruption(cfg)
            if sentence_memory_model
            else None
        ),
        "sentence_memory": (
            getattr(sentence_memory_provider, "config_summary", None)
            if sentence_memory_provider is not None
            else (
                checkpoint.get("sentence_memory_identity")
                if sentence_memory_model
                else None
            )
        ),
        "sentence_memory_query_binding": sentence_memory_query_binding,
        "factorized_attention_artifacts": factorized_attention_artifacts,
        "context_fps": float(args.context_fps),
        "sample_fps": list(fps_values),
        "retrieval_bank": retrieval_bank,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "duration_mae_seconds": float(
            sum(duration_errors) / max(len(duration_errors), 1)
        ),
        "rows": rows,
    }
    if centered_sentence_memory_enabled(cfg):
        summary["sentence_memory_checkpoint_identities"] = {
            name: checkpoint.get(f"sentence_memory_{name}_identity")
            for name in (
                "architecture",
                "behavior",
                "objective",
                "resume",
                "evaluation_control",
                "selection_aggregation",
                "validation_corruption_map",
            )
        } | {
            "relevance_calibration": checkpoint.get(
                "sentence_memory_relevance_calibration_identity"
            )
        }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    factorized_attention.close()
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "checkpoint_epoch",
                    "split",
                    "length_mode",
                    "sample_fps",
                    "num_exported",
                    "duration_mae_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
