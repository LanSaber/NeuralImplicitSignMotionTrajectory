from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


BANK_SCHEMA_NAME = "signtrajfield_sentence_motion_memory"
BANK_SCHEMA_VERSION = 1
NEIGHBOR_SCHEMA_NAME = "signtrajfield_sentence_neighbors"
NEIGHBOR_SCHEMA_VERSION = 1

REQUIRED_ARTIFACTS = (
    "group_keys",
    "group_offsets",
    "group_item_ids",
    "item_group_ids",
    "motion_mu",
    "offsets",
    "item_motion_lengths",
    "hand_valid",
    "durations",
    "metadata",
    "groups",
)

DEFAULT_ARTIFACT_FILES = {
    "group_keys": "group_keys.float32.npy",
    "group_offsets": "group_item_offsets.int64.npy",
    "group_item_ids": "group_item_ids.int32.npy",
    "item_group_ids": "item_group_ids.int32.npy",
    "motion_mu": "item_motion_mu.float16.npy",
    "offsets": "item_motion_offsets.int64.npy",
    "item_motion_lengths": "item_motion_lengths.int16.npy",
    "hand_valid": "item_hand_valid.uint8.npy",
    "durations": "item_durations.float32.npy",
    "metadata": "items.jsonl",
    "groups": "semantic_groups.jsonl",
}


class SentenceMemoryValidationError(RuntimeError):
    """Raised when a sentence-memory artifact violates its persisted contract."""


def _jsonable(value: Any):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sentence_memory_preprocessing_contract(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic motion preprocessing used by bank builders.

    Sentence banks deliberately never inherit a training-time random crop.  A
    bank identity is only reusable when the active trajectory data pipeline
    would produce the same fixed frame lengths and rotation representation.
    """

    data_cfg = dict(cfg.get("data", {}) or {})
    return {
        "rotation_rep": "rot6d",
        "min_frames": int(data_cfg.get("min_frames", 40)),
        "max_frames": int(data_cfg.get("max_frames", 400)),
        "length_multiple": int(data_cfg.get("length_multiple", 4)),
        "random_crop": False,
    }


def sentence_memory_motion_stats_paths(
    cfg: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Resolve the active rot6d normalization statistics without loading data."""

    data_cfg = dict(cfg.get("data", {}) or {})
    data_dir = data_cfg.get("data_dir")
    if not data_dir:
        raise ValueError(
            "data.data_dir is required to validate sentence-memory motion statistics"
        )
    metadata_dir = Path(data_dir) / "meta"
    return metadata_dir / "mean_rot6d.npy", metadata_dir / "std_rot6d.npy"


def require_literal_train_split(
    cfg: Mapping[str, Any], requested_split: str | None = None
) -> str:
    """Fail closed unless both configured and requested bank sources are train."""

    configured = str(dict(cfg.get("data", {}) or {}).get("train_split", "train"))
    if configured != "train":
        raise SentenceMemoryValidationError(
            "Sentence-motion memory requires data.train_split to be literally "
            f"'train', got {configured!r}"
        )
    if requested_split is not None and str(requested_split) != "train":
        raise SentenceMemoryValidationError(
            "Sentence-motion memory can only be sourced from literal split "
            f"'train', got {requested_split!r}"
        )
    return "train"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SentenceMemoryValidationError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise SentenceMemoryValidationError(
                    f"Expected an object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def normalize_sentence_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(value.casefold().split())


def sentence_text_hash(text: Any) -> str:
    return hashlib.sha256(normalize_sentence_text(text).encode("utf-8")).hexdigest()


def row_source_id(row: Mapping[str, Any]) -> str:
    for field in ("source_name", "sentence_id", "motion_path", "name"):
        value = row.get(field)
        if value is not None and str(value):
            return str(value)
    return ""


def row_group_id(row: Mapping[str, Any], group_fields: Sequence[str] | None = None) -> str:
    fields = tuple(group_fields or ("sentence_id", "source_name", "name"))
    for field in fields:
        value = row.get(field)
        if value is not None and str(value):
            return f"{field}:{value}"
    return ""


def semantic_group_id(row: Mapping[str, Any]) -> str:
    return str(row.get("semantic_group_id") or row.get("text_hash") or sentence_text_hash(row.get("text", "")))


def source_group_id(row: Mapping[str, Any], group_fields: Sequence[str] | None = None) -> str:
    return str(row.get("source_group_id") or row_group_id(row, group_fields))


def enrich_metadata_row(
    row: Mapping[str, Any],
    *,
    bank_index: int,
    encoded_frame_length: int,
    latent_length: int,
    group_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    # Keep the runtime index lightweight. Real manifests (especially
    # How2Sign) can contain large frame-level pseudo-gloss structures that are
    # useful to preprocessing but irrelevant to sentence retrieval.
    retained_fields = (
        "name",
        "motion_path",
        "source_name",
        "source_split",
        "sentence_id",
        "video_id",
        "video_name",
        "dataset",
        "signer",
        "fps",
        "num_frames",
        "duration",
    )
    output = {field: row[field] for field in retained_fields if field in row}
    output.update(
        {
            "bank_index": int(bank_index),
            "encoded_frame_length": int(encoded_frame_length),
            "latent_length": int(latent_length),
            "text_hash": sentence_text_hash(row.get("text", "")),
            "source_id": row_source_id(row),
            "semantic_group_id": sentence_text_hash(row.get("text", "")),
            "source_group_id": row_group_id(row, group_fields),
        }
    )
    return output


def _selected_text_artifacts(model_path: Path) -> list[Path]:
    """Select the files that determine a local Transformers encoder/tokenizer.

    Some checked-in model directories contain redundant PyTorch, TensorFlow,
    Flax, and safetensors weights. Transformers prefers safetensors, so hashing
    every framework copy would add minutes without strengthening the identity.
    """

    if model_path.is_file():
        return [model_path]
    if not model_path.is_dir():
        raise FileNotFoundError(f"Text encoder path does not exist: {model_path}")

    names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
    )
    selected = [model_path / name for name in names if (model_path / name).is_file()]

    safetensors_index = model_path / "model.safetensors.index.json"
    pytorch_index = model_path / "pytorch_model.bin.index.json"
    if safetensors_index.is_file():
        selected.append(safetensors_index)
        index = json.loads(safetensors_index.read_text(encoding="utf-8"))
        shard_names = sorted(set((index.get("weight_map") or {}).values()))
        selected.extend(model_path / name for name in shard_names)
    elif (model_path / "model.safetensors").is_file():
        selected.append(model_path / "model.safetensors")
    elif pytorch_index.is_file():
        selected.append(pytorch_index)
        index = json.loads(pytorch_index.read_text(encoding="utf-8"))
        shard_names = sorted(set((index.get("weight_map") or {}).values()))
        selected.extend(model_path / name for name in shard_names)
    elif (model_path / "pytorch_model.bin").is_file():
        selected.append(model_path / "pytorch_model.bin")

    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing text encoder artifacts: {missing[:3]}")
    if not selected:
        raise FileNotFoundError(f"No model or tokenizer artifacts found in {model_path}")
    return sorted(set(selected), key=lambda path: path.as_posix())


def fingerprint_text_encoder_artifacts(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    base = model_path if model_path.is_dir() else model_path.parent
    files = []
    for path in _selected_text_artifacts(model_path):
        files.append(
            {
                "path": path.relative_to(base).as_posix() if path != base else path.name,
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "digest": digest_json(files),
    }


def complete_text_encoder_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(_jsonable(identity))
    model_path = identity.get("model_path")
    if not model_path:
        raise ValueError("Text encoder identity must contain model_path")
    artifacts = fingerprint_text_encoder_artifacts(model_path)
    core_fields = (
        "model_type",
        "text_dim",
        "max_length",
        "encoder_loader",
        "tokenizer_loader",
        "tokenizer_use_fast",
        "encoder_class",
        "tokenizer_class",
        "tokenizer_is_fast",
    )
    core = {key: identity[key] for key in core_fields if key in identity}
    return {
        **core,
        "model_path": Path(model_path).as_posix(),
        "artifacts": artifacts,
        "semantic_digest": digest_json({"core": core, "artifacts": artifacts}),
    }


def compare_text_encoder_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    core_fields = (
        "model_type",
        "text_dim",
        "max_length",
        "encoder_loader",
        "tokenizer_loader",
        "tokenizer_use_fast",
        "encoder_class",
        "tokenizer_class",
        "tokenizer_is_fast",
    )
    shared_core = {
        key: (expected.get(key), actual.get(key))
        for key in core_fields
        if key in expected and key in actual
    }
    mismatches = {
        key: values for key, values in shared_core.items() if values[0] != values[1]
    }
    required = ("model_type", "text_dim", "max_length")
    missing_required = [
        key for key in required if key not in expected or key not in actual
    ]
    if mismatches or missing_required:
        raise SentenceMemoryValidationError(
            "Text encoder contract mismatch: "
            f"differences={mismatches}, missing_required={missing_required}"
        )
    expected_digest = (expected.get("artifacts") or {}).get("digest")
    actual_digest = (actual.get("artifacts") or {}).get("digest")
    if expected_digest and actual_digest and expected_digest != actual_digest:
        raise SentenceMemoryValidationError(
            "Text encoder artifacts differ from those used to build the sentence memory: "
            f"bank={expected_digest}, active={actual_digest}"
        )


def artifact_record(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "file": path.name,
        "bytes": int(path.stat().st_size),
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record["shape"] = [int(value) for value in array.shape]
        record["dtype"] = str(array.dtype)
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def compute_bank_id(manifest: Mapping[str, Any]) -> str:
    identity = dict(_jsonable(manifest))
    identity.pop("bank_id", None)
    return digest_json(identity)


def write_bank_manifest(bank_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    # A manifest rewrite invalidates a previous completed build immediately.
    # READY is recreated only after the new artifacts pass strict validation.
    (bank_dir / "READY").unlink(missing_ok=True)
    payload = dict(_jsonable(manifest))
    payload["schema_name"] = BANK_SCHEMA_NAME
    payload["schema_version"] = BANK_SCHEMA_VERSION
    payload["bank_id"] = compute_bank_id(payload)
    temporary = bank_dir / "bank.tmp.json"
    destination = bank_dir / "bank.json"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def write_build_summary(bank_dir: str | Path, summary: Mapping[str, Any]) -> Path:
    bank_dir = Path(bank_dir)
    manifest = load_bank_manifest(bank_dir)
    payload = {
        "schema_name": BANK_SCHEMA_NAME,
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_id": manifest["bank_id"],
        **dict(_jsonable(summary)),
    }
    # Required identity fields cannot be overridden by the caller.
    payload.update(
        {
            "schema_name": BANK_SCHEMA_NAME,
            "schema_version": BANK_SCHEMA_VERSION,
            "bank_id": manifest["bank_id"],
        }
    )
    temporary = bank_dir / "build_summary.tmp.json"
    destination = bank_dir / "build_summary.json"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def write_ready_marker(bank_dir: str | Path) -> Path:
    bank_dir = Path(bank_dir)
    manifest = load_bank_manifest(bank_dir)
    summary_path = bank_dir / "build_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Sentence-memory build summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("bank_id") != manifest["bank_id"]:
        raise SentenceMemoryValidationError(
            "Cannot mark sentence-memory bank ready: build summary bank_id mismatch"
        )
    payload = {
        "schema_name": BANK_SCHEMA_NAME,
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_id": manifest["bank_id"],
        "build_summary_sha256": sha256_file(summary_path),
    }
    temporary = bank_dir / "READY.tmp"
    destination = bank_dir / "READY"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _resolve_artifact_path(bank_dir: Path, record: Mapping[str, Any]) -> Path:
    relative = Path(str(record.get("file", "")))
    if not relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
        raise SentenceMemoryValidationError(f"Unsafe artifact path in bank: {relative}")
    path = (bank_dir / relative).resolve()
    root = bank_dir.resolve()
    if path != root and root not in path.parents:
        raise SentenceMemoryValidationError(f"Artifact escapes bank directory: {relative}")
    return path


def _check_array_record(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(value) for value in record.get("shape", ()))
    if expected_shape and array.shape != expected_shape:
        raise SentenceMemoryValidationError(
            f"Artifact shape mismatch for {path}: {array.shape} != {expected_shape}"
        )
    expected_dtype = record.get("dtype")
    if expected_dtype and str(array.dtype) != str(expected_dtype):
        raise SentenceMemoryValidationError(
            f"Artifact dtype mismatch for {path}: {array.dtype} != {expected_dtype}"
        )
    return array


def _rows_per_validation_chunk(
    array: np.ndarray,
    *,
    max_chunk_bytes: int,
) -> int:
    """Choose a first-axis chunk size without materializing an mmap payload."""

    if int(max_chunk_bytes) < 1:
        raise ValueError("max_chunk_bytes must be positive")
    values_per_row = int(np.prod(array.shape[1:], dtype=np.int64))
    row_bytes = max(values_per_row * int(array.dtype.itemsize), 1)
    return max(int(max_chunk_bytes) // row_bytes, 1)


def _validate_finite_array_chunked(
    array: np.ndarray,
    *,
    name: str,
    max_chunk_bytes: int,
) -> None:
    """Fail on non-finite mmap values while bounding resident scan memory."""

    rows_per_chunk = _rows_per_validation_chunk(
        array, max_chunk_bytes=max_chunk_bytes
    )
    for start in range(0, len(array), rows_per_chunk):
        end = min(start + rows_per_chunk, len(array))
        chunk = np.asarray(array[start:end])
        if not bool(np.isfinite(chunk).all()):
            raise SentenceMemoryValidationError(
                f"Non-finite values in {name} rows [{start}, {end})"
            )


def _validate_positive_array_chunked(
    array: np.ndarray,
    *,
    name: str,
    max_chunk_bytes: int,
) -> None:
    rows_per_chunk = _rows_per_validation_chunk(
        array, max_chunk_bytes=max_chunk_bytes
    )
    for start in range(0, len(array), rows_per_chunk):
        end = min(start + rows_per_chunk, len(array))
        if bool(np.any(np.asarray(array[start:end]) <= 0.0)):
            raise SentenceMemoryValidationError(
                f"{name} must be positive; invalid value in rows [{start}, {end})"
            )


def _validate_normalized_keys_chunked(
    keys: np.ndarray,
    *,
    max_chunk_bytes: int,
) -> None:
    rows_per_chunk = _rows_per_validation_chunk(
        keys, max_chunk_bytes=max_chunk_bytes
    )
    for start in range(0, len(keys), rows_per_chunk):
        end = min(start + rows_per_chunk, len(keys))
        chunk = np.asarray(keys[start:end], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1)
        if not bool(np.allclose(norms, 1.0, atol=2e-2, rtol=2e-2)):
            raise SentenceMemoryValidationError(
                f"group_keys are not L2-normalized in rows [{start}, {end})"
            )


def load_bank_manifest(bank_dir: str | Path) -> dict[str, Any]:
    path = Path(bank_dir) / "bank.json"
    if not path.is_file():
        raise FileNotFoundError(f"Sentence-memory manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != BANK_SCHEMA_NAME:
        raise SentenceMemoryValidationError(
            f"Unsupported sentence-memory schema {manifest.get('schema_name')!r}"
        )
    if int(manifest.get("schema_version", -1)) != BANK_SCHEMA_VERSION:
        raise SentenceMemoryValidationError(
            f"Unsupported sentence-memory schema version {manifest.get('schema_version')!r}; "
            f"expected {BANK_SCHEMA_VERSION}"
        )
    expected_id = compute_bank_id(manifest)
    if manifest.get("bank_id") != expected_id:
        raise SentenceMemoryValidationError(
            f"Sentence-memory bank_id is invalid: {manifest.get('bank_id')} != {expected_id}"
        )
    return manifest


def validate_sentence_memory_bank(
    bank_dir: str | Path,
    *,
    expected_bank_id: str | None = None,
    expected_manifest_path: str | Path | None = None,
    text_encoder_identity: Mapping[str, Any] | None = None,
    expected_vae_checkpoint: str | Path | None = None,
    expected_preprocessing: Mapping[str, Any] | None = None,
    expected_mean_path: str | Path | None = None,
    expected_std_path: str | Path | None = None,
    verify_hashes: bool = False,
    verify_contents: bool = False,
    content_scan_chunk_bytes: int = 64 * 1024 * 1024,
    require_ready: bool = True,
) -> dict[str, Any]:
    bank_dir = Path(bank_dir)
    manifest = load_bank_manifest(bank_dir)
    if expected_bank_id and manifest["bank_id"] != str(expected_bank_id):
        raise SentenceMemoryValidationError(
            f"Loaded sentence-memory bank {manifest['bank_id']}, expected {expected_bank_id}"
        )
    summary_path = bank_dir / "build_summary.json"
    if not summary_path.is_file():
        raise SentenceMemoryValidationError(
            f"Sentence-memory bank has no build_summary.json: {bank_dir}"
        )
    try:
        build_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SentenceMemoryValidationError(
            f"Invalid sentence-memory build summary: {summary_path}"
        ) from error
    if (
        build_summary.get("schema_name") != BANK_SCHEMA_NAME
        or int(build_summary.get("schema_version", -1)) != BANK_SCHEMA_VERSION
        or build_summary.get("bank_id") != manifest["bank_id"]
    ):
        raise SentenceMemoryValidationError(
            "Sentence-memory build summary does not match bank.json"
        )
    if require_ready:
        ready_path = bank_dir / "READY"
        if not ready_path.is_file():
            raise SentenceMemoryValidationError(
                f"Sentence-memory bank is incomplete (READY is missing): {bank_dir}"
            )
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SentenceMemoryValidationError(
                f"Invalid sentence-memory READY marker: {ready_path}"
            ) from error
        if (
            ready.get("schema_name") != BANK_SCHEMA_NAME
            or int(ready.get("schema_version", -1)) != BANK_SCHEMA_VERSION
            or ready.get("bank_id") != manifest["bank_id"]
            or ready.get("build_summary_sha256") != sha256_file(summary_path)
        ):
            raise SentenceMemoryValidationError(
                "Sentence-memory READY marker does not match the completed bank"
            )
    source = manifest.get("source") or {}
    if str(source.get("split", "")) != "train":
        raise SentenceMemoryValidationError(
            f"Sentence-motion bank must be train-only, got split={source.get('split')!r}"
        )
    if expected_manifest_path is not None:
        expected_manifest_path = Path(expected_manifest_path)
        actual_hash = sha256_file(expected_manifest_path)
        if source.get("manifest_sha256") != actual_hash:
            raise SentenceMemoryValidationError(
                "Training manifest differs from the one used to build the bank: "
                f"bank={source.get('manifest_sha256')}, active={actual_hash}"
            )
        if int(source.get("limit", 0) or 0) <= 0:
            with expected_manifest_path.open("r", encoding="utf-8") as handle:
                manifest_rows = sum(1 for line in handle if line.strip())
            if int(source.get("row_count", -1)) != manifest_rows:
                raise SentenceMemoryValidationError(
                    "Sentence-memory row count does not cover the complete training "
                    f"manifest: bank={source.get('row_count')}, manifest={manifest_rows}"
                )

    if expected_preprocessing is not None:
        persisted_preprocessing = manifest.get("preprocessing") or {}
        if digest_json(persisted_preprocessing) != digest_json(expected_preprocessing):
            raise SentenceMemoryValidationError(
                "Active motion preprocessing differs from the sentence-memory bank: "
                f"bank={canonical_json(persisted_preprocessing)}, "
                f"active={canonical_json(expected_preprocessing)}"
            )
    codec_contract = manifest.get("codec") or {}
    for label, active_path, hash_field in (
        ("mean", expected_mean_path, "mean_sha256"),
        ("std", expected_std_path, "std_sha256"),
    ):
        if active_path is None:
            continue
        active_hash = sha256_file(active_path)
        bank_hash = str(codec_contract.get(hash_field, ""))
        if not bank_hash or active_hash != bank_hash:
            raise SentenceMemoryValidationError(
                f"Active rot6d {label} statistics differ from the sentence-memory "
                f"bank: bank={bank_hash or '<missing>'}, active={active_hash}"
            )

    artifacts = manifest.get("artifacts") or {}
    missing_records = [name for name in REQUIRED_ARTIFACTS if name not in artifacts]
    if missing_records:
        raise SentenceMemoryValidationError(f"Bank is missing artifact records: {missing_records}")
    arrays = {}
    for name, record in artifacts.items():
        path = _resolve_artifact_path(bank_dir, record)
        if not path.is_file():
            raise FileNotFoundError(f"Sentence-memory artifact does not exist: {path}")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise SentenceMemoryValidationError(
                f"Artifact byte-size mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
            )
        if verify_hashes and record.get("sha256") and sha256_file(path) != record["sha256"]:
            raise SentenceMemoryValidationError(f"Artifact SHA256 mismatch: {path}")
        if path.suffix == ".npy":
            arrays[name] = _check_array_record(path, record)

    keys = arrays["group_keys"]
    group_offsets = arrays["group_offsets"]
    group_item_ids = arrays["group_item_ids"]
    item_group_ids = arrays["item_group_ids"]
    motion = arrays["motion_mu"]
    offsets = arrays["offsets"]
    item_motion_lengths = arrays["item_motion_lengths"]
    hand_valid = arrays["hand_valid"]
    durations = arrays["durations"]
    count = int(source.get("row_count", -1))
    if count < 1:
        raise SentenceMemoryValidationError(f"Invalid bank row count: {count}")
    group_count = int(source.get("semantic_group_count", -1))
    if group_count < 1:
        raise SentenceMemoryValidationError(f"Invalid semantic group count: {group_count}")
    if keys.ndim != 2 or len(keys) != group_count or str(keys.dtype) != "float32":
        raise SentenceMemoryValidationError(
            f"group_keys must be float32 [G,D] with G={group_count}"
        )
    if group_offsets.shape != (group_count + 1,):
        raise SentenceMemoryValidationError(
            f"group_offsets must have shape [{group_count + 1}]"
        )
    if group_item_ids.shape != (count,) or item_group_ids.shape != (count,):
        raise SentenceMemoryValidationError(
            f"group_item_ids and item_group_ids must each have shape [{count}]"
        )
    if motion.ndim != 2 or motion.shape[1] != int((manifest.get("codec") or {}).get("latent_dim", -1)):
        raise SentenceMemoryValidationError("motion_mu shape disagrees with codec.latent_dim")
    if offsets.shape != (count + 1,):
        raise SentenceMemoryValidationError(f"offsets must have shape [{count + 1}]")
    if item_motion_lengths.shape != (count,) or str(item_motion_lengths.dtype) != "int16":
        raise SentenceMemoryValidationError(
            f"item_motion_lengths must be int16 with shape [{count}]"
        )
    if durations.shape != (count,):
        raise SentenceMemoryValidationError(f"durations must have shape [{count}]")
    if hand_valid.shape != (len(motion), 2) or str(hand_valid.dtype) != "uint8":
        raise SentenceMemoryValidationError(
            "hand_valid must be uint8 with shape [sum_latent_tokens,2]"
        )
    offsets_array = np.asarray(offsets, dtype=np.int64)
    if offsets_array[0] != 0 or offsets_array[-1] != len(motion):
        raise SentenceMemoryValidationError("offsets endpoints do not match motion_mu")
    if bool(np.any(offsets_array[1:] < offsets_array[:-1])):
        raise SentenceMemoryValidationError("offsets must be monotonically nondecreasing")
    lengths_array = np.asarray(item_motion_lengths, dtype=np.int64)
    if bool(np.any(lengths_array <= 0)) or not np.array_equal(
        lengths_array, np.diff(offsets_array)
    ):
        raise SentenceMemoryValidationError(
            "item_motion_lengths must be positive and exactly match offsets"
        )
    group_offsets_array = np.asarray(group_offsets, dtype=np.int64)
    group_items_array = np.asarray(group_item_ids, dtype=np.int64)
    item_groups_array = np.asarray(item_group_ids, dtype=np.int64)
    if group_offsets_array[0] != 0 or group_offsets_array[-1] != count:
        raise SentenceMemoryValidationError("group_offsets endpoints do not match item count")
    if bool(np.any(group_offsets_array[1:] < group_offsets_array[:-1])):
        raise SentenceMemoryValidationError("group_offsets must be monotonically nondecreasing")
    if sorted(group_items_array.tolist()) != list(range(count)):
        raise SentenceMemoryValidationError("group_item_ids must be a permutation of item IDs")
    if bool(np.any((item_groups_array < 0) | (item_groups_array >= group_count))):
        raise SentenceMemoryValidationError("item_group_ids contains out-of-range group IDs")
    for group_index in range(group_count):
        members = group_items_array[
            group_offsets_array[group_index] : group_offsets_array[group_index + 1]
        ]
        if len(members) == 0 or bool(np.any(item_groups_array[members] != group_index)):
            raise SentenceMemoryValidationError(
                f"Semantic group membership is inconsistent at group {group_index}"
            )

    metadata_path = _resolve_artifact_path(bank_dir, artifacts["metadata"])
    metadata = read_jsonl(metadata_path)
    if len(metadata) != count:
        raise SentenceMemoryValidationError(
            f"Metadata has {len(metadata)} rows, expected {count}"
        )
    for index, row in enumerate(metadata):
        if int(row.get("bank_index", -1)) != index:
            raise SentenceMemoryValidationError(f"Metadata row {index} has wrong bank_index")
        latent_length = int(offsets_array[index + 1] - offsets_array[index])
        if int(row.get("latent_length", -1)) != latent_length:
            raise SentenceMemoryValidationError(
                f"Metadata row {index} latent_length disagrees with offsets"
            )
        if int(row.get("semantic_group_index", -1)) != int(item_groups_array[index]):
            raise SentenceMemoryValidationError(
                f"Metadata row {index} semantic_group_index disagrees with item_group_ids"
            )
        if not str(row.get("semantic_group_id", "")):
            raise SentenceMemoryValidationError(
                f"Metadata row {index} has no canonical semantic_group_id"
            )
        if not str(row.get("source_group_id", "")):
            raise SentenceMemoryValidationError(
                f"Metadata row {index} has no canonical source_group_id"
            )
    groups_path = _resolve_artifact_path(bank_dir, artifacts["groups"])
    groups = read_jsonl(groups_path)
    if len(groups) != group_count:
        raise SentenceMemoryValidationError(
            f"Group metadata has {len(groups)} rows, expected {group_count}"
        )
    for group_index, row in enumerate(groups):
        if int(row.get("semantic_group_index", -1)) != group_index:
            raise SentenceMemoryValidationError(
                f"Group metadata row {group_index} has wrong semantic_group_index"
            )

    summary_contract = {
        "row_count": count,
        "semantic_group_count": group_count,
        "latent_tokens": len(motion),
    }
    for field, expected in summary_contract.items():
        if int(build_summary.get(field, -1)) != int(expected):
            raise SentenceMemoryValidationError(
                f"build_summary.{field}={build_summary.get(field)!r}, expected {expected}"
            )

    if verify_contents:
        chunk_bytes = int(content_scan_chunk_bytes)
        for name, array in (
            ("group_keys", keys),
            ("motion_mu", motion),
            ("durations", durations),
        ):
            _validate_finite_array_chunked(
                array,
                name=name,
                max_chunk_bytes=chunk_bytes,
            )
        _validate_normalized_keys_chunked(
            keys,
            max_chunk_bytes=chunk_bytes,
        )
        _validate_positive_array_chunked(
            durations,
            name="durations",
            max_chunk_bytes=chunk_bytes,
        )

    if text_encoder_identity is not None:
        active_identity = complete_text_encoder_identity(text_encoder_identity)
        compare_text_encoder_identity(manifest.get("text_encoder") or {}, active_identity)
    if expected_vae_checkpoint is not None:
        checkpoint_hash = sha256_file(expected_vae_checkpoint)
        expected_hash = (manifest.get("codec") or {}).get("checkpoint_sha256")
        if checkpoint_hash != expected_hash:
            raise SentenceMemoryValidationError(
                f"VAE checkpoint differs from bank codec: {checkpoint_hash} != {expected_hash}"
            )
    return manifest


def exclusion_policy(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    return {
        "exclude_self": bool(cfg.get("exclude_self", True)),
        "exclude_same_source": bool(cfg.get("exclude_same_source", True)),
        "exclude_same_group": bool(cfg.get("exclude_same_group", True)),
        "exclude_exact_text_train": bool(cfg.get("exclude_exact_text_train", True)),
        "exclude_exact_text_eval": bool(cfg.get("exclude_exact_text_eval", False)),
        "group_fields": list(cfg.get("group_fields") or ("sentence_id", "source_name", "name")),
    }


def candidate_is_allowed(
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    training: bool,
    policy: Mapping[str, Any],
) -> bool:
    if policy.get("exclude_self", True):
        for field in ("name", "motion_path"):
            left = str(query.get(field, ""))
            right = str(candidate.get(field, ""))
            if left and right and left == right:
                return False
    if policy.get("exclude_same_source", True):
        left = str(query.get("source_id") or row_source_id(query))
        right = str(candidate.get("source_id") or row_source_id(candidate))
        if left and right and left == right:
            return False
    if policy.get("exclude_same_group", True):
        fields = policy.get("group_fields")
        left = source_group_id(query, fields)
        right = source_group_id(candidate, fields)
        if left and right and left == right:
            return False
    exclude_text = (
        policy.get("exclude_exact_text_train", True)
        if training
        else policy.get("exclude_exact_text_eval", False)
    )
    if exclude_text:
        left = str(query.get("text_hash") or sentence_text_hash(query.get("text", "")))
        right = str(candidate.get("text_hash") or sentence_text_hash(candidate.get("text", "")))
        if left and right and left == right:
            return False
    return True


def _npz_scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).ndim == 0:
        return np.asarray(value).item()
    return value


def _validate_neighbor_rows(ids: np.ndarray, scores: np.ndarray, *, source: str) -> None:
    """Validate the canonical top-M semantic-group row representation."""

    for row_index, (row_ids, row_scores) in enumerate(zip(ids, scores)):
        valid_mask = row_ids >= 0
        valid_ids = row_ids[valid_mask]
        if len(valid_ids) != len(np.unique(valid_ids)):
            raise ValueError(
                f"Neighbor row {row_index} contains duplicate semantic groups: {source}"
            )
        if bool(np.any(valid_mask & np.maximum.accumulate(~valid_mask))):
            raise ValueError(
                f"Neighbor row {row_index} has a valid ID after padding: {source}"
            )
        valid_scores = row_scores[valid_mask]
        if len(valid_scores) > 1 and bool(np.any(valid_scores[1:] > valid_scores[:-1])):
            raise ValueError(
                f"Neighbor row {row_index} is not sorted by cosine score: {source}"
            )


def write_neighbor_table(
    path: str | Path,
    *,
    bank_id: str,
    query_split: str,
    query_manifest_sha256: str,
    query_names: Sequence[str],
    ids: np.ndarray,
    scores: np.ndarray,
    policy: Mapping[str, Any],
    scoring: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    if ids.shape != scores.shape or ids.ndim != 2:
        raise ValueError("Neighbor ids and scores must have the same [Q,M] shape")
    if ids.shape[0] != len(query_names):
        raise ValueError("Neighbor row count must match query_names")
    if not bool(np.isfinite(scores).all()):
        raise ValueError("Neighbor scores must all be finite")
    _validate_neighbor_rows(ids, scores, source=str(path))
    max_name = max((len(str(name)) for name in query_names), default=1)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_name=np.asarray(NEIGHBOR_SCHEMA_NAME),
        schema_version=np.asarray(NEIGHBOR_SCHEMA_VERSION, dtype=np.int32),
        bank_id=np.asarray(str(bank_id)),
        query_split=np.asarray(str(query_split)),
        query_manifest_sha256=np.asarray(str(query_manifest_sha256)),
        query_order_sha256=np.asarray(digest_json([str(name) for name in query_names])),
        query_names=np.asarray([str(name) for name in query_names], dtype=f"<U{max_name}"),
        top_m=np.asarray(ids.shape[1], dtype=np.int32),
        ids=ids,
        scores=scores,
        filter_policy=np.asarray(canonical_json(policy)),
        scoring=np.asarray(canonical_json(scoring or {"metric": "cosine"})),
    )
    temporary.replace(path)
    return path


def load_neighbor_table(
    path: str | Path,
    *,
    bank_id: str,
    bank_size: int,
    expected_manifest_sha256: str | None = None,
    expected_policy: Mapping[str, Any] | None = None,
    expected_top_m: int | None = None,
) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if str(_npz_scalar(data, "schema_name", "")) != NEIGHBOR_SCHEMA_NAME:
            raise SentenceMemoryValidationError(f"Invalid neighbor schema: {path}")
        if int(_npz_scalar(data, "schema_version", -1)) != NEIGHBOR_SCHEMA_VERSION:
            raise SentenceMemoryValidationError(f"Invalid neighbor schema version: {path}")
        if str(_npz_scalar(data, "bank_id", "")) != str(bank_id):
            raise SentenceMemoryValidationError(f"Neighbor table bank_id mismatch: {path}")
        manifest_hash = str(_npz_scalar(data, "query_manifest_sha256", ""))
        if expected_manifest_sha256 and manifest_hash != expected_manifest_sha256:
            raise SentenceMemoryValidationError(
                f"Neighbor table query manifest mismatch: {manifest_hash} != {expected_manifest_sha256}"
            )
        policy = json.loads(str(_npz_scalar(data, "filter_policy", "{}")))
        if expected_policy is not None and digest_json(policy) != digest_json(expected_policy):
            raise SentenceMemoryValidationError(f"Neighbor filter policy mismatch: {path}")
        ids = np.asarray(data["ids"], dtype=np.int64)
        scores = np.asarray(data["scores"], dtype=np.float32)
        names = [str(value) for value in np.asarray(data["query_names"]).tolist()]
        if ids.ndim != 2 or scores.shape != ids.shape or len(names) != len(ids):
            raise SentenceMemoryValidationError(f"Malformed neighbor arrays: {path}")
        declared_top_m = int(_npz_scalar(data, "top_m", -1))
        if declared_top_m != ids.shape[1] or declared_top_m <= 0:
            raise SentenceMemoryValidationError(
                f"Neighbor top_m does not match its array width: {path}"
            )
        if expected_top_m is not None and declared_top_m != int(expected_top_m):
            raise SentenceMemoryValidationError(
                "Neighbor table width differs from sentence_memory.top_m: "
                f"table={declared_top_m}, configured={int(expected_top_m)}, path={path}"
            )
        if bool(np.any((ids < -1) | (ids >= int(bank_size)))):
            raise SentenceMemoryValidationError(f"Neighbor IDs out of range: {path}")
        if not bool(np.isfinite(scores).all()):
            raise SentenceMemoryValidationError(f"Non-finite neighbor scores: {path}")
        try:
            _validate_neighbor_rows(ids, scores, source=str(path))
        except ValueError as error:
            raise SentenceMemoryValidationError(str(error)) from error
        order_hash = str(_npz_scalar(data, "query_order_sha256", ""))
        if order_hash != digest_json(names):
            raise SentenceMemoryValidationError(f"Neighbor query order hash mismatch: {path}")
        return {
            "path": str(path),
            "query_split": str(_npz_scalar(data, "query_split", "")),
            "query_manifest_sha256": manifest_hash,
            "query_order_sha256": order_hash,
            "query_names": names,
            "top_m": declared_top_m,
            "ids": ids,
            "scores": scores,
            "filter_policy": policy,
            "scoring": json.loads(str(_npz_scalar(data, "scoring", "{}"))),
        }


def sentence_neighbor_table_identity(
    path: str | Path,
    *,
    bank_id: str,
    bank_size: int,
    expected_policy: Mapping[str, Any],
    expected_top_m: int | None = None,
) -> dict[str, Any]:
    """Return a path-independent identity for one validated neighbor table."""

    path = Path(path)
    table = load_neighbor_table(
        path,
        bank_id=bank_id,
        bank_size=bank_size,
        expected_policy=expected_policy,
        expected_top_m=expected_top_m,
    )
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "schema_version": NEIGHBOR_SCHEMA_VERSION,
        "query_split": table["query_split"],
        "query_manifest_sha256": table["query_manifest_sha256"],
        "query_order_sha256": table["query_order_sha256"],
        "query_count": len(table["query_names"]),
        "top_m": table["top_m"],
        "filter_policy_sha256": digest_json(table["filter_policy"]),
        "scoring": table["scoring"],
    }


def validate_neighbor_manifest_coverage(
    table: Mapping[str, Any], manifest_path: str | Path
) -> None:
    """Require one correctly ordered neighbor row for every manifest example."""

    names = list(table.get("query_names") or [])
    row_count = 0
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SentenceMemoryValidationError(
                    f"Invalid JSON at {manifest_path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise SentenceMemoryValidationError(
                    f"Expected an object at {manifest_path}:{line_number}"
                )
            if row_count >= len(names):
                raise SentenceMemoryValidationError(
                    "Neighbor table does not cover the complete query manifest: "
                    f"table_rows={len(names)}, manifest_rows>={row_count + 1}"
                )
            expected_name = str(row.get("name", ""))
            if str(names[row_count]) != expected_name:
                raise SentenceMemoryValidationError(
                    "Neighbor table query order differs from the active manifest at "
                    f"row {row_count}: table={names[row_count]!r}, "
                    f"manifest={expected_name!r}"
                )
            row_count += 1
    if row_count != len(names):
        raise SentenceMemoryValidationError(
            "Neighbor table row count differs from the complete query manifest: "
            f"table_rows={len(names)}, manifest_rows={row_count}"
        )


@dataclass
class SentenceMemoryBatch:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    token_tau: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_keys: torch.Tensor
    scores: torch.Tensor
    durations: torch.Tensor
    duration_log_gap: torch.Tensor
    part_validity: torch.Tensor
    ids: torch.Tensor
    available: torch.Tensor
    provenance: dict[str, Any]

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "sentence_motion_tokens": self.tokens,
            "sentence_motion_mask": self.token_mask,
            "sentence_motion_tau": self.token_tau,
            "sentence_candidate_mask": self.candidate_mask,
            "sentence_text_keys": self.candidate_keys,
            "sentence_scores": self.scores,
            "sentence_durations": self.durations,
            "sentence_part_validity": self.part_validity,
            "sentence_memory_available": self.available,
        }


class SentenceMemoryProvider:
    """Retrieve a small set of motion latents from a train-only disk bank.

    The complete bank remains a CPU read-only mmap. Only selected, padded
    candidate spans are copied to the target device.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        text_encoder_identity: Mapping[str, Any],
        dataset=None,
    ):
        self.cfg = dict(cfg)
        self.memory_cfg = dict(cfg.get("sentence_memory", {}))
        configured_dir = self.memory_cfg.get("bank_dir")
        override_name = str(
            self.memory_cfg.get("local_bank_env", "SIGNTRAJ_SENTENCE_MEMORY_DIR")
        )
        override = os.environ.get(override_name)
        if not override and not configured_dir:
            raise ValueError("sentence_memory.bank_dir is required")
        self.bank_dir = Path(override or configured_dir)
        expected_vae = self.memory_cfg.get("codec_checkpoint") or dict(
            cfg.get("adapter", {})
        ).get("vae_checkpoint")
        allow_partial = bool(
            self.memory_cfg.get("allow_partial_bank", False)
            or self.memory_cfg.get("allow_source_mismatch", False)
        )
        data_cfg = dict(cfg.get("data", {}))
        train_split = require_literal_train_split(cfg)
        expected_train_manifest = None
        if data_cfg and not allow_partial:
            explicit_manifest = data_cfg.get(f"{train_split}_manifest_path")
            data_dir = data_cfg.get("data_dir")
            if explicit_manifest:
                expected_train_manifest = Path(explicit_manifest)
            elif data_dir:
                expected_train_manifest = Path(data_dir) / "meta" / f"manifest_{train_split}.jsonl"
        expected_preprocessing = (
            sentence_memory_preprocessing_contract(cfg) if data_cfg else None
        )
        expected_mean_path = None
        expected_std_path = None
        if data_cfg.get("data_dir"):
            expected_mean_path, expected_std_path = sentence_memory_motion_stats_paths(
                cfg
            )
        self.manifest = validate_sentence_memory_bank(
            self.bank_dir,
            expected_bank_id=self.memory_cfg.get("expected_bank_id"),
            expected_manifest_path=expected_train_manifest,
            text_encoder_identity=text_encoder_identity,
            expected_vae_checkpoint=(
                expected_vae
                if expected_vae and bool(self.memory_cfg.get("verify_codec_checkpoint", True))
                else None
            ),
            expected_preprocessing=expected_preprocessing,
            expected_mean_path=expected_mean_path,
            expected_std_path=expected_std_path,
            verify_hashes=bool(self.memory_cfg.get("verify_artifact_hashes", False)),
            # Runtime must reject poisoned numeric payloads independently of
            # the optional, more expensive cryptographic hash verification.
            verify_contents=True,
            content_scan_chunk_bytes=int(
                self.memory_cfg.get(
                    "content_scan_chunk_bytes", 64 * 1024 * 1024
                )
            ),
        )
        source_limit = int((self.manifest.get("source") or {}).get("limit", 0) or 0)
        if source_limit > 0 and not allow_partial:
            raise SentenceMemoryValidationError(
                "Sentence-memory bank was built with a row limit; set "
                "sentence_memory.allow_partial_bank=true only for explicit debug/test runs"
            )
        artifacts = self.manifest["artifacts"]
        self.keys = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["group_keys"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.group_offsets = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["group_offsets"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.group_item_ids = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["group_item_ids"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.item_group_ids = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["item_group_ids"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.motion = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["motion_mu"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.offsets = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["offsets"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.item_motion_lengths = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["item_motion_lengths"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.hand_valid = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["hand_valid"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.durations = np.load(
            _resolve_artifact_path(self.bank_dir, artifacts["durations"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.metadata = read_jsonl(
            _resolve_artifact_path(self.bank_dir, artifacts["metadata"])
        )
        self.groups = read_jsonl(
            _resolve_artifact_path(self.bank_dir, artifacts["groups"])
        )
        self.bank_id = str(self.manifest["bank_id"])
        self.bank_size = len(self.metadata)
        self.group_count = len(self.groups)
        self.key_dim = int(self.keys.shape[1])
        self.latent_dim = int(self.motion.shape[1])
        self.k = max(int(self.memory_cfg.get("k", 8)), 1)
        self.top_m = max(int(self.memory_cfg.get("top_m", max(self.k, 64))), self.k)
        self.duration_weight = max(float(self.memory_cfg.get("duration_weight", 0.1)), 0.0)
        self.seed = int(cfg.get("seed", self.memory_cfg.get("seed", 1234)))
        self.epoch = 0
        self.policy = exclusion_policy(self.memory_cfg.get("filter"))
        bank_policy = self.manifest.get("filter_policy")
        if bank_policy and digest_json(bank_policy) != digest_json(self.policy):
            raise SentenceMemoryValidationError(
                "Configured sentence-memory filter policy differs from bank policy"
            )
        self.dataset = None
        self.neighbor_table = None
        self._dataset_manifest_sha256 = None
        self._bank_name_to_index = {
            str(row.get("name", "")): index
            for index, row in enumerate(self.metadata)
            if str(row.get("name", ""))
        }
        self._bank_path_to_index = {
            str(row.get("motion_path", "")): index
            for index, row in enumerate(self.metadata)
            if str(row.get("motion_path", ""))
        }
        self._semantic_group_ids = {
            str(row.get("semantic_group_id", ""))
            for row in self.groups
            if str(row.get("semantic_group_id", ""))
        }
        self._neighbor_tables_identity = self._collect_neighbor_tables_identity()
        if dataset is not None:
            self.set_dataset(dataset)

    def _collect_neighbor_tables_identity(self) -> dict[str, dict[str, Any]]:
        configured = dict(self.memory_cfg.get("neighbor_files") or {})
        splits = {"train", "val", "test", *map(str, configured)}
        data_cfg = dict(self.cfg.get("data") or {})
        splits.update(
            str(data_cfg[key])
            for key in ("train_split", "val_split", "test_split")
            if data_cfg.get(key)
        )
        identities = {}
        for split in sorted(splits):
            configured_path = configured.get(split)
            path = (
                Path(configured_path)
                if configured_path
                else self.bank_dir / f"neighbors_{split}.npz"
            )
            if not path.is_file():
                continue
            identities[split] = sentence_neighbor_table_identity(
                path,
                bank_id=self.bank_id,
                bank_size=self.group_count,
                expected_policy=self.policy,
                expected_top_m=self.top_m,
            )
        return identities

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "schema_version": BANK_SCHEMA_VERSION,
            "bank_dir": str(self.bank_dir),
            "row_count": self.bank_size,
            "semantic_group_count": self.group_count,
            "text_encoder": self.manifest.get("text_encoder"),
            "codec": self.manifest.get("codec"),
            "filter_policy": self.policy,
            "neighbor_tables": self._neighbor_tables_identity,
        }

    @property
    def config_summary(self) -> dict[str, Any]:
        return {
            **self.identity,
            "k": self.k,
            "top_m": self.top_m,
            "duration_weight": self.duration_weight,
            "sampling": str(self.memory_cfg.get("sampling", "weighted")),
            "sampling_temperature": float(
                self.memory_cfg.get("sampling_temperature", 0.07)
            ),
            "candidate_dropout_probability": float(
                self.memory_cfg.get("candidate_dropout_probability", 0.10)
            ),
            "query_split": getattr(self.dataset, "split", None),
            "neighbor_table": (
                self.neighbor_table.get("path") if self.neighbor_table is not None else None
            ),
        }

    def is_seen_text(self, text: Any) -> bool:
        """Return whether normalized query text has a semantic group in the train bank."""

        return sentence_text_hash(text) in self._semantic_group_ids

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _dataset_manifest_path(self, dataset) -> Path | None:
        base = getattr(dataset, "base", dataset)
        path = getattr(base, "manifest_path", None)
        return Path(path) if path is not None else None

    def _neighbor_path(self, split: str) -> Path:
        configured = self.memory_cfg.get("neighbor_files") or {}
        value = configured.get(str(split)) if isinstance(configured, Mapping) else None
        return Path(value) if value else self.bank_dir / f"neighbors_{split}.npz"

    def set_dataset(self, dataset, *, require_neighbors: bool = False):
        self.dataset = dataset
        self.neighbor_table = None
        manifest_path = self._dataset_manifest_path(dataset)
        self._dataset_manifest_sha256 = (
            sha256_file(manifest_path) if manifest_path is not None and manifest_path.is_file() else None
        )
        split = str(getattr(dataset, "split", ""))
        neighbor_path = self._neighbor_path(split)
        if neighbor_path.is_file():
            try:
                self.neighbor_table = load_neighbor_table(
                    neighbor_path,
                    bank_id=self.bank_id,
                    bank_size=self.group_count,
                    expected_manifest_sha256=self._dataset_manifest_sha256,
                    expected_policy=self.policy,
                    expected_top_m=self.top_m,
                )
                table_split = str(self.neighbor_table.get("query_split", ""))
                if table_split != split:
                    raise SentenceMemoryValidationError(
                        "Neighbor table split differs from the query dataset: "
                        f"table={table_split!r}, dataset={split!r}"
                    )
                if manifest_path is None or not manifest_path.is_file():
                    if require_neighbors:
                        raise SentenceMemoryValidationError(
                            "A readable query manifest is required to prove complete "
                            "neighbor-table coverage"
                        )
                else:
                    validate_neighbor_manifest_coverage(
                        self.neighbor_table, manifest_path
                    )
            except SentenceMemoryValidationError:
                if require_neighbors:
                    raise
                self.neighbor_table = None
        elif require_neighbors:
            raise FileNotFoundError(f"Required sentence-neighbor table does not exist: {neighbor_path}")
        return self

    def validate_query_dataset(self, dataset=None, *, require_neighbors: bool = False):
        if dataset is not None and dataset is not self.dataset:
            self.set_dataset(dataset, require_neighbors=require_neighbors)
        elif self.dataset is None:
            raise ValueError("SentenceMemoryProvider needs a query dataset")
        elif require_neighbors and self.neighbor_table is None:
            self.set_dataset(self.dataset, require_neighbors=True)
        return {
            "split": str(getattr(self.dataset, "split", "")),
            "manifest_sha256": self._dataset_manifest_sha256,
            "neighbor_table": (
                self.neighbor_table.get("path") if self.neighbor_table is not None else None
            ),
        }

    def _query_rows(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        names = [str(value) for value in batch.get("name", [])]
        size = len(names)
        paths = [str(value) for value in batch.get("motion_path", [""] * size)]
        texts = [str(value) for value in batch.get("text", [""] * size)]
        indices_value = batch.get("index")
        if torch.is_tensor(indices_value):
            indices = indices_value.detach().cpu().long().tolist()
        elif indices_value is None:
            indices = list(range(size))
        else:
            indices = list(indices_value)
        if len(indices) != size:
            raise ValueError("batch index and name fields have different lengths")
        base = getattr(self.dataset, "base", self.dataset)
        dataset_rows = getattr(base, "items", None)
        rows = []
        for position in range(size):
            index = int(indices[position])
            if dataset_rows is not None and 0 <= index < len(dataset_rows):
                row = dict(dataset_rows[index])
            else:
                row = {}
            row.setdefault("name", names[position])
            row.setdefault("motion_path", paths[position] if position < len(paths) else "")
            row.setdefault("text", texts[position] if position < len(texts) else "")
            row["query_index"] = index
            row["text_hash"] = sentence_text_hash(row.get("text", ""))
            row["semantic_group_id"] = row["text_hash"]
            row["source_id"] = row_source_id(row)
            row["source_group_id"] = row_group_id(
                row, self.policy.get("group_fields")
            )
            rows.append(row)
        return rows

    @staticmethod
    def _normalize_query_key(query_key: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(query_key):
            query_key = query_key.detach().cpu().float().numpy()
        query_key = np.asarray(query_key, dtype=np.float32)
        if query_key.ndim != 2:
            raise ValueError("query_key must have shape [B,D]")
        norm = np.linalg.norm(query_key, axis=-1, keepdims=True)
        return query_key / np.clip(norm, 1e-8, None)

    def _group_members(self, group_id: int) -> np.ndarray:
        start = int(self.group_offsets[int(group_id)])
        end = int(self.group_offsets[int(group_id) + 1])
        return np.asarray(self.group_item_ids[start:end], dtype=np.int64)

    def _eligible_items(
        self,
        query: Mapping[str, Any],
        group_id: int,
        training: bool,
    ) -> np.ndarray:
        members = self._group_members(group_id)
        keep = [
            candidate_is_allowed(
                query,
                self.metadata[int(item_id)],
                training=training,
                policy=self.policy,
            )
            for item_id in members
        ]
        return members[np.asarray(keep, dtype=np.bool_)]

    def _group_allowed(
        self,
        query: Mapping[str, Any],
        group_id: int,
        training: bool,
    ) -> bool:
        group = self.groups[int(group_id)]
        same_semantics = str(query.get("semantic_group_id", "")) == str(
            group.get("semantic_group_id", "")
        )
        exclude_semantics = (
            self.policy.get("exclude_exact_text_train", True)
            if training
            else self.policy.get("exclude_exact_text_eval", False)
        )
        if same_semantics and exclude_semantics:
            return False
        return len(self._eligible_items(query, group_id, training)) > 0

    def _select_group_item(
        self,
        query: Mapping[str, Any],
        group_id: int,
        predicted_duration: float | None,
        *,
        training: bool,
    ) -> int:
        eligible = self._eligible_items(query, group_id, training)
        if len(eligible) == 0:
            return -1
        if training:
            # Uniformly expose distinct recordings of the same sentence during
            # training, while remaining reproducible across DDP ranks/workers.
            token = (
                f"{self.seed}|{self.epoch}|variant|{query.get('name')}|"
                f"{query.get('motion_path')}|{query.get('query_index')}|{int(group_id)}"
            )
            draw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
            return int(eligible[draw % len(eligible)])

        # Evaluation is deterministic: duration compatibility wins, followed
        # by better hand tracking and finally the stable bank item ID.
        target_log_duration = (
            math.log(max(float(predicted_duration), 1e-6))
            if predicted_duration is not None
            else None
        )
        ranking = []
        for item_id in eligible:
            item_id = int(item_id)
            duration_gap = (
                abs(
                    math.log(max(float(self.durations[item_id]), 1e-6))
                    - target_log_duration
                )
                if target_log_duration is not None
                else 0.0
            )
            start = int(self.offsets[item_id])
            end = int(self.offsets[item_id + 1])
            hand_quality = float(
                np.asarray(self.hand_valid[start:end], dtype=np.float32).mean()
            )
            ranking.append((duration_gap, -hand_quality, item_id))
        return int(min(ranking)[2])

    def _online_candidates(
        self,
        query_key: np.ndarray,
        query: Mapping[str, Any],
        *,
        training: bool,
        limit: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.empty(self.group_count, dtype=np.float32)
        chunk_size = max(int(self.memory_cfg.get("search_chunk_size", 8192)), 1)
        for start in range(0, self.group_count, chunk_size):
            end = min(start + chunk_size, self.group_count)
            block = np.asarray(self.keys[start:end], dtype=np.float32)
            scores[start:end] = block @ query_key
        allowed = np.fromiter(
            (
                self._group_allowed(query, index, training)
                for index in range(self.group_count)
            ),
            dtype=np.bool_,
            count=self.group_count,
        )
        valid_ids = np.flatnonzero(allowed)
        if len(valid_ids) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        take = min(max(int(limit), 1), len(valid_ids))
        valid_scores = scores[valid_ids]
        if take < len(valid_ids):
            selected = np.argpartition(valid_scores, -take)[-take:]
            valid_ids = valid_ids[selected]
            valid_scores = valid_scores[selected]
        order = np.argsort(-valid_scores, kind="stable")
        return valid_ids[order].astype(np.int64), valid_scores[order].astype(np.float32)

    def _table_candidates(
        self,
        query: Mapping[str, Any],
        *,
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        table = self.neighbor_table
        if table is None:
            return None
        index = int(query.get("query_index", -1))
        if index < 0 or index >= len(table["ids"]):
            return None
        expected_name = str(table["query_names"][index])
        if expected_name and expected_name != str(query.get("name", "")):
            return None
        ids = table["ids"][index]
        scores = table["scores"][index]
        keep = np.asarray(
            [
                candidate_id >= 0
                and self._group_allowed(query, int(candidate_id), training)
                for candidate_id in ids
            ],
            dtype=np.bool_,
        )
        return ids[keep].astype(np.int64), scores[keep].astype(np.float32)

    def _rank_candidates(
        self,
        ids: np.ndarray,
        scores: np.ndarray,
        predicted_duration: float | None,
        query: Mapping[str, Any],
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(ids) == 0:
            return ids, scores, scores.astype(np.float64)
        rank_score = scores.astype(np.float64)
        if predicted_duration is not None and self.duration_weight > 0.0:
            candidate_duration = np.asarray(
                [
                    self.durations[
                        self._select_group_item(
                            query,
                            int(group_id),
                            predicted_duration,
                            training=training,
                        )
                    ]
                    for group_id in ids
                ],
                dtype=np.float64,
            )
            gap = np.abs(
                np.log(max(float(predicted_duration), 1e-6))
                - np.log(np.clip(candidate_duration, 1e-6, None))
            )
            rank_score = rank_score - self.duration_weight * gap
        order = np.argsort(-rank_score, kind="stable")
        return ids[order], scores[order], rank_score[order]

    def _sample_candidates(
        self,
        ids: np.ndarray,
        scores: np.ndarray,
        selection_scores: np.ndarray,
        query: Mapping[str, Any],
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        take = min(self.k, len(ids))
        strategy = str(
            self.memory_cfg.get(
                "sampling", self.memory_cfg.get("train_sampling", "weighted")
            )
        ).lower()
        if not training or strategy == "topk" or take <= 1:
            return ids[:take], scores[:take]
        if strategy not in {"weighted", "uniform"}:
            raise ValueError(
                f"Unsupported sentence_memory.sampling={strategy!r}; "
                "expected topk, weighted, or uniform"
            )
        token = f"{self.seed}|{self.epoch}|{query.get('name')}|{query.get('motion_path')}"
        seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        pool = min(len(ids), self.top_m)
        probabilities = None
        if strategy == "weighted":
            temperature = max(float(self.memory_cfg.get("sampling_temperature", 0.07)), 1e-6)
            logits = (
                selection_scores[1:pool] - selection_scores[1:pool].max()
            ) / temperature
            probabilities = np.exp(logits)
            probabilities = probabilities / probabilities.sum()
        remaining_count = min(take - 1, max(pool - 1, 0))
        if remaining_count <= 0:
            return ids[:1], scores[:1]
        chosen = 1 + rng.choice(
            pool - 1,
            size=remaining_count,
            replace=False,
            p=probabilities,
        )
        selected = np.concatenate([np.asarray([0]), np.asarray(chosen, dtype=np.int64)])
        return ids[selected], scores[selected]

    def lookup_candidate_ids(
        self,
        query_key: torch.Tensor | np.ndarray,
        query_rows: Sequence[Mapping[str, Any]],
        predicted_duration: torch.Tensor | np.ndarray | Sequence[float] | None,
        *,
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        keys = self._normalize_query_key(query_key)
        if len(keys) != len(query_rows) or keys.shape[1] != self.key_dim:
            raise ValueError(
                f"query_key shape {keys.shape} is incompatible with B={len(query_rows)}, D={self.key_dim}"
            )
        if predicted_duration is None:
            duration_values = [None] * len(query_rows)
        elif torch.is_tensor(predicted_duration):
            duration_values = predicted_duration.detach().cpu().float().view(-1).tolist()
        else:
            duration_values = np.asarray(predicted_duration, dtype=np.float32).reshape(-1).tolist()
        if len(duration_values) != len(query_rows):
            raise ValueError("predicted_duration must have one value per query")
        output_ids = np.full((len(query_rows), self.k), -1, dtype=np.int64)
        output_scores = np.zeros((len(query_rows), self.k), dtype=np.float32)
        used_table = []
        for row_index, (key, query, duration) in enumerate(
            zip(keys, query_rows, duration_values)
        ):
            candidates = self._table_candidates(query, training=training)
            from_table = candidates is not None
            if candidates is None or len(candidates[0]) < self.k:
                candidates = self._online_candidates(
                    key,
                    query,
                    training=training,
                    limit=self.top_m,
                )
                from_table = False
            ids, scores, selection_scores = self._rank_candidates(
                *candidates,
                duration,
                query,
                training,
            )
            ids, scores = self._sample_candidates(
                ids, scores, selection_scores, query, training
            )
            output_ids[row_index, : len(ids)] = ids
            output_scores[row_index, : len(scores)] = scores
            used_table.append(from_table)
        return output_ids, output_scores, used_table

    def _random_alternatives(
        self,
        query: Mapping[str, Any],
        excluded: set[int],
        count: int,
        *,
        training: bool,
        salt: str,
    ) -> list[int]:
        target = max(int(count), 0)
        if target == 0 or self.group_count <= 0:
            return []
        token = (
            f"{self.seed}|{self.epoch}|{salt}|{query.get('name')}|"
            f"{query.get('motion_path')}|{query.get('semantic_group_id')}|"
            f"{query.get('source_id')}|{query.get('source_group_id')}"
        )
        seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        start = seed % self.group_count
        if self.group_count == 1:
            step = 1
        else:
            step = 1 + ((seed >> 32) % (self.group_count - 1))
            while math.gcd(step, self.group_count) != 1:
                step += 1
                if step >= self.group_count:
                    step = 1

        selected: list[int] = []
        # A coprime stride visits every group exactly once without allocating or
        # shuffling an O(G) list. Usually only K plus a few rejected probes are
        # touched; the worst-case full pass is needed only to prove exhaustion.
        for probe in range(self.group_count):
            group_id = int((start + probe * step) % self.group_count)
            if group_id in excluded:
                continue
            if not self._group_allowed(query, group_id, training):
                continue
            selected.append(group_id)
            if len(selected) >= target:
                break
        return selected

    def _shuffle_candidate_groups(
        self,
        normal_ids: np.ndarray,
        query_keys: np.ndarray,
        query_rows: Sequence[Mapping[str, Any]],
        *,
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[list[int]]]:
        batch_size = len(query_rows)
        shuffled = np.full_like(normal_ids, -1)
        source_rows = [-1] * batch_size
        source_rows_by_rank = np.full_like(normal_ids, -1)
        for query_index, query in enumerate(query_rows):
            original = {int(value) for value in normal_ids[query_index] if int(value) >= 0}
            # The negative-control candidates depend only on stable query
            # identity, seed, and epoch—not on accidental batch partners,
            # local row position, DDP world size, or sampler partitioning.
            selected = self._random_alternatives(
                query,
                original,
                self.k,
                training=training,
                salt="shuffle",
            )
            shuffled[query_index, : len(selected)] = selected
        scores = np.zeros_like(shuffled, dtype=np.float32)
        for query_index, key in enumerate(query_keys):
            valid = shuffled[query_index] >= 0
            ids = shuffled[query_index, valid]
            if len(ids):
                scores[query_index, valid] = np.asarray(self.keys[ids], dtype=np.float32) @ key
        return shuffled, scores, source_rows, source_rows_by_rank.tolist()

    def _resolve_group_items(
        self,
        group_ids: np.ndarray,
        query_rows: Sequence[Mapping[str, Any]],
        predicted_duration: torch.Tensor | np.ndarray | Sequence[float] | None,
        *,
        training: bool,
    ) -> np.ndarray:
        if predicted_duration is None:
            durations = [None] * len(query_rows)
        elif torch.is_tensor(predicted_duration):
            durations = predicted_duration.detach().cpu().float().view(-1).tolist()
        else:
            durations = np.asarray(predicted_duration, dtype=np.float32).reshape(-1).tolist()
        item_ids = np.full_like(group_ids, -1)
        for row_index, (query, duration) in enumerate(zip(query_rows, durations)):
            for column, group_id in enumerate(group_ids[row_index]):
                if int(group_id) >= 0:
                    item_ids[row_index, column] = self._select_group_item(
                        query,
                        int(group_id),
                        duration,
                        training=training,
                    )
        return item_ids

    def _apply_candidate_dropout(
        self,
        group_ids: np.ndarray,
        item_ids: np.ndarray,
        scores: np.ndarray,
        query_rows: Sequence[Mapping[str, Any]],
        *,
        training: bool,
    ) -> list[list[int]]:
        probability = float(self.memory_cfg.get("candidate_dropout_probability", 0.10))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("candidate_dropout_probability must be in [0,1]")
        dropped: list[list[int]] = [[] for _ in query_rows]
        if not training or probability <= 0.0:
            return dropped
        for row_index, query in enumerate(query_rows):
            # Every candidate is dropped independently. It is valid for a row
            # to become all-null: the model has an explicit exact-zero fallback.
            for column in range(group_ids.shape[1]):
                if item_ids[row_index, column] < 0:
                    continue
                token = (
                    f"{self.seed}|{self.epoch}|drop|{query.get('name')}|"
                    f"{int(group_ids[row_index, column])}|{column}"
                )
                draw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
                if draw / float(16**16 - 1) < probability:
                    group_ids[row_index, column] = -1
                    item_ids[row_index, column] = -1
                    scores[row_index, column] = 0.0
                    dropped[row_index].append(column)
        return dropped

    def fetch(
        self,
        ids: np.ndarray | torch.Tensor,
        *,
        group_ids: np.ndarray | torch.Tensor | None = None,
        scores: np.ndarray | torch.Tensor | None = None,
        predicted_duration: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        available: torch.Tensor | np.ndarray | Sequence[bool] | None = None,
        device: torch.device | str = "cpu",
        provenance: dict[str, Any] | None = None,
    ) -> SentenceMemoryBatch:
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().long().numpy()
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("ids must have shape [B,K]")
        batch_size, candidate_count = ids.shape
        if group_ids is None:
            group_ids_array = np.full(ids.shape, -1, dtype=np.int64)
            valid_items = ids >= 0
            group_ids_array[valid_items] = np.asarray(
                self.item_group_ids[ids[valid_items]], dtype=np.int64
            )
        elif torch.is_tensor(group_ids):
            group_ids_array = group_ids.detach().cpu().long().numpy()
        else:
            group_ids_array = np.asarray(group_ids, dtype=np.int64)
        if group_ids_array.shape != ids.shape:
            raise ValueError("group_ids must match ids")
        if scores is None:
            scores_array = np.zeros(ids.shape, dtype=np.float32)
        elif torch.is_tensor(scores):
            scores_array = scores.detach().cpu().float().numpy()
        else:
            scores_array = np.asarray(scores, dtype=np.float32)
        if scores_array.shape != ids.shape:
            raise ValueError("scores must match ids")
        if available is None:
            requested_available = np.ones(batch_size, dtype=np.bool_)
        elif torch.is_tensor(available):
            requested_available = available.detach().cpu().bool().numpy().reshape(-1)
        else:
            requested_available = np.asarray(available, dtype=np.bool_).reshape(-1)
        if len(requested_available) != batch_size:
            raise ValueError("available must have shape [B]")
        ids = ids.copy()
        group_ids_array = group_ids_array.copy()
        scores_array = scores_array.copy()
        ids[~requested_available] = -1
        group_ids_array[~requested_available] = -1
        scores_array[~requested_available] = 0.0
        if bool(np.any((ids < -1) | (ids >= self.bank_size))):
            raise ValueError("Candidate IDs are outside the sentence-memory bank")
        if bool(np.any((group_ids_array < -1) | (group_ids_array >= self.group_count))):
            raise ValueError("Candidate group IDs are outside the sentence-memory bank")
        group_ids_array[ids < 0] = -1
        scores_array[ids < 0] = 0.0

        lengths = np.zeros(ids.shape, dtype=np.int64)
        for row in range(batch_size):
            for column in range(candidate_count):
                candidate_id = int(ids[row, column])
                if candidate_id >= 0:
                    lengths[row, column] = int(self.item_motion_lengths[candidate_id])
        max_length = max(int(lengths.max(initial=0)), 1)
        tokens = np.zeros(
            (batch_size, candidate_count, max_length, self.latent_dim), dtype=np.float16
        )
        token_mask = np.zeros((batch_size, candidate_count, max_length), dtype=np.bool_)
        token_tau = np.zeros((batch_size, candidate_count, max_length), dtype=np.float32)
        part_validity = np.zeros(
            (batch_size, candidate_count, max_length, 4), dtype=np.float32
        )
        candidate_keys = np.zeros(
            (batch_size, candidate_count, self.key_dim), dtype=np.float32
        )
        candidate_durations = np.zeros(ids.shape, dtype=np.float32)
        candidate_names: list[list[str | None]] = []
        candidate_groups: list[list[str | None]] = []
        candidate_source_ids: list[list[str | None]] = []
        candidate_source_groups: list[list[str | None]] = []
        # The same realization can be selected by multiple examples. Gather
        # each mmap span once, then fan it out into the padded batch arrays.
        item_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for candidate_id in np.unique(ids[ids >= 0]):
            candidate_id = int(candidate_id)
            start = int(self.offsets[candidate_id])
            end = int(self.offsets[candidate_id + 1])
            item_payloads[candidate_id] = (
                np.array(self.motion[start:end], dtype=np.float16, copy=True),
                np.array(self.hand_valid[start:end], dtype=np.float32, copy=True)
                / 255.0,
            )
        for row in range(batch_size):
            row_names = []
            row_groups = []
            row_source_ids = []
            row_source_groups = []
            for column in range(candidate_count):
                candidate_id = int(ids[row, column])
                length = int(lengths[row, column])
                if candidate_id < 0 or length <= 0:
                    row_names.append(None)
                    row_groups.append(None)
                    row_source_ids.append(None)
                    row_source_groups.append(None)
                    continue
                motion, hand = item_payloads[candidate_id]
                tokens[row, column, :length] = motion
                part_validity[row, column, :length, 0] = 1.0
                part_validity[row, column, :length, 1:3] = hand
                part_validity[row, column, :length, 3] = 1.0
                token_mask[row, column, :length] = True
                token_tau[row, column, :length] = (
                    np.linspace(-1.0, 1.0, length, dtype=np.float32)
                    if length > 1
                    else 0.0
                )
                group_id = int(group_ids_array[row, column])
                if group_id < 0:
                    group_id = int(self.item_group_ids[candidate_id])
                candidate_keys[row, column] = self.keys[group_id]
                candidate_durations[row, column] = self.durations[candidate_id]
                row_names.append(str(self.metadata[candidate_id].get("name", "")))
                row_groups.append(
                    str(self.metadata[candidate_id].get("semantic_group_id", ""))
                )
                row_source_ids.append(
                    str(self.metadata[candidate_id].get("source_id", ""))
                )
                row_source_groups.append(
                    str(self.metadata[candidate_id].get("source_group_id", ""))
                )
            candidate_names.append(row_names)
            candidate_groups.append(row_groups)
            candidate_source_ids.append(row_source_ids)
            candidate_source_groups.append(row_source_groups)

        candidate_mask = ids >= 0
        effective_available = requested_available & candidate_mask.any(axis=1)
        if predicted_duration is None:
            query_duration = np.ones(batch_size, dtype=np.float32)
        elif torch.is_tensor(predicted_duration):
            query_duration = predicted_duration.detach().cpu().float().view(-1).numpy()
        else:
            query_duration = np.asarray(predicted_duration, dtype=np.float32).reshape(-1)
        if len(query_duration) != batch_size:
            raise ValueError("predicted_duration must have shape [B]")
        duration_gap = np.zeros(ids.shape, dtype=np.float32)
        valid = candidate_mask
        if bool(valid.any()):
            expanded_query = np.broadcast_to(query_duration[:, None], ids.shape)
            duration_gap[valid] = np.abs(
                np.log(np.clip(expanded_query[valid], 1e-6, None))
                - np.log(np.clip(candidate_durations[valid], 1e-6, None))
            )

        target_device = torch.device(device)

        def tensor(value):
            cpu_value = torch.from_numpy(np.ascontiguousarray(value))
            if target_device.type == "cuda":
                cpu_value = cpu_value.pin_memory()
                return cpu_value.to(target_device, non_blocking=True)
            return cpu_value.to(target_device)

        provenance = dict(provenance or {})
        provenance.update(
            {
                "bank_id": self.bank_id,
                "candidate_names": candidate_names,
                "candidate_groups": candidate_groups,
                "candidate_source_ids": candidate_source_ids,
                "candidate_source_groups": candidate_source_groups,
                "candidate_group_ids": group_ids_array.tolist(),
            }
        )
        return SentenceMemoryBatch(
            tokens=tensor(tokens),
            token_mask=tensor(token_mask),
            token_tau=tensor(token_tau),
            candidate_mask=tensor(candidate_mask),
            candidate_keys=tensor(candidate_keys),
            scores=tensor(scores_array),
            durations=tensor(candidate_durations),
            duration_log_gap=tensor(duration_gap),
            part_validity=tensor(part_validity),
            ids=tensor(ids),
            available=tensor(effective_available),
            provenance=provenance,
        )

    @torch.no_grad()
    def retrieve(
        self,
        batch: Mapping[str, Any],
        query_key: torch.Tensor | np.ndarray,
        predicted_duration: torch.Tensor | np.ndarray | Sequence[float] | None,
        available: torch.Tensor | np.ndarray | Sequence[bool] | None,
        training: bool,
        device: torch.device | str,
        mode: str = "on",
    ) -> SentenceMemoryBatch:
        mode = str(mode).lower()
        aliases = {"retrieval": "on", "normal": "on", "dropout": "on"}
        mode = aliases.get(mode, mode)
        if mode not in {"off", "on", "shuffled"}:
            raise ValueError(f"Unsupported sentence-memory mode {mode!r}")
        query_rows = self._query_rows(batch)
        query_keys = self._normalize_query_key(query_key)
        if len(query_rows) != len(query_keys):
            raise ValueError("batch and query_key have different batch sizes")
        if available is None:
            availability = np.ones(len(query_rows), dtype=np.bool_)
        elif torch.is_tensor(available):
            availability = available.detach().cpu().bool().numpy().reshape(-1)
        else:
            availability = np.asarray(available, dtype=np.bool_).reshape(-1)
        if len(availability) != len(query_rows):
            raise ValueError("available must have shape [B]")
        if mode == "off":
            availability[:] = False
        if not bool(availability.any()):
            empty = np.full((len(query_rows), self.k), -1, dtype=np.int64)
            return self.fetch(
                empty,
                group_ids=empty,
                scores=np.zeros_like(empty, dtype=np.float32),
                predicted_duration=predicted_duration,
                available=availability,
                device=device,
                provenance={
                    "mode": mode,
                    "training": bool(training),
                    "query_names": [str(row.get("name", "")) for row in query_rows],
                    "candidate_source_query_rows": [-1] * len(query_rows),
                    "candidate_source_query_rows_by_rank": [
                        [-1] * self.k for _ in query_rows
                    ],
                    "used_precomputed_neighbors": [False] * len(query_rows),
                    "candidate_dropout_ranks": [[] for _ in query_rows],
                    "query_seen_text": [
                        str(row.get("semantic_group_id", ""))
                        in self._semantic_group_ids
                        for row in query_rows
                    ],
                    "candidate_exact_text": [
                        [False] * self.k for _ in query_rows
                    ],
                },
            )
        normal_groups, normal_scores, used_table = self.lookup_candidate_ids(
            query_keys,
            query_rows,
            predicted_duration,
            training=bool(training),
        )
        source_rows = list(range(len(query_rows)))
        source_rows_by_rank = [
            [row_index if int(group_id) >= 0 else -1 for group_id in normal_groups[row_index]]
            for row_index in range(len(query_rows))
        ]
        if mode == "shuffled":
            (
                selected_groups,
                selected_scores,
                source_rows,
                source_rows_by_rank,
            ) = self._shuffle_candidate_groups(
                normal_groups, query_keys, query_rows, training=bool(training)
            )
        else:
            selected_groups, selected_scores = normal_groups, normal_scores
        selected_items = self._resolve_group_items(
            selected_groups,
            query_rows,
            predicted_duration,
            training=bool(training),
        )
        dropped = self._apply_candidate_dropout(
            selected_groups,
            selected_items,
            selected_scores,
            query_rows,
            training=bool(training),
        )
        selected_groups[~availability] = -1
        selected_items[~availability] = -1
        selected_scores[~availability] = 0.0
        provenance_sources = np.asarray(source_rows_by_rank, dtype=np.int64)
        provenance_sources[selected_groups < 0] = -1
        for row_index, row_available in enumerate(availability):
            if not bool(row_available):
                source_rows[row_index] = -1
        exact_text = []
        for query, group_row in zip(query_rows, selected_groups):
            query_semantic = str(query.get("semantic_group_id", ""))
            exact_text.append(
                [
                    bool(
                        int(group_id) >= 0
                        and query_semantic
                        and query_semantic
                        == str(
                            self.groups[int(group_id)].get(
                                "semantic_group_id", ""
                            )
                        )
                    )
                    for group_id in group_row
                ]
            )
        return self.fetch(
            selected_items,
            group_ids=selected_groups,
            scores=selected_scores,
            predicted_duration=predicted_duration,
            available=availability,
            device=device,
            provenance={
                "mode": mode,
                "training": bool(training),
                "query_names": [str(row.get("name", "")) for row in query_rows],
                "candidate_source_query_rows": source_rows,
                "candidate_source_query_rows_by_rank": provenance_sources.tolist(),
                "used_precomputed_neighbors": used_table,
                "candidate_dropout_ranks": dropped,
                "query_seen_text": [
                    str(row.get("semantic_group_id", ""))
                    in self._semantic_group_ids
                    for row in query_rows
                ],
                "candidate_exact_text": exact_text,
            },
        )


def pool_validity_to_latents(
    validity: torch.Tensor,
    latent_length: int,
    downsample_factor: int,
) -> torch.Tensor:
    """Average per-frame tracking validity into the VAE token intervals."""

    validity = validity.float().flatten()
    target_frames = int(latent_length) * int(downsample_factor)
    if len(validity) < target_frames:
        validity = torch.nn.functional.pad(validity, (0, target_frames - len(validity)))
    else:
        validity = validity[:target_frames]
    return validity.view(int(latent_length), int(downsample_factor)).mean(dim=1)
