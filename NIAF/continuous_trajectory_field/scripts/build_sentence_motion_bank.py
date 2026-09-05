from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.latent_codec import LatentMotionCodec
from flow.smplx_features import rotation_rep_stats_paths
from flow.text_encoder import FrozenT5TextEncoder
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import (
    BANK_SCHEMA_NAME,
    BANK_SCHEMA_VERSION,
    DEFAULT_ARTIFACT_FILES,
    artifact_record,
    complete_text_encoder_identity,
    enrich_metadata_row,
    exclusion_policy,
    normalize_sentence_text,
    pool_validity_to_latents,
    require_literal_train_split,
    sentence_memory_motion_stats_paths,
    sentence_memory_preprocessing_contract,
    sentence_text_hash,
    sha256_file,
    validate_sentence_memory_bank,
    write_bank_manifest,
    write_build_summary,
    write_ready_marker,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a train-only, mmap-friendly sentence-motion latent bank."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", "--out-dir", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--text_device", "--text-device", default="cpu", choices=("cpu", "cuda")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_only", "--verify-only", action="store_true")
    parser.add_argument("--verify_hashes", "--verify-hashes", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(value)


def resolve_manifest_path(cfg, split: str) -> Path:
    data_cfg = cfg["data"]
    explicit = data_cfg.get(f"{split}_manifest_path")
    return (
        Path(explicit)
        if explicit
        else Path(data_cfg["data_dir"]) / "meta" / f"manifest_{split}.jsonl"
    )


def resolve_vae_checkpoint(cfg, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    memory_cfg = cfg.get("sentence_memory", {})
    value = memory_cfg.get("codec_checkpoint") or cfg.get("adapter", {}).get(
        "vae_checkpoint"
    )
    if not value:
        raise ValueError(
            "Set sentence_memory.codec_checkpoint, adapter.vae_checkpoint, or pass --vae_checkpoint"
        )
    return Path(value)


def text_encoder_identity_from_config(cfg) -> dict:
    text_cfg = cfg.get("text", {})
    model_path = Path(text_cfg.get("model_path", "deps/flan-t5-base"))
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing text model config: {config_path}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = {
        "model_type": str(model_config.get("model_type", "")),
        "model_path": model_path.as_posix(),
        "text_dim": int(model_config.get("d_model")),
        "max_length": int(text_cfg.get("max_tokens", 64)),
        "encoder_loader": "T5EncoderModel",
        "tokenizer_loader": "AutoTokenizer",
        "tokenizer_use_fast": "transformers_default",
    }
    return complete_text_encoder_identity(identity)


def conditioning_texts(batch, field: str):
    field = str(field or "text")
    if field == "gloss":
        return [
            gloss if gloss else text
            for text, gloss in zip(batch["text"], batch["gloss"])
        ]
    if field == "text_gloss":
        return [
            f"{text} {gloss}".strip() if gloss else text
            for text, gloss in zip(batch["text"], batch["gloss"])
        ]
    return batch["text"]


def manifest_length(row) -> int:
    length = int(row.get("num_frames", 0) or 0)
    if length <= 0:
        fps = max(float(row.get("fps", 20.0) or 20.0), 1.0)
        length = int(round(float(row.get("duration", 0.0) or 0.0) * fps))
    if length <= 0:
        raise ValueError(f"Manifest row {row.get('name')!r} has no usable length")
    return length


def physical_duration(row, encoded_length: int) -> float:
    # Match the trajectory pipeline's duration target. The continuous dataset
    # preserves an explicit manifest duration even when its training motion is
    # deterministically fit to the configured frame bounds.
    duration = float(row.get("duration", 0.0) or 0.0)
    if duration > 0.0:
        return duration
    fps = max(float(row.get("fps", 20.0) or 20.0), 1.0)
    return float(encoded_length) / fps


def temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.building{path.suffix}")


def save_npy_atomic(path: Path, value) -> None:
    temporary = temporary_path(path)
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_source_provenance(cfg, config_path: Path, device: torch.device) -> dict:
    """Bind a bank to the exact dirty-worktree sources that produced it."""

    repository = Path(__file__).resolve().parents[3]
    source_paths = (
        Path(__file__).resolve(),
        repository / "NIAF/continuous_trajectory_field/sentence_memory.py",
        repository / "NIAF/continuous_sign_field/config.py",
        repository / "flow/latent_codec.py",
        repository / "flow/text_encoder.py",
        repository / "flow/dataset/upper_smplx.py",
        repository / "flow/smplx_features.py",
        repository / "flow/VAE/model.py",
        Path(config_path).resolve(),
    )
    files = {}
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint missing build source: {path}")
        relative = (
            path.relative_to(repository).as_posix()
            if path.is_relative_to(repository)
            else str(path)
        )
        try:
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=repository,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            tracked = True
        except (OSError, subprocess.CalledProcessError):
            tracked = False
        modified = not tracked
        if tracked:
            try:
                modified = (
                    subprocess.run(
                        ["git", "diff", "--quiet", "HEAD", "--", relative],
                        cwd=repository,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    ).returncode
                    != 0
                )
            except OSError:
                modified = True
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "git_tracked": tracked,
            "git_status": (
                "untracked"
                if not tracked
                else "tracked_modified"
                if modified
                else "tracked_clean"
            ),
        }

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    runtime = {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda)
        if torch.version.cuda is not None
        else None,
        "transformers": package_version("transformers"),
        "sentencepiece": package_version("sentencepiece"),
        "device_type": device.type,
    }
    if device.type == "cuda":
        runtime.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    resolved_config = json.dumps(
        cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "git_commit": git_commit(),
        "git_worktree_dirty": not all(
            record["git_status"] == "tracked_clean" for record in files.values()
        ),
        "resolved_config_sha256": hashlib.sha256(resolved_config).hexdigest(),
        "source_files": files,
        "runtime": runtime,
    }


def build_dataset(cfg, split: str, limit: int):
    data_cfg = cfg["data"]
    mean_path, std_path = rotation_rep_stats_paths(data_cfg["data_dir"], "rot6d")
    return UpperSMPLXFlowDataset(
        data_cfg["data_dir"],
        split=split,
        manifest_path=resolve_manifest_path(cfg, split),
        mean_path=mean_path,
        std_path=std_path,
        min_frames=int(data_cfg.get("min_frames", 40)),
        max_frames=int(data_cfg.get("max_frames", 400)),
        length_multiple=int(data_cfg.get("length_multiple", 4)),
        random_crop=False,
        limit=max(int(limit), 0),
        rotation_rep="rot6d",
    )


def verify_only(args, cfg, out_dir, split, vae_checkpoint):
    require_literal_train_split(cfg, split)
    manifest_path = resolve_manifest_path(cfg, split)
    mean_path, std_path = sentence_memory_motion_stats_paths(cfg)
    result = validate_sentence_memory_bank(
        out_dir,
        expected_bank_id=cfg.get("sentence_memory", {}).get("expected_bank_id"),
        expected_manifest_path=manifest_path,
        text_encoder_identity=text_encoder_identity_from_config(cfg),
        expected_vae_checkpoint=vae_checkpoint,
        expected_preprocessing=sentence_memory_preprocessing_contract(cfg),
        expected_mean_path=mean_path,
        expected_std_path=std_path,
        verify_hashes=bool(args.verify_hashes),
        verify_contents=True,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "bank_dir": str(out_dir),
                "bank_id": result["bank_id"],
                "row_count": result["source"]["row_count"],
            },
            indent=2,
        )
    )


@torch.no_grad()
def build(args, cfg, out_dir: Path, split: str, vae_checkpoint: Path):
    require_literal_train_split(cfg, split)
    if (out_dir / "bank.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"Sentence-memory bank already exists: {out_dir}; pass --overwrite to replace it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted overwrite must never look like a complete old build.
    if args.overwrite:
        (out_dir / "READY").unlink(missing_ok=True)
        (out_dir / "build_summary.json").unlink(missing_ok=True)
    dataset = build_dataset(cfg, split, args.limit)
    if not len(dataset):
        raise RuntimeError("Cannot build a sentence-memory bank from an empty dataset")
    device = resolve_device(args.device)
    text_device = resolve_device(args.text_device)
    source_provenance = build_source_provenance(cfg, args.config, device)
    codec = LatentMotionCodec(vae_checkpoint, device=device)
    if codec.rotation_rep != "rot6d":
        raise ValueError(
            f"Sentence-memory codec must use rot6d, got {codec.rotation_rep}"
        )
    if codec.input_dim != dataset.mean.shape[-1]:
        raise ValueError(
            f"Codec input_dim={codec.input_dim} does not match dataset dim={dataset.mean.shape[-1]}"
        )
    text_cfg = cfg.get("text", {})
    condition_field = str(text_cfg.get("condition_field", "text"))
    if condition_field != "text":
        raise ValueError(
            "Sentence semantic groups and retrieval keys require text.condition_field=text"
        )
    text_encoder = FrozenT5TextEncoder(
        text_cfg.get("model_path", "deps/flan-t5-base"),
        device=text_device,
        max_length=int(text_cfg.get("max_tokens", 64)),
        local_files_only=bool(text_cfg.get("local_files_only", True)),
        cache=False,
    )
    text_identity = complete_text_encoder_identity(text_encoder.checkpoint_identity())

    frame_lengths = [
        dataset._target_length(manifest_length(row)) for row in dataset.items
    ]
    latent_lengths = [codec.latent_length(length) for length in frame_lengths]
    if (
        not latent_lengths
        or min(latent_lengths) <= 0
        or max(latent_lengths) > np.iinfo(np.int16).max
    ):
        raise ValueError(
            "Sentence-memory latent lengths must fit positive int16 values"
        )
    offsets = np.zeros(len(dataset) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(latent_lengths, dtype=np.int64)
    total_tokens = int(offsets[-1])

    semantic_to_group: dict[str, int] = {}
    group_members: list[list[int]] = []
    group_texts: list[str] = []
    item_group_ids = np.empty(len(dataset), dtype=np.int32)
    for item_index, row in enumerate(dataset.items):
        semantic_id = sentence_text_hash(row.get("text", ""))
        group_index = semantic_to_group.get(semantic_id)
        if group_index is None:
            group_index = len(group_members)
            semantic_to_group[semantic_id] = group_index
            group_members.append([])
            group_texts.append(str(row.get("text", "")))
        group_members[group_index].append(item_index)
        item_group_ids[item_index] = group_index
    group_offsets = np.zeros(len(group_members) + 1, dtype=np.int64)
    group_offsets[1:] = np.cumsum(
        [len(members) for members in group_members], dtype=np.int64
    )
    group_item_ids = np.asarray(
        [item_id for members in group_members for item_id in members], dtype=np.int32
    )

    final_paths = {
        name: out_dir / filename for name, filename in DEFAULT_ARTIFACT_FILES.items()
    }
    temporary_paths = {
        name: temporary_path(path)
        for name, path in final_paths.items()
        if name not in {"metadata", "groups"}
    }
    motion_mu = np.lib.format.open_memmap(
        temporary_paths["motion_mu"],
        mode="w+",
        dtype=np.float16,
        shape=(total_tokens, codec.latent_dim),
    )
    hand_valid = np.lib.format.open_memmap(
        temporary_paths["hand_valid"],
        mode="w+",
        dtype=np.uint8,
        shape=(total_tokens, 2),
    )
    group_keys = np.lib.format.open_memmap(
        temporary_paths["group_keys"],
        mode="w+",
        dtype=np.float32,
        shape=(len(group_members), text_encoder.text_dim),
    )
    durations = np.lib.format.open_memmap(
        temporary_paths["durations"],
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset),),
    )
    save_npy_atomic(final_paths["offsets"], offsets)
    save_npy_atomic(
        final_paths["item_motion_lengths"], np.asarray(latent_lengths, dtype=np.int16)
    )
    save_npy_atomic(final_paths["group_offsets"], group_offsets)
    save_npy_atomic(final_paths["group_item_ids"], group_item_ids)
    save_npy_atomic(final_paths["item_group_ids"], item_group_ids)

    text_batch_size = max(int(args.batch_size), 1)
    for start in tqdm(
        range(0, len(group_texts), text_batch_size), desc="encode semantic group keys"
    ):
        end = min(start + text_batch_size, len(group_texts))
        pooled = text_encoder.encode(group_texts[start:end])
        pooled = F.normalize(pooled.float(), dim=-1).detach().cpu().numpy()
        group_keys[start:end] = pooled.astype(np.float32)

    num_workers = max(int(args.num_workers), 0)
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_upper_smplx,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    row_cursor = 0
    for batch in tqdm(loader, desc="encode sentence-motion bank"):
        motion = batch["motion"].to(device, non_blocking=device.type == "cuda")
        mask = batch["mask"].to(device, non_blocking=device.type == "cuda")
        latent, latent_mask = codec.encode(motion, mask=mask)
        for local_index in range(len(batch["name"])):
            bank_index = row_cursor + local_index
            valid = latent_mask[local_index].bool()
            feature = latent[local_index, valid].detach().cpu().float().numpy()
            expected_length = int(offsets[bank_index + 1] - offsets[bank_index])
            if len(feature) != expected_length:
                raise RuntimeError(
                    f"Latent length mismatch at row {bank_index}: {len(feature)} != {expected_length}"
                )
            start = int(offsets[bank_index])
            end = int(offsets[bank_index + 1])
            motion_mu[start:end] = feature.astype(np.float16)
            left = pool_validity_to_latents(
                batch["left_valid"][local_index, : batch["length"][local_index]],
                expected_length,
                codec.downsample_factor,
            )
            right = pool_validity_to_latents(
                batch["right_valid"][local_index, : batch["length"][local_index]],
                expected_length,
                codec.downsample_factor,
            )
            hand_valid[start:end, 0] = np.clip(
                np.rint(left.cpu().numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            hand_valid[start:end, 1] = np.clip(
                np.rint(right.cpu().numpy() * 255.0), 0, 255
            ).astype(np.uint8)
            durations[bank_index] = physical_duration(
                dataset.items[bank_index], frame_lengths[bank_index]
            )
        row_cursor += len(batch["name"])
    if row_cursor != len(dataset):
        raise RuntimeError(f"Encoded {row_cursor} rows, expected {len(dataset)}")

    for array in (motion_mu, hand_valid, group_keys, durations):
        array.flush()
    del motion_mu, hand_valid, group_keys, durations
    for name in ("motion_mu", "hand_valid", "group_keys", "durations"):
        temporary_paths[name].replace(final_paths[name])

    policy = exclusion_policy(cfg.get("sentence_memory", {}).get("filter"))
    metadata_temporary = temporary_path(final_paths["metadata"])
    with metadata_temporary.open("w", encoding="utf-8") as handle:
        for index, (row, frame_length, latent_length) in enumerate(
            zip(dataset.items, frame_lengths, latent_lengths)
        ):
            metadata = enrich_metadata_row(
                row,
                bank_index=index,
                encoded_frame_length=frame_length,
                latent_length=latent_length,
                group_fields=policy["group_fields"],
            )
            metadata["semantic_group_index"] = int(item_group_ids[index])
            metadata["duration"] = physical_duration(row, frame_length)
            handle.write(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n"
            )
    metadata_temporary.replace(final_paths["metadata"])

    groups_temporary = temporary_path(final_paths["groups"])
    with groups_temporary.open("w", encoding="utf-8") as handle:
        for group_index, (text, members) in enumerate(zip(group_texts, group_members)):
            handle.write(
                json.dumps(
                    {
                        "semantic_group_index": group_index,
                        "semantic_group_id": sentence_text_hash(text),
                        "normalized_text": normalize_sentence_text(text),
                        "representative_text": text,
                        "item_count": len(members),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    groups_temporary.replace(final_paths["groups"])

    artifacts = {
        name: artifact_record(path, include_hash=True)
        for name, path in final_paths.items()
    }
    data_cfg = cfg["data"]
    mean_path, std_path = sentence_memory_motion_stats_paths(cfg)
    manifest_path = resolve_manifest_path(cfg, split)
    codec_config = codec.checkpoint_config()
    codec_config.update(
        {
            "checkpoint_sha256": sha256_file(vae_checkpoint),
            "mean_sha256": sha256_file(mean_path),
            "std_sha256": sha256_file(std_path),
        }
    )
    manifest = {
        "schema_name": BANK_SCHEMA_NAME,
        "schema_version": BANK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": source_provenance["git_commit"],
        "build_source": source_provenance,
        "source": {
            "data_dir": str(Path(data_cfg["data_dir"]).resolve()),
            "split": "train",
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "row_count": len(dataset),
            "semantic_group_count": len(group_members),
            "limit": max(int(args.limit), 0),
        },
        "preprocessing": sentence_memory_preprocessing_contract(cfg),
        "codec": codec_config,
        "text_encoder": text_identity,
        "key_encoding": {
            "pooling": "masked_mean_last_hidden_state",
            "normalization": "l2",
            "dtype": "float32",
            "condition_field": condition_field,
        },
        "filter_policy": policy,
        "artifacts": artifacts,
    }
    write_bank_manifest(out_dir, manifest)
    persisted = json.loads((out_dir / "bank.json").read_text(encoding="utf-8"))
    artifact_bytes = sum(int(record["bytes"]) for record in artifacts.values())
    duration_values = np.asarray(
        [
            physical_duration(row, frame_length)
            for row, frame_length in zip(dataset.items, frame_lengths)
        ],
        dtype=np.float32,
    )
    validity_values = np.load(
        final_paths["hand_valid"], mmap_mode="r", allow_pickle=False
    )
    validity_float = np.asarray(validity_values, dtype=np.float32) / 255.0
    nonempty_text_count = sum(
        bool(normalize_sentence_text(row.get("text", ""))) for row in dataset.items
    )
    write_build_summary(
        out_dir,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(dataset),
            "semantic_group_count": len(group_members),
            "latent_tokens": total_tokens,
            "latent_dim": codec.latent_dim,
            "text_dim": text_encoder.text_dim,
            "artifact_bytes": artifact_bytes,
            "text_coverage": {
                "nonempty_items": int(nonempty_text_count),
                "fraction": float(nonempty_text_count / max(len(dataset), 1)),
            },
            "duration_seconds": {
                "minimum": float(duration_values.min()),
                "mean": float(duration_values.mean()),
                "maximum": float(duration_values.max()),
            },
            "hand_validity": {
                "left_mean": float(validity_float[:, 0].mean()),
                "right_mean": float(validity_float[:, 1].mean()),
                "left_nonzero_fraction": float((validity_values[:, 0] > 0).mean()),
                "right_nonzero_fraction": float((validity_values[:, 1] > 0).mean()),
            },
            "latent_length": {
                "minimum": min(latent_lengths),
                "mean": float(np.mean(latent_lengths)),
                "maximum": max(latent_lengths),
            },
            "group_size": {
                "minimum": min(len(members) for members in group_members),
                "mean": float(np.mean([len(members) for members in group_members])),
                "maximum": max(len(members) for members in group_members),
            },
        },
    )
    validated = validate_sentence_memory_bank(
        out_dir,
        expected_manifest_path=manifest_path,
        text_encoder_identity=text_identity,
        expected_vae_checkpoint=vae_checkpoint,
        expected_preprocessing=sentence_memory_preprocessing_contract(cfg),
        expected_mean_path=mean_path,
        expected_std_path=std_path,
        verify_hashes=True,
        verify_contents=True,
        require_ready=False,
    )
    if validated["bank_id"] != persisted["bank_id"]:
        raise RuntimeError("Sentence-memory bank identity changed during validation")
    write_ready_marker(out_dir)
    print(
        json.dumps(
            {
                "bank_dir": str(out_dir),
                "bank_id": validated["bank_id"],
                "rows": len(dataset),
                "latent_tokens": total_tokens,
                "latent_dim": codec.latent_dim,
                "text_dim": text_encoder.text_dim,
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    memory_cfg = cfg.get("sentence_memory", {})
    out_dir = args.out_dir or (
        Path(memory_cfg["bank_dir"]) if memory_cfg.get("bank_dir") else None
    )
    if out_dir is None:
        raise ValueError("Pass --out_dir or set sentence_memory.bank_dir")
    split = str(args.split or cfg.get("data", {}).get("train_split", "train"))
    vae_checkpoint = resolve_vae_checkpoint(cfg, args.vae_checkpoint)
    if args.verify_only:
        verify_only(args, cfg, Path(out_dir), split, vae_checkpoint)
        return
    build(args, cfg, Path(out_dir), split, vae_checkpoint)


if __name__ == "__main__":
    main()
