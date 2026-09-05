from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import (
    candidate_is_allowed,
    load_neighbor_table,
    read_jsonl,
    require_literal_train_split,
    row_group_id,
    row_source_id,
    sentence_text_hash,
    sentence_memory_motion_stats_paths,
    sentence_memory_preprocessing_contract,
    sha256_file,
    validate_sentence_memory_bank,
    validate_neighbor_manifest_coverage,
)
from NIAF.continuous_trajectory_field.scripts.build_sentence_motion_bank import (
    resolve_manifest_path,
    resolve_vae_checkpoint,
    text_encoder_identity_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a sentence-motion bank and its optional neighbor tables."
    )
    parser.add_argument("--bank_dir", "--bank-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--verify_hashes", "--verify-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def prepare_query_row(row, policy):
    output = dict(row)
    output["text_hash"] = sentence_text_hash(row.get("text", ""))
    output["semantic_group_id"] = output["text_hash"]
    output["source_id"] = row_source_id(row)
    output["source_group_id"] = row_group_id(row, policy.get("group_fields"))
    return output


def audit_neighbor(
    path,
    *,
    bank,
    bank_dir,
    query_manifest=None,
    expected_top_m=None,
):
    artifacts = bank["artifacts"]
    metadata = read_jsonl(bank_dir / artifacts["metadata"]["file"])
    groups = read_jsonl(bank_dir / artifacts["groups"]["file"])
    group_offsets = np.load(
        bank_dir / artifacts["group_offsets"]["file"], mmap_mode="r", allow_pickle=False
    )
    group_item_ids = np.load(
        bank_dir / artifacts["group_item_ids"]["file"], mmap_mode="r", allow_pickle=False
    )
    expected_hash = sha256_file(query_manifest) if query_manifest is not None else None
    table = load_neighbor_table(
        path,
        bank_id=bank["bank_id"],
        bank_size=len(groups),
        expected_manifest_sha256=expected_hash,
        expected_policy=bank["filter_policy"],
        expected_top_m=expected_top_m,
    )
    if query_manifest is not None:
        validate_neighbor_manifest_coverage(table, query_manifest)
    violations = []
    checked = 0
    if query_manifest is not None:
        query_rows = read_jsonl(query_manifest)
        if len(query_rows) != len(table["ids"]):
            raise RuntimeError(
                f"Neighbor table has {len(table['ids'])} rows but manifest has {len(query_rows)}"
            )
        training = table["query_split"] == "train"
        for query_index, raw_query in enumerate(query_rows):
            query = prepare_query_row(raw_query, bank["filter_policy"])
            if str(raw_query.get("name", "")) != table["query_names"][query_index]:
                violations.append(
                    f"query {query_index}: name mismatch {raw_query.get('name')!r}"
                )
            for group_id in table["ids"][query_index]:
                if int(group_id) < 0:
                    continue
                start = int(group_offsets[int(group_id)])
                end = int(group_offsets[int(group_id) + 1])
                members = group_item_ids[start:end]
                if not any(
                    candidate_is_allowed(
                        query,
                        metadata[int(item_id)],
                        training=training,
                        policy=bank["filter_policy"],
                    )
                    for item_id in members
                ):
                    violations.append(
                        f"query {query_index}: group {int(group_id)} has no eligible realization"
                    )
                checked += 1
    if violations:
        preview = "; ".join(violations[:5])
        raise RuntimeError(f"Neighbor audit found {len(violations)} violations: {preview}")
    valid_per_query = (table["ids"] >= 0).sum(axis=1)
    return {
        "path": str(path),
        "query_split": table["query_split"],
        "queries": len(table["ids"]),
        "top_m": table["top_m"],
        "checked_candidates": checked,
        "minimum_valid_candidates": int(valid_per_query.min()),
        "mean_valid_candidates": float(valid_per_query.mean()),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config is not None else None
    if cfg is not None:
        require_literal_train_split(cfg)
    configured_dir = (cfg or {}).get("sentence_memory", {}).get("bank_dir")
    bank_dir = args.bank_dir or (Path(configured_dir) if configured_dir else None)
    if bank_dir is None:
        raise ValueError("Pass --bank_dir or a config containing sentence_memory.bank_dir")
    bank_dir = Path(bank_dir)
    text_identity = text_encoder_identity_from_config(cfg) if cfg is not None else None
    expected_manifest = (
        resolve_manifest_path(cfg, cfg.get("data", {}).get("train_split", "train"))
        if cfg is not None
        else None
    )
    expected_vae = resolve_vae_checkpoint(cfg) if cfg is not None else None
    expected_preprocessing = (
        sentence_memory_preprocessing_contract(cfg) if cfg is not None else None
    )
    if cfg is not None:
        expected_mean, expected_std = sentence_memory_motion_stats_paths(cfg)
        memory_cfg = cfg.get("sentence_memory", {})
        expected_top_m = max(
            int(memory_cfg.get("top_m", 64)), int(memory_cfg.get("k", 8))
        )
    else:
        expected_mean, expected_std = None, None
        expected_top_m = None
    bank = validate_sentence_memory_bank(
        bank_dir,
        expected_bank_id=(cfg or {}).get("sentence_memory", {}).get("expected_bank_id"),
        expected_manifest_path=expected_manifest,
        text_encoder_identity=text_identity,
        expected_vae_checkpoint=expected_vae,
        expected_preprocessing=expected_preprocessing,
        expected_mean_path=expected_mean,
        expected_std_path=expected_std,
        verify_hashes=bool(args.verify_hashes),
        verify_contents=True,
    )
    artifacts = bank["artifacts"]
    offsets = np.load(
        bank_dir / artifacts["offsets"]["file"], mmap_mode="r", allow_pickle=False
    )
    group_offsets = np.load(
        bank_dir / artifacts["group_offsets"]["file"], mmap_mode="r", allow_pickle=False
    )
    latent_lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    group_sizes = np.diff(np.asarray(group_offsets, dtype=np.int64))
    summary = {
        "valid": True,
        "bank_dir": str(bank_dir),
        "bank_id": bank["bank_id"],
        "items": int(bank["source"]["row_count"]),
        "semantic_groups": int(bank["source"]["semantic_group_count"]),
        "latent_tokens": int(offsets[-1]),
        "latent_length": {
            "minimum": int(latent_lengths.min()),
            "mean": float(latent_lengths.mean()),
            "maximum": int(latent_lengths.max()),
        },
        "group_size": {
            "minimum": int(group_sizes.min()),
            "mean": float(group_sizes.mean()),
            "maximum": int(group_sizes.max()),
        },
        "artifact_bytes": int(sum(int(row["bytes"]) for row in artifacts.values())),
        "neighbors": [],
    }
    splits = args.splits
    if splits is None:
        splits = []
        for path in sorted(bank_dir.glob("neighbors_*.npz")):
            splits.append(path.stem.removeprefix("neighbors_"))
    for split in dict.fromkeys(str(value) for value in splits):
        path = bank_dir / f"neighbors_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Neighbor table does not exist: {path}")
        query_manifest = resolve_manifest_path(cfg, split) if cfg is not None else None
        summary["neighbors"].append(
            audit_neighbor(
                path,
                bank=bank,
                bank_dir=bank_dir,
                query_manifest=query_manifest,
                expected_top_m=expected_top_m,
            )
        )
    payload = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp.json")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(payload)


if __name__ == "__main__":
    main()
