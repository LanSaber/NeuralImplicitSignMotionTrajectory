from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from flow.text_encoder import FrozenT5TextEncoder
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import (
    candidate_is_allowed,
    digest_json,
    exclusion_policy,
    read_jsonl,
    require_literal_train_split,
    row_group_id,
    row_source_id,
    sentence_text_hash,
    sentence_memory_motion_stats_paths,
    sentence_memory_preprocessing_contract,
    sha256_file,
    validate_sentence_memory_bank,
    write_neighbor_table,
)
from NIAF.continuous_trajectory_field.scripts.build_sentence_motion_bank import (
    resolve_device,
    resolve_manifest_path,
    resolve_vae_checkpoint,
    text_encoder_identity_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute text-only semantic-group neighbors for sentence memory."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank_dir", "--bank-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--top_m", "--top-m", type=int, default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=128)
    parser.add_argument(
        "--key_chunk_size",
        "--key-chunk-size",
        type=int,
        default=16384,
        help="Maximum number of mmap bank keys copied to the search device at once.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Reserved compatibility flag. Positive values are rejected because "
            "runtime neighbor tables must cover the complete query manifest."
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--text_device", "--text-device", default=None, choices=("cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_query_row(row, policy):
    output = dict(row)
    output["text_hash"] = sentence_text_hash(row.get("text", ""))
    output["semantic_group_id"] = output["text_hash"]
    output["source_id"] = row_source_id(row)
    output["source_group_id"] = row_group_id(row, policy.get("group_fields"))
    return output


class GroupFilterIndex:
    """Limit expensive item-level filtering to groups sharing query provenance."""

    def __init__(self, metadata, groups, group_offsets, group_item_ids, policy):
        self.metadata = metadata
        self.groups = groups
        self.group_offsets = np.asarray(group_offsets, dtype=np.int64)
        self.group_item_ids = np.asarray(group_item_ids, dtype=np.int64)
        self.policy = policy
        self.semantic_to_group = {
            str(row.get("semantic_group_id", "")): index
            for index, row in enumerate(groups)
        }
        self.source_to_groups: dict[str, set[int]] = {}
        self.source_group_to_groups: dict[str, set[int]] = {}
        self.name_to_group: dict[str, int] = {}
        self.path_to_group: dict[str, int] = {}
        for item_index, row in enumerate(metadata):
            group_index = int(row["semantic_group_index"])
            source = str(row.get("source_id", ""))
            source_group = str(row.get("source_group_id", ""))
            if source:
                self.source_to_groups.setdefault(source, set()).add(group_index)
            if source_group:
                self.source_group_to_groups.setdefault(source_group, set()).add(group_index)
            if row.get("name"):
                self.name_to_group[str(row["name"])] = group_index
            if row.get("motion_path"):
                self.path_to_group[str(row["motion_path"])] = group_index

    def members(self, group_index):
        start = int(self.group_offsets[group_index])
        end = int(self.group_offsets[group_index + 1])
        return self.group_item_ids[start:end]

    def allowed_mask(self, query, *, training):
        allowed = np.ones(len(self.groups), dtype=np.bool_)
        semantic = str(query.get("semantic_group_id", ""))
        exclude_semantic = (
            self.policy.get("exclude_exact_text_train", True)
            if training
            else self.policy.get("exclude_exact_text_eval", False)
        )
        if exclude_semantic and semantic in self.semantic_to_group:
            allowed[self.semantic_to_group[semantic]] = False

        impacted: set[int] = set()
        source = str(query.get("source_id", ""))
        source_group = str(query.get("source_group_id", ""))
        if source:
            impacted.update(self.source_to_groups.get(source, ()))
        if source_group:
            impacted.update(self.source_group_to_groups.get(source_group, ()))
        name = str(query.get("name", ""))
        path = str(query.get("motion_path", ""))
        if name in self.name_to_group:
            impacted.add(self.name_to_group[name])
        if path in self.path_to_group:
            impacted.add(self.path_to_group[path])
        if not exclude_semantic and semantic in self.semantic_to_group:
            impacted.add(self.semantic_to_group[semantic])

        for group_index in impacted:
            if not allowed[group_index]:
                continue
            allowed[group_index] = any(
                candidate_is_allowed(
                    query,
                    self.metadata[int(item_id)],
                    training=training,
                    policy=self.policy,
                )
                for item_id in self.members(group_index)
            )
        return allowed


def _canonical_topk(
    scores: torch.Tensor,
    ids: torch.Tensor,
    *,
    take: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank by descending score with a stable ascending group-ID tie break."""

    id_order = torch.argsort(ids, dim=1, descending=False, stable=True)
    scores = torch.gather(scores, 1, id_order)
    ids = torch.gather(ids, 1, id_order)
    score_order = torch.argsort(scores, dim=1, descending=True, stable=True)
    score_order = score_order[:, :take]
    return torch.gather(scores, 1, score_order), torch.gather(ids, 1, score_order)


@torch.no_grad()
def exact_chunked_cosine_topk(
    query: torch.Tensor,
    group_keys: np.ndarray,
    allowed: np.ndarray,
    *,
    top_m: int,
    search_device: torch.device,
    key_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact top-M cosine neighbors without loading all bank keys."""

    if query.ndim != 2 or group_keys.ndim != 2:
        raise ValueError("query and group_keys must be rank-two matrices")
    if query.shape[1] != group_keys.shape[1]:
        raise ValueError("query and group_keys dimensions differ")
    allowed = np.asarray(allowed, dtype=np.bool_)
    if allowed.shape != (query.shape[0], group_keys.shape[0]):
        raise ValueError("allowed must have shape [queries, group_keys]")
    if int(top_m) < 1 or int(key_chunk_size) < 1:
        raise ValueError("top_m and key_chunk_size must be positive")

    query = F.normalize(query.float(), dim=-1).to(search_device)
    if not bool(torch.isfinite(query).all()):
        raise ValueError("query keys contain non-finite values")
    take = min(int(top_m), int(group_keys.shape[0]))
    best_scores = torch.empty(
        (query.shape[0], 0), dtype=query.dtype, device=search_device
    )
    best_ids = torch.empty(
        (query.shape[0], 0), dtype=torch.long, device=search_device
    )
    for start in range(0, len(group_keys), int(key_chunk_size)):
        end = min(start + int(key_chunk_size), len(group_keys))
        # Copy only this mmap slice so torch never receives a read-only view and
        # only O(query_batch * key_chunk_size) scores are device-resident.
        key_array = np.array(group_keys[start:end], dtype=np.float32, copy=True)
        keys = torch.from_numpy(key_array).to(search_device)
        if not bool(torch.isfinite(keys).all()):
            raise ValueError(f"group keys contain non-finite values in [{start}, {end})")
        keys = F.normalize(keys, dim=-1)
        chunk_scores = query @ keys.transpose(0, 1)
        allowed_chunk = torch.from_numpy(
            np.array(allowed[:, start:end], dtype=np.bool_, copy=True)
        ).to(search_device)
        chunk_scores = chunk_scores.masked_fill(~allowed_chunk, -torch.inf)
        chunk_ids = torch.arange(start, end, device=search_device).expand(
            query.shape[0], -1
        )
        best_scores, best_ids = _canonical_topk(
            torch.cat((best_scores, chunk_scores), dim=1),
            torch.cat((best_ids, chunk_ids), dim=1),
            take=min(take, best_scores.shape[1] + chunk_scores.shape[1]),
        )

    return (
        best_scores.detach().cpu().float().numpy(),
        best_ids.detach().cpu().long().numpy(),
    )


@torch.no_grad()
def build_split(
    *,
    cfg,
    split,
    bank_dir,
    bank,
    text_encoder,
    search_device,
    top_m,
    batch_size,
    key_chunk_size,
    limit,
    overwrite,
):
    path = bank_dir / f"neighbors_{split}.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Neighbor table already exists: {path}; pass --overwrite")
    query_manifest = resolve_manifest_path(cfg, split)
    rows = read_jsonl(query_manifest)
    if int(limit) > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError(f"No query rows in {query_manifest}")
    policy = bank["filter_policy"]
    artifacts = bank["artifacts"]
    group_keys = np.load(
        bank_dir / artifacts["group_keys"]["file"], mmap_mode="r", allow_pickle=False
    )
    metadata = read_jsonl(bank_dir / artifacts["metadata"]["file"])
    groups = read_jsonl(bank_dir / artifacts["groups"]["file"])
    group_offsets = np.load(
        bank_dir / artifacts["group_offsets"]["file"], mmap_mode="r", allow_pickle=False
    )
    group_item_ids = np.load(
        bank_dir / artifacts["group_item_ids"]["file"], mmap_mode="r", allow_pickle=False
    )
    filter_index = GroupFilterIndex(
        metadata, groups, group_offsets, group_item_ids, policy
    )
    output_ids = np.full((len(rows), top_m), -1, dtype=np.int32)
    output_scores = np.zeros((len(rows), top_m), dtype=np.float32)
    training = str(split) == "train"
    query_rows = [prepare_query_row(row, policy) for row in rows]
    for start in tqdm(range(0, len(rows), batch_size), desc=f"neighbors {split}"):
        end = min(start + batch_size, len(rows))
        texts = [str(row.get("text", "")) for row in rows[start:end]]
        query = text_encoder.encode(texts).float()
        allowed = np.stack(
            [filter_index.allowed_mask(row, training=training) for row in query_rows[start:end]],
            axis=0,
        )
        values, indices = exact_chunked_cosine_topk(
            query,
            group_keys,
            allowed,
            top_m=top_m,
            search_device=search_device,
            key_chunk_size=key_chunk_size,
        )
        take = values.shape[1]
        finite = np.isfinite(values)
        output_ids[start:end, :take] = np.where(finite, indices, -1).astype(np.int32)
        output_scores[start:end, :take] = np.where(finite, values, 0.0).astype(np.float32)

    write_neighbor_table(
        path,
        bank_id=bank["bank_id"],
        query_split=split,
        query_manifest_sha256=sha256_file(query_manifest),
        query_names=[str(row.get("name", "")) for row in rows],
        ids=output_ids,
        scores=output_scores,
        policy=policy,
        scoring={
            "metric": "cosine",
            "key_space": "semantic_groups",
            "group_count": len(groups),
            "query_limit": max(int(limit), 0),
            "tie_break": "group_id_ascending",
        },
    )
    print(
        json.dumps(
            {
                "split": split,
                "path": str(path),
                "queries": len(rows),
                "top_m": top_m,
                "training_filters": training,
                "filter_policy_digest": digest_json(policy),
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    if int(args.limit) > 0:
        raise ValueError(
            "--limit cannot produce a runtime-safe neighbor table because providers "
            "require complete manifest coverage. Use a separate subset manifest/config "
            "for a smoke artifact, or omit --limit for canonical train/val/test tables."
        )
    cfg = load_config(args.config)
    require_literal_train_split(cfg)
    memory_cfg = cfg.get("sentence_memory", {})
    bank_dir = args.bank_dir or (
        Path(memory_cfg["bank_dir"]) if memory_cfg.get("bank_dir") else None
    )
    if bank_dir is None:
        raise ValueError("Pass --bank_dir or set sentence_memory.bank_dir")
    bank_dir = Path(bank_dir)
    configured_policy = exclusion_policy(memory_cfg.get("filter"))
    mean_path, std_path = sentence_memory_motion_stats_paths(cfg)
    bank = validate_sentence_memory_bank(
        bank_dir,
        expected_bank_id=memory_cfg.get("expected_bank_id"),
        expected_manifest_path=resolve_manifest_path(
            cfg, cfg.get("data", {}).get("train_split", "train")
        ),
        text_encoder_identity=text_encoder_identity_from_config(cfg),
        expected_vae_checkpoint=resolve_vae_checkpoint(cfg),
        expected_preprocessing=sentence_memory_preprocessing_contract(cfg),
        expected_mean_path=mean_path,
        expected_std_path=std_path,
        verify_hashes=False,
        verify_contents=False,
    )
    if digest_json(bank["filter_policy"]) != digest_json(configured_policy):
        raise RuntimeError("Configured filter policy differs from sentence-memory bank")
    search_device = resolve_device(args.device)
    text_device = resolve_device(args.text_device or args.device)
    text_cfg = cfg.get("text", {})
    if str(text_cfg.get("condition_field", "text")) != "text":
        raise ValueError("Sentence neighbor retrieval requires text.condition_field=text")
    text_encoder = FrozenT5TextEncoder(
        text_cfg.get("model_path", "deps/flan-t5-base"),
        device=text_device,
        max_length=int(text_cfg.get("max_tokens", 64)),
        local_files_only=bool(text_cfg.get("local_files_only", True)),
        cache=False,
    )
    splits = args.splits or [
        cfg.get("data", {}).get("train_split", "train"),
        cfg.get("data", {}).get("val_split", "val"),
        "test",
    ]
    top_m = max(
        int(args.top_m or memory_cfg.get("top_m", 64)),
        int(memory_cfg.get("k", 8)),
        1,
    )
    for split in dict.fromkeys(str(value) for value in splits):
        build_split(
            cfg=cfg,
            split=split,
            bank_dir=bank_dir,
            bank=bank,
            text_encoder=text_encoder,
            search_device=search_device,
            top_m=top_m,
            batch_size=max(int(args.batch_size), 1),
            key_chunk_size=max(int(args.key_chunk_size), 1),
            limit=max(int(args.limit), 0),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
