from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.dataset import collate_upper_smplx
from flow.latent_codec import LatentMotionCodec
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryValidationError,
    candidate_is_allowed,
    digest_json,
    load_neighbor_table,
    read_jsonl,
    require_literal_train_split,
    row_group_id,
    row_source_id,
    sentence_memory_motion_stats_paths,
    sentence_memory_preprocessing_contract,
    sentence_text_hash,
    sha256_file,
    validate_neighbor_manifest_coverage,
    validate_sentence_memory_bank,
)
from NIAF.continuous_trajectory_field.scripts.build_sentence_motion_bank import (
    build_dataset,
    physical_duration,
    resolve_device,
    resolve_manifest_path,
    resolve_vae_checkpoint,
    text_encoder_identity_from_config,
)


DIAGNOSTIC_SCHEMA_NAME = "signtrajfield_sentence_retrieval_diagnostics"
DIAGNOSTIC_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose sentence-level retrieval before training using deterministic "
            "VAE-mu motion trajectories. This is a diagnostic proxy, not a pose metric."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank_dir", "--bank-dir", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--neighbor_file", "--neighbor-file", type=Path, default=None)
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--top_k", "--top-k", type=int, default=None)
    parser.add_argument("--resample_points", "--resample-points", type=int, default=64)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--verify_hashes", "--verify-hashes", action="store_true")
    parser.add_argument("--out_json", "--out-json", type=Path, default=None)
    parser.add_argument("--details_jsonl", "--details-jsonl", type=Path, default=None)
    return parser.parse_args()


def prepare_query_row(
    row: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    output = dict(row)
    output["text_hash"] = sentence_text_hash(row.get("text", ""))
    output["semantic_group_id"] = output["text_hash"]
    output["source_id"] = row_source_id(row)
    output["source_group_id"] = row_group_id(row, policy.get("group_fields"))
    return output


def stable_query_seed(seed: int, query: Mapping[str, Any], purpose: str) -> int:
    payload = "\0".join(
        (
            str(int(seed)),
            str(purpose),
            str(query.get("name", "")),
            str(query.get("motion_path", "")),
            str(query.get("semantic_group_id", "")),
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little"
    )


def bank_latent_statistics(
    motion: np.ndarray, *, chunk_tokens: int = 65_536
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-channel statistics without materializing the mmap."""

    if motion.ndim != 2 or len(motion) <= 0:
        raise ValueError("Bank motion tokens must have non-empty shape [N,D]")
    total = np.zeros(motion.shape[1], dtype=np.float64)
    total_sq = np.zeros(motion.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(motion), max(int(chunk_tokens), 1)):
        block = np.asarray(motion[start : start + chunk_tokens], dtype=np.float64)
        if not bool(np.isfinite(block).all()):
            raise SentenceMemoryValidationError(
                "Bank motion tokens contain non-finite values"
            )
        total += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)
        count += len(block)
    mean = total / float(count)
    variance = np.maximum(total_sq / float(count) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def resample_normalized_time(sequence: np.ndarray, points: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    points = int(points)
    if sequence.ndim != 2 or len(sequence) <= 0:
        raise ValueError("A latent trajectory must have non-empty shape [T,D]")
    if points < 2:
        raise ValueError("resample_points must be at least two")
    if len(sequence) == 1:
        return np.repeat(sequence, points, axis=0)
    positions = np.linspace(0.0, float(len(sequence) - 1), points, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(sequence) - 1)
    weight = (positions - lower.astype(np.float32))[:, None]
    return sequence[lower] * (1.0 - weight) + sequence[upper] * weight


def normalized_time_latent_rmse(
    query: np.ndarray,
    candidate: np.ndarray,
    *,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    points: int = 64,
) -> float:
    """Train-bank-standardized RMSE on a shared normalized-time grid."""

    mean = np.asarray(latent_mean, dtype=np.float32).reshape(1, -1)
    std = np.maximum(np.asarray(latent_std, dtype=np.float32).reshape(1, -1), 1e-4)
    query = np.asarray(query, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if query.shape[1:] != candidate.shape[1:] or query.shape[1] != mean.shape[1]:
        raise ValueError(
            "Query, candidate, and bank statistics must share the latent dimension"
        )
    if not bool(np.isfinite(query).all() and np.isfinite(candidate).all()):
        raise ValueError("Query and candidate latent trajectories must be finite")
    query_grid = resample_normalized_time((query - mean) / std, points)
    candidate_grid = resample_normalized_time((candidate - mean) / std, points)
    return float(
        np.sqrt(np.mean(np.square(query_grid - candidate_grid), dtype=np.float64))
    )


def item_motion(item_id: int, motion: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    start = int(offsets[int(item_id)])
    end = int(offsets[int(item_id) + 1])
    if end <= start:
        raise SentenceMemoryValidationError(f"Item {item_id} has an empty motion span")
    return np.asarray(motion[start:end], dtype=np.float32)


def item_hand_validity(
    item_id: int, hand_valid: np.ndarray, offsets: np.ndarray
) -> tuple[float, float]:
    start = int(offsets[int(item_id)])
    end = int(offsets[int(item_id) + 1])
    value = np.asarray(hand_valid[start:end], dtype=np.float32) / 255.0
    if value.ndim != 2 or value.shape[1] != 2 or len(value) <= 0:
        raise SentenceMemoryValidationError(
            f"Item {item_id} has malformed hand-validity tokens"
        )
    return float(value[:, 0].mean()), float(value[:, 1].mean())


def eligible_group_items(
    query: Mapping[str, Any],
    group_id: int,
    *,
    metadata: Sequence[Mapping[str, Any]],
    group_offsets: np.ndarray,
    group_item_ids: np.ndarray,
    policy: Mapping[str, Any],
    training: bool,
) -> list[int]:
    group_id = int(group_id)
    if group_id < 0 or group_id + 1 >= len(group_offsets):
        raise SentenceMemoryValidationError(
            f"Semantic group ID is out of range: {group_id}"
        )
    start = int(group_offsets[group_id])
    end = int(group_offsets[group_id + 1])
    return [
        int(item_id)
        for item_id in group_item_ids[start:end]
        if candidate_is_allowed(
            query,
            metadata[int(item_id)],
            training=training,
            policy=policy,
        )
    ]


def select_representative_item(
    item_ids: Sequence[int], hand_valid: np.ndarray, offsets: np.ndarray
) -> int:
    """Select by joint hand coverage, then stable bank item ID.

    Unlike the runtime provider, this pre-training diagnostic has no predicted
    duration. Ground-truth query duration is therefore never used to select a
    normal top-1 or random candidate.
    """

    if not item_ids:
        raise ValueError("Cannot select a representative from an empty group")
    scored = []
    for item_id in item_ids:
        left, right = item_hand_validity(item_id, hand_valid, offsets)
        scored.append((-(left + right) / 2.0, int(item_id)))
    return min(scored)[1]


def deterministic_random_group(
    query: Mapping[str, Any],
    *,
    group_count: int,
    seed: int,
    eligible_items_for_group,
) -> tuple[int, list[int]] | None:
    """Choose the first eligible group in a deterministic affine permutation."""

    group_count = int(group_count)
    if group_count <= 0:
        return None
    rng = np.random.default_rng(stable_query_seed(seed, query, "random_group"))
    start = int(rng.integers(0, group_count))
    if group_count == 1:
        step = 1
    else:
        step = int(rng.integers(1, group_count))
        while math.gcd(step, group_count) != 1:
            step = step % group_count + 1
    for offset in range(group_count):
        group_id = (start + offset * step) % group_count
        eligible = eligible_items_for_group(group_id)
        if eligible:
            return group_id, eligible
    return None


def _duration_log_gap(query_duration: float, candidate_duration: float) -> float | None:
    if query_duration <= 0.0 or candidate_duration <= 0.0:
        return None
    return float(abs(math.log(query_duration / candidate_duration)))


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    return float(np.mean(finite, dtype=np.float64)) if finite else None


def _pairwise_cosine_distance(
    group_ids: Sequence[int], group_keys: np.ndarray
) -> float | None:
    if len(group_ids) < 2:
        return None
    keys = np.asarray(
        group_keys[np.asarray(group_ids, dtype=np.int64)], dtype=np.float32
    )
    keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    similarity = keys @ keys.T
    upper = np.triu_indices(len(keys), k=1)
    return float(np.mean(1.0 - similarity[upper], dtype=np.float64))


def validate_neighbor_ranking(table: Mapping[str, Any]) -> None:
    """Fail closed if a purported cosine top-M table is not ranked/padded."""

    ids = np.asarray(table["ids"], dtype=np.int64)
    scores = np.asarray(table["scores"], dtype=np.float32)
    if ids.ndim != 2 or scores.shape != ids.shape:
        raise SentenceMemoryValidationError("Neighbor IDs/scores must have shape [Q,M]")
    if int(table.get("top_m", -1)) != ids.shape[1]:
        raise SentenceMemoryValidationError(
            "Neighbor top_m does not match its array width"
        )
    if str((table.get("scoring") or {}).get("metric", "")) != "cosine":
        raise SentenceMemoryValidationError(
            "Direct top-1 diagnostics require a cosine neighbor table"
        )
    for row_index in range(len(ids)):
        valid = ids[row_index] >= 0
        valid_count = int(valid.sum())
        if bool(valid[valid_count:].any()):
            raise SentenceMemoryValidationError(
                f"Neighbor row {row_index} contains an ID after null padding"
            )
        row_scores = scores[row_index, :valid_count]
        if len(row_scores) > 1 and bool(np.any(np.diff(row_scores) > 1e-6)):
            raise SentenceMemoryValidationError(
                f"Neighbor row {row_index} is not descending by cosine score"
            )


def diagnose_query(
    *,
    query: Mapping[str, Any],
    query_motion: np.ndarray,
    query_duration: float,
    neighbor_ids: np.ndarray,
    neighbor_scores: np.ndarray,
    top_k: int,
    seed: int,
    training: bool,
    metadata: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    group_keys: np.ndarray,
    group_offsets: np.ndarray,
    group_item_ids: np.ndarray,
    motion: np.ndarray,
    offsets: np.ndarray,
    hand_valid: np.ndarray,
    durations: np.ndarray,
    policy: Mapping[str, Any],
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    resample_points: int,
) -> dict[str, Any]:
    query = prepare_query_row(query, policy)
    group_count = len(groups)

    def eligible(group_id: int) -> list[int]:
        return eligible_group_items(
            query,
            group_id,
            metadata=metadata,
            group_offsets=group_offsets,
            group_item_ids=group_item_ids,
            policy=policy,
            training=training,
        )

    candidates: list[dict[str, Any]] = []
    seen_group_ids: set[int] = set()
    for raw_group_id, raw_score in zip(neighbor_ids, neighbor_scores):
        group_id = int(raw_group_id)
        if group_id < 0:
            continue
        if group_id in seen_group_ids:
            raise SentenceMemoryValidationError(
                f"Query {query.get('name')!r} has duplicate semantic group {group_id}"
            )
        seen_group_ids.add(group_id)
        eligible_items = eligible(group_id)
        if not eligible_items:
            raise SentenceMemoryValidationError(
                f"Query {query.get('name')!r} has forbidden neighbor group {group_id}"
            )
        item_id = select_representative_item(eligible_items, hand_valid, offsets)
        left_valid, right_valid = item_hand_validity(item_id, hand_valid, offsets)
        candidates.append(
            {
                "group_id": group_id,
                "score": float(raw_score),
                "item_id": item_id,
                "eligible_item_ids": eligible_items,
                "left_hand_validity": left_valid,
                "right_hand_validity": right_valid,
                "duration_log_gap": _duration_log_gap(
                    query_duration, float(durations[item_id])
                ),
            }
        )
        if len(candidates) >= int(top_k):
            break

    train_semantic_ids = {str(row.get("semantic_group_id", "")) for row in groups}
    subset = (
        "exact_seen_text"
        if str(query["semantic_group_id"]) in train_semantic_ids
        else "novel_text"
    )
    output: dict[str, Any] = {
        "name": str(query.get("name", "")),
        "motion_path": str(query.get("motion_path", "")),
        "subset": subset,
        "query_duration": float(query_duration),
        "candidate_count": len(candidates),
        "top1": None,
        "deterministic_random": None,
        "oracle_top_k": None,
    }
    if not candidates:
        return output

    def attach_motion_distance(candidate: Mapping[str, Any]) -> dict[str, Any]:
        item_id = int(candidate["item_id"])
        return {
            key: value for key, value in candidate.items() if key != "eligible_item_ids"
        } | {
            "latent_ntrmse": normalized_time_latent_rmse(
                query_motion,
                item_motion(item_id, motion, offsets),
                latent_mean=latent_mean,
                latent_std=latent_std,
                points=resample_points,
            )
        }

    output["top1"] = attach_motion_distance(candidates[0])

    random_choice = deterministic_random_group(
        query,
        group_count=group_count,
        seed=seed,
        eligible_items_for_group=eligible,
    )
    if random_choice is not None:
        random_group_id, random_items = random_choice
        random_item_id = select_representative_item(random_items, hand_valid, offsets)
        left_valid, right_valid = item_hand_validity(
            random_item_id, hand_valid, offsets
        )
        output["deterministic_random"] = attach_motion_distance(
            {
                "group_id": int(random_group_id),
                "score": None,
                "item_id": int(random_item_id),
                "left_hand_validity": left_valid,
                "right_hand_validity": right_valid,
                "duration_log_gap": _duration_log_gap(
                    query_duration, float(durations[random_item_id])
                ),
            }
        )

    oracle_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        for item_id in candidate["eligible_item_ids"]:
            oracle_rows.append(
                {
                    "rank": rank,
                    "group_id": int(candidate["group_id"]),
                    "score": float(candidate["score"]),
                    "item_id": int(item_id),
                    "latent_ntrmse": normalized_time_latent_rmse(
                        query_motion,
                        item_motion(item_id, motion, offsets),
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        points=resample_points,
                    ),
                }
            )
    output["oracle_top_k"] = min(
        oracle_rows,
        key=lambda row: (row["latent_ntrmse"], row["rank"], row["item_id"]),
    )

    group_ids = [int(row["group_id"]) for row in candidates]
    source_groups = {
        str(metadata[int(row["item_id"])].get("source_group_id", ""))
        for row in candidates
        if str(metadata[int(row["item_id"])].get("source_group_id", ""))
    }
    output["candidate_set"] = {
        "mean_cosine_similarity": _mean_or_none([row["score"] for row in candidates]),
        "pairwise_key_cosine_distance": _pairwise_cosine_distance(
            group_ids, group_keys
        ),
        "unique_semantic_group_fraction": float(len(set(group_ids)) / len(group_ids)),
        "unique_source_group_fraction": float(len(source_groups) / len(group_ids)),
        "mean_abs_log_duration_gap": _mean_or_none(
            [row["duration_log_gap"] for row in candidates]
        ),
        "mean_left_hand_validity": _mean_or_none(
            [row["left_hand_validity"] for row in candidates]
        ),
        "mean_right_hand_validity": _mean_or_none(
            [row["right_hand_validity"] for row in candidates]
        ),
        "contains_exact_text": any(
            str(groups[group_id].get("semantic_group_id", ""))
            == str(query["semantic_group_id"])
            for group_id in group_ids
        ),
    }
    return output


def _numeric_summary(values: Sequence[float | None]) -> dict[str, Any]:
    values = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(value)
        ],
        dtype=np.float64,
    )
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [row for row in records if row.get("top1") is not None]

    def nested(method: str, field: str) -> list[float | None]:
        return [
            (row.get(method) or {}).get(field)
            for row in usable
            if row.get(method) is not None
        ]

    candidate_fields = (
        "mean_cosine_similarity",
        "pairwise_key_cosine_distance",
        "unique_semantic_group_fraction",
        "unique_source_group_fraction",
        "mean_abs_log_duration_gap",
        "mean_left_hand_validity",
        "mean_right_hand_validity",
    )
    top1_distance = nested("top1", "latent_ntrmse")
    random_distance = nested("deterministic_random", "latent_ntrmse")
    oracle_distance = nested("oracle_top_k", "latent_ntrmse")
    paired_top1_oracle = [
        float((row["top1"] or {})["latent_ntrmse"])
        - float((row["oracle_top_k"] or {})["latent_ntrmse"])
        for row in usable
        if row.get("top1") is not None and row.get("oracle_top_k") is not None
    ]
    paired_random_top1 = [
        float((row["deterministic_random"] or {})["latent_ntrmse"])
        - float((row["top1"] or {})["latent_ntrmse"])
        for row in usable
        if row.get("deterministic_random") is not None and row.get("top1") is not None
    ]

    def normal_candidate_summary(method: str) -> dict[str, Any]:
        result = {
            "abs_log_duration_gap": _numeric_summary(
                nested(method, "duration_log_gap")
            ),
            "left_hand_validity": _numeric_summary(
                nested(method, "left_hand_validity")
            ),
            "right_hand_validity": _numeric_summary(
                nested(method, "right_hand_validity")
            ),
        }
        if method == "top1":
            result["cosine_similarity"] = _numeric_summary(nested(method, "score"))
        return result

    return {
        "queries": int(len(records)),
        "usable_queries": int(len(usable)),
        "coverage_fraction": float(len(usable) / max(len(records), 1)),
        "candidate_count": _numeric_summary(
            [float(row.get("candidate_count", 0)) for row in records]
        ),
        "latent_motion_distance": {
            "direct_top1": _numeric_summary(top1_distance),
            "deterministic_random": _numeric_summary(random_distance),
            "diagnostic_oracle_top_k": _numeric_summary(oracle_distance),
            "paired_top1_minus_oracle": _numeric_summary(paired_top1_oracle),
            "paired_random_minus_top1": _numeric_summary(paired_random_top1),
        },
        "candidate_set": {
            field: _numeric_summary(
                [(row.get("candidate_set") or {}).get(field) for row in usable]
            )
            for field in candidate_fields
        }
        | {
            "exact_text_candidate_fraction": float(
                np.mean(
                    [
                        bool(
                            (row.get("candidate_set") or {}).get("contains_exact_text")
                        )
                        for row in usable
                    ]
                )
            )
            if usable
            else None
        },
        "top1": normal_candidate_summary("top1"),
        "deterministic_random": normal_candidate_summary("deterministic_random"),
    }


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    bank: Mapping[str, Any],
    bank_dir: Path,
    neighbor_table: Mapping[str, Any],
    neighbor_path: Path,
    query_manifest: Path,
    split: str,
    top_k: int,
    seed: int,
    resample_points: int,
    limit: int,
    vae_checkpoint: Path,
) -> dict[str, Any]:
    exact = [row for row in records if row.get("subset") == "exact_seen_text"]
    novel = [row for row in records if row.get("subset") == "novel_text"]
    return {
        "schema_name": DIAGNOSTIC_SCHEMA_NAME,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "methodology": {
            "motion_distance": "train_bank_standardized_normalized_time_vae_mu_rmse",
            "description": (
                "Each latent channel is standardized with exact statistics over all "
                "train-bank mu tokens. Query and candidate trajectories are linearly "
                "resampled to a shared normalized-time grid before RMSE."
            ),
            "resample_points": int(resample_points),
            "direct_top1": "highest-cosine eligible semantic group in the neighbor table",
            "normal_variant_selection": (
                "best mean left/right hand validity, then stable item ID; target duration "
                "and target motion are not used"
            ),
            "random": (
                "first eligible group in a seed/query-derived affine permutation of the "
                "complete train bank"
            ),
            "oracle_warning": (
                "Oracle selects the lowest-distance eligible recording among the top-K "
                "semantic groups using target motion. It is an upper-bound diagnostic "
                "only and must never be used for retrieval, training, or model selection."
            ),
            "duration_warning": (
                "Ground-truth query duration is used only to report compatibility. It is "
                "never used to rank groups or select normal candidate recordings."
            ),
        },
        "identity": {
            "bank_dir": str(bank_dir),
            "bank_id": str(bank["bank_id"]),
            "bank_source_split": str((bank.get("source") or {}).get("split", "")),
            "bank_manifest_sha256": sha256_file(bank_dir / "bank.json"),
            "neighbor_file": str(neighbor_path),
            "neighbor_sha256": sha256_file(neighbor_path),
            "neighbor_policy_digest": digest_json(neighbor_table["filter_policy"]),
            "query_manifest": str(query_manifest),
            "query_manifest_sha256": sha256_file(query_manifest),
            "vae_checkpoint": str(vae_checkpoint),
            "vae_checkpoint_sha256": sha256_file(vae_checkpoint),
        },
        "query": {
            "split": str(split),
            "limit": max(int(limit), 0),
            "evaluated": int(len(records)),
            "exact_seen_text": int(len(exact)),
            "novel_text": int(len(novel)),
            "training_exclusions": bool(str(split) == "train"),
        },
        "retrieval": {
            "top_k": int(top_k),
            "neighbor_top_m": int(neighbor_table["top_m"]),
            "seed": int(seed),
        },
        "subsets": {
            "overall": summarize_records(records),
            "novel_text": summarize_records(novel),
            "exact_seen_text": summarize_records(exact),
        },
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = load_config(args.config)
    require_literal_train_split(cfg)
    memory_cfg = dict(cfg.get("sentence_memory", {}))
    bank_dir = args.bank_dir or (
        Path(memory_cfg["bank_dir"]) if memory_cfg.get("bank_dir") else None
    )
    if bank_dir is None:
        raise ValueError("Pass --bank_dir or set sentence_memory.bank_dir")
    bank_dir = Path(bank_dir)
    split = str(args.split or cfg.get("data", {}).get("val_split", "val"))
    query_manifest = resolve_manifest_path(cfg, split)
    vae_checkpoint = resolve_vae_checkpoint(cfg, args.vae_checkpoint)
    mean_path, std_path = sentence_memory_motion_stats_paths(cfg)
    bank = validate_sentence_memory_bank(
        bank_dir,
        expected_bank_id=memory_cfg.get("expected_bank_id"),
        expected_manifest_path=resolve_manifest_path(cfg, "train"),
        text_encoder_identity=text_encoder_identity_from_config(cfg),
        expected_vae_checkpoint=vae_checkpoint,
        expected_preprocessing=sentence_memory_preprocessing_contract(cfg),
        expected_mean_path=mean_path,
        expected_std_path=std_path,
        verify_hashes=bool(args.verify_hashes),
        verify_contents=True,
    )
    source_limit = int((bank.get("source") or {}).get("limit", 0) or 0)
    if source_limit > 0 and not bool(memory_cfg.get("allow_partial_bank", False)):
        raise SentenceMemoryValidationError(
            "Retrieval diagnostics require the complete train bank; set "
            "sentence_memory.allow_partial_bank=true only for an explicit smoke test"
        )
    neighbor_path = args.neighbor_file or bank_dir / f"neighbors_{split}.npz"
    table = load_neighbor_table(
        neighbor_path,
        bank_id=bank["bank_id"],
        bank_size=int((bank.get("source") or {})["semantic_group_count"]),
        expected_manifest_sha256=sha256_file(query_manifest),
        expected_policy=bank["filter_policy"],
        expected_top_m=max(
            int(memory_cfg.get("top_m", 64)), int(memory_cfg.get("k", 8))
        ),
    )
    if str(table["query_split"]) != split:
        raise SentenceMemoryValidationError(
            f"Neighbor table split {table['query_split']!r} does not match {split!r}"
        )
    validate_neighbor_manifest_coverage(table, query_manifest)
    validate_neighbor_ranking(table)

    artifacts = bank["artifacts"]

    def load_array(name: str) -> np.ndarray:
        return np.load(
            bank_dir / artifacts[name]["file"], mmap_mode="r", allow_pickle=False
        )

    metadata = read_jsonl(bank_dir / artifacts["metadata"]["file"])
    groups = read_jsonl(bank_dir / artifacts["groups"]["file"])
    group_keys = load_array("group_keys")
    group_offsets = load_array("group_offsets")
    group_item_ids = load_array("group_item_ids")
    motion = load_array("motion_mu")
    offsets = load_array("offsets")
    hand_valid = load_array("hand_valid")
    durations = load_array("durations")
    latent_mean, latent_std = bank_latent_statistics(motion)

    top_k = max(int(args.top_k or memory_cfg.get("k", 8)), 1)
    if top_k > int(table["top_m"]):
        raise ValueError(f"top_k={top_k} exceeds neighbor table top_m={table['top_m']}")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1234))
    limit = max(int(args.limit), 0)
    dataset = build_dataset(cfg, split, limit)
    full_rows = read_jsonl(query_manifest)
    if limit > 0:
        full_rows = full_rows[:limit]
    if len(dataset) != len(full_rows):
        raise RuntimeError(
            f"Query dataset/manifest row mismatch: {len(dataset)} != {len(full_rows)}"
        )
    for index, row in enumerate(full_rows):
        if str(row.get("name", "")) != str(table["query_names"][index]):
            raise SentenceMemoryValidationError(
                f"Neighbor row {index} does not match query {row.get('name')!r}"
            )

    device = resolve_device(args.device)
    codec = LatentMotionCodec(vae_checkpoint, device=device)
    if codec.rotation_rep != "rot6d":
        raise ValueError(f"Diagnostic codec must use rot6d, got {codec.rotation_rep}")
    if int(codec.latent_dim) != int(motion.shape[1]):
        raise ValueError(
            f"Diagnostic codec latent_dim={codec.latent_dim} differs from bank {motion.shape[1]}"
        )
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        collate_fn=collate_upper_smplx,
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, Any]] = []
    row_cursor = 0
    training = split == "train"
    for batch in tqdm(loader, desc=f"diagnose sentence retrieval ({split})"):
        batch_motion = batch["motion"].to(device, non_blocking=device.type == "cuda")
        batch_mask = batch["mask"].to(device, non_blocking=device.type == "cuda")
        query_mu, query_mu_mask = codec.encode(batch_motion, mask=batch_mask)
        for local_index in range(len(batch["name"])):
            query_index = row_cursor + local_index
            row = full_rows[query_index]
            valid = query_mu_mask[local_index].bool()
            query_tokens = query_mu[local_index, valid].detach().cpu().float().numpy()
            encoded_length = int(batch["length"][local_index])
            query_duration = physical_duration(row, encoded_length)
            records.append(
                diagnose_query(
                    query=row,
                    query_motion=query_tokens,
                    query_duration=query_duration,
                    neighbor_ids=table["ids"][query_index],
                    neighbor_scores=table["scores"][query_index],
                    top_k=top_k,
                    seed=seed,
                    training=training,
                    metadata=metadata,
                    groups=groups,
                    group_keys=group_keys,
                    group_offsets=group_offsets,
                    group_item_ids=group_item_ids,
                    motion=motion,
                    offsets=offsets,
                    hand_valid=hand_valid,
                    durations=durations,
                    policy=bank["filter_policy"],
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    resample_points=max(int(args.resample_points), 2),
                )
            )
        row_cursor += len(batch["name"])
    if row_cursor != len(dataset):
        raise RuntimeError(f"Encoded {row_cursor} queries, expected {len(dataset)}")
    report = build_report(
        records,
        bank=bank,
        bank_dir=bank_dir,
        neighbor_table=table,
        neighbor_path=Path(neighbor_path),
        query_manifest=query_manifest,
        split=split,
        top_k=top_k,
        seed=seed,
        resample_points=max(int(args.resample_points), 2),
        limit=limit,
        vae_checkpoint=vae_checkpoint,
    )
    return report, records


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    report, records = run(args)
    if args.out_json is not None:
        write_json_atomic(args.out_json, report)
    if args.details_jsonl is not None:
        write_jsonl_atomic(args.details_jsonl, records)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
