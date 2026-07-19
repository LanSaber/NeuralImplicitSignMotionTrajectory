#!/usr/bin/env python
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc")
DEFAULT_VAE_CHECKPOINT = Path("experiments/flow/VAE/phoenix_rot6d_vae_jerk_b16x4_online_retry1/checkpoints/best.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a balanced word-dataset manifest by quality-gated, signer-stratified FPS pruning."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--output_suffix", default="balanced")
    parser.add_argument("--cap", type=int, default=175)
    parser.add_argument("--min_core", type=int, default=2)
    parser.add_argument("--q_drop", type=float, default=0.10)
    parser.add_argument("--conf_floor", type=float, default=0.0)
    parser.add_argument("--min_keep", type=int, default=1)
    parser.add_argument("--embedding", choices=["vae", "pose"], default="vae")
    parser.add_argument("--vae_checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument(
        "--stats_data_dir",
        type=Path,
        default=None,
        help="Stats directory for VAE normalization. Defaults to VAE checkpoint data_config.data_dir.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value, default=0.0):
    if value is None or value == "":
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def safe_int(value, default=0):
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def row_confidence(row):
    return safe_float(row.get("alignment_confidence"), 0.0)


def row_core_frames(row):
    return safe_int(row.get("core_frames"), 0)


def row_is_fallback(row):
    return bool(row.get("alignment_fallback", False) or row.get("split_method") == "even_fallback")


def row_signer(row):
    signer = str(row.get("signer") or "").strip()
    return signer if signer else "_unknown"


def row_quality_key(indexed):
    index, row = indexed
    return (
        0 if row_is_fallback(row) else 1,
        row_confidence(row),
        row_core_frames(row),
        safe_int(row.get("num_frames"), 0),
        -index,
    )


def lexicon_key(row):
    value = str(row.get("lexicon_key") or row.get("gloss") or row.get("word") or "").strip()
    return value if value else "_UNKNOWN"


def output_manifest_path(data_dir, split, suffix):
    return Path(data_dir) / "meta" / f"manifest_{split}.{suffix}.jsonl"


def distribution_stats(counts, cap=None):
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if values.size == 0:
        return {
            "total_clips": 0,
            "unique_glosses": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
            "singleton_count": 0,
            "top50_share": 0.0,
            "over_cap_count": 0,
        }
    sorted_values = sorted((int(v) for v in values), reverse=True)
    total = int(values.sum())
    return {
        "total_clips": total,
        "unique_glosses": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "singleton_count": int((values == 1).sum()),
        "top50_share": float(sum(sorted_values[:50]) / max(total, 1)),
        "over_cap_count": int(sum(1 for value in values if cap is not None and value > cap)),
    }


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return (vector / norm).astype(np.float32, copy=False)


def file_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pose_embedding_for_row(data_dir, row, mean, std):
    path = Path(data_dir) / row["motion_path"]
    with np.load(path) as data:
        motion = data["motion"].astype(np.float32)
    if motion.ndim != 2 or len(motion) == 0:
        return np.zeros(mean.shape[0] * 2, dtype=np.float32)
    motion = (motion - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-4)
    emb = np.concatenate([motion.mean(axis=0), motion.std(axis=0)], axis=0)
    return l2_normalize(emb)


def compute_pose_embeddings(data_dir, indexed_rows):
    mean = np.load(Path(data_dir) / "meta" / "mean.npy").astype(np.float32)
    std = np.load(Path(data_dir) / "meta" / "std.npy").astype(np.float32)
    embeddings = {}
    for index, row in indexed_rows:
        embeddings[index] = pose_embedding_for_row(data_dir, row, mean, std)
        if len(embeddings) % 5000 == 0:
            print(f"computed {len(embeddings)} pose embeddings...", flush=True)
    return embeddings, {
        "type": "pose",
        "dim": int(mean.shape[0] * 2),
        "stats_data_dir": str(data_dir),
    }


def resolve_device(device):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_vae_embeddings(args, indexed_rows):
    import torch

    from flow.latent_codec import LatentMotionCodec
    from flow.smplx_features import compact_to_rotation_representation, rotation_rep_stats_paths

    device = resolve_device(args.device)
    codec = LatentMotionCodec(args.vae_checkpoint, device)
    checkpoint_data_dir = codec.checkpoint.get("data_config", {}).get("data_dir")
    stats_data_dir = args.stats_data_dir or (Path(checkpoint_data_dir) if checkpoint_data_dir else args.data_dir)
    mean_path, std_path = rotation_rep_stats_paths(stats_data_dir, codec.rotation_rep)
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(
            f"Missing VAE normalization stats for rotation_rep={codec.rotation_rep}: {mean_path}, {std_path}"
        )
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    embeddings = {}
    batch_size = max(int(args.embedding_batch_size), 1)
    for start in range(0, len(indexed_rows), batch_size):
        batch = indexed_rows[start : start + batch_size]
        motions = []
        lengths = []
        for _index, row in batch:
            path = args.data_dir / row["motion_path"]
            with np.load(path) as data:
                compact = data["motion"].astype(np.float32)
            if compact.ndim != 2 or len(compact) == 0:
                compact = np.zeros((1, 133), dtype=np.float32)
            motion = compact_to_rotation_representation(compact, codec.rotation_rep)
            motion = (motion - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-4)
            motions.append(motion.astype(np.float32, copy=False))
            lengths.append(int(motion.shape[0]))

        max_len = max(lengths)
        dim = motions[0].shape[-1]
        tensor = torch.zeros(len(batch), max_len, dim, dtype=torch.float32, device=device)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)
        for item_index, motion in enumerate(motions):
            length = motion.shape[0]
            tensor[item_index, :length] = torch.from_numpy(motion).to(device=device)
            mask[item_index, :length] = True

        with torch.no_grad():
            latent, latent_mask = codec.encode(tensor, mask=mask)
            weights = latent_mask.to(device=device, dtype=latent.dtype).unsqueeze(-1)
            pooled = (latent * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1, eps=1e-6)
        pooled = pooled.detach().cpu().numpy().astype(np.float32)
        for (index, _row), emb in zip(batch, pooled):
            embeddings[index] = emb

        if len(embeddings) % 5000 < len(batch):
            print(f"computed {len(embeddings)} VAE embeddings...", flush=True)

    return embeddings, {
        "type": "vae",
        "dim": int(codec.latent_dim),
        "vae_checkpoint": str(args.vae_checkpoint),
        "vae_checkpoint_sha256": file_sha256(args.vae_checkpoint),
        "stats_data_dir": str(stats_data_dir),
        "rotation_rep": codec.rotation_rep,
        "device": str(device),
    }


def compute_embeddings(args, indexed_rows):
    if not indexed_rows:
        return {}, {"type": args.embedding, "dim": 0}
    if args.embedding == "pose":
        return compute_pose_embeddings(args.data_dir, indexed_rows)
    return compute_vae_embeddings(args, indexed_rows)


def quality_gate(indexed_rows, budget, args):
    confidences = np.asarray([row_confidence(row) for _idx, row in indexed_rows], dtype=np.float64)
    q_floor = float(args.conf_floor)
    if confidences.size > 0 and args.q_drop > 0:
        q_floor = max(q_floor, float(np.quantile(confidences, min(max(args.q_drop, 0.0), 1.0))))

    eligible = []
    rejected = []
    for indexed in indexed_rows:
        _idx, row = indexed
        passes = (
            not row_is_fallback(row)
            and row_core_frames(row) >= int(args.min_core)
            and row_confidence(row) >= q_floor
        )
        if passes:
            eligible.append(indexed)
        else:
            rejected.append(indexed)

    selected = {idx: (idx, row) for idx, row in eligible}
    relaxed_reasons = Counter()

    signers = sorted({row_signer(row) for _idx, row in indexed_rows})
    for signer in signers:
        if any(row_signer(row) == signer for _idx, row in selected.values()):
            continue
        signer_rows = [indexed for indexed in indexed_rows if row_signer(indexed[1]) == signer]
        if not signer_rows:
            continue
        best = max(signer_rows, key=row_quality_key)
        selected[best[0]] = best
        relaxed_reasons["style_coverage"] += 1

    if len(selected) < budget:
        for indexed in sorted(indexed_rows, key=row_quality_key, reverse=True):
            if indexed[0] in selected:
                continue
            selected[indexed[0]] = indexed
            relaxed_reasons["budget_fill"] += 1
            if len(selected) >= budget:
                break

    return list(selected.values()), {
        "confidence_floor": q_floor,
        "original_count": len(indexed_rows),
        "eligible_count": len(eligible),
        "candidate_count": len(selected),
        "rejected_count": len(rejected),
        "relaxed": dict(relaxed_reasons),
    }


def apportion_budget(counts, total, min_keep=1, max_counts=None):
    counts = {key: int(value) for key, value in counts.items() if int(value) > 0}
    if max_counts is None:
        max_counts = counts
    max_counts = {key: int(max_counts.get(key, 0)) for key in counts}
    total = min(int(total), sum(max_counts.values()))
    if total <= 0 or not counts:
        return {}

    base = {key: min(max_counts[key], int(min_keep)) for key in counts}
    if sum(base.values()) > total:
        ordered = sorted(counts, key=lambda key: (counts[key], key), reverse=True)
        keep = set(ordered[:total])
        return {key: (1 if key in keep else 0) for key in counts}

    raw_total = float(sum(counts.values()))
    quotas = {key: total * counts[key] / raw_total for key in counts}
    alloc = {
        key: min(max_counts[key], max(base[key], int(math.floor(quotas[key]))))
        for key in counts
    }

    while sum(alloc.values()) > total:
        choices = [key for key in counts if alloc[key] > base[key]]
        if not choices:
            break
        key = min(choices, key=lambda item: (quotas[item] - math.floor(quotas[item]), counts[item], item))
        alloc[key] -= 1

    while sum(alloc.values()) < total:
        choices = [key for key in counts if alloc[key] < max_counts[key]]
        if not choices:
            break
        key = max(choices, key=lambda item: (quotas[item] - math.floor(quotas[item]), counts[item], item))
        alloc[key] += 1

    return {key: value for key, value in alloc.items() if value > 0}


def farthest_point_sample(indexed_rows, budget, embeddings):
    if budget <= 0 or not indexed_rows:
        return []
    if budget >= len(indexed_rows):
        return sorted(indexed_rows, key=row_quality_key, reverse=True)

    seed_position, seed = max(enumerate(indexed_rows), key=lambda item: row_quality_key(item[1]))
    selected = [seed]
    candidate_positions = [pos for pos in range(len(indexed_rows)) if pos != seed_position]
    if not candidate_positions:
        return selected

    candidate_embs = np.stack([embeddings[indexed_rows[pos][0]] for pos in candidate_positions], axis=0)
    candidate_rows = [indexed_rows[pos] for pos in candidate_positions]
    max_cosine_to_selected = np.clip(candidate_embs @ embeddings[seed[0]], -1.0, 1.0)

    while len(selected) < budget and candidate_rows:
        best_offset = None
        best_score = None
        for offset, indexed in enumerate(candidate_rows):
            min_distance = float(1.0 - max_cosine_to_selected[offset])
            score = (min_distance,) + row_quality_key(indexed)
            if best_score is None or score > best_score:
                best_score = score
                best_offset = offset

        best_item = candidate_rows[best_offset]
        selected.append(best_item)
        new_cosine = np.clip(candidate_embs @ embeddings[best_item[0]], -1.0, 1.0)
        max_cosine_to_selected = np.maximum(max_cosine_to_selected, new_cosine)
        keep = np.ones(len(candidate_rows), dtype=bool)
        keep[best_offset] = False
        candidate_embs = candidate_embs[keep]
        max_cosine_to_selected = max_cosine_to_selected[keep]
        candidate_rows = [item for offset, item in enumerate(candidate_rows) if offset != best_offset]
    return selected


def stable_seed(base_seed, value):
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little", signed=False)
    return (int(base_seed) + offset) % (2**32)


def mean_pairwise_cosine_distance(indexed_rows, embeddings):
    if len(indexed_rows) < 2:
        return 0.0
    matrix = np.stack([embeddings[index] for index, _row in indexed_rows], axis=0).astype(np.float32)
    cosine = np.clip(matrix @ matrix.T, -1.0, 1.0)
    upper = np.triu_indices(len(indexed_rows), k=1)
    return float(np.mean(1.0 - cosine[upper]))


def diversity_against_random(key, selected, candidates, embeddings, seed):
    if not selected or len(candidates) <= len(selected):
        return {
            "selected_mean_pairwise_cosine_distance": mean_pairwise_cosine_distance(selected, embeddings),
            "random_mean_pairwise_cosine_distance": None,
            "delta_vs_random": None,
            "random_pool_count": int(len(candidates)),
            "random_subset_count": int(len(selected)),
            "random_seed": None,
        }

    rng_seed = stable_seed(seed, key)
    rng = np.random.default_rng(rng_seed)
    chosen_offsets = rng.choice(len(candidates), size=len(selected), replace=False)
    random_subset = [candidates[int(offset)] for offset in chosen_offsets]
    selected_distance = mean_pairwise_cosine_distance(selected, embeddings)
    random_distance = mean_pairwise_cosine_distance(random_subset, embeddings)
    return {
        "selected_mean_pairwise_cosine_distance": selected_distance,
        "random_mean_pairwise_cosine_distance": random_distance,
        "delta_vs_random": float(selected_distance - random_distance),
        "random_pool_count": int(len(candidates)),
        "random_subset_count": int(len(selected)),
        "random_seed": int(rng_seed),
    }


def annotate_row(row, kept_by, selection_rank, original_count, kept_count, signer_original, signer_kept, args):
    out = dict(row)
    out["pruned"] = {
        "kept_by": kept_by,
        "selection_rank": None if selection_rank is None else int(selection_rank),
        "gloss_original_count": int(original_count),
        "gloss_kept_count": int(kept_count),
        "signer_original_count": int(signer_original),
        "signer_kept_count": int(signer_kept),
        "cap": int(args.cap),
        "embedding": args.embedding,
    }
    return out


def mean_confidence(indexed_rows):
    if not indexed_rows:
        return 0.0
    return float(np.mean([row_confidence(row) for _idx, row in indexed_rows]))


def aggregate_diversity(per_gloss_summary):
    deltas = []
    selected_distances = []
    random_distances = []
    for item in per_gloss_summary.values():
        diversity = item.get("diversity")
        if not diversity:
            continue
        selected = diversity.get("selected_mean_pairwise_cosine_distance")
        random_value = diversity.get("random_mean_pairwise_cosine_distance")
        delta = diversity.get("delta_vs_random")
        if selected is not None:
            selected_distances.append(float(selected))
        if random_value is not None:
            random_distances.append(float(random_value))
        if delta is not None:
            deltas.append(float(delta))
    return {
        "capped_glosses_with_random_baseline": int(len(deltas)),
        "mean_selected_pairwise_cosine_distance": float(np.mean(selected_distances)) if selected_distances else None,
        "mean_random_pairwise_cosine_distance": float(np.mean(random_distances)) if random_distances else None,
        "mean_delta_vs_random": float(np.mean(deltas)) if deltas else None,
        "glosses_beating_random": int(sum(1 for value in deltas if value >= 0.0)),
    }


def process_split(args, split):
    manifest_path = args.data_dir / "meta" / f"manifest_{split}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    out_path = output_manifest_path(args.data_dir, split, args.output_suffix)
    if out_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite.")

    rows = read_jsonl(manifest_path)
    indexed_rows = list(enumerate(rows))
    groups = defaultdict(list)
    for indexed in indexed_rows:
        groups[lexicon_key(indexed[1])].append(indexed)

    before_counts = {key: len(value) for key, value in groups.items()}
    plans = {}
    embedding_need = {}
    per_gloss_summary = {}

    for key, group in groups.items():
        original_count = len(group)
        signer_counts = Counter(row_signer(row) for _idx, row in group)
        if original_count <= args.cap:
            per_gloss_summary[key] = {
                "original_count": original_count,
                "kept_count": original_count,
                "dropped_count": 0,
                "capped": False,
                "signer_before": dict(sorted(signer_counts.items())),
                "signer_after": dict(sorted(signer_counts.items())),
                "mean_confidence_before": mean_confidence(group),
                "mean_confidence_after": mean_confidence(group),
            }
            continue

        budget = min(int(args.cap), original_count)
        candidates, gate_info = quality_gate(group, budget, args)
        candidate_by_signer = defaultdict(list)
        for indexed in candidates:
            candidate_by_signer[row_signer(indexed[1])].append(indexed)
            embedding_need[indexed[0]] = indexed

        max_counts = {signer: len(items) for signer, items in candidate_by_signer.items()}
        quotas = apportion_budget(signer_counts, budget, min_keep=args.min_keep, max_counts=max_counts)
        plans[key] = {
            "group": group,
            "candidates": candidates,
            "candidate_by_signer": candidate_by_signer,
            "quotas": quotas,
            "gate": gate_info,
            "budget": budget,
            "signer_counts": signer_counts,
        }

    embedding_rows = [embedding_need[index] for index in sorted(embedding_need)]
    embeddings, embedding_info = compute_embeddings(args, embedding_rows)

    selected_ids = set()
    annotations = {}
    for key, group in groups.items():
        original_count = len(group)
        signer_counts = Counter(row_signer(row) for _idx, row in group)
        if key not in plans:
            signer_after = signer_counts
            for idx, row in group:
                selected_ids.add(idx)
                annotations[idx] = {
                    "kept_by": "uncapped",
                    "selection_rank": None,
                    "gloss_original_count": original_count,
                    "gloss_kept_count": original_count,
                    "signer_original": signer_counts[row_signer(row)],
                    "signer_kept": signer_after[row_signer(row)],
                }
            continue

        plan = plans[key]
        selected = []
        rank_by_id = {}
        for signer in sorted(plan["quotas"]):
            quota = plan["quotas"][signer]
            picked = farthest_point_sample(plan["candidate_by_signer"].get(signer, []), quota, embeddings)
            for item in picked:
                rank_by_id[item[0]] = len(selected) + 1
                selected.append(item)

        if len(selected) < plan["budget"]:
            already = {idx for idx, _row in selected}
            leftovers = [item for item in plan["candidates"] if item[0] not in already]
            picked = farthest_point_sample(leftovers, plan["budget"] - len(selected), embeddings)
            for item in picked:
                rank_by_id[item[0]] = len(selected) + 1
                selected.append(item)

        selected = selected[: plan["budget"]]
        signer_after = Counter(row_signer(row) for _idx, row in selected)
        for idx, row in selected:
            selected_ids.add(idx)
            annotations[idx] = {
                "kept_by": "fps",
                "selection_rank": rank_by_id.get(idx),
                "gloss_original_count": original_count,
                "gloss_kept_count": len(selected),
                "signer_original": signer_counts[row_signer(row)],
                "signer_kept": signer_after[row_signer(row)],
            }

        per_gloss_summary[key] = {
            "original_count": original_count,
            "kept_count": len(selected),
            "dropped_count": original_count - len(selected),
            "capped": True,
            "budget": plan["budget"],
            "quality_gate": plan["gate"],
            "signer_before": dict(sorted(signer_counts.items())),
            "signer_after": dict(sorted(signer_after.items())),
            "mean_confidence_before": mean_confidence(group),
            "mean_confidence_after": mean_confidence(selected),
            "diversity": diversity_against_random(
                key,
                selected,
                plan["candidates"],
                embeddings,
                args.seed,
            ),
        }

    output_rows = []
    for idx, row in indexed_rows:
        if idx not in selected_ids:
            continue
        ann = annotations[idx]
        output_rows.append(
            annotate_row(
                row,
                ann["kept_by"],
                ann["selection_rank"],
                ann["gloss_original_count"],
                ann["gloss_kept_count"],
                ann["signer_original"],
                ann["signer_kept"],
                args,
            )
        )

    before_pairs = {(lexicon_key(row), row_signer(row)) for row in rows}
    after_pairs = {(lexicon_key(row), row_signer(row)) for row in output_rows}
    missing_pairs = sorted(before_pairs - after_pairs)
    after_counts = Counter(lexicon_key(row) for row in output_rows)
    summary = {
        "split": split,
        "input_manifest": str(manifest_path),
        "output_manifest": str(out_path),
        "params": {
            "cap": int(args.cap),
            "min_core": int(args.min_core),
            "q_drop": float(args.q_drop),
            "conf_floor": float(args.conf_floor),
            "min_keep": int(args.min_keep),
            "seed": int(args.seed),
            "output_suffix": args.output_suffix,
        },
        "embedding": embedding_info,
        "before": distribution_stats(before_counts, cap=args.cap),
        "after": distribution_stats(after_counts, cap=args.cap),
        "total_dropped": int(len(rows) - len(output_rows)),
        "signer_pairs_before": int(len(before_pairs)),
        "signer_pairs_after": int(len(after_pairs)),
        "missing_signer_pairs": [{"lexicon_key": key, "signer": signer} for key, signer in missing_pairs],
        "diversity": aggregate_diversity(per_gloss_summary),
        "per_gloss": dict(sorted(per_gloss_summary.items())),
    }

    if not args.dry_run:
        write_jsonl(out_path, output_rows)
    return summary


def main():
    args = parse_args()
    np.random.seed(args.seed)
    args.data_dir = Path(args.data_dir)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Missing word dataset: {args.data_dir}")

    summaries = []
    for split in args.splits:
        print(f"Balancing split={split} cap={args.cap} embedding={args.embedding}", flush=True)
        summary = process_split(args, split)
        summaries.append(summary)
        print(
            f"{split}: {summary['before']['total_clips']} -> {summary['after']['total_clips']} clips; "
            f"top50_share {summary['before']['top50_share']:.3f} -> {summary['after']['top50_share']:.3f}; "
            f"max {summary['before']['max']} -> {summary['after']['max']}",
            flush=True,
        )

    combined = {
        "data_dir": str(args.data_dir),
        "splits": summaries,
        "dry_run": bool(args.dry_run),
    }
    summary_path = args.summary_path or args.data_dir / "meta" / "prune_word_dataset_summary.json"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
