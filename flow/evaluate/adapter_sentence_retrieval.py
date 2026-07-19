#!/usr/bin/env python
"""Sentence-level retrieval from adapter latents.

This diagnostic retrieves full sentence pose sequences from a sentence-pose
dictionary. The query is built from the lexical word prior:

    text -> word-concat pose -> VAE z_word -> adapter z_adapt

The retrieval ranking itself uses only latent distances to dictionary sentence
latents. The sentence name/text is used only for building the word prior and
for evaluating the rank of the paired sentence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.content_style_adapter import build_adapter_from_config
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.latent_codec import LatentMotionCodec
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import normalize_rotation_rep, rotation_rep_stats_paths


DEFAULT_CHECKPOINT = Path("experiments/flow/adapter/chatsign175_adapter_jointvae_b16_online/checkpoints/best.pt")
DEFAULT_OUT_DIR = Path("visualize/adapter_sentence_retrieval")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve full sentence pose sequences by comparing word-prior/adapter "
            "latents against a dictionary of sentence latents."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=None)
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=None)
    parser.add_argument("--word_split", "--word-split", default="")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--stats_data_dir",
        "--stats-data-dir",
        type=Path,
        default=None,
        help="Override dataset directory that owns normalization mean/std.",
    )
    parser.add_argument("--query_split", "--query-split", default="test")
    parser.add_argument("--dict_split", "--dict-split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_queries", "--num-queries", type=int, default=4)
    parser.add_argument("--limit_dict", "--limit-dict", type=int, default=0)
    parser.add_argument("--min_frames", "--min-frames", type=int, default=0)
    parser.add_argument("--max_frames", "--max-frames", type=int, default=0)
    parser.add_argument("--length_multiple", "--length-multiple", type=int, default=0)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out_dir", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default="l1", choices=["l1", "l2", "cosine"])
    parser.add_argument("--duration_weight", "--duration-weight", type=float, default=0.1)
    parser.add_argument("--top_k", "--top-k", type=int, default=5)
    return parser.parse_args()


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def path_from_config(value, default=None):
    if value is None or value == "":
        return default
    return Path(value)


def resolve_runtime(args, checkpoint):
    data_cfg = checkpoint.get("data_config", {})
    latent_cfg = checkpoint.get("latent_config", {})
    train_args = checkpoint.get("args", {})

    data_dir = args.data_dir or path_from_config(data_cfg.get("data_dir"))
    if data_dir is None:
        raise ValueError("Could not resolve data_dir from checkpoint; pass --data_dir.")

    word_data_dir = args.word_data_dir or path_from_config(data_cfg.get("word_data_dir"))
    if word_data_dir is None:
        raise ValueError("Could not resolve word_data_dir from checkpoint; pass --word_data_dir.")

    word_split = args.word_split or str(data_cfg.get("word_split") or train_args.get("word_split") or "train")
    rotation_rep = normalize_rotation_rep(
        data_cfg.get("rotation_rep")
        or latent_cfg.get("rotation_rep")
        or train_args.get("rotation_rep")
        or "axis_angle"
    )

    if args.stats_data_dir is not None:
        mean_path, std_path = rotation_rep_stats_paths(args.stats_data_dir, rotation_rep)
    else:
        mean_path = path_from_config(data_cfg.get("mean_path"))
        std_path = path_from_config(data_cfg.get("std_path"))
        if mean_path is None or std_path is None or not mean_path.is_file() or not std_path.is_file():
            stats_data_dir = path_from_config(data_cfg.get("stats_data_dir"), data_dir)
            mean_path, std_path = rotation_rep_stats_paths(stats_data_dir, rotation_rep)

    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {mean_path} and {std_path}")

    vae_checkpoint = args.vae_checkpoint or path_from_config(latent_cfg.get("vae_checkpoint"))
    if vae_checkpoint is None:
        raise ValueError("Could not resolve VAE checkpoint from adapter checkpoint; pass --vae_checkpoint.")

    latent_stats = latent_cfg.get("stats")
    if not latent_stats or "mean" not in latent_stats or "std" not in latent_stats:
        raise ValueError("Adapter checkpoint does not contain latent_config.stats.")

    max_frames = args.max_frames if args.max_frames > 0 else int(data_cfg.get("max_frames", 400))
    min_frames = args.min_frames if args.min_frames > 0 else int(data_cfg.get("min_frames", 40))
    length_multiple = args.length_multiple if args.length_multiple > 0 else int(data_cfg.get("length_multiple", 4))

    return {
        "data_dir": Path(data_dir),
        "word_data_dir": Path(word_data_dir),
        "word_split": word_split,
        "rotation_rep": rotation_rep,
        "mean_path": Path(mean_path),
        "std_path": Path(std_path),
        "vae_checkpoint": Path(vae_checkpoint),
        "latent_stats": latent_stats,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "length_multiple": length_multiple,
    }


def make_dataset(args, runtime, split, limit=0):
    return UpperSMPLXFlowDataset(
        runtime["data_dir"],
        split=split,
        mean_path=runtime["mean_path"],
        std_path=runtime["std_path"],
        min_frames=runtime["min_frames"],
        max_frames=runtime["max_frames"],
        length_multiple=runtime["length_multiple"],
        random_crop=False,
        limit=limit,
        rotation_rep=runtime["rotation_rep"],
    )


@torch.no_grad()
def encode_sentence_dictionary(dataset, codec, latent_stats, args, device):
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        collate_fn=collate_upper_smplx,
        pin_memory=device.type == "cuda",
    )
    entries = []
    for batch in tqdm(loader, desc="encode sentence dictionary"):
        motion = batch["motion"].to(device)
        mask = batch["mask"].to(device)
        z_raw, latent_mask = codec.encode(motion, mask=mask)
        z_sent = codec.normalize_latent(z_raw, latent_stats)
        for idx, name in enumerate(batch["name"]):
            valid = latent_mask[idx].bool()
            feature = z_sent[idx, valid].detach().cpu().float().numpy()
            if len(feature) == 0:
                continue
            entries.append(
                {
                    "dict_index": len(entries),
                    "name": str(name),
                    "text": str(batch["text"][idx]),
                    "frame_length": int(batch["length"][idx].item()),
                    "latent_length": int(valid.sum().item()),
                    "feature": feature.astype(np.float32, copy=False),
                }
            )
    if not entries:
        raise RuntimeError("No dictionary sentence features were encoded.")
    return entries


def resample_feature(feature, target_len):
    feature = np.asarray(feature, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if len(feature) == target_len:
        return feature.copy()
    if len(feature) == 1:
        return np.repeat(feature, target_len, axis=0)
    src = np.linspace(0.0, 1.0, num=len(feature), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, feature.shape[-1]), dtype=np.float32)
    for dim in range(feature.shape[-1]):
        out[:, dim] = np.interp(dst, src, feature[:, dim])
    return out


def sequence_distance(query, candidate, metric):
    candidate = resample_feature(candidate, len(query))
    query = np.asarray(query, dtype=np.float32)
    if metric == "l1":
        return float(np.mean(np.abs(query - candidate)))
    if metric == "l2":
        return float(np.sqrt(np.mean((query - candidate) ** 2)))
    if metric == "cosine":
        query_norm = query / np.linalg.norm(query, axis=-1, keepdims=True).clip(min=1e-8)
        cand_norm = candidate / np.linalg.norm(candidate, axis=-1, keepdims=True).clip(min=1e-8)
        return float(np.mean(1.0 - np.sum(query_norm * cand_norm, axis=-1)))
    raise ValueError(f"Unsupported metric: {metric}")


def retrieve_sentence(query_feature, dictionary, correct_name, args):
    query_len = int(len(query_feature))
    scored = []
    for entry in dictionary:
        cost = sequence_distance(query_feature, entry["feature"], args.metric)
        if args.duration_weight > 0:
            cand_len = float(entry["latent_length"])
            q_len = float(query_len)
            cost += float(args.duration_weight) * abs(cand_len - q_len) / max(cand_len, q_len, 1.0)
        scored.append((cost, entry))
    scored.sort(key=lambda item: item[0])

    correct_ranks = [rank for rank, (_cost, entry) in enumerate(scored, start=1) if entry["name"] == correct_name]
    rank = min(correct_ranks) if correct_ranks else None
    correct_distance = None
    if correct_ranks:
        correct_distance = float(scored[rank - 1][0])
    best_negative = next((float(cost) for cost, entry in scored if entry["name"] != correct_name), None)
    margin = None
    if correct_distance is not None and best_negative is not None:
        margin = float(best_negative - correct_distance)

    top_k = max(int(args.top_k), 1)
    top = [
        {
            "rank": rank_idx,
            "distance": float(cost),
            "name": entry["name"],
            "text": entry["text"],
            "frame_length": int(entry["frame_length"]),
            "latent_length": int(entry["latent_length"]),
        }
        for rank_idx, (cost, entry) in enumerate(scored[:top_k], start=1)
    ]
    return {
        "rank": int(rank) if rank is not None else None,
        "top1": bool(rank == 1) if rank is not None else False,
        "top5": bool(rank is not None and rank <= 5),
        "top10": bool(rank is not None and rank <= 10),
        "mrr": float(1.0 / rank) if rank is not None else 0.0,
        "correct_distance": correct_distance,
        "best_distance": float(scored[0][0]),
        "best_negative_distance": best_negative,
        "margin": margin,
        "top": top,
    }


@torch.no_grad()
def encode_query(item, query_dataset, prior_builder, adapter, codec, latent_stats, device):
    batch = collate_upper_smplx([item])
    motion = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    length = int(batch["length"][0].item())
    z_sent_raw, latent_mask = codec.encode(motion, mask=mask)
    z_sent = codec.normalize_latent(z_sent_raw, latent_stats)

    prior_raw, prior_stats = prior_builder.batch(
        batch["text"],
        [length],
        max_len=motion.shape[1],
        device=device,
        dtype=motion.dtype,
    )
    prior_raw = prior_raw * mask.to(device=device, dtype=prior_raw.dtype).unsqueeze(-1)
    z_word_raw, _ = codec.encode(prior_raw, mask=mask)
    z_word = codec.normalize_latent(z_word_raw, latent_stats)
    z_adapt = adapter(z_word, mask=latent_mask)["z_adapt"]

    valid = latent_mask[0].bool()
    return {
        "z_sent": z_sent[0, valid].detach().cpu().float().numpy(),
        "z_word": z_word[0, valid].detach().cpu().float().numpy(),
        "z_adapt": z_adapt[0, valid].detach().cpu().float().numpy(),
        "latent_length": int(valid.sum().item()),
        "prior_stats": prior_stats[0],
        "frame_length": length,
    }


def summarize(samples, modes):
    summary = {}
    for mode in modes:
        rows = [sample[mode] for sample in samples if mode in sample]
        if not rows:
            continue
        for key in ["top1", "top5", "top10"]:
            summary[f"{mode}_{key}"] = float(np.mean([1.0 if row[key] else 0.0 for row in rows]))
        summary[f"{mode}_mrr"] = float(np.mean([float(row["mrr"]) for row in rows]))
        ranks = [float(row["rank"]) for row in rows if row.get("rank") is not None]
        if ranks:
            summary[f"{mode}_mean_rank"] = float(np.mean(ranks))
            summary[f"{mode}_median_rank"] = float(np.median(ranks))
        for key in ["correct_distance", "best_distance", "best_negative_distance", "margin"]:
            values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
            if values:
                summary[f"{mode}_{key}"] = float(np.mean(values))
    if "z_word" in modes and "z_adapt" in modes:
        comparable = [
            sample
            for sample in samples
            if sample.get("z_word", {}).get("rank") is not None and sample.get("z_adapt", {}).get("rank") is not None
        ]
        if comparable:
            summary["adapt_rank_better_rate"] = float(
                np.mean([sample["z_adapt"]["rank"] < sample["z_word"]["rank"] for sample in comparable])
            )
            summary["adapt_rank_not_worse_rate"] = float(
                np.mean([sample["z_adapt"]["rank"] <= sample["z_word"]["rank"] for sample in comparable])
            )
    return summary


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    runtime = resolve_runtime(args, checkpoint)
    latent_stats = runtime["latent_stats"]

    query_dataset = make_dataset(args, runtime, args.query_split)
    dict_dataset = make_dataset(args, runtime, args.dict_split, limit=args.limit_dict)
    if len(query_dataset) == 0:
        raise RuntimeError(f"No query samples found in {runtime['data_dir']} split={args.query_split}")
    if len(dict_dataset) == 0:
        raise RuntimeError(f"No dictionary samples found in {runtime['data_dir']} split={args.dict_split}")
    if args.index < 0 or args.index >= len(query_dataset):
        raise IndexError(f"--index {args.index} is outside query split length {len(query_dataset)}")

    codec = LatentMotionCodec(runtime["vae_checkpoint"], device=device)
    if normalize_rotation_rep(codec.rotation_rep) != runtime["rotation_rep"]:
        raise ValueError(f"VAE rotation_rep={codec.rotation_rep} does not match {runtime['rotation_rep']}")

    adapter = build_adapter_from_config(checkpoint["model_config"]).to(device).eval()
    adapter.load_state_dict(checkpoint["model"], strict=True)
    prior_builder = WordMotionPrior(
        runtime["word_data_dir"],
        split=runtime["word_split"],
        target_mean=query_dataset.mean,
        target_std=query_dataset.std,
        rotation_rep=runtime["rotation_rep"],
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"VAE: {runtime['vae_checkpoint']}")
    print(f"Data: {runtime['data_dir']}")
    print(f"Query split: {args.query_split} samples={len(query_dataset)}")
    print(f"Dictionary split: {args.dict_split} samples={len(dict_dataset)}")
    print(f"Word prior: {runtime['word_data_dir']} split={runtime['word_split']} entries={len(prior_builder.entries)}")
    print(f"Metric: {args.metric} duration_weight={args.duration_weight}")

    dictionary = encode_sentence_dictionary(dict_dataset, codec, latent_stats, args, device)
    end = min(args.index + max(int(args.num_queries), 1), len(query_dataset))
    query_items = [query_dataset[idx] for idx in range(args.index, end)]
    modes = ["z_word", "z_adapt", "z_sent"]
    samples = []
    for local_idx, item in enumerate(tqdm(query_items, desc="retrieve queries")):
        source_index = args.index + local_idx
        query = encode_query(item, query_dataset, prior_builder, adapter, codec, latent_stats, device)
        sample = {
            "index": int(source_index),
            "name": str(item["name"]),
            "text": str(item.get("text", "")),
            "frame_length": int(query["frame_length"]),
            "latent_length": int(query["latent_length"]),
            "prior_stats": query["prior_stats"],
        }
        for mode in modes:
            sample[mode] = retrieve_sentence(query[mode], dictionary, sample["name"], args)
        samples.append(sample)
        print(
            f"[{source_index:03d}] {sample['name']} "
            f"rank word={sample['z_word']['rank']} adapt={sample['z_adapt']['rank']} "
            f"oracle={sample['z_sent']['rank']} "
            f"coverage={sample['prior_stats']['coverage']:.3f}"
        )
        print(
            "  top1 "
            f"word={sample['z_word']['top'][0]['name']} "
            f"adapt={sample['z_adapt']['top'][0]['name']} "
            f"oracle={sample['z_sent']['top'][0]['name']}"
        )

    metrics = {
        "checkpoint": str(args.checkpoint),
        "vae_checkpoint": str(runtime["vae_checkpoint"]),
        "data_dir": str(runtime["data_dir"]),
        "word_data_dir": str(runtime["word_data_dir"]),
        "word_split": runtime["word_split"],
        "query_split": args.query_split,
        "dict_split": args.dict_split,
        "index": int(args.index),
        "num_queries": len(samples),
        "num_dictionary": len(dictionary),
        "rotation_rep": runtime["rotation_rep"],
        "metric": args.metric,
        "duration_weight": float(args.duration_weight),
        "top_k": int(args.top_k),
        "modes": modes,
        "mean": summarize(samples, modes),
        "samples": samples,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
