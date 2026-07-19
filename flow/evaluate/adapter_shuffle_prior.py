#!/usr/bin/env python
"""Evaluate adapter robustness to shuffled word-prior order."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from flow.content_style_adapter import build_adapter_from_config
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.latent_codec import LatentMotionCodec
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import (
    compact_from_rotation_representation,
    fit_length,
    normalize_rotation_rep,
    rotation_rep_stats_paths,
    smplx182_from_compact,
)


DEFAULT_CHECKPOINT = Path("experiments/flow/adapter/chatsign175_adapter_jointvae_b16_online/checkpoints/best.pt")
DEFAULT_OUT_DIR = Path("visualize/adapter_shuffle_prior")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare normal and shuffled word-prior adapter outputs.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=None)
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=None)
    parser.add_argument("--word_split", "--word-split", default="")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--stats_data_dir", "--stats-data-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", "--num-samples", type=int, default=4)
    parser.add_argument("--min_frames", "--min-frames", type=int, default=0)
    parser.add_argument("--max_frames", "--max-frames", type=int, default=0)
    parser.add_argument("--length_multiple", "--length-multiple", type=int, default=0)
    parser.add_argument("--out_dir", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--shuffle_mode", "--shuffle-mode", default="random", choices=["random", "reverse"])
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
        "max_frames": max_frames,
        "min_frames": min_frames,
        "length_multiple": length_multiple,
    }


def denormalize(norm_motion, dataset):
    return norm_motion * dataset.std + dataset.mean


def diff_array(x, order):
    out = np.asarray(x, dtype=np.float32)
    for _ in range(order):
        out = np.diff(out, axis=0)
    return out


def l1_array(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def motion_metrics(prefix, motion, gt_motion):
    metrics = {f"{prefix}_pose_l1": l1_array(motion, gt_motion)}
    for order, name in [(1, "vel"), (2, "accel"), (3, "jerk")]:
        if len(motion) > order and len(gt_motion) > order:
            metrics[f"{prefix}_{name}_l1"] = l1_array(diff_array(motion, order), diff_array(gt_motion, order))
        else:
            metrics[f"{prefix}_{name}_l1"] = float("nan")
    return metrics


def masked_l1(a, b, mask):
    mask = np.asarray(mask, dtype=bool)
    return l1_array(np.asarray(a)[mask], np.asarray(b)[mask])


def summarize(samples):
    out = {}
    numeric_keys = sorted(
        {
            key
            for item in samples
            for key, value in item.items()
            if isinstance(value, (int, float)) and key not in {"index", "length"}
        }
    )
    for key in numeric_keys:
        values = [float(item[key]) for item in samples if key in item and np.isfinite(float(item[key]))]
        if values:
            out[key] = float(np.mean(values))
    for metric in ["latent_l1", "pose_l1", "vel_l1", "accel_l1", "jerk_l1"]:
        normal_key = f"normal_adapted_{metric}"
        shuffled_key = f"shuffled_adapted_{metric}"
        if normal_key in numeric_keys and shuffled_key in numeric_keys:
            values = [
                float(item[shuffled_key]) / max(float(item[normal_key]), 1e-8)
                for item in samples
                if np.isfinite(float(item.get(shuffled_key, float("nan"))))
                and np.isfinite(float(item.get(normal_key, float("nan"))))
            ]
            if values:
                out[f"shuffled_over_normal_{metric}"] = float(np.mean(values))
    return out


def save_motion_npz(path, motion, representation, rotation_rep, item, length, source_index, **extra):
    payload = {
        "motion": motion.astype(np.float32),
        "representation": representation.astype(np.float32),
        "rotation_rep": rotation_rep,
        "smplx": smplx182_from_compact(motion),
        "name": str(item["name"]),
        "text": str(item.get("text", "")),
        "length": int(length),
        "source_index": int(source_index),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)


def ordered_matches(prior_builder, text, shuffle_mode, rng):
    tokens, matches = prior_builder.match_text(text)
    ordered = list(matches)
    shuffled = list(matches)
    if len(shuffled) > 1:
        if shuffle_mode == "reverse":
            shuffled = list(reversed(shuffled))
        elif shuffle_mode == "random":
            rng.shuffle(shuffled)
        else:
            raise ValueError(f"Unsupported shuffle_mode: {shuffle_mode}")
        if [entry["name"] for entry in shuffled] == [entry["name"] for entry in ordered]:
            shuffled = list(reversed(shuffled))
    return tokens, ordered, shuffled


def compose_from_matches(prior_builder, tokens, matches, target_len, ordered_names, mode):
    target_len = int(target_len)
    if matches:
        coarse = np.concatenate([entry["motion"] for entry in matches], axis=0)
        valid = np.ones(len(coarse), dtype=np.float32)
        coarse, _, _ = fit_length(coarse, valid, valid, target_len)
    else:
        coarse = prior_builder.target_mean.repeat(target_len, axis=0).astype(np.float32)
    coarse = coarse.astype(np.float32, copy=False)
    coarse_norm = (coarse - prior_builder.target_mean) / prior_builder.target_std
    stats = {
        "mode": mode,
        "tokens": list(tokens),
        "matched": [entry["name"] for entry in matches],
        "ordered_matched": list(ordered_names),
        "matched_count": len(matches),
        "token_count": len(tokens),
        "coverage": float(len(matches) / max(len(tokens), 1)),
    }
    return coarse_norm.astype(np.float32, copy=False), stats


def build_prior_batch(batch, prior_builder, shuffle_mode, seed, max_len, device, dtype):
    lengths = [int(length) for length in batch["length"].detach().cpu().tolist()]
    normal = np.zeros((len(lengths), int(max_len), prior_builder.dim), dtype=np.float32)
    shuffled = np.zeros_like(normal)
    normal_stats = []
    shuffled_stats = []
    for idx, (text, length) in enumerate(zip(batch["text"], lengths)):
        rng = random.Random(int(seed) + idx)
        tokens, ordered, shuffled_matches = ordered_matches(prior_builder, text, shuffle_mode, rng)
        ordered_names = [entry["name"] for entry in ordered]
        normal_item, normal_item_stats = compose_from_matches(
            prior_builder,
            tokens,
            ordered,
            length,
            ordered_names,
            "normal",
        )
        shuffled_item, shuffled_item_stats = compose_from_matches(
            prior_builder,
            tokens,
            shuffled_matches,
            length,
            ordered_names,
            "shuffled",
        )
        normal[idx, :length] = normal_item[:length]
        shuffled[idx, :length] = shuffled_item[:length]
        normal_stats.append(normal_item_stats)
        shuffled_stats.append(shuffled_item_stats)
    normal_tensor = torch.from_numpy(normal).to(device=device, dtype=dtype)
    shuffled_tensor = torch.from_numpy(shuffled).to(device=device, dtype=dtype)
    return normal_tensor, shuffled_tensor, normal_stats, shuffled_stats


@torch.no_grad()
def adapt_prior(adapter, codec, prior_raw, raw_mask, latent_mask, latent_stats):
    z_prior_raw, _ = codec.encode(prior_raw, mask=raw_mask)
    z_prior = codec.normalize_latent(z_prior_raw, latent_stats)
    z_adapt = adapter(z_prior, mask=latent_mask)["z_adapt"]
    z_adapt_raw = codec.denormalize_latent(z_adapt, latent_stats)
    x_adapt = codec.decode(
        z_adapt_raw,
        target_length=prior_raw.shape[1],
        mask=raw_mask,
        latent_mask=latent_mask,
    )
    return z_prior, z_adapt, x_adapt


@torch.no_grad()
def evaluate_batch(adapter, codec, batch, prior_builder, runtime, device, seed, shuffle_mode):
    raw_x = batch["motion"].to(device)
    raw_mask = batch["mask"].to(device)
    latent_stats = {
        "mean": torch.as_tensor(runtime["latent_stats"]["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(runtime["latent_stats"]["std"], dtype=torch.float32, device=device),
    }
    z_sent_raw, latent_mask = codec.encode(raw_x, mask=raw_mask)
    z_sent = codec.normalize_latent(z_sent_raw, latent_stats)

    normal_prior, shuffled_prior, normal_stats, shuffled_stats = build_prior_batch(
        batch,
        prior_builder,
        shuffle_mode,
        seed,
        max_len=raw_x.shape[1],
        device=device,
        dtype=raw_x.dtype,
    )
    normal_prior = normal_prior * raw_mask.to(device=device, dtype=normal_prior.dtype).unsqueeze(-1)
    shuffled_prior = shuffled_prior * raw_mask.to(device=device, dtype=shuffled_prior.dtype).unsqueeze(-1)

    z_normal, z_normal_adapt, x_normal_adapt = adapt_prior(
        adapter,
        codec,
        normal_prior,
        raw_mask,
        latent_mask,
        latent_stats,
    )
    z_shuffled, z_shuffled_adapt, x_shuffled_adapt = adapt_prior(
        adapter,
        codec,
        shuffled_prior,
        raw_mask,
        latent_mask,
        latent_stats,
    )

    return {
        "raw_x": raw_x.detach().cpu().numpy(),
        "normal_prior": normal_prior.detach().cpu().numpy(),
        "shuffled_prior": shuffled_prior.detach().cpu().numpy(),
        "x_normal_adapt": x_normal_adapt.detach().cpu().numpy(),
        "x_shuffled_adapt": x_shuffled_adapt.detach().cpu().numpy(),
        "z_sent": z_sent.detach().cpu().numpy(),
        "z_normal": z_normal.detach().cpu().numpy(),
        "z_shuffled": z_shuffled.detach().cpu().numpy(),
        "z_normal_adapt": z_normal_adapt.detach().cpu().numpy(),
        "z_shuffled_adapt": z_shuffled_adapt.detach().cpu().numpy(),
        "latent_mask": latent_mask.detach().cpu().numpy(),
        "normal_stats": normal_stats,
        "shuffled_stats": shuffled_stats,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    runtime = resolve_runtime(args, checkpoint)

    dataset = UpperSMPLXFlowDataset(
        runtime["data_dir"],
        split=args.split,
        mean_path=runtime["mean_path"],
        std_path=runtime["std_path"],
        min_frames=runtime["min_frames"],
        max_frames=runtime["max_frames"],
        length_multiple=runtime["length_multiple"],
        random_crop=False,
        rotation_rep=runtime["rotation_rep"],
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {runtime['data_dir']} split={args.split}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index {args.index} is outside split length {len(dataset)}")

    end = min(args.index + max(int(args.num_samples), 1), len(dataset))
    items = [dataset[idx] for idx in range(args.index, end)]
    batch = collate_upper_smplx(items)

    codec = LatentMotionCodec(runtime["vae_checkpoint"], device=device)
    if normalize_rotation_rep(codec.rotation_rep) != runtime["rotation_rep"]:
        raise ValueError(f"VAE rotation_rep={codec.rotation_rep} does not match {runtime['rotation_rep']}")
    adapter = build_adapter_from_config(checkpoint["model_config"]).to(device).eval()
    adapter.load_state_dict(checkpoint["model"], strict=True)
    prior_builder = WordMotionPrior(
        runtime["word_data_dir"],
        split=runtime["word_split"],
        target_mean=dataset.mean,
        target_std=dataset.std,
        rotation_rep=runtime["rotation_rep"],
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"VAE: {runtime['vae_checkpoint']}")
    print(f"Data: {runtime['data_dir']} split={args.split} samples={len(dataset)}")
    print(f"Word prior: {runtime['word_data_dir']} split={runtime['word_split']} entries={len(prior_builder.entries)}")
    print(f"Shuffle mode: {args.shuffle_mode} seed={args.seed}")
    print(f"Output: {args.out_dir}")

    result = evaluate_batch(
        adapter,
        codec,
        batch,
        prior_builder,
        runtime,
        device,
        seed=args.seed + args.index,
        shuffle_mode=args.shuffle_mode,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for local_idx, item in enumerate(items):
        length = int(item["length"])
        source_index = args.index + local_idx
        latent_valid = result["latent_mask"][local_idx].astype(bool)

        gt_rep = denormalize(result["raw_x"][local_idx, :length], dataset)
        normal_prior_rep = denormalize(result["normal_prior"][local_idx, :length], dataset)
        shuffled_prior_rep = denormalize(result["shuffled_prior"][local_idx, :length], dataset)
        normal_adapt_rep = denormalize(result["x_normal_adapt"][local_idx, :length], dataset)
        shuffled_adapt_rep = denormalize(result["x_shuffled_adapt"][local_idx, :length], dataset)

        gt_motion = compact_from_rotation_representation(gt_rep, runtime["rotation_rep"])
        normal_prior_motion = compact_from_rotation_representation(normal_prior_rep, runtime["rotation_rep"])
        shuffled_prior_motion = compact_from_rotation_representation(shuffled_prior_rep, runtime["rotation_rep"])
        normal_adapt_motion = compact_from_rotation_representation(normal_adapt_rep, runtime["rotation_rep"])
        shuffled_adapt_motion = compact_from_rotation_representation(shuffled_adapt_rep, runtime["rotation_rep"])

        gt_path = args.out_dir / f"gt_{local_idx:02d}.npz"
        normal_path = args.out_dir / f"normal_adapted_{local_idx:02d}.npz"
        shuffled_path = args.out_dir / f"shuffled_adapted_{local_idx:02d}.npz"

        save_motion_npz(gt_path, gt_motion, gt_rep, runtime["rotation_rep"], item, length, source_index)
        save_motion_npz(
            normal_path,
            normal_adapt_motion,
            normal_adapt_rep,
            runtime["rotation_rep"],
            item,
            length,
            source_index,
            coarse_motion=normal_prior_motion.astype(np.float32),
            coarse_smplx=smplx182_from_compact(normal_prior_motion),
            z_sent=result["z_sent"][local_idx].astype(np.float32),
            z_prior=result["z_normal"][local_idx].astype(np.float32),
            z_adapt=result["z_normal_adapt"][local_idx].astype(np.float32),
            latent_mask=result["latent_mask"][local_idx].astype(np.bool_),
            prior_stats=json.dumps(result["normal_stats"][local_idx]),
            checkpoint=str(args.checkpoint),
            checkpoint_epoch=int(checkpoint.get("epoch", -1)),
            checkpoint_global_step=int(checkpoint.get("global_step", -1)),
        )
        save_motion_npz(
            shuffled_path,
            shuffled_adapt_motion,
            shuffled_adapt_rep,
            runtime["rotation_rep"],
            item,
            length,
            source_index,
            coarse_motion=shuffled_prior_motion.astype(np.float32),
            coarse_smplx=smplx182_from_compact(shuffled_prior_motion),
            z_sent=result["z_sent"][local_idx].astype(np.float32),
            z_prior=result["z_shuffled"][local_idx].astype(np.float32),
            z_adapt=result["z_shuffled_adapt"][local_idx].astype(np.float32),
            latent_mask=result["latent_mask"][local_idx].astype(np.bool_),
            prior_stats=json.dumps(result["shuffled_stats"][local_idx]),
            checkpoint=str(args.checkpoint),
            checkpoint_epoch=int(checkpoint.get("epoch", -1)),
            checkpoint_global_step=int(checkpoint.get("global_step", -1)),
        )

        z_sent = result["z_sent"][local_idx, latent_valid]
        z_normal = result["z_normal"][local_idx, latent_valid]
        z_shuffled = result["z_shuffled"][local_idx, latent_valid]
        z_normal_adapt = result["z_normal_adapt"][local_idx, latent_valid]
        z_shuffled_adapt = result["z_shuffled_adapt"][local_idx, latent_valid]
        sample = {
            "index": int(source_index),
            "name": str(item["name"]),
            "text": str(item.get("text", "")),
            "length": int(length),
            "gt_path": str(gt_path),
            "normal_adapted_path": str(normal_path),
            "shuffled_adapted_path": str(shuffled_path),
            "normal_prior_stats": result["normal_stats"][local_idx],
            "shuffled_prior_stats": result["shuffled_stats"][local_idx],
            "normal_prior_latent_l1": masked_l1(z_normal, z_sent, np.ones(len(z_sent), dtype=bool)),
            "shuffled_prior_latent_l1": masked_l1(z_shuffled, z_sent, np.ones(len(z_sent), dtype=bool)),
            "normal_adapted_latent_l1": masked_l1(z_normal_adapt, z_sent, np.ones(len(z_sent), dtype=bool)),
            "shuffled_adapted_latent_l1": masked_l1(z_shuffled_adapt, z_sent, np.ones(len(z_sent), dtype=bool)),
        }
        sample.update(motion_metrics("normal_prior", normal_prior_motion, gt_motion))
        sample.update(motion_metrics("shuffled_prior", shuffled_prior_motion, gt_motion))
        sample.update(motion_metrics("normal_adapted", normal_adapt_motion, gt_motion))
        sample.update(motion_metrics("shuffled_adapted", shuffled_adapt_motion, gt_motion))
        samples.append(sample)
        print(
            f"[{local_idx:02d}] index={source_index} length={length} "
            f"normal latent={sample['normal_adapted_latent_l1']:.5f} "
            f"shuffled latent={sample['shuffled_adapted_latent_l1']:.5f} "
            f"normal pose={sample['normal_adapted_pose_l1']:.5f} "
            f"shuffled pose={sample['shuffled_adapted_pose_l1']:.5f}"
        )
        print(f"  ordered: {sample['normal_prior_stats']['matched']}")
        print(f"  shuffled: {sample['shuffled_prior_stats']['matched']}")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "vae_checkpoint": str(runtime["vae_checkpoint"]),
        "data_dir": str(runtime["data_dir"]),
        "word_data_dir": str(runtime["word_data_dir"]),
        "split": args.split,
        "index": args.index,
        "num_samples": len(samples),
        "shuffle_mode": args.shuffle_mode,
        "seed": args.seed,
        "rotation_rep": runtime["rotation_rep"],
        "samples": samples,
        "mean": summarize(samples),
    }
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
