#!/usr/bin/env python
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from flow.content_style_adapter import build_adapter_from_config
from flow.dataset import UpperSMPLXFlowDataset
from flow.evaluate_adapter import load_checkpoint, resolve_device, resolve_runtime
from flow.latent_codec import LatentMotionCodec
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import (
    compact_from_rotation_representation,
    normalize_rotation_rep,
    rotation_rep_dim,
    smplx182_from_compact,
)
from flow.temporal_word_attention import WordCandidateBuilder, build_arranger_from_config
from flow.text_encoder import FrozenT5TextEncoder
from flow.train_adapter import encode_word_candidates, encode_word_text_features


DEFAULT_CHECKPOINT = Path(
    "experiments/flow/adapter/chatsign175_soft_arranger_adapter_b16_online/checkpoints/best.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark adapter pose-generation latency for one input sentence. "
            "Rendering and optional file writing are excluded from timed inference."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--text", default="")
    parser.add_argument("--length", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=None)
    parser.add_argument("--word_split", "--word-split", default="")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--text_model_path", "--text-model-path", type=Path, default=None)
    parser.add_argument("--max_text_tokens", "--max-text-tokens", type=int, default=0)
    parser.add_argument("--stats_data_dir", "--stats-data-dir", type=Path, default=None)
    parser.add_argument("--min_frames", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--length_multiple", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--shuffle_word_candidates", action="store_true")
    parser.add_argument("--candidate_seed", type=int, default=123)
    parser.add_argument(
        "--disable_text_cache",
        action="store_true",
        help="Recompute T5 features every repeat; useful for fresh-sentence latency.",
    )
    parser.add_argument(
        "--save_npz",
        type=Path,
        default=None,
        help="Optional path for one generated SMPL-X npz. This write is not included in timed inference.",
    )
    parser.add_argument("--out_json", type=Path, default=None)
    return parser.parse_args()


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now(device):
    sync_if_needed(device)
    return time.perf_counter()


def summarize(values):
    values = [float(v) for v in values]
    if not values:
        return {}
    values_sorted = sorted(values)
    p95_idx = min(len(values_sorted) - 1, int(round(0.95 * (len(values_sorted) - 1))))
    return {
        "mean_ms": float(statistics.mean(values) * 1000.0),
        "median_ms": float(statistics.median(values) * 1000.0),
        "p95_ms": float(values_sorted[p95_idx] * 1000.0),
        "min_ms": float(min(values) * 1000.0),
        "max_ms": float(max(values) * 1000.0),
    }


def pick_text_and_length(args, runtime, dataset):
    text = str(args.text).strip()
    length = int(args.length)
    source = "cli"
    if text and length > 0:
        return text, length, source

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {runtime['data_dir']} split={args.split}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index {args.index} is outside split length {len(dataset)}")

    item = dataset[args.index]
    if not text:
        text = str(item.get("text", ""))
        source = f"{args.split}[{args.index}]"
    if length <= 0:
        length = int(item["length"])
        source = f"{source}, length from {args.split}[{args.index}]"
    return text, length, source


def build_components(args, device):
    load_t0 = time.perf_counter()
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
    text, length, source = pick_text_and_length(args, runtime, dataset)
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if length > int(runtime["max_frames"]):
        raise ValueError(
            f"Requested length={length}, but runtime max_frames={runtime['max_frames']}. "
            "Pass --max_frames if the checkpoint/VAE supports a larger length."
        )

    latent_codec = LatentMotionCodec(runtime["vae_checkpoint"], device=device)
    if normalize_rotation_rep(latent_codec.rotation_rep) != runtime["rotation_rep"]:
        raise ValueError(
            f"VAE rotation_rep={latent_codec.rotation_rep} does not match adapter checkpoint "
            f"rotation_rep={runtime['rotation_rep']}"
        )

    adapter = build_adapter_from_config(checkpoint["model_config"]).to(device).eval()
    adapter.load_state_dict(checkpoint["model"], strict=True)

    prior_builder = WordMotionPrior(
        runtime["word_data_dir"],
        split=runtime["word_split"],
        target_mean=dataset.mean,
        target_std=dataset.std,
        rotation_rep=runtime["rotation_rep"],
    )

    arranger = None
    candidate_builder = None
    text_encoder = None
    if runtime["prior_mode"] == "soft_arranger":
        if "arranger_config" not in checkpoint or "arranger_model" not in checkpoint:
            raise RuntimeError("Soft-arranger checkpoint is missing arranger_config or arranger_model.")
        if runtime["text_model_path"] is None:
            raise RuntimeError("Soft-arranger checkpoint has no text model path; pass --text_model_path.")
        arranger = build_arranger_from_config(checkpoint["arranger_config"]).to(device).eval()
        arranger.load_state_dict(checkpoint["arranger_model"], strict=True)
        candidate_builder = WordCandidateBuilder(
            prior_builder,
            num_word_candidates=runtime["num_word_candidates"],
            num_negative_candidates=runtime["num_negative_candidates"],
            candidate_selection=runtime["candidate_selection"],
            max_positive_variants_per_key=runtime["max_positive_variants_per_key"],
            seed=args.candidate_seed,
        )
        text_encoder = FrozenT5TextEncoder(
            runtime["text_model_path"],
            device=device,
            max_length=runtime["max_text_tokens"],
            cache=not args.disable_text_cache,
        )

    sync_if_needed(device)
    load_seconds = time.perf_counter() - load_t0
    return {
        "checkpoint": checkpoint,
        "runtime": runtime,
        "dataset": dataset,
        "text": text,
        "length": length,
        "source": source,
        "latent_codec": latent_codec,
        "adapter": adapter,
        "prior_builder": prior_builder,
        "arranger": arranger,
        "candidate_builder": candidate_builder,
        "text_encoder": text_encoder,
        "load_seconds": load_seconds,
    }


@torch.inference_mode()
def generate_once(components, args, device):
    runtime = components["runtime"]
    dataset = components["dataset"]
    latent_codec = components["latent_codec"]
    adapter = components["adapter"]
    prior_builder = components["prior_builder"]
    arranger = components["arranger"]
    candidate_builder = components["candidate_builder"]
    text_encoder = components["text_encoder"]
    text = components["text"]
    length = int(components["length"])

    dtype = torch.float32
    dim = rotation_rep_dim(runtime["rotation_rep"])
    frame_mask = torch.ones(1, length, dtype=torch.bool, device=device)
    latent_mask = latent_codec.latent_mask(frame_mask)
    latent_stats = {
        "mean": torch.as_tensor(runtime["latent_stats"]["mean"], dtype=dtype, device=device),
        "std": torch.as_tensor(runtime["latent_stats"]["std"], dtype=dtype, device=device),
    }

    timings = {}
    t0 = now(device)
    candidate_stats = None
    if runtime["prior_mode"] == "soft_arranger":
        candidate_batch = candidate_builder.batch(
            [text],
            device=device,
            dtype=dtype,
            shuffle=args.shuffle_word_candidates,
        )
        candidate_stats = candidate_batch.stats[0]
        t1 = now(device)

        word_latents, word_latent_mask = encode_word_candidates(
            candidate_batch,
            latent_codec,
            latent_stats,
            device,
        )
        t2 = now(device)

        sentence_text = text_encoder.encode([text]).to(device=device, dtype=dtype)
        t3 = now(device)

        word_text = encode_word_text_features(text_encoder, candidate_batch.texts, device, dtype)
        t4 = now(device)

        arranger_out = arranger(
            sentence_text,
            word_text,
            word_latents,
            word_latent_mask,
            candidate_batch.candidate_mask,
            latent_mask,
        )
        z_source = arranger_out["z_prior_aligned"]
        t5 = now(device)

        timings.update(
            {
                "candidate_build": t1 - t0,
                "word_vae_encode": t2 - t1,
                "sentence_text_encode": t3 - t2,
                "word_text_encode": t4 - t3,
                "soft_arranger": t5 - t4,
            }
        )
        adapter_start = t5
    else:
        prior_raw, prior_stats = prior_builder.batch(
            [text],
            [length],
            max_len=length,
            device=device,
            dtype=dtype,
        )
        candidate_stats = prior_stats[0]
        t1 = now(device)
        z_word_raw, _ = latent_codec.encode(prior_raw, mask=frame_mask)
        z_source = latent_codec.normalize_latent(z_word_raw, latent_stats)
        t2 = now(device)
        timings.update({"word_concat_prior": t1 - t0, "word_prior_vae_encode": t2 - t1})
        adapter_start = t2

    z_adapt = adapter(z_source, mask=latent_mask)["z_adapt"]
    t_adapter = now(device)

    z_adapt_raw = latent_codec.denormalize_latent(z_adapt, latent_stats)
    x_adapt = latent_codec.decode(
        z_adapt_raw,
        target_length=length,
        mask=frame_mask,
        latent_mask=latent_mask,
    )
    t_decode = now(device)

    representation = x_adapt[0, :length].detach().cpu().numpy()
    representation = representation * dataset.std.reshape(1, dim) + dataset.mean.reshape(1, dim)
    compact = compact_from_rotation_representation(representation, runtime["rotation_rep"])
    smplx = smplx182_from_compact(compact)
    t_post = time.perf_counter()

    timings.update(
        {
            "adapter": t_adapter - adapter_start,
            "vae_decode": t_decode - t_adapter,
            "postprocess_to_smplx": t_post - t_decode,
        }
    )
    timings["total"] = sum(timings.values())

    return {
        "timings": timings,
        "compact": compact.astype(np.float32, copy=False),
        "representation": representation.astype(np.float32, copy=False),
        "smplx": smplx.astype(np.float32, copy=False),
        "latent_frames": int(latent_mask.sum().item()),
        "candidate_stats": candidate_stats,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    components = build_components(args, device)

    first = generate_once(components, args, device)
    for _ in range(max(args.warmup, 0)):
        generate_once(components, args, device)

    runs = []
    last = first
    for _ in range(max(args.repeats, 1)):
        last = generate_once(components, args, device)
        runs.append(last["timings"])

    stage_names = sorted({name for run in runs for name in run})
    stage_summary = {name: summarize([run[name] for run in runs if name in run]) for name in stage_names}
    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "batch_size": 1,
        "prior_mode": str(components["runtime"]["prior_mode"]),
        "rotation_rep": str(components["runtime"]["rotation_rep"]),
        "text_cache": not args.disable_text_cache,
        "shuffle_word_candidates": bool(args.shuffle_word_candidates),
        "candidate_seed": int(args.candidate_seed),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "load_seconds": float(components["load_seconds"]),
        "first_inference_ms": {key: value * 1000.0 for key, value in first["timings"].items()},
        "summary": stage_summary,
        "text": components["text"],
        "length": int(components["length"]),
        "latent_frames": int(last["latent_frames"]),
        "text_source": components["source"],
        "candidate_stats": last["candidate_stats"],
        "output_shapes": {
            "representation": list(last["representation"].shape),
            "compact_smplx": list(last["compact"].shape),
            "smplx182": list(last["smplx"].shape),
        },
    }

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            motion=last["compact"],
            representation=last["representation"],
            smplx=last["smplx"],
            rotation_rep=components["runtime"]["rotation_rep"],
            text=components["text"],
            length=int(components["length"]),
            checkpoint=str(args.checkpoint),
        )
        result["save_npz"] = str(args.save_npz)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
