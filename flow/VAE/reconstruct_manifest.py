#!/usr/bin/env python
"""Save evaluator-ready VAE reconstructions for a manifest subset."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flow.VAE.reconstruct_vae import build_model, load_checkpoint, resolve_device
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.smplx_features import (
    compact_from_rotation_representation,
    normalize_rotation_rep,
    smplx182_from_compact,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a random or provided manifest subset with a trained VAE "
            "and save sample_*.npz/gt_*.npz pairs for flow.evaluate metrics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--length_multiple", type=int, default=None)
    parser.add_argument("--rotation_rep", "--rotation-rep", default=None, choices=["axis_angle", "rot6d"])
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--sample_latent", action="store_true", help="Use stochastic z instead of the latent mean.")
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
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row):
    return str(row.get("motion_path", "")), str(row.get("name", ""))


def resolve_data_config(args, ckpt):
    data_config = ckpt.get("data_config", {})
    data_dir_value = args.data_dir or data_config.get("data_dir")
    if data_dir_value is None or str(data_dir_value) == "":
        raise ValueError("--data_dir is required when the checkpoint does not store data_config.data_dir")
    data_dir = Path(data_dir_value)
    min_frames = args.min_frames if args.min_frames is not None else int(data_config.get("min_frames", 40))
    max_frames = args.max_frames if args.max_frames is not None else int(data_config.get("max_frames", 400))
    length_multiple = (
        args.length_multiple
        if args.length_multiple is not None
        else int(data_config.get("length_multiple", 4))
    )
    rotation_rep = normalize_rotation_rep(
        args.rotation_rep
        or ckpt.get("model_config", {}).get("rotation_rep")
        or data_config.get("rotation_rep")
        or ckpt.get("args", {}).get("rotation_rep")
        or "axis_angle"
    )
    return data_dir, min_frames, max_frames, length_multiple, rotation_rep


def select_rows(args, data_dir):
    source_manifest = data_dir / "meta" / f"manifest_{args.split}.jsonl"
    source_rows = read_jsonl(source_manifest)
    source_index_by_key = {row_key(row): idx for idx, row in enumerate(source_rows)}

    if args.manifest is not None:
        selected = read_jsonl(args.manifest)
        indices = [int(row.get("source_index", source_index_by_key.get(row_key(row), idx))) for idx, row in enumerate(selected)]
        manifest_name = args.manifest.name
    else:
        count = len(source_rows) if args.num_samples <= 0 else min(int(args.num_samples), len(source_rows))
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(source_rows)), count)
        selected = [source_rows[idx] for idx in indices]
        manifest_name = f"manifest_{args.split}_random{count}_seed{args.seed}.jsonl"

    return source_manifest, source_rows, selected, indices, manifest_name


def finite_frame_values(rows):
    values = []
    for row in rows:
        if "num_frames" in row:
            values.append(int(row["num_frames"]))
    return values


def save_summary(path, eval_manifest, source_manifest, source_count, selected_rows, indices, seed):
    frame_values = finite_frame_values(selected_rows)
    summary = {
        "source_manifest": str(source_manifest),
        "output_manifest": str(eval_manifest),
        "seed": int(seed),
        "source_count": int(source_count),
        "sample_count": int(len(selected_rows)),
        "indices": [int(idx) for idx in indices],
    }
    if frame_values:
        summary.update(
            {
                "min_frames": int(min(frame_values)),
                "max_frames": int(max(frame_values)),
                "mean_frames": float(sum(frame_values) / len(frame_values)),
            }
        )
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def denormalize(motion, dataset):
    return (motion * dataset.std[None] + dataset.mean[None]).astype(np.float32)


def save_motion_npz(path, motion, representation, rotation_rep, batch, local_idx, source_index, length, **extra):
    payload = {
        "motion": motion.astype(np.float32),
        "representation": representation.astype(np.float32),
        "rotation_rep": rotation_rep,
        "smplx": smplx182_from_compact(motion),
        "name": str(batch["name"][local_idx]),
        "text": str(batch["text"][local_idx]),
        "gloss": str(batch.get("gloss", [""] * len(batch["name"]))[local_idx]),
        "length": int(length),
        "source_index": int(source_index),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)


@torch.no_grad()
def main():
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    data_dir, min_frames, max_frames, length_multiple, rotation_rep = resolve_data_config(args, ckpt)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest, source_rows, selected_rows, source_indices, manifest_name = select_rows(args, data_dir)
    eval_manifest = args.out_dir / manifest_name
    write_jsonl(eval_manifest, selected_rows)
    save_summary(
        args.out_dir / "sample_manifest_summary.json",
        eval_manifest,
        source_manifest,
        len(source_rows),
        selected_rows,
        source_indices,
        args.seed,
    )

    dataset = UpperSMPLXFlowDataset(
        data_dir,
        split=args.split,
        manifest_path=eval_manifest,
        min_frames=min_frames,
        max_frames=max_frames,
        length_multiple=length_multiple,
        random_crop=False,
        rotation_rep=rotation_rep,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        collate_fn=collate_upper_smplx,
    )

    device = resolve_device(args.device)
    model = build_model(ckpt).to(device).eval()

    metadata = []
    offset = 0
    for batch in loader:
        motion = batch["motion"].to(device)
        mask = batch["mask"].to(device)
        out = model(motion, mask=mask, sample=args.sample_latent)
        recon = out["recon"].detach().cpu().numpy()
        target = motion.detach().cpu().numpy()
        batch_size = len(batch["name"])

        for local_idx in range(batch_size):
            sample_idx = offset + local_idx
            suffix = f"{sample_idx:03d}"
            length = int(batch["length"][local_idx])
            source_index = source_indices[sample_idx]

            gt_rep = denormalize(target[local_idx, :length], dataset)
            recon_rep = denormalize(recon[local_idx, :length], dataset)
            gt_motion = compact_from_rotation_representation(gt_rep, rotation_rep)
            recon_motion = compact_from_rotation_representation(recon_rep, rotation_rep)

            gt_path = args.out_dir / f"gt_{suffix}.npz"
            sample_path = args.out_dir / f"sample_{suffix}.npz"
            save_motion_npz(gt_path, gt_motion, gt_rep, rotation_rep, batch, local_idx, source_index, length)
            save_motion_npz(
                sample_path,
                recon_motion,
                recon_rep,
                rotation_rep,
                batch,
                local_idx,
                source_index,
                length,
                checkpoint=str(args.checkpoint),
                checkpoint_epoch=int(ckpt.get("epoch", -1)),
                checkpoint_global_step=int(ckpt.get("global_step", -1)),
                sample_latent=bool(args.sample_latent),
            )
            metadata.append(
                {
                    "index": int(sample_idx),
                    "source_index": int(source_index),
                    "name": str(batch["name"][local_idx]),
                    "text": str(batch["text"][local_idx]),
                    "length": int(length),
                    "gt_path": str(gt_path),
                    "sample_path": str(sample_path),
                }
            )
        offset += batch_size

    (args.out_dir / "reconstruction_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": int(ckpt.get("epoch", -1)),
                "checkpoint_global_step": int(ckpt.get("global_step", -1)),
                "data_dir": str(data_dir),
                "split": args.split,
                "rotation_rep": rotation_rep,
                "deterministic": not bool(args.sample_latent),
                "num_samples": len(metadata),
                "samples": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "num_samples": len(metadata),
                "out_dir": str(args.out_dir),
                "manifest": str(eval_manifest),
                "checkpoint_epoch": int(ckpt.get("epoch", -1)),
                "checkpoint_global_step": int(ckpt.get("global_step", -1)),
                "rotation_rep": rotation_rep,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
