#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import torch

from flow.VAE.model import TemporalSMPLXVAE
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.smplx_features import (
    COMPACT_DIM,
    compact_from_rotation_representation,
    normalize_rotation_rep,
    rotation_rep_dim,
    smplx182_from_compact,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct compact SMPL-X samples with a trained VAE.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default=Path("visualize/flow/flow_vae_recon_samples"))
    parser.add_argument("--min_frames", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--length_multiple", type=int, default=None)
    parser.add_argument("--rotation_rep", "--rotation-rep", default=None, choices=["axis_angle", "rot6d"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--sample_latent", action="store_true", help="Use stochastic z instead of the latent mean.")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def build_model(ckpt):
    config = dict(ckpt["model_config"])
    config.setdefault("input_dim", COMPACT_DIM)
    config.pop("rotation_rep", None)
    config.pop("representation_dim", None)
    model = TemporalSMPLXVAE(**config)
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def denormalize(motion, dataset):
    return (motion * dataset.std[None] + dataset.mean[None]).astype(np.float32)


@torch.no_grad()
def main():
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    data_config = ckpt.get("data_config", {})
    data_dir = args.data_dir or Path(data_config.get("data_dir", ""))
    if not data_dir:
        raise ValueError("--data_dir is required when the checkpoint does not store data_config.data_dir")

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

    dataset = UpperSMPLXFlowDataset(
        data_dir,
        split=args.split,
        min_frames=min_frames,
        max_frames=max_frames,
        length_multiple=length_multiple,
        random_crop=False,
        rotation_rep=rotation_rep,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {data_dir} split={args.split}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index {args.index} is outside split length {len(dataset)}")

    end = min(args.index + max(args.num_samples, 1), len(dataset))
    items = [dataset[idx] for idx in range(args.index, end)]
    batch = collate_upper_smplx(items)

    device = resolve_device(args.device)
    model = build_model(ckpt).to(device).eval()
    motion = batch["motion"].to(device)
    mask = batch["mask"].to(device)
    out = model(motion, mask=mask, sample=args.sample_latent)
    recon = out["recon"].detach().cpu().numpy()
    target = motion.detach().cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for local_idx, item in enumerate(items):
        length = int(item["length"])
        gt_representation = denormalize(target[local_idx, :length], dataset)
        recon_representation = denormalize(recon[local_idx, :length], dataset)
        gt_motion = compact_from_rotation_representation(gt_representation, rotation_rep)
        recon_motion = compact_from_rotation_representation(recon_representation, rotation_rep)
        name = str(item["name"])
        text = str(item.get("text", ""))

        gt_path = args.out_dir / f"gt_{local_idx:02d}.npz"
        recon_path = args.out_dir / f"recon_{local_idx:02d}.npz"
        np.savez_compressed(
            gt_path,
            motion=gt_motion,
            representation=gt_representation,
            rotation_rep=rotation_rep,
            smplx=smplx182_from_compact(gt_motion),
            name=name,
            text=text,
            length=length,
            source_index=args.index + local_idx,
        )
        np.savez_compressed(
            recon_path,
            motion=recon_motion,
            representation=recon_representation,
            rotation_rep=rotation_rep,
            smplx=smplx182_from_compact(recon_motion),
            name=name,
            text=text,
            length=length,
            source_index=args.index + local_idx,
            checkpoint=str(args.checkpoint),
            checkpoint_epoch=int(ckpt.get("epoch", -1)),
            checkpoint_global_step=int(ckpt.get("global_step", -1)),
        )
        metadata.append(
            {
                "index": args.index + local_idx,
                "name": name,
                "text": text,
                "length": length,
                "gt_path": str(gt_path),
                "recon_path": str(recon_path),
            }
        )
        print(f"Saved GT: {gt_path}")
        print(f"Saved reconstruction: {recon_path}")

    print(f"checkpoint_epoch={ckpt.get('epoch', 'unknown')} global_step={ckpt.get('global_step', 'unknown')}")
    print(f"num_samples={len(metadata)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
