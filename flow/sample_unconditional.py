#!/usr/bin/env python
import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from flow.model import build_model_from_args, sample_euler, sample_heun
from flow.render import smplx182_to_vertices, write_vertices_video
from flow.smplx_features import COMPACT_DIM, smplx182_from_compact


DEFAULT_OUT_DIR = Path("visualize/flow/flow_uncond_samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample an unconditional SMPL-X flow checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=196)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sampler", default="euler", choices=["euler", "heun"])
    parser.add_argument("--noise_smoothing", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_device", default="cpu", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(args, device):
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("model_config", {})
    model_args = SimpleNamespace(
        hidden_dim=cfg.get("hidden_dim", 512),
        num_layers=cfg.get("num_layers", 8),
        num_heads=cfg.get("num_heads", 8),
        dropout=cfg.get("dropout", 0.1),
        max_frames=cfg.get("max_frames", max(400, args.length)),
    )
    model = build_model_from_args(model_args)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    data_dir = args.data_dir
    if data_dir is None:
        data_cfg = ckpt.get("data_config", {})
        data_dir = Path(data_cfg.get("data_dir", "/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx"))
    mean = np.load(data_dir / "meta" / "mean.npy").astype(np.float32)
    std = np.load(data_dir / "meta" / "std.npy").astype(np.float32)
    return model, mean, std


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = load_model(args, device)
    mask = torch.ones(args.num_samples, args.length, dtype=torch.bool, device=device)
    sampler = sample_heun if args.sampler == "heun" else sample_euler
    samples = sampler(
        model,
        (args.num_samples, args.length, COMPACT_DIM),
        steps=args.steps,
        device=device,
        mask=mask,
        noise_smoothing=args.noise_smoothing,
    )
    samples = samples.detach().cpu().numpy()
    samples = samples * std[None, None] + mean[None, None]

    for idx in range(args.num_samples):
        compact = samples[idx].astype(np.float32)
        full_smplx = smplx182_from_compact(compact)
        npz_path = args.out_dir / f"sample_{idx:02d}.npz"
        np.savez_compressed(npz_path, motion=compact, smplx=full_smplx)
        print(f"Saved: {npz_path}")

        if args.render:
            vertices, faces = smplx182_to_vertices(
                full_smplx,
                model_dir=args.model_dir,
                device=args.render_device,
                batch_size=args.smplx_batch_size,
            )
            video_path = args.out_dir / f"sample_{idx:02d}.mp4"
            write_vertices_video(
                vertices,
                faces,
                video_path,
                fps=args.fps,
                width=args.width,
                height=args.height,
                face_stride=args.software_face_stride,
                label=f"unconditional flow sample {idx:02d}",
                view_transform=args.view_transform,
            )
            print(f"Saved: {video_path}")


if __name__ == "__main__":
    main()
