from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import (
    PART_KEYS,
    build_upper_smplx_dataset,
    cache_path_for_item,
    denormalize_motion,
)
from NIAF.oracle_smplx_field.geometry.smplx_fk import DifferentiableSMPLXForward


def parse_args():
    parser = argparse.ArgumentParser(description="Cache GT SMPL-X FK joints for continuous residual flow training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested is None or requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def compute_parts_in_chunks(fk, compact6d, chunk_size):
    chunks = []
    for start in range(0, compact6d.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), compact6d.shape[0])
        with torch.no_grad():
            chunks.append(fk.parts_from_rot6d(compact6d[start:end]))
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in PART_KEYS}


def cache_split(cfg, split, device, limit=0, overwrite=False):
    dataset = build_upper_smplx_dataset(cfg, split, limit=limit or None, random_crop=False)
    cache_dir = Path(cfg["cache"]["fk_cache_dir"])
    metric_cfg = cfg.get("metrics", {})
    fk = DifferentiableSMPLXForward(
        model_dir=metric_cfg.get("model_dir", "deps/smpl_models"),
        gender=metric_cfg.get("gender", "NEUTRAL"),
        device=device,
        betas_mode=metric_cfg.get("betas_mode", "h2s_fixed"),
    )
    fk.eval()
    for param in fk.parameters():
        param.requires_grad_(False)

    mean = torch.from_numpy(dataset.mean.astype(np.float32)).to(device)
    std = torch.from_numpy(dataset.std.astype(np.float32)).to(device)
    chunk_size = int(cfg.get("cache", {}).get("fk_batch_size", 128))
    written = 0
    skipped = 0
    for index in tqdm(range(len(dataset)), desc=f"cache_fk:{split}"):
        sample = dataset[index]
        manifest_item = dataset.items[index]
        out_path = cache_path_for_item(cache_dir, manifest_item)
        if out_path.is_file() and not overwrite:
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        motion_norm = sample["motion"].to(device=device, dtype=torch.float32).unsqueeze(0)
        motion = denormalize_motion(motion_norm, mean, std).squeeze(0)
        length = int(sample["length"])
        parts = compute_parts_in_chunks(fk, motion[:length], chunk_size=chunk_size)
        np.savez_compressed(
            out_path,
            **{key: value.detach().cpu().numpy().astype(np.float32) for key, value in parts.items()},
            length=np.asarray(length, dtype=np.int32),
            name=np.asarray(str(sample["name"])),
            motion_path=np.asarray(str(manifest_item["motion_path"])),
            split=np.asarray(str(split)),
        )
        written += 1
    print(f"[{split}] wrote={written} skipped={skipped} cache_dir={cache_dir}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        cache_split(cfg, split, device=device, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

