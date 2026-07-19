from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from flow.smplx_features import compact_rot6d_to_axis_angle, smplx182_from_compact
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import ContinuousSignDataset, collate_continuous_sign
from NIAF.continuous_sign_field.flow_matching import heun_integrate
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_model,
    build_text_encoder,
    encode_batch_text,
    move_batch_to_device,
    prepare_motion,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export one continuous sign field generation as visualizer NPZ files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample_name", default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--solver_steps", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def safe_stem(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "sample"


def find_index(dataset, sample_name):
    if sample_name is None:
        return None
    for idx, item in enumerate(dataset.base.items):
        if str(item.get("name", "")) == str(sample_name):
            return idx
    raise ValueError(f"Sample name not found in split={dataset.split}: {sample_name}")


def rot6d_to_axis_and_smplx(rot6d_tensor):
    rot6d = rot6d_tensor.detach().cpu().float().numpy().astype(np.float32)
    axis = compact_rot6d_to_axis_angle(rot6d).astype(np.float32)
    smplx = smplx182_from_compact(axis).astype(np.float32)
    return rot6d, axis, smplx


def save_npz(path, motion_axis, smplx, rot6d, batch, sample_index, label, extra=None):
    extra = dict(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    name = str(batch["name"][0])
    text = str(batch["text"][0])
    gloss = str(batch["gloss"][0])
    np.savez_compressed(
        path,
        motion=motion_axis.astype(np.float32),
        smplx=smplx.astype(np.float32),
        rot6d=rot6d.astype(np.float32),
        name=np.asarray(name),
        source_name=np.asarray(name),
        text=np.asarray(text),
        gloss=np.asarray(gloss),
        split=np.asarray(str(batch.get("split", ""))),
        source_index=np.asarray(int(sample_index), dtype=np.int32),
        label=np.asarray(str(label)),
        **extra,
    )


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    text_device = resolve_device(args.text_device)

    dataset = ContinuousSignDataset(
        cfg,
        split=args.split,
        limit=0,
        random_crop=False,
        require_fk_cache=True,
    )
    sample_index = find_index(dataset, args.sample_name)
    if sample_index is None:
        sample_index = int(args.index)
    item = dataset[sample_index]
    batch = collate_continuous_sign([item])
    batch["split"] = args.split
    batch = move_batch_to_device(batch, device)

    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, dataset, device)
    model = build_model(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    x = prepare_motion(batch, dataset, device)
    lengths = batch["length"]
    mask = batch["mask"]
    scaffold, _anchor_mask = scaffold_provider.build(batch, x=x)
    tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    solver_steps = int(args.solver_steps or cfg.get("eval", {}).get("solver_steps", 4))
    pred_residual = heun_integrate(
        model,
        scaffold,
        tau,
        mask,
        text_tokens=text_tokens,
        text_mask=text_mask,
        steps=solver_steps,
    )
    pred = scaffold + pred_residual

    length = int(lengths[0].item())
    gt_rot6d, gt_axis, gt_smplx = rot6d_to_axis_and_smplx(x[0, :length])
    pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(pred[0, :length])
    scaffold_rot6d, scaffold_axis, scaffold_smplx = rot6d_to_axis_and_smplx(scaffold[0, :length])

    stem = safe_stem(batch["name"][0])
    out_dir = Path(args.out_dir)
    gt_path = out_dir / f"{stem}_gt.npz"
    pred_path = out_dir / f"{stem}_pred.npz"
    save_npz(gt_path, gt_axis, gt_smplx, gt_rot6d, batch, sample_index, label="ground_truth")
    save_npz(
        pred_path,
        pred_axis,
        pred_smplx,
        pred_rot6d,
        batch,
        sample_index,
        label="continuous_sign_field",
        extra={
            "coarse_motion": scaffold_axis.astype(np.float32),
            "coarse_smplx": scaffold_smplx.astype(np.float32),
            "coarse_rot6d": scaffold_rot6d.astype(np.float32),
            "checkpoint": np.asarray(str(args.checkpoint)),
            "checkpoint_epoch": np.asarray(int(checkpoint.get("epoch", -1)), dtype=np.int32),
            "solver_steps": np.asarray(int(solver_steps), dtype=np.int32),
            "scaffold_source": np.asarray(str(scaffold_provider.source)),
        },
    )
    summary = {
        "sample_name": batch["name"][0],
        "split": args.split,
        "index": int(sample_index),
        "length": int(length),
        "text": batch["text"][0],
        "gloss": batch["gloss"][0],
        "gt": str(gt_path),
        "pred": str(pred_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "solver_steps": int(solver_steps),
        "scaffold_source": scaffold_provider.source,
        "scaffold_config": scaffold_provider.config_summary,
    }
    summary_path = out_dir / f"{stem}_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
