#!/usr/bin/env python
"""Evaluate VAE reconstructions in memory without saving motion NPZ files."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flow.VAE.reconstruct_manifest import (
    denormalize,
    resolve_data_config,
    save_summary,
    select_rows,
    write_jsonl,
)
from flow.VAE.reconstruct_vae import build_model, load_checkpoint, resolve_device as resolve_vae_device
from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.evaluate.dtw_mpjpe_t2m_default import (
    PARTS,
    dtw_distance_default,
    dtw_distance_pa,
    resolve_betas_override,
    smplx_to_joints_vertices,
    summarize,
    t2m_default_parts,
    t2m_raw_parts,
)
from flow.evaluate.ndtw_smplx_keypoints import UPPER_BODY_JOINTS
from flow.render import DEFAULT_MODEL_DIR
from flow.smplx_features import compact_from_rotation_representation, smplx182_from_compact


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic VAE reconstruction evaluation directly in memory. "
            "Only metric JSON/CSV and manifest summary files are written."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--length_multiple", type=int, default=None)
    parser.add_argument("--rotation_rep", "--rotation-rep", default=None, choices=["axis_angle", "rot6d"])
    parser.add_argument("--vae_device", "--vae-device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--metric_device", "--metric-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--sample_latent", action="store_true", help="Use stochastic z instead of the latent mean.")
    parser.add_argument("--torch_seed", "--torch-seed", type=int, default=None)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gender", default="NEUTRAL")
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--betas_mode", default="h2s_fixed", choices=["from_params", "zero", "h2s_fixed"])
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--alignment_mode", default="default", choices=["default", "pa"])
    parser.add_argument("--comparison_name", default="vae_reconstruction")
    return parser.parse_args()


def set_torch_seed(seed):
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def default_metric_paths(out_dir, alignment_mode):
    if alignment_mode == "pa":
        return out_dir / "dtw_pa_mpjpe_t2m_h2s_betas.json", out_dir / "dtw_pa_mpjpe_t2m_h2s_betas.csv"
    return out_dir / "dtw_mpjpe_t2m_default_h2s_betas.json", out_dir / "dtw_mpjpe_t2m_default_h2s_betas.csv"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "comparison",
        "part",
        "gt_len",
        "pred_len",
        "dtw",
        "path_len",
        "ndtw",
        "ndtw_ref",
        "sample",
        "gt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def smplx_to_parts(smplx_params, args, betas_override):
    joints, vertices = smplx_to_joints_vertices(
        smplx_params,
        model_dir=args.model_dir,
        gender=args.gender,
        device=args.metric_device,
        batch_size=args.smplx_batch_size,
        betas_override=betas_override,
    )
    if args.alignment_mode == "pa":
        return t2m_raw_parts(joints, vertices)
    return t2m_default_parts(joints, vertices)


def append_metric_rows(rows, index, gt_parts, pred_parts, args, gt_len, pred_len):
    for part in args.parts:
        if args.alignment_mode == "pa":
            values = dtw_distance_pa(pred_parts, gt_parts, part)
        else:
            values = dtw_distance_default(pred_parts[part], gt_parts[part])
        rows.append(
            {
                "index": f"{index:03d}",
                "comparison": args.comparison_name,
                "part": part,
                "gt_len": int(gt_len),
                "pred_len": int(pred_len),
                "dtw": values["dtw"],
                "path_len": values["path_len"],
                "ndtw": values["ndtw"],
                "ndtw_ref": values["ndtw_ref"],
                "sample": f"memory:sample_{index:03d}",
                "gt": f"memory:gt_{index:03d}",
            }
        )


@torch.no_grad()
def main():
    args = parse_args()
    set_torch_seed(args.torch_seed)
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

    vae_device = resolve_vae_device(args.vae_device)
    model = build_model(ckpt).to(vae_device).eval()
    betas_override = resolve_betas_override(args.betas_mode)

    rows = []
    offset = 0
    for batch in loader:
        motion = batch["motion"].to(vae_device)
        mask = batch["mask"].to(vae_device)
        out = model(motion, mask=mask, sample=args.sample_latent)
        recon = out["recon"].detach().cpu().numpy()
        target = motion.detach().cpu().numpy()

        for local_idx in range(len(batch["name"])):
            sample_idx = offset + local_idx
            length = int(batch["length"][local_idx])
            gt_rep = denormalize(target[local_idx, :length], dataset)
            recon_rep = denormalize(recon[local_idx, :length], dataset)
            gt_motion = compact_from_rotation_representation(gt_rep, rotation_rep)
            recon_motion = compact_from_rotation_representation(recon_rep, rotation_rep)

            gt_parts = smplx_to_parts(smplx182_from_compact(gt_motion), args, betas_override)
            pred_parts = smplx_to_parts(smplx182_from_compact(recon_motion), args, betas_override)
            append_metric_rows(rows, sample_idx, gt_parts, pred_parts, args, length, length)
        offset += len(batch["name"])

    out_json, out_csv = default_metric_paths(args.out_dir, args.alignment_mode)
    out_json = args.out_json or out_json
    out_csv = args.out_csv or out_csv

    payload = {
        "samples_dir": str(args.out_dir),
        "sample_key": "memory:smplx",
        "gt_key": "memory:smplx",
        "prior_key": None,
        "model_dir": str(args.model_dir),
        "vae_device": str(vae_device),
        "metric_device": args.metric_device,
        "frame_metric": "mpjpe",
        "alignment_mode": args.alignment_mode,
        "metric_preset": (
            "mgpt_t2m_pa_align_idx_none"
            if args.alignment_mode == "pa"
            else "mgpt_t2m_default_align_idx_0"
        ),
        "procrustes_aligned": bool(args.alignment_mode == "pa"),
        "betas_mode": args.betas_mode,
        "betas_override": None if betas_override is None else np.asarray(betas_override).tolist(),
        "joint_sets": {
            "body": list(UPPER_BODY_JOINTS),
            "lhand": "mGPT orig_hand_regressor left layout, 21 keypoints",
            "rhand": "mGPT orig_hand_regressor right layout, 21 keypoints",
            "wholebody": "upper_body + lhand + rhand, 54 keypoints",
        },
        "definition": (
            "VAE reconstruction is decoded in memory and evaluated with the same "
            "mGPT/metrics/t2m.py-style DTW-MPJPE implementation used for saved "
            "flow samples. No reconstruction NPZ files are written."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_global_step": int(ckpt.get("global_step", -1)),
        "data_dir": str(data_dir),
        "split": args.split,
        "manifest": str(eval_manifest),
        "rotation_rep": rotation_rep,
        "deterministic": not bool(args.sample_latent),
        "torch_seed": args.torch_seed,
        "num_pairs": len(selected_rows),
        "skipped_prior": len(selected_rows),
        "parts": args.parts,
        "comparison_name": args.comparison_name,
        "summary": summarize(rows),
        "rows": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    print(json.dumps({"num_pairs": len(selected_rows), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
