from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.smplx_features import compact_rot6d_to_axis_angle, smplx182_from_compact
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import ContinuousSignDataset, collate_continuous_sign
from NIAF.continuous_sign_field.meta_learning import adapt_code, build_support_query_masks
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_eval_samples import select_manifest
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device
from NIAF.continuous_sign_field.scripts.train_meta_implicit_field import build_meta_model
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_text_encoder,
    encode_batch_text,
    move_batch_to_device,
    prepare_motion,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export MAML-style meta-implicit NIAF samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no_adapt", action="store_true", help="Export prior only; adapted output equals prior.")
    return parser.parse_args()


def rot6d_to_axis_and_smplx(rot6d_tensor):
    rot6d = rot6d_tensor.detach().cpu().float().numpy().astype(np.float32)
    axis = compact_rot6d_to_axis_angle(rot6d).astype(np.float32)
    smplx = smplx182_from_compact(axis).astype(np.float32)
    return rot6d, axis, smplx


def save_meta_npz(path, axis, smplx, rot6d, meta, label, extra=None):
    extra = dict(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        motion=axis.astype(np.float32),
        smplx=smplx.astype(np.float32),
        rot6d=rot6d.astype(np.float32),
        name=np.asarray(str(meta["name"])),
        source_name=np.asarray(str(meta["name"])),
        text=np.asarray(str(meta.get("text", ""))),
        gloss=np.asarray(str(meta.get("gloss", ""))),
        split=np.asarray(str(meta.get("split", ""))),
        source_index=np.asarray(int(meta["source_index"]), dtype=np.int32),
        label=np.asarray(str(label)),
        **extra,
    )


def initial_code_for_export(model, scaffold, mask, lengths, text_tokens, text_mask):
    with torch.no_grad():
        code = model.initial_code(scaffold, mask, lengths, text_tokens=text_tokens, text_mask=text_mask)
    return code.detach().requires_grad_(True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest, _selected_rows, manifest_summary = select_manifest(
        cfg,
        args.split,
        out_dir,
        args.num_samples,
        args.seed,
        manifest=args.manifest,
    )
    cfg.setdefault("data", {})[f"{args.split}_manifest_path"] = str(selected_manifest)
    cfg.setdefault("data", {})[f"limit_{args.split}"] = 0

    device = resolve_device(args.device)
    text_device = resolve_device(args.text_device)
    dataset = ContinuousSignDataset(cfg, split=args.split, limit=0, random_crop=False, require_fk_cache=False)
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_continuous_sign,
    )

    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, dataset, device)
    model = build_meta_model(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    rows = []
    sample_counter = 0
    for batch in tqdm(loader, desc="export meta-implicit samples"):
        batch["split"] = args.split
        batch = move_batch_to_device(batch, device)
        x = prepare_motion(batch, dataset, device)
        mask = batch["mask"]
        lengths = batch["length"]
        scaffold, anchor_mask = scaffold_provider.build(batch, x=x)
        tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
        text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
        z0 = initial_code_for_export(model, scaffold, mask, lengths, text_tokens, text_mask)
        prior_residual = model(tau, scaffold, z0, mask=mask)
        target_residual = (x - scaffold) * mask.unsqueeze(-1).to(x.dtype)
        support_mask, query_mask = build_support_query_masks(
            mask,
            anchor_mask=anchor_mask,
            support_mode=cfg.get("meta", {}).get("support_mode", "stride"),
            support_stride=int(cfg.get("meta", {}).get("support_stride", 8)),
        )
        if args.no_adapt:
            adapted_code = z0
        else:
            adapted_code, _inner_losses = adapt_code(
                model,
                z0,
                tau,
                scaffold,
                x,
                target_residual,
                support_mask,
                cfg,
            )
        adapted_residual = model(tau, scaffold, adapted_code, mask=mask)
        prior = scaffold + prior_residual
        adapted = scaffold + adapted_residual

        for local_idx, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            suffix = f"{sample_counter:04d}"
            meta = {
                "name": batch["name"][local_idx],
                "text": batch["text"][local_idx],
                "gloss": batch["gloss"][local_idx],
                "split": args.split,
                "source_index": sample_counter,
            }
            gt_rot6d, gt_axis, gt_smplx = rot6d_to_axis_and_smplx(x[local_idx, :length])
            scaffold_rot6d, scaffold_axis, scaffold_smplx = rot6d_to_axis_and_smplx(scaffold[local_idx, :length])
            prior_rot6d, prior_axis, prior_smplx = rot6d_to_axis_and_smplx(prior[local_idx, :length])
            adapted_rot6d, adapted_axis, adapted_smplx = rot6d_to_axis_and_smplx(adapted[local_idx, :length])

            support_np = support_mask[local_idx, :length].detach().cpu().numpy().astype(np.bool_)
            query_np = query_mask[local_idx, :length].detach().cpu().numpy().astype(np.bool_)
            common_extra = {
                "checkpoint": np.asarray(str(args.checkpoint)),
                "checkpoint_epoch": np.asarray(int(checkpoint.get("epoch", -1)), dtype=np.int32),
                "scaffold_source": np.asarray(str(scaffold_provider.source)),
                "support_mask": support_np,
                "query_mask": query_np,
            }

            gt_path = out_dir / f"gt_{suffix}.npz"
            scaffold_path = out_dir / f"scaffold_{suffix}.npz"
            prior_path = out_dir / f"meta_prior_{suffix}.npz"
            adapted_path = out_dir / f"meta_adapted_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_meta_npz(gt_path, gt_axis, gt_smplx, gt_rot6d, meta, "ground_truth", extra=common_extra)
            save_meta_npz(scaffold_path, scaffold_axis, scaffold_smplx, scaffold_rot6d, meta, "scaffold", extra=common_extra)
            save_meta_npz(prior_path, prior_axis, prior_smplx, prior_rot6d, meta, "meta_prior", extra=common_extra)
            save_meta_npz(adapted_path, adapted_axis, adapted_smplx, adapted_rot6d, meta, "meta_adapted", extra=common_extra)
            save_meta_npz(
                sample_path,
                gt_axis,
                gt_smplx,
                gt_rot6d,
                meta,
                "ground_truth",
                extra={
                    **common_extra,
                    "coarse_motion": scaffold_axis.astype(np.float32),
                    "coarse_smplx": scaffold_smplx.astype(np.float32),
                    "coarse_rot6d": scaffold_rot6d.astype(np.float32),
                    "meta_prior_motion": prior_axis.astype(np.float32),
                    "meta_prior_smplx": prior_smplx.astype(np.float32),
                    "meta_prior_rot6d": prior_rot6d.astype(np.float32),
                    "meta_adapted_motion": adapted_axis.astype(np.float32),
                    "meta_adapted_smplx": adapted_smplx.astype(np.float32),
                    "meta_adapted_rot6d": adapted_rot6d.astype(np.float32),
                },
            )
            rows.append(
                {
                    "index": suffix,
                    "name": meta["name"],
                    "text": meta["text"],
                    "gloss": meta["gloss"],
                    "length": length,
                    "gt": str(gt_path),
                    "scaffold": str(scaffold_path),
                    "meta_prior": str(prior_path),
                    "meta_adapted": str(adapted_path),
                    "sample": str(sample_path),
                }
            )
            sample_counter += 1

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "scaffold_source": scaffold_provider.source,
        "scaffold_config": scaffold_provider.config_summary,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "rows": rows,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("checkpoint_epoch", "split", "scaffold_source", "num_exported")}, indent=2))


if __name__ == "__main__":
    main()
