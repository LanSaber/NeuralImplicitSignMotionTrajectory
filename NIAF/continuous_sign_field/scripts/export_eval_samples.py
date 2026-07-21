from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.smplx_features import compact_rot6d_to_axis_angle, smplx182_from_compact
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import ContinuousSignDataset, collate_continuous_sign
from NIAF.continuous_sign_field.flow_matching import heun_integrate
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_model,
    build_text_encoder,
    encode_batch_text,
    move_batch_to_device,
    prepare_motion,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export NIAF continuous sign field samples in flow DTW-evaluation NPZ format."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--solver_steps", type=int, default=None)
    parser.add_argument("--num_negative_candidates", type=int, default=None)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_manifest(
    cfg,
    split,
    out_dir,
    num_samples,
    seed,
    manifest=None,
    selection_mode="random",
):
    if manifest is not None:
        selected = read_jsonl(manifest)
        out_manifest = out_dir / Path(manifest).name
        if Path(manifest).resolve(strict=False) != out_manifest.resolve(strict=False):
            write_jsonl(out_manifest, selected)
        indices = None
        source_manifest = Path(manifest)
    else:
        source_manifest = Path(cfg["data"]["data_dir"]) / "meta" / f"manifest_{split}.jsonl"
        rows = read_jsonl(source_manifest)
        count = min(int(num_samples), len(rows)) if int(num_samples) > 0 else len(rows)
        if selection_mode == "first":
            indices = list(range(count))
        elif selection_mode == "random":
            rng = random.Random(int(seed))
            indices = rng.sample(range(len(rows)), count)
        else:
            raise ValueError(f"Unsupported manifest selection mode {selection_mode!r}")
        selected = [rows[idx] for idx in indices]
        suffix = (
            f"first{count}"
            if selection_mode == "first"
            else f"random{count}_seed{seed}"
        )
        out_manifest = out_dir / f"manifest_{split}_{suffix}.jsonl"
        write_jsonl(out_manifest, selected)

    frame_counts = [int(row.get("num_frames", 0)) for row in selected if int(row.get("num_frames", 0)) > 0]
    summary = {
        "source_manifest": str(source_manifest),
        "output_manifest": str(out_manifest),
        "seed": int(seed),
        "sample_count": len(selected),
        "indices": indices,
        "min_frames": min(frame_counts) if frame_counts else None,
        "max_frames": max(frame_counts) if frame_counts else None,
        "mean_frames": (sum(frame_counts) / len(frame_counts)) if frame_counts else None,
    }
    (out_dir / "sample_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_manifest, selected, summary


def rot6d_to_axis_and_smplx(rot6d_tensor):
    rot6d = rot6d_tensor.detach().cpu().float().numpy().astype(np.float32)
    axis = compact_rot6d_to_axis_angle(rot6d).astype(np.float32)
    smplx = smplx182_from_compact(axis).astype(np.float32)
    return rot6d, axis, smplx


def save_eval_npz(path, axis, smplx, rot6d, meta, label, extra=None):
    extra = dict(extra or {})
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


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.num_negative_candidates is not None:
        cfg.setdefault("adapter", {})["num_negative_candidates"] = int(args.num_negative_candidates)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest, selected_rows, manifest_summary = select_manifest(
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
    dataset = ContinuousSignDataset(
        cfg,
        split=args.split,
        limit=0,
        random_crop=False,
        require_fk_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_continuous_sign,
    )

    text_encoder = build_text_encoder(cfg, text_device)
    scaffold_provider = ScaffoldProvider(cfg, dataset, device)
    model = build_model(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    solver_steps = int(args.solver_steps or cfg.get("eval", {}).get("solver_steps", 4))
    rows = []
    sample_counter = 0
    for batch in tqdm(loader, desc="export eval samples"):
        batch["split"] = args.split
        batch = move_batch_to_device(batch, device)
        x = prepare_motion(batch, dataset, device)
        lengths = batch["length"]
        mask = batch["mask"]
        scaffold, _anchor_mask = scaffold_provider.build(batch, x=x)
        tau = normalized_time_grid(lengths, max_len=x.shape[1], device=device, dtype=x.dtype)
        text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
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
            pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(pred[local_idx, :length])
            scaffold_rot6d, scaffold_axis, scaffold_smplx = rot6d_to_axis_and_smplx(scaffold[local_idx, :length])

            gt_path = out_dir / f"gt_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_eval_npz(gt_path, gt_axis, gt_smplx, gt_rot6d, meta, label="ground_truth")
            save_eval_npz(
                sample_path,
                pred_axis,
                pred_smplx,
                pred_rot6d,
                meta,
                label="niaf_continuous_sign_field",
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
            rows.append(
                {
                    "index": suffix,
                    "name": meta["name"],
                    "text": meta["text"],
                    "gloss": meta["gloss"],
                    "length": length,
                    "gt": str(gt_path),
                    "sample": str(sample_path),
                }
            )
            sample_counter += 1

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "solver_steps": solver_steps,
        "scaffold_source": scaffold_provider.source,
        "scaffold_config": scaffold_provider.config_summary,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "rows": rows,
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("checkpoint_epoch", "split", "solver_steps", "scaffold_source", "num_exported")}, indent=2))


if __name__ == "__main__":
    main()
