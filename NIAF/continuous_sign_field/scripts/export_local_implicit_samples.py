from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import ContinuousSignDataset, collate_continuous_sign
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_eval_samples import (
    rot6d_to_axis_and_smplx,
    save_eval_npz,
    select_manifest,
)
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device
from NIAF.continuous_sign_field.scripts.train_local_implicit_field import build_local_model
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_text_encoder,
    encode_batch_text,
    move_batch_to_device,
    prepare_motion,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export local implicit NIAF samples with predicted sequence lengths.")
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
    parser.add_argument("--length_mode", default="predicted", choices=["predicted", "ground_truth"])
    return parser.parse_args()


def generation_batch(batch, lengths, device):
    max_len = int(lengths.max().item())
    frames = torch.arange(max_len, device=device).unsqueeze(0)
    generated = {
        "name": batch["name"],
        "text": batch["text"],
        "gloss": batch["gloss"],
        "motion_path": batch["motion_path"],
        "length": lengths,
        "mask": frames < lengths.unsqueeze(1),
    }
    return generated


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("scaffold", {})["cache_only"] = False
    cfg.setdefault("scaffold", {})["prefer_cache"] = False
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
    model = build_local_model(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    duration_cfg = cfg.get("duration", {})
    min_frames = int(duration_cfg.get("min_frames", cfg.get("data", {}).get("min_frames", 40)))
    max_frames = int(duration_cfg.get("max_frames", cfg.get("data", {}).get("max_frames", 400)))
    multiple = int(duration_cfg.get("length_multiple", cfg.get("data", {}).get("length_multiple", 4)))
    rows = []
    sample_counter = 0
    for batch in tqdm(loader, desc="export local implicit samples"):
        batch = move_batch_to_device(batch, device)
        x = prepare_motion(batch, dataset, device)
        text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
        continuous_lengths = torch.exp(model.predict_log_frames(text_tokens, text_mask=text_mask))
        predicted_lengths = model.predict_lengths(
            text_tokens,
            text_mask=text_mask,
            min_frames=min_frames,
            max_frames=max_frames,
            multiple=multiple,
        )
        output_lengths = predicted_lengths if args.length_mode == "predicted" else batch["length"]
        generated = generation_batch(batch, output_lengths, device)
        scaffold, _anchor_mask = scaffold_provider.build(generated, x=None, use_cache=False)
        tau = normalized_time_grid(output_lengths, max_len=scaffold.shape[1], device=device, dtype=scaffold.dtype)
        outputs = model(tau, scaffold, generated["mask"], text_tokens, text_mask=text_mask)
        pred = outputs["prediction"]

        for local_idx in range(len(batch["name"])):
            gt_length = int(batch["length"][local_idx].item())
            pred_length = int(output_lengths[local_idx].item())
            suffix = f"{sample_counter:04d}"
            meta = {
                "name": batch["name"][local_idx],
                "text": batch["text"][local_idx],
                "gloss": batch["gloss"][local_idx],
                "split": args.split,
                "source_index": sample_counter,
            }
            gt_rot6d, gt_axis, gt_smplx = rot6d_to_axis_and_smplx(x[local_idx, :gt_length])
            pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(pred[local_idx, :pred_length])
            coarse_rot6d, coarse_axis, coarse_smplx = rot6d_to_axis_and_smplx(
                scaffold[local_idx, :pred_length]
            )
            gt_path = out_dir / f"gt_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_eval_npz(gt_path, gt_axis, gt_smplx, gt_rot6d, meta, label="ground_truth")
            save_eval_npz(
                sample_path,
                pred_axis,
                pred_smplx,
                pred_rot6d,
                meta,
                label="niaf_local_amortized_implicit",
                extra={
                    "coarse_motion": coarse_axis.astype(np.float32),
                    "coarse_smplx": coarse_smplx.astype(np.float32),
                    "coarse_rot6d": coarse_rot6d.astype(np.float32),
                    "local_implicit_motion": pred_axis.astype(np.float32),
                    "local_implicit_smplx": pred_smplx.astype(np.float32),
                    "local_implicit_rot6d": pred_rot6d.astype(np.float32),
                    "checkpoint": np.asarray(str(args.checkpoint)),
                    "checkpoint_epoch": np.asarray(int(checkpoint.get("epoch", -1)), dtype=np.int32),
                    "scaffold_source": np.asarray(str(scaffold_provider.source)),
                    "length_mode": np.asarray(str(args.length_mode)),
                    "ground_truth_length": np.asarray(gt_length, dtype=np.int32),
                    "predicted_length": np.asarray(int(predicted_lengths[local_idx].item()), dtype=np.int32),
                    "predicted_length_continuous": np.asarray(
                        float(continuous_lengths[local_idx].item()), dtype=np.float32
                    ),
                },
            )
            rows.append(
                {
                    "index": suffix,
                    "name": meta["name"],
                    "text": meta["text"],
                    "ground_truth_length": gt_length,
                    "predicted_length": int(predicted_lengths[local_idx].item()),
                    "output_length": pred_length,
                    "gt": str(gt_path),
                    "sample": str(sample_path),
                }
            )
            sample_counter += 1

    length_errors = [abs(row["predicted_length"] - row["ground_truth_length"]) for row in rows]
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "length_mode": args.length_mode,
        "scaffold_source": scaffold_provider.source,
        "scaffold_config": scaffold_provider.config_summary,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "duration_mae_frames": float(sum(length_errors) / max(len(length_errors), 1)),
        "rows": rows,
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("checkpoint_epoch", "split", "length_mode", "num_exported", "duration_mae_frames")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
