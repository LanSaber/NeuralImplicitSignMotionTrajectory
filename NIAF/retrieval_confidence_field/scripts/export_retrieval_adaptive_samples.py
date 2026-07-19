from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import ContinuousSignDataset, collate_continuous_sign
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import ARTICULATOR_NAMES
from NIAF.continuous_sign_field.scaffold import normalized_time_grid
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_eval_samples import (
    rot6d_to_axis_and_smplx,
    save_eval_npz,
    select_manifest,
)
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_text_encoder,
    encode_batch_text,
    move_batch_to_device,
    prepare_motion,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    build_retrieval_adaptive_model,
    validate_train_only_retrieval_bank,
)


__all__ = ["generation_batch", "main", "parse_args"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export retrieval-confidence adaptive NIAF samples."
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
    parser.add_argument(
        "--length_mode", default="predicted", choices=["predicted", "ground_truth"]
    )
    return parser.parse_args()


def generation_batch(batch, lengths, device):
    max_len = int(lengths.max().item())
    frames = torch.arange(max_len, device=device).unsqueeze(0)
    return {
        "name": batch["name"],
        "text": batch["text"],
        "gloss": batch["gloss"],
        "motion_path": batch["motion_path"],
        "length": lengths,
        "mask": frames < lengths.unsqueeze(1),
    }


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
    retrieval_bank = validate_train_only_retrieval_bank(cfg, scaffold_provider)
    model = build_retrieval_adaptive_model(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    duration_cfg = cfg.get("duration", {})
    min_frames = int(duration_cfg.get("min_frames", cfg.get("data", {}).get("min_frames", 40)))
    max_frames = int(duration_cfg.get("max_frames", cfg.get("data", {}).get("max_frames", 400)))
    multiple = int(
        duration_cfg.get("length_multiple", cfg.get("data", {}).get("length_multiple", 4))
    )
    rows = []
    sample_counter = 0
    for batch in tqdm(loader, desc="export retrieval-adaptive samples"):
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
        output_lengths = (
            predicted_lengths if args.length_mode == "predicted" else batch["length"]
        )
        generated = generation_batch(batch, output_lengths, device)
        scaffold, _anchor_mask, scaffold_metadata = scaffold_provider.build_with_metadata(
            generated,
            x=None,
            use_cache=False,
        )
        retrieval_features = scaffold_metadata["retrieval_features"]
        tau = normalized_time_grid(
            output_lengths,
            max_len=scaffold.shape[1],
            device=device,
            dtype=scaffold.dtype,
        )
        outputs = model(
            tau,
            scaffold,
            generated["mask"],
            text_tokens,
            retrieval_features,
            text_mask=text_mask,
        )
        prediction = outputs["prediction"]

        for local_index in range(len(batch["name"])):
            gt_length = int(batch["length"][local_index].item())
            output_length = int(output_lengths[local_index].item())
            suffix = f"{sample_counter:04d}"
            meta = {
                "name": batch["name"][local_index],
                "text": batch["text"][local_index],
                "gloss": batch["gloss"][local_index],
                "split": args.split,
                "source_index": sample_counter,
            }
            gt_rot6d, gt_axis, gt_smplx = rot6d_to_axis_and_smplx(
                x[local_index, :gt_length]
            )
            pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(
                prediction[local_index, :output_length]
            )
            scaffold_rot6d, scaffold_axis, scaffold_smplx = rot6d_to_axis_and_smplx(
                scaffold[local_index, :output_length]
            )
            gt_path = out_dir / f"gt_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_eval_npz(gt_path, gt_axis, gt_smplx, gt_rot6d, meta, label="ground_truth")
            model_type = str(
                checkpoint.get(
                    "model_type",
                    cfg.get("model", {}).get(
                        "type", "retrieval_confidence_adaptive_field"
                    ),
                )
            )
            extra = {
                "coarse_motion": scaffold_axis.astype(np.float32),
                "coarse_smplx": scaffold_smplx.astype(np.float32),
                "coarse_rot6d": scaffold_rot6d.astype(np.float32),
                "retrieval_adaptive_motion": pred_axis.astype(np.float32),
                "retrieval_adaptive_smplx": pred_smplx.astype(np.float32),
                "retrieval_adaptive_rot6d": pred_rot6d.astype(np.float32),
                "retrieval_features": retrieval_features[
                    local_index, :output_length
                ].cpu().float().numpy(),
                "retrieval_feature_names": np.asarray(RETRIEVAL_FEATURE_NAMES),
                "articulator_confidence": outputs["confidence"][
                    local_index, :output_length
                ].cpu().float().numpy(),
                "articulator_names": np.asarray(ARTICULATOR_NAMES),
                "scale_weights": outputs["scale_weights"][
                    local_index, :output_length
                ].cpu().float().numpy(),
                "code_strides": np.asarray(model.code_strides, dtype=np.int32),
                "correction_gates": outputs["gates"][
                    local_index, :output_length
                ].cpu().float().numpy(),
                "checkpoint": np.asarray(str(args.checkpoint)),
                "checkpoint_epoch": np.asarray(
                    int(checkpoint.get("epoch", -1)), dtype=np.int32
                ),
                "model_type": np.asarray(model_type),
                "scaffold_source": np.asarray(str(scaffold_provider.source)),
                "retrieval_manifest": np.asarray(str(retrieval_bank["manifest"])),
                "length_mode": np.asarray(str(args.length_mode)),
                "ground_truth_length": np.asarray(gt_length, dtype=np.int32),
                "predicted_length": np.asarray(
                    int(predicted_lengths[local_index].item()), dtype=np.int32
                ),
                "predicted_length_continuous": np.asarray(
                    float(continuous_lengths[local_index].item()), dtype=np.float32
                ),
            }
            if "correction_need" in outputs:
                extra.update(
                    {
                        "articulator_trust": outputs["trust"][
                            local_index, :output_length
                        ].cpu().float().numpy(),
                        "articulator_correction_need": outputs["correction_need"][
                            local_index, :output_length
                        ].cpu().float().numpy(),
                        "adaptive_knot_density": outputs["knot_density"][
                            local_index, :output_length
                        ].cpu().float().numpy(),
                        "adaptive_knot_coordinate": outputs["adaptive_coordinates"][
                            local_index, :output_length
                        ].cpu().float().numpy(),
                        "retrieval_uncertainty": outputs["retrieval_uncertainty"][
                            local_index, :output_length
                        ].cpu().float().numpy(),
                        "adaptive_knot_counts": outputs["knot_counts"][
                            local_index
                        ].cpu().numpy().astype(np.int32),
                        "articulator_code_strides": model.articulator_stride_tensor
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32),
                    }
                )
            if "segment_boundary_mask" in outputs:
                extra["segment_boundary_mask"] = outputs[
                    "segment_boundary_mask"
                ][local_index, :output_length].cpu().numpy().astype(np.bool_)
                extra["segment_boundary_stride"] = np.asarray(
                    int(model.segment_boundary_stride), dtype=np.int32
                )
                for scale_index, positions in enumerate(
                    outputs["segment_positions"]
                ):
                    extra[f"segment_positions_scale{scale_index}"] = positions[
                        local_index
                    ].cpu().float().numpy()
                    extra[f"segment_codes_scale{scale_index}"] = outputs[
                        "scale_codes"
                    ][scale_index][local_index].cpu().float().numpy()
                    extra[f"segment_mask_scale{scale_index}"] = outputs[
                        "scale_code_masks"
                    ][scale_index][local_index].cpu().numpy().astype(np.bool_)
            save_eval_npz(
                sample_path,
                pred_axis,
                pred_smplx,
                pred_rot6d,
                meta,
                label=f"niaf_{model_type}",
                extra=extra,
            )
            confidence_mean = outputs["confidence"][
                local_index, :output_length
            ].mean(dim=0)
            rows.append(
                {
                    "index": suffix,
                    "name": meta["name"],
                    "text": meta["text"],
                    "ground_truth_length": gt_length,
                    "predicted_length": int(predicted_lengths[local_index].item()),
                    "output_length": output_length,
                    "confidence_mean": {
                        name: float(confidence_mean[index].item())
                        for index, name in enumerate(ARTICULATOR_NAMES)
                    },
                    "gt": str(gt_path),
                    "sample": str(sample_path),
                }
            )
            sample_counter += 1

    length_errors = [
        abs(row["predicted_length"] - row["ground_truth_length"]) for row in rows
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "length_mode": args.length_mode,
        "scaffold_source": scaffold_provider.source,
        "scaffold_config": scaffold_provider.config_summary,
        "retrieval_bank": retrieval_bank,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "duration_mae_frames": float(sum(length_errors) / max(len(length_errors), 1)),
        "rows": rows,
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "checkpoint_epoch",
                    "split",
                    "length_mode",
                    "num_exported",
                    "duration_mae_frames",
                    "retrieval_bank",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
