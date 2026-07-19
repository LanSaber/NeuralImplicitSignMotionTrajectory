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
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.retrieval_confidence_field.scripts.export_retrieval_adaptive_samples import (
    generation_batch,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    validate_train_only_retrieval_bank,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export arbitrarily sampled continuous trajectory instances."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--length_mode",
        default="predicted",
        choices=["predicted", "ground_truth_sampling", "ground_truth"],
        help=(
            "predicted uses predicted context and output lengths; "
            "ground_truth_sampling keeps the predicted context but queries the "
            "trajectory at the GT frame count; ground_truth also rebuilds the "
            "context at the GT length"
        ),
    )
    parser.add_argument("--context_fps", type=float, default=20.0)
    parser.add_argument("--sample_fps", type=float, nargs="+", default=[20.0, 40.0, 80.0])
    return parser.parse_args()


def sampled_lengths(duration_seconds, fps, min_frames=2, max_frames=4000):
    return torch.round(duration_seconds * float(fps)).long().clamp(
        int(min_frames), int(max_frames)
    )


def resampled_frame_counts(
    reference_lengths,
    reference_fps,
    target_fps,
    min_frames=2,
    max_frames=4000,
):
    intervals = (reference_lengths.long() - 1).clamp_min(1)
    lengths = torch.round(
        intervals.to(torch.float32) * float(target_fps) / float(reference_fps)
    ).long() + 1
    return lengths.clamp(int(min_frames), int(max_frames))


def padded_normalized_grid(lengths, device, dtype):
    max_length = int(lengths.max().item())
    mask = torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    denominator = (lengths - 1).clamp_min(1).to(dtype).unsqueeze(1)
    coordinate = torch.arange(max_length, device=device, dtype=dtype).unsqueeze(0)
    tau = -1.0 + 2.0 * coordinate / denominator
    tau = torch.where(lengths.unsqueeze(1) > 1, tau, torch.zeros_like(tau))
    return tau.clamp(-1.0, 1.0), mask


def _fps_key(fps):
    value = float(fps)
    return f"fps{int(value)}" if value.is_integer() else f"fps{value:g}".replace(".", "p")


def _trajectory_numpy(instance, index):
    output = {}
    for key, value in instance.select(index).detach().tensor_dict().items():
        output[key] = value.squeeze(0).cpu().numpy()
    return output


@torch.no_grad()
def prepare_inference_batch(
    model,
    text_encoder,
    provider,
    batch,
    dataset,
    cfg,
    device,
    context_fps,
    length_mode,
):
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    predicted_log_duration, predicted_duration = model.predict_duration(
        text_tokens, text_mask=text_mask
    )
    duration_cfg = cfg.get("duration", {})
    min_frames = int(duration_cfg.get("min_frames", cfg.get("data", {}).get("min_frames", 40)))
    max_frames = int(duration_cfg.get("max_frames", cfg.get("data", {}).get("max_frames", 400)))
    multiple = int(duration_cfg.get("length_multiple", cfg.get("data", {}).get("length_multiple", 4)))
    predicted_context_lengths = model.predict_lengths(
        text_tokens,
        text_mask=text_mask,
        fps=float(context_fps),
        min_frames=min_frames,
        max_frames=max_frames,
        multiple=multiple,
    )
    context_lengths = (
        batch["length"]
        if length_mode == "ground_truth"
        else predicted_context_lengths
    )
    generated = generation_batch(batch, context_lengths, device)
    adapter_context, _anchors, metadata = provider.build_with_metadata(
        generated,
        x=None,
        use_cache=False,
    )
    trajectory = model.encode_trajectory(
        text_tokens=text_tokens,
        adapter_context=adapter_context,
        context_mask=generated["mask"],
        retrieval_evidence=metadata["retrieval_features"],
        text_mask=text_mask,
    )
    output_duration = (
        trajectory.duration_seconds if length_mode == "predicted" else batch["duration"]
    )
    return {
        "trajectory": trajectory,
        "adapter_context": adapter_context,
        "retrieval_features": metadata["retrieval_features"],
        "context_mask": generated["mask"],
        "context_lengths": context_lengths,
        "predicted_context_lengths": predicted_context_lengths,
        "predicted_duration": predicted_duration,
        "predicted_log_duration": predicted_log_duration,
        "output_duration": output_duration,
    }


@torch.no_grad()
def sample_trajectory_fps(
    model,
    trajectory,
    durations,
    fps_values,
    reference_lengths=None,
    reference_fps=None,
):
    samples = {}
    for fps in fps_values:
        if reference_lengths is None:
            lengths = sampled_lengths(durations, fps)
        else:
            if reference_fps is None:
                raise ValueError("reference_fps is required with reference_lengths")
            lengths = resampled_frame_counts(reference_lengths, reference_fps, fps)
        tau, mask = padded_normalized_grid(
            lengths,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        outputs = model.query_trajectory(
            trajectory,
            tau,
            time_domain="normalized",
            query_mask=mask,
            return_details=True,
        )
        samples[float(fps)] = {
            "lengths": lengths,
            "mask": mask,
            "tau": tau,
            "outputs": outputs,
        }
    return samples


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
    provider = ScaffoldProvider(cfg, dataset, device)
    retrieval_bank = validate_train_only_retrieval_bank(cfg, provider)
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    fps_values = tuple(dict.fromkeys(float(value) for value in args.sample_fps))
    rows = []
    sample_counter = 0
    for batch in tqdm(loader, desc="export continuous trajectories"):
        batch = move_batch_to_device(batch, device)
        target = prepare_motion(batch, dataset, device)
        inference = prepare_inference_batch(
            model,
            text_encoder,
            provider,
            batch,
            dataset,
            cfg,
            device,
            context_fps=args.context_fps,
            length_mode=args.length_mode,
        )
        sampled = sample_trajectory_fps(
            model,
            inference["trajectory"],
            inference["output_duration"],
            fps_values,
            reference_lengths=(
                batch["length"] if args.length_mode != "predicted" else None
            ),
            reference_fps=args.context_fps,
        )
        main_fps = fps_values[0]
        main_sample = sampled[main_fps]

        for local_index in range(len(batch["name"])):
            suffix = f"{sample_counter:04d}"
            gt_length = int(batch["length"][local_index].item())
            output_length = int(main_sample["lengths"][local_index].item())
            context_length = int(inference["context_lengths"][local_index].item())
            meta = {
                "name": batch["name"][local_index],
                "text": batch["text"][local_index],
                "gloss": batch["gloss"][local_index],
                "split": args.split,
                "source_index": sample_counter,
            }
            gt_rot6d, gt_axis, gt_smplx = rot6d_to_axis_and_smplx(
                target[local_index, :gt_length]
            )
            prediction = main_sample["outputs"]["prediction"][local_index, :output_length]
            prior = main_sample["outputs"]["prior"][local_index, :output_length]
            pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(prediction)
            prior_rot6d, prior_axis, prior_smplx = rot6d_to_axis_and_smplx(prior)
            context_rot6d, context_axis, context_smplx = rot6d_to_axis_and_smplx(
                inference["adapter_context"][local_index, :context_length]
            )
            gt_path = out_dir / f"gt_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_eval_npz(
                gt_path, gt_axis, gt_smplx, gt_rot6d, meta, label="ground_truth"
            )
            extra = {
                "continuous_prior_motion": prior_axis.astype(np.float32),
                "continuous_prior_smplx": prior_smplx.astype(np.float32),
                "continuous_prior_rot6d": prior_rot6d.astype(np.float32),
                "adapter_context_motion": context_axis.astype(np.float32),
                "adapter_context_smplx": context_smplx.astype(np.float32),
                "adapter_context_rot6d": context_rot6d.astype(np.float32),
                "retrieval_features": inference["retrieval_features"][
                    local_index, :context_length
                ].cpu().float().numpy(),
                "checkpoint": np.asarray(str(args.checkpoint)),
                "checkpoint_epoch": np.asarray(
                    int(checkpoint.get("epoch", -1)), dtype=np.int32
                ),
                "model_type": np.asarray("continuous_trajectory_field"),
                "trajectory_contract_version": np.asarray(1, dtype=np.int32),
                "length_mode": np.asarray(args.length_mode),
                "context_fps": np.asarray(float(args.context_fps), dtype=np.float32),
                "sample_fps": np.asarray(float(main_fps), dtype=np.float32),
                "ground_truth_length": np.asarray(gt_length, dtype=np.int32),
                "context_length": np.asarray(context_length, dtype=np.int32),
                "output_length": np.asarray(output_length, dtype=np.int32),
                "predicted_duration_seconds": np.asarray(
                    float(inference["trajectory"].duration_seconds[local_index].item()),
                    dtype=np.float32,
                ),
            }
            extra.update(_trajectory_numpy(inference["trajectory"], local_index))
            for fps, fps_sample in sampled.items():
                fps_length = int(fps_sample["lengths"][local_index].item())
                fps_prediction = fps_sample["outputs"]["prediction"][
                    local_index, :fps_length
                ]
                fps_rot6d, fps_axis, fps_smplx = rot6d_to_axis_and_smplx(fps_prediction)
                key = _fps_key(fps)
                extra[f"continuous_{key}_rot6d"] = fps_rot6d.astype(np.float32)
                extra[f"continuous_{key}_motion"] = fps_axis.astype(np.float32)
                extra[f"continuous_{key}_smplx"] = fps_smplx.astype(np.float32)
                extra[f"continuous_{key}_tau"] = fps_sample["tau"][
                    local_index, :fps_length
                ].cpu().float().numpy()
            save_eval_npz(
                sample_path,
                pred_axis,
                pred_smplx,
                pred_rot6d,
                meta,
                label="niaf_continuous_trajectory_field",
                extra=extra,
            )
            rows.append(
                {
                    "index": suffix,
                    "name": meta["name"],
                    "text": meta["text"],
                    "ground_truth_length": gt_length,
                    "ground_truth_duration_seconds": float(
                        batch["duration"][local_index].item()
                    ),
                    "context_length": context_length,
                    "predicted_duration_seconds": float(
                        inference["trajectory"].duration_seconds[local_index].item()
                    ),
                    "sample_lengths": {
                        _fps_key(fps): int(value["lengths"][local_index].item())
                        for fps, value in sampled.items()
                    },
                    "gt": str(gt_path),
                    "sample": str(sample_path),
                }
            )
            sample_counter += 1

    duration_errors = [
        abs(
            row["predicted_duration_seconds"]
            - row["ground_truth_duration_seconds"]
        )
        for row in rows
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "length_mode": args.length_mode,
        "context_fps": float(args.context_fps),
        "sample_fps": list(fps_values),
        "retrieval_bank": retrieval_bank,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "duration_mae_seconds": float(sum(duration_errors) / max(len(duration_errors), 1)),
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
                    "sample_fps",
                    "num_exported",
                    "duration_mae_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
