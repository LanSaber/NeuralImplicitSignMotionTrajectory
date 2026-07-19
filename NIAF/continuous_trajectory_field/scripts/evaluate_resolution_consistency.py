from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import ContinuousSignDataset, collate_continuous_sign
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_eval_samples import select_manifest
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_text_encoder,
    move_batch_to_device,
)
from NIAF.continuous_trajectory_field.derivatives import sample_padded_sequence
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory import (
    padded_normalized_grid,
    prepare_inference_batch,
    sampled_lengths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that one trajectory instance is invariant to query resolution."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--context_fps", type=float, default=20.0)
    parser.add_argument("--sample_fps", type=float, nargs="+", default=[20.0, 40.0, 80.0])
    parser.add_argument("--common_queries", type=int, default=31)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def trajectory_digest(trajectory):
    digest = hashlib.sha256()
    for key, value in sorted(trajectory.detach().tensor_dict().items()):
        array = value.cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("scaffold", {})["cache_only"] = False
    cfg.setdefault("scaffold", {})["prefer_cache"] = False
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_manifest, _rows, manifest_summary = select_manifest(
        cfg,
        args.split,
        out_path.parent,
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
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    rows = []
    common_count = max(int(args.common_queries), 3)
    for batch in tqdm(loader, desc="resolution consistency"):
        batch = move_batch_to_device(batch, device)
        inference = prepare_inference_batch(
            model,
            text_encoder,
            provider,
            batch,
            dataset,
            cfg,
            device,
            context_fps=args.context_fps,
            length_mode="predicted",
        )
        trajectory = inference["trajectory"]
        before_digest = trajectory_digest(trajectory)
        common_tau = torch.linspace(
            -1.0,
            1.0,
            common_count,
            device=device,
            dtype=trajectory.dtype,
        ).unsqueeze(0).expand(trajectory.batch_size, -1)
        direct = model.query_trajectory(trajectory, common_tau)
        batch_metrics = {}
        for fps in args.sample_fps:
            lengths = sampled_lengths(inference["output_duration"], fps)
            grid_tau, grid_mask = padded_normalized_grid(
                lengths, trajectory.device, trajectory.dtype
            )
            merged_tau = torch.cat([grid_tau, common_tau], dim=1)
            merged = model.query_trajectory(trajectory, merged_tau)
            embedded = merged[:, -common_count:]
            difference = torch.abs(direct - embedded)
            grid_prediction = model.query_trajectory(
                trajectory,
                grid_tau,
                query_mask=grid_mask,
            )
            interpolated = sample_padded_sequence(grid_prediction, common_tau, lengths)
            interpolation_difference = torch.abs(direct - interpolated)
            key = f"fps{float(fps):g}"
            batch_metrics[key] = {
                "shared_time_max_abs": float(difference.max().cpu().item()),
                "shared_time_mean_abs": float(difference.mean().cpu().item()),
                "sample_interpolation_mae": float(
                    interpolation_difference.mean().cpu().item()
                ),
            }
        permutation = torch.randperm(common_count, device=device)
        permuted = model.query_trajectory(trajectory, common_tau[:, permutation])
        inverse = torch.argsort(permutation)
        order_error = torch.abs(direct - permuted[:, inverse])
        after_digest = trajectory_digest(trajectory)
        for local_index, name in enumerate(batch["name"]):
            rows.append(
                {
                    "name": name,
                    "parameter_digest_before": before_digest,
                    "parameter_digest_after": after_digest,
                    "parameters_unchanged": before_digest == after_digest,
                    "query_order_max_abs": float(
                        order_error[local_index].max().cpu().item()
                    ),
                    "fps": batch_metrics,
                }
            )

    maximum_shared_error = max(
        metric["shared_time_max_abs"]
        for row in rows
        for metric in row["fps"].values()
    ) if rows else 0.0
    maximum_order_error = max(
        row["query_order_max_abs"] for row in rows
    ) if rows else 0.0
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "manifest": manifest_summary,
        "num_samples": len(rows),
        "sample_fps": [float(value) for value in args.sample_fps],
        "tolerance": float(args.tolerance),
        "maximum_shared_time_error": maximum_shared_error,
        "maximum_query_order_error": maximum_order_error,
        "all_parameters_unchanged": all(row["parameters_unchanged"] for row in rows),
        "passed": bool(
            maximum_shared_error <= float(args.tolerance)
            and maximum_order_error <= float(args.tolerance)
            and all(row["parameters_unchanged"] for row in rows)
        ),
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
