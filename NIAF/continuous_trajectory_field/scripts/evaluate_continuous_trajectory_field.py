from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flow.distributed import (
    barrier,
    cleanup_distributed,
    distributed_mean_scalars,
    rank_zero_print,
    resolve_device as resolve_distributed_device,
    setup_distributed,
)
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.train_residual_flow import (
    build_fk,
    build_text_encoder,
    make_loader,
)
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    evaluate,
    selection_diagnostics,
    set_seed,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    validate_train_only_retrieval_bank,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a continuous trajectory field checkpoint without training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--scaffold_mode",
        default="config",
        choices=("config", "cache", "fallback", "online"),
        help=(
            "Use config behavior, require cached scaffolds, prefer cache with "
            "online fallback, or build every scaffold online."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_device", default=None)
    parser.add_argument(
        "--distributed", default="auto", choices=("auto", "none", "ddp")
    )
    parser.add_argument(
        "--ddp_backend",
        "--ddp-backend",
        dest="ddp_backend",
        default="auto",
        choices=("auto", "nccl", "gloo"),
    )
    parser.add_argument(
        "--local_rank", "--local-rank", dest="local_rank", type=int, default=None
    )
    parser.add_argument(
        "--ddp_timeout_min",
        "--ddp-timeout-min",
        dest="ddp_timeout_min",
        type=int,
        default=60,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["eval_batch_size"] = int(args.batch_size)
    if args.device is not None:
        cfg["device"] = args.device
    if args.text_device is not None:
        cfg.setdefault("text", {})["device"] = args.text_device
    cfg.setdefault("data", {})["random_crop"] = False
    scaffold_cfg = cfg.setdefault("scaffold", {})
    if args.scaffold_mode == "online":
        scaffold_cfg["cache_only"] = False
        scaffold_cfg["prefer_cache"] = False
    elif args.scaffold_mode == "fallback":
        scaffold_cfg["cache_only"] = False
        scaffold_cfg["prefer_cache"] = True
    elif args.scaffold_mode == "cache":
        scaffold_cfg["cache_only"] = True
        scaffold_cfg["prefer_cache"] = True

    dist_info = setup_distributed(args)
    try:
        set_seed(int(cfg.get("seed", 1234)) + int(dist_info.get("rank", 0)))
        device = resolve_distributed_device(cfg.get("device", "auto"), dist_info)
        text_device = torch.device(cfg.get("text", {}).get("device", "cpu"))
        data_cfg = cfg.get("data", {})

        train_dataset, _train_loader, _train_sampler = make_loader(
            cfg,
            data_cfg.get("train_split", "train"),
            limit=0,
            shuffle=False,
            distributed=False,
        )
        eval_dataset, eval_loader, _eval_sampler = make_loader(
            cfg,
            args.split,
            limit=max(int(args.limit), 0),
            shuffle=False,
            distributed=dist_info["enabled"],
            world_size=dist_info["world_size"],
            # The provider below may initialize CUDA and transformer worker
            # threads before this loader is first iterated. Avoid a late fork,
            # which can leave every worker blocked on an inherited lock.
            num_workers=int(cfg.get("eval", {}).get("num_workers", 0)),
        )
        rank_zero_print(
            dist_info,
            f"Evaluating split={args.split} examples={len(eval_dataset)} "
            f"world_size={dist_info['world_size']} "
            f"batch_per_rank={cfg.get('train', {}).get('eval_batch_size', 1)} "
            f"workers={cfg.get('eval', {}).get('num_workers', 0)}",
        )

        text_encoder = build_text_encoder(cfg, text_device)
        provider = ScaffoldProvider(cfg, train_dataset, device)
        retrieval_bank = validate_train_only_retrieval_bank(cfg, provider)
        model = build_continuous_trajectory_field(
            cfg, text_dim=text_encoder.text_dim
        ).to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        fk = build_fk(cfg, device)

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        metrics = evaluate(
            model,
            fk,
            text_encoder,
            provider,
            eval_loader,
            eval_dataset,
            cfg,
            device,
            epoch=max(checkpoint_epoch, 1),
            max_batches=max(int(args.max_batches), 0),
            show_progress=dist_info["is_main"],
        )
        metrics = distributed_mean_scalars(metrics, device, dist_info)
        score, constraint_violation, feasible, selection_details = selection_diagnostics(
            metrics, cfg, return_details=True
        )
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
            "split": str(args.split),
            "dataset_examples": len(eval_dataset),
            "world_size": int(dist_info["world_size"]),
            "batch_size_per_rank": int(
                cfg.get("train", {}).get("eval_batch_size", 1)
            ),
            "max_batches_per_rank": max(int(args.max_batches), 0),
            "scaffold_mode": str(args.scaffold_mode),
            "scaffold": provider.config_summary,
            "retrieval_bank": retrieval_bank,
            "selection_score": float(score),
            "selection_constraint_violation": float(constraint_violation),
            "selection_feasible": bool(feasible),
            "selection_details": selection_details,
            "metrics": metrics,
        }
        if dist_info["is_main"]:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, sort_keys=True))
        barrier(dist_info)
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
