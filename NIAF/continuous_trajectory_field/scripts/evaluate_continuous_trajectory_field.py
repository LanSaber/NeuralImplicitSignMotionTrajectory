from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flow.distributed import (
    barrier,
    cleanup_distributed,
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
    bind_sentence_memory_validation_dataset,
    build_sentence_memory_provider,
    build_development_validation_loader,
    build_isolated_development_validation_loader,
    checkpoint_selection_diagnostics,
    configured_sentence_memory_eval_modes,
    configured_selection_aggregation,
    CLUSTER_EQUAL_SELECTION_AGGREGATION,
    distributed_validation_metrics,
    evaluate,
    evaluate_configured_modes,
    evaluated_loader_sample_count,
    is_dual_mode,
    isolated_development_validation_config,
    is_sentence_memory_model,
    load_isolated_development_validation_runtime,
    paired_sentence_memory_corruption_config,
    sentence_memory_enabled,
    sentence_memory_architecture_identity,
    sentence_memory_evaluation_control_identity,
    sentence_memory_provider_required,
    sentence_memory_selection_aggregation_identity,
    selection_diagnostics,
    set_seed,
    set_sentence_memory_provider_epoch_from_checkpoint,
    validate_checkpoint_contract,
    validate_sentence_memory_checkpoint_identity,
    validate_sentence_memory_architecture_identity,
    validate_sentence_memory_evaluation_control_identity,
    validate_sentence_memory_selection_aggregation_identity,
    validate_sentence_memory_validation_corruption_map_identity,
    requires_isolated_development_validation,
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
    parser.add_argument(
        "--word_prior",
        dest="word_prior_mode",
        default="auto",
        choices=("auto", "off", "on", "both"),
        help=(
            "For v2, evaluate text-only (off), prior-enabled (on), or both. "
            "auto means both for v2 and the original path for v1."
        ),
    )
    parser.add_argument(
        "--sentence_memory",
        dest="sentence_memory_mode",
        default="auto",
        choices=(
            "auto",
            "off",
            "on",
            "both",
            "shuffled",
            "motion_shuffled",
            "analytic_prior",
        ),
        help=(
            "For v3, evaluate configured modes (auto), off, on, off+on "
            "(both), a deterministic full- or motion-only-shuffled control, "
            "or analytic-prior attention."
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


def reduce_external_evaluation_metrics(
    metrics,
    loader,
    max_batches,
    device,
    dist_info,
):
    """Reduce ordinary means and paired Rmotion moments for standalone eval."""

    return distributed_validation_metrics(
        metrics,
        evaluated_loader_sample_count(loader, max_batches),
        device,
        dist_info,
    )


def build_public_evaluation_loader(cfg, *, split, limit, dist_info):
    """Build the public evaluator's query loader without exposing the holdout.

    Active factorized paired experiments are development-only here. Confirmation
    remains available exclusively through the post-lock exporter.
    """

    development_runtime = None
    loader_cfg = cfg
    if requires_isolated_development_validation(cfg):
        validation_split = str(cfg.get("data", {}).get("val_split", "val"))
        if str(split) != validation_split:
            raise ValueError(
                "Factorized paired evaluation is restricted to the sealed "
                f"development partition of split {validation_split!r}; use the "
                "post-lock confirmation exporter for confirmation data"
            )
        if int(limit) != 0:
            raise ValueError(
                "Factorized paired evaluation requires the complete sealed "
                "development manifest; --limit is not permitted"
            )
        development_runtime = load_isolated_development_validation_runtime(cfg)
        loader_cfg = isolated_development_validation_config(
            cfg, development_runtime
        )

    dataset, loader, sampler = make_loader(
        loader_cfg,
        split,
        limit=max(int(limit), 0),
        shuffle=False,
        distributed=dist_info["enabled"],
        world_size=dist_info["world_size"],
        # The provider below may initialize CUDA and transformer worker
        # threads before this loader is first iterated. Avoid a late fork,
        # which can leave every worker blocked on an inherited lock.
        num_workers=int(cfg.get("eval", {}).get("num_workers", 0)),
    )
    return dataset, loader, sampler, development_runtime


def bind_public_evaluation_development_loader(
    cfg,
    dataset,
    loader,
    sentence_memory_provider,
    dist_info,
    development_runtime,
):
    """Bind exact canonical neighbor rows and cluster the sealed dev loader."""

    if development_runtime is None:
        return loader, None, None
    if sentence_memory_provider is None:
        raise RuntimeError(
            "Factorized development evaluation requires a sentence-memory "
            "provider for exact canonical neighbor-table binding"
        )
    query_binding = bind_sentence_memory_validation_dataset(
        cfg, sentence_memory_provider, dataset, development_runtime
    )
    development_loader, sampler, _runtime = (
        build_isolated_development_validation_loader(
            cfg,
            dataset,
            loader,
            sentence_memory_provider,
            dist_info,
            development_runtime,
        )
    )
    return development_loader, sampler, query_binding


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
        dual_mode = is_dual_mode(cfg)
        sentence_memory_model = is_sentence_memory_model(cfg)
        if not dual_mode and args.word_prior_mode != "auto":
            raise ValueError("--word_prior is available only for dual-mode v2")
        if not sentence_memory_model and args.sentence_memory_mode != "auto":
            raise ValueError(
                "--sentence_memory is available only for the v3 sentence-memory model"
            )
        if sentence_memory_model:
            memory_enabled = sentence_memory_enabled(cfg)
            resolved_word_prior_mode = (
                str(
                    cfg.get("eval", {}).get(
                        "sentence_memory_word_prior_mode", "off"
                    )
                ).lower()
                if args.word_prior_mode == "auto"
                else args.word_prior_mode
            )
            if resolved_word_prior_mode not in {"off", "on"}:
                raise ValueError(
                    "v3 evaluation uses one fixed --word_prior mode ('off' or 'on')"
                )
            resolved_sentence_memory_mode = (
                args.sentence_memory_mode if memory_enabled else "off"
            )
            if memory_enabled and resolved_sentence_memory_mode == "both":
                cfg.setdefault("eval", {})["sentence_memory_modes"] = ["off", "on"]
            cfg.setdefault("eval", {})[
                "sentence_memory_word_prior_mode"
            ] = resolved_word_prior_mode
            sentence_modes = (
                ("off",)
                if not memory_enabled
                else (
                    configured_sentence_memory_eval_modes(cfg)
                    if resolved_sentence_memory_mode in {"auto", "both"}
                    else (resolved_sentence_memory_mode,)
                )
            )
        else:
            resolved_word_prior_mode = (
                "both" if dual_mode and args.word_prior_mode == "auto"
                else args.word_prior_mode
            )
            resolved_sentence_memory_mode = "not_applicable"
            sentence_modes = ()
        needs_provider = not dual_mode or resolved_word_prior_mode in {"on", "both"}
        train_dataset = None
        if needs_provider:
            train_dataset, _train_loader, _train_sampler = make_loader(
                cfg,
                data_cfg.get("train_split", "train"),
                limit=0,
                shuffle=False,
                distributed=False,
            )
        (
            eval_dataset,
            eval_loader,
            _eval_sampler,
            development_runtime,
        ) = build_public_evaluation_loader(
            cfg,
            split=args.split,
            limit=args.limit,
            dist_info=dist_info,
        )
        rank_zero_print(
            dist_info,
            f"Evaluating split={args.split} examples={len(eval_dataset)} "
            f"world_size={dist_info['world_size']} "
            f"batch_per_rank={cfg.get('train', {}).get('eval_batch_size', 1)} "
            f"workers={cfg.get('eval', {}).get('num_workers', 0)}",
        )

        text_encoder = build_text_encoder(cfg, text_device)
        provider = (
            ScaffoldProvider(cfg, train_dataset, device) if needs_provider else None
        )
        sentence_memory_provider = (
            build_sentence_memory_provider(cfg, text_encoder, dataset=eval_dataset)
            if sentence_memory_model
            and (
                development_runtime is not None
                or sentence_memory_provider_required(sentence_modes)
            )
            else None
        )
        development_query_binding = None
        if development_runtime is not None:
            (
                eval_loader,
                _eval_sampler,
                development_query_binding,
            ) = bind_public_evaluation_development_loader(
                cfg,
                eval_dataset,
                eval_loader,
                sentence_memory_provider,
                dist_info,
                development_runtime,
            )
        elif (
            sentence_memory_model
            and resolved_sentence_memory_mode in {"auto", "both"}
            and paired_sentence_memory_corruption_config(cfg)["enabled"]
            and configured_selection_aggregation(cfg)
            == CLUSTER_EQUAL_SELECTION_AGGREGATION
        ):
            validation_split = str(data_cfg.get("val_split", "val"))
            if str(args.split) != validation_split:
                raise ValueError(
                    "Cluster-equal checkpoint selection is defined only for the "
                    f"development partition of split {validation_split!r}; use "
                    "the confirmation exporter for post-lock analysis"
                )
            eval_loader, _eval_sampler, _partition = (
                build_development_validation_loader(
                    cfg,
                    eval_dataset,
                    eval_loader,
                    sentence_memory_provider,
                    dist_info,
                )
            )
        if sentence_memory_provider is not None and development_runtime is None:
            sentence_memory_provider.validate_query_dataset(
                eval_dataset,
                require_neighbors=any(mode != "off" for mode in sentence_modes),
            )
        retrieval_bank = (
            validate_train_only_retrieval_bank(cfg, provider)
            if provider is not None
            else None
        )
        model = build_continuous_trajectory_field(
            cfg, text_dim=text_encoder.text_dim
        ).to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        validate_checkpoint_contract(checkpoint, cfg, source=str(args.checkpoint))
        validate_sentence_memory_checkpoint_identity(
            checkpoint,
            sentence_memory_provider,
            source=str(args.checkpoint),
            cfg=cfg,
            text_encoder_identity=(
                text_encoder.checkpoint_identity()
                if hasattr(text_encoder, "checkpoint_identity")
                else None
            ),
        )
        if sentence_memory_model:
            validate_sentence_memory_architecture_identity(
                checkpoint, cfg, source=str(args.checkpoint)
            )
            validate_sentence_memory_evaluation_control_identity(
                checkpoint, cfg, source=str(args.checkpoint)
            )
            validate_sentence_memory_selection_aggregation_identity(
                checkpoint, cfg, source=str(args.checkpoint)
            )
            if cfg.get("validation_text_partition", {}).get(
                "evaluation_corruption_map_identity"
            ) is not None:
                validate_sentence_memory_validation_corruption_map_identity(
                    checkpoint, cfg, source=str(args.checkpoint)
                )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        fk = build_fk(cfg, device)

        checkpoint_epoch = set_sentence_memory_provider_epoch_from_checkpoint(
            sentence_memory_provider, checkpoint
        )
        evaluate_both = (
            sentence_memory_model
            and resolved_sentence_memory_mode in {"auto", "both"}
        ) or (
            not sentence_memory_model
            and dual_mode
            and resolved_word_prior_mode == "both"
        )
        evaluation_function = evaluate_configured_modes if evaluate_both else evaluate
        evaluation_kwargs = {}
        if sentence_memory_model:
            evaluation_kwargs["sentence_memory_provider"] = sentence_memory_provider
            if not evaluate_both:
                evaluation_kwargs["word_prior_mode"] = resolved_word_prior_mode
                evaluation_kwargs["sentence_memory_mode"] = sentence_modes[0]
        elif dual_mode and not evaluate_both:
            evaluation_kwargs["word_prior_mode"] = resolved_word_prior_mode
        metrics = evaluation_function(
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
            **evaluation_kwargs,
        )
        metrics = reduce_external_evaluation_metrics(
            metrics,
            eval_loader,
            args.max_batches,
            device,
            dist_info,
        )
        diagnostics_function = (
            checkpoint_selection_diagnostics if evaluate_both else selection_diagnostics
        )
        score, constraint_violation, feasible, selection_details = diagnostics_function(
            metrics,
            cfg,
            return_details=True,
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
            "word_prior_mode": resolved_word_prior_mode if dual_mode else "v1_required",
            "sentence_memory_mode": resolved_sentence_memory_mode,
            "scaffold_mode": str(args.scaffold_mode),
            "scaffold": (
                provider.config_summary if provider is not None else None
            ),
            "retrieval_bank": retrieval_bank,
            "sentence_memory": (
                getattr(sentence_memory_provider, "config_summary", None)
                if sentence_memory_provider is not None
                else None
            ),
            "development_validation_runtime": (
                development_runtime.artifact_payload
                if development_runtime is not None
                else None
            ),
            "development_validation_query_binding": development_query_binding,
            "sentence_memory_evaluation_control_identity": (
                sentence_memory_evaluation_control_identity(cfg)
                if sentence_memory_model
                else None
            ),
            "sentence_memory_architecture_identity": (
                sentence_memory_architecture_identity(cfg)
                if sentence_memory_model
                else None
            ),
            "sentence_memory_selection_aggregation_identity": (
                sentence_memory_selection_aggregation_identity(cfg)
                if sentence_memory_model
                else None
            ),
            "sentence_memory_validation_corruption_map_identity": (
                cfg.get("validation_text_partition", {}).get(
                    "evaluation_corruption_map_identity"
                )
                if sentence_memory_model
                else None
            ),
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
