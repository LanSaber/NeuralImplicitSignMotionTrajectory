from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_sign_field.data import (
    ContinuousSignDataset,
    collate_continuous_sign,
)
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_eval_samples import (
    read_jsonl,
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
from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryBatch,
    motion_only_shuffle_sentence_memory_batch,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    _sentence_memory_field,
    build_sentence_memory_provider,
    checkpoint_contract,
    is_dual_mode,
    is_sentence_memory_model,
    retrieve_sentence_memory,
    sentence_memory_enabled,
    sentence_memory_forward_kwargs,
    sentence_memory_provider_required,
    sentence_memory_query_ids,
    set_sentence_memory_provider_epoch_from_checkpoint,
    validate_checkpoint_contract,
    validate_sentence_memory_checkpoint_identity,
)
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of rows to export; 0 exports the complete split (default).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--selection_mode",
        default="random",
        choices=("random", "first"),
        help="Choose random manifest rows or the first rows used by limit_train.",
    )
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
    parser.add_argument(
        "--word_prior",
        default="auto",
        choices=("auto", "off", "on"),
        help=(
            "For v2, export text-only (off) or prior-enabled (on). auto uses "
            "text-only for v2 and the required scaffold path for v1."
        ),
    )
    parser.add_argument(
        "--sentence_memory",
        default="auto",
        choices=(
            "auto",
            "off",
            "on",
            "shuffled",
            "motion_shuffled",
            "all_null",
        ),
        help=(
            "For v3, export with memory off, retrieved memory on, or the "
            "deterministic full- or motion-only-shuffled controls, or the "
            "all-null integrity control. auto uses the configured export mode "
            "(on by default)."
        ),
    )
    parser.add_argument("--context_fps", type=float, default=20.0)
    parser.add_argument(
        "--sample_fps", type=float, nargs="+", default=[20.0, 40.0, 80.0]
    )
    return parser.parse_args()


def sampled_lengths(duration_seconds, fps, min_frames=2, max_frames=4000):
    return (
        torch.round(duration_seconds * float(fps))
        .long()
        .clamp(int(min_frames), int(max_frames))
    )


def resampled_frame_counts(
    reference_lengths,
    reference_fps,
    target_fps,
    min_frames=2,
    max_frames=4000,
):
    intervals = (reference_lengths.long() - 1).clamp_min(1)
    lengths = (
        torch.round(
            intervals.to(torch.float32) * float(target_fps) / float(reference_fps)
        ).long()
        + 1
    )
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
    return (
        f"fps{int(value)}" if value.is_integer() else f"fps{value:g}".replace(".", "p")
    )


def _trajectory_numpy(instance, index):
    output = {}
    for key, value in instance.select(index).detach().tensor_dict().items():
        output[key] = value.squeeze(0).cpu().numpy()
    return output


def _memory_row_numpy(memory_batch, field, index, *aliases):
    value = _sentence_memory_field(memory_batch, field, *aliases)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[index].detach().cpu().numpy()
    row = value[index]
    if torch.is_tensor(row):
        return row.detach().cpu().numpy()
    return np.asarray(row)


def sentence_memory_diagnostics_row(instance, index):
    """Return compact per-sample gate/null summaries for paired reports."""

    gates = getattr(instance, "sentence_memory_gates", None)
    null_mass = getattr(instance, "sentence_memory_null_mass", None)
    if gates is None or null_mass is None:
        return None
    row_gates = gates[int(index)].detach().float()
    row_null = null_mass[int(index)].detach().float()
    if row_gates.ndim != 2 or row_gates.shape[-1] != 4 or row_null.ndim != 1:
        raise ValueError("Malformed sentence-memory trajectory diagnostics")
    part_names = ("body", "lhand", "rhand", "face")
    available = getattr(instance, "sentence_memory_available", None)
    candidate_mass = getattr(instance, "sentence_memory_candidate_mass", None)
    candidate_mass_mean = None
    if candidate_mass is not None:
        row_candidate_mass = candidate_mass[int(index)].detach().float()
        if row_candidate_mass.ndim != 2:
            raise ValueError("Malformed sentence-memory candidate attention mass")
        candidate_mass_mean = float(row_candidate_mass.sum(dim=-1).mean().item())
    return {
        "available": bool(
            available is not None and available[int(index)].detach().bool().item()
        ),
        "gate_mean_by_part": {
            name: float(row_gates[:, part_index].mean().item())
            for part_index, name in enumerate(part_names)
        },
        "null_mass_mean": float(row_null.mean().item()),
        "candidate_mass_mean": candidate_mass_mean,
    }


def sentence_memory_text_subset(memory_batch, provider, index, text):
    """Classify a query only when the train-bank vocabulary is available."""

    provenance = (
        getattr(memory_batch, "provenance", {}) if memory_batch is not None else {}
    )
    seen_rows = (provenance or {}).get("query_seen_text", [])
    if index < len(seen_rows):
        return "exact_seen_text" if bool(seen_rows[index]) else "novel_text"
    if provider is not None:
        return "exact_seen_text" if provider.is_seen_text(text) else "novel_text"
    return "not_available"


def all_null_sentence_memory_batch(batch, cfg, device):
    """Create an enabled, candidate-free memory input without payload reads."""

    batch_size = len(batch.get("name", ()))
    candidate_count = int(cfg.get("sentence_memory", {}).get("k", 8))
    motion_dim = int(cfg.get("sentence_memory", {}).get("motion_dim", 256))
    key_dim = int(cfg.get("sentence_memory", {}).get("key_dim", 768))
    if batch_size < 1 or candidate_count < 1 or motion_dim < 1 or key_dim < 1:
        raise ValueError("Invalid all-null sentence-memory shape")
    return SentenceMemoryBatch(
        tokens=torch.zeros(batch_size, candidate_count, 1, motion_dim, device=device),
        token_mask=torch.zeros(
            batch_size, candidate_count, 1, dtype=torch.bool, device=device
        ),
        token_tau=torch.zeros(batch_size, candidate_count, 1, device=device),
        candidate_mask=torch.zeros(
            batch_size, candidate_count, dtype=torch.bool, device=device
        ),
        candidate_keys=torch.zeros(
            batch_size, candidate_count, key_dim, device=device
        ),
        scores=torch.zeros(batch_size, candidate_count, device=device),
        durations=torch.zeros(batch_size, candidate_count, device=device),
        duration_log_gap=torch.zeros(batch_size, candidate_count, device=device),
        part_validity=torch.zeros(
            batch_size, candidate_count, 1, 4, device=device
        ),
        ids=torch.full(
            (batch_size, candidate_count), -1, dtype=torch.int64, device=device
        ),
        available=torch.ones(batch_size, dtype=torch.bool, device=device),
        provenance={
            "mode": "all_null",
            "payload_reads": 0,
            "query_seen_text": [False] * batch_size,
            "candidate_group_ids": [[-1] * candidate_count for _ in range(batch_size)],
            "candidate_exact_text": [
                [False] * candidate_count for _ in range(batch_size)
            ],
        },
    )


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
    word_prior_mode="auto",
    sentence_memory_provider=None,
    sentence_memory_mode="auto",
    sentence_memory_epoch=0,
):
    text_tokens, text_mask = encode_batch_text(text_encoder, batch, cfg, device)
    predicted_log_duration, predicted_duration = model.predict_duration(
        text_tokens, text_mask=text_mask
    )
    duration_cfg = cfg.get("duration", {})
    min_frames = int(
        duration_cfg.get("min_frames", cfg.get("data", {}).get("min_frames", 40))
    )
    max_frames = int(
        duration_cfg.get("max_frames", cfg.get("data", {}).get("max_frames", 400))
    )
    multiple = int(
        duration_cfg.get(
            "length_multiple", cfg.get("data", {}).get("length_multiple", 4)
        )
    )
    predicted_context_lengths = model.predict_lengths(
        text_tokens,
        text_mask=text_mask,
        fps=float(context_fps),
        min_frames=min_frames,
        max_frames=max_frames,
        multiple=multiple,
    )
    context_lengths = (
        batch["length"] if length_mode == "ground_truth" else predicted_context_lengths
    )
    dual_mode = is_dual_mode(cfg)
    sentence_memory_model = is_sentence_memory_model(cfg)
    if not dual_mode and word_prior_mode != "auto":
        raise ValueError("word_prior_mode is available only for dual-mode v2")
    resolved_mode = (
        str(cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off"))
        if sentence_memory_model and word_prior_mode == "auto"
        else ("off" if dual_mode and word_prior_mode == "auto" else word_prior_mode)
    )
    if dual_mode and resolved_mode not in {"off", "on"}:
        raise ValueError("word_prior_mode must be 'off' or 'on'")
    if not sentence_memory_model and sentence_memory_mode != "auto":
        raise ValueError("sentence_memory_mode is available only for v3")
    resolved_sentence_mode = (
        "off"
        if sentence_memory_model and not sentence_memory_enabled(cfg)
        else (
            str(cfg.get("eval", {}).get("export_sentence_memory_mode", "on"))
            if sentence_memory_model and sentence_memory_mode == "auto"
            else sentence_memory_mode
        )
    )
    if sentence_memory_model and resolved_sentence_mode not in {
        "off",
        "on",
        "shuffled",
        "motion_shuffled",
        "all_null",
    }:
        raise ValueError(
            "v3 sentence_memory_mode must be 'off', 'on', 'shuffled', or "
            "'motion_shuffled', or 'all_null'"
        )

    adapter_context = None
    retrieval_features = None
    context_mask = None
    word_kwargs = {}
    if dual_mode and resolved_mode == "off":
        context_lengths = torch.zeros_like(batch["length"])
        word_kwargs = {
            "word_prior_available": torch.zeros(
                text_tokens.shape[0], dtype=torch.bool, device=device
            )
        }
    else:
        if provider is None:
            raise ValueError("Prior-enabled inference requires a ScaffoldProvider")
        generated = generation_batch(batch, context_lengths, device)
        adapter_context, _anchors, metadata = provider.build_with_metadata(
            generated,
            x=None,
            use_cache=False,
        )
        retrieval_features = metadata["retrieval_features"]
        context_mask = generated["mask"]
        if dual_mode:
            word_kwargs = {
                "word_prior_context": adapter_context,
                "word_prior_mask": context_mask,
                "word_prior_features": retrieval_features,
                "word_prior_available": torch.ones(
                    text_tokens.shape[0], dtype=torch.bool, device=device
                ),
            }
        else:
            trajectory = model.encode_trajectory(
                text_tokens=text_tokens,
                adapter_context=adapter_context,
                context_mask=context_mask,
                retrieval_evidence=retrieval_features,
                text_mask=text_mask,
            )
    sentence_memory_batch = None
    sentence_kwargs = {}
    if sentence_memory_model:
        sentence_available = torch.full(
            (text_tokens.shape[0],),
            resolved_sentence_mode != "off",
            dtype=torch.bool,
            device=device,
        )
        sentence_kwargs = {"sentence_memory_available": sentence_available}
        if bool(sentence_available.any()):
            if resolved_sentence_mode == "all_null":
                sentence_memory_batch = all_null_sentence_memory_batch(
                    batch, cfg, device
                )
            else:
                if sentence_memory_provider is None:
                    raise ValueError(
                        "Retrieval-backed inference requires a "
                        "SentenceMemoryProvider"
                    )
                sentence_memory_batch = retrieve_sentence_memory(
                    sentence_memory_provider,
                    dataset=dataset,
                    batch=batch,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    predicted_duration=predicted_duration.detach(),
                    available=sentence_available,
                    training=False,
                    device=device,
                    mode=(
                        "on"
                        if resolved_sentence_mode == "motion_shuffled"
                        else resolved_sentence_mode
                    ),
                )
            if resolved_sentence_mode == "motion_shuffled":
                sentence_memory_batch, _permutation, _informative = (
                    motion_only_shuffle_sentence_memory_batch(
                        sentence_memory_batch,
                        query_ids=sentence_memory_query_ids(batch),
                        epoch=int(sentence_memory_epoch),
                        seed=int(cfg.get("seed", 1234)),
                    )
                )
            sentence_kwargs = sentence_memory_forward_kwargs(sentence_memory_batch)
    if dual_mode:
        trajectory = model.encode_trajectory(
            text_tokens=text_tokens,
            text_mask=text_mask,
            **word_kwargs,
            **sentence_kwargs,
        )
    output_duration = (
        trajectory.duration_seconds if length_mode == "predicted" else batch["duration"]
    )
    return {
        "trajectory": trajectory,
        "adapter_context": adapter_context,
        "retrieval_features": retrieval_features,
        "context_mask": context_mask,
        "context_lengths": context_lengths,
        "predicted_context_lengths": predicted_context_lengths,
        "predicted_duration": predicted_duration,
        "predicted_log_duration": predicted_log_duration,
        "output_duration": output_duration,
        "word_prior_mode": resolved_mode if dual_mode else "v1_required",
        "sentence_memory_mode": (
            resolved_sentence_mode if sentence_memory_model else "not_applicable"
        ),
        "sentence_memory_batch": sentence_memory_batch,
        "sentence_memory_motion_shuffle_epoch": (
            int(sentence_memory_epoch)
            if resolved_sentence_mode == "motion_shuffled"
            else None
        ),
        "sentence_memory_motion_shuffle_seed": (
            int(cfg.get("seed", 1234))
            if resolved_sentence_mode == "motion_shuffled"
            else None
        ),
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
    configured_manifest = cfg.get("data", {}).get(
        f"{args.split}_manifest_path"
    )
    canonical_manifest = (
        Path(configured_manifest)
        if configured_manifest
        else Path(cfg["data"]["data_dir"])
        / "meta"
        / f"manifest_{args.split}.jsonl"
    )
    canonical_rows = read_jsonl(canonical_manifest)
    selected_manifest, _selected_rows, manifest_summary = select_manifest(
        cfg,
        args.split,
        out_dir,
        args.num_samples,
        args.seed,
        manifest=args.manifest,
        selection_mode=args.selection_mode,
    )
    complete_canonical_manifest = (
        len(_selected_rows) == len(canonical_rows)
        and sorted(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in _selected_rows
        )
        == sorted(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in canonical_rows
        )
    )
    canonical_order = _selected_rows == canonical_rows
    manifest_summary.update(
        {
            "canonical_source_manifest": str(canonical_manifest),
            "canonical_sample_count": len(canonical_rows),
            "is_complete_canonical_manifest": complete_canonical_manifest,
            "is_canonical_order": canonical_order,
        }
    )
    # A complete first-order export can consume the precomputed neighbor table
    # directly. Subsets and reordered manifests deliberately use the copied
    # manifest and fall back to audited exact retrieval by stable query fields.
    dataset_manifest = canonical_manifest if canonical_order else selected_manifest
    cfg.setdefault("data", {})[f"{args.split}_manifest_path"] = str(dataset_manifest)
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
    dual_mode = is_dual_mode(cfg)
    sentence_memory_model = is_sentence_memory_model(cfg)
    if not dual_mode and args.word_prior != "auto":
        raise ValueError("--word_prior is available only for dual-mode v2")
    resolved_word_prior_mode = (
        str(cfg.get("eval", {}).get("sentence_memory_word_prior_mode", "off"))
        if sentence_memory_model and args.word_prior == "auto"
        else ("off" if dual_mode and args.word_prior == "auto" else args.word_prior)
    )
    if not sentence_memory_model and args.sentence_memory != "auto":
        raise ValueError("--sentence_memory is available only for v3")
    resolved_sentence_memory_mode = (
        "off"
        if sentence_memory_model and not sentence_memory_enabled(cfg)
        else (
            str(cfg.get("eval", {}).get("export_sentence_memory_mode", "on"))
            if sentence_memory_model and args.sentence_memory == "auto"
            else args.sentence_memory
        )
    )
    needs_provider = not dual_mode or resolved_word_prior_mode == "on"
    provider = ScaffoldProvider(cfg, dataset, device) if needs_provider else None
    sentence_memory_provider = (
        build_sentence_memory_provider(cfg, text_encoder, dataset=dataset)
        if sentence_memory_model
        and sentence_memory_provider_required(resolved_sentence_memory_mode)
        else None
    )
    if sentence_memory_provider is not None:
        sentence_memory_provider.validate_query_dataset(
            # Export manifests are often five-row subsets whose hash cannot
            # match the precomputed full-split table. The provider safely
            # falls back to online search for this small query set.
            dataset,
            require_neighbors=False,
        )
    retrieval_bank = (
        validate_train_only_retrieval_bank(cfg, provider)
        if provider is not None
        else None
    )
    model = build_continuous_trajectory_field(cfg, text_dim=text_encoder.text_dim).to(
        device
    )
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
    checkpoint_epoch = set_sentence_memory_provider_epoch_from_checkpoint(
        sentence_memory_provider, checkpoint
    )
    model_type, contract_version = checkpoint_contract(cfg)
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
            word_prior_mode=resolved_word_prior_mode,
            sentence_memory_provider=sentence_memory_provider,
            sentence_memory_mode=resolved_sentence_memory_mode,
            sentence_memory_epoch=checkpoint_epoch,
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
        branch_samples = {}
        if bool(cfg.get("eval", {}).get("residual_branch_ablation", False)):
            branch_samples["global_only"] = model.query_trajectory(
                inference["trajectory"],
                main_sample["tau"],
                time_domain="normalized",
                query_mask=main_sample["mask"],
                include_local_residual=False,
            )
            branch_samples["local_only"] = model.query_trajectory(
                inference["trajectory"],
                main_sample["tau"],
                time_domain="normalized",
                query_mask=main_sample["mask"],
                include_global_residual=False,
            )

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
            prediction = main_sample["outputs"]["prediction"][
                local_index, :output_length
            ]
            coarse_name = "coarse" if dual_mode else "prior"
            coarse = main_sample["outputs"][coarse_name][local_index, :output_length]
            pred_rot6d, pred_axis, pred_smplx = rot6d_to_axis_and_smplx(prediction)
            coarse_rot6d, coarse_axis, coarse_smplx = rot6d_to_axis_and_smplx(coarse)
            gt_path = out_dir / f"gt_{suffix}.npz"
            sample_path = out_dir / f"sample_{suffix}.npz"
            save_eval_npz(
                gt_path, gt_axis, gt_smplx, gt_rot6d, meta, label="ground_truth"
            )
            extra = {
                f"continuous_{coarse_name}_motion": coarse_axis.astype(np.float32),
                f"continuous_{coarse_name}_smplx": coarse_smplx.astype(np.float32),
                f"continuous_{coarse_name}_rot6d": coarse_rot6d.astype(np.float32),
                "checkpoint": np.asarray(str(args.checkpoint)),
                "checkpoint_epoch": np.asarray(
                    int(checkpoint.get("epoch", -1)), dtype=np.int32
                ),
                "model_type": np.asarray(model_type),
                "trajectory_contract_version": np.asarray(
                    contract_version, dtype=np.int32
                ),
                "word_prior_mode": np.asarray(inference["word_prior_mode"]),
                "sentence_memory_mode": np.asarray(inference["sentence_memory_mode"]),
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
            if inference["adapter_context"] is not None:
                context_rot6d, context_axis, context_smplx = rot6d_to_axis_and_smplx(
                    inference["adapter_context"][local_index, :context_length]
                )
                extra.update(
                    {
                        "adapter_context_motion": context_axis.astype(np.float32),
                        "adapter_context_smplx": context_smplx.astype(np.float32),
                        "adapter_context_rot6d": context_rot6d.astype(np.float32),
                        "retrieval_features": inference["retrieval_features"][
                            local_index, :context_length
                        ]
                        .cpu()
                        .float()
                        .numpy(),
                    }
                )
            memory_batch = inference["sentence_memory_batch"]
            if memory_batch is not None:
                for output_name, field_name, aliases in (
                    ("sentence_memory_ids", "ids", ()),
                    ("sentence_memory_scores", "scores", ()),
                    ("sentence_memory_durations", "durations", ()),
                    (
                        "sentence_memory_duration_log_gap",
                        "duration_log_gap",
                        ("duration_ratio",),
                    ),
                    ("sentence_memory_candidate_mask", "candidate_mask", ()),
                ):
                    value = _memory_row_numpy(
                        memory_batch,
                        field_name,
                        local_index,
                        *aliases,
                    )
                    if value is not None:
                        extra[output_name] = value
                memory_provenance = getattr(memory_batch, "provenance", {}) or {}
                candidate_group_ids = memory_provenance.get("candidate_group_ids", [])
                candidate_exact_text = memory_provenance.get("candidate_exact_text", [])
                query_seen_text = memory_provenance.get("query_seen_text", [])
                if local_index < len(candidate_group_ids):
                    extra["sentence_memory_group_ids"] = np.asarray(
                        candidate_group_ids[local_index], dtype=np.int64
                    )
                if local_index < len(candidate_exact_text):
                    extra["sentence_memory_exact_text"] = np.asarray(
                        candidate_exact_text[local_index], dtype=np.bool_
                    )
                if local_index < len(query_seen_text):
                    extra["sentence_memory_query_seen_text"] = np.asarray(
                        bool(query_seen_text[local_index]), dtype=np.bool_
                    )
                bank_id = memory_provenance.get("bank_id")
                if bank_id:
                    extra["sentence_memory_bank_id"] = np.asarray(str(bank_id))
                for provenance_name, output_name in (
                    (
                        "motion_candidate_permutation",
                        "sentence_memory_motion_candidate_permutation",
                    ),
                    ("motion_source_ids", "sentence_memory_motion_source_ids"),
                    (
                        "motion_only_shuffle_informative",
                        "sentence_memory_motion_shuffle_informative",
                    ),
                ):
                    provenance_rows = memory_provenance.get(provenance_name, [])
                    if local_index < len(provenance_rows):
                        extra[output_name] = np.asarray(
                            provenance_rows[local_index]
                        )
                for provenance_name, output_name in (
                    (
                        "motion_only_shuffle_epoch",
                        "sentence_memory_motion_shuffle_epoch",
                    ),
                    (
                        "motion_only_shuffle_seed",
                        "sentence_memory_motion_shuffle_seed",
                    ),
                    ("payload_reads", "sentence_memory_payload_reads"),
                ):
                    if provenance_name in memory_provenance:
                        extra[output_name] = np.asarray(
                            memory_provenance[provenance_name]
                        )
            extra.update(_trajectory_numpy(inference["trajectory"], local_index))
            for branch_name, branch_prediction in branch_samples.items():
                branch_rot6d, branch_axis, branch_smplx = rot6d_to_axis_and_smplx(
                    branch_prediction[local_index, :output_length]
                )
                extra[f"{branch_name}_rot6d"] = branch_rot6d.astype(np.float32)
                extra[f"{branch_name}_motion"] = branch_axis.astype(np.float32)
                extra[f"{branch_name}_smplx"] = branch_smplx.astype(np.float32)
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
                extra[f"continuous_{key}_tau"] = (
                    fps_sample["tau"][local_index, :fps_length].cpu().float().numpy()
                )
            save_eval_npz(
                sample_path,
                pred_axis,
                pred_smplx,
                pred_rot6d,
                meta,
                label=(
                    "signtrajfield_rag_v3"
                    if sentence_memory_model
                    else (
                        "signtrajfield_v2"
                        if dual_mode
                        else "niaf_continuous_trajectory_field"
                    )
                ),
                extra=extra,
            )
            text_subset = sentence_memory_text_subset(
                inference["sentence_memory_batch"],
                sentence_memory_provider,
                local_index,
                meta["text"],
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
                    "word_prior_mode": inference["word_prior_mode"],
                    "sentence_memory_mode": inference["sentence_memory_mode"],
                    "sentence_memory_text_subset": (text_subset)
                    if sentence_memory_model
                    else "not_applicable",
                    "sentence_memory_diagnostics": (
                        sentence_memory_diagnostics_row(
                            inference["trajectory"], local_index
                        )
                        if sentence_memory_model
                        else None
                    ),
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
        abs(row["predicted_duration_seconds"] - row["ground_truth_duration_seconds"])
        for row in rows
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "config": str(args.config),
        "split": args.split,
        "length_mode": args.length_mode,
        "model_type": model_type,
        "trajectory_contract_version": contract_version,
        "word_prior_mode": resolved_word_prior_mode if dual_mode else "v1_required",
        "sentence_memory_mode": (
            resolved_sentence_memory_mode if sentence_memory_model else "not_applicable"
        ),
        "sentence_memory_motion_shuffle_epoch": (
            int(checkpoint_epoch)
            if sentence_memory_model
            and resolved_sentence_memory_mode == "motion_shuffled"
            else None
        ),
        "sentence_memory_motion_shuffle_seed": (
            int(cfg.get("seed", 1234))
            if sentence_memory_model
            and resolved_sentence_memory_mode == "motion_shuffled"
            else None
        ),
        "sentence_memory": (
            getattr(sentence_memory_provider, "config_summary", None)
            if sentence_memory_provider is not None
            else (
                checkpoint.get("sentence_memory_identity")
                if sentence_memory_model
                else None
            )
        ),
        "context_fps": float(args.context_fps),
        "sample_fps": list(fps_values),
        "retrieval_bank": retrieval_bank,
        "manifest": manifest_summary,
        "num_exported": len(rows),
        "duration_mae_seconds": float(
            sum(duration_errors) / max(len(duration_errors), 1)
        ),
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
