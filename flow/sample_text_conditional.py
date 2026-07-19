#!/usr/bin/env python
import argparse
import csv
import json
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from flow.adapter_prior import FrozenAdapterPrior
from flow.latent_codec import LatentMotionCodec
from flow.model import build_text_conditioned_model_from_args, sample_euler_text, sample_heun_text
from flow.render import smplx182_to_vertices, write_vertices_video
from flow.residual_prior import WordMotionPrior
from flow.smplx_features import (
    COMPACT_DIM,
    compact_from_rotation_representation,
    normalize_rotation_rep,
    rotation_rep_stats_paths,
    smplx182_from_compact,
)
from flow.text_encoder import FrozenT5TextEncoder


DEFAULT_OUT_DIR = Path("visualize/flow/flow_text_cond_samples")


LATENCY_METRIC_ORDER = [
    "total_wall_ms",
    "flow_text_encoder_ms",
    "residual_word_prior_build_ms",
    "residual_prior_vae_encode_ms",
    "adapter_prior_total_ms",
    "adapter_candidate_build_ms",
    "adapter_word_vae_encode_ms",
    "adapter_text_encoder_ms",
    "soft_arranger_ms",
    "adapter_concat_prior_build_ms",
    "adapter_concat_vae_encode_ms",
    "content_style_adapter_ms",
    "adapter_prior_vae_decode_ms",
    "source_prior_postprocess_ms",
    "flow_sampler_ms",
    "vae_decode_ms",
    "sample_postprocess_ms",
    "sample_smplx_convert_ms",
    "coarse_smplx_convert_ms",
    "sample_save_io_ms",
    "gt_load_ms",
    "gt_smplx_convert_ms",
    "gt_save_io_ms",
    "render_ms",
]


class LatencyProfiler:
    def __init__(self, device):
        self.device = device
        self.times = {}

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.times[name] = self.times.get(name, 0.0) + elapsed_ms

    def add_elapsed(self, name, start_time):
        self.sync()
        self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - start_time) * 1000.0

    def row(self):
        return {key: float(value) for key, value in self.times.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a text-conditioned SMPL-X flow checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--text_model_path", type=Path, default=None)
    parser.add_argument("--max_text_tokens", type=int, default=0)
    parser.add_argument("--motion_space", "--motion-space", default="", choices=["", "smplx", "latent"])
    parser.add_argument("--rotation_rep", "--rotation-rep", default="", choices=["", "axis_angle", "rot6d"])
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--source_mode", "--source-mode", default="", choices=["", "noise", "residual", "adapter_residual"])
    parser.add_argument("--adapter_checkpoint", "--adapter-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--adapter_num_negative_candidates",
        "--adapter-num-negative-candidates",
        type=int,
        default=None,
        help="Override soft-arranger adapter negative candidate count. Use 0 for clean evaluation.",
    )
    parser.add_argument(
        "--no_negative_candidates",
        "--no-negative-candidates",
        action="store_true",
        help="Disable random negative word candidates for soft-arranger adapter-residual sampling.",
    )
    parser.add_argument(
        "--shuffle_word_candidates",
        "--shuffle-word-candidates",
        action="store_true",
        help=(
            "Shuffle soft-arranger word-pose candidates during adapter-residual inference. "
            "This is an ablation/debug option; default inference keeps candidates deterministic."
        ),
    )
    parser.add_argument(
        "--adapter_prior_mode_override",
        "--adapter-prior-mode-override",
        default="",
        choices=["", "concat", "soft_arranger"],
        help="Override the adapter checkpoint prior mode for ablations without modifying the checkpoint.",
    )
    parser.add_argument(
        "--adapter_enabled_override",
        "--adapter-enabled-override",
        default="",
        choices=["", "true", "false"],
        help="Override whether the content/style adapter is applied after the word prior.",
    )
    parser.add_argument("--word_data_dir", "--word-data-dir", type=Path, default=None)
    parser.add_argument("--word_split", "--word-split", default="")
    parser.add_argument(
        "--condition_field",
        "--condition-field",
        default="",
        choices=["", "text", "gloss", "text_gloss", "label_word"],
        help="Manifest field used for dataset-prompt sampling. Defaults to the checkpoint setting, or text for older checkpoints.",
    )
    parser.add_argument("--residual_noise_scale", "--residual-noise-scale", type=float, default=-1.0)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--text", action="append", default=None, help="Prompt text. Can be passed more than once.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest JSONL for dataset-prompt sampling.")
    parser.add_argument("--num_prompts", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1, help="Repeats a single --text prompt this many times.")
    parser.add_argument(
        "--skip_save_outputs",
        "--skip-save-outputs",
        action="store_true",
        help=(
            "Run inference without writing sample_*.npz or gt_*.npz files. "
            "Useful for latency profiling; offline metric scripts need saved outputs."
        ),
    )
    parser.add_argument(
        "--prior_only",
        "--prior-only",
        action="store_true",
        help=(
            "For adapter-residual checkpoints, decode the adapted source prior and "
            "skip the final flow refinement. Useful for prior-only ablations."
        ),
    )
    parser.add_argument("--match_manifest_lengths", action="store_true")
    parser.add_argument("--length", type=int, default=196)
    parser.add_argument("--steps", "--sample_steps", dest="steps", type=int, default=0)
    parser.add_argument("--sampler", default="", choices=["", "euler", "heun"])
    parser.add_argument("--noise_smoothing", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_device", default="cpu", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model_dir", type=Path, default=Path("deps/smpl_models"))
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--software_face_stride", type=int, default=1)
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=["none", "how2sign_front", "rot_x_180", "rot_y_180", "rot_z_180", "flip_y", "flip_z"],
    )
    parser.add_argument(
        "--profile_latency",
        "--profile-latency",
        action="store_true",
        help="Measure per-prompt inference latency and write JSON/CSV summaries.",
    )
    parser.add_argument(
        "--latency_warmup",
        "--latency-warmup",
        type=int,
        default=0,
        help="Exclude the first N prompts from latency summary statistics.",
    )
    parser.add_argument(
        "--latency_out_json",
        "--latency-out-json",
        type=Path,
        default=None,
        help="Latency JSON output path. Defaults to out_dir/latency_profile.json.",
    )
    parser.add_argument(
        "--latency_out_csv",
        "--latency-out-csv",
        type=Path,
        default=None,
        help="Latency CSV output path. Defaults to out_dir/latency_profile.csv.",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_and_stats(args, device):
    ckpt = load_checkpoint(args.checkpoint)
    model_cfg = ckpt.get("model_config", {})
    text_cfg = ckpt.get("text_config", {})
    flow_max_frames = int(model_cfg.get("max_frames", 400))
    motion_space = str(args.motion_space or model_cfg.get("motion_space", "smplx"))
    raw_max_frames = int(model_cfg.get("raw_max_frames", flow_max_frames if motion_space == "smplx" else flow_max_frames * 4))
    input_dim = int(model_cfg.get("input_dim", COMPACT_DIM))
    rotation_rep = normalize_rotation_rep(
        args.rotation_rep
        or model_cfg.get("rotation_rep")
        or ckpt.get("data_config", {}).get("rotation_rep")
        or ckpt.get("latent_config", {}).get("rotation_rep")
        or ckpt.get("args", {}).get("rotation_rep")
        or "axis_angle"
    )
    text_conditioning = str(model_cfg.get("text_conditioning", text_cfg.get("conditioning", "pooled")))
    model_args = SimpleNamespace(
        text_conditioning=text_conditioning,
        model_size=str(model_cfg.get("model_size", "custom")),
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        max_frames=flow_max_frames,
        flow_max_frames=flow_max_frames,
        text_dim=int(model_cfg.get("text_dim", text_cfg.get("text_dim", 768))),
    )
    model = build_text_conditioned_model_from_args(model_args)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    data_dir = args.data_dir
    if data_dir is None:
        data_cfg = ckpt.get("data_config", {})
        data_dir = Path(data_cfg.get("data_dir", "/media/cvpr/haomian/data/SOKE_FLOW/how2sign_upper_smplx_smoke"))
    mean_path, std_path = rotation_rep_stats_paths(data_dir, rotation_rep)
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    text_model_path = args.text_model_path
    if text_model_path is None:
        text_model_path = Path(text_cfg.get("text_model_path", "deps/flan-t5-base"))
    max_text_tokens = args.max_text_tokens if args.max_text_tokens > 0 else int(text_cfg.get("max_text_tokens", 64))

    defaults = ckpt.get("args", {})
    default_steps = int(defaults.get("sample_steps", 100))
    default_sampler = str(defaults.get("sampler", "heun"))
    default_noise_smoothing = int(defaults.get("noise_smoothing", 9))
    source_cfg = ckpt.get("source_config", {})
    condition_field = str(
        args.condition_field
        or source_cfg.get("condition_field")
        or text_cfg.get("condition_field")
        or defaults.get("condition_field", "text")
    )
    default_source_mode = str(
        args.source_mode
        or source_cfg.get("source_mode")
        or defaults.get("source_mode", "noise")
    )
    word_data_dir = (
        args.word_data_dir
        or (Path(source_cfg["word_data_dir"]) if source_cfg.get("word_data_dir") else None)
        or (Path(defaults["word_data_dir"]) if defaults.get("word_data_dir") else None)
    )
    word_split = args.word_split or str(source_cfg.get("word_split") or defaults.get("word_split", "train"))
    adapter_checkpoint = (
        args.adapter_checkpoint
        or (Path(source_cfg["adapter_checkpoint"]) if source_cfg.get("adapter_checkpoint") else None)
        or (Path(defaults["adapter_checkpoint"]) if defaults.get("adapter_checkpoint") else None)
    )
    residual_noise_scale = (
        float(args.residual_noise_scale)
        if args.residual_noise_scale >= 0
        else float(source_cfg.get("residual_noise_scale", defaults.get("residual_noise_scale", 0.25)))
    )

    latent_codec = None
    latent_stats = None
    latent_cfg = ckpt.get("latent_config") or {}
    if motion_space == "latent":
        vae_checkpoint = args.vae_checkpoint or (Path(latent_cfg["vae_checkpoint"]) if latent_cfg.get("vae_checkpoint") else None)
        if vae_checkpoint is None:
            raise ValueError("Latent checkpoint has no VAE path; pass --vae_checkpoint.")
        latent_codec = LatentMotionCodec(vae_checkpoint, device=device)
        stats = latent_cfg.get("stats")
        if not stats or "mean" not in stats or "std" not in stats:
            raise ValueError("Latent checkpoint does not contain latent_config.stats.")
        latent_stats = {
            "mean": torch.as_tensor(stats["mean"], dtype=torch.float32, device=device),
            "std": torch.as_tensor(stats["std"], dtype=torch.float32, device=device),
        }
    return {
        "ckpt": ckpt,
        "model": model,
        "mean": mean,
        "std": std,
        "data_dir": Path(data_dir),
        "motion_space": motion_space,
        "rotation_rep": rotation_rep,
        "text_model_path": Path(text_model_path),
        "max_text_tokens": max_text_tokens,
        "max_frames": raw_max_frames,
        "flow_max_frames": flow_max_frames,
        "input_dim": input_dim,
        "latent_codec": latent_codec,
        "latent_stats": latent_stats,
        "text_conditioning": text_conditioning,
        "condition_field": condition_field,
        "default_steps": default_steps,
        "default_sampler": default_sampler,
        "default_noise_smoothing": default_noise_smoothing,
        "source_mode": default_source_mode,
        "word_data_dir": word_data_dir,
        "word_split": word_split,
        "adapter_checkpoint": adapter_checkpoint,
        "adapter_prior_config": source_cfg.get("adapter_prior_config"),
        "residual_noise_scale": residual_noise_scale,
    }


def prompt_specs_from_text(args):
    texts = args.text or []
    if len(texts) == 1 and args.num_samples > 1:
        texts = texts * args.num_samples
    specs = []
    for idx, text in enumerate(texts):
        specs.append(
            {
                "text": text,
                "length": args.length,
                "name": f"prompt_{idx:02d}",
                "row": None,
            }
        )
    return specs


def select_condition_text(row, field):
    text = str(row.get("text", ""))
    label_word = str(row.get("label_word", ""))
    gloss = str(row.get("gloss", ""))
    if field == "label_word" and label_word:
        return label_word
    if field == "gloss" and gloss:
        return gloss
    if field == "text_gloss" and gloss:
        return f"{text} {gloss}".strip()
    return text


def prompt_specs_from_manifest(args, max_frames, condition_field="text"):
    rows = read_jsonl(args.manifest)
    specs = []
    for idx, row in enumerate(rows[: args.num_prompts]):
        length = args.length
        if args.match_manifest_lengths:
            length = int(row.get("num_frames", length))
        if length > max_frames:
            print(f"WARNING: capping prompt {idx:02d} length from {length} to model max_frames={max_frames}")
            length = max_frames
        specs.append(
            {
                "text": select_condition_text(row, condition_field),
                "raw_text": row.get("text", ""),
                "condition_field": condition_field,
                "length": int(length),
                "name": row.get("name", f"manifest_{idx:02d}"),
                "row": row,
            }
        )
    return specs


def load_gt_motion(data_dir, row):
    if row is None or "motion_path" not in row:
        return None
    path = data_dir / row["motion_path"]
    if not path.is_file():
        print(f"WARNING: GT motion file not found: {path}")
        return None
    with np.load(path) as data:
        if "motion" not in data.files:
            print(f"WARNING: GT motion file has no 'motion' key: {path}")
            return None
        return data["motion"].astype(np.float32)


def manifest_metadata(row, condition_field=""):
    if row is None:
        return {}
    keys = (
        "gloss",
        "label_word",
        "source_name",
        "source_split",
        "source_fps",
        "source_num_frames",
        "source_pose_frames",
        "signer",
        "dataset",
        "fps",
        "duration",
        "num_frames",
        "motion_path",
    )
    metadata = {}
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            metadata[key] = value
    if condition_field:
        metadata["condition_field"] = condition_field
    if "text" in row:
        metadata["raw_text"] = row["text"]
    return metadata


@torch.no_grad()
def encode_text_condition(text_encoder, texts, text_conditioning, device, dtype):
    if text_conditioning == "token_prefix":
        text_tokens, text_mask = text_encoder.encode_tokens(texts)
        return (
            text_tokens.to(device=device, dtype=dtype),
            text_mask.to(device=device, dtype=torch.bool),
        )
    return text_encoder.encode(texts).to(device=device, dtype=dtype)


@torch.no_grad()
def sample_one(
    model,
    text_encoder,
    text,
    length,
    args,
    runtime,
    device,
    prior_builder=None,
    adapter_prior=None,
    profiler=None,
):
    if args.prior_only and runtime["source_mode"] != "adapter_residual":
        raise RuntimeError("--prior_only is only supported for adapter_residual checkpoints.")

    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    text_condition = None
    if not args.prior_only:
        if profiler is None:
            text_condition = encode_text_condition(
                text_encoder,
                [text],
                runtime["text_conditioning"],
                device,
                dtype=next(model.parameters()).dtype,
            )
        else:
            with profiler.section("flow_text_encoder_ms"):
                text_condition = encode_text_condition(
                    text_encoder,
                    [text],
                    runtime["text_conditioning"],
                    device,
                    dtype=next(model.parameters()).dtype,
                )
    sampler_name = args.sampler or runtime["default_sampler"]
    sampler = sample_heun_text if sampler_name == "heun" else sample_euler_text
    steps = args.steps if args.steps > 0 else runtime["default_steps"]
    noise_smoothing = args.noise_smoothing if args.noise_smoothing >= 0 else runtime["default_noise_smoothing"]
    prior = None
    latent_prior = None
    coarse = None
    coarse_representation = None
    if runtime["source_mode"] == "adapter_residual":
        if runtime["motion_space"] != "latent":
            raise RuntimeError("adapter_residual sampling is only supported for latent checkpoints.")
        if adapter_prior is None:
            raise RuntimeError("Adapter-residual checkpoint sampling requires a FrozenAdapterPrior.")
        raw_x = torch.zeros(1, length, runtime["mean"].shape[0], dtype=next(model.parameters()).dtype, device=device)
        if profiler is None:
            prior_out = adapter_prior.build_prior(
                [text],
                raw_x,
                mask,
                [length],
                shuffle_candidates=args.shuffle_word_candidates,
            )
        else:
            with profiler.section("adapter_prior_total_ms"):
                prior_out = adapter_prior.build_prior(
                    [text],
                    raw_x,
                    mask,
                    [length],
                    profiler=profiler,
                    shuffle_candidates=args.shuffle_word_candidates,
                )
        latent_prior = prior_out["z_prior"]
        if profiler is None:
            coarse_representation = prior_out["prior_raw"].detach().cpu().numpy()[0] * runtime["std"][None] + runtime["mean"][None]
            coarse = compact_from_rotation_representation(coarse_representation, runtime["rotation_rep"])
        else:
            with profiler.section("source_prior_postprocess_ms"):
                coarse_representation = prior_out["prior_raw"].detach().cpu().numpy()[0] * runtime["std"][None] + runtime["mean"][None]
                coarse = compact_from_rotation_representation(coarse_representation, runtime["rotation_rep"])
        if args.prior_only:
            return coarse, None, coarse_representation
    elif runtime["source_mode"] == "residual":
        if prior_builder is None:
            raise RuntimeError("Residual checkpoint sampling requires a word prior builder.")
        if profiler is None:
            prior, _ = prior_builder.batch(
                [text],
                [length],
                max_len=length,
                device=device,
                dtype=next(model.parameters()).dtype,
            )
        else:
            with profiler.section("residual_word_prior_build_ms"):
                prior, _ = prior_builder.batch(
                    [text],
                    [length],
                    max_len=length,
                    device=device,
                    dtype=next(model.parameters()).dtype,
                )
        if profiler is None:
            coarse_representation = prior.detach().cpu().numpy()[0] * runtime["std"][None] + runtime["mean"][None]
            coarse = compact_from_rotation_representation(coarse_representation, runtime["rotation_rep"])
        else:
            with profiler.section("source_prior_postprocess_ms"):
                coarse_representation = prior.detach().cpu().numpy()[0] * runtime["std"][None] + runtime["mean"][None]
                coarse = compact_from_rotation_representation(coarse_representation, runtime["rotation_rep"])

    if runtime["motion_space"] == "latent":
        latent_codec = runtime["latent_codec"]
        latent_stats = runtime["latent_stats"]
        latent_mask = latent_codec.latent_mask(mask)
        if latent_prior is None and prior is not None:
            if profiler is None:
                prior_z, _ = latent_codec.encode(prior, mask=mask)
                latent_prior = latent_codec.normalize_latent(prior_z, latent_stats)
            else:
                with profiler.section("residual_prior_vae_encode_ms"):
                    prior_z, _ = latent_codec.encode(prior, mask=mask)
                    latent_prior = latent_codec.normalize_latent(prior_z, latent_stats)
        if profiler is None:
            latent_sample = sampler(
                model,
                text_condition,
                (1, latent_mask.shape[1], latent_codec.latent_dim),
                steps=steps,
                device=device,
                mask=latent_mask,
                noise_smoothing=noise_smoothing,
                x0=latent_prior,
                source_noise_scale=runtime["residual_noise_scale"] if latent_prior is not None else 1.0,
            )
            latent_sample = latent_codec.denormalize_latent(latent_sample, latent_stats)
            decoded = latent_codec.decode(
                latent_sample,
                target_length=length,
                mask=mask,
                latent_mask=latent_mask,
            )
            sample = decoded.detach().cpu().numpy()[0]
            representation = sample * runtime["std"][None] + runtime["mean"][None]
            return compact_from_rotation_representation(representation, runtime["rotation_rep"]), coarse, representation
        with profiler.section("flow_sampler_ms"):
            latent_sample = sampler(
                model,
                text_condition,
                (1, latent_mask.shape[1], latent_codec.latent_dim),
                steps=steps,
                device=device,
                mask=latent_mask,
                noise_smoothing=noise_smoothing,
                x0=latent_prior,
                source_noise_scale=runtime["residual_noise_scale"] if latent_prior is not None else 1.0,
            )
        with profiler.section("vae_decode_ms"):
            latent_sample = latent_codec.denormalize_latent(latent_sample, latent_stats)
            decoded = latent_codec.decode(
                latent_sample,
                target_length=length,
                mask=mask,
                latent_mask=latent_mask,
            )
        with profiler.section("sample_postprocess_ms"):
            sample = decoded.detach().cpu().numpy()[0]
            representation = sample * runtime["std"][None] + runtime["mean"][None]
            compact = compact_from_rotation_representation(representation, runtime["rotation_rep"])
        return compact, coarse, representation

    if profiler is None:
        sample = sampler(
            model,
            text_condition,
            (1, length, runtime["input_dim"]),
            steps=steps,
            device=device,
            mask=mask,
            noise_smoothing=noise_smoothing,
            x0=prior,
            source_noise_scale=runtime["residual_noise_scale"] if prior is not None else 1.0,
        )
        sample = sample.detach().cpu().numpy()[0]
        representation = sample * runtime["std"][None] + runtime["mean"][None]
        return compact_from_rotation_representation(representation, runtime["rotation_rep"]), coarse, representation
    with profiler.section("flow_sampler_ms"):
        sample = sampler(
            model,
            text_condition,
            (1, length, runtime["input_dim"]),
            steps=steps,
            device=device,
            mask=mask,
            noise_smoothing=noise_smoothing,
            x0=prior,
            source_noise_scale=runtime["residual_noise_scale"] if prior is not None else 1.0,
        )
    with profiler.section("sample_postprocess_ms"):
        sample = sample.detach().cpu().numpy()[0]
        representation = sample * runtime["std"][None] + runtime["mean"][None]
        compact = compact_from_rotation_representation(representation, runtime["rotation_rep"])
    return compact, coarse, representation


def summarize_latency_rows(rows):
    included = [row for row in rows if row.get("included_in_summary", True)]
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.endswith("_ms") and isinstance(value, (int, float))
        }
    )
    summary = {}
    for name in metric_names:
        values = np.asarray(
            [float(row[name]) for row in included if isinstance(row.get(name), (int, float))],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summary[name] = {
            "count": int(values.size),
            "mean_ms": float(values.mean()),
            "std_ms": float(values.std()),
            "median_ms": float(np.median(values)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }
    return summary


def write_latency_profile(args, runtime, device, rows):
    if not rows:
        return
    out_json = args.latency_out_json or (args.out_dir / "latency_profile.json")
    out_csv = args.latency_out_csv or (args.out_dir / "latency_profile.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_latency_rows(rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "cuda_synchronized": bool(device.type == "cuda"),
        "latency_warmup": int(max(args.latency_warmup, 0)),
        "skip_save_outputs": bool(args.skip_save_outputs),
        "prior_only": bool(args.prior_only),
        "num_prompts": len(rows),
        "included_prompts": int(sum(1 for row in rows if row.get("included_in_summary", True))),
        "source_mode": runtime["source_mode"],
        "motion_space": runtime["motion_space"],
        "rotation_rep": runtime["rotation_rep"],
        "text_conditioning": runtime["text_conditioning"],
        "condition_field": runtime["condition_field"],
        "definition": (
            "Per-prompt latency in milliseconds. CUDA runs synchronize before and after "
            "each timed section. Model/checkpoint loading and word-prior index construction "
            "are startup costs and are not included in per-prompt rows."
        ),
        "summary": summary,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata_fields = [
        "index",
        "included_in_summary",
        "name",
        "length",
        "text_chars",
        "skip_save_outputs",
        "prior_only",
        "condition_field",
        "source_mode",
        "motion_space",
        "sampler",
        "steps",
        "latent_frames",
    ]
    metric_fields = [name for name in LATENCY_METRIC_ORDER if any(name in row for row in rows)]
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in metadata_fields and key not in metric_fields and key.endswith("_ms")
        }
    )
    fieldnames = metadata_fields + metric_fields + extra_fields
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Saved latency JSON: {out_json}")
    print(f"Saved latency CSV: {out_csv}")
    if summary:
        print("Latency summary mean_ms:")
        for name in [field for field in LATENCY_METRIC_ORDER if field in summary]:
            print(f"  {name}: {summary[name]['mean_ms']:.3f}")


def main():
    args = parse_args()
    if not args.text and args.manifest is None:
        raise ValueError("Pass either --text or --manifest.")
    if args.text and args.manifest is not None:
        raise ValueError("Pass only one sampling source: --text or --manifest.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_model_and_stats(args, device)
    prior_builder = None
    adapter_prior = None
    if runtime["source_mode"] == "residual":
        if runtime["word_data_dir"] is None:
            raise ValueError("Checkpoint uses residual source mode but no word_data_dir was saved; pass --word_data_dir.")
        prior_builder = WordMotionPrior(
            runtime["word_data_dir"],
            split=runtime["word_split"],
            target_mean=runtime["mean"],
            target_std=runtime["std"],
            rotation_rep=runtime["rotation_rep"],
            lazy_motions=True,
        )
    elif runtime["source_mode"] == "adapter_residual":
        if runtime["motion_space"] != "latent" or runtime["latent_codec"] is None:
            raise ValueError("adapter_residual sampling requires a latent checkpoint with a saved VAE path.")
        if runtime["adapter_checkpoint"] is None:
            raise ValueError(
                "Checkpoint uses adapter_residual source mode but no adapter checkpoint was saved; "
                "pass --adapter_checkpoint."
            )
        adapter_prior = FrozenAdapterPrior.from_checkpoint(
            runtime["adapter_checkpoint"],
            device=device,
            latent_codec=runtime["latent_codec"],
            target_mean=runtime["mean"],
            target_std=runtime["std"],
            word_data_dir=runtime["word_data_dir"],
            word_split=runtime["word_split"],
            text_model_path=runtime["text_model_path"],
            max_text_tokens=runtime["max_text_tokens"],
            candidate_seed=args.seed,
            lazy_word_motions=True,
            num_negative_candidates=0 if args.no_negative_candidates else args.adapter_num_negative_candidates,
            prior_mode_override=args.adapter_prior_mode_override or None,
            adapter_enabled_override=(
                None
                if args.adapter_enabled_override == ""
                else args.adapter_enabled_override == "true"
            ),
        )

    text_encoder = FrozenT5TextEncoder(
        runtime["text_model_path"],
        device=device,
        max_length=runtime["max_text_tokens"],
        local_files_only=True,
        cache=True,
    )
    if text_encoder.text_dim != runtime["ckpt"].get("text_config", {}).get("text_dim", text_encoder.text_dim):
        print(
            "WARNING: loaded text encoder dim "
            f"{text_encoder.text_dim} differs from checkpoint text config."
        )

    if args.manifest is not None:
        specs = prompt_specs_from_manifest(args, runtime["max_frames"], runtime["condition_field"])
    else:
        specs = prompt_specs_from_text(args)
    if not specs:
        raise RuntimeError("No prompts selected.")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Motion space: {runtime['motion_space']}")
    print(f"Rotation representation: {runtime['rotation_rep']} input_dim={runtime['input_dim']}")
    print(f"Text model: {runtime['text_model_path']}")
    print(f"Text conditioning: {runtime['text_conditioning']}")
    print(f"Condition field: {runtime['condition_field']}")
    print(f"Source mode: {runtime['source_mode']}")
    if args.prior_only:
        print("Prior only: enabled; final flow sampler is skipped.")
    if runtime["latent_codec"] is not None:
        print(
            f"VAE: {runtime['latent_codec'].checkpoint_path} "
            f"latent_dim={runtime['latent_codec'].latent_dim} flow_max_frames={runtime['flow_max_frames']}"
        )
    if prior_builder is not None:
        print(
            f"Word prior: {runtime['word_data_dir']} split={runtime['word_split']} "
            f"entries={len(prior_builder.entries)} noise_scale={runtime['residual_noise_scale']}"
        )
    if adapter_prior is not None:
        print(
            f"Adapter prior: {runtime['adapter_checkpoint']} mode={adapter_prior.prior_mode} "
            f"adapter_enabled={adapter_prior.adapter_enabled} "
            f"word_data_dir={adapter_prior.prior_builder.data_dir} split={adapter_prior.prior_builder.split} "
            f"entries={len(adapter_prior.prior_builder.entries)} "
            f"negatives={adapter_prior.candidate_builder.num_negative_candidates if adapter_prior.candidate_builder is not None else 0} "
            f"shuffle_candidates={bool(args.shuffle_word_candidates)} "
            f"noise_scale={runtime['residual_noise_scale']}"
        )
    print(f"Output: {args.out_dir}")
    print(f"Prompts: {len(specs)}")
    if args.profile_latency:
        print(
            "Latency profiling: enabled "
            f"warmup={max(args.latency_warmup, 0)} "
            f"cuda_synchronized={device.type == 'cuda'}"
        )

    latency_rows = []
    for idx, spec in enumerate(specs):
        length = int(spec["length"])
        if length > runtime["max_frames"]:
            print(f"WARNING: capping prompt {idx:02d} length from {length} to model max_frames={runtime['max_frames']}")
            length = runtime["max_frames"]
        profiler = LatencyProfiler(device) if args.profile_latency else None
        total_start = None
        if profiler is not None:
            profiler.sync()
            total_start = time.perf_counter()
        compact, coarse, representation = sample_one(
            runtime["model"],
            text_encoder,
            spec["text"],
            length,
            args,
            runtime,
            device,
            prior_builder=prior_builder,
            adapter_prior=adapter_prior,
            profiler=profiler,
        )
        full_smplx = None
        if args.render or not args.skip_save_outputs:
            if profiler is None:
                full_smplx = smplx182_from_compact(compact)
            else:
                with profiler.section("sample_smplx_convert_ms"):
                    full_smplx = smplx182_from_compact(compact)

        if args.skip_save_outputs:
            print(f"Skipped output save: sample_{idx:02d}.npz")
        else:
            npz_path = args.out_dir / f"sample_{idx:02d}.npz"
            payload = {
                "motion": compact.astype(np.float32),
                "representation": representation.astype(np.float32),
                "rotation_rep": runtime["rotation_rep"],
                "smplx": full_smplx,
                "text": spec["text"],
                "name": spec["name"],
                "length": length,
            }
            payload.update(manifest_metadata(spec["row"], runtime["condition_field"]))
            if coarse is not None:
                payload["coarse_motion"] = coarse.astype(np.float32)
                if profiler is None:
                    payload["coarse_smplx"] = smplx182_from_compact(coarse)
                else:
                    with profiler.section("coarse_smplx_convert_ms"):
                        payload["coarse_smplx"] = smplx182_from_compact(coarse)
            if profiler is None:
                np.savez_compressed(npz_path, **payload)
            else:
                with profiler.section("sample_save_io_ms"):
                    np.savez_compressed(npz_path, **payload)
            print(f"Saved: {npz_path}")

            if profiler is None:
                gt_motion = load_gt_motion(runtime["data_dir"], spec["row"])
            else:
                with profiler.section("gt_load_ms"):
                    gt_motion = load_gt_motion(runtime["data_dir"], spec["row"])
            if gt_motion is not None:
                gt_path = args.out_dir / f"gt_{idx:02d}.npz"
                if profiler is None:
                    gt_smplx = smplx182_from_compact(gt_motion)
                else:
                    with profiler.section("gt_smplx_convert_ms"):
                        gt_smplx = smplx182_from_compact(gt_motion)
                if profiler is None:
                    np.savez_compressed(
                        gt_path,
                        motion=gt_motion.astype(np.float32),
                        smplx=gt_smplx,
                        text=spec["text"],
                        name=spec["name"],
                        length=len(gt_motion),
                        **manifest_metadata(spec["row"], runtime["condition_field"]),
                    )
                else:
                    with profiler.section("gt_save_io_ms"):
                        np.savez_compressed(
                            gt_path,
                            motion=gt_motion.astype(np.float32),
                            smplx=gt_smplx,
                            text=spec["text"],
                            name=spec["name"],
                            length=len(gt_motion),
                            **manifest_metadata(spec["row"], runtime["condition_field"]),
                        )
                print(f"Saved: {gt_path}")

        if args.render:
            render_context = profiler.section("render_ms") if profiler is not None else nullcontext()
            with render_context:
                vertices, faces = smplx182_to_vertices(
                    full_smplx,
                    model_dir=args.model_dir,
                    device=args.render_device,
                    batch_size=args.smplx_batch_size,
                )
                video_path = args.out_dir / f"sample_{idx:02d}.mp4"
                write_vertices_video(
                    vertices,
                    faces,
                    video_path,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    face_stride=args.software_face_stride,
                    label=f"text flow sample {idx:02d}",
                    view_transform=args.view_transform,
                )
            print(f"Saved: {video_path}")

        if profiler is not None:
            profiler.add_elapsed("total_wall_ms", total_start)
            latent_frames = ""
            if runtime["motion_space"] == "latent" and runtime["latent_codec"] is not None:
                latent_mask = runtime["latent_codec"].latent_mask(torch.ones(1, length, dtype=torch.bool, device=device))
                latent_frames = int(latent_mask.shape[1])
            sampler_name = args.sampler or runtime["default_sampler"]
            steps = args.steps if args.steps > 0 else runtime["default_steps"]
            row = {
                "index": int(idx),
                "included_in_summary": bool(idx >= max(args.latency_warmup, 0)),
                "name": spec["name"],
                "length": int(length),
                "text_chars": int(len(str(spec["text"]))),
                "skip_save_outputs": bool(args.skip_save_outputs),
                "prior_only": bool(args.prior_only),
                "condition_field": runtime["condition_field"],
                "source_mode": runtime["source_mode"],
                "motion_space": runtime["motion_space"],
                "sampler": sampler_name,
                "steps": int(steps),
                "latent_frames": latent_frames,
            }
            row.update(profiler.row())
            latency_rows.append(row)

    if args.profile_latency:
        write_latency_profile(args, runtime, device, latency_rows)


if __name__ == "__main__":
    main()
