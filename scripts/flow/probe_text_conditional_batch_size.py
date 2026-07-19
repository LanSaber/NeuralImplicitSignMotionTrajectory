#!/usr/bin/env python
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from flow.dataset import UpperSMPLXFlowDataset, collate_upper_smplx
from flow.model import apply_model_size_preset, build_text_conditioned_model_from_args, count_parameters
from flow.text_encoder import FrozenT5TextEncoder
from flow.train_text_conditional import compute_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Probe single-GPU batch size for text-conditional flow training.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--text_model_path", type=Path, default=Path("deps/flan-t5-base"))
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    parser.add_argument("--max_text_tokens", type=int, default=64)
    parser.add_argument("--text_conditioning", "--text-conditioning", default="pooled", choices=["pooled", "token_prefix"])
    parser.add_argument(
        "--model_size",
        "--model-size",
        type=lambda value: value.lower(),
        default="custom",
        choices=["custom", "small", "base", "large", "xl"],
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--length_multiple", type=int, default=4)
    parser.add_argument("--noise_samples", type=int, default=8)
    parser.add_argument("--noise_smoothing", type=int, default=9)
    parser.add_argument("--hand_weight", type=float, default=3.0)
    parser.add_argument("--hand_valid_floor", type=float, default=0.2)
    parser.add_argument("--pose_loss_weight", type=float, default=2.0)
    parser.add_argument("--velocity_loss_weight", type=float, default=2.0)
    parser.add_argument("--accel_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def main():
    args = parse_args()
    apply_model_size_preset(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        cuda_index = 0 if device.index is None else device.index
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)

    dataset = UpperSMPLXFlowDataset(
        args.data_dir,
        split="train",
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        length_multiple=args.length_multiple,
        random_crop=False,
        limit=0,
    )
    rows = read_rows(args.data_dir / "meta" / "manifest_train.jsonl")
    dataset.items = sorted(rows, key=lambda row: int(row.get("num_frames", 0)), reverse=True)

    text_encoder = FrozenT5TextEncoder(
        args.text_model_path,
        device=device,
        max_length=args.max_text_tokens,
        local_files_only=True,
        cache=True,
    )
    train_args = Namespace(**vars(args))
    train_args.text_dim = text_encoder.text_dim
    model = build_text_conditioned_model_from_args(train_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    prop = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    if prop is not None:
        print(f"gpu={prop.name} total_gib={prop.total_memory / 1024**3:.2f}")
    print(
        "config "
        f"conditioning={args.text_conditioning} size={args.model_size} "
        f"hidden={args.hidden_dim} layers={args.num_layers} heads={args.num_heads} "
        f"max_frames={args.max_frames} noise_samples={args.noise_samples}"
    )
    print(f"params={count_parameters(model) / 1e6:.2f}M")

    last_ok = None
    for batch_size in args.batch_sizes:
        if batch_size > len(dataset):
            break
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        batch = collate_upper_smplx([dataset[index] for index in range(batch_size)])
        max_len = int(batch["motion"].shape[1])
        start = time.time()
        try:
            losses = compute_losses(model, text_encoder, batch, train_args, device)
            losses["loss"].backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
            else:
                peak_gib = 0.0
                reserved_gib = 0.0
            elapsed = time.time() - start
            print(
                f"OK batch={batch_size} effective={batch_size * args.noise_samples} "
                f"frames={max_len} peak_gib={peak_gib:.2f} reserved_gib={reserved_gib:.2f} "
                f"sec={elapsed:.2f} loss={float(losses['loss'].detach().cpu()):.4f}",
                flush=True,
            )
            last_ok = batch_size
        except RuntimeError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            status = "OOM" if is_oom(exc) else "FAIL"
            print(f"{status} batch={batch_size} effective={batch_size * args.noise_samples} frames={max_len}: {exc}", flush=True)
            break
        finally:
            del batch

    print(f"last_ok_batch={last_ok}")


if __name__ == "__main__":
    main()
