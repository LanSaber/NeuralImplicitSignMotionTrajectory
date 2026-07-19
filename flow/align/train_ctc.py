#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from flow.align.build_gloss_vocab import build_vocab
from flow.align.ctc_recognizer import GlossCTCRecognizer
from flow.align.dataset import PhoenixGlossCTCDataset, collate_ctc, load_gloss_vocab
from flow.distributed import (
    add_distributed_args,
    barrier,
    cleanup_distributed,
    rank_zero_print,
    resolve_device as resolve_distributed_device,
    setup_distributed,
    unwrap_model,
    wrap_model,
)


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx")
DEFAULT_OUT_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx/meta/ctc_recognizer")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pose-only CTC gloss recognizer for Phoenix.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vocab_path", type=Path, default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--features", choices=["motion", "motion_velocity"], default="motion")
    parser.add_argument("--append_valid", action="store_true")
    parser.add_argument("--gate_hands", action="store_true")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--conv_layers", type=int, default=3)
    parser.add_argument("--conv_kernel", type=int, default=5)
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="soke-flow-align")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--wandb_resume", default=None)
    add_distributed_args(parser)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_vocab(data_dir, vocab_path, train_split):
    path = vocab_path or data_dir / "meta" / "gloss_vocab.json"
    if not path.is_file():
        vocab = build_vocab(data_dir, train_split=train_split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote gloss vocab to {path}")
    return path


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def greedy_decode(log_probs, lengths, blank_id=0):
    predictions = log_probs.argmax(dim=-1).detach().cpu().numpy()
    lengths = lengths.detach().cpu().tolist()
    decoded = []
    for seq, length in zip(predictions, lengths):
        collapsed = []
        prev = None
        for idx in seq[: int(length)]:
            idx = int(idx)
            if idx != blank_id and idx != prev:
                collapsed.append(idx)
            prev = idx
        decoded.append(collapsed)
    return decoded


def lr_for_epoch(base_lr, epoch, epochs, warmup_epochs):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def reduce_sum_tensor(values, device, dist_info):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist_info.get("enabled", False):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.detach().cpu().tolist()


def run_epoch(model, loader, criterion, optimizer, device, blank_id, dist_info, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        motion = batch["motion"].to(device)
        frame_lengths = batch["frame_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(motion, frame_lengths)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            frame_lengths,
            target_lengths,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        count = int(motion.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    total_loss, total_items = reduce_sum_tensor([total_loss, total_items], device, dist_info)
    return total_loss / max(1.0, total_items)


@torch.no_grad()
def evaluate(model, loader, criterion, device, blank_id, dist_info=None):
    dist_info = dist_info or {"enabled": False}
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_edits = 0
    total_ref_tokens = 0
    for batch in loader:
        motion = batch["motion"].to(device)
        frame_lengths = batch["frame_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        log_probs = model(motion, frame_lengths)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            frame_lengths,
            target_lengths,
        )

        decoded = greedy_decode(log_probs, frame_lengths, blank_id=blank_id)
        for pred, ref in zip(decoded, batch["target_sequences"]):
            total_edits += edit_distance(pred, ref)
            total_ref_tokens += len(ref)

        count = int(motion.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_items += count

    total_loss, total_items, total_edits, total_ref_tokens = reduce_sum_tensor(
        [total_loss, total_items, total_edits, total_ref_tokens],
        device,
        dist_info,
    )
    return {
        "loss": total_loss / max(1.0, total_items),
        "wer": total_edits / max(1.0, total_ref_tokens),
    }


def save_checkpoint(path, model, optimizer, epoch, config, metrics, vocab):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
            "vocab": vocab,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    dist_info = setup_distributed(args)
    device = resolve_distributed_device(args.device, dist_info)
    if dist_info["is_main"]:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    vocab_path = ensure_vocab(args.data_dir, args.vocab_path, args.train_split) if dist_info["is_main"] else (args.vocab_path or args.data_dir / "meta" / "gloss_vocab.json")
    barrier(dist_info)
    vocab = load_gloss_vocab(vocab_path)

    dataset_kwargs = {
        "data_dir": args.data_dir,
        "vocab_path": vocab_path,
        "features": args.features,
        "append_valid": args.append_valid,
        "gate_hands": args.gate_hands,
    }
    train_set = PhoenixGlossCTCDataset(
        split=args.train_split,
        drop_oov=True,
        limit=args.limit_train,
        **dataset_kwargs,
    )
    val_set = PhoenixGlossCTCDataset(
        split=args.val_split,
        drop_oov=True,
        limit=args.limit_val,
        **dataset_kwargs,
    )
    if len(train_set) == 0:
        raise RuntimeError("Training set is empty after filtering.")
    if len(val_set) == 0:
        raise RuntimeError("Validation set is empty after filtering.")

    train_sampler = (
        DistributedSampler(train_set, shuffle=True, drop_last=len(train_set) >= dist_info["world_size"] * args.batch_size)
        if dist_info["enabled"]
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_ctc,
        pin_memory=device.type == "cuda",
        drop_last=train_sampler is None and len(train_set) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ctc,
        pin_memory=device.type == "cuda",
    )

    model = GlossCTCRecognizer(
        input_dim=train_set.feature_dim,
        vocab_size=vocab["vocab_size_with_blank"],
        model_dim=args.model_dim,
        conv_layers=args.conv_layers,
        conv_kernel=args.conv_kernel,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    model = wrap_model(model, dist_info, device)
    criterion = nn.CTCLoss(blank=vocab["blank_id"], zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        "data_dir": str(args.data_dir),
        "vocab_path": str(vocab_path),
        "dataset": {
            "features": args.features,
            "append_valid": bool(args.append_valid),
            "gate_hands": bool(args.gate_hands),
            "train_split": args.train_split,
            "val_split": args.val_split,
        },
        "model": unwrap_model(model).config(),
        "optimizer": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
        },
        "distributed": {
            "enabled": bool(dist_info["enabled"]),
            "world_size": int(dist_info["world_size"]),
            "backend": dist_info.get("backend"),
        },
        "wandb": {
            "enabled": bool(args.wandb),
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "id": args.wandb_id,
            "resume": args.wandb_resume,
        },
    }
    wandb_run = None
    if dist_info["is_main"]:
        (args.out_dir / "train_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rank_zero_print(
            dist_info,
            f"Training CTC recognizer on {device}; world_size={dist_info['world_size']}; "
            f"train={len(train_set)} val={len(val_set)} batch_size_per_rank={args.batch_size}",
        )
        if args.wandb:
            import wandb

            wandb_api_key = os.environ.get("WANDB_API_KEY", "")
            wandb_mode = os.environ.get("WANDB_MODE", "").lower()
            if wandb_api_key and wandb_mode not in {"disabled", "dryrun", "offline"}:
                wandb.login(key=wandb_api_key, relogin=True)
            wandb_kwargs = {
                "project": args.wandb_project,
                "name": args.wandb_run_name,
                "config": config,
            }
            if args.wandb_id:
                wandb_kwargs["id"] = args.wandb_id
            if args.wandb_resume:
                wandb_kwargs["resume"] = args.wandb_resume
            wandb_run = wandb.init(**wandb_kwargs)

    best_wer = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            lr = lr_for_epoch(args.lr, epoch, args.epochs, args.warmup_epochs)
            set_optimizer_lr(optimizer, lr)
            train_loss = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                blank_id=vocab["blank_id"],
                dist_info=dist_info,
                grad_clip=args.grad_clip,
            )
            if dist_info["is_main"]:
                val_metrics = evaluate(
                    unwrap_model(model),
                    val_loader,
                    criterion,
                    device,
                    blank_id=vocab["blank_id"],
                    dist_info={"enabled": False},
                )
            else:
                val_metrics = {"loss": float("nan"), "wer": float("nan")}
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_wer": val_metrics["wer"],
                "lr": lr,
            }
            rank_zero_print(
                dist_info,
                f"epoch={epoch:03d} lr={lr:.3e} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_wer={val_metrics['wer']:.4f}",
                flush=True,
            )

            if dist_info["is_main"]:
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": train_loss,
                            "val/loss": val_metrics["loss"],
                            "val/wer": val_metrics["wer"],
                            "lr": lr,
                            "epoch": epoch,
                        },
                        step=epoch,
                    )
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    config,
                    metrics,
                    vocab,
                )
                if val_metrics["wer"] < best_wer:
                    best_wer = val_metrics["wer"]
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        unwrap_model(model),
                        optimizer,
                        epoch,
                        config,
                        metrics,
                        vocab,
                    )
            barrier(dist_info)

        rank_zero_print(dist_info, f"Finished training. Best val WER={best_wer:.4f}; checkpoints in {checkpoint_dir}")
    finally:
        if dist_info["is_main"] and wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
