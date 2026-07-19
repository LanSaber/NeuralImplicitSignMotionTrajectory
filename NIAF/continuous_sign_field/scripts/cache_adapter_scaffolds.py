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
from NIAF.continuous_sign_field.scaffold_provider import ScaffoldProvider
from NIAF.continuous_sign_field.scripts.export_generation_npz import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Cache text-conditioned adapter scaffolds for NIAF training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cache_path(cache_dir, motion_path):
    return Path(cache_dir) / str(motion_path)


def cache_is_complete(path, require_retrieval_features=False):
    if not path.is_file():
        return False
    if not require_retrieval_features:
        return True
    try:
        with np.load(path) as data:
            return "retrieval_features" in data and "retrieval_feature_names" in data
    except (OSError, ValueError):
        return False


def save_scaffold(path, scaffold, meta, retrieval_features=None, retrieval_feature_names=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    payload = {
        "scaffold": scaffold.astype(np.float32),
        "length": np.asarray(len(scaffold), dtype=np.int32),
        "text": np.asarray(str(meta["text"])),
        "name": np.asarray(str(meta["name"])),
        "adapter_checkpoint": np.asarray(str(meta["adapter_checkpoint"])),
    }
    if retrieval_features is not None:
        payload["retrieval_features"] = retrieval_features.astype(np.float32)
        payload["retrieval_feature_names"] = np.asarray(tuple(retrieval_feature_names))
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    scaffold_cfg = cfg.setdefault("scaffold", {})
    cache_dir = scaffold_cfg.get("cache_dir")
    if not cache_dir:
        raise ValueError("The config must define scaffold.cache_dir.")
    scaffold_cfg["cache_only"] = False
    scaffold_cfg["prefer_cache"] = False
    device = resolve_device(args.device)

    datasets = {
        split: ContinuousSignDataset(cfg, split=split, limit=0, random_crop=False, require_fk_cache=False)
        for split in args.splits
    }
    provider = ScaffoldProvider(cfg, datasets[args.splits[0]], device)
    checkpoint = cfg.get("adapter", {}).get("checkpoint", "")
    word_data_dir = Path(
        cfg.get("adapter", {}).get("word_data_dir")
        or cfg.get("data", {}).get("word_data_dir", "")
    )
    word_split = str(cfg.get("adapter", {}).get("word_split", ""))
    word_manifest = word_data_dir / "meta" / f"manifest_{word_split}.jsonl"
    require_retrieval_features = bool(scaffold_cfg.get("require_retrieval_features", False))
    summary = {
        "config": str(args.config),
        "adapter_checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "word_split": word_split,
        "word_manifest": str(word_manifest.resolve()),
        "require_retrieval_features": require_retrieval_features,
        "retrieval_feature_names": list(RETRIEVAL_FEATURE_NAMES),
        "splits": {},
    }

    for split, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=max(int(args.batch_size), 1),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_continuous_sign,
        )
        written = 0
        skipped = 0
        for batch in tqdm(loader, desc=f"cache adapter scaffold {split}"):
            requested_paths = [cache_path(cache_dir, path) for path in batch["motion_path"]]
            needs_write = [
                args.overwrite or not cache_is_complete(path, require_retrieval_features)
                for path in requested_paths
            ]
            if not any(needs_write):
                skipped += len(requested_paths)
                continue
            batch["mask"] = batch["mask"].to(device)
            batch["length"] = batch["length"].to(device)
            scaffold, _anchor_mask, metadata = provider.build_with_metadata(batch, x=None, use_cache=False)
            retrieval_features = metadata.get("retrieval_features")
            retrieval_feature_names = metadata.get("retrieval_feature_names", ())
            for idx, (path, should_write) in enumerate(zip(requested_paths, needs_write)):
                if not should_write:
                    skipped += 1
                    continue
                length = int(batch["length"][idx].item())
                value = scaffold[idx, :length].detach().cpu().float().numpy()
                evidence = None
                if retrieval_features is not None:
                    evidence = retrieval_features[idx, :length].detach().cpu().float().numpy()
                save_scaffold(
                    path,
                    value,
                    {
                        "name": batch["name"][idx],
                        "text": batch["text"][idx],
                        "adapter_checkpoint": checkpoint,
                    },
                    retrieval_features=evidence,
                    retrieval_feature_names=retrieval_feature_names,
                )
                written += 1
        summary["splits"][split] = {
            "dataset_size": len(dataset),
            "written": written,
            "skipped": skipped,
        }

    summary_path = Path(cache_dir) / "cache_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
