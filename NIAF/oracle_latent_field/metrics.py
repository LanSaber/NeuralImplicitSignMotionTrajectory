from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def latent_metrics(pred, target, prefix=""):
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    diff = pred - target
    mean = target.mean(dim=0, keepdim=True)
    ss_res = diff.pow(2).sum()
    ss_tot = (target - mean).pow(2).sum().clamp_min(1e-12)
    l2_num = torch.linalg.norm(diff, dim=-1).sum()
    l2_den = torch.linalg.norm(target - mean, dim=-1).sum().clamp_min(1e-12)
    name = f"{prefix}_" if prefix else ""
    return {
        f"{name}mae_z": float(diff.abs().mean().item()),
        f"{name}l2_z": float(torch.linalg.norm(diff, dim=-1).mean().item()),
        f"{name}rel_l2_z": float((l2_num / l2_den).item()),
        f"{name}r2_z": float((1.0 - ss_res / ss_tot).item()),
    }


def adjacent_random_ratio(z, non_neighbor_margin=4, samples=4096, seed=123):
    z = z.detach().cpu().float().numpy() if torch.is_tensor(z) else np.asarray(z, dtype=np.float32)
    if len(z) < 3:
        return {"adjacent_l2_mean": 0.0, "random_l2_mean": 0.0, "smoothness_ratio": 0.0}
    adjacent = np.linalg.norm(z[1:] - z[:-1], axis=-1)
    rng = np.random.default_rng(seed)
    pairs = []
    attempts = 0
    while len(pairs) < samples and attempts < samples * 20:
        i = int(rng.integers(0, len(z)))
        j = int(rng.integers(0, len(z)))
        if abs(i - j) > non_neighbor_margin:
            pairs.append((i, j))
        attempts += 1
    if pairs:
        random_dist = np.asarray([np.linalg.norm(z[i] - z[j]) for i, j in pairs], dtype=np.float64)
        random_mean = float(random_dist.mean())
    else:
        random_mean = float(adjacent.mean())
    adjacent_mean = float(adjacent.mean())
    return {
        "adjacent_l2_mean": adjacent_mean,
        "adjacent_l2_p95": float(np.percentile(adjacent, 95.0)),
        "random_l2_mean": random_mean,
        "smoothness_ratio": adjacent_mean / max(random_mean, 1e-12),
    }


def feature_sequence_metrics(pred, target, prefix="field"):
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    out = {
        f"{prefix}_feature_mae": float((pred - target).abs().mean().item()),
        f"{prefix}_feature_l2": float(torch.linalg.norm(pred - target, dim=-1).mean().item()),
    }
    if pred.shape[0] > 1 and target.shape[0] > 1:
        out[f"{prefix}_vel_mae"] = float(((pred[1:] - pred[:-1]) - (target[1:] - target[:-1])).abs().mean().item())
        out[f"{prefix}_pose_vel_mag"] = float(torch.linalg.norm(pred[1:] - pred[:-1], dim=-1).mean().item())
    if pred.shape[0] > 2 and target.shape[0] > 2:
        pred_acc = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
        target_acc = target[2:] - 2.0 * target[1:-1] + target[:-2]
        out[f"{prefix}_acc_mae"] = float((pred_acc - target_acc).abs().mean().item())
        out[f"{prefix}_pose_acc_mag"] = float(torch.linalg.norm(pred_acc, dim=-1).mean().item())
    if pred.shape[0] > 3:
        jerk = pred[3:] - 3.0 * pred[2:-1] + 3.0 * pred[1:-2] - pred[:-3]
        out[f"{prefix}_pose_jerk_mag"] = float(torch.linalg.norm(jerk, dim=-1).mean().item())
    return out


def summarize_rows(rows, keys):
    summary = {}
    groups = {}
    for row in rows:
        group_key = tuple(row.get(key) for key in keys)
        groups.setdefault(group_key, []).append(row)
    skip = set(keys) | {"sample_id", "index", "text", "error"}
    for group_key, items in groups.items():
        label = "/".join(str(v) for v in group_key)
        values = {}
        for key in sorted(set().union(*(item.keys() for item in items)) - skip):
            nums = [item[key] for item in items if isinstance(item.get(key), (int, float))]
            if nums:
                arr = np.asarray(nums, dtype=np.float64)
                values[key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                    "count": int(len(arr)),
                }
        summary[label] = values
    return summary


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
