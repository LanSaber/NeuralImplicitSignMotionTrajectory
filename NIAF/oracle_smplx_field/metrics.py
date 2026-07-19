from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from NIAF.oracle_smplx_field.geometry.rotation import EXPR_SLICE, geodesic_loss


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def rot6d_l1(pred, target):
    return float(torch.abs(pred[..., :246] - target[..., :246]).mean().detach().cpu().item())


def expr_l1(pred, target):
    return float(torch.abs(pred[..., EXPR_SLICE] - target[..., EXPR_SLICE]).mean().detach().cpu().item())


def joint_part_l2(pred_parts, target_parts, part):
    pred = pred_parts[part]
    target = target_parts[part].to(device=pred.device, dtype=pred.dtype)
    return float(torch.linalg.norm(pred - target, dim=-1).mean().detach().cpu().item())


def joint_r2(pred, target, eps=1e-8):
    target = target.to(device=pred.device, dtype=pred.dtype)
    sse = (pred - target).pow(2).sum()
    mean = target.mean(dim=0, keepdim=True)
    sst = (target - mean).pow(2).sum()
    return float((1.0 - sse / (sst + eps)).detach().cpu().item())


def path_length(points):
    if points.shape[0] < 2:
        return points.new_tensor(0.0)
    return torch.linalg.norm(points[1:] - points[:-1], dim=-1).sum()


def path_length_ratio(pred, target, eps=1e-8):
    target = target.to(device=pred.device, dtype=pred.dtype)
    return float((path_length(pred) / (path_length(target) + eps)).detach().cpu().item())


def evaluate_pose_prediction(pred_x, target_x, pred_parts, target_parts, prefix):
    row = {}
    row[f"{prefix}_rot6d_l1"] = rot6d_l1(pred_x, target_x)
    row[f"{prefix}_expr_l1"] = expr_l1(pred_x, target_x)
    geo = geodesic_loss(pred_x, target_x)
    row[f"{prefix}_geo_rad"] = float(geo.detach().cpu().item())
    row[f"{prefix}_geo_deg"] = float(geo.detach().cpu().item() * 180.0 / np.pi)
    for part in ["body", "lhand", "rhand", "wholebody"]:
        row[f"{prefix}_{part}_jpe"] = joint_part_l2(pred_parts, target_parts, part)
    row[f"{prefix}_joint_r2"] = joint_r2(pred_parts["wholebody"], target_parts["wholebody"])
    row[f"{prefix}_plr"] = path_length_ratio(pred_parts["wholebody"], target_parts["wholebody"])
    return row


def summarize_rows(rows, group_keys):
    groups = {}
    for row in rows:
        if row.get("error"):
            continue
        key = "/".join(str(row.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, group in groups.items():
        metrics = {}
        numeric_keys = sorted(
            {
                k
                for row in group
                for k, value in row.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        for metric in numeric_keys:
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            if values.size == 0:
                continue
            metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "count": int(values.size),
            }
        summary[key] = metrics
    return summary
