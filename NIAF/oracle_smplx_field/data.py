from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flow.dataset import UpperSMPLXFlowDataset
from flow.evaluate.dtw_mpjpe_t2m_default import (
    resolve_betas_override,
    smplx_to_joints_vertices,
    t2m_default_parts,
)
from flow.render import DEFAULT_MODEL_DIR
from flow.smplx_features import (
    COMPACT6D_DIM,
    COMPACT6D_EXPRESSION,
    compact_rot6d_to_axis_angle,
    smplx182_from_compact,
)


@dataclass
class OracleSMPLXSample:
    index: int
    sample_id: str
    text: str
    x_norm: torch.Tensor
    x_rot6d: torch.Tensor
    x_axis: np.ndarray
    smplx: np.ndarray
    length: int
    fps: float | None = None


def build_dataset(cfg):
    data_cfg = cfg["data"]
    return UpperSMPLXFlowDataset(
        data_cfg["data_dir"],
        split=data_cfg.get("split", "val"),
        manifest_path=data_cfg.get("manifest_path"),
        min_frames=int(data_cfg.get("min_frames", 40)),
        max_frames=int(data_cfg.get("max_frames", 200)),
        length_multiple=int(data_cfg.get("length_multiple", 4)),
        random_crop=False,
        limit=int(data_cfg.get("limit", 0)),
        rotation_rep="rot6d",
    )


def adjusted_length_for_item(dataset, item):
    if "num_frames" in item:
        return int(dataset._target_length(int(item["num_frames"])))
    path = Path(dataset.data_dir) / item["motion_path"]
    with np.load(path) as data:
        return int(dataset._target_length(len(data["motion"])))


def select_pilot_indices(dataset, cfg):
    data_cfg = cfg["data"]
    max_sequences = int(data_cfg.get("max_sequences", 3))
    min_frames = int(data_cfg.get("min_frames", 40))
    max_frames = int(data_cfg.get("max_frames", 200))
    seed = int(cfg.get("seed", 1234))
    candidates = []
    for idx, item in enumerate(dataset.items):
        length = adjusted_length_for_item(dataset, item)
        if min_frames <= length <= max_frames:
            candidates.append((idx, length))
    if max_sequences <= 0 or len(candidates) <= max_sequences:
        return [idx for idx, _ in candidates]

    buckets = {
        "short": [idx for idx, length in candidates if length < 80],
        "medium": [idx for idx, length in candidates if 80 <= length < 160],
        "long": [idx for idx, length in candidates if length >= 160],
    }
    rng = np.random.default_rng(seed)
    selected = []
    per_bucket = max_sequences // 3
    remainder = max_sequences - per_bucket * 3
    for offset, name in enumerate(["short", "medium", "long"]):
        bucket = list(buckets[name])
        rng.shuffle(bucket)
        selected.extend(bucket[: min(len(bucket), per_bucket + (1 if offset < remainder else 0))])
    if len(selected) < max_sequences:
        selected_set = set(selected)
        remaining = [idx for idx, _ in candidates if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_sequences - len(selected)])
    return selected[:max_sequences]


def denormalize_motion(x_norm, dataset):
    x_np = x_norm.detach().cpu().float().numpy() if torch.is_tensor(x_norm) else np.asarray(x_norm, dtype=np.float32)
    return (x_np * dataset.std[None] + dataset.mean[None]).astype(np.float32)


def _item_fps(dataset, index):
    item = dataset.items[index]
    if "fps" in item:
        try:
            return float(item["fps"])
        except Exception:
            return None
    return None


def load_oracle_sample(dataset, index, device):
    item = dataset[index]
    x_norm = item["motion"].to(device=device, dtype=torch.float32)
    x_rot6d_np = denormalize_motion(x_norm, dataset)
    if x_rot6d_np.shape[-1] != COMPACT6D_DIM:
        raise ValueError(f"Expected rot6D compact dim {COMPACT6D_DIM}, got {x_rot6d_np.shape}")
    x_axis = compact_rot6d_to_axis_angle(x_rot6d_np)
    smplx = smplx182_from_compact(x_axis)
    return OracleSMPLXSample(
        index=int(index),
        sample_id=str(item["name"]),
        text=str(item.get("text", "")),
        x_norm=x_norm.detach(),
        x_rot6d=torch.from_numpy(x_rot6d_np).to(device=device, dtype=torch.float32),
        x_axis=x_axis,
        smplx=smplx,
        length=int(item["length"]),
        fps=_item_fps(dataset, index),
    )


def save_motion_npz(path, x_rot6d, sample, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_np = x_rot6d.detach().cpu().float().numpy() if torch.is_tensor(x_rot6d) else np.asarray(x_rot6d, dtype=np.float32)
    axis = compact_rot6d_to_axis_angle(x_np)
    smplx = smplx182_from_compact(axis)
    np.savez_compressed(
        path,
        motion=axis.astype(np.float32),
        representation=x_np.astype(np.float32),
        smplx=smplx.astype(np.float32),
        rotation_rep="rot6d",
        name=str(sample.sample_id),
        text=str(sample.text),
        length=int(sample.length),
        label=str(label),
        source_index=int(sample.index),
    )


def compute_joint_parts(
    smplx_params,
    model_dir=DEFAULT_MODEL_DIR,
    gender="NEUTRAL",
    device="cpu",
    batch_size=128,
    betas_mode="h2s_fixed",
):
    joints, vertices = smplx_to_joints_vertices(
        smplx_params,
        model_dir=Path(model_dir),
        gender=gender,
        device=device,
        batch_size=batch_size,
        betas_override=resolve_betas_override(betas_mode),
    )
    return t2m_default_parts(joints, vertices)


def sequence_length_from_frames(num_frames):
    return int(math.ceil(int(num_frames)))


def expression_slice():
    return COMPACT6D_EXPRESSION
