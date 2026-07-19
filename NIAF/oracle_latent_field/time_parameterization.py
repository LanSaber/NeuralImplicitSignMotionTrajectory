from __future__ import annotations

import math

import numpy as np
import torch


def _strict_unit_interval(values, eps=1e-6):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    values = values - values[0]
    total = float(values[-1])
    if not np.isfinite(total) or total <= eps:
        return np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    values = values / total
    for idx in range(1, len(values)):
        values[idx] = max(values[idx], values[idx - 1] + eps)
    values = (values - values[0]) / max(values[-1] - values[0], eps)
    return values.astype(np.float32)


def to_field_range(unit_values, field_range="-1_1"):
    values = np.asarray(unit_values, dtype=np.float32)
    if field_range in {"0_1", "[0,1]", "unit"}:
        return values
    if field_range in {"-1_1", "[-1,1]", "siren"}:
        return values * 2.0 - 1.0
    raise ValueError(f"Unsupported field_range={field_range!r}")


def uniform_progress(length, field_range="-1_1"):
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if length == 1:
        unit = np.zeros(1, dtype=np.float32)
    else:
        unit = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
    return torch.from_numpy(to_field_range(unit, field_range)).view(-1, 1)


def latent_arclength(z, field_range="-1_1"):
    z_np = z.detach().cpu().float().numpy() if torch.is_tensor(z) else np.asarray(z, dtype=np.float32)
    if len(z_np) <= 1:
        return uniform_progress(len(z_np), field_range=field_range)
    dist = np.linalg.norm(z_np[1:] - z_np[:-1], axis=-1)
    unit = _strict_unit_interval(np.concatenate([[0.0], np.cumsum(dist)]))
    return torch.from_numpy(to_field_range(unit, field_range)).view(-1, 1)


def latent_token_frame_indices(num_frames, latent_length, downsample_factor=4):
    if latent_length <= 0:
        return np.zeros(0, dtype=np.int64)
    if num_frames <= 1:
        return np.zeros(latent_length, dtype=np.int64)
    centers = (np.arange(latent_length, dtype=np.float64) + 0.5) * float(downsample_factor) - 0.5
    if latent_length > 1:
        lin = np.linspace(0.0, float(num_frames - 1), latent_length, dtype=np.float64)
        centers = np.minimum(centers, lin + float(downsample_factor))
    return np.clip(np.rint(centers), 0, num_frames - 1).astype(np.int64)


def pose_arclength(pose_points, latent_length, downsample_factor=4, field_range="-1_1"):
    points = np.asarray(pose_points, dtype=np.float32)
    if points.ndim < 2:
        raise ValueError(f"pose_points must have shape [T, ...], got {points.shape}")
    if latent_length <= 1:
        return uniform_progress(latent_length, field_range=field_range)
    frame_idx = latent_token_frame_indices(len(points), latent_length, downsample_factor=downsample_factor)
    flat = points[frame_idx].reshape(latent_length, -1)
    dist = np.linalg.norm(flat[1:] - flat[:-1], axis=-1)
    unit = _strict_unit_interval(np.concatenate([[0.0], np.cumsum(dist)]))
    return torch.from_numpy(to_field_range(unit, field_range)).view(-1, 1)


def make_time_grid(mode, z, pose_points=None, downsample_factor=4, field_range="-1_1"):
    length = int(z.shape[0])
    if mode == "uniform":
        return uniform_progress(length, field_range=field_range)
    if mode in {"latent_arclength", "latent_arc"}:
        return latent_arclength(z, field_range=field_range)
    if mode in {"pose_arclength", "pose_arc"}:
        if pose_points is None:
            raise ValueError("pose_arclength requires pose_points")
        return pose_arclength(
            pose_points,
            length,
            downsample_factor=downsample_factor,
            field_range=field_range,
        )
    raise ValueError(f"Unsupported time parameterization mode={mode!r}")


def latent_length_from_frames(num_frames, downsample_factor=4):
    return int(math.ceil(int(num_frames) / float(downsample_factor)))
