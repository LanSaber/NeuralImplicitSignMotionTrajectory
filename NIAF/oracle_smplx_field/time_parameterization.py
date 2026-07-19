from __future__ import annotations

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
    unit = np.zeros(1, dtype=np.float32) if length == 1 else np.linspace(0.0, 1.0, int(length), dtype=np.float32)
    return torch.from_numpy(to_field_range(unit, field_range)).view(-1, 1)


def joint_arclength(joints, field_range="-1_1"):
    points = np.asarray(joints, dtype=np.float32)
    if points.ndim != 3:
        raise ValueError(f"joints must have shape [T,J,3], got {points.shape}")
    if len(points) <= 1:
        return uniform_progress(len(points), field_range=field_range)
    flat = points.reshape(len(points), -1)
    dist = np.linalg.norm(flat[1:] - flat[:-1], axis=-1)
    unit = _strict_unit_interval(np.concatenate([[0.0], np.cumsum(dist)]))
    return torch.from_numpy(to_field_range(unit, field_range)).view(-1, 1)


def hand_arclength(lhand, rhand, hand_weight=1.0, field_range="-1_1"):
    left = np.asarray(lhand, dtype=np.float32)
    right = np.asarray(rhand, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError(f"hand arrays must both have shape [T,J,3], got {left.shape} and {right.shape}")
    if len(left) <= 1:
        return uniform_progress(len(left), field_range=field_range)
    dist = np.linalg.norm(left[1:] - left[:-1], axis=-1).sum(axis=-1)
    dist += np.linalg.norm(right[1:] - right[:-1], axis=-1).sum(axis=-1)
    dist *= float(hand_weight)
    unit = _strict_unit_interval(np.concatenate([[0.0], np.cumsum(dist)]))
    return torch.from_numpy(to_field_range(unit, field_range)).view(-1, 1)


def make_time_grid(mode, length, joint_parts=None, field_range="-1_1", hand_weight=1.0):
    if mode == "uniform":
        return uniform_progress(length, field_range=field_range)
    if mode in {"joint_arclength", "joint_arc"}:
        if joint_parts is None:
            raise ValueError("joint_arclength requires joint_parts")
        return joint_arclength(joint_parts["wholebody"], field_range=field_range)
    if mode in {"hand_arclength", "hand_arc"}:
        if joint_parts is None:
            raise ValueError("hand_arclength requires joint_parts")
        return hand_arclength(
            joint_parts["lhand"],
            joint_parts["rhand"],
            hand_weight=hand_weight,
            field_range=field_range,
        )
    raise ValueError(f"Unsupported time parameterization mode={mode!r}")
