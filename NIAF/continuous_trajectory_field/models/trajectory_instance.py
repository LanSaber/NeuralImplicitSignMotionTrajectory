from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Mapping, Optional

import torch


@dataclass
class TrajectoryInstance:
    """Finite parameters that identify a batch of continuous trajectories.

    The instance intentionally contains no frame-aligned scaffold. Once it has
    been created, motion queries need only this object, shared field weights, and
    continuous timestamps.
    """

    duration_seconds: torch.Tensor
    log_duration_seconds: torch.Tensor
    prior_scale: torch.Tensor
    prior_shift: torch.Tensor
    prior_output_bias: torch.Tensor
    residual_scale: torch.Tensor
    residual_shift: torch.Tensor
    residual_output_bias: torch.Tensor
    local_scale: torch.Tensor
    local_shift: torch.Tensor
    local_output_bias: torch.Tensor
    local_centers: torch.Tensor
    local_widths: torch.Tensor
    local_mask: torch.Tensor
    articulator_gates: torch.Tensor
    local_uncertainty: torch.Tensor
    context_density: Optional[torch.Tensor] = None
    context_tau: Optional[torch.Tensor] = None
    local_part_gates: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.duration_seconds.shape[0])

    @property
    def num_local_fields(self) -> int:
        return int(self.local_centers.shape[1])

    @property
    def device(self) -> torch.device:
        return self.duration_seconds.device

    @property
    def dtype(self) -> torch.dtype:
        return self.duration_seconds.dtype

    def validate(self) -> None:
        batch = self.duration_seconds.shape[0]
        if self.duration_seconds.ndim != 1:
            raise ValueError("duration_seconds must have shape [B]")
        if self.log_duration_seconds.shape != (batch,):
            raise ValueError("log_duration_seconds must have shape [B]")
        for name in (
            "prior_scale",
            "prior_shift",
            "residual_scale",
            "residual_shift",
        ):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[0] != batch:
                raise ValueError(f"{name} must have shape [B,L,H]")
        if self.prior_scale.shape != self.prior_shift.shape:
            raise ValueError("prior scale and shift shapes differ")
        if self.residual_scale.shape != self.residual_shift.shape:
            raise ValueError("residual scale and shift shapes differ")
        for name in ("prior_output_bias", "residual_output_bias", "articulator_gates"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != batch:
                raise ValueError(f"{name} must have shape [B,D]")

        local_shape = self.local_scale.shape
        if len(local_shape) != 4 or local_shape[0] != batch:
            raise ValueError("local_scale must have shape [B,M,L,H]")
        if self.local_shift.shape != local_shape:
            raise ValueError("local scale and shift shapes differ")
        local_count = local_shape[1]
        if self.local_output_bias.ndim != 3 or self.local_output_bias.shape[:2] != (batch, local_count):
            raise ValueError("local_output_bias must have shape [B,M,D]")
        for name in ("local_centers", "local_widths", "local_mask", "local_uncertainty"):
            if getattr(self, name).shape != (batch, local_count):
                raise ValueError(f"{name} must have shape [B,M]")
        if self.local_part_gates is not None and self.local_part_gates.shape != (
            batch,
            local_count,
            4,
        ):
            raise ValueError("local_part_gates must have shape [B,M,4]")
        if self.local_mask.dtype != torch.bool:
            raise ValueError("local_mask must be boolean")
        if torch.is_floating_point(self.duration_seconds):
            if not torch.isfinite(self.duration_seconds).all():
                raise ValueError("duration_seconds contains non-finite values")
            if bool((self.duration_seconds <= 0).any()):
                raise ValueError("duration_seconds must be positive")

    def _map_tensors(self, function) -> "TrajectoryInstance":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = function(value) if torch.is_tensor(value) else value
        return replace(self, **values)

    def to(self, *args, **kwargs) -> "TrajectoryInstance":
        def move(value):
            moved = value.to(*args, **kwargs)
            return moved.bool() if value.dtype == torch.bool else moved

        return self._map_tensors(move)

    def detach(self) -> "TrajectoryInstance":
        return self._map_tensors(lambda value: value.detach())

    def clone(self) -> "TrajectoryInstance":
        return self._map_tensors(lambda value: value.clone())

    def select(self, index: int) -> "TrajectoryInstance":
        index = int(index)
        return self._map_tensors(
            lambda value: value[index : index + 1]
            if value.ndim > 0 and value.shape[0] == self.batch_size
            else value
        )

    def tensor_dict(self, prefix: str = "trajectory_") -> Dict[str, torch.Tensor]:
        output = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                output[f"{prefix}{field.name}"] = value
        return output

    @classmethod
    def from_tensor_dict(
        cls,
        values: Mapping[str, torch.Tensor],
        prefix: str = "trajectory_",
    ) -> "TrajectoryInstance":
        kwargs = {}
        for field in fields(cls):
            key = f"{prefix}{field.name}"
            if key in values:
                kwargs[field.name] = values[key]
            elif field.default is None:
                kwargs[field.name] = None
            else:
                raise KeyError(f"Missing trajectory tensor {key!r}")
        return cls(**kwargs)
