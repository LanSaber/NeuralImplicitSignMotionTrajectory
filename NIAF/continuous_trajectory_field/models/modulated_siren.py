from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def identity_compact_rot6d(pose_dim: int = 256) -> torch.Tensor:
    if int(pose_dim) != 256:
        return torch.zeros(int(pose_dim), dtype=torch.float32)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return torch.cat([identity.repeat(41), torch.zeros(10)], dim=0)


class GroupModulatedSineLayer(nn.Module):
    """SIREN layer with one scale and shift per output channel."""

    def __init__(self, in_dim: int, out_dim: int, omega: float, is_first: bool):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.omega = float(omega)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / max(self.in_dim, 1)
            else:
                bound = math.sqrt(6.0 / max(self.in_dim, 1)) / max(self.omega, 1e-6)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(
        self,
        values: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if scale.ndim != 2 or scale.shape != shift.shape:
            raise ValueError("Layer modulation must have matching [B,H] scale and shift")
        preactivation = F.linear(values, self.linear.weight, self.linear.bias)
        preactivation = preactivation * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return torch.sin(self.omega * preactivation)


class GroupModulatedSiren(nn.Module):
    """Shared SIREN whose finite per-instance state is supplied at query time."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        omega0_first: float = 20.0,
        omega0_hidden: float = 1.0,
        output_init: str = "zero",
        output_weight_scale: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = max(int(depth), 1)
        layers = []
        for index in range(self.depth):
            layers.append(
                GroupModulatedSineLayer(
                    self.input_dim if index == 0 else self.hidden_dim,
                    self.hidden_dim,
                    omega0_first if index == 0 else omega0_hidden,
                    is_first=index == 0,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.uniform_(
            self.output.weight,
            -float(output_weight_scale),
            float(output_weight_scale),
        )
        if output_init == "identity_rot6d":
            with torch.no_grad():
                self.output.bias.copy_(identity_compact_rot6d(self.output_dim))
        elif output_init == "zero":
            nn.init.zeros_(self.output.bias)
        else:
            raise ValueError(f"Unsupported SIREN output initialization {output_init!r}")

    @property
    def modulation_shape(self):
        return self.depth, self.hidden_dim

    def forward(
        self,
        coordinates: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        output_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected coordinates [B,K,{self.input_dim}], got {tuple(coordinates.shape)}"
            )
        expected = (coordinates.shape[0], self.depth, self.hidden_dim)
        if scale.shape != expected or shift.shape != expected:
            raise ValueError(
                f"Expected modulation shape {expected}, got {tuple(scale.shape)} and {tuple(shift.shape)}"
            )
        hidden = coordinates
        for layer_index, layer in enumerate(self.layers):
            hidden = layer(hidden, scale[:, layer_index], shift[:, layer_index])
        output = self.output(hidden)
        if output_bias is not None:
            if output_bias.shape != (coordinates.shape[0], self.output_dim):
                raise ValueError(
                    f"Expected output bias {(coordinates.shape[0], self.output_dim)}, "
                    f"got {tuple(output_bias.shape)}"
                )
            output = output + output_bias[:, None, :]
        return output
