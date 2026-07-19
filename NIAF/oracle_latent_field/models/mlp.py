from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatureMLP(nn.Module):
    def __init__(self, dz=256, hidden=256, depth=3, num_frequencies=16, activation="gelu"):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        basis = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("basis", basis.view(1, -1))
        in_dim = 1 + 2 * self.num_frequencies
        act = nn.GELU if activation == "gelu" else nn.ReLU
        layers = []
        cur = in_dim
        for _ in range(max(int(depth), 1)):
            layers.extend([nn.Linear(cur, hidden), act()])
            cur = hidden
        layers.append(nn.Linear(cur, dz))
        self.net = nn.Sequential(*layers)

    def encode(self, s):
        if s.ndim == 1:
            s = s[:, None]
        phases = s * self.basis.to(device=s.device, dtype=s.dtype)
        return torch.cat([s, torch.sin(phases), torch.cos(phases)], dim=-1)

    def forward(self, s):
        return self.net(self.encode(s))


class ReLUMlp(nn.Module):
    def __init__(self, dz=256, hidden=256, depth=3):
        super().__init__()
        layers = []
        cur = 1
        for _ in range(max(int(depth), 1)):
            layers.extend([nn.Linear(cur, hidden), nn.ReLU()])
            cur = hidden
        layers.append(nn.Linear(cur, dz))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        if s.ndim == 1:
            s = s[:, None]
        return self.net(s)
