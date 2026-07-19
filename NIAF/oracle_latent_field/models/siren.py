from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.omega = float(omega)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SirenLatentField(nn.Module):
    def __init__(self, dz=256, hidden=256, depth=3, omega0=30.0, omega=1.0):
        super().__init__()
        layers = [SineLayer(1, hidden, omega=omega0, is_first=True)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(SineLayer(hidden, hidden, omega=omega, is_first=False))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, dz)

    def forward(self, s):
        if s.ndim == 1:
            s = s[:, None]
        return self.out(self.net(s))


def linear_interp_torch(knot_s, knot_z, query_s):
    if query_s.ndim == 1:
        query_s = query_s[:, None]
    knot_s = knot_s.reshape(-1).to(device=query_s.device, dtype=query_s.dtype)
    knot_z = knot_z.to(device=query_s.device, dtype=query_s.dtype)
    q = query_s.reshape(-1)
    if knot_s.numel() == 1:
        return knot_z[:1].expand(q.shape[0], -1)

    idx_hi = torch.searchsorted(knot_s.contiguous(), q.contiguous(), right=False)
    idx_hi = idx_hi.clamp(1, knot_s.numel() - 1)
    idx_lo = idx_hi - 1
    s0 = knot_s[idx_lo]
    s1 = knot_s[idx_hi]
    z0 = knot_z[idx_lo]
    z1 = knot_z[idx_hi]
    weight = ((q - s0) / (s1 - s0).clamp_min(1e-8)).unsqueeze(-1)
    return z0 + weight * (z1 - z0)


class ResidualLatentField(nn.Module):
    def __init__(
        self,
        knot_s,
        knot_z,
        dz=256,
        hidden=256,
        depth=3,
        omega0=30.0,
        omega=1.0,
    ):
        super().__init__()
        self.register_buffer("knot_s", knot_s.detach().float().clone().view(-1, 1))
        self.register_buffer("knot_z", knot_z.detach().float().clone())
        self.residual = SirenLatentField(dz=dz, hidden=hidden, depth=depth, omega0=omega0, omega=omega)

    def forward(self, s):
        base = linear_interp_torch(self.knot_s, self.knot_z, s)
        return base + self.residual(s)
