from __future__ import annotations

import math

import torch
from torch import nn


def fourier_encode(x, num_bands=8):
    if x.shape[-1] != 1:
        raise ValueError(f"Expected scalar input with last dim 1, got {tuple(x.shape)}")
    if int(num_bands) <= 0:
        return x
    freqs = torch.pow(
        x.new_tensor(2.0),
        torch.arange(int(num_bands), device=x.device, dtype=x.dtype),
    )
    angles = 2.0 * math.pi * x * freqs.view(*((1,) * (x.ndim - 1)), -1)
    return torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)


def temporal_differences(x, mask):
    valid = mask.unsqueeze(-1).to(x.dtype)
    velocity = torch.zeros_like(x)
    acceleration = torch.zeros_like(x)
    velocity[:, 1:] = x[:, 1:] - x[:, :-1]
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    return velocity * valid, acceleration * valid


class ResidualFlowTransformer(nn.Module):
    def __init__(
        self,
        pose_dim=256,
        text_dim=768,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        time_fourier_bands=8,
        use_text_cross_attention=True,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_fourier_bands = int(time_fourier_bands)
        self.use_text_cross_attention = bool(use_text_cross_attention)
        time_dim = 1 + 2 * self.time_fourier_bands
        input_dim = self.pose_dim * 4 + time_dim * 2

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        if self.use_text_cross_attention:
            self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                self.hidden_dim,
                int(num_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(self.hidden_dim)
            self.null_text = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        else:
            self.text_proj = None
            self.cross_attn = None
            self.cross_norm = None
            self.null_text = None
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.pose_dim)

    def forward(self, residual_t, scaffold, tau, flow_t, mask=None, text_tokens=None, text_mask=None):
        if mask is None:
            mask = torch.ones(residual_t.shape[:2], dtype=torch.bool, device=residual_t.device)
        if flow_t.ndim == 1:
            flow_t = flow_t.view(-1, 1, 1)
        if flow_t.ndim == 2:
            flow_t = flow_t[:, :, None]
        flow_grid = flow_t.expand(-1, residual_t.shape[1], 1)
        scaffold_velocity, scaffold_acceleration = temporal_differences(scaffold, mask)
        features = torch.cat(
            [
                residual_t,
                scaffold,
                scaffold_velocity,
                scaffold_acceleration,
                fourier_encode(tau, self.time_fourier_bands),
                fourier_encode(flow_grid, self.time_fourier_bands),
            ],
            dim=-1,
        )
        hidden = self.input_proj(features)
        hidden = self.temporal_encoder(hidden, src_key_padding_mask=~mask)

        if self.use_text_cross_attention:
            if text_tokens is None:
                memory = self.null_text.expand(residual_t.shape[0], -1, -1)
                key_padding_mask = None
            else:
                memory = self.text_proj(text_tokens.to(dtype=hidden.dtype))
                key_padding_mask = None if text_mask is None else ~text_mask
                hidden = hidden + 0.0 * self.null_text.sum()
            attended, _ = self.cross_attn(
                hidden,
                memory,
                memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden = self.cross_norm(hidden + attended)

        velocity = self.out(self.out_norm(hidden))
        return velocity * mask.unsqueeze(-1).to(velocity.dtype)
