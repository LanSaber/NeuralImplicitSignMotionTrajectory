from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from flow.smplx_features import (
    COMPACT6D_EXPRESSION,
    COMPACT6D_JAW,
    COMPACT6D_LEFT_HAND,
    COMPACT6D_RIGHT_HAND,
    COMPACT6D_UPPER_BODY,
)
from NIAF.continuous_sign_field.models.meta_implicit import (
    SineLayer,
    fourier_encode_scalar,
    masked_mean,
    temporal_differences,
)


def _masked_window_pool(values, mask, stride):
    stride = max(int(stride), 1)
    batch, frames, dim = values.shape
    windows = max(int(math.ceil(frames / stride)), 1)
    padded_frames = windows * stride
    if padded_frames > frames:
        values = F.pad(values, (0, 0, 0, padded_frames - frames))
        mask = F.pad(mask, (0, padded_frames - frames), value=False)
    values = values.reshape(batch, windows, stride, dim)
    window_mask = mask.reshape(batch, windows, stride)
    weights = window_mask.to(values.dtype).unsqueeze(-1)
    pooled = (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    return pooled, window_mask.any(dim=2)


def interpolate_local_codes(local_codes, local_mask, tau):
    if tau.ndim == 3:
        tau = tau.squeeze(-1)
    counts = local_mask.sum(dim=1).clamp_min(1)
    coordinate = tau.clamp(0.0, 1.0) * (counts - 1).to(tau.dtype).unsqueeze(1)
    lower = torch.floor(coordinate).long()
    upper = torch.minimum(lower + 1, (counts - 1).unsqueeze(1))
    fraction = (coordinate - lower.to(coordinate.dtype)).unsqueeze(-1)
    code_dim = local_codes.shape[-1]
    lower_code = torch.gather(local_codes, 1, lower.unsqueeze(-1).expand(-1, -1, code_dim))
    upper_code = torch.gather(local_codes, 1, upper.unsqueeze(-1).expand(-1, -1, code_dim))
    return lower_code + fraction * (upper_code - lower_code)


class TextDurationHead(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, initial_frames=80.0, dropout=0.0):
        super().__init__()
        self.text_dim = int(text_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(max(float(initial_frames), 1.0)))

    def forward(self, text_tokens, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        pooled = masked_mean(text_tokens, text_mask, dim=1)
        return self.net(pooled).squeeze(-1)


class LocalCodePredictor(nn.Module):
    def __init__(
        self,
        pose_dim=256,
        text_dim=768,
        code_dim=128,
        hidden_dim=256,
        local_stride=8,
        num_layers=2,
        num_heads=8,
        time_fourier_bands=6,
        dropout=0.0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.text_dim = int(text_dim)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.local_stride = max(int(local_stride), 1)
        self.time_fourier_bands = int(time_fourier_bands)
        time_dim = 1 + 2 * self.time_fourier_bands
        self.scaffold_proj = nn.Sequential(
            nn.Linear(self.pose_dim * 3 + time_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.code_dim),
        )

    def forward(self, scaffold, mask, tau, text_tokens, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        velocity, acceleration = temporal_differences(scaffold, mask)
        frame_features = torch.cat(
            [
                scaffold,
                velocity,
                acceleration,
                fourier_encode_scalar(tau, self.time_fourier_bands),
            ],
            dim=-1,
        )
        local_features, local_mask = _masked_window_pool(frame_features, mask, self.local_stride)
        local_features = self.scaffold_proj(local_features)
        text_features = self.text_proj(text_tokens.to(dtype=scaffold.dtype))
        attended, _attention = self.cross_attention(
            local_features,
            text_features,
            text_features,
            key_padding_mask=~text_mask.bool(),
            need_weights=False,
        )
        encoded = self.temporal_encoder(
            local_features + attended,
            src_key_padding_mask=~local_mask,
        )
        codes = self.out(encoded) * local_mask.unsqueeze(-1).to(encoded.dtype)
        return codes, local_mask


class LocalAmortizedImplicitResidualField(nn.Module):
    def __init__(
        self,
        pose_dim=256,
        text_dim=768,
        code_dim=128,
        context_hidden_dim=256,
        context_layers=2,
        context_heads=8,
        local_stride=8,
        hidden_dim=256,
        depth=4,
        time_fourier_bands=10,
        context_time_fourier_bands=6,
        omega0_first=20.0,
        omega0_hidden=1.0,
        residual_scale_init=0.1,
        residual_scale_learnable=True,
        body_gate_bias=-3.0,
        hand_gate_bias=-2.0,
        face_gate_bias=-3.0,
        duration_hidden_dim=256,
        duration_initial_frames=80.0,
        dropout=0.0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.local_stride = max(int(local_stride), 1)
        self.time_fourier_bands = int(time_fourier_bands)
        self.local_code_predictor = LocalCodePredictor(
            pose_dim=self.pose_dim,
            text_dim=int(text_dim),
            code_dim=self.code_dim,
            hidden_dim=int(context_hidden_dim),
            local_stride=self.local_stride,
            num_layers=int(context_layers),
            num_heads=int(context_heads),
            time_fourier_bands=int(context_time_fourier_bands),
            dropout=float(dropout),
        )
        self.duration_head = TextDurationHead(
            text_dim=int(text_dim),
            hidden_dim=int(duration_hidden_dim),
            initial_frames=float(duration_initial_frames),
            dropout=float(dropout),
        )

        time_dim = 1 + 2 * self.time_fourier_bands
        input_dim = time_dim + self.pose_dim * 3 + self.code_dim
        layers = [SineLayer(input_dim, hidden_dim, omega=omega0_first, is_first=True)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega=omega0_hidden, is_first=False))
        self.net = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(int(hidden_dim))
        self.out = nn.Linear(int(hidden_dim), self.pose_dim)
        self.gate = nn.Linear(int(hidden_dim), 4)
        nn.init.uniform_(self.out.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(
                torch.tensor(
                    [body_gate_bias, hand_gate_bias, hand_gate_bias, face_gate_bias],
                    dtype=self.gate.bias.dtype,
                )
            )

        scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def predict_log_frames(self, text_tokens, text_mask=None):
        return self.duration_head(text_tokens, text_mask=text_mask)

    def predict_lengths(self, text_tokens, text_mask=None, min_frames=40, max_frames=400, multiple=4):
        frames = torch.exp(self.predict_log_frames(text_tokens, text_mask=text_mask))
        frames = frames.clamp(float(min_frames), float(max_frames))
        multiple = max(int(multiple), 1)
        if multiple > 1:
            frames = torch.round(frames / multiple) * multiple
        return frames.round().long().clamp(int(min_frames), int(max_frames))

    def encode_local_codes(self, tau, scaffold, mask, text_tokens, text_mask=None):
        return self.local_code_predictor(
            scaffold,
            mask,
            tau,
            text_tokens,
            text_mask=text_mask,
        )

    def _apply_group_gates(self, raw_residual, gates):
        if self.pose_dim != 256:
            return raw_residual * gates.mean(dim=-1, keepdim=True)
        residual = torch.zeros_like(raw_residual)
        residual[..., COMPACT6D_UPPER_BODY] = raw_residual[..., COMPACT6D_UPPER_BODY] * gates[..., 0:1]
        residual[..., COMPACT6D_LEFT_HAND] = raw_residual[..., COMPACT6D_LEFT_HAND] * gates[..., 1:2]
        residual[..., COMPACT6D_RIGHT_HAND] = raw_residual[..., COMPACT6D_RIGHT_HAND] * gates[..., 2:3]
        residual[..., COMPACT6D_JAW] = raw_residual[..., COMPACT6D_JAW] * gates[..., 3:4]
        residual[..., COMPACT6D_EXPRESSION] = raw_residual[..., COMPACT6D_EXPRESSION] * gates[..., 3:4]
        return residual

    def forward(self, tau, scaffold, mask, text_tokens, text_mask=None):
        if tau.ndim == 2:
            tau = tau.unsqueeze(-1)
        local_codes, local_mask = self.encode_local_codes(
            tau,
            scaffold,
            mask,
            text_tokens,
            text_mask=text_mask,
        )
        frame_codes = interpolate_local_codes(local_codes, local_mask, tau)
        velocity, acceleration = temporal_differences(scaffold, mask)
        features = torch.cat(
            [
                fourier_encode_scalar(tau, self.time_fourier_bands),
                scaffold,
                velocity,
                acceleration,
                frame_codes,
            ],
            dim=-1,
        )
        hidden = self.out_norm(self.net(features))
        gates = torch.sigmoid(self.gate(hidden))
        raw_residual = self.out(hidden)
        residual = self.residual_scale.to(dtype=scaffold.dtype) * self._apply_group_gates(raw_residual, gates)
        residual = residual * mask.unsqueeze(-1).to(residual.dtype)
        return {
            "residual": residual,
            "prediction": scaffold + residual,
            "local_codes": local_codes,
            "local_code_mask": local_mask,
            "frame_codes": frame_codes,
            "gates": gates * mask.unsqueeze(-1).to(gates.dtype),
            "pred_log_frames": self.predict_log_frames(text_tokens, text_mask=text_mask),
        }
