from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_sign_field.models.meta_implicit import masked_mean, masked_std
from NIAF.continuous_trajectory_field.models.trajectory_instance import (
    TrajectoryInstance,
)


def _first_last(values: torch.Tensor, lengths: torch.Tensor):
    first = values[:, 0]
    last_index = (lengths.to(values.device) - 1).clamp_min(0)
    last = values[torch.arange(values.shape[0], device=values.device), last_index]
    return first, last


def _temporal_difference(values: torch.Tensor, mask: torch.Tensor):
    velocity = torch.zeros_like(values)
    acceleration = torch.zeros_like(values)
    velocity[:, 1:] = values[:, 1:] - values[:, :-1]
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    valid = mask.unsqueeze(-1).to(values.dtype)
    return velocity * valid, acceleration * valid


def _normalized_context_time(mask: torch.Tensor, dtype: torch.dtype):
    batch, frames = mask.shape
    lengths = mask.sum(dim=1).clamp_min(1)
    index = torch.arange(frames, device=mask.device, dtype=dtype).view(1, frames)
    denominator = (lengths - 1).clamp_min(1).to(dtype).unsqueeze(1)
    tau = -1.0 + 2.0 * index / denominator
    tau = torch.where(lengths.unsqueeze(1) > 1, tau, torch.zeros_like(tau))
    return tau.clamp(-1.0, 1.0) * mask.to(dtype)


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        residual = values
        values = self.norm(values).transpose(1, 2)
        values = self.depthwise(values)
        left, right = self.pointwise(values).chunk(2, dim=1)
        values = (left * torch.sigmoid(right)).transpose(1, 2)
        values = residual + self.dropout(self.out(values))
        return values * mask.unsqueeze(-1).to(values.dtype)


class TextDurationHead(nn.Module):
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        initial_seconds: float,
        minimum_seconds: float,
        maximum_seconds: float,
        dropout: float,
    ):
        super().__init__()
        self.minimum_seconds = float(minimum_seconds)
        self.maximum_seconds = float(maximum_seconds)
        self.net = nn.Sequential(
            nn.LayerNorm(int(text_dim)),
            nn.Linear(int(text_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(max(float(initial_seconds), 1e-3)))

    def forward(self, pooled_text: torch.Tensor):
        log_duration = self.net(pooled_text).squeeze(-1)
        duration = torch.exp(log_duration).clamp(self.minimum_seconds, self.maximum_seconds)
        return log_duration, duration


class TrajectoryHypernetwork(nn.Module):
    """Create finite continuous-field parameters from text and adapter context."""

    def __init__(
        self,
        text_dim: int = 768,
        pose_dim: int = COMPACT6D_DIM,
        retrieval_dim: int = len(RETRIEVAL_FEATURE_NAMES),
        context_hidden_dim: int = 256,
        context_layers: int = 3,
        field_hidden_dim: int = 256,
        field_depth: int = 4,
        residual_dim: int = 133,
        max_local_fields: int = 24,
        frames_per_local_field: int = 20,
        minimum_local_width: float = 0.06,
        maximum_local_width: float = 0.50,
        quantile_temperature: float = 0.02,
        initial_duration_seconds: float = 4.0,
        minimum_duration_seconds: float = 0.8,
        maximum_duration_seconds: float = 20.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.pose_dim = int(pose_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.context_hidden_dim = int(context_hidden_dim)
        self.field_hidden_dim = int(field_hidden_dim)
        self.field_depth = int(field_depth)
        self.residual_dim = int(residual_dim)
        self.max_local_fields = max(int(max_local_fields), 0)
        self.frames_per_local_field = max(int(frames_per_local_field), 1)
        self.minimum_local_width = float(minimum_local_width)
        self.maximum_local_width = float(maximum_local_width)
        self.quantile_temperature = float(quantile_temperature)

        frame_input_dim = self.pose_dim * 3 + self.retrieval_dim + 1
        self.frame_input = nn.Sequential(
            nn.Linear(frame_input_dim, self.context_hidden_dim),
            nn.LayerNorm(self.context_hidden_dim),
            nn.SiLU(),
        )
        self.context_blocks = nn.ModuleList(
            [
                TemporalConvBlock(self.context_hidden_dim, kernel_size=5, dropout=dropout)
                for _ in range(max(int(context_layers), 1))
            ]
        )
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.text_projection = nn.Linear(self.text_dim, self.context_hidden_dim)

        global_input_dim = self.context_hidden_dim * 5
        self.global_context = nn.Sequential(
            nn.LayerNorm(global_input_dim),
            nn.Linear(global_input_dim, self.context_hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.context_hidden_dim * 2, self.context_hidden_dim),
            nn.SiLU(),
        )
        modulation_dim = self.field_depth * self.field_hidden_dim * 2
        self.prior_head = nn.Linear(
            self.context_hidden_dim,
            modulation_dim + self.pose_dim,
        )
        self.residual_head = nn.Linear(
            self.context_hidden_dim,
            modulation_dim + self.residual_dim,
        )
        self.gate_head = nn.Linear(self.context_hidden_dim, 4)
        self.density_head = nn.Linear(self.context_hidden_dim, 1)
        self.duration_head = TextDurationHead(
            self.text_dim,
            self.context_hidden_dim,
            initial_seconds=initial_duration_seconds,
            minimum_seconds=minimum_duration_seconds,
            maximum_seconds=maximum_duration_seconds,
            dropout=dropout,
        )

        local_input_dim = self.context_hidden_dim * 3 + self.retrieval_dim
        self.local_context = nn.Sequential(
            nn.LayerNorm(local_input_dim),
            nn.Linear(local_input_dim, self.context_hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.local_head = nn.Linear(
            self.context_hidden_dim,
            modulation_dim + self.residual_dim + 1,
        )
        self._reset_output_heads()
        if self.max_local_fields == 0:
            self.density_head.requires_grad_(False)
            self.local_context.requires_grad_(False)
            self.local_head.requires_grad_(False)

    def _reset_output_heads(self):
        for layer in (self.prior_head, self.residual_head, self.local_head):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -0.5)
        nn.init.zeros_(self.density_head.weight)
        nn.init.zeros_(self.density_head.bias)

    def _split_modulation(self, values: torch.Tensor, output_dim: int):
        modulation_size = self.field_depth * self.field_hidden_dim
        scale, shift, output_bias = torch.split(
            values,
            [modulation_size, modulation_size, int(output_dim)],
            dim=-1,
        )
        target_shape = (*values.shape[:-1], self.field_depth, self.field_hidden_dim)
        return scale.reshape(target_shape), shift.reshape(target_shape), output_bias

    def predict_duration(self, text_tokens: torch.Tensor, text_mask: torch.Tensor | None = None):
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        pooled_text = masked_mean(self.text_norm(text_tokens), text_mask, dim=1)
        return self.duration_head(pooled_text)

    def _local_centers(
        self,
        density: torch.Tensor,
        context_tau: torch.Tensor,
        context_mask: torch.Tensor,
        local_mask: torch.Tensor,
    ):
        batch, frames = context_mask.shape
        local_count = local_mask.shape[1]
        if local_count == 0:
            return context_tau.new_zeros(batch, 0)
        density = density * context_mask.to(density.dtype)
        cumulative = torch.cumsum(density, dim=1) - 0.5 * density
        cumulative = cumulative / density.sum(dim=1, keepdim=True).clamp_min(1e-6)
        ranks = torch.arange(local_count, device=density.device, dtype=density.dtype)
        active_count = local_mask.sum(dim=1).clamp_min(1).to(density.dtype)
        quantiles = (ranks.unsqueeze(0) + 0.5) / active_count.unsqueeze(1)
        distance = cumulative[:, None, :] - quantiles[:, :, None]
        logits = -distance.square() / max(self.quantile_temperature, 1e-5)
        logits = logits.masked_fill(~context_mask[:, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        centers = (weights * context_tau[:, None, :]).sum(dim=-1)
        return centers * local_mask.to(centers.dtype)

    def forward(
        self,
        text_tokens: torch.Tensor,
        adapter_context: torch.Tensor,
        context_mask: torch.Tensor,
        retrieval_evidence: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> TrajectoryInstance:
        if adapter_context.ndim != 3 or adapter_context.shape[-1] != self.pose_dim:
            raise ValueError(
                f"Expected adapter context [B,T,{self.pose_dim}], got {tuple(adapter_context.shape)}"
            )
        if context_mask.shape != adapter_context.shape[:2]:
            raise ValueError("Context mask does not match adapter context")
        if retrieval_evidence.shape != (*adapter_context.shape[:2], self.retrieval_dim):
            raise ValueError(
                f"Expected retrieval evidence [B,T,{self.retrieval_dim}], "
                f"got {tuple(retrieval_evidence.shape)}"
            )
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        dtype = adapter_context.dtype
        context_mask = context_mask.bool()
        retrieval_evidence = retrieval_evidence.to(device=adapter_context.device, dtype=dtype)
        text_tokens = text_tokens.to(device=adapter_context.device, dtype=dtype)
        text_mask = text_mask.to(device=adapter_context.device).bool()
        lengths = context_mask.sum(dim=1).clamp_min(1)
        context_tau = _normalized_context_time(context_mask, dtype)
        velocity, acceleration = _temporal_difference(adapter_context, context_mask)
        frame_input = torch.cat(
            [
                adapter_context,
                velocity,
                acceleration,
                retrieval_evidence,
                context_tau.unsqueeze(-1),
            ],
            dim=-1,
        )
        frame_hidden = self.frame_input(frame_input)
        frame_hidden = frame_hidden * context_mask.unsqueeze(-1).to(dtype)
        for block in self.context_blocks:
            frame_hidden = block(frame_hidden, context_mask)

        pooled_text = masked_mean(self.text_norm(text_tokens), text_mask, dim=1)
        text_feature = self.text_projection(pooled_text)
        first, last = _first_last(frame_hidden, lengths)
        summary = torch.cat(
            [
                masked_mean(frame_hidden, context_mask, dim=1),
                masked_std(frame_hidden, context_mask, dim=1),
                first,
                last,
                text_feature,
            ],
            dim=-1,
        )
        global_context = self.global_context(summary)
        prior_scale, prior_shift, prior_output_bias = self._split_modulation(
            self.prior_head(global_context), self.pose_dim
        )
        residual_scale, residual_shift, residual_output_bias = self._split_modulation(
            self.residual_head(global_context), self.residual_dim
        )
        articulator_gates = torch.sigmoid(self.gate_head(global_context))
        log_duration, duration = self.duration_head(pooled_text)

        local_count = self.max_local_fields
        if local_count:
            density = F.softplus(self.density_head(frame_hidden).squeeze(-1)) + 0.05
            if self.retrieval_dim:
                confidence = retrieval_evidence[..., 0].clamp(0.0, 1.0)
                density = density * (1.0 + (1.0 - confidence))
            density = density * context_mask.to(dtype)
            active_count = torch.ceil(lengths.float() / self.frames_per_local_field).long()
            active_count = active_count.clamp(1, local_count)
            local_index = torch.arange(local_count, device=adapter_context.device)
            local_mask = local_index.unsqueeze(0) < active_count.unsqueeze(1)
            centers = self._local_centers(density, context_tau, context_mask, local_mask)
            base_width = (2.5 / active_count.to(dtype)).clamp(
                self.minimum_local_width, self.maximum_local_width
            )
            pool_width = base_width[:, None].expand(-1, local_count)
            distance = (context_tau[:, None, :] - centers[:, :, None]) / pool_width[:, :, None]
            pool_logits = -0.5 * distance.square()
            pool_logits = pool_logits.masked_fill(~context_mask[:, None, :], -torch.inf)
            pool_weights = torch.softmax(pool_logits, dim=-1)
            local_frame = torch.einsum("bmt,bth->bmh", pool_weights, frame_hidden)
            local_retrieval = torch.einsum(
                "bmt,btr->bmr", pool_weights, retrieval_evidence
            )
            local_text = text_feature[:, None, :].expand(-1, local_count, -1)
            local_global = global_context[:, None, :].expand(-1, local_count, -1)
            local_context = self.local_context(
                torch.cat([local_frame, local_text, local_global, local_retrieval], dim=-1)
            )
            local_values = self.local_head(local_context)
            modulation_size = self.field_depth * self.field_hidden_dim
            local_scale_flat, local_shift_flat, local_output_bias, width_logits = torch.split(
                local_values,
                [modulation_size, modulation_size, self.residual_dim, 1],
                dim=-1,
            )
            local_scale = local_scale_flat.reshape(
                adapter_context.shape[0], local_count, self.field_depth, self.field_hidden_dim
            )
            local_shift = local_shift_flat.reshape_as(local_scale)
            width_unit = torch.sigmoid(width_logits.squeeze(-1))
            local_widths = self.minimum_local_width + width_unit * (
                self.maximum_local_width - self.minimum_local_width
            )
            local_widths = 0.5 * (local_widths + pool_width)
            local_uncertainty = torch.einsum(
                "bmt,bt->bm",
                pool_weights,
                (1.0 - retrieval_evidence[..., 0].clamp(0.0, 1.0))
                if self.retrieval_dim
                else torch.zeros_like(context_tau),
            )
            mask_float = local_mask.to(dtype)
            local_scale = local_scale * mask_float[:, :, None, None]
            local_shift = local_shift * mask_float[:, :, None, None]
            local_output_bias = local_output_bias * mask_float[:, :, None]
            local_widths = local_widths * mask_float + (~local_mask).to(dtype)
            local_uncertainty = local_uncertainty * mask_float
        else:
            batch = adapter_context.shape[0]
            density = adapter_context.new_zeros(context_mask.shape)
            local_scale = adapter_context.new_zeros(
                batch, 0, self.field_depth, self.field_hidden_dim
            )
            local_shift = local_scale.clone()
            local_output_bias = adapter_context.new_zeros(batch, 0, self.residual_dim)
            centers = adapter_context.new_zeros(batch, 0)
            local_widths = adapter_context.new_zeros(batch, 0)
            local_mask = torch.zeros(batch, 0, dtype=torch.bool, device=adapter_context.device)
            local_uncertainty = adapter_context.new_zeros(batch, 0)

        return TrajectoryInstance(
            duration_seconds=duration,
            log_duration_seconds=log_duration,
            prior_scale=prior_scale,
            prior_shift=prior_shift,
            prior_output_bias=prior_output_bias,
            residual_scale=residual_scale,
            residual_shift=residual_shift,
            residual_output_bias=residual_output_bias,
            local_scale=local_scale,
            local_shift=local_shift,
            local_output_bias=local_output_bias,
            local_centers=centers,
            local_widths=local_widths,
            local_mask=local_mask,
            articulator_gates=articulator_gates,
            local_uncertainty=local_uncertainty,
            context_density=density,
            context_tau=context_tau,
        )
