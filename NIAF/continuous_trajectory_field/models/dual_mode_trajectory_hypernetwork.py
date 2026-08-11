from __future__ import annotations

import math

import torch
from torch import nn

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_sign_field.models.meta_implicit import masked_mean
from NIAF.continuous_trajectory_field.models.trajectory_hypernetwork import (
    TemporalConvBlock,
    TextDurationHead,
    _normalized_context_time,
    _temporal_difference,
)
from NIAF.continuous_trajectory_field.models.trajectory_instance import (
    TrajectoryInstance,
)


PART_COUNT = 4


def _slot_time_features(tau: torch.Tensor):
    return torch.stack(
        [
            tau,
            torch.sin(math.pi * tau),
            torch.cos(math.pi * tau),
            torch.sin(2.0 * math.pi * tau),
            torch.cos(2.0 * math.pi * tau),
        ],
        dim=-1,
    )


class TextTemporalSlotPlanner(nn.Module):
    """Build an ordered, fixed-size temporal plan from sentence tokens."""

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        slot_count: int,
        layer_count: int,
        head_count: int,
        dropout: float,
    ):
        super().__init__()
        if int(hidden_dim) % int(head_count):
            raise ValueError("temporal slot hidden_dim must be divisible by head_count")
        self.text_norm = nn.LayerNorm(int(text_dim))
        self.text_projection = nn.Linear(int(text_dim), int(hidden_dim))
        self.slot_embeddings = nn.Parameter(
            torch.zeros(int(slot_count), int(hidden_dim))
        )
        self.time_projection = nn.Sequential(
            nn.Linear(5, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=int(hidden_dim),
            nhead=int(head_count),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=max(int(layer_count), 1),
            norm=nn.LayerNorm(int(hidden_dim)),
        )
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

    @property
    def slot_count(self):
        return int(self.slot_embeddings.shape[0])

    def slot_tau(self, device, dtype):
        return torch.linspace(
            -1.0,
            1.0,
            self.slot_count,
            device=device,
            dtype=dtype,
        )

    def normalized_text(self, text_tokens):
        return self.text_norm(text_tokens)

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor):
        if text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [B,L,D]")
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_mask must match text_tokens")
        if not bool(text_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one valid text token")
        memory = self.text_projection(self.normalized_text(text_tokens))
        tau = self.slot_tau(memory.device, memory.dtype)
        slots = self.slot_embeddings.to(dtype=memory.dtype)
        slots = slots + self.time_projection(_slot_time_features(tau))
        slots = slots.unsqueeze(0).expand(memory.shape[0], -1, -1)
        planned = self.decoder(
            tgt=slots,
            memory=memory,
            memory_key_padding_mask=~text_mask.bool(),
        )
        return planned, tau


class WordPriorSlotEncoder(nn.Module):
    """Encode the optional framewise word prior onto the text slot grid."""

    def __init__(
        self,
        pose_dim: int,
        retrieval_dim: int,
        hidden_dim: int,
        layer_count: int,
        slot_count: int,
        dropout: float,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.slot_count = int(slot_count)
        input_dim = self.pose_dim * 3 + self.retrieval_dim + 1
        self.frame_input = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.context_blocks = nn.ModuleList(
            [
                TemporalConvBlock(int(hidden_dim), kernel_size=5, dropout=dropout)
                for _ in range(max(int(layer_count), 1))
            ]
        )

    def forward(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
        retrieval: torch.Tensor,
        slot_tau: torch.Tensor,
    ):
        if context.ndim != 3 or context.shape[-1] != self.pose_dim:
            raise ValueError(
                f"word_prior_context must have shape [B,T,{self.pose_dim}]"
            )
        if mask.shape != context.shape[:2]:
            raise ValueError("word_prior_mask must match word_prior_context")
        if retrieval.shape != (*context.shape[:2], self.retrieval_dim):
            raise ValueError(
                "word_prior_features must match context frames and retrieval_dim"
            )
        mask = mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every available word prior must contain a valid frame")
        tau = _normalized_context_time(mask, context.dtype)
        velocity, acceleration = _temporal_difference(context, mask)
        frame_input = torch.cat(
            [context, velocity, acceleration, retrieval, tau.unsqueeze(-1)],
            dim=-1,
        )
        hidden = self.frame_input(frame_input)
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.context_blocks:
            hidden = block(hidden, mask)

        spacing = 2.0 / max(self.slot_count - 1, 1)
        width = max(1.25 * spacing, 0.05)
        distance = (tau[:, :, None] - slot_tau[None, None, :]) / width
        logits = -0.5 * distance.square()
        logits = logits.masked_fill(~mask[:, :, None], -torch.inf)
        weights = torch.softmax(logits, dim=1)
        slot_hidden = torch.einsum("bts,bth->bsh", weights, hidden)
        slot_retrieval = torch.einsum("bts,btr->bsr", weights, retrieval)
        return slot_hidden, slot_retrieval


class DualModeTrajectoryHypernetwork(nn.Module):
    """Create trajectory parameters from text with optional word-prior guidance."""

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
        max_local_fields: int = 16,
        frames_per_local_field: int = 24,
        minimum_local_width: float = 0.05,
        maximum_local_width: float = 0.30,
        time_dependent_local_gates: bool = True,
        temporal_slot_count: int = 16,
        temporal_slot_layers: int = 2,
        temporal_slot_heads: int = 8,
        context_fps: float = 20.0,
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
        self.time_dependent_local_gates = bool(time_dependent_local_gates)
        self.context_fps = float(context_fps)

        self.text_planner = TextTemporalSlotPlanner(
            text_dim=self.text_dim,
            hidden_dim=self.context_hidden_dim,
            slot_count=int(temporal_slot_count),
            layer_count=int(temporal_slot_layers),
            head_count=int(temporal_slot_heads),
            dropout=dropout,
        )
        self.word_encoder = WordPriorSlotEncoder(
            pose_dim=self.pose_dim,
            retrieval_dim=self.retrieval_dim,
            hidden_dim=self.context_hidden_dim,
            layer_count=int(context_layers),
            slot_count=int(temporal_slot_count),
            dropout=dropout,
        )
        gate_input_dim = self.context_hidden_dim * 2 + self.retrieval_dim
        self.word_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, self.context_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.context_hidden_dim, PART_COUNT),
        )
        self.word_part_projections = nn.ModuleList(
            [
                nn.Linear(self.context_hidden_dim, self.context_hidden_dim)
                for _ in range(PART_COUNT)
            ]
        )
        self.word_fusion = nn.Linear(
            self.context_hidden_dim * PART_COUNT,
            self.context_hidden_dim,
            bias=False,
        )
        self.fusion_norm = nn.LayerNorm(self.context_hidden_dim)

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
        self.coarse_head = nn.Linear(
            self.context_hidden_dim,
            modulation_dim + self.pose_dim,
        )
        self.residual_head = nn.Linear(
            self.context_hidden_dim,
            modulation_dim + self.residual_dim,
        )
        self.gate_head = nn.Linear(self.context_hidden_dim, PART_COUNT)
        self.duration_head = TextDurationHead(
            self.text_dim,
            self.context_hidden_dim,
            initial_seconds=initial_duration_seconds,
            minimum_seconds=minimum_duration_seconds,
            maximum_seconds=maximum_duration_seconds,
            dropout=dropout,
        )

        self.local_context = nn.Sequential(
            nn.LayerNorm(self.context_hidden_dim * 3),
            nn.Linear(self.context_hidden_dim * 3, self.context_hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.local_head = nn.Linear(
            self.context_hidden_dim,
            modulation_dim + self.residual_dim + 1,
        )
        self.local_gate_head = (
            nn.Linear(self.context_hidden_dim, PART_COUNT)
            if self.time_dependent_local_gates
            else None
        )
        self._reset_output_heads()

    @property
    def temporal_slot_count(self):
        return self.text_planner.slot_count

    def _reset_output_heads(self):
        for layer in (self.coarse_head, self.residual_head):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -0.5)
        nn.init.zeros_(self.word_gate[-1].weight)
        nn.init.zeros_(self.word_gate[-1].bias)
        self.reset_local_branch()

    def reset_local_branch(self):
        for module in self.local_context.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        nn.init.normal_(self.local_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.local_head.bias)
        if self.local_gate_head is not None:
            nn.init.zeros_(self.local_gate_head.weight)
            nn.init.zeros_(self.local_gate_head.bias)

    def _pooled_text(self, text_tokens, text_mask):
        normalized = self.text_planner.normalized_text(text_tokens)
        return masked_mean(normalized, text_mask, dim=1)

    def predict_duration(self, text_tokens, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones(
                text_tokens.shape[:2],
                dtype=torch.bool,
                device=text_tokens.device,
            )
        pooled = self._pooled_text(text_tokens, text_mask.bool())
        return self.duration_head(pooled)

    def _split_modulation(self, values: torch.Tensor, output_dim: int):
        modulation_size = self.field_depth * self.field_hidden_dim
        scale, shift, output_bias = torch.split(
            values,
            [modulation_size, modulation_size, int(output_dim)],
            dim=-1,
        )
        target_shape = (*values.shape[:-1], self.field_depth, self.field_hidden_dim)
        return scale.reshape(target_shape), shift.reshape(target_shape), output_bias

    @staticmethod
    def _uniform_centers(active_count, local_count, dtype):
        ranks = torch.arange(
            local_count,
            device=active_count.device,
            dtype=dtype,
        ).unsqueeze(0)
        active = active_count.to(dtype).unsqueeze(1)
        denominator = (active - 1.0).clamp_min(1.0)
        centers = -1.0 + 2.0 * ranks / denominator
        centers = torch.where(active == 1.0, torch.zeros_like(centers), centers)
        mask = ranks < active
        return centers * mask.to(dtype), mask

    @staticmethod
    def _interpolate_slots(values: torch.Tensor, centers: torch.Tensor):
        slot_count = values.shape[1]
        position = (centers.clamp(-1.0, 1.0) + 1.0) * 0.5 * max(slot_count - 1, 1)
        left = position.floor().long().clamp(0, slot_count - 1)
        right = (left + 1).clamp(0, slot_count - 1)
        fraction = (position - left.to(position.dtype)).unsqueeze(-1)
        gather_shape = (*left.shape, values.shape[-1])
        left_values = torch.gather(
            values,
            1,
            left.unsqueeze(-1).expand(gather_shape),
        )
        right_values = torch.gather(
            values,
            1,
            right.unsqueeze(-1).expand(gather_shape),
        )
        return left_values + fraction * (right_values - left_values)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        word_prior_context: torch.Tensor | None = None,
        word_prior_mask: torch.Tensor | None = None,
        word_prior_features: torch.Tensor | None = None,
        word_prior_available: torch.Tensor | None = None,
    ) -> TrajectoryInstance:
        if text_mask is None:
            text_mask = torch.ones(
                text_tokens.shape[:2],
                dtype=torch.bool,
                device=text_tokens.device,
            )
        text_mask = text_mask.bool()
        text_slots, slot_tau = self.text_planner(text_tokens, text_mask)
        dtype = text_slots.dtype
        device = text_slots.device
        batch = text_slots.shape[0]
        pooled_text = self._pooled_text(text_tokens, text_mask)
        log_duration, duration = self.duration_head(pooled_text)

        supplied = (
            word_prior_context is not None,
            word_prior_mask is not None,
            word_prior_features is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "word_prior_context, word_prior_mask, and word_prior_features "
                "must be supplied together"
            )
        if word_prior_available is None:
            word_prior_available = torch.full(
                (batch,),
                bool(all(supplied)),
                dtype=torch.bool,
                device=device,
            )
        else:
            word_prior_available = word_prior_available.to(device=device).bool()
            if word_prior_available.shape != (batch,):
                raise ValueError("word_prior_available must have shape [B]")
        if bool(word_prior_available.any()) and not all(supplied):
            raise ValueError("Available word-prior samples require word-prior tensors")

        if all(supplied):
            context = word_prior_context.to(device=device, dtype=dtype)
            context_mask = word_prior_mask.to(device=device).bool()
            features = word_prior_features.to(device=device, dtype=dtype)
            word_slots, slot_features = self.word_encoder(
                context,
                context_mask,
                features,
                slot_tau,
            )
        else:
            word_slots = text_slots.new_zeros(text_slots.shape)
            slot_features = text_slots.new_zeros(
                batch,
                self.temporal_slot_count,
                self.retrieval_dim,
            )

        gate_input = torch.cat([text_slots, word_slots, slot_features], dim=-1)
        word_gates = torch.sigmoid(self.word_gate(gate_input))
        availability_float = word_prior_available.to(dtype)
        word_gates = word_gates * availability_float[:, None, None]
        part_deltas = [
            projection(word_slots) * word_gates[..., index : index + 1]
            for index, projection in enumerate(self.word_part_projections)
        ]
        word_delta = self.word_fusion(torch.cat(part_deltas, dim=-1))
        word_delta = word_delta * availability_float[:, None, None]
        fused_slots = self.fusion_norm(text_slots + word_delta)

        summary = torch.cat(
            [
                fused_slots.mean(dim=1),
                fused_slots.std(dim=1, unbiased=False),
                fused_slots[:, 0],
                fused_slots[:, -1],
                self.text_planner.text_projection(pooled_text),
            ],
            dim=-1,
        )
        global_context = self.global_context(summary)
        prior_scale, prior_shift, prior_output_bias = self._split_modulation(
            self.coarse_head(global_context),
            self.pose_dim,
        )
        residual_scale, residual_shift, residual_output_bias = self._split_modulation(
            self.residual_head(global_context),
            self.residual_dim,
        )
        articulator_gates = torch.sigmoid(self.gate_head(global_context))

        local_count = self.max_local_fields
        if local_count:
            predicted_frames = duration.detach() * self.context_fps
            active_count = torch.ceil(
                predicted_frames / self.frames_per_local_field
            ).long()
            active_count = active_count.clamp(1, local_count)
            centers, local_mask = self._uniform_centers(
                active_count,
                local_count,
                dtype,
            )
            local_slot = self._interpolate_slots(fused_slots, centers)
            local_text = self.text_planner.text_projection(pooled_text)
            local_text = local_text[:, None, :].expand(-1, local_count, -1)
            local_global = global_context[:, None, :].expand(-1, local_count, -1)
            local_context = self.local_context(
                torch.cat([local_slot, local_text, local_global], dim=-1)
            )
            local_values = self.local_head(local_context)
            local_part_gates = (
                torch.sigmoid(self.local_gate_head(local_context))
                if self.local_gate_head is not None
                else None
            )
            modulation_size = self.field_depth * self.field_hidden_dim
            local_scale_flat, local_shift_flat, local_output_bias, width_logits = (
                torch.split(
                    local_values,
                    [modulation_size, modulation_size, self.residual_dim, 1],
                    dim=-1,
                )
            )
            local_scale = local_scale_flat.reshape(
                batch,
                local_count,
                self.field_depth,
                self.field_hidden_dim,
            )
            local_shift = local_shift_flat.reshape_as(local_scale)
            width_unit = torch.sigmoid(width_logits.squeeze(-1))
            learned_width = self.minimum_local_width + width_unit * (
                self.maximum_local_width - self.minimum_local_width
            )
            base_width = (2.5 / active_count.to(dtype)).clamp(
                self.minimum_local_width,
                self.maximum_local_width,
            )
            local_widths = 0.5 * (learned_width + base_width[:, None])
            local_word_gates = self._interpolate_slots(word_gates, centers)
            local_uncertainty = 1.0 - local_word_gates.mean(dim=-1)
            mask_float = local_mask.to(dtype)
            local_scale = local_scale * mask_float[:, :, None, None]
            local_shift = local_shift * mask_float[:, :, None, None]
            local_output_bias = local_output_bias * mask_float[:, :, None]
            local_widths = local_widths * mask_float + (~local_mask).to(dtype)
            local_uncertainty = local_uncertainty * mask_float
            if local_part_gates is not None:
                local_part_gates = local_part_gates * mask_float[:, :, None]
        else:
            local_scale = fused_slots.new_zeros(
                batch,
                0,
                self.field_depth,
                self.field_hidden_dim,
            )
            local_shift = local_scale.clone()
            local_output_bias = fused_slots.new_zeros(batch, 0, self.residual_dim)
            centers = fused_slots.new_zeros(batch, 0)
            local_widths = fused_slots.new_zeros(batch, 0)
            local_mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            local_uncertainty = fused_slots.new_zeros(batch, 0)
            local_part_gates = (
                fused_slots.new_zeros(batch, 0, PART_COUNT)
                if self.local_gate_head is not None
                else None
            )

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
            context_density=fused_slots.norm(dim=-1),
            context_tau=slot_tau[None, :].expand(batch, -1),
            local_part_gates=local_part_gates,
            word_prior_available=word_prior_available,
            word_prior_gates=word_gates,
            temporal_slot_tau=slot_tau[None, :].expand(batch, -1),
        )
