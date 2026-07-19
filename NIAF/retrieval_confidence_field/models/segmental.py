"""Hierarchical segment-by-segment implicit residual field."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_sign_field.meta_learning import (
    masked_residual_loss,
    temporal_difference,
    temporal_mask,
)
from NIAF.continuous_sign_field.models.meta_implicit import (
    fourier_encode_scalar,
    masked_mean,
    temporal_differences,
)
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import (
    ARTICULATOR_NAMES,
)
from NIAF.retrieval_confidence_field.models.uncertainty_adaptive import (
    DEFAULT_ARTICULATOR_STRIDES,
    RetrievalUncertaintyAdaptiveKnotField,
    adaptive_time_coordinate,
)


__all__ = [
    "HierarchicalSegmentalArticulatorCodePredictor",
    "RetrievalUncertaintySegmentalField",
    "boundary_temporal_matching_loss",
    "cubic_interpolate_segment_codes",
    "segment_boundary_mask",
]


def _gather_frames(values, indices):
    feature_dim = values.shape[-1]
    return torch.gather(
        values,
        1,
        indices.unsqueeze(-1).expand(-1, -1, feature_dim),
    )


def _segment_context_pool(values, mask, density, stride, window_size):
    """Pool overlapping forward windows and retain their boundary states."""

    stride = max(int(stride), 1)
    window_size = max(int(window_size), stride)
    batch, frames, _dim = values.shape
    lengths = mask.sum(dim=1).clamp_min(1)
    counts = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
    max_segments = max(int(counts.max().item()), 1)

    segment_index = torch.arange(max_segments, device=values.device).view(1, -1)
    starts = segment_index * stride
    segment_mask = segment_index < counts.unsqueeze(1)
    starts = torch.minimum(starts.expand(batch, -1), (lengths - 1).unsqueeze(1))
    ends = torch.minimum(starts + window_size, lengths.unsqueeze(1))

    frame_index = torch.arange(frames, device=values.device).view(1, 1, -1)
    support = (
        (frame_index >= starts.unsqueeze(-1))
        & (frame_index < ends.unsqueeze(-1))
        & mask.unsqueeze(1)
        & segment_mask.unsqueeze(-1)
    )
    center = 0.5 * (starts + ends - 1).to(values.dtype)
    radius = (0.5 * (ends - starts).to(values.dtype)).clamp_min(1.0)
    distance = (frame_index.to(values.dtype) - center.unsqueeze(-1)) / radius.unsqueeze(
        -1
    )
    weights = torch.exp(-0.5 * distance.square())
    weights = weights * density.clamp_min(0.05).unsqueeze(1)
    weights = weights * support.to(values.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pooled = torch.einsum("bkt,bth->bkh", weights, values)

    start_features = _gather_frames(values, starts)
    end_features = _gather_frames(values, (ends - 1).clamp_min(0))
    context = torch.cat(
        [pooled, start_features, end_features, end_features - start_features],
        dim=-1,
    )
    denominator = (lengths - 1).clamp_min(1).to(values.dtype).unsqueeze(1)
    positions = starts.to(values.dtype) / denominator
    positions = positions * segment_mask.to(values.dtype)
    return context, segment_mask, counts, positions


def _local_text_context(text, text_mask, positions, radius):
    """Soft monotonic text alignment centered at each segment position."""

    text_mask = text_mask.bool()
    batch, tokens, _dim = text.shape
    token_index = torch.arange(tokens, device=text.device).view(1, 1, -1)
    token_counts = text_mask.sum(dim=1).clamp_min(1)
    denominator = (token_counts - 1).clamp_min(1).view(batch, 1, 1).to(text.dtype)
    token_positions = token_index.to(text.dtype) / denominator
    distance = positions.unsqueeze(-1) - token_positions
    radius = max(float(radius), 1e-3)
    logits = -0.5 * (distance / radius).square()
    logits = logits.masked_fill(
        ~text_mask.unsqueeze(1),
        -torch.finfo(logits.dtype).max,
    )
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bkl,blh->bkh", weights, text)


def _gather_codes(codes, indices):
    code_dim = codes.shape[-1]
    return torch.gather(
        codes,
        1,
        indices.unsqueeze(-1).expand(-1, -1, code_dim),
    )


def cubic_interpolate_segment_codes(segment_codes, segment_mask, coordinate):
    """Catmull-Rom interpolation with linear fallback for two-code sequences."""

    if coordinate.ndim == 3:
        coordinate = coordinate.squeeze(-1)
    counts = segment_mask.sum(dim=1).clamp_min(1)
    scaled = coordinate.clamp(0.0, 1.0) * (counts - 1).to(coordinate.dtype).unsqueeze(1)
    index_1 = torch.floor(scaled).long()
    max_index = (counts - 1).unsqueeze(1)
    index_2 = torch.minimum(index_1 + 1, max_index)
    index_0 = torch.maximum(index_1 - 1, torch.zeros_like(index_1))
    index_3 = torch.minimum(index_2 + 1, max_index)
    fraction = (scaled - index_1.to(scaled.dtype)).unsqueeze(-1)

    point_0 = _gather_codes(segment_codes, index_0)
    point_1 = _gather_codes(segment_codes, index_1)
    point_2 = _gather_codes(segment_codes, index_2)
    point_3 = _gather_codes(segment_codes, index_3)
    fraction_2 = fraction.square()
    fraction_3 = fraction_2 * fraction
    cubic = 0.5 * (
        2.0 * point_1
        + (-point_0 + point_2) * fraction
        + (2.0 * point_0 - 5.0 * point_1 + 4.0 * point_2 - point_3) * fraction_2
        + (-point_0 + 3.0 * point_1 - 3.0 * point_2 + point_3) * fraction_3
    )
    linear = point_1 + fraction * (point_2 - point_1)
    use_linear = (counts <= 2).view(-1, 1, 1)
    return torch.where(use_linear, linear, cubic)


def segment_boundary_mask(mask, stride):
    """Mark valid rollout boundaries, including both sequence endpoints."""

    stride = max(int(stride), 1)
    mask = mask.bool()
    frames = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
    boundaries = frames.remainder(stride) == 0
    lengths = mask.sum(dim=1).clamp_min(1)
    last = frames == (lengths - 1).unsqueeze(1)
    return mask & (boundaries | last)


def boundary_temporal_matching_loss(
    prediction,
    target,
    mask,
    boundaries,
    order=0,
    hand_weight=5.0,
):
    """Match pose or temporal derivatives in windows that cross a segment boundary."""

    order = max(int(order), 0)
    pred_values = temporal_difference(prediction, order=order)
    target_values = temporal_difference(target, order=order)
    valid = temporal_mask(mask, order=order)
    selected = boundaries.bool()
    for _ in range(order):
        if selected.shape[1] < 2:
            selected = selected.new_zeros(selected.shape[0], 0)
            break
        selected = selected[:, 1:] | selected[:, :-1]
    selected = selected & valid
    if pred_values.shape[1] == 0:
        return prediction.new_tensor(0.0)
    return masked_residual_loss(
        pred_values,
        target_values,
        selected,
        hand_weight=float(hand_weight),
        loss_type="l1",
    )


class CausalSegmentRollout(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.num_layers = max(int(num_layers), 1)
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.initial_state = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context, local_text, global_text, segment_mask, embedding):
        local = self.context_proj(context)
        fused = self.fuse(torch.cat([local, local_text], dim=-1))
        fused = fused + embedding.view(1, 1, -1)
        fused = fused * segment_mask.unsqueeze(-1).to(fused.dtype)
        initial = self.initial_state(global_text + embedding.view(1, -1))
        initial = initial.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        output, _hidden = self.gru(fused, initial)
        return self.out_norm(output) * segment_mask.unsqueeze(-1).to(output.dtype)


class HierarchicalSegmentalArticulatorCodePredictor(nn.Module):
    """Predict local trajectory codes sequentially at several temporal scales."""

    def __init__(
        self,
        pose_dim=COMPACT6D_DIM,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        code_dim=64,
        hidden_dim=256,
        articulator_strides=None,
        rollout_layers=2,
        time_fourier_bands=6,
        segment_window_multiplier=2.0,
        minimum_segment_frames=8,
        maximum_segment_frames=32,
        text_window_radius=0.2,
        boundary_stride=8,
        dropout=0.0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_fourier_bands = int(time_fourier_bands)
        self.segment_window_multiplier = float(segment_window_multiplier)
        self.minimum_segment_frames = max(int(minimum_segment_frames), 1)
        self.maximum_segment_frames = max(
            int(maximum_segment_frames), self.minimum_segment_frames
        )
        self.text_window_radius = float(text_window_radius)
        self.boundary_stride = max(int(boundary_stride), 1)
        configured = articulator_strides or DEFAULT_ARTICULATOR_STRIDES
        self.articulator_strides = {
            name: tuple(max(int(value), 1) for value in configured[name])
            for name in ARTICULATOR_NAMES
        }
        scale_counts = {len(value) for value in self.articulator_strides.values()}
        if len(scale_counts) != 1:
            raise ValueError("Every articulator must define the same number of scales.")
        self.num_scales = scale_counts.pop()

        time_dim = 1 + 2 * self.time_fourier_bands
        frame_dim = self.pose_dim * 3 + int(retrieval_dim) + time_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.local_mixer = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
        )
        self.text_proj = nn.Linear(int(text_dim), self.hidden_dim)
        self.rollout = CausalSegmentRollout(
            self.hidden_dim,
            num_layers=int(rollout_layers),
            dropout=float(dropout),
        )
        self.group_scale_embedding = nn.Parameter(
            torch.zeros(len(ARTICULATOR_NAMES), self.num_scales, self.hidden_dim)
        )
        nn.init.normal_(self.group_scale_embedding, std=0.02)
        self.code_heads = nn.ModuleDict(
            {
                f"{name}_{scale_index}": nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.code_dim),
                )
                for name in ARTICULATOR_NAMES
                for scale_index in range(self.num_scales)
            }
        )

    def _window_size(self, stride):
        requested = int(math.ceil(float(stride) * self.segment_window_multiplier))
        return min(
            max(requested, self.minimum_segment_frames),
            self.maximum_segment_frames,
        )

    def forward(
        self,
        scaffold,
        mask,
        tau,
        retrieval_features,
        text_tokens,
        text_mask,
        knot_density,
        velocity=None,
        acceleration=None,
    ):
        if velocity is None or acceleration is None:
            velocity, acceleration = temporal_differences(scaffold, mask)
        frame_features = torch.cat(
            [
                scaffold,
                velocity,
                acceleration,
                retrieval_features,
                fourier_encode_scalar(tau, self.time_fourier_bands),
            ],
            dim=-1,
        )
        frame_hidden = self.frame_proj(frame_features)
        mixed = self.local_mixer(frame_hidden.transpose(1, 2)).transpose(1, 2)
        frame_hidden = (frame_hidden + mixed) * mask.unsqueeze(-1).to(
            frame_hidden.dtype
        )
        text = self.text_proj(text_tokens.to(dtype=scaffold.dtype))
        global_text = masked_mean(text, text_mask, dim=1)

        group_coordinates = {
            name: adaptive_time_coordinate(knot_density[:, :, index], mask)
            for index, name in enumerate(ARTICULATOR_NAMES)
        }
        adaptive_coordinates = torch.stack(
            [group_coordinates[name] for name in ARTICULATOR_NAMES],
            dim=-1,
        )
        scale_codes = []
        scale_masks = []
        segment_positions = []
        frame_codes_by_scale = []
        knot_counts = scaffold.new_zeros(
            scaffold.shape[0],
            len(ARTICULATOR_NAMES),
            self.num_scales,
            dtype=torch.long,
        )

        for scale_index in range(self.num_scales):
            group_codes = []
            group_masks = []
            group_positions = []
            group_frame_codes = []
            for group_index, name in enumerate(ARTICULATOR_NAMES):
                stride = self.articulator_strides[name][scale_index]
                context, segment_mask, counts, positions = _segment_context_pool(
                    frame_hidden,
                    mask,
                    knot_density[:, :, group_index],
                    stride=stride,
                    window_size=self._window_size(stride),
                )
                local_text = _local_text_context(
                    text,
                    text_mask,
                    positions,
                    radius=self.text_window_radius,
                )
                states = self.rollout(
                    context,
                    local_text,
                    global_text,
                    segment_mask,
                    self.group_scale_embedding[group_index, scale_index],
                )
                codes = self.code_heads[f"{name}_{scale_index}"](states)
                codes = codes * segment_mask.unsqueeze(-1).to(codes.dtype)
                frame_codes = cubic_interpolate_segment_codes(
                    codes,
                    segment_mask,
                    group_coordinates[name],
                )
                frame_codes = frame_codes * mask.unsqueeze(-1).to(frame_codes.dtype)
                group_codes.append(codes)
                group_masks.append(segment_mask)
                group_positions.append(positions)
                group_frame_codes.append(frame_codes)
                knot_counts[:, group_index, scale_index] = counts

            max_segments = max(value.shape[1] for value in group_codes)
            padded_codes = [
                F.pad(value, (0, 0, 0, max_segments - value.shape[1]))
                for value in group_codes
            ]
            padded_masks = [
                F.pad(value, (0, max_segments - value.shape[1]), value=False)
                for value in group_masks
            ]
            padded_positions = [
                F.pad(value, (0, max_segments - value.shape[1]))
                for value in group_positions
            ]
            scale_codes.append(torch.stack(padded_codes, dim=2))
            scale_masks.append(torch.stack(padded_masks, dim=2))
            segment_positions.append(torch.stack(padded_positions, dim=2))
            frame_codes_by_scale.append(torch.stack(group_frame_codes, dim=2))

        return {
            "scale_codes": scale_codes,
            "scale_code_masks": scale_masks,
            "frame_codes_by_scale": torch.stack(frame_codes_by_scale, dim=3),
            "adaptive_coordinates": adaptive_coordinates,
            "knot_counts": knot_counts,
            "segment_positions": segment_positions,
            "segment_boundary_mask": segment_boundary_mask(mask, self.boundary_stride),
            "velocity": velocity,
            "acceleration": acceleration,
        }


class RetrievalUncertaintySegmentalField(RetrievalUncertaintyAdaptiveKnotField):
    """Uncertainty-adaptive field whose local codes are rolled out segmentally."""

    def __init__(
        self,
        segment_rollout_layers=2,
        segment_window_multiplier=2.0,
        minimum_segment_frames=8,
        maximum_segment_frames=32,
        segment_text_window_radius=0.2,
        segment_boundary_stride=8,
        **kwargs,
    ):
        kwargs["initialize_code_predictor"] = False
        super().__init__(**kwargs)
        self.segment_boundary_stride = max(int(segment_boundary_stride), 1)
        self.code_predictor = HierarchicalSegmentalArticulatorCodePredictor(
            pose_dim=self.pose_dim,
            text_dim=int(kwargs.get("text_dim", 768)),
            retrieval_dim=self.retrieval_dim,
            code_dim=self.code_dim,
            hidden_dim=int(kwargs.get("context_hidden_dim", 256)),
            articulator_strides=self.articulator_code_strides,
            rollout_layers=int(segment_rollout_layers),
            time_fourier_bands=int(kwargs.get("context_time_fourier_bands", 6)),
            segment_window_multiplier=float(segment_window_multiplier),
            minimum_segment_frames=int(minimum_segment_frames),
            maximum_segment_frames=int(maximum_segment_frames),
            text_window_radius=float(segment_text_window_radius),
            boundary_stride=self.segment_boundary_stride,
            dropout=float(kwargs.get("dropout", 0.0)),
        )
