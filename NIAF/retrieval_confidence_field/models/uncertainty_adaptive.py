"""Uncertainty-adaptive implicit sign field with articulator-specific knots."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import (
    COMPACT6D_DIM,
    COMPACT6D_EXPRESSION,
    COMPACT6D_JAW,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from NIAF.continuous_sign_field.models.local_implicit import (
    TextDurationHead,
    interpolate_local_codes,
)
from NIAF.continuous_sign_field.models.meta_implicit import (
    SineLayer,
    fourier_encode_scalar,
    masked_mean,
    temporal_differences,
)
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import (
    ARTICULATOR_NAMES,
    ROTATION_GROUPS,
)


CORRECTION_GROUP_SLICES = {
    "body": slice(0, 30),
    "lhand": slice(30, 75),
    "rhand": slice(75, 120),
    "face": slice(120, 133),
}

DEFAULT_ARTICULATOR_STRIDES = {
    "body": (4, 8, 16),
    "lhand": (2, 4, 8),
    "rhand": (2, 4, 8),
    "face": (4, 8, 16),
}

__all__ = [
    "AdaptiveKnotArticulatorCodePredictor",
    "ArticulatorTrustNeedCalibrator",
    "DEFAULT_ARTICULATOR_STRIDES",
    "RetrievalUncertaintyAdaptiveKnotField",
    "adaptive_knot_density_target",
    "adaptive_time_coordinate",
    "correction_need_target_from_error",
    "retrieval_uncertainty_proxy",
]


def _group_features(values, name):
    if name == "face":
        return torch.cat(
            [values[..., COMPACT6D_JAW], values[..., COMPACT6D_EXPRESSION]],
            dim=-1,
        )
    return values[..., ROTATION_GROUPS[name][0]]


def _group_dynamics(velocity, acceleration):
    summaries = []
    for values in (velocity, acceleration):
        for name in ARTICULATOR_NAMES:
            group = _group_features(values, name)
            summaries.append(torch.sqrt(group.square().mean(dim=-1).clamp_min(1e-8)))
    return torch.stack(summaries, dim=-1)


def _logit(value, eps=1e-4):
    value = value.clamp(float(eps), 1.0 - float(eps))
    return torch.log(value) - torch.log1p(-value)


def retrieval_uncertainty_proxy(retrieval_features):
    """Summarize ambiguity from the cache-compatible seven retrieval features."""

    if retrieval_features.shape[-1] != len(RETRIEVAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(RETRIEVAL_FEATURE_NAMES)} retrieval features, "
            f"got {retrieval_features.shape[-1]}."
        )
    indices = {name: RETRIEVAL_FEATURE_NAMES.index(name) for name in RETRIEVAL_FEATURE_NAMES}
    confidence = retrieval_features[..., indices["retrieval_confidence"]]
    lexical_max = retrieval_features[..., indices["lexical_max_attention"]]
    lexical_mass = retrieval_features[..., indices["lexical_attention_mass"]]
    null_mass = retrieval_features[..., indices["null_attention_mass"]]
    attention_change = retrieval_features[..., indices["attention_change"]]
    coverage = retrieval_features[..., indices["lexical_coverage"]]
    attention_dispersion = (lexical_mass - lexical_max).clamp(0.0, 1.0)
    values = torch.stack(
        [
            1.0 - confidence,
            attention_dispersion,
            null_mass,
            attention_change,
            1.0 - coverage,
        ],
        dim=-1,
    )
    return values.mean(dim=-1).clamp(0.0, 1.0)


def adaptive_time_coordinate(density, mask):
    """Map frame time to a cumulative density coordinate in [0, 1]."""

    if density.shape != mask.shape:
        raise ValueError(f"density={tuple(density.shape)} mask={tuple(mask.shape)}")
    mask = mask.bool()
    mass = density.clamp_min(1e-4) * mask.to(density.dtype)
    starts = torch.cumsum(mass, dim=1) - mass
    lengths = mask.sum(dim=1).clamp_min(1)
    last_index = (lengths - 1).unsqueeze(1)
    denominator = torch.gather(starts, 1, last_index).clamp_min(1e-6)
    coordinate = starts / denominator
    coordinate = torch.where(lengths.unsqueeze(1) > 1, coordinate, torch.zeros_like(coordinate))
    return coordinate.clamp(0.0, 1.0) * mask.to(density.dtype)


def _adaptive_knot_pool(frame_codes, coordinate, density, mask, stride, kernel_width):
    batch, _frames, code_dim = frame_codes.shape
    lengths = mask.sum(dim=1).clamp_min(1)
    knot_counts = torch.div(lengths + int(stride) - 1, int(stride), rounding_mode="floor")
    knot_counts = torch.where(lengths > 1, knot_counts.clamp_min(2), torch.ones_like(knot_counts))
    max_knots = max(int(knot_counts.max().item()), 1)
    knot_index = torch.arange(max_knots, device=frame_codes.device).view(1, -1)
    knot_mask = knot_index < knot_counts.unsqueeze(1)
    denominator = (knot_counts - 1).clamp_min(1).to(frame_codes.dtype).unsqueeze(1)
    knot_coordinate = knot_index.to(frame_codes.dtype) / denominator
    bandwidth = (float(kernel_width) / denominator).clamp_min(1e-3)

    distance = coordinate.unsqueeze(1) - knot_coordinate.unsqueeze(-1)
    logits = -0.5 * (distance / bandwidth.unsqueeze(-1)).square()
    logits = logits + density.clamp_min(1e-4).log().unsqueeze(1)
    logits = logits.masked_fill(~mask.unsqueeze(1), -torch.finfo(logits.dtype).max)
    weights = torch.softmax(logits, dim=-1)
    knots = torch.einsum("bkt,btc->bkc", weights, frame_codes)
    knots = knots * knot_mask.unsqueeze(-1).to(knots.dtype)
    interpolated = interpolate_local_codes(knots, knot_mask, coordinate)
    interpolated = interpolated * mask.unsqueeze(-1).to(interpolated.dtype)
    if knots.shape != (batch, max_knots, code_dim):
        raise RuntimeError("Unexpected adaptive knot shape.")
    return knots, knot_mask, knot_counts, interpolated


class ArticulatorTrustNeedCalibrator(nn.Module):
    def __init__(
        self,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        hidden_dim=128,
        text_hidden_dim=64,
        num_scales=3,
        time_fourier_bands=4,
        initial_trust=0.5,
        initial_need=0.5,
        initial_density=0.5,
        evidence_density_bias=1.0,
        density_floor=0.05,
        dropout=0.0,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.time_fourier_bands = int(time_fourier_bands)
        self.evidence_density_bias = float(evidence_density_bias)
        self.density_floor = float(density_floor)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(int(text_dim)),
            nn.Linear(int(text_dim), int(text_hidden_dim)),
            nn.SiLU(),
        )
        time_dim = 1 + 2 * self.time_fourier_bands
        input_dim = int(retrieval_dim) + 8 + int(text_hidden_dim) + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        group_count = len(ARTICULATOR_NAMES)
        self.trust_head = nn.Linear(int(hidden_dim), group_count)
        self.need_head = nn.Linear(int(hidden_dim), group_count)
        self.density_head = nn.Linear(int(hidden_dim), group_count)
        self.scale_head = nn.Linear(int(hidden_dim), group_count * self.num_scales)
        for head, initial in (
            (self.trust_head, initial_trust),
            (self.need_head, initial_need),
            (self.density_head, initial_density),
        ):
            nn.init.zeros_(head.weight)
            initial = min(max(float(initial), 1e-4), 1.0 - 1e-4)
            nn.init.constant_(head.bias, math.log(initial / (1.0 - initial)))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, tau, retrieval_features, velocity, acceleration, text_tokens, text_mask):
        pooled_text = masked_mean(text_tokens, text_mask, dim=1)
        text_context = self.text_proj(pooled_text).unsqueeze(1).expand(
            -1, retrieval_features.shape[1], -1
        )
        features = torch.cat(
            [
                retrieval_features,
                _group_dynamics(velocity, acceleration),
                text_context.to(dtype=retrieval_features.dtype),
                fourier_encode_scalar(tau, self.time_fourier_bands),
            ],
            dim=-1,
        )
        hidden = self.net(features)
        trust_logits = self.trust_head(hidden)
        need_logits = self.need_head(hidden)
        uncertainty = retrieval_uncertainty_proxy(retrieval_features)
        density_logits = self.density_head(hidden) + self.evidence_density_bias * _logit(
            uncertainty
        ).unsqueeze(-1)
        scale_logits = self.scale_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(ARTICULATOR_NAMES),
            self.num_scales,
        )
        trust = torch.sigmoid(trust_logits)
        correction_need = torch.sigmoid(need_logits)
        density = self.density_floor + (1.0 - self.density_floor) * torch.sigmoid(
            density_logits
        )
        return {
            "trust": trust,
            "trust_logits": trust_logits,
            "correction_need": correction_need,
            "correction_need_logits": need_logits,
            "knot_density": density,
            "knot_density_logits": density_logits,
            "scale_logits": scale_logits,
            "retrieval_uncertainty": uncertainty,
        }


class AdaptiveKnotArticulatorCodePredictor(nn.Module):
    def __init__(
        self,
        pose_dim=COMPACT6D_DIM,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        code_dim=64,
        hidden_dim=256,
        articulator_strides=None,
        num_layers=1,
        num_heads=8,
        time_fourier_bands=6,
        knot_kernel_width=0.75,
        dropout=0.0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.time_fourier_bands = int(time_fourier_bands)
        self.knot_kernel_width = float(knot_kernel_width)
        configured = articulator_strides or DEFAULT_ARTICULATOR_STRIDES
        self.articulator_strides = {
            name: tuple(max(int(value), 1) for value in configured[name])
            for name in ARTICULATOR_NAMES
        }
        scale_counts = {len(values) for values in self.articulator_strides.values()}
        if len(scale_counts) != 1:
            raise ValueError("Every articulator must define the same number of knot scales.")
        self.num_scales = scale_counts.pop()

        time_dim = 1 + 2 * self.time_fourier_bands
        frame_dim = self.pose_dim * 3 + int(retrieval_dim) + time_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.text_proj = nn.Linear(int(text_dim), int(hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=max(int(num_layers), 1)
        )
        self.code_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), self.code_dim),
                )
                for name in ARTICULATOR_NAMES
            }
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
        hidden = self.frame_proj(frame_features)
        text = self.text_proj(text_tokens.to(dtype=scaffold.dtype))
        attended, _ = self.cross_attention(
            hidden,
            text,
            text,
            key_padding_mask=~text_mask.bool(),
            need_weights=False,
        )
        encoded = self.temporal_encoder(
            hidden + attended,
            src_key_padding_mask=~mask.bool(),
        )
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)

        frame_codes_by_scale = []
        scale_codes = []
        scale_masks = []
        coordinates = []
        knot_counts = scaffold.new_zeros(
            scaffold.shape[0],
            len(ARTICULATOR_NAMES),
            self.num_scales,
            dtype=torch.long,
        )
        group_frame_codes = {
            name: self.code_heads[name](encoded) * mask.unsqueeze(-1).to(encoded.dtype)
            for name in ARTICULATOR_NAMES
        }
        group_coordinates = {
            name: adaptive_time_coordinate(knot_density[:, :, index], mask)
            for index, name in enumerate(ARTICULATOR_NAMES)
        }
        coordinates = torch.stack(
            [group_coordinates[name] for name in ARTICULATOR_NAMES], dim=-1
        )

        for scale_index in range(self.num_scales):
            group_knots = []
            group_masks = []
            group_interpolated = []
            for group_index, name in enumerate(ARTICULATOR_NAMES):
                knots, knot_mask, counts, interpolated = _adaptive_knot_pool(
                    group_frame_codes[name],
                    group_coordinates[name],
                    knot_density[:, :, group_index],
                    mask,
                    stride=self.articulator_strides[name][scale_index],
                    kernel_width=self.knot_kernel_width,
                )
                group_knots.append(knots)
                group_masks.append(knot_mask)
                group_interpolated.append(interpolated)
                knot_counts[:, group_index, scale_index] = counts
            max_knots = max(value.shape[1] for value in group_knots)
            padded_knots = [
                F.pad(value, (0, 0, 0, max_knots - value.shape[1]))
                for value in group_knots
            ]
            padded_masks = [
                F.pad(value, (0, max_knots - value.shape[1]), value=False)
                for value in group_masks
            ]
            scale_codes.append(torch.stack(padded_knots, dim=2))
            scale_masks.append(torch.stack(padded_masks, dim=2))
            frame_codes_by_scale.append(torch.stack(group_interpolated, dim=2))

        return {
            "scale_codes": scale_codes,
            "scale_code_masks": scale_masks,
            "frame_codes_by_scale": torch.stack(frame_codes_by_scale, dim=3),
            "adaptive_coordinates": coordinates,
            "knot_counts": knot_counts,
            "velocity": velocity,
            "acceleration": acceleration,
        }


class UncertaintyArticulatorResidualDecoder(nn.Module):
    def __init__(
        self,
        pose_group_dim,
        rotation_count,
        code_dim,
        retrieval_dim,
        hidden_dim=256,
        depth=4,
        time_fourier_bands=10,
        omega0_first=20.0,
        omega0_hidden=1.0,
        gate_bias=-1.0,
        expression_dim=0,
    ):
        super().__init__()
        self.rotation_count = int(rotation_count)
        self.expression_dim = int(expression_dim)
        self.time_fourier_bands = int(time_fourier_bands)
        time_dim = 1 + 2 * self.time_fourier_bands
        input_dim = time_dim + int(pose_group_dim) * 3 + int(code_dim) + int(retrieval_dim) + 2
        layers = [SineLayer(input_dim, int(hidden_dim), omega=omega0_first, is_first=True)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(
                SineLayer(
                    int(hidden_dim),
                    int(hidden_dim),
                    omega=omega0_hidden,
                    is_first=False,
                )
            )
        self.net = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(int(hidden_dim))
        self.rotation_head = nn.Linear(int(hidden_dim), self.rotation_count * 3)
        self.expression_head = (
            nn.Linear(int(hidden_dim), self.expression_dim)
            if self.expression_dim > 0
            else None
        )
        self.gate_head = nn.Linear(int(hidden_dim), 1)
        for head in (self.rotation_head, self.expression_head):
            if head is not None:
                nn.init.uniform_(head.weight, -1e-4, 1e-4)
                nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def forward(
        self,
        tau,
        scaffold_group,
        velocity_group,
        acceleration_group,
        frame_code,
        retrieval_features,
        trust,
        correction_need,
    ):
        features = torch.cat(
            [
                fourier_encode_scalar(tau, self.time_fourier_bands),
                scaffold_group,
                velocity_group,
                acceleration_group,
                frame_code,
                retrieval_features,
                trust,
                correction_need,
            ],
            dim=-1,
        )
        hidden = self.out_norm(self.net(features))
        expression = self.expression_head(hidden) if self.expression_head is not None else None
        return self.rotation_head(hidden), expression, torch.sigmoid(self.gate_head(hidden))


class RetrievalUncertaintyAdaptiveKnotField(nn.Module):
    def __init__(
        self,
        pose_dim=COMPACT6D_DIM,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        code_dim=64,
        context_hidden_dim=256,
        context_layers=1,
        context_heads=8,
        articulator_code_strides=None,
        knot_kernel_width=0.75,
        hidden_dim=256,
        depth=4,
        time_fourier_bands=10,
        context_time_fourier_bands=6,
        omega0_first=20.0,
        omega0_hidden=1.0,
        residual_scale_init=0.1,
        residual_scale_learnable=True,
        body_gate_bias=-2.0,
        hand_gate_bias=-0.5,
        face_gate_bias=-2.0,
        calibrator_hidden_dim=128,
        calibrator_text_dim=64,
        calibrator_time_fourier_bands=4,
        trust_initial=0.5,
        correction_need_initial=0.5,
        knot_density_initial=0.5,
        correction_need_floor=0.25,
        allocation_scale_bias=1.5,
        evidence_density_bias=1.0,
        density_floor=0.05,
        duration_hidden_dim=256,
        duration_initial_frames=80.0,
        dropout=0.0,
        initialize_code_predictor=True,
    ):
        super().__init__()
        if int(pose_dim) != COMPACT6D_DIM:
            raise ValueError(
                f"RetrievalUncertaintyAdaptiveKnotField requires pose_dim={COMPACT6D_DIM}."
            )
        if int(retrieval_dim) != len(RETRIEVAL_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(RETRIEVAL_FEATURE_NAMES)} retrieval features, got {retrieval_dim}."
            )
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.correction_need_floor = float(correction_need_floor)
        self.allocation_scale_bias = float(allocation_scale_bias)
        configured = articulator_code_strides or DEFAULT_ARTICULATOR_STRIDES
        self.articulator_code_strides = {
            name: tuple(max(int(value), 1) for value in configured[name])
            for name in ARTICULATOR_NAMES
        }
        scale_counts = {len(value) for value in self.articulator_code_strides.values()}
        if len(scale_counts) != 1:
            raise ValueError("Every articulator must define the same number of knot scales.")
        self.num_scales = scale_counts.pop()
        self.code_strides = self.articulator_code_strides["body"]

        self.calibrator = ArticulatorTrustNeedCalibrator(
            text_dim=int(text_dim),
            retrieval_dim=self.retrieval_dim,
            hidden_dim=int(calibrator_hidden_dim),
            text_hidden_dim=int(calibrator_text_dim),
            num_scales=self.num_scales,
            time_fourier_bands=int(calibrator_time_fourier_bands),
            initial_trust=float(trust_initial),
            initial_need=float(correction_need_initial),
            initial_density=float(knot_density_initial),
            evidence_density_bias=float(evidence_density_bias),
            density_floor=float(density_floor),
            dropout=float(dropout),
        )
        self.code_predictor = None
        if bool(initialize_code_predictor):
            self.code_predictor = AdaptiveKnotArticulatorCodePredictor(
                pose_dim=self.pose_dim,
                text_dim=int(text_dim),
                retrieval_dim=self.retrieval_dim,
                code_dim=self.code_dim,
                hidden_dim=int(context_hidden_dim),
                articulator_strides=self.articulator_code_strides,
                num_layers=int(context_layers),
                num_heads=int(context_heads),
                time_fourier_bands=int(context_time_fourier_bands),
                knot_kernel_width=float(knot_kernel_width),
                dropout=float(dropout),
            )
        self.duration_head = TextDurationHead(
            text_dim=int(text_dim),
            hidden_dim=int(duration_hidden_dim),
            initial_frames=float(duration_initial_frames),
            dropout=float(dropout),
        )

        gate_biases = {
            "body": float(body_gate_bias),
            "lhand": float(hand_gate_bias),
            "rhand": float(hand_gate_bias),
            "face": float(face_gate_bias),
        }
        self.decoders = nn.ModuleDict()
        for name in ARTICULATOR_NAMES:
            feature_slice = ROTATION_GROUPS[name][0]
            feature_dim = 16 if name == "face" else feature_slice.stop - feature_slice.start
            self.decoders[name] = UncertaintyArticulatorResidualDecoder(
                pose_group_dim=feature_dim,
                rotation_count=ROTATION_GROUPS[name][1],
                code_dim=self.code_dim,
                retrieval_dim=self.retrieval_dim,
                hidden_dim=int(hidden_dim),
                depth=int(depth),
                time_fourier_bands=int(time_fourier_bands),
                omega0_first=float(omega0_first),
                omega0_hidden=float(omega0_hidden),
                gate_bias=gate_biases[name],
                expression_dim=10 if name == "face" else 0,
            )

        scale = torch.full(
            (len(ARTICULATOR_NAMES),), float(residual_scale_init), dtype=torch.float32
        )
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)
        self.register_buffer(
            "scale_rank",
            torch.linspace(-1.0, 1.0, self.num_scales, dtype=torch.float32),
        )
        stride_tensor = torch.tensor(
            [self.articulator_code_strides[name] for name in ARTICULATOR_NAMES],
            dtype=torch.long,
        )
        self.register_buffer("articulator_stride_tensor", stride_tensor)

    def predict_log_frames(self, text_tokens, text_mask=None):
        return self.duration_head(text_tokens, text_mask=text_mask)

    def predict_lengths(
        self,
        text_tokens,
        text_mask=None,
        min_frames=16,
        max_frames=400,
        multiple=4,
    ):
        frames = torch.exp(self.predict_log_frames(text_tokens, text_mask=text_mask))
        frames = frames.clamp(float(min_frames), float(max_frames))
        multiple = max(int(multiple), 1)
        if multiple > 1:
            frames = torch.round(frames / multiple) * multiple
        return frames.round().long().clamp(int(min_frames), int(max_frames))

    def _compose_prediction(self, scaffold, rotation_deltas, expression_delta, mask):
        prediction = scaffold.clone()
        for name in ARTICULATOR_NAMES:
            feature_slice, rotation_count = ROTATION_GROUPS[name]
            base = scaffold[..., feature_slice].reshape(
                *scaffold.shape[:-1], rotation_count, 6
            )
            delta = rotation_deltas[name].reshape(
                *scaffold.shape[:-1], rotation_count, 3
            )
            composed = torch.matmul(rotation_6d_to_matrix(base), axis_angle_to_matrix(delta))
            prediction[..., feature_slice] = matrix_to_rotation_6d(composed).reshape(
                *scaffold.shape[:-1], rotation_count * 6
            )
        prediction[..., COMPACT6D_EXPRESSION] = (
            scaffold[..., COMPACT6D_EXPRESSION] + expression_delta
        )
        return prediction * mask.unsqueeze(-1).to(prediction.dtype)

    def forward(
        self,
        tau,
        scaffold,
        mask,
        text_tokens,
        retrieval_features,
        text_mask=None,
        trust_override=None,
        correction_need_override=None,
        knot_density_override=None,
    ):
        if tau.ndim == 2:
            tau = tau.unsqueeze(-1)
        if text_mask is None:
            text_mask = torch.ones(
                text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device
            )
        if (
            retrieval_features.shape[:2] != scaffold.shape[:2]
            or retrieval_features.shape[-1] != self.retrieval_dim
        ):
            raise ValueError(
                f"Expected retrieval features [B,T,{self.retrieval_dim}], "
                f"got {tuple(retrieval_features.shape)}"
            )
        mask = mask.bool()
        retrieval_features = retrieval_features.to(
            device=scaffold.device, dtype=scaffold.dtype
        )
        retrieval_features = retrieval_features * mask.unsqueeze(-1).to(scaffold.dtype)
        velocity, acceleration = temporal_differences(scaffold, mask)
        calibrated = self.calibrator(
            tau,
            retrieval_features,
            velocity,
            acceleration,
            text_tokens,
            text_mask,
        )
        trust = calibrated["trust"]
        correction_need = calibrated["correction_need"]
        knot_density = calibrated["knot_density"]
        if trust_override is not None:
            trust = trust_override.to(device=scaffold.device, dtype=scaffold.dtype).clamp(0.0, 1.0)
        if correction_need_override is not None:
            correction_need = correction_need_override.to(
                device=scaffold.device, dtype=scaffold.dtype
            ).clamp(0.0, 1.0)
        if knot_density_override is not None:
            knot_density = knot_density_override.to(
                device=scaffold.device, dtype=scaffold.dtype
            ).clamp_min(1e-4)
        trust = trust * mask.unsqueeze(-1).to(trust.dtype)
        correction_need = correction_need * mask.unsqueeze(-1).to(correction_need.dtype)
        knot_density = knot_density * mask.unsqueeze(-1).to(knot_density.dtype)

        code_output = self.code_predictor(
            scaffold,
            mask,
            tau,
            retrieval_features,
            text_tokens,
            text_mask,
            knot_density,
            velocity=velocity,
            acceleration=acceleration,
        )
        allocation_prior = (
            self.allocation_scale_bias
            * (trust - correction_need).unsqueeze(-1)
            * self.scale_rank.to(dtype=scaffold.dtype).view(1, 1, 1, -1)
        )
        scale_weights = torch.softmax(
            calibrated["scale_logits"] + allocation_prior, dim=-1
        )
        frame_codes = (
            code_output["frame_codes_by_scale"] * scale_weights.unsqueeze(-1)
        ).sum(dim=3)

        rotation_deltas = {}
        expression_delta = scaffold.new_zeros(*scaffold.shape[:-1], 10)
        learned_gates = []
        effective_gates = []
        for group_index, name in enumerate(ARTICULATOR_NAMES):
            raw_rotation, raw_expression, learned_gate = self.decoders[name](
                tau,
                _group_features(scaffold, name),
                _group_features(velocity, name),
                _group_features(acceleration, name),
                frame_codes[:, :, group_index],
                retrieval_features,
                trust[:, :, group_index : group_index + 1],
                correction_need[:, :, group_index : group_index + 1],
            )
            need_gate = self.correction_need_floor + (
                1.0 - self.correction_need_floor
            ) * correction_need[:, :, group_index : group_index + 1]
            effective_gate = learned_gate * need_gate
            scale = self.residual_scale[group_index].to(dtype=scaffold.dtype)
            rotation_deltas[name] = scale * effective_gate * raw_rotation
            if raw_expression is not None:
                expression_delta = scale * effective_gate * raw_expression
            learned_gates.append(learned_gate)
            effective_gates.append(effective_gate)

        prediction = self._compose_prediction(
            scaffold, rotation_deltas, expression_delta, mask
        )
        residual = (prediction - scaffold) * mask.unsqueeze(-1).to(scaffold.dtype)
        correction_axis = torch.cat(
            [
                rotation_deltas["body"],
                rotation_deltas["lhand"],
                rotation_deltas["rhand"],
                rotation_deltas["face"],
                expression_delta,
            ],
            dim=-1,
        ) * mask.unsqueeze(-1).to(scaffold.dtype)
        mask_float = mask.unsqueeze(-1).to(scaffold.dtype)
        result = {
            "prediction": prediction,
            "residual": residual,
            "correction_axis": correction_axis,
            "confidence": trust,
            "trust": trust,
            "trust_logits": calibrated["trust_logits"],
            "correction_need": correction_need,
            "correction_need_logits": calibrated["correction_need_logits"],
            "knot_density": knot_density,
            "knot_density_logits": calibrated["knot_density_logits"],
            "retrieval_uncertainty": calibrated["retrieval_uncertainty"]
            * mask.to(scaffold.dtype),
            "scale_weights": scale_weights
            * mask[:, :, None, None].to(scale_weights.dtype),
            "frame_codes": frame_codes
            * mask[:, :, None, None].to(frame_codes.dtype),
            "scale_codes": code_output["scale_codes"],
            "scale_code_masks": code_output["scale_code_masks"],
            "adaptive_coordinates": code_output["adaptive_coordinates"],
            "knot_counts": code_output["knot_counts"],
            "learned_gates": torch.cat(learned_gates, dim=-1) * mask_float,
            "gates": torch.cat(effective_gates, dim=-1) * mask_float,
            "pred_log_frames": self.predict_log_frames(text_tokens, text_mask=text_mask),
        }
        for key in ("segment_positions", "segment_boundary_mask"):
            if key in code_output:
                result[key] = code_output[key]
        return result


def correction_need_target_from_error(errors, temperatures, mask):
    temperatures = torch.as_tensor(
        temperatures, device=errors.device, dtype=errors.dtype
    )
    if temperatures.numel() != len(ARTICULATOR_NAMES):
        raise ValueError(f"Expected {len(ARTICULATOR_NAMES)} correction-need temperatures.")
    target = 1.0 - torch.exp(
        -errors / temperatures.view(1, 1, -1).clamp_min(1e-4)
    )
    return target * mask.unsqueeze(-1).to(target.dtype)


def adaptive_knot_density_target(
    target_correction,
    correction_need,
    mask,
    transition_weight=0.5,
):
    """Put knot density on large corrections and rapidly changing transitions."""

    transition = target_correction.new_zeros(
        *target_correction.shape[:2], len(ARTICULATOR_NAMES)
    )
    if target_correction.shape[1] > 1:
        for index, name in enumerate(ARTICULATOR_NAMES):
            values = target_correction[..., CORRECTION_GROUP_SLICES[name]]
            change = torch.sqrt(
                (values[:, 1:] - values[:, :-1]).square().mean(dim=-1).clamp_min(1e-8)
            )
            transition[:, 1:, index] = change
    valid = mask.unsqueeze(-1).to(transition.dtype)
    mean_change = (transition * valid).sum(dim=1, keepdim=True) / valid.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    normalized_change = (transition / (2.0 * mean_change + 1e-6)).clamp(0.0, 1.0)
    target = (correction_need + float(transition_weight) * normalized_change).clamp(0.0, 1.0)
    return target * valid
