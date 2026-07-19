"""Confidence-adaptive multi-scale implicit field with SO(3) residuals."""

from __future__ import annotations

import math

import torch
from torch import nn

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import (
    COMPACT6D_DIM,
    COMPACT6D_EXPRESSION,
    COMPACT6D_JAW,
    COMPACT6D_LEFT_HAND,
    COMPACT6D_RIGHT_HAND,
    COMPACT6D_UPPER_BODY,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from NIAF.continuous_sign_field.models.local_implicit import (
    TextDurationHead,
    _masked_window_pool,
    interpolate_local_codes,
)
from NIAF.continuous_sign_field.models.meta_implicit import (
    SineLayer,
    fourier_encode_scalar,
    masked_mean,
    temporal_differences,
)
from NIAF.oracle_smplx_field.geometry.rotation import geodesic_distance


ARTICULATOR_NAMES = ("body", "lhand", "rhand", "face")
ROTATION_GROUPS = {
    "body": (COMPACT6D_UPPER_BODY, 10),
    "lhand": (COMPACT6D_LEFT_HAND, 15),
    "rhand": (COMPACT6D_RIGHT_HAND, 15),
    "face": (COMPACT6D_JAW, 1),
}

__all__ = [
    "ARTICULATOR_NAMES",
    "ROTATION_GROUPS",
    "ArticulatorResidualDecoder",
    "MultiScaleArticulatorCodePredictor",
    "RetrievalConfidenceAdaptiveField",
    "RetrievalConfidenceCalibrator",
    "ScaleCodeEncoder",
    "articulator_scaffold_error",
    "confidence_target_from_error",
    "target_tangent_correction",
]


def _group_features(values, name):
    if name == "face":
        return torch.cat([values[..., COMPACT6D_JAW], values[..., COMPACT6D_EXPRESSION]], dim=-1)
    return values[..., ROTATION_GROUPS[name][0]]


def _group_dynamics(velocity, acceleration):
    summaries = []
    for values in (velocity, acceleration):
        for name in ARTICULATOR_NAMES:
            group = _group_features(values, name)
            summaries.append(torch.sqrt(group.square().mean(dim=-1).clamp_min(1e-8)))
    return torch.stack(summaries, dim=-1)


class ScaleCodeEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        text_hidden_dim,
        code_dim,
        hidden_dim,
        stride,
        num_groups=4,
        num_layers=1,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.stride = max(int(stride), 1)
        self.code_dim = int(code_dim)
        self.num_groups = int(num_groups)
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.text_proj = nn.Linear(int(text_hidden_dim), int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.out = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), self.num_groups * self.code_dim),
        )

    def forward(self, frame_features, mask, text_features, text_mask):
        local_features, local_mask = _masked_window_pool(frame_features, mask, self.stride)
        local_features = self.input_proj(local_features)
        text_features = self.text_proj(text_features)
        attended, _ = self.cross_attention(
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
        codes = self.out(encoded).reshape(
            encoded.shape[0],
            encoded.shape[1],
            self.num_groups,
            self.code_dim,
        )
        codes = codes * local_mask[:, :, None, None].to(codes.dtype)
        return codes, local_mask


class MultiScaleArticulatorCodePredictor(nn.Module):
    def __init__(
        self,
        pose_dim=COMPACT6D_DIM,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        code_dim=64,
        hidden_dim=256,
        strides=(4, 8, 16),
        num_layers=1,
        num_heads=8,
        time_fourier_bands=6,
        dropout=0.0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.strides = tuple(sorted({max(int(value), 1) for value in strides}))
        self.time_fourier_bands = int(time_fourier_bands)
        time_dim = 1 + 2 * self.time_fourier_bands
        frame_dim = self.pose_dim * 3 + int(retrieval_dim) + time_dim
        self.text_norm = nn.LayerNorm(int(text_dim))
        self.encoders = nn.ModuleList(
            [
                ScaleCodeEncoder(
                    input_dim=frame_dim,
                    text_hidden_dim=int(text_dim),
                    code_dim=self.code_dim,
                    hidden_dim=int(hidden_dim),
                    stride=stride,
                    num_groups=len(ARTICULATOR_NAMES),
                    num_layers=int(num_layers),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                )
                for stride in self.strides
            ]
        )

    def forward(self, scaffold, mask, tau, retrieval_features, text_tokens, text_mask):
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
        text_features = self.text_norm(text_tokens.to(dtype=scaffold.dtype))
        scale_codes = []
        scale_masks = []
        frame_codes = []
        for encoder in self.encoders:
            codes, code_mask = encoder(frame_features, mask, text_features, text_mask)
            flat_codes = codes.flatten(start_dim=2)
            interpolated = interpolate_local_codes(flat_codes, code_mask, tau).reshape(
                scaffold.shape[0],
                scaffold.shape[1],
                len(ARTICULATOR_NAMES),
                self.code_dim,
            )
            scale_codes.append(codes)
            scale_masks.append(code_mask)
            frame_codes.append(interpolated)
        return {
            "scale_codes": scale_codes,
            "scale_code_masks": scale_masks,
            "frame_codes_by_scale": torch.stack(frame_codes, dim=3),
            "velocity": velocity,
            "acceleration": acceleration,
        }


class RetrievalConfidenceCalibrator(nn.Module):
    def __init__(
        self,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        hidden_dim=128,
        text_hidden_dim=64,
        num_scales=3,
        time_fourier_bands=4,
        initial_confidence=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.time_fourier_bands = int(time_fourier_bands)
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
        self.confidence_head = nn.Linear(int(hidden_dim), len(ARTICULATOR_NAMES))
        self.scale_head = nn.Linear(int(hidden_dim), len(ARTICULATOR_NAMES) * self.num_scales)
        nn.init.zeros_(self.confidence_head.weight)
        initial_confidence = min(max(float(initial_confidence), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.confidence_head.bias, math.log(initial_confidence / (1.0 - initial_confidence)))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, tau, retrieval_features, velocity, acceleration, text_tokens, text_mask):
        pooled_text = masked_mean(text_tokens, text_mask, dim=1)
        text_context = self.text_proj(pooled_text).unsqueeze(1).expand(-1, retrieval_features.shape[1], -1)
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
        confidence_logits = self.confidence_head(hidden)
        scale_logits = self.scale_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(ARTICULATOR_NAMES),
            self.num_scales,
        )
        return confidence_logits, scale_logits


class ArticulatorResidualDecoder(nn.Module):
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
        gate_bias=-2.0,
        expression_dim=0,
    ):
        super().__init__()
        self.rotation_count = int(rotation_count)
        self.expression_dim = int(expression_dim)
        self.time_fourier_bands = int(time_fourier_bands)
        time_dim = 1 + 2 * self.time_fourier_bands
        input_dim = time_dim + int(pose_group_dim) * 3 + int(code_dim) + int(retrieval_dim) + 1
        layers = [SineLayer(input_dim, int(hidden_dim), omega=omega0_first, is_first=True)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(SineLayer(int(hidden_dim), int(hidden_dim), omega=omega0_hidden, is_first=False))
        self.net = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(int(hidden_dim))
        self.rotation_head = nn.Linear(int(hidden_dim), self.rotation_count * 3)
        self.expression_head = (
            nn.Linear(int(hidden_dim), self.expression_dim) if self.expression_dim > 0 else None
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
        confidence,
    ):
        features = torch.cat(
            [
                fourier_encode_scalar(tau, self.time_fourier_bands),
                scaffold_group,
                velocity_group,
                acceleration_group,
                frame_code,
                retrieval_features,
                confidence,
            ],
            dim=-1,
        )
        hidden = self.out_norm(self.net(features))
        expression = self.expression_head(hidden) if self.expression_head is not None else None
        return self.rotation_head(hidden), expression, torch.sigmoid(self.gate_head(hidden))


class RetrievalConfidenceAdaptiveField(nn.Module):
    def __init__(
        self,
        pose_dim=COMPACT6D_DIM,
        text_dim=768,
        retrieval_dim=len(RETRIEVAL_FEATURE_NAMES),
        code_dim=64,
        context_hidden_dim=256,
        context_layers=1,
        context_heads=8,
        code_strides=(4, 8, 16),
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
        confidence_hidden_dim=128,
        confidence_text_dim=64,
        confidence_time_fourier_bands=4,
        confidence_initial=0.5,
        confidence_residual_floor=0.1,
        confidence_scale_bias=1.0,
        duration_hidden_dim=256,
        duration_initial_frames=80.0,
        dropout=0.0,
    ):
        super().__init__()
        if int(pose_dim) != COMPACT6D_DIM:
            raise ValueError(f"RetrievalConfidenceAdaptiveField requires pose_dim={COMPACT6D_DIM}.")
        if int(retrieval_dim) != len(RETRIEVAL_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(RETRIEVAL_FEATURE_NAMES)} retrieval features, got {retrieval_dim}."
            )
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.code_strides = tuple(sorted({max(int(value), 1) for value in code_strides}))
        self.confidence_residual_floor = float(confidence_residual_floor)
        self.confidence_scale_bias = float(confidence_scale_bias)

        self.code_predictor = MultiScaleArticulatorCodePredictor(
            pose_dim=self.pose_dim,
            text_dim=int(text_dim),
            retrieval_dim=self.retrieval_dim,
            code_dim=self.code_dim,
            hidden_dim=int(context_hidden_dim),
            strides=self.code_strides,
            num_layers=int(context_layers),
            num_heads=int(context_heads),
            time_fourier_bands=int(context_time_fourier_bands),
            dropout=float(dropout),
        )
        self.confidence_calibrator = RetrievalConfidenceCalibrator(
            text_dim=int(text_dim),
            retrieval_dim=self.retrieval_dim,
            hidden_dim=int(confidence_hidden_dim),
            text_hidden_dim=int(confidence_text_dim),
            num_scales=len(self.code_strides),
            time_fourier_bands=int(confidence_time_fourier_bands),
            initial_confidence=float(confidence_initial),
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
            self.decoders[name] = ArticulatorResidualDecoder(
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

        scale = torch.full((len(ARTICULATOR_NAMES),), float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)
        scale_rank = torch.linspace(-1.0, 1.0, len(self.code_strides), dtype=torch.float32)
        self.register_buffer("scale_rank", scale_rank)

    def predict_log_frames(self, text_tokens, text_mask=None):
        return self.duration_head(text_tokens, text_mask=text_mask)

    def predict_lengths(self, text_tokens, text_mask=None, min_frames=40, max_frames=400, multiple=4):
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
            base = scaffold[..., feature_slice].reshape(*scaffold.shape[:-1], rotation_count, 6)
            delta = rotation_deltas[name].reshape(*scaffold.shape[:-1], rotation_count, 3)
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
        confidence_override=None,
    ):
        if tau.ndim == 2:
            tau = tau.unsqueeze(-1)
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        if retrieval_features.shape[:2] != scaffold.shape[:2] or retrieval_features.shape[-1] != self.retrieval_dim:
            raise ValueError(
                f"Expected retrieval features [B,T,{self.retrieval_dim}], got {tuple(retrieval_features.shape)}"
            )
        retrieval_features = retrieval_features.to(device=scaffold.device, dtype=scaffold.dtype)
        retrieval_features = retrieval_features * mask.unsqueeze(-1).to(scaffold.dtype)

        code_output = self.code_predictor(
            scaffold,
            mask,
            tau,
            retrieval_features,
            text_tokens,
            text_mask,
        )
        velocity = code_output["velocity"]
        acceleration = code_output["acceleration"]
        confidence_logits, learned_scale_logits = self.confidence_calibrator(
            tau,
            retrieval_features,
            velocity,
            acceleration,
            text_tokens.to(dtype=scaffold.dtype),
            text_mask,
        )
        confidence = torch.sigmoid(confidence_logits)
        if confidence_override is not None:
            if confidence_override.shape != confidence.shape:
                raise ValueError(
                    f"Expected confidence_override {tuple(confidence.shape)}, got {tuple(confidence_override.shape)}"
                )
            confidence = confidence_override.to(device=scaffold.device, dtype=scaffold.dtype).clamp(0.0, 1.0)

        scale_prior = (
            self.confidence_scale_bias
            * (2.0 * confidence.unsqueeze(-1) - 1.0)
            * self.scale_rank.to(dtype=scaffold.dtype).view(1, 1, 1, -1)
        )
        scale_weights = torch.softmax(learned_scale_logits + scale_prior, dim=-1)
        frame_codes = (
            code_output["frame_codes_by_scale"] * scale_weights.unsqueeze(-1)
        ).sum(dim=3)

        rotation_deltas = {}
        expression_delta = scaffold.new_zeros(*scaffold.shape[:-1], 10)
        learned_gates = []
        effective_gates = []
        for group_idx, name in enumerate(ARTICULATOR_NAMES):
            raw_rotation, raw_expression, learned_gate = self.decoders[name](
                tau,
                _group_features(scaffold, name),
                _group_features(velocity, name),
                _group_features(acceleration, name),
                frame_codes[:, :, group_idx],
                retrieval_features,
                confidence[:, :, group_idx : group_idx + 1],
            )
            confidence_gate = self.confidence_residual_floor + (
                1.0 - self.confidence_residual_floor
            ) * (1.0 - confidence[:, :, group_idx : group_idx + 1])
            effective_gate = learned_gate * confidence_gate
            scale = self.residual_scale[group_idx].to(dtype=scaffold.dtype)
            rotation_deltas[name] = scale * effective_gate * raw_rotation
            if raw_expression is not None:
                expression_delta = scale * effective_gate * raw_expression
            learned_gates.append(learned_gate)
            effective_gates.append(effective_gate)

        prediction = self._compose_prediction(scaffold, rotation_deltas, expression_delta, mask)
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
        return {
            "prediction": prediction,
            "residual": residual,
            "correction_axis": correction_axis,
            "confidence": confidence * mask_float,
            "confidence_logits": confidence_logits,
            "scale_weights": scale_weights * mask[:, :, None, None].to(scale_weights.dtype),
            "frame_codes": frame_codes * mask[:, :, None, None].to(frame_codes.dtype),
            "scale_codes": code_output["scale_codes"],
            "scale_code_masks": code_output["scale_code_masks"],
            "learned_gates": torch.cat(learned_gates, dim=-1) * mask_float,
            "gates": torch.cat(effective_gates, dim=-1) * mask_float,
            "pred_log_frames": self.predict_log_frames(text_tokens, text_mask=text_mask),
        }


def articulator_scaffold_error(scaffold, target, mask, expression_weight=0.1):
    errors = []
    for name in ARTICULATOR_NAMES:
        feature_slice, rotation_count = ROTATION_GROUPS[name]
        scaffold_rot = scaffold[..., feature_slice].reshape(*scaffold.shape[:-1], rotation_count, 6)
        target_rot = target[..., feature_slice].reshape(*target.shape[:-1], rotation_count, 6)
        error = geodesic_distance(
            rotation_6d_to_matrix(scaffold_rot),
            rotation_6d_to_matrix(target_rot),
        ).mean(dim=-1)
        if name == "face":
            expression_error = torch.abs(
                scaffold[..., COMPACT6D_EXPRESSION] - target[..., COMPACT6D_EXPRESSION]
            ).mean(dim=-1)
            error = error + float(expression_weight) * expression_error
        errors.append(error)
    output = torch.stack(errors, dim=-1)
    return output * mask.unsqueeze(-1).to(output.dtype)


def confidence_target_from_error(errors, temperatures, mask):
    temperatures = torch.as_tensor(temperatures, device=errors.device, dtype=errors.dtype)
    if temperatures.numel() != len(ARTICULATOR_NAMES):
        raise ValueError(f"Expected {len(ARTICULATOR_NAMES)} confidence temperatures.")
    target = torch.exp(-errors / temperatures.view(1, 1, -1).clamp_min(1e-4))
    return target * mask.unsqueeze(-1).to(target.dtype)


def target_tangent_correction(scaffold, target, mask):
    """Return the scaffold-to-target correction in local SO(3) tangent spaces."""

    corrections = []
    for name in ARTICULATOR_NAMES:
        feature_slice, rotation_count = ROTATION_GROUPS[name]
        scaffold_rot = scaffold[..., feature_slice].reshape(*scaffold.shape[:-1], rotation_count, 6)
        target_rot = target[..., feature_slice].reshape(*target.shape[:-1], rotation_count, 6)
        scaffold_matrix = rotation_6d_to_matrix(scaffold_rot)
        target_matrix = rotation_6d_to_matrix(target_rot)
        relative = torch.matmul(scaffold_matrix.transpose(-1, -2), target_matrix)
        corrections.append(matrix_to_axis_angle(relative).flatten(start_dim=-2))
    corrections.append(target[..., COMPACT6D_EXPRESSION] - scaffold[..., COMPACT6D_EXPRESSION])
    output = torch.cat(corrections, dim=-1)
    return output * mask.unsqueeze(-1).to(output.dtype)
