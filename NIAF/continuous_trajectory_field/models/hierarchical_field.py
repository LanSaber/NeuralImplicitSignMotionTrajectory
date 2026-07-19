from __future__ import annotations

import torch
from torch import nn

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import (
    COMPACT6D_DIM,
    COMPACT6D_EXPRESSION,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from NIAF.continuous_trajectory_field.models.modulated_siren import (
    GroupModulatedSiren,
)
from NIAF.continuous_trajectory_field.models.trajectory_hypernetwork import (
    TrajectoryHypernetwork,
)
from NIAF.continuous_trajectory_field.models.trajectory_instance import (
    TrajectoryInstance,
)


ROTATION_DIM = 246
ROTATION_COUNT = 41
TANGENT_ROTATION_DIM = 123
RESIDUAL_DIM = 133


def _articulator_gate_vector(gates: torch.Tensor):
    if gates.shape[-1] != 4:
        raise ValueError("Expected body, left hand, right hand, and face gates")
    return torch.cat(
        [
            gates[..., 0:1].expand(*gates.shape[:-1], 30),
            gates[..., 1:2].expand(*gates.shape[:-1], 45),
            gates[..., 2:3].expand(*gates.shape[:-1], 45),
            gates[..., 3:4].expand(*gates.shape[:-1], 3),
            gates[..., 3:4].expand(*gates.shape[:-1], 10),
        ],
        dim=-1,
    )


class ContinuousTrajectoryField(nn.Module):
    """Amortized finite representation of a continuous SMPL-X trajectory."""

    def __init__(
        self,
        text_dim: int = 768,
        pose_dim: int = COMPACT6D_DIM,
        retrieval_dim: int = len(RETRIEVAL_FEATURE_NAMES),
        context_hidden_dim: int = 256,
        context_layers: int = 3,
        field_hidden_dim: int = 256,
        field_depth: int = 4,
        max_local_fields: int = 24,
        frames_per_local_field: int = 20,
        minimum_local_width: float = 0.06,
        maximum_local_width: float = 0.50,
        quantile_temperature: float = 0.02,
        omega0_first: float = 20.0,
        omega0_hidden: float = 1.0,
        residual_amplitude: float = 0.10,
        residual_amplitude_learnable: bool = True,
        initial_duration_seconds: float = 4.0,
        minimum_duration_seconds: float = 0.8,
        maximum_duration_seconds: float = 20.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(pose_dim) != COMPACT6D_DIM:
            raise ValueError(f"Continuous SMPL-X field expects pose_dim={COMPACT6D_DIM}")
        self.pose_dim = int(pose_dim)
        self.residual_dim = RESIDUAL_DIM
        self.hypernetwork = TrajectoryHypernetwork(
            text_dim=text_dim,
            pose_dim=pose_dim,
            retrieval_dim=retrieval_dim,
            context_hidden_dim=context_hidden_dim,
            context_layers=context_layers,
            field_hidden_dim=field_hidden_dim,
            field_depth=field_depth,
            residual_dim=self.residual_dim,
            max_local_fields=max_local_fields,
            frames_per_local_field=frames_per_local_field,
            minimum_local_width=minimum_local_width,
            maximum_local_width=maximum_local_width,
            quantile_temperature=quantile_temperature,
            initial_duration_seconds=initial_duration_seconds,
            minimum_duration_seconds=minimum_duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
            dropout=dropout,
        )
        self.prior_field = GroupModulatedSiren(
            input_dim=1,
            output_dim=self.pose_dim,
            hidden_dim=field_hidden_dim,
            depth=field_depth,
            omega0_first=omega0_first,
            omega0_hidden=omega0_hidden,
            output_init="identity_rot6d",
        )
        self.residual_field = GroupModulatedSiren(
            input_dim=1,
            output_dim=self.residual_dim,
            hidden_dim=field_hidden_dim,
            depth=field_depth,
            omega0_first=omega0_first,
            omega0_hidden=omega0_hidden,
            output_init="zero",
        )
        amplitude = torch.tensor(float(residual_amplitude), dtype=torch.float32)
        if residual_amplitude_learnable:
            self.residual_amplitude = nn.Parameter(amplitude)
        else:
            self.register_buffer("residual_amplitude", amplitude)

    def predict_duration(self, text_tokens, text_mask=None):
        return self.hypernetwork.predict_duration(text_tokens, text_mask=text_mask)

    def predict_lengths(
        self,
        text_tokens,
        text_mask=None,
        fps: float = 20.0,
        min_frames: int = 16,
        max_frames: int = 400,
        multiple: int = 4,
    ):
        _log_duration, duration = self.predict_duration(text_tokens, text_mask=text_mask)
        frames = duration * float(fps)
        multiple = max(int(multiple), 1)
        if multiple > 1:
            frames = torch.round(frames / multiple) * multiple
        return frames.round().long().clamp(int(min_frames), int(max_frames))

    def encode_trajectory(
        self,
        text_tokens: torch.Tensor,
        adapter_context: torch.Tensor,
        context_mask: torch.Tensor,
        retrieval_evidence: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> TrajectoryInstance:
        return self.hypernetwork(
            text_tokens=text_tokens,
            adapter_context=adapter_context,
            context_mask=context_mask,
            retrieval_evidence=retrieval_evidence,
            text_mask=text_mask,
        )

    @staticmethod
    def normalize_query_times(
        trajectory: TrajectoryInstance,
        query_times: torch.Tensor,
        time_domain: str,
    ):
        if query_times.ndim == 3 and query_times.shape[-1] == 1:
            query_times = query_times.squeeze(-1)
        if query_times.ndim != 2 or query_times.shape[0] != trajectory.batch_size:
            raise ValueError(
                f"Expected query times [B,K] for B={trajectory.batch_size}, "
                f"got {tuple(query_times.shape)}"
            )
        query_times = query_times.to(device=trajectory.device, dtype=trajectory.dtype)
        domain = str(time_domain).lower()
        if domain in {"normalized", "tau", "minus_one_one"}:
            tau = query_times
        elif domain in {"unit", "zero_one"}:
            tau = 2.0 * query_times - 1.0
        elif domain in {"seconds", "physical"}:
            tau = 2.0 * query_times / trajectory.duration_seconds[:, None] - 1.0
        else:
            raise ValueError(f"Unsupported time domain {time_domain!r}")
        return tau.clamp(-1.0, 1.0)

    def _local_residual(self, trajectory: TrajectoryInstance, tau: torch.Tensor):
        batch, queries = tau.shape
        local_count = trajectory.num_local_fields
        if local_count == 0:
            return tau.new_zeros(batch, queries, self.residual_dim), tau.new_zeros(
                batch, queries, 0
            )
        width = trajectory.local_widths.clamp_min(1e-4)
        local_tau = (
            tau[:, None, :] - trajectory.local_centers[:, :, None]
        ) / width[:, :, None]
        flat_coordinates = local_tau.reshape(batch * local_count, queries, 1)
        flat_scale = trajectory.local_scale.reshape(
            batch * local_count,
            *trajectory.local_scale.shape[2:],
        )
        flat_shift = trajectory.local_shift.reshape_as(flat_scale)
        flat_bias = trajectory.local_output_bias.reshape(batch * local_count, -1)
        local_values = self.residual_field(
            flat_coordinates,
            flat_scale,
            flat_shift,
            flat_bias,
        ).reshape(batch, local_count, queries, self.residual_dim)
        logits = -0.5 * local_tau.square()
        logits = logits.masked_fill(~trajectory.local_mask[:, :, None], -torch.inf)
        all_inactive = ~trajectory.local_mask.any(dim=1)
        if bool(all_inactive.any()):
            logits = logits.clone()
            logits[all_inactive] = 0.0
        weights = torch.softmax(logits, dim=1)
        weights = weights * trajectory.local_mask[:, :, None].to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        output = (local_values * weights.unsqueeze(-1)).sum(dim=1)
        return output, weights.transpose(1, 2)

    def query_trajectory(
        self,
        trajectory: TrajectoryInstance,
        query_times: torch.Tensor,
        time_domain: str = "normalized",
        query_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        tau = self.normalize_query_times(trajectory, query_times, time_domain)
        coordinates = tau.unsqueeze(-1)
        prior_raw = self.prior_field(
            coordinates,
            trajectory.prior_scale,
            trajectory.prior_shift,
            trajectory.prior_output_bias,
        )
        global_residual = self.residual_field(
            coordinates,
            trajectory.residual_scale,
            trajectory.residual_shift,
            trajectory.residual_output_bias,
        )
        local_residual, local_weights = self._local_residual(trajectory, tau)
        residual_axis = global_residual + local_residual
        gates = _articulator_gate_vector(trajectory.articulator_gates)
        residual_axis = (
            self.residual_amplitude.to(dtype=residual_axis.dtype)
            * residual_axis
            * gates[:, None, :]
        )

        prior_rotation = prior_raw[..., :ROTATION_DIM].reshape(
            *prior_raw.shape[:-1], ROTATION_COUNT, 6
        )
        prior_matrix = rotation_6d_to_matrix(prior_rotation)
        delta_matrix = axis_angle_to_matrix(
            residual_axis[..., :TANGENT_ROTATION_DIM].reshape(
                *residual_axis.shape[:-1], ROTATION_COUNT, 3
            )
        )
        prediction_matrix = torch.matmul(prior_matrix, delta_matrix)
        prediction_rotation = matrix_to_rotation_6d(prediction_matrix).reshape(
            *prior_raw.shape[:-1], ROTATION_DIM
        )
        expression = (
            prior_raw[..., COMPACT6D_EXPRESSION]
            + residual_axis[..., TANGENT_ROTATION_DIM:]
        )
        prediction = torch.cat([prediction_rotation, expression], dim=-1)

        if query_mask is not None:
            if query_mask.shape != tau.shape:
                raise ValueError("query_mask must match query_times")
            mask_float = query_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
            prediction = prediction * mask_float
            prior_raw = prior_raw * mask_float
            residual_axis = residual_axis * mask_float
        if not return_details:
            return prediction
        return {
            "prediction": prediction,
            "prior": prior_raw,
            "correction_axis": residual_axis,
            "tau": tau,
            "local_weights": local_weights,
            "trajectory": trajectory,
        }

    def forward(
        self,
        text_tokens: torch.Tensor,
        adapter_context: torch.Tensor,
        context_mask: torch.Tensor,
        retrieval_evidence: torch.Tensor,
        query_times: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        time_domain: str = "normalized",
        query_mask: torch.Tensor | None = None,
    ):
        trajectory = self.encode_trajectory(
            text_tokens=text_tokens,
            adapter_context=adapter_context,
            context_mask=context_mask,
            retrieval_evidence=retrieval_evidence,
            text_mask=text_mask,
        )
        return self.query_trajectory(
            trajectory,
            query_times,
            time_domain=time_domain,
            query_mask=query_mask,
            return_details=True,
        )


def build_continuous_trajectory_field(cfg, text_dim: int):
    model_cfg = cfg.get("model", {})
    duration_cfg = cfg.get("duration", {})
    return ContinuousTrajectoryField(
        text_dim=int(text_dim),
        context_hidden_dim=int(model_cfg.get("context_hidden_dim", 256)),
        context_layers=int(model_cfg.get("context_layers", 3)),
        field_hidden_dim=int(model_cfg.get("field_hidden_dim", 256)),
        field_depth=int(model_cfg.get("field_depth", 4)),
        max_local_fields=int(model_cfg.get("max_local_fields", 24)),
        frames_per_local_field=int(model_cfg.get("frames_per_local_field", 20)),
        minimum_local_width=float(model_cfg.get("minimum_local_width", 0.06)),
        maximum_local_width=float(model_cfg.get("maximum_local_width", 0.50)),
        quantile_temperature=float(model_cfg.get("quantile_temperature", 0.02)),
        omega0_first=float(model_cfg.get("omega0_first", 20.0)),
        omega0_hidden=float(model_cfg.get("omega0_hidden", 1.0)),
        residual_amplitude=float(model_cfg.get("residual_amplitude", 0.10)),
        residual_amplitude_learnable=bool(
            model_cfg.get("residual_amplitude_learnable", True)
        ),
        initial_duration_seconds=float(duration_cfg.get("initial_seconds", 4.0)),
        minimum_duration_seconds=float(duration_cfg.get("min_seconds", 0.8)),
        maximum_duration_seconds=float(duration_cfg.get("max_seconds", 20.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
