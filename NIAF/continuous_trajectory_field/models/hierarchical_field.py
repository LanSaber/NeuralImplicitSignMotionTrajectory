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
from NIAF.continuous_trajectory_field.models.dual_mode_trajectory_hypernetwork import (
    DualModeTrajectoryHypernetwork,
)
from NIAF.continuous_trajectory_field.models.sentence_memory_trajectory_hypernetwork import (
    SentenceMemoryTrajectoryHypernetwork,
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
LOCAL_PART_SLICES = {
    "body": slice(0, 30),
    "left_hand": slice(30, 75),
    "right_hand": slice(75, 120),
    "face": slice(120, 133),
}


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
        local_center_mode: str = "retrieval_guided",
        use_retrieval_guidance: bool = True,
        local_window_epsilon: float = 1e-4,
        part_specific_local_experts: bool = False,
        time_dependent_local_gates: bool = False,
        local_body_omega0_first: float = 15.0,
        local_hand_omega0_first: float = 30.0,
        local_face_omega0_first: float = 20.0,
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
        self.local_window_epsilon = max(float(local_window_epsilon), 1e-12)
        self.part_specific_local_experts = bool(part_specific_local_experts)
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
            local_center_mode=local_center_mode,
            use_retrieval_guidance=use_retrieval_guidance,
            time_dependent_local_gates=time_dependent_local_gates,
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
        if self.part_specific_local_experts:
            part_omega = {
                "body": float(local_body_omega0_first),
                "left_hand": float(local_hand_omega0_first),
                "right_hand": float(local_hand_omega0_first),
                "face": float(local_face_omega0_first),
            }
            self.part_local_fields = nn.ModuleDict(
                {
                    name: GroupModulatedSiren(
                        input_dim=1,
                        output_dim=part_slice.stop - part_slice.start,
                        hidden_dim=field_hidden_dim,
                        depth=field_depth,
                        omega0_first=part_omega[name],
                        omega0_hidden=omega0_hidden,
                        output_init="zero",
                    )
                    for name, part_slice in LOCAL_PART_SLICES.items()
                }
            )
        else:
            self.part_local_fields = None
        amplitude = torch.tensor(float(residual_amplitude), dtype=torch.float32)
        if residual_amplitude_learnable:
            self.residual_amplitude = nn.Parameter(amplitude)
        else:
            self.register_buffer("residual_amplitude", amplitude)

    def predict_duration(self, text_tokens, text_mask=None):
        return self.hypernetwork.predict_duration(text_tokens, text_mask=text_mask)

    def reset_local_branch(self):
        """Reset hypernetwork local heads and optional part-specific decoders."""

        self.hypernetwork.reset_local_branch()
        if self.part_local_fields is None:
            return
        for field in self.part_local_fields.values():
            for layer in field.layers:
                layer.reset_parameters()
            nn.init.uniform_(field.output.weight, -1e-3, 1e-3)
            nn.init.zeros_(field.output.bias)

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
            return (
                tau.new_zeros(batch, queries, self.residual_dim),
                tau.new_zeros(batch, queries, 0),
                tau.new_zeros(batch, queries),
                tau.new_zeros(batch, queries, 4),
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
        if self.part_local_fields is None:
            modulated_local_values = self.residual_field(
                flat_coordinates,
                flat_scale,
                flat_shift,
                flat_bias,
            ).reshape(batch, local_count, queries, self.residual_dim)
            # A reset local head must contribute approximately zero even when the
            # shared residual SIREN was warm-started from a trained global model.
            base_local_values = self.residual_field(
                flat_coordinates,
                torch.zeros_like(flat_scale),
                torch.zeros_like(flat_shift),
                None,
            ).reshape(batch, local_count, queries, self.residual_dim)
            local_values = modulated_local_values - base_local_values
        else:
            part_values = []
            for name, part_slice in LOCAL_PART_SLICES.items():
                field = self.part_local_fields[name]
                output_dim = part_slice.stop - part_slice.start
                part_bias = flat_bias[:, part_slice]
                modulated = field(
                    flat_coordinates,
                    flat_scale,
                    flat_shift,
                    part_bias,
                ).reshape(batch, local_count, queries, output_dim)
                base = field(
                    flat_coordinates,
                    torch.zeros_like(flat_scale),
                    torch.zeros_like(flat_shift),
                    None,
                ).reshape(batch, local_count, queries, output_dim)
                part_values.append(modulated - base)
            local_values = torch.cat(part_values, dim=-1)
        if trajectory.local_part_gates is not None:
            local_values = local_values * _articulator_gate_vector(
                trajectory.local_part_gates
            )[:, :, None, :]
        raw_weights = torch.exp(-0.5 * local_tau.square())
        raw_weights = raw_weights * trajectory.local_mask[:, :, None].to(
            raw_weights.dtype
        )
        weight_sum = raw_weights.sum(dim=1, keepdim=True)
        # Keeping epsilon in the denominator gives the local mixture absolute
        # support: it fades to zero when a query is far from every local center.
        weights = raw_weights / (weight_sum + self.local_window_epsilon)
        coverage = weights.sum(dim=1)
        output = (local_values * weights.unsqueeze(-1)).sum(dim=1)
        query_part_gates = (
            torch.einsum("bmq,bmp->bqp", weights, trajectory.local_part_gates)
            if trajectory.local_part_gates is not None
            else tau.new_zeros(batch, queries, 4)
        )
        return output, weights.transpose(1, 2), coverage, query_part_gates

    def query_trajectory(
        self,
        trajectory: TrajectoryInstance,
        query_times: torch.Tensor,
        time_domain: str = "normalized",
        query_mask: torch.Tensor | None = None,
        return_details: bool = False,
        include_global_residual: bool = True,
        include_local_residual: bool = True,
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
        (
            local_residual,
            local_weights,
            local_coverage,
            local_part_gates,
        ) = self._local_residual(trajectory, tau)
        gates = _articulator_gate_vector(trajectory.articulator_gates)
        amplitude = self.residual_amplitude.to(dtype=global_residual.dtype)
        global_axis = amplitude * global_residual * gates[:, None, :]
        local_axis = amplitude * local_residual
        if trajectory.local_part_gates is None:
            local_axis = local_axis * gates[:, None, :]
        if not include_global_residual:
            global_axis = torch.zeros_like(global_axis)
        if not include_local_residual:
            local_axis = torch.zeros_like(local_axis)
        residual_axis = global_axis + local_axis

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
            global_axis = global_axis * mask_float
            local_axis = local_axis * mask_float
            local_weights = local_weights * mask_float
            local_coverage = local_coverage * query_mask.to(
                device=local_coverage.device, dtype=local_coverage.dtype
            )
            local_part_gates = local_part_gates * mask_float[..., :1]
        if not return_details:
            return prediction
        return {
            "prediction": prediction,
            "prior": prior_raw,
            "correction_axis": residual_axis,
            "global_correction_axis": global_axis,
            "local_correction_axis": local_axis,
            "tau": tau,
            "local_weights": local_weights,
            "local_coverage": local_coverage,
            "local_part_gates": local_part_gates,
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


class DualModeContinuousTrajectoryField(ContinuousTrajectoryField):
    """Text-complete trajectory field with optional gated word-prior guidance."""

    model_type = "dual_mode_continuous_trajectory_field"

    def __init__(
        self,
        *args,
        temporal_slot_count: int = 16,
        temporal_slot_layers: int = 2,
        temporal_slot_heads: int = 8,
        context_fps: float = 20.0,
        **kwargs,
    ):
        # Build the decoder exactly as v1, then replace only its conditioning
        # hypernetwork. This keeps the v1 motion representation and query path
        # byte-for-byte compatible while giving v2 a separate checkpoint graph.
        super().__init__(*args, **kwargs)
        legacy = self.hypernetwork
        duration_head = legacy.duration_head
        self.hypernetwork = DualModeTrajectoryHypernetwork(
            text_dim=legacy.text_dim,
            pose_dim=legacy.pose_dim,
            retrieval_dim=legacy.retrieval_dim,
            context_hidden_dim=legacy.context_hidden_dim,
            context_layers=len(legacy.context_blocks),
            field_hidden_dim=legacy.field_hidden_dim,
            field_depth=legacy.field_depth,
            residual_dim=legacy.residual_dim,
            max_local_fields=legacy.max_local_fields,
            frames_per_local_field=legacy.frames_per_local_field,
            minimum_local_width=legacy.minimum_local_width,
            maximum_local_width=legacy.maximum_local_width,
            time_dependent_local_gates=legacy.time_dependent_local_gates,
            temporal_slot_count=temporal_slot_count,
            temporal_slot_layers=temporal_slot_layers,
            temporal_slot_heads=temporal_slot_heads,
            context_fps=context_fps,
            initial_duration_seconds=float(
                duration_head.net[-1].bias.detach().exp().mean().item()
            ),
            minimum_duration_seconds=duration_head.minimum_seconds,
            maximum_duration_seconds=duration_head.maximum_seconds,
            dropout=float(legacy.global_context[3].p),
        )

    def encode_trajectory(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        word_prior_context: torch.Tensor | None = None,
        word_prior_mask: torch.Tensor | None = None,
        word_prior_features: torch.Tensor | None = None,
        word_prior_available: torch.Tensor | None = None,
    ) -> TrajectoryInstance:
        return self.hypernetwork(
            text_tokens=text_tokens,
            text_mask=text_mask,
            word_prior_context=word_prior_context,
            word_prior_mask=word_prior_mask,
            word_prior_features=word_prior_features,
            word_prior_available=word_prior_available,
        )

    def query_trajectory(self, *args, **kwargs):
        output = super().query_trajectory(*args, **kwargs)
        if isinstance(output, dict) and "prior" in output:
            output["coarse"] = output.pop("prior")
        return output

    def forward(
        self,
        text_tokens: torch.Tensor,
        query_times: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        time_domain: str = "normalized",
        query_mask: torch.Tensor | None = None,
        *,
        word_prior_context: torch.Tensor | None = None,
        word_prior_mask: torch.Tensor | None = None,
        word_prior_features: torch.Tensor | None = None,
        word_prior_available: torch.Tensor | None = None,
    ):
        trajectory = self.encode_trajectory(
            text_tokens=text_tokens,
            text_mask=text_mask,
            word_prior_context=word_prior_context,
            word_prior_mask=word_prior_mask,
            word_prior_features=word_prior_features,
            word_prior_available=word_prior_available,
        )
        return self.query_trajectory(
            trajectory,
            query_times,
            time_domain=time_domain,
            query_mask=query_mask,
            return_details=True,
        )


class SentenceMemoryContinuousTrajectoryField(DualModeContinuousTrajectoryField):
    """V3 trajectory field with optional top-K sentence-motion memory."""

    model_type = "sentence_memory_continuous_trajectory_field"

    def __init__(
        self,
        *args,
        sentence_motion_dim: int = 256,
        sentence_key_dim: int | None = None,
        sentence_attention_layers: int = 2,
        sentence_attention_heads: int = 8,
        sentence_score_temperature: float = 0.10,
        sentence_duration_weight: float = 0.10,
        sentence_retrieval_prior_scale: float = 1.0,
        sentence_gate_initial_bias: float = -2.2,
        **kwargs,
    ):
        # Build v2 first and copy its state into the extended hypernetwork. This
        # keeps all inherited state-dict names and makes a disabled v3 memory
        # branch execute the exact v2 path.
        super().__init__(*args, **kwargs)
        legacy = self.hypernetwork
        duration_head = legacy.duration_head
        extended = SentenceMemoryTrajectoryHypernetwork(
            text_dim=legacy.text_dim,
            pose_dim=legacy.pose_dim,
            retrieval_dim=legacy.retrieval_dim,
            context_hidden_dim=legacy.context_hidden_dim,
            context_layers=len(legacy.word_encoder.context_blocks),
            field_hidden_dim=legacy.field_hidden_dim,
            field_depth=legacy.field_depth,
            residual_dim=legacy.residual_dim,
            max_local_fields=legacy.max_local_fields,
            frames_per_local_field=legacy.frames_per_local_field,
            minimum_local_width=legacy.minimum_local_width,
            maximum_local_width=legacy.maximum_local_width,
            time_dependent_local_gates=legacy.time_dependent_local_gates,
            temporal_slot_count=legacy.temporal_slot_count,
            temporal_slot_layers=len(legacy.text_planner.decoder.layers),
            temporal_slot_heads=legacy.text_planner.decoder.layers[0].self_attn.num_heads,
            context_fps=legacy.context_fps,
            initial_duration_seconds=float(
                duration_head.net[-1].bias.detach().exp().mean().item()
            ),
            minimum_duration_seconds=duration_head.minimum_seconds,
            maximum_duration_seconds=duration_head.maximum_seconds,
            dropout=float(legacy.global_context[3].p),
            sentence_motion_dim=int(sentence_motion_dim),
            sentence_key_dim=sentence_key_dim,
            sentence_attention_layers=int(sentence_attention_layers),
            sentence_attention_heads=int(sentence_attention_heads),
            sentence_score_temperature=float(sentence_score_temperature),
            sentence_duration_weight=float(sentence_duration_weight),
            sentence_retrieval_prior_scale=float(sentence_retrieval_prior_scale),
            sentence_gate_initial_bias=float(sentence_gate_initial_bias),
        )
        incompatible = extended.load_state_dict(legacy.state_dict(), strict=False)
        if incompatible.unexpected_keys or any(
            not name.startswith("sentence_memory_") for name in incompatible.missing_keys
        ):
            raise RuntimeError(
                "Internal v2-to-v3 hypernetwork migration failed: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.hypernetwork = extended

    def encode_trajectory(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        word_prior_context: torch.Tensor | None = None,
        word_prior_mask: torch.Tensor | None = None,
        word_prior_features: torch.Tensor | None = None,
        word_prior_available: torch.Tensor | None = None,
        sentence_motion_tokens: torch.Tensor | None = None,
        sentence_motion_mask: torch.Tensor | None = None,
        sentence_motion_tau: torch.Tensor | None = None,
        sentence_text_keys: torch.Tensor | None = None,
        sentence_scores: torch.Tensor | None = None,
        sentence_durations: torch.Tensor | None = None,
        sentence_candidate_mask: torch.Tensor | None = None,
        sentence_part_validity: torch.Tensor | None = None,
        sentence_memory_available: torch.Tensor | None = None,
    ) -> TrajectoryInstance:
        return self.hypernetwork(
            text_tokens=text_tokens,
            text_mask=text_mask,
            word_prior_context=word_prior_context,
            word_prior_mask=word_prior_mask,
            word_prior_features=word_prior_features,
            word_prior_available=word_prior_available,
            sentence_motion_tokens=sentence_motion_tokens,
            sentence_motion_mask=sentence_motion_mask,
            sentence_motion_tau=sentence_motion_tau,
            sentence_text_keys=sentence_text_keys,
            sentence_scores=sentence_scores,
            sentence_durations=sentence_durations,
            sentence_candidate_mask=sentence_candidate_mask,
            sentence_part_validity=sentence_part_validity,
            sentence_memory_available=sentence_memory_available,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        query_times: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        time_domain: str = "normalized",
        query_mask: torch.Tensor | None = None,
        *,
        word_prior_context: torch.Tensor | None = None,
        word_prior_mask: torch.Tensor | None = None,
        word_prior_features: torch.Tensor | None = None,
        word_prior_available: torch.Tensor | None = None,
        sentence_motion_tokens: torch.Tensor | None = None,
        sentence_motion_mask: torch.Tensor | None = None,
        sentence_motion_tau: torch.Tensor | None = None,
        sentence_text_keys: torch.Tensor | None = None,
        sentence_scores: torch.Tensor | None = None,
        sentence_durations: torch.Tensor | None = None,
        sentence_candidate_mask: torch.Tensor | None = None,
        sentence_part_validity: torch.Tensor | None = None,
        sentence_memory_available: torch.Tensor | None = None,
    ):
        trajectory = self.encode_trajectory(
            text_tokens=text_tokens,
            text_mask=text_mask,
            word_prior_context=word_prior_context,
            word_prior_mask=word_prior_mask,
            word_prior_features=word_prior_features,
            word_prior_available=word_prior_available,
            sentence_motion_tokens=sentence_motion_tokens,
            sentence_motion_mask=sentence_motion_mask,
            sentence_motion_tau=sentence_motion_tau,
            sentence_text_keys=sentence_text_keys,
            sentence_scores=sentence_scores,
            sentence_durations=sentence_durations,
            sentence_candidate_mask=sentence_candidate_mask,
            sentence_part_validity=sentence_part_validity,
            sentence_memory_available=sentence_memory_available,
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
    model_type = str(
        model_cfg.get("type", "continuous_trajectory_field")
    ).lower()
    allowed = {
        "continuous_trajectory_field",
        "dual_mode_continuous_trajectory_field",
        "sentence_memory_continuous_trajectory_field",
    }
    if model_type not in allowed:
        raise ValueError(f"Unsupported continuous trajectory model type {model_type!r}")
    common = dict(
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
        local_center_mode=str(model_cfg.get("local_center_mode", "retrieval_guided")),
        use_retrieval_guidance=bool(model_cfg.get("use_retrieval_guidance", True)),
        local_window_epsilon=float(model_cfg.get("local_window_epsilon", 1e-4)),
        part_specific_local_experts=bool(
            model_cfg.get("part_specific_local_experts", False)
        ),
        time_dependent_local_gates=bool(
            model_cfg.get("time_dependent_local_gates", False)
        ),
        local_body_omega0_first=float(
            model_cfg.get("local_body_omega0_first", 15.0)
        ),
        local_hand_omega0_first=float(
            model_cfg.get("local_hand_omega0_first", 30.0)
        ),
        local_face_omega0_first=float(
            model_cfg.get("local_face_omega0_first", 20.0)
        ),
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
    if model_type in {
        "dual_mode_continuous_trajectory_field",
        "sentence_memory_continuous_trajectory_field",
    }:
        conditioning_cfg = cfg.get("conditioning", {})
        common.update(
            local_center_mode="uniform",
            use_retrieval_guidance=False,
            temporal_slot_count=int(conditioning_cfg.get("temporal_slot_count", 16)),
            temporal_slot_layers=int(conditioning_cfg.get("temporal_slot_layers", 2)),
            temporal_slot_heads=int(conditioning_cfg.get("temporal_slot_heads", 8)),
            context_fps=float(conditioning_cfg.get("context_fps", 20.0)),
        )
        if model_type == "sentence_memory_continuous_trajectory_field":
            sentence_cfg = cfg.get("sentence_memory", {})
            common.update(
                sentence_motion_dim=int(sentence_cfg.get("motion_dim", 256)),
                sentence_key_dim=(
                    int(sentence_cfg["key_dim"])
                    if sentence_cfg.get("key_dim") is not None
                    else None
                ),
                sentence_attention_layers=int(
                    sentence_cfg.get("attention_layers", 2)
                ),
                sentence_attention_heads=int(
                    sentence_cfg.get("attention_heads", 8)
                ),
                sentence_score_temperature=float(
                    sentence_cfg.get("score_temperature", 0.10)
                ),
                sentence_duration_weight=float(
                    sentence_cfg.get("duration_weight", 0.10)
                ),
                sentence_retrieval_prior_scale=float(
                    sentence_cfg.get("retrieval_prior_scale", 1.0)
                ),
                sentence_gate_initial_bias=float(
                    sentence_cfg.get("gate_initial_bias", -2.2)
                ),
            )
            return SentenceMemoryContinuousTrajectoryField(**common)
        return DualModeContinuousTrajectoryField(**common)
    return ContinuousTrajectoryField(**common)
