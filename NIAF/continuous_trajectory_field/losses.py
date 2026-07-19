from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from NIAF.continuous_sign_field.losses import masked_feature_l1, masked_mean
from NIAF.continuous_trajectory_field.derivatives import (
    finite_physical_derivatives,
    physical_derivatives,
    sample_padded_sequence,
)
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import (
    target_tangent_correction,
)


def masked_tangent_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hand_weight: float = 5.0,
):
    weights = prediction.new_ones(prediction.shape[-1])
    weights[30:120] = float(hand_weight)
    difference = torch.abs(prediction - target) * weights
    return masked_mean(difference, mask)


def duration_regression_loss(
    predicted_log_seconds: torch.Tensor,
    target_seconds: torch.Tensor,
    beta: float = 0.15,
):
    target_log = torch.log(target_seconds.clamp_min(1e-4))
    return F.smooth_l1_loss(predicted_log_seconds, target_log, beta=float(beta))


def prior_and_residual_losses(
    outputs,
    adapter_context: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hand_weight: float = 5.0,
):
    prior = outputs["prior"]
    correction = outputs["correction_axis"]
    target_correction = target_tangent_correction(prior.detach(), target, mask)
    return {
        "loss_prior": masked_feature_l1(
            prior, adapter_context, mask, hand_weight=hand_weight
        ),
        "loss_residual": masked_tangent_l1(
            correction, target_correction, mask, hand_weight=hand_weight
        ),
    }


def _weighted_joint_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    body_count: int,
    hand_weight: float,
):
    difference = torch.abs(prediction - target).sum(dim=-1)
    weights = difference.new_ones(difference.shape[-1])
    if difference.shape[-1] > int(body_count):
        weights[int(body_count) :] = float(hand_weight)
    return (difference * weights.view(1, 1, -1)).mean()


def _jerk_energy(values: torch.Tensor):
    return torch.linalg.vector_norm(values, dim=-1).mean()


def random_query_times(
    batch_size: int,
    count: int,
    device,
    dtype,
    randomize: bool,
):
    count = max(int(count), 2)
    base = torch.linspace(-0.95, 0.95, count, device=device, dtype=dtype)
    tau = base.unsqueeze(0).expand(int(batch_size), -1).clone()
    if randomize and count > 2:
        spacing = 1.9 / max(count - 1, 1)
        jitter = (torch.rand_like(tau[:, 1:-1]) - 0.5) * spacing
        tau[:, 1:-1] = (tau[:, 1:-1] + jitter).clamp(-0.95, 0.95)
        tau, _ = torch.sort(tau, dim=1)
    return tau


def analytic_fk_dynamics_losses(
    model,
    trajectory,
    fk,
    target_parts: Dict[str, torch.Tensor],
    lengths: torch.Tensor,
    duration_seconds: torch.Tensor,
    weights=None,
    query_count: int = 8,
    hand_weight: float = 5.0,
    smooth_kernel: int = 7,
    smooth_sigma: float = 1.5,
    jerk_target_ratio: float = 0.75,
    randomize_queries: bool = True,
):
    """Match analytic FK derivatives to denoised physical-time targets."""

    weights = dict(weights or {})
    order_weights = {
        1: float(weights.get("lambda_analytic_fk_vel", 0.0)),
        2: float(weights.get("lambda_analytic_fk_acc", 0.0)),
        3: float(weights.get("lambda_analytic_fk_jerk", 0.0)),
    }
    jerk_reg_weight = float(weights.get("lambda_analytic_fk_jerk_reg", 0.0))
    max_order = max(
        [order for order, weight in order_weights.items() if weight > 0]
        + ([3] if jerk_reg_weight > 0 else [0])
    )
    if max_order == 0:
        zero = trajectory.duration_seconds.new_tensor(0.0)
        return zero, {}

    target_whole = target_parts["wholebody"].to(
        device=trajectory.device, dtype=trajectory.dtype
    )
    tau = random_query_times(
        trajectory.batch_size,
        query_count,
        trajectory.device,
        trajectory.dtype,
        randomize=randomize_queries,
    )
    query_count = tau.shape[1]

    def fk_function(query_tau):
        compact = model.query_trajectory(
            trajectory,
            query_tau,
            time_domain="normalized",
            return_details=False,
        )
        flat = compact.reshape(-1, compact.shape[-1])
        parts = fk.parts_from_rot6d(flat)
        return parts["wholebody"].reshape(
            trajectory.batch_size,
            query_count,
            parts["wholebody"].shape[1],
            3,
        )

    _positions, prediction_derivatives = physical_derivatives(
        fk_function,
        tau,
        duration_seconds,
        max_order=max_order,
    )
    target_derivatives = finite_physical_derivatives(
        target_whole,
        lengths,
        duration_seconds,
        max_order=max_order,
        smooth_kernel=smooth_kernel,
        smooth_sigma=smooth_sigma,
    )
    body_count = int(target_parts["body"].shape[2])
    losses = {}
    total = trajectory.duration_seconds.new_tensor(0.0)
    names = {1: "vel", 2: "acc", 3: "jerk"}
    sampled_targets = {}
    for order in range(1, max_order + 1):
        sampled_targets[order] = sample_padded_sequence(
            target_derivatives[order], tau, lengths
        )
        if order_weights.get(order, 0.0) > 0:
            name = f"loss_analytic_fk_{names[order]}"
            losses[name] = _weighted_joint_l1(
                prediction_derivatives[order],
                sampled_targets[order],
                body_count=body_count,
                hand_weight=hand_weight,
            )
            total = total + order_weights[order] * losses[name]

    if jerk_reg_weight > 0:
        predicted_energy = _jerk_energy(prediction_derivatives[3])
        target_energy = _jerk_energy(sampled_targets[3]).detach()
        losses["analytic_fk_jerk_energy"] = predicted_energy
        losses["target_fk_jerk_energy"] = target_energy
        losses["loss_analytic_fk_jerk_reg"] = F.relu(
            predicted_energy - float(jerk_target_ratio) * target_energy
        )
        losses["analytic_fk_jerk_ratio"] = predicted_energy / target_energy.clamp_min(1e-8)
        total = total + jerk_reg_weight * losses["loss_analytic_fk_jerk_reg"]
    return total, losses


def local_field_regularization(trajectory):
    if trajectory.num_local_fields == 0 or not bool(trajectory.local_mask.any()):
        zero = trajectory.duration_seconds.new_tensor(0.0)
        return {
            "loss_local_modulation": zero,
            "loss_local_width": zero,
        }
    mask = trajectory.local_mask.to(trajectory.dtype)
    modulation = trajectory.local_scale.square().mean(dim=(-1, -2))
    modulation = modulation + trajectory.local_shift.square().mean(dim=(-1, -2))
    modulation = modulation + trajectory.local_output_bias.square().mean(dim=-1)
    modulation_loss = (modulation * mask).sum() / mask.sum().clamp_min(1.0)
    width_loss = (
        trajectory.local_widths.reciprocal().clamp_max(100.0) * mask
    ).sum() / mask.sum().clamp_min(1.0)
    return {
        "loss_local_modulation": modulation_loss,
        "loss_local_width": width_loss,
    }
