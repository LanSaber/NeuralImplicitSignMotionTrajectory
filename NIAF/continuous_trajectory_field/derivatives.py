from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def _directional_derivative(function: TensorFunction) -> TensorFunction:
    def derivative(coordinates: torch.Tensor):
        return torch.autograd.functional.jvp(
            function,
            (coordinates,),
            (torch.ones_like(coordinates),),
            create_graph=True,
            strict=False,
        )[1]

    return derivative


def normalized_derivatives(
    function: TensorFunction,
    tau: torch.Tensor,
    max_order: int = 3,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Evaluate a pointwise function and derivatives with respect to tau.

    The all-ones JVP returns each query's derivative because trajectory queries
    are independent across the query dimension.
    """

    max_order = max(int(max_order), 0)
    value = function(tau)
    derivatives = {}
    derivative_function = function
    for order in range(1, max_order + 1):
        derivative_function = _directional_derivative(derivative_function)
        derivatives[order] = derivative_function(tau)
    return value, derivatives


def physical_derivative_scale(
    duration_seconds: torch.Tensor,
    order: int,
    output_ndim: int,
) -> torch.Tensor:
    duration_seconds = duration_seconds.clamp_min(1e-4)
    scale = (2.0 / duration_seconds).pow(int(order))
    return scale.view(scale.shape[0], *([1] * (int(output_ndim) - 1)))


def physical_derivatives(
    function: TensorFunction,
    tau: torch.Tensor,
    duration_seconds: torch.Tensor,
    max_order: int = 3,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    value, normalized = normalized_derivatives(function, tau, max_order=max_order)
    output = {
        order: derivative
        * physical_derivative_scale(duration_seconds, order, derivative.ndim)
        for order, derivative in normalized.items()
    }
    return value, output


def gaussian_smooth_sequence(values: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5):
    """Smooth `[T,...]` values while retaining their original shape."""

    if values.shape[0] <= 2 or int(kernel_size) <= 1:
        return values
    kernel_size = min(int(kernel_size), int(values.shape[0]))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 1:
        return values
    radius = kernel_size // 2
    coordinate = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-0.5 * (coordinate / max(float(sigma), 1e-4)).square())
    kernel = kernel / kernel.sum()
    flat = values.reshape(values.shape[0], -1).transpose(0, 1).unsqueeze(0)
    flat = F.pad(flat, (radius, radius), mode="replicate")
    smoothed = F.conv1d(flat, kernel.view(1, 1, -1).expand(flat.shape[1], 1, -1), groups=flat.shape[1])
    return smoothed.squeeze(0).transpose(0, 1).reshape_as(values)


def finite_physical_derivatives(
    values: torch.Tensor,
    lengths: torch.Tensor,
    duration_seconds: torch.Tensor,
    max_order: int = 3,
    smooth_kernel: int = 7,
    smooth_sigma: float = 1.5,
) -> Dict[int, torch.Tensor]:
    """Denoised finite-difference derivative targets for padded trajectories."""

    output = {
        order: torch.zeros_like(values) for order in range(1, int(max_order) + 1)
    }
    for batch_index, length_value in enumerate(lengths.detach().cpu().tolist()):
        length = int(length_value)
        if length <= 1:
            continue
        sequence = gaussian_smooth_sequence(
            values[batch_index, :length],
            kernel_size=smooth_kernel,
            sigma=smooth_sigma,
        )
        duration = duration_seconds[batch_index].to(device=values.device, dtype=values.dtype)
        dt = float((duration / max(length - 1, 1)).detach().cpu().item())
        derivative = sequence
        for order in range(1, int(max_order) + 1):
            if derivative.shape[0] <= 1:
                derivative = torch.zeros_like(sequence)
            else:
                derivative = torch.gradient(derivative, spacing=dt, dim=0)[0]
            output[order][batch_index, :length] = derivative
    return output


def sample_padded_sequence(
    values: torch.Tensor,
    tau: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Linearly sample padded `[B,T,...]` values at normalized `[-1,1]` times."""

    if tau.ndim == 3 and tau.shape[-1] == 1:
        tau = tau.squeeze(-1)
    if tau.ndim != 2 or tau.shape[0] != values.shape[0]:
        raise ValueError("tau must have shape [B,K]")
    batch, queries = tau.shape
    coordinate = 0.5 * (tau.clamp(-1.0, 1.0) + 1.0)
    coordinate = coordinate * (lengths.to(values.device, values.dtype) - 1).clamp_min(0)[:, None]
    lower = torch.floor(coordinate).long()
    upper = torch.minimum(lower + 1, (lengths.to(values.device) - 1).clamp_min(0)[:, None])
    fraction = coordinate - lower.to(coordinate.dtype)
    flat_dim = int(values[0, 0].numel())
    flat_values = values.reshape(values.shape[0], values.shape[1], flat_dim)
    lower_values = torch.gather(
        flat_values,
        1,
        lower.unsqueeze(-1).expand(batch, queries, flat_dim),
    )
    upper_values = torch.gather(
        flat_values,
        1,
        upper.unsqueeze(-1).expand(batch, queries, flat_dim),
    )
    sampled = lower_values + fraction.unsqueeze(-1) * (upper_values - lower_values)
    return sampled.reshape(batch, queries, *values.shape[2:])
