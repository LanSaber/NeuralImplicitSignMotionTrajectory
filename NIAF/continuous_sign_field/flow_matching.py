from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_kernel1d(kernel_size, sigma, device, dtype):
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / max(float(sigma), 1e-6)) ** 2)
    return kernel / kernel.sum().clamp_min(1e-8)


def smooth_temporal(values, mask, kernel_size=9, sigma=2.0):
    if int(kernel_size) <= 1:
        return values * mask.unsqueeze(-1).to(values.dtype)
    batch, frames, dim = values.shape
    kernel = gaussian_kernel1d(kernel_size, sigma, values.device, values.dtype)
    kernel = kernel.view(1, 1, -1).expand(dim, 1, -1)
    x = values.transpose(1, 2)
    smoothed = F.conv1d(x, kernel, padding=kernel.shape[-1] // 2, groups=dim).transpose(1, 2)
    return smoothed * mask.unsqueeze(-1).to(values.dtype)


def smooth_noise_like(target, mask, sigma=0.05, kernel_size=9, smooth_sigma=2.0):
    if float(sigma) <= 0:
        return torch.zeros_like(target)
    noise = torch.randn_like(target) * float(sigma)
    noise = noise * mask.unsqueeze(-1).to(noise.dtype)
    return smooth_temporal(noise, mask, kernel_size=kernel_size, sigma=smooth_sigma)


def sample_bridge(target_residual, mask, noise_sigma=0.05, kernel_size=9, smooth_sigma=2.0):
    source_residual = smooth_noise_like(
        target_residual,
        mask,
        sigma=noise_sigma,
        kernel_size=kernel_size,
        smooth_sigma=smooth_sigma,
    )
    batch = target_residual.shape[0]
    flow_t = torch.rand(batch, 1, 1, device=target_residual.device, dtype=target_residual.dtype)
    residual_t = (1.0 - flow_t) * source_residual + flow_t * target_residual
    target_velocity = target_residual - source_residual
    return residual_t, target_velocity, source_residual, flow_t


def endpoint_from_velocity(residual_t, velocity, flow_t):
    return residual_t + (1.0 - flow_t) * velocity


def heun_integrate(model, scaffold, tau, mask, text_tokens=None, text_mask=None, steps=4, init_residual=None):
    steps = max(int(steps), 1)
    residual = torch.zeros_like(scaffold) if init_residual is None else init_residual
    dt = 1.0 / float(steps)
    for step in range(steps):
        t0 = scaffold.new_full((scaffold.shape[0], 1, 1), step * dt)
        t1 = scaffold.new_full((scaffold.shape[0], 1, 1), (step + 1) * dt)
        v0 = model(residual, scaffold, tau, t0, mask=mask, text_tokens=text_tokens, text_mask=text_mask)
        pred = residual + dt * v0
        v1 = model(pred, scaffold, tau, t1, mask=mask, text_tokens=text_tokens, text_mask=text_mask)
        residual = residual + 0.5 * dt * (v0 + v1)
        residual = residual * mask.unsqueeze(-1).to(residual.dtype)
    return residual
