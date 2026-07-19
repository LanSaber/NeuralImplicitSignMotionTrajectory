from __future__ import annotations

import torch
import torch.nn.functional as F


def finite_difference_velocity(z, s):
    z = z.float()
    s = s.float().view(-1, 1)
    if z.shape[0] <= 1:
        return torch.zeros_like(z)
    vel = torch.zeros_like(z)
    vel[0] = (z[1] - z[0]) / (s[1] - s[0]).clamp_min(1e-8)
    vel[-1] = (z[-1] - z[-2]) / (s[-1] - s[-2]).clamp_min(1e-8)
    if z.shape[0] > 2:
        vel[1:-1] = (z[2:] - z[:-2]) / (s[2:] - s[:-2]).clamp_min(1e-8)
    return vel


def finite_difference_acceleration(z, s):
    if z.shape[0] <= 2:
        return torch.zeros_like(z)
    vel = finite_difference_velocity(z, s)
    return finite_difference_velocity(vel, s)


def directional_derivatives(field, s, order=3):
    s = s.detach().clone().requires_grad_(True)
    ones = torch.ones_like(s)
    z = field(s)
    outputs = [z]

    if order >= 1:
        def f0(inp):
            return field(inp)

        _, dz = torch.autograd.functional.jvp(f0, (s,), (ones,), create_graph=True, strict=False)
        outputs.append(dz)

    if order >= 2:
        def f1(inp):
            _, cur = torch.autograd.functional.jvp(
                f0,
                (inp,),
                (torch.ones_like(inp),),
                create_graph=True,
                strict=False,
            )
            return cur

        _, ddz = torch.autograd.functional.jvp(f1, (s,), (ones,), create_graph=True, strict=False)
        outputs.append(ddz)

    if order >= 3:
        def f2(inp):
            _, cur = torch.autograd.functional.jvp(
                f1,
                (inp,),
                (torch.ones_like(inp),),
                create_graph=True,
                strict=False,
            )
            return cur

        _, dddz = torch.autograd.functional.jvp(f2, (s,), (ones,), create_graph=True, strict=False)
        outputs.append(dddz)

    while len(outputs) < 4:
        outputs.append(torch.zeros_like(z))
    return outputs[:4]


def masked_smooth_l1(pred, target, mask=None):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if mask is None:
        return loss.mean()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    loss = loss * mask.to(device=loss.device, dtype=loss.dtype)
    return loss.sum() / mask.sum().clamp_min(1.0)
