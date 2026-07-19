from __future__ import annotations

import torch

from NIAF.continuous_sign_field.losses import masked_feature_l1, masked_feature_mse


def frame_index_mask(mask, stride):
    stride = max(int(stride), 1)
    frames = torch.arange(mask.shape[1], device=mask.device)
    return mask & (frames.view(1, -1) % stride == 0)


def build_support_query_masks(
    valid_mask,
    anchor_mask=None,
    support_mode="stride",
    support_stride=8,
):
    valid_mask = valid_mask.bool()
    if anchor_mask is None:
        anchor_mask = torch.zeros_like(valid_mask)
    else:
        anchor_mask = anchor_mask.to(device=valid_mask.device, dtype=torch.bool) & valid_mask

    mode = str(support_mode or "stride")
    if mode == "anchors_only":
        support = anchor_mask
    elif mode in {"stride", "anchors_plus_stride"}:
        support = anchor_mask | frame_index_mask(valid_mask, support_stride)
    elif mode == "none":
        support = torch.zeros_like(valid_mask)
    else:
        raise ValueError(f"Unsupported support_mode={support_mode!r}")

    support = support & valid_mask
    query = valid_mask & ~support
    for batch_idx in range(valid_mask.shape[0]):
        if valid_mask[batch_idx].any() and not support[batch_idx].any():
            first = torch.nonzero(valid_mask[batch_idx], as_tuple=False)[0, 0]
            support[batch_idx, first] = True
            query[batch_idx, first] = False
        if valid_mask[batch_idx].any() and not query[batch_idx].any():
            query[batch_idx] = valid_mask[batch_idx]
            query[batch_idx] &= ~support[batch_idx]
            if not query[batch_idx].any():
                query[batch_idx] = valid_mask[batch_idx]
                support[batch_idx] = False
                first = torch.nonzero(valid_mask[batch_idx], as_tuple=False)[0, 0]
                support[batch_idx, first] = True
                query[batch_idx, first] = False
    return support, query


def temporal_difference(values, order=1):
    diff = values
    for _ in range(int(order)):
        if diff.shape[1] < 2:
            return diff.new_zeros(diff.shape[0], 0, *diff.shape[2:])
        diff = diff[:, 1:] - diff[:, :-1]
    return diff


def temporal_mask(mask, order=1):
    out = mask.bool()
    for _ in range(int(order)):
        if out.shape[1] < 2:
            return out.new_zeros(out.shape[0], 0)
        out = out[:, 1:] & out[:, :-1]
    return out


def masked_residual_loss(
    pred_residual,
    target_residual,
    mask,
    hand_weight=5.0,
    loss_type="l1",
):
    if pred_residual.shape[-1] != 256:
        values = (pred_residual - target_residual) ** 2
        if str(loss_type) != "mse":
            values = torch.abs(pred_residual - target_residual)
        mask_f = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        return (values * mask_f).sum() / mask_f.expand_as(values).sum().clamp_min(1.0)
    if str(loss_type) == "mse":
        return masked_feature_mse(pred_residual, target_residual, mask, hand_weight=hand_weight)
    return masked_feature_l1(pred_residual, target_residual, mask, hand_weight=hand_weight)


def highpass_residual_loss(pred_residual, target_residual, mask, order=2, hand_weight=5.0):
    pred_diff = temporal_difference(pred_residual, order=order)
    target_diff = temporal_difference(target_residual, order=order)
    diff_mask = temporal_mask(mask, order=order)
    if pred_diff.shape[1] == 0:
        return pred_residual.new_tensor(0.0)
    return masked_residual_loss(pred_diff, target_diff, diff_mask, hand_weight=hand_weight, loss_type="l1")


def support_adaptation_loss(
    model,
    code,
    tau,
    scaffold,
    target,
    target_residual,
    support_mask,
    cfg,
):
    loss_cfg = cfg.get("meta_loss", cfg.get("loss", {}))
    hand_weight = float(loss_cfg.get("hand_weight", cfg.get("loss", {}).get("hand_weight", 5.0)))
    pred_residual = model(tau, scaffold, code, mask=scaffold.new_ones(scaffold.shape[:2], dtype=torch.bool))
    pred = scaffold + pred_residual
    residual_loss = masked_residual_loss(
        pred_residual,
        target_residual,
        support_mask,
        hand_weight=hand_weight,
        loss_type=str(loss_cfg.get("inner_residual_loss", "l1")),
    )
    if pred.shape[-1] == 256:
        pose_loss = masked_feature_l1(
            pred,
            target,
            support_mask,
            hand_weight=hand_weight,
        )
    else:
        pose_loss = masked_residual_loss(pred, target, support_mask, hand_weight=hand_weight, loss_type="l1")
    total = (
        float(loss_cfg.get("inner_lambda_residual", 1.0)) * residual_loss
        + float(loss_cfg.get("inner_lambda_pose", 0.25)) * pose_loss
    )
    return total, {
        "inner_loss_residual": residual_loss,
        "inner_loss_pose": pose_loss,
        "inner_loss_total": total,
    }


def adapt_code(
    model,
    initial_code,
    tau,
    scaffold,
    target,
    target_residual,
    support_mask,
    cfg,
):
    meta_cfg = cfg.get("meta", {})
    code = initial_code
    inner_steps = int(meta_cfg.get("inner_steps", 3))
    inner_lr = float(meta_cfg.get("inner_lr", 1e-2))
    first_order = bool(meta_cfg.get("first_order", True))
    last_losses = {}
    for _ in range(max(inner_steps, 0)):
        loss, losses = support_adaptation_loss(
            model,
            code,
            tau,
            scaffold,
            target,
            target_residual,
            support_mask,
            cfg,
        )
        (grad,) = torch.autograd.grad(
            loss,
            code,
            create_graph=not first_order,
            retain_graph=not first_order,
        )
        if first_order:
            grad = grad.detach()
        code = code - inner_lr * grad
        last_losses = losses
    return code, last_losses
