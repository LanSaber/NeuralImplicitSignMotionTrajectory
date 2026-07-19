from __future__ import annotations

import torch
import torch.nn.functional as F

from flow.smplx_features import COMPACT6D_EXPRESSION, feature_weight_vector
from NIAF.oracle_smplx_field.geometry.rotation import geodesic_loss


def weighted_l1(pred, target, weights=None):
    diff = torch.abs(pred - target)
    if weights is not None:
        diff = diff * weights.to(device=diff.device, dtype=diff.dtype)
    return diff.mean()


def feature_loss(pred, target, hand_weight=3.0):
    weights = feature_weight_vector(hand_weight=hand_weight, device=pred.device, rotation_rep="rot6d")
    weights[COMPACT6D_EXPRESSION] = 0.0
    return weighted_l1(pred, target, weights=weights)


def expression_loss(pred, target):
    return F.l1_loss(pred[..., COMPACT6D_EXPRESSION], target[..., COMPACT6D_EXPRESSION])


def joint_l1_loss(pred_parts, target_parts, part="wholebody", hand_weight=3.0):
    pred = pred_parts[part]
    target = target_parts[part].to(device=pred.device, dtype=pred.dtype)
    diff = torch.abs(pred - target).sum(dim=-1)
    if part == "wholebody":
        body_count = pred_parts["body"].shape[1]
        lhand_count = pred_parts["lhand"].shape[1]
        rhand_count = pred_parts["rhand"].shape[1]
        weights = pred.new_ones(body_count + lhand_count + rhand_count)
        weights[body_count:] = float(hand_weight)
        diff = diff * weights.view(1, -1)
    return diff.mean()


LOSS_SCHEDULES = {
    "S1": {
        "lambda_6d": 1.0,
        "lambda_geo": 0.0,
        "lambda_joint": 10.0,
        "lambda_expr": 1.0,
        "lambda_res": 1.0e-4,
    },
    "S2": {
        "lambda_6d": 1.0,
        "lambda_geo": 0.1,
        "lambda_joint": 10.0,
        "lambda_expr": 1.0,
        "lambda_res": 1.0e-4,
    },
    "S3": {
        "lambda_6d": 1.0,
        "lambda_geo": 0.1,
        "lambda_joint": 10.0,
        "lambda_expr": 1.0,
        "lambda_res": 1.0e-4,
    },
    "S4": {
        "lambda_6d": 1.0,
        "lambda_geo": 0.0,
        "lambda_joint": 10.0,
        "lambda_expr": 1.0,
        "lambda_res": 1.0e-6,
        "lambda_dense_joint_acc": 0.1,
        "lambda_dense_joint_jerk": 0.05,
    },
}


def pose_field_loss(pred, target, fk=None, target_parts=None, schedule=None, hand_weight=3.0):
    schedule = dict(schedule or LOSS_SCHEDULES["S1"])
    losses = {}
    total = pred.new_tensor(0.0)
    if schedule.get("lambda_6d", 0.0) > 0:
        losses["loss_6d"] = feature_loss(pred, target, hand_weight=hand_weight)
        total = total + float(schedule["lambda_6d"]) * losses["loss_6d"]
    if schedule.get("lambda_geo", 0.0) > 0:
        losses["loss_geo"] = geodesic_loss(pred, target)
        total = total + float(schedule["lambda_geo"]) * losses["loss_geo"]
    if schedule.get("lambda_expr", 0.0) > 0:
        losses["loss_expr"] = expression_loss(pred, target)
        total = total + float(schedule["lambda_expr"]) * losses["loss_expr"]
    if schedule.get("lambda_joint", 0.0) > 0:
        if fk is None or target_parts is None:
            raise ValueError("Joint loss requires fk and target_parts")
        pred_parts = fk.parts_from_rot6d(pred)
        losses["loss_joint"] = joint_l1_loss(pred_parts, target_parts, hand_weight=hand_weight)
        total = total + float(schedule["lambda_joint"]) * losses["loss_joint"]
    losses["loss_total"] = total
    return total, losses


def temporal_difference(values, order):
    diff = values
    for _ in range(int(order)):
        if diff.shape[0] < 2:
            return diff.new_zeros((0,) + diff.shape[1:])
        diff = diff[1:] - diff[:-1]
    return diff


def temporal_joint_loss(joints, order=2, hand_weight=3.0, body_count=12):
    diff = temporal_difference(joints, order=order)
    if diff.numel() == 0:
        return joints.new_tensor(0.0)
    value = torch.abs(diff).sum(dim=-1)
    if value.shape[1] > body_count:
        weights = value.new_ones(value.shape[1])
        weights[body_count:] = float(hand_weight)
        value = value * weights.view(1, -1)
    return value.mean()


def dense_physical_loss(pred, fk, schedule=None, hand_weight=3.0):
    schedule = dict(schedule or {})
    losses = {}
    total = pred.new_tensor(0.0)
    if schedule.get("lambda_dense_joint_acc", 0.0) <= 0 and schedule.get("lambda_dense_joint_jerk", 0.0) <= 0:
        return total, losses
    pred_parts = fk.parts_from_rot6d(pred)
    joints = pred_parts["wholebody"]
    if schedule.get("lambda_dense_joint_acc", 0.0) > 0:
        losses["loss_dense_joint_acc"] = temporal_joint_loss(joints, order=2, hand_weight=hand_weight)
        total = total + float(schedule["lambda_dense_joint_acc"]) * losses["loss_dense_joint_acc"]
    if schedule.get("lambda_dense_joint_jerk", 0.0) > 0:
        losses["loss_dense_joint_jerk"] = temporal_joint_loss(joints, order=3, hand_weight=hand_weight)
        total = total + float(schedule["lambda_dense_joint_jerk"]) * losses["loss_dense_joint_jerk"]
    losses["loss_dense_physical"] = total
    return total, losses
