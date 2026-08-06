#!/usr/bin/env python
"""Compute mGPT t2m.py-style DTW-MPJPE metrics for saved flow samples."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from flow.evaluate.ndtw_smplx_keypoints import H2S_FIXED_BETAS, UPPER_BODY_JOINTS
from flow.render import DEFAULT_MODEL_DIR, resolve_device


LEFT_HAND_LAYOUT = (
    ("joint", 20),
    ("joint", 37),
    ("joint", 38),
    ("joint", 39),
    ("vertex", 5361),
    ("joint", 25),
    ("joint", 26),
    ("joint", 27),
    ("vertex", 4933),
    ("joint", 28),
    ("joint", 29),
    ("joint", 30),
    ("vertex", 5058),
    ("joint", 34),
    ("joint", 35),
    ("joint", 36),
    ("vertex", 5169),
    ("joint", 31),
    ("joint", 32),
    ("joint", 33),
    ("vertex", 5286),
)
RIGHT_HAND_LAYOUT = (
    ("joint", 21),
    ("joint", 52),
    ("joint", 53),
    ("joint", 54),
    ("vertex", 8079),
    ("joint", 40),
    ("joint", 41),
    ("joint", 42),
    ("vertex", 7669),
    ("joint", 43),
    ("joint", 44),
    ("joint", 45),
    ("vertex", 7794),
    ("joint", 49),
    ("joint", 50),
    ("joint", 51),
    ("vertex", 7905),
    ("joint", 46),
    ("joint", 47),
    ("joint", 48),
    ("vertex", 8022),
)
PARTS = ("body", "lhand", "rhand", "wholebody")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved SMPL-X samples with mGPT/metrics/t2m.py-style "
            "MPJPE frame costs, upper-body body joints, and 21-keypoint "
            "original hand regressors per hand."
        )
    )
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--sample_glob", default="sample_*.npz")
    parser.add_argument("--sample_key", default="smplx")
    parser.add_argument("--gt_key", default="smplx")
    parser.add_argument("--prior_key", default="coarse_smplx")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gender", default="NEUTRAL")
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument(
        "--betas_mode",
        default="h2s_fixed",
        choices=["from_params", "zero", "h2s_fixed"],
        help="SMPL-X shape source. mGPT H2S uses h2s_fixed.",
    )
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument(
        "--alignment_mode",
        default="default",
        choices=["default", "pa"],
        help=(
            "default matches t2m.py align_idx=0. pa uses partwise similarity "
            "Procrustes alignment inside each frame cost, fitting and scoring "
            "the same keypoint set."
        ),
    )
    return parser.parse_args()


def resolve_betas_override(mode):
    if mode == "from_params":
        return None
    if mode == "zero":
        return np.zeros(10, dtype=np.float32)
    if mode == "h2s_fixed":
        return np.asarray(H2S_FIXED_BETAS, dtype=np.float32)
    raise ValueError(f"Unsupported betas_mode={mode!r}")


def load_smplx(path, key):
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path}: missing key {key!r}; available keys: {data.files}")
        params = data[key].astype(np.float32)
    if params.ndim != 2 or params.shape[1] != 182:
        raise ValueError(f"{path}: {key!r} must have shape [T, 182], got {params.shape}")
    return params


def smplx_to_joints_vertices(smplx_params, model_dir, gender, device, batch_size, betas_override):
    import smplx

    smplx_params = np.asarray(smplx_params, dtype=np.float32)
    if smplx_params.ndim != 2 or smplx_params.shape[1] != 182:
        raise ValueError(f"Expected SMPL-X params with shape [T, 182], got {smplx_params.shape}")

    device = resolve_device(device)
    layer_cache = {}
    joint_chunks = []
    vertex_chunks = []

    def get_layer(cur_batch):
        if cur_batch not in layer_cache:
            layer_cache[cur_batch] = smplx.create(
                str(model_dir),
                model_type="smplx",
                gender=gender,
                use_pca=False,
                use_face_contour=True,
                num_betas=10,
                num_expression_coeffs=10,
                batch_size=cur_batch,
            ).to(device)
        return layer_cache[cur_batch]

    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        for start in range(0, len(smplx_params), batch_size):
            end = min(start + batch_size, len(smplx_params))
            cur = torch.from_numpy(smplx_params[start:end]).to(device)
            if betas_override is None:
                betas = cur[:, 159:169]
            else:
                betas = torch.as_tensor(betas_override, dtype=torch.float32, device=device).view(1, 10)
                betas = betas.expand(end - start, -1)
            layer = get_layer(end - start)
            output = layer(
                global_orient=cur[:, 0:3],
                body_pose=cur[:, 3:66],
                left_hand_pose=cur[:, 66:111],
                right_hand_pose=cur[:, 111:156],
                jaw_pose=cur[:, 156:159],
                betas=betas,
                expression=cur[:, 169:179],
                transl=cur[:, 179:182],
                leye_pose=torch.zeros((end - start, 3), dtype=torch.float32, device=device),
                reye_pose=torch.zeros((end - start, 3), dtype=torch.float32, device=device),
            )
            joint_chunks.append(output.joints.detach().cpu().numpy().astype(np.float32))
            vertex_chunks.append(output.vertices.detach().cpu().numpy().astype(np.float32))
    finally:
        torch.set_grad_enabled(was_grad_enabled)

    return np.concatenate(joint_chunks, axis=0), np.concatenate(vertex_chunks, axis=0)


def hand_from_layout(joints, vertices, layout):
    pieces = []
    for kind, index in layout:
        if kind == "joint":
            pieces.append(joints[:, index : index + 1, :])
        elif kind == "vertex":
            pieces.append(vertices[:, index : index + 1, :])
        else:
            raise ValueError(f"Unsupported hand layout entry kind={kind!r}")
    return np.concatenate(pieces, axis=1).astype(np.float32)


def normalize_first(points):
    return points - points[:, 0:1, :]


def t2m_raw_parts(joints, vertices):
    lhand = hand_from_layout(joints, vertices, LEFT_HAND_LAYOUT)
    rhand = hand_from_layout(joints, vertices, RIGHT_HAND_LAYOUT)
    wholebody = np.concatenate(
        [
            joints[:, UPPER_BODY_JOINTS, :],
            lhand,
            rhand,
        ],
        axis=1,
    )
    return {
        "full_joints": joints.astype(np.float32),
        "body": joints[:, UPPER_BODY_JOINTS, :].astype(np.float32),
        "lhand": lhand.astype(np.float32),
        "rhand": rhand.astype(np.float32),
        "wholebody": wholebody.astype(np.float32),
    }


def t2m_default_parts(joints, vertices):
    raw = t2m_raw_parts(joints, vertices)
    return {
        "body": (raw["body"] - joints[:, 0:1, :]).astype(np.float32),
        "lhand": normalize_first(raw["lhand"]).astype(np.float32),
        "rhand": normalize_first(raw["rhand"]).astype(np.float32),
        "wholebody": normalize_first(raw["wholebody"]).astype(np.float32),
    }


def frame_distance_matrix_mpjpe(a, b):
    diff = a[:, None, :, :].astype(np.float64) - b[None, :, :, :].astype(np.float64)
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def rigid_align(source, target, eps=1e-8):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (source_centered.T @ target_centered) / max(len(source), 1)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        singular_values[-1] *= -1
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    source_var = np.var(source, axis=0).sum()
    scale = 1.0 if source_var <= eps else float(np.sum(singular_values) / source_var)
    translation = target_mean - scale * (rotation @ source_mean)
    return (scale * (rotation @ source.T)).T + translation


def mpjpe(source, target):
    return float(np.linalg.norm(source - target, axis=-1).mean())


def batched_procrustes_mpjpe(pred_align, gt_align, pred_eval=None, gt_eval=None, chunk_size=8192):
    pred_align = np.asarray(pred_align, dtype=np.float64)
    gt_align = np.asarray(gt_align, dtype=np.float64)
    pred_eval = pred_align if pred_eval is None else np.asarray(pred_eval, dtype=np.float64)
    gt_eval = gt_align if gt_eval is None else np.asarray(gt_eval, dtype=np.float64)

    pred_len, gt_len = len(pred_align), len(gt_align)
    dist = np.empty((pred_len, gt_len), dtype=np.float64)
    total_pairs = pred_len * gt_len
    for start in range(0, total_pairs, chunk_size):
        end = min(start + chunk_size, total_pairs)
        flat = np.arange(start, end)
        pred_idx = flat // gt_len
        gt_idx = flat % gt_len

        source = pred_align[pred_idx]
        target = gt_align[gt_idx]
        source_eval = pred_eval[pred_idx]
        target_eval = gt_eval[gt_idx]

        source_mean = source.mean(axis=1)
        target_mean = target.mean(axis=1)
        source_centered = source - source_mean[:, None, :]
        target_centered = target - target_mean[:, None, :]
        covariance = np.einsum("bni,bnj->bij", source_centered, target_centered) / max(source.shape[1], 1)
        u, singular_values, vt = np.linalg.svd(covariance)
        rotation = np.matmul(vt.transpose(0, 2, 1), u.transpose(0, 2, 1))
        flip = np.linalg.det(rotation) < 0.0
        if np.any(flip):
            singular_values[flip, -1] *= -1.0
            vt[flip, -1, :] *= -1.0
            rotation = np.matmul(vt.transpose(0, 2, 1), u.transpose(0, 2, 1))

        source_var = np.var(source, axis=1).sum(axis=1)
        scale = np.ones_like(source_var)
        valid = source_var > 1e-8
        scale[valid] = singular_values[valid].sum(axis=1) / source_var[valid]
        aligned_eval = (
            scale[:, None, None]
            * np.einsum("bnc,bdc->bnd", source_eval - source_mean[:, None, :], rotation)
            + target_mean[:, None, :]
        )
        values = np.linalg.norm(aligned_eval - target_eval, axis=-1).mean(axis=1)
        dist[pred_idx, gt_idx] = values
    return dist


def frame_distance_matrix_pa(pred_parts, gt_parts, part):
    return batched_procrustes_mpjpe(pred_parts[part], gt_parts[part])


def dtw_from_distance_matrix(dist):
    if dist.shape[0] == 0 or dist.shape[1] == 0:
        return {"dtw": math.inf, "path_len": 0, "ndtw": math.inf, "ndtw_ref": math.inf}

    n, m = dist.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    plen = np.zeros((n + 1, m + 1), dtype=np.int32)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        prev_acc = acc[i - 1]
        cur_acc = acc[i]
        prev_len = plen[i - 1]
        cur_len = plen[i]
        for j in range(1, m + 1):
            candidates = (
                (prev_acc[j], prev_len[j]),
                (cur_acc[j - 1], cur_len[j - 1]),
                (prev_acc[j - 1], prev_len[j - 1]),
            )
            best_acc, best_len = min(candidates, key=lambda item: item[0])
            cur_acc[j] = dist[i - 1, j - 1] + best_acc
            cur_len[j] = best_len + 1

    total = float(acc[n, m])
    path_len = int(plen[n, m])
    return {
        "dtw": total,
        "path_len": path_len,
        "ndtw": total / max(path_len, 1),
        "ndtw_ref": total / max(n, 1),
    }


def dtw_distance_default(pred_part, gt_part):
    return dtw_from_distance_matrix(frame_distance_matrix_mpjpe(pred_part, gt_part))


def dtw_distance_pa(pred_parts, gt_parts, part):
    return dtw_from_distance_matrix(frame_distance_matrix_pa(pred_parts, gt_parts, part))


def pair_files(samples_dir, sample_glob):
    pairs = []
    for sample_path in sorted(samples_dir.glob(sample_glob)):
        suffix = sample_path.stem.replace("sample_", "", 1)
        gt_path = samples_dir / f"gt_{suffix}.npz"
        if gt_path.is_file():
            pairs.append((suffix, gt_path, sample_path))
    return pairs


def summarize(rows):
    summary = {}
    keys = sorted({(row["comparison"], row["part"]) for row in rows})
    for comparison, part in keys:
        subset = [row for row in rows if row["comparison"] == comparison and row["part"] == part]
        ndtw = np.asarray([row["ndtw"] for row in subset], dtype=np.float64)
        ndtw_ref = np.asarray([row["ndtw_ref"] for row in subset], dtype=np.float64)
        dtw = np.asarray([row["dtw"] for row in subset], dtype=np.float64)
        summary[f"{comparison}/{part}"] = {
            "count": int(len(subset)),
            "dtw_mean": float(dtw.mean()),
            "dtw_std": float(dtw.std()),
            "ndtw_mean": float(ndtw.mean()),
            "ndtw_std": float(ndtw.std()),
            "ndtw_median": float(np.median(ndtw)),
            "ndtw_ref_mean": float(ndtw_ref.mean()),
            "ndtw_ref_std": float(ndtw_ref.std()),
            "ndtw_ref_median": float(np.median(ndtw_ref)),
        }
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "comparison",
        "part",
        "gt_len",
        "pred_len",
        "dtw",
        "path_len",
        "ndtw",
        "ndtw_ref",
        "sample",
        "gt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    pairs = pair_files(args.samples_dir, args.sample_glob)
    if args.max_items > 0:
        pairs = pairs[: args.max_items]
    if not pairs:
        raise RuntimeError(f"No sample/gt pairs found under {args.samples_dir}")

    device = str(resolve_device(args.device))
    betas_override = resolve_betas_override(args.betas_mode)
    rows = []
    skipped_prior = 0

    for index, gt_path, sample_path in pairs:
        gt_joints, gt_vertices = smplx_to_joints_vertices(
            load_smplx(gt_path, args.gt_key),
            model_dir=args.model_dir,
            gender=args.gender,
            device=device,
            batch_size=args.smplx_batch_size,
            betas_override=betas_override,
        )
        sample_joints, sample_vertices = smplx_to_joints_vertices(
            load_smplx(sample_path, args.sample_key),
            model_dir=args.model_dir,
            gender=args.gender,
            device=device,
            batch_size=args.smplx_batch_size,
            betas_override=betas_override,
        )
        prior_outputs = None
        try:
            prior_joints, prior_vertices = smplx_to_joints_vertices(
                load_smplx(sample_path, args.prior_key),
                model_dir=args.model_dir,
                gender=args.gender,
                device=device,
                batch_size=args.smplx_batch_size,
                betas_override=betas_override,
            )
            if args.alignment_mode == "pa":
                prior_outputs = t2m_raw_parts(prior_joints, prior_vertices)
            else:
                prior_outputs = t2m_default_parts(prior_joints, prior_vertices)
        except KeyError:
            skipped_prior += 1

        if args.alignment_mode == "pa":
            gt_parts = t2m_raw_parts(gt_joints, gt_vertices)
            sample_parts = t2m_raw_parts(sample_joints, sample_vertices)
        else:
            gt_parts = t2m_default_parts(gt_joints, gt_vertices)
            sample_parts = t2m_default_parts(sample_joints, sample_vertices)
        comparisons = [("flow", sample_parts)]
        if prior_outputs is not None:
            comparisons.append(("adapter_prior", prior_outputs))

        for comparison, pred_parts in comparisons:
            for part in args.parts:
                if args.alignment_mode == "pa":
                    values = dtw_distance_pa(pred_parts, gt_parts, part)
                else:
                    values = dtw_distance_default(pred_parts[part], gt_parts[part])
                rows.append(
                    {
                        "index": index,
                        "comparison": comparison,
                        "part": part,
                        "gt_len": int(len(gt_parts[part])),
                        "pred_len": int(len(pred_parts[part])),
                        "dtw": values["dtw"],
                        "path_len": values["path_len"],
                        "ndtw": values["ndtw"],
                        "ndtw_ref": values["ndtw_ref"],
                        "sample": str(sample_path),
                        "gt": str(gt_path),
                    }
                )

    payload = {
        "samples_dir": str(args.samples_dir),
        "sample_key": args.sample_key,
        "gt_key": args.gt_key,
        "prior_key": args.prior_key,
        "model_dir": str(args.model_dir),
        "device": device,
        "frame_metric": "mpjpe",
        "alignment_mode": args.alignment_mode,
        "metric_preset": (
            "t2m_partwise_pa_same_subset"
            if args.alignment_mode == "pa"
            else "mgpt_t2m_default_align_idx_0"
        ),
        "procrustes_aligned": bool(args.alignment_mode == "pa"),
        "betas_mode": args.betas_mode,
        "betas_override": None if betas_override is None else betas_override.tolist(),
        "joint_sets": {
            "body": list(UPPER_BODY_JOINTS),
            "lhand": "mGPT orig_hand_regressor left layout, 21 keypoints",
            "rhand": "mGPT orig_hand_regressor right layout, 21 keypoints",
            "wholebody": "upper_body + lhand + rhand, 54 keypoints",
        },
        "definition": (
            "Matches mGPT/metrics/t2m.py l2_dist_align behavior. In default mode, "
            "align_idx=0 gives translated DTW-MPJPE: body is pelvis-translation "
            "aligned before selecting upper_body, each hand is wrist-translation "
            "aligned, and wholebody is aligned by its first concatenated joint, "
            "Neck. In pa mode, each predicted part is similarity "
            "Procrustes-aligned to the corresponding GT part before MPJPE, using "
            "the same keypoint set for fitting and scoring. Body uses the 12 "
            "upper-body keypoints, each hand uses its 21-keypoint layout, and "
            "wholebody uses all 54 concatenated keypoints. dtw_mean is the raw DTW "
            "value; ndtw is dtw divided by optimal path length."
        ),
        "num_pairs": len(pairs),
        "skipped_prior": skipped_prior,
        "parts": args.parts,
        "summary": summarize(rows),
        "rows": rows,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)

    print(json.dumps({"num_pairs": len(pairs), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
