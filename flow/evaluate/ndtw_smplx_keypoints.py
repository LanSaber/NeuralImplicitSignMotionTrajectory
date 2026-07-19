#!/usr/bin/env python
"""Compute nDTW in SMPL-X keypoint space for saved flow samples."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from flow.render import DEFAULT_MODEL_DIR, resolve_device


# Keep the legacy full SMPL-X body part for reproducibility. The upper-body
# subset follows mGPT.utils.human_models.smpl_x.joint_part2idx["upper_body"].
BODY_JOINTS = tuple(range(0, 22))
UPPER_BODY_JOINTS = (12, 16, 17, 18, 19, 20, 21, 59, 58, 57, 56, 55)
HAND_JOINTS = tuple(range(25, 55)) + tuple(range(66, 76))
PART_JOINTS = {
    "body": BODY_JOINTS,
    "upper_body": UPPER_BODY_JOINTS,
    "hands": HAND_JOINTS,
    "body_hands": BODY_JOINTS + HAND_JOINTS,
    "upper_body_hands": UPPER_BODY_JOINTS + HAND_JOINTS,
}
H2S_FIXED_BETAS = (
    -0.07284723,
    0.1795129,
    -0.27608207,
    0.135155,
    0.10748172,
    0.16037364,
    -0.01616933,
    -0.03450319,
    0.01369138,
    0.01108842,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert predicted and ground-truth SMPL-X params to body/hand joint "
            "positions, then compute DTW and normalized DTW."
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
        default="from_params",
        choices=["from_params", "zero", "h2s_fixed"],
        help=(
            "SMPL-X shape source: use beta columns from the params, all-zero "
            "betas, or the fixed H2S beta vector from mGPT/data/H2S.py."
        ),
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["body_hands"],
        choices=sorted(PART_JOINTS),
        help="Joint subsets to evaluate.",
    )
    parser.add_argument(
        "--root_align",
        action="store_true",
        help="Subtract pelvis joint position from each frame before DTW.",
    )
    parser.add_argument(
        "--procrustes_align",
        action="store_true",
        help="Use similarity Procrustes-aligned frame costs before DTW.",
    )
    parser.add_argument(
        "--frame_metric",
        default="rms",
        choices=["rms", "mpjpe"],
        help="Frame cost after optional alignment: coordinate RMS or mean per-joint Euclidean error.",
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


def smplx_to_joints(
    smplx_params,
    model_dir,
    gender="NEUTRAL",
    device="cpu",
    batch_size=128,
    betas_override=None,
):
    import smplx

    smplx_params = np.asarray(smplx_params, dtype=np.float32)
    if smplx_params.ndim != 2 or smplx_params.shape[1] != 182:
        raise ValueError(f"Expected SMPL-X params with shape [T, 182], got {smplx_params.shape}")

    device = resolve_device(device)
    layer_cache = {}
    chunks = []

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
            chunks.append(output.joints.detach().cpu().numpy().astype(np.float32))
    finally:
        torch.set_grad_enabled(was_grad_enabled)

    return np.concatenate(chunks, axis=0)


def select_part(joints, part, root_align=False):
    selected = joints[:, PART_JOINTS[part], :].astype(np.float32, copy=True)
    if root_align:
        selected -= joints[:, 0:1, :]
    return selected


def point_error(a, b, frame_metric):
    diff = a - b
    if frame_metric == "rms":
        return float(np.sqrt(np.mean(diff * diff, dtype=np.float64), dtype=np.float64))
    if frame_metric == "mpjpe":
        return float(np.linalg.norm(diff, axis=-1).mean())
    raise ValueError(f"Unsupported frame_metric={frame_metric!r}")


def procrustes_aligned_error(source, target, frame_metric, eps=1e-8):
    """Similarity-align source points to target points and return a joint error."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.sum(source_centered * source_centered))

    if source_var <= eps:
        aligned = source - source_mean + target_mean
        return point_error(aligned, target, frame_metric)

    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance, full_matrices=True)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        signs[-1] = -1.0
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular_values * signs) / source_var)
    aligned = scale * (source_centered @ rotation) + target_mean
    return point_error(aligned, target, frame_metric)


def frame_distance_matrix(a, b, frame_metric="rms", procrustes_align=False):
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Feature dimension mismatch: {a.shape} vs {b.shape}")
    if procrustes_align:
        dist = np.empty((len(a), len(b)), dtype=np.float64)
        for i in range(len(a)):
            for j in range(len(b)):
                dist[i, j] = procrustes_aligned_error(b[j], a[i], frame_metric)
        return dist

    if frame_metric == "rms":
        a_flat = a.reshape(a.shape[0], -1).astype(np.float64)
        b_flat = b.reshape(b.shape[0], -1).astype(np.float64)
        dim = float(a_flat.shape[1])
        aa = np.sum(a_flat * a_flat, axis=1, keepdims=True)
        bb = np.sum(b_flat * b_flat, axis=1, keepdims=True).T
        sq = np.maximum(aa + bb - 2.0 * (a_flat @ b_flat.T), 0.0)
        return np.sqrt(sq / dim, dtype=np.float64)

    if frame_metric == "mpjpe":
        diff = a[:, None, :, :].astype(np.float64) - b[None, :, :, :].astype(np.float64)
        return np.linalg.norm(diff, axis=-1).mean(axis=-1)

    raise ValueError(f"Unsupported frame_metric={frame_metric!r}")


def dtw_distance(a, b, frame_metric="rms", procrustes_align=False):
    if len(a) == 0 or len(b) == 0:
        return {"dtw": math.inf, "path_len": 0, "ndtw": math.inf, "ndtw_ref": math.inf}

    dist = frame_distance_matrix(a, b, frame_metric=frame_metric, procrustes_align=procrustes_align)
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
        "frame_metric",
        "procrustes_aligned",
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
        gt_joints = smplx_to_joints(
            load_smplx(gt_path, args.gt_key),
            model_dir=args.model_dir,
            gender=args.gender,
            device=device,
            batch_size=args.smplx_batch_size,
            betas_override=betas_override,
        )
        sample_joints = smplx_to_joints(
            load_smplx(sample_path, args.sample_key),
            model_dir=args.model_dir,
            gender=args.gender,
            device=device,
            batch_size=args.smplx_batch_size,
            betas_override=betas_override,
        )
        prior_joints = None
        try:
            prior_joints = smplx_to_joints(
                load_smplx(sample_path, args.prior_key),
                model_dir=args.model_dir,
                gender=args.gender,
                device=device,
                batch_size=args.smplx_batch_size,
                betas_override=betas_override,
            )
        except KeyError:
            skipped_prior += 1

        comparisons = [("flow", sample_joints)]
        if prior_joints is not None:
            comparisons.append(("adapter_prior", prior_joints))

        for comparison, pred_joints in comparisons:
            for part in args.parts:
                gt_part = select_part(gt_joints, part, root_align=args.root_align)
                pred_part = select_part(pred_joints, part, root_align=args.root_align)
                values = dtw_distance(
                    gt_part,
                    pred_part,
                    frame_metric=args.frame_metric,
                    procrustes_align=args.procrustes_align,
                )
                rows.append(
                    {
                        "index": index,
                        "comparison": comparison,
                        "part": part,
                        "gt_len": int(len(gt_part)),
                        "pred_len": int(len(pred_part)),
                        "dtw": values["dtw"],
                        "path_len": values["path_len"],
                        "ndtw": values["ndtw"],
                        "ndtw_ref": values["ndtw_ref"],
                        "sample": str(sample_path),
                        "gt": str(gt_path),
                        "frame_metric": args.frame_metric,
                        "procrustes_aligned": bool(args.procrustes_align),
                    }
                )

    payload = {
        "samples_dir": str(args.samples_dir),
        "sample_key": args.sample_key,
        "gt_key": args.gt_key,
        "prior_key": args.prior_key,
        "model_dir": str(args.model_dir),
        "device": device,
        "root_align": bool(args.root_align),
        "procrustes_aligned": bool(args.procrustes_align),
        "frame_metric": args.frame_metric,
        "betas_mode": args.betas_mode,
        "betas_override": None if betas_override is None else betas_override.tolist(),
        "joint_sets": {name: list(indices) for name, indices in PART_JOINTS.items() if name in args.parts},
        "definition": (
            "SMPL-X params are converted to joints. Frame cost is computed over the selected "
            "joint coordinates, optionally after similarity Procrustes alignment. "
            "DTW uses steps (i-1,j), (i,j-1), and (i-1,j-1). "
            "ndtw = dtw / optimal_path_length; ndtw_ref = dtw / len(gt). Lower is better."
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
