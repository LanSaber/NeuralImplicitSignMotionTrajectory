from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _np(x):
    return x.detach().cpu().float().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)


def pca2(points):
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def save_latent_plots(out_dir, stem, z_gt, z_pred):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_gt = _np(z_gt)
    z_pred = _np(z_pred)
    both = np.concatenate([z_gt, z_pred], axis=0)
    xy = pca2(both)
    gt_xy = xy[: len(z_gt)]
    pred_xy = xy[len(z_gt) :]
    progress = np.linspace(0.0, 1.0, len(z_gt))

    plt.figure(figsize=(6, 5))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=1.0, alpha=0.45, label="gt")
    plt.scatter(gt_xy[:, 0], gt_xy[:, 1], c=progress, cmap="viridis", s=18)
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], color="#d95f02", linewidth=1.0, label="field")
    plt.scatter(pred_xy[:, 0], pred_xy[:, 1], c=progress, cmap="magma", s=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_pca.png", dpi=160)
    plt.close()

    gt_dist = np.linalg.norm(z_gt[1:] - z_gt[:-1], axis=-1)
    pred_dist = np.linalg.norm(z_pred[1:] - z_pred[:-1], axis=-1)
    plt.figure(figsize=(7, 3))
    plt.plot(gt_dist, label="gt", linewidth=1.2)
    plt.plot(pred_dist, label="field", linewidth=1.2)
    if len(gt_dist):
        plt.axhline(np.percentile(gt_dist, 95), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_adjacent_distance.png", dpi=160)
    plt.close()
