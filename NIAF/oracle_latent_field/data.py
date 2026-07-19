from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flow.dataset import UpperSMPLXFlowDataset
from flow.evaluate.dtw_mpjpe_t2m_default import (
    dtw_distance_default,
    dtw_distance_pa,
    resolve_betas_override,
    smplx_to_joints_vertices,
    t2m_default_parts,
    t2m_raw_parts,
)
from flow.latent_codec import LatentMotionCodec
from flow.render import DEFAULT_MODEL_DIR
from flow.smplx_features import compact_from_rotation_representation, smplx182_from_compact


@dataclass
class OracleSample:
    index: int
    sample_id: str
    text: str
    x_norm: torch.Tensor
    x_vae_norm: torch.Tensor
    z_norm: torch.Tensor
    z_raw: torch.Tensor
    frame_mask: torch.Tensor
    latent_mask: torch.Tensor
    length: int
    latent_length: int
    rotation_rep: str
    fps: float | None = None


def build_dataset(cfg):
    data_cfg = cfg["data"]
    return UpperSMPLXFlowDataset(
        data_cfg["data_dir"],
        split=data_cfg.get("split", "val"),
        manifest_path=data_cfg.get("manifest_path"),
        min_frames=int(data_cfg.get("min_frames", 40)),
        max_frames=int(data_cfg.get("max_frames", 400)),
        length_multiple=int(data_cfg.get("length_multiple", 4)),
        random_crop=False,
        limit=int(data_cfg.get("limit", 0)),
        rotation_rep=cfg.get("rotation_rep", data_cfg.get("rotation_rep", "rot6d")),
    )


def build_codec(cfg, device):
    return LatentMotionCodec(cfg["vae"]["checkpoint"], device=device)


def adjusted_length_for_item(dataset, item):
    if "num_frames" in item:
        return int(dataset._target_length(int(item["num_frames"])))
    path = Path(dataset.data_dir) / item["motion_path"]
    with np.load(path) as data:
        return int(dataset._target_length(len(data["motion"])))


def select_pilot_indices(dataset, cfg, downsample_factor=4):
    data_cfg = cfg["data"]
    max_sequences = int(data_cfg.get("max_sequences", 100))
    min_latent_len = int(data_cfg.get("min_latent_len", 8))
    max_latent_len = int(data_cfg.get("max_latent_len", 10**9))
    seed = int(cfg.get("seed", 1234))

    candidates = []
    for idx, item in enumerate(dataset.items):
        length = adjusted_length_for_item(dataset, item)
        latent_length = int(math.ceil(length / float(downsample_factor)))
        if min_latent_len <= latent_length <= max_latent_len:
            candidates.append((idx, latent_length))
    if max_sequences <= 0 or len(candidates) <= max_sequences:
        return [idx for idx, _ in candidates]

    buckets = {
        "short": [idx for idx, latent_len in candidates if latent_len < 25],
        "medium": [idx for idx, latent_len in candidates if 25 <= latent_len < 60],
        "long": [idx for idx, latent_len in candidates if latent_len >= 60],
    }
    rng = np.random.default_rng(seed)
    selected = []
    per_bucket = max_sequences // 3
    remainder = max_sequences - per_bucket * 3
    for offset, name in enumerate(["short", "medium", "long"]):
        bucket = list(buckets[name])
        rng.shuffle(bucket)
        take = min(len(bucket), per_bucket + (1 if offset < remainder else 0))
        selected.extend(bucket[:take])
    if len(selected) < max_sequences:
        remaining = [idx for idx, _ in candidates if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_sequences - len(selected)])
    return selected[:max_sequences]


def _item_fps(dataset, index):
    item = dataset.items[index]
    if "fps" in item:
        try:
            return float(item["fps"])
        except Exception:
            return None
    return None


def load_oracle_sample(dataset, index, codec, latent_stats, device):
    item = dataset[index]
    x = item["motion"].to(device=device, dtype=torch.float32)
    length = int(item["length"])
    frame_mask = torch.ones(1, length, dtype=torch.bool, device=device)
    with torch.no_grad():
        z_raw, latent_mask = codec.encode(x.unsqueeze(0), mask=frame_mask)
        z_norm = codec.normalize_latent(z_raw, latent_stats)
        x_vae = codec.decode(
            z_raw,
            target_length=length,
            mask=frame_mask,
            latent_mask=latent_mask,
        )
    latent_length = int(latent_mask[0].sum().item())
    return OracleSample(
        index=int(index),
        sample_id=str(item["name"]),
        text=str(item.get("text", "")),
        x_norm=x[:length].detach(),
        x_vae_norm=x_vae[0, :length].detach(),
        z_norm=z_norm[0, :latent_length].detach(),
        z_raw=z_raw[0, :latent_length].detach(),
        frame_mask=frame_mask[0].detach(),
        latent_mask=latent_mask[0, :latent_length].detach(),
        length=length,
        latent_length=latent_length,
        rotation_rep=str(item.get("rotation_rep", dataset.rotation_rep)),
        fps=_item_fps(dataset, index),
    )


def denormalize_representation(x_norm, dataset):
    x_np = x_norm.detach().cpu().float().numpy() if torch.is_tensor(x_norm) else np.asarray(x_norm, dtype=np.float32)
    return (x_np * dataset.std[None] + dataset.mean[None]).astype(np.float32)


def normalized_to_compact_axis_angle(x_norm, dataset, rotation_rep):
    rep = denormalize_representation(x_norm, dataset)
    return compact_from_rotation_representation(rep, rotation_rep)


def sample_to_smplx(x_norm, dataset, rotation_rep):
    return smplx182_from_compact(normalized_to_compact_axis_angle(x_norm, dataset, rotation_rep))


def save_motion_npz(path, x_norm, dataset, rotation_rep, sample, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    representation = denormalize_representation(x_norm, dataset)
    compact = compact_from_rotation_representation(representation, rotation_rep)
    np.savez_compressed(
        path,
        motion=compact.astype(np.float32),
        representation=representation.astype(np.float32),
        smplx=smplx182_from_compact(compact),
        rotation_rep=str(rotation_rep),
        name=str(sample.sample_id),
        text=str(sample.text),
        length=int(sample.length),
        label=str(label),
        source_index=int(sample.index),
    )


def compute_gt_pose_points(
    sample,
    dataset,
    rotation_rep,
    model_dir=DEFAULT_MODEL_DIR,
    gender="NEUTRAL",
    device="cpu",
    batch_size=128,
    betas_mode="h2s_fixed",
):
    smplx_params = sample_to_smplx(sample.x_norm, dataset, rotation_rep)
    joints, vertices = smplx_to_joints_vertices(
        smplx_params,
        model_dir=model_dir,
        gender=gender,
        device=device,
        batch_size=batch_size,
        betas_override=resolve_betas_override(betas_mode),
    )
    parts = t2m_raw_parts(joints, vertices)
    return parts["wholebody"].astype(np.float32)


def smplx_parts(
    smplx_params,
    model_dir=DEFAULT_MODEL_DIR,
    gender="NEUTRAL",
    device="cpu",
    batch_size=128,
    betas_mode="h2s_fixed",
    alignment_mode="default",
):
    joints, vertices = smplx_to_joints_vertices(
        smplx_params,
        model_dir=model_dir,
        gender=gender,
        device=device,
        batch_size=batch_size,
        betas_override=resolve_betas_override(betas_mode),
    )
    if alignment_mode == "pa":
        return t2m_raw_parts(joints, vertices)
    return t2m_default_parts(joints, vertices)


def dtw_decoded_metrics(
    x_field_norm,
    sample,
    dataset,
    rotation_rep,
    model_dir=DEFAULT_MODEL_DIR,
    gender="NEUTRAL",
    device="cpu",
    batch_size=128,
    betas_mode="h2s_fixed",
    parts=("body", "lhand", "rhand", "wholebody"),
    alignment_mode="default",
):
    gt_parts = smplx_parts(
        sample_to_smplx(sample.x_norm, dataset, rotation_rep),
        model_dir=model_dir,
        gender=gender,
        device=device,
        batch_size=batch_size,
        betas_mode=betas_mode,
        alignment_mode=alignment_mode,
    )
    vae_parts = smplx_parts(
        sample_to_smplx(sample.x_vae_norm, dataset, rotation_rep),
        model_dir=model_dir,
        gender=gender,
        device=device,
        batch_size=batch_size,
        betas_mode=betas_mode,
        alignment_mode=alignment_mode,
    )
    field_parts = smplx_parts(
        sample_to_smplx(x_field_norm, dataset, rotation_rep),
        model_dir=model_dir,
        gender=gender,
        device=device,
        batch_size=batch_size,
        betas_mode=betas_mode,
        alignment_mode=alignment_mode,
    )
    out = {}
    for comparison, pred_parts in [("vae", vae_parts), ("field", field_parts)]:
        for part in parts:
            if alignment_mode == "pa":
                values = dtw_distance_pa(pred_parts, gt_parts, part)
            else:
                values = dtw_distance_default(pred_parts[part], gt_parts[part])
            prefix = f"{alignment_mode}_{comparison}_{part}"
            out[f"{prefix}_dtw"] = float(values["dtw"])
            out[f"{prefix}_ndtw"] = float(values["ndtw"])
            out[f"{prefix}_ndtw_ref"] = float(values["ndtw_ref"])
    return out
