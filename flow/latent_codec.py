import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flow.VAE.model import TemporalSMPLXVAE
from flow.dataset import collate_upper_smplx
from flow.smplx_features import COMPACT_DIM, normalize_rotation_rep


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_vae_from_checkpoint(checkpoint):
    config = dict(checkpoint["model_config"])
    config.setdefault("input_dim", COMPACT_DIM)
    config.pop("rotation_rep", None)
    config.pop("representation_dim", None)
    model = TemporalSMPLXVAE(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class LatentMotionCodec:
    """Frozen deterministic VAE codec for normalized compact SMPL-X motion."""

    def __init__(self, checkpoint_path, device):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"VAE checkpoint does not exist: {self.checkpoint_path}")

        self.checkpoint = load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.model = build_vae_from_checkpoint(self.checkpoint).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.input_dim = int(self.model.input_dim)
        self.rotation_rep = normalize_rotation_rep(
            self.checkpoint.get("model_config", {}).get("rotation_rep")
            or self.checkpoint.get("data_config", {}).get("rotation_rep")
            or self.checkpoint.get("args", {}).get("rotation_rep")
            or "axis_angle"
        )
        self.latent_dim = int(self.model.latent_dim)
        self.downsample_factor = int(self.model.downsample_factor)
        self.max_frames = int(self.model.max_frames)
        self.max_latent_frames = int(self.model.max_latent_frames)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def latent_length(self, frame_length):
        return int(math.ceil(int(frame_length) / float(self.downsample_factor)))

    def latent_mask(self, frame_mask):
        return self.model._latent_mask(frame_mask)

    @torch.no_grad()
    def encode(self, motion, mask=None):
        mu, _logvar, latent_mask = self.model.encode(motion, mask=mask)
        return mu, latent_mask

    def decode(self, latent, target_length=None, mask=None, latent_mask=None):
        return self.model.decode(
            latent,
            target_length=target_length,
            mask=mask,
            latent_mask=latent_mask,
        )

    def normalize_latent(self, latent, stats):
        mean, std = latent_stats_tensors(stats, latent.device, latent.dtype)
        return (latent - mean) / std

    def denormalize_latent(self, latent, stats):
        mean, std = latent_stats_tensors(stats, latent.device, latent.dtype)
        return latent * std + mean

    def checkpoint_config(self):
        return {
            "vae_checkpoint": str(self.checkpoint_path),
            "vae_model_config": dict(self.checkpoint.get("model_config", {})),
            "latent_dim": self.latent_dim,
            "downsample_factor": self.downsample_factor,
            "vae_max_frames": self.max_frames,
            "max_latent_frames": self.max_latent_frames,
            "deterministic": "mu",
            "frozen": True,
            "rotation_rep": self.rotation_rep,
            "input_dim": self.input_dim,
        }


def latent_stats_tensors(stats, device, dtype):
    mean = stats["mean"]
    std = stats["std"]
    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean)
    if not torch.is_tensor(std):
        std = torch.as_tensor(std)
    return (
        mean.to(device=device, dtype=dtype).view(1, 1, -1),
        std.to(device=device, dtype=dtype).view(1, 1, -1),
    )


def serializable_latent_stats(stats):
    mean = stats["mean"].detach().cpu() if torch.is_tensor(stats["mean"]) else torch.as_tensor(stats["mean"])
    std = stats["std"].detach().cpu() if torch.is_tensor(stats["std"]) else torch.as_tensor(stats["std"])
    return {
        "mean": mean.float().tolist(),
        "std": std.float().tolist(),
    }


@torch.no_grad()
def compute_latent_stats(dataset, codec, batch_size=8, num_workers=0, device=None):
    if device is None:
        device = codec.device
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=max(int(num_workers), 0),
        collate_fn=collate_upper_smplx,
        pin_memory=device.type == "cuda",
    )
    total = None
    total_sq = None
    count = 0.0
    for batch in loader:
        motion = batch["motion"].to(device)
        mask = batch["mask"].to(device)
        latent, latent_mask = codec.encode(motion, mask=mask)
        valid = latent_mask.to(device=device, dtype=latent.dtype).unsqueeze(-1)
        masked = latent * valid
        item_sum = masked.sum(dim=(0, 1))
        item_sq = (masked * latent).sum(dim=(0, 1))
        if total is None:
            total = item_sum
            total_sq = item_sq
        else:
            total = total + item_sum
            total_sq = total_sq + item_sq
        count += float(latent_mask.sum().item())

    if total is None or count <= 0:
        raise RuntimeError("Cannot compute latent stats from an empty dataset.")
    mean = total / count
    var = (total_sq / count - mean.pow(2)).clamp_min(1e-8)
    std = var.sqrt().clamp_min(1e-4)
    return {
        "mean": mean.detach(),
        "std": std.detach(),
    }
