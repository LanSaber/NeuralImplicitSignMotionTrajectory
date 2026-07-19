from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from flow.dataset import UpperSMPLXFlowDataset
from flow.smplx_features import COMPACT6D_DIM
from flow.smplx_features import resample_array


PART_KEYS = ("body", "lhand", "rhand", "wholebody")


def build_upper_smplx_dataset(cfg, split, limit=None, random_crop=None):
    data_cfg = cfg["data"]
    return UpperSMPLXFlowDataset(
        data_cfg["data_dir"],
        split=split,
        manifest_path=data_cfg.get(f"{split}_manifest_path"),
        min_frames=int(data_cfg.get("min_frames", 40)),
        max_frames=int(data_cfg.get("max_frames", 1000)),
        length_multiple=int(data_cfg.get("length_multiple", 4)),
        random_crop=bool(data_cfg.get("random_crop", False) if random_crop is None else random_crop),
        limit=int(limit if limit is not None else data_cfg.get(f"limit_{split}", 0)),
        rotation_rep="rot6d",
    )


def denormalize_motion(motion, mean, std):
    mean = torch.as_tensor(mean, device=motion.device, dtype=motion.dtype)
    std = torch.as_tensor(std, device=motion.device, dtype=motion.dtype)
    return motion * std.view(1, 1, -1) + mean.view(1, 1, -1)


def cache_path_for_item(cache_dir, item):
    return Path(cache_dir) / item["motion_path"]


class ContinuousSignDataset(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        limit=None,
        random_crop=None,
        require_fk_cache=True,
    ):
        self.cfg = cfg
        self.split = str(split)
        self.base = build_upper_smplx_dataset(cfg, self.split, limit=limit, random_crop=random_crop)
        self.cache_dir = Path(cfg["cache"]["fk_cache_dir"])
        self.require_fk_cache = bool(require_fk_cache)
        self.mean = self.base.mean.astype(np.float32)
        self.std = self.base.std.astype(np.float32)

    def __len__(self):
        return len(self.base)

    def _load_target_parts(self, item, target_length=None):
        path = cache_path_for_item(self.cache_dir, item)
        if not path.is_file():
            if self.require_fk_cache:
                raise FileNotFoundError(
                    f"Missing FK cache for {item['motion_path']}: {path}. "
                    "Run python -m NIAF.continuous_sign_field.scripts.cache_fk_joints first."
                )
            return None
        target_length = int(target_length or 0)
        with np.load(path) as data:
            parts = {}
            for key in PART_KEYS:
                value = data[key].astype(np.float32)
                if target_length > 0 and len(value) != target_length:
                    value = resample_array(value, target_length, nearest=False).astype(np.float32)
                parts[key] = torch.from_numpy(value)
            return parts

    def __getitem__(self, index):
        sample = self.base[index]
        manifest_item = self.base.items[index]
        sample["index"] = int(index)
        sample["motion_path"] = manifest_item["motion_path"]
        sample["fps"] = float(manifest_item.get("fps", 20.0))
        sample["duration"] = float(manifest_item.get("duration", sample["length"] / sample["fps"]))
        sample["target_parts"] = self._load_target_parts(manifest_item, target_length=sample["length"])
        return sample


def collate_continuous_sign(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_len = max(int(item["length"]) for item in batch)
    dim = batch[0]["motion"].shape[-1]
    if dim != COMPACT6D_DIM:
        raise ValueError(f"Expected compact rot6D dim {COMPACT6D_DIM}, got {dim}")

    motions = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    left_valid = torch.zeros(len(batch), max_len, dtype=torch.float32)
    right_valid = torch.zeros(len(batch), max_len, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    names = []
    texts = []
    glosses = []
    motion_paths = []
    indices = []
    fps = []
    durations = []

    has_parts = batch[0].get("target_parts") is not None
    target_parts = {}
    if has_parts:
        for key, value in batch[0]["target_parts"].items():
            target_parts[key] = torch.zeros(
                len(batch),
                max_len,
                value.shape[1],
                value.shape[2],
                dtype=torch.float32,
            )

    for idx, item in enumerate(batch):
        length = int(item["length"])
        motions[idx, :length] = item["motion"][:length]
        mask[idx, :length] = True
        left_valid[idx, :length] = item["left_valid"][:length]
        right_valid[idx, :length] = item["right_valid"][:length]
        lengths[idx] = length
        names.append(item["name"])
        texts.append(item.get("text", ""))
        glosses.append(item.get("gloss", ""))
        motion_paths.append(item.get("motion_path", ""))
        indices.append(int(item.get("index", idx)))
        fps.append(float(item.get("fps", 20.0)))
        durations.append(float(item.get("duration", length / max(float(item.get("fps", 20.0)), 1.0))))
        if has_parts:
            for key in PART_KEYS:
                value = item["target_parts"][key]
                target_parts[key][idx, :length] = value[:length]

    out = {
        "name": names,
        "text": texts,
        "gloss": glosses,
        "motion_path": motion_paths,
        "index": torch.tensor(indices, dtype=torch.long),
        "motion": motions,
        "length": lengths,
        "mask": mask,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "fps": torch.tensor(fps, dtype=torch.float32),
        "duration": torch.tensor(durations, dtype=torch.float32),
    }
    if has_parts:
        out["target_parts"] = target_parts
    return out
