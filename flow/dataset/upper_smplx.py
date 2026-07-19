import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from flow.smplx_features import fit_length
from flow.smplx_features import (
    compact_to_rotation_representation,
    normalize_rotation_rep,
    rotation_rep_dim,
    rotation_rep_stats_paths,
)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class UpperSMPLXFlowDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        manifest_path=None,
        mean_path=None,
        std_path=None,
        min_frames=40,
        max_frames=400,
        length_multiple=4,
        random_crop=True,
        limit=0,
        rotation_rep="axis_angle",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.rotation_rep = normalize_rotation_rep(rotation_rep)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.data_dir / "meta" / f"manifest_{split}.jsonl"
        )
        default_mean_path, default_std_path = rotation_rep_stats_paths(self.data_dir, self.rotation_rep)
        self.mean_path = Path(mean_path) if mean_path is not None else default_mean_path
        self.std_path = Path(std_path) if std_path is not None else default_std_path
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.length_multiple = int(length_multiple)
        self.random_crop = bool(random_crop)

        self.all_items = read_jsonl(self.manifest_path)
        self.items = list(self.all_items)
        if limit > 0:
            self.items = self.items[:limit]
        if not self.mean_path.is_file() or not self.std_path.is_file():
            self._compute_and_save_stats()
        self.mean = np.load(self.mean_path).astype(np.float32)
        self.std = np.load(self.std_path).astype(np.float32)
        expected_dim = rotation_rep_dim(self.rotation_rep)
        if self.mean.shape[-1] != expected_dim or self.std.shape[-1] != expected_dim:
            raise ValueError(
                f"Stats dimension mismatch for rotation_rep={self.rotation_rep}: "
                f"mean={self.mean.shape}, std={self.std.shape}, expected {expected_dim}"
            )

    def _compute_and_save_stats(self):
        self.mean_path.parent.mkdir(parents=True, exist_ok=True)
        total = None
        total_sq = None
        count = 0
        for item in self.all_items:
            path = self.data_dir / item["motion_path"]
            with np.load(path) as data:
                motion = data["motion"].astype(np.float32)
            motion = compact_to_rotation_representation(motion, self.rotation_rep)
            flat = motion.reshape(-1, motion.shape[-1]).astype(np.float64)
            if total is None:
                total = flat.sum(axis=0)
                total_sq = (flat * flat).sum(axis=0)
            else:
                total += flat.sum(axis=0)
                total_sq += (flat * flat).sum(axis=0)
            count += flat.shape[0]
        if total is None or count <= 0:
            raise RuntimeError(f"Cannot compute {self.rotation_rep} stats from empty manifest {self.manifest_path}")
        mean = (total / count).astype(np.float32)
        var = np.maximum(total_sq / count - mean.astype(np.float64) ** 2, 1e-8)
        std = np.sqrt(var).astype(np.float32)
        std = np.maximum(std, 1e-4).astype(np.float32)
        np.save(self.mean_path, mean)
        np.save(self.std_path, std)

    def __len__(self):
        return len(self.items)

    def _target_length(self, length):
        if length < self.min_frames:
            return self.min_frames
        if length > self.max_frames:
            return self.max_frames
        return max(self.length_multiple, (length // self.length_multiple) * self.length_multiple)

    def _adjust_length(self, motion, left_valid, right_valid):
        length = int(motion.shape[0])
        target = self._target_length(length)

        if length < self.min_frames:
            return fit_length(motion, left_valid, right_valid, target)
        if length > self.max_frames:
            if self.random_crop and self.split == "train":
                start = random.randint(0, length - self.max_frames)
                end = start + self.max_frames
                return motion[start:end], left_valid[start:end], right_valid[start:end]
            return fit_length(motion, left_valid, right_valid, target)

        if target < length:
            if self.random_crop and self.split == "train":
                start = random.randint(0, length - target)
            else:
                start = (length - target) // 2
            end = start + target
            return motion[start:end], left_valid[start:end], right_valid[start:end]

        return motion, left_valid, right_valid

    def __getitem__(self, index):
        item = self.items[index]
        path = self.data_dir / item["motion_path"]
        with np.load(path) as data:
            motion = data["motion"].astype(np.float32)
            left_valid = (
                data["left_valid"].astype(np.float32)
                if "left_valid" in data.files
                else np.ones(len(motion), dtype=np.float32)
            )
            right_valid = (
                data["right_valid"].astype(np.float32)
                if "right_valid" in data.files
                else np.ones(len(motion), dtype=np.float32)
            )
        motion, left_valid, right_valid = self._adjust_length(motion, left_valid, right_valid)
        motion = compact_to_rotation_representation(motion, self.rotation_rep)
        motion = (motion - self.mean) / self.std

        return {
            "name": item["name"],
            "text": item.get("text", ""),
            "gloss": item.get("gloss", ""),
            "label_word": item.get("label_word", ""),
            "motion": torch.from_numpy(motion).float(),
            "length": int(motion.shape[0]),
            "left_valid": torch.from_numpy(left_valid).float(),
            "right_valid": torch.from_numpy(right_valid).float(),
            "rotation_rep": self.rotation_rep,
        }


def collate_upper_smplx(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_len = max(item["length"] for item in batch)
    dim = batch[0]["motion"].shape[-1]
    motions = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    left_valid = torch.zeros(len(batch), max_len, dtype=torch.float32)
    right_valid = torch.zeros(len(batch), max_len, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    names = []
    texts = []
    glosses = []
    label_words = []

    for idx, item in enumerate(batch):
        length = item["length"]
        motions[idx, :length] = item["motion"]
        mask[idx, :length] = True
        left_valid[idx, :length] = item["left_valid"][:length]
        right_valid[idx, :length] = item["right_valid"][:length]
        lengths[idx] = length
        names.append(item["name"])
        texts.append(item["text"])
        glosses.append(item.get("gloss", ""))
        label_words.append(item.get("label_word", ""))

    return {
        "name": names,
        "text": texts,
        "gloss": glosses,
        "label_word": label_words,
        "motion": motions,
        "length": lengths,
        "mask": mask,
        "left_valid": left_valid,
        "right_valid": right_valid,
    }
