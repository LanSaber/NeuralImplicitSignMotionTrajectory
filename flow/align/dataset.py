import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Allows non-torch helpers to be used by forced_align tests.
    torch = None

    class Dataset:
        pass


LEFT_HAND_SLICE = slice(30, 75)
RIGHT_HAND_SLICE = slice(75, 120)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gloss_vocab(path):
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    gloss_to_id = {str(key): int(value) for key, value in vocab["gloss_to_id"].items()}
    id_to_gloss = list(vocab["id_to_gloss"])
    blank_id = int(vocab.get("blank_id", 0))
    if blank_id != 0:
        raise ValueError(f"Expected CTC blank id 0, got {blank_id}")
    return {
        **vocab,
        "blank_id": blank_id,
        "gloss_to_id": gloss_to_id,
        "id_to_gloss": id_to_gloss,
        "vocab_size_with_blank": int(vocab.get("vocab_size_with_blank", len(id_to_gloss))),
    }


def gloss_tokens(value):
    return str(value or "").split()


def tokens_to_ids(tokens, gloss_to_id):
    ids = []
    oov = []
    for token in tokens:
        idx = gloss_to_id.get(token)
        if idx is None:
            oov.append(token)
        else:
            ids.append(int(idx))
    return ids, oov


def load_motion_arrays(data_dir, row):
    path = Path(data_dir) / row["motion_path"]
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
    return motion, left_valid, right_valid


def prepare_motion_features(
    motion,
    left_valid,
    right_valid,
    mean,
    std,
    features="motion",
    append_valid=False,
    gate_hands=False,
):
    motion = np.asarray(motion, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if motion.ndim != 2:
        raise ValueError(f"Expected motion [T,D], got {motion.shape}")
    if mean.shape[-1] != motion.shape[-1] or std.shape[-1] != motion.shape[-1]:
        raise ValueError(
            f"Stats dimension mismatch: motion={motion.shape}, mean={mean.shape}, std={std.shape}"
        )

    x = (motion - mean) / np.maximum(std, 1e-4)
    left_valid = np.asarray(left_valid, dtype=np.float32).reshape(-1)
    right_valid = np.asarray(right_valid, dtype=np.float32).reshape(-1)
    if len(left_valid) != len(x) or len(right_valid) != len(x):
        raise ValueError(
            f"Validity length mismatch: motion={len(x)}, left={len(left_valid)}, right={len(right_valid)}"
        )

    if gate_hands and x.shape[-1] >= RIGHT_HAND_SLICE.stop:
        x = x.copy()
        x[:, LEFT_HAND_SLICE] *= left_valid[:, None]
        x[:, RIGHT_HAND_SLICE] *= right_valid[:, None]

    if features == "motion":
        feature_parts = [x]
    elif features == "motion_velocity":
        velocity = np.zeros_like(x)
        velocity[1:] = x[1:] - x[:-1]
        feature_parts = [x, velocity]
    else:
        raise ValueError(f"Unknown feature set: {features}")

    if append_valid:
        valid = np.stack([left_valid, right_valid], axis=-1).astype(np.float32)
        feature_parts.append(valid)

    return np.concatenate(feature_parts, axis=-1).astype(np.float32, copy=False)


class PhoenixGlossCTCDataset(Dataset):
    def __init__(
        self,
        data_dir,
        vocab_path,
        split="train",
        manifest_path=None,
        mean_path=None,
        std_path=None,
        features="motion",
        append_valid=False,
        gate_hands=False,
        drop_oov=True,
        limit=0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.data_dir / "meta" / f"manifest_{split}.jsonl"
        )
        self.mean_path = Path(mean_path) if mean_path is not None else self.data_dir / "meta" / "mean.npy"
        self.std_path = Path(std_path) if std_path is not None else self.data_dir / "meta" / "std.npy"
        self.features = features
        self.append_valid = bool(append_valid)
        self.gate_hands = bool(gate_hands)
        self.drop_oov = bool(drop_oov)

        self.vocab = load_gloss_vocab(vocab_path)
        self.gloss_to_id = self.vocab["gloss_to_id"]
        self.mean = np.load(self.mean_path).astype(np.float32)
        self.std = np.load(self.std_path).astype(np.float32)

        self.all_items = read_jsonl(self.manifest_path)
        self.skipped = []
        self.items = []
        for row in self.all_items:
            tokens = gloss_tokens(row.get("gloss", ""))
            ids, oov = tokens_to_ids(tokens, self.gloss_to_id)
            if not tokens:
                self.skipped.append({"name": row.get("name", ""), "reason": "empty gloss"})
                continue
            if oov and self.drop_oov:
                self.skipped.append({"name": row.get("name", ""), "reason": "oov gloss", "oov": oov})
                continue
            if len(ids) != len(tokens):
                row = {**row, "_target_ids": ids, "_tokens": tokens, "_oov": oov}
            else:
                row = {**row, "_target_ids": ids, "_tokens": tokens, "_oov": []}
            self.items.append(row)

        if limit > 0:
            self.items = self.items[: int(limit)]

        self.feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self):
        dim = int(self.mean.shape[-1])
        if self.features == "motion_velocity":
            dim *= 2
        elif self.features != "motion":
            raise ValueError(f"Unknown feature set: {self.features}")
        if self.append_valid:
            dim += 2
        return dim

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if torch is None:
            raise ModuleNotFoundError("PhoenixGlossCTCDataset requires torch.")
        row = self.items[index]
        motion, left_valid, right_valid = load_motion_arrays(self.data_dir, row)
        features = prepare_motion_features(
            motion,
            left_valid,
            right_valid,
            self.mean,
            self.std,
            features=self.features,
            append_valid=self.append_valid,
            gate_hands=self.gate_hands,
        )
        return {
            "name": row["name"],
            "text": row.get("text", ""),
            "gloss": row.get("gloss", ""),
            "tokens": row["_tokens"],
            "oov": row.get("_oov", []),
            "motion": torch.from_numpy(features).float(),
            "frame_length": int(features.shape[0]),
            "target": torch.tensor(row["_target_ids"], dtype=torch.long),
            "target_length": int(len(row["_target_ids"])),
        }


def collate_ctc(batch):
    if torch is None:
        raise ModuleNotFoundError("collate_ctc requires torch.")
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Cannot collate an empty CTC batch.")

    batch_size = len(batch)
    max_frames = max(item["frame_length"] for item in batch)
    dim = int(batch[0]["motion"].shape[-1])
    motions = torch.zeros(batch_size, max_frames, dim, dtype=torch.float32)
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    targets = []
    names = []
    glosses = []
    tokens = []

    for idx, item in enumerate(batch):
        frames = item["frame_length"]
        motions[idx, :frames] = item["motion"]
        frame_lengths[idx] = frames
        target_lengths[idx] = item["target_length"]
        targets.append(item["target"])
        names.append(item["name"])
        glosses.append(item["gloss"])
        tokens.append(item["tokens"])

    if targets:
        flat_targets = torch.cat(targets, dim=0)
    else:
        flat_targets = torch.empty(0, dtype=torch.long)

    return {
        "name": names,
        "gloss": glosses,
        "tokens": tokens,
        "motion": motions,
        "frame_lengths": frame_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "target_sequences": [target.tolist() for target in targets],
    }
