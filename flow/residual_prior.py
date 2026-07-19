import itertools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from flow.smplx_features import (
    compact_to_rotation_representation,
    normalize_rotation_rep,
    rotation_rep_dim,
    fit_length,
)


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
WORD_VARIANT_RE = re.compile(r"^([A-Za-z0-9_]+)-(\d+)$")


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_token(token):
    token = str(token).upper()
    folded = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii").upper()
    return folded if folded else token


def text_tokens(text):
    text = unicodedata.normalize("NFKC", str(text)).upper()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("—", " ")
        .replace("–", " ")
        .replace("-", " ")
        .replace("_", " ")
    )
    tokens = []
    for token in TOKEN_RE.findall(text):
        normalized = _normalize_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def token_variants(token):
    token = str(token).upper()
    variants = [token]
    if len(token) > 3 and token.endswith("IES"):
        variants.append(token[:-3] + "Y")
    if len(token) > 4 and token.endswith("IED"):
        variants.append(token[:-3] + "Y")
    if len(token) > 3 and token.endswith("ES"):
        variants.append(token[:-2])
    if len(token) > 3 and token.endswith("S"):
        variants.append(token[:-1])
    if len(token) > 4 and token.endswith("ED"):
        variants.append(token[:-2])
        variants.append(token[:-1])
    if len(token) > 5 and token.endswith("ING"):
        stem = token[:-3]
        variants.append(stem)
        variants.append(stem + "E")

    deduped = []
    seen = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def split_word_variant_name(value):
    """Split WORD-0001 style names into (WORD, 0001).

    The variant suffix is intentionally not treated as a lexicon token. This lets
    a word/gloss dataset contain multiple motion examples for the same key.
    """

    text = str(value).strip()
    match = WORD_VARIANT_RE.match(text)
    if match and match.group(1):
        return match.group(1), match.group(2)
    return text, None


def lexicon_key_from_row(row):
    name = str(row.get("name") or Path(row["motion_path"]).stem)
    path_stem = Path(row["motion_path"]).stem

    explicit_key = (
        row.get("lexicon_key")
        or row.get("word")
        or row.get("gloss")
        or row.get("label")
    )
    if explicit_key is not None and str(explicit_key).strip():
        lexicon_key = str(explicit_key).strip()
    else:
        name_key, _name_variant = split_word_variant_name(name)
        path_key, _path_variant = split_word_variant_name(path_stem)
        if name_key != name:
            lexicon_key = name_key
        elif path_key != path_stem:
            lexicon_key = path_key
        else:
            lexicon_key = name_key or path_key

    _name_key, name_variant = split_word_variant_name(name)
    _path_key, path_variant = split_word_variant_name(path_stem)
    variant_id = row.get("variant_id")
    if variant_id is None or str(variant_id) == "":
        variant_id = name_variant if name_variant is not None else path_variant
    if variant_id is not None:
        variant_id = str(variant_id)
    return lexicon_key, variant_id


class WordMotionPrior:
    """Compose a coarse compact-SMPL-X sequence from word/gloss entries.

    `entries_by_key` maps one lexicon token tuple to every available motion
    variant. `entries` remains a flat list for negative sampling and reporting.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        target_mean=None,
        target_std=None,
        max_variant_product=64,
        rotation_rep="axis_angle",
        lazy_motions=False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.rotation_rep = normalize_rotation_rep(rotation_rep)
        self.dim = rotation_rep_dim(self.rotation_rep)
        self.manifest_path = self.data_dir / "meta" / f"manifest_{split}.jsonl"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing word prior manifest: {self.manifest_path}")

        self.target_mean = (
            np.asarray(target_mean, dtype=np.float32).reshape(1, self.dim)
            if target_mean is not None
            else np.zeros((1, self.dim), dtype=np.float32)
        )
        self.target_std = (
            np.asarray(target_std, dtype=np.float32).reshape(1, self.dim)
            if target_std is not None
            else np.ones((1, self.dim), dtype=np.float32)
        )
        self.max_variant_product = int(max_variant_product)
        self.lazy_motions = bool(lazy_motions)
        self.entries = []
        self.entries_by_key = {}
        self.max_key_len = 1
        self.cache = {}
        self._load_entries()

    def _load_motion(self, motion_path):
        path = self.data_dir / motion_path
        with np.load(path) as data:
            motion = data["motion"].astype(np.float32)
        if motion.ndim != 2 or len(motion) == 0:
            return None
        motion = compact_to_rotation_representation(motion, self.rotation_rep)
        if motion.shape[1] != self.dim:
            return None
        return motion

    def entry_motion(self, entry):
        motion = entry.get("motion")
        if motion is None:
            motion = self._load_motion(entry["motion_path"])
            if motion is None:
                raise ValueError(f"Invalid word motion entry: {entry['motion_path']}")
            entry["motion"] = motion
            entry["motion_frames"] = int(motion.shape[0])
        return motion

    def _load_entries(self):
        for row_index, row in enumerate(read_jsonl(self.manifest_path)):
            name = str(row.get("name") or Path(row["motion_path"]).stem)
            lexicon_key, variant_id = lexicon_key_from_row(row)
            tokens = tuple(text_tokens(lexicon_key))
            if not tokens:
                continue
            motion = None
            motion_frames = int(row.get("num_frames") or 0)
            if not self.lazy_motions:
                motion = self._load_motion(row["motion_path"])
                if motion is None:
                    continue
                motion_frames = int(motion.shape[0])
            elif motion_frames <= 0:
                motion = self._load_motion(row["motion_path"])
                if motion is None:
                    continue
                motion_frames = int(motion.shape[0])
            entry = {
                "entry_id": len(self.entries),
                "name": name,
                "lexicon_key": lexicon_key,
                "variant_id": variant_id,
                "tokens": tokens,
                "motion": motion,
                "motion_frames": motion_frames,
                "motion_path": row["motion_path"],
                "manifest_index": row_index,
            }
            self.entries.append(entry)
            self.entries_by_key.setdefault(tokens, []).append(entry)
            self.max_key_len = max(self.max_key_len, len(tokens))

    def _candidate_keys(self, tokens):
        variants = [token_variants(token) for token in tokens]
        product = 1
        for values in variants:
            product *= max(len(values), 1)
        if product > self.max_variant_product:
            yield tuple(tokens)
            return
        for combo in itertools.product(*variants):
            yield tuple(combo)

    def match_text(self, text):
        tokens = text_tokens(text)
        matches = []
        index = 0
        while index < len(tokens):
            found = None
            max_len = min(self.max_key_len, len(tokens) - index)
            for length in range(max_len, 0, -1):
                span = tokens[index : index + length]
                for key in self._candidate_keys(span):
                    if key in self.entries_by_key:
                        found = self.entries_by_key[key][0]
                        break
                if found is not None:
                    break
            if found is None:
                index += 1
                continue
            matches.append(found)
            index += len(found["tokens"])
        return tokens, matches

    def match_text_variants(self, text):
        """Return all motion variants for every matched lexicon span."""

        tokens = text_tokens(text)
        positives = []
        spans = []
        index = 0
        while index < len(tokens):
            found_key = None
            found_entries = None
            found_length = 0
            max_len = min(self.max_key_len, len(tokens) - index)
            for length in range(max_len, 0, -1):
                span = tokens[index : index + length]
                for key in self._candidate_keys(span):
                    entries = self.entries_by_key.get(key)
                    if entries:
                        found_key = key
                        found_entries = entries
                        found_length = length
                        break
                if found_entries is not None:
                    break
            if found_entries is None:
                index += 1
                continue
            positives.extend(found_entries)
            spans.append(
                {
                    "tokens": list(found_key),
                    "text": " ".join(found_key),
                    "start": index,
                    "end": index + found_length,
                    "variant_count": len(found_entries),
                    "variants": [entry["name"] for entry in found_entries],
                }
            )
            index += found_length
        return tokens, positives, spans

    def compose(self, text, target_len):
        target_len = int(target_len)
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        cache_key = (str(text), target_len)
        if cache_key in self.cache:
            return self.cache[cache_key]

        tokens, matches = self.match_text(text)
        if matches:
            coarse = np.concatenate([self.entry_motion(entry) for entry in matches], axis=0)
            valid = np.ones(len(coarse), dtype=np.float32)
            coarse, _, _ = fit_length(coarse, valid, valid, target_len)
        else:
            coarse = self.target_mean.repeat(target_len, axis=0).astype(np.float32)

        coarse = coarse.astype(np.float32, copy=False)
        coarse_norm = (coarse - self.target_mean) / self.target_std
        stats = {
            "tokens": tokens,
            "matched": [entry["name"] for entry in matches],
            "matched_lexicon_keys": [entry.get("lexicon_key", entry["name"]) for entry in matches],
            "matched_count": len(matches),
            "matched_variant_count": len(matches),
            "token_count": len(tokens),
            "coverage": float(len(matches) / max(len(tokens), 1)),
        }
        result = (coarse_norm.astype(np.float32, copy=False), stats)
        self.cache[cache_key] = result
        return result

    def batch(self, texts, lengths, max_len=None, device=None, dtype=torch.float32):
        lengths = [int(length) for length in lengths]
        if max_len is None:
            max_len = max(lengths)
        prior = np.zeros((len(texts), int(max_len), self.dim), dtype=np.float32)
        stats = []
        for idx, (text, length) in enumerate(zip(texts, lengths)):
            coarse, item_stats = self.compose(text, length)
            prior[idx, :length] = coarse[:length]
            stats.append(item_stats)
        tensor = torch.from_numpy(prior)
        if device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        elif dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor, stats
