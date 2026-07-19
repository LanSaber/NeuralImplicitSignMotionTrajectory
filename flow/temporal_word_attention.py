import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from flow.smplx_features import fit_length


def masked_mean(x, mask, dim=1, eps=1e-8):
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(float(eps))


@dataclass
class WordCandidateBatch:
    motion: torch.Tensor
    frame_mask: torch.Tensor
    candidate_mask: torch.Tensor
    labels: torch.Tensor
    group_ids: torch.Tensor
    group_mask: torch.Tensor
    texts: list
    names: list
    group_texts: list
    stats: list

    def to(self, device):
        """Move tensor fields to `device`; list fields are returned unchanged.

        Used when the batch is built on CPU in a DataLoader worker and consumed
        on the GPU in the training loop.
        """
        return WordCandidateBatch(
            motion=self.motion.to(device),
            frame_mask=self.frame_mask.to(device),
            candidate_mask=self.candidate_mask.to(device),
            labels=self.labels.to(device),
            group_ids=self.group_ids.to(device),
            group_mask=self.group_mask.to(device),
            texts=self.texts,
            names=self.names,
            group_texts=self.group_texts,
            stats=self.stats,
        )


class WordCandidateBuilder:
    """Build unordered word-pose candidates for the soft arranger."""

    def __init__(
        self,
        prior_builder,
        num_word_candidates=32,
        num_negative_candidates=16,
        candidate_selection="flat",
        max_positive_variants_per_key=0,
        seed=42,
    ):
        self.prior_builder = prior_builder
        self.num_word_candidates = int(num_word_candidates)
        self.num_negative_candidates = int(num_negative_candidates)
        self.candidate_selection = str(candidate_selection or "flat").strip().lower()
        if self.candidate_selection not in {"flat", "round_robin"}:
            raise ValueError(
                "candidate_selection must be one of {'flat', 'round_robin'}, "
                f"got {candidate_selection!r}."
            )
        self.max_positive_variants_per_key = int(max_positive_variants_per_key or 0)
        if self.max_positive_variants_per_key < 0:
            raise ValueError("max_positive_variants_per_key must be >= 0.")
        self.rng = random.Random(int(seed))
        self.entries = list(prior_builder.entries)
        if not self.entries:
            raise RuntimeError("WordCandidateBuilder needs at least one word-prior entry.")

    @staticmethod
    def _entry_key(entry):
        key = entry.get("lexicon_key")
        if key:
            return str(key)
        tokens = entry.get("tokens")
        if tokens:
            return " ".join(str(token) for token in tokens)
        return str(entry.get("name", ""))

    def _positive_groups(self, positives, spans):
        groups = []
        if spans:
            offset = 0
            for group_id, span in enumerate(spans):
                count = int(span.get("variant_count") or 0)
                entries = positives[offset : offset + count]
                offset += count
                if entries:
                    groups.append(
                        {
                            "group_id": int(group_id),
                            "key": str(span.get("text") or self._entry_key(entries[0])),
                            "entries": list(entries),
                        }
                    )
            if offset < len(positives):
                for entry in positives[offset:]:
                    groups.append(
                        {
                            "group_id": len(groups),
                            "key": self._entry_key(entry),
                            "entries": [entry],
                        }
                    )
            return groups

        by_key = {}
        order = []
        for entry in positives:
            key = self._entry_key(entry)
            if key not in by_key:
                by_key[key] = []
                order.append(key)
            by_key[key].append(entry)
        return [
            {"group_id": group_id, "key": key, "entries": by_key[key]}
            for group_id, key in enumerate(order)
        ]

    def _cap_group(self, entries, shuffle):
        max_variants = self.max_positive_variants_per_key
        if max_variants <= 0 or len(entries) <= max_variants:
            return list(entries)
        if shuffle:
            return self.rng.sample(list(entries), max_variants)
        return list(entries[:max_variants])

    def _select_positives(self, positives, spans, max_pos, shuffle):
        if max_pos <= 0:
            return []
        groups = self._positive_groups(positives, spans)
        capped_groups = [
            {
                "group_id": int(group["group_id"]),
                "entries": self._cap_group(group["entries"], shuffle),
            }
            for group in groups
        ]
        capped_groups = [group for group in capped_groups if group["entries"]]
        if self.candidate_selection == "flat":
            selected = [
                (entry, group["group_id"])
                for group in capped_groups
                for entry in group["entries"]
            ]
            return selected[:max_pos]

        selected = []
        max_group_len = max((len(group["entries"]) for group in capped_groups), default=0)
        for variant_idx in range(max_group_len):
            for group in capped_groups:
                if variant_idx < len(group["entries"]):
                    selected.append((group["entries"][variant_idx], group["group_id"]))
                    if len(selected) >= max_pos:
                        return selected
        return selected

    def _negative_entries(self, positives, count, exclude_entries=None):
        if count <= 0:
            return []
        excluded = positives if exclude_entries is None else exclude_entries
        positive_ids = {entry.get("entry_id") for entry in excluded}
        positive_keys = {tuple(entry.get("tokens", ())) for entry in excluded}
        n = len(self.entries)
        # Fast path: rejection-sample unique negatives without scanning the whole
        # dictionary. The previous list-comprehension over self.entries was
        # O(num_entries) PER text, i.e. O(batch_size * num_entries) per training
        # step -- the dominant CPU cost on large word dictionaries (e.g. signasl's
        # 113k clips), which left the GPUs idle. Positives are few, so collisions
        # are negligible and this is effectively O(count).
        result = []
        seen_ids = set()
        max_attempts = count * 50 + 100
        attempts = 0
        while len(result) < count and attempts < max_attempts:
            attempts += 1
            entry = self.entries[self.rng.randrange(n)]
            eid = entry.get("entry_id")
            if eid in positive_ids or eid in seen_ids:
                continue
            if tuple(entry.get("tokens", ())) in positive_keys:
                continue
            seen_ids.add(eid)
            result.append(entry)
        if len(result) >= count:
            return result
        # Fallback for tiny/degenerate dictionaries where rejection cannot fill the
        # quota: exhaustive filter (original behaviour).
        pool = [
            entry
            for entry in self.entries
            if entry.get("entry_id") not in positive_ids
            and tuple(entry.get("tokens", ())) not in positive_keys
            and entry.get("entry_id") not in seen_ids
        ]
        if not pool:
            pool = [
                entry
                for entry in self.entries
                if entry.get("entry_id") not in positive_ids and entry.get("entry_id") not in seen_ids
            ]
        if not pool:
            pool = list(self.entries)
        needed = count - len(result)
        if len(pool) >= needed:
            result.extend(self.rng.sample(pool, needed))
        else:
            result.extend(self.rng.choice(pool) for _ in range(needed))
        return result

    def candidates_for_text(self, text, shuffle=True):
        if hasattr(self.prior_builder, "match_text_variants"):
            tokens, positives, spans = self.prior_builder.match_text_variants(text)
        else:
            tokens, positives = self.prior_builder.match_text(text)
            spans = []
        positives = list(positives)
        available_positives = list(positives)
        matched_span_count = len(spans) if spans else len(positives)
        available_positive_count = len(positives)
        max_pos = max(self.num_word_candidates - self.num_negative_candidates, 0)
        positive_groups = self._positive_groups(positives, spans)
        selected_positives = self._select_positives(positives, spans, max_pos, shuffle=shuffle)

        remaining = max(self.num_word_candidates - len(selected_positives), 0)
        if self.num_negative_candidates <= 0:
            neg_count = 0
        else:
            neg_count = min(max(self.num_negative_candidates, remaining), remaining)
        negatives = self._negative_entries(
            [entry for entry, _group_id in selected_positives],
            neg_count,
            exclude_entries=available_positives,
        )
        candidates = [
            (entry, 1.0, int(group_id))
            for entry, group_id in selected_positives
        ] + [(entry, 0.0, -1) for entry in negatives]

        if len(candidates) > self.num_word_candidates:
            candidates = candidates[: self.num_word_candidates]
        if shuffle:
            self.rng.shuffle(candidates)
        selected_group_ids = sorted(
            {int(group_id) for _entry, label, group_id in candidates if label > 0.5 and int(group_id) >= 0}
        )
        group_text_by_id = {int(group["group_id"]): str(group["key"]) for group in positive_groups}
        max_group_id = max(selected_group_ids, default=-1)
        group_texts = ["" for _ in range(max_group_id + 1)]
        for group_id in selected_group_ids:
            group_texts[group_id] = group_text_by_id.get(group_id, "")
        positive_entries = [entry for entry, _group_id in selected_positives]

        stats = {
            "tokens": tokens,
            "matched": [entry["name"] for entry in positive_entries],
            "matched_lexicon_keys": [entry.get("lexicon_key", entry["name"]) for entry in positive_entries],
            "matched_spans": spans,
            "matched_count": matched_span_count,
            "available_positive_variant_count": available_positive_count,
            "available_positive_group_count": len(positive_groups),
            "positive_variant_count": len(selected_positives),
            "positive_group_count": len(selected_group_ids),
            "candidate_group_ids": [int(group_id) for _entry, _label, group_id in candidates],
            "selected_group_ids": selected_group_ids,
            "group_texts": group_texts,
            "candidate_count": len(candidates),
            "positive_count": len(selected_positives),
            "negative_count": sum(1 for _entry, label, _group_id in candidates if label < 0.5),
            "token_count": len(tokens),
            "coverage": float(matched_span_count / max(len(tokens), 1)),
            "candidate_selection": self.candidate_selection,
            "max_positive_variants_per_key": self.max_positive_variants_per_key,
        }
        return candidates, stats

    def batch(self, texts, device=None, dtype=torch.float32, shuffle=True, max_motion_frames=None):
        batch_candidates = []
        stats = []
        max_frames = 1
        max_groups = 0
        dim = self.prior_builder.dim
        max_motion_frames = int(max_motion_frames) if max_motion_frames is not None else None
        if max_motion_frames is not None and max_motion_frames <= 0:
            max_motion_frames = None
        for text in texts:
            candidates, item_stats = self.candidates_for_text(text, shuffle=shuffle)
            batch_candidates.append(candidates)
            stats.append(item_stats)
            max_groups = max(max_groups, len(item_stats.get("group_texts") or []))
            for entry, _label, _group_id in candidates:
                frame_count = int(entry.get("motion_frames") or self.prior_builder.entry_motion(entry).shape[0])
                if max_motion_frames is not None:
                    frame_count = min(frame_count, max_motion_frames)
                max_frames = max(max_frames, frame_count)

        batch_size = len(texts)
        k = self.num_word_candidates
        motion = np.zeros((batch_size, k, max_frames, dim), dtype=np.float32)
        frame_mask = np.zeros((batch_size, k, max_frames), dtype=bool)
        candidate_mask = np.zeros((batch_size, k), dtype=bool)
        labels = np.zeros((batch_size, k), dtype=np.float32)
        group_ids = np.full((batch_size, k), -1, dtype=np.int64)
        group_mask = np.zeros((batch_size, max_groups), dtype=bool)
        candidate_texts = []
        candidate_names = []
        candidate_group_texts = []

        mean = self.prior_builder.target_mean.reshape(1, dim)
        std = self.prior_builder.target_std.reshape(1, dim)
        for batch_idx, candidates in enumerate(batch_candidates):
            text_row = []
            name_row = []
            group_text_row = list(stats[batch_idx].get("group_texts") or [])
            while len(group_text_row) < max_groups:
                group_text_row.append("")
            for group_idx, group_text in enumerate(group_text_row[:max_groups]):
                if group_text:
                    group_mask[batch_idx, group_idx] = True
            for cand_idx, (entry, label, group_id) in enumerate(candidates[:k]):
                word_motion = self.prior_builder.entry_motion(entry).astype(np.float32, copy=False)
                if max_motion_frames is not None and word_motion.shape[0] > max_motion_frames:
                    valid = np.ones(word_motion.shape[0], dtype=np.float32)
                    word_motion, _left_valid, _right_valid = fit_length(
                        word_motion,
                        valid,
                        valid,
                        max_motion_frames,
                    )
                length = min(int(word_motion.shape[0]), max_frames)
                motion[batch_idx, cand_idx, :length] = (word_motion[:length] - mean) / std
                frame_mask[batch_idx, cand_idx, :length] = True
                candidate_mask[batch_idx, cand_idx] = True
                labels[batch_idx, cand_idx] = float(label)
                group_ids[batch_idx, cand_idx] = int(group_id)
                text_row.append(str(entry.get("lexicon_key", entry["name"])).replace("_", " "))
                name_row.append(str(entry["name"]))
            while len(text_row) < k:
                text_row.append("")
                name_row.append("")
            candidate_texts.append(text_row)
            candidate_names.append(name_row)
            candidate_group_texts.append(group_text_row[:max_groups])

        out = WordCandidateBatch(
            motion=torch.from_numpy(motion),
            frame_mask=torch.from_numpy(frame_mask),
            candidate_mask=torch.from_numpy(candidate_mask),
            labels=torch.from_numpy(labels),
            group_ids=torch.from_numpy(group_ids),
            group_mask=torch.from_numpy(group_mask),
            texts=candidate_texts,
            names=candidate_names,
            group_texts=candidate_group_texts,
            stats=stats,
        )
        if device is not None:
            out.motion = out.motion.to(device=device, dtype=dtype)
            out.frame_mask = out.frame_mask.to(device=device)
            out.candidate_mask = out.candidate_mask.to(device=device)
            out.labels = out.labels.to(device=device, dtype=dtype)
            out.group_ids = out.group_ids.to(device=device)
            out.group_mask = out.group_mask.to(device=device)
        elif dtype is not None:
            out.motion = out.motion.to(dtype=dtype)
            out.labels = out.labels.to(dtype=dtype)
        return out


class SoftWordArranger(nn.Module):
    """Arrange unordered word-clip latent memory into a sentence-length latent prior."""

    def __init__(
        self,
        latent_dim=256,
        text_dim=768,
        hidden_dim=512,
        num_heads=8,
        dropout=0.0,
        max_frames=100,
        max_word_latent_frames=64,
        use_candidate_gates=True,
        use_null_memory=True,
        use_word_text_features=True,
        use_word_motion_latents=True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.max_frames = int(max_frames)
        self.max_word_latent_frames = int(max_word_latent_frames)
        self.use_candidate_gates = bool(use_candidate_gates)
        self.use_null_memory = bool(use_null_memory)
        self.use_word_text_features = bool(use_word_text_features)
        self.use_word_motion_latents = bool(use_word_motion_latents)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("arranger hidden_dim must be divisible by num_heads.")
        self.head_dim = self.hidden_dim // self.num_heads

        self.sentence_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.hidden_dim),
        )
        self.word_text_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.hidden_dim),
        )
        self.word_motion_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
        )
        self.query_pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, self.hidden_dim))
        self.word_phase_embed = nn.Parameter(torch.zeros(1, 1, self.max_word_latent_frames, self.hidden_dim))

        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.null_memory = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.query_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.word_phase_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.null_memory, mean=0.0, std=0.02)
        last = self.out_proj[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.bias)

    def _split_heads(self, x):
        batch, length, _dim = x.shape
        x = x.view(batch, length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        sentence_text,
        word_text,
        word_latents,
        word_latent_mask,
        candidate_mask,
        target_latent_mask,
    ):
        if sentence_text.ndim != 2:
            raise ValueError(f"Expected sentence_text [B,Dt], got {tuple(sentence_text.shape)}")
        if word_text.ndim != 3:
            raise ValueError(f"Expected word_text [B,K,Dt], got {tuple(word_text.shape)}")
        if word_latents.ndim != 4:
            raise ValueError(f"Expected word_latents [B,K,Lw,Dz], got {tuple(word_latents.shape)}")
        batch, k, word_len, latent_dim = word_latents.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        if target_latent_mask.shape[1] > self.max_frames:
            raise ValueError(f"Target latent length {target_latent_mask.shape[1]} exceeds max_frames={self.max_frames}")
        if word_len > self.max_word_latent_frames:
            raise ValueError(
                f"Word latent length {word_len} exceeds max_word_latent_frames={self.max_word_latent_frames}"
            )

        target_latent_mask = target_latent_mask.to(device=word_latents.device, dtype=torch.bool)
        word_latent_mask = word_latent_mask.to(device=word_latents.device, dtype=torch.bool)
        candidate_mask = candidate_mask.to(device=word_latents.device, dtype=torch.bool)

        sent_h = self.sentence_proj(sentence_text)
        word_text_h = self.word_text_proj(word_text)
        pooled_motion = masked_mean(
            word_latents,
            word_latent_mask,
            dim=2,
        )
        pooled_motion_h = self.word_motion_proj(pooled_motion)
        if not self.use_word_text_features:
            word_text_h = torch.zeros_like(word_text_h)
        if not self.use_word_motion_latents:
            pooled_motion_h = torch.zeros_like(pooled_motion_h)

        sent_for_gate = sent_h[:, None, :].expand(-1, k, -1)
        gate_features = torch.cat(
            [
                sent_for_gate,
                word_text_h,
                pooled_motion_h,
                sent_for_gate * word_text_h,
            ],
            dim=-1,
        )
        word_gate_logits = self.gate_mlp(gate_features).squeeze(-1)
        word_gate_logits = word_gate_logits.masked_fill(~candidate_mask, -1e4)
        if self.use_candidate_gates:
            word_gate_probs = torch.sigmoid(word_gate_logits) * candidate_mask.to(word_gate_logits.dtype)
        else:
            word_gate_logits = torch.zeros_like(word_gate_logits).masked_fill(~candidate_mask, -1e4)
            word_gate_probs = candidate_mask.to(word_gate_logits.dtype)

        word_motion_h = self.word_motion_proj(word_latents)
        if not self.use_word_motion_latents:
            word_motion_h = torch.zeros_like(word_motion_h)
        memory = word_motion_h + word_text_h[:, :, None, :]
        memory = memory + self.word_phase_embed[:, :, :word_len, :]
        memory = memory.reshape(batch, k * word_len, self.hidden_dim)
        token_mask = (word_latent_mask & candidate_mask[:, :, None]).reshape(batch, k * word_len)

        if self.use_null_memory:
            null_memory = self.null_memory.expand(batch, -1, -1)
            memory = torch.cat([memory, null_memory], dim=1)
            null_mask = torch.ones(batch, 1, dtype=torch.bool, device=word_latents.device)
            full_mask = torch.cat([token_mask, null_mask], dim=1)
        else:
            full_mask = token_mask

        target_len = target_latent_mask.shape[1]
        query = self.query_pos_embed[:, :target_len, :] + sent_h[:, None, :]
        q = self._split_heads(self.query_proj(query))
        key = self._split_heads(self.key_proj(memory))
        value = self._split_heads(self.value_proj(memory))

        scores = torch.einsum("bhld,bhnd->bhln", q, key) / (self.head_dim ** 0.5)
        if self.use_candidate_gates:
            gate_bias = torch.log(word_gate_probs.clamp_min(1e-6))
            gate_bias = gate_bias[:, :, None].expand(batch, k, word_len).reshape(batch, k * word_len)
        else:
            gate_bias = torch.zeros(batch, k * word_len, device=word_latents.device, dtype=scores.dtype)
        if self.use_null_memory:
            gate_bias = torch.cat(
                [gate_bias, torch.zeros(batch, 1, device=word_latents.device, dtype=gate_bias.dtype)],
                dim=1,
            )
        scores = scores + gate_bias[:, None, None, :]
        scores = scores.masked_fill(~full_mask[:, None, None, :], -1e4)
        scores = scores.masked_fill(~target_latent_mask[:, None, :, None], -1e4)

        attention_heads = torch.softmax(scores, dim=-1)
        attention_heads = F.dropout(attention_heads, p=self.dropout, training=self.training)
        attended = torch.einsum("bhln,bhnd->bhld", attention_heads, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, target_len, self.hidden_dim)
        z_prior_aligned = self.out_proj(attended)
        z_prior_aligned = z_prior_aligned * target_latent_mask.unsqueeze(-1).to(z_prior_aligned.dtype)

        attention = attention_heads.mean(dim=1)
        word_attention = attention[:, :, : k * word_len].reshape(batch, target_len, k, word_len)
        if self.use_null_memory:
            null_attention = attention[:, :, k * word_len]
        else:
            null_attention = attention.new_zeros(batch, target_len)
        valid_target = target_latent_mask.to(attention.dtype)
        denom = valid_target.sum(dim=1, keepdim=True).clamp_min(1.0)
        word_usage = (word_attention * valid_target[:, :, None, None]).sum(dim=(1, 3)) / denom
        if self.use_null_memory:
            null_usage = (null_attention * valid_target).sum(dim=1) / denom.squeeze(1)
        else:
            null_usage = null_attention.new_zeros(batch)

        return {
            "z_prior_aligned": z_prior_aligned,
            "attention": word_attention,
            "null_attention": null_attention,
            "word_gate_logits": word_gate_logits,
            "word_gate_probs": word_gate_probs,
            "word_usage": word_usage,
            "null_usage": null_usage,
        }


def build_arranger_from_config(config):
    return SoftWordArranger(**dict(config))
