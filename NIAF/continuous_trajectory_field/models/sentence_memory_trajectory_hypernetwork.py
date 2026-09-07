from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn

from NIAF.continuous_trajectory_field.models.dual_mode_trajectory_hypernetwork import (
    PART_COUNT,
    DualModeTrajectoryHypernetwork,
    _slot_time_features,
)
from NIAF.continuous_trajectory_field.models.trajectory_instance import (
    TrajectoryInstance,
)


SENTENCE_RETRIEVAL_FEATURE_DIM = 5
SENTENCE_CONFIDENCE_FEATURE_DIM = 4
LEGACY_SENTENCE_KEY_VALUE_MODE = "legacy_mixed_v1"
FACTORIZED_SENTENCE_KEY_VALUE_MODE = "factorized_metadata_motion_v1"
SENTENCE_KEY_VALUE_MODES = frozenset(
    {
        LEGACY_SENTENCE_KEY_VALUE_MODE,
        FACTORIZED_SENTENCE_KEY_VALUE_MODE,
    }
)
SENTENCE_TEMPORAL_PRIOR_MODES = frozenset({"none", "gaussian"})
SENTENCE_ATTENTION_MODES = frozenset({"learned", "analytic_prior"})


def _masked_candidate_softmax(logits: torch.Tensor, mask: torch.Tensor):
    """Softmax over candidates while allowing rows with no valid candidate."""

    masked = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1)
    probabilities = torch.where(
        mask,
        torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(probabilities),
    )
    return probabilities


def _token_tau(mask: torch.Tensor, dtype: torch.dtype):
    """Return per-candidate normalized token times for a padded `[B,K,U]` mask."""

    token_count = mask.shape[-1]
    positions = torch.arange(token_count, device=mask.device, dtype=dtype)
    lengths = mask.sum(dim=-1).clamp_min(1).to(dtype)
    denominator = (lengths - 1.0).clamp_min(1.0)
    tau = -1.0 + 2.0 * positions.view(1, 1, -1) / denominator.unsqueeze(-1)
    tau = torch.where(
        lengths.unsqueeze(-1) <= 1.0,
        torch.zeros_like(tau),
        tau,
    )
    return tau * mask.to(dtype)


class SentenceMemoryCrossAttention(nn.Module):
    """Cross-attend text-plan slots to unaligned sentence-motion tokens.

    The null entry has a learned key but an exactly zero value. Consequently,
    attention can reject all retrieved motion without injecting a learned null
    vector into the trajectory plan.
    """

    def __init__(self, hidden_dim: int, head_count: int, dropout: float):
        super().__init__()
        hidden_dim = int(hidden_dim)
        head_count = int(head_count)
        if hidden_dim % head_count:
            raise ValueError("sentence-memory hidden_dim must be divisible by head_count")
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        # No output bias: zero real-memory mass must yield an exact zero update.
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.null_key = nn.Parameter(torch.zeros(head_count, self.head_dim))
        self.null_logit_bias = nn.Parameter(torch.zeros(head_count))
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output_dropout = nn.Dropout(float(dropout))

        self.state_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
        )
        nn.init.normal_(self.null_key, mean=0.0, std=0.02)

    def _heads(self, values: torch.Tensor):
        batch, length, _hidden = values.shape
        return values.reshape(batch, length, self.head_count, self.head_dim).transpose(1, 2)

    def forward(
        self,
        text_slots: torch.Tensor,
        state: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_log_prior: torch.Tensor,
        null_confidence_bias: torch.Tensor,
        candidate_count: int,
        token_count: int,
    ):
        query = self._heads(
            self.query_projection(self.query_norm(text_slots + state))
        )
        normalized_memory = self.memory_norm(memory)
        key = self._heads(self.key_projection(normalized_memory))
        value = self._heads(self.value_projection(normalized_memory))

        real_logits = torch.matmul(query, key.transpose(-1, -2))
        real_logits = real_logits / math.sqrt(float(self.head_dim))
        real_logits = real_logits + candidate_log_prior[:, None, None, :]
        real_logits = real_logits.masked_fill(
            ~memory_mask[:, None, None, :],
            -torch.inf,
        )
        null_logits = torch.einsum("bhsd,hd->bhs", query, self.null_key)
        null_logits = null_logits / math.sqrt(float(self.head_dim))
        null_logits = null_logits + self.null_logit_bias[None, :, None]
        null_logits = null_logits + null_confidence_bias[:, None, None]

        probabilities = torch.softmax(
            torch.cat([null_logits.unsqueeze(-1), real_logits], dim=-1),
            dim=-1,
        )
        null_mass = probabilities[..., 0].mean(dim=1)
        real_probabilities = probabilities[..., 1:]
        dropped_probabilities = self.attention_dropout(real_probabilities)
        update = torch.matmul(dropped_probabilities, value)
        update = update.transpose(1, 2).reshape_as(state)
        state = state + self.output_dropout(self.output_projection(update))
        state = state + self.output_dropout(self.feed_forward(self.state_norm(state)))

        real_mean = real_probabilities.mean(dim=1)
        token_mass = real_mean.reshape(
            real_mean.shape[0],
            real_mean.shape[1],
            int(candidate_count),
            int(token_count),
        )
        candidate_mass = token_mass.sum(dim=-1)
        return state, null_mass, candidate_mass, token_mass


# Backward-compatible internal name used by early v3 development snapshots.
SentenceMemoryAttentionBlock = SentenceMemoryCrossAttention


class FactorizedSentenceMemoryCrossAttention(nn.Module):
    """Attend with metadata-only keys and motion-only, zero-preserving values."""

    def __init__(self, hidden_dim: int, head_count: int, dropout: float):
        super().__init__()
        hidden_dim = int(hidden_dim)
        head_count = int(head_count)
        if hidden_dim % head_count:
            raise ValueError("sentence-memory hidden_dim must be divisible by head_count")
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count

        # Query/key affine parameters cannot inject motion into the value path.
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)

        # Everything downstream of a motion value is exactly zero-preserving.
        self.value_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
        )

        self.null_key = nn.Parameter(torch.zeros(head_count, self.head_dim))
        self.null_logit_bias = nn.Parameter(torch.zeros(head_count))
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output_dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.null_key, mean=0.0, std=0.02)

    def _heads(self, values: torch.Tensor):
        batch, length, _hidden = values.shape
        return values.reshape(
            batch,
            length,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        text_slots: torch.Tensor,
        state: torch.Tensor,
        candidate_keys: torch.Tensor,
        motion_values: torch.Tensor,
        token_log_prior: torch.Tensor,
        token_mask: torch.Tensor,
        null_confidence_bias: torch.Tensor,
        candidate_count: int,
        token_count: int,
        *,
        attention_mode: str = "learned",
    ):
        """Return the same four-tensor hook contract as the legacy block.

        ``text_slots`` is the immutable text/part query for every layer. State
        carries motion evidence between layers, but is deliberately excluded
        from query construction.
        """

        attention_mode = str(attention_mode).lower()
        if attention_mode not in SENTENCE_ATTENTION_MODES:
            raise ValueError(
                "sentence-memory attention_mode must be one of "
                f"{sorted(SENTENCE_ATTENTION_MODES)}, got {attention_mode!r}"
            )
        batch, query_count, _hidden = text_slots.shape
        if candidate_keys.shape != (batch, int(candidate_count), self.hidden_dim):
            raise ValueError("candidate_keys must have shape [B,K,H]")
        if motion_values.shape != (
            batch,
            int(candidate_count),
            int(token_count),
            self.hidden_dim,
        ):
            raise ValueError("motion_values must have shape [B,K,U,H]")
        expected_prior_shape = (
            batch,
            query_count,
            int(candidate_count),
            int(token_count),
        )
        if token_log_prior.shape != expected_prior_shape:
            raise ValueError("token_log_prior must have shape [B,Q,K,U]")
        if token_mask.shape != expected_prior_shape:
            raise ValueError("token_mask must have shape [B,Q,K,U]")

        query = self._heads(
            self.query_projection(self.query_norm(text_slots))
        )
        key = self._heads(
            self.key_projection(self.key_norm(candidate_keys))
        )
        value = self._heads(
            self.value_projection(
                self.value_norm(
                    motion_values.reshape(
                        batch,
                        int(candidate_count) * int(token_count),
                        self.hidden_dim,
                    )
                )
            )
        )

        if attention_mode == "analytic_prior":
            candidate_logits = query.new_zeros(
                batch,
                self.head_count,
                query_count,
                int(candidate_count),
            )
        else:
            candidate_logits = torch.matmul(query, key.transpose(-1, -2))
            candidate_logits = candidate_logits / math.sqrt(float(self.head_dim))
        real_logits = candidate_logits.unsqueeze(-1)
        real_logits = real_logits + token_log_prior[:, None, :, :, :]
        real_logits = real_logits.masked_fill(
            ~token_mask[:, None, :, :, :],
            -torch.inf,
        )
        real_logits = real_logits.reshape(
            batch,
            self.head_count,
            query_count,
            int(candidate_count) * int(token_count),
        )

        null_logits = torch.einsum("bhqd,hd->bhq", query, self.null_key)
        null_logits = null_logits / math.sqrt(float(self.head_dim))
        null_logits = null_logits + self.null_logit_bias[None, :, None]
        null_logits = null_logits + null_confidence_bias[:, None, None]
        probabilities = torch.softmax(
            torch.cat([null_logits.unsqueeze(-1), real_logits], dim=-1),
            dim=-1,
        )
        null_mass = probabilities[..., 0].mean(dim=1)
        real_probabilities = probabilities[..., 1:]
        dropped_probabilities = self.attention_dropout(real_probabilities)
        update = torch.matmul(dropped_probabilities, value)
        update = update.transpose(1, 2).reshape_as(state)
        state = state + self.output_dropout(self.output_projection(update))
        state = state + self.output_dropout(
            self.feed_forward(self.state_norm(state))
        )

        real_mean = real_probabilities.mean(dim=1)
        token_mass = real_mean.reshape(
            batch,
            query_count,
            int(candidate_count),
            int(token_count),
        )
        candidate_mass = token_mass.sum(dim=-1)
        return state, null_mass, candidate_mass, token_mass


class SentenceMemorySlotEncoder(nn.Module):
    """Softly align top-K sentence-motion memories to the target text slots."""

    def __init__(
        self,
        motion_dim: int,
        key_dim: int,
        hidden_dim: int,
        layer_count: int,
        head_count: int,
        dropout: float,
        score_temperature: float = 0.10,
        duration_weight: float = 0.10,
        retrieval_prior_scale: float = 1.0,
        key_value_mode: str | None = None,
        temporal_prior_mode: str = "none",
        temporal_prior_sigma: float = 0.25,
        temporal_prior_scale: float = 1.0,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.key_dim = int(key_dim)
        self.hidden_dim = int(hidden_dim)
        self.score_temperature = max(float(score_temperature), 1e-4)
        self.duration_weight = max(float(duration_weight), 0.0)
        self.retrieval_prior_scale = float(retrieval_prior_scale)
        self.key_value_mode = (
            LEGACY_SENTENCE_KEY_VALUE_MODE
            if key_value_mode is None
            else str(key_value_mode).lower()
        )
        if self.key_value_mode not in SENTENCE_KEY_VALUE_MODES:
            raise ValueError(
                "sentence-memory key_value_mode must be one of "
                f"{sorted(SENTENCE_KEY_VALUE_MODES)}, got {self.key_value_mode!r}"
            )
        self.temporal_prior_mode = str(temporal_prior_mode).lower()
        if self.temporal_prior_mode not in SENTENCE_TEMPORAL_PRIOR_MODES:
            raise ValueError(
                "sentence-memory temporal_prior_mode must be one of "
                f"{sorted(SENTENCE_TEMPORAL_PRIOR_MODES)}, "
                f"got {self.temporal_prior_mode!r}"
            )
        self.temporal_prior_sigma = float(temporal_prior_sigma)
        if not math.isfinite(self.temporal_prior_sigma) or self.temporal_prior_sigma <= 0:
            raise ValueError("sentence-memory temporal_prior_sigma must be positive")
        self.temporal_prior_scale = float(temporal_prior_scale)
        if not math.isfinite(self.temporal_prior_scale) or self.temporal_prior_scale < 0:
            raise ValueError("sentence-memory temporal_prior_scale must be non-negative")

        if self.key_value_mode == LEGACY_SENTENCE_KEY_VALUE_MODE:
            # Keep this branch byte-for-byte equivalent in topology and module
            # registration order to existing v3 checkpoints.
            self.motion_projection = nn.Sequential(
                nn.LayerNorm(self.motion_dim),
                nn.Linear(self.motion_dim, self.hidden_dim),
            )
            self.key_projection = nn.Sequential(
                nn.LayerNorm(self.key_dim),
                nn.Linear(self.key_dim, self.hidden_dim),
            )
            self.time_projection = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.feature_projection = nn.Sequential(
                nn.Linear(SENTENCE_RETRIEVAL_FEATURE_DIM, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.memory_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.motion_projection = nn.Sequential(
                nn.LayerNorm(self.motion_dim, elementwise_affine=False),
                nn.Linear(self.motion_dim, self.hidden_dim, bias=False),
            )
            self.key_projection = nn.Sequential(
                nn.LayerNorm(self.key_dim),
                nn.Linear(self.key_dim, self.hidden_dim),
            )
            self.feature_projection = nn.Sequential(
                nn.Linear(SENTENCE_RETRIEVAL_FEATURE_DIM, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.part_query_embeddings = nn.Parameter(
            torch.zeros(PART_COUNT, self.hidden_dim)
        )
        self.part_query_projections = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                    bias=self.key_value_mode == LEGACY_SENTENCE_KEY_VALUE_MODE,
                )
                for _ in range(PART_COUNT)
            ]
        )
        self.null_confidence = nn.Sequential(
            nn.LayerNorm(SENTENCE_CONFIDENCE_FEATURE_DIM),
            nn.Linear(SENTENCE_CONFIDENCE_FEATURE_DIM, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.layers = nn.ModuleList(
            [
                (
                    SentenceMemoryCrossAttention(
                        hidden_dim=self.hidden_dim,
                        head_count=int(head_count),
                        dropout=float(dropout),
                    )
                    if self.key_value_mode == LEGACY_SENTENCE_KEY_VALUE_MODE
                    else FactorizedSentenceMemoryCrossAttention(
                        hidden_dim=self.hidden_dim,
                        head_count=int(head_count),
                        dropout=float(dropout),
                    )
                )
                for _ in range(max(int(layer_count), 1))
            ]
        )
        nn.init.normal_(self.part_query_embeddings, mean=0.0, std=0.02)

    def validate_inputs(
        self,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_keys: torch.Tensor,
        scores: torch.Tensor,
        durations: torch.Tensor,
        candidate_mask: torch.Tensor,
        motion_tau: torch.Tensor | None = None,
        part_validity: torch.Tensor | None = None,
    ):
        if motion_tokens.ndim != 4 or motion_tokens.shape[-1] != self.motion_dim:
            raise ValueError(
                "sentence_motion_tokens must have shape "
                f"[B,K,U,{self.motion_dim}]"
            )
        batch, candidates, tokens, _dimension = motion_tokens.shape
        if candidates < 1 or tokens < 1:
            raise ValueError("sentence memory requires at least one candidate and token")
        if motion_mask.shape != (batch, candidates, tokens):
            raise ValueError("sentence_motion_mask must have shape [B,K,U]")
        if text_keys.shape != (batch, candidates, self.key_dim):
            raise ValueError(
                f"sentence_text_keys must have shape [B,K,{self.key_dim}]"
            )
        for name, value in (
            ("sentence_scores", scores),
            ("sentence_durations", durations),
            ("sentence_candidate_mask", candidate_mask),
        ):
            if value.shape != (batch, candidates):
                raise ValueError(f"{name} must have shape [B,K]")
        if motion_tau is not None and motion_tau.shape != (batch, candidates, tokens):
            raise ValueError("sentence_motion_tau must have shape [B,K,U]")
        if part_validity is not None and part_validity.shape != (
            batch,
            candidates,
            tokens,
            PART_COUNT,
        ):
            raise ValueError("sentence_part_validity must have shape [B,K,U,4]")
        return batch, candidates, tokens

    def _retrieval_statistics(
        self,
        scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        query_duration: torch.Tensor,
        candidate_durations: torch.Tensor,
    ):
        duration_gap = torch.abs(
            torch.log(
                query_duration.detach().clamp_min(1e-4)[:, None]
                / candidate_durations.clamp_min(1e-4)
            )
        )
        adjusted_scores = scores - self.duration_weight * duration_gap
        probabilities = _masked_candidate_softmax(
            adjusted_scores / self.score_temperature,
            candidate_mask,
        )
        valid_count = candidate_mask.sum(dim=-1)
        masked_adjusted = adjusted_scores.masked_fill(~candidate_mask, -torch.inf)
        top_count = min(int(adjusted_scores.shape[1]), 2)
        top_values = torch.topk(masked_adjusted, k=top_count, dim=-1).values
        best = torch.where(
            valid_count > 0,
            top_values[:, 0],
            torch.zeros_like(top_values[:, 0]),
        )
        if top_count > 1:
            margin = torch.where(
                valid_count > 1,
                top_values[:, 0] - top_values[:, 1],
                torch.zeros_like(best),
            )
        else:
            margin = torch.zeros_like(best)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-8))
        ).sum(dim=-1)
        minimum_gap = duration_gap.masked_fill(~candidate_mask, torch.inf).amin(dim=-1)
        minimum_gap = torch.where(
            valid_count > 0,
            minimum_gap,
            torch.zeros_like(minimum_gap),
        )
        confidence = torch.stack([best, margin, entropy, minimum_gap], dim=-1)
        return adjusted_scores, probabilities, duration_gap, confidence

    def _part_queries(self, text_slots: torch.Tensor):
        batch, slot_count, _hidden = text_slots.shape
        queries = torch.stack(
            [
                projection(text_slots) + self.part_query_embeddings[index]
                for index, projection in enumerate(self.part_query_projections)
            ],
            dim=2,
        )
        return queries.reshape(
            batch,
            slot_count * PART_COUNT,
            self.hidden_dim,
        )

    def _factorized_candidate_log_prior(
        self,
        adjusted_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ):
        logits = adjusted_scores / self.score_temperature
        masked_logits = logits.masked_fill(~candidate_mask, -torch.inf)
        normalizer = torch.logsumexp(masked_logits, dim=-1, keepdim=True)
        has_candidate = candidate_mask.any(dim=-1, keepdim=True)
        # Avoid forming ``-inf - -inf`` for all-null rows.  Although the
        # subsequent where masks those values in the forward pass, the NaN can
        # otherwise leak through logsumexp's backward when candidate dropout
        # removes every real memory.
        safe_normalizer = torch.where(
            has_candidate,
            normalizer,
            torch.zeros_like(normalizer),
        )
        log_prior = torch.where(
            candidate_mask & has_candidate,
            masked_logits - safe_normalizer,
            torch.full_like(masked_logits, -torch.inf),
        )
        if self.retrieval_prior_scale == 0.0:
            return torch.where(
                candidate_mask,
                torch.zeros_like(log_prior),
                torch.full_like(log_prior, -torch.inf),
            )
        return log_prior * self.retrieval_prior_scale

    def _factorized_token_log_prior(
        self,
        *,
        motion_mask: torch.Tensor,
        part_validity: torch.Tensor,
        motion_tau: torch.Tensor,
        slot_tau: torch.Tensor | None,
        candidate_log_prior: torch.Tensor,
        slot_count: int,
    ):
        batch, candidates, tokens = motion_mask.shape
        dtype = motion_tau.dtype
        device = motion_tau.device
        if slot_tau is None:
            resolved_slot_tau = torch.linspace(
                -1.0,
                1.0,
                int(slot_count),
                dtype=dtype,
                device=device,
            ).view(1, int(slot_count))
        else:
            resolved_slot_tau = slot_tau.to(device=device, dtype=dtype)
            if resolved_slot_tau.ndim == 1:
                if resolved_slot_tau.shape != (int(slot_count),):
                    raise ValueError("slot_tau must have shape [S] or [B,S]")
                resolved_slot_tau = resolved_slot_tau.view(1, int(slot_count))
            elif resolved_slot_tau.shape != (batch, int(slot_count)):
                raise ValueError("slot_tau must have shape [S] or [B,S]")
        resolved_slot_tau = resolved_slot_tau.expand(batch, -1).clamp(-1.0, 1.0)

        validity = part_validity.permute(0, 3, 1, 2)
        validity = validity[:, None, :, :, :].expand(
            -1,
            int(slot_count),
            -1,
            -1,
            -1,
        )
        valid = motion_mask[:, None, None, :, :] & (validity > 0.0)
        if self.temporal_prior_mode == "gaussian":
            token_logits = torch.log(
                validity.clamp_min(torch.finfo(dtype).tiny)
            )
            difference = (
                resolved_slot_tau[:, :, None, None]
                - motion_tau[:, None, :, :]
            )
            gaussian = -0.5 * (
                difference / self.temporal_prior_sigma
            ).square()
            token_logits = (
                token_logits
                + self.temporal_prior_scale * gaussian[:, :, None]
            )
        else:
            # Stage 1 uses a uniform distribution over the positive-validity
            # support. Fractional validity weights are introduced only by the
            # explicit Stage-2 Gaussian prior.
            token_logits = torch.zeros_like(validity)
        token_logits = token_logits.masked_fill(~valid, -torch.inf)
        token_normalizer = torch.logsumexp(token_logits, dim=-1, keepdim=True)
        has_valid_token = valid.any(dim=-1, keepdim=True)
        safe_token_normalizer = torch.where(
            has_valid_token,
            token_normalizer,
            torch.zeros_like(token_normalizer),
        )
        token_log_prior = torch.where(
            valid & has_valid_token,
            token_logits - safe_token_normalizer,
            torch.full_like(token_logits, -torch.inf),
        )
        token_log_prior = token_log_prior + candidate_log_prior[
            :, None, None, :, None
        ]
        token_log_prior = token_log_prior.reshape(
            batch,
            int(slot_count) * PART_COUNT,
            candidates,
            tokens,
        )
        token_mask = valid.reshape_as(token_log_prior)
        return token_log_prior, token_mask

    def _forward_factorized(
        self,
        *,
        text_slots: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_keys: torch.Tensor,
        retrieval_features: torch.Tensor,
        adjusted_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        confidence: torch.Tensor,
        tau: torch.Tensor,
        part_validity: torch.Tensor | None,
        slot_tau: torch.Tensor | None,
        attention_mode: str,
    ):
        batch, candidates, tokens, _motion_dim = motion_tokens.shape
        slot_count = int(text_slots.shape[1])
        dtype = text_slots.dtype
        device = text_slots.device
        if part_validity is None:
            resolved_part_validity = torch.ones(
                batch,
                candidates,
                tokens,
                PART_COUNT,
                device=device,
                dtype=dtype,
            )
        else:
            resolved_part_validity = part_validity.to(
                device=device,
                dtype=dtype,
            ).clamp(0.0, 1.0)
        resolved_part_validity = resolved_part_validity * motion_mask[..., None].to(
            dtype
        )

        # K has only sentence text and retrieval metadata; V has only motion.
        candidate_keys = self.key_projection(text_keys)
        candidate_keys = candidate_keys + self.feature_projection(retrieval_features)
        motion_values = self.motion_projection(motion_tokens)
        motion_values = torch.where(
            motion_mask[..., None],
            motion_values,
            torch.zeros_like(motion_values),
        )
        candidate_log_prior = self._factorized_candidate_log_prior(
            adjusted_scores,
            candidate_mask,
        )
        token_log_prior, token_mask = self._factorized_token_log_prior(
            motion_mask=motion_mask,
            part_validity=resolved_part_validity,
            motion_tau=tau,
            slot_tau=slot_tau,
            candidate_log_prior=candidate_log_prior,
            slot_count=slot_count,
        )

        part_queries = self._part_queries(text_slots)
        state = part_queries.new_zeros(part_queries.shape)
        null_mass = text_slots.new_ones(batch, slot_count * PART_COUNT)
        candidate_mass = text_slots.new_zeros(
            batch,
            slot_count * PART_COUNT,
            candidates,
        )
        token_mass = text_slots.new_zeros(
            batch,
            slot_count * PART_COUNT,
            candidates,
            tokens,
        )
        query_has_memory = token_mask.any(dim=(-1, -2))
        null_bias = self.null_confidence(confidence).squeeze(-1)
        for layer in self.layers:
            state, null_mass, candidate_mass, token_mass = layer(
                part_queries,
                state,
                candidate_keys,
                motion_values,
                token_log_prior,
                token_mask,
                null_bias,
                candidates,
                tokens,
                attention_mode=attention_mode,
            )
            state = torch.where(
                query_has_memory[..., None],
                state,
                torch.zeros_like(state),
            )

        state = state.reshape(batch, slot_count, PART_COUNT, self.hidden_dim)
        part_null_mass = null_mass.reshape(batch, slot_count, PART_COUNT)
        part_candidate_mass = candidate_mass.reshape(
            batch,
            slot_count,
            PART_COUNT,
            candidates,
        )
        part_available = query_has_memory.reshape(
            batch,
            slot_count,
            PART_COUNT,
        ).to(dtype)
        return {
            "slots": state,
            "part_null_mass": part_null_mass,
            "null_mass": part_null_mass.mean(dim=2),
            "candidate_mass": part_candidate_mass.mean(dim=2),
            "part_validity": part_available,
            "confidence": confidence,
            "available": candidate_mask.any(dim=-1),
            "adjusted_scores": adjusted_scores,
        }

    def forward(
        self,
        text_slots: torch.Tensor,
        query_duration: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_keys: torch.Tensor,
        scores: torch.Tensor,
        durations: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        motion_tau: torch.Tensor | None = None,
        part_validity: torch.Tensor | None = None,
        slot_tau: torch.Tensor | None = None,
        attention_mode: str = "learned",
    ):
        attention_mode = str(attention_mode).lower()
        if attention_mode not in SENTENCE_ATTENTION_MODES:
            raise ValueError(
                "sentence-memory attention_mode must be one of "
                f"{sorted(SENTENCE_ATTENTION_MODES)}, got {attention_mode!r}"
            )
        if (
            self.key_value_mode == LEGACY_SENTENCE_KEY_VALUE_MODE
            and attention_mode != "learned"
        ):
            raise ValueError(
                "analytic_prior attention requires "
                f"key_value_mode={FACTORIZED_SENTENCE_KEY_VALUE_MODE!r}"
            )
        batch, candidates, tokens = self.validate_inputs(
            motion_tokens,
            motion_mask,
            text_keys,
            scores,
            durations,
            candidate_mask,
            motion_tau=motion_tau,
            part_validity=part_validity,
        )
        device = text_slots.device
        dtype = text_slots.dtype
        motion_tokens = motion_tokens.to(device=device, dtype=dtype)
        motion_mask = motion_mask.to(device=device).bool()
        text_keys = text_keys.to(device=device, dtype=dtype)
        scores = scores.to(device=device, dtype=dtype)
        durations = durations.to(device=device, dtype=dtype)
        candidate_mask = candidate_mask.to(device=device).bool()
        motion_mask = motion_mask & candidate_mask[:, :, None]
        candidate_mask = candidate_mask & motion_mask.any(dim=-1)
        motion_mask = motion_mask & candidate_mask[:, :, None]

        if motion_tau is None:
            tau = _token_tau(motion_mask, dtype)
        else:
            tau = motion_tau.to(device=device, dtype=dtype).clamp(-1.0, 1.0)
            tau = tau * motion_mask.to(dtype)

        adjusted, probabilities, duration_gap, confidence = self._retrieval_statistics(
            scores,
            candidate_mask,
            query_duration.to(device=device, dtype=dtype),
            durations,
        )
        # Derive rank from detached candidate evidence rather than physical
        # tensor position. This keeps retrieval conditioning invariant when a
        # caller consistently permutes candidates and all their metadata.
        greater = adjusted[:, None, :] > adjusted[:, :, None]
        greater = greater & candidate_mask[:, None, :]
        rank_numerator = greater.sum(dim=-1).to(dtype)
        if self.key_value_mode == FACTORIZED_SENTENCE_KEY_VALUE_MODE:
            rank_denominator = (
                candidate_mask.sum(dim=-1).sub(1).clamp_min(1).to(dtype)
            )[:, None]
            ranks = rank_numerator / rank_denominator
        else:
            # Preserve the legacy mixed-memory computation exactly.
            ranks = rank_numerator / max(candidates - 1, 1)
        ranks = ranks * candidate_mask.to(dtype)
        margin = confidence[:, 1:2].expand(-1, candidates)
        retrieval_features = torch.stack(
            [scores, probabilities, ranks, duration_gap, margin],
            dim=-1,
        )

        if self.key_value_mode == FACTORIZED_SENTENCE_KEY_VALUE_MODE:
            return self._forward_factorized(
                text_slots=text_slots,
                motion_tokens=motion_tokens,
                motion_mask=motion_mask,
                text_keys=text_keys,
                retrieval_features=retrieval_features,
                adjusted_scores=adjusted,
                candidate_mask=candidate_mask,
                confidence=confidence,
                tau=tau,
                part_validity=part_validity,
                slot_tau=slot_tau,
                attention_mode=attention_mode,
            )

        memory = self.motion_projection(motion_tokens)
        memory = memory + self.key_projection(text_keys)[:, :, None, :]
        memory = memory + self.time_projection(_slot_time_features(tau))
        memory = memory + self.feature_projection(retrieval_features)[:, :, None, :]
        memory = self.memory_norm(memory)
        memory = torch.where(
            motion_mask[..., None],
            memory,
            torch.zeros_like(memory),
        )
        memory = memory.reshape(batch, candidates * tokens, self.hidden_dim)
        flat_mask = motion_mask.reshape(batch, candidates * tokens)

        log_prior = torch.log(probabilities.clamp_min(1e-8))
        log_prior = self.retrieval_prior_scale * log_prior
        # The candidate prior is sentence-level. Spread it over that
        # candidate's valid tokens so longer memories do not receive extra
        # aggregate mass merely because they contain more keys.
        valid_token_count = motion_mask.sum(dim=-1).clamp_min(1).to(dtype)
        log_prior = log_prior - torch.log(valid_token_count)
        log_prior = log_prior[:, :, None].expand(-1, -1, tokens)
        log_prior = log_prior.reshape(batch, candidates * tokens)
        null_bias = self.null_confidence(confidence).squeeze(-1)

        part_queries = self._part_queries(text_slots)
        slot_count = text_slots.shape[1]
        state = part_queries.new_zeros(part_queries.shape)
        null_mass = text_slots.new_ones(batch, slot_count * PART_COUNT)
        candidate_mass = text_slots.new_zeros(
            batch,
            slot_count * PART_COUNT,
            candidates,
        )
        token_mass = text_slots.new_zeros(
            batch,
            slot_count * PART_COUNT,
            candidates,
            tokens,
        )
        for layer in self.layers:
            state, null_mass, candidate_mass, token_mass = layer(
                part_queries,
                state,
                memory,
                flat_mask,
                log_prior,
                null_bias,
                candidates,
                tokens,
            )
            # Bias-free attention is zero for an empty row, but learned
            # normalization/FFN parameters need not keep its state at zero.
            # Re-apply the semantic contract after each block: no real memory
            # means exactly zero evidence, not merely zero downstream gates.
            state = torch.where(
                flat_mask.any(dim=-1)[:, None, None],
                state,
                torch.zeros_like(state),
            )

        state = state.reshape(batch, slot_count, PART_COUNT, self.hidden_dim)
        part_null_mass = null_mass.reshape(batch, slot_count, PART_COUNT)
        part_candidate_mass = candidate_mass.reshape(
            batch,
            slot_count,
            PART_COUNT,
            candidates,
        )
        token_mass = token_mass.reshape(
            batch,
            slot_count,
            PART_COUNT,
            candidates,
            tokens,
        )

        if part_validity is None:
            part_validity = torch.ones(
                batch,
                candidates,
                tokens,
                PART_COUNT,
                device=device,
                dtype=dtype,
            )
        else:
            part_validity = part_validity.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        weighted_validity = torch.einsum(
            "bspku,bkup->bsp",
            token_mass,
            part_validity,
        )
        real_mass = token_mass.sum(dim=(-1, -2), keepdim=False)
        weighted_validity = weighted_validity / real_mass.clamp_min(1e-8)
        weighted_validity = torch.where(
            real_mass > 0,
            weighted_validity,
            torch.zeros_like(weighted_validity),
        )
        available = candidate_mask.any(dim=-1)
        return {
            "slots": state,
            "part_null_mass": part_null_mass,
            "null_mass": part_null_mass.mean(dim=2),
            "candidate_mass": part_candidate_mass.mean(dim=2),
            "part_validity": weighted_validity,
            "confidence": confidence,
            "available": available,
            "adjusted_scores": adjusted,
        }


class SentenceMemoryTrajectoryHypernetwork(DualModeTrajectoryHypernetwork):
    """SignTrajField-v3 conditioning with optional sentence-motion memory."""

    def __init__(
        self,
        *args,
        sentence_motion_dim: int = 256,
        sentence_key_dim: int | None = None,
        sentence_attention_layers: int = 2,
        sentence_attention_heads: int = 8,
        sentence_score_temperature: float = 0.10,
        sentence_duration_weight: float = 0.10,
        sentence_retrieval_prior_scale: float = 1.0,
        sentence_gate_initial_bias: float = -2.2,
        sentence_key_value_mode: str | None = None,
        sentence_temporal_prior_mode: str = "none",
        sentence_temporal_prior_sigma: float = 0.25,
        sentence_temporal_prior_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        key_dim = self.text_dim if sentence_key_dim is None else int(sentence_key_dim)
        dropout = float(self.global_context[3].p)
        self.sentence_memory_encoder = SentenceMemorySlotEncoder(
            motion_dim=int(sentence_motion_dim),
            key_dim=key_dim,
            hidden_dim=self.context_hidden_dim,
            layer_count=int(sentence_attention_layers),
            head_count=int(sentence_attention_heads),
            dropout=dropout,
            score_temperature=float(sentence_score_temperature),
            duration_weight=float(sentence_duration_weight),
            retrieval_prior_scale=float(sentence_retrieval_prior_scale),
            key_value_mode=sentence_key_value_mode,
            temporal_prior_mode=str(sentence_temporal_prior_mode),
            temporal_prior_sigma=float(sentence_temporal_prior_sigma),
            temporal_prior_scale=float(sentence_temporal_prior_scale),
        )
        factorized = (
            self.sentence_memory_encoder.key_value_mode
            == FACTORIZED_SENTENCE_KEY_VALUE_MODE
        )
        gate_input_dim = self.context_hidden_dim * 2 + SENTENCE_CONFIDENCE_FEATURE_DIM
        self.sentence_memory_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, self.context_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.context_hidden_dim, 1),
        )
        self.sentence_memory_part_projections = nn.ModuleList(
            [
                nn.Linear(
                    self.context_hidden_dim,
                    self.context_hidden_dim,
                    bias=not factorized,
                )
                for _ in range(PART_COUNT)
            ]
        )
        self.sentence_memory_fusion = nn.Linear(
            self.context_hidden_dim * PART_COUNT,
            self.context_hidden_dim,
            bias=False,
        )
        nn.init.zeros_(self.sentence_memory_gate[-1].weight)
        nn.init.constant_(
            self.sentence_memory_gate[-1].bias,
            float(sentence_gate_initial_bias),
        )
        # This is the no-regression boundary: a newly migrated v3 checkpoint is
        # exactly the v2 model even when valid memories are supplied.
        nn.init.zeros_(self.sentence_memory_fusion.weight)

    @staticmethod
    def _sentence_inputs(
        sentence_motion_tokens,
        sentence_motion_mask,
        sentence_text_keys,
        sentence_scores,
        sentence_durations,
        sentence_candidate_mask,
    ):
        return (
            sentence_motion_tokens,
            sentence_motion_mask,
            sentence_text_keys,
            sentence_scores,
            sentence_durations,
            sentence_candidate_mask,
        )

    def _off_diagnostics(self, trajectory: TrajectoryInstance):
        batch = trajectory.batch_size
        slots = self.temporal_slot_count
        dtype = trajectory.dtype
        device = trajectory.device
        return replace(
            trajectory,
            sentence_memory_available=torch.zeros(batch, dtype=torch.bool, device=device),
            sentence_memory_gates=torch.zeros(batch, slots, PART_COUNT, dtype=dtype, device=device),
            sentence_memory_null_mass=torch.ones(batch, slots, dtype=dtype, device=device),
            sentence_memory_candidate_mass=None,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        *,
        word_prior_context: torch.Tensor | None = None,
        word_prior_mask: torch.Tensor | None = None,
        word_prior_features: torch.Tensor | None = None,
        word_prior_available: torch.Tensor | None = None,
        sentence_motion_tokens: torch.Tensor | None = None,
        sentence_motion_mask: torch.Tensor | None = None,
        sentence_motion_tau: torch.Tensor | None = None,
        sentence_text_keys: torch.Tensor | None = None,
        sentence_scores: torch.Tensor | None = None,
        sentence_durations: torch.Tensor | None = None,
        sentence_candidate_mask: torch.Tensor | None = None,
        sentence_part_validity: torch.Tensor | None = None,
        sentence_memory_available: torch.Tensor | None = None,
        sentence_memory_attention_mode: str = "learned",
    ) -> TrajectoryInstance:
        sentence_memory_attention_mode = str(
            sentence_memory_attention_mode
        ).lower()
        if sentence_memory_attention_mode not in SENTENCE_ATTENTION_MODES:
            raise ValueError(
                "sentence_memory_attention_mode must be one of "
                f"{sorted(SENTENCE_ATTENTION_MODES)}, "
                f"got {sentence_memory_attention_mode!r}"
            )
        sentence_inputs = self._sentence_inputs(
            sentence_motion_tokens,
            sentence_motion_mask,
            sentence_text_keys,
            sentence_scores,
            sentence_durations,
            sentence_candidate_mask,
        )
        if any(value is not None for value in sentence_inputs) and not all(
            value is not None for value in sentence_inputs
        ):
            raise ValueError(
                "sentence_motion_tokens, sentence_motion_mask, sentence_text_keys, "
                "sentence_scores, sentence_durations, and sentence_candidate_mask "
                "must be supplied together"
            )
        if not all(value is not None for value in sentence_inputs):
            if sentence_motion_tau is not None or sentence_part_validity is not None:
                raise ValueError("sentence motion auxiliaries require sentence-memory tensors")
            if sentence_memory_available is not None:
                if sentence_memory_available.shape != (text_tokens.shape[0],):
                    raise ValueError("sentence_memory_available must have shape [B]")
                if bool(sentence_memory_available.bool().any()):
                    raise ValueError(
                        "Available sentence memory requires sentence-memory tensors"
                    )
            trajectory = super().forward(
                text_tokens=text_tokens,
                text_mask=text_mask,
                word_prior_context=word_prior_context,
                word_prior_mask=word_prior_mask,
                word_prior_features=word_prior_features,
                word_prior_available=word_prior_available,
            )
            return self._off_diagnostics(trajectory)

        if text_mask is None:
            text_mask = torch.ones(
                text_tokens.shape[:2],
                dtype=torch.bool,
                device=text_tokens.device,
            )
        text_mask = text_mask.bool()
        text_slots, slot_tau = self.text_planner(text_tokens, text_mask)
        dtype = text_slots.dtype
        device = text_slots.device
        batch = text_slots.shape[0]
        pooled_text = self._pooled_text(text_tokens, text_mask)
        log_duration, duration = self.duration_head(pooled_text)

        word_supplied = (
            word_prior_context is not None,
            word_prior_mask is not None,
            word_prior_features is not None,
        )
        if any(word_supplied) and not all(word_supplied):
            raise ValueError(
                "word_prior_context, word_prior_mask, and word_prior_features "
                "must be supplied together"
            )
        if word_prior_available is None:
            word_prior_available = torch.full(
                (batch,),
                bool(all(word_supplied)),
                dtype=torch.bool,
                device=device,
            )
        else:
            word_prior_available = word_prior_available.to(device=device).bool()
            if word_prior_available.shape != (batch,):
                raise ValueError("word_prior_available must have shape [B]")
        if bool(word_prior_available.any()) and not all(word_supplied):
            raise ValueError("Available word-prior samples require word-prior tensors")

        if all(word_supplied):
            context = word_prior_context.to(device=device, dtype=dtype)
            context_mask = word_prior_mask.to(device=device).bool()
            features = word_prior_features.to(device=device, dtype=dtype)
            word_slots, slot_features = self.word_encoder(
                context,
                context_mask,
                features,
                slot_tau,
            )
        else:
            word_slots = text_slots.new_zeros(text_slots.shape)
            slot_features = text_slots.new_zeros(
                batch,
                self.temporal_slot_count,
                self.retrieval_dim,
            )

        word_gate_input = torch.cat([text_slots, word_slots, slot_features], dim=-1)
        word_gates = torch.sigmoid(self.word_gate(word_gate_input))
        word_availability_float = word_prior_available.to(dtype)
        word_gates = word_gates * word_availability_float[:, None, None]
        word_part_deltas = [
            projection(word_slots) * word_gates[..., index : index + 1]
            for index, projection in enumerate(self.word_part_projections)
        ]
        word_delta = self.word_fusion(torch.cat(word_part_deltas, dim=-1))
        word_delta = word_delta * word_availability_float[:, None, None]

        sentence = self.sentence_memory_encoder(
            text_slots,
            duration,
            sentence_motion_tokens,
            sentence_motion_mask,
            sentence_text_keys,
            sentence_scores,
            sentence_durations,
            sentence_candidate_mask,
            motion_tau=sentence_motion_tau,
            part_validity=sentence_part_validity,
            slot_tau=slot_tau,
            attention_mode=sentence_memory_attention_mode,
        )
        if sentence_memory_available is None:
            sentence_memory_available = sentence["available"]
        else:
            sentence_memory_available = sentence_memory_available.to(
                device=device
            ).bool()
            if sentence_memory_available.shape != (batch,):
                raise ValueError("sentence_memory_available must have shape [B]")
            sentence_memory_available = sentence_memory_available & sentence["available"]
        sentence_availability_float = sentence_memory_available.to(dtype)
        confidence = sentence["confidence"][:, None, None, :].expand(
            -1,
            self.temporal_slot_count,
            PART_COUNT,
            -1,
        )
        gate_text_slots = text_slots[:, :, None, :].expand(
            -1,
            -1,
            PART_COUNT,
            -1,
        )
        sentence_gate_input = torch.cat(
            [gate_text_slots, sentence["slots"], confidence],
            dim=-1,
        )
        sentence_gates = torch.sigmoid(
            self.sentence_memory_gate(sentence_gate_input)
        ).squeeze(-1)
        sentence_gates = sentence_gates * (1.0 - sentence["part_null_mass"])
        sentence_gates = sentence_gates * sentence["part_validity"]
        sentence_gates = sentence_gates * sentence_availability_float[:, None, None]
        sentence_part_deltas = [
            projection(sentence["slots"][:, :, index, :])
            * sentence_gates[..., index : index + 1]
            for index, projection in enumerate(self.sentence_memory_part_projections)
        ]
        sentence_delta = self.sentence_memory_fusion(
            torch.cat(sentence_part_deltas, dim=-1)
        )
        sentence_delta = sentence_delta * sentence_availability_float[:, None, None]
        fused_slots = self.fusion_norm(text_slots + word_delta + sentence_delta)

        summary = torch.cat(
            [
                fused_slots.mean(dim=1),
                fused_slots.std(dim=1, unbiased=False),
                fused_slots[:, 0],
                fused_slots[:, -1],
                self.text_planner.text_projection(pooled_text),
            ],
            dim=-1,
        )
        global_context = self.global_context(summary)
        prior_scale, prior_shift, prior_output_bias = self._split_modulation(
            self.coarse_head(global_context),
            self.pose_dim,
        )
        residual_scale, residual_shift, residual_output_bias = self._split_modulation(
            self.residual_head(global_context),
            self.residual_dim,
        )
        articulator_gates = torch.sigmoid(self.gate_head(global_context))

        local_count = self.max_local_fields
        if local_count:
            predicted_frames = duration.detach() * self.context_fps
            active_count = torch.ceil(
                predicted_frames / self.frames_per_local_field
            ).long()
            active_count = active_count.clamp(1, local_count)
            centers, local_mask = self._uniform_centers(active_count, local_count, dtype)
            local_slot = self._interpolate_slots(fused_slots, centers)
            local_text = self.text_planner.text_projection(pooled_text)
            local_text = local_text[:, None, :].expand(-1, local_count, -1)
            local_global = global_context[:, None, :].expand(-1, local_count, -1)
            local_context = self.local_context(
                torch.cat([local_slot, local_text, local_global], dim=-1)
            )
            local_values = self.local_head(local_context)
            local_part_gates = (
                torch.sigmoid(self.local_gate_head(local_context))
                if self.local_gate_head is not None
                else None
            )
            modulation_size = self.field_depth * self.field_hidden_dim
            (
                local_scale_flat,
                local_shift_flat,
                local_output_bias,
                width_logits,
            ) = torch.split(
                local_values,
                [modulation_size, modulation_size, self.residual_dim, 1],
                dim=-1,
            )
            local_scale = local_scale_flat.reshape(
                batch,
                local_count,
                self.field_depth,
                self.field_hidden_dim,
            )
            local_shift = local_shift_flat.reshape_as(local_scale)
            width_unit = torch.sigmoid(width_logits.squeeze(-1))
            learned_width = self.minimum_local_width + width_unit * (
                self.maximum_local_width - self.minimum_local_width
            )
            base_width = (2.5 / active_count.to(dtype)).clamp(
                self.minimum_local_width,
                self.maximum_local_width,
            )
            local_widths = 0.5 * (learned_width + base_width[:, None])
            local_word_gates = self._interpolate_slots(word_gates, centers)
            local_sentence_gates = self._interpolate_slots(sentence_gates, centers)
            combined_retrieval_gates = 1.0 - (
                1.0 - local_word_gates
            ) * (1.0 - local_sentence_gates)
            local_uncertainty = 1.0 - combined_retrieval_gates.mean(dim=-1)
            mask_float = local_mask.to(dtype)
            local_scale = local_scale * mask_float[:, :, None, None]
            local_shift = local_shift * mask_float[:, :, None, None]
            local_output_bias = local_output_bias * mask_float[:, :, None]
            local_widths = local_widths * mask_float + (~local_mask).to(dtype)
            local_uncertainty = local_uncertainty * mask_float
            if local_part_gates is not None:
                local_part_gates = local_part_gates * mask_float[:, :, None]
        else:
            local_scale = fused_slots.new_zeros(
                batch,
                0,
                self.field_depth,
                self.field_hidden_dim,
            )
            local_shift = local_scale.clone()
            local_output_bias = fused_slots.new_zeros(batch, 0, self.residual_dim)
            centers = fused_slots.new_zeros(batch, 0)
            local_widths = fused_slots.new_zeros(batch, 0)
            local_mask = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            local_uncertainty = fused_slots.new_zeros(batch, 0)
            local_part_gates = (
                fused_slots.new_zeros(batch, 0, PART_COUNT)
                if self.local_gate_head is not None
                else None
            )

        candidate_mass = sentence["candidate_mass"]
        candidate_mass = candidate_mass * sentence_availability_float[:, None, None]
        null_mass = torch.where(
            sentence_memory_available[:, None],
            sentence["null_mass"],
            torch.ones_like(sentence["null_mass"]),
        )
        return TrajectoryInstance(
            duration_seconds=duration,
            log_duration_seconds=log_duration,
            prior_scale=prior_scale,
            prior_shift=prior_shift,
            prior_output_bias=prior_output_bias,
            residual_scale=residual_scale,
            residual_shift=residual_shift,
            residual_output_bias=residual_output_bias,
            local_scale=local_scale,
            local_shift=local_shift,
            local_output_bias=local_output_bias,
            local_centers=centers,
            local_widths=local_widths,
            local_mask=local_mask,
            articulator_gates=articulator_gates,
            local_uncertainty=local_uncertainty,
            context_density=fused_slots.norm(dim=-1),
            context_tau=slot_tau[None, :].expand(batch, -1),
            local_part_gates=local_part_gates,
            word_prior_available=word_prior_available,
            word_prior_gates=word_gates,
            temporal_slot_tau=slot_tau[None, :].expand(batch, -1),
            sentence_memory_available=sentence_memory_available,
            sentence_memory_gates=sentence_gates,
            sentence_memory_null_mass=null_mass,
            sentence_memory_candidate_mass=candidate_mass,
        )
