"""Read-only probes and metrics for temporal-slot diagnostics.

This module deliberately lives outside the trajectory model.  It uses temporary
forward hooks to observe internal tensors during validation and provides pure
metric helpers for the diagnostic script.  Importing it does not alter model
state, checkpoint schemas, or ordinary prediction exports.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from NIAF.continuous_trajectory_field.models.dual_mode_trajectory_hypernetwork import (
    PART_COUNT,
)


TensorTransform = Callable[
    [nn.Module, tuple[Any, ...], Mapping[str, Any], Any],
    Any,
]


@dataclass(frozen=True)
class SentenceLayerCapture:
    """Unaveraged output of one sentence-memory cross-attention layer.

    Shapes are ``state=[B,S,P,H]``, ``part_null_mass=[B,S,P]``,
    ``part_candidate_mass=[B,S,P,K]``, and
    ``token_mass=[B,S,P,K,U]``.  ``P`` is currently four but is inferred from
    the v3 model contract rather than from the hidden dimension.
    """

    state: torch.Tensor
    part_null_mass: torch.Tensor
    part_candidate_mass: torch.Tensor
    token_mass: torch.Tensor


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Tensors captured from exactly one trajectory-hypernetwork forward."""

    planner_slots: torch.Tensor
    slot_tau: torch.Tensor
    fusion_input: torch.Tensor
    fused_slots: torch.Tensor
    sentence_delta: torch.Tensor | None
    text_attention: torch.Tensor
    text_attention_replay_max_abs: torch.Tensor
    sentence_layers: tuple[SentenceLayerCapture, ...]


@dataclass(frozen=True)
class StratifiedSelection:
    """Deterministic, one-row-per-text causal-probe selection."""

    indices: tuple[int, ...]
    duration_bins: tuple[int, ...]
    margin_bins: tuple[int, ...]
    selection_hash: str
    requested_count: int
    available_unique_texts: int


@dataclass(frozen=True)
class BootstrapInterval:
    """Cluster-level mean and percentile confidence interval."""

    estimate: float | np.ndarray
    lower: float | np.ndarray
    upper: float | np.ndarray
    cluster_count: int
    samples: int
    seed: int
    confidence: float


def _resolve_hypernetwork(model_or_hypernetwork: nn.Module) -> nn.Module:
    module = model_or_hypernetwork
    while hasattr(module, "module") and isinstance(module.module, nn.Module):
        module = module.module
    candidate = getattr(module, "hypernetwork", None)
    if isinstance(candidate, nn.Module):
        module = candidate
    required = ("text_planner", "fusion_norm", "temporal_slot_count")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError(
            "DiagnosticCapture requires a trajectory hypernetwork; missing "
            + ", ".join(missing)
        )
    return module


def _copy_tensor(
    value: torch.Tensor,
    *,
    cpu: bool,
    clone: bool,
) -> torch.Tensor:
    result = value.detach()
    if cpu:
        result = result.cpu()
    if clone:
        result = result.clone()
    return result


class DiagnosticCapture(AbstractContextManager["DiagnosticCapture"]):
    """Temporarily capture temporal-slot and attention internals.

    Text cross-attention is recovered by replaying each frozen decoder
    cross-attention call with ``need_weights=True`` and
    ``average_attn_weights=False``.  The replay output is compared against the
    native output and a mismatch above ``replay_tolerance`` fails closed.

    Call :meth:`snapshot` after a model forward and before the next one.  All
    hooks are removed in ``__exit__`` even when the observed forward raises.
    """

    def __init__(
        self,
        model_or_hypernetwork: nn.Module,
        *,
        replay_text_attention: bool = True,
        replay_tolerance: float = 1e-6,
    ):
        if float(replay_tolerance) < 0.0:
            raise ValueError("replay_tolerance must be non-negative")
        self.hypernetwork = _resolve_hypernetwork(model_or_hypernetwork)
        self.replay_text_attention = bool(replay_text_attention)
        self.replay_tolerance = float(replay_tolerance)
        self._handles: list[Any] = []
        self._active = False
        self._replaying = False
        self.clear()

    def clear(self):
        self._planner_slots: torch.Tensor | None = None
        self._slot_tau: torch.Tensor | None = None
        self._fusion_input: torch.Tensor | None = None
        self._fused_slots: torch.Tensor | None = None
        self._sentence_delta: torch.Tensor | None = None
        self._text_attention: dict[int, torch.Tensor] = {}
        self._text_replay_error: dict[int, torch.Tensor] = {}
        self._sentence_layers: dict[int, SentenceLayerCapture] = {}

    @staticmethod
    def _first_arg(args: tuple[Any, ...], name: str) -> torch.Tensor:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError(f"Could not capture tensor input for {name}")
        return args[0]

    def _planner_hook(self, _module, _args, _kwargs, output):
        if self._replaying:
            return None
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("text planner did not return (slots, slot_tau)")
        slots, tau = output
        if not isinstance(slots, torch.Tensor) or not isinstance(tau, torch.Tensor):
            raise RuntimeError("text planner returned non-tensor diagnostics")
        self._planner_slots = slots.detach()
        self._slot_tau = tau.detach()
        return None

    def _fusion_hook(self, _module, args, _kwargs, output):
        if self._replaying:
            return None
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("fusion_norm did not return a tensor")
        self._fusion_input = self._first_arg(args, "fusion_norm").detach()
        self._fused_slots = output.detach()
        return None

    def _sentence_fusion_hook(self, _module, _args, _kwargs, output):
        if self._replaying:
            return None
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("sentence_memory_fusion did not return a tensor")
        self._sentence_delta = output.detach()
        return None

    def _sentence_layer_hook(self, layer_index: int):
        def hook(_module, _args, _kwargs, output):
            if self._replaying:
                return None
            if not isinstance(output, tuple) or len(output) != 4:
                raise RuntimeError(
                    "sentence-memory attention layer did not return four tensors"
                )
            state, null_mass, candidate_mass, token_mass = output
            if not all(
                isinstance(value, torch.Tensor)
                for value in (state, null_mass, candidate_mass, token_mass)
            ):
                raise RuntimeError("sentence-memory layer returned non-tensors")
            slot_count = int(self.hypernetwork.temporal_slot_count)
            expected_queries = slot_count * PART_COUNT
            if state.ndim != 3 or state.shape[1] != expected_queries:
                raise RuntimeError(
                    "sentence-memory state has incompatible query count: "
                    f"expected {expected_queries}, found {tuple(state.shape)}"
                )
            batch, _queries, hidden = state.shape
            candidates = int(candidate_mass.shape[-1])
            tokens = int(token_mass.shape[-1])
            expected_shapes = (
                (batch, expected_queries),
                (batch, expected_queries, candidates),
                (batch, expected_queries, candidates, tokens),
            )
            actual_shapes = (
                tuple(null_mass.shape),
                tuple(candidate_mass.shape),
                tuple(token_mass.shape),
            )
            if actual_shapes != expected_shapes:
                raise RuntimeError(
                    "sentence-memory attention diagnostics have incompatible shapes: "
                    f"expected {expected_shapes}, found {actual_shapes}"
                )
            self._sentence_layers[layer_index] = SentenceLayerCapture(
                state=state.detach().reshape(
                    batch, slot_count, PART_COUNT, hidden
                ),
                part_null_mass=null_mass.detach().reshape(
                    batch, slot_count, PART_COUNT
                ),
                part_candidate_mass=candidate_mass.detach().reshape(
                    batch, slot_count, PART_COUNT, candidates
                ),
                token_mass=token_mass.detach().reshape(
                    batch, slot_count, PART_COUNT, candidates, tokens
                ),
            )
            return None

        return hook

    @staticmethod
    def _replay_arguments(
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        replay_args = list(args)
        replay_kwargs = dict(kwargs)
        if len(replay_args) > 4:
            replay_args[4] = True
            replay_kwargs.pop("need_weights", None)
        else:
            replay_kwargs["need_weights"] = True
        if len(replay_args) > 6:
            replay_args[6] = False
            replay_kwargs.pop("average_attn_weights", None)
        else:
            replay_kwargs["average_attn_weights"] = False
        return tuple(replay_args), replay_kwargs

    def _text_attention_hook(self, layer_index: int):
        def hook(module, args, kwargs, output):
            if self._replaying or not self.replay_text_attention:
                return None
            if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
                raise RuntimeError("text cross-attention did not return a tensor tuple")
            if module.training and float(getattr(module, "dropout", 0.0)) > 0.0:
                raise RuntimeError(
                    "Text-attention replay requires evaluation mode when dropout is nonzero"
                )
            replay_args, replay_kwargs = self._replay_arguments(args, kwargs)
            self._replaying = True
            try:
                replay_output, attention = module(*replay_args, **replay_kwargs)
            finally:
                self._replaying = False
            if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
                raise RuntimeError(
                    "Per-head text cross-attention replay did not return [B,H,S,L]"
                )
            difference = (replay_output - output[0]).detach().abs().amax()
            if not bool(torch.isfinite(difference)):
                raise RuntimeError("Text-attention replay produced a non-finite error")
            if float(difference.item()) > self.replay_tolerance:
                raise RuntimeError(
                    "Text-attention replay changed the native attention output: "
                    f"max_abs={float(difference.item()):.9g}, "
                    f"tolerance={self.replay_tolerance:.9g}"
                )
            self._text_attention[layer_index] = attention.detach()
            self._text_replay_error[layer_index] = difference
            return None

        return hook

    def __enter__(self):
        if self._active:
            raise RuntimeError("DiagnosticCapture cannot be entered twice")
        self.clear()
        try:
            planner = self.hypernetwork.text_planner
            self._handles.append(
                planner.register_forward_hook(self._planner_hook, with_kwargs=True)
            )
            self._handles.append(
                self.hypernetwork.fusion_norm.register_forward_hook(
                    self._fusion_hook,
                    with_kwargs=True,
                )
            )
            sentence_fusion = getattr(
                self.hypernetwork,
                "sentence_memory_fusion",
                None,
            )
            if isinstance(sentence_fusion, nn.Module):
                self._handles.append(
                    sentence_fusion.register_forward_hook(
                        self._sentence_fusion_hook,
                        with_kwargs=True,
                    )
                )
            decoder_layers = list(getattr(planner.decoder, "layers", ()))
            if not decoder_layers:
                raise RuntimeError("text planner has no decoder layers")
            for layer_index, layer in enumerate(decoder_layers):
                self._handles.append(
                    layer.multihead_attn.register_forward_hook(
                        self._text_attention_hook(layer_index),
                        with_kwargs=True,
                    )
                )
            sentence_encoder = getattr(
                self.hypernetwork,
                "sentence_memory_encoder",
                None,
            )
            if isinstance(sentence_encoder, nn.Module):
                for layer_index, layer in enumerate(sentence_encoder.layers):
                    self._handles.append(
                        layer.register_forward_hook(
                            self._sentence_layer_hook(layer_index),
                            with_kwargs=True,
                        )
                    )
        except BaseException:
            # `__exit__` is never invoked when `__enter__` raises. Remove any
            # hooks registered before the failure so a malformed diagnostic
            # target cannot alter later ordinary forwards.
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._active = False
            self._replaying = False
            raise
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._active = False
        self._replaying = False
        return False

    def snapshot(
        self,
        *,
        clear: bool = False,
        cpu: bool = False,
        clone: bool = True,
    ) -> DiagnosticSnapshot:
        required = {
            "planner_slots": self._planner_slots,
            "slot_tau": self._slot_tau,
            "fusion_input": self._fusion_input,
            "fused_slots": self._fused_slots,
        }
        missing = [name for name, value in required.items() if value is None]
        decoder_count = len(self.hypernetwork.text_planner.decoder.layers)
        missing_text = [
            index for index in range(decoder_count) if index not in self._text_attention
        ]
        if self.replay_text_attention and missing_text:
            missing.append(f"text_attention_layers={missing_text}")
        sentence_encoder = getattr(
            self.hypernetwork,
            "sentence_memory_encoder",
            None,
        )
        expected_sentence_layers = (
            len(sentence_encoder.layers)
            if isinstance(sentence_encoder, nn.Module)
            else 0
        )
        # A memory-off v3 forward deliberately bypasses the sentence encoder.
        if self._sentence_layers and len(self._sentence_layers) != expected_sentence_layers:
            missing.append("incomplete_sentence_attention_layers")
        if missing:
            raise RuntimeError(
                "No complete diagnostic forward is available: " + ", ".join(missing)
            )

        def copied(value: torch.Tensor) -> torch.Tensor:
            return _copy_tensor(value, cpu=cpu, clone=clone)

        if self.replay_text_attention:
            text_attention = torch.stack(
                [copied(self._text_attention[index]) for index in range(decoder_count)],
                dim=1,
            )
            replay_error = torch.stack(
                [copied(self._text_replay_error[index]) for index in range(decoder_count)]
            )
        else:
            batch = int(self._planner_slots.shape[0])
            slots = int(self._planner_slots.shape[1])
            text_attention = self._planner_slots.new_empty(batch, 0, 0, slots, 0)
            replay_error = self._planner_slots.new_empty(0)
            text_attention = copied(text_attention)
            replay_error = copied(replay_error)

        layers = tuple(
            SentenceLayerCapture(
                state=copied(value.state),
                part_null_mass=copied(value.part_null_mass),
                part_candidate_mass=copied(value.part_candidate_mass),
                token_mass=copied(value.token_mass),
            )
            for _index, value in sorted(self._sentence_layers.items())
        )
        snapshot = DiagnosticSnapshot(
            planner_slots=copied(self._planner_slots),
            slot_tau=copied(self._slot_tau),
            fusion_input=copied(self._fusion_input),
            fused_slots=copied(self._fused_slots),
            sentence_delta=(
                copied(self._sentence_delta)
                if self._sentence_delta is not None
                else None
            ),
            text_attention=text_attention,
            text_attention_replay_max_abs=replay_error,
            sentence_layers=layers,
        )
        if clear:
            self.clear()
        return snapshot


class ModuleOutputIntervention(AbstractContextManager["ModuleOutputIntervention"]):
    """Apply a temporary output transform to one module.

    ``transform`` receives ``(module, args, kwargs, output)`` and must return the
    replacement output.  The hook is always removed on context exit.
    """

    def __init__(self, module: nn.Module, transform: TensorTransform):
        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        if not callable(transform):
            raise TypeError("transform must be callable")
        self.module = module
        self.transform = transform
        self._handle: Any | None = None

    def _hook(self, module, args, kwargs, output):
        return self.transform(module, args, kwargs, output)

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("ModuleOutputIntervention cannot be entered twice")
        self._handle = self.module.register_forward_hook(
            self._hook,
            with_kwargs=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _require_finite_tensor(value: torch.Tensor, *, name: str):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _pairwise_cosine(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    normalized = values / values.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    return torch.matmul(normalized, normalized.transpose(-1, -2))


def compute_slot_metrics(
    slots: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Measure redundancy and temporal variation of ``[B,S,H]`` slots."""

    _require_finite_tensor(slots, name="slots")
    if slots.ndim != 3 or slots.shape[1] < 2 or slots.shape[2] < 1:
        raise ValueError("slots must have shape [B,S>=2,H>=1]")
    if not slots.is_floating_point():
        raise TypeError("slots must use a floating dtype")
    batch, slot_count, _hidden = slots.shape
    cosine = _pairwise_cosine(slots, float(epsilon))
    diagonal = torch.eye(slot_count, dtype=torch.bool, device=slots.device)
    off_diagonal = cosine[:, ~diagonal].reshape(batch, -1)
    distance = torch.arange(slot_count, device=slots.device)
    distance = (distance[:, None] - distance[None, :]).abs()
    far_mask = distance >= max(slot_count // 2, 1)
    lag_cosine = torch.stack(
        [
            torch.diagonal(cosine, offset=lag, dim1=-2, dim2=-1).mean(dim=-1)
            for lag in range(1, slot_count)
        ],
        dim=-1,
    )

    centered = slots - slots.mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float()).to(slots.dtype)
    singular_energy = singular_values.square()
    energy_sum = singular_energy.sum(dim=-1, keepdim=True)
    energy_probability = singular_energy / energy_sum.clamp_min(float(epsilon))
    effective_rank = torch.exp(
        -(energy_probability * torch.log(energy_probability.clamp_min(epsilon))).sum(
            dim=-1
        )
    )
    effective_rank = torch.where(
        energy_sum.squeeze(-1) > float(epsilon),
        effective_rank,
        torch.zeros_like(effective_rank),
    )
    centered_energy = centered.square().mean(dim=(1, 2))
    raw_energy = slots.square().mean(dim=(1, 2))
    temporal_difference = slots[:, 1:] - slots[:, :-1]
    mean_norm = slots.norm(dim=-1).mean(dim=-1)
    temporal_total_variation = temporal_difference.norm(dim=-1).mean(dim=-1)
    temporal_total_variation = temporal_total_variation / mean_norm.clamp_min(epsilon)

    return {
        "cosine_matrix": cosine,
        "lag_cosine": lag_cosine,
        "off_diagonal_mean": off_diagonal.mean(dim=-1),
        "off_diagonal_median": off_diagonal.median(dim=-1).values,
        "off_diagonal_p90": torch.quantile(off_diagonal, 0.90, dim=-1),
        "off_diagonal_p95": torch.quantile(off_diagonal, 0.95, dim=-1),
        "adjacent_cosine_mean": lag_cosine[:, 0],
        "far_cosine_mean": cosine[:, far_mask].reshape(batch, -1).mean(dim=-1),
        "first_last_cosine": cosine[:, 0, -1],
        "across_slot_variance": centered_energy,
        "temporal_variance_ratio": centered_energy / raw_energy.clamp_min(epsilon),
        "temporal_total_variation": temporal_total_variation,
        "centered_effective_rank": effective_rank,
    }


def jensen_shannon_divergence(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    dim: int = -1,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Jensen--Shannon divergence using natural logarithms."""

    if first.shape != second.shape:
        try:
            first, second = torch.broadcast_tensors(first, second)
        except RuntimeError as error:
            raise ValueError("first and second are not broadcast-compatible") from error
    if bool((first < 0).any()) or bool((second < 0).any()):
        raise ValueError("probability tensors must be non-negative")
    first = first / first.sum(dim=dim, keepdim=True).clamp_min(epsilon)
    second = second / second.sum(dim=dim, keepdim=True).clamp_min(epsilon)
    mixture = 0.5 * (first + second)

    def kl(probability: torch.Tensor):
        terms = probability * (
            torch.log(probability.clamp_min(epsilon))
            - torch.log(mixture.clamp_min(epsilon))
        )
        return terms.sum(dim=dim)

    return 0.5 * (kl(first) + kl(second))


def _average_ranks(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Assign stable average ranks after forming consistent near-tie groups.

    Comparing with strict ``<`` while separately using ``isclose`` for ties is
    inconsistent: two nearly equal values can simultaneously count as ordered
    and tied.  Instead, sort once, form groups from adjacent gaps, and derive
    every rank solely from those groups.  Adjacent grouping also makes the
    tolerance relation deterministic when a chain of near-equal values occurs.
    """

    if values.shape[-1] < 1:
        raise ValueError("Cannot rank an empty dimension")
    sorted_values, order = torch.sort(values, dim=-1, stable=True)
    if values.shape[-1] == 1:
        return torch.zeros_like(values)
    left = sorted_values[..., :-1]
    right = sorted_values[..., 1:]
    scale = torch.maximum(left.abs(), right.abs())
    tolerance = float(epsilon) + 1e-6 * scale
    starts_group = (right - left).abs() > tolerance
    first_group = torch.ones_like(sorted_values[..., :1], dtype=torch.bool)
    group_id = torch.cat([first_group, starts_group], dim=-1).cumsum(dim=-1) - 1

    same_group = group_id.unsqueeze(-1) == group_id.unsqueeze(-2)
    ordinal = torch.arange(
        values.shape[-1],
        device=values.device,
        dtype=values.dtype,
    )
    sorted_rank = (
        same_group.to(values.dtype) * ordinal
    ).sum(dim=-1) / same_group.sum(dim=-1).to(values.dtype)
    ranks = torch.empty_like(sorted_rank)
    return ranks.scatter(dim=-1, index=order, src=sorted_rank)


def rank_correlation_last_dim(
    first_rank: torch.Tensor,
    second_rank: torch.Tensor,
) -> torch.Tensor:
    """Return the Pearson correlation of two already-ranked tensors."""

    first_rank, second_rank = torch.broadcast_tensors(first_rank, second_rank)
    first_rank = first_rank - first_rank.mean(dim=-1, keepdim=True)
    second_rank = second_rank - second_rank.mean(dim=-1, keepdim=True)
    numerator = (first_rank * second_rank).sum(dim=-1)
    denominator = torch.sqrt(
        first_rank.square().sum(dim=-1)
        * second_rank.square().sum(dim=-1)
    )
    return torch.where(
        denominator > 1e-12,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    )


def _spearman_last_dim(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first, second = torch.broadcast_tensors(first, second)
    return rank_correlation_last_dim(
        _average_ranks(first),
        _average_ranks(second),
    )


def normalized_text_attention_position_ranks(
    probability: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    mass_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expected token positions and ranks without renormalizing.

    ``probability`` is the already-normalized per-head text attention with
    shape ``[B,D,H,S,L]``.  This helper deliberately does not divide by its
    measured mass: doing so a second time can move an expected-position
    near-tie across the stable ranking tolerance.  Callers reconstructing a
    persisted diagnostic must also preserve the original batch token extent.
    """

    _require_finite_tensor(probability, name="probability")
    if probability.ndim != 5:
        raise ValueError("probability must have shape [B,D,H,S,L]")
    batch, _layers, _heads, _slots, token_count = probability.shape
    if text_mask.shape != (batch, token_count):
        raise ValueError("text_mask must have shape [B,L]")
    text_mask = text_mask.to(device=probability.device).bool()
    if not bool(text_mask.any(dim=-1).all()):
        raise ValueError("Every text row must contain at least one valid token")
    if bool((probability < 0.0).any()):
        raise ValueError("probability contains negative mass")
    valid = text_mask[:, None, None, None, :]
    invalid_mass = torch.where(
        valid,
        torch.zeros_like(probability),
        probability.abs(),
    )
    if float(invalid_mass.max().item()) != 0.0:
        raise ValueError("probability assigns mass to a masked text token")
    valid_probability = torch.where(
        valid,
        probability,
        torch.zeros_like(probability),
    )
    mass_error = (valid_probability.sum(dim=-1) - 1.0).abs()
    if float(mass_error.max().item()) > float(mass_tolerance):
        raise ValueError(
            "probability mass does not sum to one within tolerance: "
            f"max_abs={float(mass_error.max().item()):.9g}"
        )

    valid_count = text_mask.sum(dim=-1).to(probability.dtype)
    valid_rank = text_mask.cumsum(dim=-1).to(probability.dtype) - 1.0
    denominator = (valid_count - 1.0).clamp_min(1.0)
    token_position = valid_rank / denominator[:, None]
    token_position = torch.where(
        text_mask,
        token_position,
        torch.zeros_like(token_position),
    )
    expected_position = (
        valid_probability * token_position[:, None, None, None, :]
    ).sum(dim=-1)
    return expected_position, _average_ranks(expected_position)


def compute_text_attention_metrics(
    attention: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    eos_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Measure per-head slot-to-token organization.

    ``attention`` must be ``[B,D,H,S,L]`` (decoder layers, heads, slots,
    tokens), while ``text_mask`` and optional ``eos_mask`` are ``[B,L]``.
    """

    _require_finite_tensor(attention, name="attention")
    if attention.ndim != 5:
        raise ValueError("attention must have shape [B,D,H,S,L]")
    batch, _layers, _heads, slot_count, token_count = attention.shape
    if text_mask.shape != (batch, token_count):
        raise ValueError("text_mask must have shape [B,L]")
    text_mask = text_mask.to(device=attention.device).bool()
    if not bool(text_mask.any(dim=-1).all()):
        raise ValueError("Every text row must contain at least one valid token")
    if bool((attention < -float(epsilon)).any()):
        raise ValueError("attention contains negative mass")
    valid = text_mask[:, None, None, None, :]
    invalid_mass = torch.where(valid, torch.zeros_like(attention), attention.abs())
    masked = torch.where(valid, attention.clamp_min(0.0), torch.zeros_like(attention))
    mass = masked.sum(dim=-1, keepdim=True)
    if not bool((mass > float(epsilon)).all()):
        raise ValueError("attention has a slot/head row with no valid token mass")
    probability = masked / mass

    entropy = -(
        probability * torch.log(probability.clamp_min(epsilon))
    ).sum(dim=-1)
    valid_count = text_mask.sum(dim=-1).to(attention.dtype)
    max_entropy = torch.log(valid_count.clamp_min(1.0))[:, None, None, None]
    normalized_entropy = torch.where(
        max_entropy > float(epsilon),
        entropy / max_entropy.clamp_min(epsilon),
        torch.zeros_like(entropy),
    )
    expected_position, expected_position_rank = (
        normalized_text_attention_position_ranks(probability, text_mask)
    )
    slot_position = torch.linspace(
        0.0,
        1.0,
        slot_count,
        device=attention.device,
        dtype=attention.dtype,
    )
    slot_position_rank = _average_ranks(slot_position)
    correlation = rank_correlation_last_dim(
        slot_position_rank.view(1, 1, 1, -1),
        expected_position_rank,
    )
    pairwise_jsd = jensen_shannon_divergence(
        probability.unsqueeze(-2),
        probability.unsqueeze(-3),
        dim=-1,
        epsilon=epsilon,
    )
    adjacent_jsd = jensen_shannon_divergence(
        probability[..., 1:, :],
        probability[..., :-1, :],
        dim=-1,
        epsilon=epsilon,
    )
    upper = torch.triu(
        torch.ones(
            slot_count,
            slot_count,
            dtype=torch.bool,
            device=attention.device,
        ),
        diagonal=1,
    )
    earlier = expected_position.unsqueeze(-1)
    later = expected_position.unsqueeze(-2)
    inversion = (earlier > later).to(attention.dtype)
    tie = torch.isclose(earlier, later, rtol=1e-6, atol=epsilon).to(attention.dtype)
    inversion = inversion + 0.5 * tie
    inversion_rate = inversion[..., upper].mean(dim=-1)

    if eos_mask is None:
        eos_mass = attention.new_zeros(attention.shape[:-1])
    else:
        if eos_mask.shape != (batch, token_count):
            raise ValueError("eos_mask must have shape [B,L]")
        eos_mask = eos_mask.to(device=attention.device).bool() & text_mask
        eos_mass = (
            probability * eos_mask[:, None, None, None, :].to(attention.dtype)
        ).sum(dim=-1)

    return {
        "normalized_attention": probability,
        "invalid_token_mass_max": invalid_mass.amax(dim=(-1, -2, -3, -4)),
        "attention_mass_error": (mass.squeeze(-1) - 1.0).abs(),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "effective_token_count": torch.exp(entropy),
        "maximum_token_mass": probability.amax(dim=-1),
        "eos_mass": eos_mass,
        "expected_token_position": expected_position,
        # Persist this exact float32 rank tensor for downstream permutation
        # tests. Reconstructing it through another numeric backend can move a
        # near-tie across the ranking tolerance and silently change Spearman.
        "expected_token_position_rank": expected_position_rank,
        "slot_token_position_spearman": correlation,
        "inversion_rate": inversion_rate,
        "attention_span": expected_position[..., -1] - expected_position[..., 0],
        "pairwise_slot_jsd": pairwise_jsd,
        "adjacent_slot_jsd": adjacent_jsd,
    }


def analytic_retrieval_prior(
    scores: torch.Tensor,
    duration_log_gap: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    duration_weight: float = 0.05,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Compute the exact duration-adjusted candidate prior used by v3."""

    _require_finite_tensor(scores, name="scores")
    _require_finite_tensor(duration_log_gap, name="duration_log_gap")
    if scores.shape != duration_log_gap.shape or scores.shape != candidate_mask.shape:
        raise ValueError("scores, duration_log_gap, and candidate_mask must agree")
    if float(duration_weight) < 0.0:
        raise ValueError("duration_weight must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    mask = candidate_mask.to(device=scores.device).bool()
    logits = (scores - float(duration_weight) * duration_log_gap.abs()) / float(
        temperature
    )
    logits = logits.masked_fill(~mask, -torch.inf)
    result = torch.softmax(logits, dim=-1)
    return torch.where(
        mask,
        torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(result),
    )


def conditional_candidate_metrics(
    candidate_mass: torch.Tensor,
    null_mass: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    token_mass: torch.Tensor | None = None,
    token_tau: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    slot_tau: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Measure candidate selectivity before any part averaging.

    Candidate mass is ``[B,D,S,P,K]`` and null mass is ``[B,D,S,P]``.
    Optional token mass is ``[B,D,S,P,K,U]`` with ``token_tau=[B,K,U]`` and
    optional padding validity ``token_mask=[B,K,U]``.
    """

    _require_finite_tensor(candidate_mass, name="candidate_mass")
    _require_finite_tensor(null_mass, name="null_mass")
    if candidate_mass.ndim != 5:
        raise ValueError("candidate_mass must have shape [B,D,S,P,K]")
    if null_mass.shape != candidate_mass.shape[:-1]:
        raise ValueError("null_mass must have shape [B,D,S,P]")
    if bool((candidate_mass < -epsilon).any()) or bool((null_mass < -epsilon).any()):
        raise ValueError("attention masses must be non-negative")
    batch, layers, slots, parts, candidates = candidate_mass.shape
    if candidate_mask is None:
        mask = torch.ones(
            batch,
            candidates,
            dtype=torch.bool,
            device=candidate_mass.device,
        )
    else:
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask must have shape [B,K]")
        mask = candidate_mask.to(device=candidate_mass.device).bool()
    expanded_mask = mask[:, None, None, None, :]
    invalid_mass = torch.where(
        expanded_mask,
        torch.zeros_like(candidate_mass),
        candidate_mass.abs(),
    )
    real = torch.where(
        expanded_mask,
        candidate_mass.clamp_min(0.0),
        torch.zeros_like(candidate_mass),
    )
    real_mass = real.sum(dim=-1)
    conditional = torch.where(
        real_mass[..., None] > epsilon,
        real / real_mass[..., None].clamp_min(epsilon),
        torch.zeros_like(real),
    )
    entropy = -(
        conditional * torch.log(conditional.clamp_min(epsilon))
    ).sum(dim=-1)
    valid = real_mass > epsilon
    effective_k = torch.where(valid, torch.exp(entropy), torch.zeros_like(entropy))
    maximum_share = conditional.amax(dim=-1)
    argmax = conditional.argmax(dim=-1)
    argmax = torch.where(valid, argmax, torch.full_like(argmax, -1))

    slot_distribution = conditional.permute(0, 1, 3, 2, 4)
    slot_jsd_matrix = jensen_shannon_divergence(
        slot_distribution.unsqueeze(-2),
        slot_distribution.unsqueeze(-3),
        dim=-1,
        epsilon=epsilon,
    )
    adjacent_slot_jsd = jensen_shannon_divergence(
        slot_distribution[..., 1:, :],
        slot_distribution[..., :-1, :],
        dim=-1,
        epsilon=epsilon,
    )
    part_jsd_matrix = jensen_shannon_divergence(
        conditional.unsqueeze(-2),
        conditional.unsqueeze(-3),
        dim=-1,
        epsilon=epsilon,
    )
    adjacent_candidate_switch = (
        (argmax[:, :, 1:, :] != argmax[:, :, :-1, :])
        & (argmax[:, :, 1:, :] >= 0)
        & (argmax[:, :, :-1, :] >= 0)
    )
    part_argmax_left = argmax.unsqueeze(-1)
    part_argmax_right = argmax.unsqueeze(-2)
    part_candidate_switch = (
        (part_argmax_left != part_argmax_right)
        & (part_argmax_left >= 0)
        & (part_argmax_right >= 0)
    )

    result = {
        "conditional_candidate_mass": conditional,
        "real_candidate_mass": real_mass,
        "null_mass": null_mass.clamp_min(0.0),
        "mass_conservation_error": (
            real_mass + null_mass.clamp_min(0.0) - 1.0
        ).abs(),
        "invalid_candidate_mass_max": invalid_mass.amax(dim=-1),
        "candidate_entropy": entropy,
        "effective_candidate_count": effective_k,
        "maximum_candidate_share": maximum_share,
        "argmax_candidate": argmax,
        "slot_jsd_matrix": slot_jsd_matrix,
        "adjacent_slot_jsd": adjacent_slot_jsd,
        "part_jsd_matrix": part_jsd_matrix,
        "adjacent_candidate_switch": adjacent_candidate_switch,
        "part_candidate_switch": part_candidate_switch,
    }

    if token_mass is not None or token_tau is not None or token_mask is not None:
        if token_mass is None or token_tau is None:
            raise ValueError(
                "token_mass and token_tau must be supplied together when token_mask "
                "is used"
            )
        _require_finite_tensor(token_mass, name="token_mass")
        _require_finite_tensor(token_tau, name="token_tau")
        if token_mass.ndim != 6 or token_mass.shape[:-1] != candidate_mass.shape:
            raise ValueError("token_mass must have shape [B,D,S,P,K,U]")
        token_count = token_mass.shape[-1]
        if token_tau.shape != (batch, candidates, token_count):
            raise ValueError("token_tau must have shape [B,K,U]")
        if token_mask is None:
            memory_token_mask = mask[:, :, None].expand(-1, -1, token_count)
        else:
            if token_mask.shape != (batch, candidates, token_count):
                raise ValueError("token_mask must have shape [B,K,U]")
            memory_token_mask = token_mask.to(device=token_mass.device).bool()
            memory_token_mask = memory_token_mask & mask[:, :, None]
        expanded_token_mask = memory_token_mask[:, None, None, None, :, :]
        invalid_token_mass = torch.where(
            expanded_token_mask,
            torch.zeros_like(token_mass),
            token_mass.abs(),
        )
        valid_token_mass = torch.where(
            expanded_token_mask,
            token_mass.clamp_min(0.0),
            torch.zeros_like(token_mass),
        )
        valid_token_total = valid_token_mass.sum(dim=(-1, -2))
        expected_tau = (
            valid_token_mass
            * token_tau[:, None, None, None, :, :].to(
                device=token_mass.device,
                dtype=token_mass.dtype,
            )
        ).sum(dim=(-1, -2))
        expected_tau = torch.where(
            valid_token_total > epsilon,
            expected_tau / valid_token_total.clamp_min(epsilon),
            torch.zeros_like(expected_tau),
        )
        result["invalid_token_mass_max"] = invalid_token_mass.amax(dim=(-1, -2))
        result["token_candidate_mass_error"] = (
            valid_token_mass.sum(dim=-1) - real
        ).abs()
        result["expected_memory_tau"] = expected_tau
        if slot_tau is not None:
            if slot_tau.ndim == 1:
                if slot_tau.shape != (slots,):
                    raise ValueError("slot_tau must have shape [S] or [B,S]")
                target_tau = slot_tau[None, :].expand(batch, -1)
            elif slot_tau.shape == (batch, slots):
                target_tau = slot_tau
            else:
                raise ValueError("slot_tau must have shape [S] or [B,S]")
            target_tau = target_tau.to(
                device=candidate_mass.device,
                dtype=candidate_mass.dtype,
            )
            result["memory_time_absolute_error"] = (
                expected_tau - target_tau[:, None, :, None]
            ).abs()
            result["slot_memory_time_spearman"] = _spearman_last_dim(
                target_tau[:, None, None, :],
                expected_tau.permute(0, 1, 3, 2),
            )
    elif slot_tau is not None:
        raise ValueError("slot_tau requires token_mass and token_tau")
    return result


def compute_locality_metrics(
    response: torch.Tensor,
    slot_tau: torch.Tensor,
    query_tau: torch.Tensor,
    *,
    local_window: float = 0.25,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Measure temporal localization of a non-negative causal response.

    ``response`` is ``[B,S,P,T]``.  ``slot_tau`` may be ``[S]`` or ``[B,S]``;
    ``query_tau`` may be ``[T]`` or ``[B,T]``.
    """

    _require_finite_tensor(response, name="response")
    if response.ndim != 4:
        raise ValueError("response must have shape [B,S,P,T]")
    if bool((response < 0).any()):
        raise ValueError("response must be non-negative")
    if float(local_window) <= 0.0:
        raise ValueError("local_window must be positive")
    batch, slots, parts, times = response.shape
    if slot_tau.ndim == 1 and slot_tau.shape == (slots,):
        target_slot_tau = slot_tau[None, :].expand(batch, -1)
    elif slot_tau.shape == (batch, slots):
        target_slot_tau = slot_tau
    else:
        raise ValueError("slot_tau must have shape [S] or [B,S]")
    if query_tau.ndim == 1 and query_tau.shape == (times,):
        target_query_tau = query_tau[None, :].expand(batch, -1)
    elif query_tau.shape == (batch, times):
        target_query_tau = query_tau
    else:
        raise ValueError("query_tau must have shape [T] or [B,T]")
    target_slot_tau = target_slot_tau.to(device=response.device, dtype=response.dtype)
    target_query_tau = target_query_tau.to(device=response.device, dtype=response.dtype)
    distance = (
        target_query_tau[:, None, :] - target_slot_tau[:, :, None]
    ).abs()
    total = response.sum(dim=-1)
    normalized = torch.where(
        total[..., None] > epsilon,
        response / total[..., None].clamp_min(epsilon),
        torch.zeros_like(response),
    )
    center = (
        normalized * target_query_tau[:, None, None, :]
    ).sum(dim=-1)
    local_mask = distance <= float(local_window)
    far_mask = distance > 2.0 * float(local_window)
    local_fraction = (
        normalized * local_mask[:, :, None, :].to(response.dtype)
    ).sum(dim=-1)
    far_fraction = (
        normalized * far_mask[:, :, None, :].to(response.dtype)
    ).sum(dim=-1)

    expanded_distance = distance[:, :, None, :].expand(-1, -1, parts, -1)
    sorted_distance, order = torch.sort(expanded_distance, dim=-1)
    sorted_response = torch.gather(normalized, dim=-1, index=order)
    cumulative = sorted_response.cumsum(dim=-1)

    def radius_at(fraction: float):
        reached = cumulative >= float(fraction)
        first = reached.to(torch.int64).argmax(dim=-1)
        radius = torch.gather(sorted_distance, -1, first[..., None]).squeeze(-1)
        return torch.where(
            total > epsilon,
            radius,
            torch.zeros_like(radius),
        )

    target_for_correlation = target_slot_tau[:, None, :].expand(-1, parts, -1)
    center_for_correlation = center.permute(0, 2, 1)
    correlation = _spearman_last_dim(
        target_for_correlation,
        center_for_correlation,
    )
    informative = total > epsilon
    return {
        "response_total": total,
        "response_center_tau": center,
        "response_center_absolute_error": (
            center - target_slot_tau[:, :, None]
        ).abs(),
        "local_response_fraction": local_fraction,
        "far_leakage_fraction": far_fraction,
        "radius_50": radius_at(0.50),
        "radius_80": radius_at(0.80),
        "slot_response_time_spearman": correlation,
        "row_normalized_response": normalized,
        "informative": informative,
    }


def stable_int_seed(*parts: object, base_seed: int = 1234) -> int:
    """Derive a stable positive torch-compatible seed from semantic parts."""

    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little") % (2**63 - 1)


def deterministic_rademacher(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalize_last_dim: bool = True,
) -> torch.Tensor:
    """Return deterministic +/-1 probes, optionally normalized to unit norm."""

    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("shape must contain only positive dimensions")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    values = values.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    if normalize_last_dim:
        values = values / math.sqrt(float(shape[-1]))
    return values.to(device=device, dtype=dtype)


def deterministic_permutations(
    count: int,
    size: int,
    *,
    seed: int = 1234,
) -> torch.Tensor:
    """Return ``count`` deterministic permutations as a CPU int64 tensor."""

    if int(count) < 1 or int(size) < 1:
        raise ValueError("count and size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.stack(
        [torch.randperm(int(size), generator=generator) for _ in range(int(count))]
    )


def _batched_axis_permutation(
    value: torch.Tensor,
    permutation: torch.Tensor,
    *,
    axis: int,
) -> torch.Tensor:
    if axis < 0:
        axis += value.ndim
    if axis <= 0 or axis >= value.ndim:
        raise ValueError("axis must identify a non-batch tensor dimension")
    if permutation.ndim == 1:
        if permutation.shape[0] != value.shape[axis]:
            raise ValueError("permutation length does not match tensor axis")
        return value.index_select(axis, permutation.to(device=value.device))
    if permutation.ndim != 2 or permutation.shape != (
        value.shape[0],
        value.shape[axis],
    ):
        raise ValueError("batched permutation must have shape [B,N]")
    moved = value.movedim(axis, 1)
    index = permutation.to(device=value.device, dtype=torch.long)
    index = index.reshape(index.shape[0], index.shape[1], *([1] * (moved.ndim - 2)))
    index = index.expand_as(moved)
    return torch.gather(moved, 1, index).movedim(1, axis)


def permute_candidate_tensors(
    tensors: Mapping[str, torch.Tensor],
    permutation: torch.Tensor,
    *,
    candidate_axis: int = 1,
) -> dict[str, torch.Tensor]:
    """Consistently permute candidate-aligned tensors without modifying inputs."""

    if not tensors:
        raise ValueError("tensors must not be empty")
    return {
        name: _batched_axis_permutation(value, permutation, axis=candidate_axis)
        for name, value in tensors.items()
    }


def permute_token_tensors(
    tensors: Mapping[str, torch.Tensor],
    permutation: torch.Tensor,
    *,
    token_axis: int = 2,
) -> dict[str, torch.Tensor]:
    """Consistently permute token-aligned ``[B,K,U,...]`` tensors."""

    if not tensors:
        raise ValueError("tensors must not be empty")
    if permutation.ndim != 3:
        raise ValueError("token permutation must have shape [B,K,U]")
    result: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if token_axis < 0:
            actual_axis = value.ndim + token_axis
        else:
            actual_axis = token_axis
        if actual_axis != 2 or value.ndim < 3:
            raise ValueError("token-aligned tensors must use token_axis=2")
        if tuple(value.shape[:3]) != tuple(permutation.shape):
            raise ValueError(f"{name} does not match token permutation [B,K,U]")
        index = permutation.to(device=value.device, dtype=torch.long)
        index = index.reshape(*index.shape, *([1] * (value.ndim - 3)))
        index = index.expand_as(value)
        result[name] = torch.gather(value, 2, index)
    return result


def _stable_order_key(seed: int, *parts: object) -> tuple[str, ...]:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return (digest.hexdigest(), *(str(part) for part in parts))


def _rank_quantile_bins(
    values: Sequence[float],
    tie_breakers: Sequence[tuple[str, ...]],
    bins: int,
) -> list[int]:
    count = len(values)
    order = sorted(
        range(count),
        key=lambda index: (float(values[index]), tie_breakers[index]),
    )
    result = [0] * count
    for rank, index in enumerate(order):
        result[index] = min((rank * int(bins)) // max(count, 1), int(bins) - 1)
    return result


def select_stratified_queries(
    query_ids: Sequence[object],
    normalized_texts: Sequence[str],
    predicted_durations: Sequence[float],
    retrieval_margins: Sequence[float],
    novel_mask: Sequence[bool] | None = None,
    *,
    sample_count: int = 128,
    bins: int = 4,
    seed: int = 1234,
) -> StratifiedSelection:
    """Select unique novel texts over duration x retrieval-margin rank bins."""

    lengths = {
        len(query_ids),
        len(normalized_texts),
        len(predicted_durations),
        len(retrieval_margins),
    }
    if novel_mask is not None:
        lengths.add(len(novel_mask))
    if len(lengths) != 1:
        raise ValueError("All query-selection inputs must have the same length")
    row_count = len(query_ids)
    if row_count < 1:
        raise ValueError("No query rows were supplied")
    if int(sample_count) < 1 or int(bins) < 1:
        raise ValueError("sample_count and bins must be positive")
    if len({str(value) for value in query_ids}) != row_count:
        raise ValueError("query_ids must be unique")
    durations = [float(value) for value in predicted_durations]
    margins = [float(value) for value in retrieval_margins]
    if not all(math.isfinite(value) for value in (*durations, *margins)):
        raise ValueError("duration and retrieval-margin values must be finite")
    novelty = [True] * row_count if novel_mask is None else [bool(v) for v in novel_mask]

    by_text: dict[str, list[int]] = {}
    for index, (text, is_novel) in enumerate(zip(normalized_texts, novelty)):
        normalized = str(text).strip()
        if not normalized:
            raise ValueError(f"normalized_texts[{index}] is empty")
        if is_novel:
            by_text.setdefault(normalized, []).append(index)
    representatives: list[int] = []
    for text in sorted(by_text):
        choices = by_text[text]
        representative = min(
            choices,
            key=lambda index: _stable_order_key(seed, text, query_ids[index]),
        )
        representatives.append(representative)
    if len(representatives) < int(sample_count):
        raise ValueError(
            f"Requested {sample_count} unique novel texts, found {len(representatives)}"
        )
    tie_breakers = [
        _stable_order_key(seed, normalized_texts[index], query_ids[index])
        for index in representatives
    ]
    duration_bins_all = _rank_quantile_bins(
        [durations[index] for index in representatives],
        tie_breakers,
        int(bins),
    )
    margin_bins_all = _rank_quantile_bins(
        [margins[index] for index in representatives],
        tie_breakers,
        int(bins),
    )
    cells: dict[tuple[int, int], list[int]] = {
        (duration_bin, margin_bin): []
        for duration_bin in range(int(bins))
        for margin_bin in range(int(bins))
    }
    representative_position = {
        source_index: position
        for position, source_index in enumerate(representatives)
    }
    for position, source_index in enumerate(representatives):
        cell = (duration_bins_all[position], margin_bins_all[position])
        cells[cell].append(source_index)
    for values in cells.values():
        values.sort(
            key=lambda index: _stable_order_key(
                seed,
                normalized_texts[index],
                query_ids[index],
            )
        )

    cell_order = sorted(cells)
    base = int(sample_count) // len(cell_order)
    remainder = int(sample_count) % len(cell_order)
    target = {
        cell: base + (1 if position < remainder else 0)
        for position, cell in enumerate(cell_order)
    }
    selected: list[int] = []
    shortages: list[tuple[int, int]] = []
    for cell in cell_order:
        take = min(target[cell], len(cells[cell]))
        selected.extend(cells[cell][:take])
        shortages.extend([cell] * (target[cell] - take))
    selected_set = set(selected)
    remaining = [index for index in representatives if index not in selected_set]
    for target_cell in shortages:
        if not remaining:
            raise RuntimeError("Internal stratified-selection underflow")
        chosen = min(
            remaining,
            key=lambda index: (
                abs(duration_bins_all[representative_position[index]] - target_cell[0])
                + abs(margin_bins_all[representative_position[index]] - target_cell[1]),
                _stable_order_key(
                    seed,
                    normalized_texts[index],
                    query_ids[index],
                ),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    selected.sort(
        key=lambda index: (
            duration_bins_all[representative_position[index]],
            margin_bins_all[representative_position[index]],
            _stable_order_key(seed, normalized_texts[index], query_ids[index]),
        )
    )
    selected_duration_bins = tuple(
        duration_bins_all[representative_position[index]] for index in selected
    )
    selected_margin_bins = tuple(
        margin_bins_all[representative_position[index]] for index in selected
    )
    digest = hashlib.sha256()
    for index, duration_bin, margin_bin in zip(
        selected,
        selected_duration_bins,
        selected_margin_bins,
    ):
        digest.update(
            (
                f"{query_ids[index]}\0{normalized_texts[index]}\0"
                f"{duration_bin}\0{margin_bin}\n"
            ).encode("utf-8")
        )
    return StratifiedSelection(
        indices=tuple(selected),
        duration_bins=selected_duration_bins,
        margin_bins=selected_margin_bins,
        selection_hash=digest.hexdigest(),
        requested_count=int(sample_count),
        available_unique_texts=len(representatives),
    )


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


def cluster_bootstrap_interval(
    values: Sequence[float] | np.ndarray | torch.Tensor,
    clusters: Sequence[object],
    *,
    samples: int = 10_000,
    seed: int = 1234,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap an equal-weight mean after averaging rows within clusters."""

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().double().numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("values must have a row dimension")
    if array.shape[0] != len(clusters):
        raise ValueError("values and clusters must have the same row count")
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError("values must be non-empty and finite")
    if int(samples) < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    cluster_rows: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        key = str(cluster)
        if not key:
            raise ValueError(f"clusters[{index}] is empty")
        cluster_rows.setdefault(key, []).append(index)
    cluster_means = np.stack(
        [array[cluster_rows[key]].mean(axis=0) for key in sorted(cluster_rows)],
        axis=0,
    )
    cluster_count = cluster_means.shape[0]
    estimate = cluster_means.mean(axis=0)
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(samples), *array.shape[1:]), dtype=np.float64)
    chunk_size = min(256, int(samples))
    for start in range(0, int(samples), chunk_size):
        stop = min(start + chunk_size, int(samples))
        indices = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        draws[start:stop] = cluster_means[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - float(confidence))
    lower = np.quantile(draws, alpha, axis=0)
    upper = np.quantile(draws, 1.0 - alpha, axis=0)
    return BootstrapInterval(
        estimate=_scalar_or_array(np.asarray(estimate)),
        lower=_scalar_or_array(np.asarray(lower)),
        upper=_scalar_or_array(np.asarray(upper)),
        cluster_count=int(cluster_count),
        samples=int(samples),
        seed=int(seed),
        confidence=float(confidence),
    )


def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Holm--Bonferroni adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("p_values must be finite values in [0,1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    factors = np.arange(values.size, 0, -1, dtype=np.float64)
    adjusted_sorted = np.maximum.accumulate(sorted_values * factors)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


__all__ = [
    "BootstrapInterval",
    "DiagnosticCapture",
    "DiagnosticSnapshot",
    "ModuleOutputIntervention",
    "SentenceLayerCapture",
    "StratifiedSelection",
    "analytic_retrieval_prior",
    "cluster_bootstrap_interval",
    "compute_locality_metrics",
    "compute_slot_metrics",
    "compute_text_attention_metrics",
    "conditional_candidate_metrics",
    "deterministic_permutations",
    "deterministic_rademacher",
    "holm_adjust",
    "jensen_shannon_divergence",
    "permute_candidate_tensors",
    "permute_token_tensors",
    "rank_correlation_last_dim",
    "normalized_text_attention_position_ranks",
    "select_stratified_queries",
    "stable_int_seed",
]
