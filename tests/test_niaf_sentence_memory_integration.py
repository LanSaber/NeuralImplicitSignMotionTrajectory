import copy
from datetime import timedelta
import math
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.models import build_continuous_trajectory_field
from NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory import (
    sentence_memory_diagnostics_row,
    sentence_memory_text_subset,
)
from NIAF.continuous_trajectory_field.scripts.evaluate_continuous_trajectory_field import (
    reduce_external_evaluation_metrics,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    _sentence_memory_availability,
    DatasetIndexView,
    _fixed_batch_part_reduction,
    build_development_validation_loader,
    checkpoint_selection_state,
    checkpoint_contract,
    checkpoint_selection_diagnostics,
    compute_batch_losses,
    configure_sentence_memory_trainable_parameters,
    configured_sentence_memory_eval_modes,
    configured_sentence_memory_train_mode,
    deterministic_sentence_shuffle_mask,
    distributed_checkpoint_rng_state,
    distributed_sample_weighted_mean_scalars,
    distributed_validation_metrics,
    evaluate_configured_modes,
    initial_early_stopping_state,
    is_sentence_memory_model,
    is_sentence_memory_parameter,
    is_word_prior_model,
    load_sentence_memory_base_state,
    merge_sentence_memory_batches,
    paired_sentence_memory_losses,
    paired_sentence_memory_model_forward,
    paired_sentence_memory_usage_moments,
    pending_validation_resume_state,
    phase_b_combined_model_forward,
    phase_b_off_distillation,
    requires_scaffold_provider,
    requires_sentence_memory_provider,
    restore_checkpoint_rng_state,
    restore_checkpoint_selection_scores,
    restore_best_infeasible_selection_key,
    restore_early_stopping_state,
    require_finite_training_gradients,
    require_finite_training_losses,
    sentence_memory_behavior_identity,
    sentence_memory_enabled,
    sentence_memory_forward_kwargs,
    sentence_memory_off_baseline,
    sentence_memory_provider_required,
    sentence_memory_query_key,
    sentence_memory_query_ids,
    sentence_memory_association_losses,
    sentence_memory_objective_identity,
    sentence_memory_resume_identity,
    set_sentence_memory_provider_epoch_from_checkpoint,
    update_early_stopping_state,
    validate_fresh_output_directory,
    validate_phase_b_launch,
    validate_phase_b_resume_checkpoint,
    validate_phase_b_scientific_gate,
    validate_phase_b_scientific_gate_settings,
    validate_phase_b_warm_start_checkpoint,
    validate_paired_sentence_memory_training_contract,
    validate_sentence_memory_checkpoint_identity,
    validate_sentence_memory_objective_identity,
    validate_sentence_memory_resume_identity,
    validate_v2_to_v3_text_only_parity,
)
from NIAF.continuous_trajectory_field.sentence_memory import SentenceMemoryBatch
from NIAF.continuous_trajectory_field.sentence_memory import (
    motion_only_shuffle_sentence_memory_batch,
)


def _model_cfg(model_type):
    return {
        "model": {
            "type": model_type,
            "context_hidden_dim": 16,
            "context_layers": 1,
            "field_hidden_dim": 8,
            "field_depth": 1,
            "max_local_fields": 2,
            "frames_per_local_field": 16,
            "dropout": 0.0,
        },
        "conditioning": {
            "temporal_slot_count": 4,
            "temporal_slot_layers": 1,
            "temporal_slot_heads": 4,
            "context_fps": 20.0,
        },
        "sentence_memory": {
            "motion_dim": 8,
            "key_dim": 12,
            "attention_layers": 1,
            "attention_heads": 4,
        },
    }


def _paired_phase_a_cfg():
    cfg = _model_cfg("sentence_memory_continuous_trajectory_field")
    cfg.update(
        {
            "seed": 1234,
            "conditioning": {
                **cfg["conditioning"],
                "word_prior_train_mode": "off",
                "sentence_memory_train_mode": "on",
            },
            "sentence_memory_safety": {
                "enabled": True,
                "paired_corruption": {
                    "enabled": True,
                    "full_shuffle_probability": 0.10,
                    "benefit_margin_relative": 0.001,
                    "ranking_margin_relative": 0.005,
                    "detach_corrupt_ranking": True,
                    "fallback_huber_beta": 0.10,
                },
                "phase_b": {"enabled": False},
            },
            "objective": {
                "lambda_sentence_benefit": 1.0,
                "lambda_sentence_motion_rank": 1.0,
                "lambda_sentence_motion_fallback": 1.0,
                "lambda_sentence_full_shuffle_rank": 1.0,
                "lambda_sentence_full_shuffle_fallback": 1.0,
            },
            "train": {
                "freeze_base": True,
                "unfreeze_base_prefixes": [],
                "early_stopping_patience": 2,
                "early_stopping_min_epochs": 2,
            },
        }
    )
    return cfg


def _canonical_phase_b_gate_settings():
    return {
        "comparison": "flow",
        "primary_subset": "novel_text",
        "gate_metric": "ndtw",
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 1234,
        "confidence": 0.95,
        "minimum_pairs": 2,
        "duration_tolerance_seconds": 1e-7,
        "hand_path_max_relative_degradation": 0.02,
        "parity_tolerance": 1e-7,
    }


def test_v3_is_both_a_word_prior_and_sentence_memory_contract():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "conditioning": {
            "word_prior_train_mode": "off",
            "sentence_memory_train_mode": "on",
        },
        "eval": {
            "word_prior_modes": ["off"],
            "sentence_memory_modes": ["off", "on", "shuffled"],
        },
    }
    assert is_sentence_memory_model(cfg)
    assert is_word_prior_model(cfg)
    assert checkpoint_contract(cfg) == (
        "sentence_memory_continuous_trajectory_field",
        3,
    )
    assert configured_sentence_memory_eval_modes(cfg) == (
        "off",
        "on",
        "shuffled",
    )
    assert not requires_scaffold_provider(cfg)
    cfg["eval"]["sentence_memory_word_prior_mode"] = "on"
    assert requires_scaffold_provider(cfg)


def test_sentence_memory_query_and_forward_adapter():
    text = torch.tensor([[[3.0, 4.0], [0.0, 2.0]], [[8.0, 6.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True], [True, False]])
    keys = sentence_memory_query_key(text, mask)
    assert torch.allclose(keys.norm(dim=-1), torch.ones(2))

    tensors = {
        name: torch.tensor(index)
        for index, name in enumerate(
            (
                "tokens",
                "token_mask",
                "token_tau",
                "candidate_keys",
                "scores",
                "durations",
                "part_validity",
                "candidate_mask",
                "available",
            )
        )
    }
    kwargs = sentence_memory_forward_kwargs(SimpleNamespace(**tensors))
    assert kwargs["sentence_motion_tokens"] is tensors["tokens"]
    assert kwargs["sentence_text_keys"] is tensors["candidate_keys"]
    assert kwargs["sentence_memory_available"] is tensors["available"]


def test_sentence_memory_dropout_endpoints():
    device = torch.device("cpu")
    assert not _sentence_memory_availability(
        4,
        {"conditioning": {"sentence_memory_dropout_probability": 1.0}},
        device,
        "dropout",
    ).any()
    assert _sentence_memory_availability(
        4,
        {"conditioning": {"sentence_memory_dropout_probability": 0.0}},
        device,
        "dropout",
    ).all()
    assert _sentence_memory_availability(4, {}, device, "shuffled").all()


def test_sentence_memory_disabled_forces_text_only_train_and_eval_without_provider():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {"enabled": False},
        "conditioning": {"sentence_memory_train_mode": "on"},
        "eval": {
            "sentence_memory_modes": ["on", "shuffled"],
            "sentence_memory_word_prior_mode": "off",
        },
    }
    assert not sentence_memory_enabled(cfg)
    assert configured_sentence_memory_train_mode(cfg) == "off"
    assert configured_sentence_memory_eval_modes(cfg) == ("off",)
    assert not requires_sentence_memory_provider(cfg)
    for requested_mode in ("on", "shuffled", "dropout"):
        assert not _sentence_memory_availability(
            3, cfg, torch.device("cpu"), requested_mode
        ).any()

    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.evaluate",
        return_value={"pred_loss_endpoint": 1.25},
    ) as evaluate_mock:
        metrics = evaluate_configured_modes(
            None,
            None,
            None,
            None,
            None,
            None,
            cfg,
            torch.device("cpu"),
            sentence_memory_provider=None,
            show_progress=False,
        )
    assert metrics == {"text_only/pred_loss_endpoint": 1.25}
    assert evaluate_mock.call_count == 1
    assert evaluate_mock.call_args.kwargs["sentence_memory_mode"] == "off"

    score, violation, feasible, details = checkpoint_selection_diagnostics(
        {"text_only/pred_loss_endpoint": 1.25}, cfg, return_details=True
    )
    assert score == pytest.approx(1.25)
    assert violation == pytest.approx(0.0)
    assert feasible
    assert details["dual_mode"]["selection_source"] == "text_only"


def test_sentence_memory_dropout_is_sample_stable_across_batch_order():
    cfg = {
        "seed": 19,
        "conditioning": {"sentence_memory_dropout_probability": 0.5},
    }
    batch = {
        "name": ["alpha", "beta", "gamma", "delta"],
        "motion_path": ["a.npz", "b.npz", "c.npz", "d.npz"],
        "index": torch.tensor([11, 12, 13, 14]),
    }
    forward = _sentence_memory_availability(
        4,
        cfg,
        torch.device("cpu"),
        "dropout",
        batch=batch,
        epoch=7,
    )
    order = torch.tensor([3, 1, 0, 2])
    reordered_batch = {
        "name": [batch["name"][index] for index in order.tolist()],
        "motion_path": [batch["motion_path"][index] for index in order.tolist()],
        "index": batch["index"][order],
    }
    reordered = _sentence_memory_availability(
        4,
        cfg,
        torch.device("cpu"),
        "dropout",
        batch=reordered_batch,
        epoch=7,
    )
    assert torch.equal(reordered, forward[order])


def test_sentence_memory_query_ids_survive_subset_reindexing_and_reordering():
    canonical = {
        "name": ["alpha", "beta", "gamma"],
        "motion_path": ["a.npz", "b.npz", "c.npz"],
        "index": torch.tensor([10, 11, 12]),
    }
    identifiers = sentence_memory_query_ids(canonical)
    order = [2, 0]
    subset = {
        "name": [canonical["name"][index] for index in order],
        "motion_path": [canonical["motion_path"][index] for index in order],
        # A standalone subset manifest may reindex these rows locally.
        "index": torch.tensor([0, 1]),
    }
    assert sentence_memory_query_ids(subset) == [identifiers[index] for index in order]
    assert sentence_memory_query_ids({"index": torch.tensor([7])}) == [
        "index_v1:7"
    ]


def test_sentence_shuffle_mask_is_sample_stable_and_merge_preserves_rows():
    available = torch.tensor([True, False, True])
    batch = {"name": ["alpha", "beta", "gamma"]}
    always = deterministic_sentence_shuffle_mask(
        batch,
        available,
        {"sentence_memory_safety": {"shuffle_probability": 1.0}},
        epoch=3,
    )
    assert torch.equal(always, available)
    assert torch.equal(
        deterministic_sentence_shuffle_mask(
            batch,
            available,
            {"sentence_memory_safety": {"shuffle_probability": 1.0}},
            epoch=3,
        ),
        always,
    )

    def memory(offset, provenance, token_length=2):
        def value(*shape, dtype=torch.float32):
            return (
                torch.arange(int(torch.tensor(shape).prod()), dtype=dtype).reshape(
                    shape
                )
                + offset
            )

        return SentenceMemoryBatch(
            tokens=value(3, 2, token_length, 2),
            token_mask=torch.ones(3, 2, token_length, dtype=torch.bool),
            token_tau=value(3, 2, token_length),
            candidate_mask=torch.ones(3, 2, dtype=torch.bool),
            candidate_keys=value(3, 2, 4),
            scores=value(3, 2),
            durations=value(3, 2),
            duration_log_gap=value(3, 2),
            part_validity=value(3, 2, token_length, 4),
            ids=value(3, 2, dtype=torch.int64),
            available=available,
            provenance=provenance,
        )

    normal = memory(0, {"mode": "on"})
    shuffled = memory(100, {"mode": "shuffled"}, token_length=3)
    selector = torch.tensor([False, True, True])
    merged = merge_sentence_memory_batches(normal, shuffled, selector)
    assert torch.equal(merged.tokens[0, :, :2], normal.tokens[0])
    assert not merged.token_mask[0, :, 2].any()
    assert torch.equal(merged.tokens[1:], shuffled.tokens[1:])
    assert merged.provenance["sentence_memory_shuffled_mask"] == [False, True, True]


def test_paired_sentence_memory_losses_use_exact_fixed_b_by_four_formulas():
    cfg = _paired_phase_a_cfg()
    target = torch.zeros(2, 1, 256)
    correct = torch.stack(
        (torch.full((1, 256), 1.0), torch.full((1, 256), 5.0))
    ).requires_grad_()
    corrupt = torch.stack(
        (torch.full((1, 256), 3.0), torch.full((1, 256), 1.0))
    ).requires_grad_()
    off = torch.stack(
        (torch.full((1, 256), 2.0), torch.full((1, 256), 4.0))
    ).requires_grad_()
    losses, diagnostics = paired_sentence_memory_losses(
        correct_prediction=correct,
        corrupt_prediction=corrupt,
        off_prediction=off,
        target=target,
        frame_mask=torch.ones(2, 1, dtype=torch.bool),
        correct_available=torch.tensor([True, True]),
        motion_mask=torch.tensor([True, False]),
        full_shuffle_mask=torch.tensor([False, True]),
        cfg=cfg,
    )
    assert float(losses["loss_sentence_benefit"].detach()) == pytest.approx(0.1255)
    assert float(losses["loss_sentence_motion_rank"].detach()) == pytest.approx(0.0)
    assert float(losses["loss_sentence_full_shuffle_rank"].detach()) == pytest.approx(
        0.5025
    )
    assert float(
        losses["loss_sentence_motion_fallback"].detach()
    ) == pytest.approx(0.475)
    assert float(
        losses["loss_sentence_full_shuffle_fallback"].detach()
    ) == pytest.approx(1.475)
    assert torch.allclose(
        diagnostics["off_error"], torch.tensor([[2.0] * 4, [4.0] * 4])
    )

    # Ranking may improve the correct branch, but its comparator and text-off
    # denominator are explicitly stop-gradient quantities.
    losses["loss_sentence_full_shuffle_rank"].backward()
    assert correct.grad is not None
    assert corrupt.grad is None
    assert off.grad is None
    assert float(
        _fixed_batch_part_reduction(
            torch.ones(2, 4), torch.tensor([True, False])
        )
    ) == pytest.approx(0.5)


def test_paired_forward_concatenates_variable_token_lengths_once():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, text_tokens, query_times, **kwargs):
            self.calls.append((text_tokens, query_times, kwargs))
            evidence = kwargs["sentence_motion_tokens"].sum(dim=(1, 2, 3))
            prediction = (
                self.scale * evidence[:, None, None]
            ).expand(-1, query_times.shape[1], 256)
            return {"prediction": prediction}

    def branch(batch_size, token_length, offset):
        return {
            "sentence_motion_tokens": torch.full(
                (batch_size, 2, token_length, 3), float(offset)
            ),
            "sentence_motion_mask": torch.ones(
                batch_size, 2, token_length, dtype=torch.bool
            ),
            "sentence_motion_tau": torch.zeros(batch_size, 2, token_length),
            "sentence_text_keys": torch.ones(batch_size, 2, 4),
            "sentence_scores": torch.ones(batch_size, 2),
            "sentence_durations": torch.ones(batch_size, 2),
            "sentence_candidate_mask": torch.ones(
                batch_size, 2, dtype=torch.bool
            ),
            "sentence_part_validity": torch.ones(
                batch_size, 2, token_length, 4
            ),
            "sentence_memory_available": torch.ones(
                batch_size, dtype=torch.bool
            ),
        }

    model = RecordingModel()
    correct_kwargs = branch(2, 2, 1.0)
    corrupt_kwargs = branch(2, 3, 2.0)
    correct, corrupt = paired_sentence_memory_model_forward(
        model,
        text_tokens=torch.randn(2, 3, 4),
        query_times=torch.linspace(-1.0, 1.0, 5).expand(2, -1),
        text_mask=torch.ones(2, 3, dtype=torch.bool),
        query_mask=torch.ones(2, 5, dtype=torch.bool),
        word_kwargs={},
        correct_sentence_kwargs=correct_kwargs,
        corrupt_sentence_kwargs=corrupt_kwargs,
    )
    assert len(model.calls) == 1
    _text, _times, forwarded = model.calls[0]
    assert forwarded["sentence_motion_tokens"].shape == (4, 2, 3, 3)
    assert not forwarded["sentence_motion_mask"][:2, :, 2].any()
    assert forwarded["sentence_motion_mask"][2:, :, 2].all()
    assert correct["prediction"].shape == corrupt["prediction"].shape == (
        2,
        5,
        256,
    )
    (correct["prediction"].mean() + corrupt["prediction"].mean()).backward()
    assert model.scale.grad is not None


def test_corrupt_slice_cannot_change_ordinary_task_losses():
    cfg = _paired_phase_a_cfg()
    cfg["train"]["epochs"] = 4
    cfg["objective"].update(
        {
            "lambda_sentence_benefit": 0.0,
            "lambda_sentence_motion_rank": 0.0,
            "lambda_sentence_motion_fallback": 0.0,
            "lambda_sentence_full_shuffle_rank": 0.0,
            "lambda_sentence_full_shuffle_fallback": 0.0,
        }
    )
    batch = {
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "length": torch.tensor([2]),
        "duration": torch.tensor([2.0]),
    }
    correct_prediction = torch.ones(1, 2, 256)
    target = torch.zeros_like(correct_prediction)

    def prepared(corrupt_value):
        trajectory = SimpleNamespace(
            log_duration_seconds=torch.ones(1),
            duration_seconds=torch.full((1,), 2.0),
            local_mask=torch.zeros(1, 0, dtype=torch.bool),
            num_local_fields=0,
            sentence_memory_gates=torch.zeros(1, 4, 4),
            sentence_memory_null_mass=None,
            sentence_memory_candidate_mass=None,
            word_prior_gates=torch.zeros(1, 4, 4),
        )
        return {
            "target": target,
            "adapter_context": None,
            "word_prior_available": torch.zeros(1, dtype=torch.bool),
            "outputs": {
                "prediction": correct_prediction,
                "trajectory": trajectory,
                "correction_axis": torch.ones(1, 2, 133),
                "local_correction_axis": torch.ones(1, 2, 133),
                "global_correction_axis": torch.ones(1, 2, 133),
                "local_coverage": torch.ones(1, 2),
                "local_weights": torch.zeros(1, 2, 0),
            },
            "sentence_memory_corrupt_outputs": {
                "prediction": torch.full_like(correct_prediction, corrupt_value)
            },
            "sentence_memory_available": torch.ones(1, dtype=torch.bool),
            "sentence_memory_motion_mask": torch.ones(1, dtype=torch.bool),
            "sentence_memory_full_shuffle_mask": torch.zeros(1, dtype=torch.bool),
            "sentence_memory_shuffled_mask": torch.zeros(1, dtype=torch.bool),
            "sentence_memory_batch": None,
        }

    def endpoint(prediction, *_args, **_kwargs):
        value = prediction.mean()
        return value, {"loss_endpoint": value}

    def auxiliary(outputs, *_args, **_kwargs):
        value = outputs["prediction"].mean()
        return {"loss_coarse": 2.0 * value, "loss_residual": 3.0 * value}

    def finite(prediction, *_args, **_kwargs):
        value = 4.0 * prediction.mean()
        return value, {"loss_dynamics": value}

    patch_root = (
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field"
    )
    results = []
    for corrupt_value in (2.0, 9.0):
        with (
            patch(f"{patch_root}.prepare_field_batch", return_value=prepared(corrupt_value)),
            patch(f"{patch_root}.endpoint_losses", side_effect=endpoint),
            patch(f"{patch_root}.coarse_and_residual_losses", side_effect=auxiliary),
            patch(
                f"{patch_root}.duration_regression_loss",
                return_value=torch.tensor(0.25),
            ),
            patch(
                f"{patch_root}.local_field_regularization",
                return_value={
                    "loss_local_modulation": torch.tensor(0.0),
                    "loss_local_width": torch.tensor(0.0),
                },
            ),
            patch(f"{patch_root}.fk_temporal_dynamics_losses", side_effect=finite),
            patch(
                f"{patch_root}.analytic_fk_dynamics_losses",
                return_value=(torch.tensor(0.5), {"loss_analytic": torch.tensor(0.5)}),
            ),
            patch(
                f"{patch_root}.sentence_memory_off_baseline",
                return_value=torch.zeros_like(correct_prediction),
            ),
        ):
            total, losses, _prepared = compute_batch_losses(
                object(),
                None,
                None,
                None,
                batch,
                None,
                cfg,
                torch.device("cpu"),
                training=True,
            )
        results.append((float(total), {name: float(value) for name, value in losses.items()}))

    ordinary_names = (
        "loss_endpoint",
        "loss_coarse",
        "loss_residual",
        "loss_duration",
        "loss_dynamics",
        "loss_analytic",
        "loss_sentence_gate_sparsity",
        "loss_sentence_delta_sparsity",
    )
    assert results[0][0] == pytest.approx(results[1][0])
    for name in ordinary_names:
        assert results[0][1][name] == pytest.approx(results[1][1][name]), name
    assert results[0][1]["loss_sentence_motion_fallback"] != pytest.approx(
        results[1][1]["loss_sentence_motion_fallback"]
    )


def test_v2_to_v3_load_allows_only_sentence_memory_parameters():
    v2 = build_continuous_trajectory_field(
        _model_cfg("dual_mode_continuous_trajectory_field"), text_dim=12
    )
    v3 = build_continuous_trajectory_field(
        _model_cfg("sentence_memory_continuous_trajectory_field"), text_dim=12
    )
    incompatible = load_sentence_memory_base_state(v3, v2.state_dict())
    assert incompatible.missing_keys
    assert all(is_sentence_memory_parameter(name) for name in incompatible.missing_keys)
    assert not incompatible.unexpected_keys

    summary = configure_sentence_memory_trainable_parameters(
        v3,
        {
            "model": {"type": "sentence_memory_continuous_trajectory_field"},
            "train": {"freeze_base": True},
        },
    )
    assert summary["trainable"]
    assert all(is_sentence_memory_parameter(name) for name in summary["trainable"])

    parity = validate_v2_to_v3_text_only_parity(
        v3,
        {
            "config": _model_cfg("dual_mode_continuous_trajectory_field"),
            "model": v2.state_dict(),
        },
        text_dim=12,
        device=torch.device("cpu"),
    )
    assert parity["prediction_max_abs"] <= 1e-7
    assert parity["duration_max_abs"] <= 1e-7

    batch = 2
    candidates = 2
    tokens = 3
    output = v3(
        text_tokens=torch.randn(batch, 4, 12),
        text_mask=torch.ones(batch, 4, dtype=torch.bool),
        query_times=torch.linspace(-1.0, 1.0, 5).expand(batch, -1),
        sentence_motion_tokens=torch.randn(batch, candidates, tokens, 8),
        sentence_motion_mask=torch.ones(batch, candidates, tokens, dtype=torch.bool),
        sentence_motion_tau=torch.linspace(-1.0, 1.0, tokens).expand(
            batch, candidates, -1
        ),
        sentence_text_keys=torch.randn(batch, candidates, 12),
        sentence_scores=torch.randn(batch, candidates),
        sentence_durations=torch.full((batch, candidates), 2.0),
        sentence_candidate_mask=torch.ones(batch, candidates, dtype=torch.bool),
        sentence_part_validity=torch.ones(batch, candidates, tokens, 4),
        sentence_memory_available=torch.ones(batch, dtype=torch.bool),
    )
    output["prediction"].square().mean().backward()
    for name, parameter in v3.named_parameters():
        if is_sentence_memory_parameter(name):
            continue
        assert parameter.grad is None, name
    assert v3.hypernetwork.sentence_memory_fusion.weight.grad is not None


def test_v3_selection_requires_memory_to_beat_text_baseline():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "selection": {
            "weights": {"pred_loss_endpoint": 1.0},
            "constraints": None,
            "require_sentence_memory_improvement": True,
        },
    }
    metrics = {
        "text_only/pred_loss_endpoint": 2.0,
        "sentence_memory/pred_loss_endpoint": 2.1,
        "shuffled_sentence_memory/pred_loss_endpoint": 3.0,
    }
    score, violation, feasible, details = checkpoint_selection_diagnostics(
        metrics, cfg, return_details=True
    )
    assert score == pytest.approx(2.1)
    assert violation > 0
    assert not feasible
    assert details["dual_mode"]["selection_source"] == "sentence_memory"


def test_v3_selection_requires_memory_to_outperform_shuffled_control():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "selection": {
            "weights": {"pred_loss_endpoint": 1.0},
            "constraints": None,
            "sentence_memory_min_relative_improvement": 0.001,
            "sentence_memory_min_relative_improvement_over_shuffled": 0.001,
        },
    }
    metrics = {
        "text_only/pred_loss_endpoint": 2.0,
        "sentence_memory/pred_loss_endpoint": 1.8,
        "shuffled_sentence_memory/pred_loss_endpoint": 1.7,
    }
    _score, violation, feasible, details = checkpoint_selection_diagnostics(
        metrics, cfg, return_details=True
    )
    assert violation > 0
    assert not feasible
    assert details["dual_mode"]["sentence_memory_shuffled_improvement_violation"] > 0


def test_v3_selection_rejects_either_hand_path_regression():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "selection": {
            "weights": {"pred_loss_endpoint": 1.0},
            "constraints": None,
            "sentence_memory_min_relative_improvement": 0.001,
            "sentence_memory_min_relative_improvement_over_shuffled": 0.001,
            "require_hand_path_nonregression": True,
            "hand_path_max_relative_degradation": 0.02,
        },
    }
    metrics = {
        "text_only/pred_loss_endpoint": 2.0,
        "text_only/pred_loss_path_lhand": 0.5,
        "text_only/pred_loss_path_rhand": 0.5,
        "sentence_memory/pred_loss_endpoint": 1.8,
        "sentence_memory/pred_loss_path_lhand": 0.49,
        "sentence_memory/pred_loss_path_rhand": 0.52,
        "shuffled_sentence_memory/pred_loss_endpoint": 2.1,
    }
    _score, violation, feasible, details = checkpoint_selection_diagnostics(
        metrics, cfg, return_details=True
    )
    assert violation > 0
    assert not feasible
    assert details["dual_mode"]["hand_path_nonregression"]["rhand"]["violation"] > 0


def _paired_selection_metrics(*, memory=0.8, motion=1.0, shuffled=1.1, rmotion=0.1):
    values = {
        "text_only/pred_loss_endpoint": 1.0,
        "text_only/pred_loss_path_lhand": 0.5,
        "text_only/pred_loss_path_rhand": 0.5,
        "sentence_memory/pred_loss_endpoint": memory,
        "sentence_memory/pred_loss_path_lhand": 0.5,
        "sentence_memory/pred_loss_path_rhand": 0.5,
        "motion_shuffled_sentence_memory/pred_loss_endpoint": motion,
        "shuffled_sentence_memory/pred_loss_endpoint": shuffled,
        "paired_sentence_memory/Rmotion": rmotion,
    }
    for part in ("body", "left_hand", "right_hand", "face"):
        values[f"paired_sentence_memory/{part}_Rmotion"] = rmotion
    return values


def test_paired_selection_gates_off_motion_full_and_rmotion():
    cfg = _paired_phase_a_cfg()
    cfg["selection"] = {
        "weights": {"pred_loss_endpoint": 1.0},
        "constraints": None,
        "require_sentence_memory_improvement": True,
        "sentence_memory_min_relative_improvement": 0.001,
        "require_sentence_memory_outperform_shuffled": True,
        "sentence_memory_min_relative_improvement_over_shuffled": 0.001,
        "require_sentence_memory_outperform_motion_shuffled": True,
        "sentence_memory_min_relative_improvement_over_motion_shuffled": 0.001,
        "require_hand_path_nonregression": True,
        "hand_path_max_relative_degradation": 0.02,
        "sentence_memory_min_Rmotion": 0.05,
    }
    _score, violation, feasible, details = checkpoint_selection_diagnostics(
        _paired_selection_metrics(), cfg, return_details=True
    )
    assert feasible
    assert violation == pytest.approx(0.0)
    assert details["dual_mode"]["Rmotion"] == pytest.approx(0.1)

    cases = (
        (_paired_selection_metrics(memory=1.01), "text-only"),
        (_paired_selection_metrics(memory=1.01, motion=0.9), "motion-shuffled"),
        (_paired_selection_metrics(memory=1.01, shuffled=0.9), "shuffled-memory"),
        (_paired_selection_metrics(rmotion=0.049), "Rmotion"),
    )
    for metrics, reason in cases:
        _score, violation, feasible, details = checkpoint_selection_diagnostics(
            metrics, cfg, return_details=True
        )
        assert not feasible
        assert violation > 0
        assert any(
            reason in rejection
            for rejection in details["rejection_reasons"]
        )


def test_paired_selection_rejects_nonfinite_unselected_diagnostic():
    cfg = _paired_phase_a_cfg()
    cfg["selection"] = {
        "weights": {"pred_loss_endpoint": 1.0},
        "constraints": None,
    }
    metrics = _paired_selection_metrics()
    metrics["sentence_memory/sentence_memory_gate_mean_body"] = float("nan")
    score, violation, feasible, details = checkpoint_selection_diagnostics(
        metrics, cfg, return_details=True
    )
    assert math.isfinite(score)
    assert math.isfinite(violation) and violation > 0
    assert not feasible
    assert "non-finite validation" in details["rejection_reasons"][-1]


def test_training_finite_guards_cover_losses_and_frozen_base_gradients():
    model = torch.nn.Linear(2, 1)
    require_finite_training_losses(
        torch.tensor(1.0),
        {"diagnostic": torch.tensor(2.0)},
        torch.device("cpu"),
        {"enabled": False},
    )
    with pytest.raises(FloatingPointError, match="diagnostic"):
        require_finite_training_losses(
            torch.tensor(1.0),
            {"diagnostic": torch.tensor(float("nan"))},
            torch.device("cpu"),
            {"enabled": False},
        )
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="weight"):
        require_finite_training_gradients(
            model, torch.device("cpu"), {"enabled": False}
        )


def test_rmotion_uses_additive_squares_counts_and_exact_shared_mask():
    correct = torch.zeros(2, 3, 256)
    motion = torch.full_like(correct, 2.0)
    off = torch.full_like(correct, 4.0)
    mask = torch.tensor([[True, True, False], [False, True, False]])
    moments = paired_sentence_memory_usage_moments(correct, motion, off, mask)
    assert moments["all/motion_square_sum"] == pytest.approx(3 * 256 * 4)
    assert moments["all/motion_element_count"] == pytest.approx(3 * 256)
    assert moments["all/off_square_sum"] == pytest.approx(3 * 256 * 16)
    raw = {f"_paired_usage_raw/{name}": value for name, value in moments.items()}
    reduced = distributed_validation_metrics(
        {**raw, "sentence_memory/pred_loss_endpoint": 1.25},
        local_sample_count=2,
        device=torch.device("cpu"),
        dist_info={"enabled": False, "world_size": 1},
    )
    assert reduced["paired_sentence_memory/Rmotion"] == pytest.approx(0.5)
    assert reduced["paired_sentence_memory/body_Rmotion"] == pytest.approx(0.5)
    assert reduced["paired_sentence_memory/all_motion_element_count"] == pytest.approx(
        3 * 256
    )


def test_external_auto_evaluation_materializes_paired_rmotion():
    correct = torch.zeros(2, 2, 256)
    motion = torch.full_like(correct, 1.0)
    off = torch.full_like(correct, 2.0)
    moments = paired_sentence_memory_usage_moments(
        correct,
        motion,
        off,
        torch.ones(2, 2, dtype=torch.bool),
    )
    raw = {f"_paired_usage_raw/{name}": value for name, value in moments.items()}
    reduced = reduce_external_evaluation_metrics(
        {**raw, "sentence_memory/pred_loss_endpoint": 0.75},
        SimpleNamespace(sampler=None, dataset=[0, 1], batch_size=2),
        0,
        torch.device("cpu"),
        {"enabled": False, "world_size": 1},
    )
    assert reduced["paired_sentence_memory/Rmotion"] == pytest.approx(0.5)
    assert reduced["paired_sentence_memory/body_Rmotion"] == pytest.approx(0.5)
    assert not any(name.startswith("_paired_usage_raw/") for name in reduced)


def test_phase_b_selection_enforces_frozen_teacher_text_guard():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "selection": {
            "weights": {"pred_loss_endpoint": 1.0},
            "constraints": None,
            "sentence_memory_min_relative_improvement": 0.001,
            "sentence_memory_min_relative_improvement_over_shuffled": 0.001,
        },
        "sentence_memory_safety": {
            "phase_b": {
                "enabled": True,
                "teacher_validation_selection_score": 2.0,
                "text_only_max_relative_degradation": 0.005,
            }
        },
    }
    metrics = {
        "text_only/pred_loss_endpoint": 2.02,
        "sentence_memory/pred_loss_endpoint": 1.8,
        "shuffled_sentence_memory/pred_loss_endpoint": 2.2,
    }
    _score, violation, feasible, details = checkpoint_selection_diagnostics(
        metrics, cfg, return_details=True
    )
    assert violation > 0
    assert not feasible
    assert details["dual_mode"]["phase_b_text_only_guard"]["violation"] > 0


def test_v3_behavior_identity_and_phase_b_launch_are_strict():
    cfg = {
        "seed": 7,
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {
            "motion_dim": 256,
            "key_dim": 768,
            "attention_layers": 2,
            "attention_heads": 8,
            "k": 8,
            "top_m": 64,
            "sampling": "weighted",
            "filter": {"exclude_same_group": True},
        },
        "sentence_memory_safety": {
            "phase_b": {
                "enabled": True,
                "teacher_checkpoint": "v2_text_only.pt",
            }
        },
    }
    identity = sentence_memory_behavior_identity(cfg)
    checkpoint = {"sentence_memory_behavior_identity": identity}
    validate_sentence_memory_checkpoint_identity(
        checkpoint, None, cfg=cfg, source="unit checkpoint"
    )
    changed = {**cfg, "sentence_memory": {**cfg["sentence_memory"], "k": 4}}
    with pytest.raises(RuntimeError, match="behavior differs"):
        validate_sentence_memory_checkpoint_identity(
            checkpoint, None, cfg=changed, source="unit checkpoint"
        )
    with pytest.raises(ValueError, match="must start with --warm_start"):
        validate_phase_b_launch(cfg)
    with pytest.raises(ValueError, match="phase_b_gate_report"):
        validate_phase_b_launch(cfg, warm_start=Path("phase_a.pt"))
    validate_phase_b_launch(
        cfg,
        warm_start=Path("phase_a.pt"),
        phase_b_gate_report=Path("phase_b_gate.json"),
    )
    validate_phase_b_launch(cfg, resume=Path("phase_b_last.pt"))
    reset_resume_cfg = copy.deepcopy(cfg)
    reset_resume_cfg["train"] = {"reset_optimizer_on_resume": True}
    with pytest.raises(ValueError, match="Exact v3 --resume"):
        validate_phase_b_launch(
            reset_resume_cfg, resume=Path("phase_b_last.pt")
        )
    phase_a_launch_cfg = copy.deepcopy(cfg)
    phase_a_launch_cfg["sentence_memory_safety"]["phase_b"]["enabled"] = False
    with pytest.raises(ValueError, match="reserved for a model-only Phase-B"):
        validate_phase_b_launch(
            phase_a_launch_cfg, warm_start=Path("another_phase_a.pt")
        )
    with pytest.raises(RuntimeError, match="not a Phase-B checkpoint"):
        validate_phase_b_resume_checkpoint(cfg, {"config": {}})
    validate_phase_b_resume_checkpoint(
        cfg,
        {
            "config": cfg,
            "phase_b_scientific_gate": {
                "accepted": True,
                "settings": _canonical_phase_b_gate_settings(),
            },
        },
    )

    phase_a_checkpoint = {
        "v2_to_v3_text_only_parity": {
            "prediction_max_abs": 0.0,
            "duration_max_abs": 0.0,
            "tolerance": 1e-7,
            "passed": True,
        },
        "config": {
            "model": {"type": "sentence_memory_continuous_trajectory_field"},
            "sentence_memory_safety": {"phase_b": {"enabled": False}},
            "train": {
                "freeze_base": True,
                "unfreeze_base_prefixes": [],
                "base_checkpoint": "v2_text_only.pt",
            },
        },
    }
    validate_phase_b_warm_start_checkpoint(
        cfg, phase_a_checkpoint, source="phase-a checkpoint"
    )
    phase_b_checkpoint = copy.deepcopy(phase_a_checkpoint)
    phase_b_checkpoint["config"]["sentence_memory_safety"]["phase_b"]["enabled"] = True
    with pytest.raises(RuntimeError, match="is a Phase-B checkpoint"):
        validate_phase_b_warm_start_checkpoint(
            cfg, phase_b_checkpoint, source="phase-b checkpoint"
        )
    unproven_checkpoint = copy.deepcopy(phase_a_checkpoint)
    del unproven_checkpoint["config"]["sentence_memory_safety"]["phase_b"]["enabled"]
    with pytest.raises(RuntimeError, match="no explicit Phase-A marker"):
        validate_phase_b_warm_start_checkpoint(
            cfg, unproven_checkpoint, source="legacy checkpoint"
        )


def test_export_compacts_sentence_memory_gate_and_null_diagnostics():
    instance = SimpleNamespace(
        sentence_memory_available=torch.tensor([True, False]),
        sentence_memory_gates=torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        ),
        sentence_memory_null_mass=torch.tensor([[0.2, 0.4], [1.0, 1.0]]),
        sentence_memory_candidate_mass=torch.tensor(
            [
                [[0.4, 0.4], [0.3, 0.3]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )

    diagnostics = sentence_memory_diagnostics_row(instance, 0)

    assert diagnostics["available"] is True
    assert diagnostics["gate_mean_by_part"] == pytest.approx(
        {"body": 0.2, "lhand": 0.3, "rhand": 0.4, "face": 0.5}
    )
    assert diagnostics["null_mass_mean"] == pytest.approx(0.3)
    assert diagnostics["candidate_mass_mean"] == pytest.approx(0.7)


def test_phase_b_gate_is_recomputed_and_bound_to_active_bank(tmp_path):
    source_files = {}
    for mode in (
        "text_only",
        "sentence_memory",
        "shuffled_sentence_memory",
    ):
        for kind in ("export_summary", "default_dtw", "pa_dtw"):
            source_files[f"{mode}_{kind}"] = {
                "path": str(tmp_path / f"{mode}_{kind}.json")
            }
    report = {
        "schema_name": "signtrajfield_phase_b_scientific_gate",
        "schema_version": 2,
        "accepted": True,
        "settings": _canonical_phase_b_gate_settings(),
        "provenance": {"source_files": source_files},
        "gate_identity": {"algorithm": "sha256-canonical-json", "digest": "gate"},
    }
    report_path = tmp_path / "gate.json"
    report_path.write_text(__import__("json").dumps(report), encoding="utf-8")
    checkpoint_path = tmp_path / "phase_a.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    parity = {
        "prediction_max_abs": 0.0,
        "duration_max_abs": 0.0,
        "tolerance": 1e-7,
        "passed": True,
    }
    recomputed = {
        "accepted": True,
        "settings": _canonical_phase_b_gate_settings(),
        "gate_identity": report["gate_identity"],
        "checks": {
            "stored_v2_parity": {
                "passed": True,
                "prediction_max_abs": 0.0,
                "duration_max_abs": 0.0,
            }
        },
        "provenance": {
            "bank_id": "bank-1",
            "checkpoint": {"sha256": "checkpoint-sha", "epoch": 3},
        },
    }
    cfg = {
        "sentence_memory": {"resolved_identity": {"bank_id": "bank-1"}},
        "sentence_memory_safety": {"phase_b": {"enabled": True}},
    }
    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "analyze_sentence_memory_phase_b_gate.analyze_phase_b_gate",
        return_value=recomputed,
    ):
        resolved = validate_phase_b_scientific_gate(
            cfg,
            {"v2_to_v3_text_only_parity": parity},
            checkpoint_path,
            report_path,
        )
    assert resolved["accepted"] is True
    assert resolved["bank_id"] == "bank-1"
    assert (
        cfg["sentence_memory_safety"]["phase_b"]["resolved_scientific_gate"] == resolved
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("comparison", "prior", "comparison='flow'"),
        ("primary_subset", "overall", "primary_subset='novel_text'"),
        ("gate_metric", "dtw", "gate_metric='ndtw'"),
        ("bootstrap_samples", 9_999, "at least 10000"),
        ("bootstrap_seed", 17, "bootstrap_seed=1234"),
        ("confidence", 0.90, "95% confidence"),
        ("minimum_pairs", 1, "at least two"),
        ("duration_tolerance_seconds", 1e-6, "weakens"),
        ("hand_path_max_relative_degradation", 0.03, "weakens"),
        ("parity_tolerance", 1e-6, "weakens"),
    ),
)
def test_phase_b_gate_settings_fail_closed_below_canonical_rigor(name, value, message):
    settings = _canonical_phase_b_gate_settings()
    settings[name] = value
    with pytest.raises(RuntimeError, match=message):
        validate_phase_b_scientific_gate_settings(settings)


def test_phase_b_gate_settings_require_explicit_complete_provenance():
    canonical = _canonical_phase_b_gate_settings()
    assert validate_phase_b_scientific_gate_settings(canonical) == canonical
    with pytest.raises(RuntimeError, match="not canonical"):
        validate_phase_b_scientific_gate_settings({})
    stronger = dict(canonical)
    stronger.update(
        bootstrap_samples=20_000,
        minimum_pairs=8,
        duration_tolerance_seconds=1e-8,
        parity_tolerance=1e-8,
    )
    assert validate_phase_b_scientific_gate_settings(stronger) == stronger


def test_phase_b_distills_differentiable_text_only_student_for_every_row():
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, text_tokens, query_times, **kwargs):
            self.calls.append(kwargs)
            prediction = self.offset.expand(
                text_tokens.shape[0], query_times.shape[1], 256
            )
            return {"prediction": prediction}

    class Teacher(torch.nn.Module):
        def forward(self, text_tokens, query_times, **kwargs):
            del kwargs
            return {
                "prediction": torch.zeros(
                    text_tokens.shape[0], query_times.shape[1], 256
                )
            }

    student = Student()
    prepared = {
        "text_tokens": torch.randn(2, 4, 8),
        "text_mask": torch.ones(2, 4, dtype=torch.bool),
        "tau": torch.linspace(-1.0, 1.0, 3).expand(2, -1),
        # Both primary rows used sentence memory.  The auxiliary student pass
        # must nevertheless cover both rows with memory explicitly disabled.
        "sentence_memory_available": torch.ones(2, dtype=torch.bool),
    }
    batch = {"mask": torch.tensor([[True, True, True], [True, True, False]])}
    primary_outputs, student_off_prediction = phase_b_combined_model_forward(
        student,
        text_tokens=prepared["text_tokens"],
        query_times=prepared["tau"],
        text_mask=prepared["text_mask"],
        query_mask=batch["mask"],
        word_kwargs={},
        sentence_kwargs={
            "sentence_memory_available": prepared["sentence_memory_available"]
        },
    )
    prepared["phase_b_student_off_prediction"] = student_off_prediction
    teacher_prediction, student_prediction, loss = phase_b_off_distillation(
        Teacher(),
        prepared,
        batch,
        {"huber_beta": 0.1},
        torch.device("cpu"),
    )
    assert primary_outputs["prediction"].shape == (2, 3, 256)
    assert teacher_prediction.shape == student_prediction.shape == (2, 3, 256)
    assert len(student.calls) == 1
    assert student.calls[0]["word_prior_available"].shape == (4,)
    assert not student.calls[0]["word_prior_available"].any()
    assert torch.equal(
        student.calls[0]["sentence_memory_available"],
        torch.tensor([True, True, False, False]),
    )
    assert float(loss.detach()) == pytest.approx(0.95)
    loss.backward()
    assert float(student.offset.grad) == pytest.approx(1.0)


def test_phase_b_safety_baseline_uses_frozen_teacher_without_student_forward():
    class UnexpectedStudent(torch.nn.Module):
        def forward(self, **kwargs):
            del kwargs
            raise AssertionError("mutable student baseline must not run in Phase B")

    teacher_prediction = torch.randn(2, 3, 256, requires_grad=True)
    baseline = sentence_memory_off_baseline(
        UnexpectedStudent(),
        {
            "text_tokens": torch.randn(2, 4, 8),
            "text_mask": torch.ones(2, 4, dtype=torch.bool),
            "tau": torch.linspace(-1.0, 1.0, 3).expand(2, -1),
        },
        {"mask": torch.ones(2, 3, dtype=torch.bool)},
        {"model": {"type": "sentence_memory_continuous_trajectory_field"}},
        torch.device("cpu"),
        frozen_teacher_prediction=teacher_prediction,
    )
    assert torch.equal(baseline, teacher_prediction)
    assert not baseline.requires_grad


def _gloo_phase_b_single_forward_worker(rank, world_size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(31)
        cfg = _model_cfg("sentence_memory_continuous_trajectory_field")
        cfg["sentence_memory"].update(
            motion_dim=2, key_dim=3, attention_heads=2
        )
        model = build_continuous_trajectory_field(cfg, text_dim=3)
        distributed_model = DistributedDataParallel(
            model, find_unused_parameters=True
        )
        optimizer = torch.optim.SGD(distributed_model.parameters(), lr=1e-3)
        text_tokens = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]], dtype=torch.float32
        )
        text_mask = torch.ones(1, 2, dtype=torch.bool)
        query_times = torch.linspace(-1.0, 1.0, 3).view(1, -1)
        query_mask = torch.ones(1, 3, dtype=torch.bool)
        sentence_kwargs = {
            "sentence_motion_tokens": torch.randn(1, 2, 3, 2),
            "sentence_motion_mask": torch.ones(1, 2, 3, dtype=torch.bool),
            "sentence_motion_tau": torch.linspace(-1.0, 1.0, 3)
            .view(1, 1, 3)
            .expand(1, 2, 3),
            "sentence_text_keys": torch.randn(1, 2, 3),
            "sentence_scores": torch.tensor([[0.9, 0.7]]),
            "sentence_durations": torch.tensor([[2.0, 3.0]]),
            "sentence_candidate_mask": torch.ones(1, 2, dtype=torch.bool),
            "sentence_part_validity": torch.ones(1, 2, 3, 4),
            "sentence_memory_available": torch.ones(1, dtype=torch.bool),
        }
        optimizer.zero_grad(set_to_none=True)
        primary, student_off = phase_b_combined_model_forward(
            distributed_model,
            text_tokens=text_tokens,
            query_times=query_times,
            text_mask=text_mask,
            query_mask=query_mask,
            word_kwargs={},
            sentence_kwargs=sentence_kwargs,
        )
        loss = primary["prediction"].square().mean() + F.smooth_l1_loss(
            student_off, torch.zeros_like(student_off), beta=0.1
        )
        loss.backward()
        gradient = model.hypernetwork.sentence_memory_fusion.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        optimizer.step()
        weight = model.hypernetwork.sentence_memory_fusion.weight.detach()
        gathered = [torch.empty_like(weight) for _ in range(world_size)]
        dist.all_gather(gathered, weight)
        assert all(torch.equal(weight, other) for other in gathered)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_phase_b_uses_one_ddp_forward_graph(tmp_path):
    init_file = (tmp_path / "phase_b_single_forward_gloo_init").resolve()
    mp.spawn(
        _gloo_phase_b_single_forward_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )


def _gloo_paired_corruption_single_forward_worker(rank, world_size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        from NIAF.continuous_trajectory_field.models.sentence_memory_trajectory_hypernetwork import (
            FACTORIZED_SENTENCE_KEY_VALUE_MODE,
            SentenceMemorySlotEncoder,
        )

        class PairedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))
                self.base = torch.nn.Parameter(
                    torch.tensor(0.25), requires_grad=False
                )
                self.text_projection = torch.nn.Linear(3, 8, bias=False)
                self.text_projection.requires_grad_(False)
                self.sentence_memory_encoder = SentenceMemorySlotEncoder(
                    motion_dim=4,
                    key_dim=3,
                    hidden_dim=8,
                    layer_count=1,
                    head_count=2,
                    dropout=0.0,
                    score_temperature=0.10,
                    duration_weight=0.05,
                    retrieval_prior_scale=1.0,
                    key_value_mode=FACTORIZED_SENTENCE_KEY_VALUE_MODE,
                    temporal_prior_mode="none",
                    temporal_prior_sigma=0.25,
                    temporal_prior_scale=1.0,
                )
                self.sentence_memory_association_key = torch.nn.Linear(
                    3, 4, bias=False
                )
                self.sentence_memory_association_motion = torch.nn.Linear(
                    4, 4, bias=False
                )
                self.sentence_memory_association_threshold = torch.nn.Parameter(
                    torch.tensor(0.0)
                )
                self.forward_calls = 0

            def forward(self, text_tokens, query_times, **kwargs):
                self.forward_calls += 1
                text_slots = self.text_projection(text_tokens.mean(dim=1))
                text_slots = text_slots[:, None].expand(-1, 2, -1)
                evidence = self.sentence_memory_encoder(
                    text_slots=text_slots,
                    query_duration=torch.ones(
                        text_tokens.shape[0],
                        dtype=text_tokens.dtype,
                        device=text_tokens.device,
                    ),
                    motion_tokens=kwargs["sentence_motion_tokens"],
                    motion_mask=kwargs["sentence_motion_mask"],
                    text_keys=kwargs["sentence_text_keys"],
                    scores=kwargs["sentence_scores"],
                    durations=kwargs["sentence_durations"],
                    candidate_mask=kwargs["sentence_candidate_mask"],
                    motion_tau=kwargs["sentence_motion_tau"],
                    part_validity=kwargs["sentence_part_validity"],
                    slot_tau=torch.linspace(
                        -1.0,
                        1.0,
                        2,
                        dtype=text_tokens.dtype,
                        device=text_tokens.device,
                    ),
                )
                signal = evidence["slots"].mean(dim=(1, 2, 3))
                key_descriptor = F.normalize(
                    self.sentence_memory_association_key(
                        kwargs["sentence_text_keys"]
                    ),
                    dim=-1,
                )
                motion_descriptor = F.normalize(
                    self.sentence_memory_association_motion(
                        kwargs["sentence_motion_tokens"].mean(dim=2)
                    ),
                    dim=-1,
                )
                association_cosine = (key_descriptor * motion_descriptor).sum(
                    dim=-1
                )
                association_logit = (
                    association_cosine
                    - self.sentence_memory_association_threshold
                ) / 0.10
                prediction = (
                    self.base + self.scale * signal[:, None, None]
                ).expand(-1, query_times.shape[1], 256)
                return {
                    "prediction": prediction,
                    "sentence_memory_association_key_descriptor": key_descriptor,
                    "sentence_memory_association_motion_descriptor": motion_descriptor,
                    "sentence_memory_association_logit": association_logit,
                    "sentence_memory_association_mask": kwargs[
                        "sentence_candidate_mask"
                    ],
                    "sentence_memory_association_threshold": (
                        self.sentence_memory_association_threshold
                    ),
                }

        module = PairedModule()
        model = DistributedDataParallel(module)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        token_value = float(rank + 1)
        frozen_before = module.base.detach().clone()
        frozen_text_before = module.text_projection.weight.detach().clone()

        def branch(value):
            motion_value = torch.tensor(
                [value, value**2, -0.3 * value, 0.5], dtype=torch.float32
            )
            candidate_motion = torch.stack(
                (motion_value, torch.roll(motion_value, shifts=1)), dim=0
            )
            return {
                "sentence_motion_tokens": candidate_motion.view(
                    1, 2, 1, 4
                ).expand(1, 2, 2, 4).clone(),
                "sentence_motion_mask": torch.ones(1, 2, 2, dtype=torch.bool),
                "sentence_motion_tau": torch.zeros(1, 2, 2),
                "sentence_text_keys": torch.tensor(
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
                ),
                "sentence_scores": torch.ones(1, 2),
                "sentence_durations": torch.ones(1, 2),
                "sentence_candidate_mask": torch.ones(1, 2, dtype=torch.bool),
                "sentence_part_validity": torch.ones(1, 2, 2, 4),
                "sentence_memory_available": torch.ones(1, dtype=torch.bool),
                "sentence_candidate_ids": torch.tensor([[10, 11]]),
            }

        memory = SentenceMemoryBatch(
            tokens=torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2),
            token_mask=torch.ones(1, 3, 2, dtype=torch.bool),
            token_tau=torch.zeros(1, 3, 2),
            candidate_mask=torch.ones(1, 3, dtype=torch.bool),
            candidate_keys=torch.ones(1, 3, 3),
            scores=torch.tensor([[0.9, 0.8, 0.7]]),
            durations=torch.ones(1, 3),
            duration_log_gap=torch.zeros(1, 3),
            part_validity=torch.ones(1, 3, 2, 4),
            ids=torch.tensor([[10, 11, 12]]),
            available=torch.ones(1, dtype=torch.bool),
            provenance={"mode": "on"},
            group_ids=torch.tensor([[0, 1, 2]]),
            source_group_ids=torch.tensor([[100, 101, 102]]),
            motion_source_ids=torch.tensor([[10, 11, 12]]),
            motion_source_group_ids=torch.tensor([[0, 1, 2]]),
            motion_source_source_group_ids=torch.tensor([[100, 101, 102]]),
        )
        _shuffled, permutation_before, informative = (
            motion_only_shuffle_sentence_memory_batch(
                memory,
                query_ids=["stable-query"],
                epoch=3,
                seed=1234,
            )
        )
        random.seed(10_000 + rank)
        np.random.seed(20_000 + rank)
        torch.manual_seed(30_000 + rank)
        _resumed, permutation_after, resumed_informative = (
            motion_only_shuffle_sentence_memory_batch(
                memory,
                query_ids=["stable-query"],
                epoch=3,
                seed=1234,
            )
        )
        assert torch.equal(permutation_before, permutation_after)
        assert torch.equal(informative, resumed_informative)
        gathered_permutations = [
            torch.empty_like(permutation_before) for _ in range(world_size)
        ]
        dist.all_gather(gathered_permutations, permutation_before)
        assert all(
            torch.equal(permutation_before, other)
            for other in gathered_permutations
        )

        optimizer.zero_grad(set_to_none=True)
        correct, corrupt = paired_sentence_memory_model_forward(
            model,
            text_tokens=torch.ones(1, 2, 3),
            query_times=torch.linspace(-1.0, 1.0, 3).view(1, -1),
            text_mask=torch.ones(1, 2, dtype=torch.bool),
            query_mask=torch.ones(1, 3, dtype=torch.bool),
            word_kwargs={},
            correct_sentence_kwargs=branch(token_value),
            corrupt_sentence_kwargs=branch(token_value + 1.0),
        )
        losses, _diagnostics = paired_sentence_memory_losses(
            correct_prediction=correct["prediction"],
            corrupt_prediction=corrupt["prediction"],
            off_prediction=torch.zeros_like(correct["prediction"]),
            target=torch.zeros_like(correct["prediction"]),
            frame_mask=torch.ones(1, 3, dtype=torch.bool),
            correct_available=torch.ones(1, dtype=torch.bool),
            motion_mask=torch.tensor([rank == 0]),
            full_shuffle_mask=torch.tensor([rank == 1]),
            cfg=_paired_phase_a_cfg(),
        )
        association_cfg = _paired_phase_a_cfg()
        association_cfg["sentence_memory"]["association_mode"] = (
            "absolute_text_motion_v1"
        )
        association_cfg["objective"].update(
            lambda_sentence_association_bce=0.10,
            lambda_sentence_association_infonce=0.05,
            association_temperature=0.10,
        )
        correct_memory = SentenceMemoryBatch(
            tokens=branch(token_value)["sentence_motion_tokens"],
            token_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            token_tau=torch.zeros(1, 2, 2),
            candidate_mask=torch.ones(1, 2, dtype=torch.bool),
            candidate_keys=torch.ones(1, 2, 3),
            scores=torch.ones(1, 2),
            durations=torch.ones(1, 2),
            duration_log_gap=torch.zeros(1, 2),
            part_validity=torch.ones(1, 2, 2, 4),
            ids=torch.tensor([[10, 11]]),
            available=torch.ones(1, dtype=torch.bool),
            provenance={"mode": "on"},
            group_ids=torch.tensor([[0, 1]]),
            source_group_ids=torch.tensor([[100, 101]]),
            motion_source_ids=torch.tensor([[10, 11]]),
            motion_source_group_ids=torch.tensor([[0, 1]]),
            motion_source_source_group_ids=torch.tensor([[100, 101]]),
        )
        if rank == 0:
            corrupt_memory, _association_permutation, _ = (
                motion_only_shuffle_sentence_memory_batch(
                    correct_memory,
                    query_ids=["stable-query"],
                    epoch=3,
                    seed=1234,
                )
            )
        else:
            corrupt_memory = copy.deepcopy(correct_memory)
            corrupt_memory.ids = torch.tensor([[20, 21]])
            corrupt_memory.group_ids = torch.tensor([[2, 3]])
            corrupt_memory.source_group_ids = torch.tensor([[200, 201]])
            corrupt_memory.motion_source_ids = corrupt_memory.ids.clone()
            corrupt_memory.motion_source_group_ids = corrupt_memory.group_ids.clone()
            corrupt_memory.motion_source_source_group_ids = (
                corrupt_memory.source_group_ids.clone()
            )
        association_losses, association_diagnostics = (
            sentence_memory_association_losses(
                correct_outputs=correct,
                corrupt_outputs=corrupt,
                correct_memory=correct_memory,
                corrupt_memory=corrupt_memory,
                motion_row_mask=torch.tensor([rank == 0]),
                full_row_mask=torch.tensor([rank == 1]),
                cfg=association_cfg,
            )
        )
        local_counts = torch.tensor(
            [
                float(rank == 0),
                float(rank == 1),
                float(losses["loss_sentence_motion_fallback"].detach()),
                float(losses["loss_sentence_full_shuffle_fallback"].detach()),
            ]
        )
        gathered_counts = [
            torch.empty_like(local_counts) for _ in range(world_size)
        ]
        dist.all_gather(gathered_counts, local_counts)
        assert gathered_counts[0][0] == 1 and gathered_counts[0][1] == 0
        assert gathered_counts[1][0] == 0 and gathered_counts[1][1] == 1
        assert torch.isfinite(gathered_counts[0][2])
        assert float(gathered_counts[0][2]) > 0.0
        assert gathered_counts[0][3] == 0
        assert gathered_counts[1][2] == 0
        assert torch.isfinite(gathered_counts[1][3])
        assert float(gathered_counts[1][3]) > 0.0
        (
            sum(losses.values())
            + 0.10 * association_losses["loss_sentence_association_bce"]
            + 0.05 * association_losses["loss_sentence_association_infonce"]
        ).backward()
        assert module.forward_calls == 1
        assert module.scale.grad is not None
        assert torch.isfinite(module.scale.grad)
        assert module.base.grad is None
        assert module.text_projection.weight.grad is None
        assert association_diagnostics["bce"]["positive_count"].item() == 6
        assert association_diagnostics["bce"]["negative_count"].item() == 4
        for parameter in (
            module.sentence_memory_association_key.weight,
            module.sentence_memory_association_motion.weight,
            module.sentence_memory_association_threshold,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert float(parameter.grad.abs().sum()) > 0.0
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.sentence_memory_encoder.parameters()
        )
        optimizer.step()
        assert torch.equal(module.base.detach(), frozen_before)
        assert torch.equal(
            module.text_projection.weight.detach(), frozen_text_before
        )
        gathered = [torch.empty_like(module.scale) for _ in range(world_size)]
        dist.all_gather(gathered, module.scale.detach())
        assert all(torch.equal(module.scale.detach(), value) for value in gathered)
        encoder_parameter = next(module.sentence_memory_encoder.parameters())
        gathered_encoder = [
            torch.empty_like(encoder_parameter) for _ in range(world_size)
        ]
        dist.all_gather(gathered_encoder, encoder_parameter.detach())
        assert all(
            torch.equal(encoder_parameter.detach(), value)
            for value in gathered_encoder
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_factorized_paired_corruption_uses_one_ddp_forward(tmp_path):
    init_file = (tmp_path / "paired_corruption_gloo_init").resolve()
    mp.spawn(
        _gloo_paired_corruption_single_forward_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )


def test_centered_stage_b_resume_rejects_prior_factorized_objective():
    legacy = _paired_phase_a_cfg()
    legacy["sentence_memory"].update(
        key_value_mode="factorized_metadata_motion_v1",
        temporal_prior_mode="none",
    )
    checkpoint = {
        "sentence_memory_resume_identity": sentence_memory_resume_identity(legacy)
    }
    centered = copy.deepcopy(legacy)
    centered["sentence_memory"].update(
        candidate_value_mode="centered_candidate_covariance_v1",
        relevance_gate_mode="frozen_absolute_adjusted_score_v1",
        relevance_slope=2.0,
        relevance_intercept=-0.5,
        resolved_relevance_calibration_identity={"digest": "c" * 64},
        association_mode="absolute_text_motion_v1",
    )
    centered["objective"].update(
        lambda_sentence_association_bce=0.10,
        lambda_sentence_association_infonce=0.05,
        association_temperature=0.10,
    )
    centered["selection"] = {
        "aggregation": "normalized_text_cluster_equal_v1"
    }
    centered["eval"] = {
        "sentence_memory_modes": [
            "off",
            "on",
            "motion_shuffled_n0",
            "motion_shuffled_n1",
            "motion_shuffled_n2",
            "cross_query_motion",
            "full_replacement",
            "joint_tuple_permuted",
            "uniform_final_mass",
            "analytic_prior",
            "association_disabled",
        ],
        "evaluation_corruption": {
            "mode": "fixed_evidence_controls_v1",
            "seed": 1234,
        },
    }
    with pytest.raises(RuntimeError, match="cannot be resumed exactly"):
        validate_sentence_memory_resume_identity(checkpoint, centered)


def test_sentence_memory_behavior_identity_covers_non_state_factory_options():
    cfg = _model_cfg("sentence_memory_continuous_trajectory_field")
    cfg["duration"] = {
        "initial_seconds": 4.0,
        "min_seconds": 0.8,
        "max_seconds": 20.0,
    }
    baseline = sentence_memory_behavior_identity(cfg)
    assert baseline["schema_version"] == 3
    assert baseline["trajectory"]["local_center_mode"] == "uniform"
    assert baseline["trajectory"]["use_retrieval_guidance"] is False

    changes = (
        ("model", "quantile_temperature", 0.031),
        ("model", "local_window_epsilon", 2e-4),
        ("model", "local_body_omega0_first", 16.0),
        ("model", "local_hand_omega0_first", 31.0),
        ("model", "local_face_omega0_first", 21.0),
        ("model", "omega0_first", 17.0),
        ("model", "omega0_hidden", 1.5),
        ("duration", "initial_seconds", 5.0),
    )
    for section, key, value in changes:
        changed = copy.deepcopy(cfg)
        changed.setdefault(section, {})[key] = value
        assert (
            sentence_memory_behavior_identity(changed)["digest"] != baseline["digest"]
        ), key

    ignored_factory_inputs = copy.deepcopy(cfg)
    ignored_factory_inputs["model"]["local_center_mode"] = "learned"
    ignored_factory_inputs["model"]["use_retrieval_guidance"] = True
    assert (
        sentence_memory_behavior_identity(ignored_factory_inputs)["digest"]
        == baseline["digest"]
    )


def test_checkpoint_identity_binds_neighbor_table_contents():
    identity = {
        "bank_id": "bank-1",
        "neighbor_tables": {
            "val": {
                "sha256": "a" * 64,
                "bytes": 123,
                "query_split": "val",
                "top_m": 64,
            }
        },
    }
    provider = SimpleNamespace(identity=copy.deepcopy(identity))
    checkpoint = {"sentence_memory_identity": copy.deepcopy(identity)}
    validate_sentence_memory_checkpoint_identity(checkpoint, provider)

    provider.identity["neighbor_tables"]["val"]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="neighbor tables differ"):
        validate_sentence_memory_checkpoint_identity(checkpoint, provider)


def test_sentence_memory_provider_uses_checkpoint_epoch_for_standalone_retrieval():
    class Provider:
        epoch = None

        def set_epoch(self, epoch):
            self.epoch = epoch

    provider = Provider()
    assert (
        set_sentence_memory_provider_epoch_from_checkpoint(provider, {"epoch": 17})
        == 17
    )
    assert provider.epoch == 17
    assert set_sentence_memory_provider_epoch_from_checkpoint(None, {}) == 0


def test_exact_resume_restores_historical_selection_minima():
    state = checkpoint_selection_state(1.25, 3.5)
    assert restore_checkpoint_selection_scores(
        {"selection_state": state}, require=True
    ) == pytest.approx((1.25, 3.5))
    empty = checkpoint_selection_state(float("inf"), float("inf"))
    best, infeasible = restore_checkpoint_selection_scores(
        {"selection_state": empty}, require=True
    )
    assert math.isinf(best) and math.isinf(infeasible)
    with pytest.raises(RuntimeError, match="historical selection_state"):
        restore_checkpoint_selection_scores({}, require=True)


def test_paired_objective_identity_rejects_any_loss_or_corruption_drift():
    cfg = _paired_phase_a_cfg()
    baseline = sentence_memory_objective_identity(cfg)
    validate_sentence_memory_objective_identity(
        {"sentence_memory_objective_identity": baseline}, cfg
    )
    changes = (
        ("sentence_memory_safety", "benefit_margin_relative", 0.002),
        ("sentence_memory_safety", "full_shuffle_probability", 0.2),
        ("objective", "lambda_sentence_motion_rank", 0.5),
        ("objective", "lambda_sentence_sparsity", 0.0002),
    )
    for section, name, value in changes:
        changed = copy.deepcopy(cfg)
        if section == "sentence_memory_safety":
            changed[section]["paired_corruption"][name] = value
        else:
            changed[section][name] = value
        assert sentence_memory_objective_identity(changed)["digest"] != baseline[
            "digest"
        ]
        with pytest.raises(RuntimeError, match="objective differs"):
            validate_sentence_memory_objective_identity(
                {"sentence_memory_objective_identity": baseline}, changed
            )
    with pytest.raises(RuntimeError, match="no persisted.*objective identity"):
        validate_sentence_memory_objective_identity(
            {"config": copy.deepcopy(cfg)},
            cfg,
            source="old paired checkpoint",
        )

    legacy_cfg = _model_cfg("sentence_memory_continuous_trajectory_field")
    reconstructed = validate_sentence_memory_objective_identity(
        {"config": copy.deepcopy(legacy_cfg)}, legacy_cfg
    )
    assert reconstructed["mode"] == "legacy_replacement_v1"


def test_lexicographic_early_stopping_and_exact_restore():
    state = initial_early_stopping_state()
    state, improved, stopped = update_early_stopping_state(
        state,
        feasible=False,
        normalized_constraint_violation=0.5,
        selection_score_value=1.0,
        epoch=1,
        patience=2,
        minimum_epoch=2,
    )
    assert improved and not stopped
    # Lower score cannot compensate for a worse normalized violation.
    state, improved, stopped = update_early_stopping_state(
        state,
        feasible=False,
        normalized_constraint_violation=0.6,
        selection_score_value=0.1,
        epoch=2,
        patience=2,
        minimum_epoch=2,
    )
    assert not improved and not stopped
    # Any feasible point lexicographically improves over every infeasible one.
    state, improved, stopped = update_early_stopping_state(
        state,
        feasible=True,
        normalized_constraint_violation=0.0,
        selection_score_value=10.0,
        epoch=3,
        patience=2,
        minimum_epoch=2,
    )
    assert improved and not stopped
    for epoch in (4, 5):
        state, improved, stopped = update_early_stopping_state(
            state,
            feasible=True,
            normalized_constraint_violation=0.0,
            selection_score_value=11.0,
            epoch=epoch,
            patience=2,
            minimum_epoch=2,
        )
    assert not improved and stopped
    checkpoint = {
        "selection_state": checkpoint_selection_state(
            10.0, 1.0, early_stopping_state=state
        )
    }
    assert restore_early_stopping_state(checkpoint, require=True) == state

    selection_state = checkpoint_selection_state(
        float("inf"),
        0.25,
        early_stopping_state=initial_early_stopping_state(),
        best_infeasible_key=(0.05, 0.25),
    )
    assert restore_best_infeasible_selection_key(
        {"selection_state": selection_state}, require=True
    ) == pytest.approx((0.05, 0.25))
    corrupted = copy.deepcopy(selection_state)
    corrupted["best_infeasible_key"][1] = 0.3
    with pytest.raises(RuntimeError, match="disagrees"):
        restore_best_infeasible_selection_key(
            {"selection_state": corrupted}, require=True
        )


def test_pending_validation_resume_does_not_advance_or_retrain_epoch():
    checkpoint = {
        "epoch": 2,
        "metrics": {
            "epoch": 2,
            "global_step": 37,
            "train_loss_total": 1.25,
            "validation_pending": 1.0,
        },
    }
    epoch, row = pending_validation_resume_state(
        checkpoint, paired_objective_enabled=True
    )
    assert epoch == 2
    assert row == checkpoint["metrics"]
    row["validation_pending"] = 0.0
    assert checkpoint["metrics"]["validation_pending"] == 1.0
    checkpoint["selection_state"] = checkpoint_selection_state(
        float("inf"),
        float("inf"),
        early_stopping_state=initial_early_stopping_state(),
    )
    assert restore_best_infeasible_selection_key(
        checkpoint, require=True
    ) is None
    assert pending_validation_resume_state(
        {"epoch": 2, "metrics": {"validation_pending": 0.0}},
        paired_objective_enabled=True,
    ) == (None, None)
    with pytest.raises(RuntimeError, match="only for paired Phase A"):
        pending_validation_resume_state(
            checkpoint, paired_objective_enabled=False
        )


def test_development_loader_keeps_canonical_indices_and_never_reads_confirmation():
    class SourceDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.base = SimpleNamespace(
                items=[
                    {"text": "alpha"},
                    {"text": "beta"},
                    {"text": "alpha"},
                    {"text": "gamma"},
                    {"text": "seen"},
                ]
            )
            self.split = "val"
            self.estimated_lengths = [2, 3, 2, 4, 2]
            self.read_indices = []

        def __len__(self):
            return len(self.base.items)

        def __getitem__(self, index):
            self.read_indices.append(int(index))
            return {"index": int(index), "text": self.base.items[index]["text"]}

    dataset = SourceDataset()
    source_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda rows: rows,
    )
    provider = SimpleNamespace(
        identity={
            "bank_id": "bank",
            "neighbor_tables": {
                "val": {"query_manifest_sha256": "manifest"}
            },
        },
        is_seen_text=lambda text: text == "seen",
    )
    cfg = _paired_phase_a_cfg()
    cfg["train"]["epochs"] = 4
    cfg["eval"] = {"num_workers": 0}
    cfg["validation_text_partition"] = {
        "enabled": True,
        "development_text_count": 1,
        "expected_novel_text_count": 3,
        "expected_validation_rows": 5,
        "expected_validation_manifest_sha256": "manifest",
        "expected_bank_id": "bank",
        # Unsalted SHA256 ordering selects the repeated ``alpha`` cluster.
        "expected_development_rows": 2,
        "expected_confirmation_rows": 2,
    }
    loader, sampler, partition = build_development_validation_loader(
        cfg,
        dataset,
        source_loader,
        provider,
        {"enabled": False, "world_size": 1, "rank": 0},
    )
    assert sampler is None
    assert isinstance(loader.dataset, DatasetIndexView)
    assert loader.dataset.base is dataset.base
    expected_indices = [
        index
        for index, label in enumerate(partition.labels)
        if label == "development"
    ]
    loaded_indices = [row["index"] for batch in loader for row in batch]
    assert loaded_indices == expected_indices
    assert dataset.read_indices == expected_indices
    assert all(
        partition.labels[index] == "development" for index in dataset.read_indices
    )
    assert cfg["validation_text_partition"]["confirmation_evaluated_during_training"] is False
    assert cfg["validation_text_partition"]["exact_seen_evaluated_during_training"] is False
    assert cfg["validation_text_partition"]["exact_seen_row_count"] == 1


def test_exact_resume_identity_rejects_behavior_drift_but_allows_relocation():
    cfg = _model_cfg("sentence_memory_continuous_trajectory_field")
    cfg.update(
        {
            "experiment_name": "phase-a",
            "seed": 1234,
            "data": {"data_dir": "/data", "train_split": "train"},
            "loss": {"hand_weight": 5.0},
            "objective": {"lambda_sentence_safe": 1.0},
            "sentence_memory_safety": {
                "shuffle_probability": 0.25,
                "phase_b": {"enabled": False, "gate_report": None},
            },
            "train": {
                "epochs": 10,
                "batch_size": 64,
                "accumulation_steps": 1,
                "sentence_memory_lr": 1e-4,
                "freeze_base": True,
            },
            "eval": {"sentence_memory_modes": ["off", "on", "shuffled"]},
            "selection": {"require_sentence_memory_improvement": True},
            "output": {"out_dir": "/old/run"},
        }
    )
    cfg["sentence_memory"].update(
        {
            "enabled": True,
            "bank_dir": "/shared/old-bank-path",
            "local_bank_env": "SIGNTRAJ_SENTENCE_MEMORY_DIR",
        }
    )
    identity = sentence_memory_resume_identity(cfg)
    checkpoint = {"sentence_memory_resume_identity": identity}
    validate_sentence_memory_resume_identity(checkpoint, cfg)

    relocated = copy.deepcopy(cfg)
    relocated["experiment_name"] = "renamed"
    relocated["output"]["out_dir"] = "/new/run"
    relocated["sentence_memory"]["bank_dir"] = "/node-local/same-bank"
    relocated["train"]["epochs"] = 20
    relocated["train"].pop("base_checkpoint", None)
    relocated["train"]["warm_start_checkpoint"] = "/old/phase-a.pt"
    relocated["train"]["reset_local_branch_on_warm_start"] = True
    validate_sentence_memory_resume_identity(checkpoint, relocated)

    changes = (
        ("sentence_memory", "enabled", False),
        ("conditioning", "sentence_memory_train_mode", "off"),
        ("sentence_memory_safety", "shuffle_probability", 0.5),
        ("objective", "lambda_sentence_safe", 0.5),
        ("train", "batch_size", 32),
        ("train", "accumulation_steps", 2),
        ("train", "sentence_memory_lr", 2e-4),
        ("eval", "sentence_memory_modes", ["off"]),
        ("selection", "require_sentence_memory_improvement", False),
    )
    for section, key, value in changes:
        changed = copy.deepcopy(cfg)
        changed.setdefault(section, {})[key] = value
        with pytest.raises(RuntimeError, match="changed behavior sections"):
            validate_sentence_memory_resume_identity(checkpoint, changed)

    corrupted = copy.deepcopy(identity)
    corrupted["payload"]["seed"] = 99
    with pytest.raises(RuntimeError, match="invalid v3 resume-behavior digest"):
        validate_sentence_memory_resume_identity(
            {"sentence_memory_resume_identity": corrupted}, cfg
        )


def test_exact_resume_restores_python_numpy_and_torch_rng_states():
    original = distributed_checkpoint_rng_state(
        {"enabled": False, "rank": 0, "world_size": 1}
    )
    try:
        random.seed(51)
        np.random.seed(52)
        torch.manual_seed(53)
        continuation = distributed_checkpoint_rng_state(
            {"enabled": False, "rank": 0, "world_size": 1}
        )
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        assert restore_checkpoint_rng_state(
            {"rng_state": continuation},
            {"enabled": False, "rank": 0, "world_size": 1},
            require=True,
        )
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        assert actual[0] == pytest.approx(expected[0])
        assert actual[1] == pytest.approx(expected[1])
        assert torch.equal(actual[2], expected[2])
        with pytest.raises(RuntimeError, match="no exact RNG"):
            restore_checkpoint_rng_state(
                {},
                {"enabled": False, "rank": 0, "world_size": 1},
                require=True,
            )
    finally:
        restore_checkpoint_rng_state(
            {"rng_state": original},
            {"enabled": False, "rank": 0, "world_size": 1},
            require=True,
        )


def test_fresh_run_rejects_nonempty_output_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    validate_fresh_output_directory(empty)
    stale = tmp_path / "stale"
    (stale / "checkpoints").mkdir(parents=True)
    (stale / "checkpoints" / "best.pt").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="nonempty output directory"):
        validate_fresh_output_directory(stale)
    validate_fresh_output_directory(
        stale, resume=stale / "checkpoints" / "last.pt"
    )


def test_off_only_sentence_memory_needs_no_provider_and_marks_subset_unavailable():
    assert not sentence_memory_provider_required("off")
    assert not sentence_memory_provider_required(("off",))
    assert sentence_memory_provider_required(("off", "on"))
    assert sentence_memory_provider_required("shuffled")
    assert not requires_sentence_memory_provider(
        {
            "model": {
                "type": "sentence_memory_continuous_trajectory_field"
            },
            "sentence_memory": {"enabled": True},
            "conditioning": {"sentence_memory_train_mode": "off"},
            "eval": {"sentence_memory_modes": ["off"]},
        }
    )
    assert sentence_memory_text_subset(None, None, 0, "query") == "not_available"


def test_distributed_validation_reduction_weights_uneven_rank_sample_counts():
    def gather_keys(output, local):
        output[:] = [local, local]

    def add_remote_rank(tensor, op):
        del op
        # Remote rank: loss mean 3.0 over three samples.
        tensor += torch.tensor([9.0, 3.0], dtype=tensor.dtype)

    with (
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "train_continuous_trajectory_field.dist.all_gather_object",
            side_effect=gather_keys,
        ),
        patch(
            "NIAF.continuous_trajectory_field.scripts."
            "train_continuous_trajectory_field.dist.all_reduce",
            side_effect=add_remote_rank,
        ),
    ):
        reduced = distributed_sample_weighted_mean_scalars(
            {"loss": 2.0},
            local_sample_count=1,
            device=torch.device("cpu"),
            dist_info={"enabled": True, "world_size": 2},
        )
    assert reduced["loss"] == pytest.approx(2.75)


def test_v3_validation_namespaces_all_configured_controls():
    cfg = {
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "eval": {
            "sentence_memory_modes": ["off", "on", "shuffled"],
            "sentence_memory_word_prior_mode": "off",
        },
    }
    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.evaluate",
        side_effect=[
            {"pred_loss_endpoint": 2.0},
            {"pred_loss_endpoint": 1.5},
            {"pred_loss_endpoint": 2.5},
        ],
    ) as evaluate_mock:
        metrics = evaluate_configured_modes(
            None,
            None,
            None,
            None,
            None,
            None,
            cfg,
            torch.device("cpu"),
            sentence_memory_provider=object(),
            show_progress=False,
        )
    assert metrics == {
        "text_only/pred_loss_endpoint": 2.0,
        "sentence_memory/pred_loss_endpoint": 1.5,
        "shuffled_sentence_memory/pred_loss_endpoint": 2.5,
    }
    assert [
        call.kwargs["sentence_memory_mode"] for call in evaluate_mock.call_args_list
    ] == ["off", "on", "shuffled"]
    assert all(
        call.kwargs["word_prior_mode"] == "off" for call in evaluate_mock.call_args_list
    )


def test_csl_sentence_memory_phase_configs_are_explicit():
    root = Path("NIAF/continuous_trajectory_field/configs")
    phase_a = load_config(
        root / "csl_daily_signtrajfield_v3_sentence_memory_phase_a.yaml"
    )
    phase_b = load_config(
        root / "csl_daily_signtrajfield_v3_sentence_memory_phase_b.yaml"
    )
    assert phase_a["model"]["type"] == "sentence_memory_continuous_trajectory_field"
    assert phase_a["sentence_memory"]["top_m"] == 64
    assert phase_a["sentence_memory"]["sampling"] == "weighted"
    assert phase_a["sentence_memory"]["candidate_dropout_probability"] == 0.10
    assert phase_a["train"]["freeze_base"] is True
    assert phase_a["sentence_memory_safety"]["phase_b"]["enabled"] is False
    assert phase_b["train"]["base_checkpoint"] is None
    assert phase_b["train"]["unfreeze_base_prefixes"]
    assert phase_b["conditioning"]["sentence_memory_train_mode"] == "dropout"
    assert phase_b["sentence_memory_safety"]["phase_b"]["enabled"] is True
    assert (
        phase_b["sentence_memory_safety"]["phase_b"][
            "text_only_max_relative_degradation"
        ]
        == 0.005
    )
    assert phase_b["sentence_memory_safety"]["phase_b"]["gate_report"] is None


def test_motion_contrast_config_is_strict_paired_phase_a_with_four_epoch_cap():
    path = Path(
        "NIAF/continuous_trajectory_field/configs/"
        "csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml"
    )
    cfg = load_config(path)
    resolved = validate_paired_sentence_memory_training_contract(cfg)
    assert resolved["enabled"]
    assert resolved["full_shuffle_probability"] == pytest.approx(0.10)
    assert cfg["train"]["epochs"] == 4
    assert cfg["train"]["early_stopping_patience"] == 2
    assert cfg["train"]["early_stopping_min_epochs"] == 2
    assert cfg["eval"]["sentence_memory_modes"] == [
        "off",
        "on",
        "shuffled",
        "motion_shuffled",
    ]
    extended = copy.deepcopy(cfg)
    extended["train"]["epochs"] = 5
    with pytest.raises(ValueError, match="never train after epoch 4"):
        validate_paired_sentence_memory_training_contract(extended)
