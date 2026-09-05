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
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    _sentence_memory_availability,
    checkpoint_selection_state,
    checkpoint_contract,
    checkpoint_selection_diagnostics,
    configure_sentence_memory_trainable_parameters,
    configured_sentence_memory_eval_modes,
    configured_sentence_memory_train_mode,
    deterministic_sentence_shuffle_mask,
    distributed_checkpoint_rng_state,
    distributed_sample_weighted_mean_scalars,
    evaluate_configured_modes,
    is_sentence_memory_model,
    is_sentence_memory_parameter,
    is_word_prior_model,
    load_sentence_memory_base_state,
    merge_sentence_memory_batches,
    phase_b_combined_model_forward,
    phase_b_off_distillation,
    requires_scaffold_provider,
    requires_sentence_memory_provider,
    restore_checkpoint_rng_state,
    restore_checkpoint_selection_scores,
    sentence_memory_behavior_identity,
    sentence_memory_enabled,
    sentence_memory_forward_kwargs,
    sentence_memory_off_baseline,
    sentence_memory_provider_required,
    sentence_memory_query_key,
    sentence_memory_resume_identity,
    set_sentence_memory_provider_epoch_from_checkpoint,
    validate_fresh_output_directory,
    validate_phase_b_launch,
    validate_phase_b_resume_checkpoint,
    validate_phase_b_scientific_gate,
    validate_phase_b_scientific_gate_settings,
    validate_phase_b_warm_start_checkpoint,
    validate_sentence_memory_checkpoint_identity,
    validate_sentence_memory_resume_identity,
    validate_v2_to_v3_text_only_parity,
)
from NIAF.continuous_trajectory_field.sentence_memory import SentenceMemoryBatch


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
