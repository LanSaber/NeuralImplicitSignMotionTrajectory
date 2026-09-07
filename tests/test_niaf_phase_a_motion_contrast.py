from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.phase_a_motion_contrast import (
    cluster_bootstrap_indices,
    holm_adjust,
    paired_lower_is_better,
    rmotion_summary,
)
from NIAF.continuous_trajectory_field.scripts.analyze_phase_a_motion_contrast_confirmation import (
    ConfirmationInputError,
    EXPECTED_EXPERIMENT_NAME,
    _digest_json,
    _prediction_parity,
    _validate_phase_a_prime_checkpoint_contract,
    _validate_all_null_export,
    analyze_confirmation,
    load_mode_export,
)
from NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory import (
    prepare_inference_batch,
)
from NIAF.continuous_trajectory_field.sentence_memory import SentenceMemoryBatch
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    sentence_memory_behavior_identity,
    sentence_memory_objective_identity,
    sentence_memory_resume_identity,
)


def test_motion_contrast_configs_keep_full_and_smoke_contracts_separate():
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "NIAF"
        / "continuous_trajectory_field"
        / "configs"
    )
    full = load_config(
        config_dir
        / "csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml"
    )
    smoke = load_config(
        config_dir
        / "csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1_smoke.yaml"
    )
    assert full["train"]["epochs"] == 4
    assert full["train"]["early_stopping_patience"] == 2
    assert full["train"]["early_stopping_min_epochs"] == 2
    assert full["train"]["max_samples_per_memory_batch"] == 8
    assert full["train"]["max_frames_per_memory_batch"] == 2048
    assert full["selection"]["require_feasible"] is True
    assert full["selection"]["sentence_memory_min_Rmotion"] == pytest.approx(0.05)
    assert full["eval"]["sentence_memory_modes"] == [
        "off",
        "on",
        "shuffled",
        "motion_shuffled",
    ]
    assert full["validation_text_partition"]["expected_development_rows"] == 347
    assert full["validation_text_partition"]["expected_confirmation_rows"] == 728
    assert "seed" not in full["validation_text_partition"]
    assert smoke["train"]["epochs"] == 1
    assert smoke["train"]["early_stopping_patience"] == 0
    assert smoke["selection"]["require_feasible"] is False
    confirmation_launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "NIAF"
        / "confirm_csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1_sbatch.sh"
    ).read_text(encoding="utf-8")
    assert '--v2_config "$V2_CFG"' in confirmation_launcher
    assert '--run_dir "$RUN_DIR"' in confirmation_launcher


def _phase_a_prime_checkpoint_contract_fixture():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "NIAF"
        / "continuous_trajectory_field"
        / "configs"
        / f"{EXPECTED_EXPERIMENT_NAME}.yaml"
    )
    expected = load_config(config_path)
    resolved_without_digest = {
        "schema_name": "synthetic_validation_partition",
        "counts": {
            "rows": 1077,
            "novel_unique_texts": 796,
            "development_unique_texts": 256,
            "confirmation_unique_texts": 540,
            "development_rows": 347,
            "confirmation_rows": 728,
        },
    }
    partition_digest = _digest_json(resolved_without_digest)
    expected["validation_text_partition"][
        "expected_partition_digest"
    ] = partition_digest
    cfg = copy.deepcopy(expected)
    cfg["device"] = "cuda"
    cfg["output"]["out_dir"] = f"/tmp/{EXPECTED_EXPERIMENT_NAME}"
    partition_cfg = cfg["validation_text_partition"]
    partition_cfg.update(
        {
            "partition_digest": partition_digest,
            "resolved_artifact": {
                **resolved_without_digest,
                "partition_digest": partition_digest,
            },
            "development_row_count": 347,
            "exact_seen_row_count": 2,
            "exact_seen_text_count": 1,
            "exact_seen_evaluated_during_training": False,
            "confirmation_evaluated_during_training": False,
        }
    )
    checkpoint = {
        "epoch": 2,
        "config": cfg,
        "sentence_memory_objective_identity": sentence_memory_objective_identity(cfg),
        "sentence_memory_behavior_identity": sentence_memory_behavior_identity(cfg),
        "sentence_memory_resume_identity": sentence_memory_resume_identity(cfg),
        "metrics": {
            "selection_feasible": 1.0,
            "early_stopping_validation_count": 2.0,
            "val_text_only/pred_loss_total": 1.0,
            "val_sentence_memory/pred_loss_total": 0.9,
            "val_shuffled_sentence_memory/pred_loss_total": 1.0,
            "val_motion_shuffled_sentence_memory/pred_loss_total": 1.0,
        },
        "selection_state": {"best_feasible_score": 0.9},
        "rng_state": {
            "schema_version": 1,
            "world_size": 4,
            "rank_states": [{"rank": rank, "state": {}} for rank in range(4)],
        },
    }
    return checkpoint, expected, partition_digest


def test_confirmation_contract_rejects_smoke_and_wrong_objective():
    checkpoint, expected, partition_digest = (
        _phase_a_prime_checkpoint_contract_fixture()
    )
    evidence = _validate_phase_a_prime_checkpoint_contract(
        checkpoint,
        expected_cfg=expected,
        partition_digest=partition_digest,
    )
    assert evidence["development_row_count"] == 347

    smoke = copy.deepcopy(checkpoint)
    smoke["config"]["experiment_name"] = f"{EXPECTED_EXPERIMENT_NAME}_smoke"
    smoke["config"]["train"]["epochs"] = 1
    smoke["epoch"] = 1
    smoke["sentence_memory_objective_identity"] = sentence_memory_objective_identity(
        smoke["config"]
    )
    smoke["sentence_memory_behavior_identity"] = sentence_memory_behavior_identity(
        smoke["config"]
    )
    smoke["sentence_memory_resume_identity"] = sentence_memory_resume_identity(
        smoke["config"]
    )
    with pytest.raises(ConfirmationInputError, match="exact Phase-A"):
        _validate_phase_a_prime_checkpoint_contract(
            smoke,
            expected_cfg=expected,
            partition_digest=partition_digest,
        )

    wrong_objective = copy.deepcopy(checkpoint)
    wrong_objective["config"]["objective"]["lambda_sentence_motion_rank"] = 0.5
    wrong_objective["sentence_memory_objective_identity"] = (
        sentence_memory_objective_identity(wrong_objective["config"])
    )
    wrong_objective["sentence_memory_resume_identity"] = (
        sentence_memory_resume_identity(wrong_objective["config"])
    )
    with pytest.raises(ConfirmationInputError, match="exact full Phase-A"):
        _validate_phase_a_prime_checkpoint_contract(
            wrong_objective,
            expected_cfg=expected,
            partition_digest=partition_digest,
        )


def _memory_batch() -> SentenceMemoryBatch:
    candidate_mask = torch.ones(1, 4, dtype=torch.bool)
    token_mask = torch.ones(1, 4, 2, dtype=torch.bool)
    ranks = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
    tokens = ranks.expand(1, 4, 2, 3).clone()
    return SentenceMemoryBatch(
        tokens=tokens,
        token_mask=token_mask,
        token_tau=torch.zeros(1, 4, 2),
        candidate_mask=candidate_mask,
        candidate_keys=torch.arange(20, dtype=torch.float32).reshape(1, 4, 5),
        scores=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        durations=torch.ones(1, 4),
        duration_log_gap=torch.zeros(1, 4),
        part_validity=torch.ones(1, 4, 2, 4),
        ids=torch.arange(4, dtype=torch.int64).reshape(1, 4),
        available=torch.ones(1, dtype=torch.bool),
        provenance={"mode": "on", "query_seen_text": [False]},
    )


class _TinyModel:
    def __init__(self):
        self.kwargs = None

    def predict_duration(self, text_tokens, text_mask=None):
        duration = torch.full((text_tokens.shape[0],), 2.0)
        return duration.log(), duration

    def predict_lengths(self, text_tokens, text_mask=None, **_kwargs):
        return torch.full((text_tokens.shape[0],), 40, dtype=torch.long)

    def encode_trajectory(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(duration_seconds=torch.full((1,), 2.0))


def test_external_motion_shuffled_export_mode_preserves_metadata_and_deranges_motion():
    original = _memory_batch()
    model = _TinyModel()
    batch = {
        "name": ["query"],
        "motion_path": ["motion.npz"],
        "index": torch.tensor([17]),
        "length": torch.tensor([40]),
    }
    cfg = {
        "seed": 1234,
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {"enabled": True},
        "conditioning": {},
        "data": {"min_frames": 2, "max_frames": 400},
    }
    text_tokens = torch.ones(1, 3, 5)
    text_mask = torch.ones(1, 3, dtype=torch.bool)

    with patch(
        "NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory.encode_batch_text",
        return_value=(text_tokens, text_mask),
    ), patch(
        "NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory.retrieve_sentence_memory",
        return_value=original,
    ) as retrieve:
        result = prepare_inference_batch(
            model,
            object(),
            None,
            batch,
            object(),
            cfg,
            torch.device("cpu"),
            context_fps=20.0,
            length_mode="predicted",
            word_prior_mode="off",
            sentence_memory_provider=object(),
            sentence_memory_mode="motion_shuffled",
            sentence_memory_epoch=3,
        )

    assert retrieve.call_args.kwargs["mode"] == "on"
    shuffled = result["sentence_memory_batch"]
    assert result["sentence_memory_mode"] == "motion_shuffled"
    assert shuffled.provenance["mode"] == "motion_only_shuffle"
    assert result["sentence_memory_motion_shuffle_epoch"] == 3
    assert result["sentence_memory_motion_shuffle_seed"] == 1234
    assert shuffled.provenance["motion_only_shuffle_epoch"] == 3
    assert shuffled.provenance["motion_only_shuffle_seed"] == 1234
    assert torch.equal(shuffled.candidate_keys, original.candidate_keys)
    assert torch.equal(shuffled.scores, original.scores)
    assert not torch.equal(shuffled.tokens, original.tokens)
    assert torch.equal(model.kwargs["sentence_motion_tokens"], shuffled.tokens)


def test_external_all_null_mode_needs_no_provider_or_payload_lookup():
    model = _TinyModel()
    batch = {
        "name": ["query"],
        "motion_path": ["motion.npz"],
        "index": torch.tensor([17]),
        "length": torch.tensor([40]),
    }
    cfg = {
        "seed": 1234,
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {
            "enabled": True,
            "k": 4,
            "motion_dim": 3,
            "key_dim": 5,
        },
        "conditioning": {},
        "data": {"min_frames": 2, "max_frames": 400},
    }
    text_tokens = torch.ones(1, 3, 5)
    text_mask = torch.ones(1, 3, dtype=torch.bool)
    with patch(
        "NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory.encode_batch_text",
        return_value=(text_tokens, text_mask),
    ), patch(
        "NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory.retrieve_sentence_memory",
        side_effect=AssertionError("all-null must not retrieve"),
    ):
        result = prepare_inference_batch(
            model,
            object(),
            None,
            batch,
            object(),
            cfg,
            torch.device("cpu"),
            context_fps=20.0,
            length_mode="predicted",
            word_prior_mode="off",
            sentence_memory_provider=None,
            sentence_memory_mode="all_null",
        )

    memory = result["sentence_memory_batch"]
    assert memory.provenance["payload_reads"] == 0
    assert not memory.candidate_mask.any()
    assert not memory.token_mask.any()
    assert torch.all(memory.ids == -1)
    assert torch.equal(model.kwargs["sentence_motion_tokens"], memory.tokens)


def test_cluster_bootstrap_effects_and_rmotion_are_deterministic():
    indices = cluster_bootstrap_indices(6, samples=200, seed=1234)
    assert np.array_equal(indices, cluster_bootstrap_indices(6, samples=200, seed=1234))
    reference = np.asarray([10.0, 11.0, 9.0, 10.5, 12.0, 8.5])
    correct = reference * 0.99
    effect = paired_lower_is_better(correct, reference, indices)
    assert effect["relative_improvement"] == pytest.approx(0.01)
    assert effect["absolute_difference_ci95"][1] < 0.0

    rmotion = rmotion_summary(
        np.full(6, 0.04), np.full(6, 1.0), indices
    )
    assert rmotion["value"] == pytest.approx(0.2)
    assert rmotion["ci95"] == pytest.approx([0.2, 0.2])


def test_holm_family_is_monotone_and_bounded():
    adjusted = holm_adjust({"full": 0.01, "motion": 0.03})
    assert adjusted == pytest.approx({"full": 0.02, "motion": 0.03})


def test_confirmation_rejects_noncanonical_resampling_before_reading_inputs(tmp_path):
    with pytest.raises(ConfirmationInputError, match="10,000 resamples"):
        analyze_confirmation(
            checkpoint_path=tmp_path / "missing.pt",
            run_dir=tmp_path / EXPECTED_EXPERIMENT_NAME,
            config_path=tmp_path / "missing.yaml",
            v2_checkpoint_path=tmp_path / "missing_v2.pt",
            v2_config_path=tmp_path / "missing_v2.yaml",
            partition_dir=tmp_path / "partition",
            mode_dirs={
                "text_only": tmp_path,
                "sentence_memory": tmp_path,
                "shuffled_sentence_memory": tmp_path,
                "motion_shuffled_sentence_memory": tmp_path,
            },
            all_null_dir=tmp_path,
            v2_text_only_dir=tmp_path,
            bootstrap_samples=999,
            seed=1234,
        )


def test_confirmation_export_loader_rejects_test_before_other_payload_reads(tmp_path):
    mode_dir = tmp_path / "mode"
    mode_dir.mkdir()
    (mode_dir / "export_summary.json").write_text(
        json.dumps({"split": "test"}), encoding="utf-8"
    )
    with pytest.raises(ConfirmationInputError, match="not validation-only"):
        load_mode_export(
            "text_only",
            mode_dir,
            partition={"manifest_rows": []},
            checkpoint_path=tmp_path / "checkpoint.pt",
        )


def test_all_null_integrity_diagnostics_and_prediction_parity(tmp_path):
    rows = []
    for index in range(2):
        first_path = tmp_path / f"first_{index}.npz"
        second_path = tmp_path / f"second_{index}.npz"
        prediction = np.full((3 + index, 330), index + 0.25, dtype=np.float32)
        integrity = {
            "sentence_memory_ids": np.full(4, -1, dtype=np.int64),
            "sentence_memory_candidate_mask": np.zeros(4, dtype=np.bool_),
            "sentence_memory_payload_reads": np.asarray(0, dtype=np.int64),
            "trajectory_sentence_memory_available": np.asarray(False),
            "trajectory_sentence_memory_gates": np.zeros((16, 4), dtype=np.float32),
            "trajectory_sentence_memory_null_mass": np.ones(16, dtype=np.float32),
            "trajectory_sentence_memory_candidate_mass": np.zeros(
                (16, 4), dtype=np.float32
            ),
        }
        np.savez(first_path, rot6d=prediction, **integrity)
        np.savez(second_path, rot6d=prediction)
        rows.append(
            {
                "sample": str(first_path),
                "predicted_duration_seconds": 2.0 + index,
                "sample_lengths": {"fps20": 3 + index},
            }
        )
    first = {
        "dir": tmp_path,
        "rows": rows,
        "durations": np.asarray([2.0, 3.0]),
    }
    second = {
        "dir": tmp_path,
        "rows": [
            {
                **row,
                "sample": str(tmp_path / f"second_{index}.npz"),
            }
            for index, row in enumerate(rows)
        ],
        "durations": np.asarray([2.0, 3.0]),
    }
    assert _validate_all_null_export(first)["passed"] is True
    assert _prediction_parity(first, second) == {
        "prediction_max_abs": 0.0,
        "duration_max_abs": 0.0,
        "prediction_arrays_equal": True,
        "duration_values_equal": True,
        "requires_bitwise_array_equality": False,
        "prediction_tolerance": 1e-7,
        "duration_tolerance": 1e-7,
        "passed": True,
    }
    exact = _prediction_parity(first, second, require_exact=True)
    assert exact["requires_bitwise_array_equality"] is True
    assert exact["prediction_tolerance"] == 0.0
    assert exact["duration_tolerance"] == 0.0
    assert exact["passed"] is True

    perturbed_path = tmp_path / "second_0.npz"
    with np.load(perturbed_path, allow_pickle=False) as payload:
        perturbed = np.asarray(payload["rot6d"]).copy()
    perturbed[0, 0] += np.float32(5e-8)
    np.savez(perturbed_path, rot6d=perturbed)
    assert _prediction_parity(first, second)["passed"] is True
    rejected = _prediction_parity(first, second, require_exact=True)
    assert rejected["prediction_max_abs"] > 0.0
    assert rejected["prediction_max_abs"] < 1e-7
    assert rejected["prediction_arrays_equal"] is False
    assert rejected["passed"] is False
