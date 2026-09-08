from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from NIAF.continuous_trajectory_field.phase_a_motion_contrast import (
    cluster_bootstrap_indices,
)
from NIAF.continuous_trajectory_field.scripts.analyze_centered_memory_confirmation import (
    ConfirmationInputError,
    PAIR_MODES,
    compute_centered_confirmation_statistics,
    validate_centered_control_pair,
    write_outputs as write_confirmation_outputs,
)
from NIAF.continuous_trajectory_field.scripts.analyze_centered_memory_diagnostic import (
    DIAGNOSTIC_READY_SCHEMA_NAME,
    SCHEMA_NAME,
    _digest_json,
    _layer_rows,
    write_outputs,
)
from NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory import (
    CENTERED_EXPORT_MODES,
    _sentence_memory_motion_payload_digests,
    prepare_inference_batch,
)
from NIAF.continuous_trajectory_field.scripts.publish_centered_export import (
    CenteredExportPublicationError,
    rebase_and_publish_export_directory,
)
from NIAF.continuous_trajectory_field.sentence_memory import SentenceMemoryBatch


def test_centered_confirmation_statistics_apply_all_causal_gates():
    count = 16
    bootstrap = cluster_bootstrap_indices(count, samples=10_000, seed=1234)
    pa = {
        "sentence_memory": np.full(count, 0.90),
        "text_only": np.full(count, 1.00),
        "cross_query_motion": np.full(count, 0.97),
        "full_replacement": np.full(count, 0.98),
        **{mode: np.full(count, 0.96) for mode in PAIR_MODES},
    }
    result = compute_centered_confirmation_statistics(
        pa_ndtw=pa,
        pair_mse={mode: np.full(count, 0.04) for mode in PAIR_MODES},
        off_mse=np.full(count, 1.0),
        hand_path={
            hand: {
                "sentence_memory": np.full(count, 0.99),
                "text_only": np.full(count, 1.0),
            }
            for hand in ("lhand", "rhand")
        },
        bootstrap=bootstrap,
    )
    assert result["pa_ndtw_wholebody"]["passed"] is True
    assert result["Rpair"]["value"] == pytest.approx(0.2)
    assert result["Rpair"]["passed"] is True
    assert result["identity_utility"]["value"] == pytest.approx(0.6)
    assert result["identity_utility"]["passed"] is True
    assert result["hand_path"]["passed"] is True
    for name in ("mean_motion_derangement", "cross_query_motion", "full_replacement"):
        assert result["pa_ndtw_wholebody"]["comparisons"][name][
            "holm_adjusted_p"
        ] < 0.05


def test_centered_confirmation_statistics_reject_weak_pair_identity():
    count = 8
    bootstrap = cluster_bootstrap_indices(count, samples=10_000, seed=1234)
    pa = {
        "sentence_memory": np.full(count, 0.90),
        "text_only": np.full(count, 1.00),
        "cross_query_motion": np.full(count, 0.97),
        "full_replacement": np.full(count, 0.98),
        **{mode: np.full(count, 0.901) for mode in PAIR_MODES},
    }
    result = compute_centered_confirmation_statistics(
        pa_ndtw=pa,
        pair_mse={mode: np.full(count, 1e-6) for mode in PAIR_MODES},
        off_mse=np.full(count, 1.0),
        hand_path={
            hand: {
                "sentence_memory": np.ones(count),
                "text_only": np.ones(count),
            }
            for hand in ("lhand", "rhand")
        },
        bootstrap=bootstrap,
    )
    assert result["Rpair"]["passed"] is False
    assert result["identity_utility"]["passed"] is False


def _base_sample(path: Path, *, ids=(10, 20), attention_mode="learned", **extra):
    ids = np.asarray(ids, dtype=np.int64)
    payload = {
        "sentence_memory_ids": ids,
        "sentence_memory_group_ids": ids + 100,
        "sentence_memory_source_group_ids": ids + 200,
        "sentence_memory_scores": np.asarray([0.8, 0.7]),
        "sentence_memory_durations": np.asarray([1.0, 2.0]),
        "sentence_memory_duration_log_gap": np.asarray([0.1, 0.2]),
        "sentence_memory_candidate_mask": np.asarray([True, True]),
        "sentence_memory_motion_payload_digest": np.asarray(["a" * 64, "b" * 64]),
        "sentence_memory_attention_mode": np.asarray(attention_mode),
        **extra,
    }
    np.savez(path, **payload)


def _export(directory: Path) -> dict:
    return {
        "dir": directory,
        "rows": [{"sample": str(directory / "sample_0000.npz")}],
    }


def test_centered_control_provenance_validates_pair_joint_uniform_and_broadcast(tmp_path):
    correct_dir = tmp_path / "correct"
    correct_dir.mkdir()
    _base_sample(correct_dir / "sample_0000.npz")
    correct = _export(correct_dir)

    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    _base_sample(
        pair_dir / "sample_0000.npz",
        sentence_memory_motion_candidate_permutation=np.asarray([1, 0]),
        sentence_memory_motion_source_ids=np.asarray([20, 10]),
        sentence_memory_motion_payload_digest=np.asarray(["b" * 64, "a" * 64]),
        sentence_memory_motion_shuffle_informative=np.asarray(True),
        sentence_memory_evaluation_corruption_mode=np.asarray(
            "fixed_evidence_controls_v1"
        ),
        sentence_memory_evaluation_corruption_nonce=np.asarray(
            "csl_daily_pair_derangement_v1_n0"
        ),
        sentence_memory_evaluation_corruption_condition=np.asarray(
            "motion_shuffled_n0"
        ),
    )
    assert validate_centered_control_pair(
        correct, _export(pair_dir), mode="motion_shuffled_n0"
    )["passed"]

    joint_dir = tmp_path / "joint"
    joint_dir.mkdir()
    _base_sample(
        joint_dir / "sample_0000.npz",
        ids=(20, 10),
        sentence_memory_group_ids=np.asarray([120, 110]),
        sentence_memory_source_group_ids=np.asarray([220, 210]),
        sentence_memory_scores=np.asarray([0.7, 0.8]),
        sentence_memory_durations=np.asarray([2.0, 1.0]),
        sentence_memory_duration_log_gap=np.asarray([0.2, 0.1]),
        sentence_memory_motion_payload_digest=np.asarray(["b" * 64, "a" * 64]),
        sentence_memory_joint_tuple_candidate_permutation=np.asarray([1, 0]),
        sentence_memory_joint_tuple_informative=np.asarray(True),
        sentence_memory_evaluation_corruption_mode=np.asarray(
            "fixed_evidence_controls_v1"
        ),
        sentence_memory_evaluation_corruption_nonce=np.asarray(
            "csl_daily_joint_tuple_permutation_v1"
        ),
        sentence_memory_evaluation_corruption_condition=np.asarray(
            "joint_tuple_permuted"
        ),
    )
    assert validate_centered_control_pair(
        correct, _export(joint_dir), mode="joint_tuple_permuted"
    )["passed"]

    uniform_dir = tmp_path / "uniform"
    uniform_dir.mkdir()
    _base_sample(
        uniform_dir / "sample_0000.npz",
        attention_mode="uniform_final_candidate_mass",
        sentence_memory_final_part_candidate_mass=np.asarray(
            [[[0.25, 0.25]]], dtype=np.float32
        ),
        sentence_memory_candidate_support=np.asarray(
            [[[True, True]]], dtype=bool
        ),
    )
    assert validate_centered_control_pair(
        correct, _export(uniform_dir), mode="uniform_final_mass"
    )["passed"]

    broadcast_dir = tmp_path / "broadcast"
    broadcast_dir.mkdir()
    _base_sample(
        broadcast_dir / "sample_0000.npz",
        sentence_memory_broadcast_motion_source_rank=np.asarray(1),
        sentence_memory_motion_source_ids=np.asarray([20, 20]),
        sentence_memory_motion_payload_digest=np.asarray(["b" * 64, "b" * 64]),
        sentence_memory_broadcast_motion_informative=np.asarray(True),
        sentence_memory_broadcast_motion_audit_nonce=np.asarray(
            "csl_daily_broadcast_motion_payload_audit_v1"
        ),
    )
    assert validate_centered_control_pair(
        correct, _export(broadcast_dir), mode="broadcast_complete"
    )["passed"]


def test_centered_controls_reject_nonuniform_or_unbound_motion(tmp_path):
    correct_dir = tmp_path / "correct"
    correct_dir.mkdir()
    _base_sample(correct_dir / "sample_0000.npz")
    correct = _export(correct_dir)

    uniform_dir = tmp_path / "uniform"
    uniform_dir.mkdir()
    _base_sample(
        uniform_dir / "sample_0000.npz",
        attention_mode="uniform_final_candidate_mass",
        sentence_memory_final_part_candidate_mass=np.asarray([[[0.2, 0.3]]]),
        sentence_memory_candidate_support=np.asarray([[[True, True]]]),
    )
    with pytest.raises(ConfirmationInputError, match="not exactly uniform"):
        validate_centered_control_pair(
            correct, _export(uniform_dir), mode="uniform_final_mass"
        )

    joint_dir = tmp_path / "joint"
    joint_dir.mkdir()
    _base_sample(
        joint_dir / "sample_0000.npz",
        ids=(20, 10),
        sentence_memory_group_ids=np.asarray([120, 110]),
        sentence_memory_source_group_ids=np.asarray([220, 210]),
        sentence_memory_scores=np.asarray([0.7, 0.8]),
        sentence_memory_durations=np.asarray([2.0, 1.0]),
        sentence_memory_duration_log_gap=np.asarray([0.2, 0.1]),
        # Deliberately leave the motion-package digests in physical order.
        sentence_memory_joint_tuple_candidate_permutation=np.asarray([1, 0]),
        sentence_memory_joint_tuple_informative=np.asarray(True),
        sentence_memory_evaluation_corruption_mode=np.asarray(
            "fixed_evidence_controls_v1"
        ),
        sentence_memory_evaluation_corruption_nonce=np.asarray(
            "csl_daily_joint_tuple_permutation_v1"
        ),
        sentence_memory_evaluation_corruption_condition=np.asarray(
            "joint_tuple_permuted"
        ),
    )
    with pytest.raises(ConfirmationInputError, match="complete motion package"):
        validate_centered_control_pair(
            correct, _export(joint_dir), mode="joint_tuple_permuted"
        )


def test_layerwise_diagnostic_reads_centered_evidence_fields(tmp_path):
    sample = tmp_path / "sample_0000.npz"
    candidate = np.asarray(
        [
            [
                [[0.2, 0.3], [0.1, 0.4], [0.25, 0.25], [0.15, 0.35]],
                [[0.3, 0.2], [0.2, 0.3], [0.1, 0.4], [0.4, 0.1]],
            ],
            [
                [[0.1, 0.4], [0.2, 0.3], [0.3, 0.2], [0.25, 0.25]],
                [[0.4, 0.1], [0.3, 0.2], [0.2, 0.3], [0.1, 0.4]],
            ],
        ],
        dtype=np.float32,
    )
    # [L,S,P,K,U], with token mass summing to candidate mass.
    token = np.repeat(candidate[..., None] / 2.0, 2, axis=-1)
    np.savez(
        sample,
        sentence_memory_layer_part_candidate_mass=candidate,
        sentence_memory_layer_part_null_mass=1.0 - candidate.sum(axis=-1),
        sentence_memory_layer_part_token_mass=token,
        sentence_memory_centered_update=np.ones((2, 2, 4, 5), dtype=np.float32),
        sentence_memory_relevance_gate=np.asarray([0.8, 0.6]),
        sentence_memory_association_gate=np.asarray([0.8, 0.6]),
        sentence_memory_candidate_support=np.ones((2, 4, 2), dtype=bool),
        sentence_memory_token_tau=np.asarray([[-1.0, 1.0], [-1.0, 1.0]]),
        trajectory_sentence_memory_gates=np.full((2, 4), 0.5),
    )
    rows = _layer_rows(
        {"dir": tmp_path, "rows": [{"sample": str(sample)}]}, "on"
    )
    assert len(rows) == 2 * 2 * 4
    assert rows[0]["effective_k"] > 1.0
    assert rows[0]["centered_update_rms"] == pytest.approx(1.0)
    assert rows[0]["relevance_gate_mean"] == pytest.approx(0.7)
    assert rows[0]["absolute_relevance_mass"] == pytest.approx(0.75)
    assert rows[0]["absolute_association_mass"] == pytest.approx(0.5)


def test_diagnostic_ready_contract_is_canonical(tmp_path):
    out = tmp_path / "diagnostic"
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "checkpoint": {"sha256": "a" * 64},
        "authorization": {"authorization_identity": "b" * 64},
    }
    summary["identity"] = _digest_json(summary)
    rows = [{"mode": "on", "layer": 0, "metric": "effective_k", "mean": 2.0}]
    write_outputs(out, summary, rows)
    ready = json.loads((out / "READY").read_text(encoding="utf-8"))
    assert ready["schema_name"] == DIAGNOSTIC_READY_SCHEMA_NAME
    assert ready["diagnostic_identity"] == summary["identity"]
    assert ready["ready_identity"] == _digest_json(
        {key: value for key, value in ready.items() if key != "ready_identity"}
    )
    # An exact completed directory is the only idempotent restart surface.
    write_outputs(out, summary, rows)
    (out / "layerwise_metrics.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ConfirmationInputError, match="detached"):
        write_outputs(out, summary, rows)


def test_confirmation_publication_is_durable_and_idempotent(tmp_path):
    out = tmp_path / "confirmation"
    summary = {
        "schema_name": "signtrajfield_centered_memory_confirmation",
        "schema_version": 1,
        "promoted_centered_memory_checkpoint": False,
    }
    rows = [{"cluster_index": 0, "normalized_text": "x", "value": 1.0}]
    write_confirmation_outputs(out, summary, rows)
    write_confirmation_outputs(out, summary, rows)
    ready = json.loads((out / "READY").read_text(encoding="utf-8"))
    assert ready["confirmation_holdout_spent"] is True
    assert ready["ready_identity"] == _digest_json(
        {key: value for key, value in ready.items() if key != "ready_identity"}
    )


def test_interrupted_diagnostic_publication_is_never_reused(tmp_path):
    out = tmp_path / "diagnostic"
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "authorization": {"authorization_identity": "b" * 64},
    }
    summary["identity"] = _digest_json(summary)
    rows = [{"mode": "on", "layer": 0, "metric": "effective_k", "mean": 2.0}]
    target = (
        "NIAF.continuous_trajectory_field.scripts."
        "analyze_centered_memory_diagnostic._rename_directory_no_replace"
    )
    with patch(target, side_effect=OSError("injected publication failure")):
        with pytest.raises(OSError, match="injected"):
            write_outputs(out, summary, rows)
    assert not out.exists()
    with pytest.raises(ConfirmationInputError, match="Incomplete diagnostic build"):
        write_outputs(out, summary, rows)


def test_export_publication_rebases_durably_and_never_clobbers(tmp_path):
    build = tmp_path / ".on.building.1.0"
    final = tmp_path / "on"
    build.mkdir()
    for name in ("export_summary.json", "sample_manifest_summary.json"):
        (build / name).write_text(
            json.dumps({"path": str(build / "sample_0000.npz")}),
            encoding="utf-8",
        )
    (build / "sample_0000.npz").write_bytes(b"sample")
    rebase_and_publish_export_directory(build, final)
    assert not build.exists()
    assert (final / "sample_0000.npz").read_bytes() == b"sample"
    assert str(final) in (final / "export_summary.json").read_text(encoding="utf-8")

    raced_build = tmp_path / ".on.building.2.0"
    raced_build.mkdir()
    for name in ("export_summary.json", "sample_manifest_summary.json"):
        (raced_build / name).write_text("{}", encoding="utf-8")
    sentinel = final / "sentinel"
    sentinel.write_text("immutable", encoding="utf-8")
    with pytest.raises(CenteredExportPublicationError, match="Refusing to replace"):
        rebase_and_publish_export_directory(raced_build, final)
    assert raced_build.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "immutable"
    assert not (final / raced_build.name).exists()


def test_export_surface_contains_every_centered_control():
    assert set(CENTERED_EXPORT_MODES) == {
        "motion_shuffled_n0",
        "motion_shuffled_n1",
        "motion_shuffled_n2",
        "cross_query_motion",
        "full_replacement",
        "joint_tuple_permuted",
        "uniform_final_mass",
        "association_disabled",
        "broadcast_complete",
    }


def _memory_batch() -> SentenceMemoryBatch:
    ids = torch.tensor([[10, 20, 30]])
    return SentenceMemoryBatch(
        tokens=torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3),
        token_mask=torch.ones(1, 3, 2, dtype=torch.bool),
        token_tau=torch.tensor([[[-1.0, 1.0]] * 3]),
        candidate_mask=torch.ones(1, 3, dtype=torch.bool),
        candidate_keys=torch.arange(15, dtype=torch.float32).reshape(1, 3, 5),
        scores=torch.tensor([[0.9, 0.8, 0.7]]),
        durations=torch.tensor([[1.0, 1.1, 1.2]]),
        duration_log_gap=torch.tensor([[0.0, 0.1, 0.2]]),
        part_validity=torch.ones(1, 3, 2, 4),
        ids=ids,
        group_ids=ids + 100,
        available=torch.ones(1, dtype=torch.bool),
        provenance={"mode": "on", "candidate_group_ids": [[110, 120, 130]]},
    )


def test_motion_package_digest_binds_values_and_structural_fields():
    memory = _memory_batch()
    original = _sentence_memory_motion_payload_digests(memory, 0)
    changed = _memory_batch()
    changed.token_tau[0, 0, 0] = 0.25
    changed_digest = _sentence_memory_motion_payload_digests(changed, 0)
    assert original.shape == (3,)
    assert original.dtype.kind == "U"
    assert original[0] != changed_digest[0]
    assert np.array_equal(original[1:], changed_digest[1:])


class _TinyExportModel:
    def __init__(self):
        self.kwargs = None

    def predict_duration(self, text_tokens, text_mask=None):
        duration = torch.ones(text_tokens.shape[0])
        return duration.log(), duration

    def predict_lengths(self, text_tokens, text_mask=None, **_kwargs):
        return torch.full((text_tokens.shape[0],), 40, dtype=torch.long)

    def encode_trajectory(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(duration_seconds=torch.ones(1))


@pytest.mark.parametrize(
    ("mode", "expected_attention", "expected_retrieval_calls"),
    (
        ("motion_shuffled_n1", None, 1),
        ("cross_query_motion", None, 2),
        ("full_replacement", None, 1),
        ("joint_tuple_permuted", None, 1),
        ("uniform_final_mass", "uniform_final_candidate_mass", 1),
        ("association_disabled", "association_disabled", 1),
        ("analytic_prior", "analytic_prior", 1),
        ("broadcast_complete", None, 1),
    ),
)
def test_centered_export_dispatches_protocol_modes(
    mode, expected_attention, expected_retrieval_calls
):
    model = _TinyExportModel()
    memory = _memory_batch()
    source = _memory_batch()
    source.ids = source.ids + 1000
    batch = {
        "name": ["query"],
        "motion_path": ["query.npy"],
        "text": ["query text"],
        "length": torch.tensor([40]),
    }
    cfg = {
        "seed": 1234,
        "model": {"type": "sentence_memory_continuous_trajectory_field"},
        "sentence_memory": {
            "enabled": True,
            "candidate_value_mode": "centered_candidate_covariance_v1",
        },
        "conditioning": {},
        "data": {"min_frames": 2, "max_frames": 400},
        "eval": {
            "evaluation_corruption": {
                "mode": "fixed_evidence_controls_v1",
                "seed": 1234,
            }
        },
    }
    returned = [memory, source] if mode == "cross_query_motion" else [memory]
    with patch(
        "NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory.encode_batch_text",
        return_value=(torch.ones(1, 2, 5), torch.ones(1, 2, dtype=torch.bool)),
    ), patch(
        "NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory.retrieve_sentence_memory",
        side_effect=returned,
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
            sentence_memory_mode=mode,
        )
    assert retrieve.call_count == expected_retrieval_calls
    assert result["sentence_memory_mode"] == mode
    assert result["sentence_memory_attention_mode"] == (
        expected_attention or "learned"
    )
    if expected_attention is None:
        assert "sentence_memory_attention_mode" not in model.kwargs
    else:
        assert model.kwargs["sentence_memory_attention_mode"] == expected_attention
