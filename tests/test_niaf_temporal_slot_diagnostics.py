from __future__ import annotations

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from NIAF.continuous_trajectory_field.scripts import (
    diagnose_temporal_slots as diagnostic_cli,
)
from NIAF.continuous_trajectory_field.models import (
    build_continuous_trajectory_field,
)
from NIAF.continuous_trajectory_field.sentence_memory import (
    SentenceMemoryBatch,
    motion_only_shuffle_sentence_memory_batch,
)
from NIAF.continuous_trajectory_field.temporal_slot_diagnostics import (
    _average_ranks,
    DiagnosticCapture,
    ModuleOutputIntervention,
    analytic_retrieval_prior,
    cluster_bootstrap_interval,
    compute_locality_metrics,
    compute_slot_metrics,
    compute_text_attention_metrics,
    conditional_candidate_metrics,
    deterministic_permutations,
    deterministic_rademacher,
    holm_adjust,
    normalized_text_attention_position_ranks,
    rank_correlation_last_dim,
    select_stratified_queries,
)


TEXT_DIM = 24
MOTION_TOKEN_DIM = 12


def _tiny_v3_model():
    config = {
        "model": {
            "type": "sentence_memory_continuous_trajectory_field",
            "context_hidden_dim": 16,
            "context_layers": 1,
            "field_hidden_dim": 8,
            "field_depth": 1,
            "max_local_fields": 2,
            "frames_per_local_field": 16,
            "part_specific_local_experts": True,
            "time_dependent_local_gates": True,
            "dropout": 0.0,
        },
        "conditioning": {
            "temporal_slot_count": 4,
            "temporal_slot_layers": 2,
            "temporal_slot_heads": 4,
            "context_fps": 20.0,
        },
        "sentence_memory": {
            "motion_dim": MOTION_TOKEN_DIM,
            "attention_layers": 2,
            "attention_heads": 4,
            "score_temperature": 0.1,
            "duration_weight": 0.05,
        },
    }
    torch.manual_seed(1001)
    model = build_continuous_trajectory_field(config, text_dim=TEXT_DIM)
    model.eval()
    with torch.no_grad():
        model.hypernetwork.sentence_memory_fusion.weight.normal_(0.0, 0.02)
    return model


def _inputs(*, all_null: bool = False):
    torch.manual_seed(1002)
    batch, text_tokens, candidates, motion_tokens = 2, 6, 3, 5
    text = torch.randn(batch, text_tokens, TEXT_DIM)
    text_mask = torch.tensor(
        [[True, True, True, True, True, False], [True, True, True, True, False, False]]
    )
    candidate_mask = torch.zeros(batch, candidates, dtype=torch.bool)
    if not all_null:
        candidate_mask[:] = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
    motion_mask = candidate_mask[:, :, None].expand(-1, -1, motion_tokens).clone()
    if not all_null:
        motion_mask[0, 1, -2:] = False
        motion_mask[1, 2, -1] = False
    tau = torch.linspace(-1.0, 1.0, motion_tokens).view(1, 1, -1)
    memory = {
        "sentence_motion_tokens": torch.randn(
            batch, candidates, motion_tokens, MOTION_TOKEN_DIM
        ),
        "sentence_motion_mask": motion_mask,
        "sentence_motion_tau": tau.expand(batch, candidates, -1).clone(),
        "sentence_text_keys": torch.randn(batch, candidates, TEXT_DIM),
        "sentence_scores": torch.tensor(
            [[0.91, 0.72, -1.0], [0.83, 0.74, 0.51]], dtype=torch.float32
        ),
        "sentence_durations": torch.tensor(
            [[3.5, 4.0, 1.0], [3.0, 4.0, 5.0]], dtype=torch.float32
        ),
        "sentence_candidate_mask": candidate_mask,
        "sentence_part_validity": torch.ones(
            batch, candidates, motion_tokens, 4
        ),
        "sentence_memory_available": torch.ones(batch, dtype=torch.bool),
    }
    memory["sentence_part_validity"][~motion_mask] = 0.0
    query = torch.linspace(-1.0, 1.0, 9).repeat(batch, 1)
    return text, text_mask, query, memory


def _hook_count(module: torch.nn.Module) -> int:
    return sum(
        len(child._forward_hooks)
        + len(child._forward_pre_hooks)
        + len(child._backward_hooks)
        for child in module.modules()
    )


def test_capture_enter_failure_removes_partially_registered_hooks():
    model = _tiny_v3_model()
    decoder = model.hypernetwork.text_planner.decoder
    original_layers = decoder.layers
    decoder.layers = torch.nn.ModuleList()
    before = _hook_count(model)
    try:
        with pytest.raises(RuntimeError, match="no decoder layers"):
            DiagnosticCapture(model).__enter__()
        assert _hook_count(model) == before
    finally:
        decoder.layers = original_layers


def test_fk_diagnostic_hands_and_body_use_the_pelvis_reference(monkeypatch):
    def fake_parts(_joints, _vertices):
        body = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        left = torch.zeros(1, 2, 3)
        right = torch.zeros(1, 2, 3)
        # The shared helper normalizes this concatenation to body point zero
        # (joint 12), not to the pelvis. Adding body[:, :1] must reconstruct
        # the intended pelvis-relative coordinates.
        wholebody = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [19.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                ]
            ]
        )
        return {"body": body, "lhand": left, "rhand": right, "wholebody": wholebody}

    monkeypatch.setattr(diagnostic_cli, "default_joint_parts_torch", fake_parts)
    joints = torch.zeros(1, 80, 3)
    vertices = torch.zeros(1, 1, 3)
    body, left, right, face = diagnostic_cli._normalized_fk_parts(joints, vertices)
    assert torch.equal(body[..., 0], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(left[..., 0], torch.tensor([[10.0, 11.0]]))
    assert torch.equal(right[..., 0], torch.tensor([[20.0, 21.0]]))
    assert torch.count_nonzero(face) == 0


def test_analytic_prior_masks_and_is_candidate_permutation_equivariant():
    scores = torch.tensor([[0.8, 0.7, 9.0], [0.5, 0.4, 0.3]])
    gaps = torch.tensor([[0.0, 2.0, 0.0], [0.1, 0.2, 0.3]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    reference = analytic_retrieval_prior(scores, gaps, mask)
    assert torch.equal(reference[~mask], torch.zeros_like(reference[~mask]))
    assert torch.allclose(reference.sum(dim=-1), torch.ones(2), atol=1e-7)
    expected = torch.softmax(torch.tensor([0.8, 0.6]) / 0.1, dim=0)
    assert torch.allclose(reference[0, :2], expected, atol=1e-7)

    permutation = torch.tensor([2, 0, 1])
    permuted = analytic_retrieval_prior(
        scores.index_select(1, permutation),
        gaps.index_select(1, permutation),
        mask.index_select(1, permutation),
    )
    assert torch.allclose(
        permuted.index_select(1, torch.argsort(permutation)), reference, atol=1e-7
    )


def test_deterministic_random_helpers_are_repeatable_and_normalized():
    first = deterministic_rademacher((4, 3, 7), seed=19)
    second = deterministic_rademacher((4, 3, 7), seed=19)
    different = deterministic_rademacher((4, 3, 7), seed=20)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.allclose(torch.linalg.vector_norm(first, dim=-1), torch.ones(4, 3))
    assert torch.unique(first.abs()).numel() == 1

    permutations = deterministic_permutations(6, 11, seed=31)
    assert torch.equal(permutations, deterministic_permutations(6, 11, seed=31))
    expected = torch.arange(11)
    for row in permutations:
        assert torch.equal(torch.sort(row).values, expected)


def test_holm_adjust_is_monotone_and_controls_each_raw_p_value():
    raw = np.asarray([0.01, 0.04, 0.03, 0.20], dtype=np.float64)
    adjusted = np.asarray(holm_adjust(raw), dtype=np.float64)
    assert adjusted.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert np.all(adjusted >= raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0.0)


def test_average_ranks_assigns_one_consistent_rank_to_near_ties():
    values = torch.tensor([[0.0, 5e-9, 1.0], [1.0, 5e-9, 0.0]])
    ranks = _average_ranks(values)
    expected = torch.tensor([[0.5, 0.5, 2.0], [2.0, 0.5, 0.5]])
    assert torch.equal(ranks, expected)


def test_selection_and_cluster_bootstrap_are_deterministic():
    count = 96
    query_ids = [f"query-{index:03d}" for index in range(count)]
    texts = [f"unique text {index:03d}" for index in range(count)]
    durations = np.linspace(1.0, 9.0, count)
    margins = np.tile(np.linspace(0.01, 0.40, 24), 4)
    novel = np.ones(count, dtype=bool)
    novel[[3, 17]] = False

    first = select_stratified_queries(
        query_ids,
        texts,
        durations,
        margins,
        novel,
        sample_count=32,
        bins=4,
        seed=1234,
    )
    second = select_stratified_queries(
        query_ids,
        texts,
        durations,
        margins,
        novel,
        sample_count=32,
        bins=4,
        seed=1234,
    )
    assert first == second
    assert asdict(first) == asdict(second)
    assert len(first.indices) == 32
    assert len(set(first.indices)) == 32
    assert all(novel[index] for index in first.indices)
    assert len({texts[index] for index in first.indices}) == 32
    assert len(first.selection_hash) == 64

    reverse = np.arange(count - 1, -1, -1)
    reordered = select_stratified_queries(
        [query_ids[index] for index in reverse],
        [texts[index] for index in reverse],
        durations[reverse],
        margins[reverse],
        novel[reverse],
        sample_count=32,
        bins=4,
        seed=1234,
    )
    assert reordered.selection_hash == first.selection_hash
    assert {query_ids[index] for index in first.indices} == {
        query_ids[reverse[index]] for index in reordered.indices
    }

    values = np.asarray([0.0, 0.2, 0.4, 0.8, 1.0, 1.2])
    clusters = np.asarray(["a", "a", "b", "b", "c", "c"])
    interval_a = cluster_bootstrap_interval(
        values, clusters, samples=500, seed=1234
    )
    interval_b = cluster_bootstrap_interval(
        values, clusters, samples=500, seed=1234
    )
    assert interval_a == interval_b
    assert asdict(interval_a) == asdict(interval_b)
    assert interval_a.estimate == pytest.approx(0.6)
    assert interval_a.cluster_count == 3
    assert interval_a.lower <= interval_a.estimate <= interval_a.upper

    duplicated = cluster_bootstrap_interval(
        np.repeat(values, 2),
        np.repeat(clusters, 2),
        samples=500,
        seed=1234,
    )
    assert duplicated == interval_a


def test_slot_metrics_separate_collapsed_from_temporally_distinct_slots():
    collapsed = torch.ones(2, 4, 8)
    distinct = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)

    collapsed_metrics = compute_slot_metrics(collapsed)
    distinct_metrics = compute_slot_metrics(distinct)

    assert torch.allclose(
        collapsed_metrics["cosine_matrix"], torch.ones(2, 4, 4), atol=1e-7
    )
    assert torch.allclose(
        collapsed_metrics["off_diagonal_mean"], torch.ones(2), atol=1e-7
    )
    assert torch.count_nonzero(collapsed_metrics["across_slot_variance"]) == 0
    assert torch.count_nonzero(collapsed_metrics["centered_effective_rank"]) == 0

    assert torch.allclose(
        distinct_metrics["off_diagonal_mean"], torch.zeros(2), atol=1e-7
    )
    assert torch.all(distinct_metrics["centered_effective_rank"] > 2.9)
    assert torch.all(distinct_metrics["temporal_total_variation"] > 0)
    assert torch.all(
        distinct_metrics["across_slot_variance"]
        > collapsed_metrics["across_slot_variance"]
    )


def test_text_attention_metrics_preserve_heads_and_report_padding_mass():
    attention = torch.zeros(1, 1, 2, 4, 5)
    for slot in range(4):
        attention[0, 0, 0, slot, slot] = 1.0
    attention[0, 0, 1, :, :4] = 0.25
    # Invalid-token mass must be diagnosed and excluded from normalization.
    attention[0, 0, 1, 2, 4] = 0.2
    text_mask = torch.tensor([[True, True, True, True, False]])
    eos_mask = torch.tensor([[False, False, False, True, False]])

    metrics = compute_text_attention_metrics(
        attention,
        text_mask,
        eos_mask=eos_mask,
    )
    normalized = metrics["normalized_attention"]
    assert normalized.shape == attention.shape
    assert torch.count_nonzero(normalized[..., 4]) == 0
    assert torch.allclose(normalized.sum(dim=-1), torch.ones(1, 1, 2, 4))
    assert metrics["invalid_token_mass_max"].item() == pytest.approx(0.2)
    assert metrics["slot_token_position_spearman"][0, 0, 0].item() == pytest.approx(
        1.0
    )
    assert metrics["slot_token_position_spearman"][0, 0, 1].item() == pytest.approx(
        0.0
    )
    assert metrics["expected_token_position_rank"].shape == (1, 1, 2, 4)
    assert torch.equal(
        metrics["expected_token_position_rank"][0, 0, 0],
        torch.arange(4, dtype=attention.dtype),
    )
    assert metrics["normalized_entropy"][0, 0, 0].max().item() == pytest.approx(
        0.0
    )
    assert torch.allclose(
        metrics["normalized_entropy"][0, 0, 1], torch.ones(4), atol=1e-6
    )
    assert metrics["eos_mass"][0, 0, 0, -1].item() == pytest.approx(1.0)
    assert torch.count_nonzero(metrics["adjacent_slot_jsd"][0, 0, 1]) == 0
    assert metrics["pairwise_slot_jsd"].shape == (1, 1, 2, 4, 4)
    assert torch.allclose(
        metrics["pairwise_slot_jsd"],
        metrics["pairwise_slot_jsd"].transpose(-1, -2),
        atol=1e-7,
    )
    assert torch.count_nonzero(
        torch.diagonal(metrics["pairwise_slot_jsd"], dim1=-2, dim2=-1)
    ) == 0


def test_persisted_text_ranks_survive_near_tie_backend_divergence():
    # These are the two normalized attention rows that exposed the production
    # failure. The float32 Torch reduction puts their expected positions just
    # outside the ranking tolerance; NumPy's promoted float64 replay puts them
    # just inside it. Persisting the primary rank makes the estimand invariant
    # to that later numeric backend choice.
    probability = torch.tensor(
        [
            [
                0.0033767083659768105,
                0.03167921304702759,
                0.05423448607325554,
                0.027783069759607315,
                0.0022767214104533195,
                0.8643606901168823,
                0.0024984893389046192,
                0.01216097455471754,
                0.0004423671343829483,
                0.001187234534882009,
            ],
            [
                0.003145763883367181,
                0.029468007385730743,
                0.05722580850124359,
                0.030716238543391228,
                0.002285224385559559,
                0.8586874604225159,
                0.002431824803352356,
                0.014189654029905796,
                0.00046203166129998863,
                0.0013880879851058125,
            ],
        ],
        dtype=torch.float32,
    )
    torch_position = torch.arange(10, dtype=torch.float32) / 9.0
    torch_expected = (probability * torch_position).sum(dim=-1)
    tolerance = 1e-8 + 1e-6 * float(torch_expected.abs().max().item())
    assert float((torch_expected[1] - torch_expected[0]).item()) > tolerance
    persisted_rank = _average_ranks(torch_expected)
    assert torch.equal(persisted_rank, torch.tensor([0.0, 1.0]))

    mask = np.ones((1, 10), dtype=np.bool_)
    token_count = mask.sum(axis=-1)
    numpy_position = (
        np.cumsum(mask, axis=-1, dtype=np.float32) - 1.0
    ) / np.maximum(token_count[:, None] - 1.0, 1.0)
    assert numpy_position.dtype == np.float64
    numpy_expected = np.einsum(
        "sl,l->s", probability.numpy(), numpy_position[0]
    )
    assert abs(float(numpy_expected[1] - numpy_expected[0])) <= tolerance
    assert np.array_equal(
        diagnostic_cli._average_ranks_array(numpy_expected),
        np.asarray([0.5, 0.5]),
    )

    primary = rank_correlation_last_dim(
        torch.arange(2, dtype=torch.float32), persisted_rank
    )
    store = {
        "text_attention": probability.reshape(1, 1, 1, 2, 10).numpy(),
        "text_mask": np.ones((1, 10), dtype=np.uint8),
        "text_expected_token_position_rank": persisted_rank.reshape(1, 1, 1, 2)
        .numpy()
        .astype(np.float32)
    }
    result = diagnostic_cli._text_attention_permutation_test(
        store,
        [
            {
                "novel_text": True,
                "normalized_text": "near tie",
                "text_spearman": float(primary.item()),
            }
        ],
        rows=1,
        batch_size=1,
        permutations=8,
        seed=1234,
    )
    assert result["stored_vs_primary_max_abs"] == 0.0
    assert result["observed_mean_spearman"] == pytest.approx(1.0)


def test_text_rank_source_validation_preserves_passive_batch_geometry(monkeypatch):
    attention = torch.zeros(3, 1, 1, 2, 8, dtype=torch.float32)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :3] = True
    mask[2, :7] = True
    attention[0, 0, 0, 0, 0] = 1.0
    attention[0, 0, 0, 1, 4] = 1.0
    attention[1, 0, 0, 0, 2] = 1.0
    attention[1, 0, 0, 1, 0] = 1.0
    attention[2, 0, 0, 0, 1] = 1.0
    attention[2, 0, 0, 1, 5] = 1.0

    persisted = torch.empty(3, 1, 1, 2)
    for start, stop, extent in ((0, 2, 5), (2, 3, 7)):
        _expected, rank = normalized_text_attention_position_ranks(
            attention[start:stop, ..., :extent],
            mask[start:stop, :extent],
        )
        persisted[start:stop] = rank
    primary = rank_correlation_last_dim(
        torch.arange(2, dtype=torch.float32).view(1, 1, 1, 2),
        persisted,
    ).mean(dim=(1, 2))

    observed_extents = []
    canonical = diagnostic_cli.normalized_text_attention_position_ranks

    def recording_rank(probability, text_mask, **kwargs):
        observed_extents.append(int(probability.shape[-1]))
        return canonical(probability, text_mask, **kwargs)

    monkeypatch.setattr(
        diagnostic_cli,
        "normalized_text_attention_position_ranks",
        recording_rank,
    )
    result = diagnostic_cli._text_attention_permutation_test(
        {
            "text_attention": attention.numpy(),
            "text_mask": mask.numpy().astype(np.uint8),
            "text_expected_token_position_rank": persisted.numpy(),
        },
        [
            {
                "novel_text": True,
                "normalized_text": f"text-{index}",
                "text_spearman": float(primary[index].item()),
            }
            for index in range(3)
        ],
        rows=3,
        batch_size=2,
        permutations=8,
        seed=1234,
    )
    assert observed_extents == [5, 7]
    assert result["stored_vs_primary_max_abs"] == 0.0


def test_text_rank_source_validation_rejects_arbitrary_valid_sum_ranks():
    attention = torch.eye(4, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    # This corruption remains within [0, S-1] and preserves rank sum 0+1+2+3,
    # so bounds/conservation alone cannot detect it.
    corrupted_rank = np.asarray([0.0, 0.0, 3.0, 3.0], dtype=np.float32).reshape(
        1, 1, 1, 4
    )
    with pytest.raises(RuntimeError, match="do not match their source attention"):
        diagnostic_cli._text_attention_permutation_test(
            {
                "text_attention": attention.numpy(),
                "text_mask": np.ones((1, 4), dtype=np.uint8),
                "text_expected_token_position_rank": corrupted_rank,
            },
            [
                {
                    "novel_text": True,
                    "normalized_text": "corrupt",
                    "text_spearman": 1.0,
                }
            ],
            rows=1,
            batch_size=1,
            permutations=4,
            seed=1234,
        )


def _candidate_attention_fixture():
    candidate_mass = torch.tensor(
        [
            [
                [
                    [[0.4, 0.4, 0.0], [0.8, 0.0, 0.0]],
                    [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0]],
                    [[0.0, 0.8, 0.0], [0.4, 0.4, 0.0]],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    null_mass = torch.full((1, 1, 3, 2), 0.2)
    token_weights = torch.tensor([0.25, 0.75])
    token_mass = candidate_mass[..., None] * token_weights
    token_tau = torch.tensor(
        [[[-1.0, 1.0], [-0.5, 0.5], [-0.25, 0.25]]]
    )
    candidate_mask = torch.tensor([[True, True, False]])
    token_mask = candidate_mask[:, :, None].expand(-1, -1, 2).clone()
    slot_tau = torch.tensor([-1.0, 0.0, 1.0])
    return (
        candidate_mass,
        null_mass,
        token_mass,
        token_tau,
        token_mask,
        candidate_mask,
        slot_tau,
    )


def test_candidate_metrics_conserve_mass_and_are_permutation_invariant():
    (
        candidate_mass,
        null_mass,
        token_mass,
        token_tau,
        token_mask,
        candidate_mask,
        slot_tau,
    ) = _candidate_attention_fixture()
    reference = conditional_candidate_metrics(
        candidate_mass,
        null_mass,
        candidate_mask=candidate_mask,
        token_mass=token_mass,
        token_tau=token_tau,
        token_mask=token_mask,
        slot_tau=slot_tau,
    )
    assert torch.allclose(
        reference["mass_conservation_error"],
        torch.zeros_like(null_mass),
        atol=1e-7,
    )
    assert torch.count_nonzero(reference["invalid_candidate_mass_max"]) == 0
    assert torch.count_nonzero(reference["invalid_token_mass_max"]) == 0
    assert torch.count_nonzero(reference["token_candidate_mass_error"]) == 0
    assert reference["slot_jsd_matrix"].shape == (1, 1, 2, 3, 3)
    assert reference["part_jsd_matrix"].shape == (1, 1, 3, 2, 2)
    assert torch.allclose(
        reference["slot_jsd_matrix"],
        reference["slot_jsd_matrix"].transpose(-1, -2),
        atol=1e-7,
    )
    assert torch.allclose(
        reference["part_jsd_matrix"],
        reference["part_jsd_matrix"].transpose(-1, -2),
        atol=1e-7,
    )
    assert reference["effective_candidate_count"][0, 0, 0, 0].item() == pytest.approx(
        2.0
    )
    assert reference["effective_candidate_count"][0, 0, 0, 1].item() == pytest.approx(
        1.0
    )
    assert reference["maximum_candidate_share"][0, 0, 0, 0].item() == pytest.approx(
        0.5
    )

    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    permuted = conditional_candidate_metrics(
        candidate_mass.index_select(-1, permutation),
        null_mass,
        candidate_mask=candidate_mask.index_select(1, permutation),
        token_mass=token_mass.index_select(-2, permutation),
        token_tau=token_tau.index_select(1, permutation),
        token_mask=token_mask.index_select(1, permutation),
        slot_tau=slot_tau,
    )
    assert torch.allclose(
        permuted["conditional_candidate_mass"].index_select(-1, inverse),
        reference["conditional_candidate_mass"],
        atol=1e-7,
    )
    for key in (
        "effective_candidate_count",
        "maximum_candidate_share",
        "slot_jsd_matrix",
        "adjacent_slot_jsd",
        "part_jsd_matrix",
        "expected_memory_tau",
        "memory_time_absolute_error",
        "slot_memory_time_spearman",
    ):
        assert torch.allclose(permuted[key], reference[key], atol=1e-7), key

    token_permuted = conditional_candidate_metrics(
        candidate_mass,
        null_mass,
        candidate_mask=candidate_mask,
        token_mass=token_mass.flip(-1),
        token_tau=token_tau.flip(-1),
        token_mask=token_mask.flip(-1),
        slot_tau=slot_tau,
    )
    assert torch.allclose(
        token_permuted["expected_memory_tau"],
        reference["expected_memory_tau"],
        atol=1e-7,
    )

    padded_mask = token_mask.clone()
    padded_mask[:, 1, 1] = False
    padded = conditional_candidate_metrics(
        candidate_mass,
        null_mass,
        candidate_mask=candidate_mask,
        token_mass=token_mass,
        token_tau=token_tau,
        token_mask=padded_mask,
    )
    assert padded["invalid_token_mass_max"].max() > 0
    assert torch.count_nonzero(padded["token_candidate_mass_error"]) > 0


def test_v3_prediction_is_invariant_to_consistent_memory_token_permutation():
    model = _tiny_v3_model()
    text, text_mask, query, memory = _inputs()
    with torch.inference_mode():
        reference = model(text, query, text_mask=text_mask, **memory)

    token_permutation = torch.tensor([4, 1, 3, 0, 2])
    permuted = dict(memory)
    for name in (
        "sentence_motion_tokens",
        "sentence_motion_mask",
        "sentence_motion_tau",
        "sentence_part_validity",
    ):
        permuted[name] = memory[name].index_select(2, token_permutation)
    with torch.inference_mode():
        reordered = model(text, query, text_mask=text_mask, **permuted)

    assert torch.allclose(
        reordered["prediction"], reference["prediction"], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        reordered["trajectory"].sentence_memory_gates,
        reference["trajectory"].sentence_memory_gates,
        atol=1e-6,
        rtol=1e-6,
    )


def test_candidate_metrics_treat_all_null_attention_as_zero_evidence():
    candidate = torch.zeros(2, 1, 4, 4, 3)
    null = torch.ones(2, 1, 4, 4)
    metrics = conditional_candidate_metrics(candidate, null)
    assert torch.count_nonzero(metrics["conditional_candidate_mass"]) == 0
    assert torch.count_nonzero(metrics["effective_candidate_count"]) == 0
    assert torch.count_nonzero(metrics["maximum_candidate_share"]) == 0
    assert torch.count_nonzero(metrics["mass_conservation_error"]) == 0
    assert torch.all(metrics["argmax_candidate"] == -1)


def test_locality_metrics_distinguish_diagonal_global_and_reversed_response():
    slot_tau = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    query_tau = torch.linspace(-1.0, 1.0, 161)
    distance = query_tau[None, :] - slot_tau[:, None]
    local_curve = torch.exp(-0.5 * (distance / 0.035).square())
    local = local_curve[None, :, None, :].expand(1, -1, 3, -1).clone()
    global_response = torch.ones_like(local)
    reversed_response = local.flip(1)

    local_metrics = compute_locality_metrics(local, slot_tau, query_tau)
    global_metrics = compute_locality_metrics(global_response, slot_tau, query_tau)
    reversed_metrics = compute_locality_metrics(
        reversed_response, slot_tau, query_tau
    )

    assert torch.all(local_metrics["informative"])
    assert torch.all(local_metrics["local_response_fraction"] > 0.99)
    assert torch.all(local_metrics["far_leakage_fraction"] < 1e-6)
    assert torch.all(local_metrics["response_center_absolute_error"] < 1e-3)
    assert torch.all(local_metrics["radius_80"] < 0.08)
    assert torch.allclose(
        local_metrics["slot_response_time_spearman"], torch.ones(1, 3), atol=1e-7
    )

    assert torch.all(
        global_metrics["local_response_fraction"]
        < local_metrics["local_response_fraction"]
    )
    assert torch.all(
        global_metrics["far_leakage_fraction"]
        > local_metrics["far_leakage_fraction"]
    )
    assert torch.all(
        reversed_metrics["slot_response_time_spearman"] < -0.99
    )
    assert reversed_metrics["response_center_absolute_error"].mean() > 0.9


def test_capture_is_read_only_replays_attention_and_removes_every_hook():
    model = _tiny_v3_model()
    text, text_mask, query, memory = _inputs()
    before_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    before_hooks = _hook_count(model)

    with torch.inference_mode():
        baseline = model(text, query, text_mask=text_mask, **memory)
        with DiagnosticCapture(model, replay_tolerance=1e-6) as capture:
            assert _hook_count(model) > before_hooks
            observed = model(text, query, text_mask=text_mask, **memory)
            snapshot = capture.snapshot(cpu=True)
    assert _hook_count(model) == before_hooks
    assert torch.equal(observed["prediction"], baseline["prediction"])
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in before_state.items()
    )

    assert snapshot.planner_slots.shape == (2, 4, 16)
    assert snapshot.slot_tau.shape == (4,)
    assert snapshot.fusion_input.shape == (2, 4, 16)
    assert snapshot.fused_slots.shape == (2, 4, 16)
    assert snapshot.sentence_delta is not None
    assert snapshot.sentence_delta.shape == (2, 4, 16)
    assert snapshot.text_attention.shape == (2, 2, 4, 4, 6)
    assert snapshot.text_attention_replay_max_abs.shape == (2,)
    assert snapshot.text_attention_replay_max_abs.max().item() <= 1e-6
    assert torch.count_nonzero(snapshot.text_attention[0, ..., 5]) == 0
    assert torch.count_nonzero(snapshot.text_attention[1, ..., 4:]) == 0
    assert torch.allclose(
        snapshot.text_attention.sum(dim=-1),
        torch.ones(2, 2, 4, 4),
        atol=1e-6,
    )

    assert len(snapshot.sentence_layers) == 2
    motion_mask = memory["sentence_motion_mask"]
    for layer in snapshot.sentence_layers:
        assert layer.state.shape == (2, 4, 4, 16)
        assert layer.part_null_mass.shape == (2, 4, 4)
        assert layer.part_candidate_mass.shape == (2, 4, 4, 3)
        assert layer.token_mass.shape == (2, 4, 4, 3, 5)
        assert torch.allclose(
            layer.part_null_mass + layer.part_candidate_mass.sum(dim=-1),
            torch.ones_like(layer.part_null_mass),
            atol=1e-6,
        )
        assert torch.allclose(
            layer.part_candidate_mass,
            layer.token_mass.sum(dim=-1),
            atol=1e-7,
        )
        invalid = (~motion_mask)[:, None, None, :, :].expand_as(layer.token_mass)
        assert torch.count_nonzero(layer.token_mass[invalid]) == 0


def test_capture_all_null_path_and_intervention_cleanup_are_exact():
    model = _tiny_v3_model()
    text, text_mask, query, memory = _inputs(all_null=True)
    before_hooks = _hook_count(model)
    with torch.inference_mode():
        baseline = model(text, query, text_mask=text_mask)
        with DiagnosticCapture(model) as capture:
            all_null = model(text, query, text_mask=text_mask, **memory)
            snapshot = capture.snapshot(cpu=True)
    assert _hook_count(model) == before_hooks
    assert torch.equal(all_null["prediction"], baseline["prediction"])
    assert snapshot.sentence_delta is not None
    assert torch.count_nonzero(snapshot.sentence_delta) == 0
    assert len(snapshot.sentence_layers) == 2
    for layer in snapshot.sentence_layers:
        assert torch.equal(
            layer.part_null_mass,
            torch.ones_like(layer.part_null_mass),
        )
        assert torch.count_nonzero(layer.part_candidate_mass) == 0
        assert torch.count_nonzero(layer.token_mass) == 0
    trajectory = all_null["trajectory"]
    assert torch.count_nonzero(trajectory.sentence_memory_gates) == 0
    assert torch.equal(
        trajectory.sentence_memory_null_mass,
        torch.ones_like(trajectory.sentence_memory_null_mass),
    )

    identity = torch.nn.Identity()
    source = torch.tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="probe failure"):
        with ModuleOutputIntervention(
            identity,
            lambda _module, _args, _kwargs, output: output + 3.0,
        ):
            assert torch.equal(identity(source), source + 3.0)
            raise RuntimeError("probe failure")
    assert not identity._forward_hooks
    assert torch.equal(identity(source), source)


def test_eight_row_capture_smoke_is_repeatable():
    model = _tiny_v3_model()
    text, text_mask, query, memory = _inputs()
    text = text.repeat_interleave(4, dim=0)
    text_mask = text_mask.repeat_interleave(4, dim=0)
    query = query.repeat_interleave(4, dim=0)
    memory = {
        name: value.repeat_interleave(4, dim=0)
        for name, value in memory.items()
    }

    snapshots = []
    predictions = []
    with torch.inference_mode():
        for _repeat in range(2):
            with DiagnosticCapture(model) as capture:
                output = model(text, query, text_mask=text_mask, **memory)
                snapshots.append(capture.snapshot(cpu=True))
                predictions.append(output["prediction"].cpu())

    assert torch.equal(predictions[0], predictions[1])
    for field in (
        "planner_slots",
        "slot_tau",
        "fusion_input",
        "fused_slots",
        "sentence_delta",
        "text_attention",
        "text_attention_replay_max_abs",
    ):
        assert torch.equal(getattr(snapshots[0], field), getattr(snapshots[1], field))
    assert len(snapshots[0].sentence_layers) == len(snapshots[1].sentence_layers)
    for first, second in zip(
        snapshots[0].sentence_layers,
        snapshots[1].sentence_layers,
    ):
        assert torch.equal(first.state, second.state)
        assert torch.equal(first.part_null_mass, second.part_null_mass)
        assert torch.equal(first.part_candidate_mass, second.part_candidate_mass)
        assert torch.equal(first.token_mass, second.token_mass)


def _tiny_resume_array_store(root, *, reopen=False):
    return diagnostic_cli.ArrayStore(
        root,
        rows=4,
        slots=2,
        hidden=3,
        text_layers=1,
        text_heads=1,
        max_text_tokens=3,
        sentence_layers=1,
        candidates=2,
        max_motion_tokens=2,
        causal_rows=2,
        directions=1,
        query_points=3,
        reopen=reopen,
    )


def test_interrupted_progress_reopens_memmaps_and_matches_uninterrupted_result(tmp_path):
    identity = {
        "checkpoint_sha256": "fixed-checkpoint",
        "bank_id": "fixed-bank",
        "settings": {"seed": 1234},
    }
    partial = tmp_path / "interrupted"
    first = diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity=identity,
        passive_total=4,
        causal_total=2,
        resume=False,
    )
    arrays = _tiny_resume_array_store(partial)
    arrays["query_dataset_index"][:2] = np.asarray([10, 11])
    arrays["planner_slots"][:2] = 1.0
    first.commit_passive(
        0,
        2,
        {
            "queries": [{"query_id": "q0"}, {"query_id": "q1"}],
            "slots": [{"query_id": "q0", "slot": 0}],
            "text_attention": [],
            "part_attention": [],
            "candidates": [],
            "checks": {"text_attention_replay_max_abs": 1e-8},
        },
        arrays,
    )

    # Simulate a new process after interruption: reopen both progress and NPY
    # files, then continue at the first uncommitted passive row.
    resumed = diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity=identity,
        passive_total=4,
        causal_total=2,
        resume=True,
    )
    reopened = _tiny_resume_array_store(partial, reopen=True)
    resumed.validate_array_checkpoints(reopened)
    assert resumed.passive_next == 2
    assert np.array_equal(reopened["query_dataset_index"][:2], [10, 11])
    reopened["query_dataset_index"][2:] = np.asarray([12, 13])
    reopened["planner_slots"][2:] = 2.0
    resumed.commit_passive(
        2,
        4,
        {
            "queries": [{"query_id": "q2"}, {"query_id": "q3"}],
            "slots": [{"query_id": "q3", "slot": 1}],
            "text_attention": [],
            "part_attention": [],
            "candidates": [],
            "checks": {"text_attention_replay_max_abs": 2e-8},
        },
        reopened,
    )
    resumed.bind_selection("fixed-selection")
    reopened["causal_dataset_index"][0] = 12
    reopened["causal_fk_response"][0] = 3.0
    resumed.commit_causal(
        0,
        1,
        {
            "rows": [{"query_id": "q2", "stage": "planner"}],
            "selection": [{"query_id": "q2"}],
        },
        reopened,
    )

    resumed_again = diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity=identity,
        passive_total=4,
        causal_total=2,
        resume=True,
    )
    reopened_again = _tiny_resume_array_store(partial, reopen=True)
    resumed_again.validate_array_checkpoints(reopened_again)
    assert resumed_again.passive_next == 4
    assert resumed_again.causal_next == 1
    reopened_again["causal_dataset_index"][1] = 13
    reopened_again["causal_fk_response"][1] = 4.0
    resumed_again.commit_causal(
        1,
        2,
        {
            "rows": [{"query_id": "q3", "stage": "fused"}],
            "selection": [{"query_id": "q3"}],
        },
        reopened_again,
    )

    passive = resumed_again.load_passive()
    causal, selection = resumed_again.load_causal()
    assert [row["query_id"] for row in passive["queries"]] == [
        "q0",
        "q1",
        "q2",
        "q3",
    ]
    assert passive["checks"]["text_attention_replay_max_abs"] == pytest.approx(
        2e-8
    )
    assert causal == [
        {"query_id": "q2", "stage": "planner"},
        {"query_id": "q3", "stage": "fused"},
    ]
    assert selection == [{"query_id": "q2"}, {"query_id": "q3"}]
    assert np.array_equal(reopened_again["query_dataset_index"], [10, 11, 12, 13])
    assert np.array_equal(reopened_again["causal_dataset_index"], [12, 13])
    assert np.all(reopened_again["planner_slots"][:2] == 1.0)
    assert np.all(reopened_again["planner_slots"][2:] == 2.0)
    assert np.all(reopened_again["causal_fk_response"][0] == 3.0)
    assert np.all(reopened_again["causal_fk_response"][1] == 4.0)


def test_resume_rejects_stale_identity(tmp_path):
    partial = tmp_path / "stale"
    diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity={"checkpoint_sha256": "first"},
        passive_total=4,
        causal_total=2,
        resume=False,
    )
    with pytest.raises(RuntimeError, match="stale partial diagnostic"):
        diagnostic_cli._ResumeWorkspace(
            partial,
            expected_identity={"checkpoint_sha256": "second"},
            passive_total=4,
            causal_total=2,
            resume=True,
        )


def test_behavior_dependency_hash_changes_when_source_changes(tmp_path, monkeypatch):
    dependency = tmp_path / "dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(diagnostic_cli, "PROJECT_ROOT", tmp_path)
    first = diagnostic_cli._source_file_hashes({"dependency.py": dependency})
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    second = diagnostic_cli._source_file_hashes({"dependency.py": dependency})
    assert first.keys() == second.keys()
    assert first["dependency.py"] != second["dependency.py"]


def test_resume_rejects_corrupted_committed_memmap_span(tmp_path):
    identity = {"checkpoint_sha256": "fixed"}
    partial = tmp_path / "corrupted-array"
    progress = diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity=identity,
        passive_total=4,
        causal_total=2,
        resume=False,
    )
    arrays = _tiny_resume_array_store(partial)
    arrays["query_dataset_index"][:2] = [10, 11]
    arrays["planner_slots"][:2] = 1.0
    progress.commit_passive(
        0,
        2,
        {
            "queries": [{"query_id": "q0"}, {"query_id": "q1"}],
            "slots": [],
            "text_attention": [],
            "part_attention": [],
            "candidates": [],
            "checks": {},
        },
        arrays,
    )
    arrays["planner_slots"][0, 0, 0] = 9.0
    arrays.flush()

    reopened_progress = diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity=identity,
        passive_total=4,
        causal_total=2,
        resume=True,
    )
    reopened_arrays = _tiny_resume_array_store(partial, reopen=True)
    with pytest.raises(RuntimeError, match="array span hash mismatch"):
        reopened_progress.validate_array_checkpoints(reopened_arrays)


def test_resume_rejects_causal_cursor_without_selection_identity(tmp_path):
    identity = {"checkpoint_sha256": "fixed"}
    partial = tmp_path / "missing-selection"
    progress = diagnostic_cli._ResumeWorkspace(
        partial,
        expected_identity=identity,
        passive_total=1,
        causal_total=1,
        resume=False,
    )
    arrays = diagnostic_cli.ArrayStore(
        partial,
        rows=1,
        slots=1,
        hidden=1,
        text_layers=1,
        text_heads=1,
        max_text_tokens=1,
        sentence_layers=1,
        candidates=1,
        max_motion_tokens=1,
        causal_rows=1,
        directions=1,
        query_points=2,
    )
    arrays["query_dataset_index"][0] = 10
    progress.commit_passive(
        0,
        1,
        {
            "queries": [{"query_id": "q0"}],
            "slots": [],
            "text_attention": [],
            "part_attention": [],
            "candidates": [],
            "checks": {},
        },
        arrays,
    )
    progress.bind_selection("selection")
    arrays["causal_dataset_index"][0] = 10
    progress.commit_causal(
        0,
        1,
        {"rows": [], "selection": [{"query_id": "q0"}]},
        arrays,
    )
    persisted = json.loads(progress.progress_path.read_text(encoding="utf-8"))
    persisted["causal"]["selection_hash"] = None
    diagnostic_cli._atomic_write_json(progress.progress_path, persisted)
    with pytest.raises(RuntimeError, match="without a selection identity"):
        diagnostic_cli._ResumeWorkspace(
            partial,
            expected_identity=identity,
            passive_total=1,
            causal_total=1,
            resume=True,
        )


def test_ready_artifact_manifest_detects_non_array_corruption(tmp_path):
    for relative in diagnostic_cli.READY_REQUIRED_FILES:
        if relative == "provenance.json":
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact:{relative}".encode())
    manifest_sha = diagnostic_cli._write_artifact_manifest(tmp_path)
    provenance = {"artifact_manifest_sha256": manifest_sha}
    assert diagnostic_cli._validate_artifact_manifest(tmp_path, provenance)

    report = tmp_path / "report.md"
    report.write_text("corrupted\n", encoding="utf-8")
    assert not diagnostic_cli._validate_artifact_manifest(tmp_path, provenance)


def test_git_head_accepts_valid_submit_host_identity(monkeypatch):
    expected = "a" * 40
    monkeypatch.setenv("SIGNTRAJ_SOURCE_GIT_HEAD", expected.upper())
    assert diagnostic_cli._git_head() == expected


def test_git_head_rejects_invalid_submit_host_identity(monkeypatch):
    monkeypatch.setenv("SIGNTRAJ_SOURCE_GIT_HEAD", "not-a-commit")
    with pytest.raises(RuntimeError, match="Unexpected git HEAD identity"):
        diagnostic_cli._git_head()


def _phase_a_prime_checkpoint_fixture(tmp_path):
    experiment = diagnostic_cli.PHASE_A_PRIME_EXPERIMENT
    run_dir = tmp_path / experiment
    checkpoint_path = run_dir / "checkpoints/best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"locked checkpoint placeholder")

    cfg = diagnostic_cli.load_config(diagnostic_cli.PHASE_A_PRIME_CONFIG)
    cfg["device"] = "cuda"
    resolved_partition = {
        "schema_name": "csl_daily_validation_text_partition",
        "schema_version": 1,
        "counts": {
            "rows": 1_077,
            "novel_unique_texts": 796,
            "development_unique_texts": 256,
            "confirmation_unique_texts": 540,
            "development_rows": 347,
            "confirmation_rows": 728,
        },
    }
    resolved_partition["partition_digest"] = diagnostic_cli._digest_json(
        resolved_partition
    )
    cfg["validation_text_partition"].update(
        {
            "partition_digest": resolved_partition["partition_digest"],
            "resolved_artifact": resolved_partition,
            "development_row_count": 347,
            "exact_seen_row_count": 2,
            "exact_seen_text_count": 1,
            "exact_seen_evaluated_during_training": False,
            "confirmation_evaluated_during_training": False,
        }
    )
    cfg["sentence_memory"]["resolved_identity"] = {
        "bank_id": "bank",
        "neighbor_tables": {"train": {"sha256": "a"}, "val": {"sha256": "b"}},
    }
    cfg["sentence_memory"]["resolved_behavior_identity"] = (
        diagnostic_cli.sentence_memory_behavior_identity(cfg)
    )
    parity = {
        "passed": True,
        "prediction_max_abs": 0.0,
        "duration_max_abs": 0.0,
    }
    cfg["sentence_memory_safety"]["v2_to_v3_text_only_parity"] = parity

    def metric_row(epoch, score):
        return {
            "epoch": epoch,
            "validation_pending": 0.0,
            "selection_feasible": 1.0,
            "selection_score": score,
            "val_text_only/composite": 1.0,
            "val_sentence_memory/composite": score,
            "val_shuffled_sentence_memory/composite": 1.1,
            "val_motion_shuffled_sentence_memory/composite": 1.2,
        }

    rows = [metric_row(1, 0.8), metric_row(2, 0.9)]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "has_feasible_checkpoint": True,
                "best_feasible_score": 0.8,
                "required": True,
                "early_stopping": {"validation_count": 2, "stopped": True},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.resolved.json").write_text(
        json.dumps(cfg, sort_keys=True), encoding="utf-8"
    )
    checkpoint = {
        "epoch": 1,
        "config": cfg,
        "metrics": rows[0],
        "selection_state": {"best_feasible_score": 0.8},
        "v2_to_v3_text_only_parity": parity,
        "sentence_memory_identity": cfg["sentence_memory"]["resolved_identity"],
        "sentence_memory_objective_identity": (
            diagnostic_cli.sentence_memory_objective_identity(cfg)
        ),
        "sentence_memory_behavior_identity": (
            diagnostic_cli.sentence_memory_behavior_identity(cfg)
        ),
        "sentence_memory_resume_identity": (
            diagnostic_cli.sentence_memory_resume_identity(cfg)
        ),
        "rng_state": {
            "world_size": 4,
            "rank_states": [{"rank": rank} for rank in range(4)],
        },
    }
    return checkpoint, checkpoint_path, run_dir


def test_phase_a_prime_profile_requires_locked_full_run_and_binds_identities(tmp_path):
    checkpoint, checkpoint_path, run_dir = _phase_a_prime_checkpoint_fixture(tmp_path)
    evidence = diagnostic_cli._validate_phase_a_prime_locked_run(
        checkpoint,
        checkpoint_path=checkpoint_path,
        config_path=diagnostic_cli.PHASE_A_PRIME_CONFIG,
        run_dir=run_dir,
    )
    assert evidence["checkpoint_epoch"] == 1
    assert evidence["complete_validation_events"] == 2
    assert evidence["terminal_epoch"] == 2
    assert evidence["objective_identity"]["mode"] == (
        "paired_correct_motion_or_full_v1"
    )

    checkpoint["sentence_memory_identity"]["neighbor_tables"]["test"] = {
        "sha256": "forbidden"
    }
    with pytest.raises(RuntimeError, match="exactly train/val"):
        diagnostic_cli._validate_phase_a_prime_locked_run(
            checkpoint,
            checkpoint_path=checkpoint_path,
            config_path=diagnostic_cli.PHASE_A_PRIME_CONFIG,
            run_dir=run_dir,
        )


def test_diagnostic_motion_shuffle_matches_canonical_epoch_salted_utility():
    batch = {
        "index": torch.tensor([91, 92]),
        "name": ["sample-a", "sample-b"],
        "motion_path": ["motion/a.npy", "motion/b.npy"],
    }
    query_ids = diagnostic_cli._query_ids(batch)
    assert query_ids == diagnostic_cli.sentence_memory_query_ids(batch)
    torch.manual_seed(91)
    memory = SentenceMemoryBatch(
        tokens=torch.randn(2, 3, 2, 4),
        token_mask=torch.ones(2, 3, 2, dtype=torch.bool),
        token_tau=torch.randn(2, 3, 2),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
        candidate_keys=torch.randn(2, 3, 5),
        scores=torch.randn(2, 3),
        durations=torch.rand(2, 3),
        duration_log_gap=torch.rand(2, 3),
        part_validity=torch.ones(2, 3, 2, 4),
        ids=torch.arange(6).view(2, 3),
        available=torch.ones(2, dtype=torch.bool),
        provenance={"mode": "on"},
    )
    observed, permutation = diagnostic_cli._motion_only_shuffle(
        memory, query_ids, epoch=3, seed=1234
    )
    expected, expected_permutation, informative = (
        motion_only_shuffle_sentence_memory_batch(
            memory, query_ids=query_ids, epoch=3, seed=1234
        )
    )
    assert informative.all()
    assert torch.equal(permutation, expected_permutation)
    assert torch.equal(observed.tokens, expected.tokens)
    assert observed.provenance == expected.provenance
    assert observed.provenance["motion_only_shuffle_epoch"] == 3


def _factorized_authorization_fixture(tmp_path, monkeypatch, *, stage="stage1"):
    profile = diagnostic_cli.DIAGNOSTIC_PROFILES[
        diagnostic_cli.FACTORIZED_PROFILE_NAMES[stage]
    ]
    cfg = diagnostic_cli.load_config(profile.config)
    checkpoint_path = tmp_path / "run" / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"factorized checkpoint")
    authorization_path = tmp_path / "authorize_confirmation.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    development_manifest = tmp_path / "manifest_development.jsonl"
    rows = [
        {"name": f"row-{index}", "text": f"novel text {index % 256}"}
        for index in range(347)
    ]
    development_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    source_head = "a" * 40
    launch_path = tmp_path / "run_launch_identity.json"
    launch = {
        "launch_identity": "launch-id",
        "source": {
            "git_head": source_head,
            "remote_ref": "origin/fixed",
            "remote_head": source_head,
            "repository_root": str(diagnostic_cli.PROJECT_ROOT.resolve()),
            "git_directory": str(
                (diagnostic_cli.PROJECT_ROOT / ".git").resolve()
            ),
            "durable_experiments_root": str(
                (diagnostic_cli.PROJECT_ROOT / "experiments").resolve()
            ),
            "frozen_text_model_root": str(
                (diagnostic_cli.PROJECT_ROOT / "deps/mt5-base").resolve()
            ),
            "standalone_shared_clone_checked": True,
            "worktree_clean_checked": True,
            "remote_ref_exact_match_checked": True,
        },
    }
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    checkpoint_cfg = copy.deepcopy(cfg)
    checkpoint_cfg.setdefault("validation_text_partition", {}).update(
        {
            "partition_digest": (
                diagnostic_cli.EXPECTED_FACTORIZED_PARTITION_DIGEST
            ),
            "resolved_artifact": {"runtime_only": True},
            "confirmation_evaluated_during_training": False,
        }
    )
    checkpoint_cfg["device"] = "cuda"
    functions = {
        "architecture": diagnostic_cli.sentence_memory_architecture_identity,
        "behavior": diagnostic_cli.sentence_memory_behavior_identity,
        "objective": diagnostic_cli.sentence_memory_objective_identity,
        "resume": diagnostic_cli.sentence_memory_resume_identity,
        "evaluation_control": (
            diagnostic_cli.sentence_memory_evaluation_control_identity
        ),
        "selection_aggregation": (
            diagnostic_cli.sentence_memory_selection_aggregation_identity
        ),
    }
    identities = {
        name: function(checkpoint_cfg) for name, function in functions.items()
    }
    # Runtime partition resolution belongs in the resume identity only; all
    # behavior identities must still match the immutable source configuration.
    for name, function in functions.items():
        if name != "resume":
            assert identities[name] == function(cfg)
    assert identities["resume"] != functions["resume"](cfg)
    corruption = diagnostic_cli.sentence_memory_validation_corruption_map_identity(
        cfg,
        partition_digest=diagnostic_cli.EXPECTED_FACTORIZED_PARTITION_DIGEST,
        bank_id=diagnostic_cli.EXPECTED_FACTORIZED_BANK_ID,
    )
    checkpoint = {
        "epoch": 2,
        "config": checkpoint_cfg,
        "sentence_memory_identity": {
            "bank_id": diagnostic_cli.EXPECTED_FACTORIZED_BANK_ID,
            "neighbor_tables": {
                split: {"sha256": sha}
                for split, sha in (
                    diagnostic_cli.EXPECTED_FACTORIZED_NEIGHBOR_SHA256.items()
                )
            },
        },
        **{
            f"sentence_memory_{name}_identity": value
            for name, value in identities.items()
        },
        "sentence_memory_validation_corruption_map_identity": corruption,
    }
    authorized_identities = {
        name: {"digest": value["digest"], "value": value}
        for name, value in identities.items()
    }
    authorized_identities["validation_corruption_map"] = {
        "digest": corruption["digest"],
        "value": corruption,
    }
    authorization = {
        "authorization_identity": "authorization-id",
        "decision_identity": "decision-id",
        "test_data_accessed": False,
        "confirmation_manifest_opened": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": diagnostic_cli._sha256(checkpoint_path),
            "epoch": checkpoint["epoch"],
            "config": {
                "path": str(profile.config.resolve()),
                "sha256": diagnostic_cli._sha256(profile.config),
            },
            "identities": authorized_identities,
            "bank_id": diagnostic_cli.EXPECTED_FACTORIZED_BANK_ID,
            "neighbor_table_sha256": (
                diagnostic_cli.EXPECTED_FACTORIZED_NEIGHBOR_SHA256
            ),
        },
        "partition": {
            "partition_digest": diagnostic_cli.EXPECTED_FACTORIZED_PARTITION_DIGEST,
            "development_manifest": str(development_manifest),
            "development_manifest_sha256": diagnostic_cli._sha256(
                development_manifest
            ),
            "development_rows": 347,
            "development_unique_texts": 256,
        },
        "run_launch_identity": {
            "path": str(launch_path),
            "sha256": diagnostic_cli._sha256(launch_path),
            "launch_identity": launch["launch_identity"],
            "source": launch["source"],
        },
        "predecessor_authorization": None,
    }
    predecessor_authorization = None
    if stage == "stage2":
        predecessor_path = tmp_path / "authorize_stage2.json"
        predecessor_path.write_text("{}\n", encoding="utf-8")
        predecessor_authorization = {
            "authorization_identity": "stage1-infeasible-authorization",
            "decision_identity": "stage1-infeasible-decision",
        }
        authorization["predecessor_authorization"] = {
            "path": str(predecessor_path),
            "sha256": diagnostic_cli._sha256(predecessor_path),
            **predecessor_authorization,
        }
    from NIAF.continuous_trajectory_field.scripts import (
        decide_factorized_memory_stage as decision_cli,
    )

    def verify_authorization(path, *, purpose, stage):
        del path
        if purpose == "confirmation" and stage == profile.factorized_stage:
            return authorization
        if (
            predecessor_authorization is not None
            and purpose == "stage2"
            and stage == "stage1"
        ):
            return predecessor_authorization
        raise AssertionError(f"unexpected authorization request: {purpose}/{stage}")

    monkeypatch.setattr(decision_cli, "verify_authorization", verify_authorization)
    monkeypatch.setenv("SIGNTRAJ_SOURCE_GIT_HEAD", source_head)
    monkeypatch.setattr(
        diagnostic_cli,
        "_factorized_source_checkout_identity",
        lambda expected_head: {
            "repository_root": str(diagnostic_cli.PROJECT_ROOT.resolve()),
            "git_directory": str(
                (diagnostic_cli.PROJECT_ROOT / ".git").resolve()
            ),
            "git_head": expected_head,
            "durable_experiments_root": str(
                (diagnostic_cli.PROJECT_ROOT / "experiments").resolve()
            ),
            "frozen_text_model_root": str(
                (diagnostic_cli.PROJECT_ROOT / "deps/mt5-base").resolve()
            ),
            "standalone_shared_clone_checked": True,
            "worktree_clean_checked": True,
        },
    )
    return profile, cfg, checkpoint, checkpoint_path, authorization_path


def test_factorized_profiles_are_development_only_and_expose_analytic_prior():
    for stage, name in diagnostic_cli.FACTORIZED_PROFILE_NAMES.items():
        profile = diagnostic_cli.DIAGNOSTIC_PROFILES[name]
        assert profile.factorized_stage == stage
        assert profile.development_only is True
        assert profile.checkpoint.name == "best.pt"
        assert profile.output.name == "locked_development_slot_diagnostics"
        assert profile.memory_conditions == (
            "correct",
            "motion_only_shuffle",
            "shuffled",
            "analytic_prior",
        )
        diagnostic_cli._validate_config(
            diagnostic_cli.load_config(profile.config), profile
        )
    assert diagnostic_cli.DIAGNOSTIC_PROFILES[
        diagnostic_cli.LEGACY_PROFILE_NAME
    ].memory_conditions == diagnostic_cli.MEMORY_CONDITIONS


def test_factorized_diagnostic_binds_exact_name_indexed_neighbor_subset():
    subset_sha256 = "b" * 64

    class Dataset:
        split = "val"

    class Provider:
        def __init__(self, *, lookup_mode="exact_name_indexed_parent_subset_v1"):
            self.identity = {
                "neighbor_tables": {
                    "val": {
                        "sha256": diagnostic_cli.EXPECTED_FACTORIZED_NEIGHBOR_SHA256[
                            "val"
                        ],
                        "query_manifest_sha256": (
                            diagnostic_cli.EXPECTED_VALIDATION_MANIFEST_SHA256
                        ),
                        "query_count": diagnostic_cli.EXPECTED_VALIDATION_ROWS,
                        "query_order_sha256": "c" * 64,
                    }
                }
            }
            self.lookup_mode = lookup_mode
            self.neighbor_table = None
            self.bind_args = None

        def set_dataset_with_name_indexed_neighbor_subset(
            self,
            dataset,
            *,
            parent_manifest_sha256,
            expected_subset_manifest_sha256,
        ):
            self.bind_args = (
                dataset,
                parent_manifest_sha256,
                expected_subset_manifest_sha256,
            )
            query_names = [
                f"dev-{index}"
                for index in range(diagnostic_cli.EXPECTED_DEVELOPMENT_ROWS)
            ]
            self.neighbor_table = {
                "lookup_mode": self.lookup_mode,
                "parent_query_manifest_sha256": parent_manifest_sha256,
                "parent_query_order_sha256": "c" * 64,
                "parent_query_count": diagnostic_cli.EXPECTED_VALIDATION_ROWS,
                "parent_query_rows": np.arange(
                    diagnostic_cli.EXPECTED_DEVELOPMENT_ROWS, dtype=np.int64
                ),
                "query_manifest_sha256": expected_subset_manifest_sha256,
                "query_order_sha256": diagnostic_cli._digest_json(query_names),
                "query_names": query_names,
                "path": "neighbors_val.npz",
            }
            return self

        def validate_query_dataset(self, dataset, *, require_neighbors=False):
            assert require_neighbors is True
            assert dataset is self.bind_args[0]
            return {
                "split": "val",
                "manifest_sha256": subset_sha256,
                "neighbor_table": "neighbors_val.npz",
            }

    dataset = Dataset()
    provider = Provider()
    evidence = diagnostic_cli._bind_factorized_development_neighbor_subset(
        provider,
        dataset,
        parent_manifest_sha256=diagnostic_cli.EXPECTED_VALIDATION_MANIFEST_SHA256,
        subset_manifest_sha256=subset_sha256,
    )
    assert provider.bind_args == (
        dataset,
        diagnostic_cli.EXPECTED_VALIDATION_MANIFEST_SHA256,
        subset_sha256,
    )
    assert evidence["lookup_mode"] == (
        "exact_name_indexed_parent_subset_v1"
    )
    assert evidence["online_retrieval_fallback"] == "forbidden"
    assert evidence["canonical_table"]["parent_query_count"] == 1077
    assert evidence["development_subset"]["query_count"] == 347

    with pytest.raises(RuntimeError, match="exact authorized development projection"):
        diagnostic_cli._bind_factorized_development_neighbor_subset(
            Provider(lookup_mode="online"),
            dataset,
            parent_manifest_sha256=(
                diagnostic_cli.EXPECTED_VALIDATION_MANIFEST_SHA256
            ),
            subset_manifest_sha256=subset_sha256,
        )


def test_factorized_authorization_binds_checkpoint_config_source_bank_and_partition(
    tmp_path, monkeypatch
):
    profile, _cfg, checkpoint, checkpoint_path, authorization_path = (
        _factorized_authorization_fixture(tmp_path, monkeypatch)
    )
    evidence = diagnostic_cli._validate_factorized_authorized_run(
        checkpoint,
        profile=profile,
        checkpoint_path=checkpoint_path,
        config_path=profile.config,
        authorization_path=authorization_path,
    )
    assert evidence["stage"] == "stage1"
    assert evidence["development_rows"] == 347
    assert evidence["development_unique_texts"] == 256
    assert evidence["checkpoint_sha256"] == diagnostic_cli._sha256(checkpoint_path)
    assert evidence["bank_id"] == diagnostic_cli.EXPECTED_FACTORIZED_BANK_ID
    assert set(evidence["neighbor_table_sha256"]) == {"train", "val"}
    assert evidence["identities"]["architecture"]["key_value_mode"] == (
        "factorized_metadata_motion_v1"
    )


def test_factorized_authorization_rejects_best_infeasible_alias(tmp_path, monkeypatch):
    profile, _cfg, checkpoint, _checkpoint_path, authorization_path = (
        _factorized_authorization_fixture(tmp_path, monkeypatch)
    )
    bad_checkpoint = tmp_path / "run" / "checkpoints" / "best_infeasible.pt"
    bad_checkpoint.write_bytes(b"factorized checkpoint")
    with pytest.raises(RuntimeError, match="best_infeasible"):
        diagnostic_cli._validate_factorized_authorized_run(
            checkpoint,
            profile=profile,
            checkpoint_path=bad_checkpoint,
            config_path=profile.config,
            authorization_path=authorization_path,
        )


def test_factorized_stage2_authorization_requires_stage1_infeasibility_chain(
    tmp_path, monkeypatch
):
    profile, _cfg, checkpoint, checkpoint_path, authorization_path = (
        _factorized_authorization_fixture(tmp_path, monkeypatch, stage="stage2")
    )
    evidence = diagnostic_cli._validate_factorized_authorized_run(
        checkpoint,
        profile=profile,
        checkpoint_path=checkpoint_path,
        config_path=profile.config,
        authorization_path=authorization_path,
    )
    assert evidence["stage"] == "stage2"


def test_diagnostic_run_model_forwards_analytic_prior_mode():
    class RecordingModel:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"prediction": torch.zeros(1)}

    model = RecordingModel()
    text = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    tau = torch.zeros(1, 4)
    diagnostic_cli._run_model(
        model,
        text,
        mask,
        tau,
        None,
        attention_mode="analytic_prior",
    )
    assert model.kwargs["sentence_memory_attention_mode"] == "analytic_prior"
