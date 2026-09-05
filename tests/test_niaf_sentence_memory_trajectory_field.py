import pytest
import torch

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_trajectory_field.models import (
    DualModeContinuousTrajectoryField,
    SentenceMemoryContinuousTrajectoryField,
    TrajectoryInstance,
    build_continuous_trajectory_field,
)


TEXT_DIM = 24
MOTION_TOKEN_DIM = 12


def _config(model_type):
    config = {
        "model": {
            "type": model_type,
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
            "temporal_slot_layers": 1,
            "temporal_slot_heads": 4,
            "context_fps": 20.0,
        },
    }
    if model_type == "sentence_memory_continuous_trajectory_field":
        config["sentence_memory"] = {
            "motion_dim": MOTION_TOKEN_DIM,
            "attention_layers": 1,
            "attention_heads": 4,
            "score_temperature": 0.1,
            "duration_weight": 0.1,
        }
    return config


def _migrated_models():
    torch.manual_seed(401)
    v2 = build_continuous_trajectory_field(
        _config("dual_mode_continuous_trajectory_field"),
        text_dim=TEXT_DIM,
    )
    torch.manual_seed(402)
    v3 = build_continuous_trajectory_field(
        _config("sentence_memory_continuous_trajectory_field"),
        text_dim=TEXT_DIM,
    )
    incompatible = v3.load_state_dict(v2.state_dict(), strict=False)
    return v2, v3, incompatible


def _text_inputs():
    batch = 2
    text = torch.randn(batch, 5, TEXT_DIM)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    text_mask[1, -1] = False
    query = torch.linspace(-1.0, 1.0, 7).repeat(batch, 1)
    return text, text_mask, query


def _sentence_inputs(candidate_mask=None, available=None):
    batch, candidates, tokens = 2, 3, 5
    if candidate_mask is None:
        candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    if available is None:
        available = torch.ones(batch, dtype=torch.bool)
    motion_mask = candidate_mask[:, :, None].expand(-1, -1, tokens).clone()
    motion_mask[0, 1, -1] = False
    tau = torch.linspace(-1.0, 1.0, tokens).view(1, 1, -1)
    return {
        "sentence_motion_tokens": torch.randn(
            batch,
            candidates,
            tokens,
            MOTION_TOKEN_DIM,
        ),
        "sentence_motion_mask": motion_mask,
        "sentence_motion_tau": tau.expand(batch, candidates, -1).clone(),
        "sentence_text_keys": torch.randn(batch, candidates, TEXT_DIM),
        "sentence_scores": torch.tensor(
            [[0.90, 0.75, 0.60], [0.82, 0.71, 0.55]],
        ),
        "sentence_durations": torch.tensor(
            [[2.0, 2.5, 3.0], [2.25, 3.0, 4.0]],
        ),
        "sentence_candidate_mask": candidate_mask,
        "sentence_part_validity": torch.ones(
            batch,
            candidates,
            tokens,
            4,
        ),
        "sentence_memory_available": available,
    }


def test_v3_migration_preserves_exact_v2_off_path_and_state_names():
    v2, v3, incompatible = _migrated_models()
    assert isinstance(v2, DualModeContinuousTrajectoryField)
    assert isinstance(v3, SentenceMemoryContinuousTrajectoryField)
    assert incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert all(
        name.startswith("hypernetwork.sentence_memory_")
        for name in incompatible.missing_keys
    )

    v2.eval()
    v3.eval()
    text, text_mask, query = _text_inputs()
    v2_output = v2(text, query, text_mask=text_mask)
    v3_output = v3(text, query, text_mask=text_mask)

    assert torch.equal(v3_output["prediction"], v2_output["prediction"])
    assert torch.equal(
        v3_output["trajectory"].duration_seconds,
        v2_output["trajectory"].duration_seconds,
    )
    trajectory = v3_output["trajectory"]
    assert trajectory.sentence_memory_available.tolist() == [False, False]
    assert torch.count_nonzero(trajectory.sentence_memory_gates) == 0
    assert torch.equal(
        trajectory.sentence_memory_null_mass,
        torch.ones_like(trajectory.sentence_memory_null_mass),
    )

    restored = TrajectoryInstance.from_tensor_dict(trajectory.detach().tensor_dict())
    assert torch.equal(
        restored.sentence_memory_available,
        trajectory.sentence_memory_available,
    )
    assert torch.equal(restored.sentence_memory_gates, trajectory.sentence_memory_gates)
    assert torch.equal(
        restored.sentence_memory_null_mass,
        trajectory.sentence_memory_null_mass,
    )


def test_sentence_memory_has_part_specific_gates_and_zero_initialized_fusion():
    _v2, model, _incompatible = _migrated_models()
    model.eval()
    text, text_mask, query = _text_inputs()
    baseline = model(text, query, text_mask=text_mask)
    memory = _sentence_inputs(available=torch.tensor([True, False]))
    memory["sentence_part_validity"][..., 1] = 0.0

    initial = model(text, query, text_mask=text_mask, **memory)
    trajectory = initial["trajectory"]
    assert trajectory.sentence_memory_gates.shape == (2, 4, 4)
    assert trajectory.sentence_memory_null_mass.shape == (2, 4)
    assert trajectory.sentence_memory_candidate_mass.shape == (2, 4, 3)
    assert torch.count_nonzero(trajectory.sentence_memory_gates[0]) > 0
    assert torch.count_nonzero(trajectory.sentence_memory_gates[0, :, 1]) == 0
    assert torch.count_nonzero(trajectory.sentence_memory_gates[1]) == 0
    assert torch.count_nonzero(trajectory.sentence_memory_candidate_mass[1]) == 0
    assert torch.equal(initial["prediction"], baseline["prediction"])

    with torch.no_grad():
        model.hypernetwork.sentence_memory_fusion.weight.normal_(0.0, 0.02)
    output = model(text, query, text_mask=text_mask, **memory)
    assert not torch.equal(output["prediction"][0], baseline["prediction"][0])
    assert torch.equal(output["prediction"][1], baseline["prediction"][1])
    # Retrieval never changes the text-only duration branch.
    assert torch.equal(
        output["trajectory"].duration_seconds,
        baseline["trajectory"].duration_seconds,
    )

    output["prediction"].square().mean().backward()
    encoder = model.hypernetwork.sentence_memory_encoder
    assert encoder.motion_projection[1].weight.grad is not None
    assert encoder.part_query_embeddings.grad is not None
    assert encoder.part_query_projections[0].weight.grad is not None
    assert model.hypernetwork.sentence_memory_gate[-1].weight.grad is not None


def test_sentence_memory_is_candidate_permutation_invariant():
    _v2, model, _incompatible = _migrated_models()
    model.eval()
    with torch.no_grad():
        model.hypernetwork.sentence_memory_fusion.weight.normal_(0.0, 0.02)
    text, text_mask, query = _text_inputs()
    memory = _sentence_inputs()
    reference = model(text, query, text_mask=text_mask, **memory)

    permutation = torch.tensor([2, 0, 1])
    permuted = dict(memory)
    for name in (
        "sentence_motion_tokens",
        "sentence_motion_mask",
        "sentence_motion_tau",
        "sentence_text_keys",
        "sentence_scores",
        "sentence_durations",
        "sentence_candidate_mask",
        "sentence_part_validity",
    ):
        permuted[name] = memory[name].index_select(1, permutation)
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
    inverse = torch.argsort(permutation)
    assert torch.allclose(
        reordered["trajectory"].sentence_memory_candidate_mass.index_select(
            2, inverse
        ),
        reference["trajectory"].sentence_memory_candidate_mass,
        atol=1e-6,
        rtol=1e-6,
    )


def test_all_masked_sentence_memory_is_an_exact_null_fallback():
    _v2, model, _incompatible = _migrated_models()
    model.eval()
    text, text_mask, query = _text_inputs()
    baseline = model(text, query, text_mask=text_mask)
    candidate_mask = torch.zeros(2, 3, dtype=torch.bool)
    memory = _sentence_inputs(
        candidate_mask=candidate_mask,
        available=torch.ones(2, dtype=torch.bool),
    )

    with torch.no_grad():
        model.hypernetwork.sentence_memory_fusion.weight.normal_(0.0, 0.02)
    output = model(text, query, text_mask=text_mask, **memory)
    trajectory = output["trajectory"]
    assert torch.equal(output["prediction"], baseline["prediction"])
    assert not trajectory.sentence_memory_available.any()
    assert torch.count_nonzero(trajectory.sentence_memory_gates) == 0
    assert torch.count_nonzero(trajectory.sentence_memory_candidate_mass) == 0
    assert torch.equal(
        trajectory.sentence_memory_null_mass,
        torch.ones_like(trajectory.sentence_memory_null_mass),
    )

    # Even after the FFN learns a bias-like response through LayerNorm, an
    # empty memory row must expose exactly zero evidence to the fusion branch.
    encoder = model.hypernetwork.sentence_memory_encoder
    with torch.no_grad():
        encoder.layers[0].state_norm.bias.fill_(0.5)
        encoder.layers[0].feed_forward[0].weight.fill_(0.01)
        encoder.layers[0].feed_forward[3].weight.fill_(0.01)
    evidence = encoder(
        text_slots=torch.randn(2, 4, encoder.hidden_dim),
        query_duration=torch.ones(2),
        motion_tokens=memory["sentence_motion_tokens"],
        motion_mask=memory["sentence_motion_mask"],
        text_keys=memory["sentence_text_keys"],
        scores=memory["sentence_scores"],
        durations=memory["sentence_durations"],
        candidate_mask=memory["sentence_candidate_mask"],
        motion_tau=memory["sentence_motion_tau"],
        part_validity=memory["sentence_part_validity"],
    )
    assert torch.count_nonzero(evidence["slots"]) == 0


def test_sentence_memory_rejects_partial_tensor_bundle():
    _v2, model, _incompatible = _migrated_models()
    text, text_mask, query = _text_inputs()
    with pytest.raises(ValueError, match="must be supplied together"):
        model(
            text,
            query,
            text_mask=text_mask,
            sentence_motion_tokens=torch.randn(2, 3, 5, MOTION_TOKEN_DIM),
        )


def test_sentence_conditioned_instance_is_query_grid_and_resolution_invariant():
    _v2, model, _incompatible = _migrated_models()
    model.eval()
    torch.nn.init.normal_(model.hypernetwork.sentence_memory_fusion.weight, std=0.02)
    text, text_mask, _query = _text_inputs()
    memory = _sentence_inputs()
    trajectory = model.encode_trajectory(
        text,
        text_mask=text_mask,
        **memory,
    )
    common = torch.tensor([[-1.0, -0.25, 0.5, 1.0]]).repeat(2, 1)
    direct = model.query_trajectory(trajectory, common)
    dense = model.query_trajectory(
        trajectory,
        torch.cat((torch.full((2, 3), -0.75), common), dim=1),
    )
    assert torch.allclose(direct, dense[:, -common.shape[1] :], atol=1e-7)


def test_v3_preserves_word_mode_and_supports_word_plus_sentence_memory():
    v2, v3, _incompatible = _migrated_models()
    v2.eval()
    v3.eval()
    text, text_mask, query = _text_inputs()
    frames = 6
    word = {
        "word_prior_context": torch.randn(2, frames, COMPACT6D_DIM),
        "word_prior_mask": torch.ones(2, frames, dtype=torch.bool),
        "word_prior_features": torch.randn(
            2,
            frames,
            len(RETRIEVAL_FEATURE_NAMES),
        ),
        "word_prior_available": torch.ones(2, dtype=torch.bool),
    }
    v2_word = v2(text, query, text_mask=text_mask, **word)
    v3_word = v3(text, query, text_mask=text_mask, **word)
    assert torch.equal(v3_word["prediction"], v2_word["prediction"])

    with torch.no_grad():
        v3.hypernetwork.sentence_memory_fusion.weight.normal_(0.0, 0.02)
    combined = v3(
        text,
        query,
        text_mask=text_mask,
        **word,
        **_sentence_inputs(),
    )
    trajectory = combined["trajectory"]
    assert trajectory.word_prior_available.all()
    assert trajectory.sentence_memory_available.all()
    assert torch.count_nonzero(trajectory.word_prior_gates) > 0
    assert torch.count_nonzero(trajectory.sentence_memory_gates) > 0
