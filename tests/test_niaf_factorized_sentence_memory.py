from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from NIAF.continuous_trajectory_field.models import (
    SentenceMemoryContinuousTrajectoryField,
    build_continuous_trajectory_field,
)
from NIAF.continuous_trajectory_field.models.sentence_memory_trajectory_hypernetwork import (
    FACTORIZED_SENTENCE_KEY_VALUE_MODE,
    LEGACY_SENTENCE_KEY_VALUE_MODE,
    FactorizedSentenceMemoryCrossAttention,
    SentenceMemorySlotEncoder,
)


TEXT_DIM = 12
MOTION_DIM = 6
HIDDEN_DIM = 8
SLOTS = 3
PARTS = 4


def _config(*, key_value_mode="missing", temporal_prior_mode="none"):
    cfg = {
        "model": {
            "type": "sentence_memory_continuous_trajectory_field",
            "context_hidden_dim": HIDDEN_DIM,
            "context_layers": 1,
            "field_hidden_dim": 8,
            "field_depth": 1,
            "max_local_fields": 2,
            "frames_per_local_field": 16,
            "dropout": 0.0,
        },
        "conditioning": {
            "temporal_slot_count": SLOTS,
            "temporal_slot_layers": 1,
            "temporal_slot_heads": 2,
            "context_fps": 20.0,
        },
        "sentence_memory": {
            "motion_dim": MOTION_DIM,
            "attention_layers": 2,
            "attention_heads": 2,
            "score_temperature": 0.10,
            "duration_weight": 0.05,
            "retrieval_prior_scale": 1.0,
            "temporal_prior_mode": temporal_prior_mode,
            "temporal_prior_sigma": 0.25,
            "temporal_prior_scale": 1.0,
        },
    }
    if key_value_mode != "missing":
        cfg["sentence_memory"]["key_value_mode"] = key_value_mode
    return cfg


def _encoder(*, temporal_prior_mode="none", layers=2):
    return SentenceMemorySlotEncoder(
        motion_dim=MOTION_DIM,
        key_dim=TEXT_DIM,
        hidden_dim=HIDDEN_DIM,
        layer_count=layers,
        head_count=2,
        dropout=0.0,
        score_temperature=0.10,
        duration_weight=0.05,
        retrieval_prior_scale=1.0,
        key_value_mode=FACTORIZED_SENTENCE_KEY_VALUE_MODE,
        temporal_prior_mode=temporal_prior_mode,
        temporal_prior_sigma=0.25,
        temporal_prior_scale=1.0,
    ).eval()


def _encoder_inputs(*, batch=2, candidates=3, tokens=4):
    generator = torch.Generator().manual_seed(7103)
    motion_mask = torch.ones(batch, candidates, tokens, dtype=torch.bool)
    motion_mask[0, -1, -1] = False
    candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    part_validity = torch.ones(batch, candidates, tokens, PARTS)
    part_validity[0, 0, -1, 1] = 0.0
    return {
        "text_slots": torch.randn(
            batch,
            SLOTS,
            HIDDEN_DIM,
            generator=generator,
        ),
        "query_duration": torch.tensor([2.0, 3.0])[:batch],
        "motion_tokens": torch.randn(
            batch,
            candidates,
            tokens,
            MOTION_DIM,
            generator=generator,
        ),
        "motion_mask": motion_mask,
        "text_keys": torch.randn(
            batch,
            candidates,
            TEXT_DIM,
            generator=generator,
        ),
        "scores": torch.tensor(
            [[0.91, 0.73, 0.54], [0.87, 0.68, 0.49]],
        )[:batch, :candidates],
        "durations": torch.tensor(
            [[2.1, 2.7, 3.2], [2.8, 3.4, 4.1]],
        )[:batch, :candidates],
        "candidate_mask": candidate_mask,
        "motion_tau": torch.linspace(-1.0, 1.0, tokens).view(
            1,
            1,
            tokens,
        ).expand(batch, candidates, -1).clone(),
        "part_validity": part_validity,
        "slot_tau": torch.linspace(-1.0, 1.0, SLOTS),
    }


def _field_inputs():
    generator = torch.Generator().manual_seed(811)
    batch, candidates, tokens = 2, 3, 4
    text = torch.randn(batch, 5, TEXT_DIM, generator=generator)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    query = torch.linspace(-1.0, 1.0, 5).expand(batch, -1)
    values = _encoder_inputs(batch=batch, candidates=candidates, tokens=tokens)
    memory = {
        "sentence_motion_tokens": values["motion_tokens"],
        "sentence_motion_mask": values["motion_mask"],
        "sentence_motion_tau": values["motion_tau"],
        "sentence_text_keys": values["text_keys"],
        "sentence_scores": values["scores"],
        "sentence_durations": values["durations"],
        "sentence_candidate_mask": values["candidate_mask"],
        "sentence_part_validity": values["part_validity"],
        "sentence_memory_available": torch.ones(batch, dtype=torch.bool),
    }
    return text, text_mask, query, memory


def _capture_layer_inputs(encoder, inputs, *, attention_mode="learned"):
    captured = []

    def hook(_module, args, kwargs):
        captured.append(
            tuple(value.detach().clone() if torch.is_tensor(value) else value for value in args)
            + (dict(kwargs),)
        )

    handles = [
        layer.register_forward_pre_hook(hook, with_kwargs=True)
        for layer in encoder.layers
    ]
    try:
        output = encoder(**inputs, attention_mode=attention_mode)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


def _capture_layer_outputs(encoder, inputs, *, attention_mode="analytic_prior"):
    captured = []

    def hook(_module, _args, _kwargs, output):
        captured.append(tuple(value.detach().clone() for value in output))

    handles = [
        layer.register_forward_hook(hook, with_kwargs=True)
        for layer in encoder.layers
    ]
    try:
        output = encoder(**inputs, attention_mode=attention_mode)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


def test_absent_key_value_mode_is_exact_legacy_topology_and_initialization():
    torch.manual_seed(1019)
    implicit = build_continuous_trajectory_field(
        _config(),
        text_dim=TEXT_DIM,
    )
    torch.manual_seed(1019)
    explicit = build_continuous_trajectory_field(
        _config(key_value_mode=LEGACY_SENTENCE_KEY_VALUE_MODE),
        text_dim=TEXT_DIM,
    )

    assert isinstance(implicit, SentenceMemoryContinuousTrajectoryField)
    assert (
        implicit.hypernetwork.sentence_memory_encoder.key_value_mode
        == LEGACY_SENTENCE_KEY_VALUE_MODE
    )
    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert tuple(implicit_state) == tuple(explicit_state)
    assert all(
        torch.equal(implicit_state[name], explicit_state[name])
        for name in implicit_state
    )
    assert any("time_projection" in name for name in implicit_state)
    assert any("memory_norm" in name for name in implicit_state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key_value_mode", "mixed_but_not_legacy"),
        ("temporal_prior_mode", "learned"),
    ],
)
def test_factorized_mode_rejects_unknown_architecture_controls(field, value):
    kwargs = {
        "key_value_mode": FACTORIZED_SENTENCE_KEY_VALUE_MODE,
        "temporal_prior_mode": "none",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        SentenceMemorySlotEncoder(
            motion_dim=MOTION_DIM,
            key_dim=TEXT_DIM,
            hidden_dim=HIDDEN_DIM,
            layer_count=1,
            head_count=2,
            dropout=0.0,
            **kwargs,
        )


def test_factorized_layer_inputs_strictly_separate_query_key_and_value_sources():
    # Use the Gaussian arm so changing token time/fractional part validity is
    # observable in the structural prior.  The no-bias arm is intentionally
    # uniform over positive-validity support, so those changes would be a
    # no-op there unless they removed a token from the support entirely.
    encoder = _encoder(temporal_prior_mode="gaussian")
    inputs = _encoder_inputs()
    _reference_output, reference = _capture_layer_inputs(encoder, inputs)
    assert len(reference) == 2

    motion_changed = dict(inputs)
    motion_changed["motion_tokens"] = inputs["motion_tokens"] + 7.0 * torch.randn_like(
        inputs["motion_tokens"]
    )
    _output, motion_capture = _capture_layer_inputs(encoder, motion_changed)
    for layer_reference, layer_changed in zip(reference, motion_capture):
        # immutable text/part Q, candidate K, and the structural prior
        assert torch.equal(layer_reference[0], layer_changed[0])
        assert torch.equal(layer_reference[2], layer_changed[2])
        assert torch.equal(layer_reference[4], layer_changed[4])
        assert not torch.equal(layer_reference[3], layer_changed[3])

    metadata_changed = dict(inputs)
    metadata_changed["text_keys"] = inputs["text_keys"] + 3.0
    metadata_changed["scores"] = inputs["scores"].flip(-1)
    metadata_changed["durations"] = inputs["durations"] * 1.7
    _output, metadata_capture = _capture_layer_inputs(encoder, metadata_changed)
    for layer_reference, layer_changed in zip(reference, metadata_capture):
        assert torch.equal(layer_reference[0], layer_changed[0])
        assert not torch.equal(layer_reference[2], layer_changed[2])
        assert torch.equal(layer_reference[3], layer_changed[3])

    structural_changed = dict(inputs)
    structural_changed["motion_tau"] = inputs["motion_tau"].flip(-1)
    structural_changed["part_validity"] = inputs["part_validity"].clone()
    structural_changed["part_validity"][:, :, 0, 2] *= 0.25
    _output, structural_capture = _capture_layer_inputs(encoder, structural_changed)
    for layer_reference, layer_changed in zip(reference, structural_capture):
        assert torch.equal(layer_reference[0], layer_changed[0])
        assert torch.equal(layer_reference[2], layer_changed[2])
        assert torch.equal(layer_reference[3], layer_changed[3])
        assert not torch.equal(layer_reference[4], layer_changed[4])

    # Later-layer queries are exactly the same immutable Q tensor, even though
    # their carried motion state differs.
    assert torch.equal(reference[0][0], reference[1][0])
    assert torch.count_nonzero(reference[0][1]) == 0
    assert torch.count_nonzero(reference[1][1]) > 0


def test_factorized_value_and_residual_path_is_zero_preserving_after_randomization():
    encoder = _encoder(temporal_prior_mode="gaussian")
    generator = torch.Generator().manual_seed(912)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.uniform_(-0.8, 0.8, generator=generator)
    inputs = _encoder_inputs()
    inputs["motion_tokens"] = torch.zeros_like(inputs["motion_tokens"])
    evidence, layers = _capture_layer_outputs(encoder, inputs)
    assert torch.count_nonzero(evidence["slots"]) == 0
    assert all(torch.count_nonzero(layer[0]) == 0 for layer in layers)

    assert encoder.motion_projection[0].elementwise_affine is False
    assert encoder.motion_projection[1].bias is None
    assert all(projection.bias is None for projection in encoder.part_query_projections)
    for layer in encoder.layers:
        assert isinstance(layer, FactorizedSentenceMemoryCrossAttention)
        assert layer.value_norm.elementwise_affine is False
        assert layer.value_projection.bias is None
        assert layer.output_projection.bias is None
        assert layer.state_norm.elementwise_affine is False
        assert layer.feed_forward[0].bias is None
        assert layer.feed_forward[3].bias is None


def test_factorized_full_model_zero_motion_has_exact_zero_delta_and_off_prediction():
    torch.manual_seed(5201)
    model = build_continuous_trajectory_field(
        _config(
            key_value_mode=FACTORIZED_SENTENCE_KEY_VALUE_MODE,
            temporal_prior_mode="gaussian",
        ),
        text_dim=TEXT_DIM,
    ).eval()
    generator = torch.Generator().manual_seed(5202)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "sentence_memory_" in name:
                parameter.uniform_(-0.5, 0.5, generator=generator)
    text, text_mask, query, memory = _field_inputs()
    memory["sentence_motion_tokens"] = torch.zeros_like(
        memory["sentence_motion_tokens"]
    )

    captured_delta = []
    handle = model.hypernetwork.sentence_memory_fusion.register_forward_hook(
        lambda _module, _args, output: captured_delta.append(output.detach().clone())
    )
    try:
        with torch.inference_mode():
            baseline = model(text, query, text_mask=text_mask)
            observed = model(text, query, text_mask=text_mask, **memory)
    finally:
        handle.remove()
    assert len(captured_delta) == 1
    assert torch.count_nonzero(captured_delta[0]) == 0
    assert torch.equal(observed["prediction"], baseline["prediction"])
    part_projections = model.hypernetwork.sentence_memory_part_projections
    assert all(projection.bias is None for projection in part_projections)


def test_uniform_token_prior_is_uniform_over_positive_part_validity_support():
    encoder = _encoder(layers=1)
    inputs = _encoder_inputs(batch=1, candidates=1, tokens=4)
    inputs["motion_mask"][:] = True
    inputs["part_validity"][:] = 1.0
    inputs["part_validity"][0, 0, :, 0] = torch.tensor([1.0, 0.5, 0.0, 0.25])
    inputs["part_validity"][0, 0, :, 2] = 0.0
    _evidence, layers = _capture_layer_outputs(encoder, inputs)
    _state, null_mass, _candidate_mass, token_mass = layers[0]
    token_mass = token_mass.reshape(1, SLOTS, PARTS, 1, 4)
    null_mass = null_mass.reshape(1, SLOTS, PARTS)

    conditional = token_mass[0, :, 0, 0]
    conditional = conditional / conditional.sum(dim=-1, keepdim=True)
    expected = torch.tensor([1.0, 1.0, 0.0, 1.0])
    expected = expected / expected.sum()
    assert torch.allclose(conditional, expected.expand_as(conditional), atol=1e-7)
    assert torch.count_nonzero(token_mass[0, :, 0, 0, 2]) == 0
    assert torch.equal(null_mass[0, :, 2], torch.ones(SLOTS))
    assert torch.count_nonzero(token_mass[0, :, 2]) == 0
    assert torch.count_nonzero(_evidence["slots"][0, :, 2]) == 0
    assert torch.count_nonzero(_evidence["part_validity"][0, :, 2]) == 0


def test_gaussian_token_prior_uses_fractional_part_validity():
    encoder = _encoder(temporal_prior_mode="gaussian", layers=1)
    inputs = _encoder_inputs(batch=1, candidates=1, tokens=4)
    inputs["motion_mask"][:] = True
    inputs["part_validity"][:] = 1.0
    validity = torch.tensor([1.0, 0.5, 0.0, 0.25])
    inputs["part_validity"][0, 0, :, 0] = validity
    _evidence, layers = _capture_layer_outputs(encoder, inputs)
    token_mass = layers[0][3].reshape(1, SLOTS, PARTS, 1, 4)

    for slot, slot_time in enumerate(inputs["slot_tau"]):
        gaussian = -0.5 * (
            (slot_time - inputs["motion_tau"][0, 0]) / 0.25
        ).square()
        expected_logits = gaussian + torch.log(
            validity.clamp_min(torch.finfo(validity.dtype).tiny)
        )
        expected_logits = expected_logits.masked_fill(validity == 0.0, -torch.inf)
        expected = torch.softmax(expected_logits, dim=-1)
        actual = token_mass[0, slot, 0, 0]
        actual = actual / actual.sum()
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_gaussian_token_prior_matches_predeclared_normalized_formula():
    encoder = _encoder(temporal_prior_mode="gaussian", layers=1)
    inputs = _encoder_inputs(batch=1, candidates=1, tokens=4)
    inputs["motion_mask"][:] = True
    inputs["part_validity"][:] = 1.0
    _evidence, layers = _capture_layer_outputs(encoder, inputs)
    token_mass = layers[0][3].reshape(1, SLOTS, PARTS, 1, 4)

    for slot, slot_time in enumerate(inputs["slot_tau"]):
        expected_logits = -0.5 * (
            (slot_time - inputs["motion_tau"][0, 0]) / 0.25
        ).square()
        expected = torch.softmax(expected_logits, dim=-1)
        actual = token_mass[0, slot, 0, 0]
        actual = actual / actual.sum()
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_factorized_attention_is_invariant_to_invalid_candidate_padding():
    encoder = _encoder(temporal_prior_mode="gaussian")
    inputs = _encoder_inputs(batch=2, candidates=3, tokens=4)
    reference = encoder(**inputs)

    padded = dict(inputs)
    batch, _candidates, tokens, motion_dim = inputs["motion_tokens"].shape
    padded["motion_tokens"] = torch.cat(
        [inputs["motion_tokens"], torch.randn(batch, 2, tokens, motion_dim)],
        dim=1,
    )
    padded["motion_mask"] = torch.cat(
        [
            inputs["motion_mask"],
            torch.zeros(batch, 2, tokens, dtype=torch.bool),
        ],
        dim=1,
    )
    padded["text_keys"] = torch.cat(
        [inputs["text_keys"], torch.randn(batch, 2, TEXT_DIM)],
        dim=1,
    )
    padded["scores"] = torch.cat(
        [inputs["scores"], torch.randn(batch, 2)],
        dim=1,
    )
    padded["durations"] = torch.cat(
        [inputs["durations"], torch.rand(batch, 2) + 0.5],
        dim=1,
    )
    padded["candidate_mask"] = torch.cat(
        [inputs["candidate_mask"], torch.zeros(batch, 2, dtype=torch.bool)],
        dim=1,
    )
    padded["motion_tau"] = torch.cat(
        [inputs["motion_tau"], torch.randn(batch, 2, tokens)],
        dim=1,
    )
    padded["part_validity"] = torch.cat(
        [inputs["part_validity"], torch.rand(batch, 2, tokens, PARTS)],
        dim=1,
    )
    actual = encoder(**padded)

    assert torch.allclose(actual["slots"], reference["slots"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        actual["part_null_mass"],
        reference["part_null_mass"],
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        actual["candidate_mass"][..., :3],
        reference["candidate_mass"],
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.count_nonzero(actual["candidate_mass"][..., 3:]) == 0


@pytest.mark.parametrize("temporal_prior_mode", ["none", "gaussian"])
def test_factorized_token_prior_is_length_normalized_per_candidate(
    temporal_prior_mode,
):
    encoder = _encoder(temporal_prior_mode=temporal_prior_mode, layers=1)
    inputs = _encoder_inputs(batch=1, candidates=2, tokens=4)
    inputs["scores"][:] = 0.75
    inputs["durations"][:] = inputs["query_duration"][:, None]
    inputs["candidate_mask"][:] = True
    inputs["motion_mask"][:] = True
    inputs["motion_mask"][0, 1, 1:] = False
    inputs["part_validity"][:] = 1.0
    inputs["part_validity"][0, 1, 1:] = 0.0

    _evidence, layers = _capture_layer_outputs(
        encoder,
        inputs,
        attention_mode="analytic_prior",
    )
    token_mass = layers[0][3].reshape(1, SLOTS, PARTS, 2, 4)
    candidate_mass = token_mass.sum(dim=-1)
    conditional = candidate_mass / candidate_mass.sum(dim=-1, keepdim=True)

    # A four-token candidate and a one-token candidate receive the same total
    # retrieval-prior mass; token count can only redistribute mass within a
    # candidate, never multiply that candidate's weight.
    assert torch.allclose(
        conditional,
        torch.full_like(conditional, 0.5),
        atol=1e-7,
        rtol=1e-6,
    )


def test_analytic_attention_uses_exact_retrieval_prior_formula():
    encoder = _encoder(temporal_prior_mode="none", layers=1)
    inputs = _encoder_inputs(batch=1, candidates=3, tokens=4)
    inputs["motion_mask"][:] = True
    inputs["part_validity"][:] = 1.0
    _evidence, layers = _capture_layer_outputs(
        encoder,
        inputs,
        attention_mode="analytic_prior",
    )
    candidate_mass = layers[0][2].reshape(1, SLOTS, PARTS, 3)
    conditional = candidate_mass / candidate_mass.sum(dim=-1, keepdim=True)

    duration_gap = torch.abs(
        torch.log(inputs["query_duration"][:, None] / inputs["durations"])
    )
    expected = torch.softmax(
        (inputs["scores"] - 0.05 * duration_gap) / 0.10,
        dim=-1,
    )
    assert torch.allclose(
        conditional,
        expected[:, None, None, :].expand_as(conditional),
        atol=1e-7,
        rtol=1e-6,
    )


def test_analytic_prior_bypasses_only_real_qk_and_preserves_null_value_path():
    encoder = _encoder(temporal_prior_mode="gaussian", layers=1)
    inputs = _encoder_inputs()
    learned, learned_layers = _capture_layer_outputs(
        encoder,
        inputs,
        attention_mode="learned",
    )
    analytic, analytic_layers = _capture_layer_outputs(
        encoder,
        inputs,
        attention_mode="analytic_prior",
    )
    assert not torch.equal(learned_layers[0][3], analytic_layers[0][3])

    with torch.no_grad():
        encoder.layers[0].key_projection.weight.add_(9.0)
        encoder.layers[0].key_projection.bias.sub_(4.0)
    analytic_changed, changed_layers = _capture_layer_outputs(
        encoder,
        inputs,
        attention_mode="analytic_prior",
    )
    assert torch.equal(analytic_layers[0][1], changed_layers[0][1])
    assert torch.equal(analytic_layers[0][2], changed_layers[0][2])
    assert torch.equal(analytic_layers[0][3], changed_layers[0][3])
    assert torch.equal(analytic["slots"], analytic_changed["slots"])

    # The analytic control still uses motion values and the learned null path.
    zero_motion = dict(inputs)
    zero_motion["motion_tokens"] = torch.zeros_like(inputs["motion_tokens"])
    zero_output = encoder(**zero_motion, attention_mode="analytic_prior")
    assert torch.count_nonzero(zero_output["slots"]) == 0
    with torch.no_grad():
        encoder.layers[0].null_logit_bias.add_(1.0)
    _changed, null_changed_layers = _capture_layer_outputs(
        encoder,
        inputs,
        attention_mode="analytic_prior",
    )
    assert not torch.equal(changed_layers[0][1], null_changed_layers[0][1])


def test_factorized_attention_is_joint_candidate_and_token_permutation_equivariant():
    encoder = _encoder(temporal_prior_mode="gaussian")
    inputs = _encoder_inputs()
    reference, reference_layers = _capture_layer_outputs(encoder, inputs)

    candidate_permutation = torch.tensor([2, 0, 1])
    token_permutation = torch.tensor([3, 1, 0, 2])
    permuted = dict(inputs)
    for name in (
        "motion_tokens",
        "motion_mask",
        "text_keys",
        "scores",
        "durations",
        "candidate_mask",
        "motion_tau",
        "part_validity",
    ):
        permuted[name] = inputs[name].index_select(1, candidate_permutation)
    for name in ("motion_tokens", "motion_mask", "motion_tau", "part_validity"):
        permuted[name] = permuted[name].index_select(2, token_permutation)
    actual, actual_layers = _capture_layer_outputs(encoder, permuted)

    assert torch.allclose(actual["slots"], reference["slots"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        actual["part_null_mass"],
        reference["part_null_mass"],
        atol=1e-7,
        rtol=1e-6,
    )
    candidate_inverse = torch.argsort(candidate_permutation)
    assert torch.allclose(
        actual["candidate_mass"].index_select(-1, candidate_inverse),
        reference["candidate_mass"],
        atol=1e-7,
        rtol=1e-6,
    )
    token_inverse = torch.argsort(token_permutation)
    for reference_layer, actual_layer in zip(reference_layers, actual_layers):
        reference_token_mass = reference_layer[3].reshape(
            2,
            SLOTS,
            PARTS,
            3,
            4,
        )
        actual_token_mass = actual_layer[3].reshape_as(reference_token_mass)
        actual_token_mass = actual_token_mass.index_select(-2, candidate_inverse)
        actual_token_mass = actual_token_mass.index_select(-1, token_inverse)
        assert torch.allclose(
            actual_token_mass,
            reference_token_mass,
            atol=1e-7,
            rtol=1e-6,
        )


def test_factorized_all_null_is_exact_memory_off_and_attention_api_is_strict():
    model = build_continuous_trajectory_field(
        _config(key_value_mode=FACTORIZED_SENTENCE_KEY_VALUE_MODE),
        text_dim=TEXT_DIM,
    ).eval()
    text, text_mask, query, memory = _field_inputs()
    memory["sentence_candidate_mask"] = torch.zeros_like(
        memory["sentence_candidate_mask"]
    )
    memory["sentence_motion_mask"] = torch.zeros_like(
        memory["sentence_motion_mask"]
    )
    with torch.no_grad():
        model.hypernetwork.sentence_memory_fusion.weight.normal_(0.0, 0.2)
        baseline = model(text, query, text_mask=text_mask)
        observed = model(text, query, text_mask=text_mask, **memory)
    trajectory = observed["trajectory"]
    assert torch.equal(observed["prediction"], baseline["prediction"])
    assert torch.equal(
        trajectory.sentence_memory_null_mass,
        torch.ones_like(trajectory.sentence_memory_null_mass),
    )
    assert torch.count_nonzero(trajectory.sentence_memory_gates) == 0
    assert torch.count_nonzero(trajectory.sentence_memory_candidate_mass) == 0

    with pytest.raises(ValueError, match="sentence_memory_attention_mode"):
        model(
            text,
            query,
            text_mask=text_mask,
            **memory,
            sentence_memory_attention_mode="unknown",
        )
    legacy = build_continuous_trajectory_field(_config(), text_dim=TEXT_DIM)
    valid_memory = _field_inputs()[-1]
    with pytest.raises(ValueError, match="requires"):
        legacy(
            text,
            query,
            text_mask=text_mask,
            **valid_memory,
            sentence_memory_attention_mode="analytic_prior",
        )


@pytest.mark.parametrize("temporal_prior_mode", ["none", "gaussian"])
def test_factorized_all_null_backward_has_only_finite_gradients(
    temporal_prior_mode,
):
    encoder = _encoder(temporal_prior_mode=temporal_prior_mode)
    inputs = _encoder_inputs()
    inputs["candidate_mask"] = torch.zeros_like(inputs["candidate_mask"])
    inputs["motion_mask"] = torch.zeros_like(inputs["motion_mask"])
    inputs["part_validity"] = torch.zeros_like(inputs["part_validity"])

    output = encoder(**inputs)["slots"]
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    for parameter in encoder.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_factorized_config_does_not_mutate_input_and_exposes_resolved_modes():
    cfg = _config(
        key_value_mode=FACTORIZED_SENTENCE_KEY_VALUE_MODE,
        temporal_prior_mode="gaussian",
    )
    original = deepcopy(cfg)
    model = build_continuous_trajectory_field(cfg, text_dim=TEXT_DIM)
    assert cfg == original
    encoder = model.hypernetwork.sentence_memory_encoder
    assert encoder.key_value_mode == FACTORIZED_SENTENCE_KEY_VALUE_MODE
    assert encoder.temporal_prior_mode == "gaussian"
    assert encoder.temporal_prior_sigma == pytest.approx(0.25)
    assert encoder.temporal_prior_scale == pytest.approx(1.0)
