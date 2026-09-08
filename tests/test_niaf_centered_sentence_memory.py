from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from NIAF.continuous_trajectory_field.models import (
    build_continuous_trajectory_field,
)
from NIAF.continuous_trajectory_field.models.sentence_memory_trajectory_hypernetwork import (
    ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
    CENTERED_CANDIDATE_VALUE_MODE,
    FACTORIZED_SENTENCE_KEY_VALUE_MODE,
    FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
    RAW_FACTORIZED_CANDIDATE_VALUE_MODE,
    SentenceMemorySlotEncoder,
)


TEXT_DIM = 12
MOTION_DIM = 6
HIDDEN_DIM = 8
SLOTS = 3
PARTS = 4


def _encoder(
    *,
    association_mode="none",
    relevance_gate_mode="none",
    candidate_value_mode=CENTERED_CANDIDATE_VALUE_MODE,
    temporal_prior_mode="none",
    layers=2,
):
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
        candidate_value_mode=candidate_value_mode,
        temporal_prior_mode=temporal_prior_mode,
        association_mode=association_mode,
        association_temperature=0.10,
        relevance_gate_mode=relevance_gate_mode,
        relevance_slope=(3.0 if relevance_gate_mode != "none" else None),
        relevance_intercept=(-1.0 if relevance_gate_mode != "none" else None),
    ).eval()


def _inputs(*, batch=2, candidates=3, tokens=4):
    generator = torch.Generator().manual_seed(8107)
    motion_mask = torch.ones(batch, candidates, tokens, dtype=torch.bool)
    candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    part_validity = torch.ones(batch, candidates, tokens, PARTS)
    ids = torch.arange(100, 100 + batch * candidates, dtype=torch.long).reshape(
        batch,
        candidates,
    )
    return {
        "text_slots": torch.randn(
            batch,
            SLOTS,
            HIDDEN_DIM,
            generator=generator,
        ),
        "query_duration": torch.linspace(2.0, 3.0, batch),
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
        "scores": torch.linspace(0.91, 0.43, candidates).expand(batch, -1).clone(),
        "durations": torch.linspace(1.8, 3.7, candidates).expand(batch, -1).clone(),
        "candidate_mask": candidate_mask,
        "motion_tau": torch.linspace(-1.0, 1.0, tokens).view(
            1,
            1,
            tokens,
        ).expand(batch, candidates, -1).clone(),
        "part_validity": part_validity,
        "slot_tau": torch.linspace(-1.0, 1.0, SLOTS),
        "candidate_ids": ids,
    }


def _randomize(module, seed=141):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.7, 0.7, generator=generator)


def test_uniform_final_candidate_mass_is_exact_zero_after_randomization():
    encoder = _encoder(
        association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
        relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
    )
    _randomize(encoder)
    output = encoder(
        **_inputs(),
        attention_mode="uniform_final_candidate_mass",
    )

    assert torch.count_nonzero(output["slots"]) == 0
    debug = encoder._last_debug
    assert torch.count_nonzero(debug["sentence_memory_centered_update"]) == 0
    expected_null = 1.0 - debug[
        "sentence_memory_final_part_candidate_mass"
    ].sum(dim=-1)
    assert torch.equal(
        debug["sentence_memory_final_part_null_mass"],
        expected_null,
    )


def test_exactly_uniform_learned_candidate_mass_is_exact_zero_with_seven_candidates():
    encoder = _encoder(layers=1)
    _randomize(encoder)
    inputs = _inputs(candidates=7)
    inputs["text_keys"] = inputs["text_keys"][:, :1].expand(
        -1,
        7,
        -1,
    ).clone()
    inputs["scores"] = torch.full_like(inputs["scores"], 0.75)
    inputs["durations"] = inputs["query_duration"][:, None].expand(-1, 7).clone()
    output = encoder(**inputs)

    final_mass = encoder._last_debug[
        "sentence_memory_final_part_candidate_mass"
    ]
    assert torch.equal(final_mass, final_mass[..., :1].expand_as(final_mass))
    assert torch.count_nonzero(output["slots"]) == 0
    assert torch.count_nonzero(
        encoder._last_debug["sentence_memory_centered_update"]
    ) == 0


def test_broadcast_motion_is_exact_zero_for_arbitrary_candidate_weights():
    encoder = _encoder(
        association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
        relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
    )
    _randomize(encoder)
    inputs = _inputs()
    inputs["motion_tokens"] = inputs["motion_tokens"][:, :1].expand(
        -1,
        inputs["motion_tokens"].shape[1],
        -1,
        -1,
    ).clone()
    output = encoder(**inputs)

    assert torch.count_nonzero(output["slots"]) == 0
    assert torch.count_nonzero(
        encoder._last_debug["sentence_memory_centered_update"]
    ) == 0


def test_zero_and_one_candidate_parts_are_exact_all_null():
    encoder = _encoder(layers=1)
    inputs = _inputs(batch=1)
    validity = inputs["part_validity"]
    validity[..., 1] = 0.0
    validity[:, 0, :, 1] = 1.0
    validity[..., 2] = 0.0
    output = encoder(**inputs)
    debug = encoder._last_debug

    for part in (1, 2):
        assert torch.count_nonzero(output["slots"][:, :, part]) == 0
        assert torch.equal(
            debug["sentence_memory_final_part_null_mass"][:, :, part],
            torch.ones_like(
                debug["sentence_memory_final_part_null_mass"][:, :, part]
            ),
        )
        assert torch.count_nonzero(
            debug["sentence_memory_final_part_candidate_mass"][:, :, part]
        ) == 0
        assert torch.count_nonzero(output["part_validity"][:, :, part]) == 0


def test_smallest_stable_id_centering_is_candidate_permutation_equivariant():
    encoder = _encoder(
        association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
        relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
    )
    inputs = _inputs()
    inputs["candidate_ids"] = torch.tensor([[91, 4, 37], [52, 7, 31]])
    reference = encoder(**inputs)
    reference_debug = {
        name: value.detach().clone()
        for name, value in encoder._last_debug.items()
    }

    permutation = torch.tensor([2, 0, 1])
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
        "candidate_ids",
    ):
        permuted[name] = inputs[name].index_select(1, permutation)
    actual = encoder(**permuted)
    inverse = torch.argsort(permutation)

    assert torch.allclose(actual["slots"], reference["slots"], atol=2e-7, rtol=1e-6)
    actual_mass = encoder._last_debug[
        "sentence_memory_final_part_candidate_mass"
    ].index_select(-1, inverse)
    assert torch.allclose(
        actual_mass,
        reference_debug["sentence_memory_final_part_candidate_mass"],
        atol=1e-7,
        rtol=1e-6,
    )


def test_centered_layer_hook_masses_retain_physical_candidate_order():
    encoder = _encoder(
        association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
        relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
        layers=1,
    )
    inputs = _inputs(batch=1)
    inputs["candidate_ids"] = torch.tensor([[91, 4, 37]])
    inputs["motion_mask"] = torch.tensor(
        [[[True, True, True, True], [True, True, False, False], [True, True, True, False]]]
    )
    inputs["motion_tau"] = torch.tensor(
        [[[-0.9, -0.6, -0.3, 0.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]
    )
    inputs["part_validity"].zero_()
    inputs["part_validity"][:, :, :, 0] = 1.0
    inputs["part_validity"][:, :2, :, 1] = 1.0
    inputs["part_validity"][:, [0, 2], :, 2] = 1.0
    inputs["part_validity"][:, 1:, :, 3] = 1.0
    captured = []

    def capture(_module, _args, _kwargs, output):
        captured.append(tuple(value.detach().clone() for value in output))

    handle = encoder.layers[0].register_forward_hook(capture, with_kwargs=True)
    try:
        encoder(**inputs)
    finally:
        handle.remove()

    assert len(captured) == 1
    candidate_mass = captured[0][2].reshape(1, SLOTS, PARTS, 3)
    token_mass = captured[0][3].reshape(1, SLOTS, PARTS, 3, 4)
    physical_mass = encoder._last_debug[
        "sentence_memory_final_part_candidate_mass"
    ]
    assert torch.equal(candidate_mass, physical_mass)
    assert torch.allclose(
        token_mass.sum(dim=-1),
        physical_mass,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.unique(candidate_mass[0, 0, 0]).numel() > 1

    physical_token_support = (
        inputs["motion_mask"][..., None] & (inputs["part_validity"] > 0.0)
    ).permute(0, 3, 1, 2)
    physical_token_support = physical_token_support[:, None].expand(
        -1,
        SLOTS,
        -1,
        -1,
        -1,
    )
    physical_candidate_support = physical_token_support.any(dim=-1)
    assert torch.equal(token_mass > 0.0, physical_token_support)
    assert torch.equal(candidate_mass > 0.0, physical_candidate_support)
    assert torch.equal(
        encoder._last_debug["sentence_memory_candidate_support"],
        physical_candidate_support,
    )

    structural_uniform = physical_token_support.to(token_mass.dtype)
    structural_uniform = structural_uniform / structural_uniform.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)
    assert torch.allclose(
        token_mass,
        candidate_mass[..., None] * structural_uniform,
        atol=1e-7,
        rtol=1e-6,
    )
    observed_tau = (
        token_mass * inputs["motion_tau"][:, None, None]
    ).sum(dim=-1) / candidate_mass.clamp_min(1e-12)
    expected_tau = (
        structural_uniform * inputs["motion_tau"][:, None, None]
    ).sum(dim=-1)
    assert torch.allclose(
        observed_tau[physical_candidate_support],
        expected_tau[physical_candidate_support],
        atol=1e-7,
        rtol=1e-6,
    )


def test_centered_memory_is_token_permutation_and_masked_padding_equivariant():
    # Fix initialization so the float32 reduction-order tolerance is itself
    # deterministic.  Permuting tokens changes only the order of an otherwise
    # identical sum, which can differ by a few ulps.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        encoder = _encoder(
            association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
            relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
        )
    inputs = _inputs(batch=1, candidates=3, tokens=4)
    reference = encoder(**inputs)
    reference_debug = {
        name: value.detach().clone() for name, value in encoder._last_debug.items()
    }

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(inputs)
    for name in ("motion_tokens", "motion_mask", "motion_tau", "part_validity"):
        permuted[name] = inputs[name].index_select(2, permutation)
    actual = encoder(**permuted)
    assert torch.allclose(actual["slots"], reference["slots"], atol=5e-7, rtol=1e-6)
    assert torch.allclose(
        encoder._last_debug["sentence_memory_final_part_candidate_mass"],
        reference_debug["sentence_memory_final_part_candidate_mass"],
        atol=1e-7,
        rtol=1e-6,
    )

    padded = dict(inputs)
    generator = torch.Generator().manual_seed(91)
    padded["motion_tokens"] = torch.cat(
        (
            inputs["motion_tokens"],
            torch.randn(1, 3, 2, MOTION_DIM, generator=generator),
        ),
        dim=2,
    )
    padded["motion_mask"] = F.pad(inputs["motion_mask"], (0, 2), value=False)
    padded["motion_tau"] = F.pad(inputs["motion_tau"], (0, 2), value=0.73)
    padded["part_validity"] = F.pad(
        inputs["part_validity"], (0, 0, 0, 2), value=1.0
    )
    actual_padded = encoder(**padded)
    assert torch.allclose(
        actual_padded["slots"], reference["slots"], atol=5e-7, rtol=1e-6
    )


def test_centered_mode_fails_closed_without_ids_or_with_temporal_prior():
    encoder = _encoder()
    inputs = _inputs()
    inputs.pop("candidate_ids")
    with pytest.raises(ValueError, match="sentence_candidate_ids are required"):
        encoder(**inputs)

    with pytest.raises(ValueError, match="requires temporal_prior_mode='none'"):
        _encoder(temporal_prior_mode="gaussian")


def test_stage_a_registers_no_association_parameters_or_state():
    encoder = _encoder(
        association_mode="none",
        relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
    )
    assert not any(
        "association" in name for name, _parameter in encoder.named_parameters()
    )
    assert not any("association" in name for name in encoder.state_dict())


def test_raw_factorized_default_retains_exact_topology_and_output():
    kwargs = dict(
        motion_dim=MOTION_DIM,
        key_dim=TEXT_DIM,
        hidden_dim=HIDDEN_DIM,
        layer_count=2,
        head_count=2,
        dropout=0.0,
        key_value_mode=FACTORIZED_SENTENCE_KEY_VALUE_MODE,
        temporal_prior_mode="none",
    )
    torch.manual_seed(402)
    implicit = SentenceMemorySlotEncoder(**kwargs).eval()
    torch.manual_seed(402)
    explicit = SentenceMemorySlotEncoder(
        **kwargs,
        candidate_value_mode=RAW_FACTORIZED_CANDIDATE_VALUE_MODE,
    ).eval()
    assert tuple(implicit.state_dict()) == tuple(explicit.state_dict())
    assert all(
        torch.equal(implicit.state_dict()[name], explicit.state_dict()[name])
        for name in implicit.state_dict()
    )
    inputs = _inputs()
    inputs.pop("candidate_ids")
    implicit_output = implicit(**inputs)
    explicit_output = explicit(**inputs)
    for name in implicit_output:
        assert torch.equal(implicit_output[name], explicit_output[name])


def test_association_descriptors_are_normalized_and_source_separated():
    encoder = _encoder(
        association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
    )
    inputs = _inputs()
    encoder(**inputs)
    original = {
        name: value.detach().clone()
        for name, value in encoder._last_debug.items()
    }
    key_name = "sentence_memory_association_key_descriptor"
    motion_name = "sentence_memory_association_motion_descriptor"
    mask = original["sentence_memory_association_mask"]
    assert original[key_name].shape[-1] == 128
    assert original[motion_name].shape[-1] == 128
    assert torch.allclose(
        original[key_name].norm(dim=-1)[mask],
        torch.ones_like(original[key_name].norm(dim=-1)[mask]),
        atol=1e-6,
    )
    assert torch.allclose(
        original[motion_name].norm(dim=-1)[mask],
        torch.ones_like(original[motion_name].norm(dim=-1)[mask]),
        atol=1e-6,
    )
    expected_key = F.normalize(
        encoder.association_key_projection(inputs["text_keys"]),
        dim=-1,
        eps=1e-8,
    )
    assert torch.equal(original[key_name], expected_key)
    normalized_motion = encoder.association_motion_norm(inputs["motion_tokens"])
    token_weight = inputs["motion_mask"].to(normalized_motion.dtype)
    mean_motion = (
        normalized_motion * token_weight[..., None]
    ).sum(dim=-2) / token_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    expected_motion = F.normalize(
        encoder.association_motion_projection(mean_motion),
        dim=-1,
        eps=1e-8,
    )
    assert torch.equal(original[motion_name], expected_motion)

    target_changed = dict(inputs)
    target_changed["text_slots"] = torch.randn_like(inputs["text_slots"])
    target_changed["query_duration"] = inputs["query_duration"] * 4.0
    target_changed["scores"] = inputs["scores"].flip(-1)
    target_changed["durations"] = inputs["durations"].flip(-1)
    encoder(**target_changed)
    assert torch.equal(encoder._last_debug[key_name], original[key_name])
    assert torch.equal(encoder._last_debug[motion_name], original[motion_name])

    structural_changed = dict(inputs)
    structural_changed["motion_tau"] = torch.randn_like(inputs["motion_tau"])
    structural_changed["part_validity"] = torch.rand_like(inputs["part_validity"])
    structural_changed["candidate_ids"] = torch.tensor(
        [[900, 3, 71], [82, 700, 6]]
    )
    encoder(**structural_changed)
    assert torch.equal(encoder._last_debug[key_name], original[key_name])
    assert torch.equal(encoder._last_debug[motion_name], original[motion_name])

    keys_changed = dict(inputs)
    keys_changed["text_keys"] = inputs["text_keys"].flip(-1)
    encoder(**keys_changed)
    assert not torch.equal(encoder._last_debug[key_name], original[key_name])
    assert torch.equal(encoder._last_debug[motion_name], original[motion_name])

    motion_changed = dict(inputs)
    motion_changed["motion_tokens"] = inputs["motion_tokens"].flip(-1)
    encoder(**motion_changed)
    assert torch.equal(encoder._last_debug[key_name], original[key_name])
    assert not torch.equal(encoder._last_debug[motion_name], original[motion_name])


def test_association_logit_threshold_and_disabled_control_are_exact():
    encoder = _encoder(
        association_mode=ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
        relevance_gate_mode=FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
        layers=1,
    )
    with torch.no_grad():
        encoder.association_threshold_raw.fill_(100.0)
    inputs = _inputs()
    encoder(**inputs)
    enabled = {
        name: value.detach().clone()
        for name, value in encoder._last_debug.items()
    }
    threshold = enabled["sentence_memory_association_threshold"]
    assert -1.0 <= float(threshold) <= 1.0
    expected = (
        enabled["sentence_memory_association_cosine"] - threshold
    ) / 0.10
    mask = enabled["sentence_memory_association_mask"]
    assert torch.equal(
        enabled["sentence_memory_association_logit"][mask],
        expected[mask],
    )

    encoder(**inputs, attention_mode="association_disabled")
    disabled = encoder._last_debug
    assert torch.equal(
        disabled["sentence_memory_relevance_gate"],
        enabled["sentence_memory_relevance_gate"],
    )
    assert torch.equal(
        disabled["sentence_memory_association_gate"],
        enabled["sentence_memory_association_gate"],
    )
    assert torch.all(
        disabled["sentence_memory_final_part_candidate_mass"].sum(dim=-1)
        >= enabled["sentence_memory_final_part_candidate_mass"].sum(dim=-1)
    )


def _field_config():
    return {
        "model": {
            "type": "sentence_memory_continuous_trajectory_field",
            "context_hidden_dim": HIDDEN_DIM,
            "context_layers": 1,
            "field_hidden_dim": HIDDEN_DIM,
            "field_depth": 1,
            "max_local_fields": 2,
            "frames_per_local_field": 16,
            "dropout": 0.0,
        },
        "conditioning": {
            "temporal_slot_count": SLOTS,
            "temporal_slot_layers": 1,
            "temporal_slot_heads": 2,
        },
        "sentence_memory": {
            "motion_dim": MOTION_DIM,
            "key_dim": TEXT_DIM,
            "attention_layers": 2,
            "attention_heads": 2,
            "key_value_mode": FACTORIZED_SENTENCE_KEY_VALUE_MODE,
            "candidate_value_mode": CENTERED_CANDIDATE_VALUE_MODE,
            "association_mode": ABSOLUTE_TEXT_MOTION_ASSOCIATION_MODE,
            "relevance_gate_mode": FROZEN_ABSOLUTE_RELEVANCE_GATE_MODE,
            "relevance_slope": 3.0,
            "relevance_intercept": -1.0,
            "temporal_prior_mode": "none",
        },
        "objective": {"association_temperature": 0.10},
    }


def test_full_field_forwards_candidate_ids_and_exposes_flat_diagnostics():
    model = build_continuous_trajectory_field(
        _field_config(),
        text_dim=TEXT_DIM,
    ).eval()
    inputs = _inputs()
    batch = inputs["text_slots"].shape[0]
    text = torch.randn(batch, 5, TEXT_DIM)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    query = torch.linspace(-1.0, 1.0, 5).expand(batch, -1)
    memory = {
        "sentence_motion_tokens": inputs["motion_tokens"],
        "sentence_motion_mask": inputs["motion_mask"],
        "sentence_motion_tau": inputs["motion_tau"],
        "sentence_text_keys": inputs["text_keys"],
        "sentence_scores": inputs["scores"],
        "sentence_durations": inputs["durations"],
        "sentence_candidate_mask": inputs["candidate_mask"],
        "sentence_candidate_ids": inputs["candidate_ids"],
        "sentence_part_validity": inputs["part_validity"],
        "sentence_memory_available": torch.ones(batch, dtype=torch.bool),
    }
    output = model(text, query, text_mask=text_mask, **memory)

    expected = {
        "sentence_memory_association_key_descriptor": (batch, 3, 128),
        "sentence_memory_association_motion_descriptor": (batch, 3, 128),
        "sentence_memory_final_part_candidate_mass": (batch, SLOTS, PARTS, 3),
        "sentence_memory_final_part_null_mass": (batch, SLOTS, PARTS),
        "sentence_memory_centered_update": (batch, 2, SLOTS, PARTS, HIDDEN_DIM),
    }
    for name, shape in expected.items():
        assert output[name].shape == shape
    assert output["sentence_memory_association_threshold"].ndim == 0
    assert model.hypernetwork.last_sentence_memory_debug[
        "sentence_memory_centered_update"
    ] is output["sentence_memory_centered_update"]


def test_full_field_uniform_control_is_exact_memory_off_after_randomization():
    model = build_continuous_trajectory_field(
        _field_config(),
        text_dim=TEXT_DIM,
    ).eval()
    generator = torch.Generator().manual_seed(906)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "sentence_memory_" in name:
                parameter.uniform_(-0.6, 0.6, generator=generator)
    inputs = _inputs()
    batch = inputs["text_slots"].shape[0]
    text = torch.randn(batch, 5, TEXT_DIM, generator=generator)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    query = torch.linspace(-1.0, 1.0, 5).expand(batch, -1)
    memory = {
        "sentence_motion_tokens": inputs["motion_tokens"],
        "sentence_motion_mask": inputs["motion_mask"],
        "sentence_motion_tau": inputs["motion_tau"],
        "sentence_text_keys": inputs["text_keys"],
        "sentence_scores": inputs["scores"],
        "sentence_durations": inputs["durations"],
        "sentence_candidate_mask": inputs["candidate_mask"],
        "sentence_candidate_ids": inputs["candidate_ids"],
        "sentence_part_validity": inputs["part_validity"],
        "sentence_memory_available": torch.ones(batch, dtype=torch.bool),
    }
    with torch.inference_mode():
        baseline = model(text, query, text_mask=text_mask)
        controlled = model(
            text,
            query,
            text_mask=text_mask,
            sentence_memory_attention_mode="uniform_final_candidate_mass",
            **memory,
        )
    assert torch.equal(controlled["prediction"], baseline["prediction"])
    assert torch.equal(
        controlled["trajectory"].duration_seconds,
        baseline["trajectory"].duration_seconds,
    )
    assert torch.count_nonzero(controlled["sentence_memory_centered_update"]) == 0


def test_full_field_zero_motion_is_exact_memory_off_after_randomization():
    model = build_continuous_trajectory_field(
        _field_config(), text_dim=TEXT_DIM
    ).eval()
    generator = torch.Generator().manual_seed(1207)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "sentence_memory_" in name:
                parameter.uniform_(-0.8, 0.8, generator=generator)
    inputs = _inputs(batch=2, candidates=3, tokens=4)
    batch = inputs["text_slots"].shape[0]
    text = torch.randn(batch, 5, TEXT_DIM, generator=generator)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    query = torch.linspace(-1.0, 1.0, 5).expand(batch, -1)
    memory = {
        "sentence_motion_tokens": torch.zeros_like(inputs["motion_tokens"]),
        "sentence_motion_mask": inputs["motion_mask"],
        "sentence_motion_tau": inputs["motion_tau"],
        "sentence_text_keys": torch.randn(
            inputs["text_keys"].shape, generator=generator
        ),
        "sentence_scores": torch.tensor([[0.99, -0.3, 0.2], [0.1, 0.8, -0.7]]),
        "sentence_durations": torch.tensor([[0.9, 7.0, 2.1], [8.0, 1.1, 4.2]]),
        "sentence_candidate_mask": inputs["candidate_mask"],
        "sentence_candidate_ids": inputs["candidate_ids"],
        "sentence_part_validity": inputs["part_validity"],
        "sentence_memory_available": torch.ones(batch, dtype=torch.bool),
    }
    with torch.inference_mode():
        baseline = model(text, query, text_mask=text_mask)
        actual = model(text, query, text_mask=text_mask, **memory)
    assert torch.equal(actual["prediction"], baseline["prediction"])
    assert torch.equal(
        actual["trajectory"].duration_seconds,
        baseline["trajectory"].duration_seconds,
    )
    assert torch.count_nonzero(actual["sentence_memory_centered_update"]) == 0
    for name in (
        "prior_scale",
        "prior_shift",
        "prior_output_bias",
        "residual_scale",
        "residual_shift",
        "residual_output_bias",
        "local_scale",
        "local_shift",
        "local_output_bias",
    ):
        assert torch.equal(
            getattr(actual["trajectory"], name),
            getattr(baseline["trajectory"], name),
        )


def test_full_field_joint_tuple_permutation_is_within_one_e_minus_seven():
    model = build_continuous_trajectory_field(
        _field_config(),
        text_dim=TEXT_DIM,
    ).eval()
    generator = torch.Generator().manual_seed(1906)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "sentence_memory_" in name:
                parameter.uniform_(-0.6, 0.6, generator=generator)
    inputs = _inputs(candidates=7)
    batch = inputs["text_slots"].shape[0]
    text = torch.randn(batch, 5, TEXT_DIM, generator=generator)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    query = torch.linspace(-1.0, 1.0, 5).expand(batch, -1)
    memory = {
        "sentence_motion_tokens": inputs["motion_tokens"],
        "sentence_motion_mask": inputs["motion_mask"],
        "sentence_motion_tau": inputs["motion_tau"],
        "sentence_text_keys": inputs["text_keys"],
        "sentence_scores": inputs["scores"],
        "sentence_durations": inputs["durations"],
        "sentence_candidate_mask": inputs["candidate_mask"],
        "sentence_candidate_ids": inputs["candidate_ids"],
        "sentence_part_validity": inputs["part_validity"],
        "sentence_memory_available": torch.ones(batch, dtype=torch.bool),
    }
    permutation = torch.tensor([5, 0, 6, 2, 4, 1, 3])
    permuted = dict(memory)
    for name in (
        "sentence_motion_tokens",
        "sentence_motion_mask",
        "sentence_motion_tau",
        "sentence_text_keys",
        "sentence_scores",
        "sentence_durations",
        "sentence_candidate_mask",
        "sentence_candidate_ids",
        "sentence_part_validity",
    ):
        permuted[name] = memory[name].index_select(1, permutation)

    with torch.inference_mode():
        reference = model(text, query, text_mask=text_mask, **memory)
        actual = model(text, query, text_mask=text_mask, **permuted)
    assert torch.allclose(
        actual["prediction"],
        reference["prediction"],
        atol=1e-7,
        rtol=0.0,
    )
    inverse = torch.argsort(permutation)
    assert torch.allclose(
        actual["sentence_memory_final_part_candidate_mass"].index_select(
            -1,
            inverse,
        ),
        reference["sentence_memory_final_part_candidate_mass"],
        atol=1e-7,
        rtol=0.0,
    )
