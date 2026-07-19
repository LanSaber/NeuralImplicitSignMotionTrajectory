import torch

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import rotation_6d_to_matrix
from NIAF.retrieval_confidence_field import RetrievalUncertaintySegmentalField
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import ARTICULATOR_NAMES
from NIAF.retrieval_confidence_field.models.segmental import (
    boundary_temporal_matching_loss,
    cubic_interpolate_segment_codes,
    segment_boundary_mask,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    build_retrieval_adaptive_model,
)


def _model():
    torch.manual_seed(71)
    return RetrievalUncertaintySegmentalField(
        text_dim=24,
        code_dim=8,
        context_hidden_dim=32,
        context_layers=1,
        context_heads=4,
        articulator_code_strides={
            "body": (4, 8, 16),
            "lhand": (2, 4, 8),
            "rhand": (2, 4, 8),
            "face": (4, 8, 16),
        },
        hidden_dim=32,
        depth=2,
        time_fourier_bands=2,
        context_time_fourier_bands=2,
        calibrator_hidden_dim=24,
        calibrator_text_dim=12,
        calibrator_time_fourier_bands=2,
        duration_hidden_dim=16,
        segment_rollout_layers=1,
        segment_window_multiplier=2.0,
        minimum_segment_frames=4,
        maximum_segment_frames=16,
        segment_text_window_radius=0.25,
        segment_boundary_stride=8,
        dropout=0.0,
    )


def _inputs():
    batch, frames = 2, 17
    lengths = torch.tensor([17, 11])
    mask = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    compact = torch.cat([identity_6d.repeat(41), torch.zeros(10)])
    scaffold = compact.view(1, 1, -1).expand(batch, frames, -1).clone()
    scaffold = scaffold * mask.unsqueeze(-1)
    tau = torch.zeros(batch, frames, 1)
    for index, length in enumerate(lengths.tolist()):
        tau[index, :length, 0] = torch.linspace(0.0, 1.0, length)
    text = torch.randn(batch, 6, 24)
    text_mask = torch.ones(batch, 6, dtype=torch.bool)
    text_mask[1, -2:] = False
    evidence = torch.rand(batch, frames, len(RETRIEVAL_FEATURE_NAMES))
    evidence = evidence * mask.unsqueeze(-1)
    return scaffold, mask, lengths, tau, text, text_mask, evidence


def _predict_codes(model, scaffold, mask, tau, text, text_mask, evidence):
    density = torch.ones(*mask.shape, len(ARTICULATOR_NAMES))
    return model.code_predictor(
        scaffold,
        mask,
        tau,
        evidence,
        text,
        text_mask,
        density,
    )


def test_segmental_forward_contract_and_valid_rotations():
    model = _model()
    scaffold, mask, lengths, tau, text, text_mask, evidence = _inputs()
    outputs = model(tau, scaffold, mask, text, evidence, text_mask=text_mask)

    assert outputs["prediction"].shape == scaffold.shape
    assert outputs["frame_codes"].shape == (2, 17, 4, 8)
    assert outputs["scale_weights"].shape == (2, 17, 4, 3)
    assert outputs["segment_boundary_mask"].shape == mask.shape
    assert len(outputs["segment_positions"]) == 3
    assert outputs["scale_codes"][0].shape == (2, 9, 4, 8)
    assert outputs["knot_counts"][0, 0, 0] == 5
    assert outputs["knot_counts"][0, 1, 0] == 9
    assert torch.equal(
        torch.nonzero(outputs["segment_boundary_mask"][0]).flatten(),
        torch.tensor([0, 8, 16]),
    )
    assert torch.equal(
        torch.nonzero(outputs["segment_boundary_mask"][1]).flatten(),
        torch.tensor([0, 8, 10]),
    )
    assert torch.all(outputs["prediction"][1, lengths[1] :] == 0)

    matrices = rotation_6d_to_matrix(
        outputs["prediction"][mask][..., :246].reshape(-1, 41, 6)
    )
    assert torch.allclose(
        torch.linalg.det(matrices),
        torch.ones(matrices.shape[:2]),
        atol=1e-4,
    )


def test_future_scaffold_does_not_change_earlier_segment_codes():
    model = _model().eval()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    original = _predict_codes(model, scaffold, mask, tau, text, text_mask, evidence)
    changed = scaffold.clone()
    changed[:, 12:, :60] += 0.5
    perturbed = _predict_codes(model, changed, mask, tau, text, text_mask, evidence)

    # Fine left-hand segments 0 and 1 cover frames [0, 4) and [2, 6).
    assert torch.allclose(
        original["scale_codes"][0][:, :2, 1],
        perturbed["scale_codes"][0][:, :2, 1],
        atol=1e-7,
    )


def test_earlier_context_reaches_later_codes_through_rollout_state():
    model = _model().eval()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    original = _predict_codes(model, scaffold, mask, tau, text, text_mask, evidence)
    changed = scaffold.clone()
    changed[:, :4, :60] += 0.5
    perturbed = _predict_codes(model, changed, mask, tau, text, text_mask, evidence)

    # Segment 6 begins at frame 12, outside the altered local receptive field.
    difference = (
        (original["scale_codes"][0][0, 6, 1] - perturbed["scale_codes"][0][0, 6, 1])
        .abs()
        .max()
    )
    assert difference > 1e-7


def test_cubic_code_interpolation_is_continuous_at_internal_knot():
    codes = torch.tensor([[[0.0], [1.0], [0.25], [-0.5]]], dtype=torch.float64)
    code_mask = torch.ones(1, 4, dtype=torch.bool)
    knot = 1.0 / 3.0
    epsilon = 1e-5
    coordinate = torch.tensor(
        [[knot - epsilon, knot, knot + epsilon]], dtype=torch.float64
    )
    values = cubic_interpolate_segment_codes(codes, code_mask, coordinate)[0, :, 0]
    left_derivative = (values[1] - values[0]) / epsilon
    right_derivative = (values[2] - values[1]) / epsilon
    assert torch.isfinite(values).all()
    assert torch.allclose(left_derivative, right_derivative, atol=2e-3)


def test_boundary_losses_select_only_segment_transitions():
    mask = torch.ones(1, 17, dtype=torch.bool)
    boundaries = segment_boundary_mask(mask, stride=8)
    target = torch.zeros(1, 17, 256)
    prediction = target.clone()
    for order in (0, 1, 2):
        loss = boundary_temporal_matching_loss(
            prediction, target, mask, boundaries, order=order
        )
        assert loss == 0

    prediction[:, 8:, :60] = 1.0
    assert (
        boundary_temporal_matching_loss(prediction, target, mask, boundaries, order=0)
        > 0
    )
    assert (
        boundary_temporal_matching_loss(prediction, target, mask, boundaries, order=1)
        > 0
    )
    assert (
        boundary_temporal_matching_loss(
            prediction,
            target,
            mask,
            torch.zeros_like(boundaries),
            order=1,
        )
        == 0
    )


def test_segmental_backward_reaches_rollout_and_code_heads():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    outputs = model(tau, scaffold, mask, text, evidence, text_mask=text_mask)
    loss = (
        outputs["prediction"][mask].square().mean()
        + outputs["frame_codes"][mask].square().mean()
        + outputs["trust"][mask].mean()
    )
    loss.backward()
    assert model.code_predictor.rollout.gru.weight_ih_l0.grad is not None
    assert model.code_predictor.code_heads["lhand_0"][2].weight.grad is not None
    assert model.calibrator.trust_head.weight.grad is not None
    assert torch.isfinite(model.code_predictor.rollout.gru.weight_ih_l0.grad).all()


def test_builder_and_package_export_select_segmental_model():
    model = build_retrieval_adaptive_model(
        {
            "model": {
                "type": "retrieval_uncertainty_segmental_field",
                "code_dim": 8,
                "context_hidden_dim": 32,
                "context_heads": 4,
                "hidden_dim": 32,
                "depth": 2,
                "segment_rollout_layers": 1,
                "minimum_segment_frames": 4,
                "maximum_segment_frames": 16,
                "articulator_code_strides": {
                    "body": [4, 8, 16],
                    "lhand": [2, 4, 8],
                    "rhand": [2, 4, 8],
                    "face": [4, 8, 16],
                },
            },
            "duration": {"hidden_dim": 16},
        },
        text_dim=24,
    )
    assert isinstance(model, RetrievalUncertaintySegmentalField)
    assert model.code_strides == (4, 8, 16)
    assert model.segment_boundary_stride == 8
