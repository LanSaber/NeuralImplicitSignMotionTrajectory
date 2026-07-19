import torch

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import rotation_6d_to_matrix
from NIAF.continuous_sign_field.losses import (
    endpoint_losses,
    fk_temporal_dynamics_losses,
    prediction_parts_from_rot6d,
    wrist_relative_hand_l1,
)
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import (
    ARTICULATOR_NAMES,
)
from NIAF.retrieval_confidence_field.models.uncertainty_adaptive import (
    RetrievalUncertaintyAdaptiveKnotField,
    adaptive_knot_density_target,
    adaptive_time_coordinate,
    correction_need_target_from_error,
    retrieval_uncertainty_proxy,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    build_retrieval_adaptive_model,
    validation_selection_score,
)


def _model():
    torch.manual_seed(31)
    return RetrievalUncertaintyAdaptiveKnotField(
        text_dim=24,
        code_dim=8,
        context_hidden_dim=32,
        context_layers=1,
        context_heads=4,
        articulator_code_strides={
            "body": (2, 4, 8),
            "lhand": (1, 2, 4),
            "rhand": (1, 2, 4),
            "face": (2, 4, 8),
        },
        hidden_dim=32,
        depth=2,
        time_fourier_bands=2,
        context_time_fourier_bands=2,
        calibrator_hidden_dim=24,
        calibrator_text_dim=12,
        calibrator_time_fourier_bands=2,
        duration_hidden_dim=16,
        dropout=0.0,
    )


def _inputs():
    batch, frames = 2, 9
    lengths = torch.tensor([9, 6])
    mask = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    compact = torch.cat([identity_6d.repeat(41), torch.zeros(10)])
    scaffold = compact.view(1, 1, -1).expand(batch, frames, -1).clone()
    scaffold = scaffold * mask.unsqueeze(-1)
    tau = torch.zeros(batch, frames, 1)
    for index, length in enumerate(lengths.tolist()):
        tau[index, :length, 0] = torch.linspace(0.0, 1.0, length)
    text = torch.randn(batch, 5, 24)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    text_mask[1, -1] = False
    evidence = torch.rand(batch, frames, len(RETRIEVAL_FEATURE_NAMES))
    evidence = evidence * mask.unsqueeze(-1)
    return scaffold, mask, lengths, tau, text, text_mask, evidence


def test_uncertainty_adaptive_forward_shapes_and_rotations():
    model = _model()
    scaffold, mask, lengths, tau, text, text_mask, evidence = _inputs()
    outputs = model(tau, scaffold, mask, text, evidence, text_mask=text_mask)

    assert outputs["prediction"].shape == scaffold.shape
    assert outputs["trust"].shape == (2, 9, len(ARTICULATOR_NAMES))
    assert outputs["correction_need"].shape == (2, 9, len(ARTICULATOR_NAMES))
    assert outputs["knot_density"].shape == (2, 9, len(ARTICULATOR_NAMES))
    assert outputs["adaptive_coordinates"].shape == (2, 9, len(ARTICULATOR_NAMES))
    assert outputs["scale_weights"].shape == (2, 9, len(ARTICULATOR_NAMES), 3)
    assert outputs["frame_codes"].shape == (2, 9, len(ARTICULATOR_NAMES), 8)
    assert outputs["knot_counts"].shape == (2, len(ARTICULATOR_NAMES), 3)
    assert outputs["knot_counts"][0, 0, 0] == 5
    assert outputs["knot_counts"][0, 1, 0] == lengths[0]
    assert torch.all(outputs["prediction"][1, 6:] == 0)
    assert torch.allclose(
        outputs["scale_weights"][mask].sum(dim=-1),
        torch.ones_like(outputs["scale_weights"][mask][..., 0]),
        atol=1e-6,
    )

    matrices = rotation_6d_to_matrix(
        outputs["prediction"][mask][..., :246].reshape(-1, 41, 6)
    )
    assert torch.allclose(
        torch.linalg.det(matrices),
        torch.ones(matrices.shape[0], matrices.shape[1]),
        atol=1e-4,
    )


def test_correction_need_opens_gate_independently_of_trust():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    trust = torch.full((*mask.shape, len(ARTICULATOR_NAMES)), 0.8)
    density = torch.ones_like(trust)
    low_need = torch.zeros_like(trust)
    high_need = torch.ones_like(trust)
    low = model(
        tau,
        scaffold,
        mask,
        text,
        evidence,
        text_mask=text_mask,
        trust_override=trust,
        correction_need_override=low_need,
        knot_density_override=density,
    )
    high = model(
        tau,
        scaffold,
        mask,
        text,
        evidence,
        text_mask=text_mask,
        trust_override=trust,
        correction_need_override=high_need,
        knot_density_override=density,
    )
    assert high["gates"][mask].mean() > low["gates"][mask].mean() * 2.0


def test_trust_controls_scale_without_directly_defining_need():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    low_trust = torch.zeros(*mask.shape, len(ARTICULATOR_NAMES))
    high_trust = torch.ones_like(low_trust)
    need = torch.full_like(low_trust, 0.5)
    density = torch.ones_like(low_trust)
    low = model(
        tau,
        scaffold,
        mask,
        text,
        evidence,
        text_mask=text_mask,
        trust_override=low_trust,
        correction_need_override=need,
        knot_density_override=density,
    )
    high = model(
        tau,
        scaffold,
        mask,
        text,
        evidence,
        text_mask=text_mask,
        trust_override=high_trust,
        correction_need_override=need,
        knot_density_override=density,
    )
    assert high["scale_weights"][mask][..., -1].mean() > low[
        "scale_weights"
    ][mask][..., -1].mean()


def test_density_warps_knot_coordinate_toward_high_need_region():
    mask = torch.ones(1, 9, dtype=torch.bool)
    uniform = adaptive_time_coordinate(torch.ones(1, 9), mask)
    front_loaded = adaptive_time_coordinate(
        torch.tensor([[8.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        mask,
    )
    assert front_loaded[0, 4] > uniform[0, 4]
    assert torch.allclose(front_loaded[:, :1], torch.zeros(1, 1))
    assert torch.allclose(front_loaded[:, -1:], torch.ones(1, 1))


def test_targets_and_uncertainty_proxy_are_bounded():
    mask = torch.ones(1, 4, dtype=torch.bool)
    errors = torch.tensor(
        [[[0.0, 0.1, 0.3, 0.8], [0.1, 0.2, 0.4, 0.9], [0.2, 0.3, 0.5, 1.0], [0.3, 0.4, 0.6, 1.1]]]
    )
    need = correction_need_target_from_error(errors, [0.25] * 4, mask)
    correction = torch.zeros(1, 4, 133)
    correction[:, 2, 30:75] = 1.0
    density = adaptive_knot_density_target(correction, need, mask)
    evidence = torch.zeros(1, 4, len(RETRIEVAL_FEATURE_NAMES))
    uncertainty = retrieval_uncertainty_proxy(evidence)
    assert torch.all((need >= 0.0) & (need <= 1.0))
    assert torch.all((density >= 0.0) & (density <= 1.0))
    assert density[0, 2, 1] > need[0, 2, 1]
    assert torch.all((uncertainty >= 0.0) & (uncertainty <= 1.0))


def test_uncertainty_field_backward_reaches_calibrator_and_knot_encoder():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    outputs = model(tau, scaffold, mask, text, evidence, text_mask=text_mask)
    loss = (
        outputs["prediction"][mask].square().mean()
        + outputs["trust"][mask].mean()
        + outputs["correction_need"][mask].mean()
        + outputs["knot_density"][mask].mean()
        + outputs["frame_codes"][mask].square().mean()
    )
    loss.backward()
    assert model.calibrator.trust_head.weight.grad is not None
    assert model.calibrator.need_head.weight.grad is not None
    assert model.calibrator.density_head.weight.grad is not None
    assert model.code_predictor.code_heads["lhand"][1].weight.grad is not None


def test_wrist_relative_hand_loss_ignores_global_translation():
    target = {
        "wholebody": torch.zeros(2, 5, 3),
        "lhand": torch.randn(2, 21, 3),
        "rhand": torch.randn(2, 21, 3),
    }
    translation = torch.tensor([[[1.0, -2.0, 3.0]]])
    pred = {
        "wholebody": target["wholebody"],
        "lhand": target["lhand"] + translation,
        "rhand": target["rhand"] + translation,
    }
    assert torch.allclose(wrist_relative_hand_l1(pred, target), torch.tensor(0.0), atol=1e-6)


def test_shared_prediction_parts_avoid_duplicate_fk_without_changing_losses():
    class CountingFK:
        def __init__(self):
            self.calls = 0

        def parts_from_rot6d(self, compact):
            self.calls += 1
            joints = compact[:, :12].reshape(-1, 4, 3)
            return {
                "body": joints[:, :2],
                "lhand": joints[:, 1:3],
                "rhand": joints[:, 2:4],
                "wholebody": joints,
            }

    torch.manual_seed(47)
    pred = torch.randn(2, 6, 256, requires_grad=True)
    target = torch.randn_like(pred)
    lengths = torch.tensor([6, 4])
    mask = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    target_fk = CountingFK()
    flat_target = target_fk.parts_from_rot6d(target[mask])
    target_parts = {}
    for key, value in flat_target.items():
        padded = value.new_zeros((2, 6) + value.shape[1:])
        padded[mask] = value
        target_parts[key] = padded

    endpoint_weights = {
        "lambda_joint": 1.0,
        "lambda_hand": 1.0,
        "lambda_hand_relative": 1.0,
        "lambda_vel": 1.0,
        "lambda_vel_hand": 1.0,
        "lambda_acc": 1.0,
        "lambda_path": 1.0,
    }
    dynamics_weights = {
        "lambda_fk_vel": 1.0,
        "lambda_fk_acc": 1.0,
        "lambda_fk_jerk": 1.0,
        "fk_temporal_include_hand_parts": True,
    }

    shared_fk = CountingFK()
    shared_parts = prediction_parts_from_rot6d(
        pred, mask, target_parts, shared_fk
    )
    shared_endpoint, _ = endpoint_losses(
        pred,
        target,
        mask,
        lengths,
        target_parts,
        fk=shared_fk,
        weights=endpoint_weights,
        pred_parts=shared_parts,
    )
    shared_dynamics, _ = fk_temporal_dynamics_losses(
        pred,
        mask,
        lengths,
        target_parts,
        shared_fk,
        weights=dynamics_weights,
        pred_parts=shared_parts,
    )

    separate_fk = CountingFK()
    separate_endpoint, _ = endpoint_losses(
        pred,
        target,
        mask,
        lengths,
        target_parts,
        fk=separate_fk,
        weights=endpoint_weights,
    )
    separate_dynamics, _ = fk_temporal_dynamics_losses(
        pred,
        mask,
        lengths,
        target_parts,
        separate_fk,
        weights=dynamics_weights,
    )

    assert shared_fk.calls == 1
    assert separate_fk.calls == 2
    assert torch.allclose(shared_endpoint, separate_endpoint)
    assert torch.allclose(shared_dynamics, separate_dynamics)
    (shared_endpoint + shared_dynamics).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_builder_and_composite_selection_use_new_model_path():
    model = build_retrieval_adaptive_model(
        {
            "model": {
                "type": "retrieval_uncertainty_adaptive_knot_field",
                "code_dim": 8,
                "context_hidden_dim": 32,
                "context_heads": 4,
                "hidden_dim": 32,
                "depth": 2,
                "articulator_code_strides": {
                    "body": [2, 4, 8],
                    "lhand": [1, 2, 4],
                    "rhand": [1, 2, 4],
                    "face": [2, 4, 8],
                },
            },
            "duration": {"hidden_dim": 16},
        },
        text_dim=24,
    )
    assert isinstance(model, RetrievalUncertaintyAdaptiveKnotField)
    score = validation_selection_score(
        {"val_a": 2.0, "val_b": 3.0},
        {"selection": {"weights": {"a": 0.5, "val_b": 2.0}}},
    )
    assert score == 7.0
