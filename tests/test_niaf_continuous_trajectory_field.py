import inspect

import torch

from flow.smplx_features import rotation_6d_to_matrix
from NIAF.continuous_trajectory_field.derivatives import (
    finite_physical_derivatives,
    physical_derivatives,
    sample_padded_sequence,
)
from NIAF.continuous_trajectory_field.models import (
    ContinuousTrajectoryField,
    TrajectoryInstance,
    build_continuous_trajectory_field,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    selection_diagnostics,
    selection_score,
)


def _model(max_local_fields=4):
    torch.manual_seed(101)
    return ContinuousTrajectoryField(
        text_dim=24,
        context_hidden_dim=32,
        context_layers=1,
        field_hidden_dim=32,
        field_depth=2,
        max_local_fields=max_local_fields,
        frames_per_local_field=4,
        minimum_local_width=0.1,
        maximum_local_width=0.6,
        quantile_temperature=0.05,
        residual_amplitude=0.1,
        dropout=0.0,
    )


def _inputs():
    batch, frames = 2, 11
    lengths = torch.tensor([11, 7])
    mask = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    identity = torch.cat(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(41),
            torch.zeros(10),
        ]
    )
    context = identity.view(1, 1, -1).repeat(batch, frames, 1)
    context = context + 0.01 * torch.randn_like(context)
    context = context * mask.unsqueeze(-1)
    evidence = torch.rand(batch, frames, 7) * mask.unsqueeze(-1)
    text = torch.randn(batch, 5, 24)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    text_mask[1, -2:] = False
    return context, mask, lengths, evidence, text, text_mask


def _instance(model):
    context, mask, _lengths, evidence, text, text_mask = _inputs()
    return model.encode_trajectory(text, context, mask, evidence, text_mask=text_mask)


def test_forward_shape_and_valid_rotations():
    model = _model()
    context, mask, _lengths, evidence, text, text_mask = _inputs()
    query = torch.linspace(-1.0, 1.0, 17).repeat(2, 1)
    outputs = model(
        text,
        context,
        mask,
        evidence,
        query,
        text_mask=text_mask,
    )
    assert outputs["prediction"].shape == (2, 17, 256)
    assert outputs["prior"].shape == (2, 17, 256)
    assert outputs["correction_axis"].shape == (2, 17, 133)
    matrices = rotation_6d_to_matrix(
        outputs["prediction"][..., :246].reshape(2, 17, 41, 6)
    )
    assert torch.allclose(
        torch.linalg.det(matrices),
        torch.ones(2, 17, 41),
        atol=1e-4,
    )


def test_same_instance_is_query_grid_and_order_invariant():
    model = _model().eval()
    trajectory = _instance(model)
    common = torch.tensor([[-1.0, -0.4, 0.0, 0.35, 1.0]]).repeat(2, 1)
    direct = model.query_trajectory(trajectory, common)
    extra = torch.tensor([[-0.8, -0.2, 0.7]]).repeat(2, 1)
    merged = model.query_trajectory(trajectory, torch.cat([extra, common], dim=1))
    assert torch.allclose(direct, merged[:, -common.shape[1] :], atol=1e-7)

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model.query_trajectory(trajectory, common[:, permutation])
    assert torch.allclose(direct, permuted[:, torch.argsort(permutation)], atol=1e-7)


def test_trajectory_serialization_round_trip():
    model = _model()
    trajectory = _instance(model).detach()
    restored = TrajectoryInstance.from_tensor_dict(trajectory.tensor_dict())
    assert restored.batch_size == trajectory.batch_size
    assert restored.num_local_fields == trajectory.num_local_fields
    for key, value in trajectory.tensor_dict().items():
        restored_value = restored.tensor_dict()[key]
        if value.dtype == torch.bool:
            assert torch.equal(value, restored_value)
        else:
            assert torch.allclose(value, restored_value)
    query = torch.linspace(-1.0, 1.0, 9).repeat(2, 1)
    assert torch.allclose(
        model.query_trajectory(trajectory, query),
        model.query_trajectory(restored, query),
    )
    half = trajectory.to(dtype=torch.float16)
    assert half.duration_seconds.dtype == torch.float16
    assert half.local_mask.dtype == torch.bool


def test_local_centers_widths_masks_and_partition():
    model = _model(max_local_fields=4)
    trajectory = _instance(model)
    assert torch.equal(trajectory.local_mask.sum(dim=1), torch.tensor([3, 2]))
    for batch_index in range(2):
        active = trajectory.local_mask[batch_index]
        centers = trajectory.local_centers[batch_index, active]
        widths = trajectory.local_widths[batch_index, active]
        assert torch.all(centers[1:] >= centers[:-1])
        assert torch.all(centers >= -1.0)
        assert torch.all(centers <= 1.0)
        assert torch.all(widths >= 0.1)
        assert torch.all(widths <= 0.6)
    query = torch.linspace(-1.0, 1.0, 13).repeat(2, 1)
    outputs = model.query_trajectory(trajectory, query, return_details=True)
    assert outputs["local_weights"].shape == (2, 13, 4)
    assert torch.allclose(outputs["local_weights"].sum(dim=-1), torch.ones(2, 13))


def test_global_only_field_has_empty_local_state():
    model = _model(max_local_fields=0)
    trajectory = _instance(model)
    inactive_modules = (
        model.hypernetwork.density_head,
        model.hypernetwork.local_context,
        model.hypernetwork.local_head,
    )
    assert not any(
        parameter.requires_grad
        for module in inactive_modules
        for parameter in module.parameters()
    )
    assert torch.count_nonzero(trajectory.context_density) == 0
    assert trajectory.local_scale.shape == (2, 0, 2, 32)
    assert trajectory.local_mask.shape == (2, 0)
    query = torch.linspace(-1.0, 1.0, 7).repeat(2, 1)
    outputs = model.query_trajectory(trajectory, query, return_details=True)
    assert outputs["prediction"].shape == (2, 7, 256)
    assert outputs["local_weights"].shape == (2, 7, 0)


def test_backward_reaches_hypernetwork_and_shared_fields():
    model = _model()
    context, mask, _lengths, evidence, text, text_mask = _inputs()
    query = torch.linspace(-1.0, 1.0, 9).repeat(2, 1)
    outputs = model(text, context, mask, evidence, query, text_mask=text_mask)
    loss = outputs["prediction"].square().mean()
    loss.backward()
    assert model.hypernetwork.frame_input[0].weight.grad is not None
    assert model.hypernetwork.local_head.weight.grad is not None
    assert model.prior_field.output.weight.grad is not None
    assert model.residual_field.output.weight.grad is not None
    assert torch.isfinite(model.hypernetwork.frame_input[0].weight.grad).all()


def test_physical_derivative_scaling_is_exact():
    tau = torch.linspace(-1.0, 1.0, 7).repeat(2, 1)

    def cubic(values):
        return values.unsqueeze(-1).pow(3)

    _value, derivatives = physical_derivatives(
        cubic,
        tau,
        torch.tensor([2.0, 4.0]),
        max_order=3,
    )
    assert torch.allclose(derivatives[1][0, :, 0], 3.0 * tau[0].square())
    assert torch.allclose(derivatives[3][0, :, 0], torch.full((7,), 6.0))
    assert torch.allclose(derivatives[3][1, :, 0], torch.full((7,), 0.75))


def test_finite_derivative_targets_and_sampling_cover_valid_frames():
    time = torch.linspace(0.0, 1.0, 9)
    values = torch.zeros(2, 9, 1, 1)
    values[0, :, 0, 0] = time.square()
    values[1, :6, 0, 0] = torch.linspace(0.0, 1.0, 6).square()
    derivatives = finite_physical_derivatives(
        values,
        torch.tensor([9, 6]),
        torch.tensor([1.0, 1.0]),
        max_order=3,
        smooth_kernel=1,
    )
    assert derivatives[1].shape == values.shape
    query = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    sampled = sample_padded_sequence(values, query, torch.tensor([9, 6]))
    assert sampled.shape == (2, 3, 1, 1)
    assert torch.allclose(sampled[:, 0], torch.zeros(2, 1, 1))
    assert torch.allclose(sampled[:, -1], torch.ones(2, 1, 1))


def test_builder_and_inference_signature_keep_gt_out_of_model_contract():
    model = build_continuous_trajectory_field(
        {
            "model": {
                "context_hidden_dim": 32,
                "context_layers": 1,
                "field_hidden_dim": 32,
                "field_depth": 2,
                "max_local_fields": 0,
            },
            "duration": {"initial_seconds": 3.0},
        },
        text_dim=24,
    )
    assert isinstance(model, ContinuousTrajectoryField)
    signature = inspect.signature(model.encode_trajectory)
    assert "target" not in signature.parameters
    assert "ground_truth" not in signature.parameters
    assert set(signature.parameters) == {
        "text_tokens",
        "adapter_context",
        "context_mask",
        "retrieval_evidence",
        "text_mask",
    }


def test_selection_constraints_are_normalized_and_report_feasibility():
    cfg = {
        "selection": {
            "weights": {"pred_loss_endpoint": 1.0},
            "constraint_penalty": 100.0,
            "constraints": {
                "pred_loss_endpoint": {"max": 1.8, "scale": 1.8},
                "pred_dense_analytic_fk_jerk_ratio": {"max": 1.0, "scale": 1.0},
            },
        }
    }
    feasible_metrics = {
        "pred_loss_endpoint": 1.5,
        "pred_dense_analytic_fk_jerk_ratio": 0.9,
    }
    score, violation, feasible = selection_diagnostics(feasible_metrics, cfg)
    assert score == 1.5
    assert violation == 0.0
    assert feasible
    assert selection_score(feasible_metrics, cfg) == score

    violating_metrics = {
        "pred_loss_endpoint": 2.7,
        "pred_dense_analytic_fk_jerk_ratio": 1.5,
    }
    score, violation, feasible = selection_diagnostics(violating_metrics, cfg)
    assert abs(violation - 1.0) < 1e-8
    assert abs(score - 102.7) < 1e-8
    assert not feasible
