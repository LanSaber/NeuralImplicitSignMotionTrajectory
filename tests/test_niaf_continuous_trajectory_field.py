import inspect
from unittest.mock import patch

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
    configure_wandb_metrics,
    evaluate,
    load_warm_start_state,
    memory_microbatch_size,
    selection_diagnostics,
    selection_score,
    validation_microbatch_size,
    wandb_train_batch_payload,
    wandb_train_epoch_payload,
    wandb_validation_payload,
    wandb_validation_pending_payload,
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
    weight_sum = outputs["local_weights"].sum(dim=-1)
    assert torch.all(weight_sum >= 0.0)
    assert torch.all(weight_sum <= 1.0)
    assert torch.allclose(weight_sum, outputs["local_coverage"])


def test_local_windows_have_absolute_support_and_fade_outside_centers():
    model = _model(max_local_fields=4).eval()
    trajectory = _instance(model).detach().clone()
    trajectory.local_centers.zero_()
    trajectory.local_widths.fill_(0.05)
    query = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    outputs = model.query_trajectory(trajectory, query, return_details=True)
    assert torch.all(outputs["local_coverage"][:, (0, 2)] < 1e-6)
    assert torch.all(outputs["local_coverage"][:, 1] > 0.99)
    assert torch.allclose(
        outputs["local_correction_axis"][:, (0, 2)],
        torch.zeros_like(outputs["local_correction_axis"][:, (0, 2)]),
        atol=1e-7,
    )


def test_reset_local_branch_starts_as_small_delta_from_shared_field():
    model = _model(max_local_fields=4).eval()
    model.hypernetwork.reset_local_branch()
    trajectory = _instance(model)
    query = torch.linspace(-1.0, 1.0, 11).repeat(2, 1)
    outputs = model.query_trajectory(trajectory, query, return_details=True)
    assert torch.sqrt(outputs["local_correction_axis"].square().mean()) < 1e-3


def test_global_and_local_residual_branches_can_be_ablated_independently():
    model = _model(max_local_fields=4).eval()
    trajectory = _instance(model)
    query = torch.linspace(-1.0, 1.0, 11).repeat(2, 1)
    full = model.query_trajectory(trajectory, query, return_details=True)
    global_only = model.query_trajectory(
        trajectory,
        query,
        return_details=True,
        include_local_residual=False,
    )
    local_only = model.query_trajectory(
        trajectory,
        query,
        return_details=True,
        include_global_residual=False,
    )
    assert torch.count_nonzero(global_only["local_correction_axis"]) == 0
    assert torch.count_nonzero(local_only["global_correction_axis"]) == 0
    assert torch.allclose(
        full["correction_axis"],
        global_only["correction_axis"] + local_only["correction_axis"],
    )


def test_part_specific_local_experts_and_time_gates_are_serializable_and_trainable():
    torch.manual_seed(103)
    model = ContinuousTrajectoryField(
        text_dim=24,
        context_hidden_dim=32,
        context_layers=1,
        field_hidden_dim=32,
        field_depth=2,
        max_local_fields=4,
        frames_per_local_field=4,
        minimum_local_width=0.1,
        maximum_local_width=0.6,
        part_specific_local_experts=True,
        time_dependent_local_gates=True,
        dropout=0.0,
    )
    trajectory = _instance(model)
    assert trajectory.local_part_gates.shape == (2, 4, 4)
    assert torch.all(trajectory.local_part_gates[trajectory.local_mask] > 0.0)
    assert torch.count_nonzero(trajectory.local_part_gates[~trajectory.local_mask]) == 0
    restored = TrajectoryInstance.from_tensor_dict(trajectory.detach().tensor_dict())
    assert torch.allclose(restored.local_part_gates, trajectory.local_part_gates)

    query = torch.linspace(-1.0, 1.0, 11).repeat(2, 1)
    outputs = model.query_trajectory(trajectory, query, return_details=True)
    assert outputs["local_part_gates"].shape == (2, 11, 4)
    outputs["prediction"].square().mean().backward()
    assert model.hypernetwork.local_gate_head.weight.grad is not None
    for field in model.part_local_fields.values():
        assert field.output.weight.grad is not None


def test_stage2_warm_start_accepts_only_new_part_expert_parameters():
    common = {
        "text_dim": 24,
        "context_hidden_dim": 32,
        "context_layers": 1,
        "field_hidden_dim": 32,
        "field_depth": 2,
        "max_local_fields": 4,
        "frames_per_local_field": 4,
        "dropout": 0.0,
    }
    stage1 = ContinuousTrajectoryField(**common)
    stage2 = ContinuousTrajectoryField(
        **common,
        part_specific_local_experts=True,
        time_dependent_local_gates=True,
    )
    incompatible = load_warm_start_state(stage2, stage1.state_dict())
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        name.startswith(("part_local_fields.", "hypernetwork.local_gate_head."))
        for name in incompatible.missing_keys
    )
    assert torch.allclose(
        stage2.residual_field.output.weight,
        stage1.residual_field.output.weight,
    )


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


def test_memory_microbatch_is_bounded_by_samples_and_padded_frames():
    cfg = {
        "train": {
            "max_samples_per_memory_batch": 32,
            "max_frames_per_memory_batch": 4096,
        }
    }
    short_batch = {"name": [str(index) for index in range(64)], "motion": torch.zeros(64, 40, 1)}
    long_batch = {"name": [str(index) for index in range(32)], "motion": torch.zeros(32, 376, 1)}

    assert memory_microbatch_size(short_batch, cfg) == 32
    assert memory_microbatch_size(long_batch, cfg) == 10


def test_validation_microbatch_defaults_to_one_sample():
    batch = {
        "name": [str(index) for index in range(8)],
        "motion": torch.zeros(8, 40, 1),
    }

    assert validation_microbatch_size(batch, {"train": {}}) == 1


def test_evaluate_streams_a_logical_batch_as_validation_microbatches():
    batch = {
        "name": [str(index) for index in range(4)],
        "motion": torch.zeros(4, 40, 1),
    }
    calls = []

    class Model:
        def eval(self):
            return self

    def fake_evaluate_microbatch(*args, **kwargs):
        microbatch = args[4]
        calls.append(len(microbatch["name"]))
        return {"pred_loss_total": 2.0}

    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.evaluate_microbatch",
        side_effect=fake_evaluate_microbatch,
    ):
        metrics = evaluate(
            Model(),
            None,
            None,
            None,
            [batch],
            None,
            {"eval": {"max_samples_per_memory_batch": 1}},
            torch.device("cpu"),
            show_progress=False,
        )

    assert calls == [1, 1, 1, 1]
    assert metrics["pred_loss_total"] == 2.0


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


def test_selection_supports_scaffold_relative_metrics_and_rejection_reasons():
    cfg = {
        "selection": {
            "weights": {
                "pred_loss_endpoint": {
                    "weight": 1.0,
                    "relative_to": "scaffold_loss_endpoint",
                }
            },
            "constraint_penalty": 10.0,
            "constraints": {
                "pred_loss_endpoint": {
                    "relative_to": "scaffold_loss_endpoint",
                    "max": 0.0,
                    "scale": 1.0,
                }
            },
        }
    }
    feasible_metrics = {
        "pred_loss_endpoint": 10.0,
        "scaffold_loss_endpoint": 11.0,
    }
    score, violation, feasible, details = selection_diagnostics(
        feasible_metrics, cfg, return_details=True
    )
    assert score == -1.0
    assert violation == 0.0
    assert feasible
    assert details["rejection_reasons"] == []

    rejected_metrics = {
        "pred_loss_endpoint": 12.0,
        "scaffold_loss_endpoint": 11.0,
    }
    score, violation, feasible, details = selection_diagnostics(
        rejected_metrics, cfg, return_details=True
    )
    assert score == 11.0
    assert violation == 1.0
    assert not feasible
    assert len(details["rejection_reasons"]) == 1


def test_wandb_payloads_keep_train_and_validation_namespaces_separate():
    row = {
        "epoch": 5,
        "global_step": 555,
        "elapsed_sec": 123.0,
        "lr_global": 2e-5,
        "lr_local": 1e-4,
        "train_loss_total": 9.0,
        "train_loss_path": 0.4,
        "val_pred_loss_total": 10.0,
        "val_scaffold_loss_total": 11.0,
        "validation_pending": 0.0,
        "selection_score": -0.5,
        "selection_feasible": 1.0,
        "selection_rejection_reasons": "",
    }

    train = wandb_train_epoch_payload(row)
    assert train["train/epoch/loss_total"] == 9.0
    assert train["train/epoch/loss_path"] == 0.4
    assert train["optimizer/global_lr"] == 2e-5
    assert train["optimizer/local_lr"] == 1e-4
    assert not any(name.startswith("validation/") for name in train)

    validation = wandb_validation_payload(row)
    assert validation["validation/pred_loss_total"] == 10.0
    assert validation["validation/scaffold_loss_total"] == 11.0
    assert validation["validation/selection/score"] == -0.5
    assert validation["validation/pending"] == 0.0
    assert not any(name.startswith("train/") for name in validation)

    pending = wandb_validation_pending_payload(epoch=5, global_step=555)
    assert pending == {
        "validation/epoch_step": 5,
        "validation/global_step": 555,
        "validation/pending": 1.0,
    }


def test_wandb_batch_payload_uses_optimizer_step_and_selected_metrics():
    payload = wandb_train_batch_payload(
        {
            "loss_total": 9.5,
            "loss_path": 0.3,
            "residual_rms": 0.1,
            "unbounded_debug_metric": 123.0,
        },
        epoch=5,
        optimizer_step=445,
        logical_batch=2,
        logical_batches=222,
    )
    assert payload["train/optimizer_step"] == 445
    assert payload["train/batch/epoch"] == 5
    assert payload["train/batch/logical_batch"] == 2
    assert payload["train/batch/loss_total"] == 9.5
    assert payload["train/batch/loss_path"] == 0.3
    assert "train/batch/unbounded_debug_metric" not in payload


def test_wandb_metric_definitions_use_independent_custom_steps():
    class Run:
        def __init__(self):
            self.calls = []

        def define_metric(self, name, **kwargs):
            self.calls.append((name, kwargs))

    run = Run()
    configure_wandb_metrics(run)
    definitions = dict(run.calls)
    assert definitions["train/batch/*"] == {
        "step_metric": "train/optimizer_step"
    }
    assert definitions["train/epoch/*"] == {"step_metric": "train/epoch_step"}
    assert definitions["validation/*"] == {
        "step_metric": "validation/epoch_step"
    }
