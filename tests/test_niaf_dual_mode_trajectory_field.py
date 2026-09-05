from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES
from flow.smplx_features import COMPACT6D_DIM
from NIAF.continuous_sign_field.config import load_config
from NIAF.continuous_trajectory_field.losses import coarse_and_residual_losses
from NIAF.continuous_trajectory_field.models import (
    DualModeContinuousTrajectoryField,
    TrajectoryInstance,
    build_continuous_trajectory_field,
)
from NIAF.continuous_trajectory_field.scripts.train_continuous_trajectory_field import (
    _word_prior_availability,
    checkpoint_selection_diagnostics,
    configured_word_prior_eval_modes,
    configured_word_prior_train_mode,
    evaluate_configured_modes,
    memory_microbatch_size,
    requires_scaffold_provider,
    synchronized_memory_microbatch_size,
    validate_checkpoint_contract,
    wandb_train_batch_payload,
    wandb_train_epoch_payload,
)


def _ddp_microbatch_schedule_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        # The ranks deliberately receive different padded frame counts. The
        # frame budget therefore permits four samples on rank 0 but only two
        # on rank 1 before synchronization.
        padded_frames = 2 if rank == 0 else 4
        batch = {
            "name": [f"rank-{rank}-sample-{index}" for index in range(4)],
            "motion": torch.empty(4, padded_frames, 1),
        }
        cfg = {
            "train": {
                "max_samples_per_memory_batch": 4,
                "max_frames_per_memory_batch": 8,
            }
        }
        assert memory_microbatch_size(batch, cfg) == (4 if rank == 0 else 2)
        shared_size = synchronized_memory_microbatch_size(
            batch,
            cfg,
            {
                "enabled": True,
                "rank": rank,
                "world_size": world_size,
                "backend": "gloo",
            },
            torch.device("cpu"),
        )
        assert shared_size == 2

        model = DistributedDataParallel(torch.nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(batch["name"]), shared_size):
            end = min(start + shared_size, len(batch["name"]))
            sample_count = end - start
            inputs = torch.full((sample_count, 1), float(rank + 1))
            loss = (
                model(inputs).square().mean()
                * sample_count
                / len(batch["name"])
            )
            loss.backward()
        optimizer.step()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch Gloo distributed backend is unavailable",
)
def test_ddp_microbatch_schedule_synchronizes_different_rank_lengths(tmp_path):
    init_file = (tmp_path / "ddp_microbatch_init").resolve()
    mp.spawn(
        _ddp_microbatch_schedule_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )


def _dual_model():
    torch.manual_seed(211)
    return build_continuous_trajectory_field(
        {
            "model": {
                "type": "dual_mode_continuous_trajectory_field",
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
                "temporal_slot_count": 16,
                "temporal_slot_layers": 2,
                "temporal_slot_heads": 8,
                "context_fps": 20.0,
            },
        },
        text_dim=24,
    )


def _dual_inputs():
    batch, text_length, frames = 2, 5, 7
    text = torch.randn(batch, text_length, 24)
    text_mask = torch.ones(batch, text_length, dtype=torch.bool)
    text_mask[1, -1] = False
    context = torch.randn(batch, frames, COMPACT6D_DIM)
    context_mask = torch.ones(batch, frames, dtype=torch.bool)
    features = torch.randn(batch, frames, len(RETRIEVAL_FEATURE_NAMES))
    query = torch.linspace(-1.0, 1.0, 9).repeat(batch, 1)
    return text, text_mask, context, context_mask, features, query


def test_dual_mode_builder_supports_text_only_without_prior_tensors():
    model = _dual_model().eval()
    text, text_mask, _context, _context_mask, _features, query = _dual_inputs()
    outputs = model(text, query, text_mask=text_mask)

    assert isinstance(model, DualModeContinuousTrajectoryField)
    assert outputs["prediction"].shape == (2, 9, COMPACT6D_DIM)
    assert outputs["coarse"].shape == outputs["prediction"].shape
    assert "prior" not in outputs
    trajectory = outputs["trajectory"]
    assert trajectory.temporal_slot_tau.shape == (2, 16)
    assert trajectory.word_prior_available.tolist() == [False, False]
    assert torch.count_nonzero(trajectory.word_prior_gates) == 0

    restored = TrajectoryInstance.from_tensor_dict(trajectory.detach().tensor_dict())
    assert torch.equal(restored.word_prior_available, trajectory.word_prior_available)
    assert torch.allclose(restored.word_prior_gates, trajectory.word_prior_gates)


def test_dual_mode_mixed_batch_zeroes_dropped_prior_and_is_differentiable():
    model = _dual_model()
    text, text_mask, context, context_mask, features, query = _dual_inputs()
    outputs = model(
        text,
        query,
        text_mask=text_mask,
        word_prior_context=context,
        word_prior_mask=context_mask,
        word_prior_features=features,
        word_prior_available=torch.tensor([True, False]),
    )
    gates = outputs["trajectory"].word_prior_gates
    assert torch.count_nonzero(gates[0]) > 0
    assert torch.count_nonzero(gates[1]) == 0

    outputs["prediction"].square().mean().backward()
    assert model.hypernetwork.text_planner.text_projection.weight.grad is not None
    assert model.hypernetwork.word_encoder.frame_input[0].weight.grad is not None
    assert model.hypernetwork.word_fusion.weight.grad is not None


def test_dual_mode_same_instance_remains_query_grid_invariant():
    model = _dual_model().eval()
    text, text_mask, _context, _context_mask, _features, _query = _dual_inputs()
    trajectory = model.encode_trajectory(text, text_mask=text_mask)
    common = torch.tensor([[-1.0, -0.25, 0.5, 1.0]]).repeat(2, 1)
    direct = model.query_trajectory(trajectory, common)
    merged = model.query_trajectory(
        trajectory,
        torch.cat([torch.zeros(2, 2), common], dim=1),
    )
    assert torch.allclose(direct, merged[:, -common.shape[1] :], atol=1e-7)


def test_word_prior_dropout_handles_both_deterministic_endpoints():
    assert not _word_prior_availability(
        8,
        {"conditioning": {"word_prior_dropout_probability": 1.0}},
        torch.device("cpu"),
        "dropout",
    ).any()
    assert _word_prior_availability(
        8,
        {"conditioning": {"word_prior_dropout_probability": 0.0}},
        torch.device("cpu"),
        "dropout",
    ).all()


def test_word_prior_modes_preserve_legacy_defaults_and_support_pure_text():
    legacy = {"model": {"type": "dual_mode_continuous_trajectory_field"}}
    assert configured_word_prior_train_mode(legacy) == "dropout"
    assert configured_word_prior_eval_modes(legacy) == ("off", "on")
    assert requires_scaffold_provider(legacy)

    pure_text = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "conditioning": {"word_prior_train_mode": "off"},
        "eval": {"word_prior_modes": ["off"]},
    }
    assert configured_word_prior_train_mode(pure_text) == "off"
    assert configured_word_prior_eval_modes(pure_text) == ("off",)
    assert not requires_scaffold_provider(pure_text)


def test_word_prior_modes_reject_stochastic_validation():
    cfg = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "eval": {"word_prior_modes": ["dropout"]},
    }
    with pytest.raises(ValueError, match="deterministic"):
        configured_word_prior_eval_modes(cfg)


def test_coarse_auxiliary_is_supervised_against_ground_truth():
    model = _dual_model().eval()
    text, text_mask, _context, _context_mask, _features, query = _dual_inputs()
    outputs = model(text, query, text_mask=text_mask)
    mask = torch.ones(query.shape, dtype=torch.bool)
    target = outputs["coarse"].detach().clone()
    losses = coarse_and_residual_losses(outputs, target, mask)
    assert torch.allclose(losses["loss_coarse"], torch.tensor(0.0))


def test_dual_checkpoint_selection_is_text_led_with_two_percent_gate():
    cfg = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "selection": {
            "weights": {"pred_loss_endpoint": 1.0},
            "constraints": None,
            "word_prior_max_relative_degradation": 0.02,
        },
    }
    acceptable = {
        "text_only/pred_loss_endpoint": 10.0,
        "word_prior/pred_loss_endpoint": 10.2,
    }
    score, violation, feasible, details = checkpoint_selection_diagnostics(
        acceptable, cfg, return_details=True
    )
    assert score == 10.0
    assert violation <= 1e-12
    assert feasible
    assert details["dual_mode"]["selection_source"] == "text_only"

    rejected = dict(acceptable)
    rejected["word_prior/pred_loss_endpoint"] = 10.21
    score, violation, feasible, details = checkpoint_selection_diagnostics(
        rejected, cfg, return_details=True
    )
    assert score == 10.0
    assert violation > 0.0
    assert not feasible
    assert details["rejection_reasons"]


@pytest.mark.parametrize(
    ("namespace", "expected_source"),
    (("text_only", "text_only"), ("word_prior", "word_prior")),
)
def test_dual_checkpoint_selection_supports_one_configured_mode(
    namespace, expected_source
):
    cfg = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "selection": {"weights": {"pred_loss_endpoint": 1.0}},
    }
    metrics = {f"{namespace}/pred_loss_endpoint": 2.5}
    score, violation, feasible, details = checkpoint_selection_diagnostics(
        metrics, cfg, return_details=True
    )
    assert score == 2.5
    assert violation == 0.0
    assert feasible
    assert details["dual_mode"]["selection_source"] == expected_source


def test_v2_checkpoint_contract_cannot_load_v1_identity():
    cfg = {"model": {"type": "dual_mode_continuous_trajectory_field"}}
    validate_checkpoint_contract(
        {
            "model_type": "dual_mode_continuous_trajectory_field",
            "trajectory_contract_version": 2,
        },
        cfg,
    )
    with pytest.raises(RuntimeError, match="cannot load v1 weights"):
        validate_checkpoint_contract(
            {
                "model_type": "continuous_trajectory_field",
                "trajectory_contract_version": 1,
            },
            cfg,
        )


def test_checkpoint_contract_rejects_different_text_encoder():
    cfg = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "text": {"model_path": "deps/mt5-base"},
    }
    with pytest.raises(RuntimeError, match="semantically incompatible"):
        validate_checkpoint_contract(
            {
                "model_type": "dual_mode_continuous_trajectory_field",
                "trajectory_contract_version": 2,
                "config": {"text": {"model_path": "deps/flan-t5-base"}},
            },
            cfg,
        )


def test_v2_wandb_payloads_separate_mode_statistics():
    batch = wandb_train_batch_payload(
        {
            "loss_total": 1.0,
            "mode_text_only_compact_l1": 2.0,
            "mode_word_prior_compact_l1": 1.5,
            "word_prior_gate_mean_body": 0.25,
        },
        epoch=1,
        optimizer_step=2,
        logical_batch=3,
        logical_batches=4,
    )
    assert batch["train/text_only/batch/compact_l1"] == 2.0
    assert batch["train/word_prior/batch/compact_l1"] == 1.5
    assert batch["train/word_prior/gates/mean_body"] == 0.25

    epoch = wandb_train_epoch_payload(
        {
            "epoch": 1,
            "global_step": 2,
            "elapsed_sec": 3.0,
            "train_mode_text_only_compact_l1": 2.0,
            "train_mode_word_prior_compact_l1": 1.5,
        }
    )
    assert epoch["train/text_only/epoch/compact_l1"] == 2.0
    assert epoch["train/word_prior/epoch/compact_l1"] == 1.5


def test_v2_configs_resolve_the_approved_contract():
    config_root = Path("NIAF/continuous_trajectory_field/configs")
    for filename in (
        "phoenix_signtrajfield_v2_overfit5.yaml",
        "phoenix_signtrajfield_v2_pilot500.yaml",
        "phoenix_signtrajfield_v2_full.yaml",
        "how2sign_signtrajfield_v2_full.yaml",
        "csl_daily_signtrajfield_v2_full.yaml",
    ):
        cfg = load_config(config_root / filename)
        assert cfg["model"]["type"] == "dual_mode_continuous_trajectory_field"
        assert cfg["conditioning"]["temporal_slot_count"] == 16
        assert cfg["conditioning"]["temporal_slot_layers"] == 2
        assert cfg["conditioning"]["temporal_slot_heads"] == 8
        assert cfg["conditioning"]["word_prior_dropout_probability"] == 0.5
        assert cfg["train"]["local_warmup_epochs"] == 0
        assert cfg["selection"]["word_prior_max_relative_degradation"] == 0.02


def test_csl_mt5_ablation_configs_are_matched_except_for_prior_mode():
    config_root = Path("NIAF/continuous_trajectory_field/configs")
    text_only = load_config(
        config_root / "csl_daily_signtrajfield_v2_mt5_text_only_full.yaml"
    )
    word_prior = load_config(
        config_root / "csl_daily_signtrajfield_v2_mt5_word_prior_full.yaml"
    )

    assert text_only["text"]["model_path"] == "deps/mt5-base"
    assert word_prior["text"]["model_path"] == "deps/mt5-base"
    assert text_only["conditioning"]["word_prior_train_mode"] == "off"
    assert word_prior["conditioning"]["word_prior_train_mode"] == "dropout"
    assert text_only["eval"]["word_prior_modes"] == ["off"]
    assert word_prior["eval"]["word_prior_modes"] == ["off", "on"]
    assert not requires_scaffold_provider(text_only)
    assert requires_scaffold_provider(word_prior)


def test_dual_validation_runs_off_and_on_with_separate_namespaces():
    cfg = {"model": {"type": "dual_mode_continuous_trajectory_field"}}
    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.evaluate",
        side_effect=[
            {"pred_loss_endpoint": 2.0},
            {"pred_loss_endpoint": 1.5},
        ],
    ) as evaluate_mock:
        metrics = evaluate_configured_modes(
            None,
            None,
            None,
            None,
            None,
            None,
            cfg,
            torch.device("cpu"),
            show_progress=False,
        )

    assert metrics == {
        "text_only/pred_loss_endpoint": 2.0,
        "word_prior/pred_loss_endpoint": 1.5,
    }
    assert [
        call.kwargs["word_prior_mode"] for call in evaluate_mock.call_args_list
    ] == ["off", "on"]


def test_dual_validation_runs_only_the_configured_text_mode():
    cfg = {
        "model": {"type": "dual_mode_continuous_trajectory_field"},
        "eval": {"word_prior_modes": ["off"]},
    }
    with patch(
        "NIAF.continuous_trajectory_field.scripts."
        "train_continuous_trajectory_field.evaluate",
        return_value={"pred_loss_endpoint": 2.0},
    ) as evaluate_mock:
        metrics = evaluate_configured_modes(
            None,
            None,
            None,
            None,
            None,
            None,
            cfg,
            torch.device("cpu"),
            show_progress=False,
        )

    assert metrics == {"text_only/pred_loss_endpoint": 2.0}
    assert evaluate_mock.call_args.kwargs["word_prior_mode"] == "off"
