import numpy as np
import torch

from NIAF.continuous_sign_field.meta_learning import adapt_code, build_support_query_masks
from NIAF.continuous_sign_field.models import MetaImplicitResidualField
from NIAF.continuous_sign_field.scripts.export_meta_implicit_samples import save_meta_npz


def _model():
    torch.manual_seed(7)
    return MetaImplicitResidualField(
        pose_dim=16,
        text_dim=12,
        code_dim=8,
        context_hidden_dim=16,
        hidden_dim=24,
        depth=2,
        time_fourier_bands=3,
    )


def test_meta_implicit_forward_shape():
    model = _model()
    batch, frames, pose_dim = 2, 5, 16
    scaffold = torch.randn(batch, frames, pose_dim)
    mask = torch.ones(batch, frames, dtype=torch.bool)
    lengths = torch.tensor([5, 4])
    mask[1, -1] = False
    tau = torch.linspace(0, 1, frames).view(1, frames, 1).expand(batch, -1, -1)
    text = torch.randn(batch, 3, 12)
    text_mask = torch.ones(batch, 3, dtype=torch.bool)

    code = model.initial_code(scaffold, mask, lengths, text_tokens=text, text_mask=text_mask)
    residual = model(tau, scaffold, code, mask=mask)

    assert code.shape == (batch, 8)
    assert residual.shape == (batch, frames, pose_dim)
    assert torch.all(residual[1, -1] == 0)


def test_meta_implicit_gap_conditioning_is_optional_and_shape_checked():
    model = MetaImplicitResidualField(
        pose_dim=16,
        text_dim=12,
        code_dim=8,
        context_hidden_dim=16,
        hidden_dim=24,
        depth=2,
        time_fourier_bands=3,
        condition_dim=4,
    )
    scaffold = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    lengths = torch.tensor([5])
    tau = torch.linspace(0, 1, 5).view(1, 5, 1)
    code = model.initial_code(scaffold, mask, lengths)
    unconditioned = model(tau, scaffold, code, mask=mask)
    conditioned = model(tau, scaffold, code, mask=mask, condition=torch.ones(1, 5, 4))
    assert unconditioned.shape == conditioned.shape == (1, 5, 16)
    with np.testing.assert_raises(ValueError):
        model(tau, scaffold, code, mask=mask, condition=torch.ones(1, 5, 3))


def test_inner_loop_adaptation_changes_only_code():
    model = _model()
    params_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    batch, frames, pose_dim = 1, 6, 16
    scaffold = torch.zeros(batch, frames, pose_dim)
    target = torch.randn(batch, frames, pose_dim)
    target_residual = target - scaffold
    mask = torch.ones(batch, frames, dtype=torch.bool)
    tau = torch.linspace(0, 1, frames).view(1, frames, 1)
    code = torch.zeros(batch, 8, requires_grad=True)
    cfg = {
        "meta": {"inner_steps": 2, "inner_lr": 0.5, "first_order": True},
        "meta_loss": {"inner_lambda_residual": 1.0, "inner_lambda_pose": 0.0},
        "loss": {"hand_weight": 1.0},
    }

    adapted, _losses = adapt_code(model, code, tau, scaffold, target, target_residual, mask, cfg)

    assert not torch.allclose(adapted.detach(), code.detach())
    for name, value in model.named_parameters():
        assert torch.allclose(value.detach(), params_before[name])
        assert value.grad is None


def test_support_query_masks_do_not_overlap():
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 7:] = False
    anchors = torch.zeros_like(valid)
    anchors[:, 0] = True
    anchors[0, -1] = True
    support, query = build_support_query_masks(valid, anchors, support_mode="stride", support_stride=4)

    assert not torch.any(support & query)
    assert torch.all((support | query) == valid)
    assert torch.all(support[anchors])


def test_save_meta_npz_contains_required_visualization_keys(tmp_path):
    frames = 3
    axis = np.zeros((frames, 133), dtype=np.float32)
    smplx = np.zeros((frames, 182), dtype=np.float32)
    rot6d = np.zeros((frames, 256), dtype=np.float32)
    meta = {"name": "sample", "text": "hello", "gloss": "HELLO", "split": "train", "source_index": 0}
    path = tmp_path / "sample_0000.npz"

    save_meta_npz(
        path,
        axis,
        smplx,
        rot6d,
        meta,
        "meta_adapted",
        extra={
            "coarse_smplx": smplx,
            "meta_prior_smplx": smplx,
            "meta_adapted_smplx": smplx,
        },
    )

    with np.load(path) as data:
        assert "smplx" in data
        assert "coarse_smplx" in data
        assert "meta_prior_smplx" in data
        assert "meta_adapted_smplx" in data
