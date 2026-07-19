import math

import torch

from NIAF.continuous_sign_field.models import LocalAmortizedImplicitResidualField


def _model():
    torch.manual_seed(17)
    return LocalAmortizedImplicitResidualField(
        pose_dim=256,
        text_dim=32,
        code_dim=16,
        context_hidden_dim=32,
        context_layers=1,
        context_heads=4,
        local_stride=4,
        hidden_dim=48,
        depth=2,
        time_fourier_bands=3,
        context_time_fourier_bands=2,
        duration_hidden_dim=24,
        duration_initial_frames=80,
    )


def _inputs():
    batch, frames = 2, 17
    lengths = torch.tensor([17, 11])
    mask = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    scaffold = torch.randn(batch, frames, 256) * mask.unsqueeze(-1)
    tau = torch.zeros(batch, frames, 1)
    for idx, length in enumerate(lengths.tolist()):
        tau[idx, :length, 0] = torch.linspace(0, 1, length)
    text = torch.randn(batch, 6, 32)
    text_mask = torch.ones(batch, 6, dtype=torch.bool)
    text_mask[1, -2:] = False
    return scaffold, mask, lengths, tau, text, text_mask


def test_local_implicit_forward_shapes_and_padding():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask = _inputs()
    outputs = model(tau, scaffold, mask, text, text_mask=text_mask)

    assert outputs["prediction"].shape == scaffold.shape
    assert outputs["residual"].shape == scaffold.shape
    assert outputs["local_codes"].shape == (2, math.ceil(scaffold.shape[1] / 4), 16)
    assert outputs["frame_codes"].shape == (2, scaffold.shape[1], 16)
    assert outputs["gates"].shape == (2, scaffold.shape[1], 4)
    assert torch.all(outputs["residual"][1, 11:] == 0)
    assert torch.all(outputs["local_codes"][1, 3:] == 0)


def test_duration_prediction_uses_text_only_and_quantizes():
    model = _model()
    _scaffold, _mask, _lengths, _tau, text, text_mask = _inputs()
    predicted = model.predict_lengths(text, text_mask=text_mask, min_frames=40, max_frames=400, multiple=4)

    assert predicted.shape == (2,)
    assert torch.all(predicted == 80)
    assert torch.all(predicted % 4 == 0)


def test_duration_loss_updates_duration_head():
    model = _model()
    _scaffold, _mask, lengths, _tau, text, text_mask = _inputs()
    pred_log_frames = model.predict_log_frames(text, text_mask=text_mask)
    loss = torch.nn.functional.smooth_l1_loss(pred_log_frames, torch.log(lengths.float()))
    loss.backward()

    grads = [parameter.grad for parameter in model.duration_head.parameters()]
    assert any(grad is not None and torch.any(grad != 0) for grad in grads)
