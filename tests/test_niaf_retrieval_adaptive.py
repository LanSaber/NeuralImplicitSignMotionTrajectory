import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from flow.adapter_prior import RETRIEVAL_FEATURE_NAMES, summarize_arranger_retrieval
from flow.smplx_features import rotation_6d_to_matrix
from NIAF.retrieval_confidence_field.models.retrieval_adaptive import (
    ARTICULATOR_NAMES,
    RetrievalConfidenceAdaptiveField,
    confidence_target_from_error,
    target_tangent_correction,
)
from NIAF.retrieval_confidence_field.scripts.train_retrieval_adaptive_field import (
    validate_train_only_retrieval_bank,
)


def _model():
    torch.manual_seed(23)
    return RetrievalConfidenceAdaptiveField(
        text_dim=24,
        code_dim=8,
        context_hidden_dim=32,
        context_layers=1,
        context_heads=4,
        code_strides=(2, 4, 8),
        hidden_dim=32,
        depth=2,
        time_fourier_bands=2,
        context_time_fourier_bands=2,
        confidence_hidden_dim=24,
        confidence_text_dim=12,
        confidence_time_fourier_bands=2,
        duration_hidden_dim=16,
        dropout=0.0,
    )


def _inputs():
    batch, frames = 2, 9
    lengths = torch.tensor([9, 6])
    mask = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rotations = identity_6d.repeat(41)
    compact = torch.cat([rotations, torch.zeros(10)])
    scaffold = compact.view(1, 1, -1).expand(batch, frames, -1).clone()
    scaffold = scaffold * mask.unsqueeze(-1)
    tau = torch.zeros(batch, frames, 1)
    for index, length in enumerate(lengths.tolist()):
        tau[index, :length, 0] = torch.linspace(0.0, 1.0, length)
    text = torch.randn(batch, 5, 24)
    text_mask = torch.ones(batch, 5, dtype=torch.bool)
    text_mask[1, -1] = False
    evidence = torch.rand(batch, frames, len(RETRIEVAL_FEATURE_NAMES)) * mask.unsqueeze(-1)
    return scaffold, mask, lengths, tau, text, text_mask, evidence


def test_retrieval_summary_distinguishes_concentrated_attention():
    attention = torch.tensor(
        [
            [
                [[0.8, 0.0], [0.0, 0.0]],
                [[0.2, 0.2], [0.2, 0.2]],
            ]
        ]
    )
    arranger_out = {
        "attention": attention,
        "null_attention": torch.full((1, 2), 0.2),
        "word_gate_probs": torch.tensor([[0.75, 0.25]]),
    }
    candidates = SimpleNamespace(
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        stats=[{"coverage": 0.5}],
    )
    word_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    features, names = summarize_arranger_retrieval(arranger_out, candidates, word_mask)

    assert names == RETRIEVAL_FEATURE_NAMES
    assert features.shape == (1, 2, len(RETRIEVAL_FEATURE_NAMES))
    assert features[0, 0, 0] > features[0, 1, 0]
    assert torch.allclose(features[0, :, 5], torch.full((2,), 0.5))


def test_retrieval_adaptive_forward_shapes_and_valid_rotations():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    outputs = model(tau, scaffold, mask, text, evidence, text_mask=text_mask)

    assert outputs["prediction"].shape == scaffold.shape
    assert outputs["correction_axis"].shape == (2, 9, 133)
    assert outputs["confidence"].shape == (2, 9, len(ARTICULATOR_NAMES))
    assert outputs["scale_weights"].shape == (2, 9, len(ARTICULATOR_NAMES), 3)
    assert outputs["frame_codes"].shape == (2, 9, len(ARTICULATOR_NAMES), 8)
    assert torch.all(outputs["prediction"][1, 6:] == 0)
    valid_scale_weights = outputs["scale_weights"][mask]
    assert torch.allclose(
        valid_scale_weights.sum(dim=-1),
        torch.ones_like(valid_scale_weights[..., 0]),
        atol=1e-6,
    )

    matrices = rotation_6d_to_matrix(
        outputs["prediction"][mask][..., :246].reshape(-1, 41, 6)
    )
    determinants = torch.linalg.det(matrices)
    assert torch.allclose(determinants, torch.ones_like(determinants), atol=1e-4)


def test_high_confidence_suppresses_corrections_and_coarsens_scale():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    low = torch.zeros(scaffold.shape[0], scaffold.shape[1], len(ARTICULATOR_NAMES))
    high = torch.ones_like(low)
    low_outputs = model(
        tau,
        scaffold,
        mask,
        text,
        evidence,
        text_mask=text_mask,
        confidence_override=low,
    )
    high_outputs = model(
        tau,
        scaffold,
        mask,
        text,
        evidence,
        text_mask=text_mask,
        confidence_override=high,
    )

    assert high_outputs["gates"][mask].mean() < low_outputs["gates"][mask].mean()
    assert (
        high_outputs["scale_weights"][mask][..., -1].mean()
        > low_outputs["scale_weights"][mask][..., -1].mean()
    )


def test_tangent_target_and_confidence_target_are_well_behaved():
    scaffold, mask, _lengths, _tau, _text, _text_mask, _evidence = _inputs()
    correction = target_tangent_correction(scaffold, scaffold, mask)
    assert correction.shape == (2, 9, 133)
    assert torch.allclose(correction, torch.zeros_like(correction), atol=1e-6)

    errors = torch.tensor([[[0.0, 0.1, 0.5, 1.0]]])
    target = confidence_target_from_error(errors, [0.5] * 4, torch.ones(1, 1, dtype=torch.bool))
    assert torch.all(target[:, :, :-1] > target[:, :, 1:])


def test_forward_backward_updates_field_parameters():
    model = _model()
    scaffold, mask, _lengths, tau, text, text_mask, evidence = _inputs()
    outputs = model(tau, scaffold, mask, text, evidence, text_mask=text_mask)
    loss = outputs["prediction"][mask].square().mean() + outputs["confidence"][mask].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.any(gradient != 0) for gradient in gradients)


def test_train_only_retrieval_manifest_is_enforced():
    word_dir = "/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc"
    manifest = f"{word_dir}/meta/manifest_train.balanced.jsonl"
    prior_builder = SimpleNamespace(
        manifest_path=manifest,
        entries=[1, 2, 3],
        entries_by_key={"A": [1], "B": [2, 3]},
    )
    provider = SimpleNamespace(
        adapter_prior=SimpleNamespace(prior_builder=prior_builder),
    )
    summary = validate_train_only_retrieval_bank(
        {
            "adapter": {"word_data_dir": word_dir},
            "retrieval": {"require_train_only_bank": True},
        },
        provider,
    )
    assert summary["manifest"] == manifest
    assert summary["entries"] == 3
    assert summary["lexicon_keys"] == 2


def test_explicit_external_retrieval_manifest_is_enforced():
    with tempfile.TemporaryDirectory() as temporary:
        word_dir = Path(temporary)
        manifest = word_dir / "meta" / "manifest_all.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            '{"lexicon_key": "HELLO"}\n{"lexicon_key": "THANK-YOU"}\n',
            encoding="utf-8",
        )
        prior_builder = SimpleNamespace(
            manifest_path=manifest,
            entries=[1, 2],
            entries_by_key={"HELLO": [1], "THANK-YOU": [2]},
        )
        provider = SimpleNamespace(
            adapter_prior=SimpleNamespace(prior_builder=prior_builder),
        )
        summary = validate_train_only_retrieval_bank(
            {
                "adapter": {"word_data_dir": str(word_dir)},
                "retrieval": {
                    "require_train_only_bank": False,
                    "expected_word_split": "all",
                },
            },
            provider,
        )
        assert summary["manifest"] == str(manifest)
        assert summary["entries"] == 2
        assert summary["lexicon_keys"] == 2


def test_cache_only_split_audit_uses_cache_summary():
    word_dir = "/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc"
    with tempfile.TemporaryDirectory() as temporary:
        cache_dir = Path(temporary)
        (cache_dir / "cache_summary.json").write_text(
            json.dumps(
                {
                    "word_split": "train.balanced",
                    "word_manifest": (
                        f"{word_dir}/meta/manifest_train.balanced.jsonl"
                    ),
                    "require_retrieval_features": True,
                    "retrieval_feature_names": list(RETRIEVAL_FEATURE_NAMES),
                }
            ),
            encoding="utf-8",
        )
        provider = SimpleNamespace(
            adapter_prior=None,
            cache_only=True,
            scaffold_cache_dir=cache_dir,
        )
        summary = validate_train_only_retrieval_bank(
            {
                "adapter": {
                    "word_data_dir": word_dir,
                    "word_split": "train.balanced",
                },
                "retrieval": {"require_train_only_bank": True},
            },
            provider,
        )
        assert summary["entries"] == 29323
        assert summary["lexicon_keys"] == 1085
        assert summary["cache_summary"] == str(cache_dir / "cache_summary.json")


def test_legacy_model_import_forwards_to_new_package():
    from NIAF.continuous_sign_field.models.retrieval_adaptive import (
        RetrievalConfidenceAdaptiveField as LegacyField,
    )

    assert LegacyField is RetrievalConfidenceAdaptiveField
