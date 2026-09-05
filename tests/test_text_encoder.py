from unittest.mock import Mock, patch

import torch

from flow.text_encoder import FrozenT5TextEncoder


def test_frozen_text_encoder_preserves_legacy_loader_and_reports_identity():
    tokenizer = Mock()
    tokenizer.is_fast = True
    model = Mock()
    model.config.d_model = 768
    model.config.model_type = "mt5"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
        patch("transformers.T5EncoderModel.from_pretrained", return_value=model) as load_t5,
    ):
        encoder = FrozenT5TextEncoder(
            "deps/mt5-base",
            device="cpu",
            max_length=96,
            local_files_only=True,
        )

    load_tokenizer.assert_called_once_with(
        encoder.model_path,
        local_files_only=True,
    )
    load_t5.assert_called_once_with(
        encoder.model_path,
        local_files_only=True,
    )
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()
    model.requires_grad_.assert_called_once_with(False)
    assert encoder.tokenizer is tokenizer
    assert encoder.checkpoint_identity() == {
        "model_type": "mt5",
        "model_path": "deps/mt5-base",
        "text_dim": 768,
        "max_length": 96,
        "encoder_loader": "T5EncoderModel",
        "tokenizer_loader": "AutoTokenizer",
        "tokenizer_use_fast": "transformers_default",
        "encoder_class": "Mock",
        "tokenizer_class": "Mock",
        "tokenizer_is_fast": True,
    }
