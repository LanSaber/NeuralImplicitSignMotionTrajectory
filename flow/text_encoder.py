from pathlib import Path

import torch


class FrozenT5TextEncoder:
    """Frozen local T5 encoder with pooled and token-level embeddings.

    The loader deliberately retains the historical ``T5EncoderModel`` and
    default ``AutoTokenizer`` path. Existing SignTrajField checkpoints were
    trained through that frontend, so changing either class would invalidate
    full-pipeline text-only parity even when all trajectory weights match.
    """

    def __init__(self, model_path, device, max_length=64, local_files_only=True, cache=True):
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as exc:
            raise ImportError("Text-conditioned flow requires the transformers package.") from exc

        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.cache = bool(cache)
        self._pool_cache = {}
        self._token_cache = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
        )
        self.model = T5EncoderModel.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model_type = str(getattr(self.model.config, "model_type", "t5")).lower()
        self.text_dim = int(self.model.config.d_model)

    def checkpoint_identity(self):
        """Return the semantic text-encoder contract persisted with checkpoints."""

        return {
            "model_type": self.model_type,
            "model_path": self.model_path.as_posix(),
            "text_dim": self.text_dim,
            "max_length": self.max_length,
            "encoder_loader": "T5EncoderModel",
            "tokenizer_loader": "AutoTokenizer",
            "tokenizer_use_fast": "transformers_default",
            "encoder_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(self.tokenizer, "is_fast", False)),
        }

    @torch.no_grad()
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [text if text is not None else "" for text in texts]
        if self.cache:
            missing = list(dict.fromkeys(text for text in texts if text not in self._pool_cache))
            if missing:
                encoded = self._encode_uncached(missing)
                for text, emb in zip(missing, encoded):
                    self._pool_cache[text] = emb.detach().cpu()
            return torch.stack([self._pool_cache[text] for text in texts], dim=0).to(self.device)
        return self._encode_uncached(texts)

    @torch.no_grad()
    def encode_tokens(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [text if text is not None else "" for text in texts]
        if not self.cache:
            return self._encode_tokens_uncached(texts)

        missing = list(dict.fromkeys(text for text in texts if text not in self._token_cache))
        if missing:
            hidden, attention_mask = self._encode_tokens_uncached(missing)
            lengths = attention_mask.sum(dim=1).detach().cpu().long().tolist()
            for text, token_hidden, length in zip(missing, hidden.detach().cpu(), lengths):
                length = max(int(length), 1)
                self._token_cache[text] = token_hidden[:length].contiguous()

        lengths = [self._token_cache[text].shape[0] for text in texts]
        max_len = max(lengths) if lengths else 1
        tokens = torch.zeros(len(texts), max_len, self.text_dim, dtype=torch.float32)
        mask = torch.zeros(len(texts), max_len, dtype=torch.bool)
        for idx, text in enumerate(texts):
            cached = self._token_cache[text]
            length = cached.shape[0]
            tokens[idx, :length] = cached
            mask[idx, :length] = True
        return tokens.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def _encode_uncached(self, texts):
        hidden, attention_mask = self._encode_tokens_uncached(texts)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.float()

    @torch.no_grad()
    def _encode_tokens_uncached(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.model(**encoded).last_hidden_state
        return hidden.float(), encoded["attention_mask"].bool()
