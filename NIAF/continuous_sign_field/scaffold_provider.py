from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flow.adapter_prior import FrozenAdapterPrior, RETRIEVAL_FEATURE_NAMES
from flow.latent_codec import LatentMotionCodec, load_checkpoint
from NIAF.continuous_sign_field.data import denormalize_motion
from NIAF.continuous_sign_field.scaffold import build_batch_scaffold


def conditioning_texts(batch, field):
    field = str(field or "text")
    if field == "gloss":
        return [gloss if gloss else text for text, gloss in zip(batch["text"], batch["gloss"])]
    if field == "text_gloss":
        return [
            f"{text} {gloss}".strip() if gloss else text
            for text, gloss in zip(batch["text"], batch["gloss"])
        ]
    return batch["text"]


def _resolve_adapter_vae_checkpoint(adapter_cfg):
    explicit = adapter_cfg.get("vae_checkpoint")
    if explicit:
        return Path(explicit)

    checkpoint_path = Path(adapter_cfg["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    latent_cfg = checkpoint.get("latent_config") or {}
    vae_checkpoint = latent_cfg.get("vae_checkpoint")
    if not vae_checkpoint:
        raise ValueError(
            "Adapter scaffold needs adapter.vae_checkpoint, or an adapter checkpoint "
            "with latent_config.vae_checkpoint."
        )
    return Path(vae_checkpoint)


def _checkpoint_condition_field(adapter_cfg):
    checkpoint_path = Path(adapter_cfg["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_cfg = checkpoint.get("data_config") or {}
    return data_cfg.get("condition_field")


class ScaffoldProvider:
    """Build the conditioning scaffold without hiding its source.

    `gt_anchors` is an oracle scaffold used only as an upper-bound baseline.
    `adapter_pred` builds a scaffold from the frozen adapter/word prior and
    never reads target poses except for batch shape, mask, and supervision loss.
    """

    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.scaffold_cfg = cfg.get("scaffold", {})
        self.adapter_cfg = cfg.get("adapter", {})
        self.source = str(self.scaffold_cfg.get("source", "gt_anchors"))
        self.device = torch.device(device)
        self.mean = torch.from_numpy(dataset.mean).to(device=self.device, dtype=torch.float32)
        self.std = torch.from_numpy(dataset.std).to(device=self.device, dtype=torch.float32)
        self.adapter_prior = None
        self.latent_codec = None
        self.adapter_condition_field = None
        cache_dir = self.scaffold_cfg.get("cache_dir")
        self.scaffold_cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_only = bool(self.scaffold_cfg.get("cache_only", False))
        self.prefer_cache = bool(self.scaffold_cfg.get("prefer_cache", True))
        self.require_retrieval_features = bool(self.scaffold_cfg.get("require_retrieval_features", False))

        if self.source in {"adapter_pred", "adapter_prior", "adapter"}:
            if not self.adapter_cfg.get("checkpoint"):
                raise ValueError("scaffold.source=adapter_pred requires adapter.checkpoint.")
            if self.cache_only:
                if self.scaffold_cache_dir is None:
                    raise ValueError("scaffold.cache_only=true requires scaffold.cache_dir.")
                self.adapter_condition_field = (
                    self.adapter_cfg.get("condition_field")
                    or _checkpoint_condition_field(self.adapter_cfg)
                    or self.cfg.get("text", {}).get("condition_field", "text")
                )
                return
            vae_checkpoint = _resolve_adapter_vae_checkpoint(self.adapter_cfg)
            self.latent_codec = LatentMotionCodec(vae_checkpoint, device=self.device)
            self.adapter_condition_field = (
                self.adapter_cfg.get("condition_field")
                or _checkpoint_condition_field(self.adapter_cfg)
                or self.cfg.get("text", {}).get("condition_field", "text")
            )
            self.adapter_prior = FrozenAdapterPrior.from_checkpoint(
                self.adapter_cfg["checkpoint"],
                device=self.device,
                latent_codec=self.latent_codec,
                target_mean=dataset.mean,
                target_std=dataset.std,
                word_data_dir=self.adapter_cfg.get("word_data_dir") or self.cfg.get("data", {}).get("word_data_dir"),
                word_split=self.adapter_cfg.get("word_split"),
                text_model_path=self.adapter_cfg.get("text_model_path") or self.cfg.get("text", {}).get("model_path"),
                max_text_tokens=self.adapter_cfg.get("max_text_tokens") or self.cfg.get("text", {}).get("max_tokens"),
                candidate_seed=int(self.adapter_cfg.get("candidate_seed", self.cfg.get("seed", 1234))),
                lazy_word_motions=bool(self.adapter_cfg.get("lazy_word_motions", False)),
                num_negative_candidates=self.adapter_cfg.get("num_negative_candidates"),
                prior_mode_override=self.adapter_cfg.get("prior_mode_override"),
                adapter_enabled_override=self.adapter_cfg.get("adapter_enabled"),
            )
        elif self.source not in {"gt_anchors", "neutral"}:
            raise ValueError(f"Unsupported scaffold.source={self.source!r}")

    @property
    def config_summary(self):
        summary = {
            "source": self.source,
            "cache_dir": str(self.scaffold_cache_dir) if self.scaffold_cache_dir is not None else None,
            "cache_only": self.cache_only,
            "require_retrieval_features": self.require_retrieval_features,
        }
        if self.adapter_prior is not None:
            summary.update(self.adapter_prior.checkpoint_config())
            summary["condition_field"] = self.adapter_condition_field
            summary["shuffle_word_candidates"] = bool(self.adapter_cfg.get("shuffle_word_candidates", False))
            if self.adapter_prior.candidate_builder is not None:
                summary["effective_num_negative_candidates"] = int(
                    self.adapter_prior.candidate_builder.num_negative_candidates
                )
                summary["effective_num_word_candidates"] = int(
                    self.adapter_prior.candidate_builder.num_word_candidates
                )
        return summary

    def _cache_path(self, motion_path):
        if self.scaffold_cache_dir is None:
            return None
        return self.scaffold_cache_dir / str(motion_path)

    def _build_from_cache(self, batch, mask, dtype, return_metadata=False):
        if self.scaffold_cache_dir is None or not self.prefer_cache:
            return None, None, []
        batch_size, max_len = mask.shape
        scaffold = torch.zeros(
            batch_size,
            max_len,
            self.mean.numel(),
            device=self.device,
            dtype=dtype,
        )
        retrieval_features = scaffold.new_zeros(batch_size, max_len, len(RETRIEVAL_FEATURE_NAMES))
        feature_names = RETRIEVAL_FEATURE_NAMES
        missing = []
        for idx, (motion_path, length_tensor) in enumerate(zip(batch["motion_path"], batch["length"])):
            length = int(length_tensor.item())
            cache_path = self._cache_path(motion_path)
            if cache_path is None or not cache_path.is_file():
                missing.append(str(cache_path))
                continue
            with np.load(cache_path) as data:
                value = data["scaffold"].astype(np.float32)
                if len(value) != length:
                    missing.append(f"{cache_path} (cached={len(value)}, requested={length})")
                    continue
                scaffold[idx, :length] = torch.from_numpy(value).to(device=self.device, dtype=dtype)
                if return_metadata and "retrieval_features" in data:
                    cached_features = data["retrieval_features"].astype(np.float32)
                    if cached_features.shape != (length, len(RETRIEVAL_FEATURE_NAMES)):
                        missing.append(
                            f"{cache_path} (retrieval_features={cached_features.shape}, "
                            f"expected={(length, len(RETRIEVAL_FEATURE_NAMES))})"
                        )
                        continue
                    retrieval_features[idx, :length] = torch.from_numpy(cached_features).to(
                        device=self.device,
                        dtype=dtype,
                    )
                    if "retrieval_feature_names" in data:
                        cached_names = tuple(str(name) for name in data["retrieval_feature_names"].tolist())
                        if cached_names != RETRIEVAL_FEATURE_NAMES:
                            missing.append(f"{cache_path} (retrieval feature names do not match)")
                            continue
                        feature_names = cached_names
                elif return_metadata and self.require_retrieval_features:
                    missing.append(f"{cache_path} (missing retrieval_features)")
        if missing:
            return None, None, missing
        scaffold = scaffold * mask.unsqueeze(-1).to(dtype)
        metadata = None
        if return_metadata:
            metadata = {
                "retrieval_features": retrieval_features * mask.unsqueeze(-1).to(dtype),
                "retrieval_feature_names": feature_names,
                "from_cache": True,
            }
        return scaffold, metadata, []

    def _frame_retrieval_features(self, latent_features, lengths, max_len, dtype):
        output = torch.zeros(
            len(lengths),
            int(max_len),
            len(RETRIEVAL_FEATURE_NAMES),
            device=self.device,
            dtype=dtype,
        )
        for idx, length in enumerate(lengths):
            length = int(length)
            latent_len = self.latent_codec.latent_length(length)
            values = latent_features[idx, :latent_len].transpose(0, 1).unsqueeze(0)
            resized = F.interpolate(values, size=length, mode="linear", align_corners=False)
            output[idx, :length] = resized.squeeze(0).transpose(0, 1).to(dtype=dtype)
        return output

    @staticmethod
    def _return(scaffold, anchor_mask, metadata, return_metadata):
        if return_metadata:
            return scaffold, anchor_mask, metadata
        return scaffold, anchor_mask

    @torch.no_grad()
    def build(self, batch, x=None, use_cache=True, return_metadata=False):
        mask = batch["mask"].to(self.device)
        lengths = batch["length"].to(self.device)
        dtype = x.dtype if x is not None else torch.float32
        batch_size, max_len = mask.shape

        if self.source == "gt_anchors":
            if x is None:
                raise ValueError("GT-anchor scaffold requires denormalized target motion x.")
            scaffold, anchor_mask = build_batch_scaffold(
                x,
                lengths,
                stride=int(self.scaffold_cfg.get("anchor_stride", 4)),
                kind=self.scaffold_cfg.get("interpolation", "slerp"),
            )
            metadata = {
                "retrieval_features": scaffold.new_zeros(
                    batch_size,
                    max_len,
                    len(RETRIEVAL_FEATURE_NAMES),
                ),
                "retrieval_feature_names": RETRIEVAL_FEATURE_NAMES,
                "from_cache": False,
            }
            return self._return(scaffold, anchor_mask, metadata, return_metadata)

        if self.source == "neutral":
            scaffold = self.mean.view(1, 1, -1).to(dtype=dtype).expand(batch_size, max_len, -1).clone()
            scaffold = scaffold * mask.unsqueeze(-1).to(dtype)
            metadata = {
                "retrieval_features": scaffold.new_zeros(
                    batch_size,
                    max_len,
                    len(RETRIEVAL_FEATURE_NAMES),
                ),
                "retrieval_feature_names": RETRIEVAL_FEATURE_NAMES,
                "from_cache": False,
            }
            return self._return(
                scaffold,
                torch.zeros_like(mask, dtype=torch.bool),
                metadata,
                return_metadata,
            )

        if use_cache:
            cached, metadata, missing = self._build_from_cache(
                batch,
                mask,
                dtype,
                return_metadata=return_metadata,
            )
            if cached is not None:
                return self._return(
                    cached,
                    torch.zeros_like(mask, dtype=torch.bool),
                    metadata,
                    return_metadata,
                )
            if self.cache_only:
                preview = ", ".join(missing[:3])
                raise FileNotFoundError(f"Missing or incompatible adapter scaffold cache entries: {preview}")

        if self.latent_codec is None or self.adapter_prior is None:
            raise RuntimeError("Adapter scaffold provider was not initialized.")
        if max_len > int(self.latent_codec.max_frames):
            raise ValueError(
                f"Adapter scaffold max_len={max_len} exceeds VAE max_frames={self.latent_codec.max_frames}. "
                "Use a shorter data.max_frames or a VAE/adapter checkpoint trained for longer sequences."
            )

        texts = conditioning_texts(batch, self.adapter_condition_field)
        raw_x = torch.zeros(
            batch_size,
            max_len,
            self.mean.numel(),
            dtype=dtype,
            device=self.device,
        )
        prior_out = self.adapter_prior.build_prior(
            texts,
            raw_x,
            mask,
            lengths.detach().cpu().tolist(),
            shuffle_candidates=bool(self.adapter_cfg.get("shuffle_word_candidates", False)),
        )
        scaffold = denormalize_motion(prior_out["prior_raw"].to(dtype=dtype), self.mean, self.std)
        scaffold = scaffold * mask.unsqueeze(-1).to(dtype)
        latent_features = prior_out.get("retrieval_features_latent")
        if latent_features is None:
            if self.require_retrieval_features:
                raise RuntimeError("The active adapter prior did not return retrieval confidence features.")
            retrieval_features = scaffold.new_zeros(batch_size, max_len, len(RETRIEVAL_FEATURE_NAMES))
        else:
            names = tuple(prior_out.get("retrieval_feature_names") or ())
            if names != RETRIEVAL_FEATURE_NAMES:
                raise RuntimeError(f"Unexpected retrieval feature layout: {names}")
            retrieval_features = self._frame_retrieval_features(
                latent_features,
                lengths.detach().cpu().tolist(),
                max_len,
                dtype,
            )
            retrieval_features = retrieval_features * mask.unsqueeze(-1).to(dtype)
        metadata = {
            "retrieval_features": retrieval_features,
            "retrieval_feature_names": RETRIEVAL_FEATURE_NAMES,
            "retrieval_stats": prior_out.get("stats", []),
            "from_cache": False,
        }
        return self._return(
            scaffold,
            torch.zeros_like(mask, dtype=torch.bool),
            metadata,
            return_metadata,
        )

    @torch.no_grad()
    def build_with_metadata(self, batch, x=None, use_cache=True):
        return self.build(batch, x=x, use_cache=use_cache, return_metadata=True)
