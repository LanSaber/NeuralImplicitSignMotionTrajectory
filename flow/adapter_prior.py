from contextlib import nullcontext
from pathlib import Path

import torch

from flow.content_style_adapter import build_adapter_from_config
from flow.latent_codec import load_checkpoint
from flow.residual_prior import WordMotionPrior
from flow.temporal_word_attention import WordCandidateBuilder, build_arranger_from_config
from flow.text_encoder import FrozenT5TextEncoder


RETRIEVAL_FEATURE_NAMES = (
    "retrieval_confidence",
    "lexical_max_attention",
    "lexical_attention_mass",
    "null_attention_mass",
    "attention_change",
    "lexical_coverage",
    "candidate_gate_mean",
)


def _freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _path_equal(left, right):
    if left is None or right is None:
        return True
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def _latent_stats_from_checkpoint(checkpoint, device):
    stats = (checkpoint.get("latent_config") or {}).get("stats") or {}
    if "mean" not in stats or "std" not in stats:
        raise ValueError("Adapter checkpoint does not contain latent_config.stats.")
    return {
        "mean": torch.as_tensor(stats["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(stats["std"], dtype=torch.float32, device=device),
    }


def _checkpoint_adapter_enabled(checkpoint):
    def as_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    if "adapter_enabled" in checkpoint:
        return as_bool(checkpoint["adapter_enabled"], default=True)
    args = checkpoint.get("args") or {}
    if "disable_adapter" in args:
        return not as_bool(args["disable_adapter"], default=False)
    return True


def _encode_word_candidates(candidate_batch, latent_codec, latent_stats):
    word_motion = candidate_batch.motion
    word_mask = candidate_batch.frame_mask
    batch_size, num_candidates, word_frames, dim = word_motion.shape
    flat_motion = word_motion.reshape(batch_size * num_candidates, word_frames, dim)
    flat_mask = word_mask.reshape(batch_size * num_candidates, word_frames)
    flat_candidate_mask = candidate_batch.candidate_mask.reshape(batch_size * num_candidates)
    if not bool(flat_candidate_mask.all().item()):
        flat_motion = flat_motion.clone()
        flat_mask = flat_mask.clone()
        flat_mask[~flat_candidate_mask, 0] = True
    word_z_raw, word_z_mask = latent_codec.encode(flat_motion, mask=flat_mask)
    word_z = latent_codec.normalize_latent(word_z_raw, latent_stats)
    word_z = word_z.reshape(batch_size, num_candidates, word_z.shape[1], word_z.shape[2])
    word_z_mask = word_z_mask.reshape(batch_size, num_candidates, word_z_mask.shape[1])
    word_z = word_z * candidate_batch.candidate_mask[:, :, None, None].to(
        device=word_z.device,
        dtype=word_z.dtype,
    )
    word_z_mask = word_z_mask & candidate_batch.candidate_mask[:, :, None]
    return word_z, word_z_mask


def _encode_word_text_features(text_encoder, candidate_texts, device, dtype):
    flat_texts = [text for row in candidate_texts for text in row]
    word_text = text_encoder.encode(flat_texts).to(device=device, dtype=dtype)
    batch_size = len(candidate_texts)
    num_candidates = len(candidate_texts[0]) if candidate_texts else 0
    return word_text.reshape(batch_size, num_candidates, -1)


def _profile_section(profiler, name):
    return profiler.section(name) if profiler is not None else nullcontext()


def summarize_arranger_retrieval(arranger_out, candidate_batch, word_latent_mask):
    """Compress SoftArranger attention into inference-time confidence evidence."""

    attention = arranger_out["attention"]
    null_attention = arranger_out["null_attention"]
    batch, target_len = attention.shape[:2]
    flat_attention = attention.reshape(batch, target_len, -1)
    lexical_mass = flat_attention.sum(dim=-1).clamp(0.0, 1.0)
    lexical_max = (
        flat_attention.max(dim=-1).values
        if flat_attention.shape[-1]
        else lexical_mass.new_zeros(lexical_mass.shape)
    )

    token_mask = (
        word_latent_mask.to(device=attention.device, dtype=torch.bool)
        & candidate_batch.candidate_mask.to(device=attention.device, dtype=torch.bool)[:, :, None]
    ).reshape(batch, -1)
    support_count = token_mask.sum(dim=-1).to(attention.dtype) + 1.0
    probabilities = torch.cat([flat_attention, null_attention.unsqueeze(-1)], dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_denom = support_count.clamp_min(2.0).log().unsqueeze(1)
    concentration = (1.0 - entropy / entropy_denom).clamp(0.0, 1.0)
    retrieval_confidence = concentration * lexical_mass

    attention_change = flat_attention.new_zeros(batch, target_len)
    if target_len > 1:
        attention_change[:, 1:] = 0.5 * torch.abs(flat_attention[:, 1:] - flat_attention[:, :-1]).sum(dim=-1)

    coverage = (
        attention.new_tensor(
            [float(item.get("coverage", 0.0)) for item in candidate_batch.stats]
        )
        .unsqueeze(1)
        .expand(-1, target_len)
    )
    candidate_mask = candidate_batch.candidate_mask.to(device=attention.device, dtype=attention.dtype)
    gate_probs = arranger_out["word_gate_probs"].to(dtype=attention.dtype)
    gate_mean = (gate_probs * candidate_mask).sum(dim=-1) / candidate_mask.sum(dim=-1).clamp_min(1.0)
    gate_mean = gate_mean.unsqueeze(1).expand(-1, target_len)

    features = torch.stack(
        [
            retrieval_confidence,
            lexical_max,
            lexical_mass,
            null_attention,
            attention_change,
            coverage,
            gate_mean,
        ],
        dim=-1,
    )
    return features, RETRIEVAL_FEATURE_NAMES


class FrozenAdapterPrior:
    """Frozen soft-arranger/content-style adapter used as a latent flow source."""

    def __init__(
        self,
        checkpoint_path,
        checkpoint,
        adapter,
        arranger,
        prior_builder,
        candidate_builder,
        text_encoder,
        latent_stats,
        latent_codec,
        prior_mode,
        adapter_enabled=True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = checkpoint
        self.adapter = adapter
        self.arranger = arranger
        self.prior_builder = prior_builder
        self.candidate_builder = candidate_builder
        self.text_encoder = text_encoder
        self.latent_stats = latent_stats
        self.latent_codec = latent_codec
        self.prior_mode = str(prior_mode)
        self.adapter_enabled = bool(adapter_enabled)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        device,
        latent_codec,
        target_mean,
        target_std,
        word_data_dir=None,
        word_split=None,
        text_model_path=None,
        max_text_tokens=None,
        candidate_seed=42,
        lazy_word_motions=False,
        num_negative_candidates=None,
        prior_mode_override=None,
        adapter_enabled_override=None,
    ):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        latent_cfg = checkpoint.get("latent_config") or {}
        vae_checkpoint = latent_cfg.get("vae_checkpoint")
        if vae_checkpoint and not _path_equal(vae_checkpoint, latent_codec.checkpoint_path):
            raise ValueError(
                "Adapter checkpoint VAE does not match the active flow VAE: "
                f"adapter={vae_checkpoint}, flow={latent_codec.checkpoint_path}"
            )
        rotation_rep = latent_cfg.get("rotation_rep") or checkpoint.get("data_config", {}).get("rotation_rep")
        if rotation_rep and str(rotation_rep) != str(latent_codec.rotation_rep):
            raise ValueError(
                "Adapter checkpoint rotation representation does not match flow VAE: "
                f"adapter={rotation_rep}, flow={latent_codec.rotation_rep}"
            )

        if adapter_enabled_override is None:
            adapter_enabled = _checkpoint_adapter_enabled(checkpoint)
        else:
            adapter_enabled = bool(adapter_enabled_override)
        adapter = None
        if adapter_enabled:
            adapter = build_adapter_from_config(checkpoint["model_config"]).to(device)
            adapter.load_state_dict(checkpoint["model"], strict=True)
            adapter = _freeze(adapter)

        prior_mode = str(
            prior_mode_override
            or checkpoint.get("prior_mode")
            or checkpoint.get("args", {}).get("prior_mode", "concat")
        )
        data_cfg = checkpoint.get("data_config") or {}
        resolved_word_data_dir = word_data_dir or data_cfg.get("word_data_dir")
        if resolved_word_data_dir is None or not str(resolved_word_data_dir).strip():
            raise ValueError("Adapter prior needs a word_data_dir; pass --word_data_dir or use a checkpoint that stores it.")
        word_data_dir = Path(resolved_word_data_dir)
        word_split = str(word_split or data_cfg.get("word_split", "train"))
        prior_builder = WordMotionPrior(
            word_data_dir,
            split=word_split,
            target_mean=target_mean,
            target_std=target_std,
            rotation_rep=latent_codec.rotation_rep,
            lazy_motions=lazy_word_motions,
        )

        arranger = None
        candidate_builder = None
        text_encoder = None
        text_cfg = checkpoint.get("text_config") or {}
        if prior_mode == "soft_arranger":
            if "arranger_config" not in checkpoint or "arranger_model" not in checkpoint:
                raise ValueError("Soft-arranger adapter checkpoint is missing arranger_config or arranger_model.")
            arranger = build_arranger_from_config(checkpoint["arranger_config"]).to(device)
            arranger.load_state_dict(checkpoint["arranger_model"], strict=True)
            arranger = _freeze(arranger)
            candidate_cfg = checkpoint.get("candidate_config") or {}
            default_num_negative_candidates = int(candidate_cfg.get("num_negative_candidates", 16))
            if num_negative_candidates is None:
                num_negative_candidates = default_num_negative_candidates
            candidate_builder = WordCandidateBuilder(
                prior_builder,
                num_word_candidates=int(candidate_cfg.get("num_word_candidates", 32)),
                num_negative_candidates=int(num_negative_candidates),
                candidate_selection=str(candidate_cfg.get("candidate_selection", "flat")),
                max_positive_variants_per_key=int(candidate_cfg.get("max_positive_variants_per_key", 0)),
                seed=int(candidate_seed),
            )
            text_model_path = Path(text_model_path or text_cfg.get("text_model_path", "deps/flan-t5-base"))
            max_text_tokens = int(max_text_tokens or text_cfg.get("max_text_tokens", 64))
            text_encoder = FrozenT5TextEncoder(
                text_model_path,
                device=device,
                max_length=max_text_tokens,
                local_files_only=True,
                cache=True,
            )
        elif prior_mode != "concat":
            raise ValueError(f"Unsupported adapter prior_mode: {prior_mode}")

        latent_stats = _latent_stats_from_checkpoint(checkpoint, device)
        return cls(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            adapter=adapter,
            arranger=arranger,
            prior_builder=prior_builder,
            candidate_builder=candidate_builder,
            text_encoder=text_encoder,
            latent_stats=latent_stats,
            latent_codec=latent_codec,
            prior_mode=prior_mode,
            adapter_enabled=adapter_enabled,
        )

    @torch.no_grad()
    def build_prior(self, texts, raw_x, raw_mask, lengths, profiler=None, shuffle_candidates=False):
        device = raw_x.device
        dtype = raw_x.dtype
        latent_mask = self.latent_codec.latent_mask(raw_mask)

        retrieval_features = None
        retrieval_feature_names = ()
        if self.prior_mode == "soft_arranger":
            max_word_frames = min(
                int(self.latent_codec.max_frames),
                int(self.arranger.max_word_latent_frames) * int(self.latent_codec.downsample_factor),
            )
            with _profile_section(profiler, "adapter_candidate_build_ms"):
                candidate_batch = self.candidate_builder.batch(
                    texts,
                    device=device,
                    dtype=dtype,
                    shuffle=bool(shuffle_candidates),
                    max_motion_frames=max_word_frames,
                )
            with _profile_section(profiler, "adapter_word_vae_encode_ms"):
                word_latents, word_latent_mask = _encode_word_candidates(
                    candidate_batch,
                    self.latent_codec,
                    self.latent_stats,
                )
            with _profile_section(profiler, "adapter_text_encoder_ms"):
                sentence_text = self.text_encoder.encode(texts).to(device=device, dtype=dtype)
                word_text = _encode_word_text_features(self.text_encoder, candidate_batch.texts, device, dtype)
            with _profile_section(profiler, "soft_arranger_ms"):
                arranger_out = self.arranger(
                    sentence_text,
                    word_text,
                    word_latents,
                    word_latent_mask,
                    candidate_batch.candidate_mask,
                    latent_mask,
                )
            z_source = arranger_out["z_prior_aligned"]
            retrieval_features, retrieval_feature_names = summarize_arranger_retrieval(
                arranger_out,
                candidate_batch,
                word_latent_mask,
            )
            stats = candidate_batch.stats
        else:
            with _profile_section(profiler, "adapter_concat_prior_build_ms"):
                prior_raw, stats = self.prior_builder.batch(
                    texts,
                    lengths,
                    max_len=raw_x.shape[1],
                    device=device,
                    dtype=dtype,
                )
            prior_raw = prior_raw * raw_mask.to(device=device, dtype=dtype).unsqueeze(-1)
            with _profile_section(profiler, "adapter_concat_vae_encode_ms"):
                z_source_raw, _ = self.latent_codec.encode(prior_raw, mask=raw_mask)
                z_source = self.latent_codec.normalize_latent(z_source_raw, self.latent_stats)

        if self.adapter_enabled:
            with _profile_section(profiler, "content_style_adapter_ms"):
                z_adapt = self.adapter(z_source, mask=latent_mask)["z_adapt"]
        else:
            z_adapt = z_source * latent_mask.to(device=device, dtype=z_source.dtype).unsqueeze(-1)
        z_adapt = z_adapt * latent_mask.to(device=device, dtype=z_adapt.dtype).unsqueeze(-1)
        with _profile_section(profiler, "adapter_prior_vae_decode_ms"):
            decoded = self.latent_codec.decode(
                self.latent_codec.denormalize_latent(z_adapt, self.latent_stats),
                target_length=raw_x.shape[1],
                mask=raw_mask,
                latent_mask=latent_mask,
            )
        decoded = decoded * raw_mask.to(device=device, dtype=decoded.dtype).unsqueeze(-1)
        output = {
            "z_prior": z_adapt,
            "prior_raw": decoded,
            "stats": stats,
        }
        if retrieval_features is not None:
            output["retrieval_features_latent"] = retrieval_features
            output["retrieval_feature_names"] = retrieval_feature_names
        return output

    def checkpoint_config(self):
        return {
            "adapter_checkpoint": str(self.checkpoint_path),
            "adapter_prior_mode": self.prior_mode,
            "adapter_enabled": self.adapter_enabled,
            "candidate_config": dict(self.checkpoint.get("candidate_config") or {}),
            "text_config": dict(self.checkpoint.get("text_config") or {}),
            "data_config": dict(self.checkpoint.get("data_config") or {}),
            "frozen": True,
        }
