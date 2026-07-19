import math

import torch
import torch.nn.functional as F
from torch import nn

from flow.smplx_features import (
    COMPACT6D_DIM,
    COMPACT6D_EXPRESSION,
    COMPACT6D_JAW,
    COMPACT6D_LEFT_HAND,
    COMPACT6D_RIGHT_HAND,
    COMPACT6D_UPPER_BODY,
    COMPACT_DIM,
    COMPACT_EXPRESSION,
    COMPACT_JAW,
    COMPACT_LEFT_HAND,
    COMPACT_RIGHT_HAND,
    COMPACT_UPPER_BODY,
)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DownsampleBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SMPLXOutputHeads(nn.Module):
    def __init__(self, hidden_dim, output_dim=COMPACT_DIM):
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim == COMPACT6D_DIM:
            upper_body = COMPACT6D_UPPER_BODY
            left_hand = COMPACT6D_LEFT_HAND
            right_hand = COMPACT6D_RIGHT_HAND
            jaw = COMPACT6D_JAW
            expression = COMPACT6D_EXPRESSION
        elif self.output_dim == COMPACT_DIM:
            upper_body = COMPACT_UPPER_BODY
            left_hand = COMPACT_LEFT_HAND
            right_hand = COMPACT_RIGHT_HAND
            jaw = COMPACT_JAW
            expression = COMPACT_EXPRESSION
        else:
            upper_body = left_hand = right_hand = jaw = expression = None

        if upper_body is None:
            self.generic = nn.Linear(hidden_dim, self.output_dim)
            self.upper_body = None
            self.left_hand = None
            self.right_hand = None
            self.jaw = None
            self.expression = None
        else:
            self.generic = None
            self.upper_body = nn.Linear(hidden_dim, upper_body.stop - upper_body.start)
            self.left_hand = nn.Linear(hidden_dim, left_hand.stop - left_hand.start)
            self.right_hand = nn.Linear(hidden_dim, right_hand.stop - right_hand.start)
            self.jaw = nn.Linear(hidden_dim, jaw.stop - jaw.start)
            self.expression = nn.Linear(hidden_dim, expression.stop - expression.start)

    def forward(self, h):
        if self.generic is not None:
            return self.generic(h)
        return torch.cat(
            [
                self.upper_body(h),
                self.left_hand(h),
                self.right_hand(h),
                self.jaw(h),
                self.expression(h),
            ],
            dim=-1,
        )


class TemporalSMPLXVAE(nn.Module):
    """High-fidelity temporal VAE for normalized compact SMPL-X sequences.

    The latent is a temporal sequence with length T / downsample_factor, rather
    than a single global vector, so it can preserve signing timing and hand detail.
    """

    def __init__(
        self,
        input_dim=COMPACT_DIM,
        hidden_dim=512,
        latent_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.0,
        max_frames=400,
        downsample_factor=4,
        logvar_min=-10.0,
        logvar_max=10.0,
    ):
        super().__init__()
        if downsample_factor != 4:
            raise ValueError("TemporalSMPLXVAE currently expects downsample_factor=4.")
        if max_frames % downsample_factor != 0:
            raise ValueError("max_frames must be divisible by downsample_factor.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.max_frames = int(max_frames)
        self.downsample_factor = int(downsample_factor)
        self.max_latent_frames = self.max_frames // self.downsample_factor
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, self.hidden_dim))
        self.encoder_down = nn.Sequential(
            DownsampleBlock(self.hidden_dim, dropout=self.dropout),
            DownsampleBlock(self.hidden_dim, dropout=self.dropout),
        )
        self.encoder_latent_pos = nn.Parameter(torch.zeros(1, self.max_latent_frames, self.hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.encoder_norm = nn.LayerNorm(self.hidden_dim)
        self.mu_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.hidden_dim, self.latent_dim)

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_latent_pos = nn.Parameter(torch.zeros(1, self.max_latent_frames, self.hidden_dim))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(dec_layer, num_layers=self.num_layers)
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.decoder_up = nn.Sequential(
            UpsampleBlock(self.hidden_dim, dropout=self.dropout),
            UpsampleBlock(self.hidden_dim, dropout=self.dropout),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_heads = SMPLXOutputHeads(self.hidden_dim, output_dim=self.input_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.input_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_latent_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_latent_pos, mean=0.0, std=0.02)
        for module in self.output_heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def _pad_to_factor(self, x, mask=None):
        length = x.shape[1]
        pad = (self.downsample_factor - length % self.downsample_factor) % self.downsample_factor
        if pad == 0:
            return x, mask, length
        x = F.pad(x, (0, 0, 0, pad))
        if mask is not None:
            mask = F.pad(mask, (0, pad), value=False)
        return x, mask, length

    def _latent_mask(self, mask):
        if mask is None:
            return None
        batch, length = mask.shape
        latent_length = int(math.ceil(length / self.downsample_factor))
        padded = F.pad(
            mask,
            (0, latent_length * self.downsample_factor - length),
            value=False,
        )
        return padded.view(batch, latent_length, self.downsample_factor).any(dim=-1)

    def encode(self, x, mask=None):
        x, mask, _ = self._pad_to_factor(x, mask)
        if x.shape[1] > self.max_frames:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_frames={self.max_frames}")

        h = self.input_proj(x) + self.input_pos_embed[:, : x.shape[1]]
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)

        h = self.encoder_down(h.transpose(1, 2)).transpose(1, 2)
        latent_mask = self._latent_mask(mask)
        if h.shape[1] > self.max_latent_frames:
            raise ValueError(f"Latent length {h.shape[1]} exceeds max_latent_frames={self.max_latent_frames}")
        h = h + self.encoder_latent_pos[:, : h.shape[1]]
        key_padding_mask = ~latent_mask.bool() if latent_mask is not None else None
        h = self.encoder_transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.encoder_norm(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(self.logvar_min, self.logvar_max)
        if latent_mask is not None:
            keep = latent_mask.unsqueeze(-1).to(mu.dtype)
            mu = mu * keep
            logvar = logvar * keep
        return mu, logvar, latent_mask

    def reparameterize(self, mu, logvar, sample=True):
        if not self.training and not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_length=None, mask=None, latent_mask=None):
        if z.shape[1] > self.max_latent_frames:
            raise ValueError(f"Latent length {z.shape[1]} exceeds max_latent_frames={self.max_latent_frames}")
        h = self.latent_proj(z) + self.decoder_latent_pos[:, : z.shape[1]]
        key_padding_mask = ~latent_mask.bool() if latent_mask is not None else None
        h = self.decoder_transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.decoder_norm(h)
        h = self.decoder_up(h.transpose(1, 2)).transpose(1, 2)
        if target_length is not None:
            h = h[:, :target_length]
        h = self.output_norm(h)
        out = self.output_heads(h)
        if mask is not None:
            out = out * mask[:, : out.shape[1]].unsqueeze(-1).to(out.dtype)
        return out

    def forward(self, x, mask=None, sample=True):
        original_length = x.shape[1]
        mu, logvar, latent_mask = self.encode(x, mask=mask)
        z = self.reparameterize(mu, logvar, sample=sample)
        recon = self.decode(z, target_length=original_length, mask=mask, latent_mask=latent_mask)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "latent_mask": latent_mask,
        }
