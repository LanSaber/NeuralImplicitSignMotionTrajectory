import torch
from torch import nn


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(x, scale=1.0):
    return GradientReverseFunction.apply(x, float(scale))


def masked_mean(x, mask=None, dim=1, eps=1e-8):
    """Mean-pool a temporal tensor while ignoring padded frames."""

    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    total = (x * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(float(eps))
    return total / denom


class ResidualMLPHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout=0.0, zero_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class ContentStyleAdapter(nn.Module):
    """Adapt word-concat VAE latents toward sentence latents.

    The adapter is intentionally residual: z_adapt = z_in + delta. The delta
    head is zero-initialized so training starts from the raw word prior.
    """

    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        content_dim=256,
        style_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.0,
        max_frames=100,
        num_domains=2,
        use_content_domain_classifier=False,
        gradient_reversal_lambda=1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.max_frames = int(max_frames)
        self.num_domains = int(num_domains)
        self.use_content_domain_classifier = bool(use_content_domain_classifier)
        self.gradient_reversal_lambda = float(gradient_reversal_lambda)

        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.delta_head = ResidualMLPHead(
            self.hidden_dim,
            self.latent_dim,
            dropout=self.dropout,
            zero_init=True,
        )
        self.content_head = ResidualMLPHead(
            self.hidden_dim,
            self.content_dim,
            dropout=self.dropout,
            zero_init=False,
        )
        self.style_head = ResidualMLPHead(
            self.hidden_dim,
            self.style_dim,
            dropout=self.dropout,
            zero_init=False,
        )
        self.style_classifier = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.style_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.style_dim, self.num_domains),
        )
        self.content_domain_classifier = None
        if self.use_content_domain_classifier:
            self.content_domain_classifier = nn.Sequential(
                nn.LayerNorm(self.content_dim),
                nn.Linear(self.content_dim, self.content_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.content_dim, self.num_domains),
            )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, z, mask=None):
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B, T, D], got {tuple(z.shape)}")
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z.shape[-1]}")
        if z.shape[1] > self.max_frames:
            raise ValueError(f"Latent length {z.shape[1]} exceeds max_frames={self.max_frames}")

        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        else:
            mask = mask.to(device=z.device, dtype=torch.bool)
            if mask.shape != z.shape[:2]:
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match z shape {tuple(z.shape[:2])}")

        h = self.input_proj(z)
        h = h + self.pos_embed[:, : z.shape[1]]
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h)

        delta = self.delta_head(h)
        valid = mask.unsqueeze(-1).to(dtype=delta.dtype)
        delta = delta * valid
        z_adapt = (z + delta) * valid

        content_tokens = self.content_head(h) * valid
        style_tokens = self.style_head(h) * valid
        content_pooled = masked_mean(content_tokens, mask=mask)
        style_pooled = masked_mean(style_tokens, mask=mask)
        style_logits = self.style_classifier(style_pooled)

        out = {
            "z_adapt": z_adapt,
            "delta": delta,
            "content_tokens": content_tokens,
            "content_pooled": content_pooled,
            "style_tokens": style_tokens,
            "style_pooled": style_pooled,
            "style_logits": style_logits,
        }
        if self.content_domain_classifier is not None:
            content_adv = gradient_reverse(content_pooled, self.gradient_reversal_lambda)
            out["content_domain_logits"] = self.content_domain_classifier(content_adv)
        return out


def build_adapter_from_config(config):
    return ContentStyleAdapter(**dict(config))
