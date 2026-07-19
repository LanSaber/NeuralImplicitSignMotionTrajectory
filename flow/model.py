import math

import torch
from torch import nn

from flow.smplx_features import COMPACT_DIM

MODEL_SIZE_PRESETS = {
    "small": {"hidden_dim": 256, "num_layers": 4, "num_heads": 4},
    "base": {"hidden_dim": 512, "num_layers": 8, "num_heads": 8},
    "large": {"hidden_dim": 768, "num_layers": 12, "num_heads": 12},
    "xl": {"hidden_dim": 1024, "num_layers": 16, "num_heads": 16},
}

TEXT_CONDITIONING_MODES = {"pooled", "token_prefix"}


def normalize_model_size(value):
    value = str(value or "custom").lower()
    if value == "XL".lower():
        return "xl"
    return value


def apply_model_size_preset(args):
    model_size = normalize_model_size(getattr(args, "model_size", "custom"))
    if model_size == "custom":
        return args
    if model_size not in MODEL_SIZE_PRESETS:
        raise ValueError(f"Unsupported model_size={model_size!r}; expected custom, small, base, large, or xl.")
    for key, value in MODEL_SIZE_PRESETS[model_size].items():
        setattr(args, key, value)
    setattr(args, "model_size", model_size)
    return args


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, value):
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=value.device, dtype=torch.float32) * -scale)
        args = value.float()[:, None] * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class UnconditionalFlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim=COMPACT_DIM,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_frames=400,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, mask=None):
        if x.shape[1] > self.max_frames:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_frames={self.max_frames}")
        h = self.input_proj(x)
        h = h + self.pos_embed[:, : x.shape[1]]
        h = h + self.time_embed(t).unsqueeze(1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        out = self.output_proj(h)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class TextConditionalFlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim=COMPACT_DIM,
        text_dim=768,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_frames=400,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, text_emb, mask=None, text_mask=None):
        if x.shape[1] > self.max_frames:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_frames={self.max_frames}")
        if text_emb.ndim != 2:
            raise ValueError(f"Expected text_emb with shape [B, D], got {tuple(text_emb.shape)}")
        if text_emb.shape[0] != x.shape[0]:
            raise ValueError(f"text_emb batch {text_emb.shape[0]} does not match motion batch {x.shape[0]}")

        h = self.input_proj(x)
        h = h + self.pos_embed[:, : x.shape[1]]
        h = h + self.time_embed(t).unsqueeze(1)
        h = h + self.text_proj(text_emb).unsqueeze(1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        out = self.output_proj(h)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class TextPrefixFlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim=COMPACT_DIM,
        text_dim=768,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_frames=400,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )
        self.motion_pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.time_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.motion_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.motion_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.text_type_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.time_type_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.motion_type_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, text_tokens, mask=None, text_mask=None):
        if x.shape[1] > self.max_frames:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_frames={self.max_frames}")
        if text_tokens.ndim != 3:
            raise ValueError(f"Expected text_tokens with shape [B, L, D], got {tuple(text_tokens.shape)}")
        if text_tokens.shape[0] != x.shape[0]:
            raise ValueError(f"text_tokens batch {text_tokens.shape[0]} does not match motion batch {x.shape[0]}")
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        else:
            text_mask = text_mask.to(device=text_tokens.device, dtype=torch.bool)
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError(f"text_mask shape {tuple(text_mask.shape)} does not match text tokens {tuple(text_tokens.shape[:2])}")

        motion_h = self.input_proj(x)
        motion_h = motion_h + self.motion_pos_embed[:, : x.shape[1]]
        motion_h = motion_h + self.motion_type_embed
        text_h = self.text_proj(text_tokens) + self.text_type_embed
        time_h = self.time_embed(t).unsqueeze(1) + self.time_type_embed

        h = torch.cat([text_h, time_h, motion_h], dim=1)
        time_mask = torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)
        if mask is None:
            motion_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            motion_mask = mask.to(device=x.device, dtype=torch.bool)
        src_mask = torch.cat([text_mask.to(x.device), time_mask, motion_mask], dim=1)
        key_padding_mask = ~src_mask

        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        motion_start = text_tokens.shape[1] + 1
        h = h[:, motion_start:]
        h = self.norm(h)
        out = self.output_proj(h)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


def build_model_from_args(args):
    return UnconditionalFlowTransformer(
        input_dim=COMPACT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_frames=args.max_frames,
    )


def build_text_conditioned_model_from_args(args):
    args = apply_model_size_preset(args)
    text_conditioning = getattr(args, "text_conditioning", "pooled")
    input_dim = int(getattr(args, "input_dim", COMPACT_DIM))
    max_frames = int(getattr(args, "flow_max_frames", getattr(args, "max_frames", 400)))
    if text_conditioning == "pooled":
        model_cls = TextConditionalFlowTransformer
    elif text_conditioning == "token_prefix":
        model_cls = TextPrefixFlowTransformer
    else:
        raise ValueError(f"Unsupported text_conditioning={text_conditioning!r}; expected pooled or token_prefix.")
    return model_cls(
        input_dim=input_dim,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_frames=max_frames,
    )


def _gaussian_kernel1d(kernel_size, device):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return None
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(float(kernel_size) / 6.0, 1e-3)
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def smooth_time(x, kernel_size):
    """Apply light temporal Gaussian smoothing to [B, T, D] tensors."""

    kernel = _gaussian_kernel1d(kernel_size, x.device)
    if kernel is None:
        return x
    pad = kernel.numel() // 2
    channels = x.shape[-1]
    y = x.transpose(1, 2)
    y = torch.nn.functional.pad(y, (pad, pad), mode="replicate")
    weight = kernel.view(1, 1, -1).expand(channels, 1, -1)
    y = torch.nn.functional.conv1d(y, weight, groups=channels)
    return y.transpose(1, 2)


def make_sequence_noise(shape, device=None, mask=None, smoothing=0, dtype=torch.float32):
    """Create optional temporally correlated Gaussian source noise."""

    x = torch.randn(shape, device=device, dtype=dtype)
    if smoothing and smoothing > 1:
        x = smooth_time(x, smoothing)
        if mask is not None:
            valid = mask.to(device=device, dtype=x.dtype).unsqueeze(-1)
            denom = (valid.sum(dim=(1, 2), keepdim=True) * x.shape[-1]).clamp_min(1.0)
            mean = (x * valid).sum(dim=(1, 2), keepdim=True) / denom
            var = ((x - mean).pow(2) * valid).sum(dim=(1, 2), keepdim=True) / denom
            x = (x - mean) / var.sqrt().clamp_min(1e-4)
        else:
            x = (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    if mask is not None:
        x = x * mask.to(device=device, dtype=x.dtype).unsqueeze(-1)
    return x


@torch.no_grad()
def sample_euler(model, shape, steps=20, device=None, mask=None, noise_smoothing=0):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if mask is None:
        mask = torch.ones(shape[0], shape[1], dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)
    x = make_sequence_noise(
        shape,
        device=device,
        mask=mask,
        smoothing=noise_smoothing,
        dtype=next(model.parameters()).dtype,
    )
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = torch.full((shape[0],), step / float(steps), device=device, dtype=torch.float32)
        v = model(x, t, mask=mask)
        x = x + dt * v
        x = x * mask.unsqueeze(-1).to(x.dtype)
    return x


@torch.no_grad()
def sample_heun(model, shape, steps=20, device=None, mask=None, noise_smoothing=0):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if mask is None:
        mask = torch.ones(shape[0], shape[1], dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)
    x = make_sequence_noise(
        shape,
        device=device,
        mask=mask,
        smoothing=noise_smoothing,
        dtype=next(model.parameters()).dtype,
    )
    dt = 1.0 / float(steps)
    for step in range(steps):
        t0 = torch.full((shape[0],), step / float(steps), device=device, dtype=torch.float32)
        t1 = torch.full((shape[0],), min((step + 1) / float(steps), 1.0), device=device, dtype=torch.float32)
        v0 = model(x, t0, mask=mask)
        x_pred = (x + dt * v0) * mask.unsqueeze(-1).to(x.dtype)
        v1 = model(x_pred, t1, mask=mask)
        x = x + 0.5 * dt * (v0 + v1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
    return x


def move_text_condition(text_condition, device, dtype):
    if isinstance(text_condition, (tuple, list)):
        text_tokens, text_mask = text_condition
        return (
            text_tokens.to(device=device, dtype=dtype),
            text_mask.to(device=device, dtype=torch.bool),
        )
    return text_condition.to(device=device, dtype=dtype)


def call_text_model(model, x, t, text_condition, mask=None):
    if isinstance(text_condition, (tuple, list)):
        text_tokens, text_mask = text_condition
        return model(x, t, text_tokens, mask=mask, text_mask=text_mask)
    return model(x, t, text_condition, mask=mask)


def make_initial_state(shape, device, dtype, mask=None, noise_smoothing=0, x0=None, source_noise_scale=1.0):
    if x0 is None:
        return make_sequence_noise(
            shape,
            device=device,
            mask=mask,
            smoothing=noise_smoothing,
            dtype=dtype,
        )
    x = x0.to(device=device, dtype=dtype)
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"x0 shape {tuple(x.shape)} does not match requested shape {tuple(shape)}")
    scale = float(source_noise_scale)
    if scale > 0:
        x = x + scale * make_sequence_noise(
            shape,
            device=device,
            mask=mask,
            smoothing=noise_smoothing,
            dtype=dtype,
        )
    if mask is not None:
        x = x * mask.to(device=device, dtype=dtype).unsqueeze(-1)
    return x


@torch.no_grad()
def sample_euler_text(
    model,
    text_condition,
    shape,
    steps=20,
    device=None,
    mask=None,
    noise_smoothing=0,
    x0=None,
    source_noise_scale=1.0,
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    text_condition = move_text_condition(text_condition, device=device, dtype=next(model.parameters()).dtype)
    text_batch = text_condition[0].shape[0] if isinstance(text_condition, (tuple, list)) else text_condition.shape[0]
    if text_batch != shape[0]:
        raise ValueError(f"text condition batch {text_batch} does not match sample batch {shape[0]}")
    if mask is None:
        mask = torch.ones(shape[0], shape[1], dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)
    x = make_initial_state(
        shape,
        device=device,
        dtype=dtype,
        mask=mask,
        noise_smoothing=noise_smoothing,
        x0=x0,
        source_noise_scale=source_noise_scale,
    )
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = torch.full((shape[0],), step / float(steps), device=device, dtype=torch.float32)
        v = call_text_model(model, x, t, text_condition, mask=mask)
        x = x + dt * v
        x = x * mask.unsqueeze(-1).to(x.dtype)
    return x


@torch.no_grad()
def sample_heun_text(
    model,
    text_condition,
    shape,
    steps=20,
    device=None,
    mask=None,
    noise_smoothing=0,
    x0=None,
    source_noise_scale=1.0,
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    text_condition = move_text_condition(text_condition, device=device, dtype=dtype)
    text_batch = text_condition[0].shape[0] if isinstance(text_condition, (tuple, list)) else text_condition.shape[0]
    if text_batch != shape[0]:
        raise ValueError(f"text condition batch {text_batch} does not match sample batch {shape[0]}")
    if mask is None:
        mask = torch.ones(shape[0], shape[1], dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)
    x = make_initial_state(
        shape,
        device=device,
        dtype=dtype,
        mask=mask,
        noise_smoothing=noise_smoothing,
        x0=x0,
        source_noise_scale=source_noise_scale,
    )
    dt = 1.0 / float(steps)
    for step in range(steps):
        t0 = torch.full((shape[0],), step / float(steps), device=device, dtype=torch.float32)
        t1 = torch.full((shape[0],), min((step + 1) / float(steps), 1.0), device=device, dtype=torch.float32)
        v0 = call_text_model(model, x, t0, text_condition, mask=mask)
        x_pred = (x + dt * v0) * mask.unsqueeze(-1).to(x.dtype)
        v1 = call_text_model(model, x_pred, t1, text_condition, mask=mask)
        x = x + 0.5 * dt * (v0 + v1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
    return x
