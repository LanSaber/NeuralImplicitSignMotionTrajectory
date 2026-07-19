import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return residual + y


class GlossCTCRecognizer(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        model_dim=256,
        conv_layers=3,
        conv_kernel=5,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.conv_layers = int(conv_layers)
        self.conv_kernel = int(conv_kernel)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.dropout_p = float(dropout)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.conv = nn.ModuleList(
            [
                TemporalConvBlock(
                    self.model_dim,
                    kernel_size=self.conv_kernel,
                    dropout=self.dropout_p,
                )
                for _ in range(self.conv_layers)
            ]
        )
        self.lstm = nn.LSTM(
            input_size=self.model_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.head = nn.Linear(self.lstm_hidden * 2, self.vocab_size)

    def config(self):
        return {
            "input_dim": self.input_dim,
            "vocab_size": self.vocab_size,
            "model_dim": self.model_dim,
            "conv_layers": self.conv_layers,
            "conv_kernel": self.conv_kernel,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout_p,
        }

    def forward(self, motion, lengths):
        frame_mask = (
            torch.arange(motion.shape[1], device=motion.device).unsqueeze(0)
            < lengths.to(device=motion.device).unsqueeze(1)
        ).unsqueeze(-1)
        frame_mask = frame_mask.to(dtype=motion.dtype)

        x = self.input_proj(motion)
        x = x * frame_mask
        for block in self.conv:
            x = block(x)
            x = x * frame_mask

        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=motion.shape[1],
        )
        logits = self.head(self.dropout(x))
        return F.log_softmax(logits, dim=-1)
