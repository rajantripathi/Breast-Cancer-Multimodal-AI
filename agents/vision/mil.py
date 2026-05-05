from __future__ import annotations

"""Bag-level vision aggregation modules for pathology slide embeddings."""

import torch
from torch import nn


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Convert per-sample lengths into a boolean validity mask."""
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


class AttentionMILPool(nn.Module):
    """Gated attention MIL pooling after Ilse et al."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, attention_dim: int = 128):
        super().__init__()
        self.instance_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.attn_v = nn.Linear(hidden_dim, attention_dim)
        self.attn_u = nn.Linear(hidden_dim, attention_dim)
        self.attn_w = nn.Linear(attention_dim, 1, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, bag: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.instance_proj(bag)
        logits = self.attn_w(torch.tanh(self.attn_v(encoded)) * torch.sigmoid(self.attn_u(encoded))).squeeze(-1)
        if lengths is None:
            mask = torch.ones(logits.shape, dtype=torch.bool, device=logits.device)
        else:
            mask = lengths_to_mask(lengths, encoded.shape[1])
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        empty_rows = ~mask.any(dim=1, keepdim=True)
        logits = torch.where(empty_rows, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
        return self.output_norm(pooled)


class TransformerMILPool(nn.Module):
    """Transformer bag pooling with a learned CLS token."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.instance_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, bag: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.instance_proj(bag)
        batch_size, max_length, _ = encoded.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, encoded], dim=1)
        padding_mask = None
        if lengths is not None:
            instance_mask = lengths_to_mask(lengths, max_length)
            cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=encoded.device)
            padding_mask = ~torch.cat([cls_mask, instance_mask], dim=1)
        transformed = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.output_norm(transformed[:, 0, :])
