"""Pose-sequence Transformer encoder (paper Section 3.2).

Input:  (B, N, F) keypoint sequences, F = 150 (x,y) or 225 (x,y,z)
Steps:  linear pose embedding + ReLU (Eq. 1)  ->  sinusoidal positional
        encoding (Eqs. 2-3)  ->  prepend learnable [CLS] token  ->
        nn.TransformerEncoder (post-norm, as in Vaswani et al.)  ->
        take final [CLS] hidden state  ->  optional linear projection
        (the "Linear" head shown in the paper's Figure 3).
Output: (B, D) embedding vectors.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(base) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SignTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pe_base: float = 10000.0,
        max_len: int = 512,
        proj_dim: int | None = 128,
    ):
        super().__init__()
        self.pose_embed = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, pe_base)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,  # post-norm: LayerNorm(x + Sublayer(x)), Eq. 4
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, proj_dim) if proj_dim else nn.Identity()
        self.embed_dim = proj_dim or d_model

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, N, F) -> (B, embed_dim)."""
        h = self.pos_enc(self.pose_embed(x))                    # (B, N, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)          # (B, 1, D)
        h = torch.cat([cls, h], dim=1)                          # (B, N+1, D)
        if padding_mask is not None:
            pad = torch.zeros(
                h.size(0), 1, dtype=torch.bool, device=h.device
            )
            padding_mask = torch.cat([pad, padding_mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.proj(h[:, 0])                                # [CLS]


def build_model(cfg: dict, input_dim: int) -> SignTransformer:
    m = cfg["model"]
    return SignTransformer(
        input_dim=input_dim,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        pe_base=float(m["pe_base"]),
        max_len=m["max_len"],
        proj_dim=m.get("proj_dim"),
    )
