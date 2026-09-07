# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Transformer Backbone for ACT (Action Chunking with Transformers)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal 1D Positional Embedding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1)]


class ACTTransformer(nn.Module):
    """Transformer Encoder-Decoder architecture for ACT."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.pos_emb = SinusoidalPositionEmbedding(d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, src_len, d_model) - Context tokens (obs + style latent z)
            tgt: (B, tgt_len, d_model) - Query tokens for action sequence
            src_key_padding_mask: (B, src_len)
        Returns:
            out: (B, tgt_len, d_model)
        """
        src = self.pos_emb(src)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        tgt = self.pos_emb(tgt)
        out = self.decoder(tgt, memory)
        return out
