# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior Cloning (BC) Policies.

Includes MLP-based and RNN-based policy networks for state-based imitation learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.vision_encoder import VisionEncoder

class MLPBCPolicy(nn.Module):
    """Multi-Layer Perceptron Behavior Cloning Policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [512, 512, 256],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.vision_encoder = VisionEncoder(feature_dim=512)
        in_dim = obs_dim + 512

        act_fn = nn.ReLU if activation.lower() == "relu" else nn.GELU

        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, rgb_image: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            rgb_image = rgb_image.unsqueeze(0)
            
        img_features = self.vision_encoder(rgb_image)
        combined = torch.cat([obs, img_features], dim=-1)
        
        out = self.net(combined)
        if out.shape[0] == 1 and obs.dim() == 1:
            return out.squeeze(0)
        return out

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_act = self.forward(batch["obs"], batch["rgb_image"])
        loss = F.mse_loss(pred_act, batch["action"])
        return {"loss": loss, "mse": loss.detach()}


class RNNBCPolicy(nn.Module):
    """Recurrent (LSTM) Behavior Cloning Policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.vision_encoder = VisionEncoder(feature_dim=512)
        in_dim = obs_dim + 512

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

    def forward(self, obs_seq: torch.Tensor, rgb_seq: torch.Tensor, hidden=None) -> tuple[torch.Tensor, tuple]:
        img_features = self.vision_encoder(rgb_seq) # shape: (B, T, 512)
        combined = torch.cat([obs_seq, img_features], dim=-1)
        
        out, hidden = self.lstm(combined, hidden)
        last_out = out[:, -1, :]  # Take output of last timestep
        action = self.head(last_out)
        return action, hidden

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = batch["obs"]
        img = batch["rgb_image"]
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)
            img = img.unsqueeze(1)
        pred_act, _ = self.forward(obs, img)
        loss = F.mse_loss(pred_act, batch["action"])
        return {"loss": loss, "mse": loss.detach()}
