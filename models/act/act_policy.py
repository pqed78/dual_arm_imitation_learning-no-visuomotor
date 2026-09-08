# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Action Chunking with Transformers (ACT) Policy (Zhao et al. 2023)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import ACTTransformer


from models.vision_encoder import VisionEncoder

class ACTPolicy(nn.Module):
    """CVAE + Transformer Action Chunking Policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        chunk_size: int = 24,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        dropout: float = 0.1,
        temporal_ensembling: bool = True,
        temporal_ensemble_coeff: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.d_model = d_model
        self.temporal_ensembling = temporal_ensembling
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        
        self.vision_encoder = VisionEncoder(feature_dim=512)
        in_dim = obs_dim + 512

        # --- CVAE Encoder (Obs + Action Chunk -> Latent z) ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.obs_proj_enc = nn.Linear(in_dim, d_model)
        self.act_proj_enc = nn.Linear(act_dim, d_model)

        cvae_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_encoder_layer, num_layers=2)
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)  # mu and logvar

        # --- Decoder (Obs + Latent z -> Action Chunk) ---
        self.obs_proj_dec = nn.Linear(in_dim, d_model)
        self.latent_proj_dec = nn.Linear(latent_dim, d_model)

        # Learnable action sequence query tokens
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model))

        # Core Transformer
        self.transformer = ACTTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output projection to action space
        self.action_head = nn.Linear(d_model, act_dim)

        # Temporal Ensembling buffer for inference
        self.reset_temporal_ensemble()

    def reset_temporal_ensemble(self):
        """Reset the temporal ensembling action buffer."""
        self.ensemble_buffer = {}  # maps future timestep t -> list of (predicted_action, weight)

    def encode(self, obs: torch.Tensor, img: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode demonstration trajectory into latent style distribution."""
        batch_size = obs.shape[0]

        img_features = self.vision_encoder(img)
        combined_obs = torch.cat([obs, img_features], dim=-1)

        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        obs_tok = self.obs_proj_enc(combined_obs).unsqueeze(1)    # (B, 1, d_model)
        act_tok = self.act_proj_enc(actions)             # (B, chunk_size, d_model)

        seq = torch.cat([cls, obs_tok, act_tok], dim=1)  # (B, 2 + chunk_size, d_model)
        enc_out = self.cvae_encoder(seq)

        cls_out = enc_out[:, 0, :]                       # (B, d_model)
        latent_params = self.latent_proj(cls_out)
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent z ~ N(mu, exp(logvar))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs: torch.Tensor,
        img: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size = obs.shape[0]

        if actions is not None:
            # Training mode: sample z from encoder
            mu, logvar = self.encode(obs, img, actions)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference mode: set latent z to prior mean (0)
            mu, logvar = None, None
            z = torch.zeros((batch_size, self.latent_dim), device=obs.device, dtype=obs.dtype)

        # Prepare Decoder inputs: Condition tokens [obs, z]
        img_features = self.vision_encoder(img)
        combined_obs = torch.cat([obs, img_features], dim=-1)
        obs_tok = self.obs_proj_dec(combined_obs).unsqueeze(1)       # (B, 1, d_model)
        z_tok = self.latent_proj_dec(z).unsqueeze(1)        # (B, 1, d_model)
        context = torch.cat([obs_tok, z_tok], dim=1)        # (B, 2, d_model)

        queries = self.action_queries.expand(batch_size, -1, -1)  # (B, chunk_size, d_model)

        dec_out = self.transformer(src=context, tgt=queries)
        pred_actions = self.action_head(dec_out)

        return pred_actions, mu, logvar

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute ACT loss: L1/MSE reconstruction + KL divergence."""
        obs = batch["obs"]
        img = batch["rgb_image"]
        target_actions = batch["action"]

        pred_actions, mu, logvar = self.forward(obs, img, target_actions)

        # L1 / L2 action reconstruction loss
        recon_loss = F.l1_loss(pred_actions, target_actions)

        # KL divergence: D_KL(q(z|o,a) || p(z)) where p(z) ~ N(0, I)
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    @torch.no_grad()
    def predict_action_step(
        self,
        obs: torch.Tensor,
        img: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            img = img.unsqueeze(0)

        # Predict fresh chunk of actions for the next chunk_size steps
        pred_chunk, _, _ = self.forward(obs, img, actions=None)  # (1, chunk_size, act_dim)
        chunk_actions = pred_chunk.squeeze(0)                # (chunk_size, act_dim)

        if not self.temporal_ensembling:
            # Without ensembling: just return the first action
            return chunk_actions[0]

        # Temporal Ensembling: Add predicted trajectory to buffer
        k = self.temporal_ensemble_coeff
        for i in range(self.chunk_size):
            t = current_step + i
            # Exponential decay weight: w = exp(-k * i)
            weight = np.exp(-k * i)
            action_i = chunk_actions[i].detach()

            if t not in self.ensemble_buffer:
                self.ensemble_buffer[t] = []
            self.ensemble_buffer[t].append((action_i, weight))

        # Compute weighted average for the current step
        entries = self.ensemble_buffer.pop(current_step, None)
        if entries is None:
            return chunk_actions[0]

        total_weight = sum(w for _, w in entries)
        blended_action = sum(act * (w / total_weight) for act, w in entries)

        # Clean old timesteps from buffer to prevent memory growth
        old_keys = [k for k in self.ensemble_buffer.keys() if k < current_step]
        for k_old in old_keys:
            del self.ensemble_buffer[k_old]

        return blended_action
