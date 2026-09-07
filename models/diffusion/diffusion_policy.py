# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion Policy (Chi et al. 2023) implementation for State-based Dual Arm Manipulation."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet1d import ConditionalUnet1D


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in Nichol and Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionPolicy(nn.Module):
    """Diffusion Policy model supporting training and fast closed-loop inference."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.num_train_timesteps = num_train_timesteps

        # Global conditioning dimension = obs_horizon * obs_dim
        self.cond_dim = obs_horizon * obs_dim

        # 1D Temporal UNet backbone
        self.model = ConditionalUnet1D(
            act_dim=act_dim,
            cond_dim=self.cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        # Setup diffusion noise scheduler constants
        if beta_schedule == "squaredcos_cap_v2":
            betas = cosine_beta_schedule(num_train_timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers so they are automatically moved with model.to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8),
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward diffusion training step.

        Args:
            batch: dict with:
                "obs": (B, obs_horizon, obs_dim)
                "action": (B, pred_horizon, act_dim)
        """
        obs_seq = batch["obs"]
        act_seq = batch["action"]
        batch_size = act_seq.shape[0]

        # Flatten observation history into single conditioning vector
        global_cond = obs_seq.reshape(batch_size, -1)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=act_seq.device, dtype=torch.long
        )

        # Sample noise
        noise = torch.randn_like(act_seq)

        # Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        noisy_actions = sqrt_alpha * act_seq + sqrt_one_minus_alpha * noise

        # Predict noise
        pred_noise = self.model(noisy_actions, timesteps, global_cond)

        # Loss: MSE between true noise and predicted noise
        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss, "mse": loss.detach()}

    @torch.no_grad()
    def predict_action(
        self,
        obs_seq: torch.Tensor,
        num_inference_steps: int = 15,
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """Sample actions using reverse diffusion process.

        Args:
            obs_seq: (B, obs_horizon, obs_dim) or (obs_horizon, obs_dim)
            num_inference_steps: Number of sampling steps (fast DDIM acceleration)
            use_ddim: Whether to use DDIM (fast) or standard DDPM
        Returns:
            actions: (B, pred_horizon, act_dim)
        """
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        batch_size = obs_seq.shape[0]
        device = obs_seq.device

        global_cond = obs_seq.reshape(batch_size, -1)

        # Start from pure Gaussian noise
        x = torch.randn((batch_size, self.pred_horizon, self.act_dim), device=device)

        if use_ddim and num_inference_steps < self.num_train_timesteps:
            # DDIM accelerated sampling
            timesteps = torch.linspace(
                self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
            )

            for i in range(len(timesteps)):
                t = timesteps[i].repeat(batch_size)
                pred_noise = self.model(x, t, global_cond)

                alpha_bar = self.alphas_cumprod[t].view(batch_size, 1, 1)
                t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=device)
                alpha_bar_prev = self.alphas_cumprod[t_prev.repeat(batch_size)].view(batch_size, 1, 1)

                # Predict x_0 from current noise prediction
                pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

                if i < len(timesteps) - 1:
                    dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                else:
                    x = pred_x0
        else:
            # Standard DDPM reverse process
            for t_idx in reversed(range(self.num_train_timesteps)):
                t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
                pred_noise = self.model(x, t, global_cond)

                beta = self.betas[t].view(batch_size, 1, 1)
                alpha = self.alphas[t].view(batch_size, 1, 1)
                alpha_bar = self.alphas_cumprod[t].view(batch_size, 1, 1)

                # DDPM mean
                mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise)

                if t_idx > 0:
                    var = self.posterior_variance[t].view(batch_size, 1, 1)
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(var) * noise
                else:
                    x = mean

        return x
