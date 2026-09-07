# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""1D Temporal Conditional UNet for Diffusion Policy (Chi et al. 2023)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning on timesteps and global observations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)

        # FiLM generator: predicts scale and bias from condition embedding
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        """
        out = self.conv1(x)

        # FiLM scale & bias
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, 2 * out_channels, 1)
        scale, bias = torch.chunk(cond_embed, 2, dim=1)

        out = out * (1.0 + scale) + bias
        out = self.conv2(out)

        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D Temporal UNet conditional on observation history and diffusion step."""

    def __init__(
        self,
        act_dim: int,
        cond_dim: int,
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.cond_dim = cond_dim

        # Timestep embedding MLP
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Combined condition dimension (timestep + observation condition)
        total_cond_dim = diffusion_step_embed_dim + cond_dim

        # Encoder (Downsampling)
        all_dims = [act_dim] + list(down_dims)
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims)):
            in_d = all_dims[i]
            out_d = all_dims[i + 1]
            self.down_blocks.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(in_d, out_d, total_cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1D(out_d, out_d, total_cond_dim, kernel_size, n_groups),
                    nn.Conv1d(out_d, out_d, kernel_size=3, stride=2, padding=1),  # Downsample
                ])
            )

        # Mid block
        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(mid_dim, mid_dim, total_cond_dim, kernel_size, n_groups)
        self.mid_block2 = ConditionalResidualBlock1D(mid_dim, mid_dim, total_cond_dim, kernel_size, n_groups)

        # Decoder (Upsampling)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(down_dims))):
            dim_in = down_dims[i]
            dim_out = down_dims[i - 1] if i > 0 else down_dims[0]
            self.up_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose1d(dim_in, dim_in, kernel_size=4, stride=2, padding=1),  # Upsample
                    ConditionalResidualBlock1D(dim_in * 2, dim_out, total_cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1D(dim_out, dim_out, total_cond_dim, kernel_size, n_groups),
                ])
            )

        # Final projection to action dimension
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(down_dims[0], act_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, T, act_dim) noisy actions
            timestep: (B,) diffusion timesteps
            global_cond: (B, cond_dim) observation embedding
        Returns:
            predicted noise or action: (B, T, act_dim)
        """
        # Permute for 1D convolution: (B, act_dim, T)
        x = sample.transpose(1, 2)
        orig_len = x.shape[-1]

        # Compute combined condition vector
        time_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([time_emb, global_cond], dim=-1)

        skips = []
        # Downward pass
        for res1, res2, downsample in self.down_blocks:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        # Mid pass
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        # Upward pass
        for upsample, res1, res2 in self.up_blocks:
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        # Final projection
        x = self.final_conv(x)

        if x.shape[-1] != orig_len:
            x = F.interpolate(x, size=orig_len, mode="linear", align_corners=False)

        # Permute back: (B, T, act_dim)
        return x.transpose(1, 2)

