"""Shared 1-D convolutional blocks for the temporal-convolution generator.

The MLP generator emits the whole trajectory from two linear heads and shares
nothing across timesteps. The convolutional generator instead decodes along the
horizon axis with these residual temporal blocks, which share weights across
timesteps and give a local receptive field in time.
"""
from __future__ import annotations

import torch
from torch import nn


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, embed_dim: int, kernel: int = 5):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel)
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out_ch))
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.block1(x) + self.time_mlp(t).unsqueeze(-1)
        out = self.block2(out)
        return out + self.residual(x)
