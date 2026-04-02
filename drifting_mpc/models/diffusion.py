from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            timesteps = timesteps.reshape(-1)
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        dtype = torch.float32
        if half_dim == 0:
            return timesteps.to(dtype).unsqueeze(-1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=device, dtype=dtype)
            * (-(math.log(10000.0) / max(half_dim - 1, 1)))
        )
        args = timesteps.to(dtype).unsqueeze(-1) * frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.norm(x)
        hidden = self.act(self.fc1(hidden))
        hidden = self.fc2(hidden)
        return residual + hidden


class ConditionalTrajectoryDiffusionModel(nn.Module):
    """MLP denoiser over trajectory tensors.

    The denoiser is unconditional in the Diffuser sense: conditioning is handled
    externally by clamping known observation slices in the trajectory tensor at
    every diffusion step. The model therefore only consumes the noisy trajectory
    and the timestep embedding.
    """

    def __init__(
        self,
        horizon_steps: int,
        transition_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.transition_dim = int(transition_dim)
        self.trajectory_dim = self.horizon_steps * self.transition_dim
        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(self.trajectory_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_act = nn.SiLU()
        self.output = nn.Linear(hidden_dim, self.trajectory_dim)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del x0, omega
        if noisy_trajectory.ndim != 3:
            raise ValueError("noisy_trajectory must have shape [batch, horizon_steps, transition_dim].")
        if noisy_trajectory.shape[1] != self.horizon_steps or noisy_trajectory.shape[2] != self.transition_dim:
            raise ValueError(
                f"Expected noisy_trajectory shape [batch, {self.horizon_steps}, {self.transition_dim}], got {tuple(noisy_trajectory.shape)}."
            )
        batch_size = noisy_trajectory.shape[0]
        flat = noisy_trajectory.reshape(batch_size, self.trajectory_dim)
        time_features = self.time_mlp(self.time_embedding(timesteps))
        hidden = self.input_proj(torch.cat([flat, time_features], dim=-1))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_act(self.final_norm(hidden))
        output = self.output(hidden)
        return output.reshape(batch_size, self.horizon_steps, self.transition_dim)
