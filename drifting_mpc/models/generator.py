from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from drifting_mpc.data.trajectory import interleave_trajectory_parts


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.norm(inputs)
        hidden = self.act(self.fc1(hidden))
        hidden = self.fc2(hidden)
        return residual + hidden


class ConditionalTrajectoryGenerator(nn.Module):
    """Conditional residual MLP that directly generates full relative trajectories."""

    def __init__(
        self,
        horizon: int,
        state_dim: int,
        action_dim: int,
        omega_dim: int,
        eps_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.omega_dim = omega_dim
        self.eps_dim = eps_dim
        self.hidden_dim = hidden_dim
        input_dim = eps_dim + state_dim + omega_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_act = nn.SiLU()
        self.action_head = nn.Linear(hidden_dim, horizon * action_dim)
        self.offset_head = nn.Linear(hidden_dim, horizon * state_dim)

    def forward(self, eps: torch.Tensor, x0: torch.Tensor, omega: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps.shape[0] != x0.shape[0]:
            raise ValueError("eps and x0 must share the batch dimension.")
        if self.omega_dim > 0:
            if omega is None:
                raise ValueError("omega must be provided when omega_dim > 0.")
            if x0.shape[0] != omega.shape[0]:
                raise ValueError("eps, x0, and omega must share the batch dimension.")
            hidden = torch.cat([eps, x0, omega], dim=-1)
        else:
            hidden = torch.cat([eps, x0], dim=-1)
        hidden = self.input_proj(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_act(self.final_norm(hidden))
        actions = self.action_head(hidden).reshape(-1, self.horizon, self.action_dim)
        deltas = self.offset_head(hidden).reshape(-1, self.horizon, self.state_dim)
        return interleave_trajectory_parts(actions, deltas)


class TemporalConvTrajectoryGenerator(nn.Module):
    """Conditional generator with a temporal convolutional decoder.

    The MLP generator emits the whole trajectory from two linear heads whose size
    scales with horizon x (state_dim + action_dim), and shares nothing across
    timesteps. This variant encodes the conditioning (eps, x0, omega) into a
    latent, broadcasts it along the horizon, and decodes with residual temporal
    convolutions -- the same inductive bias that fixed the diffusion denoiser.
    Interface is identical to ConditionalTrajectoryGenerator.
    """

    def __init__(
        self,
        horizon: int,
        state_dim: int,
        action_dim: int,
        omega_dim: int,
        eps_dim: int = 32,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        from drifting_mpc.models.blocks import Conv1dBlock, ResidualTemporalBlock
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.omega_dim = int(omega_dim)
        self.eps_dim = int(eps_dim)
        self.hidden_dim = int(hidden_dim)
        cond_dim = eps_dim + state_dim + omega_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos = nn.Parameter(torch.zeros(1, hidden_dim, self.horizon))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, hidden_dim, hidden_dim, kernel_size)
             for _ in range(num_blocks)]
        )
        self.final = nn.Sequential(
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size),
            nn.Conv1d(hidden_dim, action_dim + state_dim, 1),
        )

    def forward(self, eps: torch.Tensor, x0: torch.Tensor, omega: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps.shape[0] != x0.shape[0]:
            raise ValueError("eps and x0 must share the batch dimension.")
        if self.omega_dim > 0:
            if omega is None:
                raise ValueError("omega must be provided when omega_dim > 0.")
            cond = torch.cat([eps, x0, omega], dim=-1)
        else:
            cond = torch.cat([eps, x0], dim=-1)
        c = self.cond_mlp(cond)                                   # (B, C)
        h = c.unsqueeze(-1) + self.pos                            # (B, C, H)
        for block in self.blocks:
            h = block(h, c)
        out = self.final(h).transpose(1, 2)                       # (B, H, du+dx)
        actions = out[..., : self.action_dim]
        deltas = out[..., self.action_dim:]
        return interleave_trajectory_parts(actions, deltas)
