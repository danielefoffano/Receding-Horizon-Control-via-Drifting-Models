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
