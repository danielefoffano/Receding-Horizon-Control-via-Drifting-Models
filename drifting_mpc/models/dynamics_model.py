"""Explicit one-step dynamics model learned from the offline dataset.

Drift RHC uses this only for the training-time consistency regulariser: the
model is fit on the dataset transitions, then frozen, and the generated plans
are penalised for deviating from it. Inference stays model-free.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualDynamicsModel(nn.Module):
    """MLP predicting the state increment: x' = x + g(x, u)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        layers: list[nn.Module] = [nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.net(torch.cat([state, action], dim=-1))

    @torch.no_grad()
    def rollout(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll a batch of action sequences. state (B,dx), actions (B,H,du) -> (B,H+1,dx)."""
        states = [state]
        current = state
        for t in range(actions.shape[1]):
            current = self.forward(current, actions[:, t])
            states.append(current)
        return torch.stack(states, dim=1)
