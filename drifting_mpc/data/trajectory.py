from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch

TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)


def trajectory_dim(horizon: int, state_dim: int = 2, action_dim: int = 1) -> int:
    return horizon * (state_dim + action_dim)


def encode_trajectory(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Encode a state-action trajectory as (u0, dx1, u1, dx2, ..., u_{H-1}, dx_H)."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.shape[0] != actions.shape[0] + 1:
        raise ValueError("states must have length H+1 when actions have length H.")
    x0 = states[0]
    deltas = states[1:] - x0[None, :]
    chunks = []
    for t in range(actions.shape[0]):
        chunks.append(actions[t])
        chunks.append(deltas[t])
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _split_zeta_np(zeta: np.ndarray, horizon: int, state_dim: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(zeta, dtype=np.float32).reshape(horizon, action_dim + state_dim)
    actions = blocks[:, :action_dim]
    deltas = blocks[:, action_dim:]
    return actions, deltas


def _split_zeta_torch(zeta: torch.Tensor, horizon: int, state_dim: int, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = zeta.reshape(*zeta.shape[:-1], horizon, action_dim + state_dim)
    actions = blocks[..., :, :action_dim]
    deltas = blocks[..., :, action_dim:]
    return actions, deltas


def decode_trajectory_np(
    x0: np.ndarray,
    zeta: np.ndarray,
    horizon: int,
    state_dim: int = 2,
    action_dim: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a relative-coordinate trajectory back to full states and actions."""
    x0 = np.asarray(x0, dtype=np.float32)
    actions, deltas = _split_zeta_np(zeta, horizon=horizon, state_dim=state_dim, action_dim=action_dim)
    states = np.concatenate([x0[None, :], x0[None, :] + deltas], axis=0)
    return states.astype(np.float32), actions.astype(np.float32)


def decode_trajectory_torch(
    x0: torch.Tensor,
    zeta: torch.Tensor,
    horizon: int,
    state_dim: int = 2,
    action_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions, deltas = _split_zeta_torch(zeta, horizon=horizon, state_dim=state_dim, action_dim=action_dim)
    base = x0.unsqueeze(-2)
    states = torch.cat([base, base + deltas], dim=-2)
    return states, actions


def interleave_trajectory_parts(actions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Interleave action and state-offset tokens and flatten the result."""
    pieces = []
    for step in range(actions.shape[-2]):
        pieces.append(actions[..., step, :])
        pieces.append(deltas[..., step, :])
    return torch.cat(pieces, dim=-1)
