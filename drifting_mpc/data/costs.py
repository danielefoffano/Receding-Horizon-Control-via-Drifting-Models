from __future__ import annotations

import math

import numpy as np
import torch


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(math.log(low), math.log(high))))


def omega_ranges(config: dict) -> list[tuple[float, float]]:
    """Log-uniform sampling ranges for omega = [q (state_dim), r (action_dim)].

    Accepts either the mass-spring-damper keys (q_pos_range/q_vel_range/r_u_range)
    or the general form (q_ranges: list of [lo, hi], r_ranges: list of [lo, hi]).
    """
    cost_cfg = config["cost"]
    if "q_ranges" in cost_cfg:
        return [tuple(x) for x in cost_cfg["q_ranges"]] + [tuple(x) for x in cost_cfg["r_ranges"]]
    return [tuple(cost_cfg["q_pos_range"]), tuple(cost_cfg["q_vel_range"]), tuple(cost_cfg["r_u_range"])]


def sample_omega(rng: np.random.Generator, config: dict) -> np.ndarray:
    return np.array([sample_log_uniform(rng, lo, hi) for lo, hi in omega_ranges(config)],
                    dtype=np.float32)


def omega_to_cost_matrices(
    omega: np.ndarray, state_dim: int = 2, action_dim: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal Q, R from omega = [q (state_dim), r (action_dim)]."""
    omega = np.asarray(omega, dtype=np.float32)
    q_mat = np.diag(omega[:state_dim]).astype(np.float32)
    r_mat = np.diag(omega[state_dim:state_dim + action_dim]).astype(np.float32)
    return q_mat, r_mat


def trajectory_cost_np(states: np.ndarray, actions: np.ndarray, omega: np.ndarray) -> float:
    """Quadratic cost with diagonal Q, R for an arbitrary state/action dimension.

    omega packs the diagonals as [q (state_dim), r (action_dim)]; for the
    mass-spring-damper this is exactly (q_pos, q_vel, r_u).
    """
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    omega = np.asarray(omega, dtype=np.float32)
    state_dim, action_dim = states.shape[-1], actions.shape[-1]
    q = omega[:state_dim]
    r = omega[state_dim:state_dim + action_dim]
    stage = (np.square(states[:-1]) * q).sum() + (np.square(actions) * r).sum()
    terminal = (np.square(states[-1]) * q).sum()
    return float(stage + terminal)


def trajectory_cost_torch(states: torch.Tensor, actions: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Broadcasted quadratic cost for (..., H+1, dx) states and (..., H, du) actions.

    omega packs [q (dx), r (du)] along its last axis.
    """
    state_dim, action_dim = states.shape[-1], actions.shape[-1]
    q = omega[..., :state_dim].unsqueeze(-2)                      # (..., 1, dx)
    r = omega[..., state_dim:state_dim + action_dim].unsqueeze(-2)  # (..., 1, du)
    stage = (states[..., :-1, :].square() * q).sum(dim=(-1, -2))
    stage = stage + (actions.square() * r).sum(dim=(-1, -2))
    terminal = (states[..., -1, :].square() * omega[..., :state_dim]).sum(dim=-1)
    return stage + terminal
