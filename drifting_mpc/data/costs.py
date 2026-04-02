from __future__ import annotations

import math

import numpy as np
import torch


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(math.log(low), math.log(high))))


def sample_omega(rng: np.random.Generator, config: dict) -> np.ndarray:
    cost_cfg = config["cost"]
    omega = np.array(
        [
            sample_log_uniform(rng, *cost_cfg["q_pos_range"]),
            sample_log_uniform(rng, *cost_cfg["q_vel_range"]),
            sample_log_uniform(rng, *cost_cfg["r_u_range"]),
        ],
        dtype=np.float32,
    )
    return omega


def omega_to_cost_matrices(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(omega, dtype=np.float32)
    q_pos, q_vel, r_u = omega.tolist()
    q_mat = np.diag([q_pos, q_vel]).astype(np.float32)
    r_mat = np.array([[r_u]], dtype=np.float32)
    return q_mat, r_mat


def trajectory_cost_np(states: np.ndarray, actions: np.ndarray, omega: np.ndarray) -> float:
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    omega = np.asarray(omega, dtype=np.float32)
    q_pos, q_vel, r_u = omega.tolist()
    position = states[:-1, 0]
    velocity = states[:-1, 1]
    control = actions[:, 0]
    terminal = states[-1]
    stage = q_pos * np.square(position) + q_vel * np.square(velocity) + r_u * np.square(control)
    terminal_cost = q_pos * float(terminal[0] ** 2) + q_vel * float(terminal[1] ** 2)
    return float(stage.sum() + terminal_cost)


def trajectory_cost_torch(states: torch.Tensor, actions: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Broadcasted quadratic trajectory cost for (..., H+1, 2) states and (..., H, 1) actions."""
    q_pos = omega[..., 0]
    q_vel = omega[..., 1]
    r_u = omega[..., 2]
    position = states[..., :-1, 0]
    velocity = states[..., :-1, 1]
    control = actions[..., :, 0]
    terminal = states[..., -1, :]
    stage_cost = q_pos.unsqueeze(-1) * position.square()
    stage_cost = stage_cost + q_vel.unsqueeze(-1) * velocity.square()
    stage_cost = stage_cost + r_u.unsqueeze(-1) * control.square()
    terminal_cost = q_pos * terminal[..., 0].square() + q_vel * terminal[..., 1].square()
    return stage_cost.sum(dim=-1) + terminal_cost
