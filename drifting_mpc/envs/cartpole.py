"""Nonlinear cart-pole benchmark: 4-D state, 1-D input.

State x = (p, p_dot, theta, theta_dot) with theta = 0 upright; action u is the
cart force. Stabilisation about the origin, so the same diagonal quadratic cost
used for the mass-spring-damper applies unchanged with
omega = [q (4), r (1)].

Integrated with RK4 at dt. Unlike the mass-spring-damper there is no exact
discretisation and no closed-form optimal controller -- the reference is iLQR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleSpec:
    m_c: float
    m_p: float
    length: float
    gravity: float
    dt: float
    horizon: int
    episode_len: int
    init_low: np.ndarray
    init_high: np.ndarray
    force_limit: float = 0.0  # 0 disables clipping

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 1


def build_cartpole_spec(config: dict) -> CartPoleSpec:
    env_cfg = config["environment"]
    init = env_cfg["initial_state"]
    low = np.array([init["position"][0], init["velocity"][0],
                    init["angle"][0], init["angular_velocity"][0]], dtype=np.float32)
    high = np.array([init["position"][1], init["velocity"][1],
                     init["angle"][1], init["angular_velocity"][1]], dtype=np.float32)
    return CartPoleSpec(
        m_c=float(env_cfg.get("m_c", 1.0)),
        m_p=float(env_cfg.get("m_p", 0.1)),
        length=float(env_cfg.get("length", 0.5)),
        gravity=float(env_cfg.get("gravity", 9.81)),
        dt=float(env_cfg["dt"]),
        horizon=int(env_cfg["horizon"]),
        episode_len=int(env_cfg["episode_len"]),
        init_low=low,
        init_high=high,
        force_limit=float(env_cfg.get("force_limit", 0.0)),
    )


class CartPoleEnv:
    """Deterministic RK4-integrated cart-pole."""

    def __init__(self, spec: CartPoleSpec):
        self.spec = spec
        self.state = np.zeros(spec.state_dim, dtype=np.float32)
        self.t = 0

    # --- dynamics (batched core) ----------------------------------------
    def _accel(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Continuous-time derivative. state (...,4), force (...,) -> (...,4)."""
        p_dot = state[..., 1]; th = state[..., 2]; th_dot = state[..., 3]
        s, c = np.sin(th), np.cos(th)
        mc, mp, l, g = self.spec.m_c, self.spec.m_p, self.spec.length, self.spec.gravity
        total = mc + mp
        temp = (force + mp * l * th_dot ** 2 * s) / total
        th_acc = (g * s - c * temp) / (l * (4.0 / 3.0 - mp * c ** 2 / total))
        p_acc = temp - mp * l * th_acc * c / total
        return np.stack([p_dot, p_acc, th_dot, th_acc], axis=-1)

    def f_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """RK4 step for a batch. states (...,4), actions (...,1) -> (...,4)."""
        x = np.asarray(states, dtype=np.float64)
        u = np.asarray(actions, dtype=np.float64)[..., 0]
        if self.spec.force_limit > 0.0:
            u = np.clip(u, -self.spec.force_limit, self.spec.force_limit)
        dt = self.spec.dt
        k1 = self._accel(x, u)
        k2 = self._accel(x + 0.5 * dt * k1, u)
        k3 = self._accel(x + 0.5 * dt * k2, u)
        k4 = self._accel(x + dt * k3, u)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def f(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One RK4 step of the true continuous dynamics (single transition)."""
        x = np.asarray(state, dtype=np.float64).reshape(self.spec.state_dim)
        u = np.asarray(action, dtype=np.float64).reshape(self.spec.action_dim)
        return self.f_batch(x, u).astype(np.float32)

    # --- env interface --------------------------------------------------
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.spec.init_low, self.spec.init_high).astype(np.float32)

    def reset(self, initial_state: np.ndarray | None = None,
              rng: np.random.Generator | None = None) -> np.ndarray:
        if initial_state is None:
            if rng is None:
                raise ValueError("An RNG is required when initial_state is not provided.")
            initial_state = self.sample_initial_state(rng)
        self.state = np.asarray(initial_state, dtype=np.float32).copy()
        self.t = 0
        return self.state.copy()

    def step(self, action: np.ndarray | float) -> np.ndarray:
        self.state = self.f(self.state, np.asarray(action, dtype=np.float32).reshape(self.spec.action_dim))
        self.t += 1
        return self.state.copy()

    def rollout_open_loop(self, x0: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.reset(initial_state=x0)
        states = [self.state.copy()]
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.spec.action_dim)
        for action in actions[: self.spec.episode_len]:
            states.append(self.step(action))
        return np.stack(states), actions[: self.spec.episode_len].copy()

    # --- linearisation used by iLQR ------------------------------------
    def linearize_traj(self, states: np.ndarray, actions: np.ndarray, eps: float = 1e-5):
        """Central-difference Jacobians along a trajectory.

        states (H,4), actions (H,1) -> A (H,4,4), B (H,4,1). Every perturbation
        is evaluated in one batched RK4 call, which is what makes iLQR affordable
        inside a receding-horizon loop.
        """
        nx, nu = self.spec.state_dim, self.spec.action_dim
        x = np.asarray(states, dtype=np.float64)
        u = np.asarray(actions, dtype=np.float64).reshape(-1, nu)
        h = x.shape[0]
        a = np.zeros((h, nx, nx)); b = np.zeros((h, nx, nu))
        for i in range(nx):
            d = np.zeros(nx); d[i] = eps
            a[:, :, i] = (self.f_batch(x + d, u) - self.f_batch(x - d, u)) / (2 * eps)
        for j in range(nu):
            d = np.zeros(nu); d[j] = eps
            b[:, :, j] = (self.f_batch(x, u + d) - self.f_batch(x, u - d)) / (2 * eps)
        return a, b

    def rollout_batch(self, x0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll a batch of action sequences. x0 (B,4), actions (B,H,1) -> (B,H+1,4)."""
        x = np.asarray(x0, dtype=np.float64)
        out = [x]
        for t in range(actions.shape[1]):
            x = self.f_batch(x, actions[:, t])
            out.append(x)
        return np.stack(out, axis=1)
