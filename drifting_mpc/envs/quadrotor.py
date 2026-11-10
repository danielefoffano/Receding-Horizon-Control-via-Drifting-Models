"""Planar quadrotor benchmark: 6-D state, 2-D input.

State x = (p_x, v_x, p_z, v_z, theta, omega), ordered position/rate-interleaved
to match the mass-spring-damper and cart-pole conventions so the generic
behaviour policies apply unchanged.

The action is the thrust DEVIATION from hover, u = (du_L, du_R), so that the
origin (x = 0, u = 0) is an equilibrium and the same diagonal quadratic cost
used everywhere else applies with omega = [q (6), r (2)].

    T  = m g + du_L + du_R          (total thrust)
    px'' = -T sin(theta) / m
    pz'' =  T cos(theta) / m - g
    theta'' = arm (du_L - du_R) / I

Underactuated (2 inputs, 3 configuration variables) and unstable in pitch.
Integrated with RK4; the reference controller is iLQR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarQuadrotorSpec:
    mass: float
    inertia: float
    arm: float
    gravity: float
    dt: float
    horizon: int
    episode_len: int
    init_low: np.ndarray
    init_high: np.ndarray
    thrust_limit: float = 0.0   # >0 saturates |du| per rotor
    state_clip: float = 1e4     # numerical guard for divergent open-loop data

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 2


def build_quadrotor_spec(config: dict) -> PlanarQuadrotorSpec:
    env_cfg = config["environment"]
    init = env_cfg["initial_state"]
    keys = ["position_x", "velocity_x", "position_z", "velocity_z", "angle", "angular_velocity"]
    low = np.array([init[k][0] for k in keys], dtype=np.float32)
    high = np.array([init[k][1] for k in keys], dtype=np.float32)
    return PlanarQuadrotorSpec(
        mass=float(env_cfg.get("mass", 1.0)),
        inertia=float(env_cfg.get("inertia", 0.05)),
        arm=float(env_cfg.get("arm", 0.25)),
        gravity=float(env_cfg.get("gravity", 9.81)),
        dt=float(env_cfg["dt"]),
        horizon=int(env_cfg["horizon"]),
        episode_len=int(env_cfg["episode_len"]),
        init_low=low,
        init_high=high,
        thrust_limit=float(env_cfg.get("thrust_limit", 0.0)),
        state_clip=float(env_cfg.get("state_clip", 1e4)),
    )


class PlanarQuadrotorEnv:
    def __init__(self, spec: PlanarQuadrotorSpec):
        self.spec = spec
        self.state = np.zeros(spec.state_dim, dtype=np.float32)
        self.t = 0

    # --- dynamics (batched core) ----------------------------------------
    def _deriv(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        m, inertia, arm, g = self.spec.mass, self.spec.inertia, self.spec.arm, self.spec.gravity
        vx = state[..., 1]; vz = state[..., 3]
        th = state[..., 4]; om = state[..., 5]
        du_l = u[..., 0]; du_r = u[..., 1]
        thrust = m * g + du_l + du_r
        ax = -thrust * np.sin(th) / m
        az = thrust * np.cos(th) / m - g
        alpha = arm * (du_l - du_r) / inertia
        return np.stack([vx, ax, vz, az, om, alpha], axis=-1)

    def f_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float64)
        u = np.asarray(actions, dtype=np.float64)
        if self.spec.thrust_limit > 0.0:
            u = np.clip(u, -self.spec.thrust_limit, self.spec.thrust_limit)
        dt = self.spec.dt
        k1 = self._deriv(x, u)
        k2 = self._deriv(x + 0.5 * dt * k1, u)
        k3 = self._deriv(x + 0.5 * dt * k2, u)
        k4 = self._deriv(x + dt * k3, u)
        nxt = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.spec.state_clip > 0.0:
            nxt = np.clip(nxt, -self.spec.state_clip, self.spec.state_clip)
        return nxt

    def f(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(self.spec.state_dim)
        u = np.asarray(action, dtype=np.float64).reshape(self.spec.action_dim)
        return self.f_batch(x, u).astype(np.float32)

    # --- env interface ---------------------------------------------------
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

    def step(self, action: np.ndarray) -> np.ndarray:
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

    # --- iLQR support -----------------------------------------------------
    def linearize_traj(self, states: np.ndarray, actions: np.ndarray, eps: float = 1e-5):
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
        x = np.asarray(x0, dtype=np.float64)
        out = [x]
        for t in range(actions.shape[1]):
            x = self.f_batch(x, actions[:, t])
            out.append(x)
        return np.stack(out, axis=1)
