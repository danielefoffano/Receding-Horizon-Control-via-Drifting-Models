from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class MassSpringDamperSpec:
    """Static configuration for the 1D mass-spring-damper benchmark."""

    m: float
    k_s: float
    c: float
    dt: float
    horizon: int
    episode_len: int
    init_low: np.ndarray
    init_high: np.ndarray

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 1


def discretize_linear_system(a_cont: np.ndarray, b_cont: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order hold discretization for a continuous-time linear system."""
    nx = a_cont.shape[0]
    nu = b_cont.shape[1]
    augmented = np.zeros((nx + nu, nx + nu), dtype=np.float64)
    augmented[:nx, :nx] = a_cont
    augmented[:nx, nx:] = b_cont
    matrix_exp = expm(augmented * dt)
    a_disc = matrix_exp[:nx, :nx].astype(np.float32)
    b_disc = matrix_exp[:nx, nx:].astype(np.float32)
    return a_disc, b_disc


def build_msd_spec(config: dict) -> MassSpringDamperSpec:
    env_cfg = config["environment"]
    init_cfg = env_cfg["initial_state"]
    return MassSpringDamperSpec(
        m=float(env_cfg["m"]),
        k_s=float(env_cfg["k_s"]),
        c=float(env_cfg["c"]),
        dt=float(env_cfg["dt"]),
        horizon=int(env_cfg["horizon"]),
        episode_len=int(env_cfg["episode_len"]),
        init_low=np.array([init_cfg["position"][0], init_cfg["velocity"][0]], dtype=np.float32),
        init_high=np.array([init_cfg["position"][1], init_cfg["velocity"][1]], dtype=np.float32),
    )


class MassSpringDamperEnv:
    """Deterministic discrete-time mass-spring-damper environment."""

    def __init__(self, spec: MassSpringDamperSpec):
        self.spec = spec
        a_cont = np.array(
            [[0.0, 1.0], [-spec.k_s / spec.m, -spec.c / spec.m]],
            dtype=np.float64,
        )
        b_cont = np.array([[0.0], [1.0 / spec.m]], dtype=np.float64)
        self.a_disc, self.b_disc = discretize_linear_system(a_cont, b_cont, spec.dt)
        self.state = np.zeros(spec.state_dim, dtype=np.float32)
        self.t = 0

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.spec.init_low, self.spec.init_high).astype(np.float32)

    def reset(self, initial_state: np.ndarray | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        if initial_state is None:
            if rng is None:
                raise ValueError("An RNG is required when initial_state is not provided.")
            initial_state = self.sample_initial_state(rng)
        self.state = np.asarray(initial_state, dtype=np.float32).copy()
        self.t = 0
        return self.state.copy()

    def step(self, action: np.ndarray | float) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32).reshape(self.spec.action_dim)
        self.state = (self.a_disc @ self.state + self.b_disc @ action_array).astype(np.float32)
        self.t += 1
        return self.state.copy()

    def rollout_open_loop(self, x0: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.reset(initial_state=x0)
        states = [self.state.copy()]
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.spec.action_dim)
        for action in actions[: self.spec.episode_len]:
            states.append(self.step(action))
        return np.stack(states), actions[: self.spec.episode_len].copy()
