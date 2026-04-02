from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drifting_mpc.data.costs import omega_to_cost_matrices
from drifting_mpc.envs.mass_spring_damper import MassSpringDamperEnv, MassSpringDamperSpec


def finite_horizon_lqr_gains(
    a_disc: np.ndarray,
    b_disc: np.ndarray,
    q_mat: np.ndarray,
    r_mat: np.ndarray,
    horizon: int,
) -> list[np.ndarray]:
    """Compute exact finite-horizon LQR gains by backward Riccati recursion."""
    if horizon <= 0:
        return []
    p_next = q_mat.astype(np.float64)
    gains: list[np.ndarray] = [np.zeros((r_mat.shape[0], a_disc.shape[0]), dtype=np.float64) for _ in range(horizon)]
    a64 = a_disc.astype(np.float64)
    b64 = b_disc.astype(np.float64)
    q64 = q_mat.astype(np.float64)
    r64 = r_mat.astype(np.float64)
    for t in reversed(range(horizon)):
        bt_p_b = b64.T @ p_next @ b64
        system_matrix = r64 + bt_p_b
        gain = np.linalg.solve(system_matrix, b64.T @ p_next @ a64)
        gains[t] = gain.astype(np.float32)
        closed_loop = a64 - b64 @ gain
        p_next = q64 + gain.T @ r64 @ gain + closed_loop.T @ p_next @ closed_loop
    return gains


@dataclass
class OracleLQRController:
    """Oracle benchmark controller with access to the true linear dynamics."""

    spec: MassSpringDamperSpec

    def __post_init__(self) -> None:
        env = MassSpringDamperEnv(self.spec)
        self.a_disc = env.a_disc
        self.b_disc = env.b_disc

    def first_action(self, state: np.ndarray, omega: np.ndarray, horizon: int | None = None) -> np.ndarray:
        planning_horizon = self.spec.horizon if horizon is None else int(horizon)
        q_mat, r_mat = omega_to_cost_matrices(omega)
        gains = finite_horizon_lqr_gains(self.a_disc, self.b_disc, q_mat, r_mat, planning_horizon)
        if not gains:
            return np.zeros(self.spec.action_dim, dtype=np.float32)
        action = -(gains[0] @ np.asarray(state, dtype=np.float32))
        return action.astype(np.float32)

    def rollout(
        self,
        env: MassSpringDamperEnv,
        x0: np.ndarray,
        omega: np.ndarray,
        action_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = env.reset(initial_state=x0)
        states = [state.copy()]
        actions = []
        for _ in range(self.spec.episode_len):
            action = self.first_action(state, omega, horizon=self.spec.horizon)
            if action_noise_std > 0.0:
                if rng is None:
                    raise ValueError("An RNG is required for noisy oracle rollouts.")
                action = (action + rng.normal(0.0, action_noise_std, size=action.shape)).astype(np.float32)
            state = env.step(action)
            actions.append(action.copy())
            states.append(state.copy())
        return np.stack(states), np.stack(actions)
