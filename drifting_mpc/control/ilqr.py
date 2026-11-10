"""iLQR reference controller for nonlinear systems.

The mass-spring-damper oracle is an exact finite-horizon LQR. No such closed form
exists once the dynamics are nonlinear, so the reference here is iLQR run in the
same receding-horizon fashion: re-solved at every step, warm-started from the
previous solution. It is a strong reference, not a certified optimum -- report it
as such.
"""
from __future__ import annotations

import numpy as np

from drifting_mpc.data.costs import omega_to_cost_matrices


class ILQRController:
    def __init__(self, spec, env, max_iters: int = 50, tol: float = 1e-6,
                 reg_init: float = 1e-6, reg_max: float = 1e8, warm_start: bool = True):
        self.spec = spec
        self.env = env
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.reg_init = float(reg_init)
        self.reg_max = float(reg_max)
        self.warm_start = bool(warm_start)
        self._u_prev: np.ndarray | None = None

    # --- helpers --------------------------------------------------------
    def _rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).reshape(1, -1)
        return self.env.rollout_batch(x, u_seq[None, ...])[0]

    @staticmethod
    def _cost(x_seq: np.ndarray, u_seq: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        stage = np.einsum("ti,ij,tj->", x_seq[:-1], q, x_seq[:-1])
        stage += np.einsum("ti,ij,tj->", u_seq, r, u_seq)
        return float(stage + x_seq[-1] @ q @ x_seq[-1])

    # --- core solve -----------------------------------------------------
    def solve(self, x0: np.ndarray, omega: np.ndarray, horizon: int,
              u_init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        nx, nu = self.spec.state_dim, self.spec.action_dim
        q, r = omega_to_cost_matrices(omega, nx, nu)
        q = q.astype(np.float64); r = r.astype(np.float64)
        u = np.zeros((horizon, nu)) if u_init is None else np.array(u_init, dtype=np.float64).reshape(horizon, nu)
        x = self._rollout(x0, u)
        cost = self._cost(x, u, q, r)
        reg = self.reg_init

        for _ in range(self.max_iters):
            a_mats, b_mats = self.env.linearize_traj(x[:-1], u)
            v_x = 2.0 * q @ x[-1]
            v_xx = 2.0 * q
            k_ff = np.zeros((horizon, nu))
            k_fb = np.zeros((horizon, nu, nx))
            diverged = False

            for t in reversed(range(horizon)):
                a, b = a_mats[t], b_mats[t]
                q_x = 2.0 * q @ x[t] + a.T @ v_x
                q_u = 2.0 * r @ u[t] + b.T @ v_x
                q_xx = 2.0 * q + a.T @ v_xx @ a
                q_uu = 2.0 * r + b.T @ v_xx @ b + reg * np.eye(nu)
                q_ux = b.T @ v_xx @ a
                try:
                    q_uu_inv = np.linalg.inv(q_uu)
                except np.linalg.LinAlgError:
                    diverged = True
                    break
                k_ff[t] = -q_uu_inv @ q_u
                k_fb[t] = -q_uu_inv @ q_ux
                v_x = q_x + k_fb[t].T @ q_uu @ k_ff[t] + k_fb[t].T @ q_u + q_ux.T @ k_ff[t]
                v_xx = q_xx + k_fb[t].T @ q_uu @ k_fb[t] + k_fb[t].T @ q_ux + q_ux.T @ k_fb[t]
                v_xx = 0.5 * (v_xx + v_xx.T)

            if diverged:
                reg = min(reg * 10.0, self.reg_max)
                if reg >= self.reg_max:
                    break
                continue

            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
                x_new = np.zeros_like(x)
                u_new = np.zeros_like(u)
                x_new[0] = x[0]
                for t in range(horizon):
                    u_new[t] = u[t] + alpha * k_ff[t] + k_fb[t] @ (x_new[t] - x[t])
                    x_new[t + 1] = self.env.f_batch(x_new[t][None, :], u_new[t][None, :])[0]
                cost_new = self._cost(x_new, u_new, q, r)
                if np.isfinite(cost_new) and cost_new < cost:
                    improved = True
                    break
            if not improved:
                reg = min(reg * 10.0, self.reg_max)
                if reg >= self.reg_max:
                    break
                continue
            reg = max(reg / 10.0, self.reg_init)
            if abs(cost - cost_new) < self.tol * max(abs(cost), 1.0):
                x, u, cost = x_new, u_new, cost_new
                break
            x, u, cost = x_new, u_new, cost_new
        return u, x, cost

    # --- OracleLQRController-compatible interface ------------------------
    def reset_warm_start(self) -> None:
        self._u_prev = None

    def first_action(self, state: np.ndarray, omega: np.ndarray, horizon: int | None = None) -> np.ndarray:
        h = self.spec.horizon if horizon is None else int(horizon)
        u_init = None
        if self.warm_start and self._u_prev is not None and len(self._u_prev) == h:
            u_init = np.vstack([self._u_prev[1:], self._u_prev[-1:]])
        u, _, _ = self.solve(state, omega, h, u_init=u_init)
        self._u_prev = u
        return u[0].astype(np.float32)

    def rollout(self, env, x0: np.ndarray, omega: np.ndarray,
                action_noise_std: float = 0.0, rng: np.random.Generator | None = None):
        self.reset_warm_start()
        state = env.reset(initial_state=x0)
        states, actions = [state.copy()], []
        for _ in range(self.spec.episode_len):
            action = self.first_action(state, omega, horizon=self.spec.horizon)
            if action_noise_std > 0.0:
                if rng is None:
                    raise ValueError("An RNG is required for noisy reference rollouts.")
                action = (action + rng.normal(0.0, action_noise_std, size=action.shape)).astype(np.float32)
            state = env.step(action)
            actions.append(action.copy())
            states.append(state.copy())
        return np.stack(states), np.stack(actions)
