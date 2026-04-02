from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LTISystem:
    name: str
    A: np.ndarray
    B: np.ndarray
    u_max: float
    x0_scale: np.ndarray
    horizon: int

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.B.shape[1]

    def sample_x0(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.x0_scale, self.x0_scale).astype(np.float32)

    def step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator | None = None, process_noise_std: float = 0.0) -> np.ndarray:
        u = np.clip(u, -self.u_max, self.u_max).astype(np.float32)
        x_next = (self.A @ x + self.B @ u).astype(np.float32)
        if process_noise_std > 0.0:
            if rng is None:
                raise ValueError('rng must be provided when process_noise_std > 0.0')
            x_next = (x_next + rng.normal(0.0, process_noise_std, size=(self.nx,))).astype(np.float32)
        return x_next

    def rollout(
        self,
        x0: np.ndarray,
        policy,
        rng: np.random.Generator,
        process_noise_std: float,
        action_noise_std: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = x0.astype(np.float32).copy()
        states = [x.copy()]
        actions = []
        for t in range(self.horizon):
            u = np.asarray(policy(t, x), dtype=np.float32)
            if action_noise_std > 0.0:
                u = (u + rng.normal(0.0, action_noise_std, size=(self.nu,))).astype(np.float32)
            u = np.clip(u, -self.u_max, self.u_max).astype(np.float32)
            x = self.step(x, u, rng=rng, process_noise_std=process_noise_std)
            actions.append(u.copy())
            states.append(x.copy())
        return np.stack(states), np.stack(actions)


def make_systems(horizon: int = 18, dt: float =0.2) -> dict[str, LTISystem]:
    systems: dict[str, LTISystem] = {}

    Umax = 3
    T = 4
    q0 = Umax * (T ** 2) / 2
    v0 = Umax * T

    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
    B = np.array([[0.5 * dt * dt], [dt]], dtype=np.float32)
    systems['double_integrator'] = LTISystem('double_integrator', A, B, Umax, np.array([q0, v0], dtype=np.float32), horizon)

    k, c, m = 0.6, 0.15, 1
    q0 = 0.8 * Umax /k 
    v0 = 0.5 * Umax * T / m
    A = np.array([[1.0, dt], [-k * dt, 1.0 - c * dt]], dtype=np.float32)
    B = np.array([[0.0], [dt]], dtype=np.float32)
    systems['mass_spring_damper'] = LTISystem('mass_spring_damper', A, B, Umax, np.array([q0, v0], dtype=np.float32), horizon)

    k, c = 0.6, 0.15
    m = 1.0
    alpha = c / (2.0 * m)
    wd = np.sqrt(k / m - alpha**2)

    exp_term = np.exp(-alpha * dt)
    cos_term = np.cos(wd * dt)
    sin_term = np.sin(wd * dt)

    A = exp_term * np.array([
        [cos_term + (alpha / wd) * sin_term, (1.0 / wd) * sin_term],
        [-(k / m) / wd * sin_term, cos_term - (alpha / wd) * sin_term]
    ], dtype=np.float32)

    B = np.array([
        [(1.0 / k) * (1.0 - exp_term * (cos_term + (alpha / wd) * sin_term))],
        [exp_term * (sin_term / (m * wd))]
    ], dtype=np.float32)

    systems['mass_spring_damper_exact'] = LTISystem('mass_spring_damper_exact', A, B, Umax, np.array([q0, v0], dtype=np.float32), horizon)



    m = 1
    p0 = Umax * (T ** 2) / (m * 4)
    v0 = Umax * T / (2 * m)
    A1 = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
    B1 = np.array([[0.5 * dt * dt / m], [dt / m]], dtype=np.float32)
    A = np.block([[A1, np.zeros_like(A1)], [np.zeros_like(A1), A1]]).astype(np.float32)
    B = np.block([[B1, np.zeros_like(B1)], [np.zeros_like(B1), B1]]).astype(np.float32)
    systems['point_mass_2d'] = LTISystem('point_mass_2d', A, B, Umax, np.array([p0, v0, p0, v0], dtype=np.float32), horizon)
    
    
    
    
    
    return systems