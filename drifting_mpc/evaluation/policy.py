from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from drifting_mpc.data.costs import trajectory_cost_torch
from drifting_mpc.data.dataset import TrajectoryNormalizer
from drifting_mpc.data.trajectory import decode_trajectory_torch
from drifting_mpc.envs.registry import build_spec
from drifting_mpc.methods import get_method_spec
from drifting_mpc.models.generator import ConditionalTrajectoryGenerator


@dataclass
class PlannerSample:
    action: np.ndarray
    predicted_states: np.ndarray
    predicted_actions: np.ndarray
    predicted_cost: float


class LearnedMPCPlanner:
    def __init__(
        self,
        model: ConditionalTrajectoryGenerator,
        normalizer: TrajectoryNormalizer,
        config: dict,
        device: torch.device | str,
        m_plan: int,
    ):
        self.model = model
        self.normalizer = normalizer
        self.config = config
        self.device = torch.device(device)
        self.spec = build_spec(config)
        self.m_plan = int(m_plan)
        self.method = get_method_spec(config)

    @torch.no_grad()
    def act(self, state: np.ndarray, omega: np.ndarray) -> PlannerSample:
        x_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        omega_tensor = torch.as_tensor(omega, dtype=torch.float32, device=self.device).reshape(1, -1)
        eps = torch.randn(self.m_plan, self.model.eps_dim, device=self.device)
        x_batch = x_tensor.repeat(self.m_plan, 1)
        omega_batch = omega_tensor.repeat(self.m_plan, 1)
        if self.method.condition_on_omega:
            zeta = self.model(eps, x_batch, omega_batch)
        else:
            zeta = self.model(eps, x_batch, None)
        predicted_states, predicted_actions = decode_trajectory_torch(
            x_batch,
            zeta,
            horizon=self.spec.horizon,
            state_dim=self.spec.state_dim,
            action_dim=self.spec.action_dim,
        )
        candidate_cost = trajectory_cost_torch(predicted_states, predicted_actions, omega_batch)
        best_index = int(torch.argmin(candidate_cost).item())
        return PlannerSample(
            action=predicted_actions[best_index, 0].detach().cpu().numpy().astype(np.float32),
            predicted_states=predicted_states[best_index].detach().cpu().numpy().astype(np.float32),
            predicted_actions=predicted_actions[best_index].detach().cpu().numpy().astype(np.float32),
            predicted_cost=float(candidate_cost[best_index].detach().cpu()),
        )


def load_planner_from_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str,
    m_plan: int | None = None,
) -> tuple[LearnedMPCPlanner, dict]:
    """Rebuild the receding-horizon planner stored in a training checkpoint."""
    payload = torch.load(Path(ckpt_path).expanduser(), map_location=device, weights_only=False)
    config = payload["config"]
    spec = build_spec(config)
    method = get_method_spec(config)
    model_family = payload.get("model_family", method.model_family)
    normalizer = TrajectoryNormalizer(
        mean=payload["normalizer"]["mean"].astype(np.float32),
        std=payload["normalizer"]["std"].astype(np.float32),
    )
    if model_family == "drifting":
        from drifting_mpc.training.trainer import _build_model as _build_gen
        model = _build_gen(config, spec, device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        planner = LearnedMPCPlanner(
            model=model,
            normalizer=normalizer,
            config=config,
            device=device,
            m_plan=m_plan if m_plan is not None else int(config["evaluation"]["m_plan"]),
        )
        return planner, config
    raise ValueError(f"Unsupported checkpoint model_family: {model_family}")
