from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from drifting_mpc.data.costs import trajectory_cost_torch
from drifting_mpc.data.dataset import TrajectoryNormalizer
from drifting_mpc.data.trajectory import decode_trajectory_torch
from drifting_mpc.envs.mass_spring_damper import build_msd_spec
from drifting_mpc.methods import get_method_spec
from drifting_mpc.models.diffusion import ConditionalTrajectoryDiffusionModel
from drifting_mpc.models.generator import ConditionalTrajectoryGenerator
from drifting_mpc.training.diffusion import DiffusionScheduler, apply_conditioning


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
        self.spec = build_msd_spec(config)
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


class DiffusionMPCPlanner:
    """Receding-horizon planner using Diffuser-style conditioning and ancestral sampling."""

    def __init__(
        self,
        model: ConditionalTrajectoryDiffusionModel,
        normalizer: TrajectoryNormalizer,
        scheduler: DiffusionScheduler,
        config: dict,
        device: torch.device | str,
        m_plan: int,
    ):
        self.model = model
        self.normalizer = normalizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.spec = build_msd_spec(config)
        self.m_plan = int(m_plan)
        self.sample_steps = int(config.get("diffusion", {}).get("sample_steps", scheduler.num_train_steps))
        self.clip_sample = float(config.get("diffusion", {}).get("clip_sample", 5.0))
        self.action_dim = self.spec.action_dim
        self.state_dim = self.spec.state_dim
        self.transition_horizon = self.spec.horizon + 1
        self.transition_dim = self.action_dim + self.state_dim
        mean, std = self.normalizer.as_torch(self.device)
        self.mean_seq = mean.reshape(self.transition_horizon, self.transition_dim)
        self.std_seq = std.reshape(self.transition_horizon, self.transition_dim)

    def _conditions(self, x_batch: torch.Tensor) -> dict[int, torch.Tensor]:
        obs0 = (x_batch - self.mean_seq[0, self.action_dim:]) / self.std_seq[0, self.action_dim:]
        return {0: obs0}

    @torch.no_grad()
    def _sample_normalized_transition_tensor(self, x_batch: torch.Tensor, omega_batch: torch.Tensor) -> torch.Tensor:
        del omega_batch
        cond = self._conditions(x_batch)
        return self.scheduler.sample_ancestral(
            model=self.model,
            shape=(self.m_plan, self.transition_horizon, self.transition_dim),
            cond=cond,
            action_dim=self.action_dim,
            clip_denoised=self.clip_sample,
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, omega: np.ndarray) -> PlannerSample:
        x_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        omega_tensor = torch.as_tensor(omega, dtype=torch.float32, device=self.device).reshape(1, -1)
        x_batch = x_tensor.repeat(self.m_plan, 1)
        omega_batch = omega_tensor.repeat(self.m_plan, 1)
        normalized_transitions = self._sample_normalized_transition_tensor(x_batch, omega_batch)
        flat = normalized_transitions.reshape(self.m_plan, -1)
        transitions = self.normalizer.denormalize_torch(flat).reshape(self.m_plan, self.transition_horizon, self.transition_dim)
        predicted_states = transitions[:, :, self.action_dim:]
        predicted_actions = transitions[:, :-1, :self.action_dim]
        candidate_cost = trajectory_cost_torch(predicted_states, predicted_actions, omega_batch)
        best_index = int(torch.argmin(candidate_cost).item())
        return PlannerSample(
            action=predicted_actions[best_index, 0].detach().cpu().numpy().astype(np.float32),
            predicted_states=predicted_states[best_index].detach().cpu().numpy().astype(np.float32),
            predicted_actions=predicted_actions[best_index].detach().cpu().numpy().astype(np.float32),
            predicted_cost=float(candidate_cost[best_index].detach().cpu()),
        )


class GuidedDiffusionMPCPlanner(DiffusionMPCPlanner):
    def __init__(
        self,
        model: ConditionalTrajectoryDiffusionModel,
        normalizer: TrajectoryNormalizer,
        scheduler: DiffusionScheduler,
        config: dict,
        device: torch.device | str,
        m_plan: int,
    ):
        super().__init__(model=model, normalizer=normalizer, scheduler=scheduler, config=config, device=device, m_plan=m_plan)
        diffusion_cfg = config.get("diffusion", {})
        self.guidance_scale = float(diffusion_cfg.get("guidance_scale", 1.0))
        self.guidance_n_steps = int(diffusion_cfg.get("guidance_n_steps", 1))
        self.guidance_t_stopgrad = int(diffusion_cfg.get("guidance_t_stopgrad", 0))
        self.guidance_scale_grad_by_std = bool(diffusion_cfg.get("guidance_scale_grad_by_std", True))

    @torch.no_grad()
    def _sample_normalized_transition_tensor(self, x_batch: torch.Tensor, omega_batch: torch.Tensor) -> torch.Tensor:
        cond = self._conditions(x_batch)
        return self.scheduler.sample_ancestral(
            model=self.model,
            shape=(self.m_plan, self.transition_horizon, self.transition_dim),
            cond=cond,
            action_dim=self.action_dim,
            clip_denoised=self.clip_sample,
            omega_batch=omega_batch,
            mean_seq=self.mean_seq,
            std_seq=self.std_seq,
            guidance_scale=self.guidance_scale,
            n_guide_steps=self.guidance_n_steps,
            t_stopgrad=self.guidance_t_stopgrad,
            scale_grad_by_std=self.guidance_scale_grad_by_std,
        )


AnyPlanner = Union[LearnedMPCPlanner, DiffusionMPCPlanner, GuidedDiffusionMPCPlanner]


def load_planner_from_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str,
    m_plan: int | None = None,
) -> tuple[AnyPlanner, dict]:
    payload = torch.load(Path(ckpt_path).expanduser(), map_location=device, weights_only=False)
    config = payload["config"]
    spec = build_msd_spec(config)
    method = get_method_spec(config)
    model_family = payload.get("model_family", method.model_family)
    normalizer = TrajectoryNormalizer(
        mean=payload["normalizer"]["mean"].astype(np.float32),
        std=payload["normalizer"]["std"].astype(np.float32),
    )
    if model_family == "drifting":
        model_cfg = config["model"]
        model = ConditionalTrajectoryGenerator(
            horizon=spec.horizon,
            state_dim=spec.state_dim,
            action_dim=spec.action_dim,
            omega_dim=3 if method.condition_on_omega else 0,
            eps_dim=int(model_cfg["eps_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_blocks=int(model_cfg["num_blocks"]),
        ).to(device)
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
    if model_family == "diffusion":
        model_cfg = config["model"]
        diffusion_cfg = config.get("diffusion", {})
        model = ConditionalTrajectoryDiffusionModel(
            horizon_steps=spec.horizon + 1,
            transition_dim=spec.action_dim + spec.state_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_blocks=int(model_cfg["num_blocks"]),
            time_embed_dim=int(diffusion_cfg.get("time_embed_dim", 64)),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        scheduler = DiffusionScheduler(
            num_train_steps=int(diffusion_cfg.get("num_diffusion_steps", 64)),
            beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
            beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
            device=device,
        )
        planner_cls = GuidedDiffusionMPCPlanner if method.variant == "guided_diffusion_behavior_prior" else DiffusionMPCPlanner
        planner = planner_cls(
            model=model,
            normalizer=normalizer,
            scheduler=scheduler,
            config=config,
            device=device,
            m_plan=m_plan if m_plan is not None else int(config["evaluation"]["m_plan"]),
        )
        return planner, config
    raise ValueError(f"Unsupported checkpoint model_family: {model_family}")
