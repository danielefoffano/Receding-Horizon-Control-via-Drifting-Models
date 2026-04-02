from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from drifting_mpc.data.dataset import TorchOfflineTrajectorySplit, TrajectoryNormalizer
from drifting_mpc.methods import MethodSpec, get_method_spec


@dataclass
class DiffusionBatchStatistics:
    loss: float


class TrajectoryDiffusionObjective:
    """DDPM-style objective over normalized relative trajectories."""

    def __init__(
        self,
        split: TorchOfflineTrajectorySplit,
        normalizer: TrajectoryNormalizer,
        config: dict,
        device: torch.device | str,
    ):
        self.split = split
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.method: MethodSpec = get_method_spec(config)
        train_cfg = config["training"]
        diffusion_cfg = config.get("diffusion", {})
        self.batch_size = int(train_cfg["batch_size"])
        self.num_diffusion_steps = int(diffusion_cfg.get("num_diffusion_steps", 64))
        beta_start = float(diffusion_cfg.get("beta_start", 1e-4))
        beta_end = float(diffusion_cfg.get("beta_end", 0.02))
        self.betas = torch.linspace(beta_start, beta_end, self.num_diffusion_steps, device=self.device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.mean, self.std = normalizer.as_torch(self.device)
        self.normalized_split_zeta = (self.split.zeta - self.mean) / self.std
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(config["experiment"]["seed"]))

    def sample_batch(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.batch_size if batch_size is None else int(batch_size)
        indices = torch.randint(0, self.split.size, (size,), generator=self.generator, device=self.device)
        return self.split.x0[indices], self.split.omega[indices], self.normalized_split_zeta[indices]

    def compute_loss(
        self,
        model: torch.nn.Module,
        x0_batch: torch.Tensor,
        omega_batch: torch.Tensor,
        normalized_zeta_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, DiffusionBatchStatistics]:
        del omega_batch  # diffusion-based priors in this codebase do not condition on omega during training.
        batch_size = normalized_zeta_batch.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(batch_size,),
            generator=self.generator,
            device=self.device,
        )
        noise = torch.randn(
            normalized_zeta_batch.shape,
            generator=self.generator,
            device=self.device,
            dtype=normalized_zeta_batch.dtype,
        )
        alpha_bar_t = self.alpha_bars[timesteps].unsqueeze(-1)
        noisy_zeta = torch.sqrt(alpha_bar_t) * normalized_zeta_batch + torch.sqrt(1.0 - alpha_bar_t) * noise
        pred_noise = model(noisy_zeta, timesteps, x0_batch, None)
        loss = torch.mean((pred_noise - noise) ** 2)
        return loss, DiffusionBatchStatistics(loss=float(loss.detach().cpu()))


class DiffusionScheduler:
    def __init__(self, num_train_steps: int, beta_start: float, beta_end: float, device: torch.device | str):
        self.device = torch.device(device)
        self.num_train_steps = int(num_train_steps)
        self.betas = torch.linspace(beta_start, beta_end, self.num_train_steps, device=self.device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _normalized_reward_gradient(
        pred_x0_normalized: torch.Tensor,
        x0_batch: torch.Tensor,
        omega_batch: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        horizon: int,
        state_dim: int,
        action_dim: int,
    ) -> torch.Tensor:
        """Closed-form gradient of cumulative reward w.r.t. normalized relative trajectory.

        The reward is the negative quadratic trajectory cost. In the relative-coordinate
        representation zeta = (u_0, dx_1, ..., u_{H-1}, dx_H), each action contributes
        2 * r_u * u_t to the cost gradient and each future state offset contributes
        2 * Q * x_t. Converting from zeta-space to normalized zeta-space multiplies by std.
        """
        if action_dim != 1 or state_dim != 2:
            raise NotImplementedError(
                "Closed-form guidance currently supports the mass-spring-damper setting with action_dim=1 and state_dim=2."
            )
        zeta = pred_x0_normalized * std + mean
        blocks = zeta.reshape(-1, horizon, action_dim + state_dim)
        actions = blocks[..., :, :action_dim]
        deltas = blocks[..., :, action_dim:]
        future_states = x0_batch.unsqueeze(1) + deltas

        q_pos = omega_batch[:, 0].unsqueeze(-1)
        q_vel = omega_batch[:, 1].unsqueeze(-1)
        r_u = omega_batch[:, 2].unsqueeze(-1)

        grad_actions = 2.0 * r_u.unsqueeze(-1) * actions
        grad_deltas = torch.empty_like(deltas)
        grad_deltas[..., 0] = 2.0 * q_pos * future_states[..., 0]
        grad_deltas[..., 1] = 2.0 * q_vel * future_states[..., 1]

        grad_blocks = torch.cat([grad_actions, grad_deltas], dim=-1)
        grad_zeta = grad_blocks.reshape(pred_x0_normalized.shape)
        grad_normalized_cost = grad_zeta * std
        grad_normalized_reward = -grad_normalized_cost
        return grad_normalized_reward

    def sample_ddim(
        self,
        model: torch.nn.Module,
        x0_batch: torch.Tensor,
        num_samples: int,
        trajectory_dim: int,
        sample_steps: int,
        clip_sample: float = 5.0,
        *,
        omega_batch: Optional[torch.Tensor] = None,
        normalizer: Optional[TrajectoryNormalizer] = None,
        horizon: Optional[int] = None,
        state_dim: int = 2,
        action_dim: int = 1,
        guidance_scale: float = 0.0,
        guidance_norm: bool = True,
        guidance_max_step: float = 0.0,
        guidance_start_fraction: float = 0.0,
        guidance_end_fraction: float = 1.0,
    ) -> torch.Tensor:
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive.")
        x_t = torch.randn(num_samples, trajectory_dim, device=self.device, dtype=torch.float32)
        x0_rep = x0_batch
        if x0_rep.shape[0] != num_samples:
            raise ValueError("x0_batch must have shape [num_samples, state_dim].")
        if sample_steps >= self.num_train_steps:
            timesteps = torch.arange(self.num_train_steps - 1, -1, -1, device=self.device, dtype=torch.long)
        else:
            timesteps = torch.linspace(self.num_train_steps - 1, 0, sample_steps, device=self.device).round().to(torch.long)
            timesteps = torch.unique_consecutive(timesteps)
        timesteps_list = timesteps.tolist()

        mean_t: Optional[torch.Tensor] = None
        std_t: Optional[torch.Tensor] = None
        if normalizer is not None:
            mean_t, std_t = normalizer.as_torch(self.device)

        apply_guidance = (
            guidance_scale > 0.0
            and omega_batch is not None
            and mean_t is not None
            and std_t is not None
            and horizon is not None
        )
        num_sampling_steps = len(timesteps_list)
        start_index = int(round(guidance_start_fraction * max(num_sampling_steps - 1, 0)))
        end_index = int(round(guidance_end_fraction * max(num_sampling_steps - 1, 0)))
        start_index = max(0, min(start_index, num_sampling_steps - 1))
        end_index = max(start_index, min(end_index, num_sampling_steps - 1))

        for index, t_int in enumerate(timesteps_list):
            t = int(t_int)
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            pred_noise = model(x_t, t_batch, x0_rep, None)
            alpha_bar_t = self.alpha_bars[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
            if clip_sample > 0.0:
                pred_x0 = pred_x0.clamp(min=-clip_sample, max=clip_sample)

            if apply_guidance and start_index <= index <= end_index:
                reward_grad = self._normalized_reward_gradient(
                    pred_x0_normalized=pred_x0,
                    x0_batch=x0_rep,
                    omega_batch=omega_batch,
                    mean=mean_t,
                    std=std_t,
                    horizon=horizon,
                    state_dim=state_dim,
                    action_dim=action_dim,
                )
                if guidance_norm:
                    reward_grad = reward_grad / reward_grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                step_scale = guidance_scale * float(1.0 - alpha_bar_t)
                if guidance_max_step > 0.0:
                    step_scale = min(step_scale, guidance_max_step)
                pred_x0 = pred_x0 + step_scale * reward_grad
                if clip_sample > 0.0:
                    pred_x0 = pred_x0.clamp(min=-clip_sample, max=clip_sample)

            if index == len(timesteps_list) - 1:
                x_t = pred_x0
                continue
            t_prev = int(timesteps_list[index + 1])
            alpha_bar_prev = self.alpha_bars[t_prev]
            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        return x_t
