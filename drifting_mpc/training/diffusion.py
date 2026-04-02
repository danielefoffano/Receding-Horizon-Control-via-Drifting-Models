from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from drifting_mpc.data.dataset import TorchOfflineTrajectorySplit, TrajectoryNormalizer
from drifting_mpc.methods import MethodSpec, get_method_spec


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Cosine beta schedule as used in Diffuser / Nichol & Dhariwal."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=dtype, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=0.0, max=0.999)


@dataclass
class DiffusionBatchStatistics:
    loss: float


def build_transition_tensor_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Build a Diffuser-style trajectory tensor.

    Each timestep stores [action_t, state_t]. A dummy zero action is appended at
    the terminal state so the tensor has shape [batch, H+1, action_dim + state_dim].
    """
    batch, horizon, action_dim = actions.shape
    final_action = torch.zeros(batch, 1, action_dim, device=actions.device, dtype=actions.dtype)
    padded_actions = torch.cat([actions, final_action], dim=1)
    return torch.cat([padded_actions, states], dim=-1)


def apply_conditioning(x: torch.Tensor, conditions: dict[int, torch.Tensor], action_dim: int) -> torch.Tensor:
    for timestep, value in conditions.items():
        x[:, timestep, action_dim:] = value.clone()
    return x


class TrajectoryDiffusionObjective:
    """DDPM-style objective over normalized trajectory tensors."""

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
        del beta_start, beta_end
        self.betas = cosine_beta_schedule(self.num_diffusion_steps, device=self.device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.transition_horizon = split.actions.shape[1] + 1
        self.action_dim = split.actions.shape[-1]
        self.state_dim = split.states.shape[-1]
        self.transition_dim = self.action_dim + self.state_dim
        self.mean, self.std = normalizer.as_torch(self.device)
        self.mean_seq = self.mean.reshape(self.transition_horizon, self.transition_dim)
        self.std_seq = self.std.reshape(self.transition_horizon, self.transition_dim)
        self.transition_tensor = build_transition_tensor_torch(self.split.states, self.split.actions)
        flat = self.transition_tensor.reshape(self.split.size, -1)
        self.normalized_transition_tensor = ((flat - self.mean) / self.std).reshape(
            self.split.size, self.transition_horizon, self.transition_dim
        )
        self.cond_state0 = (self.split.x0[:1] - self.mean_seq[0, self.action_dim:]) / self.std_seq[0, self.action_dim:]
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(config["experiment"]["seed"]))

    def _condition_values(self, x0_batch: torch.Tensor) -> dict[int, torch.Tensor]:
        value = (x0_batch - self.mean_seq[0, self.action_dim:]) / self.std_seq[0, self.action_dim:]
        return {0: value}

    def sample_batch(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.batch_size if batch_size is None else int(batch_size)
        indices = torch.randint(0, self.split.size, (size,), generator=self.generator, device=self.device)
        return self.split.x0[indices], self.split.omega[indices], self.normalized_transition_tensor[indices]

    def compute_loss(
        self,
        model: torch.nn.Module,
        x0_batch: torch.Tensor,
        omega_batch: torch.Tensor,
        normalized_transition_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, DiffusionBatchStatistics]:
        del omega_batch
        batch_size = normalized_transition_batch.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(batch_size,),
            generator=self.generator,
            device=self.device,
        )
        noise = torch.randn(
            normalized_transition_batch.shape,
            generator=self.generator,
            device=self.device,
            dtype=normalized_transition_batch.dtype,
        )
        alpha_bar_t = self.alpha_bars[timesteps].view(-1, 1, 1)
        x_noisy = torch.sqrt(alpha_bar_t) * normalized_transition_batch + torch.sqrt(1.0 - alpha_bar_t) * noise
        cond = self._condition_values(x0_batch)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        x_recon = model(x_noisy, timesteps, None, None)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)
        loss = torch.mean((x_recon - noise) ** 2)
        return loss, DiffusionBatchStatistics(loss=float(loss.detach().cpu()))


class DiffusionScheduler:
    def __init__(self, num_train_steps: int, beta_start: float, beta_end: float, device: torch.device | str):
        self.device = torch.device(device)
        self.num_train_steps = int(num_train_steps)
        del beta_start, beta_end
        self.betas = cosine_beta_schedule(self.num_train_steps, device=self.device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def _reward_gradient_on_normalized_transitions(
        x: torch.Tensor,
        omega_batch: torch.Tensor,
        mean_seq: torch.Tensor,
        std_seq: torch.Tensor,
        action_dim: int,
    ) -> torch.Tensor:
        """Closed-form gradient of cumulative reward w.r.t. normalized transition tensor."""
        mean = mean_seq.unsqueeze(0)
        std = std_seq.unsqueeze(0)
        transitions = x * std + mean
        actions = transitions[:, :-1, :action_dim]
        states = transitions[:, :, action_dim:]
        q_pos = omega_batch[:, 0].view(-1, 1)
        q_vel = omega_batch[:, 1].view(-1, 1)
        r_u = omega_batch[:, 2].view(-1, 1, 1)
        grad = torch.zeros_like(transitions)
        grad[:, :-1, :action_dim] = -2.0 * r_u * actions
        grad[:, :, action_dim + 0] = -2.0 * q_pos * states[:, :, 0]
        grad[:, :, action_dim + 1] = -2.0 * q_vel * states[:, :, 1]
        return grad * std

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_recon = self.predict_start_from_noise(x, t=t, noise=model(x, t, None, None))
        if clip_denoised > 0.0:
            x_recon = x_recon.clamp(-clip_denoised, clip_denoised)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def _default_p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        cond: dict[int, torch.Tensor],
        t: torch.Tensor,
        action_dim: int,
        clip_denoised: float,
    ) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(model, x=x, t=t, clip_denoised=clip_denoised)
        model_std = torch.exp(0.5 * model_log_variance)
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        out = model_mean + model_std * noise
        return apply_conditioning(out, cond, action_dim)

    @torch.no_grad()
    def _guided_p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        cond: dict[int, torch.Tensor],
        t: torch.Tensor,
        action_dim: int,
        clip_denoised: float,
        omega_batch: torch.Tensor,
        mean_seq: torch.Tensor,
        std_seq: torch.Tensor,
        scale: float,
        n_guide_steps: int,
        t_stopgrad: int,
        scale_grad_by_std: bool,
    ) -> torch.Tensor:
        model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
        for _ in range(n_guide_steps):
            grad = self._reward_gradient_on_normalized_transitions(x, omega_batch, mean_seq, std_seq, action_dim)
            if scale_grad_by_std:
                grad = model_var * grad
            grad[t < t_stopgrad] = 0
            x = x + scale * grad
            x = apply_conditioning(x, cond, action_dim)
        model_mean, _, _ = self.p_mean_variance(model, x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        out = model_mean + model_std * noise
        return apply_conditioning(out, cond, action_dim)

    @torch.no_grad()
    def sample_ancestral(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int],
        cond: dict[int, torch.Tensor],
        action_dim: int,
        clip_denoised: float,
        *,
        omega_batch: Optional[torch.Tensor] = None,
        mean_seq: Optional[torch.Tensor] = None,
        std_seq: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        n_guide_steps: int = 1,
        t_stopgrad: int = 0,
        scale_grad_by_std: bool = True,
    ) -> torch.Tensor:
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        x = apply_conditioning(x, cond, action_dim)
        guided = guidance_scale > 0 and omega_batch is not None and mean_seq is not None and std_seq is not None
        for i in reversed(range(0, self.num_train_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if guided:
                x = self._guided_p_sample(
                    model=model,
                    x=x,
                    cond=cond,
                    t=t,
                    action_dim=action_dim,
                    clip_denoised=clip_denoised,
                    omega_batch=omega_batch,
                    mean_seq=mean_seq,
                    std_seq=std_seq,
                    scale=guidance_scale,
                    n_guide_steps=n_guide_steps,
                    t_stopgrad=t_stopgrad,
                    scale_grad_by_std=scale_grad_by_std,
                )
            else:
                x = self._default_p_sample(model=model, x=x, cond=cond, t=t, action_dim=action_dim, clip_denoised=clip_denoised)
        return x
