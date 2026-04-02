from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class MethodSpec:
    variant: str
    condition_on_omega: bool
    use_cost_tilt: bool
    model_family: str
    description: str


_METHODS: Final[dict[str, MethodSpec]] = {
    "cost_aware": MethodSpec(
        variant="cost_aware",
        condition_on_omega=True,
        use_cost_tilt=True,
        model_family="drifting",
        description="Cost-aware drifting with omega-conditioned generator and exponentially tilted positives.",
    ),
    "cost_conditioned_prior": MethodSpec(
        variant="cost_conditioned_prior",
        condition_on_omega=True,
        use_cost_tilt=False,
        model_family="drifting",
        description="Cost-conditioned behavior prior: omega-conditioned generator without exponential cost tilt.",
    ),
    "behavior_prior": MethodSpec(
        variant="behavior_prior",
        condition_on_omega=False,
        use_cost_tilt=False,
        model_family="drifting",
        description="True behavior prior: generator conditioned only on x0, with cost used only at evaluation time.",
    ),
    "diffusion_behavior_prior": MethodSpec(
        variant="diffusion_behavior_prior",
        condition_on_omega=False,
        use_cost_tilt=False,
        model_family="diffusion",
        description="Diffusion behavior prior: x0-conditioned diffusion model trained without omega, with cost used only at evaluation time.",
    ),
    "guided_diffusion_behavior_prior": MethodSpec(
        variant="guided_diffusion_behavior_prior",
        condition_on_omega=False,
        use_cost_tilt=False,
        model_family="diffusion",
        description="Guided diffusion behavior prior: x0-conditioned diffusion model trained without omega and guided at sampling time by the analytical gradient of cumulative reward.",
    ),
}


def get_method_variant(config: dict | None) -> str:
    if config is None:
        return "cost_aware"
    method_cfg = config.get("method", {})
    variant = str(method_cfg.get("variant", "cost_aware")).strip().lower()
    if variant not in _METHODS:
        valid = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method variant '{variant}'. Expected one of: {valid}.")
    return variant



def get_method_spec(config: dict | None) -> MethodSpec:
    return _METHODS[get_method_variant(config)]



def set_method_variant(config: dict, variant: str) -> None:
    variant = variant.strip().lower()
    if variant not in _METHODS:
        valid = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method variant '{variant}'. Expected one of: {valid}.")
    config.setdefault("method", {})["variant"] = variant
