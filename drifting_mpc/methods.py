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
