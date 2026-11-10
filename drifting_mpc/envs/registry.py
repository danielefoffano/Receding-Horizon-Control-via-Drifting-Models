"""Environment dispatch so the pipeline is not hard-wired to one system."""
from __future__ import annotations

from drifting_mpc.envs.cartpole import CartPoleEnv, build_cartpole_spec
from drifting_mpc.envs.mass_spring_damper import MassSpringDamperEnv, build_msd_spec
from drifting_mpc.envs.quadrotor import PlanarQuadrotorEnv, build_quadrotor_spec

_BUILDERS = {"msd": (build_msd_spec, MassSpringDamperEnv),
             "mass_spring_damper": (build_msd_spec, MassSpringDamperEnv),
             "cartpole": (build_cartpole_spec, CartPoleEnv),
             "quadrotor": (build_quadrotor_spec, PlanarQuadrotorEnv),
             "planar_quadrotor": (build_quadrotor_spec, PlanarQuadrotorEnv)}


def system_name(config: dict) -> str:
    return str(config.get("environment", {}).get("system", "msd")).strip().lower()


def build_spec(config: dict):
    name = system_name(config)
    if name not in _BUILDERS:
        raise ValueError(f"Unknown environment.system '{name}'. Expected one of: {sorted(_BUILDERS)}.")
    return _BUILDERS[name][0](config)


def make_env(config: dict, spec=None):
    name = system_name(config)
    if name not in _BUILDERS:
        raise ValueError(f"Unknown environment.system '{name}'. Expected one of: {sorted(_BUILDERS)}.")
    builder, env_cls = _BUILDERS[name]
    return env_cls(spec if spec is not None else builder(config))


def make_reference_controller(config: dict, spec=None, env=None):
    """Reference ('oracle') controller for the configured system.

    Exact finite-horizon LQR for the linear mass-spring-damper; iLQR for the
    nonlinear cart-pole, which has no closed-form optimum.
    """
    spec = spec if spec is not None else build_spec(config)
    name = system_name(config)
    if name in ("msd", "mass_spring_damper"):
        from drifting_mpc.control.oracle import OracleLQRController
        return OracleLQRController(spec)
    if name in ("cartpole", "quadrotor", "planar_quadrotor"):
        from drifting_mpc.control.ilqr import ILQRController
        ilqr_cfg = config.get("reference", {}) or {}
        return ILQRController(spec, env if env is not None else make_env(config, spec),
                              max_iters=int(ilqr_cfg.get("max_iters", 30)))
    raise ValueError(f"No reference controller for environment.system '{name}'.")
