# Drifting MPC on Mass-Spring-Damper

This repository implements five offline trajectory-generation baselines for the mass-spring-damper system under unknown dynamics:

1. **cost_aware**: Alternative A from the paper. The generator is conditioned on `(x0, omega)` and the positive field uses exponential cost tilting.
2. **cost_conditioned_prior**: the generator is still conditioned on `(x0, omega)`, but the positive field does **not** use exponential cost tilting.
3. **behavior_prior**: the generator is conditioned only on `x0`; the query cost `omega` is used only at evaluation time to score sampled trajectories.
4. **diffusion_behavior_prior**: a diffusion baseline conditioned only on `x0`; it does **not** see `omega` during training and uses unguided diffusion sampling at test time before scoring candidates with the query cost.
5. **guided_diffusion_behavior_prior**: the same `x0`-conditioned diffusion prior, but sampled with Diffuser-style reward guidance using the closed-form gradient of the cumulative reward (equivalently, negative cost) with respect to the predicted denoised trajectory.

All five variants:
- use the same offline dataset format,
- generate full trajectories directly in relative coordinates,
- use the true dynamics only for dataset collection, evaluation rollouts, and the oracle benchmark.

The first three variants use drifting for training; the last two use diffusion instead.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Choosing the variant

You can choose the method in **either** of two ways.

### Option 1: choose it in the YAML file

Set

```yaml
method:
  variant: cost_aware
```

or

```yaml
method:
  variant: cost_conditioned_prior
```

or

```yaml
method:
  variant: behavior_prior
```

or

```yaml
method:
  variant: diffusion_behavior_prior
```

or

```yaml
method:
  variant: guided_diffusion_behavior_prior
```

### Option 2: override it from the CLI

```bash
python3 scripts/train_model.py --config configs/msd_default.yaml --variant guided_diffusion_behavior_prior
```

The checkpoint stores the chosen variant, and evaluation automatically rebuilds the matching architecture from the checkpoint.

## Provided configs

Default configs:
- `configs/msd_default.yaml` → `cost_aware`
- `configs/msd_cost_conditioned_prior.yaml`
- `configs/msd_behavior_prior.yaml`
- `configs/msd_diffusion_behavior_prior.yaml`
- `configs/msd_guided_diffusion_behavior_prior.yaml`

Smoke configs:
- `configs/msd_smoke.yaml` → `cost_aware`
- `configs/msd_cost_conditioned_prior_smoke.yaml`
- `configs/msd_behavior_prior_smoke.yaml`
- `configs/msd_diffusion_behavior_prior_smoke.yaml`
- `configs/msd_guided_diffusion_behavior_prior_smoke.yaml`

## Exact smoke-run commands

### Cost-aware drifting

```bash
python3 scripts/collect_dataset.py --config configs/msd_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run
python3 scripts/train_model.py --config configs/msd_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --output-dir artifacts/runs/msd_cost_aware_smoke_run
python3 scripts/eval_model.py --config configs/msd_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --ckpt artifacts/runs/msd_cost_aware_smoke_run/best.pt --output-dir artifacts/eval/msd_cost_aware_smoke_run
```

### Cost-conditioned prior

```bash
python3 scripts/collect_dataset.py --config configs/msd_cost_conditioned_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run
python3 scripts/train_model.py --config configs/msd_cost_conditioned_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --output-dir artifacts/runs/msd_cc_prior_smoke_run
python3 scripts/eval_model.py --config configs/msd_cost_conditioned_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --ckpt artifacts/runs/msd_cc_prior_smoke_run/best.pt --output-dir artifacts/eval/msd_cc_prior_smoke_run
```

### Behavior prior without omega conditioning

```bash
python3 scripts/collect_dataset.py --config configs/msd_behavior_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run
python3 scripts/train_model.py --config configs/msd_behavior_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --output-dir artifacts/runs/msd_behavior_prior_smoke_run
python3 scripts/eval_model.py --config configs/msd_behavior_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --ckpt artifacts/runs/msd_behavior_prior_smoke_run/best.pt --output-dir artifacts/eval/msd_behavior_prior_smoke_run
```

### Diffusion behavior prior without omega conditioning

```bash
python3 scripts/collect_dataset.py --config configs/msd_diffusion_behavior_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run
python3 scripts/train_model.py --config configs/msd_diffusion_behavior_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --output-dir artifacts/runs/msd_diffusion_behavior_prior_smoke_run
python3 scripts/eval_model.py --config configs/msd_diffusion_behavior_prior_smoke.yaml --dataset-dir artifacts/datasets/msd_smoke_run --ckpt artifacts/runs/msd_diffusion_behavior_prior_smoke_run/best.pt --output-dir artifacts/eval/msd_diffusion_behavior_prior_smoke_run
```

## Backwards compatibility

The original script names still work:

```bash
python3 scripts/train_alt_a.py --config configs/msd_default.yaml
python3 scripts/eval_vs_oracle.py --config configs/msd_default.yaml --ckpt <path>
```

They now honor `method.variant` and dispatch automatically to either drifting or diffusion training/evaluation.

## What changes across variants

| Variant | Condition on omega during training? | Exponential cost tilt? | Training family | Use omega at planning time? |
|---|---:|---:|---|---:|
| `cost_aware` | yes | yes | drifting | yes |
| `cost_conditioned_prior` | yes | no | drifting | yes |
| `behavior_prior` | no | no | drifting | yes |
| `diffusion_behavior_prior` | no | no | diffusion | yes |
| `guided_diffusion_behavior_prior` | no | no | diffusion + reward guidance | yes |

The planner remains a best-of-`M` receding-horizon sampler for all variants. For drifting models, candidate trajectories come from one-shot generation. For the diffusion baseline, candidates come from DDIM-style sampling in normalized trajectory space.

## Important modules

- `drifting_mpc/methods.py`: variant definitions and config helpers
- `drifting_mpc/models/generator.py`: conditional residual MLP trajectory generator for drifting variants
- `drifting_mpc/models/diffusion.py`: trajectory denoiser used by the diffusion baseline
- `drifting_mpc/training/drifting.py`: generic drifting objective for drifting variants
- `drifting_mpc/training/diffusion.py`: diffusion objective and DDIM sampler
- `drifting_mpc/training/trainer.py`: generic trainer dispatch for drifting/diffusion families
- `drifting_mpc/training/diffusion_trainer.py`: trainer for both diffusion priors
- `drifting_mpc/training/diffusion.py`: diffusion objective, scheduler, and analytical reward-guided sampling
- `drifting_mpc/evaluation/policy.py`: planners that load the correct architecture from the checkpoint
- `drifting_mpc/evaluation/evaluate.py`: learned-vs-oracle evaluation
- `drifting_mpc/data/collection.py`: offline dataset generation
- `drifting_mpc/control/oracle.py`: finite-horizon LQR oracle

## Notes

- `cost_aware` is the paper's Alternative A.
- `cost_conditioned_prior` is the clean ablation where the model still sees `omega`, but no exponential reweighting is used.
- `behavior_prior` removes `omega` from the generator entirely while keeping drifting.
- `diffusion_behavior_prior` removes `omega` from training and replaces drifting with diffusion, yielding a planner more similar in spirit to diffusion-based trajectory baselines.
- `guided_diffusion_behavior_prior` keeps the same training objective as `diffusion_behavior_prior` but adds Diffuser-style guidance at sampling time using the analytical gradient of the known cumulative reward.
- The oracle is exact finite-horizon LQR because there are no hard action bounds in this benchmark.


## Comparing multiple checkpoints

Use `scripts/eval_compare_models.py` to evaluate multiple checkpoints on the same held-out episodes and save combined plots:

```bash
python scripts/eval_compare_models.py \
  --config configs/msd_default.yaml \
  --ckpts path/to/cost_aware.pt path/to/behavior_prior.pt path/to/diffusion_prior.pt \
  --labels cost_aware behavior_prior diffusion_prior \
  --output-dir artifacts/eval/compare_run
```

This writes a combined `metrics.json`, `metrics_summary.csv`, `episode_metrics.csv`, side-by-side cost histograms/scatter plots, and rollout plots containing all specified methods plus the oracle.
