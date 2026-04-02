from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting_mpc.config import clone_config, load_config
from drifting_mpc.methods import set_method_variant
from drifting_mpc.evaluation.evaluate import evaluate_vs_oracle
from drifting_mpc.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a drifting trajectory model against the oracle controller.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint saved by train_alt_a.py.")
    parser.add_argument("--variant", type=str, default=None, choices=["cost_aware", "cost_conditioned_prior", "behavior_prior", "diffusion_behavior_prior", "guided_diffusion_behavior_prior"], help="Optional override for output config bookkeeping. The checkpoint architecture is loaded from the checkpoint itself.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional override for dataset.output_dir.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for evaluation.output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = clone_config(load_config(args.config))
    if args.variant is not None:
        set_method_variant(config, args.variant)
    set_global_seed(int(config["experiment"]["seed"]))
    if args.dataset_dir is not None:
        config["dataset"]["output_dir"] = args.dataset_dir
    if args.output_dir is not None:
        config["evaluation"]["output_dir"] = args.output_dir
    outputs = evaluate_vs_oracle(
        ckpt_path=args.ckpt,
        config=config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    print(f"eval_dir={outputs['eval_dir']}")
    print(f"metrics={outputs['metrics']}")
    print(f"histogram={outputs['histogram']}")
    print(f"scatter={outputs['scatter']}")


if __name__ == "__main__":
    main()
