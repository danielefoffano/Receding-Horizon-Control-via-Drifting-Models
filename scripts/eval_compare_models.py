from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting_mpc.config import clone_config, load_config
from drifting_mpc.evaluation.evaluate import evaluate_multiple_vs_oracle
from drifting_mpc.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple learned planners against the oracle on the same evaluation episodes.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config controlling evaluation episodes/output.")
    parser.add_argument("--ckpts", nargs="+", required=True, help="One or more checkpoint paths to compare.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching the checkpoint list. Defaults to checkpoint method variants.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional override for dataset.output_dir.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for evaluation.output_dir.")
    parser.add_argument("--m-plan", type=int, default=None, help="Optional shared candidate count used for all loaded planners.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = clone_config(load_config(args.config))
    set_global_seed(int(config["experiment"]["seed"]))
    if args.dataset_dir is not None:
        config["dataset"]["output_dir"] = args.dataset_dir
    if args.output_dir is not None:
        config["evaluation"]["output_dir"] = args.output_dir
    outputs = evaluate_multiple_vs_oracle(
        ckpt_paths=args.ckpts,
        labels=args.labels,
        config=config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        m_plan=args.m_plan,
    )
    print(f"eval_dir={outputs['eval_dir']}")
    print(f"metrics={outputs['metrics']}")
    print(f"summary={outputs['summary']}")
    print(f"histograms={outputs['histograms']}")
    print(f"scatter={outputs['scatter']}")


if __name__ == "__main__":
    main()
