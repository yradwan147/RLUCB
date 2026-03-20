#!/usr/bin/env python3
"""
Real Data Experiment Runner

Runs bandit algorithms on real educational datasets (Duolingo, ASSISTments).

Usage:
    python run_real_data.py --dataset duolingo --data-dir data/
    python run_real_data.py --dataset assistments --data-dir data/ --algorithms fucb bkt_bandit oracle
    python run_real_data.py --dataset duolingo --replay-only --max-students 500
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiment.real_data import run_real_data_experiment
from experiment.selectors import SELECTOR_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run bandit algorithms on real educational datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["duolingo", "assistments"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (contains duolingo/ and assistments/ subdirs)",
    )
    parser.add_argument(
        "--algorithms", "-a",
        type=str,
        nargs="+",
        default=None,  # None = use all from registry
        help="Algorithms to evaluate",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/real_data",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-students-fit",
        type=int,
        default=200,
        help="Max students for parameter fitting",
    )
    parser.add_argument(
        "--max-students-replay",
        type=int,
        default=200,
        help="Max students for replay evaluation",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=500,
        help="Max interactions per student in replay",
    )
    parser.add_argument(
        "--fitted-params",
        type=str,
        default=None,
        help="Path to pre-fitted params JSON (skip fitting if provided)",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Only run replay evaluation (skip fitted simulation)",
    )
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="Only run fitted-parameter simulation (skip replay)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Use all algorithms if none specified
    algorithms = args.algorithms if args.algorithms else list(SELECTOR_REGISTRY.keys())

    # Validate algorithms
    for algo in algorithms:
        if algo not in SELECTOR_REGISTRY:
            print(f"Error: Unknown algorithm '{algo}'. "
                  f"Available: {list(SELECTOR_REGISTRY.keys())}")
            return 1

    run_real_data_experiment(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        algorithms=algorithms,
        output_dir=args.output_dir,
        max_students_fit=args.max_students_fit,
        max_students_replay=args.max_students_replay,
        max_interactions=args.max_interactions,
        fitted_params_path=args.fitted_params,
        run_replay=not args.sim_only,
        run_fitted_sim=not args.replay_only,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
