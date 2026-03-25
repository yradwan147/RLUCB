#!/usr/bin/env python3
"""
UCB vs Random Quizzing Experiment Runner

This script runs the experiment comparing UCB-based adaptive quizzing
against random quizzing for simulated student learning.

Usage:
    python run_experiment.py                    # Run with defaults
    python run_experiment.py --students 100     # 100 students per group
    python run_experiment.py --questions 500    # 500 questions per session
    python run_experiment.py --dashboard        # Generate full dashboard
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiment.config import ExperimentConfig
from experiment.simulation import (
    Experiment, run_multiple_experiments,
    MultiAlgorithmExperiment, MultiAlgorithmResults,
)
from experiment.selectors import SELECTOR_REGISTRY
from experiment.visualization import (
    plot_knowledge_comparison,
    plot_category_heatmaps,
    plot_exposure_distribution,
    plot_weakest_category_improvement,
    plot_knowledge_variance,
    plot_final_knowledge_distribution,
    plot_accuracy_over_time,
    create_dashboard,
    print_summary_statistics,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run UCB vs Random Quizzing Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Population settings
    parser.add_argument(
        "--students", "-s",
        type=int,
        default=50,
        help="Number of students per group",
    )
    parser.add_argument(
        "--categories", "-c",
        type=int,
        default=6,
        help="Number of knowledge categories",
    )
    
    # Learning dynamics
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.12,
        help="Learning rate (alpha) for knowledge gain on correct answer",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.02,
        help="Penalty rate (beta) for knowledge loss on incorrect answer",
    )
    
    # Forgetting dynamics
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.01,
        help="Forgetting decay rate per timestep",
    )
    parser.add_argument(
        "--base-knowledge",
        type=float,
        default=0.10,
        help="Minimum knowledge after forgetting",
    )
    
    # Experiment settings
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=200,
        help="Number of questions per student session",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=1.414,
        help="UCB exploration parameter (c)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate comprehensive dashboard (all plots in one figure)",
    )
    parser.add_argument(
        "--individual-plots",
        action="store_true",
        help="Generate individual plot files",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (only save)",
    )
    parser.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress progress bars and summary output",
    )
    
    # Multiple runs
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of experiment runs (for statistical analysis)",
    )

    # Algorithm selection (multi-algorithm mode)
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        nargs="+",
        default=None,
        help="Algorithm(s) to run. Available: {}. "
             "If not specified, runs legacy UCB vs Random mode.".format(
                 ", ".join(SELECTOR_REGISTRY.keys())
             ),
    )
    parser.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Run all available algorithms",
    )
    parser.add_argument(
        "--num-students",
        type=int,
        default=None,
        help="Alias for --students (for CLI compatibility)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Alias for --output (for CLI compatibility)",
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=None,
        help="Log metrics every N timesteps (default: auto-set based on questions)",
    )
    parser.add_argument(
        "--algorithm-decay-rate",
        type=float,
        default=None,
        help="Decay rate for algorithms (mis-specification test). If set, algorithms use this instead of --decay-rate.",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle aliases
    num_students = args.num_students if args.num_students else args.students
    output_path = args.output_dir if args.output_dir else args.output

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-set log frequency to keep output manageable (~1000 data points max)
    if args.log_frequency is not None:
        log_freq = args.log_frequency
    else:
        log_freq = max(1, args.questions // 1000)

    # Create configuration
    config = ExperimentConfig(
        num_students_per_group=num_students,
        num_categories=args.categories,
        initial_knowledge_mean=0.3,
        initial_knowledge_std=0.1,
        learning_rate=args.learning_rate,
        incorrect_penalty=args.penalty,
        base_knowledge=args.base_knowledge,
        decay_rate=args.decay_rate,
        algorithm_decay_rate=args.algorithm_decay_rate,
        questions_per_session=args.questions,
        exploration_param=args.exploration,
        random_seed=args.seed,
        log_frequency=log_freq,
    )

    # Determine algorithms to run
    algorithms = None
    if args.all_algorithms:
        algorithms = list(SELECTOR_REGISTRY.keys())
    elif args.algorithm:
        algorithms = args.algorithm

    if not args.quiet:
        print("=" * 60)
        if algorithms:
            print("Multi-Algorithm Adaptive Quizzing Experiment")
        else:
            print("UCB vs Random Quizzing Experiment")
        print("=" * 60)
        print()
        print("Configuration:")
        print(f"  Students per group: {config.num_students_per_group}")
        print(f"  Categories: {config.num_categories}")
        print(f"  Questions per session: {config.questions_per_session}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Decay rate: {config.decay_rate}")
        print(f"  UCB exploration: {config.exploration_param}")
        print(f"  Random seed: {config.random_seed}")
        if algorithms:
            print(f"  Algorithms: {algorithms}")
        print()

    # Multi-algorithm mode
    if algorithms:
        exp = MultiAlgorithmExperiment(config, algorithms)
        results = exp.run(show_progress=not args.quiet)

        if not args.quiet:
            print("\nResults:")
            print("{:20s} | {:>14s} | {:>12s} | {:>10s}".format(
                "Algorithm", "Avg Knowledge", "Weakest Cat", "Accuracy"))
            print("-" * 65)
            for algo in algorithms:
                final = results.metrics[algo][-1]
                total = final.cumulative_correct + final.cumulative_incorrect
                acc = final.cumulative_correct / total if total > 0 else 0
                print("{:20s} | {:14.4f} | {:12.4f} | {:10.4f}".format(
                    algo, final.average_knowledge, final.weakest_category_avg, acc))

        # Export CSV
        csv_path = output_dir / "experiment_results.csv"
        results.save_to_csv(str(csv_path))
        if not args.quiet:
            print(f"\nResults saved to: {csv_path}")

        return 0

    # Legacy UCB vs Random mode
    if args.runs > 1:
        if not args.quiet:
            print(f"Running {args.runs} experiments...")
        results_list = run_multiple_experiments(
            config,
            num_runs=args.runs,
            show_progress=not args.quiet,
        )
        results = results_list[0]
        if not args.quiet:
            print(f"\nCompleted {args.runs} experiment runs.")
    else:
        experiment = Experiment(config)
        results = experiment.run(show_progress=not args.quiet)

    # Print summary
    if not args.quiet:
        print()
        print(print_summary_statistics(results))

    # Export CSV if requested
    if args.csv:
        csv_path = output_dir / "experiment_results.csv"
        results.save_to_csv(str(csv_path))
        if not args.quiet:
            print(f"\nResults saved to: {csv_path}")

    # Generate visualizations
    import matplotlib.pyplot as plt

    if args.dashboard:
        if not args.quiet:
            print("\nGenerating dashboard...")
        dashboard_path = output_dir / "dashboard.png"
        create_dashboard(results, save_path=str(dashboard_path))
        if not args.quiet:
            print(f"Dashboard saved to: {dashboard_path}")

    if args.individual_plots:
        if not args.quiet:
            print("\nGenerating individual plots...")

        plot_knowledge_comparison(results, save_path=str(output_dir / "knowledge_comparison.png"))
        plot_category_heatmaps(results, save_path=str(output_dir / "category_heatmaps.png"))
        plot_exposure_distribution(results, save_path=str(output_dir / "exposure_distribution.png"))
        plot_weakest_category_improvement(results, save_path=str(output_dir / "weakest_category.png"))
        plot_knowledge_variance(results, save_path=str(output_dir / "knowledge_variance.png"))
        plot_final_knowledge_distribution(results, save_path=str(output_dir / "final_distribution.png"))
        plot_accuracy_over_time(results, save_path=str(output_dir / "accuracy.png"))

        if not args.quiet:
            print(f"Plots saved to: {output_dir}/")

    if not args.no_show and (args.dashboard or args.individual_plots):
        plt.show()

    if not args.dashboard and not args.individual_plots and not args.csv:
        if not args.quiet:
            print("\nGenerating dashboard (use --dashboard to save)...")
        create_dashboard(results)
        if not args.no_show:
            plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
