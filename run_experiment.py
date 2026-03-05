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
from experiment.simulation import Experiment, run_multiple_experiments
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = ExperimentConfig(
        num_students_per_group=args.students,
        num_categories=args.categories,
        initial_knowledge_mean=0.3,
        initial_knowledge_std=0.1,
        learning_rate=args.learning_rate,
        incorrect_penalty=args.penalty,
        base_knowledge=args.base_knowledge,
        decay_rate=args.decay_rate,
        questions_per_session=args.questions,
        exploration_param=args.exploration,
        random_seed=args.seed,
    )
    
    if not args.quiet:
        print("=" * 60)
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
        print()
    
    # Run experiment(s)
    if args.runs > 1:
        if not args.quiet:
            print(f"Running {args.runs} experiments...")
        results_list = run_multiple_experiments(
            config, 
            num_runs=args.runs,
            show_progress=not args.quiet,
        )
        # Use first result for visualization
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
        
        # Knowledge comparison
        plot_knowledge_comparison(
            results, 
            save_path=str(output_dir / "knowledge_comparison.png")
        )
        
        # Category heatmaps
        plot_category_heatmaps(
            results,
            save_path=str(output_dir / "category_heatmaps.png")
        )
        
        # Exposure distribution
        plot_exposure_distribution(
            results,
            save_path=str(output_dir / "exposure_distribution.png")
        )
        
        # Weakest category
        plot_weakest_category_improvement(
            results,
            save_path=str(output_dir / "weakest_category.png")
        )
        
        # Knowledge variance
        plot_knowledge_variance(
            results,
            save_path=str(output_dir / "knowledge_variance.png")
        )
        
        # Final distribution
        plot_final_knowledge_distribution(
            results,
            save_path=str(output_dir / "final_distribution.png")
        )
        
        # Accuracy
        plot_accuracy_over_time(
            results,
            save_path=str(output_dir / "accuracy.png")
        )
        
        if not args.quiet:
            print(f"Plots saved to: {output_dir}/")
    
    # Show plots if not suppressed
    if not args.no_show and (args.dashboard or args.individual_plots):
        plt.show()
    
    # Default behavior: show dashboard if no output options specified
    if not args.dashboard and not args.individual_plots and not args.csv:
        if not args.quiet:
            print("\nGenerating dashboard (use --dashboard to save)...")
        create_dashboard(results)
        if not args.no_show:
            plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
