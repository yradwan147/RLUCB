"""Visualization functions for experiment results."""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .simulation import ExperimentResults, GroupMetrics


# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", palette="muted")


def plot_knowledge_comparison(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot average knowledge over time for UCB vs Control groups.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    ucb_knowledge = [m.average_knowledge for m in results.ucb_metrics]
    ucb_std = [m.knowledge_std for m in results.ucb_metrics]
    
    control_timesteps = [m.timestep for m in results.control_metrics]
    control_knowledge = [m.average_knowledge for m in results.control_metrics]
    control_std = [m.knowledge_std for m in results.control_metrics]
    
    # Plot lines
    ax.plot(ucb_timesteps, ucb_knowledge, label="UCB (Adaptive)", 
            color="#2ecc71", linewidth=2)
    ax.plot(control_timesteps, control_knowledge, label="Random (Control)", 
            color="#e74c3c", linewidth=2)
    
    # Add confidence bands
    ucb_knowledge = np.array(ucb_knowledge)
    ucb_std = np.array(ucb_std)
    control_knowledge = np.array(control_knowledge)
    control_std = np.array(control_std)
    
    ax.fill_between(ucb_timesteps, 
                    ucb_knowledge - ucb_std, 
                    ucb_knowledge + ucb_std,
                    alpha=0.2, color="#2ecc71")
    ax.fill_between(control_timesteps,
                    control_knowledge - control_std,
                    control_knowledge + control_std,
                    alpha=0.2, color="#e74c3c")
    
    ax.set_xlabel("Questions Answered", fontsize=12)
    ax.set_ylabel("Average Knowledge", fontsize=12)
    ax.set_title("Knowledge Acquisition: UCB vs Random Quizzing", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_category_heatmaps(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-category knowledge evolution as heatmaps.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    num_categories = results.config.num_categories
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Build data matrices
    ucb_matrix = np.array([m.per_category_knowledge for m in results.ucb_metrics]).T
    control_matrix = np.array([m.per_category_knowledge for m in results.control_metrics]).T
    
    timesteps = [m.timestep for m in results.ucb_metrics]
    
    # Determine common color scale
    vmin = min(ucb_matrix.min(), control_matrix.min())
    vmax = max(ucb_matrix.max(), control_matrix.max())
    
    # UCB heatmap
    sns.heatmap(ucb_matrix, ax=axes[0], cmap="YlGn", vmin=vmin, vmax=vmax,
                xticklabels=20, yticklabels=[f"Cat {i}" for i in range(num_categories)],
                cbar_kws={"label": "Knowledge"})
    axes[0].set_title("UCB Group - Per-Category Knowledge", fontsize=12)
    axes[0].set_xlabel("Timestep", fontsize=10)
    axes[0].set_ylabel("Category", fontsize=10)
    
    # Control heatmap
    sns.heatmap(control_matrix, ax=axes[1], cmap="YlOrRd", vmin=vmin, vmax=vmax,
                xticklabels=20, yticklabels=[f"Cat {i}" for i in range(num_categories)],
                cbar_kws={"label": "Knowledge"})
    axes[1].set_title("Control Group - Per-Category Knowledge", fontsize=12)
    axes[1].set_xlabel("Timestep", fontsize=10)
    axes[1].set_ylabel("Category", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_exposure_distribution(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how often each category was quizzed (exposure distribution).
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_categories = results.config.num_categories
    categories = [f"Category {i}" for i in range(num_categories)]
    
    # Get final exposure counts
    ucb_exposure = results.ucb_metrics[-1].exposure_distribution
    control_exposure = results.control_metrics[-1].exposure_distribution
    
    x = np.arange(num_categories)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ucb_exposure, width, label="UCB (Adaptive)",
                   color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x + width/2, control_exposure, width, label="Random (Control)",
                   color="#e74c3c", alpha=0.8)
    
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Number of Exposures", fontsize=12)
    ax.set_title("Category Exposure Distribution", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_weakest_category_improvement(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how the weakest category's knowledge improves over time.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    ucb_weakest = [m.weakest_category_avg for m in results.ucb_metrics]
    control_weakest = [m.weakest_category_avg for m in results.control_metrics]
    
    ax.plot(ucb_timesteps, ucb_weakest, label="UCB - Weakest Category",
            color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_weakest, label="Random - Weakest Category",
            color="#e74c3c", linewidth=2)
    
    ax.set_xlabel("Questions Answered", fontsize=12)
    ax.set_ylabel("Weakest Category Knowledge", fontsize=12)
    ax.set_title("Improvement in Weakest Category", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_knowledge_variance(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot knowledge variance over time (measure of uniformity).
    
    Lower variance means more uniform knowledge across categories.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    ucb_variance = [m.knowledge_variance_avg for m in results.ucb_metrics]
    control_variance = [m.knowledge_variance_avg for m in results.control_metrics]
    
    ax.plot(ucb_timesteps, ucb_variance, label="UCB (Adaptive)",
            color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_variance, label="Random (Control)",
            color="#e74c3c", linewidth=2)
    
    ax.set_xlabel("Questions Answered", fontsize=12)
    ax.set_ylabel("Knowledge Variance (lower = more uniform)", fontsize=12)
    ax.set_title("Knowledge Uniformity Across Categories", fontsize=14)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_final_knowledge_distribution(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of final knowledge using violin plots.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect final knowledge per student
    ucb_knowledge = [s["average_knowledge"] for s in results.ucb_final_students]
    control_knowledge = [s["average_knowledge"] for s in results.control_final_students]
    
    # Create DataFrame for seaborn
    data = pd.DataFrame({
        "Knowledge": ucb_knowledge + control_knowledge,
        "Group": ["UCB"] * len(ucb_knowledge) + ["Random"] * len(control_knowledge)
    })
    
    # Create violin plot
    sns.violinplot(data=data, x="Group", y="Knowledge", hue="Group", ax=ax,
                   palette={"UCB": "#2ecc71", "Random": "#e74c3c"}, legend=False)
    
    ax.set_ylabel("Final Average Knowledge", fontsize=12)
    ax.set_title("Distribution of Final Knowledge Across Students", fontsize=14)
    ax.set_ylim(0, 1)
    
    # Add mean markers
    means = data.groupby("Group")["Knowledge"].mean()
    for i, group in enumerate(["UCB", "Random"]):
        ax.scatter([i], [means[group]], color="white", s=100, zorder=10,
                   edgecolors="black", linewidths=2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_accuracy_over_time(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative accuracy over time.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    
    ucb_accuracy = []
    control_accuracy = []
    
    for m in results.ucb_metrics:
        total = m.cumulative_correct + m.cumulative_incorrect
        ucb_accuracy.append(m.cumulative_correct / total if total > 0 else 0)
    
    for m in results.control_metrics:
        total = m.cumulative_correct + m.cumulative_incorrect
        control_accuracy.append(m.cumulative_correct / total if total > 0 else 0)
    
    ax.plot(ucb_timesteps, ucb_accuracy, label="UCB (Adaptive)",
            color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_accuracy, label="Random (Control)",
            color="#e74c3c", linewidth=2)
    
    ax.set_xlabel("Questions Answered", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy", fontsize=12)
    ax.set_title("Answer Accuracy Over Time", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def create_dashboard(
    results: ExperimentResults,
    figsize: Tuple[int, int] = (16, 20),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive dashboard with all key visualizations.
    
    Args:
        results: Experiment results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)
    
    # 1. Knowledge comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_knowledge_on_ax(results, ax1)
    
    # 2. Exposure distribution (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_exposure_on_ax(results, ax2)
    
    # 3. Category heatmaps (second row, full width)
    ax3a = fig.add_subplot(gs[1, 0])
    ax3b = fig.add_subplot(gs[1, 1])
    _plot_heatmaps_on_ax(results, ax3a, ax3b)
    
    # 4. Weakest category (third row left)
    ax4 = fig.add_subplot(gs[2, 0])
    _plot_weakest_on_ax(results, ax4)
    
    # 5. Knowledge variance (third row right)
    ax5 = fig.add_subplot(gs[2, 1])
    _plot_variance_on_ax(results, ax5)
    
    # 6. Final distribution (bottom left)
    ax6 = fig.add_subplot(gs[3, 0])
    _plot_final_dist_on_ax(results, ax6)
    
    # 7. Accuracy (bottom right)
    ax7 = fig.add_subplot(gs[3, 1])
    _plot_accuracy_on_ax(results, ax7)
    
    # Add title
    fig.suptitle("UCB vs Random Quizzing Experiment Results", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


# Helper functions for dashboard
def _plot_knowledge_on_ax(results: ExperimentResults, ax: plt.Axes) -> None:
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    ucb_knowledge = [m.average_knowledge for m in results.ucb_metrics]
    control_knowledge = [m.average_knowledge for m in results.control_metrics]
    
    ax.plot(ucb_timesteps, ucb_knowledge, label="UCB", color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_knowledge, label="Random", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Questions")
    ax.set_ylabel("Avg Knowledge")
    ax.set_title("Knowledge Over Time")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1)


def _plot_exposure_on_ax(results: ExperimentResults, ax: plt.Axes) -> None:
    num_categories = results.config.num_categories
    ucb_exposure = results.ucb_metrics[-1].exposure_distribution
    control_exposure = results.control_metrics[-1].exposure_distribution
    
    x = np.arange(num_categories)
    width = 0.35
    
    ax.bar(x - width/2, ucb_exposure, width, label="UCB", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, control_exposure, width, label="Random", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Category")
    ax.set_ylabel("Exposures")
    ax.set_title("Exposure Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_categories)])
    ax.legend(fontsize=9)


def _plot_heatmaps_on_ax(results: ExperimentResults, ax_ucb: plt.Axes, ax_ctrl: plt.Axes) -> None:
    num_categories = results.config.num_categories
    
    ucb_matrix = np.array([m.per_category_knowledge for m in results.ucb_metrics]).T
    control_matrix = np.array([m.per_category_knowledge for m in results.control_metrics]).T
    
    vmin = min(ucb_matrix.min(), control_matrix.min())
    vmax = max(ucb_matrix.max(), control_matrix.max())
    
    sns.heatmap(ucb_matrix, ax=ax_ucb, cmap="YlGn", vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=[f"C{i}" for i in range(num_categories)],
                cbar_kws={"label": "K"})
    ax_ucb.set_title("UCB - Category Knowledge")
    ax_ucb.set_xlabel("Time")
    
    sns.heatmap(control_matrix, ax=ax_ctrl, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=[f"C{i}" for i in range(num_categories)],
                cbar_kws={"label": "K"})
    ax_ctrl.set_title("Random - Category Knowledge")
    ax_ctrl.set_xlabel("Time")


def _plot_weakest_on_ax(results: ExperimentResults, ax: plt.Axes) -> None:
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    ucb_weakest = [m.weakest_category_avg for m in results.ucb_metrics]
    control_weakest = [m.weakest_category_avg for m in results.control_metrics]
    
    ax.plot(ucb_timesteps, ucb_weakest, label="UCB", color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_weakest, label="Random", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Questions")
    ax.set_ylabel("Weakest Cat. Knowledge")
    ax.set_title("Weakest Category Improvement")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1)


def _plot_variance_on_ax(results: ExperimentResults, ax: plt.Axes) -> None:
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    ucb_variance = [m.knowledge_variance_avg for m in results.ucb_metrics]
    control_variance = [m.knowledge_variance_avg for m in results.control_metrics]
    
    ax.plot(ucb_timesteps, ucb_variance, label="UCB", color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_variance, label="Random", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Questions")
    ax.set_ylabel("Variance (lower=uniform)")
    ax.set_title("Knowledge Uniformity")
    ax.legend(loc="upper right", fontsize=9)


def _plot_final_dist_on_ax(results: ExperimentResults, ax: plt.Axes) -> None:
    ucb_knowledge = [s["average_knowledge"] for s in results.ucb_final_students]
    control_knowledge = [s["average_knowledge"] for s in results.control_final_students]
    
    data = pd.DataFrame({
        "Knowledge": ucb_knowledge + control_knowledge,
        "Group": ["UCB"] * len(ucb_knowledge) + ["Random"] * len(control_knowledge)
    })
    
    sns.violinplot(data=data, x="Group", y="Knowledge", hue="Group", ax=ax,
                   palette={"UCB": "#2ecc71", "Random": "#e74c3c"}, legend=False)
    ax.set_ylabel("Final Avg Knowledge")
    ax.set_title("Final Knowledge Distribution")
    ax.set_ylim(0, 1)


def _plot_accuracy_on_ax(results: ExperimentResults, ax: plt.Axes) -> None:
    ucb_timesteps = [m.timestep for m in results.ucb_metrics]
    
    ucb_accuracy = []
    control_accuracy = []
    
    for m in results.ucb_metrics:
        total = m.cumulative_correct + m.cumulative_incorrect
        ucb_accuracy.append(m.cumulative_correct / total if total > 0 else 0)
    
    for m in results.control_metrics:
        total = m.cumulative_correct + m.cumulative_incorrect
        control_accuracy.append(m.cumulative_correct / total if total > 0 else 0)
    
    ax.plot(ucb_timesteps, ucb_accuracy, label="UCB", color="#2ecc71", linewidth=2)
    ax.plot(ucb_timesteps, control_accuracy, label="Random", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Questions")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Answer Accuracy")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1)


def print_summary_statistics(results: ExperimentResults) -> str:
    """
    Generate a text summary of experiment results.
    
    Args:
        results: Experiment results
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "EXPERIMENT SUMMARY",
        "=" * 60,
        "",
        "Configuration:",
        f"  Students per group: {results.config.num_students_per_group}",
        f"  Categories: {results.config.num_categories}",
        f"  Questions per session: {results.config.questions_per_session}",
        f"  Learning rate: {results.config.learning_rate}",
        f"  Decay rate: {results.config.decay_rate}",
        "",
        "-" * 60,
        "Final Results:",
        "-" * 60,
        "",
    ]
    
    # UCB group stats
    ucb_final_knowledge = [s["average_knowledge"] for s in results.ucb_final_students]
    ucb_final_accuracy = [s["overall_accuracy"] for s in results.ucb_final_students]
    
    lines.extend([
        "UCB Group:",
        f"  Final Avg Knowledge: {np.mean(ucb_final_knowledge):.4f} (+/- {np.std(ucb_final_knowledge):.4f})",
        f"  Final Accuracy: {np.mean(ucb_final_accuracy):.4f}",
        f"  Weakest Category (final): {results.ucb_metrics[-1].weakest_category_avg:.4f}",
        "",
    ])
    
    # Control group stats
    ctrl_final_knowledge = [s["average_knowledge"] for s in results.control_final_students]
    ctrl_final_accuracy = [s["overall_accuracy"] for s in results.control_final_students]
    
    lines.extend([
        "Control Group (Random):",
        f"  Final Avg Knowledge: {np.mean(ctrl_final_knowledge):.4f} (+/- {np.std(ctrl_final_knowledge):.4f})",
        f"  Final Accuracy: {np.mean(ctrl_final_accuracy):.4f}",
        f"  Weakest Category (final): {results.control_metrics[-1].weakest_category_avg:.4f}",
        "",
    ])
    
    # Comparison
    knowledge_diff = np.mean(ucb_final_knowledge) - np.mean(ctrl_final_knowledge)
    knowledge_pct = (knowledge_diff / np.mean(ctrl_final_knowledge)) * 100
    
    lines.extend([
        "-" * 60,
        "Comparison:",
        "-" * 60,
        f"  UCB Knowledge Advantage: {knowledge_diff:+.4f} ({knowledge_pct:+.1f}%)",
        f"  UCB Weakest Cat Advantage: {results.ucb_metrics[-1].weakest_category_avg - results.control_metrics[-1].weakest_category_avg:+.4f}",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)
