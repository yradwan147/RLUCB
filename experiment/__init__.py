"""UCB vs Random Quizzing RL Experiment Package."""

from .config import ExperimentConfig
from .student import Student
from .selectors import RandomSelector, UCBSelectorAdapter
from .environment import QuizEnvironment
from .simulation import Experiment
from .visualization import (
    plot_knowledge_comparison,
    plot_category_heatmaps,
    plot_exposure_distribution,
    create_dashboard,
)

__all__ = [
    "ExperimentConfig",
    "Student",
    "RandomSelector",
    "UCBSelectorAdapter",
    "QuizEnvironment",
    "Experiment",
    "plot_knowledge_comparison",
    "plot_category_heatmaps",
    "plot_exposure_distribution",
    "create_dashboard",
]
