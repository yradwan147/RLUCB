"""Question selection strategies for the experiment."""

import sys
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

# Add UCBQuizMaster to path for importing
sys.path.insert(0, str(__file__).replace("/experiment/selectors.py", ""))

from UCBQuizMaster.ucb_algorithm import UCBSelector
from .config import ExperimentConfig


class BaseSelector(ABC):
    """Abstract base class for question selectors."""
    
    @abstractmethod
    def select_category(self) -> int:
        """Select the next category to quiz on."""
        pass
    
    @abstractmethod
    def update(self, category: int, correct: bool) -> None:
        """Update selector state after a question is answered."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the selector to initial state."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> dict:
        """Get selector statistics."""
        pass


class RandomSelector(BaseSelector):
    """
    Randomly selects categories with uniform probability.
    
    This serves as the control/baseline for comparison with UCB.
    """
    
    def __init__(
        self,
        num_categories: int,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize random selector.
        
        Args:
            num_categories: Number of categories to select from
            rng: Random number generator for reproducibility
        """
        self.num_categories = num_categories
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Track statistics for comparison with UCB
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
    
    def select_category(self) -> int:
        """Select a random category with uniform probability."""
        return int(self.rng.integers(0, self.num_categories))
    
    def update(self, category: int, correct: bool) -> None:
        """Update statistics after a question is answered."""
        self.attempts[category] += 1
        if correct:
            self.correct[category] += 1
        self.total_questions += 1
    
    def reset(self) -> None:
        """Reset selector to initial state."""
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
    
    def get_statistics(self) -> dict:
        """Get selector statistics."""
        stats = {}
        for i in range(self.num_categories):
            attempts = int(self.attempts[i])
            correct = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": attempts,
                "correct": correct,
                "incorrect": attempts - correct,
                "correctness_rate": correct / attempts if attempts > 0 else 0.0,
            }
        return stats
    
    def get_exposure_distribution(self) -> np.ndarray:
        """Get the distribution of exposures across categories."""
        return self.attempts.copy()


class UCBSelectorAdapter(BaseSelector):
    """
    Adapter for the existing UCBSelector to work with integer category indices.
    
    This wraps the UCBSelector from UCBQuizMaster to provide a consistent
    interface with RandomSelector.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        """
        Initialize UCB selector adapter.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.num_categories = config.num_categories
        
        # Create category names for UCBSelector
        self.category_names = config.get_category_names()
        
        # Create name-to-index mapping
        self.name_to_index = {name: i for i, name in enumerate(self.category_names)}
        self.index_to_name = {i: name for i, name in enumerate(self.category_names)}
        
        # Initialize the underlying UCBSelector
        self.ucb_selector = UCBSelector(
            subtopics=self.category_names,
            exploration_param=config.exploration_param,
        )
    
    def select_category(self) -> int:
        """Select the next category using UCB algorithm."""
        subtopic_name = self.ucb_selector.select_next_subtopic()
        return self.name_to_index[subtopic_name]
    
    def update(self, category: int, correct: bool) -> None:
        """Update UCB statistics after a question is answered."""
        subtopic_name = self.index_to_name[category]
        self.ucb_selector.update(subtopic_name, correct)
    
    def reset(self) -> None:
        """Reset the UCB selector to initial state."""
        self.ucb_selector = UCBSelector(
            subtopics=self.category_names,
            exploration_param=self.config.exploration_param,
        )
    
    def get_statistics(self) -> dict:
        """Get UCB selector statistics."""
        return self.ucb_selector.get_statistics()
    
    def get_ucb_scores(self) -> dict:
        """Get current UCB scores for visualization."""
        return self.ucb_selector.get_ucb_data_for_visualization()
    
    def get_weakest_category(self) -> int:
        """Get the index of the weakest category according to UCB."""
        weakest_name = self.ucb_selector.get_weakest_subtopic()
        return self.name_to_index[weakest_name]
    
    def get_exposure_distribution(self) -> np.ndarray:
        """Get the distribution of exposures across categories."""
        stats = self.ucb_selector.get_statistics()
        exposures = np.zeros(self.num_categories, dtype=np.int32)
        for name, data in stats.items():
            idx = self.name_to_index[name]
            exposures[idx] = data["attempts"]
        return exposures
