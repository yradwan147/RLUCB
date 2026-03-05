"""Configuration dataclass for the UCB vs Random quizzing experiment."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ExperimentConfig:
    """
    Configuration for the UCB vs Random quizzing experiment.
    
    All parameters are configurable to allow exploration of different
    simulation scenarios.
    """
    
    # Population settings
    num_students_per_group: int = 50
    num_categories: int = 6
    
    # Category difficulty (affects initial knowledge distribution)
    # If None, will be randomly generated
    # Values represent how "hard" each category is (lower = harder)
    category_difficulties: Optional[List[float]] = None
    
    # Student heterogeneity - initial knowledge distribution
    initial_knowledge_mean: float = 0.4
    initial_knowledge_std: float = 0.1
    
    # Learning dynamics
    learning_rate: float = 0.12  # alpha: how much knowledge increases on correct answer
    incorrect_penalty: float = 0.02  # beta: how much knowledge decreases on incorrect answer
    
    # Forgetting dynamics (Ebbinghaus-inspired)
    base_knowledge: float = 0.10  # minimum retained knowledge after forgetting
    decay_rate: float = 0.01  # exponential decay rate per timestep
    
    # Experiment duration
    questions_per_session: int = 200  # total questions per student
    
    # UCB algorithm parameter
    exploration_param: float = 1.414  # typically sqrt(2)
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    
    # Logging frequency (log metrics every N timesteps)
    log_frequency: int = 1
    
    def __post_init__(self):
        """Validate and set default values after initialization."""
        # Generate category difficulties if not provided
        if self.category_difficulties is None:
            rng = np.random.default_rng(self.random_seed)
            self.category_difficulties = rng.uniform(0.3, 0.7, self.num_categories).tolist()
        
        # Validate category difficulties length
        if len(self.category_difficulties) != self.num_categories:
            raise ValueError(
                f"category_difficulties length ({len(self.category_difficulties)}) "
                f"must match num_categories ({self.num_categories})"
            )
        
        # Validate ranges
        if not 0 < self.initial_knowledge_mean < 1:
            raise ValueError("initial_knowledge_mean must be between 0 and 1")
        
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        
        if not 0 <= self.incorrect_penalty <= 1:
            raise ValueError("incorrect_penalty must be between 0 and 1")
        
        if not 0 <= self.base_knowledge < 1:
            raise ValueError("base_knowledge must be between 0 and 1")
        
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        
        if self.questions_per_session < 1:
            raise ValueError("questions_per_session must be at least 1")
        
        if self.exploration_param < 0:
            raise ValueError("exploration_param must be non-negative")
    
    def get_category_names(self) -> List[str]:
        """Generate category names for use with UCBSelector."""
        return [f"Category_{i}" for i in range(self.num_categories)]
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/saving."""
        return {
            "num_students_per_group": self.num_students_per_group,
            "num_categories": self.num_categories,
            "category_difficulties": self.category_difficulties,
            "initial_knowledge_mean": self.initial_knowledge_mean,
            "initial_knowledge_std": self.initial_knowledge_std,
            "learning_rate": self.learning_rate,
            "incorrect_penalty": self.incorrect_penalty,
            "base_knowledge": self.base_knowledge,
            "decay_rate": self.decay_rate,
            "questions_per_session": self.questions_per_session,
            "exploration_param": self.exploration_param,
            "random_seed": self.random_seed,
            "log_frequency": self.log_frequency,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**data)
