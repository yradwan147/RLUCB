"""Student class with knowledge dynamics for the learning simulation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from .config import ExperimentConfig


@dataclass
class StudentHistory:
    """Record of a single interaction in a student's history."""
    timestep: int
    category: int
    correct: bool
    knowledge_before: List[float]
    knowledge_after: List[float]


class Student:
    """
    Simulates a student with knowledge that evolves through learning and forgetting.
    
    Knowledge is represented as a vector of probabilities (one per category),
    where each probability represents the chance of answering correctly
    in that category.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        student_id: int = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize a student with knowledge based on config.
        
        Args:
            config: Experiment configuration
            student_id: Unique identifier for this student
            rng: Random number generator (for reproducibility)
        """
        self.config = config
        self.student_id = student_id
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Initialize knowledge vector
        self.knowledge = self._init_knowledge()
        
        # Track time since each category was last exposed
        self.time_since_exposure = np.zeros(config.num_categories, dtype=np.int32)
        
        # Track history of interactions
        self.history: List[StudentHistory] = []
        
        # Track exposure counts per category
        self.exposure_counts = np.zeros(config.num_categories, dtype=np.int32)
        
        # Track correct/incorrect per category
        self.correct_counts = np.zeros(config.num_categories, dtype=np.int32)
        self.incorrect_counts = np.zeros(config.num_categories, dtype=np.int32)
    
    def _init_knowledge(self) -> np.ndarray:
        """
        Initialize knowledge vector based on config.
        
        Knowledge is sampled from a normal distribution centered on the
        category difficulty, with student-specific variation.
        """
        knowledge = np.zeros(self.config.num_categories)
        
        for i in range(self.config.num_categories):
            # Base knowledge affected by category difficulty
            category_difficulty = self.config.category_difficulties[i]
            
            # Sample from normal distribution
            base = self.config.initial_knowledge_mean * category_difficulty
            k = self.rng.normal(base, self.config.initial_knowledge_std)
            
            # Clamp to valid range [0.05, 0.95]
            knowledge[i] = np.clip(k, 0.05, 0.95)
        
        return knowledge
    
    def answer(self, category: int) -> bool:
        """
        Simulate answering a question in the given category.
        
        The probability of answering correctly is based on the student's
        current knowledge in that category.
        
        Args:
            category: Index of the category being quizzed
            
        Returns:
            True if the answer is correct, False otherwise
        """
        if not 0 <= category < self.config.num_categories:
            raise ValueError(f"Invalid category index: {category}")
        
        # Probabilistic answer based on knowledge
        correct = self.rng.random() < self.knowledge[category]
        return correct
    
    def update_knowledge(self, category: int, correct: bool, timestep: int = 0) -> None:
        """
        Update knowledge after answering a question.
        
        If correct: knowledge increases asymptotically toward 1.0
        If incorrect: knowledge decreases slightly (penalty)
        
        Args:
            category: Index of the category that was quizzed
            correct: Whether the answer was correct
            timestep: Current timestep (for history tracking)
        """
        knowledge_before = self.knowledge.copy()
        
        if correct:
            # Asymptotic learning toward 1.0
            # k_new = k + alpha * (1 - k)
            self.knowledge[category] += self.config.learning_rate * (1 - self.knowledge[category])
            self.correct_counts[category] += 1
        else:
            # Small penalty for incorrect answer
            # k_new = k - beta * k
            self.knowledge[category] -= self.config.incorrect_penalty * self.knowledge[category]
            self.incorrect_counts[category] += 1
        
        # Clamp to valid range
        self.knowledge[category] = np.clip(self.knowledge[category], 0.01, 0.99)
        
        # Reset time since exposure for this category
        self.time_since_exposure[category] = 0
        
        # Update exposure count
        self.exposure_counts[category] += 1
        
        # Record history
        self.history.append(StudentHistory(
            timestep=timestep,
            category=category,
            correct=correct,
            knowledge_before=knowledge_before.tolist(),
            knowledge_after=self.knowledge.copy().tolist(),
        ))
    
    def apply_forgetting(self) -> None:
        """
        Apply forgetting dynamics to all categories.
        
        Uses Ebbinghaus-inspired exponential decay applied incrementally:
        k_new = base + (k - base) * exp(-decay_rate)
        
        This is applied once per timestep for categories not exposed this step.
        The decay compounds naturally over multiple timesteps.
        """
        # Decay factor applied once per timestep
        decay_factor = np.exp(-self.config.decay_rate)
        base = self.config.base_knowledge
        
        for i in range(self.config.num_categories):
            if self.time_since_exposure[i] > 0:
                # Incremental exponential decay toward base knowledge
                current = self.knowledge[i]
                self.knowledge[i] = base + (current - base) * decay_factor
                
                # Clamp to valid range
                self.knowledge[i] = np.clip(self.knowledge[i], 0.01, 0.99)
        
        # Increment time since exposure for all categories
        self.time_since_exposure += 1
    
    def get_knowledge_snapshot(self) -> np.ndarray:
        """Get a copy of the current knowledge vector."""
        return self.knowledge.copy()
    
    def get_average_knowledge(self) -> float:
        """Get the average knowledge across all categories."""
        return float(np.mean(self.knowledge))
    
    def get_weakest_category(self) -> int:
        """Get the index of the category with lowest knowledge."""
        return int(np.argmin(self.knowledge))
    
    def get_strongest_category(self) -> int:
        """Get the index of the category with highest knowledge."""
        return int(np.argmax(self.knowledge))
    
    def get_knowledge_variance(self) -> float:
        """Get the variance in knowledge across categories."""
        return float(np.var(self.knowledge))
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics about this student."""
        return {
            "student_id": self.student_id,
            "knowledge": self.knowledge.tolist(),
            "average_knowledge": self.get_average_knowledge(),
            "knowledge_variance": self.get_knowledge_variance(),
            "weakest_category": self.get_weakest_category(),
            "strongest_category": self.get_strongest_category(),
            "exposure_counts": self.exposure_counts.tolist(),
            "correct_counts": self.correct_counts.tolist(),
            "incorrect_counts": self.incorrect_counts.tolist(),
            "total_questions": int(np.sum(self.exposure_counts)),
            "overall_accuracy": (
                float(np.sum(self.correct_counts) / max(1, np.sum(self.exposure_counts)))
            ),
        }
    
    def reset(self) -> None:
        """Reset student to initial state."""
        self.knowledge = self._init_knowledge()
        self.time_since_exposure = np.zeros(self.config.num_categories, dtype=np.int32)
        self.history = []
        self.exposure_counts = np.zeros(self.config.num_categories, dtype=np.int32)
        self.correct_counts = np.zeros(self.config.num_categories, dtype=np.int32)
        self.incorrect_counts = np.zeros(self.config.num_categories, dtype=np.int32)
