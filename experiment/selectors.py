"""Question selection strategies for the experiment."""

import math
import sys
from abc import ABC, abstractmethod
from collections import deque
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


class FUCBSelector(BaseSelector):
    """
    Forgetting-aware UCB (F-UCB) selector.

    Extends UCB1 by incorporating time-since-exposure into the score:
      score(i) = (1 - r_i) * exp(-λ * t_i) + γ * (1 - exp(-λ * t_i)) + c * sqrt(ln(N) / n_i)

    The first term decays the observed weakness as it becomes stale.
    The second term (forgetting urgency) increases priority for categories
    that haven't been visited recently, reflecting expected knowledge decay.
    """

    def __init__(
        self,
        num_categories: int,
        exploration_param: float = 1.414,
        decay_rate: float = 0.01,
        urgency_weight: float = 0.5,
    ):
        self.num_categories = num_categories
        self.exploration_param = exploration_param
        self.decay_rate = decay_rate
        self.urgency_weight = urgency_weight

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)

    def select_category(self) -> int:
        # Prioritize unvisited categories
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            r_i = self.correct[i] / self.attempts[i]
            t_i = self.time_since_exposure[i]
            decay = math.exp(-self.decay_rate * t_i)

            # Time-decayed weakness + forgetting urgency
            weakness = (1 - r_i) * decay
            urgency = self.urgency_weight * (1 - decay)

            # Exploration bonus
            exploration = self.exploration_param * math.sqrt(
                math.log(self.total_questions + 1) / self.attempts[i]
            )

            score = weakness + urgency + exploration
            if score > best_score:
                best_score = score
                best_cat = i

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        self.attempts[category] += 1
        if correct:
            self.correct[category] += 1
        self.total_questions += 1
        # Increment time for all, then reset the quizzed category
        self.time_since_exposure += 1
        self.time_since_exposure[category] = 0

    def reset(self) -> None:
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(self.num_categories, dtype=np.int32)

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class BKTBanditSelector(BaseSelector):
    """
    Bayesian Knowledge-Tracing Bandit (BKT-Bandit) selector.

    Maintains a Beta posterior per category for latent knowledge probability.
    Selection uses an optimistic estimate based on posterior uncertainty:
      score(i) = (1 - mean(Beta_i)) + c * std(Beta_i)

    Between observations, forgetting flattens the posterior (increases variance),
    creating a principled mechanism for revisiting neglected categories.
    """

    def __init__(
        self,
        num_categories: int,
        exploration_param: float = 1.414,
        decay_rate: float = 0.01,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.num_categories = num_categories
        self.exploration_param = exploration_param
        self.decay_rate = decay_rate
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Beta posterior params per category
        self.alpha = np.full(num_categories, prior_alpha)
        self.beta_ = np.full(num_categories, prior_beta)

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)

    def _apply_forgetting(self) -> None:
        """Flatten posteriors for categories not recently visited."""
        for i in range(self.num_categories):
            if self.time_since_exposure[i] > 0:
                t = self.time_since_exposure[i]
                # Shrink posterior toward prior proportionally to time elapsed
                decay = math.exp(-self.decay_rate * t)
                self.alpha[i] = self.prior_alpha + (self.alpha[i] - self.prior_alpha) * decay
                self.beta_[i] = self.prior_beta + (self.beta_[i] - self.prior_beta) * decay

    def select_category(self) -> int:
        self._apply_forgetting()

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            a, b = self.alpha[i], self.beta_[i]
            mean = a / (a + b)
            std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

            # Target weak areas with uncertainty bonus
            score = (1 - mean) + self.exploration_param * std
            if score > best_score:
                best_score = score
                best_cat = i

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        if correct:
            self.alpha[category] += 1
            self.correct[category] += 1
        else:
            self.beta_[category] += 1
        self.attempts[category] += 1
        self.total_questions += 1
        self.time_since_exposure += 1
        self.time_since_exposure[category] = 0

    def reset(self) -> None:
        self.alpha = np.full(self.num_categories, self.prior_alpha)
        self.beta_ = np.full(self.num_categories, self.prior_beta)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(self.num_categories, dtype=np.int32)

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class BKTThompsonSelector(BaseSelector):
    """
    BKT-Thompson Sampling selector.

    Like BKT-Bandit but uses Thompson Sampling: samples knowledge from each
    category's Beta posterior, then selects the category with lowest sampled
    knowledge (i.e., highest sampled weakness).

    Forgetting flattens posteriors the same way as BKT-Bandit.
    """

    def __init__(
        self,
        num_categories: int,
        decay_rate: float = 0.01,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_categories = num_categories
        self.decay_rate = decay_rate
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = rng if rng is not None else np.random.default_rng()

        self.alpha = np.full(num_categories, prior_alpha)
        self.beta_ = np.full(num_categories, prior_beta)

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)

    def _apply_forgetting(self) -> None:
        for i in range(self.num_categories):
            if self.time_since_exposure[i] > 0:
                t = self.time_since_exposure[i]
                decay = math.exp(-self.decay_rate * t)
                self.alpha[i] = self.prior_alpha + (self.alpha[i] - self.prior_alpha) * decay
                self.beta_[i] = self.prior_beta + (self.beta_[i] - self.prior_beta) * decay

    def select_category(self) -> int:
        self._apply_forgetting()

        # Sample knowledge from each category's posterior
        samples = np.array([
            self.rng.beta(self.alpha[i], self.beta_[i])
            for i in range(self.num_categories)
        ])
        # Select the weakest sampled knowledge
        return int(np.argmin(samples))

    def update(self, category: int, correct: bool) -> None:
        if correct:
            self.alpha[category] += 1
            self.correct[category] += 1
        else:
            self.beta_[category] += 1
        self.attempts[category] += 1
        self.total_questions += 1
        self.time_since_exposure += 1
        self.time_since_exposure[category] = 0

    def reset(self) -> None:
        self.alpha = np.full(self.num_categories, self.prior_alpha)
        self.beta_ = np.full(self.num_categories, self.prior_beta)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(self.num_categories, dtype=np.int32)

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class ThompsonSelector(BaseSelector):
    """
    Standard Thompson Sampling selector (Beta-Bernoulli, no forgetting).

    Maintains Beta posterior per category based on correct/incorrect counts.
    Samples from each posterior and selects the category with lowest sampled
    knowledge (highest weakness).
    """

    def __init__(
        self,
        num_categories: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_categories = num_categories
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = rng if rng is not None else np.random.default_rng()

        self.alpha = np.full(num_categories, prior_alpha)
        self.beta_ = np.full(num_categories, prior_beta)

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def select_category(self) -> int:
        samples = np.array([
            self.rng.beta(self.alpha[i], self.beta_[i])
            for i in range(self.num_categories)
        ])
        return int(np.argmin(samples))

    def update(self, category: int, correct: bool) -> None:
        if correct:
            self.alpha[category] += 1
            self.correct[category] += 1
        else:
            self.beta_[category] += 1
        self.attempts[category] += 1
        self.total_questions += 1

    def reset(self) -> None:
        self.alpha = np.full(self.num_categories, self.prior_alpha)
        self.beta_ = np.full(self.num_categories, self.prior_beta)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class EpsilonGreedySelector(BaseSelector):
    """
    Epsilon-greedy selector.

    With probability ε, selects a random category (exploration).
    With probability 1-ε, selects the category with highest weakness score
    (1 - correctness_rate).
    """

    def __init__(
        self,
        num_categories: int,
        epsilon: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_categories = num_categories
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def select_category(self) -> int:
        # Explore unvisited first
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_categories))

        # Greedy: pick weakest category
        rates = self.correct / np.maximum(self.attempts, 1)
        return int(np.argmin(rates))

    def update(self, category: int, correct: bool) -> None:
        self.attempts[category] += 1
        if correct:
            self.correct[category] += 1
        self.total_questions += 1

    def reset(self) -> None:
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class SlidingWindowUCBSelector(BaseSelector):
    """
    Sliding-Window UCB (SW-UCB) selector.

    Like UCB1 but only uses the last W observations per category,
    addressing non-stationarity from forgetting.

    Reference: Garivier & Moulines (2011), "On Upper-Confidence Bound
    Policies for Switching Bandit Problems."
    """

    def __init__(
        self,
        num_categories: int,
        window_size: int = 100,
        exploration_param: float = 1.414,
    ):
        self.num_categories = num_categories
        self.window_size = window_size
        self.exploration_param = exploration_param

        # Sliding window of (correct: bool) per category
        self.windows: List[deque] = [
            deque(maxlen=window_size) for _ in range(num_categories)
        ]
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def select_category(self) -> int:
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            w = self.windows[i]
            n_w = len(w)
            if n_w == 0:
                return i
            r_w = sum(w) / n_w

            weakness = 1 - r_w
            exploration = self.exploration_param * math.sqrt(
                math.log(self.total_questions + 1) / n_w
            )
            score = weakness + exploration
            if score > best_score:
                best_score = score
                best_cat = i

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        self.windows[category].append(1 if correct else 0)
        self.attempts[category] += 1
        if correct:
            self.correct[category] += 1
        self.total_questions += 1

    def reset(self) -> None:
        self.windows = [
            deque(maxlen=self.window_size) for _ in range(self.num_categories)
        ]
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class LeitnerSelector(BaseSelector):
    """
    Leitner system selector (spaced repetition heuristic).

    Categories are assigned to boxes (1 to max_box). Correct answers promote
    a category to the next box; incorrect answers demote it to box 1.
    Categories in lower boxes are reviewed more frequently.

    At each step, the category with the lowest box that is "due" is selected.
    A category in box b is due every 2^(b-1) steps since its last review.
    """

    def __init__(
        self,
        num_categories: int,
        max_box: int = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_categories = num_categories
        self.max_box = max_box
        self.rng = rng if rng is not None else np.random.default_rng()

        self.boxes = np.ones(num_categories, dtype=np.int32)  # All start in box 1
        self.last_reviewed = np.full(num_categories, -1, dtype=np.int32)
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def select_category(self) -> int:
        # Prioritize never-reviewed categories
        for i in range(self.num_categories):
            if self.last_reviewed[i] < 0:
                return i

        # Find due categories: time since last review >= 2^(box-1)
        time_since = self.total_questions - self.last_reviewed
        intervals = np.power(2, self.boxes - 1)
        urgency = time_since / intervals  # Higher = more overdue

        # Among the most overdue, pick the one in the lowest box
        # (ties broken by urgency)
        best_cat = int(np.argmax(urgency))
        return best_cat

    def update(self, category: int, correct: bool) -> None:
        if correct:
            self.boxes[category] = min(self.boxes[category] + 1, self.max_box)
            self.correct[category] += 1
        else:
            self.boxes[category] = 1  # Demote to box 1
        self.last_reviewed[category] = self.total_questions
        self.attempts[category] += 1
        self.total_questions += 1

    def reset(self) -> None:
        self.boxes = np.ones(self.num_categories, dtype=np.int32)
        self.last_reviewed = np.full(self.num_categories, -1, dtype=np.int32)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class OracleSelector(BaseSelector):
    """
    Oracle selector that has access to the student's true knowledge vector.

    Always selects the category with the lowest true knowledge. This provides
    an upper bound on performance — no real algorithm can do better.

    The oracle requires a reference to the student's knowledge array,
    which must be set via set_knowledge_ref() before use.
    """

    def __init__(self, num_categories: int):
        self.num_categories = num_categories
        self._knowledge_ref: Optional[np.ndarray] = None
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def set_knowledge_ref(self, knowledge: np.ndarray) -> None:
        """Set reference to the student's knowledge array."""
        self._knowledge_ref = knowledge

    def select_category(self) -> int:
        if self._knowledge_ref is None:
            return 0
        return int(np.argmin(self._knowledge_ref))

    def update(self, category: int, correct: bool) -> None:
        self.attempts[category] += 1
        if correct:
            self.correct[category] += 1
        self.total_questions += 1

    def reset(self) -> None:
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a,
                "correct": c,
                "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class WhittleIndexSelector(BaseSelector):
    """
    Whittle Index selector for the education-forgetting restless bandit.

    Precomputes the Whittle index table from known model parameters,
    then at each step selects the category with highest index based on
    its estimated knowledge state.

    Uses observed correctness rates as knowledge estimates (like UCB),
    but selects via principled Whittle index instead of heuristic score.
    """

    def __init__(
        self,
        num_categories: int,
        learning_rate: float = 0.12,
        incorrect_penalty: float = 0.02,
        decay_rate: float = 0.01,
        base_knowledge: float = 0.10,
    ):
        from .whittle import approximate_whittle_index

        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.incorrect_penalty = incorrect_penalty
        self.decay_rate = decay_rate
        self.base_knowledge = base_knowledge
        self._approx_whittle = approximate_whittle_index

        # Beta posterior per category for knowledge estimation
        self.alpha = np.ones(num_categories)
        self.beta_arr = np.ones(num_categories)

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)

    def _apply_posterior_forgetting(self) -> None:
        """Shrink posteriors toward prior for categories not recently quizzed."""
        for i in range(self.num_categories):
            if self.time_since_exposure[i] > 0:
                t = self.time_since_exposure[i]
                decay = math.exp(-self.decay_rate * t)
                self.alpha[i] = 1.0 + (self.alpha[i] - 1.0) * decay
                self.beta_arr[i] = 1.0 + (self.beta_arr[i] - 1.0) * decay

    def select_category(self) -> int:
        # Explore unvisited first
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        self._apply_posterior_forgetting()

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            a, b = self.alpha[i], self.beta_arr[i]
            k_est = a / (a + b)
            k_est = max(0.01, min(0.99, k_est))

            # Whittle index = active gain + forgetting opportunity cost
            W = self._approx_whittle(
                k_est, self.learning_rate, self.incorrect_penalty,
                self.decay_rate, self.base_knowledge,
            )

            # Add urgency for time since exposure
            t = self.time_since_exposure[i]
            urgency = (k_est - self.base_knowledge) * (1 - math.exp(-self.decay_rate * t))
            W += urgency

            if W > best_score:
                best_score = W
                best_cat = i

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        if correct:
            self.alpha[category] += 1
            self.correct[category] += 1
        else:
            self.beta_arr[category] += 1
        self.attempts[category] += 1
        self.total_questions += 1
        self.time_since_exposure += 1
        self.time_since_exposure[category] = 0

    def reset(self) -> None:
        self.alpha = np.ones(self.num_categories)
        self.beta_arr = np.ones(self.num_categories)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(self.num_categories, dtype=np.int32)

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a, "correct": c, "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class PDWhittleSelector(BaseSelector):
    """
    Parametric-Decay Whittle (PD-Whittle) selector.

    Online learning version that maintains Bayesian estimates of per-arm
    decay parameters and recomputes Whittle indices as estimates improve.

    Uses approximate Whittle index (closed-form) for efficiency, with
    decay-aware exploration bonus.
    """

    def __init__(
        self,
        num_categories: int,
        learning_rate: float = 0.12,
        incorrect_penalty: float = 0.02,
        decay_rate_prior: float = 0.01,
        base_knowledge: float = 0.10,
        exploration_param: float = 1.414,
        rng: Optional[np.random.Generator] = None,
    ):
        from .whittle import approximate_whittle_index

        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.incorrect_penalty = incorrect_penalty
        self.base_knowledge = base_knowledge
        self.exploration_param = exploration_param
        self._approx_whittle = approximate_whittle_index
        self.rng = rng if rng is not None else np.random.default_rng()

        # Per-category Bayesian estimates
        # Track running correctness as knowledge proxy
        self.alpha = np.ones(num_categories)  # Beta posterior: correct + 1
        self.beta_ = np.ones(num_categories)  # Beta posterior: incorrect + 1

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)

        # Per-arm decay rate estimate (start with prior)
        self.decay_estimates = np.full(num_categories, decay_rate_prior)
        self.decay_observations = [[] for _ in range(num_categories)]

    def select_category(self) -> int:
        # Explore unvisited first
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            # Knowledge estimate from Beta posterior, adjusted for decay
            a, b = self.alpha[i], self.beta_[i]
            k_est = a / (a + b)
            t = self.time_since_exposure[i]
            decay = math.exp(-self.decay_estimates[i] * t)
            k_est = k_est * decay + self.base_knowledge * (1 - decay)
            k_est = max(0.01, min(0.99, k_est))

            # Approximate Whittle index
            W = self._approx_whittle(
                k_est, self.learning_rate, self.incorrect_penalty,
                self.decay_estimates[i], self.base_knowledge,
            )

            # Decay-aware exploration bonus
            n_i = self.attempts[i]
            posterior_std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            urgency = 1.0 + self.decay_estimates[i] * t
            bonus = self.exploration_param * posterior_std * urgency * math.sqrt(
                math.log(self.total_questions + 1) / n_i
            )

            score = W + bonus
            if score > best_score:
                best_score = score
                best_cat = i

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        if correct:
            self.alpha[category] += 1
            self.correct[category] += 1
        else:
            self.beta_[category] += 1
        self.attempts[category] += 1
        self.total_questions += 1

        # Update decay estimate: if we haven't seen this category in a while
        # and the student gets it wrong more than expected, decay is higher
        t = self.time_since_exposure[category]
        if t > 0 and self.attempts[category] > 3:
            expected_k = (self.alpha[category] - 1) / max(1, self.attempts[category] - 1)
            actual = 1.0 if correct else 0.0
            # Simple exponential smoothing of decay estimate
            if not correct and expected_k > 0.5:
                # Student was expected to know this but got it wrong → decay higher
                self.decay_estimates[category] *= 1.05
            elif correct and expected_k < 0.5:
                # Student was expected to NOT know but got it right → decay lower
                self.decay_estimates[category] *= 0.95
            self.decay_estimates[category] = max(1e-4, min(0.5, self.decay_estimates[category]))

        self.time_since_exposure += 1
        self.time_since_exposure[category] = 0

    def reset(self) -> None:
        self.alpha = np.ones(self.num_categories)
        self.beta_ = np.ones(self.num_categories)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(self.num_categories, dtype=np.int32)

    def get_statistics(self) -> dict:
        stats = {}
        for i in range(self.num_categories):
            a = int(self.attempts[i])
            c = int(self.correct[i])
            stats[f"Category_{i}"] = {
                "attempts": a, "correct": c, "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


# Registry mapping algorithm names to selector factory functions
SELECTOR_REGISTRY = {
    "random": lambda cfg, rng: RandomSelector(cfg.num_categories, rng=rng),
    "ucb1": lambda cfg, rng: UCBSelectorAdapter(cfg),
    "fucb": lambda cfg, rng: FUCBSelector(
        cfg.num_categories,
        exploration_param=cfg.exploration_param,
        decay_rate=cfg.decay_rate,
        urgency_weight=0.5,
    ),
    "bkt_bandit": lambda cfg, rng: BKTBanditSelector(
        cfg.num_categories,
        exploration_param=cfg.exploration_param,
        decay_rate=cfg.decay_rate,
    ),
    "bkt_thompson": lambda cfg, rng: BKTThompsonSelector(
        cfg.num_categories,
        decay_rate=cfg.decay_rate,
        rng=rng,
    ),
    "thompson": lambda cfg, rng: ThompsonSelector(
        cfg.num_categories,
        rng=rng,
    ),
    "epsilon_greedy": lambda cfg, rng: EpsilonGreedySelector(
        cfg.num_categories,
        epsilon=0.1,
        rng=rng,
    ),
    "sw_ucb": lambda cfg, rng: SlidingWindowUCBSelector(
        cfg.num_categories,
        window_size=100,
        exploration_param=cfg.exploration_param,
    ),
    "leitner": lambda cfg, rng: LeitnerSelector(
        cfg.num_categories,
        rng=rng,
    ),
    "oracle": lambda cfg, rng: OracleSelector(cfg.num_categories),
    "whittle": lambda cfg, rng: WhittleIndexSelector(
        cfg.num_categories,
        learning_rate=cfg.learning_rate,
        incorrect_penalty=cfg.incorrect_penalty,
        decay_rate=cfg.decay_rate,
        base_knowledge=cfg.base_knowledge,
    ),
    "pd_whittle": lambda cfg, rng: PDWhittleSelector(
        cfg.num_categories,
        learning_rate=cfg.learning_rate,
        incorrect_penalty=cfg.incorrect_penalty,
        decay_rate_prior=cfg.decay_rate,
        base_knowledge=cfg.base_knowledge,
        exploration_param=cfg.exploration_param,
        rng=rng,
    ),
}


def create_selector(algorithm: str, config: ExperimentConfig,
                    rng: Optional[np.random.Generator] = None) -> BaseSelector:
    """Create a selector by algorithm name."""
    if algorithm not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(SELECTOR_REGISTRY.keys())}"
        )
    return SELECTOR_REGISTRY[algorithm](config, rng)
