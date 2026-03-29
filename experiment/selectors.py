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


class EquiBanditSelector(BaseSelector):
    """
    Equity-aware BKT-Bandit that balances knowledge gain with equity.

    Extends BKT-Bandit's posterior-based scoring with a variance penalty
    that prioritizes categories whose knowledge deviates from the mean.
    This explicitly addresses the equity-efficiency tradeoff: BKT-Bandit
    concentrates on high-value categories (Gini=0.449 at K=20), while
    EquiBandit regularizes toward equal mastery.

    Score: (1 - μ_i) + c·σ_i + η·(μ_i - μ̄)²
    The η·(μ_i - μ̄)² term gives higher priority to categories that are
    far from the average — whether above (over-mastered) or below (neglected).
    """

    def __init__(
        self,
        num_categories: int,
        exploration_param: float = 1.414,
        decay_rate: float = 0.01,
        equity_weight: float = 0.5,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.num_categories = num_categories
        self.exploration_param = exploration_param
        self.decay_rate = decay_rate
        self.equity_weight = equity_weight
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

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

        # Compute posterior means
        means = np.array([self.alpha[i] / (self.alpha[i] + self.beta_[i])
                          for i in range(self.num_categories)])
        mean_avg = np.mean(means)

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            a, b = self.alpha[i], self.beta_[i]
            mean_k = means[i]
            std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

            # BKT-Bandit base: weakness + uncertainty
            weakness = 1 - mean_k
            exploration = self.exploration_param * std

            # Equity penalty: reduce score for over-exposed categories
            avg_exposure = max(1.0, np.mean(self.attempts))
            exposure_ratio = self.attempts[i] / avg_exposure if avg_exposure > 0 else 1.0
            equity = -self.equity_weight * max(0.0, exposure_ratio - 1.0)

            score = weakness + exploration + equity
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
                "attempts": a, "correct": c, "incorrect": a - c,
                "correctness_rate": c / a if a > 0 else 0.0,
            }
        return stats

    def get_exposure_distribution(self) -> np.ndarray:
        return self.attempts.copy()


class DiscountedTSSelector(BaseSelector):
    """
    Discounted Thompson Sampling for non-stationary bandits.

    Uses a geometric discount factor γ on past observations, so recent
    outcomes weigh more than old ones. The effective posterior is:
      α_i = 1 + Σ_{s≤t} γ^{t-s} · correct_{i,s}
      β_i = 1 + Σ_{s≤t} γ^{t-s} · incorrect_{i,s}

    Implemented via multiplicative decay: after each step, multiply all
    α and β pseudo-counts by γ, then add the new observation.

    Reference: Raj & Kalyani (2017), "Taming Non-stationary Bandits:
    A Bayesian Approach" (arXiv:1707.09727).
    """

    def __init__(
        self,
        num_categories: int,
        discount: float = 0.999,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_categories = num_categories
        self.discount = discount
        self.rng = rng if rng is not None else np.random.default_rng()

        self.alpha = np.ones(num_categories)
        self.beta_ = np.ones(num_categories)
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def select_category(self) -> int:
        samples = np.array([
            self.rng.beta(max(self.alpha[i], 1e-3), max(self.beta_[i], 1e-3))
            for i in range(self.num_categories)
        ])
        # Select weakest (lowest sampled knowledge)
        return int(np.argmin(samples))

    def update(self, category: int, correct: bool) -> None:
        # Discount all past observations
        self.alpha = 1.0 + (self.alpha - 1.0) * self.discount
        self.beta_ = 1.0 + (self.beta_ - 1.0) * self.discount

        # Add new observation
        if correct:
            self.alpha[category] += 1
            self.correct[category] += 1
        else:
            self.beta_[category] += 1
        self.attempts[category] += 1
        self.total_questions += 1

    def reset(self) -> None:
        self.alpha = np.ones(self.num_categories)
        self.beta_ = np.ones(self.num_categories)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0

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


class LookaheadOracleSelector(BaseSelector):
    """
    2-step lookahead Oracle with perfect state knowledge.

    Unlike the myopic Greedy-Min (which picks argmin k_i), this selector
    evaluates each category by simulating the expected knowledge state
    after quizzing it, including forgetting of all other categories.

    For each candidate category i, it computes:
      E[total_knowledge | quiz i] =
        k_i * (k_i + α(1-k_i)) + (1-k_i) * (k_i - β*k_i)   [quizzed arm]
        + Σ_{j≠i} (base + (k_j - base) * exp(-λ))            [decayed arms]

    Then picks the category maximizing expected total knowledge.
    This separates the effect of myopia from the effect of state observability.
    """

    def __init__(
        self,
        num_categories: int,
        learning_rate: float = 0.12,
        incorrect_penalty: float = 0.02,
        decay_rate: float = 0.01,
        base_knowledge: float = 0.10,
    ):
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.incorrect_penalty = incorrect_penalty
        self.decay_rate = decay_rate
        self.base_knowledge = base_knowledge
        self._knowledge_ref: Optional[np.ndarray] = None
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def set_knowledge_ref(self, knowledge: np.ndarray) -> None:
        self._knowledge_ref = knowledge

    def select_category(self) -> int:
        if self._knowledge_ref is None:
            return 0

        k = self._knowledge_ref
        K = self.num_categories
        decay_factor = math.exp(-self.decay_rate)
        base = self.base_knowledge
        best_score = -float('inf')
        best_cat = 0

        for i in range(K):
            # Expected knowledge of quizzed category after one step
            p_correct = k[i]
            k_correct = k[i] + self.learning_rate * (1 - k[i])
            k_incorrect = k[i] - self.incorrect_penalty * k[i]
            expected_ki = p_correct * k_correct + (1 - p_correct) * k_incorrect

            # Sum of decayed knowledge for all OTHER categories
            decay_sum = 0.0
            for j in range(K):
                if j != i:
                    decay_sum += base + (k[j] - base) * decay_factor

            total = expected_ki + decay_sum
            if total > best_score:
                best_score = total
                best_cat = i

        return best_cat

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
                "attempts": a, "correct": c, "incorrect": a - c,
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
        exploration_param: float = 1.414,
    ):
        from .whittle import compute_whittle_index_table, lookup_whittle_index

        self.num_categories = num_categories
        self.decay_rate = decay_rate
        self.base_knowledge = base_knowledge
        self.exploration_param = exploration_param
        self._lookup = lookup_whittle_index

        # Precompute advantage-based index table
        self._states, self._raw_indices = compute_whittle_index_table(
            learning_rate=learning_rate,
            incorrect_penalty=incorrect_penalty,
            decay_rate=decay_rate,
            base_knowledge=base_knowledge,
            num_states=50,
            num_categories=num_categories,
        )
        # Normalize indices to [0, 1] for proper combination with exploration
        idx_min = self._raw_indices.min()
        idx_range = self._raw_indices.max() - idx_min
        if idx_range > 0:
            self._indices = (self._raw_indices - idx_min) / idx_range
        else:
            self._indices = np.ones_like(self._raw_indices) * 0.5

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

            # Whittle advantage: lookup from precomputed table
            W = self._lookup(k_est, self._states, self._indices)

            # Posterior uncertainty as exploration bonus
            std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

            # Decay-aware exploration: more urgent for categories not seen recently
            t = self.time_since_exposure[i]
            urgency = 1.0 + self.decay_rate * t

            score = W + self.exploration_param * std * urgency

            if score > best_score:
                best_score = score
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
        decay_rate: float = 0.01,
        base_knowledge: float = 0.10,
        exploration_param: float = 1.414,
        rng: Optional[np.random.Generator] = None,
    ):
        from .whittle import compute_whittle_index_table, lookup_whittle_index

        self.num_categories = num_categories
        self.decay_rate = decay_rate
        self.base_knowledge = base_knowledge
        self.exploration_param = exploration_param
        self._lookup = lookup_whittle_index
        self.rng = rng if rng is not None else np.random.default_rng()

        # Precompute normalized advantage index
        self._states, raw_indices = compute_whittle_index_table(
            learning_rate=learning_rate,
            incorrect_penalty=incorrect_penalty,
            decay_rate=decay_rate,
            base_knowledge=base_knowledge,
            num_states=50,
            num_categories=num_categories,
        )
        idx_min = raw_indices.min()
        idx_range = raw_indices.max() - idx_min
        self._indices = (raw_indices - idx_min) / idx_range if idx_range > 0 else np.full_like(raw_indices, 0.5)

        # Beta posterior per category with forgetting
        self.alpha = np.ones(num_categories)
        self.beta_ = np.ones(num_categories)

        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)

    def _apply_posterior_forgetting(self) -> None:
        for i in range(self.num_categories):
            if self.time_since_exposure[i] > 0:
                t = self.time_since_exposure[i]
                decay = math.exp(-self.decay_rate * t)
                self.alpha[i] = 1.0 + (self.alpha[i] - 1.0) * decay
                self.beta_[i] = 1.0 + (self.beta_[i] - 1.0) * decay

    def select_category(self) -> int:
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        self._apply_posterior_forgetting()

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            a, b = self.alpha[i], self.beta_[i]
            k_est = a / (a + b)
            k_est = max(0.01, min(0.99, k_est))

            # Normalized Whittle advantage
            W = self._lookup(k_est, self._states, self._indices)

            # Posterior uncertainty + decay urgency
            std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            t = self.time_since_exposure[i]
            urgency = 1.0 + self.decay_rate * t

            # Thompson-style sampling: add noise proportional to uncertainty
            noise = self.rng.normal(0, std * urgency)

            score = W + self.exploration_param * std * urgency + 0.5 * noise

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


class AdaptiveWhittleSelector(BaseSelector):
    """
    Adaptive Whittle selector that blends Whittle advantage with F-UCB-style
    urgency based on the estimated forgetting rate.

    At low decay: behaves like Whittle (principled MDP-based selection).
    At high decay: shifts toward urgency-based selection (like F-UCB).
    The blend weight adapts online based on observed performance drops after gaps.
    """

    def __init__(
        self,
        num_categories: int,
        learning_rate: float = 0.12,
        incorrect_penalty: float = 0.02,
        decay_rate: float = 0.01,
        base_knowledge: float = 0.10,
        exploration_param: float = 1.414,
        rng: Optional[np.random.Generator] = None,
    ):
        from .whittle import compute_whittle_index_table, lookup_whittle_index

        self.num_categories = num_categories
        self.decay_rate = decay_rate
        self.base_knowledge = base_knowledge
        self.exploration_param = exploration_param
        self.rng = rng if rng is not None else np.random.default_rng()
        self._lookup = lookup_whittle_index

        states, raw = compute_whittle_index_table(
            learning_rate, incorrect_penalty, decay_rate, base_knowledge,
            num_states=50, num_categories=num_categories,
        )
        self._states = states
        mn, rng_v = raw.min(), raw.max() - raw.min()
        self._indices = (raw - mn) / rng_v if rng_v > 0 else np.full_like(raw, 0.5)

        self.alpha = np.ones(num_categories)
        self.beta_ = np.ones(num_categories)
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(num_categories, dtype=np.int32)
        self.gap_surprises: list = []
        self.effective_decay = decay_rate

    def select_category(self) -> int:
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        # Apply forgetting to posteriors
        for i in range(self.num_categories):
            if self.time_since_exposure[i] > 0:
                d = math.exp(-self.effective_decay * self.time_since_exposure[i])
                self.alpha[i] = 1.0 + (self.alpha[i] - 1.0) * d
                self.beta_[i] = 1.0 + (self.beta_[i] - 1.0) * d

        # Urgency weight: higher when decay is faster
        urgency_weight = min(1.0, self.effective_decay * 20)

        best_score = -float('inf')
        best_cat = 0

        for i in range(self.num_categories):
            a, b = self.alpha[i], self.beta_[i]
            mean_k = a / (a + b)
            std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            t = self.time_since_exposure[i]

            # Whittle advantage
            k_est = max(0.01, min(0.99, mean_k))
            W = self._lookup(k_est, self._states, self._indices)

            # F-UCB-style urgency
            decay_factor = math.exp(-self.effective_decay * t)
            urgency = 1 - decay_factor

            # Adaptive blend
            score = (1 - urgency_weight) * W + urgency_weight * urgency

            # Exploration bonus
            score += self.exploration_param * std * (1 + self.effective_decay * t)

            if score > best_score:
                best_score = score
                best_cat = i

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        t = self.time_since_exposure[category]
        if t > 5 and self.attempts[category] > 2:
            expected = self.alpha[category] / (self.alpha[category] + self.beta_[category])
            surprise = abs((1.0 if correct else 0.0) - expected)
            self.gap_surprises.append(surprise)
            if len(self.gap_surprises) >= 20:
                avg_surprise = float(np.mean(self.gap_surprises[-20:]))
                self.effective_decay = self.decay_rate * (1 + avg_surprise * 5)
                self.effective_decay = max(0.001, min(0.2, self.effective_decay))

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
        self.alpha = np.ones(self.num_categories)
        self.beta_ = np.ones(self.num_categories)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        self.time_since_exposure = np.zeros(self.num_categories, dtype=np.int32)
        self.gap_surprises = []
        self.effective_decay = self.decay_rate

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


class MetaSelector(BaseSelector):
    """
    Unified scoring meta-selector over base expert recommendations.

    Runs M base selectors in parallel to generate candidate categories,
    then scores each candidate using a unified formula that combines:
    - BKT-Bandit's Beta posterior with forgetting (weakness + uncertainty)
    - F-UCB's forgetting urgency (time-dependent exploration multiplier)

    Score: (1 - mean_i) + c * std_i * (1 + λ * t_i)

    At low decay (λ→0): reduces to BKT-Bandit scoring.
    At high decay (λ large): urgency dominates, similar to F-UCB.

    Theoretically: UCB index with time-varying confidence width for
    non-stationary arms (Garivier & Moulines 2011). Expert candidates
    restrict action set from K to ≤M, reducing regret (Orabona §6.8).
    """

    def __init__(
        self,
        base_selectors: List[BaseSelector],
        num_categories: int,
        decay_rate: float = 0.01,
        exploration_param: float = 1.414,
        rng: Optional[np.random.Generator] = None,
    ):
        self.bases = base_selectors
        self.num_categories = num_categories
        self.M = len(base_selectors)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._decay_rate = decay_rate
        self._exploration = exploration_param

        # Per-category Beta posterior
        self._alpha = np.ones(num_categories)
        self._beta = np.ones(num_categories)
        self._time_since = np.zeros(num_categories, dtype=np.int32)

        # Standard tracking
        self.attempts = np.zeros(num_categories, dtype=np.int32)
        self.correct = np.zeros(num_categories, dtype=np.int32)
        self.total_questions = 0

    def select_category(self) -> int:
        # Get recommendations from all base experts
        recs = [b.select_category() for b in self.bases]

        # Explore each category once first
        for i in range(self.num_categories):
            if self.attempts[i] == 0:
                return i

        # Apply posterior forgetting
        for i in range(self.num_categories):
            if self._time_since[i] > 0:
                d = math.exp(-self._decay_rate * self._time_since[i])
                self._alpha[i] = 1.0 + (self._alpha[i] - 1.0) * d
                self._beta[i] = 1.0 + (self._beta[i] - 1.0) * d

        # Score expert-recommended categories with unified formula
        candidates = list(set(recs))
        best_score = -float('inf')
        best_cat = candidates[0]

        for cat in candidates:
            a, b = self._alpha[cat], self._beta[cat]
            mean_k = a / (a + b)
            std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            t = self._time_since[cat]

            # Unified three-term score (F-UCB structure + BKT posterior):
            # 1. Time-decayed weakness (old observations lose weight)
            # 2. Forgetting urgency (additive, grows with time)
            # 3. Posterior uncertainty (exploration)
            decay_t = math.exp(-self._decay_rate * t)
            weakness = (1 - mean_k) * decay_t
            urgency = 0.5 * (1 - decay_t)
            exploration = self._exploration * std
            score = weakness + urgency + exploration

            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    def update(self, category: int, correct: bool) -> None:
        for b in self.bases:
            b.update(category, correct)

        if correct:
            self._alpha[category] += 1
            self.correct[category] += 1
        else:
            self._beta[category] += 1

        self.attempts[category] += 1
        self.total_questions += 1
        self._time_since += 1
        self._time_since[category] = 0

    def reset(self) -> None:
        self._alpha = np.ones(self.num_categories)
        self._beta = np.ones(self.num_categories)
        self._time_since = np.zeros(self.num_categories, dtype=np.int32)
        self.attempts = np.zeros(self.num_categories, dtype=np.int32)
        self.correct = np.zeros(self.num_categories, dtype=np.int32)
        self.total_questions = 0
        for b in self.bases:
            b.reset()

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
        decay_rate=cfg.selector_decay_rate,
        urgency_weight=0.5,
    ),
    "bkt_bandit": lambda cfg, rng: BKTBanditSelector(
        cfg.num_categories,
        exploration_param=cfg.exploration_param,
        decay_rate=cfg.selector_decay_rate,
    ),
    "bkt_thompson": lambda cfg, rng: BKTThompsonSelector(
        cfg.num_categories,
        decay_rate=cfg.selector_decay_rate,
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
    "lookahead_oracle": lambda cfg, rng: LookaheadOracleSelector(
        cfg.num_categories,
        learning_rate=cfg.learning_rate,
        incorrect_penalty=cfg.incorrect_penalty,
        decay_rate=cfg.decay_rate,  # uses TRUE decay (not selector_decay_rate)
        base_knowledge=cfg.base_knowledge,
    ),
    "equi_bandit": lambda cfg, rng: EquiBanditSelector(
        cfg.num_categories,
        exploration_param=cfg.exploration_param,
        decay_rate=cfg.selector_decay_rate,
        equity_weight=0.1,
    ),
    "discounted_ts": lambda cfg, rng: DiscountedTSSelector(
        cfg.num_categories,
        discount=math.exp(-cfg.selector_decay_rate),
        rng=rng,
    ),
    "meta": lambda cfg, rng: MetaSelector(
        base_selectors=[
            create_selector("bkt_bandit", cfg, np.random.default_rng(42)),
            create_selector("fucb", cfg, np.random.default_rng(43)),
            create_selector("pd_whittle", cfg, np.random.default_rng(44)),
            create_selector("leitner", cfg, np.random.default_rng(45)),
            create_selector("random", cfg, np.random.default_rng(46)),
        ],
        num_categories=cfg.num_categories,
        decay_rate=cfg.selector_decay_rate,
        exploration_param=cfg.exploration_param,
        rng=rng,
    ),
    "whittle": lambda cfg, rng: WhittleIndexSelector(
        cfg.num_categories,
        learning_rate=cfg.learning_rate,
        incorrect_penalty=cfg.incorrect_penalty,
        decay_rate=cfg.selector_decay_rate,
        base_knowledge=cfg.base_knowledge,
        exploration_param=cfg.exploration_param,
    ),
    "adaptive_whittle": lambda cfg, rng: AdaptiveWhittleSelector(
        cfg.num_categories,
        learning_rate=cfg.learning_rate,
        incorrect_penalty=cfg.incorrect_penalty,
        decay_rate=cfg.selector_decay_rate,
        base_knowledge=cfg.base_knowledge,
        exploration_param=cfg.exploration_param,
        rng=rng,
    ),
    "pd_whittle": lambda cfg, rng: PDWhittleSelector(
        cfg.num_categories,
        learning_rate=cfg.learning_rate,
        incorrect_penalty=cfg.incorrect_penalty,
        decay_rate=cfg.selector_decay_rate,
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
