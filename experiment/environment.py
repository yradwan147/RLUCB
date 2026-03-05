"""Gymnasium-compatible RL environment for quiz simulation."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import ExperimentConfig
from .student import Student


class QuizEnvironment(gym.Env):
    """
    Gymnasium-compatible environment for simulating a quiz session.
    
    This environment models a single student answering questions across
    multiple categories. The agent (question selector) chooses which
    category to quiz next, and receives rewards based on the student's
    knowledge improvement.
    
    Observation Space:
        Box(0, 1, shape=(num_categories,)) - Student's current knowledge vector
    
    Action Space:
        Discrete(num_categories) - Which category to quiz next
    
    Reward:
        Configurable - can be based on knowledge gain, correct answers, etc.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: ExperimentConfig,
        reward_type: str = "knowledge_gain",
        render_mode: Optional[str] = None,
        student_id: int = 0,
    ):
        """
        Initialize the quiz environment.
        
        Args:
            config: Experiment configuration
            reward_type: Type of reward function to use:
                - "knowledge_gain": reward = sum(new_knowledge) - sum(old_knowledge)
                - "correct_answer": reward = 1.0 if correct else -0.1
                - "balanced": reward = 2 * weak_improvement + overall_improvement
            render_mode: How to render the environment
            student_id: ID for the student in this environment
        """
        super().__init__()
        
        self.config = config
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.student_id = student_id
        
        # Create random generator
        seed = config.random_seed
        if seed is not None:
            seed = seed + student_id  # Different seed per student
        self.rng = np.random.default_rng(seed)
        
        # Create student
        self.student = Student(config, student_id=student_id, rng=self.rng)
        
        # Define observation space: knowledge vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.num_categories,),
            dtype=np.float32,
        )
        
        # Define action space: which category to quiz
        self.action_space = spaces.Discrete(config.num_categories)
        
        # Track episode state
        self.current_step = 0
        self.max_steps = config.questions_per_session
        
        # Track metrics
        self.episode_rewards = []
        self.episode_knowledge_history = []
        self.episode_correct_count = 0
        self.episode_incorrect_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.student = Student(
                self.config,
                student_id=self.student_id,
                rng=self.rng,
            )
        else:
            self.student.reset()
        
        self.current_step = 0
        self.episode_rewards = []
        self.episode_knowledge_history = [self.student.get_knowledge_snapshot()]
        self.episode_correct_count = 0
        self.episode_incorrect_count = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Category index to quiz
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        category = action
        previous_knowledge = self.student.get_knowledge_snapshot()
        previous_weakest = self.student.get_weakest_category()
        previous_weakest_knowledge = previous_knowledge[previous_weakest]
        
        # Student answers question
        correct = self.student.answer(category)
        
        # Update knowledge based on answer
        self.student.update_knowledge(category, correct, timestep=self.current_step)
        
        # Apply forgetting to all categories
        self.student.apply_forgetting()
        
        # Get new state
        current_knowledge = self.student.get_knowledge_snapshot()
        
        # Calculate reward
        reward = self._calculate_reward(
            previous_knowledge,
            current_knowledge,
            previous_weakest,
            previous_weakest_knowledge,
            correct,
        )
        
        # Update tracking
        self.current_step += 1
        self.episode_rewards.append(reward)
        self.episode_knowledge_history.append(current_knowledge)
        
        if correct:
            self.episode_correct_count += 1
        else:
            self.episode_incorrect_count += 1
        
        # Check if episode is done
        terminated = False  # No natural termination condition
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        info["correct"] = correct
        info["category_quizzed"] = category
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (knowledge vector)."""
        return self.student.knowledge.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary."""
        return {
            "step": self.current_step,
            "average_knowledge": self.student.get_average_knowledge(),
            "knowledge_variance": self.student.get_knowledge_variance(),
            "weakest_category": self.student.get_weakest_category(),
            "strongest_category": self.student.get_strongest_category(),
            "total_correct": self.episode_correct_count,
            "total_incorrect": self.episode_incorrect_count,
        }
    
    def _calculate_reward(
        self,
        previous_knowledge: np.ndarray,
        current_knowledge: np.ndarray,
        previous_weakest: int,
        previous_weakest_knowledge: float,
        correct: bool,
    ) -> float:
        """
        Calculate reward based on configured reward type.
        
        Args:
            previous_knowledge: Knowledge before this step
            current_knowledge: Knowledge after this step
            previous_weakest: Index of weakest category before step
            previous_weakest_knowledge: Knowledge in weakest category before step
            correct: Whether the answer was correct
            
        Returns:
            Reward value
        """
        if self.reward_type == "knowledge_gain":
            # Reward = total knowledge gained
            reward = float(np.sum(current_knowledge) - np.sum(previous_knowledge))
            
        elif self.reward_type == "correct_answer":
            # Simple binary reward
            reward = 1.0 if correct else -0.1
            
        elif self.reward_type == "balanced":
            # Focus on weak areas + overall improvement
            weak_improvement = current_knowledge[previous_weakest] - previous_weakest_knowledge
            overall_improvement = np.sum(current_knowledge) - np.sum(previous_knowledge)
            reward = 2.0 * weak_improvement + overall_improvement
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        return reward
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "human":
            print(self._render_text())
            return None
        elif self.render_mode == "ansi":
            return self._render_text()
        return None
    
    def _render_text(self) -> str:
        """Generate text representation of current state."""
        lines = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Knowledge: {np.round(self.student.knowledge, 3)}",
            f"Average: {self.student.get_average_knowledge():.3f}",
            f"Correct: {self.episode_correct_count}, Incorrect: {self.episode_incorrect_count}",
        ]
        return "\n".join(lines)
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the completed episode."""
        return {
            "total_steps": self.current_step,
            "total_reward": sum(self.episode_rewards),
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "final_knowledge": self.student.knowledge.tolist(),
            "final_average_knowledge": self.student.get_average_knowledge(),
            "final_knowledge_variance": self.student.get_knowledge_variance(),
            "total_correct": self.episode_correct_count,
            "total_incorrect": self.episode_incorrect_count,
            "accuracy": (
                self.episode_correct_count / 
                max(1, self.episode_correct_count + self.episode_incorrect_count)
            ),
            "knowledge_history": [k.tolist() for k in self.episode_knowledge_history],
            "student_statistics": self.student.get_statistics(),
        }
