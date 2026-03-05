"""Main experiment runner for UCB vs Random quizzing comparison."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ExperimentConfig
from .student import Student
from .selectors import RandomSelector, UCBSelectorAdapter


@dataclass
class GroupMetrics:
    """Metrics collected for a group of students at each timestep."""
    timestep: int
    average_knowledge: float
    knowledge_std: float
    per_category_knowledge: List[float]
    per_category_knowledge_std: List[float]
    weakest_category_avg: float
    strongest_category_avg: float
    knowledge_variance_avg: float
    cumulative_correct: int
    cumulative_incorrect: int
    exposure_distribution: List[int]


@dataclass
class ExperimentResults:
    """Complete results from an experiment run."""
    config: ExperimentConfig
    ucb_metrics: List[GroupMetrics]
    control_metrics: List[GroupMetrics]
    ucb_final_students: List[Dict]
    control_final_students: List[Dict]
    
    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert results to pandas DataFrames for analysis."""
        ucb_data = self._metrics_to_dict(self.ucb_metrics, "ucb")
        control_data = self._metrics_to_dict(self.control_metrics, "control")
        
        ucb_df = pd.DataFrame(ucb_data)
        control_df = pd.DataFrame(control_data)
        
        return ucb_df, control_df
    
    def _metrics_to_dict(self, metrics: List[GroupMetrics], group_name: str) -> Dict:
        """Convert list of GroupMetrics to dictionary for DataFrame."""
        data = {
            "timestep": [],
            "group": [],
            "average_knowledge": [],
            "knowledge_std": [],
            "weakest_category_avg": [],
            "strongest_category_avg": [],
            "knowledge_variance_avg": [],
            "cumulative_correct": [],
            "cumulative_incorrect": [],
            "accuracy": [],
        }
        
        # Add per-category columns
        num_categories = len(metrics[0].per_category_knowledge) if metrics else 0
        for i in range(num_categories):
            data[f"category_{i}_knowledge"] = []
            data[f"category_{i}_exposure"] = []
        
        for m in metrics:
            data["timestep"].append(m.timestep)
            data["group"].append(group_name)
            data["average_knowledge"].append(m.average_knowledge)
            data["knowledge_std"].append(m.knowledge_std)
            data["weakest_category_avg"].append(m.weakest_category_avg)
            data["strongest_category_avg"].append(m.strongest_category_avg)
            data["knowledge_variance_avg"].append(m.knowledge_variance_avg)
            data["cumulative_correct"].append(m.cumulative_correct)
            data["cumulative_incorrect"].append(m.cumulative_incorrect)
            
            total = m.cumulative_correct + m.cumulative_incorrect
            data["accuracy"].append(m.cumulative_correct / total if total > 0 else 0)
            
            for i in range(num_categories):
                data[f"category_{i}_knowledge"].append(m.per_category_knowledge[i])
                data[f"category_{i}_exposure"].append(m.exposure_distribution[i])
        
        return data
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """Get a single DataFrame with both groups."""
        ucb_df, control_df = self.to_dataframe()
        return pd.concat([ucb_df, control_df], ignore_index=True)
    
    def save_to_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        df = self.get_combined_dataframe()
        df.to_csv(filepath, index=False)


class Experiment:
    """
    Runs the UCB vs Random quizzing experiment.
    
    This class manages two groups of simulated students:
    - UCB Group: Questions selected using UCB algorithm (adaptive)
    - Control Group: Questions selected uniformly at random
    
    It tracks and compares how each group's knowledge evolves over time.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Create random generator for reproducibility
        self.rng = np.random.default_rng(config.random_seed)
        
        # Initialize student groups
        self.ucb_group: List[Student] = []
        self.control_group: List[Student] = []
        
        # Initialize selectors (one per student for individual tracking)
        self.ucb_selectors: List[UCBSelectorAdapter] = []
        self.control_selectors: List[RandomSelector] = []
        
        # Metrics storage
        self.ucb_metrics: List[GroupMetrics] = []
        self.control_metrics: List[GroupMetrics] = []
        
        # Initialize groups
        self._init_groups()
    
    def _init_groups(self) -> None:
        """Initialize both student groups and their selectors."""
        for i in range(self.config.num_students_per_group):
            # Create student RNG with different seed for each
            student_seed = None
            if self.config.random_seed is not None:
                student_seed = self.config.random_seed + i
            
            student_rng = np.random.default_rng(student_seed)
            
            # UCB group
            ucb_student = Student(
                self.config,
                student_id=i,
                rng=np.random.default_rng(student_seed),
            )
            self.ucb_group.append(ucb_student)
            self.ucb_selectors.append(UCBSelectorAdapter(self.config))
            
            # Control group (same initial conditions)
            control_student = Student(
                self.config,
                student_id=i + self.config.num_students_per_group,
                rng=np.random.default_rng(student_seed),  # Same seed for fair comparison
            )
            self.control_group.append(control_student)
            
            control_rng = np.random.default_rng(
                student_seed + 1000000 if student_seed else None
            )
            self.control_selectors.append(
                RandomSelector(self.config.num_categories, rng=control_rng)
            )
    
    def run(self, show_progress: bool = True) -> ExperimentResults:
        """
        Run the complete experiment.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            ExperimentResults containing all metrics and final states
        """
        # Record initial state
        self._log_metrics(timestep=0)
        
        # Create progress bar
        timesteps = range(1, self.config.questions_per_session + 1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Running experiment")
        
        for timestep in timesteps:
            # Process UCB group
            self._process_group_step(
                students=self.ucb_group,
                selectors=self.ucb_selectors,
                is_ucb=True,
                timestep=timestep,
            )
            
            # Process Control group
            self._process_group_step(
                students=self.control_group,
                selectors=self.control_selectors,
                is_ucb=False,
                timestep=timestep,
            )
            
            # Log metrics at specified frequency
            if timestep % self.config.log_frequency == 0:
                self._log_metrics(timestep)
        
        # Collect final student statistics
        ucb_final = [s.get_statistics() for s in self.ucb_group]
        control_final = [s.get_statistics() for s in self.control_group]
        
        return ExperimentResults(
            config=self.config,
            ucb_metrics=self.ucb_metrics,
            control_metrics=self.control_metrics,
            ucb_final_students=ucb_final,
            control_final_students=control_final,
        )
    
    def _process_group_step(
        self,
        students: List[Student],
        selectors: List,
        is_ucb: bool,
        timestep: int,
    ) -> None:
        """
        Process one timestep for a group of students.
        
        Args:
            students: List of students in the group
            selectors: List of selectors (one per student)
            is_ucb: Whether this is the UCB group
            timestep: Current timestep
        """
        for student, selector in zip(students, selectors):
            # Select category
            category = selector.select_category()
            
            # Student answers question
            correct = student.answer(category)
            
            # Update student knowledge
            student.update_knowledge(category, correct, timestep)
            
            # Update selector statistics
            selector.update(category, correct)
            
            # Apply forgetting
            student.apply_forgetting()
    
    def _log_metrics(self, timestep: int) -> None:
        """
        Log metrics for both groups at current timestep.
        
        Args:
            timestep: Current timestep
        """
        ucb_metrics = self._compute_group_metrics(
            self.ucb_group,
            self.ucb_selectors,
            timestep,
        )
        control_metrics = self._compute_group_metrics(
            self.control_group,
            self.control_selectors,
            timestep,
        )
        
        self.ucb_metrics.append(ucb_metrics)
        self.control_metrics.append(control_metrics)
    
    def _compute_group_metrics(
        self,
        students: List[Student],
        selectors: List,
        timestep: int,
    ) -> GroupMetrics:
        """
        Compute aggregate metrics for a group of students.
        
        Args:
            students: List of students
            selectors: List of selectors
            timestep: Current timestep
            
        Returns:
            GroupMetrics object
        """
        # Collect knowledge vectors
        knowledge_matrix = np.array([s.knowledge for s in students])
        
        # Average knowledge across students
        avg_knowledge = float(np.mean(knowledge_matrix))
        std_knowledge = float(np.std(knowledge_matrix))
        
        # Per-category statistics
        per_category_avg = np.mean(knowledge_matrix, axis=0).tolist()
        per_category_std = np.std(knowledge_matrix, axis=0).tolist()
        
        # Weakest and strongest category averages
        weakest_indices = [s.get_weakest_category() for s in students]
        strongest_indices = [s.get_strongest_category() for s in students]
        
        weakest_knowledge = [
            students[i].knowledge[weakest_indices[i]]
            for i in range(len(students))
        ]
        strongest_knowledge = [
            students[i].knowledge[strongest_indices[i]]
            for i in range(len(students))
        ]
        
        # Knowledge variance per student, averaged
        variances = [s.get_knowledge_variance() for s in students]
        
        # Cumulative correct/incorrect
        total_correct = sum(int(np.sum(s.correct_counts)) for s in students)
        total_incorrect = sum(int(np.sum(s.incorrect_counts)) for s in students)
        
        # Exposure distribution (aggregate across all students)
        exposure_dist = np.zeros(self.config.num_categories, dtype=np.int32)
        for selector in selectors:
            exposure_dist += selector.get_exposure_distribution()
        
        return GroupMetrics(
            timestep=timestep,
            average_knowledge=avg_knowledge,
            knowledge_std=std_knowledge,
            per_category_knowledge=per_category_avg,
            per_category_knowledge_std=per_category_std,
            weakest_category_avg=float(np.mean(weakest_knowledge)),
            strongest_category_avg=float(np.mean(strongest_knowledge)),
            knowledge_variance_avg=float(np.mean(variances)),
            cumulative_correct=total_correct,
            cumulative_incorrect=total_incorrect,
            exposure_distribution=exposure_dist.tolist(),
        )
    
    def reset(self) -> None:
        """Reset the experiment to initial state."""
        self.ucb_group = []
        self.control_group = []
        self.ucb_selectors = []
        self.control_selectors = []
        self.ucb_metrics = []
        self.control_metrics = []
        self._init_groups()


def run_multiple_experiments(
    config: ExperimentConfig,
    num_runs: int = 10,
    show_progress: bool = True,
) -> List[ExperimentResults]:
    """
    Run multiple experiments with different random seeds.
    
    This is useful for computing confidence intervals and
    statistical significance.
    
    Args:
        config: Base experiment configuration
        num_runs: Number of experiment runs
        show_progress: Whether to show progress
        
    Returns:
        List of ExperimentResults from each run
    """
    results = []
    
    runs = range(num_runs)
    if show_progress:
        runs = tqdm(runs, desc="Running experiments")
    
    for run_idx in runs:
        # Create config with different seed
        run_config = ExperimentConfig(
            num_students_per_group=config.num_students_per_group,
            num_categories=config.num_categories,
            category_difficulties=config.category_difficulties,
            initial_knowledge_mean=config.initial_knowledge_mean,
            initial_knowledge_std=config.initial_knowledge_std,
            learning_rate=config.learning_rate,
            incorrect_penalty=config.incorrect_penalty,
            base_knowledge=config.base_knowledge,
            decay_rate=config.decay_rate,
            questions_per_session=config.questions_per_session,
            exploration_param=config.exploration_param,
            random_seed=(config.random_seed or 0) + run_idx * 10000,
            log_frequency=config.log_frequency,
        )
        
        experiment = Experiment(run_config)
        result = experiment.run(show_progress=False)
        results.append(result)
    
    return results
