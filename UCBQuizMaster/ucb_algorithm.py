"""Implementation of the Upper Confidence Bound (UCB) algorithm for adaptive learning."""
import math
from typing import Dict, List
import numpy as np


class UCBSelector:
    """
    Implements the Upper Confidence Bound algorithm to select which subtopic
    to quiz the student on next, prioritizing subtopics where the student struggles most.
    """
    
    def __init__(self, subtopics: List[str], exploration_param: float = 1.414):
        """
        Initialize the UCB selector.
        
        Args:
            subtopics: List of all subtopic names
            exploration_param: Exploration parameter (typically sqrt(2) ≈ 1.414)
                              Higher values increase exploration of less-tested subtopics
        """
        self.subtopics = subtopics
        self.exploration_param = exploration_param
        
        # Track statistics for each subtopic
        self.stats = {
            subtopic: {
                'attempts': 0,
                'correct': 0,
                'incorrect': 0,
                'correctness_rate': 0.0
            }
            for subtopic in subtopics
        }
        
        self.total_questions = 0
    
    def update(self, subtopic: str, is_correct: bool):
        """
        Update statistics after a question is answered.
        
        Args:
            subtopic: The subtopic of the answered question
            is_correct: Whether the answer was correct
        """
        self.stats[subtopic]['attempts'] += 1
        if is_correct:
            self.stats[subtopic]['correct'] += 1
        else:
            self.stats[subtopic]['incorrect'] += 1
        
        # Update correctness rate
        attempts = self.stats[subtopic]['attempts']
        correct = self.stats[subtopic]['correct']
        self.stats[subtopic]['correctness_rate'] = correct / attempts if attempts > 0 else 0.0
        
        self.total_questions += 1
    
    def select_next_subtopic(self) -> str:
        """
        Select the next subtopic to quiz on using UCB algorithm.
        
        The algorithm balances:
        1. Exploitation: Focus on subtopics with lower correctness rates (where student struggles)
        2. Exploration: Give chances to less-tested subtopics
        
        Returns:
            The name of the selected subtopic
        """
        # If any subtopic hasn't been tried yet, prioritize it
        for subtopic in self.subtopics:
            if self.stats[subtopic]['attempts'] == 0:
                return subtopic
        
        best_subtopic = None
        best_score = -float('inf')
        
        for subtopic in self.subtopics:
            # Calculate UCB score
            # We use (1 - correctness_rate) because we want to focus on weak areas
            # Lower correctness = higher priority
            weakness_score = 1 - self.stats[subtopic]['correctness_rate']
            
            # Exploration bonus: explore less-tested subtopics
            attempts = self.stats[subtopic]['attempts']
            exploration_bonus = self.exploration_param * math.sqrt(
                math.log(self.total_questions + 1) / attempts
            )
            
            # UCB score = weakness score + exploration bonus
            ucb_score = weakness_score + exploration_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_subtopic = subtopic
        
        return best_subtopic
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics for all subtopics.
        
        Returns:
            Dictionary with statistics for each subtopic
        """
        return self.stats.copy()
    
    def get_weakest_subtopic(self) -> str:
        """
        Get the subtopic with the lowest correctness rate (where student struggles most).
        
        Returns:
            Name of the weakest subtopic
        """
        # Only consider subtopics that have been attempted
        attempted_subtopics = {
            k: v for k, v in self.stats.items() if v['attempts'] > 0
        }
        
        if not attempted_subtopics:
            return self.subtopics[0]
        
        weakest = min(
            attempted_subtopics.items(),
            key=lambda x: x[1]['correctness_rate']
        )
        return weakest[0]
    
    def get_progress_summary(self) -> str:
        """
        Get a formatted summary of the student's progress.
        
        Returns:
            Formatted string with progress information
        """
        summary = "\n" + "="*60 + "\n"
        summary += "📊 LEARNING PROGRESS SUMMARY\n"
        summary += "="*60 + "\n\n"
        
        # Sort subtopics by correctness rate (weakest first)
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: (x[1]['attempts'] == 0, x[1]['correctness_rate'])
        )
        
        for subtopic, stats in sorted_stats:
            if stats['attempts'] == 0:
                summary += f"📚 {subtopic}\n"
                summary += f"   Not attempted yet\n\n"
            else:
                rate = stats['correctness_rate'] * 100
                emoji = "🔴" if rate < 50 else "🟡" if rate < 75 else "🟢"
                summary += f"{emoji} {subtopic}\n"
                summary += f"   Attempts: {stats['attempts']} | "
                summary += f"Correct: {stats['correct']} | "
                summary += f"Incorrect: {stats['incorrect']} | "
                summary += f"Rate: {rate:.1f}%\n\n"
        
        summary += "="*60 + "\n"
        return summary
    
    def get_ucb_data_for_visualization(self) -> Dict:
        """
        Get UCB data formatted for visualization.
        
        Returns:
            Dictionary with visualization data for each subtopic:
            {
                'subtopic': {
                    'correctness_rate': float,
                    'weakness_score': float (1 - correctness_rate),
                    'exploration_bonus': float,
                    'ucb_score': float,
                    'attempts': int
                }
            }
        """
        viz_data = {}
        
        for subtopic in self.subtopics:
            stats = self.stats[subtopic]
            attempts = stats['attempts']
            
            if attempts == 0:
                # Not yet attempted
                viz_data[subtopic] = {
                    'correctness_rate': 0.5,  # Neutral value for display
                    'weakness_score': 0.5,
                    'exploration_bonus': float('inf'),  # Will be capped in visualization
                    'ucb_score': float('inf'),
                    'attempts': 0
                }
            else:
                correctness_rate = stats['correctness_rate']
                weakness_score = 1 - correctness_rate
                exploration_bonus = self.exploration_param * math.sqrt(
                    math.log(self.total_questions + 1) / attempts
                )
                ucb_score = weakness_score + exploration_bonus
                
                viz_data[subtopic] = {
                    'correctness_rate': correctness_rate,
                    'weakness_score': weakness_score,
                    'exploration_bonus': exploration_bonus,
                    'ucb_score': ucb_score,
                    'attempts': attempts
                }
        
        return viz_data

