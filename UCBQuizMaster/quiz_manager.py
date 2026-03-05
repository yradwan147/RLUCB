"""Quiz session manager that coordinates question presentation and learning."""
from typing import Dict, List, Optional
from question_generator import QuestionGenerator
from ucb_algorithm import UCBSelector
import config


class QuizManager:
    """
    Manages a quiz session including question bank, student progress,
    and adaptive question selection.
    """
    
    def __init__(self, topic: str):
        """
        Initialize a new quiz session.
        
        Args:
            topic: The main topic for the quiz
        """
        self.topic = topic
        self.generator = QuestionGenerator()
        self.ucb_selector = None
        self.subtopics = []
        
        # Question bank: {subtopic: [list of question dicts]}
        self.question_bank = {}
        
        # Track which questions have been asked: {subtopic: [list of indices]}
        self.asked_questions = {}
        
        self.current_question = None
        self.current_subtopic = None
        self.questions_answered = 0
    
    def initialize(self, num_subtopics: int = 5):
        """
        Initialize the quiz by generating subtopics and initial questions.
        
        Args:
            num_subtopics: Number of subtopics to generate
        """
        print(f"\n{'='*60}")
        print(f"🎓 Initializing Quiz on: {self.topic}")
        print(f"{'='*60}\n")
        
        # Generate subtopics
        self.subtopics = self.generator.generate_subtopics(self.topic, num_subtopics)
        
        print("📚 Subtopics:")
        for i, subtopic in enumerate(self.subtopics, 1):
            print(f"  {i}. {subtopic}")
        print()
        
        # Initialize UCB selector
        self.ucb_selector = UCBSelector(self.subtopics)
        
        # Generate initial questions for each subtopic
        print(f"Generating {config.INITIAL_QUESTIONS_PER_SUBTOPIC} questions per subtopic...\n")
        for subtopic in self.subtopics:
            questions = self.generator.generate_questions(
                self.topic, 
                subtopic, 
                config.INITIAL_QUESTIONS_PER_SUBTOPIC
            )
            self.question_bank[subtopic] = questions
            self.asked_questions[subtopic] = []
        
        print(f"✅ Quiz initialized with {len(self.subtopics)} subtopics and "
              f"{sum(len(q) for q in self.question_bank.values())} total questions!\n")
    
    def get_next_question(self) -> Optional[Dict]:
        """
        Get the next question using UCB algorithm.
        
        Returns:
            Question dictionary or None if no questions available
        """
        # Use UCB to select the next subtopic
        self.current_subtopic = self.ucb_selector.select_next_subtopic()
        
        # Get available questions for this subtopic
        available_questions = self._get_available_questions(self.current_subtopic)
        
        # If no questions available, generate more
        if not available_questions:
            print(f"\n🔄 Generating more questions for '{self.current_subtopic}'...\n")
            new_questions = self.generator.generate_questions(
                self.topic,
                self.current_subtopic,
                config.ADDITIONAL_QUESTIONS_BATCH
            )
            self.question_bank[self.current_subtopic].extend(new_questions)
            available_questions = self._get_available_questions(self.current_subtopic)
        
        if not available_questions:
            return None
        
        # Select the next question from available ones
        question_index = available_questions[0]
        self.asked_questions[self.current_subtopic].append(question_index)
        self.current_question = self.question_bank[self.current_subtopic][question_index]
        
        return self.current_question
    
    def _get_available_questions(self, subtopic: str) -> List[int]:
        """
        Get indices of questions that haven't been asked yet for a subtopic.
        
        Args:
            subtopic: The subtopic to check
            
        Returns:
            List of available question indices
        """
        total_questions = len(self.question_bank[subtopic])
        asked = set(self.asked_questions[subtopic])
        available = [i for i in range(total_questions) if i not in asked]
        return available
    
    def submit_answer(self, user_answer: str) -> Dict:
        """
        Process the user's answer and update statistics.
        
        Args:
            user_answer: The user's answer choice (A, B, C, or D)
            
        Returns:
            Dictionary with is_correct, correct_answer, and explanation
        """
        if not self.current_question:
            return {"error": "No active question"}
        
        correct_answer = self.current_question['correct_answer']
        is_correct = user_answer.upper() == correct_answer.upper()
        
        # Update UCB statistics
        self.ucb_selector.update(self.current_subtopic, is_correct)
        self.questions_answered += 1
        
        result = {
            'is_correct': is_correct,
            'correct_answer': correct_answer,
            'explanation': self.current_question['explanation'],
            'subtopic': self.current_subtopic
        }
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get current learning statistics."""
        return self.ucb_selector.get_statistics()
    
    def get_progress_summary(self) -> str:
        """Get formatted progress summary."""
        return self.ucb_selector.get_progress_summary()
    
    def get_weakest_subtopic(self) -> str:
        """Get the subtopic where the student struggles most."""
        return self.ucb_selector.get_weakest_subtopic()
    
    def get_questions_answered(self) -> int:
        """Get total number of questions answered."""
        return self.questions_answered

