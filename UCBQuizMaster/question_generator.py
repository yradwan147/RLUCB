"""Module for generating subtopics and questions using OpenAI API."""
import json
from typing import List, Dict
from openai import OpenAI
import config


class QuestionGenerator:
    """Handles generation of subtopics and questions using OpenAI."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
    
    def generate_subtopics(self, topic: str, num_subtopics: int = 5) -> List[str]:
        """
        Generate a list of subtopics for a given topic.
        
        Args:
            topic: The main topic to generate subtopics for
            num_subtopics: Number of subtopics to generate
            
        Returns:
            List of subtopic names
        """
        print(f"🤖 Generating subtopics for '{topic}'...")
        
        prompt = f"""You are an educational expert. Generate {num_subtopics} distinct and important subtopics for the topic: "{topic}".

The subtopics should:
- Cover the most important areas of this topic
- Be specific enough to create meaningful quiz questions
- Be ordered from fundamental to advanced concepts
- Be comprehensive and non-overlapping

Return ONLY a JSON array of subtopic names, nothing else. Format: ["subtopic1", "subtopic2", ...]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        subtopics = json.loads(content)
        print(f"✅ Generated {len(subtopics)} subtopics\n")
        return subtopics
    
    def generate_questions(self, topic: str, subtopic: str, num_questions: int = 25) -> List[Dict]:
        """
        Generate multiple-choice questions for a given subtopic.
        
        Args:
            topic: The main topic
            subtopic: The specific subtopic to generate questions for
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries with question, options, correct_answer, and explanation
        """
        print(f"📝 Generating {num_questions} questions for '{subtopic}'...")
        
        prompt = f"""You are an educational expert creating quiz questions about: {topic} - {subtopic}

Generate {num_questions} multiple-choice questions that:
- Test understanding at various difficulty levels (easy, medium, hard mix)
- Are clear and unambiguous
- Have 4 answer options (A, B, C, D)
- Have only ONE correct answer
- Include a detailed explanation for the correct answer

Return ONLY a JSON array with this exact structure:
[
  {{
    "question": "The question text",
    "options": {{
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    }},
    "correct_answer": "A",
    "explanation": "Detailed explanation of why this answer is correct"
  }}
]

Make sure all questions are unique and cover different aspects of {subtopic}."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        questions = json.loads(content)
        print(f"✅ Generated {len(questions)} questions\n")
        return questions

