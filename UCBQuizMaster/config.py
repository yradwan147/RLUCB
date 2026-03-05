"""Configuration settings for the quiz application."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective model, can be changed to gpt-4o for better quality
INITIAL_QUESTIONS_PER_SUBTOPIC = 10
ADDITIONAL_QUESTIONS_BATCH = 10

