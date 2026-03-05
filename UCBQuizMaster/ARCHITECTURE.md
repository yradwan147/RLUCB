# 🏗️ System Architecture

## Overview

UCB Quiz Master is a modular adaptive learning system that combines AI-powered content generation with intelligent question selection using the Upper Confidence Bound algorithm.

## Components

### 1. Main Application (`main.py`)
**Purpose**: User interface and application flow control

**Responsibilities**:
- Display welcome screen and collect user input
- Initialize quiz session
- Present questions to the user
- Collect and validate answers
- Display results and explanations
- Show progress summaries
- Handle user commands (quit, stats)

**Flow**:
```
Start → Get Topic → Initialize Quiz → Loop:
  ├─ Get Next Question (from QuizManager)
  ├─ Display Question
  ├─ Get User Answer
  ├─ Submit Answer
  ├─ Display Result & Explanation
  └─ Continue or Exit
→ Show Final Summary → End
```

### 2. Quiz Manager (`quiz_manager.py`)
**Purpose**: Central coordinator for quiz sessions

**Responsibilities**:
- Initialize quiz with topic
- Coordinate with QuestionGenerator for content
- Maintain question bank (all questions for all subtopics)
- Track which questions have been asked
- Select next question using UCB algorithm
- Generate additional questions when needed
- Track overall progress
- Provide statistics

**Key Methods**:
- `initialize()`: Set up subtopics and generate initial questions
- `get_next_question()`: Use UCB to select next question
- `submit_answer()`: Process answer and update statistics
- `get_progress_summary()`: Format and return progress data

**Data Structures**:
```python
question_bank = {
    'Subtopic 1': [question1, question2, ...],
    'Subtopic 2': [question1, question2, ...],
    ...
}

asked_questions = {
    'Subtopic 1': [0, 3, 7, ...],  # indices of asked questions
    'Subtopic 2': [1, 2, 4, ...],
    ...
}
```

### 3. Question Generator (`question_generator.py`)
**Purpose**: OpenAI API integration for content generation

**Responsibilities**:
- Generate subtopics for a given topic
- Generate questions with multiple-choice options
- Generate correct answers and explanations
- Parse and validate OpenAI responses

**Key Methods**:
- `generate_subtopics(topic, num_subtopics)`: Create subtopic list
- `generate_questions(topic, subtopic, num_questions)`: Create question set

**Question Format**:
```python
{
    "question": "What is ...?",
    "options": {
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
    },
    "correct_answer": "B",
    "explanation": "The answer is B because..."
}
```

### 4. UCB Algorithm (`ucb_algorithm.py`)
**Purpose**: Implement adaptive question selection

**Responsibilities**:
- Track performance statistics per subtopic
- Calculate UCB scores for each subtopic
- Select next subtopic to quiz on
- Identify weakest subtopic
- Generate progress summaries

**Statistics Tracked**:
```python
stats = {
    'Subtopic': {
        'attempts': 10,
        'correct': 6,
        'incorrect': 4,
        'correctness_rate': 0.6
    }
}
```

**UCB Formula**:
```
UCB Score = (1 - correctness_rate) + c × √(ln(total) / attempts)
           └─── exploitation ───┘   └───── exploration ──────┘
```

**Selection Logic**:
1. If any subtopic hasn't been tried → select it (exploration)
2. Otherwise → select subtopic with highest UCB score
3. Higher UCB score = more urgent to quiz on

### 5. Configuration (`config.py`)
**Purpose**: Centralized configuration management

**Settings**:
- OpenAI API key (from .env)
- Model selection (gpt-4o-mini, gpt-4o, etc.)
- Initial questions per subtopic (25)
- Additional questions batch size (20)

## Data Flow

```
User Input (Topic)
    ↓
QuestionGenerator.generate_subtopics()
    ↓
[Subtopics List]
    ↓
For each subtopic:
    QuestionGenerator.generate_questions()
    ↓
[Question Bank Created]
    ↓
UCBSelector initialized with subtopics
    ↓
Quiz Loop:
    UCBSelector.select_next_subtopic() → Subtopic
        ↓
    QuizManager.get_next_question() → Question
        ↓
    User answers question
        ↓
    QuizManager.submit_answer()
        ↓
    UCBSelector.update(subtopic, is_correct)
        ↓
    Display result & explanation
        ↓
    If questions running low:
        QuestionGenerator.generate_questions()
    ↓
    Repeat
```

## Algorithm Details

### Upper Confidence Bound (UCB)

**Goal**: Identify and focus on the student's weak areas while ensuring all topics are adequately tested.

**Exploitation Component**: `(1 - correctness_rate)`
- Higher value when student performs poorly
- Focuses on weak areas

**Exploration Component**: `c × √(ln(total) / attempts)`
- Higher value when subtopic has fewer attempts
- Ensures all topics get fair representation
- `c` (exploration parameter) controls balance

**Selection Strategy**:
- Early phase: Explores all subtopics to build accurate statistics
- Mid phase: Balances between weak areas and exploration
- Late phase: Heavily focuses on identified weak areas

**Adaptive Behavior**:
- If student improves in a weak area → algorithm shifts focus to next weakest
- If student struggles everywhere → rotates between all weak areas
- If student excels everywhere → explores to find any remaining gaps

## Question Generation Strategy

### Subtopic Generation Prompt
- Asks for comprehensive coverage of topic
- Requests non-overlapping subtopics
- Orders from fundamental to advanced

### Question Generation Prompt
- Mixed difficulty levels (easy, medium, hard)
- Clear and unambiguous questions
- Exactly 4 options
- Only one correct answer
- Detailed explanations

### Quality Assurance
- JSON validation of responses
- Markdown code block removal
- Error handling for API failures

## Extensibility

### Adding New Features

**Progress Persistence**:
```python
# Add to QuizManager
def save_progress(self, filename):
    with open(filename, 'w') as f:
        json.dump(self.ucb_selector.stats, f)

def load_progress(self, filename):
    with open(filename, 'r') as f:
        stats = json.load(f)
        self.ucb_selector.stats = stats
```

**Different Question Types**:
```python
# Extend QuestionGenerator
def generate_true_false_questions(self, topic, subtopic, num):
    # Implementation
    pass

def generate_fill_in_blank_questions(self, topic, subtopic, num):
    # Implementation
    pass
```

**Custom UCB Variants**:
```python
# Extend UCBSelector
class AdaptiveUCBSelector(UCBSelector):
    def select_next_subtopic(self):
        # Custom selection logic
        # E.g., adjust exploration_param based on progress
        pass
```

**Web Interface**:
- Replace `main.py` with Flask/FastAPI backend
- Add frontend (React/Vue)
- Keep QuizManager, QuestionGenerator, UCBSelector as-is

## Performance Considerations

**API Calls**:
- Initial setup: ~6 API calls (1 for subtopics + 5 for questions)
- During quiz: 1 API call per 25 questions (only when generating more)
- Cost: ~$0.10-0.20 per full quiz session with gpt-4o-mini

**Memory**:
- Question bank grows over time
- For 5 subtopics × 50 questions = ~250KB memory
- Negligible for modern systems

**Response Time**:
- Question selection: < 1ms (pure computation)
- Question generation: 2-10 seconds (API call)
- Overall: Highly responsive during quiz

## Security & Privacy

**API Key**:
- Stored in `.env` file (gitignored)
- Never hardcoded
- Loaded via environment variables

**User Data**:
- All data kept in memory (not persisted)
- No user information sent to OpenAI
- Only topic and questions sent to API

**Dependencies**:
- Minimal external dependencies
- All from trusted sources (OpenAI, numpy, python-dotenv)

## Testing Strategy

**Unit Tests** (can be added):
```python
# test_ucb_algorithm.py
def test_ucb_selection():
    selector = UCBSelector(['Topic1', 'Topic2'])
    # Test selection logic
    
# test_quiz_manager.py
def test_question_tracking():
    # Test question tracking logic
```

**Integration Tests**:
```python
# test_integration.py
def test_full_quiz_flow():
    # Test end-to-end flow (with mocked API)
```

**Manual Testing**:
- Test with various topics
- Test question regeneration
- Test statistics accuracy
- Test edge cases (all correct, all incorrect)

## Future Enhancements

1. **Spaced Repetition**: Combine UCB with spaced repetition algorithm
2. **Difficulty Adaptation**: Track question difficulty and adapt
3. **Multi-user Support**: Track multiple students' progress
4. **Analytics Dashboard**: Visualize learning progress over time
5. **Custom Question Import**: Allow importing existing question banks
6. **Collaborative Features**: Compare progress with peers
7. **Gamification**: Add points, badges, leaderboards
8. **Mobile App**: iOS/Android versions

---

This architecture provides a solid foundation for adaptive learning while remaining simple, maintainable, and extensible.

