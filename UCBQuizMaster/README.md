# 🎓 UCB Quiz Master

An intelligent, adaptive learning quiz application that uses the **Upper Confidence Bound (UCB)** algorithm to identify your weak areas and help you improve efficiently.

## 🌟 Features

- **Beautiful GUI Interface**: Modern, intuitive graphical interface (NEW!)
- **Real-Time UCB Visualization**: Watch the algorithm work with interactive plots (NEW!)
- **Debug Mode**: See how the algorithm decides which questions to ask next (NEW!)
- **AI-Powered Content Generation**: Automatically generates subtopics and questions using OpenAI's GPT models
- **Adaptive Learning**: Uses the UCB algorithm to focus on areas where you struggle most
- **Automatic Grading**: Multiple-choice questions with instant feedback
- **Detailed Explanations**: Get explanations for incorrect answers to help you learn
- **Dynamic Question Generation**: Generates more questions automatically when needed
- **Progress Tracking**: Detailed statistics showing your performance across all subtopics
- **CLI Version Available**: Classic terminal interface for those who prefer it

## 🧠 How It Works

### The Upper Confidence Bound Algorithm

The UCB algorithm balances two key strategies:

1. **Exploitation**: Focus on subtopics where you have low correctness rates (your weak areas)
2. **Exploration**: Ensure all subtopics get tested to accurately assess your knowledge

The formula used is:
```
UCB Score = (1 - correctness_rate) + c * sqrt(ln(total_questions) / attempts)
```

Where:
- `correctness_rate`: Your success rate on a subtopic (0-1)
- `c`: Exploration parameter (default: 1.414, or √2)
- `total_questions`: Total questions answered across all subtopics
- `attempts`: Questions answered for this specific subtopic

**Result**: The algorithm intelligently selects questions from subtopics where you need the most improvement while still testing all areas to maintain accuracy.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   
   Create a `.env` file in the project directory:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```
   
   Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Running the Application

**GUI Version (Recommended)** - Beautiful interface with UCB visualization:
```bash
python gui_app.py
```

**CLI Version** - Simple terminal interface:
```bash
python main.py
```

Or make them executable:
```bash
chmod +x gui_app.py main.py
./gui_app.py  # or ./main.py
```

## 📖 Usage Guide

### GUI Version (Recommended)

1. Run `python gui_app.py`
2. Enter your topic and number of subtopics
3. **Enable Debug Mode** to see the UCB algorithm visualization
4. Click "Start Quiz"
5. Answer questions and watch the algorithm adapt!

**See [GUI_GUIDE.md](GUI_GUIDE.md) for detailed GUI documentation and visualization explanation.**

### CLI Version

1. Run `python main.py`
2. Enter the topic you want to study (e.g., "Machine Learning", "Calculus", "Python Programming")
3. Optionally specify the number of subtopics (default: 5)
4. Wait for the AI to generate subtopics and questions
5. Start answering questions!

### During the Quiz

- **Answer questions**: Type A, B, C, or D and press Enter
- **View statistics**: Type `s` to see your current progress
- **Quit**: Type `q` to end the quiz and see final results

### Understanding Your Results

The progress summary shows:
- 🔴 Red: Correctness rate < 50% (needs significant improvement)
- 🟡 Yellow: Correctness rate 50-75% (needs some work)
- 🟢 Green: Correctness rate > 75% (good understanding)

## 🏗️ Project Structure

```
UCBQuizMaster/
├── gui_app.py             # GUI application with UCB visualization (NEW!)
├── main.py                # CLI application
├── quiz_manager.py        # Quiz session management
├── question_generator.py  # OpenAI integration for content generation
├── ucb_algorithm.py       # UCB algorithm implementation
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── GUI_GUIDE.md           # GUI usage guide (NEW!)
├── QUICKSTART.md          # Quick start guide
└── .env                   # Your API key (create this)
```

## ⚙️ Configuration

Edit `config.py` to customize:

- `OPENAI_MODEL`: The OpenAI model to use (default: "gpt-4o-mini")
  - `gpt-4o-mini`: Fast and cost-effective
  - `gpt-4o`: Higher quality, more expensive
  - `gpt-4-turbo`: Good balance of quality and speed
- `INITIAL_QUESTIONS_PER_SUBTOPIC`: Questions generated initially (default: 10)
- `ADDITIONAL_QUESTIONS_BATCH`: Questions generated when more are needed (default: 10)

## 💡 Tips for Best Results

1. **Be Specific**: Use specific topics like "Linear Algebra - Matrix Operations" rather than just "Math"
2. **Answer Honestly**: The algorithm works best when you answer authentically
3. **Review Explanations**: Always read the explanations for incorrect answers
4. **Check Progress**: Use the `s` command periodically to see your improvement
5. **Multiple Sessions**: Run multiple sessions on the same topic to track improvement over time

## 🔧 Advanced Usage

### Customizing the UCB Algorithm

In `ucb_algorithm.py`, you can adjust the `exploration_param`:

```python
ucb_selector = UCBSelector(subtopics, exploration_param=2.0)
```

- **Lower values** (e.g., 1.0): More exploitation (focus heavily on weak areas)
- **Higher values** (e.g., 2.0): More exploration (test all areas more evenly)

### Using Different AI Models

For better quality questions (at higher cost), edit `config.py`:

```python
OPENAI_MODEL = "gpt-4o"  # or "gpt-4-turbo"
```

## 📊 Example Session

```
🎓 UCB QUIZ MASTER

Enter the topic you want to quiz on: Python Programming

Generating subtopics for 'Python Programming'...
✅ Generated 5 subtopics

📚 Subtopics:
  1. Python Basics and Syntax
  2. Data Structures
  3. Object-Oriented Programming
  4. File I/O and Exception Handling
  5. Libraries and Modules

Generating 25 questions per subtopic...
✅ Quiz initialized with 5 subtopics and 125 total questions!

Question 1
Subtopic: Python Basics and Syntax
==================================================

What is the output of print(type([]))?

  A. <class 'tuple'>
  B. <class 'list'>
  C. <class 'dict'>
  D. <class 'set'>

Your answer (A/B/C/D) or 'q' to quit, 's' for stats: B

✅ Correct! Well done!
```

## 🐛 Troubleshooting

### "OpenAI API key not configured"
- Ensure your `.env` file exists and contains `OPENAI_API_KEY=your_key`
- Check that the `.env` file is in the same directory as `main.py`

### API Rate Limits
- If you hit rate limits, wait a few minutes or upgrade your OpenAI plan
- Consider using `gpt-4o-mini` which has higher rate limits

### Poor Question Quality
- Try using `gpt-4o` or `gpt-4-turbo` instead of `gpt-4o-mini`
- Be more specific with your topic

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and customize it for your needs!

## 📧 Support

For issues or questions, please create an issue in the repository.

---

**Happy Learning! 🚀📚**

