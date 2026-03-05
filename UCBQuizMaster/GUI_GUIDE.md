# 🎨 GUI Application Guide

## Overview

The GUI version of UCB Quiz Master provides a beautiful, interactive interface with **real-time visualization** of the Upper Confidence Bound algorithm in action!

## Features

### 🖥️ Modern Interface
- Clean, intuitive design
- Easy-to-use question interface
- Real-time feedback and explanations
- Progress tracking

### 🔬 Debug Mode (UCB Visualization)
The **Debug Mode** shows a live visualization of how the UCB algorithm works:

#### What You'll See:
- **X-axis**: All subtopics
- **Y-axis**: Score (0-2 range)
- **Colored circles (●)**: Empirical weakness score (1 - correctness rate)
  - 🔴 Red: < 50% correct (struggling)
  - 🟡 Orange: 50-75% correct (moderate)
  - 🟢 Green: > 75% correct (doing well)
  - ⚫ Gray: Not yet attempted
- **Vertical lines (|)**: Exploration bonus
- **Upper whisker (_)**: Total UCB score (used to select next question)
- **Yellow highlight**: Next subtopic to be selected

#### How to Read the Plot:
1. **Longer vertical lines** = More exploration bonus = Less tested
2. **Higher circles** = Lower correctness = Struggling more
3. **The algorithm picks** the subtopic with the highest upper whisker
4. **Watch it adapt** as you answer questions!

## Running the GUI

### Quick Start

```bash
python gui_app.py
```

Or if made executable:
```bash
./gui_app.py
```

### Steps:

1. **Enter a Topic**
   - Type any topic you want to learn (e.g., "Machine Learning", "Calculus")

2. **Set Number of Subtopics**
   - Choose 2-10 subtopics (default: 5)

3. **Enable Debug Mode** (optional)
   - Check the "Debug Mode" checkbox in the top-right
   - The UCB visualization will appear on the right side

4. **Click "Start Quiz"**
   - Wait while the AI generates subtopics and questions
   - Usually takes 30-60 seconds

5. **Answer Questions**
   - Read the question
   - Select A, B, C, or D
   - Click "Submit Answer"
   - See instant feedback
   - Click "Next Question" to continue

6. **Watch the Algorithm Work**
   - If debug mode is on, watch the plot update after each answer
   - See how the algorithm focuses on your weak areas
   - Notice the exploration bonus decrease as topics get more attempts

## Understanding the UCB Algorithm

### The Formula Visualized:

```
UCB Score = Weakness Score + Exploration Bonus
            └─ circle (●) ─┘   └─ line length ─┘
```

### Example Interpretation:

```
Subtopic A: ●────|  (High UCB score = will be selected soon)
            ↑    ↑
            |    └─ High exploration (not tested much)
            └────── High weakness (struggling)

Subtopic B: ●|     (Low UCB score = doing well, tested enough)
            ↑
            └────── Low weakness (high correctness)
```

### Algorithm Behavior:

**Early Phase (First few questions):**
- All subtopics have high exploration bonus
- Algorithm explores all areas
- Vertical lines are very long

**Mid Phase:**
- Exploration bonus decreases
- Weakness scores become apparent
- Algorithm balances exploration and exploitation

**Late Phase:**
- Focuses heavily on weak areas
- Continues to check strong areas occasionally
- Helps you improve where you need it most!

## UI Elements

### Top Bar
- **Title**: Current application status
- **Debug Mode Checkbox**: Toggle visualization on/off

### Left Panel (Quiz Interface)
- **Question Number**: Current question count
- **Subtopic Badge**: Current subtopic being tested
- **Question Text**: The question to answer
- **Options**: Radio buttons for A, B, C, D
- **Submit Button**: Check your answer
- **Next Button**: Move to next question
- **Statistics Button**: View detailed progress

### Right Panel (Debug Mode)
- **UCB Visualization**: Real-time algorithm visualization
- **Updates automatically** after each question
- **Shows next selection** with yellow highlight

## Keyboard Shortcuts

- **Enter**: Submit answer (when an option is selected)
- **Tab**: Navigate between options

## Tips for Using Debug Mode

1. **Watch the Pattern**: Notice how the algorithm shifts focus based on your performance

2. **Identify Weak Areas**: Red circles show where you're struggling

3. **See Exploration**: Long vertical lines mean "we need more data on this topic"

4. **Next Question Prediction**: The yellow highlight shows what's coming next

5. **Track Improvement**: Watch circles move down (get greener) as you improve!

## Statistics View

Click "📊 View Statistics" to see:
- Total attempts per subtopic
- Correct/incorrect counts
- Correctness percentages
- Color-coded performance indicators

## Troubleshooting

### Window is too small
- Resize the window - it's fully responsive
- Debug panel automatically adjusts

### Visualization not updating
- Make sure Debug Mode checkbox is checked
- Try toggling Debug Mode off and on

### API Key Error
- Create a `.env` file with `OPENAI_API_KEY=your_key`
- Restart the application

### Slow Loading
- Initial quiz setup takes 30-60 seconds
- This is normal - AI is generating content
- Subsequent questions are instant

## Advanced: Understanding the Visualization Math

### Weakness Score (Circle Position)
```python
weakness_score = 1 - (correct_answers / total_attempts)
```
- Perfect score (100%): weakness = 0 (circle at bottom)
- No knowledge (0%): weakness = 1 (circle at top)

### Exploration Bonus (Line Length)
```python
exploration_bonus = sqrt(2) × sqrt(ln(total_questions) / attempts)
```
- More total questions: Slight increase
- Fewer attempts on this subtopic: Large increase
- Encourages testing all areas

### UCB Score (Upper Whisker)
```python
ucb_score = weakness_score + exploration_bonus
```
- **Highest UCB score wins** = Next question comes from this subtopic
- Balances "help weak areas" + "test all areas fairly"

## Educational Value

This visualization helps you understand:
- **How the algorithm thinks**: See the decision-making process
- **Where you struggle**: Immediate visual feedback
- **Why you got a question**: Understand the selection logic
- **Reinforcement learning**: See UCB algorithm (from ML) in action!

## Performance Notes

- Initial generation: ~30-60 seconds (API calls)
- Question display: Instant
- Visualization update: < 100ms
- Memory usage: ~50MB typical

## Exiting

- Close the window anytime
- Progress is not saved (by design)
- Start fresh each session

---

**Enjoy watching the algorithm learn about you as you learn the material!** 🎓📊

