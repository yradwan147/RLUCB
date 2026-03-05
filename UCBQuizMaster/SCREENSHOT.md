# 📸 GUI Screenshots & Visual Guide

## Main Interface Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  🎓 UCB Quiz Master                        ☑ Debug Mode (Show UCB Viz)      │
├──────────────────────────────────┬───────────────────────────────────────────┤
│                                  │                                           │
│  Question 5                      │  UCB Algorithm Visualization              │
│  📚 Neural Networks              │                                           │
│  ──────────────────────────────  │   2.0┤                                    │
│                                  │      │     ●                               │
│  What is backpropagation?        │      │     │                               │
│                                  │   1.5┤     │   ●─┐                        │
│  ┌────────────────────────────┐  │      │  ┌──┘    │                        │
│  │ Backpropagation is...      │  │   1.0┤  │       │    ●─┐                 │
│  │ used for training neural   │  │      │  │       │    │ │    ●            │
│  │ networks...                │  │   0.5┤  │       │    │ │    │            │
│  └────────────────────────────┘  │      │  │       │    │ │    │            │
│                                  │   0.0└──┴───────┴────┴─┴────┴────        │
│  Select your answer:             │       Sub1  Sub2  Sub3 Sub4 Sub5          │
│  ┌────────────────────────────┐  │                                           │
│  │ ○ A. Forward pass only     │  │  Legend:                                  │
│  │ ● B. Gradient calculation  │  │  ● Green: >75% correct                    │
│  │ ○ C. Data preprocessing    │  │  ● Orange: 50-75% correct                 │
│  │ ○ D. Model deployment      │  │  ● Red: <50% correct                      │
│  └────────────────────────────┘  │  ● Gray: Not attempted                    │
│                                  │                                           │
│  [Submit Answer]  [Next →]      │  Yellow highlight = Next topic             │
│              [📊 Statistics]     │                                           │
└──────────────────────────────────┴───────────────────────────────────────────┘
```

## Debug Mode Visualization Explained

The right panel shows the UCB algorithm's decision-making process:

### Visual Elements

```
   Score
    2.0 ┤
        │   ● ← Red circle (struggling, <50%)
    1.5 ┤   │
        │   │ ← Vertical line (exploration bonus)
    1.0 ┤   │
        │   _ ← Upper whisker (total UCB score)
    0.5 ┤
        │
    0.0 └─────────────────────────────
        Subtopic Name
```

### Color Coding

- 🔴 **Red**: Performance < 50% (Struggling!)
- 🟡 **Orange**: Performance 50-75% (Needs work)
- 🟢 **Green**: Performance > 75% (Good!)
- ⚫ **Gray**: Not yet attempted

### What the Algorithm Shows

1. **Circle Position (●)**: Your empirical weakness score
   - Higher = Struggling more
   - Lower = Doing better

2. **Vertical Line (|)**: Exploration bonus
   - Longer = Less tested, needs more data
   - Shorter = Well tested

3. **Upper Whisker (_)**: Total UCB score
   - Highest wins = Next question from this topic

4. **Yellow Background**: Next selected subtopic

## Example Scenarios

### Scenario 1: Early in Quiz
```
All topics have long vertical lines (high exploration)
Algorithm testing all areas to gather data
```

### Scenario 2: Identified Weakness
```
One topic has:
- High circle (red, struggling)
- Short line (well tested)
- Still highest UCB score
→ Algorithm focuses here to help you improve
```

### Scenario 3: Uniform Performance
```
All topics have similar scores
Algorithm balances between all areas
Continues to explore
```

## Result Display

### Correct Answer:
```
┌─────────────────────────────────┐
│ ✅ Correct! Well done!          │
└─────────────────────────────────┘
```

### Incorrect Answer:
```
┌─────────────────────────────────────────────────────────┐
│ ❌ Incorrect. The correct answer is: B                  │
│                                                         │
│ 💡 Explanation:                                         │
│ ┌─────────────────────────────────────────────────────┐│
│ │ Backpropagation is the algorithm used to calculate ││
│ │ gradients of the loss function with respect to     ││
│ │ the neural network weights, enabling learning      ││
│ │ through gradient descent...                        ││
│ └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Statistics View

Click "📊 Statistics" to see:

```
╔════════════════════════════════════════════════════════════╗
║              📊 LEARNING PROGRESS SUMMARY                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  🔴 Neural Networks                                        ║
║     Attempts: 12 | Correct: 5 | Incorrect: 7 | Rate: 41.7%║
║                                                            ║
║  🟡 Data Structures                                        ║
║     Attempts: 8 | Correct: 5 | Incorrect: 3 | Rate: 62.5% ║
║                                                            ║
║  🟢 Python Basics                                          ║
║     Attempts: 10 | Correct: 9 | Incorrect: 1 | Rate: 90.0%║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

## Setup Screen

When you first launch:

```
┌────────────────────────────────────────────────────┐
│                                                    │
│        Welcome to UCB Quiz Master!                 │
│                                                    │
│  Adaptive Learning with Upper Confidence Bound     │
│             Algorithm                              │
│                                                    │
│  ─────────────────────────────────────────────    │
│                                                    │
│  Enter Topic:                                      │
│  ┌──────────────────────────────────────────────┐ │
│  │ Machine Learning                             │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Number of Subtopics:                              │
│  ┌────────┐                                        │
│  │   5    │                                        │
│  └────────┘                                        │
│                                                    │
│           [  🚀 Start Quiz  ]                      │
│                                                    │
│  This application will:                            │
│  • Generate subtopics for your topic               │
│  • Create 10 questions per subtopic                │
│  • Adapt to find your weak areas                   │
│  • Focus on helping you improve                    │
│                                                    │
│  Enable Debug Mode to see the UCB algorithm!       │
└────────────────────────────────────────────────────┘
```

## Tips for Using the GUI

### Best Practices:
1. **Start with Debug Mode ON** - Watch and learn how the algorithm works
2. **Observe the patterns** - See which topics get more questions
3. **Track the colors** - Watch circles change from red → orange → green
4. **Notice the balance** - Algorithm tests all areas while focusing on weaknesses
5. **Use statistics** - Regular check-ins show your progress

### Understanding the Flow:
```
Setup → Initialize (30-60s) → Question 1 
  ↓
Debug Plot Updates → Answer → Feedback → Next
  ↓
Algorithm Adapts → Selects Weak Area → Next Question
  ↓
Repeat → Final Statistics → Complete
```

## Window Management

- **Minimum Size**: 1200x800 pixels recommended
- **Resizable**: Yes, fully responsive
- **Debug Panel**: Toggle on/off anytime
- **Layout**: Paned window - adjust split position by dragging

## Keyboard Navigation

- **Tab**: Move between options
- **Arrow Keys**: Select options
- **Enter**: Submit answer (when answer selected)
- **Escape**: Close statistics window

---

**The visualization makes learning about learning visible!** 🎓📊🔬

