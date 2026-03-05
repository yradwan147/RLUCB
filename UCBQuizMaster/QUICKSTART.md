# 🚀 Quick Start Guide

Get started with UCB Quiz Master in under 2 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script (creates a virtual environment):
```bash
chmod +x setup.sh
./setup.sh
```

## Step 2: Configure Your API Key

Create a `.env` file in this directory:

```bash
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

**Get your API key**: https://platform.openai.com/api-keys

## Step 3: Run the Application

**GUI Version (Recommended):**
```bash
python gui_app.py
```

**CLI Version:**
```bash
python main.py
```

Or if you made them executable:
```bash
./gui_app.py  # or ./main.py
```

## That's It! 🎉

The application will:
1. Ask for a topic (e.g., "Machine Learning", "Calculus", "Python")
2. Generate subtopics automatically
3. Generate 10 questions per subtopic
4. Quiz you adaptively, focusing on your weak areas
5. Show explanations for incorrect answers
6. Generate more questions automatically when needed
7. **[GUI Only]** Visualize the UCB algorithm in real-time with Debug Mode!

## During the Quiz

- **Answer**: Type A, B, C, or D
- **Stats**: Type `s` to see your progress
- **Quit**: Type `q` to exit

## Example Session

```
Enter the topic: Python Programming
Number of subtopics (default 5): 5

[Generating subtopics and questions...]

Question 1
Subtopic: Python Basics
What is the correct way to create a list?
  A. list = ()
  B. list = []
  C. list = {}
  D. list = <>

Your answer: B
✅ Correct!
```

## Tips

- Be specific with topics: "Linear Algebra" is better than "Math"
- Answer honestly for best adaptive learning results
- Review explanations when you get questions wrong
- Check progress every 10 questions with the `s` command

## Troubleshooting

**API Key Error?**
- Make sure `.env` file is in the same directory as `main.py`
- Check that your API key is valid at https://platform.openai.com/

**Rate Limit?**
- Wait a few minutes between sessions
- The default model (`gpt-4o-mini`) has generous rate limits

**Need Better Questions?**
- Edit `config.py` and change `OPENAI_MODEL` to `"gpt-4o"`

---

For more details, see the full [README.md](README.md)

