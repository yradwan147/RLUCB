#!/bin/bash
# Quick launcher for the GUI application

echo "🎓 Starting UCB Quiz Master GUI..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo ""
    echo "Please create a .env file with your OpenAI API key:"
    echo "  echo 'OPENAI_API_KEY=your_key_here' > .env"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "❌ tkinter not found. Please install Python with tkinter support."
    exit 1
fi

if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the GUI
echo "🚀 Launching GUI..."
echo ""
python3 gui_app.py

