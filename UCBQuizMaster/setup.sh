#!/bin/bash
# Setup script for UCB Quiz Master

echo "🎓 Setting up UCB Quiz Master..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your OpenAI API key:"
echo "   echo 'OPENAI_API_KEY=your_key_here' > .env"
echo ""
echo "2. Run the application:"
echo "   python main.py"
echo ""
echo "To activate the virtual environment in the future:"
echo "   source venv/bin/activate"
echo ""

