#!/bin/bash

# Jupyter Notebook Launcher for Substack Analyzer
# Opens the interactive LSTM analysis notebook

echo "🚀 Starting Jupyter Notebook for Substack Analyzer..."
echo ""

# Check if dependencies are installed
if ! python3 -c "import jupyter" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing now..."
    pip3 install -r requirements.txt
    echo ""
fi

# Download NLTK data if needed
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null

echo "✅ Setup complete!"
echo ""
echo "📊 Opening Jupyter Notebook..."
echo "   The notebook will open in your browser"
echo "   Press Ctrl+C twice to stop the server"
echo ""
echo "📝 Notebook: Interactive_Substack_Analysis.ipynb"
echo "   - LSTM predictions with time-decay weighting"
echo "   - Interactive Python environment"
echo "   - Real-time testing with your Substack"
echo ""

# Run Jupyter
jupyter notebook Interactive_Substack_Analysis.ipynb
