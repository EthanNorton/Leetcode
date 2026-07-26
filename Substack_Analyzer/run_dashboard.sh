#!/bin/bash

# Substack Analyzer Dashboard Launcher
# Starts the interactive Streamlit dashboard

echo "🚀 Starting Substack Analyzer Dashboard..."
echo ""

# Check if dependencies are installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing now..."
    pip3 install -r requirements.txt
    echo ""
fi

# Download NLTK data if needed
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo "✅ Setup complete!"
echo ""
echo "📊 Opening dashboard in your browser..."
echo "   Press Ctrl+C to stop the dashboard"
echo ""

# Run the dashboard
streamlit run dashboard.py --server.port 8501 --server.headless true
