@echo off
REM Substack Analyzer Dashboard Launcher for Windows

echo Starting Substack Analyzer Dashboard...
echo.

REM Check if dependencies are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo Setup complete!
echo.
echo Opening dashboard in your browser...
echo Press Ctrl+C to stop the dashboard
echo.

REM Run the dashboard
streamlit run dashboard.py --server.port 8501 --server.headless true
