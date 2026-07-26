@echo off
REM Jupyter Notebook Launcher for Windows

echo Starting Jupyter Notebook for Substack Analyzer...
echo.

REM Check if dependencies are installed
python -c "import jupyter" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>nul

echo Setup complete!
echo.
echo Opening Jupyter Notebook...
echo The notebook will open in your browser
echo Press Ctrl+C to stop the server
echo.
echo Notebook: Interactive_Substack_Analysis.ipynb
echo   - LSTM predictions with time-decay weighting
echo   - Interactive Python environment
echo   - Real-time testing with your Substack
echo.

REM Run Jupyter
jupyter notebook Interactive_Substack_Analysis.ipynb
