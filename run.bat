@echo off
REM Gen2Seg Frontend - Quick Start Script (Windows)
REM This script installs dependencies and runs the Streamlit app

echo.
echo Gen2Seg Image Segmentation Frontend
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully
echo.

REM Run Streamlit app
echo Starting Streamlit app...
echo The app will open in your browser automatically
echo If not, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py
