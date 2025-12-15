#!/bin/bash

# Gen2Seg Frontend - Quick Start Script
# This script installs dependencies and runs the Streamlit app

echo "🔬 Gen2Seg Image Segmentation Frontend"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip is not installed"
    echo "Please install pip: https://pip.pypa.io/en/stable/installation/"
    exit 1
fi

echo "✅ pip found"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Dependencies installed successfully"
echo ""

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
echo "The app will open in your browser automatically"
echo "If not, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
