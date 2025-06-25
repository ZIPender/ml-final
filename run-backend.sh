#!/bin/bash

# YouTube Timestamp Generator - Backend Script
echo "🚀 Starting FastAPI backend..."

# Check if we're in the right directory
if [ ! -f "backend/app/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Setup Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment and install dependencies
echo "🔧 Installing Python dependencies..."
source .venv/bin/activate
pip install -r backend/requirements.txt

# Start backend server
echo "🔧 Starting FastAPI backend server on http://localhost:8000..."
uvicorn backend.app.main:app --reload --port 8000 