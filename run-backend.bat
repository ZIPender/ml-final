@echo off
setlocal enabledelayedexpansion

REM YouTube Timestamp Generator - Backend Script
echo 🚀 Starting FastAPI backend...

REM Check if we're in the right directory
if not exist "backend\app\main.py" (
    echo ❌ Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if Python is installed and accessible
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not available
    echo Please ensure Python is properly installed with pip
    pause
    exit /b 1
)

REM Setup Python virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo 🔧 Installing Python dependencies...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

pip install -r backend\requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Start backend server
echo 🔧 Starting FastAPI backend server on http://localhost:8000...
uvicorn backend.app.main:app --reload --port 8000

pause 