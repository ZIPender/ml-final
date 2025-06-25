@echo off
setlocal enabledelayedexpansion

REM YouTube Timestamp Generator - Frontend Script
echo 🌐 Starting Next.js frontend...

REM Check if we're in the right directory
if not exist "src" (
    echo ❌ Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    echo Make sure to check "Add to PATH" during installation
    pause
    exit /b 1
)

REM Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not available
    echo Please ensure Node.js is properly installed with npm
    pause
    exit /b 1
)

echo ✅ Node.js version:
node --version
echo ✅ npm version:
npm --version
echo.

REM Change to src directory and stay there
cd src
echo 📁 Changed to directory: %CD%

REM Install frontend dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo 📦 Installing Node.js dependencies...
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ✅ Dependencies already installed
)

REM Start frontend server
echo 🌐 Starting Next.js frontend server on http://localhost:3000...
echo 📝 Running: npm run dev
echo.

REM Run the dev server (this will keep running)
npm run dev

REM This line will only execute if npm run dev fails or is stopped
echo.
echo ❌ Server stopped or failed to start
pause 