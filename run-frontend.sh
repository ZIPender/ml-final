#!/bin/bash

# YouTube Timestamp Generator - Frontend Script
echo "🌐 Starting Next.js frontend..."

# Check if we're in the right directory
if [ ! -d "src" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "src/node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd src
    npm install
    cd ..
fi

# Start frontend server
echo "🌐 Starting Next.js frontend server on http://localhost:3000..."
cd src
npm run dev 