#!/bin/bash

echo "🚀 Starting Optimizer Lens..."
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm is not installed. Please install pnpm."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install Node dependencies
echo "📦 Installing Node dependencies..."
pnpm install

# Start the Python API in the background
echo "🐍 Starting Python API on http://localhost:8000..."
python core/api.py &
API_PID=$!

# Wait for API to start
sleep 3

# Start the Next.js frontend
echo "⚛️  Starting Next.js frontend on http://localhost:3000..."
pnpm dev

# Cleanup on exit
trap "kill $API_PID" EXIT
