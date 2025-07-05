#!/bin/bash

echo "Starting Clothing Detection App..."
echo "================================="

# Optimize for M4 MacBook Pro
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Enable MPS with CPU fallback

echo "Running with MPS GPU acceleration enabled"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Install Flask dependencies if needed
echo "Checking Flask dependencies..."
pip install flask flask-cors replicate python-dotenv requests google-generativeai >/dev/null 2>&1

# Kill any existing process on port 5001
echo "Checking for existing Flask process on port 5001..."
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
sleep 1

# Start Flask API in background
echo "Starting Flask API server..."
cd "$PROJECT_ROOT"
python react-app/flask_api.py &
FLASK_PID=$!

# Wait for Flask to start
sleep 3

# Check if Flask started successfully
if ! ps -p $FLASK_PID > /dev/null; then
    echo "Flask failed to start. Check the error messages above."
    exit 1
fi

# Start React app
echo "Starting React app..."
cd "$SCRIPT_DIR"
npm start

# Kill Flask when React exits
kill $FLASK_PID