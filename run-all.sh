#!/bin/bash

echo "Starting KapdaAI Production Mode..."
echo "========================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Clean up any existing processes on required ports
echo "Cleaning up existing processes on ports..."
lsof -ti:3000,5001 | xargs kill -9 2>/dev/null || true
sleep 2

# Start Flask backend
echo ""
echo "Starting Flask backend..."
source /Users/mukulpathak/PROJECTS/KapdaAI/kapdaai/bin/activate

# Kill any existing Flask processes first
echo "Killing any existing Flask processes..."
pkill -f flask_api.py || true
sleep 2

cd backend && python flask_api.py 2>&1 | tee flask.log &
FLASK_PID=$!

# Start React frontend
echo ""
echo "Starting React frontend..."
npm start &
REACT_PID=$!

echo ""
echo "========================================"
echo "All services started!"
echo ""
echo "React App: http://localhost:3000"
echo "Flask API: http://localhost:5001"
echo ""
echo "NOTE: Using production Firebase. Make sure you have:"
echo "1. serviceAccountKey.json in the project root"
echo "2. .env file with GOOGLE_APPLICATION_CREDENTIALS=./serviceAccountKey.json"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================"

# Keep script running
wait