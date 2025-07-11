#!/bin/bash

echo "Starting KapdaAI with Firebase Local..."
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
lsof -ti:3000,4000,4400,4500,5001,9099,9199 | xargs kill -9 2>/dev/null || true
sleep 2

# Start Firebase emulators in background
echo "Starting Firebase Emulators..."
firebase emulators:start --only auth,storage &
FIREBASE_PID=$!

# Wait for Firebase to start
echo "Waiting for Firebase to initialize..."
sleep 8

# Start Flask backend
echo ""
echo "Starting Flask backend..."
source /Users/mukulpathak/PROJECTS/KapdaAI/kapdaai/bin/activate
python flask_api.py &
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
echo "Firebase Emulator UI: http://localhost:4000"
echo "React App: http://localhost:3000"
echo "Flask API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================"

# Keep script running
wait