#!/bin/bash

echo "Starting KapdaAI Mobile App with Firebase & Backend..."
echo "=================================================="
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
lsof -ti:3000,4000,4400,4500,5001,8081,9099,9199 | xargs kill -9 2>/dev/null || true
sleep 2

# Start Firebase emulators in background with data persistence
echo "Starting Firebase Emulators..."
firebase emulators:start --only auth,storage --import=./firebase-data --export-on-exit &
FIREBASE_PID=$!

# Wait for Firebase to start
echo "Waiting for Firebase to initialize..."
sleep 8

# Start Flask backend
echo ""
echo "Starting Flask backend..."
# Activate virtual environment if it exists
if [ -f "backend/venv/bin/activate" ]; then
    source backend/venv/bin/activate
elif [ -f "../kapdaai/bin/activate" ]; then
    source ../kapdaai/bin/activate
fi

# Kill any existing Flask processes first
echo "Killing any existing Flask processes..."
pkill -f flask_api.py || true
sleep 2

cd backend && python flask_api.py 2>&1 | tee flask.log &
FLASK_PID=$!
cd ..

echo ""
echo "=================================================="
echo "All backend services started!"
echo ""
echo "Firebase Emulator UI: http://localhost:4000"
echo "Flask API: http://localhost:5001"
echo ""
echo "Now starting Mobile App..."
echo "The Expo QR code will appear below"
echo "=================================================="

# Start React Native mobile app (in foreground to show QR code)
if [ -d "mobile-app/mobile-app" ]; then
    cd mobile-app/mobile-app && npm start
elif [ -d "mobile-app" ]; then
    cd mobile-app && npm start
else
    echo "ERROR: Cannot find mobile-app directory!"
fi