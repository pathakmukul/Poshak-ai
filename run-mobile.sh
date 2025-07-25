#!/bin/bash

echo "Starting KapdaAI Mobile Development Environment..."
echo "================================================"
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
lsof -ti:5001,9099,9199 | xargs kill -9 2>/dev/null || true
sleep 2

# Start Firebase emulators in background
echo "Starting Firebase Emulators..."
firebase emulators:start --only auth,storage --import=./firebase-data --export-on-exit &
FIREBASE_PID=$!

# Wait for Firebase to start
echo "Waiting for Firebase to initialize..."
sleep 5

# Start Flask backend with proper host binding
echo ""
echo "Starting Flask backend..."
cd backend && python flask_api.py &
FLASK_PID=$!

# Wait for Flask to start
sleep 3

# Start React Native app
echo ""
echo "Starting React Native Mobile App..."
cd ../mobile-app/mobile-app && npm start &
EXPO_PID=$!

echo ""
echo "================================================"
echo "All services started!"
echo ""
echo "Firebase Emulator UI: http://localhost:4000"
echo "Flask API: http://localhost:5001"
echo "Expo Dev Server: http://localhost:8081"
echo ""
echo "Press 'a' in the Expo terminal to open Android"
echo "Press 'i' in the Expo terminal to open iOS"
echo ""
echo "Press Ctrl+C to stop all services"
echo "================================================"

# Keep script running
wait