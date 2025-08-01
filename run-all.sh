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

cd backend && python -u flask_api.py 2>&1 | tee -a flask.log &
FLASK_PID=$!

# Clear Chrome's localStorage for localhost:3000
echo ""
echo "Clearing browser cache for localhost:3000..."

# Create a Python script to clear Chrome's localStorage
python3 << 'EOF'
import os
import sqlite3
import shutil
from pathlib import Path

# Find Chrome's profile directory
home = Path.home()
chrome_paths = [
    home / "Library/Application Support/Google/Chrome/Default/Local Storage",  # macOS
    home / ".config/google-chrome/Default/Local Storage",  # Linux
    home / "AppData/Local/Google/Chrome/User Data/Default/Local Storage"  # Windows
]

cleared = False
for chrome_path in chrome_paths:
    if chrome_path.exists():
        # Look for localhost files
        for file in chrome_path.glob("*localhost*"):
            try:
                print(f"Removing cache file: {file.name}")
                file.unlink()
                cleared = True
            except Exception as e:
                print(f"Could not remove {file.name}: {e}")

if cleared:
    print("✓ Browser cache cleared!")
else:
    print("⚠ No cache files found or unable to clear")
EOF

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