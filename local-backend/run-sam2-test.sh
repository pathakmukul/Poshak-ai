#!/bin/bash

echo "Starting SAM2 + CLIP Testing Server..."
echo "=================================="
echo "Edit SAM2_TEST_CONFIG in sam2_clip_test.py to test different configurations"
echo "Access the interface at: http://localhost:5001"
echo "=================================="

# Skip venv activation - use current conda environment

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Load .env if it exists
if [ -f "../backend/.env" ]; then
    export $(cat ../backend/.env | grep -v '^#' | xargs)
fi

# Run the test server
python sam2_clip_test.py