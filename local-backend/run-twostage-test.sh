#!/bin/bash

echo "Starting Two-Stage SAM2 + CLIP Testing Server..."
echo "============================================"
echo "Stage 1: Light SAM2 to find person (largest segment)"
echo "Stage 2: Detailed SAM2 on cropped person for clothing"
echo "Stage 3: CLIP classification of all segments"
echo "============================================"
echo "Access the interface at: http://localhost:5002"
echo "============================================"

# Skip venv activation - use current conda environment

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Load .env if it exists
if [ -f "../backend/.env" ]; then
    export $(cat ../backend/.env | grep -v '^#' | xargs)
fi

# Run the test server
python sam2_twostage_test.py