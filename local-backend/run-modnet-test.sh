#!/bin/bash

echo "Starting Segmentation + SAM2 + CLIP Testing Server..."
echo "============================================"
echo "Stage 1: Segmentation model for person extraction"
echo "Stage 2: SAM2 for detailed clothing segmentation"
echo "Stage 3: CLIP classification of segments"
echo "============================================"
echo "Access the interface at: http://localhost:5003"
echo "============================================"

# Skip venv activation - use current conda environment

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Load .env if it exists
if [ -f "../backend/.env" ]; then
    export $(cat ../backend/.env | grep -v '^#' | xargs)
fi

# Run the test server
python modnet_sam2_test.py