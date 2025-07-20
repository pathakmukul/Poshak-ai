#!/bin/bash

echo "Starting OpenAI Local Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Note: OpenAI API key is set directly in openai_flask_api.py

# Start the Flask server
echo "Starting Flask server on port 5002..."
python openai_flask_api.py