# Use Python 3.9 for better MediaPipe compatibility
FROM python:3.9-slim

# Install system dependencies required for MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libglfw3 \
    libgles2-mesa \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    opencv-python \
    numpy \
    Pillow \
    torch \
    torchvision \
    transformers

# Pre-download ML models to avoid Hugging Face rate limits
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import CLIPProcessor, CLIPModel; \
    import os; \
    print('Cache dir:', os.environ.get('HF_HOME')); \
    print('Downloading CLIP model...'); \
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14'); \
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); \
    print('Model downloaded successfully'); \
    print('Cache contents:', os.listdir('/app/.cache/huggingface') if os.path.exists('/app/.cache/huggingface') else 'No cache dir')"

# Copy only Python files first (most likely to change)
COPY *.py ./
COPY services/ ./services/

# Then copy everything else
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the Flask app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_api:app