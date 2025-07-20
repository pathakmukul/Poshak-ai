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
    print('Downloading CLIP models...'); \
    # Download the smaller base model for faster CPU inference \
    print('Downloading clip-vit-base-patch32 (88M params, 4x faster)...'); \
    processor_base = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    model_base = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    print('Base model downloaded successfully'); \
    # Also download the large model as backup \
    print('Downloading clip-vit-large-patch14 (428M params) as backup...'); \
    processor_large = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14'); \
    model_large = CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); \
    print('Large model downloaded successfully'); \
    print('Both models downloaded successfully'); \
    print('Cache contents:', os.listdir('/app/.cache/huggingface') if os.path.exists('/app/.cache/huggingface') else 'No cache dir')"

# Force rebuild with new timestamp and cache bust
ARG CACHEBUST=1
RUN echo "Build timestamp: $(date) - Cache bust: ${CACHEBUST}" > /tmp/build_time.txt

# Clear any potential Python cache
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy only Python files first (most likely to change)
COPY *.py ./
COPY services/ ./services/

# Verify critical files are updated
RUN echo "Verifying SAM2 service update..." && \
    grep "pred_iou_thresh.*0.88" services/sam2_service.py || \
    (echo "ERROR: SAM2 service not updated!" && exit 1)

# Then copy everything else
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the Flask app without preload (causes issues in Cloud Run)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_api:app