#!/bin/bash

# Deploy KapdaAI Backend to Google Cloud Run

echo "🚀 Deploying KapdaAI Backend to Cloud Run..."

# Set project variables
PROJECT_ID="poshakai"
SERVICE_NAME="kapdaai-backend"
REGION="us-central1"

# Build with caching
echo "📦 Building Docker image with caching..."
gcloud builds submit --config cloudbuild.yaml

# Check if secrets exist
echo "🔐 Checking secrets..."
for secret in replicate-api-token fal-api-key gemini-api-key; do
    if ! gcloud secrets describe $secret &>/dev/null; then
        echo "❌ Secret '$secret' not found. Please create it first:"
        echo "   echo -n 'your-token' | gcloud secrets create $secret --data-file=-"
        exit 1
    fi
done

# Deploy to Cloud Run with secrets
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
    --platform managed \
    --region $REGION \
    --memory 8Gi \
    --timeout 540 \
    --allow-unauthenticated \
    --update-secrets="REPLICATE_API_TOKEN=replicate-api-token:latest,FAL_KEY=fal-api-key:latest,GOOGLE_API_KEY=gemini-api-key:latest"

echo "✅ Deployment complete!"
echo "🌐 Service URL: https://$SERVICE_NAME-<hash>-uc.a.run.app"