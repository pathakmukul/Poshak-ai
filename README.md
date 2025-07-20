# KapdaAI - AI-Powered Wardrobe Management

A React + Flask application that uses computer vision to detect, segment, and manage clothing items in your digital wardrobe.

## What We're Building

KapdaAI is an intelligent wardrobe management system that automatically detects and categorizes clothing items from photos, allowing users to build and manage their digital wardrobe with AI-powered segmentation, classification, and virtual try-on capabilities.

## Features

- **User Authentication**: Firebase-based login system with multi-user support and dummy accounts
- **Automatic Clothing Detection**: Uses SAM2 (Segment Anything Model 2) via Replicate API for precise clothing segmentation
- **Smart Classification**: CLIP-based classification for accurate clothing categorization (shirts, pants, shoes)
- **Digital Wardrobe**: Personal clothing collection with Firebase storage and detailed item view
- **Closet View**: NEW! Browse extracted clothing items in a clean grid layout with transparent backgrounds
- **Interactive Mask Editing**: Edit and refine AI detections with visual feedback and category-focused editing
- **Virtual Try-On**: Gemini API integration for AI-powered outfit visualization
- **Enhanced Segmentation**: Improved SAM2 parameters for better clothing detection (32 points grid, optimized thresholds)
- **Person Detection**: MediaPipe-based person extraction to reduce background noise before segmentation
- **Transparent Backgrounds**: All cropped clothing items now have transparent backgrounds for better visualization
- **Mask Persistence**: Intelligent caching system for instant re-access
- **Data Persistence**: Firebase emulator data persists across restarts
- **Multi-item Support**: Allows multiple items per category (e.g., pair of shoes)

## Tech Stack & Pipeline

### Frontend
- **React** - UI framework
- **Firebase SDK** - Authentication and storage
- **CSS Modules** - Component styling

### Backend
- **Flask** - Python web framework
- **SAM2** - Meta's Segment Anything Model for image segmentation
- **CLIP** - OpenAI's model for zero-shot image classification
- **Replicate API** - Cloud inference for fast processing
- **OpenCV** - Image processing
- **NumPy** - Numerical operations

### Infrastructure
- **Firebase Auth** - User authentication
- **Firebase Storage** - Cloud storage for images and masks
- **Firebase Local Emulator** - Local development environment
- **Replicate API** - Cloud-based SAM2 inference
- **Gemini API** - AI-powered virtual try-on and fashion advice

## Project Structure

### Frontend Files
- `src/App.js` - Clean landing page with navigation to Wardrobe and Closet views
- `src/Login.js` - User login/signup interface
- `src/Wardrobe.js` - Digital wardrobe with full photos and virtual try-on capabilities
- `src/Closet.js` - NEW! Clothing item browser showing extracted items with transparent backgrounds
- `src/closetService.js` - NEW! Service for fetching and organizing clothing items from Firebase
- `src/UploadSegmentModal.js` - Full-screen workflow for image upload and segmentation
- `src/firebase.js` - Firebase configuration and initialization
- `src/storageService.js` - Firebase storage operations for images and masks

### Backend Files
- `backend/flask_api.py` - Main Flask server with all API endpoints (SAM2 via Replicate only)
- `backend/config_improved.py` - Simplified configuration with enhanced SAM2 parameters
- `backend/services/gemini_service.py` - NEW! Gemini API service for virtual try-on functionality
- `backend/services/sam2_service.py` - SAM2 segmentation via Replicate API with mask conversion
- `backend/clip_classifier.py` - CLIP-based clothing classification (fast and accurate)
- `backend/services/person_extractor.py` - MediaPipe person detection to reduce background noise

### Configuration Files
- `.env` - Environment variables (API keys)
- `package.json` - Node.js dependencies
- `requirements.txt` - Python dependencies
- `firebase.json` - Firebase emulator configuration
- `storage.rules` - Firebase storage security rules

### Scripts
- `run-all.sh` - Starts both frontend and backend servers
- `start-firebase.sh` - Starts Firebase local emulator

## Processing Pipeline

1. **Image Upload** → User uploads photo locally (no Firebase until save)
2. **Segmentation** → SAM2 via Replicate API generates masks with enhanced parameters
3. **Classification** → CLIP classifies each mask as clothing type with original labels preserved
4. **Post-processing** → Creates visualizations with transparent backgrounds for closet view
5. **Manual Refinement** → User can edit AI selections with category-focused interface
6. **Storage** → Final selections saved to Firebase with metadata, visualizations, and closet-specific views
7. **Display** → Items viewable in Wardrobe (full photos) or Closet (extracted items)

## API Endpoints

- `POST /process` - Process image with SAM2 (Replicate API) and classify segments
- `POST /update-mask-labels` - Update mask classifications after manual editing
- `POST /get-gemini-data` - Get outfit details and fashion advice from Gemini
- `POST /prepare-wardrobe-gemini` - Prepare wardrobe items for virtual try-on
- `POST /gemini-tryon` - Perform virtual try-on with selected items

## Running the Application

```bash
# Simply run everything with one command
./run-all.sh
```

This script will:
- Clean up any existing processes on required ports
- Start Firebase emulators with data persistence
- Start the Flask backend
- Start the React frontend

The app runs on:
- Frontend: http://localhost:3000
- Backend: http://localhost:5001
- Firebase Emulator: http://localhost:4000

## Environment Variables

Required in `.env`:
```
REPLICATE_API_TOKEN=your_replicate_token  # For Replicate provider
FAL_KEY=your_fal_key                      # For FAL provider
GOOGLE_API_KEY=your_gemini_key            # Optional for try-on
```

## Switching SAM2 Providers

KapdaAI supports two providers for SAM2 segmentation:
- **Replicate** (default): The original implementation
- **FAL**: Alternative provider with similar performance

To switch providers, update the `provider` field in `backend/config_improved.py`:
```python
SAM2_PROVIDER_CONFIG = {
    "provider": "fal",  # Options: "replicate" or "fal"
    "fal_endpoint": "https://fal.run/fal-ai/sam2/auto-segment"
}
```

## Key Features Implementation

- **Multi-user Support**: Each user has isolated wardrobe storage with Firebase auth
- **Local-first Processing**: Images processed locally until explicit save to Firebase
- **Enhanced SAM2 Parameters**: Improved segmentation with 32x32 point grid, optimized thresholds, and multimask output
- **Transparent Backgrounds**: All clothing extractions use RGBA format with transparent backgrounds
- **Dual Views**: Wardrobe (full photos) and Closet (extracted items) for different use cases
- **Visual Feedback**: Color-coded mask editing (blue=shirt, red=pants, orange=shoes)
- **Batch Operations**: Multiple items can be selected per category
- **Persistent Storage**: Firebase emulator data saved between sessions
- **Mask Preservation**: Original AI labels preserved during manual edits
- **Category Isolation**: Edit one clothing category at a time without affecting others
- **Virtual Try-On**: AI-powered outfit visualization using Gemini API
- **Intelligent Processing**: Always uses Replicate API for consistent, fast results

## Cloud Run Deployment (Backend)

The backend is deployed on Google Cloud Run for scalability and cost-effectiveness:

```bash
# Option 1: Use the deployment script (recommended)
cd backend
./deploy.sh

# Option 2: Manual deployment
# Build with caching (ALWAYS use this for fast builds)
cd backend
gcloud builds submit --config cloudbuild.yaml

# Deploy to Cloud Run
gcloud run deploy kapdaai-backend \
  --image gcr.io/poshakai/kapdaai-backend:latest \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --timeout 540 \
  --allow-unauthenticated \
  --update-secrets="REPLICATE_API_TOKEN=replicate-api-token:latest,FAL_KEY=fal-api-key:latest,GOOGLE_API_KEY=gemini-api-key:latest"
```

### Setting up Secrets in Google Secret Manager

```bash
# Create secrets if they don't exist
echo -n "your-replicate-token" | gcloud secrets create replicate-api-token --data-file=-
echo -n "your-fal-key" | gcloud secrets create fal-api-key --data-file=-
echo -n "your-gemini-key" | gcloud secrets create gemini-api-key --data-file=-

# Update existing secrets
echo -n "new-replicate-token" | gcloud secrets versions add replicate-api-token --data-file=-
echo -n "new-fal-key" | gcloud secrets versions add fal-api-key --data-file=-
echo -n "new-gemini-key" | gcloud secrets versions add gemini-api-key --data-file=-

# Grant Cloud Run service account access to secrets
SERVICE_ACCOUNT="560568328203-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding replicate-api-token --member="serviceAccount:$SERVICE_ACCOUNT" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding fal-api-key --member="serviceAccount:$SERVICE_ACCOUNT" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding gemini-api-key --member="serviceAccount:$SERVICE_ACCOUNT" --role="roles/secretmanager.secretAccessor"
```

Features:
- **Docker Layer Caching**: Fast rebuilds with cloudbuild.yaml (saves 10-15 minutes)
- **8GB Memory**: Handles large models (CLIP, MediaPipe)
- **Auto-scaling**: Scales to zero when not in use
- **Environment**: Uses Google Secret Manager for API keys
- **SAM2 Service**: Modular architecture with proper mask format conversion