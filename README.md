# KapdaAI - AI-Powered Wardrobe Management

A React + Flask application that uses computer vision to detect, segment, and manage clothing items in your digital wardrobe.

## What We're Building

KapdaAI is an intelligent wardrobe management system that automatically detects and categorizes clothing items from photos, allowing users to build and manage their digital wardrobe with AI-powered segmentation and classification.

## Features

- **User Authentication**: Firebase-based login system with multi-user support and dummy accounts
- **Automatic Clothing Detection**: Uses SAM2 (Segment Anything Model 2) for precise clothing segmentation
- **Smart Classification**: CLIP-based classification for accurate clothing categorization (shirts, pants, shoes)
- **Digital Wardrobe**: Personal clothing collection with Firebase storage and detailed item view
- **Interactive Mask Editing**: Edit and refine AI detections with visual feedback and category-focused editing
- **Fast Processing**: Replicate API integration for cloud-based processing
- **Local Processing**: Support for local SAM2 models (Tiny, Small, Base, Large)
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

## Project Structure

### Frontend Files
- `src/App.js` - Main app component with authentication flow
- `src/Login.js` - User login/signup interface
- `src/Wardrobe.js` - Digital wardrobe grid view with item management
- `src/UploadSegmentModal.js` - Full-screen workflow for image upload and segmentation
- `src/FileUpload.js` - (Legacy) Original file upload component
- `src/firebase.js` - Firebase configuration and initialization
- `src/storageService.js` - Firebase storage operations for images and masks

### Backend Files
- `flask_api.py` - Main Flask server with all API endpoints
- `config_improved.py` - Configuration for models and processing parameters
- `improved_siglip_classification.py` - SigLIP-based clothing classification
- `clip_classifier.py` - CLIP-based clothing classification (preferred)
- `fashion_siglip_classifier.py` - Fashion-specific SigLIP implementation

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
2. **Segmentation** → SAM2 generates masks for all objects in image
3. **Classification** → CLIP classifies each mask as clothing type with original labels preserved
4. **Post-processing** → Best masks selected per category (shirt, pants, shoes) with proper state tracking
5. **Manual Refinement** → User can edit AI selections with category-focused interface
6. **Storage** → Final selections saved to Firebase with metadata and visualizations

## API Endpoints

- `POST /process` - Process image with SAM2 and classify segments
- `POST /update-mask-labels` - Update mask classifications
- `POST /apply-mask-edits` - Apply user edits to masks
- `GET /get-all-masks-with-classes` - Retrieve all masks for editing

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
REPLICATE_API_TOKEN=your_replicate_token
GOOGLE_API_KEY=your_gemini_key  # Optional for try-on
```

## Key Features Implementation

- **Multi-user Support**: Each user has isolated wardrobe storage with Firebase auth
- **Local-first Processing**: Images processed locally until explicit save to Firebase
- **Smart Caching**: Processed masks cached for instant re-access
- **Visual Feedback**: Color-coded mask editing (blue=shirt, red=pants, orange=shoes)
- **Batch Operations**: Multiple items can be selected per category
- **Persistent Storage**: Firebase emulator data saved between sessions
- **Mask Preservation**: Original AI labels preserved during manual edits
- **Category Isolation**: Edit one clothing category at a time without affecting others
- **Wardrobe Details**: Click any item to view segmentation breakdown with visualizations
- **Intelligent Defaults**: Always uses Replicate API for wardrobe operations (fast mode)