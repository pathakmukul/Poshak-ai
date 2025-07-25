# KapdaAI - AI-Powered Wardrobe Management

A full-stack application with React web app, React Native mobile app, and Flask backend that uses computer vision to detect, segment, and manage clothing items in your digital wardrobe.

## What We're Building

KapdaAI is an intelligent wardrobe management system that automatically detects and categorizes clothing items from photos, allowing users to build and manage their digital wardrobe with AI-powered segmentation, classification, and virtual try-on capabilities. Now with a unified architecture where Flask serves as the central API gateway for all Firebase operations, ensuring consistency between web and mobile apps.

## Features

### Core Features
- **User Authentication**: Firebase-based login system with multi-user support
- **Automatic Clothing Detection**: Uses Segformer B2 Clothes model for precise clothing segmentation and classification in one step
- **All-in-One AI Model**: Single model that segments and classifies 18 clothing categories (shirts, pants, shoes, dresses, etc.)
- **Digital Wardrobe**: Personal clothing collection with Firebase storage and detailed item view
- **Closet View**: Browse extracted clothing items in a clean grid layout with transparent backgrounds
- **Interactive Mask Editing**: Edit and refine AI detections with visual feedback and category-focused editing
- **Virtual Try-On**: Gemini API integration for AI-powered outfit visualization

### Architecture Features
- **Unified Backend Architecture**: Flask serves as the single API gateway for all Firebase operations
- **Cross-Platform Consistency**: Web and mobile apps use identical Flask endpoints
- **Mobile App with Local Caching**: React Native app with AsyncStorage for lightning-fast performance
- **Smart Image Resizing**: Automatic resizing at upload time for efficient processing
- **Content-Aware Cropping**: Generous padding ensures full clothing items are visible
- **Production Firebase**: Migrated from emulator to production Firebase with proper authentication
- **Optimized API Endpoints**: Single endpoint returns all clothing items (shirts, pants, shoes) for better performance

## Tech Stack & Pipeline

### Frontend
- **React** - UI framework
- **Firebase SDK** - Authentication and storage
- **CSS Modules** - Component styling

### Backend
- **Flask** - Python web framework
- **Segformer B2 Clothes** - Specialized model for clothing segmentation and classification
- **Hugging Face Transformers** - Model inference framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **PyTorch** - Deep learning framework

### Infrastructure
- **Firebase Auth** - User authentication (production)
- **Firebase Storage** - Cloud storage for images and masks
- **Firebase Admin SDK** - Server-side Firebase operations
- **Hugging Face API** - Optional cloud-based inference
- **Gemini API** - AI-powered virtual try-on and fashion advice

### Mobile App
- **React Native** - Cross-platform mobile framework
- **Expo** - Development and build tools
- **AsyncStorage** - Local caching for instant loading
- **React Navigation** - Native navigation

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
- `backend/flask_api.py` - Main Flask server with all API endpoints and Firebase gateway
- `backend/config_improved.py` - Application configuration
- `backend/services/gemini_service.py` - Gemini API service for virtual try-on functionality
- `backend/services/segformer_service.py` - Segformer model service for local inference
- `backend/services/segformer_api_service.py` - Hugging Face API integration for Segformer
- `backend/services/firebase_service.py` - Centralized Firebase operations service
- `backend/services/visualization_service.py` - Image processing and closet visualizations
- `backend/services/image_processing_service.py` - Image utilities and transformations

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
2. **Segmentation & Classification** → Segformer B2 Clothes model performs both tasks in one pass
3. **Post-processing** → Creates visualizations with transparent backgrounds for closet view
4. **Manual Refinement** → User can edit AI selections with category-focused interface
5. **Storage** → Final selections saved to Firebase with metadata, visualizations, and closet-specific views
6. **Display** → Items viewable in Wardrobe (full photos) or Closet (extracted items)

## API Endpoints

### Processing Endpoints
- `POST /process` - Process image with Segformer model for segmentation and classification
- `POST /update-mask-labels` - Update mask classifications after manual editing

### Firebase Gateway Endpoints (New)
- `POST /firebase/save-results` - Save processed results to Firebase (used by both web and mobile)
- `GET /firebase/images/<user_id>` - Get user's wardrobe images
- `GET /firebase/clothing-items/<user_id>` - Get all clothing items for closet (optimized)
- `GET /firebase/clothing-counts/<user_id>` - Get just the counts for smart sync (lightweight)
- `GET /firebase/mask-data/<user_id>/<image_name>` - Get mask data for specific image
- `POST /firebase/delete-image` - Delete image and associated data

### Gemini Endpoints
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
# API Keys
GOOGLE_API_KEY=your_gemini_key            # For virtual try-on

# Firebase Configuration (Production)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
FIREBASE_STORAGE_BUCKET=your-project.appspot.com

# Optional for Hugging Face API mode
USE_HF_API=false                          # Set to 'true' to use HF API instead of local model
HUGGINGFACE_API_TOKEN=your_hf_token       # Optional, for private models or higher rate limits
```

## Model Inference Options

KapdaAI uses the Segformer B2 Clothes model (MIT licensed) with two deployment options:

### Local Model (Default)
- Downloads and runs the model locally (~50MB)
- No internet required after initial download
- Best for development and privacy

### Hugging Face API
- Uses Hugging Face's inference API
- No local GPU/memory requirements
- Set `USE_HF_API=true` in your `.env` file

## Key Features Implementation

### Architecture
- **Unified Backend**: Flask serves as the single source of truth for all Firebase operations
- **No Duplicate Code**: Web and mobile apps share the same backend endpoints
- **Production Firebase**: Migrated from emulator to production with proper authentication
- **Smart Image Processing**: Automatic resizing at upload for efficient processing (max 1024px)

### User Experience
- **Multi-user Support**: Each user has isolated wardrobe storage with Firebase auth
- **Mobile App**: Full-featured React Native app with local caching for instant loading
- **User-Controlled Saving**: Preview segmentation results before saving (mobile)
- **Content-Aware Display**: Generous padding (50%) ensures full clothing items are visible
- **Pull-to-Refresh**: Manual sync option in mobile app

### Technical Features
- **Single Model Architecture**: Segformer B2 handles both segmentation and classification
- **18 Clothing Categories**: Detects shirts, pants, dresses, shoes, accessories, and more
- **Transparent Backgrounds**: All clothing extractions use RGBA format
- **Dual Views**: Wardrobe (full photos) and Closet (extracted items)
- **Visual Feedback**: Color-coded mask editing in web app
- **Optimized Endpoints**: Single API call returns all clothing items
- **Local Caching**: Mobile app uses AsyncStorage for offline support
- **Smart Sync Strategy**: Intelligent count-based synchronization only when needed

## Segformer Model Details

KapdaAI uses the **mattmdjaga/segformer_b2_clothes** model:
- **License**: MIT (free for commercial use)
- **Model Size**: ~50MB
- **Categories**: 18 clothing/body part classes
- **Architecture**: SegFormer B2 fine-tuned on ATR dataset
- **Performance**: Runs on CPU or GPU (MPS on Apple Silicon)

### Supported Categories:
- Clothing: Upper-clothes, Pants, Skirt, Dress, Belt, Left-shoe, Right-shoe
- Accessories: Hat, Sunglasses, Bag, Scarf
- Body parts: Hair, Face, Arms, Legs (marked as non-clothing)

## Local Development Tips

- **First Run**: Model download happens on first run (~50MB, cached afterward)
- **Apple Silicon**: Automatically uses MPS (Metal) for faster inference
- **Multiple Terminals**: Run Flask and React in separate terminals for faster restarts
- **Mobile Development**: Use physical device for best performance, ensure backend is accessible
- **Image Processing**: Images automatically resized to max 1024px for efficiency
- **Closet Loading**: Mobile app loads instantly from cache, syncs in background

## Mobile App

The mobile app is located in `/mobile-app` directory. See the [Mobile App README](mobile-app/README.md) for detailed setup and features.

Key mobile features:
- Lightning-fast loading with local caching
- Works offline with cached data
- User-controlled save after segmentation
- Pull-to-refresh for manual sync
- Smart sync strategy:
  - On app open: Sync count and images
  - On closet reopen: Check count only
  - If count matches: Use cached data
  - If count differs: Sync only differences
  - No sync during active session
- Optimized for both iOS and Android