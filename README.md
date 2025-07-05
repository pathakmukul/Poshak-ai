# SAM2 + SigLIP + Gemini Clothing Detection & Virtual Try-On

A React + Flask application that uses SAM2 for segmentation, SigLIP for clothing classification, and Google Gemini for virtual try-on.

## Features

- **Automatic Clothing Detection**: Uses SAM2 to segment clothing items automatically
- **Zero-shot Classification**: SigLIP classifies detected segments into shirts, pants, and shoes
- **Virtual Try-On**: Google Gemini 2.0 Flash generates realistic try-on results
- **Multiple SAM2 Models**: Support for Tiny, Small, Base, and Large models (both v2.0 and v2.1)
- **Cloud Processing**: Replicate API integration for faster processing
- **Persistent Masks**: Automatically saves masks for reuse without re-processing

## Prerequisites

- Python 3.8+
- Node.js 14+
- SAM2 model checkpoints (download from Meta)

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install React dependencies:
   ```bash
   cd react-app
   npm install
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Running the Application

```bash
cd react-app
./run.sh
```

This will start both the Flask backend (port 5001) and React frontend (port 3000).

## API Keys Required

- **Replicate API Token**: For cloud-based SAM2 processing
- **Google Gemini API Key**: For virtual try-on functionality

## Usage

1. Select a person image from the dropdown
2. Choose a SAM2 model size (or use Replicate for faster processing)
3. Click "Generate Masks" to detect clothing items
4. Select a clothing type tab (Shirt, Pants, or Shoes)
5. Choose a garment for virtual try-on
6. Click "Try On with Gemini" to see the result

## Saved Masks

Once masks are generated, they're automatically saved as:
- `person_mask_shirt.png`
- `person_mask_pants.png`
- `person_mask_shoes.png`

These can be loaded instantly on subsequent runs using the "Use Saved Masks" button.