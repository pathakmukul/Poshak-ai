from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import os
import sys
import numpy as np
import base64
from io import BytesIO
import io
from PIL import Image
import torch
import replicate
from dotenv import load_dotenv
import requests
import json
import pickle
from datetime import datetime
import threading

# Load environment variables from .env file
load_dotenv()

# Removed Firebase Admin - using client-side Firebase instead

# Import Google GenerativeAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI package not installed. Run: pip install google-generativeai")

# Optimize for Apple Silicon
torch.set_num_threads(4)  # Use 4 CPU threads (better balance)
if torch.backends.mps.is_available():
    print(f"MPS (Metal Performance Shaders) is available!")
    print(f"Using Apple Silicon GPU acceleration")

from transformers import AutoProcessor, AutoModel
import time

# Import configuration
from config_improved import CLASSIFICATION_CONFIG, DEBUG_CONFIG, PERSON_EXTRACTION_CONFIG
# Removed SigLIP imports - using CLIP only
from clip_classifier import classify_with_clip, classify_batch_with_clip

# Import Gemini service
from services.gemini_service import GeminiService

# Import person extractor
from services.person_extractor import extract_person_from_image

# Import SAM2 service and provider config
from services.sam2_service import process_with_replicate, process_with_fal
from config_improved import SAM2_PROVIDER_CONFIG

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global models for CLIP classification
processor = None
model = None
_model_lock = threading.Lock()  # Thread lock for safe model loading

# Initialize Gemini service
gemini_service = GeminiService()

# Models will be loaded on first use to avoid Cloud Run preload issues

# No Firebase Admin needed - client handles Firebase

def load_classification_models():
    """Load classification models with thread safety"""
    global processor, model
    
    if processor is None or model is None:
        with _model_lock:  # Ensure only one thread loads the model
            # Double-check after acquiring lock
            if processor is None or model is None:
                print("Thread acquired lock, loading CLIP models...")
                from clip_classifier import load_clip_model
                processor, model = load_clip_model()
                print("CLIP models loaded and cached in memory!")

def image_to_base64(image):
    """Convert numpy array to base64 string"""
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def file_to_base64(file_path):
    """Convert file content directly to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def create_clothing_visualization(image_rgb, masks, clothing_type, for_gemini=False):
    """Create visualization for specific clothing type
    
    Args:
        image_rgb: Original image
        masks: List of detected masks
        clothing_type: Type of clothing to visualize
        for_gemini: If True, creates clean mask without labels for AI processing
    """
    # Filter for specific clothing type
    if clothing_type == "shirt":
        type_labels = ["shirt"]
    elif clothing_type == "pants":
        type_labels = ["pants"]
    elif clothing_type == "shoes":
        type_labels = ["shoes"]
    else:  # all items
        type_labels = ["shirt", "pants", "shoes"]
    
    clothing_masks = [m for m in masks if m.get('label', '') in type_labels and not m.get('skip_viz', False)]
    
    # Special handling for shoes - but respect manual selections
    if clothing_type == "shoes" and len(clothing_masks) > 0:
        # Check if these are manual selections (confidence = 1.0)
        manual_selections = [m for m in clothing_masks if m.get('confidence', 0) == 1.0]
        
        if manual_selections:
            # User has manually selected shoes - use ALL of them
            clothing_masks = manual_selections
            print(f"  Using {len(manual_selections)} manually selected shoes")
        else:
            # Automatic selection - use the old logic but allow up to 2 shoes
            if len(clothing_masks) > 2:
                # Sort by confidence and take top 2
                clothing_masks = sorted(clothing_masks, key=lambda x: x.get('confidence', 0), reverse=True)[:2]
    
    # Create visualization
    overlay = image_rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, mask_dict in enumerate(clothing_masks):
        mask = mask_dict['segmentation']
        label = mask_dict.get('label', 'unknown')
        conf = mask_dict.get('confidence', 0.0)
        
        # Apply colored mask
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask] = colors[i % len(colors)]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Add label text only if not for Gemini
        if not for_gemini:
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                cy = int(y_indices.mean())
                cx = int(x_indices.mean())
                
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                text = f"{label.upper()} {conf:.0%}"
                
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(overlay_bgr, (cx-5, cy-text_h-5), (cx+text_w+5, cy+5), (0,0,0), -1)
                cv2.putText(overlay_bgr, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                
                overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return overlay, len(clothing_masks)

def create_closet_visualization(image_rgb, masks, clothing_type):
    """Create visualization for closet - shows only clothing items with transparent background
    
    Args:
        image_rgb: Original image
        masks: List of detected masks
        clothing_type: Type of clothing to visualize ('shirt', 'pants', 'shoes', or 'all')
    """
    h, w = image_rgb.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Filter for specific clothing type
    if clothing_type == "all":
        type_labels = ["shirt", "pants", "shoes"]
    else:
        type_labels = [clothing_type]
    
    clothing_masks = [m for m in masks if m.get('label', '') in type_labels and not m.get('skip_viz', False)]
    
    # Apply all relevant masks
    for mask_dict in clothing_masks:
        mask = mask_dict['segmentation']
        # Copy RGB values where mask is True
        result[mask, :3] = image_rgb[mask]
        # Set alpha to opaque where mask is True
        result[mask, 3] = 255
    
    # Convert to PIL and save as base64
    pil_img = Image.fromarray(result, mode='RGBA')
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode(), len(clothing_masks)

def generate_all_visualizations(image_rgb, masks):
    """Generate all clothing visualizations at once to avoid code duplication
    
    Returns:
        dict: Dictionary containing all visualization images and counts
    """
    all_items_img, all_items_count = create_clothing_visualization(image_rgb, masks, "all", for_gemini=True)
    shirt_img, shirt_count = create_clothing_visualization(image_rgb, masks, "shirt", for_gemini=True)
    pants_img, pants_count = create_clothing_visualization(image_rgb, masks, "pants", for_gemini=True)
    shoes_img, shoes_count = create_clothing_visualization(image_rgb, masks, "shoes", for_gemini=True)
    
    # Create closet visualizations (transparent background, original positions)
    closet_all_img, _ = create_closet_visualization(image_rgb, masks, "all")
    closet_shirt_img, _ = create_closet_visualization(image_rgb, masks, "shirt")
    closet_pants_img, _ = create_closet_visualization(image_rgb, masks, "pants")
    closet_shoes_img, _ = create_closet_visualization(image_rgb, masks, "shoes")
    
    return {
        'all_items_img': image_to_base64(all_items_img),
        'all_items_count': all_items_count,
        'shirt_img': image_to_base64(shirt_img),
        'shirt_count': shirt_count,
        'pants_img': image_to_base64(pants_img),
        'pants_count': pants_count,
        'shoes_img': image_to_base64(shoes_img),
        'shoes_count': shoes_count,
        'closet_visualizations': {
            'all': closet_all_img,
            'shirt': closet_shirt_img,
            'pants': closet_pants_img,
            'shoes': closet_shoes_img
        }
    }

def create_raw_sam2_visualization(image_rgb, masks):
    """Create grid visualization of ALL raw SAM2 masks"""
    h, w = image_rgb.shape[:2]
    
    # Calculate grid size
    n_masks = len(masks)
    cols = min(4, n_masks)  # Max 4 columns
    rows = (n_masks + cols - 1) // cols
    
    # Create a large canvas
    cell_size = 300
    canvas_w = cols * cell_size + (cols + 1) * 10  # 10px padding
    canvas_h = rows * cell_size + (rows + 1) * 10 + 50  # Extra space for title
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add title
    title = f"All SAM2 Segments: {n_masks} masks detected"
    cv2.putText(canvas, title, (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Sort masks by area (largest first)
    masks_sorted = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
    
    # Colors for masks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128)
    ]
    
    for idx, mask_dict in enumerate(masks_sorted):
        row = idx // cols
        col = idx % cols
        
        # Calculate position in canvas
        x_start = col * cell_size + (col + 1) * 10
        y_start = row * cell_size + (row + 1) * 10 + 50
        
        # Create individual mask visualization
        mask_viz = image_rgb.copy()
        mask = mask_dict['segmentation']
        
        # Apply colored overlay to mask area only
        mask_color = colors[idx % len(colors)]
        mask_viz[mask] = mask_viz[mask] * 0.3 + np.array(mask_color) * 0.7
        
        # Draw mask boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_viz, contours, -1, mask_color, 2)
        
        # Resize to cell size
        scale = min(cell_size / h, cell_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        mask_viz_resized = cv2.resize(mask_viz, (new_w, new_h))
        
        # Center in cell
        y_offset = (cell_size - new_h) // 2
        x_offset = (cell_size - new_w) // 2
        
        # Place in canvas
        canvas[y_start + y_offset:y_start + y_offset + new_h,
               x_start + x_offset:x_start + x_offset + new_w] = mask_viz_resized
        
        # Add mask info
        info_text = f"#{idx + 1}"
        area_pct = (mask_dict['area'] / (h * w)) * 100
        size_text = f"{area_pct:.1f}%"
        
        cv2.putText(canvas, info_text, (x_start + 10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, size_text, (x_start + 10, y_start + cell_size - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return canvas

# Helper functions for mask storage
def save_mask_images(image_path, image_rgb, masks):
    """Save mask images as PNG files for each clothing type in organized folders"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists in current dir, otherwise check parent
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        else:
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Create dedicated folder for this image's masks
        masks_dir = os.path.join(data_dir, 'saved_masks', base_name)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Save masks for each clothing type
        clothing_types = ['shirt', 'pants', 'shoes']
        
        for clothing_type in clothing_types:
            # Create visualization without labels
            mask_img, count = create_clothing_visualization(image_rgb, masks, clothing_type, for_gemini=True)
            
            if count > 0:
                # Save the masked image in the dedicated folder
                mask_filename = f"mask_{clothing_type}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                # Convert to PIL and save
                mask_pil = Image.fromarray(mask_img)
                mask_pil.save(mask_path)
                print(f"Saved {clothing_type} mask to {os.path.join(base_name, mask_filename)}")
        
        # Save the raw masks data in the same folder
        masks_pkl_file = os.path.join(masks_dir, "masks.pkl")
        serializable_masks = []
        for mask in masks:
            mask_copy = mask.copy()
            mask_copy['segmentation'] = mask_copy['segmentation'].tolist()
            serializable_masks.append(mask_copy)
            
        with open(masks_pkl_file, 'wb') as f:
            pickle.dump({
                'masks': serializable_masks,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }, f)
            
        print(f"Saved raw masks data to {os.path.join(base_name, 'masks.pkl')}")
        
    except Exception as e:
        print(f"Error saving masks: {str(e)}")

def load_masks_from_file(image_path):
    """Load masks from the new organized folder structure"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists in current dir, otherwise check parent
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), 'data')):
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        else:
            return None, None  # No data directory found
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Look in the dedicated folder for this image
        masks_dir = os.path.join(data_dir, 'saved_masks', base_name)
        masks_file = os.path.join(masks_dir, "masks.pkl")
        
        if os.path.exists(masks_file):
            with open(masks_file, 'rb') as f:
                data = pickle.load(f)
                
            # Convert lists back to numpy arrays
            masks = []
            for mask in data['masks']:
                mask['segmentation'] = np.array(mask['segmentation'], dtype=bool)
                masks.append(mask)
                
            return masks, data.get('timestamp')
        return None, None
    except Exception as e:
        print(f"Error loading masks: {str(e)}")
        return None, None

@app.route('/')
def home():
    return jsonify({"status": "Flask API is running"})

@app.route('/health')
def health():
    """Health check endpoint that also pre-warms the model"""
    # Load models if not already loaded
    load_classification_models()
    return jsonify({
        "status": "healthy",
        "models_loaded": processor is not None and model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/static/<filename>')
def serve_person_image(filename):
    """Serve person images"""
    try:
        print(f"Serving person image: {filename}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists in current dir, otherwise check parent
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), 'data')):
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        else:
            return jsonify({"error": "Data directory not found"}), 404
            
        image_dir = os.path.join(data_dir, 'sample_images', 'people')
        
        if not os.path.exists(image_dir):
            return jsonify({"error": "People images directory not found"}), 404
            
        if not os.path.exists(os.path.join(image_dir, filename)):
            return jsonify({"error": f"Image {filename} not found"}), 404
            
        response = send_from_directory(image_dir, filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/garments/<filename>')
def serve_garment_image(filename):
    """Serve garment images"""
    try:
        print(f"Serving garment image: {filename}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists in current dir, otherwise check parent
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), 'data')):
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        else:
            return jsonify({"error": "Data directory not found"}), 404
            
        image_dir = os.path.join(data_dir, 'sample_images', 'garments')
        
        if not os.path.exists(image_dir):
            return jsonify({"error": "Garments directory not found"}), 404
            
        if not os.path.exists(os.path.join(image_dir, filename)):
            return jsonify({"error": f"Garment {filename} not found"}), 404
            
        response = send_from_directory(image_dir, filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    global processor, model
    
    try:
        # Get request data
        data = request.json
        image_path = data.get('image_path')
        image_url = data.get('image_url')  # Firebase Storage URL
        image_data = data.get('image_data')  # Base64 data URL
        model_size = data.get('model_size', 'large')
        
        # Validate input
        if not image_path and not image_url and not image_data:
            return jsonify({'error': 'No image source provided'}), 400
        
        # Load and process image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if image_data:
            # Handle base64 data URL
            print("Processing base64 image data")
            try:
                # Remove data URL prefix
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Failed to decode base64 image'}), 500
                    
                # For Replicate, we need to save temporarily
                if model_size == 'replicate':
                    temp_path = os.path.join(base_dir, 'temp_upload_image.jpg')
                    cv2.imwrite(temp_path, image)
                    full_image_path = temp_path
                else:
                    full_image_path = None
                    
            except Exception as e:
                return jsonify({'error': f'Failed to process base64 image: {str(e)}'}), 500
        elif image_url:
            # Download from Firebase Storage URL
            print(f"Downloading image from Firebase Storage: {image_url}")
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Convert to numpy array
                nparr = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Failed to decode image from URL'}), 500
                    
                # For Replicate, we need to save temporarily
                if model_size == 'replicate':
                    temp_path = os.path.join(base_dir, 'temp_firebase_image.jpg')
                    cv2.imwrite(temp_path, image)
                    full_image_path = temp_path
                else:
                    full_image_path = None
                    
            except Exception as e:
                return jsonify({'error': f'Failed to download image: {str(e)}'}), 500
        else:
            # Local file path
            full_image_path = os.path.join(base_dir, image_path)
            print(f"Loading image from: {full_image_path}")
            
            if not os.path.exists(full_image_path):
                return jsonify({'error': f'Image not found: {full_image_path}'}), 404
                
            image = cv2.imread(full_image_path)
            if image is None:
                return jsonify({'error': f'Failed to read image: {full_image_path}'}), 500
                
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_rgb = image_rgb.copy()  # Keep original for mask adjustment
        
        # Apply person extraction if enabled
        person_bbox = None
        person_extraction_viz = None
        if PERSON_EXTRACTION_CONFIG.get('use_person_extraction', True):
            print("Extracting person from image using MediaPipe...")
            mediapipe_start = time.time()
            try:
                cropped_person, person_bbox, person_mask = extract_person_from_image(
                    image_rgb, 
                    padding_percent=PERSON_EXTRACTION_CONFIG.get('padding_percent', 10)
                )
                
                if cropped_person is not None:
                    mediapipe_time = time.time() - mediapipe_start
                    print(f"⏱️  MediaPipe extraction completed in: {mediapipe_time:.2f} seconds")
                    print(f"Person detected! Bbox: {person_bbox}")
                    
                    # Create visualization of person extraction
                    viz_img = original_image_rgb.copy()
                    # Draw bounding box
                    x, y, w, h = person_bbox
                    cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Apply mask overlay
                    if person_mask is not None:
                        mask_overlay = np.zeros_like(viz_img)
                        mask_overlay[person_mask > 0] = [0, 255, 0]  # Green overlay
                        viz_img = cv2.addWeighted(viz_img, 0.7, mask_overlay, 0.3, 0)
                    
                    # Add text
                    cv2.putText(viz_img, "MediaPipe Person Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(viz_img, f"Bbox: {person_bbox}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    person_extraction_viz = image_to_base64(viz_img)
                    
                    # Save cropped person for SAM2 processing
                    temp_person_path = os.path.join(base_dir, 'temp_person_cropped.jpg')
                    cv2.imwrite(temp_person_path, cv2.cvtColor(cropped_person, cv2.COLOR_RGB2BGR))
                    # Update paths and image for SAM2
                    full_image_path = temp_person_path
                    image_rgb = cropped_person
                    print(f"✅ MediaPipe detected person! Processing cropped region: {cropped_person.shape}")
                    print(f"✅ Cropped person saved to: {temp_person_path}")
                    print(f"✅ SAM2 will process ONLY the person region, not the full image")
                else:
                    mediapipe_time = time.time() - mediapipe_start
                    print(f"⏱️  MediaPipe extraction completed in: {mediapipe_time:.2f} seconds")
                    print("No person detected, processing full image")
                    # Create visualization showing no detection
                    viz_img = original_image_rgb.copy()
                    cv2.putText(viz_img, "MediaPipe: No Person Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    person_extraction_viz = image_to_base64(viz_img)
            except Exception as e:
                print(f"Person extraction failed: {str(e)}, processing full image")
                import traceback
                traceback.print_exc()
                # Create error visualization
                viz_img = original_image_rgb.copy()
                cv2.putText(viz_img, f"MediaPipe Error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                person_extraction_viz = image_to_base64(viz_img)
        
        # Use configured provider for SAM2
        try:
            # Ensure we have a saved file path for the API
            if not full_image_path:
                # Create a temporary file if we only have image data
                temp_path = os.path.join(base_dir, 'temp_process_image.jpg')
                cv2.imwrite(temp_path, image)
                full_image_path = temp_path
            
            # Pass whether image was cropped by MediaPipe
            is_person_cropped = person_bbox is not None
            
            # Choose provider based on configuration
            provider = SAM2_PROVIDER_CONFIG.get("provider", "replicate")
            print(f"Using SAM2 provider: {provider}")
            
            if provider == "fal":
                masks, sam2_time = process_with_fal(full_image_path, is_person_cropped)
                print(f"Generated {len(masks)} masks via FAL in {sam2_time:.2f} seconds")
            else:
                masks, sam2_time = process_with_replicate(full_image_path, is_person_cropped)
                print(f"Generated {len(masks)} masks via Replicate in {sam2_time:.2f} seconds")
        except Exception as e:
            provider_name = SAM2_PROVIDER_CONFIG.get("provider", "replicate").upper()
            print(f"{provider_name} API error: {str(e)}")
            return jsonify({'error': f'{provider_name} API failed: {str(e)}'}), 500
        
        # Adjust mask coordinates back to original image space if person was extracted
        if person_bbox is not None:
            print(f"Adjusting {len(masks)} masks back to original image coordinates...")
            x_offset, y_offset = person_bbox[0], person_bbox[1]
            
            for mask in masks:
                # Create a new full-size mask
                h_orig, w_orig = original_image_rgb.shape[:2]
                full_mask = np.zeros((h_orig, w_orig), dtype=bool)
                
                # Get the cropped mask dimensions
                h_crop, w_crop = mask['segmentation'].shape
                
                # Calculate the region where to place the mask
                y_start = y_offset
                y_end = min(y_offset + h_crop, h_orig)
                x_start = x_offset
                x_end = min(x_offset + w_crop, w_orig)
                
                # Place the cropped mask in the full image space
                full_mask[y_start:y_end, x_start:x_end] = mask['segmentation'][:y_end-y_start, :x_end-x_start]
                
                # Update mask data
                mask['segmentation'] = full_mask
                
                # Adjust bbox coordinates
                old_bbox = mask['bbox']
                mask['bbox'] = [
                    old_bbox[0] + x_offset,
                    old_bbox[1] + y_offset,
                    old_bbox[2] + x_offset,
                    old_bbox[3] + y_offset
                ]
                
                # Recalculate area
                mask['area'] = int(full_mask.sum())
            
            # Use original image for visualization
            image_rgb = original_image_rgb
        
        # Sort by area
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Print mask statistics for debugging
        if DEBUG_CONFIG.get("print_mask_stats", False) and masks:
            print("\n=== Mask Statistics ===")
            print(f"Total masks: {len(masks)}")
            
            # Analyze mask quality
            iou_scores = [m['predicted_iou'] for m in masks]
            areas = [m['area'] for m in masks]
            
            print(f"IoU scores: min={min(iou_scores):.2f}, max={max(iou_scores):.2f}, avg={np.mean(iou_scores):.2f}")
            print(f"Areas: min={min(areas)}, max={max(areas)}, avg={int(np.mean(areas))}")
            
            # Only show stability scores if available (not from Replicate)
            if masks[0].get('stability_score') is not None:
                stability_scores = [m['stability_score'] for m in masks]
                print(f"Stability: min={min(stability_scores):.2f}, max={max(stability_scores):.2f}, avg={np.mean(stability_scores):.2f}")
            
            # Count by size
            small = sum(1 for a in areas if a < 5000)
            medium = sum(1 for a in areas if 5000 <= a < 50000)
            large = sum(1 for a in areas if a >= 50000)
            print(f"Size distribution: small={small}, medium={medium}, large={large}")
            print("======================\n")
        
        # Filter out obvious background masks before refinement
        print(f"Filtering {len(masks)} raw masks...")
        filtered_pre = []
        image_area = image_rgb.shape[0] * image_rgb.shape[1]
        
        for mask_dict in masks:
            area_ratio = mask_dict['area'] / image_area
            
            # Skip if too large (likely background)
            if area_ratio > 0.8:
                print(f"  Skipping large mask with area ratio {area_ratio:.2f}")
                continue
                
            # Skip if mask touches all edges (likely background)
            mask = mask_dict['segmentation']
            if (mask[0, :].any() and mask[-1, :].any() and 
                mask[:, 0].any() and mask[:, -1].any()):
                print(f"  Skipping edge-touching mask")
                continue
            
            filtered_pre.append(mask_dict)
        
        masks = filtered_pre
        print(f"Pre-filtered to {len(masks)} potential clothing masks")
        
        # Save raw masks before refinement for visualization
        raw_masks_for_viz = [m.copy() for m in masks]
        
        # Refine masks with advanced edge-aware processing
        print("Refining mask edges with advanced processing...")
        refined_masks = []
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            
            # Convert boolean mask to uint8
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Remove small noise with connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            
            # Keep only the largest connected component (excluding background)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
                if len(areas) > 0:
                    largest_label = np.argmax(areas) + 1
                    mask_clean = (labels == largest_label).astype(np.uint8) * 255
                else:
                    mask_clean = mask_uint8
            else:
                mask_clean = mask_uint8
            
            # Apply bilateral filter for edge-preserving smoothing
            mask_bilateral = cv2.bilateralFilter(mask_clean, 9, 75, 75)
            
            # Morphological gradient for edge detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gradient = cv2.morphologyEx(mask_bilateral, cv2.MORPH_GRADIENT, kernel)
            
            # Combine original mask with gradient for better edges
            mask_enhanced = cv2.addWeighted(mask_bilateral, 0.8, gradient, 0.2, 0)
            
            # Final threshold with OTSU for optimal binarization
            _, mask_final = cv2.threshold(mask_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # One more closing operation to ensure solid masks
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_close)
            
            # Update the mask and recompute area
            new_mask = mask_final > 128
            new_area = int(new_mask.sum())
            
            # Only keep if area is reasonable
            if new_area > 100:  # Minimum area threshold
                mask_dict['segmentation'] = new_mask
                mask_dict['area'] = new_area
                refined_masks.append(mask_dict)
        
        masks = refined_masks
        print(f"Refined to {len(masks)} high-quality masks")
        
        # Define ALL labels like the working version
        all_labels = [
            # Upper body
            "shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "jacket", "coat", "vest",
            # Lower body
            "pants", "jeans", "trousers", "shorts", "skirt", "leggings",
            # Full body
            "dress", "suit", "jumpsuit",
            # Footwear
            "shoes", "sneakers", "boots", "sandals", "heels",
            # Accessories
            "hat", "cap", "sunglasses", "glasses", "watch", "belt", "tie", "scarf",
            "bag", "backpack", "purse",
            # Non-clothing (for filtering)
            "person", "face", "hair", "hand", "arm", "leg"
        ]
        
        # Clothing-only labels for filtering
        non_clothing = ["person", "face", "hair", "hand", "arm", "leg"]
        clothing_labels = [l for l in all_labels if l not in non_clothing]
        
        # Load CLIP for classification if not already loaded
        if processor is None or model is None:
            load_classification_models()
        
        # Classify each mask
        classification_model = "CLIP"  # Always using CLIP now
        print(f"Classifying {len(masks)} masks with {classification_model}...")
        print(f"Using batch processing for faster classification")
        start_classification = time.time()
        clothing_detections = {"shirt": 0, "pants": 0, "shoes": 0}
        
        # Prepare data for batch processing
        mask_images = []
        mask_metadata = []
        masks_to_classify = []
        
        # First pass: prepare images and filter out obvious non-clothing
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']
            
            # Skip very large masks based on config
            if mask_dict['area'] > CLASSIFICATION_CONFIG["max_background_ratio"] * image_rgb.shape[0] * image_rgb.shape[1]:
                mask_dict['label'] = 'background'
                mask_dict['confidence'] = 1.0
                mask_dict['skip_classification'] = True
                continue
            
            # Skip very small masks (likely noise)
            if mask_dict['area'] < 500:  # Too small to be clothing
                mask_dict['label'] = 'noise'
                mask_dict['confidence'] = 1.0
                mask_dict['skip_classification'] = True
                continue
            
            # Position hints
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                mask_dict['skip_classification'] = True
                continue
            
            center_y = y_indices.mean() / image_rgb.shape[0]
            aspect_ratio = (x_indices.max() - x_indices.min()) / (y_indices.max() - y_indices.min() + 1)
            
            # Create a proper isolated mask for classification
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Add padding around the crop
            pad = 20
            y_min_pad = max(0, y_min - pad)
            y_max_pad = min(image_rgb.shape[0], y_max + pad)
            x_min_pad = max(0, x_min - pad)
            x_max_pad = min(image_rgb.shape[1], x_max + pad)
            
            # Create RGBA image with transparent background for classification
            crop_height = y_max_pad - y_min_pad
            crop_width = x_max_pad - x_min_pad
            masked_rgba = np.zeros((crop_height, crop_width, 4), dtype=np.uint8)
            
            # Copy RGB values and set alpha where mask is True
            mask_crop = mask[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            masked_rgba[mask_crop, :3] = image_rgb[y_min_pad:y_max_pad, x_min_pad:x_max_pad][mask_crop]
            masked_rgba[mask_crop, 3] = 255  # Set alpha to opaque where mask is True
            
            # Convert to PIL - use RGB mode for classification models
            pil_img = Image.fromarray(masked_rgba[:, :, :3])  # Convert RGBA to RGB for models
            
            # Store for batch processing
            mask_images.append(pil_img)
            mask_metadata.append({
                'index': i,
                'center_y': center_y,
                'area_ratio': aspect_ratio,
                'mask_dict': mask_dict
            })
            masks_to_classify.append(i)
        
        # Batch classify all masks at once
        if mask_images:
            print(f"  Batch processing {len(mask_images)} masks...")
            batch_results = classify_batch_with_clip(mask_images, processor, model)
            
            # Apply results back to masks
            for j, result in enumerate(batch_results):
                i = masks_to_classify[j]
                mask_dict = masks[i]
                metadata = mask_metadata[j]
                
                detected_label = result['label']
                mask_dict['full_label'] = detected_label
                mask_dict['confidence'] = result['confidence']
                all_scores = result.get('all_scores', {})
            
                # For debug
                descriptive_hints = list(all_scores.keys())
                
                # Store debug info for CLIP
                mask_dict['debug_info'] = {
                    'prompts': descriptive_hints,
                    'all_scores': all_scores,
                    'position_y': metadata['center_y'],
                    'mask_area': mask_dict['area'],
                    'aspect_ratio': metadata['area_ratio'],
                    'model': 'CLIP'
                }
                
                # Add mask ID for tracking
                mask_dict['mask_id'] = i
                
                # Store the original detected label
                mask_dict['original_label'] = detected_label
                mask_dict['original_confidence'] = mask_dict['confidence']
                
                # Mark non-clothing items but don't skip them
                if detected_label in non_clothing:
                    mask_dict['label'] = 'non_clothing'
                    mask_dict['skip_viz'] = True  # Flag for visualization to skip
                # Map to simple categories for visualization
                elif detected_label in ["shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "jacket", "coat", "vest"]:
                    mask_dict['label'] = 'shirt'
                elif detected_label in ["pants", "jeans", "trousers", "shorts", "skirt", "leggings"]:
                    mask_dict['label'] = 'pants'
                elif detected_label in ["shoes", "sneakers", "boots", "sandals", "heels"]:
                    mask_dict['label'] = 'shoes'
                elif detected_label in ["dress", "suit", "jumpsuit"]:
                    mask_dict['label'] = 'dress'
                else:
                    # Accessories or other
                    mask_dict['label'] = detected_label
                
                # Count what we detected
                if mask_dict['label'] in ['shirt', 'pants', 'shoes']:
                    clothing_detections[mask_dict['label']] += 1
                
                # Add the input image to debug info
                if 'debug_info' in mask_dict:
                    mask_dict['debug_info']['input_image'] = image_to_base64(np.array(mask_images[j]))
        
        # Handle masks that were skipped
        for i, mask_dict in enumerate(masks):
            if 'label' not in mask_dict:
                mask_dict['label'] = 'unknown'
                mask_dict['confidence'] = 0.0
                mask_dict['skip_viz'] = True
        
        classification_time = time.time() - start_classification
        print(f"⏱️  CLIP batch classification completed in: {classification_time:.2f} seconds")
        print(f"   Processed {len(mask_images)} masks in a single batch")
        
        # Post-process to keep only best masks for each category
        print("\nPost-processing: Keeping best masks for each clothing type...")
        
        # Store ALL masks before filtering for debug
        # First mark all masks as skip_viz by default
        for mask in masks:
            if 'skip_viz' not in mask:
                mask['skip_viz'] = True
        
        # We'll update this after filtering
        process_image.last_all_masks = masks.copy()
        
        # Group masks by category
        category_masks = {
            'shirt': [],
            'pants': [],
            'shoes': [],
            'dress': [],
            'other': []
        }
        
        for mask in masks:
            label = mask.get('label', '')
            if label in category_masks:
                category_masks[label].append(mask)
            elif label not in ['non_clothing', 'background'] and not mask.get('skip_viz', False):
                category_masks['other'].append(mask)
        
        # Keep only the best mask for each category (except shoes)
        filtered_masks = []
        
        # For shirt, pants, dress - keep only the highest confidence one
        # Category-specific confidence thresholds
        MIN_CONFIDENCE = {
            'shirt': 0.20,  # 20% for shirts
            'pants': 0.05,  # 5% for pants/shorts (they often have low confidence)
            'dress': 0.15   # 15% for dresses
        }
        
        for category in ['shirt', 'pants', 'dress']:
            if category_masks[category]:
                # Sort by confidence and take the best one
                best_mask = max(category_masks[category], key=lambda x: x.get('confidence', 0))
                
                # Only include if confidence is high enough
                if best_mask.get('confidence', 0) >= MIN_CONFIDENCE.get(category, 0.15):
                    best_mask['skip_viz'] = False  # Ensure this mask is marked for visualization
                    filtered_masks.append(best_mask)
                    print(f"  Best {category}: {best_mask['full_label']} ({best_mask['confidence']:.1%})")
                else:
                    print(f"  Skipping low confidence {category}: {best_mask['full_label']} ({best_mask['confidence']:.1%})")
        
        # For shoes - keep up to 2 (left and right)
        if category_masks['shoes']:
            # Filter out low confidence shoes (likely floor/background)
            MIN_SHOE_CONFIDENCE = 0.30  # 30% minimum for shoes
            shoe_masks = [m for m in category_masks['shoes'] if m.get('confidence', 0) >= MIN_SHOE_CONFIDENCE]
            
            if shoe_masks:
                shoe_masks = sorted(shoe_masks, key=lambda x: x.get('confidence', 0), reverse=True)
                
                # Try to identify left and right shoes
                if len(shoe_masks) >= 2:
                    # Sort by x-position to get left and right
                    shoe_masks_by_x = sorted(shoe_masks[:4], key=lambda x: np.where(x['segmentation'])[1].mean())
                
                    # Take leftmost and rightmost with good confidence
                    left_shoe = shoe_masks_by_x[0]
                    right_shoe = shoe_masks_by_x[-1]
                    
                    # Make sure they're actually different shoes
                    left_x = np.where(left_shoe['segmentation'])[1].mean()
                    right_x = np.where(right_shoe['segmentation'])[1].mean()
                    
                    if abs(right_x - left_x) > image_rgb.shape[1] * 0.1:  # At least 10% image width apart
                        left_shoe['skip_viz'] = False
                        right_shoe['skip_viz'] = False
                        filtered_masks.extend([left_shoe, right_shoe])
                        print(f"  Left shoe: {left_shoe['full_label']} ({left_shoe['confidence']:.1%})")
                        print(f"  Right shoe: {right_shoe['full_label']} ({right_shoe['confidence']:.1%})")
                    else:
                        # Too close, just take the best one
                        shoe_masks[0]['skip_viz'] = False
                        filtered_masks.append(shoe_masks[0])
                        print(f"  Best shoe: {shoe_masks[0]['full_label']} ({shoe_masks[0]['confidence']:.1%})")
                else:
                    # Only one shoe detected
                    shoe_masks[0]['skip_viz'] = False
                    filtered_masks.append(shoe_masks[0])
                    print(f"  Single shoe: {shoe_masks[0]['full_label']} ({shoe_masks[0]['confidence']:.1%})")
        
        # Add non-clothing masks back (but marked as skip_viz)
        for mask in masks:
            if mask.get('label') == 'non_clothing' or mask.get('skip_viz', False):
                filtered_masks.append(mask)
        
        # Update masks and counts
        masks = filtered_masks
        clothing_detections = {
            'shirt': len([m for m in masks if m.get('label') == 'shirt']),
            'pants': len([m for m in masks if m.get('label') == 'pants']),
            'shoes': len([m for m in masks if m.get('label') == 'shoes'])
        }
        
        total_time = sam2_time + classification_time
        
        print(f"\nFinal Detection Summary:")
        print(f"  Shirts: {clothing_detections['shirt']}")
        print(f"  Pants: {clothing_detections['pants']}")
        print(f"  Shoes: {clothing_detections['shoes']}")
        
        # Store masks globally for Gemini and editing
        process_image.last_masks = masks
        process_image.last_image_path = image_path if image_path else "temp_firebase_image.jpg"
        process_image.last_image_rgb = image_rgb  # Store for mask editor
        
        # Update the stored all_masks with the final labels from filtered masks
        # This ensures that when we edit, we know which masks were selected
        for filtered_mask in filtered_masks:
            if filtered_mask.get('label') in ['shirt', 'pants', 'shoes'] and not filtered_mask.get('skip_viz', False):
                # Find this mask in the all_masks list and update its skip_viz
                for stored_mask in process_image.last_all_masks:
                    if abs(stored_mask['area'] - filtered_mask['area']) < 10 and \
                       abs(stored_mask['bbox'][0] - filtered_mask['bbox'][0]) < 5:
                        stored_mask['skip_viz'] = False
                        stored_mask['label'] = filtered_mask['label']
                        break
        
        # Save mask images permanently (only if we have a local path)
        if image_path:
            save_mask_images(image_path, image_rgb, masks)
        
        # Create RAW SAM2 masks visualization (before classification filtering)
        raw_sam2_img = create_raw_sam2_visualization(image_rgb, raw_masks_for_viz)
        
        # Create visualizations without labels (all in Gemini format)
        visualizations = generate_all_visualizations(image_rgb, masks)
        
        # Log timing breakdown
        provider = SAM2_PROVIDER_CONFIG.get("provider", "replicate")
        print(f"\n=== Timing Breakdown ===")
        print(f"{provider.upper()} API used for SAM2")
        print(f"SAM2 inference: {sam2_time:.2f}s")
        print(f"{classification_model} classification: {classification_time:.2f}s")
        print(f"Total processing: {total_time:.2f}s")
        print(f"=======================\n")
        
        # Prepare masks data with cropped images for editor
        # IMPORTANT: Use last_all_masks which has the updated labels
        all_masks_for_editor = process_image.last_all_masks
        masks_with_crops = []
        print(f"Preparing {len(all_masks_for_editor)} masks for editor (showing all detected masks)")
        
        # Debug: show what masks have which labels
        print("Mask labels in all_masks_for_editor:")
        for i, m in enumerate(all_masks_for_editor[:5]):  # Show first 5
            print(f"  Mask {i}: label={m.get('label', 'NONE')}, skip_viz={m.get('skip_viz', 'NONE')}, original_label={m.get('original_label', 'NONE')}")
        
        for idx, mask in enumerate(all_masks_for_editor):
            # Create cropped image of the mask area
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            x, y, w, h = [int(v) for v in bbox]
            
            # Create RGBA image with transparent background
            h_full, w_full = image_rgb.shape[:2]
            masked_rgba = np.zeros((h_full, w_full, 4), dtype=np.uint8)
            
            # Copy RGB values where mask is True
            masked_rgba[segmentation, :3] = image_rgb[segmentation]
            # Set alpha channel to 255 where mask is True (opaque)
            masked_rgba[segmentation, 3] = 255
            
            # Crop to bounding box
            cropped = masked_rgba[y:y+h, x:x+w]
            
            # Convert to base64 (using RGBA)
            pil_crop = Image.fromarray(cropped, mode='RGBA')
            buffer = BytesIO()
            pil_crop.save(buffer, format="PNG")
            crop_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Get the classification info directly from the mask
            # Since raw_masks_for_viz is just last_all_masks, it should have all the data
            label = mask.get('label', 'unknown')
            full_label = mask.get('full_label', mask.get('original_label', label))
            confidence = mask.get('confidence', mask.get('original_confidence', 0))
            original_label = mask.get('original_label', label)
            mask_id = mask.get('mask_id', idx)
            
            masks_with_crops.append({
                'label': label,
                'full_label': full_label,
                'confidence': confidence,
                'original_label': original_label,
                'original_confidence': mask.get('original_confidence', confidence),
                'mask_id': mask_id,
                'cropped_img': crop_base64,
                'bbox': bbox,
                'area': mask['area'],
                'index': idx,  # Track original index
                'skip_viz': mask.get('skip_viz', label == 'non_clothing')  # Track if it's currently shown
            })
        
        # Return results
        return jsonify({
            'sam2_time': sam2_time,
            'classification_time': classification_time,
            'total_time': total_time,
            **visualizations,  # Spread the visualization results
            'raw_sam2_img': image_to_base64(raw_sam2_img),
            'raw_masks_count': len(raw_masks_for_viz),
            'masks': masks_with_crops,  # Include for editor
            'person_extraction_viz': person_extraction_viz  # MediaPipe visualization
        })
    
    except Exception as e:
        import traceback
        print(f"Error in process_image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/quick-load-masks/<filename>', methods=['GET'])
def quick_load_masks(filename):
    """Quick load pre-generated masks for an image"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), 'data')):
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        else:
            return jsonify({'success': False, 'has_masks': False})
            
        base_name = filename.split('.')[0]
        
        # Look in the dedicated folder for this image
        masks_dir = os.path.join(data_dir, 'saved_masks', base_name)
        
        # Check if the folder exists
        if not os.path.exists(masks_dir):
            # No masks found for this image
            return jsonify({'success': False, 'has_masks': False})
        
        # Check if mask images exist in new location
        mask_files = {}
        for clothing_type in ['shirt', 'pants', 'shoes']:
            mask_filename = f"mask_{clothing_type}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            if os.path.exists(mask_path):
                # Read and convert to base64
                mask_files[clothing_type] = file_to_base64(mask_path)
        
        if mask_files:
            # Load the raw masks data
            masks_pkl = os.path.join(masks_dir, "masks.pkl")
            if os.path.exists(masks_pkl):
                with open(masks_pkl, 'rb') as f:
                    data = pickle.load(f)
                    masks = []
                    for mask in data['masks']:
                        mask['segmentation'] = np.array(mask['segmentation'], dtype=bool)
                        masks.append(mask)
                    
                    # Store masks globally
                    process_image.last_masks = masks
                    process_image.last_image_path = filename
            
            # Count items for each type
            return jsonify({
                'success': True,
                'has_masks': True,
                'all_items_img': mask_files.get('shirt', ''),  # Use shirt as default "all"
                'shirt_img': mask_files.get('shirt', ''),
                'pants_img': mask_files.get('pants', ''),
                'shoes_img': mask_files.get('shoes', ''),
                'all_items_count': len(mask_files),
                'shirt_count': 1 if 'shirt' in mask_files else 0,
                'pants_count': 1 if 'pants' in mask_files else 0,
                'shoes_count': 1 if 'shoes' in mask_files else 0,
                'from_saved': True
            })
        else:
            return jsonify({'success': False, 'has_masks': False})
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/load-saved-masks', methods=['POST'])
def load_saved_masks():
    """Load saved masks and generate visualizations"""
    global processor, model
    
    try:
        data = request.json
        image_path = data.get('image_path')
        model_size = data.get('model_size')
        
        # Extract just the filename
        filename = os.path.basename(image_path)
        
        # Load masks
        masks, timestamp = load_masks_from_file(filename, model_size)
        
        if masks is None:
            return jsonify({'error': 'No saved masks found'}), 404
        
        # Load original image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_image_path = os.path.join(base_dir, image_path)
        image = cv2.imread(full_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store masks globally for Gemini and editing
        process_image.last_masks = masks
        process_image.last_image_path = image_path
        process_image.last_image_rgb = image_rgb  # Store for mask editor
        
        # Create visualizations without labels (all in Gemini format)
        visualizations = generate_all_visualizations(image_rgb, masks)
        
        # Return results
        return jsonify({
            'sam2_time': 0,  # No processing time for loaded masks
            'classification_time': 0,
            'total_time': 0,
            'loaded_from_cache': True,
            'cache_timestamp': timestamp,
            **visualizations  # Spread the visualization results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/garments', methods=['GET'])
def get_garments():
    """Get list of available garment images"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), 'data')):
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        else:
            return jsonify({'garments': []})
            
        garments_dir = os.path.join(data_dir, 'sample_images', 'garments')
        
        garments = []
        if os.path.exists(garments_dir):
            for file in os.listdir(garments_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.avif')):
                    garments.append(file)
        
        return jsonify({'garments': sorted(garments)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get-all-masks-with-classes', methods=['GET'])
def get_all_masks_with_classes():
    """Get all masks with their classifications for editing"""
    try:
        if hasattr(process_image, 'last_all_masks') and process_image.last_all_masks:
            # Get the stored image
            if hasattr(process_image, 'last_image_rgb'):
                image_rgb = process_image.last_image_rgb
            else:
                # Load image if not stored
                base_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(base_dir, process_image.last_image_path) if hasattr(process_image, 'last_image_path') else None
                if full_path and os.path.exists(full_path):
                    image = cv2.imread(full_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    return jsonify({'masks': []})
            
            mask_data = []
            for i, mask in enumerate(process_image.last_all_masks):
                # Create individual mask image
                mask_seg = mask.get('segmentation')
                if mask_seg is None:
                    continue
                    
                # Extract the masked region
                y_indices, x_indices = np.where(mask_seg)
                if len(y_indices) == 0:
                    continue
                    
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                
                # Add padding
                pad = 20
                y_min_pad = max(0, y_min - pad)
                y_max_pad = min(image_rgb.shape[0], y_max + pad)
                x_min_pad = max(0, x_min - pad)
                x_max_pad = min(image_rgb.shape[1], x_max + pad)
                
                # Create RGBA image with transparent background
                crop_height = y_max_pad - y_min_pad
                crop_width = x_max_pad - x_min_pad
                masked_rgba = np.zeros((crop_height, crop_width, 4), dtype=np.uint8)
                
                # Copy RGB values and set alpha where mask is True
                mask_crop = mask_seg[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                masked_rgba[mask_crop, :3] = image_rgb[y_min_pad:y_max_pad, x_min_pad:x_max_pad][mask_crop]
                masked_rgba[mask_crop, 3] = 255  # Set alpha to opaque where mask is True
                
                # Convert to base64 (using RGBA)
                pil_img = Image.fromarray(masked_rgba, mode='RGBA')
                img_base64 = image_to_base64(np.array(pil_img))
                
                # Get classification info
                current_label = mask.get('label', 'unknown')
                confidence = mask.get('confidence', 0.0)
                
                # Get top predictions from debug info if available
                top_predictions = []
                if 'debug_info' in mask and 'all_scores' in mask['debug_info']:
                    scores = mask['debug_info']['all_scores']
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_predictions = [label for label, score in sorted_scores[:5] 
                                     if label not in ['person', 'face', 'background', 'rock', 'stone', 'ground']]
                
                mask_data.append({
                    'index': i,
                    'image': img_base64,
                    'current_label': current_label,
                    'confidence': confidence,
                    'top_predictions': top_predictions,
                    'area': mask.get('area', 0),
                    'position_y': mask.get('debug_info', {}).get('position_y', 0.5)
                })
            
            return jsonify({'masks': mask_data})
        
        return jsonify({'masks': []})
    except Exception as e:
        print(f"Error in get_all_masks_with_classes: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'masks': [], 'error': str(e)})


@app.route('/get-debug-info', methods=['GET'])
def get_debug_info():
    """Get debug info from last processing - ALL masks before filtering"""
    try:
        if hasattr(process_image, 'last_all_masks') and process_image.last_all_masks:
            debug_data = []
            
            # Create a set of kept mask IDs for faster lookup
            kept_mask_ids = set()
            if hasattr(process_image, 'last_masks'):
                for mask in process_image.last_masks:
                    # Use area and position as unique identifier
                    mask_id = (mask.get('area', 0), 
                              mask.get('bbox', [0,0,0,0])[0],
                              mask.get('bbox', [0,0,0,0])[1])
                    kept_mask_ids.add(mask_id)
            
            # Show ALL masks, not just filtered ones
            for i, mask in enumerate(process_image.last_all_masks):
                if 'debug_info' in mask:
                    # Check if this mask was kept
                    mask_id = (mask.get('area', 0),
                              mask.get('bbox', [0,0,0,0])[0], 
                              mask.get('bbox', [0,0,0,0])[1])
                    was_kept = mask_id in kept_mask_ids
                    
                    debug_data.append({
                        'mask_number': i + 1,
                        'label': mask.get('label', 'unknown'),
                        'full_label': mask.get('full_label', mask.get('label', 'unknown')),
                        'confidence': mask.get('confidence', 0),
                        'debug': mask['debug_info'],
                        'kept': was_kept
                    })
            return jsonify({'debug_data': debug_data})
        return jsonify({'debug_data': []})
    except Exception as e:
        print(f"Error in get_debug_info: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'debug_data': [], 'error': str(e)})

@app.route('/people', methods=['GET'])
def get_people():
    """Get list of available person images"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if data directory exists
        if os.path.exists(os.path.join(base_dir, 'data')):
            data_dir = os.path.join(base_dir, 'data')
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), 'data')):
            data_dir = os.path.join(os.path.dirname(base_dir), 'data')
        else:
            return jsonify({'people': []})
            
        people_dir = os.path.join(data_dir, 'sample_images', 'people')
        
        people = []
        if os.path.exists(people_dir):
            for file in os.listdir(people_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not '_mask_' in file:
                    people.append(file)
        
        return jsonify({'people': sorted(people)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update-mask-labels', methods=['POST'])
def update_mask_labels():
    """Update mask labels based on user selection and regenerate visualizations"""
    try:
        data = request.json
        mask_selections = data.get('mask_selections', {})
        
        if not hasattr(process_image, 'last_all_masks'):
            return jsonify({'error': 'No masks available'}), 404
            
        # Use ALL masks, not just filtered ones
        all_masks = process_image.last_all_masks
        image_rgb = process_image.last_image_rgb
        
        print(f"Updating mask labels - total masks: {len(all_masks)}, selections: {mask_selections}")
        
        # First, preserve the current good detections
        current_detections = {}
        for idx, mask in enumerate(all_masks):
            if mask.get('label') in ['shirt', 'pants', 'shoes'] and not mask.get('skip_viz', False):
                category = mask['label']
                if category not in current_detections:
                    current_detections[category] = []
                current_detections[category].append(idx)
        
        print(f"Current detections before update: {current_detections}")
        
        # Store original confidence scores before resetting
        original_scores = {}
        for idx, mask in enumerate(all_masks):
            original_scores[idx] = {
                'confidence': mask.get('confidence', 0),
                'full_label': mask.get('full_label', ''),
                'original_label': mask.get('original_label', mask.get('label', 'unknown'))
            }
        
        # Reset all labels first
        for mask in all_masks:
            mask['label'] = 'non_clothing'
            mask['skip_viz'] = True
        
        # Apply ALL selections (both old and new)
        for category, indices in mask_selections.items():
            print(f"  Category {category}: applying to indices {indices}")
            for idx in indices:
                if 0 <= idx < len(all_masks):
                    all_masks[idx]['label'] = category
                    all_masks[idx]['skip_viz'] = False
                    
                    # Check if this is a new manual selection or keeping existing
                    was_already_this_category = (idx in current_detections.get(category, []))
                    if not was_already_this_category:
                        # This is a manual override
                        all_masks[idx]['confidence'] = 1.0  # Manual selection = 100% confidence
                        all_masks[idx]['full_label'] = f'{category} (manual)'  # Mark as manually selected
                        all_masks[idx]['original_label'] = original_scores[idx]['original_label']
                    else:
                        # Keep original confidence and label
                        all_masks[idx]['confidence'] = original_scores[idx]['confidence']
                        all_masks[idx]['full_label'] = original_scores[idx]['full_label']
                    
                    print(f"    Set mask {idx} to {category} (confidence: {all_masks[idx]['confidence']:.1%})")
        
        # Filter masks to only include selected ones
        filtered_masks = [m for m in all_masks if not m.get('skip_viz', False)]
        print(f"Filtered to {len(filtered_masks)} masks for visualization")
        
        # Update the stored masks with manual edits
        process_image.last_all_masks = all_masks
        
        # Recreate visualizations
        visualizations = generate_all_visualizations(image_rgb, filtered_masks)
        
        return jsonify(visualizations)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Removed /user-clothing-items endpoint - handled client-side

@app.route('/login', methods=['POST'])
def login():
    """Handle user login with dummy authentication"""
    try:
        data = request.json
        username = data.get('username')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        # Dummy users list (matching the frontend)
        dummy_users = [
            'John Doe',
            'Jane Smith',
            'Test User',
            'Fashion Designer',
            'Demo Account'
        ]
        
        # Check if user exists
        if username not in dummy_users:
            return jsonify({'error': 'User not found'}), 404
        
        # For now, just return success with user info
        # In a real app, you'd create a session, generate a token, etc.
        user_info = {
            'username': username,
            'id': dummy_users.index(username) + 1,
            'role': 'designer' if 'Designer' in username else 'user',
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'user': user_info
        })
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

# Gemini routes
@app.route('/get-gemini-data', methods=['POST'])
def get_gemini_data():
    """Get image and mask data for Gemini processing"""
    data = request.json
    
    # Get the stored masks from the process_image function attributes
    last_masks = getattr(process_image, 'last_masks', None)
    last_image_rgb = getattr(process_image, 'last_image_rgb', None)
    
    response, status_code = gemini_service.get_gemini_data(
        data.get('image_path'),
        data.get('clothing_type'),
        last_masks,
        last_image_rgb
    )
    return jsonify(response), status_code

@app.route('/prepare-wardrobe-gemini', methods=['POST'])
def prepare_wardrobe_gemini():
    """Prepare wardrobe item for Gemini try-on using stored masks"""
    data = request.json
    response, status_code = gemini_service.prepare_wardrobe_gemini(
        data.get('image_url'),
        data.get('mask_data'),
        data.get('clothing_type')
    )
    return jsonify(response), status_code

@app.route('/gemini-tryon', methods=['POST'])
def gemini_tryon():
    """Perform virtual try-on using Gemini"""
    data = request.json
    response, status_code = gemini_service.gemini_tryon(
        data.get('person_image'),
        data.get('mask_image'),
        data.get('garment_file'),
        data.get('clothing_type', 'shirt')
    )
    return jsonify(response), status_code

def cleanup():
    """Clean up classification models from memory on shutdown"""
    global processor, model
    print("\nCleaning up classification models from memory...")
    
    if processor is not None:
        del processor
    if model is not None:
        del model
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("Memory cleanup complete!")

if __name__ == '__main__':
    import atexit
    import os
    atexit.register(cleanup)  # Register cleanup on exit
    
    # Get port from environment variable (for GCP) or use default
    port = int(os.environ.get('PORT', 5001))
    
    # Check if running in production (GCP)
    is_production = os.environ.get('GAE_ENV', '').startswith('standard') or os.environ.get('K_SERVICE')
    
    if is_production:
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(debug=True, port=port)