from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import sys
import numpy as np
import base64
from io import BytesIO
import io
from PIL import Image
from dotenv import load_dotenv
import requests
import json
import pickle
from datetime import datetime

# Load environment variables from .env file in parent directory
load_dotenv('../.env')

# Import Google GenerativeAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI package not installed. Run: pip install google-generativeai")

# Using API mode - no local model optimization needed
print(f"Running in API mode - using external APIs for ML processing")

import time

# Import configuration
from config_improved import DEBUG_CONFIG

# Import Gemini service
from services.gemini_service import GeminiService

# Import Style Assistant service
from services.assistantllm import StyleAssistantService

# Import OpenAI TryOn service
from services.openai_tryon_service import OpenAITryOnService

# Import LangChain Agent service
from services.langchain_agent_fixed import (
    handle_agent_chat, 
    clear_agent_session,
    get_or_create_agent
)

# Import Firebase service for centralized Firebase operations
from services.firebase_service import FirebaseService, bucket
from google.cloud import firestore
from services.fashion_trends_service import FashionTrendsService
from services.moodboard import TrendsService

# Import visualization and image processing services
from services.visualization_service import (
    image_to_base64,
    create_clothing_visualization,
    create_closet_visualization,
    create_person_only_image,
    generate_all_visualizations,
    create_raw_segformer_visualization
)
from services.image_processing_service import (
    file_to_base64,
    save_mask_images,
    load_masks_from_file
)

# Import Segformer service (replaces MediaPipe + SAM2 + CLIP)
# Check if we should use API or local model
USE_HF_API = os.getenv('USE_HF_API', 'false').lower() == 'true'

print("=" * 60)
print(f"🔧 SEGFORMER MODE: {'HUGGING FACE API' if USE_HF_API else 'LOCAL MODEL'}")
print(f"   USE_HF_API environment variable: {os.getenv('USE_HF_API', 'not set')}")
if USE_HF_API:
    print(f"   HF Token configured: {'Yes' if os.getenv('HUGGINGFACE_API_TOKEN') else 'No'}")
print("=" * 60)

# Delay imports until after checking USE_HF_API to avoid loading heavy dependencies
process_with_segformer = None
filter_best_clothing_items = None
MODEL_INFO = None
init_segmentation = None
cleanup_model = None

if USE_HF_API:
    print("📡 Using Hugging Face API for Segformer")
    # Import inside if block to prevent loading segformer_service.py
    from services.segformer_api_service import (
        process_with_segformer, 
        filter_best_clothing_items, 
        MODEL_INFO,
        init_segmentation,
        cleanup_model
    )
else:
    print("💻 Using local Segformer model")
    # This import won't happen in API mode, preventing transformers/torch loading
    # Commented out for Firebase deployment - uncomment if using local model
    # from services.segformer_service import (
    #     process_with_segformer, 
    #     filter_best_clothing_items, 
    #     MODEL_INFO,
    #     init_segmentation,
    #     cleanup_model
    # )
    raise NotImplementedError("Local model mode is disabled for Firebase deployment. Use USE_HF_API=true")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Gemini service
gemini_service = GeminiService()


@app.route('/')
def home():
    return jsonify({"status": "Flask API is running with Segformer"})

@app.route('/health')
def health():
    """Health check endpoint that also pre-warms the model"""
    # Pre-initialize Segformer model
    init_segmentation()
    return jsonify({
        "status": "healthy",
        "model_info": MODEL_INFO,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/static/<filename>')
def serve_person_image(filename):
    """Serve person images"""
    try:
        print(f"Serving person image: {filename}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        if not data_dir:
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

@app.route('/static/garments/<path:filename>')
def serve_garment_image(filename):
    """Serve garment images including from subfolders"""
    try:
        print(f"Serving garment image: {filename}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        if not data_dir:
            return jsonify({"error": "Data directory not found"}), 404
            
        image_dir = os.path.join(data_dir, 'sample_images', 'garments')
        
        if not os.path.exists(image_dir):
            return jsonify({"error": "Garments directory not found"}), 404
            
        # Handle subfolder paths
        full_path = os.path.join(image_dir, filename)
        if not os.path.exists(full_path):
            return jsonify({"error": f"Garment {filename} not found"}), 404
            
        # Get directory and filename separately for send_from_directory
        file_dir = os.path.dirname(full_path)
        file_name = os.path.basename(full_path)
        
        response = send_from_directory(file_dir, file_name)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Get request data
        data = request.json
        image_path = data.get('image_path')
        image_url = data.get('image_url')  # Firebase Storage URL
        image_data = data.get('image_data')  # Base64 data URL
        user_id = data.get('user_id')  # For mobile app
        save_to_firebase = data.get('save_to_firebase', False)  # Mobile app flag
        
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
                image_pil = Image.open(BytesIO(img_bytes))
                
                # Convert to RGB if needed
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                    
                # Save temporarily for processing
                temp_path = os.path.join(base_dir, 'temp_upload_image.jpg')
                image_pil.save(temp_path, 'JPEG')
                full_image_path = temp_path
                    
            except Exception as e:
                return jsonify({'error': f'Failed to process base64 image: {str(e)}'}), 500
        elif image_url:
            # Download from Firebase Storage URL
            print(f"Downloading image from Firebase Storage: {image_url}")
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Open image from URL response
                image_pil = Image.open(BytesIO(response.content))
                
                # Convert to RGB if needed
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                    
                # Save temporarily
                temp_path = os.path.join(base_dir, 'temp_firebase_image.jpg')
                image_pil.save(temp_path, 'JPEG')
                full_image_path = temp_path
                    
            except Exception as e:
                return jsonify({'error': f'Failed to download image: {str(e)}'}), 500
        else:
            # Local file path
            full_image_path = os.path.join(base_dir, image_path)
            print(f"Loading image from: {full_image_path}")
            
            if not os.path.exists(full_image_path):
                return jsonify({'error': f'Image not found: {full_image_path}'}), 404
                
            image_pil = Image.open(full_image_path)
            
            # Convert to RGB if needed
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
                
        # Get original dimensions
        original_w, original_h = image_pil.size
        
        # Resize image if too large (for efficiency)
        MAX_PROCESS_SIZE = 1024  # Maximum dimension for processing
        if max(original_h, original_w) > MAX_PROCESS_SIZE:
            # Calculate resize ratio
            ratio = MAX_PROCESS_SIZE / max(original_h, original_w)
            new_w = int(original_w * ratio)
            new_h = int(original_h * ratio)
            print(f"Resizing image from {original_w}x{original_h} to {new_w}x{new_h} for efficient processing")
            image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        # Convert to numpy array for processing (already in RGB)
        image_rgb = np.array(image_pil)
        
        # Store dimensions for response
        processed_h, processed_w = image_rgb.shape[:2]
        
        # Process with Segformer
        print("\n=== Processing with Segformer B2 Clothes Model ===")
        try:
            # Run Segformer segmentation
            all_masks, segmentation_time = process_with_segformer(image_rgb)
            print(f"Generated {len(all_masks)} masks in {segmentation_time:.2f} seconds")
            
            # Save raw masks for visualization
            raw_masks_for_viz = [m.copy() for m in all_masks]
            
            # Filter to keep best clothing items
            masks = filter_best_clothing_items(all_masks)
            print(f"Filtered to {len(masks)} clothing items")
            
        except Exception as e:
            print(f"Segformer error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500
        
        # Count detections
        clothing_detections = {
            'shirt': len([m for m in masks if m.get('label') == 'shirt' and not m.get('skip_viz', False)]),
            'pants': len([m for m in masks if m.get('label') == 'pants' and not m.get('skip_viz', False)]),
            'shoes': len([m for m in masks if m.get('label') == 'shoes' and not m.get('skip_viz', False)])
        }
        
        print(f"\nFinal Detection Summary:")
        print(f"  Shirts: {clothing_detections['shirt']}")
        print(f"  Pants: {clothing_detections['pants']}")
        print(f"  Shoes: {clothing_detections['shoes']}")
        
        # Debug: Check mask quality
        print(f"\nMask Quality Check:")
        for i, mask in enumerate(masks[:5]):  # Check first 5 masks
            mask_pixels = np.sum(mask['segmentation'])
            total_pixels = mask['segmentation'].size
            coverage = (mask_pixels / total_pixels) * 100
            print(f"  Mask {i} ({mask.get('label', 'unknown')}): {mask_pixels} pixels ({coverage:.1f}% of image)")
        
        # Store masks globally for Gemini and editing
        process_image.last_masks = masks
        process_image.last_all_masks = all_masks
        process_image.last_image_path = image_path if image_path else "temp_firebase_image.jpg"
        process_image.last_image_rgb = image_rgb
        
        # Save mask images permanently (only if we have a local path)
        if image_path:
            save_mask_images(image_path, image_rgb, masks)
        
        # Create RAW Segformer masks visualization
        raw_segformer_img = create_raw_segformer_visualization(image_rgb, raw_masks_for_viz)
        
        # Create visualizations
        visualizations = generate_all_visualizations(image_rgb, masks)
        
        # Prepare masks data with cropped images for editor
        masks_with_crops = []
        for idx, mask in enumerate(all_masks):
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
            
            masks_with_crops.append({
                'label': mask.get('label', 'unknown'),
                'full_label': mask.get('original_label', mask.get('label', 'unknown')),
                'confidence': mask.get('confidence', 1.0),
                'original_label': mask.get('original_label', mask.get('label', 'unknown')),
                'original_confidence': 1.0,
                'mask_id': idx,
                'cropped_img': crop_base64,
                'bbox': bbox,
                'area': mask['area'],
                'index': idx,
                'skip_viz': mask.get('skip_viz', False)
            })
        
        # Prepare response
        response_data = {
            'sam2_time': segmentation_time,  # Actually Segformer time
            'classification_time': 0,  # No separate classification needed
            'total_time': segmentation_time,
            **visualizations,
            'raw_segformer_img': image_to_base64(raw_segformer_img),
            'raw_masks_count': len(raw_masks_for_viz),
            'masks': masks_with_crops,  # Include for editor
            'person_extraction_viz': None,  # No MediaPipe needed
            'model_info': MODEL_INFO,
            'processing_info': {
                'original_dimensions': f"{original_w}x{original_h}",
                'processed_dimensions': f"{processed_w}x{processed_h}",
                'was_resized': original_w != processed_w or original_h != processed_h
            }
        }
        
        # Mobile should handle saving on client side like web does
        if save_to_firebase and user_id:
            print(f"WARNING: save_to_firebase flag is deprecated. Mobile should save on client side like web app does.")
        
        return jsonify(response_data)
    
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
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        if not data_dir:
            return jsonify({'success': False, 'has_masks': False})
            
        base_name = filename.split('.')[0]
        masks_dir = os.path.join(data_dir, 'saved_masks', base_name)
        
        if not os.path.exists(masks_dir):
            return jsonify({'success': False, 'has_masks': False})
        
        # Check if mask images exist
        mask_files = {}
        for clothing_type in ['shirt', 'pants', 'shoes']:
            mask_filename = f"mask_{clothing_type}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            if os.path.exists(mask_path):
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
            
            return jsonify({
                'success': True,
                'has_masks': True,
                'all_items_img': mask_files.get('shirt', ''),
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

@app.route('/garments', methods=['GET'])
def get_garments():
    """Get list of available garment images organized by type"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        print(f"Looking for garments in: {data_dir}")
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return jsonify({'garments': {}, 'flat_list': []})
            
        garments_dir = os.path.join(data_dir, 'sample_images', 'garments')
        print(f"Garments directory: {garments_dir}")
        
        garments_by_type = {}
        flat_list = []  # For backward compatibility
        
        if os.path.exists(garments_dir):
            print(f"Found garments directory, scanning folders...")
            
            # Check for category folders
            for category in os.listdir(garments_dir):
                category_path = os.path.join(garments_dir, category)
                if os.path.isdir(category_path) and category.upper() in ['SHIRT', 'PANT', 'SHOES', 'ACCESSORIES']:
                    category_key = 'pants' if category.upper() == 'PANT' else category.lower()
                    garments_by_type[category_key] = []
                    
                    print(f"  Scanning {category} folder...")
                    for file in os.listdir(category_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.avif')):
                            # Store with relative path from garments folder
                            relative_path = f"{category}/{file}"
                            garments_by_type[category_key].append(relative_path)
                            flat_list.append(relative_path)
                            print(f"    Found: {file}")
        else:
            print(f"Garments directory not found: {garments_dir}")
        
        print(f"Returning garments by type: {[(k, len(v)) for k, v in garments_by_type.items()]}")
        return jsonify({
            'garments': garments_by_type,
            'flat_list': sorted(flat_list)  # For backward compatibility
        })
    except Exception as e:
        print(f"Error in get_garments: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/people', methods=['GET'])
def get_people():
    """Get list of available person images"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        if not data_dir:
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
            
        all_masks = process_image.last_all_masks
        image_rgb = process_image.last_image_rgb
        
        print(f"Updating mask labels - total masks: {len(all_masks)}, selections: {mask_selections}")
        
        # Reset all labels first
        for mask in all_masks:
            mask['label'] = 'non_clothing'
            mask['skip_viz'] = True
        
        # Apply selections
        for category, indices in mask_selections.items():
            print(f"  Category {category}: applying to indices {indices}")
            for idx in indices:
                if 0 <= idx < len(all_masks):
                    all_masks[idx]['label'] = category
                    all_masks[idx]['skip_viz'] = False
                    all_masks[idx]['confidence'] = 1.0  # Manual selection
        
        # Filter masks to only include selected ones
        filtered_masks = [m for m in all_masks if not m.get('skip_viz', False)]
        
        # Update stored masks
        process_image.last_all_masks = all_masks
        
        # Recreate visualizations
        visualizations = generate_all_visualizations(image_rgb, filtered_masks)
        
        return jsonify(visualizations)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Handle user login with dummy authentication"""
    try:
        data = request.json
        username = data.get('username')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        # Map dummy users to actual Firebase UIDs from console
        # Using the first UID for each email from Firebase Console
        dummy_users = {
            'John Doe': 'ZtLk8Ct3XvXNeQepqnSME8g',  # john.doe@poshakai.test
            'Jane Smith': 'xZqtHBbe0hVLsoEaWOU58W',  # jane.smith@poshakai.test
            'Test User': '0PmexvZDQrPus4LvZoADDxL',  # test.user@poshakai.test
            'Fashion Designer': 'ATTDFkAwTdfYBl4V36vhRjhrqEd2',  # fashion.designer@poshakai.test (first UID)
            'Demo Account': 'omXo3iWdEeTCbCusQVIvIAeSaGG2'  # demo.account@poshakai.test
        }
        
        if username not in dummy_users:
            return jsonify({'error': 'User not found'}), 404
        
        user_info = {
            'username': username,
            'uid': dummy_users[username],  # Use consistent UID
            'id': list(dummy_users.keys()).index(username) + 1,
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

# Firebase endpoints for centralized operations
@app.route('/firebase/save-results', methods=['POST'])
def save_firebase_results():
    """Save processed results to Firebase - used by mobile and web"""
    print("\n=== SAVE-RESULTS ENDPOINT CALLED ===")
    try:
        data = request.json
        user_id = data.get('user_id')
        file_name = data.get('file_name')
        segmentation_results = data.get('segmentation_results')
        original_image = data.get('original_image')  # base64
        
        print(f"User ID: {user_id}")
        print(f"File name: {file_name}")
        print(f"Has segmentation results: {bool(segmentation_results)}")
        print(f"Has original image: {bool(original_image)}")
        
        if segmentation_results:
            print(f"Detected items counts:")
            print(f"  Shirts: {segmentation_results.get('shirt_count', 0)}")
            print(f"  Pants: {segmentation_results.get('pants_count', 0)}")
            print(f"  Shoes: {segmentation_results.get('shoes_count', 0)}")
        
        if not user_id or not segmentation_results:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Generate descriptions for detected clothing items
        try:
            # Get the original image (either from segmentation results or passed directly)
            image_for_description = original_image or segmentation_results.get('original_image')
            
            if image_for_description:
                # Build detected items dict from segmentation results
                detected_items = {}
                if segmentation_results.get('shirt_count', 0) > 0:
                    detected_items['shirt'] = ""
                if segmentation_results.get('pants_count', 0) > 0:
                    detected_items['pants'] = ""
                if segmentation_results.get('shoes_count', 0) > 0:
                    detected_items['shoes'] = ""
                
                if detected_items:
                    print(f"\n=== DESCRIPTION GENERATION DEBUG ===")
                    print(f"Detected items for description: {list(detected_items.keys())}")
                    print(f"Image data available: {bool(image_for_description)}")
                    print(f"Image data length: {len(image_for_description) if image_for_description else 0}")
                    
                    # Call Gemini to generate descriptions
                    descriptions, desc_status = gemini_service.generate_clothing_descriptions(
                        image_for_description,
                        detected_items
                    )
                    
                    print(f"Gemini response status: {desc_status}")
                    print(f"Descriptions returned: {descriptions}")
                    
                    if desc_status == 200:
                        # Add descriptions to segmentation results
                        segmentation_results['descriptions'] = descriptions
                        print(f"✅ Successfully added descriptions to segmentation_results")
                    else:
                        print(f"❌ Failed to generate descriptions: {descriptions}")
                        # Continue without descriptions if generation fails
                    print(f"=== END DESCRIPTION DEBUG ===\n")
        except Exception as desc_error:
            print(f"Error generating descriptions: {str(desc_error)}")
            # Continue without descriptions
        
        # Save to Firebase
        result = FirebaseService.save_processed_results(
            user_id,
            file_name or f"{int(datetime.now().timestamp() * 1000)}_upload.png",
            segmentation_results,
            original_image
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/firebase/images/<user_id>', methods=['GET'])
def get_firebase_images(user_id):
    """Get user images from Firebase"""
    try:
        result = FirebaseService.get_user_images(user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/firebase/mask-data/<user_id>/<image_name>', methods=['GET'])
def get_firebase_mask_data(user_id, image_name):
    """Get mask data for specific image"""
    try:
        result = FirebaseService.get_mask_data(user_id, image_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/firebase/copy-user-data', methods=['POST'])
def copy_user_data():
    """Copy all data from one user to another - useful for demo/testing"""
    try:
        data = request.json
        from_user_id = data.get('from_user_id')
        to_user_id = data.get('to_user_id')
        
        if not from_user_id or not to_user_id:
            return jsonify({'error': 'Both from_user_id and to_user_id are required'}), 400
        
        # Get all images from source user
        source_images = FirebaseService.get_user_images(from_user_id)
        if not source_images['success']:
            return jsonify({'error': 'Failed to get source user images'}), 500
        
        copied_count = 0
        for image in source_images['images']:
            image_name = image['name'].split('.')[0]
            
            # Get mask data
            mask_result = FirebaseService.get_mask_data(from_user_id, image_name)
            if mask_result['success'] and mask_result['data']:
                # Save to target user
                # Note: This is a simplified copy - in production you'd want to copy the actual image files too
                FirebaseService.save_mask_data(to_user_id, image_name, mask_result['data'])
                copied_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Copied {copied_count} items from {from_user_id} to {to_user_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/firebase/delete-image', methods=['POST'])
def delete_firebase_image():
    """Delete image and associated masks"""
    try:
        data = request.json
        user_id = data.get('user_id')
        image_path = data.get('image_path')
        
        if not user_id or not image_path:
            return jsonify({'error': 'Missing required data'}), 400
        
        result = FirebaseService.delete_user_image(user_id, image_path)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def fix_data_url_encoding(image_data):
    """Fix various forms of double-encoded data URLs"""
    if not image_data:
        return image_data
        
    # Handle different forms of double encoding
    if image_data.startswith('data:image/png;base64,data:'):
        # Fix double-encoded PNG data URL
        return 'data:image/png;base64,' + image_data.split('data:image/png;base64,')[2]
    elif image_data.startswith('data:image/jpeg;base64,data:') or image_data.startswith('data:image/jpg;base64,data:'):
        # Fix double-encoded JPEG data URL
        parts = image_data.split(',', 2)
        if len(parts) >= 3:
            return parts[0] + ',' + parts[2]
    elif image_data.startswith('data:image'):
        # Already has proper data URL prefix
        # Check if it's still double-encoded in a different way
        if ',data:image' in image_data:
            parts = image_data.split(',data:image')
            if len(parts) > 1:
                # Extract the actual base64 from the nested URL
                nested_parts = ('data:image' + parts[1]).split(',', 1)
                if len(nested_parts) == 2:
                    return 'data:image/png;base64,' + nested_parts[1]
        return image_data
    else:
        # Add data URL prefix if missing
        return f"data:image/png;base64,{image_data}"

@app.route('/test-closet-viz', methods=['GET'])
def test_closet_viz():
    """Test endpoint to verify closet visualizations are working"""
    try:
        # Use a test image
        import numpy as np
        
        # Create a simple test image (red shirt on white background)
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
        # Add a red "shirt" rectangle
        test_image[50:150, 70:130] = [255, 0, 0]  # Red color
        
        # Create a simple mask
        test_mask = {
            'segmentation': np.zeros((200, 200), dtype=bool),
            'label': 'shirt',
            'area': 5000,
            'bbox': [70, 50, 60, 100]
        }
        # Set the mask for the red rectangle
        test_mask['segmentation'][50:150, 70:130] = True
        
        # Generate closet visualization
        closet_result = create_closet_visualization(test_image, [test_mask], 'shirt')
        
        return jsonify({
            'success': True,
            'test_image_base64': closet_result['image'],
            'image_length': len(closet_result['image']),
            'content_bounds': closet_result['content_bounds']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/firebase/clothing-counts/<user_id>', methods=['GET'])
def get_clothing_counts(user_id):
    """Get just the counts of clothing items - lightweight for sync check"""
    try:
        # Get all user images
        images_result = FirebaseService.get_user_images(user_id)
        if not images_result['success']:
            return jsonify({'success': False, 'shirts': 0, 'pants': 0, 'shoes': 0})
        
        shirt_count = 0
        pants_count = 0
        shoes_count = 0
        
        # Process each image to count items
        for image in images_result['images']:
            image_name = image['name'].split('.')[0]
            
            # Get mask data
            mask_result = FirebaseService.get_mask_data(user_id, image_name)
            if not mask_result['success'] or not mask_result['data']:
                continue
                
            classifications = mask_result['data'].get('classifications', {})
            
            shirt_count += classifications.get('shirt', 0)
            pants_count += classifications.get('pants', 0)
            shoes_count += classifications.get('shoes', 0)
        
        return jsonify({
            'success': True,
            'shirts': shirt_count,
            'pants': pants_count,
            'shoes': shoes_count
        })
        
    except Exception as e:
        print(f"ERROR getting clothing counts: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'shirts': 0, 'pants': 0, 'shoes': 0}), 500

@app.route('/firebase/clothing-items/<user_id>', methods=['GET'])
def get_user_clothing_items(user_id):
    """Get all clothing items for user's closet - optimized endpoint"""
    try:
        print(f"\n=== DEBUG: Getting clothing items for user: {user_id} ===", flush=True)
        
        # Get all user images
        images_result = FirebaseService.get_user_images(user_id)
        print(f"  Found {len(images_result.get('images', []))} images for user", flush=True)
        if not images_result['success']:
            return jsonify({'success': False, 'shirts': [], 'pants': [], 'shoes': []})
        
        all_shirts = []
        all_pants = []
        all_shoes = []
        
        # Process each image
        for image in images_result['images']:
            image_name = image['name'].split('.')[0]
            
            # Get mask data
            mask_result = FirebaseService.get_mask_data(user_id, image_name)
            if not mask_result['success'] or not mask_result['data']:
                continue
                
            mask_data = mask_result['data']
            closet_visualizations = mask_data.get('closet_visualizations', {})
            visualizations = mask_data.get('visualizations', {})
            classifications = mask_data.get('classifications', {})
            metadata = mask_data.get('metadata', {})
            descriptions = mask_data.get('descriptions', {})
            
            # Process shirts
            if classifications.get('shirt', 0) > 0:
                shirt_image = closet_visualizations.get('shirt') or visualizations.get('shirt')
                if shirt_image:
                    # Fix any encoding issues
                    shirt_image = fix_data_url_encoding(shirt_image)
                    
                    all_shirts.append({
                        'id': f"{image_name}_shirt",
                        'image': shirt_image,
                        'type': 'shirt',
                        'source_image': image_name,
                        'isClosetViz': bool(closet_visualizations.get('shirt')),
                        'contentBounds': metadata.get('shirt'),
                        'description': descriptions.get('shirt', '')
                    })
            
            # Process pants
            if classifications.get('pants', 0) > 0:
                pants_image = closet_visualizations.get('pants') or visualizations.get('pants')
                if pants_image:
                    # Fix any encoding issues
                    pants_image = fix_data_url_encoding(pants_image)
                    
                    all_pants.append({
                        'id': f"{image_name}_pants",
                        'image': pants_image,
                        'type': 'pants',
                        'source_image': image_name,
                        'isClosetViz': bool(closet_visualizations.get('pants')),
                        'contentBounds': metadata.get('pants'),
                        'description': descriptions.get('pants', '')
                    })
            
            # Process shoes
            if classifications.get('shoes', 0) > 0:
                shoes_image = closet_visualizations.get('shoes') or visualizations.get('shoes')
                if shoes_image:
                    # Fix any encoding issues
                    shoes_image = fix_data_url_encoding(shoes_image)
                    
                    all_shoes.append({
                        'id': f"{image_name}_shoes",
                        'image': shoes_image,
                        'type': 'shoes',
                        'source_image': image_name,
                        'isClosetViz': bool(closet_visualizations.get('shoes')),
                        'contentBounds': metadata.get('shoes'),
                        'description': descriptions.get('shoes', '')
                    })
        
        print(f"\n=== DEBUG: Returning {len(all_shirts)} shirts, {len(all_pants)} pants, {len(all_shoes)} shoes ===", flush=True)
        
        return jsonify({
            'success': True,
            'shirts': all_shirts,
            'pants': all_pants,
            'shoes': all_shoes
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'shirts': [],
            'pants': [],
            'shoes': []
        })

@app.route('/firebase/clothing-items-stream/<user_id>', methods=['GET'])
def get_user_clothing_items_stream(user_id):
    """Stream clothing items as they're processed for progressive loading"""
    try:
        def generate():
            # Get all user images
            images_result = FirebaseService.get_user_images(user_id)
            if not images_result['success']:
                yield json.dumps({'type': 'error', 'message': 'Failed to load images'}) + '\n'
                return
            
            # Process each image and yield items as they're ready
            for image in images_result['images']:
                image_name = image['name'].split('.')[0]
                
                # Get mask data
                mask_result = FirebaseService.get_mask_data(user_id, image_name)
                if not mask_result['success'] or not mask_result['data']:
                    continue
                    
                mask_data = mask_result['data']
                closet_visualizations = mask_data.get('closet_visualizations', {})
                visualizations = mask_data.get('visualizations', {})
                classifications = mask_data.get('classifications', {})
                metadata = mask_data.get('metadata', {})
                descriptions = mask_data.get('descriptions', {})
                
                # Process and yield shirts
                if classifications.get('shirt', 0) > 0:
                    shirt_image = closet_visualizations.get('shirt') or visualizations.get('shirt')
                    if shirt_image:
                        shirt_image = fix_data_url_encoding(shirt_image)
                        yield json.dumps({
                            'type': 'item',
                            'category': 'shirt',
                            'data': {
                                'id': f"{image_name}_shirt",
                                'image': shirt_image,
                                'type': 'shirt',
                                'source_image': image_name,
                                'isClosetViz': bool(closet_visualizations.get('shirt')),
                                'contentBounds': metadata.get('shirt'),
                                'description': descriptions.get('shirt', '')
                            }
                        }) + '\n'
                
                # Process and yield pants
                if classifications.get('pants', 0) > 0:
                    pants_image = closet_visualizations.get('pants') or visualizations.get('pants')
                    if pants_image:
                        pants_image = fix_data_url_encoding(pants_image)
                        yield json.dumps({
                            'type': 'item',
                            'category': 'pants',
                            'data': {
                                'id': f"{image_name}_pants",
                                'image': pants_image,
                                'type': 'pants',
                                'source_image': image_name,
                                'isClosetViz': bool(closet_visualizations.get('pants')),
                                'contentBounds': metadata.get('pants'),
                                'description': descriptions.get('pants', '')
                            }
                        }) + '\n'
                
                # Process and yield shoes
                if classifications.get('shoes', 0) > 0:
                    shoes_image = closet_visualizations.get('shoes') or visualizations.get('shoes')
                    if shoes_image:
                        shoes_image = fix_data_url_encoding(shoes_image)
                        yield json.dumps({
                            'type': 'item',
                            'category': 'shoes',
                            'data': {
                                'id': f"{image_name}_shoes",
                                'image': shoes_image,
                                'type': 'shoes',
                                'source_image': image_name,
                                'isClosetViz': bool(closet_visualizations.get('shoes')),
                                'contentBounds': metadata.get('shoes'),
                                'description': descriptions.get('shoes', '')
                            }
                        }) + '\n'
            
            # Send completion signal
            yield json.dumps({'type': 'complete'}) + '\n'
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Virtual Closet endpoints
@app.route('/firebase/virtual-closet', methods=['POST'])
def save_virtual_closet_item():
    """Save virtual try-on result to Firebase Storage"""
    try:
        data = request.json
        user_id = data.get('userId')
        item = data.get('item')
        
        if not user_id or not item:
            return jsonify({'error': 'Missing userId or item data'}), 400
        
        # Store in Firebase Storage as JSON
        item_id = item.get('id', str(int(datetime.now().timestamp() * 1000)))
        blob_path = f"users/{user_id}/virtual-closet/{item_id}.json"
        blob = bucket.blob(blob_path)
        
        # Convert item to JSON
        json_data = json.dumps(item)
        blob.upload_from_string(json_data, content_type='application/json')
        
        return jsonify({
            'success': True,
            'id': item_id
        })
        
    except Exception as e:
        print(f"Error saving virtual closet item: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/firebase/virtual-closet/<user_id>', methods=['GET'])
def get_virtual_closet_items(user_id):
    """Get all virtual closet items for a user"""
    try:
        prefix = f"users/{user_id}/virtual-closet/"
        blobs = bucket.list_blobs(prefix=prefix)
        
        items = []
        for blob in blobs:
            # Skip directories
            if blob.name.endswith('/'):
                continue
            
            # Download and parse JSON
            json_data = blob.download_as_text()
            item = json.loads(json_data)
            
            # Ensure ID is set
            if 'id' not in item:
                item['id'] = os.path.basename(blob.name).replace('.json', '')
            
            items.append(item)
        
        # Sort by createdAt (newest first)
        items.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'items': items
        })
        
    except Exception as e:
        print(f"Error getting virtual closet items: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'items': []
        })

@app.route('/firebase/virtual-closet/<user_id>/<item_id>', methods=['DELETE'])
def delete_virtual_closet_item(user_id, item_id):
    """Delete a virtual closet item"""
    try:
        blob_path = f"users/{user_id}/virtual-closet/{item_id}.json"
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
            return jsonify({'error': 'Item not found'}), 404
        
        blob.delete()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error deleting virtual closet item: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Style Assistant routes
@app.route('/style-assistant/chat', methods=['POST'])
def style_assistant_chat():
    """Handle style assistant chat requests"""
    try:
        data = request.json
        user_id = data.get('user_id')
        query = data.get('query')
        clothing_items = data.get('clothing_items', {})
        conversation_history = data.get('conversation_history', [])
        
        print(f"\n=== STYLE ASSISTANT REQUEST ===")
        print(f"User ID: {user_id}")
        print(f"Query: {query}")
        print(f"Conversation history length: {len(conversation_history)}")
        print(f"Clothing items counts:")
        print(f"  Shirts: {len(clothing_items.get('shirts', []))}")
        print(f"  Pants: {len(clothing_items.get('pants', []))}")
        print(f"  Shoes: {len(clothing_items.get('shoes', []))}")
        
        if not user_id or not query:
            return jsonify({
                'success': False,
                'error': 'Missing user_id or query'
            }), 400
        
        # Initialize style assistant
        style_assistant = StyleAssistantService()
        
        # Get recommendation with conversation history
        result = style_assistant.get_outfit_recommendation(query, clothing_items, conversation_history)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in style assistant chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'recommendation': "I'm having trouble processing your request. Please try again.",
            'matched_items': []
        }), 500

# LangChain Agent routes
@app.route('/agent/chat', methods=['POST'])
def agent_chat():
    """Chat endpoint for LangChain style agent"""
    try:
        data = request.json
        result = handle_agent_chat(data)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        print(f"Error in agent chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/agent/clear-session/<session_id>', methods=['POST'])
def agent_clear_session(session_id):
    """Clear chat history for a specific session"""
    try:
        result = clear_agent_session(session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Gemini routes
@app.route('/get-gemini-data', methods=['POST'])
def get_gemini_data():
    """Get image and mask data for Gemini processing"""
    data = request.json
    
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
    print(f"\n=== /prepare-wardrobe-gemini called ===")
    print(f"Request from: {request.remote_addr}")
    print(f"Clothing type: {data.get('clothing_type')}")
    print(f"Has image_url: {bool(data.get('image_url'))}")
    print(f"Has mask_data: {bool(data.get('mask_data'))}")
    if data.get('mask_data'):
        print(f"Mask data keys: {list(data.get('mask_data', {}).keys())}")
    
    response, status_code = gemini_service.prepare_wardrobe_gemini(
        data.get('image_url'),
        data.get('mask_data'),
        data.get('clothing_type')
    )
    
    print(f"Response status: {status_code}")
    if status_code == 200:
        print(f"Success - image sizes in response: original={len(response.get('original_image', ''))}, mask={len(response.get('mask_image', ''))}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    return jsonify(response), status_code

@app.route('/gemini-tryon', methods=['POST'])
def gemini_tryon():
    """Perform virtual try-on using Gemini"""
    data = request.json
    print(f"\n=== /gemini-tryon called ===")
    print(f"Request from: {request.remote_addr}")
    print(f"Garment file: {data.get('garment_file')}")
    print(f"Clothing type: {data.get('clothing_type', 'shirt')}")
    print(f"Has person_image: {bool(data.get('person_image'))}")
    print(f"Has mask_image: {bool(data.get('mask_image'))}")
    if data.get('person_image'):
        print(f"Person image length: {len(data.get('person_image'))}")
    if data.get('mask_image'):
        print(f"Mask image length: {len(data.get('mask_image'))}")
    
    response, status_code = gemini_service.gemini_tryon(
        data.get('person_image'),
        data.get('mask_image'),
        data.get('garment_file'),
        data.get('clothing_type', 'shirt')
    )
    
    print(f"Response status: {status_code}")
    if status_code != 200:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    return jsonify(response), status_code

@app.route('/generate-clothing-descriptions', methods=['POST'])
def generate_clothing_descriptions():
    """Generate descriptions for detected clothing items using Gemini"""
    try:
        data = request.json
        image_base64 = data.get('image')
        detected_items = data.get('detected_items')
        
        if not image_base64 or not detected_items:
            return jsonify({'error': 'Missing image or detected_items'}), 400
        
        # Call Gemini service
        descriptions, status_code = gemini_service.generate_clothing_descriptions(
            image_base64,
            detected_items
        )
        
        return jsonify(descriptions), status_code
        
    except Exception as e:
        print(f"Error generating descriptions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/gemini-tryon-multiple', methods=['POST'])
def gemini_tryon_multiple():
    """Test endpoint for multi-item virtual try-on using Gemini"""
    try:
        data = request.json
        
        # Debug incoming request
        print(f"\n=== /gemini-tryon-multiple called ===")
        print(f"Request from: {request.remote_addr}")
        print(f"Clothing types: {data.get('clothing_types', [])}")
        print(f"Has person_image: {bool(data.get('person_image'))}")
        print(f"Mask images provided: {list(data.get('mask_images', {}).keys())}")
        print(f"Garment files provided: {list(data.get('garment_files', {}).keys())}")
        
        # Validate required fields
        if not data.get('person_image') or not data.get('mask_images') or not data.get('garment_files'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Call Gemini service
        result, status_code = gemini_service.gemini_tryon_multiple(
            person_image=data['person_image'],
            mask_images=data['mask_images'],
            garment_files=data['garment_files'],
            clothing_types=data.get('clothing_types', [])
        )
        
        print(f"Response status: {status_code}")
        if status_code != 200:
            print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"Successfully replaced: {result.get('items_replaced', [])}")
        
        return jsonify(result), status_code
    except Exception as e:
        print(f"[gemini-tryon-multiple] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/openai-tryon', methods=['POST'])
def openai_tryon():
    """Virtual try-on using OpenAI's image API"""
    try:
        data = request.json
        
        # Debug incoming request
        print(f"\n=== /openai-tryon called ===")
        print(f"Request from: {request.remote_addr}")
        print(f"Has person_image: {bool(data.get('person_image'))}")
        print(f"Garment files provided: {list(data.get('garment_files', {}).keys())}")
        
        # Validate required fields
        if not data.get('person_image') or not data.get('garment_files'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Initialize OpenAI service
        openai_service = OpenAITryOnService()
        
        # Process try-on
        result = openai_service.process_tryon(
            person_image_base64=data['person_image'],
            garment_images=data['garment_files']
        )
        
        if result['success']:
            print(f"OpenAI TryOn successful")
            return jsonify(result), 200
        else:
            print(f"OpenAI TryOn failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
            
    except Exception as e:
        print(f"[openai-tryon] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def cleanup():
    """Clean up models from memory on shutdown"""
    print("\nCleaning up Segformer model from memory...")
    cleanup_model()
    print("Memory cleanup complete!")

# Fashion Trends and Recommendations Routes
@app.route('/fashion-trends/<location>', methods=['GET'])
def get_fashion_trends(location):
    """Get fashion trends for a specific location"""
    print(f"\n[fashion-trends] Getting trends for location: {location}")
    
    try:
        context = request.args.get('context', 'home')
        
        # Use the new TrendsService from moodboard
        trends_service = TrendsService()
        trends_data = trends_service.get_trending_items(location, context)
        
        # Get time context from old service
        old_service = FashionTrendsService()
        time_context = old_service.get_current_time_context()
        
        # Combine the results
        result = {
            **trends_data,
            'time_context': time_context,
            'season': time_context['season']
        }
        
        if result.get('success'):
            # Cache in Firebase - disabled due to Firestore mode issue
            # firebase_service = FirebaseService()
            # cache_key = f"trends_{location}_{context}_{datetime.now().strftime('%Y%m%d')}"
            # firebase_service.save_trends_cache(cache_key, result)
            pass
            
        return jsonify(result)
    except Exception as e:
        print(f"[fashion-trends] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/shopping-recommendations', methods=['POST'])
def get_shopping_recommendations():
    """Get shopping recommendations based on location and needs"""
    print("\n[shopping-recommendations] Processing request")
    print(f"[shopping-recommendations] SERPER_API_KEY exists: {bool(os.getenv('SERPER_API_KEY'))}")
    print(f"[shopping-recommendations] SERPER_API_KEY length: {len(os.getenv('SERPER_API_KEY', ''))}")
    
    try:
        data = request.json
        location = data.get('location', 'New York')
        shopping_need = data.get('shopping_need', 'casual wear')
        budget = data.get('budget', 'medium')
        user_id = data.get('user_id')
        
        # Get user's wardrobe data if user_id provided
        wardrobe_data = None
        if user_id:
            firebase_service = FirebaseService()
            wardrobe_result = firebase_service.get_all_user_clothing_data(user_id)
            if wardrobe_result.get('success'):
                wardrobe_data = wardrobe_result
        
        trends_service = FashionTrendsService()
        result = trends_service.get_shopping_recommendations(
            location, shopping_need, budget, wardrobe_data
        )
        
        if result.get('success'):
            # Cache in Firebase - disabled due to Firestore mode issue
            # firebase_service = FirebaseService()
            # cache_key = f"recs_{location}_{shopping_need}_{datetime.now().strftime('%Y%m%d')}"
            # firebase_service.save_recommendations_cache(user_id, cache_key, result)
            pass
            
        return jsonify(result)
    except Exception as e:
        print(f"[shopping-recommendations] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/tribe-picks', methods=['POST'])
def get_tribe_picks():
    """Get fashion brand recommendations based on cultural preferences using Qloo API"""
    try:
        data = request.json
        music = data.get('music', 'Rock')
        movies = data.get('movies', 'Die Hard')
        location = data.get('location', 'New York')
        
        print(f"[Flask] Getting tribe picks for music: {music}, movies: {movies}, location: {location}")
        
        # Check if we have Qloo API key
        qloo_api_key = os.getenv('QLOO_API_KEY', '')
        if not qloo_api_key:
            print("[Flask] No QLOO API key configured")
            return jsonify({
                'success': False,
                'error': 'QLOO API key not configured',
                'brands': []
            }), 500
        
        # Use Qloo insights API with proper parameters
        headers = {'X-Api-Key': qloo_api_key}
        qloo_url = 'https://hackathon.api.qloo.com/v2/insights'
        
        # Build query parameters for RETAIL brands with cultural signals
        params = {
            'filter.type': 'urn:entity:brand',
            'filter.tags': 'urn:tag:genre:brand:retail',  # Retail stores where tribe shops
            'signal.location.query': location,
            'limit': 10,
            'sort': 'affinity'
        }
        
        # Add only one preference as interest tag (Qloo hackathon API limitation)
        # Testing shows the API returns same brands regardless of interests
        if music and music.lower() != 'not set':
            params['signal.interests.tags'] = music.lower()
        elif movies and movies.lower() != 'not set':
            params['signal.interests.tags'] = movies.lower()
        
        print(f"[Flask] Calling Qloo insights API")
        print(f"[Flask] Music: {music}, Movies: {movies}")
        print(f"[Flask] Query params: {params}")
        
        response = requests.get(qloo_url, headers=headers, params=params, timeout=10)
        print(f"[Flask] Qloo response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[Flask] Raw Qloo response: {json.dumps(data, indent=2)[:500]}...")  # First 500 chars
            
            # Check different possible response structures
            brands_data = []
            if 'results' in data and 'entities' in data['results']:
                brands_data = data['results']['entities']
            elif 'entities' in data:
                brands_data = data['entities']
            elif 'data' in data:
                brands_data = data['data']
            
            print(f"[Flask] Got {len(brands_data)} brands from Qloo")
            
            # Process brands with scores
            brands = []
            for i, brand in enumerate(brands_data[:10]):  # Try to get more brands
                brand_name = brand.get('name', '')
                # Use popularity or affinity score, scale to percentage
                popularity = brand.get('popularity', 0.8)
                score = round(popularity * 100, 1)
                
                print(f"[Flask] Brand {i+1}: {brand_name} (score: {score})")
                
                if brand_name:  # Only add if we have a name
                    brands.append({
                        'name': brand_name,
                        'score': score
                    })
                
                if len(brands) >= 4:  # Stop after 4 brands
                    break
            
            # Only return brands we actually got from Qloo
            
            return jsonify({
                'success': True,
                'description': f"{music} + {movies} fans shop at:",
                'brands': brands,
                'source': 'qloo'
            })
            
        else:
            print(f"[Flask] Qloo API returned {response.status_code}")
            try:
                error_data = response.json()
                print(f"[Flask] Qloo error response: {error_data}")
            except:
                print(f"[Flask] Qloo error text: {response.text}")
            
            return jsonify({
                'success': False,
                'error': f'QLOO API returned {response.status_code}',
                'brands': []
            }), 500
            
    except Exception as e:
        print(f"[Flask] Error in tribe picks: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'brands': []
        }), 500

@app.route('/search-products', methods=['POST'])
def search_products():
    """Simple product search using Serper API"""
    print("\n[search-products] Processing request")
    
    try:
        data = request.json
        query = data.get('query', '')
        num_results = data.get('num_results', 3)
        
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            return jsonify({
                'success': False,
                'error': 'Serper API key not configured',
                'products': []
            })
        
        # Direct Serper API call
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results,
            'gl': 'us'
        }
        
        print(f"[search-products] Searching for: {query}")
        response = requests.post('https://google.serper.dev/shopping', headers=headers, json=payload)
        
        if response.status_code == 200:
            results = response.json()
            products = []
            
            # Process all results to ensure brand diversity
            brand_count = {}
            max_per_brand = 2  # Maximum items per brand
            
            for item in results.get('shopping', []):
                title = item.get('title', '')
                # Use source as brand - it's the actual brand/retailer name
                brand = item.get('source', 'Unknown')
                
                # Track brand count
                if brand not in brand_count:
                    brand_count[brand] = 0
                
                # Only add if we haven't exceeded max per brand
                if brand_count[brand] < max_per_brand and len(products) < num_results:
                    brand_count[brand] += 1
                    
                    products.append({
                        'title': title,
                        'price': item.get('price', 'N/A'),
                        'image': item.get('imageUrl', ''),
                        'link': item.get('link', ''),
                        'source': brand,
                        'brand': brand,  # Use source as brand
                        'name': title  # alias for compatibility
                    })
                    
                    print(f"[search-products] Added: {title[:50]}... Brand: {brand}")
                    
                # Stop if we have enough products
                if len(products) >= num_results:
                    break
            
            print(f"[search-products] Found {len(products)} products")
            return jsonify({
                'success': True,
                'products': products
            })
        else:
            print(f"[search-products] Serper API error: {response.status_code}")
            return jsonify({
                'success': False,
                'error': f'Serper API error: {response.status_code}',
                'products': []
            })
            
    except Exception as e:
        print(f"[search-products] Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'products': []
        }), 500

@app.route('/moodboard/latest/<user_id>', methods=['GET'])
def get_latest_moodboard(user_id):
    """Get the latest saved moodboard from Firebase Storage"""
    print(f"\n[moodboard/latest] Getting latest moodboard for user {user_id}")
    
    try:
        # Try to get the latest moodboard from Storage
        blob_name = f"users/{user_id}/moodboard/latest.json"
        blob = bucket.blob(blob_name)
        
        if blob.exists():
            # Download as string
            json_data = blob.download_as_text()
            moodboard_data = json.loads(json_data)
            
            print(f"[moodboard/latest] Found latest moodboard from {moodboard_data.get('saved_at')}")
            
            return jsonify({
                'success': True,
                'moodboard': moodboard_data
            })
        else:
            print(f"[moodboard/latest] No saved moodboard found for user {user_id}")
            return jsonify({
                'success': False,
                'message': 'No saved moodboard found'
            })
            
    except Exception as e:
        print(f"[moodboard/latest] Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/gemini-analyze', methods=['POST', 'OPTIONS'])
def gemini_analyze():
    """Analyze data using Gemini"""
    if request.method == 'OPTIONS':
        return '', 200
    
    print("\n[gemini-analyze] Processing request")
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        print(f"[gemini-analyze] Prompt length: {len(prompt)} chars")
        print(f"[gemini-analyze] Prompt preview: {prompt[:200]}...")
            
        gemini_service = GeminiService()
        
        # Check if Gemini is initialized
        if not gemini_service.genai:
            print("[gemini-analyze] Gemini not initialized")
            return jsonify({'error': 'Gemini service not initialized'}), 500
        
        # Create a text-only model for analysis
        model = gemini_service.genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"[gemini-analyze] Gemini response: {response_text}")
        
        return jsonify({'success': True, 'response': response_text})
    except Exception as e:
        print(f"[gemini-analyze] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/moodboard/save', methods=['POST'])
def save_moodboard_data():
    """Save moodboard data to Firebase Storage as JSON"""
    print("\n[moodboard/save] Processing save request")
    
    try:
        data = request.json
        user_id = data.get('user_id')
        moodboard_data = data.get('data')
        
        if not user_id or not moodboard_data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Debug the data being saved
        print(f"[moodboard/save] Data keys: {list(moodboard_data.keys())}")
        if 'wardrobeGapProducts' in moodboard_data:
            print(f"[moodboard/save] Wardrobe gap products count: {len(moodboard_data.get('wardrobeGapProducts', []))}")
        
        # Add metadata
        moodboard_data['saved_at'] = datetime.now().isoformat()
        
        # Convert to JSON
        json_data = json.dumps(moodboard_data)
        
        # Save to Firebase Storage
        blob_name = f"users/{user_id}/moodboard/latest.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json_data, content_type='application/json')
        
        # Also save a timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_blob_name = f"users/{user_id}/moodboard/backup_{timestamp}.json"
        backup_blob = bucket.blob(backup_blob_name)
        backup_blob.upload_from_string(json_data, content_type='application/json')
        
        print(f"[moodboard/save] Saved moodboard data for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Moodboard saved successfully',
            'path': blob_name
        })
        
    except Exception as e:
        print(f"[moodboard/save] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import atexit
    import os
    atexit.register(cleanup)  # Register cleanup on exit
    
    # Get port from environment variable (for GCP) or use default
    port = int(os.environ.get('PORT', 5001))
    
    # Check if running in production (GCP)
    is_production = os.environ.get('GAE_ENV', '').startswith('standard') or os.environ.get('K_SERVICE')
    
    # Always bind to all interfaces for mobile app development
    app.run(host='0.0.0.0', port=port, debug=not is_production)