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

# Load environment variables from .env file
load_dotenv()

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

# Add parent directory to path for imports (for SAM2 imports)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoProcessor, AutoModel
import time

# Import configuration
from config_improved import get_active_sam2_config, MODEL_OVERRIDES, SIGLIP_CONFIG, CLASSIFICATION_CONFIG, PERFORMANCE_CONFIG, DEBUG_CONFIG
from improved_siglip_classification import classify_with_improved_prompts, classify_with_position_hints
from fashion_siglip_classifier import classify_with_fashion_siglip, FashionSigLIPClassifier
from clip_classifier import classify_with_clip

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global models
mask_gen = None
processor = None
model = None

def load_models(model_size='large'):
    global mask_gen, processor, model
    
    # Model configurations
    model_configs = {
        # SAM 2 models
        'tiny': ('sam2_hiera_t.yaml', 'checkpoints/sam2_hiera_tiny.pt'),
        'small': ('sam2_hiera_s.yaml', 'checkpoints/sam2_hiera_small.pt'),
        'base': ('sam2_hiera_b+.yaml', 'checkpoints/sam2_hiera_base_plus.pt'),
        'large': ('sam2_hiera_l.yaml', 'checkpoints/sam2_hiera_large.pt'),
        # SAM 2.1 models - configs are in configs/sam2.1/ subfolder
        'tiny_v2.1': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'checkpoints/sam2.1_hiera_tiny.pt'),
        'small_v2.1': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'checkpoints/sam2.1_hiera_small.pt'),
        'base_v2.1': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'checkpoints/sam2.1_hiera_base_plus.pt'),
        'large_v2.1': ('configs/sam2.1/sam2.1_hiera_l.yaml', 'checkpoints/sam2.1_hiera_large.pt')
    }
    
    if mask_gen is None or getattr(mask_gen, 'model_size', None) != model_size:
        # Clear old model from memory if switching
        if mask_gen is not None:
            print(f"Clearing previous model from memory...")
            del mask_gen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # Force memory cleanup on MPS (Apple Silicon)
                import gc
                gc.collect()
                torch.mps.empty_cache()
        
        print(f"Loading SAM2 {model_size.upper()} model...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get config and checkpoint paths
        config_file, checkpoint_file = model_configs.get(model_size, model_configs['large'])
        
        # Change to parent directory to match working demo
        original_cwd = os.getcwd()
        sam2_dir = os.path.dirname(base_dir)  # Go up one level to find SAM2
        os.chdir(sam2_dir)
        
        try:
            # Load the selected model with optimizations
            sam2 = build_sam2(config_file, checkpoint_file, device="mps", apply_postprocessing=True)
            
            # Enable optimizations for M4
            sam2.eval()  # Set to evaluation mode
            
            # Get active configuration (Facebook's parameters)
            config = get_active_sam2_config()
            
            # Apply model-specific overrides if any
            model_key = model_size.replace('_v2.1', '').replace('large', 'large')
            if model_key in MODEL_OVERRIDES:
                config.update(MODEL_OVERRIDES[model_key])
            
            # Create mask generator with all Facebook parameters
            mask_gen = SAM2AutomaticMaskGenerator(
                sam2,
                points_per_side=config["points_per_side"],
                points_per_batch=config["points_per_batch"],
                pred_iou_thresh=config["pred_iou_thresh"],
                stability_score_thresh=config["stability_score_thresh"],
                stability_score_offset=config.get("stability_score_offset", 1.0),  # NEW!
                mask_threshold=0.0,  # Default
                box_nms_thresh=config.get("box_nms_thresh", 0.7),  # NEW! Critical for deduplication
                crop_n_layers=config.get("crop_n_layers", 0),
                crop_nms_thresh=0.7,  # Default
                crop_overlap_ratio=512/1500,  # Default
                crop_n_points_downscale_factor=config.get("crop_n_points_downscale_factor", 1),
                min_mask_region_area=config.get("min_mask_region_area", 0),
                use_m2m=config.get("use_m2m", False),  # NEW! Mask refinement
                multimask_output=True,  # Generate multiple masks per point
            )
            
            print(f"SAM2 configured with {config['points_per_side']}x{config['points_per_side']} = {config['points_per_side']**2} points")
        finally:
            # Change back to original directory
            os.chdir(original_cwd)
            
        mask_gen.model_size = model_size  # Store which model is loaded
        
        print("Loading SigLIP...")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast=True)
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        model.eval()
        
        # Move SigLIP to MPS for GPU acceleration
        if torch.backends.mps.is_available():
            model = model.to("mps")
            print("SigLIP moved to MPS for GPU acceleration")

def image_to_base64(image):
    """Convert numpy array to base64 string"""
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def process_with_replicate(image_path):
    """Process image using Replicate API"""
    print("Processing with Replicate API...")
    
    # Check if Replicate API token is set
    if not os.environ.get('REPLICATE_API_TOKEN'):
        raise Exception("REPLICATE_API_TOKEN environment variable not set. Get your token from https://replicate.com/account/api-tokens")
    
    # Convert image to base64 for upload
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
        img_uri = f"data:image/png;base64,{img_data}"
    
    try:
        # Get active configuration to use with Replicate
        config = get_active_sam2_config()
        
        # Run SAM-2 on Replicate with our config
        start_time = time.time()
        output = replicate.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={
                "image": img_uri,
                "use_m2m": config.get("use_m2m", True),
                "points_per_side": config.get("points_per_side", 32),
                "pred_iou_thresh": config.get("pred_iou_thresh", 0.88),
                "stability_score_thresh": config.get("stability_score_thresh", 0.95)
            }
        )
        sam2_time = time.time() - start_time
        print(f"Replicate SAM2 completed in {sam2_time:.2f} seconds")
        
        # Process the output masks
        masks = []
        
        # Debug: print output structure
        print(f"Replicate output type: {type(output)}")
        if isinstance(output, dict):
            print(f"Output keys: {output.keys()}")
        
        # Handle different output formats from Replicate
        if output:
            # Check if output is a direct URL (string)
            if isinstance(output, str):
                # Single mask URL
                mask_urls = [output]
            elif isinstance(output, list):
                # List of mask URLs
                mask_urls = output
            elif isinstance(output, dict):
                # Dictionary with masks
                if 'masks' in output:
                    mask_urls = output['masks']
                elif 'individual_masks' in output:
                    mask_urls = output['individual_masks']
                else:
                    print(f"Unexpected output format: {output}")
                    mask_urls = []
            else:
                mask_urls = []
            
            # Process each mask URL
            for i, mask_url in enumerate(mask_urls):
                try:
                    # Download mask image
                    response = requests.get(mask_url)
                    mask_img = Image.open(BytesIO(response.content))
                    mask_array = np.array(mask_img.convert('L'))
                    
                    # Convert to boolean mask
                    mask_bool = mask_array > 128
                    
                    # Calculate area and other properties
                    y_indices, x_indices = np.where(mask_bool)
                    if len(y_indices) > 0:
                        masks.append({
                            'segmentation': mask_bool,
                            'area': int(mask_bool.sum()),
                            'bbox': [int(x_indices.min()), int(y_indices.min()), 
                                    int(x_indices.max()), int(y_indices.max())],
                            'predicted_iou': 0.9  # Default value
                        })
                except Exception as mask_error:
                    print(f"Error processing mask {i}: {str(mask_error)}")
                    continue
        
        print(f"Processed {len(masks)} masks from Replicate")
        return masks, sam2_time
        
    except Exception as e:
        print(f"Replicate API error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

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
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Create dedicated folder for this image's masks
        masks_dir = os.path.join(base_dir, 'data', 'saved_masks', base_name)
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
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Look in the dedicated folder for this image
        masks_dir = os.path.join(base_dir, 'data', 'saved_masks', base_name)
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

@app.route('/static/<filename>')
def serve_person_image(filename):
    """Serve person images"""
    try:
        print(f"Serving person image: {filename}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, 'data', 'sample_images', 'people')
        
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
        image_dir = os.path.join(base_dir, 'data', 'sample_images', 'garments')
        
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
        
        # Check if using Replicate API
        if model_size == 'replicate':
            try:
                masks, sam2_time = process_with_replicate(full_image_path)
                print(f"Generated {len(masks)} masks via Replicate in {sam2_time:.2f} seconds")
            except Exception as e:
                print(f"Replicate API error: {str(e)}")
                return jsonify({'error': f'Replicate API failed: {str(e)}'}), 500
        else:
            # Load models if not loaded or if model size changed
            model_load_time = 0
            if mask_gen is None or getattr(mask_gen, 'model_size', None) != model_size:
                print(f"Loading models for {model_size}...")
                start_load = time.time()
                try:
                    load_models(model_size)
                except Exception as e:
                    print(f"Model loading error: {str(e)}")
                    return jsonify({'error': f'Model loading failed: {str(e)}'}), 500
                model_load_time = time.time() - start_load
                print(f"Model loading took {model_load_time:.2f} seconds")
            
            # Generate masks with SAM2
            print(f"Generating masks with {model_size} model...")
            start_sam2 = time.time()
            
            # First attempt with original image
            masks = mask_gen.generate(image_rgb)
            
            # If we get very few masks, try different strategies
            if len(masks) < 5:
                print(f"Only {len(masks)} masks found. Trying alternative approaches...")
                
                # Strategy 1: Try with enhanced image
                lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                image_enhanced = cv2.merge([l, a, b])
                image_enhanced = cv2.cvtColor(image_enhanced, cv2.COLOR_LAB2RGB)
                
                masks_enhanced = mask_gen.generate(image_enhanced)
                
                if len(masks_enhanced) > len(masks):
                    masks = masks_enhanced
                    print(f"Enhanced image gave {len(masks)} masks")
                
                # Strategy 2: If still few masks, temporarily adjust thresholds
                if len(masks) < 5:
                    print("Still too few masks. Temporarily lowering thresholds...")
                    
                    # Save original thresholds
                    orig_iou = mask_gen.predictor.model.mask_threshold
                    orig_stability = getattr(mask_gen, 'stability_score_thresh', 0.85)
                    
                    # Lower thresholds temporarily
                    mask_gen.predictor.model.mask_threshold = 0.7
                    mask_gen.pred_iou_thresh = 0.7
                    mask_gen.stability_score_thresh = 0.8
                    
                    # Generate with lower thresholds
                    masks_lowthresh = mask_gen.generate(image_rgb)
                    
                    # Restore original thresholds
                    mask_gen.predictor.model.mask_threshold = orig_iou
                    config = get_active_sam2_config()
                    mask_gen.pred_iou_thresh = config["pred_iou_thresh"]
                    mask_gen.stability_score_thresh = orig_stability
                    
                    if len(masks_lowthresh) > len(masks):
                        masks = masks_lowthresh
                        print(f"Lower thresholds gave {len(masks)} masks")
            
            sam2_time = time.time() - start_sam2
            print(f"Generated {len(masks)} masks in {sam2_time:.2f} seconds")
            
            # ADAPTIVE RETRY WITH FACEBOOK'S DENSE CONFIG
            if PERFORMANCE_CONFIG.get("adaptive_quality", True) and len(masks) < 5:
                print(f"Low mask count ({len(masks)}), trying Facebook's dense configuration...")
                
                # Save current config
                original_params = {
                    'points_per_side': mask_gen.points_per_side,
                    'pred_iou_thresh': mask_gen.pred_iou_thresh,
                    'stability_score_thresh': mask_gen.stability_score_thresh,
                    'crop_n_layers': mask_gen.crop_n_layers,
                    'min_mask_region_area': mask_gen.min_mask_region_area,
                    'use_m2m': getattr(mask_gen, 'use_m2m', False)
                }
                
                # Apply Facebook's dense configuration
                mask_gen.points_per_side = 64  # FB dense: 64x64 = 4096 points!
                mask_gen.points_per_batch = 128
                mask_gen.pred_iou_thresh = 0.7
                mask_gen.stability_score_thresh = 0.92
                mask_gen.stability_score_offset = 0.7
                mask_gen.crop_n_layers = 1
                mask_gen.min_mask_region_area = 25
                if hasattr(mask_gen, 'use_m2m'):
                    mask_gen.use_m2m = True
                
                # Retry generation
                retry_start = time.time()
                retry_masks = mask_gen.generate(image_rgb)
                retry_time = time.time() - retry_start
                
                print(f"Dense config gave {len(retry_masks)} masks in {retry_time:.2f}s")
                
                if len(retry_masks) > len(masks):
                    masks = retry_masks
                    sam2_time += retry_time
                
                # Restore original settings
                for key, value in original_params.items():
                    if hasattr(mask_gen, key):
                        setattr(mask_gen, key, value)
        
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
        
        # Load CLIP/SigLIP if using Replicate (since we need it for classification)
        if model_size == 'replicate' and (processor is None or model is None):
            USE_CLIP = DEBUG_CONFIG.get("use_clip_instead", False)
            
            if USE_CLIP:
                print("Loading CLIP for classification (fast mode)...")
                from transformers import CLIPProcessor, CLIPModel
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
                model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                model.eval()
                
                # Move CLIP to MPS for GPU acceleration
                if torch.backends.mps.is_available():
                    model = model.to("mps")
                    print("CLIP moved to MPS for GPU acceleration")
            else:
                print("Loading SigLIP for classification...")
                processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast=True)
                model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
                model.eval()
                
                # Move SigLIP to MPS for GPU acceleration
                if torch.backends.mps.is_available():
                    model = model.to("mps")
                    print("SigLIP moved to MPS for GPU acceleration")
        
        # Classify each mask
        classification_model = "CLIP" if DEBUG_CONFIG.get("use_clip_instead", False) else "SigLIP"
        print(f"Classifying {len(masks)} masks with {classification_model}...")
        print(f"Using labels: {all_labels[:10]}... (showing first 10 of {len(all_labels)})")
        start_siglip = time.time()
        clothing_detections = {"shirt": 0, "pants": 0, "shoes": 0}
        
        for i, mask_dict in enumerate(masks):
            if i % 10 == 0:
                print(f"  Processing mask {i}/{len(masks)}...")
            mask = mask_dict['segmentation']
            
            # Skip very large masks based on config
            if mask_dict['area'] > CLASSIFICATION_CONFIG["max_background_ratio"] * image_rgb.shape[0] * image_rgb.shape[1]:
                mask_dict['label'] = 'background'
                mask_dict['confidence'] = 1.0
                continue
            
            # Position hints
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
            center_y = y_indices.mean() / image_rgb.shape[0]
            
            # Check if this is a SAM 2.1 model (more sensitive, smaller segments)
            # Replicate uses its own optimized model, not v2.1
            is_v21 = 'v2.1' in model_size and model_size != 'replicate'
            
            # Adjust thresholds for SAM 2.1
            if is_v21:
                # SAM 2.1 produces smaller segments, adjust thresholds
                shirt_area_min = 10000  # Lower threshold
                shirt_area_max = 50000
                pants_area_min = 10000
                pants_area_max = 35000
                shoe_area_max = 8000  # Higher for SAM 2.1
                shoe_area_min = 300
            else:
                # Original thresholds for SAM 2.0
                shirt_area_min = 20000
                shirt_area_max = 60000
                pants_area_min = 20000
                pants_area_max = 40000
                shoe_area_max = 5000
                shoe_area_min = 500
            
            # Better classification logic
            aspect_ratio = (x_indices.max() - x_indices.min()) / (y_indices.max() - y_indices.min() + 1)
            
            # Use ALL labels for classification (like the working version)
            descriptive_hints = all_labels  # All 40+ options
            
            # Create a proper isolated mask for classification
            # Use the ORIGINAL image, not enhanced
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Add padding around the crop
            pad = 20
            y_min_pad = max(0, y_min - pad)
            y_max_pad = min(image_rgb.shape[0], y_max + pad)
            x_min_pad = max(0, x_min - pad)
            x_max_pad = min(image_rgb.shape[1], x_max + pad)
            
            # Create a white background image
            crop_height = y_max_pad - y_min_pad
            crop_width = x_max_pad - x_min_pad
            masked_crop = np.ones((crop_height, crop_width, 3), dtype=np.uint8) * 255
            
            # Copy only the masked pixels from the original image
            mask_crop = mask[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            masked_crop[mask_crop] = image_rgb[y_min_pad:y_max_pad, x_min_pad:x_max_pad][mask_crop]
            
            # Convert to PIL
            pil_img = Image.fromarray(masked_crop)
            
            # Choose classification method based on config
            USE_CLIP = DEBUG_CONFIG.get("use_clip_instead", False)
            USE_FASHION_SIGLIP = DEBUG_CONFIG.get("use_fashion_siglip", True)
            
            if USE_CLIP:
                # Use CLIP for classification
                classification_result = classify_with_clip(
                    pil_img, processor, model, position_y=center_y, area_ratio=area_ratio
                )
                
                detected_label = classification_result['label']
                mask_dict['full_label'] = detected_label
                mask_dict['confidence'] = classification_result['confidence']
                all_scores = classification_result.get('all_scores', {})
                
                # For debug
                descriptive_hints = list(all_scores.keys())
                
                # Store debug info for CLIP
                mask_dict['debug_info'] = {
                    'prompts': descriptive_hints,
                    'all_scores': all_scores,
                    'position_y': center_y,
                    'mask_area': mask_dict['area'],
                    'aspect_ratio': aspect_ratio,
                    'model': 'CLIP'
                }
                
            elif USE_FASHION_SIGLIP:
                # Use Fashion SigLIP (best for clothing)
                classification_result = classify_with_fashion_siglip(
                    pil_img, processor, model, position_y=center_y, area_ratio=area_ratio
                )
                
                detected_label = classification_result['label']
                mask_dict['full_label'] = detected_label
                mask_dict['confidence'] = classification_result['confidence']
                all_scores = classification_result['all_scores']
                
                # Store multi-label results if available
                if 'multi_labels' in classification_result:
                    mask_dict['multi_labels'] = classification_result['multi_labels']
                
                # For debug
                descriptive_hints = list(all_scores.keys())
                
            elif DEBUG_CONFIG.get("use_improved_siglip", False):
                # Use improved classification with position hints
                classification_result = classify_with_position_hints(
                    pil_img, processor, model, mask_dict['area'], center_y, aspect_ratio
                )
                
                detected_label = classification_result['label']
                mask_dict['full_label'] = classification_result['raw_prompt']
                mask_dict['confidence'] = classification_result['confidence']
                all_scores = classification_result['all_scores']
                
                # For consistency with old code
                descriptive_hints = list(all_scores.keys())
                
            else:
                # Old classification method (backup)
                inputs = processor(text=descriptive_hints, images=pil_img, return_tensors="pt", padding=True)
                
                # Move inputs to MPS if available for GPU acceleration
                if torch.backends.mps.is_available():
                    inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image[0]
                    probs = torch.softmax(logits, dim=0)
                
                best_idx = probs.argmax().item()
                mask_dict['full_label'] = descriptive_hints[best_idx]  # Keep full label for debug
                mask_dict['confidence'] = probs[best_idx].item()
                detected_label = descriptive_hints[best_idx]
                
                # Store all scores for old method
                all_scores = {label: probs[i].item() for i, label in enumerate(descriptive_hints)}
            
            mask_dict['debug_info'] = {
                'prompts': descriptive_hints,
                'all_scores': all_scores,
                'position_y': center_y,
                'mask_area': mask_dict['area'],
                'aspect_ratio': aspect_ratio
            }
            
            # Add mask ID for tracking
            mask_dict['mask_id'] = i
            
            # Just use what SigLIP says, no corrections based on position
            print(f"    Mask {i}: detected as '{detected_label}' ({mask_dict['confidence']:.1%}), y_pos={center_y:.2f}")
            
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
                debug_img_buffer = BytesIO()
                pil_img.save(debug_img_buffer, format="PNG")
                debug_img_base64 = base64.b64encode(debug_img_buffer.getvalue()).decode()
                mask_dict['debug_info']['input_image'] = debug_img_base64
        
        siglip_time = time.time() - start_siglip
        
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
        
        total_time = sam2_time + siglip_time
        
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
        all_items_img, all_items_count = create_clothing_visualization(image_rgb, masks, "all", for_gemini=True)
        shirt_img, shirt_count = create_clothing_visualization(image_rgb, masks, "shirt", for_gemini=True)
        pants_img, pants_count = create_clothing_visualization(image_rgb, masks, "pants", for_gemini=True)
        shoes_img, shoes_count = create_clothing_visualization(image_rgb, masks, "shoes", for_gemini=True)
        
        # Log timing breakdown
        print(f"\n=== Timing Breakdown ===")
        print(f"Model loading: {model_load_time:.2f}s" if 'model_load_time' in locals() else "Model already loaded")
        print(f"SAM2 inference: {sam2_time:.2f}s")
        print(f"{classification_model} classification: {siglip_time:.2f}s")
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
            
            # Create masked region
            masked = image_rgb.copy()
            masked[~segmentation] = 255  # White background
            
            # Crop to bounding box
            cropped = masked[y:y+h, x:x+w]
            
            # Convert to base64
            pil_crop = Image.fromarray(cropped)
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
            'siglip_time': siglip_time,
            'total_time': total_time,
            'all_items_img': image_to_base64(all_items_img),
            'all_items_count': all_items_count,
            'shirt_img': image_to_base64(shirt_img),
            'shirt_count': shirt_count,
            'pants_img': image_to_base64(pants_img),
            'pants_count': pants_count,
            'shoes_img': image_to_base64(shoes_img),
            'shoes_count': shoes_count,
            'raw_sam2_img': image_to_base64(raw_sam2_img),
            'raw_masks_count': len(raw_masks_for_viz),
            'masks': masks_with_crops  # Include for editor
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
        base_name = filename.split('.')[0]
        
        # Look in the dedicated folder for this image
        masks_dir = os.path.join(base_dir, 'data', 'saved_masks', base_name)
        
        # Check if the folder exists
        if not os.path.exists(masks_dir):
            # Try old location for backward compatibility
            return quick_load_masks_legacy(filename)
        
        # Check if mask images exist in new location
        mask_files = {}
        for clothing_type in ['shirt', 'pants', 'shoes']:
            mask_filename = f"mask_{clothing_type}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            if os.path.exists(mask_path):
                # Read and convert to base64
                with open(mask_path, 'rb') as f:
                    mask_data = base64.b64encode(f.read()).decode()
                mask_files[clothing_type] = mask_data
        
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

def quick_load_masks_legacy(filename):
    """Load masks from old location for backward compatibility"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = filename.split('.')[0]
        
        # Check old location
        mask_files = {}
        for clothing_type in ['shirt', 'pants', 'shoes']:
            mask_filename = f"{base_name}_mask_{clothing_type}.png"
            mask_path = os.path.join(base_dir, 'data', 'sample_images', 'people', mask_filename)
            if os.path.exists(mask_path):
                with open(mask_path, 'rb') as f:
                    mask_data = base64.b64encode(f.read()).decode()
                mask_files[clothing_type] = mask_data
        
        if mask_files:
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
                'from_saved': True,
                'legacy': True
            })
        return jsonify({'success': False, 'has_masks': False})
    except Exception as e:
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
        all_items_img, all_items_count = create_clothing_visualization(image_rgb, masks, "all", for_gemini=True)
        shirt_img, shirt_count = create_clothing_visualization(image_rgb, masks, "shirt", for_gemini=True)
        pants_img, pants_count = create_clothing_visualization(image_rgb, masks, "pants", for_gemini=True)
        shoes_img, shoes_count = create_clothing_visualization(image_rgb, masks, "shoes", for_gemini=True)
        
        # Return results
        return jsonify({
            'sam2_time': 0,  # No processing time for loaded masks
            'siglip_time': 0,
            'total_time': 0,
            'loaded_from_cache': True,
            'cache_timestamp': timestamp,
            'all_items_img': image_to_base64(all_items_img),
            'all_items_count': all_items_count,
            'shirt_img': image_to_base64(shirt_img),
            'shirt_count': shirt_count,
            'pants_img': image_to_base64(pants_img),
            'pants_count': pants_count,
            'shoes_img': image_to_base64(shoes_img),
            'shoes_count': shoes_count
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
        garments_dir = os.path.join(base_dir, 'data', 'sample_images', 'garments')
        
        garments = []
        if os.path.exists(garments_dir):
            for file in os.listdir(garments_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.avif')):
                    garments.append(file)
        
        return jsonify({'garments': sorted(garments)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-sam2-only', methods=['POST'])
def process_sam2_only():
    """Process image with SAM2 only - no classification, just return all masks"""
    global mask_gen
    
    try:
        data = request.json
        image_path = data.get('image_path')
        model_size = data.get('model_size', 'large')
        
        # Load image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_image_path = os.path.join(base_dir, image_path)
        image = cv2.imread(full_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size
        orig_h, orig_w = image_rgb.shape[:2]
        
        # Resize for faster processing if image is large
        max_size = 768  # Process at lower resolution
        if orig_h > max_size or orig_w > max_size:
            scale = max_size / max(orig_h, orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            image_rgb_small = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"Resized image from {orig_w}x{orig_h} to {new_w}x{new_h} for faster processing")
        else:
            image_rgb_small = image_rgb
            scale = 1.0
        
        # Load model if needed
        if mask_gen is None or getattr(mask_gen, 'model_size', None) != model_size:
            return jsonify({'error': 'Model not loaded. Run full process first.'}), 400
        
        # Generate masks
        start_time = time.time()
        masks = mask_gen.generate(image_rgb)
        inference_time = time.time() - start_time
        
        # Create visualization with all masks in different colors
        overlay = image_rgb.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
                  (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128)]
        
        # Only show top 20 masks by area
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)[:20]
        
        for i, mask_dict in enumerate(masks_sorted):
            mask = mask_dict['segmentation']
            color = colors[i % len(colors)]
            
            # Apply colored mask
            mask_colored = np.zeros_like(overlay)
            mask_colored[mask] = color
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Convert to base64
        img_base64 = image_to_base64(overlay)
        
        return jsonify({
            'success': True,
            'mask_count': len(masks),
            'inference_time': inference_time,
            'visualization': img_base64,
            'top_20_masks': len(masks_sorted)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
                
                # Create white background with mask
                crop_height = y_max_pad - y_min_pad
                crop_width = x_max_pad - x_min_pad
                masked_crop = np.ones((crop_height, crop_width, 3), dtype=np.uint8) * 255
                mask_crop = mask_seg[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                masked_crop[mask_crop] = image_rgb[y_min_pad:y_max_pad, x_min_pad:x_max_pad][mask_crop]
                
                # Convert to base64
                pil_img = Image.fromarray(masked_crop)
                buffer = BytesIO()
                pil_img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
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

@app.route('/update-mask-classifications', methods=['POST'])
def update_mask_classifications():
    """Update mask classifications based on user edits"""
    try:
        data = request.json
        updates = data.get('updates', {})
        preserve_others = data.get('preserveOthers', False)
        
        if not hasattr(process_image, 'last_all_masks') or not hasattr(process_image, 'last_image_rgb'):
            return jsonify({'error': 'No masks available'}), 400
        
        if preserve_others:
            # Only reset masks for the categories being updated
            for category in updates.keys():
                for mask in process_image.last_all_masks:
                    if mask.get('label') == category:
                        mask['label'] = 'unknown'
                        mask['skip_viz'] = True
        else:
            # Reset all mask labels
            for mask in process_image.last_all_masks:
                if mask.get('label') in ['shirt', 'pants', 'shoes']:
                    mask['label'] = 'unknown'
                    mask['skip_viz'] = True
        
        # Apply new classifications
        for category, indices in updates.items():
            for idx in indices:
                if 0 <= idx < len(process_image.last_all_masks):
                    process_image.last_all_masks[idx]['label'] = category
                    process_image.last_all_masks[idx]['skip_viz'] = False
        
        # Update the filtered masks
        process_image.last_masks = [m for m in process_image.last_all_masks 
                                   if m.get('label') in ['shirt', 'pants', 'shoes'] and not m.get('skip_viz', False)]
        
        # Recreate visualizations
        image_rgb = process_image.last_image_rgb
        
        # Create visualizations
        all_items_img, all_items_count = create_clothing_visualization(image_rgb, process_image.last_masks, "all", for_gemini=True)
        shirt_img, shirt_count = create_clothing_visualization(image_rgb, process_image.last_masks, "shirt", for_gemini=True)
        pants_img, pants_count = create_clothing_visualization(image_rgb, process_image.last_masks, "pants", for_gemini=True)
        shoes_img, shoes_count = create_clothing_visualization(image_rgb, process_image.last_masks, "shoes", for_gemini=True)
        
        return jsonify({
            'all_items_img': image_to_base64(all_items_img),
            'all_items_count': all_items_count,
            'shirt_img': image_to_base64(shirt_img),
            'shirt_count': shirt_count,
            'pants_img': image_to_base64(pants_img),
            'pants_count': pants_count,
            'shoes_img': image_to_base64(shoes_img),
            'shoes_count': shoes_count,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in update_mask_classifications: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
        people_dir = os.path.join(base_dir, 'data', 'sample_images', 'people')
        
        people = []
        if os.path.exists(people_dir):
            for file in os.listdir(people_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not '_mask_' in file:
                    people.append(file)
        
        return jsonify({'people': sorted(people)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-gemini-data', methods=['POST'])
def get_gemini_data():
    """Get original image and mask for Gemini (no overlay, just raw data)"""
    global processor, model
    
    try:
        data = request.json
        image_path = data.get('image_path')
        clothing_type = data.get('clothing_type')
        
        # Get the stored masks from the last process call
        if not hasattr(process_image, 'last_masks') or not process_image.last_masks:
            return jsonify({'error': 'No masks available. Generate masks first.'}), 400
        
        # Use stored image if available, otherwise load from path
        if hasattr(process_image, 'last_image_rgb') and process_image.last_image_rgb is not None:
            image_rgb = process_image.last_image_rgb
        elif image_path:
            # Load original image from path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_image_path = os.path.join(base_dir, image_path)
            image = cv2.imread(full_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return jsonify({'error': 'No image available. Process an image first.'}), 400
        
        # Filter for specific clothing type
        if clothing_type == "shirt":
            type_labels = ["shirt", "t-shirt", "hoodie"]
        elif clothing_type == "pants":
            type_labels = ["pants", "jeans", "trousers"]
        elif clothing_type == "shoes":
            type_labels = ["shoes", "sneakers"]
        
        clothing_masks = [m for m in process_image.last_masks if m.get('label', '') in type_labels]
        
        # Get the largest mask for the clothing type
        if clothing_masks:
            # Sort by area and get the largest
            largest_mask = max(clothing_masks, key=lambda x: x['area'])
            mask = largest_mask['segmentation']
            
            # Convert mask to RGBA format - EXACT SAME AS streamlit_app.py
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[:, :, 3] = (~mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_rgba, mode='RGBA')
            
            # Convert to base64
            buffer = BytesIO()
            mask_pil.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Also convert original image to base64
            original_pil = Image.fromarray(image_rgb)
            buffer2 = BytesIO()
            original_pil.save(buffer2, format="PNG")
            original_base64 = base64.b64encode(buffer2.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'original_image': original_base64,
                'mask_image': mask_base64
            })
        else:
            return jsonify({'error': f'No {clothing_type} mask found'}), 400
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/prepare-wardrobe-gemini', methods=['POST'])
def prepare_wardrobe_gemini():
    """Prepare wardrobe item for Gemini try-on using stored masks"""
    try:
        data = request.json
        image_url = data.get('image_url')  # Firebase Storage URL
        mask_data = data.get('mask_data')  # The stored mask data from Firebase
        clothing_type = data.get('clothing_type')
        
        if not image_url or not mask_data:
            return jsonify({'error': 'Missing image_url or mask_data'}), 400
            
        # Download image from Firebase
        print(f"Downloading wardrobe image from: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Convert to numpy array
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image from URL'}), 500
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get the visualization for the selected clothing type
        visualizations = mask_data.get('visualizations', {})
        mask_base64 = visualizations.get(clothing_type)
        
        if not mask_base64:
            return jsonify({'error': f'No {clothing_type} mask found in stored data'}), 400
            
        # The stored visualization is already a mask image, but we need to convert it to RGBA format
        # Decode the stored mask visualization
        mask_img_data = base64.b64decode(mask_base64)
        mask_pil = Image.open(BytesIO(mask_img_data))
        mask_np = np.array(mask_pil)
        
        # Extract the mask from the visualization
        # The visualization has colored regions for clothing items
        # We need to create a binary mask where clothing pixels are True
        if len(mask_np.shape) == 3:
            # Convert to grayscale if needed
            mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask_np
            
        # Create binary mask where non-zero pixels are clothing
        binary_mask = mask_gray > 0
        
        # Convert to RGBA format for Gemini (transparent where clothing is)
        mask_rgba = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[:, :, 3] = (~binary_mask * 255).astype(np.uint8)  # Inverted for transparency
        mask_pil_rgba = Image.fromarray(mask_rgba, mode='RGBA')
        
        # Convert to base64
        buffer = BytesIO()
        mask_pil_rgba.save(buffer, format="PNG")
        mask_base64_rgba = base64.b64encode(buffer.getvalue()).decode()
        
        # Convert original image to base64
        original_pil = Image.fromarray(image_rgb)
        buffer2 = BytesIO()
        original_pil.save(buffer2, format="PNG")
        original_base64 = base64.b64encode(buffer2.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'original_image': original_base64,
            'mask_image': mask_base64_rgba
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/gemini-tryon', methods=['POST'])
def gemini_tryon():
    """Perform virtual try-on using Gemini - EXACT IMPLEMENTATION FROM streamlit_app.py"""
    try:
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Google GenAI SDK not installed'}), 500
        
        # Get request data
        data = request.json
        person_image = data.get('person_image')  # base64 - ORIGINAL IMAGE
        mask_image = data.get('mask_image')  # base64 - MASK AS RGBA
        garment_file = data.get('garment_file')
        clothing_type = data.get('clothing_type', 'shirt')
        
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({'error': 'GEMINI_API_KEY not set in .env file'}), 500
        
        # Configure Gemini with IMAGE GENERATION model - EXACT SAME AS streamlit_app.py
        genai.configure(api_key=api_key)
        generation_config = {
            "response_modalities": ["TEXT", "IMAGE"]
        }
        model = genai.GenerativeModel(
            'gemini-2.0-flash-preview-image-generation',
            generation_config=generation_config
        )
        
        # Decode person image from base64
        person_img_data = base64.b64decode(person_image.split(',')[1])
        person_pil = Image.open(BytesIO(person_img_data)).convert('RGB')
        
        # Decode mask image from base64
        mask_img_data = base64.b64decode(mask_image.split(',')[1])
        mask_pil = Image.open(BytesIO(mask_img_data))  # Keep as RGBA
        
        # Load garment image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        garment_path = os.path.join(base_dir, 'data', 'sample_images', 'garments', garment_file)
        garment_pil = Image.open(garment_path).convert('RGB')
        
        # Create prompt - EXACT SAME AS streamlit_app.py
        prompt = f"""Replace the {clothing_type} in the person image with the {clothing_type} from the reference image. 
        Use the provided mask to identify exactly which area to replace. 
        Maintain the person's pose, lighting, skin tone, and background. 
        Ensure the new {clothing_type} fits naturally and matches the original image style.
        Only change the clothing in the masked area, keep everything else identical."""
        
        # Prepare inputs for Gemini - EXACT SAME AS streamlit_app.py
        inputs = [prompt, person_pil, garment_pil, mask_pil]
        
        # Call Gemini API
        start_time = time.time()
        print(f"Sending to Gemini: {len(inputs)} inputs (prompt + {len(inputs)-1} images)")
        response = model.generate_content(inputs)
        
        # Check if response has candidates - EXACT SAME AS streamlit_app.py
        if response.candidates and len(response.candidates) > 0:
            # Iterate through parts to find image data
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Get the generated image data directly (no base64 decoding needed)
                    try:
                        result_image = Image.open(io.BytesIO(part.inline_data.data))
                        
                        # Convert to base64 for frontend
                        buffer = BytesIO()
                        result_image.save(buffer, format="PNG")
                        result_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        processing_time = time.time() - start_time
                        
                        return jsonify({
                            'success': True,
                            'result_image': result_base64,
                            'processing_time': processing_time
                        })
                    except Exception as e:
                        print(f"Failed to process image data: {str(e)}")
                        # Try with base64 decoding as fallback
                        try:
                            image_data = base64.b64decode(part.inline_data.data)
                            result_image = Image.open(io.BytesIO(image_data))
                            
                            # Convert to base64 for frontend
                            buffer = BytesIO()
                            result_image.save(buffer, format="PNG")
                            result_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            processing_time = time.time() - start_time
                            
                            return jsonify({
                                'success': True,
                                'result_image': result_base64,
                                'processing_time': processing_time
                            })
                        except:
                            continue
            
            return jsonify({'error': 'Gemini returned no image data in response'}), 500
        else:
            return jsonify({'error': 'Gemini returned no candidates'}), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Gemini API error: {str(e)}'}), 500

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
        all_items_img, all_items_count = create_clothing_visualization(image_rgb, filtered_masks, "all", for_gemini=True)
        shirt_img, shirt_count = create_clothing_visualization(image_rgb, filtered_masks, "shirt", for_gemini=True)
        pants_img, pants_count = create_clothing_visualization(image_rgb, filtered_masks, "pants", for_gemini=True)
        shoes_img, shoes_count = create_clothing_visualization(image_rgb, filtered_masks, "shoes", for_gemini=True)
        
        return jsonify({
            'all_items_img': image_to_base64(all_items_img),
            'all_items_count': all_items_count,
            'shirt_img': image_to_base64(shirt_img),
            'shirt_count': shirt_count,
            'pants_img': image_to_base64(pants_img),
            'pants_count': pants_count,
            'shoes_img': image_to_base64(shoes_img),
            'shoes_count': shoes_count
        })
        
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

def cleanup():
    """Clean up models from memory on shutdown"""
    global mask_gen, processor, model
    print("\nCleaning up models from memory...")
    
    if mask_gen is not None:
        del mask_gen
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
    atexit.register(cleanup)  # Register cleanup on exit
    app.run(debug=True, port=5001)