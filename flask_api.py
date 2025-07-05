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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoProcessor, AutoModel
import time

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
        print(f"Loading SAM2 {model_size.upper()} model...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Get config and checkpoint paths
        config_file, checkpoint_file = model_configs.get(model_size, model_configs['large'])
        
        # Change to parent directory to match working demo
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        
        try:
            # Load the selected model with optimizations
            sam2 = build_sam2(config_file, checkpoint_file, device="mps", apply_postprocessing=False)
            
            # Enable optimizations for M4
            sam2.eval()  # Set to evaluation mode
            
            # For SAM 2.1 models, use tuned parameters for better clothing detection
            if 'v2.1' in model_size:
                if 'tiny' in model_size or 'small' in model_size:
                    # Balanced parameters for tiny/small models
                    mask_gen = SAM2AutomaticMaskGenerator(
                        sam2,
                        points_per_side=32,  # Keep default
                        points_per_batch=64,  # Default batch size
                        pred_iou_thresh=0.82,  # Slightly lower threshold
                        stability_score_thresh=0.88,  # Slightly lower stability
                        crop_n_layers=0,  # Disable crops for speed
                        crop_n_points_downscale_factor=1,
                        min_mask_region_area=100,  # Standard min area
                    )
                else:
                    # Base and Large models - balanced parameters
                    mask_gen = SAM2AutomaticMaskGenerator(
                        sam2,
                        points_per_side=32,
                        pred_iou_thresh=0.86,
                        stability_score_thresh=0.92,
                        crop_n_layers=0,
                        crop_n_points_downscale_factor=1,
                        min_mask_region_area=100,
                    )
            else:
                # SAM 2.0 uses default parameters
                mask_gen = SAM2AutomaticMaskGenerator(sam2)
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
        # Run SAM-2 on Replicate
        start_time = time.time()
        output = replicate.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={
                "image": img_uri,
                "use_m2m": True,
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95
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
        type_labels = ["shirt", "t-shirt", "hoodie"]
    elif clothing_type == "pants":
        type_labels = ["pants", "jeans", "trousers"]
    elif clothing_type == "shoes":
        type_labels = ["shoes", "sneakers"]
    else:  # all items
        type_labels = ["shirt", "t-shirt", "hoodie", "pants", "jeans", "trousers", "dress", "shoes", "sneakers"]
    
    clothing_masks = [m for m in masks if m.get('label', '') in type_labels]
    
    # Special handling for shoes - merge nearby shoe detections
    if clothing_type == "shoes" and len(clothing_masks) > 0:
        # For SAM 2.1, also check for foot-like shapes in lower part of image
        if len(clothing_masks) < 2:
            # Look for any masks in the foot region that might be shoes
            foot_region_masks = []
            for m in masks:
                if m.get('label') not in ['background', 'pants', 'jeans', 'trousers', 'shirt', 't-shirt', 'hoodie']:
                    mask = m['segmentation']
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        center_y = y_indices.mean() / image_rgb.shape[0]
                        area = m['area']
                        # Look for masks in foot region with reasonable size
                        if center_y > 0.7 and area > 200 and area < 10000:
                            m['label'] = 'shoes'
                            m['confidence'] = 0.7
                            foot_region_masks.append(m)
            
            clothing_masks.extend(foot_region_masks)
        
        # Group shoes by proximity (left vs right foot)
        shoe_groups = {'left': [], 'right': []}
        image_center_x = image_rgb.shape[1] / 2
        
        for mask_dict in clothing_masks:
            mask = mask_dict['segmentation']
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0:
                center_x = x_indices.mean()
                if center_x < image_center_x:
                    shoe_groups['right'].append(mask_dict)  # Right foot appears on left side of image
                else:
                    shoe_groups['left'].append(mask_dict)
        
        # Keep best shoe from each group
        final_shoes = []
        for group in shoe_groups.values():
            if group:
                # Sort by area and keep the largest
                best_shoe = max(group, key=lambda x: x['area'])
                final_shoes.append(best_shoe)
        
        clothing_masks = final_shoes
    
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

# Helper functions for mask storage
def save_mask_images(image_path, image_rgb, masks):
    """Save mask images as PNG files for each clothing type"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Save masks for each clothing type
        clothing_types = ['shirt', 'pants', 'shoes']
        
        for clothing_type in clothing_types:
            # Create visualization without labels
            mask_img, count = create_clothing_visualization(image_rgb, masks, clothing_type, for_gemini=True)
            
            if count > 0:
                # Save the masked image
                mask_filename = f"{base_name}_mask_{clothing_type}.png"
                mask_path = os.path.join(base_dir, 'data', 'sample_images', 'people', mask_filename)
                
                # Convert to PIL and save
                mask_pil = Image.fromarray(mask_img)
                mask_pil.save(mask_path)
                print(f"Saved {clothing_type} mask to {mask_filename}")
        
        # Also save the raw masks data
        masks_dir = os.path.join(base_dir, 'data', 'saved_masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        masks_file = os.path.join(masks_dir, f"{base_name}_masks.pkl")
        serializable_masks = []
        for mask in masks:
            mask_copy = mask.copy()
            mask_copy['segmentation'] = mask_copy['segmentation'].tolist()
            serializable_masks.append(mask_copy)
            
        with open(masks_file, 'wb') as f:
            pickle.dump({
                'masks': serializable_masks,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }, f)
            
        print(f"Saved raw masks data to {masks_file}")
        
    except Exception as e:
        print(f"Error saving masks: {str(e)}")

def load_masks_from_file(image_path, model_size):
    """Load masks from a pickle file"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        masks_dir = os.path.join(base_dir, 'saved_masks')
        filename = get_mask_filename(image_path, model_size)
        filepath = os.path.join(masks_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        model_size = data.get('model_size', 'large')
        
        # Construct full path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_image_path = os.path.join(base_dir, image_path)
        
        # Load and process image
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
            try:
                load_models(model_size)
            except Exception as e:
                print(f"Model loading error: {str(e)}")
                return jsonify({'error': f'Model loading failed: {str(e)}'}), 500
            
            # Generate masks with SAM2
            print(f"Generating masks with {model_size} model...")
            start_sam2 = time.time()
            masks = mask_gen.generate(image_rgb)
            sam2_time = time.time() - start_sam2
            print(f"Generated {len(masks)} masks in {sam2_time:.2f} seconds")
        
        # Sort by area
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Define clothing labels
        clothing_labels = ["shirt", "t-shirt", "hoodie", "pants", "jeans", "trousers", "dress", "shoes", "sneakers", "person", "background", "face", "sky", "ground"]
        
        # Load SigLIP if using Replicate (since we need it for classification)
        if model_size == 'replicate' and (processor is None or model is None):
            print("Loading SigLIP for classification...")
            processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast=True)
            model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            model.eval()
            
            # Move SigLIP to MPS for GPU acceleration
            if torch.backends.mps.is_available():
                model = model.to("mps")
                print("SigLIP moved to MPS for GPU acceleration")
        
        # Classify each mask with SigLIP
        print(f"Classifying {len(masks)} masks with SigLIP...")
        start_siglip = time.time()
        for i, mask_dict in enumerate(masks):
            if i % 10 == 0:
                print(f"  Processing mask {i}/{len(masks)}...")
            mask = mask_dict['segmentation']
            
            # Skip very large masks
            if mask_dict['area'] > 0.5 * image_rgb.shape[0] * image_rgb.shape[1]:
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
            
            # Position-based classification with adjusted thresholds
            if mask_dict['area'] > shirt_area_min and mask_dict['area'] < shirt_area_max and center_y < 0.5:
                position_hint = ["shirt", "hoodie", "t-shirt", "jacket"]
            elif mask_dict['area'] > pants_area_min and mask_dict['area'] < pants_area_max and center_y > 0.4 and center_y < 0.8:
                position_hint = ["pants", "jeans", "trousers"]
            elif mask_dict['area'] < shoe_area_max and center_y > 0.75:  # Lowered from 0.8
                if mask_dict['area'] < shoe_area_min:
                    mask_dict['label'] = 'fragment'
                    mask_dict['confidence'] = 0.0
                    continue
                position_hint = ["shoes", "sneakers"]
            else:
                position_hint = clothing_labels
            
            # Create masked image
            masked_image = np.ones_like(image_rgb) * 255
            masked_image[mask] = image_rgb[mask]
            
            # Get bounding box
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            cropped = masked_image[y_min:y_max+1, x_min:x_max+1]
            
            # Convert to PIL
            pil_img = Image.fromarray(cropped.astype(np.uint8))
            
            # Classify
            inputs = processor(text=position_hint, images=pil_img, return_tensors="pt", padding=True)
            
            # Move inputs to MPS if available for GPU acceleration
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0)
            
            best_idx = probs.argmax().item()
            mask_dict['label'] = position_hint[best_idx]
            mask_dict['confidence'] = probs[best_idx].item()
        
        siglip_time = time.time() - start_siglip
        total_time = sam2_time + siglip_time
        
        # Store masks globally for Gemini
        process_image.last_masks = masks
        process_image.last_image_path = image_path
        
        # Save mask images permanently
        save_mask_images(image_path, image_rgb, masks)
        
        # Create visualizations without labels (all in Gemini format)
        all_items_img, all_items_count = create_clothing_visualization(image_rgb, masks, "all", for_gemini=True)
        shirt_img, shirt_count = create_clothing_visualization(image_rgb, masks, "shirt", for_gemini=True)
        pants_img, pants_count = create_clothing_visualization(image_rgb, masks, "pants", for_gemini=True)
        shoes_img, shoes_count = create_clothing_visualization(image_rgb, masks, "shoes", for_gemini=True)
        
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
            'shoes_count': shoes_count
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_name = filename.split('.')[0]
        
        # Check if mask images exist
        mask_files = {}
        for clothing_type in ['shirt', 'pants', 'shoes']:
            mask_filename = f"{base_name}_mask_{clothing_type}.png"
            mask_path = os.path.join(base_dir, 'data', 'sample_images', 'people', mask_filename)
            if os.path.exists(mask_path):
                # Read and convert to base64
                with open(mask_path, 'rb') as f:
                    mask_data = base64.b64encode(f.read()).decode()
                mask_files[clothing_type] = mask_data
        
        if mask_files:
            # Also load the raw masks data if available
            masks_file = os.path.join(base_dir, 'data', 'saved_masks', f"{base_name}_masks.pkl")
            if os.path.exists(masks_file):
                with open(masks_file, 'rb') as f:
                    data = pickle.load(f)
                    masks = []
                    for mask in data['masks']:
                        mask['segmentation'] = np.array(mask['segmentation'], dtype=bool)
                        masks.append(mask)
                    
                    # Store masks globally
                    process_image.last_masks = masks
                    process_image.last_image_path = filename
            
            # Count items for each type
            counts = {}
            for clothing_type in ['shirt', 'pants', 'shoes']:
                if clothing_type in mask_files:
                    counts[f"{clothing_type}_count"] = 1  # Simplified - just say 1 if mask exists
            
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_image_path = os.path.join(base_dir, image_path)
        image = cv2.imread(full_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store masks globally for Gemini
        process_image.last_masks = masks
        process_image.last_image_path = image_path
        
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        garments_dir = os.path.join(base_dir, 'data', 'sample_images', 'garments')
        
        garments = []
        if os.path.exists(garments_dir):
            for file in os.listdir(garments_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    garments.append(file)
        
        return jsonify({'garments': sorted(garments)})
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
        
        # Load original image
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_image_path = os.path.join(base_dir, image_path)
        image = cv2.imread(full_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)