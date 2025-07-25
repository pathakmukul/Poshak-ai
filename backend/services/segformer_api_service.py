"""
Segformer API Service - Uses Hugging Face API instead of local model
"""

import os
import time
import numpy as np
from PIL import Image
import requests
import base64
from io import BytesIO

# Hugging Face API configuration
HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', '')  # Optional for public models
HF_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Model info (same as local)
MODEL_INFO = {
    "name": "mattmdjaga/segformer_b2_clothes",
    "license": "MIT (Free for commercial use)",
    "description": "SegFormer B2 fine-tuned on ATR dataset for clothes segmentation",
    "labels": ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
               "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face",
               "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
}

# Label mapping to our categories
LABEL_MAPPING = {
    'Upper-clothes': 'shirt',
    'Skirt': 'skirt', 
    'Pants': 'pants',
    'Dress': 'dress',
    'Left-shoe': 'shoes',
    'Right-shoe': 'shoes',
    'Hat': 'non_clothing',
    'Hair': 'non_clothing',
    'Sunglasses': 'non_clothing',
    'Belt': 'non_clothing',
    'Face': 'non_clothing',
    'Left-leg': 'non_clothing',
    'Right-leg': 'non_clothing',
    'Left-arm': 'non_clothing',
    'Right-arm': 'non_clothing',
    'Bag': 'non_clothing',
    'Scarf': 'non_clothing',
    'Background': 'non_clothing'
}

def process_with_segformer(image_path_or_array):
    """Process image using Hugging Face API"""
    print("\n" + "="*60)
    print("🌐 PROCESSING WITH HUGGING FACE API")
    print(f"   API URL: {HF_API_URL}")
    print(f"   Token configured: {'Yes' if HF_API_TOKEN else 'No'}")
    print("="*60 + "\n")
    start_time = time.time()
    
    # Convert to PIL Image if needed
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array)
    else:
        image = Image.fromarray(image_path_or_array)
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Prepare headers
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": img_str}, timeout=30)
        response.raise_for_status()
        
        # Parse response
        api_results = response.json()
        
        # Note: The exact response format depends on the model
        # This is a placeholder implementation
        masks = []
        h, w = np.array(image).shape[:2]
        
        # If API returns an error about model loading
        if isinstance(api_results, dict) and 'error' in api_results:
            print(f"API Error: {api_results['error']}")
            raise Exception(api_results['error'])
        
        # Process results (format depends on the specific model)
        # For semantic segmentation models, the response might be different
        # This needs to be adjusted based on actual API response
        
        print(f"API Response type: {type(api_results)}")
        if isinstance(api_results, list):
            print(f"Got {len(api_results)} results")
            # Debug: Print first result to understand structure
            if api_results:
                print(f"First result structure: {api_results[0].keys() if isinstance(api_results[0], dict) else type(api_results[0])}")
                if isinstance(api_results[0], dict):
                    for key, value in api_results[0].items():
                        print(f"  {key}: {type(value)} - {str(value)[:100] if isinstance(value, str) else value}")
        
        # Parse actual segmentation masks from API response
        for idx, result in enumerate(api_results):
            label_name = result.get('label', 'Unknown')
            score = result.get('score', 1.0)
            mask_b64 = result.get('mask', '')
            
            if not mask_b64:
                continue
                
            # Decode mask from base64
            try:
                mask_img = Image.open(BytesIO(base64.b64decode(mask_b64)))
                # Convert to binary mask
                mask_array = np.array(mask_img.resize((w, h))) > 128
                
                # Map to our categories
                our_label = LABEL_MAPPING.get(label_name, 'non_clothing')
                
                # Calculate area
                area = np.sum(mask_array)
                
                # Skip very small masks
                if area < 100:
                    continue
                
                print(f"Segment {idx}: {label_name} -> {our_label} (area: {area:,} pixels)")
                
                # Calculate bounding box
                y_indices, x_indices = np.where(mask_array)
                if len(y_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                else:
                    bbox = [0, 0, 0, 0]
                
                masks.append({
                    'segmentation': mask_array,
                    'label': our_label,
                    'original_label': label_name,
                    'confidence': float(score),
                    'area': int(area),
                    'bbox': bbox,
                    'skip_viz': our_label == 'non_clothing'
                })
            except Exception as e:
                print(f"Error processing mask for {label_name}: {e}")
                continue
        
        processing_time = time.time() - start_time
        print(f"HF API processing completed in {processing_time:.2f}s")
        
        # Return empty masks for now - needs proper implementation
        return masks, processing_time
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            error_msg = "Model is loading on Hugging Face servers. Please try again in a minute."
            print(error_msg)
            raise Exception(error_msg)
        else:
            raise Exception(f"Hugging Face API error: {e}")
    except Exception as e:
        print(f"API request failed: {str(e)}")
        raise

# Re-export the filter function from local service
from services.segformer_service import filter_best_clothing_items

# Stub functions for compatibility
def init_segmentation():
    """No initialization needed for API mode"""
    print("Using Hugging Face API - no local model initialization needed")

def cleanup_model():
    """No cleanup needed for API mode"""
    pass