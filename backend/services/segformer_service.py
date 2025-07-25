"""
Segformer B2 Clothes Segmentation Service
Replaces MediaPipe + SAM2 + CLIP with a single model
License: MIT (Free for commercial use)
"""

import numpy as np
from PIL import Image
import time
from transformers import pipeline
import torch

# Global segmentation pipeline
_segmentation_pipeline = None

# Model info
MODEL_INFO = {
    "name": "mattmdjaga/segformer_b2_clothes",
    "license": "MIT (Free for commercial use)",
    "description": "SegFormer B2 fine-tuned on ATR dataset for clothes segmentation",
    "labels": ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", 
               "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", 
               "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"],
    "hosting": "Self-hosted via Hugging Face transformers (no API costs)",
    "size": "~100MB model download"
}

# Label mapping to simplified categories
LABEL_MAPPING = {
    # Upper body
    "Upper-clothes": "shirt",
    "Dress": "dress",
    
    # Lower body
    "Pants": "pants",
    "Skirt": "skirt",
    
    # Footwear
    "Left-shoe": "shoes",
    "Right-shoe": "shoes",
    
    # Accessories
    "Hat": "hat",
    "Sunglasses": "glasses",
    "Belt": "belt",
    "Bag": "bag",
    "Scarf": "scarf",
    
    # Body parts (non-clothing)
    "Hair": "non_clothing",
    "Face": "non_clothing",
    "Left-leg": "non_clothing",
    "Right-leg": "non_clothing",
    "Left-arm": "non_clothing",
    "Right-arm": "non_clothing",
    "Background": "background"
}

def init_segmentation():
    """Initialize segmentation model"""
    global _segmentation_pipeline
    if _segmentation_pipeline is None:
        print("Loading Segformer B2 Clothes model...")
        start_time = time.time()
        
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        if torch.backends.mps.is_available():
            device = "mps"
        
        _segmentation_pipeline = pipeline(
            "image-segmentation", 
            model=MODEL_INFO["name"],
            device=device,
            image_processor_kwargs={"use_fast": True}
        )
        
        load_time = time.time() - start_time
        print(f"Segformer model loaded in {load_time:.2f}s on device: {device}")
    
    return _segmentation_pipeline

def process_with_segformer(image_path_or_array):
    """
    Process image with Segformer to get clothing segments
    
    Args:
        image_path_or_array: Either a file path or numpy array of the image
        
    Returns:
        tuple: (masks, processing_time)
        where masks is a list of dictionaries with keys:
        - segmentation: boolean numpy array
        - label: simplified label (shirt, pants, shoes, etc.)
        - original_label: original model label
        - confidence: always 1.0 (model doesn't provide scores)
        - area: number of pixels
        - bbox: [x, y, w, h] bounding box
    """
    start_time = time.time()
    
    # Load image
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        # Assume numpy array
        image = Image.fromarray(image_path_or_array)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Initialize pipeline
    pipeline = init_segmentation()
    
    # Run segmentation
    print(f"Processing image of size: {image.size}")
    results = pipeline(image)
    
    # Convert results to our format
    masks = []
    
    for i, result in enumerate(results):
        original_label = result['label']
        
        # Skip background
        if original_label == "Background":
            continue
            
        # Get simplified label
        simple_label = LABEL_MAPPING.get(original_label, "other")
        
        # Convert mask to numpy array
        mask = result['mask']
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert('L')) > 127
        else:
            mask_array = mask > 127
        
        # Calculate area
        area = int(np.sum(mask_array))
        
        # Skip very small masks (noise)
        if area < 500:
            continue
        
        # Calculate bounding box
        y_indices, x_indices = np.where(mask_array)
        if len(y_indices) == 0:
            continue
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        
        # Calculate content bounds for display optimization
        content_bounds = {
            'minX': int(x_min),
            'maxX': int(x_max),
            'minY': int(y_min),
            'maxY': int(y_max),
            'width': w,
            'height': h
        }
        
        # Calculate position info for filtering
        center_y = y_indices.mean() / h
        
        mask_dict = {
            'segmentation': mask_array,
            'label': simple_label,
            'original_label': original_label,
            'full_label': original_label,
            'confidence': 1.0,  # Segformer doesn't provide confidence scores
            'area': area,
            'bbox': bbox,
            'content_bounds': content_bounds,  # New: exact pixel bounds for display
            'predicted_iou': 0.95,  # High quality masks
            'stability_score': 0.95,  # High quality masks
            'position_y': center_y,
            'skip_viz': simple_label == "non_clothing"
        }
        
        masks.append(mask_dict)
        print(f"Segment {i+1}: {original_label} -> {simple_label} (area: {area:,} pixels)")
    
    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    processing_time = time.time() - start_time
    print(f"Segformer processing completed in {processing_time:.2f}s, found {len(masks)} masks")
    
    return masks, processing_time

def filter_best_clothing_items(masks):
    """
    Filter to keep only the best mask for each clothing category
    Similar to the post-processing in the original code
    
    Args:
        masks: List of mask dictionaries
        
    Returns:
        List of filtered masks
    """
    # Group masks by category
    category_masks = {
        'shirt': [],
        'pants': [],
        'shoes': [],
        'dress': [],
        'skirt': [],
        'other': []
    }
    
    for mask in masks:
        label = mask.get('label', '')
        if label in category_masks:
            category_masks[label].append(mask)
        elif label not in ['non_clothing', 'background']:
            category_masks['other'].append(mask)
    
    filtered_masks = []
    
    # For shirt, pants, dress, skirt - keep only the largest one
    for category in ['shirt', 'pants', 'dress', 'skirt']:
        if category_masks[category]:
            # Sort by area and take the largest
            best_mask = max(category_masks[category], key=lambda x: x.get('area', 0))
            best_mask['skip_viz'] = False
            filtered_masks.append(best_mask)
            print(f"  Best {category}: {best_mask['original_label']} (area: {best_mask['area']:,})")
    
    # For shoes - keep up to 2 (left and right)
    if category_masks['shoes']:
        shoe_masks = sorted(category_masks['shoes'], key=lambda x: x.get('area', 0), reverse=True)
        
        if len(shoe_masks) >= 2:
            # Check if we have distinct left and right shoes
            left_shoes = [m for m in shoe_masks if 'Left' in m['original_label']]
            right_shoes = [m for m in shoe_masks if 'Right' in m['original_label']]
            
            if left_shoes and right_shoes:
                # Take best of each
                left_shoes[0]['skip_viz'] = False
                right_shoes[0]['skip_viz'] = False
                filtered_masks.extend([left_shoes[0], right_shoes[0]])
                print(f"  Left shoe: area {left_shoes[0]['area']:,}")
                print(f"  Right shoe: area {right_shoes[0]['area']:,}")
            else:
                # Just take the two largest
                shoe_masks[0]['skip_viz'] = False
                shoe_masks[1]['skip_viz'] = False
                filtered_masks.extend(shoe_masks[:2])
        elif shoe_masks:
            # Only one shoe
            shoe_masks[0]['skip_viz'] = False
            filtered_masks.append(shoe_masks[0])
    
    # Add back non-clothing masks (but keep them marked as skip_viz)
    filtered_mask_ids = {id(mask) for mask in filtered_masks}
    for mask in masks:
        if mask.get('label') == 'non_clothing' and id(mask) not in filtered_mask_ids:
            filtered_masks.append(mask)
    
    return filtered_masks

def cleanup_model():
    """Clean up the model from memory"""
    global _segmentation_pipeline
    if _segmentation_pipeline is not None:
        del _segmentation_pipeline
        _segmentation_pipeline = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print("Segformer model cleaned up from memory")