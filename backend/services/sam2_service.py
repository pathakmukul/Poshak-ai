"""
SAM2 (Segment Anything Model 2) Service
Handles all SAM2 segmentation via Replicate API or FAL
"""

import os
import time
import base64
import cv2
import replicate
import requests
from io import BytesIO
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


# SAM2 Configuration - Enhanced for better clothing detection
SAM2_CONFIG = {
    "points_per_side": 64,  # 4096 points for much better coverage
    "points_per_batch": 128,  # Process more points at once
    "pred_iou_thresh": 0.6,  # Lower threshold to catch more segments
    "stability_score_thresh": 0.85,  # More permissive for clothing variations
    "stability_score_offset": 0.7,  # Lower offset for more mask variations
    "mask_threshold": -0.5,  # Negative for softer boundaries
    "box_nms_thresh": 0.3,  # Lower NMS to keep overlapping clothing items
    "crop_n_layers": 1,  # Multi-scale helps with shoes and accessories
    "crop_nms_thresh": 0.7,  # Standard crop NMS
    "crop_overlap_ratio": 0.4,  # More overlap for better boundaries
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,  # Filter out noise but keep small items
    "multimask_output": True,  # Get 3 masks per point
    "use_m2m": True,  # Mask-to-mask refinement for cleaner boundaries
}

# Import provider config from main config file
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_improved import SAM2_PROVIDER_CONFIG as PROVIDER_CONFIG


def get_active_sam2_config():
    """Returns the SAM2 configuration"""
    return SAM2_CONFIG


def file_to_base64(file_path):
    """Convert file content directly to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def convert_replicate_masks_to_sam_format(mask_images, original_image_shape):
    """
    Convert Replicate mask images to SAM format with metadata
    
    Args:
        mask_images: List of dicts with 'image' (PIL Image)
        original_image_shape: Shape of the original image (h, w, c)
        
    Returns:
        List of mask dicts in SAM format
    """
    import numpy as np
    
    h, w = original_image_shape[:2]
    sam_masks = []
    
    for i, mask_data in enumerate(mask_images):
        # Get PIL image
        mask_img = mask_data['image']
        
        # Convert to numpy array
        mask_np = np.array(mask_img)
        
        # If RGB, convert to grayscale
        if len(mask_np.shape) == 3:
            # Use any channel that has variation (they should all be the same for masks)
            mask_np = mask_np[:, :, 0]
        
        # Convert to binary mask
        binary_mask = mask_np > 127  # Threshold at middle value
        
        # Find bounding box
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not rows.any() or not cols.any():
            continue  # Skip empty masks
            
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Create SAM-format mask dict
        sam_mask = {
            'segmentation': binary_mask,
            'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'area': int(binary_mask.sum()),
            'predicted_iou': 0.9,  # Default high IoU since these are from SAM2
            'stability_score': 0.9,  # Default high stability
            'mask_index': i
        }
        
        sam_masks.append(sam_mask)
    
    return sam_masks


def process_with_replicate(image_path, is_person_cropped=False):
    """
    Process image using Replicate API for SAM2 segmentation
    
    Args:
        image_path: Path to the image file
        is_person_cropped: Whether the image is already cropped to person by MediaPipe
        
    Returns:
        tuple: (masks, processing_time)
    """
    print("Processing with Replicate API...")
    
    # Check if Replicate API token is set
    if not os.environ.get('REPLICATE_API_TOKEN'):
        raise Exception("REPLICATE_API_TOKEN environment variable not set. Get your token from https://replicate.com/account/api-tokens")
    
    # Log the image being processed
    img = cv2.imread(image_path)
    print(f"🎯 SAM2 processing image from: {image_path}")
    print(f"🎯 Image dimensions: {img.shape if img is not None else 'Unknown'}")
    print(f"🎯 Person pre-cropped: {is_person_cropped}")
    
    # Convert image to base64 for upload
    img_data = file_to_base64(image_path)
    img_uri = f"data:image/png;base64,{img_data}"
    
    try:
        # Get active configuration
        config = get_active_sam2_config()
        
        # Log the configuration being used
        print(f"\n=== Using SAM2 Configuration ===")
        print(f"Key parameters:")
        print(f"  - points_per_side: {config.get('points_per_side', 32)}")
        print(f"  - pred_iou_thresh: {config.get('pred_iou_thresh', 0.88)}")
        print(f"  - box_nms_thresh: {config.get('box_nms_thresh', 0.7)}")
        print(f"  - mask_threshold: {config.get('mask_threshold', 0.0)}")
        print(f"  - multimask_output: {config.get('multimask_output', True)}")
        print(f"=======================\n")
        
        # Run SAM-2 on Replicate with our config
        start_time = time.time()
        print("🚀 Sending request to Replicate SAM2...")
        output = replicate.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={
                "image": img_uri,
                "use_m2m": config.get("use_m2m", True),
                "points_per_side": config.get("points_per_side", 32),
                "points_per_batch": config.get("points_per_batch", 64),
                "pred_iou_thresh": config.get("pred_iou_thresh", 0.88),
                "stability_score_thresh": config.get("stability_score_thresh", 0.95),
                "stability_score_offset": config.get("stability_score_offset", 1.0),
                "mask_threshold": config.get("mask_threshold", 0.0),
                "box_nms_thresh": config.get("box_nms_thresh", 0.7),
                "crop_n_layers": config.get("crop_n_layers", 0),
                "crop_nms_thresh": config.get("crop_nms_thresh", 0.7),
                "crop_overlap_ratio": config.get("crop_overlap_ratio", 0.41),
                "crop_n_points_downscale_factor": config.get("crop_n_points_downscale_factor", 1),
                "min_mask_region_area": config.get("min_mask_region_area", 0),
                "multimask_output": config.get("multimask_output", True)
            }
        )
        sam2_time = time.time() - start_time
        print(f"⏱️  Replicate SAM2 API completed in: {sam2_time:.2f} seconds")
        
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
                    # Try to find any key that might contain URLs
                    for key, value in output.items():
                        if isinstance(value, list) and value:
                            if isinstance(value[0], str) and value[0].startswith('http'):
                                mask_urls = value
                                break
                    else:
                        mask_urls = []
            else:
                mask_urls = []
            
            print(f"Found {len(mask_urls) if isinstance(mask_urls, list) else 1} mask URLs")
            
            # Download and process each mask
            for i, mask_url in enumerate(mask_urls if isinstance(mask_urls, list) else [mask_urls]):
                if not mask_url:
                    continue
                    
                try:
                    # Download the mask
                    response = requests.get(mask_url)
                    if response.status_code == 200:
                        # Load mask as grayscale
                        mask_img = Image.open(BytesIO(response.content))
                        
                        # Convert to RGB if needed
                        if mask_img.mode != 'RGB':
                            mask_img = mask_img.convert('RGB')
                        
                        masks.append({
                            'url': mask_url,
                            'image': mask_img,
                            'index': i
                        })
                    else:
                        print(f"Failed to download mask {i}: HTTP {response.status_code}")
                except Exception as e:
                    print(f"Error processing mask {i}: {str(e)}")
        
        print(f"Successfully processed {len(masks)} masks")
        
        # Convert masks to SAM format
        if masks and img is not None:
            sam_format_masks = convert_replicate_masks_to_sam_format(masks, img.shape)
            print(f"Converted {len(sam_format_masks)} masks to SAM format")
            return sam_format_masks, sam2_time
        else:
            return [], sam2_time
        
    except Exception as e:
        print(f"Replicate API error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


def download_mask(mask_url, index):
    """Download a single mask from URL"""
    try:
        response = requests.get(mask_url, timeout=5)
        if response.status_code == 200:
            return index, Image.open(BytesIO(response.content))
        else:
            print(f"Failed to download mask {index}: HTTP {response.status_code}")
            return index, None
    except Exception as e:
        print(f"Error downloading mask {index}: {str(e)}")
        return index, None


def convert_fal_masks_to_sam_format(mask_data, original_image_shape, use_base64=False):
    """
    Convert FAL mask URLs or base64 data to SAM format with metadata
    
    Args:
        mask_data: List of mask image URLs or base64 data URLs
        original_image_shape: Shape of the original image (h, w, c)
        use_base64: Whether mask_data contains base64 data URLs
        
    Returns:
        List of mask dicts in SAM format
    """
    import numpy as np
    
    h, w = original_image_shape[:2]
    sam_masks = []
    
    print(f"DEBUG: convert_fal_masks_to_sam_format called with use_base64={use_base64}, mask_data length={len(mask_data) if mask_data else 0}")
    
    # Handle parallel downloads for URLs
    if not use_base64 and mask_data:
        print(f"Downloading {len(mask_data)} masks in parallel...")
        download_start = time.time()
        
        # Download all masks in parallel
        mask_images = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(download_mask, url, i): i 
                             for i, url in enumerate(mask_data)}
            
            for future in as_completed(future_to_index):
                index, mask_img = future.result()
                if mask_img:
                    mask_images[index] = mask_img
        
        print(f"Downloaded {len(mask_images)} masks in {time.time() - download_start:.2f}s")
        
        # Process downloaded masks in order
        for i in sorted(mask_images.keys()):
            mask_img = mask_images[i]
            try:
                # Convert to numpy array
                mask_np = np.array(mask_img)
                
                # If RGB, convert to grayscale
                if len(mask_np.shape) == 3:
                    mask_np = mask_np[:, :, 0]
                
                # Convert to binary mask
                binary_mask = mask_np > 127
                
                # Find bounding box
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                
                if not rows.any() or not cols.any():
                    continue
                    
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # Create SAM-format mask dict
                sam_mask = {
                    'segmentation': binary_mask,
                    'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                    'area': int(binary_mask.sum()),
                    'predicted_iou': 0.9,
                    'stability_score': 0.9,
                    'mask_index': i
                }
                
                sam_masks.append(sam_mask)
            except Exception as e:
                print(f"Error processing mask {i}: {str(e)}")
        
        return sam_masks
    
    # Original sequential processing for base64 or fallback
    for i, mask_item in enumerate(mask_data):
        try:
            if use_base64:
                # Extract base64 data from data URL
                if mask_item.startswith('data:image'):
                    header, data = mask_item.split(',', 1)
                    mask_bytes = base64.b64decode(data)
                    mask_img = Image.open(BytesIO(mask_bytes))
                else:
                    print(f"Invalid base64 data URL format for mask {i}")
                    continue
            
            # Convert to numpy array
            mask_np = np.array(mask_img)
            
            # If RGB, convert to grayscale
            if len(mask_np.shape) == 3:
                # Use any channel that has variation (they should all be the same for masks)
                mask_np = mask_np[:, :, 0]
            
            # Convert to binary mask
            binary_mask = mask_np > 127  # Threshold at middle value
            
            # Find bounding box
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            
            if not rows.any() or not cols.any():
                continue  # Skip empty masks
                
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Create SAM-format mask dict
            sam_mask = {
                'segmentation': binary_mask,
                'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'area': int(binary_mask.sum()),
                'predicted_iou': 0.9,  # Default high IoU since these are from SAM2
                'stability_score': 0.9,  # Default high stability
                'mask_index': i
            }
            
            sam_masks.append(sam_mask)
            
        except Exception as e:
            print(f"Error processing mask {i}: {str(e)}")
    
    return sam_masks


def process_with_fal(image_path, is_person_cropped=False):
    """
    Process image using FAL API for SAM2 segmentation
    
    Args:
        image_path: Path to the image file
        is_person_cropped: Whether the image is already cropped to person by MediaPipe
        
    Returns:
        tuple: (masks, processing_time)
    """
    print("Processing with FAL API...")
    
    # Check if FAL API key is set
    if not os.environ.get('FAL_KEY'):
        raise Exception("FAL_KEY environment variable not set. Get your token from https://fal.ai")
    
    # Log the image being processed
    img = cv2.imread(image_path)
    print(f"🎯 SAM2 processing image from: {image_path}")
    print(f"🎯 Image dimensions: {img.shape if img is not None else 'Unknown'}")
    print(f"🎯 Person pre-cropped: {is_person_cropped}")
    
    try:
        # Get active configuration
        config = get_active_sam2_config()
        
        # Log the configuration being used
        print(f"\n=== Using SAM2 Configuration ===")
        print(f"Key parameters:")
        print(f"  - points_per_side: {config.get('points_per_side', 32)}")
        print(f"  - pred_iou_thresh: {config.get('pred_iou_thresh', 0.88)}")
        print(f"  - stability_score_thresh: {config.get('stability_score_thresh', 0.95)}")
        print(f"  - min_mask_region_area: {config.get('min_mask_region_area', 100)}")
        print(f"=======================\n")
        
        # Upload image to a temporary hosting service or use base64
        # For FAL, we need to provide an image URL
        # Option 1: Convert to base64 data URL
        img_data = file_to_base64(image_path)
        img_url = f"data:image/png;base64,{img_data}"
        
        # Prepare FAL API request
        headers = {
            "Authorization": f"Key {os.environ.get('FAL_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Map our config to FAL parameters
        fal_params = {
            "image_url": img_url,
            "output_format": "png",
            "points_per_side": config.get("points_per_side", 32),
            "pred_iou_thresh": config.get("pred_iou_thresh", 0.88),
            "stability_score_thresh": config.get("stability_score_thresh", 0.95),
            "min_mask_region_area": config.get("min_mask_region_area", 100),
            "sync_mode": True  # Return base64 data URLs instead of regular URLs
        }
        
        # Run SAM-2 on FAL
        start_time = time.time()
        print("🚀 Sending request to FAL SAM2...")
        
        response = requests.post(
            PROVIDER_CONFIG["fal_endpoint"],
            headers=headers,
            json=fal_params
        )
        
        if response.status_code != 200:
            raise Exception(f"FAL API error: {response.status_code} - {response.text}")
        
        result = response.json()
        sam2_time = time.time() - start_time
        print(f"⏱️  FAL SAM2 API completed in: {sam2_time:.2f} seconds")
        
        # Process the output masks
        masks = []
        
        # FAL returns individual_masks as a list of image objects
        if 'individual_masks' in result and result['individual_masks']:
            # Check if masks are base64 data URLs (sync_mode=true) or regular URLs
            first_mask = result['individual_masks'][0] if result['individual_masks'] else {}
            
            # Check if we have 'url' or direct data
            if 'url' in first_mask and first_mask['url'].startswith('data:image'):
                # Base64 data URLs
                mask_data = [mask['url'] for mask in result['individual_masks'] if 'url' in mask]
                use_base64 = True
                print(f"Found {len(mask_data)} base64 mask data URLs from FAL")
            elif 'url' in first_mask:
                # Regular URLs
                mask_data = [mask['url'] for mask in result['individual_masks'] if 'url' in mask]
                use_base64 = False
                print(f"Found {len(mask_data)} mask URLs from FAL")
            else:
                # Try to find any data field that might contain the mask
                mask_data = []
                use_base64 = False
                print("Unexpected FAL response format for masks")
            
            # Convert FAL masks to SAM format
            if mask_data and img is not None:
                sam_format_masks = convert_fal_masks_to_sam_format(mask_data, img.shape, use_base64)
                print(f"Converted {len(sam_format_masks)} masks to SAM format")
                return sam_format_masks, sam2_time
            else:
                return [], sam2_time
        else:
            print("No individual masks found in FAL response")
            return [], sam2_time
            
    except Exception as e:
        print(f"FAL API error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e