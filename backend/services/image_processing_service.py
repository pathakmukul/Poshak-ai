# Image processing service for file operations
import os
import base64
import pickle
import numpy as np
from PIL import Image
from datetime import datetime
from .visualization_service import create_clothing_visualization




def file_to_base64(file_path):
    """Convert file content directly to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Helper functions for mask storage
def save_mask_images(image_path, image_rgb, masks):
    """Save mask images as PNG files for each clothing type"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        base_name = os.path.basename(image_path).split('.')[0]
        masks_dir = os.path.join(data_dir, 'saved_masks', base_name)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Save masks for each clothing type
        clothing_types = ['shirt', 'pants', 'shoes']
        
        for clothing_type in clothing_types:
            mask_img, count = create_clothing_visualization(image_rgb, masks, clothing_type, for_gemini=True)
            
            if count > 0:
                mask_filename = f"mask_{clothing_type}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                mask_pil = Image.fromarray(mask_img)
                mask_pil.save(mask_path)
                print(f"Saved {clothing_type} mask to {os.path.join(base_name, mask_filename)}")
        
        # Save the raw masks data
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
    """Load masks from saved files"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for data directory in parent (react-app) folder
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        if not data_dir:
            return None, None
            
        base_name = os.path.basename(image_path).split('.')[0]
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
