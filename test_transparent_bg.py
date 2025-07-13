#!/usr/bin/env python3
"""Test script to verify transparent background implementation"""

import requests
import base64
import json
from PIL import Image
from io import BytesIO
import numpy as np

def test_transparent_background():
    """Test that cropped images have transparent backgrounds"""
    
    # Test endpoint
    url = "http://localhost:5000/process"
    
    # Create a simple test image (red square on blue background)
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    test_img[:, :] = [0, 0, 255]  # Blue background
    test_img[50:150, 50:150] = [255, 0, 0]  # Red square
    
    # Convert to base64
    pil_img = Image.fromarray(test_img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    img_data = f"data:image/png;base64,{img_base64}"
    
    print("Testing transparent background implementation...")
    
    # Make request
    response = requests.post(url, json={
        "image_data": img_data,
        "model_size": "replicate"
    })
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    
    # Check if masks have cropped images
    if 'masks' in result and len(result['masks']) > 0:
        mask = result['masks'][0]
        if 'cropped_img' in mask:
            # Decode the cropped image
            crop_data = base64.b64decode(mask['cropped_img'])
            crop_img = Image.open(BytesIO(crop_data))
            
            # Check if image has alpha channel
            if crop_img.mode == 'RGBA':
                print("✓ Cropped image has alpha channel (RGBA)")
                
                # Convert to numpy to check transparency
                crop_array = np.array(crop_img)
                alpha_channel = crop_array[:, :, 3]
                
                # Check if there are transparent pixels (alpha = 0)
                transparent_pixels = np.sum(alpha_channel == 0)
                opaque_pixels = np.sum(alpha_channel == 255)
                
                print(f"  - Transparent pixels: {transparent_pixels}")
                print(f"  - Opaque pixels: {opaque_pixels}")
                
                if transparent_pixels > 0:
                    print("✓ Image has transparent background!")
                else:
                    print("✗ No transparent pixels found")
            else:
                print(f"✗ Image mode is {crop_img.mode}, not RGBA")
    
    # Check closet visualizations
    if 'closet_visualizations' in result:
        print("\n✓ Closet visualizations found in response")
        for key in result['closet_visualizations']:
            print(f"  - {key}")
    else:
        print("\n✗ No closet visualizations in response")

if __name__ == "__main__":
    test_transparent_background()