from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import os
import sys
import numpy as np
from PIL import Image
import io
import time
import json
import cv2

# Add backend to path to import services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import SAM2 and CLIP services
from services.sam2_service import process_with_fal, convert_fal_masks_to_sam_format
from clip_classifier import classify_batch_with_clip
from config_improved import CLASSIFICATION_CONFIG

app = Flask(__name__)
CORS(app)

# Configurable SAM2 parameters - EDIT THESE TO TEST!
SAM2_TEST_CONFIG = {
    "points_per_side": 64,  # Try: 16, 32, 64, 128
    "points_per_batch": 128,
    "pred_iou_thresh": 0.6,  # Try: 0.3, 0.5, 0.7, 0.9
    "stability_score_thresh": 0.85,  # Try: 0.7, 0.8, 0.9, 0.95
    "stability_score_offset": 0.7,
    "mask_threshold": -0.5,  # Try: -1.0, 0.0, 0.5
    "box_nms_thresh": 0.3,  # Try: 0.1, 0.3, 0.5, 0.7
    "crop_n_layers": 1,  # Try: 0, 1, 2
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 0.4,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,  # Try: 50, 100, 500
    "multimask_output": True,
    "use_m2m": True,
}

# HTML template for testing interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>SAM2 + CLIP Testing</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .upload-area { 
            border: 2px dashed #ccc; 
            padding: 20px; 
            text-align: center; 
            margin: 20px 0;
        }
        .results { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-top: 20px;
        }
        .mask-item {
            border: 2px solid #ddd;
            padding: 10px;
            text-align: center;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        .mask-item:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .mask-item img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 4px;
        }
        .mask-item.clothing {
            border-color: #4CAF50;
            background-color: #f1f8f4;
        }
        .mask-item.clothing h4 {
            color: #2e7d32;
            margin: 10px 0 5px 0;
        }
        .config { 
            background: #f5f5f5; 
            padding: 15px; 
            margin: 20px 0;
            border-radius: 5px;
        }
        .config pre { 
            background: white; 
            padding: 10px; 
            overflow-x: auto;
        }
        #loading { display: none; color: blue; }
        .stats { background: #e8f4f8; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SAM2 + CLIP Local Testing</h1>
        
        <div class="config">
            <h3>Current SAM2 Config (edit sam2_clip_test.py to change):</h3>
            <pre>{{ config }}</pre>
        </div>
        
        <div class="upload-area">
            <input type="file" id="imageFile" accept="image/*">
            <button onclick="processImage()">Process Image</button>
            <div id="loading">Processing... this may take a while...</div>
        </div>
        
        <div id="stats" class="stats" style="display:none;">
            <h3>Processing Stats:</h3>
            <div id="statsContent"></div>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        async function processImage() {
            const fileInput = document.getElementById('imageFile');
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select an image');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            document.getElementById('stats').style.display = 'none';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Show stats
                document.getElementById('stats').style.display = 'block';
                document.getElementById('statsContent').innerHTML = `
                    <p>Total masks generated: ${data.total_masks}</p>
                    <p>SAM2 time: ${data.sam2_time}s</p>
                    <p>CLIP classification time: ${data.clip_time}s</p>
                    <p>Total processing time: ${data.total_time}s</p>
                `;
                
                // Show results
                const resultsDiv = document.getElementById('results');
                data.masks.forEach((mask, idx) => {
                    const maskDiv = document.createElement('div');
                    const isClothing = mask.priority <= 6;
                    maskDiv.className = isClothing ? 'mask-item clothing' : 'mask-item';
                    maskDiv.innerHTML = `
                        <img src="data:image/png;base64,${mask.mask_image}" alt="Mask ${idx}">
                        <h4>${mask.label}</h4>
                        <p>Confidence: ${(mask.score * 100).toFixed(1)}%</p>
                        <p style="font-size: 0.9em; color: #666;">Area: ${mask.area.toLocaleString()} px</p>
                    `;
                    resultsDiv.appendChild(maskDiv);
                });
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error processing image: ' + error);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    config_str = json.dumps(SAM2_TEST_CONFIG, indent=2)
    return render_template_string(HTML_TEMPLATE, config=config_str)

@app.route('/process', methods=['POST'])
def process():
    try:
        start_time = time.time()
        
        # Get uploaded image
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Save temporarily
        temp_path = '/tmp/test_image.jpg'
        image_file.save(temp_path)
        
        # Load image to get dimensions
        img = Image.open(temp_path)
        # Convert to RGB if needed (remove alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        # Process with SAM2 using FAL
        print(f"Processing with SAM2 config: {json.dumps(SAM2_TEST_CONFIG, indent=2)}")
        sam2_start = time.time()
        
        # Temporarily override the config
        import services.sam2_service as sam2_module
        original_config = sam2_module.SAM2_CONFIG.copy()
        sam2_module.SAM2_CONFIG.update(SAM2_TEST_CONFIG)
        
        try:
            # Use the real process_with_fal function
            all_masks, sam2_api_time = process_with_fal(temp_path)
            sam2_time = time.time() - sam2_start
            print(f"SAM2 returned {len(all_masks)} masks in {sam2_time:.2f}s (API time: {sam2_api_time:.2f}s)")
            
        finally:
            # Restore original config
            sam2_module.SAM2_CONFIG.update(original_config)
        
        # Classify with CLIP
        clip_start = time.time()
        
        # Extract image crops from masks for classification
        mask_images = []
        valid_masks = []
        
        for mask in all_masks:
            try:
                # Get the mask segmentation
                mask_seg = mask['segmentation']
                
                # Apply mask to original image
                masked_img = img_array.copy()
                masked_img[mask_seg == 0] = 0
                
                # Crop to bounding box if available
                if 'bbox' in mask:
                    x, y, w, h = mask['bbox']
                    crop = masked_img[y:y+h, x:x+w]
                    if crop.size > 0:
                        mask_images.append(crop)
                        valid_masks.append(mask)
                else:
                    # Use full masked image
                    mask_images.append(masked_img)
                    valid_masks.append(mask)
                    
            except Exception as e:
                print(f"Error processing mask for classification: {e}")
                continue
        
        # Classify all masks
        if mask_images:
            classification_results = classify_batch_with_clip(mask_images)
            clip_time = time.time() - clip_start
            print(f"CLIP classified {len(classification_results)} masks in {clip_time:.2f}s")
            
            # Combine mask data with classification results
            classified_masks = []
            for i, (mask, result) in enumerate(zip(valid_masks, classification_results)):
                mask['label'] = result.get('label', 'unknown')
                mask['score'] = result.get('confidence', 0.0)
                classified_masks.append(mask)
        else:
            classified_masks = []
            clip_time = time.time() - clip_start
            print("No masks to classify")
        
        # Define clothing priorities
        clothing_priorities = {
            'shirt': 1, 't-shirt': 1, 'top': 1, 'blouse': 1, 'jacket': 1, 'coat': 1, 'sweater': 1,
            'pants': 2, 'jeans': 2, 'trousers': 2, 'shorts': 2, 'skirt': 2, 'dress': 2,
            'shoes': 3, 'sneakers': 3, 'boots': 3, 'sandals': 3, 'heels': 3,
            'hat': 4, 'cap': 4, 'bag': 5, 'backpack': 5, 'purse': 5,
            'watch': 6, 'glasses': 6, 'jewelry': 6, 'belt': 6
        }
        
        # Prepare response with colored segments
        results = []
        for i, mask in enumerate(classified_masks):
            # Create colored segment overlay
            mask_array = mask['segmentation'].astype(np.uint8)
            
            # Create RGB image with random color for this segment
            color = np.array([
                (i * 67) % 255,  # Red channel
                (i * 123) % 255,  # Green channel  
                (i * 189) % 255   # Blue channel
            ], dtype=np.uint8)
            
            # Create colored mask
            colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
            colored_mask[mask_array > 0] = color
            
            # Blend with original image
            masked_region = img_array.copy()
            
            # Ensure we're working with RGB (3 channels)
            if masked_region.shape[-1] == 4:
                masked_region = masked_region[:, :, :3]
            
            blend_alpha = 0.5
            masked_region[mask_array > 0] = (
                blend_alpha * masked_region[mask_array > 0] + 
                (1 - blend_alpha) * colored_mask[mask_array > 0]
            ).astype(np.uint8)
            
            # Crop to bounding box if available
            if 'bbox' in mask:
                x, y, w, h = mask['bbox']
                cropped = masked_region[y:y+h, x:x+w]
                # Also add a border to show the segment clearly
                cropped = cv2.copyMakeBorder(cropped, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color.tolist())
            else:
                cropped = masked_region
            
            # Convert to PIL and base64
            mask_pil = Image.fromarray(cropped)
            buffer = io.BytesIO()
            mask_pil.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            label = mask.get('label', 'unknown')
            priority = clothing_priorities.get(label, 999)
            
            results.append({
                'mask_image': mask_base64,
                'label': label,
                'score': float(mask.get('score', 0)),
                'area': int(mask.get('area', 0)),
                'priority': priority
            })
        
        # Sort by priority (clothing items first)
        results.sort(key=lambda x: (x['priority'], -x['score']))
        
        total_time = time.time() - start_time
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'masks': results,
            'total_masks': len(results),
            'sam2_time': round(sam2_time, 2),
            'clip_time': round(clip_time, 2),
            'total_time': round(total_time, 2)
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("SAM2 + CLIP Testing Server")
    print("Edit SAM2_TEST_CONFIG in this file to test different configurations")
    print("Running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)