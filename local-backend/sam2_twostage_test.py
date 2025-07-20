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

# Stage 1: Light SAM2 config to find person
SAM2_PERSON_CONFIG = {
    "points_per_side": 16,  # Less points for faster processing
    "points_per_batch": 64,
    "pred_iou_thresh": 0.9,  # High threshold for confident segments
    "stability_score_thresh": 0.95,  # Very stable segments only
    "stability_score_offset": 0.8,
    "mask_threshold": 0.0,
    "box_nms_thresh": 0.7,  # Less aggressive NMS
    "crop_n_layers": 0,  # No multi-scale
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 0.3,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 1000,  # Large segments only
    "multimask_output": False,
    "use_m2m": False,
}

# Stage 2: Detailed SAM2 config for clothing
SAM2_CLOTHING_CONFIG = {
    "points_per_side": 64,  # Dense points for detailed segmentation
    "points_per_batch": 128,
    "pred_iou_thresh": 0.6,  # Lower threshold for more segments
    "stability_score_thresh": 0.85,  # More permissive
    "stability_score_offset": 0.7,
    "mask_threshold": -0.5,
    "box_nms_thresh": 0.3,  # Keep overlapping items
    "crop_n_layers": 1,  # Multi-scale for details
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 0.4,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,  # Small items too
    "multimask_output": True,
    "use_m2m": True,
}

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Two-Stage SAM2 + CLIP Testing</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .upload-area { 
            border: 2px dashed #ccc; 
            padding: 20px; 
            text-align: center; 
            margin: 20px 0;
        }
        .stage-info {
            background: #f0f8ff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
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
        .person-preview {
            max-width: 400px;
            margin: 20px auto;
            text-align: center;
        }
        .person-preview img {
            max-width: 100%;
            border: 3px solid #2196F3;
            border-radius: 8px;
        }
        #loading { display: none; color: blue; }
        .stats { background: #e8f4f8; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Two-Stage SAM2 + CLIP Testing</h1>
        
        <div class="stage-info">
            <h3>How it works:</h3>
            <ol>
                <li><strong>Stage 1:</strong> Run SAM2 with light config to find the person (largest segment)</li>
                <li><strong>Stage 2:</strong> Crop to person and run detailed SAM2 for clothing segmentation</li>
                <li><strong>Stage 3:</strong> Classify all segments with CLIP</li>
            </ol>
        </div>
        
        <div class="upload-area">
            <input type="file" id="imageFile" accept="image/*">
            <button onclick="processImage()">Process Image</button>
            <div id="loading">Processing... this may take a while...</div>
        </div>
        
        <div id="personPreview" class="person-preview" style="display:none;">
            <h3>Stage 1: Top 3 Segments by Area</h3>
            <div id="topSegments" style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
            </div>
            <p id="personInfo"></p>
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
            document.getElementById('personPreview').style.display = 'none';
            
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
                
                // Show top segments preview
                if (data.top_segments && data.top_segments.length > 0) {
                    document.getElementById('personPreview').style.display = 'block';
                    const segmentsDiv = document.getElementById('topSegments');
                    segmentsDiv.innerHTML = '';
                    
                    data.top_segments.forEach((seg, idx) => {
                        const segDiv = document.createElement('div');
                        segDiv.style = 'text-align: center; flex: 1; min-width: 200px;';
                        segDiv.innerHTML = `
                            <h4>${idx === 0 ? '🏆 Selected' : '#' + (idx + 1)}</h4>
                            <img src="data:image/png;base64,${seg.image}" style="max-width: 100%; border: 3px solid ${idx === 0 ? '#4CAF50' : '#ddd'}; border-radius: 8px;">
                            <p><strong>${seg.area.toLocaleString()} pixels</strong></p>
                            <p style="font-size: 0.9em; color: #666;">${seg.classification || 'Unclassified'}</p>
                        `;
                        segmentsDiv.appendChild(segDiv);
                    });
                    
                    document.getElementById('personInfo').textContent = 
                        `Stage 1 found ${data.stage1_mask_count} total masks. Using segment #1 with ${data.person_area.toLocaleString()} pixels`;
                }
                
                // Show stats
                document.getElementById('stats').style.display = 'block';
                document.getElementById('statsContent').innerHTML = `
                    <p><strong>Stage 1:</strong> Person detection - ${data.stage1_time}s (${data.stage1_mask_count} masks)</p>
                    <p><strong>Stage 2:</strong> Clothing segmentation - ${data.stage2_time}s (${data.stage2_mask_count} masks)</p>
                    <p><strong>Stage 3:</strong> CLIP classification - ${data.clip_time}s</p>
                    <p><strong>Total time:</strong> ${data.total_time}s</p>
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
    return render_template_string(HTML_TEMPLATE)

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
        
        # Load image
        img = Image.open(temp_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        print("\n=== STAGE 1: Find Person ===")
        stage1_start = time.time()
        
        # Override SAM2 config for person detection
        import services.sam2_service as sam2_module
        original_config = sam2_module.SAM2_CONFIG.copy()
        sam2_module.SAM2_CONFIG.update(SAM2_PERSON_CONFIG)
        
        try:
            # Run SAM2 to find person
            person_masks, sam2_api_time = process_with_fal(temp_path)
            stage1_time = time.time() - stage1_start
            print(f"Stage 1 completed in {stage1_time:.2f}s, found {len(person_masks)} masks")
            
            # Sort masks by area and get top 3
            sorted_masks = sorted(person_masks, key=lambda m: m.get('area', 0), reverse=True)
            top_masks = sorted_masks[:3] if len(sorted_masks) >= 3 else sorted_masks
            
            if not top_masks:
                return jsonify({'error': 'No segments found in image'}), 400
            
            # Quick classify top 3 to help identify person
            top_segments_data = []
            for i, mask in enumerate(top_masks):
                mask_array = mask['segmentation']
                masked_img = img_array.copy()
                masked_img[mask_array == 0] = 255  # White background
                
                # Create preview
                preview_img = Image.fromarray(masked_img)
                preview_img.thumbnail((300, 300))
                buffer = io.BytesIO()
                preview_img.save(buffer, format='PNG')
                preview_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Quick classification
                try:
                    result = classify_batch_with_clip([masked_img])
                    classification = result[0].get('label', 'unknown') if result else 'unknown'
                except:
                    classification = 'unknown'
                
                top_segments_data.append({
                    'image': preview_base64,
                    'area': int(mask.get('area', 0)),
                    'classification': classification
                })
                
                print(f"Segment #{i+1}: Area={mask.get('area', 0)}, Classification={classification}")
            
            # For now, still use the largest (but we could add logic to pick "person" classified one)
            largest_mask = top_masks[0]
            largest_area = largest_mask.get('area', 0)
            
            print(f"Using largest segment with area: {largest_area} pixels")
            
            # Crop image to person bounding box
            mask_array = largest_mask['segmentation']
            bbox = largest_mask.get('bbox')
            
            if bbox:
                x, y, w, h = bbox
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_array.shape[1] - x, w + 2 * padding)
                h = min(img_array.shape[0] - y, h + 2 * padding)
                
                # Crop the image
                person_crop = img_array[y:y+h, x:x+w]
                
                # Also crop the mask
                person_mask = mask_array[y:y+h, x:x+w]
                
                # Apply mask to remove background
                person_crop[person_mask == 0] = [255, 255, 255]  # White background
            else:
                # Use full image with mask
                person_crop = img_array.copy()
                person_crop[mask_array == 0] = [255, 255, 255]
            
            # Save cropped person for stage 2
            person_img = Image.fromarray(person_crop)
            person_path = '/tmp/person_crop.jpg'
            person_img.save(person_path)
            
            # Create preview for UI
            preview_buffer = io.BytesIO()
            person_img.save(preview_buffer, format='PNG')
            person_preview_base64 = base64.b64encode(preview_buffer.getvalue()).decode()
            
        finally:
            # Restore original config
            sam2_module.SAM2_CONFIG.update(original_config)
        
        print("\n=== STAGE 2: Segment Clothing ===")
        stage2_start = time.time()
        
        # Override config for detailed segmentation
        sam2_module.SAM2_CONFIG.update(SAM2_CLOTHING_CONFIG)
        
        try:
            # Run detailed SAM2 on cropped person
            clothing_masks, sam2_api_time2 = process_with_fal(person_path)
            stage2_time = time.time() - stage2_start
            print(f"Stage 2 completed in {stage2_time:.2f}s, found {len(clothing_masks)} masks")
            
        finally:
            # Restore original config
            sam2_module.SAM2_CONFIG.update(original_config)
        
        print("\n=== STAGE 3: Classify with CLIP ===")
        clip_start = time.time()
        
        # Extract image crops from masks
        mask_images = []
        valid_masks = []
        
        for mask in clothing_masks:
            try:
                mask_seg = mask['segmentation']
                masked_img = person_crop.copy()
                masked_img[mask_seg == 0] = 0
                
                if 'bbox' in mask:
                    x, y, w, h = mask['bbox']
                    crop = masked_img[y:y+h, x:x+w]
                    if crop.size > 0:
                        mask_images.append(crop)
                        valid_masks.append(mask)
                else:
                    mask_images.append(masked_img)
                    valid_masks.append(mask)
                    
            except Exception as e:
                print(f"Error processing mask: {e}")
                continue
        
        # Classify all masks
        if mask_images:
            classification_results = classify_batch_with_clip(mask_images)
            clip_time = time.time() - clip_start
            print(f"CLIP classified {len(classification_results)} masks in {clip_time:.2f}s")
            
            # Combine results
            classified_masks = []
            for i, (mask, result) in enumerate(zip(valid_masks, classification_results)):
                mask['label'] = result.get('label', 'unknown')
                mask['score'] = result.get('confidence', 0.0)
                classified_masks.append(mask)
        else:
            classified_masks = []
            clip_time = time.time() - clip_start
        
        # Prepare response with colored segments
        clothing_priorities = {
            'shirt': 1, 't-shirt': 1, 'top': 1, 'blouse': 1, 'jacket': 1, 'coat': 1, 'sweater': 1,
            'pants': 2, 'jeans': 2, 'trousers': 2, 'shorts': 2, 'skirt': 2, 'dress': 2,
            'shoes': 3, 'sneakers': 3, 'boots': 3, 'sandals': 3, 'heels': 3,
            'hat': 4, 'cap': 4, 'bag': 5, 'backpack': 5, 'purse': 5,
            'watch': 6, 'glasses': 6, 'jewelry': 6, 'belt': 6
        }
        
        results = []
        for i, mask in enumerate(classified_masks):
            mask_array = mask['segmentation'].astype(np.uint8)
            
            # Create colored visualization
            color = np.array([
                (i * 67) % 255,
                (i * 123) % 255,  
                (i * 189) % 255
            ], dtype=np.uint8)
            
            colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
            colored_mask[mask_array > 0] = color
            
            masked_region = person_crop.copy()
            blend_alpha = 0.5
            masked_region[mask_array > 0] = (
                blend_alpha * masked_region[mask_array > 0] + 
                (1 - blend_alpha) * colored_mask[mask_array > 0]
            ).astype(np.uint8)
            
            if 'bbox' in mask:
                x, y, w, h = mask['bbox']
                cropped = masked_region[y:y+h, x:x+w]
                cropped = cv2.copyMakeBorder(cropped, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color.tolist())
            else:
                cropped = masked_region
            
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
        
        # Sort by priority
        results.sort(key=lambda x: (x['priority'], -x['score']))
        
        total_time = time.time() - start_time
        
        # Clean up
        os.remove(temp_path)
        os.remove(person_path)
        
        return jsonify({
            'masks': results,
            'top_segments': top_segments_data,
            'person_crop': person_preview_base64,
            'person_area': int(largest_area),
            'stage1_mask_count': len(person_masks),
            'stage2_mask_count': len(clothing_masks),
            'stage1_time': round(stage1_time, 2),
            'stage2_time': round(stage2_time, 2),
            'clip_time': round(clip_time, 2),
            'total_time': round(total_time, 2)
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Two-Stage SAM2 + CLIP Testing Server")
    print("Stage 1: Light SAM2 to find person")
    print("Stage 2: Detailed SAM2 on person crop")
    print("Stage 3: CLIP classification")
    print("Running on http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True)