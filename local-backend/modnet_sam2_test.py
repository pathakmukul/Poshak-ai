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
from transformers import pipeline

# Add backend to path to import services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import SAM2 and CLIP services (commented out for now)
# from services.sam2_service import process_with_fal, convert_fal_masks_to_sam_format
# from clip_classifier import classify_batch_with_clip
# from config_improved import CLASSIFICATION_CONFIG

app = Flask(__name__)
CORS(app)

# Global segmentation pipeline
segmentation_pipeline = None

def init_segmentation():
    """Initialize segmentation model using Hugging Face transformers"""
    global segmentation_pipeline
    if segmentation_pipeline is None:
        print("Loading segmentation model...")
        # Use a reliable image segmentation model from Hugging Face
        segmentation_pipeline = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")
        print("Segmentation model loaded successfully")
    return segmentation_pipeline

def run_person_segmentation(image):
    """Run segmentation to detect person/clothing - returns raw results"""
    pipeline = init_segmentation()
    
    # Run segmentation
    results = pipeline(image)
    
    return results

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

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>MODNet + SAM2 + CLIP Testing</title>
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
        <h1>Segformer B2 Clothes - Raw Output Test</h1>
        
        <div class="stage-info">
            <h3>Segformer B2 Clothes Model Info:</h3>
            <ul>
                <li><strong>License:</strong> MIT (Free for commercial use)</li>
                <li><strong>Model:</strong> mattmdjaga/segformer_b2_clothes</li>
                <li><strong>Hosting:</strong> Self-hosted, no API costs</li>
                <li><strong>Size:</strong> ~100MB one-time download</li>
            </ul>
            <p><strong>Labels it can detect:</strong> Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt, Shoes, Face, Arms, Legs, Bag, Scarf</p>
        </div>
        
        <div class="upload-area">
            <input type="file" id="imageFile" accept="image/*">
            <button onclick="processImage()">Process Image</button>
            <div id="loading">Processing... this may take a while...</div>
        </div>
        
        <div id="rawOutput" style="display:none;">
            <h3>Raw Model Output:</h3>
            <pre id="rawJson" style="background: #f5f5f5; padding: 10px; overflow-x: auto;"></pre>
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
            document.getElementById('rawOutput').style.display = 'none';
            
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
                
                // Show raw output
                document.getElementById('rawOutput').style.display = 'block';
                document.getElementById('rawJson').textContent = JSON.stringify(data.raw_output, null, 2);
                
                // Show visual results
                const resultsDiv = document.getElementById('results');
                
                // Add summary
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'stats';
                summaryDiv.innerHTML = `
                    <h3>Summary:</h3>
                    <p>Total segments: ${data.total_segments}</p>
                    <p>Processing time: ${data.processing_time}s</p>
                    <p>Detected labels: ${data.unique_labels.join(', ')}</p>
                `;
                resultsDiv.appendChild(summaryDiv);
                
                // Show each segment
                data.segments.forEach((segment, idx) => {
                    const maskDiv = document.createElement('div');
                    maskDiv.className = 'mask-item';
                    maskDiv.innerHTML = `
                        <h4>${segment.label}</h4>
                        <img src="data:image/png;base64,${segment.mask_image}" alt="Mask">
                        <img src="data:image/png;base64,${segment.overlay_image}" alt="Overlay">
                        <p>Score: ${segment.score ? segment.score.toFixed(3) : 'N/A'}</p>
                        <p>Pixels: ${segment.pixel_count.toLocaleString()}</p>
                        <p>Coverage: ${segment.coverage_percent.toFixed(1)}%</p>
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
        
        print("\n=== Running Segformer B2 Clothes Segmentation ===")
        segmentation_start = time.time()
        
        # Run segmentation to get raw results
        results = run_person_segmentation(img)
        segmentation_time = time.time() - segmentation_start
        print(f"Segmentation completed in {segmentation_time:.2f}s")
        print(f"Found {len(results)} segments")
        
        # Process results
        segments = []
        unique_labels = set()
        raw_output = []
        
        for i, result in enumerate(results):
            label = result['label']
            score = result.get('score', None)  # May not have score
            mask = result['mask']
            
            unique_labels.add(label)
            
            # Convert mask to numpy array
            if isinstance(mask, Image.Image):
                mask_array = np.array(mask.convert('L'))
            else:
                mask_array = mask
            
            # Calculate statistics
            pixel_count = np.sum(mask_array > 127)
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            coverage_percent = (pixel_count / total_pixels) * 100
            
            # Create mask visualization (black and white)
            mask_viz = Image.fromarray((mask_array > 127).astype(np.uint8) * 255)
            mask_buffer = io.BytesIO()
            mask_viz.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
            
            # Create overlay visualization
            overlay = img_array.copy()
            mask_bool = mask_array > 127
            # Generate unique color for this segment
            color = np.array([
                (i * 67) % 255,
                (i * 123) % 255,  
                (i * 189) % 255
            ], dtype=np.uint8)
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
            
            overlay_img = Image.fromarray(overlay.astype(np.uint8))
            overlay_buffer = io.BytesIO()
            overlay_img.save(overlay_buffer, format='PNG')
            overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode()
            
            segments.append({
                'label': label,
                'score': float(score) if score is not None else 1.0,
                'pixel_count': int(pixel_count),
                'coverage_percent': float(coverage_percent),
                'mask_image': mask_base64,
                'overlay_image': overlay_base64
            })
            
            # Store raw output info
            raw_output.append({
                'index': i,
                'label': label,
                'score': float(score) if score is not None else 1.0,
                'mask_shape': list(mask_array.shape),
                'pixel_count': int(pixel_count),
                'coverage_percent': float(coverage_percent)
            })
            
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"Segment {i+1}: {label} (score: {score_str}, pixels: {pixel_count:,}, coverage: {coverage_percent:.1f}%)")
        
        processing_time = time.time() - start_time
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'segments': segments,
            'total_segments': len(segments),
            'unique_labels': sorted(list(unique_labels)),
            'processing_time': round(processing_time, 2),
            'raw_output': raw_output,
            'model_info': MODEL_INFO
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Segformer B2 Clothes Model Test")
    print(f"Model: {MODEL_INFO['name']}")
    print(f"License: {MODEL_INFO['license']}")
    print(f"Hosting: {MODEL_INFO['hosting']}")
    print("Initializing segmentation model...")
    init_segmentation()  # Pre-load model
    print("Running on http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=True)