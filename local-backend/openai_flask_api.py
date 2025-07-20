from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
import json
from datetime import datetime
import tempfile
import time

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
# Replace with your actual OpenAI API key
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Base directory for images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENAI_TEST_DIR = os.path.join(BASE_DIR, 'openai-test')

@app.route('/api/list-images/<category>', methods=['GET'])
def list_images(category):
    """List all images in a specific category folder"""
    try:
        category_path = os.path.join(OPENAI_TEST_DIR, category)
        
        if not os.path.exists(category_path):
            return jsonify({'images': []})
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        
        for file in os.listdir(category_path):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                # Return relative path from openai-test folder
                images.append(f'/openai-test/{category}/{file}')
        
        return jsonify({'images': sorted(images)})
    
    except Exception as e:
        print(f"Error listing images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/openai-outfit-variation', methods=['POST'])
def openai_outfit_variation():
    """Generate outfit variations using OpenAI's image API"""
    try:
        data = request.get_json()
        selections = data.get('selections', [])
        user_id = data.get('userId')
        
        if not selections:
            return jsonify({'error': 'No selections provided'}), 400
        
        # Find the model image (base image)
        model_selection = next((s for s in selections if s['category'] == 'MODEL'), None)
        if not model_selection:
            return jsonify({'error': 'Model image is required'}), 400
        
        # Load the model image
        model_path = os.path.join(BASE_DIR, model_selection['imagePath'].lstrip('/'))
        with open(model_path, 'rb') as f:
            model_image = f.read()
        
        # Prepare input images as list of tuples (like in the docs)
        input_images = [('model.png', model_image, 'image/png')]
        
        for selection in selections:
            if selection['category'] != 'MODEL':
                # Load each clothing item image
                item_path = os.path.join(BASE_DIR, selection['imagePath'].lstrip('/'))
                with open(item_path, 'rb') as f:
                    item_image = f.read()
                    filename = f"{selection['category'].lower()}.png"
                    input_images.append((filename, item_image, 'image/png'))
        
        # Build prompt similar to the docs examples
        selected_categories = [s['category'] for s in selections if s['category'] != 'MODEL']
        
        if len(selected_categories) == 1:
            # Single item addition
            category = selected_categories[0].lower()
            if category == 'accessories':
                prompt = "Add the accessory to the outfit."
            else:
                prompt = f"Replace the existing {category} with the {category} from the provided image."
        else:
            # Multiple items
            items = []
            for cat in selected_categories:
                if cat == 'SHIRT':
                    items.append("the shirt")
                elif cat == 'PANT':
                    items.append("the pants")
                elif cat == 'SHOES':
                    items.append("the shoes")
                elif cat == 'Accessories':
                    items.append("the accessories")
            
            prompt = f"Update the outfit with {', '.join(items)} from the provided images."
        
        # Debug: Print what we're sending
        print("\n=== OPENAI API PAYLOAD ===")
        print(f"Model image: {model_selection['imagePath']}")
        print(f"Other images: {[s['imagePath'] for s in selections if s['category'] != 'MODEL']}")
        print(f"Prompt: {prompt}")
        print(f"Input images count: {len(input_images)}")
        print("========================\n")
        
        # Time the API call
        start_time = time.time()
        
        # Call OpenAI API with high input fidelity
        # If we have multiple images, pass them all, otherwise just the model
        if len(input_images) > 1:
            result = client.images.edit(
                model="gpt-image-1",
                image=input_images,  # Pass all images
                prompt=prompt,
                input_fidelity="high",
                quality="high",
                output_format="jpeg"
            )
        else:
            result = client.images.edit(
                model="gpt-image-1",
                image=input_images[0],  # Just the model
                prompt=prompt,
                input_fidelity="high",
                quality="high",
                output_format="jpeg"
            )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        print(f"OpenAI generation time: {generation_time:.2f} seconds")
        
        # Get the generated image
        image_base64 = result.data[0].b64_json
        
        # Save the image temporarily and return URL
        # In production, you'd upload to Firebase Storage or similar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'openai_output_{user_id}_{timestamp}.png'
        output_path = os.path.join(BASE_DIR, 'temp', output_filename)
        
        # Create temp directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        image_bytes = base64.b64decode(image_base64)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        
        # Return the URL with generation time
        return jsonify({
            'success': True,
            'imageUrl': f'/temp/{output_filename}',
            'timestamp': timestamp,
            'generationTime': f"{generation_time:.2f} seconds"
        })
    
    except Exception as e:
        print(f"Error in outfit variation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/openai-test/<category>/<filename>')
def serve_test_image(category, filename):
    """Serve images from the openai-test directory"""
    try:
        file_path = os.path.join(OPENAI_TEST_DIR, category, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Determine content type
            ext = filename.lower().split('.')[-1]
            content_type = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'webp': 'image/webp'
            }.get(ext, 'image/jpeg')
            
            return image_data, 200, {'Content-Type': content_type}
        else:
            return 'Image not found', 404
    except Exception as e:
        return str(e), 500

@app.route('/temp/<filename>')
def serve_temp_image(filename):
    """Serve temporary generated images"""
    try:
        file_path = os.path.join(BASE_DIR, 'temp', filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                image_data = f.read()
            return image_data, 200, {'Content-Type': 'image/png'}
        else:
            return 'Image not found', 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)