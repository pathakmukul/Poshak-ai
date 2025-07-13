"""
Gemini Service for Virtual Try-On functionality

This service handles three main endpoints:
1. get_gemini_data - Used by App.js for local images with masks in memory
2. prepare_wardrobe_gemini - Used by Wardrobe.js for Firebase images with stored masks
3. gemini_tryon - Used by both App.js and Wardrobe.js for the actual try-on

The key difference:
- App.js uses get_gemini_data because it has masks in memory from processing
- Wardrobe.js uses prepare_wardrobe_gemini because it loads masks from Firebase storage
"""

# Removed Flask dependencies - this is now a pure service class
import cv2
import os
import numpy as np
import base64
from io import BytesIO
import io
from PIL import Image
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Google GenerativeAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI package not installed. Run: pip install google-generativeai")


class GeminiService:
    def __init__(self):
        """Initialize Gemini service with API key"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
    
    def get_gemini_data(self, image_path, clothing_type, last_masks, last_image_rgb=None):
        """Get original image and mask for Gemini (no overlay, just raw data)
        
        Args:
            image_path: Path to the image file
            clothing_type: Type of clothing to extract (shirt, pants, shoes)
            last_masks: List of mask data from previous processing
            last_image_rgb: Optional pre-loaded image in RGB format
            
        Returns:
            dict: Success response with base64 images or error response
            int: HTTP status code
        """
        try:
            # Get the stored masks from the last process call
            if not last_masks:
                return {'error': 'No masks available. Generate masks first.'}, 400
            
            # Use stored image if available, otherwise load from path
            if last_image_rgb is not None:
                image_rgb = last_image_rgb
            elif image_path:
                # Load original image from path
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                full_image_path = os.path.join(base_dir, image_path)
                image = cv2.imread(full_image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return {'error': 'No image available. Process an image first.'}, 400
            
            # Filter for specific clothing type
            if clothing_type == "shirt":
                type_labels = ["shirt", "t-shirt", "hoodie"]
            elif clothing_type == "pants":
                type_labels = ["pants", "jeans", "trousers"]
            elif clothing_type == "shoes":
                type_labels = ["shoes", "sneakers"]
            else:
                type_labels = [clothing_type]
            
            clothing_masks = [m for m in last_masks if m.get('label', '') in type_labels]
            
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
                
                return {
                    'success': True,
                    'original_image': original_base64,
                    'mask_image': mask_base64
                }, 200
            else:
                return {'error': f'No {clothing_type} mask found'}, 400
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}, 500

    def prepare_wardrobe_gemini(self, image_url, mask_data, clothing_type):
        """Prepare wardrobe item for Gemini try-on using stored masks
        
        Args:
            image_url: Firebase Storage URL of the image
            mask_data: Stored mask data from Firebase
            clothing_type: Type of clothing to extract
            
        Returns:
            dict: Success response with base64 images or error response
            int: HTTP status code
        """
        try:
            if not image_url or not mask_data:
                return {'error': 'Missing image_url or mask_data'}, 400
                
            # Download image from Firebase
            print(f"Downloading wardrobe image from: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Convert to numpy array
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Failed to decode image from URL'}, 500
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get the visualization for the selected clothing type
            visualizations = mask_data.get('visualizations', {})
            mask_base64 = visualizations.get(clothing_type)
            
            if not mask_base64:
                return {'error': f'No {clothing_type} mask found in stored data'}, 400
                
            # The stored visualization is already a mask image, but we need to convert it to RGBA format
            # Decode the stored mask visualization
            mask_img_data = base64.b64decode(mask_base64)
            mask_pil = Image.open(BytesIO(mask_img_data))
            mask_np = np.array(mask_pil)
            
            # Extract the mask from the visualization
            # The visualization has colored regions for clothing items
            # We need to create a binary mask where clothing pixels are True
            if len(mask_np.shape) == 3:
                # Convert to grayscale if needed
                mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            else:
                mask_gray = mask_np
                
            # Create binary mask where non-zero pixels are clothing
            binary_mask = mask_gray > 0
            
            # Convert to RGBA format for Gemini (transparent where clothing is)
            mask_rgba = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[:, :, 3] = (~binary_mask * 255).astype(np.uint8)  # Inverted for transparency
            mask_pil_rgba = Image.fromarray(mask_rgba, mode='RGBA')
            
            # Convert to base64
            buffer = BytesIO()
            mask_pil_rgba.save(buffer, format="PNG")
            mask_base64_rgba = base64.b64encode(buffer.getvalue()).decode()
            
            # Convert original image to base64
            original_pil = Image.fromarray(image_rgb)
            buffer2 = BytesIO()
            original_pil.save(buffer2, format="PNG")
            original_base64 = base64.b64encode(buffer2.getvalue()).decode()
            
            return {
                'success': True,
                'original_image': original_base64,
                'mask_image': mask_base64_rgba
            }, 200
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}, 500

    def gemini_tryon(self, person_image, mask_image, garment_file, clothing_type='shirt'):
        """Perform virtual try-on using Gemini - EXACT IMPLEMENTATION FROM streamlit_app.py
        
        Args:
            person_image: Base64 encoded original person image
            mask_image: Base64 encoded RGBA mask image
            garment_file: Filename of the garment to try on
            clothing_type: Type of clothing (default: 'shirt')
            
        Returns:
            dict: Success response with result image or error response
            int: HTTP status code
        """
        try:
            if not GEMINI_AVAILABLE:
                return {'error': 'Google GenAI SDK not installed'}, 500
            
            # Get API key
            if not self.api_key:
                return {'error': 'GEMINI_API_KEY not set in .env file'}, 500
            
            # Configure Gemini with IMAGE GENERATION model - EXACT SAME AS streamlit_app.py
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
                            
                            return {
                                'success': True,
                                'result_image': result_base64,
                                'processing_time': processing_time
                            }, 200
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
                                
                                return {
                                    'success': True,
                                    'result_image': result_base64,
                                    'processing_time': processing_time
                                }, 200
                            except:
                                continue
                
                return {'error': 'Gemini returned no image data in response'}, 500
            else:
                return {'error': 'Gemini returned no candidates'}, 500
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'Gemini API error: {str(e)}'}, 500
