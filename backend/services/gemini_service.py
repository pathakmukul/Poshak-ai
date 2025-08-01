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
import os
import numpy as np
import base64
from io import BytesIO
import io
from PIL import Image
import time
import requests
import json
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
        self.genai = None
        
        if self.api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.genai = genai
            print(f"[GeminiService] Initialized with API key: {self.api_key[:10]}...")
        else:
            print(f"[GeminiService] NOT initialized - API key: {bool(self.api_key)}, Package available: {GEMINI_AVAILABLE}")
    
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
                # Use PIL instead of cv2
                image_pil = Image.open(full_image_path)
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                image_rgb = np.array(image_pil)
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
                
            # Check if we have person_only image in visualizations
            visualizations = mask_data.get('visualizations', {})
            person_only_base64 = visualizations.get('person_only')
            
            if person_only_base64:
                # Use person_only image which was generated at same resolution as masks
                print(f"Using person_only image from visualizations")
                person_img_data = base64.b64decode(person_only_base64)
                person_pil = Image.open(BytesIO(person_img_data))
                
                # Convert RGBA to RGB if needed
                if person_pil.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', person_pil.size, (255, 255, 255))
                    background.paste(person_pil, mask=person_pil.split()[3])
                    person_pil = background
                elif person_pil.mode != 'RGB':
                    person_pil = person_pil.convert('RGB')
                
                image_rgb = np.array(person_pil)
                h, w = image_rgb.shape[:2]
                print(f"\n=== DIMENSION CHECK IN prepare_wardrobe_gemini ===")
                print(f"Using person_only image: {w}x{h} (width x height)")
                print(f"Image shape: {image_rgb.shape}")
            else:
                # Fallback to downloading from URL
                print(f"person_only not found, downloading from: {image_url}")
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Use PIL instead of cv2
                image_pil = Image.open(io.BytesIO(response.content))
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                image_rgb = np.array(image_pil)
                h, w = image_rgb.shape[:2]
                print(f"\n=== DIMENSION CHECK IN prepare_wardrobe_gemini ===")
                print(f"Downloaded image from Firebase: {w}x{h} (width x height)")
                print(f"Image shape: {image_rgb.shape}")
            
            # Get the binary mask for the selected clothing type
            binary_masks = mask_data.get('binary_masks', {})
            mask_base64 = binary_masks.get(clothing_type)
            
            # Fallback to visualizations if binary_masks not available (for backward compatibility)
            if not mask_base64:
                visualizations = mask_data.get('visualizations', {})
                mask_base64 = visualizations.get(clothing_type)
                
                if not mask_base64:
                    return {'error': f'No {clothing_type} mask found in stored data'}, 400
                    
                # If using visualization, we need to extract the mask
                print(f"WARNING: Using visualization instead of binary mask for {clothing_type}")
                mask_img_data = base64.b64decode(mask_base64)
                mask_pil = Image.open(BytesIO(mask_img_data))
                mask_np = np.array(mask_pil)
                
                # Resize to match original image if needed
                if mask_np.shape[:2] != (h, w):
                    print(f"Resizing visualization from {mask_np.shape[:2]} to {(h, w)}")
                    mask_pil = mask_pil.resize((w, h), Image.Resampling.LANCZOS)
                    mask_np = np.array(mask_pil)
                
                # Create RGBA mask for Gemini - transparent where to keep original
                # This matches the old implementation
                mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                
                if len(mask_np.shape) == 3:
                    diff = np.abs(mask_np.astype(float) - image_rgb.astype(float))
                    binary_mask = np.any(diff > 30, axis=2)
                else:
                    binary_mask = mask_np > 0
                
                # Set alpha channel - transparent (0) where to keep original, opaque (255) where to replace
                mask_rgba[:, :, 3] = (binary_mask * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_rgba, mode='RGBA')
            else:
                # Using stored binary mask - convert to RGBA
                print(f"Using stored binary mask for {clothing_type}")
                mask_img_data = base64.b64decode(mask_base64)
                mask_pil = Image.open(BytesIO(mask_img_data))
                
                print(f"Loaded binary mask size: {mask_pil.size} (width x height)")
                print(f"Mask mode: {mask_pil.mode}")
                
                # Resize if needed
                if mask_pil.size != (w, h):
                    print(f"WARNING: Mask size {mask_pil.size} doesn't match image size {(w, h)}")
                    print(f"Resizing mask from {mask_pil.size} to {(w, h)}")
                    mask_pil = mask_pil.resize((w, h), Image.Resampling.LANCZOS)
                
                # Convert binary mask to RGBA - transparent where to keep original
                mask_np = np.array(mask_pil)
                mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                # Where mask is white (255), make opaque (alpha=255) for replacement
                # Where mask is black (0), make transparent (alpha=0) to keep original
                mask_rgba[:, :, 3] = mask_np
                mask_pil = Image.fromarray(mask_rgba, mode='RGBA')
            
            # Debug: Save mask to verify
            import os
            debug_dir = os.path.join(os.path.dirname(__file__), 'debug_masks')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f'mask_{clothing_type}_{int(time.time())}.png')
            mask_pil.save(debug_path)
            print(f"DEBUG: Saved mask to {debug_path}")
            print(f"DEBUG: Mask size: {mask_pil.size}")
            print(f"DEBUG: Original image shape: {image_rgb.shape}")
            print(f"DEBUG: Mask mode: {mask_pil.mode} (RGBA - transparent=keep, opaque=replace)")
            
            # Ensure mask and image have EXACT same dimensions
            if mask_pil.size != (w, h):
                print(f"ERROR: Mask size {mask_pil.size} doesn't match image size {(w, h)}")
                mask_pil = mask_pil.resize((w, h), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            mask_pil.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Convert original image to base64
            original_pil = Image.fromarray(image_rgb)
            print(f"\n=== FINAL CHECK BEFORE SENDING TO GEMINI ===")
            print(f"Original image size: {original_pil.size} (width x height)")
            print(f"Mask size: {mask_pil.size} (width x height)")
            print(f"Do they match? {original_pil.size == mask_pil.size}")
            
            buffer2 = BytesIO()
            original_pil.save(buffer2, format="PNG")
            original_base64 = base64.b64encode(buffer2.getvalue()).decode()
            
            return {
                'success': True,
                'original_image': original_base64,
                'mask_image': mask_base64
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
            
            # Add safety settings to reduce false positives
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            model = genai.GenerativeModel(
                'gemini-2.0-flash-preview-image-generation',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Decode person image from base64
            try:
                if ',' in person_image:
                    person_img_data = base64.b64decode(person_image.split(',')[1])
                else:
                    person_img_data = base64.b64decode(person_image)
                person_pil = Image.open(BytesIO(person_img_data)).convert('RGB')
                print(f"Person image size: {person_pil.size}")
            except Exception as e:
                print(f"Error decoding person image: {str(e)}")
                return {'error': f'Failed to decode person image: {str(e)}'}, 400
            
            # Decode mask image from base64
            try:
                if ',' in mask_image:
                    mask_img_data = base64.b64decode(mask_image.split(',')[1])
                else:
                    mask_img_data = base64.b64decode(mask_image)
                mask_pil = Image.open(BytesIO(mask_img_data))
                print(f"Mask image size: {mask_pil.size}, mode: {mask_pil.mode}")
                
                # CRITICAL: Ensure mask is same size as person image
                if mask_pil.size != person_pil.size:
                    print(f"WARNING: Mask size {mask_pil.size} doesn't match person size {person_pil.size}")
                    print(f"Resizing mask to match person image")
                    mask_pil = mask_pil.resize(person_pil.size, Image.Resampling.LANCZOS)
                    
                # Keep mask as RGBA if it's already RGBA, otherwise convert
                if mask_pil.mode == 'RGBA':
                    print(f"Mask is already in RGBA mode")
                elif mask_pil.mode == 'L':
                    print(f"Converting grayscale mask to RGBA")
                    # Convert L mode to RGBA - use the grayscale value as alpha
                    mask_np = np.array(mask_pil)
                    mask_rgba = np.zeros((mask_pil.size[1], mask_pil.size[0], 4), dtype=np.uint8)
                    mask_rgba[:, :, 3] = mask_np  # Use grayscale as alpha
                    mask_pil = Image.fromarray(mask_rgba, mode='RGBA')
                else:
                    print(f"Converting mask from {mask_pil.mode} to RGBA")
                    mask_pil = mask_pil.convert('RGBA')
                    
            except Exception as e:
                print(f"Error decoding mask image: {str(e)}")
                return {'error': f'Failed to decode mask image: {str(e)}'}, 400
            
            # Load garment image - look in parent directory (react-app)
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parent_dir = os.path.dirname(backend_dir)  # react-app directory
            garment_path = os.path.join(parent_dir, 'data', 'sample_images', 'garments', garment_file)
            garment_pil = Image.open(garment_path).convert('RGB')
            
            # Create prompt - clearer about what each image is
            prompt = f"""Virtual try-on task: Replace the {clothing_type} in the person image.

Images provided (in order):
1. Person wearing original clothing
2. New {clothing_type} to try on
3. Mask showing which area to replace (opaque areas = replace, transparent = keep original)

Please replace only the masked area with the new {clothing_type}, keeping everything else unchanged."""
            
            # FINAL VERIFICATION before sending to Gemini
            print(f"FINAL VERIFICATION before Gemini:")
            print(f"  Person: size={person_pil.size}, mode={person_pil.mode}")
            print(f"  Garment: size={garment_pil.size}, mode={garment_pil.mode}")
            print(f"  Mask: size={mask_pil.size}, mode={mask_pil.mode}")
            
            if person_pil.size != mask_pil.size:
                print(f"CRITICAL ERROR: Person and mask sizes don't match!")
                return {'error': 'Person and mask sizes do not match'}, 500
            
            # Prepare inputs for Gemini - CORRECT ORDER: Person, Garment, Mask
            inputs = [prompt, person_pil, garment_pil, mask_pil]
            
            # Call Gemini API
            start_time = time.time()
            print(f"Sending to Gemini: {len(inputs)} inputs (prompt + {len(inputs)-1} images)", flush=True)
            print(f"Garment file: {garment_file}", flush=True)
            print(f"Clothing type: {clothing_type}", flush=True)
            
            try:
                response = model.generate_content(inputs)
                
                # Debug the response
                print(f"Gemini response received", flush=True)
                if hasattr(response, 'prompt_feedback'):
                    print(f"Prompt feedback: {response.prompt_feedback}", flush=True)
                if hasattr(response, 'candidates'):
                    print(f"Number of candidates: {len(response.candidates) if response.candidates else 0}", flush=True)
                    if not response.candidates:
                        print(f"Full response object: {vars(response)}", flush=True)
            except Exception as api_error:
                print(f"Gemini API call failed: {str(api_error)}", flush=True)
                import traceback
                traceback.print_exc()
                return {'error': f'Gemini API call failed: {str(api_error)}'}, 500
            
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
    
    def gemini_tryon_multiple(self, person_image, mask_images, garment_files, clothing_types):
        """Test multi-item virtual try-on using Gemini
        
        Args:
            person_image: Base64 encoded original person image
            mask_images: Dict of clothing_type -> base64 encoded RGBA mask image
            garment_files: Dict of clothing_type -> garment filename
            clothing_types: List of clothing types to replace
            
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
            
            # Configure Gemini with IMAGE GENERATION model
            generation_config = {
                "response_modalities": ["TEXT", "IMAGE"]
            }
            
            # Add safety settings to reduce false positives
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            model = genai.GenerativeModel(
                'gemini-2.0-flash-preview-image-generation',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Decode person image from base64
            try:
                if ',' in person_image:
                    person_img_data = base64.b64decode(person_image.split(',')[1])
                else:
                    person_img_data = base64.b64decode(person_image)
                person_pil = Image.open(BytesIO(person_img_data)).convert('RGB')
                print(f"Person image size: {person_pil.size}")
            except Exception as e:
                print(f"Error decoding person image: {str(e)}")
                return {'error': f'Failed to decode person image: {str(e)}'}, 400
            
            # Create a combined RGBA mask
            w, h = person_pil.size  # PIL returns (width, height)
            print(f"Person PIL image: width={w}, height={h}")
            combined_mask_np = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA mask
            print(f"Created RGBA mask array with shape: {combined_mask_np.shape}")
            
            # Process each mask and combine them
            for clothing_type in clothing_types:
                if clothing_type not in mask_images:
                    continue
                    
                mask_image = mask_images[clothing_type]
                try:
                    if ',' in mask_image:
                        mask_img_data = base64.b64decode(mask_image.split(',')[1])
                    else:
                        mask_img_data = base64.b64decode(mask_image)
                    mask_pil = Image.open(BytesIO(mask_img_data))
                    
                    # Resize mask to match person image size if needed
                    if mask_pil.size != person_pil.size:
                        print(f"WARNING: Mask for {clothing_type} has size {mask_pil.size} but person has {person_pil.size}")
                        print(f"Resizing mask to match person image")
                        mask_pil = mask_pil.resize(person_pil.size, Image.Resampling.LANCZOS)
                        print(f"After resize: mask size = {mask_pil.size}")
                    
                    mask_array = np.array(mask_pil)
                    
                    # Combine RGBA masks
                    if len(mask_array.shape) == 2:  # Grayscale mask
                        # Convert to RGBA - use grayscale as alpha
                        combined_mask_np[:, :, 3] = np.maximum(combined_mask_np[:, :, 3], mask_array)
                    elif mask_array.shape[2] >= 4:  # RGBA mask
                        # Combine alpha channels - take maximum (union of masks)
                        combined_mask_np[:, :, 3] = np.maximum(combined_mask_np[:, :, 3], mask_array[:, :, 3])
                    else:  # RGB mask
                        # Convert to grayscale and use as alpha
                        # Convert RGB to grayscale using PIL
                        mask_pil = Image.fromarray(mask_array)
                        mask_gray = np.array(mask_pil.convert('L'))
                        combined_mask_np[:, :, 3] = np.maximum(combined_mask_np[:, :, 3], mask_gray)
                    
                except Exception as e:
                    print(f"Error processing mask for {clothing_type}: {str(e)}")
                    continue
            
            combined_mask_pil = Image.fromarray(combined_mask_np, mode='RGBA')
            
            # Debug: Save combined mask
            debug_dir = os.path.join(os.path.dirname(__file__), 'debug_masks')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f'combined_mask_{int(time.time())}.png')
            combined_mask_pil.save(debug_path)
            print(f"DEBUG: Saved combined mask to {debug_path}")
            print(f"DEBUG: Combined mask shape: {combined_mask_np.shape}")
            print(f"DEBUG: Combined mask type: RGBA (transparent=keep, opaque=replace)")
            
            # Load all garment images
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parent_dir = os.path.dirname(backend_dir)  # react-app directory
            
            garment_images = []
            for clothing_type in clothing_types:
                if clothing_type not in garment_files:
                    continue
                    
                garment_file = garment_files[clothing_type]
                
                # Check if it's a base64 image or a file path
                if garment_file.startswith('data:image'):
                    # It's a base64 data URL
                    try:
                        # Extract base64 data from data URL
                        if ',' in garment_file:
                            garment_data = base64.b64decode(garment_file.split(',')[1])
                        else:
                            garment_data = base64.b64decode(garment_file)
                        
                        garment_pil = Image.open(BytesIO(garment_data)).convert('RGB')
                        garment_images.append((clothing_type, garment_pil))
                        print(f"Loaded base64 garment for {clothing_type}")
                    except Exception as e:
                        print(f"Error loading base64 garment for {clothing_type}: {str(e)[:100]}...")  # Truncate error
                        continue
                else:
                    # It's a file path (existing behavior)
                    garment_path = os.path.join(parent_dir, 'data', 'sample_images', 'garments', garment_file)
                    try:
                        garment_pil = Image.open(garment_path).convert('RGB')
                        garment_images.append((clothing_type, garment_pil))
                        print(f"Loaded file garment for {clothing_type}: {garment_file}")
                    except Exception as e:
                        print(f"Error loading garment file for {clothing_type}: {str(e)[:100]}...")  # Truncate error
                        continue
            
            # Create prompt for multiple items - clearer about image order
            items_str = " and ".join(clothing_types)
            garment_list = "\n".join([f"{i+2}. New {clothing_type} to try on" 
                                     for i, (clothing_type, _) in enumerate(garment_images)])
            
            prompt = f"""Virtual try-on task: Replace the {items_str} in the person image.

Images provided (in order):
1. Person wearing original clothing
{garment_list}
{len(garment_images)+2}. Combined mask showing all areas to replace (opaque = replace, transparent = keep)

Please replace only the masked areas with the corresponding new items, keeping everything else unchanged."""
            
            print(f"\n{'='*50}", flush=True)
            print(f"FULL PROMPT BEING SENT TO GEMINI:", flush=True)
            print(f"{'='*50}", flush=True)
            print(prompt, flush=True)
            print(f"{'='*50}\n", flush=True)
            
            # Prepare inputs for Gemini - CORRECT ORDER: Person, Garments, then Mask
            inputs = [prompt, person_pil]
            for _, garment_img in garment_images:
                inputs.append(garment_img)
            inputs.append(combined_mask_pil)  # Mask goes last
            
            # Call Gemini API
            start_time = time.time()
            print(f"Sending to Gemini: {len(inputs)} inputs (prompt + person + mask + {len(garment_images)} garments)", flush=True)
            print(f"Replacing: {items_str}", flush=True)
            
            try:
                response = model.generate_content(inputs)
                
                # Debug the response
                print(f"Gemini response received", flush=True)
                if hasattr(response, 'prompt_feedback'):
                    print(f"Prompt feedback: {response.prompt_feedback}", flush=True)
                if hasattr(response, 'candidates'):
                    print(f"Number of candidates: {len(response.candidates) if response.candidates else 0}", flush=True)
                    if not response.candidates:
                        print(f"Full response object: {vars(response)}", flush=True)
            except Exception as api_error:
                print(f"Gemini API call failed: {str(api_error)}", flush=True)
                import traceback
                traceback.print_exc()
                return {'error': f'Gemini API call failed: {str(api_error)}'}, 500
            
            # Check if response has candidates
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
                                'processing_time': processing_time,
                                'items_replaced': clothing_types
                            }, 200
                        except Exception as e:
                            print(f"Failed to process image data: {str(e)}")
                            continue
                
                return {'error': 'Gemini returned no image data in response'}, 500
            else:
                return {'error': 'Gemini returned no candidates'}, 500
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'Gemini API error: {str(e)}'}, 500

    def generate_clothing_descriptions(self, image_base64, detected_items):
        """
        Generate descriptions for detected clothing items using Gemini
        
        Args:
            image_base64: Base64 encoded image
            detected_items: Dict with clothing types as keys, empty values
                          e.g., {"shirt": "", "pants": "", "shoes": ""}
        
        Returns:
            dict: Same structure with filled descriptions
        """
        if not self.genai:
            return {'error': 'Gemini API not initialized'}, 500
        
        try:
            # Decode base64 image
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Create prompt for Gemini
            prompt = f"""You are a fashion AI assistant. Analyze this image and provide brief, comma-separated descriptions for each detected clothing item.

Detected items to describe:
{json.dumps(detected_items, indent=2)}

For each item, provide a concise description focusing on:
- Color/pattern
- Style/type
- Material/texture (if visible)
- Key features

IMPORTANT: 
1. Respond ONLY with a valid JSON object in the exact same format as the input
2. Fill in the empty string values with descriptions
3. Use comma-separated attributes in each description
4. Keep descriptions concise (3-5 attributes max)
5. If an item is not clearly visible, put "not clearly visible"

Example response:
{{"shirt": "white, cotton, button-down, long-sleeve", "pants": "blue, denim, straight-fit", "shoes": "black, leather, formal"}}

Respond with ONLY the JSON object, no other text."""

            # Use Gemini 2.0 Flash for quick analysis
            model = self.genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Configure for JSON response
            generation_config = {
                "temperature": 0.3,  # Lower temperature for consistency
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 500,
            }
            
            print(f"\n[GeminiService] Generating descriptions...")
            print(f"  Detected items: {list(detected_items.keys())}")
            
            # Generate descriptions
            response = model.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            print(f"  Gemini raw response: {response.text}")
            
            # Parse JSON response
            try:
                # Clean the response text
                response_text = response.text.strip()
                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                descriptions = json.loads(response_text)
                
                print(f"  Parsed descriptions: {descriptions}")
                
                # Validate response has same keys as input
                if set(descriptions.keys()) != set(detected_items.keys()):
                    print(f"  Warning: Response keys don't match input keys")
                    # Fill missing keys with empty strings
                    for key in detected_items:
                        if key not in descriptions:
                            descriptions[key] = "not detected"
                
                print(f"  Final descriptions being returned: {descriptions}")
                return descriptions, 200
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse Gemini response as JSON: {response.text}")
                return {
                    'error': 'Invalid JSON response from Gemini',
                    'raw_response': response.text
                }, 500
        
        except Exception as e:
            print(f"Error generating descriptions: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': 'Failed to generate descriptions',
                'details': str(e)
            }, 500
