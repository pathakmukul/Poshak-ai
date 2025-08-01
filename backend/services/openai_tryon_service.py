import os
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAITryOnService:
    def __init__(self):
        # Initialize OpenAI client
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        
    def process_tryon(self, person_image_base64, garment_images):
        """
        Process virtual try-on using OpenAI's image API
        
        Args:
            person_image_base64: Base64 encoded person image or URL (the model)
            garment_images: Dict of garment type -> base64 encoded images or URLs
                           e.g. {'shirt': 'data:image/png;base64,...', 'pants': '...', 'shoes': '...'}
        
        Returns:
            dict with success status and result image
        """
        try:
            # Handle person image - could be base64 or URL
            if person_image_base64.startswith('data:image'):
                # It's a data URL
                person_image_base64 = person_image_base64.split(',')[1]
                model_image_bytes = base64.b64decode(person_image_base64)
            elif person_image_base64.startswith('http'):
                # It's a URL - fetch it
                import requests
                response = requests.get(person_image_base64)
                model_image_bytes = response.content
            else:
                # Assume it's raw base64
                model_image_bytes = base64.b64decode(person_image_base64)
            
            # Prepare input images list (model first)
            input_images = [('model.png', model_image_bytes, 'image/png')]
            
            # Add garment images in specific order: shirt, pants, shoes
            garment_order = ['shirt', 'pants', 'shoes']
            selected_items = []
            
            for garment_type in garment_order:
                if garment_type in garment_images:
                    garment_data = garment_images[garment_type]
                    
                    # Handle different formats
                    if garment_data.startswith('data:image'):
                        # Data URL
                        garment_bytes = base64.b64decode(garment_data.split(',')[1])
                    elif garment_data.startswith('http'):
                        # URL - fetch it
                        import requests
                        response = requests.get(garment_data)
                        garment_bytes = response.content
                    else:
                        # Raw base64
                        garment_bytes = base64.b64decode(garment_data)
                    filename = f"{garment_type}.png"
                    input_images.append((filename, garment_bytes, 'image/png'))
                    selected_items.append(garment_type)
            
            # Build prompt based on selected items
            if len(selected_items) == 0:
                return {
                    'success': False,
                    'error': 'No garment items provided'
                }
            elif len(selected_items) == 1:
                # Single item
                item = selected_items[0]
                prompt = f"Replace the existing {item} with the {item} from the provided image."
            else:
                # Multiple items
                items_text = []
                for item in selected_items:
                    if item == 'shirt':
                        items_text.append("the shirt")
                    elif item == 'pants':
                        items_text.append("the pants")
                    elif item == 'shoes':
                        items_text.append("the shoes")
                
                prompt = f"Update the outfit with {', '.join(items_text)} from the provided images."
            
            logger.info(f"OpenAI TryOn - Items: {selected_items}, Prompt: {prompt}")
            
            # Call OpenAI API
            if len(input_images) > 1:
                result = self.client.images.edit(
                    model="gpt-image-1",
                    image=input_images,  # Pass all images
                    prompt=prompt,
                    # input_fidelity="high",
                    quality="high",
                    output_format="jpeg"
                )
            else:
                result = self.client.images.edit(
                    model="gpt-image-1",
                    image=input_images[0],  # Just the model
                    prompt=prompt,
                    # input_fidelity="high",
                    quality="high",
                    output_format="jpeg"
                )
            
            # Get the generated image
            # The response format might be different
            if hasattr(result.data[0], 'b64_json'):
                image_base64 = result.data[0].b64_json
            else:
                # If URL is returned instead of base64
                image_url = result.data[0].url
                import requests
                response = requests.get(image_url)
                image_base64 = base64.b64encode(response.content).decode('utf-8')
            
            return {
                'success': True,
                'result_image': image_base64  # Already base64 encoded
            }
            
        except Exception as e:
            logger.error(f"OpenAI TryOn error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full error details: {repr(e)}")
            
            # Log which parameters might be causing the issue
            if "unexpected keyword argument" in str(e):
                logger.error("This OpenAI client version doesn't support these parameters")
                logger.error("The reference code might be using a different openai package version")
            
            return {
                'success': False,
                'error': str(e)
            }