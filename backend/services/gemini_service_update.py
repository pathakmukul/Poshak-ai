# This is the update needed for gemini_service.py
# Replace the garment loading section (around lines 615-628) with this:

"""
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
"""

# Also add this import at the top of the file if not already present:
# from io import BytesIO