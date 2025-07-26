# Visualization service for clothing detection
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64




def image_to_base64(image):
    """Convert numpy array to base64 string"""
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def create_binary_mask(image_shape, masks, clothing_type):
    """Create a binary mask for a specific clothing type
    
    Args:
        image_shape: (height, width) tuple
        masks: List of detected masks
        clothing_type: Type of clothing to create mask for
        
    Returns:
        Binary mask as numpy array (255 where clothing is, 0 elsewhere)
    """
    h, w = image_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Filter for specific clothing type
    if clothing_type == "shirt":
        type_labels = ["shirt", "dress"]
    elif clothing_type == "pants":
        type_labels = ["pants", "skirt"]
    elif clothing_type == "shoes":
        type_labels = ["shoes"]
    else:  # all items
        type_labels = ["shirt", "pants", "shoes", "dress", "skirt"]
    
    clothing_masks = [m for m in masks if m.get('label', '') in type_labels and not m.get('skip_viz', False)]
    
    # Special handling for shoes
    if clothing_type == "shoes" and len(clothing_masks) > 2:
        clothing_masks = sorted(clothing_masks, key=lambda x: x.get('area', 0), reverse=True)[:2]
    
    # Create binary mask
    for mask_dict in clothing_masks:
        mask = mask_dict['segmentation']
        binary_mask[mask] = 255
    
    return binary_mask




def create_clothing_visualization(image_rgb, masks, clothing_type, for_gemini=False):
    """Create visualization for specific clothing type
    
    Args:
        image_rgb: Original image
        masks: List of detected masks
        clothing_type: Type of clothing to visualize
        for_gemini: If True, creates clean mask without labels for AI processing
    """
    # Filter for specific clothing type
    if clothing_type == "shirt":
        type_labels = ["shirt", "dress"]  # Include dress as it covers upper body
    elif clothing_type == "pants":
        type_labels = ["pants", "skirt"]
    elif clothing_type == "shoes":
        type_labels = ["shoes"]
    else:  # all items
        type_labels = ["shirt", "pants", "shoes", "dress", "skirt"]
    
    clothing_masks = [m for m in masks if m.get('label', '') in type_labels and not m.get('skip_viz', False)]
    
    # Special handling for shoes
    if clothing_type == "shoes" and len(clothing_masks) > 2:
        # Keep only the two largest shoes
        clothing_masks = sorted(clothing_masks, key=lambda x: x.get('area', 0), reverse=True)[:2]
    
    # Create visualization
    overlay = image_rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, mask_dict in enumerate(clothing_masks):
        mask = mask_dict['segmentation']
        label = mask_dict.get('label', 'unknown')
        
        # Apply colored mask
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask] = colors[i % len(colors)]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Add label text only if not for Gemini
        if not for_gemini:
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                cy = int(y_indices.mean())
                cx = int(x_indices.mean())
                
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                text = f"{label.upper()}"
                
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(overlay_bgr, (cx-5, cy-text_h-5), (cx+text_w+5, cy+5), (0,0,0), -1)
                cv2.putText(overlay_bgr, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                
                overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return overlay, len(clothing_masks)

def create_closet_visualization(image_rgb, masks, clothing_type):
    """Create visualization for closet - shows only clothing items with transparent background"""
    h, w = image_rgb.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    
    print(f"\n[create_closet_visualization] Creating visualization for {clothing_type}")
    print(f"  Image dimensions: {w}x{h}")
    print(f"  Total masks received: {len(masks)}")
    
    # Filter for specific clothing type
    if clothing_type == "all":
        type_labels = ["shirt", "pants", "shoes", "dress", "skirt"]
    else:
        type_labels = [clothing_type]
    
    clothing_masks = [m for m in masks if m.get('label', '') in type_labels and not m.get('skip_viz', False)]
    print(f"  Filtered masks for {clothing_type}: {len(clothing_masks)}")
    
    # Track overall content bounds
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    
    # Apply all relevant masks and calculate bounds
    has_any_content = False
    for mask_dict in clothing_masks:
        mask = mask_dict['segmentation']
        # Check if mask has any True values
        mask_count = np.sum(mask)
        print(f"    Mask for {mask_dict.get('label', 'unknown')}: {mask_count} pixels")
        
        if mask_count > 0:
            has_any_content = True
            # Copy RGB values where mask is True
            result[mask, :3] = image_rgb[mask]
            # Set alpha to opaque where mask is True
            result[mask, 3] = 255
        
        # Update overall bounds if available
        if 'content_bounds' in mask_dict:
            bounds = mask_dict['content_bounds']
            min_x = min(min_x, bounds['minX'])
            max_x = max(max_x, bounds['maxX'])
            min_y = min(min_y, bounds['minY'])
            max_y = max(max_y, bounds['maxY'])
    
    # Calculate final content bounds
    content_bounds = None
    if clothing_masks and max_x > min_x and max_y > min_y:
        content_bounds = {
            'minX': min_x,
            'maxX': max_x,
            'minY': min_y,
            'maxY': max_y,
            'width': w,
            'height': h
        }
    
    # If no content was found, create a placeholder image
    if not has_any_content:
        print(f"  WARNING: No content found for {clothing_type}, creating placeholder")
        # Create a small placeholder image with a colored square
        placeholder_size = 100
        result = np.zeros((placeholder_size, placeholder_size, 4), dtype=np.uint8)
        # Add a colored square based on clothing type
        color_map = {'shirt': [100, 150, 255], 'pants': [150, 100, 50], 'shoes': [50, 50, 50]}
        color = color_map.get(clothing_type, [128, 128, 128])
        result[20:80, 20:80, :3] = color
        result[20:80, 20:80, 3] = 255
    
    # Convert to PIL
    pil_img = Image.fromarray(result, mode='RGBA')
    
    # Crop with VERY generous padding to preserve the full clothing item
    if clothing_masks and has_any_content:
        # Find the actual content bounds (non-transparent pixels)
        alpha = result[:, :, 3]
        rows = np.any(alpha, axis=1)
        cols = np.any(alpha, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            print(f"  Content bounds before padding: x({cmin},{cmax}) y({rmin},{rmax})")
            
            # Add VERY generous padding (50% of dimensions) to preserve context
            content_width = cmax - cmin
            content_height = rmax - rmin
            padding_x = int(0.5 * content_width)  # 50% padding
            padding_y = int(0.5 * content_height)  # 50% padding
            
            rmin = max(0, rmin - padding_y)
            rmax = min(h - 1, rmax + padding_y)
            cmin = max(0, cmin - padding_x)
            cmax = min(w - 1, cmax + padding_x)
            
            print(f"  Content bounds after padding: x({cmin},{cmax}) y({rmin},{rmax})")
            print(f"  Cropping from {w}x{h} to {cmax-cmin+1}x{rmax-rmin+1}")
            
            # Crop the image to content bounds
            pil_img = pil_img.crop((cmin, rmin, cmax + 1, rmax + 1))
    
    # Resize for mobile if image is too large
    max_size = 512  # Max dimension for mobile compatibility
    if max(pil_img.size) > max_size:
        # Calculate new size maintaining aspect ratio
        ratio = max_size / max(pil_img.size)
        new_size = tuple(int(dim * ratio) for dim in pil_img.size)
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    
    return {
        'image': base64.b64encode(buffer.getvalue()).decode(),
        'count': len(clothing_masks),
        'content_bounds': content_bounds
    }

def create_person_only_image(image_rgb, masks):
    """Create an image with only the person (no background)"""
    h, w = image_rgb.shape[:2]
    
    # Create a combined mask of all human-related segments
    person_mask = np.zeros((h, w), dtype=bool)
    
    for mask_dict in masks:
        # Include all segments (clothing and body parts) to get the full person
        mask = mask_dict['segmentation']
        person_mask = person_mask | mask
    
    # Create RGBA image with transparent background
    result = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Copy RGB values where person mask is True
    result[person_mask, :3] = image_rgb[person_mask]
    # Set alpha to opaque where person mask is True
    result[person_mask, 3] = 255
    
    # Find bounding box of the person
    y_indices, x_indices = np.where(person_mask)
    if len(y_indices) > 0:
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Add small padding (10 pixels or 2% of image dimensions)
        padding = max(10, int(0.02 * max(h, w)))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop to person bounds
        person_cropped = result[y_min:y_max, x_min:x_max]
    else:
        # If no person found, return original with transparency
        person_cropped = result
    
    return person_cropped

def generate_all_visualizations(image_rgb, masks):
    """Generate all clothing visualizations at once"""
    all_items_img, all_items_count = create_clothing_visualization(image_rgb, masks, "all", for_gemini=True)
    shirt_img, shirt_count = create_clothing_visualization(image_rgb, masks, "shirt", for_gemini=True)
    pants_img, pants_count = create_clothing_visualization(image_rgb, masks, "pants", for_gemini=True)
    shoes_img, shoes_count = create_clothing_visualization(image_rgb, masks, "shoes", for_gemini=True)
    
    # Create closet visualizations
    closet_all = create_closet_visualization(image_rgb, masks, "all")
    closet_shirt = create_closet_visualization(image_rgb, masks, "shirt")
    closet_pants = create_closet_visualization(image_rgb, masks, "pants")
    closet_shoes = create_closet_visualization(image_rgb, masks, "shoes")
    
    # Create person-only image (no background)
    person_only_img = create_person_only_image(image_rgb, masks)
    
    # Create binary masks for Gemini
    shirt_mask = create_binary_mask(image_rgb.shape, masks, "shirt")
    pants_mask = create_binary_mask(image_rgb.shape, masks, "pants")
    shoes_mask = create_binary_mask(image_rgb.shape, masks, "shoes")
    
    return {
        'all_items_img': image_to_base64(all_items_img),
        'all_items_count': all_items_count,
        'shirt_img': image_to_base64(shirt_img),
        'shirt_count': shirt_count,
        'pants_img': image_to_base64(pants_img),
        'pants_count': pants_count,
        'shoes_img': image_to_base64(shoes_img),
        'shoes_count': shoes_count,
        'person_only_img': image_to_base64(person_only_img),  # NEW!
        'closet_visualizations': {
            'all': closet_all['image'],
            'shirt': closet_shirt['image'],
            'pants': closet_pants['image'],
            'shoes': closet_shoes['image']
        },
        'closet_metadata': {
            'all': closet_all['content_bounds'],
            'shirt': closet_shirt['content_bounds'],
            'pants': closet_pants['content_bounds'],
            'shoes': closet_shoes['content_bounds']
        },
        'binary_masks': {
            'shirt': image_to_base64(shirt_mask) if shirt_count > 0 else None,
            'pants': image_to_base64(pants_mask) if pants_count > 0 else None,
            'shoes': image_to_base64(shoes_mask) if shoes_count > 0 else None
        }
    }

def create_raw_segformer_visualization(image_rgb, masks):
    """Create grid visualization of ALL raw Segformer masks"""
    h, w = image_rgb.shape[:2]
    
    # Handle empty masks
    if not masks:
        # Return a simple image with "No masks detected" text
        canvas = np.ones((400, 600, 3), dtype=np.uint8) * 240
        cv2.putText(canvas, "No masks detected", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
        return canvas
    
    # Calculate grid size
    n_masks = len(masks)
    cols = min(4, n_masks)  # Max 4 columns
    rows = (n_masks + cols - 1) // cols
    
    # Create a large canvas
    cell_size = 300
    canvas_w = cols * cell_size + (cols + 1) * 10  # 10px padding
    canvas_h = rows * cell_size + (rows + 1) * 10 + 50  # Extra space for title
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add title
    title = f"All Segformer Segments: {n_masks} masks detected"
    cv2.putText(canvas, title, (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Sort masks by area (largest first)
    masks_sorted = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
    
    # Colors for masks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128)
    ]
    
    for idx, mask_dict in enumerate(masks_sorted):
        row = idx // cols
        col = idx % cols
        
        # Calculate position in canvas
        x_start = col * cell_size + (col + 1) * 10
        y_start = row * cell_size + (row + 1) * 10 + 50
        
        # Create individual mask visualization
        mask_viz = image_rgb.copy()
        mask = mask_dict['segmentation']
        
        # Apply colored overlay to mask area only
        mask_color = colors[idx % len(colors)]
        mask_viz[mask] = mask_viz[mask] * 0.3 + np.array(mask_color) * 0.7
        
        # Draw mask boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_viz, contours, -1, mask_color, 2)
        
        # Resize to cell size
        scale = min(cell_size / h, cell_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        mask_viz_resized = cv2.resize(mask_viz, (new_w, new_h))
        
        # Center in cell
        y_offset = (cell_size - new_h) // 2
        x_offset = (cell_size - new_w) // 2
        
        # Place in canvas
        canvas[y_start + y_offset:y_start + y_offset + new_h,
               x_start + x_offset:x_start + x_offset + new_w] = mask_viz_resized
        
        # Add mask info
        info_text = f"{mask_dict.get('original_label', 'Unknown')}"
        area_pct = (mask_dict['area'] / (h * w)) * 100
        size_text = f"{area_pct:.1f}%"
        
        cv2.putText(canvas, info_text, (x_start + 10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, size_text, (x_start + 10, y_start + cell_size - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return canvas
