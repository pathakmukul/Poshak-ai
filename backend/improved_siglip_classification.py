"""
Improved SigLIP classification using best practices from research
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

def classify_with_improved_prompts(image_crop, processor, model, use_template=False, position_hint=None):
    """
    Classify clothing using improved prompt engineering
    
    Args:
        image_crop: PIL Image of the masked region
        processor: SigLIP processor
        model: SigLIP model
        use_template: Whether to use "This is a photo of" template (generally False is better)
        position_hint: List of likely labels based on position/size
    
    Returns:
        dict with label, confidence, and all scores
    """
    
    # Define all clothing labels (matching sam2_clothing_detector_auto.py)
    all_labels = [
        # Upper body
        "shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "jacket", "coat", "vest",
        # Lower body
        "pants", "jeans", "trousers", "shorts", "skirt", "leggings",
        # Full body
        "dress", "suit", "jumpsuit",
        # Footwear
        "shoes", "sneakers", "boots", "sandals", "heels",
        # Accessories
        "hat", "cap", "sunglasses", "glasses", "watch", "belt", "tie", "scarf",
        "bag", "backpack", "purse",
        # Non-clothing (for filtering)
        "person", "face", "hair", "hand", "arm", "leg", "background"
    ]
    
    # Use position hints if provided, otherwise use all labels
    if position_hint:
        clothing_prompts = position_hint
    else:
        clothing_prompts = all_labels
    
    # Don't use templates - direct labels work better
    if use_template:
        # Override - templates generally perform worse
        use_template = False
    
    # Process with proper padding
    inputs = processor(
        text=clothing_prompts, 
        images=image_crop, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to device if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        model = model.to(device)
    elif torch.backends.mps.is_available():
        device = "mps"
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        model = model.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        # Use softmax for multi-class classification
        probs = torch.softmax(logits, dim=0)
    
    # Get best match
    best_idx = probs.argmax().item()
    best_label = clothing_prompts[best_idx]
    confidence = probs[best_idx].item()
    
    # Get all scores
    all_scores = {label: probs[i].item() for i, label in enumerate(clothing_prompts)}
    
    return {
        'label': best_label,
        'confidence': confidence,
        'all_scores': all_scores,
        'raw_prompt': best_label
    }


def classify_with_position_hints(image_crop, processor, model, mask_area, center_y, aspect_ratio):
    """
    Smart classification using position and size hints
    """
    
    # Use area ratio instead of absolute area for better generalization
    # Assuming typical image is around 1024x1024 = ~1M pixels
    area_ratio = mask_area / 1000000  # Normalize to typical image size
    
    # Determine likely categories based on position and size
    if area_ratio > 0.02 and area_ratio < 0.06 and center_y < 0.6:
        # Upper body clothing
        position_hint = ["shirt", "t-shirt", "hoodie", "jacket", "sweater", "blouse", "top", "vest", "coat"]
    elif area_ratio > 0.02 and area_ratio < 0.04 and center_y > 0.4 and center_y < 0.85:
        # Lower body clothing
        position_hint = ["pants", "jeans", "trousers", "shorts", "skirt", "leggings"]
    elif area_ratio < 0.008 and center_y > 0.75:
        # Footwear - smaller area, lower position
        position_hint = ["shoes", "sneakers", "boots", "sandals", "heels"]
    elif area_ratio > 0.04 and center_y > 0.3 and center_y < 0.8:
        # Full body clothing - larger area
        position_hint = ["dress", "suit", "jumpsuit"]
    else:
        # Fallback to all clothing categories
        position_hint = [
            "shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "jacket", "coat",
            "pants", "jeans", "trousers", "shorts", "skirt", "leggings",
            "dress", "suit", "jumpsuit",
            "shoes", "sneakers", "boots", "sandals",
            "person", "face", "background"
        ]
    
    # Use the improved classification with position hints
    return classify_with_improved_prompts(
        image_crop, processor, model, 
        use_template=False,  # Direct labels work better
        position_hint=position_hint
    )


def classify_with_context_prompts(image_crop, processor, model):
    """
    Alternative approach using more contextual prompts
    """
    
    # Simpler, more direct context prompts
    context_prompts = [
        # Direct clothing descriptions
        "person wearing shirt",
        "person wearing t-shirt", 
        "person wearing pants",
        "person wearing jeans",
        "person wearing dress",
        "person wearing shoes",
        "person wearing sneakers",
        
        # Non-clothing
        "person's face",
        "person's hand",
        "background"
    ]
    
    # Process inputs
    inputs = processor(
        text=context_prompts,
        images=image_crop,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    if torch.backends.mps.is_available():
        inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=0)  # Use softmax here too
    
    # Return results
    best_idx = probs.argmax().item()
    best_prompt = context_prompts[best_idx]
    
    # Extract clothing type from prompt
    if "wearing" in best_prompt:
        label = best_prompt.split("wearing ")[-1]
    else:
        label = best_prompt
    
    return {
        'label': label,
        'confidence': probs[best_idx].item(),
        'raw_prompt': best_prompt,
        'all_scores': {context_prompts[i]: probs[i].item() for i in range(len(context_prompts))}
    }


def ensemble_classification(image_crop, processor, model, mask_area=None, center_y=None, aspect_ratio=None):
    """
    Ensemble approach using multiple strategies
    """
    
    results = []
    
    # Strategy 1: Position-based hints (if available)
    if mask_area is not None and center_y is not None:
        pos_result = classify_with_position_hints(image_crop, processor, model, mask_area, center_y, aspect_ratio)
        results.append(('position', pos_result))
    
    # Strategy 2: Simple labels
    simple_result = classify_with_improved_prompts(image_crop, processor, model, use_template=False)
    results.append(('simple', simple_result))
    
    # Strategy 3: Context prompts
    context_result = classify_with_context_prompts(image_crop, processor, model)
    results.append(('context', context_result))
    
    # Choose best result based on confidence
    best_strategy, best_result = max(results, key=lambda x: x[1]['confidence'])
    
    return {
        'label': best_result['label'],
        'confidence': best_result['confidence'],
        'strategy': best_strategy,
        'all_results': {name: result for name, result in results}
    }


# Utility function to test different approaches
def compare_classification_approaches(image_path, mask):
    """
    Compare different classification approaches
    """
    from PIL import Image
    import cv2
    
    # Load models
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    model.eval()
    
    # Load and crop image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract masked region
    y_indices, x_indices = np.where(mask)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Add padding
    pad = 20
    y_min_pad = max(0, y_min - pad)
    y_max_pad = min(image_rgb.shape[0], y_max + pad)
    x_min_pad = max(0, x_min - pad)
    x_max_pad = min(image_rgb.shape[1], x_max + pad)
    
    # Create white background crop
    crop_height = y_max_pad - y_min_pad
    crop_width = x_max_pad - x_min_pad
    masked_crop = np.ones((crop_height, crop_width, 3), dtype=np.uint8) * 255
    mask_crop = mask[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
    masked_crop[mask_crop] = image_rgb[y_min_pad:y_max_pad, x_min_pad:x_max_pad][mask_crop]
    
    # Convert to PIL
    pil_img = Image.fromarray(masked_crop)
    
    # Calculate mask properties
    mask_area = np.sum(mask)
    center_y = y_indices.mean() / image_rgb.shape[0]
    aspect_ratio = (x_indices.max() - x_indices.min()) / (y_indices.max() - y_indices.min() + 1)
    
    # Test different approaches
    print("1. Simple labels (no template):")
    simple_result = classify_with_improved_prompts(pil_img, processor, model, use_template=False)
    print(f"   Result: {simple_result['label']} ({simple_result['confidence']:.1%})")
    
    print("\n2. Position-based hints:")
    pos_result = classify_with_position_hints(pil_img, processor, model, mask_area, center_y, aspect_ratio)
    print(f"   Result: {pos_result['label']} ({pos_result['confidence']:.1%})")
    
    print("\n3. Context-aware prompts:")
    context_result = classify_with_context_prompts(pil_img, processor, model)
    print(f"   Result: {context_result['label']} ({context_result['confidence']:.1%})")
    
    print("\n4. Ensemble classification:")
    ensemble_result = ensemble_classification(pil_img, processor, model, mask_area, center_y, aspect_ratio)
    print(f"   Result: {ensemble_result['label']} ({ensemble_result['confidence']:.1%}) via {ensemble_result['strategy']}")
    
    return {
        'simple': simple_result,
        'position': pos_result,
        'context': context_result,
        'ensemble': ensemble_result
    }