"""
CLIP-based clothing classifier as an alternative to SigLIP
Uses OpenAI's CLIP model which might perform better for clothing
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# REMOVED CLIPClothingClassifier to prevent duplicate model loading
# The CLIP model is already loaded in flask_api.py's load_classification_models()
# This class was causing duplicate loading and 429 errors

# Global CLIP model instances (loaded once, reused everywhere)
_clip_processor = None
_clip_model = None

def load_clip_model():
    """Load CLIP model once and reuse it"""
    global _clip_processor, _clip_model
    
    if _clip_processor is None or _clip_model is None:
        import os
        from transformers import CLIPProcessor, CLIPModel
        
        print("Loading CLIP model for classification...")
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
        
        try:
            # Try loading from cache first
            _clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=cache_dir,
                local_files_only=True,
                use_fast=True
            )
            _clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=cache_dir,
                local_files_only=True
            )
            print("Loaded CLIP from local cache with FAST processor")
        except:
            # Download if needed
            print("Local cache not found, downloading CLIP...")
            _clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=cache_dir,
                use_fast=True
            )
            _clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=cache_dir
            )
        
        _clip_model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _clip_model = _clip_model.to("cuda")
            print("CLIP moved to CUDA")
        elif torch.backends.mps.is_available():
            _clip_model = _clip_model.to("mps") 
            print("CLIP moved to MPS")
    
    return _clip_processor, _clip_model


def classify_with_clip(image_crop, processor=None, model=None, position_y=None, area_ratio=None):
    """
    Drop-in replacement for SigLIP classification using CLIP
    Uses the provided processor and model if available, otherwise loads from this module
    """
    # If processor and model not provided, load them from this module
    if processor is None or model is None:
        processor, model = load_clip_model()
    
    # Now we have processor and model, use them for classification
    # Direct classification using provided CLIP model
    from PIL import Image
    import torch
    
    # Ensure image is PIL Image
    if not isinstance(image_crop, Image.Image):
        image_crop = Image.fromarray(image_crop)
    
    # Define simple clothing prompts
    text_prompts = [
        "a photo of a shirt",
        "a photo of pants", 
        "a photo of shoes",
        "a photo of a dress",
        "a photo of a jacket",
        "a photo of a person",
        "background"
    ]
    
    # Process image and text - work around the _valid_processor_keys bug
    # Use the processor's components directly
    image_inputs = processor.image_processor(images=image_crop, return_tensors="pt")
    text_inputs = processor.tokenizer(text=text_prompts, return_tensors="pt", padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**image_inputs, **text_inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Get best match
    best_idx = probs[0].argmax().item()
    confidence = probs[0][best_idx].item()
    label = text_prompts[best_idx].replace("a photo of ", "").replace("a ", "")
    
    # Map to standard categories
    if label in ["shirt", "jacket", "dress"]:
        label = "shirt"
    elif label == "pants":
        label = "pants"
    elif label == "shoes":
        label = "shoes"
    else:
        label = "non_clothing"
        
    return {
        'label': label,
        'confidence': confidence,
        'all_scores': {text_prompts[i]: probs[0][i].item() for i in range(len(text_prompts))}
    }
    

if __name__ == "__main__":
    # Test
    print("CLIP classifier ready for use")