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
        
        # Use smaller model for CPU inference (4x faster)
        model_name = "openai/clip-vit-base-patch32"  # 88M params vs 428M
        # Fallback to large model if base not available
        fallback_model = "openai/clip-vit-large-patch14"
        
        try:
            # Try loading from cache first
            _clip_processor = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
                use_fast=True
            )
            _clip_model = CLIPModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True
            )
            print(f"Loaded {model_name} from local cache with FAST processor")
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            # Try fallback model
            try:
                print(f"Trying fallback model {fallback_model}...")
                _clip_processor = CLIPProcessor.from_pretrained(
                    fallback_model,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    use_fast=True
                )
                _clip_model = CLIPModel.from_pretrained(
                    fallback_model,
                    cache_dir=cache_dir,
                    local_files_only=True
                )
                print(f"Loaded fallback {fallback_model} from local cache")
            except:
                # Download if needed (should not happen with proper Docker cache)
                print(f"ERROR: No models found in cache. This should not happen!")
                raise
        
        _clip_model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _clip_model = _clip_model.to("cuda")
            print("CLIP moved to CUDA")
        elif torch.backends.mps.is_available():
            _clip_model = _clip_model.to("mps") 
            print("CLIP moved to MPS")
        else:
            print("Running CLIP on CPU (no GPU available)")
            # Set CPU optimizations
            torch.set_num_threads(8)  # Use multiple CPU threads
            # Additional CPU optimizations
            torch.set_grad_enabled(False)  # Disable gradients for inference
            if hasattr(torch, 'set_flush_denormal'):
                torch.set_flush_denormal(True)  # Flush denormal numbers to zero
    
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
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    image_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in text_inputs.items()}
    
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


def classify_batch_with_clip(image_crops, processor=None, model=None):
    """
    Batch processing version of classify_with_clip
    Process multiple images in a single forward pass for much better performance
    
    Args:
        image_crops: List of PIL images or numpy arrays
        processor: CLIP processor
        model: CLIP model
        
    Returns:
        List of classification results
    """
    import time
    start_time = time.time()
    
    # If processor and model not provided, load them
    if processor is None or model is None:
        processor, model = load_clip_model()
    
    from PIL import Image
    import torch
    
    # Ensure all images are PIL Images
    pil_images = []
    for img in image_crops:
        if not isinstance(img, Image.Image):
            pil_images.append(Image.fromarray(img))
        else:
            pil_images.append(img)
    
    print(f"   Prepared {len(pil_images)} images for batch processing")
    
    # Define text prompts (same as single version)
    text_prompts = [
        "a photo of a shirt",
        "a photo of pants", 
        "a photo of shoes",
        "a photo of a dress",
        "a photo of a jacket",
        "a photo of a person",
        "background"
    ]
    
    # Process all images at once
    preprocess_start = time.time()
    image_inputs = processor.image_processor(images=pil_images, return_tensors="pt")
    text_inputs = processor.tokenizer(text=text_prompts, return_tensors="pt", padding=True)
    print(f"   Image preprocessing took: {time.time() - preprocess_start:.2f}s")
    
    # Move to device
    device_start = time.time()
    device = next(model.parameters()).device
    image_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in text_inputs.items()}
    print(f"   Moving to device took: {time.time() - device_start:.2f}s")
    
    # Get predictions for all images at once
    inference_start = time.time()
    with torch.no_grad():
        outputs = model(**image_inputs, **text_inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    print(f"   Model inference took: {time.time() - inference_start:.2f}s")
    
    # Process results for each image
    results = []
    for i in range(len(image_crops)):
        best_idx = probs[i].argmax().item()
        confidence = probs[i][best_idx].item()
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
        
        results.append({
            'label': label,
            'confidence': confidence,
            'all_scores': {text_prompts[j]: probs[i][j].item() for j in range(len(text_prompts))}
        })
    
    print(f"   Total batch classification time: {time.time() - start_time:.2f}s")
    return results


if __name__ == "__main__":
    # Test
    print("CLIP classifier ready for use")