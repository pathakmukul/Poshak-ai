"""
CLIP-based clothing classifier as an alternative to SigLIP
Uses OpenAI's CLIP model which might perform better for clothing
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPClothingClassifier:
    def __init__(self):
        """Initialize CLIP model for clothing classification"""
        print("Loading CLIP model for clothing classification...")
        
        # Use a better CLIP model - larger and more accurate
        model_name = "openai/clip-vit-large-patch14"  # Much better than base-patch32
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = self.model.to(self.device)
        
        # Define clothing prompts - more natural language
        self.clothing_prompts = [
            # Upper body - natural descriptions
            "a photo of a shirt",
            "a photo of a t-shirt",
            "a photo of a blouse",
            "a photo of a sweater",
            "a photo of a hoodie",
            "a photo of a jacket",
            "a photo of a coat",
            
            # Lower body
            "a photo of pants",
            "a photo of jeans",
            "a photo of shorts",
            "a photo of a skirt",
            "a photo of leggings",
            
            # Full body
            "a photo of a dress",
            
            # Footwear - more specific
            "a photo of shoes",
            "a photo of sneakers",
            "a photo of boots",
            "a photo of sandals",
            "a photo of bare feet",
            "a photo of a person wearing shoes",
            "footwear on feet",
            
            # Context-aware alternatives
            "clothing item: shirt",
            "clothing item: pants", 
            "clothing item: dress",
            "footwear: shoes",
            
            # Negative classes - more specific
            "a photo of a person",
            "a photo of a face",
            "a photo of hands",
            "a photo of bare skin",
            "a photo of a rock",
            "a photo of a stone",
            "a photo of the ground",
            "a photo of the floor",
            "background"
        ]
        
        # Simple label mapping
        self.label_map = {
            "a photo of a shirt": "shirt",
            "a photo of a t-shirt": "t-shirt",
            "a photo of a blouse": "blouse",
            "a photo of a sweater": "sweater",
            "a photo of a hoodie": "hoodie",
            "a photo of a jacket": "jacket",
            "a photo of a coat": "coat",
            "a photo of pants": "pants",
            "a photo of jeans": "jeans",
            "a photo of shorts": "shorts",
            "a photo of a skirt": "skirt",
            "a photo of leggings": "leggings",
            "a photo of a dress": "dress",
            "a photo of shoes": "shoes",
            "a photo of sneakers": "sneakers",
            "a photo of boots": "boots",
            "a photo of sandals": "sandals",
            "a photo of bare feet": "bare_feet",
            "a photo of a person wearing shoes": "shoes",
            "footwear on feet": "shoes",
            "clothing item: shirt": "shirt",
            "clothing item: pants": "pants",
            "clothing item: dress": "dress",
            "footwear: shoes": "shoes",
            "a photo of a person": "person",
            "a photo of a face": "face",
            "a photo of hands": "hands",
            "a photo of bare skin": "skin",
            "a photo of a rock": "rock",
            "a photo of a stone": "stone",
            "a photo of the ground": "ground",
            "a photo of the floor": "floor",
            "background": "background"
        }
        
    def classify(self, image_crop, threshold=0.15):
        """
        Classify clothing using CLIP
        
        Args:
            image_crop: PIL Image
            threshold: Confidence threshold
            
        Returns:
            Classification results
        """
        # Process inputs
        inputs = self.processor(
            text=self.clothing_prompts,
            images=image_crop,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLIP uses different output structure
            logits_per_image = outputs.logits_per_image
            # Use softmax for CLIP (it's designed for it)
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Get best match
        best_idx = probs.argmax().item()
        best_prompt = self.clothing_prompts[best_idx]
        best_label = self.label_map.get(best_prompt, best_prompt)
        best_conf = probs[best_idx].item()
        
        # Get top-k predictions
        top_k = 5
        top_indices = probs.topk(top_k).indices
        top_labels = []
        for idx in top_indices:
            prompt = self.clothing_prompts[idx]
            label = self.label_map.get(prompt, prompt)
            conf = probs[idx].item()
            if label not in ["person", "face", "background", "hands", "skin", "rock", "stone", "ground", "floor", "bare_feet"] and conf > threshold:
                top_labels.append((label, conf))
        
        # Create all scores
        all_scores = {}
        for i, prompt in enumerate(self.clothing_prompts):
            label = self.label_map.get(prompt, prompt)
            # Average scores for duplicate labels
            if label in all_scores:
                all_scores[label] = max(all_scores[label], probs[i].item())
            else:
                all_scores[label] = probs[i].item()
        
        return {
            'label': best_label,
            'confidence': best_conf,
            'top_labels': top_labels,
            'all_scores': all_scores,
            'model': 'CLIP'
        }
    
    def classify_with_position(self, image_crop, center_y, area_ratio):
        """Position-aware classification"""
        # Select relevant prompts based on position
        if center_y < 0.4:
            # Upper body
            selected_prompts = [
                "a photo of a shirt",
                "a photo of a t-shirt", 
                "a photo of a blouse",
                "a photo of a sweater",
                "a photo of a hoodie",
                "a photo of a jacket",
                "clothing item: shirt"
            ]
        elif center_y > 0.75:
            # Footwear area - be more specific
            selected_prompts = [
                "a photo of shoes",
                "a photo of sneakers",
                "a photo of boots",
                "a photo of sandals",
                "a photo of a person wearing shoes",
                "footwear on feet",
                # Add negative options that might appear in foot area
                "a photo of bare feet",
                "a photo of a rock",
                "a photo of a stone",
                "a photo of the ground",
                "a photo of the floor"
            ]
        else:
            # Lower/mid body
            selected_prompts = [
                "a photo of pants",
                "a photo of jeans",
                "a photo of shorts",
                "a photo of a skirt",
                "a photo of a dress",
                "clothing item: pants",
                "clothing item: dress"
            ]
        
        # Add negative classes
        selected_prompts.extend(["a photo of a person", "background"])
        
        # Process with selected prompts
        inputs = self.processor(
            text=selected_prompts,
            images=image_crop,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0]
        
        best_idx = probs.argmax().item()
        best_prompt = selected_prompts[best_idx]
        best_label = self.label_map.get(best_prompt, best_prompt)
        best_conf = probs[best_idx].item()
        
        return {
            'label': best_label,
            'confidence': best_conf,
            'model': 'CLIP-position'
        }


# Global instance
_clip_classifier = None

def get_clip_classifier():
    """Get or create global CLIP classifier"""
    global _clip_classifier
    if _clip_classifier is None:
        _clip_classifier = CLIPClothingClassifier()
    return _clip_classifier


def classify_with_clip(image_crop, processor=None, model=None, position_y=None, area_ratio=None):
    """
    Drop-in replacement for SigLIP classification using CLIP
    """
    classifier = get_clip_classifier()
    
    if position_y is not None and area_ratio is not None:
        return classifier.classify_with_position(image_crop, position_y, area_ratio)
    else:
        return classifier.classify(image_crop)
    

if __name__ == "__main__":
    # Test
    print("CLIP classifier ready for use")