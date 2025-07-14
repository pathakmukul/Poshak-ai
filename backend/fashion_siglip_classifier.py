"""
Fashion-specific SigLIP classification using Marqo-FashionSigLIP
This model is specifically fine-tuned for fashion items and uses sigmoid for multi-label classification
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

class FashionSigLIPClassifier:
    def __init__(self, use_fashion_model=False):  # Disabled due to loading issues
        """
        Initialize the classifier
        
        Args:
            use_fashion_model: If True, use Marqo-FashionSigLIP (currently disabled)
                             If False, use standard SigLIP with better prompts
        """
        # For now, always use standard SigLIP due to Fashion model loading issues
        model_name = 'google/siglip-base-patch16-224'
        print("Loading SigLIP with optimized clothing prompts...")
            
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = self.model.to(self.device)
        
        # Define clothing categories with descriptive prompts
        self.clothing_categories = [
            # Upper body - more descriptive
            "clothing shirt",
            "clothing t-shirt", 
            "clothing blouse",
            "clothing sweater",
            "clothing hoodie",
            "clothing jacket",
            "clothing coat",
            "clothing top",
            
            # Lower body - more descriptive
            "clothing pants",
            "clothing jeans",
            "clothing shorts",
            "clothing skirt",
            "clothing leggings",
            "clothing trousers",
            
            # Full body
            "clothing dress",
            "full body suit",
            
            # Footwear - more descriptive
            "footwear shoes",
            "footwear sneakers",
            "footwear boots",
            "footwear sandals",
            
            # Not clothing (for filtering)
            "human person",
            "human face",
            "background wall floor"
        ]
        
        # Mapping back to simple labels
        self.label_mapping = {
            "clothing shirt": "shirt",
            "clothing t-shirt": "t-shirt",
            "clothing blouse": "blouse",
            "clothing sweater": "sweater",
            "clothing hoodie": "hoodie",
            "clothing jacket": "jacket",
            "clothing coat": "coat",
            "clothing top": "top",
            "clothing pants": "pants",
            "clothing jeans": "jeans",
            "clothing shorts": "shorts",
            "clothing skirt": "skirt",
            "clothing leggings": "leggings",
            "clothing trousers": "trousers",
            "clothing dress": "dress",
            "full body suit": "suit",
            "footwear shoes": "shoes",
            "footwear sneakers": "sneakers",
            "footwear boots": "boots",
            "footwear sandals": "sandals",
            "human person": "person",
            "human face": "face",
            "background wall floor": "background"
        }
        
    def classify_clothing(self, image_crop, use_sigmoid=False, threshold=0.15):
        """
        Classify a clothing item using SigLIP
        
        Args:
            image_crop: PIL Image of the masked region
            use_sigmoid: If True, use sigmoid (multi-label). If False, use softmax (single-label)
            threshold: Confidence threshold for sigmoid mode
            
        Returns:
            dict with label, confidence, and multi-label predictions
        """
        # Process inputs
        inputs = self.processor(
            text=self.clothing_categories,
            images=image_crop,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            # SigLIP uses logits_per_image
            logits = outputs.logits_per_image[0]
            
            # SigLIP logits need temperature scaling
            temperature = 100.0  # SigLIP uses temperature=100 by default
            logits = logits * temperature
        
        if use_sigmoid:
            # Use sigmoid for multi-label (SigLIP's native approach)
            # Note: sigmoid without proper loss training often gives very low values
            probs = torch.sigmoid(logits / temperature)  # Scale back for sigmoid
            
            # Get all labels above threshold
            high_conf_indices = torch.where(probs > threshold)[0]
            multi_labels = []
            for idx in high_conf_indices:
                raw_label = self.clothing_categories[idx.item()]
                simple_label = self.label_mapping.get(raw_label, raw_label)
                conf = probs[idx].item()
                if simple_label not in ["person", "face", "background"]:  # Filter non-clothing
                    multi_labels.append((simple_label, conf))
            
            # Sort by confidence
            multi_labels.sort(key=lambda x: x[1], reverse=True)
            
            # Get best single label
            best_idx = probs.argmax().item()
            raw_best_label = self.clothing_categories[best_idx]
            best_label = self.label_mapping.get(raw_best_label, raw_best_label)
            best_conf = probs[best_idx].item()
            
            # Create all_scores with simple labels
            all_scores = {}
            for i, raw_cat in enumerate(self.clothing_categories):
                simple_cat = self.label_mapping.get(raw_cat, raw_cat)
                all_scores[simple_cat] = probs[i].item()
            
            return {
                'label': best_label,
                'confidence': best_conf,
                'multi_labels': multi_labels,
                'all_scores': all_scores,
                'mode': 'sigmoid'
            }
        else:
            # Use softmax for single-label (works better in practice)
            # Don't use temperature for softmax
            probs = torch.softmax(logits / temperature, dim=0)
            
            best_idx = probs.argmax().item()
            raw_best_label = self.clothing_categories[best_idx]
            best_label = self.label_mapping.get(raw_best_label, raw_best_label)
            best_conf = probs[best_idx].item()
            
            # Create all_scores with simple labels
            all_scores = {}
            for i, raw_cat in enumerate(self.clothing_categories):
                simple_cat = self.label_mapping.get(raw_cat, raw_cat)
                all_scores[simple_cat] = probs[i].item()
            
            return {
                'label': best_label,
                'confidence': best_conf,
                'all_scores': all_scores,
                'mode': 'softmax'
            }
    
    def classify_with_position_context(self, image_crop, center_y, area_ratio):
        """
        Classify with position-based context
        
        Args:
            image_crop: PIL Image
            center_y: Y position (0-1)
            area_ratio: Area ratio (0-1)
        """
        # Create position-aware prompts with descriptive labels
        if center_y < 0.4:
            # Upper body
            context_labels = ["clothing shirt", "clothing t-shirt", "clothing blouse", 
                            "clothing sweater", "clothing hoodie", "clothing jacket", "clothing top"]
        elif center_y > 0.75:
            # Footwear
            context_labels = ["footwear shoes", "footwear sneakers", "footwear boots", "footwear sandals"]
        elif center_y > 0.4 and center_y < 0.75:
            # Lower body
            if area_ratio > 0.04:
                # Larger area - could be dress
                context_labels = ["clothing pants", "clothing jeans", "clothing shorts", 
                                "clothing skirt", "clothing dress"]
            else:
                context_labels = ["clothing pants", "clothing jeans", "clothing shorts", 
                                "clothing skirt", "clothing trousers"]
        else:
            # Use all categories
            context_labels = self.clothing_categories
        
        # Process with context labels
        inputs = self.processor(
            text=context_labels,
            images=image_crop,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            
            # SigLIP logits need temperature scaling
            temperature = 100.0
            logits = logits * temperature
        
        # Use softmax for single-label classification (works better)
        # Even though this is position-based, we want single best label
        probs = torch.softmax(logits / temperature, dim=0)
        
        best_idx = probs.argmax().item()
        raw_best_label = context_labels[best_idx]
        # Map back to simple label
        best_label = self.label_mapping.get(raw_best_label, raw_best_label)
        best_conf = probs[best_idx].item()
        
        # Create all_scores with simple labels
        all_scores = {}
        for i, raw_cat in enumerate(context_labels):
            simple_cat = self.label_mapping.get(raw_cat, raw_cat)
            all_scores[simple_cat] = probs[i].item()
        
        return {
            'label': best_label,
            'confidence': best_conf,
            'context_labels': [self.label_mapping.get(l, l) for l in context_labels],
            'all_scores': all_scores
        }


# Global classifier instance (singleton pattern for efficiency)
_fashion_classifier = None

def get_fashion_classifier():
    """Get or create the global Fashion SigLIP classifier"""
    global _fashion_classifier
    if _fashion_classifier is None:
        _fashion_classifier = FashionSigLIPClassifier(use_fashion_model=False)  # Use standard SigLIP
    return _fashion_classifier


# Simplified function for Flask integration
def classify_with_fashion_siglip(image_crop, processor=None, model=None, position_y=None, area_ratio=None):
    """
    Classify clothing using Fashion SigLIP
    
    This is a drop-in replacement for the improved_siglip_classification functions
    Note: processor and model args are ignored since we use the Fashion model
    """
    # Get the global classifier instance
    classifier = get_fashion_classifier()
    
    # Use position context if available
    if position_y is not None and area_ratio is not None:
        result = classifier.classify_with_position_context(image_crop, position_y, area_ratio)
    else:
        # Use softmax for single-label classification (more reliable)
        result = classifier.classify_clothing(image_crop, use_sigmoid=False, threshold=0.15)
    
    return result


# Test function
if __name__ == "__main__":
    # Test with a sample image
    test_image_path = "test_clothing.jpg"
    
    # Initialize classifier
    classifier = FashionSigLIPClassifier(use_fashion_model=True)
    
    # Test classification
    try:
        from PIL import Image
        img = Image.open(test_image_path)
        result = classifier.classify_clothing(img, use_sigmoid=True)
        print(f"Classification result: {result}")
    except FileNotFoundError:
        print("Test image not found. Please provide a test image.")