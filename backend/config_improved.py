"""
Improved configuration for SAM2 - optimized for clothing segmentation
"""

# SAM2 configuration moved to services/sam2_service.py

# Removed SigLIP config - using CLIP only

# Classification thresholds
CLASSIFICATION_CONFIG = {
    "shirt_max_y": 0.6,
    "pants_min_y": 0.4,
    "pants_max_y": 0.85,
    "shoes_min_y": 0.8,
    "max_background_ratio": 0.5,
    "max_shoe_area": 10000,
}

# Performance config removed - not used

# Person extraction configuration
PERSON_EXTRACTION_CONFIG = {
    "use_person_extraction": True,  # Enable MediaPipe person extraction
    "padding_percent": 10,  # Padding around detected person
    "model_selection": 1,  # 0 for general, 1 for landscape (better for full body)
}

# Debug options
DEBUG_CONFIG = {
    "save_intermediate_images": False,
    "print_all_scores": False,
    "timing_details": True,
    "print_mask_stats": True,
    "use_clip_instead": True,
}