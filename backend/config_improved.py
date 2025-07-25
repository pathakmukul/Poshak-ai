"""
Configuration for KapdaAI - Segformer-based clothing detection
"""

# Segformer configuration
SEGFORMER_CONFIG = {
    "model": "mattmdjaga/segformer_b2_clothes",
    "use_gpu": True,  # Use GPU if available (MPS on Apple Silicon)
    "min_mask_area": 100,  # Minimum area for a valid mask in pixels
}

# Classification thresholds (for backward compatibility with mask editing)
CLASSIFICATION_CONFIG = {
    "shirt_max_y": 0.6,
    "pants_min_y": 0.4,
    "pants_max_y": 0.85,
    "shoes_min_y": 0.8,
    "max_background_ratio": 0.5,
    "max_shoe_area": 10000,
}

# Debug options
DEBUG_CONFIG = {
    "save_intermediate_images": False,
    "print_all_scores": False,
    "timing_details": True,
    "print_mask_stats": True,
}