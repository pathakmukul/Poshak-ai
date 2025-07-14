"""
Improved configuration for SAM2 - optimized for clothing segmentation
"""

# SAM2 Configuration - Enhanced for better clothing detection
SAM2_CONFIG = {
    "points_per_side": 32,  # 2304 points for better coverage
    "points_per_batch": 128,  # Process more points at once
    "pred_iou_thresh": 0.75,  # Lower threshold to catch more segments
    "stability_score_thresh": 0.88,  # More permissive for clothing
    "stability_score_offset": 0.7,  # Lower offset for more mask variations
    "mask_threshold": -0.5,  # Negative for softer boundaries
    "box_nms_thresh": 0.5,  # Lower NMS to keep overlapping clothing items
    "crop_n_layers": 1,  # Multi-scale helps with shoes and accessories
    "crop_nms_thresh": 0.7,  # Standard crop NMS
    "crop_overlap_ratio": 0.4,  # More overlap for better boundaries
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,  # Filter out noise but keep small items
    "multimask_output": True,  # Get 3 masks per point
    "use_m2m": True,  # Mask-to-mask refinement for cleaner boundaries
}

# Get the active configuration
def get_active_sam2_config():
    """Returns the SAM2 configuration"""
    return SAM2_CONFIG

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

# Performance optimizations
PERFORMANCE_CONFIG = {
    "resize_for_detection": False,
    "max_image_size": 1024,
    "skip_very_large_masks": True,
    "batch_classification": False,
    "adaptive_quality": True,
    "max_retry_attempts": 2,
}

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