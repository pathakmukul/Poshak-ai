"""
Improved configuration using Facebook's SAM2 parameters
Based on analysis of FB's automatic_mask_generator_example.ipynb
"""

# SAM2 Configuration - Using Facebook's proven parameters
SAM2_CONFIG = {
    # STANDARD MODE - Facebook's default settings
    "standard": {
        "points_per_side": 32,  # FB default: 32x32 = 1024 points
        "points_per_batch": 64,  # FB default
        "pred_iou_thresh": 0.88,  # FB default
        "stability_score_thresh": 0.95,  # FB default
        "stability_score_offset": 1.0,  # FB default (we were missing this!)
        "box_nms_thresh": 0.7,  # FB default (critical for deduplication!)
        "crop_n_layers": 0,  # FB default
        "crop_n_points_downscale_factor": 1,  # FB default
        "min_mask_region_area": 0,  # FB default (no filtering)
        "use_m2m": False,  # FB default (can enable for better quality)
    },
    
    # HIGH QUALITY MODE - Facebook's dense configuration
    "high_quality": {
        "points_per_side": 64,  # FB dense: 64x64 = 4096 points!
        "points_per_batch": 128,  # More parallel processing
        "pred_iou_thresh": 0.7,  # Lower threshold for more masks
        "stability_score_thresh": 0.92,  # Slightly lower
        "stability_score_offset": 0.7,  # More aggressive offset
        "box_nms_thresh": 0.7,  # Keep deduplication
        "crop_n_layers": 1,  # Multi-scale for small objects
        "crop_n_points_downscale_factor": 2,  # FB dense setting
        "min_mask_region_area": 25.0,  # Small cleanup threshold
        "use_m2m": True,  # Enable mask-to-mask refinement
    },
    
    # FAST MODE - Balanced for speed
    "fast": {
        "points_per_side": 24,  # 24x24 = 576 points
        "points_per_batch": 64,
        "pred_iou_thresh": 0.85,  # Slightly lower than FB
        "stability_score_thresh": 0.93,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 50,  # Filter tiny fragments
        "use_m2m": False,
    },
    
    # CLOTHING OPTIMIZED - Best for fashion items
    "clothing": {
        "points_per_side": 40,  # 1600 points - good coverage
        "points_per_batch": 128,
        "pred_iou_thresh": 0.80,  # Lower to catch more clothing
        "stability_score_thresh": 0.90,  # More permissive
        "stability_score_offset": 0.8,
        "box_nms_thresh": 0.65,  # Slightly lower for overlapping clothes
        "crop_n_layers": 1,  # Important for shoes!
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100,  # Small enough for accessories
        "use_m2m": True,  # Better boundaries
    }
}

# Model-specific overrides
MODEL_OVERRIDES = {
    "tiny": {
        # Tiny model needs more aggressive settings
        "pred_iou_thresh": 0.75,
        "stability_score_thresh": 0.88,
    },
    "small": {
        # Small model with balanced settings
        "pred_iou_thresh": 0.80,
    },
    "base": {
        # Base model can use standard settings
    },
    "large": {
        # Large model can be more selective
        "pred_iou_thresh": 0.85,
        "points_per_side": 48,  # Can handle more points
    }
}

# Active configuration selection
ACTIVE_CONFIG = "standard"  # Options: "standard", "high_quality", "fast", "clothing"

# Get the active configuration
def get_active_sam2_config():
    """Returns the currently active SAM2 configuration"""
    return SAM2_CONFIG.get(ACTIVE_CONFIG, SAM2_CONFIG["standard"])

# SigLIP Classification Configuration (unchanged)
SIGLIP_CONFIG = {
    "use_context": True,
    "context_padding": 20,
    "context_dim_factor": 0.3,
}

# Classification thresholds (unchanged)
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
    
    # New options
    "adaptive_quality": True,  # Automatically adjust based on initial results
    "max_retry_attempts": 2,  # Retry with higher quality if poor results
}

# Debug options
DEBUG_CONFIG = {
    "save_intermediate_images": False,
    "print_all_scores": False,
    "timing_details": True,
    "print_mask_stats": True,  # Show mask count and quality metrics
    "use_improved_siglip": False,  # Use improved SigLIP classification with templates
    "use_fashion_siglip": False,  # Use optimized SigLIP with better prompts
    "use_clip_instead": True,  # Use CLIP instead of SigLIP (experimental)
}