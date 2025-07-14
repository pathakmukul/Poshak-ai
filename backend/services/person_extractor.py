"""
Person extraction service using MediaPipe Selfie Segmentation.
Extracts person from background to reduce noise for SAM2 processing.
"""

import numpy as np
import cv2
import mediapipe as mp


class PersonExtractor:
    def __init__(self, model_selection=1):
        """
        Initialize MediaPipe selfie segmentation.
        
        Args:
            model_selection: 0 for general model, 1 for landscape model (better for full body)
        """
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
    
    def extract_person(self, image, padding_percent=10):
        """
        Extract person from image using MediaPipe selfie segmentation.
        
        Args:
            image: Input image as numpy array (RGB)
            padding_percent: Percentage of padding to add around bounding box
            
        Returns:
            tuple: (cropped_image, bbox, mask) or (None, None, None) if no person detected
            - cropped_image: Cropped RGB image containing the person
            - bbox: (x, y, width, height) of the person in original image
            - mask: Binary mask of the person in original image
        """
        if image is None or image.size == 0:
            return None, None, None
            
        # Process with MediaPipe
        results = self.selfie_segmentation.process(image)
        
        if results.segmentation_mask is None:
            return None, None, None
            
        # Convert mask to binary (0 or 255)
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Find contours to get bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
            
        # Get the largest contour (assume it's the person)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        height, width = image.shape[:2]
        padding_x = int(w * padding_percent / 100)
        padding_y = int(h * padding_percent / 100)
        
        # Calculate padded bbox with bounds checking
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(width, x + w + padding_x)
        y_end = min(height, y + h + padding_y)
        
        # Update bbox with padding
        bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
        
        # Crop the image
        cropped_image = image[y_start:y_end, x_start:x_end]
        
        return cropped_image, bbox, mask
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'selfie_segmentation'):
            self.selfie_segmentation.close()


def extract_person_from_image(image_np, padding_percent=10):
    """
    Convenience function to extract person from a single image.
    
    Args:
        image_np: Input image as numpy array (RGB)
        padding_percent: Percentage of padding to add around bounding box
        
    Returns:
        tuple: (cropped_image, bbox, mask) or (None, None, None) if no person detected
    """
    extractor = PersonExtractor(model_selection=1)  # Use landscape model for full body
    result = extractor.extract_person(image_np, padding_percent)
    del extractor  # Clean up resources
    return result