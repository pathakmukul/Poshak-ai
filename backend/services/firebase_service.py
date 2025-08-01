"""
Firebase Service - Centralized Firebase operations for Flask backend
All Firebase Storage and Firestore operations should go through this service.
This ensures web, mobile, and any future clients use the same backend logic.
"""

import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import json
import base64
from datetime import datetime
import io
from PIL import Image
import numpy as np

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    # Production: Use service account credentials
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
    else:
        # Use default credentials (for GCP environments)
        cred = credentials.ApplicationDefault()
    
    # Initialize with production bucket
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET', 'poshakai.appspot.com')
    })

# Get references
db = firestore.client()
bucket = storage.bucket()

# Export for use in other modules
__all__ = ['FirebaseService', 'bucket', 'db']

class FirebaseService:
    """Handle all Firebase operations for the application"""
    
    @staticmethod
    def upload_user_image(user_id, image_data, file_name=None):
        """
        Upload user image to Firebase Storage
        
        Args:
            user_id: User ID
            image_data: Either base64 string, PIL Image, or numpy array
            file_name: Optional filename (will generate if not provided)
            
        Returns:
            dict: {success: bool, url: str, path: str, fileName: str}
        """
        try:
            # Generate filename if not provided
            if not file_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name = f"{timestamp}_upload.png"
            
            # Ensure filename has extension
            if '.' not in file_name:
                file_name += '.png'
            
            # Convert image data to bytes
            if isinstance(image_data, str):
                # Handle base64 string
                if image_data.startswith('data:'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            elif isinstance(image_data, np.ndarray):
                # Convert numpy array to bytes
                pil_image = Image.fromarray(image_data)
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            elif isinstance(image_data, Image.Image):
                # Convert PIL Image to bytes
                buffer = io.BytesIO()
                image_data.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            else:
                # Assume it's already bytes
                image_bytes = image_data
            
            # Upload to Firebase Storage
            blob_path = f"users/{user_id}/images/{file_name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(image_bytes, content_type='image/png')
            
            # Make publicly accessible and get URL
            blob.make_public()
            url = blob.public_url
            
            return {
                'success': True,
                'url': url,
                'downloadURL': url,
                'path': blob_path,
                'fileName': file_name
            }
            
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def save_mask_data(user_id, image_name, mask_data, metadata=None):
        """
        Save mask data to Firebase Storage as JSON
        
        Args:
            user_id: User ID
            image_name: Image name (without extension)
            mask_data: Mask data dictionary
            metadata: Optional metadata
            
        Returns:
            dict: {success: bool}
        """
        try:
            # Prepare data to save
            data_to_save = {
                **mask_data,
                'timestamp': datetime.now().isoformat()
            }
            
            if metadata:
                data_to_save['metadata'] = metadata
            
            # Convert to JSON
            json_data = json.dumps(data_to_save)
            
            # Upload to Storage
            blob_path = f"users/{user_id}/masks/{image_name}/masks.json"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(json_data, content_type='application/json')
            
            return {'success': True}
            
        except Exception as e:
            print(f"Error saving mask data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_user_images(user_id):
        """
        Get all images for a user
        
        Args:
            user_id: User ID
            
        Returns:
            dict: {success: bool, images: list}
        """
        try:
            prefix = f"users/{user_id}/images/"
            blobs = bucket.list_blobs(prefix=prefix)
            
            images = []
            for blob in blobs:
                # Skip directories
                if blob.name.endswith('/'):
                    continue
                    
                # Get public URL
                url = blob.public_url
                    
                images.append({
                    'name': os.path.basename(blob.name),
                    'url': url,
                    'path': blob.name,
                    'updated': blob.updated.isoformat() if blob.updated else None
                })
            
            # Sort by updated time (newest first)
            images.sort(key=lambda x: x['updated'] or '', reverse=True)
            
            return {
                'success': True,
                'images': images
            }
            
        except Exception as e:
            print(f"Error getting user images: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'images': []
            }
    
    @staticmethod
    def get_mask_data(user_id, image_name):
        """
        Get mask data for a specific image
        
        Args:
            user_id: User ID
            image_name: Image name (without extension)
            
        Returns:
            dict: {success: bool, data: dict}
        """
        try:
            blob_path = f"users/{user_id}/masks/{image_name}/masks.json"
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                return {
                    'success': False,
                    'error': 'Mask data not found',
                    'data': None
                }
            
            # Download and parse JSON
            json_data = blob.download_as_text()
            mask_data = json.loads(json_data)
            
            return {
                'success': True,
                'data': mask_data
            }
            
        except Exception as e:
            print(f"Error getting mask data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    @staticmethod
    def delete_user_image(user_id, image_path):
        """
        Delete user image and associated masks
        
        Args:
            user_id: User ID
            image_path: Full path to image in Storage
            
        Returns:
            dict: {success: bool}
        """
        try:
            # Delete the main image
            blob = bucket.blob(image_path)
            if blob.exists():
                blob.delete()
            
            # Extract image name for mask deletion
            image_name = os.path.basename(image_path).split('.')[0]
            
            # Delete all associated masks
            mask_prefix = f"users/{user_id}/masks/{image_name}/"
            mask_blobs = bucket.list_blobs(prefix=mask_prefix)
            
            for mask_blob in mask_blobs:
                mask_blob.delete()
            
            return {'success': True}
            
        except Exception as e:
            print(f"Error deleting image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def save_processed_results(user_id, file_name, segmentation_results, original_image_base64=None):
        """
        Save complete processed results including person-only image and mask data
        
        Args:
            user_id: User ID
            file_name: Original filename
            segmentation_results: Results from image processing
            original_image_base64: Original image in base64 (optional)
            
        Returns:
            dict: {success: bool, imageUrl: str, imagePath: str, fileName: str}
        """
        try:
            # Save person-only image if available, otherwise save original
            image_to_save = None
            if segmentation_results.get('person_only_img'):
                # Person-only image is base64
                image_to_save = segmentation_results['person_only_img']
            elif original_image_base64:
                image_to_save = original_image_base64
            else:
                return {
                    'success': False,
                    'error': 'No image data to save'
                }
            
            # Upload the main image
            upload_result = FirebaseService.upload_user_image(
                user_id, 
                image_to_save,
                file_name
            )
            
            if not upload_result['success']:
                return upload_result
            
            # Prepare mask data
            image_name = file_name.split('.')[0]
            
            # Process masks to remove numpy arrays (can't serialize to JSON)
            masks_for_storage = []
            for mask in segmentation_results.get('masks', []):
                mask_copy = {
                    'label': mask.get('label'),
                    'bbox': mask.get('bbox'),
                    'area': mask.get('area'),
                    'confidence': mask.get('confidence', 1.0)
                }
                masks_for_storage.append(mask_copy)
            
            mask_data = {
                'masks': masks_for_storage,
                'classifications': {
                    'shirt': segmentation_results.get('shirt_count', 0),
                    'pants': segmentation_results.get('pants_count', 0),
                    'shoes': segmentation_results.get('shoes_count', 0)
                },
                'visualizations': {
                    'shirt': segmentation_results.get('shirt_img'),
                    'pants': segmentation_results.get('pants_img'),
                    'shoes': segmentation_results.get('shoes_img'),
                    'all': segmentation_results.get('all_items_img'),
                    'person_only': segmentation_results.get('person_only_img')
                },
                'closet_visualizations': segmentation_results.get('closet_visualizations', {}),
                'binary_masks': segmentation_results.get('binary_masks', {}),
                'descriptions': segmentation_results.get('descriptions', {}),
                'originalImageUrl': upload_result['url']
            }
            
            # Debug logging for descriptions
            print(f"\n[FirebaseService] Saving mask data for {image_name}")
            print(f"  Descriptions in segmentation_results: {segmentation_results.get('descriptions', {})}")
            print(f"  Descriptions being saved: {mask_data['descriptions']}")
            
            # Debug logging
            print(f"\n[save_processed_results] Saving closet visualizations:")
            for item_type, viz_data in mask_data['closet_visualizations'].items():
                if viz_data:
                    print(f"  {item_type}: {len(viz_data)} chars")
            
            metadata = segmentation_results.get('closet_metadata', {})
            
            # Save mask data
            mask_result = FirebaseService.save_mask_data(
                user_id,
                image_name,
                mask_data,
                metadata
            )
            
            if not mask_result['success']:
                # Rollback image upload if mask save fails
                FirebaseService.delete_user_image(user_id, upload_result['path'])
                return mask_result
            
            return {
                'success': True,
                'imageUrl': upload_result['url'],
                'imagePath': upload_result['path'],
                'fileName': upload_result['fileName']
            }
            
        except Exception as e:
            print(f"Error saving processed results: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def get_all_user_clothing_data(user_id):
        """Get all clothing data for a user in a single optimized call"""
        try:
            # Get all mask files in one go
            mask_blobs = bucket.list_blobs(prefix=f'users/{user_id}/masks/')
            
            all_data = {}
            
            for blob in mask_blobs:
                if blob.name.endswith('.json'):
                    # Download and parse the JSON
                    mask_data = json.loads(blob.download_as_text())
                    
                    # Extract image name from path
                    image_name = blob.name.split('/')[-1].replace('.json', '')
                    
                    all_data[image_name] = mask_data
            
            return {'success': True, 'data': all_data}
            
        except Exception as e:
            print(f"Error getting all user clothing data: {str(e)}")
            return {'success': False, 'error': str(e), 'data': {}}
    
    def save_trends_cache(self, cache_key, data):
        """Save fashion trends to Firebase cache"""
        try:
            doc_ref = db.collection('trends_cache').document(cache_key)
            doc_ref.set({
                'data': data,
                'timestamp': datetime.now(),
                'created_at': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Error saving trends cache: {str(e)}")
            return False
    
    def save_recommendations_cache(self, user_id, cache_key, data):
        """Save shopping recommendations to Firebase cache"""
        try:
            if user_id:
                doc_ref = db.collection('users').document(user_id).collection('recommendations_cache').document(cache_key)
            else:
                doc_ref = db.collection('recommendations_cache').document(cache_key)
                
            doc_ref.set({
                'data': data,
                'timestamp': datetime.now(),
                'created_at': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Error saving recommendations cache: {str(e)}")
            return False

# Legacy function for backwards compatibility
def save_segmentation_results(user_id, image_data, segmentation_results, original_image_base64):
    """
    Legacy function - redirects to new FirebaseService
    """
    return FirebaseService.save_processed_results(
        user_id,
        f"{int(datetime.now().timestamp() * 1000)}_upload.jpg",
        segmentation_results,
        original_image_base64
    )