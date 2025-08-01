// All Firebase operations go through Flask backend
// No direct Firebase imports needed
import { clearUserClothingCache } from './closetService';

// Simple cache for wardrobe images
const wardrobeCache = new Map();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Upload user image - This should be handled by UploadSegmentModal which calls Flask
export const uploadUserImage = async (userId, file) => {
  console.warn('Direct upload not supported. Use UploadSegmentModal component.');
  return { 
    success: false, 
    error: 'Please use the upload modal which processes images through Flask' 
  };
};

// Get user's images via Flask backend with caching
export const getUserImages = async (userId, forceRefresh = false) => {
  try {
    // Check cache first unless force refresh
    if (!forceRefresh) {
      const cached = wardrobeCache.get(userId);
      if (cached && cached.timestamp > Date.now() - CACHE_DURATION) {
        console.log('Using cached wardrobe images');
        return cached.data;
      }
    }
    
    console.log('Fetching wardrobe images from server...');
    const response = await fetch(`http://localhost:5001/firebase/images/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch images');
    }
    const data = await response.json();
    
    // Cache successful response
    if (data.success) {
      wardrobeCache.set(userId, {
        data: data,
        timestamp: Date.now()
      });
      console.log('Wardrobe images cached successfully');
    }
    
    return data;
  } catch (error) {
    console.error('Error getting user images:', error);
    return { success: false, error: error.message, images: [] };
  }
};

// Clear wardrobe cache for a user
export const clearWardrobeCache = (userId) => {
  wardrobeCache.delete(userId);
  console.log('Wardrobe cache cleared for user:', userId);
};

// Save mask data - This should be handled by Flask
export const saveMaskData = async (userId, imageName, maskData, metadata = {}) => {
  console.warn('Direct mask save not supported. Use Flask endpoints.');
  return { 
    success: false, 
    error: 'Mask data should be saved through Flask during image processing' 
  };
};

// Get person-only image URL for a specific image
export const getPersonOnlyUrl = async (userId, imageName) => {
  try {
    // Get mask data from Flask
    const maskData = await getMaskData(userId, imageName);
    if (!maskData.success || !maskData.data) {
      return null;
    }
    
    // Check if person-only visualization exists
    if (maskData.data.visualizations?.person_only) {
      return `data:image/png;base64,${maskData.data.visualizations.person_only}`;
    }
    
    return null;
  } catch (error) {
    return null;
  }
};

// Get mask data via Flask backend
export const getMaskData = async (userId, imageName) => {
  try {
    const response = await fetch(`http://localhost:5001/firebase/mask-data/${userId}/${imageName}`);
    if (!response.ok) {
      throw new Error('Failed to fetch mask data');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting mask data:', error);
    return { success: false, error: error.message, data: null };
  }
};

// Save mask image - This should be handled by Flask
export const saveMaskImage = async (userId, imageName, clothingType, imageBlob) => {
  console.warn('Direct mask image save not supported. Use Flask endpoints.');
  return { 
    success: false, 
    error: 'Mask images are saved through Flask during processing' 
  };
};

// Get shared garments via Flask
export const getSharedGarments = async () => {
  try {
    const response = await fetch('http://localhost:5001/garments');
    if (!response.ok) {
      throw new Error('Failed to fetch garments');
    }
    const data = await response.json();
    
    // Transform to expected format
    const garments = (data.garments || []).map(name => ({
      name: name,
      url: `http://localhost:5001/static/garments/${name}`,
      path: `shared/garments/${name}`
    }));
    
    return { success: true, garments };
  } catch (error) {
    console.error('Error getting shared garments:', error);
    return { success: false, error: error.message, garments: [] };
  }
};

// Delete user image via Flask backend
export const deleteUserImage = async (userId, imagePath) => {
  try {
    const response = await fetch('http://localhost:5001/firebase/delete-image', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        image_path: imagePath
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to delete image');
    }
    
    const data = await response.json();
    
    // Clear cache after successful deletion
    if (data.success) {
      clearWardrobeCache(userId);
      clearUserClothingCache(userId);
      console.log('Wardrobe and clothing cache cleared after image deletion');
    }
    
    return data;
  } catch (error) {
    console.error('Error deleting image:', error);
    return { success: false, error: error.message };
  }
};