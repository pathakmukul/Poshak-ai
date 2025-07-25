import config from '../config';

// Service that calls Flask backend instead of direct Firebase access

// This function is deprecated - use Flask backend instead
export const uploadUserImage = async (userId, imageUri, fileName) => {
  console.warn('uploadUserImage is deprecated. Use Flask /firebase/save-results endpoint instead');
  return { success: false, error: 'Please use Flask backend for uploads' };
};

// Get user's images via Flask backend
export const getUserImages = async (userId) => {
  try {
    const response = await fetch(`${config.API_URL}/firebase/images/${userId}`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting user images:', error);
    return { success: false, error: error.message, images: [] };
  }
};

// This function is deprecated - use Flask backend instead
export const saveMaskData = async (userId, imageName, maskData, metadata = {}) => {
  console.warn('saveMaskData is deprecated. Use Flask /firebase/save-results endpoint instead');
  return { success: false, error: 'Please use Flask backend for saving data' };
};

// Get mask data via Flask backend
export const getMaskData = async (userId, imageName) => {
  try {
    const response = await fetch(`${config.API_URL}/firebase/mask-data/${userId}/${imageName}`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting mask data:', error);
    return { success: false, error: error.message, data: null };
  }
};

// This function is deprecated - Flask handles all mask saving
export const saveMaskImage = async (userId, imageName, clothingType, base64String) => {
  console.warn('saveMaskImage is deprecated. Masks are saved automatically via Flask');
  return { success: false, error: 'Please use Flask backend' };
};

// Get shared garments via Flask backend
export const getSharedGarments = async () => {
  try {
    const response = await fetch(`${config.API_URL}/garments`);
    const data = await response.json();
    return { 
      success: true, 
      garments: data.garments?.map(g => ({
        name: g,
        url: `${config.API_URL}/static/garments/${g}`,
        path: `shared/garments/${g}`
      })) || [] 
    };
  } catch (error) {
    console.error('Error getting shared garments:', error);
    return { success: false, error: error.message, garments: [] };
  }
};

// Delete user image via Flask backend
export const deleteUserImage = async (userId, imagePath) => {
  try {
    const response = await fetch(`${config.API_URL}/firebase/delete-image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        image_path: imagePath
      }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error deleting image:', error);
    return { success: false, error: error.message };
  }
};