import { storage } from './firebase';
import { ref, uploadBytes, getDownloadURL, listAll, deleteObject } from 'firebase/storage';

// Storage paths
const PATHS = {
  userImages: (userId) => `users/${userId}/images`,
  userMasks: (userId) => `users/${userId}/masks`,
  sharedGarments: () => `shared/garments`
};

// Upload user image
export const uploadUserImage = async (userId, file) => {
  try {
    const timestamp = Date.now();
    const fileName = `${timestamp}_${file.name}`;
    const storageRef = ref(storage, `${PATHS.userImages(userId)}/${fileName}`);
    
    const snapshot = await uploadBytes(storageRef, file);
    const downloadURL = await getDownloadURL(snapshot.ref);
    
    return {
      success: true,
      fileName,
      downloadURL,
      path: snapshot.ref.fullPath
    };
  } catch (error) {
    console.error('Error uploading image:', error);
    return { success: false, error: error.message };
  }
};

// Get user's images
export const getUserImages = async (userId) => {
  try {
    const listRef = ref(storage, PATHS.userImages(userId));
    const result = await listAll(listRef);
    
    const images = await Promise.all(
      result.items.map(async (itemRef) => {
        const url = await getDownloadURL(itemRef);
        return {
          name: itemRef.name,
          url,
          path: itemRef.fullPath
        };
      })
    );
    
    return { success: true, images };
  } catch (error) {
    console.error('Error getting user images:', error);
    return { success: false, error: error.message, images: [] };
  }
};

// Save mask data
export const saveMaskData = async (userId, imageName, maskData) => {
  try {
    const maskRef = ref(storage, `${PATHS.userMasks(userId)}/${imageName}/masks.json`);
    const blob = new Blob([JSON.stringify(maskData)], { type: 'application/json' });
    
    await uploadBytes(maskRef, blob);
    
    return { success: true };
  } catch (error) {
    console.error('Error saving mask data:', error);
    return { success: false, error: error.message };
  }
};

// Get mask data
export const getMaskData = async (userId, imageName) => {
  try {
    const maskRef = ref(storage, `${PATHS.userMasks(userId)}/${imageName}/masks.json`);
    const url = await getDownloadURL(maskRef);
    
    const response = await fetch(url);
    const data = await response.json();
    
    return { success: true, data };
  } catch (error) {
    console.error('Error getting mask data:', error);
    return { success: false, error: error.message, data: null };
  }
};

// Save mask image
export const saveMaskImage = async (userId, imageName, clothingType, imageBlob) => {
  try {
    const maskRef = ref(storage, `${PATHS.userMasks(userId)}/${imageName}/mask_${clothingType}.png`);
    
    await uploadBytes(maskRef, imageBlob);
    const downloadURL = await getDownloadURL(maskRef);
    
    return { success: true, downloadURL };
  } catch (error) {
    console.error('Error saving mask image:', error);
    return { success: false, error: error.message };
  }
};

// Get shared garments
export const getSharedGarments = async () => {
  try {
    const listRef = ref(storage, PATHS.sharedGarments());
    const result = await listAll(listRef);
    
    const garments = await Promise.all(
      result.items.map(async (itemRef) => {
        const url = await getDownloadURL(itemRef);
        return {
          name: itemRef.name,
          url,
          path: itemRef.fullPath
        };
      })
    );
    
    return { success: true, garments };
  } catch (error) {
    console.error('Error getting shared garments:', error);
    return { success: false, error: error.message, garments: [] };
  }
};

// Delete user image and associated masks
export const deleteUserImage = async (userId, imagePath) => {
  try {
    // Delete the image
    const imageRef = ref(storage, imagePath);
    await deleteObject(imageRef);
    
    // Get image name for mask deletion
    const imageName = imagePath.split('/').pop().split('.')[0];
    
    // Delete associated masks
    const maskFolderRef = ref(storage, `${PATHS.userMasks(userId)}/${imageName}`);
    const maskList = await listAll(maskFolderRef);
    
    await Promise.all(
      maskList.items.map(item => deleteObject(item))
    );
    
    return { success: true };
  } catch (error) {
    console.error('Error deleting image:', error);
    return { success: false, error: error.message };
  }
};