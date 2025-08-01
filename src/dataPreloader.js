// Preload user data in the background after login
import { getUserClothingItems } from './closetService';
import { getUserImages } from './storageService';
import { getVirtualClosetItems } from './virtualClosetService';

// Track preloading status
const preloadStatus = {
  clothing: false,
  wardrobe: false,
  virtualCloset: false
};

// Preload all user data in the background
export const preloadUserData = async (userId) => {
  console.log('Starting background data preload for user:', userId);
  
  // Reset status
  preloadStatus.clothing = false;
  preloadStatus.wardrobe = false;
  preloadStatus.virtualCloset = false;
  
  // Load all data in parallel
  const promises = [
    // Preload clothing items
    getUserClothingItems(userId).then(data => {
      preloadStatus.clothing = true;
      console.log('Clothing items preloaded');
      return data;
    }).catch(err => {
      console.error('Failed to preload clothing:', err);
      preloadStatus.clothing = true; // Mark as done even on error
    }),
    
    // Preload wardrobe images
    getUserImages(userId).then(data => {
      preloadStatus.wardrobe = true;
      console.log('Wardrobe images preloaded');
      return data;
    }).catch(err => {
      console.error('Failed to preload wardrobe:', err);
      preloadStatus.wardrobe = true;
    }),
    
    // Preload virtual closet items
    getVirtualClosetItems(userId).then(data => {
      preloadStatus.virtualCloset = true;
      console.log('Virtual closet items preloaded');
      return data;
    }).catch(err => {
      console.error('Failed to preload virtual closet:', err);
      preloadStatus.virtualCloset = true;
    })
  ];
  
  // Don't await - let it run in background
  Promise.all(promises).then(() => {
    console.log('All data preloaded successfully');
  }).catch(err => {
    console.error('Error during preload:', err);
  });
};

// Get preload status
export const getPreloadStatus = () => {
  return { ...preloadStatus };
};

// Check if all data is preloaded
export const isAllDataPreloaded = () => {
  return preloadStatus.clothing && preloadStatus.wardrobe && preloadStatus.virtualCloset;
};