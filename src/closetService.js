// All Firebase operations go through Flask backend
import { closetCache } from './closetCache';

// Get all clothing items for a user - WITH CACHING
export const getUserClothingItems = async (userId, forceRefresh = false) => {
  try {
    // Check cache first unless force refresh is requested
    if (!forceRefresh) {
      const cached = closetCache.get(userId);
      if (cached) {
        console.log('Using cached clothing items');
        return { ...cached, fromCache: true };
      }
    }
    
    console.log('Fetching clothing items from server...');
    const response = await fetch(`http://localhost:5001/firebase/clothing-items/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch clothing items');
    }
    
    const data = await response.json();
    
    // Cache successful response
    if (data.success) {
      closetCache.set(userId, data);
      console.log('Clothing items cached successfully');
    }
    
    return data;
  } catch (error) {
    console.error('Error loading clothing items:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

// Clear cache for a user (call after add/delete operations)
export const clearUserClothingCache = (userId) => {
  closetCache.clear(userId);
  console.log('Clothing cache cleared for user:', userId);
};