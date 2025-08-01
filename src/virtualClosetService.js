import API_URL from './config';

const VIRTUAL_CLOSET_COLLECTION = 'virtualCloset';
const LOCAL_STORAGE_KEY = 'virtualCloset_';

// Helper to estimate localStorage usage
const getStorageSize = () => {
  let total = 0;
  for (let key in localStorage) {
    if (localStorage.hasOwnProperty(key)) {
      total += localStorage[key].length + key.length;
    }
  }
  return (total / 1024).toFixed(2) + ' KB';
};

// Save virtual try-on result via Flask backend (same as mobile)
export const saveVirtualTryOn = async (userId, tryOnData) => {
  try {
    const item = {
      id: Date.now().toString(),
      userId,
      ...tryOnData,
      createdAt: new Date().toISOString(),
    };
    
    const response = await fetch(`${API_URL}/firebase/virtual-closet`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ userId, item }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to save virtual try-on');
    }
    
    const result = await response.json();
    
    // Clear localStorage cache to force refresh
    const localKey = `${LOCAL_STORAGE_KEY}${userId}`;
    localStorage.removeItem(localKey);
    
    return { success: true, item };
  } catch (error) {
    console.error('Error saving virtual try-on:', error);
    return { success: false, error: error.message };
  }
};

// Clear localStorage if it's full - more aggressive cleanup
export const clearLocalStorageIfNeeded = () => {
  try {
    // Try to set a test item
    localStorage.setItem('test', 'test');
    localStorage.removeItem('test');
  } catch (e) {
    console.log(`Storage full (${getStorageSize()}), clearing old data...`);
    // Storage is full, clear ALL cached data to make room
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
      // Clear virtual closet, wardrobe, and other cached data
      if (key.startsWith(LOCAL_STORAGE_KEY) || 
          key.startsWith('wardrobe_') || 
          key.startsWith('clothingItems_') ||
          key.startsWith('masks_')) {
        localStorage.removeItem(key);
      }
    });
    
    // Also clear old/expired items
    keys.forEach(key => {
      try {
        const data = localStorage.getItem(key);
        if (data && data.includes('timestamp')) {
          const parsed = JSON.parse(data);
          // Remove items older than 7 days
          if (parsed.timestamp && Date.now() - parsed.timestamp > 7 * 24 * 60 * 60 * 1000) {
            localStorage.removeItem(key);
          }
        }
      } catch (e) {
        // If we can't parse it, it might be corrupted, remove it
        localStorage.removeItem(key);
      }
    });
  }
};

// Get all virtual closet items with caching
export const getVirtualClosetItems = async (userId, forceRefresh = false) => {
  try {
    const localKey = `${LOCAL_STORAGE_KEY}${userId}`;
    
    // Check cache first unless force refresh
    if (!forceRefresh) {
      const cachedData = localStorage.getItem(localKey);
      if (cachedData) {
        try {
          const parsed = JSON.parse(cachedData);
          console.log('[virtualClosetService] Loaded from cache');
          return {
            success: true,
            items: parsed.items || [],
            fromCache: true
          };
        } catch (e) {
          console.error('Error parsing cached data:', e);
          localStorage.removeItem(localKey);
        }
      }
    }
    
    // Fetch from backend if no cache or force refresh
    console.log('[virtualClosetService] Fetching from backend...');
    const response = await fetch(`${API_URL}/firebase/virtual-closet/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch virtual closet items');
    }
    
    const data = await response.json();
    
    // Cache the results
    if (data.success && data.items) {
      const dataToCache = JSON.stringify({
        items: data.items,
        timestamp: Date.now()
      });
      
      try {
        // First, try to clear some space proactively
        clearLocalStorageIfNeeded();
        
        // Try to save
        localStorage.setItem(localKey, dataToCache);
        console.log('[virtualClosetService] Cached successfully');
      } catch (e) {
        console.warn('Failed to cache virtual closet data:', e);
        
        // If still failing, clear more aggressively
        try {
          // Clear all virtual closet cache
          const keys = Object.keys(localStorage);
          keys.forEach(key => {
            if (key.startsWith(LOCAL_STORAGE_KEY) || key.includes('wardrobe') || key.includes('clothing')) {
              localStorage.removeItem(key);
            }
          });
          
          // Try one more time
          localStorage.setItem(localKey, dataToCache);
          console.log('[virtualClosetService] Cached after cleanup');
        } catch (e2) {
          console.error('Storage is completely full, running without cache');
        }
      }
    }
    
    return data;
  } catch (error) {
    console.error('Error loading virtual closet:', error);
    return {
      success: false,
      error: error.message,
      items: []
    };
  }
};

// Delete virtual closet item (same as mobile)
export const deleteVirtualClosetItem = async (userId, itemId) => {
  try {
    const response = await fetch(`${API_URL}/firebase/virtual-closet/${userId}/${itemId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      throw new Error('Failed to delete item');
    }
    
    return { success: true };
  } catch (error) {
    console.error('Error deleting virtual closet item:', error);
    return { success: false, error: error.message };
  }
};