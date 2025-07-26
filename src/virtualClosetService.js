import API_URL from './config';

const VIRTUAL_CLOSET_COLLECTION = 'virtualCloset';
const LOCAL_STORAGE_KEY = 'virtualCloset_';

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

// Clear localStorage if it's full
export const clearLocalStorageIfNeeded = () => {
  try {
    // Try to set a test item
    localStorage.setItem('test', 'test');
    localStorage.removeItem('test');
  } catch (e) {
    // Storage is full, clear virtual closet items
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
      if (key.startsWith(LOCAL_STORAGE_KEY)) {
        localStorage.removeItem(key);
      }
    });
  }
};

// Get all virtual closet items (same as mobile)
export const getVirtualClosetItems = async (userId, forceRefresh = false) => {
  try {
    // Always fetch from Firebase via Flask (no localStorage for web due to size)
    console.log('[virtualClosetService] Fetching from backend...');
    
    const response = await fetch(`${API_URL}/firebase/virtual-closet/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch virtual closet items');
    }
    
    const data = await response.json();
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