import config from '../config';
import { CacheService } from './cacheService';
import AsyncStorage from '@react-native-async-storage/async-storage';

const VIRTUAL_CLOSET_KEY = 'virtualCloset_';

// Save virtual try-on result
export const saveVirtualTryOn = async (userId, tryOnData) => {
  try {
    // Create item with unique ID
    const item = {
      id: Date.now().toString(),
      userId,
      ...tryOnData,
      createdAt: new Date().toISOString(),
    };
    
    // Save to local storage first
    const key = `${VIRTUAL_CLOSET_KEY}${userId}`;
    const existingData = await AsyncStorage.getItem(key);
    const items = existingData ? JSON.parse(existingData) : [];
    items.unshift(item); // Add to beginning
    await AsyncStorage.setItem(key, JSON.stringify(items));
    
    // Sync to Firebase in background
    syncToFirebase(userId, item);
    
    return { success: true, item };
  } catch (error) {
    console.error('Error saving virtual try-on:', error);
    return { success: false, error: error.message };
  }
};

// Get all virtual closet items
export const getVirtualClosetItems = async (userId, forceRefresh = false) => {
  try {
    // First, try to get from local storage
    if (!forceRefresh) {
      const key = `${VIRTUAL_CLOSET_KEY}${userId}`;
      const cached = await AsyncStorage.getItem(key);
      if (cached) {
        console.log('[virtualClosetService] Returning cached data');
        // Sync in background
        syncFromFirebase(userId);
        return {
          success: true,
          items: JSON.parse(cached),
          fromCache: true
        };
      }
    }
    
    console.log('[virtualClosetService] Fetching from backend...');
    
    // Fetch from Firebase via Flask
    const response = await fetch(`${config.API_URL}/firebase/virtual-closet/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch virtual closet items');
    }
    
    const data = await response.json();
    
    // Save to local storage
    if (data.success && data.items) {
      const key = `${VIRTUAL_CLOSET_KEY}${userId}`;
      await AsyncStorage.setItem(key, JSON.stringify(data.items));
    }
    
    return data;
  } catch (error) {
    console.error('Error loading virtual closet:', error);
    
    // If network error, try to return cached data
    const key = `${VIRTUAL_CLOSET_KEY}${userId}`;
    const cached = await AsyncStorage.getItem(key);
    if (cached) {
      console.log('[virtualClosetService] Network error, returning cached data');
      return {
        success: true,
        items: JSON.parse(cached),
        fromCache: true,
        error: 'Using cached data due to network error'
      };
    }
    
    return {
      success: false,
      error: error.message,
      items: []
    };
  }
};

// Delete virtual closet item
export const deleteVirtualClosetItem = async (userId, itemId) => {
  try {
    // Remove from local storage first
    const key = `${VIRTUAL_CLOSET_KEY}${userId}`;
    const existingData = await AsyncStorage.getItem(key);
    if (existingData) {
      const items = JSON.parse(existingData);
      const filtered = items.filter(item => item.id !== itemId);
      await AsyncStorage.setItem(key, JSON.stringify(filtered));
    }
    
    // Delete from Firebase in background
    fetch(`${config.API_URL}/firebase/virtual-closet/${userId}/${itemId}`, {
      method: 'DELETE'
    }).catch(err => console.error('Error deleting from Firebase:', err));
    
    return { success: true };
  } catch (error) {
    console.error('Error deleting virtual closet item:', error);
    return { success: false, error: error.message };
  }
};

// Sync single item to Firebase
async function syncToFirebase(userId, item) {
  try {
    await fetch(`${config.API_URL}/firebase/virtual-closet`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ userId, item })
    });
  } catch (error) {
    console.error('Error syncing to Firebase:', error);
  }
}

// Sync all items from Firebase
async function syncFromFirebase(userId) {
  try {
    const response = await fetch(`${config.API_URL}/firebase/virtual-closet/${userId}`);
    if (response.ok) {
      const data = await response.json();
      if (data.success && data.items) {
        const key = `${VIRTUAL_CLOSET_KEY}${userId}`;
        await AsyncStorage.setItem(key, JSON.stringify(data.items));
      }
    }
  } catch (error) {
    console.error('Error syncing from Firebase:', error);
  }
}