import AsyncStorage from '@react-native-async-storage/async-storage';

const CACHE_KEYS = {
  CLOSET_ITEMS: 'closet_items_',
  CLOSET_METADATA: 'closet_metadata_',
  LAST_SYNC: 'last_sync_',
  SESSION_ID: 'session_id_',
  SESSION_COUNT_CHECKED: 'session_count_checked_'
};

export const CacheService = {
  // Save closet items to local storage
  saveClosetItems: async (userId, items) => {
    try {
      const key = CACHE_KEYS.CLOSET_ITEMS + userId;
      await AsyncStorage.setItem(key, JSON.stringify(items));
      
      // Update metadata
      const metaKey = CACHE_KEYS.CLOSET_METADATA + userId;
      const metadata = {
        shirtsCount: items.shirts?.length || 0,
        pantsCount: items.pants?.length || 0,
        shoesCount: items.shoes?.length || 0,
        lastUpdated: new Date().toISOString()
      };
      await AsyncStorage.setItem(metaKey, JSON.stringify(metadata));
      
      return true;
    } catch (error) {
      console.error('Error saving closet cache:', error);
      return false;
    }
  },

  // Get cached closet items
  getCachedClosetItems: async (userId) => {
    try {
      const key = CACHE_KEYS.CLOSET_ITEMS + userId;
      const cached = await AsyncStorage.getItem(key);
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      console.error('Error reading closet cache:', error);
      return null;
    }
  },

  // Get closet metadata (counts, last update)
  getClosetMetadata: async (userId) => {
    try {
      const key = CACHE_KEYS.CLOSET_METADATA + userId;
      const metadata = await AsyncStorage.getItem(key);
      return metadata ? JSON.parse(metadata) : null;
    } catch (error) {
      console.error('Error reading closet metadata:', error);
      return null;
    }
  },

  // Add a single new item to cache
  addItemToCache: async (userId, type, item) => {
    try {
      const cached = await CacheService.getCachedClosetItems(userId) || {
        shirts: [],
        pants: [],
        shoes: []
      };
      
      // Add to appropriate category
      if (cached[type]) {
        cached[type].unshift(item); // Add to beginning
      }
      
      await CacheService.saveClosetItems(userId, cached);
      return true;
    } catch (error) {
      console.error('Error adding item to cache:', error);
      return false;
    }
  },

  // Clear cache for a user
  clearCache: async (userId) => {
    try {
      await AsyncStorage.removeItem(CACHE_KEYS.CLOSET_ITEMS + userId);
      await AsyncStorage.removeItem(CACHE_KEYS.CLOSET_METADATA + userId);
      await AsyncStorage.removeItem(CACHE_KEYS.LAST_SYNC + userId);
      return true;
    } catch (error) {
      console.error('Error clearing cache:', error);
      return false;
    }
  },

  // Check if we need to sync with backend - only on app open
  shouldSync: async (userId) => {
    try {
      // This function is now only used on app open or manual refresh
      // We don't do automatic periodic syncs
      return false;
    } catch (error) {
      return false;
    }
  },
  
  // Check if counts match backend
  needsCountSync: async (userId, backendCounts) => {
    try {
      const metadata = await CacheService.getClosetMetadata(userId);
      if (!metadata) return true;
      
      // Compare counts
      const countsMatch = 
        metadata.shirtsCount === backendCounts.shirts &&
        metadata.pantsCount === backendCounts.pants &&
        metadata.shoesCount === backendCounts.shoes;
      
      return !countsMatch;
    } catch (error) {
      return true;
    }
  },

  // Mark sync completed
  markSynced: async (userId) => {
    try {
      const key = CACHE_KEYS.LAST_SYNC + userId;
      await AsyncStorage.setItem(key, new Date().toISOString());
    } catch (error) {
      console.error('Error marking sync:', error);
    }
  },
  
  // Session management for smart sync
  startNewSession: async (userId) => {
    try {
      const sessionId = Date.now().toString();
      await AsyncStorage.setItem(CACHE_KEYS.SESSION_ID + userId, sessionId);
      // Clear session count checked flag
      await AsyncStorage.removeItem(CACHE_KEYS.SESSION_COUNT_CHECKED + userId);
      return sessionId;
    } catch (error) {
      console.error('Error starting session:', error);
      return null;
    }
  },
  
  // Check if we already checked counts in this session
  hasCheckedCountsThisSession: async (userId) => {
    try {
      const currentSessionId = await AsyncStorage.getItem(CACHE_KEYS.SESSION_ID + userId);
      const checkedSessionId = await AsyncStorage.getItem(CACHE_KEYS.SESSION_COUNT_CHECKED + userId);
      return currentSessionId === checkedSessionId;
    } catch (error) {
      return false;
    }
  },
  
  // Mark that we checked counts this session
  markCountsChecked: async (userId) => {
    try {
      const currentSessionId = await AsyncStorage.getItem(CACHE_KEYS.SESSION_ID + userId);
      if (currentSessionId) {
        await AsyncStorage.setItem(CACHE_KEYS.SESSION_COUNT_CHECKED + userId, currentSessionId);
      }
    } catch (error) {
      console.error('Error marking counts checked:', error);
    }
  }
};