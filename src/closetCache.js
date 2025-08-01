// Cache management for closet items
const CACHE_KEY_PREFIX = 'closet_items_';
const CACHE_TIMESTAMP_KEY = 'closet_cache_timestamp_';
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

export const closetCache = {
  // Get cached items for a user
  get: (userId) => {
    try {
      const timestampKey = CACHE_TIMESTAMP_KEY + userId;
      const timestamp = localStorage.getItem(timestampKey);
      
      if (!timestamp) return null;
      
      const age = Date.now() - parseInt(timestamp);
      if (age > CACHE_DURATION) {
        // Cache expired
        closetCache.clear(userId);
        return null;
      }
      
      const dataKey = CACHE_KEY_PREFIX + userId;
      const cached = localStorage.getItem(dataKey);
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      console.error('Error reading from cache:', error);
      return null;
    }
  },
  
  // Set cached items for a user
  set: (userId, items) => {
    try {
      const dataKey = CACHE_KEY_PREFIX + userId;
      const timestampKey = CACHE_TIMESTAMP_KEY + userId;
      
      localStorage.setItem(dataKey, JSON.stringify(items));
      localStorage.setItem(timestampKey, Date.now().toString());
    } catch (error) {
      console.error('Error writing to cache:', error);
      // If storage is full, clear old cache entries
      if (error.name === 'QuotaExceededError') {
        closetCache.clearAll();
      }
    }
  },
  
  // Clear cache for a specific user
  clear: (userId) => {
    try {
      const dataKey = CACHE_KEY_PREFIX + userId;
      const timestampKey = CACHE_TIMESTAMP_KEY + userId;
      
      localStorage.removeItem(dataKey);
      localStorage.removeItem(timestampKey);
    } catch (error) {
      console.error('Error clearing cache:', error);
    }
  },
  
  // Clear all closet caches
  clearAll: () => {
    try {
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith(CACHE_KEY_PREFIX) || key.startsWith(CACHE_TIMESTAMP_KEY)) {
          localStorage.removeItem(key);
        }
      });
    } catch (error) {
      console.error('Error clearing all caches:', error);
    }
  },
  
  // Check if cache exists and is valid
  isValid: (userId) => {
    try {
      const timestampKey = CACHE_TIMESTAMP_KEY + userId;
      const timestamp = localStorage.getItem(timestampKey);
      
      if (!timestamp) return false;
      
      const age = Date.now() - parseInt(timestamp);
      return age <= CACHE_DURATION;
    } catch (error) {
      console.error('Error checking cache validity:', error);
      return false;
    }
  }
};