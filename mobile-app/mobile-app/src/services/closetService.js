import config from '../config';
import { CacheService } from './cacheService';

// Get all clothing items for a user - OPTIMIZED version using Flask backend
export const getUserClothingItems = async (userId, forceRefresh = false) => {
  try {
    // First, try to get from cache if not forcing refresh
    if (!forceRefresh) {
      const cached = await CacheService.getCachedClosetItems(userId);
      if (cached) {
        console.log('[closetService] Returning cached data');
        
        // Check counts in background (only on closet open, not during session)
        checkCountsInBackground(userId);
        
        return {
          success: true,
          shirts: cached.shirts || [],
          pants: cached.pants || [],
          shoes: cached.shoes || [],
          fromCache: true
        };
      }
    }
    
    console.log('[closetService] Fetching from backend...');
    
    // Use optimized Flask endpoint that gets all clothing items in one call
    const response = await fetch(`${config.API_URL}/firebase/clothing-items/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch clothing items');
    }
    
    const data = await response.json();
    
    // Log the raw data for debugging
    console.log('[closetService] Raw response data:', {
      success: data.success,
      shirtsCount: data.shirts?.length || 0,
      pantsCount: data.pants?.length || 0,
      shoesCount: data.shoes?.length || 0
    });
    
    // Log sample items to check data structure
    if (data.shirts?.length > 0) {
      console.log('[closetService] Sample shirt item:', {
        id: data.shirts[0].id,
        type: data.shirts[0].type,
        imageLength: data.shirts[0].image?.length || 0,
        imagePrefix: data.shirts[0].image?.substring(0, 50),
        hasContentBounds: !!data.shirts[0].contentBounds
      });
    }
    
    // Save to cache
    if (data.success) {
      await CacheService.saveClosetItems(userId, {
        shirts: data.shirts || [],
        pants: data.pants || [],
        shoes: data.shoes || []
      });
      await CacheService.markSynced(userId);
    }
    
    return data;
  } catch (error) {
    console.error('Error loading clothing items:', error);
    
    // If network error, try to return cached data
    const cached = await CacheService.getCachedClosetItems(userId);
    if (cached) {
      console.log('[closetService] Network error, returning cached data');
      return {
        success: true,
        shirts: cached.shirts || [],
        pants: cached.pants || [],
        shoes: cached.shoes || [],
        fromCache: true,
        error: 'Using cached data due to network error'
      };
    }
    
    return {
      success: false,
      error: error.message,
      shirts: [],
      pants: [],
      shoes: []
    };
  }
};

// Check counts and sync if needed - smart sync strategy
async function checkCountsInBackground(userId) {
  try {
    // Only check counts once per session (on app open)
    const alreadyChecked = await CacheService.hasCheckedCountsThisSession(userId);
    if (alreadyChecked) {
      console.log('[closetService] Already checked counts this session, skipping');
      return;
    }
    
    console.log('[closetService] Checking counts for smart sync...');
    
    // Get counts from backend
    const response = await fetch(`${config.API_URL}/firebase/clothing-counts/${userId}`);
    if (!response.ok) return;
    
    const backendCounts = await response.json();
    if (!backendCounts.success) return;
    
    // Mark that we checked counts this session
    await CacheService.markCountsChecked(userId);
    
    // Check if counts match
    const needsSync = await CacheService.needsCountSync(userId, backendCounts);
    
    if (needsSync) {
      console.log('[closetService] Count mismatch detected, syncing...');
      // Full sync needed
      await syncInBackground(userId);
    } else {
      console.log('[closetService] Counts match, no sync needed');
    }
  } catch (error) {
    console.error('[closetService] Count check failed:', error);
  }
}

// Background sync function - only called when counts mismatch
async function syncInBackground(userId) {
  try {
    console.log('[closetService] Starting full background sync...');
    const response = await fetch(`${config.API_URL}/firebase/clothing-items/${userId}`);
    if (!response.ok) return;
    
    const data = await response.json();
    if (data.success) {
      await CacheService.saveClosetItems(userId, {
        shirts: data.shirts || [],
        pants: data.pants || [],
        shoes: data.shoes || []
      });
      await CacheService.markSynced(userId);
      console.log('[closetService] Background sync completed');
    }
  } catch (error) {
    console.error('[closetService] Background sync failed:', error);
  }
}