// All Firebase operations go through Flask backend

// Get all clothing items for a user - OPTIMIZED for speed
export const getUserClothingItems = async (userId) => {
  try {
    // Use optimized endpoint that gets all clothing items in one call
    const response = await fetch(`http://localhost:5001/firebase/clothing-items/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch clothing items');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error loading clothing items:', error);
    return {
      success: false,
      error: error.message,
      shirts: [],
      pants: [],
      shoes: []
    };
  }
};