import { storage } from './firebase';
import { ref, listAll, getDownloadURL } from 'firebase/storage';

// Get all clothing items for a user
export const getUserClothingItems = async (userId) => {
  try {
    // List all mask folders
    const masksRef = ref(storage, `users/${userId}/masks`);
    const maskFolders = await listAll(masksRef);
    
    const allShirts = [];
    const allPants = [];
    const allShoes = [];
    
    // Process each image folder
    for (const folderRef of maskFolders.prefixes) {
      const imageName = folderRef.name;
      
      try {
        // Get the masks.json file
        const maskDataRef = ref(storage, `users/${userId}/masks/${imageName}/masks.json`);
        const maskDataUrl = await getDownloadURL(maskDataRef);
        
        // Fetch and parse the mask data
        const response = await fetch(maskDataUrl);
        const maskData = await response.json();
        
        const closetVisualizations = maskData.closet_visualizations || {};
        const classifications = maskData.classifications || {};
        
        // Use closet visualizations if available, otherwise fall back to old format
        if (closetVisualizations.shirt || closetVisualizations.pants || closetVisualizations.shoes) {
          // New format with proper visualizations
          if (closetVisualizations.shirt && classifications.shirt > 0) {
            allShirts.push({
              id: `${imageName}_shirt`,
              image: closetVisualizations.shirt,
              type: 'shirt',
              source_image: imageName,
              isClosetViz: true
            });
          }
          
          if (closetVisualizations.pants && classifications.pants > 0) {
            allPants.push({
              id: `${imageName}_pants`,
              image: closetVisualizations.pants,
              type: 'pants',
              source_image: imageName,
              isClosetViz: true
            });
          }
          
          if (closetVisualizations.shoes && classifications.shoes > 0) {
            allShoes.push({
              id: `${imageName}_shoes`,
              image: closetVisualizations.shoes,
              type: 'shoes',
              source_image: imageName,
              isClosetViz: true
            });
          }
        } else {
          // Fall back to old visualization format for existing data
          const visualizations = maskData.visualizations || {};
          
          if (visualizations.shirt && classifications.shirt > 0) {
            allShirts.push({
              id: `${imageName}_shirt`,
              image: visualizations.shirt,
              type: 'shirt',
              source_image: imageName,
              isClosetViz: false
            });
          }
          
          if (visualizations.pants && classifications.pants > 0) {
            allPants.push({
              id: `${imageName}_pants`,
              image: visualizations.pants,
              type: 'pants',
              source_image: imageName,
              isClosetViz: false
            });
          }
          
          if (visualizations.shoes && classifications.shoes > 0) {
            allShoes.push({
              id: `${imageName}_shoes`,
              image: visualizations.shoes,
              type: 'shoes',
              source_image: imageName,
              isClosetViz: false
            });
          }
        }
      } catch (error) {
        console.error(`Error loading masks for ${imageName}:`, error);
        // Continue with next folder
      }
    }
    
    return {
      success: true,
      shirts: allShirts,
      pants: allPants,
      shoes: allShoes
    };
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