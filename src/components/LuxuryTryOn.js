import { getMaskData } from '../storageService';
import API_URL from '../config';

// Try-on method configuration - switch between 'openai' and 'gemini'
const TRYON_METHOD = 'gemini'; // Change this to 'gemini' to use Gemini instead

// Handle Virtual Try On - OpenAI Implementation
export const handleOpenAITryOn = async ({ displayedItems, wardrobeItems, user, setTryOnLoading, setTryOnResult, setShowTryOnOverlay }) => {
  console.log('🎯 HANDLE OPENAI TRY ON STARTING');
  console.log('📦 Displayed Items:', displayedItems);
  console.log('👔 Has Shirt:', !!displayedItems.shirts);
  console.log('👖 Has Pants:', !!displayedItems.pants);
  console.log('👟 Has Shoes:', !!displayedItems.shoes);
  console.log('📸 Wardrobe items count:', wardrobeItems?.length || 0);
  console.log('📸 Wardrobe items:', wardrobeItems);
  
  if (!displayedItems.shirts && !displayedItems.pants && !displayedItems.shoes) {
    alert('Please select at least one item before trying on');
    return;
  }

  setTryOnLoading(true);
  try {
    // Use first wardrobe item as the model/person image (no mask needed for OpenAI)
    if (!wardrobeItems || wardrobeItems.length === 0) {
      throw new Error('No wardrobe items found. Please upload some photos to your wardrobe first.');
    }
    
    // Just use the first wardrobe item as model
    const selectedItem = wardrobeItems[0];
    const personImageUrl = selectedItem.url || selectedItem.imageUrl || selectedItem.image;
    
    console.log('Using wardrobe item as model:', selectedItem.name);

    // Prepare garment files in the order: shirt, pants, shoes
    const garmentFiles = {};
    
    // Process garments in specific order
    const garmentOrder = [
      { type: 'shirt', key: 'shirts' },
      { type: 'pants', key: 'pants' },
      { type: 'shoes', key: 'shoes' }
    ];
    
    for (const { type, key } of garmentOrder) {
      if (displayedItems[key]) {
        const garmentImage = displayedItems[key].image || displayedItems[key].url || displayedItems[key].imageUrl;
        
        if (garmentImage.startsWith('data:image')) {
          garmentFiles[type] = garmentImage;
          console.log(`📦 ${type}: Using data URL directly`);
        } else if (garmentImage.includes('firebasestorage.googleapis.com')) {
          // For Firebase URLs, fetch and convert to base64
          try {
            console.log(`🌐 ${type}: Fetching Firebase image...`);
            const response = await fetch(garmentImage);
            const blob = await response.blob();
            const reader = new FileReader();
            const base64Promise = new Promise((resolve) => {
              reader.onloadend = () => resolve(reader.result);
              reader.readAsDataURL(blob);
            });
            garmentFiles[type] = await base64Promise;
            console.log(`✅ ${type}: Converted Firebase URL to base64`);
          } catch (error) {
            console.log(`⚠️ ${type}: Failed to fetch Firebase image, passing URL directly`);
            garmentFiles[type] = garmentImage;
          }
        } else {
          // Assume it's a static garment filename
          garmentFiles[type] = garmentImage;
          console.log(`📁 ${type}: Using static filename`);
        }
      }
    }
    
    console.log('🎨 Garments collected:', Object.keys(garmentFiles));
    
    if (Object.keys(garmentFiles).length === 0) {
      throw new Error('No garments were selected. Please select at least one item.');
    }

    // Prepare payload for OpenAI - backend will handle URL fetching
    const tryOnPayload = {
      person_image: personImageUrl,  // Send URL directly, backend will fetch it
      garment_files: garmentFiles
    };
    
    console.log('🚀 === FINAL PAYLOAD TO OPENAI-TRYON ===');
    console.log('👤 Person image URL:', tryOnPayload.person_image);
    console.log('👗 Garment files collected:', Object.keys(tryOnPayload.garment_files));
    
    // Call OpenAI try-on endpoint
    const tryOnResponse = await fetch(`${API_URL}/openai-tryon`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tryOnPayload),
    });

    const tryOnData = await tryOnResponse.json();
    
    if (tryOnData.success) {
      setTryOnResult(tryOnData.result_image);
      setShowTryOnOverlay(true); // Show overlay when new result is ready
    } else {
      throw new Error(tryOnData.error || 'Try-on failed');
    }
  } catch (error) {
    console.error('OpenAI Try-on error:', error);
    alert(`Failed to generate virtual try-on: ${error.message}`);
  } finally {
    setTryOnLoading(false);
  }
};

// Handle Virtual Try On - Gemini Implementation
export const handleGeminiTryOn = async ({ displayedItems, wardrobeItems, user, setTryOnLoading, setTryOnResult, setShowTryOnOverlay }) => {
  console.log('🎯 HANDLE TRY ON STARTING');
  console.log('📦 Displayed Items:', displayedItems);
  console.log('👔 Has Shirt:', !!displayedItems.shirts);
  console.log('👖 Has Pants:', !!displayedItems.pants);
  console.log('👟 Has Shoes:', !!displayedItems.shoes);
  
  if (!displayedItems.shirts || !displayedItems.pants || !displayedItems.shoes) {
    alert('Please select all three items (shirt, pants, and shoes) before trying on');
    return;
  }

  setTryOnLoading(true);
  try {
    // Use wardrobe items (photos of people) as the person image
    if (wardrobeItems.length === 0) {
      throw new Error('No wardrobe items found. Please upload some photos to your wardrobe first.');
    }
    
    // Find a wardrobe item that has mask data (has been segmented)
    let selectedItem = null;
    let maskData = null;
    
    for (const item of wardrobeItems) {
      // Try to get mask data for this item
      let imageName;
      if (item.name) {
        imageName = item.name.split('.')[0];
      } else {
        console.log('Wardrobe item missing name:', item);
        continue;
      }
      
      console.log('Checking mask data for:', imageName);
      const maskResult = await getMaskData(user.uid, imageName);
      
      if (maskResult.success && maskResult.data) {
        // Found an item with mask data!
        selectedItem = item;
        maskData = maskResult.data;
        console.log('Found wardrobe item with mask data:', item.name);
        break;
      }
    }
    
    if (!selectedItem || !maskData) {
      throw new Error('No segmented wardrobe items found. Please go to Wardrobe and segment some photos first.');
    }
    
    const personImageUrl = selectedItem.url || selectedItem.imageUrl || selectedItem.image;
    
    // We already have maskData from the loop above
    console.log('Using mask data:', maskData);

    const maskImages = {};
    const garmentFiles = {};

    // Get mask data for each clothing type using prepare-wardrobe-gemini
    const clothingTypes = ['shirt', 'pants', 'shoes'];
    
    console.log('🎭 === STARTING MASK PROCESSING ===');
    console.log('📊 Mask data available:', maskData);
    console.log('🖼️ Visualizations:', maskData.visualizations ? Object.keys(maskData.visualizations) : 'none');
    console.log('📋 Classifications:', maskData.classifications);
    console.log('🎯 Processing clothing types:', clothingTypes);
    
    for (const type of clothingTypes) {
      console.log(`🔄 Loop iteration for ${type}`);
      // Handle pluralization: shirt->shirts, pants->pants, shoes->shoes
      const categoryKey = type === 'pants' || type === 'shoes' ? type : type + 's';
      console.log(`📌 Checking displayedItems[${categoryKey}]:`, displayedItems[categoryKey]);
      
      // For each type we want to try on, check if we have a garment selected
      if (displayedItems[categoryKey]) {
        console.log(`🔍 Processing ${type}: has displayed item`);
        
        // Don't check masks - send all selected items to backend
        console.log(`🎯 Processing ${type} - sending to backend`);
        
        // Get mask for this type using prepare_wardrobe_gemini (following Wardrobe.js pattern)
        console.log(`📤 Calling prepare-wardrobe-gemini for ${type}...`);
        try {
          const geminiDataResponse = await fetch(`${API_URL}/prepare-wardrobe-gemini`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_url: personImageUrl,
              mask_data: maskData,
              clothing_type: type
            }),
          });

          if (!geminiDataResponse.ok) {
            console.log(`❌ Failed to call prepare-wardrobe-gemini for ${type}: ${geminiDataResponse.status}`);
            continue;
          }
          
          const geminiData = await geminiDataResponse.json();
          console.log(`📨 Response for ${type}:`, {
            success: geminiData.success,
            hasError: !!geminiData.error,
            error: geminiData.error
          });
          
          // Get the garment image URL regardless of mask success
          const garmentImage = displayedItems[categoryKey].image || displayedItems[categoryKey].url || displayedItems[categoryKey].imageUrl;
          console.log(`👕 Garment for ${type}:`, garmentImage.substring(0, 100) + '...');
          
          if (geminiData.success) {
            maskImages[type] = `data:image/png;base64,${geminiData.mask_image}`;
            console.log(`✅ Got mask for ${type}`);
          } else {
            console.log(`⚠️ No mask available for ${type}:`, geminiData.error);
            // For luxury closet, we'll create a placeholder mask or let backend handle it
            // Still add the garment!
          }
          
          // ALWAYS add the garment if it's selected, regardless of mask availability
          if (garmentImage.startsWith('data:image')) {
            garmentFiles[type] = garmentImage;
            console.log(`📦 ${type}: Using data URL directly`);
          } else if (garmentImage.includes('firebasestorage.googleapis.com')) {
            // For Firebase URLs, try to fetch and convert to base64
            try {
              console.log(`🌐 ${type}: Fetching Firebase image...`);
              const response = await fetch(garmentImage);
              const blob = await response.blob();
              const reader = new FileReader();
              const base64Promise = new Promise((resolve) => {
                reader.onloadend = () => resolve(reader.result);
                reader.readAsDataURL(blob);
              });
              garmentFiles[type] = await base64Promise;
              console.log(`✅ ${type}: Converted Firebase URL to base64`);
            } catch (error) {
              console.log(`⚠️ ${type}: Failed to fetch Firebase image, passing URL directly`);
              garmentFiles[type] = garmentImage;
            }
          } else {
            // Assume it's a static garment filename
            garmentFiles[type] = garmentImage;
            console.log(`📁 ${type}: Using static filename`);
          }
        } catch (error) {
          console.log(`💥 Exception calling prepare-wardrobe-gemini for ${type}:`, error);
        }
      } else {
        console.log(`⏭️ Skipping ${type}: no displayed item`);
      }
    }

    console.log('🏁 === MASK COLLECTION COMPLETE ===');
    console.log('✅ Masks collected:', Object.keys(maskImages));
    console.log('📦 Garments collected:', Object.keys(garmentFiles));
    console.log('🔍 Mask count:', Object.keys(maskImages).length);
    console.log('🔍 Garment count:', Object.keys(garmentFiles).length);
    
    // For luxury closet, we want to send all garments even if masks aren't available
    // The backend will need to handle creating appropriate masks or use a different approach
    console.log(`🎨 Preparing to send ${Object.keys(garmentFiles).length} garments for try-on`);
    
    if (Object.keys(garmentFiles).length === 0) {
      throw new Error('No garments were collected. Please try again.');
    }
    
    // Check which items we couldn't process
    const missingTypes = [];
    for (const type of clothingTypes) {
      const categoryKey = type === 'shoes' ? 'shoes' : type + 's';
      if (displayedItems[categoryKey] && !maskImages[type]) {
        missingTypes.push(type);
      }
    }
    
    if (missingTypes.length > 0) {
      console.log(`⚠️ Missing types: ${missingTypes.join(', ')}`);
      console.log(`⚠️ Note: Cannot try on ${missingTypes.join(', ')} because the person in the photo isn't wearing these items`);
    }

    // Get original image in base64 using the same endpoint
    const originalResponse = await fetch(`${API_URL}/prepare-wardrobe-gemini`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: personImageUrl,
        mask_data: maskData,
        clothing_type: Object.keys(maskImages)[0] // Just to get the original image
      }),
    });

    const originalData = await originalResponse.json();
    if (!originalData.success) {
      throw new Error('Failed to prepare original image');
    }

    // Prepare final payload - use garmentFiles keys for clothing_types
    const tryOnPayload = {
      person_image: `data:image/png;base64,${originalData.original_image}`,
      mask_images: maskImages,
      garment_files: garmentFiles,
      clothing_types: Object.keys(garmentFiles) // Use garments, not masks!
    };
    
    console.log('🚀 === FINAL PAYLOAD TO GEMINI-TRYON-MULTIPLE ===');
    console.log('👤 Person image: base64 data (length:', tryOnPayload.person_image.length, ')');
    console.log('🎭 Mask images collected:', Object.keys(tryOnPayload.mask_images));
    console.log('👗 Garment files collected:', Object.keys(tryOnPayload.garment_files));
    console.log('📋 Clothing types being sent:', tryOnPayload.clothing_types);
    console.log('🎯 Full garment files object:', tryOnPayload.garment_files);
    console.log('📏 Mask images count:', Object.keys(maskImages).length);
    console.log('📏 Garment files count:', Object.keys(garmentFiles).length);
    
    // Log each garment file
    Object.entries(garmentFiles).forEach(([type, file]) => {
      console.log(`🔍 ${type} garment: ${file.substring(0, 100)}...`);
    });
    
    // Call gemini-tryon-multiple endpoint (following Wardrobe.js pattern exactly)
    const tryOnResponse = await fetch(`${API_URL}/gemini-tryon-multiple`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tryOnPayload),
    });

    const tryOnData = await tryOnResponse.json();
    
    if (tryOnData.success) {
      setTryOnResult(tryOnData.result_image);
      setShowTryOnOverlay(true); // Show overlay when new result is ready
    } else {
      throw new Error(tryOnData.error || 'Try-on failed');
    }
  } catch (error) {
    console.error('Try-on error:', error);
    alert(`Failed to generate virtual try-on: ${error.message}`);
  } finally {
    setTryOnLoading(false);
  }
};

// Unified try-on handler that uses the configured method
export const handleTryOn = async (params) => {
  console.log(`🎨 Using ${TRYON_METHOD.toUpperCase()} for virtual try-on`);
  if (TRYON_METHOD === 'openai') {
    return handleOpenAITryOn(params);
  } else {
    return handleGeminiTryOn(params);
  }
};