// Hugging Face API service for direct segmentation
import { db, storage } from '../firebase';
import { doc, setDoc } from 'firebase/firestore';
import { ref, uploadString, getDownloadURL } from 'firebase/storage';

const HF_API_URL = 'https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes';
const HF_API_TOKEN = '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'; // You'll need to add your token

// Label mapping from HF API response to our categories
const LABEL_MAPPING = {
  'Upper-clothes': 'shirt',
  'Skirt': 'skirt', 
  'Pants': 'pants',
  'Dress': 'dress',
  'Left-shoe': 'shoes',
  'Right-shoe': 'shoes',
  'Hat': 'non_clothing',
  'Hair': 'non_clothing',
  'Sunglasses': 'non_clothing',
  'Belt': 'non_clothing',
  'Face': 'non_clothing',
  'Left-leg': 'non_clothing',
  'Right-leg': 'non_clothing',
  'Left-arm': 'non_clothing',
  'Right-arm': 'non_clothing',
  'Bag': 'non_clothing',
  'Scarf': 'non_clothing',
  'Background': 'non_clothing'
};

const CLOTHING_LABELS = ['shirt', 'pants', 'shoes', 'dress', 'skirt'];

export const processImageWithHuggingFace = async (userId, imageBase64) => {
  try {
    // Extract base64 data from data URL if needed
    let base64Data = imageBase64;
    if (imageBase64.includes(',')) {
      base64Data = imageBase64.split(',')[1];
    }
    
    // Call Hugging Face API
    const response = await fetch(HF_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${HF_API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: base64Data
      })
    });

    if (!response.ok) {
      throw new Error('Failed to process image with Hugging Face');
    }

    // Get segmentation masks
    const result = await response.json();
    
    // Process the segmentation results
    const processedResults = processSegmentationResults(result);
    
    // Save to Firebase
    const saveResult = await saveToFirebase(userId, imageBase64, processedResults);
    
    return {
      success: true,
      ...saveResult
    };
  } catch (error) {
    console.error('HuggingFace processing error:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

const processSegmentationResults = (hfResult) => {
  // Parse HuggingFace response and extract clothing items
  const clothingMasks = [];
  const counts = {
    shirt: 0,
    pants: 0,
    shoes: 0
  };
  
  // HF API returns array of segments with label, score, and mask (base64)
  if (hfResult && Array.isArray(hfResult)) {
    hfResult.forEach((segment) => {
      const originalLabel = segment.label;
      const mappedLabel = LABEL_MAPPING[originalLabel];
      
      // Skip non-clothing items
      if (mappedLabel === 'non_clothing' || !mappedLabel) {
        return;
      }
      
      clothingMasks.push({
        label: mappedLabel,
        original_label: originalLabel,
        score: segment.score || 1.0,
        mask: segment.mask // base64 encoded mask image
      });
      
      // Update counts based on mapped label
      if (mappedLabel === 'shirt' || mappedLabel === 'dress') {
        counts.shirt++;
      } else if (mappedLabel === 'pants' || mappedLabel === 'skirt') {
        counts.pants++;
      } else if (mappedLabel === 'shoes') {
        counts.shoes++;
      }
    });
  }
  
  console.log('Processed segments:', counts);
  
  return {
    masks: clothingMasks,
    counts: counts
  };
};

const saveToFirebase = async (userId, originalImage, processedResults) => {
  try {
    // Generate timestamp for unique naming
    const timestamp = Date.now();
    const imageName = `${timestamp}_upload`;
    
    // 1. Save original image to Storage
    const imageRef = ref(storage, `users/${userId}/images/${imageName}.jpg`);
    await uploadString(imageRef, originalImage, 'data_url');
    const imageUrl = await getDownloadURL(imageRef);
    
    // 2. Prepare mask data for Firestore
    const maskData = {
      masks: processedResults.masks.map(m => ({
        label: m.label,
        confidence: m.score,
        area: 0, // HF API might not provide this
        bbox: [] // HF API might not provide this
      })),
      classifications: processedResults.counts,
      visualizations: {}, // We won't have visualizations from HF API
      closet_visualizations: {}, // We won't have these either
      timestamp: new Date().toISOString(),
      originalImageUrl: imageUrl,
      processedWithHF: true // Flag to indicate HF processing
    };
    
    // 3. Save to Firestore
    const maskDocRef = doc(db, 'users', userId, 'masks', imageName);
    await setDoc(maskDocRef, maskData);
    
    return {
      imageId: imageName,
      imageUrl: imageUrl,
      counts: processedResults.counts
    };
  } catch (error) {
    console.error('Firebase save error:', error);
    throw error;
  }
};