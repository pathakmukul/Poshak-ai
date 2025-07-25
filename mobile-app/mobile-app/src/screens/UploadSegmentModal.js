import React, { useState } from 'react';
import {
  View,
  Text,
  Modal,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  Dimensions,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import config from '../config';
import { CacheService } from '../services/cacheService';

const { width: screenWidth } = Dimensions.get('window');

function UploadSegmentModal({ visible, onClose, user, onSuccess, navigation }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [segmentationResults, setSegmentationResults] = useState(null);
  const [selectedItems, setSelectedItems] = useState({
    shirt: true,
    pants: true,
    shoes: true
  });

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,  // Don't force cropping
      quality: 1,
      base64: true,
    });

    if (!result.canceled) {
      setSelectedImage({
        uri: result.assets[0].uri,
        base64: result.assets[0].base64,
      });
    }
  };

  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,  // Don't force cropping
      quality: 1,
      base64: true,
    });

    if (!result.canceled) {
      setSelectedImage({
        uri: result.assets[0].uri,
        base64: result.assets[0].base64,
      });
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;

    setProcessing(true);
    setStatus('Processing image...');

    try {
      // Step 1: Process image with Flask (same as web)
      console.log('Processing image with Flask...');
      const response = await fetch(`${config.API_URL}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_data: `data:image/jpeg;base64,${selectedImage.base64}`,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to process image');
      }

      const segmentResults = await response.json();
      console.log('Segmentation complete');
      console.log('Segmentation results keys:', Object.keys(segmentResults));
      console.log('Has closet visualizations:', !!segmentResults.closet_visualizations);
      console.log('Classifications:', {
        shirt: segmentResults.shirt_count,
        pants: segmentResults.pants_count,
        shoes: segmentResults.shoes_count
      });

      // Show results in the same modal
      setShowResults(true);
      setSegmentationResults(segmentResults);
      setProcessing(false);
      
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to process image');
      console.error('Process error:', error);
    } finally {
      setProcessing(false);
      setStatus('');
    }
  };

  const toggleItem = (type) => {
    setSelectedItems(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const handleSave = async () => {
    const itemsToSave = Object.keys(selectedItems).filter(key => selectedItems[key]);
    if (itemsToSave.length === 0) {
      Alert.alert('No items selected', 'Please select at least one item to save');
      return;
    }
    
    setProcessing(true);
    setStatus('Saving to wardrobe...');
    
    try {
      const fileName = `${Date.now()}_mobile.png`;
      
      const saveResponse = await fetch(`${config.API_URL}/firebase/save-results`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: user.uid,
          file_name: fileName,
          segmentation_results: segmentationResults,
          original_image: selectedImage.base64 ? `data:image/jpeg;base64,${selectedImage.base64}` : null
        }),
      });

      if (!saveResponse.ok) {
        const error = await saveResponse.json();
        throw new Error(error.error || 'Failed to save results');
      }

      const saveResult = await saveResponse.json();
      
      // Immediately add to cache for instant display
      const timestamp = Date.now().toString();
      if (selectedItems.shirt && segmentationResults.closet_visualizations?.shirt) {
        await CacheService.addItemToCache(user.uid, 'shirts', {
          id: `${timestamp}_mobile_shirt`,
          image: 'data:image/png;base64,' + segmentationResults.closet_visualizations.shirt,
          type: 'shirt',
          source_image: fileName,
          isClosetViz: true,
          contentBounds: segmentationResults.closet_metadata?.shirt
        });
      }
      
      if (selectedItems.pants && segmentationResults.closet_visualizations?.pants) {
        await CacheService.addItemToCache(user.uid, 'pants', {
          id: `${timestamp}_mobile_pants`,
          image: 'data:image/png;base64,' + segmentationResults.closet_visualizations.pants,
          type: 'pants',
          source_image: fileName,
          isClosetViz: true,
          contentBounds: segmentationResults.closet_metadata?.pants
        });
      }
      
      if (selectedItems.shoes && segmentationResults.closet_visualizations?.shoes) {
        await CacheService.addItemToCache(user.uid, 'shoes', {
          id: `${timestamp}_mobile_shoes`,
          image: 'data:image/png;base64,' + segmentationResults.closet_visualizations.shoes,
          type: 'shoes',
          source_image: fileName,
          isClosetViz: true,
          contentBounds: segmentationResults.closet_metadata?.shoes
        });
      }

      onSuccess();
      onClose();
      // Reset state
      setSelectedImage(null);
      setShowResults(false);
      setSegmentationResults(null);
      setSelectedItems({ shirt: true, pants: true, shoes: true });
      
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to save items');
    } finally {
      setProcessing(false);
      setStatus('');
    }
  };

  const handleReset = () => {
    setShowResults(false);
    setSelectedImage(null);
    setSegmentationResults(null);
    setSelectedItems({ shirt: true, pants: true, shoes: true });
  };

  const handleModalClose = () => {
    onClose();
    // Reset state
    setSelectedImage(null);
    setShowResults(false);
    setSegmentationResults(null);
    setSelectedItems({ shirt: true, pants: true, shoes: true });
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={true}
      onRequestClose={handleModalClose}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalContent}>
          <View style={styles.header}>
            <Text style={styles.title}>{showResults ? 'Select Items to Save' : 'Upload & Segment'}</Text>
            <TouchableOpacity onPress={handleModalClose} style={styles.closeButton}>
              <Text style={styles.closeButtonText}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.scrollView}>
            {showResults && segmentationResults ? (
              <View style={styles.resultsSection}>
                {/* Show original image */}
                <View style={styles.originalImageContainer}>
                  <Image 
                    source={{ uri: selectedImage.uri }} 
                    style={styles.originalImage}
                    resizeMode="contain"
                  />
                  <Text style={styles.originalImageLabel}>Original</Text>
                </View>
                
                <Text style={styles.sectionTitle}>Select items to save:</Text>
                
                <View style={styles.itemsGrid}>
                  {['shirt', 'pants', 'shoes'].map((type) => {
                    const image = segmentationResults.closet_visualizations?.[type];
                    if (!image) return null;
                    
                    return (
                      <TouchableOpacity
                        key={type}
                        style={[styles.resultItem, selectedItems[type] && styles.resultItemSelected]}
                        onPress={() => toggleItem(type)}
                        activeOpacity={0.7}
                      >
                        <Image 
                          source={{ uri: `data:image/png;base64,${image}` }}
                          style={styles.resultImage}
                          resizeMode="contain"
                        />
                        {selectedItems[type] && (
                          <View style={styles.checkmark}>
                            <Text style={styles.checkmarkText}>✓</Text>
                          </View>
                        )}
                        <Text style={styles.itemLabel}>
                          {type === 'shirt' ? '👔' : type === 'pants' ? '👖' : '👟'}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                
                <View style={styles.resultActions}>
                  <TouchableOpacity style={styles.secondaryButton} onPress={handleReset}>
                    <Text style={styles.secondaryButtonText}>Try Again</Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={[styles.primaryButton, processing && styles.disabledButton]} 
                    onPress={handleSave}
                    disabled={processing}
                  >
                    {processing ? (
                      <ActivityIndicator color="#fff" />
                    ) : (
                      <Text style={styles.primaryButtonText}>Save Selected</Text>
                    )}
                  </TouchableOpacity>
                </View>
                {status ? <Text style={styles.statusText}>{status}</Text> : null}
              </View>
            ) : !selectedImage ? (
              <View style={styles.uploadSection}>
                <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
                  <Text style={styles.uploadButtonText}>📷 Choose from Gallery</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.uploadButton} onPress={takePhoto}>
                  <Text style={styles.uploadButtonText}>📸 Take Photo</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View>
                <Image source={{ uri: selectedImage.uri }} style={styles.previewImage} />
                <View style={styles.actionSection}>
                  <TouchableOpacity
                    style={[styles.processButton, processing && styles.processingButton]}
                    onPress={processImage}
                    disabled={processing}
                  >
                    {processing ? (
                      <ActivityIndicator color="#fff" />
                    ) : (
                      <Text style={styles.processButtonText}>Process & Save</Text>
                    )}
                  </TouchableOpacity>
                  {status ? <Text style={styles.statusText}>{status}</Text> : null}
                  <TouchableOpacity style={styles.changeButton} onPress={() => setSelectedImage(null)}>
                    <Text style={styles.changeButtonText}>Change Image</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: screenWidth - 40,
    maxHeight: '80%',
    backgroundColor: '#2a2a2a',
    borderRadius: 20,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#444',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  closeButton: {
    padding: 5,
  },
  closeButtonText: {
    fontSize: 24,
    color: '#999',
  },
  scrollView: {
    maxHeight: 600,
  },
  uploadSection: {
    padding: 20,
    alignItems: 'center',
  },
  uploadButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 10,
    marginVertical: 10,
    width: '100%',
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  previewImage: {
    width: '100%',
    height: 300,
    resizeMode: 'contain',
    backgroundColor: '#1a1a1a',
  },
  actionSection: {
    padding: 20,
  },
  processButton: {
    backgroundColor: '#2196F3',
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 10,
  },
  processingButton: {
    opacity: 0.7,
  },
  processButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  statusText: {
    color: '#999',
    textAlign: 'center',
    marginBottom: 10,
  },
  changeButton: {
    paddingVertical: 10,
    alignItems: 'center',
  },
  changeButtonText: {
    color: '#999',
    fontSize: 14,
  },
  resultsSection: {
    padding: 20,
  },
  originalImageContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  originalImage: {
    width: 150,
    height: 200,
    backgroundColor: '#2a2a2a',
    borderRadius: 8,
  },
  originalImageLabel: {
    color: '#999',
    fontSize: 12,
    marginTop: 4,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 15,
    textAlign: 'center',
  },
  itemsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 30,
  },
  resultItem: {
    width: 100,
    height: 120,
    backgroundColor: '#333',
    borderRadius: 8,
    padding: 4,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  resultItemSelected: {
    borderColor: '#4CAF50',
  },
  resultImage: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 4,
  },
  checkmark: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkmarkText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  itemLabel: {
    textAlign: 'center',
    marginTop: 4,
    fontSize: 16,
  },
  resultActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: '#4CAF50',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: '#444',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  disabledButton: {
    opacity: 0.7,
  },
});

export default UploadSegmentModal;