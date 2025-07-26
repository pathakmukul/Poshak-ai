import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  Modal,
  SafeAreaView,
  FlatList,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import * as ImagePicker from 'expo-image-picker';
import { getUserImages, uploadUserImage, deleteUserImage, getMaskData, getSharedGarments } from '../services/storageService';
import UploadSegmentModal from './UploadSegmentModal';
import config from '../config';
import { saveVirtualTryOn } from '../services/virtualClosetService';

function Wardrobe({ navigation, route, user }) {
  const [wardrobeItems, setWardrobeItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedItemMasks, setSelectedItemMasks] = useState(null);
  const [loadingMasks, setLoadingMasks] = useState(false);
  const [showTryOnModal, setShowTryOnModal] = useState(false);
  const [tryOnProcessing, setTryOnProcessing] = useState(false);
  const [tryOnResult, setTryOnResult] = useState(null);
  const [selectedGarment, setSelectedGarment] = useState('');
  const [garments, setGarments] = useState({});
  const [selectedClothingType, setSelectedClothingType] = useState('shirt');
  const [selectedReplacements, setSelectedReplacements] = useState({}); // {shirt: 'SHIRT/file.png', pants: 'PANT/file.png'}
  const [showStoredTick, setShowStoredTick] = useState(false);

  // Load user's wardrobe items
  useEffect(() => {
    loadWardrobeItems();
    loadGarments();
  }, [user]);

  const loadGarments = async () => {
    try {
      const response = await fetch(`${config.API_URL}/garments`);
      const data = await response.json();
      if (data.garments) {
        // Store the categorized garments
        setGarments(data.garments);
        // For now, we'll use flat_list for backward compatibility
        // We'll filter based on clothing type when displaying
      }
    } catch (error) {
      console.error('Error loading garments:', error);
    }
  };

  const loadWardrobeItems = async () => {
    setLoading(true);
    try {
      const result = await getUserImages(user.uid);
      if (result.success) {
        setWardrobeItems(result.images);
      }
    } catch (error) {
      console.error('Error loading wardrobe:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSuccess = () => {
    setShowUploadModal(false);
    loadWardrobeItems(); // Refresh the wardrobe
  };

  const handleDeleteItem = (item) => {
    Alert.alert(
      'Delete Item',
      'Are you sure you want to delete this item?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            const result = await deleteUserImage(user.uid, item.path);
            if (result.success) {
              loadWardrobeItems();
            } else {
              Alert.alert('Error', 'Failed to delete item');
            }
          },
        },
      ]
    );
  };

  const handleItemPress = async (item) => {
    setSelectedItem(item);
    setLoadingMasks(true);
    
    // Load mask data for this item
    const imageName = item.name.split('.')[0];
    const maskData = await getMaskData(user.uid, imageName);
    if (maskData.success && maskData.data) {
      setSelectedItemMasks(maskData.data);
      
      // Set first available clothing type as selected
      if (maskData.data.classifications) {
        const availableTypes = Object.entries(maskData.data.classifications)
          .filter(([type, count]) => count > 0)
          .map(([type]) => type);
        
        if (availableTypes.length > 0) {
          setSelectedClothingType(availableTypes[0]);
        }
      }
    }
    setLoadingMasks(false);
  };

  const handleStoreTryOn = async () => {
    if (!tryOnResult || !selectedItem) return;
    
    try {
      // Store the try-on result
      const tryOnData = {
        originalImage: selectedItem.url,
        resultImage: `data:image/png;base64,${tryOnResult}`,
        garments: selectedReplacements,
        masks: selectedItemMasks,
      };
      
      // Store using service (handles local storage + Firebase sync)
      const result = await saveVirtualTryOn(user.uid, tryOnData);
      
      if (result.success) {
        // Show green tick
        setShowStoredTick(true);
        
        // Navigate to Virtual Closet after 1 second
        setTimeout(() => {
          setShowTryOnModal(false);
          setTryOnResult(null);
          setSelectedReplacements({});
          setSelectedGarment('');
          setShowStoredTick(false);
          navigation.navigate('VirtualCloset');
        }, 1000);
      } else {
        Alert.alert('Error', 'Failed to store try-on result');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to store try-on result');
      console.error('Store error:', error);
    }
  };

  const performGeminiTryOn = async () => {
    if (!selectedItem || !selectedItemMasks) {
      Alert.alert('Error', 'Please wait for mask data to load');
      return;
    }

    if (Object.keys(selectedReplacements).length === 0) {
      Alert.alert('Error', 'Please select at least one garment to try on');
      return;
    }

    setTryOnProcessing(true);
    setTryOnResult(null);

    try {
      const maskImages = {};
      const garmentFiles = {};

      // Get mask data for each selected type
      for (const [type, garmentFile] of Object.entries(selectedReplacements)) {
        if (selectedItemMasks.visualizations && selectedItemMasks.visualizations[type]) {
          // Get mask for this type using prepare_wardrobe_gemini
          const geminiDataResponse = await fetch(`${config.API_URL}/prepare-wardrobe-gemini`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_url: selectedItem.url,
              mask_data: selectedItemMasks,
              clothing_type: type
            }),
          });

          const geminiData = await geminiDataResponse.json();
          
          if (geminiData.success) {
            maskImages[type] = `data:image/png;base64,${geminiData.mask_image}`;
            garmentFiles[type] = garmentFile;
          }
        }
      }

      if (Object.keys(maskImages).length === 0) {
        Alert.alert('Error', 'No valid clothing items found');
        return;
      }

      // Get original image in base64
      const originalResponse = await fetch(`${config.API_URL}/prepare-wardrobe-gemini`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: selectedItem.url,
          mask_data: selectedItemMasks,
          clothing_type: Object.keys(maskImages)[0] // Just to get the original image
        }),
      });

      const originalData = await originalResponse.json();
      if (!originalData.success) {
        Alert.alert('Error', 'Failed to prepare original image');
        return;
      }

      // Always use multi-item endpoint (works for single item too)
      const tryOnResponse = await fetch(`${config.API_URL}/gemini-tryon-multiple`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          person_image: `data:image/png;base64,${originalData.original_image}`,
          mask_images: maskImages,
          garment_files: garmentFiles,
          clothing_types: Object.keys(maskImages)
        }),
      });

      const tryOnData = await tryOnResponse.json();
      
      if (tryOnData.success) {
        setTryOnResult(tryOnData.result_image);
        setShowTryOnModal(true);
      } else {
        Alert.alert('Error', tryOnData.error || 'Try-on failed');
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setTryOnProcessing(false);
    }
  };

  const renderUploadCard = () => (
    <TouchableOpacity
      style={[styles.wardrobeItem, styles.uploadCard]}
      onPress={() => setShowUploadModal(true)}
    >
      <View style={styles.uploadCardContent}>
        <Text style={styles.uploadIcon}>+</Text>
        <Text style={styles.uploadText}>Add New Item</Text>
      </View>
    </TouchableOpacity>
  );

  const renderItem = ({ item, index }) => {
    // Render upload card as first item
    if (index === 0) {
      return renderUploadCard();
    }

    // Adjust index for actual items
    const actualItem = wardrobeItems[index - 1];
    
    return (
      <TouchableOpacity
        style={styles.wardrobeItem}
        onPress={() => handleItemPress(actualItem)}
        onLongPress={() => handleDeleteItem(actualItem)}
      >
        <Image source={{ uri: actualItem.url }} style={styles.itemImage} />
        <View style={styles.itemOverlay}>
          <TouchableOpacity
            style={styles.deleteButton}
            onPress={(e) => {
              e.stopPropagation();
              handleDeleteItem(actualItem);
            }}
          >
            <Text style={styles.deleteButtonText}>🗑️</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.itemName} numberOfLines={1}>
          {actualItem.name}
        </Text>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Loading wardrobe...</Text>
      </View>
    );
  }

  // Add upload card as first item in the data
  const dataWithUploadCard = [{ isUploadCard: true }, ...wardrobeItems];

  return (
    <SafeAreaView style={styles.container}>
      <FlatList
        data={dataWithUploadCard}
        renderItem={renderItem}
        keyExtractor={(item, index) => item.isUploadCard ? 'upload' : item.path}
        numColumns={2}
        contentContainerStyle={styles.gridContainer}
        columnWrapperStyle={styles.row}
      />

      {/* Upload and Segment Modal */}
      <UploadSegmentModal
        visible={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        user={user}
        onSuccess={handleUploadSuccess}
        navigation={navigation}
      />

      {/* Detail modal */}
      <Modal
        visible={!!selectedItem && !showTryOnModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => {
          setSelectedItem(null);
          setSelectedItemMasks(null);
        }}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => {
            setSelectedItem(null);
            setSelectedItemMasks(null);
          }}
        >
          <View style={styles.modalContent}>
            {selectedItem && (
              <>
                <Image
                  source={{ uri: selectedItem.url }}
                  style={styles.modalImage}
                  resizeMode="contain"
                />
                
                
                <TouchableOpacity
                  style={[styles.tryOnButton, (!selectedItemMasks || loadingMasks) && styles.disabledButton]}
                  onPress={() => {
                    if (selectedItemMasks) {
                      setShowTryOnModal(true);
                    }
                  }}
                  disabled={!selectedItemMasks || loadingMasks}
                >
                  <Text style={styles.tryOnButtonText}>🎨 Virtual Try-On</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.closeButton}
                  onPress={() => {
                    setSelectedItem(null);
                    setSelectedItemMasks(null);
                  }}
                >
                  <Text style={styles.closeButtonText}>Close</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Try-On Modal */}
      <Modal
        visible={showTryOnModal && !!selectedItem}
        transparent={true}
        animationType="fade"
        onRequestClose={() => {
          setShowTryOnModal(false);
          setTryOnResult(null);
          setSelectedReplacements({});
          setSelectedGarment('');
        }}
      >
        <View style={styles.modalOverlay}>
        <View style={styles.tryOnModal}>
          <View style={styles.tryOnHeader}>
            <TouchableOpacity
              onPress={() => {
                setShowTryOnModal(false);
                setTryOnResult(null);
                setSelectedReplacements({});
                setSelectedGarment('');
              }}
            >
              <Text style={styles.closeButtonText}>✕</Text>
            </TouchableOpacity>
            <Text style={styles.tryOnTitle}>🎨 Virtual Try-On</Text>
            <View style={{ width: 40 }} />
          </View>

          <ScrollView style={styles.tryOnContent} showsVerticalScrollIndicator={false}>
            {!tryOnResult && (
              <View style={styles.tryOnControls}>
                <Text style={styles.controlLabel}>Select Clothing Type:</Text>
                <ScrollView horizontal style={styles.segmentSelector} showsHorizontalScrollIndicator={false}>
                  {selectedItemMasks && selectedItemMasks.visualizations && 
                    Object.entries(selectedItemMasks.classifications || {}).map(([type, count]) => {
                      if (count > 0 && selectedItemMasks.visualizations[type]) {
                        const visualization = selectedItemMasks.visualizations[type];
                        let imageSrc;
                        
                        // Handle both base64 and URL formats
                        if (visualization.startsWith('data:')) {
                          imageSrc = visualization;
                        } else if (visualization.startsWith('http')) {
                          imageSrc = visualization;
                        } else {
                          imageSrc = `data:image/png;base64,${visualization}`;
                        }
                        
                        return (
                          <TouchableOpacity
                            key={type}
                            style={[
                              styles.segmentImageButton,
                              selectedClothingType === type && styles.selectedSegmentImageButton
                            ]}
                            onPress={() => {
                              setSelectedClothingType(type);
                              // Set the previously selected garment for this type if any
                              setSelectedGarment(selectedReplacements[type] || '');
                            }}
                          >
                            <Image 
                              source={{ uri: imageSrc }}
                              style={styles.segmentImage}
                              resizeMode="contain"
                            />
                            <Text style={styles.segmentImageLabel}>
                              {type.charAt(0).toUpperCase() + type.slice(1)}
                            </Text>
                            {selectedReplacements[type] && (
                              <View style={styles.checkmarkBadge}>
                                <Text style={styles.checkmarkText}>✓</Text>
                              </View>
                            )}
                          </TouchableOpacity>
                        );
                      }
                      return null;
                    })
                  }
                </ScrollView>

                <Text style={styles.controlLabel}>Select {selectedClothingType.charAt(0).toUpperCase() + selectedClothingType.slice(1)} to Try:</Text>
                <ScrollView horizontal style={styles.garmentScroll} showsHorizontalScrollIndicator={false}>
                  <View style={[styles.garmentGrid, {
                    width: garments[selectedClothingType] ? Math.ceil(garments[selectedClothingType].length / 2) * 120 : 0
                  }]}>
                    {garments[selectedClothingType] && garments[selectedClothingType].map((garment, index) => {
                      const column = Math.floor(index / 2);
                      const row = index % 2;
                      const leftPosition = column * 120;
                      
                      return (
                        <TouchableOpacity
                          key={garment}
                          style={[
                            styles.garmentItem,
                            selectedGarment === garment && styles.selectedGarment,
                            {
                              left: leftPosition,
                              top: row * 140,
                            }
                          ]}
                          onPress={() => {
                            if (selectedGarment === garment) {
                              // Clicking the same garment deselects it
                              setSelectedGarment('');
                              setSelectedReplacements(prev => {
                                const newReplacements = {...prev};
                                delete newReplacements[selectedClothingType];
                                return newReplacements;
                              });
                            } else {
                              // Select new garment
                              setSelectedGarment(garment);
                              setSelectedReplacements(prev => ({
                                ...prev,
                                [selectedClothingType]: garment
                              }));
                            }
                          }}
                        >
                          <Image
                            source={{ uri: `${config.API_URL}/static/garments/${garment}` }}
                            style={styles.garmentImage}
                          />
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </ScrollView>

                <TouchableOpacity
                  style={[styles.generateButton, tryOnProcessing && styles.disabledButton]}
                  onPress={performGeminiTryOn}
                  disabled={tryOnProcessing || Object.keys(selectedReplacements).length === 0}
                >
                  <Text style={styles.generateButtonText}>
                    {tryOnProcessing ? 'Processing...' : '✨ Generate Try-On'}
                  </Text>
                </TouchableOpacity>
              </View>
            )}

            {tryOnResult && (
              <View style={styles.tryOnResults}>
                <Text style={styles.mainResultLabel}>Generated Result</Text>
                <View style={{paddingHorizontal: 12}}>
                  <Image 
                    source={{ uri: `data:image/png;base64,${tryOnResult}` }} 
                    style={styles.mainResultImage}
                    resizeMode="contain"
                  />
                </View>
                <Text style={styles.tilesTitle}>Items Used</Text>
                  <ScrollView 
                    horizontal 
                    style={styles.tilesScrollView}
                    showsHorizontalScrollIndicator={false}
                  >
                    <View style={styles.tilesContainer}>
                      {/* Original image */}
                      <View style={styles.tile}>
                        <Text style={styles.tileLabel}>Original</Text>
                        <Image 
                          source={{ uri: selectedItem.url }} 
                          style={styles.tileImage}
                          resizeMode="contain"
                        />
                      </View>
                      
                      {/* All selected garments */}
                      {Object.entries(selectedReplacements).map(([type, garmentFile]) => (
                        <View key={type} style={styles.tile}>
                          <Text style={styles.tileLabel}>{type.charAt(0).toUpperCase() + type.slice(1)}</Text>
                          <Image 
                            source={{ uri: `${config.API_URL}/static/garments/${garmentFile}` }} 
                            style={styles.tileImage}
                            resizeMode="contain"
                          />
                        </View>
                      ))}
                    </View>
                  </ScrollView>

                <View style={styles.resultButtonsContainer}>
                  <TouchableOpacity
                    style={styles.iconButton}
                    onPress={() => performGeminiTryOn()}
                  >
                    <Text style={styles.iconButtonText}>↻</Text>
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={[styles.storeButton, showStoredTick && styles.storedButton]}
                    onPress={() => handleStoreTryOn()}
                    disabled={showStoredTick}
                  >
                    {showStoredTick ? (
                      <Text style={styles.storeButtonText}>✓ Stored</Text>
                    ) : (
                      <Text style={styles.storeButtonText}>Store</Text>
                    )}
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={styles.iconButton}
                    onPress={() => {
                      setTryOnResult(null);
                      setSelectedReplacements({});
                      setSelectedGarment('');
                      setSelectedClothingType('shirt');
                    }}
                  >
                    <Text style={styles.iconButtonText}>←</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </ScrollView>
        </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
  },
  loadingText: {
    color: '#999',
    marginTop: 10,
  },
  gridContainer: {
    padding: 8,
  },
  row: {
    justifyContent: 'space-between',
  },
  wardrobeItem: {
    flex: 0.48,
    aspectRatio: 0.75,
    marginVertical: 8,
    backgroundColor: '#2a2a2a',
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
  },
  uploadCard: {
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#4CAF50',
    borderStyle: 'dashed',
  },
  uploadCardContent: {
    alignItems: 'center',
  },
  uploadIcon: {
    fontSize: 48,
    color: '#4CAF50',
    marginBottom: 8,
  },
  uploadText: {
    color: '#4CAF50',
    fontSize: 14,
    fontWeight: '600',
  },
  itemImage: {
    width: '100%',
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  itemOverlay: {
    position: 'absolute',
    top: 0,
    right: 0,
    padding: 8,
  },
  deleteButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 20,
    width: 30,
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  deleteButtonText: {
    fontSize: 16,
  },
  itemName: {
    color: '#fff',
    fontSize: 12,
    padding: 8,
    textAlign: 'center',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '90%',
    maxHeight: '80%',
    alignItems: 'center',
  },
  modalImage: {
    width: '100%',
    height: 400,
    marginBottom: 20,
  },
  tryOnButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  tryOnButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  closeButton: {
    backgroundColor: '#666',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  maskInfo: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: 16,
    borderRadius: 8,
    marginBottom: 20,
  },
  maskInfoText: {
    color: '#fff',
    fontSize: 14,
    marginVertical: 4,
  },
  disabledButton: {
    opacity: 0.5,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 10,
  },
  tryOnModal: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    width: '95%',
    maxHeight: '85%',
    overflow: 'hidden',
  },
  tryOnHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  tryOnTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  tryOnContent: {
    maxHeight: '100%',
    paddingBottom: 0,
  },
  tryOnControls: {
    marginTop: 5,
    paddingHorizontal: 12,
    paddingBottom: 10,
  },
  controlLabel: {
    color: '#fff',
    fontSize: 14,
    marginBottom: 5,
    fontWeight: '600',
  },
  segmentSelector: {
    marginBottom: 8,
    height: 130,
  },
  segmentImageButton: {
    marginRight: 12,
    alignItems: 'center',
    padding: 6,
    borderRadius: 8,
    backgroundColor: '#2a2a2a',
    borderWidth: 3,
    borderColor: 'transparent',
    width: 110,
  },
  selectedSegmentImageButton: {
    borderColor: '#4CAF50',
  },
  segmentImage: {
    width: 90,
    height: 90,
    borderRadius: 4,
    backgroundColor: '#1a1a1a',
  },
  segmentImageLabel: {
    color: '#fff',
    fontSize: 13,
    marginTop: 4,
    fontWeight: '600',
    textAlign: 'center',
  },
  garmentScroll: {
    marginBottom: 8,
    height: 280,
  },
  garmentGrid: {
    height: 280,
    position: 'relative',
  },
  garmentItem: {
    alignItems: 'center',
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#2a2a2a',
    borderWidth: 2,
    borderColor: 'transparent',
    position: 'absolute',
    width: 110,
    height: 130,
  },
  selectedGarment: {
    borderColor: '#4CAF50',
  },
  garmentImage: {
    width: 100,
    height: 100,
    borderRadius: 4,
    backgroundColor: '#1a1a1a',
  },
  garmentName: {
    color: '#fff',
    fontSize: 12,
    width: 100,
    textAlign: 'center',
  },
  generateButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20,
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  tryOnResults: {
    // Remove flex: 1 to prevent taking full height
  },
  mainResultContainer: {
    marginBottom: 20,
  },
  mainResultLabel: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    marginTop: 12,
    paddingHorizontal: 12,
  },
  mainResultImage: {
    width: '100%',
    aspectRatio: 1, // Square aspect ratio, will adjust based on actual image
    resizeMode: 'contain',
    marginBottom: 4,
  },
  tilesWrapper: {
    marginBottom: 16,
  },
  tilesTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
    paddingHorizontal: 12,
  },
  tilesScrollView: {
    marginBottom: 4,
    paddingHorizontal: 12,
  },
  tilesContainer: {
    flexDirection: 'row',
  },
  tile: {
    width: 120,
    alignItems: 'center',
    marginRight: 12,
  },
  tileLabel: {
    color: '#999',
    fontSize: 12,
    marginBottom: 6,
    textAlign: 'center',
  },
  tileImage: {
    width: 120,
    height: 120,
    borderRadius: 8,
    backgroundColor: '#2a2a2a',
  },
  resultButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
    paddingHorizontal: 12,
    gap: 8,
  },
  iconButton: {
    backgroundColor: '#333',
    width: 50,
    height: 50,
    borderRadius: 25,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  storeButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 8,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 8,
  },
  storedButton: {
    backgroundColor: '#4CAF50',
  },
  storeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  checkmarkBadge: {
    position: 'absolute',
    top: 5,
    right: 5,
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkmarkText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default Wardrobe;