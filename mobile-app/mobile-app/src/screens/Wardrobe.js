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
import * as ImagePicker from 'expo-image-picker';
import { getUserImages, uploadUserImage, deleteUserImage } from '../services/storageService';
import UploadSegmentModal from './UploadSegmentModal';

function Wardrobe({ navigation, route, user }) {
  const [wardrobeItems, setWardrobeItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);

  // Load user's wardrobe items
  useEffect(() => {
    loadWardrobeItems();
  }, [user]);

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
        onPress={() => setSelectedItem(actualItem)}
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
        visible={!!selectedItem}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setSelectedItem(null)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setSelectedItem(null)}
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
                  style={styles.tryOnButton}
                  onPress={() => {
                    setSelectedItem(null);
                    // Navigate to try-on screen
                    Alert.alert('Coming Soon', 'Virtual try-on feature coming soon!');
                  }}
                >
                  <Text style={styles.tryOnButtonText}>🎨 Virtual Try-On</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.closeButton}
                  onPress={() => setSelectedItem(null)}
                >
                  <Text style={styles.closeButtonText}>Close</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        </TouchableOpacity>
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
});

export default Wardrobe;