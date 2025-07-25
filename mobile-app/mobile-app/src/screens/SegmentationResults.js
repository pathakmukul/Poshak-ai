import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  Image,
  TouchableOpacity,
  Dimensions,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import config from '../config';

const { width: screenWidth } = Dimensions.get('window');

function SegmentationResults({ route, navigation }) {
  const { results, originalImage, originalBase64, userId } = route.params;
  const [saving, setSaving] = useState(false);
  
  // Track which items are selected (all selected by default)
  const [selectedItems, setSelectedItems] = useState({
    shirt: results.closet_visualizations?.shirt ? true : false,
    pants: results.closet_visualizations?.pants ? true : false,
    shoes: results.closet_visualizations?.shoes ? true : false,
  });

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
    
    setSaving(true);
    
    try {
      // Filter results to only include selected items
      const filteredResults = {
        ...results,
        closet_visualizations: Object.fromEntries(
          Object.entries(results.closet_visualizations || {}).filter(
            ([key]) => selectedItems[key] || key === 'all'
          )
        )
      };
      
      const fileName = `${Date.now()}_mobile.png`;
      
      // Call Flask endpoint to save everything
      const saveResponse = await fetch(`${config.API_URL}/firebase/save-results`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          file_name: fileName,
          segmentation_results: filteredResults,
          original_image: originalBase64 ? `data:image/jpeg;base64,${originalBase64}` : null
        }),
      });

      if (!saveResponse.ok) {
        const error = await saveResponse.json();
        throw new Error(error.error || 'Failed to save results');
      }

      const saveResult = await saveResponse.json();
      console.log('Save result:', saveResult);
      
      Alert.alert('Success', 'Items saved to your wardrobe!', [
        { text: 'OK', onPress: () => navigation.navigate('Home') }
      ]);
      
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to save items');
      console.error('Save error:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleRedo = () => {
    navigation.goBack();
  };

  const handleCancel = () => {
    navigation.navigate('Home');
  };

  const renderSegmentTile = (type, label, emoji) => {
    const image = results.closet_visualizations?.[type];
    const isSelected = selectedItems[type];
    
    if (!image) return null;
    
    return (
      <TouchableOpacity 
        style={[styles.tile, isSelected && styles.tileSelected]}
        onPress={() => toggleItem(type)}
        activeOpacity={0.7}
      >
        <Image 
          source={{ uri: `data:image/png;base64,${image}` }}
          style={styles.tileImage}
          resizeMode="contain"
        />
        {isSelected && (
          <View style={styles.checkmark}>
            <Ionicons name="checkmark-circle" size={32} color="#4CAF50" />
          </View>
        )}
        <Text style={styles.tileLabel}>{emoji} {label}</Text>
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Select Items to Save</Text>
        <Text style={styles.subtitle}>Tap to deselect items you don't want</Text>
      </View>

      <View style={styles.tilesContainer}>
        {renderSegmentTile('shirt', 'Shirt', '👔')}
        {renderSegmentTile('pants', 'Pants', '👖')}
        {renderSegmentTile('shoes', 'Shoes', '👟')}
      </View>

      <View style={styles.actionsContainer}>
        <TouchableOpacity style={styles.actionButton} onPress={handleCancel}>
          <Ionicons name="arrow-back" size={24} color="#999" />
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.actionButton} onPress={handleRedo}>
          <Ionicons name="refresh" size={24} color="#999" />
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.actionButton, styles.saveButton]} 
          onPress={handleSave}
        >
          <Ionicons name="checkmark" size={24} color="#fff" />
        </TouchableOpacity>
      </View>

      {saving && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#4CAF50" />
            <Text style={styles.loadingText}>Saving to wardrobe...</Text>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  header: {
    padding: 20,
    alignItems: 'center',
  },
  title: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    color: '#999',
    fontSize: 14,
  },
  tilesContainer: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  tile: {
    width: (screenWidth - 60) / 2,
    height: (screenWidth - 60) / 2,
    margin: 10,
    backgroundColor: '#2a2a2a',
    borderRadius: 12,
    padding: 8,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  tileSelected: {
    borderColor: '#4CAF50',
  },
  tileImage: {
    flex: 1,
    width: '100%',
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
  },
  checkmark: {
    position: 'absolute',
    top: 8,
    right: 8,
  },
  tileLabel: {
    color: '#fff',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 40,
    paddingBottom: 40,
  },
  actionButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#2a2a2a',
    justifyContent: 'center',
    alignItems: 'center',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingBox: {
    backgroundColor: '#2a2a2a',
    padding: 30,
    borderRadius: 12,
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 12,
    fontSize: 16,
  },
});

export default SegmentationResults;