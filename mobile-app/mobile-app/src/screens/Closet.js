import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  FlatList,
  RefreshControl,
} from 'react-native';
import { getUserClothingItems } from '../services/closetService';
import SmartCropImage from '../components/SmartCropImage';

function Closet({ user, navigation }) {
  const [activeTab, setActiveTab] = useState('shirts');
  const [clothingItems, setClothingItems] = useState({
    shirts: [],
    pants: [],
    shoes: []
  });
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Load from cache first, then sync
    loadClothingItems(false);
  }, [user]);
  
  // Focus effect to reload when screen is focused
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      // Only refresh from cache on focus
      loadClothingItems(false);
    });
    return unsubscribe;
  }, [navigation]);

  const loadClothingItems = async (forceRefresh = false) => {
    if (forceRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    
    try {
      const result = await getUserClothingItems(user.uid, forceRefresh);
      if (result.success) {
        setClothingItems({
          shirts: result.shirts,
          pants: result.pants,
          shoes: result.shoes
        });
        
        if (result.fromCache) {
          console.log('[Closet] Loaded from cache - instant!');
        } else {
          console.log('[Closet] Loaded from backend');
        }
        
        console.log('[Closet] Items loaded - Shirts:', result.shirts.length, 'Pants:', result.pants.length, 'Shoes:', result.shoes.length);
      }
    } catch (error) {
      console.error('Error loading clothing items:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const tabs = [
    { key: 'shirts', label: '👔 Shirts', count: clothingItems.shirts.length },
    { key: 'pants', label: '👖 Pants', count: clothingItems.pants.length },
    { key: 'shoes', label: '👟 Shoes', count: clothingItems.shoes.length },
  ];

  const renderItem = ({ item }) => {
    // Handle base64 images properly
    let imageSource = null;
    
    if (item.image) {
      // Debug logging
      console.log(`[Closet] Rendering ${item.type} item:`, {
        id: item.id,
        imageLength: item.image.length,
        isUrlBased: item.isUrlBased,
        startsWithDataImage: item.image.startsWith('data:image'),
        first50Chars: item.image.substring(0, 50)
      });
      
      // Check if it's a URL or base64
      if (item.isUrlBased) {
        imageSource = { uri: item.image };
      } else if (item.image.startsWith('data:image')) {
        // For base64 images, check size and warn if too large
        if (item.image.length > 50000) {
          console.warn(`[Closet] Large base64 image detected (${item.image.length} chars) for ${item.type}`);
        }
        imageSource = { uri: item.image };
      } else {
        // If it's just the base64 string, add the data URL prefix
        // Check if we have format info
        const format = item.format || 'png';
        imageSource = { uri: `data:image/${format};base64,${item.image}` };
      }
    }
    
    return (
      <View style={styles.itemContainer}>
        {imageSource ? (
          <Image 
            source={imageSource} 
            style={styles.itemImage}
            resizeMode="contain"
            onError={(e) => {
              console.error(`[Closet] Image load error:`, e.nativeEvent.error);
            }}
            onLoad={() => console.log(`[Closet] Image loaded successfully for ${item.type}`)}
          />
        ) : (
          <View style={[styles.itemImageContainer, styles.placeholderImage]}>
            <Text style={styles.placeholderText}>No Image</Text>
          </View>
        )}
        {item.isClosetViz && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>✨</Text>
          </View>
        )}
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Loading your closet...</Text>
      </View>
    );
  }

  const currentItems = clothingItems[activeTab];

  return (
    <SafeAreaView style={styles.container}>
      {/* Tab Bar */}
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.activeTab]}
            onPress={() => setActiveTab(tab.key)}
          >
            <Text style={[styles.tabText, activeTab === tab.key && styles.activeTabText]}>
              {tab.label}
            </Text>
            <Text style={[styles.tabCount, activeTab === tab.key && styles.activeTabCount]}>
              ({tab.count})
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Content */}
      {currentItems.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No {activeTab} in your closet yet</Text>
          <Text style={styles.emptySubtext}>Upload photos to see items here</Text>
        </View>
      ) : (
        <FlatList
          data={currentItems}
          renderItem={renderItem}
          keyExtractor={(item) => item.id}
          numColumns={2}
          contentContainerStyle={styles.gridContainer}
          columnWrapperStyle={styles.row}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => loadClothingItems(true)}
              tintColor="#4CAF50"
            />
          }
        />
      )}
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
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 12,
    marginHorizontal: 4,
    borderRadius: 8,
  },
  activeTab: {
    backgroundColor: '#3a3a3a',
  },
  tabText: {
    color: '#999',
    fontSize: 16,
    fontWeight: '600',
    marginRight: 4,
  },
  activeTabText: {
    color: '#4CAF50',
  },
  tabCount: {
    color: '#666',
    fontSize: 14,
  },
  activeTabCount: {
    color: '#4CAF50',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  emptySubtext: {
    color: '#999',
    fontSize: 16,
  },
  gridContainer: {
    padding: 8,
  },
  row: {
    justifyContent: 'space-between',
  },
  itemContainer: {
    flex: 0.48,
    aspectRatio: 1,
    marginVertical: 8,
    backgroundColor: '#2a2a2a',
    borderRadius: 8,
    position: 'relative',
    overflow: 'hidden',
  },
  itemImage: {
    width: '100%',
    height: '100%',
    backgroundColor: '#1a1a1a',
  },
  itemImageContainer: {
    width: '100%',
    height: '100%',
  },
  badge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    fontSize: 14,
  },
  placeholderImage: {
    backgroundColor: '#333',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    color: '#666',
    fontSize: 12,
  },
});

export default Closet;