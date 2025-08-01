import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  SafeAreaView,
  ActivityIndicator,
  Modal,
  ScrollView,
  Alert,
  RefreshControl,
  Platform,
} from 'react-native';
import ScreenHeader from '../components/ScreenHeader';
import config from '../config';
import { getVirtualClosetItems, deleteVirtualClosetItem } from '../services/virtualClosetService';

function VirtualCloset({ navigation, user }) {
  const [virtualItems, setVirtualItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (user) {
      loadVirtualCloset(false);
    }
  }, [user]);

  const loadVirtualCloset = async (forceRefresh = false) => {
    if (forceRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    
    try {
      const result = await getVirtualClosetItems(user.uid, forceRefresh);
      if (result.success) {
        setVirtualItems(result.items || []);
      } else {
        Alert.alert('Error', 'Failed to load virtual closet');
      }
    } catch (error) {
      console.error('Error loading virtual closet:', error);
      Alert.alert('Error', 'Failed to load virtual closet');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const deleteItem = async (itemId) => {
    Alert.alert(
      'Delete Item',
      'Are you sure you want to delete this try-on result?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const result = await deleteVirtualClosetItem(user.uid, itemId);
              if (result.success) {
                setVirtualItems(items => items.filter(item => item.id !== itemId));
                setShowDetailModal(false);
              } else {
                Alert.alert('Error', 'Failed to delete item');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete item');
            }
          }
        }
      ]
    );
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity
      style={styles.itemCard}
      onPress={() => {
        setSelectedItem(item);
        setShowDetailModal(true);
      }}
    >
      <Image 
        source={{ uri: item.resultImage }} 
        style={styles.itemImage}
        resizeMode="cover"
      />
      <Text style={styles.itemDate}>
        {new Date(item.createdAt).toLocaleDateString()}
      </Text>
    </TouchableOpacity>
  );

  const renderEmptyList = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyIcon}>VC</Text>
      <Text style={styles.emptyTitle}>No Saved Try-Ons Yet</Text>
      <Text style={styles.emptySubtitle}>
        Your virtual try-on results will appear here
      </Text>
      <TouchableOpacity
        style={styles.goToWardrobeButton}
        onPress={() => navigation.navigate('Wardrobe')}
      >
        <Text style={styles.goToWardrobeText}>Go to Wardrobe</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <ScreenHeader 
        title="Virtual Closet"
        onBack={() => navigation.goBack()}
      />
      
      {loading ? (
        <ActivityIndicator size="large" color="#4CAF50" style={styles.loader} />
      ) : (
        <FlatList
          data={virtualItems}
          renderItem={renderItem}
          keyExtractor={item => item.id}
          numColumns={2}
          contentContainerStyle={styles.listContent}
          ListEmptyComponent={renderEmptyList}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => loadVirtualCloset(true)}
              tintColor="#4CAF50"
            />
          }
        />
      )}

      {/* Detail Modal */}
      {showDetailModal && selectedItem && (
        <Modal
          visible={true}
          animationType="slide"
          transparent={true}
          onRequestClose={() => setShowDetailModal(false)}
        >
          <View style={styles.modalContainer}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <TouchableOpacity onPress={() => setShowDetailModal(false)}>
                  <Text style={styles.closeButton}>✕</Text>
                </TouchableOpacity>
                <Text style={styles.modalTitle}>Try-On Details</Text>
                <TouchableOpacity onPress={() => deleteItem(selectedItem.id)}>
                  <Text style={styles.deleteButton}>Delete</Text>
                </TouchableOpacity>
              </View>

              <ScrollView showsVerticalScrollIndicator={false}>
                <Image 
                  source={{ uri: selectedItem.resultImage }} 
                  style={styles.detailImage}
                  resizeMode="contain"
                />

                <Text style={styles.sectionTitle}>Items Used</Text>
                <ScrollView 
                  horizontal 
                  style={styles.itemsUsedScroll}
                  showsHorizontalScrollIndicator={false}
                >
                  <View style={styles.itemsUsedContainer}>
                    {/* Original */}
                    <View style={styles.usedItem}>
                      <Text style={styles.usedItemLabel}>Original</Text>
                      <Image 
                        source={{ uri: selectedItem.originalImage }} 
                        style={styles.usedItemImage}
                        resizeMode="contain"
                      />
                    </View>

                    {/* Garments */}
                    {Object.entries(selectedItem.garments || {}).map(([type, garmentFile]) => (
                      <View key={type} style={styles.usedItem}>
                        <Text style={styles.usedItemLabel}>
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </Text>
                        <Image 
                          source={{ uri: `${config.API_URL}/static/garments/${garmentFile}` }} 
                          style={styles.usedItemImage}
                          resizeMode="contain"
                        />
                      </View>
                    ))}
                  </View>
                </ScrollView>

                <Text style={styles.dateText}>
                  Created on {new Date(selectedItem.createdAt).toLocaleString()}
                </Text>
              </ScrollView>
            </View>
          </View>
        </Modal>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  loader: {
    flex: 1,
  },
  listContent: {
    padding: 8,
  },
  itemCard: {
    flex: 1,
    margin: 8,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#2a2a2a',
  },
  itemImage: {
    width: '100%',
    aspectRatio: 0.75,
    backgroundColor: '#333',
  },
  itemDate: {
    color: '#999',
    fontSize: 12,
    padding: 8,
    textAlign: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 100,
  },
  emptyIcon: {
    fontSize: 60,
    marginBottom: 16,
  },
  emptyTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  emptySubtitle: {
    color: '#999',
    fontSize: 16,
    marginBottom: 24,
  },
  goToWardrobeButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  goToWardrobeText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  modalContent: {
    flex: 1,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  closeButton: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  modalTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  deleteButton: {
    color: '#ff4444',
    fontSize: 16,
    fontWeight: '600',
  },
  detailImage: {
    width: '100%',
    aspectRatio: 0.75,
    marginVertical: 16,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    paddingHorizontal: 16,
  },
  itemsUsedScroll: {
    marginBottom: 16,
    paddingHorizontal: 16,
  },
  itemsUsedContainer: {
    flexDirection: 'row',
  },
  usedItem: {
    marginRight: 12,
    alignItems: 'center',
  },
  usedItemLabel: {
    color: '#999',
    fontSize: 12,
    marginBottom: 6,
  },
  usedItemImage: {
    width: 100,
    height: 100,
    borderRadius: 8,
    backgroundColor: '#2a2a2a',
  },
  dateText: {
    color: '#999',
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 16,
  },
});

export default VirtualCloset;