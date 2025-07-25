import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  Alert,
} from 'react-native';

function UploadSegment({ navigation, route, user }) {
  const { imageUri, downloadURL, fileName, userId } = route.params;

  const handleProcess = () => {
    Alert.alert(
      'Coming Soon',
      'Image segmentation and processing feature will be implemented soon!',
      [
        {
          text: 'OK',
          onPress: () => navigation.navigate('Wardrobe')
        }
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Process Image</Text>
        <Text style={styles.subtitle}>
          Image uploaded successfully! Segmentation coming soon.
        </Text>
        
        <TouchableOpacity
          style={styles.processButton}
          onPress={handleProcess}
        >
          <Text style={styles.processButtonText}>Process Image</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.navigate('Wardrobe')}
        >
          <Text style={styles.backButtonText}>Back to Wardrobe</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#999',
    marginBottom: 40,
    textAlign: 'center',
  },
  processButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 8,
    marginBottom: 16,
  },
  processButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  backButton: {
    backgroundColor: '#666',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
  },
});

export default UploadSegment;