import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Platform,
} from 'react-native';
import ScreenHeader from '../components/ScreenHeader';

function Home({ navigation, user, onLogout }) {
  const navigateToScreen = (screen) => {
    navigation.navigate(screen);
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <ScreenHeader 
          title="KapdaAI"
          showBack={false}
          largeTitle={true}
          rightIcon={
            <TouchableOpacity style={styles.profileButton} onPress={onLogout}>
              <Text style={styles.profileIcon}>👤</Text>
            </TouchableOpacity>
          }
        />

        {/* Welcome Section */}
        <View style={styles.welcomeSection}>
          <Text style={styles.welcomeTitle}>Welcome to KapdaAI</Text>
          <Text style={styles.welcomeSubtitle}>Your AI-powered virtual wardrobe assistant</Text>
        </View>

        {/* Action Cards */}
        <View style={styles.actionCards}>
          <TouchableOpacity 
            style={styles.actionCard} 
            onPress={() => navigateToScreen('Wardrobe')}
          >
            <Text style={styles.cardIcon}>📸</Text>
            <Text style={styles.cardTitle}>Wardrobe</Text>
            <Text style={styles.cardDescription}>Upload photos and try on virtual outfits</Text>
            <View style={styles.cardButton}>
              <Text style={styles.cardButtonText}>Open Wardrobe →</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.actionCard} 
            onPress={() => navigateToScreen('Closet')}
          >
            <Text style={styles.cardIcon}>👗</Text>
            <Text style={styles.cardTitle}>My Closet</Text>
            <Text style={styles.cardDescription}>View your collection of clothing items</Text>
            <View style={styles.cardButton}>
              <Text style={styles.cardButtonText}>Open Closet →</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.actionCard} 
            onPress={() => navigateToScreen('VirtualCloset')}
          >
            <Text style={styles.cardIcon}>VC</Text>
            <Text style={styles.cardTitle}>Virtual Closet</Text>
            <Text style={styles.cardDescription}>View your saved try-on results</Text>
            <View style={styles.cardButton}>
              <Text style={styles.cardButtonText}>View Collection →</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.actionCard} 
            onPress={() => navigateToScreen('StyleAssistant')}
          >
            <Text style={styles.cardIcon}>✨</Text>
            <Text style={styles.cardTitle}>Style Assistant</Text>
            <Text style={styles.cardDescription}>Get AI-powered outfit recommendations</Text>
            <View style={styles.cardButton}>
              <Text style={styles.cardButtonText}>Chat Now →</Text>
            </View>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 40 : 20,
    marginBottom: 30,
  },
  headerTitle: {
    fontSize: 34,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  profileButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileIcon: {
    fontSize: 18,
  },
  welcomeSection: {
    alignItems: 'center',
    marginBottom: 40,
  },
  welcomeTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  welcomeSubtitle: {
    fontSize: 16,
    color: '#999',
  },
  actionCards: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionCard: {
    backgroundColor: '#2a2a2a',
    borderRadius: 12,
    padding: 20,
    width: '48%',
    marginBottom: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  comingSoonCard: {
    opacity: 0.6,
  },
  cardIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
    textAlign: 'center',
  },
  cardDescription: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    marginBottom: 16,
  },
  cardButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  cardButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  comingSoonBadge: {
    backgroundColor: '#666',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  comingSoonText: {
    color: '#fff',
    fontWeight: '600',
  },
});

export default Home;