import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
} from 'react-native';

function Home({ navigation, user, onLogout }) {
  const navigateToScreen = (screen) => {
    navigation.navigate(screen);
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.userInfo}>
            <Text style={styles.userIcon}>👤</Text>
            <Text style={styles.userName}>{user.username}</Text>
          </View>
          <TouchableOpacity style={styles.logoutButton} onPress={onLogout}>
            <Text style={styles.logoutText}>Logout</Text>
          </TouchableOpacity>
        </View>

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
            onPress={() => navigateToScreen('OpenAISwitch')}
          >
            <Text style={styles.cardIcon}>🎨</Text>
            <Text style={styles.cardTitle}>OpenAI Switch</Text>
            <Text style={styles.cardDescription}>Generate outfits with AI-powered variations</Text>
            <View style={styles.cardButton}>
              <Text style={styles.cardButtonText}>Try Now →</Text>
            </View>
          </TouchableOpacity>

          <View style={[styles.actionCard, styles.comingSoonCard]}>
            <Text style={styles.cardIcon}>✨</Text>
            <Text style={styles.cardTitle}>Style Assistant</Text>
            <Text style={styles.cardDescription}>Get AI-powered outfit recommendations</Text>
            <View style={styles.comingSoonBadge}>
              <Text style={styles.comingSoonText}>Coming Soon</Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 30,
  },
  userInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  userIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  userName: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  logoutButton: {
    backgroundColor: '#ff4444',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 8,
  },
  logoutText: {
    color: '#fff',
    fontWeight: '600',
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