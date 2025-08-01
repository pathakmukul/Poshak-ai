import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  ScrollView,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { CacheService } from '../services/cacheService';
import config from '../config';
import ScreenHeader from '../components/ScreenHeader';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function StyleAssistant({ navigation, user, onLogout }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [clothingItems, setClothingItems] = useState({});
  const [wardrobeLoading, setWardrobeLoading] = useState(true);
  const [showIntroCard, setShowIntroCard] = useState(true);
  const flatListRef = useRef(null);

  useEffect(() => {
    // Load saved conversation from AsyncStorage
    loadConversationHistory();

    // Load user's wardrobe from cache
    if (user && user.uid) {
      loadWardrobe();
    }
  }, [user]);

  const loadConversationHistory = async () => {
    if (!user?.uid) return;
    
    try {
      const savedConversation = await AsyncStorage.getItem(`style_chat_${user.uid}`);
      if (savedConversation) {
        const parsed = JSON.parse(savedConversation);
        // Convert timestamp strings back to Date objects
        const messagesWithDates = (parsed.messages || []).map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(messagesWithDates);
        setShowIntroCard(false); // Hide intro if we have history
        console.log('Loaded conversation from memory');
      }
    } catch (error) {
      console.error('Error loading conversation history:', error);
    }
  };

  // Save conversation to AsyncStorage whenever messages update
  useEffect(() => {
    if (user?.uid && messages.length > 0) {
      const saveConversation = async () => {
        try {
          await AsyncStorage.setItem(`style_chat_${user.uid}`, JSON.stringify({
            messages: messages,
            timestamp: Date.now()
          }));
        } catch (error) {
          console.error('Error saving conversation:', error);
        }
      };
      saveConversation();
    }
  }, [messages, user]);

  const loadWardrobe = async () => {
    setWardrobeLoading(true);
    try {
      // First try to load from cache
      const cachedData = await CacheService.getCachedClosetItems(user.uid);
      
      if (cachedData && (cachedData.shirts || cachedData.pants || cachedData.shoes)) {
        // Use the cached data directly, limiting to 20 items per category
        const grouped = {
          shirts: (cachedData.shirts || []).slice(0, 20),
          pants: (cachedData.pants || []).slice(0, 20),
          shoes: (cachedData.shoes || []).slice(0, 20)
        };
        
        setClothingItems(grouped);
        console.log('Wardrobe loaded from cache:', {
          shirts: grouped.shirts.length,
          pants: grouped.pants.length,
          shoes: grouped.shoes.length
        });
      }
    } catch (error) {
      console.error('Error loading wardrobe:', error);
    } finally {
      setWardrobeLoading(false);
    }
  };

  const handleClearChat = async () => {
    // Clear AsyncStorage
    try {
      await AsyncStorage.removeItem(`style_chat_${user.uid}`);
      setMessages([]);
      setShowIntroCard(true);
      console.log('Chat memory cleared');
    } catch (error) {
      console.error('Error clearing chat:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || loading || wardrobeLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      text: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);
    setShowIntroCard(false); // Hide intro card on first message

    // Prepare conversation history (text only, no images)
    const conversationHistory = messages.map(msg => ({
      role: msg.type,
      content: msg.text,
      timestamp: msg.timestamp
    }));

    try {
      const response = await fetch(`${config.API_URL}/style-assistant/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: user.uid,
          query: inputMessage,
          clothing_items: clothingItems,
          conversation_history: conversationHistory
        })
      });

      const data = await response.json();

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        text: data.recommendation,
        items: data.matched_items || [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
      
      // Scroll to bottom
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        text: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };


  const renderMessage = ({ item }) => {
    const isUser = item.type === 'user';
    
    return (
      <View style={[styles.messageContainer, isUser && styles.userMessageContainer]}>
        {isUser ? (
          <View style={[styles.messageBubble, styles.userMessage]}>
            <LinearGradient
              colors={['#5B4CFF', '#7B68EE']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.userMessageGradient}
            >
              <Text style={[styles.messageText, styles.userMessageText]}>
                {item.text}
              </Text>
            </LinearGradient>
            <Text style={styles.messageTime}>
              {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </Text>
          </View>
        ) : (
          <View style={styles.messageBubble}>
            <View style={styles.assistantMessage}>
              <Text style={styles.messageText}>
                {item.text}
              </Text>
              
              {item.items && item.items.length > 0 && (
                <ScrollView 
                  horizontal 
                  showsHorizontalScrollIndicator={false}
                  style={styles.recommendedItems}
                >
                  {item.items.map((clothingItem, index) => (
                    <View key={index} style={styles.clothingItem}>
                      <Image 
                        source={{ uri: clothingItem.image }} 
                        style={styles.itemImage}
                      />
                    </View>
                  ))}
                </ScrollView>
              )}
            </View>
            
            <Text style={styles.messageTime}>
              {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </Text>
          </View>
        )}
      </View>
    );
  };

  return (
    <LinearGradient
      colors={['#1C1C1E', '#000000']}
      style={styles.container}
    >
      <ScreenHeader 
        title="Style Assistant"
        onBack={() => navigation.goBack()}
        rightButton={{
          icon: '🔄',
          onPress: handleClearChat
        }}
      />
      
      <KeyboardAvoidingView 
        style={styles.chatContainer}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 0}
      >
        {wardrobeLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Loading your wardrobe...</Text>
          </View>
        ) : (
          <>
            {showIntroCard && messages.length === 0 && (
              <View style={styles.introCard}>
                <LinearGradient
                  colors={['rgba(91, 76, 255, 0.2)', 'rgba(123, 104, 238, 0.1)']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.introCardGradient}
                >
                  <Text style={styles.introTitle}>Ask me for drip recs! ✨</Text>
                  <Text style={styles.introText}>
                     Try:
                    {'\n'}• "What should I wear today?"
                    {'\n'}• "Show me casual outfits"
                    {'\n'}• "Match my blue jeans"
                  </Text>
                </LinearGradient>
              </View>
            )}
            
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderMessage}
              keyExtractor={item => item.id}
              contentContainerStyle={styles.messagesList}
              onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
            />

            {loading && (
              <View style={styles.typingIndicator}>
                <View style={styles.dot} />
                <View style={[styles.dot, { animationDelay: '0.2s' }]} />
                <View style={[styles.dot, { animationDelay: '0.4s' }]} />
              </View>
            )}

            <View style={styles.inputContainer}>
              <TextInput
                style={styles.textInput}
                value={inputMessage}
                onChangeText={setInputMessage}
                placeholder={wardrobeLoading ? "Loading wardrobe..." : "Ask about outfits..."}
                placeholderTextColor="#999999"
                editable={!loading && !wardrobeLoading}
                onSubmitEditing={sendMessage}
              />
              <TouchableOpacity 
                style={[styles.sendButton, (!inputMessage.trim() || loading || wardrobeLoading) && styles.sendButtonDisabled]}
                onPress={sendMessage}
                disabled={!inputMessage.trim() || loading || wardrobeLoading}
              >
                <LinearGradient
                  colors={(!inputMessage.trim() || loading || wardrobeLoading) 
                    ? ['#4A4A5E', '#5A5A6E'] 
                    : ['#5B4CFF', '#7B68EE']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.sendButtonGradient}
                >
                  <Text style={[styles.sendButtonText, (!inputMessage.trim() || loading || wardrobeLoading) && styles.sendButtonTextDisabled]}>
                    {loading ? '...' : '↑'}
                  </Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </>
        )}
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  chatContainer: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: 'rgba(255, 255, 255, 0.7)',
    marginTop: 12,
    fontSize: 15,
    fontWeight: '500',
  },
  messagesList: {
    paddingHorizontal: 12,
    paddingTop: 16,
    paddingBottom: 16,
  },
  messageContainer: {
    marginVertical: 4,
  },
  userMessageContainer: {
    alignItems: 'flex-end',
  },
  messageBubble: {
    maxWidth: '85%',
    overflow: 'hidden',
  },
  userMessage: {
    alignSelf: 'flex-end',
  },
  userMessageGradient: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 18,
    borderTopRightRadius: 18,
    borderBottomRightRadius: 4,
  },
  assistantMessage: {
    backgroundColor: 'rgba(42, 42, 47, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 18,
    borderTopLeftRadius: 18,
    borderBottomLeftRadius: 4,
  },
  messageText: {
    color: '#FFFFFF',
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '400',
  },
  userMessageText: {
    color: '#FFFFFF',
  },
  messageTime: {
    fontSize: 11,
    color: 'rgba(255, 255, 255, 0.4)',
    marginTop: 4,
    paddingHorizontal: 4,
  },
  recommendedItems: {
    marginTop: 10,
    marginHorizontal: -8,
    paddingVertical: 8,
    paddingHorizontal: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    borderRadius: 12,
  },
  clothingItem: {
    marginHorizontal: 4,
    alignItems: 'center',
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  itemImage: {
    width: 80,
    height: 80,
  },
  itemDescription: {
    fontSize: 11,
    color: 'rgba(255, 255, 255, 0.7)',
    marginTop: 4,
    maxWidth: 80,
    textAlign: 'center',
    paddingHorizontal: 4,
    paddingBottom: 4,
  },
  typingIndicator: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 10,
    gap: 4,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
  inputContainer: {
    backgroundColor: '#2A2A2E',
    flexDirection: 'row',
    alignItems: 'center',
    paddingLeft: 20,
    paddingRight: 8,
    paddingVertical: 8,
    paddingBottom: Platform.OS === 'ios' ? 34 : 8,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  textInput: {
    flex: 1,
    color: '#FFFFFF',
    fontSize: 17,
    paddingVertical: 12,
    backgroundColor: '#2A2A2E',
  },
  sendButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
    overflow: 'hidden',
  },
  sendButtonDisabled: {
    opacity: 0.8,
  },
  sendButtonGradient: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonText: {
    color: '#FFD700',
    fontSize: 20,
    fontWeight: '600',
  },
  sendButtonTextDisabled: {
    color: '#B8B8C8',
  },
  introCard: {
    margin: 16,
    overflow: 'hidden',
    borderRadius: 18,
  },
  introCardGradient: {
    padding: 20,
    alignItems: 'center',
  },
  introTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 10,
  },
  introText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 20,
  },
});