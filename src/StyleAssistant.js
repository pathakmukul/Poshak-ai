import React, { useState, useEffect, useRef } from 'react';
import './globalStyles.css';
import './StyleAssistant.css';
import { getUserClothingItems } from './closetService';
import API_URL from './config';
import VoiceAgentPopup from './VoiceAgentPopup';

function StyleAssistant({ user }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [clothingItems, setClothingItems] = useState({});
  const [wardrobeLoading, setWardrobeLoading] = useState(true);
  const [showVoiceAgent, setShowVoiceAgent] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Try to load existing conversation from sessionStorage
    const savedConversation = sessionStorage.getItem(`style_chat_${user?.uid}`);
    
    if (savedConversation) {
      try {
        const parsed = JSON.parse(savedConversation);
        // Convert timestamp strings back to Date objects
        const messagesWithDates = (parsed.messages || []).map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(messagesWithDates);
        console.log('Loaded conversation from memory');
      } catch (e) {
        console.error('Error loading saved conversation:', e);
        // Fall back to greeting message
        addGreetingMessage();
      }
    } else {
      // Add greeting message for new conversation
      addGreetingMessage();
    }

    // Load user's wardrobe
    if (user && user.uid) {
      loadWardrobe();
    }
  }, [user]);

  const addGreetingMessage = () => {
    // Don't add greeting as a message anymore
    setMessages([]);
  };

  useEffect(() => {
    scrollToBottom();
    
    // Save conversation to sessionStorage whenever messages update
    if (user?.uid && messages.length > 0) {
      try {
        sessionStorage.setItem(`style_chat_${user.uid}`, JSON.stringify({
          messages: messages,
          timestamp: Date.now()
        }));
      } catch (e) {
        console.error('Error saving conversation:', e);
      }
    }
  }, [messages, user]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadWardrobe = async () => {
    setWardrobeLoading(true);
    try {
      const data = await getUserClothingItems(user.uid);
      
      // Check if data has success property (the actual API response structure)
      if (data && (data.shirts || data.pants || data.shoes)) {
        const items = {
          shirts: data.shirts || [],
          pants: data.pants || [],
          shoes: data.shoes || []
        };
        setClothingItems(items);
        console.log('Wardrobe loaded:', {
          shirts: items.shirts.length,
          pants: items.pants.length,
          shoes: items.shoes.length
        });
      } else {
        console.error('No clothing data received:', data);
        setClothingItems({
          shirts: [],
          pants: [],
          shoes: []
        });
      }
    } catch (error) {
      console.error('Error loading wardrobe:', error);
      setClothingItems({
        shirts: [],
        pants: [],
        shoes: []
      });
    } finally {
      setWardrobeLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    // Debug log
    console.log('Sending to new platform-agnostic agent:', {
      user_id: user.uid,
      message: inputMessage,
      wardrobeItemCounts: {
        shirts: clothingItems.shirts?.length || 0,
        pants: clothingItems.pants?.length || 0,
        shoes: clothingItems.shoes?.length || 0
      }
    });

    try {
      // Use the new platform-agnostic agent through Flask
      const response = await fetch(`${API_URL}/agent/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: user.uid,
          message: inputMessage,
          wardrobe_data: clothingItems,  // Send cached wardrobe data
          session_id: sessionStorage.getItem(`style_agent_session_${user.uid}`),
          context: {
            location: null,  // Let user specify location if needed
            preferences: {}
          }
        })
      });

      const data = await response.json();
      
      if (data.success) {
        // Store session ID for continuity
        if (data.session_id) {
          sessionStorage.setItem(`style_agent_session_${user.uid}`, data.session_id);
        }

        // Use matched items from the agent response
        const agentResponse = data.response;
        const matchedItems = data.matched_items || [];

        console.log('Agent Response:', {
          response: agentResponse,
          matchedItemsCount: matchedItems.length,
          matchedItems: matchedItems
        });

        const assistantMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          text: agentResponse,
          items: matchedItems,  // Use matched items from agent with images
          timestamp: new Date()
        };

        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(data.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Fallback to old backend if new agent fails - DISABLED
      /*
      try {
        console.log('Falling back to old style assistant...');
        const fallbackResponse = await fetch(`${API_URL}/style-assistant/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: user.uid,
            query: inputMessage,
            clothing_items: clothingItems,
            conversation_history: []
          })
        });

        const fallbackData = await fallbackResponse.json();
        
        const assistantMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          text: fallbackData.recommendation,
          items: fallbackData.matched_items || [],
          timestamp: new Date()
        };

        setMessages(prev => [...prev, assistantMessage]);
      } catch (fallbackError) {
        const errorMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          text: 'Sorry, I encountered an error. Please make sure the style agent is running.',
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, errorMessage]);
      }
      */
      
      // Just show error message without fallback
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        text: 'Sorry, I encountered an error. Please make sure the Flask backend is running on port 5001.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const ClothingItemCard = ({ item }) => {
    const [isHovered, setIsHovered] = useState(false);
    
    // Check if this is a shopping item
    const isShoppingItem = item.isShoppingItem || item.type === 'shopping';
    
    const handleClick = () => {
      if (isShoppingItem && item.link) {
        window.open(item.link, '_blank');
      }
    };
    
    return (
      <div 
        className={`clothing-item-card ${isShoppingItem ? 'shopping-item' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={handleClick}
        style={{ cursor: isShoppingItem ? 'pointer' : 'default' }}
      >
        <img 
          src={item.image} 
          alt={item.description}
          className="item-thumbnail"
        />
        {isHovered && (
          <div className="item-hover-info">
            <p className="item-description">{item.description}</p>
            {item.brand && <p className="item-brand">{item.brand}</p>}
            {item.price && <p className="item-price">{item.price}</p>}
            {isShoppingItem && <p className="click-to-shop">Click to shop →</p>}
          </div>
        )}
      </div>
    );
  };

  const handleClearChat = async () => {
    if (window.confirm('Clear chat history? Your wardrobe data will remain.')) {
      // Clear sessionStorage
      sessionStorage.removeItem(`style_chat_${user.uid}`);
      const sessionId = sessionStorage.getItem(`style_agent_session_${user.uid}`);
      
      // Clear backend session if exists
      if (sessionId) {
        try {
          const response = await fetch(`${API_URL}/agent/clear-session/${sessionId}`, {
            method: 'POST'
          });
          const data = await response.json();
          console.log('Backend session cleared:', data);
          
          // Remove the session ID from storage
          sessionStorage.removeItem(`style_agent_session_${user.uid}`);
        } catch (error) {
          console.error('Error clearing backend session:', error);
        }
      }
      
      // Reset to empty messages (will show greeting card)
      setMessages([]);
      
      console.log('Chat memory cleared');
    }
  };

  return (
    <div className="page-container">
      <div className="style-assistant-container">
        <div className="chat-container">
        {wardrobeLoading && (
          <div className="wardrobe-loading">
            <span className="loading-text">Loading your wardrobe...</span>
          </div>
        )}
        
        <div className="messages-area">
          {messages.length === 0 && (
            <div className="greeting-card">
              <h2 className="greeting-title">Hello! I'm your personal style assistant</h2>
              <p className="greeting-subtitle">I can help you choose the perfect outfit from your wardrobe.</p>
              
              <div className="greeting-suggestions">
                <p className="suggestions-title">Try asking me questions like:</p>
                <ul className="suggestion-list">
                  <li>"What should I wear for a sunny day?"</li>
                  <li>"Suggest an outfit for a business meeting"</li>
                  <li>"What goes well with my blue jeans?"</li>
                  <li>"Help me pick something casual for the weekend"</li>
                </ul>
              </div>
              
              <p className="greeting-prompt">What outfit are you looking for today?</p>
            </div>
          )}
          
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`message ${message.type}-message`}
            >
              <div className="message-content">
                <div className="message-text">{message.text}</div>
                
                {message.items && message.items.length > 0 && (
                  <div className="recommended-items">
                    {message.items.map((item, index) => (
                      <ClothingItemCard key={index} item={item} />
                    ))}
                  </div>
                )}
                
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message assistant-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <div className="input-area-inner">
            <button 
              onClick={handleClearChat} 
              className="clear-chat-button" 
              title="Clear chat history"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                <line x1="10" y1="11" x2="10" y2="17"></line>
                <line x1="14" y1="11" x2="14" y2="17"></line>
              </svg>
            </button>
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={wardrobeLoading ? "Loading wardrobe..." : "Ask me about outfit recommendations..."}
              className="message-input"
              disabled={loading || wardrobeLoading}
            />
            <button 
              onClick={sendMessage} 
              className="send-button"
              disabled={!inputMessage.trim() || loading || wardrobeLoading}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <button
              onClick={() => setShowVoiceAgent(true)}
              className="voice-button-enhanced"
              title="Use voice assistant"
              disabled={wardrobeLoading}
            >
              <div className="voice-sphere">
                <div className="voice-sphere-inner"></div>
              </div>
              <span className="voice-label">Try Voice!</span>
            </button>
          </div>
        </div>
      </div>
      </div>
      
      {showVoiceAgent && (
        <VoiceAgentPopup 
          user={user}
          wardrobeData={clothingItems}
          onClose={() => setShowVoiceAgent(false)}
        />
      )}
    </div>
  );
}

export default StyleAssistant;