import React, { useState, useEffect, useRef } from 'react';
import { getUserClothingItems } from './closetService';
import API_URL from './config';
import './VoiceAgent.css';
import { useConversation } from '@elevenlabs/react';

function VoiceAgent({ user, onBack }) {
  const [isLoading, setIsLoading] = useState(true);
  const [wardrobeData, setWardrobeData] = useState(null);
  const [conversationActive, setConversationActive] = useState(false);
  const [displayedItems, setDisplayedItems] = useState([]);
  const [conversationStatus, setConversationStatus] = useState('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const elevenLabsContainerRef = useRef(null);
  const conversationRef = useRef(null);
  
  // Initialize ElevenLabs conversation
  const conversation = useConversation({
    onConnect: () => {
      console.log('Connected to ElevenLabs');
      setConversationStatus('connected');
    },
    onDisconnect: () => {
      console.log('Disconnected from ElevenLabs');
      setConversationStatus('disconnected');
      setConversationActive(false);
    },
    onError: (error) => {
      console.error('ElevenLabs error:', error);
      setErrorMessage(error.message || 'Connection error');
      setConversationStatus('error');
    },
    onMessage: (message) => {
      console.log('Message from ElevenLabs:', message);
      // The React SDK handles tool calls differently - we need to check the actual message structure
    },
    onDebug: (debugInfo) => {
      console.log('ElevenLabs debug:', debugInfo);
    }
  });

  // Load wardrobe data on mount
  useEffect(() => {
    if (user && user.uid) {
      loadWardrobe();
    }
  }, [user]);

  const loadWardrobe = async () => {
    setIsLoading(true);
    try {
      const data = await getUserClothingItems(user.uid);
      
      if (data && (data.shirts || data.pants || data.shoes)) {
        const items = {
          shirts: data.shirts || [],
          pants: data.pants || [],
          shoes: data.shoes || []
        };
        setWardrobeData(items);
        console.log('Wardrobe loaded for voice agent:', {
          shirts: items.shirts.length,
          pants: items.pants.length,
          shoes: items.shoes.length
        });
      } else {
        console.error('No clothing data received');
        setWardrobeData({
          shirts: [],
          pants: [],
          shoes: []
        });
      }
    } catch (error) {
      console.error('Error loading wardrobe:', error);
      setErrorMessage('Failed to load wardrobe data');
      setWardrobeData({
        shirts: [],
        pants: [],
        shoes: []
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Create wardrobe description for ElevenLabs
  const createWardrobeDescription = () => {
    if (!wardrobeData) return '';

    let description = 'User Wardrobe:\n\n';
    
    // Shirts
    if (wardrobeData.shirts.length > 0) {
      description += `SHIRTS (${wardrobeData.shirts.length} items):\n`;
      wardrobeData.shirts.forEach((item, index) => {
        description += `- ID: ${item.id || `shirt_${index}`} - ${item.description}\n`;
      });
      description += '\n';
    }

    // Pants
    if (wardrobeData.pants.length > 0) {
      description += `PANTS (${wardrobeData.pants.length} items):\n`;
      wardrobeData.pants.forEach((item, index) => {
        description += `- ID: ${item.id || `pants_${index}`} - ${item.description}\n`;
      });
      description += '\n';
    }

    // Shoes
    if (wardrobeData.shoes.length > 0) {
      description += `SHOES (${wardrobeData.shoes.length} items):\n`;
      wardrobeData.shoes.forEach((item, index) => {
        description += `- ID: ${item.id || `shoes_${index}`} - ${item.description}\n`;
      });
    }

    return description;
  };

  // Initialize ElevenLabs conversation
  const startConversation = async () => {
    if (!wardrobeData || conversationActive) return;

    setConversationStatus('initializing');
    setErrorMessage('');

    try {
      // Request microphone permission first
      await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Get ElevenLabs agent ID from environment or config
      const agentId = process.env.REACT_APP_ELEVENLABS_AGENT_ID;
      
      if (!agentId) {
        throw new Error('ElevenLabs Agent ID not configured');
      }

      // Create wardrobe description
      const wardrobeDescription = createWardrobeDescription();

      // Start conversation session with the React SDK
      const sessionId = await conversation.startSession({
        agentId: agentId,
        connectionType: 'webrtc', // or 'websocket'
        // Pass wardrobe data as custom data
        customLlmData: {
          wardrobe_data: wardrobeDescription,
          user_id: user.uid,
          user_name: user.username || user.displayName
        }
      });

      console.log('Conversation started with session ID:', sessionId);
      conversationRef.current = sessionId;
      setConversationActive(true);

    } catch (error) {
      console.error('Failed to start conversation:', error);
      setErrorMessage(error.message || 'Failed to start voice conversation');
      setConversationStatus('error');
      setConversationActive(false);
    }
  };

  // Stop conversation
  const stopConversation = async () => {
    try {
      await conversation.endSession();
      conversationRef.current = null;
    } catch (error) {
      console.error('Error stopping conversation:', error);
    }
    setConversationActive(false);
    setConversationStatus('idle');
    setDisplayedItems([]);
  };

  // Display outfit items handler
  const displayOutfitItems = (params) => {
    const { shirt_id, pant_id, shoe_id } = params;
    const items = [];

    // Find and display shirt
    if (shirt_id && wardrobeData.shirts) {
      const shirt = wardrobeData.shirts.find(item => 
        item.id === shirt_id || item.id === `shirt_${wardrobeData.shirts.indexOf(item)}`
      );
      if (shirt) {
        items.push({ ...shirt, type: 'shirt' });
      }
    }

    // Find and display pants
    if (pant_id && wardrobeData.pants) {
      const pant = wardrobeData.pants.find(item => 
        item.id === pant_id || item.id === `pants_${wardrobeData.pants.indexOf(item)}`
      );
      if (pant) {
        items.push({ ...pant, type: 'pants' });
      }
    }

    // Find and display shoes
    if (shoe_id && wardrobeData.shoes) {
      const shoe = wardrobeData.shoes.find(item => 
        item.id === shoe_id || item.id === `shoes_${wardrobeData.shoes.indexOf(item)}`
      );
      if (shoe) {
        items.push({ ...shoe, type: 'shoes' });
      }
    }

    setDisplayedItems(items);
  };

  // Display shopping items handler
  const displayShoppingItems = (items) => {
    if (!items || !Array.isArray(items)) return;

    const formattedItems = items.map((item, index) => ({
      ...item,
      id: `shopping_${index}`,
      type: 'shopping',
      isShoppingItem: true
    }));

    setDisplayedItems(formattedItems);
  };

  // Clothing item card component
  const ClothingItemCard = ({ item }) => {
    const [isHovered, setIsHovered] = useState(false);
    const isShoppingItem = item.isShoppingItem || item.type === 'shopping';

    const handleClick = () => {
      if (isShoppingItem && item.link) {
        window.open(item.link, '_blank');
      }
    };

    return (
      <div 
        className={`voice-clothing-card ${isShoppingItem ? 'shopping-item' : ''} ${item.type}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={handleClick}
        style={{ cursor: isShoppingItem ? 'pointer' : 'default' }}
      >
        <img 
          src={item.image} 
          alt={item.description}
          className="item-image"
        />
        {isHovered && (
          <div className="item-overlay">
            <p className="item-description">{item.description}</p>
            {item.brand && <p className="item-brand">{item.brand}</p>}
            {item.price && <p className="item-price">{item.price}</p>}
            {isShoppingItem && <p className="click-to-shop">Click to shop →</p>}
          </div>
        )}
        <div className="item-type-badge">{item.type}</div>
      </div>
    );
  };

  return (
    <div className="voice-agent-container">
      <button onClick={onBack} className="back-button">
        ← Back
      </button>
      
      <div className="voice-agent-header">
        <h1>Voice Style Assistant</h1>
        <p className="subtitle">Talk to me about your outfit choices!</p>
      </div>

      {isLoading && (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading your wardrobe...</p>
        </div>
      )}

      {!isLoading && wardrobeData && (
        <>
          <div className="wardrobe-summary">
            <div className="summary-item">
              <span className="count">{wardrobeData.shirts.length}</span>
              <span className="label">Shirts</span>
            </div>
            <div className="summary-item">
              <span className="count">{wardrobeData.pants.length}</span>
              <span className="label">Pants</span>
            </div>
            <div className="summary-item">
              <span className="count">{wardrobeData.shoes.length}</span>
              <span className="label">Shoes</span>
            </div>
          </div>

          {!conversationActive && (
            <div className="start-conversation">
              <button 
                onClick={startConversation}
                className="start-button"
                disabled={conversationStatus === 'initializing'}
              >
                {conversationStatus === 'initializing' ? (
                  <>
                    <div className="button-spinner"></div>
                    Connecting...
                  </>
                ) : (
                  <>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                      <line x1="12" y1="19" x2="12" y2="23"></line>
                      <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                    Start Voice Conversation
                  </>
                )}
              </button>
              
              {errorMessage && (
                <div className="error-message">
                  <p>{errorMessage}</p>
                </div>
              )}
            </div>
          )}

          {conversationActive && (
            <div className="conversation-active">
              <div className="voice-status">
                <div className="voice-indicator">
                  <div className="voice-wave"></div>
                  <p>Voice conversation active</p>
                  <p className="status-text">Status: {conversationStatus}</p>
                  {conversation.isSpeaking && <p className="speaking-indicator">AI is speaking...</p>}
                </div>
              </div>

              <button 
                onClick={stopConversation}
                className="stop-button"
              >
                End Conversation
              </button>
            </div>
          )}

          {displayedItems.length > 0 && (
            <div className="displayed-items">
              <h3>Recommended Items</h3>
              <div className="items-grid">
                {displayedItems.map((item, index) => (
                  <ClothingItemCard key={item.id || index} item={item} />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default VoiceAgent;