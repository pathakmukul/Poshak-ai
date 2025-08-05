import React, { useState, useEffect } from 'react';
import { useConversation } from '@elevenlabs/react';
import './VoiceAgentPopup.css';

function VoiceAgentPopup({ user, wardrobeData, onClose }) {
  const [conversationActive, setConversationActive] = useState(false);
  const [conversationStatus, setConversationStatus] = useState('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [displayedItems, setDisplayedItems] = useState([]);
  
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
      // Handle messages if needed
    },
    onDebug: (debugInfo) => {
      console.log('ElevenLabs debug:', debugInfo);
    }
  });

  // Create wardrobe description for ElevenLabs
  const createWardrobeDescription = () => {
    if (!wardrobeData) return '';

    let description = 'User Wardrobe:\n\n';
    
    // Shirts
    if (wardrobeData.shirts && wardrobeData.shirts.length > 0) {
      description += `SHIRTS (${wardrobeData.shirts.length} items):\n`;
      wardrobeData.shirts.forEach((item, index) => {
        description += `- ID: ${item.id || `shirt_${index}`} - ${item.description}\n`;
      });
      description += '\n';
    }

    // Pants
    if (wardrobeData.pants && wardrobeData.pants.length > 0) {
      description += `PANTS (${wardrobeData.pants.length} items):\n`;
      wardrobeData.pants.forEach((item, index) => {
        description += `- ID: ${item.id || `pants_${index}`} - ${item.description}\n`;
      });
      description += '\n';
    }

    // Shoes
    if (wardrobeData.shoes && wardrobeData.shoes.length > 0) {
      description += `SHOES (${wardrobeData.shoes.length} items):\n`;
      wardrobeData.shoes.forEach((item, index) => {
        description += `- ID: ${item.id || `shoes_${index}`} - ${item.description}\n`;
      });
    }

    return description;
  };

  // Start conversation
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

      // Start conversation session
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
    } catch (error) {
      console.error('Error stopping conversation:', error);
    }
    setConversationActive(false);
    setConversationStatus('idle');
    setDisplayedItems([]);
  };

  // Close popup and clean up
  const handleClose = () => {
    if (conversationActive) {
      stopConversation();
    }
    onClose();
  };

  // Display outfit items handler
  const displayOutfitItems = (params) => {
    const { shirt_id, pant_id, shoe_id } = params;
    const items = [];

    // Find and display items
    if (shirt_id && wardrobeData.shirts) {
      const shirt = wardrobeData.shirts.find(item => 
        item.id === shirt_id || item.id === `shirt_${wardrobeData.shirts.indexOf(item)}`
      );
      if (shirt) items.push({ ...shirt, type: 'shirt' });
    }

    if (pant_id && wardrobeData.pants) {
      const pant = wardrobeData.pants.find(item => 
        item.id === pant_id || item.id === `pants_${wardrobeData.pants.indexOf(item)}`
      );
      if (pant) items.push({ ...pant, type: 'pants' });
    }

    if (shoe_id && wardrobeData.shoes) {
      const shoe = wardrobeData.shoes.find(item => 
        item.id === shoe_id || item.id === `shoes_${wardrobeData.shoes.indexOf(item)}`
      );
      if (shoe) items.push({ ...shoe, type: 'shoes' });
    }

    setDisplayedItems(items);
  };

  return (
    <div className="voice-popup-overlay" onClick={handleClose}>
      <div className="voice-popup-container" onClick={(e) => e.stopPropagation()}>
        <div className="voice-popup-header">
          <h2>Voice Style Assistant</h2>
          <button className="close-button" onClick={handleClose}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div className="voice-popup-content">
          {!conversationActive && (
            <div className="voice-start-section">
              <div className="voice-icon-large">
                <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                  <line x1="12" y1="19" x2="12" y2="23"></line>
                  <line x1="8" y1="23" x2="16" y2="23"></line>
                </svg>
              </div>
              <p className="voice-description">
                Talk to me about your outfit choices! I can help you pick the perfect combination from your wardrobe.
              </p>
              <button 
                onClick={startConversation}
                className="voice-start-button"
                disabled={conversationStatus === 'initializing'}
              >
                {conversationStatus === 'initializing' ? (
                  <>
                    <div className="button-spinner"></div>
                    Connecting...
                  </>
                ) : (
                  'Start Voice Conversation'
                )}
              </button>
              
              {errorMessage && (
                <div className="voice-error-message">
                  <p>{errorMessage}</p>
                </div>
              )}
            </div>
          )}

          {conversationActive && (
            <div className="voice-active-section">
              <div className="voice-status-indicator">
                <div className="voice-pulse"></div>
                <p>Voice conversation active</p>
                <p className="voice-status-text">Status: {conversationStatus}</p>
                {conversation.isSpeaking && <p className="voice-speaking">AI is speaking...</p>}
              </div>

              <button 
                onClick={stopConversation}
                className="voice-stop-button"
              >
                End Conversation
              </button>
            </div>
          )}

          {displayedItems.length > 0 && (
            <div className="voice-items-display">
              <h3>Recommended Items</h3>
              <div className="voice-items-grid">
                {displayedItems.map((item, index) => (
                  <div key={item.id || index} className={`voice-item-card ${item.type}`}>
                    <img src={item.image} alt={item.description} />
                    <div className="voice-item-info">
                      <p>{item.description}</p>
                      <span className="voice-item-type">{item.type}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default VoiceAgentPopup;