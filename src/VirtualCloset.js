import React, { useState, useEffect } from 'react';
import './VirtualCloset.css';
import { getVirtualClosetItems, deleteVirtualClosetItem } from './virtualClosetService';
import API_URL from './config';

function VirtualCloset({ user, onBack }) {
  const [virtualItems, setVirtualItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
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
        alert('Failed to load virtual closet');
      }
    } catch (error) {
      console.error('Error loading virtual closet:', error);
      alert('Failed to load virtual closet');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const deleteItem = async (itemId) => {
    if (window.confirm('Are you sure you want to delete this try-on result?')) {
      try {
        const result = await deleteVirtualClosetItem(user.uid, itemId);
        if (result.success) {
          setVirtualItems(items => items.filter(item => item.id !== itemId));
          setSelectedItem(null);
        } else {
          alert('Failed to delete item');
        }
      } catch (error) {
        alert('Failed to delete item');
      }
    }
  };

  return (
    <div className="virtual-closet-container">
      <header className="virtual-closet-header">
        <button onClick={onBack} className="back-button">
          ← Back
        </button>
        <h1>Virtual Closet</h1>
        <button onClick={() => loadVirtualCloset(true)} className="refresh-button">
          🔄 Refresh
        </button>
      </header>

      <div className="virtual-closet-content">
        {loading ? (
          <div className="loading-spinner">Loading virtual closet...</div>
        ) : virtualItems.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">VC</div>
            <h2>No Saved Try-Ons Yet</h2>
            <p>Your virtual try-on results will appear here</p>
            <button onClick={onBack} className="go-to-wardrobe-button">
              Go to Wardrobe
            </button>
          </div>
        ) : (
          <div className="virtual-closet-grid">
            {virtualItems.map((item) => (
              <div 
                key={item.id}
                className="virtual-closet-item"
                onClick={() => setSelectedItem(item)}
              >
                <img 
                  src={item.resultImage} 
                  alt="Try-on result"
                />
                <div className="item-date">
                  {new Date(item.createdAt).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedItem && (
        <div className="item-detail-modal" onClick={() => setSelectedItem(null)}>
          <div className="modal-content expanded" onClick={(e) => e.stopPropagation()}>
            <button 
              className="close-button"
              onClick={() => setSelectedItem(null)}
            >
              ✕
            </button>
            
            <h2>Try-On Details</h2>
            
            <div className="detail-content">
              <div className="result-section">
                <img 
                  src={selectedItem.resultImage} 
                  alt="Try-on result"
                  className="detail-result-image"
                />
              </div>

              <div className="items-used-section">
                <h3>Items Used</h3>
                <div className="items-used-grid">
                  <div className="used-item">
                    <h4>Original</h4>
                    <img 
                      src={selectedItem.originalImage} 
                      alt="Original"
                    />
                  </div>
                  
                  {Object.entries(selectedItem.garments || {}).map(([type, garmentFile]) => (
                    <div key={type} className="used-item">
                      <h4>{type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                      <img 
                        src={`${API_URL}/static/garments/${garmentFile}`} 
                        alt={`${type} garment`}
                      />
                    </div>
                  ))}
                </div>

                <p className="created-date">
                  Created on {new Date(selectedItem.createdAt).toLocaleString()}
                </p>

                <button 
                  className="delete-button-modal"
                  onClick={() => deleteItem(selectedItem.id)}
                >
                  Delete Try-On
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default VirtualCloset;