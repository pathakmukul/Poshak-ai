import React, { useState, useEffect } from 'react';
import './globalStyles.css';
import './Closet.css';
import BackButton from './components/BackButton';
import { getUserClothingItems } from './closetService';
import Loader from './components/Loader';

function Closet({ user, onBack }) {
  const [clothingItems, setClothingItems] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (user && user.uid) {
      loadClothingItems();
    }
  }, [user]);

  const loadClothingItems = async () => {
    setLoading(true);
    try {
      const data = await getUserClothingItems(user.uid);
      
      if (data.success) {
        // Build items from response
        const items = {};
        Object.keys(data).forEach(key => {
          if (key !== 'success' && key !== 'error' && Array.isArray(data[key])) {
            items[key] = data[key];
          }
        });
        setClothingItems(items);
      }
    } catch (error) {
      console.error('Error loading clothing items:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      const data = await getUserClothingItems(user.uid);
      
      if (data.success) {
        // Build items from response
        const items = {};
        Object.keys(data).forEach(key => {
          if (key !== 'success' && key !== 'error' && Array.isArray(data[key])) {
            items[key] = data[key];
          }
        });
        setClothingItems(items);
      }
    } catch (error) {
      console.error('Error refreshing clothing items:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const ClothingSection = ({ title, items, icon }) => {
    if (items.length === 0) return null;
    
    return (
      <div className="clothing-section">
        <div className="section-header">
          <span className="section-icon">{icon}</span>
          <h2>{title}</h2>
          <span className="item-count">{items.length}</span>
        </div>
        
        <div className="clothing-carousel">
          <div className="carousel-track">
            {items.map((item, index) => (
              <div 
                key={item.id || index} 
                className="clothing-item"
                onClick={() => setSelectedItem(item)}
                style={{ cursor: 'pointer' }}
              >
                <div 
                  className="item-image-wrapper"
                  style={{
                    backgroundImage: `url(${item.image})`,
                    backgroundSize: '125%',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat'
                  }}
                >
                </div>
                <div className="item-details">
                  <p className="item-source">from {item.source_image}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const getIcon = (category) => {
    const icons = {
      shirts: '👔',
      pants: '👖',
      shoes: '👟',
      dress: '👗',
      jacket: '🧥',
      hat: '🧢',
      // Add more as needed
    };
    return icons[category] || '👕';
  };

  const formatTitle = (category) => {
    return category.charAt(0).toUpperCase() + category.slice(1);
  };

  return (
    <div className="page-container">
      <div className="closet-container">
        <header className="closet-header">
          <BackButton onClick={onBack} />
          <h1 className="page-title">My Closet</h1>
          <button 
            onClick={handleRefresh} 
            className="refresh-button"
            disabled={isRefreshing}
          >
            {isRefreshing ? '↻ Refreshing...' : '↻ Refresh'}
          </button>
        </header>

      {loading ? (
        <div className="loading-container">
          <Loader message="Loading your closet" />
        </div>
      ) : (
        <div className="closet-content">
          {Object.keys(clothingItems).length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">👔</div>
              <h3>Your closet is empty</h3>
              <p>Upload photos in Wardrobe to start building your digital closet</p>
            </div>
          ) : (
            <>
              {Object.entries(clothingItems).map(([category, items]) => (
                <ClothingSection 
                  key={category}
                  title={formatTitle(category)} 
                  items={items} 
                  icon={getIcon(category)}
                />
              ))}
            </>
          )}
        </div>
      )}
      
      {/* Description Popup */}
      {selectedItem && (
        <div className="description-popup-overlay" onClick={() => setSelectedItem(null)}>
          <div className="description-popup" onClick={(e) => e.stopPropagation()}>
            <button className="close-popup" onClick={() => setSelectedItem(null)}>×</button>
            
            <div className="popup-content">
              <img 
                src={selectedItem.image} 
                alt={selectedItem.type}
                className="popup-image"
              />
              
              <div className="popup-details">
                <h3>{selectedItem.type.charAt(0).toUpperCase() + selectedItem.type.slice(1)}</h3>
                <p className="popup-source">From: {selectedItem.source_image}</p>
                
                {selectedItem.description && (
                  <div className="description-section">
                    <h4>Description</h4>
                    <p className="item-description">{selectedItem.description}</p>
                  </div>
                )}
                
                {!selectedItem.description && (
                  <div className="description-section">
                    <p className="no-description">No description available</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}

export default Closet;