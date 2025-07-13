import React, { useState, useEffect } from 'react';
import './Closet.css';
import { getUserClothingItems } from './closetService';

function Closet({ user, onBack }) {
  const [clothingItems, setClothingItems] = useState({
    shirts: [],
    pants: [],
    shoes: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadClothingItems();
  }, [user]);

  const loadClothingItems = async () => {
    setLoading(true);
    try {
      const data = await getUserClothingItems(user.uid);
      
      if (data.success) {
        setClothingItems({
          shirts: data.shirts || [],
          pants: data.pants || [],
          shoes: data.shoes || []
        });
      }
    } catch (error) {
      console.error('Error loading clothing items:', error);
    } finally {
      setLoading(false);
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
              <div key={item.id || index} className="clothing-item">
                <div className="item-image-wrapper">
                  <img 
                    src={`data:image/png;base64,${item.image}`} 
                    alt={`${item.type} ${index + 1}`}
                    className="item-image"
                  />
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

  return (
    <div className="closet-container">
      <header className="closet-header">
        <button onClick={onBack} className="back-button">
          ← Back
        </button>
        <h1>My Closet</h1>
        <div className="header-spacer"></div>
      </header>

      {loading ? (
        <div className="loading-container">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Loading your closet...</p>
          </div>
        </div>
      ) : (
        <div className="closet-content">
          {clothingItems.shirts.length === 0 && 
           clothingItems.pants.length === 0 && 
           clothingItems.shoes.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">👔</div>
              <h3>Your closet is empty</h3>
              <p>Upload photos in Wardrobe to start building your digital closet</p>
            </div>
          ) : (
            <>
              <ClothingSection 
                title="Shirts" 
                items={clothingItems.shirts} 
                icon="👔" 
              />
              <ClothingSection 
                title="Pants" 
                items={clothingItems.pants} 
                icon="👖" 
              />
              <ClothingSection 
                title="Shoes" 
                items={clothingItems.shoes} 
                icon="👟" 
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default Closet;