import React, { useState, useEffect } from 'react';
import './Wardrobe.css';
import UploadSegmentModal from './UploadSegmentModal';
import { getUserImages, deleteUserImage, getMaskData } from './storageService';

function Wardrobe({ user, onBack }) {
  const [wardrobeItems, setWardrobeItems] = useState([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedItemMasks, setSelectedItemMasks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingMasks, setLoadingMasks] = useState(false);

  // Load user's wardrobe items
  useEffect(() => {
    loadWardrobeItems();
  }, [user]);

  const loadWardrobeItems = async () => {
    setLoading(true);
    try {
      const result = await getUserImages(user.uid);
      if (result.success) {
        setWardrobeItems(result.images);
      }
    } catch (error) {
      console.error('Error loading wardrobe:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSuccess = () => {
    setShowUploadModal(false);
    loadWardrobeItems(); // Refresh the wardrobe
  };

  const handleDeleteItem = async (item) => {
    if (window.confirm('Are you sure you want to delete this item?')) {
      const result = await deleteUserImage(user.uid, item.path);
      if (result.success) {
        loadWardrobeItems();
      } else {
        alert('Failed to delete item');
      }
    }
  };

  return (
    <div className="wardrobe-container">
      <header className="wardrobe-header">
        <button onClick={onBack} className="back-button">
          ← Back
        </button>
        <h1>My Wardrobe</h1>
        <div className="header-spacer"></div>
      </header>

      <div className="wardrobe-content">
        {loading ? (
          <div className="loading-spinner">Loading wardrobe...</div>
        ) : (
          <div className="wardrobe-grid">
            {/* Upload card - always first */}
            <div 
              className="wardrobe-item upload-card"
              onClick={() => setShowUploadModal(true)}
            >
              <div className="upload-card-content">
                <svg
                  className="upload-icon"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                <p>Add New Item</p>
              </div>
            </div>

            {/* Wardrobe items */}
            {wardrobeItems.map((item, index) => (
              <div 
                key={item.path}
                className="wardrobe-item"
                onClick={async () => {
                  setSelectedItem(item);
                  setLoadingMasks(true);
                  // Load mask data for this item
                  const imageName = item.name.split('.')[0];
                  const maskData = await getMaskData(user.uid, imageName);
                  if (maskData.success) {
                    setSelectedItemMasks(maskData.data);
                  }
                  setLoadingMasks(false);
                }}
              >
                <img src={item.url} alt={`Wardrobe item ${index + 1}`} />
                <div className="item-overlay">
                  <button
                    className="delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteItem(item);
                    }}
                  >
                    🗑️
                  </button>
                </div>
                <div className="item-info">
                  <p>{item.name}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload and Segment Modal */}
      {showUploadModal && (
        <UploadSegmentModal
          user={user}
          onClose={() => setShowUploadModal(false)}
          onSuccess={handleUploadSuccess}
        />
      )}

      {/* Item Detail Modal */}
      {selectedItem && (
        <div className="item-detail-modal" onClick={() => {
          setSelectedItem(null);
          setSelectedItemMasks(null);
        }}>
          <div className="modal-content expanded" onClick={(e) => e.stopPropagation()}>
            <button 
              className="close-button"
              onClick={() => {
                setSelectedItem(null);
                setSelectedItemMasks(null);
              }}
            >
              ✕
            </button>
            
            <div className="item-detail-grid">
              <div className="original-image-section">
                <h3>Original Image</h3>
                <img src={selectedItem.url} alt={selectedItem.name} />
              </div>
              
              <div className="segments-section">
                <h3>Detected Clothing</h3>
                {loadingMasks ? (
                  <p>Loading segments...</p>
                ) : selectedItemMasks ? (
                  <div className="segments-info">
                    <div className="segment-count">
                      <p>👔 Shirts: {selectedItemMasks.classifications?.shirt || 0}</p>
                      <p>👖 Pants: {selectedItemMasks.classifications?.pants || 0}</p>
                      <p>👟 Shoes: {selectedItemMasks.classifications?.shoes || 0}</p>
                    </div>
                    <div className="segment-images">
                      {/* Show mask images if available */}
                      {['shirt', 'pants', 'shoes'].map(type => {
                        const imageName = selectedItem.name.split('.')[0];
                        const maskUrl = `${selectedItem.url.replace(selectedItem.name, `${imageName}/mask_${type}.png`)}`;
                        return selectedItemMasks.classifications?.[type] > 0 && (
                          <div key={type} className="segment-preview">
                            <h4>{type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                            <img 
                              src={maskUrl} 
                              alt={`${type} mask`}
                              onError={(e) => {
                                e.target.style.display = 'none';
                              }}
                            />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <p>No segmentation data available</p>
                )}
              </div>
            </div>
            
            <div className="item-actions">
              <button className="try-on-button">
                ✨ Virtual Try-On
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Wardrobe;