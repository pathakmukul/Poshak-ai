import React, { useState, useEffect } from 'react';
import './Wardrobe.css';
import UploadSegmentModal from './UploadSegmentModal';
import { getUserImages, deleteUserImage, getMaskData } from './storageService';
import API_URL from './config';

function Wardrobe({ user, onBack }) {
  const [wardrobeItems, setWardrobeItems] = useState([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedItemMasks, setSelectedItemMasks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingMasks, setLoadingMasks] = useState(false);
  const [showTryOnModal, setShowTryOnModal] = useState(false);
  const [tryOnProcessing, setTryOnProcessing] = useState(false);
  const [tryOnResult, setTryOnResult] = useState(null);
  const [selectedGarment, setSelectedGarment] = useState('');
  const [garments, setGarments] = useState([]);
  const [selectedClothingType, setSelectedClothingType] = useState('shirt');
  const [lastTryOnParams, setLastTryOnParams] = useState(null); // Store last try-on parameters
  
  // TEST: Using Segformer Model - Updated at 2:35 AM
  console.log("🚀 WARDROBE UPDATED: Now using Segformer model instead of MediaPipe+SAM2+CLIP");

  // Load user's wardrobe items
  useEffect(() => {
    loadWardrobeItems();
  }, [user]);

  // Load available garments when component mounts
  useEffect(() => {
    fetch(`${API_URL}/garments`)
      .then(res => res.json())
      .then(data => {
        if (data.garments) {
          setGarments(data.garments);
          if (data.garments.length > 0) {
            setSelectedGarment(data.garments[0]);
          }
        }
      })
      .catch(err => console.error('Error loading garments:', err));
  }, []);

  const loadWardrobeItems = async () => {
    setLoading(true);
    try {
      const result = await getUserImages(user.uid);
      if (result.success) {
        console.log('Loaded wardrobe items:', result.images);
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

  const redoTryOn = async () => {
    if (!lastTryOnParams) return;
    
    setTryOnProcessing(true);
    
    try {
      // Reuse the stored parameters
      const tryOnResponse = await fetch(`${API_URL}/gemini-tryon`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(lastTryOnParams),
      });

      const tryOnData = await tryOnResponse.json();
      
      if (tryOnData.success) {
        setTryOnResult(tryOnData.result_image);
      } else {
        alert(`Error: ${tryOnData.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setTryOnProcessing(false);
    }
  };

  const performGeminiTryOn = async () => {
    if (!selectedItem || !selectedGarment || !selectedItemMasks) {
      alert('Please select an item with mask data and a garment');
      return;
    }

    setTryOnProcessing(true);
    setTryOnResult(null);

    try {
      // Use the new endpoint that works with stored masks - no reprocessing!
      const geminiDataResponse = await fetch(`${API_URL}/prepare-wardrobe-gemini`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: selectedItem.url,  // Firebase Storage URL
          mask_data: selectedItemMasks,  // Already loaded mask data
          clothing_type: selectedClothingType
        }),
      });

      const geminiData = await geminiDataResponse.json();
      
      if (!geminiData.success) {
        throw new Error(geminiData.error || 'Failed to prepare image data');
      }

      // Perform the try-on
      const tryOnResponse = await fetch(`${API_URL}/gemini-tryon`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          person_image: `data:image/png;base64,${geminiData.original_image}`,
          mask_image: `data:image/png;base64,${geminiData.mask_image}`,
          garment_file: selectedGarment,
          clothing_type: selectedClothingType
        }),
      });

      const tryOnData = await tryOnResponse.json();
      
      if (tryOnData.success) {
        setTryOnResult(tryOnData.result_image);
        setShowTryOnModal(true);
        // Store parameters for redo
        setLastTryOnParams({
          person_image: `data:image/png;base64,${geminiData.original_image}`,
          mask_image: `data:image/png;base64,${geminiData.mask_image}`,
          garment_file: selectedGarment,
          clothing_type: selectedClothingType
        });
      } else {
        alert(`Error: ${tryOnData.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setTryOnProcessing(false);
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
                <img 
                  src={item.url} 
                  alt={`Wardrobe item ${index + 1}`} 
                  style={{ backgroundColor: '#f5f5f5' }}
                />
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
                        const count = selectedItemMasks.classifications?.[type] || 0;
                        const visualization = selectedItemMasks.visualizations?.[type];
                        
                        if (count === 0 || !visualization) return null; // Don't show empty categories
                        
                        // Handle both base64 and URL-based visualizations
                        let imageSrc;
                        if (selectedItemMasks.isUrlBased) {
                          // For URL-based saves, visualization contains the storage path
                          const encodedPath = encodeURIComponent(visualization);
                          imageSrc = `https://firebasestorage.googleapis.com/v0/b/kapdaai-local.appspot.com/o/${encodedPath}?alt=media`;
                        } else if (visualization.startsWith('http') || visualization.startsWith('/')) {
                          imageSrc = visualization;  // It's already a URL
                        } else {
                          imageSrc = `data:image/png;base64,${visualization}`; // It's base64
                        }
                        
                        return (
                          <div key={type} className="segment-preview">
                            <h4>{type.charAt(0).toUpperCase() + type.slice(1)} ({count})</h4>
                            <img 
                              src={imageSrc}
                              alt={`${type} mask`}
                              style={{ backgroundColor: '#f8f8f8' }}
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
              <button 
                className="try-on-button"
                onClick={() => {
                  if (selectedItemMasks) {
                    setShowTryOnModal(true);
                  } else {
                    alert('Please wait for mask data to load');
                  }
                }}
                disabled={!selectedItemMasks || loadingMasks}
              >
                ✨ Virtual Try-On
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Virtual Try-On Modal */}
      {showTryOnModal && selectedItem && (
        <div className="item-detail-modal" onClick={() => {
          setShowTryOnModal(false);
          setTryOnResult(null);
        }}>
          <div className="modal-content expanded" onClick={(e) => e.stopPropagation()}>
            <button 
              className="close-button"
              onClick={() => {
                setShowTryOnModal(false);
                setTryOnResult(null);
              }}
            >
              ✕
            </button>
            
            <h2>🎨 Virtual Try-On</h2>
            
            {!tryOnResult && (
              <div className="try-on-controls">
                <div className="control-group">
                  <label>Clothing Type:</label>
                  <select
                    value={selectedClothingType}
                    onChange={(e) => setSelectedClothingType(e.target.value)}
                    disabled={tryOnProcessing}
                  >
                    <option value="shirt">Shirt</option>
                    <option value="pants">Pants</option>
                    <option value="shoes">Shoes</option>
                  </select>
                </div>
                
                <div className="control-group">
                  <label>Select Garment:</label>
                  <select
                    value={selectedGarment}
                    onChange={(e) => setSelectedGarment(e.target.value)}
                    disabled={tryOnProcessing}
                  >
                    <option value="">-- Select a garment --</option>
                    {garments.map(garment => (
                      <option key={garment} value={garment}>
                        {garment}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  className="try-on-button"
                  onClick={performGeminiTryOn}
                  disabled={tryOnProcessing || !selectedGarment}
                  style={{ marginTop: '20px' }}
                >
                  {tryOnProcessing ? 'Processing...' : '✨ Generate Try-On'}
                </button>
              </div>
            )}
            
            {tryOnResult && (
              <div className="try-on-results">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginTop: '20px' }}>
                  <div>
                    <h4>Original</h4>
                    <img 
                      src={selectedItem.url} 
                      alt="Original" 
                      style={{ width: '100%', borderRadius: '8px' }}
                    />
                  </div>
                  <div>
                    <h4>Garment</h4>
                    {selectedGarment && (
                      <img 
                        src={`${API_URL}/static/garments/${selectedGarment}`} 
                        alt="Garment" 
                        style={{ width: '100%', borderRadius: '8px', backgroundColor: '#f8f8f8' }}
                      />
                    )}
                  </div>
                  <div>
                    <h4>Try-On Result</h4>
                    <img 
                      src={`data:image/png;base64,${tryOnResult}`} 
                      alt="Try-on result" 
                      style={{ width: '100%', borderRadius: '8px' }}
                    />
                  </div>
                </div>
                
                <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginTop: '20px' }}>
                  <button
                    className="try-on-button"
                    onClick={redoTryOn}
                    disabled={tryOnProcessing}
                    style={{ 
                      backgroundColor: '#9c27b0',
                      opacity: tryOnProcessing ? 0.7 : 1
                    }}
                  >
                    {tryOnProcessing ? 'Regenerating...' : '🔄 Redo Try-On'}
                  </button>
                  
                  <button
                    className="try-on-button"
                    onClick={() => {
                      setTryOnResult(null);
                      setSelectedClothingType('shirt');
                    }}
                  >
                    Try Another Garment
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Wardrobe;