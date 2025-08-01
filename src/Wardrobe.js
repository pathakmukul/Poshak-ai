import React, { useState, useEffect } from 'react';
import './globalStyles.css';
import './Wardrobe.css';
import UploadSegmentModal from './UploadSegmentModal';
import { getUserImages, deleteUserImage, getMaskData } from './storageService';
import { saveVirtualTryOn, clearLocalStorageIfNeeded } from './virtualClosetService';
import API_URL from './config';
import Loader from './components/Loader';

function Wardrobe({ user }) {
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
  const [garments, setGarments] = useState({});
  const [selectedClothingType, setSelectedClothingType] = useState('shirt');
  const [selectedReplacements, setSelectedReplacements] = useState({}); // {shirt: 'SHIRT/file.png', pants: 'PANT/file.png'}
  const [lastTryOnParams, setLastTryOnParams] = useState(null); // Store last try-on parameters
  const [showStoredTick, setShowStoredTick] = useState(false);
  const [showOriginalImage, setShowOriginalImage] = useState(false); // Toggle between original and generated
  
  // TEST: Using Segformer Model - Updated at 2:35 AM

  // Load user's wardrobe items
  useEffect(() => {
    // Clear localStorage if needed to prevent quota errors
    clearLocalStorageIfNeeded();
    loadWardrobeItems();
  }, [user]);

  // Load available garments when component mounts
  useEffect(() => {
    fetch(`${API_URL}/garments`)
      .then(res => res.json())
      .then(data => {
        if (data.garments) {
          setGarments(data.garments);
          // Don't pre-select any garment
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

  const handleUploadSuccess = async () => {
    setShowUploadModal(false);
    // Clear cache and force refresh
    const { clearWardrobeCache } = await import('./storageService');
    clearWardrobeCache(user.uid);
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
    performGeminiTryOn();
  };

  const handleStoreTryOn = async () => {
    if (!tryOnResult || !selectedItem) return;
    
    try {
      // Store the try-on result
      const tryOnData = {
        originalImage: selectedItem.url,
        resultImage: `data:image/png;base64,${tryOnResult}`,
        garments: selectedReplacements,
        masks: selectedItemMasks,
      };
      
      // Store using service (handles localStorage + Firestore)
      const result = await saveVirtualTryOn(user.uid, tryOnData);
      
      if (result.success) {
        // Show green tick
        setShowStoredTick(true);
        
        // Reset after 2 seconds
        setTimeout(() => {
          setShowStoredTick(false);
        }, 2000);
        
        alert('Try-on result saved to Virtual Closet!');
      } else {
        console.error('Store failed:', result);
        alert(`Failed to store try-on result: ${result.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Store error:', error);
      alert(`Failed to store try-on result: ${error.message}`);
    }
  };

  const performGeminiTryOn = async () => {
    if (!selectedItem || !selectedItemMasks) {
      alert('Please wait for mask data to load');
      return;
    }

    if (Object.keys(selectedReplacements).length === 0) {
      alert('Please select at least one garment to try on');
      return;
    }

    setTryOnProcessing(true);
    setTryOnResult(null);

    try {
      const maskImages = {};
      const garmentFiles = {};

      // Get mask data for each selected type
      for (const [type, garmentFile] of Object.entries(selectedReplacements)) {
        if (selectedItemMasks.visualizations && selectedItemMasks.visualizations[type]) {
          // Get mask for this type using prepare_wardrobe_gemini
          const geminiDataResponse = await fetch(`${API_URL}/prepare-wardrobe-gemini`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_url: selectedItem.url,
              mask_data: selectedItemMasks,
              clothing_type: type
            }),
          });

          const geminiData = await geminiDataResponse.json();
          
          if (geminiData.success) {
            maskImages[type] = `data:image/png;base64,${geminiData.mask_image}`;
            garmentFiles[type] = garmentFile;
          }
        }
      }

      if (Object.keys(maskImages).length === 0) {
        alert('No valid clothing items found');
        return;
      }

      // Get original image in base64
      const originalResponse = await fetch(`${API_URL}/prepare-wardrobe-gemini`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: selectedItem.url,
          mask_data: selectedItemMasks,
          clothing_type: Object.keys(maskImages)[0] // Just to get the original image
        }),
      });

      const originalData = await originalResponse.json();
      if (!originalData.success) {
        alert('Failed to prepare original image');
        return;
      }

      // Always use multi-item endpoint
      const tryOnResponse = await fetch(`${API_URL}/gemini-tryon-multiple`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          person_image: `data:image/png;base64,${originalData.original_image}`,
          mask_images: maskImages,
          garment_files: garmentFiles,
          clothing_types: Object.keys(maskImages)
        }),
      });

      const tryOnData = await tryOnResponse.json();
      
      if (tryOnData.success) {
        setTryOnResult(tryOnData.result_image);
        setShowTryOnModal(true);
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
    <div className="page-container">
      <div className="wardrobe-container">
        <div className="wardrobe-tagline">
          Capture your style – Upload full-body shots and watch your digital wardrobe come to life
        </div>
        <div className="wardrobe-content">
        {loading ? (
          <Loader message="Loading wardrobe" />
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

                <button
                  className="try-on-button"
                  onClick={performGeminiTryOn}
                  disabled={tryOnProcessing || Object.keys(selectedReplacements).length === 0}
                  style={{ marginTop: '20px' }}
                >
                  {tryOnProcessing ? 'Processing...' : '✨ Generate Try-On'}
                </button>
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
                <img src={selectedItem.url} alt={selectedItem.name} />
              </div>
              
              <div className="segments-section">
                <h3>Detected Clothing</h3>
                {loadingMasks ? (
                  <p>Loading segments...</p>
                ) : selectedItemMasks ? (
                  <div className="segments-info">
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
                  // Reset selections when opening try-on modal
                  setSelectedReplacements({});
                  setSelectedGarment('');
                  setSelectedClothingType('shirt');
                  setShowOriginalImage(false); // Reset toggle
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
                <div className="clothing-type-selector">
                  <h3>Select Clothing Type:</h3>
                  <div className="segment-tiles">
                    {selectedItemMasks && selectedItemMasks.visualizations && 
                      Object.entries(selectedItemMasks.classifications || {}).map(([type, count]) => {
                        if (count > 0 && selectedItemMasks.visualizations[type]) {
                          const visualization = selectedItemMasks.visualizations[type];
                          let imageSrc;
                          
                          // Handle both base64 and URL formats
                          if (selectedItemMasks.isUrlBased) {
                            const encodedPath = encodeURIComponent(visualization);
                            imageSrc = `https://firebasestorage.googleapis.com/v0/b/kapdaai-local.appspot.com/o/${encodedPath}?alt=media`;
                          } else if (visualization.startsWith('http')) {
                            imageSrc = visualization;
                          } else {
                            imageSrc = `data:image/png;base64,${visualization}`;
                          }
                          
                          return (
                            <div
                              key={type}
                              className={`segment-tile ${selectedClothingType === type ? 'selected' : ''}`}
                              onClick={() => {
                                setSelectedClothingType(type);
                                setSelectedGarment(selectedReplacements[type] || '');
                              }}
                            >
                              <img src={imageSrc} alt={`${type} mask`} />
                              <p>{type.charAt(0).toUpperCase() + type.slice(1)}</p>
                              {selectedReplacements[type] && (
                                <div className="checkmark">✓</div>
                              )}
                            </div>
                          );
                        }
                        return null;
                      })
                    }
                  </div>
                </div>

                <div className="garment-selector">
                  <h3>Select {selectedClothingType.charAt(0).toUpperCase() + selectedClothingType.slice(1)} to Try:</h3>
                  <div className="garment-grid">
                    {garments[selectedClothingType] && garments[selectedClothingType].map((garment) => (
                      <div
                        key={garment}
                        className={`garment-tile ${selectedGarment === garment ? 'selected' : ''}`}
                        onClick={() => {
                          if (selectedGarment === garment) {
                            // Clicking the same garment deselects it
                            setSelectedGarment('');
                            setSelectedReplacements(prev => {
                              const newReplacements = {...prev};
                              delete newReplacements[selectedClothingType];
                              return newReplacements;
                            });
                          } else {
                            // Select new garment
                            setSelectedGarment(garment);
                            setSelectedReplacements(prev => ({
                              ...prev,
                              [selectedClothingType]: garment
                            }));
                          }
                        }}
                      >
                        <img 
                          src={`${API_URL}/static/garments/${garment}`} 
                          alt={garment}
                        />
                      </div>
                    ))}
                  </div>
                </div>
                
                <button
                  className="try-on-button"
                  onClick={performGeminiTryOn}
                  disabled={tryOnProcessing || Object.keys(selectedReplacements).length === 0}
                  style={{ marginTop: '20px' }}
                >
                  {tryOnProcessing ? 'Processing...' : '✨ Generate Try-On'}
                </button>
              </div>
            )}
            
            {tryOnResult && (
              <div className="try-on-results">
                <div className="results-layout">
                  <div className="result-main">
                    <div className="result-image-container">
                      <label className="switch-toggle image-toggle">
                        <input 
                          type="checkbox"
                          checked={showOriginalImage}
                          onChange={() => setShowOriginalImage(!showOriginalImage)}
                        />
                        <span className="slider"></span>
                      </label>
                      <img 
                        src={showOriginalImage ? selectedItem.url : `data:image/png;base64,${tryOnResult}`} 
                        alt={showOriginalImage ? "Original" : "Try-on result"} 
                        className="result-image"
                      />
                    </div>
                  </div>

                  <div className="items-used">
                    <h3>Items Used</h3>
                    <div className="items-tiles">
                    <div className="item-tile">
                      <h4>Original</h4>
                      <img 
                        src={selectedItem.url} 
                        alt="Original"
                      />
                    </div>
                    
                    {Object.entries(selectedReplacements).map(([type, garmentFile]) => (
                      <div key={type} className="item-tile">
                        <h4>{type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                        <img 
                          src={`${API_URL}/static/garments/${garmentFile}`} 
                          alt={`${type} garment`}
                        />
                      </div>
                    ))}
                  </div>
                  
                  <div className="result-actions">
                    <button
                      className="try-on-button secondary"
                      onClick={redoTryOn}
                      disabled={tryOnProcessing}
                    >
                      {tryOnProcessing ? 'Regenerating...' : '🔄 Redo Try-On'}
                    </button>
                    
                    <button
                      className={`try-on-button store-button ${showStoredTick ? 'stored' : ''}`}
                      onClick={handleStoreTryOn}
                      disabled={showStoredTick}
                    >
                      {showStoredTick ? '✓ Stored' : '💾 Store'}
                    </button>
                    
                    <button
                      className="try-on-button"
                      onClick={() => {
                        setTryOnResult(null);
                        setSelectedReplacements({});
                        setSelectedGarment('');
                        setSelectedClothingType('shirt');
                      }}
                    >
                      Try Another Outfit
                    </button>
                  </div>
                </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </div>
  );
}

export default Wardrobe;