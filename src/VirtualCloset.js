import React, { useState, useEffect } from 'react';
import './globalStyles.css';
import './VirtualCloset.css';
import { getVirtualClosetItems, deleteVirtualClosetItem, saveVirtualTryOn } from './virtualClosetService';
import { getUserImages, getMaskData } from './storageService';
import API_URL from './config';
import Loader from './components/Loader';

function VirtualCloset({ user }) {
  const [virtualItems, setVirtualItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [showTryOnModal, setShowTryOnModal] = useState(false);
  const [tryOnProcessing, setTryOnProcessing] = useState(false);
  const [tryOnResult, setTryOnResult] = useState(null);
  const [wardrobeImages, setWardrobeImages] = useState([]);
  const [selectedWardrobeItem, setSelectedWardrobeItem] = useState(null);
  const [selectedItemMasks, setSelectedItemMasks] = useState(null);
  const [selectedClothingType, setSelectedClothingType] = useState('shirt');
  const [selectedGarment, setSelectedGarment] = useState('');
  const [selectedReplacements, setSelectedReplacements] = useState({});
  const [loadingMasks, setLoadingMasks] = useState(false);
  const [showWardrobeSelector, setShowWardrobeSelector] = useState(false);
  const [garments, setGarments] = useState({});
  const [showOriginalImage, setShowOriginalImage] = useState(false);

  useEffect(() => {
    if (user) {
      loadVirtualCloset(false);
      loadWardrobeImages();
    }
  }, [user]);

  // Load available garments when component mounts (same as Wardrobe)
  useEffect(() => {
    fetch(`${API_URL}/garments`)
      .then(res => res.json())
      .then(data => {
        if (data.garments) {
          setGarments(data.garments);
        }
      })
      .catch(err => console.error('Error loading garments:', err));
  }, []);

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

  // Load wardrobe images for try-on
  const loadWardrobeImages = async () => {
    try {
      const result = await getUserImages(user.uid);
      if (result.success && result.images.length > 0) {
        setWardrobeImages(result.images);
      }
    } catch (error) {
      console.error('Error loading wardrobe:', error);
    }
  };

  // Load mask data for selected wardrobe item
  const loadMaskData = async (item) => {
    setLoadingMasks(true);
    try {
      console.log('Loading mask data for item:', item);
      // Extract mask path from item name (same as Wardrobe)
      const imageName = item.name.split('.')[0];
      const maskData = await getMaskData(user.uid, imageName);
      console.log('Mask data response:', maskData);
      if (maskData.success) {
        setSelectedItemMasks(maskData.data); // Note: it's .data not .maskData
      } else {
        console.error('Failed to load mask data:', maskData.error);
        setSelectedItemMasks(null);
      }
    } catch (error) {
      console.error('Error loading mask data:', error);
      setSelectedItemMasks(null);
    } finally {
      setLoadingMasks(false);
    }
  };

  // Open try-on modal
  const openTryOnModal = () => {
    if (!wardrobeImages.length) {
      alert('No images in wardrobe. Please add images to wardrobe first.');
      return;
    }
    // Don't show try-on modal yet - need to select an item first
    setShowWardrobeSelector(true);
  };

  // Perform try-on (same as Wardrobe)
  const performGeminiTryOn = async () => {
    if (!selectedWardrobeItem || !selectedItemMasks) {
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
              image_url: selectedWardrobeItem.url,
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
          image_url: selectedWardrobeItem.url,
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
      } else {
        alert(`Error: ${tryOnData.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setTryOnProcessing(false);
    }
  };

  // Store try-on result
  const handleStoreTryOn = async () => {
    if (!tryOnResult || !selectedWardrobeItem) return;
    
    try {
      // Store the try-on result
      const tryOnData = {
        originalImage: selectedWardrobeItem.url,
        resultImage: `data:image/png;base64,${tryOnResult}`,
        garments: selectedReplacements,
        masks: selectedItemMasks,
      };
      
      // Store using service
      const result = await saveVirtualTryOn(user.uid, tryOnData);
      
      if (result.success) {
        alert('Try-on result saved!');
        setShowTryOnModal(false);
        loadVirtualCloset(true); // Refresh the list
      } else {
        alert(`Failed to store try-on result: ${result.error || 'Unknown error'}`);
      }
    } catch (error) {
      alert(`Failed to store try-on result: ${error.message}`);
    }
  };

  // Select wardrobe item and load masks
  const selectWardrobeItem = async (item) => {
    setSelectedWardrobeItem(item);
    await loadMaskData(item);
    setShowWardrobeSelector(false);
    setShowTryOnModal(true);
    setSelectedReplacements({});
    setTryOnResult(null);
  };

  return (
    <div className="page-container">
      <div className="virtual-closet-container">
        <div className="virtual-closet-content">
        {loading ? (
          <Loader message="Loading virtual closet" />
        ) : virtualItems.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">VC</div>
            <h2>No Saved Try-Ons Yet</h2>
            <p>Your virtual try-on results will appear here</p>
          </div>
        ) : (
          <div className="virtual-closet-grid">
            {/* New Try-On Tile */}
            <div 
              className="virtual-closet-item new-tryon-tile"
              onClick={openTryOnModal}
            >
              <div className="new-tryon-content">
                <svg
                  className="new-tryon-icon"
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
                <h3>New Try-On</h3>
                <p>Create a new virtual try-on</p>
              </div>
            </div>
            
            {/* Existing try-on results */}
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

      {/* Wardrobe Selector Modal */}
      {showWardrobeSelector && (
        <div className="item-detail-modal" onClick={() => setShowWardrobeSelector(false)}>
          <div className="modal-content expanded" onClick={(e) => e.stopPropagation()}>
            <button 
              className="close-button"
              onClick={() => setShowWardrobeSelector(false)}
            >
              ✕
            </button>
            
            <h2>Select Image for Try-On</h2>
            
            <div className="wardrobe-selector-grid">
              {wardrobeImages.map((item) => (
                <div 
                  key={item.path}
                  className="wardrobe-selector-item"
                  onClick={() => selectWardrobeItem(item)}
                >
                  <img src={item.url} alt={item.name} />
                  <p>{item.name}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Virtual Try-On Modal (same as Wardrobe) */}
      {showTryOnModal && selectedWardrobeItem && (
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
            
            {loadingMasks ? (
              <div className="loading-masks">
                <Loader message="Loading clothing data" />
              </div>
            ) : !tryOnResult && selectedItemMasks ? (
              <div className="try-on-controls">
                <div className="clothing-type-selector">
                  <h3>Select Clothing Type:</h3>
                  <div className="segment-tiles">
                    {selectedItemMasks.visualizations && 
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
                          onError={(e) => {
                            console.error(`Failed to load garment image: ${garment}`);
                            e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzMzMyIvPjx0ZXh0IHg9IjUwIiB5PSI1MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iIGZpbGw9IiM5OTkiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCI+Tm8gSW1hZ2U8L3RleHQ+PC9zdmc+';
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <button 
                  className="try-on-button primary"
                  onClick={performGeminiTryOn}
                  disabled={tryOnProcessing || Object.keys(selectedReplacements).length === 0}
                >
                  {tryOnProcessing ? 'Processing...' : 'Generate Try-On'}
                </button>
              </div>
            ) : !tryOnResult && !selectedItemMasks ? (
              <div className="no-masks-message">
                <p>No clothing items detected in this image. Please select another image.</p>
                <button 
                  onClick={() => {
                    setShowTryOnModal(false);
                    setShowWardrobeSelector(true);
                  }}
                  className="select-another-button"
                >
                  Select Another Image
                </button>
              </div>
            ) : null}

            {tryOnProcessing && (
              <div className="processing-overlay">
                <Loader message="Generating try-on" />
              </div>
            )}

            {tryOnResult && (
              <div className="try-on-results">
                <div className="results-layout">
                  <div className="result-main">
                    <div className="result-header">
                      <h3>{showOriginalImage ? 'Original Image' : 'Generated Result'}</h3>
                      <label className="switch-toggle">
                        <input 
                          type="checkbox"
                          checked={showOriginalImage}
                          onChange={() => setShowOriginalImage(!showOriginalImage)}
                        />
                        <span className="slider"></span>
                      </label>
                    </div>
                    <img 
                      src={showOriginalImage ? selectedWardrobeItem.url : `data:image/png;base64,${tryOnResult}`} 
                      alt={showOriginalImage ? "Original" : "Try-on result"} 
                      className="result-image"
                    />
                  </div>

                  <div className="items-used">
                    <h3>Items Used</h3>
                    <div className="items-tiles">
                    <div className="item-tile">
                      <h4>Original</h4>
                      <img 
                        src={selectedWardrobeItem.url} 
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
                      onClick={performGeminiTryOn}
                      disabled={tryOnProcessing}
                    >
                      {tryOnProcessing ? 'Regenerating...' : '🔄 Redo Try-On'}
                    </button>
                    
                    <button
                      className="try-on-button store-button"
                      onClick={handleStoreTryOn}
                    >
                      💾 Save Result
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

export default VirtualCloset;