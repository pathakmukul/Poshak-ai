import React, { useState, useEffect, useRef } from 'react';
import './LuxuryClosetHome.css';
import { getUserClothingItems } from '../closetService';
import { getUserImages } from '../storageService';
import { saveVirtualTryOn } from '../virtualClosetService';
import { handleTryOn } from './LuxuryTryOn';

function LuxuryClosetHome({ user }) {
  const [clothingItems, setClothingItems] = useState({
    shirts: [],
    pants: [],
    shoes: []
  });
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState(null);
  const [displayedItems, setDisplayedItems] = useState({
    shirts: null,
    pants: null,
    shoes: null
  });
  const [imageBounds, setImageBounds] = useState({});
  const [imagesLoaded, setImagesLoaded] = useState({});
  const [tryOnLoading, setTryOnLoading] = useState(false);
  const [tryOnResult, setTryOnResult] = useState(null);
  const [wardrobeItems, setWardrobeItems] = useState([]); // Person images from wardrobe
  const [showStoredTick, setShowStoredTick] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [showTryOnOverlay, setShowTryOnOverlay] = useState(true); // Toggle for try-on overlay
  const canvasRef = useRef(null);
  const stackRef = useRef(null);

  useEffect(() => {
    loadClothingItems();
    loadWardrobeItems();
  }, [user.uid]);

  const loadClothingItems = async () => {
    try {
      setLoading(true);
      
      // Fetch from API (which will use cache if available)
      const response = await getUserClothingItems(user.uid);
      
      if (response.success) {
        setClothingItems({
          shirts: response.shirts || [],
          pants: response.pants || [],
          shoes: response.shoes || []
        });
      } else {
        console.error('Failed to load clothing items');
        setClothingItems({
          shirts: [],
          pants: [],
          shoes: []
        });
      }
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading clothing items:', error);
      setLoading(false);
    }
  };

  const loadWardrobeItems = async () => {
    try {
      const result = await getUserImages(user.uid);
      if (result.success) {
        console.log('Loaded wardrobe items (person photos):', result.images);
        setWardrobeItems(result.images || []);
      } else {
        console.log('Failed to load wardrobe items:', result);
        setWardrobeItems([]);
      }
    } catch (error) {
      console.error('Error loading wardrobe items:', error);
    }
  };

  const handleItemClick = (item, category) => {
    console.log('Item clicked:', item, category);
    // Add category to item if not present
    const itemWithCategory = {
      ...item,
      category: item.category || category
    };
    
    // Clear the bounds for this category so it recalculates
    setImageBounds(prev => {
      const newBounds = { ...prev };
      delete newBounds[category];
      return newBounds;
    });
    
    setImagesLoaded(prev => {
      const newLoaded = { ...prev };
      delete newLoaded[category];
      return newLoaded;
    });
    
    // Update only the specific category
    setDisplayedItems(prev => {
      const updated = {
        ...prev,
        [category]: itemWithCategory
      };
      console.log('Updated displayed items:', updated);
      return updated;
    });
  };

  // Calculate actual content bounds of an image - PROPERLY
  const calculateImageBounds = (img, type) => {
    if (!canvasRef.current) {
      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Get displayed size vs natural size ratio
    const displayRatio = img.offsetHeight / img.naturalHeight;
    
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    
    let minY = canvas.height;
    let maxY = 0;
    let foundContent = false;
    
    // Scan more efficiently - check every 4th pixel for speed
    for (let y = 0; y < canvas.height; y += 2) {
      let rowHasContent = false;
      for (let x = 0; x < canvas.width; x += 4) {
        const idx = (y * canvas.width + x) * 4;
        const alpha = pixels[idx + 3];
        
        if (alpha > 30) { // Higher threshold for actual content
          rowHasContent = true;
          foundContent = true;
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
          break;
        }
      }
    }
    
    if (!foundContent) {
      console.warn(`No content found in ${type} image`);
      return null;
    }
    
    const bounds = {
      naturalTop: minY,
      naturalBottom: maxY,
      naturalHeight: maxY - minY,
      // Convert to display pixels
      displayTop: minY * displayRatio,
      displayBottom: maxY * displayRatio,
      displayHeight: (maxY - minY) * displayRatio,
      // Store the actual element position
      elementTop: img.offsetTop,
      elementHeight: img.offsetHeight
    };
    
    // Bounds calculated successfully
    
    setImageBounds(prev => ({
      ...prev,
      [type]: bounds
    }));
    
    setImagesLoaded(prev => ({
      ...prev,
      [type]: true
    }));
    
    return bounds;
  };

  // Recalculate all positions when any image loads
  useEffect(() => {
    if (Object.keys(imagesLoaded).length > 0) {
      // Small delay to ensure DOM is ready
      setTimeout(() => {
        forceRecalculatePositions();
      }, 100);
    }
  }, [imagesLoaded, displayedItems]);

  // Force recalculation of all positions
  const forceRecalculatePositions = () => {
    const items = ['shirts', 'pants', 'shoes'];
    items.forEach(type => {
      const element = stackRef.current?.querySelector(`.display-${type} img`);
      if (element && element.complete) {
        calculateImageBounds(element, type);
      }
    });
  };

  // Get dynamic position based on actual content
  const getItemPosition = (type) => {
    if (type === 'shirts') {
      return {}; // Let CSS handle it
    } 
    
    if (type === 'pants') {
      const shirtBounds = imageBounds.shirts;
      if (shirtBounds && imagesLoaded.shirts) {
        // Position pants where shirt content ends
        const shirtContentEnd = shirtBounds.displayTop + shirtBounds.displayHeight;
        return { top: `${shirtContentEnd}px` };
      }
      return {};
    }
    
    if (type === 'shoes') {
      const pantsBounds = imageBounds.pants;
      const shirtBounds = imageBounds.shirts;
      
      if (pantsBounds && imagesLoaded.pants) {
        // First calculate where pants are positioned
        let pantsPosition = 0;
        if (shirtBounds && imagesLoaded.shirts) {
          pantsPosition = shirtBounds.displayTop + shirtBounds.displayHeight;
        }
        // Then position shoes where pants content ends
        const pantsContentEnd = pantsPosition + pantsBounds.displayTop + pantsBounds.displayHeight;
        return { top: `${pantsContentEnd}px` };
      }
      return {};
    }
    
    return {};
  };

  // Handle Virtual Try On
  const handleTryOnClick = async () => {
    await handleTryOn({
      displayedItems,
      wardrobeItems,
      user,
      setTryOnLoading,
      setTryOnResult,
      setShowTryOnOverlay
    });
  };

  // Handle storing try-on result to Virtual Closet
  const handleStoreTryOn = async () => {
    if (!tryOnResult || !wardrobeItems.length) return;
    
    try {
      // Get the person image used
      const personItem = wardrobeItems.find(item => {
        const imageName = item.name?.split('.')[0];
        if (imageName) {
          // Check if this was the item we used (has mask data)
          return true; // We used the first item with mask data
        }
        return false;
      });
      
      if (!personItem) {
        alert('Could not find original person image');
        return;
      }
      
      // Prepare try-on data
      const tryOnData = {
        originalImage: personItem.url || personItem.imageUrl || personItem.image,
        resultImage: tryOnResult.startsWith('data:') ? tryOnResult : `data:image/png;base64,${tryOnResult}`,
        garments: {
          shirt: displayedItems.shirts ? (displayedItems.shirts.image || displayedItems.shirts.url || displayedItems.shirts.imageUrl) : null,
          pants: displayedItems.pants ? (displayedItems.pants.image || displayedItems.pants.url || displayedItems.pants.imageUrl) : null,
          shoes: displayedItems.shoes ? (displayedItems.shoes.image || displayedItems.shoes.url || displayedItems.shoes.imageUrl) : null
        },
        masks: null, // Luxury closet doesn't track individual masks
      };
      
      // Remove null garments
      Object.keys(tryOnData.garments).forEach(key => {
        if (!tryOnData.garments[key]) {
          delete tryOnData.garments[key];
        }
      });
      
      // Store using service (handles localStorage + Firestore)
      const result = await saveVirtualTryOn(user.uid, tryOnData);
      
      if (result.success) {
        // Show green tick
        setShowStoredTick(true);
        
        // Show toast message
        setToastMessage('Saved to Virtual Closet');
        
        // Reset after 2 seconds
        setTimeout(() => {
          setShowStoredTick(false);
        }, 2000);
        
        // Hide toast after 3 seconds
        setTimeout(() => {
          setToastMessage('');
        }, 3000);
      } else {
        console.error('Store failed:', result);
        setToastMessage('Failed to save');
        setTimeout(() => {
          setToastMessage('');
        }, 3000);
      }
    } catch (error) {
      console.error('Store error:', error);
      setToastMessage('Failed to save');
      setTimeout(() => {
        setToastMessage('');
      }, 3000);
    }
  };



  if (loading) {
    return (
      <div className="luxury-closet-loading">
        <div className="loading-spinner"></div>
        <p>Opening your luxury closet...</p>
      </div>
    );
  }

  return (
    <div className="luxury-closet-container">
      {/* Ambient lighting overlay */}
      <div className="ambient-light-overlay"></div>
      
      {/* Main closet structure */}
      <div className="closet-room">
        {/* Ceiling with luxury lighting */}
        <div className="closet-ceiling">
          <div className="ceiling-light"></div>
          <div className="ceiling-light"></div>
          <div className="ceiling-light"></div>
        </div>

        {/* Left Wall - Shirts Section */}
        <div className="closet-wall wall-left">
          <div className="clothing-rail">
            <div className="rail-bar"></div>
            <div className="hangers-container">
              {clothingItems.shirts.length === 0 ? (
                <div className="empty-section">
                  <p>No shirts in your collection yet</p>
                </div>
              ) : (
                clothingItems.shirts.map((item, index) => (
                  <div 
                    key={item.id || index} 
                    className="hanger-item"
                    style={{ '--delay': `${index * 0.1}s` }}
                    onClick={() => handleItemClick(item, 'shirts')}
                  >
                    <div className="hanger">
                      <div className="hanger-hook"></div>
                      <div className="hanger-body"></div>
                    </div>
                    <div className="clothing-item shirt-item">
                      <img 
                        src={item.image || item.url || item.imageUrl} 
                        alt={item.category || item.type || 'Shirt'}
                        loading="lazy"
                        onError={(e) => {
                          console.error('Failed to load image:', item.image || item.url || item.imageUrl);
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Glass Display Case */}
        <div className="closet-center-space">
          <div className="glass-display-case">
            <div className={`glass-case-frame ${tryOnLoading ? 'loading' : ''}`}>
              <div className="glass-top"></div>
              <div className="glass-panels">
                {/* Spotlights */}
                <div className={`spotlight spotlight-top ${(displayedItems.shirts || displayedItems.pants) ? 'active' : ''}`}></div>
                <div className={`spotlight spotlight-bottom ${displayedItems.shoes ? 'active' : ''}`}></div>
                
                {/* Video Display when nothing is selected */}
                {!displayedItems.shirts && !displayedItems.pants && !displayedItems.shoes && !tryOnResult && (
                  <div className="video-container">
                    <video 
                      className="display-video"
                      autoPlay 
                      loop 
                      muted 
                      playsInline
                    >
                      <source src="/videos/rampwalk.mp4" type="video/mp4" />
                      Your browser does not support the video tag.
                    </video>
                  </div>
                )}

                {/* Stacked Display Items with Dynamic Positioning */}
                <div className="outfit-stack" ref={stackRef}>
                  {displayedItems.shirts && (
                    <div 
                      className="outfit-item display-shirt" 
                      style={getItemPosition('shirts')}
                      key={`shirt-${displayedItems.shirts.id || displayedItems.shirts.imageUrl}`}
                    >
                      <img 
                        src={displayedItems.shirts.image || displayedItems.shirts.url || displayedItems.shirts.imageUrl} 
                        alt="Shirt"
                        onLoad={(e) => {
                          calculateImageBounds(e.target, 'shirts');
                        }}
                      />
                    </div>
                  )}
                  {displayedItems.pants && (
                    <div 
                      className="outfit-item display-pants" 
                      style={getItemPosition('pants')}
                      key={`pants-${displayedItems.pants.id || displayedItems.pants.imageUrl}`}
                    >
                      <img 
                        src={displayedItems.pants.image || displayedItems.pants.url || displayedItems.pants.imageUrl} 
                        alt="Pants"
                        onLoad={(e) => {
                          calculateImageBounds(e.target, 'pants');
                        }}
                      />
                    </div>
                  )}
                  {displayedItems.shoes && (
                    <div 
                      className="outfit-item display-shoes" 
                      style={getItemPosition('shoes')}
                      key={`shoes-${displayedItems.shoes.id || displayedItems.shoes.imageUrl}`}
                    >
                      <img 
                        src={displayedItems.shoes.image || displayedItems.shoes.url || displayedItems.shoes.imageUrl} 
                        alt="Shoes"
                        onLoad={(e) => {
                          calculateImageBounds(e.target, 'shoes');
                        }}
                      />
                    </div>
                  )}
                </div>
                
                {/* Corner Buttons - show different buttons based on try-on state */}
                {!tryOnResult ? (
                  <>
                    <button 
                      className="corner-button left-corner"
                      onClick={() => {
                        setDisplayedItems({
                          shirts: null,
                          pants: null,
                          shoes: null
                        });
                        setTryOnResult(null);
                      }}
                      title="Clear all items"
                    >
                      ×
                    </button>
                    <button 
                      className={`corner-button right-corner ${tryOnLoading ? 'processing' : ''}`}
                      onClick={handleTryOnClick}
                      disabled={tryOnLoading || !displayedItems.shirts || !displayedItems.pants || !displayedItems.shoes}
                      title={tryOnLoading ? "Processing..." : "Virtual try on"}
                    >
                      {tryOnLoading ? '⏳' : '✨'}
                    </button>
                  </>
                ) : null}
                
                {/* Try On Result Display */}
                {tryOnResult && showTryOnOverlay && (
                  <div className="tryon-result-overlay">
                    <button className="close-tryon" onClick={() => {
                      setTryOnResult(null);
                      setShowTryOnOverlay(true); // Reset toggle when closing
                    }}>×</button>
                    <img 
                      src={tryOnResult.startsWith('data:') ? tryOnResult : `data:image/png;base64,${tryOnResult}`} 
                      alt="Try on result" 
                      className="tryon-result-image" 
                    />
                  </div>
                )}
                
                {/* Toggle and Save buttons when try-on result exists */}
                {tryOnResult && (
                  <>
                    <button 
                      className="corner-button left-corner toggle-button"
                      onClick={() => setShowTryOnOverlay(!showTryOnOverlay)}
                      title={showTryOnOverlay ? "Hide try-on result" : "Show try-on result"}
                    >
                      {showTryOnOverlay ? '◉' : '○'}
                    </button>
                    <button
                      className={`corner-button right-corner save-button ${showStoredTick ? 'stored' : ''}`}
                      onClick={handleStoreTryOn}
                      disabled={showStoredTick}
                      title="Save to Virtual Closet"
                    >
                      {showStoredTick ? '✓' : '⬇'}
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Wall - Pants Section */}
        <div className="closet-wall wall-right">
          <div className="clothing-rail">
            <div className="rail-bar"></div>
            <div className="hangers-container">
              {clothingItems.pants.length === 0 ? (
                <div className="empty-section">
                  <p>No pants in your collection yet</p>
                </div>
              ) : (
                clothingItems.pants.slice().reverse().map((item, index) => (
                  <div 
                    key={item.id || index} 
                    className="hanger-item"
                    style={{ '--delay': `${index * 0.1}s` }}
                    onClick={() => handleItemClick(item, 'pants')}
                  >
                    <div className="hanger">
                      <div className="hanger-hook"></div>
                      <div className="hanger-body"></div>
                    </div>
                    <div className="clothing-item pants-item">
                      <img 
                        src={item.image || item.url || item.imageUrl} 
                        alt={item.category || item.type || 'Pants'}
                        loading="lazy"
                        onError={(e) => {
                          console.error('Failed to load image:', item.image || item.url || item.imageUrl);
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Shoe Rack at Bottom */}
      <div className="shoe-rack-container">
        <div className="shoe-rack">
          <div className="shoe-rack-display">
            {clothingItems.shoes.length === 0 ? (
              <div className="empty-section">
                <p>No shoes in your collection yet</p>
              </div>
            ) : (
              <div className="shoes-horizontal-grid">
                {clothingItems.shoes.map((item, index) => (
                  <div 
                    key={item.id || index} 
                    className="shoe-display"
                    style={{ '--delay': `${index * 0.1}s` }}
                    onClick={() => handleItemClick(item, 'shoes')}
                  >
                    <div className="shoe-platform">
                      <div className="shoe-item">
                        <img 
                          src={item.image || item.url || item.imageUrl} 
                          alt={item.category || item.type || 'Shoe'}
                          loading="lazy"
                          onError={(e) => {
                            console.error('Failed to load image:', item.image || item.url || item.imageUrl);
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                      <div className="shoe-reflection"></div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toast Notification */}
      {toastMessage && (
        <div className={`toast-notification ${toastMessage.includes('Failed') ? 'error' : 'success'}`}>
          {toastMessage}
        </div>
      )}

    </div>
  );
}

export default LuxuryClosetHome;