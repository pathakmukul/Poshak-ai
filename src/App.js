import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Login from './Login';
import FileUpload from './FileUpload';
import Wardrobe from './Wardrobe';
import { auth, logoutUser } from './firebase';
import { onAuthStateChanged } from 'firebase/auth';
import { getUserImages, getSharedGarments } from './storageService';

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [currentView, setCurrentView] = useState('main'); // 'main' or 'wardrobe'
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedModel, setSelectedModel] = useState('large');
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [status, setStatus] = useState('');
  const [timings, setTimings] = useState({ sam2: 0, siglip: 0, total: 0 });
  const [activeTab, setActiveTab] = useState('all');
  const [garments, setGarments] = useState([]);
  const [selectedGarment, setSelectedGarment] = useState('');
  const [tryonProcessing, setTryonProcessing] = useState(false);
  const [tryonResult, setTryonResult] = useState(null);
  const [tryonTime, setTryonTime] = useState(0);
  const [availableMasks, setAvailableMasks] = useState([]);
  const [hasSavedMasks, setHasSavedMasks] = useState(false);
  const [personImages, setPersonImages] = useState([]);
  const [userImages, setUserImages] = useState([]);
  const [showUpload, setShowUpload] = useState(false);
  const [sam2Preview, setSam2Preview] = useState(null);
  const [sam2Time, setSam2Time] = useState(0);
  const [showDebug, setShowDebug] = useState(false);
  const [debugInfo, setDebugInfo] = useState(null);
  const [showRawSAM2, setShowRawSAM2] = useState(false);
  const [showMaskEditor, setShowMaskEditor] = useState(false);
  const [editableMasks, setEditableMasks] = useState([]);
  const [selectedMaskIndices, setSelectedMaskIndices] = useState({ shirt: [], pants: [], shoes: [] });
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);
  const statusIntervalRef = useRef(null);

  // Handle user login
  const handleLogin = (user) => {
    setCurrentUser(user);
  };

  // Handle logout
  const handleLogout = async () => {
    try {
      await logoutUser();
      setCurrentUser(null);
      // Reset app state
      setSelectedImage(null);
      setResults(null);
      setTryonResult(null);
      setUserImages([]);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };
  
  // Handle file upload success
  const handleUploadSuccess = async (uploadedFile) => {
    // Refresh user images
    const result = await getUserImages(currentUser.uid);
    if (result.success) {
      setUserImages(result.images);
      // Select the newly uploaded image
      setSelectedImage(uploadedFile.url);
      setShowUpload(false);
    }
  };

  // Listen to Firebase auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        // User is signed in
        setCurrentUser({
          uid: user.uid,
          email: user.email,
          displayName: user.displayName || user.email.split('@')[0],
          username: user.displayName || user.email.split('@')[0]
        });
      } else {
        // User is signed out
        setCurrentUser(null);
      }
    });

    // Cleanup subscription
    return () => unsubscribe();
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Fetch available images
  useEffect(() => {
    if (!currentUser) return; // Skip if not logged in
    
    // Fetch user's Firebase Storage images
    const fetchUserImages = async () => {
      const result = await getUserImages(currentUser.uid);
      if (result.success) {
        setUserImages(result.images);
      }
    };
    
    // Fetch shared garments from Firebase Storage
    const fetchGarments = async () => {
      const result = await getSharedGarments();
      if (result.success) {
        setGarments(result.garments);
        if (result.garments.length > 0) {
          setSelectedGarment(result.garments[0].name);
        }
      }
    };
    
    fetchUserImages();
    fetchGarments();
    
    // Still fetch local demo images as fallback
    fetch('/people')
      .then(res => res.json())
      .then(data => {
        if (data.people) {
          setPersonImages(data.people);
        }
      })
      .catch(err => console.error('Failed to fetch demo people:', err));
  }, [currentUser]); // Re-run when user logs in

  // Check for available masks when image changes
  useEffect(() => {
    if (!currentUser) return; // Skip if not logged in
    
    checkAvailableMasks();
    
    // Check if saved masks exist in new or old location
    if (selectedImage) {
      fetch(`/quick-load-masks/${selectedImage}`)
        .then(res => res.json())
        .then(data => {
          setHasSavedMasks(data.has_masks || false);
        })
        .catch(err => console.error('Error checking masks:', err));
    }
  }, [selectedImage, currentUser]);

  // If not logged in, show login page
  if (!currentUser) {
    return <Login onLogin={handleLogin} />;
  }

  // Show wardrobe view if selected
  if (currentView === 'wardrobe') {
    return (
      <Wardrobe 
        user={currentUser} 
        onBack={() => setCurrentView('main')} 
      />
    );
  }

  // Start timer
  const startTimer = () => {
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      setElapsedTime(elapsed);
    }, 100); // Update every 100ms
  };

  // Stop timer
  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // Check for available mask images
  const checkAvailableMasks = () => {
    if (!selectedImage) return;
    
    const baseName = selectedImage.split('.')[0];
    const maskTypes = ['shirt', 'pants', 'shoes'];
    const masks = [];
    
    // Check which mask images exist
    maskTypes.forEach(type => {
      const maskImage = `${baseName}_mask_${type}.png`;
      // We'll assume they exist if we've processed this image before
      masks.push({ type, image: maskImage });
    });
    
    setAvailableMasks(masks);
  };

  // Process image
  const processImage = async () => {
    setProcessing(true);
    setResults(null);
    setStatus('Loading models...');
    startTimer();

    try {
      // Determine if we're using a user image or demo image
      const isUserImage = selectedImage && selectedImage.startsWith('http');
      
      const requestBody = isUserImage ? {
        image_url: selectedImage,
        model_size: selectedModel
      } : {
        image_path: `data/sample_images/people/${selectedImage}`,
        model_size: selectedModel
      };
      
      const response = await fetch('/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Processing failed');
      }

      const data = await response.json();
      stopTimer();
      clearInterval(statusIntervalRef.current);
      
      setResults(data);
      setTimings({
        sam2: data.sam2_time,
        siglip: data.siglip_time,
        total: data.total_time
      });
      setStatus('Processing complete!');
    } catch (error) {
      stopTimer();
      clearInterval(statusIntervalRef.current);
      setStatus(`Error: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  // Gemini Virtual Try-on
  const performGeminiTryon = async () => {
    if (!selectedImage || !selectedGarment) {
      alert('Please select both an image and a garment');
      return;
    }

    // If no results but we have saved masks, load them first
    if (!results && hasSavedMasks) {
      try {
        const res = await fetch(`/quick-load-masks/${selectedImage}`);
        const data = await res.json();
        if (data.success) {
          setResults(data);
          setActiveTab('shirt'); // Default to shirt
          // Continue with try-on after masks are loaded
          setTimeout(() => performGeminiTryon(), 100);
          return;
        }
      } catch (err) {
        console.error('Failed to load masks:', err);
      }
    }

    if (!results) {
      alert('Please generate masks first! Click "Generate Masks" button above.');
      return;
    }

    if (activeTab === 'all') {
      alert('Please select a specific clothing type (Shirt, Pants, or Shoes) from the Detection Results tabs');
      return;
    }

    setTryonProcessing(true);
    setTryonResult(null);
    const startTime = Date.now();

    try {
      // First, get original image and mask separately
      const geminiDataResponse = await fetch('/get-gemini-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_path: `data/sample_images/people/${selectedImage}`,
          clothing_type: activeTab
        }),
      });

      const geminiData = await geminiDataResponse.json();
      
      if (!geminiData.success) {
        throw new Error(geminiData.error || 'Failed to get image data');
      }

      const response = await fetch('/gemini-tryon', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          person_image: `data:image/png;base64,${geminiData.original_image}`,
          mask_image: `data:image/png;base64,${geminiData.mask_image}`,
          garment_file: selectedGarment,
          clothing_type: activeTab
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        setTryonResult(data.result_image);
        setTryonTime(data.processing_time);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setTryonProcessing(false);
      const elapsed = (Date.now() - startTime) / 1000;
      setTryonTime(elapsed);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1><img src="/images/logo.png" alt="KapdaAI" style={{ height: '40px', marginRight: '10px', verticalAlign: 'middle' }} />SAM2 Clothing Detection</h1>
            <p>Select a person image to detect and segment clothing items</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <span style={{ color: '#ecf0f1', fontSize: '16px' }}>
              👤 {currentUser.username}
            </span>
            <button
              onClick={() => setCurrentView('wardrobe')}
              style={{
                padding: '8px 16px',
                backgroundColor: '#27ae60',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                marginRight: '10px'
              }}
            >
              👗 My Wardrobe
            </button>
            <button
              onClick={handleLogout}
              style={{
                padding: '8px 16px',
                backgroundColor: '#e74c3c',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                transition: 'background-color 0.3s'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#c0392b'}
              onMouseLeave={(e) => e.target.style.backgroundColor = '#e74c3c'}
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="container">
        <div className="sidebar">
          <div className="timing-panel">
            <h3>⏱️ Performance Metrics</h3>
            {processing && (
              <div className="timer-live">
                <div className="elapsed-time">{elapsedTime.toFixed(1)}s</div>
                <div className="status">{status}</div>
              </div>
            )}
            {!processing && timings.total > 0 && (
              <div className="timing-results">
                <div className="metric">
                  <span>SAM2 Time:</span>
                  <span>{timings.sam2.toFixed(2)}s</span>
                </div>
                <div className="metric">
                  <span>SigLIP Time:</span>
                  <span>{timings.siglip.toFixed(2)}s</span>
                </div>
                <div className="metric total">
                  <span>Total Time:</span>
                  <span>{timings.total.toFixed(2)}s</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="main-content">
          <div className="control-panel">
            <div className="image-selector">
              <label>Select Image:</label>
              <select 
                value={selectedImage || ''} 
                onChange={(e) => setSelectedImage(e.target.value)}
                disabled={processing}
                style={{ marginBottom: '10px' }}
              >
                <option value="">Choose an image...</option>
                {userImages.length > 0 && (
                  <optgroup label="Your Images">
                    {userImages.map(img => (
                      <option key={img.path} value={img.url}>{img.name}</option>
                    ))}
                  </optgroup>
                )}
                {personImages.length > 0 && (
                  <optgroup label="Demo Images">
                    {personImages.map(img => (
                      <option key={img} value={img}>{img}</option>
                    ))}
                  </optgroup>
                )}
              </select>
              <button
                onClick={() => setShowUpload(!showUpload)}
                style={{
                  padding: '6px 12px',
                  marginLeft: '10px',
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                📤 Upload New
              </button>
            </div>
            
            <div className="model-selector">
              <label>Model Size:</label>
              <select 
                value={selectedModel} 
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={processing}
              >
                <option value="tiny">SAM2 Tiny (Fast, ~10s)</option>
                <option value="small">SAM2 Small (Better, ~20s)</option>
                <option value="base">SAM2 Base+ (Good, ~40s)</option>
                <option value="large">SAM2 Large (Best, ~60s)</option>
                <optgroup label="SAM 2.1 (Better Accuracy)">
                  <option value="tiny_v2.1">SAM2.1 Tiny (Fast, ~10s)</option>
                  <option value="small_v2.1">SAM2.1 Small (Better, ~20s)</option>
                  <option value="base_v2.1">SAM2.1 Base+ (Good, ~40s)</option>
                  <option value="large_v2.1">SAM2.1 Large (Best, ~60s)</option>
                </optgroup>
                <optgroup label="Cloud Processing">
                  <option value="replicate">Replicate API (Fast, ~5-10s)</option>
                </optgroup>
              </select>
            </div>
            
            <button 
              className="process-btn"
              onClick={processImage}
              disabled={!selectedImage || processing}
            >
              {processing ? 'Processing...' : '🚀 Generate Masks'}
            </button>
            
            {results && (
              <button 
                className="debug-btn"
                onClick={async () => {
                  const res = await fetch('/get-debug-info');
                  const data = await res.json();
                  setDebugInfo(data.debug_data);
                  setShowDebug(true);
                }}
                style={{
                  marginLeft: '10px',
                  padding: '8px 20px',
                  backgroundColor: '#e74c3c',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                🔍 Show SigLIP Debug
              </button>
            )}
            
            {results && results.raw_sam2_img && (
              <button
                onClick={() => {
                  setShowRawSAM2(!showRawSAM2);
                }}
                style={{
                  marginLeft: '10px',
                  padding: '8px 20px',
                  backgroundColor: '#3498db',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                🎯 Show All SAM2 Segments ({results.raw_masks_count})
              </button>
            )}

          </div>

          {selectedImage && (
            <div className="content-row">
              <div className="content-column">
                <h3>📷 Original Image</h3>
                <div className="image-container">
                  <div className="image-frame">
                    <img 
                      src={`/static/${selectedImage}`} 
                      alt="Selected" 
                    />
                  </div>
                  <div className="image-info">
                    <p>File: {selectedImage}</p>
                  </div>
                </div>
              </div>

              {results && (
                <div className="content-column">
                  <div className="detection-container">
                    <div className="detection-header">
                      <h3>🎯 Detection Results</h3>
                      <div className="tabs">
                        <button 
                          className={`tab ${activeTab === 'all' ? 'active' : ''}`}
                          onClick={() => setActiveTab('all')}
                        >
                          All
                        </button>
                        <button 
                          className={`tab ${activeTab === 'shirt' ? 'active' : ''}`}
                          onClick={() => setActiveTab('shirt')}
                        >
                          Shirt
                        </button>
                        <button 
                          className={`tab ${activeTab === 'pants' ? 'active' : ''}`}
                          onClick={() => setActiveTab('pants')}
                        >
                          Pants
                        </button>
                        <button 
                          className={`tab ${activeTab === 'shoes' ? 'active' : ''}`}
                          onClick={() => setActiveTab('shoes')}
                        >
                          Shoes
                        </button>
                      </div>
                    </div>

                    <div className="results-display">
                    <div className="image-container">
                      {activeTab === 'all' && (
                        <>
                          <div className="image-frame">
                            <img src={`data:image/png;base64,${results.all_items_img}`} alt="All items" />
                          </div>
                          <div className="image-info">
                            <p>Total items detected: {results.all_items_count}</p>
                          </div>
                        </>
                      )}

                      {activeTab === 'shirt' && (
                        <>
                          <div className="image-frame">
                            <img src={`data:image/png;base64,${results.shirt_img}`} alt="Shirt" />
                          </div>
                          <div className="image-info">
                            <p>Shirts detected: {results.shirt_count}</p>
                            <button
                              onClick={async () => {
                                const res = await fetch('/get-all-masks-with-classes');
                                const data = await res.json();
                                setEditableMasks(data.masks || []);
                                setSelectedMaskIndices({ shirt: [], pants: [], shoes: [] });
                                setShowMaskEditor(true);
                              }}
                              style={{
                                marginTop: '10px',
                                padding: '6px 15px',
                                backgroundColor: '#e74c3c',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '14px'
                              }}
                            >
                              🔄 Redo Selection
                            </button>
                          </div>
                        </>
                      )}

                      {activeTab === 'pants' && (
                        <>
                          <div className="image-frame">
                            <img src={`data:image/png;base64,${results.pants_img}`} alt="Pants" />
                          </div>
                          <div className="image-info">
                            <p>Pants detected: {results.pants_count}</p>
                            <button
                              onClick={async () => {
                                const res = await fetch('/get-all-masks-with-classes');
                                const data = await res.json();
                                setEditableMasks(data.masks || []);
                                setSelectedMaskIndices({ shirt: [], pants: [], shoes: [] });
                                setShowMaskEditor(true);
                              }}
                              style={{
                                marginTop: '10px',
                                padding: '6px 15px',
                                backgroundColor: '#e74c3c',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '14px'
                              }}
                            >
                              🔄 Redo Selection
                            </button>
                          </div>
                        </>
                      )}

                      {activeTab === 'shoes' && (
                        <>
                          <div className="image-frame">
                            <img src={`data:image/png;base64,${results.shoes_img}`} alt="Shoes" />
                          </div>
                          <div className="image-info">
                            <p>Shoes detected: {results.shoes_count}</p>
                            <button
                              onClick={async () => {
                                const res = await fetch('/get-all-masks-with-classes');
                                const data = await res.json();
                                setEditableMasks(data.masks || []);
                                setSelectedMaskIndices({ shirt: [], pants: [], shoes: [] });
                                setShowMaskEditor(true);
                              }}
                              style={{
                                marginTop: '10px',
                                padding: '6px 15px',
                                backgroundColor: '#e74c3c',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '14px'
                              }}
                            >
                              🔄 Redo Selection
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Gemini Tryon Panel */}
          {selectedImage && (
            <div className="gemini-panel">
              <h2>🎨 Gemini Virtual Try-On</h2>
              {!results && (
                <div>
                  <p className="info-message">
                    ⚠️ Please generate masks first by clicking "Generate Masks" above
                  </p>
                  <p className="info-note">
                    Once generated, masks are automatically saved as {selectedImage?.split('.')[0]}_mask_shirt.png, etc.
                  </p>
                  {hasSavedMasks && (
                    <button 
                      className="quick-load-btn"
                      style={{ marginTop: '20px', width: '100%' }}
                      onClick={async () => {
                        try {
                          const res = await fetch(`/quick-load-masks/${selectedImage}`);
                          const data = await res.json();
                          if (data.success) {
                            setResults(data);
                            setTimings({ sam2: 0, siglip: 0, total: 0 });
                            setStatus('Loaded saved masks');
                            setActiveTab('shirt'); // Default to shirt tab
                          }
                        } catch (err) {
                          console.error('Failed to load masks:', err);
                        }
                      }}
                    >
                      ⚡ Use Saved Masks from Previous Run
                    </button>
                  )}
                </div>
              )}
              {results && activeTab === 'all' && (
                <p className="info-message">
                  ⚠️ Please select a specific clothing type (Shirt, Pants, or Shoes) from the Detection Results tabs above
                </p>
              )}
              
              {/* Show available saved masks info */}
              {hasSavedMasks && results && (
                <p className="info-note" style={{ color: '#27ae60', fontWeight: 'bold', marginBottom: '15px' }}>
                  ✅ Using saved masks from previous run
                </p>
              )}
              
              <div className="gemini-controls">
                <div className="control-group">
                  <label>Select Garment:</label>
                  <select
                    value={selectedGarment}
                    onChange={(e) => setSelectedGarment(e.target.value)}
                    disabled={tryonProcessing}
                  >
                    {garments.map(garment => (
                      <option key={garment} value={garment}>
                        {garment}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  className="tryon-btn"
                  onClick={performGeminiTryon}
                  disabled={tryonProcessing || !selectedGarment || (!results && !hasSavedMasks) || activeTab === 'all'}
                >
                  {tryonProcessing ? 'Processing...' : 
                   !results ? 'Generate masks first' :
                   activeTab === 'all' ? 'Select a clothing type above' : 
                   '✨ Try On with Gemini'}
                </button>

                {tryonTime > 0 && !tryonProcessing && (
                  <div className="tryon-time">
                    Processed in {tryonTime.toFixed(2)}s
                  </div>
                )}
              </div>

              {tryonResult && (
                <div className="tryon-results">
                  <div className="content-row">
                    <div className="content-column">
                      <h3>👔 Selected Garment</h3>
                      <div className="image-frame">
                        <img 
                          src={`/static/garments/${selectedGarment}`} 
                          alt="Garment" 
                        />
                      </div>
                    </div>
                    <div className="content-column">
                      <h3>✨ Virtual Try-On Result</h3>
                      <div className="image-frame">
                        <img 
                          src={`data:image/png;base64,${tryonResult}`} 
                          alt="Try-on result" 
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Debug Popup */}
      {showDebug && debugInfo && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            maxWidth: '800px',
            maxHeight: '80vh',
            overflow: 'auto',
            position: 'relative'
          }}>
            <button 
              onClick={() => setShowDebug(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                fontSize: '24px',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              ✕
            </button>
            
            <h2>🔍 SigLIP Classification Debug Info - ALL {debugInfo.length} Masks</h2>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Showing what we sent to SigLIP and what it returned for ALL masks (not just filtered ones)
            </p>
            
            {debugInfo.map((item, idx) => (
              <div key={idx} style={{
                marginBottom: '25px',
                padding: '15px',
                backgroundColor: item.kept ? '#e8f5e9' : '#ffebee',
                borderRadius: '5px',
                border: item.kept ? '2px solid #4caf50' : '2px solid #f44336',
                opacity: item.kept ? 1 : 0.8
              }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#2c3e50' }}>
                  Mask #{item.mask_number}: {item.full_label} ({(item.confidence * 100).toFixed(1)}%)
                  {item.kept ? ' ✅ KEPT' : ' ❌ FILTERED OUT'}
                </h3>
                
                <div style={{ display: 'flex', gap: '20px', marginBottom: '15px' }}>
                  <div>
                    <strong>Image sent to SigLIP:</strong>
                    <div style={{ marginTop: '5px' }}>
                      <img 
                        src={`data:image/png;base64,${item.debug.input_image}`} 
                        alt="Input to SigLIP"
                        style={{ 
                          maxWidth: '200px', 
                          maxHeight: '200px',
                          border: '2px solid #ddd',
                          borderRadius: '5px'
                        }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <strong>Mask Properties:</strong>
                    <ul style={{ margin: '5px 0' }}>
                      <li>Area: {item.debug.mask_area} pixels</li>
                      <li>Y Position: {(item.debug.position_y * 100).toFixed(1)}% from top</li>
                      <li>Aspect Ratio: {item.debug.aspect_ratio ? item.debug.aspect_ratio.toFixed(2) : 'N/A'}</li>
                    </ul>
                  </div>
                </div>
                
                <div>
                  <strong>Prompts we sent to SigLIP:</strong>
                  <div style={{ 
                    backgroundColor: 'white', 
                    padding: '10px', 
                    borderRadius: '3px',
                    fontFamily: 'monospace',
                    fontSize: '14px',
                    marginTop: '5px'
                  }}>
                    [{item.debug.prompts.map(p => `"${p}"`).join(', ')}]
                  </div>
                </div>
                
                <div style={{ marginTop: '10px' }}>
                  <strong>SigLIP Scores (what it thinks each prompt matches):</strong>
                  <div style={{ marginTop: '5px' }}>
                    {Object.entries(item.debug.all_scores)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 3)
                      .map(([prompt, score]) => (
                        <div key={prompt} style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '3px 0',
                          fontWeight: prompt === item.label ? 'bold' : 'normal',
                          color: prompt === item.label ? '#2c3e50' : '#7f8c8d'
                        }}>
                          <span>{prompt}:</span>
                          <span>{(score * 100).toFixed(1)}%</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
                
                {!item.kept && (
                  <div style={{ 
                    marginTop: '10px', 
                    padding: '10px', 
                    backgroundColor: '#fff3cd',
                    borderRadius: '5px',
                    color: '#856404'
                  }}>
                    <strong>Why filtered out:</strong> 
                    {item.confidence < 0.15 ? ' Low confidence' : 
                     item.label === 'background' ? ' Detected as background' :
                     item.label === 'non_clothing' ? ' Not clothing' :
                     ' Not the best mask for this category'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Raw SAM2 Segments Popup */}
      {showRawSAM2 && results && results.raw_sam2_img && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            maxWidth: '90%',
            maxHeight: '90vh',
            overflow: 'auto',
            position: 'relative'
          }}>
            <button 
              onClick={() => setShowRawSAM2(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                fontSize: '24px',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              ✕
            </button>
            
            <h2 style={{ marginTop: '0', marginBottom: '20px' }}>
              🎯 All SAM2 Detected Segments
            </h2>
            
            <div style={{ textAlign: 'center' }}>
              <img 
                src={`data:image/png;base64,${results.raw_sam2_img}`}
                alt="All SAM2 segments"
                style={{
                  maxWidth: '100%',
                  maxHeight: '70vh',
                  objectFit: 'contain'
                }}
              />
              <p style={{ marginTop: '20px', color: '#7f8c8d' }}>
                Each colored region with a number represents a segment detected by SAM2.
                <br />
                Total segments: {results.raw_masks_count}
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Mask Editor Popup */}
      {showMaskEditor && editableMasks.length > 0 && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          overflow: 'auto'
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            maxWidth: '90%',
            maxHeight: '90vh',
            overflow: 'auto',
            position: 'relative'
          }}>
            <button 
              onClick={() => setShowMaskEditor(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                fontSize: '24px',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              ✕
            </button>
            
            <h2 style={{ marginTop: '0', marginBottom: '20px' }}>
              🔄 Redo {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Selection
            </h2>
            
            <p style={{ marginBottom: '20px', color: '#666' }}>
              Select which segments should be classified as <strong>{activeTab}</strong>. 
              Click masks to select them. Green = currently selected.
            </p>
            
            {/* Quick select button */}
            <div style={{ marginBottom: '20px' }}>
              <button
                onClick={() => {
                  // Select all masks currently labeled as this category
                  const currentCategoryMasks = editableMasks
                    .map((mask, idx) => mask.current_label === activeTab ? idx : -1)
                    .filter(idx => idx !== -1);
                  setSelectedMaskIndices({
                    ...selectedMaskIndices,
                    [activeTab]: currentCategoryMasks
                  });
                }}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#3498db',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                Select current {activeTab} masks
              </button>
              
              <button
                onClick={() => {
                  setSelectedMaskIndices({
                    ...selectedMaskIndices,
                    [activeTab]: []
                  });
                }}
                style={{
                  padding: '10px 20px',
                  marginLeft: '10px',
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                Clear selection
              </button>
            </div>
            
            {/* Mask grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
              gap: '15px',
              marginBottom: '20px'
            }}>
              {editableMasks.map((mask, idx) => (
                <div 
                  key={idx}
                  onClick={() => {
                    // Toggle selection for active category only
                    const currentSelections = selectedMaskIndices[activeTab] || [];
                    const isSelected = currentSelections.includes(idx);
                    
                    if (isSelected) {
                      setSelectedMaskIndices({
                        ...selectedMaskIndices,
                        [activeTab]: currentSelections.filter(i => i !== idx)
                      });
                    } else {
                      setSelectedMaskIndices({
                        ...selectedMaskIndices,
                        [activeTab]: [...currentSelections, idx]
                      });
                    }
                  }}
                  style={{
                    cursor: 'pointer',
                    border: `3px solid ${
                      (selectedMaskIndices[activeTab] || []).includes(idx) ? '#27ae60' : '#ddd'
                    }`,
                    borderRadius: '5px',
                    padding: '5px',
                    position: 'relative'
                  }}
                >
                  <img 
                    src={`data:image/png;base64,${mask.image}`}
                    alt={`Mask ${idx}`}
                    style={{ width: '100%', height: 'auto' }}
                  />
                  <div style={{
                    position: 'absolute',
                    top: '5px',
                    left: '5px',
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    color: 'white',
                    padding: '2px 5px',
                    borderRadius: '3px',
                    fontSize: '12px'
                  }}>
                    #{idx + 1}
                  </div>
                  <div style={{
                    marginTop: '5px',
                    fontSize: '12px',
                    textAlign: 'center'
                  }}>
                    <strong>{mask.current_label}</strong> ({(mask.confidence * 100).toFixed(0)}%)
                    <br />
                    {mask.top_predictions && (
                      <span style={{ color: '#666' }}>
                        Also: {mask.top_predictions.slice(0, 2).join(', ')}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
            
            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
              <button
                onClick={async () => {
                  // Apply the new classifications for active category only
                  const updates = {
                    [activeTab]: selectedMaskIndices[activeTab] || []
                  };
                  
                  const response = await fetch('/update-mask-classifications', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                      updates,
                      preserveOthers: true  // Don't reset other categories
                    })
                  });
                  
                  if (response.ok) {
                    const data = await response.json();
                    setResults(data);
                    setShowMaskEditor(false);
                  }
                }}
                style={{
                  padding: '12px 30px',
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                ✓ Apply {activeTab} Selection
              </button>
              
              <button
                onClick={() => setShowMaskEditor(false)}
                style={{
                  padding: '12px 30px',
                  backgroundColor: '#e74c3c',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* File Upload Modal */}
      {showUpload && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            maxWidth: '600px',
            width: '90%',
            position: 'relative'
          }}>
            <button 
              onClick={() => setShowUpload(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                fontSize: '24px',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              ✕
            </button>
            
            <h2 style={{ marginTop: 0, marginBottom: '20px', color: '#2c3e50' }}>
              Upload Your Image
            </h2>
            
            <FileUpload 
              userId={currentUser.uid}
              onUploadSuccess={handleUploadSuccess}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;