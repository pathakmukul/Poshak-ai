import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
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
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);
  const statusIntervalRef = useRef(null);

  // Available images
  const personImages = [
    'person.png',
    'person2.png'
  ];

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

    // Update status periodically
    let statusSteps = 0;
    statusIntervalRef.current = setInterval(() => {
      if (statusSteps === 0) setStatus('Loading models...');
      else if (statusSteps === 1) setStatus('Running SAM2 segmentation...');
      else if (statusSteps === 2) setStatus('Classifying with SigLIP...');
      statusSteps++;
    }, 5000);

    try {
      const response = await fetch('/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_path: `data/sample_images/people/${selectedImage}`,
          model_size: selectedModel
        }),
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

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Fetch available garments and mask images
  useEffect(() => {
    // Fetch garments
    fetch('/garments')
      .then(res => res.json())
      .then(data => {
        if (data.garments) {
          setGarments(data.garments);
          if (data.garments.length > 0) {
            setSelectedGarment(data.garments[0]);
          }
        }
      })
      .catch(err => console.error('Failed to fetch garments:', err));
      
    // Fetch available mask images
    fetch('/list-mask-images')
      .then(res => res.json())
      .then(data => {
        if (data.mask_images) {
          console.log('Available mask images:', data.mask_images);
        }
      })
      .catch(err => console.error('Failed to fetch mask images:', err));
  }, []);

  // Check for available masks when image changes
  useEffect(() => {
    checkAvailableMasks();
    
    // Check if saved masks exist
    if (selectedImage) {
      fetch(`/quick-load-masks/${selectedImage}`)
        .then(res => res.json())
        .then(data => {
          setHasSavedMasks(data.has_masks || false);
        })
        .catch(err => console.error('Error checking masks:', err));
    }
  }, [selectedImage]);

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
        <h1>🎯 SAM2 Clothing Detection</h1>
        <p>Select a person image to detect and segment clothing items</p>
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
              >
                <option value="">Choose an image...</option>
                {personImages.map(img => (
                  <option key={img} value={img}>{img}</option>
                ))}
              </select>
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
    </div>
  );
}

export default App;