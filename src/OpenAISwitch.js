import React, { useState, useEffect } from 'react';
import './OpenAISwitch.css';

function OpenAISwitch({ user, onBack }) {
  const [selections, setSelections] = useState({
    MODEL: '',
    SHIRT: '',
    PANT: '',
    SHOES: '',
    Accessories: ''
  });
  
  const [availableImages, setAvailableImages] = useState({
    MODEL: [],
    SHIRT: [],
    PANT: [],
    SHOES: [],
    Accessories: []
  });
  
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [generationTime, setGenerationTime] = useState(null);
  
  // Load available images from folders
  useEffect(() => {
    loadAvailableImages();
  }, []);
  
  const loadAvailableImages = async () => {
    const categories = ['MODEL', 'SHIRT', 'PANT', 'SHOES', 'Accessories'];
    const images = {};
    
    for (const category of categories) {
      try {
        const response = await fetch(`http://localhost:5002/api/list-images/${category}`);
        if (response.ok) {
          const data = await response.json();
          images[category] = data.images || [];
        }
      } catch (error) {
        console.error(`Error loading ${category} images:`, error);
        images[category] = [];
      }
    }
    
    setAvailableImages(images);
  };
  
  const handleSelectionChange = (category, value) => {
    setSelections(prev => ({
      ...prev,
      [category]: value
    }));
  };
  
  const handleProcessOutfit = async () => {
    const selectedItems = Object.entries(selections)
      .filter(([_, value]) => value !== '')
      .map(([category, imagePath]) => ({ category, imagePath }));
    
    if (selectedItems.length === 0) {
      alert('Please select at least one item to process');
      return;
    }
    
    if (!selections.MODEL) {
      alert('Please select a model image');
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5002/api/openai-outfit-variation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selections: selectedItems,
          userId: user.uid
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process outfit');
      }
      
      const data = await response.json();
      setProcessedImage(data.imageUrl);
      setGenerationTime(data.generationTime);
    } catch (error) {
      console.error('Error processing outfit:', error);
      alert('Failed to process outfit. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="openai-switch">
      <div className="openai-header">
        <button className="back-button" onClick={onBack}>
          ← Back
        </button>
        <h2>OpenAI Outfit Generator</h2>
      </div>
      
      <div className="openai-content">
        <div className="selection-area">
          <h3>Select Items</h3>
          <div className="dropdowns-container">
            {Object.keys(selections).map(category => (
              <div key={category} className="dropdown-group">
                <label>{category}</label>
                <select
                  value={selections[category]}
                  onChange={(e) => handleSelectionChange(category, e.target.value)}
                  disabled={loading}
                >
                  <option value="">-- Select {category} --</option>
                  {availableImages[category].map((image, index) => (
                    <option key={index} value={image}>
                      {image.split('/').pop()}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
          
          <button 
            className="process-button"
            onClick={handleProcessOutfit}
            disabled={loading || !selections.MODEL}
          >
            {loading ? 'Processing...' : 'Generate Outfit'}
          </button>
        </div>
        
        <div className="preview-area">
          <h3>Preview</h3>
          <div className="preview-images">
            {Object.entries(selections).map(([category, imagePath]) => (
              imagePath && (
                <div key={category} className="preview-item">
                  <p>{category}</p>
                  <img 
                    src={`http://localhost:5002${imagePath}`} 
                    alt={category}
                    className={category === 'MODEL' ? 'model-image' : 'item-image'}
                  />
                </div>
              )
            ))}
          </div>
          
          {processedImage && (
            <div className="result-section">
              <h3>Generated Outfit</h3>
              {generationTime && (
                <p className="generation-time">Generated in {generationTime}</p>
              )}
              <img 
                src={`http://localhost:5002${processedImage}`} 
                alt="Generated outfit"
                className="result-image"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default OpenAISwitch;