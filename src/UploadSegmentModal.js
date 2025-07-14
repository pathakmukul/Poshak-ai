import React, { useState, useRef } from 'react';
import './UploadSegmentModal.css';
import { uploadUserImage, saveMaskData, saveMaskImage } from './storageService';
import API_URL from './config';

function UploadSegmentModal({ user, onClose, onSuccess }) {
  const [currentStep, setCurrentStep] = useState('upload'); // upload, segment, edit
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [segmentResults, setSegmentResults] = useState(null);
  const [activeTab, setActiveTab] = useState('all');
  const [editMode, setEditMode] = useState(false);
  const [editableMasks, setEditableMasks] = useState([]);
  const [selectedMaskIndices, setSelectedMaskIndices] = useState({ shirt: [], pants: [], shoes: [] });
  const [editCategory, setEditCategory] = useState('shirt'); // Separate state for edit mode category
  const [status, setStatus] = useState('');
  const [personExtractionViz, setPersonExtractionViz] = useState(null);
  const fileInputRef = useRef(null);

  const handleLocalFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Read file locally without uploading to Firebase
    const reader = new FileReader();
    reader.onload = (event) => {
      setUploadedImage({
        file: file,
        localDataUrl: event.target.result,
        fileName: file.name
      });
      setCurrentStep('segment');
    };
    reader.readAsDataURL(file);
  };

  const processSegmentation = async () => {
    if (!uploadedImage) return;

    setProcessing(true);
    setStatus('Processing with Replicate API (fast mode)...');

    try {
      // Call Flask API to process segmentation
      const response = await fetch(`${API_URL}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_data: uploadedImage.localDataUrl, // Send base64 data directly
          model_size: 'replicate' // Always use Replicate for wardrobe - it's fast!
        }),
      });

      if (!response.ok) {
        throw new Error('Segmentation failed');
      }

      const data = await response.json();
      setSegmentResults(data);
      setStatus('Segmentation complete!');
      
      // Store person extraction visualization if available
      if (data.person_extraction_viz) {
        setPersonExtractionViz(data.person_extraction_viz);
      }
      
      // If we need to edit masks, prepare the editable data
      if (data.masks) {
        setEditableMasks(data.masks);
      }
    } catch (error) {
      console.error('Segmentation error:', error);
      setStatus('Segmentation failed. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  const handleEditMasks = () => {
    setEditMode(true);
    // Load all masks for editing
    if (segmentResults && segmentResults.masks) {
      setEditableMasks(segmentResults.masks);
      
      // Initialize selectedMaskIndices with current selections
      const currentSelections = { shirt: [], pants: [], shoes: [] };
      
      // Debug: log all masks to see their state
      console.log('All masks before initializing edit:', segmentResults.masks.map((m, i) => ({
        idx: i,
        label: m.label,
        skip_viz: m.skip_viz,
        original_label: m.original_label
      })));
      
      segmentResults.masks.forEach((mask, idx) => {
        if (mask.label === 'shirt' && !mask.skip_viz) {
          currentSelections.shirt.push(idx);
        } else if (mask.label === 'pants' && !mask.skip_viz) {
          currentSelections.pants.push(idx);
        } else if (mask.label === 'shoes' && !mask.skip_viz) {
          currentSelections.shoes.push(idx);
        }
      });
      
      console.log('Initializing edit mode with selections:', currentSelections);
      setSelectedMaskIndices(currentSelections);
      
      // Set edit category based on active tab (but not 'all')
      if (activeTab !== 'all') {
        setEditCategory(activeTab);
      } else {
        setEditCategory('shirt'); // Default to shirt if viewing all
      }
    }
  };

  const applyMaskEdits = async () => {
    setProcessing(true);
    setStatus('Applying changes...');
    
    try {
      console.log('Sending mask selections to server:', selectedMaskIndices);
      // Use the simpler update-mask-labels endpoint
      const response = await fetch(`${API_URL}/update-mask-labels`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mask_selections: selectedMaskIndices
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to apply mask edits');
      }

      const data = await response.json();
      
      // Update the editableMasks with new labels based on selectedMaskIndices
      const updatedMasks = editableMasks.map((mask, idx) => {
        // Reset the mask label first
        const updatedMask = { ...mask, label: 'non_clothing', skip_viz: true };
        
        // Check if this mask is selected in any category
        if (selectedMaskIndices.shirt?.includes(idx)) {
          updatedMask.label = 'shirt';
          updatedMask.skip_viz = false;
        } else if (selectedMaskIndices.pants?.includes(idx)) {
          updatedMask.label = 'pants';
          updatedMask.skip_viz = false;
        } else if (selectedMaskIndices.shoes?.includes(idx)) {
          updatedMask.label = 'shoes';
          updatedMask.skip_viz = false;
        }
        
        return updatedMask;
      });
      
      // Update the segment results with new visualizations
      setSegmentResults({
        ...segmentResults,
        ...data,
        masks: updatedMasks // Use the updated mask list
      });
      setEditableMasks(updatedMasks); // Also update editableMasks
      
      setEditMode(false);
      setStatus('Changes applied!');
    } catch (error) {
      console.error('Error applying mask edits:', error);
      setStatus('Failed to apply changes');
    } finally {
      setProcessing(false);
    }
  };

  const saveToWardrobe = async () => {
    if (!segmentResults || !uploadedImage) return;

    setProcessing(true);
    setStatus('Saving to wardrobe...');

    try {
      // First, upload the original image to Firebase
      const uploadResult = await uploadUserImage(user.uid, uploadedImage.file);
      if (!uploadResult.success) {
        throw new Error('Failed to upload image');
      }

      // Save mask data including the visualization images
      const maskData = {
        masks: segmentResults.masks,
        classifications: {
          shirt: segmentResults.shirt_count,
          pants: segmentResults.pants_count,
          shoes: segmentResults.shoes_count
        },
        visualizations: {
          shirt: segmentResults.shirt_img,
          pants: segmentResults.pants_img,
          shoes: segmentResults.shoes_img,
          all: segmentResults.all_items_img
        },
        closet_visualizations: segmentResults.closet_visualizations || {},
        timestamp: new Date().toISOString(),
        originalImageUrl: uploadResult.downloadURL
      };

      // Use the fileName from uploadResult which includes the timestamp
      const imageName = uploadResult.fileName.split('.')[0];
      await saveMaskData(user.uid, imageName, maskData);

      // Save mask images
      for (const type of ['shirt', 'pants', 'shoes']) {
        if (segmentResults[`${type}_img`]) {
          // Convert base64 to blob
          const base64Data = segmentResults[`${type}_img`];
          const byteCharacters = atob(base64Data);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: 'image/png' });

          await saveMaskImage(user.uid, imageName, type, blob);
        }
      }

      setStatus('Saved to wardrobe!');
      setTimeout(() => {
        onSuccess();
      }, 1000);
    } catch (error) {
      console.error('Save error:', error);
      setStatus('Failed to save. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="upload-segment-modal">
      <div className="modal-header">
        <h2>Add to Wardrobe</h2>
        <button className="close-modal-button" onClick={onClose}>
          ✕
        </button>
      </div>

      <div className="modal-body">
        {currentStep === 'upload' && (
          <div className="upload-step">
            <h3>Upload Your Image</h3>
            <p>Select a photo of yourself to analyze and add to your wardrobe</p>
            <div className="local-upload-container">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleLocalFileSelect}
                style={{ display: 'none' }}
              />
              <button 
                className="upload-button-large"
                onClick={() => fileInputRef.current?.click()}
              >
                📸 Choose Photo
              </button>
              <p className="upload-hint">Your image will be processed locally until you save to wardrobe</p>
            </div>
          </div>
        )}

        {currentStep === 'segment' && uploadedImage && (
          <div className="segment-step">
            <div className="image-preview">
              <img src={uploadedImage.localDataUrl} alt="Uploaded" />
            </div>
            
            {!segmentResults ? (
              <div className="segment-controls">
                <h3>Ready to Analyze</h3>
                <p>Click the button below to identify clothing items in your image</p>
                <button 
                  className="segment-button-large"
                  onClick={processSegmentation}
                  disabled={processing}
                >
                  {processing ? (
                    <>
                      <span className="spinner"></span>
                      Processing...
                    </>
                  ) : (
                    '🎯 Segment Clothing'
                  )}
                </button>
                {status && <p className="status-message">{status}</p>}
                
                {/* Show MediaPipe person extraction visualization */}
                {personExtractionViz && (
                  <div className="person-extraction-viz" style={{ marginTop: '20px', textAlign: 'center' }}>
                    <h4 style={{ color: '#4CAF50' }}>MediaPipe Person Detection:</h4>
                    <img 
                      src={`data:image/png;base64,${personExtractionViz}`} 
                      alt="Person extraction visualization"
                      style={{ maxWidth: '100%', maxHeight: '400px', marginTop: '10px', border: '2px solid #4CAF50', borderRadius: '8px' }}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="results-section">
                <h3>Detected Clothing Items</h3>
                
                {/* Show MediaPipe person extraction visualization */}
                {personExtractionViz && (
                  <div className="person-extraction-viz" style={{ marginTop: '20px', textAlign: 'center' }}>
                    <h4 style={{ color: '#4CAF50' }}>MediaPipe Person Detection:</h4>
                    <img 
                      src={`data:image/png;base64,${personExtractionViz}`} 
                      alt="Person extraction visualization"
                      style={{ maxWidth: '100%', maxHeight: '400px', marginTop: '10px', border: '2px solid #4CAF50', borderRadius: '8px' }}
                    />
                  </div>
                )}
                
                {/* Tabs for categories */}
                <div className="category-tabs">
                  <button 
                    className={`tab ${activeTab === 'all' ? 'active' : ''}`}
                    onClick={() => setActiveTab('all')}
                  >
                    All Items
                  </button>
                  <button 
                    className={`tab ${activeTab === 'shirt' ? 'active' : ''}`}
                    onClick={() => setActiveTab('shirt')}
                  >
                    Shirts ({segmentResults.shirt_count || 0})
                  </button>
                  <button 
                    className={`tab ${activeTab === 'pants' ? 'active' : ''}`}
                    onClick={() => setActiveTab('pants')}
                  >
                    Pants ({segmentResults.pants_count || 0})
                  </button>
                  <button 
                    className={`tab ${activeTab === 'shoes' ? 'active' : ''}`}
                    onClick={() => setActiveTab('shoes')}
                  >
                    Shoes ({segmentResults.shoes_count || 0})
                  </button>
                </div>

                {/* Display segmented images */}
                <div className="segmented-images">
                  {activeTab === 'all' && segmentResults.all_items_img && (
                    <img src={`data:image/png;base64,${segmentResults.all_items_img}`} alt="All items" />
                  )}
                  {activeTab === 'shirt' && segmentResults.shirt_img && (
                    <img src={`data:image/png;base64,${segmentResults.shirt_img}`} alt="Shirts" />
                  )}
                  {activeTab === 'pants' && segmentResults.pants_img && (
                    <img src={`data:image/png;base64,${segmentResults.pants_img}`} alt="Pants" />
                  )}
                  {activeTab === 'shoes' && segmentResults.shoes_img && (
                    <img src={`data:image/png;base64,${segmentResults.shoes_img}`} alt="Shoes" />
                  )}
                </div>

                {/* Action buttons */}
                <div className="action-buttons">
                  <button 
                    className="edit-button"
                    onClick={handleEditMasks}
                  >
                    ✏️ Edit Selections
                  </button>
                  <button 
                    className="save-button"
                    onClick={saveToWardrobe}
                    disabled={processing}
                  >
                    {processing ? 'Saving...' : '💾 Save to Wardrobe'}
                  </button>
                </div>

                {status && <p className="status-message">{status}</p>}
              </div>
            )}
          </div>
        )}

        {/* Edit Mode Overlay */}
        {editMode && (
          <div className="edit-overlay">
            <div className="edit-header">
              <h3>Edit {editCategory.charAt(0).toUpperCase() + editCategory.slice(1)} Selection</h3>
              <p>Click masks to add/remove them from {editCategory}</p>
              <div className="edit-category-selector">
                <label>Switch category:</label>
                <button 
                  className={`category-btn shirt-btn ${editCategory === 'shirt' ? 'active' : ''}`}
                  onClick={() => setEditCategory('shirt')}
                >
                  👔 Shirt
                </button>
                <button 
                  className={`category-btn pants-btn ${editCategory === 'pants' ? 'active' : ''}`}
                  onClick={() => setEditCategory('pants')}
                >
                  👖 Pants
                </button>
                <button 
                  className={`category-btn shoes-btn ${editCategory === 'shoes' ? 'active' : ''}`}
                  onClick={() => setEditCategory('shoes')}
                >
                  👟 Shoes
                </button>
              </div>
            </div>

            <div className="masks-grid">
              {editableMasks.map((mask, idx) => {
                // Only show selection color for current category
                const isSelectedInCurrentCategory = selectedMaskIndices[editCategory]?.includes(idx);
                const isSelectedInOtherCategory = ['shirt', 'pants', 'shoes']
                  .filter(cat => cat !== editCategory)
                  .some(cat => selectedMaskIndices[cat]?.includes(idx));
                
                const selectionClass = isSelectedInCurrentCategory ? `selected-${editCategory}` : 
                                     isSelectedInOtherCategory ? 'selected-other' : '';
                
                return (
                  <div
                    key={idx}
                    className={`mask-item ${selectionClass}`}
                    onClick={() => {
                      const newSelections = { ...selectedMaskIndices };
                      
                      if (isSelectedInCurrentCategory) {
                        // Remove from current category
                        newSelections[editCategory] = newSelections[editCategory].filter(i => i !== idx);
                      } else {
                        // First remove from ALL categories to ensure no duplicates
                        ['shirt', 'pants', 'shoes'].forEach(cat => {
                          if (newSelections[cat]) {
                            newSelections[cat] = newSelections[cat].filter(i => i !== idx);
                          }
                        });
                        // Then add to current category
                        newSelections[editCategory] = [...(newSelections[editCategory] || []), idx];
                      }
                      
                      setSelectedMaskIndices(newSelections);
                    }}
                  >
                    <img 
                      src={`data:image/png;base64,${mask.cropped_img}`} 
                      alt={`Mask ${idx}`} 
                    />
                    <div className="mask-info">
                      <div>{mask.original_label || mask.full_label || mask.label}</div>
                      <div className="mask-confidence">({((mask.original_confidence || mask.confidence || 0) * 100).toFixed(0)}%)</div>
                      {isSelectedInCurrentCategory && (
                        <span className="category-tag">
                          {editCategory === 'shirt' ? '👔' : editCategory === 'pants' ? '👖' : '👟'}
                        </span>
                      )}
                      {isSelectedInOtherCategory && (
                        <span className="category-tag other-category">
                          {selectedMaskIndices.shirt?.includes(idx) ? '👔' : 
                           selectedMaskIndices.pants?.includes(idx) ? '👖' : '👟'}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="edit-actions">
              <button onClick={() => setEditMode(false)}>Cancel</button>
              <button onClick={applyMaskEdits} className="apply-button">
                Apply Changes
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadSegmentModal;