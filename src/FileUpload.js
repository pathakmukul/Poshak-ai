import React, { useState } from 'react';
import { uploadUserImage } from './storageService';
import './FileUpload.css';

function FileUpload({ userId, onUploadSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    setUploading(true);
    
    try {
      const result = await uploadUserImage(userId, file);
      
      if (result.success) {
        // Notify parent component
        onUploadSuccess({
          fileName: result.fileName,
          url: result.downloadURL,
          path: result.path
        });
        
        // Clear the input
        const input = document.getElementById('file-upload-input');
        if (input) input.value = '';
      } else {
        alert(`Upload failed: ${result.error}`);
      }
    } catch (error) {
      alert(`Upload error: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <form 
        className={`file-upload-form ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onSubmit={(e) => e.preventDefault()}
      >
        <input
          id="file-upload-input"
          type="file"
          accept="image/*"
          onChange={handleChange}
          disabled={uploading}
          className="file-upload-input"
        />
        
        <label 
          htmlFor="file-upload-input" 
          className="file-upload-label"
        >
          <div>
            {uploading ? (
              <>
                <div className="upload-spinner"></div>
                <p>Uploading...</p>
              </>
            ) : (
              <>
                <svg
                  className="upload-icon"
                  fill="currentColor"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                >
                  <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
                </svg>
                <p>Drag & drop your image here</p>
                <p className="upload-text-small">or click to browse</p>
              </>
            )}
          </div>
        </label>
      </form>
      
      <div className="upload-info">
        <p>Supported formats: JPG, PNG, GIF</p>
        <p>Maximum file size: 10MB</p>
      </div>
    </div>
  );
}

export default FileUpload;