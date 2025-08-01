import React from 'react';
import './BackButton.css';

function BackButton({ onClick, label = 'Back' }) {
  return (
    <button className="luxury-back-button" onClick={onClick}>
      <svg 
        className="back-icon" 
        width="20" 
        height="20" 
        viewBox="0 0 24 24" 
        fill="none"
      >
        <path 
          d="M15 18L9 12L15 6" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round"
        />
      </svg>
      <span className="back-label">{label}</span>
      <span className="back-line"></span>
    </button>
  );
}

export default BackButton;