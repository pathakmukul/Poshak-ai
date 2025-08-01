import React from 'react';
import loaderGif from '../assets/poshakloader.gif';
import './Loader.css';

function Loader({ message = "Loading" }) {
  return (
    <div className="loader-container">
      <img src={loaderGif} alt="Loading" className="loader-gif" />
      {message && (
        <p className="loader-message">
          {message}
          <span className="loading-dots">
            <span className="dot">.</span>
            <span className="dot">.</span>
            <span className="dot">.</span>
          </span>
        </p>
      )}
    </div>
  );
}

export default Loader;