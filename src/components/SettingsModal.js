import React, { useState, useEffect } from 'react';
import './SettingsModal.css';
import { auth } from '../firebase';
import { closetCache } from '../closetCache';

function SettingsModal({ user, onClose }) {
  const [activeSection, setActiveSection] = useState('user-info');
  const [loading, setLoading] = useState(false);
  const [imageCount, setImageCount] = useState(0);
  
  const [userPreferences, setUserPreferences] = useState({
    location: localStorage.getItem(`userLocation_${user.uid}`) || '',
    favoriteBrands: JSON.parse(localStorage.getItem(`favoriteBrands_${user.uid}`) || '[]'),
    favoriteGenres: JSON.parse(localStorage.getItem(`favoriteGenres_${user.uid}`) || '[]'),
    favoriteMovies: JSON.parse(localStorage.getItem(`favoriteMovies_${user.uid}`) || '[]')
  });
  
  const [newBrand, setNewBrand] = useState('');
  const [newGenre, setNewGenre] = useState('');
  const [newMovie, setNewMovie] = useState('');
  const [isAddingBrand, setIsAddingBrand] = useState(false);
  const [isAddingGenre, setIsAddingGenre] = useState(false);
  const [isAddingMovie, setIsAddingMovie] = useState(false);

  useEffect(() => {
    fetchImageCount();
  }, [user.uid]);

  const fetchImageCount = async () => {
    try {
      const cachedData = closetCache.get(user.uid);
      if (cachedData) {
        const totalCount = (cachedData.shirts?.length || 0) + 
                          (cachedData.pants?.length || 0) + 
                          (cachedData.shoes?.length || 0);
        setImageCount(totalCount);
      } else {
        const response = await fetch(`http://localhost:5001/firebase/clothing-counts/${user.uid}`);
        if (response.ok) {
          const counts = await response.json();
          const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
          setImageCount(total);
        }
      }
    } catch (error) {
      console.error('Error fetching image count:', error);
    }
  };

  const handleLocationChange = (e) => {
    const newLocation = e.target.value;
    setUserPreferences(prev => ({ ...prev, location: newLocation }));
    localStorage.setItem(`userLocation_${user.uid}`, newLocation);
  };

  const handleAddBrand = () => {
    if (newBrand.trim() && !userPreferences.favoriteBrands.includes(newBrand.trim())) {
      const updatedBrands = [...userPreferences.favoriteBrands, newBrand.trim()];
      setUserPreferences(prev => ({ ...prev, favoriteBrands: updatedBrands }));
      localStorage.setItem(`favoriteBrands_${user.uid}`, JSON.stringify(updatedBrands));
      setNewBrand('');
      setIsAddingBrand(false);
    }
  };

  const handleRemoveBrand = (brand) => {
    const updatedBrands = userPreferences.favoriteBrands.filter(b => b !== brand);
    setUserPreferences(prev => ({ ...prev, favoriteBrands: updatedBrands }));
    localStorage.setItem(`favoriteBrands_${user.uid}`, JSON.stringify(updatedBrands));
  };

  const handleAddGenre = () => {
    if (newGenre.trim() && !userPreferences.favoriteGenres.includes(newGenre.trim())) {
      const updatedGenres = [...userPreferences.favoriteGenres, newGenre.trim()];
      setUserPreferences(prev => ({ ...prev, favoriteGenres: updatedGenres }));
      localStorage.setItem(`favoriteGenres_${user.uid}`, JSON.stringify(updatedGenres));
      setNewGenre('');
      setIsAddingGenre(false);
    }
  };

  const handleRemoveGenre = (genre) => {
    const updatedGenres = userPreferences.favoriteGenres.filter(g => g !== genre);
    setUserPreferences(prev => ({ ...prev, favoriteGenres: updatedGenres }));
    localStorage.setItem(`favoriteGenres_${user.uid}`, JSON.stringify(updatedGenres));
  };

  const handleAddMovie = () => {
    if (newMovie.trim() && !userPreferences.favoriteMovies.includes(newMovie.trim())) {
      const updatedMovies = [...userPreferences.favoriteMovies, newMovie.trim()];
      setUserPreferences(prev => ({ ...prev, favoriteMovies: updatedMovies }));
      localStorage.setItem(`favoriteMovies_${user.uid}`, JSON.stringify(updatedMovies));
      setNewMovie('');
      setIsAddingMovie(false);
    }
  };

  const handleRemoveMovie = (movie) => {
    const updatedMovies = userPreferences.favoriteMovies.filter(m => m !== movie);
    setUserPreferences(prev => ({ ...prev, favoriteMovies: updatedMovies }));
    localStorage.setItem(`favoriteMovies_${user.uid}`, JSON.stringify(updatedMovies));
  };

  const sections = [
    {
      id: 'user-info',
      label: 'User Info',
      icon: '👤'
    },
    {
      id: 'personalization',
      label: 'Personalization',
      icon: '✨'
    }
  ];

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2 className="settings-title">Settings</h2>
          <button className="settings-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>
        
        <div className="settings-body">
          <div className="settings-sidebar">
            {sections.map(section => (
              <button
                key={section.id}
                className={`settings-nav-item ${activeSection === section.id ? 'active' : ''}`}
                onClick={() => setActiveSection(section.id)}
              >
                <span className="settings-nav-icon">{section.icon}</span>
                <span className="settings-nav-label">{section.label}</span>
              </button>
            ))}
          </div>
          
          <div className="settings-content">
            {activeSection === 'user-info' && (
              <div className="settings-section">
                <h3 className="section-title">User Information</h3>
                
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Name</span>
                    <span className="info-value">{user.username || user.displayName || 'User'}</span>
                  </div>
                  
                  <div className="info-item">
                    <span className="info-label">Email</span>
                    <span className="info-value">{user.email || auth.currentUser?.email || 'Not available'}</span>
                  </div>
                  
                  <div className="info-item">
                    <span className="info-label">User ID</span>
                    <span className="info-value">{user.uid}</span>
                  </div>
                  
                  <div className="info-item highlight">
                    <span className="info-label">Total Clothing Items</span>
                    <span className="info-value large">{imageCount}</span>
                  </div>
                </div>
                
                <div className="stats-section">
                  <h4 className="subsection-title">Wardrobe Statistics</h4>
                  <div className="stats-info">
                    <p>You have {imageCount} items in your digital wardrobe.</p>
                    <p className="stats-note">Keep adding more items to build your complete virtual closet!</p>
                  </div>
                </div>
              </div>
            )}
            
            {activeSection === 'personalization' && (
              <div className="settings-section">
                <h3 className="section-title">Personalization</h3>
                
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-icon">📍</span>
                    Location
                  </label>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Enter your city or region"
                    value={userPreferences.location}
                    onChange={handleLocationChange}
                  />
                  <p className="form-hint">Help us provide weather-appropriate outfit suggestions</p>
                </div>
                
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-icon">🏷️</span>
                    Favorite Brands
                  </label>
                  <div className="bubble-container">
                    {userPreferences.favoriteBrands.map((brand, index) => (
                      <div key={index} className="bubble-tag">
                        <span>{brand}</span>
                        <button 
                          className="bubble-remove-btn"
                          onClick={() => handleRemoveBrand(brand)}
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                    {isAddingBrand ? (
                      <div className="bubble-tag bubble-input-tag">
                        <input
                          type="text"
                          className="bubble-input"
                          placeholder="Type brand..."
                          value={newBrand}
                          onChange={(e) => setNewBrand(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleAddBrand();
                            if (e.key === 'Escape') {
                              setIsAddingBrand(false);
                              setNewBrand('');
                            }
                          }}
                          onBlur={() => {
                            if (newBrand.trim()) {
                              handleAddBrand();
                            } else {
                              setIsAddingBrand(false);
                              setNewBrand('');
                            }
                          }}
                          autoFocus
                        />
                      </div>
                    ) : (
                      <button 
                        className="bubble-tag bubble-add-tag"
                        onClick={() => setIsAddingBrand(true)}
                      >
                        <span>+ Add more</span>
                      </button>
                    )}
                  </div>
                </div>
                
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-icon">🎵</span>
                    Favorite Music Genres
                  </label>
                  <div className="bubble-container">
                    {userPreferences.favoriteGenres.map((genre, index) => (
                      <div key={index} className="bubble-tag">
                        <span>{genre}</span>
                        <button 
                          className="bubble-remove-btn"
                          onClick={() => handleRemoveGenre(genre)}
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                    {isAddingGenre ? (
                      <div className="bubble-tag bubble-input-tag">
                        <input
                          type="text"
                          className="bubble-input"
                          placeholder="Type music genre..."
                          value={newGenre}
                          onChange={(e) => setNewGenre(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleAddGenre();
                            if (e.key === 'Escape') {
                              setIsAddingGenre(false);
                              setNewGenre('');
                            }
                          }}
                          onBlur={() => {
                            if (newGenre.trim()) {
                              handleAddGenre();
                            } else {
                              setIsAddingGenre(false);
                              setNewGenre('');
                            }
                          }}
                          autoFocus
                        />
                      </div>
                    ) : (
                      <button 
                        className="bubble-tag bubble-add-tag"
                        onClick={() => setIsAddingGenre(true)}
                      >
                        <span>+ Add more</span>
                      </button>
                    )}
                  </div>
                  <p className="form-hint">Examples: hip-hop, rock, indie, electronic, jazz, K-pop</p>
                </div>
                
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-icon">🎬</span>
                    Favorite Movies
                  </label>
                  <div className="bubble-container">
                    {userPreferences.favoriteMovies.map((movie, index) => (
                      <div key={index} className="bubble-tag">
                        <span>{movie}</span>
                        <button 
                          className="bubble-remove-btn"
                          onClick={() => handleRemoveMovie(movie)}
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                    {isAddingMovie ? (
                      <div className="bubble-tag bubble-input-tag">
                        <input
                          type="text"
                          className="bubble-input"
                          placeholder="Type movie..."
                          value={newMovie}
                          onChange={(e) => setNewMovie(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleAddMovie();
                            if (e.key === 'Escape') {
                              setIsAddingMovie(false);
                              setNewMovie('');
                            }
                          }}
                          onBlur={() => {
                            if (newMovie.trim()) {
                              handleAddMovie();
                            } else {
                              setIsAddingMovie(false);
                              setNewMovie('');
                            }
                          }}
                          autoFocus
                        />
                      </div>
                    ) : (
                      <button 
                        className="bubble-tag bubble-add-tag"
                        onClick={() => setIsAddingMovie(true)}
                      >
                        <span>+ Add more</span>
                      </button>
                    )}
                  </div>
                  <p className="form-hint">Help us understand your style through your movie taste</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SettingsModal;