import React, { useState, useEffect } from 'react';
import './Navigation.css';
import SettingsModal from './SettingsModal';

function Navigation({ user, currentView, onViewChange, onLogout }) {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.user-menu')) {
        setUserDropdownOpen(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const navItems = [
    { id: 'main', label: 'Closet', icon: '🏠' },
    { id: 'wardrobe', label: 'Wardrobe', icon: '📸' },
    { id: 'virtualcloset', label: 'Virtual Closet', icon: '✨' },
    { id: 'styleassistant', label: 'Style Assistant', icon: '💬' },
    { id: 'moodboard', label: 'MoodBoard', icon: '🎨' }
  ];

  return (
    <>
      <nav className={`luxury-nav ${scrolled ? 'scrolled' : ''}`}>
        <div className="nav-container">
          <div className="nav-brand">
            <h1 className="brand-name">PoshakAI</h1>
            <span className="brand-tagline">Luxury Fashion AI</span>
          </div>

          <div className="nav-menu desktop-menu">
            {navItems.map(item => (
              <button
                key={item.id}
                className={`nav-item ${currentView === item.id ? 'active' : ''}`}
                onClick={() => {
                  onViewChange(item.id);
                  setMobileMenuOpen(false);
                }}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </button>
            ))}
          </div>

          <div className="nav-actions">
            <div className="user-menu">
              <button 
                className="user-info"
                onClick={(e) => {
                  e.stopPropagation();
                  setUserDropdownOpen(!userDropdownOpen);
                }}
              >
                <span className="user-avatar">👤</span>
                <span className="user-name">{user.username}</span>
                <span className="dropdown-arrow">▼</span>
              </button>
              
              {userDropdownOpen && (
                <div className="user-dropdown">
                  <button 
                    className="dropdown-item"
                    onClick={() => {
                      setUserDropdownOpen(false);
                      setSettingsModalOpen(true);
                    }}
                  >
                    <span className="dropdown-icon">⚙️</span>
                    <span>Settings</span>
                  </button>
                  <div className="dropdown-divider"></div>
                  <button 
                    className="dropdown-item logout-item"
                    onClick={() => {
                      setUserDropdownOpen(false);
                      onLogout();
                    }}
                  >
                    <span className="dropdown-icon">⎋</span>
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>

            <button 
              className="mobile-menu-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <span className={`menu-bar ${mobileMenuOpen ? 'open' : ''}`}></span>
              <span className={`menu-bar ${mobileMenuOpen ? 'open' : ''}`}></span>
              <span className={`menu-bar ${mobileMenuOpen ? 'open' : ''}`}></span>
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      <div className={`mobile-menu-overlay ${mobileMenuOpen ? 'open' : ''}`}>
        <div className="mobile-menu-content">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`mobile-nav-item ${currentView === item.id ? 'active' : ''}`}
              onClick={() => {
                onViewChange(item.id);
                setMobileMenuOpen(false);
              }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
          
          <div className="mobile-user-section">
            <div className="mobile-user-info">
              <span className="user-avatar">👤</span>
              <span className="user-name">{user.username}</span>
            </div>
            <button className="mobile-logout-btn" onClick={() => {
              onLogout();
              setMobileMenuOpen(false);
            }}>
              Logout
            </button>
          </div>
        </div>
      </div>
      
      {settingsModalOpen && (
        <SettingsModal 
          user={user}
          onClose={() => setSettingsModalOpen(false)}
        />
      )}
    </>
  );
}

export default Navigation;