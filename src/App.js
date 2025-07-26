import React, { useState, useEffect } from 'react';
import './App.css';
import Login from './Login';
import Wardrobe from './Wardrobe';
import Closet from './Closet';
import VirtualCloset from './VirtualCloset';
import OpenAISwitch from './OpenAISwitch';
import { auth, logoutUser } from './firebase';
import { onAuthStateChanged } from 'firebase/auth';

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [currentView, setCurrentView] = useState('main'); // 'main', 'wardrobe', 'closet', 'virtualcloset', or 'openai'

  // Handle user login
  const handleLogin = (user) => {
    setCurrentUser(user);
  };

  // Handle logout
  const handleLogout = async () => {
    try {
      await logoutUser();
      setCurrentUser(null);
      setCurrentView('main');
    } catch (error) {
      console.error('Logout error:', error);
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

  // Show login if no user
  if (!currentUser) {
    return <Login onLogin={handleLogin} />;
  }

  // Show wardrobe view
  if (currentView === 'wardrobe') {
    return <Wardrobe user={currentUser} onBack={() => setCurrentView('main')} />;
  }

  // Show closet view
  if (currentView === 'closet') {
    return <Closet user={currentUser} onBack={() => setCurrentView('main')} />;
  }

  // Show virtual closet view
  if (currentView === 'virtualcloset') {
    return <VirtualCloset user={currentUser} onBack={() => setCurrentView('main')} />;
  }

  // Show OpenAI switch view
  if (currentView === 'openai') {
    return <OpenAISwitch user={currentUser} onBack={() => setCurrentView('main')} />;
  }

  // Main view - clean homepage
  return (
    <div className="App">
      {/* Navigation Header */}
      <header className="app-header">
        <nav className="nav-bar">
          <div className="nav-left">
            <h1 className="app-title">KapdaAI</h1>
          </div>
          <div className="nav-center">
            <button 
              className={`nav-button ${currentView === 'main' ? 'active' : ''}`}
              onClick={() => setCurrentView('main')}
            >
              Home
            </button>
            <button 
              className={`nav-button ${currentView === 'wardrobe' ? 'active' : ''}`}
              onClick={() => setCurrentView('wardrobe')}
            >
              Wardrobe
            </button>
            <button 
              className={`nav-button ${currentView === 'closet' ? 'active' : ''}`}
              onClick={() => setCurrentView('closet')}
            >
              Closet
            </button>
            <button 
              className={`nav-button ${currentView === 'virtualcloset' ? 'active' : ''}`}
              onClick={() => setCurrentView('virtualcloset')}
            >
              Virtual Closet
            </button>
          </div>
          <div className="nav-right">
            <span className="user-name">👤 {currentUser.username}</span>
            <button className="logout-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="welcome-section">
          <h2>Welcome to KapdaAI</h2>
          <p>Your AI-powered virtual wardrobe assistant</p>
          
          <div className="action-cards">
            <div className="action-card" onClick={() => setCurrentView('wardrobe')}>
              <div className="card-icon">📸</div>
              <h3>Wardrobe</h3>
              <p>Upload photos and try on virtual outfits</p>
              <button className="card-button">Open Wardrobe →</button>
            </div>
            
            <div className="action-card" onClick={() => setCurrentView('closet')}>
              <div className="card-icon">👗</div>
              <h3>My Closet</h3>
              <p>View your collection of clothing items</p>
              <button className="card-button">Open Closet →</button>
            </div>
            
            <div className="action-card" onClick={() => setCurrentView('virtualcloset')}>
              <div className="card-icon">VC</div>
              <h3>Virtual Closet</h3>
              <p>View your saved try-on results</p>
              <button className="card-button">View Collection →</button>
            </div>
            
            <div className="action-card" onClick={() => setCurrentView('openai')}>
              <div className="card-icon">🎨</div>
              <h3>OpenAI Switch</h3>
              <p>Generate outfits with AI-powered variations</p>
              <button className="card-button">Try Now →</button>
            </div>
            
            <div className="action-card coming-soon">
              <div className="card-icon">✨</div>
              <h3>Style Assistant</h3>
              <p>Get AI-powered outfit recommendations</p>
              <span className="coming-soon-badge">Coming Soon</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;