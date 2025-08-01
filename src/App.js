import React, { useState, useEffect } from 'react';
import './globalStyles.css';
import './App.css';
import Navigation from './components/Navigation';
import LuxuryClosetHome from './components/LuxuryClosetHome';
import Login from './Login';
import Wardrobe from './Wardrobe';
import Closet from './Closet';
import VirtualCloset from './VirtualCloset';
import OpenAISwitch from './OpenAISwitch';
import StyleAssistant from './StyleAssistant';
import MoodBoard from './MoodBoard';
import { auth, logoutUser } from './firebase';
import { onAuthStateChanged } from 'firebase/auth';
import { preloadUserData } from './dataPreloader';

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [currentView, setCurrentView] = useState('main'); // 'main', 'wardrobe', 'closet', 'virtualcloset', 'openai', 'styleassistant', or 'moodboard'

  // Handle user login
  const handleLogin = (user) => {
    setCurrentUser(user);
    
    // Start preloading data if we have the real uid
    if (user.uid && !user.isAuthenticating) {
      preloadUserData(user.uid);
    }
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
        const userData = {
          uid: user.uid,
          email: user.email,
          displayName: user.displayName || user.email.split('@')[0],
          username: user.displayName || user.email.split('@')[0]
        };
        setCurrentUser(userData);
        
        // Start preloading data in the background
        preloadUserData(user.uid);
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

  // Render different views based on currentView
  const renderView = () => {
    switch (currentView) {
      case 'wardrobe':
        return <Wardrobe user={currentUser} onBack={() => setCurrentView('main')} />;
      case 'closet':
        return <Closet user={currentUser} onBack={() => setCurrentView('main')} />;
      case 'virtualcloset':
        return <VirtualCloset user={currentUser} onBack={() => setCurrentView('main')} />;
      case 'openai':
        return <OpenAISwitch user={currentUser} onBack={() => setCurrentView('main')} />;
      case 'styleassistant':
        return <StyleAssistant user={currentUser} onBack={() => setCurrentView('main')} />;
      case 'moodboard':
        return <MoodBoard user={currentUser} onBack={() => setCurrentView('main')} />;
      default:
        return renderMainView();
    }
  };

  // Main view content - now shows luxury closet
  const renderMainView = () => (
    <LuxuryClosetHome user={currentUser} />
  );

  // Main app render
  return (
    <div className="page-container">
      <Navigation 
        user={currentUser}
        currentView={currentView}
        onViewChange={setCurrentView}
        onLogout={handleLogout}
      />
      {renderView()}
    </div>
  );
}

export default App;