import React, { useState, useEffect } from 'react';
import './globalStyles.css';
import './Login.css';
import { loginUser, DUMMY_USERS } from './firebase';

function Login({ onLogin }) {
  const [selectedUser, setSelectedUser] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // REMOVED: Auto-creation of dummy users to prevent duplicates
  useEffect(() => {
    // Users should already exist in Firebase
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedUser) {
      setError('Please select a user');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Find the selected user's email
      const userConfig = DUMMY_USERS.find(u => u.displayName === selectedUser);
      if (!userConfig) {
        throw new Error('User configuration not found');
      }

      // Pass user info to parent immediately (optimistic UI)
      onLogin({
        uid: userConfig.email, // Temporary ID until Firebase responds
        email: userConfig.email,
        displayName: userConfig.displayName,
        username: userConfig.displayName,
        isAuthenticating: true // Flag to show we're still authenticating
      });
      
      // Login with Firebase in background
      loginUser(userConfig.email, userConfig.password).then(user => {
        // Update with real user data once authenticated
        onLogin({
          uid: user.uid,
          email: user.email,
          displayName: userConfig.displayName,
          username: userConfig.displayName,
          isAuthenticating: false
        });
      }).catch(err => {
        console.error('Background auth failed:', err);
        // Could show a subtle notification here
      });
    } catch (err) {
      console.error('Login error:', err);
      if (err.code === 'auth/user-not-found') {
        setError('User not found. Please ensure test users exist in Firebase Console.');
      } else {
        setError('Login failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Get display names from Firebase config
  // const dummyUsers = DUMMY_USERS.map(user => user.displayName);
  // Only show Demo Account
  const dummyUsers = ['Demo Account'];

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-header">
          <img src="/images/logo.png" alt="PoshakAI" className="login-logo" />
          <h1>PoshakAI</h1>
          <p>AI-Powered Clothing Detection, Virtual Try-On & Recommendations</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Select User Account</label>
            <select
              id="username"
              value={selectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
              disabled={loading}
              className="user-select"
            >
              <option value="">-- Choose a user --</option>
              {dummyUsers.map(user => (
                <option key={user} value={user}>{user}</option>
              ))}
            </select>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <button 
            type="submit" 
            disabled={loading || !selectedUser}
            className="login-button"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="login-footer">
          <p>This is a demo application with pre-configured accounts.</p>
          <p>No password required for testing.</p>
        </div>
      </div>
    </div>
  );
}

export default Login;