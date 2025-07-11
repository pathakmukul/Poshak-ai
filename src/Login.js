import React, { useState, useEffect } from 'react';
import './Login.css';
import { loginUser, DUMMY_USERS, initializeDummyUsers } from './firebase';

function Login({ onLogin }) {
  const [selectedUser, setSelectedUser] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Initialize dummy users on component mount
  useEffect(() => {
    initializeDummyUsers().catch(console.error);
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

      // Login with Firebase
      const user = await loginUser(userConfig.email, userConfig.password);
      
      // Pass user info to parent
      onLogin({
        uid: user.uid,
        email: user.email,
        displayName: userConfig.displayName,
        username: userConfig.displayName
      });
    } catch (err) {
      console.error('Login error:', err);
      if (err.code === 'auth/user-not-found') {
        setError('User not found. Initializing users...');
        await initializeDummyUsers();
        setError('Please try again');
      } else {
        setError('Login failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Get display names from Firebase config
  const dummyUsers = DUMMY_USERS.map(user => user.displayName);

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-header">
          <img src="/images/logo.png" alt="KapdaAI" className="login-logo" />
          <h1>Welcome to KapdaAI</h1>
          <p>AI-Powered Clothing Detection & Virtual Try-On</p>
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