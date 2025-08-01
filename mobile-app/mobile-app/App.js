import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { auth, logoutUser } from './src/firebase';
import { onAuthStateChanged } from 'firebase/auth';
import { CacheService } from './src/services/cacheService';

// Import screens
import Login from './src/screens/Login';
import Home from './src/screens/Home';
import Wardrobe from './src/screens/Wardrobe';
import VirtualCloset from './src/screens/VirtualCloset';
import Closet from './src/screens/Closet';
import OpenAISwitch from './src/screens/OpenAISwitch';
import UploadSegment from './src/screens/UploadSegment';
import StyleAssistant from './src/screens/StyleAssistant';

const Stack = createStackNavigator();

export default function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Handle user login
  const handleLogin = (user) => {
    setCurrentUser(user);
  };

  // Handle logout
  const handleLogout = async () => {
    try {
      await logoutUser();
      setCurrentUser(null);
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
        
        // Start a new session for smart sync
        CacheService.startNewSession(user.uid).then(sessionId => {
          console.log('[App] Started new session:', sessionId);
        });
      } else {
        // User is signed out
        setCurrentUser(null);
      }
      setIsLoading(false);
    });

    // Cleanup subscription
    return () => unsubscribe();
  }, []);

  if (isLoading) {
    return null; // Or a loading screen
  }

  return (
    <>
      <StatusBar style="light" />
      <NavigationContainer>
        <Stack.Navigator
          screenOptions={{
            headerShown: false, // Hide ALL headers
          }}
        >
          {!currentUser ? (
            <Stack.Screen name="Login">
              {props => <Login {...props} onLogin={handleLogin} />}
            </Stack.Screen>
          ) : (
            <>
              <Stack.Screen name="Home">
                {props => <Home {...props} user={currentUser} onLogout={handleLogout} />}
              </Stack.Screen>
              <Stack.Screen name="Wardrobe">
                {props => <Wardrobe {...props} user={currentUser} />}
              </Stack.Screen>
              <Stack.Screen name="Closet">
                {props => <Closet {...props} user={currentUser} />}
              </Stack.Screen>
              <Stack.Screen name="VirtualCloset">
                {props => <VirtualCloset {...props} user={currentUser} />}
              </Stack.Screen>
              <Stack.Screen name="OpenAISwitch">
                {props => <OpenAISwitch {...props} user={currentUser} />}
              </Stack.Screen>
              <Stack.Screen name="StyleAssistant">
                {props => <StyleAssistant {...props} user={currentUser} onLogout={handleLogout} />}
              </Stack.Screen>
              <Stack.Screen name="UploadSegment">
                {props => <UploadSegment {...props} user={currentUser} />}
              </Stack.Screen>
            </>
          )}
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}