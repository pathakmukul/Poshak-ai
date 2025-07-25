import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { auth, logoutUser } from './src/firebase';
import { onAuthStateChanged } from 'firebase/auth';

// Import screens
import Login from './src/screens/Login';
import Home from './src/screens/Home';
import Wardrobe from './src/screens/Wardrobe';
import Closet from './src/screens/Closet';
import OpenAISwitch from './src/screens/OpenAISwitch';

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
            headerStyle: {
              backgroundColor: '#2a2a2a',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          }}
        >
          {!currentUser ? (
            <Stack.Screen 
              name="Login" 
              options={{ headerShown: false }}
            >
              {props => <Login {...props} onLogin={handleLogin} />}
            </Stack.Screen>
          ) : (
            <>
              <Stack.Screen 
                name="Home" 
                options={{ title: 'KapdaAI' }}
              >
                {props => <Home {...props} user={currentUser} onLogout={handleLogout} />}
              </Stack.Screen>
              <Stack.Screen 
                name="Wardrobe" 
                options={{ title: 'My Wardrobe' }}
              >
                {props => <Wardrobe {...props} user={currentUser} />}
              </Stack.Screen>
              <Stack.Screen 
                name="Closet" 
                options={{ title: 'My Closet' }}
              >
                {props => <Closet {...props} user={currentUser} />}
              </Stack.Screen>
              <Stack.Screen 
                name="OpenAISwitch" 
                options={{ title: 'OpenAI Switch' }}
              >
                {props => <OpenAISwitch {...props} user={currentUser} />}
              </Stack.Screen>
            </>
          )}
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}