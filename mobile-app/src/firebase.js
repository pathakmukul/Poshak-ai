import { initializeApp } from 'firebase/app';
import { getAuth, initializeAuth, getReactNativePersistence, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from 'firebase/auth';
import { getStorage } from 'firebase/storage';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Firebase configuration for local development
const firebaseConfig = {
  apiKey: "fake-api-key",
  authDomain: "localhost",
  projectId: "kapdaai-local",
  storageBucket: "kapdaai-local.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abcdef123456"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Auth with persistence for React Native
export const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage)
});

// Initialize Storage
export const storage = getStorage(app);

// Note: For React Native, we'll connect to the emulator using the host machine's IP
// This will need to be configured based on the development environment
export const EMULATOR_HOST = '10.0.2.2'; // For Android emulator
// export const EMULATOR_HOST = 'localhost'; // For iOS simulator

// Dummy users configuration
export const DUMMY_USERS = [
  { email: 'john.doe@kapdaai.local', password: 'password123', displayName: 'John Doe' },
  { email: 'jane.smith@kapdaai.local', password: 'password123', displayName: 'Jane Smith' },
  { email: 'test.user@kapdaai.local', password: 'password123', displayName: 'Test User' },
  { email: 'fashion.designer@kapdaai.local', password: 'password123', displayName: 'Fashion Designer' },
  { email: 'demo.account@kapdaai.local', password: 'password123', displayName: 'Demo Account' }
];

// Helper function to create dummy users in emulator
export const initializeDummyUsers = async () => {
  for (const user of DUMMY_USERS) {
    try {
      await createUserWithEmailAndPassword(auth, user.email, user.password);
      console.log(`Created user: ${user.displayName}`);
    } catch (error) {
      if (error.code !== 'auth/email-already-in-use') {
        console.error(`Error creating user ${user.displayName}:`, error);
      }
    }
  }
};

// Auth functions
export const loginUser = async (email, password) => {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    return userCredential.user;
  } catch (error) {
    throw error;
  }
};

export const logoutUser = async () => {
  try {
    await signOut(auth);
  } catch (error) {
    throw error;
  }
};

export default app;