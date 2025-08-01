import { initializeApp } from 'firebase/app';
import { getAuth, initializeAuth, getReactNativePersistence, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from 'firebase/auth';
import { getStorage } from 'firebase/storage';
import { getFirestore } from 'firebase/firestore';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Firebase configuration for PoshakAI production
const firebaseConfig = {
  apiKey: "AIzaSyCsKGTED7HNVO35Ky0ms5O49w4KRmJlw7Y",
  authDomain: "poshakai.firebaseapp.com",
  projectId: "poshakai",
  storageBucket: "poshakai.appspot.com",
  messagingSenderId: "560568328203",
  appId: "1:560568328203:web:YOUR_APP_ID" // You'll need to get this from Firebase Console
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Auth with persistence for React Native
export const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage)
});

// Initialize Storage
export const storage = getStorage(app);

// Initialize Firestore
export const db = getFirestore(app);

// Test users configuration
export const DUMMY_USERS = [
  { email: 'john.doe@poshakai.test', password: 'password123', displayName: 'John Doe' },
  { email: 'jane.smith@poshakai.test', password: 'password123', displayName: 'Jane Smith' },
  { email: 'test.user@poshakai.test', password: 'password123', displayName: 'Test User' },
  { email: 'fashion.designer@poshakai.test', password: 'password123', displayName: 'Fashion Designer' },
  { email: 'demo.account@poshakai.test', password: 'password123', displayName: 'Demo Account' }
];

// REMOVED: initializeDummyUsers function to prevent duplicate user creation
// Use existing users from Firebase Console instead

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