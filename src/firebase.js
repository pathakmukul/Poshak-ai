import { initializeApp } from 'firebase/app';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from 'firebase/auth';
import { getStorage } from 'firebase/storage';
import { getFirestore } from 'firebase/firestore';

// Firebase configuration for PoshakAI production
const firebaseConfig = {
  apiKey: "AIzaSyCsKGTED7HNVO35Ky0ms5O49w4KRmJlw7Y",
  authDomain: "poshakai.firebaseapp.com",
  projectId: "poshakai",
  storageBucket: "poshakai.appspot.com",
  messagingSenderId: "560568328203",
  appId: "1:560568328203:web:c5c4d5e6f7g8h9i0" // Replace with actual app ID from Firebase Console
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize services
export const auth = getAuth(app);
export const storage = getStorage(app);
export const db = getFirestore(app);

// Test users configuration
export const DUMMY_USERS = [
  { email: 'john.doe@poshakai.test', password: 'password123', displayName: 'John Doe' },
  { email: 'jane.smith@poshakai.test', password: 'password123', displayName: 'Jane Smith' },
  { email: 'test.user@poshakai.test', password: 'password123', displayName: 'Test User' },
  { email: 'fashion.designer@poshakai.test', password: 'password123', displayName: 'Fashion Designer' },
  { email: 'demo.account@poshakai.test', password: 'password123', displayName: 'Demo Account' }
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