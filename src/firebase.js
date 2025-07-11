import { initializeApp } from 'firebase/app';
import { getAuth, connectAuthEmulator, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from 'firebase/auth';
import { getStorage, connectStorageEmulator } from 'firebase/storage';

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

// Initialize services
export const auth = getAuth(app);
export const storage = getStorage(app);

// Connect to emulators if in development
if (process.env.NODE_ENV === 'development') {
  // Check if emulators are already connected
  if (!auth.emulatorConfig) {
    connectAuthEmulator(auth, 'http://localhost:9099', { disableWarnings: true });
  }
  if (!storage._delegate?._host?.includes('localhost')) {
    connectStorageEmulator(storage, 'localhost', 9199);
  }
}

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