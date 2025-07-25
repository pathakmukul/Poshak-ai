// API Configuration
const API_URL = process.env.NODE_ENV === 'production' 
  ? 'http://localhost:5001'  // Using local backend
  : 'http://localhost:5001';  // Always use local now!

export default API_URL;