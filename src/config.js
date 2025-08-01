// API Configuration
const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://us-central1-poshakai.cloudfunctions.net/backend'  // Firebase Functions URL
  : 'http://localhost:5001';  // Local development

export default API_URL;