// API Configuration
const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://kapdaai-backend-560568328203.us-central1.run.app'
  : 'https://kapdaai-backend-560568328203.us-central1.run.app'; // Temporarily use Cloud Run for testing

export default API_URL;