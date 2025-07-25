import { Platform } from 'react-native';

// API Configuration
// For React Native, we need to use the actual IP address instead of localhost
const API_URL = Platform.OS === 'ios' ? 'http://192.168.1.122:5001' : 'http://10.0.2.2:5001';

const config = {
  API_URL,
};

export default config;