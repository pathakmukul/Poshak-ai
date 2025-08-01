# Poshaak AI - AI-Powered Digital Wardrobe & Fashion Assistant

An intelligent wardrobe management system that combines computer vision, AI-powered virtual try-on, and personalized fashion recommendations powered by multiple AI services including QLOO's Taste AI.

## 🎯 Key Features

### Digital Wardrobe Management
- **Automatic Clothing Detection**: Uses Segformer B2 Clothes model for precise segmentation
- **Smart Categorization**: Automatically classifies shirts, pants, shoes, dresses, and more
- **Multi-View Experience**: 
  - Wardrobe View: Full outfit photos
  - Closet View: Individual clothing items with transparent backgrounds
  - Virtual Closet: Saved virtual try-on combinations

### AI-Powered Virtual Try-On
- **Multi-Item Try-On**: Combine shirts, pants, and shoes simultaneously
- **Powered by Gemini 2.0**: High-quality virtual outfit visualization
- **Mix & Match**: Experiment with different combinations from your wardrobe

### Intelligent Style Assistant
- **LangChain + Vertex AI Agent**: Natural language fashion advice
- **Context-Aware Recommendations**: 
  - Weather-based suggestions
  - Location-specific trends via QLOO API
  - Occasion-appropriate outfit selection from your wardrobe
- **Shopping Suggestions**: Curated recommendations from trending brands

### Fashion Intelligence Moodboard
- **QLOO-Powered Trends**: Real-time fashion trends for your location
- **Smart Wardrobe Analysis**: AI identifies gaps in your collection
- **Personalized Shopping**: Product recommendations based on:
  - Current trends in your area
  - Your existing wardrobe
  - Style preferences
- **Travel Mode**: Location-specific fashion insights for trips

## 🛠️ Tech Stack

### Frontend
- React.js with modern hooks
- Firebase Authentication & Storage
- Responsive CSS design
- React Native mobile app

### Backend
- Flask API server
- Segformer B2 (MIT licensed) for clothing detection
- Multiple AI integrations:
  - Google Gemini for virtual try-on
  - LangChain + Vertex AI for style assistance
  - QLOO Taste AI for fashion trends
- Firebase Admin SDK

### AI Services
- **QLOO API**: Fashion trends, brand affinities, location-based insights
- **Gemini 2.0**: Virtual try-on and fashion advice
- **Vertex AI**: Powers the conversational style assistant
- **Serper API**: Real-time shopping recommendations
- **Hugging Face**: Optional cloud inference for Segformer

## 🚀 Getting Started

### Prerequisites
- Node.js 14+
- Python 3.8+
- Firebase project
- API Keys for: QLOO, Google (Gemini), Serper

### Installation

1. Clone the repository
```bash
git clone https://github.com/pathakmukul/Poshak-ai.git
cd Poshak-ai
```

2. Install dependencies
```bash
# Frontend
npm install

# Backend
cd backend
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create .env file in root
REACT_APP_FIREBASE_API_KEY=your_firebase_key
# ... other Firebase config

# Create backend/.env
GOOGLE_API_KEY=your_gemini_key
QLOO_API_KEY=your_qloo_key
SERPER_API_KEY=your_serper_key
# ... other backend config
```

4. Run the application
```bash
# Use the provided script
./run-all.sh

# Or run separately
# Terminal 1 - Backend
cd backend && python flask_api.py

# Terminal 2 - Frontend
npm start
```

## 📱 Mobile App

A React Native app is included in the `mobile-app` directory with:
- Local caching for instant loading
- Offline support
- Cross-platform compatibility (iOS & Android)

## 🏗️ Architecture

- **Unified Backend**: Flask serves as the API gateway for all operations
- **Smart Processing**: Images resized automatically for efficient processing
- **Caching Strategy**: Multi-level caching for optimal performance
- **Cross-Platform**: Web and mobile apps share the same backend

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Segformer B2 Clothes model by mattmdjaga
- QLOO for fashion intelligence API
- Google Gemini for virtual try-on capabilities
- All open-source libraries used in this project