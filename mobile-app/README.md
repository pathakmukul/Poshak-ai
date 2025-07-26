# KapdaAI Mobile App

A React Native mobile application for KapdaAI - your AI-powered wardrobe assistant. Segment clothing from photos, build your digital closet, and manage your wardrobe on the go.

## ✨ Features

### Core Features
- **🤖 AI Clothing Segmentation**: Upload photos and automatically detect clothing items (shirts, pants, shoes)
- **👕 Digital Wardrobe**: View and manage your full-body clothing photos with multi-item virtual try-on
- **🗄️ Digital Closet**: Browse individual clothing items with smart categorization
- **👗 Virtual Try-On**: Try multiple garments at once using Gemini AI
- **📸 Virtual Closet**: Save and manage your favorite try-on results
- **🔐 User Authentication**: Secure Firebase-based login system

### Performance Features
- **⚡ Lightning-Fast Loading**: Local caching system for instant closet access
- **🔄 Smart Sync**: Intelligent count-based synchronization strategy
  - Only syncs when item counts differ from backend
  - Checks counts once per app session
  - No automatic periodic syncs
  - Pull-to-refresh for manual sync
- **📱 Offline Support**: Browse your wardrobe even without internet
- **💾 Intelligent Caching**: Stores clothing items locally for blazing-fast performance

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (Mac) or Android Emulator
- Backend Flask API running (see main project README)

### Installation

1. Navigate to mobile app directory:
```bash
cd mobile-app
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Configure Firebase (update `src/firebase.js`):
   - Use your Firebase project credentials
   - Ensure Firebase Storage is enabled

4. Configure API endpoint (update `src/config.js`):
```javascript
export default {
  API_URL: 'http://localhost:5001'  // Update with your backend URL
};
```

5. Start the development server:
```bash
npx expo start
# or
npm start
```

6. Run on your device:
   - **iOS**: Press `i` in terminal or scan QR with Expo Go app
   - **Android**: Press `a` in terminal or scan QR with Expo Go app
   - **Physical Device**: Ensure backend URL is accessible from device

## 📁 Project Structure

```
mobile-app/
├── App.js                           # Root component with navigation
├── src/
│   ├── screens/                     # App screens
│   │   ├── Home.js                 # Main dashboard
│   │   ├── Login.js                # Authentication
│   │   ├── Wardrobe.js             # Full person images gallery & virtual try-on
│   │   ├── Closet.js               # Individual clothing items
│   │   ├── VirtualCloset.js        # Saved virtual try-on results
│   │   ├── OpenAISwitch.js         # AI outfit generation
│   │   └── UploadSegmentModal.js   # Image upload & segmentation
│   ├── services/                    # Business logic
│   │   ├── storageService.js       # Firebase integration
│   │   ├── closetService.js        # Closet data management with smart sync
│   │   ├── virtualClosetService.js # Virtual closet data management
│   │   └── cacheService.js         # Local caching with session tracking
│   ├── components/                  # Reusable components
│   │   └── SmartCropImage.js       # Intelligent image display
│   ├── config.js                   # App configuration
│   └── firebase.js                 # Firebase setup
├── assets/                         # Images and icons
└── package.json                    # Dependencies
```

## 🎯 Key Features Implementation

### Image Segmentation Flow
1. **Upload**: Camera or gallery selection (no forced cropping)
2. **Process**: AI segments clothing using Segformer model
3. **Preview**: Shows original + 3 detected items (shirt, pants, shoes)
4. **Select**: Tap to deselect unwanted items
5. **Save**: Items instantly appear in closet

### Caching System
- **AsyncStorage**: Stores clothing items locally
- **Smart Sync Strategy**:
  - **Session-based**: New session starts on app open
  - **Count Check**: Lightweight API call checks item counts
  - **Differential Sync**: Only syncs when counts mismatch
  - **No Periodic Sync**: Sync only happens on app open or manual refresh
- **Instant Updates**: New items cached immediately after segmentation
- **Offline Mode**: Full functionality without internet

### Virtual Try-On System
- **Multi-Item Selection**: Select multiple garments (shirt, pants, shoes) to try on simultaneously
- **Visual Feedback**: Green checkmarks show selected items
- **Try-On Results**: View generated image with items used
- **Save Results**: Store favorite try-ons to Virtual Closet
- **Quick Actions**: Redo, back, and store buttons for seamless workflow

### Virtual Closet System
- **Local-First Storage**: Virtual try-on results saved to AsyncStorage instantly
- **Background Sync**: Automatic sync to Firebase Storage via Flask endpoints
- **Offline Support**: Browse saved try-ons without internet connection
- **Same Caching Strategy**: Uses identical smart sync as My Closet feature
- **Flask Integration**: Uses `/firebase/virtual-closet` endpoints for consistency

### UI/UX Highlights
- **Dark Theme**: Consistent dark mode design
- **Modal Experience**: Try-on interface in popup overlay
- **Fast Navigation**: Instant screen transitions
- **Pull to Refresh**: Manual sync option
- **Smart Image Display**: Content-aware with padding
- **Responsive Design**: Adaptive layouts for different screen sizes

## 🛠️ Development

### Running in Development
```bash
# Start Metro bundler
npx expo start

# Run on iOS
npx expo run:ios

# Run on Android
npx expo run:android
```

### Building for Production

**iOS**:
```bash
eas build -p ios
```

**Android**:
```bash
eas build -p android
```

## 🐛 Troubleshooting

### Common Issues

1. **Closet items not showing**
   - Clear AsyncStorage cache
   - Pull down to refresh
   - Check backend connection

2. **Slow initial load**
   - First load fetches from backend
   - Subsequent loads use cache
   - Check network speed

3. **Image upload fails**
   - Ensure image < 10MB
   - Check backend is running
   - Verify API URL in config

4. **Login issues**
   - Verify Firebase configuration
   - Check internet connection
   - Ensure user exists in Firebase

## 🔮 Future Enhancements

- [x] Virtual Closet for saving try-on results
- [x] Multi-item virtual try-on
- [ ] Outfit recommendations based on weather
- [ ] Social sharing features
- [ ] Advanced search and filters
- [ ] Outfit planning calendar
- [ ] Mix & match outfit creator
- [ ] Color coordination suggestions
- [ ] Wardrobe statistics
- [ ] Clothing care reminders

## 📄 License

This project is proprietary and confidential.

## 💬 Support

For issues or questions:
- Create an issue in the repository
- Contact the development team
- Check the main project documentation