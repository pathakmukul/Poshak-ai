# 🚀 Quick Start Guide for KapdaAI Mobile App

## To Run the App:

### Option 1: Use the run script (Recommended)
```bash
cd /Users/mukulpathak/PROJECTS/KapdaAI/react-app/mobile-app
./run-mobile.sh
```

### Option 2: Manual start
```bash
cd /Users/mukulpathak/PROJECTS/KapdaAI/react-app/mobile-app/mobile-app
npm start
```

## After Starting:
- Press `i` to open in iOS Simulator
- Press `a` to open in Android Emulator  
- Press `w` to open in web browser
- Or scan the QR code with Expo Go app on your phone

## Important Notes:
1. Make sure the Firebase emulator is running (from the main react-app directory)
2. Make sure the Flask backend is running on port 5001
3. For physical devices, update the API URL in `src/config.js` with your machine's IP

## Troubleshooting:
- If you see dependency errors, run: `npm install --force`
- If Metro bundler has issues, clear cache: `npx expo start --clear`
- For iOS simulator issues: Open Xcode > Preferences > Locations and ensure Command Line Tools is set

The app is now ready to run! 🎉