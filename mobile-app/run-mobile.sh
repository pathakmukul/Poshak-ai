#!/bin/bash

echo "🚀 Starting KapdaAI Mobile App..."
echo ""

# Navigate to the correct Expo project directory
cd mobile-app

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install --force
    echo ""
fi

# Install additional dependencies if missing
echo "📦 Installing additional React Native dependencies..."
npm install @react-native-async-storage/async-storage @react-native-picker/picker --force

# Start the Expo development server
echo ""
echo "🎉 Starting Expo development server..."
echo "---------------------------------------------"
echo "Press 'i' for iOS simulator"
echo "Press 'a' for Android emulator"
echo "Press 'w' for web browser"
echo "Scan QR code with Expo Go app on your phone"
echo "---------------------------------------------"
echo ""

npm start