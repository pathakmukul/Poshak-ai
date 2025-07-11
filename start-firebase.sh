#!/bin/bash

echo "Starting Firebase Emulators..."
echo "================================"
echo ""
echo "Auth Emulator: http://localhost:9099"
echo "Storage Emulator: http://localhost:9199" 
echo "Emulator UI: http://localhost:4000"
echo ""
echo "================================"

# Start emulators
firebase emulators:start --only auth,storage