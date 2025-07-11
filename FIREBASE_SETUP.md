# Firebase Local Setup for KapdaAI

## Overview
KapdaAI now uses Firebase Local Emulator Suite for user authentication and file storage. This provides a seamless local development experience that can easily migrate to production Firebase.

## Features
- **User Authentication**: Each user has their own account with email/password
- **User-Specific Storage**: Images are stored per user in Firebase Storage
- **Shared Resources**: Garment images are shared across all users
- **Local Development**: Everything runs locally using Firebase emulators

## Dummy Users
The following test accounts are pre-configured:
- john.doe@kapdaai.local (password: password123)
- jane.smith@kapdaai.local (password: password123)
- test.user@kapdaai.local (password: password123)
- fashion.designer@kapdaai.local (password: password123)
- demo.account@kapdaai.local (password: password123)

## Running the Application

### Option 1: Run Everything Together
```bash
./run-all.sh
```
This starts:
- Firebase Auth Emulator (port 9099)
- Firebase Storage Emulator (port 9199)
- Firebase Emulator UI (port 4000)
- Flask Backend (port 5001)
- React Frontend (port 3000)

### Option 2: Run Services Separately

1. Start Firebase Emulators:
```bash
./start-firebase.sh
```

2. Start Flask Backend:
```bash
source /Users/mukulpathak/PROJECTS/KapdaAI/kapdaai/bin/activate
python flask_api.py
```

3. Start React Frontend:
```bash
npm start
```

## Storage Structure
```
Firebase Storage:
├── users/
│   ├── {userId}/
│   │   ├── images/       # User uploaded images
│   │   └── masks/        # Generated masks
│   │       └── {imageName}/
│   │           ├── masks.json
│   │           ├── mask_shirt.png
│   │           ├── mask_pants.png
│   │           └── mask_shoes.png
└── shared/
    └── garments/         # Shared garment images
```

## Key Files
- `src/firebase.js` - Firebase configuration and auth functions
- `src/storageService.js` - Firebase Storage operations
- `src/FileUpload.js` - Image upload component
- `flask_api.py` - Updated to handle Firebase Storage URLs

## Future Migration
To migrate to production Firebase:
1. Create a Firebase project at https://console.firebase.google.com
2. Update `src/firebase.js` with real Firebase config
3. Remove emulator connection code
4. Deploy security rules
5. Migrate data if needed

## Troubleshooting
- If emulators fail to start, ensure ports 9099, 9199, and 4000 are free
- Clear browser localStorage/sessionStorage if auth issues occur
- Check Firebase Emulator UI at http://localhost:4000 for debugging