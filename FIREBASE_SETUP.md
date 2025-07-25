# Firebase Setup Guide for PoshakAI

## Prerequisites
- Firebase project created (PoshakAI - Project ID: poshakai)
- Firebase CLI installed (`npm install -g firebase-tools`)

## 1. Download Service Account Key

1. Go to [Firebase Console](https://console.firebase.google.com/project/poshakai/settings/serviceaccounts/adminsdk)
2. Click "Generate New Private Key"
3. Save the downloaded file as `serviceAccountKey.json` in the project root directory
4. This file is gitignored and should NEVER be committed

## 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy from .env.example
cp .env.example .env
```

Update the `.env` file with your values:
```
# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS=./serviceAccountKey.json
FIREBASE_STORAGE_BUCKET=poshakai.appspot.com

# Other API keys as needed...
```

## 3. Set up Firebase Services

### Enable Required Services:
1. **Authentication**: Enable Email/Password authentication
2. **Firestore**: Create database in production mode
3. **Storage**: Enable Firebase Storage

### Security Rules

#### Firestore Rules:
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId}/{document=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

#### Storage Rules:
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Users can only access their own files
    match /users/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Shared garments are public read
    match /shared/garments/{fileName} {
      allow read: if true;
      allow write: if false;
    }
  }
}
```

## 4. Create Test Users

Since we're using production Firebase, create test users in the Firebase Console:

1. Go to Authentication → Users
2. Add users with these emails:
   - john.doe@poshakai.test
   - jane.smith@poshakai.test
   - test.user@poshakai.test
   - fashion.designer@poshakai.test
   - demo.account@poshakai.test

Set password: `password123` for all test users

## 5. Update Firebase Config

The app ID is still needed. Get it from:
1. Go to Project Settings → General
2. Under "Your apps", click "Add app" → Web
3. Register app with nickname "PoshakAI Web"
4. Copy the `appId` from the config
5. Update both `src/firebase.js` and `mobile-app/mobile-app/src/firebase.js` with the appId

## 6. Running the Application

```bash
# Install dependencies
npm install
cd backend && pip install -r requirements.txt && cd ..

# Run the application
./run-all.sh
```

## 7. Verify Setup

1. Check Flask logs for successful Firebase initialization
2. Try uploading an image from web or mobile
3. Check Firebase Console → Storage to see uploaded files
4. Check Firestore for mask data

## Storage Structure
```
Firebase Storage:
├── users/
│   ├── {userId}/
│   │   ├── images/       # User uploaded images (person-only)
│   │   └── masks/        # Generated masks and visualizations
│   │       └── {imageName}/
│   │           └── masks.json
└── shared/
    └── garments/         # Shared garment images
```

## Architecture
- **Flask Backend**: Handles all Firebase operations via Admin SDK
- **Web/Mobile Apps**: Send requests to Flask, display results
- **No Direct Firebase Access**: Clients don't directly access Firebase

## Troubleshooting

### "The specified bucket does not exist"
- Make sure Storage is enabled in Firebase Console
- Verify the bucket name in `.env` matches your project

### Authentication errors
- Ensure Email/Password auth is enabled
- Check that test users are created

### Permission denied
- Review and update security rules
- Make sure users are authenticated before accessing resources

### Images not showing
- Check Flask logs for upload errors
- Verify service account has proper permissions
- Ensure storage bucket name is correct