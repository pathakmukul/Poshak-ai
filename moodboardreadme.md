# MoodBoard Feature

## Overview
The MoodBoard is an AI-powered fashion trends dashboard that provides personalized fashion insights based on your location, current trends, and wardrobe inventory. It combines multiple data sources to create a comprehensive fashion intelligence platform.

## Cards & Data Sources

### 1. **Trending in [Location]** Card
- **Purpose**: Shows current fashion trends specific to your location
- **Data Source**: QLOO API (real-time fashion trends by location)
- **Data Flow**: 
  - Fetches trends based on Home/Trip location
  - Falls back to curated data if API unavailable
  - Updates on "Run" button click
- **Display**: Fashion item pills (e.g., "oversized blazers", "wide-leg pants")

### 2. **Hot Brands** Card
- **Purpose**: Displays trending fashion brands with popularity scores
- **Data Source**: QLOO API (brand affinity data)
- **Data Flow**:
  - Retrieves top brands with affinity scores (0-100%)
  - Sorted by popularity in your location
  - Cached with other trend data
- **Display**: Brand names with percentage scores

### 3. **Right Now** Card
- **Purpose**: Shows seasonal and time-based fashion context
- **Data Source**: Calculated from current date/time + QLOO trends
- **Display**: Season vibes (e.g., "summer vibes") and time context (e.g., "perfect for evening")

### 4. **Recommended for You** Card
- **Purpose**: Personalized shopping recommendations
- **Data Source**: 
  - Primary: Serper Shopping API
  - Trends: QLOO brands for targeting
  - Query: Combines trending items + brands + location
- **Features**:
  - Brand diversity (max 2 items per brand)
  - Hover shows brand + price
  - Click opens detailed product popup
  - Links to actual purchase pages

### 5. **[Location] Style** Card
- **Purpose**: Location-specific style insights
- **Data Source**: Calculated based on wardrobe alignment with local trends
- **Display**: 
  - Home mode: "Your wardrobe is X% aligned with local trends"
  - Trip mode: "Pack light with versatile pieces for [city]"

### 6. **You Need** Card
- **Purpose**: Intelligent wardrobe gap analysis
- **Data Source**:
  - Wardrobe: User's closet inventory from localStorage
  - AI: Gemini LLM for gap analysis
  - Products: Serper API for shopping suggestions
- **Process**:
  1. Analyzes your wardrobe categories (shirts, pants, shoes)
  2. Compares with current QLOO trends
  3. Identifies 3 missing items
  4. Fetches actual products from trending brands
- **Display**: Analysis text + product recommendations

## Core Features

### Run Button
- Fetches fresh data from all APIs
- Updates all cards simultaneously
- Shows loading states during fetch
- Caches results in localStorage

### Save Button
- Saves current moodboard state to Firebase Storage
- Includes all card data + timestamps
- Creates timestamped backups
- Enables cross-device access

### Home/Trip Toggle
- **Home Mode**: Uses saved home location
- **Trip Mode**: Allows custom city input
- Clears cache when switching modes
- All cards update based on selected location

## Data Flow & Caching

### Initial Load
1. Check localStorage for cached data
2. If no cache → Load from Firebase Storage
3. If no Firebase data → Empty state
4. Display cached/loaded data immediately

### Run Process
1. Fetch QLOO trends for location
2. Fetch Serper shopping recommendations
3. Analyze wardrobe gaps with Gemini
4. Fetch products for identified gaps
5. Update all state + localStorage cache

### API Calls per Run
- QLOO Trends API: 1 call
- Serper Shopping API: 2 calls (recommendations + gap products)
- Gemini API: 1 call
- Total: 4 API calls

### Caching Strategy
- localStorage key: `moodboard_cache_${userId}`
- Contains all card data + timestamps
- Persists between sessions
- Cleared on mode switch (Home ↔ Trip)

## Technical Implementation

### State Management
```javascript
- trends: QLOO trend data
- recommendations: Shopping products
- wardrobeGaps: Identified missing items
- wardrobeGapProducts: Products for gaps
- wardrobeAnalysis: Gemini's analysis text
- lastRunData: Complete state snapshot
```

### Key Components
- Absolute positioned cards for artistic layout
- Common product-item class (28% width for ~3.5 items visible)
- Hover overlays for product info
- Responsive design with percentage-based sizing

### Performance Optimizations
- Single Serper call for multiple items using OR queries
- Brand diversity limits (max 2 per brand)
- Lazy loading with cached data
- Debounced API calls

## Firebase Integration
- Storage path: `users/{userId}/moodboard/latest.json`
- Backup path: `users/{userId}/moodboard/backup_{timestamp}.json`
- No Firestore usage (project in Datastore mode)

## Future Enhancements
- Weather-based recommendations
- Outfit suggestions from wardrobe
- Price range filtering
- Brand preference learning