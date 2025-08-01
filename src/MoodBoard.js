import React, { useState, useEffect } from 'react';
import './MoodBoard.css';

function MoodBoard({ user, onBack }) {
  const [viewMode, setViewMode] = useState('home');
  const [tripCity, setTripCity] = useState('');
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(true);
  const [saving, setSaving] = useState(false);
  const [trends, setTrends] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [lastRunData, setLastRunData] = useState(null);
  const [hasInitialized, setHasInitialized] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [wardrobeGaps, setWardrobeGaps] = useState([]);
  const [wardrobeGapProducts, setWardrobeGapProducts] = useState([]);
  const [loadingGapProducts, setLoadingGapProducts] = useState(false);
  const [wardrobeAnalysis, setWardrobeAnalysis] = useState('');
  const [tribePicks, setTribePicks] = useState(null);
  const [loadingTribePicks, setLoadingTribePicks] = useState(false);
  
  // Get home location from localStorage
  const homeLocation = localStorage.getItem(`userLocation_${user.uid}`) || 'New York';
  
  // Hardcoded Style Twin wardrobe items
  const getTwinWardrobeItems = () => {
    return [
      {
        id: 1,
        name: 'Vintage Denim Jacket',
        image: '/twins/52DUOuEa6p5JC7jL-gB6t.jpeg',
        price: '$85',
        brand: 'Thrifted Find',
        twinCount: 3,
        description: 'Classic 90s oversized denim jacket with distressed details. Style twins pair it with black turtlenecks and wide-leg trousers for a timeless look.',
        category: 'jacket',
        affinity_score: 0.92
      },
      {
        id: 2,
        name: 'Minimalist White Sneakers',
        image: '/twins/white_jeans_shoes.png',
        price: '$120',
        brand: 'Common Projects',
        twinCount: 5,
        description: 'Clean white leather sneakers that go with everything. Your style twins wear these with both casual jeans and tailored pants.',
        category: 'shoes',
        affinity_score: 0.88
      },
      {
        id: 3,
        name: 'Olive Green Utility Pants',
        image: '/twins/olives.png',
        price: '$95',
        brand: 'Carhartt WIP',
        twinCount: 4,
        description: 'Versatile cargo pants in olive green. Style twins love these for both streetwear and smart-casual looks.',
        category: 'pants',
        affinity_score: 0.85
      },
      {
        id: 4,
        name: 'Striped Breton Shirt',
        image: '/twins/btECU7IiA90NNnrC7VJlc.jpeg',
        price: '$65',
        brand: 'Saint James',
        twinCount: 7,
        description: 'Classic French marinière in navy and white stripes. A wardrobe staple that your style twins pair with everything from jeans to blazers.',
        category: 'shirt',
        affinity_score: 0.90
      }
    ];
  };

  // Analyze wardrobe gaps using Gemini
  const analyzeWardrobeGaps = async (trendsData, runData = null) => {
    try {
      console.log('[MoodBoard] analyzeWardrobeGaps called with trends:', trendsData ? 'yes' : 'no');
      
      // Get wardrobe data from the correct key
      const cachedWardrobe = localStorage.getItem(`closet_items_${user.uid}`);
      
      if (!cachedWardrobe || !trendsData) {
        console.log('[MoodBoard] No wardrobe data or trends available for analysis');
        console.log('[MoodBoard] Looking for key:', `closet_items_${user.uid}`);
        console.log('[MoodBoard] Available localStorage keys:', Object.keys(localStorage).filter(k => k.includes(user.uid)));
        return;
      }
      
      const wardrobeData = JSON.parse(cachedWardrobe);
      const currentLocation = viewMode === 'home' ? homeLocation : tripCity || homeLocation;
      
      console.log('[MoodBoard] Analyzing wardrobe gaps with Gemini...');
      console.log('[MoodBoard] Wardrobe data structure:', Object.keys(wardrobeData));
      
      // Create wardrobe inventory string matching the vertex agent format
      let wardrobeInventory = "User's wardrobe inventory:\n";
      let itemIndex = 1;
      let totalItems = 0;
      
      // Categories from vertex agent: shirts, pants, shoes, etc.
      Object.entries(wardrobeData).forEach(([category, items]) => {
        if (Array.isArray(items) && items.length > 0) {
          wardrobeInventory += `\n${category.toUpperCase()} (${items.length} items):\n`;
          items.forEach(item => {
            // Use description field like vertex agent
            const description = item.description || item.display_name || item.name || `${category} item`;
            wardrobeInventory += `  ${itemIndex}. ${description}\n`;
            itemIndex++;
            totalItems++;
          });
        }
      });
      
      console.log(`[MoodBoard] Total wardrobe items: ${totalItems}`);
      
      const prompt = `Based on this wardrobe inventory and current fashion context, analyze gaps and suggest items.

${wardrobeInventory}

Current Fashion Context from QLOO API:
- Location: ${currentLocation}
- Season: ${trendsData.season || 'current season'}
- Trending items in ${currentLocation}: ${trendsData.trending_items?.join(', ') || 'casual wear'}
- Hot brands locally: ${trendsData.trending_brands?.map(b => b.name).join(', ') || 'various'}
- Time context: ${trendsData.time_context?.time_of_day || 'day'} wear

IMPORTANT: Only suggest CLOTHING items from these categories:
- Shirts (t-shirts, button-ups, blouses, tops)
- Pants (jeans, trousers, chinos, leggings)
- Shoes (sneakers, boots, loafers, heels)

DO NOT suggest accessories like sunglasses, bags, belts, watches, jewelry, etc.

Return a JSON response with this EXACT format:
{
  "analysis": "2-3 line explanation of what's missing and why these items would complete the wardrobe based on local trends",
  "items": ["item1", "item2", "item3"]
}

Example:
{
  "analysis": "Your wardrobe lacks versatile layering pieces for San Jose's variable weather. Adding these items would give you more outfit options for both casual tech offices and weekend activities.",
  "items": ["denim jacket", "black jeans", "white sneakers"]
}`;

      console.log('[MoodBoard] Sending prompt to Gemini:', prompt);

      const response = await fetch('http://localhost:5001/gemini-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('[MoodBoard] Gemini response:', data.response);
        
        let gaps = [];
        let analysis = '';
        
        try {
          // Clean the response - remove markdown if present
          let cleanedResponse = data.response.trim();
          
          // Remove markdown code blocks
          if (cleanedResponse.includes('```json')) {
            cleanedResponse = cleanedResponse.replace(/```json\s*/g, '').replace(/```/g, '');
          } else if (cleanedResponse.includes('```')) {
            cleanedResponse = cleanedResponse.replace(/```/g, '');
          }
          
          // Try to parse as JSON
          const parsed = JSON.parse(cleanedResponse);
          if (parsed.items && Array.isArray(parsed.items)) {
            gaps = parsed.items.slice(0, 3);
            analysis = parsed.analysis || '';
          }
        } catch (e) {
          // Fallback to comma/newline parsing if not valid JSON
          console.log('[MoodBoard] Response not JSON, falling back to text parsing');
          console.log('[MoodBoard] Parse error:', e);
          
          // Remove any JSON artifacts
          let cleanText = data.response.replace(/[{}"\[\]]/g, '');
          
          if (cleanText.includes(',')) {
            gaps = cleanText.split(',').map(item => item.trim()).filter(item => item.length > 0);
          } else {
            gaps = cleanText.split('\n').map(item => item.trim()).filter(item => item.length > 0);
          }
          gaps = gaps.slice(0, 3).map(item => {
            return item.replace(/^\d+\.\s*/, '').replace(/^[-*]\s*/, '').trim();
          });
        }
        
        console.log('[MoodBoard] Wardrobe gaps identified:', gaps);
        console.log('[MoodBoard] Analysis:', analysis);
        setWardrobeGaps(gaps);
        setWardrobeAnalysis(analysis);
        
        // Update the cache with wardrobe gaps and analysis
        const dataToCache = runData || lastRunData;
        if (dataToCache) {
          const updatedData = { ...dataToCache, wardrobeGaps: gaps, wardrobeAnalysis: analysis };
          setLastRunData(updatedData);
          localStorage.setItem(`moodboard_cache_${user.uid}`, JSON.stringify(updatedData));
        }
        
        // Fetch products for the identified gaps
        if (gaps.length > 0 && trendsData.trending_brands) {
          fetchWardrobeGapProducts(gaps, trendsData.trending_brands, runData);
        }
      } else {
        console.error('[MoodBoard] Gemini API failed:', response.status);
      }
    } catch (error) {
      console.error('[MoodBoard] Error analyzing wardrobe gaps:', error);
    }
  };

  // Fetch actual products for wardrobe gaps using hot brands
  const fetchWardrobeGapProducts = async (gaps, trendingBrands, runData = null) => {
    console.log('[MoodBoard] Fetching products for wardrobe gaps:', gaps);
    console.log('[MoodBoard] Using trending brands:', trendingBrands.map(b => b.name));
    
    setLoadingGapProducts(true);
    
    try {
      const currentLocation = viewMode === 'home' ? homeLocation : tripCity || homeLocation;
      
      // Build a single search query mixing brands and items intelligently
      const searchParts = [];
      
      // Distribute items across different brands for variety
      gaps.slice(0, 3).forEach((gap, index) => {
        const brand = trendingBrands[index % trendingBrands.length];
        if (brand) {
          searchParts.push(`${brand.name} mens ${gap}`);
        }
      });
      
      // If we have fewer than 3 brands, add generic search terms
      if (searchParts.length < 3) {
        gaps.slice(searchParts.length).forEach(gap => {
          searchParts.push(`mens ${gap} trending ${currentLocation}`);
        });
      }
      
      const combinedQuery = searchParts.join(' OR ');
      console.log(`[MoodBoard] Single Serper search: ${combinedQuery}`);
      
      // Single Serper API call
      const response = await fetch('http://localhost:5001/search-products', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: combinedQuery,
          num_results: 10  // Get more results to pick best 6
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.products) {
          // Process and categorize products
          const processedProducts = data.products.slice(0, 6).map((product, idx) => {
            // Try to match product to a gap category
            let matchedGap = gaps[0]; // default
            const productTitle = product.title.toLowerCase();
            
            gaps.forEach(gap => {
              if (productTitle.includes(gap.toLowerCase())) {
                matchedGap = gap;
              }
            });
            
            // Find which brand this might be from
            let matchedBrand = trendingBrands[0];
            trendingBrands.slice(0, 2).forEach(brand => {
              if (productTitle.includes(brand.name.toLowerCase())) {
                matchedBrand = brand;
              }
            });
            
            return {
              ...product,
              category: matchedGap,
              search_brand: matchedBrand.name,
              brand_affinity: matchedBrand.affinity || 0.8
            };
          });
          
          console.log(`[MoodBoard] Found ${processedProducts.length} products from single search`);
          setWardrobeGapProducts(processedProducts);
          
          // Update cache
          const dataToCache = runData || lastRunData;
          if (dataToCache) {
            const updatedData = { ...dataToCache, wardrobeGapProducts: processedProducts };
            setLastRunData(updatedData);
            localStorage.setItem(`moodboard_cache_${user.uid}`, JSON.stringify(updatedData));
          }
        }
      } else {
        console.error('[MoodBoard] Search API failed:', response.status);
      }
    } catch (error) {
      console.error('[MoodBoard] Error fetching wardrobe gap products:', error);
    } finally {
      setLoadingGapProducts(false);
    }
  };

  // Fetch tribe picks based on music and movie preferences
  const fetchTribePicks = async () => {
    console.log('[MoodBoard] ========== FETCHING TRIBE PICKS ==========');
    console.log('[MoodBoard] fetchTribePicks called!');
    setLoadingTribePicks(true);
    
    try {
      // Get first item from the arrays stored by the bubble UI
      const favoriteGenres = JSON.parse(localStorage.getItem(`favoriteGenres_${user.uid}`) || '[]');
      const favoriteMovies = JSON.parse(localStorage.getItem(`favoriteMovies_${user.uid}`) || '[]');
      
      const userMusic = favoriteGenres[0] || 'Rock';
      const userMovies = favoriteMovies[0] || 'Die Hard';
      const currentLocation = viewMode === 'home' ? homeLocation : tripCity || homeLocation;
      
      console.log(`[MoodBoard] User preferences - Music: ${userMusic}, Movies: ${userMovies}`);
      
      const response = await fetch('http://localhost:5001/tribe-picks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          music: userMusic,
          movies: userMovies,
          location: currentLocation
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('[MoodBoard] Tribe picks data:', data);
        console.log('[MoodBoard] Number of brands received:', data.brands?.length || 0);
        console.log('[MoodBoard] All brands:', JSON.stringify(data.brands, null, 2));
        setTribePicks(data);
        
        // Update cache
        if (lastRunData) {
          const updatedData = { ...lastRunData, tribePicks: data };
          setLastRunData(updatedData);
          localStorage.setItem(`moodboard_cache_${user.uid}`, JSON.stringify(updatedData));
        }
      } else {
        console.error('[MoodBoard] Tribe picks API failed:', response.status);
        setTribePicks(null);
      }
    } catch (error) {
      console.error('[MoodBoard] Error fetching tribe picks:', error);
      setTribePicks(null);
    } finally {
      setLoadingTribePicks(false);
    }
  };

  const fetchTrendsAndRecommendations = async () => {
    console.log('[MoodBoard] fetchTrendsAndRecommendations called');
    setLoading(true);
    const currentLocation = viewMode === 'home' ? homeLocation : tripCity;
    
    if (viewMode === 'trip' && !tripCity) {
      console.log('[MoodBoard] Trip mode but no city entered, aborting');
      setLoading(false);
      return;
    }
    
    console.log(`[MoodBoard] Fetching trends for ${currentLocation} (${viewMode} mode)`);
    console.log('[MoodBoard] API URL:', `http://localhost:5001/fashion-trends/${encodeURIComponent(currentLocation)}?context=${viewMode}`);
    
    try {
      // Fetch fashion trends
      const trendsResponse = await fetch(`http://localhost:5001/fashion-trends/${encodeURIComponent(currentLocation)}?context=${viewMode}`);
      console.log(`[MoodBoard] Trends API response status: ${trendsResponse.status}`);
      
      let trendsData = null;
      let recsData = null;
      
      if (trendsResponse.ok) {
        trendsData = await trendsResponse.json();
        console.log('[MoodBoard] Trends data:', trendsData);
        setTrends(trendsData);
      } else {
        console.error('[MoodBoard] Trends API failed:', trendsResponse.status);
      }
      
      // Fetch shopping recommendations
      console.log('[MoodBoard] Fetching shopping recommendations...');
      const recsResponse = await fetch('http://localhost:5001/shopping-recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.uid,
          location: currentLocation,
          shopping_need: 'wardrobe essentials',
          budget: 'medium'
        })
      });
      
      console.log('[MoodBoard] Shopping recommendations response status:', recsResponse.status);
      
      if (recsResponse.ok) {
        recsData = await recsResponse.json();
        console.log('[MoodBoard] Recommendations data received:', recsData);
        console.log('[MoodBoard] Number of recommendations:', recsData.recommendations?.length || 0);
        setRecommendations(recsData.recommendations || []);
      } else {
        const errorText = await recsResponse.text();
        console.error('[MoodBoard] Recommendations API failed:', recsResponse.status, errorText);
      }

      // Store the fresh data for save functionality and cache
      if (trendsData || recsData) {
        const runData = {
          location: currentLocation,
          viewMode,
          trends: trendsData,
          recommendations: recsData?.recommendations || [],
          wardrobeGaps: [],  // Will be populated by Gemini
          wardrobeGapProducts: [],  // Will be populated after Gemini
          timestamp: new Date().toISOString()
        };
        setLastRunData(runData);
        console.log('[MoodBoard] Updated lastRunData:', runData);
        
        // Cache the data in localStorage
        try {
          localStorage.setItem(`moodboard_cache_${user.uid}`, JSON.stringify(runData));
          console.log('[MoodBoard] Cached data to localStorage');
        } catch (e) {
          console.error('[MoodBoard] Error caching data:', e);
        }
        
        // Analyze wardrobe gaps after setting runData
        if (trendsData) {
          setTimeout(() => analyzeWardrobeGaps(trendsData, runData), 500);
        }
        
        // Fetch tribe picks
        console.log('[MoodBoard] About to call fetchTribePicks...');
        fetchTribePicks();
      }
      
    } catch (error) {
      console.error('[MoodBoard] Error fetching data:', error);
      console.error('[MoodBoard] Error details:', error.message, error.stack);
      
      // Set some fallback data so UI shows something
      setTrends({
        success: true,
        trending_items: ['oversized blazers', 'wide-leg pants', 'platform shoes'],
        trending_brands: [
          { name: 'Zara', affinity: 0.9 },
          { name: 'Uniqlo', affinity: 0.85 }
        ],
        season: 'winter',
        time_context: { time_of_day: 'evening' },
        source: 'fallback-error'
      });
    } finally {
      setLoading(false);
      setInitialLoad(false);
      console.log('[MoodBoard] Loading complete');
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      // Build complete data object with current state
      const dataToSave = {
        location: viewMode === 'home' ? homeLocation : tripCity,
        viewMode,
        trends,
        recommendations,
        wardrobeGaps,
        wardrobeGapProducts,
        wardrobeAnalysis,
        tribePicks,
        timestamp: new Date().toISOString()
      };
      
      console.log('[MoodBoard] Saving data:', dataToSave);
      console.log('[MoodBoard] Wardrobe gap products count:', wardrobeGapProducts.length);
      console.log('[MoodBoard] Wardrobe analysis:', wardrobeAnalysis);
      console.log('[MoodBoard] Tribe picks being saved:', tribePicks);
      console.log('[MoodBoard] Number of tribe brands being saved:', tribePicks?.brands?.length || 0);
      
      const response = await fetch('http://localhost:5001/moodboard/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.uid,
          data: dataToSave
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('[MoodBoard] Save successful:', result);
        // Update lastRunData with saved data
        setLastRunData(dataToSave);
        // Also update localStorage
        localStorage.setItem(`moodboard_cache_${user.uid}`, JSON.stringify(dataToSave));
      } else {
        const error = await response.text();
        console.error('[MoodBoard] Save failed:', error);
      }
    } catch (error) {
      console.error('[MoodBoard] Error saving moodboard data:', error);
    } finally {
      setSaving(false);
    }
  };

  // Load cached data on mount
  useEffect(() => {
    if (!hasInitialized) {
      // Try to load cached data first
      const cachedData = localStorage.getItem(`moodboard_cache_${user.uid}`);
      if (cachedData) {
        try {
          const parsed = JSON.parse(cachedData);
          console.log('[MoodBoard] Loading cached data:', parsed);
          console.log('[MoodBoard] Cached recommendations count:', parsed.recommendations?.length || 0);
          
          // Only load cache if it matches current view mode
          if (parsed.viewMode === viewMode) {
            setTrends(parsed.trends);
            setRecommendations(parsed.recommendations || []);
            setLastRunData(parsed);
            
            // Also check if we have cached wardrobe gaps
            if (parsed.wardrobeGaps) {
              console.log('[MoodBoard] Loading cached wardrobe gaps:', parsed.wardrobeGaps);
              setWardrobeGaps(parsed.wardrobeGaps);
            }
            
            // Load cached wardrobe gap products
            if (parsed.wardrobeGapProducts) {
              console.log('[MoodBoard] Loading cached wardrobe gap products:', parsed.wardrobeGapProducts.length);
              setWardrobeGapProducts(parsed.wardrobeGapProducts);
            }
            
            // Load cached wardrobe analysis
            if (parsed.wardrobeAnalysis) {
              console.log('[MoodBoard] Loading cached wardrobe analysis');
              setWardrobeAnalysis(parsed.wardrobeAnalysis);
            }
            
            // Load cached tribe picks
            if (parsed.tribePicks) {
              console.log('[MoodBoard] Loading cached tribe picks:', parsed.tribePicks);
              console.log('[MoodBoard] Number of cached tribe brands:', parsed.tribePicks?.brands?.length || 0);
              console.log('[MoodBoard] All cached brands:', JSON.stringify(parsed.tribePicks?.brands, null, 2));
              setTribePicks(parsed.tribePicks);
            }
          }
        } catch (e) {
          console.error('[MoodBoard] Error parsing cached data:', e);
        }
      } else {
        // If no localStorage cache, try to load from Firebase
        loadLatestMoodboardFromFirebase();
      }
      setHasInitialized(true);
    }
  }, []);
  
  // Load latest moodboard from Firebase
  const loadLatestMoodboardFromFirebase = async () => {
    try {
      console.log('[MoodBoard] Loading latest moodboard from Firebase...');
      const response = await fetch(`http://localhost:5001/moodboard/latest/${user.uid}`);
      
      if (response.ok) {
        const data = await response.json();
        console.log('[MoodBoard] Firebase response:', data);
        
        if (data.success && data.moodboard) {
          const moodboard = data.moodboard;
          console.log('[MoodBoard] Loaded moodboard from Firebase:', moodboard);
          console.log('[MoodBoard] Firebase wardrobe gaps:', moodboard.wardrobeGaps);
          console.log('[MoodBoard] Firebase gap products count:', moodboard.wardrobeGapProducts?.length || 0);
          console.log('[MoodBoard] Firebase analysis:', moodboard.wardrobeAnalysis);
          
          setTrends(moodboard.trends);
          setRecommendations(moodboard.recommendations || []);
          setWardrobeGaps(moodboard.wardrobeGaps || []);
          setWardrobeGapProducts(moodboard.wardrobeGapProducts || []);
          setWardrobeAnalysis(moodboard.wardrobeAnalysis || '');
          setTribePicks(moodboard.tribePicks || null);
          setLastRunData(moodboard);
          
          // Also cache it locally
          localStorage.setItem(`moodboard_cache_${user.uid}`, JSON.stringify(moodboard));
        } else {
          console.log('[MoodBoard] No moodboard data found in Firebase');
        }
      } else {
        console.error('[MoodBoard] Firebase load failed:', response.status);
      }
    } catch (error) {
      console.error('[MoodBoard] Error loading from Firebase:', error);
    }
  };
  
  // Only clear data when user actually changes the mode
  const handleModeChange = (newMode) => {
    if (newMode !== viewMode) {
      console.log('[MoodBoard] Mode changing from', viewMode, 'to', newMode);
      setViewMode(newMode);
      
      // Reload cached data from localStorage
      const cachedData = localStorage.getItem(`moodboard_cache_${user.uid}`);
      if (cachedData) {
        try {
          const parsed = JSON.parse(cachedData);
          // Check if cached data matches the new mode
          if (parsed.viewMode === newMode) {
            console.log(`[MoodBoard] Found cached data for ${newMode} mode, loading...`);
            setTrends(parsed.trends);
            setRecommendations(parsed.recommendations || []);
            setLastRunData(parsed);
            setWardrobeGaps(parsed.wardrobeGaps || []);
            setWardrobeGapProducts(parsed.wardrobeGapProducts || []);
            setWardrobeAnalysis(parsed.wardrobeAnalysis || '');
            setTribePicks(parsed.tribePicks || null);
          } else {
            console.log(`[MoodBoard] Cached data is for ${parsed.viewMode} mode, clearing for ${newMode}`);
            // Clear everything if mode doesn't match
            setTrends(null);
            setRecommendations([]);
            setLastRunData(null);
            setWardrobeGaps([]);
            setWardrobeGapProducts([]);
            setWardrobeAnalysis('');
            setTribePicks(null);
          }
        } catch (e) {
          console.error('[MoodBoard] Error parsing cached data:', e);
        }
      } else {
        // No cache, clear everything
        setTrends(null);
        setRecommendations([]);
        setLastRunData(null);
        setWardrobeGaps([]);
        setWardrobeGapProducts([]);
        setWardrobeAnalysis('');
        setTribePicks(null);
      }
      
      setLoadingGapProducts(false);
    }
  };

  // Get posh welcome message based on season and time
  const getPoshWelcomeMessage = (season, timeOfDay) => {
    const messages = {
      winter: {
        morning: "Cashmere thoughts and morning frost",
        afternoon: "Wool coats and champagne toasts",
        evening: "Velvet nights and city lights",
        night: "Midnight luxe and starlit dreams"
      },
      spring: {
        morning: "Silk scarves and sunrise moments",
        afternoon: "Linen lunches and garden parties",
        evening: "Chiffon sunsets and rooftop views",
        night: "Moonlit strolls in couture"
      },
      summer: {
        morning: "Crisp cottons and yacht club mornings",
        afternoon: "Parasols and poolside glamour",
        evening: "Golden hour in flowing fabrics",
        night: "Terrace soirées under stars"
      },
      fall: {
        morning: "Tweed jackets and misty mornings",
        afternoon: "Leather boots and vintage finds",
        evening: "Burgundy hues and gallery views",
        night: "Candlelit dinners in tailored fits"
      }
    };
    
    return messages[season]?.[timeOfDay] || messages.winter.evening;
  };

  return (
    <div className="moodboard-container">
      <div className="moodboard-controls">
        <button 
          className="action-btn run-btn"
          onClick={() => {
            console.log('[MoodBoard] Run button clicked - fetching fresh data');
            fetchTrendsAndRecommendations();
          }}
          disabled={loading || (viewMode === 'trip' && !tripCity)}
        >
          {loading ? '...' : 'Run'}
        </button>
        <button 
          className="action-btn save-btn"
          onClick={handleSave}
          disabled={saving || !lastRunData}
        >
          {saving ? '...' : 'Save'}
        </button>
        <div className="mode-toggle">
          <button 
            className={`mode-btn ${viewMode === 'home' ? 'active' : ''}`}
            onClick={() => handleModeChange('home')}
          >
            Home
          </button>
          <button 
            className={`mode-btn ${viewMode === 'trip' ? 'active' : ''}`}
            onClick={() => handleModeChange('trip')}
          >
            Trip
          </button>
        </div>
        {viewMode === 'trip' && (
          <input
            type="text"
            className="trip-input"
            placeholder="City"
            value={tripCity}
            onChange={(e) => setTripCity(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && fetchTrendsAndRecommendations()}
          />
        )}
        <span className="location-text">
          {viewMode === 'home' ? homeLocation : (tripCity || '...')}
        </span>
      </div>

      <div className="moodboard-grid">
            {/* Trends Card */}
            <div className={`glass-card trends-card ${loading ? 'loading-glow' : ''}`}>
              <h3 className="card-title">
                trending in {viewMode === 'home' ? homeLocation.toLowerCase() : (tripCity.toLowerCase() || 'your destination')}
                {trends?.source === 'qloo' && <span className="data-source"> • live</span>}
              </h3>
              <div className="card-content">
                <div className="items-list">
                  {trends?.success && trends.trending_items?.length > 0 ? (
                    trends.trending_items.slice(0, 5).map((item, idx) => (
                      <span key={idx} className="trend-pill">{item}</span>
                    ))
                  ) : (
                    <>
                      <span className="trend-pill">minimalist blazers</span>
                      <span className="trend-pill">wide-leg pants</span>
                      <span className="trend-pill">platform shoes</span>
                      <span className="trend-pill">crossbody bags</span>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Top Brands Card - Enhanced with Qloo data */}
            <div className={`glass-card brands-card ${loading ? 'loading-glow' : ''}`}>
              <h3 className="card-title">hot brands</h3>
              <div className="card-content">
                <div className="brand-pills">
                  {trends?.success && trends.trending_brands?.length > 0 ? (
                    trends.trending_brands.map((brand, idx) => (
                      <span key={idx} className="brand-pill">
                        {brand.name}
                        <span className="brand-pill-score">{Math.round(brand.affinity * 100)}%</span>
                      </span>
                    ))
                  ) : (
                    <>
                      <span className="brand-pill">
                        Zara
                        <span className="brand-pill-score">92%</span>
                      </span>
                      <span className="brand-pill">
                        Uniqlo
                        <span className="brand-pill-score">88%</span>
                      </span>
                      <span className="brand-pill">
                        Everlane
                        <span className="brand-pill-score">85%</span>
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Season Context Card */}
            <div className={`glass-card context-card ${loading ? 'loading-glow' : ''}`}>
              <h3 className="card-title">right now</h3>
              <div className="card-content">
                <div className="context-info">
                  <p className="season-text">
                    {getPoshWelcomeMessage(
                      trends?.season || 'winter',
                      trends?.time_context?.time_of_day || 'evening'
                    )}
                  </p>
                </div>
              </div>
            </div>

            {/* Recommendations Card - All in one */}
            {console.log('[MoodBoard] Rendering, recommendations count:', recommendations.length)}
            <div className={`glass-card recommendations-card ${loading ? 'loading-glow' : ''}`}>
              <h3 className="card-title">
                recommended for you based on {(viewMode === 'home' ? homeLocation : tripCity).toLowerCase()} trends
              </h3>
              <div className="recommendations-grid">
                {loading && recommendations.length === 0 ? (
                  // Show placeholders while loading
                  [1, 2, 3, 4, 5, 6].map(idx => (
                    <div key={`placeholder-${idx}`} className="product-item recommendation-item">
                      <div className="recommendation-image-placeholder"></div>
                    </div>
                  ))
                ) : (
                  recommendations.slice(0, 6).map((item, idx) => (
                    <div 
                      key={idx} 
                      className="product-item recommendation-item" 
                      onClick={() => setSelectedProduct(item)}
                    >
                      <div className="product-image recommendation-image">
                        {item.image ? (
                          <img 
                            src={item.image} 
                            alt={item.name}
                            onError={(e) => {
                              console.log('[MoodBoard] Image failed to load:', item.image);
                              e.target.style.display = 'none';
                            }}
                          />
                        ) : (
                          <div className="image-placeholder"></div>
                        )}
                      </div>
                      <div className="recommendation-overlay">
                        <p className="overlay-brand">{item.brand}</p>
                        <p className="overlay-price">{item.price}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Your Tribe's Picks Card */}
            <div className={`glass-card tribe-picks-card ${loadingTribePicks ? 'loading-glow' : ''}`}>
              <h3 className="card-title">where your tribe shops</h3>
              <div className="card-content">
                {/* Debug info to show preferences */}
                <p className="debug-info" style={{fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)', marginBottom: '0.5rem'}}>
                  Genre: {JSON.parse(localStorage.getItem(`favoriteGenres_${user.uid}`) || '[]')[0] || 'Not set'} | 
                  Movie: {JSON.parse(localStorage.getItem(`favoriteMovies_${user.uid}`) || '[]')[0] || 'Not set'}
                </p>
                {tribePicks && tribePicks.success ? (
                  <>
                    <p className="tribe-subtitle">{tribePicks.description}</p>
                    <div className="brand-pills">
                      {console.log('[MoodBoard] Rendering tribe brands:', tribePicks.brands)}
                      {console.log('[MoodBoard] Number of brands to render:', tribePicks.brands?.length || 0)}
                      {tribePicks.brands.map((brand, idx) => (
                        <span key={idx} className="brand-pill">
                          {brand.name}
                          <span className="brand-pill-score">{brand.score}%</span>
                        </span>
                      ))}
                    </div>
                  </>
                ) : loadingTribePicks ? (
                  <div className="brand-pills">
                    {[1, 2, 3, 4, 5].map(idx => (
                      <span key={idx} className="brand-pill" style={{background: 'rgba(255, 255, 255, 0.05)'}}>
                        <span className="text-placeholder short" style={{height: '16px', width: '60px'}}></span>
                        <span className="text-placeholder short" style={{height: '16px', width: '30px'}}></span>
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="no-data-message">Set your music and movie preferences in settings to see your tribe's picks</p>
                )}
              </div>
            </div>

            {/* Style Twin Wardrobes Card */}
            <div className={`glass-card location-card ${loading ? 'loading-glow' : ''}`}>
              <h3 className="card-title">
                style twin wardrobes <span className="subtitle-inline">community-powered</span>
              </h3>
              <div className="card-content">
                <p className="twins-description">Peek into wardrobes of people who match your taste profile</p>
                <div className="twins-grid">
                  {getTwinWardrobeItems().map((item, idx) => (
                    <div 
                      key={idx} 
                      className="product-item"
                      onClick={() => setSelectedProduct(item)}
                    >
                      <div className="product-image">
                        <img 
                          src={item.image} 
                          alt={item.name}
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                      <div className="recommendation-overlay">
                        <p className="overlay-brand">{item.twinCount} style twins</p>
                        <p className="overlay-price">own this</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Missing Items Card - Enhanced with actual products */}
            <div className={`glass-card missing-card ${loading || loadingGapProducts ? 'loading-glow' : ''}`}>
              <h3 className="card-title">
                you need <span className="subtitle-inline">based on your wardrobe analysis</span>
              </h3>
              <div className="card-content">
                {wardrobeAnalysis && (
                  <p className="wardrobe-analysis-text">{wardrobeAnalysis}</p>
                )}
                {loadingGapProducts ? (
                  <div className="wardrobe-gap-products">
                    <div className="gap-products-scroll">
                      {[1, 2, 3, 4, 5, 6].map(idx => (
                        <div key={`loading-${idx}`} className="product-item gap-product-item">
                          <div className="product-image gap-product-image">
                            <div className="image-placeholder"></div>
                          </div>
                          <div className="gap-product-info">
                            <div className="text-placeholder short" style={{height: '12px', marginBottom: '4px'}}></div>
                            <div className="text-placeholder" style={{height: '10px', marginBottom: '4px'}}></div>
                            <div className="text-placeholder short" style={{height: '14px'}}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : wardrobeGapProducts.length > 0 ? (
                  <div className="wardrobe-gap-products">
                    <div className="gap-products-scroll">
                      {wardrobeGapProducts.map((product, idx) => (
                        <div 
                          key={idx} 
                          className="product-item gap-product-item"
                          onClick={() => setSelectedProduct(product)}
                        >
                          <div className="product-image gap-product-image">
                            {product.image ? (
                              <img 
                                src={product.image} 
                                alt={product.name || product.title}
                                onError={(e) => {
                                  e.target.style.display = 'none';
                                }}
                              />
                            ) : (
                              <div className="image-placeholder"></div>
                            )}
                          </div>
                          <div className="recommendation-overlay">
                            {product.brand && <p className="overlay-brand">{product.brand}</p>}
                            <p className="overlay-price">{product.price}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="missing-items">
                    {wardrobeGaps.length > 0 && !wardrobeGapProducts.length ? (
                      wardrobeGaps.map((gap, idx) => (
                        <span key={idx} className="missing-pill">{gap.toLowerCase()}</span>
                      ))
                    ) : !wardrobeGapProducts.length ? (
                      loading ? (
                        <>
                          <span className="missing-pill shimmer">analyzing...</span>
                          <span className="missing-pill shimmer">analyzing...</span>
                          <span className="missing-pill shimmer">analyzing...</span>
                        </>
                      ) : (
                        <>
                          <span className="missing-pill">versatile blazer</span>
                          <span className="missing-pill">white sneakers</span>
                          <span className="missing-pill">casual friday shirt</span>
                        </>
                      )
                    ) : null}
                  </div>
                )}
              </div>
            </div>
      </div>

      {/* Product Detail Popup */}
      {selectedProduct && (
        <div className="product-popup-overlay" onClick={() => setSelectedProduct(null)}>
          <div className="product-popup" onClick={(e) => e.stopPropagation()}>
            <button className="popup-close" onClick={() => setSelectedProduct(null)}>×</button>
            <div className="popup-content">
              <div className="popup-image">
                {selectedProduct.image && <img src={selectedProduct.image} alt={selectedProduct.name || selectedProduct.title} />}
              </div>
              <div className="popup-details">
                {selectedProduct.brand && <h3 className="popup-brand">{selectedProduct.brand}</h3>}
                <h2 className="popup-name">{selectedProduct.name || selectedProduct.title}</h2>
                <p className="popup-price">{selectedProduct.price}</p>
                {selectedProduct.description && (
                  <p className="popup-description">{selectedProduct.description}</p>
                )}
                <p className="popup-affinity">
                  brand affinity: {Math.round((selectedProduct.affinity_score || selectedProduct.brand_affinity || 0.8) * 100)}%
                </p>
                {selectedProduct.link && (
                  <a 
                    href={selectedProduct.link} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="popup-visit-btn"
                  >
                    Visit Now →
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MoodBoard;