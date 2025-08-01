import os
import requests
from datetime import datetime
import json

class TrendsService:
    def __init__(self):
        self.qloo_api_key = os.getenv('QLOO_API_KEY', '')
        self.qloo_api_url = 'https://hackathon.api.qloo.com'
        
    def get_trending_items(self, location, context='home'):
        """Get trending fashion items for a location using QLOO API"""
        print(f"\n[TrendsService] Getting trends for {location} (context: {context})")
        
        if not self.qloo_api_key:
            print("[TrendsService] No QLOO API key configured")
            return self._get_fallback_trends(location)
            
        try:
            headers = {'X-Api-Key': self.qloo_api_key}
            
            # Use v2 insights endpoint with proper parameters
            url = f"{self.qloo_api_url}/v2/insights"
            params = {
                'filter.type': 'urn:entity:brand',
                'filter.tags': 'urn:tag:genre:brand:fashion',
                'signal.location.query': location,
                'limit': 10,
                'sort': 'popularity'
            }
            
            print(f"[TrendsService] Calling QLOO API: {url}")
            print(f"[TrendsService] Location: {location}")
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                brands_data = data.get('results', {}).get('entities', [])
                
                print(f"[TrendsService] Got {len(brands_data)} brands from QLOO")
                
                # Extract trending items and brands
                trending_items = []
                trending_brands = []
                
                # Process brands and extract style tags
                for brand in brands_data[:8]:
                    brand_name = brand.get('name', '')
                    popularity = brand.get('popularity', 0.0)
                    
                    # Get tags to identify trending items
                    tags = brand.get('tags', [])
                    for tag in tags[:15]:
                        tag_name = tag.get('name', '').lower()
                        
                        # Extract fashion item types from tags
                        fashion_items = ['blazer', 'pants', 'shoes', 'bag', 'jacket', 'shirt', 
                                       'dress', 'sweater', 'coat', 'jeans', 'sneakers', 'boots',
                                       'accessories', 'watch', 'sunglasses', 'scarf']
                        
                        for item in fashion_items:
                            if item in tag_name and tag_name not in trending_items:
                                trending_items.append(tag_name)
                    
                    # Add brand with affinity score
                    trending_brands.append({
                        'name': brand_name,
                        'affinity': popularity,
                        'location_specific': True
                    })
                
                # If we didn't get enough trending items from tags, add some intelligent defaults
                if len(trending_items) < 4:
                    season = self._get_season()
                    if season == 'winter':
                        trending_items.extend(['wool coats', 'leather boots', 'cashmere scarves'])
                    elif season == 'summer':
                        trending_items.extend(['linen shirts', 'canvas sneakers', 'straw hats'])
                    elif season == 'spring':
                        trending_items.extend(['trench coats', 'white sneakers', 'denim jackets'])
                    else:  # fall
                        trending_items.extend(['blazers', 'ankle boots', 'knit sweaters'])
                
                # Add location-specific items
                location_lower = location.lower()
                if 'miami' in location_lower or 'los angeles' in location_lower:
                    trending_items.append('swimwear')
                    trending_items.append('sunglasses')
                elif 'new york' in location_lower or 'chicago' in location_lower:
                    trending_items.append('structured bags')
                    trending_items.append('statement coats')
                elif 'san francisco' in location_lower or 'seattle' in location_lower:
                    trending_items.append('tech-wear jackets')
                    trending_items.append('waterproof sneakers')
                
                # Remove duplicates and limit
                trending_items = list(dict.fromkeys(trending_items))[:8]
                
                print(f"[TrendsService] Extracted {len(trending_items)} trending items")
                print(f"[TrendsService] Top brands: {[b['name'] for b in trending_brands[:3]]}")
                
                return {
                    'success': True,
                    'trending_items': trending_items,
                    'trending_brands': trending_brands[:5],
                    'location': location,
                    'context': context,
                    'source': 'qloo'
                }
                
            else:
                print(f"[TrendsService] QLOO API returned {response.status_code}")
                return self._get_fallback_trends(location)
                
        except Exception as e:
            print(f"[TrendsService] Error fetching trends: {str(e)}")
            return self._get_fallback_trends(location)
    
    def _get_season(self):
        """Get current season"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_fallback_trends(self, location):
        """Fallback trends when API is not available"""
        print("[TrendsService] Using fallback trends")
        
        # Location-based fallbacks
        location_trends = {
            'new york': {
                'items': ['oversized blazers', 'chelsea boots', 'structured bags', 'wool coats', 'minimalist jewelry'],
                'brands': [
                    {'name': 'The Row', 'affinity': 0.92},
                    {'name': 'Everlane', 'affinity': 0.88},
                    {'name': 'COS', 'affinity': 0.85}
                ]
            },
            'los angeles': {
                'items': ['vintage denim', 'canvas sneakers', 'bucket hats', 'crop tops', 'sustainable tees'],
                'brands': [
                    {'name': 'Reformation', 'affinity': 0.94},
                    {'name': 'Patagonia', 'affinity': 0.89},
                    {'name': 'Veja', 'affinity': 0.86}
                ]
            },
            'miami': {
                'items': ['linen shirts', 'espadrilles', 'panama hats', 'swimwear', 'resort wear'],
                'brands': [
                    {'name': 'Orlebar Brown', 'affinity': 0.91},
                    {'name': 'Zimmermann', 'affinity': 0.88},
                    {'name': 'Solid & Striped', 'affinity': 0.84}
                ]
            }
        }
        
        # Default trends
        default_trends = {
            'items': ['minimalist blazers', 'wide-leg pants', 'chunky sneakers', 'crossbody bags', 'knit sets'],
            'brands': [
                {'name': 'Zara', 'affinity': 0.90},
                {'name': 'Uniqlo', 'affinity': 0.87},
                {'name': 'H&M', 'affinity': 0.83}
            ]
        }
        
        # Get location-specific or default
        location_key = location.lower()
        trends_data = default_trends
        
        for key in location_trends:
            if key in location_key:
                trends_data = location_trends[key]
                break
        
        return {
            'success': True,
            'trending_items': trends_data['items'],
            'trending_brands': trends_data['brands'],
            'location': location,
            'context': 'home',
            'source': 'fallback'
        }