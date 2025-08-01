import os
import requests
from datetime import datetime
import json
from .moodboard import TrendsService

class FashionTrendsService:
    def __init__(self):
        self.qloo_api_key = os.getenv('QLOO_API_KEY', '')
        self.qloo_api_url = os.getenv('QLOO_API_URL', 'https://hackathon.api.qloo.com')
        self.serper_api_key = os.getenv('SERPER_API_KEY', os.getenv('SERPAPI_KEY', ''))
        
    def get_current_time_context(self):
        """Get time-based context for recommendations"""
        now = datetime.now()
        hour = now.hour
        month = now.month
        
        # Time of day
        if 6 <= hour < 12:
            time_context = "morning"
        elif 12 <= hour < 17:
            time_context = "afternoon"
        elif 17 <= hour < 21:
            time_context = "evening"
        else:
            time_context = "night"
            
        # Season
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"
            
        return {
            "time_of_day": time_context,
            "season": season,
            "hour": hour,
            "month": month
        }
    
    def get_fashion_trends(self, location, context="home"):
        """Get fashion trends from Qloo API"""
        if not self.qloo_api_key:
            return {"success": False, "error": "QLOO API key not configured"}
            
        try:
            headers = {'X-Api-Key': self.qloo_api_key}
            
            # Get insights for fashion
            insights_url = f"{self.qloo_api_url}/insights"
            insights_params = {
                'category': 'Fashion',
                'city': location.replace(' ', ''),
                'limit': 10
            }
            
            response = requests.get(insights_url, headers=headers, params=insights_params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract brand and style trends
                trending_brands = []
                trending_items = []
                
                # Get brands with affinity scores
                if 'results' in data:
                    for result in data.get('results', [])[:5]:
                        brand_info = {
                            'name': result.get('name', ''),
                            'affinity': result.get('affinity', 0),
                            'category': result.get('tags', ['Fashion'])[0] if result.get('tags') else 'Fashion'
                        }
                        trending_brands.append(brand_info)
                
                # Extract trending items from tags
                for result in data.get('results', []):
                    for tag in result.get('tags', []):
                        if tag.lower() not in ['fashion', 'brand', 'store'] and tag not in trending_items:
                            trending_items.append(tag)
                
                time_context = self.get_current_time_context()
                
                return {
                    "success": True,
                    "location": location,
                    "context": context,
                    "time_context": time_context,
                    "trending_brands": trending_brands[:5],
                    "trending_items": trending_items[:8],
                    "season": time_context["season"]
                }
            else:
                return {"success": False, "error": f"QLOO API returned {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_shopping_recommendations(self, location, shopping_need, budget="medium", wardrobe_data=None):
        """Get shopping recommendations using Qloo trends and Serper search"""
        print(f"[FashionTrendsService] get_shopping_recommendations called")
        print(f"[FashionTrendsService] Location: {location}, Need: {shopping_need}, Budget: {budget}")
        print(f"[FashionTrendsService] SERPER_API_KEY configured: {bool(self.serper_api_key)}")
        
        # First get trends from Qloo using TrendsService
        trends_service = TrendsService()
        trends = trends_service.get_trending_items(location, "shopping")
        if not trends.get("success"):
            return {"success": False, "error": "Failed to get fashion trends"}
        
        trending_brands = trends.get("trending_brands", [])
        print(f"[FashionTrendsService] Got {len(trending_brands)} trending brands from Qloo")
        
        if not self.serper_api_key:
            print(f"[FashionTrendsService] No SERPER_API_KEY, using fallback recommendations")
            # Return intelligent fallback recommendations
            return self._get_fallback_recommendations(location, shopping_need, budget, trends)
            
        try:
            recommendations = []
            
            # Map budget to price ranges
            price_ranges = {
                "low": "under $50",
                "medium": "$50-$150", 
                "high": "over $150"
            }
            price_filter = price_ranges.get(budget, "$50-$150")
            
            # Build single search query with variety
            search_parts = []
            brand_map = {}  # Track which brand is which for later
            
            # Create varied search terms
            if len(trending_brands) >= 3:
                # If we have 3+ brands, use different ones
                search_parts.append(f"{trending_brands[0]['name']} mens {shopping_need}")
                search_parts.append(f"{trending_brands[1]['name']} mens clothing")
                search_parts.append(f"{trending_brands[2]['name']} mens fashion")
                for brand_info in trending_brands[:3]:
                    brand_map[brand_info['name'].lower()] = brand_info
            else:
                # If fewer brands, vary the search terms
                for i, brand_info in enumerate(trending_brands):
                    brand = brand_info['name']
                    brand_map[brand.lower()] = brand_info
                    if i == 0:
                        search_parts.append(f"{brand} mens {shopping_need}")
                    else:
                        search_parts.append(f"mens {shopping_need} {brand}")
            
            # Add location context
            search_parts.append(f"trending mens fashion {location}")
            
            combined_query = f"{' OR '.join(search_parts)} {price_filter}"
            print(f"[FashionTrendsService] Single Serper search: {combined_query}")
            
            # Single Serper API call
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            search_url = "https://google.serper.dev/shopping"
            
            payload = {
                'q': combined_query,
                'num': 10,  # Get more results to have variety
                'gl': 'us'
            }
            
            response = requests.post(search_url, headers=headers, json=payload)
            print(f"[FashionTrendsService] Serper response status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                print(f"[FashionTrendsService] Got {len(results.get('shopping', []))} shopping results")
                
                # Extract shopping results with brand diversity
                brand_count = {}
                max_per_brand = 2
                
                for item in results.get('shopping', []):
                    if len(recommendations) >= 6:
                        break
                        
                    item_title = item.get('title', '')
                    item_title_lower = item_title.lower()
                    
                    # Use source as the brand - it's the actual retailer/brand name
                    brand = item.get('source', 'Unknown')
                    
                    # Try to match to trending brands for affinity score
                    matched_brand_info = None
                    for brand_lower, brand_info in brand_map.items():
                        if brand_lower in brand.lower() or brand_lower in item_title_lower:
                            matched_brand_info = brand_info
                            break
                    
                    # Track brand count
                    if brand not in brand_count:
                        brand_count[brand] = 0
                    
                    # Only add if we haven't exceeded max per brand
                    if brand_count[brand] < max_per_brand:
                        brand_count[brand] += 1
                        
                        rec = {
                            'name': item_title,
                            'brand': brand,  # Use source as brand
                            'price': item.get('price', 'N/A'),
                            'image': item.get('imageUrl', ''),
                            'link': item.get('link', ''),
                            'source': brand,
                            'affinity_score': matched_brand_info.get('affinity', 0) if matched_brand_info else 0
                        }
                        recommendations.append(rec)
            else:
                print(f"[FashionTrendsService] Serper API error: {response.text}")
            
            # Analyze what's missing from wardrobe
            missing_categories = self._analyze_wardrobe_gaps(wardrobe_data, shopping_need)
            
            print(f"[FashionTrendsService] Total recommendations found: {len(recommendations)}")
            
            return {
                "success": True,
                "recommendations": recommendations[:6],
                "missing_categories": missing_categories,
                "location": location,
                "trends": trends
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_fallback_recommendations(self, location, shopping_need, budget, trends):
        """Get fallback recommendations when Serper API is not available"""
        print(f"[FashionTrendsService] Using recommendations based on QLOO trends for {location}")
        
        # Get trending brands and items from QLOO trends
        trending_brands = trends.get('trending_brands', [])
        trending_items = trends.get('trending_items', [])
        
        recommendations = []
        
        # Create recommendations based on actual trending items
        if trending_items:
            for i, item_type in enumerate(trending_items[:3]):
                # Map QLOO item types to specific products
                item_mapping = {
                    'jeans': {'name': 'Premium Denim Jeans', 'price': '$98'},
                    'accessories': {'name': 'Minimalist Leather Wallet', 'price': '$65'},
                    'bags & leather': {'name': 'Crossbody Leather Bag', 'price': '$185'},
                    'home accessories': {'name': 'Tech Organizer Pouch', 'price': '$45'},
                    'footwear': {'name': 'Versatile Sneakers', 'price': '$125'},
                    'clothing': {'name': 'Essential Cotton Shirt', 'price': '$75'}
                }
                
                # Get product details or use generic
                product = item_mapping.get(item_type, {'name': f'Trending {item_type.title()}', 'price': '$85'})
                brand = trending_brands[i % len(trending_brands)] if trending_brands else {'name': 'Local Brand', 'affinity': 0.8}
                
                rec = {
                    'name': product['name'],
                    'brand': brand.get('name'),
                    'price': product['price'],
                    'category': item_type,
                    'image': '',  # No placeholder images
                    'link': '',
                    'source': 'qloo-trending',
                    'affinity_score': brand.get('affinity', 0.8)
                }
                recommendations.append(rec)
        
        # Add essentials based on shopping need
        if shopping_need == 'wardrobe essentials' and len(recommendations) < 3:
            # Add missing essentials with top brands
            essentials = [
                {'name': 'Classic White Shirt', 'category': 'shirt', 'price': '$75'},
                {'name': 'Versatile Chinos', 'category': 'pants', 'price': '$95'}
            ]
            
            for i, item in enumerate(essentials):
                if len(recommendations) >= 6:
                    break
                brand = trending_brands[(len(recommendations) + i) % len(trending_brands)] if trending_brands else {'name': 'Local Brand', 'affinity': 0.8}
                
                rec = {
                    'name': item['name'],
                    'brand': brand.get('name'),
                    'price': item['price'],
                    'category': item['category'],
                    'image': '',
                    'link': '',
                    'source': 'essential',
                    'affinity_score': brand.get('affinity', 0.8)
                }
                recommendations.append(rec)
        
        # Add location-specific items
        if 'miami' in location.lower() or 'los angeles' in location.lower():
            recommendations.append({
                'name': 'Linen Beach Shirt',
                'brand': 'Everlane',
                'price': '$68',
                'category': 'shirt',
                'image': 'https://via.placeholder.com/300x400?text=Linen+Shirt',
                'link': '#',
                'source': 'location-based',
                'affinity_score': 0.9
            })
        elif 'new york' in location.lower() or 'chicago' in location.lower():
            recommendations.append({
                'name': 'Wool Overcoat',
                'brand': 'COS',
                'price': '$275',
                'category': 'outerwear',
                'image': 'https://via.placeholder.com/300x400?text=Wool+Coat',
                'link': '#',
                'source': 'location-based',
                'affinity_score': 0.88
            })
        
        return {
            "success": True,
            "recommendations": recommendations[:6],
            "missing_categories": ['versatile blazer', 'white sneakers', 'crossbody bag'],
            "location": location,
            "trends": trends,
            "source": "fallback"
        }
    
    def _analyze_wardrobe_gaps(self, wardrobe_data, shopping_need):
        """Analyze what's missing from user's wardrobe"""
        if not wardrobe_data:
            return []
            
        gaps = []
        
        # Check for common missing items
        shirt_count = len(wardrobe_data.get('shirts', []))
        pants_count = len(wardrobe_data.get('pants', []))
        shoes_count = len(wardrobe_data.get('shoes', []))
        
        if shirt_count < 5:
            gaps.append("more shirt varieties")
        if pants_count < 3:
            gaps.append("pants options")
        if shoes_count < 2:
            gaps.append("shoe variety")
            
        # Check for specific needs
        if "formal" in shopping_need.lower() and shirt_count < 2:
            gaps.append("formal wear")
        if "athletic" in shopping_need.lower():
            gaps.append("athletic gear")
        if "accessories" in shopping_need.lower():
            gaps.append("accessories")
            
        return gaps[:3]