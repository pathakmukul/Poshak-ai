"""
ElevenLabs Webhook Endpoints for Voice Assistant
"""

from flask import jsonify, request
import json
from services.langchain_agent_fixed import (
    get_fashion_trends as get_fashion_trends_internal,
    get_shopping_recommendations as get_shopping_recommendations_internal
)


def register_elevenlabs_routes(app):
    """Register all ElevenLabs webhook endpoints with the Flask app"""
    
    @app.route('/elevenlabs/fashion-trends', methods=['POST'])
    def elevenlabs_fashion_trends():
        """
        ElevenLabs webhook endpoint for location-based fashion trends
        Expected body params: location
        """
        try:
            data = request.json
            location = data.get('location', 'San Francisco')
            
            print(f"[ElevenLabs] Fashion trends request for: {location}")
            
            # Call the internal QLOO function
            trends_response = get_fashion_trends_internal(location)
            
            # Parse the response to extract key information
            voice_response = f"Here are the trending fashion styles in {location}. "
            
            # Extract brand and trend info from the response string
            if "Fashion trends in" in trends_response:
                lines = trends_response.split('\n')
                brands = []
                for line in lines:
                    if line.strip().startswith('-') and 'affinity:' in line:
                        # Extract brand name
                        brand_part = line.split('(affinity:')[0].strip('- ').strip()
                        brands.append(brand_part)
                
                if brands:
                    voice_response += f"Popular brands include {', '.join(brands[:3])}. "
            
            return jsonify({
                "response": voice_response,
                "raw_data": trends_response
            })
            
        except Exception as e:
            print(f"[ElevenLabs] Error in fashion trends: {e}")
            return jsonify({
                "response": f"Sorry, I couldn't get fashion trends for that location.",
                "error": str(e)
            }), 500


    @app.route('/elevenlabs/shopping-search', methods=['POST'])
    def elevenlabs_shopping_search():
        """
        ElevenLabs webhook endpoint for shopping recommendations
        Expected body params: location, shopping_need, budget
        """
        try:
            data = request.json
            location = data.get('location', 'San Francisco')
            shopping_need = data.get('shopping_need', 'casual wear')
            budget = data.get('budget', 'medium')
            
            print(f"[ElevenLabs] Shopping search: {shopping_need} in {location}, budget: {budget}")
            
            # Call the internal shopping function
            shopping_response = get_shopping_recommendations_internal(
                location=location,
                shopping_need=shopping_need,
                budget=budget
            )
            
            # Parse the JSON response
            try:
                shopping_data = json.loads(shopping_response)
                recommendations = shopping_data.get('data', {}).get('recommendations', [])
                
                voice_response = f"I found some great {shopping_need} options for you. "
                shopping_items = []
                
                for brand_data in recommendations[:2]:  # Top 2 brands
                    brand = brand_data.get('brand', '')
                    products = brand_data.get('products', [])
                    
                    if products:
                        voice_response += f"{brand} has {len(products)} items. "
                        
                        for product in products[:3]:  # Top 3 products per brand
                            shopping_items.append({
                                'title': product.get('title', ''),
                                'price': product.get('price', ''),
                                'image': product.get('image', ''),
                                'link': product.get('link', ''),
                                'brand': brand,
                                'source': product.get('source', ''),
                                'isShoppingItem': True
                            })
                
                voice_response += "I'll show them on your screen now."
                
                # Return response that ElevenLabs expects
                # The client tool will handle displaying the items
                return jsonify({
                    "response": voice_response,
                    "shopping_items": shopping_items  # This will be passed to client tool
                })
                
            except json.JSONDecodeError:
                # If response is not JSON, return as is
                return jsonify({
                    "response": shopping_response
                })
            
        except Exception as e:
            print(f"[ElevenLabs] Error in shopping search: {e}")
            return jsonify({
                "response": "Sorry, I had trouble searching for shopping options.",
                "error": str(e)
            }), 500


    @app.route('/elevenlabs/outfit-recommendation', methods=['POST'])
    def elevenlabs_outfit_recommendation():
        """
        ElevenLabs webhook endpoint for outfit recommendations
        This endpoint expects the wardrobe data to be passed in the request
        """
        try:
            data = request.json
            occasion = data.get('occasion', 'casual')
            weather = data.get('weather', '')
            wardrobe_data = data.get('wardrobe_data', {})
            
            print(f"[ElevenLabs] Outfit recommendation for {occasion}, weather: {weather}")
            
            # Simple outfit selection logic
            # In production, this would use your existing recommendation logic
            voice_response = f"I've selected a great {occasion} outfit for you. "
            recommended_items = []
            
            # Select items based on occasion
            if wardrobe_data:
                # Try to find appropriate items
                shirts = wardrobe_data.get('shirts', [])
                pants = wardrobe_data.get('pants', [])
                shoes = wardrobe_data.get('shoes', [])
                
                selected_ids = {}
                
                # Select shirt
                if shirts:
                    if occasion == 'formal':
                        # Look for formal shirt
                        formal_shirt = next((s for s in shirts if any(word in s.get('description', '').lower() 
                                           for word in ['dress', 'formal', 'button', 'white'])), None)
                        if formal_shirt:
                            selected_ids['shirt_id'] = formal_shirt.get('id')
                            voice_response += f"Start with the {formal_shirt.get('description', 'formal shirt')}. "
                    else:
                        # Pick first casual shirt
                        selected_ids['shirt_id'] = shirts[0].get('id')
                        voice_response += f"I recommend the {shirts[0].get('description', 'shirt')}. "
                
                # Select pants
                if pants:
                    selected_ids['pant_id'] = pants[0].get('id')
                    voice_response += f"Pair it with {pants[0].get('description', 'pants')}. "
                
                # Select shoes
                if shoes:
                    selected_ids['shoe_id'] = shoes[0].get('id')
                    voice_response += f"Complete the look with {shoes[0].get('description', 'shoes')}."
                
                return jsonify({
                    "response": voice_response,
                    "outfit_ids": selected_ids  # This will be used by client tool
                })
            else:
                return jsonify({
                    "response": "I need access to your wardrobe to make recommendations. Please make sure your wardrobe is loaded."
                })
            
        except Exception as e:
            print(f"[ElevenLabs] Error in outfit recommendation: {e}")
            return jsonify({
                "response": "Sorry, I had trouble creating an outfit recommendation.",
                "error": str(e)
            }), 500


    @app.route('/elevenlabs/weather', methods=['POST'])
    def elevenlabs_weather():
        """
        ElevenLabs webhook endpoint for weather information
        Expected body params: location
        """
        try:
            data = request.json
            location = data.get('location', 'San Francisco')
            
            print(f"[ElevenLabs] Weather request for: {location}")
            
            # Use the existing weather function from langchain agent
            from services.langchain_agent_fixed import get_weather
            weather_response = get_weather(location)
            
            # Add clothing context to the response
            if "°F" in weather_response:
                temp_str = weather_response.split("°F")[0].split()[-1]
                try:
                    temp = float(temp_str)
                    if temp > 80:
                        weather_response += " It's quite warm, so lightweight fabrics would be perfect."
                    elif temp > 60:
                        weather_response += " The temperature is comfortable for most outfits."
                    else:
                        weather_response += " It's a bit cool, so you might want to layer up."
                except:
                    pass
            
            return jsonify({
                "response": weather_response
            })
            
        except Exception as e:
            print(f"[ElevenLabs] Error getting weather: {e}")
            return jsonify({
                "response": f"Sorry, I couldn't get the weather for {location}.",
                "error": str(e)
            }), 500


    print("[ElevenLabs] Webhook endpoints registered successfully")