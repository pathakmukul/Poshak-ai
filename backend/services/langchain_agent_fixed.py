"""
KapdaAI LangChain Agent - Fixed Implementation
Uses proper LangChain with tool execution
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool, StructuredTool
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

load_dotenv()

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'poshakai')
LOCATION = os.getenv('LOCATION', 'us-central1')
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5001')
QLOO_API_KEY = os.getenv('QLOO_API_KEY', '')
QLOO_API_URL = os.getenv('QLOO_API_URL', 'https://hackathon.api.qloo.com')


# Tool Functions
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    print(f"\n[TOOL CALL] get_weather(location='{location}')")
    try:
        # Clean location string - remove quotes if present
        location = location.strip().strip('"').strip("'")
        if location.startswith("location="):
            location = location.replace("location=", "").strip()
        
        # Get coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        print(f"[WEATHER API] Calling geocoding API: {geo_url}?name={location}&count=1")
        geo_resp = requests.get(geo_url, params={"name": location, "count": 1})
        print(f"[WEATHER API] Geocoding response: {geo_resp.status_code}")
        
        if geo_resp.status_code == 200 and geo_resp.json().get("results"):
            geo_data = geo_resp.json()
            geo_size = len(json.dumps(geo_data))
            print(f"[WEATHER API] Geocoding response size: {geo_size} bytes")
            coords = geo_data["results"][0]
            
            # Get weather
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "current": ["temperature_2m", "weather_code"],
                "timezone": "auto"
            }
            print(f"[WEATHER API] Calling weather API: {weather_url}")
            print(f"[WEATHER API] Weather params: {weather_params}")
            weather_resp = requests.get(weather_url, params=weather_params)
            print(f"[WEATHER API] Weather response: {weather_resp.status_code}")
            
            if weather_resp.status_code == 200:
                weather_data = weather_resp.json()
                weather_size = len(json.dumps(weather_data))
                print(f"[WEATHER API] Weather response size: {weather_size} bytes")
                data = weather_data["current"]
                
                # Weather code mapping
                weather_codes = {
                    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
                    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow"
                }
                
                temp_c = data["temperature_2m"]
                temp_f = (temp_c * 9/5) + 32
                
                result = f"Weather in {location}: {round(temp_f, 1)}°F, {weather_codes.get(data['weather_code'], 'Unknown')}"
                
                # Size comparison
                total_api_size = geo_size + weather_size
                result_size = len(result)
                print(f"\n[WEATHER API] SIZE COMPARISON:")
                print(f"  Total API responses: {total_api_size} bytes")
                print(f"  Final tool response: {result_size} bytes")
                print(f"  Reduction: {((total_api_size - result_size) / total_api_size * 100):.1f}%")
                print(f"[WEATHER API] Final result: {result}")
                return result
        
        return f"Could not get weather for {location}"
        
    except Exception as e:
        print(f"[WEATHER API] Error: {e}")
        return f"Weather error: {str(e)}"


# Global wardrobe storage for current request
current_wardrobe_data = {}
selected_indices = {}  # DEPRECATED - use selected_ids instead
selected_ids = {}  # Store selected item IDs
qloo_trends = {}  # Store QLOO insights for current request

def get_wardrobe(user_id: str = "omXo3iWdEeTCbCusQVIvIAeSaGG2") -> str:
    """Get user's wardrobe items from the client-provided data."""
    print(f"\n[TOOL CALL] get_wardrobe(user_id='{user_id}')")
    global current_wardrobe_data
    
    if not current_wardrobe_data:
        return "No wardrobe data available"
    
    # Create summary WITH IDs
    summary = []
    for category in ["shirts", "pants", "shoes"]:
        items = current_wardrobe_data.get(category, [])
        if items:
            summary.append(f"\n{category.upper()} ({len(items)} items):")
            for i, item in enumerate(items):  # Show ALL items with IDs
                item_id = item.get('id', f'item_{i}')
                desc = item.get('description', 'No description')
                summary.append(f"  ID: {item_id}")
                summary.append(f"     Description: {desc}")
    
    return "\n".join(summary) if summary else "No items in wardrobe"


def get_fashion_trends(location: str, weather_condition: str = None) -> str:
    """Get trending fashion items for a location using QLOO API."""
    print(f"\n[TOOL CALL] get_fashion_trends(location='{location}', weather='{weather_condition}')")
    global qloo_trends
    
    if not QLOO_API_KEY:
        print("[QLOO API] No API key configured")
        return "QLOO API key not configured"
    
    try:
        # Use QLOO Insights API to get fashion brand trends
        headers = {'X-Api-Key': QLOO_API_KEY}
        print(f"[QLOO API] Using API key: {QLOO_API_KEY[:10]}...")
        
        # Get popular fashion brands - only request what we need
        params = {
            'filter.type': 'urn:entity:brand',
            'filter.tags': 'urn:tag:genre:brand:fashion',
            'signal.location.query': location,  # ADD LOCATION SIGNAL
            'limit': 5,  # Get 5 to have better selection
            'sort': 'popularity'
        }
        
        url = f"{QLOO_API_URL}/v2/insights"
        print(f"[QLOO API] Calling: {url}")
        print(f"[QLOO API] Headers: {{'X-Api-Key': '{QLOO_API_KEY[:10]}...'}}")
        print(f"[QLOO API] Params: {params}")
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        print(f"[QLOO API] Response status: {response.status_code}")
        print(f"[QLOO API] Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            # Don't print full response - it's huge!
            original_size = len(response.text)
            print(f"[QLOO API] Original response size: {original_size} bytes, Brands: {len(data.get('results', {}).get('entities', []))}")
            
            # IMMEDIATELY extract only essential data to avoid context explosion
            brands_raw = data.get('results', {}).get('entities', [])
            brands = []
            
            # Process and slim down the data
            for brand in brands_raw[:5]:  # Only process top 5
                # Extract key style tags (not all tags!)
                tags = brand.get('tags', [])
                style_tags = []
                for tag in tags[:10]:  # Check first 10 tags only
                    tag_name = tag.get('name', '').lower()
                    if any(keyword in tag_name for keyword in ['casual', 'formal', 'athletic', 'luxury', 'streetwear', 'footwear', 'clothing']):
                        style_tags.append(tag_name)
                        if len(style_tags) >= 3:  # Max 3 style tags
                            break
                
                # Create slim brand object
                slim_brand = {
                    'name': brand.get('name', ''),
                    'popularity': brand.get('popularity', 0.0),
                    'style_tags': style_tags[:3]  # Only top 3 relevant tags, NO full tags!
                }
                brands.append(slim_brand)
            
            print(f"[QLOO API] Processed data size: ~{len(str(brands))} bytes (98% reduction!)")
            
            # Extract trending fashion insights with affinity scores
            trending_items = []
            trending_brands = []
            
            # Print raw data for debugging
            print(f"\n[QLOO API] Location-specific results for {location}:")
            
            for i, brand in enumerate(brands[:5]):  # Top 5 brands
                brand_name = brand['name']
                affinity_score = brand['popularity']
                style_tags = brand.get('style_tags', [])
                
                # Print each brand's details
                print(f"\n[QLOO API] Brand #{i+1}:")
                print(f"  Name: {brand_name}")
                print(f"  Affinity Score: {affinity_score}")
                print(f"  Style Tags: {style_tags}")
                
                # Build slim brand object for storage
                trending_brands.append({
                    'name': brand_name,
                    'affinity': affinity_score,
                    'style_tags': style_tags  # Only 3 tags instead of all!
                })
                
                # Collect unique style tags
                for tag in style_tags:
                    if tag not in trending_items:
                        trending_items.append(tag)
            
            # Just use what QLOO returns, no hardcoded suggestions
            
            # Store trends with affinity data
            qloo_trends = {
                "location": location,
                "trending_items": trending_items[:6],  # Top 6 items
                "trending_brands": trending_brands[:5],  # Store all 5 brands with affinity
                "timestamp": "current"
            }
            
            # Return a more natural, helpful response with affinity scores
            result = f"Fashion trends in {location}:\n"
            
            # Show top 3 brands with affinity scores
            for brand in trending_brands[:3]:
                result += f"- {brand['name']} (affinity: {brand['affinity']:.2f} - "
                if brand['affinity'] > 0.8:
                    result += "very popular"
                elif brand['affinity'] > 0.6:
                    result += "popular"
                else:
                    result += "moderately popular"
                result += f" in {location})\n"
            
            # Only include meaningful style categories
            meaningful_styles = [item for item in trending_items if len(item.split()) <= 2 and 'fashion' not in item]
            if meaningful_styles:
                result += f"\nTrending styles: {', '.join(meaningful_styles[:3])}"
            
            # CRITICAL: Print size comparison
            original_response_size = original_size
            final_response_size = len(result)
            reduction_percent = ((original_response_size - final_response_size) / original_response_size) * 100
            
            print(f"\n[QLOO API] SIZE COMPARISON:")
            print(f"  Original QLOO response: {original_response_size:,} bytes")
            print(f"  Final tool response: {final_response_size} bytes")
            print(f"  Reduction: {reduction_percent:.1f}%")
            print(f"\n[QLOO API] Final formatted response:\n{result}")
            
            return result
            
        else:
            print(f"[QLOO API] Error response: {response.status_code}")
            print(f"[QLOO API] Error body: {response.text}")
            return f"Could not fetch fashion trends from QLOO API (status: {response.status_code})"
        
    except Exception as e:
        print(f"[ERROR] QLOO API error: {e}")
        return f"Could not fetch fashion trends: {str(e)}"



def get_shopping_recommendations(location: str, shopping_need: str = None, budget: str = "medium") -> str:
    """Get shopping recommendations with actual products using QLOO + SerpAPI
    
    Args:
        location: City/location for trends
        shopping_need: What user is looking for (e.g., "beach clothes", "office wear", "athletic gear")
        budget: User's budget level
    """
    print(f"\n[TOOL CALL] get_shopping_recommendations(location='{location}', need='{shopping_need}', budget='{budget}')")
    
    global qloo_trends
    
    # Step 1: Get location-specific brands from QLOO (or use cached)
    if not qloo_trends or qloo_trends.get("location") != location:
        trends_response = get_fashion_trends(location)
    
    trending_brands = qloo_trends.get("trending_brands", [])
    
    if not trending_brands:
        return "Unable to get trending brands from QLOO"
    
    # Step 2: Use Serper API to find actual products
    try:
        recommendations = []
        serper_api_key = os.getenv('SERPER_API_KEY', os.getenv('SERPAPI_KEY', ''))  # Support both env var names
        
        if not serper_api_key:
            # If no Serper API, just return QLOO brands
            result = f"Shopping recommendations for {location}:\n\n"
            for brand in trending_brands[:3]:
                result += f"- {brand['name']} (popularity: {brand['affinity']:.2f})\n"
                if brand.get('style_tags'):
                    result += f"  Styles: {', '.join(brand['style_tags'])}\n"
            return result
        
        # For top 3 trending brands, search for products
        for brand in trending_brands[:3]:
            # Build search query: brand + need + location
            search_query = f"{brand['name']}"
            if shopping_need:
                search_query += f" {shopping_need}"
            
            print(f"[SERPER] Searching for: {search_query}")
            
            # Search Google Shopping using Serper API
            url = "https://google.serper.dev/shopping"
            headers = {
                'X-API-KEY': serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                "q": search_query,
                "location": location,
                "num": 3  # Get top 3 products per brand
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                results = response.json()
                shopping_results = results.get("shopping", [])
            
                brand_products = {
                    "brand": brand['name'],
                    "popularity": brand['affinity'],
                    "style_tags": brand.get('style_tags', []),
                    "products": []
                }
                
                # Extract product info from Serper API response
                for product in shopping_results[:3]:
                    brand_products["products"].append({
                        "title": product.get("title", ""),
                        "price": product.get("price", ""),
                        "source": product.get("source", ""),
                        "link": product.get("link", ""),
                        "image": product.get("imageUrl", product.get("image", ""))  # Serper may use imageUrl or image
                    })
                
                recommendations.append(brand_products)
            else:
                print(f"[SERPER] Error {response.status_code}: {response.text}")
        
        # Return structured data as JSON string
        shopping_data = {
            "location": location,
            "shopping_need": shopping_need or 'fashion items',
            "recommendations": recommendations
        }
        
        # Also include a formatted text summary for the LLM to use
        text_summary = f"Found shopping recommendations for {shopping_need or 'fashion items'} in {location}. "
        text_summary += f"Top brands: {', '.join([r['brand'] for r in recommendations[:3]])}. "
        text_summary += f"Total products found: {sum(len(r['products']) for r in recommendations)}."
        
        # Return both structured data and text summary
        return json.dumps({
            "summary": text_summary,
            "data": shopping_data
        })
        
    except Exception as e:
        print(f"[ERROR] Shopping search failed: {e}")
        return f"Could not fetch product recommendations: {str(e)}"


def format_structured_response(
    text: str,
    wardrobe_ids: Optional[Dict[str, str]] = None,
    additional_text: Optional[str] = None
) -> str:
    """
    REQUIRED: Format ALL responses using this structure.
    
    Args:
        text: Your main response text
        wardrobe_ids: When recommending outfits, use EXACT IDs from wardrobe
                     e.g. {"shirts": "image1_shirt", "pants": "image2_pants", "shoes": "image3_shoes"}
        additional_text: Any additional text after outfit items
    
    Returns:
        JSON formatted response
    """
    response = {
        "text": text,
        "wardrobe_ids": wardrobe_ids or {},
        "additional_text": additional_text or ""
    }
    
    return json.dumps(response)


# Create LangChain tools
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Get current weather for a location. Use when user mentions a city."
)

wardrobe_tool = Tool(
    name="get_wardrobe",
    func=get_wardrobe,
    description="Get user's wardrobe items"
)

fashion_trends_tool = Tool(
    name="get_fashion_trends",
    func=get_fashion_trends,
    description="Get location-specific trending fashion brands with popularity scores. ALWAYS use this when user mentions a location to get brands that are actually popular in that specific city. Returns brand names with affinity scores showing how popular they are in that location."
)

# Removed outfit_tool - LLM should use individual tools and make its own recommendations

# Create Pydantic model for shopping tool
class ShoppingRecommendationsInput(BaseModel):
    location: str = Field(description="City/location for trends")
    shopping_need: Optional[str] = Field(default=None, description="What user is looking for (e.g., 'beach clothes', 'office wear', 'athletic gear')")
    budget: Optional[str] = Field(default="medium", description="User's budget level: 'low', 'medium', or 'high'")

shopping_tool = StructuredTool.from_function(
    func=get_shopping_recommendations,
    name="get_shopping_recommendations",
    description="Get shopping recommendations with actual products based on location trends. Use when user asks about shopping or buying clothes. This tool uses QLOO to find trending brands and optionally searches for real products.",
    args_schema=ShoppingRecommendationsInput
)

# Create Pydantic model for structured tool
class FormatResponseInput(BaseModel):
    text: str = Field(description="Your main response message")
    wardrobe_ids: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of exact wardrobe item IDs when recommending outfits. Example: {'shirts': 'image1_shirt', 'pants': 'image2_pants', 'shoes': 'image3_shoes'}"
    )
    additional_text: Optional[str] = Field(
        default=None,
        description="Any additional suggestions or notes"
    )
    
    model_config = {"extra": "forbid"}  # Updated for Pydantic v2

format_response_tool = StructuredTool.from_function(
    func=format_structured_response,
    name="format_structured_response",
    description="REQUIRED: Format your final response. When recommending outfits, include the exact wardrobe item IDs you selected. Your response after this tool MUST be ONLY the JSON output - no additional text!",
    args_schema=FormatResponseInput
)

# Create agent
def create_langchain_agent():
    """Create LangChain agent with tools"""
    
    # Initialize LLM
    llm = ChatVertexAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        project=PROJECT_ID,
        location=LOCATION
    )
    
    # Tools
    tools = [weather_tool, wardrobe_tool, fashion_trends_tool, shopping_tool]  # Removed format_response_tool
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are KapdaAI, a fashion assistant. Who helps users with their wardrobe and fashion needs. You recommend what to wear to users based on their wardrobe. You have ability to recommend outfits based on weather and location. When user provides a location, you can call weather api for that location and get weather information. You can also call QLOO api to get trending fashion items for that location. These are this agents main features. so use them when user mentions a location.
         Be cautious of recommendation. You can ask user occasion as well to better understand their needs. But dont ask user about weather or things you already have access to. 
        As a final response, you will reply with a message talking about the outfit you recommended, give details why this works. And ingrain knowledge gained from the QLOO API to make response look better. If you feel user may not have the items in their wardrobe, you can recommend them to buy them based on QLOO suggestions for that location. But then again, always recommend never say that you dont have the items in their wardrobe so buy. bad strategy. buying recommendation comes as an add on message. You dont have to recommend whole outfit purchase but an item or two is fine as well.
        
When users ask about shopping or buying clothes, ALWAYSuse the get_shopping_recommendations tool to find actual products. thsi is very important..
        
CRITICAL FORMAT FOR FINAL RESPONSE:
Your final response MUST be in this EXACT JSON format for images to display:
```json
{{
  "text": "Your outfit recommendation message here with details about why this works",
  "wardrobe_ids": {{
    "shirts": "exact_shirt_id_from_wardrobe",
    "pants": "exact_pant_id_from_wardrobe", 
    "shoes": "exact_shoe_id_from_wardrobe"
  }},
  "additional_text": "Optional styling/shopping suggestions or extra tips based on QLOO response if available or needed"
}}
```

IMPORTANT: Use the EXACT IDs from get_wardrobe() output, NOT descriptions in wardrobe_ids. ALSO Dont include wardrobe_ids in the text or additional_text fields as it would look ugly to the user.

TOOLS AND HOW TO USE THEM:

1. get_wardrobe() - Returns all user's clothing items
   Example: get_wardrobe()

2. get_weather(location) - Gets weather for a city
   Example: get_weather(location="Miami")
   Example: get_weather(location="San Jose")

3. get_fashion_trends(location) - Gets location-specific trending brands from QLOO with affinity scores
   Example: get_fashion_trends(location="New York")
   Returns brands with popularity scores (0.8+ = very popular, 0.6-0.8 = popular in that location)

4. get_shopping_recommendations(location, shopping_need, budget) - Gets shopping recommendations
   Example: get_shopping_recommendations(location="Miami", shopping_need="beach clothes", budget="medium")
   Use when user asks about shopping or what to buy. Returns trending brands for that location.

HOW TO HANDLE DIFFERENT REQUESTS:

FOR OUTFIT RECOMMENDATIONS (from wardrobe):
1. Call get_wardrobe() to see all available items with their IDs
2. If user mentions a location, call get_weather() and get_fashion_trends()
3. Select appropriate items from wardrobe
4. Return response with wardrobe_ids filled in

FOR SHOPPING RECOMMENDATIONS: (always use this when user asks about shopping or buying clothes)
1. Call get_shopping_recommendations() with location and shopping_need
2. Include the shopping results in your response
3. Leave wardrobe_ids empty since these are not wardrobe items"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# No Flask imports needed - this is now a service module

# Global agent factory (we'll create per-session agents)
def get_or_create_agent(session_id: str = "default"):
    """Get or create an agent for a specific session"""
    if not hasattr(get_or_create_agent, "agents"):
        get_or_create_agent.agents = {}
    
    if session_id not in get_or_create_agent.agents:
        get_or_create_agent.agents[session_id] = create_langchain_agent()
    
    return get_or_create_agent.agents[session_id]


def handle_agent_chat(data):
    """Handle agent chat request"""
    global current_wardrobe_data
    
    try:
        message = data.get('message', '')
        user_id = data.get('user_id', 'omXo3iWdEeTCbCusQVIvIAeSaGG2')
        wardrobe_data = data.get('wardrobe_data', {})
        session_id = data.get('session_id', user_id)  # Use user_id as session_id if not provided
        
        # Store wardrobe data for tools to access
        current_wardrobe_data = wardrobe_data
        global selected_indices, selected_ids
        selected_indices = {}  # Reset for this request
        selected_ids = {}  # Reset IDs for this request!
        
        # Get agent for this session
        agent_executor = get_or_create_agent(session_id)
        
        # Debug: Check available tools
        if hasattr(agent_executor, 'tools'):
            tool_names = [tool.name for tool in agent_executor.tools]
            print(f"[DEBUG] Available tools: {tool_names}")
        
        # Debug: Check if this is a shopping request
        if any(word in message.lower() for word in ['buy', 'shopping', 'purchase']):
            print(f"[DEBUG] Shopping request detected: '{message}'")
        
        # Debug: Check memory size
        if hasattr(agent_executor, 'memory') and agent_executor.memory:
            memory_buffer = agent_executor.memory.buffer
            if isinstance(memory_buffer, list):
                print(f"\n[DEBUG] Memory buffer has {len(memory_buffer)} messages")
                buffer_str = str(memory_buffer)
                print(f"[DEBUG] Memory buffer size: {len(buffer_str)} bytes")
                # Show first 200 chars of buffer
                print(f"[DEBUG] Buffer preview: {buffer_str[:200]}...")
            else:
                print(f"[DEBUG] Memory buffer type: {type(memory_buffer)}")
        
        # Execute agent
        print(f"[DEBUG] User message: '{message}'")
        
        # Try to catch what's being sent to LLM
        try:
            # Get the full input that would be sent
            full_input = {
                "input": message,
                "chat_history": agent_executor.memory.buffer if hasattr(agent_executor, 'memory') else [],
                "agent_scratchpad": []  # This gets populated during execution
            }
            print(f"[DEBUG] Initial context size: {len(str(full_input))} bytes")
        except:
            pass
        
        result = agent_executor.invoke({
            "input": message
        })
        
        # Extract output
        agent_output = result.get('output', '')
        print(f"[DEBUG] Agent output: {agent_output[:200]}...")
        
        # Parse JSON response
        try:
            # Clean up markdown if present
            cleaned_output = agent_output.strip()
            
            # Find the last JSON block (the actual response, not tool output)
            json_blocks = []
            
            # Look for all ```json blocks
            start_pos = 0
            while True:
                json_start = cleaned_output.find('```json', start_pos)
                if json_start == -1:
                    break
                json_start += 7
                json_end = cleaned_output.find('```', json_start)
                if json_end != -1:
                    json_blocks.append(cleaned_output[json_start:json_end].strip())
                start_pos = json_end + 3
            
            # Use the last JSON block (the agent's final response)
            if json_blocks:
                cleaned_output = json_blocks[-1]
            elif '```json' in cleaned_output:
                # Fallback to original logic
                start = cleaned_output.find('```json') + 7
                end = cleaned_output.find('```', start)
                if end != -1:
                    cleaned_output = cleaned_output[start:end].strip()
            elif '```' in cleaned_output:
                # Extract content between ``` and ```
                start = cleaned_output.find('```') + 3
                end = cleaned_output.find('```', start)
                if end != -1:
                    cleaned_output = cleaned_output[start:end].strip()
            
            # Additional cleanup for any remaining backticks
            cleaned_output = cleaned_output.strip('`').strip()
            
            # Parse JSON
            parsed_response = json.loads(cleaned_output)
            
            # Handle wrapped response (format_structured_response_response)
            if 'format_structured_response_response' in parsed_response:
                parsed_response = parsed_response['format_structured_response_response']
            
            response_text = parsed_response.get('text', '')
            wardrobe_ids = parsed_response.get('wardrobe_ids', {})
            additional_text = parsed_response.get('additional_text', '')
            
            # Build matched items from IDs
            matched_items = []
            if wardrobe_ids and current_wardrobe_data:
                print(f"[DEBUG] Using wardrobe IDs: {wardrobe_ids}")
                for category, item_id in wardrobe_ids.items():
                    if category in current_wardrobe_data and item_id:
                        items = current_wardrobe_data.get(category, [])
                        # Find item by ID
                        item_found = False
                        for item in items:
                            if item.get('id') == item_id:
                                print(f"[DEBUG] Found {category} with ID {item_id}: {item.get('description')}")
                                matched_items.append({
                                    **item,
                                    'type': category.rstrip('s')
                                })
                                item_found = True
                                break
                        if not item_found:
                            print(f"[DEBUG] WARNING: Could not find {category} with ID {item_id}")
                print(f"[DEBUG] Total matched items: {len(matched_items)}")
            
            # Check if response contains shopping data in additional_text
            shopping_items = []
            if additional_text and "SHOPPING_JSON:" in additional_text:
                try:
                    # Extract JSON data after SHOPPING_JSON:
                    json_start = additional_text.find("SHOPPING_JSON:") + len("SHOPPING_JSON:")
                    json_str = additional_text[json_start:].strip()
                    shopping_data = json.loads(json_str)
                    
                    # Extract products from the structured data
                    if 'data' in shopping_data and 'recommendations' in shopping_data['data']:
                        for brand_data in shopping_data['data']['recommendations']:
                            brand_name = brand_data.get('brand', '')
                            for product in brand_data.get('products', []):
                                shopping_items.append({
                                    'id': f'shopping_{len(shopping_items)}',
                                    'description': product.get('title', ''),
                                    'brand': brand_name,
                                    'price': product.get('price', ''),
                                    'type': 'shopping',
                                    'image': product.get('image', '/api/placeholder/200/200'),
                                    'link': product.get('link', ''),
                                    'source': product.get('source', ''),
                                    'isShoppingItem': True
                                })
                    
                    print(f"[DEBUG] Extracted {len(shopping_items)} shopping items with images")
                    
                    # Remove the JSON data from additional_text for display
                    additional_text = additional_text[:additional_text.find("SHOPPING_JSON:")].strip()
                    
                except Exception as e:
                    print(f"[DEBUG] Error parsing shopping JSON data: {e}")
            
            # Combine text for frontend
            full_response = response_text
            if additional_text:
                full_response += f"\n\n{additional_text}"
            
            # Use shopping_items if found, otherwise use matched wardrobe items
            items_to_display = shopping_items if shopping_items else matched_items
            
            return {
                "success": True,
                "response": full_response,
                "matched_items": items_to_display,
                "user_id": user_id
            }
            
        except json.JSONDecodeError:
            # Fallback if agent didn't return JSON
            print("[DEBUG] Agent didn't return JSON, using raw output")
            return {
                "success": True,
                "response": agent_output,
                "matched_items": [],
                "user_id": user_id
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def clear_agent_session(session_id):
    """Clear chat history for a specific session"""
    if hasattr(get_or_create_agent, "agents") and session_id in get_or_create_agent.agents:
        # Remove the agent for this session
        del get_or_create_agent.agents[session_id]
        return {"success": True, "message": f"Session {session_id} cleared"}
    return {"success": False, "message": "Session not found"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test agent")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing LangChain Agent")
        print("="*50)
        
        agent = create_langchain_agent()
        
        test_queries = [
            "Hello!",
            "What should I wear in New York today?",
            "I need a formal outfit for Miami"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            result = agent.invoke({"input": query})
            print(f"Agent: {result['output']}")
    
    else:
        print("Use --test")