"""
Style Assistant LLM Service
Provides outfit recommendations based on user queries and wardrobe analysis
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import re

# Import Google GenerativeAI with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI package not installed. Run: pip install google-generativeai")

class StyleAssistantService:
    def __init__(self):
        """Initialize the Style Assistant with Gemini configuration"""
        # Use the same env variable as gemini_service.py
        self.api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI package not available")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.0 Flash for better understanding
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # System prompt for style recommendations
        self.system_prompt = """You are a friendly and conversational style assistant helping users with their digital wardrobe.

Your behavior:
1. Be conversational and natural - not every message needs outfit recommendations
2. For greetings (hi, hello, hey), respond warmly WITHOUT suggesting outfits
3. For general chat, respond appropriately without forcing outfit suggestions
4. Only recommend outfits when the user asks about what to wear or requests styling help
5. Remember the conversation context and build on previous discussions

CRITICAL: You MUST respond ONLY with a JSON object (no other text before or after).

For general conversation (greetings, questions, chat):
{
  "message": "Your conversational response",
  "selections": {},
  "styling_tip": ""
}

For outfit recommendations:
{
  "message": "Your friendly response to the user (2-3 sentences)",
  "selections": {
    "shirt": {
      "id": "exact_id_from_wardrobe",
      "description": "exact description from wardrobe"
    },
    "pants": {
      "id": "exact_id_from_wardrobe", 
      "description": "exact description from wardrobe"
    },
    "shoes": {
      "id": "exact_id_from_wardrobe",
      "description": "exact description from wardrobe"
    }
  },
  "styling_tip": "A brief styling tip or explanation"
}

EXAMPLES:

User: "Hi!"
{
  "message": "Hello! I'm here to help you look your best. How are you doing today? Is there any particular occasion or style you're thinking about?",
  "selections": {},
  "styling_tip": ""
}

User: "I'm good, thanks!"
{
  "message": "Great to hear! I'm here whenever you need help putting together an outfit or want some style advice. Feel free to ask me anything about your wardrobe!",
  "selections": {},
  "styling_tip": ""
}

User: "What should I wear to the beach?"
{
  "message": "For a casual beach day, I recommend a simple yet stylish outfit. This combination will keep you comfortable and cool.",
  "selections": {
    "shirt": {
      "id": "1753425551684_mobile_shirt",
      "description": "white, crew neck, short-sleeve, casual"
    },
    "pants": {
      "id": "1753424587126_mobile_pants",
      "description": "blue, denim, relaxed fit, casual"
    },
    "shoes": {
      "id": "1753426090807_mobile_shoes",
      "description": "white, canvas, lace-up, sneakers"
    }
  },
  "styling_tip": "Roll up the cuffs of your jeans for a relaxed beach vibe. Consider adding sunglasses and a hat for sun protection."
}

Important:
- Respond naturally to greetings and general conversation
- Only suggest outfits when explicitly asked about clothing/style
- Use the EXACT id and description from the wardrobe when making selections
- If a category is not needed, omit it from selections
- Always maintain proper JSON format
- Consider conversation context from previous messages
"""

    def analyze_wardrobe(self, clothing_items: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """
        Analyze wardrobe and extract key features for matching
        
        Args:
            clothing_items: Dict with 'shirts', 'pants', 'shoes' containing item data
            
        Returns:
            Dict with categorized descriptions
        """
        wardrobe_summary = {
            'shirts': [],
            'pants': [],
            'shoes': []
        }
        
        for category in ['shirts', 'pants', 'shoes']:
            if category in clothing_items:
                # Debug: check first few items
                if category == 'shirts' and len(clothing_items[category]) > 0:
                    print(f"\nDEBUG: First 5 shirts in wardrobe:")
                    for i, item in enumerate(clothing_items[category][:5]):
                        print(f"  {i}: id={item.get('id')}, desc={item.get('description')}")
                
                for item in clothing_items[category]:
                    if item.get('description'):
                        wardrobe_summary[category].append({
                            'id': item['id'],
                            'description': item['description'],
                            'source_image': item.get('source_image', '')
                        })
        
        return wardrobe_summary

    def parse_gemini_response(self, response_text: str) -> Dict:
        """
        Parse Gemini's JSON response
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed JSON object or error dict
        """
        try:
            # Try to extract JSON from the response
            # Sometimes Gemini adds extra text before/after JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, return a structured response
                return {
                    "message": response_text,
                    "selections": {},
                    "styling_tip": ""
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response as JSON: {e}")
            print(f"Response text: {response_text}")
            return {
                "message": response_text,
                "selections": {},
                "styling_tip": ""
            }

    def get_outfit_recommendation(self, user_query: str, clothing_items: Dict[str, List[Dict]], conversation_history: List[Dict] = None) -> Dict:
        """
        Get outfit recommendations based on user query and wardrobe
        
        Args:
            user_query: User's question about what to wear
            clothing_items: User's wardrobe items with descriptions
            
        Returns:
            Dict with recommendation text and matched items
        """
        try:
            print(f"\n=== STYLE ASSISTANT DEBUG ===")
            print(f"Query: {user_query}")
            print(f"Clothing items received:")
            print(f"  Shirts: {len(clothing_items.get('shirts', []))}")
            print(f"  Pants: {len(clothing_items.get('pants', []))}")
            print(f"  Shoes: {len(clothing_items.get('shoes', []))}")
            
            # Analyze wardrobe
            wardrobe_summary = self.analyze_wardrobe(clothing_items)
            
            # Create wardrobe context for LLM with IDs
            wardrobe_data = {
                "shirts": [],
                "pants": [],
                "shoes": []
            }
            
            for shirt in wardrobe_summary.get('shirts', []):
                wardrobe_data['shirts'].append({
                    "id": shirt['id'],
                    "description": shirt['description']
                })
            
            for pants in wardrobe_summary.get('pants', []):
                wardrobe_data['pants'].append({
                    "id": pants['id'],
                    "description": pants['description']
                })
            
            for shoes in wardrobe_summary.get('shoes', []):
                wardrobe_data['shoes'].append({
                    "id": shoes['id'],
                    "description": shoes['description']
                })
            
            # Debug: show first few items being sent
            print(f"\n=== WARDROBE DATA BEING SENT TO GEMINI ===")
            print(f"First 3 shirts: {wardrobe_data['shirts'][:3]}")
            print(f"First 3 pants: {wardrobe_data['pants'][:3]}")
            print(f"First 3 shoes: {wardrobe_data['shoes'][:3]}")
            
            # Build conversation context if history exists
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "\n\nPrevious conversation:\n"
                # Include last 10 messages for context (5 exchanges)
                recent_history = conversation_history[-10:]
                for msg in recent_history:
                    role = "User" if msg.get('role') == 'user' else "Assistant"
                    conversation_context += f"{role}: {msg.get('content', '')}\n"
            
            # Create prompt with structured wardrobe data and conversation history
            prompt = f"""{self.system_prompt}

{conversation_context}

User's Wardrobe (with IDs):
{json.dumps(wardrobe_data, indent=2)}

Current User Query: {user_query}

IMPORTANT: First understand what the user is asking:
- If it's a greeting or casual conversation, respond conversationally without outfit suggestions
- If they're asking about clothes/outfits/what to wear, then provide specific recommendations
- Use conversation history to understand context (e.g., "show me something more formal" refers to previous suggestions)

Remember to respond with a properly formatted JSON object. Only include outfit selections when the user is actually asking for clothing recommendations."""

            # Get recommendation from Gemini
            response = self.model.generate_content(prompt)
            recommendation_text = response.text
            
            print(f"\n=== GEMINI RAW RESPONSE ===")
            print(recommendation_text)
            print("=== END GEMINI RESPONSE ===\n")
            
            # Parse the JSON response
            parsed_response = self.parse_gemini_response(recommendation_text)
            
            # Extract selected items and find their images
            final_items = []
            selections = parsed_response.get('selections', {})
            
            print(f"\n=== MATCHING SELECTIONS ===")
            print(f"Selections from Gemini: {list(selections.keys())}")
            
            for category, item_data in selections.items():
                if isinstance(item_data, dict) and 'id' in item_data:
                    # Find the actual item in clothing_items
                    # Handle pluralization correctly
                    category_plural = category
                    if category == 'shirt':
                        category_plural = 'shirts'
                    elif category == 'pants':
                        category_plural = 'pants'  # already plural
                    elif category == 'shoes':
                        category_plural = 'shoes'  # already plural
                    
                    print(f"  Looking for {category} with id: {item_data['id']}")
                    print(f"  Searching in {len(clothing_items.get(category_plural, []))} {category_plural}")
                    
                    found = False
                    for item in clothing_items.get(category_plural, []):
                        if item['id'] == item_data['id']:
                            print(f"  ✓ Found matching {category}!")
                            final_items.append({
                                'category': category,
                                'id': item['id'],
                                'image': item['image'],
                                'description': item.get('description', ''),
                                'source_image': item.get('source_image', '')
                            })
                            found = True
                            break
                    
                    if not found:
                        print(f"  ✗ Could not find {category} with id: {item_data['id']}")
            
            # Combine message and styling tip for display
            display_message = parsed_response.get('message', '')
            
            # Add descriptions of selected items in a natural way
            if len(final_items) > 0:
                item_descriptions = []
                for item in final_items:
                    desc = item.get('description', '')
                    if desc:
                        item_descriptions.append(f"{desc} {item['category']}")
                
                if item_descriptions:
                    display_message += f"\n\nI've selected: {', '.join(item_descriptions)}."
            
            if parsed_response.get('styling_tip'):
                display_message += f"\n\n{parsed_response['styling_tip']}"
            
            print(f"\n=== RETURNING TO FRONTEND ===")
            print(f"Number of matched items: {len(final_items)}")
            for item in final_items:
                print(f"  {item['category']}: {item['id']}")
            
            return {
                'success': True,
                'recommendation': display_message,
                'matched_items': final_items,
                'query': user_query
            }
            
        except Exception as e:
            print(f"Error in style recommendation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'recommendation': "I'm having trouble accessing your wardrobe right now. Please try again.",
                'matched_items': []
            }

    def get_greeting_message(self) -> str:
        """Get a friendly greeting message for the chat"""
        return """Hello! I'm your personal style assistant. I can help you choose the perfect outfit from your wardrobe.

Try asking me questions like:
- "What should I wear for a sunny day?"
- "Suggest an outfit for a business meeting"
- "What goes well with my blue jeans?"
- "Help me pick something casual for the weekend"

What outfit are you looking for today?"""