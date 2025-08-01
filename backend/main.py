"""
Firebase Functions entry point for KapdaAI Backend
"""
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Set API mode environment variable for Firebase deployment
os.environ['USE_HF_API'] = 'true'

import functions_framework
from flask_api import app

@functions_framework.http
def backend(request):
    """
    Firebase Functions entry point
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Create a Flask context and dispatch the request
    with app.request_context(request.environ):
        try:
            # Use Flask's routing
            response = app.full_dispatch_request()
            
            # Add CORS headers to all responses
            if hasattr(response, 'headers'):
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            
            return response
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            # Return error with CORS headers
            return {'error': str(e)}, 500, {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            }

# For local testing
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)