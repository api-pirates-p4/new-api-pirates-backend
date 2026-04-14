"""
=============================================================================
PVO Veterans Chatbot API
=============================================================================
Dedicated API for the Poway Veterans Organization virtual assistant chatbot.

SETUP REQUIRED:
1. Ensure GROQ_API_KEY is set in .env file
2. This API provides Anthropic-compatible response format for frontend compatibility

ENDPOINT PROVIDED:
- POST /api/chat - Chat with PVO virtual assistant

USAGE EXAMPLE (JavaScript frontend):
    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
            messages: [
                { role: 'user', content: 'Hello!' }
            ]
        })
    })
=============================================================================
"""

from __init__ import app
from flask import Blueprint, request, jsonify, current_app
from flask_restful import Api, Resource
import requests
import os

# Blueprint setup
pvo_chatbot_api = Blueprint('pvo_chatbot_api', __name__, url_prefix='/api')
api = Api(pvo_chatbot_api)

# =============================================================================
# CONFIGURATION
# =============================================================================

SYSTEM_PROMPT = """You are a warm, respectful virtual assistant for the Poway Veterans Organization (PVO), a 501c3 non-profit in Poway, CA. Help veterans and families, answer questions about PVO services, and guide users to the right website pages.
ABOUT PVO: Assists veterans and dependents in Poway, Ramona, and surrounding areas with financial need due to illness, injury, unemployment or hardship. Founded 2014. 95%+ of funds go to veteran assistance. All volunteers. Phone: (858) 206-8854. Email: contact@powayveterans.org. Address: PO Box 563, Poway CA 92064.
SERVICES: Home repairs, Medical, Counseling, Legal, Transportation (vehicle registration aid and transit vouchers), Family support, Meals and food, VA Services navigation, Other hardship needs.
KEY PAGES:
- Home: https://powayveterans.org/
- About: https://powayveterans.org/about-pvo/
- Apply For Assistance: https://powayveterans.org/request-assistance/
- Donate: https://powayveterans.org/donate/
- Volunteer: https://powayveterans.org/volunteer/
- Scholarships: https://powayveterans.org/scholarships/
- Events: https://powayveterans.org/calendar/
- Golf Tournament: https://powayveterans.org/annual-golf-tournament/
- Newsletter: https://powayveterans.org/newsletter/
- Blog: https://powayveterans.org/blog-2/
RULES: Be warm and respectful. Direct veterans needing help to https://powayveterans.org/request-assistance/ . Use Markdown links [text](url). If unsure, say to call (858) 206-8854. Do not invent information."""

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000


def get_groq_server():
    """Get Groq server URL from app config"""
    return current_app.config.get('GROQ_SERVER')


def get_groq_api_key():
    """Get Groq API key from app config or environment"""
    return current_app.config.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')


# =============================================================================
# ENDPOINTS
# =============================================================================

class _Chat(Resource):
    """
    PVO Chatbot endpoint - POST /api/chat
    Returns Anthropic-compatible response format for frontend compatibility.
    """
    def post(self):
        try:
            api_key = get_groq_api_key()
            if not api_key:
                return {
                    'error': 'GROQ_API_KEY not configured. Add it to .env file.'
                }, 500

            data = request.get_json() or {}
            messages = data.get('messages', [])

            if not messages:
                return {
                    'error': 'messages array is required'
                }, 400

            # Prepare messages with system prompt
            full_messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + messages

            response = requests.post(
                get_groq_server(),
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    "model": DEFAULT_MODEL,
                    "messages": full_messages,
                    "max_tokens": MAX_TOKENS
                },
                timeout=60
            )

            if response.status_code == 200:
                api_data = response.json()
                text = api_data['choices'][0]['message']['content']
                # Return in Anthropic-compatible format
                return {
                    'content': [{'text': text}]
                }, 200
            else:
                return {
                    'error': f'Groq API error: {response.status_code} - {response.text}'
                }, response.status_code

        except requests.Timeout:
            return {
                'error': 'Request timed out'
            }, 504
        except Exception as e:
            return {
                'error': str(e)
            }, 500


# Register endpoint
api.add_resource(_Chat, '/chat')