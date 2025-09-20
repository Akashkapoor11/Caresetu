import os
import requests

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = os.getenv("GEMINI_API_URL", 
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
)

def gemini_generate(prompt: str) -> str:
    if not API_KEY:
        return "[Error] GEMINI_API_KEY not set"

    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(API_URL, headers=headers, params=params, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[REST API call failed] {e}"
