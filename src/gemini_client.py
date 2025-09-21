# src/gemini_client.py
import os
import requests
import json

API_KEY = os.getenv("GEMINI_API_KEY")  # must be set on the host (Render / Streamlit Cloud / etc.)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # default to flash; change via env if needed

if not API_KEY:
    # Raise on import so missing key fails quickly in deployment with a clear message.
    raise RuntimeError(
        "Missing GEMINI_API_KEY environment variable. "
        "Set GEMINI_API_KEY on the host (Render secrets / Streamlit secrets / .env locally)."
    )

BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

def ask_gemini(prompt: str, timeout: int = 30) -> str:
    """
    Send a prompt to Gemini (AI Studio API key). Returns the main text response.
    Raises RuntimeError on non-200 responses.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 200:
        j = resp.json()
        # safety checks
        try:
            return j["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected response format from Gemini: {e}\n{json.dumps(j)}")
    else:
        # include body in error for easier debugging on host
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
