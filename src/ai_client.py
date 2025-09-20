# ai_client.py
import os

# this module delegates to the top-level call_gemini function if available;
# otherwise it provides a simple fallback.

try:
    # sometimes app.py defines call_gemini in the same process; try import
    from app import call_gemini  # type: ignore
except Exception:
    call_gemini = None

class OpenAIClient:
    def __init__(self):
        self.available = call_gemini is not None

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2):
        """
        Returns the model text. If call_gemini isn't available, returns a fallback message.
        """
        if self.available:
            try:
                return call_gemini(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                return f"[LLM call failed: {e}]"
        # fallback:
        return ("[LLM not configured] Fallback answer. "
                "Install google-generativeai and set GEMINI_API_KEY to enable full LLM responses.")
