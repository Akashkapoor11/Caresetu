# test_gemini.py
import os
try:
    import google.generativeai as genai
except Exception as e:
    print("google.generativeai import failed:", e)
    raise SystemExit(1)

key = os.getenv("GEMINI_API_KEY")
if not key:
    print("GEMINI_API_KEY not set in environment.")
    raise SystemExit(1)

# configure SDK (newer versions)
try:
    genai.configure(api_key=key)
except Exception:
    try:
        genai.api_key = key
    except Exception as e:
        print("Could not configure genai:", e)
        raise SystemExit(1)

print("SDK configured, calling a small prompt...")

try:
    model = genai.GenerativeModel("gemini-1.0")  # or "gemini-2.0" if available
    resp = model.generate_content(
        "Say hello in one short sentence and mention 'CareSetu test'.",
        generation_config={"max_output_tokens": 50, "temperature": 0.0}
    )
    if hasattr(resp, "text") and resp.text:
        print("Response (text):", resp.text)
    else:
        print("Response object:", resp)
except Exception as e:
    print("Call failed:", e)
