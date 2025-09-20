# app.py - CareSetu Streamlit app (updated OCR + VLM heuristics, minimal UI tips)
# Save and run: streamlit run app.py

import streamlit as st
import os
import json
import time
import hashlib
import hmac
import binascii
import base64
from datetime import datetime, date
from pathlib import Path
import tempfile

# Optional libs (defensive imports)
try:
    import google.generativeai as genai_legacy
except Exception:
    genai_legacy = None

try:
    import openai
except Exception:
    openai = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# PIL / numpy
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
    import numpy as np
except Exception:
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageOps = None
    ImageEnhance = None
    np = None

# OpenCV (optional, helps preprocessing)
try:
    import cv2
except Exception:
    cv2 = None

# pytesseract (preferred OCR)
try:
    import pytesseract
except Exception:
    pytesseract = None

# easyocr fallback
try:
    import easyocr
except Exception:
    easyocr = None

# Mongo (optional)
try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
except Exception:
    MongoClient = None
    ObjectId = None

import pandas as pd
import altair as alt

# dotenv loader (optional)
try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None
    find_dotenv = None

def safe_load_dotenv():
    if load_dotenv is None:
        return False
    try:
        load_dotenv()
        return True
    except Exception:
        try:
            env_path = find_dotenv() or ".env"
        except Exception:
            env_path = ".env"
        if not os.path.exists(env_path):
            return False
        try:
            with open(env_path, "rb") as f:
                raw = f.read()
            decoded = raw.decode("utf-8", errors="replace")
            tmp = tempfile.NamedTemporaryFile(delete=False, prefix="env_sanitized_", suffix=".env", mode="w", encoding="utf-8")
            tmp.write(decoded)
            tmp.close()
            load_dotenv(tmp.name, override=True)
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            return True
        except Exception:
            try:
                load_dotenv(encoding="latin-1")
                return True
            except Exception:
                return False

safe_load_dotenv()

# Env keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if GEMINI_API_KEY and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

if openai and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        try:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        except Exception:
            pass

# -----------------------
# LLM wrappers (unchanged logic, defensive)
# -----------------------
def call_gemini(prompt: str, max_tokens: int = 400, temperature: float = 0.2, model_hint: str = None) -> str:
    """
    Try modern google.genai then fallback to older google.generativeai.
    Returns text or raises RuntimeError with brief message.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set.")

    model_name = model_hint or "gemini-2.0-flash"

    # Try modern SDK
    try:
        try:
            from google import genai as genai_modern  # type: ignore
        except Exception:
            genai_modern = None

        if genai_modern is not None:
            client = genai_modern.Client(api_key=GEMINI_API_KEY)
            # Try a commonly used call; some SDK versions differ.
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    max_output_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
            except TypeError:
                # fallback without max_output_tokens param
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    temperature=float(temperature),
                )
            # Normalize response
            if hasattr(resp, "text") and resp.text:
                return resp.text
            if hasattr(resp, "outputs") and resp.outputs:
                out0 = resp.outputs[0]
                if hasattr(out0, "content"):
                    return out0.content
            if hasattr(resp, "candidates") and resp.candidates:
                first = resp.candidates[0]
                if hasattr(first, "content"):
                    return first.content
                if isinstance(first, dict):
                    return first.get("content") or first.get("text") or str(first)
            return str(resp)
    except Exception:
        pass

    # Fallback: older google.generativeai
    try:
        if genai_legacy is not None:
            try:
                try:
                    genai_legacy.configure(api_key=GEMINI_API_KEY)
                except Exception:
                    pass
                if hasattr(genai_legacy, "generate_text"):
                    resp = genai_legacy.generate_text(model=model_name, prompt=prompt, max_output_tokens=int(max_tokens))
                    if isinstance(resp, dict):
                        for k in ("candidates", "output", "text", "content"):
                            if k in resp:
                                v = resp[k]
                                if isinstance(v, list) and v:
                                    return v[0].get("content") or v[0].get("text") or str(v[0])
                                return str(v)
                    if hasattr(resp, "text") and resp.text:
                        return resp.text
                    return str(resp)
                elif hasattr(genai_legacy, "GenerativeModel"):
                    model_obj = genai_legacy.GenerativeModel(model_name)
                    resp = model_obj.generate_content(prompt, generation_config={"max_output_tokens": int(max_tokens), "temperature": float(temperature)})
                    if hasattr(resp, "text") and resp.text:
                        return resp.text
                    if isinstance(resp, dict):
                        for k in ("text", "output", "content", "candidates"):
                            if k in resp:
                                v = resp[k]
                                if isinstance(v, list) and v:
                                    c = v[0]
                                    if isinstance(c, dict):
                                        return c.get("content") or c.get("text") or str(c)
                                    return str(c)
                                return str(v)
                    return str(resp)
            except Exception:
                pass
    except Exception:
        pass

    # If both fail, raise friendly error (caller will show fallback UI)
    raise RuntimeError("Gemini SDK not available or call patterns failed. Ensure GEMINI_API_KEY and google-genai are installed/compatible.")

def call_openai_chat(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 400, temperature: float = 0.2) -> str:
    if openai is None:
        raise RuntimeError("OpenAI library not installed.")
    if not (OPENAI_API_KEY or getattr(openai, "api_key", None)):
        raise RuntimeError("OPENAI_API_KEY not set.")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful pediatric assistant. Be concise and cautious."},
            {"role": "user", "content": prompt}
        ]
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        choices = resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            return content
        return str(resp)
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

# ---------------------------
# Utility: robust OCR function (tries pytesseract -> easyocr)
# ---------------------------
def ocr_image_to_text(path: str) -> str:
    """
    Try to extract text from an image using:
      1. pytesseract (requires tesseract installed and pytesseract)
      2. easyocr (if installed)
    Returns a string with text OR a clear error message indicating what to install.
    """
    if Image is None:
        return "OCR unavailable: Pillow (PIL) not installed."

    # small preprocessing to improve OCR
    def preprocess_for_ocr_pil(im_pil):
        try:
            # convert to grayscale, enhance contrast and sharpen
            gray = im_pil.convert("L")
            enhancer = ImageEnhance.Contrast(gray)
            gray = enhancer.enhance(1.6)
            gray = gray.filter(ImageFilter.SHARPEN)
            # optional equalize / autocontrast
            gray = ImageOps.autocontrast(gray)
            return gray
        except Exception:
            return im_pil.convert("L")

    try:
        im = Image.open(path)
    except Exception as e:
        return f"OCR failed: cannot open image ({e})"

    # Try pytesseract
    if pytesseract is not None:
        # Check that tesseract engine exists: pytesseract.pytesseract.tesseract_cmd or default in PATH
        try:
            # Do a quick call; if tesseract not installed, this raises OSError
            pre = preprocess_for_ocr_pil(im)
            txt = pytesseract.image_to_string(pre, lang='eng')
            txt = (txt or "").strip()
            if txt:
                return txt
            # If empty, try color->thresholding with opencv (if available)
            if cv2 is not None:
                try:
                    arr = np.array(pre)
                    _, thr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thr_pil = Image.fromarray(thr)
                    txt2 = pytesseract.image_to_string(thr_pil, lang='eng')
                    return (txt2 or "").strip()
                except Exception:
                    pass
            # empty result but engine worked
            return ""  # no text found
        except Exception as e:
            # Could be tesseract missing or other error; fall through to easyocr if available
            pyt_err = str(e)
    else:
        pyt_err = "pytesseract not installed"

    # Try easyocr if available
    if easyocr is not None:
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            # easyocr wants an image path or array
            res = reader.readtext(path, detail=0)
            txt = "\n".join(res).strip()
            return txt
        except Exception as e:
            easy_err = str(e)
    else:
        easy_err = "easyocr not installed"

    # If we reached here, neither worked - return an instructive message
    install_msg_lines = [
        "OCR failed: no working OCR engine detected.",
        "Options to fix:",
        "1) Install Tesseract OCR engine and pytesseract:",
        "   - Ubuntu/Debian: sudo apt-get install tesseract-ocr",
        "   - macOS (homebrew): brew install tesseract",
        "   - Windows: install from Tesseract project releases and add to PATH",
        "   Then: pip install pytesseract",
        "OR",
        "2) Install easyocr (pip install easyocr) and its deps (may need torch).",
        "Detected errors:",
        f" - pytesseract: {pyt_err}",
        f" - easyocr: {easy_err if 'easy_err' in locals() else 'not attempted'}",
    ]
    return "\n".join(install_msg_lines)

# ---------------------------
# Enhanced VLM/local image analysis (returns explicit flags)
# ---------------------------
def analyze_image_symptom(image_path: str):
    """
    Heuristic analysis returning:
      - finding (text summary)
      - confidence (0..1)
      - details (diagnostic numbers)
      - annotated_path (image saved locally with label)
      - rash_on_face (True/False/None)
      - swelling_detected (True/False/None)
      - wound_detected (True/False/None)
    """
    result = {
        "finding": "Unclear",
        "confidence": 0.0,
        "details": "",
        "annotated_path": "",
        "rash_on_face": None,
        "swelling_detected": None,
        "wound_detected": None
    }

    if Image is None or np is None:
        result["finding"] = "Unavailable"
        result["details"] = "PIL or numpy not installed."
        return result

    try:
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        # scale down for analysis
        max_dim = 900
        if max(w, h) > max_dim:
            im_small = im.copy()
            im_small.thumbnail((max_dim, max_dim))
        else:
            im_small = im.copy()

        arr = np.array(im_small).astype(np.float32)
        brightness = arr.mean() / 255.0
        r_mean = arr[:, :, 0].mean()
        g_mean = arr[:, :, 1].mean()
        b_mean = arr[:, :, 2].mean()
        rgb_mean = (r_mean + g_mean + b_mean) / 3.0
        red_ratio = (r_mean - rgb_mean) / (rgb_mean + 1e-6)
        std_total = arr.std()
        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        thr = max(10, int(gray.mean() - 30))
        mask = (gray < thr).astype(np.uint8)
        spots = mask.sum() / (mask.shape[0] * mask.shape[1])

        findings = []
        confidence = 0.0
        if red_ratio > 0.18 and brightness > 0.2:
            findings.append("Redness / inflammation")
            confidence += min(0.55, red_ratio)
        if spots > 0.02 and std_total > 25:
            findings.append("Small dark spots / possible rash/scabbing")
            confidence += min(0.45, spots * 4.5)
        if std_total > 55 and red_ratio > 0.08:
            findings.append("High texture / possible swelling")
            confidence += 0.2
        if brightness < 0.12:
            findings.append("Low-light — limited reliability")
            confidence += 0.15
        if not findings:
            findings.append("No clear abnormal signs detected")
            confidence += 0.05

        # Heuristic face-region detection: top-center box (approximate)
        H, W = gray.shape
        top_h = max(1, int(H * 0.40))
        left = max(0, int(W * 0.20))
        right = min(W, int(W * 0.80))
        face_block = arr[0:top_h, left:right, :]
        face_gray = np.dot(face_block[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        face_thr = max(10, int(face_gray.mean() - 30))
        face_mask_dark = (face_gray < face_thr).astype(np.uint8)
        face_spots = face_mask_dark.sum() / (face_mask_dark.shape[0] * face_mask_dark.shape[1])
        face_r_mean = face_block[:, :, 0].mean()
        face_g_mean = face_block[:, :, 1].mean()
        face_b_mean = face_block[:, :, 2].mean()
        face_rgb_mean = (face_r_mean + face_g_mean + face_b_mean) / 3.0
        face_red_ratio = (face_r_mean - face_rgb_mean) / (face_rgb_mean + 1e-6)
        face_std = face_block.std()

        rash_on_face = False
        if (face_red_ratio > 0.15 and face_spots > 0.02) or (face_spots > 0.05):
            rash_on_face = True

        swelling_detected = False
        if face_std > 40 and face_red_ratio > 0.08:
            swelling_detected = True
        if not swelling_detected and std_total > 55 and red_ratio > 0.12:
            swelling_detected = True

        wound_detected = False
        if face_spots > 0.04 and face_std > 30:
            wound_detected = True
        if not wound_detected and spots > 0.05 and std_total > 30:
            wound_detected = True

        # annotated image (top overlay text)
        ann_path = ""
        try:
            ann = im_small.copy()
            draw = ImageDraw.Draw(ann)
            label = ", ".join(findings[:2])
            flags = [
                f"Rash_on_face={'YES' if rash_on_face else 'NO'}",
                f"Swelling={'YES' if swelling_detected else 'NO'}",
                f"Wound={'YES' if wound_detected else 'NO'}"
            ]
            txt = f"{label} ({int(min(1.0, confidence)*100)}% conf) — {'; '.join(flags)}"
            draw.rectangle([(0, 0), (ann.width, 36)], fill=(0, 0, 0, 160))
            try:
                draw.text((6, 6), txt, fill=(255, 255, 255))
            except Exception:
                pass
            ann = ann.filter(ImageFilter.SHARPEN)
            ann_fname = PHOTOS_DIR / f"annot_{int(time.time())}.jpg"
            ann.save(ann_fname, quality=80)
            ann_path = str(ann_fname)
        except Exception:
            ann_path = ""

        details_lines = [
            f"Brightness: {brightness:.2f}",
            f"Red prominence: {red_ratio:.2f}",
            f"Texture std dev: {std_total:.1f}",
            f"Dark-pixel fraction: {spots:.3f}"
        ]
        details = "; ".join(details_lines)
        result.update({
            "finding": "; ".join(findings),
            "confidence": float(min(1.0, confidence)),
            "details": details,
            "annotated_path": ann_path,
            "rash_on_face": rash_on_face,
            "swelling_detected": swelling_detected,
            "wound_detected": wound_detected
        })
        return result
    except Exception as e:
        result["finding"] = "Error"
        result["details"] = f"Processing failed: {e}"
        return result

def vlm_symptom_flow(image_tmp_path: str, ocr_text: str = ""):
    out = {"local": None, "llm": None}
    local = analyze_image_symptom(image_tmp_path)
    out["local"] = local
    # Optionally call Gemini/OpenAI to enrich (if available)
    if GEMINI_API_KEY:
        try:
            prompt_parts = [
                "You are a cautious pediatric assistant. Parent uploaded a photo of a child's skin/area.",
                "Local heuristics summary:",
                f"Finding: {local['finding']}. Confidence: {local['confidence']:.2f}. Details: {local['details']}.",
                f"Flags: rash_on_face={local.get('rash_on_face')}, swelling={local.get('swelling_detected')}, wound={local.get('wound_detected')}.",
            ]
            if ocr_text:
                prompt_parts.append("OCR text (short):")
                prompt_parts.append(ocr_text[:1200])
            prompt_parts.append("Based on this, list likely benign causes, possible concerning causes, and a short action plan with red flags. Be concise and cautious.")
            prompt = "\n\n".join(prompt_parts)
            try:
                llm_resp = call_gemini(prompt, max_tokens=300, temperature=0.0, model_hint="gemini-2.0")
                out["llm"] = llm_resp
            except Exception:
                out["llm"] = None
        except Exception:
            out["llm"] = None
    return out

# ---------------------------
# App config / storage
# ---------------------------
st.set_page_config(page_title="CareSetu – AI Baby Care Companion", page_icon="🤱", layout="wide")
ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / ".data"
DATA_DIR.mkdir(exist_ok=True)
PHOTOS_DIR = DATA_DIR / "photos"
PHOTOS_DIR.mkdir(exist_ok=True)

HEALTH_LOG_FILE = DATA_DIR / "health_log.json"
GROWTH_FILE = DATA_DIR / "growth.json"
VACCINE_FILE = DATA_DIR / "vaccines.json"
HOSPITALS_FILE = DATA_DIR / "hospitals.json"
CHILD_FILE = DATA_DIR / "child.json"
EXTRACTED_TEXT_FILE = DATA_DIR / "1000_days_extracted.txt"

_candidate_bg = Path("/mnt/data/bg1.webp")
if _candidate_bg.exists():
    BG_IMAGE_PATH = _candidate_bg
else:
    BG_IMAGE_PATH = ROOT / "bg1.webp"

def build_data_uri(path: Path) -> str:
    try:
        if not path or not path.exists():
            return ""
        b = path.read_bytes()
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "webp":
            mime = "image/webp"
        elif suffix in ("jpg", "jpeg"):
            mime = "image/jpeg"
        elif suffix == "png":
            mime = "image/png"
        elif suffix == "gif":
            mime = "image/gif"
        else:
            mime = "application/octet-stream"
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def months_between(dob: date, now: date):
    return (now.year - dob.year) * 12 + (now.month - dob.month) + (now.day - dob.day)/30.0

def safe_rerun():
    try:
        st.rerun()
        return
    except Exception:
        pass
    st.session_state["_force_refresh_ts"] = str(time.time())
    st.stop()

# MongoDB defensive connect
def get_mongo_client():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    if MongoClient is None:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client
    except Exception:
        return None

mongo_client = get_mongo_client()
mongo_available = False
users_col = child_col = growth_col = health_col = vaccines_col = None
if mongo_client:
    try:
        db = mongo_client.get_database("caresetu")
        users_col = db.get_collection("users")
        child_col = db.get_collection("child")
        growth_col = db.get_collection("growth")
        health_col = db.get_collection("health_log")
        vaccines_col = db.get_collection("vaccines")
        mongo_available = True
    except Exception:
        mongo_available = False
        users_col = child_col = growth_col = health_col = vaccines_col = None

# password hashing helpers (unchanged)
def hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return binascii.hexlify(salt).decode() + "$" + binascii.hexlify(dk).decode()

def verify_password(stored_hash: str, password_attempt: str) -> bool:
    try:
        salt_hex, dk_hex = stored_hash.split("$")
        salt = binascii.unhexlify(salt_hex)
        expected = binascii.unhexlify(dk_hex)
        attempt = hashlib.pbkdf2_hmac("sha256", password_attempt.encode("utf-8"), salt, 200_000)
        return hmac.compare_digest(attempt, expected)
    except Exception:
        return False

# Minimal user API (unchanged)
def create_user(username: str, password: str):
    user_doc = {
        "username": username,
        "password_hash": hash_password(password),
        "created_at": datetime.utcnow().isoformat(),
        "email": "",
        "phone": "",
        "photo_path": ""
    }
    if mongo_available:
        if users_col.find_one({"username": username}):
            return False, "Username already exists."
        res = users_col.insert_one(user_doc)
        return True, str(res.inserted_id)
    else:
        users_file = DATA_DIR / "users.json"
        users = load_json(users_file, [])
        if any(u.get("username") == username for u in users):
            return False, "Username already exists."
        user_doc["id"] = len(users) + 1
        users.append(user_doc)
        save_json(users_file, users)
        return True, str(user_doc["id"])

def authenticate_user(username: str, password: str):
    if mongo_available:
        u = users_col.find_one({"username": username})
        if not u:
            return False, "No such user"
        if verify_password(u.get("password_hash",""), password):
            return True, str(u.get("_id"))
        return False, "Invalid password"
    else:
        users_file = DATA_DIR / "users.json"
        users = load_json(users_file, [])
        for u in users:
            if u.get("username") == username:
                if verify_password(u.get("password_hash",""), password):
                    return True, str(u.get("id"))
                else:
                    return False, "Invalid password"
        return False, "No such user"

# sample hospitals (unchanged)
if not HOSPITALS_FILE.exists():
    sample_hospitals = {
        "kanpur": [
            {"name": "Government Hospital A - Kanpur", "address": "Civil Lines, Kanpur", "phone": "0512-XXXXXXX"},
            {"name": "Private Hospital B - Kanpur", "address": "G.T. Road, Kanpur", "phone": "0512-YYYYYYY"},
        ],
        "delhi": [
            {"name": "All India Institute of Medical Sciences (AIIMS)", "address": "Ansari Nagar, New Delhi", "phone": "011-2658XXXX"},
            {"name": "Safdarjung Hospital", "address": "Ring Road, New Delhi", "phone": "011-2673XXXX"},
        ],
        "default": [
            {"name": "District Hospital", "address": "Nearest Town", "phone": "102"},
            {"name": "Primary Health Center (PHC)", "address": "Nearest Village", "phone": "102"},
        ],
    }
    save_json(HOSPITALS_FILE, sample_hospitals)

STANDARD_VACCINES = [
    {"name": "BCG", "due_month": 0},
    {"name": "OPV (birth)", "due_month": 0},
    {"name": "Hepatitis B (birth)", "due_month": 0},
    {"name": "OPV (6 weeks)", "due_month": 1.5},
    {"name": "DPT (6 weeks)", "due_month": 1.5},
    {"name": "Hep B (6 weeks)", "due_month": 1.5},
    {"name": "OPV (10 weeks)", "due_month": 2.5},
    {"name": "DPT (10 weeks)", "due_month": 2.5},
    {"name": "OPV (14 weeks)", "due_month": 3.5},
    {"name": "DPT (14 weeks)", "due_month": 3.5},
    {"name": "Measles (9 months)", "due_month": 9},
    {"name": "Vitamin A (9 months)", "due_month": 9},
    {"name": "MMR (15-18 months)", "due_month": 15},
]

hospitals_data = load_json(HOSPITALS_FILE, {})
vaccine_state_local = load_json(VACCINE_FILE, {})
vaccine_state_local.setdefault("done", {})
vaccine_state_local.setdefault("notes", "")

# styling (kept)
_bg_data_uri = build_data_uri(BG_IMAGE_PATH)
hero_bg_css = f'url("{_bg_data_uri}")' if _bg_data_uri else "linear-gradient(135deg,#84fab0,#8fd3f4)"

st.markdown(
    f"""
    <style>
    :root {{
      --primary-grad: linear-gradient(90deg,#2b2140 0%, #2f2a55 100%);
      --card-bg: linear-gradient(180deg,#ffffff,#fbfeff);
      --muted: #667;
    }}
    html, body, [class*="css"] {{
      font-family: 'Segoe UI', Roboto, Arial, sans-serif;
      color: var(--muted);
    }}
    .top-header {{
      position: sticky;
      top: 10px;
      z-index: 9999;
      display:flex; align-items:center; justify-content:space-between;
      background: var(--primary-grad);
      color: #fff;
      padding: 16px 20px; border-radius: 10px; margin-bottom: 18px;
      box-shadow: 0 10px 30px rgba(16,24,40,0.25);
      border: 1px solid rgba(255,255,255,0.04);
    }}
    .branding {{ display:flex; align-items:center; gap:12px; }}
    .app-icon {{
      width:48px; height:48px; border-radius:10px; background: linear-gradient(180deg,#ffb3c7,#ff8fb2);
      display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    }}
    .hero {{
      background-image: {hero_bg_css};
      background-size: cover;
      background-position: center;
      border-radius: 12px;
      padding: 52px 32px;
      margin-bottom: 18px;
      color: #fff;
      box-shadow: inset 0 0 30px rgba(0,0,0,0.25);
    }}
    .card {{
      background: var(--card-bg);
      border-radius:12px; padding:14px; margin:12px 0; box-shadow: 0 10px 30px rgba(44,37,77,0.04);
      border: 1px solid rgba(45,58,80,0.02);
    }}
    .muted {{ color: var(--muted); font-size:13px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# User/profile helpers and UI pages: keep logic but use our OCR and VLM functions in the Verified Data Assistant page.
def get_user_profile_record():
    uid = st.session_state.get("user_id")
    if not uid:
        return {}
    if mongo_available:
        try:
            try:
                objid = ObjectId(uid)
            except Exception:
                objid = uid
            u = users_col.find_one({"_id": objid})
            if u:
                u["id"] = str(u.get("_id"))
                return u
        except Exception:
            return {}
    else:
        users_file = DATA_DIR / "users.json"
        users = load_json(users_file, [])
        for u in users:
            if str(u.get("id")) == str(uid) or u.get("username") == st.session_state.get("user"):
                return u
    return {}

def save_user_profile_updates(updates: dict):
    uid = st.session_state.get("user_id")
    if not uid:
        return False
    if mongo_available:
        try:
            try:
                objid = ObjectId(uid)
            except Exception:
                objid = uid
            users_col.update_one({"_id": objid}, {"$set": updates})
            return True
        except Exception:
            return False
    else:
        users_file = DATA_DIR / "users.json"
        users = load_json(users_file, [])
        changed = False
        for i, u in enumerate(users):
            if str(u.get("id")) == str(uid) or u.get("username") == st.session_state.get("user"):
                users[i].update(updates)
                changed = True
                break
        if changed:
            save_json(users_file, users)
            return True
        return False

def save_uploaded_user_photo(uploaded_file, uid):
    if not uploaded_file:
        return ""
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        ext = ".png"
    fname = f"user_photo_{uid}{ext}"
    dest = PHOTOS_DIR / fname
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)

def render_header():
    profile = get_user_profile_record()
    avatar_html = ""
    if profile.get("photo_path") and Path(profile.get("photo_path")).exists():
        avatar_uri = build_data_uri(Path(profile.get("photo_path")))
        if avatar_uri:
            avatar_html = f'<div style="width:46px;height:46px;border-radius:10px;overflow:hidden"><img src="{avatar_uri}" style="width:100%;height:100%;object-fit:cover"/></div>'
    st.markdown(f"""
      <div class="top-header">
        <div style="display:flex;align-items:center;gap:12px;">
          <div class="app-icon">🤱</div>
          <div>
            <div style="font-weight:800;font-size:18px">CareSetu – Baby Care Companion</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.85)">Personalized tips, trackers & records</div>
          </div>
        </div>
        <div>{avatar_html}</div>
      </div>
    """, unsafe_allow_html=True)

# session init
if "user" not in st.session_state:
    st.session_state.user = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = None
if "gemini_cache" not in st.session_state:
    st.session_state["gemini_cache"] = {}
if "last_gemini_call_ts" not in st.session_state:
    st.session_state["last_gemini_call_ts"] = 0.0

render_header()

# Authentication UI (unchanged)
if not st.session_state.user:
    st.sidebar.title("Account")
    auth_page = st.sidebar.radio("Sign Up / Login", ["Login", "Sign up", "Continue as guest", "Profile (view only)"], key="auth_radio")

    if auth_page in ("Login", "Sign up"):
        st.markdown('<div class="hero"><h1 style="margin:0">Care for your baby — simple, reliable, personalised</h1></div>', unsafe_allow_html=True)

    if auth_page == "Sign up":
        st.markdown('<div class="card"><h3>Create an account</h3>', unsafe_allow_html=True)
        with st.form("signup", clear_on_submit=False):
            su_name = st.text_input("Username", key="su_name")
            su_pass = st.text_input("Password", type="password", key="su_pass")
            su_confirm = st.text_input("Confirm Password", type="password", key="su_confirm")
            s_sub = st.form_submit_button("Sign up", key="su_submit")
            if s_sub:
                if not su_name or not su_pass:
                    st.error("Enter username & password")
                elif su_pass != su_confirm:
                    st.error("Passwords do not match")
                else:
                    ok, res = create_user(su_name, su_pass)
                    if ok:
                        st.success("Account created — please log in.")
                    else:
                        st.error(res)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    elif auth_page == "Login":
        st.markdown('<div class="card"><h3>Login</h3>', unsafe_allow_html=True)
        with st.form("login", clear_on_submit=False):
            li_name = st.text_input("Username", key="li_name")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            l_sub = st.form_submit_button("Log in", key="li_submit")
            if l_sub:
                ok, res = authenticate_user(li_name, li_pass)
                if ok:
                    st.success("Login successful")
                    st.session_state.user = li_name
                    st.session_state.user_id = res
                    safe_rerun()
                else:
                    st.error(res)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    elif auth_page == "Profile (view only)":
        st.markdown('<div class="card"><h3>Profile (no account)</h3>', unsafe_allow_html=True)
        st.info("Sign up or login to edit a persistent profile (phone, email, photo).")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    else:
        st.session_state.user = None
        st.session_state.user_id = "guest"

# user-scoped helpers (unchanged)
def user_insert_child(profile: dict):
    profile_rec = profile.copy()
    profile_rec["user_id"] = st.session_state.user_id or "guest"
    profile_rec["created_at"] = datetime.utcnow().isoformat()
    if mongo_available:
        res = child_col.insert_one(profile_rec)
        return str(res.inserted_id)
    else:
        child_file = DATA_DIR / f"child_{st.session_state.user_id or 'guest'}.json"
        save_json(child_file, profile_rec)
        return "local:" + str(child_file)

def user_get_child():
    if mongo_available:
        rec = child_col.find_one({"user_id": st.session_state.user_id})
        if rec and "_id" in rec:
            rec["id"] = str(rec["_id"])
        return rec or {}
    else:
        child_file = DATA_DIR / f"child_{st.session_state.user_id or 'guest'}.json"
        return load_json(child_file, {})

def user_insert_growth(rec):
    record = rec.copy()
    record["user_id"] = st.session_state.user_id or "guest"
    record["created_at"] = datetime.utcnow().isoformat()
    if mongo_available:
        res = growth_col.insert_one(record)
        return str(res.inserted_id)
    else:
        gf = DATA_DIR / f"growth_{st.session_state.user_id or 'guest'}.json"
        arr = load_json(gf, [])
        arr.append(record)
        save_json(gf, arr)
        return "local"

def user_get_growth():
    if mongo_available:
        return list(growth_col.find({"user_id": st.session_state.user_id}))
    else:
        gf = DATA_DIR / f"growth_{st.session_state.user_id or 'guest'}.json"
        return load_json(gf, [])

def user_insert_health(rec):
    record = rec.copy()
    record["user_id"] = st.session_state.user_id or "guest"
    record["created_at"] = datetime.utcnow().isoformat()
    if mongo_available:
        res = health_col.insert_one(record)
        return str(res.inserted_id)
    else:
        hf = DATA_DIR / f"health_{st.session_state.user_id or 'guest'}.json"
        arr = load_json(hf, [])
        arr.insert(0, record)
        save_json(hf, arr)
        return "local"

def user_get_health():
    if mongo_available:
        return list(health_col.find({"user_id": st.session_state.user_id}))
    else:
        hf = DATA_DIR / f"health_{st.session_state.user_id or 'guest'}.json"
        return load_json(hf, [])

def user_get_vaccines():
    if mongo_available:
        rec = vaccines_col.find_one({"user_id": st.session_state.user_id})
        return rec or {"done": {}, "notes": ""}
    else:
        vf = DATA_DIR / f"vaccines_{st.session_state.user_id or 'guest'}.json"
        return load_json(vf, {"done": {}, "notes": ""})

def user_save_vaccines(obj):
    obj_c = obj.copy()
    obj_c["user_id"] = st.session_state.user_id or "guest"
    if mongo_available:
        vaccines_col.update_one({"user_id": obj_c["user_id"]}, {"$set": obj_c}, upsert=True)
    else:
        vf = DATA_DIR / f"vaccines_{st.session_state.user_id or 'guest'}.json"
        save_json(vf, obj_c)

# UI helpers
def card_start(title: str = None, subtitle: str = None):
    if title:
        st.markdown(f"<div class='card'><h3 style='margin-top:0'>{title}</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='muted'>{subtitle}</div><br/>", unsafe_allow_html=True)

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

# BMI classifier (unchanged)
def classify_bmi(weight_kg: float, height_cm: float):
    if not weight_kg or not height_cm:
        return "Unknown", "Invalid weight or height provided.", []
    try:
        height_m = height_cm / 100.0
        if height_m <= 0:
            return "Unknown", "Invalid height.", []
        bmi = weight_kg / (height_m ** 2)
    except Exception:
        return "Unknown", "Could not compute BMI from inputs.", []
    if bmi < 18.5:
        status = "Underweight"
        explanation = f"BMI = {bmi:.1f}. This is below general healthy BMI thresholds."
        suggestions = [
            "Ensure adequate, frequent feedings and nutrient-rich foods.",
            "Monitor weight regularly and track feeding patterns.",
            "Seek pediatric advice if weight faltering continues."
        ]
    elif bmi > 24.9:
        status = "Overweight"
        explanation = f"BMI = {bmi:.1f}. This is above general healthy BMI thresholds."
        suggestions = [
            "Encourage age-appropriate active play and limit sugary drinks/snacks.",
            "Review portion sizes and feeding routines.",
            "Discuss growth and diet with a pediatrician if concerned."
        ]
    else:
        status = "Normal"
        explanation = f"BMI = {bmi:.1f}. This is within a general healthy range."
        suggestions = [
            "Continue balanced feeding and routine growth monitoring.",
            "Encourage active play and healthy eating habits."
        ]
    return status, explanation, suggestions

# RAG init (defensive - leave original logic)
try:
    from ai_client import OpenAIClient
    from db_helpers import VectorDB
    from vaccine_data import prepare_and_index
    from model import RAGPipeline
    from utils import transcribe_audio_local_placeholder  # we use our own OCR above
    AI_HELPERS_AVAILABLE = True
except Exception:
    AI_HELPERS_AVAILABLE = False
    def transcribe_audio_local_placeholder(_path):
        return "Transcript not available (utils missing)."
    try:
        # keep utils.ocr_image_to_text from earlier replaced by our function above
        pass
    except Exception:
        pass

try:
    if AI_HELPERS_AVAILABLE:
        vector_db = VectorDB()
        try:
            prepare_and_index(vector_db)
        except Exception:
            pass
        ai_client = OpenAIClient()
        rag_pipeline = RAGPipeline(vector_db=vector_db, ai_client=ai_client)
    else:
        rag_pipeline = None
except Exception:
    rag_pipeline = None

# ---------------------------
# Pages (assistant page keeps no-tip text)
# ---------------------------
def page_assistant():
    card_start("🤖 AI Parenting Assistant")
    child = user_get_child()
    if child:
        try:
            dob_default = date.fromisoformat(child.get("dob"))
        except Exception:
            dob_default = date(2023,9,16)
        city_default = child.get("city", "")
    else:
        dob_default = date(2023,9,16)
        city_default = "Kanpur"

    COOLDOWN_SECONDS = 2.0

    col_dbg1, col_dbg2 = st.columns([1,3])
    with col_dbg1:
        use_ai_toggle = st.checkbox(
            "Enable LLM",
            value=bool(GEMINI_API_KEY or OPENAI_API_KEY),
            key="use_ai_toggle"
        )
    with col_dbg2:
        st.write("")  # blank per request

    with st.form("assistant_form", clear_on_submit=True):
        user_q = st.text_input("Ask a parenting question", placeholder="E.g., 'My 6-month-old has fever. What should I do?'", key="assistant_q")
        col_a, col_b = st.columns(2)
        with col_a:
            dob_input = st.date_input("Baby's date of birth", value=dob_default, key="assistant_dob")
        with col_b:
            city_input = st.text_input("Your City / District", value=city_default, key="assistant_city")
        submit = st.form_submit_button("Get Personalized Advice", key="assistant_submit")

        if submit:
            if not user_q or not user_q.strip():
                st.error("Type a question first.")
            else:
                prompt = (
                    "You are a friendly pediatric assistant. Provide safe, simple, India-context advice.\n\n"
                    f"Child DOB: {dob_input.isoformat()}\n"
                    f"City: {city_input}\n"
                    f"Question: {user_q}\n\n"
                    "Provide concise, actionable suggestions and mention red flags that need immediate medical attention."
                )

                key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                cache = st.session_state.get("gemini_cache", {})
                now_ts = time.time()
                last_ts = st.session_state.get("last_gemini_call_ts", 0.0)
                if now_ts - last_ts < COOLDOWN_SECONDS:
                    st.warning("Please wait a moment before asking again (rate-limiting to avoid hitting API quotas).")
                    if key in cache:
                        st.write(cache[key])
                    else:
                        st.info("Showing fallback guidance until cooldown expires.")
                        st.write("- Check temperature; keep baby hydrated; seek urgent care for red-flag symptoms.")
                    card_end()
                    return

                if key in cache:
                    st.write(cache[key])
                    card_end()
                    return

                if not use_ai_toggle:
                    st.info("AI usage disabled (checkbox). Showing a safe fallback answer.")
                    st.write("- Measure temperature. If >38.5°C, consult pediatrician.")
                    st.write("- Keep baby hydrated and comfortable.")
                    st.write("- If severe symptoms, seek immediate care.")
                    card_end()
                    return

                llm_answer = None
                # Spinner "Thinking..."
                if GEMINI_API_KEY:
                    try:
                        with st.spinner("Thinking..."):
                            llm_answer = call_gemini(prompt, max_tokens=400, temperature=0.2, model_hint="gemini-2.0-flash")
                    except Exception:
                        llm_answer = None

                if llm_answer is None and openai and (OPENAI_API_KEY or getattr(openai, "api_key", None)):
                    try:
                        with st.spinner("Thinking..."):
                            llm_answer = call_openai_chat(prompt, model="gpt-3.5-turbo", max_tokens=400, temperature=0.2)
                    except Exception:
                        llm_answer = None

                if llm_answer:
                    cache[key] = llm_answer
                    if len(cache) > 200:
                        try:
                            first_key = next(iter(cache))
                            cache.pop(first_key, None)
                        except Exception:
                            cache.clear()
                    st.session_state["gemini_cache"] = cache
                    st.session_state["last_gemini_call_ts"] = time.time()
                    st.write(llm_answer)
                else:
                    st.info("AI not configured or LLM call failed — showing a safe fallback answer.")
                    st.write("- Measure temperature. If >38.5°C, consult pediatrician.")
                    st.write("- Keep baby hydrated and comfortable.")
                    st.write("- If severe symptoms, seek immediate care.")
    card_end()

# ... (other pages unchanged except Verified Data Assistant uses our OCR and VLM)
def page_vaccine():
    card_start("💉 Vaccine Tracker", "Recommended vaccines & status for your child.")
    child = user_get_child()
    if child:
        try:
            dob_for_vacc = date.fromisoformat(child.get("dob"))
        except Exception:
            dob_for_vacc = date.today()
    else:
        dob_for_vacc = date.today()
    age_months = months_between(dob_for_vacc, date.today())
    st.markdown(f"**Child DOB:** {dob_for_vacc.isoformat()}  •  **Age ≈ {age_months:.1f} months**")
    st.markdown("---")
    vacc = user_get_vaccines()
    vacc.setdefault("done", {})
    vacc.setdefault("notes", "")
    for v in STANDARD_VACCINES:
        due = v["due_month"]
        due_flag = age_months + 0.01 >= due - 0.5
        name = v["name"]
        done = name in vacc["done"]
        col1, col2 = st.columns([3,1])
        with col1:
            status = "✅ Done" if done else ("⏳ Due" if due_flag else "🕒 Upcoming")
            st.markdown(f"**{name}** — <span style='color:gray'>{status}</span>", unsafe_allow_html=True)
        with col2:
            if not done:
                if st.button(f"Mark {name} done", key=f"v_mark_{name}"):
                    vacc["done"][name] = datetime.now().isoformat()
                    user_save_vaccines(vacc)
                    safe_rerun()
            else:
                if st.button(f"Undo {name}", key=f"v_undo_{name}"):
                    vacc["done"].pop(name, None)
                    user_save_vaccines(vacc)
                    safe_rerun()
    st.markdown("---")
    notes = st.text_area("Vaccination notes", value=vacc.get("notes",""), key="vacc_notes")
    if st.button("Save vaccine notes", key="save_vacc_notes"):
        vacc["notes"] = notes
        user_save_vaccines(vacc)
        st.success("Saved vaccine notes.")
    card_end()

def page_health():
    card_start("🩺 Baby Health Log", "Log symptoms, treatments and visits.")
    health_log = user_get_health()
    with st.form("health_form", clear_on_submit=True):
        d = st.date_input("Date", value=date.today(), key="health_date")
        symptom = st.text_input("Symptoms / Reason (e.g., fever, cough)", key="health_symptom")
        notes = st.text_area("Notes / Treatment", key="health_notes")
        submitted = st.form_submit_button("Add Health Entry", key="health_submit")
        if submitted:
            entry = {"date": d.isoformat(), "symptom": symptom, "notes": notes}
            user_insert_health(entry)
            st.success("Entry added.")
            safe_rerun()
    if health_log:
        st.subheader("Recent entries")
        entries = health_log if isinstance(health_log, list) else sorted(health_log, key=lambda x: x.get("created_at",""), reverse=True)
        for e in entries:
            st.markdown(f"- **{e.get('date','')}** — {e.get('symptom','')} — {e.get('notes','')}")
    else:
        st.info("No health log entries yet.")
    card_end()

def page_nearby():
    card_start("🏥 Nearby Hospitals", "Quick lookup of hospitals in your area.")
    search_city = st.text_input("Enter your city/district to find nearby hospitals", value=(user_get_child() or {}).get("city", "Kanpur"), key="nearby_city")
    if st.button("Search hospitals", key="search_hospitals") or search_city:
        key = search_city.strip().lower()
        list_h = hospitals_data.get(key, hospitals_data.get("default", []))
        st.markdown(f"Showing results for **{search_city.title()}**")
        for h in list_h:
            name = h.get("name")
            addr = h.get("address")
            phone = h.get("phone", "N/A")
            q = f"{name} {addr} {search_city}"
            maps = f"https://www.google.com/maps/search/{q.replace(' ', '+')}"
            st.markdown(f"**{name}**  •  {addr}  •  Tel: {phone}  —  [Open in Maps]({maps})")
    card_end()

def page_growth():
    card_start("📈 Growth Tracker", "Weight and height tracking with BMI-based guidance.")
    st.caption(f"User: {st.session_state.user or 'Guest'}")
    growth = user_get_growth()
    with st.form("growth_form_page", clear_on_submit=True):
        gdate = st.date_input("Date", value=date.today(), key="growth_date")
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.01, format="%.2f", key="growth_weight")
        height_cm = st.number_input("Height / Length (cm)", min_value=0.0, step=0.1, format="%.1f", key="growth_height")
        submit_g = st.form_submit_button("Add growth record", key="growth_submit")
        if submit_g:
            rec = {"date": gdate.isoformat(), "weight": weight, "height_cm": height_cm}
            user_insert_growth(rec)
            st.success("Saved growth record.")
            safe_rerun()
    if growth:
        df = pd.DataFrame(growth)
        if df.empty:
            st.info("No valid growth records yet.")
            card_end()
            return
        df['weight'] = pd.to_numeric(df.get('weight', 0), errors='coerce').fillna(0.0)
        df['height_cm'] = pd.to_numeric(df.get('height_cm', 0), errors='coerce').fillna(0.0)
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df.sort_values("date")
        if df.empty or df['date'].isnull().all():
            st.info("No valid growth records yet.")
            card_end()
            return
        st.subheader("Weight & Height trend")
        chart_df = df.set_index("date")[["weight", "height_cm"]].rename(columns={"height_cm":"height_cm (cm)"})
        st.line_chart(chart_df)
        latest = df.loc[df["date"].idxmax()]
        latest_weight = float(latest['weight'])
        latest_height = float(latest['height_cm'])
        st.markdown("---")
        st.subheader("Growth Guidance (latest record)")
        status, explanation, suggestions = classify_bmi(latest_weight, latest_height)
        colors = {"Underweight": "#ffb3b3", "Overweight": "#bcdffb", "Normal": "#d1f7de", "Unknown": "#eee"}
        icons = {"Underweight": "⚠️", "Overweight": "ℹ️", "Normal": "✅", "Unknown": "❓"}
        bg = colors.get(status, "#eee")
        ic = icons.get(status, "ℹ️")
        st.markdown(
            f"""<div style="border-radius:8px;padding:12px 18px;background:{bg};color:#073642;">
                <strong style="font-size:16px">{ic} {status}</strong>
                <div style="margin-top:8px;color:#02111a">{explanation}</div>
               </div>""",
            unsafe_allow_html=True
        )
        if suggestions:
            st.markdown("**Suggestions**")
            for s in suggestions:
                st.markdown(f"- {s}")
        st.markdown(
            "<small style='color:#777'>⚠️ This classification is general and informational only — not medical advice. "
            "For personalized concerns, consult a certified pediatrician.</small>",
            unsafe_allow_html=True
        )
    else:
        st.info("No growth records yet.")
    card_end()

def page_voice():
    card_start("🎙️ Voice Ask (mic + upload fallback)", "Speak or upload audio to transcribe and (optionally) get an AI answer.")
    st.write("Option A: Use microphone (requires PyAudio). Option B: Upload audio file (works without PyAudio).")
    if sr is None:
        st.info("SpeechRecognition not installed. Use text inputs or upload audio.")
    else:
        mic_available = False
        try:
            import pyaudio  # noqa: F401
            mic_available = True
        except Exception:
            mic_available = False
        if mic_available and st.button("🎤 Start voice capture (mic)", key="start_voice_capture"):
            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("Listening for 6 seconds...")
                    audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)
                text = recognizer.recognize_google(audio, language="en-IN")
                st.success("You said: " + text)
                if GEMINI_API_KEY:
                    try:
                        prompt = f"You are a helpful pediatric assistant. User said: {text}"
                        with st.spinner("Thinking..."):
                            answer = call_gemini(prompt, max_tokens=300, temperature=0.2)
                        st.write(answer)
                        if gTTS:
                            tts = gTTS(answer, lang="en")
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                            tts.save(tmp.name)
                            st.audio(tmp.name)
                    except Exception:
                        st.error("Gemini call failed. Showing transcription only.")
                else:
                    st.info("AI not configured. Showing transcription only.")
            except Exception as e:
                st.error(f"Voice capture failed: {e}")
        uploaded_audio = st.file_uploader("📂 Or upload a recorded audio file", type=["wav","mp3","m4a","ogg"], key="voice_upload")
        if uploaded_audio is not None:
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(uploaded_audio) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="en-IN")
                st.success("Transcription: " + text)
                if GEMINI_API_KEY:
                    try:
                        prompt = f"You are a helpful pediatric assistant. User said: {text}"
                        with st.spinner("Thinking..."):
                            answer = call_gemini(prompt, max_tokens=300, temperature=0.2)
                        st.write(answer)
                        if gTTS:
                            tts = gTTS(answer, lang="en")
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                            tts.save(tmp.name)
                            st.audio(tmp.name)
                    except Exception:
                        st.error("Gemini call failed. Showing transcription only.")
                else:
                    st.info("AI not configured. Showing transcription only.")
            except Exception as e:
                st.error(f"Failed to process uploaded audio: {e}")
    card_end()

def page_guide():
    card_start("Quick actionable guidance — First 1000 days (conception → 2 years)")
    st.markdown(
        """
        **Why it matters**
        - The first 1000 days (from conception through a child's second birthday) are critical for brain growth, immune development and lifelong health.
        - Timely interventions in nutrition, health and stimulation produce lasting gains in learning, productivity and well-being.
        """
    )
    st.subheader("📅 Month-wise Guidance for First 24 Months")
    monthwise = {
        "0-1 month": [
            "Exclusive breastfeeding (8–12 times/day).",
            "Skin-to-skin contact, bonding and warmth.",
            "First checkup and vaccines (BCG, OPV, Hepatitis B)."
        ],
        "2 months": [
            "Continue exclusive breastfeeding.",
            "Vaccines: DPT, OPV, Hep B (as per schedule).",
            "Tummy time for motor development."
        ],
        "3 months": [
            "Baby starts smiling & cooing — encourage interaction.",
            "Vaccines booster as per schedule.",
            "Monitor weight gain."
        ],
        "4 months": [
            "Exclusive breastfeeding, no solids yet.",
            "Introduce rattles, colorful toys for stimulation.",
            "Look for head control milestones."
        ],
        "6 months": [
            "Introduce complementary feeding (semi-solid foods).",
            "Continue breastfeeding on demand.",
            "Ensure safe water and hygiene during feeding."
        ],
        "9 months": [
            "Baby crawls, sits with support.",
            "Introduce finger foods & variety (cereals, pulses, fruits).",
            "Vaccines: Measles, Vitamin A dose."
        ],
        "12 months (1 year)": [
            "Encourage walking with support.",
            "Family foods mashed/soft given 3–4 times a day.",
            "First birthday vaccines (MMR, boosters)."
        ],
        "15–18 months": [
            "Active playtime, speech development.",
            "Vaccines: MMR booster, DPT booster.",
            "Promote independence with self-feeding."
        ],
        "24 months (2 years)": [
            "Child runs, climbs stairs, speaks 2-word phrases.",
            "Balanced family diet with milk, fruits, vegetables.",
            "Regular health checkups and growth monitoring."
        ]
    }
    for age, tips in monthwise.items():
        st.markdown(f"### {age}")
        for t in tips:
            st.markdown(f"- {t}")
    st.info("⚠️ This month-wise plan is a **general guideline**. Always follow your pediatrician’s advice.")
    card_end()

def page_edit_profile():
    card_start("Edit child profile / Logout")
    profile = user_get_child() or {}
    st.write("Current profile:", profile if profile else "No child profile yet")
    with st.form("edit_profile"):
        n = st.text_input("Name", value=profile.get("name",""), key="edit_name")
        dob_default = date.fromisoformat(profile.get("dob")) if profile.get("dob") else date(2023,9,16)
        dob_val = st.date_input("DOB", value=dob_default, key="edit_dob")
        sex_list = ["Prefer not to say", "Male", "Female"]
        current_sex = profile.get("sex","Prefer not to say")
        try:
            init_index = sex_list.index(current_sex)
        except Exception:
            init_index = 0
        sex_opt = st.selectbox("Sex", sex_list, index=init_index, key="edit_sex")
        city_val = st.text_input("City", value=profile.get("city",""), key="edit_city")
        submitted = st.form_submit_button("Save profile", key="edit_save")
        if submitted:
            new_profile = {"name": n, "dob": dob_val.isoformat(), "sex": sex_opt, "city": city_val}
            if mongo_available:
                child_col.delete_many({"user_id": st.session_state.user_id})
                child_col.insert_one({**new_profile, "user_id": st.session_state.user_id, "created_at": datetime.utcnow().isoformat()})
            else:
                cf = DATA_DIR / f"child_{st.session_state.user_id or 'guest'}.json"
                save_json(cf, new_profile)
            st.success("Saved profile.")
            safe_rerun()
    if st.button("Logout (clear session)", key="logout_btn"):
        st.session_state.user = None
        st.session_state.user_id = None
        safe_rerun()
    card_end()

def page_account_profile():
    card_start("Account Profile")
    profile = get_user_profile_record()
    st.write("Username:", profile.get("username", st.session_state.get("user", "Guest")))
    with st.form("account_profile"):
        email = st.text_input("Email", value=profile.get("email",""), key="acct_email")
        phone = st.text_input("Phone", value=profile.get("phone",""), key="acct_phone")
        uploaded = st.file_uploader("Upload profile photo", type=["png","jpg","jpeg","webp"], accept_multiple_files=False, key="acct_photo")
        submitted = st.form_submit_button("Save account profile", key="acct_save")
        if submitted:
            saved_photo_path = profile.get("photo_path","")
            if uploaded is not None:
                saved_photo_path = save_uploaded_user_photo(uploaded, st.session_state.user_id or "guest")
            updates = {"email": email, "phone": phone, "photo_path": saved_photo_path}
            ok = save_user_profile_updates(updates)
            if ok:
                st.success("Profile saved.")
            else:
                st.error("Failed to save profile (DB may be unreachable).")
            safe_rerun()
    if profile.get("photo_path") and Path(profile.get("photo_path")).exists():
        avatar_uri = build_data_uri(Path(profile.get("photo_path")))
        if avatar_uri:
            st.image(avatar_uri, width=160)
    else:
        st.info("No profile photo. Upload one to personalize the app.")
    card_end()

def page_rag_assistant():
    card_start("🧠 Verified Data Assistant", "Ask questions based on the verified local dataset (vaccines, clinics, schemes).")
    if rag_pipeline is None:
        st.warning("RAG pipeline not initialized. It may fail if GEMINI_API_KEY or VECTOR_DB not configured.")
    col_q, col_btn = st.columns([6,1])
    with col_q:
        q = st.text_input("Ask dataset-powered question", placeholder="E.g., 'When is measles vaccine due?'", key="rag_q")
    with col_btn:
        ask = st.button("Ask", key="rag_ask")
    if ask and q:
        try:
            if rag_pipeline:
                ans = rag_pipeline.answer_from_dataset(q)
                st.markdown("**Answer (from dataset):**")
                st.write(ans)
            else:
                st.info("RAG pipeline not available — returning simple fallback.")
                st.write("Fallback: Measles vaccine is typically given at 9 months (check local schedule).")
        except Exception as e:
            st.error(f"RAG query failed: {e}")
    st.markdown("---")
    st.subheader("Upload a vaccination card / prescription photo")
    uploaded_img = st.file_uploader("Upload image (photo of card)", type=["png","jpg","jpeg","webp"], accept_multiple_files=False, key="rag_card")
    ocr_text = ""
    if uploaded_img is not None:
        fd, tmp = tempfile.mkstemp(suffix=Path(uploaded_img.name).suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded_img.getbuffer())
        st.info("Running OCR on the uploaded image...")
        ocr_text = ocr_image_to_text(tmp)
        if not ocr_text:
            st.info("OCR returned no text or OCR engine not available.")
        # show the OCR output (or the helpful install message)
        st.text_area("OCR text (preview)", value=ocr_text, height=200, key="ocr_preview")
        if st.button("Extract fields from OCR", key="extract_ocr"):
            if not ocr_text:
                st.error("Cannot extract fields: OCR did not return text. See the OCR preview for details / installation instructions.")
            else:
                if GEMINI_API_KEY:
                    with st.spinner("Thinking..."):
                        try:
                            prompt = (
                                "Extract vaccination card fields (name, dob, vaccine, dose, date, batch) and return JSON only. OCR follows.\n\n"
                                f"OCR:\n{ocr_text}"
                            )
                            extracted = call_gemini(prompt, max_tokens=400, temperature=0.0)
                            st.markdown("**Extracted (LLM):**")
                            st.code(extracted, language="json")
                        except Exception:
                            st.error("Extraction failed (AI unavailable).")
                else:
                    st.info("AI not configured — cannot run extraction.")

    st.markdown("---")
    st.subheader("Upload voice note (wav/mp3/m4a)")
    uploaded_audio = st.file_uploader("Upload voice note", type=["wav","mp3","m4a","ogg"], key="rag_voice")
    if uploaded_audio:
        fd, tmp_audio = tempfile.mkstemp(suffix=Path(uploaded_audio.name).suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        st.info("Transcribing (placeholder)...")
        try:
            transcript = transcribe_audio_local_placeholder(tmp_audio)
            st.text_area("Transcript", value=transcript, height=120, key="rag_transcript")
            if st.button("Ask dataset with transcript", key="rag_transcript_ask"):
                try:
                    if rag_pipeline:
                        ans = rag_pipeline.answer_from_dataset(transcript)
                        st.write(ans)
                    else:
                        st.info("RAG pipeline not available — fallback: check vaccine schedule at 9 months.")
                except Exception as e:
                    st.error(f"Failed to answer from transcript: {e}")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
    st.markdown("---")
    st.subheader("Upload a symptom photo (rash, swelling, wound)")
    uploaded_sym = st.file_uploader("Upload symptom image (photo of rash/skin/etc)", type=["png","jpg","jpeg","webp"], accept_multiple_files=False, key="symptom_photo")
    if uploaded_sym is not None:
        fd, tmp_sym = tempfile.mkstemp(suffix=Path(uploaded_sym.name).suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded_sym.getbuffer())
        st.info("Analyzing image locally...")
        local_ocr = ocr_text if ocr_text else ""
        vlm_out = vlm_symptom_flow(tmp_sym, local_ocr)
        local = vlm_out.get("local", {})
        if local:
            st.markdown(f"**Local analysis:** {local.get('finding','')}  — Confidence {local.get('confidence',0):.2f}")
            st.markdown(f"<small style='color:#666'>{local.get('details','')}</small>", unsafe_allow_html=True)
            st.markdown("**Quick flags (heuristic)**")
            st.markdown(f"- Rash on face: **{'Yes' if local.get('rash_on_face') else 'No'}**")
            st.markdown(f"- Swelling detected: **{'Yes' if local.get('swelling_detected') else 'No'}**")
            st.markdown(f"- Wound-like signs: **{'Yes' if local.get('wound_detected') else 'No'}**")
            if local.get("annotated_path"):
                st.image(local.get("annotated_path"), caption="Annotated preview", use_column_width=True)
        else:
            st.info("No local analysis result.")

        # LLM enriched analysis button (optional)
        if GEMINI_API_KEY:
            if st.button("Ask AI for more detail from this image", key="ask_ai_image"):
                with st.spinner("Thinking..."):
                    try:
                        vlm_out2 = vlm_symptom_flow(tmp_sym, local_ocr)
                        if vlm_out2.get("llm"):
                            st.markdown("**LLM enriched analysis**")
                            st.write(vlm_out2.get("llm"))
                        else:
                            st.info("AI did not return additional info.")
                    except Exception:
                        st.error("AI call failed.")
        else:
            st.info("AI not configured — only local analysis shown.")
    card_end()

# Navigation & routing (unchanged)
st.sidebar.title("Navigation")
pages = [
    "AI Parenting Assistant",
    "Vaccine Tracker",
    "Baby Health Log",
    "Nearby Hospitals",
    "Growth Tracker",
    "Voice Ask",
    "Baby 1000 Days Guide",
    "Verified Data Assistant",
    "Edit Profile / Logout",
    "Account Profile"
]
if "page" not in st.session_state or not st.session_state.page:
    st.session_state.page = pages[0]

page_choice = st.sidebar.radio(
    "Go to",
    pages,
    index=pages.index(st.session_state.page),
    key="nav_radio",
)
st.session_state.page = page_choice

route_map = {
    "AI Parenting Assistant": page_assistant,
    "Vaccine Tracker": page_vaccine,
    "Baby Health Log": page_health,
    "Nearby Hospitals": page_nearby,
    "Growth Tracker": page_growth,
    "Voice Ask": page_voice,
    "Baby 1000 Days Guide": page_guide,
    "Verified Data Assistant": page_rag_assistant,
    "Edit Profile / Logout": page_edit_profile,
    "Account Profile": page_account_profile
}

route_map.get(st.session_state.page, page_assistant)()

# Footer
st.markdown("---")
st.markdown("<div class='muted'>🔒 <strong>All advice is for awareness only.</strong> For emergencies, consult a certified pediatrician or call local emergency services.</div>", unsafe_allow_html=True)
