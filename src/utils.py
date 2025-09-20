# utils.py
import os
import tempfile
from pathlib import Path

# optional imports
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

def ocr_image_to_text(path_or_bytes):
    """
    Try to run pytesseract OCR on the provided file path.
    If pytesseract / tesseract binary not available, return a helpful placeholder.
    Accepts either a path string or bytes-like object.
    """
    # Accept bytes (e.g., uploaded file), or path
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            # write to temp file and run
            fd, tmp = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(path_or_bytes)
            img_path = tmp
        else:
            img_path = str(path_or_bytes)

        # Use pytesseract if present
        if pytesseract is None or Image is None:
            return f"OCR placeholder: would extract text from {img_path} (no OCR library configured)."

        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang="eng")
            if not text or text.strip() == "":
                return f"OCR ran but returned no text for {img_path}."
            return text
        except Exception as e:
            return f"OCR failed: {e}"

    finally:
        # if we created a temp file for bytes, attempt to remove it
        try:
            if isinstance(path_or_bytes, (bytes, bytearray)):
                os.remove(tmp)
        except Exception:
            pass


def transcribe_audio_local_placeholder(path):
    """
    Placeholder transcription when speech libraries aren't installed.
    """
    return "Transcription placeholder: audio file processed locally (no speech lib configured)."

def save_media_from_url(url, dest=None):
    """
    Minimal helper: download a remote media file to dest and return path.
    This function is intentionally minimal (no requests import) and used only as a placeholder.
    """
    raise NotImplementedError("save_media_from_url is not implemented in local stub.")
