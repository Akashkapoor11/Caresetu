# src/repositories.py
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / ".data"

def find_clinics_in_city(city_name: str, limit: int = 10):
    try:
        with open(DATA_DIR / "hospitals.json", "r", encoding="utf-8") as f:
            hospitals = json.load(f)
    except Exception:
        hospitals = {}
    key = city_name.strip().lower()
    out = hospitals.get(key, hospitals.get("default", []))
    return out[:limit]
