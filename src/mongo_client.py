# src/mongo_client.py
import os
from .config import settings

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

def get_mongo_client():
    uri = os.getenv("MONGO_URI", settings.MONGO_URI or "") or ""
    if not uri or MongoClient is None:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client
    except Exception:
        return None

# Usage:
# client = get_mongo_client()
# if client: db = client.get_database("caresetu")
