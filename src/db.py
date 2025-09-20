# src/db.py
from pathlib import Path
import sqlite3
from .config import settings

DB_PATH = Path(settings.DB_PATH).resolve()

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT,
        email TEXT,
        phone TEXT,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vaccines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        vaccine_name TEXT,
        given_on TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS health_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        created_on TEXT,
        note TEXT
    )""")
    conn.commit()
    conn.close()
