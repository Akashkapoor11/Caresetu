# db_helpers.py
"""
Simple in-memory VectorDB stub.

This stores documents (text + meta) in a list and does a naive ranking
by counting common tokens. It's intentionally tiny and dependency-free,
just to allow the UI/testing flow to work without a real vector DB.
"""

from typing import List, Dict, Any
import re

class VectorDB:
    def __init__(self):
        self._docs = []  # list of dicts: {"id": id, "text": text, "meta": {...}}

    def add(self, text: str, meta: Dict[str, Any] = None):
        meta = meta or {}
        doc = {"id": len(self._docs)+1, "text": text, "meta": meta}
        self._docs.append(doc)
        return doc["id"]

    def search(self, query: str, top_k: int = 3):
        """
        Naive search: score by number of shared tokens.
        Returns list of docs sorted by score descending.
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for d in self._docs:
            tokens = set(re.findall(r"\w+", d["text"].lower()))
            score = len(q_tokens & tokens)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [d for score, d in scored if score > 0][:top_k]
        # if no matches, return top_k docs as fallback with score 0
        if not results:
            return [d for _, d in sorted([(0,d) for d in self._docs], key=lambda x: x[0], reverse=True)][:top_k]
        return results

    def all_docs(self):
        return list(self._docs)
