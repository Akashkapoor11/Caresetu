# model.py
from typing import List
import json

class RAGPipeline:
    def __init__(self, vector_db, ai_client=None):
        self.vector_db = vector_db
        self.ai_client = ai_client

    def answer_from_dataset(self, question: str) -> str:
        """
        Search the vector DB, then either ask the AI client to synthesize an answer
        or return a simple summary of top documents.
        """
        docs = self.vector_db.search(question, top_k=3)
        if not docs:
            return "No dataset documents available."

        # Prepare context
        ctx = "\n\n---\n\n".join([f"Doc {d['id']}:\n{d['text']}" for d in docs])

        if self.ai_client:
            prompt = (
                "You are a helpful assistant that answers questions using only the provided context below.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely and cite any doc numbers used (e.g., Doc 1). If uncertain, say you are unsure."
            )
            try:
                resp = self.ai_client.generate(prompt, max_tokens=250, temperature=0.0)
                # try to ensure it's a plain string
                if isinstance(resp, (dict, list)):
                    return json.dumps(resp)
                return resp
            except Exception as e:
                return f"[AI client failed: {e}]\n\nContext summary:\n" + ctx
        # No AI client: return text summary
        summary_lines = []
        for d in docs:
            txt = d.get("text","").strip().replace("\n", " ")
            summary_lines.append(f"Doc {d['id']}: {txt[:300]}{'...' if len(txt)>300 else ''}")
        return "No LLM available — showing matched documents:\n\n" + "\n\n".join(summary_lines)
