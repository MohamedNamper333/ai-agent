import os
from pathlib import Path
from typing import Optional

from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Optional[Embedder] = None, store: Optional[VectorStore] = None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def load(self):
        self.embedder.load()
        self.store.load()

    def add_document(self, text: str, metadata: Optional[dict] = None) -> int:
        embedding = self.embedder.embed(text)
        return self.store.add(text, embedding, metadata)

    def add_document_batch(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[int]:
        embeddings = self.embedder.embed_batch(texts)
        ids = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            meta = metadatas[i] if metadatas else None
            ids.append(self.store.add(text, emb, meta))
        return ids

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        q_emb = self.embedder.embed(question)
        return self.store.search(q_emb, top_k=top_k)

    def query_text(self, question: str, top_k: int = 3) -> str:
        results = self.query(question, top_k=top_k)
        if not results:
            return ""
        parts = []
        for r in results:
            src = r["metadata"].get("source", "unknown")
            parts.append(f"[From {src}] {r['text']}")
        return "\n\n".join(parts)

    def index_file(self, filepath: str) -> int:
        path = Path(filepath)
        if not path.exists():
            return 0

        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(text)
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]
        ids = self.add_document_batch(chunks, metadatas)
        self.store.save()
        return len(ids)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def save(self):
        self.store.save()

    def load_or_init(self):
        self.load()
