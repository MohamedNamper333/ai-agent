"""Retriever - with hybrid search, semantic chunking, and re-ranking"""

import os
import math
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Optional[Embedder] = None, store: Optional[VectorStore] = None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.doc_count = 0
        self.doc_lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.corpus_tokens: list[list[str]] = []
        self._query_cache: Dict[str, tuple[float, list[dict]]] = {}
        self._cache_expiry = 300

    def load(self):
        self.embedder.load()
        self.store.load()
        self._build_bm25_index()

    def _build_bm25_index(self):
        all_entries = self.store.get_all_entries()
        self.doc_count = len(all_entries)
        self.doc_lengths = []
        self.idf = {}
        self.corpus_tokens = []

        for entry in all_entries:
            tokens = self._tokenize(entry["text"])
            self.doc_lengths.append(len(tokens))
            self.corpus_tokens.append(tokens)
            for token in set(tokens):
                self.idf[token] = self.idf.get(token, 0) + 1

        for token in self.idf:
            self.idf[token] = math.log((self.doc_count + 1) / (self.idf[token] + 1)) + 1

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        score = 0.0
        k1 = 1.5
        b = 0.75
        doc_len = len(doc_tokens)
        avg_dl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        for token in query_tokens:
            idf = self.idf.get(token, 0)
            if idf == 0:
                continue
            tf = doc_tokens.count(token)
            if tf == 0:
                continue
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
        return score

    def add_document(self, text: str, metadata: Optional[dict] = None) -> int:
        chunks = self._chunk_text_semantic(text)
        ids = []
        new_tokens = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedder.embed(chunk)
            meta = dict(metadata or {})
            meta["chunk"] = i
            meta["total_chunks"] = len(chunks)
            idx = self.store.add(chunk, embedding, meta)
            ids.append(idx)
            tokens = self._tokenize(chunk)
            new_tokens.append(tokens)
            self.doc_count += 1
            self.doc_lengths.append(len(tokens))
            for token in set(tokens):
                self.idf[token] = self.idf.get(token, 0) + 1
            self.corpus_tokens.append(tokens)
        for token in list(self.idf.keys()):
            self.idf[token] = math.log((self.doc_count + 1) / (self.idf[token] + 1)) + 1
        self._query_cache.clear()
        return len(chunks)

    def add_document_batch(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[int]:
        all_ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else None
            num_chunks = self.add_document(text, meta)
            all_ids.append(num_chunks)
        return all_ids

    def query(self, question: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
        if question in self._query_cache:
            timestamp, results = self._query_cache[question]
            if time.time() - timestamp < self._cache_expiry:
                return results

        q_emb = self.embedder.embed(question)
        semantic_results = self.store.search(q_emb, top_k=top_k * 2)

        query_tokens = self._tokenize(question)
        keyword_results = []

        if hasattr(self, 'doc_lengths') and self.doc_count > 0:
            scored_docs = []
            for i, doc_tokens in enumerate(self.corpus_tokens):
                score = self._bm25_score(query_tokens, doc_tokens)
                if score > 0:
                    scored_docs.append((score, i))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            for score, idx in scored_docs[:top_k * 2]:
                if idx < len(self.store.entries):
                    entry = self.store.entries[idx]
                    keyword_results.append({
                        "score": score,
                        "text": entry["text"],
                        "metadata": entry["metadata"],
                        "id": entry["id"],
                    })

        combined_results = {}
        entry_map = {e["id"]: e for e in self.store.entries}
        for res in semantic_results:
            eid = res["id"]
            combined_results[eid] = combined_results.get(eid, 0) * 0.5 + (res["score"] * alpha)
            if eid not in entry_map:
                entry_map[eid] = {"text": res["text"], "metadata": res["metadata"], "id": eid}

        for i, res in enumerate(keyword_results):
            eid = res["id"]
            keyword_score = (1.0 / (i + 1)) * (1 - alpha)
            combined_results[eid] = combined_results.get(eid, 0) + keyword_score
            if eid not in entry_map:
                entry_map[eid] = {"text": res["text"], "metadata": res["metadata"], "id": eid}

        final_scored = []
        for eid, combined_score in combined_results.items():
            entry = entry_map.get(eid)
            if entry:
                final_scored.append({
                    "text": entry["text"],
                    "score": combined_score,
                    "metadata": entry.get("metadata", {}),
                    "id": eid,
                })

        final_scored.sort(key=lambda x: x["score"], reverse=True)
        results = final_scored[:top_k]

        self._query_cache[question] = (time.time(), results)
        return results

    def query_text(self, question: str, top_k: int = 3) -> str:
        results = self.query(question, top_k=top_k)
        if not results:
            return ""
        parts = []
        for r in results:
            src = r["metadata"].get("source", "unknown")
            score = r.get("score", 0)
            parts.append(f"[From {src} (score: {score:.3f})] {r['text']}")
        return "\n\n".join(parts)

    def index_file(self, filepath: str) -> int:
        path = Path(filepath)
        if not path.exists():
            return 0
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks = self._chunk_text_semantic(text)
            metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]
            for i, chunk in enumerate(chunks):
                embedding = self.embedder.embed(chunk)
                self.store.add(chunk, embedding, metadatas[i])
            self._build_bm25_index()
            return len(chunks)
        except Exception:
            return 0

    def index_directory(self, directory: str, patterns: list[str] = None) -> dict:
        root = Path(directory)
        if not root.exists():
            return {"error": f"Directory not found: {directory}"}
        if not patterns:
            patterns = ["*.txt", "*.md", "*.py", "*.json", "*.csv"]

        stats = {"total_files": 0, "total_chunks": 0, "files": []}
        for pattern in patterns:
            for filepath in root.rglob(pattern):
                if any(x in str(filepath) for x in ("__pycache__", ".venv", ".git", "venv")):
                    continue
                num_chunks = self.index_file(str(filepath))
                if num_chunks > 0:
                    stats["total_files"] += 1
                    stats["total_chunks"] += num_chunks
                    stats["files"].append({"path": str(filepath), "chunks": num_chunks})

        self.store.save()
        return stats

    def delete_source(self, source: str) -> int:
        deleted = self.store.delete_by_source(source)
        self.store.save()
        self._build_bm25_index()
        return deleted

    def get_stats(self) -> dict:
        return {
            "vector_entries": len(self.store.entries),
            "chunked": True,
            "embedder": self.embedder.get_info(),
            "store": self.store.get_stats(),
        }

    def _chunk_text_semantic(self, text: str,
                              chunk_size: int = 512,
                              overlap: int = 64,
                              min_chunk: int = 100) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                if len(para) > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) < chunk_size:
                            current_chunk += sent + " "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sent + " "
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        final_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk) < min_chunk and i > 0:
                final_chunks[-1] += "\n\n" + chunk
            else:
                final_chunks.append(chunk)

        return final_chunks

    def save(self):
        self.store.save()

    def load_or_init(self):
        self.load()
