"""
Hybrid retriever combining dense FAISS search, BM25 sparse search,
Reciprocal Rank Fusion (RRF), and cross-encoder reranking with
bidirectional call-graph expansion.
"""

import json
import os
from typing import List, Tuple, Dict, Optional

import networkx as nx
import numpy as np
from rapidfuzz import fuzz

from .indexer import (
    embed_texts,
    l2_normalize,
    load_bm25_index,
    _tokenize_for_bm25,
    chunk_embed_text,
)

# Lazy-loaded cross-encoder for reranking
_cross_encoder = None


def _get_cross_encoder():
    """Lazily load a lightweight cross-encoder for CPU reranking."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


class Retriever:
    """Hybrid retriever with RRF fusion, cross-encoder reranking,
    and bidirectional call-graph expansion."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chunks: List[dict] = []
        self.embeddings = None
        self.bm25 = None
        self.call_graph: Optional[nx.DiGraph] = None
        self.chunk_by_name: Dict[str, List[dict]] = {}
        self.load()

    def load(self) -> None:
        """Load all persisted indices and data."""
        index_path = os.path.join(self.data_dir, "chunks.npy")
        chunks_path = os.path.join(self.data_dir, "chunks.json")
        graph_path = os.path.join(self.data_dir, "call_graph.gml")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.embeddings = np.load(index_path)
            with open(chunks_path) as f:
                self.chunks = json.load(f)

            self.chunk_by_name = {}
            for c in self.chunks:
                self.chunk_by_name.setdefault(c["name"], []).append(c)
                # Also index the base name for qualified names like Class.method
                if "." in c["name"]:
                    base = c["name"].split(".")[-1]
                    self.chunk_by_name.setdefault(base, []).append(c)

        # Load BM25
        self.bm25 = load_bm25_index(self.data_dir)

        # Load call graph
        if os.path.exists(graph_path):
            self.call_graph = nx.read_gml(graph_path)
        else:
            self.call_graph = nx.DiGraph()

    # ── Search methods ────────────────────────────────────────────────

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """Pure dense semantic search via exact cosine similarity using NumPy."""
        if self.embeddings is None or not self.chunks:
            return []
        q_emb = l2_normalize(embed_texts([query])).astype("float32")
        scores = np.dot(self.embeddings, q_emb[0])
        idxs = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[i], float(scores[i]))
            for i in idxs
        ]

    def bm25_search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """Sparse BM25 keyword search."""
        if self.bm25 is None or not self.chunks:
            return []
        tokens = _tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[int(i)], float(scores[i]))
            for i in top_idxs
            if scores[i] > 0
        ]

    def keyword_score(self, query: str, chunk: dict) -> float:
        """Fuzzy lexical match score against chunk name and file path."""
        name_score = fuzz.partial_ratio(query.lower(), chunk["name"].lower()) / 100.0
        path_score = fuzz.partial_ratio(query.lower(), chunk["file"].lower()) / 100.0
        return max(name_score, path_score)

    # ── Hybrid retrieval with RRF ─────────────────────────────────────

    def hybrid_retrieve(
        self,
        query: str,
        k: int = 5,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> List[Tuple[dict, float]]:
        """Reciprocal Rank Fusion of dense + BM25, with optional cross-encoder reranking.

        RRF score for each chunk = 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_bm25)

        After fusion, the top candidates are optionally reranked with a
        cross-encoder, then deduplicated per file (max 2 per file).
        """
        # Get candidate lists
        pool_size = k * 6
        dense_results = self.semantic_search(query, k=pool_size)
        bm25_results = self.bm25_search(query, k=pool_size)

        # Build chunk index -> RRF score mapping
        # Use chunk identity by (file, start_line, name) tuple
        def _chunk_key(c):
            return (c["file"], c["start_line"], c["name"])

        rrf_scores: Dict[tuple, float] = {}
        chunk_map: Dict[tuple, dict] = {}

        # Score from dense ranking
        for rank, (chunk, _) in enumerate(dense_results):
            key = _chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_map[key] = chunk

        # Score from BM25 ranking
        for rank, (chunk, _) in enumerate(bm25_results):
            key = _chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_map[key] = chunk

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        candidates = [(chunk_map[key], rrf_scores[key]) for key in sorted_keys[:pool_size]]

        if not candidates:
            return []

        # ── Cross-encoder reranking ──
        if rerank and len(candidates) > 1:
            try:
                ce = _get_cross_encoder()
                pairs = [[query, chunk_embed_text(c)] for c, _ in candidates[:20]]
                ce_scores = ce.predict(pairs)
                # Convert raw logits to probabilities (0.0 to 1.0)
                ce_scores = 1.0 / (1.0 + np.exp(-ce_scores))
                reranked = sorted(
                    zip([c for c, _ in candidates[:20]], ce_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                candidates = [(c, float(s)) for c, s in reranked]
            except Exception:
                pass  # Fall back to RRF order if cross-encoder fails

        # ── File diversity: max 2 chunks per file ──
        best_per_file: Dict[str, List[Tuple[dict, float]]] = {}
        for chunk, score in candidates:
            f = chunk["file"]
            if f not in best_per_file:
                best_per_file[f] = []
            if len(best_per_file[f]) < 2:
                best_per_file[f].append((chunk, score))

        deduped = []
        for entries in best_per_file.values():
            deduped.extend(entries)
        deduped.sort(key=lambda x: x[1], reverse=True)

        return deduped[:k]

    # ── Call-graph expansion ──────────────────────────────────────────

    def expand_with_call_graph(
        self,
        top_chunks: List[Tuple[dict, float]],
        hops: int = 1,
    ) -> List[dict]:
        """Expand retrieved chunks via bidirectional call-graph traversal.

        Traverses both successors (callees) AND predecessors (callers)
        to capture the full context around a function.
        """
        expanded = {c["name"]: c for c, _ in top_chunks}
        frontier = list(expanded.keys())

        for _ in range(hops):
            next_frontier = []
            for name in frontier:
                # Also check the base name for qualified names
                names_to_check = [name]
                if "." in name:
                    names_to_check.append(name.split(".")[-1])

                for check_name in names_to_check:
                    if self.call_graph and check_name in self.call_graph:
                        # Bidirectional: callees + callers
                        neighbors = set(self.call_graph.successors(check_name))
                        neighbors |= set(self.call_graph.predecessors(check_name))
                        for neighbor in neighbors:
                            if neighbor not in expanded and neighbor in self.chunk_by_name:
                                expanded[neighbor] = self.chunk_by_name[neighbor][0]
                                next_frontier.append(neighbor)
            frontier = next_frontier

        return list(expanded.values())
