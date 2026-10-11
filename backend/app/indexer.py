"""
Dense (FAISS) + Sparse (BM25) indexing, call-graph construction,
and embedding utilities for the RAG Codebase Assistant.
"""

import json
import os
import re
import ast
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DEVICE = "cpu"
MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
embed_model = None


def get_embed_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model."""
    global embed_model
    if embed_model is None:
        torch.set_num_threads(4)
        embed_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return embed_model


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode a list of texts into normalized embedding vectors."""
    model = get_embed_model()
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)


def chunk_embed_text(c: dict) -> str:
    """Build a rich text representation of a chunk for embedding.

    Includes file path words, qualified name, parent class,
    docstring, and code for maximum retrieval signal.
    """
    path_words = c["file"].replace("/", " ").replace("_", " ").replace(".py", "")
    parts = [path_words, c["name"]]
    if c.get("parent_name"):
        parts.append(f"class {c['parent_name']}")
    if c.get("docstring"):
        parts.append(c["docstring"])
    parts.append(c["code"])
    return "\n".join(parts)


# ── BM25 tokenization ────────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 by splitting on whitespace, underscores,
    and camelCase boundaries."""
    # Split camelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on underscores, dots, slashes
    text = re.sub(r"[_./\\]", " ", text)
    return [t.lower() for t in text.split() if len(t) >= 2]


def build_bm25_index(chunks: List[dict], data_dir: str) -> None:
    """Build a BM25 index and save the tokenized corpus to disk."""
    corpus = [_tokenize_for_bm25(chunk_embed_text(c)) for c in chunks]
    corpus_path = os.path.join(data_dir, "bm25_corpus.json")
    with open(corpus_path, "w") as f:
        json.dump(corpus, f)


def load_bm25_index(data_dir: str) -> Optional[BM25Okapi]:
    """Load a saved BM25 corpus and reconstruct the index."""
    corpus_path = os.path.join(data_dir, "bm25_corpus.json")
    if not os.path.exists(corpus_path):
        return None
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    if not corpus:
        return None
    return BM25Okapi(corpus)


# ── Call-graph construction ───────────────────────────────────────────

class CallVisitor(ast.NodeVisitor):
    """AST visitor that extracts function-to-function call edges."""

    def __init__(self, known_functions: set):
        self.current_func = None
        self.edges: List[Tuple[str, str]] = []
        self.known_functions = known_functions

    def visit_FunctionDef(self, node):
        prev = self.current_func
        self.current_func = node.name
        self.generic_visit(node)
        self.current_func = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self.current_func:
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee and callee in self.known_functions:
                self.edges.append((self.current_func, callee))
        self.generic_visit(node)


def find_calls_in_file(filepath: str, known_functions: set) -> List[Tuple[str, str]]:
    """Extract call edges from a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []
    visitor = CallVisitor(known_functions)
    visitor.visit(tree)
    return visitor.edges


# ── Main build function ───────────────────────────────────────────────

SKIP_DIRS = {"venv", ".venv", "__pycache__", "tests", "test", ".git", "migrations", "node_modules"}


def build_index_and_graph(chunks: List[dict], repo_dir: str, data_dir: str) -> None:
    """Build dense FAISS index, sparse BM25 index, and call graph.

    Saves everything to data_dir:
    - chunks.index  (FAISS)
    - chunks.json   (chunk metadata)
    - bm25_corpus.json (BM25 tokenized corpus)
    - call_graph.gml (NetworkX call graph)
    """
    os.makedirs(data_dir, exist_ok=True)

    # ── Dense index (Pure NumPy) ──
    chunk_texts = [chunk_embed_text(c) for c in chunks]
    chunk_embeddings = l2_normalize(embed_texts(chunk_texts))

    np.save(os.path.join(data_dir, "chunks.npy"), chunk_embeddings.astype("float32"))
    with open(os.path.join(data_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2)

    # ── BM25 index ──
    build_bm25_index(chunks, data_dir)

    # ── Call graph ──
    # Extract the base function name (without class prefix) for matching
    known_functions = set()
    for c in chunks:
        if c["type"] in ("function",):
            name = c["name"]
            # Handle qualified names like ClassName.method
            if "." in name:
                known_functions.add(name.split(".")[-1])
            known_functions.add(name)

    call_graph = nx.DiGraph()
    call_graph.add_nodes_from(known_functions)

    repo_root = Path(repo_dir)
    for py_file in repo_root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        call_graph.add_edges_from(find_calls_in_file(str(py_file), known_functions))

    nx.write_gml(call_graph, os.path.join(data_dir, "call_graph.gml"))
