"""
AST-based Python code chunker with advanced features:
- Decorator extraction (includes @decorators in chunk code)
- Nested-class qualified names (e.g. ClassName.method_name)
- Module-level docstrings for file overviews
- Import extraction per file
- Sliding-window fallback for very large chunks (>100 lines)
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

SKIP_DIRS = {
    "venv", ".venv", "__pycache__", "tests", "test", ".git",
    "migrations", "node_modules", ".tox", "dist", "build", "egg-info",
}
MIN_CODE_LINES = 3
MAX_CHUNK_LINES = 100
OVERLAP_LINES = 10


def _get_decorator_start(node: ast.AST, source_lines: list) -> int:
    """Find the actual start line including decorators above the node."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _get_parent_map(tree: ast.Module) -> dict:
    """Build a mapping from child node id -> parent node for qualified names."""
    parent_map = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node
    return parent_map


def _qualified_name(node: ast.AST, parent_map: dict) -> str:
    """Build a qualified name like ClassName.method_name for nested definitions."""
    parts = [node.name]
    current = node
    while id(current) in parent_map:
        parent = parent_map[id(current)]
        if isinstance(parent, ast.ClassDef):
            parts.insert(0, parent.name)
        elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.insert(0, parent.name)
        current = parent
    return ".".join(parts)


def _extract_module_docstring(tree: ast.Module) -> Optional[str]:
    """Extract the module-level docstring if present."""
    return ast.get_docstring(tree)


def _extract_imports(tree: ast.Module) -> List[str]:
    """Extract all import statements from the AST."""
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            imports.append(f"from {module} import {names}")
    return imports


def _split_large_chunk(chunk: dict) -> List[dict]:
    """Split a chunk that exceeds MAX_CHUNK_LINES into overlapping windows."""
    code_lines = chunk["code"].splitlines()
    total = len(code_lines)

    if total <= MAX_CHUNK_LINES:
        return [chunk]

    sub_chunks = []
    window_start = 0
    part_idx = 0

    while window_start < total:
        window_end = min(window_start + MAX_CHUNK_LINES, total)
        sub_code = "\n".join(code_lines[window_start:window_end])
        abs_start = chunk["start_line"] + window_start
        abs_end = chunk["start_line"] + window_end - 1

        sub_chunks.append({
            "name": f"{chunk['name']}__part{part_idx}",
            "type": chunk["type"],
            "file": chunk["file"],
            "start_line": abs_start,
            "end_line": abs_end,
            "docstring": chunk["docstring"] if part_idx == 0 else None,
            "code": sub_code,
            "parent_name": chunk.get("parent_name"),
        })

        part_idx += 1
        # Advance with overlap, break if we already reached the end
        if window_end >= total:
            break
        window_start = window_end - OVERLAP_LINES

    return sub_chunks


def extract_chunks_from_file(filepath: str, repo_root: str) -> List[dict]:
    """Parse one Python file and extract every function/class as a chunk.

    Features:
    - Includes decorators in the extracted code
    - Nested methods get qualified names (e.g. User.verify_password)
    - Large chunks (>100 lines) are split into overlapping windows
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    rel_path = os.path.relpath(filepath, repo_root)
    rel_path = rel_path.replace("\\", "/")
    parent_map = _get_parent_map(tree)
    raw_chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = _get_decorator_start(node, source_lines)
            end_line = getattr(node, "end_lineno", start_line)
            qualified = _qualified_name(node, parent_map)

            # Determine parent class name if nested
            parent_name = None
            if id(node) in parent_map:
                parent = parent_map[id(node)]
                if isinstance(parent, ast.ClassDef):
                    parent_name = parent.name

            chunk = {
                "name": qualified,
                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                "file": rel_path,
                "start_line": start_line,
                "end_line": end_line,
                "docstring": ast.get_docstring(node),
                "code": "\n".join(source_lines[start_line - 1:end_line]),
                "parent_name": parent_name,
            }
            raw_chunks.append(chunk)

    # Split large chunks and filter tiny ones
    chunks = []
    for chunk in raw_chunks:
        for sub in _split_large_chunk(chunk):
            if len(sub["code"].splitlines()) >= MIN_CODE_LINES:
                chunks.append(sub)

    return chunks


def chunk_repository(repo_root: str) -> List[dict]:
    """Chunk all Python files in a repository.

    Returns a list of chunks including:
    - Function/class chunks with decorators and qualified names
    - File overview chunks with imports, module docstrings, and symbol lists
    """
    repo_root = Path(repo_root)
    all_chunks = []

    for py_file in repo_root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        all_chunks.extend(extract_chunks_from_file(str(py_file), str(repo_root)))

    # Build file overviews with richer metadata
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for c in all_chunks:
        by_file[c["file"]].append(c)

    overview_chunks = []
    for file_path, file_chunks in by_file.items():
        full_path = repo_root / file_path
        names = sorted(set(c["name"] for c in file_chunks))
        docs = [c["docstring"] for c in file_chunks if c["docstring"]]

        # Try to get module-level info
        module_doc = None
        imports = []
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
            tree = ast.parse(source)
            module_doc = _extract_module_docstring(tree)
            imports = _extract_imports(tree)
        except (SyntaxError, OSError):
            pass

        # Build rich overview
        summary_parts = [f"File: {file_path}"]
        if module_doc:
            summary_parts.append(f"Module description: {module_doc}")
        if imports:
            summary_parts.append(f"Imports: {', '.join(imports[:15])}")
        summary_parts.append(f"Contains: {', '.join(names)}")
        if docs:
            summary_parts.append("Docstrings: " + " | ".join(docs[:5]))

        overview_chunks.append({
            "name": f"__file_overview__:{file_path}",
            "type": "file_overview",
            "file": file_path,
            "start_line": 1,
            "end_line": 1,
            "docstring": module_doc,
            "code": "\n".join(summary_parts),
            "parent_name": None,
        })

    return all_chunks + overview_chunks
