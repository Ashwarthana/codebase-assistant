"""Pydantic models for the RAG Codebase Assistant API."""

from pydantic import BaseModel
from typing import List, Optional, Any


class IndexRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    question: str
    mode: str = "proposed"  # "proposed" or "baseline"
    k: int = 5  # number of chunks to retrieve


class CitationModel(BaseModel):
    file: str
    start: int
    end: int


class AskResponse(BaseModel):
    status: str
    answer: str
    confidence: float
    citations: List[CitationModel]
    citation_valid: bool
    evidence: List[dict]
    mode: str
    retrieved_files: List[str] = []
    overlap_score: float = 0.0


class StatsResponse(BaseModel):
    total_chunks: int
    total_files: int
    index_loaded: bool
    generator_type: str
