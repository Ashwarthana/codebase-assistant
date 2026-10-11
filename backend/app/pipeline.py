"""
Grounded generation pipeline with:
- Chain-of-thought prompting
- Structured output parsing (STATUS / CITATIONS / ANSWER)
- Citation validation against retrieved evidence
- Evidence-overlap scoring to catch ungrounded answers
- Abstention detection
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35  # Lower than before since RRF scores differ

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_GENERIC_CODE_WORDS = {
    "def", "self", "return", "class", "import", "from", "the", "and", "for",
    "that", "this", "with", "function", "method", "app", "flask", "none",
    "true", "false", "not", "are", "does", "then", "else", "elif", "try",
    "except", "raise", "pass", "yield", "print", "value", "values", "object",
    "objects", "using", "used", "use", "when", "which", "into", "get", "set",
    "new", "via", "user", "users", "str", "int", "list", "dict", "bool",
    "type", "name", "file", "line", "code", "data", "test", "init", "main",
    "args", "kwargs", "None", "True", "False", "has", "have", "been", "will",
}

_ABSTENTION_PHRASES = (
    "does not contain", "cannot determine", "not enough information",
    "i don't know", "i do not know", "cannot answer", "no information",
    "not possible to determine", "insufficient", "not clear from",
    "no relevant", "outside the scope", "not related", "unable to find",
    "cannot find", "no evidence", "not provided", "beyond the scope",
)


# ── Prompt construction ──────────────────────────────────────────────

def build_grounded_prompt(query: str, context_chunks: List[dict]) -> str:
    """Build a grounded, chain-of-thought prompt for the LLM.

    Instructs the LLM to:
    1. Reason about which evidence is relevant
    2. Output structured STATUS / CITATIONS / ANSWER
    3. Never invent files, functions, or line numbers
    """
    blocks = [
        f"# {c['file']}:{c['start_line']}-{c['end_line']} ({c['name']})\n{c['code']}"
        for c in context_chunks
    ]
    context_str = "\n\n".join(blocks) if blocks else "(no relevant code retrieved)"

    return (
        "You are a repository-grounded codebase assistant.\n"
        "Use ONLY the supplied repository evidence below. Do not use pretrained "
        "knowledge to fill missing information, and do not guess.\n"
        "Every repository-specific factual statement must be supported by the evidence.\n"
        "If the evidence does not support an answer, or the question is unrelated to this "
        "repository, abstain by returning STATUS: INSUFFICIENT_EVIDENCE.\n"
        "Never invent files, functions, classes, variables, APIs, or line numbers.\n\n"
        "INSTRUCTIONS:\n"
        "1. First, reason step-by-step about which pieces of evidence are relevant to "
        "the question. Think about what the code does and how it connects.\n"
        "2. Then output your response in PLAIN TEXT ONLY (no markdown, no bold, no "
        "bullet points, no code fences) in EXACTLY this format.\n"
        "3. You MUST always include all three sections IN THIS ORDER:\n\n"
        "STATUS: SUPPORTED or INSUFFICIENT_EVIDENCE\n\n"
        "CITATIONS:\n"
        "<one per line as file.py:start-end, taken only from the file/line ranges "
        "shown in the evidence below, or 'None'>\n\n"
        "ANSWER:\n"
        "<answer grounded only in the evidence, or the sentence "
        "'The repository does not contain sufficient evidence to answer this question.'>\n\n"
        "EXAMPLE (copy the SHAPE and ORDER only, never this content):\n"
        "STATUS: SUPPORTED\n\n"
        "CITATIONS:\n"
        "app/models.py:120-135\n"
        "app/auth/views.py:42-58\n\n"
        "ANSWER:\n"
        "The Comment model stores a body, a timestamp, and a foreign key to the "
        "author who posted it. The login view authenticates via check_password.\n\n"
        "---\n\n"
        f"REPOSITORY EVIDENCE:\n{context_str}\n\n"
        f"QUESTION: {query}\n"
    )


# ── Output normalization & parsing ───────────────────────────────────

def normalize_llm_output(text: str) -> str:
    """Strip markdown artifacts and normalize whitespace."""
    # Remove code fences
    text = re.sub(r"```[a-zA-Z]*\n?", "\n", text)
    text = text.replace("```", "")
    # Remove markdown formatting
    text = text.replace("**", "").replace("`", "")
    # Remove heading markers
    text = re.sub(r"^\s{0,3}#+\s*", "", text, flags=re.MULTILINE)
    # Remove bullet points
    text = re.sub(r"^\s{0,3}[-*•]\s+", "", text, flags=re.MULTILINE)
    # Remove numbered lists
    text = re.sub(r"^\s{0,3}\d+\.\s+", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# Citation regexes — handle many formats:
#   file.py:10-20, file.py:L10-L20, file.py lines 10-20, file.py (lines 10-20)
_RANGE_RE = re.compile(
    r"([\w\-/\.]+\.py)\s*(?::?\s*(?:L|lines?\s*)?)(\d+)\s*(?:[-–—]|to)\s*(?:L?)(\d+)",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(
    r"([\w\-/\.]+\.py)\s*(?::?\s*(?:L|line\s*)?)\s*(\d+)(?!\s*[-–—\d]|(?:\s*to))",
    re.IGNORECASE,
)

_STATUS_RE = re.compile(
    r"STATUS\s*:?\s*(SUPPORTED|INSUFFICIENT[_ ]EVIDENCE)",
    re.IGNORECASE,
)


def extract_citations(text: str) -> List[Tuple[str, int, int]]:
    """Extract file:start-end citation tuples from LLM output."""
    citations, seen_spans = [], []

    for m in _RANGE_RE.finditer(text):
        citations.append((m.group(1), int(m.group(2)), int(m.group(3))))
        seen_spans.append(m.span())

    for m in _SINGLE_RE.finditer(text):
        if any(m.start() >= s0 and m.end() <= s1 for s0, s1 in seen_spans):
            continue
        line = int(m.group(2))
        citations.append((m.group(1), line, line))

    return citations


def parse_structured_output(raw_text: str) -> dict:
    """Parse the LLM's structured output into status, answer, and citations."""
    text = normalize_llm_output(raw_text)

    # Parse status
    status_match = _STATUS_RE.search(text)
    if status_match:
        status = status_match.group(1).upper().replace(" ", "_")
    else:
        status = "INSUFFICIENT_EVIDENCE"

    # Parse answer — try ANSWER: section first, then fallback
    answer_match = re.search(
        r"ANSWER\s*:?\s*(.*?)(?:CITATIONS\s*:|STATUS\s*:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if answer_match and answer_match.group(1).strip():
        answer = answer_match.group(1).strip()
    elif "ANSWER" in text.upper():
        # Split on ANSWER and take everything after
        parts = re.split(r"ANSWER\s*:?\s*", text, flags=re.IGNORECASE)
        answer = parts[-1].strip() if len(parts) > 1 else text.strip()
    else:
        answer = text.strip()

    # Parse citations
    citations = extract_citations(text)

    return {
        "status": status,
        "answer": answer,
        "citations": citations,
        "normalized_text": text,
    }


# ── Citation validation ──────────────────────────────────────────────

def validate_citations(
    citations: List[Tuple[str, int, int]],
    context_chunks: List[dict],
) -> bool:
    """Check that every citation references lines within the retrieved evidence."""
    if not citations:
        return False

    by_file: Dict[str, List[Tuple[int, int]]] = {}
    for c in context_chunks:
        by_file.setdefault(c["file"], []).append((c["start_line"], c["end_line"]))

    for file, start, end in citations:
        if file not in by_file:
            return False
        intervals = by_file[file]
        overlaps_any = any(
            max(start, cs) <= min(end, ce) for cs, ce in intervals
        )
        union_lo = min(cs for cs, ce in intervals)
        union_hi = max(ce for cs, ce in intervals)
        within_union = union_lo <= start and end <= union_hi
        if not (overlaps_any or within_union):
            return False
    return True


# ── Evidence overlap scoring ─────────────────────────────────────────

def extract_evidence_tokens(context_chunks: List[dict]) -> set:
    """Extract meaningful tokens from evidence chunks."""
    tokens = set()
    for c in context_chunks:
        if c.get("name") and not c["name"].startswith("__"):
            tokens.add(c["name"].lower())
        for text in (c.get("code") or "", c.get("docstring") or ""):
            tokens.update(w.lower() for w in _WORD_RE.findall(text))
    return tokens - _GENERIC_CODE_WORDS


def evidence_overlap_score(answer: str, context_chunks: List[dict]) -> float:
    """Compute what fraction of specific identifiers in the answer
    appear in the evidence. Returns 0.0-1.0.

    Used to detect when the LLM invents information not in the evidence.
    """
    answer_tokens = {w.lower() for w in _WORD_RE.findall(answer)}
    answer_specific = answer_tokens - _GENERIC_CODE_WORDS
    if not answer_specific:
        return 0.0

    evidence_tokens = extract_evidence_tokens(context_chunks)
    if not evidence_tokens:
        return 0.0

    matched = len(answer_specific & evidence_tokens)
    return matched / len(answer_specific)


def infer_citations_from_overlap(
    answer: str,
    context_chunks: List[dict],
    min_overlap: int = 2,
    max_citations: int = 3,
) -> List[Tuple[str, int, int]]:
    """Infer citations by finding which chunks share the most specific
    identifiers with the answer text."""
    answer_tokens = {w.lower() for w in _WORD_RE.findall(answer)}
    scored = []
    for c in context_chunks:
        tokens = extract_evidence_tokens([c])
        if not tokens:
            continue
        overlap = len(tokens & answer_tokens)
        if overlap >= min_overlap:
            scored.append((overlap, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        (c["file"], c["start_line"], c["end_line"])
        for _, c in scored[:max_citations]
    ]


def looks_like_abstention(raw_text: str) -> bool:
    """Detect if the LLM's response is effectively an abstention."""
    lowered = raw_text.lower()
    return any(p in lowered for p in _ABSTENTION_PHRASES)


# ── Main pipeline functions ──────────────────────────────────────────

def answer_question_proposed(
    query: str,
    retriever,
    generator,
    k: int = 5,
    hops: int = 1,
) -> dict:
    """PROPOSED pipeline: hybrid retrieval → call-graph expansion →
    confidence gate → grounded generation → citation validation →
    evidence overlap check.
    """
    top = retriever.hybrid_retrieve(query, k=k)
    top_score = top[0][1] if top else -1.0
    retrieved_files = [c["file"] for c, _ in top]

    # Confidence gate
    if top_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": "The repository does not contain sufficient evidence to answer this question.",
            "citations": [],
            "confidence": top_score,
            "retrieved_files": retrieved_files,
            "citation_valid": False,
            "evidence": [],
            "mode": "proposed",
            "overlap_score": 0.0,
        }

    # Expand context via call graph and limit to 10 chunks to allow more file context
    context_chunks = retriever.expand_with_call_graph(top, hops=hops)[:10]
    prompt = build_grounded_prompt(query, context_chunks)

    # Generate
    try:
        raw = generator.generate(prompt)
    except Exception as e:
        logger.error(f"Generator error: {e}")
        raw = (
            "STATUS: INSUFFICIENT_EVIDENCE\n\n"
            "CITATIONS:\nNone\n\n"
            "ANSWER:\nThe repository does not contain sufficient evidence "
            "to answer this question."
        )

    # Parse
    parsed = parse_structured_output(raw)
    citation_valid = validate_citations(parsed["citations"], context_chunks)

    final_status = parsed["status"]
    final_answer = parsed["answer"]
    final_citations = parsed["citations"]

    # Guard: SUPPORTED but citations are hallucinated → flip to INSUFFICIENT
    if final_status == "SUPPORTED" and not citation_valid:
        final_status = "INSUFFICIENT_EVIDENCE"
        final_answer = (
            "The repository does not contain sufficient evidence "
            "to answer this question."
        )
        final_citations = []

    # Check format compliance
    format_compliant = (
        raw is not None
        and _STATUS_RE.search(normalize_llm_output(raw)) is not None
    )

    # Salvage unformatted but substantive responses
    if (
        final_status == "INSUFFICIENT_EVIDENCE"
        and raw
        and not format_compliant
        and len(raw.strip()) > 60
        and not looks_like_abstention(raw)
    ):
        inferred = infer_citations_from_overlap(raw, context_chunks)
        if inferred:
            final_status = "SUPPORTED"
            final_answer = raw.strip()
            final_citations = inferred
            citation_valid = True

    # Evidence overlap check: if SUPPORTED but answer is ungrounded, flip
    overlap_score = evidence_overlap_score(final_answer, context_chunks)
    if final_status == "SUPPORTED" and overlap_score < 0.15:
        final_status = "INSUFFICIENT_EVIDENCE"
        final_answer = (
            "The repository does not contain sufficient evidence "
            "to answer this question."
        )
        final_citations = []
        citation_valid = False

    return {
        "status": final_status,
        "answer": final_answer,
        "citations": [
            {"file": f, "start": s, "end": e} for f, s, e in final_citations
        ],
        "confidence": top_score,
        "retrieved_files": retrieved_files,
        "citation_valid": citation_valid,
        "evidence": context_chunks,
        "mode": "proposed",
        "overlap_score": round(overlap_score, 3),
    }


def answer_question_baseline(
    query: str,
    retriever,
    generator,
    k: int = 5,
) -> dict:
    """BASELINE pipeline: semantic-only retrieval → simple prompt → raw answer."""
    top = retriever.semantic_search(query, k=k)
    top_score = top[0][1] if top else -1.0
    retrieved_files = [c["file"] for c, _ in top[:10]]
    context_chunks = [c for c, _ in top[:10]]

    blocks = [
        f"# {c['file']}:{c['start_line']}-{c['end_line']} ({c['name']})\n{c['code']}"
        for c in context_chunks
    ]
    context_str = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    prompt = (
        "You are a codebase assistant. Answer the question using ONLY the "
        "code context provided below. Be specific and reference the actual "
        "code when possible.\n\n"
        f"CODE CONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"
    )

    try:
        raw = generator.generate(prompt)
    except Exception as e:
        logger.error(f"Generator error: {e}")
        raw = "Error generating answer."

    citations = [
        {"file": c["file"], "start": c["start_line"], "end": c["end_line"]}
        for c in context_chunks
    ]
    overlap = evidence_overlap_score(raw, context_chunks)

    return {
        "status": "SUPPORTED",
        "answer": raw,
        "citations": citations,
        "confidence": top_score,
        "retrieved_files": retrieved_files,
        "citation_valid": True,
        "evidence": context_chunks,
        "mode": "baseline",
        "overlap_score": round(overlap, 3),
    }
