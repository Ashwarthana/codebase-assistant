import pytest
from app.pipeline import (
    parse_structured_output,
    validate_citations,
    infer_citations_from_overlap,
    looks_like_abstention,
    answer_question_proposed,
    answer_question_baseline,
    evidence_overlap_score,
    extract_citations
)
from tests.conftest import MockGenerator


def test_parse_structured_output_supported():
    text = "STATUS: SUPPORTED\n\nCITATIONS:\napp/models.py:10-15\n\nANSWER:\nThis is the answer."
    result = parse_structured_output(text)
    assert result["status"] == "SUPPORTED"
    assert result["answer"] == "This is the answer."
    assert result["citations"] == [("app/models.py", 10, 15)]

def test_parse_structured_output_insufficient():
    text = "STATUS: INSUFFICIENT_EVIDENCE\n\nCITATIONS:\nNONE\n\nANSWER:\nI cannot answer this."
    result = parse_structured_output(text)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["answer"] == "I cannot answer this."
    assert result["citations"] == []

def test_parse_structured_output_malformed():
    text = "Here is the answer but no status line."
    result = parse_structured_output(text)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"

def test_extract_citations():
    text = "Check out app/models.py:L10-L15 and also app/views.py line 20."
    c = extract_citations(text)
    assert ("app/models.py", 10, 15) in c
    assert ("app/views.py", 20, 20) in c

def test_validate_citations_valid(mock_context_chunks):
    citations = [("app/models.py", 10, 15)]
    assert validate_citations(citations, mock_context_chunks) is True

def test_validate_citations_overlap(mock_context_chunks):
    # Chunk is 5-50. Citation 40-60 partially overlaps.
    citations = [("app/models.py", 40, 60)]
    assert validate_citations(citations, mock_context_chunks) is True

def test_validate_citations_invalid_file(mock_context_chunks):
    citations = [("app/nonexistent.py", 10, 15)]
    assert validate_citations(citations, mock_context_chunks) is False

def test_validate_citations_invalid_lines(mock_context_chunks):
    citations = [("app/models.py", 100, 150)]
    assert validate_citations(citations, mock_context_chunks) is False

def test_validate_citations_empty(mock_context_chunks):
    citations = []
    assert validate_citations(citations, mock_context_chunks) is False

def test_infer_citations_from_overlap(mock_context_chunks):
    answer = "The User model uses generate_password_hash."
    citations = infer_citations_from_overlap(answer, mock_context_chunks)
    assert len(citations) > 0
    assert citations[0][0] == "app/models.py"

def test_infer_citations_no_overlap(mock_context_chunks):
    answer = "The system is running well."
    citations = infer_citations_from_overlap(answer, mock_context_chunks)
    assert len(citations) == 0

def test_looks_like_abstention_positive():
    assert looks_like_abstention("I don't know the answer to that.") is True
    assert looks_like_abstention("The provided context does not contain information about this.") is True

def test_looks_like_abstention_negative():
    assert looks_like_abstention("The User class is located in app/models.py.") is False

def test_answer_question_proposed_supported(mock_retriever):
    generator = MockGenerator("STATUS: SUPPORTED\n\nCITATIONS:\napp/models.py:10-15\n\nANSWER:\nThe answer is check_password.")
    result = answer_question_proposed("What is the answer?", mock_retriever, generator)
    assert result["status"] == "SUPPORTED"
    assert result["answer"] == "The answer is check_password."

def test_answer_question_proposed_insufficient(mock_retriever):
    generator = MockGenerator("STATUS: INSUFFICIENT_EVIDENCE\n\nCITATIONS:\nNONE\n\nANSWER:\nI cannot answer.")
    result = answer_question_proposed("What is the unknown?", mock_retriever, generator)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"

def test_answer_question_proposed_hallucinated_citations(mock_retriever):
    # Overlap score will be 0 since answer words aren't in chunks
    generator = MockGenerator("STATUS: SUPPORTED\n\nCITATIONS:\napp/hallucinated.py:1-10\n\nANSWER:\nBlah blah nonexistent.")
    result = answer_question_proposed("Query", mock_retriever, generator)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(result.get("citations", [])) == 0

def test_answer_question_baseline(mock_retriever):
    generator = MockGenerator("check_password")
    result = answer_question_baseline("Query", mock_retriever, generator)
    assert "answer" in result
    assert result["answer"] == "check_password"

def test_evidence_overlap_score(mock_context_chunks):
    assert evidence_overlap_score("This uses check_password_hash", mock_context_chunks) > 0.0
    assert evidence_overlap_score("Nothing related", mock_context_chunks) == 0.0
