"""Tests for chunking argument validation."""

import pytest

from data_designer_retrieval_sdg.chunking import chunks_to_sections_structured


def test_chunks_to_sections_structured_rejects_non_positive_num_sections():
    """Zero or negative num_sections must raise ValueError."""
    chunks = [{"text": "one"}, {"text": "two"}]
    for invalid in (0, -1, -5):
        with pytest.raises(ValueError, match="num_sections must be positive"):
            chunks_to_sections_structured(chunks, num_sections=invalid)


def test_chunks_to_sections_structured_accepts_positive_num_sections():
    """Positive num_sections continues to work normally."""
    chunks = [{"text": "one"}, {"text": "two"}]
    result = chunks_to_sections_structured(chunks, num_sections=1)
    assert isinstance(result, list)
    assert len(result) == 1

    result = chunks_to_sections_structured(chunks, num_sections=2)
