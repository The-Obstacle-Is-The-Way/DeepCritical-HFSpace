"""Unit tests for query preprocessing utilities."""

import pytest

from src.tools.query_utils import expand_synonyms, preprocess_query, strip_question_words


@pytest.mark.unit
class TestQueryPreprocessing:
    """Tests for query preprocessing."""

    def test_strip_question_words(self) -> None:
        """Test removal of question words."""
        assert strip_question_words("What drugs treat cancer") == "drugs treat cancer"
        assert strip_question_words("Which medications help diabetes") == "medications diabetes"
        assert strip_question_words("How can we cure alzheimer") == "we cure alzheimer"
        assert strip_question_words("Is metformin effective") == "metformin"

    def test_strip_preserves_medical_terms(self) -> None:
        """Test that medical terms are preserved."""
        result = strip_question_words("What is the mechanism of metformin")
        assert "metformin" in result
        assert "mechanism" in result

    def test_expand_synonyms_long_covid(self) -> None:
        """Test Long COVID synonym expansion."""
        result = expand_synonyms("long covid treatment")
        assert "PASC" in result or "post-COVID" in result

    def test_expand_synonyms_alzheimer(self) -> None:
        """Test Alzheimer's synonym expansion."""
        result = expand_synonyms("alzheimer drug")
        assert "Alzheimer" in result

    def test_expand_synonyms_preserves_unknown(self) -> None:
        """Test that unknown terms are preserved."""
        result = expand_synonyms("metformin diabetes")
        assert "metformin" in result
        assert "diabetes" in result

    def test_preprocess_query_full_pipeline(self) -> None:
        """Test complete preprocessing pipeline."""
        raw = "What medications show promise for Long COVID?"
        result = preprocess_query(raw)

        # Should not contain question words
        assert "what" not in result.lower()
        assert "show" not in result.lower()
        assert "promise" not in result.lower()

        # Should contain expanded terms
        assert "PASC" in result or "post-COVID" in result or "long covid" in result.lower()
        assert "medications" in result.lower() or "drug" in result.lower()

    def test_preprocess_query_removes_punctuation(self) -> None:
        """Test that question marks are removed."""
        result = preprocess_query("Is metformin safe?")
        assert "?" not in result

    def test_preprocess_query_handles_empty(self) -> None:
        """Test handling of empty/whitespace queries."""
        assert preprocess_query("") == ""
        assert preprocess_query("   ") == ""

    def test_preprocess_query_already_clean(self) -> None:
        """Test that clean queries pass through."""
        clean = "metformin diabetes mechanism"
        result = preprocess_query(clean)
        assert "metformin" in result
        assert "diabetes" in result
        assert "mechanism" in result
