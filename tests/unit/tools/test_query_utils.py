"""Unit tests for query preprocessing utilities."""

import pytest

from src.tools.query_utils import expand_synonyms, preprocess_query, strip_question_words


@pytest.mark.unit
class TestQueryPreprocessing:
    """Tests for query preprocessing."""

    def test_strip_question_words(self) -> None:
        """Test removal of question words."""
        assert strip_question_words("What drugs treat HSDD") == "drugs treat hsdd"
        assert strip_question_words("Which medications help low libido") == "medications low libido"
        assert strip_question_words("How can we treat ED") == "we treat ed"
        assert strip_question_words("Is sildenafil effective") == "sildenafil"

    def test_strip_preserves_medical_terms(self) -> None:
        """Test that medical terms are preserved."""
        result = strip_question_words("What is the mechanism of sildenafil")
        assert "sildenafil" in result
        assert "mechanism" in result

    def test_expand_synonyms_low_libido(self) -> None:
        """Test Low Libido synonym expansion."""
        result = expand_synonyms("low libido treatment")
        assert "HSDD" in result or "hypoactive sexual desire" in result

    def test_expand_synonyms_ed(self) -> None:
        """Test ED synonym expansion."""
        result = expand_synonyms("erectile dysfunction drug")
        assert "impotence" in result

    def test_expand_synonyms_preserves_unknown(self) -> None:
        """Test that unknown terms are preserved."""
        result = expand_synonyms("sildenafil unknowncondition")
        assert "sildenafil" in result
        assert "unknowncondition" in result

    def test_preprocess_query_full_pipeline(self) -> None:
        """Test complete preprocessing pipeline."""
        raw = "What medications show promise for Low Libido?"
        result = preprocess_query(raw)

        # Should not contain question words
        assert "what" not in result.lower()
        assert "show" not in result.lower()
        assert "promise" not in result.lower()

        # Should contain expanded terms
        assert "HSDD" in result or "hypoactive" in result or "low libido" in result.lower()
        assert "medications" in result.lower() or "drug" in result.lower()

    def test_preprocess_query_removes_punctuation(self) -> None:
        """Test that question marks are removed."""
        result = preprocess_query("Is sildenafil safe?")
        assert "?" not in result

    def test_preprocess_query_handles_empty(self) -> None:
        """Test handling of empty/whitespace queries."""
        assert preprocess_query("") == ""
        assert preprocess_query("   ") == ""

    def test_preprocess_query_already_clean(self) -> None:
        """Test that clean queries pass through."""
        clean = "sildenafil ed mechanism"
        result = preprocess_query(clean)
        assert "sildenafil" in result
        assert "ed" in result
        assert "mechanism" in result
