"""Unit tests for HFInferenceJudgeHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_factory.judges import HFInferenceJudgeHandler
from src.utils.models import Citation, Evidence


@pytest.mark.unit
class TestHFInferenceJudgeHandler:
    """Tests for HFInferenceJudgeHandler."""

    @pytest.fixture
    def mock_client(self):
        """Mock HuggingFace InferenceClient."""
        with patch("src.agent_factory.judges.InferenceClient") as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance
            yield client_instance

    @pytest.fixture
    def handler(self, mock_client):
        """Create a handler instance with mocked client."""
        return HFInferenceJudgeHandler()

    @pytest.mark.asyncio
    async def test_assess_success(self, handler, mock_client):
        """Test successful assessment with primary model."""
        import json

        # Construct valid JSON payload
        data = {
            "details": {
                "mechanism_score": 8,
                "mechanism_reasoning": "Good mechanism",
                "clinical_evidence_score": 7,
                "clinical_reasoning": "Good clinical",
                "drug_candidates": ["Drug A"],
                "key_findings": ["Finding 1"],
            },
            "sufficient": True,
            "confidence": 0.85,
            "recommendation": "synthesize",
            "next_search_queries": [],
            "reasoning": (
                "Sufficient evidence provided to support the hypothesis with high confidence."
            ),
        }

        # Mock chat_completion response structure
        mock_message = MagicMock()
        mock_message.content = f"""Here is the analysis:
```json
{json.dumps(data)}
```"""
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Setup async mock for run_in_executor
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

            evidence = [
                Evidence(
                    content="test", citation=Citation(source="pubmed", title="t", url="u", date="d")
                )
            ]
            result = await handler.assess("test question", evidence)

            assert result.sufficient is True
            assert result.confidence == 0.85
            assert result.details.drug_candidates == ["Drug A"]

    @pytest.mark.asyncio
    async def test_assess_fallback_logic(self, handler, mock_client):
        """Test fallback to secondary model when primary fails."""

        # Setup async mock to fail first, then succeed
        with patch("asyncio.get_running_loop"):
            # We need to mock the _call_with_retry method directly to test the loop in assess
            # but _call_with_retry is decorated with tenacity,
            # which makes it harder to mock partial failures easily
            # without triggering the tenacity retry loop first.
            # Instead, let's mock run_in_executor to raise exception on first call

            # This is tricky because assess loops over models,
            # and for each model _call_with_retry retries.
            # We want to simulate: Model 1 fails (retries exhausted) -> Model 2 succeeds.

            # Let's patch _call_with_retry to avoid waiting for real retries
            side_effect = [
                Exception("Model 1 failed"),
                Exception("Model 2 failed"),
                Exception("Model 3 failed"),
            ]
            with patch.object(handler, "_call_with_retry", side_effect=side_effect) as mock_call:
                evidence = []
                result = await handler.assess("test", evidence)

                # Should have tried all 3 fallback models
                assert mock_call.call_count == 3
                # Fallback assessment should indicate failure
                assert result.sufficient is False
                assert "failed" in result.reasoning.lower() or "error" in result.reasoning.lower()

    def test_extract_json_robustness(self, handler):
        """Test JSON extraction with various inputs."""

        # 1. Clean JSON
        assert handler._extract_json('{"a": 1}') == {"a": 1}

        # 2. Markdown block
        assert handler._extract_json('```json\n{"a": 1}\n```') == {"a": 1}

        # 3. Text preamble/postamble
        text = """
        Sure, here is the JSON:
        {
            "a": 1,
            "b": {
                "c": 2
            }
        }
        Hope that helps!
        """
        assert handler._extract_json(text) == {"a": 1, "b": {"c": 2}}

        # 4. Nested braces
        nested = '{"a": {"b": "}"}}'
        assert handler._extract_json(nested) == {"a": {"b": "}"}}

        # 5. Invalid JSON
        assert handler._extract_json("Not JSON") is None
        assert handler._extract_json("{Incomplete") is None
