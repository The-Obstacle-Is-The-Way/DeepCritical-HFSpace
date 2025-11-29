"""Unit tests for HFInferenceJudgeHandler Circuit Breaker."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent_factory.judges import HFInferenceJudgeHandler
from src.utils.models import Citation, Evidence


@pytest.mark.unit
class TestJudgeCircuitBreaker:
    """Tests specifically for the circuit breaker logic."""

    @pytest.fixture
    def handler(self):
        """Create a handler with mocked dependencies."""
        with patch("src.agent_factory.judges.InferenceClient"):
            return HFInferenceJudgeHandler()

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers_after_max_failures(self, handler):
        """Verify it switches to 'synthesize' after 3 consecutive failures."""

        # Mock _call_with_retry to always fail
        with patch.object(handler, "_call_with_retry", side_effect=Exception("Model failed")):
            evidence = [
                Evidence(
                    content="test",
                    citation=Citation(source="pubmed", title="t", url="u", date="2025"),
                )
            ]

            # Call 1: Fails
            result1 = await handler.assess("test", evidence)
            assert result1.recommendation == "continue"
            assert handler.consecutive_failures == 1

            # Call 2: Fails
            result2 = await handler.assess("test", evidence)
            assert result2.recommendation == "continue"
            assert handler.consecutive_failures == 2

            # Call 3: Fails
            result3 = await handler.assess("test", evidence)
            assert result3.recommendation == "continue"
            assert handler.consecutive_failures == 3

            # Call 4: Circuit Breaker SHOULD trigger
            # Because failures >= MAX (3)
            result4 = await handler.assess("test", evidence)

            assert result4.recommendation == "synthesize"
            assert result4.sufficient is True
            # The message contains "failed 3 times" or "Unavailable"
            reasoning_lower = result4.reasoning.lower()
            assert "failed" in reasoning_lower or "unavailable" in reasoning_lower

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, handler):
        """Verify failures reset if a call succeeds."""

        evidence = [
            Evidence(
                content="t",
                citation=Citation(source="pubmed", title="t", url="u", date="d"),
            )
        ]

        # 1. Fail once
        with patch.object(handler, "_call_with_retry", side_effect=Exception("Fail")):
            await handler.assess("test", evidence)
            assert handler.consecutive_failures == 1

        # 2. Succeed
        valid_assessment = MagicMock(recommendation="continue", sufficient=False)
        with patch.object(handler, "_call_with_retry", return_value=valid_assessment):
            await handler.assess("test", evidence)
            assert handler.consecutive_failures == 0  # Should reset

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_new_question(self, handler):
        """Verify failures reset if question changes."""

        evidence = []

        # 1. Fail on Question A
        with patch.object(handler, "_call_with_retry", side_effect=Exception("Fail")):
            await handler.assess("Question A", evidence)
            assert handler.consecutive_failures == 1

            # 2. Fail on Question B (Should reset first, then increment to 1)
            await handler.assess("Question B", evidence)
            # Reset happens at start of assess:
            # if "Question B" != "Question A" -> failures = 0
            # Then it tries and fails -> failures = 1
            assert handler.consecutive_failures == 1
