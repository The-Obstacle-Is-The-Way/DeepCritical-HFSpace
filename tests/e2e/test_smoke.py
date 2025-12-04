"""
Smoke tests for regression prevention.

These tests run against REAL APIs and verify end-to-end functionality.
They are slow (2-5 minutes) and should NOT run in CI.

Usage:
    make smoke-free   # Test Free Tier (HuggingFace)
    make smoke-paid   # Test Paid Tier (OpenAI BYOK)
"""

import os

import pytest

from src.orchestrators.advanced import AdvancedOrchestrator


@pytest.mark.e2e
@pytest.mark.timeout(600)  # 10 minute timeout for Free Tier
async def test_free_tier_synthesis():
    """Verify Free Tier produces actual synthesis (not just 'Research complete.')"""
    # Use a simple query that is likely to yield results quickly
    orch = AdvancedOrchestrator(max_rounds=2)

    events = []
    print("\nRunning Free Tier Smoke Test...")
    async for event in orch.run("What is libido?"):
        if event.type == "complete":
            events.append(event)
            print(f"Received complete event: {event.message[:50]}...")

    # MUST have a complete event
    assert len(events) >= 1, "No complete event received"

    # Complete event MUST have substantive content (not just signal)
    final = events[-1]

    # P2 Bug Regression Check: Ensure content isn't just "Research complete."
    assert len(final.message) > 100, f"Synthesis too short: {len(final.message)} chars"

    # P1 Bug Regression Check: Ensure we got actual text
    assert "Research complete." not in final.message or len(final.message) > 50, (
        "Got empty synthesis signal instead of actual report"
    )


@pytest.mark.e2e
@pytest.mark.timeout(300)  # 5 minute timeout for Paid Tier
async def test_paid_tier_synthesis():
    """Verify Paid Tier (BYOK) produces synthesis."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    orch = AdvancedOrchestrator(max_rounds=2, api_key=api_key)

    events = []
    print("\nRunning Paid Tier Smoke Test...")
    async for event in orch.run("What is libido?"):
        if event.type == "complete":
            events.append(event)

    assert len(events) >= 1, "No complete event received"
    assert len(events[-1].message) > 100, "Synthesis too short"
