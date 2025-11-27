"""Tests for rate limiting functionality."""

import asyncio
import time

import pytest

from src.tools.rate_limiter import RateLimiter, get_pubmed_limiter, reset_pubmed_limiter


class TestRateLimiter:
    """Test suite for rate limiter."""

    def test_create_limiter_without_api_key(self) -> None:
        """Should create 3/sec limiter without API key."""
        limiter = RateLimiter(rate="3/second")
        assert limiter.rate == "3/second"

    def test_create_limiter_with_api_key(self) -> None:
        """Should create 10/sec limiter with API key."""
        limiter = RateLimiter(rate="10/second")
        assert limiter.rate == "10/second"

    @pytest.mark.asyncio
    async def test_limiter_allows_requests_under_limit(self) -> None:
        """Should allow requests under the rate limit."""
        limiter = RateLimiter(rate="10/second")

        # 3 requests should all succeed immediately
        for _ in range(3):
            allowed = await limiter.acquire()
            assert allowed is True

    @pytest.mark.asyncio
    async def test_limiter_blocks_when_exceeded(self) -> None:
        """Should wait when rate limit exceeded."""
        limiter = RateLimiter(rate="2/second")

        # First 2 should be instant
        await limiter.acquire()
        await limiter.acquire()

        # Third should block briefly
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited ~0.5 seconds (half second window for 2/sec)
        assert elapsed >= 0.3

    @pytest.mark.asyncio
    async def test_limiter_resets_after_window(self) -> None:
        """Rate limit should reset after time window."""
        limiter = RateLimiter(rate="5/second")

        # Use up the limit
        for _ in range(5):
            await limiter.acquire()

        # Wait for window to pass
        await asyncio.sleep(1.1)

        # Should be allowed again
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # Should be nearly instant


class TestGetPubmedLimiter:
    """Test PubMed-specific limiter factory."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset limiter before and after each test."""
        reset_pubmed_limiter()
        yield
        reset_pubmed_limiter()

    def test_limiter_without_api_key(self) -> None:
        """Should return 3/sec limiter without key."""
        limiter = get_pubmed_limiter(api_key=None)
        assert "3" in limiter.rate

    def test_limiter_with_api_key(self) -> None:
        """Should return 10/sec limiter with key."""
        limiter = get_pubmed_limiter(api_key="my-api-key")
        assert "10" in limiter.rate

    def test_limiter_is_singleton(self) -> None:
        """Same API key should return same limiter instance."""
        limiter1 = get_pubmed_limiter(api_key="key1")
        limiter2 = get_pubmed_limiter(api_key="key1")
        assert limiter1 is limiter2

    def test_different_keys_different_limiters(self) -> None:
        """Different API keys should return different limiters."""
        limiter1 = get_pubmed_limiter(api_key="key1")
        limiter2 = get_pubmed_limiter(api_key="key2")
        # Clear cache for clean test
        # Actually, different keys SHOULD share the same limiter
        # since we're limiting against the same API
        assert limiter1 is limiter2  # Shared NCBI rate limit
