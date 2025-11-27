"""Rate limiting utilities using the limits library."""

import asyncio
from typing import ClassVar

from limits import RateLimitItem, parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter


class RateLimiter:
    """
    Async-compatible rate limiter using limits library.

    Uses moving window algorithm for smooth rate limiting.
    """

    def __init__(self, rate: str) -> None:
        """
        Initialize rate limiter.

        Args:
            rate: Rate string like "3/second" or "10/second"
        """
        self.rate = rate
        self._storage = MemoryStorage()
        self._limiter = MovingWindowRateLimiter(self._storage)
        self._rate_limit: RateLimitItem = parse(rate)
        self._identity = "default"  # Single identity for shared limiting

    async def acquire(self, wait: bool = True) -> bool:
        """
        Acquire permission to make a request.

        ASYNC-SAFE: Uses asyncio.sleep(), never time.sleep().
        The polling pattern allows other coroutines to run while waiting.

        Args:
            wait: If True, wait until allowed. If False, return immediately.

        Returns:
            True if allowed, False if not (only when wait=False)
        """
        while True:
            # Check if we can proceed (synchronous, fast - ~microseconds)
            if self._limiter.hit(self._rate_limit, self._identity):
                return True

            if not wait:
                return False

            # CRITICAL: Use asyncio.sleep(), NOT time.sleep()
            # This yields control to the event loop, allowing other
            # coroutines (UI, parallel searches) to run.
            # Using 0.01s for fine-grained responsiveness.
            await asyncio.sleep(0.01)

    def reset(self) -> None:
        """Reset the rate limiter (for testing)."""
        self._storage.reset()


# Singleton limiter for PubMed/NCBI
_pubmed_limiter: RateLimiter | None = None


def get_pubmed_limiter(api_key: str | None = None) -> RateLimiter:
    """
    Get the shared PubMed rate limiter.

    Rate depends on whether API key is provided:
    - Without key: 3 requests/second
    - With key: 10 requests/second

    Args:
        api_key: NCBI API key (optional)

    Returns:
        Shared RateLimiter instance
    """
    global _pubmed_limiter

    if _pubmed_limiter is None:
        rate = "10/second" if api_key else "3/second"
        _pubmed_limiter = RateLimiter(rate)

    return _pubmed_limiter


def reset_pubmed_limiter() -> None:
    """Reset the PubMed limiter (for testing)."""
    global _pubmed_limiter
    _pubmed_limiter = None


# Factory for other APIs
class RateLimiterFactory:
    """Factory for creating/getting rate limiters for different APIs."""

    _limiters: ClassVar[dict[str, RateLimiter]] = {}

    @classmethod
    def get(cls, api_name: str, rate: str) -> RateLimiter:
        """
        Get or create a rate limiter for an API.

        Args:
            api_name: Unique identifier for the API
            rate: Rate limit string (e.g., "10/second")

        Returns:
            RateLimiter instance (shared for same api_name)
        """
        if api_name not in cls._limiters:
            cls._limiters[api_name] = RateLimiter(rate)
        return cls._limiters[api_name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all limiters (for testing)."""
        cls._limiters.clear()
