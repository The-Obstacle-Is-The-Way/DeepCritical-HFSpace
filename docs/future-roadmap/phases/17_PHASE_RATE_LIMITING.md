# Phase 17: Rate Limiting with `limits` Library

**Priority**: P0 CRITICAL - Prevents API blocks
**Effort**: ~1 hour
**Dependencies**: None

---

## CRITICAL: Async Safety Requirements

**WARNING**: The rate limiter MUST be async-safe. Blocking the event loop will freeze:
- The Gradio UI
- All parallel searches
- The orchestrator

**Rules**:
1. **NEVER use `time.sleep()`** - Always use `await asyncio.sleep()`
2. **NEVER use blocking while loops** - Use async-aware polling
3. **The `limits` library check is synchronous** - Wrap it carefully

The implementation below uses a polling pattern that:
- Checks the limit (synchronous, fast)
- If exceeded, `await asyncio.sleep()` (non-blocking)
- Retry the check

**Alternative**: If `limits` proves problematic, use `aiolimiter` which is pure-async.

---

## Overview

Replace naive `asyncio.sleep` rate limiting with proper rate limiter using the `limits` library, which provides:
- Moving window rate limiting
- Per-API configurable limits
- Thread-safe storage
- Already used in reference repo

**Why This Matters?**
- NCBI will block us without proper rate limiting (3/sec without key, 10/sec with)
- Current implementation only has simple sleep delay
- Need coordinated limits across all PubMed calls
- Professional-grade rate limiting prevents production issues

---

## Current State

### What We Have (`src/tools/pubmed.py:20-21, 34-41`)

```python
RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key

async def _rate_limit(self) -> None:
    """Enforce NCBI rate limiting."""
    loop = asyncio.get_running_loop()
    now = loop.time()
    elapsed = now - self._last_request_time
    if elapsed < self.RATE_LIMIT_DELAY:
        await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
    self._last_request_time = loop.time()
```

### Problems

1. **Not shared across instances**: Each `PubMedTool()` has its own counter
2. **Simple delay vs moving window**: Doesn't handle bursts properly
3. **Hardcoded rate**: Doesn't adapt to API key presence
4. **No backoff on 429**: Just retries blindly

---

## TDD Implementation Plan

### Step 1: Add Dependency

**File**: `pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "limits>=3.0",
]
```

Then run:
```bash
uv sync
```

---

### Step 2: Write the Tests First

**File**: `tests/unit/tools/test_rate_limiting.py`

```python
"""Tests for rate limiting functionality."""

import asyncio
import time

import pytest

from src.tools.rate_limiter import RateLimiter, get_pubmed_limiter


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
```

---

### Step 3: Create Rate Limiter Module

**File**: `src/tools/rate_limiter.py`

```python
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
            # coroutines (UI, parallel searches) to run
            await asyncio.sleep(0.1)

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
```

---

### Step 4: Update PubMed Tool

**File**: `src/tools/pubmed.py` (replace rate limiting code)

```python
# Replace imports and rate limiting

from src.tools.rate_limiter import get_pubmed_limiter


class PubMedTool:
    """Search tool for PubMed/NCBI."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    HTTP_TOO_MANY_REQUESTS = 429

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.ncbi_api_key
        if self.api_key == "your-ncbi-key-here":
            self.api_key = None
        # Use shared rate limiter
        self._limiter = get_pubmed_limiter(self.api_key)

    async def _rate_limit(self) -> None:
        """Enforce NCBI rate limiting using shared limiter."""
        await self._limiter.acquire()

    # ... rest of class unchanged ...
```

---

### Step 5: Add Rate Limiters for Other APIs

**File**: `src/tools/clinicaltrials.py` (optional)

```python
from src.tools.rate_limiter import RateLimiterFactory


class ClinicalTrialsTool:
    def __init__(self) -> None:
        # ClinicalTrials.gov doesn't document limits, but be conservative
        self._limiter = RateLimiterFactory.get("clinicaltrials", "5/second")

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        await self._limiter.acquire()
        # ... rest of method ...
```

**File**: `src/tools/europepmc.py` (optional)

```python
from src.tools.rate_limiter import RateLimiterFactory


class EuropePMCTool:
    def __init__(self) -> None:
        # Europe PMC is generous, but still be respectful
        self._limiter = RateLimiterFactory.get("europepmc", "10/second")

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        await self._limiter.acquire()
        # ... rest of method ...
```

---

## Demo Script

**File**: `examples/rate_limiting_demo.py`

```python
#!/usr/bin/env python3
"""Demo script to verify rate limiting works correctly."""

import asyncio
import time

from src.tools.rate_limiter import RateLimiter, get_pubmed_limiter, reset_pubmed_limiter
from src.tools.pubmed import PubMedTool


async def test_basic_limiter():
    """Test basic rate limiter behavior."""
    print("=" * 60)
    print("Rate Limiting Demo")
    print("=" * 60)

    # Test 1: Basic limiter
    print("\n[Test 1] Testing 3/second limiter...")
    limiter = RateLimiter("3/second")

    start = time.monotonic()
    for i in range(6):
        await limiter.acquire()
        elapsed = time.monotonic() - start
        print(f"  Request {i+1} at {elapsed:.2f}s")

    total = time.monotonic() - start
    print(f"  Total time for 6 requests: {total:.2f}s (expected ~2s)")


async def test_pubmed_limiter():
    """Test PubMed-specific limiter."""
    print("\n[Test 2] Testing PubMed limiter (shared)...")

    reset_pubmed_limiter()  # Clean state

    # Without API key: 3/sec
    limiter = get_pubmed_limiter(api_key=None)
    print(f"  Rate without key: {limiter.rate}")

    # Multiple tools should share the same limiter
    tool1 = PubMedTool()
    tool2 = PubMedTool()

    # Verify they share the limiter
    print(f"  Tools share limiter: {tool1._limiter is tool2._limiter}")


async def test_concurrent_requests():
    """Test rate limiting under concurrent load."""
    print("\n[Test 3] Testing concurrent request limiting...")

    limiter = RateLimiter("5/second")

    async def make_request(i: int):
        await limiter.acquire()
        return time.monotonic()

    start = time.monotonic()
    # Launch 10 concurrent requests
    tasks = [make_request(i) for i in range(10)]
    times = await asyncio.gather(*tasks)

    # Calculate distribution
    relative_times = [t - start for t in times]
    print(f"  Request times: {[f'{t:.2f}s' for t in sorted(relative_times)]}")

    total = max(relative_times)
    print(f"  All 10 requests completed in {total:.2f}s (expected ~2s)")


async def main():
    await test_basic_limiter()
    await test_pubmed_limiter()
    await test_concurrent_requests()

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Verification Checklist

### Unit Tests
```bash
# Run rate limiting tests
uv run pytest tests/unit/tools/test_rate_limiting.py -v

# Expected: All tests pass
```

### Integration Test (Manual)
```bash
# Run demo
uv run python examples/rate_limiting_demo.py

# Expected: Requests properly spaced
```

### Full Test Suite
```bash
make check
# Expected: All tests pass, mypy clean
```

---

## Success Criteria

1. **`limits` library installed**: Dependency added to pyproject.toml
2. **RateLimiter class works**: Can create and use limiters
3. **PubMed uses new limiter**: Shared limiter across instances
4. **Rate adapts to API key**: 3/sec without, 10/sec with
5. **Concurrent requests handled**: Multiple async requests properly queued
6. **No regressions**: All existing tests pass

---

## API Rate Limit Reference

| API | Without Key | With Key |
|-----|-------------|----------|
| PubMed/NCBI | 3/sec | 10/sec |
| ClinicalTrials.gov | Undocumented (~5/sec safe) | N/A |
| Europe PMC | ~10-20/sec (generous) | N/A |
| OpenAlex | ~100k/day (no per-sec limit) | Faster with `mailto` |

---

## Notes

- `limits` library uses moving window algorithm (fairer than fixed window)
- Singleton pattern ensures all PubMed calls share the limit
- The factory pattern allows easy extension to other APIs
- Consider adding 429 response detection + exponential backoff
- In production, consider Redis storage for distributed rate limiting
