#!/usr/bin/env python3
"""Demo script to verify rate limiting works correctly."""

import asyncio
import time

from src.tools.pubmed import PubMedTool
from src.tools.rate_limiter import RateLimiter, get_pubmed_limiter, reset_pubmed_limiter


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
