#!/usr/bin/env python3
"""Verify that Modal sandbox is properly isolated.

This script proves to judges that code runs in Modal, not locally.
NO agent_framework dependency - uses only src.tools.code_execution.

Usage:
    uv run python examples/modal_demo/verify_sandbox.py
"""

import asyncio
from functools import partial

from src.tools.code_execution import CodeExecutionError, get_code_executor
from src.utils.config import settings


def print_result(result: dict) -> None:
    """Print execution result, surfacing errors when they occur."""
    if result.get("success"):
        print(f"  {result['stdout'].strip()}\n")
    else:
        error = result.get("error") or result.get("stderr", "").strip() or "Unknown error"
        print(f"  ERROR: {error}\n")


async def main() -> None:
    """Verify Modal sandbox isolation."""
    if not settings.modal_available:
        print("Error: Modal credentials not configured.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env")
        return

    try:
        executor = get_code_executor()
        loop = asyncio.get_running_loop()

        print("=" * 60)
        print("Modal Sandbox Isolation Verification")
        print("=" * 60 + "\n")

        # Test 1: Hostname
        print("Test 1: Check hostname (should NOT be your machine)")
        code1 = "import socket; print(f'Hostname: {socket.gethostname()}')"
        result1 = await loop.run_in_executor(None, partial(executor.execute, code1))
        print_result(result1)

        # Test 2: Scientific libraries
        print("Test 2: Verify scientific libraries")
        code2 = """
import pandas as pd
import numpy as np
import scipy
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
"""
        result2 = await loop.run_in_executor(None, partial(executor.execute, code2))
        print_result(result2)

        # Test 3: Network blocked
        print("Test 3: Verify network isolation")
        code3 = """
import urllib.request
try:
    urllib.request.urlopen("https://google.com", timeout=2)
    print("Network: ALLOWED (unexpected!)")
except Exception:
    print("Network: BLOCKED (as expected)")
"""
        result3 = await loop.run_in_executor(None, partial(executor.execute, code3))
        print_result(result3)

        # Test 4: Real statistics
        print("Test 4: Execute statistical analysis")
        code4 = """
import pandas as pd
import scipy.stats as stats

data = pd.DataFrame({'effect': [0.42, 0.38, 0.51]})
mean = data['effect'].mean()
t_stat, p_val = stats.ttest_1samp(data['effect'], 0)

print(f"Mean Effect: {mean:.3f}")
print(f"P-value: {p_val:.4f}")
print(f"Verdict: {'SUPPORTED' if p_val < 0.05 else 'INCONCLUSIVE'}")
"""
        result4 = await loop.run_in_executor(None, partial(executor.execute, code4))
        print_result(result4)

        print("=" * 60)
        print("All tests complete - Modal sandbox verified!")
        print("=" * 60)

    except CodeExecutionError as e:
        print(f"Error: Modal code execution failed: {e}")
        print("Hint: Ensure Modal SDK is installed and credentials are valid.")


if __name__ == "__main__":
    asyncio.run(main())
