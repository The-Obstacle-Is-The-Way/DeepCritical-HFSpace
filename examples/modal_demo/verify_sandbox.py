"""Verification script to prove code is running in Modal sandboxes, not locally.

This script runs tests that would behave differently in a sandbox vs local execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.code_execution import SANDBOX_LIBRARIES, get_code_executor


def test_1_hostname_check():
    """Test 1: Check hostname - should be different in sandbox."""
    print("\n" + "=" * 60)
    print("TEST 1: Hostname Check")
    print("=" * 60)

    executor = get_code_executor()

    # Get local hostname
    import socket

    local_hostname = socket.gethostname()
    print(f"Local hostname: {local_hostname}")

    # Get sandbox hostname
    code = """
import socket
hostname = socket.gethostname()
print(f"Sandbox hostname: {hostname}")
"""

    result = executor.execute(code)
    print(f"\n{result['stdout']}")

    if local_hostname in result["stdout"]:
        print("⚠️  WARNING: Hostnames match - might be running locally!")
        return False
    else:
        print("✅ SUCCESS: Different hostnames - running in sandbox!")
        return True


def test_2_file_system_isolation():
    """Test 2: Try to access local files - should fail in sandbox."""
    print("\n" + "=" * 60)
    print("TEST 2: File System Isolation")
    print("=" * 60)

    executor = get_code_executor()

    # Try to read our own source file
    local_file = Path(__file__).resolve()
    print(f"Local file exists: {local_file}")
    print(f"Can read locally: {local_file.exists()}")

    # Try to access it from sandbox (use POSIX path for Windows compatibility)
    code = f"""
from pathlib import Path
file_path = Path("{local_file.as_posix()}")
exists = file_path.exists()
print(f"File exists in sandbox: {{exists}}")
if exists:
    print("⚠️  Can access local filesystem!")
else:
    print("✅ Filesystem is isolated!")
"""

    result = executor.execute(code)
    print(f"\n{result['stdout']}")

    if "File exists in sandbox: True" in result["stdout"]:
        print("\n⚠️  WARNING: Can access local files - not properly sandboxed!")
        return False
    else:
        print("\n✅ SUCCESS: Cannot access local files - properly sandboxed!")
        return True


def test_3_process_information():
    """Test 3: Check process and container info."""
    print("\n" + "=" * 60)
    print("TEST 3: Process Information")
    print("=" * 60)

    executor = get_code_executor()

    code = """
import os
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"Process ID: {os.getpid()}")
print(f"User: {os.getenv('USER', 'unknown')}")
print(f"Home: {os.getenv('HOME', 'unknown')}")
print(f"Working directory: {os.getcwd()}")

# Check if running in container
in_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
print(f"In container: {in_container}")
"""

    result = executor.execute(code)
    print(f"\n{result['stdout']}")

    if "In container: True" in result["stdout"]:
        print("\n✅ SUCCESS: Running in containerized environment!")
        return True
    else:
        print("\n⚠️  WARNING: Not detecting container environment")
        return False


def test_4_library_versions():
    """Test 4: Check if scientific libraries match Modal image specs."""
    print("\n" + "=" * 60)
    print("TEST 4: Library Versions (Should match Modal image)")
    print("=" * 60)

    executor = get_code_executor()

    code = """
import pandas as pd
import numpy as np
import scipy
import matplotlib
import sklearn
import statsmodels

print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"statsmodels: {statsmodels.__version__}")
"""

    result = executor.execute(code)
    print(f"\n{result['stdout']}")

    # Check if versions match what we specified in code_execution.py
    expected_versions = {
        f"pandas: {SANDBOX_LIBRARIES['pandas']}": True,
        f"numpy: {SANDBOX_LIBRARIES['numpy']}": True,
        f"scipy: {SANDBOX_LIBRARIES['scipy']}": True,
    }

    matches = 0
    for expected in expected_versions:
        if expected in result["stdout"]:
            matches += 1
            print(f"✅ {expected}")

    if matches >= 2:
        print(f"\n✅ SUCCESS: Library versions match Modal image spec ({matches}/3)")
        return True
    else:
        print(f"\n⚠️  WARNING: Library versions don't match ({matches}/3)")
        return False


def test_5_destructive_operations():
    """Test 5: Try destructive operations that would be dangerous locally."""
    print("\n" + "=" * 60)
    print("TEST 5: Destructive Operations (Safe in sandbox)")
    print("=" * 60)

    executor = get_code_executor()

    code = """
import os
import tempfile

# Try to write to /tmp (should work)
tmp_file = "/tmp/test_modal_sandbox.txt"
try:
    with open(tmp_file, 'w') as f:
        f.write("Test write to /tmp")
    print(f"✅ Can write to /tmp: {tmp_file}")
    os.remove(tmp_file)
    print("✅ Can delete from /tmp")
except Exception as e:
    print(f"❌ Error with /tmp: {e}")

# Try to write to /root (might fail due to permissions)
try:
    test_file = "/root/test.txt"
    with open(test_file, 'w') as f:
        f.write("Test")
    print(f"✅ Can write to /root (running as root in container)")
    os.remove(test_file)
except Exception as e:
    print(f"⚠️  Cannot write to /root: {e}")

# Check what user we're running as
print(f"Running as UID: {os.getuid()}")
print(f"Running as GID: {os.getgid()}")
"""

    result = executor.execute(code)
    print(f"\n{result['stdout']}")

    if "Can write to /tmp" in result["stdout"]:
        print("\n✅ SUCCESS: Sandbox has expected filesystem permissions!")
        return True
    else:
        print("\n⚠️  WARNING: Unexpected filesystem behavior")
        return False


def test_6_network_isolation():
    """Test 6: Check network access (should be allowed by default in our config)."""
    print("\n" + "=" * 60)
    print("TEST 6: Network Access Check")
    print("=" * 60)

    executor = get_code_executor()

    code = """
import socket

# Try to resolve a hostname
try:
    ip = socket.gethostbyname('google.com')
    print(f"✅ Can resolve DNS: google.com -> {ip}")
    print("(Network is enabled - can be disabled for security)")
except Exception as e:
    print(f"❌ Cannot resolve DNS: {e}")
    print("(Network is blocked)")
"""

    result = executor.execute(code)
    print(f"\n{result['stdout']}")

    return True  # Either result is valid


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "MODAL SANDBOX VERIFICATION")
    print("=" * 70)
    print("\nThese tests verify code is running in Modal sandboxes, not locally.")
    print("=" * 70)

    tests = [
        ("Hostname Isolation", test_1_hostname_check),
        ("Filesystem Isolation", test_2_file_system_isolation),
        ("Container Detection", test_3_process_information),
        ("Library Versions", test_4_library_versions),
        ("Destructive Operations", test_5_destructive_operations),
        ("Network Access", test_6_network_isolation),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("=" * 70)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed >= 4:
        print("\n🎉 Modal sandboxing is working correctly!")
    elif passed >= 2:
        print("\n⚠️  Some tests failed - review output above")
    else:
        print("\n❌ Modal sandboxing may not be working - check configuration")

    print("=" * 70)


if __name__ == "__main__":
    main()
