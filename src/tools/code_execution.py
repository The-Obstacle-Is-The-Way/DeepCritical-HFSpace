"""Modal-based secure code execution tool for statistical analysis.

This module provides sandboxed Python code execution using Modal's serverless infrastructure.
It's designed for running LLM-generated statistical analysis code safely.
"""

import os
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Shared library versions for Modal sandbox - used by both executor and LLM prompts
# Keep these in sync to avoid version mismatch between generated code and execution
SANDBOX_LIBRARIES: dict[str, str] = {
    "pandas": "2.2.0",
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "matplotlib": "3.8.2",
    "scikit-learn": "1.4.0",
    "statsmodels": "0.14.1",
}


def get_sandbox_library_list() -> list[str]:
    """Get list of library==version strings for Modal image."""
    return [f"{lib}=={ver}" for lib, ver in SANDBOX_LIBRARIES.items()]


def get_sandbox_library_prompt() -> str:
    """Get formatted library versions for LLM prompts."""
    return "\n".join(f"- {lib}=={ver}" for lib, ver in SANDBOX_LIBRARIES.items())


class CodeExecutionError(Exception):
    """Raised when code execution fails."""

    pass


class ModalCodeExecutor:
    """Execute Python code securely using Modal sandboxes.

    This class provides a safe environment for executing LLM-generated code,
    particularly for scientific computing and statistical analysis tasks.

    Features:
    - Sandboxed execution (isolated from host system)
    - Pre-installed scientific libraries (numpy, scipy, pandas, matplotlib)
    - Network isolation for security
    - Timeout protection
    - Stdout/stderr capture

    Example:
        >>> executor = ModalCodeExecutor()
        >>> result = executor.execute('''
        ... import pandas as pd
        ... df = pd.DataFrame({'a': [1, 2, 3]})
        ... result = df['a'].sum()
        ... ''')
        >>> print(result['stdout'])
        6
    """

    def __init__(self) -> None:
        """Initialize Modal code executor.

        Note:
            Logs a warning if Modal credentials are not configured.
            Execution will fail at runtime without valid credentials.
        """
        # Check for Modal credentials
        self.modal_token_id = os.getenv("MODAL_TOKEN_ID")
        self.modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")

        if not self.modal_token_id or not self.modal_token_secret:
            logger.warning(
                "Modal credentials not found. Code execution will fail unless modal setup is run."
            )

    def execute(self, code: str, timeout: int = 60, allow_network: bool = False) -> dict[str, Any]:
        """Execute Python code in a Modal sandbox.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds (default: 60)
            allow_network: Whether to allow network access (default: False for security)

        Returns:
            Dictionary containing:
                - stdout: Standard output from code execution
                - stderr: Standard error from code execution
                - success: Boolean indicating if execution succeeded
                - error: Error message if execution failed

        Raises:
            CodeExecutionError: If execution fails or times out
        """
        try:
            import modal
        except ImportError as e:
            raise CodeExecutionError(
                "Modal SDK not installed. Run: uv sync or pip install modal>=0.63.0"
            ) from e

        logger.info("executing_code", code_length=len(code), timeout=timeout)

        try:
            # Create or lookup Modal app
            app = modal.App.lookup("deepboner-code-execution", create_if_missing=True)

            # Define scientific computing image with common libraries
            scientific_image = modal.Image.debian_slim(python_version="3.11").pip_install(
                *get_sandbox_library_list()
            )

            # Create sandbox with security restrictions
            sandbox = modal.Sandbox.create(
                app=app,
                image=scientific_image,
                timeout=timeout,
                block_network=not allow_network,  # Wire the network control
            )

            try:
                # Execute the code
                # Wrap code to capture result
                wrapped_code = f"""
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

stdout_io = io.StringIO()
stderr_io = io.StringIO()

try:
    with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
        {self._indent_code(code, 8)}
    print("__EXECUTION_SUCCESS__")
except Exception as e:
    print(f"__EXECUTION_ERROR__: {{type(e).__name__}}: {{e}}", file=sys.stderr)

print("__STDOUT_START__")
print(stdout_io.getvalue())
print("__STDOUT_END__")
print("__STDERR_START__")
print(stderr_io.getvalue(), file=sys.stderr)
print("__STDERR_END__", file=sys.stderr)
"""

                # Run the wrapped code
                process = sandbox.exec("python", "-c", wrapped_code, timeout=timeout)

                # Read output
                stdout_raw = process.stdout.read()
                stderr_raw = process.stderr.read()
            finally:
                # Always clean up sandbox to prevent resource leaks
                sandbox.terminate()

            # Parse output
            success = "__EXECUTION_SUCCESS__" in stdout_raw

            # Extract actual stdout/stderr
            stdout = self._extract_output(stdout_raw, "__STDOUT_START__", "__STDOUT_END__")
            stderr = self._extract_output(stderr_raw, "__STDERR_START__", "__STDERR_END__")

            result = {
                "stdout": stdout,
                "stderr": stderr,
                "success": success,
                "error": stderr if not success else None,
            }

            logger.info(
                "code_execution_completed",
                success=success,
                stdout_length=len(stdout),
                stderr_length=len(stderr),
            )

            return result

        except Exception as e:
            logger.error("code_execution_failed", error=str(e), error_type=type(e).__name__)
            raise CodeExecutionError(f"Code execution failed: {e}") from e

    def execute_with_return(self, code: str, timeout: int = 60) -> Any:
        """Execute code and return the value of the 'result' variable.

        Convenience method that executes code and extracts a return value.
        The code should assign its final result to a variable named 'result'.

        Args:
            code: Python code to execute (must set 'result' variable)
            timeout: Maximum execution time in seconds

        Returns:
            The value of the 'result' variable from the executed code

        Example:
            >>> executor.execute_with_return("result = 2 + 2")
            4
        """
        # Modify code to print result as JSON
        wrapped = f"""
import json
{code}
print(json.dumps({{"__RESULT__": result}}))
"""

        execution_result = self.execute(wrapped, timeout=timeout)

        if not execution_result["success"]:
            raise CodeExecutionError(f"Execution failed: {execution_result['error']}")

        # Parse result from stdout
        import json

        try:
            output = execution_result["stdout"].strip()
            if "__RESULT__" in output:
                # Extract JSON line
                for line in output.split("\n"):
                    if "__RESULT__" in line:
                        data = json.loads(line)
                        return data["__RESULT__"]
            raise ValueError("Result not found in output")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "failed_to_parse_result", error=str(e), stdout=execution_result["stdout"]
            )
            return execution_result["stdout"]

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        return "\n".join(indent + line if line.strip() else line for line in code.split("\n"))

    def _extract_output(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract content between markers."""
        try:
            start_idx = text.index(start_marker) + len(start_marker)
            end_idx = text.index(end_marker)
            return text[start_idx:end_idx].strip()
        except ValueError:
            # Markers not found, return original text
            return text.strip()


@lru_cache(maxsize=1)
def get_code_executor() -> ModalCodeExecutor:
    """Get or create singleton code executor instance (thread-safe via lru_cache)."""
    return ModalCodeExecutor()
