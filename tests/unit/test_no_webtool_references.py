import ast
import pathlib

import pytest


def test_examples_no_webtool_imports():
    """No example files should import WebTool or the websearch module."""
    examples_dir = pathlib.Path("examples")

    for py_file in examples_dir.rglob("*.py"):
        content = py_file.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "websearch" in module:
                    pytest.fail(f"{py_file} imports websearch (should be removed)")
                # Also check for `from src.tools import WebTool`
                for alias in node.names:
                    if alias.name == "WebTool":
                        pytest.fail(f"{py_file} imports WebTool (should be removed)")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "websearch" in alias.name:
                        pytest.fail(f"{py_file} imports websearch (should be removed)")
