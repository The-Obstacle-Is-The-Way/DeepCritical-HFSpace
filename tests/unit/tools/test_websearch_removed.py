def test_websearch_module_deleted():
    """WebTool should no longer exist."""
    import pytest

    with pytest.raises(ImportError):
        from src.tools.websearch import WebTool  # noqa: F401
