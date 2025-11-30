# Code Quality Audit Findings - 2025-11-30

**Auditor:** Senior Staff Engineer (Gemini)
**Date:** 2025-11-30
**Scope:** `src/` (services, tools, agents, orchestrators)
**Focus:** Configuration validation, Error handling, Defensive programming anti-patterns

## Summary

The codebase is generally clean and modern, but exhibits specific anti-patterns related to configuration management and defensive error handling. The most critical finding is the reliance on manual `os.getenv` calls and "silent default" fallbacks which obscure configuration errors, directly contributing to the `OpenAIError` observed in production.

## Findings

### 1. Defensive Pass Block (Silent Failure) - MEDIUM
**File:** `src/services/statistical_analyzer.py:246-247`
```python
            try:
                min_p = min(float(p) for p in p_values)
                # ... logic ...
            except ValueError:
                pass
```
**Problem:** If p-values are found by regex but fail to parse, the error is swallowed silently. This makes debugging parser issues impossible.
**Fix:** Replace `pass` with `logger.warning("Failed to parse p-values: %s", p_values)` to aid debugging.

### 2. Missing Pydantic Validation (Manual Config) - MEDIUM
**File:** `src/tools/code_execution.py:75-76`
```python
        self.modal_token_id = os.getenv("MODAL_TOKEN_ID")
        self.modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")
```
**Problem:** Secrets are manually fetched from env vars, bypassing the centralized `Settings` validation.
**Fix:** Move to `src/utils/config.py` in the `Settings` class and inject `settings` into `ModalCodeExecutor`.

### 3. Broad Exception Swallowing - MEDIUM
**File:** `src/tools/pubmed.py:129-130`
```python
            except Exception:
                continue  # Skip malformed articles
```
**Problem:** Catching `Exception` hides potential bugs (like `NameError` or `TypeError` in our own code), not just malformed data.
**Fix:** Catch specific exceptions (e.g., `(KeyError, AttributeError, TypeError)`) OR log the error before continuing: `logger.debug(f"Skipping malformed article {pmid}: {e}")`.

### 4. Missing Pydantic Validation (UI Layer) - LOW
**File:** `src/app.py:115, 119`
```python
    elif os.getenv("OPENAI_API_KEY"):
        # ...
    elif os.getenv("ANTHROPIC_API_KEY"):
```
**Problem:** Application logic relies on raw environment variable checks to determine available backends, creating duplication and potential inconsistency with `config.py`.
**Fix:** Centralize this logic in `src/utils/config.py` (e.g., `settings.has_openai`, `settings.has_anthropic`).

### 5. Try/Except for Flow Control - LOW
**File:** `src/tools/code_execution.py:244-249`
```python
        try:
            start_idx = text.index(start_marker) + len(start_marker)
            # ...
        except ValueError:
            return text.strip()
```
**Problem:** Using exceptions for expected "not found" cases is slower and less explicit.
**Fix:** Use `find()` which returns `-1` on failure.

## Action Plan

1.  **Refactor Configuration:** Eliminate `os.getenv` in favor of `src/utils/config.py` `Settings` model.
2.  **Fix Error Handling:** Remove empty `pass` blocks; add logging.
3.  **Address P0 Bug:** Fix the `OpenAIError` in synthesis (caused by Finding #4/General Config issue) by injecting the correct model into the orchestrator.
