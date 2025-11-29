# Phase 10 Implementation Spec: ClinicalTrials.gov Integration

**Goal**: Add clinical trial search for drug repurposing evidence.
**Philosophy**: "Clinical trials are the bridge from hypothesis to therapy."
**Prerequisite**: Phase 9 complete (DuckDuckGo removed)
**Estimated Time**: 2-3 hours

---

## 1. Why ClinicalTrials.gov?

### Scientific Value

| Feature | Value for Drug Repurposing |
|---------|---------------------------|
| **400,000+ studies** | Massive evidence base |
| **Trial phase data** | Phase I/II/III = evidence strength |
| **Intervention details** | Exact drug + dosing |
| **Outcome measures** | What was measured |
| **Status tracking** | Completed vs recruiting |
| **Free API** | No cost, no key required |

### Example Query Response

Query: "metformin Alzheimer's"

```json
{
  "studies": [
    {
      "nctId": "NCT04098666",
      "briefTitle": "Metformin in Alzheimer's Dementia Prevention",
      "phase": "Phase 2",
      "status": "Recruiting",
      "conditions": ["Alzheimer Disease"],
      "interventions": ["Drug: Metformin"]
    }
  ]
}
```

**This is GOLD for drug repurposing** - actual trials testing the hypothesis!

---

## 2. API Specification

### Endpoint

```
Base URL: https://clinicaltrials.gov/api/v2/studies
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `query.cond` | Condition/disease | `Alzheimer` |
| `query.intr` | Intervention/drug | `Metformin` |
| `query.term` | General search | `metformin alzheimer` |
| `pageSize` | Results per page | `20` |
| `fields` | Fields to return | See below |

### Fields We Need

```
NCTId, BriefTitle, Phase, OverallStatus, Condition,
InterventionName, StartDate, CompletionDate, BriefSummary
```

### Rate Limits

- ~50 requests/minute per IP
- No authentication required
- Paginated (100 results max per call)

### Documentation

- [API v2 Docs](https://clinicaltrials.gov/data-api/api)
- [Migration Guide](https://www.nlm.nih.gov/pubs/techbull/ma24/ma24_clinicaltrials_api.html)

---

## 3. Data Model

### 3.1 Update Citation Source Type (`src/utils/models.py`)

```python
# BEFORE
source: Literal["pubmed", "web"]

# AFTER
source: Literal["pubmed", "clinicaltrials", "biorxiv"]
```

### 3.2 Evidence from Clinical Trials

Clinical trial data maps to our existing `Evidence` model:

```python
Evidence(
    content=f"{brief_summary}. Phase: {phase}. Status: {status}.",
    citation=Citation(
        source="clinicaltrials",
        title=brief_title,
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        date=start_date or "Unknown",
        authors=[]  # Trials don't have authors in the same way
    ),
    relevance=0.8  # Trials are highly relevant for repurposing
)
```

---

## 4. Implementation

### 4.0 Important: HTTP Client Selection

**ClinicalTrials.gov's WAF blocks `httpx`'s TLS fingerprint.** Use `requests` instead.

| Library | Status | Notes |
|---------|--------|-------|
| `httpx` | ❌ 403 Blocked | TLS/JA3 fingerprint flagged |
| `httpx[http2]` | ❌ 403 Blocked | HTTP/2 doesn't help |
| `requests` | ✅ Works | Industry standard, not blocked |
| `urllib` | ✅ Works | Stdlib alternative |

We use `requests` wrapped in `asyncio.to_thread()` for async compatibility.

### 4.1 ClinicalTrials Tool (`src/tools/clinicaltrials.py`)

```python
"""ClinicalTrials.gov search tool using API v2."""

import asyncio
from typing import Any, ClassVar

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class ClinicalTrialsTool:
    """Search tool for ClinicalTrials.gov.

    Note: Uses `requests` library instead of `httpx` because ClinicalTrials.gov's
    WAF blocks httpx's TLS fingerprint. The `requests` library is not blocked.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    FIELDS: ClassVar[list[str]] = [
        "NCTId",
        "BriefTitle",
        "Phase",
        "OverallStatus",
        "Condition",
        "InterventionName",
        "StartDate",
        "BriefSummary",
    ]

    @property
    def name(self) -> str:
        return "clinicaltrials"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """Search ClinicalTrials.gov for studies."""
        params = {
            "query.term": query,
            "pageSize": min(max_results, 100),
            "fields": "|".join(self.FIELDS),
        }

        try:
            # Run blocking requests.get in a separate thread for async compatibility
            response = await asyncio.to_thread(
                requests.get,
                self.BASE_URL,
                params=params,
                headers={"User-Agent": "DeepBoner-Research-Agent/1.0"},
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            studies = data.get("studies", [])
            return [self._study_to_evidence(study) for study in studies[:max_results]]

        except requests.HTTPError as e:
            raise SearchError(f"ClinicalTrials.gov API error: {e}") from e
        except requests.RequestException as e:
            raise SearchError(f"ClinicalTrials.gov request failed: {e}") from e

    def _study_to_evidence(self, study: dict) -> Evidence:
        """Convert a clinical trial study to Evidence."""
        # Navigate nested structure
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        desc_module = protocol.get("descriptionModule", {})
        design_module = protocol.get("designModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})

        nct_id = id_module.get("nctId", "Unknown")
        title = id_module.get("briefTitle", "Untitled Study")
        status = status_module.get("overallStatus", "Unknown")
        start_date = status_module.get("startDateStruct", {}).get("date", "Unknown")

        # Get phase (might be a list)
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else "Not Applicable"

        # Get conditions
        conditions = conditions_module.get("conditions", [])
        conditions_str = ", ".join(conditions[:3]) if conditions else "Unknown"

        # Get interventions
        interventions = arms_module.get("interventions", [])
        intervention_names = [i.get("name", "") for i in interventions[:3]]
        interventions_str = ", ".join(intervention_names) if intervention_names else "Unknown"

        # Get summary
        summary = desc_module.get("briefSummary", "No summary available.")

        # Build content with key trial info
        content = (
            f"{summary[:500]}... "
            f"Trial Phase: {phase}. "
            f"Status: {status}. "
            f"Conditions: {conditions_str}. "
            f"Interventions: {interventions_str}."
        )

        return Evidence(
            content=content[:2000],
            citation=Citation(
                source="clinicaltrials",
                title=title[:500],
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                date=start_date,
                authors=[],  # Trials don't have traditional authors
            ),
            relevance=0.85,  # Trials are highly relevant for repurposing
        )
```

---

## 5. TDD Test Suite

### 5.1 Unit Tests (`tests/unit/tools/test_clinicaltrials.py`)

Uses `unittest.mock.patch` to mock `requests.get` (not `respx` since we're not using `httpx`).

```python
"""Unit tests for ClinicalTrials.gov tool."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tools.clinicaltrials import ClinicalTrialsTool
from src.utils.exceptions import SearchError
from src.utils.models import Evidence


@pytest.fixture
def mock_clinicaltrials_response() -> dict:
    """Mock ClinicalTrials.gov API response."""
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT04098666",
                        "briefTitle": "Metformin in Alzheimer's Dementia Prevention",
                    },
                    "statusModule": {
                        "overallStatus": "Recruiting",
                        "startDateStruct": {"date": "2020-01-15"},
                    },
                    "descriptionModule": {
                        "briefSummary": "This study evaluates metformin for Alzheimer's prevention."
                    },
                    "designModule": {"phases": ["PHASE2"]},
                    "conditionsModule": {"conditions": ["Alzheimer Disease", "Dementia"]},
                    "armsInterventionsModule": {
                        "interventions": [{"name": "Metformin", "type": "Drug"}]
                    },
                }
            }
        ]
    }


class TestClinicalTrialsTool:
    """Tests for ClinicalTrialsTool."""

    def test_tool_name(self) -> None:
        """Tool should have correct name."""
        tool = ClinicalTrialsTool()
        assert tool.name == "clinicaltrials"

    @pytest.mark.asyncio
    async def test_search_returns_evidence(
        self, mock_clinicaltrials_response: dict
    ) -> None:
        """Search should return Evidence objects."""
        with patch("src.tools.clinicaltrials.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_clinicaltrials_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            tool = ClinicalTrialsTool()
            results = await tool.search("metformin alzheimer", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert results[0].citation.source == "clinicaltrials"
            assert "NCT04098666" in results[0].citation.url
            assert "Metformin" in results[0].citation.title

    @pytest.mark.asyncio
    async def test_search_api_error(self) -> None:
        """Search should raise SearchError on API failure."""
        with patch("src.tools.clinicaltrials.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "500 Server Error"
            )
            mock_get.return_value = mock_response

            tool = ClinicalTrialsTool()

            with pytest.raises(SearchError):
                await tool.search("metformin alzheimer")


class TestClinicalTrialsIntegration:
    """Integration tests (marked for separate run)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_call(self) -> None:
        """Test actual API call (requires network)."""
        tool = ClinicalTrialsTool()
        results = await tool.search("metformin diabetes", max_results=3)

        assert len(results) > 0
        assert all(isinstance(r, Evidence) for r in results)
        assert all(r.citation.source == "clinicaltrials" for r in results)
```

---

## 6. Integration with SearchHandler

### 6.1 Update Example Files

```python
# examples/search_demo/run_search.py
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler

search_handler = SearchHandler(
    tools=[PubMedTool(), ClinicalTrialsTool()],
    timeout=30.0
)
```

### 6.2 Update SearchResult Type

```python
# src/utils/models.py
sources_searched: list[Literal["pubmed", "clinicaltrials"]]
```

---

## 7. Definition of Done

Phase 10 is **COMPLETE** when:

- [ ] `src/tools/clinicaltrials.py` implemented
- [ ] Unit tests in `tests/unit/tools/test_clinicaltrials.py`
- [ ] Integration test marked with `@pytest.mark.integration`
- [ ] SearchHandler updated to include ClinicalTrialsTool
- [ ] Type definitions updated in models.py
- [ ] Example files updated
- [ ] All unit tests pass
- [ ] Lints pass
- [ ] Manual verification with real API

---

## 8. Verification Commands

```bash
# 1. Run unit tests
uv run pytest tests/unit/tools/test_clinicaltrials.py -v

# 2. Run integration test (requires network)
uv run pytest tests/unit/tools/test_clinicaltrials.py -v -m integration

# 3. Run full test suite
uv run pytest tests/unit/ -v

# 4. Run example
source .env && uv run python examples/search_demo/run_search.py "metformin alzheimer"
# Should show results from BOTH PubMed AND ClinicalTrials.gov
```

---

## 9. Value Delivered

| Before | After |
|--------|-------|
| Papers only | Papers + Clinical Trials |
| "Drug X might help" | "Drug X is in Phase II trial" |
| No trial status | Recruiting/Completed/Terminated |
| No phase info | Phase I/II/III evidence strength |

**Demo pitch addition**:
> "DeepBoner searches PubMed for peer-reviewed evidence AND ClinicalTrials.gov for 400,000+ clinical trials."
