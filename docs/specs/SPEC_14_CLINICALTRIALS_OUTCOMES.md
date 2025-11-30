# SPEC_14: Add Outcome Measures to ClinicalTrials.gov Fields

**Status**: Draft (Validated via API Documentation Review)
**Priority**: P1
**GitHub Issue**: #95
**Estimated Effort**: Small (~40 lines of code)
**Last Updated**: 2025-11-30

---

## Problem Statement

The `ClinicalTrialsTool` retrieves trial metadata but **misses critical efficacy data**:

### Current Fields Retrieved

```python
# src/tools/clinicaltrials.py:24-33
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
```

### Missing Data (Critical for Research)

| Data | Location in Response | Purpose |
|------|---------------------|---------|
| Primary Outcomes | `protocolSection.outcomesModule.primaryOutcomes[].measure` | Main efficacy endpoint |
| Secondary Outcomes | `protocolSection.outcomesModule.secondaryOutcomes[].measure` | Additional endpoints |
| Has Results | `study.hasResults` (top-level) | Whether results are posted |
| Results Date | `protocolSection.statusModule.resultsFirstPostDateStruct.date` | When results posted |

### Impact

**Current Output**:
```
Trial Phase: PHASE3. Status: COMPLETED. Conditions: Erectile Dysfunction.
Interventions: Sildenafil.
```

**Desired Output**:
```
Trial Phase: PHASE3. Status: COMPLETED. Conditions: Erectile Dysfunction.
Interventions: Sildenafil.
Primary Outcome: Change from baseline in IIEF-EF domain score at Week 12.
Results Available: Yes (posted 2024-01-15).
```

---

## API Documentation Review (2025-11-30)

### ClinicalTrials.gov API v2 Response Structure

**Source**: [Stack Overflow - ClinicalTrials.gov API v2](https://stackoverflow.com/questions/78415818)

The API returns nested JSON. Key findings:

1. **`hasResults`** is a **top-level** field on each study object (NOT inside `protocolSection`)
2. **Outcomes** are in `protocolSection.outcomesModule`:
   ```python
   study['protocolSection']['outcomesModule']['primaryOutcomes']  # List
   study['protocolSection']['outcomesModule']['secondaryOutcomes']  # List
   ```
3. **Results date** is in `protocolSection.statusModule.resultsFirstPostDateStruct.date`

### `fields` Parameter Behavior (VERIFIED VIA LIVE API TESTING)

The `fields` query parameter filters what the API returns. **If you don't request a field, you don't get it.**

**Live API Test Results (2025-11-30):**

```bash
# Test 1: With limited fields - NO outcomesModule returned
curl "...&fields=NCTId,BriefTitle"
# → Returns ONLY: protocolSection.identificationModule.{nctId, briefTitle}

# Test 2: Without fields param - outcomesModule IS present
curl "...&pageSize=1"
# → Returns: hasResults: false, outcomesModule: {primaryOutcomes, secondaryOutcomes, otherOutcomes}

# Test 3: Valid field names for outcomes
curl "...&fields=NCTId,OutcomesModule"  # ✅ Works - returns full outcomesModule
curl "...&fields=NCTId,PrimaryOutcome"  # ✅ Works - returns only primaryOutcomes
curl "...&fields=NCTId,HasResults"      # ✅ Works - returns hasResults at top level
```

**Valid Field Names (Tested):**
- `OutcomesModule` → Returns full `protocolSection.outcomesModule` with all outcomes
- `PrimaryOutcome` → Returns only `primaryOutcomes` array
- `SecondaryOutcome` → Returns only `secondaryOutcomes` array
- `HasResults` → Returns `hasResults` at study top level

---

## Proposed Solution

### ✅ UPDATE FIELDS Constant (REQUIRED)

The current implementation explicitly passes `fields=",".join(self.FIELDS)` at line 67.
**The API ONLY returns requested fields.** We MUST add the new field names.

```python
# src/tools/clinicaltrials.py - UPDATE FIELDS
FIELDS: ClassVar[list[str]] = [
    "NCTId",
    "BriefTitle",
    "Phase",
    "OverallStatus",
    "Condition",
    "InterventionName",
    "StartDate",
    "BriefSummary",
    # NEW: Outcome measures (verified via live API testing 2025-11-30)
    "OutcomesModule",  # Returns protocolSection.outcomesModule.{primaryOutcomes, secondaryOutcomes}
    "HasResults",      # Returns study.hasResults (top-level boolean)
]
```

### ✅ Update `_study_to_evidence()` Method

```python
def _study_to_evidence(self, study: dict[str, Any]) -> Evidence:
    """Convert a clinical trial study to Evidence."""
    # Navigate nested structure
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    desc_module = protocol.get("descriptionModule", {})
    design_module = protocol.get("designModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    outcomes_module = protocol.get("outcomesModule", {})  # NEW

    # ... existing field extraction (nct_id, title, status, phase, etc.) ...

    # NEW: Extract outcome measures
    primary_outcomes = outcomes_module.get("primaryOutcomes", [])
    primary_outcome_str = ""
    if primary_outcomes:
        # Get first primary outcome measure and timeframe
        first = primary_outcomes[0]
        measure = first.get("measure", "")
        timeframe = first.get("timeFrame", "")
        # Truncate long outcome descriptions
        primary_outcome_str = measure[:200]
        if timeframe:
            primary_outcome_str += f" (measured at {timeframe})"

    secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
    secondary_count = len(secondary_outcomes)

    # NEW: Check if results are available (hasResults is TOP-LEVEL, not in protocol!)
    has_results = study.get("hasResults", False)

    # Results date is in statusModule (nested inside date struct)
    results_date_struct = status_module.get("resultsFirstPostDateStruct", {})
    results_date = results_date_struct.get("date", "")

    # Build content with key trial info (UPDATED)
    content_parts = [
        f"{summary[:400]}...",
        f"Trial Phase: {phase}.",
        f"Status: {status}.",
        f"Conditions: {conditions_str}.",
        f"Interventions: {interventions_str}.",
    ]

    if primary_outcome_str:
        content_parts.append(f"Primary Outcome: {primary_outcome_str}.")

    if secondary_count > 0:
        content_parts.append(f"Secondary Outcomes: {secondary_count} additional endpoints.")

    if has_results:
        results_info = "Results Available: Yes"
        if results_date:
            results_info += f" (posted {results_date})"
        content_parts.append(results_info + ".")
    else:
        content_parts.append("Results Available: Not yet posted.")

    content = " ".join(content_parts)

    return Evidence(
        content=content[:2000],
        citation=Citation(
            source="clinicaltrials",
            title=title[:500],
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            date=start_date,
            authors=[],
        ),
        relevance=0.90 if has_results else 0.85,  # Boost relevance for trials with results
    )
```

---

## API Reference

The ClinicalTrials.gov API v2 returns nested JSON:

```json
{
  "protocolSection": {
    "outcomesModule": {
      "primaryOutcomes": [
        {
          "measure": "Change from Baseline in IIEF-EF Domain Score",
          "description": "...",
          "timeFrame": "Baseline to Week 12"
        }
      ],
      "secondaryOutcomes": [
        {
          "measure": "Subject Global Assessment Question",
          "timeFrame": "Week 12"
        }
      ]
    }
  },
  "hasResults": true
}
```

See: https://clinicaltrials.gov/data-api/api

---

## Test Plan

### Unit Tests (`tests/unit/tools/test_clinicaltrials.py`)

```python
@pytest.mark.unit
class TestClinicalTrialsOutcomes:
    """Tests for outcome measure extraction."""

    @pytest.mark.asyncio
    async def test_extracts_primary_outcome(self, tool: ClinicalTrialsTool) -> None:
        """Test that primary outcome is extracted from response."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test"},
                "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2023"}},
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": ["PHASE3"]},
                "conditionsModule": {"conditions": ["ED"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "Change in IIEF-EF score",
                            "timeFrame": "Week 12"
                        }
                    ]
                },
            },
            "hasResults": True,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=1)

            assert len(results) == 1
            assert "Primary Outcome" in results[0].content
            assert "IIEF-EF" in results[0].content
            assert "Week 12" in results[0].content

    @pytest.mark.asyncio
    async def test_includes_results_status(self, tool: ClinicalTrialsTool) -> None:
        """Test that results availability is shown."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test"},
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2023"},
                    # Note: resultsFirstPostDateStruct, not resultsFirstSubmitDate
                    "resultsFirstPostDateStruct": {"date": "2024-06-15"},
                },
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": ["PHASE3"]},
                "conditionsModule": {"conditions": ["ED"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": True,  # Note: hasResults is TOP-LEVEL
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=1)

            assert "Results Available: Yes" in results[0].content
            assert "2024-06-15" in results[0].content

    @pytest.mark.asyncio
    async def test_shows_no_results_when_missing(self, tool: ClinicalTrialsTool) -> None:
        """Test that missing results are indicated."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test"},
                "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2024"}},
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": ["PHASE2"]},
                "conditionsModule": {"conditions": ["ED"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=1)

            assert "Results Available: Not yet posted" in results[0].content

    @pytest.mark.asyncio
    async def test_boosts_relevance_for_results(self, tool: ClinicalTrialsTool) -> None:
        """Trials with results should have higher relevance score."""
        with_results = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT11111111", "briefTitle": "With Results"},
                "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2023"}},
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": []},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": True,
        }
        without_results = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT22222222", "briefTitle": "No Results"},
                "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2024"}},
                "descriptionModule": {"briefSummary": "Summary"},
                "designModule": {"phases": []},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {},
            },
            "hasResults": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [with_results, without_results]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=2)

            assert results[0].relevance == 0.90  # With results
            assert results[1].relevance == 0.85  # Without results
```

### Integration Test

```python
@pytest.mark.integration
class TestClinicalTrialsOutcomesIntegration:
    """Integration tests with real API."""

    @pytest.mark.asyncio
    async def test_real_completed_trial_has_outcome(self) -> None:
        """Real completed Phase 3 trials should have outcome measures."""
        tool = ClinicalTrialsTool()

        # Search for completed Phase 3 ED trials (likely to have outcomes)
        results = await tool.search(
            "sildenafil erectile dysfunction Phase 3 COMPLETED",
            max_results=3
        )

        # At least one should have primary outcome
        has_outcome = any("Primary Outcome" in r.content for r in results)
        assert has_outcome, "No completed trials with outcome measures found"
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/tools/clinicaltrials.py` | **ADD** `OutcomesModule` and `HasResults` to `FIELDS`, update `_study_to_evidence()` |
| `tests/unit/tools/test_clinicaltrials.py` | Add outcome parsing tests |

---

## Acceptance Criteria

### FIELDS Constant (REQUIRED CHANGE)
- [ ] `FIELDS` includes `"OutcomesModule"` (returns full outcomesModule)
- [ ] `FIELDS` includes `"HasResults"` (returns top-level boolean)

### `_study_to_evidence()` Method
- [ ] Extracts `protocolSection.outcomesModule.primaryOutcomes`
- [ ] Accesses `study.hasResults` at TOP LEVEL (not inside protocolSection)
- [ ] Results date extracted from `statusModule.resultsFirstPostDateStruct.date`
- [ ] Evidence content includes primary outcome measure when available
- [ ] Evidence content shows results availability status
- [ ] Outcome measure text truncated to 200 chars
- [ ] Trials with results have boosted relevance (0.90 vs 0.85)

### Testing
- [ ] All unit tests pass
- [ ] Integration test confirms real trials return outcome data
- [ ] Live API test confirms `OutcomesModule` and `HasResults` fields work

---

## Edge Cases

1. **No outcomes defined**: Some early-phase trials don't have outcomes yet
   - Solution: Gracefully skip outcome section if `outcomesModule` is empty or missing

2. **Multiple primary outcomes**: Some trials have 2-3 primary outcomes
   - Solution: Show first outcome only, mention count of others

3. **Long outcome descriptions**: Some measures are very verbose (500+ chars)
   - Solution: Truncate measure to 200 chars with `[:200]`

4. **hasResults without resultsFirstPostDateStruct**: Some completed trials may have results without a posted date
   - Solution: Show "Results Available: Yes" without date

5. **outcomesModule missing entirely**: Not all API responses include this module
   - Solution: Use `.get("outcomesModule", {})` for safe access

---

## Rollback Plan

If outcome extraction causes issues:
1. DO NOT modify `FIELDS` - nothing to revert there
2. Remove outcome extraction code from `_study_to_evidence()`
3. Existing tests should still pass

---

## References

- GitHub Issue #95
- [ClinicalTrials.gov API v2 Studies Endpoint](https://clinicaltrials.gov/data-api/api)
- [Stack Overflow - ClinicalTrials.gov API v2 Response Structure](https://stackoverflow.com/questions/78415818)
- `TOOL_ANALYSIS_CRITICAL.md` - "Tool 2: ClinicalTrials.gov > Current Implementation Gaps"
