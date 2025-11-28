# ClinicalTrials.gov Tool: Current State & Future Improvements

**Status**: Currently Implemented
**Priority**: High (Core Data Source for Drug Repurposing)

---

## Current Implementation

### What We Have (`src/tools/clinicaltrials.py`)

- V2 API search via `clinicaltrials.gov/api/v2/studies`
- Filters: `INTERVENTIONAL` study type, `RECRUITING` status
- Returns: NCT ID, title, conditions, interventions, phase, status
- Query preprocessing via shared `query_utils.py`

### Current Strengths

1. **Good Filtering**: Already filtering for interventional + recruiting
2. **V2 API**: Using the modern API (v1 deprecated)
3. **Phase Info**: Extracting trial phases for drug development context

### Current Limitations

1. **No Outcome Data**: Missing primary/secondary outcomes
2. **No Eligibility Criteria**: Missing inclusion/exclusion details
3. **No Sponsor Info**: Missing who's running the trial
4. **No Result Data**: For completed trials, no efficacy data
5. **Limited Drug Mapping**: No integration with drug databases

---

## API Capabilities We're Not Using

### Fields We Could Request

```python
# Current fields
fields = ["NCTId", "BriefTitle", "Condition", "InterventionName", "Phase", "OverallStatus"]

# Additional valuable fields
additional_fields = [
    "PrimaryOutcomeMeasure",      # What are they measuring?
    "SecondaryOutcomeMeasure",    # Secondary endpoints
    "EligibilityCriteria",        # Who can participate?
    "LeadSponsorName",            # Who's funding?
    "ResultsFirstPostDate",       # Has results?
    "StudyFirstPostDate",         # When started?
    "CompletionDate",             # When finished?
    "EnrollmentCount",            # Sample size
    "InterventionDescription",    # Drug details
    "ArmGroupLabel",              # Treatment arms
    "InterventionOtherName",      # Drug aliases
]
```

### Filter Enhancements

```python
# Current
aggFilters = "studyType:INTERVENTIONAL,status:RECRUITING"

# Could add
"status:RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED"  # Include completed for results
"phase:PHASE2,PHASE3"  # Only later-stage trials
"resultsFirstPostDateRange:2020-01-01_"  # Trials with posted results
```

---

## Recommended Improvements

### Phase 1: Richer Metadata

```python
EXTENDED_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "Condition",
    "InterventionName",
    "InterventionDescription",
    "InterventionOtherName",  # Drug synonyms!
    "Phase",
    "OverallStatus",
    "PrimaryOutcomeMeasure",
    "EnrollmentCount",
    "LeadSponsorName",
    "StudyFirstPostDate",
]
```

### Phase 2: Results Retrieval

For completed trials, we can get actual efficacy data:

```python
async def get_trial_results(nct_id: str) -> dict | None:
    """Fetch results for completed trials."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    params = {
        "fields": "ResultsSection",
    }
    # Returns outcome measures and statistics
```

### Phase 3: Drug Name Normalization

Map intervention names to standard identifiers:

```python
# Problem: "Metformin", "Metformin HCl", "Glucophage" are the same drug
# Solution: Use RxNorm or DrugBank for normalization

async def normalize_drug_name(intervention: str) -> str:
    """Normalize drug name via RxNorm API."""
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={intervention}"
    # Returns standardized RxCUI
```

---

## Integration Opportunities

### With PubMed

Cross-reference trials with publications:
```python
# ClinicalTrials.gov provides PMID links
# Can correlate trial results with published papers
```

### With DrugBank/ChEMBL

Map interventions to:
- Mechanism of action
- Known targets
- Adverse effects
- Drug-drug interactions

---

## Python Libraries to Consider

| Library | Purpose | Notes |
|---------|---------|-------|
| [pytrials](https://pypi.org/project/pytrials/) | CT.gov wrapper | V2 API support unclear |
| [clinicaltrials](https://github.com/ebmdatalab/clinicaltrials-act-tracker) | Data tracking | More for analysis |
| [drugbank-downloader](https://pypi.org/project/drugbank-downloader/) | Drug mapping | Requires license |

---

## API Quirks & Gotchas

1. **Rate Limiting**: Undocumented, be conservative
2. **Pagination**: Max 1000 results per request
3. **Field Names**: Case-sensitive, camelCase
4. **Empty Results**: Some fields may be null even if requested
5. **Status Changes**: Trials change status frequently

---

## Example Enhanced Query

```python
async def search_drug_repurposing_trials(
    drug_name: str,
    condition: str,
    include_completed: bool = True,
) -> list[Evidence]:
    """Search for trials repurposing a drug for a new condition."""

    statuses = ["RECRUITING", "ACTIVE_NOT_RECRUITING"]
    if include_completed:
        statuses.append("COMPLETED")

    params = {
        "query.intr": drug_name,
        "query.cond": condition,
        "filter.overallStatus": ",".join(statuses),
        "filter.studyType": "INTERVENTIONAL",
        "fields": ",".join(EXTENDED_FIELDS),
        "pageSize": 50,
    }
```

---

## Sources

- [ClinicalTrials.gov API Documentation](https://clinicaltrials.gov/data-api/api)
- [CT.gov Field Definitions](https://clinicaltrials.gov/data-api/about-api/study-data-structure)
- [RxNorm API](https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.findRxcuiByString.html)
