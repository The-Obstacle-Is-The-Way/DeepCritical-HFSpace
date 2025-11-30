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
    See: https://clinicaltrials.gov/data-api/api
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    # Fields to retrieve
    FIELDS: ClassVar[list[str]] = [
        "NCTId",
        "BriefTitle",
        "Phase",
        "OverallStatus",
        "Condition",
        "InterventionName",
        "StartDate",
        "BriefSummary",
        # NEW: Outcome measures
        "OutcomesModule",
        "HasResults",
    ]

    # Status filter: Only active/completed studies with potential data
    STATUS_FILTER = "COMPLETED,ACTIVE_NOT_RECRUITING,RECRUITING,ENROLLING_BY_INVITATION"

    # Study type filter: Only interventional (drug/treatment studies)
    STUDY_TYPE_FILTER = "INTERVENTIONAL"

    @property
    def name(self) -> str:
        return "clinicaltrials"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """Search ClinicalTrials.gov for interventional studies.

        Args:
            query: Search query (e.g., "testosterone libido")
            max_results: Maximum results to return (max 100)

        Returns:
            List of Evidence objects from clinical trials
        """
        # Add study type filter to query string (parameter is not supported)
        # AREA[StudyType]INTERVENTIONAL restricts to interventional studies
        final_query = f"{query} AND AREA[StudyType]INTERVENTIONAL"

        params: dict[str, Any] = {
            "query.term": final_query,
            "pageSize": min(max_results, 100),
            "fields": ",".join(self.FIELDS),
            # FILTERS - Only active/completed studies
            "filter.overallStatus": self.STATUS_FILTER,
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

    def _extract_primary_outcome(self, outcomes_module: dict[str, Any]) -> str:
        """Extract and format primary outcome from outcomes module."""
        primary_outcomes = outcomes_module.get("primaryOutcomes", [])
        if not primary_outcomes:
            return ""
        # Get first primary outcome measure and timeframe
        first = primary_outcomes[0]
        measure = first.get("measure", "")
        timeframe = first.get("timeFrame", "")
        # Build full outcome string first, then truncate
        result = f"{measure} (measured at {timeframe})" if timeframe else measure
        # Truncate long outcome descriptions with ellipsis
        return result[:197] + "..." if len(result) > 200 else result

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
        outcomes_module = protocol.get("outcomesModule", {})

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

        # Extract outcome measures
        primary_outcome_str = self._extract_primary_outcome(outcomes_module)
        secondary_count = len(outcomes_module.get("secondaryOutcomes", []))

        # Check if results are available (hasResults is TOP-LEVEL, not in protocol!)
        has_results = study.get("hasResults", False)

        # Results date is in statusModule (nested inside date struct)
        results_date_struct = status_module.get("resultsFirstPostDateStruct", {})
        results_date = results_date_struct.get("date", "")

        # Build content with key trial info
        summary_text = summary[:400] + "..." if len(summary) > 400 else summary
        content_parts = [
            summary_text,
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
                authors=[],  # Trials don't have traditional authors
            ),
            relevance=0.90 if has_results else 0.85,  # Boost relevance for trials with results
        )
