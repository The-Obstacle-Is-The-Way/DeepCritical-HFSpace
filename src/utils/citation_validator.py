"""Citation validation to prevent LLM hallucination.

CRITICAL: Medical research requires accurate citations.
This module validates that all references exist in collected evidence.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.models import Evidence, ResearchReport

logger = logging.getLogger(__name__)

# Max characters to display for URLs in log messages
_MAX_URL_DISPLAY_LENGTH = 80


def validate_references(report: "ResearchReport", evidence: list["Evidence"]) -> "ResearchReport":
    """Ensure all references actually exist in collected evidence.

    CRITICAL: Prevents LLM hallucination of citations.

    Note:
        This function MUTATES report.references in-place and returns the same
        report object. This is intentional for efficiency.

    Args:
        report: The generated research report (will be mutated)
        evidence: All evidence collected during research

    Returns:
        The same report object with references updated in-place
    """
    # Build set of valid URLs from evidence
    valid_urls = {e.citation.url for e in evidence}
    # Also check titles (case-insensitive, exact match) as fallback
    valid_titles = {e.citation.title.lower() for e in evidence}

    validated_refs = []
    removed_count = 0

    for ref in report.references:
        ref_url = ref.get("url", "")
        ref_title = ref.get("title", "").lower()

        # Check if URL matches collected evidence
        if ref_url in valid_urls:
            validated_refs.append(ref)
        # Fallback: exact title match (case-insensitive)
        elif ref_title and ref_title in valid_titles:
            validated_refs.append(ref)
        else:
            removed_count += 1
            # Truncate URL for display
            if len(ref_url) > _MAX_URL_DISPLAY_LENGTH:
                url_display = ref_url[:_MAX_URL_DISPLAY_LENGTH] + "..."
            else:
                url_display = ref_url
            logger.warning(
                f"Removed hallucinated reference: '{ref.get('title', 'Unknown')}' "
                f"(URL: {url_display})"
            )

    if removed_count > 0:
        logger.info(
            f"Citation validation removed {removed_count} hallucinated references. "
            f"{len(validated_refs)} valid references remain."
        )

    # Update report with validated references
    report.references = validated_refs
    return report


def build_reference_from_evidence(evidence: "Evidence") -> dict[str, str]:
    """Build a properly formatted reference from evidence.

    Use this to ensure references match the original evidence exactly.
    """
    return {
        "title": evidence.citation.title,
        "authors": ", ".join(evidence.citation.authors or ["Unknown"]),
        "source": evidence.citation.source,
        "date": evidence.citation.date or "n.d.",
        "url": evidence.citation.url,
    }
