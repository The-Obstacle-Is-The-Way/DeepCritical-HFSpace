"""Data models for the Search feature."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation to a source document."""

    source: Literal["pubmed", "web"] = Field(description="Where this came from")
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(description="URL to the source")
    date: str = Field(description="Publication date (YYYY-MM-DD or 'Unknown')")
    authors: list[str] = Field(default_factory=list)

    MAX_AUTHORS_IN_CITATION: ClassVar[int] = 3

    @property
    def formatted(self) -> str:
        """Format as a citation string."""
        author_str = ", ".join(self.authors[: self.MAX_AUTHORS_IN_CITATION])
        if len(self.authors) > self.MAX_AUTHORS_IN_CITATION:
            author_str += " et al."
        return f"{author_str} ({self.date}). {self.title}. {self.source.upper()}"


class Evidence(BaseModel):
    """A piece of evidence retrieved from search."""

    content: str = Field(min_length=1, description="The actual text content")
    citation: Citation
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score 0-1")

    model_config = {"frozen": True}


class SearchResult(BaseModel):
    """Result of a search operation."""

    query: str
    evidence: list[Evidence]
    sources_searched: list[Literal["pubmed", "web"]]
    total_found: int
    errors: list[str] = Field(default_factory=list)
