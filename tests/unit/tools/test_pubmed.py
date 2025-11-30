"""Unit tests for PubMed tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.pubmed import PubMedTool

# Sample PubMed XML response for mocking
SAMPLE_PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Testosterone Therapy for HSDD</ArticleTitle>
                <Abstract>
                    <AbstractText>Testosterone shows efficacy in HSDD...</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                </AuthorList>
                <Journal>
                    <JournalIssue>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>01</Month>
                        </PubDate>
                    </JournalIssue>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedTool:
    """Tests for PubMedTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """PubMedTool should return Evidence objects from search."""
        # Mock the HTTP responses
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": ["12345678"]}}
        mock_search_response.raise_for_status = MagicMock()

        mock_fetch_xml = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Testosterone and Libido</ArticleTitle>
                        <Abstract>
                            <AbstractText>Testosterone improves libido.</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author><LastName>Doe</LastName><ForeName>John</ForeName></Author>
                        </AuthorList>
                        <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="pubmed">12345678</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """

        mock_fetch_response = MagicMock()
        mock_fetch_response.text = mock_fetch_xml
        mock_fetch_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_search_response, mock_fetch_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Act
        tool = PubMedTool()
        results = await tool.search("testosterone libido")

        # Assert
        assert len(results) == 1
        assert results[0].citation.source == "pubmed"
        assert "Testosterone" in results[0].citation.title
        assert "12345678" in results[0].citation.url

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mocker):
        """PubMedTool should return empty list when no results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        results = await tool.search("xyznonexistentquery123")

        assert results == []

    def test_parse_pubmed_xml(self):
        """PubMedTool should correctly parse XML."""
        tool = PubMedTool()
        results = tool._parse_pubmed_xml(SAMPLE_PUBMED_XML)

        assert len(results) == 1
        assert results[0].citation.source == "pubmed"
        assert "Smith John" in results[0].citation.authors

    @pytest.mark.asyncio
    async def test_search_preprocesses_query(self, mocker):
        """Test that queries are preprocessed before search."""
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_search_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_search_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        await tool.search("What medications help with Low Libido?")

        # Verify call args
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        term = params["term"]

        # "what" and "help" should be stripped
        assert "what" not in term.lower()
        assert "help" not in term.lower()
        # "low libido" should be expanded
        assert "HSDD" in term or "hypoactive" in term
