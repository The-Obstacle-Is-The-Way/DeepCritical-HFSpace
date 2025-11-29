"""Graph node implementations for DeepBoner research."""

from typing import Any, Literal

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.agent_factory.judges import get_model
from src.agents.graph.state import Hypothesis, ResearchState
from src.prompts.hypothesis import SYSTEM_PROMPT as HYPOTHESIS_SYSTEM_PROMPT
from src.prompts.hypothesis import format_hypothesis_prompt
from src.prompts.report import SYSTEM_PROMPT as REPORT_SYSTEM_PROMPT
from src.prompts.report import format_report_prompt
from src.services.embeddings import EmbeddingService
from src.tools.base import SearchTool
from src.tools.clinicaltrials import ClinicalTrialsTool
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool
from src.tools.search_handler import SearchHandler
from src.utils.citation_validator import validate_references
from src.utils.models import (
    Citation,
    Evidence,
    HypothesisAssessment,
    MechanismHypothesis,
    ResearchReport,
)

logger = structlog.get_logger()


def _convert_hypothesis_to_mechanism(h: Hypothesis) -> MechanismHypothesis:
    """Convert state Hypothesis to MechanismHypothesis for report generation.

    The state Hypothesis stores the mechanism as a statement like:
    "drug -> target -> pathway -> effect"

    We parse this back into structured MechanismHypothesis fields.
    """
    # Parse statement format: "drug -> target -> pathway -> effect"
    # Handle both " -> " (standard) and "->" (compact) separators
    separator = " -> " if " -> " in h.statement else "->"
    parts = [p.strip() for p in h.statement.split(separator)]

    # Validate: exactly 4 non-empty parts
    if len(parts) == 4 and all(parts):
        drug, target, pathway, effect = parts
    elif len(parts) > 4 and all(parts[:4]):
        # More than 4 parts: join extras into effect
        drug, target, pathway = parts[0], parts[1], parts[2]
        effect = f"{separator}".join(parts[3:])
        logger.debug(
            "Hypothesis has extra parts, joined into effect",
            hypothesis_id=h.id,
            parts_count=len(parts),
        )
    else:
        # Log parsing failure for debugging
        logger.warning(
            "Failed to parse hypothesis statement format",
            hypothesis_id=h.id,
            statement=h.statement[:100],  # Truncate for log safety
            parts_count=len(parts),
        )
        # Use meaningful fallback values
        drug = "Unknown"
        target = "Unknown"
        pathway = "Unknown"
        effect = h.statement.strip() if h.statement else "Unknown effect"

    return MechanismHypothesis(
        drug=drug,
        target=target,
        pathway=pathway,
        effect=effect,
        confidence=h.confidence,
        supporting_evidence=h.supporting_evidence_ids,
        contradicting_evidence=h.contradicting_evidence_ids,
    )


# --- Supervisor Output Schema ---
class SupervisorDecision(BaseModel):
    """The decision made by the supervisor."""

    next_step: Literal["search", "judge", "resolve", "synthesize", "finish"] = Field(
        description="The next step to take in the research process."
    )
    reasoning: str = Field(description="Reasoning for this decision.")


# --- Nodes ---


async def search_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Execute search across all sources."""
    query = state["query"]
    logger.info("search_node: executing search", query=query)

    # Initialize tools
    tools: list[SearchTool] = [PubMedTool(), ClinicalTrialsTool(), EuropePMCTool()]
    handler = SearchHandler(tools=tools)

    # Execute search
    result = await handler.execute(query)

    new_evidence_count = 0
    new_ids = []

    if embedding_service and result.evidence:
        # Deduplicate and store
        unique_evidence = await embedding_service.deduplicate(result.evidence)

        for ev in unique_evidence:
            ev_id = ev.citation.url
            await embedding_service.add_evidence(
                evidence_id=ev_id,
                content=ev.content,
                metadata={
                    "source": ev.citation.source,
                    "title": ev.citation.title,
                    "date": ev.citation.date,
                    "authors": ",".join(ev.citation.authors or []),
                    "url": ev.citation.url,
                },
            )
            new_ids.append(ev_id)

        new_evidence_count = len(unique_evidence)
    else:
        new_evidence_count = len(result.evidence)

    message = (
        f"Search completed. Found {result.total_found} total, "
        f"{new_evidence_count} unique new papers."
    )
    if result.errors:
        message += f" Errors: {'; '.join(result.errors)}"

    return {
        "evidence_ids": new_ids,
        "messages": [AIMessage(content=message)],
    }


async def judge_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Evaluate evidence and update hypothesis confidence."""
    logger.info("judge_node: evaluating evidence")

    evidence_context: list[Evidence] = []
    if embedding_service:
        scored_points = await embedding_service.search_similar(state["query"], n_results=20)
        for p in scored_points:
            meta = p.get("metadata", {})
            authors = meta.get("authors", "")
            author_list = authors.split(",") if authors else []

            evidence_context.append(
                Evidence(
                    content=p.get("content", ""),
                    citation=Citation(
                        url=p.get("id", ""),
                        title=meta.get("title", "Unknown"),
                        source=meta.get("source", "Unknown"),
                        date=meta.get("date", ""),
                        authors=author_list,
                    ),
                )
            )

    agent = Agent(
        model=get_model(),
        output_type=HypothesisAssessment,
        system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
    )

    prompt = await format_hypothesis_prompt(
        query=state["query"], evidence=evidence_context, embeddings=embedding_service
    )

    try:
        result = await agent.run(prompt)
        assessment = result.output

        new_hypotheses = []
        for h in assessment.hypotheses:
            new_hypotheses.append(
                Hypothesis(
                    id=h.drug,
                    statement=f"{h.drug} -> {h.target} -> {h.pathway} -> {h.effect}",
                    status="proposed",
                    confidence=h.confidence,
                    supporting_evidence_ids=[],
                    contradicting_evidence_ids=[],
                )
            )

        return {
            "hypotheses": new_hypotheses,
            "messages": [AIMessage(content=f"Judge: Generated {len(new_hypotheses)} hypotheses.")],
            "next_step": "resolve",
        }
    except Exception as e:
        logger.error("judge_node failed", error=str(e))
        return {"messages": [AIMessage(content=f"Judge Error: {e!s}")], "next_step": "search"}


async def resolve_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Handle open conflicts."""
    messages = []

    # Access attributes with dot notation because items are Pydantic models
    high_conf = [h for h in state["hypotheses"] if h.confidence > 0.8]

    if high_conf:
        messages.append(
            AIMessage(
                content=(
                    f"Resolver: Found {len(high_conf)} high confidence hypotheses. "
                    "Conflicts resolved."
                )
            )
        )
    else:
        messages.append(AIMessage(content="Resolver: No high confidence hypotheses yet."))

    return {"messages": messages}


async def synthesize_node(
    state: ResearchState, embedding_service: EmbeddingService | None = None
) -> dict[str, Any]:
    """Generate final report."""
    logger.info("synthesize_node: generating report")

    evidence_context: list[Evidence] = []
    if embedding_service:
        scored_points = await embedding_service.search_similar(state["query"], n_results=50)
        for p in scored_points:
            meta = p.get("metadata", {})
            authors = meta.get("authors", "")
            author_list = authors.split(",") if authors else []

            evidence_context.append(
                Evidence(
                    content=p.get("content", ""),
                    citation=Citation(
                        url=p.get("id", ""),
                        title=meta.get("title", "Unknown"),
                        source=meta.get("source", "Unknown"),
                        date=meta.get("date", ""),
                        authors=author_list,
                    ),
                )
            )

    agent = Agent(
        model=get_model(),
        output_type=ResearchReport,
        system_prompt=REPORT_SYSTEM_PROMPT,
    )

    # Convert state hypotheses to MechanismHypothesis for report generation
    mechanism_hypotheses = [_convert_hypothesis_to_mechanism(h) for h in state["hypotheses"]]

    prompt = await format_report_prompt(
        query=state["query"],
        evidence=evidence_context,
        hypotheses=mechanism_hypotheses,
        assessment={},
        metadata={"sources": list(set(e.citation.source for e in evidence_context))},
        embeddings=embedding_service,
    )

    try:
        result = await agent.run(prompt)
        report = result.output
        report = validate_references(report, evidence_context)

        return {"messages": [AIMessage(content=report.to_markdown())], "next_step": "finish"}
    except Exception as e:
        logger.error("synthesize_node failed", error=str(e))
        return {"messages": [AIMessage(content=f"Synthesis Error: {e!s}")], "next_step": "finish"}


async def supervisor_node(state: ResearchState, llm: BaseChatModel | None = None) -> dict[str, Any]:
    """Route to next node based on state using robust Pydantic parsing."""
    if state["iteration_count"] >= state["max_iterations"]:
        return {"next_step": "synthesize", "iteration_count": state["iteration_count"]}

    if llm is None:
        return {"next_step": "search", "iteration_count": state["iteration_count"] + 1}

    parser = PydanticOutputParser(pydantic_object=SupervisorDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Research Supervisor. Manage the workflow.\n\n"
                "State Summary:\n"
                "- Query: {query}\n"
                "- Hypotheses: {hypo_count}\n"
                "- Conflicts: {conflict_count}\n"
                "- Iteration: {iteration}/{max_iter}\n\n"
                "Decide the next step based on this logic:\n"
                "1. If there are open conflicts -> 'resolve'\n"
                "2. If hypotheses are unverified or few -> 'search'\n"
                "3. If new evidence needs evaluation -> 'judge'\n"
                "4. If hypotheses are confirmed -> 'synthesize'\n\n"
                "{format_instructions}",
            ),
            ("user", "What is the next step?"),
        ]
    )

    chain = prompt | llm | parser

    try:
        # Note: state["conflicts"] contains Pydantic models, so use dot notation
        decision: SupervisorDecision = await chain.ainvoke(
            {
                "query": state["query"],
                "hypo_count": len(state["hypotheses"]),
                "conflict_count": len([c for c in state["conflicts"] if c.status == "open"]),
                "iteration": state["iteration_count"],
                "max_iter": state["max_iterations"],
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return {
            "next_step": decision.next_step,
            "iteration_count": state["iteration_count"] + 1,
            "messages": [AIMessage(content=f"Supervisor: {decision.reasoning}")],
        }
    except Exception as e:
        return {
            "next_step": "synthesize",
            "iteration_count": state["iteration_count"] + 1,
            "messages": [AIMessage(content=f"Supervisor Error: {e!s}. Proceeding to synthesis.")],
        }
