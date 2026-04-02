"""State definitions and data models for the topic selection agent."""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models – used as structured output from LLM calls and as the
# final response presented to the user.
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    """A single academic paper with essential metadata."""
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    cited_by_count: int = 0
    abstract: str = ""
    url: str = ""
    source: Literal["openalex", "arxiv", "semantic_scholar"] = "openalex"
    openalex_id: str = ""

    @property
    def credibility(self) -> Literal["high", "medium", "low"]:
        if self.source == "arxiv":
            if self.cited_by_count >= 10:
                return "medium"
            return "low"
        return "high"


class Citation(BaseModel):
    """A reference to a paper used as evidence for a claim."""
    paper: Paper
    relevance: str = ""  # why this paper is cited here


class SubField(BaseModel):
    """A cluster / sub-area identified during landscape analysis."""
    name: str
    description: str
    paper_count: int = 0
    trend: Literal["rising", "stable", "declining", "emerging"] = "stable"
    key_papers: list[Paper] = Field(default_factory=list)


class LandscapeMap(BaseModel):
    """High-level landscape of the research area."""
    summary: str
    subfields: list[SubField] = Field(default_factory=list)
    total_papers_analyzed: int = 0


class TopicSuggestion(BaseModel):
    """A single candidate research topic."""
    title: str
    one_liner: str
    gap_evidence: list[Citation] = Field(default_factory=list)
    related_work: list[Citation] = Field(default_factory=list)
    trend: Literal["rising", "stable", "declining", "emerging"] = "emerging"
    feasibility: str = ""
    confidence: float = Field(ge=0, le=1, default=0.5)
    reasoning_chain: str = ""


class FinalOutput(BaseModel):
    """The complete output handed back to the user."""
    landscape: LandscapeMap
    suggestions: list[TopicSuggestion] = Field(default_factory=list)
    comparison_table: str = ""
    methodology_note: str = ""


# ---------------------------------------------------------------------------
# LangGraph State – the single mutable dict that flows through every node.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # --- user input ---
    interest: str                               # raw user interest
    constraints: str                            # optional constraints
    depth: str                                  # "quick" or "deep"

    # --- expand node ---
    search_queries: list[str]                   # generated search queries

    # --- retrieve node ---
    papers: Annotated[list[dict], add]          # accumulated papers (dicts for serializability)

    # --- organize node ---
    landscape: dict                             # LandscapeMap as dict

    # --- compare node ---
    candidates: list[dict]                      # list of TopicSuggestion dicts

    # --- converge node ---
    suggestions: list[dict]                     # final top-3 suggestions
    comparison_table: str
    methodology_note: str

    # --- control flow ---
    retry_count: int                            # how many times we retried retrieve
    status: str                                 # "running" | "needs_confirmation" | "done"
    error: str                                  # last error message, if any
