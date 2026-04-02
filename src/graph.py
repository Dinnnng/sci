"""LangGraph workflow – the core graph that orchestrates the topic selection pipeline."""

from __future__ import annotations

import functools
import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from src.state import AgentState
from src.nodes.expand import expand
from src.nodes.retrieve import retrieve
from src.nodes.organize import organize
from src.nodes.compare import compare
from src.nodes.converge import converge

load_dotenv()


def _should_retry(state: AgentState) -> str:
    """Conditional edge: check if any suggestion lacks evidence and needs more retrieval."""
    suggestions = state.get("suggestions", [])
    retry_count = state.get("retry_count", 0)

    if retry_count >= 2:
        return END

    # Check if any top suggestion has fewer than 2 evidence papers
    for s in suggestions:
        gap_evidence = s.get("gap_evidence", [])
        if len(gap_evidence) < 2:
            return "retrieve"

    return END


def _make_llm(model_name: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance, respecting env vars for Qwen/other providers."""
    model = model_name or os.getenv("LLM_MODEL", "qwen3.5-flash")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    kwargs = {
        "model": model,
        "temperature": 0.3,
        "extra_body": {"enable_thinking": False},
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_graph(model_name: str | None = None) -> StateGraph:
    """Build and compile the LangGraph workflow."""
    llm = _make_llm(model_name)

    # Bind LLM to nodes that need it
    expand_node = functools.partial(expand, llm=llm)
    organize_node = functools.partial(organize, llm=llm)
    compare_node = functools.partial(compare, llm=llm)
    converge_node = functools.partial(converge, llm=llm)

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("expand", expand_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("organize", organize_node)
    graph.add_node("compare", compare_node)
    graph.add_node("converge", converge_node)

    # Linear flow: expand → retrieve → organize → compare → converge
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "retrieve")
    graph.add_edge("retrieve", "organize")
    graph.add_edge("organize", "compare")
    graph.add_edge("compare", "converge")

    # Conditional: converge → END or back to retrieve for more evidence
    graph.add_conditional_edges("converge", _should_retry)

    # Compile with in-memory checkpointing
    app = graph.compile(checkpointer=MemorySaver())
    return app


async def run_topic_agent(
    interest: str,
    constraints: str = "",
    depth: str = "quick",
    model_name: str | None = None,
) -> dict:
    """Run the full topic selection pipeline.

    Args:
        interest: The user's research interest / question.
        constraints: Optional constraints (e.g. "prefer empirical studies").
        depth: "quick" (~30 papers) or "deep" (~100 papers).
        model_name: LLM model to use (defaults to LLM_MODEL env var).

    Returns:
        The final AgentState dict containing suggestions, landscape, etc.
    """
    app = build_graph(model_name=model_name)

    initial_state: AgentState = {
        "interest": interest,
        "constraints": constraints,
        "depth": depth,
        "search_queries": [],
        "papers": [],
        "landscape": {},
        "candidates": [],
        "suggestions": [],
        "comparison_table": "",
        "methodology_note": "",
        "retry_count": 0,
        "status": "running",
        "error": "",
    }

    config = {"configurable": {"thread_id": "default"}}
    result = await app.ainvoke(initial_state, config=config)
    return result
