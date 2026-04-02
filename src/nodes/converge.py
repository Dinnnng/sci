"""Node: converge – select top 3 topics, verify evidence, assign confidence."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState

_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "converge.txt").read_text()


def _papers_to_text(papers: list[dict], limit: int = 60) -> str:
    lines = []
    for i, p in enumerate(papers[:limit]):
        lines.append(
            f"[{i}] {p.get('title', 'Untitled')} ({p.get('year', '?')}) "
            f"cited: {p.get('cited_by_count', 0)}"
        )
    return "\n".join(lines)


async def converge(state: AgentState, *, llm) -> dict:
    """Converge to top-3 topics with verified evidence chains."""
    papers = state.get("papers", [])
    candidates = state.get("candidates", [])

    prompt = _PROMPT_TEMPLATE.format(
        candidates=json.dumps(candidates, ensure_ascii=False, indent=2, default=str),
        papers_text=_papers_to_text(papers),
    )

    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Select the top 3 and produce the final comparison."),
    ])

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(text)

    # Build final suggestions from selected candidates
    top_indices = result.get("top_3_indices", [0, 1, 2])
    confidence_scores = result.get("confidence_scores", [0.5, 0.5, 0.5])

    suggestions = []
    for rank, idx in enumerate(top_indices):
        if idx < len(candidates):
            candidate = dict(candidates[idx])
            candidate["confidence"] = (
                confidence_scores[rank] if rank < len(confidence_scores) else 0.5
            )
            suggestions.append(candidate)

    queries_used = state.get("search_queries", [])
    methodology = result.get("methodology_note", "")
    if not methodology:
        methodology = (
            f"Retrieved {len(papers)} papers using {len(queries_used)} search queries "
            f"across OpenAlex and arXiv. Queries: {', '.join(queries_used[:5])}"
        )

    return {
        "suggestions": suggestions,
        "comparison_table": result.get("comparison_table", ""),
        "methodology_note": methodology,
        "status": "done",
    }
