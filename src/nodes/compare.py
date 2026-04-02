"""Node: compare – generate candidate research topics."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState

_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "compare.txt").read_text()


def _papers_to_text(papers: list[dict], limit: int = 60) -> str:
    lines = []
    for i, p in enumerate(papers[:limit]):
        abstract = (p.get("abstract", "") or "")[:200]
        lines.append(
            f"[{i}] {p.get('title', 'Untitled')} ({p.get('year', '?')}) "
            f"cited: {p.get('cited_by_count', 0)}\n    {abstract}"
        )
    return "\n\n".join(lines)


async def compare(state: AgentState, *, llm) -> dict:
    """Generate 5 candidate topics from the landscape + papers."""
    papers = state.get("papers", [])
    landscape = state.get("landscape", {})

    prompt = _PROMPT_TEMPLATE.format(
        landscape=json.dumps(landscape, ensure_ascii=False, indent=2),
        interest=state["interest"],
        constraints=state.get("constraints", ""),
        papers_text=_papers_to_text(papers),
    )

    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the candidate research topics now."),
    ])

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    candidates_raw = json.loads(text)

    # Resolve paper indices to actual paper references
    candidates = []
    for c in candidates_raw:
        gap_indices = c.pop("gap_evidence_indices", [])
        related_indices = c.pop("related_work_indices", [])
        c["gap_evidence"] = [
            {"paper": papers[i], "relevance": "supports gap claim"}
            for i in gap_indices if i < len(papers)
        ]
        c["related_work"] = [
            {"paper": papers[i], "relevance": "related work"}
            for i in related_indices if i < len(papers)
        ]
        candidates.append(c)

    return {"candidates": candidates}
