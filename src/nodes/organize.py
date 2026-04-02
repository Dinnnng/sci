"""Node: organize – cluster papers into a landscape map."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState

_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "organize.txt").read_text()


def _papers_to_text(papers: list[dict], limit: int = 60) -> str:
    """Format papers into a numbered text block for the LLM prompt."""
    lines = []
    for i, p in enumerate(papers[:limit]):
        abstract = (p.get("abstract", "") or "")[:300]
        lines.append(
            f"[{i}] {p.get('title', 'Untitled')} ({p.get('year', '?')}) "
            f"– cited: {p.get('cited_by_count', 0)} – {p.get('venue', '')}\n"
            f"    {abstract}"
        )
    return "\n\n".join(lines)


async def organize(state: AgentState, *, llm) -> dict:
    """Analyze papers and produce a landscape map."""
    papers = state.get("papers", [])
    papers_text = _papers_to_text(papers)

    prompt = _PROMPT_TEMPLATE.format(papers_text=papers_text)
    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Analyze these papers and produce the landscape map."),
    ])

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    landscape = json.loads(text)
    landscape["total_papers_analyzed"] = len(papers)

    # Attach key papers to subfields
    for sf in landscape.get("subfields", []):
        key_indices = sf.pop("key_paper_indices", [])
        sf["key_papers"] = [
            papers[i] for i in key_indices
            if i < len(papers)
        ]

    return {"landscape": landscape}
