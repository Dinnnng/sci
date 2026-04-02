"""Node: expand – turn user interest into search queries."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState

_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "expand.txt").read_text()


async def expand(state: AgentState, *, llm) -> dict:
    """Generate search queries from the user's research interest."""
    prompt = _PROMPT_TEMPLATE.format(
        interest=state["interest"],
        constraints=state.get("constraints", ""),
    )
    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the search queries now."),
    ])

    # Parse the JSON array from the response
    text = response.content.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    queries = json.loads(text)
    if not isinstance(queries, list):
        queries = [state["interest"]]

    return {"search_queries": queries}
