"""Node: retrieve – fetch papers from OpenAlex + arXiv."""

from __future__ import annotations

import asyncio

from src.state import AgentState
from src.tools import openalex, arxiv_search


def _dedup_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by title similarity (lowercased)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for p in papers:
        key = p.get("title", "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


async def retrieve(state: AgentState, **_kwargs) -> dict:
    """Fetch papers for all search queries from OpenAlex + arXiv."""
    queries = state.get("search_queries", [])
    depth = state.get("depth", "quick")
    per_query = 10 if depth == "quick" else 25

    # Fan out: search OpenAlex and arXiv concurrently for all queries
    tasks = []
    for q in queries:
        tasks.append(openalex.search_papers(q, per_page=per_query))
        tasks.append(arxiv_search.search_papers(q, max_results=max(3, per_query // 3)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: list[dict] = []
    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)
        # silently skip errors – partial results are fine

    # Dedup and sort by citations
    unique = _dedup_papers(all_papers)
    unique.sort(key=lambda p: p.get("cited_by_count", 0), reverse=True)

    return {"papers": unique}
