"""OpenAlex API client – primary data source for paper search."""

from __future__ import annotations

import os
import httpx

_BASE = "https://api.openalex.org"
_TIMEOUT = 30.0


def _email_param() -> dict:
    email = os.getenv("OPENALEX_EMAIL", "")
    return {"mailto": email} if email else {}


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


async def search_papers(
    query: str,
    per_page: int = 25,
    sort: str = "relevance_score:desc",
) -> list[dict]:
    """Search OpenAlex for papers matching *query*.

    Returns a list of dicts with keys:
        title, authors, year, venue, cited_by_count, abstract, url,
        source ("openalex"), openalex_id
    """
    params = {
        "search": query,
        "per_page": per_page,
        "sort": sort,
        "select": "id,title,authorships,publication_year,cited_by_count,"
                  "primary_location,abstract_inverted_index",
        **_email_param(),
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/works", params=params)
        resp.raise_for_status()
        data = resp.json()

    results: list[dict] = []
    for work in data.get("results", []):
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])[:5]
        ]
        venue = ""
        loc = work.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name", "")

        results.append({
            "title": work.get("title", ""),
            "authors": authors,
            "year": work.get("publication_year"),
            "venue": venue,
            "cited_by_count": work.get("cited_by_count", 0),
            "abstract": _reconstruct_abstract(
                work.get("abstract_inverted_index")
            ),
            "url": work.get("id", ""),
            "source": "openalex",
            "openalex_id": work.get("id", ""),
        })
    return results


async def get_related_works(openalex_id: str, limit: int = 10) -> list[dict]:
    """Fetch works related to a given OpenAlex work ID."""
    params = {
        "filter": f"related_to:{openalex_id}",
        "per_page": limit,
        "sort": "cited_by_count:desc",
        "select": "id,title,publication_year,cited_by_count",
        **_email_param(),
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/works", params=params)
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "title": w.get("title", ""),
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
            "url": w.get("id", ""),
            "source": "openalex",
            "openalex_id": w.get("id", ""),
        }
        for w in data.get("results", [])
    ]
