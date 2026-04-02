"""arXiv API client – supplementary source for recent preprints."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

import httpx

_BASE = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}
_TIMEOUT = 30.0
_MAX_RETRIES = 3


async def search_papers(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """Search arXiv for preprints matching *query*.

    Returns a list of dicts compatible with the Paper model.
    Returns empty list on persistent failure (graceful degradation).
    """
    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(_BASE, params=params)
                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                break
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError):
            if attempt == _MAX_RETRIES - 1:
                return []  # graceful degradation
            await asyncio.sleep(3)
    else:
        return []

    root = ET.fromstring(resp.text)
    results: list[dict] = []

    for entry in root.findall("a:entry", _NS):
        title_el = entry.find("a:title", _NS)
        summary_el = entry.find("a:summary", _NS)
        published_el = entry.find("a:published", _NS)
        id_el = entry.find("a:id", _NS)

        authors = [
            name_el.text
            for author in entry.findall("a:author", _NS)
            if (name_el := author.find("a:name", _NS)) is not None
            and name_el.text
        ]

        year = None
        if published_el is not None and published_el.text:
            year = int(published_el.text[:4])

        results.append({
            "title": (title_el.text or "").strip().replace("\n", " "),
            "authors": authors[:5],
            "year": year,
            "venue": "arXiv",
            "cited_by_count": 0,  # arXiv doesn't provide citation counts
            "abstract": (summary_el.text or "").strip().replace("\n", " "),
            "url": (id_el.text or "").strip(),
            "source": "arxiv",
            "openalex_id": "",
        })

    return results
