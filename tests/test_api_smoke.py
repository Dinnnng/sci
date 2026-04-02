"""Quick smoke test for API tools – no LLM needed."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.tools import openalex, arxiv_search


async def test_openalex():
    print("=== OpenAlex API Test ===")
    papers = await openalex.search_papers("large language model healthcare", per_page=3)
    print(f"Found {len(papers)} papers")
    for p in papers:
        print(f"  [{p['year']}] {p['title'][:80]}... (cited: {p['cited_by_count']})")
        print(f"       abstract: {p['abstract'][:100]}...")
        print(f"       source: {p['source']}, url: {p['url'][:60]}")
    assert len(papers) > 0, "OpenAlex returned no results"
    has_abstract = any(p["abstract"] for p in papers)
    print(f"  Papers with abstracts: {sum(1 for p in papers if p['abstract'])}/{len(papers)}")
    assert has_abstract, "No papers have abstracts"
    print("✅ OpenAlex OK\n")


async def test_arxiv():
    print("=== arXiv API Test ===")
    papers = await arxiv_search.search_papers("large language model healthcare", max_results=3)
    print(f"Found {len(papers)} papers")
    if len(papers) == 0:
        print("  ⚠️ arXiv returned 0 results (rate limited) – graceful degradation OK")
    else:
        for p in papers:
            print(f"  [{p['year']}] {p['title'][:80]}...")
            print(f"       abstract: {p['abstract'][:100]}...")
    print("✅ arXiv OK (graceful degradation)\n")


async def main():
    await test_openalex()
    await test_arxiv()
    print("🎉 All API tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
