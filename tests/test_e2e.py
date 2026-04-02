"""End-to-end test of the full LangGraph pipeline.

Requires OPENAI_API_KEY to be set.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.graph import run_topic_agent


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping e2e test.")
        print("   Set it and rerun: export OPENAI_API_KEY=sk-...")
        return

    print("🚀 Running end-to-end test: 'LLM在医疗领域的应用'\n")

    result = await run_topic_agent(
        interest="LLM在医疗领域的应用",
        constraints="偏好实证研究",
        depth="quick",
    )

    # --- Validate output ---
    print(f"Status: {result.get('status')}")
    print(f"Search queries: {len(result.get('search_queries', []))}")
    print(f"Papers retrieved: {len(result.get('papers', []))}")

    landscape = result.get("landscape", {})
    print(f"Landscape subfields: {len(landscape.get('subfields', []))}")
    print(f"Landscape summary: {landscape.get('summary', '')[:200]}...")

    suggestions = result.get("suggestions", [])
    print(f"\n🎯 Suggestions: {len(suggestions)}")
    for i, s in enumerate(suggestions):
        print(f"\n--- Suggestion {i+1} ---")
        print(f"  Title: {s.get('title', '')}")
        print(f"  One-liner: {s.get('one_liner', '')}")
        print(f"  Trend: {s.get('trend', '')}")
        print(f"  Feasibility: {s.get('feasibility', '')}")
        print(f"  Confidence: {s.get('confidence', 'N/A')}")
        gap = s.get("gap_evidence", [])
        print(f"  Gap evidence papers: {len(gap)}")
        for g in gap[:2]:
            paper = g.get("paper", g) if isinstance(g, dict) else g
            print(f"    - {paper.get('title', '')[:60]}... ({paper.get('year', '?')})")

    print(f"\n📋 Comparison table:\n{result.get('comparison_table', 'N/A')}")
    print(f"\n📝 Methodology: {result.get('methodology_note', '')[:300]}")

    # Basic assertions
    assert result.get("status") == "done", f"Expected status 'done', got '{result.get('status')}'"
    assert len(suggestions) >= 1, "Expected at least 1 suggestion"
    assert len(result.get("papers", [])) > 0, "Expected papers"
    assert landscape.get("summary"), "Expected landscape summary"

    print("\n✅ End-to-end test PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
