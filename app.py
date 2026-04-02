"""Streamlit app for the research topic selection agent."""

import asyncio
import json
import streamlit as st

from src.graph import run_topic_agent


st.set_page_config(page_title="科研选题 Agent", page_icon="🔬", layout="wide")

st.title("科研选题 Agent")
st.caption("从模糊兴趣到具体选题方向 — 基于 LangGraph 的智能调研助手")

# --- Sidebar: inputs ---
with st.sidebar:
    st.header("输入你的研究兴趣")
    interest = st.text_area(
        "研究兴趣 / 初步想法",
        placeholder="例如：LLM在医疗领域的应用、强化学习与机器人控制...",
        height=100,
    )
    constraints = st.text_input(
        "约束条件（可选）",
        placeholder="例如：偏好实证研究、排除纯综述...",
    )
    depth = st.radio("检索深度", ["quick", "deep"], index=0,
                     format_func=lambda x: "快速（~30篇）" if x == "quick" else "深度（~100篇）")
    model = st.selectbox("LLM 模型", ["qwen3.5-flash"], index=0)
    run_btn = st.button("开始调研", type="primary", use_container_width=True)


def render_paper(paper: dict, prefix: str = "") -> str:
    """Format a paper as a markdown string."""
    title = paper.get("title", "Untitled")
    year = paper.get("year", "?")
    cited = paper.get("cited_by_count", 0)
    venue = paper.get("venue", "")
    url = paper.get("url", "")
    source = paper.get("source", "")
    link = f"[{title}]({url})" if url else title
    return f"{prefix}{link} ({year}) — cited: {cited} | {venue} | `{source}`"


# --- Main area ---
if run_btn and interest:
    with st.status("正在调研...", expanded=True) as status:
        st.write("🔍 扩展搜索关键词...")

        # Run the async pipeline
        result = asyncio.run(
            run_topic_agent(
                interest=interest,
                constraints=constraints,
                depth=depth,
                model_name=model,
            )
        )

        status.update(label="调研完成!", state="complete", expanded=False)

    # --- Display results ---

    # 1. Methodology note
    st.info(result.get("methodology_note", ""))

    # 2. Landscape
    st.header("📊 研究全景图")
    landscape = result.get("landscape", {})
    st.markdown(landscape.get("summary", ""))

    subfields = landscape.get("subfields", [])
    if subfields:
        cols = st.columns(min(len(subfields), 3))
        for i, sf in enumerate(subfields):
            with cols[i % len(cols)]:
                trend_emoji = {
                    "rising": "📈", "stable": "➡️",
                    "declining": "📉", "emerging": "🌱",
                }.get(sf.get("trend", ""), "")
                st.metric(
                    label=sf.get("name", ""),
                    value=f"{sf.get('paper_count', 0)} 篇",
                    delta=f"{trend_emoji} {sf.get('trend', '')}",
                )
                st.caption(sf.get("description", ""))

    # 3. Top-3 suggestions
    st.header("🎯 推荐选题方向（Top 3）")

    suggestions = result.get("suggestions", [])
    for i, s in enumerate(suggestions):
        confidence = s.get("confidence", 0.5)
        conf_bar = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"

        with st.expander(
            f"**{i+1}. {s.get('title', '')}** — 置信度 {conf_bar} {confidence:.0%}",
            expanded=(i == 0),
        ):
            st.markdown(f"**一句话**: {s.get('one_liner', '')}")
            st.markdown(f"**趋势**: {s.get('trend', '')} | **可行性**: {s.get('feasibility', '')}")

            st.markdown("**推理链**:")
            st.markdown(f"> {s.get('reasoning_chain', '')}")

            # Gap evidence
            gap_evidence = s.get("gap_evidence", [])
            if gap_evidence:
                st.markdown("**Gap 证据论文**:")
                for cite in gap_evidence:
                    paper = cite.get("paper", cite) if isinstance(cite, dict) else cite
                    st.markdown(render_paper(paper, prefix="- "))

            # Related work
            related = s.get("related_work", [])
            if related:
                st.markdown("**相关工作**:")
                for cite in related:
                    paper = cite.get("paper", cite) if isinstance(cite, dict) else cite
                    st.markdown(render_paper(paper, prefix="- "))

    # 4. Comparison table
    st.header("📋 对比表")
    table = result.get("comparison_table", "")
    if table:
        st.markdown(table)

    # 5. Raw data (collapsible)
    with st.expander("查看原始数据（调试用）"):
        st.json(result, expanded=False)

elif run_btn:
    st.warning("请先输入你的研究兴趣。")

else:
    st.markdown("""
    ### 使用方法
    1. 在左侧输入你的研究兴趣
    2. 点击"开始调研"
    3. 系统将自动完成：**关键词扩展 → 论文检索 → 全景梳理 → 候选比较 → 收敛推荐**
    4. 查看 Top-3 推荐选题，验证证据链

    ### 数据来源
    - **OpenAlex**: 4.7亿篇论文索引，免费开放 API
    - **arXiv**: STEM 领域最新预印本
    """)
