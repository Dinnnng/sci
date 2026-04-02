# Sci Topic Agent

一个基于 LangGraph 和 Streamlit 的科研选题助手。输入研究兴趣与约束后，应用会自动完成关键词扩展、论文检索、研究全景梳理、候选方向比较，并给出 Top 3 选题建议。

## Features

- 从模糊研究兴趣出发，生成可执行的选题方向
- 集成 OpenAlex 与 arXiv 检索
- 展示研究全景、趋势、相关工作与 gap 证据
- 提供可直接运行的 Streamlit 界面

## Requirements

- Python 3.11+

## Installation

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果需要开发依赖：

```bash
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Environment

先参考 `.env.example` 创建本地环境变量文件：

```bash
cp .env.example .env
```

需要配置的核心变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `LLM_MODEL`
- `OPENALEX_EMAIL`（推荐）

## Run

```bash
streamlit run app.py
```

默认会启动一个本地 Web 页面。

## Test

```bash
pytest
```

## Project Structure

```text
.
├── app.py
├── src/
│   ├── graph.py
│   ├── nodes/
│   ├── prompts/
│   ├── state.py
│   └── tools/
├── tests/
├── pyproject.toml
└── .env.example
```

## Notes

- 本仓库只保留运行项目所需代码与最小文档
- 不包含本地私有配置、Claude 相关目录和设计文档
