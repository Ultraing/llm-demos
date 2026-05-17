# LLM 学习 Demo 集合

6 个循序渐进的 LLM 实战 Demo，从基础 API 调用到 Agent 开发，再到 API 服务部署。

## 项目简介

本项目通过一系列可运行的 Demo，帮助你从零掌握 LLM 应用开发的核心技能：

- **API 调用** — 理解 LLM 的基本调用方式和参数含义
- **对话管理** — 构建能记住上下文的聊天机器人
- **RAG** — 用外部知识库增强 LLM 的回答质量
- **Agent** — 让 LLM 调用工具、执行操作
- **Prompt Engineering** — 掌握提示词优化技巧
- **API 服务** — 用 FastAPI 将 LLM 能力封装为 HTTP 服务

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Demo 1: 基础 API 调用
python 01_basic_api.py

# 运行 Demo 2: 聊天机器人（自动演示）
python 02_chatbot.py

# 运行 Demo 2: 聊天机器人（交互模式）
python 02_chatbot.py --interactive

# 运行 Demo 3: RAG 检索增强生成
python 03_rag_demo.py

# 运行 Demo 4: Agent 工具调用
python 04_agent_demo.py

# 运行 Demo 5: Prompt Engineering
python 05_prompt_engineering.py

# 运行 Demo 6: FastAPI 服务（访问 http://127.0.0.1:8001/docs）
python 06_fastapi_demo.py
```

## Demo 列表

| Demo | 主题 | 核心知识点 |
|------|------|-----------|
| 01 | 基础 API 调用 | messages、system prompt、temperature、token |
| 02 | 聊天机器人 | 对话管理、上下文窗口、历史截断 |
| 03 | RAG 检索增强生成 | 文档切分、Embedding、向量检索、增强生成 |
| 04 | Agent 工具调用 | Function Calling、ReAct 循环、工具执行 |
| 05 | Prompt Engineering | Zero-shot、Few-shot、CoT、角色扮演 |
| 06 | FastAPI 服务 | Web 框架、请求校验、流式输出（SSE）、Swagger 文档 |

## 学习路线

建议按顺序学习：01 → 02 → 03 → 04 → 05 → 06

- **01-02**: LLM 基础（API 调用和对话管理）
- **03-04**: LLM 应用（RAG 和 Agent）
- **05**: Prompt 优化（提升输出质量）
- **06**: 服务部署（将 LLM 能力封装为 API 服务）
