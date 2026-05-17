# LLM 学习 Demo 集合

5 个循序渐进的 LLM 实战 Demo，从基础 API 调用到 Agent 开发。

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
```

## Demo 列表

| Demo | 主题 | 核心知识点 |
|------|------|-----------|
| 01 | 基础 API 调用 | messages、system prompt、temperature、token |
| 02 | 聊天机器人 | 对话管理、上下文窗口、历史截断 |
| 03 | RAG 检索增强生成 | 文档切分、Embedding、向量检索、增强生成 |
| 04 | Agent 工具调用 | Function Calling、ReAct 循环、工具执行 |
| 05 | Prompt Engineering | Zero-shot、Few-shot、CoT、角色扮演 |

## 学习路线

建议按顺序学习：01 → 02 → 03 → 04 → 05

- **01-02**: LLM 基础（API 调用和对话管理）
- **03-04**: LLM 应用（RAG 和 Agent）
- **05**: Prompt 优化（提升输出质量）
