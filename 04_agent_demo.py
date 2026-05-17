"""
Demo 4: 简单 AI Agent（工具调用）
=================================
学习目标:
  - 理解 Agent 的核心原理：感知 → 思考 → 行动 → 观察
  - 学会使用 Function Calling（工具调用）
  - 实现一个能使用工具的 AI Agent

Agent vs 普通 LLM:
  - 普通 LLM: 用户提问 → AI 回答（一问一答）
  - Agent:    用户提问 → AI 思考 → 调用工具 → 观察结果 → 继续思考 → ... → 最终回答

Agent 的核心循环（ReAct 模式）:
  1. Thought（思考）: AI 分析问题，决定下一步行动
  2. Action（行动）:  AI 调用一个工具
  3. Observation（观察）: 获取工具返回的结果
  4. 重复 1-3 直到得出最终答案

Anthropic 的 Tool Use 流程:
  1. 定义工具: 告诉 LLM 有哪些工具可用（名称、描述、参数）
  2. LLM 决策: LLM 分析问题，决定是否调用工具
  3. 执行工具: 代码执行 LLM 请求的工具调用
  4. 返回结果: 把工具结果发回给 LLM
  5. 最终回答: LLM 基于工具结果生成最终回答
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import math
from datetime import datetime

# API 配置
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"
MODEL = "mimo-v2.5-pro"


# ============================================================
# 第一部分: 定义工具（Functions）
# ============================================================
# 工具就是普通的 Python 函数，关键是给 LLM 提供清晰的描述

def calculator(expression: str) -> str:
    """
    计算器工具：计算数学表达式。

    支持: 加(+)、减(-)、乘(*)、除(/)、幂(**)、三角函数等

    参数:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    # 安全限制：只允许数学相关的字符
    allowed = set("0123456789+-*/.() sin cos tan sqrt pi e ")
    if not all(c in allowed for c in expression.replace(" ", "")):
        return f"错误: 不支持的字符在表达式中"

    try:
        # 提供安全的数学函数
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
            "abs": abs, "pow": pow, "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def search_knowledge_base(query: str) -> str:
    """
    知识库搜索工具：从本地知识库中搜索信息。

    这是一个模拟工具，实际项目中会连接真实的数据库或搜索引擎。
    """
    # 模拟知识库
    knowledge = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "javascript": "JavaScript 是 Web 的编程语言，由 Brendan Eich 于 1995 年创建。",
        "rust": "Rust 是一种系统编程语言，由 Mozilla 研究院开发，注重安全性和性能。",
        "agent": "AI Agent 是能够自主感知环境、做出决策并执行行动的智能体。",
        "rag": "RAG（检索增强生成）是结合信息检索和文本生成的技术，用于减少 LLM 幻觉。",
    }

    query_lower = query.lower()
    results = []
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            results.append(value)

    if results:
        return "\n".join(results)
    return f"未找到与 '{query}' 相关的信息。"


# ============================================================
# 第二部分: 工具定义（给 LLM 看的描述）
# ============================================================
# Anthropic Tool Use 格式：用 JSON Schema 描述工具

TOOLS = [
    {
        "name": "calculator",
        "description": "计算数学表达式。支持加减乘除、幂运算、三角函数等。输入应为有效的数学表达式字符串。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "获取当前的日期和时间。不需要任何参数。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "从知识库中搜索信息。输入搜索关键词，返回相关知识。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    }
]

# 工具名称到函数的映射（执行时用）
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_knowledge_base": search_knowledge_base,
}


# ============================================================
# 第三部分: Agent 核心逻辑
# ============================================================

class Agent:
    """
    AI Agent：能思考并使用工具的智能体。

    核心循环:
      while 没有最终答案:
        1. 发送消息给 LLM（包含工具定义）
        2. LLM 返回: 要么直接回答，要么请求调用工具
        3. 如果请求调用工具 → 执行工具 → 把结果发回 LLM
        4. 如果直接回答 → 返回答案
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or """你是一个智能助手，可以使用以下工具来帮助回答问题：

1. calculator: 计算数学表达式
2. get_current_time: 获取当前时间
3. search_knowledge_base: 搜索知识库

使用工具的原则：
- 需要精确计算时使用 calculator
- 需要知道当前时间时使用 get_current_time
- 需要查找事实信息时使用 search_knowledge_base
- 如果不需要工具就能回答，直接回答即可"""
        self.max_iterations = 5  # 防止无限循环

    def run(self, user_message: str) -> str:
        """
        运行 Agent 处理用户请求。

        这是 Agent 的核心方法，实现了 "思考-行动-观察" 循环。

        参数:
            user_message: 用户输入

        返回:
            Agent 的最终回答
        """
        print(f"\n[Agent] 收到问题: {user_message}")
        print("-" * 50)

        # 初始消息列表
        messages = [{"role": "user", "content": user_message}]

        # Agent 循环（最多 max_iterations 次，防止死循环）
        for iteration in range(self.max_iterations):
            print(f"\n[迭代 {iteration + 1}] 调用 LLM...")

            # 调用 LLM（带工具定义）
            response = self._call_llm(messages)

            # 分析 LLM 的响应
            # Anthropic 响应中，content 可能包含多种类型的 block:
            # - {"type": "text", "text": "..."}: 文本回复
            # - {"type": "tool_use", "id": "...", "name": "...", "input": {...}}: 工具调用请求
            content_blocks = response["content"]

            # 检查是否包含工具调用
            tool_calls = [b for b in content_blocks if b["type"] == "tool_use"]
            text_blocks = [b for b in content_blocks if b["type"] == "text"]

            # 如果没有工具调用，说明 LLM 给出了最终回答
            if not tool_calls:
                final_text = " ".join([b["text"] for b in text_blocks])
                print(f"\n[Agent] 最终回答: {final_text}")
                return final_text

            # 有工具调用：打印思考过程
            if text_blocks:
                print(f"[思考] {text_blocks[0]['text']}")

            # 把 LLM 的回复加入消息列表
            messages.append({"role": "assistant", "content": content_blocks})

            # 执行每个工具调用
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_id = tool_call["id"]

                print(f"[行动] 调用工具: {tool_name}({tool_input})")

                # 执行工具
                result = self._execute_tool(tool_name, tool_input)
                print(f"[观察] 工具返回: {result}")

                # 构建工具结果消息（Anthropic 格式）
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })

            # 把工具结果加入消息列表
            messages.append({"role": "user", "content": tool_results})

        return "抱歉，经过多次尝试未能得出答案。"

    def _call_llm(self, messages: list) -> dict:
        """调用 LLM API（带工具定义）"""
        headers = {
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = {
            "model": MODEL,
            "max_tokens": 1024,
            "system": self.system_prompt,
            "messages": messages,
            "tools": TOOLS,  # 传入工具定义
        }

        response = requests.post(
            f"{BASE_URL}/v1/messages",
            headers=headers,
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        执行工具调用。

        这是 Agent 与外部世界交互的桥梁。
        """
        func = TOOL_FUNCTIONS.get(tool_name)
        if not func:
            return f"错误: 未知工具 '{tool_name}'"

        try:
            result = func(**tool_input)
            return result
        except Exception as e:
            return f"工具执行错误: {e}"


# ============================================================
# 主程序：运行 Agent 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 4: 简单 AI Agent（工具调用）")
    print("=" * 60)

    # 创建 Agent 实例
    agent = Agent()

    # 测试不同类型的请求
    test_questions = [
        "现在几点了？",                              # 需要调用时间工具
        "计算一下 (15 + 27) * 3 - 18 等于多少",       # 需要调用计算器工具
        "Python 是什么？",                            # 需要调用知识库工具
        "帮我算一下 sqrt(144) + 23，然后告诉我现在几点", # 需要调用多个工具
    ]

    for question in test_questions:
        answer = agent.run(question)
        print(f"\n{'='*60}")

    print("\n" + "=" * 60)
    print("Demo 4 完成！你已理解 Agent 的工具调用机制。")
    print("=" * 60)
