"""
Demo 1: LLM 基础 API 调用
=========================
学习目标:
  - 理解 LLM API 的基本调用流程
  - 了解 messages、system prompt、temperature 等核心概念
  - 学会解析 API 响应

核心概念:
  1. Messages（消息）: LLM 的输入是一组消息，包含 role（角色）和 content（内容）
     - system: 系统提示词，设定 AI 的行为规则
     - user: 用户输入
     - assistant: AI 的回复
  2. Temperature（温度）: 控制输出的随机性，0=确定性，1=高随机性
  3. Max Tokens（最大 token 数）: 控制输出长度，1 token ≈ 1.5 个中文字符
"""

import sys
import io
# Windows 终端中文编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

# ============================================================
# 第一部分: 最简单的 API 调用
# ============================================================

# API 配置
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"
MODEL = "mimo-v2.5-pro"


def basic_call(user_message: str) -> str:
    """
    最基础的 LLM 调用：发送一条消息，获取回复。

    流程:
      用户输入 → 构建请求 → 发送 API → 解析响应 → 返回结果

    参数:
        user_message: 用户输入的文本

    返回:
        LLM 的回复文本
    """
    # 构建请求头（认证信息）
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # 构建请求体
    # messages 是一个列表，每个元素包含 role 和 content
    body = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": user_message}  # 用户消息
        ],
    }

    # 发送 POST 请求
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )

    # 解析响应
    data = response.json()

    # Anthropic 响应格式:
    # {
    #   "content": [{"type": "text", "text": "回复内容"}],
    #   "model": "模型名",
    #   "usage": {"input_tokens": 10, "output_tokens": 50}
    # }
    return data["content"][0]["text"]


# ============================================================
# 第二部分: 带系统提示词的调用
# ============================================================

def call_with_system_prompt(user_message: str, system_prompt: str) -> str:
    """
    带系统提示词的调用：通过 system prompt 控制 AI 的行为。

    系统提示词的作用:
      - 设定 AI 的角色（如：你是一个 Python 专家）
      - 定义输出格式（如：用 JSON 格式回复）
      - 设定行为规则（如：只用中文回复）

    参数:
        user_message:  用户输入
        system_prompt: 系统提示词
    """
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": system_prompt,  # 系统提示词：在消息之外单独设置
        "messages": [
            {"role": "user", "content": user_message}
        ],
    }

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ============================================================
# 第三部分: 多轮对话
# ============================================================

def multi_turn_chat(messages: list) -> str:
    """
    多轮对话：通过 messages 列表维护上下文。

    关键点:
      - LLM 本身是无状态的，每次调用都是独立的
      - 我们通过把历史消息都发过去来模拟"记忆"
      - 消息必须交替出现: user → assistant → user → assistant ...

    参数:
        messages: 完整的对话历史
    """
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": "你是一个友好的 AI 助手，用中文回复。",
        "messages": messages,  # 传入完整对话历史
    }

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ============================================================
# 第四部分: 解析 Token 使用量（成本控制）
# ============================================================

def get_usage_info(user_message: str) -> dict:
    """
    获取 API 调用的 token 使用量信息。

    为什么要关注 token？
      - API 按 token 计费
      - 输入 token + 输出 token = 总 token
      - 控制 token 用量是成本优化的关键
    """
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": user_message}
        ],
    }

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    # 返回结构化的使用量信息
    return {
        "reply": data["content"][0]["text"],
        "model": data["model"],
        "input_tokens": data["usage"]["input_tokens"],   # 输入消耗的 token
        "output_tokens": data["usage"]["output_tokens"], # 输出消耗的 token
        "stop_reason": data["stop_reason"],               # 停止原因（end_turn/stop_sequence/max_tokens）
    }


# ============================================================
# 主程序：运行所有演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: LLM 基础 API 调用")
    print("=" * 60)

    # --- 演示 1: 最简单的调用 ---
    print("\n[演示 1] 最简单的 API 调用")
    print("-" * 40)
    reply = basic_call("用一句话解释什么是大语言模型")
    print(f"问题: 用一句话解释什么是大语言模型")
    print(f"回答: {reply}")

    # --- 演示 2: 系统提示词的作用 ---
    print("\n[演示 2] 系统提示词控制 AI 行为")
    print("-" * 40)

    # 同一个问题，不同的系统提示词，得到不同的回答风格
    question = "解释 Python 的装饰器"

    # 情况 A: 面向初学者
    system_a = "你是一个耐心的编程老师，用简单的比喻和例子向初学者解释概念。"
    reply_a = call_with_system_prompt(question, system_a)
    print(f"[面向初学者]\n{reply_a}\n")

    # 情况 B: 面向专家
    system_b = "你是一个资深 Python 开发者，用简洁的技术语言回答，直接给出代码示例。"
    reply_b = call_with_system_prompt(question, system_b)
    print(f"[面向专家]\n{reply_b}\n")

    # --- 演示 3: 多轮对话 ---
    print("\n[演示 3] 多轮对话（模拟上下文记忆）")
    print("-" * 40)

    # 构建多轮对话历史
    conversation = [
        {"role": "user", "content": "我叫小明，请记住我的名字"},
    ]

    # 第一轮
    reply1 = multi_turn_chat(conversation)
    print(f"用户: {conversation[0]['content']}")
    print(f"AI: {reply1}")

    # 把 AI 的回复加入历史，再问新问题
    conversation.append({"role": "assistant", "content": reply1})
    conversation.append({"role": "user", "content": "我的名字是什么？"})

    # 第二轮（AI 应该能记住名字）
    reply2 = multi_turn_chat(conversation)
    print(f"\n用户: {conversation[2]['content']}")
    print(f"AI: {reply2}")

    # --- 演示 4: Token 使用量 ---
    print("\n[演示 4] Token 使用量分析")
    print("-" * 40)

    usage = get_usage_info("写一首关于春天的五言绝句")
    print(f"回答: {usage['reply']}")
    print(f"模型: {usage['model']}")
    print(f"输入 token: {usage['input_tokens']}")
    print(f"输出 token: {usage['output_tokens']}")
    print(f"总 token: {usage['input_tokens'] + usage['output_tokens']}")
    print(f"停止原因: {usage['stop_reason']}")

    print("\n" + "=" * 60)
    print("Demo 1 完成！你已掌握 LLM API 的基本调用方法。")
    print("=" * 60)
