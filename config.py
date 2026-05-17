"""
LLM Demo 共享配置文件
====================
所有 demo 共用的 API 配置和工具函数。
使用小米 Mimo API（兼容 Anthropic 格式）。
"""

# ============================================================
# API 配置
# ============================================================
# Mimo API 的基础地址（兼容 Anthropic 格式）
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"

# API 密钥（认证身份用）
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"

# 默认使用的模型名称
MODEL = "mimo-v2.5-pro"

# 生成参数：控制 LLM 输出的随机性
# temperature 越高（如 1.0），输出越随机、越有创意
# temperature 越低（如 0.0），输出越确定、越保守
TEMPERATURE = 0.7

# 最大输出 token 数（1 个 token ≈ 1.5 个中文字符）
MAX_TOKENS = 2048


def call_mimo_api(messages: list, system: str = "", tools: list = None) -> dict:
    """
    调用 Mimo API（兼容 Anthropic 格式）的核心函数。

    参数:
        messages: 对话消息列表，格式为 [{"role": "user/assistant", "content": "..."}]
        system:   系统提示词（可选），用于设定 AI 的行为和角色
        tools:    工具定义列表（可选），用于 Agent 的 function calling

    返回:
        API 的完整响应（dict 格式）

    Anthropic API 的请求格式:
    {
        "model": "模型名",
        "max_tokens": 最大输出长度,
        "system": "系统提示词",
        "messages": [{"role": "user", "content": "用户输入"}]
    }
    """
    import requests

    # 构建请求头：包含认证信息和内容类型
    headers = {
        "x-api-key": API_KEY,                    # Anthropic 格式的 API key 认证
        "anthropic-version": "2023-06-01",       # API 版本号
        "content-type": "application/json",      # 请求体格式
    }

    # 构建请求体
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": messages,
    }

    # 如果有系统提示词，加入请求体
    if system:
        body["system"] = system

    # 如果有工具定义（Agent 场景），加入请求体
    if tools:
        body["tools"] = tools

    # 发送 POST 请求到 API
    response = requests.post(
        f"{BASE_URL}/v1/messages",  # Anthropic 的消息端点
        headers=headers,
        json=body,
        timeout=60,  # 超时时间 60 秒
    )

    # 检查 HTTP 状态码，非 2xx 会抛出异常
    response.raise_for_status()

    return response.json()


def print_stream(text: str, end: str = "\n"):
    """模拟流式输出效果，逐字符打印（教学演示用）"""
    import sys
    import time
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.02)  # 每个字符间隔 20ms
    sys.stdout.write(end)


def format_messages_display(messages: list) -> str:
    """格式化显示对话历史（调试用）"""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # 截断过长的内容
        if len(content) > 100:
            content = content[:100] + "..."
        result.append(f"  [{role}]: {content}")
    return "\n".join(result)
