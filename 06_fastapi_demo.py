"""
Demo 6: FastAPI 构建 LLM API 服务
==================================
学习目标:
  - 用 FastAPI 将 LLM 封装为 HTTP API 服务
  - 理解请求/响应模型（Pydantic）
  - 实现普通调用和流式输出（SSE）两个端点

核心概念:
  1. FastAPI: Python 高性能 Web 框架，自动生成 API 文档
  2. Pydantic: 数据校验库，用于定义请求和响应的数据结构
  3. SSE (Server-Sent Events): 服务端向客户端推送流式数据的协议
     - ChatGPT 网页版打字机效果就是用的 SSE
     - 客户端逐 token 接收，而非等全部生成完

运行方式:
  python 06_fastapi_demo.py
  或
  uvicorn 06_fastapi_demo:app --reload

然后访问:
  - http://127.0.0.1:8001           # 欢迎页
  - http://127.0.0.1:8001/docs      # Swagger 交互式文档
  - http://127.0.0.1:8001/redoc     # ReDoc 文档
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import requests
from typing import Optional
from config import BASE_URL, API_KEY, MODEL, TEMPERATURE, MAX_TOKENS

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ============================================================
# 第一部分: 定义数据模型（Pydantic）
# ============================================================
# Pydantic 模型的作用：
#   - 自动校验请求数据（类型、必填、默认值）
#   - 自动生成 JSON Schema（用于 Swagger 文档）
#   - 自动序列化/反序列化 JSON


class ChatRequest(BaseModel):
    """聊天请求模型

    客户端发送的 JSON 格式:
    {
        "message": "你好",
        "system": "你是一个友好的助手",  // 可选
        "temperature": 0.7,              // 可选
        "max_tokens": 1024               // 可选
    }
    """
    message: str = Field(..., min_length=1, max_length=10000, description="用户输入的消息")
    system: Optional[str] = Field(None, description="系统提示词（可选）")
    temperature: Optional[float] = Field(None, ge=0, le=1, description="温度参数，0-1")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="最大输出 token 数")


class ChatResponse(BaseModel):
    """聊天响应模型

    服务端返回的 JSON 格式:
    {
        "reply": "你好！有什么可以帮助你的吗？",
        "model": "mimo-v2.5-pro",
        "input_tokens": 15,
        "output_tokens": 20
    }
    """
    reply: str = Field(..., description="AI 的回复内容")
    model: str = Field(..., description="使用的模型名称")
    input_tokens: int = Field(..., description="输入消耗的 token 数")
    output_tokens: int = Field(..., description="输出消耗的 token 数")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = "ok"
    model: str = MODEL


# ============================================================
# 第二部分: 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title="LLM Demo API",
    description="基于 FastAPI 的 LLM 聊天服务 Demo，支持普通调用和流式输出。",
    version="1.0.0",
)


# ============================================================
# 第三部分: LLM 调用核心逻辑
# ============================================================

def call_llm(message: str, system: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> dict:
    """调用 LLM API 并返回原始响应"""
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": MODEL,
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": temperature if temperature is not None else TEMPERATURE,
        "messages": [{"role": "user", "content": message}],
    }

    if system:
        body["system"] = system

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def call_llm_stream(message: str, system: Optional[str] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None):
    """调用 LLM API 的流式接口，逐 token 产出"""
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": MODEL,
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": temperature if temperature is not None else TEMPERATURE,
        "messages": [{"role": "user", "content": message}],
        "stream": True,  # 开启流式输出
    }

    if system:
        body["system"] = system

    # stream=True 让 requests 不要一次性下载完响应
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=120,
        stream=True,
    )
    response.raise_for_status()

    # 逐行读取 SSE 事件流
    # Anthropic 的 SSE 格式: 每行以 "data: " 开头，后跟 JSON
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data = json.loads(decoded[6:])
            # content_block_delta 包含实际的文本片段
            if data.get("type") == "content_block_delta":
                text = data.get("delta", {}).get("text", "")
                if text:
                    yield text


# ============================================================
# 第四部分: 定义 API 端点（路由）
# ============================================================

@app.get("/", summary="欢迎页")
async def root():
    """返回欢迎信息和可用端点列表"""
    return {
        "message": "LLM Demo API 已启动！",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "普通聊天（等待完整回复）",
            "POST /chat/stream": "流式聊天（逐 token 返回）",
            "GET /health": "健康检查",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """检查服务是否正常运行"""
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse, summary="普通聊天")
async def chat(request: ChatRequest):
    """普通聊天端点：等待 LLM 生成完整回复后一次性返回。

    适用场景：
      - 后端服务之间的调用
      - 不需要实时打字效果的场景
      - 简单的请求-响应模式
    """
    try:
        data = call_llm(
            message=request.message,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ChatResponse(
            reply=data["content"][0]["text"],
            model=data["model"],
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", summary="流式聊天")
async def chat_stream(request: ChatRequest):
    """流式聊天端点：逐 token 返回 LLM 的回复（SSE）。

    适用场景：
      - 网页端实现打字机效果
      - 长文本生成时让用户尽早看到内容
      - 提升用户体验（降低感知等待时间）

    SSE 协议说明：
      - 响应的 Content-Type 是 text/event-stream
      - 客户端通过 EventSource API 或 fetch + ReadableStream 读取
      - 每条消息格式: data: {"text": "你"}
    """
    def event_generator():
        try:
            for text in call_llm_stream(
                message=request.message,
                system=request.system,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                # SSE 格式: data: <json>\n\n
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            # 发送结束标记
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 禁用缓存
            "Connection": "keep-alive",         # 保持连接
            "X-Accel-Buffering": "no",          # 禁用 Nginx 缓冲
        },
    )


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Demo 6: FastAPI LLM 服务")
    print("=" * 60)
    print()
    print("启动服务中...")
    print()
    print("访问以下地址:")
    print("  http://127.0.0.1:8001        # 欢迎页")
    print("  http://127.0.0.1:8001/docs   # Swagger 交互式文档")
    print("  http://127.0.0.1:8001/redoc  # ReDoc 文档")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 60)

    # 启动 uvicorn 服务器
    # host="0.0.0.0" 表示监听所有网络接口（局域网可访问）
    # reload=True 开启热重载（修改代码后自动重启）
    uvicorn.run(
        "06_fastapi_demo:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
