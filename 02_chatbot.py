"""
Demo 2: 带上下文记忆的聊天机器人
================================
学习目标:
  - 理解对话上下文管理的核心原理
  - 学会构建一个可交互的聊天机器人
  - 了解上下文窗口限制及应对策略

核心概念:
  1. 上下文窗口（Context Window）: LLM 能处理的最大 token 数
     - 输入 token + 输出 token ≤ 上下文窗口大小
     - 超出限制需要截断或压缩历史
  2. 对话管理: 维护消息列表，每轮把新消息追加进去
  3. 系统人设: 通过 system prompt 定义机器人的性格和行为
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
from datetime import datetime

# API 配置
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"
MODEL = "mimo-v2.5-pro"


class ChatBot:
    """
    聊天机器人类：封装对话管理逻辑。

    属性:
        system_prompt: 系统提示词（定义机器人人设）
        history:       对话历史列表
        max_history:   最大保留的历史轮数（防止超出上下文窗口）
    """

    def __init__(self, system_prompt: str = "你是一个友好的 AI 助手", max_history: int = 20):
        """
        初始化聊天机器人。

        参数:
            system_prompt: 定义机器人的角色和行为
            max_history:   最多保留多少轮对话（防止 token 超限）
        """
        self.system_prompt = system_prompt
        self.history = []          # 存储对话历史
        self.max_history = max_history  # 历史轮数上限

    def chat(self, user_message: str) -> str:
        """
        发送消息并获取回复。

        流程:
          1. 把用户消息加入历史
          2. 截断过长的历史（保留最近的对话）
          3. 调用 API 获取回复
          4. 把 AI 回复加入历史
          5. 返回回复文本

        参数:
            user_message: 用户输入的文本

        返回:
            AI 的回复文本
        """
        # 第 1 步: 把用户消息加入历史
        self.history.append({"role": "user", "content": user_message})

        # 第 2 步: 截断历史，保留最近的对话
        # 为什么要截断？因为上下文窗口有大小限制
        # 如果历史太长，API 会报错或费用飙升
        trimmed_history = self._trim_history()

        # 第 3 步: 调用 API
        reply = self._call_api(trimmed_history)

        # 第 4 步: 把 AI 回复加入历史
        self.history.append({"role": "assistant", "content": reply})

        return reply

    def _trim_history(self) -> list:
        """
        截断对话历史，保留最近的对话。

        策略: 保留最近 max_history 轮对话（每轮 = 1 user + 1 assistant）
        更复杂的策略可以是:
          - 按 token 数截断
          - 保留摘要而非完整历史
          - 保留系统提示 + 最近 N 轮
        """
        # 每轮对话有 2 条消息（user + assistant）
        max_messages = self.max_history * 2

        if len(self.history) > max_messages:
            # 只保留最近的消息
            return self.history[-max_messages:]
        return self.history.copy()

    def _call_api(self, messages: list) -> str:
        """调用 Mimo API 获取回复"""
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
        }

        response = requests.post(
            f"{BASE_URL}/v1/messages",
            headers=headers,
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def get_history(self) -> list:
        """获取完整对话历史"""
        return self.history.copy()

    def clear_history(self):
        """清空对话历史（开始新对话）"""
        self.history = []
        print("[系统] 对话历史已清空")

    def export_history(self, filename: str = None) -> str:
        """
        导出对话历史为 JSON 文件。

        用途:
          - 调试和分析对话质量
          - 保存有趣的对话
          - 用于微调训练数据
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"

        export_data = {
            "system_prompt": self.system_prompt,
            "messages": self.history,
            "total_turns": len(self.history) // 2,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return filename


# ============================================================
# 交互式聊天循环
# ============================================================

def run_interactive_chat():
    """
    运行交互式聊天机器人。

    用户输入消息 → 机器人回复 → 循环直到用户输入 'quit'
    """
    print("=" * 60)
    print("Demo 2: 带上下文记忆的聊天机器人")
    print("=" * 60)

    # 创建机器人实例，设定人设
    bot = ChatBot(
        system_prompt="""你是一个叫"小智"的 AI 助手。
你的特点:
- 说话简洁有趣
- 喜欢用比喻来解释复杂概念
- 会记住用户之前说过的话
- 用中文回复""",
        max_history=10,  # 保留最近 10 轮对话
    )

    print("\n[小智] 你好！我是小智，有什么可以帮你的？")
    print("[提示] 输入 'quit' 退出，输入 'clear' 清空历史，输入 'export' 导出对话\n")

    while True:
        # 获取用户输入
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 再见！")
            break

        # 处理特殊命令
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("[系统] 再见！")
            break
        if user_input.lower() == "clear":
            bot.clear_history()
            continue
        if user_input.lower() == "export":
            filename = bot.export_history()
            print(f"[系统] 对话已导出到 {filename}")
            continue

        # 调用机器人获取回复
        try:
            reply = bot.chat(user_input)
            print(f"小智: {reply}\n")
        except requests.exceptions.RequestException as e:
            print(f"[错误] API 调用失败: {e}\n")


# ============================================================
# 自动演示模式（不需要用户输入）
# ============================================================

def run_auto_demo():
    """
    自动演示模式：展示聊天机器人的上下文记忆能力。
    """
    print("=" * 60)
    print("Demo 2: 聊天机器人 - 自动演示模式")
    print("=" * 60)

    bot = ChatBot(
        system_prompt="你是一个叫'小智'的 AI 助手，说话简洁有趣，用中文回复。",
        max_history=10,
    )

    # 模拟多轮对话，展示上下文记忆
    demo_messages = [
        "你好，我叫小明",
        "我喜欢 Python 编程",
        "你还记得我叫什么名字吗？",  # 测试记忆
        "我最喜欢的编程语言是什么？",  # 测试记忆
        "总结一下你对我的了解",       # 测试综合记忆
    ]

    for msg in demo_messages:
        print(f"\n你: {msg}")
        reply = bot.chat(msg)
        print(f"小智: {reply}")

    # 显示对话历史统计
    history = bot.get_history()
    print(f"\n[统计] 共 {len(history)} 条消息，{len(history) // 2} 轮对话")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import sys

    # 命令行参数控制模式
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_chat()
    else:
        run_auto_demo()
