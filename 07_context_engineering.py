"""
Demo 7: Context Engineering（上下文工程）
==========================================
学习目标:
  - 理解 Context Engineering 的核心思想
  - 掌握上下文组装、压缩、动态选择等关键技术
  - 学会在有限的上下文窗口中放入最有价值的信息

Context Engineering vs Prompt Engineering:
  - Prompt Engineering 关注"怎么问"（措辞、格式、指令）
  - Context Engineering 关注"给什么"（信息选择、组装、压缩）
  - 两者互补：好的 Prompt 需要好的 Context 才能发挥最大效果

本 Demo 演示的技术:
  1. Context Assembly（上下文组装）: 从多个来源拼装上下文
  2. Context Window Management（窗口管理）: 在 token 限制内优先放入重要信息
  3. Dynamic Context Selection（动态选择）: 根据用户问题选择相关上下文
  4. Context Compression（上下文压缩）: 压缩冗余信息，腾出空间
  5. Structured Context（结构化上下文）: 用结构化格式提升 LLM 理解效率
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
from typing import Optional

BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"
MODEL = "mimo-v2.5-pro"


def call_llm(user_message: str, system: str = "", max_tokens: int = 1024) -> str:
    """调用 LLM API"""
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system:
        body["system"] = system
    response = requests.post(f"{BASE_URL}/v1/messages", headers=headers, json=body, timeout=60)
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ============================================================
# 第一部分: Context Assembly（上下文组装）
# ============================================================
# 核心思想: LLM 本身没有外部知识，需要我们在 prompt 中提供。
# 上下文组装就是从多个来源（文档、数据库、API）收集信息，拼装成一个完整的上下文。

# 模拟：多个知识片段（实际项目中可能来自向量数据库、文档系统等）
KNOWLEDGE_BASE = {
    "company_policy": """
    公司假期政策：
    1. 年假：入职满1年后享受5天年假，满3年后10天，满5年后15天
    2. 病假：每年最多10天带薪病假，需提供医院证明
    3. 事假：每月最多2天无薪事假
    4. 调休：加班可按1:1比例兑换调休
    """,
    "product_info": """
    产品信息 - AI 助手 Pro：
    - 版本：v3.2
    - 价格：基础版 99元/月，专业版 299元/月，企业版定制
    - 功能：智能问答、文档分析、代码生成、多语言支持
    - 支持的模型：GPT-4、Claude、Mimo 等主流模型
    - 部署方式：SaaS / 私有化部署
    """,
    "technical_docs": """
    API 接口文档：
    - POST /api/chat: 聊天接口，支持流式和非流式
    - GET /api/models: 获取可用模型列表
    - POST /api/documents: 上传文档到知识库
    - 认证方式：Bearer Token
    - 速率限制：基础版 100次/分钟，专业版 1000次/分钟
    """,
}


def demo_basic_assembly():
    """
    演示: 基础上下文组装

    做法: 把所有相关知识片段拼接在一起，作为上下文发给 LLM。
    问题: 如果知识片段很多，会超出上下文窗口限制。
    """
    print("Demo 1: 基础上下文组装")
    print("-" * 50)
    print("思路: 把所有知识拼在一起，让 LLM 回答问题\n")

    question = "公司年假政策是什么？入职2年有多少天年假？"

    # 组装上下文：把所有知识片段拼起来
    context = "\n---\n".join(KNOWLEDGE_BASE.values())

    prompt = f"""请根据以下参考资料回答用户问题。如果资料中没有相关信息，请说明。

参考资料:
{context}

用户问题: {question}"""

    reply = call_llm(prompt)
    print(f"问题: {question}")
    print(f"回答: {reply}")
    print(f"\n[上下文长度: {len(context)} 字符]\n")


# ============================================================
# 第二部分: Dynamic Context Selection（动态上下文选择）
# ============================================================
# 核心思想: 不是每次都把所有知识塞进去，而是根据用户问题，
#           只选择最相关的知识片段。
# 优势: 节省 token、减少噪音、提高回答质量。

# 模拟: 简单的关键词匹配（实际项目中会用向量检索/Embedding）
def select_relevant_context(question: str) -> str:
    """根据问题动态选择相关上下文"""
    question_lower = question.lower()
    selected = []

    # 关键词匹配策略（简化版，实际用 Embedding 相似度）
    if any(kw in question_lower for kw in ["假期", "年假", "病假", "事假", "调休", "请假"]):
        selected.append(("公司政策", KNOWLEDGE_BASE["company_policy"]))

    if any(kw in question_lower for kw in ["产品", "价格", "功能", "版本", "ai助手", "助手"]):
        selected.append(("产品信息", KNOWLEDGE_BASE["product_info"]))

    if any(kw in question_lower for kw in ["api", "接口", "文档", "调用", "部署"]):
        selected.append(("技术文档", KNOWLEDGE_BASE["technical_docs"]))

    if not selected:
        # 没匹配到关键词时，返回全部（兜底策略）
        return "\n---\n".join(KNOWLEDGE_BASE.values())

    # 拼接选中的上下文
    parts = [f"[{name}]\n{content}" for name, content in selected]
    return "\n---\n".join(parts)


def demo_dynamic_selection():
    """
    演示: 动态上下文选择

    做法: 根据用户问题的关键词，只选择相关的知识片段。
    效果: 上下文更精简，回答更聚焦。
    """
    print("Demo 2: 动态上下文选择")
    print("-" * 50)
    print("思路: 根据问题只选择相关的知识片段，避免无关信息干扰\n")

    questions = [
        "公司年假政策是什么？",         # 应匹配 company_policy
        "AI 助手 Pro 有哪些功能？",     # 应匹配 product_info
        "API 的速率限制是多少？",       # 应匹配 technical_docs
    ]

    for q in questions:
        context = select_relevant_context(q)
        prompt = f"""请根据以下参考资料回答用户问题。如果资料中没有相关信息，请说明。

参考资料:
{context}

用户问题: {q}"""

        reply = call_llm(prompt)
        print(f"问题: {q}")
        print(f"选中的上下文: {len(context)} 字符（vs 全量 {sum(len(v) for v in KNOWLEDGE_BASE.values())} 字符）")
        print(f"回答: {reply}\n")


# ============================================================
# 第三部分: Context Compression（上下文压缩）
# ============================================================
# 核心思想: 当上下文太长时，先用 LLM 对其进行压缩/摘要，
#           再把压缩后的版本作为上下文使用。
# 场景: 长文档、大量历史对话、多轮检索结果合并。

def compress_context(context: str, focus: str = "") -> str:
    """用 LLM 压缩上下文，保留关键信息"""
    focus_instruction = f"重点关注与「{focus}」相关的信息。" if focus else ""
    prompt = f"""请将以下内容压缩为简短的摘要，保留关键事实和数据，去掉冗余描述。{focus_instruction}

原始内容:
{context}

要求:
- 保留所有具体数字、规则、条件
- 用简洁的条目格式输出
- 长度不超过原文的 30%"""

    return call_llm(prompt)


def demo_context_compression():
    """
    演示: 上下文压缩

    做法: 先用 LLM 对长上下文进行摘要压缩，再用于回答问题。
    效果: 在有限的 token 预算内，塞入更多信息。
    """
    print("Demo 3: 上下文压缩")
    print("-" * 50)
    print("思路: 先压缩上下文，再用压缩后的版本回答问题\n")

    # 模拟一段很长的上下文
    long_context = """
    公司成立于2018年，总部位于北京。公司专注于人工智能领域，主要产品包括AI助手Pro、
    智能客服系统、文档分析平台等。公司目前有员工约200人，其中研发团队占60%。

    公司的假期政策如下：入职满1年后享受5天年假，满3年后10天，满5年后15天。
    每年最多10天带薪病假，需要提供医院证明。每月最多2天无薪事假。
    加班可以按1:1比例兑换调休。公司还提供弹性工作制，核心工作时间为10:00-16:00。

    AI助手Pro是公司的旗舰产品，当前版本为v3.2。定价方面，基础版99元/月，
    专业版299元/月，企业版根据需求定制。功能包括智能问答、文档分析、代码生成、
    多语言支持等。支持GPT-4、Claude、Mimo等主流模型。部署方式支持SaaS和私有化部署。

    公司的API接口包括：POST /api/chat（聊天接口，支持流式和非流式），
    GET /api/models（获取可用模型列表），POST /api/documents（上传文档到知识库）。
    认证方式为Bearer Token。速率限制方面，基础版100次/分钟，专业版1000次/分钟。
    """

    question = "年假有几天？"
    print(f"问题: {question}")
    print(f"原始上下文: {len(long_context)} 字符")

    # 第一步: 压缩上下文（聚焦于问题相关的信息）
    compressed = compress_context(long_context, focus="年假")
    print(f"压缩后上下文: {len(compressed)} 字符")
    print(f"压缩率: {len(compressed)/len(long_context)*100:.1f}%\n")
    print(f"压缩结果:\n{compressed}\n")

    # 第二步: 用压缩后的上下文回答问题
    prompt = f"""根据以下参考资料回答问题。

参考资料:
{compressed}

问题: {question}"""

    reply = call_llm(prompt)
    print(f"回答: {reply}\n")


# ============================================================
# 第四部分: Structured Context（结构化上下文）
# ============================================================
# 核心思想: 同样的信息，用结构化格式（XML、JSON、Markdown）组织，
#           比纯文本更容易被 LLM 理解和引用。
# 原因: LLM 在训练时见过大量结构化数据，对格式敏感。

def demo_structured_context():
    """
    演示: 结构化上下文 vs 自由文本

    同样的信息，用 XML 标签包裹后，LLM 更容易准确引用。
    """
    print("Demo 4: 结构化上下文")
    print("-" * 50)
    print("对比: 自由文本 vs XML 结构化格式\n")

    question = "专业版的价格和速率限制分别是多少？"

    # 方式 A: 自由文本（信息混杂，LLM 容易搞混）
    free_text_context = """
    AI助手Pro产品有三个版本，基础版99元/月，专业版299元/月，企业版定制。
    API速率限制基础版100次/分钟，专业版1000次/分钟。
    """

    # 方式 B: XML 结构化（信息边界清晰，LLM 可以精确定位）
    structured_context = """
<product>
  <name>AI 助手 Pro</name>
  <pricing>
    <plan name="基础版" price="99元/月" />
    <plan name="专业版" price="299元/月" />
    <plan name="企业版" price="定制" />
  </pricing>
</product>

<api>
  <rate_limit plan="基础版" value="100次/分钟" />
  <rate_limit plan="专业版" value="1000次/分钟" />
</api>
"""

    # 对比两种方式的回答
    prompt_a = f"""根据以下资料回答问题。

资料:
{free_text_context}

问题: {question}"""

    prompt_b = f"""根据以下资料回答问题。

资料:
{structured_context}

问题: {question}"""

    print(f"问题: {question}\n")

    print("[自由文本上下文]")
    reply_a = call_llm(prompt_a)
    print(f"回答: {reply_a}\n")

    print("[XML 结构化上下文]")
    reply_b = call_llm(prompt_b)
    print(f"回答: {reply_b}\n")

    print("观察: XML 结构化格式让信息边界更清晰，LLM 不容易搞混不同字段的值。")


# ============================================================
# 第五部分: Context Window Management（上下文窗口管理）
# ============================================================
# 核心思想: 上下文窗口是有限的（如 128K tokens），需要策略性地
#           分配空间给不同部分：系统提示、知识、对话历史、用户问题。
#
# 分配策略（类比内存管理）:
#   [系统提示词]  [知识上下文]  [对话历史]  [用户问题]
#      10%           50%          30%         10%

def demo_window_management():
    """
    演示: 上下文窗口的空间分配

    在有限的 token 预算内，如何分配空间给不同部分。
    """
    print("Demo 5: 上下文窗口管理")
    print("-" * 50)
    print("思路: 在有限的 token 预算内，策略性地分配空间\n")

    # 模拟一个有 token 预算限制的场景
    TOKEN_BUDGET = 2000  # 假设我们只有 2000 tokens 的预算
    CHARS_PER_TOKEN = 1.5  # 中文大约 1 token ≈ 1.5 字符
    char_budget = int(TOKEN_BUDGET * CHARS_PER_TOKEN)

    print(f"Token 预算: {TOKEN_BUDGET} tokens ≈ {char_budget} 字符\n")

    # 各部分的预算分配
    allocation = {
        "系统提示词": 0.10,   # 10%: 角色设定和规则
        "知识上下文": 0.50,   # 50%: 检索到的知识（最重要）
        "对话历史":  0.30,   # 30%: 多轮对话的上下文
        "用户问题":  0.10,   # 10%: 当前问题
    }

    print("空间分配策略:")
    for name, ratio in allocation.items():
        chars = int(char_budget * ratio)
        print(f"  {name}: {ratio*100:.0f}% = {chars} 字符")

    print(f"\n实际应用中的优化技巧:")
    print(f"  1. 系统提示词: 精简但明确，避免冗长的规则说明")
    print(f"  2. 知识上下文: 只放最相关的片段（配合动态选择）")
    print(f"  3. 对话历史: 超出窗口时截断或摘要（保留最近的几轮）")
    print(f"  4. 用户问题: 通常最短，不需要特别优化")
    print(f"  5. 预留空间: 留 10-20% 给模型的输出（max_tokens）")

    # 实际演示: 用压缩后的对话历史
    print(f"\n示例: 对话历史压缩")
    print("-" * 30)

    # 模拟一段较长的对话历史
    conversation_history = [
        {"role": "user", "content": "你好，我想了解一下你们的产品"},
        {"role": "assistant", "content": "您好！我们主要产品是 AI 助手 Pro，支持智能问答、文档分析、代码生成等功能。"},
        {"role": "user", "content": "价格是多少？"},
        {"role": "assistant", "content": "基础版99元/月，专业版299元/月，企业版根据需求定制。"},
        {"role": "user", "content": "有什么区别？"},
        {"role": "assistant", "content": "主要区别在速率限制和功能范围。基础版100次/分钟，专业版1000次/分钟，企业版无限制。"},
        {"role": "user", "content": "我们团队有50人，推荐哪个版本？"},
        {"role": "assistant", "content": "50人团队建议专业版或企业版。专业版性价比高，适合中小团队。"},
    ]

    # 策略1: 全量保留（消耗大量 token）
    full_history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
    print(f"全量保留: {len(full_history)} 字符")

    # 策略2: 只保留最近2轮（节省空间）
    recent_history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history[-4:]])
    print(f"最近2轮: {len(recent_history)} 字符")

    # 策略3: 摘要 + 最近1轮（平衡信息量和空间）
    summary = "用户了解了产品功能和价格，50人团队，已推荐专业版或企业版。"
    last_round = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history[-2:]])
    compressed_history = f"[对话摘要] {summary}\n\n[最近对话]\n{last_round}"
    print(f"摘要+最近1轮: {len(compressed_history)} 字符")

    print(f"\n策略3通常是最佳选择：保留了关键上下文，又节省了空间。")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 7: Context Engineering（上下文工程）")
    print("=" * 60)
    print()

    demo_basic_assembly()
    print("=" * 60 + "\n")

    demo_dynamic_selection()
    print("=" * 60 + "\n")

    demo_context_compression()
    print("=" * 60 + "\n")

    demo_structured_context()
    print("=" * 60 + "\n")

    demo_window_management()
    print("=" * 60 + "\n")

    print("Demo 7 完成！你已掌握 Context Engineering 的核心技巧。")
    print("=" * 60)
