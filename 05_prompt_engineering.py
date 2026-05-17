"""
Demo 5: Prompt Engineering（提示词工程）
========================================
学习目标:
  - 掌握常用的 Prompt 技术
  - 理解不同 Prompt 策略的效果差异
  - 学会设计高质量的 Prompt

Prompt Engineering 是什么？
  通过设计和优化输入提示词，引导 LLM 产生更好的输出。
  这是使用 LLM 最重要的技能之一，直接影响输出质量。

本 Demo 演示的技术:
  1. Zero-shot（零样本）: 直接提问，不给示例
  2. Few-shot（少样本）: 给几个示例，让 LLM 学习模式
  3. Chain-of-Thought（思维链）: 引导 LLM 分步思考
  4. Role-playing（角色扮演）: 设定特定角色来回答
  5. Output Format（输出格式）: 指定输出的结构
  6. Self-consistency（自一致性）: 多次采样取最优
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

# API 配置
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"
MODEL = "mimo-v2.5-pro"


def call_llm(user_message: str, system: str = "", temperature: float = 0.7) -> str:
    """调用 LLM API 的辅助函数"""
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": MODEL,
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system:
        body["system"] = system

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ============================================================
# 技术 1: Zero-shot（零样本提示）
# ============================================================

def demo_zero_shot():
    """
    Zero-shot: 直接提问，不给任何示例。

    适用场景: 简单任务、通用问题
    优点: 简单直接
    缺点: 对复杂任务效果不稳定
    """
    print("\n[技术 1] Zero-shot（零样本提示）")
    print("-" * 50)

    # 简单的零样本提示
    result = call_llm("将以下句子分类为正面或负面情感：'这家餐厅的菜太好吃了！'")
    print(f"输入: 将以下句子分类为正面或负面情感：'这家餐厅的菜太好吃了！'")
    print(f"输出: {result}")


# ============================================================
# 技术 2: Few-shot（少样本提示）
# ============================================================

def demo_few_shot():
    """
    Few-shot: 给 LLM 几个示例，让它学习输入-输出的模式。

    适用场景: 需要特定格式输出、风格迁移
    优点: 比 zero-shot 更稳定
    缺点: 占用上下文窗口
    """
    print("\n[技术 2] Few-shot（少样本提示）")
    print("-" * 50)

    # 通过示例教 LLM 学习"情感分析"的格式和标准
    prompt = """请分析以下评论的情感，只回答"正面"或"负面"。

评论: 这个手机拍照效果很棒
情感: 正面

评论: 服务态度太差了，等了一个小时
情感: 负面

评论: 价格便宜质量又好
情感: 正面

评论: 包装破损，东西也有问题
情感: 负面

评论: 这本书内容很丰富，值得推荐
情感:"""

    result = call_llm(prompt)
    print(f"输出: {result}")


# ============================================================
# 技术 3: Chain-of-Thought（思维链）
# ============================================================

def demo_chain_of_thought():
    """
    Chain-of-Thought (CoT): 引导 LLM 分步骤思考问题。

    核心思想: 不要直接要答案，让 LLM 展示推理过程。
    通过加一句"让我们一步一步思考"就能显著提升推理能力。

    适用场景: 数学题、逻辑推理、复杂分析
    """
    print("\n[技术 3] Chain-of-Thought（思维链）")
    print("-" * 50)

    # 不用 CoT 的效果
    print("[不用 CoT]")
    result_no_cot = call_llm("一个商店有 15 个苹果，卖掉了 3 箱，每箱 4 个，又进了 2 箱，每箱 6 个，现在有多少个苹果？")
    print(f"回答: {result_no_cot}\n")

    # 用 CoT 的效果
    print("[用 CoT]")
    prompt_cot = """一个商店有 15 个苹果，卖掉了 3 箱，每箱 4 个，又进了 2 箱，每箱 6 个，现在有多少个苹果？

让我们一步一步思考："""
    result_cot = call_llm(prompt_cot)
    print(f"回答: {result_cot}")


# ============================================================
# 技术 4: Role-playing（角色扮演）
# ============================================================

def demo_role_playing():
    """
    Role-playing: 通过 system prompt 设定特定角色。

    为什么有效？
      - 角色设定会影响 LLM 的回答风格、用词、深度
      - 相当于给 LLM 一个"身份"，让它更专注
    """
    print("\n[技术 4] Role-playing（角色扮演）")
    print("-" * 50)

    question = "什么是机器学习？"

    # 角色 1: 面向儿童
    print("[角色: 儿童科普老师]")
    result_child = call_llm(
        question,
        system="你是一个儿童科普老师，用 6 岁孩子能听懂的语言解释复杂概念，多用比喻和例子。"
    )
    print(f"回答: {result_child}\n")

    # 角色 2: 面向专业人士
    print("[角色: 机器学习教授]")
    result_prof = call_llm(
        question,
        system="你是一个顶级大学的机器学习教授，用严谨的学术语言回答，可以引用经典论文和算法。"
    )
    print(f"回答: {result_prof}")


# ============================================================
# 技术 5: Output Format（输出格式控制）
# ============================================================

def demo_output_format():
    """
    Output Format: 指定 LLM 输出的格式。

    常用格式:
      - JSON: 结构化数据，方便程序解析
      - Markdown: 文档格式，方便阅读
      - 表格: 对比信息
      - 列表: 枚举信息
    """
    print("\n[技术 5] Output Format（输出格式控制）")
    print("-" * 50)

    # JSON 格式输出
    prompt_json = """分析以下产品的优缺点，以 JSON 格式输出。

产品：iPhone 15

输出格式要求：
{
  "product": "产品名",
  "pros": ["优点1", "优点2", ...],
  "cons": ["缺点1", "缺点2", ...],
  "rating": 评分(1-10),
  "summary": "一句话总结"
}

只输出 JSON，不要其他内容。"""

    result_json = call_llm(prompt_json)
    print("[JSON 格式输出]")
    print(result_json)

    # 验证是否为有效 JSON
    try:
        data = json.loads(result_json)
        print(f"\n✓ JSON 解析成功！评分: {data.get('rating')}/10")
    except json.JSONDecodeError:
        print("\n⚠ 输出不是有效 JSON（LLM 有时会加 markdown 标记）")


# ============================================================
# 技术 6: Self-consistency（自一致性）
# ============================================================

def demo_self_consistency():
    """
    Self-consistency: 多次采样，取最常见的答案。

    原理:
      - 设置较高的 temperature，让 LLM 多次生成不同答案
      - 统计哪个答案出现最多，选它作为最终答案
      - 类似"投票"机制，提高答案可靠性
    """
    print("\n[技术 6] Self-consistency（自一致性）")
    print("-" * 50)

    question = "15 + 27 * 3 - 18 = ?，只输出数字结果。"

    # 多次采样
    results = []
    for i in range(3):
        result = call_llm(question, temperature=0.8)
        results.append(result.strip())
        print(f"采样 {i+1}: {result.strip()}")

    # 统计最常见的答案
    from collections import Counter
    counter = Counter(results)
    most_common = counter.most_common(1)[0][0]
    print(f"\n最终答案（多数投票）: {most_common}")


# ============================================================
# Prompt 设计原则总结
# ============================================================

def print_prompt_principles():
    """打印 Prompt 设计的最佳实践"""
    print("\n" + "=" * 60)
    print("Prompt 设计最佳实践总结")
    print("=" * 60)

    principles = """
1. 明确具体: 不要说"写点东西"，要说"写一篇 300 字的产品介绍"

2. 提供上下文: 告诉 LLM 你的背景和目的
   ❌ "帮我写个函数"
   ✅ "我是一个 Python 初学者，请帮我写一个计算列表平均值的函数，要有注释"

3. 指定输出格式: 明确你想要什么样的输出
   ❌ "分析一下这个数据"
   ✅ "分析这个数据，用表格对比，包含增长率列"

4. 使用分隔符: 用 --- 或三引号把不同部分分开
   ✅ 请翻译以下文本：\n---\n{text}\n---

5. 给出示例: 一个好示例胜过千言万语

6. 分步指令: 复杂任务拆分成步骤
   ✅ "第一步：分析问题\n第二步：列出方案\n第三步：给出建议"

7. 设定角色: 告诉 LLM 它是谁
   ✅ "你是一个有 10 年经验的 Python 工程师"

8. 控制长度: 明确字数要求
   ✅ "用 100 字以内总结"
"""
    print(principles)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 5: Prompt Engineering（提示词工程）")
    print("=" * 60)

    # 运行所有演示
    demo_zero_shot()
    demo_few_shot()
    demo_chain_of_thought()
    demo_role_playing()
    demo_output_format()
    demo_self_consistency()

    # 打印最佳实践总结
    print_prompt_principles()

    print("\n" + "=" * 60)
    print("Demo 5 完成！你已掌握常用的 Prompt Engineering 技术。")
    print("=" * 60)
