"""
Demo 3: RAG（检索增强生成）
===========================
学习目标:
  - 理解 RAG 的核心原理：检索 → 增强 → 生成
  - 学会使用 Embedding 将文本转为向量
  - 实现一个简单的文档问答系统

RAG 流程:
  1. 文档处理: 把长文档切分成小块（chunks）
  2. 向量化:  用 Embedding 模型把文本块转为向量
  3. 存储:     把向量存入向量数据库
  4. 检索:     用户问题也转为向量，找到最相似的文本块
  5. 增强:     把检索到的文本块作为上下文，拼接到 prompt 中
  6. 生成:     LLM 基于上下文生成回答

为什么需要 RAG？
  - LLM 的知识有截止日期（训练数据的时间点）
  - LLM 不知道你公司的内部文档
  - RAG 让 LLM 能"查阅资料"再回答，减少幻觉
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import numpy as np
from typing import List, Tuple

# API 配置
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
API_KEY = "tp-sloug5cuh06qejzd22q09ihuwhdvhn34uhq4nh5ej19feuwo"
MODEL = "mimo-v2.5-pro"


# ============================================================
# 第一部分: 文档处理 - 切分（Chunking）
# ============================================================

def split_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    将长文本切分成小块。

    为什么要切分？
      - LLM 的上下文窗口有限，不能把整个文档塞进去
      - 小块文本更容易精准检索
      - 需要在精度和上下文之间找平衡

    参数:
        text:       原始文本
        chunk_size: 每个块的最大字符数
        overlap:    相邻块的重叠字符数（保持上下文连贯）

    切分策略示意（chunk_size=10, overlap=3）:
      文本: "ABCDEFGHIJKLMNOPQR"
      块1:  "ABCDEFGHIJ"
      块2:  "HIJKLMNOP"  ← 与块1重叠了 "HIJ"
      块3:  "PQR..."

    返回:
        切分后的文本块列表
    """
    chunks = []
    start = 0

    while start < len(text):
        # 计算当前块的结束位置
        end = start + chunk_size

        # 切出一个块
        chunk = text[start:end]

        # 只保留非空块
        if chunk.strip():
            chunks.append(chunk)

        # 下一个块的起始位置（向前移动，但保留 overlap 重叠）
        start = end - overlap

    return chunks


# ============================================================
# 第二部分: 向量化 - 简单的 TF-IDF Embedding
# ============================================================

class SimpleEmbedding:
    """
    简单的文本向量化工具（基于字符频率，教学演示用）。

    生产环境中应该使用:
      - OpenAI Embedding API (text-embedding-3-small)
      - 本地模型: sentence-transformers, BGE, M3E 等
      - 中文推荐: text2vec-chinese, m3e-base

    这里用字符频率向量来演示原理，无需额外 API 调用。
    """

    def __init__(self):
        # 构建字符表（常用中文字符 + 英文字母 + 数字）
        self.chars = list("的一是不了人我在有他这为之大来以个中上们到说国和地也子时")
        self.chars += list("abcdefghijklmnopqrstuvwxyz0123456789")
        self.dim = len(self.chars)  # 向量维度

    def encode(self, text: str) -> np.ndarray:
        """
        将文本编码为向量。

        方法: 统计每个字符出现的频率，归一化后作为向量。
        这是一种简化的 "词袋模型"（Bag of Words）。

        参数:
            text: 输入文本

        返回:
            向量（numpy 数组）
        """
        vector = np.zeros(self.dim)
        text = text.lower()

        # 统计字符频率
        for i, char in enumerate(self.chars):
            vector[i] = text.count(char)

        # L2 归一化（让向量长度为 1，方便计算余弦相似度）
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码多段文本"""
        return np.array([self.encode(t) for t in texts])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。

    余弦相似度:
      - 衡量两个向量的方向是否一致
      - 值域 [-1, 1]，越接近 1 越相似
      - 公式: cos(θ) = (A·B) / (|A| × |B|)

    直觉理解:
      - 两个向量指向同一个方向 → 相似度 = 1
      - 两个向量垂直（无关）    → 相似度 = 0
      - 两个向量方向相反        → 相似度 = -1
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# 第三部分: 向量数据库（内存版）
# ============================================================

class VectorStore:
    """
    简单的向量数据库（内存存储，教学演示用）。

    生产环境应该使用:
      - ChromaDB: 轻量级，适合原型
      - Milvus:   分布式，适合大规模
      - Pinecone: 托管服务，无需运维
      - FAISS:    Facebook 出品，性能优秀
    """

    def __init__(self, embedder: SimpleEmbedding):
        self.embedder = embedder
        self.documents = []    # 存储原始文本
        self.vectors = []      # 存储对应的向量

    def add_documents(self, texts: List[str]):
        """
        添加文档到向量数据库。

        流程: 文本 → Embedding → 存储
        """
        for text in texts:
            vector = self.embedder.encode(text)
            self.documents.append(text)
            self.vectors.append(vector)

        print(f"[向量库] 已添加 {len(texts)} 个文档块，总计 {len(self.documents)} 个")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        相似度检索：找到与查询最相关的文档。

        流程:
          1. 把查询文本也转为向量
          2. 计算查询向量与所有文档向量的相似度
          3. 按相似度排序，返回 top_k 个结果

        参数:
            query: 查询文本
            top_k: 返回最相似的 K 个结果

        返回:
            [(文档文本, 相似度分数), ...] 按相似度降序排列
        """
        if not self.documents:
            return []

        # 把查询转为向量
        query_vec = self.embedder.encode(query)

        # 计算与所有文档的相似度
        similarities = []
        for i, doc_vec in enumerate(self.vectors):
            sim = cosine_similarity(query_vec, doc_vec)
            similarities.append((self.documents[i], sim))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


# ============================================================
# 第四部分: RAG 完整流程
# ============================================================

class RAGSystem:
    """
    RAG 系统：整合检索和生成。

    流程:
      1. 索引阶段（离线）: 文档 → 切分 → 向量化 → 存储
      2. 查询阶段（在线）: 问题 → 检索相关文档 → 拼接 prompt → LLM 生成
    """

    def __init__(self):
        self.embedder = SimpleEmbedding()
        self.vector_store = VectorStore(self.embedder)

    def index_document(self, document: str):
        """
        索引一个文档（离线处理）。

        步骤:
          1. 切分文档为小块
          2. 添加到向量数据库
        """
        print("[RAG] 正在切分文档...")
        chunks = split_text(document, chunk_size=200, overlap=50)
        print(f"[RAG] 文档切分为 {len(chunks)} 个块")

        print("[RAG] 正在向量化并存储...")
        self.vector_store.add_documents(chunks)
        print("[RAG] 索引完成！")

    def query(self, question: str, top_k: int = 3) -> str:
        """
        查询 RAG 系统（在线处理）。

        步骤:
          1. 用问题检索最相关的文档块
          2. 把文档块拼接到 prompt 中作为上下文
          3. 调用 LLM 基于上下文生成回答

        参数:
            question: 用户问题
            top_k:    检索几个相关文档块

        返回:
            LLM 的回答
        """
        # 第 1 步: 检索
        print(f"[RAG] 正在检索相关文档...")
        results = self.vector_store.search(question, top_k=top_k)

        if not results:
            return "抱歉，没有找到相关文档。"

        # 第 2 步: 构建增强 prompt
        # 这是 RAG 的核心：把检索到的文档作为上下文
        context = "\n\n---\n\n".join([doc for doc, score in results])
        print(f"[RAG] 找到 {len(results)} 个相关文档块")

        # 第 3 步: 调用 LLM
        prompt = f"""基于以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请明确说明"根据现有资料无法回答"，不要编造答案。

参考资料:
{context}

用户问题: {question}"""

        headers = {
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = {
            "model": MODEL,
            "max_tokens": 1024,
            "system": "你是一个基于文档的问答助手，回答要准确、简洁。",
            "messages": [
                {"role": "user", "content": prompt}
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
# 主程序：运行 RAG 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 3: RAG 检索增强生成")
    print("=" * 60)

    # 准备示例文档（模拟公司内部知识库）
    sample_document = """
    Python 编程语言简介

    Python 是一种高级、解释型、通用编程语言，由 Guido van Rossum 于 1991 年创建。
    Python 的设计哲学强调代码的可读性和简洁性。它的语法允许程序员用更少的代码行
    来表达概念，相比 C++ 或 Java 等语言更加简洁。

    Python 的主要特点包括：
    1. 简单易学：语法清晰直观，适合初学者
    2. 丰富的标准库：内置大量模块，覆盖文件处理、网络编程、数据库等
    3. 跨平台：支持 Windows、Linux、macOS 等操作系统
    4. 动态类型：变量不需要声明类型，运行时自动推断
    5. 面向对象：支持类、继承、多态等 OOP 特性

    Python 在以下领域广泛应用：
    - Web 开发：Django、Flask 等框架
    - 数据科学：NumPy、Pandas、Matplotlib
    - 机器学习：TensorFlow、PyTorch、scikit-learn
    - 自动化运维：Ansible、SaltStack
    - 网络爬虫：Scrapy、BeautifulSoup

    Python 的版本历史：
    - Python 2.0（2000年）：引入列表推导式、垃圾回收
    - Python 3.0（2008年）：重大更新，不完全向后兼容
    - Python 3.12（2023年）：性能提升，改进错误消息

    Python 虚拟环境：
    虚拟环境是 Python 的隔离运行环境，允许不同项目使用不同版本的依赖包。
    创建虚拟环境命令：python -m venv myenv
    激活虚拟环境（Windows）：myenv\\Scripts\\activate
    激活虚拟环境（Linux/Mac）：source myenv/bin/activate

    pip 包管理器：
    pip 是 Python 的官方包管理工具，用于安装和管理第三方库。
    安装包命令：pip install package_name
    查看已安装包：pip list
    导出依赖：pip freeze > requirements.txt
    安装依赖：pip install -r requirements.txt
    """

    # 创建 RAG 系统并索引文档
    rag = RAGSystem()
    rag.index_document(sample_document)

    # 测试查询
    questions = [
        "Python 是谁创建的？",
        "Python 有哪些 Web 开发框架？",
        "如何创建 Python 虚拟环境？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")
        answer = rag.query(q)
        print(f"回答: {answer}")

    print("\n" + "=" * 60)
    print("Demo 3 完成！你已理解 RAG 的核心流程。")
    print("=" * 60)
