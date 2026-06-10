"""完整的 RAG 问答系统：检索 + 生成 + 引用溯源"""

import logging
from typing import AsyncGenerator, List, Optional

from app.config import settings
from app.hybrid_retriever import HybridRetriever
from app.query_transformer import QueryTransformer
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# RAG 提示词模板
RAG_PROMPT_TEMPLATE = """你是一个专业的知识问答助手。请严格根据以下参考资料回答用户问题。

## 回答要求
1. 只基于参考资料回答，不要编造信息
2. 如果参考资料不足以回答，请明确告知用户"根据现有资料无法回答该问题"
3. 在回答中用 [1][2] 等标注引用来源
4. 回答要结构化、简洁明了
5. 使用中文回答

## 参考资料
{context}

## 用户问题
{question}

## 回答
"""


def format_context(docs: List[Document]) -> str:
    """格式化检索结果，带引用编号和来源信息"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        filename = doc.metadata.get("filename", "")
        header = f"[{i}] 来源: {filename or source}"
        if page:
            header += f" (第{page}页)"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGSystem:
    """
    完整 RAG 问答系统

    特性：
    - 混合检索 + Reranker 精排
    - 多查询扩展 / HyDE
    - 流式输出
    - 引用溯源
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: Optional[ChatOpenAI] = None,
    ):
        self.retriever = retriever
        self.query_transformer = QueryTransformer(llm)
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            streaming=True,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_tokens=settings.max_output_tokens,
            timeout=30,
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        use_multi_query: bool = False,
        use_hyde: bool = False,
    ) -> dict:
        """同步问答，返回答案和引用"""
        # Query 扩展
        queries = [question]
        if use_multi_query:
            queries = self.query_transformer.multi_query(question)
        elif use_hyde:
            hyde_doc = self.query_transformer.hyde(question)
            if hyde_doc:
                queries = [question, hyde_doc]

        # 检索：对每个查询分别检索，合并结果
        all_docs = []
        seen_ids = set()
        for q in queries:
            docs = self.retriever.retrieve(q, top_k=top_k)
            for doc in docs:
                doc_id = id(doc)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
                    if len(all_docs) >= top_k:
                        break
            if len(all_docs) >= top_k:
                break

        if not all_docs:
            return {
                "question": question,
                "answer": "根据现有资料无法回答该问题。",
                "citations": [],
                "retrieved_chunks": [],
            }

        # 生成答案
        context = format_context(all_docs)
        prompt_value = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(prompt_value)
        answer = response.content if hasattr(response, "content") else str(response)

        # 构建引用信息
        citations = []
        for i, doc in enumerate(all_docs, 1):
            citations.append(
                {
                    "index": i,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", None),
                    "content": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                    "score": doc.metadata.get("rerank_score", None),
                }
            )

        retrieved_chunks = [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
            }
            for doc in all_docs
        ]

        logger.info(f"问答完成: {len(citations)} 个引用")

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }

    async def ask_stream(
        self,
        question: str,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """流式问答"""
        docs = self.retriever.retrieve(question, top_k=top_k)
        context = format_context(docs)

        chain = (
            {"context": lambda _: context, "question": lambda _: question}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        async for chunk in chain.astream({}):
            yield chunk
