"""查询转换器：多查询扩展、HyDE、查询改写"""

import logging
import re
from typing import List, Optional

from app.config import settings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 多查询扩展提示词
MULTI_QUERY_PROMPT = """你是一个专业的知识问答助手。请从不同角度将以下用户问题改写为3个语义相近但表述不同的查询。
每个查询占一行，直接输出查询内容，不要编号。

原始问题：{question}
"""

# HyDE 提示词
HYDE_PROMPT = """请针对以下问题，写一段简短的假设性回答（约100字）。
不需要保证准确性，只需要包含相关的术语和概念。回答要专业、有条理。

问题：{question}
"""

# 查询改写提示词
QUERY_REWRITE_PROMPT = """请将以下用户查询改写得更加清晰、完整，便于检索系统找到相关信息。
保持原意不变，补充可能缺失的上下文信息。

原始查询：{question}

改写后的查询：
"""


class QueryTransformer:
    """查询改写与扩展，提升检索效果"""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=15,
        )

    def multi_query(self, original_query: str) -> List[str]:
        """生成多个角度的查询，用于提升召回率"""
        prompt = MULTI_QUERY_PROMPT.format(question=original_query)
        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            queries = [
                q.strip().lstrip("0123456789.、- ")
                for q in content.strip().split("\n")
                if q.strip()
            ]
            # 去重并限制数量
            seen = set()
            unique_queries = []
            for q in [original_query] + queries[:3]:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            logger.info(f"多查询扩展: {original_query} -> {len(unique_queries)} 个查询")
            return unique_queries
        except Exception as e:
            logger.warning(f"多查询扩展失败: {e}")
            return [original_query]

    def hyde(self, query: str) -> Optional[str]:
        """HyDE: 先生成假设性答案，用答案去检索"""
        prompt = HYDE_PROMPT.format(question=query)
        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            logger.info(f"HyDE 生成完成: {len(content)} 字符")
            return content.strip()
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {e}")
            return None

    def rewrite(self, query: str) -> str:
        """查询改写：使查询更清晰完整"""
        prompt = QUERY_REWRITE_PROMPT.format(question=query)
        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            rewritten = content.strip().strip('"').strip("'")
            logger.info(f"查询改写: {query} -> {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return query
