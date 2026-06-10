"""智能分块策略：递归字符分块 + 父子文档索引"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SmartChunker:
    """
    多级分块策略：
    1. 递归字符分块（基础）
    2. 父子文档索引（检索小块，返回大块上下文）

    分块大小选择依据：
    - 512 tokens：在检索精度和上下文完整性间取得最佳平衡
    - overlap 64：约 12.5% 的重叠率，防止关键信息在分块边界丢失
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        parent_chunk_size: int = 2048,
    ):
        # 子块：用于精确检索
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            length_function=len,
        )
        # 父块：用于提供上下文
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=128,
            separators=["\n\n", "\n", "。", " ", ""],
        )

    def chunk_with_parent(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """返回 (子块列表, 父块列表)"""
        parent_docs = self.parent_splitter.split_documents(documents)

        child_docs = []
        for i, parent in enumerate(parent_docs):
            parent.metadata["parent_id"] = f"parent_{i}"
            children = self.child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = f"parent_{i}"
            child_docs.extend(children)

        logger.info(f"分块完成: {len(parent_docs)} 父块, {len(child_docs)} 子块")
        return child_docs, parent_docs

    def chunk_simple(self, documents: List[Document]) -> List[Document]:
        """简单分块，适用于不需要父子索引的场景"""
        chunks = self.child_splitter.split_documents(documents)
        logger.info(f"简单分块完成: {len(chunks)} 块")
        return chunks
