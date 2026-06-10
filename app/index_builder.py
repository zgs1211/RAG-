"""索引构建器：向量索引构建与管理"""

import logging
import os
from typing import List, Optional

from app.chunker import SmartChunker
from app.config import settings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

logger = logging.getLogger(__name__)


class IndexBuilder:
    """向量索引构建器，支持 Chroma (开发) 和 Milvus (生产)"""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.collection_name
        self.embeddings = self._init_embeddings()
        self.vector_store = None

    def _init_embeddings(self):
        """初始化 Embedding 模型"""
        import warnings

        model_name = settings.embedding_model
        logger.info(f"加载 Embedding 模型: {model_name}")

        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": settings.embedding_device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": settings.embedding_batch_size,
                },
            )
        except Exception as e:
            logger.warning(f"Embedding 模型 {model_name} 加载失败: {e}")
            logger.info("尝试使用 OpenAI Embedding 作为回退...")
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

    def _init_vector_store(self):
        """初始化向量数据库"""
        if settings.vector_db_type == "milvus":
            return self._init_milvus()
        else:
            return self._init_chroma()

    def _init_chroma(self) -> Chroma:
        """初始化 Chroma（开发环境）"""
        persist_dir = settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=persist_dir,
        )

    def _init_milvus(self):
        """初始化 Milvus（生产环境）"""
        from langchain_milvus import Milvus as LangchainMilvus

        return LangchainMilvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={
                "host": settings.milvus_host,
                "port": settings.milvus_port,
            },
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            },
        )

    def build_index(self, documents: List[Document]):
        """批量构建索引"""
        self.vector_store = self._init_vector_store()
        batch_size = 500
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            logger.info(f"已索引 {min(i + batch_size, total)}/{total}")

        logger.info(f"索引构建完成: 共 {total} 个文档块")

    def build_parent_child_index(self, documents: List[Document]) -> InMemoryStore:
        """构建父子文档索引，返回父文档存储器"""
        chunker = SmartChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            parent_chunk_size=settings.parent_chunk_size,
        )
        child_docs, parent_docs = chunker.chunk_with_parent(documents)

        # 父文档存内存
        parent_store = InMemoryStore()
        for doc in parent_docs:
            pid = doc.metadata.get("parent_id")
            if pid:
                parent_store.mset([(pid, doc)])

        # 子文档入向量库
        self.build_index(child_docs)

        logger.info(
            f"父子索引构建完成: {len(child_docs)} 子块, {len(parent_docs)} 父块"
        )
        return parent_store

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """获取检索器实例"""
        if self.vector_store is None:
            self.vector_store = self._init_vector_store()

        kwargs = search_kwargs or {"k": settings.retrieve_top_k}
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=kwargs,
        )
