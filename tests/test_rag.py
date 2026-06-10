"""RAG 系统单元测试"""

import pytest
from app.chunker import SmartChunker
from app.config import settings
from app.document_processor import DocumentProcessor
from langchain.schema import Document


class TestConfig:
    """配置测试"""

    def test_settings_load(self):
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 64
        assert settings.rerank_top_k == 5


class TestChunker:
    """分块测试"""

    def setup_method(self):
        self.chunker = SmartChunker(chunk_size=100, chunk_overlap=20)

    def test_simple_chunk(self):
        docs = [Document(page_content="这是测试文档内容。" * 20)]
        chunks = self.chunker.chunk_simple(docs)
        assert len(chunks) > 1

    def test_parent_child_chunk(self):
        docs = [Document(page_content="这是测试文档内容。" * 50)]
        children, parents = self.chunker.chunk_with_parent(docs)
        assert len(children) > 0
        assert len(parents) > 0
        for child in children:
            assert "parent_id" in child.metadata
