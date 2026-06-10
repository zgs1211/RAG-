"""文档处理器：支持 PDF/Word/HTML/Markdown 格式解析"""

import hashlib
import logging
import os
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """统一文档处理器，支持多格式解析和元数据提取"""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".md": TextLoader,
        ".txt": TextLoader,
    }

    def load(self, file_path: str) -> List[Document]:
        """加载并解析文档"""
        suffix = os.path.splitext(file_path)[1].lower()
        loader_cls = self.LOADER_MAP.get(suffix)
        if not loader_cls:
            raise ValueError(f"不支持的文件格式: {suffix}")

        logger.info(f"正在解析文档: {file_path}")
        try:
            loader = loader_cls(file_path)
            docs = loader.load()
        except Exception as e:
            logger.error(f"文档解析失败 {file_path}: {e}")
            raise

        # 添加元数据
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        for doc in docs:
            doc.metadata["source"] = file_path
            doc.metadata["doc_id"] = file_hash
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["filetype"] = suffix.lstrip(".")

        logger.info(f"解析完成: {len(docs)} 页/段")
        return docs

    def load_directory(self, dir_path: str) -> List[Document]:
        """批量加载目录下的所有支持文档"""
        all_docs = []
        supported_exts = set(self.LOADER_MAP.keys())

        for root, _, files in os.walk(dir_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_exts:
                    fpath = os.path.join(root, fname)
                    try:
                        docs = self.load(fpath)
                        all_docs.extend(docs)
                    except Exception as e:
                        logger.warning(f"跳过文件 {fpath}: {e}")
                        continue

        logger.info(f"批量加载完成: 共 {len(all_docs)} 文档块")
        return all_docs

    @staticmethod
    def extract_metadata(doc: Document) -> dict:
        """提取并标准化文档元数据"""
        return {
            "source": doc.metadata.get("source", ""),
            "doc_id": doc.metadata.get("doc_id", ""),
            "filename": doc.metadata.get("filename", ""),
            "filetype": doc.metadata.get("filetype", ""),
            "page": doc.metadata.get("page", None),
            "total_pages": doc.metadata.get("total_pages", None),
        }
