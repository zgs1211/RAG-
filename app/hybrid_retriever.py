"""混合检索器：稠密检索 + BM25 + RRF 融合 + Reranker"""

import logging
from typing import List, Optional

from app.config import settings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def rrf_fusion(
    doc_lists: List[List[Document]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_k: int = 10,
) -> List[Document]:
    """
    RRF (Reciprocal Rank Fusion) 分数融合

    公式: score(d) = sum(w_i * 1/(k + rank_i(d)))
    其中 rank_i(d) 是文档 d 在第 i 个检索结果中的排名
    """
    import math
    from collections import OrderedDict

    if weights is None:
        weights = [1.0 / len(doc_lists)] * len(doc_lists)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    fusion_scores = OrderedDict()

    for docs, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(docs, 1):
            doc_id = doc.page_content[:100]  # 使用内容前缀作为 ID
            score = weight * (1.0 / (k + rank))
            if doc_id in fusion_scores:
                existing_score, existing_doc = fusion_scores[doc_id]
                fusion_scores[doc_id] = (existing_score + score, doc)
            else:
                fusion_scores[doc_id] = (score, doc)

    # 按融合分数排序
    sorted_docs = sorted(
        fusion_scores.values(),
        key=lambda x: x[0],
        reverse=True,
    )

    results = [doc for score, doc in sorted_docs[:top_k]]
    # 保存融合分数到元数据
    for doc, (score, _) in zip(results, sorted_docs[:top_k]):
        doc.metadata["rrf_score"] = round(score, 4)

    return results


class HybridRetriever:
    """
    混合检索器：稠密检索 + BM25 + RRF 融合 + Reranker

    检索流程：
    1. 稠密检索（向量相似度），召回 top_k 候选
    2. 稀疏检索（BM25 关键词），召回 top_k 候选
    3. RRF 分数融合
    4. Reranker 二次精排，取最终结果
    """

    def __init__(self, vector_store, corpus_docs: Optional[List[Document]] = None):
        # 稠密检索
        self.dense_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieve_top_k},
        )

        # 稀疏检索 (BM25)
        if corpus_docs:
            self.bm25_retriever = BM25Retriever.from_documents(
                corpus_docs, k=settings.retrieve_top_k
            )
        else:
            self.bm25_retriever = None

        # Reranker
        self.reranker = None

    def _lazy_init_reranker(self):
        """延迟初始化 Reranker（首次使用时加载）"""
        if self.reranker is None:
            try:
                from FlagEmbedding import FlagReranker

                self.reranker = FlagReranker(
                    settings.reranker_model,
                    use_fp16=settings.reranker_use_fp16,
                )
                logger.info(f"Reranker 加载完成: {settings.reranker_model}")
            except Exception as e:
                logger.warning(f"Reranker 加载失败（跳过）: {str(e)[:80]}")
                self.reranker = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """完整检索流程：混合检索 + Reranking"""
        k = top_k or settings.rerank_top_k

        # Step 1: 稠密检索
        dense_docs = self.dense_retriever.invoke(query)
        logger.debug(f"稠密检索: {len(dense_docs)} 个结果")

        # Step 2: BM25 检索
        all_results = [dense_docs]
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
                logger.debug(f"BM25 检索: {len(bm25_docs)} 个结果")
                all_results.append(bm25_docs)
            except Exception as e:
                logger.warning(f"BM25 检索失败: {e}")

        # Step 3: RRF 融合
        weights = [settings.dense_weight, settings.sparse_weight]
        candidates = rrf_fusion(
            all_results,
            weights=weights[: len(all_results)],
            top_k=min(k * 3, settings.retrieve_top_k * 2),
        )

        if not candidates:
            logger.warning(f"检索结果为空: {query}")
            return []

        # Step 4: Reranking，精排到 top_k
        self._lazy_init_reranker()
        if self.reranker and len(candidates) > 1:
            try:
                pairs = [[query, doc.page_content] for doc in candidates]
                scores = self.reranker.compute_score(pairs)

                if isinstance(scores, list) and len(scores) > 0:
                    if isinstance(scores[0], list):
                        scores = [s[0] for s in scores]
                elif isinstance(scores, float):
                    scores = [scores]

                scored_docs = list(zip(candidates, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                top_docs = [doc for doc, _ in scored_docs[:k]]

                score_map = {id(doc): score for doc, score in scored_docs[:k]}
                for doc in top_docs:
                    doc.metadata["rerank_score"] = float(score_map.get(id(doc), 0.0))

                logger.info(
                    f"检索完成: {len(candidates)} -> {len(top_docs)} (Reranked)"
                )
                return top_docs
            except Exception as e:
                logger.warning(f"Reranker 执行失败，回退到 RRF 结果: {e}")
                return candidates[:k]
        else:
            return candidates[:k]

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[tuple]:
        """检索并返回带分数的文档列表"""
        k = top_k or settings.rerank_top_k
        docs = self.retrieve(query, k)
        return [
            (doc, doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0.0)))
            for doc in docs
        ]
