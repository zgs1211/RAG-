"""FastAPI 应用入口：RAG 知识问答系统 API"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from app.chunker import SmartChunker
from app.config import settings
from app.document_processor import DocumentProcessor
from app.evaluator import RAGEvaluator
from app.hybrid_retriever import HybridRetriever
from app.index_builder import IndexBuilder
from app.models import (
    Citation,
    DocumentUploadResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.rag_system import RAGSystem
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

document_processor: Optional[DocumentProcessor] = None
index_builder: Optional[IndexBuilder] = None
hybrid_retriever: Optional[HybridRetriever] = None
rag_system: Optional[RAGSystem] = None
evaluator: Optional[RAGEvaluator] = None
parent_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, index_builder, hybrid_retriever, rag_system, evaluator
    logger.info("正在初始化 RAG 系统...")
    document_processor = DocumentProcessor()
    index_builder = IndexBuilder()
    try:
        vector_store = index_builder._init_vector_store()
        hybrid_retriever = HybridRetriever(vector_store)
        rag_system = RAGSystem(hybrid_retriever)
        evaluator = RAGEvaluator(rag_system)
        logger.info("RAG 系统初始化完成（使用已有索引）")
    except Exception as e:
        logger.warning(f"初始化索引失败（首次运行正常）: {e}")
        hybrid_retriever = None
        rag_system = None
        evaluator = None
    yield
    logger.info("RAG 系统关闭")


app = FastAPI(title="RAG 知识问答系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = None
_static_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_path):
    STATIC_DIR = _static_path


def ensure_system_ready():
    if rag_system is None:
        raise HTTPException(
            status_code=503, detail="系统尚未初始化，请先上传文档建立索引"
        )


@app.get("/api/health")
async def health_check():
    status = "ok"
    vector_db_status = "unknown"
    embedding_status = "unknown"
    if index_builder is not None:
        try:
            vs = index_builder._init_vector_store()
            vs._collection.count()
            vector_db_status = "ok"
        except Exception:
            vector_db_status = "error"
        try:
            index_builder.embeddings.embed_query("test")
            embedding_status = "ok"
        except Exception:
            embedding_status = "error"
    return HealthResponse(
        status=status, vector_db=vector_db_status, embeddings=embedding_status
    )


@app.get("/api/settings")
async def get_settings():
    return {
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model.replace("BAAI/", ""),
        "vector_db": settings.vector_db_type,
        "chunk_size": settings.chunk_size,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    ensure_system_ready()
    try:
        result = rag_system.ask(
            question=request.question,
            top_k=request.top_k,
            use_multi_query=request.use_multi_query,
            use_hyde=request.use_hyde,
        )
        citations = [Citation(**c) for c in result["citations"]]
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            citations=citations,
            retrieved_chunks=result["retrieved_chunks"],
        )
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    ensure_system_ready()
    if not request.stream:
        result = await query(request)
        return result

    async def generate():
        async for chunk in rag_system.ask_stream(
            question=request.question, top_k=request.top_k
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global hybrid_retriever, rag_system, evaluator
    upload_dir = "./data/documents"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    try:
        docs = document_processor.load(file_path)
        chunk_size = settings.chunk_size
        chunker = SmartChunker(
            chunk_size=chunk_size, chunk_overlap=settings.chunk_overlap
        )
        chunks = chunker.chunk_simple(docs)
        index_builder.build_index(chunks)
        vector_store = index_builder._init_vector_store()
        hybrid_retriever = HybridRetriever(vector_store, corpus_docs=chunks)
        rag_system = RAGSystem(hybrid_retriever)
        evaluator = RAGEvaluator(rag_system)
        return DocumentUploadResponse(
            filename=file.filename,
            doc_id=docs[0].metadata.get("doc_id", ""),
            chunks_count=len(chunks),
            status="success",
        )
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload-batch")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            result = await upload_document(file)
            results.append(result.model_dump())
        except Exception as e:
            results.append(
                {"filename": file.filename, "status": "error", "error": str(e)}
            )
    return {"results": results}


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_system(request: EvaluationRequest):
    ensure_system_ready()
    try:
        scores = evaluator.evaluate(request.test_cases)
        evaluator.print_report(scores)
        return EvaluationResponse(
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            context_precision=scores.get("context_precision", 0.0),
            context_recall=scores.get("context_recall", 0.0),
            total_cases=scores.get("total_cases", 0),
        )
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    upload_dir = "./data/documents"
    if not os.path.exists(upload_dir):
        return {"documents": []}
    files = []
    for fname in os.listdir(upload_dir):
        fpath = os.path.join(upload_dir, fname)
        if os.path.isfile(fpath):
            files.append(
                {
                    "filename": fname,
                    "size": os.path.getsize(fpath),
                    "last_modified": os.path.getmtime(fpath),
                }
            )
    return {"documents": files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )

# Serve frontend (after all API routes)
if STATIC_DIR:

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        from fastapi.responses import FileResponse

        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    logger.info(f"静态文件目录: {STATIC_DIR}")
