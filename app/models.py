"""Pydantic 数据模型"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """问答请求"""

    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    stream: bool = Field(default=False, description="是否流式输出")
    use_hyde: bool = Field(default=False, description="是否使用 HyDE 查询改写")
    use_multi_query: bool = Field(default=False, description="是否使用多查询扩展")


class Citation(BaseModel):
    """引用来源"""

    index: int = Field(..., description="引用编号")
    source: str = Field("", description="来源文件路径")
    page: Optional[str] = Field(None, description="页码")
    content: str = Field("", description="引用内容片段")
    score: Optional[float] = Field(None, description="相关性分数")


class QueryResponse(BaseModel):
    """问答响应"""

    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="生成的回答")
    citations: List[Citation] = Field(default_factory=list, description="引用来源列表")
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="检索到的原始文档块"
    )


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    filename: str = Field(..., description="文件名")
    doc_id: str = Field(..., description="文档 ID")
    chunks_count: int = Field(0, description="分块数量")
    status: str = Field("success", description="处理状态")


class EvaluationRequest(BaseModel):
    """评估请求"""

    test_cases: List[Dict[str, str]] = Field(..., description="测试用例列表")


class EvaluationResponse(BaseModel):
    """评估响应"""

    faithfulness: float = Field(0.0, description="忠实度分数")
    answer_relevancy: float = Field(0.0, description="答案相关性分数")
    context_precision: float = Field(0.0, description="上下文精确率")
    context_recall: float = Field(0.0, description="上下文召回率")
    total_cases: int = Field(0, description="总测试用例数")


class ErrorResponse(BaseModel):
    """错误响应"""

    detail: str = Field(..., description="错误详情")
    error_code: str = Field("UNKNOWN_ERROR", description="错误码")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field("ok", description="服务状态")
    vector_db: str = Field("unknown", description="向量数据库状态")
    embeddings: str = Field("unknown", description="Embedding 模型状态")
