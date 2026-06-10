"""应用配置模块"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，优先从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_model_cheap: str = "gpt-4o-mini"

    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    embedding_dim: int = 1024

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_use_fp16: bool = False

    vector_db_type: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    collection_name: str = "knowledge_base"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    cache_ttl: int = 3600

    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 2048

    retrieve_top_k: int = 20
    rerank_top_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4

    log_level: str = "INFO"
    max_input_tokens: int = 4096
    max_output_tokens: int = 2048

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
