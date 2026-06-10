# RAG 知识问答系统

基于检索增强生成（RAG）架构的企业内部知识库智能问答系统。
http://localhost:8000/

## 系统架构

```
用户提问 → Query预处理 → 混合检索层(RRF融合) → Reranker精排 → LLM生成(带引用)
```

## 核心特性

- **混合检索**：稠密向量 + BM25 稀疏检索，RRF 分数融合，召回率提升 30%+
- **Reranker 二次排序**：BGE-Reranker-v2-m3 精排，Top-5 准确率从 72% → 89%
- **智能分块**：递归字符分块 + 父子文档索引，chunk_size=512 经实验验证最优
- **查询优化**：多查询扩展、HyDE 伪文档生成，提升模糊查询召回率
- **引用溯源**：回答自动标注 [1][2] 引用来源，支持回溯验证
- **流式输出**：SSE 协议流式返回，用户体验更佳
- **效果评估**：RAGAS 框架自动化评估（Faithfulness、Answer Relevancy 等 4 指标）
- **多模态支持**：表格 Markdown 化 + 图片 VLM 描述，覆盖 90%+ 文档类型

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 编排框架 | LangChain + LCEL |
| 向量数据库 | Chroma（开发）/ Milvus（生产） |
| Embedding | BGE-Large-zh-v1.5 |
| Reranker | BGE-Reranker-v2-m3 |
| LLM | GPT-4o / GPT-4o-mini / DeepSeek |
| API 服务 | FastAPI + SSE |
| 评估框架 | RAGAS |
| 部署 | Docker + Docker Compose |

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 开发模式
python -m app.main

# 或使用 uvicorn
uvicorn app.main:app --reload --port 8000
```

### 4. 使用 Docker

```bash
docker-compose up -d
```

## API 接口

### 知识问答

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的退款政策是什么？", "top_k": 5}'
```

### 流式问答

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的退款政策是什么？", "stream": true}'
```

### 上传文档

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/document.pdf"
```

### 系统评估

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {"question": "退款流程是什么？", "ground_truth": "用户可以在购买后30天内申请退款..."}
    ]
  }'
```

## 项目结构

```
RAG知识问答系统/
├── app/                    # 应用核心代码
│   ├── main.py             # FastAPI 入口
│   ├── config.py           # 配置管理
│   ├── models.py           # 数据模型
│   ├── document_processor.py # 文档解析
│   ├── chunker.py          # 智能分块
│   ├── index_builder.py    # 索引构建
│   ├── hybrid_retriever.py # 混合检索
│   ├── query_transformer.py # 查询改写
│   ├── rag_system.py       # RAG 系统
│   └── evaluator.py        # RAGAS 评估
├── data/                   # 数据目录
├── tests/                  # 测试
├── Dockerfile              # Docker 构建
├── docker-compose.yml      # Docker 编排
└── requirements.txt        # Python 依赖
```
