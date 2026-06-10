# RAG 知识问答系统 - 管理命令手册
http://localhost:8000/docs#/
## 服务管理

### 启动服务

```bash
wsl bash -c "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 tmux new-session -d -s rag-server \
  'cd /home/zgs/projects/RAG知识问答系统 && \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zgs/miniconda3/envs/agent-env/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8000 2>&1'"
```

### 查看服务日志

```bash
wsl tmux capture-pane -t rag-server -p
```

### 实时查看日志（attach）

```bash
wsl tmux attach -t rag-server
# 退出 attach: Ctrl+B, 然后按 D
```

### 重启服务

```bash
wsl bash -c "tmux kill-session -t rag-server 2>/dev/null; sleep 1; \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 tmux new-session -d -s rag-server \
  'cd /home/zgs/projects/RAG知识问答系统 && \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zgs/miniconda3/envs/agent-env/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8000 2>&1'"
```

### 停止服务

```bash
wsl bash -c "tmux kill-session -t rag-server"
```

### 检查服务状态

```bash
wsl bash -c "ps aux | grep uvicorn | grep -v grep; echo ---; ss -tlnp | grep 8000"
```

---

## API 接口

### 健康检查

```bash
curl -s http://localhost:8000/health
```

### 上传文档

```bash
curl -s -F "file=@/home/zgs/projects/RAG知识问答系统/data/documents/company_policy.md" \
  http://localhost:8000/documents/upload
```

### 批量上传文档

```bash
curl -s -F "files=@/path/to/doc1.pdf" -F "files=@/path/to/doc2.docx" \
  http://localhost:8000/documents/upload-batch
```

### 知识问答

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是Agent？", "top_k": 3}'
```

### 使用多查询扩展

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "员工请假有什么规定？", "top_k": 5, "use_multi_query": true}'
```

### 使用 HyDE 查询改写

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "加班怎么算钱？", "top_k": 5, "use_hyde": true}'
```

### 流式问答

```bash
curl -s -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "退款政策是什么？", "stream": true}'
```

### 列出已索引文档

```bash
curl -s http://localhost:8000/documents
```

### 系统评估

```bash
curl -s -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {"question": "退款流程是什么？", "ground_truth": "用户可以在购买后30天内申请退款，需要订单号和购买凭证..."},
      {"question": "年假有多少天？", "ground_truth": "员工每年享有15天带薪年假"}
    ]
  }'
```

### Swagger 交互式文档

在浏览器中打开：http://localhost:8000/docs

---

## 数据管理

### 查看文档目录

```bash
wsl ls -la /home/zgs/projects/RAG知识问答系统/data/documents/
```

### 手动放文档

将 PDF/Word/HTML/Markdown 文件放到 `data/documents/` 目录后，通过 API 上传：

```bash
curl -F "file=@/home/zgs/projects/RAG知识问答系统/data/documents/你的文件.pdf" \
  http://localhost:8000/documents/upload
```

### 清空向量数据库（重新索引）

```bash
wsl bash -c "rm -rf /home/zgs/projects/RAG知识问答系统/data/chroma_db/*; \
  echo '向量数据库已清空'"
```
清空后需要重启服务并重新上传文档。

### 查看已缓存的 Embedding 模型

```bash
wsl ls -la ~/.cache/huggingface/hub/ | grep models
```

---

## 开发命令

### 进入项目目录

```bash
wsl bash
cd /home/zgs/projects/RAG知识问答系统
```

### 激活 conda 环境

```bash
wsl bash -c "export PATH=~/miniconda3/bin:\$PATH && \
  source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate agent-env"
```

### 安装/更新依赖

```bash
wsl ~/miniconda3/envs/agent-env/bin/pip install -r \
  /home/zgs/projects/RAG知识问答系统/requirements.txt
```

### 运行测试

```bash
cd /home/zgs/projects/RAG知识问答系统
wsl ~/miniconda3/envs/agent-env/bin/python -m pytest tests/ -v
```

### 语法检查

```bash
wsl bash -c 'cd /home/zgs/projects/RAG知识问答System && \
  for f in app/*.py; do \
    /home/zgs/miniconda3/envs/agent-env/bin/python -m py_compile "\$f" && \
    echo "OK: \$f"; \
  done'
```

### 单独测试模块

```bash
wsl ~/miniconda3/envs/agent-env/bin/python -c "
import sys
sys.path.insert(0, '/home/zgs/projects/RAG知识问答系统')
from app.config import settings
print(f'模型: {settings.llm_model}')
print(f'分块大小: {settings.chunk_size}')
print(f'向量数据库: {settings.vector_db_type}')
"
```

---

## Docker 部署

### 构建并启动（需在项目目录）

```bash
cd /home/zgs/projects/RAG知识问答系统
docker-compose up -d
```

### 仅构建

```bash
cd /home/zgs/projects/RAG知识问答系统
docker build -t rag-knowledge-qa .
```

### 查看容器日志

```bash
docker logs -f rag-knowledge-qa
```

---

## 配置文件说明

`.env` 文件位于项目根目录，关键配置项：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `OPENAI_API_KEY` | API 密钥 | `sk-xxx` |
| `OPENAI_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 主模型 | `deepseek-chat` / `gpt-4o` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-small-zh-v1.5` |
| `VECTOR_DB_TYPE` | 向量数据库类型 | `chroma` / `milvus` |
| `CHUNK_SIZE` | 分块大小（tokens） | `512` |
| `CHUNK_OVERLAP` | 分块重叠 | `64` |
| `RETRIEVE_TOP_K` | 检索召回数量 | `20` |
| `RERANK_TOP_K` | 重排序后数量 | `5` |

---

## 常见问题

### 嵌入模型加载失败

```bash
# 设置离线模式，使用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 服务端口被占用

```bash
# 查看占用端口的进程
wsl ss -tlnp | grep 8000
# 或者启动时指定其他端口
--port 8001
```

### 向量数据库数据清除

```bash
wsl rm -rf /home/zgs/projects/RAG知识问答系统/data/chroma_db/
```
