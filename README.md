# Image Semantic Search | 图像语义检索系统

A production-ready multimodal image retrieval system built on [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) and [Qwen3-VL-Reranker](https://github.com/QwenLM/Qwen3-VL-Embedding). Upload images, describe what you're looking for in natural language, and get ranked results instantly.

基于 [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) 与 [Qwen3-VL-Reranker](https://github.com/QwenLM/Qwen3-VL-Embedding) 构建的多模态图像检索系统。上传图片，用自然语言描述目标内容，即刻获取排序结果。

---

## Features | 核心特性

- **Two-Stage Retrieval Pipeline | 两阶段检索流水线** — FAISS HNSW coarse ranking → Reranker cross-attention fine ranking, toggle on demand
  FAISS HNSW 粗排 → Reranker Cross-Attention 精排，按需启用
- **Multimodal Understanding | 多模态理解** — Unified visual-text embedding space supporting 30+ languages
  统一的视觉-文本嵌入空间，支持 30+ 语言自然语言搜索
- **Model Hot-Switch | 模型热切换** — Switch between 8B (high accuracy) and 2B (low VRAM) at runtime via API
  运行时通过 API 切换 8B（高精度）/ 2B（低显存）模型
- **Batch Indexing with Progress | 批量索引** — Drag-and-drop upload with real-time embedding progress tracking
  拖拽批量上传，实时追踪嵌入进度
- **Hardware Adaptive | 硬件自适应** — Auto-detect CUDA / Apple MPS / CPU with optimal dtype (bfloat16 / float16 / float32)
  自动检测 CUDA / Apple MPS / CPU，选用最优精度

---

## Architecture | 系统架构

```
                         ┌─────────────────────────────────────────────────┐
                         │              Indexing Pipeline                  │
                         │              索引构建流程                        │
                         │                                                 │
                         │  Image ──▸ Qwen3-VL-Embedding ──▸ L2 Norm     │
                         │              (Visual Encoder)        │         │
                         │                                    ▼          │
                         │                           FAISS HNSW Index     │
                         │                          (Inner Product)       │
                         └─────────────────────────────────────────────────┘

                         ┌─────────────────────────────────────────────────┐
                         │            Search Pipeline                      │
                         │            检索流程                              │
                         │                                                 │
  Query ──▸ Embedding ──▸ FAISS HNSW ─────┬──▸ Top-K Results (粗排)       │
             (Text         (Stage 1:       │                                │
              Encoder)      Coarse Rank)   │   Reranker enabled?           │
                             top_k × 3     ├──▸ YES ──▸ Qwen3-VL-Reranker │
                             candidates     │           (Stage 2:          │
                                            │            Cross-Attention)  │
                                            │            Fine Rank)        │
                                            │                │             │
                                            │                ▼             │
                                            └──▸ Re-ranked Top-K Results  │
                                                 (精排结果)                │
                         └─────────────────────────────────────────────────┘
```

**Stage 1 — Coarse Rank / 粗排：** Dual-tower architecture. Query and images are independently encoded into the same embedding space, matched via cosine similarity (inner product on L2-normalized vectors). Fast, suitable for large-scale retrieval.

双塔架构。Query 和图像独立编码到同一向量空间，通过余弦相似度（L2 归一化向量的内积）匹配。速度快，适合大规模召回。

**Stage 2 — Fine Rank / 精排 (Optional)：** Single-tower architecture. Query-image pairs are jointly processed through cross-attention for deep inter-modal interaction. Outputs a binary relevance score (sigmoid of `yes - no` logit). Higher precision, activated on demand.

单塔架构。Query-图像对联合输入 Cross-Attention 进行深度跨模态交互，输出二值相关性分数。精度更高，按需启用。

---

## Quick Start | 快速开始

### 1. Install Dependencies / 安装依赖

> Requires Python 3.10 (`faiss-gpu` does not support Python 3.12+)

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

For GPU environments, install the CUDA-matched PyTorch from [pytorch.org](https://pytorch.org/).

### 2. Download Models / 下载模型

**Option A: ModelScope (recommended for China / 国内推荐)**

```bash
export MODELSCOPE_ENDPOINT='https://mirror.alibaba.com/modelscope'
export MODELSCOPE_CACHE=./models

# Embedding model (required / 必需)
python3 download_model.py --source modelscope --type embedding --size 8b

# Reranker model (optional / 可选)
python3 download_model.py --source modelscope --type reranker --size 2b
```

**Option B: HuggingFace**

```bash
python3 download_model.py --source huggingface --type embedding --size 8b
python3 download_model.py --source huggingface --type reranker --size 2b
```

**Option C: Low VRAM / 低显存模式 (use 2B models / 使用 2B 模型)**

```bash
python3 download_model.py --source modelscope --type embedding --size 2b
```

### 3. Launch / 启动服务

```bash
# Use locally downloaded model / 使用本地模型
export QWEN3VL_MODEL_PATH=./models/Qwen3-VL-Embedding-8B
python3 app.py

# Or auto-download on first run / 或首次启动时自动下载
python3 app.py
```

Open `http://localhost:5200` in your browser.

---

## Usage | 使用方法

1. Click **Load Model** to load the embedding model into GPU memory
   点击 **Load Model** 加载嵌入模型至显存
2. Drag-and-drop or click to upload images — indexing starts automatically
   拖拽或点击上传图片，系统自动建立向量索引
3. Type a natural language query, e.g. `children playing in water`, `日落`, `a dog running`
   输入自然语言描述进行搜索
4. Toggle **Reranker** in the header to enable two-stage retrieval for higher precision
   在页头切换 **Reranker** 开关，启用两阶段精排提升精度

---

## Project Structure | 项目结构

```
image-search/
├── app.py                      # Flask web server & API gateway
├── embedder.py                 # Core engine: embedding, FAISS indexing, search, reranker orchestration
├── qwen3_vl_embedding.py       # Qwen3-VL-Embedding model wrapper (official repo)
├── qwen3_vl_reranker.py        # Qwen3-VL-Reranker model wrapper (official repo)
├── download_model.py           # CLI model downloader (ModelScope / HuggingFace)
├── requirements.txt
├── templates/
│   └── index.html              # Single-page Web UI (dark theme)
├── data/                       # FAISS index + metadata (auto-generated)
│   ├── faiss.index
│   └── metadata.json
├── images/uploaded/            # Original uploaded images (auto-generated)
├── static/thumbnails/          # 256×256 JPEG thumbnails (auto-generated)
└── models/                     # Local model weights (downloaded separately)
    ├── Qwen3-VL-Embedding-8B/
    ├── Qwen3-VL-Embedding-2B/
    └── Qwen3-VL-Reranker-2B/
```

---

## API Reference | 接口文档

### Model Management / 模型管理

| Endpoint | Method | Description | 说明 |
|----------|--------|-------------|------|
| `/api/status` | GET | Server status, model info, reranker state | 服务状态、模型信息、Reranker 状态 |
| `/api/load_model` | POST | Trigger async model loading | 异步加载模型 |
| `/api/switch_model` | POST | Switch model size `{"model_size": "8b"\|"2b"}` | 切换模型规格，切换后需重新加载 |

### Image Indexing / 图像索引

| Endpoint | Method | Description | 说明 |
|----------|--------|-------------|------|
| `/api/index` | POST | Upload & embed images (multipart/form-data, field: `files`) | 上传图片并生成嵌入，支持批量 |
| `/api/index/progress` | GET | Query indexing progress `{"current", "total", "done"}` | 查询索引进度 |

### Search / 检索

| Endpoint | Method | Description | 说明 |
|----------|--------|-------------|------|
| `/api/search` | POST | Search with text query | 文本语义检索 |
| | | `{"query": "...", "top_k": 5, "use_reranker": false}` | |
| | | Response includes `"reranker_used": true/false` | 返回结果标记是否经过精排 |

### Reranker / 精排控制

| Endpoint | Method | Description | 说明 |
|----------|--------|-------------|------|
| `/api/reranker/toggle` | POST | Enable/disable reranker `{"enabled": true\|false}` | 开关 Reranker |
| `/api/reranker/status` | GET | Reranker state: enabled, loaded, loading, model path | Reranker 状态查询 |

### System / 系统

| Endpoint | Method | Description | 说明 |
|----------|--------|-------------|------|
| `/api/stats` | GET | Index statistics (image count, dimension, model info) | 索引统计信息 |
| `/api/reset` | POST | Reset entire index (deletes all data) | 重置全部索引数据 |

---

## Configuration | 配置参数

### Environment Variables / 环境变量

| Variable | Default | Description | 说明 |
|----------|---------|-------------|------|
| `PORT` | `5200` | Web server port | 服务端口 |
| `QWEN3VL_MODEL_PATH` | `Qwen/Qwen3-VL-Embedding-8B` | Embedding model path (local or HF ID) | 嵌入模型路径 |
| `QWEN3VL_MODEL_SIZE` | `8b` | Default model size (8b / 2b) | 默认模型规格 |
| `QWEN3VL_RERANKER_ENABLED` | `0` | Enable reranker by default (`1` = on) | 默认启用 Reranker |
| `QWEN3VL_RERANKER_MODEL_PATH` | `Qwen/Qwen3-VL-Reranker-2B` | Reranker model path (local or HF ID) | Reranker 模型路径 |

### FAISS HNSW Parameters / 索引参数

| Parameter | Value | Description | 说明 |
|-----------|-------|-------------|------|
| M | 32 | Neighbors per node, balances recall & memory | 每节点邻居数，平衡召回率与内存 |
| efConstruction | 200 | Build-time search width, higher = better graph quality | 构建时搜索宽度 |
| efSearch | 128 | Search-time beam width, higher = better recall | 搜索时搜索宽度 |
| Metric | `METRIC_INNER_PRODUCT` | Cosine similarity on L2-normalized vectors | L2 归一化后的内积 = 余弦相似度 |

### Reranker Parameters / 精排参数

| Parameter | Value | Description | 说明 |
|-----------|-------|-------------|------|
| Candidate Multiplier | 3 | Retrieve `top_k × 3` candidates from FAISS for reranking | FAISS 召回候选倍数 |

---

## Hardware Requirements / 硬件需求

### Embedding Model Only / 仅嵌入模型

| Model | Min VRAM | Recommended GPU |
|-------|----------|-----------------|
| Qwen3-VL-Embedding-8B | 16 GB | NVIDIA A100 / RTX 4090 |
| Qwen3-VL-Embedding-2B | 6 GB | NVIDIA RTX 3060+ |
| CPU mode | 32 GB RAM | Very slow, not recommended |

### With Reranker Enabled / 同时启用 Reranker

| Configuration | Min VRAM | Note |
|---------------|----------|------|
| Embedding-8B + Reranker-2B | 20 GB | Recommended production config |
| Embedding-2B + Reranker-2B | 10 GB | Budget config |
| CPU mode | 48 GB RAM | Extremely slow |

> The reranker model is lazy-loaded: it only occupies GPU memory when first invoked after being enabled.
> Reranker 模型采用懒加载策略，仅在首次启用并调用时才占用 GPU 显存。

---

## Tech Stack | 技术栈

| Component | Technology | Role |
|-----------|-----------|------|
| Web Framework | Flask | API gateway & static file serving |
| Embedding Model | Qwen3-VL-Embedding (8B/2B) | Dual-tower visual-text encoder |
| Reranker Model | Qwen3-VL-Reranker (8B/2B) | Single-tower cross-attention scorer |
| Vector Index | FAISS HNSW (GPU) | Approximate nearest neighbor search |
| Model Loading | HuggingFace Transformers | Model download & inference |
| Model Source | ModelScope / HuggingFace | Distribution channels |

---

## License

This project uses models from [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding). Please refer to the Qwen license for model usage terms. Application code is provided as-is.

本项目使用 [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) 模型，请参阅 Qwen 许可协议了解模型使用条款。应用代码按原样提供。
