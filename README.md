# Image Semantic Search

基于 [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) 的图片语义搜索工具。上传图片，用自然语言描述你要找的内容，即刻检索匹配结果。

## Features

- 上传批量图片，自动生成视觉嵌入向量
- 自然语言搜索（中英文均可），返回 Top-K 匹配结果
- Web UI，支持拖拽上传、实时搜索、图片预览
- 支持 GPU (CUDA) / Apple Silicon (MPS) / CPU
- 可选 8B（高精度）或 2B（低显存）模型

## Quick Start

### 1. 安装依赖

**注意：需要 Python 3.10（faiss-gpu 不支持 Python 3.12+）**

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

GPU 环境建议参照 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版本的 torch。

### 2. 下载模型

**方式 A：从 ModelScope 下载（推荐国内用户）**

建议配置阿里云镜像加速：
```bash
export MODELSCOPE_ENDPOINT='https://mirror.alibaba.com/modelscope'
export MODELSCOPE_CACHE=./models
python3 download_model.py --source modelscope --size 8b
```

**方式 B：从 HuggingFace 下载**

```bash
python3 download_model.py --source huggingface --size 8b
```

**方式 C：使用 2B 小模型（显存 < 8GB）**

```bash
python3 download_model.py --source modelscope --size 2b
```

### 3. 启动服务

```bash
# 使用本地已下载的模型
export QWEN3VL_MODEL_PATH=./models/Qwen3-VL-Embedding-8B
python3 app.py

# 或直接使用 HuggingFace/ModelScope 自动下载（首次启动会自动下载）
python3 app.py
```

打开浏览器访问 `http://localhost:5200`

## Usage

1. 点击 **Load Model** 加载嵌入模型（首次启动需要等模型加载到显存）
2. 拖拽或点击上传图片，系统自动建立索引
3. 在搜索框输入描述，如 `孩子在玩水`、`sunset over the ocean`、`a dog running in the park`
4. 查看相似度排序的搜索结果

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Upload Image │────▸│ Qwen3-VL-Embed   │────▸│ Embedding    │
│              │     │ (visual encoder) │     │ Vector Store │
└──────────────┘     └──────────────────┘     └──────┬───────┘
                                                       │
┌──────────────┐     ┌──────────────────┐              │
│ Text Query   │────▸│ Qwen3-VL-Embed   │              │
│              │     │ (text encoder)   │────▸ cosine  │
└──────────────┘     └──────────────────┘     sim ────▸┘
```

图片和文本通过同一个多模态嵌入模型映射到共享的向量空间，通过余弦相似度匹配。

## Project Structure

```
image-search/
├── app.py                   # Flask web server + API
├── embedder.py              # Embedding logic (index, search, store)
├── qwen3_vl_embedding.py   # Qwen3-VL-Embedding model wrapper (from official repo)
├── download_model.py        # Model download script
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Web UI
├── data/                    # Embedding index (auto-generated)
├── images/uploaded/         # Uploaded images (auto-generated)
└── static/thumbnails/       # Image thumbnails (auto-generated)
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Server status, model info |
| `/api/load_model` | POST | Trigger model loading |
| `/api/index` | POST | Upload & index images (multipart/form-data) |
| `/api/search` | POST | Search with text query (JSON) |
| `/api/stats` | GET | Index statistics |
| `/api/reset` | POST | Reset entire index |

## Hardware Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| Qwen3-VL-Embedding-8B | 16 GB | NVIDIA A100 / RTX 4090 |
| Qwen3-VL-Embedding-2B | 6 GB | NVIDIA RTX 3060+ |
| CPU mode | 32 GB RAM | Very slow, not recommended |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3VL_MODEL_PATH` | `Qwen/Qwen3-VL-Embedding-8B` | Model path (local or HuggingFace ID) |
| `PORT` | `5200` | Web server port |

## License

This project uses the [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) model. The code in this repository is provided as-is. Please refer to the Qwen license for model usage terms.
