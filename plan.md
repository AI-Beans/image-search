# Image Search — Optimization Roadmap | 优化路线图

## Completed | 已完成

### P0: HNSW Index Upgrade / 索引升级

- [x] FAISS `IndexFlatIP` integration (replaced numpy brute-force search) / FAISS 内积索引集成
- [x] HNSW upgrade — `IndexHNSWFlat(M=32, efConstruction=200, efSearch=128)`
- [x] `_load_index` / `_save_index` exception protection / 异常保护
- [x] `faiss-gpu` as default dependency / 默认依赖

### P1: Quantization & Compatibility / 量化与兼容

- [x] INT8 quantization — `IndexHNSWSQ (ScalarQuantizer QT_fp16)`, memory halved / INT8 量化，内存减半
- [x] BFloat16 compatibility — `embed_text`/`embed_image` return float32 / BFloat16 兼容修复
- [x] Model switching API — `POST /api/switch_model`, 8B/2B hot-swap / 模型热切换 API
- [x] ModelScope dependency / 依赖补充
- [x] Cleanup — removed redundant report file / 删除冗余文件

### P2: Reranker (On-Demand) / 精排（按需）

- [x] Qwen3-VL-Reranker model wrapper — `qwen3_vl_reranker.py` (from official repo)
  Qwen3-VL-Reranker 模型封装，来源官方仓库
- [x] Two-stage search — FAISS retrieves `top_k × 3` candidates → Reranker scores each → sort → return `top_k`
  两阶段检索：FAISS 召回 3× 候选 → Reranker 逐对精排 → 排序返回
- [x] Lazy loading — Reranker model loads into GPU only on first use after being enabled
  懒加载策略，仅首次启用时占用显存
- [x] Runtime toggle — `POST /api/reranker/toggle`, frontend switch in header
  运行时开关，前端页头可切换
- [x] Graceful fallback — if Reranker fails, returns FAISS coarse-ranking results
  容错降级：Reranker 异常时回退到粗排结果
- [x] Download support — `python3 download_model.py --type reranker --size 2b`
  下载支持 2B/8B Reranker 模型

---

## TODO | 待办

- [ ] Frontend model dropdown selector / 前端模型下拉选择器
- [ ] `favicon.svg`

---

## P3: Rejected | 不执行

| Proposal | Reason |
|----------|--------|
| Qdrant migration | Over-engineering for current scale |
| FastAPI replacing Flask | No measurable benefit |
| Hybrid search (BM25 + Dense) | Not needed for current use case |
| Distributed architecture | Not needed for current scale |

---

## Technical Configuration | 技术配置

See [README.md — Configuration](./README.md#configuration--配置参数) for full environment variable reference.

完整环境变量参考见 [README.md — Configuration](./README.md#configuration--配置参数)。

### FAISS HNSW

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 32 | Neighbors per node |
| efConstruction | 200 | Build-time search width |
| efSearch | 128 | Search-time beam width |
| Metric | `METRIC_INNER_PRODUCT` | Cosine similarity on L2-normalized vectors |

### Reranker Pipeline

| Parameter | Value | Description |
|-----------|-------|-------------|
| Candidate Multiplier | 3 | FAISS retrieves `top_k × 3` for reranking |
| Scoring Mechanism | `sigmoid(linear(last_hidden_state[-1]))` | Binary yes/no relevance via cross-attention |
| Loading Strategy | Lazy | Model loaded on first invocation after enable |
