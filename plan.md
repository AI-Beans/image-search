# Image Search Optimization Plan

## Completed

- [x] FAISS IndexFlatIP 集成 (替换 numpy 暴力搜索)
- [x] requirements.txt 默认 faiss-gpu
- [x] P0: HNSW 索引升级 — IndexHNSWFlat(M=32, efConstruction=200, efSearch=128)
- [x] P1: INT8 量化 — IndexHNSWSQ (ScalarQuantizer QT_fp16, 内存减半)
- [x] BFloat16 兼容修复 (embed_text/embed_image 返回 float32)
- [x] 模型切换后端 API (/api/switch_model, 8B/2B)
- [x] _load_index / _save_index 异常保护
- [x] modelscope 依赖
- [x] 删除冗余文件 (REPORT_检查报告.md)

## TODO

- [ ] 前端下拉选择器 (模型切换 UI)
- [ ] favicon.svg
- [ ] P2: Reranker (按需)

## P3: 不执行

- Qdrant 迁移: 过度设计
- FastAPI 替换 Flask: 无收益
- 混合搜索 (BM25 + Dense): 当前不需要
- 分布式架构: 当前不需要

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| FAISS_USE_SQ | 1 | 是否启用 fp16 量化 |
| QWEN3VL_MODEL_PATH | Qwen/Qwen3-VL-Embedding-8B | 模型路径 |
| QWEN3VL_MODEL_SIZE | 8b | 模型大小 (8b/2b) |

## HNSW 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| M | 32 | 每个节点的邻居数, 平衡精度与内存 |
| efConstruction | 200 | 构建时搜索宽度, 越高图质量越好 |
| efSearch | 128 | 搜索时搜索宽度, 越高召回率越高 |
| ScalarQuantizer | QT_fp16 | float32 → float16, 内存减半 |
