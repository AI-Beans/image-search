import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

from qwen3_vl_embedding import Qwen3VLEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = Path(__file__).parent / "images" / "uploaded"
THUMBS_DIR = Path(__file__).parent / "static" / "thumbnails"

for d in [DATA_DIR, IMAGES_DIR, THUMBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_FILE = DATA_DIR / "faiss.index"
METADATA_FILE = DATA_DIR / "metadata.json"

_instruction = "Retrieve relevant images based on the user's description."

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
USE_SQ = False

MODEL_MAP = {
    "8b": "Qwen/Qwen3-VL-Embedding-8B",
    "2b": "Qwen/Qwen3-VL-Embedding-2B",
}

RERANKER_MODEL_MAP = {
    "2b": "Qwen/Qwen3-VL-Reranker-2B",
    "8b": "Qwen/Qwen3-VL-Reranker-8B",
}

RERANKER_CANDIDATE_MULTIPLIER = 3

_model = None
_reranker_model = None
_reranker_enabled = os.environ.get("QWEN3VL_RERANKER_ENABLED", "0") == "1"
_reranker_loading = False
_faiss_index = None
_metadata = []
_current_model_size = os.environ.get("QWEN3VL_MODEL_SIZE", "2b")

_index_lock = threading.Lock()


def _model_path():
    return os.environ.get("QWEN3VL_MODEL_PATH", MODEL_MAP.get(_current_model_size, MODEL_MAP["8b"]))


def get_model():
    global _model
    if _model is None:
        path = _model_path()
        logger.info("Loading embedding model from %s ...", path)
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float16
        _model = Qwen3VLEmbedder(
            model_name_or_path=path,
            torch_dtype=dtype,
        )
        logger.info("Embedding model loaded.")
    return _model


def switch_model(size: str) -> str:
    global _model, _faiss_index, _metadata, _current_model_size
    if size not in MODEL_MAP:
        raise ValueError(f"Unknown model size: {size}, choose from {list(MODEL_MAP.keys())}")
    with _index_lock:
        _current_model_size = size
        _model = None
        _faiss_index = None
        _metadata = []
        for f in FAISS_INDEX_FILE, METADATA_FILE:
            if f.exists():
                f.unlink()
    logger.info("Switched model to %s", MODEL_MAP[size])
    return MODEL_MAP[size]


def get_current_model_info() -> dict:
    path = _model_path()
    name = path.split("/")[-1]
    return {
        "model_path": path,
        "model_name": name,
        "model_size": _current_model_size,
    }


def _create_index(dim):
    if USE_SQ:
        index = faiss.IndexHNSWSQ(
            dim, faiss.ScalarQuantizer_QT_fp16, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
    else:
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


def _ensure_index(dim):
    global _faiss_index
    if _faiss_index is None or _faiss_index.d != dim:
        _faiss_index = _create_index(dim)


def _load_index():
    global _faiss_index, _metadata
    try:
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                _metadata = json.load(f)
    except Exception as e:
        logger.warning("Failed to load metadata: %s", e)
        _metadata = []
    try:
        if FAISS_INDEX_FILE.exists():
            _faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
            if hasattr(_faiss_index, "hnsw"):
                _faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
            logger.info("Loaded FAISS index with %d vectors", _faiss_index.ntotal)
    except Exception as e:
        logger.warning("Failed to load FAISS index: %s", e)
        _faiss_index = None


def _save_index():
    with _index_lock:
        try:
            if _faiss_index is not None and _faiss_index.ntotal > 0:
                faiss.write_index(_faiss_index, str(FAISS_INDEX_FILE))
            with open(METADATA_FILE, "w") as f:
                json.dump(_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save index: %s", e)


_load_index()


def embed_text(text: str) -> np.ndarray:
    model = get_model()
    result = model.process([{"text": text, "instruction": _instruction}])
    return result.cpu().to(torch.float32).numpy().flatten()


def embed_image(image_path: str) -> np.ndarray:
    model = get_model()
    result = model.process([{"image": image_path, "instruction": _instruction}])
    return result.cpu().to(torch.float32).numpy().flatten()


def _generate_thumbnail(image_path: str, filename: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
    thumb_name = f"thumb_{filename}"
    thumb_path = THUMBS_DIR / thumb_name
    img.save(str(thumb_path), "JPEG", quality=85)
    return thumb_name


def add_image(image_path: str, filename: str) -> dict:
    global _faiss_index, _metadata

    emb = embed_image(image_path)
    thumb_name = _generate_thumbnail(image_path, filename)

    entry = {
        "id": filename,
        "filename": filename,
        "original_path": str(image_path),
        "thumbnail": f"/static/thumbnails/{thumb_name}",
        "indexed_at": datetime.now().isoformat(),
    }

    with _index_lock:
        _ensure_index(emb.shape[0])
        _faiss_index.add(emb.reshape(1, -1).astype("float32"))
        _metadata.append(entry)
    _save_index()

    return entry


def add_image_batch(image_paths: list, batch_size: int = 8, progress_callback=None) -> list:
    global _faiss_index
    model = get_model()
    all_embeddings = []
    results = []
    total = len(image_paths)

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        inputs = [{"image": p, "instruction": _instruction} for p in batch_paths]
        embeddings = model.process(inputs).cpu().to(torch.float32).numpy()
        all_embeddings.append(embeddings)

        for path in batch_paths:
            filename = Path(path).name
            thumb_name = _generate_thumbnail(path, filename)

            entry = {
                "id": filename,
                "filename": filename,
                "original_path": str(path),
                "thumbnail": f"/static/thumbnails/{thumb_name}",
                "indexed_at": datetime.now().isoformat(),
            }
            with _index_lock:
                _metadata.append(entry)
            results.append(entry)

        if progress_callback:
            processed = min(batch_start + batch_size, total)
            progress_callback(processed, total)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    with _index_lock:
        _ensure_index(all_embeddings.shape[1])
        _faiss_index.add(all_embeddings)

    _save_index()
    return results


def search(query: str, top_k: int = 5) -> list:
    if top_k < 1:
        top_k = 1
    with _index_lock:
        if _faiss_index is None or _faiss_index.ntotal == 0:
            return []
        k = min(top_k, _faiss_index.ntotal)
        q_emb = embed_text(query).astype("float32").reshape(1, -1)
        scores, indices = _faiss_index.search(q_emb, k)

        results = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            break
        entry = dict(_metadata[idx])
        entry["score"] = float(score)
        results.append(entry)

    return results


def get_stats() -> dict:
    with _index_lock:
        meta_count = len(_metadata)
        dim = int(_faiss_index.d) if _faiss_index is not None and _faiss_index.ntotal > 0 else 0
        idx_type = type(_faiss_index).__name__ if _faiss_index is not None else "none"
    return {
        "total_images": meta_count,
        "embedding_dim": dim,
        "model_loaded": _model is not None,
        "has_gpu": torch.cuda.is_available(),
        "index_type": idx_type,
        "reranker_enabled": _reranker_enabled,
        "reranker_loaded": _reranker_model is not None,
        "reranker_loading": _reranker_loading,
    }


def _reranker_model_path() -> str:
    return os.environ.get(
        "QWEN3VL_RERANKER_MODEL_PATH",
        RERANKER_MODEL_MAP.get(_current_model_size, RERANKER_MODEL_MAP["2b"]),
    )


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from qwen3_vl_reranker import Qwen3VLReranker

        path = _reranker_model_path()
        logger.info("Loading reranker model from %s ...", path)
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float16
        _reranker_model = Qwen3VLReranker(
            model_name_or_path=path,
            torch_dtype=dtype,
        )
        logger.info("Reranker model loaded.")
    return _reranker_model


def preload_reranker():
    global _reranker_loading
    _reranker_loading = True
    try:
        get_reranker()
        logger.info("Reranker preloaded successfully.")
    except Exception as e:
        logger.error("Reranker preload failed: %s", e)
        global _reranker_enabled
        _reranker_enabled = False
    finally:
        _reranker_loading = False


def set_reranker_enabled(enabled: bool) -> dict:
    global _reranker_enabled
    _reranker_enabled = enabled
    return {
        "reranker_enabled": _reranker_enabled,
        "reranker_loaded": _reranker_model is not None,
    }


def get_reranker_status() -> dict:
    return {
        "reranker_enabled": _reranker_enabled,
        "reranker_loaded": _reranker_model is not None,
        "reranker_loading": _reranker_loading,
        "reranker_model_path": _reranker_model_path(),
    }


def search_with_rerank(query: str, top_k: int = 5) -> list:
    global _reranker_loading
    if top_k < 1:
        top_k = 1
    with _index_lock:
        if _faiss_index is None or _faiss_index.ntotal == 0:
            return []
        candidate_k = min(top_k * RERANKER_CANDIDATE_MULTIPLIER, _faiss_index.ntotal)
        q_emb = embed_text(query).astype("float32").reshape(1, -1)
        scores, indices = _faiss_index.search(q_emb, candidate_k)
        current_metadata = list(_metadata)

    candidates = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            break
        entry = dict(current_metadata[idx])
        entry["embedding_score"] = float(score)
        entry["score"] = float(score)
        candidates.append((idx, entry))

    if not candidates:
        return []

    _reranker_loading = True
    try:
        reranker = get_reranker()
        documents = []
        valid_candidates = []
        for idx, entry in candidates:
            image_path = entry.get("original_path", "")
            if image_path and Path(image_path).exists():
                documents.append({"image": image_path})
                valid_candidates.append((idx, entry))

        if not documents:
            return [e for _, e in candidates[:top_k]]

        reranker_inputs = {
            "instruction": _instruction,
            "query": {"text": query},
            "documents": documents,
        }
        reranker_scores = reranker.process(reranker_inputs)

        for i, (_idx, entry) in enumerate(valid_candidates):
            if i < len(reranker_scores):
                entry["reranker_score"] = float(reranker_scores[i])
                entry["score"] = float(reranker_scores[i])

        valid_candidates.sort(key=lambda x: x[1].get("reranker_score", 0), reverse=True)
        return [e for _, e in valid_candidates[:top_k]]
    except Exception as e:
        logger.warning("Reranker failed, falling back to embedding scores: %s", e)
        return [e for _, e in candidates[:top_k]]
    finally:
        _reranker_loading = False


def reset():
    global _faiss_index, _metadata
    with _index_lock:
        _faiss_index = None
        _metadata = []
        for f in FAISS_INDEX_FILE, METADATA_FILE:
            if f.exists():
                f.unlink()
    for d in [IMAGES_DIR, THUMBS_DIR]:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
    logger.info("Index and images reset.")
