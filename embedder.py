from qwen3_vl_embedding import Qwen3VLEmbedder
import numpy as np
import os
import json
import faiss
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch

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

_model = None
_faiss_index = None
_metadata = []
_current_model_size = os.environ.get("QWEN3VL_MODEL_SIZE", "8b")


def _model_path():
    return os.environ.get(
        "QWEN3VL_MODEL_PATH", MODEL_MAP.get(_current_model_size, MODEL_MAP["8b"])
    )


def get_model():
    global _model
    if _model is None:
        path = _model_path()
        print(f"Loading model from {path} ...")
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float16
        _model = Qwen3VLEmbedder(
            model_name_or_path=path,
            torch_dtype=dtype,
        )
        print("Model loaded.")
    return _model


def switch_model(size: str) -> str:
    global _model, _faiss_index, _metadata, _current_model_size
    if size not in MODEL_MAP:
        raise ValueError(
            f"Unknown model size: {size}, choose from {list(MODEL_MAP.keys())}"
        )
    _current_model_size = size
    _model = None
    _faiss_index = None
    _metadata = []
    for f in FAISS_INDEX_FILE, METADATA_FILE:
        if f.exists():
            f.unlink()
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
            with open(METADATA_FILE, "r") as f:
                _metadata = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load metadata: {e}")
        _metadata = []
    try:
        if FAISS_INDEX_FILE.exists():
            _faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
            if hasattr(_faiss_index, "hnsw"):
                _faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
    except Exception as e:
        print(f"Warning: failed to load FAISS index: {e}")
        _faiss_index = None


def _save_index():
    try:
        if _faiss_index is not None and _faiss_index.ntotal > 0:
            faiss.write_index(_faiss_index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "w") as f:
            json.dump(_metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save index: {e}")


_load_index()


def embed_text(text: str) -> np.ndarray:
    model = get_model()
    result = model.process([{"text": text, "instruction": _instruction}])
    return result.cpu().to(torch.float32).numpy().flatten()


def embed_image(image_path: str) -> np.ndarray:
    model = get_model()
    result = model.process([{"image": image_path, "instruction": _instruction}])
    return result.cpu().to(torch.float32).numpy().flatten()


def add_image(image_path: str, filename: str) -> dict:
    global _faiss_index, _metadata

    emb = embed_image(image_path)

    img = Image.open(image_path)
    thumb_size = (256, 256)
    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
    thumb_name = f"thumb_{filename}"
    thumb_path = THUMBS_DIR / thumb_name
    img.save(str(thumb_path), "JPEG", quality=85)

    entry = {
        "id": filename,
        "filename": filename,
        "original_path": str(image_path),
        "thumbnail": f"/static/thumbnails/{thumb_name}",
        "indexed_at": datetime.now().isoformat(),
    }

    _ensure_index(emb.shape[0])
    _faiss_index.add(emb.reshape(1, -1).astype("float32"))

    _metadata.append(entry)
    _save_index()

    return entry


def add_image_batch(image_paths: list) -> list:
    model = get_model()
    inputs = [{"image": p, "instruction": _instruction} for p in image_paths]
    embeddings = model.process(inputs).cpu().to(torch.float32).numpy()

    results = []
    for i, path in enumerate(image_paths):
        filename = Path(path).name
        img = Image.open(path)
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        thumb_name = f"thumb_{filename}"
        thumb_path = THUMBS_DIR / thumb_name
        img.save(str(thumb_path), "JPEG", quality=85)

        entry = {
            "id": filename,
            "filename": filename,
            "original_path": str(path),
            "thumbnail": f"/static/thumbnails/{thumb_name}",
            "indexed_at": datetime.now().isoformat(),
        }
        _metadata.append(entry)
        results.append(entry)

    global _faiss_index
    _ensure_index(embeddings.shape[1])
    _faiss_index.add(embeddings)

    _save_index()
    return results


def search(query: str, top_k: int = 5) -> list:
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []

    q_emb = embed_text(query).astype("float32").reshape(1, -1)
    k = min(top_k, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            break
        entry = dict(_metadata[idx])
        entry["score"] = float(score)
        results.append(entry)

    return results


def get_stats() -> dict:
    return {
        "total_images": len(_metadata),
        "embedding_dim": int(_faiss_index.d)
        if _faiss_index is not None and _faiss_index.ntotal > 0
        else 0,
        "model_loaded": _model is not None,
        "has_gpu": torch.cuda.is_available(),
        "index_type": type(_faiss_index).__name__
        if _faiss_index is not None
        else "none",
    }


def reset():
    global _faiss_index, _metadata
    _faiss_index = None
    _metadata = []
    for f in FAISS_INDEX_FILE, METADATA_FILE:
        if f.exists():
            f.unlink()
    for d in [IMAGES_DIR, THUMBS_DIR]:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
