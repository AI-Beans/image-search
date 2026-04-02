from qwen3_vl_embedding import Qwen3VLEmbedder
import numpy as np
import os
import json
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = Path(__file__).parent / "images" / "uploaded"
THUMBS_DIR = Path(__file__).parent / "static" / "thumbnails"
INDEX_FILE = DATA_DIR / "index.json"

for d in [DATA_DIR, IMAGES_DIR, THUMBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
METADATA_FILE = DATA_DIR / "metadata.json"

_instruction = "Retrieve relevant images based on the user's description."

_model = None
_embeddings = None
_metadata = []


def get_model():
    global _model
    if _model is None:
        model_path = os.environ.get("QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B")
        print(f"Loading model from {model_path} ...")
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float16
        _model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=dtype,
        )
        print("Model loaded.")
    return _model


def _load_index():
    global _embeddings, _metadata
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            _metadata = json.load(f)
    if EMBEDDINGS_FILE.exists():
        _embeddings = np.load(str(EMBEDDINGS_FILE))
    else:
        _embeddings = np.empty((0, 0))


def _save_index():
    if _embeddings is not None and len(_embeddings) > 0:
        np.save(str(EMBEDDINGS_FILE), _embeddings)
    with open(METADATA_FILE, "w") as f:
        json.dump(_metadata, f, ensure_ascii=False, indent=2)


_load_index()


def embed_text(text: str) -> np.ndarray:
    model = get_model()
    result = model.process([{"text": text, "instruction": _instruction}])
    return result.cpu().numpy().flatten()


def embed_image(image_path: str) -> np.ndarray:
    model = get_model()
    result = model.process([{"image": image_path, "instruction": _instruction}])
    return result.cpu().numpy().flatten()


def add_image(image_path: str, filename: str) -> dict:
    global _embeddings, _metadata

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

    if _embeddings is not None and len(_embeddings) > 0:
        _embeddings = np.vstack([_embeddings, emb.reshape(1, -1)])
    else:
        _embeddings = emb.reshape(1, -1)

    _metadata.append(entry)
    _save_index()

    return entry


def add_image_batch(image_paths: list) -> list:
    model = get_model()
    inputs = [{"image": p, "instruction": _instruction} for p in image_paths]
    embeddings = model.process(inputs).cpu().numpy()

    results = []
    new_embs = []
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
        new_embs.append(embeddings[i])
        results.append(entry)

    global _embeddings
    if _embeddings is not None and len(_embeddings) > 0:
        _embeddings = np.vstack([_embeddings] + [e.reshape(1, -1) for e in new_embs])
    else:
        _embeddings = np.vstack([e.reshape(1, -1) for e in new_embs])

    _save_index()
    return results


def search(query: str, top_k: int = 5) -> list:
    global _embeddings, _metadata

    if _embeddings is None or len(_embeddings) == 0:
        return []

    q_emb = embed_text(query)

    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = _embeddings / norms
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

    scores = (normed @ q_norm).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        entry = dict(_metadata[idx])
        entry["score"] = float(scores[idx])
        results.append(entry)

    return results


def get_stats() -> dict:
    return {
        "total_images": len(_metadata),
        "embedding_dim": int(_embeddings.shape[1])
        if _embeddings is not None and len(_embeddings) > 0
        else 0,
        "model_loaded": _model is not None,
        "has_gpu": torch.cuda.is_available(),
    }


def reset():
    global _embeddings, _metadata
    _embeddings = None
    _metadata = []
    for f in EMBEDDINGS_FILE, METADATA_FILE:
        if f.exists():
            f.unlink()
    for d in [IMAGES_DIR, THUMBS_DIR]:
        for f in d.iterdir():
            f.unlink()
