import os
import sys
import json
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "images" / "uploaded"
DATA_DIR = BASE_DIR / "data"

for d in [UPLOAD_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_LOADED = False
MODEL_LOADING = False
SEARCH_ENGINE = None


def get_engine():
    global SEARCH_ENGINE
    if SEARCH_ENGINE is None:
        from embedder import (
            get_model,
            search,
            add_image_batch,
            add_image,
            get_stats,
            reset,
        )

        get_model()
        SEARCH_ENGINE = True
    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    has_index = (DATA_DIR / "metadata.json").exists()
    total = 0
    if has_index:
        try:
            meta = json.loads((DATA_DIR / "metadata.json").read_text())
            total = len(meta)
        except:
            pass

    import torch
    import faiss

    dim = 0
    faiss_file = DATA_DIR / "faiss.index"
    if faiss_file.exists():
        try:
            idx = faiss.read_index(str(faiss_file))
            dim = idx.d
        except:
            pass

    return jsonify(
        {
            "model_loaded": MODEL_LOADED or SEARCH_ENGINE is not None,
            "model_loading": MODEL_LOADING,
            "has_gpu": torch.cuda.is_available(),
            "device": "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            ),
            "total_images": total,
            "embedding_dim": dim,
            "model_path": os.environ.get(
                "QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B"
            ),
        }
    )


@app.route("/api/load_model", methods=["POST"])
def load_model():
    global MODEL_LOADED, MODEL_LOADING
    if MODEL_LOADED or SEARCH_ENGINE is not None:
        return jsonify({"status": "already_loaded"})
    if MODEL_LOADING:
        return jsonify({"status": "loading"})

    def _load():
        global MODEL_LOADED, MODEL_LOADING
        MODEL_LOADING = True
        try:
            get_engine()
            MODEL_LOADED = True
        except Exception as e:
            print(f"Model load error: {e}")
        finally:
            MODEL_LOADING = False

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return jsonify({"status": "loading_started"})


@app.route("/api/index", methods=["POST"])
def index_images():
    global MODEL_LOADED
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    if not MODEL_LOADED and SEARCH_ENGINE is None:
        try:
            get_engine()
            MODEL_LOADED = True
        except Exception as e:
            return jsonify({"error": f"Model not loaded: {e}"}), 500

    from embedder import add_image

    results = []
    saved_paths = []
    for f in files:
        if not f.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            results.append(
                {
                    "name": f.filename,
                    "status": "skipped",
                    "reason": "unsupported format",
                }
            )
            continue

        ext = Path(f.filename).suffix
        file_id = f"{uuid.uuid4().hex[:8]}{ext}"
        filepath = UPLOAD_DIR / file_id
        f.save(str(filepath))
        saved_paths.append((str(filepath), f.filename))

    if saved_paths:
        try:
            from embedder import add_image_batch

            paths = [p[0] for p in saved_paths]
            batch_results = add_image_batch(paths)
            for br, (_, orig_name) in zip(batch_results, saved_paths):
                br["original_name"] = orig_name
                results.append(br)
        except Exception as e:
            return jsonify({"error": f"Embedding failed: {e}"}), 500

    return jsonify(
        {
            "results": results,
            "total_indexed": len(
                [
                    r
                    for r in results
                    if "status" not in r or r.get("status") != "skipped"
                ]
            ),
        }
    )


@app.route("/api/search", methods=["POST"])
def search_images():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not MODEL_LOADED and SEARCH_ENGINE is None:
        try:
            get_engine()
            MODEL_LOADED = True
        except Exception as e:
            return jsonify({"error": f"Model not loaded: {e}"}), 500

    from embedder import search

    try:
        results = search(query, top_k=top_k)
        return jsonify(
            {"query": query, "results": results, "total_searched": len(results)}
        )
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500


@app.route("/api/stats")
def stats():
    from embedder import get_stats

    try:
        s = get_stats()
        return jsonify(s)
    except:
        return jsonify(
            {
                "total_images": 0,
                "embedding_dim": 0,
                "model_loaded": False,
                "has_gpu": False,
            }
        )


@app.route("/api/reset", methods=["POST"])
def reset_index():
    global SEARCH_ENGINE
    from embedder import reset

    reset()
    SEARCH_ENGINE = None
    return jsonify({"status": "reset complete"})


@app.route("/images/uploaded/<filename>")
def serve_image(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5200))
    model_path = os.environ.get("QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B")
    print(f"\n{'=' * 60}")
    print(f"  Image Semantic Search - Qwen3-VL-Embedding")
    print(f"  Model: {model_path}")
    print(f"  URL: http://localhost:{port}")
    print(f"{'=' * 60}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
