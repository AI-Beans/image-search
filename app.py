import os
import sys
import json
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    send_file,
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "images" / "uploaded"
DATA_DIR = BASE_DIR / "data"

for d in [UPLOAD_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_LOADED = False
MODEL_LOADING = False
SEARCH_ENGINE = None

INDEX_PROGRESS = {"current": 0, "total": 0, "done": False}
INDEX_PROGRESS_LOCK = threading.Lock()
INDEX_TASK_RESULT = {"results": None, "error": None}
INDEX_TASK_LOCK = threading.Lock()


def get_engine():
    global SEARCH_ENGINE
    if SEARCH_ENGINE is None:
        from embedder import get_model

        get_model()
        SEARCH_ENGINE = True
    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.svg")
def favicon():
    return send_file(str(BASE_DIR / "static" / "favicon.svg"), "image/svg+xml")


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

    try:
        import torch

        device = "cuda"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        has_gpu = torch.cuda.is_available()
    except Exception:
        device = "cpu"
        has_gpu = False

    model_info = {}
    try:
        from embedder import get_current_model_info

        model_info = get_current_model_info()
    except Exception:
        pass

    reranker_status = {}
    try:
        from embedder import get_reranker_status

        reranker_status = get_reranker_status()
    except Exception:
        pass

    return jsonify(
        {
            "model_loaded": MODEL_LOADED or SEARCH_ENGINE is not None,
            "model_loading": MODEL_LOADING,
            "has_gpu": has_gpu,
            "device": device,
            "total_images": total,
            "model_path": model_info.get("model_path", "Qwen/Qwen3-VL-Embedding-8B"),
            "model_name": model_info.get("model_name", "Qwen3-VL-Embedding-8B"),
            "model_size": model_info.get("model_size", "8b"),
            "reranker_enabled": reranker_status.get("reranker_enabled", False),
            "reranker_loaded": reranker_status.get("reranker_loaded", False),
            "reranker_loading": reranker_status.get("reranker_loading", False),
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


@app.route("/api/switch_model", methods=["POST"])
def switch_model_endpoint():
    global MODEL_LOADED, MODEL_LOADING, SEARCH_ENGINE
    data = request.get_json() or {}
    size = data.get("model_size", "8b")
    if size not in ("8b", "2b"):
        return jsonify({"error": "Invalid model_size, use '8b' or '2b'"}), 400
    if MODEL_LOADING:
        return jsonify({"error": "Model is currently loading"}), 409

    try:
        from embedder import switch_model

        new_path = switch_model(size)
        MODEL_LOADED = False
        SEARCH_ENGINE = None
        return jsonify(
            {
                "status": "switched",
                "model_size": size,
                "model_path": new_path,
                "message": "Model switched. Call /api/load_model to load the new model.",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/index", methods=["POST"])
def index_images():
    global MODEL_LOADED, MODEL_LOADING, SEARCH_ENGINE, INDEX_PROGRESS, INDEX_TASK_RESULT
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

            with INDEX_PROGRESS_LOCK:
                INDEX_PROGRESS = {
                    "current": 0,
                    "total": len(saved_paths),
                    "done": False,
                }
            with INDEX_TASK_LOCK:
                INDEX_TASK_RESULT = {"results": None, "error": None}

            def progress_callback(current, total):
                with INDEX_PROGRESS_LOCK:
                    INDEX_PROGRESS["current"] = current
                    INDEX_PROGRESS["done"] = current >= total

            def background_task():
                try:
                    paths = [p[0] for p in saved_paths]
                    batch_results = add_image_batch(
                        paths, progress_callback=progress_callback
                    )
                    task_results = []
                    for br, (_, orig_name) in zip(batch_results, saved_paths):
                        br["original_name"] = orig_name
                        task_results.append(br)
                    with INDEX_TASK_LOCK:
                        INDEX_TASK_RESULT["results"] = task_results
                    with INDEX_PROGRESS_LOCK:
                        INDEX_PROGRESS["done"] = True
                except Exception as e:
                    with INDEX_TASK_LOCK:
                        INDEX_TASK_RESULT["error"] = str(e)
                    with INDEX_PROGRESS_LOCK:
                        INDEX_PROGRESS["done"] = True

            t = threading.Thread(target=background_task, daemon=True)
            t.start()
            return jsonify({"status": "processing", "task_id": "index"})
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


@app.route("/api/index/progress")
def index_progress():
    global INDEX_PROGRESS, INDEX_TASK_RESULT
    with INDEX_PROGRESS_LOCK:
        prog = dict(INDEX_PROGRESS)
    with INDEX_TASK_LOCK:
        task_result = (
            dict(INDEX_TASK_RESULT)
            if INDEX_TASK_RESULT["results"] is not None
            else None
        )
        task_error = INDEX_TASK_RESULT["error"]
    result = {"progress": prog}
    if task_result is not None:
        result["results"] = task_result
        result["total_indexed"] = len(
            [r for r in task_result if r.get("status") != "skipped"]
        )
    if task_error:
        result["error"] = task_error
    return jsonify(result)


@app.route("/api/search", methods=["POST"])
def search_images():
    global MODEL_LOADED, MODEL_LOADING, SEARCH_ENGINE
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 5)
    use_reranker = data.get("use_reranker", False)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not MODEL_LOADED and SEARCH_ENGINE is None:
        try:
            get_engine()
            MODEL_LOADED = True
        except Exception as e:
            return jsonify({"error": f"Model not loaded: {e}"}), 500

    import time

    try:
        start_time = time.time()
        if use_reranker:
            from embedder import search_with_rerank

            results = search_with_rerank(query, top_k=top_k)
        else:
            from embedder import search

            results = search(query, top_k=top_k)
        elapsed_ms = (time.time() - start_time) * 1000

        reranker_used = use_reranker and any("reranker_score" in r for r in results)

        return jsonify(
            {
                "query": query,
                "results": results,
                "total_searched": len(results),
                "time_ms": round(elapsed_ms, 2),
                "reranker_used": reranker_used,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500


@app.route("/api/reranker/toggle", methods=["POST"])
def toggle_reranker():
    data = request.get_json() or {}
    enabled = data.get("enabled", False)
    try:
        from embedder import set_reranker_enabled

        result = set_reranker_enabled(enabled)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reranker/status")
def reranker_status():
    try:
        from embedder import get_reranker_status

        return jsonify(get_reranker_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    try:
        from embedder import get_stats, get_current_model_info

        s = get_stats()
        s.update(get_current_model_info())
        return jsonify(s)
    except Exception:
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
