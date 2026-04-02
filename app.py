import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "images" / "uploaded"
DATA_DIR = BASE_DIR / "data"

ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
MAX_TOP_K = 50

for d in [UPLOAD_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_LOADED = False
MODEL_LOADING = False
SEARCH_ENGINE = None

_index_progress = {"current": 0, "total": 0, "done": False, "error": None}
_index_lock = threading.Lock()


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


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    has_index = (DATA_DIR / "metadata.json").exists()
    total = 0
    if has_index:
        try:
            meta = json.loads((DATA_DIR / "metadata.json").read_text())
            total = len(meta)
        except (json.JSONDecodeError, OSError):
            pass

    try:
        import torch

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
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Model load error: %s", e)
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
    global MODEL_LOADED, MODEL_LOADING, SEARCH_ENGINE
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

    saved_paths = []
    skipped = []
    for f in files:
        filename = f.filename or ""
        if not filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            skipped.append(
                {
                    "name": filename,
                    "status": "skipped",
                    "reason": "unsupported format",
                }
            )
            continue
        ext = Path(filename).suffix
        file_id = f"{uuid.uuid4().hex[:8]}{ext}"
        filepath = UPLOAD_DIR / file_id
        f.save(str(filepath))
        saved_paths.append((str(filepath), filename))

    if not saved_paths:
        return jsonify({"results": skipped, "total_indexed": 0})

    global _index_progress
    with _index_lock:
        _index_progress = {
            "current": 0,
            "total": len(saved_paths),
            "done": False,
            "error": None,
        }

    def _background_embed():
        global _index_progress
        try:
            from embedder import add_image_batch

            def _on_progress(current, total):
                with _index_lock:
                    _index_progress["current"] = current
                    _index_progress["total"] = total

            paths = [p[0] for p in saved_paths]
            batch_results = add_image_batch(paths, batch_size=4, progress_callback=_on_progress)
            for br, (_, orig_name) in zip(batch_results, saved_paths, strict=True):
                br["original_name"] = orig_name
            with _index_lock:
                _index_progress["results"] = batch_results + skipped
                _index_progress["total_indexed"] = len(batch_results)
                _index_progress["done"] = True
            logger.info("Indexed %d images", len(batch_results))
        except Exception as e:
            logger.error("Background embedding failed: %s", e)
            with _index_lock:
                _index_progress["error"] = str(e)
                _index_progress["done"] = True

    t = threading.Thread(target=_background_embed, daemon=True)
    t.start()
    return jsonify({"status": "processing", "total": len(saved_paths)})


@app.route("/api/index/progress")
def index_progress():
    with _index_lock:
        prog = dict(_index_progress)
    return jsonify(prog)


@app.route("/api/search", methods=["POST"])
def search_images():
    global MODEL_LOADED, MODEL_LOADING, SEARCH_ENGINE
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    query = data.get("query", "").strip()
    top_k = data.get("top_k", 5)
    use_reranker = data.get("use_reranker", False)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not isinstance(top_k, int) or top_k < 1:
        return jsonify({"error": "top_k must be a positive integer"}), 400
    top_k = min(top_k, MAX_TOP_K)

    if not MODEL_LOADED and SEARCH_ENGINE is None:
        try:
            get_engine()
            MODEL_LOADED = True
        except Exception as e:
            return jsonify({"error": f"Model not loaded: {e}"}), 500

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
        logger.error("Search failed: %s", e)
        return jsonify({"error": f"Search failed: {e}"}), 500


@app.route("/api/reranker/toggle", methods=["POST"])
def toggle_reranker():
    global MODEL_LOADED, MODEL_LOADING, SEARCH_ENGINE
    data = request.get_json() or {}
    enabled = data.get("enabled", False)
    try:
        from embedder import get_reranker_status, set_reranker_enabled

        set_reranker_enabled(enabled)
        status = get_reranker_status()

        if enabled and not status["reranker_loaded"] and not status["reranker_loading"]:
            if not MODEL_LOADED and SEARCH_ENGINE is None:
                try:
                    get_engine()
                    MODEL_LOADED = True
                except Exception as e:
                    return jsonify({"error": f"Embedding model not loaded: {e}"}), 500

            def _load_reranker():
                from embedder import preload_reranker

                preload_reranker()

            t = threading.Thread(target=_load_reranker, daemon=True)
            t.start()
            return jsonify(
                {
                    "reranker_enabled": True,
                    "reranker_loading": True,
                    "reranker_loaded": False,
                }
            )

        return jsonify(
            {
                "reranker_enabled": enabled,
                "reranker_loading": status["reranker_loading"],
                "reranker_loaded": status["reranker_loaded"],
            }
        )
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
        from embedder import get_current_model_info, get_stats

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
    logger.info("Index reset")
    return jsonify({"status": "reset complete"})


@app.route("/images/uploaded/<filename>")
def serve_image(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5200))
    model_path = os.environ.get("QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-8B")
    logger.info(
        "Image Semantic Search - Qwen3-VL-Embedding | Model: %s | URL: http://localhost:%d",
        model_path,
        port,
    )
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
