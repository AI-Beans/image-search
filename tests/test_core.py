import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAppEndpoints:
    @pytest.fixture
    def client(self):
        os.environ.setdefault("QWEN3VL_MODEL_SIZE", "2b")
        from app import app

        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Image Search" in resp.data

    def test_favicon(self, client):
        resp = client.get("/favicon.svg")
        assert resp.status_code == 200

    def test_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_loaded" in data
        assert "has_gpu" in data
        assert "total_images" in data

    def test_search_empty_query(self, client):
        resp = client.post(
            "/api/search",
            data=json.dumps({"query": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_search_missing_body(self, client):
        resp = client.post("/api/search", content_type="application/json")
        assert resp.status_code == 400

    def test_search_invalid_top_k(self, client):
        resp = client.post(
            "/api/search",
            data=json.dumps({"query": "test", "top_k": -1}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_switch_model_invalid(self, client):
        resp = client.post(
            "/api/switch_model",
            data=json.dumps({"model_size": "4b"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_index_no_files(self, client):
        resp = client.post("/api/index")
        assert resp.status_code == 400

    def test_reranker_status(self, client):
        resp = client.get("/api/reranker/status")
        assert resp.status_code == 200

    def test_stats(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_images" in data


class TestVLUtils:
    def test_is_image_path_jpg(self):
        from vl_utils import is_image_path

        assert is_image_path("photo.jpg") is True
        assert is_image_path("photo.PNG") is True
        assert is_image_path("photo.webp") is True
        assert is_image_path("photo.txt") is False
        assert is_image_path("photo") is False

    def test_is_image_path_url(self):
        from vl_utils import is_image_path

        assert is_image_path("https://example.com/photo.jpg") is True
        assert is_image_path("https://example.com/photo.jpg?w=100") is True
        assert is_image_path("https://example.com/doc.pdf") is False

    def test_is_video_input(self):
        from vl_utils import is_video_input

        assert is_video_input("video.mp4") is True
        assert is_video_input(None) is False
        assert is_video_input([]) is False

    def test_sample_frames_short(self):
        from vl_utils import sample_frames

        frames = ["a", "b", "c"]
        assert sample_frames(frames, 5) == frames

    def test_sample_frames_long(self):
        from vl_utils import sample_frames

        frames = list(range(100))
        result = sample_frames(frames, 10)
        assert len(result) == 10
        assert result[0] == 0
        assert result[-1] == 99


class TestEmbedderHelpers:
    def test_model_map_keys(self):
        from embedder import MODEL_MAP, RERANKER_MODEL_MAP

        assert "8b" in MODEL_MAP
        assert "2b" in MODEL_MAP
        assert "2b" in RERANKER_MODEL_MAP

    def test_get_current_model_info(self):
        from embedder import get_current_model_info

        info = get_current_model_info()
        assert "model_path" in info
        assert "model_name" in info
        assert "model_size" in info

    def test_get_reranker_status(self):
        from embedder import get_reranker_status

        status = get_reranker_status()
        assert "reranker_enabled" in status
        assert "reranker_loaded" in status
        assert "reranker_loading" in status

    def test_search_empty_index(self):
        from embedder import search

        results = search("test query", top_k=5)
        assert results == []

    def test_search_with_rerank_empty_index(self):
        from embedder import search_with_rerank

        results = search_with_rerank("test query", top_k=5)
        assert results == []

    def test_get_stats(self):
        from embedder import get_stats

        stats = get_stats()
        assert "total_images" in stats
        assert "model_loaded" in stats
        assert "has_gpu" in stats
        assert "index_type" in stats
