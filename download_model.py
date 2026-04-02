#!/usr/bin/env python3
"""Download Qwen3-VL-Embedding or Qwen3-VL-Reranker model from ModelScope or HuggingFace."""

import os
import sys
import argparse


EMBEDDING_MODELS = {
    "8b": "Qwen/Qwen3-VL-Embedding-8B",
    "2b": "Qwen/Qwen3-VL-Embedding-2B",
}

RERANKER_MODELS = {
    "2b": "Qwen/Qwen3-VL-Reranker-2B",
    "8b": "Qwen/Qwen3-VL-Reranker-8B",
}


def download_model(
    source="modelscope", model_name="Qwen/Qwen3-VL-Embedding-8B", save_dir=None
):
    if save_dir:
        os.environ["QWEN3VL_MODEL_PATH"] = save_dir
    else:
        save_dir = model_name.replace("/", "_")

    if source == "modelscope":
        _download_modelscope(model_name, save_dir)
    else:
        _download_huggingface(model_name, save_dir)

    print(f"\nModel saved to: {save_dir}")
    print(f"Set environment variable to use local model:")
    print(f"  export QWEN3VL_MODEL_PATH={os.path.abspath(save_dir)}")
    print(f"\nOr just run the app (it will auto-detect the local path):")
    print(f"  python3 app.py")
    return save_dir


def _download_modelscope(model_name, save_dir):
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("Installing modelscope...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope import snapshot_download

    print(f"Downloading {model_name} from ModelScope...")
    print("This may take a while (model size ~16GB for 8B).")
    path = snapshot_download(
        model_name, cache_dir=save_dir if save_dir != model_name else None
    )
    print(f"Downloaded to: {path}")
    return path


def _download_huggingface(model_name, save_dir):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} from HuggingFace...")
    print("This may take a while (model size ~16GB for 8B).")
    path = snapshot_download(
        model_name, local_dir=save_dir if save_dir != model_name else None
    )
    print(f"Downloaded to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen3-VL model")
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Download source (default: modelscope, faster in China)",
    )
    parser.add_argument(
        "--type",
        choices=["embedding", "reranker"],
        default="embedding",
        help="Model type (default: embedding)",
    )
    parser.add_argument(
        "--size",
        choices=["2b", "8b"],
        default="8b",
        help="Model size (default: 8b for embedding, 2b for reranker)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save model (default: auto)",
    )
    args = parser.parse_args()

    if args.type == "embedding":
        model_map = EMBEDDING_MODELS
    else:
        model_map = RERANKER_MODELS

    if args.size not in model_map:
        print(f"Error: size '{args.size}' not available for type '{args.type}'")
        print(f"Available sizes for {args.type}: {list(model_map.keys())}")
        sys.exit(1)

    model_name = model_map[args.size]
    download_model(source=args.source, model_name=model_name, save_dir=args.save_dir)
