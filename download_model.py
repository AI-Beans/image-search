#!/usr/bin/env python3
"""Download Qwen3-VL-Embedding model from ModelScope or HuggingFace."""

import os
import sys
import argparse


def download_model(
    source="modelscope", model_name="Qwen/Qwen3-VL-Embedding-8B", save_dir=None
):
    if save_dir:
        os.environ["QWEN3VL_MODEL_PATH"] = save_dir
    else:
        save_dir = model_name

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
    parser = argparse.ArgumentParser(description="Download Qwen3-VL-Embedding model")
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Download source (default: modelscope, faster in China)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-Embedding-8B",
        help="Model name (default: Qwen/Qwen3-VL-Embedding-8B)",
    )
    parser.add_argument(
        "--size",
        choices=["8b", "2b"],
        default="8b",
        help="Model size (default: 8b, use 2b for less VRAM)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save model (default: auto)",
    )
    args = parser.parse_args()

    model_map = {
        "8b": "Qwen/Qwen3-VL-Embedding-8B",
        "2b": "Qwen/Qwen3-VL-Embedding-2B",
    }
    model_name = model_map[args.size]
    download_model(source=args.source, model_name=model_name, save_dir=args.save_dir)
