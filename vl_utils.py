import os
from urllib.parse import urlparse

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
)


def is_image_path(path: str) -> bool:
    if path.startswith(("http://", "https://")):
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in IMAGE_EXTENSIONS


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and len(video) > 0:
        first_elem = video[0]
        if isinstance(first_elem, Image.Image):
            return True
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    return False


def sample_frames(
    frames: list[str | Image.Image], max_segments: int
) -> list[str | Image.Image]:
    duration = len(frames)
    if duration <= max_segments:
        return frames
    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    return [frames[idx] for idx in frame_id_array.tolist()]
