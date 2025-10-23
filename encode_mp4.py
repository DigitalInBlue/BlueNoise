from __future__ import annotations
from typing import List, Dict
import imageio
import os
import subprocess
import shutil
import logging
import numpy as np  
from PIL import Image 

logger = logging.getLogger(__name__)

Frame = Dict[str, object]  # {"image": PIL.Image, "duration_ms": int}

def _to_frame_array(obj):
    if isinstance(obj, Image.Image):
        if obj.mode != "RGB":
            obj = obj.convert("RGB")
        return np.asarray(obj)
    # already an array-like
    arr = np.asarray(obj)
    # enforce 3-channel uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def encode_frames_to_mp4(frames: List[Frame], out_path: str, fps: int = 24, crf: int = 20) -> None:
    if not frames:
        raise ValueError("No frames to encode")

    # Ensure even dimensions
    first = frames[0]["image"]
    try:
        import PIL.Image as _PIL
        w, h = first.size  # type: ignore
    except Exception as e:
        raise ValueError("Invalid frame image") from e

    pad_w = w % 2
    pad_h = h % 2

    tmpdir = os.path.dirname(out_path)
    os.makedirs(tmpdir, exist_ok=True)

    # Prefer calling ffmpeg directly for control
    if shutil.which("ffmpeg"):
        # Write frames to a pipe via imageio to keep things simple
        with imageio.get_writer(
            out_path,
            format="ffmpeg",
            mode="I",
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_params=["-crf", str(crf)] + (["-vf", f"pad=iw+{pad_w}:ih+{pad_h}"] if (pad_w or pad_h) else []),
        ) as writer:
            for fr in frames:
                writer.append_data(_to_frame_array(fr["image"]))
        return

    # Fallback: try imageio-ffmpeg bundled executable (imageio controls this internally)
    with imageio.get_writer(
        out_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(crf)],
    ) as writer:
        for fr in frames:
            writer.append_data(_to_frame_array(fr["image"]))
