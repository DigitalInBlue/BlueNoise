from __future__ import annotations
from typing import Optional, Tuple
from PIL import Image, ImageOps, ImageCms
import os
import math
import logging

logger = logging.getLogger(__name__)

MAX_PREVIEW_DIM = 1024


def ensure_srgb(img: Image.Image) -> Image.Image:
    try:
        if "icc_profile" in img.info:
            icc = img.info.get("icc_profile")
            if icc:
                srgb = ImageCms.createProfile("sRGB")
                src = ImageCms.ImageCmsProfile(BytesIO(icc))  # type: ignore
                return ImageCms.profileToProfile(img, src, srgb, outputMode=img.mode)
    except Exception:
        pass
    return img


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    # Pillow lazy loads; ensure it's loaded now
    img.load()
    return img


def split_rgba(img: Image.Image) -> Tuple[Image.Image, Optional[Image.Image]]:
    if img.mode == "RGBA":
        rgb = img.convert("RGB")
        a = img.split()[3]
        return rgb, a
    if img.mode == "LA":
        rgb = img.convert("RGB")
        a = img.split()[1]
        return rgb, a
    if img.mode == "P":
        # Paletted with transparency -> convert
        alpha = None
        if "transparency" in img.info:
            img = img.convert("RGBA")
            return split_rgba(img)
        return img.convert("RGB"), None
    return img.convert("RGB"), None


def recombine_rgb_alpha(rgb: Image.Image, alpha: Optional[Image.Image]) -> Image.Image:
    if alpha is None:
        return rgb
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def save_png(img: Image.Image, path: str, overwrite: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    img.save(path, format="PNG")


def make_output_path(out_dir: str, in_path: str, variant_name: str, ext: str) -> str:
    base = os.path.splitext(os.path.basename(in_path))[0]
    filename = f"{base}_{variant_name}.{ext}"
    return os.path.join(out_dir, filename)


def downscale_for_preview(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = min(MAX_PREVIEW_DIM / max(w, h), 1.0)
    if scale < 1.0:
        nw = max(1, int(math.floor(w * scale)))
        nh = max(1, int(math.floor(h * scale)))
        return img.resize((nw, nh), Image.LANCZOS)
    return img