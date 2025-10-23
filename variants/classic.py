from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import random, math, numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageChops, ImageEnhance, ImageStat

from variants_core import variant, Bool, Int, Float, Enum, Color

# ========= size-aware helpers (no pixel assumptions) =========
# All geometry-sensitive parameters use normalized units:
# - *_w: fraction of width  [0..1]
# - *_h: fraction of height [0..1]
# - *_min: fraction of min(width,height) [0..1]
# Density-like params are per-image counts, not per-pixel.

def fit_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")

def px_w(im: Image.Image, frac: float) -> int:
    return max(1, int(round(float(im.width) * max(0.0, frac))))

def px_h(im: Image.Image, frac: float) -> int:
    return max(1, int(round(float(im.height) * max(0.0, frac))))

def px_min(im: Image.Image, frac: float) -> int:
    return max(1, int(round(float(min(im.width, im.height)) * max(0.0, frac))))

def hex_to_rgb(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(c*2 for c in s)
    return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))

def duotone(im, c0, c1):
    g = ImageOps.grayscale(im); g = ImageOps.autocontrast(g)
    g = np.asarray(g, dtype=np.float32) / 255.0
    c0 = np.array(c0, dtype=np.float32); c1 = np.array(c1, dtype=np.float32)
    out = (c0*(1-g[...,None]) + c1*g[...,None]).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")

# -------- scanlines and masks with normalized pitch --------
def add_scanlines(im: Image.Image, opacity: float = 0.2, pitch_h: float = 1/512):
    """
    Draw horizontal scanlines with a pitch specified as a fraction of image height.
    """
    im = im.convert("RGBA")
    period = px_h(im, pitch_h)
    ov = Image.new("RGBA", im.size, (0,0,0,0)); d = ImageDraw.Draw(ov)
    for y in range(0, im.height, period):
        d.line([(0, y), (im.width, y)], fill=(0,0,0,int(255*opacity)))
    return Image.alpha_composite(im, ov).convert("RGB")

def add_slot_mask(im: Image.Image, triad_w: float = 1/512, strength: float = 0.18,
                  scan_opacity: float = 0.25, scan_pitch_h: float = 1/512,
                  gap_every: int = 3, gap_opacity: float = 0.24):
    """
    Vertical RGB triads spaced by a fraction of width.
    """
    w, h = im.size
    triad = max(1, px_w(im, triad_w))
    mask = Image.new("RGB", (w, h), (0,0,0))
    d = ImageDraw.Draw(mask)
    for x in range(0, w, triad):
        cols = [(255,0,0),(0,255,0),(0,0,255)]
        for i, c in enumerate(cols):
            if x+i < w:
                d.line([(x+i, 0), (x+i, h)], fill=c, width=1)
    m = Image.blend(im, mask, strength)
    m = add_scanlines(m, opacity=scan_opacity, pitch_h=scan_pitch_h)
    if gap_every > 0:
        ov = Image.new("RGBA", (w,h), (0,0,0,0))
        d2 = ImageDraw.Draw(ov)
        for x in range(0, w, gap_every*triad):
            d2.line([(x,0),(x,h)], fill=(0,0,0,int(255*gap_opacity)))
        m = Image.alpha_composite(m.convert("RGBA"), ov).convert("RGB")
    return m

# -------- effects rewritten with normalized geometry --------
def rgb_split(im: Image.Image, max_shift_w: float = 0.006, *,
              mode: str = "symmetric", seed: Optional[int] = None):
    img = fit_rgb(im)
    r, g, b = img.split()
    dx = max(0, px_w(img, max_shift_w))
    rng = random.Random((img.width, img.height, seed))
    if mode == "symmetric":
        r_off = -dx
        b_off = +dx
    elif mode == "fixed":
        r_off = dx
        b_off = -dx
    else:  # "random" (legacy)
        r_off = rng.randint(-dx, dx)
        b_off = rng.randint(-dx, dx)
    r = ImageChops.offset(r, r_off, 0)
    b = ImageChops.offset(b, b_off, 0)
    return Image.merge("RGB", (r, g, b))

def jitter_rows(im: Image.Image, strength_w: float = 0.01, band_h: float = 0.01,
                p: float = 0.22, *, seed: Optional[int] = None):
    arr = np.array(im)
    h, w, _ = arr.shape
    band = max(1, px_h(im, band_h))
    dx_max = max(0, px_w(im, strength_w))
    rng = random.Random((w, h, seed))
    for y in range(0, h, band):
        if rng.random() < p:
            dx = rng.randint(-dx_max, dx_max)
            arr[y:y+band, :, :] = np.roll(arr[y:y+band, :, :], dx, axis=1)
    return Image.fromarray(arr).convert("RGB")


def bloom(im: Image.Image, amount: float = 0.6, radius_min: float = 0.006):
    radius = px_min(im, radius_min)
    blur = im.filter(ImageFilter.GaussianBlur(radius))
    return Image.blend(im, blur, amount)

def grain(im: Image.Image, sigma: float = 10):
    arr = np.asarray(im, dtype=np.float32)
    n = np.random.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + n, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def vignette(im: Image.Image, strength: float = 0.6):
    w,h = im.size
    y,x = np.ogrid[:h,:w]
    cx,cy = w/2,h/2
    r = np.sqrt((x-cx)**2+(y-cy)**2)
    mask = (r/r.max())**2
    mask = (1 - strength*mask)[...,None]
    arr = np.asarray(im, dtype=np.float32)
    out = np.clip(arr*mask, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")

def pixelate(im: Image.Image, cell_min: float = 0.01):
    """
    Pixelate using a cell size equal to a fraction of min(width,height).
    """
    cell = px_min(im, cell_min)
    small = im.resize((max(1, im.width//max(1, im.width//cell)),
                       max(1, im.height//max(1, im.height//cell))),
                      Image.Resampling.NEAREST)
    return small.resize(im.size, Image.Resampling.NEAREST)

def barrel_distort(im: Image.Image, k: float = 0.12):
    w, h = im.size
    cx, cy = w/2, h/2
    max_r = math.sqrt(cx*cx + cy*cy)
    src = im.load()
    out = Image.new("RGB", im.size)
    dst = out.load()
    for y in range(h):
        for x in range(w):
            nx = (x - cx) / max_r
            ny = (y - cy) / max_r
            r = math.sqrt(nx*nx + ny*ny)
            if r == 0:
                sx, sy = cx, cy
            else:
                factor = 1 + k * (r*r)
                sx = cx + nx / factor * max_r
                sy = cy + ny / factor * max_r
            if 0 <= sx < w and 0 <= sy < h:
                dst[x, y] = src[int(sx), int(sy)]
    return out

def dmrc(im: Image.Image, pitch_min: float = 0.01, dot_scale: float = 0.95, gamma: float = 0.9, bg=(248, 245, 240)):
    """
    Round color halftone with pitch set as a fraction of min(width,height).
    """
    im = im.convert("RGB")
    w, h = im.size
    pitch = px_min(im, pitch_min)
    out = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(out)
    for y in range(0, h, pitch):
        for x in range(0, w, pitch):
            box = (x, y, min(x+pitch, w), min(y+pitch, h))
            stat = ImageStat.Stat(im.crop(box))
            r, g, b = [c for c in stat.mean]
            L = (0.2126*r + 0.7152*g + 0.0722*b) / 255.0
            radius = (1.0 - pow(L, gamma)) * (pitch * 0.5 * dot_scale)
            if radius <= 0.3:
                continue
            cx = x + pitch * 0.5
            cy = y + pitch * 0.5
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(int(r), int(g), int(b)))
    return out

# --- Perceptual helpers for high-res images ---

def add_scanlines_perceptual_deprecated(im: Image.Image, opacity: float, pitch_h: float) -> Image.Image:
    """
    Perceptual scanlines:
      - pitch_h is fraction of height
      - line thickness scales with pitch and clamps in px to avoid 1px-only at 6k
    """
    im_rgba = im.convert("RGBA")
    w, h = im_rgba.size
    period = max(2, px_h(im_rgba, pitch_h))                 # e.g., 6k * (1/512) ~= 12 px
    # thickness ~8% of pitch, clamped [1..max(3, period//6)] to avoid razor-thin lines at 6k
    thick = max(1, min(max(3, period // 6), int(round(period * 0.08))))
    a = int(255 * max(0.0, min(1.0, opacity)))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # Center the first line to reduce moiré when downsampling
    y0 = (period - thick) // 2
    for y in range(y0, h, period):
        d.rectangle((0, y, w, y + thick - 1), fill=(0, 0, 0, a))
    return Image.alpha_composite(im_rgba, overlay).convert("RGB")

# put near the other helpers
def add_scanlines_perceptual_balanced(im: Image.Image, opacity: float, pitch_h: float) -> Image.Image:
    img = fit_rgb(im)
    w, h = img.size
    period = max(2, px_h(img, pitch_h))
    thick  = max(1, min(max(3, period // 6), int(round(period * 0.08))))
    # density compensation so perceived darkness is stable across resolutions
    fill   = thick / float(period)
    a_in   = float(max(0.0, min(1.0, opacity)))
    a_eff  = 1.0 - pow(1.0 - a_in, 1.0 / max(1e-6, fill))

    # randomized phase to avoid content lock-in
    phase = (hash((w, h)) % period) // 2

    m = np.zeros((h,), dtype=np.uint8)
    y = phase
    while y < h:
        y2 = min(h, y + thick); m[y:y2] = 1; y += period
    M = m[:, None].astype(np.float32)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr_lin = np.power(arr, 2.2)
    out_lin = arr_lin * (1.0 - a_eff * M)[..., None]
    out = np.clip(np.power(out_lin, 1/2.2) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

# micro-sharpen that scales gently with resolution
def sharpen_perceptual(im: Image.Image, amount: float = 0.6, radius_min: float = 0.002) -> Image.Image:
    r = max(1, px_min(im, radius_min))
    if amount <= 0.0 or r <= 0:
        return im
    return im.filter(ImageFilter.UnsharpMask(radius=r, percent=int(150*amount), threshold=0))

def bloom_perceptual(im: Image.Image, amount: float, radius_min: float,
                     hi_thresh: float = 0.70, knee: float = 0.15) -> Image.Image:
    """
    Perceptual bloom:
      - radius scales sublinearly with resolution (sqrt) to hold apparent softness
      - only bright regions contribute (soft knee)
    """
    amount = max(0.0, min(1.0, float(amount)))
    if amount == 0.0:
        return im

    w, h = im.size
    mdim = float(min(w, h))
    # base radius from normalized min-dim, then sublinear growth with sqrt scale
    r_base = max(1.0, float(px_min(im, max(0.0, radius_min))))
    r = int(round(min(512.0, r_base * math.sqrt(mdim / 1024.0))))  # e.g., at 6k => ~2.4× r_base
    if r < 2:
        r = 2

    src = fit_rgb(im)
    arr = np.asarray(src, dtype=np.float32)
    # luminance for highlight mask
    lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) / 255.0
    # soft knee
    t0 = max(0.0, min(1.0, hi_thresh))
    k = max(1e-6, knee)
    mask = np.clip((lum - t0) / (1.0 - t0), 0.0, 1.0)
    mask = np.power(mask, 1.5)                                 # slightly concentrate on the top end
    mask_img = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")

    # blur only the highlights via masked pre-blend to avoid global haze
    # Convert highlights to isolated layer, blur, then composite back
    hi = Image.composite(src, Image.new("RGB", src.size, (0, 0, 0)), mask_img)
    hi_blur = hi.filter(ImageFilter.GaussianBlur(radius=r))

    # mix with original, weighting by mask and amount
    hb = np.asarray(hi_blur, dtype=np.float32)
    out = np.clip(arr * (1.0 - amount * mask[..., None]) + hb * (amount * mask[..., None]), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# ========= variants (ported, with GUI parameters) =========

# --- GLITCH: preserve dark bands at high-res
@variant(
    name="glitch",
    options={
        "scan_alpha": Float(0.22, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "jitter_strength_w": Float(0.01, 0.0, 0.25, 0.001),
        "jitter_band_h": Float(0.012, 0.001, 0.25, 0.001),
        "jitter_p": Float(0.18, 0.0, 1.0, 0.01),
        "max_rgb_shift_w": Float(0.006, 0.0, 0.05, 0.0005),
    },
)
def _v_glitch(im: Image.Image, *, scan_alpha: float, scan_pitch_h: float, jitter_strength_w: float, jitter_band_h: float, jitter_p: float, max_rgb_shift_w: float):
    base = rgb_split(im, max_shift_w=max_rgb_shift_w, mode="symmetric", seed=0)
    # apply scanlines before jitter so bands don’t get smeared out of existence
    base = add_scanlines_perceptual_balanced(base, scan_alpha, scan_pitch_h)
    return jitter_rows(base, strength_w=jitter_strength_w, band_h=jitter_band_h, p=jitter_p, seed=0)


@variant(
    name="pixel",
    options={"cell_min": Float(0.01, 0.001, 0.25, 0.001)},  # fraction of min dimension
)
def _v_pixel(im: Image.Image, *, cell_min: float):
    return pixelate(im, cell_min=cell_min)

@variant(
    name="vignette_grain",
    options={
        "sigma": Float(10.0, 0.0, 50.0, 0.1),
        "strength": Float(0.6, 0.0, 1.0, 0.01),
    },
)
def _v_vignette_grain(im: Image.Image, *, sigma: float, strength: float):
    return grain(vignette(im, strength=strength), sigma)

# @variant(
#     name="dot_matrix_round_color",
#     options={
#         "pitch_min": Float(0.01, 0.002, 0.1, 0.001),  # fraction of min dimension
#         "dot_scale": Float(0.95, 0.1, 2.0, 0.01),
#         "gamma": Float(0.9, 0.1, 3.0, 0.01),
#         "bg": Color("#f8f5f0"),
#     },
# )
# def _v_dot_matrix_round_color(im: Image.Image, *, pitch_min: float, dot_scale: float, gamma: float, bg: str):
#     return dmrc(fit_rgb(im), pitch_min=pitch_min, dot_scale=dot_scale, gamma=gamma, bg=hex_to_rgb(bg))

@variant(
    name="pix_patch_gray",
    options={
        "seed": Int(1, 0, 1_000_000, 1),
        "min_frac": Float(0.25, 0.05, 0.9, 0.01),
        "max_frac": Float(0.33, 0.06, 0.95, 0.01),
        "down_cell_min": Float(0.02, 0.005, 0.25, 0.001),  # pixelation cell as fraction of min
        "grayscale": Bool(True),
    },
)
def _v_pix_patch_gray(im: Image.Image, *, seed: int, min_frac: float, max_frac: float, down_cell_min: float, grayscale: bool):
    rnd = random.Random(seed)
    out = im.copy(); w,h = out.size
    pw = rnd.randint(max(1, int(w*min_frac)), max(1, int(w*max_frac)))
    ph = rnd.randint(max(1, int(h*min_frac)), max(1, int(h*max_frac)))
    x0 = rnd.randint(0, max(1, w - pw)); y0 = rnd.randint(0, max(1, h - ph))
    patch = out.crop((x0,y0,x0+pw,y0+ph))
    if grayscale:
        patch = ImageOps.grayscale(patch).convert("RGB")
    cell = px_min(patch, down_cell_min)
    small = patch.resize(
        (max(1, patch.width // max(1, patch.width // cell)),
         max(1, patch.height // max(1, patch.height // cell))),
        Image.Resampling.NEAREST,
    )
    patch = small.resize((pw, ph), Image.Resampling.NEAREST)
    out.paste(patch, (x0,y0))
    return out

@variant(
    name="noir_threshold",
    options={"threshold": Int(128, 0, 255, 1)},
)
def _v_noir_threshold(im: Image.Image, *, threshold: int):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    return g.point(lambda v: 255 if v >= threshold else 0).convert("RGB")

# @variant(
#     name="dot_matrix_round",
#     options={"cell_min": Float(0.012, 0.002, 0.1, 0.001)},
# )
# def _v_dot_matrix_round(im: Image.Image, *, cell_min: float):
#     g = ImageOps.grayscale(im); g = ImageOps.autocontrast(g)
#     w,h = g.size
#     out = Image.new("RGB", (w,h), "black"); draw = ImageDraw.Draw(out)
#     arr = np.asarray(g, dtype=np.float32)/255.0
#     cell = px_min(im, cell_min)
#     for y in range(0,h,cell):
#         for x in range(0,w,cell):
#             patch = arr[y:min(y+cell,h), x:min(x+cell,w)]
#             if patch.size == 0: continue
#             lum = patch.mean()
#             r = int((1.0 - lum) * (cell/2))
#             if r > 0:
#                 cx, cy = x + cell//2, y + cell//2
#                 draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(240,240,240))
#     return out

# @variant(
#     name="shadow_red_bleed",
#     options={
#         "red": Color("#7a0e0e"),
#         "power": Float(1.7, 0.1, 4.0, 0.05),
#         "mix": Float(0.22, 0.0, 1.0, 0.01),
#     },
# )
# def _v_shadow_red_bleed(im: Image.Image, *, red: str, power: float, mix: float):
#     red_t = hex_to_rgb(red)
#     g = ImageOps.grayscale(im); m = np.asarray(g, dtype=np.float32)/255.0
#     shadow = (1.0 - m) ** power
#     base = np.asarray(fit_rgb(im), dtype=np.float32)
#     tint = np.zeros_like(base); tint[...,0] = red_t[0]; tint[...,1] = red_t[1]; tint[...,2] = red_t[2]
#     tint *= shadow[...,None]
#     out = np.clip(base*(1.0 - mix) + tint*mix, 0, 255).astype(np.uint8)
#     return Image.fromarray(out).convert("RGB")

# @variant(
#     name="mono_red_burnt",
#     options={
#         "red": Color("#7a0e0e"),
#         "vignette": Float(0.7, 0.0, 1.0, 0.01),
#         "leak_blur_min": Float(0.06, 0.0, 0.6, 0.005),  # blur radius as fraction of min
#         "leak_blend": Float(0.18, 0.0, 1.0, 0.01),
#         "grain_sigma": Float(6.0, 0.0, 30.0, 0.1),
#         "contrast": Float(1.35, 0.5, 3.0, 0.01),
#         "leak_count": Int(3, 0, 12, 1),
#     },
# )
# def _v_mono_red_burnt(im: Image.Image, *, red: str, vignette: float, leak_blur_min: float, leak_blend: float, grain_sigma: float, contrast: float, leak_count: int):
#     red_t = hex_to_rgb(red)
#     dt = duotone(im, (5,5,8), red_t)
#     dt = ImageEnhance.Contrast(dt).enhance(contrast)
#     dt = vignette_fn(dt, strength=vignette) if (vignette is not None) else dt
# 
#     leak = Image.new("RGB", dt.size, (0,0,0)); d = ImageDraw.Draw(leak)
#     for _ in range(max(0, leak_count)):
#         x0 = random.randint(-int(dt.width*0.2), int(dt.width*0.2))
#         y0 = random.randint(-int(dt.height*0.2), int(dt.height*0.2))
#         r  = random.randint(max(1, dt.width//3), dt.width)
#         col = (min(255, red_t[0]+80), red_t[1]//3, red_t[2]//3)
#         d.ellipse([x0-r, y0-r, x0+r, y0+r], fill=col)
#     rb = px_min(dt, leak_blur_min)
#     if rb > 0:
#         leak = leak.filter(ImageFilter.GaussianBlur(radius=rb))
#     out = Image.blend(dt, leak, leak_blend)
#     return grain(out, sigma=grain_sigma)

def vignette_fn(im: Image.Image, strength: float) -> Image.Image:
    return vignette(im, strength=strength)

@variant(
    name="scan_jitter_gif",
    options={
        "n_frames": Int(8, 1, 120, 1),
        "dur_ms": Int(90, 10, 500, 5),
        "shift_base_w": Float(0.002, 0.0, 0.05, 0.0005),
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
    },
    animated=True, preview_safe=False,
)
def _v_scan_jitter_gif(im: Image.Image, *, n_frames: int, dur_ms: int, shift_base_w: float, scan_opacity: float, scan_pitch_h: float):
    frames: list[dict[str, Any]] = []
    base = fit_rgb(im)
    for i in range(int(n_frames)):
        f = rgb_split(base, max_shift_w=(shift_base_w + (0.0005 if (i % 2) else 0.0)))
        f = add_scanlines(f, scan_opacity + 0.02*(i%2), scan_pitch_h)
        frames.append({"image": f, "duration_ms": int(dur_ms)})
    return frames

# --- AMBER MONITOR: stronger visibility at 6k with density-comp scanlines and mild sharpen
@variant(
    name="amber_monitor",
    options={
        "contrast": Float(1.15, 0.5, 3.0, 0.01),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "bloom_amount": Float(0.35, 0.0, 1.0, 0.01),
        "bloom_radius_min": Float(0.006, 0.0, 0.08, 0.0005),
        "vignette": Float(0.40, 0.0, 1.0, 0.01),
    },
)
def _v_amber_monitor(im: Image.Image, *, contrast: float, scan_opacity: float, scan_pitch_h: float, bloom_amount: float, bloom_radius_min: float, vignette: float):
    amber_dark = (5, 3, 0); amber_light = (255, 170, 40)
    dt = duotone(im, amber_dark, amber_light)
    dt = ImageEnhance.Contrast(dt).enhance(contrast)
    dt = add_scanlines_perceptual_balanced(dt, scan_opacity, scan_pitch_h)
    dt = bloom_perceptual(dt, amount=bloom_amount, radius_min=bloom_radius_min, hi_thresh=0.72, knee=0.15)
    dt = sharpen_perceptual(dt, amount=0.5, radius_min=0.0015)
    return vignette_fn(dt, strength=vignette)

@variant(
    name="amber_monitor_strong",
    options={
        "contrast": Float(1.25, 0.5, 3.0, 0.01),
        "color_enh": Float(0.9, 0.0, 3.0, 0.01),
        "scan_opacity": Float(0.35, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "bloom_amount": Float(0.42, 0.0, 1.0, 0.01),
        "bloom_radius_min": Float(0.008, 0.0, 0.12, 0.0005),
        "vignette": Float(0.48, 0.0, 1.0, 0.01),
    },
)
def _v_amber_monitor_strong(im: Image.Image, *, contrast: float, color_enh: float, scan_opacity: float, scan_pitch_h: float, bloom_amount: float, bloom_radius_min: float, vignette: float):
    amber_dark = (4, 2, 0); amber_light = (255, 180, 60)
    dt = duotone(im, amber_dark, amber_light)
    dt = ImageEnhance.Contrast(dt).enhance(contrast)
    dt = ImageEnhance.Color(dt).enhance(color_enh)
    dt = add_scanlines_perceptual_balanced(dt, opacity=scan_opacity, pitch_h=scan_pitch_h)
    dt = bloom_perceptual(dt, amount=bloom_amount, radius_min=bloom_radius_min, hi_thresh=0.72, knee=0.15)
    dt = sharpen_perceptual(dt, amount=0.6, radius_min=0.0015)
    return vignette_fn(dt, strength=vignette)

@variant(
    name="gray_color_glitch",
    options={
        "edge_sharp": Float(2.0, 0.0, 5.0, 0.1),
        "edge_thresh": Int(40, 0, 255, 1),
        "mix": Float(0.65, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "max_shift_w": Float(0.01, 0.0, 0.08, 0.0005),      # frac of width
        "jitter_strength_w": Float(0.012, 0.0, 0.25, 0.001),# frac of width
        "jitter_band_h": Float(0.012, 0.001, 0.25, 0.001),  # frac of height
        "jitter_p": Float(0.18, 0.0, 1.0, 0.01),
    },
)
def _v_gray_color_glitch(im: Image.Image, *, edge_sharp: float, edge_thresh: int, mix: float, scan_opacity: float, scan_pitch_h: float, max_shift_w: float, jitter_strength_w: float, jitter_band_h: float, jitter_p: float):
    rgbImage = fit_rgb(im)
    base = ImageOps.grayscale(rgbImage).convert("RGB")
    color = rgb_split(rgbImage, max_shift_w=max_shift_w, mode="symmetric", seed=0)
    edges = ImageEnhance.Sharpness(rgbImage).enhance(edge_sharp).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    edges = ImageOps.grayscale(edges)
    edges = ImageEnhance.Brightness(edges).enhance(1.4)
    edges = edges.point(lambda v: 255 if v > edge_thresh else 0)
    out = Image.blend(base, color, mix)
    out = add_scanlines_perceptual_balanced(out, scan_opacity, scan_pitch_h)

    # decouple row jitter scale from scanline period to avoid moiré lock
    out = jitter_rows(out, strength_w=jitter_strength_w, band_h=jitter_band_h, p=jitter_p, seed=0)
    return out


@variant(
    name="gray_contrast",
    options={
        "cutoff": Int(2, 0, 20, 1),
        "contrast": Float(1.6, 0.2, 4.0, 0.01),
        "brightness": Float(0.98, 0.2, 3.0, 0.01),
        "usm_radius_min": Float(0.004, 0.0, 0.08, 0.0005),  # frac of min dim
        "usm_percent": Int(130, 0, 500, 1),
        "usm_threshold": Int(2, 0, 64, 1),
    },
)
def _v_gray_contrast(im: Image.Image, *, cutoff: int, contrast: float, brightness: float, usm_radius_min: float, usm_percent: int, usm_threshold: int):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=cutoff)
    g = ImageEnhance.Contrast(g).enhance(contrast)
    g = ImageEnhance.Brightness(g).enhance(brightness)
    im2 = g.convert("RGB")
    r_px = px_min(im2, usm_radius_min)
    if r_px > 0 and usm_percent > 0:
        im2 = im2.filter(ImageFilter.UnsharpMask(radius=r_px, percent=usm_percent, threshold=usm_threshold))
    return im2



@variant(
    name="gray_contrast_redshadows",
    options={
        "red": Color("#7a0e0e"),
        "shadow_power": Float(1.6, 0.1, 4.0, 0.05),
        "base_mix": Float(0.92, 0.0, 1.0, 0.01),
        "tint_mix": Float(0.20, 0.0, 1.0, 0.01),
    },
)
def _v_gray_contrast_redshadows(im: Image.Image, *, red: str, shadow_power: float, base_mix: float, tint_mix: float):
    red_t = hex_to_rgb(red)
    g = ImageOps.grayscale(im); g = ImageOps.autocontrast(g, cutoff=2)
    g = ImageEnhance.Contrast(g).enhance(1.7)
    g_rgb = g.convert("RGB")
    m = np.asarray(g, dtype=np.float32) / 255.0
    shadow = (1.0 - m) ** shadow_power
    tint = np.zeros((g.height, g.width, 3), dtype=np.float32); tint[...,0]=red_t[0]; tint[...,1]=red_t[1]; tint[...,2]=red_t[2]
    tint *= shadow[...,None]
    base = np.asarray(g_rgb, dtype=np.float32)
    out = np.clip(base*base_mix + tint*tint_mix, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")



@variant(
    name="posterize_redblack",
    options={
        "bits": Int(3, 1, 8, 1),
        "red": Color("#a01414"),
        "contrast": Float(1.2, 0.2, 3.0, 0.01),
    },
)
def _v_posterize_redblack(im: Image.Image, *, bits: int, red: str, contrast: float):
    g = ImageOps.grayscale(im)
    g = ImageOps.posterize(g.convert("L"), max(1, int(bits)))
    rb = ImageOps.colorize(g, black="black", white=f"{red}")
    return ImageEnhance.Contrast(rb).enhance(contrast)



@variant(
    name="neon_edges",
    options={
        "black": Color("#050508"),
        "white": Color("#00E5FF"),
        "mix": Float(0.65, 0.0, 1.0, 0.01),
        "contrast": Float(2.0, 0.0, 5.0, 0.1),
    },
)
def _v_neon_edges(im: Image.Image, *, black: str, white: str, mix: float, contrast: float):
    e = im.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(contrast)
    e = ImageOps.colorize(ImageOps.grayscale(e), black=black, white=white)
    return Image.blend(ImageOps.grayscale(im).convert("RGB"), e, mix)


@variant(
    name="neon_edges_amber",
    options={"tint": Color("#ffb43c"), "mix": Float(0.70, 0.0, 1.0, 0.01), "scan_opacity": Float(0.20, 0.0, 1.0, 0.01), "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096)},
)
def _v_neon_edges_amber(im: Image.Image, *, tint: str, mix: float, scan_opacity: float, scan_pitch_h: float):
    return _neon_edges_tint_custom(im, hex_to_rgb(tint), mix, scan_opacity, scan_pitch_h)

@variant(
    name="neon_edges_red",
    options={"tint": Color("#c81e1e"), "mix": Float(0.70, 0.0, 1.0, 0.01), "scan_opacity": Float(0.20, 0.0, 1.0, 0.01), "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096)},
)
def _v_neon_edges_red(im: Image.Image, *, tint: str, mix: float, scan_opacity: float, scan_pitch_h: float):
    return _neon_edges_tint_custom(im, hex_to_rgb(tint), mix, scan_opacity, scan_pitch_h)

def _neon_edges_tint_custom(im: Image.Image, tint_rgb: Tuple[int,int,int], mix: float, scan_opacity: float, scan_pitch_h: float):
    e = im.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(2.2)
    e = ImageOps.grayscale(e)
    e = ImageOps.colorize(e, black="#050508", white=f"rgb{tint_rgb}")
    base = ImageOps.grayscale(im).convert("RGB")
    out = Image.blend(base, e, mix)
    return add_scanlines_perceptual_balanced(out, scan_opacity, scan_pitch_h)

@variant(
    name="data_streaks",
    options={
        "count": Int(4, 0, 200, 1),
        "band_min_h": Float(0.003, 0.001, 0.1, 0.0005),  # frac of height
        "band_max_h": Float(0.02,  0.002, 0.2, 0.0005),
        "dx_max_w":   Float(0.05,  0.0,   0.5,  0.001),   # frac of width
    },
)
def _v_data_streaks(im: Image.Image, *, count: int, band_min_h: float, band_max_h: float, dx_max_w: float):
    arr = np.asarray(im).copy(); h,w,_ = arr.shape
    band_min = max(1, px_h(im, band_min_h)); band_max = max(band_min+1, px_h(im, band_max_h))
    dx_max = px_w(im, dx_max_w)
    n = max(0, count if count>0 else max(4, h//120))
    for _ in range(n):
        y = random.randint(0, h-2)
        band = random.randint(band_min, band_max)
        dx = random.randint(-dx_max, dx_max)
        arr[y:y+band,:,:] = np.roll(arr[y:y+band,:,:], dx, axis=1)
    return Image.fromarray(arr).convert("RGB")


# --- CRT MASK: triad thickness scales with triad pitch; slight soften to avoid aliasing
@variant(
    name="crt_mask",
    options={
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "triad_w": Float(1/512, 1/4096, 1/64, 1/4096),
        "mask_strength": Float(0.08, 0.0, 1.0, 0.01),
    },
)
def _v_crt_mask(im: Image.Image, *, scan_opacity: float, scan_pitch_h: float, triad_w: float, mask_strength: float):
    im2 = add_scanlines_perceptual_balanced(im, scan_opacity, scan_pitch_h)
    w, h = im2.size
    triad = max(1, px_w(im2, triad_w))
    lw = max(1, int(round(triad * 0.33)))  # thickness scales with triad
    phase = (hash((w, h, triad)) % triad) // 2

    mask = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(mask)
    for x in range(phase, w, triad):
        for i, c in enumerate(((255,0,0),(0,255,0),(0,0,255))):
            xi = x + i*lw
            if xi < w:
                d.line([(xi, 0), (xi, h)], fill=c, width=lw)
    # feather one pixel to avoid hard moiré at 6k
    mask = mask.filter(ImageFilter.GaussianBlur(0.5))
    return Image.blend(im2, mask, mask_strength)


@variant(
    name="pixel_sort",
    options={
        "step_min_h": Float(0.004, 0.001, 0.1, 0.0005),  # fraction of height
        "step_max_h": Float(0.02, 0.002, 0.5, 0.0005),
        "prob": Float(0.3, 0.0, 1.0, 0.01),
    },
)
def _v_pixel_sort_rows(im: Image.Image, *, step_min_h: float, step_max_h: float, prob: float):
    arr = np.asarray(fit_rgb(im)).copy()
    h,w,_ = arr.shape
    step_min = max(1, px_h(im, step_min_h))
    step_max = max(step_min + 1, px_h(im, step_max_h))
    y = 0
    while y < h:
        step = random.randint(step_min, step_max)
        if random.random() < prob:
            row = arr[y]
            keys = row.mean(axis=1).argsort()
            arr[y] = row[keys]
        y += step
    return Image.fromarray(arr).convert("RGB")


@variant(
    name="halftone",
    options={"blur_min": Float(0.003, 0.0, 0.05, 0.0005)},  # frac of min dim
)
def _v_halftone(im: Image.Image, *, blur_min: float):
    g = ImageOps.grayscale(im)
    r = px_min(g, blur_min)
    if r > 0:
        g = g.filter(ImageFilter.GaussianBlur(radius=r))
    arr = np.asarray(g, dtype=np.float32)/255.0
    bayer = (1/64.0)*np.array([
        [0,48,12,60,3,51,15,63],[32,16,44,28,35,19,47,31],
        [8,56,4,52,11,59,7,55],[40,24,36,20,43,27,39,23],
        [2,50,14,62,1,49,13,61],[34,18,46,30,33,17,45,29],
        [10,58,6,54,9,57,5,53],[42,26,38,22,41,25,37,21],
    ], dtype=np.float32)
    h,w = arr.shape; tiled = np.tile(bayer, (h//8+1, w//8+1))[:h,:w]
    dots = (arr > tiled).astype(np.uint8)*255
    return Image.merge("RGB", (Image.fromarray(dots),)*3)

@variant(
    name="cga_palette",
    options={"dither": Bool(True)},
)
def _v_cga_palette(im: Image.Image, *, dither: bool):
    palette = [
        (0,0,0),(0,0,170),(0,170,0),(0,170,170),
        (170,0,0),(170,0,170),(170,85,0),(170,170,170),
        (85,85,85),(85,85,255),(85,255,85),(85,255,255),
        (255,85,85),(255,85,255),(255,255,85),(255,255,255)
    ]
    pal = []; 
    for r,g,b in palette: pal.extend([r,g,b])
    pal += [0,0,0]*(256-len(palette))
    pimg = Image.new("P", (1,1)); pimg.putpalette(pal)
    q = fit_rgb(im).quantize(palette=pimg, dither=Image.FLOYDSTEINBERG if dither else Image.NONE)
    return q.convert("RGB")



@variant(
    name="blueprint_cyan",
    options={
        "dark": Color("#001015"),
        "light": Color("#00e0ff"),
        "bloom_amount": Float(0.35, 0.0, 1.0, 0.01),
        "bloom_radius_min": Float(0.008, 0.0, 0.12, 0.0005),
        "vignette": Float(0.35, 0.0, 1.0, 0.01),
    },
)
def _v_blueprint_cyan(im: Image.Image, *, dark: str, light: str, bloom_amount: float, bloom_radius_min: float, vignette: float):
    inv = ImageOps.invert(fit_rgb(im))
    out = duotone(inv, hex_to_rgb(dark), hex_to_rgb(light))
    out = bloom_perceptual(out, amount=bloom_amount, radius_min=bloom_radius_min)
    return vignette_fn(out, strength=vignette)

@variant(
    name="wireframe_edges_red",
    options={
        "edge_color": Color("#E02020"),
        "base_brightness": Float(0.6, 0.0, 3.0, 0.01),
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
    },
)
def _v_wireframe_edges_red(im: Image.Image, *, edge_color: str, base_brightness: float, scan_opacity: float, scan_pitch_h: float):
    base = ImageOps.grayscale(im).convert("RGB")
    base = ImageEnhance.Brightness(base).enhance(base_brightness)
    edges = ImageEnhance.Sharpness(im).enhance(2).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.grayscale(edges); edges = ImageOps.autocontrast(edges)
    edges = ImageOps.colorize(edges, black="black", white=edge_color)
    out = ImageChops.screen(base, edges)
    return add_scanlines_perceptual_balanced(out, scan_opacity, scan_pitch_h)

@variant(
    name="cinema_21x9",
    options={
        "bar_frac": Float(0.04, 0.0, 0.2, 0.005),
        "grain_sigma": Float(6.0, 0.0, 30.0, 0.1),
    },
)
def _v_cinema_21x9(im: Image.Image, *, bar_frac: float, grain_sigma: float):
    img = fit_rgb(im); w,h = img.size
    target_ratio = 21/9; cur_ratio = w/h
    if cur_ratio > target_ratio:
        new_w = int(h*target_ratio); x0 = (w - new_w)//2; img = img.crop((x0, 0, x0+new_w, h))
    else:
        new_h = int(w/target_ratio); y0 = (h - new_h)//2; img = img.crop((0, y0, w, y0+new_h))
    bar_h = max(1, int(img.height * max(0.0, min(0.2, bar_frac)) * 0.5))
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,img.width,bar_h], fill=(0,0,0))
    d.rectangle([0,img.height-bar_h,img.width,img.height], fill=(0,0,0))
    return grain(img, sigma=grain_sigma)

@variant(
    name="copper_duotone",
    options={
        "dark": Color("#0a0a0d"),
        "light": Color("#b87333"),
        "contrast": Float(1.12, 0.5, 3.0, 0.01),
    },
)
def _v_copper_duotone(im: Image.Image, *, dark: str, light: str, contrast: float):
    out = duotone(im, hex_to_rgb(dark), hex_to_rgb(light))
    return ImageEnhance.Contrast(out).enhance(contrast)



# @variant(
#     name="vhs_tape",
#     options={
#         "desat": Float(0.85, 0.0, 2.0, 0.01),
#         "blur": Float(0.6, 0.0, 10.0, 0.05),
#         "jitter_strength": Int(18, 0, 128, 1),
#         "jitter_band": Int(6, 1, 64, 1),
#         "jitter_p": Float(0.12, 0.0, 1.0, 0.01),
#         "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
#         "scan_period": Int(2, 1, 8, 1),
#         "grain_sigma": Float(8.0, 0.0, 30.0, 0.1),
#         "brightness": Float(1.04, 0.0, 3.0, 0.01),
#         "vignette": Float(0.40, 0.0, 1.0, 0.01),
#     },
# )
# def _v_vhs_tape(im: Image.Image, *, desat: float, blur: float, jitter_strength: int, jitter_band: int, jitter_p: float, scan_opacity: float, scan_period: int, grain_sigma: float, brightness: float, vignette: float):
#     base = fit_rgb(im)
#     base = ImageEnhance.Color(base).enhance(desat)
#     if blur > 0:
#         base = base.filter(ImageFilter.GaussianBlur(blur))
#     r,g,b = base.split(); r = ImageChops.offset(r, 1, 0); b = ImageChops.offset(b, -1, 0); base = Image.merge("RGB", (r,g,b))
#     base = jitter_rows(base, strength=jitter_strength, band=jitter_band, p=jitter_p)
# 
#     w,h = base.size
#     ov = Image.new("RGBA", (w,h), (0,0,0,0)); d = ImageDraw.Draw(ov)
#     band_h = max(2, h//36)
#     for _ in range(2):
#         y = h - random.randint(band_h, band_h*3)
#         for x in range(0, w, 6):
#             if random.random() < 0.6:
#                 d.line([(x, y), (x+4, y)], fill=(255,255,255, random.randint(30,90)), width=1)
#     for _ in range(random.randint(3,7)):
#         y = random.randint(0, h-1)
#         d.line([(0,y),(w,y)], fill=(255,255,255, random.randint(20,70)), width=1)
#     base = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
# 
#     base = add_scanlines(base, opacity=scan_opacity, period=scan_period)
#     base = grain(base, sigma=grain_sigma)
#     base = ImageEnhance.Brightness(base).enhance(brightness)
#     return vignette_fn(base, strength=vignette)

# --- VHS: less blur at 4k–8k, regain detail with perceptual USM
@variant(
    name="vhs_tape",
    options={
        "desat": Float(0.85, 0.0, 2.0, 0.01),
        "blur_min": Float(0.003, 0.0, 0.06, 0.0005),
        "jitter_strength_w": Float(0.02, 0.0, 0.25, 0.001),
        "jitter_band_h": Float(0.008, 0.001, 0.25, 0.001),
        "jitter_p": Float(0.12, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "grain_sigma": Float(8.0, 0.0, 30.0, 0.1),
        "brightness": Float(1.04, 0.0, 3.0, 0.01),
        "vignette": Float(0.40, 0.0, 1.0, 0.01),
    },
)
def _v_vhs_tape(im: Image.Image, *, desat: float, blur_min: float, jitter_strength_w: float, jitter_band_h: float, jitter_p: float, scan_opacity: float, scan_pitch_h: float, grain_sigma: float, brightness: float, vignette: float):
    base = fit_rgb(im)
    base = ImageEnhance.Color(base).enhance(desat)

    # sublinear blur for high-res
    w, h = base.size
    md = float(min(w, h))
    r_base = max(1.0, float(px_min(base, max(0.0, blur_min))))
    # sublinear growth with sqrt of size vs. 1k reference, then clamp
    r = int(round(min(9.0, r_base * math.sqrt(md / 1024.0))))
    if r > 0:
        base = base.filter(ImageFilter.GaussianBlur(r))

    # chroma mis-register
    r_,g_,b_ = base.split()
    r_ = ImageChops.offset(r_,  1, 0)
    b_ = ImageChops.offset(b_, -1, 0)
    base = Image.merge("RGB", (r_, g_, b_))

    # line jitter
    base = jitter_rows(base, strength_w=jitter_strength_w, band_h=jitter_band_h, p=jitter_p)

    # scanlines with density compensation
    base = add_scanlines_perceptual_balanced(base, opacity=scan_opacity, pitch_h=scan_pitch_h)

    # film grain then perceptual sharpen to recover fine detail at 6k
    base = grain(base, sigma=grain_sigma)
    base = sharpen_perceptual(base, amount=0.7, radius_min=0.0015)

    base = ImageEnhance.Brightness(base).enhance(brightness)
    return vignette_fn(base, strength=vignette)


@variant(
    name="crt_zoomed",
    options={
        "cell_min": Float(0.01, 0.002, 0.1, 0.001),         # pixelation cell as frac of min
        "triad_w": Float(1/512, 1/4096, 1/64, 1/4096),
        "mask_strength": Float(0.12, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.26, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
        "gap_every": Int(4, 0, 32, 1),
        "gap_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "barrel_k": Float(0.075, 0.0, 0.5, 0.001),
        "contrast": Float(1.06, 0.5, 3.0, 0.01),
        "vignette": Float(0.55, 0.0, 1.0, 0.01),
    },
)
def _v_crt_zoomed(im: Image.Image, *, cell_min: float, triad_w: float, mask_strength: float, scan_opacity: float, scan_pitch_h: float, gap_every: int, gap_opacity: float, barrel_k: float, contrast: float, vignette: float):
    pximg = pixelate(fit_rgb(im), cell_min=cell_min)
    pximg = add_slot_mask(pximg, triad_w=triad_w, strength=mask_strength, scan_opacity=scan_opacity, scan_pitch_h=scan_pitch_h, gap_every=gap_every, gap_opacity=gap_opacity)
    pximg = barrel_distort(pximg, k=barrel_k)
    pximg = ImageEnhance.Contrast(pximg).enhance(contrast)
    return vignette_fn(pximg, strength=vignette)


# ==== CYBER-NOIR ADDITIONS ====

@variant(
    name="neon_sign_halo",
    options={
        "edge_contrast": Float(2.2, 0.0, 6.0, 0.1),
        "halo_radius_min": Float(0.012, 0.0, 0.12, 0.0005), # frac of min dim
        "halo_amount": Float(0.35, 0.0, 1.0, 0.01),
        "tint": Color("#00b7ff"),
        "base_mix": Float(0.65, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.14, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
    },
)
def _v_neon_sign_halo(im: Image.Image, *, edge_contrast: float, halo_radius_min: float, halo_amount: float, tint: str, base_mix: float, scan_opacity: float, scan_pitch_h: float):
    base = ImageOps.grayscale(im).convert("RGB")
    edges = ImageEnhance.Contrast(im).enhance(edge_contrast).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.grayscale(edges)
    edges_c = ImageOps.colorize(edges, black="#000000", white=tint)
    halo = edges_c.filter(ImageFilter.GaussianBlur(radius=px_min(im, halo_radius_min)))
    mixed = Image.blend(base, halo, halo_amount)
    mixed = Image.blend(mixed, fit_rgb(im), base_mix)
    return add_scanlines_perceptual_balanced(mixed, scan_opacity, scan_pitch_h)

@variant(
    name="hologram_shift",
    options={
        "rgb_shift_w": Float(0.003, 0.0, 0.05, 0.0005),     # frac of width
        "quant_levels": Int(6, 2, 64, 1),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "gap_pitch_h": Float(1/256, 1/1024, 1/32, 1/4096),  # vertical gap pitch
        "gap_opacity": Float(0.22, 0.0, 1.0, 0.01),
    },
)
def _v_hologram_shift(im: Image.Image, *, rgb_shift_w: float, quant_levels: int, scan_opacity: float, gap_pitch_h: float, gap_opacity: float):
    img = fit_rgb(im)
    shift = px_w(img, rgb_shift_w)
    r,g,b = img.split()
    r = ImageChops.offset(r, -shift, 0)
    b = ImageChops.offset(b,  shift, 0)
    img = Image.merge("RGB", (r,g,b))
    q = ImageOps.posterize(img, int(max(2, min(8, math.log2(quant_levels))))) if quant_levels <= 256 else img
    q = add_scanlines_perceptual_balanced(q, scan_opacity, 2 / img.height)
    ov = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(ov)
    pitch = max(1, px_h(img, gap_pitch_h))
    for y in range(0, img.height, pitch):
        d.line([(0,y),(img.width,y)], fill=(0,0,0, int(255*gap_opacity)), width=1)
    return Image.alpha_composite(q.convert("RGBA"), ov).convert("RGB")

@variant(
    name="sodium_vapor_night",
    options={
        "dark": Color("#040409"),
        "light": Color("#ffb400"),
        "fog_strength": Float(0.22, 0.0, 1.0, 0.01),
        "fog_height": Float(0.55, 0.0, 1.0, 0.01),
        "bloom_amount": Float(0.28, 0.0, 1.0, 0.01),
        "bloom_radius_min": Float(0.01, 0.0, 0.12, 0.0005),
        "contrast": Float(1.06, 0.5, 3.0, 0.01),
    },
)
def _v_sodium_vapor_night(im: Image.Image, *, dark: str, light: str, fog_strength: float, fog_height: float, bloom_amount: float, bloom_radius_min: float, contrast: float):
    base = duotone(im, hex_to_rgb(dark), hex_to_rgb(light))
    base = ImageEnhance.Contrast(base).enhance(contrast)
    w,h = base.size
    y = np.linspace(0,1,h, dtype=np.float32)
    center = np.clip(fog_height, 0.0, 1.0)
    dist = np.abs(y - center)
    fog = np.clip(1.0 - (dist / max(1e-6, center if center>0.5 else 1-center)), 0, 1) ** 1.8
    fog = (fog * fog_strength).astype(np.float32)
    arr = np.asarray(base, dtype=np.float32)
    fog_col = np.array(hex_to_rgb(light), np.float32)
    arr = np.clip(arr*(1.0 - fog[:,None]) + fog[:,None]*fog_col, 0, 255)
    out = Image.fromarray(arr.astype(np.uint8))
    return bloom_perceptual(out, amount=bloom_amount, radius_min=bloom_radius_min)

@variant(
    name="glitch_grid_tiles",
    options={
        "grid": Int(6, 2, 32, 1),
        "max_offset_min": Float(0.02, 0.0, 0.5, 0.001),  # frac of min dim
        "swap_prob": Float(0.08, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.16, 0.0, 1.0, 0.01),
    },
)
def _v_glitch_grid_tiles(im: Image.Image, *, grid: int, max_offset_min: float, swap_prob: float, scan_opacity: float):
    img = fit_rgb(im).copy()
    w,h = img.size
    gw = max(2, grid); gh = max(2, grid)
    cw, ch = w//gw, h//gh
    rng = random.Random(9901)
    tiles = []
    for gy in range(gh):
        row = []
        for gx in range(gw):
            box = (gx*cw, gy*ch, (gx+1)*cw if gx<gw-1 else w, (gy+1)*ch if gy<gh-1 else h)
            row.append(img.crop(box))
        tiles.append(row)
    for gy in range(gh):
        for gx in range(gw):
            if rng.random() < swap_prob:
                gx2 = min(gw-1, max(0, gx + rng.randint(-1,1)))
                gy2 = min(gh-1, max(0, gy + rng.randint(-1,1)))
                tiles[gy][gx], tiles[gy2][gx2] = tiles[gy2][gx2], tiles[gy][gx]
    out = Image.new("RGB", (w,h))
    max_off = px_min(img, max_offset_min)
    for gy in range(gh):
        for gx in range(gw):
            dx = rng.randint(-max_off, max_off)
            dy = rng.randint(-max_off, max_off)
            x = gx*cw + dx; y = gy*ch + dy
            out.paste(tiles[gy][gx], (x,y))
    return add_scanlines_perceptual_balanced(out, scan_opacity, 2 / h)

@variant(
    name="billboard_ghosting",
    options={
        "trails": Int(3, 0, 12, 1),
        "trail_shift_w": Float(0.005, 0.0, 0.1, 0.0005),   # frac of width
        "trail_decay": Float(0.55, 0.0, 1.0, 0.01),
        "bloom_amount": Float(0.24, 0.0, 1.0, 0.01),
        "bloom_radius_min": Float(0.01, 0.0, 0.12, 0.0005),
    },
)
def _v_billboard_ghosting(im: Image.Image, *, trails: int, trail_shift_w: float, trail_decay: float, bloom_amount: float, bloom_radius_min: float):
    base = fit_rgb(im)
    acc = base.copy()
    shift = px_w(base, trail_shift_w)
    for i in range(1, max(1,trails)+1):
        shifted = ImageChops.offset(base, i*shift, 0)
        acc = ImageChops.screen(acc, ImageEnhance.Brightness(shifted).enhance(trail_decay**i))
    acc = bloom(acc, amount=bloom_amount, radius_min=bloom_radius_min)
    return acc

@variant(
    name="smog_depth_fade",
    options={
        "strength": Float(0.28, 0.0, 1.0, 0.01),
        "bias": Float(0.35, 0.0, 1.0, 0.01),
        "tint": Color("#0c1220"),
    },
)
def _v_smog_depth_fade(im: Image.Image, *, strength: float, bias: float, tint: str):
    img = fit_rgb(im)
    w,h = img.size
    y = np.linspace(0,1,h, dtype=np.float32)
    fade = np.clip((y - bias) / max(1e-6, (1.0 - bias)), 0.0, 1.0) ** 1.6
    fade *= strength
    arr = np.asarray(img, dtype=np.float32)
    tcol = np.array(hex_to_rgb(tint), np.float32)
    arr = np.clip(arr*(1.0 - fade[:,None]) + fade[:,None]*tcol, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

@variant(
    name="dirac_static_hologram",
    options={
        "rgb_shift_w": Float(0.004, 0.0, 0.06, 0.0005),
        "dither_pitch_min": Float(0.01, 0.002, 0.1, 0.001), # ordered dither pitch
        "moire_pitch_min": Float(0.008, 0.002, 0.1, 0.001),
        "moire_mix": Float(0.20, 0.0, 1.0, 0.01),
        "shutter_amp_w": Float(0.004, 0.0, 0.1, 0.0005),    # per-row horizontal wobble
        "shutter_freq": Float(1.6, 0.1, 6.0, 0.05),
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "tint": Color("#52ffe7"),
        "quant_levels": Int(6, 2, 16, 1),
    },
)
def _v_dirac_static_hologram(im: Image.Image, *,
                             rgb_shift_w: float, dither_pitch_min: float, moire_pitch_min: float, moire_mix: float,
                             shutter_amp_w: float, shutter_freq: float, scan_opacity: float,
                             tint: str, quant_levels: int):
    img = fit_rgb(im)
    w, h = img.size

    shift = px_w(img, rgb_shift_w)
    r, g, b = img.split()
    r = ImageChops.offset(r,  shift, 0)
    b = ImageChops.offset(b, -shift, 0)
    img = Image.merge("RGB", (r, g, b))

    arr = np.asarray(img).copy()
    wob = px_w(img, shutter_amp_w)
    y = np.arange(h, dtype=np.float32)
    phase = (np.sin(2*np.pi*(y/h)*shutter_freq) * wob).astype(np.int32)
    for yy in range(h):
        arr[yy] = np.roll(arr[yy], int(phase[yy]), axis=0)
    img = Image.fromarray(arr)

    g1 = ImageOps.grayscale(img)
    p_d = max(1, px_min(img, dither_pitch_min))
    if p_d > 1:
        small = g1.resize((max(1, w//p_d), max(1, h//p_d)), Image.Resampling.BILINEAR)
        g1 = small.resize((w, h), Image.Resampling.NEAREST)
    g1 = ImageOps.posterize(g1.convert("L"), max(1, int(math.log2(quant_levels))))
    holo = ImageOps.colorize(g1, black="#000000", white=tint)

    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    p_m = max(1, px_min(img, moire_pitch_min))
    for x in range(0, w, p_m):
        a = int(255 * moire_mix * 0.65)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, a), width=1)
    for yline in range(0, h, p_m):
        a = int(255 * moire_mix * 0.45)
        draw.line([(0, yline), (w, yline)], fill=(255, 255, 255, a), width=1)
    holo = Image.alpha_composite(holo.convert("RGBA"), grid).convert("RGB")

    holo = add_scanlines_perceptual_balanced(holo, scan_opacity, 2 / h)
    return grain(holo, sigma=4)

@variant(
    name="smog_slick_reflections",
    options={
        "horizon": Float(0.58, 0.2, 0.9, 0.01),
        "reflect_mix": Float(0.45, 0.0, 1.0, 0.01),
        "ripple_amp_w": Float(0.01, 0.0, 0.2, 0.0005),  # frac of width
        "ripple_freq": Float(3.2, 0.2, 12.0, 0.1),
        "sodium": Color("#ffb400"),
        "cyan": Color("#00e5ff"),
        "smog": Float(0.22, 0.0, 1.0, 0.01),
        "bloom_amount": Float(0.28, 0.0, 1.0, 0.01),
        "bloom_radius_min": Float(0.012, 0.0, 0.12, 0.0005),
        "grain": Float(7.0, 0.0, 30.0, 0.1),
        "scan_opacity": Float(0.14, 0.0, 1.0, 0.01),
        "scan_pitch_h": Float(1/512, 1/2048, 1/64, 1/4096),
    },
)
def _v_smog_slick_reflections(im: Image.Image, *,
                              horizon: float, reflect_mix: float, ripple_amp_w: float, ripple_freq: float,
                              sodium: str, cyan: str, smog: float,
                              bloom_amount: float, bloom_radius_min: float, grain: float, scan_opacity: float, scan_pitch_h: float):
    img = fit_rgb(im)
    w, h = img.size
    y0 = int(h * horizon); y0 = min(max(1, y0), h-1)

    top = img.crop((0, 0, w, y0))
    bot = img.crop((0, y0, w, h))

    neon = duotone(img, hex_to_rgb(sodium), hex_to_rgb(cyan))

    refl_src = top.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    arr = np.asarray(refl_src)
    hh, ww, _ = arr.shape
    yy = np.arange(hh, dtype=np.float32)
    amp = px_w(img, ripple_amp_w)
    phase = (np.sin(2*np.pi*(yy/hh)*ripple_freq) * amp).astype(np.int32)
    for i in range(hh):
        arr[i] = np.roll(arr[i], int(phase[i]), axis=0)
    ripple = Image.fromarray(arr)

    neon_bot = neon.crop((0, y0, w, h)).resize((w, h - y0), Image.Resampling.BILINEAR)
    refl = Image.blend(ripple, neon_bot, reflect_mix)

    mask = Image.new("L", refl.size, 0)
    grad = np.linspace(200, 40, h - y0).astype(np.uint8)
    mask = Image.fromarray(np.vstack([grad]*w).T)
    mask = mask.filter(ImageFilter.GaussianBlur(3))

    out = img.copy()
    region = Image.composite(refl, bot, mask)
    out.paste(region, (0, y0))

    out = Image.blend(out, neon, smog*0.25)
    out = bloom_perceptual(out, amount=bloom_amount, radius_min=bloom_radius_min)
    out = add_scanlines_perceptual_balanced(out, scan_opacity, scan_pitch_h)
    return ImageEnhance.Color(grain(out, sigma=grain)).enhance(0.95)


from scanlines import apply as scanlines_v5

@variant(
    name="scanlines_heavy_v5",
    options={
        "seed": Int(2025, 0, 1000000, 1),
        "triad_period_px": Float(3.6, 2.4, 8.0, 0.1),
        "triad_jitter": Float(0.01, 0.0, 0.05, 0.001),
        "triad_strength": Float(0.18, 0.0, 1.0, 0.01),
        "triad_luma_mode": Bool(True),
        "scan_gap": Float(0.22, 0.0, 0.6, 0.01),
        "scan_strength": Float(0.85, 0.0, 1.0, 0.01),
        "wobble_amp": Float(0.18, 0.0, 1.0, 0.01),
        "sigma_x": Float(0.30, 0.1, 1.2, 0.01),
        "sigma_y0": Float(0.8, 0.2, 2.0, 0.01),
        "sigma_y_bright_scale": Float(0.7, 0.0, 2.0, 0.01),
        "bloom_lambda_y": Float(2.8, 0.5, 8.0, 0.1),
        "bloom_sigma_x": Float(0.5, 0.1, 2.0, 0.05),
        "trails_D": Int(3, 0, 12, 1),
        "trails_tau": Float(2.2, 0.5, 6.0, 0.1),
        "defect_rate": Float(0.006, 0.0, 0.05, 0.001),
        "banding_strength": Float(0.008, 0.0, 0.05, 0.001),
        "pink_amount": Float(0.05, 0.0, 0.3, 0.01),
        "poisson_amount": Float(0.05, 0.0, 0.5, 0.01),
        "chroma_skew": Float(0.06, 0.0, 0.3, 0.01),
        "quant_bits": Int(8, 6, 10, 1),
        "dither_amount": Float(0.4, 0.0, 1.0, 0.01),
        "gamma_shadow": Float(0.92, 0.7, 1.0, 0.01),
        "gamma_high": Float(1.30, 1.0, 1.8, 0.01),
        "glass_amount": Float(0.04, 0.0, 0.2, 0.005),
        "vignette": Float(0.16, 0.0, 0.5, 0.01),
        "shaft_x": Float(0.50, 0.0, 1.0, 0.001),
        "shaft_sigma": Float(0.055, 0.01, 0.2, 0.001),
    },
)
def _v_scanlines_heavy_v5(im: Image.Image, **opts):
    base = fit_rgb(im)
    return scanlines_v5(base, **opts)
