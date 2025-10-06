from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import random, math, numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageChops, ImageEnhance, ImageStat

from variants_core import variant, Bool, Int, Float, Enum, Color

# ========= helper utils =========

def fit_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")

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

def add_scanlines(im: Image.Image, opacity=0.2, period=3):
    im = im.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0,0,0,0)); d = ImageDraw.Draw(ov)
    for y in range(0, im.height, period):
        d.line([(0,y), (im.width,y)], fill=(0,0,0,int(255*opacity)))
    return Image.alpha_composite(im, ov).convert("RGB")

def rgb_split(im: Image.Image, max_shift=6):
    rgbImage = fit_rgb(im)
    try:
        r,g,b = rgbImage.split()
        r = ImageChops.offset(r, random.randint(-max_shift,max_shift), random.randint(-max_shift,max_shift))
        b = ImageChops.offset(b, random.randint(-max_shift,max_shift), random.randint(-max_shift,max_shift))
        return Image.merge("RGB", (r,g,b))
    except Exception as e:
        print("Error in rgb_split:", e)
        return im

def jitter_rows(im: Image.Image, strength=8, band=8, p=0.22):
    arr = np.array(im)
    h,w,_ = arr.shape
    for y in range(0,h,band):
        if random.random() < p:
            dx = random.randint(-strength, strength)
            arr[y:y+band,:,:] = np.roll(arr[y:y+band,:,:], dx, axis=1)
    return Image.fromarray(arr).convert("RGB")

def bloom(im: Image.Image, amount=0.6, radius=6):
    blur = im.filter(ImageFilter.GaussianBlur(radius))
    return Image.blend(im, blur, amount)

def grain(im: Image.Image, sigma=10):
    arr = np.asarray(im, dtype=np.float32)
    n = np.random.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + n, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def vignette(im: Image.Image, strength=0.6):
    w,h = im.size
    y,x = np.ogrid[:h,:w]
    cx,cy = w/2,h/2
    r = np.sqrt((x-cx)**2+(y-cy)**2)
    mask = (r/r.max())**2
    mask = (1 - strength*mask)[...,None]
    arr = np.asarray(im, dtype=np.float32)
    out = np.clip(arr*mask, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")

def pixelate(im: Image.Image, scale=6):
    small = im.resize((max(1, im.width//scale), max(1, im.height//scale)), Image.Resampling.NEAREST)
    return small.resize(im.size, Image.Resampling.NEAREST)

def add_slot_mask(im: Image.Image, triad=3, strength=0.18, scan_opacity=0.25, scan_period=2, gap_every=3, gap_opacity=0.24):
    w, h = im.size
    mask = Image.new("RGB", (w, h), (0,0,0))
    d = ImageDraw.Draw(mask)
    for x in range(0, w, triad):
        cols = [(255,0,0),(0,255,0),(0,0,255)]
        for i, c in enumerate(cols):
            if x+i < w:
                d.line([(x+i, 0), (x+i, h)], fill=c, width=1)
    m = Image.blend(im, mask, strength)
    m = add_scanlines(m, opacity=scan_opacity, period=scan_period)
    if gap_every > 0:
        ov = Image.new("RGBA", (w,h), (0,0,0,0))
        d2 = ImageDraw.Draw(ov)
        for x in range(0, w, gap_every*triad):
            d2.line([(x,0),(x,h)], fill=(0,0,0,int(255*gap_opacity)))
        m = Image.alpha_composite(m.convert("RGBA"), ov).convert("RGB")
    return m

def barrel_distort(im: Image.Image, k=0.12):
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

def dmrc(im: Image.Image, pitch=6, dot_scale=0.95, gamma=0.9, bg=(248, 245, 240)):
    im = im.convert("RGB")
    w, h = im.size
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

# ========= variants (ported, with GUI parameters) =========

@variant(
    name="glitch",
    options={
        "scan_alpha": Float(0.22, 0.0, 1.0, 0.01),
        "scan_thickness": Int(2, 1, 8, 1),
        "jitter_strength": Int(12, 0, 64, 1),
        "jitter_band": Int(10, 1, 64, 1),
        "jitter_p": Float(0.18, 0.0, 1.0, 0.01),
    },
)
def _v_glitch(im: Image.Image, *, scan_alpha: float, scan_thickness: int, jitter_strength: int, jitter_band: int, jitter_p: float):
    base = rgb_split(im)
    base = add_scanlines(base, scan_alpha, scan_thickness)
    return jitter_rows(base, strength=jitter_strength, band=jitter_band, p=jitter_p)
# glitch – already parameterized (no change)

@variant(
    name="pixel",
    options={"factor": Int(16, 2, 128, 1)},
)
def _v_pixel(im: Image.Image, *, factor: int):
    w, h = im.size
    f = max(1, int(factor))
    small = im.resize((max(1, w // f), max(1, h // f)), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)


@variant(
    name="vignette_grain",
    options={
        "sigma": Float(10.0, 0.0, 50.0, 0.1),
        "strength": Float(0.6, 0.0, 1.0, 0.01),
    },
)
def _v_vignette_grain(im: Image.Image, *, sigma: float, strength: float):
    return grain(vignette(im, strength=strength), sigma)


@variant(
    name="dot_matrix_round_color",
    options={
        "pitch": Int(6, 2, 64, 1),
        "dot_scale": Float(0.95, 0.1, 2.0, 0.01),
        "gamma": Float(0.9, 0.1, 3.0, 0.01),
        "bg": Color("#f8f5f0"),
    },
)
def _v_dot_matrix_round_color(im: Image.Image, *, pitch: int, dot_scale: float, gamma: float, bg: str):
    return dmrc(fit_rgb(im), pitch=pitch, dot_scale=dot_scale, gamma=gamma, bg=hex_to_rgb(bg))


@variant(
    name="pix_patch_gray",
    options={
        "seed": Int(1, 0, 1_000_000, 1),
        "min_frac": Float(0.25, 0.05, 0.9, 0.01),
        "max_frac": Float(0.33, 0.06, 0.95, 0.01),
        "down_factor": Int(6, 2, 32, 1),
        "grayscale": Bool(True),
    },
)
def _v_pix_patch_gray(im: Image.Image, *, seed: int, min_frac: float, max_frac: float, down_factor: int, grayscale: bool):
    rnd = random.Random(seed)
    out = im.copy(); w,h = out.size
    pw = rnd.randint(max(1, int(w*min_frac)), max(1, int(w*max_frac)))
    ph = rnd.randint(max(1, int(h*min_frac)), max(1, int(h*max_frac)))
    x0 = rnd.randint(0, max(1, w - pw)); y0 = rnd.randint(0, max(1, h - ph))
    patch = out.crop((x0,y0,x0+pw,y0+ph))
    if grayscale:
        patch = ImageOps.grayscale(patch).convert("RGB")
    small = patch.resize((max(1,pw//down_factor), max(1,ph//down_factor)), Image.Resampling.NEAREST)
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


@variant(
    name="dot_matrix_round",
    options={"cell": Int(8, 2, 64, 1)},
)
def _v_dot_matrix_round(im: Image.Image, *, cell: int):
    g = ImageOps.grayscale(im); g = ImageOps.autocontrast(g)
    w,h = g.size
    out = Image.new("RGB", (w,h), "black"); draw = ImageDraw.Draw(out)
    arr = np.asarray(g, dtype=np.float32)/255.0
    for y in range(0,h,cell):
        for x in range(0,w,cell):
            patch = arr[y:min(y+cell,h), x:min(x+cell,w)]
            if patch.size == 0: continue
            lum = patch.mean()
            r = int((1.0 - lum) * (cell/2))
            if r > 0:
                cx, cy = x + cell//2, y + cell//2
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(240,240,240))
    return out


@variant(
    name="shadow_red_bleed",
    options={
        "red": Color("#7a0e0e"),
        "power": Float(1.7, 0.1, 4.0, 0.05),
        "mix": Float(0.22, 0.0, 1.0, 0.01),
    },
)
def _v_shadow_red_bleed(im: Image.Image, *, red: str, power: float, mix: float):
    red_t = hex_to_rgb(red)
    g = ImageOps.grayscale(im); m = np.asarray(g, dtype=np.float32)/255.0
    shadow = (1.0 - m) ** power
    base = np.asarray(fit_rgb(im), dtype=np.float32)
    tint = np.zeros_like(base); tint[...,0] = red_t[0]; tint[...,1] = red_t[1]; tint[...,2] = red_t[2]
    tint *= shadow[...,None]
    out = np.clip(base*(1.0 - mix) + tint*mix, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


@variant(
    name="mono_red_burnt",
    options={
        "red": Color("#7a0e0e"),
        "vignette": Float(0.7, 0.0, 1.0, 0.01),
        "leak_blur": Int(80, 0, 200, 1),
        "leak_blend": Float(0.18, 0.0, 1.0, 0.01),
        "grain_sigma": Float(6.0, 0.0, 30.0, 0.1),
        "contrast": Float(1.35, 0.5, 3.0, 0.01),
        "leak_count": Int(3, 0, 12, 1),
    },
)
def _v_mono_red_burnt(im: Image.Image, *, red: str, vignette: float, leak_blur: int, leak_blend: float, grain_sigma: float, contrast: float, leak_count: int):
    red_t = hex_to_rgb(red)
    dt = duotone(im, (5,5,8), red_t)
    dt = ImageEnhance.Contrast(dt).enhance(contrast)
    dt = vignette_fn(dt, strength=vignette) if (vignette is not None) else dt

    leak = Image.new("RGB", dt.size, (0,0,0)); d = ImageDraw.Draw(leak)
    for _ in range(max(0, leak_count)):
        x0 = random.randint(-int(dt.width*0.2), int(dt.width*0.2))
        y0 = random.randint(-int(dt.height*0.2), int(dt.height*0.2))
        r  = random.randint(max(1, dt.width//3), dt.width)
        col = (min(255, red_t[0]+80), red_t[1]//3, red_t[2]//3)
        d.ellipse([x0-r, y0-r, x0+r, y0+r], fill=col)
    if leak_blur > 0:
        leak = leak.filter(ImageFilter.GaussianBlur(radius=leak_blur))
    out = Image.blend(dt, leak, leak_blend)
    return grain(out, sigma=grain_sigma)

# small helper so name doesn't shadow function above
def vignette_fn(im: Image.Image, strength: float) -> Image.Image:
    return vignette(im, strength=strength)


@variant(
    name="scan_jitter_gif",
    options={
        "n_frames": Int(8, 1, 120, 1),
        "dur_ms": Int(90, 10, 500, 5),
        "shift_base": Int(2, 0, 16, 1),
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
    },
    animated=True, preview_safe=False,
)
def _v_scan_jitter_gif(im: Image.Image, *, n_frames: int, dur_ms: int, shift_base: int, scan_opacity: float, scan_period: int):
    frames: list[dict[str, Any]] = []
    base = fit_rgb(im)
    for i in range(int(n_frames)):
        f = rgb_split(base, max_shift=shift_base + (i % 2))
        f = add_scanlines(f, scan_opacity + 0.02*(i%2), scan_period)
        frames.append({"image": f, "duration_ms": int(dur_ms)})
    return frames


@variant(
    name="amber_monitor",
    options={
        "contrast": Float(1.15, 0.5, 3.0, 0.01),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
        "bloom_amount": Float(0.35, 0.0, 1.0, 0.01),
        "bloom_radius": Int(3, 0, 32, 1),
        "vignette": Float(0.4, 0.0, 1.0, 0.01),
    },
)
def _v_amber_monitor(im: Image.Image, *, contrast: float, scan_opacity: float, scan_period: int, bloom_amount: float, bloom_radius: int, vignette: float):
    amber_dark = (5, 3, 0); amber_light = (255, 170, 40)
    dt = duotone(im, amber_dark, amber_light)
    dt = ImageEnhance.Contrast(dt).enhance(contrast)
    dt = add_scanlines(dt, scan_opacity, scan_period)
    dt = bloom(dt, amount=bloom_amount, radius=bloom_radius)
    return vignette_fn(dt, strength=vignette)


@variant(
    name="amber_monitor_strong",
    options={
        "contrast": Float(1.25, 0.5, 3.0, 0.01),
        "color_enh": Float(0.9, 0.0, 3.0, 0.01),
        "scan_opacity": Float(0.35, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
        "bloom_amount": Float(0.42, 0.0, 1.0, 0.01),
        "bloom_radius": Int(3, 0, 32, 1),
        "vignette": Float(0.48, 0.0, 1.0, 0.01),
    },
)
def _v_amber_monitor_strong(im: Image.Image, *, contrast: float, color_enh: float, scan_opacity: float, scan_period: int, bloom_amount: float, bloom_radius: int, vignette: float):
    amber_dark = (4, 2, 0); amber_light = (255, 180, 60)
    dt = duotone(im, amber_dark, amber_light)
    dt = ImageEnhance.Contrast(dt).enhance(contrast)
    dt = ImageEnhance.Color(dt).enhance(color_enh)
    dt = add_scanlines(dt, opacity=scan_opacity, period=scan_period)
    dt = bloom(dt, amount=bloom_amount, radius=bloom_radius)
    return vignette_fn(dt, strength=vignette)


@variant(
    name="gray_color_glitch",
    options={
        "edge_sharp": Float(2.0, 0.0, 5.0, 0.1),
        "edge_thresh": Int(40, 0, 255, 1),
        "mix": Float(0.65, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "max_shift": Int(10, 0, 32, 1),
        "jitter_strength": Int(12, 0, 64, 1),
        "jitter_band": Int(10, 1, 64, 1),
        "jitter_p": Float(0.18, 0.0, 1.0, 0.01),
    },
)
def _v_gray_color_glitch(im: Image.Image, *, edge_sharp: float, edge_thresh: int, mix: float, scan_opacity: float, max_shift: int, jitter_strength: int, jitter_band: int, jitter_p: float):
    # Convert to RGB.
    rgbImage = fit_rgb(im)
    base = ImageOps.grayscale(rgbImage).convert("RGB")
    color = rgb_split(rgbImage, max_shift=max_shift)
    edges = ImageEnhance.Sharpness(rgbImage).enhance(edge_sharp).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    edges = ImageOps.grayscale(edges)
    edges = ImageEnhance.Brightness(edges).enhance(1.4)
    edges = edges.point(lambda v: 255 if v > edge_thresh else 0)
    out = Image.blend(base, color, mix)
    out = add_scanlines(out, scan_opacity, 2)
    return jitter_rows(out, strength=jitter_strength, band=jitter_band, p=jitter_p)


@variant(
    name="gray_contrast",
    options={
        "cutoff": Int(2, 0, 20, 1),
        "contrast": Float(1.6, 0.2, 4.0, 0.01),
        "brightness": Float(0.98, 0.2, 3.0, 0.01),
        "usm_radius": Float(2.0, 0.0, 8.0, 0.1),
        "usm_percent": Int(130, 0, 500, 1),
        "usm_threshold": Int(2, 0, 64, 1),
    },
)
def _v_gray_contrast(im: Image.Image, *, cutoff: int, contrast: float, brightness: float, usm_radius: float, usm_percent: int, usm_threshold: int):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=cutoff)
    g = ImageEnhance.Contrast(g).enhance(contrast)
    g = ImageEnhance.Brightness(g).enhance(brightness)
    im2 = g.convert("RGB")
    if usm_radius > 0 and usm_percent > 0:
        im2 = im2.filter(ImageFilter.UnsharpMask(radius=usm_radius, percent=usm_percent, threshold=usm_threshold))
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
    options={"tint": Color("#ffb43c"), "mix": Float(0.70, 0.0, 1.0, 0.01), "scan_opacity": Float(0.20, 0.0, 1.0, 0.01)},
)
def _v_neon_edges_amber(im: Image.Image, *, tint: str, mix: float, scan_opacity: float):
    return _neon_edges_tint_custom(im, hex_to_rgb(tint), mix, scan_opacity)

@variant(
    name="neon_edges_red",
    options={"tint": Color("#c81e1e"), "mix": Float(0.70, 0.0, 1.0, 0.01), "scan_opacity": Float(0.20, 0.0, 1.0, 0.01)},
)
def _v_neon_edges_red(im: Image.Image, *, tint: str, mix: float, scan_opacity: float):
    return _neon_edges_tint_custom(im, hex_to_rgb(tint), mix, scan_opacity)

def _neon_edges_tint_custom(im: Image.Image, tint_rgb: Tuple[int,int,int], mix: float, scan_opacity: float):
    e = im.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(2.2)
    e = ImageOps.grayscale(e)
    e = ImageOps.colorize(e, black="#050508", white=f"rgb{tint_rgb}")
    base = ImageOps.grayscale(im).convert("RGB")
    out = Image.blend(base, e, mix)
    return add_scanlines(out, scan_opacity, 2)


@variant(
    name="data_streaks",
    options={
        "count": Int(4, 0, 200, 1),
        "band_min": Int(2, 1, 64, 1),
        "band_max": Int(16, 2, 128, 1),
        "dx_max": Int(64, 0, 512, 1),
    },
)
def _v_data_streaks(im: Image.Image, *, count: int, band_min: int, band_max: int, dx_max: int):
    arr = np.asarray(im).copy(); h,w,_ = arr.shape
    for _ in range(max(0, count if count>0 else max(4, h//120))):
        y = random.randint(0, h-2)
        band = random.randint(band_min, band_max)
        dx = random.randint(-dx_max, dx_max)
        arr[y:y+band,:,:] = np.roll(arr[y:y+band,:,:], dx, axis=1)
    return Image.fromarray(arr).convert("RGB")


@variant(
    name="crt_mask",
    options={
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
        "triad": Int(3, 1, 8, 1),
        "mask_strength": Float(0.08, 0.0, 1.0, 0.01),
    },
)
def _v_crt_mask(im: Image.Image, *, scan_opacity: float, scan_period: int, triad: int, mask_strength: float):
    im2 = add_scanlines(im, scan_opacity, scan_period)
    w,h = im2.size; mask = Image.new("RGB", (w,h)); draw = ImageDraw.Draw(mask)
    for x in range(0,w,triad):
        for i,c in enumerate([(255,0,0),(0,255,0),(0,0,255)]):
            if x+i < w:
                draw.line([(x+i,0),(x+i,h)], fill=c, width=1)
    return Image.blend(im2, mask, mask_strength)


@variant(
    name="pixel_sort",
    options={
        "step_min": Int(4, 1, 64, 1),
        "step_max": Int(14, 2, 256, 1),
        "prob": Float(0.3, 0.0, 1.0, 0.01),
    },
)
def _v_pixel_sort_rows(im: Image.Image, *, step_min: int, step_max: int, prob: float):
    arr = np.asarray(fit_rgb(im)).copy()
    h,w,_ = arr.shape
    step_min = max(1, step_min); step_max = max(step_min+1, step_max)
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
    options={"blur": Float(0.8, 0.0, 5.0, 0.05)},
)
def _v_halftone(im: Image.Image, *, blur: float):
    g = ImageOps.grayscale(im)
    if blur > 0:
        g = g.filter(ImageFilter.GaussianBlur(radius=blur))
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
    options={
        "dither": Bool(True),
    },
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
        "bloom_radius": Int(3, 0, 32, 1),
        "vignette": Float(0.35, 0.0, 1.0, 0.01),
    },
)
def _v_blueprint_cyan(im: Image.Image, *, dark: str, light: str, bloom_amount: float, bloom_radius: int, vignette: float):
    inv = ImageOps.invert(fit_rgb(im))
    out = duotone(inv, hex_to_rgb(dark), hex_to_rgb(light))
    out = bloom(out, amount=bloom_amount, radius=bloom_radius)
    return vignette_fn(out, strength=vignette)


@variant(
    name="wireframe_edges_red",
    options={
        "edge_color": Color("#E02020"),
        "base_brightness": Float(0.6, 0.0, 3.0, 0.01),
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "scan_period": Int(3, 1, 8, 1),
    },
)
def _v_wireframe_edges_red(im: Image.Image, *, edge_color: str, base_brightness: float, scan_opacity: float, scan_period: int):
    base = ImageOps.grayscale(im).convert("RGB")
    base = ImageEnhance.Brightness(base).enhance(base_brightness)
    edges = ImageEnhance.Sharpness(im).enhance(2).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.grayscale(edges); edges = ImageOps.autocontrast(edges)
    edges = ImageOps.colorize(edges, black="black", white=edge_color)
    out = ImageChops.screen(base, edges)
    return add_scanlines(out, scan_opacity, scan_period)


@variant(
    name="cinema_21x9",
    options={
        "bar_frac": Float(0.04, 0.0, 0.2, 0.005),  # total bar fraction of height
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


@variant(
    name="vhs_tape",
    options={
        "desat": Float(0.85, 0.0, 2.0, 0.01),
        "blur": Float(0.6, 0.0, 10.0, 0.05),
        "jitter_strength": Int(18, 0, 128, 1),
        "jitter_band": Int(6, 1, 64, 1),
        "jitter_p": Float(0.12, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
        "grain_sigma": Float(8.0, 0.0, 30.0, 0.1),
        "brightness": Float(1.04, 0.0, 3.0, 0.01),
        "vignette": Float(0.40, 0.0, 1.0, 0.01),
    },
)
def _v_vhs_tape(im: Image.Image, *, desat: float, blur: float, jitter_strength: int, jitter_band: int, jitter_p: float, scan_opacity: float, scan_period: int, grain_sigma: float, brightness: float, vignette: float):
    base = fit_rgb(im)
    base = ImageEnhance.Color(base).enhance(desat)
    if blur > 0:
        base = base.filter(ImageFilter.GaussianBlur(blur))
    r,g,b = base.split(); r = ImageChops.offset(r, 1, 0); b = ImageChops.offset(b, -1, 0); base = Image.merge("RGB", (r,g,b))
    base = jitter_rows(base, strength=jitter_strength, band=jitter_band, p=jitter_p)

    w,h = base.size
    ov = Image.new("RGBA", (w,h), (0,0,0,0)); d = ImageDraw.Draw(ov)
    band_h = max(2, h//36)
    for _ in range(2):
        y = h - random.randint(band_h, band_h*3)
        for x in range(0, w, 6):
            if random.random() < 0.6:
                d.line([(x, y), (x+4, y)], fill=(255,255,255, random.randint(30,90)), width=1)
    for _ in range(random.randint(3,7)):
        y = random.randint(0, h-1)
        d.line([(0,y),(w,y)], fill=(255,255,255, random.randint(20,70)), width=1)
    base = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")

    base = add_scanlines(base, opacity=scan_opacity, period=scan_period)
    base = grain(base, sigma=grain_sigma)
    base = ImageEnhance.Brightness(base).enhance(brightness)
    return vignette_fn(base, strength=vignette)


@variant(
    name="crt_zoomed",
    options={
        "scale": Int(2, 1, 16, 1),
        "triad": Int(2, 1, 8, 1),
        "mask_strength": Float(0.12, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.26, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
        "gap_every": Int(4, 0, 32, 1),
        "gap_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "barrel_k": Float(0.075, 0.0, 0.5, 0.001),
        "contrast": Float(1.06, 0.5, 3.0, 0.01),
        "vignette": Float(0.55, 0.0, 1.0, 0.01),
    },
)
def _v_crt_zoomed(im: Image.Image, *, scale: int, triad: int, mask_strength: float, scan_opacity: float, scan_period: int, gap_every: int, gap_opacity: float, barrel_k: float, contrast: float, vignette: float):
    px = pixelate(fit_rgb(im), scale=max(1, scale))
    px = add_slot_mask(px, triad=triad, strength=mask_strength, scan_opacity=scan_opacity, scan_period=scan_period, gap_every=gap_every, gap_opacity=gap_opacity)
    px = barrel_distort(px, k=barrel_k)
    px = ImageEnhance.Contrast(px).enhance(contrast)
    return vignette_fn(px, strength=vignette)

# ==== CYBER-NOIR ADDITIONS ====

@variant(
    name="neon_sign_halo",
    options={
        "edge_contrast": Float(2.2, 0.0, 6.0, 0.1),
        "halo_radius": Int(12, 0, 64, 1),
        "halo_amount": Float(0.35, 0.0, 1.0, 0.01),
        "tint": Color("#00b7ff"),   # neon cyan by default
        "base_mix": Float(0.65, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.14, 0.0, 1.0, 0.01),
        "scan_period": Int(2, 1, 8, 1),
    },
)
def _v_neon_sign_halo(im: Image.Image, *, edge_contrast: float, halo_radius: int, halo_amount: float, tint: str, base_mix: float, scan_opacity: float, scan_period: int):
    base = ImageOps.grayscale(im).convert("RGB")
    edges = ImageEnhance.Contrast(im).enhance(edge_contrast).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.grayscale(edges)
    edges_c = ImageOps.colorize(edges, black="#000000", white=tint)
    halo = edges_c.filter(ImageFilter.GaussianBlur(radius=halo_radius))
    mixed = Image.blend(base, halo, halo_amount)
    mixed = Image.blend(mixed, fit_rgb(im), base_mix)
    return add_scanlines(mixed, scan_opacity, scan_period)

@variant(
    name="hologram_shift",
    options={
        "rgb_shift": Int(3, 0, 16, 1),
        "quant_levels": Int(6, 2, 64, 1),
        "scan_opacity": Float(0.22, 0.0, 1.0, 0.01),
        "gap_period": Int(6, 2, 32, 1),   # missing-scan gaps
        "gap_opacity": Float(0.22, 0.0, 1.0, 0.01),
    },
)
def _v_hologram_shift(im: Image.Image, *, rgb_shift: int, quant_levels: int, scan_opacity: float, gap_period: int, gap_opacity: float):
    img = fit_rgb(im)
    r,g,b = img.split()
    r = ImageChops.offset(r, -rgb_shift, 0)
    b = ImageChops.offset(b, rgb_shift, 0)
    img = Image.merge("RGB", (r,g,b))
    # posterize/quantize
    q = ImageOps.posterize(img, int(max(2, min(8, math.log2(quant_levels))))) if quant_levels <= 256 else img
    q = add_scanlines(q, scan_opacity, 2)
    if gap_period > 0:
        ov = Image.new("RGBA", img.size, (0,0,0,0))
        d = ImageDraw.Draw(ov)
        for y in range(0, img.height, gap_period):
            d.line([(0,y),(img.width,y)], fill=(0,0,0, int(255*gap_opacity)), width=1)
        q = Image.alpha_composite(q.convert("RGBA"), ov).convert("RGB")
    return q


@variant(
    name="sodium_vapor_night",
    options={
        "dark": Color("#040409"),
        "light": Color("#ffb400"),       # street-lamp amber
        "fog_strength": Float(0.22, 0.0, 1.0, 0.01),
        "fog_height": Float(0.55, 0.0, 1.0, 0.01),  # 0=top fog, 1=bottom fog
        "bloom_amount": Float(0.28, 0.0, 1.0, 0.01),
        "bloom_radius": Int(6, 0, 64, 1),
        "contrast": Float(1.06, 0.5, 3.0, 0.01),
    },
)
def _v_sodium_vapor_night(im: Image.Image, *, dark: str, light: str, fog_strength: float, fog_height: float, bloom_amount: float, bloom_radius: int, contrast: float):
    base = duotone(im, hex_to_rgb(dark), hex_to_rgb(light))
    base = ImageEnhance.Contrast(base).enhance(contrast)
    # vertical fog ramp
    w,h = base.size
    y = np.linspace(0,1,h, dtype=np.float32)
    # fog from fog_height toward edges
    center = np.clip(fog_height, 0.0, 1.0)
    dist = np.abs(y - center)
    fog = np.clip(1.0 - (dist / max(1e-6, center if center>0.5 else 1-center)), 0, 1) ** 1.8
    fog = (fog * fog_strength).astype(np.float32)
    arr = np.asarray(base, dtype=np.float32)
    fog_col = np.array(hex_to_rgb(light), np.float32)
    arr = np.clip(arr*(1.0 - fog[:,None]) + fog[:,None]*fog_col, 0, 255)
    out = Image.fromarray(arr.astype(np.uint8))
    return bloom(out, amount=bloom_amount, radius=bloom_radius)


@variant(
    name="glitch_grid_tiles",
    options={
        "grid": Int(6, 2, 32, 1),
        "max_offset": Int(14, 0, 128, 1),
        "swap_prob": Float(0.08, 0.0, 1.0, 0.01),
        "scan_opacity": Float(0.16, 0.0, 1.0, 0.01),
    },
)
def _v_glitch_grid_tiles(im: Image.Image, *, grid: int, max_offset: int, swap_prob: float, scan_opacity: float):
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
    # random swaps / offsets
    for gy in range(gh):
        for gx in range(gw):
            if rng.random() < swap_prob:
                gx2 = min(gw-1, max(0, gx + rng.randint(-1,1)))
                gy2 = min(gh-1, max(0, gy + rng.randint(-1,1)))
                tiles[gy][gx], tiles[gy2][gx2] = tiles[gy2][gx2], tiles[gy][gx]
    # paste back with offsets
    out = Image.new("RGB", (w,h))
    for gy in range(gh):
        for gx in range(gw):
            dx = rng.randint(-max_offset, max_offset)
            dy = rng.randint(-max_offset, max_offset)
            x = gx*cw + dx; y = gy*ch + dy
            out.paste(tiles[gy][gx], (x,y))
    return add_scanlines(out, scan_opacity, 2)


@variant(
    name="billboard_ghosting",
    options={
        "trails": Int(3, 0, 12, 1),
        "trail_shift": Int(3, 0, 32, 1),
        "trail_decay": Float(0.55, 0.0, 1.0, 0.01),
        "bloom_amount": Float(0.24, 0.0, 1.0, 0.01),
        "bloom_radius": Int(5, 0, 64, 1),
    },
)
def _v_billboard_ghosting(im: Image.Image, *, trails: int, trail_shift: int, trail_decay: float, bloom_amount: float, bloom_radius: int):
    base = fit_rgb(im)
    acc = base.copy()
    for i in range(1, max(1,trails)+1):
        shifted = ImageChops.offset(base, i*trail_shift, 0)
        acc = ImageChops.screen(acc, ImageEnhance.Brightness(shifted).enhance(trail_decay**i))
    acc = bloom(acc, amount=bloom_amount, radius=bloom_radius)
    return acc


@variant(
    name="smog_depth_fade",
    options={
        "strength": Float(0.28, 0.0, 1.0, 0.01),
        "bias": Float(0.35, 0.0, 1.0, 0.01),   # where fade starts vertically
        "tint": Color("#0c1220"),              # bluish smog
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


# ---- Animated: diagonal rain with parallax feel ----
@variant(
    name="rain_parallax_anim",
    options={
        "n_frames": Int(24, 1, 240, 1),
        "dur_ms": Int(42, 10, 200, 1),
        "density": Int(260, 0, 1500, 10),
        "len_px": Int(22, 2, 120, 1),
        "angle": Float(15.0, -45.0, 45.0, 0.5),
        "opacity": Float(0.22, 0.0, 1.0, 0.01),
        "layer_parallax": Float(0.5, 0.0, 2.0, 0.01),
    },
    animated=True,
    preview_safe=False,
)
def _v_rain_parallax_anim(im: Image.Image, *, n_frames: int, dur_ms: int, density: int, len_px: int, angle: float, opacity: float, layer_parallax: float):
    base = fit_rgb(im)
    w,h = base.size
    ang = math.radians(angle)
    dx, dy = math.cos(ang), math.sin(ang)
    rng = random.Random(4242)
    # pre-generate drops: (x,y, layer_speed)
    drops = []
    for _ in range(max(0, density)):
        x = rng.randint(-w//4, int(w*1.25))
        y = rng.randint(-h//4, int(h*1.25))
        layer = 1.0 + layer_parallax*rng.random()
        drops.append((x,y,layer))
    frames: list[dict[str, Any]] = []
    for t in range(int(n_frames)):
        img = base.copy().convert("RGBA")
        ov = Image.new("RGBA", (w,h), (0,0,0,0))
        d = ImageDraw.Draw(ov)
        for (x0,y0,layer) in drops:
            # move with time
            offx = int(dx * len_px * t * 0.2 * layer)
            offy = int(dy * len_px * t * 0.2 * layer)
            x1 = x0 + offx; y1 = y0 + offy
            x2 = int(x1 + dx*len_px); y2 = int(y1 + dy*len_px)
            a = int(255*opacity)
            d.line([(x1,y1),(x2,y2)], fill=(255,255,255,a), width=1)
        img = Image.alpha_composite(img, ov)
        frames.append({"image": img.convert("RGB"), "duration_ms": int(dur_ms)})
    return frames

@variant(
    name="dirac_static_hologram",
    options={
        "rgb_shift": Int(4, 0, 24, 1),
        "dither_scale": Int(8, 2, 32, 1),   # pixel pitch for ordered dither
        "moire_pitch": Int(6, 2, 32, 1),
        "moire_mix": Float(0.20, 0.0, 1.0, 0.01),
        "shutter_amp": Int(6, 0, 48, 1),    # rolling shutter vertical wobble
        "shutter_freq": Float(1.6, 0.1, 6.0, 0.05),
        "scan_opacity": Float(0.18, 0.0, 1.0, 0.01),
        "tint": Color("#52ffe7"),           # aqua hologram bias
        "quant_levels": Int(6, 2, 16, 1),
    },
)
def _v_dirac_static_hologram(im: Image.Image, *,
                             rgb_shift: int, dither_scale: int, moire_pitch: int, moire_mix: float,
                             shutter_amp: int, shutter_freq: float, scan_opacity: float,
                             tint: str, quant_levels: int):
    img = fit_rgb(im)
    w, h = img.size

    # 1) RGB mis-register
    r, g, b = img.split()
    r = ImageChops.offset(r,  rgb_shift, 0)
    b = ImageChops.offset(b, -rgb_shift, 0)
    img = Image.merge("RGB", (r, g, b))

    # 2) Rolling shutter warp (per-row horizontal phase)
    arr = np.asarray(img).copy()
    y = np.arange(h, dtype=np.float32)
    phase = (np.sin(2*np.pi*(y/h)*shutter_freq) * shutter_amp).astype(np.int32)
    for yy in range(h):
        arr[yy] = np.roll(arr[yy], int(phase[yy]), axis=0)
    img = Image.fromarray(arr)

    # 3) Ordered dither plane (simulate bit-depth stepping)
    g1 = ImageOps.grayscale(img)
    if dither_scale > 1:
        small = g1.resize((max(1, w//dither_scale), max(1, h//dither_scale)), Image.Resampling.BILINEAR)
        g1 = small.resize((w, h), Image.Resampling.NEAREST)
    g1 = ImageOps.posterize(g1.convert("L"), max(1, int(math.log2(quant_levels))))  # 2..16 levels
    holo = ImageOps.colorize(g1, black="#000000", white=tint)

    # 4) Moiré grid overlay
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    p = max(1, moire_pitch)
    for x in range(0, w, p):
        a = int(255 * moire_mix * 0.65)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, a), width=1)
    for yline in range(0, h, p):
        a = int(255 * moire_mix * 0.45)
        draw.line([(0, yline), (w, yline)], fill=(255, 255, 255, a), width=1)
    holo = Image.alpha_composite(holo.convert("RGBA"), grid).convert("RGB")

    # 5) Scanlines + gentle grain
    holo = add_scanlines(holo, scan_opacity, 2)
    return grain(holo, sigma=4)

@variant(
    name="smog_slick_reflections",
    options={
        "horizon": Float(0.58, 0.2, 0.9, 0.01),   # where reflection begins (fraction of height)
        "reflect_mix": Float(0.45, 0.0, 1.0, 0.01),
        "ripple_amp": Int(6, 0, 32, 1),
        "ripple_freq": Float(3.2, 0.2, 12.0, 0.1),
        "sodium": Color("#ffb400"),
        "cyan": Color("#00e5ff"),
        "smog": Float(0.22, 0.0, 1.0, 0.01),
        "bloom_amount": Float(0.28, 0.0, 1.0, 0.01),
        "bloom_radius": Int(7, 0, 64, 1),
        "grain": Float(7.0, 0.0, 30.0, 0.1),
        "scan_opacity": Float(0.14, 0.0, 1.0, 0.01),
    },
)
def _v_smog_slick_reflections(im: Image.Image, *,
                              horizon: float, reflect_mix: float, ripple_amp: int, ripple_freq: float,
                              sodium: str, cyan: str, smog: float,
                              bloom_amount: float, bloom_radius: int, grain: float, scan_opacity: float):
    img = fit_rgb(im)
    w, h = img.size
    y0 = int(h * horizon)
    y0 = min(max(1, y0), h-1)

    top = img.crop((0, 0, w, y0))
    bot = img.crop((0, y0, w, h))

    # 1) Make neon-biased duotone for reflections (cyan vs sodium)
    neon = duotone(img, hex_to_rgb(sodium), hex_to_rgb(cyan))

    # 2) Build reflection from flipped top with ripples
    refl_src = top.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    arr = np.asarray(refl_src)
    hh, ww, _ = arr.shape
    yy = np.arange(hh, dtype=np.float32)
    phase = (np.sin(2*np.pi*(yy/hh)*ripple_freq) * ripple_amp).astype(np.int32)
    for i in range(hh):
        arr[i] = np.roll(arr[i], int(phase[i]), axis=0)
    ripple = Image.fromarray(arr)

    # 3) Mix neon duotone hint into reflection
    neon_bot = neon.crop((0, y0, w, h)).resize((w, h - y0), Image.Resampling.BILINEAR)
    refl = Image.blend(ripple, neon_bot, 0.35)

    # 4) Puddle mask via soft noise falloff
    mask = Image.new("L", refl.size, 0)
    m = ImageDraw.Draw(mask)
    m.rectangle([0, 0, w, (h - y0)], fill=140)  # base wetness
    # feather edges with a vertical gradient
    grad = np.linspace(200, 40, h - y0).astype(np.uint8)
    mask = Image.fromarray(np.vstack([grad]*w).T)
    mask = mask.filter(ImageFilter.GaussianBlur(3))

    # 5) Composite reflection onto lower half
    out = img.copy()
    region = Image.composite(refl, bot, mask)
    out.paste(region, (0, y0))

    # 6) Smog lift + bloom + scanlines + grain
    out = Image.blend(out, neon, smog*0.25)
    out = bloom(out, amount=bloom_amount, radius=bloom_radius)
    out = add_scanlines(out, scan_opacity, 2)
    return ImageEnhance.Color(grain(out, sigma=grain)).enhance(0.95)
