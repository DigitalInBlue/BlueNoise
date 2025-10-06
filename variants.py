#!/usr/bin/env python3
import argparse, random
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
from PIL import (
    Image, ImageOps, ImageFilter, ImageDraw, ImageChops, ImageEnhance, ImageStat
)
import concurrent.futures as cf
import math

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

def hex_to_rgb(hex_color: str) -> Tuple[int,int,int]:
    s = hex_color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c*2 for c in s)
    return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))

def fit_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")

# === Building blocks ===
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
    r,g,b = im.split()
    r = ImageChops.offset(r, random.randint(-max_shift,max_shift), random.randint(-max_shift,max_shift))
    b = ImageChops.offset(b, random.randint(-max_shift,max_shift), random.randint(-max_shift,max_shift))
    return Image.merge("RGB", (r,g,b))

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
    # vertical RGB triads + scanlines + sparse vertical gaps
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

def dot_matrix_round_color(im: Image.Image, pitch=6, dot_scale=0.95, gamma=0.9, bg=(248, 245, 240)):
    im = im.convert("RGB")
    w, h = im.size
    out = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(out)

    for y in range(0, h, pitch):
        for x in range(0, w, pitch):
            box = (x, y, min(x+pitch, w), min(y+pitch, h))
            stat = ImageStat.Stat(im.crop(box))
            r, g, b = [c for c in stat.mean]
            # luminance (0=black, 1=white) -> darker = bigger dot
            L = (0.2126*r + 0.7152*g + 0.0722*b) / 255.0
            radius = (1.0 - pow(L, gamma)) * (pitch * 0.5 * dot_scale)
            if radius <= 0.3:
                continue
            cx = x + pitch * 0.5
            cy = y + pitch * 0.5
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(int(r), int(g), int(b)))
    return out


# === Variants ===
def var_grayscale_contrast(im: Image.Image):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=2)
    g = ImageEnhance.Contrast(g).enhance(1.6)
    g = ImageEnhance.Brightness(g).enhance(0.98)
    im2 = g.convert("RGB")
    im2 = im2.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))
    return im2

def var_gray_color_glitch(im: Image.Image):
    base = ImageOps.grayscale(im).convert("RGB")
    color = rgb_split(im, max_shift=10)
    edges = ImageEnhance.Sharpness(im).enhance(2).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    edges = ImageOps.grayscale(edges)
    edges = ImageEnhance.Brightness(edges).enhance(1.4)
    edges = edges.point(lambda v: 255 if v > 40 else 0)
    out = Image.composite(color, base, edges)
    out = add_scanlines(out, 0.22, 2)
    return jitter_rows(out, strength=12, band=10, p=0.18)

def var_mono_red_burnt(im: Image.Image, red=(122,14,14)):
    dt = duotone(im, (5,5,8), red)
    dt = ImageEnhance.Contrast(dt).enhance(1.35)
    dt = vignette(dt, strength=0.7)
    leak = Image.new("RGB", dt.size, (0,0,0))
    d = ImageDraw.Draw(leak)
    for _ in range(3):
        x0 = random.randint(-int(dt.width*0.2), int(dt.width*0.2))
        y0 = random.randint(-int(dt.height*0.2), int(dt.height*0.2))
        r  = random.randint(dt.width//3, dt.width)
        col = (min(255, red[0]+80), red[1]//3, red[2]//3)
        d.ellipse([x0-r, y0-r, x0+r, y0+r], fill=col)
    leak = leak.filter(ImageFilter.GaussianBlur(radius=80))
    out = Image.blend(dt, leak, 0.18)
    out = grain(out, sigma=6)
    return out

def var_amber_monitor(im: Image.Image):
    amber_dark = (5, 3, 0); amber_light = (255, 170, 40)
    dt = duotone(im, amber_dark, amber_light)
    dt = ImageEnhance.Contrast(dt).enhance(1.15)
    dt = add_scanlines(dt, 0.22, 2)
    dt = bloom(dt, amount=0.35, radius=3)
    return vignette(dt, strength=0.4)

def var_random_pix_patch_gray(im: Image.Image, seed=None):
    rnd = random.Random(seed)
    out = im.copy()
    w,h = out.size
    pw, ph = rnd.randint(w//4, w//3), rnd.randint(h//4, h//3)
    x0 = rnd.randint(0, max(1,w-pw)); y0 = rnd.randint(0, max(1,h-ph))
    patch = out.crop((x0,y0,x0+pw,y0+ph))
    patch = ImageOps.grayscale(patch).convert("RGB")
    small = patch.resize((max(1,pw//6), max(1,ph//6)), Image.Resampling.NEAREST)
    patch = small.resize((pw, ph), Image.Resampling.NEAREST)
    out.paste(patch, (x0,y0))
    return out

def var_halftone(im: Image.Image):
    g = ImageOps.grayscale(im)
    g = g.filter(ImageFilter.GaussianBlur(radius=0.8))
    arr = np.asarray(g, dtype=np.float32)/255.0
    bayer = (1/64.0)*np.array([
        [0,48,12,60,3,51,15,63],
        [32,16,44,28,35,19,47,31],
        [8,56,4,52,11,59,7,55],
        [40,24,36,20,43,27,39,23],
        [2,50,14,62,1,49,13,61],
        [34,18,46,30,33,17,45,29],
        [10,58,6,54,9,57,5,53],
        [42,26,38,22,41,25,37,21],
    ], dtype=np.float32)
    h,w = arr.shape
    tiled = np.tile(bayer, (h//8+1, w//8+1))[:h,:w]
    dots = (arr > tiled).astype(np.uint8)*255
    return Image.merge("RGB", (Image.fromarray(dots),)*3)

def var_neon_edges(im: Image.Image):
    e = im.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(2.0)
    e = ImageOps.colorize(ImageOps.grayscale(e), black="#050508", white="#00E5FF")
    return Image.blend(ImageOps.grayscale(im).convert("RGB"), e, 0.65)

def var_crt_mask(im: Image.Image):
    im = add_scanlines(im, 0.18, 2)
    w,h = im.size
    mask = Image.new("RGB", (w,h))
    mpx = 3
    draw = ImageDraw.Draw(mask)
    for x in range(0,w,mpx):
        for i,c in enumerate([(255,0,0),(0,255,0),(0,0,255)]):
            if x+i < w:
                draw.line([(x+i,0),(x+i,h)], fill=c, width=1)
    return Image.blend(im, mask, 0.08)

def var_pixel_sort_rows(im: Image.Image):
    arr = np.asarray(fit_rgb(im)).copy()
    h,w,_ = arr.shape
    for y in range(0,h, random.randint(4,14)):
        if random.random() < 0.3:
            row = arr[y]
            keys = row.mean(axis=1).argsort()
            arr[y] = row[keys]
    return Image.fromarray(arr).convert("RGB")

def neon_edges_tint(im: Image.Image, tint_rgb):
    e = im.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(2.2)
    e = ImageOps.grayscale(e)
    e = ImageOps.colorize(e, black="#050508", white=f"rgb{tint_rgb}")
    base = ImageOps.grayscale(im).convert("RGB")
    out = Image.blend(base, e, 0.70)
    return add_scanlines(out, 0.20, 2)

def var_neon_edges_amber(im: Image.Image):
    return neon_edges_tint(im, (255, 180, 60))

def var_neon_edges_red(im: Image.Image):
    return neon_edges_tint(im, (200, 30, 30))

def var_gray_contrast_redshadows(im: Image.Image, red=(122,14,14)):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=2)
    g = ImageEnhance.Contrast(g).enhance(1.7)
    g_rgb = g.convert("RGB")
    m = np.asarray(g, dtype=np.float32) / 255.0
    shadow = (1.0 - m) ** 1.6
    sr, sg, sb = red
    tint = np.zeros((g.height, g.width, 3), dtype=np.float32)
    tint[...,0] = sr; tint[...,1] = sg; tint[...,2] = sb
    tint *= shadow[...,None]
    base = np.asarray(g_rgb, dtype=np.float32)
    out = np.clip(base*0.92 + tint*0.20, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")

def var_posterize_redblack(im: Image.Image, red=(160,20,20)):
    g = ImageOps.grayscale(im)
    g = ImageOps.posterize(g.convert("L"), 3)
    rb = ImageOps.colorize(g, black="black", white=f"rgb{red}")
    return ImageEnhance.Contrast(rb).enhance(1.2)

def var_cyber_teal_redglow(im: Image.Image, red=(122,14,14)):
    teal = (0, 180, 170); dark = (5,5,8)
    base = duotone(im, dark, teal)
    base = add_scanlines(base, 0.15, 3)
    g = ImageOps.grayscale(im); g = ImageOps.autocontrast(g)
    glow = ImageOps.colorize(g, "black", f"rgb{red}").filter(ImageFilter.GaussianBlur(3))
    out = Image.blend(base, glow, 0.18)
    return vignette(out, 0.45)

def var_data_streaks(im: Image.Image):
    arr = np.asarray(im).copy()
    h,w,_ = arr.shape
    for _ in range(max(4, h//120)):
        y = random.randint(0, h-2)
        band = random.randint(2, 16)
        dx = random.randint(-w//8, w//8)
        arr[y:y+band,:,:] = np.roll(arr[y:y+band,:,:], dx, axis=1)
    return Image.fromarray(arr).convert("RGB")

# 1) noir_threshold (hard B/W) + ordered-dither option
def var_noir_threshold(im: Image.Image, threshold=128):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    bw = g.point(lambda v: 255 if v >= threshold else 0).convert("RGB")
    return bw

def var_noir_threshold_dither(im: Image.Image):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    # Bayer 8x8 ordered dither
    bayer = (1/64.0)*np.array([
        [0,48,12,60,3,51,15,63],
        [32,16,44,28,35,19,47,31],
        [8,56,4,52,11,59,7,55],
        [40,24,36,20,43,27,39,23],
        [2,50,14,62,1,49,13,61],
        [34,18,46,30,33,17,45,29],
        [10,58,6,54,9,57,5,53],
        [42,26,38,22,41,25,37,21],
    ], dtype=np.float32)
    arr = np.asarray(g, dtype=np.float32)/255.0
    h,w = arr.shape
    tiled = np.tile(bayer, (h//8+1, w//8+1))[:h,:w]
    dots = (arr > tiled).astype(np.uint8)*255
    return Image.merge("RGB", (Image.fromarray(dots),)*3)

# 2) cga_palette (fixed tiny retro palette)
def _palette_image(colors):
    pal = []
    for r,g,b in colors:
        pal.extend([r,g,b])
    pal += [0,0,0] * (256 - len(colors))
    pimg = Image.new("P", (1,1))
    pimg.putpalette(pal)
    return pimg

def var_cga_palette(im: Image.Image):
    # classic-ish 16-color CGA/EGA-ish vibe
    palette = [
        (0,0,0),(0,0,170),(0,170,0),(0,170,170),
        (170,0,0),(170,0,170),(170,85,0),(170,170,170),
        (85,85,85),(85,85,255),(85,255,85),(85,255,255),
        (255,85,85),(255,85,255),(255,255,85),(255,255,255)
    ]
    pimg = _palette_image(palette)
    q = fit_rgb(im).quantize(palette=pimg, dither=Image.FLOYDSTEINBERG)
    return q.convert("RGB")

# 3) blueprint_cyan
def var_blueprint_cyan(im: Image.Image):
    inv = ImageOps.invert(fit_rgb(im))
    out = duotone(inv, (0,16,21), (0,224,255))
    out = bloom(out, amount=0.35, radius=3)
    return vignette(out, strength=0.35)

# 4) wireframe_edges_red
def var_wireframe_edges_red(im: Image.Image):
    base = ImageOps.grayscale(im).convert("RGB")
    base = ImageEnhance.Brightness(base).enhance(0.6)
    edges = ImageEnhance.Sharpness(im).enhance(2).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.grayscale(edges)
    edges = ImageOps.autocontrast(edges)
    edges = ImageOps.colorize(edges, black="black", white="#E02020")
    out = ImageChops.screen(base, edges)
    return add_scanlines(out, 0.18, 3)

# 6) dot_matrix_round (circular halftone)
def var_dot_matrix_round(im: Image.Image, cell=8):
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g)
    w,h = g.size
    out = Image.new("RGB", (w,h), "black")
    draw = ImageDraw.Draw(out)
    arr = np.asarray(g, dtype=np.float32)/255.0
    for y in range(0,h,cell):
        for x in range(0,w,cell):
            patch = arr[y:min(y+cell,h), x:min(x+cell,w)]
            if patch.size == 0: continue
            lum = patch.mean()  # 0..1
            r = int((1.0 - lum) * (cell/2))  # darker -> bigger dot
            cx, cy = x + cell//2, y + cell//2
            if r > 0:
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(240,240,240))
    return out

# 10) cinema_21x9 (letterbox + grain)
def var_cinema_21x9(im: Image.Image):
    img = fit_rgb(im)
    w,h = img.size
    target_ratio = 21/9
    cur_ratio = w/h
    if cur_ratio > target_ratio:
        # crop width
        new_w = int(h*target_ratio)
        x0 = (w - new_w)//2
        img = img.crop((x0, 0, x0+new_w, h))
    else:
        # crop height
        new_h = int(w/target_ratio)
        y0 = (h - new_h)//2
        img = img.crop((0, y0, w, y0+new_h))
    # add thin bars (4% height total)
    bar_h = max(2, img.height//25)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,img.width,bar_h], fill=(0,0,0))
    draw.rectangle([0,img.height-bar_h,img.width,img.height], fill=(0,0,0))
    return grain(img, sigma=6)

# 13) scan_jitter_gif (animated WebP)
def var_scan_jitter_gif(im: Image.Image, n_frames=8, dur_ms=90):
    frames = []
    base = fit_rgb(im)
    for i in range(n_frames):
        f = rgb_split(base, max_shift=2+i%2)
        f = add_scanlines(f, 0.18 + 0.02*(i%2), 2)
        frames.append(f)
    # attach metadata for the saver
    return {"frames": frames, "duration": dur_ms, "loop": 0}

# 14) copper_duotone
def var_copper_duotone(im: Image.Image):
    out = duotone(im, (10,10,13), (184,115,51))
    return ImageEnhance.Contrast(out).enhance(1.12)

# 15) shadow_red_bleed
def var_shadow_red_bleed(im: Image.Image, red=(122,14,14), power=1.7, mix=0.22):
    g = ImageOps.grayscale(im)
    m = np.asarray(g, dtype=np.float32)/255.0
    shadow = (1.0 - m) ** power
    base = np.asarray(fit_rgb(im), dtype=np.float32)
    tint = np.zeros_like(base)
    tint[...,0] = red[0]; tint[...,1] = red[1]; tint[...,2] = red[2]
    tint *= shadow[...,None]
    out = np.clip(base*(1.0 - mix) + tint*mix, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")

def var_amber_monitor_strong(im: Image.Image):
    amber_dark = (4, 2, 0); amber_light = (255, 180, 60)
    dt = duotone(im, amber_dark, amber_light)
    dt = ImageEnhance.Contrast(dt).enhance(1.25)
    dt = ImageEnhance.Color(dt).enhance(0.9)
    dt = add_scanlines(dt, opacity=0.35, period=2)   # more pronounced
    dt = bloom(dt, amount=0.42, radius=3)
    return vignette(dt, strength=0.48)

def var_vhs_tape(im: Image.Image):
    base = fit_rgb(im)
    # slight blur + desat like tape
    base = ImageEnhance.Color(base).enhance(0.85)
    base = base.filter(ImageFilter.GaussianBlur(0.6))

    # chroma bleed / misregistration
    r,g,b = base.split()
    r = ImageChops.offset(r, 1, 0)
    b = ImageChops.offset(b, -1, 0)
    base = Image.merge("RGB", (r,g,b))

    # intermittent horizontal jitter bands
    base = jitter_rows(base, strength=18, band=6, p=0.12)

    # head-switching noise at bottom
    w,h = base.size
    ov = Image.new("RGBA", (w,h), (0,0,0,0))
    d = ImageDraw.Draw(ov)
    band_h = max(2, h//36)
    for _ in range(2):
        y = h - random.randint(band_h, band_h*3)
        for x in range(0, w, 6):
            if random.random() < 0.6:
                d.line([(x, y), (x+4, y)], fill=(255,255,255, random.randint(30,90)), width=1)
    # occasional dropout lines
    for _ in range(random.randint(3,7)):
        y = random.randint(0, h-1)
        d.line([(0,y),(w,y)], fill=(255,255,255, random.randint(20,70)), width=1)
    base = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")

    # interlace-like scanlines + tape grain
    base = add_scanlines(base, opacity=0.22, period=2)
    base = grain(base, sigma=8)
    # mild lift in blacks + slight gamma
    base = ImageEnhance.Brightness(base).enhance(1.04)
    return vignette(base, strength=0.40)

def var_crt_zoomed(im: Image.Image):
    # zoomed pixel look
    px = pixelate(fit_rgb(im), scale=2)
    # add slot mask & scanlines with small gaps
    px = add_slot_mask(px, triad=2, strength=0.12, scan_opacity=0.26, scan_period=2, gap_every=4, gap_opacity=0.18)
    # gentle barrel curvature
    px = barrel_distort(px, k=0.075)
    # slight center pop + edge falloff
    px = ImageEnhance.Contrast(px).enhance(1.06)
    return vignette(px, strength=0.55)

def var_dot_matrix_round_color(im: Image.Image):
    return dot_matrix_round_color(fit_rgb(im), pitch=6, dot_scale=0.95, gamma=0.9)

# --- New: parameterized GLITCH (replaces the old lambda) ---
def var_glitch(
    im: Image.Image,
    scan_alpha: float = 0.22,
    scan_thickness: int = 2,
    jitter_strength: int = 12,
    jitter_band: int = 10,
    jitter_p: float = 0.18,
) -> Image.Image:
    """
    Original behavior:
      jitter_rows(add_scanlines(rgb_split(im), 0.22, 2), 12, 10, 0.18)
    Now tunable via kwargs with the same defaults.
    """
    base = rgb_split(im)
    base = add_scanlines(base, scan_alpha, scan_thickness)
    out = jitter_rows(base, strength=jitter_strength, band=jitter_band, p=jitter_p)
    return out


# --- New: parameterized PIXEL (replaces the old lambda) ---
def var_pixel(
    im: Image.Image,
    factor: int = 16,
) -> Image.Image:
    """
    Original behavior:
      im.resize((w//16, h//16), NEAREST).resize((w, h), NEAREST)
    """
    w, h = im.size
    f = max(1, int(factor))
    small = im.resize((max(1, w // f), max(1, h // f)), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)


# --- New: parameterized VIGNETTE+GRAIN (replaces the old lambda) ---
def var_vignette_grain(
    im: Image.Image,
    sigma: float = 10.0,
    strength: float = 0.6,
) -> Image.Image:
    """
    Original behavior:
      grain(vignette(im), 10)
    Default strength matches vignette() default; tweakable if desired.
    """
    return grain(vignette(im, strength=strength), sigma)


# --- Updated: parameterized DOT-MATRIX COLOR WRAPPER (was fixed args) ---
def var_dot_matrix_round_color(
    im: Image.Image,
    pitch: int = 6,
    dot_scale: float = 0.95,
    gamma: float = 0.9,
    bg: Tuple[int, int, int] = (248, 245, 240),
) -> Image.Image:
    """
    Original wrapper called:
      dot_matrix_round_color(fit_rgb(im), pitch=6, dot_scale=0.95, gamma=0.9)
    Now exposes pitch/dot_scale/gamma/bg while keeping defaults identical.
    """
    return dot_matrix_round_color(fit_rgb(im), pitch=pitch, dot_scale=dot_scale, gamma=gamma, bg=bg)


# --- (Optional but handy): generic seedable pix-patch (in addition to A/B/C presets) ---
def var_pix_patch_gray(
    im: Image.Image,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Convenience wrapper around var_random_pix_patch_gray with a consistent name.
    """
    return var_random_pix_patch_gray(im, seed=seed)


# === Pipeline ===
VARIANTS = {
    "amber_monitor": var_amber_monitor,
    "crt_mask": var_crt_mask,
    "cyber_teal_redglow": var_cyber_teal_redglow,
    "data_streaks": var_data_streaks,
    "glitch": var_glitch,
    "gray_color_glitch": var_gray_color_glitch,
    "gray_contrast_redshadows": var_gray_contrast_redshadows,
    "gray_contrast": var_grayscale_contrast,
    "halftone": var_halftone,
    "mono_red_burnt": var_mono_red_burnt,
    "neon_edges_amber": var_neon_edges_amber,
    "neon_edges_red": var_neon_edges_red,
    "neon_edges": var_neon_edges,
    "pix_patch_gray_a": lambda im: var_random_pix_patch_gray(im, seed=1),
    "pix_patch_gray_b": lambda im: var_random_pix_patch_gray(im, seed=2),
    "pix_patch_gray_c": lambda im: var_random_pix_patch_gray(im, seed=3),
    "pix_patch_gray": var_pix_patch_gray,
    "pixel_sort": var_pixel_sort_rows,
    "pixel": var_pixel,
    "posterize_redblack": var_posterize_redblack,
    "vignette_grain": var_vignette_grain,
    "noir_threshold": var_noir_threshold,
    "noir_threshold_dither": var_noir_threshold_dither,
    "cga_palette": var_cga_palette,
    "blueprint_cyan": var_blueprint_cyan,
    "wireframe_edges_red": var_wireframe_edges_red,
    "dot_matrix_round": var_dot_matrix_round,
    "cinema_21x9": var_cinema_21x9,
    "scan_jitter_gif": var_scan_jitter_gif,   # animated WebP
    "copper_duotone": var_copper_duotone,
    "shadow_red_bleed": var_shadow_red_bleed,
    "amber_monitor_strong": var_amber_monitor_strong,
    "vhs_tape": var_vhs_tape,
    "crt_zoomed": var_crt_zoomed,
    "dot_matrix_round_color": var_dot_matrix_round_color,
}

def select_inputs(src: Path):
    """Return a list of files to process, preferring *_1920.* when both exist."""
    candidates = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    by_key: Dict[str, Path] = {}
    for p in sorted(candidates):
        stem = p.stem
        if stem.endswith("_1920"):
            key = stem[:-5]  # drop '_1920'
            # prefer 1920 version
            by_key[key] = p
        else:
            key = stem
            # only set if no 1920 chosen yet
            by_key.setdefault(key, p)
    return [by_key[k] for k in sorted(by_key.keys())]

def _save_variant(dst: Path, img_or_dict):
    if isinstance(img_or_dict, dict) and "frames" in img_or_dict:  # animated WebP
        frames = img_or_dict["frames"]
        duration = int(img_or_dict.get("duration", 100))
        loop = int(img_or_dict.get("loop", 0))
        frames[0].save(
            dst, format="WEBP", save_all=True, append_images=frames[1:],
            duration=duration, loop=loop, method=6
        )
    else:
        img_or_dict.save(dst, format="WEBP", quality=90, method=6)

def process_image(path: Path, out_dir: Path, red_rgb=(122,14,14), workers: int = 8):
    with Image.open(path) as im:
        base = fit_rgb(im)  # single load
        stem = path.stem

        def run(name, fn):
            # isolate per-thread image; no shared mutable state
            dst = out_dir / f"{stem}_{name}.webp"
            if dst.exists():
                return None

            im_local = base.copy()
            img = fn(im_local) if name != "mono_red_burnt" else var_mono_red_burnt(im_local, red_rgb)
            _save_variant(dst, img)
            return dst.name

        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(run, name, fn) for name, fn in VARIANTS.items()]
            for fut in cf.as_completed(futures):
                try:
                    print("Wrote", fut.result())
                except Exception as e:
                    print("Variant failed:", e)

def main():
    ap = argparse.ArgumentParser(description="Creative cyber-noir image variants (prefers *_1920 inputs; outputs WebP).")
    ap.add_argument("subdir")
    ap.add_argument("--out", default="_Variants")
    ap.add_argument("--red", default="#7a0e0e", help="brand red (hex) for mono_red_burnt and related variants")
    ap.add_argument("--workers", type=int, default=16, help="threads per file")
    args = ap.parse_args()

    src = Path(args.subdir)
    out = src / args.out
    out.mkdir(exist_ok=True)
    red = hex_to_rgb(args.red)

    files = select_inputs(src)
    if not files:
        print("No images found."); return

    for p in files:
        process_image(p, out, red, workers=args.workers)

    print("Done ->", out)

if __name__ == "__main__":
    main()
