# scanlines_v5.py
import numpy as np
from PIL import Image
from numpy.fft import rfft2, irfft2

# ---------- utilities ----------
def to_np(img):
    if img.mode != "RGB": img = img.convert("RGB")
    return (np.asarray(img).astype(np.float32) / 255.0)

def from_np(a):
    a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGB")

def g2l(a, g=2.2): return np.power(a, g)
def l2g(a, g=2.2): return np.power(np.clip(a, 0, 1), 1.0 / g)
def luma(lin): return 0.2126*lin[...,0] + 0.7152*lin[...,1] + 0.0722*lin[...,2]

def gaussian1d_kernel(sigma):
    r = max(1, int(3*sigma))
    x = np.arange(-r, r+1, dtype=np.float32)
    k = np.exp(-(x*x)/(2*sigma*sigma)); k /= k.sum()
    return k, r

def convolve_y(arr, sigma):
    if sigma <= 0: return arr
    k, r = gaussian1d_kernel(sigma)
    if arr.ndim == 1:
        p = np.pad(arr, (r,r), 'reflect')
        out = np.empty_like(arr)
        for i in range(arr.shape[0]): out[i] = np.dot(k, p[i:i+len(k)])
        return out
    if arr.ndim == 2:
        p = np.pad(arr, ((r,r),(0,0)), 'reflect'); out = np.empty_like(arr)
        for i in range(arr.shape[0]): out[i,:] = np.tensordot(k, p[i:i+len(k),:], axes=(0,0))
        return out
    p = np.pad(arr, ((r,r),(0,0),(0,0)), 'reflect'); out = np.empty_like(arr)
    for i in range(arr.shape[0]): out[i,:,:] = np.tensordot(k, p[i:i+len(k),:,:], axes=(0,0))
    return out

def convolve_x(arr, sigma):
    if sigma <= 0:
        return arr
    if arr.ndim == 2:
        return convolve_y(arr.T, sigma).T
    # arr.ndim == 3  -> (H,W,C)
    return np.transpose(convolve_y(np.transpose(arr, (1, 0, 2)), sigma), (1, 0, 2))


rng_cache = {}
def RNG(seed):  # reproducible
    global rng_cache
    if seed not in rng_cache: rng_cache[seed] = np.random.default_rng(seed)
    return rng_cache[seed]

# ---------- stochastic fields ----------
def perlin(h, w, scale, rng):
    gy = max(1, int(h/scale))
    gx = max(1, int(w/scale))
    # gradient grid (gy+1, gx+1, 2)
    grads = rng.random((gy+1, gx+1, 2), dtype=np.float32)*2 - 1

    # pixel coords in grid space
    y = np.linspace(0, gy, h, dtype=np.float32)
    x = np.linspace(0, gx, w, dtype=np.float32)
    yi = np.floor(y).astype(np.int32); xi = np.floor(x).astype(np.int32)
    yf = (y - yi)[:, None]              # (H,1)
    xf = (x - xi)[None, :]              # (1,W)

    yi1 = np.clip(yi + 1, 0, gy); xi1 = np.clip(xi + 1, 0, gx)

    # corner gradients via advanced indexing -> (H,W,2)
    g00 = grads[yi[:, None], xi[None, :]]
    g10 = grads[yi1[:, None], xi[None, :]]
    g01 = grads[yi[:, None], xi1[None, :]]
    g11 = grads[yi1[:, None], xi1[None, :]]

    # displacement vectors to each corner, broadcast to (H,W)
    d00x, d00y = xf,      yf
    d10x, d10y = xf,      yf - 1.0
    d01x, d01y = xf - 1.0, yf
    d11x, d11y = xf - 1.0, yf - 1.0

    # dot products (H,W)
    n00 = g00[...,0]*d00x + g00[...,1]*d00y
    n10 = g10[...,0]*d10x + g10[...,1]*d10y
    n01 = g01[...,0]*d01x + g01[...,1]*d01y
    n11 = g11[...,0]*d11x + g11[...,1]*d11y

    # smooth interpolation
    def fade(t): return t*t*(3.0 - 2.0*t)
    ux = fade(xf)        # (1,W)
    uy = fade(yf)        # (H,1)

    nx0 = n00*(1-ux) + n01*ux
    nx1 = n10*(1-ux) + n11*ux
    n = nx0*(1-uy) + nx1*uy

    # normalize to [0,1]
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    return n.astype(np.float32)


def worley(h, w, cells, rng):
    gy, gx = max(1,int(cells)), max(1,int(cells*w/h))
    pts = rng.random((gy,gx,2), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = yy/h; xx = xx/w
    yy = yy*gy; xx = xx*gx
    yi = np.clip(np.floor(yy).astype(int),0,gy-1); xi = np.clip(np.floor(xx).astype(int),0,gx-1)
    out = np.full((h,w), 10.0, dtype=np.float32)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            yc = np.clip(yi+dy,0,gy-1); xc = np.clip(xi+dx,0,gx-1)
            py = (yc + pts[yc, xc, 1])
            px = (xc + pts[yc, xc, 0])
            d = np.hypot(yy - py, xx - px)
            out = np.minimum(out, d)
    out = (out - out.min())/(out.max()-out.min()+1e-6)
    return out

from numpy.fft import rfft2, irfft2

def blue_noise_mask(h, w, rng):
    base = rng.random((h, w), dtype=np.float32)
    F = rfft2(base)                                   # (H, W//2+1)
    ky = np.fft.fftfreq(h)[:, None]                   # (H,1)
    kx = np.fft.rfftfreq(w)[None, :]                  # (1,W//2+1)
    kk = np.sqrt(kx**2 + ky**2) + 1e-6
    BN = np.real(irfft2(F / kk, s=(h, w)))            # back to (H,W)
    BN = (BN - BN.min()) / (BN.max() - BN.min() + 1e-6)
    return BN.astype(np.float32)

def pink_noise(h, w, rng):
    white = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    F = rfft2(white)                                  # (H, W//2+1)
    ky = np.fft.fftfreq(h)[:, None]                   # (H,1)
    kx = np.fft.rfftfreq(w)[None, :]                  # (1,W//2+1)
    kk = np.sqrt(kx**2 + ky**2) + 1e-3
    P = np.real(irfft2(F / kk, s=(h, w)))
    P = (P - P.min()) / (P.max() - P.min() + 1e-6)
    return P.astype(np.float32)


def bandlimited_1d(h, rng, sigma_rows):
    x = rng.normal(0,1,(h,)).astype(np.float32)
    return convolve_y(x, sigma_rows)

# --- NEW: resolution-invariant horizontal scan mask
def _scan_mask(H, W, period_px=2.0, duty=0.52, depth=0.28, jitter_px=0.15, seed=0):
    rng = RNG(seed+11)
    y = np.arange(H, dtype=np.float32)
    # per-row subpixel jitter
    j = rng.normal(0.0, jitter_px, size=H).astype(np.float32)
    phase = (y + j) / max(1e-6, period_px)                # cycles
    # smooth “thin dark bar” gate
    s = 0.35                                              # softness in cycles
    wave = 0.5*(1.0 + np.cos(2*np.pi*(phase - duty)))     # 0..1
    gate = 1.0 - np.exp(- (wave/s)**2 )                   # darker near crest
    mask = 1.0 - depth*gate                               # 1 in bright rows, (1-depth) in dark rows
    return np.tile(mask[:, None], (1, W)).astype(np.float32)


# ---------- core renderer ----------
def apply(
    image: Image.Image,
    *,
    seed: int = 2025,
    # triad / scan
    triad_period_px: float = 3.6,
    triad_jitter: float = 0.01,
    duty_r: float = 0.31, duty_g: float = 0.34, duty_b: float = 0.31,
    scan_gap: float = 0.22,           # line dark gap [0..1]
    wobble_amp: float = 0.10,         # px
    # beam/bloom
    sigma_x: float = 0.35,
    sigma_y0: float = 0.9,
    sigma_y_bright_scale: float = 0.8,
    bloom_lambda_y: float = 2.6,
    bloom_sigma_x: float = 0.5,
    trails_D: int = 4,
    trails_tau: float = 2.5,
    # defects/grain
    defect_rate: float = 0.008,
    banding_strength: float = 0.01,
    pink_amount: float = 0.04,
    poisson_amount: float = 0.04,
    chroma_skew: float = 0.08,
    quant_bits: int = 8,
    dither_amount: float = 0.6,
    # grading & masks
    gamma_shadow: float = 0.9,
    gamma_high: float = 1.35,
    glass_amount: float = 0.03,
    vignette: float = 0.18,
    shaft_x: float = 0.50,
    shaft_sigma: float = 0.06,
    triad_strength: float = 0.18,   # 0=off … 1=full per-channel
    triad_luma_mode: bool = True,   # apply mask to L only
    scan_strength: float = 0.85,    # line dark gap multiplier
    scan_period_px: float = 2.0,
    scan_duty: float = 0.52,
    scan_depth: float = 0.22,
    scan_jitter_px: float = 0.15,
):
    glass_amount *= 0.5            # halves it
    pink_amount  *= 0.5
    poisson_amount *= 0.5
    dither_amount *= 0.5

    rng = RNG(seed)
    img = to_np(image); lin = g2l(img)
    H,W,_ = lin.shape

    # 0) stochastic fields
    N_per = perlin(H, W, scale=220, rng=rng)
    N_w = worley(H, W, cells=42, rng=rng)
    N_blue = blue_noise_mask(H, W, rng=rng)
    N_pink = pink_noise(H, W, rng=rng)
    V = bandlimited_1d(H, rng, sigma_rows=7.0)  # per-line jitter
    J = rng.normal(0.0, 0.015, (H,W)).astype(np.float32)
    G = convolve_y(convolve_x(pink_noise(H,W,rng)*0.5 + perlin(H,W,600,rng)*0.5, 9.0), 9.0)

    # 1) luma–chroma
    L = luma(lin); C = lin / np.maximum(L[...,None], 1e-4)

    # 2) tone curve + cross-print glass
    Ls = np.power(np.clip(L,0,1), gamma_shadow)
    Lh = 1.0 - np.power(1.0 - Ls, gamma_high)
    L = np.clip(Lh + glass_amount*(G-0.5), 0, 1)

    rgb = C * L[...,None]

    # 3) triad mask (irregular)
    x = np.arange(W, dtype=np.float32)[None,:]
    y = np.arange(H, dtype=np.float32)[:,None]
    p = triad_period_px * (1.0 + triad_jitter*N_per[:,0:1])
    phi = 2*np.pi*(V[:,None]/(triad_period_px+1e-6))
    def trapezoid(u, period, duty):
        # u in pixels; duty fraction
        u = (u % period) / np.maximum(period,1e-6)
        # soft trapezoid 0..1
        ramp = 0.5 + 0.5*np.cos(np.clip((u - duty)*np.pi/duty, -np.pi, np.pi))
        core = (u < duty).astype(np.float32)
        return 0.5*core + 0.5*core*ramp
    dutyR = np.clip(duty_r + 0.03*J, 0.0, 1.0)
    dutyG = np.clip(duty_g + 0.03*J, 0.0, 1.0)
    dutyB = np.clip(duty_b + 0.03*J, 0.0, 1.0)
    T_R = trapezoid(x + phi, p, dutyR)
    T_G = trapezoid(x + phi + p/3.0, p, dutyG)
    T_B = trapezoid(x + phi + 2.0*p/3.0, p, dutyB)
    triad_mod = np.stack([T_R, T_G, T_B], axis=-1)
    triad_mod *= (1.0 + 0.04*N_w[...,None])  # was 0.08  # tiny cellular specks

    if triad_luma_mode:
        # collapse to scalar mask to avoid vertical rainbow columns
        triad_scalar = np.clip(np.mean(triad_mod, axis=-1), 0.0, 2.0)[..., None]
        rgb = rgb * (1.0 - 0.15 + 0.15*triad_scalar)   # 15% strength, luma-like
    else:
        rgb = rgb * (1.0 - triad_strength + triad_strength*np.clip(triad_mod, 0.0, 2.0))

    rgb *= np.clip(triad_mod, 0.0, 2.0)

    # 4) beam PSF + scanline gain + wobble
    sigma_y = sigma_y0*(1.0 + sigma_y_bright_scale*L)[...,None]
    rgb = convolve_x(rgb, sigma_x)
   
    # per-row variable vertical blur
    out = np.empty_like(rgb)
    for r in range(H):
        out[r:r+1,:,:] = convolve_y(rgb[r:r+1,:,:], float(sigma_y[r,0,0]) )
    rgb = out

    scan = _scan_mask(H, W, period_px=scan_period_px, duty=scan_duty,
                  depth=scan_depth, jitter_px=scan_jitter_px, seed=seed)
    # luma-first: apply to L, then rebuild RGB to avoid color tearing
    L_scan = np.clip(L * scan, 0, 1)
    rgb = C * L_scan[..., None]

    # line blanking
    base_gap = 0.5 + 0.5*np.sin(2*np.pi*y)

    # wobble sampling (per-row)
    # make a single shift per scanline from V plus a slow N_per component
    wobble_amp = min(wobble_amp, 0.15)
    shift_row = np.round(wobble_amp * (V + 0.2*np.mean(N_per, axis=1))).astype(int)
    for r in range(H):
        s = int(shift_row[r])
        if s: rgb[r,:,:] = np.roll(rgb[r,:,:], s, axis=1)

    # 5) vertical bloom / streak
    Hmask = (L_scan > 0.6).astype(np.float32)
    # directional kernel via separable exp along y
    # approximate with recursive blur:
    lam = bloom_lambda_y*(1.0 + 0.6*Hmask)
    bloom = np.zeros((H,W), dtype=np.float32)
    # forward pass
    acc = np.zeros(W, dtype=np.float32)
    for r in range(H):
        acc = acc*np.exp(-1.0/np.maximum(lam[r,:],1e-3)) + Hmask[r,:]
        bloom[r,:] = acc
    # backward tail
    acc[:] = 0
    for r in range(H-1, -1, -1):
        acc = acc*np.exp(-1.0/np.maximum(lam[r,:],1e-3)) + 0.3*Hmask[r,:]
        bloom[r,:] += acc
    
    bloom = convolve_x(bloom, bloom_sigma_x)
    rgb += 0.65 * bloom[...,None] * np.array([1.0-0.7*chroma_skew, 1.0, 1.0], np.float32)  # was bloom_gain

    grain_mono = (0.5*pink_amount*(N_pink-0.5)).astype(np.float32)  # halve pink
    poisson = RNG(seed+7).poisson(lam=0.5*poisson_amount*np.clip(L,0,1))*0.003

    # 6) phosphor decay trails
    trail = np.zeros_like(L)
    for d in range(1, int(trails_D)+1):
        trail += np.roll(Hmask, shift=d, axis=0) * np.exp(-d/np.maximum(trails_tau,1e-3))
    rgb += 0.25*trail[...,None]

    # 7) defects and banding
    if defect_rate > 0:
        sel = RNG(seed+3).random(H) < defect_rate
        rgb[sel,:,:] *= RNG(seed+4).uniform(0.92, 0.97, (sel.sum(),1,1)).astype(np.float32)
    P = RNG(seed+5).uniform(120, 320)
    rgb *= (1.0 + banding_strength*np.sin(2*np.pi*y/P + RNG(seed+6).uniform(0,2*np.pi)))[...,None]
    rgb = np.clip(rgb, 0.0, 2.0)

    # 8) grain
    grain_mono = (pink_amount*(N_pink-0.5)).astype(np.float32)
    poisson = RNG(seed+7).poisson(lam=poisson_amount*np.clip(L,0,1))*0.005
    rgb += (grain_mono + poisson)[...,None]
    rgb += 0.12*grain_mono[...,None]*np.array([0.0,1.0,0.0],np.float32)  # slight chroma grain

    # 9) chromatic mis-registration
    def subpix_shift(c, dx):
        j = int(np.floor(dx)); a = dx - j
        return (np.roll(c, j, axis=1)*(1-a) + np.roll(c, j+1, axis=1)*a)
    dr = RNG(seed+8).uniform(-0.25,0.25); db = RNG(seed+9).uniform(-0.25,0.25)
    rgb[...,0] = subpix_shift(rgb[...,0], dr)
    rgb[...,2] = subpix_shift(rgb[...,2], -db)

    # 10) quantize + blue-noise dither
    qlev = max(2, 2**quant_bits - 1)
    dither = (0.5*dither_amount)*(N_blue-0.5)[...,None]/qlev
    q = np.round((rgb + dither)*qlev)/qlev
    rgb = np.clip(q, 0, 1)

    # 11) color grade (simple teal split)
    M = np.array([[0.82,-0.05,0.05],
                  [0.00,0.95,0.05],
                  [0.03,0.05,1.05]], dtype=np.float32)
    rgb = np.tensordot(rgb, M.T, axes=1)

    # vignette + shaft
    yy, xx = np.mgrid[0:H,0:W].astype(np.float32)
    xx = (xx/W - shaft_x)
    shaft = np.exp(-(xx*xx)/(2*shaft_sigma*shaft_sigma))
    rgb *= (0.9 + 0.1*shaft)[...,None]
    r = np.sqrt(((yy-H/2)/(H/2))**2 + ((xx*W+W*shaft_x)/W - 0.5)**2)
    rgb *= (1.0 - vignette*np.clip(r, 0, 1))[...,None]

    # 12) two-scale unsharp (masked out of highlights)
    Hm = (L < 0.9).astype(np.float32)[...,None]
    def unsharp(img, rad, amt):
        blur = convolve_y(convolve_x(img, rad), rad)
        return np.clip(img + amt*(img - blur), 0, 1)
    rgb = rgb*Hm + unsharp(rgb, 0.6, 0.25)*(1.0*Hm) + unsharp(rgb, 2.2, 0.12)*(1.0*Hm)

    return from_np(l2g(rgb))
