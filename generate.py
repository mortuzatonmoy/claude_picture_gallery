# Procedural NATURE scenes - 5 images at 3840x2160 (4K UHD), pure NumPy
import os
import numpy as np
from PIL import Image, ImageFilter

W, H = 3840, 2160
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
Y /= H; X /= H                      # y in [0,1], x in [0,1.78]
AR = W / H


def save(img, name):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    im.save(f"images/{name}.jpg", quality=94)
    im.resize((1280, 720), Image.LANCZOS).save(f"images/thumbs/{name}.jpg", quality=85)
    print("saved", name)


_tables = {}
def vnoise(x, y, seed):
    if seed not in _tables:
        t = np.random.default_rng(seed).random((257, 257)).astype(np.float32); t[256, :] = t[0, :]; t[:, 256] = t[:, 0]; _tables[seed] = t
    r = _tables[seed]
    fx, fy = np.floor(x), np.floor(y)
    xi, yi = fx.astype(np.int32) % 256, fy.astype(np.int32) % 256
    u, v = x - fx, y - fy
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v)
    return (r[yi, xi] * (1 - u) + r[yi, xi + 1] * u) * (1 - v) + (r[yi + 1, xi] * (1 - u) + r[yi + 1, xi + 1] * u) * v


def fbm(x, y, seed, octaves=6, lac=2.0, gain=0.5):
    s, a, f, norm = 0, 0.5, 1, 0
    for o in range(octaves):
        s = s + a * vnoise(x * f, y * f, seed + o * 7); norm += a; a *= gain; f *= lac
    return s / norm


def ridge1d(x, seed, octaves=8):
    return fbm(x, np.full_like(x, seed * 1.37), seed, octaves)


def mix(a, b, t): return a + (b - a) * t
def C(*c): return np.array(c, np.float32)
def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
def blur(img, r):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im).astype(np.float32) / 255


def pine_mask(cx, base, h, w, seed):
    """Return (slices, mask) for one jagged pine-tree silhouette."""
    rng = np.random.default_rng(seed)
    y0, y1 = int(max(base - h, 0)), int(min(base + h * 0.04, H))
    x0, x1 = int(max(cx - w, 0)), int(min(cx + w, W))
    if y1 <= y0 or x1 <= x0:
        return None, None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    t = (yy - (base - h)) / h
    tiers = rng.integers(6, 10)
    saw = (t * tiers) % 1.0
    half = w * t * (0.55 + 0.45 * saw) * (0.85 + 0.3 * vnoise(xx * 0.08, yy * 0.08, seed % 97))
    m = np.abs(xx - cx) < half
    trunk = (np.abs(xx - cx) < w * 0.06) & (t > 0.9)
    return (slice(y0, y1), slice(x0, x1)), (m | trunk)


def paint(img, sl, m, col):
    if sl is not None:
        region = img[sl]
        region[m] = col


# ------------------------------------------------ 1. Mountain lake at sunrise
def mountain_lake():
    hz = 0.64
    sky = mix(C(1.0, .78, .55), C(.22, .38, .70), (smooth(0.0, hz, hz - Y) ** 0.7)[..., None])
    d = np.hypot(X - 1.15, Y - 0.52)
    sky = sky + (np.exp(-d * 9) * 0.6)[..., None] * C(1, .75, .45) + np.exp(-d * 40)[..., None] * C(1, .9, .7)
    cl = fbm(X * 2.2 + fbm(X * 3, Y * 3, 5) * 0.8, Y * 7, 11)
    cmask = smooth(0.52, 0.72, cl) * smooth(0.05, 0.25, Y) * (1 - smooth(0.40, 0.55, Y))
    ccol = mix(C(1, .62, .55), C(1, .9, .8), smooth(0, .4, np.exp(-d * 3))[..., None])
    img = mix(sky, ccol, cmask[..., None] * 0.85)
    xs = X[0]
    layers = [(0.40, 0.30, C(.42, .40, .52), 1.0, 0.45, 21),
              (0.50, 0.20, C(.30, .32, .40), 0.6, 0.28, 33),
              (0.58, 0.12, C(.20, .25, .28), 0.0, 0.14, 45)]
    for base, amp, rock, snow, fog, sd in layers:
        r = ridge1d(xs * 1.6, sd)
        r = (1 - np.abs(r * 2 - 1)) ** 1.6
        top = base - amp * r + 0.05
        m = Y > top[None, :]
        tex = fbm(X * 9, Y * 6, sd + 3, 6)
        gx = np.gradient(fbm(X * 5 + tex * 0.3, Y * 9, sd + 9, 6), axis=1) * H
        shade = np.clip(0.75 + gx * 0.9, 0.25, 1.4)
        col = rock * shade[..., None] * (0.8 + 0.3 * tex[..., None])
        depth = Y - top[None, :]
        if snow > 0:
            sn = smooth(0.0, 1.0, (amp * 0.45 * snow - depth) / (amp * 0.25) + (tex - .5) * 0.5)
            snowc = mix(C(.55, .58, .75), C(1, .88, .82), np.clip(shade - .5, 0, 1)[..., None])
            col = mix(col, snowc, sn[..., None])
        col = mix(col, C(.85, .75, .75), fog)
        img = np.where(m[..., None], col, img)
    tl = hz - 0.025 - 0.03 * ridge1d(xs * 12, 77, 5)
    img = np.where((Y > tl[None, :])[..., None], C(.05, .09, .07), img)
    rng = np.random.default_rng(3)
    for i in range(420):
        cx = rng.uniform(0, W); xi = int(np.clip(cx, 0, W - 1))
        b = (tl[xi] + 0.02) * H; h = rng.uniform(0.03, 0.09) * H
        sl, m = pine_mask(cx, b, h, h * 0.22, i + 1000)
        paint(img, sl, m, C(.04, .07, .06) * rng.uniform(.8, 1.3))
    # lake reflection with ripples
    rows = np.arange(H)
    ripple = (np.sin(Y * H * 0.35 + vnoise(X * 40, Y * 400, 9) * 6) * 3
              + (fbm(X * 30, Y * 300, 17, 3) - .5) * 14).astype(np.int32)
    src_r = np.broadcast_to(np.clip(2 * hz * H - rows, 0, H - 1).astype(np.int32)[:, None], (H, W))
    src_c = np.clip(np.arange(W)[None, :] + ripple, 0, W - 1)
    refl = blur(img[src_r, src_c] * C(.62, .72, .80) + C(.02, .04, .07), 2)
    img = np.where((Y > hz)[..., None], refl, img)
    save(img, "01_mountain_lake_sunrise")


# ------------------------------------------------ 2. Misty pine forest with sun rays
def misty_forest():
    img = mix(C(.86, .90, .86), C(.55, .68, .70), Y[..., None])
    sx, sy = 0.55, -0.15
    ang = np.arctan2(Y - sy, X - sx)
    rays = smooth(0.45, 0.8, fbm(ang * 14, np.zeros_like(ang), 3, 3)) * smooth(1.4, 0.2, np.hypot(X - sx, Y - sy))
    rng = np.random.default_rng(8)
    nl = 7
    for L in range(nl):
        depth = L / (nl - 1)
        col = mix(C(.70, .78, .76), C(.03, .07, .05), depth ** 0.8)
        ground = 0.50 + 0.42 * depth ** 1.3 + 0.04 * ridge1d(X[0] * 3, L + 50, 4)
        img = np.where((Y > ground[None, :])[..., None], col, img)
        for i in range(int(140 - 14 * L)):
            cx = rng.uniform(-50, W + 50); xi = int(np.clip(cx, 0, W - 1))
            h = rng.uniform(0.10, 0.20) * (0.5 + 2.6 * depth ** 2) * H
            sl, m = pine_mask(cx, ground[xi] * H + 10, h, h * rng.uniform(.13, .19), L * 1000 + i)
            paint(img, sl, m, col * rng.uniform(.9, 1.1))
        img = mix(img, C(.93, .95, .88), np.clip(rays * (0.35 - 0.045 * L), 0, 1)[..., None])
        if L < nl - 1:
            fog = smooth(0.35, 0.9, Y) * 0.12 * fbm(X * 2, Y * 4 + L, 200 + L, 4)
            img = mix(img, C(.85, .9, .86), fog[..., None])
    grass = fbm(X * 200, Y * 20, 99, 3)
    fg = smooth(0.85, 1, Y)
    img = mix(img, C(.05, .12, .04), fg[..., None] * 0.5)
    img = img * (0.9 + 0.3 * (grass - .5) * fg)[..., None]
    save(img, "02_misty_pine_forest")


# ------------------------------------------------ 3. Ocean sunset
def ocean_sunset():
    hz = 0.55
    sky = mix(C(.18, .14, .38), C(.95, .45, .22), (smooth(0, hz, Y) ** 1.6)[..., None])
    sky = mix(sky, C(1, .78, .40), smooth(hz - 0.12, hz, Y)[..., None] * 0.8)
    sx = AR * 0.5
    d = np.hypot(X - sx, Y - (hz - 0.07))
    sky = sky + np.exp(-d * 6)[..., None] * C(1, .55, .2) * 0.6
    sky = np.where((d < 0.055)[..., None], C(1, .93, .75), sky)
    cl = fbm(X * 1.8, Y * 9, 31, 7)
    cm = smooth(0.5, 0.75, cl) * (1 - smooth(hz - 0.12, hz - 0.02, Y))
    lit = np.exp(-np.abs(X - sx) * 1.3)
    ccol = mix(C(.35, .15, .30), C(1, .55, .35), lit[..., None] * (1 - cm[..., None] * 0.5))
    img = mix(sky, ccol, cm[..., None])
    p = np.maximum(Y - hz, 1e-3)
    u = (X - sx) / p; v = 1 / p
    w = fbm(u * 0.6, v * 1.5, 61, 6)
    sea = mix(C(.30, .18, .30), C(.05, .05, .15), smooth(hz, 1, Y)[..., None])
    band = np.exp(-np.abs(X - sx) / (0.04 + p * 0.9))
    glit = smooth(0.55, 0.8, w) * band
    sea = sea + glit[..., None] * C(1, .75, .4) * 1.4 + band[..., None] * C(.5, .25, .1) * 0.4
    sea = sea * (0.85 + 0.3 * w)[..., None]
    img = np.where((Y > hz)[..., None], sea, img)
    shore = 0.86 + 0.03 * np.sin(X[0] * 3) + 0.02 * ridge1d(X[0] * 4, 7, 4)
    sand = mix(C(.45, .28, .22), C(.15, .08, .09), smooth(0.86, 1, Y)[..., None])
    sand = sand * (0.9 + 0.2 * fbm(X * 300, Y * 300, 4, 2))[..., None]
    foam = smooth(0.012, 0, np.abs(Y - shore[None, :])) * fbm(X * 60, Y * 60, 8, 3)
    img = np.where((Y > shore[None, :])[..., None], sand, img)
    img = mix(img, C(1, .85, .75), foam[..., None] * 0.9)
    save(img, "03_ocean_sunset")


# ------------------------------------------------ 4. Milky Way over the hills
def starry_night():
    img = mix(C(.01, .015, .04), C(.06, .09, .18), smooth(0.2, 0.9, Y)[..., None])
    bx = (X * 0.55 + Y * 0.84) - 0.95
    along = X * 0.84 - Y * 0.55
    band = np.exp(-(bx / 0.30) ** 2)
    neb = fbm(along * 3 + fbm(X * 4, Y * 4, 1) * 0.5, bx * 6, 2, 7)
    dust = smooth(0.45, 0.7, fbm(along * 6, bx * 14, 3, 6))
    glow = band * (0.4 + neb) * (1 - dust * 0.8 * np.exp(-(bx / 0.08) ** 2))
    img = img + glow[..., None] * mix(C(.35, .30, .45), C(.95, .8, .65), np.exp(-(bx / 0.09) ** 2)[..., None]) * 0.9
    rng = np.random.default_rng(1)
    n = 26000
    sx, sy = rng.integers(0, W, n), rng.integers(0, int(H * .85), n)
    bright = rng.pareto(2.2, n) * 0.25 + 0.1
    keep = rng.random(n) < (0.35 + band[sy, sx])
    stars = np.zeros((H, W), np.float32)
    np.add.at(stars, (sy[keep], sx[keep]), bright[keep])
    halo = blur(np.repeat(np.clip(stars, 0, 1)[..., None], 3, 2), 2.5)[..., 0]
    img = img + (stars + halo * 3.5)[..., None] * C(.95, .95, 1.0)
    img = img + (smooth(0.5, 0.75, Y) * smooth(0.95, 0.75, Y))[..., None] * C(.10, .08, .12)
    h1 = 0.82 - 0.10 * ridge1d(X[0] * 1.2, 5, 6)
    h2 = 0.90 - 0.07 * ridge1d(X[0] * 2.0, 9, 7)
    img = np.where((Y > h1[None, :])[..., None], C(.025, .03, .05), img)
    img = np.where((Y > h2[None, :])[..., None], C(.008, .01, .015), img)
    trng = np.random.default_rng(4)
    for i in range(40):
        cx = trng.uniform(0, W); xi = int(np.clip(cx, 0, W - 1)); h = trng.uniform(.06, .16) * H
        sl, m = pine_mask(cx, h2[xi] * H + 8, h, h * 0.2, 5000 + i)
        paint(img, sl, m, C(.006, .008, .012))
    save(img, "04_milky_way_night")


# ------------------------------------------------ 5. Green meadow & cumulus clouds
def green_meadow():
    img = mix(C(.25, .50, .88), C(.70, .85, .97), smooth(0, 0.6, Y)[..., None])
    n = fbm(X * 1.6, Y * 3.2, 70, 7)
    n2 = fbm(X * 1.6 + 0.01, Y * 3.2 - 0.02, 70, 7)
    dens = smooth(0.50, 0.68, n) * smooth(0.55, 0.15, Y)
    light = np.clip((n - n2) * 25 + 0.75, 0.5, 1.1)
    img = mix(img, mix(C(.62, .66, .76), C(1, 1, .98), light[..., None]), dens[..., None])
    hills = [(0.58, .08, C(.45, .62, .55), 1), (0.66, .07, C(.38, .62, .28), 2),
             (0.74, .09, C(.30, .55, .16), 3), (0.84, .10, C(.20, .45, .08), 4)]
    rng = np.random.default_rng(12)
    for base, amp, col, s in hills:
        top = base - amp * (0.5 + 0.5 * np.sin(X[0] * (1.5 + s * 0.4) + s)) * (0.6 + 0.8 * ridge1d(X[0], s + 40, 3))
        tex = fbm(X * (30 + s * 40), Y * (30 + s * 40), s + 90, 4)
        sh = 0.85 + 0.35 * smooth(0.12, 0, Y - top[None, :]) + 0.2 * (tex - .5)
        img = np.where((Y > top[None, :])[..., None], col * sh[..., None], img)
        if s <= 2:  # round trees on far hills
            for i in range(260):
                cx = int(rng.uniform(40, W - 40)); rr = int(rng.uniform(.005, .011) * H / s)
                cy = int(top[cx] * H + rng.uniform(rr, rr * 4))
                ys, xs_ = np.ogrid[-rr:rr, -rr:rr]
                dm = (xs_ ** 2 + (ys * 1.2) ** 2) < rr * rr
                if cy + rr < H and cx - rr >= 0 and cx + rr < W:
                    paint(img, (slice(cy - rr, cy + rr), slice(cx - rr, cx + rr)), dm, C(.16, .30, .16) * rng.uniform(.8, 1.1))
    gf = fbm(X * 500, Y * 25, 300, 3)
    fg = smooth(0.84, 1, Y)
    img = img * (1 + (gf - .5) * 0.6 * fg)[..., None]
    frng = np.random.default_rng(21); nf = 9000
    fy = (0.80 + 0.2 * frng.random(nf) ** 0.6) * H; fx = frng.uniform(0, W, nf)
    sz = ((fy / H - 0.78) * 30).astype(int) + 1
    cols = [C(1, .95, .2), C(1, 1, 1), C(.9, .25, .3), C(.7, .4, .95)]
    for x_, y_, s_, k in zip(fx.astype(int), fy.astype(int), sz, frng.integers(0, 4, nf)):
        if y_ > top[min(x_, W - 1)] * H + 5:
            img[max(y_ - s_, 0):y_ + s_, max(x_ - s_, 0):x_ + s_] = cols[k] * frng.uniform(.85, 1)
    save(img, "05_green_meadow")


if __name__ == "__main__":
    os.makedirs("images/thumbs", exist_ok=True)
    for f in (mountain_lake, misty_forest, ocean_sunset, starry_night, green_meadow):
        f()
