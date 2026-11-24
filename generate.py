# Procedural art generator - 5 images at 3840x2160 (4K UHD)
import numpy as np
from PIL import Image
W, H = 3840, 2160
def save(a, name):
    Image.fromarray(np.clip(a*255,0,255).astype(np.uint8)).save(f"images/{name}.jpg", quality=95)
    print("saved", name)
def grid(cx, cy, scale):
    x = np.linspace(cx-scale, cx+scale, W)
    y = np.linspace(cy-scale*H/W, cy+scale*H/W, H)
    return x[None,:] + 1j*y[:,None]
def palette(t, a, b, c, d):
    return a + b*np.cos(2*np.pi*(c*t[...,None]+d))

# 1. Mandelbrot (smooth coloring)
c = grid(-0.7436447860, 0.1318252536, 0.0035); z = np.zeros_like(c); n = np.zeros(c.shape)
alive = np.ones(c.shape, bool)
for i in range(1200):
    z[alive] = z[alive]**2 + c[alive]
    esc = alive & (np.abs(z) > 256); n[esc] = i + 1 - np.log2(np.log(np.abs(z[esc]))); alive &= ~esc
t = np.sqrt(n/ n.max())*4
img = palette(t,(.5,.5,.5),(.5,.5,.5),(1,1,1),(0,.1,.2)); img[alive]=0
save(img, "01_mandelbrot_spiral")

# 2. Julia set
z = grid(0,0,1.6); cj = -0.8+0.156j; n = np.zeros(z.shape); alive = np.ones(z.shape,bool)
for i in range(500):
    z[alive] = z[alive]**2 + cj
    esc = alive & (np.abs(z) > 64); n[esc] = i + 1 - np.log2(np.log(np.abs(z[esc]))); alive &= ~esc
t = (n/60)**0.6
img = palette(t,(.5,.5,.5),(.5,.5,.5),(1,1,.5),(.8,.9,.3)); img[alive]=(0.02,0,0.05)
save(img, "02_julia_bloom")

# 3. Nebula - domain-warped fractal noise
rng = np.random.default_rng(7)
def vnoise(x, y, seed):
    r = np.random.default_rng(seed).random((257,257))
    xi, yi = np.floor(x).astype(int)%256, np.floor(y).astype(int)%256
    xf, yf = x-np.floor(x), y-np.floor(y); u, v = xf*xf*(3-2*xf), yf*yf*(3-2*yf)
    return (r[yi,xi]*(1-u)+r[yi,xi+1]*u)*(1-v)+(r[yi+1,xi]*(1-u)+r[yi+1,xi+1]*u)*v
def fbm(x, y, seed, oct=7):
    s, a = 0, .5
    for o in range(oct): s += a*vnoise(x*2**o, y*2**o, seed+o); a *= .5
    return s
yy, xx = np.mgrid[0:H,0:W]/H*3.0
qx, qy = fbm(xx,yy,1), fbm(xx+5.2,yy+1.3,20)
f = fbm(xx+4*qx, yy+4*qy, 40)
img = np.stack([f**2*1.6+qx*.3, f**3*1.2+qy*.4, f*0.9+qx*qy*.8],-1)
img = img**1.3*1.4
stars = rng.random((H,W)); img[stars>0.9993] = 1
save(img, "03_cosmic_nebula")

# 4. Mountain sunset landscape
Y, X = np.mgrid[0:H,0:W]/np.array([H,W])[:,None,None]
sky = np.stack([.98-.6*Y**.6+0*X, .55-.4*Y, .35+.3*(1-Y)],-1)
sky = np.stack([1-0.7*(1-Y), .45*Y+.1, .55*(1-Y)+.2],-1)
d = np.hypot((X-.5)*W/H, Y-.62); img = sky + np.clip(1-d*4,0,1)[...,None]**2*np.array([1,.8,.4])
img += np.exp(-d*6)[...,None]*np.array([.5,.3,.1])
x1 = np.linspace(0,8,W)
for k,(base,amp,col) in enumerate([(.55,.12,(.55,.25,.45)),(.65,.10,(.35,.15,.35)),(.75,.09,(.2,.08,.22)),(.86,.07,(.07,.02,.08))]):
    ridge = base - amp*fbm(x1*(1+k*.6), np.full(W,k*3.3), 100+k*10, 8)*1.5 + amp*.6
    mask = Y > ridge[None,:]
    haze = np.clip((Y-ridge[None,:])*3,0,1)[...,None]
    img = np.where(mask[...,None], np.array(col)*(1-.3*haze)+0.15*(1-haze)*img, img)
save(img, "04_sunset_ranges")

# 5. Quasicrystal mandala (9-fold wave interference)
Y5, X = (np.mgrid[0:H,0:W]-np.array([H/2,W/2])[:,None,None])/H*140
s = np.zeros((H,W))
for k in range(9):
    a = np.pi*k/9; s += np.cos(X*np.cos(a)+Y5*np.sin(a))
s = s/9; r = np.hypot(X,Y5)
t = s*0.8 + r*.004
img = palette(t,(.5,.5,.5),(.5,.5,.5),(1,1,1),(.0,.33,.67))*(0.15+0.85*((s+1)/2)[...,None]**1.5)
img *= np.clip(1.25-r/80,0,1)[...,None]**0.7
save(img, "05_quasicrystal_mandala")
