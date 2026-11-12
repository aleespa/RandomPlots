"""
Inside Out

A blue sphere turns itself inside out. No cut, no tear, no crease: when it is
round again, the gold that was its inside is now its outside.

In 1957 Stephen Smale proved that this is possible, as long as the surface is
allowed to pass through itself, and for years nobody could picture how. The
obvious move, pushing the north pole down through the south, fails: the belt
around the equator ends up pinched into a tight crease, and creases are not
allowed. The trick, found by Bill Thurston, is to first ripple the sphere into
waves. The waves give the surface room to slide through itself, so the two
poles can pass, the belt can turn half a twist, and every wave can be smoothed
away again on the other side.

The music follows the turning: low strings grow tense as the poles push
through, a bell marks each step, and the home chord lands when the sphere is
whole again, gold side out.

Made with Python & PyTorch. Sound synthesized in NumPy.
#sphereeversion #topology #smaleparadox #mathart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F

from common import audio
from common.gpu import select_device
from common.music import Scale
from common.image_processing import ImageProcessingSettings
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# Thurston's corrugated eversion, after Nathaniel Thurston's `evert`
# (Geometry Center, 1995, Apache 2.0). Times are on evert's own 0..1 clock.
N_STRIPS = 8
STAGES = (  # name, evert time it starts at, video second it starts at
    ("corrugate", 0.0, 0.35),
    ("push", 0.1, 2.6),
    ("twist", 0.23, 5.4),
    ("unpush", 0.6, 9.4),
    ("uncorrugate", 0.93, 12.6),
)
END = 14.0  # s: round again, resting until DURATION
NU, NV = 192, 96  # grid of one patch (u: pole to equator, v: one strip)
EPS = 1e-5  # step of the central differences that stand in for evert's jets

# Camera
ELEVATION = np.radians(24.0)
AZIMUTH = np.radians(35.0)
ORBIT = np.radians(70.0)  # turned over the whole video
DISTANCE = 7.0  # sphere radii
RADIUS_PX = 420.0  # on-screen radius of the round sphere
MARGIN = 0.9  # widest the shape may get, as a fraction of the frame width
CENTRE = (WIDTH / 2, HEIGHT * 0.47)
SS = 2  # supersampling
CHUNK = 5_000_000  # candidate pixels per rasterizer pass

# Look
BACKGROUND = "#05070d"
OUTSIDE = ("#0e1f5c", "#2f5fe0", "#8fb1ff")  # shadow, body, light
INSIDE = ("#4a2406", "#d98a22", "#ffe0a0")
LIGHT = (-0.45, 0.55, 0.70)  # key light, world-ish (towards the viewer, up-left)
FILL = (0.7, -0.2, 0.2)
SHINE = 70.0
LINES_U, LINES_V = 8, 2  # grid lines per hemisphere / per strip
LINE_DARK = 0.32

# Sound
ROOTS = range(50, 56)  # D3 .. G3: low strings
BELL_OCTAVE = 2  # bells this many octaves over the root
ROOM = (3.2, 0.36, 4500.0)  # rt60, wet, tone: a dark hall

TAU = 2 * np.pi


# ---------------------------------------------------------------- the eversion
# Angles are in turns (evert's Sin/Cos multiply by 2 pi).
def _sin(x):
    return torch.sin(TAU * x)


def _cos(x):
    return torch.cos(TAU * x)


def _lerp(a, b, w):
    w = torch.as_tensor(w, dtype=a.dtype, device=a.device)
    if w.ndim:
        w = w[..., None]
    return a * (1 - w) + b * w


def _normalize(p):
    return p / torch.sqrt((p * p).sum(-1, keepdim=True).clamp_min(1e-30))


def _rotate_z(p, a):
    s, c = _sin(a), _cos(a)
    x, y, z = p.unbind(-1)
    return torch.stack((x * c + y * s, -x * s + y * c, z), -1)


def _rotate_y(p, a):
    s, c = _sin(a), _cos(a)
    x, y, z = p.unbind(-1)
    return torch.stack((x * c - z * s, y, x * s + z * c), -1)


def _arc(u, v, xs, ys, zs):
    u = u * 0.25
    return torch.stack(
        (_sin(u) * _sin(v) * xs, _sin(u) * _cos(v) * ys, _cos(u) * zs), -1
    )


def _fold(x, period):
    """x mod period, then the two halves of [0, 2] (evert's `x %= ...; if x > 1`)."""
    x = torch.remainder(x, period)
    return torch.where(x > 1, 2 - x, x)


def _param1(x):
    x = torch.remainder(x, 4)
    high = x > 2
    x = torch.where(high, x - 2, x)
    return torch.where(x <= 1, 2 * x - x * x, x * x - 2 * x + 2) + 2.0 * high


def _param2(x):
    x = torch.remainder(x, 4)
    high = x > 2
    x = torch.where(high, x - 2, x)
    return torch.where(x <= 1, x * x, -x * x + 4 * x - 2) + 2.0 * high


def _u_interp(x):
    x = _fold(x, 2)
    return 3 * x**2 - 2 * x**3


def _ff_interp(x):
    x = _fold(x, 2) * 1.06 - 0.05
    return torch.where(x < 0, 0.0, torch.where(x > 1, 1.0, 3 * x**2 - 2 * x**3))


def _fs_interp(x):
    x = _fold(x, 2)
    return (3 * x**2 - 2 * x**3) * -0.2


def _stage1(u, v):
    return _arc(u, v, 1, 1, 1)


def _stage2(u, v):
    return _lerp(
        _arc(_param1(u), v, 0.9, 0.9, -1), _arc(_param2(u), v, 1, 1, 0.5), _u_interp(u)
    )


def _stage3(u, v):
    return _lerp(
        _arc(_param1(u), v, -0.9, -0.9, -1),
        _arc(_param2(u), v, -1, 1, -0.5),
        _u_interp(u),
    )


def _stage4(u, v):
    return _arc(u, v, -1, -1, -1)


def _scene23(u, v, t, side):
    t = t * 0.5
    tt = torch.where(side <= 1, t, -t)
    return _lerp(
        _rotate_z(_arc(_param1(u), v, 0.9, 0.9, -1), tt),
        _rotate_y(_arc(_param2(u), v, 1, 1, 0.5), torch.full_like(u, t)),
        _u_interp(u),
    )


def _stage(time: float):
    """evert's scene at `time` in 0..1: (name, local time 0..1)."""
    starts = [start for _, start, _ in STAGES] + [1.0]
    for (name, start, _), end in zip(STAGES, starts[1:]):
        if time < end or name == "uncorrugate":
            return name, min(max((time - start) / (end - start), 0.0), 1.0)


def _base(u, v, name, t, side):
    """
    The surface the corrugations ride on (a ThreeJetVec in evert); `side` is the
    u that picks the branch of the twist (u itself, unless differentiating).
    """
    if name in ("corrugate", "uncorrugate"):
        return _stage1(u, v) if name == "corrugate" else _stage4(u, v)
    if name == "push":
        return _lerp(_stage1(u, v), _stage2(u, v), t)
    if name == "twist":
        return _scene23(u, v, t, side)
    return _lerp(_stage3(u, v), _stage4(u, v), t)


def _form(u, name, t):
    if name == "corrugate":
        return _ff_interp(u) * t
    if name == "uncorrugate":
        return _ff_interp(u) * (1 - t)
    return _ff_interp(u)


def patch(u, v, time: float):
    """
    One patch of the everting sphere at evert time `time`: u in 0..1 (pole to
    equator) by v in 0..1 (one of N_STRIPS strips) -> (NU, NV, 3).
    evert's AddFigureEight + FigureEight, with its jets (values + derivatives)
    replaced by central differences in u and v, all in one batch.
    """
    name, t = _stage(time)
    n, e, z = u.numel(), torch.full_like(u, EPS), torch.zeros_like(u)
    at = _base(
        torch.cat((u, u - e, u + e, u, u)),
        torch.cat((z, z, z, -e, e)),
        name,
        t,
        u.repeat(5),
    ).view(5, n, 3)
    p, p_u, p_v = at[0], (at[2] - at[1]) / (2 * EPS), (at[4] - at[3]) / (2 * EPS)
    near = torch.cat((u, u - e, u + e))
    size = (_form(near, name, t) * _fs_interp(near)).view(3, n)
    size, size_u = size[0], (size[2] - size[1]) / (2 * EPS)
    form = _form(u, name, t)
    form = form * 2 - form * form

    du = _normalize(p_u)
    h = _normalize(torch.cross(du, p_v, dim=-1)) * size[:, None]
    w = _normalize(torch.cross(h, du, dim=-1)) * (size * 1.1)[:, None]
    bend = du * size_u[:, None]

    vv = torch.remainder(v, 1)
    height = 1 - _cos(2 * vv)
    height = torch.where((vv > 0.25) & (vv < 0.75), 4 - height, height) * 0.6
    hh = h[:, None] + bend[:, None] * (height * height / 64)[None, :, None]
    loop = 2 * (1 - _cos(vv))[None, :]
    profile = loop + (height[None, :] - loop) * form[:, None]
    surface = (
        p[:, None] + w[:, None] * _sin(2 * vv)[None, :, None] + hh * profile[..., None]
    )
    return _rotate_z(surface, (v / N_STRIPS)[None, :])


def replicas(device):
    """evert -whole: the 2 x N_STRIPS rigid copies of a patch, as (16, 3, 3) for p @ M."""
    mats = []
    for j in (1, -1):
        for k in range(N_STRIPS):
            a = TAU * (k if j > 0 else N_STRIPS - 1 - k) / N_STRIPS
            s, c = np.sin(a), np.cos(a)
            mats.append([[j * c, -s, 0], [j * s, c, 0], [0, 0, j]])
    return torch.tensor(mats, dtype=torch.float64, device=device)


def evert_time(seconds: float) -> float:
    """Video seconds -> evert's clock, easing through every stage."""
    keys = [(start, at) for _, start, at in STAGES] + [(1.0, END)]
    if seconds <= keys[0][1]:
        return 0.0
    if seconds >= END:
        return 1.0
    for i in range(len(keys) - 1):
        (a, s0), (b, s1) = keys[i], keys[i + 1]
        if seconds < s1:
            x = (seconds - s0) / (s1 - s0)
            ease = float(smoothstep(x))
            if i == 0:  # ease in from rest
                x = ease if x > 0.5 else 0.5 * ease + 0.5 * x * x
            elif i == len(keys) - 2:  # ease out to rest
                x = ease if x < 0.5 else 0.5 * ease + 0.5 * (1 - (1 - x) ** 2)
            else:
                x = 0.55 * x + 0.45 * ease
            return a + (b - a) * x
    return 1.0


# ------------------------------------------------------------------ rendering
def _rgb(colour: str):
    """'#rrggbb' -> linear RGB (the frame is gamma-encoded at the end)."""
    return [(int(colour[i : i + 2], 16) / 255.0) ** 2.2 for i in (1, 3, 5)]


class Renderer:
    """Triangle rasterizer with a z-buffer on the GPU, shading both sides."""

    def __init__(self, device):
        self.device = device
        f32 = dict(device=device, dtype=torch.float32)
        f64 = dict(device=device, dtype=torch.float64)
        self.u = torch.linspace(1e-6, 1.0, NU, **f64)
        self.v = torch.linspace(0.0, 1.0, NV, **f64)
        self.mats = replicas(device)
        copies = self.mats.shape[0]

        # Two triangles per grid quad, per copy.
        idx = torch.arange(NU * NV, device=device).view(NU, NV)
        a, b = idx[:-1, :-1].reshape(-1), idx[1:, :-1].reshape(-1)
        c, d = idx[1:, 1:].reshape(-1), idx[:-1, 1:].reshape(-1)
        tris = torch.cat((torch.stack((a, b, c), 1), torch.stack((a, c, d), 1)))
        offset = torch.arange(copies, device=device)[:, None, None] * (NU * NV)
        self.tris = (tris[None] + offset).reshape(-1, 3)
        uu = self.u.float()[:, None].expand(NU, NV)
        vv = self.v.float()[None, :].expand(NU, NV)
        self.uv = torch.stack((uu, vv), -1).reshape(1, -1, 2).expand(copies, -1, -1)
        self.uv = self.uv.reshape(-1, 2)

        # Which way the grid normal points out of the round sphere, per copy.
        p = self._world(0.0)
        n = self._normals(p)
        self.orient = torch.sign((n * p).sum(-1).mean((1, 2)))[:, None, None, None]

        # Zoom: as large as the frame allows at every moment of the eversion.
        clock = np.linspace(0.0, 1.0, 161)
        fit = []
        for time in clock:
            q = self._world(float(time))
            across = float(torch.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2).max())
            tall = float(q[..., 2].abs().max())
            tall = tall * np.cos(ELEVATION) + across * np.sin(ELEVATION)
            fit.append(
                min(RADIUS_PX, MARGIN * WIDTH / 2 / across, 0.8 * HEIGHT / 2 / tall)
            )
        fit = np.array(fit)
        fit = np.array([fit[max(0, i - 8) : i + 9].min() for i in range(len(fit))])
        kernel = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2)
        fit = np.convolve(np.pad(fit, 12, mode="edge"), kernel / kernel.sum(), "valid")
        self.zoom_clock, self.zoom = clock, fit

        self.width, self.height = WIDTH * SS, HEIGHT * SS
        self.background = torch.tensor(_rgb(BACKGROUND), **f32)
        self.outside = [torch.tensor(_rgb(c), **f32) for c in OUTSIDE]
        self.inside = [torch.tensor(_rgb(c), **f32) for c in INSIDE]
        ys = (torch.arange(self.height, **f32) + 0.5) / self.height - 0.47
        xs = (torch.arange(self.width, **f32) + 0.5) / self.width - 0.5
        r2 = xs[None, :] ** 2 * (WIDTH / HEIGHT) ** 2 + ys[:, None] ** 2
        self.glow = torch.exp(-r2 / 0.06)[..., None]

    def _world(self, time):
        p = patch(self.u, self.v, time)  # (NU, NV, 3)
        return torch.einsum("uvi,cij->cuvj", p, self.mats)  # (copies, NU, NV, 3)

    @staticmethod
    def _normals(p):
        pu = torch.gradient(p, dim=1)[0]
        pv = torch.gradient(p, dim=2)[0]
        n = torch.cross(pu, pv, dim=-1)
        n[:, 0] = n[:, 1]  # the pole row is degenerate
        return _normalize(n)

    def camera(self, seconds):
        az = AZIMUTH + ORBIT * seconds / DURATION
        eye = DISTANCE * np.array(
            [
                np.cos(ELEVATION) * np.cos(az),
                np.cos(ELEVATION) * np.sin(az),
                np.sin(ELEVATION),
            ]
        )
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, forward, right, up

    def frame(self, seconds):
        dev = self.device
        time = evert_time(seconds)
        p = self._world(time)
        n = (self._normals(p) * self.orient).float().reshape(-1, 3)
        p = p.float().reshape(-1, 3)

        eye, forward, right, up = self.camera(seconds)
        basis = torch.tensor(
            np.stack((right, up, forward), 1), device=dev, dtype=torch.float32
        )
        cam = (p - torch.tensor(eye, device=dev, dtype=torch.float32)) @ basis
        focal = float(np.interp(time, self.zoom_clock, self.zoom)) * SS * DISTANCE
        sx = CENTRE[0] * SS + focal * cam[:, 0] / cam[:, 2]
        sy = CENTRE[1] * SS - focal * cam[:, 1] / cam[:, 2]
        depth = cam[:, 2]

        key = self._rasterize(sx, sy, depth)
        image = self._shade(key, sx, sy, n, forward, right, up)
        image = F.avg_pool2d(image.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)
        return (image.clamp(0, 1) ** (1 / 2.2) * 255.0).flip(-1)  # BGR

    def _rasterize(self, sx, sy, depth):
        """
        Nearest triangle per pixel, packed as (depth << 21) | triangle. Every
        pixel centre in a triangle's bounding box is a candidate, tested with
        edge functions written relative to the box corner (float32-safe).
        """
        dev, W, H = self.device, self.width, self.height
        tris = self.tris
        x, y, z = sx[tris], sy[tris], depth[tris]  # (T, 3)
        x0 = torch.ceil(x.min(1).values - 0.5).clamp(0, W)
        x1 = torch.floor(x.max(1).values - 0.5).clamp(-1, W - 1)
        y0 = torch.ceil(y.min(1).values - 0.5).clamp(0, H)
        y1 = torch.floor(y.max(1).values - 0.5).clamp(-1, H - 1)
        w = (x1 - x0 + 1).clamp_min(0).int()
        h = (y1 - y0 + 1).clamp_min(0).int()
        x, y = x - x0[:, None], y - y0[:, None]
        area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
            y[:, 1] - y[:, 0]
        )
        counts = torch.where(area.abs() > 1e-9, w * h, 0)
        inv = 1.0 / torch.where(area.abs() > 1e-9, area, 1.0)
        # b_i(px, py) = a_i px + b_i py + c_i, with i = 0 opposite vertex 0 etc.
        j, k = [1, 2, 0], [2, 0, 1]
        ea = (y[:, j] - y[:, k]) * inv[:, None]
        eb = (x[:, k] - x[:, j]) * inv[:, None]
        ec = (x[:, j] * y[:, k] - x[:, k] * y[:, j]) * inv[:, None]
        near, far = float(depth.min()), float(depth.max())
        scale = (2**30 - 2) / max(far - near, 1e-6)
        q = (far - z) * scale  # depth as key units, larger = nearer
        # b_2 = 1 - b_0 - b_1, so two edges are enough; then the depth plane.
        plane = [(e * q).sum(1, True) for e in (ea, eb, ec)]
        coef = torch.cat((ea[:, :2], eb[:, :2], ec[:, :2], *plane), 1)
        coef = coef.T.contiguous()  # (9, T): column gathers are far faster than rows
        base = (y0 * W + x0).int()

        ends = torch.cumsum(counts.long(), 0)
        total = int(ends[-1])
        starts = (ends - counts).int() if total < 2**31 else None
        keys = torch.zeros(H * W, dtype=torch.int64, device=dev)
        cuts = torch.searchsorted(
            ends, torch.arange(CHUNK, total + CHUNK, CHUNK, device=dev)
        )
        first = 0
        for cut in cuts.tolist():
            last = min(cut + 1, tris.shape[0])
            if last <= first:
                continue
            c = counts[first:last]
            n_cand = int(ends[last - 1] - (ends[first - 1] if first else 0))
            if n_cand:
                ids = torch.arange(first, last, device=dev, dtype=torch.int32)
                tid = torch.repeat_interleave(ids, c, output_size=n_cand)
                local = torch.arange(n_cand, device=dev, dtype=torch.int32)
                local = local - (starts[tid] - starts[first])
                tw = w[tid]
                ly = torch.div(local, tw, rounding_mode="floor")
                lx = local - ly * tw
                cf = torch.index_select(coef, 1, tid)
                px, py = lx.float() + 0.5, ly.float() + 0.5
                b = torch.addcmul(torch.addcmul(cf[4:6], cf[0:2], px), cf[2:4], py)
                inside = (b[0] >= -1e-5) & (b[1] >= -1e-5) & (b[0] + b[1] <= 1 + 1e-5)
                d = torch.addcmul(torch.addcmul(cf[8], cf[6], px), cf[7], py)
                d, tid = d[inside], tid[inside]
                pix = (base[tid] + ly[inside] * W + lx[inside]).long()
                key = (d.long().clamp(0, 2**30 - 2) + 1 << 21) | tid.long()
                keys.scatter_reduce_(0, pix, key, reduce="amax")
                del tid, local, tw, ly, lx, cf, px, py, b, inside, d, pix, key
            first = last
        return keys.view(H, W)

    def _shade(self, key, sx, sy, n, forward, right, up):
        dev, W, H = self.device, self.width, self.height
        image = (self.background * (1 + 1.6 * self.glow)).expand(H, W, 3).clone()
        flat = key.view(-1)
        covered = torch.nonzero(flat > 0).squeeze(1)
        if covered.numel() == 0:
            return image
        tid = flat[covered] & ((1 << 21) - 1)
        tri = self.tris[tid]
        px = (covered % W).float() + 0.5
        py = torch.div(covered, W, rounding_mode="floor").float() + 0.5
        xt, yt = sx[tri], sy[tri]
        area = (xt[:, 1] - xt[:, 0]) * (yt[:, 2] - yt[:, 0]) - (xt[:, 2] - xt[:, 0]) * (
            yt[:, 1] - yt[:, 0]
        )
        b0 = (
            (xt[:, 1] - px) * (yt[:, 2] - py) - (xt[:, 2] - px) * (yt[:, 1] - py)
        ) / area
        b1 = (
            (xt[:, 2] - px) * (yt[:, 0] - py) - (xt[:, 0] - px) * (yt[:, 2] - py)
        ) / area
        bary = torch.stack((b0, b1, 1 - b0 - b1), 1).clamp(0, 1)
        bary = bary / bary.sum(1, keepdim=True)
        normal = _normalize((n[tri] * bary[..., None]).sum(1))
        uv = (self.uv[tri] * bary[..., None]).sum(1)

        to = dict(device=dev, dtype=torch.float32)
        view = -torch.tensor(forward, **to)
        facing = (normal @ view) >= 0
        normal = torch.where(facing[:, None], normal, -normal)

        # Lights fixed to the camera, so the shading reads the same all the way round.
        cam = torch.tensor(np.stack((right, up, -forward), 0), **to)
        light = _normalize(torch.tensor(LIGHT, **to) @ cam)
        fill = _normalize(torch.tensor(FILL, **to) @ cam)
        half = _normalize(light + view)
        diffuse = (normal @ light).clamp_min(0)
        fill_d = (normal @ fill).clamp_min(0)
        spec = (normal @ half).clamp_min(0) ** SHINE
        rim = (1 - (normal @ view).abs()) ** 3

        def paint(ramp):
            shadow, body, bright = ramp
            base = shadow + (body - shadow) * diffuse[:, None]
            base = base + (bright - body) * (diffuse**3)[:, None]
            return base + 0.25 * body * fill_d[:, None]

        colour = torch.where(facing[:, None], paint(self.outside), paint(self.inside))
        rim_colour = torch.where(facing[:, None], self.outside[2], self.inside[2])

        # Faint grid on the surface: latitude rings and strip edges.
        def line(x, count):
            d = (x * count) % 1.0
            d = torch.minimum(d, 1 - d) / count
            return torch.exp(-((d / 0.006) ** 2))

        grid = torch.maximum(line(uv[:, 0], LINES_U), line(uv[:, 1], LINES_V))
        colour = colour * (1 - LINE_DARK * grid[:, None])
        colour = colour + 0.55 * spec[:, None] + 0.35 * rim[:, None] * rim_colour
        image.view(-1, 3)[covered] = colour
        return image


# ------------------------------------------------------------------- the sound
def pole_crossing() -> float:
    """Second at which the north pole passes the equatorial plane (the push)."""
    u = torch.tensor([1e-6], dtype=torch.float64)
    v = torch.tensor([0.0], dtype=torch.float64)
    seconds = np.linspace(STAGES[1][2], STAGES[2][2], 400)
    z = np.array([float(patch(u, v, evert_time(s))[0, 0, 2]) for s in seconds])
    flips = np.nonzero(np.sign(z[1:]) != np.sign(z[:-1]))[0]
    return float(seconds[flips[0] + 1]) if len(flips) else STAGES[2][2]


def soundtrack(crossing: float, rng: np.random.Generator) -> np.ndarray:
    """
    Rise & resolve. A bell for each of the N_STRIPS waves as they form (and
    again, falling, as they smooth away), panned by where the strip faces; a
    low string drone; a minor chord with a riser up to `crossing`, where the
    poles pass and a deep bell lands; a bell for every 1/16 turn of the twist
    over a moving chord; a second riser through the unpush; and the home chord,
    turned major (strings and choir), when the sphere is round again.
    """
    root = int(rng.choice(ROOTS))
    minor, major = Scale(root, "minor"), Scale(root, "major")
    hz = lambda note: float(audio.midi_to_hz(note))
    track = audio.silence(DURATION)
    at = {name: start for name, _, start in STAGES}
    kw = dict(circular=False)

    def chord(notes, start, length, velocity=0.22, brightness=0.35):
        for note in notes:
            sound = audio.strings(hz(note), length, velocity, 1.2, 1.2, brightness)
            audio.add(track, sound, start, **kw)

    def strip_pan(k, seconds):
        az = AZIMUTH + ORBIT * seconds / DURATION
        return 0.7 * np.sin(TAU * (k + 0.5) / N_STRIPS - az)

    # The waves form: one bell per strip, climbing.
    a, b = at["corrugate"], at["push"] - 0.35
    for k in range(N_STRIPS):
        when = a + (b - a) * k / N_STRIPS
        bell = audio.bell(hz(minor.note(k, BELL_OCTAVE)), 0.3 + 0.03 * k)
        audio.add(track, audio.pan(bell, strip_pan(k, when)), when, **kw)
    drone = audio.strings(hz(minor.note(0, -1)), END - 0.2, 0.3, 2.0, 1.0, 0.25)
    audio.add(track, drone, 0.2, **kw)

    # The push: tension up to the moment the poles pass.
    chord(minor.chord(0, 3), at["push"] - 0.4, at["twist"] - at["push"])
    rise = audio.riser(crossing - at["push"], rng, 140.0, 700.0, 0.22)
    audio.add(track, audio.pan(rise, 0.0), at["push"], **kw)
    audio.add(track, audio.bell(hz(minor.note(0, 0)), 0.8, 5.0), crossing, **kw)
    audio.add(track, audio.kick(0.5, 44.0), crossing, **kw)

    # The twist: half a turn, a bell every sixteenth of a turn, falling.
    a, b = at["twist"], at["unpush"]
    for i, degree in enumerate((5, 3, 4)):  # VI, iv, v under the twist
        third = (b - a) / 3
        chord(minor.chord(degree, 3, -1 if degree > 3 else 0), a + i * third, third)
    for k in range(N_STRIPS):
        when = a + (b - a) * (k + 0.5) / N_STRIPS
        bell = audio.bell(hz(minor.note(7 - k, BELL_OCTAVE)), 0.32)
        audio.add(track, audio.pan(bell, 0.5 * (-1) ** k), when, **kw)

    # The unpush: the long rise to the end.
    chord(minor.chord(4, 4, -1), at["unpush"], at["uncorrugate"] - at["unpush"], 0.2)
    rise = audio.riser(END - at["unpush"] - 0.05, rng, 200.0, 1600.0, 0.24)
    audio.add(track, audio.pan(rise, 0.0), at["unpush"], **kw)

    # The waves smooth away: the bells again, falling, in major now.
    a, b = at["uncorrugate"], END - 0.15
    for k in range(N_STRIPS):
        when = a + (b - a) * k / N_STRIPS
        bell = audio.bell(hz(major.note(7 - k, BELL_OCTAVE)), 0.34)
        audio.add(track, audio.pan(bell, strip_pan(k, when)), when, **kw)

    # Round again, gold side out: the home chord, major.
    hold = DURATION - END
    for note in major.chord(0, 3) + [major.note(0, -1)]:
        audio.add(track, audio.strings(hz(note), hold, 0.3, 0.15, 1.0, 0.55), END, **kw)
    for note in major.chord(0, 3, 1):
        audio.add(track, audio.choir(hz(note), hold, 0.22, 0.4, 1.0, "o"), END, **kw)
    audio.add(track, audio.bell(hz(major.note(0, BELL_OCTAVE)), 0.6, 4.0), END, **kw)
    return audio.master(
        audio.reverb(track, rng, *ROOM[:2], tone=ROOM[2], circular=False), loop=False
    )


def generate(settings: ImageProcessingSettings = None):
    """
    Smale's paradox, Thurston's corrugated sphere eversion.

    The geometry is a port of Nathaniel Thurston's `evert` (the program behind
    the film "Outside In", Geometry Center 1995): a patch of the sphere (u: pole
    to equator, v: 1/N_STRIPS of the way round) rides a base surface that is
    pushed (Stage1 -> Stage2), twisted half a turn (Scene23) and unpushed
    (Stage3 -> Stage4, the antipodal sphere), carrying a figure-eight
    corrugation along the normal (AddFigureEight) that grows in "corrugate" and
    dies in "uncorrugate". evert's jets (value + derivatives) become plain
    tensors plus torch.func.jvp for the derivatives in u and v; 2 x N_STRIPS
    rigid copies make the whole sphere. Every frame is a closed-form function
    of time: `evert_time` maps the video seconds onto evert's stage clock, easing
    through each stage boundary.

    Rendering: the copies' NU x NV grids are triangulated and rasterized on the
    GPU (bounding-box candidates, edge functions, an int64 z-buffer of
    (depth << 21 | triangle) by scatter_reduce amax) at SS x resolution, shaded
    per pixel with interpolated normals: which side faces the camera picks the
    ramp (outside cobalt, inside amber), plus camera-fixed key/fill lights, a
    Blinn highlight, a rim and a faint u/v grid. The camera orbits ORBIT.
    The picture uses no randomness; the seed only picks the soundtrack's key
    and its riser noise (`soundtrack`).
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    renderer = Renderer(device)

    def render(frame: int):
        with torch.no_grad():
            return renderer.frame(frame / FPS)

    render_frames(settings, render, FRAME_COUNT, parallel=False)
    del renderer
    gc.collect()
    track = soundtrack(pole_crossing(), settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)


if __name__ == "__main__":
    generate()
