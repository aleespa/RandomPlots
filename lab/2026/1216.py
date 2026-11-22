"""
Cut a Möbius Strip in Half

What happens if you cut a Möbius strip down the middle? You might expect two
strips. You get one: a single loop, twice as long, as the light running along
it shows by needing two laps to get round.

A Möbius strip has only one side and only one edge, so the cut never really
divides it. Cut it a third of the way in instead and the scissors go round
twice before they meet their starting point. Now there are two pieces: a thin
Möbius strip from the middle (gold), and a loop twice as long (violet), and the
two are linked together, impossible to pull apart.

Every cut plays a glassy bell as it travels; when the light runs along the new
loops, the bells climb with it.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#mobiusstrip #topology #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from numba import njit

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
TAU = 2.0 * np.pi

# The strip: M(u, v) = ((R + W v cos u/2) cos u, (R + W v cos u/2) sin u, W v sin u/2).
R, W = 1.0, 0.42
NU = 720  # samples round the strip (u in [0, 2 pi])
ROWS = np.array([0.0, 0.035, 0.1, 0.25, 0.5, 0.75, 0.9, 0.965, 1.0])  # across a piece
NF = len(ROWS)
THIRD = 1.0 / 3.0
RAMP = 0.55  # radians behind the spark over which a cut opens
GAP_CUT, GAP_OPEN = 0.045, 0.11  # half-gap of a cut (in v) while cutting / once open

# Timeline (seconds).
CUT1 = (0.15, 3.0)
OPEN1 = (3.0, 3.8)
PAINT1 = (3.4, 6.3)
SWAP = (6.7, 7.2)  # cross-fade to a fresh strip
CUT2 = (7.4, 11.3)
OPEN2 = (11.3, 12.0)
PAINT2 = (11.6, 14.0)
FADE = 0.35

# Camera and frame.
DISTANCE = 6.0
ELEVATION = np.radians(38.0)
AZIMUTH0 = np.radians(-60.0)
ORBIT = np.radians(110.0)  # over the whole video
SPIN = 0.18  # extra slow roll of the strip about its axis, turns per video
CENTRE = (WIDTH / 2, 940.0)
SCALE_PX = 515.0  # screen radius of R + W at the target distance
SS = 2
CROP = (340, 1540)  # rows of the frame the strip can reach
TITLE_Y, LABEL_Y, NOTE_Y = 300, 1470, 1520

# Palette (hex, converted to BGR below).
BACKGROUND = ("#04060d", "#0c1426")
STRIP = "#2c7f9e"
EDGE = "#9fe8f5"
SPARK = "#fff1c9"
GOLD = "#f2b049"
VIOLET = "#a66bf0"
TEXT = "#eef3f8"
TEXT_DIM = "#9fb2c8"
LIGHT = np.array([-0.45, 0.55, 0.70])  # world space, from above and to the side


def _bgr(colour: str) -> np.ndarray:
    return np.array(hex_to_bgr(colour), np.float32)


def _ease(x):
    """Half linear, half smoothstep: starts moving at once, still lands gently."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * x + 0.5 * smoothstep(x)


def _ramp(sec, span):
    return float(smoothstep((sec - span[0]) / (span[1] - span[0])))


def mobius(u, v):
    """Points of the strip; u and v broadcast. M(u + 2 pi, v) = M(u, -v)."""
    r = R + W * v * np.cos(u / 2)
    return np.stack((r * np.cos(u), r * np.sin(u), W * v * np.sin(u / 2)), -1)


@njit(nogil=True, cache=True, fastmath=True)
def _raster(sx, sy, sz, col, tris, image, zbuf):
    """z-buffered triangles, colours interpolated (Gouraud) at pixel centres."""
    h, w = zbuf.shape
    for t in range(tris.shape[0]):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        x0, y0, x1, y1, x2, y2 = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area
        xa = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        xb = min(int(np.ceil(max(x0, max(x1, x2)))), w - 1)
        ya = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        yb = min(int(np.ceil(max(y0, max(y1, y2)))), h - 1)
        for py in range(ya, yb + 1):
            fy = py + 0.5
            for px in range(xa, xb + 1):
                fx = px + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * inv
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                z = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
                if z < zbuf[py, px]:
                    zbuf[py, px] = z
                    for k in range(3):
                        image[py, px, k] = (
                            w0 * col[a, k] + w1 * col[b, k] + w2 * col[c, k]
                        )


def _grid_tris(pieces: int) -> np.ndarray:
    idx = np.arange((NU + 1) * NF).reshape(NU + 1, NF)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    one = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    offset = np.arange(pieces)[:, None, None] * (NU + 1) * NF
    return (one[None] + offset).reshape(-1, 3).astype(np.int32)


def _comet(d):
    """Brightness of a travelling light at distance d behind its head (d < 0: ahead)."""
    ahead = np.exp(-((np.minimum(d, 0.0) / 0.12) ** 2))
    behind = np.exp(-np.maximum(d, 0.0) / 0.8)
    return np.where(d < 0, ahead, behind)


class Scene:
    """Which pieces exist at a given second, their colours, and the spark."""

    def __init__(self):
        self.u = np.linspace(0.0, TAU, NU + 1)
        self.strip, self.edge = _bgr(STRIP), _bgr(EDGE)
        self.gold, self.violet = _bgr(GOLD), _bgr(VIOLET)

    def cut_progress(self, sec, act):
        span, total = (CUT1, TAU) if act == 1 else (CUT2, 2 * TAU)
        x = (sec - span[0]) / (span[1] - span[0])
        return (total + RAMP) * float(_ease(x)), total

    def pieces(self, sec, act):
        """
        A list of (lo, hi, lo_edge, hi_edge, base, glow): v-bounds per u, edge
        strengths per u, a base colour and an added glow per u (NU + 1, 3).
        """
        u = self.u
        p, total = self.cut_progress(sec, act)
        if act == 1:
            gap = GAP_CUT + (GAP_OPEN - GAP_CUT) * _ramp(sec, OPEN1)
            g = gap * smoothstep((p - u) / RAMP)
            cut = smoothstep(g / GAP_CUT)
            head = (2 * TAU + 0.35) * float(
                _ease((sec - PAINT1[0]) / (PAINT1[1] - PAINT1[0]))
            )
            on = 1.0 if sec > PAINT1[0] else 0.0
            fade = 1.0 - _ramp(sec, (PAINT1[1] - 0.2, PAINT1[1] + 0.5))
            out = []
            for s, lo, hi, le, he in (
                (u, g, np.ones_like(u), cut, 0.5 + 0 * u),
                (u + TAU, -np.ones_like(u), -g, 0.5 + 0 * u, cut),
            ):
                d = head - s
                paint = on * smoothstep(d / 0.3)
                base = self.strip + paint[:, None] * (self.gold - self.strip)
                glow = (on * fade * _comet(d))[:, None] * (_bgr(SPARK) * 0.75)
                out.append((lo, hi, le, he, base, glow))
            return out
        gap = GAP_CUT + (GAP_OPEN - GAP_CUT) * _ramp(sec, OPEN2)
        gp = gap * smoothstep((p - u) / RAMP)
        gm = gap * smoothstep((p - TAU - u) / RAMP)
        cp, cm = smoothstep(gp / GAP_CUT), smoothstep(gm / GAP_CUT)
        x = float(_ease((sec - PAINT2[0]) / (PAINT2[1] - PAINT2[0])))
        on = 1.0 if sec > PAINT2[0] else 0.0
        fade = 1.0 - _ramp(sec, (PAINT2[1] - 0.2, PAINT2[1] + 0.5))
        band_head, core_head = (2 * TAU + 0.35) * x, (TAU + 0.35) * x
        one = np.ones_like(u)
        out = []
        for s, head, colour, lo, hi, le, he in (
            (u, band_head, self.violet, THIRD + gp, one, cp, 0.5 * one),
            (u, core_head, self.gold, -THIRD + gm, THIRD - gp, cm, cp),
            (u + TAU, band_head, self.violet, -one, -THIRD - gm, 0.5 * one, cm),
        ):
            d = head - s
            paint = on * smoothstep(d / 0.3)
            base = self.strip + paint[:, None] * (colour - self.strip)
            glow = (on * fade * _comet(d))[:, None] * (_bgr(SPARK) * 0.75)
            out.append((lo, hi, le, he, base, glow))
        return out

    def spark(self, sec, act):
        """World position and brightness of the cutting point, or None."""
        p, total = self.cut_progress(sec, act)
        if p <= 0 or p >= total:
            return None
        if act == 1:
            u, v = p, 0.0
        else:
            u, v = (p, THIRD) if p < TAU else (p - TAU, -THIRD)
        fade = min(1.0, p / 0.3, (total - p) / 0.3)
        return mobius(np.array(u), np.array(v)), fade


class Renderer:
    def __init__(self):
        self.scene = Scene()
        self.tris = {2: _grid_tris(2), 3: _grid_tris(3)}
        self.light = LIGHT / np.linalg.norm(LIGHT)
        self.focal = SCALE_PX * DISTANCE / (R + W)

        # Background: dark navy, a soft glow behind the strip.
        ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        r2 = ((xs - CENTRE[0]) / WIDTH) ** 2 + ((ys - CENTRE[1]) / WIDTH) ** 2
        glow = np.exp(-r2 / 0.18)[..., None]
        lo, hi = _bgr(BACKGROUND[0]), _bgr(BACKGROUND[1])
        self.background = (lo + (hi - lo) * glow).astype(np.float32)
        crop = self.background[CROP[0] : CROP[1]]
        self.crop_ss = cv2.resize(
            crop, (WIDTH * SS, (CROP[1] - CROP[0]) * SS), interpolation=cv2.INTER_LINEAR
        )
        k = np.arange(-60, 61, dtype=np.float32)
        self.sprite = (
            0.8 * np.exp(-(k[:, None] ** 2 + k[None] ** 2) / (2 * 14.0**2))
            + 1.2 * np.exp(-(k[:, None] ** 2 + k[None] ** 2) / (2 * 3.5**2))
        )[..., None]

        size = 58
        self.texts = {
            "title1": latex_mask(
                r"$\mathrm{Cut\ a\ M\ddot{o}bius\ strip\ down\ the\ middle}$", size
            ),
            "label1": latex_mask(r"$\mathrm{one\ loop{,}\ twice\ as\ long}$", 58),
            "title2": latex_mask(
                r"$\mathrm{Now\ cut\ it\ a\ third\ of\ the\ way\ in}$", size
            ),
            "note2": latex_mask(r"$\mathrm{the\ scissors\ go\ round\ twice}$", 40),
            "label2": latex_mask(r"$\mathrm{two\ loops{,}\ linked\ together}$", 58),
        }

        # Compile the rasterizer once here, not in every worker at once.
        image = np.zeros((4, 4, 3), np.float32)
        zbuf = np.full((4, 4), np.inf, np.float32)
        tri = np.array([[0, 1, 2]], np.int32)
        pts = np.array([0.0, 3.0, 0.0], np.float32)
        _raster(
            pts, pts[::-1].copy(), pts, np.ones((3, 3), np.float32), tri, image, zbuf
        )

    @staticmethod
    def camera(sec):
        az = AZIMUTH0 + ORBIT * sec / DURATION
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

    def project(self, p, sec):
        eye, forward, right, up = self.camera(sec)
        q = p - eye
        zc = q @ forward
        sx = (CENTRE[0] + self.focal * (q @ right) / zc) * SS
        sy = (CENTRE[1] - CROP[0] - self.focal * (q @ up) / zc) * SS
        return sx, sy, zc, eye

    @staticmethod
    def roll(p, sec):
        """The strip turns slowly in its own plane (about z)."""
        a = TAU * SPIN * sec / DURATION
        c, s = np.cos(a), np.sin(a)
        x, y = p[..., 0], p[..., 1]
        return np.stack((c * x - s * y, s * x + c * y, p[..., 2]), -1)

    def strip_image(self, sec, act):
        """The strip alone, on the background crop, plus the frame's z-buffer."""
        pieces = self.scene.pieces(sec, act)
        u = self.scene.u
        pts, cols = [], []
        for lo, hi, le, he, base, glow in pieces:
            v = lo[:, None] + (hi - lo)[:, None] * ROWS[None, :]
            p = self.roll(mobius(u[:, None], v), sec)  # (NU + 1, NF, 3)
            pu = np.gradient(p, axis=0)
            pv = np.gradient(p, axis=1)
            n = np.cross(pu, pv)
            n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
            edge = np.zeros((NU + 1, NF))
            edge[:, 0], edge[:, -1] = le, he
            pts.append(p)
            cols.append((n, base, glow, edge))
        eye = self.camera(sec)[0]
        image = self.crop_ss.copy()
        zbuf = np.full(image.shape[:2], np.inf, np.float32)
        allp = np.concatenate([p.reshape(-1, 3) for p in pts])
        colours = []
        for p, (n, base, glow, edge) in zip(pts, cols):
            view = eye - p
            view /= np.linalg.norm(view, axis=-1, keepdims=True)
            facing = np.sum(n * view, -1, keepdims=True)
            n = np.where(facing < 0, -n, n)
            diffuse = np.clip(n @ self.light, 0, None)[..., None]
            half = self.light + view
            half /= np.linalg.norm(half, axis=-1, keepdims=True)
            spec = np.clip(np.sum(n * half, -1), 0, None)[..., None] ** 40
            rim = (1 - np.abs(np.sum(n * view, -1)))[..., None] ** 3
            body = base[:, None, :]
            c = body * (0.2 + 0.75 * diffuse) + 0.4 * rim * body + 85.0 * spec
            c += glow[:, None, :]
            edge_col = 0.5 * body + 0.6 * self.scene.edge
            c += edge[..., None] * edge_col
            colours.append(c.reshape(-1, 3))
        sx, sy, sz, _ = self.project(allp, sec)
        _raster(
            sx.astype(np.float32),
            sy.astype(np.float32),
            sz.astype(np.float32),
            np.concatenate(colours).astype(np.float32),
            self.tris[len(pieces)],
            image,
            zbuf,
        )
        small = cv2.resize(
            image, (WIDTH, CROP[1] - CROP[0]), interpolation=cv2.INTER_AREA
        )
        return small, zbuf

    def add_spark(self, frame, zbuf, sec, act):
        spark = self.scene.spark(sec, act)
        if spark is None:
            return
        p, fade = spark
        p = self.roll(p, sec)
        sx, sy, sz, _ = self.project(p[None], sec)
        x, y = int(round(sx[0])), int(round(sy[0]))
        if not (0 <= x < zbuf.shape[1] and 0 <= y < zbuf.shape[0]):
            return
        hidden = zbuf[y, x] < sz[0] - 0.02
        gain = fade * (0.3 if hidden else 1.0)
        cx, cy = x // SS, y // SS + CROP[0]
        h = self.sprite.shape[0] // 2
        y0, y1 = max(cy - h, 0), min(cy + h + 1, HEIGHT)
        x0, x1 = max(cx - h, 0), min(cx + h + 1, WIDTH)
        patch = self.sprite[y0 - cy + h : y1 - cy + h, x0 - cx + h : x1 - cx + h]
        frame[y0:y1, x0:x1] += gain * 255.0 * patch * (_bgr(SPARK) / 255.0)

    def scene_frame(self, sec, act):
        frame = self.background.copy()
        strip, zbuf = self.strip_image(sec, act)
        frame[CROP[0] : CROP[1]] = strip
        self.add_spark(frame, zbuf, sec, act)
        return frame

    def text(self, frame, sec):
        def alpha(start, end, fade=0.3):
            return float(
                smoothstep((sec - start) / fade) * smoothstep((end - sec) / fade)
            )

        t, d = _bgr(TEXT), _bgr(TEXT_DIM)
        blits = (
            ("title1", TITLE_Y, t, alpha(-1.0, SWAP[0] + 0.25)),
            (
                "label1",
                LABEL_Y,
                _bgr(GOLD) * 0.4 + t * 0.6,
                alpha(PAINT1[0] + 0.4, SWAP[0] + 0.25),
            ),
            ("title2", TITLE_Y, t, alpha(SWAP[1] - 0.25, DURATION + 1)),
            ("note2", NOTE_Y - 40, d, alpha(CUT2[0] + 1.2, CUT2[1] + 0.2)),
            ("label2", LABEL_Y, t, alpha(PAINT2[0] + 0.3, DURATION + 1)),
        )
        for key, y, colour, a in blits:
            if a > 0:
                blit_mask(frame, self.texts[key], WIDTH // 2, y, colour, a, "centre")

    def frame(self, sec):
        if sec < SWAP[0]:
            frame = self.scene_frame(sec, 1)
        elif sec > SWAP[1]:
            frame = self.scene_frame(sec, 2)
        else:
            w = float(smoothstep((sec - SWAP[0]) / (SWAP[1] - SWAP[0])))
            frame = (1 - w) * self.scene_frame(sec, 1) + w * self.scene_frame(sec, 2)

        # Bloom on everything brighter than the background.
        lit = np.clip(frame - self.background, 0, None)
        small = cv2.resize(lit, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 7.0)
        frame += 0.45 * cv2.resize(small, (WIDTH, HEIGHT))

        self.text(frame, sec)
        out = float(smoothstep((DURATION - sec) / FADE))
        if out < 1:
            frame = self.background + out * (frame - self.background)
        return frame


# ------------------------------------------------------------------- the sound
STYLE = "music_box"  # bells in lydian over a quiet choir


def _when(f, span, levels):
    """First second in `span` at which f(sec) reaches each level (from the closed forms)."""
    secs = np.linspace(span[0] - 0.2, span[1] + 0.6, 6000)
    values = np.array([f(s) for s in secs])
    return [float(secs[np.argmax(values >= max(level, 1e-3))]) for level in levels]


def _pan(u, v, sec):
    """Stereo position of the strip point M(u, v): where it is on screen."""
    p = Renderer.roll(mobius(np.array(u), np.array(v)), sec)
    eye, forward, right, _ = Renderer.camera(sec)
    q = p - eye
    x = (q @ right) / (q @ forward) * SCALE_PX * DISTANCE / (R + W)
    return float(np.clip(0.75 * x / (WIDTH / 2), -0.8, 0.8))


def soundtrack(rng: np.random.Generator) -> np.ndarray:
    """
    Music box: one glassy bell per eighth of a turn of each cut, panned to where
    the spark is on screen. The midline cut climbs a scale once and lands on a
    fifth when it meets itself; the light then climbs nine notes over its two
    laps. The third-cut repeats its eight-note figure an octave higher on its
    second lap (the scissors go round twice), and when the two new loops light
    up, the long band climbs low while the Möbius core climbs high, ending on
    the home chord. Underneath, a quiet choir holds I, then vi, II, I.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    hz = lambda note: float(audio.midi_to_hz(note))
    scene = Scene()
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    def ring(degree, octave, velocity, when, pan, length=3.0):
        bell = style.voice(scale.hz(degree, octave), velocity, length)
        audio.add(track, audio.pan(bell, pan), when, **kw)

    def head_pan(s, sec):
        return _pan(s % TAU, 0.6 if s % (2 * TAU) < TAU else -0.6, sec)

    # Act 1: the midline cut, an eighth of a turn per bell, then the meeting.
    cut = lambda sec: scene.cut_progress(sec, 1)[0]
    for k, when in enumerate(_when(cut, CUT1, TAU * np.arange(8) / 8)):
        ring(k, 0, 0.32 + 0.02 * k, when, _pan(k * TAU / 8, 0.0, when))
    (meet,) = _when(cut, CUT1, [TAU])
    ring(0, 1, 0.42, meet, 0.0, 4.0)
    ring(4, 0, 0.36, meet, 0.0, 4.0)

    # The light runs two laps of the one long band: nine notes, climbing.
    span = PAINT1
    head = lambda sec: (2 * TAU + 0.35) * float(
        _ease((sec - span[0]) / (span[1] - span[0]))
    )
    for k, when in enumerate(_when(head, span, np.pi / 2 * np.arange(9))):
        ring(k, -1 if k < 4 else 0, 0.3, when, head_pan(k * np.pi / 2, when))

    # Act 2: the third-cut goes round twice: the same figure, an octave up.
    figure = (0, 2, 4, 6, 7, 6, 4, 2)
    cut = lambda sec: scene.cut_progress(sec, 2)[0]
    for k, when in enumerate(_when(cut, CUT2, TAU * np.arange(16) / 8)):
        v = THIRD if k < 8 else -THIRD
        ring(figure[k % 8], k // 8 - 1, 0.3, when, _pan(k * TAU / 8, v, when))
    (meet,) = _when(cut, CUT2, [2 * TAU])
    ring(5, 0, 0.38, meet, 0.0, 4.0)
    ring(2, 0, 0.32, meet, 0.0, 4.0)

    # Two loops light up: the long band low, the Möbius core high.
    span = PAINT2
    x = lambda sec: float(_ease((sec - span[0]) / (span[1] - span[0])))
    band = lambda sec: (2 * TAU + 0.35) * x(sec)
    core = lambda sec: (TAU + 0.35) * x(sec)
    for k, when in enumerate(_when(band, span, np.pi / 2 * np.arange(9))):
        ring(k, -1, 0.3, when, head_pan(k * np.pi / 2, when))
    for k, when in enumerate(_when(core, span, np.pi / 2 * np.arange(1, 5))):
        ring(4 + k, 1, 0.26, when, _pan(k * np.pi / 2, 0.0, when))

    # The home chord as the painting finishes.
    end = span[1]
    for degree, octave in ((0, 0), (4, 0), (0, 1)):
        ring(degree, octave, 0.4, end, 0.0, 5.0)

    # The bed: a soft choir, I -> vi -> II -> I, and a low string root.
    for degree, start, stop in (
        (0, 0.0, SWAP[1]),
        (5, SWAP[0], CUT2[1]),
        (1, CUT2[1] - 0.3, end),
        (0, end - 0.3, DURATION),
    ):
        for note in scale.chord(degree, 3, -2):
            voice = audio.choir(hz(note), stop - start, 0.12, 0.8, 1.2, "u")
            audio.add(track, voice, start, **kw)
        low = audio.strings(
            hz(scale.note(degree, -2)), stop - start, 0.14, 0.8, 1.2, 0.3
        )
        audio.add(track, low, start, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Cutting a Möbius strip along its midline, then a third of the way in.

    The strip is M(u, v) = ((R + W v cos u/2) cos u, (R + W v cos u/2) sin u,
    W v sin u/2) for u in [0, 2 pi], v in [-1, 1]; M(u + 2 pi, v) = M(u, -v) is
    the half twist. A cut along the curve v = c is drawn as a gap of half-width
    g(u) = gap * smoothstep((p - u) / RAMP) behind the cut point p(t), so the
    strip is split into pieces with v-bounds that depend on u, each an
    (NU + 1) x NF grid.

    Act 1 (midline, v = 0): p runs 0 -> 2 pi. The two pieces v > 0 and v < 0
    are one band, parametrised by s in [0, 4 pi) (s = u on the upper piece,
    s = u + 2 pi on the lower), and a light with a comet tail runs along s from
    0 to 4 pi, painting it gold: two laps round.

    Act 2 (v = 1/3, which continues as v = -1/3 after one lap): p runs 0 -> 4 pi,
    and the cut opens at v = +1/3 when p > u and at v = -1/3 when p > u + 2 pi.
    The pieces are the core |v| < 1/3 (a Möbius strip, painted gold, s = u) and
    the outer band |v| > 1/3 (one band of length 4 pi, painted violet), linked.

    Every frame is a closed-form function of the time: the grids are lit per
    vertex (two-sided Lambert, Blinn highlight, rim, bright edges for the
    strip's single edge and the cut edges), projected through an orbiting
    perspective camera, rasterized with a z-buffer by a Numba kernel at 2x and
    downsampled; a spark sprite marks the cut point, a quarter-res bloom and
    LaTeX labels go on top. The picture uses no randomness; the seed only picks
    the soundtrack's key and room (`soundtrack`).
    """
    settings = settings or ImageProcessingSettings(1)
    renderer = Renderer()

    def render(frame):
        return renderer.frame(frame / (FRAME_COUNT - 1) * DURATION)

    render_frames(settings, render, FRAME_COUNT)
    del renderer
    gc.collect()
    track = soundtrack(settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)


if __name__ == "__main__":
    generate()
