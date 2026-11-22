"""
The Road to Chaos

The logistic map, x → r·x(1 − x), is one line of algebra, and this bifurcation
diagram shows where it leads. Each column is a value of r; the dots are where x
ends up if you apply the rule again and again. At first it settles on one value.
Turn r up and it flips between two, then four, then eight, until at r ≈ 3.57 it
never repeats: chaos.

Deep inside the chaos, a window of order opens: a cycle of three. Zoom in and the
whole diagram is there again, in miniature, doubling its way back into chaos.

Below, the same rule is drawn as a staircase, and every step strikes a marimba
note, its height on screen: you hear the map settle, trill, and fall apart, twice.

Made with Python & PyTorch. Sound synthesized in NumPy.
#logisticmap #chaostheory #mathart #generativeart #randomplots
"""

import gc
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

from common import audio
from common.gpu import blur, gaussian_kernel, select_device
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale, beat
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'One line of algebra. Then chaos.'
HOOK_Y = 240
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- Timeline (s): 96 BPM, a bar is 2.5 s, one map step per sixteenth note -----------
BPM = 96.0
STEP = beat(BPM) / 4  # s per iterate of the melody orbit
SWEEP1 = (  # (time, r): the cursor crosses the whole diagram, slowing at the doublings
    (0.0, 2.86),
    (1.0, 3.0),
    (2.3, 3.449),
    (3.3, 3.5441),
    (4.0, 3.5644),
    (4.5, 3.5699),
    (5.5, 3.80),
    (6.3, 3.8335),
    (6.6, 3.8540),
    (7.5, 4.0),
)
GLIDE = (7.5, 8.3)  # the cursor flies back into the period-3 window
ZOOM = (7.6, 10.2)  # the camera dives into the window
SWEEP2 = (  # inside the window: 3 -> 6 -> 12 -> chaos -> crisis
    (10.2, 3.8350),
    (10.9, 3.8412),
    (11.8, 3.8472),
    (12.4, 3.8490),
    (13.1, 3.8530),
    (13.6, 3.8566),
    (13.8, 3.8582),
)
NOTES_END = 13.8  # s: the last map step that sounds; the image rests after it
FADE = 0.45  # s fade to black at the very end

# --- Views (r range, x range) ---------------------------------------------------------
MAIN_VIEW = (2.84, 4.01, -0.03, 1.03)
WINDOW_R = 3.8350  # where the cursor waits during the dive (stable 3-cycle)
ZOOM_VIEW = (3.8262, 3.8594, 0.4315, 0.5685)  # the middle branch of the window

# --- Layout (px) ----------------------------------------------------------------------
DIAG_Y, DIAG_H = 300, 940  # the diagram band, full width
READ_Y = 1300  # baseline of the r / period readout
WEB_X, WEB_Y, WEB_S = 300, 1345, 480  # cobweb panel: left, top, size
SS = 2  # supersampling

# --- Diagram sampling -----------------------------------------------------------------
SAMPLES = 48  # orbits per (2x) pixel column
TRANSIENT = 700  # iterations discarded
RECORDED = 112  # iterations plotted per orbit
DENSITY = 5.0  # counts per 2x pixel that reach ~63% brightness
GHOST = 0.22  # brightness of the not-yet-swept diagram

# --- Look -----------------------------------------------------------------------------
BG = "#07060b"
DENSITY_COLOURS = ["#07060b", "#2a1240", "#6a2a6e", "#c2457a", "#f59e6b", "#fff1d6"]
ACCENT = "#7fe7ff"  # ice cyan: the cursor, the orbit, the staircase
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

# --- Sound ----------------------------------------------------------------------------
STYLE = (
    "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
MODE = "dorian"  # the melody's scale (overrides the style's)
SPAN = 21  # scale degrees from the bottom of the view to the top (three octaves)
LOWEST = -9  # degree at the bottom of the view
DAMP = 0.45  # s a melody note rings before it is damped


PARABOLA = np.linspace(0.0, 1.0, 160)


def hex_to_rgb(colour):
    """'#rrggbb' -> [r, g, b] floats in 0..255."""
    return [float(int(colour[i : i + 2], 16)) for i in (1, 3, 5)]


def timeline():
    """Fine-grid curves of time: cursor r, zoom progress, and how far the diagram is lit."""
    tau = np.linspace(0.0, DURATION, 60001)
    s1 = PchipInterpolator(*np.array(SWEEP1).T)
    s2 = PchipInterpolator(*np.array(SWEEP2).T)
    r = np.empty_like(tau)
    a = tau <= SWEEP1[-1][0]
    r[a] = s1(tau[a])
    glide = smoothstep((tau - GLIDE[0]) / (GLIDE[1] - GLIDE[0]))
    b = (tau > SWEEP1[-1][0]) & (tau < SWEEP2[0][0])
    r[b] = SWEEP1[-1][1] + (WINDOW_R - SWEEP1[-1][1]) * glide[b]
    c = tau >= SWEEP2[0][0]
    r[c] = s2(np.minimum(tau[c], SWEEP2[-1][0]))
    lit = np.maximum.accumulate(r)
    zoom = smoothstep((tau - ZOOM[0]) / (ZOOM[1] - ZOOM[0]))
    return tau, r, lit, zoom


def view_at(z):
    """(r0, r1, x0, x1) at zoom progress z: geometric zoom about each axis's fixed point."""
    out = []
    for lo0, hi0, lo1, hi1 in (
        (MAIN_VIEW[0], MAIN_VIEW[1], ZOOM_VIEW[0], ZOOM_VIEW[1]),
        (MAIN_VIEW[2], MAIN_VIEW[3], ZOOM_VIEW[2], ZOOM_VIEW[3]),
    ):
        w0, w1 = hi0 - lo0, hi1 - lo1
        c0, c1 = 0.5 * (lo0 + hi0), 0.5 * (lo1 + hi1)
        p = (c1 * w0 - c0 * w1) / (w0 - w1)  # stays put on screen
        w = w0 ** (1 - z) * w1**z
        c = p + (c0 - p) * w / w0
        out += [c - w / 2, c + w / 2]
    return tuple(out)


def melody_orbit(curve_tau, curve_r):
    """One orbit x_{k+1} = r(t_k) x_k (1 - x_k), a step per sixteenth: (times, r, x)."""
    times = np.arange(0.0, NOTES_END + 1e-9, STEP)
    rs = np.interp(times, curve_tau, curve_r)
    x = 0.5
    for _ in range(300):  # settle on the attractor of the first r
        x = rs[0] * x * (1 - x)
    xs = np.empty(len(times) + 1)
    xs[0] = x
    for k, r in enumerate(rs):
        xs[k + 1] = r * xs[k] * (1 - xs[k])
    return times, rs, xs


def attractor(rs, count=64, transient=6000):
    """Points of the attractor at every r in `rs` (float64): (len(rs), count)."""
    x = np.full(len(rs), 0.3141)
    for _ in range(transient):
        x = rs * x * (1 - x)
    pts = np.empty((len(rs), count))
    for i in range(count):
        x = rs * x * (1 - x)
        pts[:, i] = x
    return pts


def period_of(points, tol=2e-4):
    """Smallest p <= 32 with x_{i+p} = x_i along each row, else 0 (chaos)."""
    out = np.zeros(len(points), int)
    for p in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        same = np.abs(points[:, p:] - points[:, :-p]).max(axis=1) < tol
        out[(out == 0) & same] = p
    return out


class Diagram:
    """The bifurcation diagram of any (r, x) window, as a density image, on the GPU."""

    def __init__(self, rng, device):
        self.device = device
        self.w2, self.h2 = WIDTH * SS, DIAG_H * SS
        jitter = rng.random((SAMPLES, self.w2))
        self.u = torch.tensor(
            (np.arange(SAMPLES)[:, None] + jitter) / SAMPLES, device=device
        ).double()
        self.cols = torch.arange(self.w2, device=device).double()[None, :]
        self.x0 = torch.tensor(
            rng.uniform(0.05, 0.95, (SAMPLES, self.w2)), device=device
        ).double()
        self.col_index = torch.arange(self.w2, device=device)[None, :].expand(
            SAMPLES, -1
        )
        self.soften = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=device
            ).view(1, 1, 3, 3)
            / 16.0
        )  # widens one-pixel cycle lines to ~3 sub-pixels
        self.cache = {}
        self.keep = {self.key(view_at(0.0)), self.key(view_at(1.0))}  # static views

    @staticmethod
    def key(view):
        return tuple(round(float(v), 12) for v in view)

    @torch.no_grad()
    def __call__(self, view):
        """Brightness (DIAG_H, WIDTH) in [0, 1] for view = (r0, r1, x0, x1)."""
        key = self.key(view)
        if key in self.cache:
            return self.cache[key]
        r0, r1, y0, y1 = view
        r = r0 + (r1 - r0) * (self.cols + self.u) / self.w2
        x = self.x0.clone()
        for _ in range(TRANSIENT):
            x = r * x * (1 - x)
        record = torch.empty((RECORDED,) + x.shape, device=self.device, dtype=x.dtype)
        for i in range(RECORDED):
            x = r * x * (1 - x)
            record[i] = x
        row = ((y1 - record) * (self.h2 / (y1 - y0))).floor().long()
        ok = (row >= 0) & (row < self.h2)
        idx = (row * self.w2 + self.col_index[None])[ok]
        counts = torch.bincount(idx, minlength=self.h2 * self.w2).float()
        counts = torch.nn.functional.conv2d(
            counts.view(1, 1, self.h2, self.w2), self.soften, padding=1
        )[0, 0]
        v = 1 - torch.exp(-counts / DENSITY)
        v = torch.nn.functional.avg_pool2d(v[None, None], SS)[0, 0]
        if key in self.keep:
            self.cache[key] = v
        return v


class Fonts:
    def __init__(self):
        self.fonts = {}

    def __call__(self, face, size):
        key = face, int(size)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(TTF / f"{face}.ttf"), int(size))
        return self.fonts[key]


def generate(settings: ImageProcessingSettings = None):
    """
    The logistic map x -> r x (1 - x) and its bifurcation diagram, a one-shot.

    Time drives one cursor value r(τ): SWEEP1 crosses r = 2.86 .. 4 with a monotone
    PCHIP time-warp that slows through the period doublings (3, 3.449, 3.544,
    3.5644, accumulating at r∞ = 3.5699) and the period-3 window (1 + √8 = 3.8284
    to its crisis at 3.8568); the cursor then glides back to r = 3.835 while the
    camera dives geometrically (each axis about its own fixed point, eased) into
    the middle branch of the window, where the whole diagram reappears in
    miniature (the third iterate f³ is renormalisable there), and SWEEP2 crosses
    the copy: 3 -> 6 -> 12 -> chaos -> crisis.

    Diagram (GPU, torch): for every one of 2160 sub-pixel columns, SAMPLES orbits
    with stratified r and random starts (both drawn once from settings.rng) are
    iterated TRANSIENT times in float64 (float32 orbits collapse onto rounding cycles), then RECORDED iterates are histogrammed
    into rows; brightness 1 - exp(-count / DENSITY), 2x2 averaged, coloured
    through DENSITY_COLOURS with a soft bloom. The main and final views are
    computed once; during the dive each frame recomputes its view (~30 ms).
    Columns beyond the furthest r the cursor has reached show at GHOST.

    Overlays (cv2 at SS x, PIL text): the cursor line with the attractor at r(τ)
    (64 float64 iterates after 6000), sparks where the melody orbit lands, the
    cobweb (staircase) of that orbit under the parabola of the current r, and
    the readout r = …, period p (detected to 2e-4 after 6000 iterations, else
    "chaos").

    The melody orbit is one sequential orbit x_{k+1} = r(t_k) x_k (1 - x_k) with
    a step every sixteenth note at BPM, precomputed up front; every frame is a
    pure function of τ and that list. Frames render in order on the GPU.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    curve_tau, curve_r, curve_lit, curve_zoom = timeline()
    note_t, note_r, note_x = melody_orbit(curve_tau, curve_r)
    frame_tau = np.arange(FRAME_COUNT) / (FRAME_COUNT - 1) * DURATION
    frame_r = np.interp(frame_tau, curve_tau, curve_r)
    frame_pts = attractor(frame_r)
    frame_period = period_of(frame_pts)

    diagram = Diagram(rng, device)
    lut = torch.tensor(palette_lut(DENSITY_COLOURS, 1024), device=device)
    kernel = gaussian_kernel(5.0, device)
    fonts = Fonts()
    columns = torch.arange(WIDTH, device=device).float()

    # Static base: the background and the rule, RGB on the GPU.
    head = Image.new("RGBA", (WIDTH, DIAG_Y), (0, 0, 0, 0))
    pen = ImageDraw.Draw(head)
    it, rm = fonts("STIXGeneralItalic", 84), fonts("STIXGeneral", 84)
    parts = [("x", it), ("  →  ", rm), ("r", it), (" x", it), (" (1 − ", rm)]
    parts += [("x", it), (")", rm)]
    width = sum(f.getlength(s) for s, f in parts)
    left = (WIDTH - width) / 2
    for s, f in parts:
        pen.text((left, 205), s, font=f, fill=(236, 228, 214, 255), anchor="ls")
        left += f.getlength(s)
    head = torch.tensor(np.asarray(head, np.float32), device=device)
    bg = torch.tensor(hex_to_rgb(BG), device=device)
    accent = torch.tensor(hex_to_rgb(ACCENT), device=device)
    base = bg.expand(HEIGHT, WIDTH, 3).clone()
    base[:DIAG_Y] += (head[..., :3] - base[:DIAG_Y]) * (head[..., 3:] / 255.0)
    glow_kernel = gaussian_kernel(7.0, device)
    accent_rgb = tuple(int(ACCENT[i : i + 2], 16) for i in (1, 3, 5))
    S = SS * 16  # cv2 fixed point (shift 4) on the supersampled canvas

    def pt(px, py):
        return int(round(px * S)), int(round(py * S))

    def to_screen(r, x, view):
        r0, r1, y0, y1 = view
        return (r - r0) / (r1 - r0) * WIDTH, DIAG_Y + (y1 - x) / (y1 - y0) * DIAG_H

    def web(x, y):
        return WEB_X + x * WEB_S, WEB_Y + (1 - y) * WEB_S

    def readout(r_now, period, z, view):
        text = Image.new("RGBA", (WIDTH, 80), (0, 0, 0, 0))
        pen = ImageDraw.Draw(text)
        f_it, f_rm = fonts("STIXGeneralItalic", 46), fonts("STIXGeneral", 46)
        cream = (236, 228, 214, 235)
        pen.text((60, 60), "r", font=f_it, fill=cream, anchor="ls")
        pen.text(
            (60 + f_it.getlength("r"), 60),
            f" = {r_now:.4f}",
            font=f_rm,
            fill=cream,
            anchor="ls",
        )
        label = "settles" if period == 1 else f"period {period}"
        label = "chaos" if period == 0 else label
        pen.text(
            (WIDTH - 60, 60), label, font=f_rm, fill=accent_rgb + (255,), anchor="rs"
        )
        if z > 0.02:
            zoom = (MAIN_VIEW[1] - MAIN_VIEW[0]) / (view[1] - view[0])
            pen.text(
                (WIDTH / 2, 60),
                f"×{zoom:.0f}",
                font=f_rm,
                fill=(236, 228, 214, int(200 * min(1, z * 4))),
                anchor="ms",
            )
        return np.asarray(text)

    def render(frame):
        tau = frame_tau[frame]
        r_now = frame_r[frame]
        z = float(np.interp(tau, curve_tau, curve_zoom))
        view = view_at(z)
        lit = float(np.interp(tau, curve_tau, curve_lit))

        # --- the diagram
        v = diagram(view)
        col_r = view[0] + (view[1] - view[0]) * (columns + 0.5) / WIDTH
        edge = (view[1] - view[0]) * 3.0 / WIDTH  # a 3 px soft edge
        shown = GHOST + (1 - GHOST) * torch.clamp((lit - col_r) / edge + 0.5, 0, 1)
        v = (v * shown[None, :]).clamp(0, 1)
        rgb = lut[(v**0.8 * 1023).long()] * 255.0
        rgb = rgb + 0.55 * blur(rgb, kernel)
        image = base.clone()
        image[DIAG_Y : DIAG_Y + DIAG_H] = torch.maximum(rgb, bg)

        # --- accent strokes (cv2, uint8): sharp at SS x, glow at 1 x
        lines = np.zeros((HEIGHT * SS, WIDTH * SS), np.uint8)
        glow = np.zeros((HEIGHT, WIDTH), np.uint8)

        cx, _ = to_screen(r_now, 0.0, view)
        if -5 < cx < WIDTH + 5:
            top, bottom = pt(cx, DIAG_Y), pt(cx, DIAG_Y + DIAG_H)
            cv2.line(lines, top, bottom, 56, SS, cv2.LINE_AA, 4)
            for x in frame_pts[frame]:
                px, py = to_screen(r_now, x, view)
                if DIAG_Y < py < DIAG_Y + DIAG_H:
                    cv2.circle(lines, pt(px, py), int(3.0 * S), 230, -1, cv2.LINE_AA, 4)
                    cv2.circle(glow, (int(px), int(py)), 3, 64, -1, cv2.LINE_AA)

        k_now = int(np.searchsorted(note_t, tau, side="right")) - 1
        for k in range(max(0, k_now - 8), k_now + 1):
            age = tau - note_t[k]
            if age < 0 or age > 0.9:
                continue
            fade = np.exp(-age / 0.25)
            px, py = to_screen(note_r[k], note_x[k + 1], view)
            if DIAG_Y < py < DIAG_Y + DIAG_H and -10 < px < WIDTH + 10:
                radius = int((4.0 + 10.0 * fade) * S)
                cv2.circle(
                    lines, pt(px, py), radius, int(255 * fade), -1, cv2.LINE_AA, 4
                )
                cv2.circle(
                    glow,
                    (int(px), int(py)),
                    int(14 * fade + 4),
                    int(230 * fade),
                    -1,
                    cv2.LINE_AA,
                )

        box = np.array([web(0, 0), web(1, 0), web(1, 1), web(0, 1)]) * S
        cv2.polylines(lines, [box.astype(np.int32)], True, 46, SS, cv2.LINE_AA, 4)
        cv2.line(lines, pt(*web(0, 0)), pt(*web(1, 1)), 64, SS, cv2.LINE_AA, 4)
        curve = np.stack(web(PARABOLA, r_now * PARABOLA * (1 - PARABOLA)), 1) * S
        cv2.polylines(
            lines, [curve.astype(np.int32)], False, 140, 2 * SS, cv2.LINE_AA, 4
        )
        if k_now >= 0:
            for k in range(max(0, k_now - 14), k_now + 1):
                a, b = note_x[k], note_x[k + 1]
                strength = int(230 * 0.82 ** (k_now - k))
                grow = 1.0
                if k == k_now:
                    grow = float(smoothstep((tau - note_t[k]) / (0.6 * STEP)))
                p0, p1, p2 = web(a, a), web(a, b), web(b, b)  # up to f(a), across
                g1, g2 = min(1.0, 2 * grow), max(0.0, 2 * grow - 1)
                q1 = (p0[0], p0[1] + (p1[1] - p0[1]) * g1)
                cv2.line(lines, pt(*p0), pt(*q1), strength, SS, cv2.LINE_AA, 4)
                if g2 > 0:
                    q2 = (p1[0] + (p2[0] - p1[0]) * g2, p1[1])
                    cv2.line(lines, pt(*p1), pt(*q2), strength, SS, cv2.LINE_AA, 4)
            a, b = note_x[k_now], note_x[k_now + 1]
            fade = 0.4 + 0.6 * np.exp(-(tau - note_t[k_now]) / 0.2)
            hx, hy = web(a, b)
            cv2.circle(
                lines, pt(hx, hy), int(5 * S), int(255 * fade), -1, cv2.LINE_AA, 4
            )
            cv2.circle(glow, (int(hx), int(hy)), 12, int(180 * fade), -1, cv2.LINE_AA)

        sharp = torch.from_numpy(lines).to(device).float()[None, None] / 255.0
        sharp = torch.nn.functional.avg_pool2d(sharp, SS)[0, 0]
        soft = torch.from_numpy(glow).to(device).float()[..., None] / 255.0
        soft = blur(soft, glow_kernel)[..., 0]
        light = (sharp + soft).clamp(0, 1.4)[..., None]
        image += (accent - image) * light.clamp(0, 1) + 60.0 * (light - 1).clamp(min=0)

        # --- readout
        text = torch.from_numpy(np.array(readout(r_now, frame_period[frame], z, view)))
        text = text.to(device).float()
        top = READ_Y - 60
        alpha = text[..., 3:] / 255.0
        image[top : top + 80] += (text[..., :3] - image[top : top + 80]) * alpha

        end = float(smoothstep((DURATION - tau) / FADE))
        return bg + (image - bg) * end

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(
        curve_tau, curve_r, curve_zoom, note_t, note_r, note_x, style_name,
        settings.audio_rng,
    )  # fmt: skip
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


BASS_DEGREE = {1: 0, 2: 4, 3: 3, 4: 2, 6: 5, 8: 6, 12: 5, 16: 6, 24: 6, 32: 6, 0: 1}


def soundtrack(curve_tau, curve_r, curve_zoom, note_t, note_r, note_x, style_name, rng):
    """
    The melody orbit played note for note, one-shot (circular=False everywhere).

    Every sixteenth the orbit takes one step x -> r x (1 - x) and the style's voice
    strikes it: the pitch is the dot's height in the view on screen (SPAN degrees
    of MODE from LOWEST at the bottom), so a fixed point is one repeated note, a
    2-cycle a trill, a 4-cycle an arpeggio, chaos a melody that never repeats; in
    the dive the two off-screen points of the 3-cycle are clipped to the edges
    and played softly. Pan follows the cursor across the screen. Each change of
    the attractor's period (sampled every 20 ms along the sweeps, independent of
    FPS, kept if it lasts 0.12 s) strikes a bass note (BASS_DEGREE). Under it a
    home-chord pad dims while a cluster pad (adjacent degrees) swells with the
    chaos, both from the same period curve; when the notes stop, the home chord
    rings out over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), MODE)
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    damp_t = np.arange(int(4.0 * audio.SAMPLE_RATE)) / audio.SAMPLE_RATE
    damp = np.exp(-np.maximum(damp_t - DAMP, 0.0) / 0.12)

    def strike(degree, velocity, when, x=0.0, octave=0, voice=None, damped=False):
        velocity = round(float(np.clip(velocity, 0.05, 1.0)) * 20) / 20  # cache hits
        note = (voice or style.voice)(scale.hz(int(degree), octave), velocity)
        if damped:  # a dead stroke: the melody stays legible at six notes a second
            note = note * damp[: len(note)]
        audio.add(track, audio.pan(note, float(np.clip(x, -1, 1))), when, **kw)

    # melody: one strike per step of the orbit
    for k, when in enumerate(note_t):
        view = view_at(float(np.interp(when, curve_tau, curve_zoom)))
        height = (note_x[k + 1] - view[2]) / (view[3] - view[2])
        seen = 0.0 <= height <= 1.0
        degree = LOWEST + int(round(np.clip(height, 0, 1) * SPAN))
        cx = (note_r[k] - view[0]) / (view[1] - view[0])
        velocity = (0.36 + 0.14 * (k % 4 == 0)) if seen else 0.26
        strike(degree, velocity, when, 0.6 * (2 * np.clip(cx, 0, 1) - 1), damped=True)

    # the attractor's period along the sweeps, every 20 ms
    grid = np.arange(0.0, NOTES_END, 0.02)
    in_glide = (grid > SWEEP1[-1][0]) & (grid < SWEEP2[0][0])
    period = period_of(attractor(np.interp(grid, curve_tau, curve_r)))
    chaos = (period == 0).astype(float)
    changes, current = [], period[0]
    for i in range(1, len(grid)):
        if in_glide[i]:
            continue
        if period[i] != current:
            hold = (period[i : i + 6] == period[i]).all()  # lasts 0.12 s
            if hold:
                changes.append((grid[i], period[i]))
                current = period[i]
    logger.info(
        "Period changes: "
        + ", ".join(f"{t:.2f}s->{'chaos' if p == 0 else p}" for t, p in changes)
    )
    strike(0, 0.55, 0.0, octave=-2, voice=style.bass)
    for when in (GLIDE[1], 0.5 * (GLIDE[1] + SWEEP2[0][0])):  # the dive: period 3
        strike(BASS_DEGREE[3], 0.5, when, octave=-2, voice=style.bass)
    for when, p in changes:
        strike(BASS_DEGREE.get(int(p), 1), 0.5, when, octave=-2, voice=style.bass)

    # pads: order dims, a cluster swells with the chaos
    t_audio = np.arange(len(track)) / audio.SAMPLE_RATE
    smooth = np.convolve(chaos, np.ones(15) / 15, mode="same")
    env = np.interp(t_audio, grid, smooth, right=0.0)
    for degrees, gain in (((0, 2, 4), 1.0 - 0.7 * env), ((0, 1, 3, 4), 0.9 * env)):
        for degree in degrees:
            freq = scale.hz(degree, -1)
            pad = audio.pad(freq, NOTES_END + 0.2, 0.15, 0.6, 0.8, 0.35)[: len(track)]
            pad = pad * gain[: len(pad), None]
            audio.add(track, pad, 0.0, **kw)

    # the notes stop: home
    end = NOTES_END + STEP
    strike(0, 0.6, end, octave=-2, voice=style.bass)
    for i, degree in enumerate((0, 2, 4, 7)):
        strike(degree, 0.4, end + 0.06 * i, 0.3 * (i % 2) - 0.15)
    for degree in (0, 2, 4):
        audio.add(track, audio.pad(scale.hz(degree, -1), 0.6, 0.15, 0.3, 0.6, 0.3),
                  end, **kw)  # fmt: skip
    track = audio.highpass(style.room(track, rng, circular=False), 60.0, circular=False)
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
