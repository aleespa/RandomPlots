"""
It Always Becomes a Circle

Curve-shortening flow: every point of this closed curve moves inwards at a
speed equal to how sharply the curve bends there. Tight corners race, straight
walls wait. The faint lines are where the curve was, at equal steps of time.

In 1986–87 Michael Gage, Richard Hamilton and Matthew Grayson proved that any
closed curve that never crosses itself, however tangled, untangles under this
flow, turns convex, then round, and shrinks to a single point. Nothing of the
starting shape survives.

And a second surprise: the area inside drains at exactly the same rate, whatever
the shape, dA/dt = −2π. So the moment of collapse is known from the very
start: T = A₀ / 2π.

Every faint line strikes a handpan, pitched by the length of the curve: low
while it is long, rising as it shortens. An uneasy chord holds until the very
moment the curve turns convex, and a bright one lands as it vanishes.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#curveshorteningflow #differentialgeometry #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from numba import njit

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FLOW_END = 13.2  # s: the curve collapses here, then the contour map rests
FADE = 0.4  # s of fade to black at the very end
FLOW_POWER = 1.8  # video time -> flow time: more screen time for the tangled start

COLS, ROWS = 7, 12  # maze cells
CORRIDOR = 0.42  # corridor width, as a fraction of a cell
GHOSTS = 60  # contour lines left behind, at equal steps of flow time
DS_MAX, DS_MIN, N_MIN = 3.0, 0.4, 48  # polygon spacing (px) and minimum points
END_AREA = 2.0  # px^2: the curve counts as a point below this

STYLE = "handpan"  # common.music.STYLES: a handpan stroke on every ghost contour
PITCH_SPAN = 14  # scale degrees climbed as the curve's length shrinks
RISER = 1.8  # s of build-up into the collapse

SS = 2  # supersampling
SHIFT = 4  # cv2 sub-pixel bits
PALETTE = ["#3a2a8c", "#1f8fa3", "#d0448c", "#f5c04a"]  # indigo, teal, magenta, gold
GHOST_GAIN = 0.55
CURV_BINS = 8


def maze_tree(rng, cols, rows):
    """Edges of a random spanning tree of the grid (randomised depth-first search)."""
    seen = np.zeros((rows, cols), bool)
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    stack, edges = [start], []
    seen[start] = True
    while stack:
        r, c = stack[-1]
        options = [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < rows and 0 <= c + dc < cols and not seen[r + dr, c + dc]
        ]
        if not options:
            stack.pop()
            continue
        nxt = options[int(rng.integers(len(options)))]
        seen[nxt] = True
        edges.append(((r, c), nxt))
        stack.append(nxt)
    return edges


def initial_curve(rng):
    """The outline of the thickened maze tree: a simple closed curve, (N, 2) px."""
    cell = (WIDTH - 2 * 80) / COLS
    x0 = (WIDTH - cell * (COLS - 1)) / 2
    y0 = (HEIGHT - cell * (ROWS - 1)) / 2
    width = CORRIDOR * cell
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for (r1, c1), (r2, c2) in maze_tree(rng, COLS, ROWS):
        p1 = (int(round(x0 + c1 * cell)), int(round(y0 + r1 * cell)))
        p2 = (int(round(x0 + c2 * cell)), int(round(y0 + r2 * cell)))
        cv2.line(mask, p1, p2, 255, int(round(width)), cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), width * 0.18)
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    return resample(contour, DS_MAX, DS_MIN, N_MIN)


@njit(cache=True, nogil=True)
def resample(x, ds_max, ds_min, n_min):
    """Uniform arclength resampling of a closed polygon."""
    n = x.shape[0]
    cum = np.empty(n + 1)
    cum[0] = 0.0
    for i in range(n):
        j = (i + 1) % n
        cum[i + 1] = cum[i] + np.hypot(x[j, 0] - x[i, 0], x[j, 1] - x[i, 1])
    length = cum[n]
    m = max(n_min, int(length / ds_max))
    if length / m < ds_min:
        m = max(8, int(length / ds_min))
    out = np.empty((m, 2))
    k = 0
    for i in range(m):
        s = length * i / m
        while cum[k + 1] < s:
            k += 1
        seg = cum[k + 1] - cum[k]
        a = (s - cum[k]) / seg if seg > 0 else 0.0
        j = (k + 1) % n
        out[i, 0] = x[k, 0] + a * (x[j, 0] - x[k, 0])
        out[i, 1] = x[k, 1] + a * (x[j, 1] - x[k, 1])
    return out


@njit(cache=True, nogil=True)
def area(x):
    n = x.shape[0]
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += x[i, 0] * x[j, 1] - x[j, 0] * x[i, 1]
    return abs(0.5 * a)


@njit(cache=True, nogil=True)
def curvature_vector(x):
    """kappa * N at each vertex (points towards the centre of curvature)."""
    n = x.shape[0]
    k = np.empty((n, 2))
    for i in range(n):
        p, q = x[i - 1], x[(i + 1) % n]
        ex, ey = q[0] - x[i, 0], q[1] - x[i, 1]
        fx, fy = p[0] - x[i, 0], p[1] - x[i, 1]
        le, lf = np.hypot(ex, ey), np.hypot(fx, fy)
        s = 2.0 / (le + lf)
        k[i, 0] = s * (ex / le + fx / lf)
        k[i, 1] = s * (ey / le + fy / lf)
    return k


@njit(cache=True, nogil=True)
def flow_until(x, target):
    """Run curve-shortening flow until the enclosed area drops to `target`."""
    while True:
        x = resample(x, DS_MAX, DS_MIN, N_MIN)
        if area(x) <= target:
            return x
        n = x.shape[0]
        length = 0.0
        for i in range(n):
            j = (i + 1) % n
            length += np.hypot(x[j, 0] - x[i, 0], x[j, 1] - x[i, 1])
        ds = length / n
        dt = 0.2 * ds * ds
        for _ in range(10):
            x = x + dt * curvature_vector(x)
            if area(x) <= target:
                return x


def flow_schedule(t):
    """Fraction of the flow done at normalised video time t in [0, 1]: slow start."""
    return np.clip(t, 0.0, 1.0) ** FLOW_POWER


def perimeter(x):
    return float(np.hypot(*(np.roll(x, -1, axis=0) - x).T).sum())


def soundtrack(curves, ghost_fraction, frame_fraction, rng):
    """
    Rise & resolve on a handpan. Ghost contour k appears at video time
    FLOW_END (k / GHOSTS)^(1 / FLOW_POWER) and strikes one note: pitch from the
    log of the curve's length (long = low), climbing PITCH_SPAN degrees by the
    last ghost, panned by the curve's centroid. A whole-tone pad holds until the
    curve turns convex (area / convex-hull area > 0.999 from then on), a minor
    string chord takes over, a riser builds into the collapse at FLOW_END and a
    major choir chord with a high bell lands there and rings out.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    root = scale.root
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    def hz(midi):
        return float(audio.midi_to_hz(midi))

    convex = [
        area(curves[f].astype(np.float64))
        >= 0.999 * cv2.contourArea(cv2.convexHull(curves[f]))
        for f in frame_fraction
    ]
    bent = [i for i, c in enumerate(convex) if not c]
    turn = (bent[-1] + 1) / FPS if bent else 0.0
    logger.info(f"The curve turns convex at {turn:.2f} s")

    lengths = np.log([perimeter(curves[g]) for g in ghost_fraction])
    climb = (lengths[0] - lengths) / (lengths[0] - lengths[-1])
    for k, g in enumerate(ghost_fraction):
        when = max(0.03, FLOW_END * g ** (1 / FLOW_POWER))
        degree = -3 + int(round(PITCH_SPAN * climb[k]))
        x = curves[g][:, 0].mean() / WIDTH * 2 - 1
        note = style.voice(scale.hz(degree), 0.3 + 0.3 * (1 - climb[k]))
        audio.add(track, audio.pan(note, 0.6 * x), when, **kw)
    audio.add(track, style.bass(scale.hz(0, -2), 0.6), 0.03, **kw)

    for semitones in (0, 2, 6, 10):  # whole-tone: no home anywhere
        voice = audio.pad(hz(root - 12 + semitones), turn, 0.1, 0.4, 1.2, 0.35)
        audio.add(track, audio.lowpass(voice, 1800.0, circular=False), 0.0, **kw)
    for semitones in (-12, 0, 3, 7):  # minor home chord once convex
        voice = audio.strings(
            hz(root + semitones), FLOW_END - turn, 0.2, 1.0, 0.6, 0.35
        )
        audio.add(track, voice, turn, **kw)
    audio.add(
        track, audio.riser(RISER, rng, 180.0, 1800.0, 0.22), FLOW_END - RISER, **kw
    )

    ring = DURATION - FLOW_END
    for semitones in (0, 4, 7, 12):  # the circle: a major chord
        voice = audio.choir(hz(root + semitones), ring - 1.0, 0.32, 0.05, 1.5, "o")
        audio.add(track, voice, FLOW_END, **kw)
    audio.add(
        track,
        audio.pad(hz(root - 24), ring - 1.0, 0.35, 0.05, 1.5, 0.3),
        FLOW_END,
        **kw,
    )
    audio.add(track, audio.bell(hz(root + 24), 0.5), FLOW_END, **kw)
    audio.add(track, audio.bell(hz(root + 31), 0.35), FLOW_END + 0.12, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Curve-shortening flow x_t = kappa N of a simple closed curve: the outline of a
    thickened random spanning tree on a 7x12 grid (randomised DFS from
    `settings.rng`, so the seed picks the maze).

    The flow is a cheap sequential state update, done once up front: a polygon
    with ~3 px uniform spacing (resampled every 10 explicit steps, dt = 0.2 ds^2)
    in numba. Since dA/dt = -2pi exactly, the area is the clock: video frame f is
    the curve at area A0 (1 - h(tau / FLOW_END)), h(u) = u^FLOW_POWER (the
    fine, fast early motion gets more screen time), so the curve dies at FLOW_END
    s. Ghost contours are the curve at area
    A0 (1 - k / GHOSTS), i.e. at equal steps of flow time, coloured by time with
    an indigo-teal-magenta-gold palette. The live curve is brightened towards
    white where |kappa| is large and gets a blurred glow. At the collapse a flash
    blooms at the vanishing point; the contour map rests and fades out.

    Frames are drawn in parallel with cv2 at 2x and downsampled.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    curve = initial_curve(rng)
    a0 = area(curve)
    width = CORRIDOR * (WIDTH - 2 * 80) / COLS
    logger.info(f"Initial curve: {len(curve)} points, area {a0:.0f} px^2")

    # Every area at which a curve is needed: frames, then ghosts.
    flow_frames = [f for f in range(FRAME_COUNT) if f / FPS < FLOW_END]
    frame_fraction = flow_schedule(np.array(flow_frames) / FPS / FLOW_END)
    ghost_fraction = np.arange(GHOSTS) / GHOSTS
    fractions = np.unique(np.concatenate([frame_fraction, ghost_fraction]))
    curves = {}
    x = curve
    for fraction in fractions:  # ascending: area descending
        target = max(a0 * (1 - fraction), END_AREA)
        x = flow_until(x, target)
        curves[fraction] = x.astype(np.float32)
    end_point = curves[fractions[-1]].mean(axis=0)
    logger.info(f"Flow done: collapses at ({end_point[0]:.0f}, {end_point[1]:.0f})")

    lut = (palette_lut(PALETTE) * 255)[:, ::-1]  # BGR, 0..255

    def colour(fraction, gain=1.0):
        c = lut[min(255, int(fraction * 255))] * gain
        return tuple(float(v) for v in c)

    ghost_polys = [
        (
            np.round(curves[g] * SS * (1 << SHIFT)).astype(np.int32),
            colour(g, GHOST_GAIN),
        )
        for g in ghost_fraction
    ]
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    dist2 = (xx - end_point[0]) ** 2 + (yy - end_point[1]) ** 2
    del xx, yy

    def render(frame):
        tau = frame / FPS
        flowing = tau < FLOW_END
        fraction = frame_fraction[frame] if flowing else 1.0

        big = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
        for g, (poly, c) in zip(ghost_fraction, ghost_polys):
            if g <= fraction:
                cv2.polylines(big, [poly], True, c, SS, cv2.LINE_AA, SHIFT)

        glow = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        if flowing:
            x = curves[fraction]
            kappa = np.hypot(*curvature_vector(x.astype(np.float64)).T)
            hot = np.clip(kappa * width / 2, 0, 1) ** 0.7
            bins = np.minimum((hot * CURV_BINS).astype(int), CURV_BINS - 1)
            base = np.array(colour(fraction))
            segs = np.stack([x, np.roll(x, -1, axis=0)], axis=1)
            fine = np.round(segs * SS * (1 << SHIFT)).astype(np.int32)
            coarse = np.round(segs * (1 << SHIFT)).astype(np.int32)
            for b in range(CURV_BINS):
                sel = bins == b
                if not sel.any():
                    continue
                w = (b + 0.5) / CURV_BINS
                c = tuple(float(v) for v in base * (1 - w) * 1.3 + 255 * w)
                c = tuple(min(255.0, v) for v in c)
                cv2.polylines(
                    big, list(fine[sel]), False, c, 2 * SS, cv2.LINE_AA, SHIFT
                )
                cv2.polylines(glow, list(coarse[sel]), False, c, 5, cv2.LINE_AA, SHIFT)

        image = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        if flowing:
            image += 1.1 * cv2.GaussianBlur(glow, (0, 0), 7).astype(np.float32)

        since = tau - FLOW_END
        if since >= 0:  # the flash at the vanishing point
            radius = 25 + 160 * (1 - np.exp(-since / 0.25))
            strength = np.exp(-since / 0.45) * smoothstep(since / 0.04)
            flash = strength * np.exp(-dist2 / (2 * radius * radius))
            core = strength * np.exp(-dist2 / (2 * 6.0**2))
            image += flash[..., None] * np.array(colour(1.0)) * 1.4
            image += core[..., None] * 255
        image *= smoothstep((DURATION - tau) / FADE)
        return image

    render_frames(settings, render, FRAME_COUNT)
    track = soundtrack(curves, ghost_fraction, frame_fraction, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
