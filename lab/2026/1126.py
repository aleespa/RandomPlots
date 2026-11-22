"""
A Graph That Untangles Itself

A random graph of 584 dots and 1,685 links: a hairball with over three
hundred thousand crossings. Two rules untangle it. Pin the outer ring of dots
to a circle, and let every other dot slide to the average position of its
neighbours. The crossings melt away until there are none.

This always works. Tutte's spring theorem (1963): make every link a rubber
band, nail the outer face to a convex shape, and any well-connected graph
that can be drawn without crossings settles into exactly such a drawing. It
is how computers lay out networks and flatten 3D meshes.

Each time the number of crossings halves, a harp string is plucked one
step higher, over a chord that only resolves when the last crossing goes.

Made with Python, NumPy, SciPy, Numba & OpenCV. Sound synthesized in NumPy.
#planargraph #graphtheory #mathart #generativeart #randomplots
"""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from loguru import logger
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    hex_to_bgr,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = '340,000 crossings. Two rules. Zero left.'
HOOK_Y = 440
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

# The graph: a Delaunay triangulation of a disc, the rim on the unit circle.
RIM = 64  # nodes pinned to the circle
INTERIOR = 520  # free nodes
SPACING = 0.055  # minimum distance between interior nodes (unit disc)
INNER = 0.93  # interior nodes lie within this radius

# Timeline (s).
PIN_START, PIN_END = 0.2, 2.3  # the rim nodes glide to the circle, one after another
PIN_GLIDE = 0.7  # each rim node's glide
RELAX_START = 1.9  # the springs let go
ZERO_AT = 9.0  # the last crossing disappears (the relaxation is paced around it)
RELAX_END = 12.2  # the drawing has (visibly) converged
TAU_START, TAU_END = 0.006, 220.0  # spring time at RELAX_START / RELAX_END

# Layout (output px).
CENTRE = (540.0, 1080.0)
RADIUS = 468.0
COUNTER_Y = 330  # baseline-ish centre of the crossing counter
NOTE_Y = 1700  # the closing line of text
SS = 2  # supersampling
SHIFT = 4  # cv2 sub-pixel bits

BACKGROUND = "#07060b"
HALO = "#140f20"  # faint light behind the disc
# Edge colour by how many other edges it crosses: none .. the most.
SILK = "#eef6d8"
HEAT = ["#ffd36e", "#ff8a47", "#ff3d86", "#c63cff"]
CALM = ["#ffeeb0", "#e9f6ff", "#b7a6ff"]  # crossing-free edges, centre to rim
RIM_COLOUR = "#ffffff"
TEXT_COLOUR = "#eef6d8"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
FONT_NOTE = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
NOTE = "every dot at the average of its neighbours"

# common.music.STYLES; an --audio-only tag naming one overrides it
STYLE = "harp"


# --- The graph -------------------------------------------------------------------


def build_graph(rng):
    """Nodes (rim first) of a triangulated disc, their edges, and the start tangle."""
    points = []
    while len(points) < INTERIOR:
        p = rng.uniform(-1.0, 1.0, 2)
        if p @ p > INNER**2:
            continue
        if points and np.min(np.sum((np.asarray(points) - p) ** 2, 1)) < SPACING**2:
            continue
        points.append(p)
    angles = 2 * np.pi * np.arange(RIM) / RIM
    rim = np.c_[np.cos(angles), np.sin(angles)]
    nodes = np.vstack([rim, np.asarray(points)])
    edges = set()
    for a, b, c in Delaunay(nodes).simplices:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    edges = np.array(sorted(edges), dtype=np.int64)
    # The tangle: every node (rim included) dropped uniformly in the disc.
    n = len(nodes)
    r = 0.97 * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2 * np.pi, n)
    start = np.c_[r * np.cos(theta), r * np.sin(theta)]
    return rim, edges, start


class Springs:
    """
    The spring flow of the interior with the rim held at `b`:
    dx/dτ = -(L_II x + L_IB b), solved in closed form from the eigenvectors of L_II.
    """

    def __init__(self, n, edges):
        weights = np.zeros((n, n))
        weights[edges[:, 0], edges[:, 1]] = 1.0
        weights[edges[:, 1], edges[:, 0]] = 1.0
        laplacian = np.diag(weights.sum(1)) - weights
        inner = np.arange(RIM, n)
        l_ii = laplacian[np.ix_(inner, inner)]
        l_ib = laplacian[np.ix_(inner, np.arange(RIM))]
        self.harmonic = -np.linalg.solve(l_ii, l_ib)  # Tutte: x_I = H b
        self.lam, self.modes = np.linalg.eigh(l_ii)

    def interior(self, start_inner, rim, tau):
        rest = self.harmonic @ rim
        offset = self.modes.T @ (start_inner - rest)
        return rest + self.modes @ (np.exp(-self.lam * tau)[:, None] * offset)


@njit(nogil=True, cache=True)
def crossings(x, edges):
    """Per edge, how many other edges it properly crosses (shared ends don't count)."""
    m = len(edges)
    count = np.zeros(m, np.int32)
    for i in range(m):
        a, b = edges[i, 0], edges[i, 1]
        ax, ay, bx, by = x[a, 0], x[a, 1], x[b, 0], x[b, 1]
        lox, hix = min(ax, bx), max(ax, bx)
        loy, hiy = min(ay, by), max(ay, by)
        for j in range(i + 1, m):
            c, d = edges[j, 0], edges[j, 1]
            if c == a or c == b or d == a or d == b:
                continue
            cx, cy, dx, dy = x[c, 0], x[c, 1], x[d, 0], x[d, 1]
            if max(cx, dx) < lox or min(cx, dx) > hix:
                continue
            if max(cy, dy) < loy or min(cy, dy) > hiy:
                continue
            d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
            if d1 * d2 >= 0:
                continue
            d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
            d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
            if d3 * d4 >= 0:
                continue
            count[i] += 1
            count[j] += 1
    return count


def graph_distance(n, edges, source):
    """Breadth-first hop count from `source` to every node."""
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    dist = np.full(n, -1)
    dist[source] = 0
    frontier = [source]
    while frontier:
        nxt = []
        for u in frontier:
            for v in neighbours[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


class Motion:
    """Positions of every node as a pure function of the time in seconds."""

    def __init__(self, rim, edges, start):
        self.rim, self.edges, self.start = rim, edges, start
        self.n = len(start)
        self.springs = Springs(self.n, edges)
        self.final = np.vstack([rim, self.springs.harmonic @ rim])
        # Each rim node glides to its place on the circle, in order round it.
        order = np.arange(RIM)
        span = PIN_END - PIN_START - PIN_GLIDE
        self.pin_at = PIN_START + span * order / (RIM - 1)
        self.log_tau = None

    def rim_at(self, seconds):
        u = smoothstep((seconds - self.pin_at) / PIN_GLIDE)[:, None]
        return (1 - u) * self.start[:RIM] + u * self.rim

    def tau(self, seconds):
        if seconds <= RELAX_START:
            return 0.0
        if seconds >= RELAX_END:
            return np.inf
        return 10.0 ** float(self.log_tau(seconds))

    def at(self, seconds):
        rim = self.rim_at(seconds)
        tau = self.tau(seconds)
        if np.isinf(tau):
            inner = self.final[RIM:]
        else:
            inner = self.springs.interior(self.start[RIM:], rim, tau)
            # Land exactly on the drawing at RELAX_END (the leftover is far below a pixel).
            land = smoothstep((seconds - (RELAX_END - 0.6)) / 0.6)
            inner = (1 - land) * inner + land * self.final[RIM:]
        return np.vstack([rim, inner])

    def pace(self):
        """
        Spring time τ(s): log-paced from TAU_START to TAU_END, with the spring time at
        which this graph's last crossing disappears placed exactly at ZERO_AT.
        """
        rim = self.rim
        lo, hi = np.log10(0.05), np.log10(50.0)
        for _ in range(40):  # bisection on the (eventually monotone) crossing count
            mid = 0.5 * (lo + hi)
            inner = self.springs.interior(self.start[RIM:], rim, 10.0**mid)
            if crossings(np.vstack([rim, inner]), self.edges).sum() == 0:
                hi = mid
            else:
                lo = mid
        tau_zero = hi + 0.01
        logger.info(f"Last crossing goes at spring time τ = {10 ** tau_zero:.3f}")
        knots = [RELAX_START, RELAX_START + 0.9, ZERO_AT, RELAX_END]
        values = [
            np.log10(TAU_START),
            np.log10(TAU_START) + 1.0,
            tau_zero,
            np.log10(TAU_END),
        ]
        self.log_tau = PchipInterpolator(knots, values)


# --- Drawing ---------------------------------------------------------------------


class Text:
    """Counter and caption bitmaps, cached per string (alpha masks in 0..1)."""

    def __init__(self):
        self.big = ImageFont.truetype(str(FONT), 132)
        self.small = ImageFont.truetype(str(FONT), 34)
        self.note = ImageFont.truetype(str(FONT_NOTE), 40)
        self.cache = {}

    def counter(self, number):
        key = "c", number
        if key not in self.cache:
            canvas = Image.new("L", (WIDTH, 260), 0)
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (WIDTH / 2, 150), f"{number:,}".replace(",", " "), 255, self.big, "ms"
            )
            draw.text((WIDTH / 2, 190), "c r o s s i n g s", 150, self.small, "mt")
            self.cache[key] = np.asarray(canvas, np.float32) / 255.0
        return self.cache[key]

    def line(self):
        key = "n", NOTE
        if key not in self.cache:
            canvas = Image.new("L", (WIDTH, 80), 0)
            ImageDraw.Draw(canvas).text((WIDTH / 2, 40), NOTE, 255, self.note, "mm")
            self.cache[key] = np.asarray(canvas, np.float32) / 255.0
        return self.cache[key]


def blend(image, alpha, top, colour, strength=1.0):
    """Paint `colour` (BGR floats) through an alpha mask whose top row is `top`."""
    region = image[top : top + alpha.shape[0]]
    a = (strength * alpha)[..., None]
    region += (np.asarray(colour, np.float32) - region) * a


def generate(settings: ImageProcessingSettings = None):
    """
    Tutte's spring embedding, animated as a closed-form gradient flow.

    The graph is the Delaunay triangulation of RIM points evenly spaced on the unit
    circle plus INTERIOR jittered points inside it (a triangulated disc: planar,
    3-connected, rim = outer face). Every node starts at a uniform random point of the
    disc. From PIN_START to PIN_END the rim nodes glide (smoothstep, staggered round
    the circle) to their circle positions b. Then the interior follows the spring flow
    dx_I/dτ = -(L_II x_I + L_IB b), whose solution is x_I(τ) = H b + V e^{-Λτ} Vᵀ
    (x_I(0) - H b) with H = -L_II⁻¹ L_IB (the Tutte / barycentric drawing, every
    interior node at the mean of its neighbours) and L_II = V Λ Vᵀ. So a frame is a
    pure function of t. The spring time τ(s) is a monotone PCHIP in log τ from
    TAU_START at RELAX_START to TAU_END at RELAX_END, with the τ at which this graph's
    crossing count first reaches 0 (bisection, up front) pinned to ZERO_AT; the last
    0.6 s blend onto H b exactly.

    Each frame counts, per edge, the proper crossings with every other edge (numba
    O(E²) with bounding-box rejection, ~13 ms) and colours the edge by log(1 + count)
    through HEAT, crossing-free edges in SILK; the rim cycle is drawn brighter once
    pinned. At ZERO_AT a brightness ripple runs out from the node nearest the centre
    by graph distance, and a line of text fades in. Lines are drawn with cv2 at 2x
    and downsampled, plus a Gaussian glow; frames render on the thread pool.

    The soundtrack (STYLE, key from settings.audio_rng) is computed from the same
    crossing counts sampled at 60 Hz; see `soundtrack`. Posted with seed 0.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    rim, edges, start = build_graph(rng)
    motion = Motion(rim, edges, start)
    motion.pace()
    n = motion.n
    crossings(start, edges)  # compile before the thread pool starts
    reference = crossings(start, edges).max()
    logger.info(
        f"{n} nodes, {len(edges)} edges, {crossings(start, edges).sum() // 2} "
        "crossings at the start"
    )

    rim_edge = np.array(
        [(b - a == 1) or (a == 0 and b == RIM - 1) for a, b in edges]
    ) & (edges[:, 1] < RIM)
    centre_node = RIM + int(np.argmin(np.sum(motion.final[RIM:] ** 2, 1)))
    hops = graph_distance(n, edges, centre_node)
    edge_hops = 0.5 * (hops[edges[:, 0]] + hops[edges[:, 1]])

    heat_lut = palette_lut(HEAT, 256)[:, ::-1] * 255.0  # BGR
    silk = np.array(hex_to_bgr(SILK), np.float32)
    calm_lut = palette_lut(CALM, 256)[:, ::-1] * 255.0
    calm = calm_lut[np.round(255 * edge_hops / edge_hops.max()).astype(int)]
    rim_colour = np.array(hex_to_bgr(RIM_COLOUR), np.float32)
    text_colour = np.array(hex_to_bgr(TEXT_COLOUR), np.float32)

    # Background: near black with a faint halo behind the disc.
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    rr = np.hypot(xx - CENTRE[0], yy - CENTRE[1]) / (1.25 * RADIUS)
    halo = np.exp(-(rr**2))[..., None]
    background = np.asarray(hex_to_bgr(BACKGROUND), np.float32)
    background = background + halo * (
        np.asarray(hex_to_bgr(HALO), np.float32) - background
    )
    text = Text()
    fade_frames = FADE * FPS

    def to_px(x):
        px = CENTRE[0] + RADIUS * x[:, 0]
        py = CENTRE[1] - RADIUS * x[:, 1]
        return np.round(np.c_[px, py] * SS * (1 << SHIFT)).astype(np.int64)

    def render(frame):
        t = frame / (FRAME_COUNT - 1)
        seconds = t * DURATION
        x = motion.at(seconds)
        count = crossings(x, edges)
        total = int(count.sum()) // 2

        # Edge colours.
        heat = np.log1p(count) / np.log1p(reference)
        colours = heat_lut[np.clip((heat * 255).astype(int), 0, 255)]
        free = count == 0
        colours[free] = calm[free]
        pinned = smoothstep(
            (seconds - motion.pin_at.max() - PIN_GLIDE) / 0.4
        ) * rim_edge.astype(np.float32)
        colours = colours + pinned[:, None] * (rim_colour - colours)
        ripple = np.exp(-(((seconds - ZERO_AT - 0.07 * edge_hops) / 0.18) ** 2))
        colours = colours + 0.85 * ripple[:, None] * (255.0 - colours)
        # Long edges of the tangle are fainter, so the hairball reads as a texture.
        p = x[edges]
        length = np.hypot(*(p[:, 1] - p[:, 0]).T)
        dim = np.clip(0.09 / np.maximum(length, 1e-6), 0.4, 1.0) ** 0.8
        dim = np.where(free, 1.0, dim)
        colours = colours * dim[:, None]

        canvas = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
        px = to_px(x)
        order = np.argsort(count, kind="stable")  # the most crossed on top
        for e in order:
            a, b = edges[e]
            thick = 3 if rim_edge[e] else (2 if free[e] else 1)
            cv2.line(
                canvas,
                tuple(px[a]),
                tuple(px[b]),
                tuple(float(c) for c in colours[e]),
                thick,
                cv2.LINE_AA,
                SHIFT,
            )
        # Nodes: small beads, the pinned rim larger.
        settle = smoothstep((seconds - RELAX_START) / 3.0)
        bead = (0.55 + 0.45 * settle) * silk
        for i in range(n):
            radius = (5 if i < RIM else 3) * (1 << SHIFT)
            cv2.circle(
                canvas,
                tuple(px[i]),
                radius,
                tuple(float(c) for c in bead),
                -1,
                cv2.LINE_AA,
                SHIFT,
            )
        lines = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        lines = lines.astype(np.float32)
        glow = cv2.GaussianBlur(lines, (0, 0), 7.0)
        image = background + lines + 0.7 * glow

        # The counter, and the closing line.
        alpha = text.counter(total)
        blend(image, alpha, COUNTER_Y - 150, text_colour, 0.92)
        show = smoothstep((seconds - ZERO_AT - 0.8) / 1.0)
        if show > 0:
            blend(image, text.line(), NOTE_Y - 40, text_colour, 0.8 * show)

        fade = min(1.0, (FRAME_COUNT - 1 - frame) / fade_frames)
        return np.clip(image * fade, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)

    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    totals = crossing_curve(motion, edges)
    track = soundtrack(totals, motion, edges, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


# --- Sound -----------------------------------------------------------------------


def crossing_curve(motion, edges, rate=60):
    """(seconds, total crossings) sampled at `rate` Hz, the same counts the frames show."""
    seconds = np.arange(int(DURATION * rate) + 1) / rate

    def total(s):
        return int(crossings(motion.at(s), edges).sum()) // 2

    with ThreadPoolExecutor(max_workers=16) as pool:
        totals = np.array(list(pool.map(total, seconds)))
    return seconds, totals


def halvings(curve, motion, edges):
    """
    (seconds, x) each time the crossing count first falls below start / 2^k, k = 1, 2,
    ..., and finally to 0; x (-1..1) is where the remaining crossings are, on average.
    """
    seconds, totals = curve
    start = totals[0]
    events, k = [], 1
    for s, total in zip(seconds, totals):
        if s < RELAX_START:
            continue
        while k <= np.log2(start) + 1 and total < start / 2**k:
            x = motion.at(s)
            count = crossings(x, edges)
            if count.sum():
                mid = 0.5 * (x[edges[:, 0], 0] + x[edges[:, 1], 0])
                where = float(np.average(mid, weights=count))
            else:
                where = 0.0
            events.append((s, where))
            k += 1
        if total == 0:
            break
    return events


def soundtrack(curve, motion, edges, style_name, rng):
    """
    A one-shot, Rise & resolve, computed from the crossing counts the frames show.

    While the rim nodes are pinned to the circle, every eighth one lands with a soft
    note of the style's voice, rising round the ring and panned to where it lands.
    Then each halving of the crossing count strikes the voice one scale step higher
    (from two steps under the root to two octaves above it), louder as it climbs,
    panned to where the remaining crossings are. Under it all an augmented
    (whole-tone) pad, fading as the crossings go, and a riser into ZERO_AT. When the
    last crossing goes, the home chord lands (strings and a sung "o" over the bass)
    and the voice rolls up the chord in time with the ripple; it rings to the end.
    """
    style = STYLES[style_name]
    root = int(rng.choice(style.roots))
    scale = Scale(root, style.mode)
    home = Scale(root, "major")
    kw = dict(circular=False)
    track = audio.silence(DURATION)
    hz = lambda midi: float(audio.midi_to_hz(midi))
    video = (FRAME_COUNT - 1) / FPS / DURATION  # model seconds -> video seconds
    zero = ZERO_AT * video

    # Pinning the rim.
    for j in range(0, RIM, 8):
        when = (motion.pin_at[j] + 0.8 * PIN_GLIDE) * video
        note = style.voice(scale.hz(j // 8 - 3), 0.32)
        audio.add(track, audio.pan(note, 0.75 * motion.rim[j, 0]), when, **kw)
    audio.add(track, style.bass(scale.hz(0, octave=-1), 0.35), 0.05, **kw)

    # The tension, thinning out as the crossings go.
    tension = Scale(root, "whole_tone").chord(0, 3)
    for i, (begin, end) in enumerate(
        ((0.0, zero), (0.0, 0.7 * zero), (0.0, 0.45 * zero))
    ):
        for note in tension:
            voice = audio.pad(
                hz(note + 12 * (i == 2)), end - begin, 0.09, 1.2, 1.4, 0.35
            )
            audio.add(track, voice, begin, **kw)
    audio.add(track, audio.riser(2.2, rng, 180.0, 1600.0, 0.22), zero - 2.2, **kw)

    # Every halving, one step up.
    events = halvings(curve, motion, edges)[:-1]  # the last (zero) gets the chord
    last = -np.inf
    for k, (when, x) in enumerate(events):
        when *= video
        if when - last < 0.1 or zero - when < 0.12:
            continue
        degree = int(round(-2 + 12 * k / max(1, len(events) - 1)))
        velocity = 0.5 + 0.4 * k / max(1, len(events) - 1)
        audio.add(
            track,
            audio.pan(style.voice(scale.hz(degree), velocity), 0.7 * x),
            when,
            **kw,
        )
        if k % 6 == 0:
            audio.add(track, style.bass(scale.hz(0, octave=-1), 0.3), when, **kw)
        last = when

    # Zero crossings: the home chord, and the voice rolling up it with the ripple.
    ring = DURATION - zero - 0.3
    for note in home.chord(0, 3):
        audio.add(
            track,
            audio.strings(hz(note), ring - 1.2, 0.35, 0.05, 1.6, 0.45),
            zero,
            **kw,
        )
        audio.add(
            track,
            audio.choir(hz(note + 12), ring - 1.2, 0.25, 0.3, 1.6, "o"),
            zero,
            **kw,
        )
    audio.add(
        track, audio.pad(hz(root - 12), ring - 1.5, 0.15, 0.05, 1.8, 0.3), zero, **kw
    )
    audio.add(track, style.bass(home.hz(0, octave=-1), 0.6), zero, **kw)
    for i, note in enumerate(home.chord(0, 3, octave=1) + [home.note(0, 2)]):
        roll = style.voice(hz(note), 0.55 - 0.06 * i)
        audio.add(track, audio.pan(roll, 0.3 * (-1) ** i), zero + 0.14 * (i + 1), **kw)

    logger.info(f"Soundtrack: {style_name}, root MIDI {root}, {len(events)} halvings")
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, FADE)
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
