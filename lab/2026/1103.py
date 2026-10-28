"""
Every Road Leads to 1

Pick any whole number. If it is even, halve it. If it is odd, triple it and
add one. Repeat. Start from 7 and you go 22, 11, 34, 17, 52, 26, 13, 40,
20, 10, 5, 16, 8, 4, 2, 1. Start from 27 and you climb as high as 9232
before falling, 111 steps later, to the same 1.

Here are the roads of every number up to 30,000, drawn backwards from where
they all end. Each road bends a little one way when it halves and the
other way when it triples, and since every road that reaches 1 shares its
last stretch with others, they merge into a single plant, rooted at 1.

Every number ever tested, up to about 2^71, ends at 1. Nobody has proved
that all of them do. It is the Collatz conjecture, posed in 1937, and Paul
Erdős said mathematics is not yet ready for problems like it.

Listen as it grows: a bell for each new layer of roads, higher the wider
the plant has spread, over strings that swell as it fills in, and the
home chord the moment the last road arrives.

Made with Python & OpenCV. Sound synthesized in NumPy.
#collatz #numbertheory #unsolvedproblems #mathart #randomplots
"""

import gc

import io

import cv2
import numpy as np
from loguru import logger
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties

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

DURATION = 18.0
FRAME_COUNT = frame_count(DURATION)

# --- The tree -----------------------------------------------------------------------
LIMIT = 30000  # every starting number up to this
EVEN_TURN, ODD_TURN = 8.65, -16.0  # degrees per halving / per tripling step

# --- Motion -------------------------------------------------------------------------
GROW = (0.8, 11.0)  # s: the growth front runs from STALK to the deepest node
GROW_EASE = 2.0  # > 1 lingers on the first, labelled steps
STALK = 4  # depth shown from the start: 1, 2, 4, 8, 16
SWAY = 0.12  # degrees per step: a wave travelling up every strand
SWAY_WAVE, SWAY_PERIOD = 0.045, 6.0  # rad per step, s
SETTLE = (14.0, 17.5)  # s: the sway fades to a stop

# --- Camera -------------------------------------------------------------------------
MARGIN = 0.86  # of the frame the tree may fill
MAX_ZOOM = 130.0  # px per step, at the start
CAMERA_SMOOTH = 0.6  # s

# --- Look ---------------------------------------------------------------------------
PALETTE = ["#2a9aa0", "#3cc4a4", "#9be8b8", "#f4f8c8"]  # root to tips
LINE_LEVEL = 0.9
LINE_SCALE = 0.08  # line width at 2x, px per px-per-step (1 to 8)
LABEL_ZOOM = (14.0, 30.0)  # px per step: labels fade in over this range
LABEL_HEIGHT, LABEL_MAX = 0.2, 26.0  # px of a digit per px-per-step, and at most
LABEL_BASE = 96.0  # pt: font size the LaTeX labels are typeset at
LABEL_CROWD = (30, 90)  # labelled nodes: labels fade out over this range
TIP_LEVEL = 1.0
BLOOM_SHRINK, BLOOM_SIGMA, BLOOM_GAIN = 4, 4.0, 0.6
COLOUR_BINS = 48

# --- Sound -------------------------------------------------------------------------
STYLE = "gamelan"  # common.music.STYLES; an --audio-only tag naming one overrides it
NOTE_GAP = 0.22  # s: closest two layer notes may come


def collatz_tree(limit: int):
    """
    The union of the Collatz roads of 1..limit, as a tree rooted at 1.
    Returns (values, parent, depth); nodes are ordered so a parent always
    comes before its children, and parent[0] = -1 for the root.
    """
    index = {1: 0}
    values, parent = [1], [-1]
    for n in range(2, limit + 1):
        path, v = [], n
        while v not in index:
            path.append(v)
            v = v // 2 if v % 2 == 0 else 3 * v + 1
        p = index[v]
        for u in reversed(path):
            index[u] = len(values)
            values.append(u)
            parent.append(p)
            p = index[u]
    values, parent = np.array(values, np.int64), np.array(parent)
    depth = np.zeros(len(values), np.int64)
    for i in range(1, len(values)):
        depth[i] = depth[parent[i]] + 1
    return values, parent, depth


def latex_sprite(value: int) -> np.ndarray:
    """The number typeset in Computer Modern ($n$) at LABEL_BASE pt, as a float mask."""
    buffer = io.BytesIO()
    prop = FontProperties(size=LABEL_BASE, math_fontfamily="cm")
    mathtext.math_to_image(
        f"${value}$", buffer, prop=prop, dpi=100 / 0.72, format="png"
    )
    grey = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    mask = 1 - grey.astype(np.float32) / 255  # black ink on white
    rows = np.flatnonzero(mask.max(1) > 0)
    cols = np.flatnonzero(mask.max(0) > 0)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def front(t):
    """Depth of the growth front at time t (fractional: the last edge grows)."""
    s = smoothstep(
        np.clip((np.asarray(t, float) - GROW[0]) / (GROW[1] - GROW[0]), 0, 1)
    )
    return STALK + s**GROW_EASE * (front.deepest - STALK)


def soundtrack(
    style_name: str, rng: np.random.Generator, counts: np.ndarray, pans: np.ndarray
) -> np.ndarray:
    """
    A one-shot that grows with the tree (counts[d]: nodes at depth d;
    pans[d]: where they sit across the frame, -1 to 1).

    - The layers: each time the growth front reaches a new depth, the
      style's voice strikes a note, pitched by how wide the tree has become
      there (two scale degrees per doubling of the layer's node count) and
      panned with the layer. The stalk on screen from the start (1, 2, 4, 8,
      16) plays as a rising figure; after it every layer sounds until
      they come faster than NOTE_GAP, then only the next one after the gap.
    - The swell: a string chord under the growth, louder as the tree fills
      in (the share of nodes already drawn).
    - The root: when the last road arrives, the home chord, struck from the
      bottom up over a low bass, with strings holding it while the plant
      settles; two soft notes fall during the hold.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    sr = audio.SAMPLE_RATE
    kw = dict(circular=False)
    track = audio.silence(DURATION)

    grid = np.linspace(0, DURATION, int(DURATION * 1000) + 1)
    reach = front(grid)
    last, notes = -np.inf, 0
    for d in range(len(counts)):
        if d <= STALK:  # the opening stalk, 1 2 4 8 16, rising one degree a node
            when, degree = 0.1 + 0.18 * d, d
        else:
            crossed = np.flatnonzero(reach >= d)
            if not len(crossed):
                break
            when = float(grid[crossed[0]])
            if when - last < NOTE_GAP or when > GROW[1] - 0.3:
                continue
            degree = int(round(2 * np.log2(counts[d])))
        velocity = 0.5 if when < 4.5 else 0.32
        note = style.voice(scale.hz(degree % 7, degree // 7), velocity)
        audio.add(track, audio.pan(note, 0.6 * pans[d]), when, **kw)
        last, notes = when, notes + 1

    # The swell, shaped by the share of the tree already drawn.
    drawn = np.cumsum(counts) / counts.sum()
    share = drawn[
        np.clip(
            np.floor(front(np.arange(len(track)) / sr)).astype(int), 0, len(counts) - 1
        )
    ]
    swell = audio.silence(DURATION)
    start = 0.5
    for n in scale.chord(0, 4, octave=-1):
        hz = float(audio.midi_to_hz(n))
        audio.add(
            swell, audio.strings(hz, GROW[1] - start, 0.22, 2.0, 1.0, 0.35), start, **kw
        )
    track += swell * (0.15 + 0.85 * share[:, None] ** 0.7)

    home = GROW[1]
    for k, n in enumerate(scale.chord(0, 4)):
        note = style.voice(float(audio.midi_to_hz(n)), 0.5)
        audio.add(track, audio.pan(note, 0.25 * (k - 1.5)), home + 0.09 * k, **kw)
    bass = scale.hz(0, -2)
    bass = 2 * bass if bass < 65 else bass  # phone speakers stop around 60 Hz
    audio.add(track, style.bass(bass, 0.5, length=5.0), home, **kw)
    for n in scale.chord(0, 3, octave=-1):
        hz = float(audio.midi_to_hz(n))
        audio.add(
            track,
            audio.strings(hz, DURATION - home - 2.5, 0.2, 0.6, 2.5, 0.3),
            home - 0.2,
            **kw,
        )
    for when, degree in ((13.6, 4), (15.4, 2)):
        audio.add(
            track, audio.pan(style.voice(scale.hz(degree), 0.25), 0.3), when, **kw
        )

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} layer notes"
    )
    track = style.room(track, rng, circular=False)
    return audio.master(audio.fade(track, 0.0, 1.5), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The Collatz tree: every number up to LIMIT, joined to its successor
    (n/2 or 3n+1), is a tree rooted at 1 (`collatz_tree`). Drawn from the
    root out, after Edmund Harriss: each node is one unit step from its
    successor, turned EVEN_TURN degrees if it is even and ODD_TURN if odd;
    since odd steps are always followed by even ones the turns nearly
    balance, so strands curl gently instead of spiralling. The whole tree is
    rotated so the root-to-centroid direction points up.

    A frame is a pure function of t: the growth front `front(t)` eases from
    depth STALK to the deepest node over GROW, drawing every edge at
    shallower depth and the front edges partway, with a bright dot at their
    tips. A sway wave travels up all strands: the turn at depth d gets
    SWAY sin(2π t / SWAY_PERIOD - SWAY_WAVE d) extra degrees, fading out
    over SETTLE; since it depends on depth only, positions are a cumulative
    sum layer by layer. The camera fits the rest-pose bounding box of the
    visible depths (precomputed per depth, smoothed over CAMERA_SMOOTH).
    Lines are drawn with cv2 at 2x in COLOUR_BINS depth groups coloured by
    depth through PALETTE, downsampled with INTER_AREA and bloomed. While
    the camera is close, nodes carry their numbers, typeset in Computer
    Modern with matplotlib mathtext once and scaled per frame; they fade out
    as the zoom drops below LABEL_ZOOM or the count passes LABEL_CROWD. Nothing
    is random: --seed does not change the picture. The posted soundtrack is
    gamelan with --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    values, parent, depth = collatz_tree(LIMIT)
    deepest = int(depth.max())
    front.deepest = deepest
    turn = np.radians(np.where(values % 2 == 0, EVEN_TURN, ODD_TURN))
    turn[0] = 0.0
    layers = [np.flatnonzero(depth == d) for d in range(deepest + 1)]
    logger.info(
        f"{len(values)} nodes, deepest {deepest} (from {values[depth.argmax()]})"
    )

    def pose(sway):
        """Node positions (complex, in steps) for per-depth extra turns `sway`."""
        angle = np.zeros(len(values))
        pos = np.zeros(len(values), complex)
        for d in range(1, deepest + 1):
            nodes = layers[d]
            p = parent[nodes]
            angle[nodes] = angle[p] + turn[nodes] + sway[d]
            pos[nodes] = pos[p] + np.exp(1j * angle[nodes])
        return pos

    # Turn the tree so it fills the portrait frame best, root at the bottom.
    rest = pose(np.zeros(deepest + 1))
    best = 0.0
    for a in np.radians(np.arange(0, 360, 1.0)):
        r = rest * np.exp(1j * a)
        if r.imag.mean() <= 0:
            continue
        fit = min(WIDTH / np.ptp(r.real), HEIGHT / np.ptp(r.imag))
        if fit > best:
            best, rotation = fit, np.exp(1j * a)

    # Camera: bounding box of the visible depths at rest, smoothed in time.
    rest = rest * rotation
    lo = np.full((deepest + 1, 2), np.inf)
    hi = np.full((deepest + 1, 2), -np.inf)
    for d in range(deepest + 1):
        pts = np.stack([rest[layers[d]].real, rest[layers[d]].imag], 1)
        lo[d], hi[d] = pts.min(0), pts.max(0)
    lo, hi = np.minimum.accumulate(lo), np.maximum.accumulate(hi)
    times = np.arange(FRAME_COUNT) / FPS
    d = np.clip(np.ceil(front(times)).astype(int), 0, deepest)
    box_lo, box_hi = lo[d], hi[d]
    span = np.maximum(box_hi - box_lo, 1e-6)
    zoom = np.minimum(
        MAX_ZOOM, np.minimum(MARGIN * WIDTH / span[:, 0], MARGIN * HEIGHT / span[:, 1])
    )
    centre = (box_lo + box_hi) / 2
    kernel = np.exp(
        -0.5 * (np.arange(-3 * FPS, 3 * FPS + 1) / (CAMERA_SMOOTH * FPS)) ** 2
    )
    kernel /= kernel.sum()

    def smooth(x):
        padded = np.pad(x, (len(kernel) // 2,), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    zoom = np.exp(smooth(np.log(zoom)))
    centre = np.stack([smooth(centre[:, 0]), smooth(centre[:, 1])], 1)

    lut = palette_lut(PALETTE, COLOUR_BINS)[:, ::-1] * 255  # BGR
    bins = np.minimum((depth * COLOUR_BINS) // (deepest + 1), COLOUR_BINS - 1)
    edge_bins = [np.flatnonzero((bins == b) & (depth > 0)) for b in range(COLOUR_BINS)]

    # LaTeX labels for every node that can be labelled before they crowd out.
    order = np.argsort(depth, kind="stable")
    shallow = depth <= depth[order[LABEL_CROWD[1]]]
    sprites = {int(v): latex_sprite(int(v)) for v in values[shallow]}
    digit = latex_sprite(8).shape[0]  # px

    def render(frame: int) -> np.ndarray:
        t = frame / FPS
        amplitude = SWAY * (
            1 - smoothstep(np.clip((t - SETTLE[0]) / (SETTLE[1] - SETTLE[0]), 0, 1))
        )
        k = np.arange(deepest + 1)
        sway = np.radians(
            amplitude * np.sin(2 * np.pi * t / SWAY_PERIOD - SWAY_WAVE * k)
        )
        pos = pose(sway) * rotation
        x = WIDTH / 2 + (pos.real - centre[frame, 0]) * zoom[frame]
        y = HEIGHT / 2 - (pos.imag - centre[frame, 1]) * zoom[frame]
        xy = np.round(np.stack([x, y], 1) * 2 * 4).astype(
            np.int32
        )  # 2x, 2 bits subpixel

        # cv2 widths are whole pixels: draw at the next one up, dimmed by the
        # fraction, so the line's weight follows the zoom smoothly.
        exact = float(np.clip(LINE_SCALE * zoom[frame], 1, 8))
        width = int(np.ceil(exact))
        level = LINE_LEVEL * exact / width
        reach = float(front(t))
        whole = int(np.floor(reach))
        canvas = np.zeros((2 * HEIGHT, 2 * WIDTH, 3), np.float32)
        for b, nodes in enumerate(edge_bins):
            nodes = nodes[depth[nodes] <= whole]
            if not len(nodes):
                continue
            segments = np.stack([xy[parent[nodes]], xy[nodes]], 1)
            cv2.polylines(
                canvas,
                list(segments),
                False,
                tuple(float(c) for c in level * lut[b] / 255),
                width,
                cv2.LINE_AA,
                2,
            )

        if whole < deepest:  # the growing edges and their tips
            nodes = layers[whole + 1]
            frac = reach - whole
            start = xy[parent[nodes]]
            tip = np.round(start + frac * (xy[nodes] - start)).astype(np.int32)
            colour = tuple(float(c) for c in level * lut[bins[nodes[0]]] / 255)
            cv2.polylines(
                canvas,
                list(np.stack([start, tip], 1)),
                False,
                colour,
                width,
                cv2.LINE_AA,
                2,
            )
            for p in tip:
                cv2.circle(
                    canvas,
                    tuple(map(int, p)),
                    int(4 * max(2.0, 1.6 * exact)),
                    (TIP_LEVEL,) * 3,
                    -1,
                    cv2.LINE_AA,
                    2,
                )

        image = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        small = cv2.resize(
            image,
            (WIDTH // BLOOM_SHRINK, HEIGHT // BLOOM_SHRINK),
            interpolation=cv2.INTER_AREA,
        )
        bloom = cv2.resize(
            cv2.GaussianBlur(small, (0, 0), BLOOM_SIGMA), (WIDTH, HEIGHT)
        )
        image = image + BLOOM_GAIN * bloom

        # Numbers on the nodes while the camera is close.
        alpha = float(
            np.clip(
                (zoom[frame] - LABEL_ZOOM[0]) / (LABEL_ZOOM[1] - LABEL_ZOOM[0]), 0, 1
            )
        )
        labelled = np.flatnonzero(depth <= whole)
        alpha *= float(
            np.clip(
                (LABEL_CROWD[1] - len(labelled)) / (LABEL_CROWD[1] - LABEL_CROWD[0]),
                0,
                1,
            )
        )
        if alpha > 0:
            scale = min(LABEL_HEIGHT * zoom[frame], LABEL_MAX) / digit
            for i in labelled:
                mask = cv2.resize(
                    sprites[values[i]],
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
                h, w = mask.shape
                left = int(round(x[i] + 0.25 * zoom[frame]))
                top = int(round(y[i] - h / 2))
                if left < 0 or top < 0 or left + w > WIDTH or top + h > HEIGHT:
                    continue
                image[top : top + h, left : left + w] += (0.85 * alpha * mask)[
                    ..., None
                ]
        return np.clip(image, 0, 1) * 255

    render_frames(settings, render, FRAME_COUNT)
    counts = np.bincount(depth)
    middle, half = (lo[-1, 0] + hi[-1, 0]) / 2, (hi[-1, 0] - lo[-1, 0]) / 2
    pans = np.array([(rest[layer].real.mean() - middle) / half for layer in layers])
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(style_name, settings.audio_rng, counts, pans)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
