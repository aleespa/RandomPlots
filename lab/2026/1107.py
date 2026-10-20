"""
Exactly One Half

A grid of dots, and between every pair of neighbours a link that is either open or shut,
decided by a coin weighted to come up "open" with probability p. As p creeps upward,
links open, little islands of connected dots merge into bigger ones, and their colours
warm with their size.

Below one half, every island is finite, however large the grid. Above it, a single
island spans everything. The switch happens at exactly p = ½, a fact Harry Kesten
proved in 1980, and at that moment the islands are ragged fractals at every scale. This
is percolation: the same tipping point decides whether water seeps through coffee,
whether a forest fire jumps from tree to tree, and whether an epidemic fizzles or spreads.

Every time two islands join, a vibraphone strikes: small ones high and soft, big ones
lower and louder, panned to where they meet, over a chord that won't settle. The moment
one island crosses from top to bottom, it resolves.

Made with Python, NumPy, SciPy & OpenCV. Sound synthesized in NumPy.
#percolation #graphtheory #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

CELL = 5  # px per node: a 3 px dot and a 2 px gap that an open link fills
NX, NY = WIDTH // CELL, HEIGHT // CELL  # 216 x 384 nodes

BACKGROUND = "#030508"
# Clusters other than the largest, by size: dust -> islands -> continents.
SIZES = ["#0a141e", "#14304a", "#21557a", "#3683a3", "#7cc3cf"]  # cool only
GIANT = ["#8a1d08", "#ff5a1f", "#ffd166"]  # the largest cluster: ember -> gold

KNOBS = dict(
    p_start=0.40,
    p_end=0.57,
    span_at=9.0,  # s: the largest cluster first joins top and bottom here
    settle_at=14.0,  # s: p reaches p_end and holds
    linger=1.6,  # > 1: how slowly p creeps through the critical point
    glow=0.15,
)

# --- Sound -------------------------------------------------------------------------
STYLE = "vibes"  # common.music.STYLES; an --audio-only tag naming one overrides it
MIN_MERGE = 20  # nodes: merges of smaller clusters stay silent
NOTE_GAP = 0.17  # s: at most one struck note per window, the biggest merge in it
TOP = 9  # scale degree of the smallest audible merge; each doubling steps one down
BIG_MERGE = 2000  # nodes: a merge this big into the largest cluster adds a low note
BASS_GAP = 0.8  # s: shortest gap between bass notes
RISER = 3.0  # s: the build-up into the spanning moment


def clusters(open_h, open_v):
    """Component label of every node and the size of every component."""
    n = NX * NY
    index = np.arange(n).reshape(NY, NX)
    rows = np.concatenate([index[:, :-1][open_h], index[:-1, :][open_v]])
    cols = np.concatenate([index[:, 1:][open_h], index[1:, :][open_v]])
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels, np.bincount(labels)


def spans(labels, sizes) -> bool:
    """Does the largest cluster touch both the top and the bottom row?"""
    biggest = sizes.argmax()
    grid = labels.reshape(NY, NX)
    return bool((grid[0] == biggest).any() and (grid[-1] == biggest).any())


def merges(u_h, u_v, p_lo, p_hi):
    """
    Every merge of two clusters as p sweeps from p_lo to p_hi, in order: links open
    one by one in order of u (union-find), so this is exact. Returns rows of (p, size
    of the smaller cluster, x of its centre in -1..1, 1 if the merge made the new
    largest cluster).
    """
    ys, xs = np.mgrid[0:NY, 0:NX]
    index = ys * NX + xs
    left = np.concatenate([index[:, :-1].ravel(), index[:-1].ravel()])
    right = np.concatenate([index[:, 1:].ravel(), index[1:].ravel()])
    u = np.concatenate([u_h.ravel(), u_v.ravel()])
    order = np.argsort(u)
    order = order[u[order] < p_hi]
    parent = list(range(NX * NY))
    size = [1] * (NX * NY)
    x_sum = [float(x) for x in xs.ravel()]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    events, largest = [], 1
    for k in order:
        a, b = find(int(left[k])), find(int(right[k]))
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        small, x = size[b], x_sum[b] / size[b]
        parent[b] = a
        size[a] += small
        x_sum[a] += x_sum[b]
        record = size[a] > largest
        largest = max(largest, size[a])
        if u[k] >= p_lo and small >= MIN_MERGE:
            events.append((u[k], small, 2 * x / (NX - 1) - 1, record))
    return np.array(events)


def soundtrack(events, style_name, rng):
    """
    A one-shot, Rise & resolve. `events` are the merges with their times (seconds,
    absorbed size, x, record). Each window of NOTE_GAP strikes the style's voice once,
    for its biggest merge: the bigger the absorbed cluster, the lower and louder the
    note (one scale degree per doubling), panned to where it was. A merge of BIG_MERGE
    nodes or more into the largest cluster adds a bass note (BASS_GAP apart). Under them an augmented (whole-tone) pad grows
    louder and brighter towards the spanning moment, with a riser; at span_at it
    resolves onto the home chord in a sung "o" over the root, which rings out.
    """
    knobs = KNOBS
    style = STYLES[style_name]
    root = int(rng.choice(style.roots))
    scale = Scale(root, style.mode)
    home = Scale(root, "major")
    kw = dict(circular=False)
    track = audio.silence(DURATION)
    span = knobs["span_at"]
    hz = lambda midi: float(audio.midi_to_hz(midi))

    # The tension: an augmented triad an octave down, swelling in three layers.
    tension = Scale(root, "whole_tone").chord(0, 3, octave=-1)
    for i, start in enumerate((0.0, 3.0, 6.0)):
        brightness = 0.25 + 0.25 * i
        for note in tension:
            voice = audio.pad(hz(note), span - start, 0.22, 2.5, 0.25, brightness)
            audio.add(track, voice, start, **kw)
    audio.add(track, audio.riser(RISER, rng, 150.0, 1800.0, 0.25), span - RISER, **kw)

    # The resolution: the home chord, sung, with the root below.
    ring = DURATION - span - 0.4
    for note in home.chord(0, 3) + [home.note(0, 1)]:
        audio.add(
            track, audio.choir(hz(note), ring - 1.0, 0.4, 0.15, 1.8, "o"), span, **kw
        )
    audio.add(
        track, audio.pad(hz(root - 12), ring - 1.5, 0.35, 0.1, 2.0, 0.3), span, **kw
    )
    audio.add(track, style.bass(scale.hz(0, octave=-2), 0.8), span, **kw)

    struck, last_bass = 0, -np.inf
    for start in np.arange(0.0, DURATION - 1.0, NOTE_GAP):
        window = events[(events[:, 0] >= start) & (events[:, 0] < start + NOTE_GAP)]
        if not len(window):
            continue
        when, small, x, _ = window[window[:, 1].argmax()]
        if abs(when - span) < 0.12:
            continue  # the chord has the spanning moment to itself
        doublings = np.log2(small / MIN_MERGE)
        degree = TOP - int(round(doublings))
        velocity = min(0.85, 0.3 + 0.06 * doublings)
        note = style.voice(scale.hz(degree), velocity)
        audio.add(track, audio.pan(note, 0.7 * x), when, **kw)
        record = window[:, 3] > 0
        if (window[record, 1] >= BIG_MERGE).any() and when - last_bass >= BASS_GAP:
            audio.add(track, style.bass(scale.hz(0, octave=-2), 0.4), when, **kw)
            last_bass = when
        struck += 1
    logger.info(f"Soundtrack: {style_name}, root MIDI {root}, {struck} struck notes")
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.4)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Bond percolation on a 216 x 384 square lattice, swept through p = 1/2.

    Every link (horizontal and vertical) gets one uniform number u from settings.rng; at
    probability p the open links are those with u < p, so clusters only ever merge as p
    grows. The spanning threshold of this sample, p_span (the smallest p at which the
    largest cluster joins the top and bottom rows), is found by bisection up front, and
    the sweep is built around it: p(s) = p_span - a |s|^linger before, p_span +
    b s^linger after, s = seconds - span_at, from p_start at 0 s to p_end at
    settle_at, then held. p creeps through the critical point and the spanning moment
    lands at span_at exactly.

    A frame is a pure function of p: links open, components from
    scipy.sparse.csgraph.connected_components, then every pixel looks up its element
    (a node, a link or the gap) in a precomputed map. Clusters other than the largest
    are coloured by log size through SIZES (cool colours only); the largest ignites into
    GIANT as it grows from 0.5 % to 3 % of the nodes, ember at its ragged frontier and
    gold where it is dense (a blur of its own mask), with a glow, and flashes at the
    spanning moment.
    Frames render in parallel on the thread pool; the last 0.4 s fade out.

    Sound (see `soundtrack`): the merges come from the same u by union-find, timed by
    inverting p(s). The key comes from settings.audio_rng; the posted soundtrack uses
    --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    knobs = KNOBS
    rng = settings.rng
    u_h = rng.random((NY, NX - 1))
    u_v = rng.random((NY - 1, NX))

    lo, hi = 0.4, 0.6
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if spans(*clusters(u_h < mid, u_v < mid)):
            hi = mid
        else:
            lo = mid
    p_span = hi
    logger.info(f"This lattice first spans at p = {p_span:.5f}")

    before = (p_span - knobs["p_start"]) / knobs["span_at"] ** knobs["linger"]
    after = (knobs["p_end"] - p_span) / (
        knobs["settle_at"] - knobs["span_at"]
    ) ** knobs["linger"]

    def p_at(seconds):
        s = min(seconds, knobs["settle_at"]) - knobs["span_at"]
        if s < 0:
            return p_span - before * (-s) ** knobs["linger"]
        return p_span + after * s ** knobs["linger"]

    # Element map: every pixel is a node (index i), a horizontal link (n + j), a
    # vertical link (n + m_h + k) or the gap (-1).
    n, m_h = NX * NY, NY * (NX - 1)
    element = np.full((HEIGHT, WIDTH), -1, np.int64)
    ys, xs = np.mgrid[0:NY, 0:NX]
    hy, hx = ys[:, :-1], xs[:, :-1]
    vy, vx = ys[:-1], xs[:-1]
    for a in (1, 2, 3):  # across the dot / link
        for b in (1, 2, 3):
            element[ys * CELL + a, xs * CELL + b] = ys * NX + xs
        for g in (4, 5):  # along the gap to the next node
            element[hy * CELL + a, hx * CELL + g] = n + hy * (NX - 1) + hx
            element[vy * CELL + g, vx * CELL + a] = n + m_h + vy * NX + vx
    left_node = np.concatenate(
        [(ys[:, :-1] * NX + xs[:, :-1]).ravel(), (ys[:-1] * NX + xs[:-1]).ravel()]
    )
    u_links = np.concatenate([u_h.ravel(), u_v.ravel()])

    size_lut = palette_lut(SIZES, 256) * 255
    giant_lut = palette_lut(GIANT, 256) * 255
    background = np.array([int(BACKGROUND[i : i + 2], 16) for i in (1, 3, 5)])
    fade_frames = 0.4 * FPS

    def render(frame):
        seconds = frame / (FRAME_COUNT - 1) * DURATION
        p = p_at(seconds)
        open_links = u_links < p
        labels, sizes = clusters(
            open_links[:m_h].reshape(NY, NX - 1), open_links[m_h:].reshape(NY - 1, NX)
        )
        biggest = sizes.argmax()
        level = np.log(sizes[labels]) / np.log(0.02 * n)
        node_rgb = size_lut[np.clip(level * 255, 0, 255).astype(int)]
        giant = labels == biggest
        # The giant's colour: ember at its frontier, gold where it is dense, measured
        # by a blur of its own mask on the node grid.
        mask = giant.reshape(NY, NX).astype(np.float32)
        density = cv2.GaussianBlur(mask, (0, 0), 6.0).ravel()
        heat = np.clip(density[giant] * 1.4, 0, 1)
        # The giant ignites as it grows from 0.5 % to 3 % of the lattice.
        ignite = np.clip((sizes[biggest] / n - 0.005) / 0.025, 0, 1)
        ignite = ignite * ignite * (3 - 2 * ignite)
        fire = giant_lut[(heat * 255).astype(int)]
        node_rgb[giant] = node_rgb[giant] + ignite * (fire - node_rgb[giant])
        link_rgb = np.where(
            open_links[:, None], node_rgb[left_node], background[None, :]
        )
        colours = np.concatenate([node_rgb, link_rgb, background[None, :]])
        image = colours[element].astype(np.float32)
        # Glow of the giant, brighter at the spanning flash.
        flash = np.exp(-(((seconds - knobs["span_at"]) / 0.25) ** 2))
        small = cv2.resize(mask, (NX, NY))
        glow = cv2.GaussianBlur(small, (0, 0), 3.0)
        glow = cv2.resize(glow, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        amount = knobs["glow"] * ignite * (1 + 1.5 * flash)
        image += amount * glow[..., None] * np.array([255, 110, 40], np.float32)
        image *= 1 + 0.35 * flash * (element >= 0)[..., None]
        fade = min(1.0, (FRAME_COUNT - 1 - frame) / fade_frames)
        return np.clip(image * fade, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT, rgb=True)
    events = merges(u_h, u_v, knobs["p_start"], knobs["p_end"])
    grid = np.linspace(0.0, knobs["settle_at"], 4000)
    events[:, 0] = np.interp(events[:, 0], [p_at(s) for s in grid], grid)
    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(events, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
