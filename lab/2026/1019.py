"""
One Way Through

A random maze, and a spark of light at its centre. The light floods every
corridor at once, forking at each junction, until the whole maze glows like
an aurora. Then the colours fade and one path lights up: the only way from
the last corner the light reached back to the start.

This maze is a spanning tree: a network of corridors that reaches every
cell with no loops at all. So between any two points there is exactly one
path, never a choice. It was grown with Wilson's algorithm, which builds it
out of random walks with their loops erased, and it is perfectly random:
every possible maze on this grid was equally likely.

The colour bands count steps along the corridors, like tree rings. The
last cell the light reaches is 133 cells from the centre as the crow
flies, but the only way there is 1,110 steps long: more than eight times
the straight line, a wandering trail as jagged at every scale.

Listen: each time the light splits into two big branches, a marimba bar
is struck, lower for bigger branches, over a soft patter of dead ends being
reached. The one path plays back as a rising run that lands on the home
chord.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#spanningtree #graphtheory #maze #mathart #randomplots
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
    hex_to_bgr,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- Maze ------------------------------------------------------------------------
PITCH = 8  # pixels per cell; 1080 and 1920 are whole numbers of cells
CORRIDOR = 5  # corridor width in pixels; the rest of the pitch is wall

# --- Timeline (seconds) ---------------------------------------------------------
FLOOD = (0.15, 10.0)  # the fire runs from the centre to the farthest cell
EMBERS = (10.0, 11.4)  # the maze dims to embers
PATH = (10.3, 12.9)  # the one path lights from the last cell back to the centre
FLOOD_EASE = 0.5  # 0: lit area eases in and out; 1: it grows linearly

# --- Look ------------------------------------------------------------------------
# name: (cyclic bands by distance, unlit corridors, the front, the one path)
PALETTES = {
    "ember": (
        ["#ffb347", "#ef6a3a", "#c22d4a", "#6d2470", "#2f2a6e", "#ffb347"],
        "#15161d",
        "#fff3d6",
        "#ffe7a8",
    ),
    "glacier": (
        ["#e8f6ff", "#8fd3f4", "#3a8fc9", "#24467f", "#151f45", "#e8f6ff"],
        "#12151c",
        "#f4fbff",
        "#ffffff",
    ),
    "abyss": (
        ["#c8ff6e", "#3fe0a0", "#14a3a8", "#1b5c8c", "#1a2b5e", "#c8ff6e"],
        "#0f1418",
        "#f2ffe0",
        "#e9ff9a",
    ),
    "neon": (
        ["#ff3cac", "#b03cff", "#5a46ff", "#2bd2ff", "#2bffcf", "#ff3cac"],
        "#14121c",
        "#ffffff",
        "#ffffff",
    ),
    "gilt": (
        ["#f6e3a1", "#d4a84b", "#8c6424", "#3e2a14", "#8c6424", "#f6e3a1"],
        "#141210",
        "#fffaf0",
        "#fff6d8",
    ),
    "aurora": (
        ["#7dffb0", "#2fd6c3", "#3a7bd5", "#8a4fd8", "#e85fb0", "#7dffb0"],
        "#11141a",
        "#f0fff7",
        "#f7f0ff",
    ),
}
PALETTE = "aurora"
BAND_STEPS = 90.0  # corridor steps per full colour cycle
FRONT_STEPS = 18.0  # length of the white-hot front, in steps
EMBER_LEVEL = 0.22  # brightness of the maze once it has burnt down
GLOW_SCALE = 4  # the glow layer is computed at 1/4 resolution
GLOW_SIGMA = 5.0  # in glow-layer pixels
GLOW = 0.9

# --- Sound -----------------------------------------------------------------------
STYLE = (
    "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
SPLIT_MIN = 150  # cells: a split sounds when its smaller branch is at least this big
SPLIT_GAP = 0.16  # s, the closest two split notes
SPLIT_SPAN = 7  # scale degrees from the biggest split (lowest) to the smallest
RAIN = 9.0  # dead-end ticks per second at the peak
PATH_NOTES = 10  # notes in the run along the one path


@njit(cache=True)
def wilson(cols, rows, root, order, seed):
    """
    Uniform spanning tree of the cols x rows grid by Wilson's algorithm:
    from each cell not yet in the tree (in `order`), walk at random until the
    tree is hit, remembering only the last exit from every cell (which erases
    the loops), then add the walk's path. Returns parent[cell] (root: -1).
    """
    np.random.seed(seed)
    n = cols * rows
    in_tree = np.zeros(n, np.bool_)
    parent = np.full(n, -1, np.int64)
    exit_to = np.full(n, -1, np.int64)
    in_tree[root] = True
    dx = np.array([1, -1, 0, 0])
    dy = np.array([0, 0, 1, -1])
    for start in order:
        u = start
        while not in_tree[u]:
            x, y = u % cols, u // cols
            while True:
                k = np.random.randint(4)
                nx, ny = x + dx[k], y + dy[k]
                if 0 <= nx < cols and 0 <= ny < rows:
                    break
            exit_to[u] = ny * cols + nx
            u = exit_to[u]
        u = start
        while not in_tree[u]:
            in_tree[u] = True
            parent[u] = exit_to[u]
            u = exit_to[u]
    return parent


@njit(cache=True)
def depths(parent):
    """Steps from every cell to the root along the tree."""
    n = len(parent)
    depth = np.full(n, -1, np.int64)
    stack = np.empty(n, np.int64)
    for c in range(n):
        if parent[c] == -1:
            depth[c] = 0
    for c in range(n):
        size, u = 0, c
        while depth[u] < 0:
            stack[size] = u
            size += 1
            u = parent[u]
        d = depth[u]
        for i in range(size - 1, -1, -1):
            d += 1
            depth[stack[i]] = d
    return depth


def pixel_distance(parent, depth, cols, cells=None):
    """
    Distance along the maze, in steps, of every pixel (inf on walls). Each
    corridor between a cell and its parent is a CORRIDOR-wide strip from the
    far side of the parent's block to the far side of the cell's, with
    distance running linearly from depth(parent) to depth(cell); where strips
    overlap, the smaller distance wins. With `cells`, only the strips into
    those cells are drawn.
    """
    offset = (PITCH - CORRIDOR) // 2
    children = np.flatnonzero(parent >= 0) if cells is None else np.asarray(cells)
    children = children[parent[children] >= 0]
    px, py = parent[children] % cols, parent[children] // cols
    cx, cy = children % cols, children // cols
    flat = np.full(HEIGHT * WIDTH, np.inf, np.float32)
    along = np.arange(PITCH + CORRIDOR)[None, :, None]  # (1, L, 1)
    across = np.arange(CORRIDOR)[None, None, :]  # (1, 1, C)
    for horizontal in (True, False):
        pick = (cy == py) if horizontal else (cx == px)
        p, c = (px, cx) if horizontal else (py, cy)
        p, c = p[pick][:, None, None], c[pick][:, None, None]
        side = ((py if horizontal else px)[pick] * PITCH + offset)[:, None, None]
        run = np.minimum(p, c) * PITCH + offset + along  # pixels along the strip
        # steps from the parent's centre: depth grows by |s| either way from it
        s = np.abs(run + 0.5 - (p * PITCH + offset + CORRIDOR / 2)) / PITCH
        value = depth[parent[children[pick]]][:, None, None] + s
        x, y = (run, side + across) if horizontal else (side + across, run)
        x, y, value = np.broadcast_arrays(x, y, value)
        np.minimum.at(flat, (y * WIDTH + x).ravel(), value.ravel().astype(np.float32))
    return flat.reshape(HEIGHT, WIDTH)


def flood_events(parent, depth, cols, front, path):
    """
    The soundtrack's events, from the tree and the same front D(t) as the
    frames: when the front reaches each major split (a cell whose second
    biggest child subtree has at least SPLIT_MIN cells) and each dead end,
    how many cells sit right behind the front over time, and the times and
    places of PATH_NOTES evenly spaced points along the one path as the
    second front runs it back to the root.
    """
    n = len(parent)
    grid = np.arange(0.0, DURATION, 1e-3)
    reach = np.maximum.accumulate([front(s) for s in grid])

    def when(d):
        return grid[np.minimum(np.searchsorted(reach, d), len(grid) - 1)]

    size = np.ones(n, np.int64)
    for c in np.argsort(-depth, kind="stable"):
        if parent[c] >= 0:
            size[parent[c]] += size[c]
    first, second = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for c in np.flatnonzero(parent >= 0):
        p, s = parent[c], size[c]
        if s > first[p]:
            first[p], second[p] = s, first[p]
        elif s > second[p]:
            second[p] = s
    splits = np.flatnonzero(second >= SPLIT_MIN)
    leaves = np.flatnonzero(size == 1)

    sorted_depth = np.sort(depth)
    times = np.arange(0.0, DURATION, 0.01)
    reached = np.array([front(s) for s in times])
    frontier = np.searchsorted(sorted_depth, reached, "right") - np.searchsorted(
        sorted_depth, reached - FRONT_STEPS, "right"
    )

    # the second front: S = max_depth (1 - smoothstep(v)) - 2 over PATH
    max_depth = depth.max()
    u = np.arange(1, PATH_NOTES + 1) / PATH_NOTES
    xs = np.linspace(0, 1, 2001)
    v = np.interp(u, smoothstep(xs), xs)
    along = np.clip(np.round(u * max_depth).astype(int), 0, len(path) - 1)
    path_cells = np.array(path)[along]

    def screen_x(cells):
        return 2 * (cells % cols + 0.5) / cols - 1

    return dict(
        split_time=when(depth[splits]),
        split_x=screen_x(splits),
        split_size=second[splits],
        leaf_time=when(depth[leaves]),
        frontier_time=times,
        frontier=frontier,
        path_time=PATH[0] + (PATH[1] - PATH[0]) * v,
        path_x=screen_x(path_cells),
    )


def soundtrack(events, style_name, rng):
    """
    A one-shot following the flood.

    - The spark: the root note over a low root.
    - Major splits: one note each, panned to where the split is. Bigger
      branches (log of the smaller branch's size) sound lower and louder;
      the biggest are kept first and no two notes come closer than SPLIT_GAP.
    - Dead ends: soft noise ticks, their rate following how many dead ends
      the front reaches (up to RAIN per second).
    - A pad on the home chord whose loudness follows the number of cells just
      behind the front, fading with the embers.
    - The one path: a run of PATH_NOTES rising notes as the second front
      sweeps back to the centre, panned along it, landing on the home chord
      (low root, rolled chord, pad) as it reaches the centre, which rings out.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    lift = (
        1 if scale.root < 66 else 0
    )  # spark and landing chord: an octave up, below MIDI ~84
    track = audio.silence(DURATION)
    n = len(track)
    kw = dict(circular=False)

    # The spark.
    audio.add(track, style.voice(scale.hz(0, lift), 0.5), FLOOD[0], **kw)
    audio.add(track, style.bass(scale.hz(0, -2), 0.5, length=3.0), FLOOD[0], **kw)

    # Major splits, the biggest first.
    t, x, size = events["split_time"], events["split_x"], events["split_size"]
    big = np.log(size / SPLIT_MIN) / np.log(max(size.max() / SPLIT_MIN, 1.0001))
    kept = []
    for i in np.argsort(-size, kind="stable"):
        if t[i] < FLOOD[1] and all(abs(t[i] - t[j]) >= SPLIT_GAP for j in kept):
            kept.append(i)
    for i in kept:
        degree = int(round((1 - big[i]) * SPLIT_SPAN))
        note = style.voice(scale.hz(degree), 0.22 + 0.45 * big[i])
        audio.add(track, audio.pan(note, 0.8 * x[i]), t[i], **kw)

    # Dead ends: a patter of ticks.
    step = 0.05
    counts, edges = np.histogram(
        events["leaf_time"], bins=np.arange(0, DURATION + step, step)
    )
    rate = RAIN * counts / max(counts.max(), 1)
    ticks = 0
    for k, r in enumerate(rate):
        for when in edges[k] + step * rng.uniform(size=rng.poisson(r * step)):
            tick = audio.tick(rng, rng.uniform(0.04, 0.08), 0.02, 7000.0)
            audio.add(track, audio.pan(tick, rng.uniform(-0.6, 0.6)), when, **kw)
            ticks += 1

    # A pad that swells with the front.
    fading = smoothstep((events["frontier_time"] - EMBERS[0]) / (EMBERS[1] - EMBERS[0]))
    level = np.sqrt(events["frontier"] / events["frontier"].max()) * (1 - fading)
    seconds = np.arange(n) / audio.SAMPLE_RATE
    envelope = np.interp(seconds, events["frontier_time"], level)[:, None]
    swell = audio.silence(DURATION)
    for note in scale.chord(0, 3, octave=-1):
        f = float(audio.midi_to_hz(note))
        audio.add(swell, audio.pad(f, DURATION, 0.45, 0.3, 0.5, 0.35), 0.0, **kw)
    track += envelope * swell

    # The one path, rising to the home chord.
    for k, (when, px) in enumerate(zip(events["path_time"], events["path_x"])):
        degree = k - PATH_NOTES // 2
        note = style.voice(scale.hz(degree), 0.3 + 0.25 * k / PATH_NOTES)
        audio.add(track, audio.pan(note, 0.7 * px), when, **kw)
    home = PATH[1]
    audio.add(track, style.bass(scale.hz(0, -2), 0.7, length=4.0), home, **kw)
    for i, note in enumerate(scale.chord(0, 3, octave=lift)):
        voice = style.voice(float(audio.midi_to_hz(note)), 0.45)
        audio.add(track, audio.pan(voice, 0.5 * (i - 1)), home + 0.05 * i, **kw)
    for note in scale.chord(0, 3, octave=-1):
        f = float(audio.midi_to_hz(note))
        audio.add(track, audio.pad(f, DURATION - home, 0.4, 0.4, 1.5, 0.3), home, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {len(kept)} split notes "
        f"of {len(t)} splits, {ticks} ticks"
    )
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.6)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A uniform spanning tree of a 135x240 cell grid, flooded from the centre.

    Wilson's algorithm (loop-erased random walks, numba) builds the tree
    rooted at the centre cell; the tree depth of each cell is its distance to
    the centre through the maze. Each corridor strip between a cell and its
    parent is rasterised once with a linearly interpolated distance, giving
    a float distance map over the pixels (walls inf).

    Every frame is a closed form in t. The front sits at distance
    D(t), chosen so the lit fraction of the corridors grows along a curve
    between linear and smoothstep (FLOOD_EASE) over FLOOD; pixels
    with distance <= D take a cyclic ember colour by distance (BAND_STEPS per
    cycle), blended toward white by exp(-(D - d) / FRONT_STEPS) near the
    front. Over EMBERS the lit maze dims to EMBER_LEVEL, and over PATH a second
    front runs from the deepest cell back to the root along its unique path,
    which is drawn in the palette's path colour. Hot pixels feed a quarter-resolution
    Gaussian glow added on top. The seed picks the maze.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    cols, rows = WIDTH // PITCH, HEIGHT // PITCH
    root = (rows // 2) * cols + cols // 2
    order = rng.permutation(cols * rows)
    parent = wilson(cols, rows, root, order, int(rng.integers(2**31)))
    depth = depths(parent)
    deepest = int(np.argmax(depth))
    path = [deepest]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    max_depth = float(depth.max())
    far = np.hypot(
        (deepest % cols - root % cols) * PITCH, (deepest // cols - root // cols) * PITCH
    )
    logger.info(
        f"Maze {cols}x{rows}, max depth {max_depth:.0f} steps; the deepest cell is "
        f"{far / PITCH:.0f} cells from the centre ({max_depth * PITCH / far:.1f}x)"
    )

    distance = pixel_distance(parent, depth, cols)
    on_path = np.isfinite(pixel_distance(parent, depth, cols, path))
    corridor = np.isfinite(distance)
    distance = np.where(corridor, np.maximum(distance, 0.0), np.inf).astype(np.float32)

    # t-independent colours (BGR float, 0..1)
    bands, unlit, hot, path_colour = PALETTES[PALETTE]
    lut = palette_lut(bands, 1024)[:, ::-1]
    band = (np.where(corridor, distance, 0.0) / BAND_STEPS) % 1.0
    lit_colour = lut[(band * 1023).astype(np.int32)] * corridor[..., None]
    unlit, hot, path_colour = (
        np.array(hex_to_bgr(c), np.float32) / 255 for c in (unlit, hot, path_colour)
    )
    glow_colour = lut[0] * 0.6 + hot * 0.4
    corridor_bgr = corridor[..., None].astype(np.float32)

    # The front is paced by area: the lit fraction of the corridors follows an
    # eased curve, and D(t) is that quantile of the corridor distances.
    ranked = np.sort(distance[corridor])

    def front(seconds):
        u = np.clip((seconds - FLOOD[0]) / (FLOOD[1] - FLOOD[0]), 0, 1)
        area = FLOOD_EASE * u + (1 - FLOOD_EASE) * smoothstep(u)
        return float(ranked[min(int(area * len(ranked)), len(ranked) - 1)])

    def render(frame):
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        D = front(seconds) + 3 * FRONT_STEPS * max(seconds - FLOOD[1], 0)  # cools
        behind = D - distance  # >= 0 where lit
        lit = behind >= 0
        heat = np.where(lit, np.exp(-np.maximum(behind, 0) / FRONT_STEPS), 0.0)
        ember = 1 - (1 - EMBER_LEVEL) * smoothstep(
            (seconds - EMBERS[0]) / (EMBERS[1] - EMBERS[0])
        )
        image = unlit * corridor_bgr * (1 - lit[..., None])
        image = image + lit[..., None] * ember * lit_colour
        image = image + heat[..., None] * (hot - image)

        # the one path, lit from the deepest cell back to the root
        u = smoothstep((seconds - PATH[0]) / (PATH[1] - PATH[0]))
        if u > 0:
            S = max_depth * (1 - u) - 2.0
            ahead = distance - S  # >= 0 where the path is lit
            path_lit = on_path & (ahead >= 0)
            path_heat = np.where(path_lit, np.exp(-np.maximum(ahead, 0) / 6.0), 0)
            path_mix = path_lit[..., None] * (path_colour + 0.6 * path_heat[..., None])
            image = np.where(path_lit[..., None], np.minimum(path_mix, 1.2), image)
            heat = np.maximum(heat, path_lit * (0.35 + 0.65 * path_heat))

        small = cv2.resize(
            heat.astype(np.float32),
            (WIDTH // GLOW_SCALE, HEIGHT // GLOW_SCALE),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(small, (0, 0), GLOW_SIGMA)
        glow = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        image = image + GLOW * glow[..., None] * glow_colour
        return 255.0 * image

    render_frames(settings, render, FRAME_COUNT)
    tag = settings.audio_tag if hasattr(settings, "audio_tag") else ""
    style_name = tag if tag in STYLES else STYLE
    events = flood_events(parent, depth, cols, front, path)
    track = soundtrack(events, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
