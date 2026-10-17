"""
The Einstein Tile

One shape, copied over and over, tiles the whole plane without a gap, and the
pattern never repeats. For sixty years mathematicians asked whether a single
tile could do that: an "einstein", from the German ein Stein, one stone. In
2022 David Smith, a retired print technician from Yorkshire, found it while
playing with paper shapes: the hat, thirteen sides glued from eight kites of
a hexagon grid. With Myers, Kaplan and Goodman-Strauss they proved it can tile
forever but never periodically.

Here the hats spread outward from a single tile while the view pulls back to
thousands of them. Shift the pattern however you like and it never lands on
itself. The yellow hats are flipped over, mirror images of the rest. You can't
avoid them: about one hat in eight has to be reversed, and they never repeat
either.

At the end the outlines light up: hats gather into clusters, clusters into
bigger clusters of the same four kinds, and so on up. That self-similar
hierarchy is how the proof shows the tiling never repeats.

Listen too: every flipped hat strikes a gamelan bell where it lands, higher
the farther out it is, while the air rushes with the zoom. Each level of
clusters lands as a deep gong-like shock, lower and heavier as they grow.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#einsteintile #tessellation #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    hex_to_bgr,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- Tiling ------------------------------------------------------------------
LEVELS = 5  # substitutions of the four metatiles: ~60k hats, far beyond the frame
CENTRE_CHOICES = 24  # the seed picks the starting hat among the ones nearest the middle

# --- Timeline (seconds) --------------------------------------------------------
REVEAL_START, REVEAL_END = 0.8, 10.0  # the hats spread out while the view pulls back
ARRIVAL = 0.45  # how long one hat takes to fade and grow in
HIERARCHY = (10.2, 13.6)  # cluster outlines light up, level by level
SHOWN_LEVELS = (1, 2, 3)  # level 0 (a few hats each) is too fine to read
FADE_OUT = 0.4  # to black over the last moment

# --- Camera ----------------------------------------------------------------------
START_HAT_PX = 520  # the first hat's width on screen
END_HAT_PX = 32  # a hat's width once the view has pulled all the way back
HOLD_ZOOM = 0.06  # a slow push in while the hierarchy shows

# --- Appearance ----------------------------------------------------------------
# Each palette: background, seam, one fill per direction a hat can point in
# (six), the flipped hats' accent, and the hierarchy lines. Hand-mixed, not ramps.
PALETTES = {
    # Deep sea teals and slates, amber flips.
    "tide": (
        "#0b0d12", "#06070a",
        ("#123e4a", "#17525c", "#1d6670", "#2a5578", "#223f63", "#1a2f4d"),
        "#f2a541", "#ffe3b0",
    ),
    # Swiss-poster charcoal greys, signal-yellow flips.
    "graphite": (
        "#0c0c0d", "#050506",
        ("#2a2c30", "#36393e", "#43464c", "#24262a", "#4e5158", "#303237"),
        "#ffd23f", "#fff3c4",
    ),
    # Ultramarine and ink, lemon flips: blue and yellow, the strongest pair.
    "ultramarine": (
        "#07081a", "#03040d",
        ("#1a1f5c", "#222b7a", "#2c3792", "#151947", "#2f2a80", "#1d2c6e"),
        "#ffe14d", "#fff6c9",
    ),
    # Aubergine, plum and dusk rose, gold flips.
    "plum": (
        "#0d0710", "#060308",
        ("#2a1633", "#3d1f47", "#5a2d5c", "#4a1a33", "#241a45", "#6b3a5a"),
        "#ffc94a", "#ffe6b8",
    ),
    # Burnt earth, rust and oxblood, the flips turned electric cyan.
    "ember": (
        "#0e0907", "#060403",
        ("#5a2a1a", "#7a3417", "#3b2418", "#8c4a2f", "#4a1717", "#2e1d16"),
        "#3fe0d0", "#c8fff7",
    ),
    # Pine, moss and olive, vermilion flips.
    "forest": (
        "#070c09", "#030604",
        ("#10372b", "#2c4a22", "#0f4d3a", "#1f5a4a", "#18392f", "#3b4a1e"),
        "#ff5a36", "#ffd9c9",
    ),
}  # fmt: skip
PALETTE = "ultramarine"
VIGNETTE = 0.28
GLOW = 0.9
FLASH = 0.25  # how far an arriving hat brightens towards white

# --- Sound -----------------------------------------------------------------------
STYLE = "gamelan"  # common.music.STYLES; an --audio-only tag naming one overrides it
BELL_GAP = (0.3, 0.11)  # s between flipped-hat bells: calm start, fastest mid-zoom
BELL_SPAN = 14  # scale degrees the bells climb from the first hat to the frame's edge
RAIN = 9.0  # most soft ticks per second, at the peak of the arrivals
RISER = 4.2  # s of build-up that stops dead on the first shock
WIND = 0.22  # loudness of the rushing air that follows the zoom speed

SHIFT = 4  # cv2 sub-pixel bits
SCALE = 1 << SHIFT

R3 = np.sqrt(3.0)
HR3 = R3 / 2.0


# --- The hat and its metatiles (after C. Kaplan's hatviz) -----------------------


def hex_point(x, y):
    return np.array([x + 0.5 * y, HR3 * y])


HAT = np.array(
    [
        hex_point(*p)
        for p in [
            (0, 0), (-1, -1), (0, -2), (2, -2), (2, -1), (4, -2), (5, -1),
            (4, 0), (3, 0), (2, 2), (0, 3), (0, 2), (-1, 2),
        ]
    ]
)  # fmt: skip


def affine(a, b, c, d, e, f):
    return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]])


def translate(x, y):
    return affine(1, 0, x, 0, 1, y)


def rotate(angle):
    c, s = np.cos(angle), np.sin(angle)
    return affine(c, -s, 0, s, c, 0)


def rotate_about(p, angle):
    return translate(*p) @ rotate(angle) @ translate(*-p)


def apply(m, p):
    return m[:2, :2] @ p + m[:2, 2]


def match_segment(p, q):
    """The map sending the unit segment (0,0)-(1,0) to p-q (a similarity)."""
    return affine(q[0] - p[0], p[1] - q[1], p[0], q[1] - p[1], q[0] - p[0], p[1])


def match_two(p1, q1, p2, q2):
    return match_segment(p2, q2) @ np.linalg.inv(match_segment(p1, q1))


def intersect(p1, q1, p2, q2):
    d = (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1])
    u = ((q2[0] - p2[0]) * (p1[1] - p2[1]) - (q2[1] - p2[1]) * (p1[0] - p2[0])) / d
    return p1 + u * (q1 - p1)


class Hat:
    def __init__(self, label):
        self.label = label


class MetaTile:
    def __init__(self, shape):
        self.shape = [np.asarray(p, dtype=float) for p in shape]
        self.children = []  # (transform, Hat or MetaTile)

    def add(self, m, child):
        self.children.append((m, child))

    def child_point(self, n, i):
        m, child = self.children[n]
        return apply(m, child.shape[i])

    def recentre(self):
        c = np.mean(self.shape, axis=0)
        self.shape = [p - c for p in self.shape]
        self.children = [(translate(*-c) @ m, child) for m, child in self.children]


def base_metatiles():
    """The four level-0 metatiles H, T, P, F built from hats (H holds the flipped one)."""
    h_hat, t_hat, p_hat, f_hat, h1_hat = (Hat(s) for s in ("H", "T", "P", "F", "H1"))
    pt = lambda x, y: np.array([x, y], dtype=float)

    h_out = [
        pt(0, 0),
        pt(4, 0),
        pt(4.5, HR3),
        pt(2.5, 5 * HR3),
        pt(1.5, 5 * HR3),
        pt(-0.5, HR3),
    ]
    h = MetaTile(h_out)
    h.add(match_two(HAT[5], HAT[7], h_out[5], h_out[0]), h_hat)
    h.add(match_two(HAT[9], HAT[11], h_out[1], h_out[2]), h_hat)
    h.add(match_two(HAT[5], HAT[7], h_out[3], h_out[4]), h_hat)
    h.add(
        translate(2.5, HR3)
        @ affine(-0.5, -HR3, 0, HR3, -0.5, 0)
        @ affine(0.5, 0, 0, 0, -0.5, 0),
        h1_hat,
    )

    t = MetaTile([pt(0, 0), pt(3, 0), pt(1.5, 3 * HR3)])
    t.add(affine(0.5, 0, 0.5, 0, 0.5, HR3), t_hat)

    second = (
        translate(0, 2 * HR3)
        @ affine(0.5, HR3, 0, -HR3, 0.5, 0)
        @ affine(0.5, 0, 0, 0, 0.5, 0)
    )
    p = MetaTile([pt(0, 0), pt(4, 0), pt(3, 2 * HR3), pt(-1, 2 * HR3)])
    p.add(affine(0.5, 0, 1.5, 0, 0.5, HR3), p_hat)
    p.add(second, p_hat)

    f = MetaTile([pt(0, 0), pt(3, 0), pt(3.5, HR3), pt(3, 2 * HR3), pt(-1, 2 * HR3)])
    f.add(affine(0.5, 0, 1.5, 0, 0.5, HR3), f_hat)
    f.add(second, f_hat)
    return h, t, p, f


PATCH_RULES = [
    ["H"],
    [0, 0, "P", 2], [1, 0, "H", 2], [2, 0, "P", 2], [3, 0, "H", 2],
    [4, 4, "P", 2], [0, 4, "F", 3], [2, 4, "F", 3], [4, 1, 3, 2, "F", 0],
    [8, 3, "H", 0], [9, 2, "P", 0], [10, 2, "H", 0], [11, 4, "P", 2],
    [12, 0, "H", 2], [13, 0, "F", 3], [14, 2, "F", 1], [15, 3, "H", 4],
    [8, 2, "F", 1], [17, 3, "H", 0], [18, 2, "P", 0], [19, 2, "H", 2],
    [20, 4, "F", 3], [20, 0, "P", 2], [22, 0, "H", 2], [23, 4, "F", 3],
    [23, 0, "F", 3], [16, 0, "P", 2], [9, 4, 0, 2, "T", 2], [4, 0, "F", 3],
]  # fmt: skip


def substitute(h, t, p, f):
    """One level up: glue 29 metatiles into a patch and cut the next H, T, P, F from it."""
    shapes = {"H": h, "T": t, "P": p, "F": f}
    patch = MetaTile([])
    for rule in PATCH_RULES:
        if len(rule) == 1:
            patch.add(np.eye(3), shapes[rule[0]])
            continue
        if len(rule) == 4:
            m, child = patch.children[rule[0]]
            poly = child.shape
            a = apply(m, poly[(rule[1] + 1) % len(poly)])
            b = apply(m, poly[rule[1]])
        else:
            mp, cp = patch.children[rule[0]]
            mq, cq = patch.children[rule[2]]
            a = apply(mq, cq.shape[rule[3]])
            b = apply(mp, cp.shape[rule[1]])
        new = shapes[rule[-2]]
        k = rule[-1]
        patch.add(
            match_two(new.shape[k], new.shape[(k + 1) % len(new.shape)], a, b), new
        )

    point = patch.child_point
    bps1, bps2 = point(8, 2), point(21, 2)
    rbps = apply(rotate_about(bps1, -2.0 * np.pi / 3.0), bps2)
    p72, p252 = point(7, 2), point(25, 2)
    llc = intersect(bps1, rbps, point(6, 2), p72)
    w = point(6, 2) - llc

    h_out = [llc, bps1]
    w = apply(rotate(-np.pi / 3), w)
    h_out.append(h_out[1] + w)
    h_out.append(point(14, 2))
    w = apply(rotate(-np.pi / 3), w)
    h_out.append(h_out[3] - w)
    h_out.append(point(6, 2))

    def cut(outline, indices):
        tile = MetaTile(outline)
        for i in indices:
            tile.add(*patch.children[i])
        tile.recentre()
        return tile

    new_h = cut(h_out, [0, 9, 16, 27, 26, 6, 1, 8, 10, 15])
    new_p = cut([p72, p72 + (bps1 - llc), bps1, llc], [7, 2, 3, 4, 28])
    new_f = cut(
        [bps2, point(24, 2), point(25, 0), p252, p252 + (llc - bps1)],
        [21, 20, 22, 23, 24, 25],
    )
    aaa = h_out[2]
    bbb = h_out[1] + (h_out[4] - h_out[5])
    ccc = apply(rotate_about(bbb, -np.pi / 3), aaa)
    new_t = cut([bbb, ccc, aaa], [11])
    return new_h, new_t, new_p, new_f


def flatten(tile, levels):
    """
    Every hat of `tile` as (transform, flipped), and every metatile outline
    with its level (0 = the base metatiles), all in the tile's frame.
    """
    hats, outlines = [], []

    def walk(node, m, level):
        if isinstance(node, Hat):
            hats.append(m)
            return
        outlines.append((level, np.array([apply(m, p) for p in node.shape])))
        for cm, child in node.children:
            walk(child, m @ cm, level - 1)

    walk(tile, np.eye(3), levels)
    return np.array(hats), outlines


def build_tiling():
    tiles = base_metatiles()
    for _ in range(LEVELS):
        tiles = substitute(*tiles)
    return flatten(tiles[0], LEVELS)


# --- Video -----------------------------------------------------------------------


def soundtrack(events, style_name, rng):
    """
    A one-shot that follows the camera and lands on the hierarchy shocks.

    `events` comes from the same closed forms as the frames: the times, screen
    x and distance (in hats) at which every flipped hat arrives on screen, the
    arrival times of all visible hats, the zoom speed |d log scale / dt| on a
    time grid, and the moments the three hierarchy levels light up.

    - The first hat: a bell on the root over a low note and a pad that swells.
    - Flipped hats: a bell each, panned to where it lands, climbing BELL_SPAN
      degrees with log distance (the farther out, the higher), thinned so they
      are never closer than BELL_GAP, which tightens with the zoom speed.
    - Every hat: soft noise ticks like rain, their rate following the log of
      the arrival rate (up to RAIN per second).
    - The zoom: stereo wind crossfading from dark to bright noise with the
      zoom speed, then a riser over the last RISER seconds before the first
      shock, cut dead by it.
    - The shocks: per hierarchy level a low kick, a crash of noise, a bass
      note, a rolled bell chord and a pad held to the next shock, on chords
      vi, IV and finally I (lower and louder as the clusters grow); the last
      rings out under the fade.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    size = len(scale.steps)
    track = audio.silence(DURATION)
    n = len(track)
    kw = dict(circular=False)

    # The first hat.
    audio.add(track, audio.bell(scale.hz(0), 0.55), 0.0, **kw)
    audio.add(track, style.bass(scale.hz(0, octave=-2), 0.5), 0.0, **kw)
    for degree in (0, 4):
        audio.add(
            track,
            audio.pad(scale.hz(degree, -1), 9.0, 0.22, 2.5, 1.5, 0.25),
            0.05,
            **kw,
        )

    # Flipped hats: thinned bells that climb outward.
    grid, speed = events["grid"], events["speed"]
    level = speed / speed.max()
    reach = np.log1p(events["flip_distance"]) / np.log1p(events["flip_distance"].max())
    last, bells = -1.0, 0
    for when, x, r in zip(events["flip_time"], events["flip_x"], reach):
        gap = np.interp(np.interp(when, grid, level), [0, 1], BELL_GAP)
        if when - last < gap or when > HIERARCHY[0] - 0.1:
            continue
        last = when
        degree = int(round(BELL_SPAN * r)) + int(rng.integers(0, 2)) * 2
        velocity = 0.3 + 0.2 * r
        note = audio.bell(scale.hz(degree), velocity)
        audio.add(track, audio.pan(note, 0.8 * x), when, **kw)
        bells += 1

    # Rain: one soft tick per many arrivals.
    counts, edges = np.histogram(
        events["all_time"], bins=np.arange(0, DURATION + 0.1, 0.1)
    )
    rate = np.log1p(counts / 0.1 / 20.0)
    rate = RAIN * rate / max(rate.max(), 1e-9)
    ticks = 0
    for k, r in enumerate(rate):
        for when in edges[k] + 0.1 * rng.uniform(size=rng.poisson(r * 0.1)):
            tick = audio.tick(rng, rng.uniform(0.05, 0.1), 0.025, 6500.0)
            audio.add(track, audio.pan(tick, rng.uniform(-0.7, 0.7)), when, **kw)
            ticks += 1

    # Wind with the zoom speed, then the riser into the first shock.
    t = np.arange(n) / audio.SAMPLE_RATE
    gust = np.interp(t, grid, level)[:, None]
    noise = rng.standard_normal((n, 2))
    dark = audio.lowpass(noise, 500.0, **kw)
    bright = audio.highpass(audio.lowpass(noise, 4500.0, **kw), 900.0, **kw)
    dark /= np.abs(dark).max()
    bright /= np.abs(bright).max()
    track += WIND * gust**1.5 * (dark + gust * (bright - dark))
    shocks = events["shocks"]
    rise = audio.riser(RISER, rng, scale.hz(0, -2), scale.hz(0, 1), 0.7)
    audio.add(track, rise[:, None] * np.ones(2), shocks[0] - RISER, **kw)

    # The shocks.
    roots = (5, 3, 0) if size == 7 else (3, 1, 0)
    for k, when in enumerate(shocks):
        last_shock = k == len(shocks) - 1
        hold = (DURATION - when) if last_shock else (shocks[k + 1] - when)
        weight = 0.75 + 0.15 * k
        audio.add(track, audio.kick(weight, 52.0 - 4.0 * k), when, **kw)
        crash = audio.tick(rng, 0.35 * weight, 0.22, 1800.0)
        audio.add(track, audio.pan(crash, -0.3 + 0.3 * k), when, **kw)
        audio.add(
            track, style.bass(scale.hz(roots[k], octave=-2), weight, 3.5), when, **kw
        )
        for i, degree in enumerate(scale.chord(roots[k], 4, octave=1)):
            bell = audio.bell(float(audio.midi_to_hz(degree)), 0.45 * weight)
            audio.add(track, audio.pan(bell, -0.5 + 0.33 * i), when + 0.035 * i, **kw)
        for note in scale.chord(roots[k], 3, octave=-1):
            pad = audio.pad(
                float(audio.midi_to_hz(note)), hold, 0.3 * weight, 0.25, 1.2, 0.4
            )
            audio.add(track, pad, when, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {bells} bells, "
        f"{ticks} ticks, shocks at " + ", ".join(f"{w:.2f} s" for w in shocks)
    )
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, FADE_OUT + 0.2)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The hat monotile (Smith, Myers, Kaplan & Goodman-Strauss 2023) revealed
    outward from one tile while the camera pulls back, then its metatile
    hierarchy.

    The tiling is built once by the H/T/P/F metatile substitution of the
    paper (as in Kaplan's hatviz): LEVELS rounds of gluing 29 metatiles into
    a patch and cutting the next level's four metatiles from it, then the top
    H is flattened into ~60k hat transforms plus every metatile outline with
    its level. Reflected hats are those whose transform determinant has the
    minority sign.

    Every frame is a closed form in t. The camera scale goes exponentially
    from START_HAT_PX to END_HAT_PX per hat width (eased), and the reveal
    radius in screen pixels grows from the first hat to past the frame's
    corners, so hat i arrives at the time its centroid distance d_i is passed
    (tabulated once and inverted by interpolation). Arriving hats fade, grow
    from 60 % about their centroid and flash towards white. Hats are filled
    with cv2.fillPoly (antialiased, 4 sub-pixel bits) grouped by colour,
    seams are polylines scaled with the zoom, and flipped hats, flashes and
    the hierarchy outlines go into a quarter-resolution glow layer that is
    blurred and added on top. Over HIERARCHY the metatile outlines of
    SHOWN_LEVELS light up one after another, leaving a faint trace. The seed rotates
    the tiling and picks the starting hat.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    hats, outlines = build_tiling()
    lin = hats[:, :2, :2]
    det = np.linalg.det(lin)
    flipped = np.sign(det) != np.sign(np.median(det))
    logger.info(
        f"{len(hats)} hats, {flipped.sum()} flipped "
        f"(1 in {len(hats) / flipped.sum():.2f}), {len(outlines)} metatiles"
    )

    # World-space hat polygons and centroids.
    polys = np.einsum("nij,kj->nki", lin, HAT) + hats[:, None, :2, 2]
    centroids = polys.mean(axis=1)
    hat_width = np.ptp(HAT[:, 0]) * np.sqrt(abs(np.median(det)))

    # Orientation class: the angle of the transform's first column, in sixths of a turn.
    angle = np.arctan2(lin[:, 1, 0], lin[:, 0, 0])
    orientation = np.round(angle / (np.pi / 3)).astype(int) % 6

    # Pick the starting hat (unflipped) near the middle of the patch; rotate the world.
    middle = centroids.mean(axis=0)
    near = np.argsort(np.linalg.norm(centroids - middle, axis=1))
    near = near[~flipped[near]][:CENTRE_CHOICES]
    start = near[rng.integers(len(near))]
    origin = centroids[start]
    turn = rotate(rng.uniform(0, 2 * np.pi))[:2, :2]
    polys = (polys - origin) @ turn.T
    centroids = (centroids - origin) @ turn.T
    outlines = [(level, (o - origin) @ turn.T) for level, o in outlines]
    distance = np.linalg.norm(centroids, axis=1)
    logger.info(f"Patch covers radius {distance.max() / hat_width:.0f} hats")

    # Colours (BGR).
    background, seam_colour, orientation_fills, accent, line_colour = PALETTES[PALETTE]
    bg = np.array(hex_to_bgr(background), float)
    fills = np.array([hex_to_bgr(c) for c in orientation_fills + (accent,)], float)
    colour_index = np.where(flipped, 6, orientation)
    seam = hex_to_bgr(seam_colour)
    line = np.array(hex_to_bgr(line_colour), float)

    # Camera and reveal timing.
    s0 = START_HAT_PX / hat_width
    s1 = END_HAT_PX / hat_width
    corner = 0.5 * np.hypot(WIDTH, HEIGHT)

    def camera(sec, push=True):
        u = smoothstep((sec - REVEAL_START) / (REVEAL_END - REVEAL_START))
        if not push:  # line widths ignore the final push in
            return s0 * (s1 / s0) ** u
        drift = 1 + HOLD_ZOOM * smoothstep((sec - REVEAL_END) / (DURATION - REVEAL_END))
        return s0 * (s1 / s0) ** u * drift

    def strokes(image, lines, colour, width):
        """
        cv2 strokes come in whole pixels (and antialiased ones in uneven
        steps), so a width that grows smoothly would jump. Draw the two whole
        widths around it and blend them by the fraction: no jumps.
        """
        width = max(width, 1.0)
        low = int(np.floor(width))
        frac = width - low
        out = np.zeros(image.shape, np.float32)
        for pixels, weight in ((low, 1 - frac), (low + 1, frac)):
            if weight > 1e-3:
                layer = np.zeros_like(image)
                cv2.polylines(layer, lines, True, colour, pixels, cv2.LINE_AA, SHIFT)
                out += weight * layer
        return out

    def reveal_px(sec):
        u = np.clip((sec - REVEAL_START) / (REVEAL_END - REVEAL_START), 0, 1)
        u = u**0.8 * (1 - 0.35 * u) / 0.65  # fast start, eased finish
        return 0.02 * START_HAT_PX + u * (1.08 * corner - 0.02 * START_HAT_PX)

    grid = np.linspace(0, DURATION, 4000)
    reach = reveal_px(grid) / camera(grid)  # world radius revealed by each moment
    reach = np.maximum.accumulate(reach)
    arrival = np.interp(distance, reach, grid, right=np.inf)
    arrival[start] = -0.5 * ARRIVAL  # already glowing on the first frame

    # Hierarchy outlines: levels 0..3, each with its own moment.
    level_times = np.linspace(*HIERARCHY, len(SHOWN_LEVELS) + 1)[:-1]
    level_outlines = []
    for k in SHOWN_LEVELS:
        polys_k = [o for level, o in outlines if level == k]
        centres = np.array([o.mean(axis=0) for o in polys_k])
        radii = np.array(
            [np.linalg.norm(o - o.mean(axis=0), axis=1).max() for o in polys_k]
        )
        level_outlines.append((polys_k, centres, radii))

    yy = np.linspace(-1, 1, HEIGHT)[:, None]
    xx = np.linspace(-1, 1, WIDTH)[None, :] * (WIDTH / HEIGHT)
    vignette = np.clip(1 - VIGNETTE * (xx**2 + yy**2), 0, 1)[..., None].astype(
        np.float32
    )
    gw, gh = WIDTH // 4, HEIGHT // 4

    def to_screen(points, scale, offset=(WIDTH / 2, HEIGHT / 2), factor=1.0):
        return np.round((points * scale + offset) * factor * SCALE).astype(np.int32)

    def render(frame):
        sec = frame / (FRAME_COUNT - 1) * DURATION
        scale = camera(sec)
        margin = hat_width * scale
        screen_c = centroids * scale
        visible = (np.abs(screen_c[:, 0]) < WIDTH / 2 + margin) & (
            np.abs(screen_c[:, 1]) < HEIGHT / 2 + margin
        )
        a = np.clip((sec - arrival) / ARRIVAL, 0, 1)
        shown = visible & (a > 0)
        settled = shown & (a >= 1)
        arriving = np.flatnonzero(shown & (a < 1))

        canvas = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        canvas[:] = bg.astype(np.uint8)
        glow = np.zeros((gh, gw, 3), np.uint8)
        overlay = np.zeros((HEIGHT, WIDTH, 3), np.float32)  # hierarchy lines, added
        line_scale = camera(sec, push=False)

        idx_settled = np.flatnonzero(settled)
        pts = to_screen(polys[idx_settled], scale)
        for c in range(7):
            sel = colour_index[idx_settled] == c
            if sel.any():
                cv2.fillPoly(
                    canvas, list(pts[sel]), fills[c].tolist(), cv2.LINE_AA, SHIFT
                )
        flipped_settled = idx_settled[flipped[idx_settled]]
        if len(flipped_settled):
            gpts = to_screen(polys[flipped_settled], scale, factor=0.25)
            cv2.fillPoly(glow, list(gpts), fills[6].tolist(), cv2.LINE_AA, SHIFT)

        for i in arriving:
            ai = smoothstep(a[i])
            grow = 0.6 + 0.4 * ai
            poly = centroids[i] + (polys[i] - centroids[i]) * grow
            flash = 4 * ai * (1 - ai)
            colour = bg + ai * (fills[colour_index[i]] - bg)
            colour = colour + FLASH * flash * (255 - colour)
            p = to_screen(poly[None], scale)
            cv2.fillPoly(canvas, list(p), colour.tolist(), cv2.LINE_AA, SHIFT)
            g = to_screen(poly[None], scale, factor=0.25)
            glow_colour = (fills[6] * ai if flipped[i] else np.zeros(3)) + 60 * flash
            cv2.fillPoly(
                glow, list(g), np.minimum(glow_colour, 255).tolist(), cv2.LINE_AA, SHIFT
            )

        image = canvas.astype(np.float32)
        shown_idx = np.flatnonzero(shown)
        if len(shown_idx):
            ink = strokes(
                np.zeros((HEIGHT, WIDTH), np.uint8),
                list(to_screen(polys[shown_idx], scale)),
                255,
                0.045 * hat_width * line_scale,
            )
            image += (ink[..., None] / 255) * (np.float32(seam) - image)

        # Hierarchy: a pulse per level, leaving a faint trace.
        for n, (k, t0) in enumerate(zip(SHOWN_LEVELS, level_times)):
            dt = sec - t0
            if dt < 0:
                continue
            trace = 0.6 if n == len(level_times) - 1 else 0.1  # the last one stays
            pulse = smoothstep(dt / 0.35) * (
                trace + (1 - trace) * np.exp(-max(0.0, dt - 0.35) / 0.5)
            )
            polys_k, centres, radii = level_outlines[n]
            near = (np.abs(centres[:, 0]) * scale < WIDTH / 2 + radii * scale) & (
                np.abs(centres[:, 1]) * scale < HEIGHT / 2 + radii * scale
            )
            polys_k = [polys_k[j] for j in np.flatnonzero(near)]
            pk = [to_screen(o, scale) for o in polys_k]
            width_k = (1.0 + 0.6 * k) * line_scale * hat_width * 0.06
            overlay += strokes(canvas, pk, (pulse * 0.75 * line).tolist(), width_k)
            gk = [to_screen(o, scale, factor=0.25) for o in polys_k]
            cv2.polylines(
                glow, gk, True, (pulse * line).tolist(), 1 + k // 2, cv2.LINE_AA, SHIFT
            )

        bloom = cv2.GaussianBlur(glow, (0, 0), 3.0)
        bloom = cv2.resize(bloom, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        # Less bloom once the hats are small, or the glow turns into haze.
        gain = GLOW * np.clip(hat_width * scale / 90, 0.3, 1.0)
        image += overlay
        image += gain * bloom.astype(np.float32)
        image *= vignette
        fade = 1 - smoothstep((sec - (DURATION - FADE_OUT)) / FADE_OUT)
        return image * fade

    render_frames(settings, render, FRAME_COUNT)

    # The sound's events, from the same camera and arrival times as the frames.
    def on_screen(i):
        sc = camera(np.maximum(arrival[i], 0.0))[:, None]
        p = centroids[i] * sc
        inside = (np.abs(p[:, 0]) < WIDTH / 2) & (np.abs(p[:, 1]) < HEIGHT / 2)
        return i[inside], p[inside, 0] / (WIDTH / 2)

    timed = np.flatnonzero(np.isfinite(arrival))
    flips, flip_x = on_screen(timed[flipped[timed]])
    order = np.argsort(arrival[flips])
    seen, _ = on_screen(timed)
    log_scale = np.log(camera(grid))
    events = {
        "flip_time": arrival[flips][order],
        "flip_x": flip_x[order],
        "flip_distance": distance[flips][order] / hat_width,
        "all_time": arrival[seen],
        "grid": grid,
        "speed": np.abs(np.gradient(log_scale, grid)),
        "shocks": level_times,
    }
    tag = settings.audio_tag if hasattr(settings, "audio_tag") else ""
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(events, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
