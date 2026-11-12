"""
The Knight's Tour

A chess knight visits all 64 squares of a chessboard exactly once, then jumps
back to where it started: a closed knight's tour, a loop it can run forever.

Join every pair of squares a knight move apart and you get the knight's graph,
the faint web behind the board. The tour is one loop through that network that
touches every square once, a Hamiltonian cycle. Euler studied them in 1759; in
1997 a computer counted all the closed tours: 13,267,364,410,532.

This one is special: turn the board upside down and it looks the same, so a
second knight half a tour behind always mirrors the first.

Listen: a slow chord moves every eight jumps, and a kalimba chimes when the
knights land near the corners and edges, where squares have the fewest moves.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#knightstour #graphtheory #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds, one loop
FRAME_COUNT = frame_count(DURATION)

# --- The tour ----------------------------------------------------------------------
N = 8  # board size
KNIGHT = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
HOPS = 32  # each knight makes half the tour per loop (the other half is its mirror)
HOP_TIME = DURATION / HOPS  # 0.469 s per jump
MOVE = 0.6  # fraction of a hop spent in the air; the rest is a pause on the square
SEARCH_LIMIT = 200_000  # backtracking nodes before a restart with new tie-breaks

# --- Layout (px) -------------------------------------------------------------------
CELL = 112
BOARD = N * CELL  # 896
BOARD_X, BOARD_Y = (WIDTH - BOARD) // 2, (HEIGHT - BOARD) // 2
PAD = 40  # the drawn region extends this far past the board (glow)
TITLE_Y, SUBTITLE_Y = 300, 392
FOOT_Y, FOOT2_Y = 1500, 1572

# --- Look --------------------------------------------------------------------------
BG_TOP, BG_BOTTOM = "#0b0d1c", "#04050a"
SQUARES = ("#0d1022", "#12162c")  # dark and light squares, barely apart
WEB = "#5260b0"  # the knight's graph, drawn faint
WEB_ALPHA = 0.3
TRAIL = ["#1a1d4a", "#2e4f96", "#5d93cf", "#a9d2f0", "#f4ecf6"]  # old -> fresh
HEAD = "#ff8a5e"  # the knight, and the edge it is drawing
TEXT, TEXT_DIM = "#ecebf4", "#9097bd"
TAU = 7.0  # hops for a trail to fade by e
FLOOR = 0.12  # brightness of the oldest trail
CELL_GLOW = 1.4  # hops for a landed square to fade by e
KNIGHT_R = 18.0  # px
GLOW_SCALE, GLOW_SIGMA, GLOW = 4, 4.0, 1.1

# --- Sound -------------------------------------------------------------------------
ROOTS = range(50, 56)  # MIDI root of the chords (D3 .. G3)
PROGRESSION = (0, 3, 5, 4)  # I IV vi V, one chord per 8 hops
CHIME_DEGREE = 4  # landings on squares with at most this many knight moves chime


def square_xy(square):
    """Pixel centre of a square (0..63, row-major from the top-left)."""
    r, c = divmod(square, N)
    return BOARD_X + (c + 0.5) * CELL, BOARD_Y + (r + 0.5) * CELL


def neighbours(square):
    r, c = divmod(square, N)
    return [
        (r + dr) * N + c + dc
        for dr, dc in KNIGHT
        if 0 <= r + dr < N and 0 <= c + dc < N
    ]


def symmetric_tour(rng):
    """
    A closed knight's tour that maps onto itself under a half turn of the board
    (square s -> 63 - s). Backtracking over the first half, p0 .. p31: each square
    claims its mirror image too, and the half closes if p31 is a knight move from
    63 - p0. The full tour is p0 .. p31, 63 - p0 .. 63 - p31. Moves are tried in
    Warnsdorff order (fewest onward moves first), ties broken by random priorities.
    """
    adjacency = [neighbours(s) for s in range(N * N)]
    for attempt in range(100):
        priority = rng.random(N * N)
        start = int(rng.integers(N * N))
        used = np.zeros(N * N, bool)
        path = [start]
        used[start] = used[63 - start] = True
        nodes = 0

        def options(square):
            free = [q for q in adjacency[square] if not used[q]]
            onward = [sum(not used[x] for x in adjacency[q]) for q in free]
            return [q for _, _, q in sorted(zip(onward, priority[free], free))]

        def extend():
            nonlocal nodes
            nodes += 1
            if nodes > SEARCH_LIMIT:
                return False
            if len(path) == HOPS:
                return (63 - path[0]) in adjacency[path[-1]]
            for q in options(path[-1]):
                path.append(q)
                used[q] = used[63 - q] = True
                if extend():
                    return True
                path.pop()
                used[q] = used[63 - q] = False
            return False

        if extend():
            logger.info(f"Symmetric tour found on attempt {attempt + 1}, {nodes} nodes")
            return path + [63 - s for s in path]
    raise RuntimeError("No symmetric tour found")


def hop_state(s):
    """
    Knight A's place at s hops into the loop: the hop index m and the eased
    fraction u of that jump travelled (0 on take-off, 1 once landed).
    """
    m = int(np.floor(s)) % (2 * HOPS)
    u = float(smoothstep((s - np.floor(s)) / MOVE))
    return m, u


def generate(settings: ImageProcessingSettings = None):
    """
    A closed knight's tour with half-turn symmetry, travelled forever by two knights
    half a tour apart. A seamless 15 s loop.

    The tour: a Hamiltonian cycle on the 8x8 knight's graph (168 edges, drawn faint)
    found by backtracking with Warnsdorff ordering, constrained so that the tour is
    invariant under the half turn s -> 63 - s (see symmetric_tour; tie-breaks and the
    start square come from settings.rng, so every seed gives a different tour). The
    second half of the tour is the mirror of the first, so knight B, 32 hops behind
    knight A, is always A's mirror image.

    Time: t = frame / FRAME_COUNT in [0, 1), s = 32 t hops. Knight A is on hop
    m = floor(s) from square p_m to p_{m+1}, with smoothstep easing over the first 60%
    of each 0.469 s hop and a pause on the square after; B is on hop m + 32. After one
    loop A stands where B started and vice versa, so the frame repeats exactly. Each
    tour edge k has age (s - k - 0.6) mod 32 hops since a knight last drew it;
    brightness FLOOR + (1 - FLOOR) exp(-age / 7) through an indigo-to-ice LUT. The edge
    in flight is drawn coral from p_m to the knight. Landed squares glow and fade with
    the same age. Knights are coral discs that swell mid-jump.

    Render: cv2 lines (LINE_AA, 4-bit subpixel) on a float32 crop of the board, plus a
    glow (1/4-resolution Gaussian blur) of the light layer, over a precomputed
    background (gradient, board, faint knight's graph, LaTeX text). Frames are pure
    functions of t on the render_frames thread pool.

    Sound (posted: --seed 0, no --audio-only tag, i.e. the "choir" variant of
    ambient(), audio seed = seed, root F3): an ambient loop-periodic bed of pad and
    "u" choir chords, I IV vi V in major, one chord per eight hops, with a kalimba
    chime whenever the knights land on squares with at most four knight moves
    (corner and edge regions): two tines for the pair of mirrored landings (pitch
    by height on the board, pan by x), an octave doubled at the corners. Long
    reverb and a hop-synced echo; circular throughout, master(loop=True).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    tour = symmetric_tour(rng)
    points = np.array([square_xy(s) for s in tour])  # (64, 2) px, frame coords
    degree = np.array([len(neighbours(s)) for s in range(N * N)])

    # --- Static background ------------------------------------------------------
    top, bottom = np.array(hex_to_bgr(BG_TOP)), np.array(hex_to_bgr(BG_BOTTOM))
    ramp = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    background = np.broadcast_to(
        (1 - ramp) * top + ramp * bottom, (HEIGHT, WIDTH, 3)
    ).copy()
    for r in range(N):
        for c in range(N):
            y0, x0 = BOARD_Y + r * CELL, BOARD_X + c * CELL
            background[y0 : y0 + CELL, x0 : x0 + CELL] = hex_to_bgr(
                SQUARES[(r + c) % 2]
            )
    web = np.zeros((HEIGHT, WIDTH), np.uint8)
    for s in range(N * N):
        for q in neighbours(s):
            if q > s:
                a, b = ((np.array(square_xy(x)) * 16).astype(int) for x in (s, q))
                cv2.line(web, tuple(a), tuple(b), 255, 1, cv2.LINE_AA, shift=4)
    alpha = WEB_ALPHA * web.astype(np.float32)[..., None] / 255
    background += alpha * (np.array(hex_to_bgr(WEB), np.float32) - background)

    for tex, size, y, colour in (
        (r"$\mathrm{Every\ square\ once{,}\ then\ home}$", 64, TITLE_Y, TEXT),
        (
            r"$\mathrm{a\ closed\ tour\ of\ the\ knight\ graph}$",
            42,
            SUBTITLE_Y,
            TEXT_DIM,
        ),
        (r"$13{,}267{,}364{,}410{,}532\ \mathrm{such\ tours}$", 44, FOOT_Y, TEXT),
        (
            r"$\mathrm{this\ one\ looks\ the\ same\ upside\ down}$",
            40,
            FOOT2_Y,
            TEXT_DIM,
        ),
    ):
        mask = latex_mask(tex, size)
        blit_mask(background, mask, WIDTH // 2, y, hex_to_bgr(colour), anchor="centre")
    background_u8 = np.clip(background, 0, 255).astype(np.uint8)

    # The drawn region: the board plus a margin, in its own coordinates.
    cx0, cy0 = BOARD_X - PAD, BOARD_Y - PAD
    side = BOARD + 2 * PAD
    crop = background[cy0 : cy0 + side, cx0 : cx0 + side].copy()
    local = (points - [cx0, cy0]) * 16  # 4-bit subpixel coordinates
    lut = palette_lut(TRAIL, 256)[:, ::-1] * 255  # BGR
    head = np.array(hex_to_bgr(HEAD), np.float32)
    head_c = tuple(float(v) for v in head)
    cell_xy = [
        (int(points[j, 0]) - cx0 - CELL // 2, int(points[j, 1]) - cy0 - CELL // 2)
        for j in range(2 * HOPS)
    ]

    def line(layer, a, b, colour, thickness):
        cv2.line(
            layer,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            tuple(float(v) for v in colour),
            thickness,
            cv2.LINE_AA,
            shift=4,
        )

    def render(frame):
        t = frame / FRAME_COUNT
        s = HOPS * t
        m, u = hop_state(s)
        light = np.zeros((side, side, 3), np.float32)

        # Squares glow as a knight lands on them (square k + 1 at the end of hop k).
        age = (s - np.arange(2 * HOPS) - MOVE) % HOPS
        for k in np.flatnonzero(age < 6 * CELL_GLOW):
            x, y = cell_xy[(k + 1) % (2 * HOPS)]
            glow = 0.12 * np.exp(-age[k] / CELL_GLOW) * min(1.0, age[k] / 0.12)
            light[y + 3 : y + CELL - 3, x + 3 : x + CELL - 3] += glow * lut[200]

        # Trail edges, oldest first so fresh ones draw on top.
        strength = FLOOR + (1 - FLOOR) * np.exp(-age / TAU)
        for k in np.argsort(-age):
            colour = lut[int(255 * strength[k])] * (0.25 + 0.75 * strength[k])
            line(light, local[k], local[(k + 1) % (2 * HOPS)], colour, 3)

        # The two knights and the edge each is drawing right now.
        lift = np.sin(np.pi * u)
        # During the pause on the square the coral edge cools into the colour it
        # has as a finished trail edge at the start of the next hop: no pop.
        settle = float(smoothstep((s - np.floor(s) - MOVE) / (1 - MOVE)))
        done = FLOOR + (1 - FLOOR) * np.exp(-(1 - MOVE) / TAU)
        done = lut[int(255 * done)] * (0.25 + 0.75 * done)
        edge = head + settle * (done - head)
        for k in (m, (m + HOPS) % (2 * HOPS)):
            a, b = local[k], local[(k + 1) % (2 * HOPS)]
            pos = a + u * (b - a)
            if u > 0:
                line(light, a, pos, edge, 3)
            radius = KNIGHT_R * (1 + 0.45 * lift)
            centre = (int(pos[0]), int(pos[1]))
            cv2.circle(light, centre, int(16 * radius), head_c, -1, cv2.LINE_AA, 4)
            cv2.circle(
                light,
                centre,
                int(16 * radius * 0.4),
                (235.0, 240.0, 255.0),
                -1,
                cv2.LINE_AA,
                4,
            )

        small = cv2.resize(
            light,
            (side // GLOW_SCALE, side // GLOW_SCALE),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(small, (0, 0), GLOW_SIGMA)
        glow = cv2.resize(small, (side, side), interpolation=cv2.INTER_LINEAR)
        region = crop + light + GLOW * glow
        image = background_u8.copy()
        image[cy0 : cy0 + side, cx0 : cx0 + side] = np.clip(region, 0, 255)
        return image

    render_frames(settings, render, FRAME_COUNT)
    track = ambient(tour, degree, settings.audio_tag, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def ambient(tour, degree, variant, rng):
    """
    The posted soundtrack: a sustained bed with sparse chimes, all loop-periodic.

    Bed: I IV vi V in major, one chord per eight hops (3.75 s), as soft pads (and a
    quiet "u" choir; the "strings" variant swaps the choir for strings and chimes
    only on squares with at most three moves). Chimes:
    every hop both knights land at once on mirrored squares p and 63 - p, which have
    the same number of knight moves. If that number is at most CHIME_DEGREE (corner
    and edge regions), two kalimba tines sound, one per knight: pitch from the
    square's height on the board within the current chord's tones, pan from its x.
    Corners (2 moves) ring loudest with an octave above, squares with 4 moves
    softest. A hop-synced echo and a long room; circular throughout,
    master(loop=True).
    """
    root = int(rng.choice(ROOTS))
    scale = Scale(root, "major")
    track = audio.silence(DURATION)
    bar = 8 * HOP_TIME
    bed = audio.silence(DURATION)
    for i, chord_degree in enumerate(PROGRESSION):
        start = i * bar
        notes = scale.chord(chord_degree, 3)
        for j, midi in enumerate(notes):
            hz = float(audio.midi_to_hz(midi))
            held = audio.pad(hz, bar, 0.30, attack=1.4, release=2.6, brightness=0.35)
            audio.add(bed, held, start - 0.6)
            if variant == "strings":
                voice = audio.strings(
                    float(audio.midi_to_hz(midi + 12)),
                    bar,
                    0.16,
                    attack=1.6,
                    release=2.4,
                    brightness=0.3,
                )
            else:
                voice = audio.choir(
                    float(audio.midi_to_hz(midi + 12)),
                    bar,
                    0.17,
                    attack=1.6,
                    release=2.4,
                    vowel="u",
                )
            audio.add(bed, voice, start - 0.6)
    bed = audio.lowpass(bed, 2200.0)
    track += bed

    chimes = audio.silence(DURATION)
    threshold = CHIME_DEGREE - (variant == "strings")  # strings: corner regions only
    count = 0
    for k in range(HOPS):
        landing = (k + MOVE) * HOP_TIME
        chord_degree = PROGRESSION[int(k // 8) % len(PROGRESSION)]
        tones = sorted(
            {m + 12 * o for m in scale.chord(chord_degree, 3) for o in (1, 2, 3)}
        )
        tones = [m for m in tones if root + 16 <= m <= root + 36]
        for square in (tour[(k + 1) % len(tour)], tour[(k + 1 + HOPS) % len(tour)]):
            moves = degree[square]
            if moves > threshold:
                continue
            r, c = divmod(square, N)
            midi = tones[round((N - 1 - r) / (N - 1) * (len(tones) - 1))]
            velocity = {2: 0.55, 3: 0.42, 4: 0.28}[moves]
            position = 0.8 * (c - (N - 1) / 2) / ((N - 1) / 2)
            note = audio.kalimba(float(audio.midi_to_hz(midi)), velocity, 3.0)
            audio.add(chimes, audio.pan(note, position), landing)
            if moves == 2:
                high = audio.kalimba(float(audio.midi_to_hz(midi + 12)), 0.3, 3.0)
                audio.add(chimes, audio.pan(high, position), landing + 0.03)
            count += 1
    chimes = audio.lowpass(chimes, 5000.0, order=4)
    chimes = audio.delay(chimes, 2 * HOP_TIME, feedback=0.35, wet=0.3)
    track += chimes
    track = audio.highpass(track, 80.0)
    track = audio.reverb(track, rng, rt60=4.5, wet=0.42, tone=4200.0)
    logger.info(
        f"Ambient soundtrack ({variant or 'choir'}): root {root}, {count} chimes"
    )
    return audio.master(track)


if __name__ == "__main__":
    generate()
