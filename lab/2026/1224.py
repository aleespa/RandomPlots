"""
A Triangle Folds Into a Square

An equilateral triangle, cut into four pieces and joined at three hinges, swings shut
into a perfect square, then back. Nothing is stretched or trimmed: the same four pieces
cover exactly the same area both ways.

This is Henry Dudeney's hinged dissection, a puzzle he published in 1902. A deeper
theorem (Wallace, Bolyai and Gerwien) says any two polygons with the same area can be
cut into finitely many pieces that rearrange from one into the other. Dudeney found the
triangle-to-square case with only four pieces, and on a chain, so they fold like a
bracelet. Behind it, the triangle tiling of the floor turns into a square tiling made
of the same four cuts.

Each hinge strikes a vibraphone bar as it lets go and again as its piece clicks into
place, rising as the square closes and falling as the triangle returns, over a soft
chord that changes with the shape.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#hingeddissection #geometry #mathart #generativeart #randomplots
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
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    render_frames,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'A triangle that folds into a square'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # s, one loop: triangle -> square -> triangle
FRAME_COUNT = frame_count(DURATION)

# --- The dissection (unit triangle of side 2, y up) -------------------------------
R3 = np.sqrt(3.0)
SIDE = 3**0.25  # the square's side: same area as the triangle, √3
A, B, C = np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, R3])
E, D = (A + C) / 2, (B + C) / 2  # midpoints: two of the hinges
F = np.array([1.5 - np.sqrt(SIDE**2 - 0.75), 0.0])  # |DF| = SIDE
G = F + np.array([1.0, 0.0])  # the third hinge, on the base
U = (F - D) / SIDE
P = D + np.dot(E - D, U) * U  # feet of the perpendiculars dropped onto DF
Q = D + np.dot(G - D, U) * U
PIECES = [
    np.array([A, F, P, E]),  # swings about E
    np.array([E, P, Q, D, C]),  # holds still
    np.array([F, G, Q, P]),  # swings about G, carried by the next
    np.array([G, B, D, Q]),  # swings about D
]
SQUARE_TILT = -(np.arctan2(U[1], U[0]) % (np.pi / 2))  # turn that lays the square flat

# --- Timing (s) -------------------------------------------------------------------
FOLD_START, UNFOLD_START = 0.3, 7.8
SWING = 4.8  # each hinge's half turn
STAGGER = 0.6  # hinge after hinge
ECHOES, ECHO_DT = 14, 0.022  # ghost outlines at earlier times

# --- Look -------------------------------------------------------------------------
SS = 2
SCALE = 470.0  # px per unit at rest: the triangle is 940 px wide
FIT = (520.0, 780.0)  # px: the open chain stays inside these half-extents (zooming out)
ZOOM_WINDOW = 0.9  # s: smoothing of the zoom
CENTRE = (WIDTH / 2, HEIGHT * 0.49)
BACKGROUND = "#0c0d18"
PIECE_COLOURS = ["#ff5e57", "#ffc23c", "#34d399", "#6c8cff"]
HINGE = "#fff6e6"
TEXT = "#c9c6d6"
TILE_ALPHA = 0.42  # the background tilings
BLOOM = 0.32

# --- Sound ------------------------------------------------------------------------
STYLE = "vibes"  # common.music.STYLES; an --audio-only tag naming one overrides it
HINGE_DEGREES = (2, 4, 6)  # hinges E, D, G: the note each lets go on
CLOSE_STEP = 3  # degrees higher when a piece lands in the square
HINGE_PAN = (-0.45, 0.45, 0.1)


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)  # smootherstep


def progress(seconds):
    """Half-turn fraction of hinges E, D, G at a time in seconds (wrapped)."""
    s = np.mod(seconds, DURATION)
    out = []
    for k in range(3):
        fold = ease((s - FOLD_START - k * STAGGER) / SWING)
        unfold = ease((s - UNFOLD_START - (2 - k) * STAGGER) / SWING)
        out.append(fold - unfold)
    return out


def rotate(points, centre, angle):
    c, s = np.cos(angle), np.sin(angle)
    return (points - centre) @ np.array([[c, s], [-s, c]]) + centre


def pose(e, d, g):
    """The four pieces with hinges E, D, G turned by e, d, g half turns."""
    p1 = rotate(PIECES[0], E, -np.pi * e)
    p4 = rotate(PIECES[3], D, np.pi * d)
    g4 = rotate(G, D, np.pi * d)
    p3 = rotate(rotate(PIECES[2], D, np.pi * d), g4, np.pi * g)
    return [p1, PIECES[1], p3, p4], [E, D, g4]


def area_centroid(polys):
    total, moment = 0.0, np.zeros(2)
    for p in polys:
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        a = cross.sum() / 2
        total += a
        moment += np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / 6
    return moment / total


def tilings():
    """The triangle tiling and the square tiling, each cell cut into the four pieces."""
    triangle_cells, square_cells = [], []
    square, _ = pose(1, 1, 1)
    across, up = -U * SIDE, np.array([-U[1], U[0]]) * SIDE
    for i in range(-7, 8):
        for j in range(-7, 8):
            o = i * np.array([2.0, 0.0]) + j * np.array([1.0, R3])
            triangle_cells += [p + o for p in PIECES]
            triangle_cells += [rotate(p, D, np.pi) + o for p in PIECES]
            square_cells += [p + i * across + j * up for p in square]
    return triangle_cells, square_cells


def soundtrack(style_name, rng):
    """
    Events from the same hinge schedule as the picture: every hinge strikes a soft
    note when it starts to swing, a softer one (panned opposite) as it passes the
    half turn, and a louder one when its piece lands (the end of its smootherstep). Folding, hinges E, D, G let go on degrees 2, 4, 6 and land
    CLOSE_STEP higher, an ascending figure into the square; unfolding (G, D, E)
    reverses it. When the last piece lands (square closed, triangle closed) a bass
    note sounds two octaves down. Underneath, a pad holds the home triad while the
    figure is a triangle and the IV triad while it is a square, crossfading during
    the swings. All onsets are loop-periodic and every effect circular.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    notes = 0
    for k in range(3):
        base = HINGE_DEGREES[k]
        fold = FOLD_START + k * STAGGER
        unfold = UNFOLD_START + (2 - k) * STAGGER
        for start, let_go, land in (
            (fold, base, base + CLOSE_STEP),
            (unfold, base + CLOSE_STEP, base),
        ):
            soft = style.voice(scale.hz(let_go), 0.28)
            half = style.voice(scale.hz((let_go + land + 1) // 2), 0.2)
            loud = style.voice(scale.hz(land), 0.5 + 0.08 * k)
            audio.add(track, audio.pan(soft, HINGE_PAN[k]), start)
            audio.add(track, audio.pan(half, -HINGE_PAN[k]), start + SWING / 2)
            audio.add(track, audio.pan(loud, HINGE_PAN[k]), start + SWING)
            notes += 3
    square_closed = FOLD_START + 2 * STAGGER + SWING
    triangle_closed = UNFOLD_START + 2 * STAGGER + SWING
    for when, degree in ((square_closed, 3), (triangle_closed, 0)):
        bass = style.bass(scale.hz(degree, octave=-2), 0.55)
        audio.add(track, audio.pan(bass, 0.0), when)
    for degree, start, held in (
        # each chord swells in mid-swing and is held to the middle of the next swing
        (3, FOLD_START + SWING / 2, UNFOLD_START - FOLD_START),
        (0, UNFOLD_START + SWING / 2, DURATION - UNFOLD_START + FOLD_START),
    ):
        for midi in scale.chord(degree, 3, octave=-1):
            bed = audio.pad(float(audio.midi_to_hz(midi)), held, 0.13, 2.0, 2.5, 0.3)
            audio.add(track, bed, start)
    logger.info(f"Soundtrack: {style_name}, root {scale.root}, {notes} hinge notes")
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    Dudeney's hinged dissection of an equilateral triangle into a square, as a loop.

    Triangle ABC of side 2 (area √3); the square has side 3^(1/4). D, E are the
    midpoints of BC and AC, F lies on AB with |DF| = 3^(1/4), G = F + (1, 0), and P, Q
    are the feet of the perpendiculars from E and G onto DF. The pieces AFPE, EPQDC,
    FGQP and GBDQ are hinged at E, D and G; turning hinge E by −π and hinges D and G by
    +π (G carried along by the D piece) closes the square without any overlap. Each
    hinge follows a smootherstep in seconds, staggered by STAGGER: fold from
    FOLD_START, unfold in reverse order from UNFOLD_START, rest on both shapes, so the
    loop closes exactly (frame 0 and t = 1 are the resting triangle). The whole figure
    turns by g·SQUARE_TILT (g the mean hinge progress) so the square lands
    axis-aligned, and is re-centred on its area centroid every frame.

    Behind it, the triangle tiling and the square tiling of the plane, each cell cut
    into the same four pieces, crossfade as (1 − g)³ and g³ in the figure's moving
    frame, so they show at the rests and step back during the swings. The scale is
    SCALE at rest and zooms out just enough to keep the open chain inside FIT (a
    periodic table of the needed scale, eroded and smoothed). Each
    frame is a pure function of t, drawn with cv2 at 2x on the thread pool: tiling
    lines and ECHOES ghost outlines at earlier times (additive), the filled pieces with
    lighter rims, the hinges, then INTER_AREA downsample, a quarter-resolution bloom
    and two LaTeX lines. --seed only permutes the piece colours (and seeds the
    soundtrack's root and room via settings.audio_rng).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    colours = [
        np.array(hex_to_bgr(PIECE_COLOURS[i]), np.float32) for i in rng.permutation(4)
    ]
    rims = [(0.55 * c + 0.45 * 255).tolist() for c in colours]
    fills = [c.tolist() for c in colours]
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    hinge = hex_to_bgr(HINGE)
    shift = 3
    one = 1 << shift
    W2, H2 = WIDTH * SS, HEIGHT * SS
    background = np.full((H2, W2, 3), hex_to_bgr(BACKGROUND), np.uint8)
    triangle_cells, square_cells = tilings()
    cell_colours = [colours[k % 4] for k in range(len(triangle_cells))]
    square_colours = [colours[k % 4] for k in range(len(square_cells))]

    top = latex_mask(r"$\mathrm{four\ pieces{,}\ three\ hinges}$", 46)
    bottom = latex_mask(r"$\frac{\sqrt{3}}{4}\,a^2\ =\ s^2$", 58)
    text = np.array(hex_to_bgr(TEXT), np.float32)

    def placed(seconds):
        e, d, g = progress(seconds)
        polys, hinges = pose(e, d, g)
        centre = area_centroid(polys)
        angle = (e + d + g) / 3 * SQUARE_TILT
        return polys, hinges, centre, angle, (e + d + g) / 3

    # The zoom: the largest scale (up to SCALE) at which the open chain fits inside
    # FIT, eroded and smoothed over ZOOM_WINDOW on a fine periodic grid of times.
    grid = np.arange(4000) / 4000 * DURATION
    need = np.empty(len(grid))
    for k, s in enumerate(grid):
        polys, _, centre, angle, _ = placed(s)
        local = np.abs(rotate(np.concatenate(polys), centre, angle) - centre).max(0)
        need[k] = min(SCALE, FIT[0] / local[0], FIT[1] / local[1])
    half = int(ZOOM_WINDOW / 2 / DURATION * len(grid))
    padded = np.concatenate([need[-half:], need, need[:half]])
    eroded = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).min(1)
    eroded = np.concatenate([eroded[-half:], eroded, eroded[:half]])
    kernel = np.hanning(2 * half + 1)
    zoom_table = np.convolve(eroded, kernel / kernel.sum(), mode="valid")

    def zoom(seconds):
        return float(np.interp(seconds % DURATION, grid, zoom_table, period=DURATION))

    def frame_transform(seconds):
        polys, hinges, centre, angle, mix = placed(seconds)
        scale = zoom(seconds)

        def to_px(points):
            local = rotate(np.asarray(points, float), centre, angle) - centre
            x = CENTRE[0] + scale * local[..., 0]
            y = CENTRE[1] - scale * local[..., 1]
            return np.round(np.stack([x, y], -1) * SS * one).astype(np.int32)

        return polys, hinges, to_px, mix

    def render(frame):
        seconds = frame / FRAME_COUNT * DURATION
        polys, hinges, to_px, mix = frame_transform(seconds)
        canvas = background.copy()
        glow = np.zeros_like(canvas)

        # The two tilings, crossfading.
        for cells, cols, weight in (
            (triangle_cells, cell_colours, (1 - mix) ** 3),
            (square_cells, square_colours, mix**3),
        ):
            if weight < 0.01:
                continue
            middle = to_px(np.array([p.mean(0) for p in cells])) / (SS * one)
            margin = 1.6 * zoom(seconds)
            visible = (
                (middle[:, 0] > -margin)
                & (middle[:, 0] < WIDTH + margin)
                & (middle[:, 1] > -margin)
                & (middle[:, 1] < HEIGHT + margin)
            )
            for p, c in zip(
                [p for p, v in zip(cells, visible) if v],
                [c for c, v in zip(cols, visible) if v],
            ):
                cv2.polylines(
                    glow,
                    [to_px(p)],
                    True,
                    (TILE_ALPHA * weight * c).tolist(),
                    SS,
                    cv2.LINE_AA,
                    shift,
                )

        # Ghost outlines at earlier times.
        for j in range(ECHOES, 0, -1):
            ghost, _, ghost_px, _ = frame_transform(seconds - j * ECHO_DT)
            level = 0.42 * (1 - j / (ECHOES + 1)) ** 1.5
            for p, c in zip(ghost, colours):
                cv2.polylines(
                    glow,
                    [ghost_px(p)],
                    True,
                    (level * c).tolist(),
                    SS,
                    cv2.LINE_AA,
                    shift,
                )
        canvas = cv2.add(canvas, glow)

        # The pieces.
        for p, fill, rim in zip(polys, fills, rims):
            pts = to_px(p)
            cv2.fillPoly(canvas, [pts], fill, cv2.LINE_AA, shift)
            cv2.polylines(canvas, [pts], True, rim, SS * 3, cv2.LINE_AA, shift)
        for h in hinges:
            cv2.circle(
                canvas, tuple(to_px(h)), 11 * SS * one, hinge, -1, cv2.LINE_AA, shift
            )

        image = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        small = cv2.resize(
            image, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA
        )
        small = cv2.GaussianBlur(np.maximum(small - bg, 0), (0, 0), 7)
        image += BLOOM * cv2.resize(
            small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        blit_mask(image, top, WIDTH // 2, 170, text, 0.8, anchor="centre")
        blit_mask(image, bottom, WIDTH // 2, HEIGHT - 300, text, 0.9, anchor="centre")
        return np.clip(image, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
