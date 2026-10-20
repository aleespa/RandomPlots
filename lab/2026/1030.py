"""
Harmony, Drawn

Six dots circle along the top, at speeds 1 to 6. Eleven circle down the
side, at speeds 1 to 11. Every cell of the table follows its column's dot
left and right and its row's dot up and down, and draws what it gets.

Equal speeds draw a circle. Speeds 2 and 1 draw a figure eight, 3 and 2 a
pretzel, and the more complicated the ratio, the more tangled the knot.
These ratios are musical intervals: 2:1 is an octave, 3:2 a fifth, 4:3 a
fourth. Feed two notes into an oscilloscope and this is exactly what it
draws, so a table of Lissajous figures is a picture of harmony. The colour
says how simple each ratio is: gold for the sweetest intervals, blue for
the harshest. The slow drift that makes every figure seem to turn in space
is the two notes sliding in and out of phase.

Listen to the top row: each dot plucks its overtone of one low note every
time it passes the top of its circle, the first once every five seconds,
the sixth six times as often. Together they play the harmonic series, the
same notes the table draws, over a drone on the fundamental.

Made with Python & OpenCV. Sound synthesized in NumPy.
#lissajous #harmonics #musictheory #mathart #randomplots
"""

import gc
from math import gcd

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    hex_to_bgr,
    palette_lut,
    render_frames,
)

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The table ----------------------------------------------------------------------
COLUMNS, ROWS = 6, 11  # frequencies 1..COLUMNS across the top, 1..ROWS down the side
CELL = 150  # px
RADIUS = 0.4  # of a cell: the figures' half-size
CYCLES = 3  # turns of the slowest dot per loop
DRIFT = 1  # turns of the phase between columns and rows per loop

# --- Look ---------------------------------------------------------------------------
BACKGROUND = "#05060a"
CONSONANCE = ["#ffd27a", "#ff8f8f", "#b48cff", "#5aa9ff"]  # simple ratios to complex
TRAIL = 0.16  # of the loop: how far back each comet reaches
TRAIL_POINTS, TRAIL_CHUNKS = 320, 10
LINE = 3  # px at 2x
HEAD = 5  # px at 2x: the pen's dot
GUIDE_LEVEL = 0.12  # brightness of the guide lines through the header dots
RING_LEVEL = 0.22  # brightness of the header circles
BLOOM_SHRINK, BLOOM_SIGMA, BLOOM_GAIN = 4, 3.0, 0.7

# --- Sound -------------------------------------------------------------------------
STYLE = "felt_piano"  # common.music.STYLES; an --audio-only tag naming one overrides it
FUNDAMENTAL = -24  # semitones from the style's lowest root: partial 1
DRONE = (1, 2, 3)  # partials held underneath


def tenney(a: int, b: int) -> float:
    """log2(a b / gcd^2): 0 for a unison, larger for less consonant ratios."""
    g = gcd(a, b)
    return float(np.log2(a * b / g / g))


def layout():
    """Pixel centres of the header circles and the cells (the table is centred)."""
    left = (WIDTH - (COLUMNS + 1) * CELL) / 2
    top = (HEIGHT - (ROWS + 1) * CELL) / 2
    xs = left + CELL * (np.arange(COLUMNS + 1) + 0.5)  # xs[0] is the header column
    ys = top + CELL * (np.arange(ROWS + 1) + 0.5)  # ys[0] is the header row
    return xs, ys


def angles(t):
    """Base angle and phase drift at loop time t (any shape): both close at t = 1."""
    t = np.asarray(t, float)
    return 2 * np.pi * CYCLES * t, 2 * np.pi * DRIFT * t


def soundtrack(style_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    A loop in pure overtones of one fundamental f0 (FUNDAMENTAL semitones
    under the style's lowest root, the octave picked from the audio rng).

    - The plucks: column a's header dot is at the top of its circle when
      sin(a θ) = -1, i.e. at t = (3/4 + k) / (CYCLES a) for k = 0 .. CYCLES a - 1,
      and each time it plucks partial a (frequency a f0) on the style's voice:
      a 1:2:3:4:5:6 polyrhythm in the harmonic series, softer for higher
      partials, panned with the column.
    - The drone: a soft pad on the DRONE partials for the whole loop.
    All of it circular, so the sound loops with the frames.
    """
    style = STYLES[style_name]
    root = int(rng.choice(style.roots)) + FUNDAMENTAL
    f0 = float(audio.midi_to_hz(root))
    f0 = 2 * f0 if f0 < 65 else f0  # phone speakers stop around 60 Hz
    track = audio.silence(DURATION)
    xs, _ = layout()
    plucks = 0
    for a in range(1, COLUMNS + 1):
        pan = 0.7 * (xs[a] - WIDTH / 2) / (WIDTH / 2)
        voice = style.voice(a * f0, 0.55 / np.sqrt(a))
        for k in range(CYCLES * a):
            when = DURATION * (0.75 + k) / (CYCLES * a)
            audio.add(track, audio.pan(voice, pan), when)
            plucks += 1
    for n in DRONE:
        audio.add(track, audio.pad(n * f0, DURATION, 0.09 / n, 2.0, 2.0, 0.25), 0.0)
    logger.info(f"Soundtrack: {style_name}, f0 {f0:.1f} Hz, {plucks} plucks per loop")
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    A Lissajous table: cell (a, b) traces x = cos(a θ), y = sin(b θ + δ).

    Column a's header dot runs round its circle at angle a θ, row b's at
    b θ + δ, with θ = 2π CYCLES t and δ = 2π DRIFT t over the loop t in
    [0, 1); every frequency is an integer, so all of it closes exactly at
    t = 1. A cell follows its column's x and its row's y. The ratio a:b is a
    just interval, and each cell is coloured by its Tenney height
    log2(a b / gcd(a, b)^2) through CONSONANCE: unisons and octaves gold,
    complex ratios blue. The drifting δ turns every figure as if in 3D.

    A frame is a pure function of t, drawn with cv2 at 2x on the thread pool:
    each pen's comet is the closed form at TRAIL_POINTS times reaching TRAIL
    of the loop back (so frame 0 already has full trails), drawn in
    TRAIL_CHUNKS pieces brightening toward the head; faint guide lines through
    the header dots, the header circles and the pens' dots on top. Downsampled
    with INTER_AREA, bloomed with a blur at 1/BLOOM_SHRINK size. Nothing is
    random: --seed does not change the picture. The posted soundtrack is
    felt_piano with --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    xs, ys = layout()
    r = RADIUS * CELL
    lut = palette_lut(CONSONANCE)
    worst = max(tenney(a, b) for a in range(1, COLUMNS + 1) for b in range(1, ROWS + 1))
    cells = []
    for b in range(1, ROWS + 1):
        for a in range(1, COLUMNS + 1):
            rgb = lut[int(round(255 * tenney(a, b) / worst))]
            cells.append((a, b, xs[a], ys[b], rgb[::-1].copy()))  # BGR
    background = np.array(hex_to_bgr(BACKGROUND), np.float32) / 255
    ramp = np.linspace(0.08, 1.0, TRAIL_CHUNKS) ** 1.6
    back = np.linspace(TRAIL, 0.0, TRAIL_POINTS)  # oldest first
    logger.info(f"{len(cells)} cells, worst Tenney height {worst:.2f}")

    def to2x(x, y):
        return np.round(np.stack([x, y], -1) * 2 * 4).astype(
            np.int32
        )  # 2 bits subpixel

    def pt(x, y):
        return tuple(int(v) for v in to2x(x, y))

    def render(frame: int) -> np.ndarray:
        t = frame / FRAME_COUNT
        theta, delta = angles(t)
        canvas = np.zeros((2 * HEIGHT, 2 * WIDTH, 3), np.uint8)

        # Guide lines through the header dots, then the header circles and dots.
        grey = int(255 * GUIDE_LEVEL)
        for a in range(1, COLUMNS + 1):
            x = xs[a] + r * np.cos(a * theta)
            p = to2x(np.array([x, x]), np.array([ys[0], ys[-1] + 0.5 * CELL]))
            cv2.line(
                canvas,
                tuple(map(int, p[0])),
                tuple(map(int, p[1])),
                (grey,) * 3,
                2,
                cv2.LINE_AA,
                2,
            )
        for b in range(1, ROWS + 1):
            y = ys[b] + r * np.sin(b * theta + delta)
            p = to2x(np.array([xs[0], xs[-1] + 0.5 * CELL]), np.array([y, y]))
            cv2.line(
                canvas,
                tuple(map(int, p[0])),
                tuple(map(int, p[1])),
                (grey,) * 3,
                2,
                cv2.LINE_AA,
                2,
            )
        ring = int(255 * RING_LEVEL)
        heads = []
        for a in range(1, COLUMNS + 1):
            cv2.circle(
                canvas, pt(xs[a], ys[0]), int(2 * 4 * r), (ring,) * 3, 2, cv2.LINE_AA, 2
            )
            heads.append(
                (
                    xs[a] + r * np.cos(a * theta),
                    ys[0] + r * np.sin(a * theta),
                    (255,) * 3,
                )
            )
        for b in range(1, ROWS + 1):
            cv2.circle(
                canvas, pt(xs[0], ys[b]), int(2 * 4 * r), (ring,) * 3, 2, cv2.LINE_AA, 2
            )
            heads.append(
                (
                    xs[0] + r * np.cos(b * theta + delta),
                    ys[b] + r * np.sin(b * theta + delta),
                    (255,) * 3,
                )
            )

        # The comets.
        th, de = angles(t - back)
        for a, b, cx, cy, bgr in cells:
            points = to2x(cx + r * np.cos(a * th), cy + r * np.sin(b * th + de))
            for k, piece in enumerate(
                np.array_split(np.arange(TRAIL_POINTS), TRAIL_CHUNKS)
            ):
                if k:  # overlap one point so the chunks join
                    piece = np.concatenate([[piece[0] - 1], piece])
                colour = tuple(float(c) for c in 255 * ramp[k] * bgr)
                cv2.polylines(
                    canvas, [points[piece]], False, colour, LINE, cv2.LINE_AA, 2
                )
            heads.append(
                (
                    cx + r * np.cos(a * theta),
                    cy + r * np.sin(b * theta + delta),
                    tuple(float(c) for c in 255 * (0.5 + 0.5 * bgr)),
                )
            )
        for x, y, colour in heads:
            cv2.circle(canvas, pt(x, y), HEAD * 4, colour, -1, cv2.LINE_AA, 2)

        image = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255
        small = cv2.resize(
            image,
            (WIDTH // BLOOM_SHRINK, HEIGHT // BLOOM_SHRINK),
            interpolation=cv2.INTER_AREA,
        )
        bloom = cv2.resize(
            cv2.GaussianBlur(small, (0, 0), BLOOM_SIGMA), (WIDTH, HEIGHT)
        )
        image = background + image + BLOOM_GAIN * bloom
        return np.clip(image, 0, 1) * 255

    render_frames(settings, render, FRAME_COUNT)
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
