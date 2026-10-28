"""
The Map of All Julia Sets

At the top, the Mandelbrot set, with a small dot walking round its edge.
Below, the fractal that dot points to: as it moves, the shape below turns
into a rabbit, a basilica, a seahorse's tail, and every time the dot steps
outside the black shape, it bursts into dust.

Both come from one rule: take a number z, square it, add c, repeat. For each
c, the points z that never fly away form a Julia set. The Mandelbrot set is
the map of all of them: pick c inside the black shape and its Julia set is
one connected piece; pick c outside and it shatters into infinitely many
specks. And near the edge, each Julia set looks like the Mandelbrot set right
where the dot is standing.

Made with Python & Numba.
#juliaset #mandelbrot #fractal #mathart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from numba import njit

from common.image_processing import ImageProcessingSettings
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
)

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The walk -----------------------------------------------------------------------
STRAY, WEAVES = (
    0.03,
    5,
)  # c = boundary(θ) + STRAY sin(WEAVES θ) along the outward normal
MAX_ITER = 256
BAILOUT = 1000.0  # as in the Java original
SETTLED = 1e-12  # an orbit back within this (squared) of a checkpoint is inside
BAND = 1.75  # units above and below the Julia centre that are computed

# --- Layout -------------------------------------------------------------------------
JULIA_CENTRE = (WIDTH / 2, 0.62 * HEIGHT)  # px
JULIA_SCALE = 330.0  # px per unit
MAP_CENTRE, MAP_SCALE = (WIDTH / 2, 0.17 * HEIGHT), 200.0  # the Mandelbrot inset
MAP_OFFSET = -0.6  # the inset's centre, real axis

# --- Look ---------------------------------------------------------------------------
PALETTE = ["#021a1a", "#0e6f6a", "#5fe3c0", "#f3fff8"]
FILAMENT, HALO = 1.1, 9.0  # px: widths of the boundary glow
HALO_GAIN, HAZE_GAIN = 0.35, 0.25
MAP_LEVEL = 0.55  # the inset's brightness
DOT, TRAIL, TRAIL_POINTS = 7, 0.06, 40  # px, loop fraction, samples


@njit(nogil=True, fastmath=True, cache=True)
def escape(cx, cy, xs, ys, julia):
    """
    Escape time and distance estimate on the grid xs x ys (a port of
    Julia.java's per-pixel early exit; single-threaded and GIL-free, since
    whole frames run in parallel on the thread pool). julia: z0 = pixel,
    fixed c; otherwise Mandelbrot: z0 = 0, c = pixel. Returns (smooth count,
    distance), count -1 and distance 0 inside.
    """
    rows, cols = len(ys), len(xs)
    count = np.empty((rows, cols), np.float32)
    distance = np.empty((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            if julia:
                zx, zy, ax, ay = xs[j], ys[i], cx, cy
                dx, dy = 1.0, 0.0
            else:
                zx, zy, ax, ay = 0.0, 0.0, xs[j], ys[i]
                dx, dy = 0.0, 0.0
            k = 0
            m2 = zx * zx + zy * zy
            px, py, checkpoint = zx, zy, 8
            settled = False
            while k < MAX_ITER and m2 <= BAILOUT:
                # dz <- 2 z dz (+ 1 for the Mandelbrot set)
                ndx = 2.0 * (zx * dx - zy * dy) + (0.0 if julia else 1.0)
                dy = 2.0 * (zx * dy + zy * dx)
                dx = ndx
                zx, zy = zx * zx - zy * zy + ax, 2.0 * zx * zy + ay
                m2 = zx * zx + zy * zy
                k += 1
                # Periodicity check: an orbit that returns to its last
                # checkpoint has settled on a cycle and never escapes.
                if (zx - px) ** 2 + (zy - py) ** 2 < SETTLED:
                    settled = True
                    break
                if k == checkpoint:
                    px, py, checkpoint = zx, zy, 2 * checkpoint
            if settled or m2 <= BAILOUT:
                count[i, j] = -1.0
                distance[i, j] = 0.0
            else:
                m = np.sqrt(m2)
                count[i, j] = k + 1.0 - np.log(np.log(m)) / np.log(2.0)
                distance[i, j] = m * np.log(m) / max(np.sqrt(dx * dx + dy * dy), 1e-30)
    return count, distance


def walk(t):
    """
    c at loop time t: on the main cardioid's edge, e^{iθ}/2 - e^{2iθ}/4,
    nudged STRAY sin(WEAVES θ) along its outward normal -i e^{3iθ/2} (out of
    the set, then back in). The nudge vanishes at the cusp, θ = 0, where the
    normal flips, so the walk stays continuous and the loop closes.
    """
    theta = 2 * np.pi * np.mod(np.asarray(t, float), 1.0)
    w = np.exp(1j * theta)
    normal = -1j * np.exp(1.5j * theta)
    return w / 2 - w * w / 4 + STRAY * np.sin(WEAVES * theta) * normal


def glow(count, distance, scale):
    """Light in [0, 1]: a crisp filament and halo from the distance, a faint haze."""
    d = distance * scale  # px
    light = np.exp(-d / np.float32(FILAMENT)) + np.float32(HALO_GAIN) * np.exp(
        -d / np.float32(HALO)
    )
    haze = HAZE_GAIN * np.clip(count / 40.0, 0, 1) ** 1.5  # only close to the set
    return np.clip(np.where(count < 0, 0.0, light + haze), 0, 1)


def generate(settings: ImageProcessingSettings = None):
    """
    Julia sets of z -> z^2 + c, with c walking the edge of the Mandelbrot
    set's main cardioid (`walk`): c(θ) = e^{iθ}/2 - e^{2iθ}/4 plus
    STRAY sin(WEAVES θ) along the outward normal, θ = 2πt, so it steps out of
    the set WEAVES times per loop (Julia set dust) and back in (connected),
    and the loop closes exactly.

    Each frame computes the Julia set with `escape`, a numba port of the
    user's Julia.java kernel (per-pixel early exit at |z|^2 > BAILOUT),
    extended with the derivative dz for the distance estimate
    |z| ln|z| / |dz| and a periodicity check (checkpoints at doubling
    iterations) that stops settled interior orbits early. Only the band BAND
    units round the Julia centre is computed, and whole frames run in
    parallel on the thread pool. The boundary is drawn from the distance (a
    FILAMENT-px line plus a HALO-px glow), the exterior gets a faint haze
    that grows close to the set, the interior stays dark. The Mandelbrot inset
    is computed once, drawn dim, with a dot at c and a fading trail of its past
    positions. Nothing is random: --seed does not change the picture.
    """
    settings = settings or ImageProcessingSettings(1)
    lut = (palette_lut(PALETTE, 1024)[:, ::-1] * 255).astype(np.float32)  # BGR
    xs = (np.arange(WIDTH) - JULIA_CENTRE[0] + 0.5) / JULIA_SCALE
    ys = (JULIA_CENTRE[1] - np.arange(HEIGHT) - 0.5) / JULIA_SCALE

    band = slice(
        max(0, int(JULIA_CENTRE[1] - BAND * JULIA_SCALE)),
        min(HEIGHT, int(JULIA_CENTRE[1] + BAND * JULIA_SCALE)),
    )

    # The inset, once.
    half_w, half_h = 1.6, 1.15
    mx = MAP_OFFSET + np.arange(-half_w * MAP_SCALE, half_w * MAP_SCALE) / MAP_SCALE
    my = -np.arange(-half_h * MAP_SCALE, half_h * MAP_SCALE) / MAP_SCALE
    count, distance = escape(0.0, 0.0, mx, my, False)
    inside = (count < 0).astype(np.float32)
    inset = np.maximum(
        MAP_LEVEL * 0.35 * inside, MAP_LEVEL * glow(count, distance, MAP_SCALE)
    )
    top = int(MAP_CENTRE[1] - half_h * MAP_SCALE)
    left = int(MAP_CENTRE[0] - half_w * MAP_SCALE)
    inset_box = (slice(top, top + len(my)), slice(left, left + len(mx)))

    def on_map(c):
        x = MAP_CENTRE[0] + (np.real(c) - MAP_OFFSET) * MAP_SCALE
        y = MAP_CENTRE[1] - np.imag(c) * MAP_SCALE
        return x, y

    escape(0.3, 0.0, xs[:8], ys[:8], True)  # compile before timing
    logger.info(f"c starts at {complex(walk(0.0)):.3f}; inset {inset.shape}")

    def render(frame: int) -> np.ndarray:
        t = frame / FRAME_COUNT
        c = complex(walk(t))
        light = np.zeros((HEIGHT, WIDTH))
        count, distance = escape(c.real, c.imag, xs, ys[band], True)
        light[band] = glow(count, distance, JULIA_SCALE)
        light[inset_box] = np.maximum(light[inset_box], inset)
        image = lut[(light * 1023).astype(np.int32)]

        # The dot and its trail, at 4x subpixel precision.
        past = walk(t - TRAIL * np.linspace(1, 0, TRAIL_POINTS))
        px, py = on_map(past)
        points = np.round(np.stack([px, py], 1) * 4).astype(np.int32)
        for k in range(TRAIL_POINTS - 1):
            level = 255.0 * (k + 1) / TRAIL_POINTS
            cv2.line(
                image,
                tuple(map(int, points[k])),
                tuple(map(int, points[k + 1])),
                (level,) * 3,
                2,
                cv2.LINE_AA,
                2,
            )
        x, y = on_map(c)
        cv2.circle(
            image, (int(x * 4), int(y * 4)), DOT * 4, (255.0,) * 3, -1, cv2.LINE_AA, 2
        )
        return image

    render_frames(settings, render, FRAME_COUNT)
    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
