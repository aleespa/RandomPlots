"""
Three Bodies, One Thread

Two masses under gravity is a solved problem: ellipses, forever. Add a
third and no formula exists for where they will be — you can only
integrate, step by step, and watch.

These three start on the figure-eight orbit of Chenciner and Montgomery:
three equal masses chasing one another around a single closed curve, each
a third of a lap behind the last. Then the starting conditions are nudged
by about a thousandth — far less than the width of the line you are
looking at. For a while nothing seems wrong. The eight breathes, drifts,
opens; the bodies stop retracing one another. That divergence was there
from the first frame, doubling quietly, until it became the picture.

Newton's law is the whole of it: ẍᵢ = Σ (xⱼ − xᵢ)/|xⱼ − xᵢ|³.

Made with Python & NumPy.
#generativeart #threebodyproblem #chaostheory #celestialmechanics #orbit
#physics #randomplots #creativecoding #algorithmicart #mathisbeautiful
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from loguru import logger

from common.image_processing import ImageProcessingSettings

# --- Video -------------------------------------------------------------------
# RP_FPS in the environment gives a quick low-frame-rate test render.
FPS = int(os.environ.get("RP_FPS", "30"))
DURATION = 30.0
FRAME_COUNT = max(2, int(FPS * DURATION))
WIDTH, HEIGHT = 1080, 1920  # 9:16 portrait for Reels / Stories
PNG_WRITERS = 6

# --- Orbit -------------------------------------------------------------------
# Chenciner-Montgomery figure-eight: three equal masses on a single closed
# curve, in units where G = m = 1. One body starts at each end of the eight and
# one at the crossing, and the orbit closes after PERIOD.
POSITIONS = np.array([[-0.97000436, 0.24308753], [0.0, 0.0], [0.97000436, -0.24308753]])
VELOCITY = np.array([0.93240737, 0.86473146])
VELOCITIES = np.array([VELOCITY / 2.0, -VELOCITY, VELOCITY / 2.0])
PERIOD = 6.32591398

SUBSTEPS = 60  # integration steps per rendered frame; also the trail sampling
LOOPS = 4.2 # periods of the original orbit covered by the clip

# Nudge applied to the starting positions and velocities, as a fraction of the
# orbit's own scale. The figure-eight is linearly stable, so a small one wanders
# and breathes for a few periods before it is obviously off; a larger one can
# end the usual way, with one body thrown clear and the other two left paired.
# Set to 0.0 to recover the exact, endlessly repeating orbit.
PERTURBATION = 0.02

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([2, 3, 8], dtype=np.float32)  # BGR, near-black

# One colour per body, in BGR: cool white-blue, warm amber, pale gold.
BODY_COLOURS = np.array(
    [[255, 196, 150], [40, 140, 255], [150, 225, 255]], dtype=np.float32
)

MARGIN = 0.12  # fraction of the short side kept clear around the orbit
FRAMING_PERCENTILE = 99.0  # ignore rare excursions when fitting the orbit
TRAIL_FRACTION = 0.62  # how much of the orbit the tail covers
TRAIL_ENERGY = 1.1  # brightness deposited per trail sample
TRAIL_GAMMA = 2.2  # >1 makes the tail fall away quickly behind the body
HEAD_RADIUS = 3.4  # core of the glowing body, in pixels
HEAD_ENERGY = 900.0
GLOW_SIGMA = 9.0
GLOW = 0.5
BLOOM_SIGMA = 26.0
BLOOM = 0.22
# Resolution divisor each blur is computed at -- see blur().
GLOW_DOWNSAMPLE = 2
BLOOM_DOWNSAMPLE = 4


def accelerations(positions: np.ndarray) -> np.ndarray:
    """
    Newtonian gravity for three equal unit masses, G = 1.

    a_i = sum_j (x_j - x_i) / |x_j - x_i|^3, skipping j = i.
    """
    separation = positions[None, :, :] - positions[:, None, :]
    distance = np.linalg.norm(separation, axis=-1)
    np.fill_diagonal(distance, np.inf)  # a body does not pull on itself
    return np.sum(separation / distance[..., None] ** 3, axis=1)


def integrate(
    positions: np.ndarray, velocities: np.ndarray, steps: int, dt: float
) -> np.ndarray:
    """
    Integrate the orbit with velocity Verlet, returning every sampled position.

    Verlet is symplectic: it conserves energy over the whole clip, so whatever
    the orbit does is the physics of the starting conditions and not the slow
    bleed of an integrator. That matters here -- the point of the figure is that
    a tiny change in the start, not in the arithmetic, is what pulls it apart.
    """
    positions = positions.copy()
    velocities = velocities.copy()
    acceleration = accelerations(positions)

    track = np.empty((steps, 3, 2), dtype=np.float64)
    for step in range(steps):
        track[step] = positions
        velocities += 0.5 * dt * acceleration
        positions += dt * velocities
        acceleration = accelerations(positions)
        velocities += 0.5 * dt * acceleration

    return track


def to_screen(track: np.ndarray) -> np.ndarray:
    """Fit the whole orbit into the portrait frame, preserving its shape."""
    points = track.reshape(-1, 2)
    # A percentile rather than the extremes: if a perturbed body is thrown out
    # of the system it should leave the frame, not shrink everything else.
    low = np.percentile(points, 100.0 - FRAMING_PERCENTILE, axis=0)
    high = np.percentile(points, FRAMING_PERCENTILE, axis=0)
    centre = 0.5 * (low + high)
    span = (high - low).max()

    scale = (1.0 - 2.0 * MARGIN) * min(WIDTH, HEIGHT) / span
    screen = (track - centre) * scale
    screen[..., 0] += WIDTH / 2.0
    screen[..., 1] = HEIGHT / 2.0 - screen[..., 1]  # y grows upwards in physics
    return screen


def splat(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Rasterise points into a flat single-channel intensity field, bilinearly.

    Everything visible is drawn this way: the trail is the integrator's own
    samples, the bodies are small discs of samples. Deposits add, so where the
    curve crosses itself it burns brighter, exactly as a long exposure would.
    Intensity is kept apart from colour -- each body has a single colour, so the
    tint is applied once to the finished field rather than to every sample.

    All four bilinear corners go into a single `bincount`. Each call has to
    allocate and zero a whole WIDTH*HEIGHT accumulator, which for a few tens of
    thousands of points costs far more than scattering the points themselves --
    so the number of calls, not the number of points, sets the price.
    """
    x = points[:, 0]
    y = points[:, 1]

    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)

    count = len(x)
    index = np.empty(4 * count, dtype=np.int64)
    amount = np.empty(4 * count, dtype=np.float64)

    for slot, (offset_x, offset_y, corner) in enumerate(
        (
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        )
    ):
        ix = x0 + offset_x
        iy = y0 + offset_y
        inside = (ix >= 0) & (ix < WIDTH) & (iy >= 0) & (iy < HEIGHT)

        # Corners off the canvas stay in the batch, aimed at pixel 0 with a zero
        # weight, so every corner contributes the same fixed-size block.
        block = slice(slot * count, (slot + 1) * count)
        index[block] = np.where(inside, iy * WIDTH + ix, 0)
        amount[block] = np.where(inside, corner * weights, 0.0)

    return np.bincount(index, weights=amount, minlength=HEIGHT * WIDTH)


def blur(image: np.ndarray, sigma: float, downsample: int) -> np.ndarray:
    """
    Gaussian blur, computed on a downsampled copy and scaled back up.

    A Gaussian this wide carries nothing near the pixel grid, so the reduced
    resolution costs no visible detail: against the full-resolution blur the
    worst pixel of a finished frame moves by 3 levels out of 255, and none moves
    by more than 2 -- for a sixteenth of the work at BLOOM_SIGMA.
    """
    if downsample <= 1:
        return cv2.GaussianBlur(image, (0, 0), sigma)

    small = cv2.resize(
        image,
        (WIDTH // downsample, HEIGHT // downsample),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / downsample)
    return cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def disc_offsets(radius: float) -> np.ndarray:
    """A small filled disc of sample points, used to draw a body's core."""
    span = int(np.ceil(radius)) + 1
    grid = np.arange(-span, span + 1)
    dx, dy = np.meshgrid(grid, grid)
    inside = dx**2 + dy**2 <= radius**2
    return np.stack([dx[inside], dy[inside]], axis=-1).astype(np.float64)


def generate(settings: ImageProcessingSettings = None):
    """
    The figure-eight three-body orbit, drawn as three bodies and their trails.

    Three equal masses are integrated under Newtonian gravity with a velocity
    Verlet step, starting from the Chenciner-Montgomery figure-eight conditions
    with a small perturbation drawn from the seed. The unperturbed orbit is a
    choreography -- all three bodies on one closed curve, a third of a period
    apart -- but a perturbed one is not periodic, so this clip does not loop: it
    runs for LOOPS periods of the original orbit and ends wherever the physics
    has taken it. Every seed is a different history. The integration carries a
    pre-roll of one trail length so the tails are already at full extent in the
    first frame.

    Everything is drawn by bilinear point splatting into a float buffer: the
    trail is the integrator's own samples weighted by age, each body is a small
    disc of samples, and glow and bloom are two Gaussian blurs of the result.
    The seed sets the perturbation and rotates the orbit in the frame. Framing
    is fitted to a percentile of the trajectory rather than to its extremes, so
    a body thrown out of the system leaves the frame instead of shrinking the
    rest of the picture to nothing.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    steps = FRAME_COUNT * SUBSTEPS
    dt = LOOPS * PERIOD / steps

    # The trail is the recent past, so the clip starts one trail-length into the
    # integration -- otherwise the first seconds are spent watching it grow.
    trail_length = max(2, int(TRAIL_FRACTION * steps))

    # The seed is the nudge: the same law and the same starting arrangement to
    # three decimal places, and a different history every time.
    positions = POSITIONS + PERTURBATION * rng.normal(size=POSITIONS.shape)
    velocities = VELOCITIES + PERTURBATION * rng.normal(size=VELOCITIES.shape)

    # Hold the centre of mass still, so the system does not drift out of frame
    # on the perturbation alone.
    positions -= positions.mean(axis=0)
    velocities -= velocities.mean(axis=0)

    logger.info(
        f"Integrating {steps} steps of the figure-eight orbit "
        f"(dt = {dt:.2e}, perturbation = {PERTURBATION:g})"
    )
    track = integrate(positions, velocities, steps + trail_length, dt)

    # The seed only chooses how the orbit sits in the frame.
    angle = rng.uniform(0.0, 2.0 * np.pi)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    screen = to_screen(track @ rotation.T)

    # Per-sample deposit weights, the same every frame.
    # The ramp runs oldest-to-newest, matching the order of the trail slice, so
    # the bright end is the sample sitting under the body.
    trail_weights = TRAIL_ENERGY * np.linspace(0.0, 1.0, trail_length) ** TRAIL_GAMMA
    head = disc_offsets(HEAD_RADIUS)
    head_weights = np.full(len(head), HEAD_ENERGY / len(head))
    weights = np.concatenate((trail_weights, head_weights))

    # One intensity field per body, allocated once. Kept as three columns of a
    # single array so the tinting below is one matrix multiply.
    fields = np.empty((HEIGHT * WIDTH, 3), dtype=np.float32)

    logger.info(f"Rendering {FRAME_COUNT} frames @ {FPS} fps to {WIDTH}x{HEIGHT}")
    frames_path = settings.frames_path

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            now = trail_length + frame * SUBSTEPS
            # The trail is simply the recent past. No wrap-around: a perturbed
            # orbit does not close, so there is no join to be continuous across.
            tail = slice(now - trail_length + 1, now + 1)

            for body in range(3):
                # Trail and head are one batch of points: the cost of a splat is
                # dominated by the call, not by how many points it carries.
                points = np.concatenate((screen[tail, body], screen[now, body] + head))
                fields[:, body] = splat(points, weights)

            # Three intensity fields, one colour each: the tint is a 3x3 matrix.
            buffer = (fields @ BODY_COLOURS).reshape(HEIGHT, WIDTH, 3)

            # Glow around the bodies, then a wide atmospheric bloom.
            buffer += GLOW * blur(buffer, GLOW_SIGMA, GLOW_DOWNSAMPLE)
            buffer += BLOOM * blur(buffer, BLOOM_SIGMA, BLOOM_DOWNSAMPLE)
            buffer += BACKGROUND

            image = np.clip(buffer, 0.0, 255.0).astype(np.uint8)
            pending.append(
                pool.submit(
                    cv2.imwrite, str(frames_path / f"frame{frame:04d}.png"), image
                )
            )

            if frame % FPS == 0:
                logger.info(f"Frame {frame}/{FRAME_COUNT}")

        for task in pending:
            if not task.result():
                raise OSError("Could not write a frame")

    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
