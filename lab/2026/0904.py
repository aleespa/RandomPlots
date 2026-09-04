"""Double Pendulum Bloom -- a 60 fps portrait study of sensitive dependence."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from common.image_processing import ImageProcessingSettings


FPS = 60
SECONDS = 36
FRAMES = FPS * SECONDS
# Simulation seconds advanced for each displayed second. Lower values make the
# pendulums evolve more slowly without sacrificing the 60 fps output cadence.
MOTION_RATE = 0.5
PALETTE = ("#FFF5E1", "#F6C445", "#E84855", "#313A7D")
WIDTH, HEIGHT = 1080, 1920
ASPECT = WIDTH / HEIGHT
N_WORKERS = min(os.cpu_count() or 1, 16)

_STATE = {}


def _palette_colours(palette: tuple[str, ...], count: int) -> np.ndarray:
    """Linearly interpolate the hex palette without constructing a colourmap."""
    anchors = np.array(
        [tuple(bytes.fromhex(colour.removeprefix("#"))) for colour in palette],
        dtype=np.float32,
    ) / 255.0
    positions = np.linspace(0.0, len(anchors) - 1, count, dtype=np.float32)
    left = np.floor(positions).astype(np.intp)
    right = np.minimum(left + 1, len(anchors) - 1)
    fraction = (positions - left)[:, None]
    return anchors[left] * (1.0 - fraction) + anchors[right] * fraction


def _simulate(frames: int, count: int, dt: float, centre: np.ndarray, spread: float):
    """Simulate a tight family of equal-mass, unequal-arm double pendulums."""
    l1, l2, gravity = 1.0, 0.72, 9.81
    offsets = np.linspace(-spread, spread, count)
    theta1 = centre[0] + offsets
    theta2 = centre[1] - offsets
    omega1 = np.zeros(count)
    omega2 = np.zeros(count)
    x = np.empty((frames, count), dtype=np.float32)
    y = np.empty_like(x)

    for frame in range(frames):
        delta = theta1 - theta2
        denominator_1 = l1 * (2.0 - np.cos(2.0 * delta))
        denominator_2 = l2 * (2.0 - np.cos(2.0 * delta))
        alpha1 = (
            -2.0 * gravity * np.sin(theta1)
            - gravity * np.sin(theta1 - 2.0 * theta2)
            - 2.0
            * np.sin(delta)
            * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
        ) / denominator_1
        alpha2 = (
            2.0
            * np.sin(delta)
            * (omega1**2 * l1 + gravity * np.cos(theta1) + omega2**2 * l2 * np.cos(delta))
        ) / denominator_2
        # Semi-implicit Euler preserves the energetic, hand-drawn quality.
        omega1 += alpha1 * dt
        omega2 += alpha2 * dt
        theta1 += omega1 * dt
        theta2 += omega2 * dt
        x[frame] = l1 * np.sin(theta1) + l2 * np.sin(theta2)
        y[frame] = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
    return x, y


def _splat(px: np.ndarray, py: np.ndarray, weights: np.ndarray, rgb: np.ndarray):
    """Bilinear additive rasterisation, matching the hairline treatment in 0902."""
    fx, fy = px * HEIGHT - 0.5, (1.0 - py) * HEIGHT - 0.5
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    dx, dy = fx - x0, fy - y0
    buffer = np.zeros((3, HEIGHT * WIDTH), dtype=np.float64)
    for ox, oy, share in (
        (0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy), (1, 1, dx * dy),
    ):
        xi, yi = x0 + ox, y0 + oy
        valid = (xi >= 0) & (xi < WIDTH) & (yi >= 0) & (yi < HEIGHT)
        pixels = (yi[valid] * WIDTH + xi[valid]).ravel()
        contribution = (share * weights)[valid].ravel()
        for channel in range(3):
            buffer[channel] += np.bincount(
                pixels, contribution * rgb[channel][valid].ravel(), minlength=HEIGHT * WIDTH
            )
    return buffer.reshape(3, HEIGHT, WIDTH)


def _tone_map(buffer: np.ndarray, gain: float, bloom_weight: float, bloom_sigma: float):
    image = 1.0 - np.exp(-gain * buffer)
    bloom = np.stack([gaussian_filter(channel, bloom_sigma) for channel in image])
    return np.clip(image + bloom_weight * bloom, 0, 1).transpose(1, 2, 0).astype(np.float32)


def _densify(px: np.ndarray, py: np.ndarray, target_step_px: float = 0.65):
    """Fill gaps at sub-pixel spacing, even during the pendulum's fast swings."""
    if px.shape[0] < 2:
        return px, py
    largest_step = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)).max() * HEIGHT
    samples_per_segment = max(1, int(np.ceil(largest_step / target_step_px)))
    phase = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    dense_x = px[:-1, None, :] + phase[None, :, :] * (px[1:, None, :] - px[:-1, None, :])
    dense_y = py[:-1, None, :] + phase[None, :, :] * (py[1:, None, :] - py[:-1, None, :])
    return (
        np.concatenate((dense_x.reshape(-1, px.shape[1]), px[-1:])),
        np.concatenate((dense_y.reshape(-1, py.shape[1]), py[-1:])),
    )


def _render_buffer(index, x, y, colours, trail):
    """Continuously rasterised pendulum trails that fade smoothly with age."""
    start = max(0, index - trail)
    # The view is deliberately zoomed out: the bob's full +/-1.72 reach sits
    # well inside this -2.05..2.05 by -3.65..3.65 portrait viewport.
    px = (x[start : index + 1] + 2.05) / 4.10 * ASPECT
    py = (y[start : index + 1] + 3.65) / 7.30
    px, py = _densify(px, py)
    points = px.shape[0]
    distance = np.linspace(-1.0, 0.0, points)[:, None]
    # The newest position is bright; older parts of the trajectory dissolve
    # evenly to the background rather than breaking into pulsing dots.
    weights = (0.04 + 0.96 * np.cos(0.5 * np.pi * distance) ** 2).astype(np.float32)
    rgb = np.broadcast_to(colours.T[:, None, :], (3, points, x.shape[1]))
    # Slightly broaden the physical one-pixel splat before the larger halo is
    # added by _tone_map. This makes each continuous path visibly line-like.
    return np.stack([gaussian_filter(channel, 0.58) for channel in _splat(px, py, weights, rgb)])


def _init_worker(state):
    """Install the read-only animation state once per rendering process."""
    _STATE.update(state)
    _STATE["frames_path"] = Path(state["frames_path"])


def _render_chunk(frame_indices):
    """Render a contiguous run of frames directly to PNG files."""
    state = _STATE
    for index in frame_indices:
        buffer = _render_buffer(
            index, state["x"], state["y"], state["colours"], state["trail"]
        )
        rgb = _tone_map(buffer, state["gain"], state["bloom_weight"], state["bloom_sigma"])
        # OpenCV encodes BGR uint8 PNGs directly; no figure, renderer, or
        # Matplotlib process state is involved.
        frame = np.ascontiguousarray(np.rint(rgb * 255.0).astype(np.uint8)[..., ::-1])
        if not cv2.imwrite(str(state["frames_path"] / f"frame{index:04d}.png"), frame):
            raise OSError(f"Could not write frame {index}")


def generate(settings: ImageProcessingSettings = None):
    """Render a 12-second, 60 fps bloom of diverging double-pendulum trails."""
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # The narrow initial-angle band is the key lever: increase it for earlier
    # disorder; reduce it for a longer, more recognisably petal-like opening.
    pendulums = 280  # denser family so divergence becomes a woven bloom, not dots
    trail = 96  # 1.6 seconds of smoothly dimming history
    centre = np.array([2.19, 1.37]) + rng.uniform(-0.025, 0.025, 2)
    x, y = _simulate(FRAMES, pendulums, MOTION_RATE / FPS, centre, spread=0.015)
    colours = _palette_colours(PALETTE, pendulums)
    bloom_weight, bloom_sigma = 0.34, 5.5
    # A single exposure for the whole clip prevents chaotic density changes
    # from making the image visibly brighten or dim between frames.
    probes = [_render_buffer(index, x, y, colours, trail) for index in (120, 360, 600)]
    gain = 2.8 / max(float(np.max([np.percentile(buffer.max(axis=0), 99.7) for buffer in probes])), 1e-6)

    state = dict(
        x=x,
        y=y,
        colours=colours,
        trail=trail,
        gain=gain,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=str(settings.frames_path),
    )
    n_workers = min(N_WORKERS, FRAMES)
    chunks = [chunk for chunk in np.array_split(np.arange(FRAMES), n_workers * 4) if len(chunk)]
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(state,)
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
