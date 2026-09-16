"""
Truchet Breath

One tile: a square with two quarter-circles, each joining the midpoints
of two neighbouring sides. Lay it down in a grid and the arcs link up
across the joins into loops, meanders and long wandering paths. Turn any
tile a quarter turn and the paths reconnect a different way. Sébastien
Truchet drew the first of these in 1704 while thinking about floor
tiles; every generative artist since has drawn them again.

Here nothing is placed by hand. Two slow waves cross the grid and each
tile turns as they pass, dwelling in one state until the crest arrives
and then rolling over to the other. Between states the arcs part and
rejoin, so the maze is forever unmaking and remaking itself. The waves
complete whole cycles over the clip, so the last frame is the first.

Made with Python, PyTorch & NumPy.
#generativeart #truchet #tiling #maze #geometry #mathematics #pattern
#randomplots #creativecoding #algorithmicart #mathisbeautiful
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from loguru import logger

from common.image_processing import ImageProcessingSettings

# --- Video -------------------------------------------------------------------
# RP_FPS in the environment gives a quick low-frame-rate test render.
FPS = int(os.environ.get("RP_FPS", "30"))
DURATION = 20.0
FRAME_COUNT = max(2, int(FPS * DURATION))
WIDTH, HEIGHT = 1080, 1920  # 9:16 portrait for Reels / Stories
PNG_WRITERS = 6
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

# --- Tiles -------------------------------------------------------------------
COLUMNS = 9  # tiles across; rows follow from the aspect ratio
STROKE = 0.16  # main arc stroke width as a fraction of the tile
THIN_STROKE = 0.035  # the two thin companion arcs
THIN_OFFSET = 0.20  # radial offset of the companions from the main arc
EDGE_SOFTNESS = 1.2  # anti-aliasing width in pixels
FLIP_SHARPNESS = 5.0  # how abruptly a tile rolls over as a wave crest passes
# Waves: (cycles per loop, wavelength in tiles, direction angle in radians).
WAVES = (
    (1, 6.0, 0.55),
    (-1, 9.5, 2.05),
)

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([244, 240, 231], dtype=np.float32)  # warm off-white, RGB
# The main arc's colour follows the tile's wave value through this ramp.
PALETTE = np.array(
    [
        [0.11, 0.11, 0.16],  # ink
        [0.16, 0.30, 0.48],  # slate
        [0.75, 0.25, 0.22],  # brick
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
THIN_COLOUR = np.array([0.11, 0.11, 0.16], dtype=np.float32)
THIN_ALPHA = 0.55


def select_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("no CUDA device, rendering on CPU")
    return torch.device("cpu")


def palette_lut(device: torch.device) -> torch.Tensor:
    positions = np.linspace(0.0, 1.0, len(PALETTE))
    t = np.linspace(0.0, 1.0, PALETTE_SIZE)
    lut = np.stack([np.interp(t, positions, PALETTE[:, c]) for c in range(3)], axis=1)
    return torch.tensor(lut * 255.0, dtype=torch.float32, device=device)


def arc_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Distance, in tile units, from local offset (u, v) in [-1/2, 1/2]^2 to the
    nearer of the two quarter-circles centred on opposite corners.
    """
    d1 = (torch.hypot(u + 0.5, v + 0.5) - 0.5).abs()
    d2 = (torch.hypot(u - 0.5, v - 0.5) - 0.5).abs()
    return torch.minimum(d1, d2)


def generate(settings: ImageProcessingSettings = None):
    """
    A Truchet tiling whose tiles rotate under two travelling waves.

    Each tile carries two quarter-circle arcs centred on opposite corners.
    Its orientation is a quarter turn times a soft step of the local wave
    value: w = mean of two plane waves with integer cycles per loop, and
    s = sigmoid(FLIP_SHARPNESS * w), so a tile rests in one state or the
    other and rolls over as a crest passes. A random base parity per tile,
    fixed by the seed, decides which state each tile rests in first, so the
    resting tiling is itself a random Truchet maze.

    Rendering is per pixel: the offset from the tile centre is rotated by
    the tile's angle, the distance to the arcs is evaluated analytically,
    and stroke coverage is a one-pixel ramp on that distance. A wide main
    stroke, coloured by the wave value, and two thin companion arcs are
    composited over the off-white ground. Because the arcs stay inside their
    tile under rotation about its centre, no neighbour lookup is needed.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)

    logger.info(f"Rendering Truchet Breath: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)

    tile = WIDTH / COLUMNS
    rows = int(np.ceil(HEIGHT / tile))
    parity = torch.tensor(
        rng.integers(0, 2, size=(rows, COLUMNS)), dtype=torch.float32, device=device
    )

    py, px = torch.meshgrid(
        (torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5) / tile,
        (torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5) / tile,
        indexing="ij",
    )
    ti = torch.floor(px)
    tj = torch.floor(py)
    u0 = px - ti - 0.5
    v0 = py - tj - 0.5
    ii = ti.to(torch.long).clamp(0, COLUMNS - 1)
    jj = tj.to(torch.long).clamp(0, rows - 1)

    ci = torch.arange(COLUMNS, device=device, dtype=torch.float32) + 0.5
    cj = torch.arange(rows, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(cj, ci, indexing="ij")

    background = torch.tensor(BACKGROUND, device=device)
    thin_colour = torch.tensor(THIN_COLOUR * 255.0, device=device)
    soft = EDGE_SOFTNESS / tile  # anti-aliasing width in tile units

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            w = torch.zeros_like(grid_x)
            for cycles, wavelength, angle in WAVES:
                along = grid_x * np.cos(angle) + grid_y * np.sin(angle)
                w = w + torch.sin(2.0 * np.pi * (cycles * t - along / wavelength))
            w = w / len(WAVES)
            state = torch.sigmoid(FLIP_SHARPNESS * w)  # 0..1, resting near the ends
            angle = 0.5 * np.pi * (state + parity)
            shade = 0.5 * (1.0 + w)
            colour = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)]  # (rows, cols, 3)

            c = torch.cos(angle)[jj, ii]
            s = torch.sin(angle)[jj, ii]
            u = u0 * c + v0 * s
            v = -u0 * s + v0 * c

            d = arc_distance(u, v)
            main = torch.clamp((0.5 * STROKE - d) / soft + 0.5, 0.0, 1.0)
            thin = torch.clamp(
                (0.5 * THIN_STROKE - (d - THIN_OFFSET).abs()) / soft + 0.5, 0.0, 1.0
            )

            ground = background.expand(HEIGHT, WIDTH, 3)
            composite = (
                ground * (1.0 - THIN_ALPHA * thin)[..., None]
                + thin_colour * (THIN_ALPHA * thin)[..., None]
            )
            composite = (
                composite * (1.0 - main)[..., None] + colour[jj, ii] * main[..., None]
            )

            out = composite.clamp_(0.0, 255.0).to(torch.uint8).flip(-1).cpu().numpy()
            pending.append(
                pool.submit(
                    cv2.imwrite,
                    str(frames_path / f"frame{frame:04d}.png"),
                    out,
                    PNG_PARAMS,
                )
            )
            while len(pending) > 2 * PNG_WRITERS:
                if not pending.pop(0).result():
                    raise OSError("Could not write a frame")
            if frame % FPS == 0:
                logger.info(f"Frame {frame}/{FRAME_COUNT}")

        for task in pending:
            if not task.result():
                raise OSError("Could not write a frame")

    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
