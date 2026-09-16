"""
Kinetic Lattice

A plain grid of squares, and two ripples running through it. Each square
turns and shrinks by an amount set only by where the ripples are as they
pass, so nothing moves across the frame — every square stays in its cell
— and yet the eye sees waves rolling over the surface, sheets of tiles
tilting together like a field of wind vanes.

This is the trick of the kinetic art of the 1960s: motion made from
static things, order made from a rule applied identically everywhere.
Two ripple sources circle the frame in opposite directions and complete
their orbits exactly as the clip ends, so the last frame is the first.

Turn(x, y, t) = A · Σ sin(2π f t − k |x − c(t)|)

Made with Python, PyTorch & NumPy.
#generativeart #kineticart #opart #vasarely #geometry #grid #tiling
#mathematics #randomplots #creativecoding #algorithmicart
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

# --- Lattice -----------------------------------------------------------------
COLUMNS = 13  # cells across the width; rows follow from the aspect ratio
SIZE_MIN = 0.22  # square half-side, as a fraction of the cell, at full shrink
SIZE_MAX = 0.43
MAX_TURN = np.pi / 4.0  # a square is invariant under a quarter turn
EDGE_SOFTNESS = 1.0  # anti-aliasing width in pixels

# Two ripple sources: (orbit radius as a fraction of the half-width, orbit turns
# per loop, wave cycles per loop, wavelength in cells). Integer turns and cycles
# are what make the loop exact.
RIPPLES = (
    (0.65, 1, 3, 5.5),
    (0.65, -1, 2, 8.0),
)

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([244, 240, 231], dtype=np.float32)  # warm off-white, RGB
# Tile colour follows the local wave value through this ramp.
PALETTE = np.array(
    [
        [0.10, 0.10, 0.18],  # ink
        [0.16, 0.36, 0.55],  # slate blue
        [0.90, 0.38, 0.28],  # coral
        [0.96, 0.68, 0.30],  # amber
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
SHADOW = 0.10  # faint drop shadow below each square, as a fraction of the cell
SHADOW_STRENGTH = 0.18


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


def wave(cell_x: torch.Tensor, cell_y: torch.Tensor, t: float):
    """
    Sum of the ripples at each cell centre, in [-1, 1] per ripple.

    Each source moves on a circle about the frame centre and emits a wave
    whose phase advances an integer number of cycles per loop.
    """
    total = torch.zeros_like(cell_x)
    for orbit, turns, cycles, wavelength in RIPPLES:
        angle = 2.0 * np.pi * turns * t
        sx = 0.5 * COLUMNS + orbit * 0.5 * COLUMNS * np.cos(angle)
        sy = 0.5 * COLUMNS * HEIGHT / WIDTH + orbit * 0.5 * COLUMNS * np.sin(angle)
        distance = torch.hypot(cell_x - sx, cell_y - sy)
        total = total + torch.sin(2.0 * np.pi * (cycles * t - distance / wavelength))
    return total / len(RIPPLES)


def generate(settings: ImageProcessingSettings = None):
    """
    A lattice of squares rotated and scaled by two travelling ripples.

    Cell (i, j) holds one square whose rotation is MAX_TURN * w and whose
    half-side interpolates SIZE_MIN..SIZE_MAX by (1 + w) / 2, where w(i, j, t)
    is the mean of two circular waves radiating from sources that orbit the
    frame centre. Orbit turns and wave cycles are integers, so w is exactly
    periodic in t and the clip closes on itself.

    Rasterisation is analytic: for every pixel the signed distance to the
    square in its own cell and in the eight neighbouring cells is evaluated
    (a rotated square's SDF is the Chebyshev norm of the rotated offset minus
    the half-side), converted to coverage with a one-pixel ramp, and the
    covered colour is composited over the off-white ground. Tile colour is the
    local wave value through a four-stop ramp, and a faint offset copy of the
    coverage supplies a drop shadow. The RNG seed sets the phase at which the
    loop starts and mirrors the ripples' direction of travel.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    mirror = -1.0 if rng.uniform() < 0.5 else 1.0

    logger.info(f"Rendering Kinetic Lattice: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)

    cell = WIDTH / COLUMNS
    rows = int(np.ceil(HEIGHT / cell))

    # Pixel centres in cell units.
    py, px = torch.meshgrid(
        (torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5) / cell,
        (torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5) / cell,
        indexing="ij",
    )
    px = px * mirror + (COLUMNS if mirror < 0 else 0.0)
    home_i = torch.floor(px)
    home_j = torch.floor(py)

    # Cell centres for the whole lattice, padded by one so the neighbour lookup
    # never indexes outside.
    ci = torch.arange(-1, COLUMNS + 1, device=device, dtype=torch.float32) + 0.5
    cj = torch.arange(-1, rows + 1, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(cj, ci, indexing="ij")

    shadow_dx = SHADOW * 0.6
    shadow_dy = SHADOW

    background = torch.tensor(BACKGROUND, device=device)
    neighbours = [(di, dj) for dj in (-1, 0, 1) for di in (-1, 0, 1)]

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            w = wave(grid_x, grid_y, t)  # (rows+2, COLUMNS+2)
            turn = MAX_TURN * w
            half = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * 0.5 * (1.0 + w)
            cos_t = torch.cos(turn)
            sin_t = torch.sin(turn)
            shade = 0.5 * (1.0 + w)
            colour = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)]  # (.., .., 3)

            coverage = torch.zeros(HEIGHT, WIDTH, device=device)
            shadow = torch.zeros(HEIGHT, WIDTH, device=device)
            tint = torch.zeros(HEIGHT, WIDTH, 3, device=device)

            for di, dj in neighbours:
                # Index of the neighbouring cell in the padded lattice arrays.
                ii = (home_i + di + 1).to(torch.long).clamp(0, COLUMNS + 1)
                jj = (home_j + dj + 1).to(torch.long).clamp(0, rows + 1)
                c = cos_t[jj, ii]
                s = sin_t[jj, ii]
                h = half[jj, ii]
                cx = grid_x[jj, ii]
                cy = grid_y[jj, ii]

                # Offset from the square centre, rotated into the square's frame.
                dx = px - cx
                dy = py - cy
                u = dx * c + dy * s
                v = -dx * s + dy * c
                sdf = (torch.maximum(u.abs(), v.abs()) - h) * cell  # pixels
                cover = torch.clamp(0.5 - sdf / EDGE_SOFTNESS, 0.0, 1.0)

                # Same square, displaced, for the shadow.
                us = (dx - shadow_dx) * c + (dy - shadow_dy) * s
                vs = -(dx - shadow_dx) * s + (dy - shadow_dy) * c
                sdf_s = (torch.maximum(us.abs(), vs.abs()) - h) * cell
                shadow = torch.maximum(
                    shadow, torch.clamp(0.5 - sdf_s / (6.0 * EDGE_SOFTNESS), 0.0, 1.0)
                )

                tint = tint + cover[..., None] * colour[jj, ii]
                coverage = coverage + cover

            # Where squares overlap, average their colours rather than summing.
            tint = tint / coverage.clamp(min=1e-6)[..., None]
            coverage = coverage.clamp(max=1.0)

            ground = background * (1.0 - SHADOW_STRENGTH * shadow)[..., None]
            composite = (
                ground * (1.0 - coverage)[..., None] + tint * coverage[..., None]
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
