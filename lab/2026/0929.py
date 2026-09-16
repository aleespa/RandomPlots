"""
Hex Tumble

Three rhombuses make a hexagon; shade them light, medium and dark and
the hexagon becomes a cube seen corner-on. Tile the plane with them and
you have the tumbling blocks of a thousand quilts and Roman floors —
a pattern the eye cannot settle: stare, and the cubes flip, the top
faces become bottoms, the stack pops inside out.

Here the shading is not fixed. A slow wave rolls across the tiling and
as it passes, the three shades of every cube rotate: light slides to
where medium was, medium to dark. Nothing moves, no edge is redrawn —
only the tones change — and the whole floor seems to turn over, row by
row, as the wave goes by. The wave completes whole cycles over the
clip, so the last frame is the first.

Made with Python, PyTorch & NumPy.
#generativeart #tumblingblocks #isometric #opart #tessellation #hexagon
#geometry #mathematics #randomplots #creativecoding #algorithmicart
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

# --- Tiling ------------------------------------------------------------------
COLUMNS = 7.5  # hexagons across the width (pointy-top, so columns interleave)
EDGE = 1.6  # dark seam between rhombuses, in pixels
EDGE_SOFTNESS = 1.0
# Waves: (cycles per loop, wavelength in hex widths, direction angle). Their
# mean drives the shade phase; integer cycles make the loop exact.
WAVES = (
    (1, 4.0, 1.25),
    (-1, 6.5, -0.35),
)
WAVE_DEPTH = 0.5  # how far (in face cycles) the wave rotates the shades

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([10, 9, 16], dtype=np.float32)  # only ever seen at seams
PALETTE = np.array(  # dark face -> light face
    [
        [0.08, 0.07, 0.18],
        [0.20, 0.16, 0.45],
        [0.55, 0.28, 0.62],
        [0.95, 0.55, 0.45],
        [1.00, 0.88, 0.62],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
VIGNETTE = 0.20


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


def hex_lookup(px: torch.Tensor, py: torch.Tensor, size: float):
    """
    Nearest pointy-top hexagon of circumradius `size` for every pixel.

    Returns the hexagon centre and the pixel's offset from it. Uses axial
    coordinates with cube rounding.
    """
    q = (np.sqrt(3.0) / 3.0 * px - 1.0 / 3.0 * py) / size
    r = (2.0 / 3.0 * py) / size
    s = -q - r
    rq = torch.round(q)
    rr = torch.round(r)
    rs = torch.round(s)
    dq = (rq - q).abs()
    dr = (rr - r).abs()
    ds = (rs - s).abs()
    fix_q = (dq > dr) & (dq > ds)
    fix_r = (~fix_q) & (dr > ds)
    rq = torch.where(fix_q, -rr - rs, rq)
    rr = torch.where(fix_r, -rq - rs, rr)
    cx = size * (np.sqrt(3.0) * rq + np.sqrt(3.0) / 2.0 * rr)
    cy = size * 1.5 * rr
    return cx, cy, px - cx, py - cy


def generate(settings: ImageProcessingSettings = None):
    """
    Tumbling blocks with the face shades rotated by a travelling wave.

    Every pixel is assigned to the nearest pointy-top hexagon and, within it,
    to one of three rhombuses by the 120-degree sector of its offset from the
    centre (sectors split at the vertices at 90, 210 and 330 degrees). Face
    k of the hexagon at c takes the shade 0.5 + 0.5 cos(2 pi (phi(c, t) + k/3))
    where phi is WAVE_DEPTH times the mean of two plane waves with integer
    cycles per loop; advancing phi by 1/3 permutes the three shades, which is
    the visible tumble. The shade is mapped through a five-stop palette.

    Seams between rhombuses and between hexagons are drawn as a thin dark
    line from the analytic distance to the sector boundaries and the hexagon
    edge. Everything is evaluated per pixel on the GPU; a vignette finishes
    the frame. The seed sets the starting phase and mirrors the wave
    directions.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    mirror = -1.0 if rng.uniform() < 0.5 else 1.0

    logger.info(f"Rendering Hex Tumble: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)

    size = WIDTH / (COLUMNS * np.sqrt(3.0))  # circumradius
    apothem = size * np.sqrt(3.0) / 2.0

    py, px = torch.meshgrid(
        torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5,
        torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    cx, cy, dx, dy = hex_lookup(px, py, size)

    # Which rhombus: sectors of 120 degrees, boundaries towards the vertices
    # at 90, 210 and 330 degrees (screen y grows downwards, so flip dy).
    angle = torch.atan2(-dy, dx)  # -pi..pi
    sector = torch.floor(((angle - np.pi / 2.0) / (2.0 * np.pi / 3.0)) % 3.0)

    # Seams: distance to the three sector rays, and to the hexagon edge.
    seam = torch.full_like(dx, float("inf"))
    for k in range(3):
        a = np.pi / 2.0 + k * 2.0 * np.pi / 3.0
        along = dx * np.cos(a) - dy * np.sin(a)
        across = (dx * np.sin(a) + dy * np.cos(a)).abs()
        seam = torch.minimum(
            seam, torch.where(along > 0, across, torch.full_like(across, float("inf")))
        )
    hex_edge = torch.full_like(dx, -float("inf"))
    for a in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        hex_edge = torch.maximum(hex_edge, (dx * np.cos(a) - dy * np.sin(a)).abs())
    seam = torch.minimum(seam, apothem - hex_edge)
    seam_mask = torch.clamp((0.5 * EDGE - seam) / EDGE_SOFTNESS + 0.5, 0.0, 1.0)

    # Wave coordinates in hex widths.
    wx = cx / (2.0 * apothem)
    wy = cy / (2.0 * apothem)
    background = torch.tensor(BACKGROUND, device=device)
    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            phase = torch.zeros_like(wx)
            for cycles, wavelength, direction in WAVES:
                along = wx * np.cos(direction) + mirror * wy * np.sin(direction)
                phase = phase + torch.sin(
                    2.0 * np.pi * (cycles * t - along / wavelength)
                )
            phase = WAVE_DEPTH * phase / len(WAVES)

            shade = 0.5 + 0.5 * torch.cos(2.0 * np.pi * (phase + sector / 3.0))
            colour = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)]
            composite = (
                colour * (1.0 - seam_mask)[..., None]
                + background * seam_mask[..., None]
            )
            composite = composite * vignette

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
