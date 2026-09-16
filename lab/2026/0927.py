"""
Voronoi Drift

Scatter a few dozen points. Give every place on the plane to whichever
point is nearest, and the plane shatters into cells: straight-edged,
convex, fitted together without a gap. This is the Voronoi diagram, the
shape of territories, of cells in a leaf, of the cracks in drying mud.

Here the points do not sit still. Each one rides its own closed orbit —
a Lissajous figure, a whole number of swings in each direction — so the
cells stretch, shrink, gain neighbours and lose them, while the pattern
as a whole never tears. Every orbit closes at the end of the clip, so
the last frame is the first.

Made with Python, PyTorch & NumPy.
#generativeart #voronoi #tessellation #geometry #mathematics #lissajous
#stainedglass #randomplots #creativecoding #algorithmicart
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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

# --- Sites -------------------------------------------------------------------
SITES = 56
ORBIT = 0.16  # orbit half-size as a fraction of the width
ORBIT_CYCLES = (1, 2, 3)  # each site draws its x and y swing counts from these
MARGIN = 0.10  # keep site centres this far inside the frame, as a fraction

# --- Appearance --------------------------------------------------------------
EDGE_WIDTH = 7.0  # pixels, the dark leading between cells
EDGE_SOFTNESS = 1.5
LEADING = np.array([12, 10, 18], dtype=np.float32)  # RGB
LIGHT_FALLOFF = 0.55  # cells darken towards their edge by this much
SITE_GLOW = 0.5  # bright core at each site
SITE_RADIUS = 42.0
PALETTE = np.array(
    [
        [0.93, 0.35, 0.25],
        [0.98, 0.65, 0.20],
        [0.95, 0.85, 0.40],
        [0.35, 0.70, 0.55],
        [0.20, 0.50, 0.75],
        [0.45, 0.30, 0.65],
        [0.85, 0.45, 0.60],
    ],
    dtype=np.float32,
)
BLOOM = 0.18
BLOOM_SIGMA = 6.0
VIGNETTE = 0.25


def select_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("no CUDA device, rendering on CPU")
    return torch.device("cpu")


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    return kernel / kernel.sum()


def blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    planes = image.permute(2, 0, 1)[None]
    channels = planes.shape[1]
    radius = (kernel.numel() - 1) // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.conv2d(
        F.pad(planes, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    out = F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels
    )
    return out[0].permute(1, 2, 0)


def generate(settings: ImageProcessingSettings = None):
    """
    A Voronoi tessellation of sites on closed Lissajous orbits, exact loop.

    Site k moves as c_k + ORBIT * (sin(2 pi (a_k t + phi_k)), sin(2 pi (b_k t
    + psi_k))) with integer a_k, b_k, so every orbit closes over the loop and
    the diagram at t = 1 equals t = 0. Centres are stratified over the frame
    so the cells start evenly sized.

    Per pixel, the nearest and second-nearest sites are found by a running
    top-two over all sites (memory stays at a few frames' worth regardless of
    the site count). The cell takes its site's palette colour, darkened by the
    ratio of the distance to the nearest site over the distance to the cell
    edge, which reads as light through glass; the edge itself is drawn as
    dark leading where d2 - d1 is below EDGE_WIDTH, and a soft glow marks each
    site. A single bloom pass and a vignette finish the frame. The seed sets
    the orbits, phases and colour assignment.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    # Stratified centres: a jittered grid, so no region starts empty.
    cols = 6
    rows_n = int(np.ceil(SITES / cols))
    grid = [(i, j) for j in range(rows_n) for i in range(cols)][:SITES]
    centres = np.array(
        [
            [
                MARGIN * WIDTH
                + (WIDTH * (1 - 2 * MARGIN)) * (i + rng.uniform(0.15, 0.85)) / cols,
                MARGIN * HEIGHT
                + (HEIGHT * (1 - 2 * MARGIN)) * (j + rng.uniform(0.15, 0.85)) / rows_n,
            ]
            for i, j in grid
        ]
    )
    cycles_x = rng.choice(ORBIT_CYCLES, size=SITES) * rng.choice([-1, 1], size=SITES)
    cycles_y = rng.choice(ORBIT_CYCLES, size=SITES) * rng.choice([-1, 1], size=SITES)
    phase_x = rng.uniform(0.0, 1.0, size=SITES)
    phase_y = rng.uniform(0.0, 1.0, size=SITES)
    seed_phase = rng.uniform(0.0, 1.0)
    palette = torch.tensor(PALETTE * 255.0, device=device)
    colour_index = torch.tensor(
        rng.integers(0, len(PALETTE), size=SITES), device=device
    )
    site_colours = palette[colour_index]  # (SITES, 3)

    logger.info(f"Rendering Voronoi Drift: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)

    py, px = torch.meshgrid(
        torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5,
        torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]
    leading = torch.tensor(LEADING, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            sx = centres[:, 0] + ORBIT * WIDTH * np.sin(
                2.0 * np.pi * (cycles_x * t + phase_x)
            )
            sy = centres[:, 1] + ORBIT * WIDTH * np.sin(
                2.0 * np.pi * (cycles_y * t + phase_y)
            )

            d1 = torch.full((HEIGHT, WIDTH), float("inf"), device=device)
            d2 = torch.full((HEIGHT, WIDTH), float("inf"), device=device)
            nearest = torch.zeros((HEIGHT, WIDTH), dtype=torch.long, device=device)
            for k in range(SITES):
                d = torch.hypot(px - float(sx[k]), py - float(sy[k]))
                closer = d < d1
                d2 = torch.where(closer, d1, torch.minimum(d2, d))
                d1 = torch.where(closer, d, d1)
                nearest = torch.where(closer, torch.full_like(nearest, k), nearest)

            # Distance from the pixel to its cell edge along the bisector is (d2 - d1) / 2.
            edge = 0.5 * (d2 - d1)
            lead = torch.clamp(
                (0.5 * EDGE_WIDTH - edge) / EDGE_SOFTNESS + 0.5, 0.0, 1.0
            )
            # Light falls off from the site towards the edge.
            depth = d1 / (d1 + edge).clamp(min=1e-6)
            light = 1.0 - LIGHT_FALLOFF * depth**1.5
            glow = SITE_GLOW * torch.exp(-0.5 * (d1 / SITE_RADIUS) ** 2)

            fill = site_colours[nearest] * (light + glow)[..., None]
            image = fill * (1.0 - lead)[..., None] + leading * lead[..., None]
            composite = (image + BLOOM * blur(image, bloom_kernel)) * vignette

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
