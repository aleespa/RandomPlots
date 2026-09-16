"""
Golden Angle Drift

Put down seeds one at a time, each a fixed turn further round than the
last and a little further out. If the turn is the golden angle — about
137.5° — the seeds pack as tightly as they can, and the eye picks out
spirals curving both ways: the 13, 21 and 34 arms of a sunflower head.

That arrangement is fragile. The thousandth seed turns a thousand times
further than the first for the same change of angle, so a shift of a few
hundredths of a degree leaves the centre untouched and rearranges the
outer arms completely. This clip swings the angle by that much either
side of the golden angle and nothing else: every seed keeps its place in
the sequence, only its bearing changes. The spirals that appear and
dissolve are not drawn — they are what the eye finds in the pattern.

r = c √n,  θ = n · α,  α = 137.508° ± 0.016°

Made with Python, PyTorch & NumPy.
#generativeart #phyllotaxis #goldenangle #fibonacci #sunflower #geometry
#mathematics #randomplots #creativecoding #algorithmicart #mathisbeautiful
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

# --- Seeds -------------------------------------------------------------------
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))  # 137.507... degrees
# How far the divergence angle wanders. Seed n turns by n times the change, so
# a few hundredths of a degree already rearranges the outer arms completely.
ANGLE_SWING = np.deg2rad(0.016)
SWING_CYCLES = 1  # integer cycles per loop -> exact loop
SEED_COUNT = 5800
SPACING = 19.0  # r = SPACING * sqrt(n), in pixels; fills the portrait height
DISC_RADIUS = 6.8  # base seed radius, pixels
PULSE_DEPTH = 0.45  # radius swells by this fraction as the pulse passes
PULSE_CYCLES = 2  # pulses travelling outward per loop
PULSE_WAVELENGTH = 520.0  # pixels from one pulse crest to the next

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([6, 12, 16], dtype=np.float32)  # RGB
PALETTE = np.array(
    [
        [0.02, 0.35, 0.40],
        [0.05, 0.62, 0.60],
        [0.55, 0.85, 0.60],
        [1.00, 0.82, 0.30],
        [1.00, 0.95, 0.75],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
COLOUR_CYCLES = 1  # palette drift over the loop
SEED_ENERGY = 1.25  # brightness per unit of disc coverage
BLOOM = 0.30
BLOOM_SIGMA = 5.0
VIGNETTE = 0.28


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
    return torch.tensor(lut, dtype=torch.float32, device=device)


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


def disc_offsets(radius: float, device: torch.device) -> torch.Tensor:
    """
    A grid of offsets covering the largest disc, with the fractional coverage
    of each cell computed for a unit-radius disc so it can be scaled per seed.
    """
    span = int(np.ceil(radius)) + 1
    grid = torch.arange(-span, span + 1, device=device, dtype=torch.float32)
    dy, dx = torch.meshgrid(grid, grid, indexing="ij")
    return torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)


def generate(settings: ImageProcessingSettings = None):
    """
    Vogel's phyllotaxis with an oscillating divergence angle.

    Seed n sits at r = SPACING*sqrt(n), theta = n*alpha(t), with alpha(t) =
    golden angle + ANGLE_SWING*sin(2*pi*SWING_CYCLES*t + phi). The sqrt law
    keeps the area per seed constant, and small departures from the golden
    angle make different Fibonacci parastichy families dominate, which is the
    visible reorganisation. A radial pulse, r-dependent and integer-periodic,
    modulates the seed radius so the change in packing is also read through
    seed size.

    Each seed is rasterised as an anti-aliased disc: a fixed stencil of pixel
    offsets is evaluated against the seed's radius with a one-pixel soft edge
    and deposited by bilinear splatting, so discs overlap additively where the
    packing becomes tight. Colour follows the seed index through a teal-to-
    gold palette that drifts one cycle per loop. A single bloom pass and a
    vignette finish the frame. The seed of the RNG sets the loop's starting
    phase and the orientation of the whole head.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    orientation = rng.uniform(0.0, 2.0 * np.pi)

    logger.info(f"Rendering Golden Angle Drift: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.55, 1.0)[..., None]

    n = torch.arange(1, SEED_COUNT + 1, device=device, dtype=torch.float32)
    radius = SPACING * torch.sqrt(n)
    shade_base = n / SEED_COUNT

    max_radius = DISC_RADIUS * (1.0 + PULSE_DEPTH)
    offsets = disc_offsets(max_radius, device)  # (K, 2)
    offset_norm = torch.hypot(offsets[:, 0], offsets[:, 1])  # (K,)

    background = torch.tensor(BACKGROUND, device=device)
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            alpha = GOLDEN_ANGLE + ANGLE_SWING * np.sin(2.0 * np.pi * SWING_CYCLES * t)

            theta = n * alpha + orientation
            cx = WIDTH / 2.0 + radius * torch.cos(theta)
            cy = HEIGHT / 2.0 - radius * torch.sin(theta)

            pulse = torch.sin(
                2.0 * np.pi * (PULSE_CYCLES * t - radius / PULSE_WAVELENGTH)
            )
            seed_radius = DISC_RADIUS * (1.0 + PULSE_DEPTH * pulse)  # (N,)

            # Ping-pong through the palette so the drift never shows a seam.
            shade = 0.5 - 0.5 * torch.cos(
                2.0 * np.pi * (shade_base + COLOUR_CYCLES * t)
            )
            colours = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)]  # (N, 3)

            # Coverage of every stencil cell for every seed: soft edge one pixel wide.
            coverage = torch.clamp(
                seed_radius[:, None] + 0.5 - offset_norm[None, :], 0.0, 1.0
            )
            px = (cx[:, None] + offsets[None, :, 0]).reshape(-1)
            py = (cy[:, None] + offsets[None, :, 1]).reshape(-1)
            weight = (coverage * SEED_ENERGY * 255.0).reshape(-1)
            deposit = (
                colours[:, None, :].expand(-1, offsets.shape[0], -1).reshape(-1, 3)
                * weight[:, None]
            )

            # Stencil cells sit on integer offsets from a fractional centre, so the
            # bilinear split is the same for every cell of a seed: do it explicitly.
            flat.zero_()
            x0 = torch.floor(px)
            y0 = torch.floor(py)
            fx = px - x0
            fy = py - y0
            x0 = x0.to(torch.long)
            y0 = y0.to(torch.long)
            for ox, oy, corner in (
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ):
                ix = x0 + ox
                iy = y0 + oy
                inside = (ix >= 0) & (ix < WIDTH) & (iy >= 0) & (iy < HEIGHT)
                index = iy.clamp(0, HEIGHT - 1) * WIDTH + ix.clamp(0, WIDTH - 1)
                flat.index_add_(0, index, deposit * (corner * inside)[:, None])

            image = flat.view(HEIGHT, WIDTH, 3)
            composite = image + BLOOM * blur(image, bloom_kernel) + background
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
