"""
Isoline Tide

A contour map shows a landscape as lines of equal height. Here the
landscape is nothing but a handful of soft hills — smooth mounds of
the same shape, each one riding its own slow closed orbit — and the
lines are drawn fresh every frame.

Watch what the lines do as the hills meet. Two separate rings of
contours approach, kiss, and at the moment of contact a single line
pinches through the saddle between them and they become one. Pull the
hills apart and the ring splits again. Nothing is drawn as a curve;
every line is just the set of points where the height happens to equal
one fixed value. Every orbit closes after a whole number of swings, so
the tide runs back into itself exactly.

Made with Python, PyTorch & NumPy.
#generativeart #contours #isolines #topography #metaballs #geometry
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

# --- Landscape ---------------------------------------------------------------
HILLS = 9
SIGMA = 0.26  # hill width, in units where the half-width of the frame is 1
ORBIT = 0.40  # orbit half-size in the same units
ORBIT_CYCLES = (1, 2)  # each hill draws its x and y swing counts from these
LEVELS = 16  # contour levels between the floor and the tallest possible peak
LEVEL_FLOOR = 0.08
LEVEL_CEILING = 2.0
LINE_WIDTH = 2.2  # contour half-width in pixels, measured as |f - c| / |grad f|

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([5, 8, 16], dtype=np.float32)  # RGB
FILL = np.array([14, 34, 60], dtype=np.float32)  # low-lying ground tint
FILL_GAIN = 0.55
PALETTE = np.array(  # low level -> high level
    [
        [0.10, 0.45, 0.70],
        [0.10, 0.80, 0.85],
        [0.75, 0.98, 0.80],
        [1.00, 0.85, 0.45],
        [1.00, 0.98, 0.95],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
LINE_GAIN = 255.0
BLOOM = 0.45
BLOOM_SIGMA = 3.5
WIDE_BLOOM = 0.22
WIDE_BLOOM_SIGMA = 20.0
VIGNETTE = 0.25


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


def generate(settings: ImageProcessingSettings = None):
    """
    Contour lines of a sum of Gaussian hills on closed Lissajous orbits.

    The height field is f(p) = sum_k exp(-|p - c_k(t)|^2 / (2 SIGMA^2)) with
    c_k(t) = o_k + ORBIT (sin 2 pi (a_k t + phi_k), sin 2 pi (b_k t + psi_k))
    for integer a_k, b_k, so f is exactly periodic over the loop. The
    analytic gradient is accumulated alongside f, and each of LEVELS contour
    values c_l is drawn with constant screen width by the first-order
    distance |f - c_l| / |grad f|, so lines stay crisp whether the slope is
    steep or gentle. Each level takes its colour from a blue-to-white ramp,
    and the field itself tints the ground faintly so the hills read as
    volumes between the lines. Two-scale bloom and a vignette finish the
    frame. The seed sets orbit centres, swing counts and phases.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    aspect = HEIGHT / WIDTH
    centres = np.stack(
        [
            rng.uniform(-0.6, 0.6, size=HILLS),
            rng.uniform(-aspect + 0.5, aspect - 0.5, size=HILLS),
        ],
        axis=1,
    )
    cycles_x = rng.choice(ORBIT_CYCLES, size=HILLS) * rng.choice([-1, 1], size=HILLS)
    cycles_y = rng.choice(ORBIT_CYCLES, size=HILLS) * rng.choice([-1, 1], size=HILLS)
    phase_x = rng.uniform(0.0, 1.0, size=HILLS)
    phase_y = rng.uniform(0.0, 1.0, size=HILLS)
    seed_phase = rng.uniform(0.0, 1.0)

    logger.info(f"Rendering Isoline Tide: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    y, x = torch.meshgrid(
        torch.linspace(-aspect, aspect, HEIGHT, device=device),
        torch.linspace(-1.0, 1.0, WIDTH, device=device),
        indexing="ij",
    )
    pixel = 2.0 / WIDTH
    vignette = torch.clamp(1.0 - VIGNETTE * (x**2 + (y / aspect) ** 2), 0.55, 1.0)[
        ..., None
    ]
    levels = np.linspace(LEVEL_FLOOR, LEVEL_CEILING, LEVELS)
    level_colours = lut[
        (torch.linspace(0.0, 1.0, LEVELS, device=device) * (PALETTE_SIZE - 1)).to(
            torch.long
        )
    ]

    background = torch.tensor(BACKGROUND, device=device)
    fill = torch.tensor(FILL, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            hx = centres[:, 0] + ORBIT * np.sin(2.0 * np.pi * (cycles_x * t + phase_x))
            hy = centres[:, 1] + ORBIT * np.sin(2.0 * np.pi * (cycles_y * t + phase_y))

            f = torch.zeros_like(x)
            fx = torch.zeros_like(x)
            fy = torch.zeros_like(x)
            for k in range(HILLS):
                dx = x - float(hx[k])
                dy = y - float(hy[k])
                g = torch.exp(-(dx * dx + dy * dy) / (2.0 * SIGMA * SIGMA))
                f = f + g
                fx = fx - g * dx / (SIGMA * SIGMA)
                fy = fy - g * dy / (SIGMA * SIGMA)
            gradient = torch.hypot(fx, fy).clamp(min=1e-6) * pixel  # per pixel

            lines = torch.zeros(HEIGHT, WIDTH, 3, device=device)
            for l, c in enumerate(levels):
                distance = (f - float(c)).abs() / gradient
                line = torch.exp(-0.5 * (distance / LINE_WIDTH) ** 2)
                lines = lines + line[..., None] * level_colours[l]

            ground = background + FILL_GAIN * torch.tanh(f)[..., None] * fill
            glow = lines * LINE_GAIN
            composite = (
                ground
                + glow
                + BLOOM * blur(glow, bloom_kernel)
                + WIDE_BLOOM * blur(glow, wide_kernel)
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
