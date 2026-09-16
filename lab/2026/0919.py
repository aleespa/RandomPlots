"""
Spirograph Bloom

A small wheel rolls inside a bigger one, and a pen fixed somewhere on the
small wheel draws. That is the whole toy: the curve it leaves is a
hypotrochoid, and whether it makes a flower, a star or a tangle of loops
depends only on the two radii and on how far the pen sits from the
centre of its wheel.

Here dozens of pens ride the same wheels at once, each a little further
out than the last, and the distances swell and shrink as a slow wave
passing from the innermost pen to the outermost. Three wheel pairs stack
up the frame — seven, thirteen and nine lobes — and turn at the pace of
their own symmetry, so the whole picture repeats itself exactly and the
last frame is the first.

x = (R − r) cos θ + d cos((R − r) θ / r)
y = (R − r) sin θ − d sin((R − r) θ / r)

Made with Python, PyTorch & NumPy.
#generativeart #spirograph #hypotrochoid #geometry #mathematics
#randomplots #creativecoding #algorithmicart #mathisbeautiful
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

# Frames rendered before frame 0 so the motion trail is at steady state.
WARMUP_FRAMES = 20

# --- Wheels ------------------------------------------------------------------
# One entry per spirograph: lobes p, turns q (R/r = p/q), vertical position of
# its centre as a fraction of the half-height, outer radius as a fraction of the
# half-width, and how many lobes it advances over the loop. The curve has
# p-fold symmetry, so turning it by any whole number of lobes closes exactly.
WHEELS = (
    (7, 3, 0.62, 0.62, 1),
    (13, 5, 0.0, 0.92, -1),
    (9, 4, -0.62, 0.62, 1),
)
CURVES = 28  # pens per wheel
SAMPLES_PER_TURN = 900  # samples per 2*pi of theta
PEN_MIN = 0.35  # pen distance as a fraction of the small wheel's radius
PEN_MAX = 1.55
PEN_SWING = 0.28  # how far the travelling wave moves each pen
PEN_WAVE_CYCLES = 1  # cycles of the wave over the loop (integer -> exact)

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([12, 4, 18], dtype=np.float32)  # RGB
PALETTE = np.array(
    [
        [0.28, 0.04, 0.45],
        [0.70, 0.08, 0.55],
        [0.97, 0.21, 0.35],
        [1.00, 0.55, 0.15],
        [1.00, 0.88, 0.40],
        [1.00, 1.00, 0.92],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
COLOUR_CYCLES = 1  # palette drift cycles over the loop

LINE_ENERGY = 24.0  # brightness deposited per pixel of curve
TRAIL_DECAY = 0.86
BLOOM = 0.35
BLOOM_SIGMA = 3.0
WIDE_BLOOM = 0.12
WIDE_BLOOM_SIGMA = 18.0
VIGNETTE = 0.22


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
    """Separable Gaussian blur of an (H, W, C) tensor."""
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


def splat(
    flat: torch.Tensor, px: torch.Tensor, py: torch.Tensor, colours: torch.Tensor
):
    """
    Bilinear point splatting into a flat (H*W, 3) buffer. Deposits add, so
    crossings burn brighter; the bilinear weights are the anti-aliasing.
    """
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
        flat.index_add_(0, index, colours * (corner * inside)[:, None])


def sample_lengths(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Screen-space length of curve carried by each sample, per row of samples."""
    steps = torch.hypot(px[:, 1:] - px[:, :-1], py[:, 1:] - py[:, :-1])
    lengths = torch.empty_like(px)
    lengths[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    lengths[:, 0] = steps[:, 0]
    lengths[:, -1] = steps[:, -1]
    return lengths


def hypotrochoid(
    theta: torch.Tensor, p: int, q: int, pen: torch.Tensor, spin: torch.Tensor
):
    """
    Hypotrochoid family with R = 1, r = q/p, one curve per row of `pen`.

    `pen` is the pen distance as a fraction of r, `spin` the rotation of each
    curve. The curve is scaled so that its outer radius is 1 whatever the pen.
    """
    r = q / p
    k = (1.0 - r) / r  # (R - r) / r
    x = (1.0 - r) * torch.cos(theta) + pen * r * torch.cos(k * theta)
    y = (1.0 - r) * torch.sin(theta) - pen * r * torch.sin(k * theta)
    scale = 1.0 / ((1.0 - r) + pen * r)
    x = x * scale
    y = y * scale
    c = torch.cos(spin)
    s = torch.sin(spin)
    return x * c - y * s, x * s + y * c


def generate(settings: ImageProcessingSettings = None):
    """
    Three stacked hypotrochoid families, animated as an exact loop.

    Each wheel pair (p, q) draws CURVES hypotrochoids whose pen distance d_i
    runs from PEN_MIN to PEN_MAX across the family and is modulated by a wave
    travelling along the family index: d_i(t) = d_i + PEN_SWING * sin(2*pi*
    (PEN_WAVE_CYCLES * t + i / CURVES)). The whole family also turns by an
    integer number of lobes (2*pi*n/p) over the loop, which is a symmetry of
    the curve, so the geometry at t = 1 is identical to t = 0.

    Curves are drawn by bilinear point splatting on the GPU: each sample
    carries the screen length of curve it represents, so brightness per pixel
    is independent of sampling density, and crossings accumulate. Colour comes
    from the pen index through a warm palette that drifts one cycle per loop.
    The frame buffer decays between frames for a short motion trail; two-scale
    bloom and a vignette are applied to a copy for output. The seed rotates the
    whole loop in time and picks the direction of the pen wave.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    wave_direction = 1.0 if rng.uniform() < 0.5 else -1.0

    logger.info(
        f"Rendering Spirograph Bloom: {FRAME_COUNT} frames @ {FPS} fps, {WIDTH}x{HEIGHT}"
    )
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(HEIGHT, WIDTH, 3).clone()
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)

    index_row = torch.arange(CURVES, device=device, dtype=torch.float32)[:, None]
    base_pen = PEN_MIN + (PEN_MAX - PEN_MIN) * index_row / (CURVES - 1)

    # Per wheel: parameter samples over q turns (the curve closes after q).
    thetas = [
        torch.linspace(0.0, 2.0 * np.pi * q, SAMPLES_PER_TURN * q, device=device)[
            None, :
        ]
        for _, q, _, _, _ in WHEELS
    ]

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)
            flat.zero_()

            for (p, q, cy, radius, lobes), theta in zip(WHEELS, thetas):
                pen = base_pen + PEN_SWING * torch.sin(
                    2.0
                    * np.pi
                    * (PEN_WAVE_CYCLES * t + wave_direction * index_row / CURVES)
                )
                spin = (2.0 * np.pi / p) * (lobes * t + index_row / CURVES)
                x, y = hypotrochoid(theta, p, q, pen, spin)

                px = WIDTH / 2.0 + x * radius * WIDTH / 2.0
                py = HEIGHT / 2.0 - (y * radius * WIDTH / 2.0 + cy * HEIGHT / 2.0)
                lengths = sample_lengths(px, py)

                # Ping-pong through the palette so the drift never shows a seam.
                shade = 0.5 - 0.5 * torch.cos(
                    2.0 * np.pi * (index_row / CURVES + COLOUR_CYCLES * t)
                )
                shade = shade.expand_as(px)
                colours = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)].reshape(
                    -1, 3
                )
                weights = (lengths * LINE_ENERGY).reshape(-1)
                splat(flat, px.reshape(-1), py.reshape(-1), colours * weights[:, None])

            image.add_(flat.view(HEIGHT, WIDTH, 3))

            if frame < 0:
                continue

            highlights = torch.clamp(image - 120.0, min=0.0)
            composite = image + BLOOM * blur(image, bloom_kernel)
            composite = composite + WIDE_BLOOM * blur(highlights, wide_kernel)
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
