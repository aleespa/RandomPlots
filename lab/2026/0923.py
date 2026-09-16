"""
Whirling Squares

Take a square. Mark a point a little way along each side, always the same
fraction, and join the four marks: a smaller square, slightly turned.
Do it again inside that one, and again, forty times, and the corners
spiral inward to a point — four curves that were never drawn, only
implied by the straight lines.

Tile the frame with these, mirror the twist on alternating cells so the
spirals lock into one another, and let the fraction ripple across the
tiling as a slow wave. Where it is small the squares nest almost
straight; where it is large they twist hard and the spirals tighten.
Every line is straight. The curves are in your eye.

vₖ₊₁ = vₖ + τ (vₖ₊₁ − vₖ)

Made with Python, PyTorch & NumPy.
#generativeart #whirlingsquares #pursuitcurve #tessellation #geometry
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

# --- Tiling ------------------------------------------------------------------
COLUMNS = 5  # cells across the width; rows follow from the aspect ratio
DEPTH = 64  # nested squares per cell
SAMPLES_PER_EDGE = 220
INSET_MIN = 0.03  # fraction of each edge walked before the next corner
INSET_MAX = 0.24
# The inset ripple: (cycles per loop, wavelength in cells, direction angle).
RIPPLES = (
    (1, 4.5, 0.35),
    (-1, 7.0, 2.30),
)
SLOW_TURN_CYCLES = 1  # whole-tiling rotation cycles (a wobble, not a spin)
SLOW_TURN = 0.0  # radians of wobble; 0 keeps the lattice square to the frame

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([5, 4, 14], dtype=np.float32)  # RGB
PALETTE = np.array(
    [
        [0.12, 0.05, 0.40],
        [0.35, 0.10, 0.75],
        [0.15, 0.55, 0.95],
        [0.25, 0.95, 0.90],
        [0.95, 1.00, 1.00],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
LINE_ENERGY = 115.0  # brightness per pixel of line
BLOOM = 0.30
BLOOM_SIGMA = 2.5
WIDE_BLOOM = 0.10
WIDE_BLOOM_SIGMA = 16.0
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


def splat(
    flat: torch.Tensor, px: torch.Tensor, py: torch.Tensor, colours: torch.Tensor
):
    """Bilinear point splatting into a flat (H*W, 3) buffer; deposits add."""
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


def nested_squares(corners: torch.Tensor, inset: torch.Tensor) -> torch.Tensor:
    """
    Iterate the inset rule on a batch of squares.

    corners: (C, 4, 2) the outer square of every cell; inset: (C,) the fraction
    tau per cell, signed -- negative walks the edges the other way round, which
    is the mirrored chirality. Returns (DEPTH + 1, C, 4, 2).
    """
    levels = [corners]
    tau = inset[:, None, None]
    for _ in range(DEPTH):
        current = levels[-1]
        forward = current.roll(-1, dims=1)
        backward = current.roll(1, dims=1)
        # tau > 0 walks towards the next corner, tau < 0 towards the previous.
        nxt = torch.where(
            tau > 0,
            current + tau * (forward - current),
            current - tau * (backward - current),
        )
        levels.append(nxt)
    return torch.stack(levels, dim=0)


def generate(settings: ImageProcessingSettings = None):
    """
    A tiling of whirling squares whose inset ratio is a travelling wave.

    Each cell of a square lattice holds DEPTH nested squares built by the
    pursuit rule v' = v + tau (v_next - v). Cells alternate chirality in a
    checkerboard, so the implied spiral curves meet across cell edges. The
    inset tau varies per cell as INSET_MIN..INSET_MAX by the mean of two plane
    waves, each with an integer number of cycles per loop, so the whole
    tiling is exactly periodic in time.

    Every edge of every square is sampled uniformly and splatted bilinearly on
    the GPU, with each sample weighted by the screen length it represents, so
    brightness per pixel of line is constant and the dense inner squares do
    not burn out. Colour follows nesting depth through a violet-to-cyan ramp,
    cooled slightly on one chirality and warmed on the other. Two-scale bloom
    and a vignette finish the frame. The RNG seed sets the loop's starting
    phase and which chirality the checkerboard begins with.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    parity = int(rng.integers(0, 2))

    logger.info(f"Rendering Whirling Squares: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    cell = WIDTH / COLUMNS
    rows = int(np.ceil(HEIGHT / cell))
    # Centre the lattice vertically so the crop is symmetric.
    offset_y = (HEIGHT - rows * cell) / 2.0

    ij = [(i, j) for j in range(rows) for i in range(COLUMNS)]
    ci = torch.tensor([i for i, _ in ij], device=device, dtype=torch.float32)
    cj = torch.tensor([j for _, j in ij], device=device, dtype=torch.float32)
    chirality = torch.tensor(
        [1.0 if (i + j + parity) % 2 == 0 else -1.0 for i, j in ij], device=device
    )
    x0 = ci * cell
    y0 = offset_y + cj * cell
    corners = torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x0 + cell, y0], dim=-1),
            torch.stack([x0 + cell, y0 + cell], dim=-1),
            torch.stack([x0, y0 + cell], dim=-1),
        ],
        dim=1,
    )  # (C, 4, 2)
    centre_x = ci + 0.5
    centre_y = cj + 0.5

    depth_shade = torch.linspace(0.0, 1.0, DEPTH + 1, device=device)
    u = torch.linspace(0.0, 1.0, SAMPLES_PER_EDGE, device=device)[
        None, None, None, :, None
    ]

    background = torch.tensor(BACKGROUND, device=device)
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            wave = torch.zeros_like(centre_x)
            for cycles, wavelength, angle in RIPPLES:
                along = centre_x * np.cos(angle) + centre_y * np.sin(angle)
                wave = wave + torch.sin(2.0 * np.pi * (cycles * t - along / wavelength))
            wave = wave / len(RIPPLES)
            inset = INSET_MIN + (INSET_MAX - INSET_MIN) * 0.5 * (1.0 + wave)

            levels = nested_squares(corners, inset * chirality)  # (D+1, C, 4, 2)
            starts = levels
            ends = levels.roll(-1, dims=2)
            points = (
                starts[..., None, :] + u * (ends - starts)[..., None, :]
            )  # (D+1, C, 4, S, 2)
            edge_length = torch.hypot(*(ends - starts).unbind(-1))  # (D+1, C, 4)
            weight = (edge_length / SAMPLES_PER_EDGE * LINE_ENERGY)[..., None].expand(
                -1, -1, -1, SAMPLES_PER_EDGE
            )

            shade = depth_shade[:, None, None, None].expand(
                -1, len(ij), 4, SAMPLES_PER_EDGE
            )
            shade = (shade + 0.06 * chirality[None, :, None, None]).clamp(0.0, 1.0)
            colours = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)].reshape(-1, 3)

            flat.zero_()
            splat(
                flat,
                points[..., 0].reshape(-1),
                points[..., 1].reshape(-1),
                colours * weight.reshape(-1)[:, None],
            )
            image = flat.view(HEIGHT, WIDTH, 3)

            highlights = torch.clamp(image - 140.0, min=0.0)
            composite = (
                image
                + BLOOM * blur(image, bloom_kernel)
                + WIDE_BLOOM * blur(highlights, wide_kernel)
            )
            composite = (composite + background) * vignette

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
