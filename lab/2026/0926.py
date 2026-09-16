"""
Superformula Bloom

In 1997 the botanist Johan Gielis noticed that one small formula could
draw an astonishing share of the shapes in nature: starfish, diatoms,
petals, seed cases, the cross-sections of stems. Give it a symmetry and
three exponents and it returns an outline. Nudge the exponents and the
outline swells into a flower, sharpens into a star, softens into a
rounded polygon.

Here the same outline is drawn sixty times, each copy a little smaller
and a little turned, so the shape becomes a nested throat. The three
exponents breathe in slow whole cycles and the whole bloom opens, folds
and opens again, arriving back where it began.

r(θ) = ( |cos(mθ/4)|ⁿ² + |sin(mθ/4)|ⁿ³ )^(−1/n₁)

Made with Python, PyTorch & NumPy.
#generativeart #superformula #gielis #botany #geometry #mathematics
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
WARMUP_FRAMES = 16

# --- Shape -------------------------------------------------------------------
SYMMETRIES = (5, 6, 7, 8, 9)  # the seed picks m from these
LAYERS = 60  # nested copies
SAMPLES = 2400  # samples per outline
INNER_SCALE = 0.06  # radius of the innermost copy, as a fraction of the outer
TWIST = 0.55  # total rotation from outer to inner copy, radians
TWIST_CYCLES = 1  # the twist itself swings with this many cycles per loop
# Exponents: (centre, swing, cycles per loop, phase). Integer cycles -> exact loop.
N1 = (0.9, 0.5, 1, 0.00)
N2 = (1.6, 1.1, 1, 0.33)
N3 = (1.6, 1.1, -1, 0.66)
RADIUS = 0.90  # outer copy's radius as a fraction of the half-width

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([8, 6, 10], dtype=np.float32)  # RGB
PALETTE = np.array(  # outer -> inner
    [
        [0.10, 0.25, 0.30],
        [0.10, 0.60, 0.55],
        [0.55, 0.90, 0.60],
        [1.00, 0.85, 0.45],
        [1.00, 0.98, 0.90],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
LINE_ENERGY = 70.0
TRAIL_DECAY = 0.84
BLOOM = 0.35
BLOOM_SIGMA = 3.0
WIDE_BLOOM = 0.14
WIDE_BLOOM_SIGMA = 20.0
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


def sample_lengths(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    steps = torch.hypot(px[:, 1:] - px[:, :-1], py[:, 1:] - py[:, :-1])
    lengths = torch.empty_like(px)
    lengths[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    lengths[:, 0] = steps[:, 0]
    lengths[:, -1] = steps[:, -1]
    return lengths


def exponent(spec, t: float) -> float:
    centre, swing, cycles, phase = spec
    return centre + swing * np.sin(2.0 * np.pi * (cycles * t + phase))


def superformula(
    theta: torch.Tensor, m: int, n1: float, n2: float, n3: float
) -> torch.Tensor:
    """Gielis's superformula with a = b = 1, normalised to a maximum radius of 1."""
    c = torch.cos(m * theta / 4.0).abs() ** n2
    s = torch.sin(m * theta / 4.0).abs() ** n3
    r = (c + s) ** (-1.0 / n1)
    return r / r.max()


def generate(settings: ImageProcessingSettings = None):
    """
    Nested, twisted copies of a superformula outline with breathing exponents.

    The outline r(theta) = (|cos(m theta/4)|^n2 + |sin(m theta/4)|^n3)^(-1/n1)
    is evaluated once per frame with exponents that follow sinusoids of
    integer frequency, then drawn LAYERS times at radii from RADIUS down to
    INNER_SCALE * RADIUS, each copy rotated by a share of a twist angle that
    itself swings once per loop. Everything is periodic, so the loop closes.

    Outlines are splatted bilinearly on the GPU with samples weighted by the
    screen length they carry. Colour follows the layer index through a
    teal-to-cream palette. The frame buffer decays between frames for a short
    trail; two-scale bloom and a vignette finish the frame. The RNG seed
    chooses the symmetry m, the starting phase and the base rotation.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    m = int(SYMMETRIES[int(rng.integers(0, len(SYMMETRIES)))])
    seed_phase = rng.uniform(0.0, 1.0)
    base_angle = rng.uniform(0.0, 2.0 * np.pi)

    logger.info(
        f"Rendering Superformula Bloom (m = {m}): {FRAME_COUNT} frames @ {FPS} fps"
    )
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    theta = torch.linspace(0.0, 2.0 * np.pi, SAMPLES + 1, device=device)
    layer = torch.linspace(0.0, 1.0, LAYERS, device=device)[:, None]  # 0 outer, 1 inner
    scales = 1.0 - (1.0 - INNER_SCALE) * layer**1.35
    colours = (
        lut[(layer * (PALETTE_SIZE - 1)).to(torch.long)]
        .expand(-1, SAMPLES + 1, -1)
        .reshape(-1, 3)
    )

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(HEIGHT, WIDTH, 3).clone()
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            n1 = exponent(N1, t)
            n2 = exponent(N2, t)
            n3 = exponent(N3, t)
            twist = TWIST * np.sin(2.0 * np.pi * TWIST_CYCLES * t)

            r = superformula(theta, m, n1, n2, n3)[None, :] * scales  # (LAYERS, S+1)
            angle = theta[None, :] + base_angle + twist * layer
            px = WIDTH / 2.0 + r * torch.cos(angle) * RADIUS * WIDTH / 2.0
            py = HEIGHT / 2.0 - r * torch.sin(angle) * RADIUS * WIDTH / 2.0
            weight = (sample_lengths(px, py) * LINE_ENERGY).reshape(-1)

            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)
            flat.zero_()
            splat(flat, px.reshape(-1), py.reshape(-1), colours * weight[:, None])
            image.add_(flat.view(HEIGHT, WIDTH, 3))

            if frame < 0:
                continue

            highlights = torch.clamp(image - 130.0, min=0.0)
            composite = (
                image
                + BLOOM * blur(image, bloom_kernel)
                + WIDE_BLOOM * blur(highlights, wide_kernel)
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
