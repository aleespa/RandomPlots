"""
Chord Loom

Mark a few hundred points evenly round a circle and number them. Draw a
line from every point to the one with twice its number, counting round
again when you run out. A heart-shaped curve appears out of the straight
lines — a cardioid, the same curve light makes in the bottom of a cup.
Use three times the number and it grows a second lobe; four gives three.

Nothing says the multiplier has to be whole. Here it slides smoothly
from two up to five and back, and between the tidy whole-number pictures
the loom passes through wild, unnamed tangles that are just as exact.
Two wheels turn in the frame half a cycle apart, so one is at rest when
the other is at its most chaotic, and the sweep closes on itself.

chord: i → k·i (mod N)

Made with Python, PyTorch & NumPy.
#generativeart #cardioid #timestable #stringart #modulararithmetic
#geometry #mathematics #randomplots #creativecoding #algorithmicart
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

# --- Loom --------------------------------------------------------------------
POINTS = 420  # points round each circle
SAMPLES = 700  # samples per chord
K_MIN, K_MAX = 2.0, 5.0  # the multiplier sweeps K_MIN..K_MAX..K_MIN once per loop
# Two wheels: (vertical centre as a fraction of the half-height, radius as a
# fraction of the half-width, phase offset in loops).
WHEELS = ((0.48, 0.88, 0.0), (-0.48, 0.88, 0.5))
SPIN_TURNS = 0  # whole turns of each wheel over the loop; 0 keeps them still

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([4, 6, 12], dtype=np.float32)  # RGB
PALETTE = np.array(  # by chord length: short -> long
    [
        [0.05, 0.30, 0.55],
        [0.00, 0.65, 0.80],
        [0.40, 0.95, 0.85],
        [1.00, 0.90, 0.55],
        [1.00, 1.00, 0.95],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
LINE_ENERGY = 30.0
TRAIL_DECAY = 0.86
BLOOM = 0.40
BLOOM_SIGMA = 3.0
WIDE_BLOOM = 0.16
WIDE_BLOOM_SIGMA = 22.0
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


def generate(settings: ImageProcessingSettings = None):
    """
    The times-table cardioid with a continuously sweeping multiplier.

    POINTS points sit at angles 2 pi i / N on a circle. A chord joins point i
    to the angle 2 pi k i / N, with k real: k(t) = mid + half * cos(2 pi t),
    so it sweeps K_MIN..K_MAX..K_MIN exactly once per loop and the figure
    closes. Whole-number k gives the familiar epicycloids (k = 2 cardioid,
    3 nephroid, ...); between them the chord ends run round the circle at
    incommensurate rates and the envelope dissolves. Two wheels run half a
    loop apart.

    Chords are straight lines splatted bilinearly on the GPU, each sample
    weighted by the chord length it represents so brightness is per pixel of
    line. Colour follows chord length through a blue-to-white palette, which
    makes the envelope, where many chords are tangent, stand out from the
    crossing fill. A decaying frame buffer gives a short trail; two-scale
    bloom and a vignette finish the frame. The seed sets the starting phase
    and a fixed rotation of each wheel.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    wheel_angles = rng.uniform(0.0, 2.0 * np.pi, size=len(WHEELS))

    logger.info(f"Rendering Chord Loom: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    i = torch.arange(POINTS, device=device, dtype=torch.float32)[:, None]
    u = torch.linspace(0.0, 1.0, SAMPLES, device=device)[None, :]

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(HEIGHT, WIDTH, 3).clone()
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)
    k_mid = 0.5 * (K_MIN + K_MAX)
    k_half = 0.5 * (K_MAX - K_MIN)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)
            flat.zero_()

            for (cy, radius, offset), wheel_angle in zip(WHEELS, wheel_angles):
                phase = (t + offset) % 1.0
                k = k_mid + k_half * np.cos(2.0 * np.pi * phase)
                spin = wheel_angle + 2.0 * np.pi * SPIN_TURNS * t
                a0 = 2.0 * np.pi * i / POINTS + spin
                a1 = 2.0 * np.pi * k * i / POINTS + spin
                r = radius * WIDTH / 2.0
                x0 = WIDTH / 2.0 + r * torch.cos(a0)
                y0 = HEIGHT / 2.0 - (r * torch.sin(a0) + cy * HEIGHT / 2.0)
                x1 = WIDTH / 2.0 + r * torch.cos(a1)
                y1 = HEIGHT / 2.0 - (r * torch.sin(a1) + cy * HEIGHT / 2.0)

                length = torch.hypot(x1 - x0, y1 - y0)  # (POINTS, 1)
                shade = (length / (2.0 * r)).clamp(0.0, 1.0)
                colours = lut[
                    (shade * (PALETTE_SIZE - 1)).to(torch.long)
                ]  # (POINTS, 1, 3)
                weight = length / SAMPLES * LINE_ENERGY
                deposit = (
                    (colours * weight[..., None]).expand(-1, SAMPLES, -1).reshape(-1, 3)
                )

                px = (x0 + u * (x1 - x0)).reshape(-1)
                py = (y0 + u * (y1 - y0)).reshape(-1)
                splat(flat, px, py, deposit)

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
