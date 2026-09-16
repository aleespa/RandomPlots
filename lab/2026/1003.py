"""
Maurer Walk

A rose curve is the simplest flower mathematics knows: r = sin(nθ),
petals laid out round a point. In 1987 Peter Maurer found a way to
make it far stranger. Walk round the rose taking big fixed strides —
say 71 degrees at a time — and join each footfall to the next with a
straight line. The 360 chords weave a lattice inside the petals that
looks nothing like the rose, yet is entirely determined by it.

Here the stride is not held fixed. It drifts by half a degree either
side of its whole number, and because the three-hundredth footfall
moves three hundred times further than the first, that tiny drift
sends the far end of the walk sweeping round the rose, unweaving the
lattice and weaving a new one. Two roses turn in the frame, half a
cycle apart. The drift is one whole swing, so the loop is exact.

r = sin(nθ),  θₖ = k·d,  k = 0 … 360

Made with Python, PyTorch & NumPy.
#generativeart #maurerrose #rosecurve #stringart #geometry #mathematics
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

# --- Roses -------------------------------------------------------------------
PETALS = (2, 3, 4, 5, 6, 7)  # n, per rose, chosen by the seed
STRIDES = (29, 31, 37, 41, 47, 71, 97)  # d in degrees, per rose, chosen by the seed
STEPS = 360  # footfalls per walk
STRIDE_DRIFT = 0.5  # degrees either side of the whole-number stride
DRIFT_CYCLES = 1  # integer -> exact loop
SAMPLES = 600  # samples per chord
# Two roses: (vertical centre as a fraction of the half-height, radius as a
# fraction of the half-width, phase offset in loops).
ROSES = ((0.48, 0.88, 0.0), (-0.48, 0.88, 0.5))
ROSE_ENERGY = 40.0  # the rose curve itself, drawn faintly behind the walk
ROSE_SAMPLES = 4000

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([8, 4, 10], dtype=np.float32)  # RGB
PALETTE = np.array(  # along the walk, ping-ponged
    [
        [0.35, 0.05, 0.45],
        [0.85, 0.10, 0.50],
        [1.00, 0.45, 0.35],
        [1.00, 0.85, 0.50],
        [1.00, 1.00, 0.92],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
COLOUR_BANDS = 2
LINE_ENERGY = 26.0
ROSE_COLOUR = np.array([0.45, 0.20, 0.55], dtype=np.float32)
TRAIL_DECAY = 0.84
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


def sample_lengths(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    steps = torch.hypot(px[:, 1:] - px[:, :-1], py[:, 1:] - py[:, :-1])
    lengths = torch.empty_like(px)
    lengths[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    lengths[:, 0] = steps[:, 0]
    lengths[:, -1] = steps[:, -1]
    return lengths


def generate(settings: ImageProcessingSettings = None):
    """
    Maurer roses with a drifting stride, as an exact loop.

    For a rose r = sin(n theta), the Maurer walk visits theta_k = k d degrees
    for k = 0..STEPS and joins consecutive points with chords. With d whole
    the walk closes and forms the classic lattice; here d(t) = d0 +
    STRIDE_DRIFT sin(2 pi DRIFT_CYCLES t), so the far footfalls, whose angle
    is k times the drift, sweep round the rose while the near ones barely
    move, and the lattice continuously unweaves and reweaves. Two roses run
    half a loop apart; the seed picks n and d0 for each and the starting
    phase.

    Chords are splatted bilinearly on the GPU, each sample weighted by the
    chord length it represents. Colour follows the footfall index through a
    plum-to-cream palette that ping-pongs COLOUR_BANDS times. The rose curve
    itself is drawn faintly beneath. A decaying buffer gives a short trail;
    two-scale bloom and a vignette finish the frame.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    petals = [
        int(PETALS[i]) for i in rng.choice(len(PETALS), size=len(ROSES), replace=False)
    ]
    strides = [
        float(STRIDES[i])
        for i in rng.choice(len(STRIDES), size=len(ROSES), replace=False)
    ]
    rotations = rng.uniform(0.0, 2.0 * np.pi, size=len(ROSES))

    logger.info(
        f"Rendering Maurer Walk n={petals} d={strides}: {FRAME_COUNT} frames @ {FPS} fps"
    )
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    k = torch.arange(STEPS + 1, device=device, dtype=torch.float32)
    u = torch.linspace(0.0, 1.0, SAMPLES, device=device)[None, :]
    along = k[:-1] / STEPS
    shade = 0.5 - 0.5 * torch.cos(2.0 * np.pi * COLOUR_BANDS * along)
    chord_colours = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)][
        :, None, :
    ]  # (STEPS, 1, 3)
    rose_colour = torch.tensor(ROSE_COLOUR, device=device)
    rose_theta = torch.linspace(0.0, 2.0 * np.pi, ROSE_SAMPLES, device=device)[None, :]

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(HEIGHT, WIDTH, 3).clone()
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)
            flat.zero_()

            for (cy, radius, offset), n, d0, rot in zip(
                ROSES, petals, strides, rotations
            ):
                phase = (t + offset) % 1.0
                d = np.deg2rad(
                    d0 + STRIDE_DRIFT * np.sin(2.0 * np.pi * DRIFT_CYCLES * phase)
                )
                r_px = radius * WIDTH / 2.0
                centre_y = HEIGHT / 2.0 - cy * HEIGHT / 2.0

                # The rose itself, faintly.
                rr = torch.sin(n * rose_theta)
                rx = WIDTH / 2.0 + r_px * rr * torch.cos(rose_theta + rot)
                ry = centre_y - r_px * rr * torch.sin(rose_theta + rot)
                weight = (sample_lengths(rx, ry) * ROSE_ENERGY).reshape(-1)
                splat(
                    flat,
                    rx.reshape(-1),
                    ry.reshape(-1),
                    rose_colour[None, :] * weight[:, None],
                )

                # The walk.
                theta = k * d
                rad = torch.sin(n * theta)
                x = WIDTH / 2.0 + r_px * rad * torch.cos(theta + rot)
                y = centre_y - r_px * rad * torch.sin(theta + rot)
                x0, x1 = x[:-1, None], x[1:, None]
                y0, y1 = y[:-1, None], y[1:, None]
                length = torch.hypot(x1 - x0, y1 - y0)  # (STEPS, 1)
                weight = length / SAMPLES * LINE_ENERGY
                deposit = (
                    (chord_colours * weight[..., None])
                    .expand(-1, SAMPLES, -1)
                    .reshape(-1, 3)
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
