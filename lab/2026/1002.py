"""
Dragon Unfold

Fold a strip of paper in half, and in half again, always the same way,
a dozen times. Open it so that every crease is a right angle and look
down on the edge: it has become the dragon curve, a jagged coastline
that never crosses itself and tiles the plane with copies of itself.
It was found by NASA physicists in the 1960s and made famous in the
margins of Jurassic Park.

Here the creases are not held at right angles. The fold opens and
closes together — every crease at the same angle, swinging between
gently bent and fully square — so the dragon breathes: unfolding
towards a loose spiral of paper, then folding back into its tight,
self-similar coil. The swing is one whole cycle, so the loop is exact.

Made with Python, PyTorch & NumPy.
#generativeart #dragoncurve #fractal #paperfolding #geometry
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

# Frames rendered before frame 0 so the motion trail is at steady state.
WARMUP_FRAMES = 16

# --- Curve -------------------------------------------------------------------
ORDER = 13  # 2^ORDER segments
SAMPLES_PER_SEGMENT = 28
FOLD_MIN = np.deg2rad(52.0)  # loosest crease
FOLD_MAX = np.deg2rad(90.0)  # the true dragon
FOLD_CYCLES = 1  # integer -> exact loop
MARGIN = 0.07  # fraction of the frame kept clear around the curve
TURN_CYCLES = 1  # slow whole-curve rotation, whole turns per loop

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([6, 5, 12], dtype=np.float32)  # RGB
PALETTE = np.array(  # along the curve, ping-ponged
    [
        [0.15, 0.10, 0.55],
        [0.50, 0.10, 0.75],
        [0.95, 0.25, 0.55],
        [1.00, 0.65, 0.30],
        [1.00, 0.95, 0.80],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
COLOUR_BANDS = 3  # how many times the palette ping-pongs along the curve
LINE_ENERGY = 170.0
TRAIL_DECAY = 0.70
BLOOM = 0.35
BLOOM_SIGMA = 2.5
WIDE_BLOOM = 0.14
WIDE_BLOOM_SIGMA = 18.0
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


def folding_sequence(order: int) -> np.ndarray:
    """
    The regular paper-folding sequence: turn direction at each crease,
    +1 left / -1 right, for 2^order - 1 creases.
    """
    turns = np.array([], dtype=np.int64)
    for _ in range(order):
        turns = np.concatenate([turns, [1], -turns[::-1]])
    return turns


def generate(settings: ImageProcessingSettings = None):
    """
    The dragon curve with a breathing crease angle, as an exact loop.

    The curve is built from the regular paper-folding sequence: 2^ORDER unit
    segments with a turn of +/- alpha at every crease. alpha = 90 degrees is
    the dragon; smaller alpha unfolds it continuously. Here alpha swings
    between FOLD_MIN and FOLD_MAX as a cosine with FOLD_CYCLES cycles per
    loop, and the whole curve turns TURN_CYCLES whole turns, so the geometry
    closes on itself. Each frame the curve is fitted to the frame by its
    bounding box, so the unfolding reads as a change of shape rather than of
    size.

    Segments are splatted bilinearly on the GPU, each sample weighted by the
    screen length it carries. Colour follows position along the curve
    through a violet-to-cream palette that ping-pongs COLOUR_BANDS times, so
    the curve's self-similar halves read in matching bands. The frame buffer
    decays for a short trail; two-scale bloom and a vignette finish the
    frame. The seed sets the starting phase and the base orientation.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    base_angle = rng.uniform(0.0, 2.0 * np.pi)

    logger.info(
        f"Rendering Dragon Unfold (order {ORDER}): {FRAME_COUNT} frames @ {FPS} fps"
    )
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    turns = torch.tensor(folding_sequence(ORDER), dtype=torch.float32, device=device)
    segments = len(turns) + 1
    u = torch.linspace(0.0, 1.0, SAMPLES_PER_SEGMENT, device=device)[None, :]

    along = torch.linspace(0.0, 1.0, segments, device=device)
    shade = 0.5 - 0.5 * torch.cos(2.0 * np.pi * COLOUR_BANDS * along)
    colours = lut[(shade * (PALETTE_SIZE - 1)).to(torch.long)]  # (segments, 3)
    colours = colours[:, None, :].expand(-1, SAMPLES_PER_SEGMENT, -1).reshape(-1, 3)

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(HEIGHT, WIDTH, 3).clone()
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)
    fit = 1.0 - 2.0 * MARGIN

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            alpha = 0.5 * (FOLD_MAX + FOLD_MIN) + 0.5 * (FOLD_MAX - FOLD_MIN) * np.cos(
                2.0 * np.pi * FOLD_CYCLES * t
            )
            heading = torch.cat(
                [torch.zeros(1, device=device), torch.cumsum(turns * alpha, dim=0)]
            )
            heading = heading + base_angle + 2.0 * np.pi * TURN_CYCLES * t
            steps = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
            vertices = torch.cat(
                [torch.zeros(1, 2, device=device), torch.cumsum(steps, dim=0)]
            )

            # Fit the bounding box into the frame, preserving aspect.
            low = vertices.min(dim=0).values
            high = vertices.max(dim=0).values
            centre = 0.5 * (low + high)
            span = high - low
            scale = min(fit * WIDTH / float(span[0]), fit * HEIGHT / float(span[1]))
            screen = (vertices - centre) * scale
            sx = WIDTH / 2.0 + screen[:, 0]
            sy = HEIGHT / 2.0 - screen[:, 1]

            px = sx[:-1, None] + u * (sx[1:] - sx[:-1])[:, None]
            py = sy[:-1, None] + u * (sy[1:] - sy[:-1])[:, None]
            weight = torch.full(
                (segments, SAMPLES_PER_SEGMENT),
                scale / SAMPLES_PER_SEGMENT * LINE_ENERGY,
                device=device,
            )

            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)
            flat.zero_()
            splat(
                flat,
                px.reshape(-1),
                py.reshape(-1),
                colours * weight.reshape(-1)[:, None],
            )
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
