"""
Chladni Drift

Scatter sand on a metal plate and bow its edge. The plate rings, and the
sand hops away from wherever the surface is moving and settles where it
is still. Those still lines — the nodes — draw themselves: diamonds,
stars, nested rings, each note its own figure. Ernst Chladni showed them
to Napoleon in 1808 and they have been hypnotising people since.

A real plate only rings at its own notes, one figure at a time. This one
is allowed to hum several at once, and the mix of them slides slowly
round a closed cycle, so the figures do not jump from one to the next but
flow: a star folds into a lattice, the lattice opens into rings. The
bright lines are exactly where the surface would be still.

f(x, y) = Σ aₖ cos(nₖπx) cos(mₖπy) + bₖ cos(mₖπx) cos(nₖπy) = 0

Made with Python, PyTorch & NumPy.
#generativeart #chladni #cymatics #standingwaves #nodallines #physics
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

# --- Plate -------------------------------------------------------------------
# The plate spans x in [-1, 1] and y in [-ASPECT, ASPECT]; modes are cosines
# with free edges. One entry per mode: (n, m, weight cycles per loop, phase
# offset in cycles). Integer cycles make every weight, and so the whole field,
# exactly periodic over the loop.
ASPECT = HEIGHT / WIDTH
MODES = (
    (3, 5, 1, 0.00),
    (5, 2, 1, 0.37),
    (2, 7, -1, 0.61),
    (6, 4, 2, 0.15),
    (4, 8, -2, 0.82),
    (7, 1, 1, 0.50),
)
SYMMETRIC_MIX = 0.6  # weight of the swapped (m, n) partner in each mode
LINE_WIDTH = 2.6  # nodal line half-width in pixels, measured as |f| / |grad f|
CORE_WIDTH = 0.9  # hot white core of the line

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([4, 6, 14], dtype=np.float32)  # RGB
# Faint two-tone field: the plate's regions of opposite sign.
FIELD_POSITIVE = np.array([16, 34, 64], dtype=np.float32)
FIELD_NEGATIVE = np.array([46, 14, 52], dtype=np.float32)
FIELD_STRENGTH = 0.9
LINE_COLOUR = np.array([120, 210, 255], dtype=np.float32)
CORE_COLOUR = np.array([255, 255, 255], dtype=np.float32)
LINE_GAIN = 1.0
BLOOM = 0.55
BLOOM_SIGMA = 4.0
WIDE_BLOOM = 0.30
WIDE_BLOOM_SIGMA = 22.0
VIGNETTE = 0.30


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


def field(x: torch.Tensor, y: torch.Tensor, t: float, phases: np.ndarray):
    """
    The plate displacement and its gradient at every pixel.

    f = sum_k w_k(t) [cos(n pi x) cos(m pi y / ASPECT) + s cos(m pi x) cos(n pi y / ASPECT)]
    with w_k(t) = cos(2 pi (c_k t + phi_k)). The gradient is analytic, so the
    distance to the nodal line, |f| / |grad f|, is exact to first order.
    """
    f = torch.zeros_like(x)
    fx = torch.zeros_like(x)
    fy = torch.zeros_like(x)
    for (n, m, cycles, offset), phase in zip(MODES, phases):
        w = np.cos(2.0 * np.pi * (cycles * t + offset + phase))
        for a, b, s in ((n, m, 1.0), (m, n, SYMMETRIC_MIX)):
            ka = a * np.pi
            kb = b * np.pi / ASPECT
            ca = torch.cos(ka * x)
            cb = torch.cos(kb * y)
            f = f + w * s * ca * cb
            fx = fx - w * s * ka * torch.sin(ka * x) * cb
            fy = fy - w * s * kb * ca * torch.sin(kb * y)
    return f, fx, fy


def generate(settings: ImageProcessingSettings = None):
    """
    Nodal lines of a superposition of rectangular plate modes, exact loop.

    The displacement field is a weighted sum of free-edge cosine modes
    cos(n pi x) cos(m pi y), each paired with its (m, n) partner at a fixed
    ratio, and every weight is a cosine of time with an integer number of
    cycles per loop. The nodal set {f = 0} is drawn as a line of constant
    screen width by the first-order distance |f| / |grad f|, computed from the
    analytic gradient, with a narrow white core inside a wider coloured line.
    The sign of f tints the background faintly so the regions between lines
    read as the two halves of the vibrating plate. Two-scale bloom and a
    vignette finish the frame. The RNG seed sets the loop's starting phase and
    a fixed random phase offset per mode, so different seeds walk different
    closed paths through the same mode space.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    phases = rng.uniform(0.0, 1.0, size=len(MODES))

    logger.info(f"Rendering Chladni Drift: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    y, x = torch.meshgrid(
        torch.linspace(-ASPECT, ASPECT, HEIGHT, device=device),
        torch.linspace(-1.0, 1.0, WIDTH, device=device),
        indexing="ij",
    )
    pixel = 2.0 / WIDTH  # plate units per pixel
    vignette = torch.clamp(1.0 - VIGNETTE * (x**2 + (y / ASPECT) ** 2), 0.55, 1.0)[
        ..., None
    ]

    background = torch.tensor(BACKGROUND, device=device)
    positive = torch.tensor(FIELD_POSITIVE, device=device)
    negative = torch.tensor(FIELD_NEGATIVE, device=device)
    line_colour = torch.tensor(LINE_COLOUR, device=device)
    core_colour = torch.tensor(CORE_COLOUR, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0

            f, fx, fy = field(x, y, t, phases)
            gradient = torch.hypot(fx, fy).clamp(min=1e-6)
            distance = f.abs() / gradient / pixel  # pixels to the nodal line

            line = torch.exp(-0.5 * (distance / LINE_WIDTH) ** 2)
            core = torch.exp(-0.5 * (distance / CORE_WIDTH) ** 2)

            # Soft two-tone field from the sign and strength of f.
            amplitude = torch.tanh(2.0 * f)[..., None]
            plate = FIELD_STRENGTH * (
                positive * amplitude.clamp(min=0.0)
                + negative * (-amplitude).clamp(min=0.0)
            )

            image = background + plate
            image = image + LINE_GAIN * (
                line[..., None] * line_colour + core[..., None] * core_colour
            )

            glow = line[..., None] * line_colour + core[..., None] * core_colour
            composite = (
                image
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
