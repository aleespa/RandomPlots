"""
Ricci's Thread

A perfectly regular grid is transformed by a nonlinear mathematical field.
Every visible mark begins as a simple straight line, but coupled sinusoidal
deformations fold the grid into waves, knots and dense woven structures.

The colour is not decoration: each thread is lit by how much the field
squeezes it. Where the deformation crowds the grid together the threads
run white-hot, where it pulls them apart they cool to deep blue, so the
folds of the field light themselves up as they move.

The animation travels a closed path of transformations. Nothing is
simulated and nothing is random frame by frame: the complexity comes
entirely from the geometry.

Made with Python, PyTorch & NumPy.
#generativeart #mathematics #nonlinear #generativegeometry #randomplots
#creativecoding #algorithmicart #mathisbeautiful
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
FPS = int(os.environ.get("RP_FPS", "60"))
DURATION = 20.0
FRAME_COUNT = int(FPS * DURATION)

WIDTH, HEIGHT = 1080, 1920  # 9:16 portrait for Reels / Stories

# Threads are splatted at this resolution and the frame is upscaled from it.
RENDER_WIDTH, RENDER_HEIGHT = 540, 960
PNG_WRITERS = 6
# Frames are an intermediate for ffmpeg, so encode them fast rather than small.
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

# Frames rendered before frame 0 is written, so the trail buffer is already at
# its steady state when the clip starts. Without this the opening second fades
# up from an empty canvas and the loop join visibly jumps.
WARMUP_FRAMES = 24

# --- Grid --------------------------------------------------------------------
N_HORIZONTAL = 105
N_VERTICAL = 105

# Samples per thread. Splatting has no line segments -- a thread is drawn by its
# samples alone, so they must land closer together than a pixel or it beads.
SAMPLES = 1600

# --- Transformation ----------------------------------------------------------
WARP_X = 0.34
WARP_Y = 0.34
FREQUENCY_X = 4.2
FREQUENCY_Y = 5.0
QUADRATIC_WARP = 0.20
RADIAL_WARP = 0.14
CROSS_COUPLING = 0.16  # how strongly the two coordinate directions interact

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([3, 10, 19], dtype=np.float32)  # RGB

PALETTE = np.array(
    [
        [0.00, 0.28, 0.44],
        [0.00, 0.52, 0.68],
        [0.00, 0.75, 0.82],
        [0.00, 0.95, 0.78],
        [0.55, 1.00, 0.88],
        [1.00, 1.00, 1.00],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256

# Brightness deposited per pixel of thread length. Each sample carries the
# length of thread it represents, so brightness does not depend on how densely
# the thread happens to be sampled.
LINE_ENERGY = 95.0
TRAIL_DECAY = 0.82
BLOOM = 0.16
BLOOM_SIGMA = 2.4
WIDE_BLOOM = 0.07
WIDE_BLOOM_SIGMA = 14.0
BLOOM_THRESHOLD = 90.0  # only genuinely bright pixels feed the wide glow

# Colour from local compression: t = 1/2 - gain * log(length / rest length).
# Below 1 the field is crowding the grid together and the thread lights up;
# above 1 it is being pulled apart and cools.
COMPRESSION_GAIN = 0.55
FAMILY_TINT = (-0.04, 0.04)  # keeps the two families distinguishable
TINT_DRIFT = 0.06  # slow palette breathing over the loop

# The weave: each family lays a soft dark mask along its own threads before it
# is added, knocking back whatever is underneath, so the second family passes
# visibly *over* the first instead of merging with it.
HALO_SIGMA = 1.6
HALO_GAIN = 12.0  # how quickly coverage saturates the mask
HALO_STRENGTH = 0.75


def select_device() -> torch.device:
    """CUDA when there is a GPU, CPU otherwise. Everything else is identical."""
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("no CUDA device, rendering on CPU")
    return torch.device("cpu")


def palette_lut(device: torch.device) -> torch.Tensor:
    """The oceanic palette as a 256-entry RGB lookup table, 0-1."""
    positions = np.linspace(0.0, 1.0, len(PALETTE))
    t = np.linspace(0.0, 1.0, PALETTE_SIZE)
    lut = np.stack(
        [np.interp(t, positions, PALETTE[:, channel]) for channel in range(3)], axis=1
    )
    return torch.tensor(lut, dtype=torch.float32, device=device)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """1-D Gaussian, normalised, for separable blurring."""
    radius = max(1, int(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    return kernel / kernel.sum()


def blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of an (H, W) or (H, W, C) tensor."""
    single_channel = image.dim() == 2
    planes = image[None] if single_channel else image.permute(2, 0, 1)
    channels = planes.shape[0]
    radius = (kernel.numel() - 1) // 2

    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    out = F.conv2d(
        F.pad(planes[None], (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    out = F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels
    )
    out = out[0]
    return out[0] if single_channel else out.permute(1, 2, 0)


def create_grid(device: torch.device):
    """
    Build the grid as two families of polylines, one row per line.

    Horizontal lines run along x at fixed y; vertical lines along y at fixed x.
    """
    x = torch.linspace(-2.0, 2.0, SAMPLES, device=device)
    y = torch.linspace(-3.0, 3.0, SAMPLES, device=device)
    horizontal_values = torch.linspace(-3.0, 3.0, N_HORIZONTAL, device=device)
    vertical_values = torch.linspace(-2.0, 2.0, N_VERTICAL, device=device)

    horizontal_x = x[None, :].expand(N_HORIZONTAL, SAMPLES).contiguous()
    horizontal_y = horizontal_values[:, None].expand(N_HORIZONTAL, SAMPLES).contiguous()
    vertical_x = vertical_values[:, None].expand(N_VERTICAL, SAMPLES).contiguous()
    vertical_y = y[None, :].expand(N_VERTICAL, SAMPLES).contiguous()

    return (horizontal_x, horizontal_y), (vertical_x, vertical_y)


def transform(x: torch.Tensor, y: torch.Tensor, phase: float):
    """
    Apply a coupled nonlinear transformation to a whole collection of points.

    Sinusoidal displacement, quadratic phase distortion, cross-coupling and
    radial deformation, all driven by the single angle theta = 2*pi*phase, so
    every parameter follows a closed trajectory over one loop.
    """
    theta = 2.0 * np.pi * phase

    # Smoothly varying amplitudes.
    ax = WARP_X * (0.72 + 0.28 * np.sin(theta))
    ay = WARP_Y * (0.72 + 0.28 * np.cos(theta))

    # Frequencies follow a closed trajectory.
    bx = FREQUENCY_X + 0.55 * np.sin(theta)
    by = FREQUENCY_Y + 0.65 * np.cos(theta)

    # Primary nonlinear deformation.
    dx = ax * torch.sin(by * y + CROSS_COUPLING * torch.sin(bx * x + theta))
    dy = ay * torch.sin(bx * x + CROSS_COUPLING * torch.cos(by * y - theta))

    # Quadratic deformation -- curvature that grows away from the centre.
    dx = dx + QUADRATIC_WARP * torch.sin(by * y + 2.2 * y * y + theta)
    dy = dy + QUADRATIC_WARP * torch.cos(bx * x + 1.8 * x * x - theta)

    # Radial deformation.
    r = torch.sqrt(x * x + y * y)
    radial = RADIAL_WARP * torch.sin(3.0 * r - theta)
    safe_r = torch.clamp(r, min=1e-5)
    dx = dx + radial * x / safe_r
    dy = dy + radial * y / safe_r

    return x + dx, y + dy


def map_to_screen(x: torch.Tensor, y: torch.Tensor):
    """Map mathematical coordinates to the portrait render canvas."""
    px = RENDER_WIDTH / 2.0 + x * RENDER_WIDTH / 4.15
    py = RENDER_HEIGHT / 2.0 - y * RENDER_HEIGHT / 6.65
    return px, py


def sample_lengths(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """
    Screen-space length of thread carried by each sample.

    Each sample takes the mean of the two steps meeting at it; the ends take the
    single step they have.
    """
    steps = torch.hypot(px[:, 1:] - px[:, :-1], py[:, 1:] - py[:, :-1])
    lengths = torch.empty_like(px)
    lengths[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    lengths[:, 0] = steps[:, 0]
    lengths[:, -1] = steps[:, -1]
    return lengths


def splat(
    flat_colour: torch.Tensor,
    flat_coverage: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    colours: torch.Tensor,
    weights: torch.Tensor,
):
    """
    Deposit every sample of a family into an accumulation buffer.

    This replaces line rasterisation entirely. Each sample is spread over the
    four pixels around it with bilinear weights -- which is where the
    anti-aliasing comes from -- and deposits add, so threads crossing each other
    build up instead of painting over one another. `flat_colour` collects
    colour, `flat_coverage` the same deposits without it, for the weave mask;
    both are flat (HEIGHT*WIDTH, ...) views of the frame.
    """
    px = px.reshape(-1)
    py = py.reshape(-1)
    weights = weights.reshape(-1)

    x0 = torch.floor(px)
    y0 = torch.floor(py)
    fx = px - x0
    fy = py - y0
    x0 = x0.to(torch.long)
    y0 = y0.to(torch.long)

    # Colour only ever lands pre-multiplied by the sample weight, so the product
    # is formed once and each corner just scales it.
    weighted_colour = colours.reshape(-1, 3) * weights[:, None]

    for offset_x, offset_y, corner in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        ix = x0 + offset_x
        iy = y0 + offset_y
        inside = (ix >= 0) & (ix < RENDER_WIDTH) & (iy >= 0) & (iy < RENDER_HEIGHT)

        # Samples off the canvas stay in the batch with a clamped index and a
        # zeroed weight. Compacting them out with a boolean mask instead costs a
        # host synchronisation and four gathers on every corner of every frame.
        index = iy.clamp_(0, RENDER_HEIGHT - 1) * RENDER_WIDTH + ix.clamp_(
            0, RENDER_WIDTH - 1
        )
        scale = corner * inside
        amount = scale * weights

        flat_coverage.index_add_(0, index, amount)
        flat_colour.index_add_(0, index, weighted_colour * scale[:, None])


def draw_family(
    image: torch.Tensor,
    family,
    phase: float,
    reference: float,
    tint: float,
    lut: torch.Tensor,
    halo_kernel: torch.Tensor,
    background: torch.Tensor,
    flat_colour: torch.Tensor,
    flat_coverage: torch.Tensor,
):
    """
    Deform, colour and splat one family of threads on top of `image`.

    The family's own coverage, blurred, is the weave mask: it darkens what is
    already in the frame before the family's colour is added, so the family
    reads as lying over everything drawn before it.
    """
    x, y = transform(*family, phase)
    px, py = map_to_screen(x, y)
    lengths = sample_lengths(px, py)

    # Colour by compression; weight by the length of thread each sample carries.
    ratio = torch.clamp(lengths, min=1e-6) / reference
    t = torch.clamp(0.5 - COMPRESSION_GAIN * torch.log(ratio) + tint, 0.0, 1.0)
    colours = lut[(t * (PALETTE_SIZE - 1)).to(torch.long)]
    weights = lengths * (LINE_ENERGY / reference)

    flat_colour.zero_()
    flat_coverage.zero_()
    splat(flat_colour, flat_coverage, px, py, colours, weights)
    buffer = flat_colour.view(RENDER_HEIGHT, RENDER_WIDTH, 3)
    coverage = flat_coverage.view(RENDER_HEIGHT, RENDER_WIDTH)

    # Weave, then add.
    mask = torch.clamp(blur(coverage, halo_kernel) * HALO_GAIN, 0.0, 1.0)
    image.sub_(background).mul_(1.0 - HALO_STRENGTH * mask[..., None]).add_(background)
    image.add_(buffer)


def generate(settings: ImageProcessingSettings = None):
    """
    A 9:16 looping animation built from a nonlinear deformation of a grid.

    Every visible element is a mathematical line. The two families of lines are
    pushed through coupled sinusoidal, quadratic and radial displacements; the
    woven structures are simply where the deformed families cross. All of the
    transformation parameters are functions of theta = 2*pi*phase, so the
    geometry at the end of the clip is identical to the geometry at the start.

    Rendering is GPU point splatting rather than line rasterisation: each thread
    is sampled finely enough that consecutive samples land under a pixel apart,
    and every sample is deposited into the frame with bilinear weights. Crossing
    threads therefore accumulate, anti-aliasing comes from the splat weights,
    and each sample carries its own colour -- taken from the local Jacobian of
    the map, measured as the ratio of the thread's screen-space length to its
    undeformed length, so compression drives a thread towards white and
    extension towards deep blue. Each sample also carries the length of thread
    it represents, which keeps brightness independent of sampling density.

    The frame buffer persists between frames with a decay factor, leaving a
    short motion trail; two-scale bloom and the vignette are applied to a copy,
    so they colour the frame without compounding in the buffer. The whole
    pipeline stays on the GPU until the finished frame is copied back for PNG
    encoding, which the writer pool overlaps with the next frame.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    # The seed picks where in the closed cycle the clip starts. The animation
    # stays perfectly periodic -- it is a rotation of the same loop.
    seed_phase = rng.uniform(0.0, 1.0)

    logger.info(
        f"Rendering Ricci's Thread: {FRAME_COUNT} frames @ {FPS} fps "
        f"({DURATION:.1f}s), {RENDER_WIDTH}x{RENDER_HEIGHT} -> {WIDTH}x{HEIGHT}"
    )

    frames_path = settings.frames_path
    horizontal, vertical = create_grid(device)
    lut = palette_lut(device)

    # Undeformed sample spacing per family, in screen pixels -- the reference
    # the compression colouring is measured against.
    horizontal_reference = float(sample_lengths(*map_to_screen(*horizontal)).mean())
    vertical_reference = float(sample_lengths(*map_to_screen(*vertical)).mean())

    halo_kernel = gaussian_kernel(HALO_SIGMA, device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_bloom_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, RENDER_HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, RENDER_WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - 0.18 * (xx**2 + yy**2), 0.72, 1.0)[..., None]

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(RENDER_HEIGHT, RENDER_WIDTH, 3).clone()

    # Splat accumulators, allocated once and cleared per family rather than
    # reallocated twice a frame.
    flat_colour = torch.zeros(RENDER_HEIGHT * RENDER_WIDTH, 3, device=device)
    flat_coverage = torch.zeros(RENDER_HEIGHT * RENDER_WIDTH, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for index in range(-WARMUP_FRAMES, FRAME_COUNT):
            phase = (2.0 * index / FRAME_COUNT + seed_phase) % 1.0
            drift = TINT_DRIFT * np.sin(2.0 * np.pi * phase)

            # Fade the previous geometry, but never below the background.
            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)

            # Horizontals first, so the verticals weave over them.
            draw_family(
                image,
                horizontal,
                phase,
                horizontal_reference,
                FAMILY_TINT[0] + drift,
                lut,
                halo_kernel,
                background,
                flat_colour,
                flat_coverage,
            )
            draw_family(
                image,
                vertical,
                phase,
                vertical_reference,
                FAMILY_TINT[1] - drift,
                lut,
                halo_kernel,
                background,
                flat_colour,
                flat_coverage,
            )

            if index < 0:  # warm-up pass: fill the trail buffer, write nothing
                continue

            # Two-scale bloom: a tight halo on everything, and a wide
            # atmospheric glow fed only by the bright crossings.
            highlights = torch.clamp(image - BLOOM_THRESHOLD, min=0.0)
            composite = image + BLOOM * blur(image, bloom_kernel)
            composite = composite + WIDE_BLOOM * blur(highlights, wide_bloom_kernel)
            composite = composite * vignette

            frame = (
                F.interpolate(
                    composite.permute(2, 0, 1)[None],
                    size=(HEIGHT, WIDTH),
                    mode="bicubic",
                    align_corners=False,
                )[0]
                .permute(1, 2, 0)
                .clamp_(0.0, 255.0)
                .to(torch.uint8)
                .flip(-1)  # RGB -> BGR for OpenCV
                .cpu()
                .numpy()
            )

            pending.append(
                pool.submit(
                    cv2.imwrite,
                    str(frames_path / f"frame{index:04d}.png"),
                    frame,
                    PNG_PARAMS,
                )
            )

            # Bound the queue: without this every finished frame stays resident
            # as an 1080x1920 BGR array until the render ends.
            while len(pending) > 2 * PNG_WRITERS:
                if not pending.pop(0).result():
                    raise OSError("Could not write a frame")

            if index % FPS == 0:
                logger.info(f"Frame {index}/{FRAME_COUNT}")

        for task in pending:
            if not task.result():
                raise OSError("Could not write a frame")

    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
