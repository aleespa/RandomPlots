"""
Torus Knot Turning

Wind a string around a doughnut: three times through the hole while it
goes seven times around the outside, and join the ends. The result is a
torus knot, a closed curve that cannot be untangled without cutting it,
and one of the first shapes knot theory learned to name.

Here the knot is a hollow tube drawn as a cage of parallel strands,
stretched tall to stand in the frame. It turns once about its long axis
over the length of the clip while rocking gently towards you and away,
so it arrives back exactly where it started. Nothing else changes. Depth is the only shading: the strands nearest you burn
white, the ones passing behind fall to violet, and where the tube crosses
in front of itself you can read which strand is over and which under.

x = (R + r cos qφ) cos pφ,  y = (R + r cos qφ) sin pφ,  z = r sin qφ

Made with Python, PyTorch & NumPy.
#generativeart #torusknot #knottheory #topology #wireframe #geometry
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

# --- Knot --------------------------------------------------------------------
# (p, q) pairs the seed chooses from: p times through the hole, q around.
KNOTS = ((3, 7), (2, 7), (3, 8), (4, 9), (5, 8))
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.42  # the doughnut's tube
TUBE_RADIUS = 0.13  # the knot's own tube, drawn as a cage of strands
STRANDS = 10
SAMPLES = 24000  # samples along the curve; under a pixel apart once stretched
RINGS = 90  # cross-section rings drawn along the tube, for the cage look
RING_SAMPLES = 64
STRETCH = 1.75  # the torus is stretched along the frame's long axis
# Motion over the loop: whole turns about the long (vertical) axis, and a rock
# about the horizontal axis that swings ROCK radians with ROCK_CYCLES cycles.
TURNS_Y = 1
ROCK = 0.45
ROCK_CYCLES = 1
TILT = 0.55  # mean tilt so the knot never sits flat to the screen
CAMERA_DISTANCE = 4.2  # perspective: eye distance in knot units
SCALE = 0.30  # fraction of the width the unit radius maps to

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([6, 4, 16], dtype=np.float32)  # RGB
PALETTE = np.array(  # far -> near
    [
        [0.18, 0.05, 0.42],
        [0.45, 0.10, 0.70],
        [0.85, 0.20, 0.60],
        [1.00, 0.55, 0.45],
        [1.00, 0.95, 0.85],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
LINE_ENERGY = 95.0
RING_ENERGY = 24.0
DEPTH_GAIN = 1.6  # near strands are this much brighter than far ones
TRAIL_DECAY = 0.55
BLOOM = 0.35
BLOOM_SIGMA = 3.0
WIDE_BLOOM = 0.12
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
    """Screen-space length carried by each sample, per row of samples."""
    steps = torch.hypot(px[:, 1:] - px[:, :-1], py[:, 1:] - py[:, :-1])
    lengths = torch.empty_like(px)
    lengths[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    lengths[:, 0] = steps[:, 0]
    lengths[:, -1] = steps[:, -1]
    return lengths


def torus_knot_frame(phi: torch.Tensor, p: int, q: int):
    """
    The knot curve and an orthonormal frame (normal, binormal) along it.

    The frame is built from the curve's tangent and the direction to the
    torus's central circle, which is smooth everywhere on a torus knot and
    avoids the twisting that a Frenet frame develops.
    """
    tube = MAJOR_RADIUS + MINOR_RADIUS * torch.cos(q * phi)
    x = tube * torch.cos(p * phi)
    y = tube * torch.sin(p * phi)
    z = MINOR_RADIUS * torch.sin(q * phi)
    curve = torch.stack([x, y, z], dim=-1)

    # Tangent by central differences on the closed curve.
    tangent = curve.roll(-1, dims=0) - curve.roll(1, dims=0)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)

    # Outward direction from the torus core circle.
    core = torch.stack(
        [MAJOR_RADIUS * torch.cos(p * phi), MAJOR_RADIUS * torch.sin(p * phi), 0 * phi],
        dim=-1,
    )
    outward = curve - core
    outward = outward - (outward * tangent).sum(-1, keepdim=True) * tangent
    normal = outward / outward.norm(dim=-1, keepdim=True)
    binormal = torch.cross(tangent, normal, dim=-1)
    return curve, normal, binormal


def rotation(t: float) -> torch.Tensor:
    """Turn about the long axis, then rock about the horizontal one; exact per loop."""
    ax = TILT + ROCK * np.sin(2.0 * np.pi * ROCK_CYCLES * t)
    ay = 2.0 * np.pi * TURNS_Y * t
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
    )
    return torch.tensor(ry @ rx, dtype=torch.float32)


def project(points: torch.Tensor):
    """Perspective projection of (..., 3) points to screen pixels and depth."""
    z = points[..., 2]
    scale = CAMERA_DISTANCE / (CAMERA_DISTANCE - z)
    px = WIDTH / 2.0 + points[..., 0] * scale * SCALE * WIDTH
    py = HEIGHT / 2.0 - points[..., 1] * scale * SCALE * WIDTH
    return px, py, z


def generate(settings: ImageProcessingSettings = None):
    """
    A (p, q) torus knot drawn as a cage of strands and rings, rotating in 3D.

    The knot curve is sampled finely, and a smooth orthonormal frame along it
    (tangent, outward-from-core normal, binormal) places STRANDS parallel
    curves on a tube of radius TUBE_RADIUS around it, plus RINGS cross-section
    circles. The whole cage is turned about the vertical axis by an integer
    number of turns, rocked about the horizontal axis by a sinusoid of integer
    frequency, stretched by STRETCH along the frame's long axis and projected
    with perspective, so the geometry at t = 1 equals t = 0 exactly.

    Every strand and ring is splatted bilinearly on the GPU, each sample
    weighted by the screen length it represents. Colour and brightness come
    from depth after rotation, through a violet-to-white palette, which gives
    the over/under reading at crossings. The frame buffer decays between
    frames for a short trail; two-scale bloom and a vignette finish the frame.
    The RNG seed chooses the (p, q) pair, the starting phase, and the strand
    twist offset.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    p, q = KNOTS[int(rng.integers(0, len(KNOTS)))]
    seed_phase = rng.uniform(0.0, 1.0)
    twist = rng.uniform(0.0, 2.0 * np.pi)

    logger.info(
        f"Rendering Torus Knot Turning ({p}, {q}): {FRAME_COUNT} frames @ {FPS} fps"
    )
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    phi = torch.linspace(0.0, 2.0 * np.pi, SAMPLES + 1, device=device)[:-1]
    curve, normal, binormal = torus_knot_frame(phi, p, q)

    # Strands: (STRANDS, SAMPLES, 3), closed along SAMPLES.
    angles = torch.linspace(0.0, 2.0 * np.pi, STRANDS + 1, device=device)[:-1] + twist
    strands = (
        curve[None]
        + TUBE_RADIUS * torch.cos(angles)[:, None, None] * normal[None]
        + TUBE_RADIUS * torch.sin(angles)[:, None, None] * binormal[None]
    )
    strands = torch.cat([strands, strands[:, :1]], dim=1)  # close each strand

    # Rings: (RINGS, RING_SAMPLES + 1, 3).
    ring_index = torch.linspace(0, SAMPLES, RINGS + 1, device=device)[:-1].to(
        torch.long
    )
    ring_angles = torch.linspace(0.0, 2.0 * np.pi, RING_SAMPLES + 1, device=device)
    rings = (
        curve[ring_index][:, None, :]
        + TUBE_RADIUS
        * torch.cos(ring_angles)[None, :, None]
        * normal[ring_index][:, None, :]
        + TUBE_RADIUS
        * torch.sin(ring_angles)[None, :, None]
        * binormal[ring_index][:, None, :]
    )

    background = torch.tensor(BACKGROUND, device=device)
    image = background.expand(HEIGHT, WIDTH, 3).clone()
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)
    z_extent = MAJOR_RADIUS + MINOR_RADIUS + TUBE_RADIUS

    stretch = torch.tensor([1.0, STRETCH, 1.0], device=device)

    def draw(points: torch.Tensor, energy: float, rot: torch.Tensor):
        rotated = (points @ rot.T) * stretch
        px, py, z = project(rotated)
        lengths = sample_lengths(px, py)
        depth = ((z / z_extent) * 0.5 + 0.5).clamp(0.0, 1.0)  # 0 far, 1 near
        colours = lut[(depth * (PALETTE_SIZE - 1)).to(torch.long)].reshape(-1, 3)
        weight = (lengths * energy * (1.0 + DEPTH_GAIN * depth)).reshape(-1)
        splat(flat, px.reshape(-1), py.reshape(-1), colours * weight[:, None])

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            rot = rotation(t).to(device)

            image.mul_(TRAIL_DECAY)
            torch.maximum(image, background, out=image)
            flat.zero_()
            draw(strands, LINE_ENERGY, rot)
            draw(rings, RING_ENERGY, rot)
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
