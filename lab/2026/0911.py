"""
Harmonic Nebula

A breathing, rotating spiral manifold born from non-linear harmonic modulation:

    r(θ) = θ / (2 + sin(u θ + φ))

When a spiral's radius is folded by an oscillating denominator, simple polar
growth breaks into intricate resonant folds. As the modulation frequency u and
phase shift φ traverse a slow, periodic orbit, the petals twist, bloom, and
unfurl into delicate moiré caustics before smoothly returning to their origin.

Luminous filaments are colored along their winding arc length, surrounded by
a celestial haze and screen-blended optical bloom.

Made with Python, PyTorch and OpenCV. 60 fps, exact loop.

#generativeart #creativecoding #mathart #spirals #moire #lissajous #looping
#pythonart #generative #glow #bloom #digitalart
"""

import gc
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings

# RP_FPS in the environment allows quick test runs (e.g. RP_FPS=5)
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 8  # 6-second exact loop period
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 3  # play the rendered loop 3 times back-to-back in the final mp4
WIDTH, HEIGHT = 1080, 1920  # 9:16 for Reels/Stories
HALF_WIDTH = 165.0  # Expands the view box so the entire figure fits with comfortable breathing margin
HALF_HEIGHT = HALF_WIDTH * HEIGHT / WIDTH

# Saturated, vivid cosmic palette tailored for vibrant mobile screens:
# Deep electric teal -> luminous cyan -> pure electric violet -> magenta pink -> radiant neon amber -> warm gold
VIVID_NEBULA = [
    "#00f0ff",  # electric cyan
    "#00ffaa",  # radiant mint / emerald
    "#7000ff",  # electric violet
    "#ff007f",  # neon magenta
    "#ff4400",  # blazing orange
    "#ffcc00",  # radiant gold
    "#00f0ff",  # loop back seamlessly
]

PNG_WRITERS = 4


def _device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.warning("CUDA not available -- rendering on the CPU with torch")
    return torch.device("cpu")


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def _blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) tensor with reflection padding."""
    radius = (kernel.numel() - 1) // 2
    channels = image.shape[0]
    x = image[None]
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(
        x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels
    )
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(
        x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels
    )
    return x[0]


class _NebulaRenderer:
    """Evaluates modulated spiral ribbons and renders screen-blended glowing frames."""

    def __init__(self, knobs: dict, device: torch.device):
        self.k = knobs
        self.device = device

        lut = LinearSegmentedColormap.from_list("vivid", VIVID_NEBULA, N=2048)(
            np.linspace(0.0, 1.0, 2048)
        )[:, :3]
        self.lut = torch.tensor(lut, dtype=torch.float32, device=device)

        self.glow_kernel_tight = _gaussian_kernel(knobs["glow_sigma_tight"], device)
        self.glow_kernel_wide = _gaussian_kernel(knobs["glow_sigma_wide"], device)

        # Precompute coordinate grids and radial atmospheric background mask
        y_coords = torch.linspace(
            -HALF_HEIGHT, HALF_HEIGHT, HEIGHT, device=device, dtype=torch.float32
        )
        x_coords = torch.linspace(
            -HALF_WIDTH, HALF_WIDTH, WIDTH, device=device, dtype=torch.float32
        )
        gy, gx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        r_field = torch.hypot(gx, gy)
        atmosphere = torch.exp(-((r_field / (HALF_WIDTH * 0.95)) ** 2))
        self.background = (
            atmosphere[None, ...].expand(3, HEIGHT, WIDTH)
            * torch.tensor([0.005, 0.009, 0.020], device=device)[:, None, None]
        )

        # High-density discretization of the spiral ribbon
        n_points = knobs["n_points"]
        theta_max = knobs["theta_max"]
        self.theta = torch.linspace(
            0.0, theta_max, n_points, device=device, dtype=torch.float32
        )
        self.cos_theta = torch.cos(self.theta)
        self.sin_theta = torch.sin(self.theta)

        # Ribbon phase offsets across multiple strand arms
        n_strands = knobs["n_strands"]
        self.strand_phases = torch.arange(
            n_strands, device=device, dtype=torch.float32
        ) * (2 * np.pi / n_strands)
        self.strand_weights = torch.linspace(0.94, 1.06, n_strands, device=device)

        # Arc length color index along the spiral
        color_idx = (
            (self.theta / theta_max * knobs["color_cycles"])
            % 1.0
            * (self.lut.shape[0] - 1)
        ).long()
        self.strand_colors = self.lut[color_idx]  # (n_points, 3)

        # Soft radial ramp to gently balance the core brightness against the outer filaments
        # Points near θ=0 naturally superimpose heavily; this prevents the core blowing out.
        theta_norm = self.theta / theta_max
        core_balance = 0.45 + 0.55 * torch.clamp(theta_norm * 4.0, 0.0, 1.0)
        self.strand_colors = self.strand_colors * core_balance[:, None]

        # 5x5 smooth Gaussian footprint for thick, prominent lines on mobile screens
        radius = knobs.get("splat_radius", 2)
        sigma = knobs.get("splat_sigma", 1.2)
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = np.hypot(dx, dy)
                weight = np.exp(-0.5 * (dist / sigma) ** 2)
                offsets.append((dx, dy, float(weight)))
        total_w = sum(w for _, _, w in offsets)
        self.splat_offsets = [(dx, dy, w / total_w) for dx, dy, w in offsets]
        self.margin = radius + 1

    def render_frame(self, tau: float) -> np.ndarray:
        """Render a single frame at normalized loop phase tau in [0, 1)."""
        k = self.k

        # Slow, gentle harmonic frequency and phase modulation (seamlessly periodic)
        u_t = k["u_center"] + k["u_amp"] * np.cos(2.0 * np.pi * k["u_cycles"] * tau)
        phi_t = k["phi_amp"] * np.sin(2.0 * np.pi * k["phi_cycles"] * tau)
        rot_angle = k["rot_amp"] * np.sin(2.0 * np.pi * tau)
        cos_rot = float(np.cos(rot_angle))
        sin_rot = float(np.sin(rot_angle))

        # Accumulator image tensor for splatted particles/ribbons: (3, HEIGHT, WIDTH)
        canvas = torch.zeros(
            (3, HEIGHT, WIDTH), dtype=torch.float32, device=self.device
        )
        flat_canvas = canvas.view(3, -1)

        # Accumulate across multiple strands
        for s_idx, strand_phi in enumerate(self.strand_phases):
            eff_u = u_t * self.strand_weights[s_idx]
            arg_x = eff_u * self.theta + phi_t + strand_phi
            arg_y = eff_u * self.theta + phi_t + strand_phi + np.pi / 4

            denom_x = torch.sin(arg_x) + 2.0
            denom_y = torch.cos(arg_y) + 2.0

            x = (self.cos_theta / denom_x) * self.theta
            y = (self.sin_theta / denom_y) * self.theta

            # Breathing scale
            breathe = 1.0 + k["breathe_amp"] * np.sin(
                2.0 * np.pi * k["breathe_cycles"] * tau
            )
            x = x * breathe
            y = y * breathe

            # Smooth rigid rotation
            x_rot = cos_rot * x - sin_rot * y
            y_rot = sin_rot * x + cos_rot * y

            # Map from world space to pixel coordinates
            px = (
                ((x_rot + HALF_WIDTH) / (2.0 * HALF_WIDTH) * (WIDTH - 1)).round().long()
            )
            py = (
                ((y_rot + HALF_HEIGHT) / (2.0 * HALF_HEIGHT) * (HEIGHT - 1))
                .round()
                .long()
            )

            m = self.margin
            valid = (px >= m) & (px < WIDTH - m) & (py >= m) & (py < HEIGHT - m)
            px_val = px[valid]
            py_val = py[valid]
            colors_val = self.strand_colors[valid]

            # Accumulate across thick footprint for visible, bold filaments
            for dx, dy, w in self.splat_offsets:
                idx = (py_val + dy) * WIDTH + (px_val + dx)
                for c in range(3):
                    flat_canvas[c].scatter_add_(0, idx, colors_val[:, c] * w)

        # Normalization and gentle tone compression
        density = canvas / float(len(self.strand_phases))
        level = torch.tanh(k["gain"] * density)

        # Composite background atmosphere
        rgb = torch.maximum(level, self.background)

        # Screen-blended dual-pass Gaussian optical bloom with controlled core
        if k["glow_weight"] > 0:
            bright = (level - k["glow_floor"]).clamp(min=0.0)
            tight_glow = _blur(bright, self.glow_kernel_tight)
            wide_glow = _blur(bright, self.glow_kernel_wide)
            bloom = (
                k["glow_weight_tight"] * tight_glow + k["glow_weight_wide"] * wide_glow
            )
            rgb = 1.0 - (1.0 - rgb) * torch.exp(-bloom)

        # Convert to BGR uint8 for fast OpenCV PNG writing
        frame = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _write_png(path: Path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


def generate(settings: ImageProcessingSettings = None):
    """
    Harmonic Nebula -- modulated spiral manifold with non-linear harmonic folding.

    Extends the non-linear denominator modulation of 0308:
        x(θ) = [cos(θ) / (sin(u θ + φ) + 2)] * θ
        y(θ) = [sin(θ) / (cos(u θ + φ + π/4) + 2)] * θ

    When u and φ oscillate across an exact integer loop period, the spiral folds
    into high-order multi-petal resonance caustics and breathing moiré waves.
    Rendered with PyTorch on the GPU using multi-strand accumulation, arc-length
    palette indexing, thick 5x5 anti-aliased footprint splatting, and dual-scale bloom.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    knobs = dict(
        u_center=4.0,  # center frequency
        u_amp=0.12,  # very gentle frequency modulation: u swings subtly between 3.82 and 4.18
        u_cycles=1,  # 1 smooth periodic wave across the 6s
        phi_cycles=1,  # 1 smooth phase shift
        phi_amp=0.30,  # smooth subtle phase excursion
        rot_cycles=0,  # no disorienting full spin
        rot_amp=0.08,  # gentle subtle angular oscillation (+/- ~4.5 degrees)
        breathe_amp=0.015,  # barely perceptible organic breathing pulsation
        breathe_cycles=1,
        theta_max=160,  # spiral winding extent
        n_points=240000,  # dense, smooth filament trace
        n_strands=2,  # bundle of interwoven ribbon strands
        color_cycles=2.0,  # palette repeats along spiral length
        splat_radius=2,  # 5x5 footprint for thicker, bolder lines on mobile
        splat_sigma=1.2,  # smooth Gaussian line weight
        gain=2.2,  # rich filament saturation
        glow_sigma_tight=4.0,
        glow_sigma_wide=14.0,
        glow_floor=0.15,  # higher floor prevents the center from washing out
        glow_weight=0.9,  # moderated glow to preserve delicate structure in the center
        glow_weight_tight=0.55,
        glow_weight_wide=0.35,
    )

    device = _device()
    renderer = _NebulaRenderer(knobs, device)
    frames_path = settings.frames_path

    logger.info(
        f"rendering {LOOP_FRAMES} frames for Harmonic Nebula at {FPS} fps ({LOOP_SECONDS}s loop)..."
    )

    with ThreadPoolExecutor(PNG_WRITERS) as pool:
        pending = []
        for i in range(LOOP_FRAMES):
            tau = i / LOOP_FRAMES
            frame = renderer.render_frame(tau)
            out_file = frames_path / f"frame{i:04d}.png"
            pending.append(pool.submit(_write_png, out_file, frame))

            if len(pending) >= 2 * PNG_WRITERS:
                pending.pop(0).result()

            if (i + 1) % (LOOP_FRAMES // 8 or 1) == 0 or i == LOOP_FRAMES - 1:
                logger.info(f"rendered {i + 1}/{LOOP_FRAMES} frames (tau={tau:.3f})")

        for job in pending:
            job.result()

    # Replicate seamless loop back-to-back for Instagram Reels/Stories
    logger.info(f"replicating loop for {REPEATS} iterations...")
    for repeat in range(1, REPEATS):
        for i in range(LOOP_FRAMES):
            src = frames_path / f"frame{i:04d}.png"
            dst = frames_path / f"frame{repeat * LOOP_FRAMES + i:04d}.png"
            shutil.copyfile(src, dst)

    del renderer
    gc.collect()

    logger.info("encoding video with ffmpeg...")
    settings.save_video(FPS, crf=18)
    logger.info("Harmonic Nebula animation complete.")


if __name__ == "__main__":
    generate()
