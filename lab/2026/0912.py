"""
Non-Hermitian Spectral Bloom

A modern re-imagination of non-Hermitian random matrix eigenvalue dynamics:

    M₁(t) = [[t₀, i], [ c, t₁]]
    M₂(t) = [[t₀, i], [-c, t₁]]

When complex parameter pairs (t₀, t₁) are sampled across a 2-torus 𝕋² with
a rhythmic, musical breathing envelope r(τ), the non-Hermitian eigenvalues
Z₁ = eig(M₁) and Z₂ = eig(M₂) trace out geometric envelope curves and caustic
cusps in the complex plane.

Cross-projecting the real and imaginary components (Re(Z₁), Im(Z₂)) and
(Re(Z₂), Im(Z₁)) weaves an intricate, glowing mandala of spectral caustics
that pulses, rebounds, bifurcates, and loops seamlessly to an 8-beat cadence.

Made with Python, PyTorch and OpenCV. 60 fps, exact loop.

#generativeart #creativecoding #mathart #spectraltheory #nonhermitian
#eigenvalues #looping #pythonart #glow #bloom #digitalart
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

from common.image_processing import ImageProcessingSettings

# RP_FPS in the environment allows rapid test runs (e.g. RP_FPS=5)
FPS = int(os.environ.get("RP_FPS", "60"))
BPM = 96
BEATS_PER_LOOP = 12
LOOP_SECONDS = BEATS_PER_LOOP * 60 / BPM  # 7.5 seconds
LOOP_FRAMES = round(FPS * LOOP_SECONDS) # exact loop period, in frames
REPEATS = 3  # play the rendered loop 3 times back-to-back in the final mp4
WIDTH, HEIGHT = 1080, 1920  # 9:16 for Reels/Stories
HALF_WIDTH = 4.2  # domain span in complex plane
HALF_HEIGHT = HALF_WIDTH * HEIGHT / WIDTH

PNG_WRITERS = 8


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


def _musical_rhythm(tau: float) -> float:
    """
    Dynamic but smooth 96 BPM envelope.

    7.5 seconds = exactly 12 beats at 96 BPM.
    Large-scale phrase movement explores the full figure,
    while individual beats provide softer rhythmic modulation.
    """
    BPM = 96.0
    t = tau * LOOP_SECONDS
    beat = BPM / 60.0 * t

    phase = beat % 1.0

    # Soft beat pulse
    beat_pulse = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
    beat_pulse = beat_pulse ** 0.7

    # 4-beat bar
    bar_phase = (beat % 4.0) / 4.0
    bar_env = 0.5 - 0.5 * np.cos(2.0 * np.pi * bar_phase)

    # Full 12-beat phrase
    phrase = 0.5 - 0.5 * np.cos(2.0 * np.pi * tau)

    # Stronger global expansion/contraction
    baseline = 0.08 + 0.72 * phrase

    # Gentle beat modulation
    rhythm = (
        baseline
        + 0.14 * beat_pulse
        + 0.06 * bar_env
    )

    return float(np.clip(rhythm, 0.0, 1.0))


class _SpectralRenderer:
    """
    Evaluates exact closed-form eigenvalues for 2x2 non-Hermitian matrix pencils,
    cross-projects (Re(Z1), Im(Z2)) and (Re(Z2), Im(Z1)), and renders screen-blended
    luminous bloom frames.
    """

    def __init__(self, knobs: dict, rng: np.random.Generator, device: torch.device):
        self.k = knobs
        self.device = device

        self.glow_kernel_tight = _gaussian_kernel(knobs["glow_sigma_tight"], device)
        self.glow_kernel_wide = _gaussian_kernel(knobs["glow_sigma_wide"], device)

        # Precompute coordinate grids and radial atmospheric background mask
        atm_y = torch.linspace(
            -HALF_HEIGHT, HALF_HEIGHT, HEIGHT, device=device, dtype=torch.float32
        )
        atm_x = torch.linspace(
            -HALF_WIDTH, HALF_WIDTH, WIDTH, device=device, dtype=torch.float32
        )
        gy, gx = torch.meshgrid(atm_y, atm_x, indexing="ij")
        r_field = torch.hypot(gx, gy)
        atmosphere = torch.exp(-((r_field / (HALF_WIDTH * 0.95)) ** 2))
        self.background = (
            atmosphere[None, ...].expand(3, HEIGHT, WIDTH)
            * torch.tensor([0.006, 0.004, 0.015], device=device)[:, None, None]
        )

        # Sample random angle pairs on the 2-torus T^2 (reproducible via settings.rng)
        n_points = knobs["n_points"]
        thetas = rng.uniform(0.0, 2.0 * np.pi, size=(n_points, 2)).astype(np.float32)
        self.thetas = torch.tensor(thetas, device=device)

        # Color channels for the two cross-projected branches (RGB float):
        # Branch 1: Luminous Electric Cyan
        self.col_branch1 = torch.tensor(
            [0.05, 0.88, 1.00], device=device, dtype=torch.float32
        )
        # Branch 2: Neon Magenta / Hot Pink
        self.col_branch2 = torch.tensor(
            [1.00, 0.12, 0.68], device=device, dtype=torch.float32
        )
        # Direct eigenvalues subtle backdrop (Gold/Amber)
        self.col_direct = torch.tensor(
            [0.95, 0.70, 0.15], device=device, dtype=torch.float32
        )

        # 3x3 smooth anti-aliased splat footprint
        self.splat_offsets = [
            (-1, 0, 0.12),
            (1, 0, 0.12),
            (0, -1, 0.12),
            (0, 1, 0.12),
            (0, 0, 0.52),
        ]

    def render_frame(self, tau: float) -> np.ndarray:
        """Render a single frame at normalized loop phase tau in [0, 1)."""
        k = self.k
        dev = self.device

        # Musical rhythmic breathing radius and coupling modulation
        r_norm = _musical_rhythm(tau)
        r = k["r_min"] + (k["r_max"] - k["r_min"]) * r_norm
        c = k["c_base"] + k["c_amp"] * np.sin(2.0 * np.pi * tau) * (0.6 + 0.4 * r_norm)
        rot_angle = k["rot_amp"] * np.sin(2.0 * np.pi * tau)

        # Phase modulation along torus
        th0 = self.thetas[:, 0] + 2.0 * np.pi * k["th0_cycles"] * tau
        th1 = self.thetas[:, 1] + 2.0 * np.pi * k["th1_cycles"] * tau

        t0 = r * torch.complex(torch.cos(th0), torch.sin(th0))
        t1 = r * torch.complex(torch.cos(th1), torch.sin(th1))

        # Analytical eigenvalues for 2x2 matrix pencils:
        # M1 = [[t0, i], [ c, t1]] -> disc1 = (t0 - t1)^2 + 4i*c
        # M2 = [[t0, i], [-c, t1]] -> disc2 = (t0 - t1)^2 - 4i*c
        diff_sq = (t0 - t1) ** 2
        disc1 = torch.sqrt(diff_sq + 4j * c)
        disc2 = torch.sqrt(diff_sq - 4j * c)
        sum_t = t0 + t1

        z1_a = (sum_t + disc1) * 0.5
        z1_b = (sum_t - disc1) * 0.5
        z2_a = (sum_t + disc2) * 0.5
        z2_b = (sum_t - disc2) * 0.5

        # Cross-projections:
        # Branch 1: (Re(Z1), Im(Z2))
        x1_a, y1_a = z1_a.real, z2_a.imag
        x1_b, y1_b = z1_b.real, z2_b.imag

        # Branch 2: (Re(Z2), Im(Z1))
        x2_a, y2_a = z2_a.real, z1_a.imag
        x2_b, y2_b = z2_b.real, z1_b.imag

        # Direct eigenvalues: (Re(Z1), Im(Z1)) and (Re(Z2), Im(Z2))
        xd_1, yd_1 = z1_a.real, z1_a.imag
        xd_2, yd_2 = z2_a.real, z2_a.imag

        # Smooth rigid rotation
        cos_rot = float(np.cos(rot_angle))
        sin_rot = float(np.sin(rot_angle))

        def rotate(x, y):
            return cos_rot * x - sin_rot * y, sin_rot * x + cos_rot * y

        x1_a, y1_a = rotate(x1_a, y1_a)
        x1_b, y1_b = rotate(x1_b, y1_b)
        x2_a, y2_a = rotate(x2_a, y2_a)
        x2_b, y2_b = rotate(x2_b, y2_b)
        xd_1, yd_1 = rotate(xd_1, yd_1)
        xd_2, yd_2 = rotate(xd_2, yd_2)

        # Splatting onto accumulator canvas
        canvas = torch.zeros((3, HEIGHT, WIDTH), dtype=torch.float32, device=dev)
        flat_canvas = canvas.view(3, -1)

        def splat_points(px, py, col, weight):
            sx = (((px + HALF_WIDTH) / (2.0 * HALF_WIDTH)) * (WIDTH - 1)).round().long()
            sy = (
                (((py + HALF_HEIGHT) / (2.0 * HALF_HEIGHT)) * (HEIGHT - 1))
                .round()
                .long()
            )
            valid = (sx >= 2) & (sx < WIDTH - 2) & (sy >= 2) & (sy < HEIGHT - 2)
            sx_v = sx[valid]
            sy_v = sy[valid]
            val = (col * weight).unsqueeze(1).expand(-1, sx_v.shape[0])
            for dx, dy, w in self.splat_offsets:
                idx = (sy_v + dy) * WIDTH + (sx_v + dx)
                flat_canvas.scatter_add_(1, idx.unsqueeze(0).expand(3, -1), val * w)

        # Splat direct eigenvalues (subtle halo)
        if k.get("direct_weight", 0.0) > 0:
            splat_points(xd_1, yd_1, self.col_direct, k["direct_weight"])
            splat_points(xd_2, yd_2, self.col_direct, k["direct_weight"])

        # Splat cross-projected branches (primary luminous structures)
        w_branch = k["branch_weight"]
        splat_points(x1_a, y1_a, self.col_branch1, w_branch)
        splat_points(x1_b, y1_b, self.col_branch1, w_branch)
        splat_points(x2_a, y2_a, self.col_branch2, w_branch)
        splat_points(x2_b, y2_b, self.col_branch2, w_branch)

        # Tone mapping and atmospheric composite
        level = torch.tanh(k["gain"] * canvas)
        rgb = torch.maximum(level, self.background)

        # Dual-kernel separable Gaussian optical bloom
        if k["glow_weight"] > 0:
            bright = (level - k["glow_floor"]).clamp(min=0.0)
            tight_glow = _blur(bright, self.glow_kernel_tight)
            wide_glow = _blur(bright, self.glow_kernel_wide)
            bloom = (
                k["glow_weight_tight"] * tight_glow + k["glow_weight_wide"] * wide_glow
            )
            rgb = 1.0 - (1.0 - rgb) * torch.exp(-bloom)

        # Convert to BGR uint8 for OpenCV video writer
        frame = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _write_png(path: Path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


def generate(settings: ImageProcessingSettings = None):
    """
    Non-Hermitian Spectral Bloom -- modernized eigenvalue dynamics of lab.2023.1013.

    Samples complex parameter pairs (t0, t1) on a 2-torus T^2 with breathing radius r(tau).
    For each pair, exact analytical eigenvalues are computed for the coupled non-Hermitian
    matrix families:
        M1 = [[t0, i], [ c, t1]] -> Z1 = (t0 + t1 +/- sqrt((t0 - t1)^2 + 4i*c)) / 2
        M2 = [[t0, i], [-c, t1]] -> Z2 = (t0 + t1 +/- sqrt((t0 - t1)^2 - 4i*c)) / 2

    Cross-projecting (Re(Z1), Im(Z2)) and (Re(Z2), Im(Z1)) produces sharp envelope
    caustics and breathing geometric lobes with exceptional point bifurcations.

    Rendered with PyTorch on the GPU using vectorized point splatting, subpixel anti-aliasing,
    and dual-scale screen-blended optical bloom.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    knobs = dict(
        # Torus & parameter dynamics
        r_min=0.45,  # inner breathing radius
        r_max=3.35,  # outer breathing radius
        c_base=0.60,  # non-Hermitian off-diagonal coupling (from 1013)
        c_amp=0.22,  # subtle harmonic coupling modulation
        th0_cycles=0,  # fixed toroidal orientation
        th1_cycles=0,
        rot_amp=1,  # subtle angular sway (+/- ~6.8 degrees)
        # Point ensemble & resolution
        n_points=1_000_000,  # dense, smooth spectral cloud
        branch_weight=0.75,  # brightness per point for cross branches
        direct_weight=0.15,  # subtle ambient direct spectrum
        gain=0.045,  # tone compression gain
        # Optical bloom
        glow_sigma_tight=4.0,
        glow_sigma_wide=16.0,
        glow_floor=0.05,
        glow_weight=1.0,
        glow_weight_tight=0.55,
        glow_weight_wide=0.35,
    )

    device = _device()
    renderer = _SpectralRenderer(knobs, rng, device)
    frames_path = settings.frames_path

    logger.info(
        f"rendering {LOOP_FRAMES} frames for Non-Hermitian Spectral Bloom at {FPS} fps ({LOOP_SECONDS}s loop)..."
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
    logger.info("Non-Hermitian Spectral Bloom animation complete.")


if __name__ == "__main__":
    generate()
