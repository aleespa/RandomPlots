"""Conjugate Bloom -- chaos-game density of z -> w_k / (a z + b conj(z) + c), c on a loop."""

import gc
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings

# RP_FPS=5 in the environment gives a quick 30-frame test loop with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 6
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 3  # play the rendered loop this many times back-to-back in the mp4
WIDTH, HEIGHT = 1080, 1920  # 9:16 for Reels/Stories
HALF_WIDTH = 2.15  # the frame spans Re(z) in [-HALF_WIDTH, HALF_WIDTH]
HALF_HEIGHT = HALF_WIDTH * HEIGHT / WIDTH

# The map of the reference figure: two branches w_k = exp(2 pi i k / 2), k = 0, 1.
A = 1.02 + 1.93j
B = 0.345 + 0.515j
C = -0.22 + 0.57j

# Density ramp: wine-dark void, teal haze, cyan silk, white edge, then the
# densest cores run back through orange and ember to near-black.
PALETTE = [
    "#000000",
    "#2a0810",
    "#173a45",
    "#2fa0a8",
    "#a9ece9",
    "#fff6ee",
    "#f2a070",
    "#c8451c",
    "#3a0c08",
    "#000000"
]

PNG_WRITERS = 4  # threads encoding PNGs while the GPU renders the next frame


def _device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.warning("CUDA not available -- rendering on the CPU with torch")
    return torch.device("cpu")


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def _blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) tensor with reflected borders."""
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


class _Renderer:
    """The chaos game, histogram and tone mapping, all resident on one device."""

    def __init__(self, knobs: dict, seed: int, device: torch.device):
        self.k = knobs
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        # a z + b conj(z) is real-linear in (x, y); expand it once into a 2x2 block.
        self.m11 = float(A.real + B.real)
        self.m12 = float(-A.imag + B.imag)
        self.m21 = float(A.imag + B.imag)
        self.m22 = float(A.real - B.real)
        lut = LinearSegmentedColormap.from_list("bloom", PALETTE, N=1024)(
            np.linspace(0.0, 1.0, 1024)
        )[:, :3]
        self.lut = torch.tensor(lut, dtype=torch.float32, device=device)
        self.soft_kernel = _gaussian_kernel(knobs["softness"], device)
        self.glow_kernel = _gaussian_kernel(knobs["glow_sigma"], device)
        self.log_ceiling = 1.0

    def constant(self, tau: float) -> complex:
        """The additive constant for loop phase tau: C plus a small circular orbit."""
        return C + self.k["orbit_radius"] * np.exp(
            2j * np.pi * (tau + self.k["orbit_phase"])
        )

    def step(self, x, y, c: complex):
        """One chaos-game step: z -> +-1 / (a z + b conj(z) + c), branch drawn per point."""
        dr = self.m11 * x + self.m12 * y + c.real
        di = self.m21 * x + self.m22 * y + c.imag
        inv = 1.0 / (dr * dr + di * di)
        sign = (
            torch.randint(
                0,
                2,
                x.shape,
                generator=self.gen,
                device=self.device,
                dtype=torch.float32,
            )
            * 2.0
            - 1.0
        )
        return sign * dr * inv, -sign * di * inv

    def burn_in(self, c: complex):
        """A fresh point cloud settled onto the attractor of the map at constant c."""
        n = self.k["n_points"]
        x = torch.rand(n, generator=self.gen, device=self.device) * 2.0 - 1.0
        y = torch.rand(n, generator=self.gen, device=self.device) * 2.0 - 1.0
        for _ in range(self.k["burn_in"]):
            x, y = self.step(x, y, c)
        return torch.nan_to_num(x), torch.nan_to_num(y)

    def frame_density(self, x, y, c: complex):
        """Advance the cloud at constant c, histogramming every visited position."""
        density = torch.zeros(HEIGHT * WIDTH, dtype=torch.float32, device=self.device)
        ones = None
        for _ in range(self.k["iterations_per_frame"]):
            x, y = self.step(x, y, c)
            px = ((x + HALF_WIDTH) * (WIDTH / (2 * HALF_WIDTH))).long()
            py = ((HALF_HEIGHT - y) * (HEIGHT / (2 * HALF_HEIGHT))).long()
            inside = (px >= 0) & (px < WIDTH) & (py >= 0) & (py < HEIGHT)
            idx = (py * WIDTH + px)[inside]
            if ones is None or ones.numel() < idx.numel():
                ones = torch.ones(idx.numel(), device=self.device)
            density.index_put_((idx,), ones[: idx.numel()], accumulate=True)
        return density.view(HEIGHT, WIDTH), torch.nan_to_num(x), torch.nan_to_num(y)

    def log_density(self, density):
        return torch.log1p(_blur(density[None], self.soft_kernel)[0])

    def compose(self, density) -> np.ndarray:
        """Log-density tone mapping through the palette with a soft glow, as BGR uint8."""
        level = self.log_density(density) / self.log_ceiling
        index = (level.clamp(0.0, 1.0) * (self.lut.shape[0] - 1)).long()
        rgb = self.lut[index].permute(2, 0, 1)  # (3, H, W)
        if self.k["glow_weight"] > 0:
            # Only the bright silk glows; the wine-dark background stays flat.
            lift = (level - 0.35).clamp(0.0, 1.0)
            rgb = rgb + self.k["glow_weight"] * _blur(rgb, self.glow_kernel) * lift
        frame = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _write_png(path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


def generate(settings: ImageProcessingSettings = None):
    """
    Conjugate Bloom -- the chaos-game attractor of the two-branch map

        z -> w_k / (a z + b conj(z) + c),   w_k = exp(2 pi i k / 2), k = 0, 1,

    with a = 1.02 + 1.93i, b = 0.345 + 0.515i and c = -0.22 + 0.57i, rendered as
    an exactly looping 9:16 clip.

    The denominator is a real-linear (not holomorphic) function of z because of
    the conj(z) term, so the map folds the plane anisotropically before the
    reciprocal turns it inside out; the two branches differ only by the sign
    w_1 = -w_0 and are picked with equal probability at every step. A cloud of
    points iterated this way settles on the attractor of the iterated function
    system, and the picture is its invariant measure: each frame histograms
    every visited position into the pixel grid, the log of the density is sent
    through a colour ramp (dark void -> teal -> cyan -> white -> orange ->
    ember), and a soft glow is added to the bright silk.

    The loop moves the constant c around a small circle, c(tau) = c_0 + r
    exp(2 pi i tau), which deforms the folds of the attractor continuously and
    returns exactly at tau = 1. The cloud is carried from frame to frame so it
    tracks the slowly moving attractor with a single burn-in.

    Everything numerical runs on the GPU with torch: the map is applied to the
    whole cloud in real arithmetic, positions are scatter-added into the
    histogram, and the blur, palette lookup and glow are convolutions on the
    device. Only the finished 8-bit frame crosses back to the host, where a
    small thread pool encodes the PNGs while the next frame renders.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    knobs = dict(
        n_points=1_200_000,  # chaos-game points in the cloud
        iterations_per_frame=48,  # map steps histogrammed into each frame
        burn_in=80,  # steps before a fresh cloud is trusted to sit on the attractor
        orbit_radius=0.15,  # radius of the circle c travels; larger = wilder morph
        orbit_phase=float(rng.uniform(0.0, 1.0)),  # where on the circle the loop starts
        softness=0.7,  # px, blur of the raw histogram
        glow_weight=0.25,
        glow_sigma=6.0,
    )
    exposure_pct = 99.8  # log-density percentile that reaches the ramp's end

    renderer = _Renderer(knobs, int(rng.integers(0, 2**31)), _device())

    # --- one exposure for the whole clip, from probe frames around the loop ---
    ceilings = []
    for tau in np.arange(3) / 3:
        c = renderer.constant(tau)
        density, _, _ = renderer.frame_density(*renderer.burn_in(c), c)
        ceilings.append(
            torch.quantile(
                renderer.log_density(density).flatten()[::7], exposure_pct / 100
            ).item()
        )
    renderer.log_ceiling = float(max(ceilings))
    logger.info(f"log-density ceiling {renderer.log_ceiling:.3f}")

    frames_path = settings.frames_path
    logger.info(f"rendering {LOOP_FRAMES} frames")
    x, y = renderer.burn_in(renderer.constant(0.0))
    with ThreadPoolExecutor(PNG_WRITERS) as pool:
        pending = []
        for index in range(LOOP_FRAMES):
            density, x, y = renderer.frame_density(
                x, y, renderer.constant(index / LOOP_FRAMES)
            )
            frame = renderer.compose(density)
            pending.append(
                pool.submit(_write_png, frames_path / f"frame{index:04d}.png", frame)
            )
            if len(pending) > 2 * PNG_WRITERS:
                pending.pop(0).result()
        for job in pending:
            job.result()

    for repeat in range(1, REPEATS):
        for i in range(LOOP_FRAMES):
            shutil.copyfile(
                frames_path / f"frame{i:04d}.png",
                frames_path / f"frame{repeat * LOOP_FRAMES + i:04d}.png",
            )

    del renderer, x, y
    gc.collect()
    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
