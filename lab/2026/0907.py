"""Caustic Tide -- sunlight refracted through a looping wave spectrum onto the pool floor."""

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

from colors.palettes import GLACIER
from common.image_processing import ImageProcessingSettings

# RP_FPS=4 in the environment gives a quick 32-frame test loop with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 3  # play the rendered loop this many times back-to-back in the mp4
WIDTH, HEIGHT = 1080, 1920  # 9:16 for Reels/Stories
ASPECT = WIDTH / HEIGHT  # the floor spans x in [0, ASPECT], y in [0, 1]

WATER_IOR = 1.333  # refractive index of water at ~589 nm
PALETTE = GLACIER  # ice-blue haze, gold held back for the densest caustic cores

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


def _wave_spectrum(rng, knobs: dict):
    """
    A deep-water gravity-wave spectrum whose every component is LOOP_SECONDS-periodic.

    Each component gets an integer harmonic m of the loop frequency,
    omega_m = 2 pi m / LOOP_SECONDS, so the surface repeats exactly. The
    dispersion relation for deep water, omega^2 = g k, then fixes the
    wavenumber: k_m = m^2 k_1. Choosing the fundamental wavelength therefore
    chooses the effective gravity, and the harmonics are automatically the
    physically correct, ever shorter and faster waves rather than an arbitrary
    set of frequencies.

    Returns (kx, ky, omega, amp, phase) as float32 arrays, with the amplitudes
    scaled so the RMS surface slope is exactly knobs["slope_rms"].
    """
    k_fundamental = 2.0 * np.pi / knobs["base_wavelength"]
    omega_fundamental = 2.0 * np.pi / LOOP_SECONDS
    gravity = omega_fundamental**2 / k_fundamental
    mean_direction = rng.uniform(0.0, 2.0 * np.pi)
    spread = np.deg2rad(knobs["direction_spread_deg"])

    kx, ky, omega, amp, phase = [], [], [], [], []
    for m in knobs["harmonics"]:
        w = 2.0 * np.pi * m / LOOP_SECONDS
        k = w * w / gravity  # == m^2 * k_fundamental
        for _ in range(knobs["waves_per_harmonic"]):
            theta = mean_direction + rng.uniform(-spread, spread)
            kx.append(k * np.cos(theta))
            ky.append(k * np.sin(theta))
            omega.append(w)
            amp.append(float(m) ** -knobs["amplitude_falloff"])
            phase.append(rng.uniform(0.0, 2.0 * np.pi))

    kx = np.array(kx, dtype=np.float64)
    ky = np.array(ky, dtype=np.float64)
    amp = np.array(amp, dtype=np.float64)
    # h = sum a cos(psi) has E[|grad h|^2] = sum a^2 k^2 / 2; scale to the target.
    rms = np.sqrt(0.5 * np.sum(amp**2 * (kx**2 + ky**2)))
    amp *= knobs["slope_rms"] / rms
    wavelengths = 2 * np.pi / np.hypot(kx, ky)
    logger.info(
        f"{len(amp)} waves, lambda {wavelengths.min():.3f}-{wavelengths.max():.3f}, "
        f"peak-to-trough {2 * amp.sum():.4f}, g={gravity:.3f}"
    )
    return (
        kx.astype(np.float32),
        ky.astype(np.float32),
        np.array(omega, dtype=np.float32),
        amp.astype(np.float32),
        np.array(phase, dtype=np.float32),
    )


class _Renderer:
    """Photon refraction, floor histogram and tone mapping, all on one device."""

    def __init__(self, knobs: dict, spectrum, seed: int, device: torch.device):
        self.k = knobs
        self.device = device
        self.kx, self.ky, self.omega, self.amp, self.phase = (
            torch.tensor(a, device=device) for a in spectrum
        )
        # The launch grid covers a margin beyond the frame, because refraction
        # moves a photon sideways: light from outside the frame lands inside it.
        self.margin = knobs["launch_margin"]
        oversample = float(np.sqrt(knobs["samples_per_pixel"]))
        self.grid_w = int(round(WIDTH * oversample * (1 + 2 * self.margin / ASPECT)))
        self.grid_h = int(round(HEIGHT * oversample * (1 + 2 * self.margin)))
        self.rows_per_chunk = max(1, knobs["chunk_points"] // self.grid_w)
        self.jitter_seed = seed
        lut = LinearSegmentedColormap.from_list("caustic", PALETTE, N=1024)(
            np.linspace(0.0, 1.0, 1024)
        )[:, :3]
        self.lut = torch.tensor(lut, dtype=torch.float32, device=device)
        self.soft_kernel = _gaussian_kernel(knobs["softness"], device)
        self.glow_kernel = _gaussian_kernel(knobs["glow_sigma"], device)
        self.black_level = 0.0
        self.white_level = 1.0
        chunks = int(np.ceil(self.grid_h / self.rows_per_chunk))
        logger.info(
            f"{self.grid_w} x {self.grid_h} = "
            f"{self.grid_w * self.grid_h / 1e6:.1f}M photons per frame, "
            f"{chunks} chunks"
        )

    def launch_positions(self, row0: int, row1: int):
        """The (fixed for the whole clip) jittered launch points of a chunk of rows."""
        # Seeding per chunk rather than per frame keeps the sampling identical
        # from frame to frame, so the residual noise sits still instead of boiling.
        gen = torch.Generator(device=self.device).manual_seed(self.jitter_seed + row0)
        rows = row1 - row0
        jitter = torch.rand(2, rows, self.grid_w, generator=gen, device=self.device)
        cols = torch.arange(self.grid_w, device=self.device, dtype=torch.float32)
        idx_r = torch.arange(row0, row1, device=self.device, dtype=torch.float32)
        u = (cols[None, :] + jitter[0]).reshape(-1)
        v = (idx_r[:, None] + jitter[1]).reshape(-1)
        x = u * ((ASPECT + 2 * self.margin) / self.grid_w) - self.margin
        y = v * ((1.0 + 2 * self.margin) / self.grid_h) - self.margin
        return x, y

    def surface(self, x, y, t: float):
        """Height and slope of the wave field at time t: h, dh/dx, dh/dy."""
        h = torch.zeros_like(x)
        hx = torch.zeros_like(x)
        hy = torch.zeros_like(x)
        for j in range(self.amp.numel()):
            psi = self.kx[j] * x + self.ky[j] * y - self.omega[j] * t + self.phase[j]
            h += self.amp[j] * torch.cos(psi)
            slope = -self.amp[j] * torch.sin(psi)
            hx += slope * self.kx[j]
            hy += slope * self.ky[j]
        return h, hx, hy

    def refract(self, x, y, h, hx, hy):
        """
        Snell refraction of a vertical ray, and where it lands on the floor.

        The incident direction is d = (0, 0, -1) and the unit surface normal is
        n = (-hx, -hy, 1) / sqrt(1 + |grad h|^2), so cos(incidence) = n_z. The
        vector form of Snell's law with eta = 1 / n_water gives the transmitted
        direction

            t = eta d + (eta cos_i - sqrt(1 - eta^2 (1 - cos_i^2))) n,

        which is then walked from (x, y, h) down to the floor at z = -depth.
        """
        inv_norm = torch.rsqrt(1.0 + hx * hx + hy * hy)
        nx, ny, nz = -hx * inv_norm, -hy * inv_norm, inv_norm
        eta = 1.0 / WATER_IOR
        cos_i = nz  # = -n . d
        radicand = (1.0 - eta * eta * (1.0 - cos_i * cos_i)).clamp_min(1e-9)
        scale = eta * cos_i - torch.sqrt(radicand)
        tx, ty = scale * nx, scale * ny
        tz = -eta + scale * nz
        # Light keeps going down, so tz < 0 everywhere and this stays finite.
        march = (self.k["depth"] + h) / (-tz)
        return x + march * tx, y + march * ty

    def frame_density(self, t: float) -> torch.Tensor:
        """Histogram every photon's landing point on the floor, a chunk at a time."""
        density = torch.zeros(HEIGHT * WIDTH, dtype=torch.float32, device=self.device)
        ones = torch.ones(1, device=self.device)
        for row0 in range(0, self.grid_h, self.rows_per_chunk):
            row1 = min(row0 + self.rows_per_chunk, self.grid_h)
            x, y = self.launch_positions(row0, row1)
            fx, fy = self.refract(x, y, *self.surface(x, y, t))
            px = (fx * (WIDTH / ASPECT)).long()
            py = ((1.0 - fy) * HEIGHT).long()
            inside = (px >= 0) & (px < WIDTH) & (py >= 0) & (py < HEIGHT)
            idx = (py * WIDTH + px)[inside]
            density.index_put_((idx,), ones.expand(idx.numel()), accumulate=True)
        return density.view(HEIGHT, WIDTH)

    def log_density(self, density):
        return torch.log1p(_blur(density[None], self.soft_kernel)[0])

    def levels(self, density):
        """Log density mapped to [0, 1] between the clip's black and white points."""
        span = max(self.white_level - self.black_level, 1e-6)
        return ((self.log_density(density) - self.black_level) / span).clamp(0.0, 1.0)

    def compose(self, density) -> np.ndarray:
        """Tone map through the palette with a soft glow, as BGR uint8 for cv2."""
        level = self.levels(density)
        index = (level * (self.lut.shape[0] - 1)).long()
        rgb = self.lut[index].permute(2, 0, 1)  # (3, H, W)
        if self.k["glow_weight"] > 0:
            # Only the focused filaments bloom; the defocused floor stays flat.
            lift = (level - 0.45).clamp(0.0, 1.0)
            rgb = rgb + self.k["glow_weight"] * _blur(rgb, self.glow_kernel) * lift
        frame = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _write_png(path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


def generate(settings: ImageProcessingSettings = None):
    """
    Caustic Tide -- the net of light that refracted sunlight draws on the floor
    of a pool, as an exactly looping 9:16 clip.

    The water surface is a sum of deep-water gravity waves,

        h(x, y, t) = sum_j a_j cos(k_j . x - omega_j t + phi_j),

    in which every frequency is an integer harmonic of the loop frequency,
    omega_j = 2 pi m_j / LOOP_SECONDS. That alone makes the surface exactly
    periodic; the dispersion relation omega^2 = g k then fixes each wavenumber
    at k_j = m_j^2 k_1, so the harmonics are the physically correct shorter,
    faster waves rather than an arbitrary set of ripples, and the spectrum as a
    whole is scaled to a chosen RMS surface slope.

    Light arrives vertically. At each point of the surface the ray is refracted
    by Snell's law in vector form and walked down to the floor at z = -depth.
    That map from surface to floor is smooth but not injective: where the
    surface is convex it focuses, and the caustic is exactly the fold locus
    where the Jacobian determinant of the map vanishes and the intensity
    1 / |det J| diverges. Rather than solve for the folds, the picture is the
    Monte Carlo estimate of that intensity -- a fixed jittered grid of photons
    is refracted and histogrammed into the pixel grid, so the density *is* the
    caustic. Keeping the launch grid fixed across frames (each chunk reseeded
    from the same value every time) makes the residual sampling noise sit still
    instead of boiling.

    Tone mapping sets a black point as well as a white point, both from
    percentiles of the log density probed around the loop: the floor between
    the filaments is lit but defocused, and pushing that level to black is what
    leaves the caustic net alone on a dark ground. The ramp then runs ice blue
    through white to gold at the brightest crossings, and only the focused
    filaments are allowed to bloom.

    Everything numerical runs on the GPU with torch, in chunks of photons, so
    the sample count is limited by time rather than memory. Only the finished
    8-bit frame crosses back to the host, where a small thread pool encodes the
    PNGs while the next frame renders.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    knobs = dict(
        # the wave spectrum
        harmonics=(1, 2, 3, 5),  # integer multiples of the loop frequency
        waves_per_harmonic=6,  # directions drawn per harmonic
        base_wavelength=1.10,  # wavelength of harmonic 1, in frame heights
        amplitude_falloff=2.4,  # a_m ~ m^-falloff; lower = more fine chop
        direction_spread_deg=55.0,  # angular spread about a random mean heading
        slope_rms=0.20,  # RMS surface slope -- the single strongest knob
        # the optics
        depth=0.42,  # floor depth below the mean surface, in frame heights
        # sampling
        samples_per_pixel=9,  # photons per output pixel, before the margin
        launch_margin=0.10,  # extra launch area beyond the frame, in frame heights
        chunk_points=4_000_000,  # photons refracted per GPU batch
        # look
        softness=0.7,  # px, blur of the raw histogram
        glow_weight=0.22,
        glow_sigma=7.0,
    )
    black_pct = 40.0  # log-density percentile pushed to black
    white_pct = 99.7  # log-density percentile driven to the end of the ramp

    device = _device()
    renderer = _Renderer(
        knobs, _wave_spectrum(rng, knobs), int(rng.integers(0, 2**31)), device
    )

    # --- one exposure for the whole clip, from probe frames around the loop ---
    lows, highs = [], []
    for tau in np.arange(3) / 3:
        sample = renderer.log_density(
            renderer.frame_density(tau * LOOP_SECONDS)
        ).flatten()[::7]
        lows.append(torch.quantile(sample, black_pct / 100).item())
        highs.append(torch.quantile(sample, white_pct / 100).item())
    renderer.black_level = float(np.mean(lows))
    renderer.white_level = float(np.max(highs))
    logger.info(
        f"log-density black {renderer.black_level:.3f} "
        f"white {renderer.white_level:.3f}"
    )

    frames_path = settings.frames_path
    logger.info(f"rendering {LOOP_FRAMES} frames")
    with ThreadPoolExecutor(PNG_WRITERS) as pool:
        pending = []
        for index in range(LOOP_FRAMES):
            frame = renderer.compose(
                renderer.frame_density(index / LOOP_FRAMES * LOOP_SECONDS)
            )
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

    del renderer
    gc.collect()
    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
