"""
Conway's Loom

A single rule governs every cell: too few or too many neighbours, die;
exactly three, be born. From that sentence emerge gliders, guns, still
lifes, and collapses — the whole zoo of Life, woven on a torus.

This is John Conway's Game of Life (1970): a zero-player game on a grid
where the universe computes itself. After the first random breath there
is no chance — every generation is the honest output of the last. The
violet glow is where the field is still burning; the mint-green specks
are ash, the small machines that survived and then went quiet forever.

Made with Python & PyTorch.
#generativeart #gameoflife #conway #cellularautomata #mathematics
#randomplots #creativecoding #algorithmicart #mathisbeautiful
"""

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from colors.palettes import ULTRAVIOLET_BLOOM
from common.image_processing import ImageProcessingSettings

# Single knob: clip length in seconds. One Life generation per frame at 60 fps.
# RP_FPS=5 in the environment gives a quick test render at lower frame rate.
RUN_SECONDS = 30.0
FPS = int(os.environ.get("RP_FPS", "60"))
FRAME_COUNT = max(2, int(RUN_SECONDS * FPS))
STEPS_PER_FRAME = 1
WIDTH, HEIGHT = 1080, 1920  # 9:16 portrait for Reels / Stories
CELL = 6  # screen pixels per cell -- structures have to read on a phone
SIM_WIDTH, SIM_HEIGHT = WIDTH // CELL, HEIGHT // CELL
INITIAL_DENSITY = 0.22
PNG_WRITERS = 4

# Colour: every layer takes its own slice of one palette ramp. Cells run violet
# (newborn) -> white (burning) -> mint (ancient, i.e. still lifes); the phosphor
# and scar fields sit underneath them in the indigo end.
PALETTE = ULTRAVIOLET_BLOOM
AGE_RANGE = (0.32, 0.78)
AGE_FULL = 96.0  # generations to reach the top of the age ramp
PHOSPHOR_RANGE = (0.09, 0.33)
PHOSPHOR_DECAY = 0.985  # ~0.8 s half-life at 60 fps
PHOSPHOR_GAIN = 1.2
PHOSPHOR_SIGMA = 2.6
SCAR_RANGE = (0.10, 0.26)
SCAR_GAIN = 0.35
SCAR_SIGMA = 2.0
BLOOM_GAIN = 0.5
BLOOM_SIGMA = 2.2
PERSISTENCE = 0.55  # CRT-style frame memory; stops blinkers strobing at 60 fps
VIGNETTE = 0.3


def _device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.warning("CUDA not available — rendering on the CPU with torch")
    return torch.device("cpu")


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    return kernel / kernel.sum()


def _blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) tensor with circular borders."""
    radius = (kernel.numel() - 1) // 2
    channels = image.shape[0]
    x = image[None]
    x = F.pad(x, (radius, radius, 0, 0), mode="circular")
    x = F.conv2d(
        x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels
    )
    x = F.pad(x, (0, 0, radius, radius), mode="circular")
    x = F.conv2d(
        x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels
    )
    return x[0]


def _vignette(device: torch.device) -> torch.Tensor:
    """Gentle radial falloff, so the torus reads as a lit object, not a tile."""
    y = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    x = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    radius = ((x * x + y * y) / 2.0).clamp(0.0, 1.0)
    return (1.0 - VIGNETTE * radius**1.5)[None]


def _write_png(path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


class _LifeRenderer:
    """Conway's Game of Life on a torus, lit by age, phosphor and burn scars."""

    def __init__(self, seed: int, device: torch.device):
        self.device = device
        gen = torch.Generator(device=device).manual_seed(seed)

        kernel = torch.ones(1, 1, 3, 3, device=device, dtype=torch.float32)
        kernel[0, 0, 1, 1] = 0.0
        self.neighbor_kernel = kernel

        lut = LinearSegmentedColormap.from_list("life", PALETTE, N=1024)(
            np.linspace(0.0, 1.0, 1024)
        )[:, :3]
        self.lut = torch.tensor(lut, dtype=torch.float32, device=device)
        self.phosphor_kernel = _gaussian_kernel(PHOSPHOR_SIGMA, device)
        self.scar_kernel = _gaussian_kernel(SCAR_SIGMA, device)
        self.bloom_kernel = _gaussian_kernel(BLOOM_SIGMA, device)
        self.vignette = _vignette(device)

        alive = (
            torch.rand(SIM_HEIGHT, SIM_WIDTH, generator=gen, device=device)
            < INITIAL_DENSITY
        )
        self.grid = alive.to(torch.float32)
        self.age = self.grid.clone()  # generations this cell has been alive
        self.phosphor = self.grid.clone()  # recently alive, decaying
        self.scar = self.grid.clone()  # how often this cell has ever lived
        self.generation = 0
        self.previous = None

    def _neighbor_count(self, grid: torch.Tensor) -> torch.Tensor:
        field = grid[None, None]
        padded = F.pad(field, (1, 1, 1, 1), mode="circular")
        return F.conv2d(padded, self.neighbor_kernel)[0, 0]

    def _ramp(self, value: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """Map a 0-1 field onto the [lo, hi] slice of the palette, as (3, H, W)."""
        index = ((lo + (hi - lo) * value.clamp(0.0, 1.0)) * 1023).long()
        return self.lut[index].permute(2, 0, 1)

    def step(self, n_steps: int = 1):
        """Advance B3/S23 Life on a torus for n_steps generations."""
        for _ in range(n_steps):
            alive = self.grid > 0.5
            neighbors = self._neighbor_count(self.grid)
            birth = (~alive) & (neighbors == 3)
            survive = alive & ((neighbors == 2) | (neighbors == 3))
            next_alive = birth | survive

            self.age = torch.where(
                next_alive,
                torch.where(alive, self.age + 1.0, torch.ones_like(self.age)),
                torch.zeros_like(self.age),
            )
            self.grid = next_alive.to(torch.float32)
            self.phosphor = torch.maximum(self.phosphor * PHOSPHOR_DECAY, self.grid)
            self.scar = self.scar + self.grid
            self.generation += 1

    def compose(self) -> np.ndarray:
        """Stack scar, phosphor and living cells into a WIDTH×HEIGHT BGR frame."""
        alive = self.grid > 0.5

        # Soft layers, blurred at simulation resolution and upscaled smoothly.
        phosphor = _blur(self.phosphor[None], self.phosphor_kernel)[0]
        ground = self._ramp(phosphor, *PHOSPHOR_RANGE) * (
            PHOSPHOR_GAIN * phosphor
        ).unsqueeze(0)

        scar = torch.log1p(self.scar) / np.log1p(0.35 * max(self.generation, 1) + 1.0)
        scar = _blur(scar.clamp(0.0, 1.0)[None], self.scar_kernel)[0]
        ground = ground + self._ramp(scar, *SCAR_RANGE) * (SCAR_GAIN * scar).unsqueeze(
            0
        )

        # Living cells: hard-edged, one crisp block of CELL pixels each.
        age_norm = torch.log1p(self.age) / np.log1p(AGE_FULL)
        cells = self._ramp(age_norm, *AGE_RANGE) * alive.unsqueeze(0).float()
        ground = ground + BLOOM_GAIN * _blur(cells, self.bloom_kernel)

        rgb = F.interpolate(
            ground[None], scale_factor=CELL, mode="bilinear", align_corners=False
        )[0]
        rgb = rgb + F.interpolate(cells[None], scale_factor=CELL, mode="nearest")[0]
        rgb = (rgb * self.vignette).clamp(0.0, 1.0)

        if self.previous is not None:
            rgb = torch.maximum(rgb, self.previous * PERSISTENCE)
        self.previous = rgb

        frame = (rgb * 255.0).round().to(torch.uint8)
        return frame.permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1]


def generate(settings: ImageProcessingSettings = None):
    """Conway's Game of Life on a toroidal 180×320 grid, six screen pixels a cell.

    Each cell follows the B3/S23 rule: a dead cell with exactly three live
    neighbours is born; a live cell with two or three neighbours survives;
    all other live cells die. Neighbour counts use a 3×3 convolution with
    circular padding, so the grid wraps at every edge.

    Three fields are stacked, each taking its own slice of the
    ULTRAVIOLET_BLOOM ramp. A cumulative burn *scar* map washes the ground in
    dark indigo wherever the colony has ever lived; a *phosphor* field (the max
    of the live grid and its own 0.985 decay, blurred) glows violet where the
    field is still churning; the live cells sit on top, tinted by log-age from
    violet (newborn) through white to mint (ancient, so still lifes read as
    green ash), and blurred once more for bloom. The soft layers are upscaled
    bilinearly and the cells with nearest neighbour, so the nebula stays smooth
    while the cells stay crisp. Each frame then keeps 55% of the previous one,
    which turns the 30 Hz strobe of a field full of blinkers into a pulse.

    RUN_SECONDS sets clip length at 60 fps with one generation per frame.
    Rendering runs on the GPU; PNG encoding uses a thread pool.
    """
    settings = settings or ImageProcessingSettings(1)
    device = _device()
    renderer = _LifeRenderer(int(settings.rng.integers(0, 2**31)), device)

    frames_path = settings.frames_path
    logger.info(
        f"rendering {FRAME_COUNT} frames ({RUN_SECONDS:.1f}s @ {FPS} fps) "
        f"at {SIM_WIDTH}×{SIM_HEIGHT}, {STEPS_PER_FRAME} gen/frame"
    )

    with ThreadPoolExecutor(PNG_WRITERS) as pool:
        pending = []
        for index in range(FRAME_COUNT):
            renderer.step(STEPS_PER_FRAME)
            frame = renderer.compose()
            pending.append(
                pool.submit(_write_png, frames_path / f"frame{index:04d}.png", frame)
            )
            if len(pending) > 2 * PNG_WRITERS:
                pending.pop(0).result()
        for job in pending:
            job.result()

    del renderer
    gc.collect()
    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
