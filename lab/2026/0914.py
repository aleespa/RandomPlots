"""
Turing Lace

Two chemicals share a periodic plane and obey the Gray-Scott equations:

    du/dt = Du ∇²u − uv² + F(1 − u)
    dv/dt = Dv ∇²v + uv² − (F + k) v

U is fed in at rate F and consumed by the autocatalytic reaction
U + 2V → 3V; V is removed at rate F + k. Stirred together, the two settle
into a flat, uniform state. Let them diffuse and that calm breaks: the
substrate spreads twice as fast as the catalyst, so a narrow band of
wavenumbers grows while every other scale decays. This is the
diffusion-driven instability Alan Turing described in 1952, and the
fastest-growing mode sets the spacing of the lace.

The feed and kill rates drift slowly along a path through the (F, k) plane,
carrying the field across bifurcations — spots divide, stretch into
filaments, knit into dense lace, then dissolve. The colour traces level
sets of V.

Made with Python, PyTorch & Matplotlib.

#generativeart #reactiondiffusion #turingpatterns #grayscott #mathart
#pde #creativecoding #algorithmicart #simulation #randomplots
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

from common.image_processing import ImageProcessingSettings


# ---------------------------------------------------------------------------
# VIDEO
# ---------------------------------------------------------------------------

RUN_SECONDS = 14
FPS = int(os.environ.get("RP_FPS", "60"))

FRAME_COUNT = max(2, int(RUN_SECONDS * FPS))
STEPS_PER_FRAME = 36

WIDTH, HEIGHT = 1080, 1920
SIM_WIDTH = 1080
SIM_HEIGHT = 1920

PNG_WRITERS = 8


# ---------------------------------------------------------------------------
# GRAY-SCOTT PARAMETERS
# ---------------------------------------------------------------------------

DU = 0.16
DV = 0.08
DT = 1.0


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------

PALETTE = [
    "#071A2B",  # deep ocean
    "#004E7A",  # blue
    "#0077B6",  # vivid blue
    "#00B4D8",  # cyan
    "#00F5D4",  # turquoise
    "#00FFAA",  # aqua-green
    "#7BFF00",  # electric green
    "#D8FF00",  # lime
    "#F9FF00",  # yellow
    "#FFFFFF",  # white
]

CONTOUR_LEVELS = 18
CONTOUR_WIDTH = 0.010


# ---------------------------------------------------------------------------
# DETERMINISTIC INITIAL CONDITION
# ---------------------------------------------------------------------------

SEED_POINTS = [
    (0.20, 0.18),
    (0.72, 0.16),
    (0.48, 0.30),
    (0.27, 0.48),
    (0.76, 0.47),
    (0.45, 0.62),
    (0.18, 0.78),
    (0.72, 0.82),
]

SEED_RADII = [7] * len(SEED_POINTS)


# ---------------------------------------------------------------------------
# PARAMETER TRAJECTORY
# ---------------------------------------------------------------------------

FK_KEYFRAMES = np.array(
    [
        (0.040, 0.060),
        (0.035, 0.065),
        (0.030, 0.062),
        (0.054, 0.062),
        (0.047, 0.061),
    ],
    dtype=np.float64,
)

FK_TIMES = np.array([
    0.00,
    0.13,
    0.33,
    0.57,
    0.73,
])


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(
            f"rendering on {torch.cuda.get_device_name(0)}"
        )
        return torch.device("cuda")

    logger.warning(
        "CUDA not available — rendering on the CPU with torch"
    )
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# GAUSSIAN BLUR
# ---------------------------------------------------------------------------

def _gaussian_kernel(
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))

    t = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=torch.float32,
    )

    kernel = torch.exp(-0.5 * (t / sigma) ** 2)

    return kernel / kernel.sum()


def _blur(
    image: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Separable Gaussian blur with circular borders."""

    radius = (kernel.numel() - 1) // 2
    channels = image.shape[0]

    x = image[None]

    x = F.pad(
        x,
        (radius, radius, 0, 0),
        mode="circular",
    )

    x = F.conv2d(
        x,
        kernel.view(1, 1, 1, -1).expand(
            channels, 1, 1, -1
        ),
        groups=channels,
    )

    x = F.pad(
        x,
        (0, 0, radius, radius),
        mode="circular",
    )

    x = F.conv2d(
        x,
        kernel.view(1, 1, -1, 1).expand(
            channels, 1, -1, 1
        ),
        groups=channels,
    )

    return x[0]


# ---------------------------------------------------------------------------
# PARAMETER INTERPOLATION
# ---------------------------------------------------------------------------

def _interp_fk(t: float) -> tuple[float, float]:
    """Smoothly interpolate feed and kill along the choreographed path."""

    t = float(np.clip(t, 0.0, 1.0))

    if t >= 1.0:
        return (
            float(FK_KEYFRAMES[-1, 0]),
            float(FK_KEYFRAMES[-1, 1]),
        )

    index = np.searchsorted(
        FK_TIMES,
        t,
        side="right",
    ) - 1

    index = int(
        np.clip(
            index,
            0,
            len(FK_KEYFRAMES) - 2,
        )
    )

    t0 = FK_TIMES[index]
    t1 = FK_TIMES[index + 1]

    frac = (t - t0) / (t1 - t0)

    # Cosine easing.
    smooth = 0.5 - 0.5 * np.cos(np.pi * frac)

    f0, k0 = FK_KEYFRAMES[index]
    f1, k1 = FK_KEYFRAMES[index + 1]

    feed = f0 + smooth * (f1 - f0)
    kill = k0 + smooth * (k1 - k0)

    return float(feed), float(kill)


# ---------------------------------------------------------------------------
# PNG WRITER
# ---------------------------------------------------------------------------

def _write_png(path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


# ---------------------------------------------------------------------------
# RENDERER
# ---------------------------------------------------------------------------

class _Renderer:
    """Gray–Scott integrator and lace renderer."""

    def __init__(self, device: torch.device):
        self.device = device

        # ---------------------------------------------------------------
        # LAPLACIAN
        # ---------------------------------------------------------------

        lap = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )

        self.lap_kernel = lap.view(1, 1, 3, 3)

        # ---------------------------------------------------------------
        # SOBEL
        # ---------------------------------------------------------------

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = sobel_x.transpose(-1, -2)

        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

        # ---------------------------------------------------------------
        # COLOUR LUT
        # ---------------------------------------------------------------

        lut = LinearSegmentedColormap.from_list(
            "lace",
            PALETTE,
            N=1024,
        )(
            np.linspace(0.0, 1.0, 1024)
        )[:, :3]

        self.lut = torch.tensor(
            lut,
            dtype=torch.float32,
            device=device,
        )

        # ---------------------------------------------------------------
        # CONTOURS
        # ---------------------------------------------------------------

        self.levels = torch.linspace(
            0.12,
            0.88,
            CONTOUR_LEVELS,
            device=device,
        )

        level_colors = self.lut[
            (
                torch.linspace(
                    0.08,
                    0.95,
                    CONTOUR_LEVELS,
                    device=device,
                ) * 1023
            ).long()
        ]

        self.level_colors = level_colors

        # ---------------------------------------------------------------
        # BLUR KERNELS
        # ---------------------------------------------------------------

        self.glow_kernel = _gaussian_kernel(2.4, device)
        self.soft_kernel = _gaussian_kernel(0.9, device)

        # ---------------------------------------------------------------
        # INITIAL CHEMICAL STATE
        # ---------------------------------------------------------------

        u = torch.ones(
            1,
            1,
            SIM_HEIGHT,
            SIM_WIDTH,
            device=device,
        )

        v = torch.zeros(
            1,
            1,
            SIM_HEIGHT,
            SIM_WIDTH,
            device=device,
        )

        yy, xx = torch.meshgrid(
            torch.arange(
                SIM_HEIGHT,
                device=device,
                dtype=torch.float32,
            ),
            torch.arange(
                SIM_WIDTH,
                device=device,
                dtype=torch.float32,
            ),
            indexing="ij",
        )

        # ---------------------------------------------------------------
        # DELIBERATELY PLACED SEEDS
        # ---------------------------------------------------------------

        for (x_frac, y_frac), radius in zip(
            SEED_POINTS,
            SEED_RADII,
        ):
            cx = x_frac * SIM_WIDTH
            cy = y_frac * SIM_HEIGHT

            mask = (
                (xx - cx) ** 2
                + (yy - cy) ** 2
                <= radius ** 2
            )

            v = torch.where(
                mask,
                torch.ones_like(v),
                v,
            )

            u = torch.where(
                mask,
                torch.full_like(u, 0.50),
                u,
            )

        self.u = u
        self.v = v

    # ------------------------------------------------------------------
    # LAPLACIAN
    # ------------------------------------------------------------------

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            F.pad(
                field,
                (1, 1, 1, 1),
                mode="circular",
            ),
            self.lap_kernel,
        )

    # ------------------------------------------------------------------
    # GRAY-SCOTT STEP
    # ------------------------------------------------------------------

    def step(
        self,
        feed: float,
        kill: float,
        n_steps: int,
    ):
        for _ in range(n_steps):
            u = self.u
            v = self.v

            uvv = u * v * v

            du = (
                DU * self._laplacian(u)
                - uvv
                + feed * (1.0 - u)
            )

            dv = (
                DV * self._laplacian(v)
                + uvv
                - (feed + kill) * v
            )

            self.u = (u + DT * du).clamp(0.0, 1.0)
            self.v = (v + DT * dv).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # COMPOSITION
    # ------------------------------------------------------------------

    def compose(self, t: float) -> np.ndarray:
        v = self.v[0, 0]

        # ---------------------------------------------------------------
        # CONTRAST STRETCH
        # ---------------------------------------------------------------

        v_min = v.quantile(0.02)
        v_max = v.quantile(0.98)

        v = (
            (v - v_min)
            / (v_max - v_min).clamp(min=1e-4)
        ).clamp(0.0, 1.0)

        field = v[None]

        # ---------------------------------------------------------------
        # SOBEL EDGES
        # ---------------------------------------------------------------

        gx = F.conv2d(
            F.pad(
                field,
                (1, 1, 1, 1),
                mode="circular",
            ),
            self.sobel_x,
        )

        gy = F.conv2d(
            F.pad(
                field,
                (1, 1, 1, 1),
                mode="circular",
            ),
            self.sobel_y,
        )

        edge = torch.sqrt(gx * gx + gy * gy)[0]

        edge = edge / edge.quantile(0.985).clamp(min=1e-4)

        # ---------------------------------------------------------------
        # CINEMATIC INTENSITY
        # ---------------------------------------------------------------

        intensity = np.exp(
            -0.5 * ((t - 0.68) / 0.27) ** 2
        )

        intro = np.clip(t / 0.10, 0.0, 1.0)
        outro = np.clip((1.0 - t) / 0.10, 0.0, 1.0)

        intensity *= intro * outro
        intensity = 0.35 + 0.65 * intensity

        # ---------------------------------------------------------------
        # SUBTLE BREATHING
        # ---------------------------------------------------------------

        breathing = (
            1.0
            + 0.08 * np.sin(2.0 * np.pi * 1.25 * t)
        )

        dynamic_width = CONTOUR_WIDTH * breathing

        # ---------------------------------------------------------------
        # ISO-CONTOURS
        # ---------------------------------------------------------------

        rgb = torch.zeros(
            3,
            SIM_HEIGHT,
            SIM_WIDTH,
            device=self.device,
        )

        for index, level in enumerate(self.levels):
            band = torch.exp(
                -((v - level) / dynamic_width) ** 2
            )

            color = self.level_colors[index]

            rgb += (
                band.unsqueeze(0)
                * color.view(3, 1, 1)
            )

        rgb = rgb.clamp(0.0, 1.0)

        # ---------------------------------------------------------------
        # DYNAMIC GLOW
        # ---------------------------------------------------------------

        glow_strength = 0.16 + 0.32 * intensity
        edge_strength = 0.28 + 0.45 * intensity

        rgb = (
            rgb
            + glow_strength
            * _blur(rgb, self.glow_kernel)
            * rgb
        )

        rgb = (
            rgb
            + edge_strength
            * edge.unsqueeze(0)
            * self.lut[900].view(3, 1, 1)
        )

        # ---------------------------------------------------------------
        # ATMOSPHERIC HAZE
        # ---------------------------------------------------------------

        haze = self.lut[
            (
                _blur(
                    v[None],
                    self.soft_kernel,
                )[0].clamp(0.0, 1.0)
                * 1023
            ).long()
        ].permute(2, 0, 1)

        haze_strength = 0.10 + 0.16 * intensity

        rgb = (
            haze_strength * haze
            + (1.0 - haze_strength) * rgb
        )

        # ---------------------------------------------------------------
        # GLOBAL INTENSITY
        # ---------------------------------------------------------------

        rgb *= 0.72 + 0.28 * intensity
        rgb = rgb.clamp(0.0, 1.0)

        # ---------------------------------------------------------------
        # SUBTLE SCALE BREATHING
        # ---------------------------------------------------------------

        zoom = 1.0 + 0.035 * np.sin(np.pi * t)

        if abs(zoom - 1.0) > 1e-5:
            crop_h = int(SIM_HEIGHT / zoom)
            crop_w = int(SIM_WIDTH / zoom)

            y0 = (SIM_HEIGHT - crop_h) // 2
            x0 = (SIM_WIDTH - crop_w) // 2

            rgb = rgb[
                :,
                y0:y0 + crop_h,
                x0:x0 + crop_w,
            ]

        # ---------------------------------------------------------------
        # UPSCALE
        # ---------------------------------------------------------------

        rgb = F.interpolate(
            rgb[None],
            size=(HEIGHT, WIDTH),
            mode="bilinear",
            align_corners=False,
        )[0]

        # ---------------------------------------------------------------
        # RGB → BGR
        # ---------------------------------------------------------------

        frame = (
            rgb.clamp(0.0, 1.0)
            * 255.0
        ).round().to(torch.uint8)

        return (
            frame
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()[:, :, ::-1]
        )


# ---------------------------------------------------------------------------
# GENERATE
# ---------------------------------------------------------------------------

def generate(
    settings: ImageProcessingSettings = None,
):
    """
    Generate the deterministic Turing Lace animation.

    The simulation progresses continuously through:

        emergence
            ↓
        multiplication
            ↓
        transformation
            ↓
        dense lace / climax
            ↓
        dissolution

    There is no random initialisation and no simulation reset.
    """

    settings = (
        settings
        or ImageProcessingSettings(1)
    )

    device = _device()

    renderer = _Renderer(device)

    # ---------------------------------------------------------------
    # SHORT BURN-IN
    # ---------------------------------------------------------------

    f0, k0 = _interp_fk(0.0)

    logger.info(
        f"short burn-in at "
        f"f={f0:.4f}, "
        f"k={k0:.4f}"
    )

    renderer.step(
        f0,
        k0,
        STEPS_PER_FRAME * 8,
    )

    frames_path = settings.frames_path

    logger.info(
        f"rendering "
        f"{FRAME_COUNT} frames "
        f"({RUN_SECONDS:.1f}s @ {FPS} fps) "
        f"at {SIM_WIDTH}×{SIM_HEIGHT}, "
        f"{STEPS_PER_FRAME} steps/frame"
    )

    # ---------------------------------------------------------------
    # FRAME LOOP
    # ---------------------------------------------------------------

    with ThreadPoolExecutor(PNG_WRITERS) as pool:
        pending = []

        for index in range(FRAME_COUNT):
            t = index / max(FRAME_COUNT - 1, 1)

            feed, kill = _interp_fk(t)

            renderer.step(
                feed,
                kill,
                STEPS_PER_FRAME,
            )

            frame = renderer.compose(t)

            pending.append(
                pool.submit(
                    _write_png,
                    frames_path / f"frame{index:04d}.png",
                    frame,
                )
            )

            if len(pending) > 2 * PNG_WRITERS:
                pending.pop(0).result()

        for job in pending:
            job.result()

    del renderer
    gc.collect()

    settings.save_video(
        FPS,
        crf=18,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate()
