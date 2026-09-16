"""
Moiré Tide

Straight, evenly spaced waves are laid over one another, bent through a
slow-moving current, and left to interfere. Nothing here is drawn as a
curve — every ingredient is a plain cosine — yet where the crests agree
and disagree, great soft bands swell up, roll across the frame and
dissolve. This is moiré: interference you can see with the naked eye.

A ring of waves spreads from a point that circles the frame, and three
straight gratings turn against it. The current that warps them is itself
made of waves, travelling at their own speeds, so the fringes wander and
break the way a swell does under wind. Every rate is a whole number of
cycles per loop, so the tide closes on itself exactly: the last frame is
the first frame, with nothing faded or cut.

Nothing here knows what water is. There is no fluid, no surface, no light
— only cosines added together and a colour ramp that spends most of its
range in the dark and saves the last of it for the crests. That is enough
to make the eye see deep water under a low sun.

Made with Python, PyTorch & NumPy.
#generativeart #moire #interference #waves #water #ocean #mathematics
#randomplots #creativecoding #algorithmicart #mathisbeautiful
"""

import gc
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings

RUN_SECONDS = 20.0
FPS = int(os.environ.get("RP_FPS", "30"))
FRAME_COUNT = max(2, int(RUN_SECONDS * FPS))
WIDTH, HEIGHT = 1080, 1920  # 9:16 portrait for Reels / Stories
PNG_WRITERS = 10
PNG_QUEUE = PNG_WRITERS * 3  # cap the in-flight frames so RAM stays bounded
PNG_PARAMS = (cv2.IMWRITE_PNG_COMPRESSION, 1)  # lossless; level 1 encodes ~4x faster

# One entry per straight grating: spatial frequency, starting angle, half-turns per
# loop, and phase cycles per loop. Integer half-turns and integer phase cycles are
# what make the animation seamless.
GRATINGS = (
    (52.0, 0.0, 1, 0),
    (52.0, np.pi / 3.0, -1, 1),
    (26.0, np.pi / 7.0, 2, -1),
)

# The circular grating: frequency, how far its centre orbits, and how many turns
# that orbit makes over the loop.
RING_FREQ = 62.0
RING_ORBIT = 0.85
RING_TURNS = 1
RING_PHASE_CYCLES = -2

# The current that bends everything. Each entry is (amplitude, frequency, direction,
# static phase, cycles per loop) — a travelling plane wave of displacement.
WARP_WAVES = (
    (0.085, 2.3, 0.4, 0.0, 1),
    (0.055, 4.1, 2.1, 1.7, -1),
    (0.032, 7.3, 3.9, 0.6, 2),
)
WARP_SWING = 0.55  # how much the current strengthens and slackens over the loop

BREATH_DEPTH = 0.14  # how much the frequencies swell over the loop
SHARPNESS = 2.2  # drives the bands from soft gradients towards hard contours
MOIRE_BIAS = 0.45  # weight of the ring x line product — the big slow beat envelope
VIGNETTE = 0.42  # falloff towards the top and bottom of the portrait frame
GLOW = 0.20  # strength of the blurred highlight pass — the sun's sheen on the water
GLOW_RADIUS = 13  # tight, so the glints stay points of light instead of hazing over
UNSHARP = 1.15  # high-frequency emphasis: the bite in the ripple edges
UNSHARP_RADIUS = 5

# Monotonic ramp, trough to crest. Water is far less colourful than memory insists:
# near-neutral slate with only a trace of blue-green, most of the range crowded into
# the shadows, and the light confined to the top few percent so the crests read as
# sparse specular glints rather than a lit surface. Nothing reaches white.
CMAP = LinearSegmentedColormap.from_list(
    "open_water",
    [
        (0.00, "#05080b"),
        (0.20, "#0a1116"),
        (0.40, "#121d24"),
        (0.55, "#1b2a32"),
        (0.68, "#273941"),
        (0.78, "#374a52"),
        (0.86, "#4e6067"),
        (0.92, "#6d7d83"),
        (0.96, "#96a1a5"),
        (1.00, "#dfe4e5"),
    ],
)
LUT = (np.asarray(CMAP(np.linspace(0.0, 1.0, 256)))[:, 2::-1] * 255).astype(np.uint8)


def generate(settings: ImageProcessingSettings = None):
    """
    Animated warped multi-grating moiré, rendered straight to pixels.

    Each frame builds a displacement field from a handful of travelling plane waves,
    d(r, t) = Σ_m a_m sin(f_m (r · n_m) + p_m + 2π q_m t), and samples the gratings at
    the displaced coordinates r + d. Three straight gratings, cos(k_j(t) · r + φ_j(t)),
    are summed there and a circular grating, cos(k_r |r - c(t)| + φ_r(t)), is added
    from a centre c(t) orbiting the frame. Their product supplies the slow beat
    envelope that the straight-only version lacked. Exactness of the loop is
    structural: every wavevector rotates an integer number of half turns (a cosine
    grating is invariant under a half turn), every phase advances a whole number of
    cycles, and the orbit closes, so frame N is frame 0 with no crossfade. The result
    is steepened with tanh, vignetted, mapped through a 256-entry colour LUT and given
    a blurred highlight pass.

    There is no matplotlib figure in the loop. Every pixel operation is elementwise, so
    the whole frame is built as one torch expression on the GPU when CUDA is present
    (torch on CPU, which is itself threaded, otherwise), including the glow -- a
    separable Gaussian run as two grouped convolutions. Only the finished uint8 frame
    crosses back to the host, where a pool of writer threads encodes the PNGs while the
    GPU is already computing the next frames.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"rendering {FRAME_COUNT} frames on {device}")

    aspect = HEIGHT / WIDTH
    x, y = torch.meshgrid(
        torch.linspace(-1.0, 1.0, WIDTH, device=device),
        torch.linspace(-aspect, aspect, HEIGHT, device=device),
        indexing="xy",
    )
    # Applied to the rendered colour, not the field: it should sink the far water into
    # shadow, not flatten the waves there to the middle of the ramp.
    vignette = torch.exp(-VIGNETTE * (x * x + y * y)).view(1, 1, HEIGHT, WIDTH)
    lut = torch.from_numpy(LUT).to(device=device, dtype=torch.float32)

    # Separable Gaussians, run as grouped 1D convolutions over the 3 channels: a wide
    # one for the glow, a tight one whose difference from the image is the unsharp mask.
    def gaussian(radius):
        kernel = torch.from_numpy(cv2.getGaussianKernel(radius, 0).astype(np.float32))
        kernel = kernel.flatten().to(device)
        return (
            kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1).contiguous(),
            kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1).contiguous(),
            radius // 2,
        )

    def blur(image, kernels):
        kernel_h, kernel_v, pad = kernels
        image = F.conv2d(
            F.pad(image, (pad, pad, 0, 0), "replicate"), kernel_h, groups=3
        )
        return F.conv2d(F.pad(image, (0, 0, pad, pad), "replicate"), kernel_v, groups=3)

    glow_kernels = gaussian(GLOW_RADIUS)
    unsharp_kernels = gaussian(UNSHARP_RADIUS)

    # A small per-seed nudge so different seeds are visibly different compositions.
    angle_jitter = rng.uniform(-0.12, 0.12, size=len(GRATINGS))
    freq_jitter = rng.uniform(0.9, 1.1, size=len(GRATINGS))
    warp_jitter = rng.uniform(0.0, 2.0 * np.pi, size=len(WARP_WAVES))

    # Projections of the grid onto each warp wave's direction: static, so precompute.
    warp_projections = []
    for index, (_, freq, direction, _, _) in enumerate(WARP_WAVES):
        theta = direction + 0.15 * np.sin(warp_jitter[index])
        warp_projections.append(freq * (x * np.cos(theta) + y * np.sin(theta)))

    frames_path = settings.frames_path
    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool, torch.inference_mode():
        pending = deque()
        for frame in range(FRAME_COUNT):
            t = frame / FRAME_COUNT
            tau = 2.0 * np.pi * t
            breath = 1.0 + BREATH_DEPTH * np.sin(tau)
            swing = 1.0 + WARP_SWING * np.sin(tau - 1.1)

            # The current: one displacement field, applied along both axes with a
            # quarter-cycle offset so it swirls rather than shears.
            u = x.clone()
            v = y.clone()
            for index, (amp, _, _, phase0, cycles) in enumerate(WARP_WAVES):
                projection = warp_projections[index]
                drift = phase0 + warp_jitter[index] + 2.0 * np.pi * cycles * t
                u += (amp * swing) * torch.sin(projection + drift)
                v += (amp * swing) * torch.cos(projection * 0.87 - drift)

            lines = torch.zeros_like(x)
            for index, (freq, angle0, half_turns, phase_cycles) in enumerate(GRATINGS):
                angle = angle0 + angle_jitter[index] + np.pi * half_turns * t
                phase = 2.0 * np.pi * phase_cycles * t
                k = freq * freq_jitter[index] * breath
                lines += torch.cos(
                    (k * np.cos(angle)) * u + (k * np.sin(angle)) * v + phase
                )
            lines /= len(GRATINGS)

            cx = RING_ORBIT * np.cos(2.0 * np.pi * RING_TURNS * t)
            cy = RING_ORBIT * aspect * np.sin(2.0 * np.pi * RING_TURNS * t)
            u -= cx
            v -= cy
            radius = torch.sqrt(u * u + v * v)
            ring = torch.cos(
                (RING_FREQ * breath) * radius + 2.0 * np.pi * RING_PHASE_CYCLES * t
            )

            # Sum for the fringes, product for the slow beat that rolls across them.
            field = (1.0 - MOIRE_BIAS) * 0.5 * (
                lines + ring
            ) + MOIRE_BIAS * lines * ring
            field = torch.tanh(SHARPNESS * field)

            indices = torch.clamp((field + 1.0) * 127.5, 0, 255).long()
            image = lut[indices].permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W
            image *= vignette

            # Unsharp mask puts the bite back into the ripple edges; the wider blur is
            # added on top as the sheen. Sharpen first, so the glow is not re-sharpened.
            image += UNSHARP * (image - blur(image, unsharp_kernels))
            image = torch.clamp(image + GLOW * blur(image, glow_kernels), 0, 255)
            image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

            # Keep the writer queue shallow so finished frames don't pile up in RAM.
            while len(pending) >= PNG_QUEUE:
                pending.popleft().result()
            pending.append(
                pool.submit(
                    cv2.imwrite,
                    str(frames_path / f"frame{frame:04d}.png"),
                    image,
                    PNG_PARAMS,
                )
            )
            if frame % 60 == 0:
                logger.info(f"frame {frame + 1}/{FRAME_COUNT}")
        for task in pending:
            task.result()

    gc.collect()
    settings.save_video(fps=FPS, crf=18)


if __name__ == "__main__":
    generate()
