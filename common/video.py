"""
Shared plumbing for videos: timing, colours, and rendering frames to disk.

A video generator only supplies the pixels of one frame; `render_frames` runs it
for every frame (in parallel by default), writes the PNGs where
`ImageProcessingSettings.save_video` reads them, logs progress and reports how
long the frames took. Torch helpers live in `common.gpu`, so CPU-only videos
don't pay for importing torch.
"""

from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Sequence

import cv2
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from common.image_processing import ImageProcessingSettings

# RP_FPS in the environment gives a quick low-frame-rate preview with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
WIDTH, HEIGHT = 1080, 1920  # 9:16 for Reels
WORKERS = min(16, os.cpu_count() or 4)
PNG_WRITERS = 8
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def frame_count(duration: float, fps: int = FPS) -> int:
    """Number of frames in `duration` seconds."""
    return max(2, int(round(fps * duration)))


def smoothstep(x):
    """Ease-in-out on [0, 1] (3x^2 - 2x^3), clipped; works on scalars and arrays."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def hex_to_bgr(colour: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (b, g, r) ints, the colour order cv2 drawing expects."""
    return tuple(int(colour[i : i + 2], 16) for i in (5, 3, 1))


def palette_lut(colours: Sequence, size: int = 256) -> np.ndarray:
    """
    A smooth (size, 3) float32 RGB gradient in [0, 1] through `colours`, which are
    '#rrggbb' strings (e.g. a `colors.palettes` constant) or rows of RGB in [0, 1],
    spaced evenly. Index it with a scalar field scaled to 0 .. size - 1.
    """
    if isinstance(colours[0], str):
        colours = [[int(c[i : i + 2], 16) / 255.0 for i in (1, 3, 5)] for c in colours]
    colours = np.asarray(colours, dtype=np.float32)
    anchors = np.linspace(0.0, 1.0, len(colours))
    u = np.linspace(0.0, 1.0, size)
    lut = [np.interp(u, anchors, colours[:, c]) for c in range(3)]
    return np.stack(lut, axis=1).astype(np.float32)


def _to_bgr_uint8(image, rgb: bool) -> np.ndarray:
    """A frame as a uint8 BGR numpy array; tensors are moved off the GPU."""
    if not isinstance(image, np.ndarray):  # a torch tensor
        if rgb:
            image = image.flip(-1)
        return image.clamp(0, 255).byte().cpu().numpy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _write(path: str, image: np.ndarray):
    if not cv2.imwrite(path, image, PNG_PARAMS):
        raise OSError(f"Could not write {path}")


def render_frames(
    settings: ImageProcessingSettings,
    render: Callable[[int], object],
    count: int,
    *,
    parallel: bool = True,
    rgb: bool = False,
    workers: int = WORKERS,
):
    """
    Call `render(frame)` for frame = 0 .. count - 1 and write each result to
    frames/frame%04d.png, ready for `settings.save_video`.

    `render` returns a (HEIGHT, WIDTH, 3) image with values in 0..255: a numpy
    array (uint8, or float which is clipped) or a torch tensor on any device.
    Channels are BGR like cv2, or RGB with `rgb=True`.

    parallel=True: frames are rendered concurrently on `workers` threads, so
    `render` must be a pure function of the frame index (NumPy and cv2 release
    the GIL). Draw all randomness up front, never inside `render`.

    parallel=False: frames are rendered in order on this thread (GPU work, or a
    renderer that carries state from frame to frame) while a small pool writes
    the PNGs. Do any warm-up before calling.

    In an `--audio-only` run nothing is rendered; the frames on disk are kept.
    """
    frames_path = settings.frames_path
    if getattr(settings, "audio_only", False):
        found = len(list(frames_path.glob("frame*.png")))
        if found != count:
            raise RuntimeError(
                f"--audio-only needs the {count} frames of a full run in "
                f"{frames_path}, found {found}: render the video first "
                "(same --name and RP_FPS)"
            )
        logger.info(f"Audio only: keeping the {count} frames already rendered")
        return
    step = max(1, 5 * FPS)  # log every 5 s of video
    mode = f"{workers} threads" if parallel else "in order"
    logger.info(f"Rendering {count} frames ({count / FPS:.1f} s @ {FPS} fps), {mode}")
    start = time.perf_counter()

    def path(frame):
        return str(frames_path / f"frame{frame:04d}.png")

    if parallel:

        def task(frame):
            _write(path(frame), _to_bgr_uint8(render(frame), rgb))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, frame) for frame in range(count)]
            try:
                for done, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if done % step == 0:
                        logger.info(f"{done}/{count} frames")
            except BaseException:
                pool.shutdown(cancel_futures=True)  # fail now, not after every frame
                raise
    else:
        with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
            pending = deque()
            for frame in range(count):
                image = _to_bgr_uint8(render(frame), rgb)
                pending.append(pool.submit(_write, path(frame), image))
                while len(pending) > 2 * PNG_WRITERS:  # keep memory flat
                    pending.popleft().result()
                if (frame + 1) % step == 0:
                    logger.info(f"{frame + 1}/{count} frames")
            for future in pending:
                future.result()

    elapsed = time.perf_counter() - start
    logger.info(
        f"Rendered {count} frames in {elapsed:.1f} s ({count / elapsed:.0f} frames/s)"
    )
