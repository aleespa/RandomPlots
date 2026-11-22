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


def latex_mask(tex: str, size: float) -> np.ndarray:
    r"""
    On-screen text typeset as LaTeX: `tex` (mathtext, e.g. r"$\frac{\sin 3x}{3}$" or
    r"$\mathrm{the\ jump{,}\ magnified}$") in Computer Modern at `size` px per em, as a
    float32 alpha mask in [0, 1] cropped to the ink with a 2 px margin. Build masks once,
    outside the frame function, and composite them with `blit_mask`. Words go in
    \mathrm{...} with "\ " for spaces; write commas as {,} so math mode adds no gap.
    Mathtext draws a bare ' as a prime; write apostrophes as \text{'}.
    """
    import io

    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties

    buffer = io.BytesIO()
    prop = FontProperties(size=size, math_fontfamily="cm")
    mathtext.math_to_image(tex, buffer, prop=prop, dpi=72, format="png")
    grey = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    mask = 1 - grey.astype(np.float32) / 255  # black ink on white
    rows = np.flatnonzero(mask.max(1) > 0)
    cols = np.flatnonzero(mask.max(0) > 0)
    return np.pad(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], 2)


def blit_mask(image, mask, x, y, colour, alpha=1.0, anchor="left"):
    """
    Composite a text mask in place onto a float (H, W, 3) image at top-left (x, y), or
    top-centre with anchor="centre"; `colour` is in the image's channel order. Clipped
    to the frame.
    """
    h, w = mask.shape
    if anchor == "centre":
        x -= w // 2
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha * mask[y0 - y : y1 - y, x0 - x : x1 - x, None]
    region = image[y0:y1, x0:x1]
    region *= 1 - a
    region += a * np.asarray(colour, np.float32)


# The opening hook: one short line of text over the first seconds of a Reel (see
# docs/AGENT_INSTRUCTIONS.md, "What the account's numbers say"). Fully visible at frame 0,
# held, then faded; placed below Instagram's top bar and clear of its bottom/right buttons.
HOOK_HOLD = 2.5  # seconds fully visible
HOOK_FADE = 0.5  # seconds to fade out
HOOK_Y = 300  # top of the text block, px
HOOK_SIZE = 62  # px per em
HOOK_COLOUR = "#f4f0e7"
HOOK_MAX_CHARS = 28  # per line; longer hooks wrap onto a second line


def hook_masks(text: str, size: float = HOOK_SIZE) -> list[np.ndarray]:
    """The hook as LaTeX line masks, wrapped at word boundaries ("\\n" forces a break)."""
    lines = []
    for chunk in text.split("\n"):
        line = ""
        for word in chunk.split():
            if line and len(line) + 1 + len(word) > HOOK_MAX_CHARS:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)

    def tex(line):
        escaped = line.replace(" ", r"\ ").replace(",", "{,}").replace("'", r"\text{'}")
        escaped = escaped.replace("%", r"\%").replace("-", r"\text{-}")  # hyphen, not minus
        return rf"$\mathrm{{{escaped}}}$"

    return [latex_mask(tex(line), size) for line in lines]


def hook_alpha(frame: int, fps: int = FPS) -> float:
    """Opacity of the hook at `frame`: 1 while held, then a smoothstep fade to 0."""
    t = frame / fps
    if t <= HOOK_HOLD:
        return 1.0
    return float(1 - smoothstep(np.clip((t - HOOK_HOLD) / HOOK_FADE, 0, 1)))


def hook_frames(fps: int = FPS) -> int:
    return int(np.ceil((HOOK_HOLD + HOOK_FADE) * fps))


def draw_hook(image: np.ndarray, masks, alpha: float, y: int = HOOK_Y, colour=HOOK_COLOUR):
    """
    Composite the hook onto a uint8 BGR frame and return it. A soft halo of the opposite
    tone under the letters (dark under light ink, light under dark ink) keeps them legible.
    """
    if alpha <= 0:
        return image
    out = image.astype(np.float32)
    bgr = hex_to_bgr(colour) if isinstance(colour, str) else colour
    halo_bgr = (0, 0, 0) if sum(bgr) > 3 * 128 else (255, 255, 255)
    top = y
    for mask in masks:
        halo = cv2.GaussianBlur(np.pad(mask, 14), (0, 0), 7)
        halo = np.clip(halo * 2.2, 0, 1)
        blit_mask(out, halo, image.shape[1] // 2, top - 14, halo_bgr, 0.6 * alpha, "centre")
        blit_mask(out, mask, image.shape[1] // 2, top, bgr, alpha, "centre")
        top += int(mask.shape[0] * 1.25)
    return np.clip(out, 0, 255).astype(np.uint8)


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
    hook: str | None = None,
    hook_y: int = HOOK_Y,
    hook_colour: str = HOOK_COLOUR,
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

    `hook` is the opening line (the script's HOOK constant): drawn over the first
    HOOK_HOLD + HOOK_FADE seconds at `hook_y` in `hook_colour`, after `render`, so the
    maths never sees it. `tools.add_hook --place` picks y and colour from the frames.

    In an `--audio-only` run nothing is rendered; the frames on disk are kept.
    """
    if os.environ.get("RP_NO_HOOK"):  # clean frames for tools.add_hook --place
        hook = None
    masks = hook_masks(hook) if hook else None
    n_hook = hook_frames() if hook else 0
    render_raw, rgb_raw = render, rgb

    def render(frame):  # noqa: F811 -- the scene, then the hook on top
        image = _to_bgr_uint8(render_raw(frame), rgb_raw)
        if frame < n_hook:
            image = draw_hook(image, masks, hook_alpha(frame), hook_y, hook_colour)
        return image

    rgb = False  # already converted above
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
