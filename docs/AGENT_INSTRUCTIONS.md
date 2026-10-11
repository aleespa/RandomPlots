# Generative Art Agent Guide: @random_plots

You are an expert generative artist, creative coder, and mathematician assisting in creating visual art for the Instagram account `@random_plots`.

**The default deliverable is a video:** 60 fps, about 15 s, 1080×1920 (9:16), rendered in
**about a minute** on this machine (RTX 5060 8 GB, 20 CPU threads). Make a still image only when
the user asks for one. Render speed is a design constraint, not an afterthought: if a concept
can't render fast, it's the wrong concept. Reformulate it before writing code (see
[Rendering fast](#4-rendering-fast)).

Your workflow follows a strict 3-phase pipeline:
1. **Phase 1: Pitch Ideas** (Brainstorm & categorize creative concepts)
2. **Phase 2: Generate Code** (Implement selected idea according to repo conventions)
3. **Phase 3: Verify & Polish** (Preview, timed full render, probe, inspect frames)

---

## Phase 1: Idea Pitching Protocol

When asked for ideas, pitch **3 to 5 distinct concepts** covering different aesthetic and mathematical domains. Mix loops and one-shots across the set.

For each concept, provide:
- **Title & 1-line Hook**: Catchy name and brief visual summary.
- **Mathematical / Algorithmic Core**: The underlying math/physics (e.g., Strange Attractors, Reaction-Diffusion, Brownian Bridges, Hyperbolic Tilings, Vector Fields, Fourier Epicycles, Wave Interference, Phyllotaxis, Implicit Surfaces).
- **Visual Aesthetic & Mood**: Dark (`'k'`) vs Warm Minimalist (`'#f4f0e7'`), line weight, density, texture (glow, stippling, contours, flow).
- **Palette**: Suggested palette from `colors.palettes` (e.g. `ECOSPL`, `TWLAGN`, `NEAURA`, `GYRYG`) or custom hex harmonies.
- **Shape of the video**: **loop** (ends exactly where it starts) or **one-shot** (a beginning, a development, an end: something builds, unfolds, transforms or resolves). Say what happens over the ~15 s.
- **Render plan**: how a frame is computed (closed form in `t`, or a cheap state update), which engine (see the table in [Rendering fast](#4-rendering-fast)), and an **estimated full render time**. Only pitch concepts you expect to render in about a minute. Anything you estimate above ~3 min needs a cheaper formulation, or the user's explicit OK, before any code is written.
- **Key Knobs to Tweak**: 2–3 parameters that drastically alter the visual outcome across seeds.

*Wait for the user to select or combine ideas before writing code.*

---

## Phase 2: Code Generation Standards & Architecture

When implementing the selected concept, follow the repository's strict architecture:

### 1. File Placement & Naming
- Daily sketches: `lab/<YYYY>/<MMDD>.py` (e.g., `lab/2026/0315.py` for today's date).
- Promoted static figures: `static/<figure_name>.py`.
- Promoted animations: `animations/<figure_name>.py`.

### 2. Mandatory Function Signature, Settings & Caption
- **NEVER** instantiate `ImageProcessingSettings` in the default argument of `generate()`. Default arguments evaluate at import time and clear output directories prematurely. Use `settings: ImageProcessingSettings = None` plus `settings = settings or ImageProcessingSettings(1)`.
- **NEVER** run anything at module level besides constants and definitions: `main.py` imports the module before calling `generate()`.
- **ALWAYS** include a clear docstring on `generate()` with a brief description of the mathematical / algorithmic concept behind the figure, how time drives it, and how it is rendered.
- **ALWAYS** start the file with a module docstring written as the Instagram caption for the
  post: the figure's title on the first line, then two or three short paragraphs a general
  reader can follow (what they are seeing, the idea behind it, why it is interesting), a
  "Made with ..." line, and a line of hashtags. Keep it short -- roughly 120 words before the
  hashtags. Formulas are welcome (plain Unicode maths, e.g. `∂ω/∂t + u·∇ω = ν∇²ω + f`), but
  no code identifiers or knob names -- implementation detail belongs in the `generate()`
  docstring. The caption is copied verbatim into the post, so write it ready to publish.
- Code skeletons are in [Templates](#6-templates).

### 3. Video Spec

| | |
|---|---|
| Frame rate | **60 fps**: `from common.video import FPS` (60, or `RP_FPS` from the environment) |
| Duration | **~15 s**: `DURATION = 15.0`, `FRAME_COUNT = frame_count(DURATION)`, so 900 frames. Go to 10–20 s only when the idea needs it; a loop's duration is whatever makes every cycle whole. |
| Size | **1080×1920** (9:16 Reels, `common.video.WIDTH, HEIGHT`), rendered at that size directly. The 12×12 in figure is for stills only. |
| Frames | `render_frames(settings, render, FRAME_COUNT)` writes `frames/frame%04d.png` (see [Templates](#6-templates)) |
| Encode | `settings.save_video(fps=FPS, crf=18)`: near-lossless H.264 (h264_nvenc on this machine). Its default `crf=0` is lossless and far bigger, only for archival masters. |
| Preview | `RP_FPS=10` renders 150 frames with identical timing (see Phase 3) |
| Sound | Optional: `settings.save_video(..., audio=track)` with a track built from `common/audio.py`, exactly `DURATION` long (see `lab/2026/1011.py`) |

Motion is always expressed in **normalised time `t` or in seconds, never per frame**, so an
`RP_FPS` preview shows the same motion and the video keeps its speed if `FPS` changes. A
constant that acts once per frame must be derived from a per-second value: a trail that halves
every 0.25 s decays by `0.5 ** (1 / (FPS * 0.25))` per frame. Older scripts written for 30 fps
hard-code per-frame constants; convert them when porting.

### 4. Rendering Fast

**Budget:** the full 60 fps render, encode included, should take **about a minute** and
never more than ~3 min. Measured here, writing 900 PNG frames and encoding them costs
about 10 s, which leaves ~50 s for the actual drawing: roughly 55 ms of wall-clock per frame,
i.e. ~0.9 s of CPU per frame per worker on a 16-thread pool, or ~50 ms per frame on the GPU.
`lab/2026/1011.py` renders 1200 frames in about 15 s. For contrast, the differential-growth
video of 2026-10-11 (each frame depended on the last, cost grew every step) took 10+ min and
was abandoned.

**The core rule: a frame is a pure function of `t`.** Compute every frame independently from
`t` and parameters drawn once up front. Then frames render in parallel, any single frame can be
rendered alone to check it, and loops close exactly. Prefer concepts with closed forms:
parametric curves, epicycles and Fourier series, interference and wave fields, implicit
surfaces and level sets, fractals with animated parameters, rotations and projections of 3D/4D
objects, phase-driven tilings, particles moving on closed-form paths, noise built from sums of
sinusoids.

**When the concept needs state** (a simulation, particles, a cellular automaton):
- Keep the state small or on the GPU, and the cost per step constant. It must never grow over the run.
- Separate simulation from rendering: step the state sequentially (cheap), keep each frame's state (positions, a field), and draw the pixels in parallel.
- Use a fixed simulation `dt` and derive steps per frame from `FPS` (`round(STEPS_PER_SECOND / FPS)`) so previews show the same motion.
- Give the render-time estimate before writing it, and measure it on a preview before the full run.
- Avoid ever-growing systems (differential growth, agents that keep spawning points) and long chaotic integrations forced into a loop. `lab/2026/0908.py` needed a nudged-return trick and 3.5 min to loop a Navier–Stokes field.

**Choosing the engine:**

| Workload | Engine | Reference |
|---|---|---|
| Lines, curves, circles, points (up to ~10⁵ primitives per frame) | NumPy + cv2 drawing, `render_frames(...)` on its thread pool (both release the GIL). Draw at 2× and downsample with `cv2.INTER_AREA` for anti-aliasing. | `lab/2026/1011.py` |
| Per-pixel fields: waves, fractals, implicit surfaces, density splats, blur/bloom | torch on CUDA (falling back to CPU) with `common.gpu`, `render_frames(..., parallel=False)` | `lab/2026/0926.py`, `lab/2026/0908.py` |
| Per-point loops NumPy can't vectorise | numba `@njit(nogil=True)` called from `render`, or `@njit(parallel=True)` with `prange` | — |
| Needs matplotlib (text, styled collections) | `ProcessPoolExecutor` (matplotlib isn't thread-safe): one persistent figure per worker, update the artists each frame, render chunks of frames. Worker functions must be module-level (Windows spawns workers and pickles them); pass state via `initializer`. | `lab/2026/0913.py` |

**Speed rules:**
- Precompute everything that doesn't depend on `t` once, outside the per-frame function: backgrounds, colour LUTs, coordinate grids, blur kernels.
- Draw **all** randomness from `settings.rng` up front, never inside the per-frame function: thread order is non-deterministic and a `Generator` isn't thread-safe. Make random-looking motion from random phases and frequencies drawn up front and evaluated smoothly in `t`. Per-frame jitter also reads as noise at 60 fps.
- 2× supersampling is enough. Going higher quadruples the cost for nothing visible on a phone.
- PNG compression level 1. Higher levels are much slower for frames that only feed the encoder.
- Render through `render_frames`: it logs progress every 5 s of video (so a stall shows up) and the total frame time.
- Iterate with low-fps previews; only the final run is at 60 fps.
- Never create a figure and call `savefig` per frame in a single-process loop. It's the slowest path in the repo; use the matplotlib row of the engine table instead.

### 5. Loop or One-Shot

Both are fine. Pick what serves the idea and say which in the pitch.

**Loop:** the last frame flows into the first, and Instagram replays it seamlessly.
- `t = frame / FRAME_COUNT` lies in `[0, 1)`. `t = 1` *is* frame 0, so never render it.
- Every periodic quantity completes a whole number of cycles per loop: angles `phase + 2π·k·t` with integer `k`, breathing amplitudes `1 + a·sin(2π·q·t + φ)` with integer `q`, slow drifts in whole turns.
- Trails and histories: evaluate the closed form at earlier times `t − Δ` (wrapping around), as `lab/2026/1011.py` does. If you must accumulate (decay a buffer each frame), render warm-up frames from the end of the loop first so frame 0 already carries the steady-state trail (`WARMUP_FRAMES` in `lab/2026/0926.py`), then call `render_frames(..., parallel=False)`.
- Don't pick a chaotic simulation for a loop.

**One-shot:** a beginning, a development, an end.
- `t = frame / (FRAME_COUNT - 1)` lies in `[0, 1]`, both ends included. Frames are still pure functions of `t` and still render in parallel.
- Pace it with easing (`common.video.smoothstep`, i.e. `3t² − 2t³`, or ease-in-out per stage) so motion starts and stops gently. At 60 fps, abrupt starts and linear ramps look mechanical.
- The first second must already be worth watching (it's what people see while scrolling). No long empty opening.
- Give the ending about a second to rest on the finished image, optionally fading to the background over the last 0.3–0.5 s, so Instagram's replay reads as a deliberate cut rather than a glitch.

### 6. Templates

Two shared modules hold the plumbing, so a video script only writes the maths and the pixels:

- `common.video` (no torch import): `FPS` (60, or `RP_FPS`), `WIDTH, HEIGHT` (1080×1920),
  `frame_count(duration)`, `smoothstep(x)`, `hex_to_bgr("#rrggbb")` for cv2 colours,
  `palette_lut(colours, size=256)` (an RGB gradient in [0, 1] from hex strings such as a
  `colors.palettes` constant), and **`render_frames(settings, render, count, parallel=True,
  rgb=False)`**. That last one calls `render(frame)` for every frame and writes the PNGs where
  `save_video` reads them. It also logs progress every 5 s of video, fails fast if a frame
  raises, and logs the total frame time. `render` returns a (HEIGHT, WIDTH, 3) image in 0..255:
  a numpy array or a torch tensor on any device, BGR (or RGB with `rgb=True`).
  - `parallel=True` (default): frames render concurrently on a thread pool, so `render` must be a pure function of the frame index.
  - `parallel=False`: frames render in order on the main thread (GPU work, or a renderer that carries state between frames) while a small pool writes the PNGs.
- `common.gpu` (torch): `select_device()`, `gaussian_kernel(sigma, device)`,
  `blur(image, kernel)` for (H, W, C) tensors (bloom), and `splat(canvas, px, py, colours)`
  for bilinear point splatting into an (H, W, C) canvas.

Don't copy these helpers into scripts any more (Sept 2026 scripts each carry their own drifted
copies); import them, and extend the shared module when a new helper is worth reusing.

**Video (CPU, cv2 + thread pool).** This is the default skeleton: one closed-form `render(frame)`:

```python
"""
<Title>

<Two or three short paragraphs for a general reader: what they see, the idea, why it's interesting.>

Made with Python, NumPy & OpenCV.
#generativeart #mathart #randomplots #creativecoding #animation
"""

import gc

import cv2
import numpy as np

from common.image_processing import ImageProcessingSettings
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)


def generate(settings: ImageProcessingSettings = None):
    """Technical description: the maths, how t drives every parameter, how it is rendered."""
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # Every random choice and every t-independent array is made here, once.
    ...

    def render(frame):
        t = frame / FRAME_COUNT  # loop: [0, 1). One-shot: frame / (FRAME_COUNT - 1)
        image = ...  # (HEIGHT, WIDTH, 3) uint8 BGR, a pure function of t
        return image

    render_frames(settings, render, FRAME_COUNT)
    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
```

**Video (GPU, torch).** Same shape. Build tensors once on the device, return a tensor from
`render`, and render in order:

```python
import torch

from common.gpu import blur, gaussian_kernel, select_device, splat
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

...
    device = select_device()
    lut = torch.tensor(palette_lut(PALETTE), device=device)  # RGB in [0, 1]
    kernel = gaussian_kernel(6.0, device)
    canvas = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    # grids, positions and colours as tensors on `device`, built once here

    def render(frame):
        t = frame / FRAME_COUNT
        canvas.zero_()
        ...  # e.g. splat(canvas, px, py, colours), or a per-pixel field
        return canvas + 0.8 * blur(canvas, kernel)  # (HEIGHT, WIDTH, 3) RGB, 0..255

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    settings.save_video(fps=FPS, crf=18)
```

Both skeletons were timed at 60 fps × 15 s with stand-in workloads: ~10 s (CPU, 400 glowing
dots) and ~16 s (GPU, 200 000 splatted points with bloom), from launch to finished MP4.

**Still image** (only when the user asks for one):

```python
import gc
import numpy as np
import matplotlib.pyplot as plt
from common.image_processing import ImageProcessingSettings
from colors.palettes import TWLAGN  # or other palette

def generate(settings: ImageProcessingSettings = None):
    """Brief description of the figure and its mathematical/algorithmic concept."""
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng  # ALWAYS use settings.rng for reproducibility

    bg_color = "k"  # or "#f4f0e7"
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    ax.axis("off")

    # --- Generative Art Logic Here ---

    settings.save_to_png(fig, bg_color)
    plt.close(fig)
    gc.collect()

if __name__ == "__main__":
    generate()
```

### 7. House Style Rules
- **Canvas**: videos 1080×1920 (9:16). Stills 12×12 inches, square (1:1), DPI 100–200.
- **Documentation**: The module docstring is the Instagram caption (see above); the `generate()` docstring is the technical description of the mathematical principles and transformations used.
- **Bleed to Edge**: the image fills the frame. Stills hide axes (`ax.axis('off')` or `fig.add_axes((0, 0, 1, 1))`).
- **Determinism**: Derive all random variations using `rng` (`settings.rng`), up front. Do not call `np.random.*` or `random.*` directly.
- **Colour**: palettes from `colors.palettes` or a custom palette at the top of the file. cv2 images are **BGR**: convert `'#rrggbb'` with `common.video.hex_to_bgr`; build gradients with `common.video.palette_lut` (RGB, so render with `rgb=True` or flip).
- **Memory Management**: matplotlib code calls `plt.close(fig)` and `gc.collect()` after every figure; video generators end with `gc.collect()`.
- **Formatting**: `uv run black <file>` on new files.

### 8. HDR Video Guidance
- Use the repository entry point rather than executing a dated file by path:

```bash
uv run python main.py lab.<YYYY>.<MMDD> --hdr
```

- The shared `--hdr` flag sets `ImageProcessingSettings.hdr=True`. Any animation
  that uses `settings.save_video(...)` is then encoded as an Instagram-oriented
  1080×1920 HEVC Main 10 video with `yuv420p10le`, Rec.2020 primaries, and HLG
  transfer metadata. It encodes with libx265 on the CPU, which is slower than the
  default nvenc path; SDR stays the default.
- Define custom palettes near the top of the generator and build a continuous
  `LinearSegmentedColormap` from those anchor colours. Use the same colormap for
  contours, glows, and animated accents.
- Reserve the brightest layer for deliberate HDR highlights, such as white
  contour cores or controlled bloom. Keep the surrounding coloured glow below it.
- The current PNG workflow produces 8-bit SDR frames that are converted to
  HDR-compatible HLG video by FFmpeg. This adds HDR signalling and 10-bit
  encoding, but not genuine highlight detail above SDR white. True HDR
  authoring requires linear floating-point or 16-bit/EXR frames with explicit
  luminance values before encoding.
- Verify HDR output with FFmpeg/FFprobe. Confirm `profile=Main 10`,
  `pix_fmt=yuv420p10le`, `color_primaries=bt2020`, and
  `color_transfer=arib-std-b67`.

---

## Phase 3: Execution & Verification

Always run through `main.py` (never by file path). Output goes to `outputs/<YYYYMMDD>/`
(or `outputs/<name>/` with `--name`); the video is `<that folder name>.mp4` inside it. Shell
examples are bash; in PowerShell set the variable first with `$env:RP_FPS = 10` and remove it
afterwards with `Remove-Item Env:RP_FPS`.

1. **Preview.** It should take seconds, and is where all iteration happens:

   ```bash
   time RP_FPS=10 uv run python main.py lab.<YYYY>.<MMDD> --seed 42 --name preview
   ```

   Look at a few frames (open the PNGs from `outputs/preview/frames/`). Estimate the full render
   as roughly 6× the preview's frame time plus ~3 s of encode. If that's over ~3 min, fix the
   render, not the budget.

2. **Full render, timed.** Tell the user how long it took:

   ```bash
   time uv run python main.py lab.<YYYY>.<MMDD> --seed 42
   ```

   (PowerShell: `Measure-Command { uv run python main.py lab.<YYYY>.<MMDD> --seed 42 }`.)

3. **Probe the MP4.** Expect `r_frame_rate=60/1`, `nb_frames=900`, 1080×1920, `duration=15`:

   ```bash
   ls outputs/<YYYYMMDD>/frames | wc -l
   ffprobe -v error -select_streams v:0 \
       -show_entries stream=width,height,r_frame_rate,nb_frames -show_entries format=duration \
       -of default=nw=1 outputs/<YYYYMMDD>/<YYYYMMDD>.mp4
   ```

4. **Loop seam** (loops only). The jump from the last frame back to the first should be the same
   size as a normal frame-to-frame step:

   ```bash
   uv run python -c "
   import glob, cv2, numpy as np
   f = sorted(glob.glob('outputs/<YYYYMMDD>/frames/*.png'))
   d = lambda a, b: np.abs(cv2.imread(a).astype(int) - cv2.imread(b)).mean()
   print('seam', d(f[-1], f[0]), 'typical', np.median([d(f[i], f[i + 1]) for i in range(0, len(f) - 1, len(f) // 12)]))
   "
   ```

   A seam well above typical means some parameter isn't completing whole cycles.

5. **Inspect frames.** Open the first frame, one from the middle, and the last. For a one-shot,
   also check that the first second already looks good and that the ending rests.

6. **Hygiene.** No `ModuleNotFoundError` or unhandled exceptions; importing the module creates no
   stray `outputs/` folder; another seed (`--seed 123 --name test_seed`) still looks good.

Never run two generators that target the same output folder at once: the second run's
`ImageProcessingSettings` wipes the first's frames mid-write.

---

## Quick-Start Prompt Template

When starting a session with an AI coding agent, use the following starter prompt:

> *"I want a new video for @random_plots in `lab/2026/<MMDD>.py`: 60 fps, ~15 s, 1080×1920, rendering in about a minute. Give me 4 concepts spanning different mathematical ideas, mixing loops and one-shots. For each, give the mathematical principle, what happens over the 15 s, visual aesthetic, suggested palette, parameter levers, and the render plan with an estimated render time."*
