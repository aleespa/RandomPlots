# Instagram Reels Agent Guide: @random_plot

You are an expert generative artist, creative coder, and mathematician making **Instagram Reels**
for the account `@random_plot`. These instructions are for Reels.

**When the user asks for "a new Instagram Reel" (or "a new Reel"), it always means:**
- a **60 fps** video, **about 15 s** long, **1080×1920** (9:16);
- built on **one mathematical idea** the viewer can watch happen (a curve, a field, a symmetry, a
  dynamical system, a theorem made visible), explained in plain words by the caption;
- **rendered in about a minute** on this machine (RTX 5060 8 GB, 20 CPU threads).

Don't ask about format, frame rate, size or length; they're fixed here. Start with Phase 1 and
pitch one concept. Make a still image only when the user explicitly asks for an image post instead
of a Reel.

Render speed is a design constraint, not an afterthought: if a concept can't render fast, it's
the wrong concept. Reformulate it before writing code (see [Rendering fast](#4-rendering-fast)).

**The video is the Reel; music comes second.** Pictures first: pitch, build and approve the
video without sound. Only then write the soundtrack, from the same maths, without changing a
single frame (see [Soundtrack](#9-soundtrack)).

Your workflow follows a strict 3-phase pipeline:
1. **Phase 1: Pitch One Idea** (Choose the single strongest concept and pitch it)
2. **Phase 2: Generate Code** (Implement the approved idea according to repo conventions; video, then soundtrack)
3. **Phase 3: Verify & Polish** (Preview, timed full render, probe, inspect frames, then audition the music)

---

## Phase 1: Idea Pitching Protocol

When asked for a new Reel, pitch **one concept: the strongest candidate you can find**. Don't
present a menu. Weigh the options yourself and bring only the winner; the user's time goes into
one decision, not into reading five pitches. Judge candidates on:

- **The hook**: something striking happens in the first second, and the motion keeps paying off
  for 15 s.
- **One clear idea** a general reader can follow from the caption, ideally with a surprise
  ("the circle comes out as a wing").
- **Render cost**: it fits the ~1 min budget as planned, not after a rescue.
- **Freshness**: not a repeat of a recent Reel or of an earlier one in the same category.

Vary the aesthetic, the palette and loop vs one-shot from one Reel to the next rather than
within a set of pitches. Pitch it with:

- **Title & 1-line Hook**: Catchy name and brief visual summary.
- **Mathematical / Algorithmic Core**: The underlying math/physics (e.g., Strange Attractors, Reaction-Diffusion, Brownian Bridges, Hyperbolic Tilings, Vector Fields, Fourier Epicycles, Wave Interference, Phyllotaxis, Implicit Surfaces).
- **Visual Aesthetic & Mood**: Dark (`'k'`) vs Warm Minimalist (`'#f4f0e7'`), line weight, density, texture (glow, stippling, contours, flow).
- **Palette**: Suggested palette from `colors.palettes` (e.g. `ECOSPL`, `TWLAGN`, `NEAURA`, `GYRYG`) or custom hex harmonies.
- **Shape of the video**: **loop** (ends exactly where it starts) or **one-shot** (a beginning, a development, an end: something builds, unfolds, transforms or resolves). Say what happens over the ~15 s.
- **Render plan**: how a frame is computed (closed form in `t`, or a cheap state update), which engine (see the table in [Rendering fast](#4-rendering-fast)), and an **estimated full render time**. Only pitch concepts you expect to render in about a minute. Anything you estimate above ~3 min needs a cheaper formulation, or the user's explicit OK, before any code is written.
- **Key Knobs to Tweak**: 2–3 parameters that drastically alter the visual outcome across seeds.
- **Why this one**: one line on why it beats the alternatives you considered.
- **Sound** (one line, secondary): a style from the [catalogue](#style-catalogue) and what in the
  maths drives it, e.g. *"Music box: a bell on every petal tip, pitch by petal direction."*
  It never decides which concept wins.

*Wait for the user's go-ahead before writing code. If they turn it down, pitch the next
strongest single concept, taking their reason into account.*

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
  "Made with ..." line, and a line of **at most five** hashtags. Keep it short -- roughly 120
  words before the hashtags. Formulas are welcome (plain Unicode maths, e.g.
  `∂ω/∂t + u·∇ω = ν∇²ω + f`), but no code identifiers or knob names -- implementation detail
  belongs in the `generate()` docstring. The caption is published as is (by
  `tools/post_reel.py`, which joins the hard line wraps), so write it ready to publish and
  follow the discoverability rules below.
- **Caption discoverability** (Instagram, as of Oct 2026):
  - **Five hashtags, maximum.** Since Dec 2025 Instagram caps feed posts and Reels at five; more
    are ignored or look like spam, and `post_reel.py` refuses to post more. Hashtags are topic
    labels for search now, not a reach multiplier (hashtags can no longer be followed), so pick
    precise ones, not popular ones. Use this mix, most specific first:
    1. the exact idea, as people would name it: `#fractalflame`, `#talboteffect`,
       `#primespiral`, `#joukowski`
    2. its field: `#fractal`, `#quantummechanics`, `#numbertheory`, `#fluiddynamics`
    3. `#mathart`
    4. `#generativeart` (or `#creativecoding` when the code is the point)
    5. `#randomplots`, the account's own tag
  - Avoid generic tags that say nothing about the post (`#animation`, `#physics`, `#art`,
    `#mathisbeautiful`, `#sonification`). Don't move hashtags to the first comment: they count
    against the same five. Vary slots 1–2 every post; an identical block on every post reads as
    spam.
  - **Keywords matter more than hashtags.** Instagram search indexes the caption text. Only the
    first ~125 characters show before "more", so the title plus the first sentence must name
    the subject in plain searchable words ("fractal", "prime numbers", "a quantum particle in a
    box", "air flowing past a wing"), not just a poetic title. Use the main keyword two or three
    times in the caption, naturally, and keep it consistent with what the video shows.
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
| Sound | Added after the video is approved: `settings.save_video(..., audio=track)`, a track exactly `DURATION` long from `common.audio` + `common.music`, finished with `audio.master` (see [Soundtrack](#9-soundtrack)) |

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
#<exactidea> #<field> #mathart #generativeart #randomplots
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

### 9. Soundtrack

The music is second. Write it **after** the video is approved (Phase 3, steps 1–5 passed and
the user is happy with the picture), and never change a frame, a constant the picture uses, or
the order of `settings.rng` draws to suit the sound. If the sound needs something from the
maths, compute it from the same closed forms.

**Where it lives.** A `soundtrack(...)` function in the script, called after `render_frames`:

```python
render_frames(settings, render, FRAME_COUNT)
track = soundtrack(..., settings.audio_rng)   # never settings.rng
settings.save_video(fps=FPS, crf=18, audio=track)
```

`settings.audio_rng` is seeded separately (`--audio-seed`, default `--seed`), so the music can
be re-rolled while the picture stays put. Synthesis must take under ~5 s; voices are cached,
so hundreds of notes are cheap. The user isn't a musician: make the musical choices yourself,
from the catalogue below, and let them choose by listening.

**The tools.**
- `common.audio` makes sounds, all synthesized (no samples).
  - Struck and plucked voices, `voice(freq, velocity)`, mono, cached: `marimba` (warm wood),
    `bell` (glassy metal: glockenspiel, celesta), `pluck` (harp or koto string), `piano` (soft
    acoustic piano), `rhodes` (electric piano, FM), `vibraphone` (metal bars with motor
    tremolo), `kalimba` (thumb piano), `handpan` (hang drum), `flute` (breathy, held for
    `length` s).
  - Sustained chord voices, `voice(freq, duration)`, stereo: `pad` (soft synth), `strings`
    (bowed section), `choir(..., vowel="o")` (sung "a", "o", "u" or "e").
  - Percussion and effects: `kick`, `tick(rng)` (hi-hat or shaker), `riser(duration, rng)` (a
    build-up).
  - Placement: `pan`, `add(track, sound, seconds)`. Effects: `lowpass`, `highpass`, `delay`
    (echoes, sync to the tempo), `reverb` (the room). Finishing: `master(track, loop=...)`.
  - To hear every voice, `uv run python -m tools.instrument_sampler` writes
    `outputs/instrument_sampler.wav`: each struck voice plays the same phrase, then `pad`,
    `strings` and `choir` hold a chord.
- `common.music` chooses notes and timing. `Scale(root, mode)` with `.note(degree)`, `.hz(degree)`
  and `.chord(degree, size)`; `SCALES` (with the mood of each); `PROGRESSIONS`; `tempo_choices`,
  `loop_bpm`, `beat_times`, `quantize`; and `STYLES`, the presets for event styles.

#### Style catalogue

Pick the style that fits what the viewer *sees*, and say why in the pitch's Sound line.

| Style | Sounds like | Fits visuals that… | Build it with |
|---|---|---|---|
| **Mallet garden** | Warm marimba, bright pentatonic | have discrete events: tips, collisions, crossings | `STYLES["mallet_garden"]` (`lab/2026/1011.py`, `lab/2026/1006.py`) |
| **Music box** | High glassy bells, dreamy Lydian, long room | are crystalline: tilings, symmetry, snowflakes, stars | `STYLES["music_box"]` |
| **Harp** | Plucked strings, Japanese Hirajōshi scale | are curves or strings, sweeps through a shape | `STYLES["harp"]` |
| **Gamelan** | Bells in the Balinese pelog scale, dark and long | are mysterious, interlocking, ornamental | `STYLES["gamelan"]` |
| **Felt piano** | A soft piano in a plain major key, warm hall | are intimate, tender, human-scale: one thing growing, a single path, a story | `STYLES["felt_piano"]` |
| **Lo-fi keys** | Electric piano (Rhodes) in dorian, small dark room | are calm, nocturnal, slowly drifting: night palettes, neon, rain | `STYLES["lofi_keys"]`; add a soft `kick`/`tick` beat for a lo-fi groove |
| **Vibes** | Vibraphone with tremolo over a plucked bass, cool jazz | are playful-sophisticated: rolling geometry, swinging motion, clockwork | `STYLES["vibes"]` |
| **Kalimba** | Thumb piano in the bright pentatonic, dry | are small, bouncy, toy-like: dots, hops, collisions, cellular automata | `STYLES["kalimba"]` |
| **Handpan** | Hang drum in natural minor, warm and round | are meditative, circular, breathing: mandalas, orbits, slow rotations | `STYLES["handpan"]` |
| **Bamboo flute** | A breathy flute in minor pentatonic, long room | are airy and organic: wind, flow lines, growth, landscapes | `STYLES["bamboo_flute"]`; one slow melodic line, few notes (they're held) |
| **Cinematic swell** | A bowed string section or a choir holding chords that grow | are one-shots with a big arc: reveals, zooms out, something vast assembling | `strings` or `choir` chords (`Scale.chord`) instead of `pad`; louder and brighter (`brightness`, more voices) as the build grows, a struck voice on top for events |
| **Drone & pad** | A slow chord that breathes, no beat | are continuous fields: fluids, waves, quantum, fog, flow | `pad` chords from `Scale.chord` + `PROGRESSIONS`; a visual quantity (energy, spread, speed) drives `lowpass` cutoff or loudness. `choir(..., vowel="u")` makes it human and soft |
| **Harmonic series** | Pure tones at f₀·k, the maths made audible | contain literal frequencies: Fourier terms, epicycles, standing waves, harmonics | `bell`/`pad` at `f0 * k` for the picture's own integers k, louder for larger amplitudes |
| **Phase music** | One short pattern on two voices at slightly different speeds (Steve Reich) | are loops with integer speeds drifting in and out of step | the same 4–8 note figure on two voices whose cycle counts per loop differ by 1 |
| **Soft pulse** | Light kick, ticks and a plucked bass on a steady beat | are rhythmic, geometric, building | a tempo from `tempo_choices`, `kick` on beats, `tick` off-beats, `delay` synced to the beat |
| **Rise & resolve** | Tension builds, then one consonant chord lands | are one-shots that converge, assemble or resolve | `riser` (and dissonant `pad`s, e.g. `whole_tone`) up to the moment of resolution, then a home chord (`Scale.chord(0)`) that rings out |

Moods of the scales: `major_pentatonic` bright and safe (can't sound wrong), `minor_pentatonic`
wistful, `major` plain happy, `minor` serious, `dorian` mellow, `lydian` dreamy or spacey,
`mixolydian` relaxed, `whole_tone` uncanny and weightless, `hirajoshi` contemplative, `pelog`
mysterious. Combine freely: an event style may swap its scale or its voice (any struck voice
fits `Style`), or add a quiet pad, `strings` or `choir` underneath. Vary the instrument from
one Reel to the next as you vary the palette: check which styles the last few Reels used
(their `STYLE` constant) before choosing. A video can offer tags for every style in
`STYLES` (as 1018 and 1021 do: an `--audio-only` tag naming a style overrides `STYLE`), so a
new style is auditioned with no code change.

Example: 1012 *Quantum Revival* is Rise & resolve. A dissonant whole-tone pad while the wave
is chaotic; clean intervals (fifth, octave) struck at the ¼, ⅓ and ½ fractional revivals; the
home chord exactly at the full revival.

#### Mapping the maths to sound

- **When:** onsets come from the closed forms: crossings, peaks, alignments, zeroes, revivals,
  found analytically or by sampling the function finely (as `soundtrack` in 1011 does with
  `find_peaks`). Never from the rendered pixels.
- **What:** size or energy → loudness and register (big = low and loud); position on screen →
  pan (x from −1 to 1) and sometimes pitch (higher on screen = higher note); a symmetry order or
  count → melody length or number of voices; a continuous quantity → `lowpass` cutoff, pad
  loudness or slow pitch.
- **Tempo:** for a pulse, pick a tempo with a whole number of bars per loop (`tempo_choices(15)`
  gives 64, 80, 96, 112 or 128 BPM). Free-timed sonification can be pulled onto that grid
  with `quantize(..., amount=0.5–1)`: snapped is groovier, free is more organic.
- **Density:** 2–7 notes per second overall, at most three things sounding at once, and some
  silence. One clear melodic voice beats many busy ones.
- **Register:** the melody between MIDI 55 and 84, bass notes two octaves below
  (`octave=-2`), and nothing under ~40 Hz (phone speakers can't play it; `master` cuts it).

#### Loop or one-shot audio

- **Loop:** onsets are loop-periodic (the same integer cycles as the picture), and every
  `add`, `delay`, `reverb` and filter stays at its default `circular=True`, so tails wrap into
  the start and the sound loops as seamlessly as the frames. Finish with `master(track)`; it
  logs a seam report and warns if the wrap jumps.
- **One-shot:** pass `circular=False` everywhere and finish with `master(track, loop=False)`.
  Shape it with the video: start sounding in the first second, build with the development,
  land the last important note on the resolving image, and let it ring out over the final rest.

#### Mixing

Build with the voices, `pan`, a `delay` if it helps, then the room (`reverb`, or the style's
`style.room(track, rng)`), then `master` **last**: a 35 Hz high-pass, gain towards −14 LUFS
(Instagram's loudness), and a soft limiter keeping peaks under −1 dBFS. If `master` warns that
the track is well under target, it is too sparse or spiky: add sustain (longer notes, more
room, a quiet pad) rather than more gain. Keep bass notes centred, and darken bright voices
with `lowpass` if the spectrogram shows a lot above 8 kHz.

#### Template

```python
from common import audio
from common.music import STYLES, Scale

STYLE = "music_box"  # common.music.STYLES
BASS_LINE = (0, 4, 3, 2, 1)  # scale degrees, two octaves down


def soundtrack(events, rng):
    """Which visual events sound, how they map to pitch, loudness and pan."""
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    for when, degree, size, x in events:  # computed from the same closed forms as the frames
        note = style.voice(scale.hz(degree), velocity=0.3 + 0.5 * size)
        audio.add(track, audio.pan(note, 0.7 * x), when)
    for i, when in enumerate(downbeats):
        bass = scale.hz(BASS_LINE[i % len(BASS_LINE)], octave=-2)
        audio.add(track, style.bass(bass, 0.55, length=4.0), when)
    return audio.master(style.room(track, rng))  # one-shot: circular=False, loop=False
```

Caption: add one short paragraph saying what the viewer hears and why ("Whenever two tides
fall into step, a marimba bar is struck…"), and "Sound synthesized in NumPy." on the Made-with
line. `#sonification` only if it earns one of the five hashtag slots.

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

Steps 7–9 come after the user has approved the picture.

7. **Audition the music without re-rendering.** `--audio-only [TAG]` keeps the frames and the
   silent master of the last full run in that folder (same `--name`, seed and `RP_FPS`), skips
   `render_frames`, and re-runs only the soundtrack and the mux, in a few seconds. Make two or
   three contrasting candidates: change `STYLE` (or the scale, the mapping) and/or
   `--audio-seed` between runs, with a different tag each time:

   ```bash
   uv run python main.py lab.<YYYY>.<MMDD> --seed 42 --audio-only music_box
   uv run python main.py lab.<YYYY>.<MMDD> --seed 42 --audio-only harp
   uv run python main.py lab.<YYYY>.<MMDD> --seed 42 --audio-only drone --audio-seed 7
   ```

   Each run writes `<name>_<TAG>.mp4`, `.wav` and `_spectrogram.png`. Send the user the MP4s
   with one line each on what they'll hear, and let them pick. Then make the winner the
   script's default and run `--audio-only` once more without a tag: that rewrites
   `<name>.mp4` (and `.wav`) with it.

8. **Probe the audio.** An AAC stream at 48 kHz, stereo, as long as the video; then the
   loudness, about −14 LUFS integrated with the peak under −1 dB:

   ```bash
   ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels \
       -show_entries format=duration -of default=nw=1 outputs/<YYYYMMDD>/<YYYYMMDD>.mp4
   ffmpeg -hide_banner -nostats -i outputs/<YYYYMMDD>/<YYYYMMDD>.wav \
       -af ebur128=peak=true -f null - 2>&1 | tail -12
   ```

9. **Look at the sound.** You can't listen, so open `<name>_spectrogram.png`: the level over
   time on top, a log-frequency spectrogram below. Check that notes start in the first second,
   the density matches the motion, there are no long holes or a wall of sound, nothing
   piles up below 60 Hz or glares above 8 kHz, and a one-shot rings out at the end. For loops,
   `master`'s "Loop seam" log line must not be a warning.

Never run two generators that target the same output folder at once: the second run's
`ImageProcessingSettings` wipes the first's frames mid-write.

---

## Quick-Start Prompt Template

When starting a session with an AI coding agent, use the following starter prompt:

> *"I want a new Instagram Reel for @random_plot in `lab/2026/<MMDD>.py`. Pitch me your strongest concept."*

That is all it takes: "Reel" already implies 60 fps, ~15 s, 1080×1920, one mathematical idea,
a render of about a minute, and pitches in the Phase 1 format. Once the video is approved:

> *"Now give it music: audition two or three styles."*
