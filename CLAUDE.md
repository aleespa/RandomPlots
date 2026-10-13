# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Generative-art code for the Instagram account `@random_plot`. Every "figure" is a standalone
matplotlib script exposing a `generate(settings)` function that writes PNGs (and, for animations,
an MP4) into `outputs/`. There are no tests, no library API, and no packaging — the deliverable is
images.

## Generating new figures

When brainstorming or implementing new figures, follow the pitch → code → verify workflow, the
caption format, and the code templates in [`docs/AGENT_INSTRUCTIONS.md`](docs/AGENT_INSTRUCTIONS.md)
(it also covers HDR/looping guidance for animations). In short: every figure script starts with a
module docstring that is the ready-to-post Instagram caption (title, plain-language paragraphs,
"Made with" line, at most five specific hashtags; see the doc's "Caption discoverability"); the technical description goes in the `generate()` docstring. New work
goes in `lab/<YYYY>/<MMDD>.py`, dated.

**New figures are Instagram Reels. "A new Instagram Reel" (or "a new Reel") always means a 60 fps,
~15 s (900 frames), 1080×1920 video built on one mathematical idea, either a loop or a one-shot,
whose full render takes about a minute (3 min at most).** Don't ask about format, frame rate, size
or length. Start by pitching concepts (Phase 1 of the doc). Get there by making
each frame a pure function of time, rendered in parallel (cv2 on a thread pool, or torch on the
GPU), never with long sequential simulations. Pitch each video with its render plan and a time
estimate, iterate on `RP_FPS=10` previews, and report the full render time. The rules,
templates and verification steps are in the doc's "Video Spec", "Rendering Fast" and "Loop or
One-Shot" sections. Stills only when asked.

**Posting schedule.** `SCHEDULE.md` lists every planned Reel through 2026-12-31: date,
math category, script path, title and status. "The next one" / "next Reel" means the earliest
open slot there. Pitch only within that slot's category, and update the row's status as work
progresses. The `next-reel` skill (`.claude/skills/next-reel/`) has the steps.

**Posting.** Reels are published only when the user says to ("post it", "post tomorrow's"):
run `tools.post_reel <date> --dry-run`, show the caption, then run it for real and report the
permalink. It uses the Instagram Graph API with credentials in `.env` (gitignored), reads the
caption from the module docstring, needs status `music ok`, and sets the row to `posted`.
The API only accepts a `video_url`, so the MP4 is served for ~1 min through a temporary
`cloudflared` quick tunnel (installed via winget) at a secret path; it closes once fetched.
Never post on your own initiative or on a schedule.

**The video comes first; music is second.** Add a soundtrack only once the picture is approved,
computed from the same maths and never changing a frame. Pick a style from the doc's
"Soundtrack" catalogue (`common.music.STYLES` and friends), audition 2–3 candidates with
`--audio-only <tag>` (seconds each, no re-render) and let the user choose by listening.

## Commands

Dependencies are managed with `uv` (Python 3.13, see `pyproject.toml` / `uv.lock`). `torch` is
pulled from the PyTorch CUDA 12.8 index; many 2026 animations render on the GPU via torch (falling
back to CPU) or use `numba`.

```bash
uv sync                                          # install/refresh .venv
uv run python main.py lab.2026.0314              # -> outputs/20260314/
uv run python main.py lab.2026.0314 --seed 7     # RNG seed (default 0)
uv run python main.py lab.2026.0314 --name test  # output folder name override
uv run python main.py lab.2026.0908 --hdr        # animations: 10-bit HEVC HLG instead of H.264
RP_FPS=10 uv run python main.py lab.2026.1011 --name preview  # low-fps preview, same timing
uv run python main.py lab.2026.1011 --audio-only harp     # soundtrack only, over the last full run's frames -> 20261011_harp.mp4
uv run python main.py lab.2026.1011 --audio-only --audio-seed 5  # re-roll the music, rewrite 20261011.mp4
uv run python -m lab.2026.0314                   # standalone run via the __main__ block (seed 1)
uv run black <path>                              # formatter (defaults, no config)
uv run python -m tools.post_reel 2026-10-14 --dry-run  # caption + video checks for a scheduled Reel
uv run python -m tools.post_reel 2026-10-14            # publish it to Instagram, row -> posted
```

Always run figures as modules (`main.py` or `python -m`), never by file path —
`uv run python lab/2026/0314.py` fails with `ModuleNotFoundError: No module named 'common'`.

`main.py` imports the module, builds one `ImageProcessingSettings(seed, filename, hdr, audio_only,
audio_seed)` and calls
`generate(settings)`. The output folder is the module path minus its top-level package
(`lab.2026.0314` -> `20260314`, `static.polygons` -> `polygons`).

Animations need `ffmpeg` on `PATH`. `common.technology.images_to_video` uses `h264_nvenc` when
`nvidia-smi` is present, else `libx264`; lossless by default (`crf=0`).

There is no test suite. "Verifying" a figure means: `main.py` exits 0 and the output exists,
importing the module creates no stray `outputs/` folder, and for animations the frame count and
`ffprobe` of the MP4 look right — the exact commands are in `REFACTOR.md` ("How to verify a
converted module"). For new videos, follow Phase 3 of `docs/AGENT_INSTRUCTIONS.md`, which adds the
timed render and the loop-seam check.

## Architecture

- `lab/<YYYY>/<MMDD>.py` — the daily sketch archive (~270 scripts, 2019–2026), filename is the date.
- `static/`, `animations/` — sketches promoted to named figures. Same interface as `lab/`.
- `common/image_processing.py` — `ImageProcessingSettings`, the single object every figure routes
  its randomness and output through:
  - `settings.rng` — seeded `numpy.random.Generator`; the only source of randomness a figure should
    use (pass it into helpers), so `--seed` is meaningful.
  - `settings.save_to_png(fig, bg)` — static output, timestamp-named.
  - `settings.save_numbered_frame(i, bg)` + `settings.save_video(fps, crf=0)` — animations. Frames
    go to `outputs/<name>/frames/frame%04d.png` (the pattern ffmpeg reads); the frame call also does
    `plt.close()` + `gc.collect()`. `save_video` switches to `images_to_hdr_video` when `hdr=True`.
    With `audio=track` it keeps `<name>_silent.mp4`, writes `<name>.wav` + `_spectrogram.png`
    and muxes them into `<name>.mp4`.
  - `settings.audio_rng` — the soundtrack's own generator (`--audio-seed`, default `--seed`).
  - `__init__` **creates and clears** `outputs/<filename>/`, except with `audio_only`, which
    keeps the folder and makes `render_frames` skip and `save_video` only re-mux.
- `common/audio.py` (voices, effects, `master` to −14 LUFS, spectrogram) and `common/music.py`
  (scales, chords, tempo, `STYLES`) — soundtracks; see the doc's "Soundtrack" section.
- `common/video.py` — video plumbing: `FPS` (60 / `RP_FPS`), `WIDTH, HEIGHT`, `frame_count`,
  colour helpers, and `render_frames(settings, render, count, parallel=True, rgb=False)`, which
  renders every frame (thread pool, or in order for GPU/stateful renders), writes the PNGs and
  logs the render time. `common/gpu.py` — torch helpers (`select_device`, `gaussian_kernel`,
  `blur`, `splat`); kept separate so CPU-only videos don't import torch. New videos import these
  instead of copying them (the Sept 2026 scripts carry drifted private copies).
- `common/technology.py` (ffmpeg encoding, GPU detection), `common/simulation.py` (`get_rng`, random
  walks, brownian bridge), `common/fractal.py` (Julia set variants).
- `colors/palettes.py` — hex palettes as module-level constants (e.g. `ECOSPL`, `TWLAGN`, `NEAURA`).
  `SUNWAV` is a list of `(hex, count)` pairs, not plain hex strings — expand it before use.

The canonical generator shape:

```python
def generate(settings: ImageProcessingSettings = None):
    """Technical description."""
    settings = settings or ImageProcessingSettings(1)
    ...
    settings.save_to_png(fig, bg_color)   # or save_numbered_frame(i) in a loop + save_video(fps)


if __name__ == '__main__':
    generate()
```

### House style for a figure

12×12 in figure, DPI 100–200, black (`'k'`) or off-white (`'#f4f0e7'`) background, axes off / added
via `fig.add_axes((0, 0, 1, 1))` so the plot bleeds to the edge (Instagram square), colours from
`colors.palettes` or a custom palette at the top of the file, `loguru` for logging, and
`plt.close()` + `gc.collect()` after each figure — frame loops leak memory badly without it.
That is for stills. Videos are 1080×1920 frames drawn directly with cv2/NumPy or torch, written
by `common.video.render_frames` and encoded with `save_video(FPS, crf=18)`. Loops close
exactly (integer cycles of every periodic parameter); one-shots ease in and rest at the end.

## Gotchas

- Never write `def generate(settings=ImageProcessingSettings(1))`. The default is evaluated at
  **import** time, so merely importing the module creates and clears a stray `outputs/` folder.
  Use `= None` plus `settings = settings or ImageProcessingSettings(1)`. No module-level side
  effects either — `main.py` imports before calling.
- Don't run two generators targeting the same output folder concurrently: the second
  `ImageProcessingSettings.__init__` wipes the first's frames mid-write.
- `settings.save_frame()` and `settings.images_to_video()` are legacy (timestamped frames, cv2
  writer, doubled output filename; the root `openh264-*.dll` files exist only for it). Use the
  numbered-frame/`save_video` pair.
- Known non-runnable modules: `lab/2022/1107.py` and `lab/2022/1226.py` are empty;
  `lab.2024.1104` and `lab.2025.0712` depend on external programs not in the repo;
  `common/layout_image.py` loads the deleted `config.toml` and crashes on import. Some old lab
  scripts still contain dead `sys.argv` / `/mnt/c/.../Pictures/RandomPlots/` variables — leftovers,
  not where output goes.
- The repo is not black-formatted (single quotes in older files). Black new files, but don't
  reformat whole existing files as part of an unrelated edit.
- `outputs/` and `*.png` are gitignored.
