# Repository Guidelines

## Project Structure & Module Organization

RandomPlots creates generative art for `@random_plot` using Python and Matplotlib.

- `lab/<YYYY>/<MMDD>.py`: dated experimental sketches; place new figures here.
- `static/` and `animations/`: named, reusable figure generators.
- `common/`: image settings, rendering, video encoding, simulation, and fractal helpers.
- `colors/palettes.py`: shared palette constants.
- `main.py`: CLI for generators accepting an `ImageProcessingSettings` object.
- `outputs/`: generated images, frames, and videos; ignored by Git.
- `docs/AGENT_INSTRUCTIONS.md`: figure workflow, caption requirements, and rendering guidance.

## Build, Test, and Development Commands

Use Python 3.13 or newer and run commands from the repository root:

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py --help`: inspect CLI options.
- `uv run python main.py lab.2026.0314 --seed 42 --name review_0314`: render a compatible sketch into `outputs/review_0314/`.
- `uv run black <path>`: format changed Python files with Black defaults.
- `uv run black --check <path>`: check formatting without modifying files.
- `ffmpeg -version`: verify the external encoder required for animations.

There is no separate build step. Inspect legacy generators before running them: some lack the current settings interface or execute on import. Use module entry points where supported; running dated files directly can break imports.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase constants. Follow Black formatting.

New figures are Instagram Reels. "A new Instagram Reel" (or "a new Reel") always means a 60 fps, about 15 s, 1080×1920 video built on one mathematical idea, either a seamless loop or a one-shot with a beginning and an end; don't ask about format, frame rate, size or length. A full render must take about a minute (3 minutes at most), so make each frame a pure function of time and render frames in parallel with cv2 on a thread pool or torch on the GPU, through `common.video.render_frames` (torch helpers in `common.gpu`). Avoid long sequential simulations. Follow the "Video Spec", "Rendering Fast" and "Loop or One-Shot" sections of `docs/AGENT_INSTRUCTIONS.md`.

The video comes first and music second: add a soundtrack only after the picture is approved, computed from the same maths with `settings.audio_rng`, never changing a frame. Pick a style from the "Soundtrack" section of `docs/AGENT_INSTRUCTIONS.md` (`common.audio`, `common.music`), audition 2–3 candidates with `main.py <module> --audio-only <tag>` (seconds each, the frames are reused) and let the user choose by listening.

New generators should expose `generate(settings: ImageProcessingSettings = None)` and initialize settings inside the function. Use `settings.rng` for reproducible randomness. Start figure modules with a publication-ready caption and document the algorithm in `generate()`. Reuse shared saving helpers and palettes; close Matplotlib figures and release resources after rendering.

## Testing Guidelines

No automated test suite or coverage threshold is configured. Render changed generators with a fixed seed, inspect PNGs or MP4s, and check logs for errors. Try another seed and verify animation loop continuity when relevant. Use a unique `--name`: settings initialization clears the selected output directory.

## Commit & Pull Request Guidelines

Recent commits use lowercase action-led subjects, such as `add 0908 vorticity tide animation script` and `tweak 0906 conjugate bloom animation parameters`. Keep commits focused.

PRs should describe the visual or behavioral change, include reproduction commands and seeds, link relevant issues, and attach previews for visual changes because generated outputs are ignored.
