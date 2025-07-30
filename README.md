# RandomPlots
Code for the instagram account of @random_plots

## Running Plot Generator Scripts via CLI

This project includes a command-line interface (CLI) tool to dynamically run any plot generator script's `generate()` function.

### Usage

```bash
python main.py <script_module>.<figure_name> 
```

Requirements: FFmpeg & GPU Support
====================================

This project depends on **FFmpeg** to convert image sequences into videos.

FFmpeg Installation
-------------------

You must have FFmpeg installed and accessible from your system’s command line (`ffmpeg` should work in a terminal or command prompt).

- **Windows**: Download from https://www.gyan.dev/ffmpeg/builds/ and add the `bin/` folder to your system `PATH`.
- **macOS**: Install via Homebrew:
  
```bash
  brew install ffmpeg
```

- **Linux (Debian/Ubuntu)**:

```bash
  sudo apt update && sudo apt install ffmpeg
```

You can verify installation with:


```bash
  ffmpeg -version
```

GPU Acceleration (Optional)
-----------------------------

If an **NVIDIA GPU** is available on your system and `nvidia-smi` is installed, the project will use **hardware-accelerated video encoding** via `h264_nvenc`, which is significantly faster than CPU encoding.

If no compatible GPU is detected, it will gracefully fall back to **CPU-based encoding** using `libx264`.

The output is still lossless in both cases — the only difference is performance.
 