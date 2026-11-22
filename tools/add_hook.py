"""
Draw a script's opening HOOK onto a finished Reel's frames, without re-rendering anything.

    uv run python -m tools.add_hook 2026-11-19 --place     # choose HOOK_Y/HOOK_COLOUR + sheet
    uv run python -m tools.add_hook 2026-11-19 --preview   # contact sheet only, nothing changed
    uv run python -m tools.add_hook 2026-11-19 2026-11-20  # apply, re-encode, keep the audio

For Reels rendered before hooks existed. New Reels pass `hook=HOOK` to `render_frames`, which
draws the same thing at render time, so a full re-render reproduces this result exactly.

Applying a hook:
- copies the untouched opening frames to `frames_nohook/` once and always draws from them,
  so running it again (with new wording or a new HOOK_Y) never stacks two hooks;
- re-encodes the frames like `save_video(FPS, crf=18)` into `<name>_silent.mp4`;
- copies the soundtrack stream bit for bit from the current `<name>.mp4` (the chosen one),
  so the music is untouched. The pre-hook video is kept as `<name>_nohook.mp4`.
"""

import argparse
import datetime as dt
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from common.technology import images_to_video
from common.video import (
    FPS,
    HOOK_COLOUR,
    HOOK_Y,
    draw_hook,
    hook_alpha,
    hook_frames,
    hook_masks,
)

ROOT = Path(__file__).resolve().parent.parent
CRF = 18
PLACE_TOP, PLACE_BOTTOM = 230, 1250  # below IG's top bar, above its caption area
DARK_INK = "#1c1b22"


def load_hook(date: dt.date):
    name = f"lab.{date:%Y}.{date:%m%d}"
    module = importlib.import_module(name)
    module = importlib.reload(module)  # --place rewrites the script mid-run
    hook = getattr(module, "HOOK", None)
    if not hook:
        raise SystemExit(f"lab/{date:%Y}/{date:%m%d}.py has no HOOK")
    return (
        hook,
        getattr(module, "HOOK_Y", HOOK_Y),
        getattr(module, "HOOK_COLOUR", HOOK_COLOUR),
    )


def place(date: dt.date):
    """
    Pick HOOK_Y and HOOK_COLOUR from the opening frames and write them into the script:
    the band (y in TOP..BOTTOM) where the hook covers the least detail over its 3 s, with
    a mild preference for higher bands, and dark ink where that band is light.
    """
    hook, _, _ = load_hook(date)
    _, frames, backup = folders(date)
    masks = hook_masks(hook)
    height = sum(int(m.shape[0] * 1.25) for m in masks)
    width = max(m.shape[1] for m in masks) + 60
    detail = None
    for frame in range(0, hook_frames(), max(1, FPS // 3)):
        grey = cv2.cvtColor(original(frames, backup, frame), cv2.COLOR_BGR2GRAY)
        grey = grey.astype(np.float32)
        edges = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
        detail = edges if detail is None else np.maximum(detail, edges)
    x0 = max(0, detail.shape[1] // 2 - width // 2)
    rows = detail[:, x0 : x0 + width].sum(1)
    margin = 30
    costs = {}
    for y in range(PLACE_TOP, PLACE_BOTTOM - height, 10):
        cost = rows[y - margin : y + height + margin].sum()
        costs[y] = cost * (1 + (y - PLACE_TOP) / 1500)  # prefer higher, all else equal
    y = min(costs, key=costs.get)
    band = cv2.cvtColor(original(frames, backup, 0), cv2.COLOR_BGR2GRAY)
    light = band[y : y + height, x0 : x0 + width].mean() > 140
    colour = DARK_INK if light else HOOK_COLOUR

    script = ROOT / "lab" / f"{date:%Y}" / f"{date:%m%d}.py"
    text = script.read_text(encoding="utf-8")
    text = re.sub(r"\nHOOK_Y = .*", "", text)
    text = re.sub(r"\nHOOK_COLOUR = .*", "", text)
    text = re.sub(
        r"(\nHOOK = .*)", rf'\1\nHOOK_Y = {y}\nHOOK_COLOUR = "{colour}"', text, count=1
    )
    text = re.sub(
        r"hook=HOOK(, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)?\)",
        "hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)",
        text,
    )
    script.write_text(text, encoding="utf-8")
    logger.info(f"{date}: HOOK_Y = {y}, HOOK_COLOUR = {colour}")


def folders(date: dt.date):
    out = ROOT / "outputs" / f"{date:%Y%m%d}"
    return out, out / "frames", out / "frames_nohook"


def original(frames: Path, backup: Path, frame: int) -> np.ndarray:
    source = backup if (backup / f"frame{frame:04d}.png").exists() else frames
    image = cv2.imread(str(source / f"frame{frame:04d}.png"))
    if image is None:
        raise SystemExit(f"Missing frame {frame} in {source}")
    return image


def preview(dates):
    tiles = []
    for date in dates:
        hook, y, colour = load_hook(date)
        _, frames, backup = folders(date)
        masks = hook_masks(hook)
        tile = draw_hook(original(frames, backup, 0), masks, 1.0, y, colour)
        tile = cv2.resize(tile, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{date:%m-%d}", (8, tile.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # fmt: skip
        tiles.append(tile)
    columns = 6
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    path = ROOT / "outputs" / "hook_preview.png"
    cv2.imwrite(str(path), np.vstack(rows))
    logger.success(f"Preview: {path}")


def apply(date: dt.date):
    hook, y, colour = load_hook(date)
    out, frames, backup = folders(date)
    name = out.name
    final = out / f"{name}.mp4"
    if not final.exists():
        raise SystemExit(f"Missing {final}")
    count = len(list(frames.glob("frame*.png")))
    if count != round(15 * FPS):
        raise SystemExit(f"{frames} has {count} frames; set RP_FPS to the full-run rate")

    n = hook_frames()
    backup.mkdir(exist_ok=True)
    for frame in range(n):
        target = backup / f"frame{frame:04d}.png"
        if not target.exists():
            shutil.copy2(frames / f"frame{frame:04d}.png", target)
    masks = hook_masks(hook)
    for frame in range(n):
        image = draw_hook(
            original(frames, backup, frame), masks, hook_alpha(frame), y, colour
        )
        cv2.imwrite(str(frames / f"frame{frame:04d}.png"), image)

    nohook = out / f"{name}_nohook.mp4"
    if not nohook.exists():
        shutil.copy2(final, nohook)
    silent = f"{name}_silent.mp4"
    images_to_video(frames, silent, FPS, crf=CRF)
    muxed = out / f"{name}.hooking.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(out / silent), "-i", str(nohook),
           "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest",
           "-movflags", "+faststart", str(muxed)]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg mux failed:\n{result.stderr[-2000:]}")
    muxed.replace(final)
    logger.success(f"{date}: hook drawn on {n} frames -> {final.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Add a script's HOOK to a finished Reel.")
    parser.add_argument("dates", nargs="+", help="posting dates YYYY-MM-DD")
    parser.add_argument("--preview", action="store_true", help="contact sheet only")
    parser.add_argument(
        "--place", action="store_true", help="pick HOOK_Y/HOOK_COLOUR, write to script"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dates = [dt.date.fromisoformat(d) for d in args.dates]
    if args.place:
        for date in dates:
            place(date)
    if args.preview or args.place:
        preview(dates)
        return
    for date in dates:
        apply(date)


if __name__ == "__main__":
    main()
