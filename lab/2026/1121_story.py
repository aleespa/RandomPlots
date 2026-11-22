"""
The Proof Is 17 Million Lines Long: three Instagram Story cards.

Stills from the Reel of 2026-11-21 (lab/2026/1121.py) with one line of text each: the whole
proof lit up, the longest chain with the totals, and the statement turning from hypothesis
into theorem, pointing to tomorrow's Part 2. Needs that Reel's frames
(outputs/20261121/frames) from a full render.
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from common.image_processing import ImageProcessingSettings
from common.video import blit_mask, hex_to_bgr, latex_mask

FRAMES = Path("outputs") / "20261121" / "frames"
FPS = 60
INK = "#f4f0e7"
GOLD = "#ffd37a"
# (seconds into the Reel, lines of (LaTeX, size, colour)); text sits low, above the
# Story's reply bar (content stays between y = 270 and 1640)
CARDS = [
    (
        12.0,
        [
            (r"$\mathrm{Every\ file\ of\ the\ proof{,}\ in\ one\ picture}$", 46, INK),
            (
                r"$\mathrm{New\ Reel\ \cdot\ Part\ 1\ of\ 3\ \cdot\ on\ the\ profile}$",
                40,
                GOLD,
            ),
        ],
    ),
    (
        26.8,
        [
            (r"$\mathrm{Checked\ by\ a\ computer{,}\ file\ by\ file}$", 46, INK),
            (r"$\mathrm{The\ full\ Reel\ is\ on\ the\ profile}$", 40, GOLD),
        ],
    ),
    (
        35.0,
        [
            (r"$\mathrm{Part\ 2\ tomorrow}$", 46, GOLD),
            (r"$\mathrm{the\ Riemann\ hypothesis\ in\ one\ picture}$", 46, INK),
        ],
    ),
]
TEXT_BOTTOM = 1630


def halo(mask, sigma=8):
    pad = 2 * sigma
    return np.clip(cv2.GaussianBlur(np.pad(mask, pad), (0, 0), sigma) * 2.0, 0, 1)


def generate(settings: ImageProcessingSettings = None):
    """Story cards: a Reel frame, a dark fade at the bottom, two centred lines of text."""
    settings = settings or ImageProcessingSettings(1, "20261121_story")
    folder = settings.output_path / settings.filename
    rows = np.arange(1920, dtype=np.float32)[:, None, None]
    for n, (seconds, lines) in enumerate(CARDS, start=1):
        frame = FRAMES / f"frame{round(seconds * FPS):04d}.png"
        if not frame.exists():
            raise FileNotFoundError(f"{frame}: run the full render of lab.2026.1121")
        image = cv2.imread(str(frame)).astype(np.float32)
        image *= 1 - 0.75 * np.clip((rows - 1380) / 200, 0, 1)  # darken under the text
        masks = [(latex_mask(tex, size), colour) for tex, size, colour in lines]
        y = TEXT_BOTTOM - sum(m.shape[0] + 22 for m, _ in masks)
        for mask, colour in masks:
            g = halo(mask)
            pad = (g.shape[0] - mask.shape[0]) // 2
            blit_mask(image, g, 540, y - pad, (0, 0, 0), 0.7, "centre")
            blit_mask(image, mask, 540, y, hex_to_bgr(colour), 1.0, "centre")
            y += mask.shape[0] + 22
        path = folder / f"story_{n}.png"
        cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
        logger.info(f"{path}")


if __name__ == "__main__":
    generate()
