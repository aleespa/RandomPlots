"""A fast, looping Penrose-tile animation for @random_plots."""

import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings

FPS = 60
LOOP_SECONDS = 6
LOOP_FRAMES = FPS * LOOP_SECONDS
REPEATS = 3
FIGURE_SIZE = (7.2, 12.8)  # 1080 x 1920 at 150 dpi
DPI = 150
BG_COLOR = "#f4f0e7"
PALETTE = ["#f4f0e7", "#5092B8", "#228B46", "#c9d06c", "#e26000", "#ff9b9b"]
N_WORKERS = min(os.cpu_count() or 1, 6)

_STATE = {}


def _init_worker(tiles, centers, wave_vectors, wave_phase, wave_speed, weights, frames_path):
    """Keep the fixed geometry in each rendering process, not each work item."""
    _STATE.update(
        tiles=tiles,
        centers=centers,
        wave_vectors=wave_vectors,
        wave_phase=wave_phase,
        wave_speed=wave_speed,
        weights=weights,
        frames_path=Path(frames_path),
        cmap=LinearSegmentedColormap.from_list("penrose_weather", PALETTE, N=256),
    )


def _tile_colours(frame):
    """A periodic, band-limited colour field; it closes exactly after LOOP_FRAMES."""
    st = _STATE
    t = 2 * np.pi * frame / LOOP_FRAMES
    argument = (
        st["centers"] @ st["wave_vectors"].T
        + st["wave_phase"]
        + t * st["wave_speed"]
    )
    field = (np.cos(argument) * st["weights"]).sum(axis=1)
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    colours = st["cmap"](field)
    # A little temporal breathing gives the coloured weather fronts a pulse.
    colours[:, 3] = 0.76 + 0.24 * (0.5 + 0.5 * np.cos(t + field * 2 * np.pi))
    return colours


def _render_chunk(frame_indices):
    """Render independent frame ranges. Geometry is made once per process."""
    st = _STATE
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=BG_COLOR)
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-4.15, 4.15)
    # `equal` shrinks this 9:16 axes by a few pixels because the data limits
    # are not precisely 9:16, exposing the figure's off-white face colour as
    # bars.  Fill the video canvas instead; the resulting sub-1% stretch is
    # invisible at this scale but guarantees edge-to-edge tiles.
    ax.set_aspect("auto")
    ax.axis("off")

    tiles = PolyCollection(
        st["tiles"],
        facecolors=_tile_colours(0),
        edgecolors="#0d3356",
        linewidths=5,
        antialiaseds=True,
    )
    ax.add_collection(tiles)
    for frame in frame_indices:
        tiles.set_facecolors(_tile_colours(frame))
        fig.savefig(st["frames_path"] / f"frame{frame:04d}.png", facecolor=BG_COLOR)

    plt.close(fig)
    gc.collect()


def _penrose_rhombs(rng, radius=10, crop_x=4, crop_y=5):
    """Construct a finite Penrose rhomb patch with the de Bruijn pentagrid dual."""
    angles = np.pi * np.arange(5) / 5
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    # Generic offsets prevent triple line intersections while preserving fivefold order.
    offsets = rng.uniform(-0.45, 0.45, 5)
    dual = normals.copy()
    tiles = []

    for i in range(5):
        for j in range(i + 1, 5):
            system = np.stack((normals[i], normals[j]))
            if abs(np.linalg.det(system)) < 1e-8:
                continue
            inverse = np.linalg.inv(system)
            for ni in range(-radius, radius + 1):
                for nj in range(-radius, radius + 1):
                    point = inverse @ np.array((ni - offsets[i], nj - offsets[j]))
                    if abs(point[0]) > crop_x or abs(point[1]) > crop_y:
                        continue

                    # The three non-crossing strips choose the dual lattice base.
                    q = np.floor(point @ normals.T + offsets).astype(int)
                    q[i], q[j] = ni - 1, nj - 1
                    base = q @ dual
                    rhomb = np.array(
                        [base, base + dual[i], base + dual[i] + dual[j], base + dual[j]]
                    )
                    centroid = rhomb.mean(axis=0)
                    if abs(centroid[0]) <= crop_x and abs(centroid[1]) <= crop_y:
                        tiles.append(rhomb)

    return np.asarray(tiles)


def generate(settings: ImageProcessingSettings = None):
    """Animate a Penrose pentagrid with a smoothly drifting, periodic colour field.

    Five families of parallel lines form de Bruijn's pentagrid.  Taking its dual
    produces thick and thin Penrose rhombs: an ordered pattern with no translational
    repeat.  The tile geometry is computed once; only face colours change at 60 fps,
    so rendering stays light enough for a short vertical loop.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    tiles = _penrose_rhombs(rng)
    centers = tiles.mean(axis=1)
    n_waves = 3
    directions = rng.uniform(0, 2 * np.pi, n_waves)
    wave_vectors = np.column_stack((np.cos(directions), np.sin(directions)))
    wave_vectors *= rng.uniform(0.45, 1.65, (n_waves, 1))
    wave_phase = rng.uniform(0, 2 * np.pi, n_waves)
    wave_speed = rng.integers(1, 4, n_waves) * rng.choice([-1, 1], n_waves)
    weights = rng.uniform(0.35, 1.0, n_waves)
    weights /= weights.sum()

    frames_path = settings.frames_path
    chunks = [chunk for chunk in np.array_split(np.arange(LOOP_FRAMES), min(N_WORKERS, LOOP_FRAMES)) if len(chunk)]
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=_init_worker,
        initargs=(tiles, centers, wave_vectors, wave_phase, wave_speed, weights, str(frames_path)),
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    # Repeat an exact six-second loop without paying to render duplicate frames.
    for repeat in range(1, REPEATS):
        for frame in range(LOOP_FRAMES):
            shutil.copyfile(
                frames_path / f"frame{frame:04d}.png",
                frames_path / f"frame{repeat * LOOP_FRAMES + frame:04d}.png",
            )
    settings.save_video(fps=FPS, crf=20)


if __name__ == "__main__":
    generate()
