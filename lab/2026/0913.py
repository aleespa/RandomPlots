"""
Rose Memory

Nested trigonometric ribbons breathe and twist in continuous silence. The
geometry unfolds from an exact butterfly contour, x = r cos(u) sin(u) and
y = r cos(u) sin(2u), whose layered petals ripple across radius-dependent
phase shifts. Two pulses of pure light travel along the contours, tracing
the curves as they fold into an undulating silken knot before returning
to their origin.

Surrounding the core, a delicate field of luminous motes drifts along the
same harmonic deformation field. Every motion shares a single global clock
with integer temporal periodicity, completing a seamless closed loop with
no seams, reversals, or artificial crossfading.

Made with Python, NumPy, SciPy and Matplotlib. 60 fps, exact loop.

#generativeart #creativecoding #mathart #parametric #pythonart #animation
#loop #geometry #differentialgeometry #digitalart #randomplots
"""

import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from common.image_processing import ImageProcessingSettings

FPS = 60
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS
REPEATS = 2

# Vertical 9:16 format (1080x1920 @ 150 DPI)
WIDTH, HEIGHT = 1080, 1920
FIGURE_SIZE = (7.2, 12.8)
DPI = 150
ASPECT = WIDTH / HEIGHT  # 9/16 = 0.5625

CURVE_COUNT = 72
CURVE_SAMPLES = 600
BACKGROUND = "#08040d"
ROSE_MEMORY = ["#481660", "#7a1e78", "#b52c8c", "#ee4f9d", "#ff8fbc", "#ffe6ec"]
COLORMAP = LinearSegmentedColormap.from_list("rose_memory", ROSE_MEMORY)

PARTICLE_COUNT = 72
N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


class _RoseMemory:
    def __init__(self, rng):
        self.radii = np.linspace(0.065, 1.0, CURVE_COUNT)[:, None]
        self.parameter = np.linspace(0.0, 2.0 * np.pi, CURVE_SAMPLES)[None, :]
        self.wave_offset = float(rng.uniform(0.0, 2.0 * np.pi))
        self.twist = float(rng.uniform(0.30, 0.42))
        self.phase_depth = float(rng.uniform(0.85, 1.05))
        self.light_offset = float(rng.uniform(0.0, 2.0 * np.pi))

        # Ambient floating stardust motes riding the same harmonic deformation field
        self.particle_radii = rng.uniform(0.10, 1.05, PARTICLE_COUNT)[:, None]
        self.particle_u0 = rng.uniform(0.0, 2.0 * np.pi, PARTICLE_COUNT)[:, None]
        self.particle_speeds = rng.choice([-2, -1, 1, 2], size=(PARTICLE_COUNT, 1))
        self.particle_sizes = rng.uniform(4.0, 18.0, PARTICLE_COUNT)

    def coordinates(self, parameter, radii, phase):
        """Deform the original butterfly (r cos(u) sin(u), r cos(u) sin(2u)).

        Uses strictly isotropic scaling so the natural butterfly wings retain
        their true mathematical proportions without distortion.
        """
        envelope = np.sin(phase)
        radial_wave = np.sin(2.0 * np.pi * radii - phase + self.wave_offset)
        lag = envelope * (self.phase_depth + 0.32 * radial_wave)
        shear = 0.24 * np.sin(2.0 * phase) * radii
        radius = radii * (1.0 + 0.065 * envelope * radial_wave)
        horizontal = radius * np.cos(parameter) * np.sin(parameter + shear)
        vertical = radius * np.cos(parameter) * np.sin(2.0 * parameter + lag)
        angle = 0.16 * envelope + self.twist * (radii - 0.5) * np.sin(2.0 * phase)
        rotated_horizontal = horizontal * np.cos(angle) - vertical * np.sin(angle)
        rotated_vertical = horizontal * np.sin(angle) + vertical * np.cos(angle)

        # Uniform isotropic scaling: perfectly preserves aspect ratio
        scale = 1.62
        return np.stack((rotated_horizontal * scale, rotated_vertical * scale), axis=-1)

    def frame_data(self, phase):
        curves = self.coordinates(self.parameter, self.radii, phase)
        radii = self.radii[:, 0]
        colour_position = (
            0.18
            + 0.62 * radii
            + 0.10 * np.sin(2.0 * np.pi * radii - phase + self.wave_offset)
        )
        colours = COLORMAP(np.clip(colour_position, 0.0, 1.0))
        colours[:, 3] = 0.40 + 0.50 * radii**0.6

        # Travelling bright highlight threads
        trail_parameter = np.linspace(-0.42, 0.0, 36)
        head = phase + self.light_offset + 1.6 * self.radii
        trails = []
        for offset in (0.0, np.pi):
            points = self.coordinates(
                head + offset + trail_parameter, self.radii, phase
            )
            trails.append(np.stack((points[:, :-1], points[:, 1:]), axis=2))
        highlights = np.concatenate(trails, axis=0).reshape(-1, 2, 2)
        light_colours = COLORMAP(np.minimum(colour_position + 0.22, 1.0))
        light_colours = np.repeat(light_colours[:, None, :], 35, axis=1)
        light_colours[:, :, 3] = (0.18 + 0.72 * radii[:, None]) * np.linspace(
            0.0, 1.0, 35
        )[None, :] ** 1.8
        light_colours = np.tile(light_colours.reshape(-1, 4), (2, 1))

        # Ambient floating stardust motes
        particle_u = self.particle_u0 + self.particle_speeds * phase
        particle_pts = self.coordinates(particle_u, self.particle_radii, phase).reshape(
            PARTICLE_COUNT, 2
        )
        p_radii = self.particle_radii[:, 0]
        p_pulse = 0.5 + 0.5 * np.sin(phase + 3.0 * p_radii)
        p_cols = COLORMAP(np.clip(0.35 + 0.55 * p_radii, 0.0, 1.0))
        p_cols[:, 3] = 0.30 + 0.55 * p_pulse

        return curves, colours, highlights, light_colours, particle_pts, p_cols


def _init_worker(state):
    _STATE.update(state)
    _STATE["frames_path"] = Path(state["frames_path"])
    renderer = _RoseMemory.__new__(_RoseMemory)
    renderer.radii = state["radii"]
    renderer.parameter = state["parameter"]
    renderer.wave_offset = state["wave_offset"]
    renderer.twist = state["twist"]
    renderer.phase_depth = state["phase_depth"]
    renderer.light_offset = state["light_offset"]
    renderer.particle_radii = state["particle_radii"]
    renderer.particle_u0 = state["particle_u0"]
    renderer.particle_speeds = state["particle_speeds"]
    renderer.particle_sizes = state["particle_sizes"]
    _STATE["renderer"] = renderer


def _bloom(rgb, weight=0.42, sigma=6.0):
    """Add a screen-like optical bloom so highlights and curves softly glow."""
    if weight <= 0:
        return rgb
    emission = np.maximum(rgb - 0.12, 0.0)
    glow = np.stack(
        [gaussian_filter(ch, sigma) for ch in emission.transpose(2, 0, 1)]
    ).transpose(1, 2, 0)
    return np.clip(rgb + weight * glow, 0.0, 1.0)


def _render_chunk(frame_indices):
    """Render a sequence of frames, reusing a single Matplotlib Figure for speed."""
    st = _STATE
    renderer = st["renderer"]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI, facecolor=BACKGROUND)
    try:
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_facecolor(BACKGROUND)
        ax.set(
            xlim=(-st["half_width"], st["half_width"]),
            ylim=(-st["half_height"], st["half_height"]),
            aspect="equal",
        )
        ax.axis("off")

        # Layer collections for multi-pass luminous rendering
        layers = []
        for width, opacity in ((6.5, 0.04), (3.0, 0.10), (0.85, 1.0)):
            lc = LineCollection(
                [],
                linewidths=width,
                capstyle="round",
                joinstyle="round",
            )
            ax.add_collection(lc)
            layers.append((lc, opacity))

        highlight_lc = LineCollection(
            [],
            linewidths=1.2,
            capstyle="round",
        )
        ax.add_collection(highlight_lc)

        # Ambient stardust motes (diffuse halo + bright core)
        particle_halo = ax.scatter([], [], s=1, linewidths=0, zorder=4)
        particle_core = ax.scatter([], [], s=1, linewidths=0, zorder=5)
        particle_halo.set_sizes(renderer.particle_sizes * 4.0)
        particle_core.set_sizes(renderer.particle_sizes)

        for f in frame_indices:
            phase = 2.0 * np.pi * f / st["loop_frames"]
            (
                curves,
                colours,
                highlights,
                light_colours,
                particles,
                p_cols,
            ) = renderer.frame_data(phase)

            for lc, opacity in layers:
                layer_colours = colours.copy()
                layer_colours[:, 3] *= opacity
                lc.set_segments(curves)
                lc.set_color(layer_colours)

            highlight_lc.set_segments(highlights)
            highlight_lc.set_color(light_colours)

            particle_halo.set_offsets(particles)
            halo_cols = p_cols.copy()
            halo_cols[:, 3] *= 0.25
            particle_halo.set_color(halo_cols)

            particle_core.set_offsets(particles)
            particle_core.set_color(p_cols)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
            enhanced = _bloom(frame, weight=st["bloom_weight"], sigma=st["bloom_sigma"])
            plt.imsave(st["frames_path"] / f"frame{f:04d}.png", enhanced)
    finally:
        plt.close(fig)
        gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """Animate the nested trigonometric curves in a vertical 9:16 format with parallel rendering.

    The parametric curve family:
        x = r * cos(u) * sin(u)
        y = r * cos(u) * sin(2u)
    defines the classical butterfly shape at stationary phases (phase = 0 and pi).
    A smooth, radius-dependent wave deformation unfolds the petals into rippling ribbons,
    introducing differential rotation and periodic shear before snapping back into the
    pristine butterfly geometry.

    Enhancements:
        - True isotropic scale and aspect="equal" preserving natural butterfly proportions.
        - Ambient floating stardust motes traversing the deformation field in exact harmonic cycles.
        - Multi-pass luminous depth with optical bloom for cinematic glow.
        - Parallel chunked rendering via ProcessPoolExecutor and persistent Matplotlib Figure reuse.

    RP_FPS and RP_DPI environment variables allow rapid low-overhead preview generation.
    """
    fps = int(os.environ.get("RP_FPS", FPS))
    dpi = int(os.environ.get("RP_DPI", DPI))
    if fps <= 0 or dpi <= 0:
        raise ValueError("RP_FPS and RP_DPI must be positive integers")

    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    base_renderer = _RoseMemory(rng)
    loop_frames = fps * LOOP_SECONDS

    # In 9:16 (aspect = 0.5625), xlim = (-0.95, 0.95), ylim = (-1.69, 1.69)
    # With aspect='equal', the butterfly (height ~1.24, width ~0.80) fits with
    # balanced natural margins without any vertical stretching.
    half_width = 0.95
    half_height = half_width / ASPECT

    bloom_weight = 0.40
    bloom_sigma = 5.5

    state = dict(
        radii=base_renderer.radii,
        parameter=base_renderer.parameter,
        wave_offset=base_renderer.wave_offset,
        twist=base_renderer.twist,
        phase_depth=base_renderer.phase_depth,
        light_offset=base_renderer.light_offset,
        particle_radii=base_renderer.particle_radii,
        particle_u0=base_renderer.particle_u0,
        particle_speeds=base_renderer.particle_speeds,
        particle_sizes=base_renderer.particle_sizes,
        loop_frames=loop_frames,
        half_width=half_width,
        half_height=half_height,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=str(settings.frames_path),
    )

    n_workers = min(N_WORKERS, loop_frames)
    chunks = np.array_split(np.arange(loop_frames), n_workers * 3)
    chunks = [list(c) for c in chunks if len(c) > 0]
    logger.info(
        f"Rendering Rose Memory: {loop_frames} vertical frames (9:16) at {fps} fps across {n_workers} workers"
    )

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(state,)
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    # Duplicate seamless loop cycles for seamless playback
    frames_path = settings.frames_path
    for repeat in range(1, REPEATS):
        for i in range(loop_frames):
            shutil.copyfile(
                frames_path / f"frame{i:04d}.png",
                frames_path / f"frame{repeat * loop_frames + i:04d}.png",
            )

    gc.collect()
    settings.save_video(fps=fps, crf=18)


if __name__ == "__main__":
    generate()
