import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings

FPS = 60
DURATION_SECONDS = 10
LOOP_REPEATS = 3  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150

# Custom palette (not a stock colormap, not one of colors.palettes): deep
# electric violet at low speed, sweeping through magenta into a saturated
# cyan flash at peak speed -- cool-toned but vivid, for a more electric,
# high-contrast "current" feel than a muted gradient gives.
CURRENT_PULSE = ["#170b3b", "#5b0ce0", "#a504e0", "#ff2ec4", "#00eaff", "#ffffff", "#ff2ec4", "#a504e0"]

# Rendering is CPU-bound per frame and each frame is independent given the
# precomputed trajectories below, so frames are split into chunks and rendered
# in parallel worker processes. Cap workers modestly -- most of the per-worker
# cost is matplotlib/Agg rasterization, which doesn't benefit from going far
# past the physical core count.
N_WORKERS = min(os.cpu_count() or 1, 16)

# Populated once per worker process by _init_worker; read by _render_chunk.
_STATE = {}


def _init_worker(pos, speed_norm, offset, cmap, frames_path, bg_color,
                  domain_x, domain_y, loop_frames, trail_len, fade_frames):
    """Runs once per worker process: stash the (large, read-only) precomputed
    data in module globals so it's sent across the process boundary once,
    rather than re-pickled with every chunk."""
    _STATE.update(
        pos=pos,
        speed_norm=speed_norm,
        offset=offset,
        particle_idx=np.arange(pos.shape[1]),
        cmap=cmap,
        frames_path=Path(frames_path),
        bg_color=bg_color,
        domain_x=domain_x,
        domain_y=domain_y,
        loop_frames=loop_frames,
        trail_len=trail_len,
        fade_frames=fade_frames,
    )


def _frame_segments(f):
    """Build the (segments, colors) comet-trail arrays for frame f from the
    precomputed per-particle trajectories in _STATE. Returns (None, None) if
    nothing is visible yet (only possible in the very first fade-in window)."""
    st = _STATE
    loop_frames, trail_len = st["loop_frames"], st["trail_len"]
    age = (f + st["offset"]) % loop_frames
    # smoothstep fade in/out around each particle's own loop seam
    fade = np.clip(np.minimum(age, loop_frames - age) / st["fade_frames"], 0, 1)

    segments = []
    colors = []
    for k in range(trail_len - 1):
        idx0 = age - k
        idx1 = idx0 - 1
        valid = idx1 >= 0
        if not np.any(valid):
            break
        idx0c = np.clip(idx0, 0, loop_frames - 1)
        idx1c = np.clip(idx1, 0, loop_frames - 1)
        recency = 1.0 - k / (trail_len - 1)
        seg_alpha = fade * recency * valid
        keep = seg_alpha > 0.02
        if not np.any(keep):
            continue
        p0 = st["pos"][idx0c[keep], st["particle_idx"][keep]]
        p1 = st["pos"][idx1c[keep], st["particle_idx"][keep]]
        rgba = st["cmap"](st["speed_norm"][idx0c[keep], st["particle_idx"][keep]])
        rgba[:, 3] = seg_alpha[keep] * 0.9
        segments.append(np.stack([p0, p1], axis=1))
        colors.append(rgba)

    if not segments:
        return None, None
    return np.concatenate(segments, axis=0), np.concatenate(colors, axis=0)


def _render_chunk(frame_indices):
    """Render a contiguous run of frames, reusing one Figure/Axes/LineCollection
    across all of them -- only the segment data changes frame to frame, so
    there's no need to rebuild the matplotlib scaffolding (and pay its layout
    and Agg-canvas setup cost) on every single frame."""
    st = _STATE
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(st["bg_color"])
    ax.axis("off")
    ax.set_xlim(-st["domain_x"], st["domain_x"])
    ax.set_ylim(-st["domain_y"], st["domain_y"])
    ax.set_aspect("equal")
    lc = LineCollection([], linewidths=1.3)
    ax.add_collection(lc)

    for f in frame_indices:
        segments, colors = _frame_segments(f)
        lc.set_segments(segments if segments is not None else [])
        if colors is not None:
            lc.set_color(colors)
        fig.savefig(st["frames_path"] / f"frame{f:04d}.png", facecolor=st["bg_color"])

    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Curl-noise flow field ("Riverbed"), animated as a perfectly looping 9:16 clip.

    A scalar potential Psi(x, y) is a sum of random plane waves (band-limited
    noise across coarse-to-fine octaves) plus a handful of Gaussian "vortex"
    bumps. The velocity field u = dPsi/dy, v = -dPsi/dx is divergence-free by
    construction and steady (time-independent), so particles braid, split, and
    swirl around the vortex centers without ever bunching into sinks.

    Each particle gets one deterministic lifetime of `loop_frames` positions,
    precomputed once via RK2 integration from a fixed spawn point. At render
    time particle i's visible state at frame f is read off at
    age = (f + offset_i) % loop_frames -- a periodic function of f with period
    `loop_frames` -- so every particle (and hence the whole rendered scene)
    repeats exactly after one period: a mathematically exact loop, no crossfade
    needed. Random per-particle phase offsets spread respawns across the whole
    clip, and each particle's opacity is smoothstep-faded in/out around its own
    seam so the respawn is invisible. A short trailing comet (fading by
    recency) is drawn behind each particle's current position, colored by
    local flow speed.

    Frame rendering (the actual bottleneck at high particle counts / long
    durations) is split across worker processes: each frame is fully
    independent given the precomputed trajectories, so this parallelizes with
    no approximation or change to the output.

    Only one loop period's worth of frames (`loop_frames`) is ever rendered;
    since it's an exact loop, the frames are then copied `LOOP_REPEATS` times
    to pad the delivered clip to `DURATION_SECONDS * LOOP_REPEATS` seconds
    without paying render cost for the repeats.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    bg_color = "k"

    # --- knobs ---
    n_particles = 2_300
    loop_frames = FPS * DURATION_SECONDS  # exact loop period, in frames
    dt = 0.01
    n_waves = 25                    # more waves -> finer turbulence
    freq_range = (0.6, 5.0)          # low freq = broad braids, high freq = ripples
    n_vortices = 15                   # eddy count
    vortex_strength_range = (0.4, 1.1)
    vortex_sigma_range = (0.25, 0.6)
    trail_len = 16                   # comet length, in frames
    fade_frames = trail_len * 2      # fade in/out around each particle's own seam
    domain_y = 1.6
    domain_x = domain_y * 9 / 16     # match the 9:16 frame

    # --- random terms of the steady potential ---
    angles = rng.uniform(0, 2 * np.pi, n_waves)
    freqs = rng.uniform(*freq_range, n_waves)
    kx = freqs * np.cos(angles)
    ky = freqs * np.sin(angles)
    amp = rng.uniform(0.3, 1.0, n_waves) / freqs  # softer amplitude at high freq
    phase = rng.uniform(0, 2 * np.pi, n_waves)

    vcx = rng.uniform(-domain_x * 0.7, domain_x * 0.7, n_vortices)
    vcy = rng.uniform(-domain_y * 0.7, domain_y * 0.7, n_vortices)
    vstrength = rng.uniform(*vortex_strength_range, n_vortices) * rng.choice(
        [-1, 1], n_vortices
    )
    vsigma = rng.uniform(*vortex_sigma_range, n_vortices)

    def velocity(x, y):
        phase_term = (
            x[:, None] * kx[None, :] + y[:, None] * ky[None, :] + phase[None, :]
        )
        cos_term = np.cos(phase_term)
        u = (amp[None, :] * ky[None, :] * cos_term).sum(axis=1)
        v = -(amp[None, :] * kx[None, :] * cos_term).sum(axis=1)

        dx = x[:, None] - vcx[None, :]
        dy = y[:, None] - vcy[None, :]
        r2 = dx**2 + dy**2
        bump = vstrength[None, :] * np.exp(-r2 / (2 * vsigma[None, :] ** 2))
        u += (bump * (-dy) / vsigma[None, :] ** 2).sum(axis=1)
        v += (bump * dx / vsigma[None, :] ** 2).sum(axis=1)
        return u, v

    # --- precompute one full, deterministic lifetime per particle (RK2) ---
    x = rng.uniform(-domain_x, domain_x, n_particles)
    y = rng.uniform(-domain_y, domain_y, n_particles)

    pos = np.empty((loop_frames, n_particles, 2))
    speed = np.empty((loop_frames, n_particles))
    for i in range(loop_frames):
        pos[i, :, 0] = x
        pos[i, :, 1] = y
        u1, v1 = velocity(x, y)
        speed[i] = np.hypot(u1, v1)
        xm, ym = x + 0.5 * dt * u1, y + 0.5 * dt * v1
        u2, v2 = velocity(xm, ym)
        x = x + dt * u2
        y = y + dt * v2

    speed_norm = speed / (speed.max() + 1e-9)
    cmap = LinearSegmentedColormap.from_list("riverbed", CURRENT_PULSE)

    # random phase so particles aren't all born/reset on the same frame
    offset = rng.integers(0, loop_frames, n_particles)

    frames_path = settings.frames_path  # creates the frames/ dir up front
    n_workers = min(N_WORKERS, loop_frames)
    chunks = np.array_split(np.arange(loop_frames), n_workers)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            pos, speed_norm, offset, cmap, str(frames_path), bg_color,
            domain_x, domain_y, loop_frames, trail_len, fade_frames,
        ),
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    # The loop is exact, so pad the clip by copying the rendered period rather
    # than re-rendering it: frame{0..loop_frames-1} -> repeated LOOP_REPEATS
    # times, renumbered to a single contiguous frame%04d.png sequence.
    for repeat in range(1, LOOP_REPEATS):
        for i in range(loop_frames):
            src = frames_path / f"frame{i:04d}.png"
            dst = frames_path / f"frame{repeat * loop_frames + i:04d}.png"
            shutil.copyfile(src, dst)

    # crf=28: no visible quality loss on this content (thin lines over a flat
    # black background compress very cleanly) but a small fraction of the
    # lossless file size -- Instagram recompresses on upload anyway, so a
    # multi-GB master buys nothing there and just makes the upload slower.
    settings.save_video(FPS, crf=20)


if __name__ == "__main__":
    generate()
