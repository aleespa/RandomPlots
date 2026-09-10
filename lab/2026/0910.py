"""
Kuramoto Fireflies

Seven hundred oscillators, each with its own natural rhythm, and one rule:

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ − θᵢ)

Every firefly nudges its phase toward the others. When the coupling K is weak
they flash in disorder; as K rises past a critical value, order emerges out of
nothing and they lock into a single pulse. Then K falls and synchrony dissolves
again.

Outer ring: the fireflies in order of natural frequency, flashing. Inner
circle: their phases. Silken threads tie each firefly to its phase; the bright
point in the middle is the mean field, the order parameter r that measures
how much of the swarm is in step. The strip below traces r over the loop.

Made with Python, numpy and matplotlib. 60 fps, exact loop.

#generativeart #creativecoding #kuramoto #synchronization #emergence
#complexsystems #mathart #python #fireflies
"""

import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from common.image_processing import ImageProcessingSettings

# RP_FPS=5 in the environment gives a quick 40-frame test loop with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 3  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150
HALF_WIDTH, HALF_HEIGHT = 1.0, 16 / 9
BG_COLOR = "#000000"

# Colour follows natural frequency, so the outer ring reads as one gradient and
# the inner circle shows which frequencies have joined the synchronised cluster.
PALETTE = ["#64f5d2", "#39badb", "#6679ed", "#bf92f3", "#f5bb83", "#64f5d2"]

# Layout, in axis units (x in [-1, 1]).
RING_CENTER = (0.0, 0.28)
R_OUTER = 0.86
R_INNER = 0.5
STRIP_Y = (-1.22, -0.99)
STRIP_X = (-0.72, 0.72)

N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class _Swarm:
    """The Kuramoto model in mean-field form, O(N) per step, integrated with RK4."""

    def __init__(self, knobs: dict, rng: np.random.Generator):
        self.k = knobs
        n = knobs["n"]
        # Natural frequencies in the rotating frame of their mean; sorted so the
        # ring slot (and the colour) of an oscillator is its frequency rank.
        omega = np.sort(rng.normal(0.0, knobs["sigma"], n))
        self.omega = omega - omega.mean()
        self.nudge = None  # (lambda, target phases) while returning to the loop

    def coupling(self, tau: float) -> float:
        """K over the loop: weak at the join, strongest half-way through."""
        k = self.k
        return k["k_min"] + (k["k_max"] - k["k_min"]) * 0.5 * (
            1.0 - np.cos(2 * np.pi * tau)
        )

    @staticmethod
    def order(theta):
        z = np.exp(1j * theta).mean()
        return abs(z), np.angle(z)

    def rhs(self, theta, tau: float):
        r, psi = self.order(theta)
        d = self.omega + self.coupling(tau) * r * np.sin(psi - theta)
        if self.nudge is not None:
            lam, target = self.nudge
            d = d + lam * np.sin(target - theta)
        return d

    def step(self, theta, tau: float, dtau: float):
        dt = dtau * self.k["loop_time"]
        k1 = self.rhs(theta, tau)
        k2 = self.rhs(theta + 0.5 * dt * k1, tau + 0.5 * dtau)
        k3 = self.rhs(theta + 0.5 * dt * k2, tau + 0.5 * dtau)
        k4 = self.rhs(theta + dt * k3, tau + dtau)
        return theta + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def advance_frame(self, theta, tau: float, nudge=None):
        """Advance one frame (1 / LOOP_FRAMES of the loop) in several RK4 substeps."""
        sub = self.k["substeps"]
        dtau = 1.0 / (LOOP_FRAMES * sub)
        for i in range(sub):
            if nudge is not None:
                lam_fn, lo, hi = nudge
                frac = (i + 0.5) / sub
                self.nudge = (
                    lam_fn(tau + frac / LOOP_FRAMES),
                    lo + frac * _wrap(hi - lo),
                )
            theta = self.step(theta, tau + i * dtau, dtau)
        self.nudge = None
        return theta


def _init_worker(state):
    _STATE.update(state)
    _STATE["frames_path"] = Path(state["frames_path"])


def _render_chunk(frame_indices):
    """Draw luminous filaments and periodic trails, then bloom bright pixels."""
    st = _STATE
    n = st["theta"].shape[1]
    base = st["colors"]
    slot = 2 * np.pi * np.arange(n) / n + np.pi / 2
    cx, cy = RING_CENTER
    center = np.array([cx, cy])
    directions = np.column_stack([np.cos(slot), np.sin(slot)])
    outer = center + R_OUTER * directions
    tangent = np.column_stack([-np.sin(slot), np.cos(slot)])
    # Cubic control points give the connections a common, gently swept grain.
    control = outer - 0.27 * directions + 0.19 * tangent
    t = np.linspace(0, 1, 28)[None, :, None]
    trail_steps = max(3, round(FPS * 0.16))

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
    ax.set_ylim(-HALF_HEIGHT, HALF_HEIGHT)
    ax.axis("off")

    # A very faint atmospheric field lifts the silhouette off the midnight canvas.
    gx, gy = np.meshgrid(
        np.linspace(-1, 1, 360), np.linspace(-HALF_HEIGHT, HALF_HEIGHT, 640)
    )
    radius = np.hypot(gx - cx, gy - cy)
    atmosphere = np.exp(-(((radius - 0.65) / 0.37) ** 2))
    background = np.empty((*radius.shape, 3))
    background[:] = 0.0
    background += atmosphere[..., None] * np.array([0.001, 0.003, 0.005])
    ax.imshow(
        background, extent=(-1, 1, -HALF_HEIGHT, HALF_HEIGHT), origin="lower", zorder=0
    )

    angles = np.linspace(0, 2 * np.pi, 721)
    for radius, alpha in [(R_OUTER + 0.034, 0.18), (R_INNER, 0.13)]:
        ax.plot(
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles),
            color="#8ebcca",
            lw=0.35,
            alpha=alpha,
            zorder=1,
        )
    ticks = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    tick_dirs = np.column_stack([np.cos(ticks), np.sin(ticks)])
    tick_ends = R_OUTER + 0.047 + np.where(np.arange(120) % 10 == 0, 0.022, 0.009)
    tick_lines = np.stack(
        [
            center + (R_OUTER + 0.047) * tick_dirs,
            center + tick_ends[:, None] * tick_dirs,
        ],
        axis=1,
    )
    ax.add_collection(
        LineCollection(tick_lines, colors="#7aabbc", linewidths=0.4, alpha=0.23)
    )

    # Slightly heavier filaments survive downsampling to a phone-sized image.
    chords = LineCollection([], linewidths=0.78, capstyle="round", zorder=2)
    ax.add_collection(chords)
    trails = LineCollection([], linewidths=1.0, capstyle="round", zorder=3)
    ax.add_collection(trails)
    outer_glow = ax.scatter(outer[:, 0], outer[:, 1], s=24, linewidths=0, zorder=4)
    outer_core = ax.scatter(outer[:, 0], outer[:, 1], s=3.2, linewidths=0, zorder=5)
    inner_dots = ax.scatter([], [], s=5, linewidths=0, zorder=5)
    (spoke,) = ax.plot([], [], color="#d6f7ec", lw=0.85, zorder=4)
    field_core = ax.scatter([cx], [cy], s=1, color="#edfff5", linewidths=0, zorder=6)
    (field_ring,) = ax.plot([], [], color="#aceddc", lw=0.55, zorder=5)
    ax.scatter([cx], [cy], s=3, color="#7294a0", alpha=0.55, linewidths=0)

    # The trace closes at t=1; a traveling highlight follows the cyclic timeline.
    xs = np.linspace(*STRIP_X, LOOP_FRAMES + 1)
    ys = STRIP_Y[0] + (STRIP_Y[1] - STRIP_Y[0]) * np.r_[st["r"], st["r"][0]]
    trace_points = np.column_stack([xs, ys])
    trace_segments = np.stack([trace_points[:-1], trace_points[1:]], axis=1)
    ax.fill_between(xs, STRIP_Y[0], ys, color="#5acdbd", alpha=0.035)
    ax.plot(xs, np.full_like(xs, STRIP_Y[0]), color="#7fa9b7", lw=0.4, alpha=0.22)
    trace = LineCollection(trace_segments, linewidths=1.55, zorder=4)
    ax.add_collection(trace)
    marker = ax.scatter([], [], s=14, color="#e4fff4", linewidths=0, zorder=6)

    for f in frame_indices:
        theta = st["theta"][f]
        r, psi = st["r"][f], st["psi"][f]
        flash = 0.5 * (1.0 + np.cos(theta))
        white = 0.18 * flash[:, None] ** 6  # retain saturated colour in bright pulses
        rgb = base * (1.0 - white) + white
        inner_dirs = np.column_stack([np.cos(theta), np.sin(theta)])
        inner = center + R_INNER * inner_dirs
        destination_control = center + (R_INNER - 0.18) * inner_dirs
        curves = (
            (1 - t) ** 3 * outer[:, None, :]
            + 3 * (1 - t) ** 2 * t * control[:, None, :]
            + 3 * (1 - t) * t**2 * destination_control[:, None, :]
            + t**3 * inner[:, None, :]
        )
        chords.set_segments(curves)
        # Raise colour density for mobile viewing while keeping the dark field open.
        chord_rgb = np.clip(base * 1.14, 0.0, 1.0)
        chords.set_color(np.column_stack([chord_rgb, 0.07 + 0.31 * flash**2]))

        # Read history cyclically so trails remain present across the loop seam.
        history = st["theta"][(f - np.arange(trail_steps, -1, -1)) % LOOP_FRAMES]
        history_xy = center + R_INNER * np.stack(
            [np.cos(history), np.sin(history)], axis=-1
        )
        trail_segments = np.stack([history_xy[:-1], history_xy[1:]], axis=2)
        trails.set_segments(trail_segments.reshape(-1, 2, 2))
        trail_colors = np.empty((trail_steps, n, 4))
        trail_colors[..., :3] = np.clip(base * 1.18, 0.0, 1.0)
        trail_colors[..., 3] = np.linspace(0.02, 0.25, trail_steps)[:, None]
        trails.set_color(trail_colors.reshape(-1, 4))

        outer_glow.set_color(
            np.column_stack([np.clip(base * 1.12, 0.0, 1.0), 0.045 + 0.16 * flash**3])
        )
        outer_glow.set_sizes(8 + 32 * flash**3)
        outer_core.set_color(
            np.column_stack([np.clip(rgb * 1.08, 0.0, 1.0), 0.3 + 0.7 * flash**2])
        )
        outer_core.set_sizes(1.8 + 6.0 * flash**4)
        inner_dots.set_offsets(inner)
        inner_dots.set_color(np.column_stack([rgb, np.full(n, 0.8)]))

        field = center + R_INNER * r * np.array([np.cos(psi), np.sin(psi)])
        spoke.set_data([cx, field[0]], [cy, field[1]])
        spoke.set_alpha(0.12 + 0.35 * r)
        field_core.set_offsets([field])
        field_core.set_sizes([5 + 30 * r**2])
        ring_radius = 0.018 + 0.018 * r
        field_ring.set_data(
            field[0] + ring_radius * np.cos(angles),
            field[1] + ring_radius * np.sin(angles),
        )
        field_ring.set_alpha(0.2 + 0.4 * r)

        age = (f / LOOP_FRAMES - np.arange(LOOP_FRAMES) / LOOP_FRAMES) % 1.0
        trace_colors = np.empty((LOOP_FRAMES, 4))
        trace_colors[:, :3] = np.array([0.42, 0.88, 0.78])
        trace_colors[:, 3] = 0.22 + 0.7 * np.exp(-age / 0.12)
        trace.set_color(trace_colors)
        marker.set_offsets([[xs[f], ys[f]]])

        # Screen-blended optical bloom: soft light without opaque scatter discs.
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255
        emission = np.maximum(pixels - 0.18, 0.0)
        bloom = 0.65 * gaussian_filter(
            emission, sigma=(2.2, 2.2, 0)
        ) + 1.15 * gaussian_filter(emission, sigma=(9, 9, 0))
        pixels = 1.0 - (1.0 - pixels) * np.exp(-bloom)
        plt.imsave(st["frames_path"] / f"frame{f:04d}.png", np.clip(pixels, 0, 1))
    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Kuramoto Fireflies -- the Kuramoto model of N coupled phase oscillators,

        dtheta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i),

    rendered as an exactly looping 9:16 clip. In mean-field form the sum is
    K r sin(psi - theta_i) with r e^{i psi} = mean(e^{i theta}), so each RK4
    step is O(N); the whole simulation is negligible next to drawing.

    Natural frequencies are normal with spread sigma (in the rotating frame of
    their mean), sorted so an oscillator's ring slot and colour are its
    frequency rank. The coupling sweeps K_min -> K_max -> K_min over the loop
    as a raised cosine; with K_c ~ 1.6 sigma inside that range the swarm
    crosses the synchronisation transition twice per loop.

    The loop: an incoherent oscillator drifts at its own frequency, so the
    state is not periodic in K's period by itself. As in 0908, one reference
    period A(t) is recorded after a warm-up and the clip is a second run,
    started from A(T), that is nudged back onto A by an extra term
    lambda(t) sin(A_i - theta_i) with lambda ramping up from zero; a short
    blend over the last frames (in wrapped phase) closes what remains. The
    residual phase mismatch is logged.

    Drawing: phases are shown in the lab frame, theta + 2 pi m tau with integer
    m flashes per loop, so the synchronised swarm visibly pulses together.
    Outer ring of fireflies whose brightness is (1 + cos theta) / 2,
    inner circle with cyclic phase trails, cubic filaments joining each firefly
    to its phase, the mean-field point r e^{i psi}, and a luminous synchrony
    trace. A midnight atmosphere, fine orbital ticks and screen-blended bloom
    give the diagram depth. Frames use one persistent Figure per worker.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    knobs = dict(
        n=720,  # oscillators
        sigma=1.0,  # spread of natural frequencies
        k_min=0.4,  # coupling at the loop's join (below K_c ~ 1.6)
        k_max=2.8,  # coupling half-way (above K_c; higher = tighter cluster)
        flashes=5,  # flashes per loop of the synchronised swarm (integer)
        loop_time=36.0,  # simulated time units per loop
        substeps=max(4, int(np.ceil(240 / FPS))),  # stable at preview frame rates
        warmup_periods=2,
        nudge_lambda=3.0,  # peak relaxation rate toward the reference, per time unit
        nudge_ramp=0.45,  # fraction of the loop over which the rate ramps up
        join_fraction=0.08,  # tail of the loop blended onto the reference exactly
    )

    swarm = _Swarm(knobs, rng)
    theta = rng.uniform(0.0, 2 * np.pi, knobs["n"])

    # --- warm-up onto the driven swarm's attractor (tau only matters modulo 1) ---
    for i in range(knobs["warmup_periods"] * LOOP_FRAMES):
        theta = swarm.advance_frame(theta, (i % LOOP_FRAMES) / LOOP_FRAMES)

    # --- reference period ---
    ref = [theta.copy()]
    for i in range(LOOP_FRAMES):
        theta = swarm.advance_frame(theta, i / LOOP_FRAMES)
        ref.append(theta.copy())
    ref = np.array(ref)  # (N_frames + 1, n); ref[-1] is the loop's first state

    # --- the loop, nudged back onto the reference ---
    lam_max, ramp = knobs["nudge_lambda"], knobs["nudge_ramp"]

    def lam(phase):
        return lam_max * _smoothstep(phase / ramp)

    join_frames = max(2, int(round(knobs["join_fraction"] * LOOP_FRAMES)))
    frames_theta = np.empty((LOOP_FRAMES, knobs["n"]))
    residuals = []
    for i in range(LOOP_FRAMES):
        shown = theta
        k_join = i - (LOOP_FRAMES - join_frames)
        if k_join >= 0:
            sj = _smoothstep((k_join + 1) / (join_frames + 1))
            shown = theta + sj * _wrap(ref[i] - theta)
        frames_theta[i] = shown
        if i % (LOOP_FRAMES // 8) == 0 or i == LOOP_FRAMES - 1:
            residuals.append(np.abs(_wrap(theta - ref[i])).mean())
        theta = swarm.advance_frame(
            theta, i / LOOP_FRAMES, nudge=(lam, ref[i], ref[i + 1])
        )
    logger.info(
        "mean |phase - reference| (rad) at 8 phases + last frame: "
        + ", ".join(f"{r:.3f}" for r in residuals)
    )

    # Show the phases in the lab frame: the swarm's mean frequency makes an
    # integer number of turns per loop, so a synchronised swarm flashes
    # together `flashes` times per loop and the join stays exact.
    frames_theta += (
        2 * np.pi * knobs["flashes"] * (np.arange(LOOP_FRAMES) / LOOP_FRAMES)[:, None]
    )
    z = np.exp(1j * frames_theta).mean(axis=1)
    r, psi = np.abs(z), np.angle(z)
    logger.info(f"order parameter r: min {r.min():.2f}, max {r.max():.2f}")

    cmap = LinearSegmentedColormap.from_list("fireflies", PALETTE, N=1024)
    colors = cmap(np.linspace(0.0, 1.0, knobs["n"]))[:, :3]

    frames_path = settings.frames_path
    state = dict(
        theta=frames_theta,
        r=r,
        psi=psi,
        colors=colors,
        frames_path=str(frames_path),
    )
    n_workers = max(1, min(N_WORKERS, LOOP_FRAMES))
    chunks = [list(c) for c in np.array_split(np.arange(LOOP_FRAMES), n_workers * 3)]
    chunks = [c for c in chunks if c]
    logger.info(f"rendering {LOOP_FRAMES} frames on {n_workers} workers")
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(state,)
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    for repeat in range(1, REPEATS):
        for i in range(LOOP_FRAMES):
            shutil.copyfile(
                frames_path / f"frame{i:04d}.png",
                frames_path / f"frame{repeat * LOOP_FRAMES + i:04d}.png",
            )

    gc.collect()
    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
