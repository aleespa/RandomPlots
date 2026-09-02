"""Curl-Noise Silk -- hair-thin streamlines drifting through a breathing incompressible field."""

import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter

from common.image_processing import ImageProcessingSettings

FPS = 60
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 2  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150
WIDTH, HEIGHT = 1080, 1920
BG_COLOR = "#000000"

# Slow-to-fast: cold violet for the still water, hot mint/cyan where the flow
# converges and accelerates, white reserved for the brightest pulse cores.
PALETTE = [
    "#031A2B",  # deep ocean
    "#064663",  # dark blue-teal
    "#0A7EA4",  # ocean blue
    "#00B4D8",  # tropical cyan
    "#48E0C2",  # seafoam
    "#E8FFFF",  # ocean mist
]
N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


def _make_modes(rng, n_modes, k_min, k_max, beta, max_harmonic):
    """Draw the random Fourier modes that make up the stream function.

    psi(x, y, t) = sum_i a_i * cos(kx_i x + ky_i y + w_i t + phi_i)

    Wavenumber magnitudes are log-uniform between k_min and k_max (cycles per
    frame height), directions isotropic, amplitudes ~ |k|^-beta so the field
    has a fractal (fBm-like) spectrum: big lazy swirls carrying finer eddies.
    Every temporal frequency w_i is an integer number of cycles per loop, so
    the field is exactly periodic in LOOP_SECONDS.
    """
    mag = np.exp(rng.uniform(np.log(k_min), np.log(k_max), n_modes)) * 2 * np.pi
    theta = rng.uniform(0, 2 * np.pi, n_modes)
    kx, ky = mag * np.cos(theta), mag * np.sin(theta)
    amp = mag ** (-beta)
    amp *= rng.uniform(0.6, 1.0, n_modes)
    harmonics = rng.integers(-max_harmonic, max_harmonic + 1, n_modes)
    harmonics[harmonics == 0] = rng.choice([-1, 1], (harmonics == 0).sum())
    omega = 2 * np.pi * harmonics / LOOP_SECONDS
    phi = rng.uniform(0, 2 * np.pi, n_modes)
    return dict(kx=kx, ky=ky, amp=amp, omega=omega, phi=phi)


def _velocity(x, y, t, modes):
    """Curl of the stream function: u = d(psi)/dy, v = -d(psi)/dx.

    Evaluated analytically mode by mode, so the field is divergence-free to
    machine precision -- no finite differences, no numerical sources or sinks.
    """
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    for kx, ky, a, w, p in zip(
        modes["kx"], modes["ky"], modes["amp"], modes["omega"], modes["phi"]
    ):
        s = np.sin(kx * x + ky * y + w * t + p)
        u -= a * ky * s
        v += a * kx * s
    return u, v


def _streamlines(seeds_x, seeds_y, t, modes, n_steps, step):
    """Trace a unit-speed streamline through the frozen field at time t,
    n_steps forward and n_steps backward from every seed (midpoint rule).

    Returns arrays of shape (2*n_steps+1, n_seeds): positions along the line
    (in order from tail to head), and the local speed |u| for colouring.
    """
    n_pts = 2 * n_steps + 1
    px = np.empty((n_pts, seeds_x.size), dtype=np.float32)
    py = np.empty_like(px)
    speed = np.empty_like(px)

    def unit(x, y):
        u, v = _velocity(x, y, t, modes)
        mag = np.hypot(u, v)
        return u / (mag + 1e-9), v / (mag + 1e-9), mag

    for direction, sign in ((1, 1.0), (-1, -1.0)):
        x, y = seeds_x.copy(), seeds_y.copy()
        ux, uy, mag = unit(x, y)
        px[n_steps], py[n_steps], speed[n_steps] = x, y, mag
        for i in range(1, n_steps + 1):
            hx, hy = x + 0.5 * sign * step * ux, y + 0.5 * sign * step * uy
            ux, uy, _ = unit(hx, hy)
            x, y = x + sign * step * ux, y + sign * step * uy
            ux, uy, mag = unit(x, y)
            j = n_steps + direction * i
            px[j], py[j], speed[j] = x, y, mag
    return px, py, speed


def _splat(px, py, weights, rgb):
    """Bilinear additive rasterisation of weighted, coloured points into a
    float RGB buffer. Each point is split across its four neighbouring pixels
    so lines are antialiased rather than stair-stepped."""
    fx = px * HEIGHT - 0.5  # image coords: x across WIDTH, y down HEIGHT
    fy = (1.0 - py) * HEIGHT - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    dx = fx - x0
    dy = fy - y0
    buf = np.zeros((3, HEIGHT * WIDTH), dtype=np.float64)
    for ox, oy, wgt in (
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi, yi = x0 + ox, y0 + oy
        ok = (xi >= 0) & (xi < WIDTH) & (yi >= 0) & (yi < HEIGHT)
        idx = (yi[ok] * WIDTH + xi[ok]).ravel()
        w = (wgt * weights)[ok].ravel()
        for c in range(3):
            buf[c] += np.bincount(idx, w * rgb[c][ok].ravel(), minlength=HEIGHT * WIDTH)
    return buf.reshape(3, HEIGHT, WIDTH)


def _render_buffer(t, st):
    """Streamlines -> weighted coloured points -> raw additive RGB buffer."""
    px, py, speed = _streamlines(
        st["seeds_x"], st["seeds_y"], t, st["modes"], st["n_steps"], st["step"]
    )
    n_pts = px.shape[0]
    # Arc length from the tail (index 0) in pixels, per point.
    s = (np.arange(n_pts) - st["n_steps"]) * st["step"] * HEIGHT
    # Cosine-squared taper so each thread fades out at both ends.
    envelope = np.cos(0.5 * np.pi * s / (s[-1] + 1e-9)) ** 2
    # A pulse travelling along the thread in the flow direction, one full
    # wavelength per `pulse_wavelength` px, `pulse_cycles` laps per loop.
    phase = (
        2
        * np.pi
        * (s[:, None] / st["pulse_wavelength"] - st["pulse_cycles"] * t / LOOP_SECONDS)
        + st["line_phase"][None, :]
    )
    pulse = st["pulse_floor"] + (1 - st["pulse_floor"]) * (0.5 + 0.5 * np.cos(phase))
    weights = (envelope[:, None] * pulse).astype(np.float32)

    # Colour by local speed through the palette.
    c = np.clip(speed / st["speed_scale"], 0, 1) ** st["colour_gamma"]
    lut = st["lut"]
    ci = (c * (lut.shape[0] - 1)).astype(np.int64)
    rgb = lut[ci].transpose(2, 0, 1)  # (3, n_pts, n_seeds)
    return _splat(px, py, weights, rgb)


def _tone_map(buf, gain, bloom_weight, bloom_sigma):
    """Soft additive exposure (1 - exp(-gain*x)) plus a blurred bloom layer."""
    img = 1.0 - np.exp(-gain * buf)
    if bloom_weight > 0:
        bloom = np.stack([gaussian_filter(ch, bloom_sigma) for ch in img])
        img = img + bloom_weight * bloom
    return np.clip(img, 0, 1).transpose(1, 2, 0).astype(np.float32)


def _init_worker(state):
    _STATE.update(state)
    _STATE["frames_path"] = Path(state["frames_path"])


def _render_chunk(frame_indices):
    """Render a run of frames, reusing one Figure/Axes/AxesImage."""
    st = _STATE
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")
    image = ax.imshow(
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    for f in frame_indices:
        t = f / FPS
        buf = _render_buffer(t, st)
        image.set_data(
            _tone_map(buf, st["gain"], st["bloom_weight"], st["bloom_sigma"])
        )
        fig.savefig(st["frames_path"] / f"frame{f:04d}.png", facecolor=BG_COLOR)
    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Curl-Noise Silk -- thousands of hair-thin streamlines drifting through a
    slowly breathing, incompressible flow field, as an exactly looping 9:16 clip.

    The velocity field is the curl of a scalar stream function, so it is
    divergence-free by construction: no sources or sinks, only swirl and
    shear, which is what gives fluid its silk-like folding. The stream
    function is a random superposition of plane waves,

        psi(x, y, t) = sum_i a_i cos(k_i . x + w_i t + phi_i),

    with wavenumbers log-uniform over roughly a decade and amplitudes falling
    as |k|^-beta -- a synthetic fractal spectrum that reads like curl noise but
    whose derivatives are exact. Each temporal frequency w_i is an integer
    number of cycles per loop, so the field at t = LOOP_SECONDS is identical
    to the field at t = 0.

    Each frame traces a streamline of fixed pixel length through the frozen
    field from every point of one fixed seed set (unit-speed midpoint
    integration, forward and backward). The threads are splatted additively
    with bilinear antialiasing, weighted by an end taper and by a pulse that
    travels along each thread in the flow direction an integer number of times
    per loop. Colour is local speed through the palette. Because seeds, field
    and pulses are all exactly periodic, frame LOOP_FRAMES == frame 0 and the
    clip loops without a crossfade.

    Exposure is fixed once for the whole clip from a probe of the raw density,
    so the picture does not pump. Frames are independent, so rendering is
    spread across worker processes and one period is copied REPEATS times.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    n_lines = 5_000  # threads per frame: airy vs dense fabric
    line_len_px = 230  # thread length in pixels: dust vs long silk
    step_px = 1.2  # integration step in pixels (finer = smoother curves)
    n_modes = 48  # Fourier modes in the stream function
    k_min, k_max = 0.7, 9.0  # wavenumber range, cycles per frame height
    beta = 1.9  # spectral slope: high = big lazy swirls, low = turbulence
    max_harmonic = 2  # max |cycles per loop| of any mode: how much it morphs
    pulse_wavelength = 90.0  # px between travelling highlights on a thread
    pulse_cycles = 6  # laps each pulse makes along its thread per loop
    pulse_floor = 0.15  # thread brightness between pulses (1 = no pulse)
    speed_pct = 97.0  # speed percentile mapped to the palette's hot end
    colour_gamma = 0.8  # <1 warms the midtones sooner
    exposure_pct = 99.5  # raw density percentile driven to full white
    bloom_weight = 0.35
    bloom_sigma = 6.0
    margin = 0.08  # seeds spawn this far outside the frame (frame height units)

    aspect = WIDTH / HEIGHT
    modes = _make_modes(rng, n_modes, k_min, k_max, beta, max_harmonic)
    seeds_x = rng.uniform(-margin, aspect + margin, n_lines).astype(np.float32)
    seeds_y = rng.uniform(-margin, 1 + margin, n_lines).astype(np.float32)
    line_phase = rng.uniform(0, 2 * np.pi, n_lines).astype(np.float32)

    lut = np.array(
        [
            to_rgb(c)
            for c in LinearSegmentedColormap.from_list("silk", PALETTE, N=512)(
                np.linspace(0, 1, 512)
            )
        ],
        dtype=np.float32,
    )

    step = step_px / HEIGHT
    n_steps = max(1, int(round(line_len_px / (2 * step_px))))

    state = dict(
        modes=modes,
        seeds_x=seeds_x,
        seeds_y=seeds_y,
        line_phase=line_phase,
        n_steps=n_steps,
        step=step,
        pulse_wavelength=pulse_wavelength,
        pulse_cycles=pulse_cycles,
        pulse_floor=pulse_floor,
        colour_gamma=colour_gamma,
        lut=lut,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=str(settings.frames_path),
    )

    # --- one exposure for the whole clip, from a few probe phases ---
    # Speed scale: percentile of |u| over the frame at several phases.
    gx, gy = np.meshgrid(np.linspace(0, aspect, 120), np.linspace(0, 1, 200))
    speeds = []
    for phase in np.arange(4) / 4:
        u, v = _velocity(gx.ravel(), gy.ravel(), phase * LOOP_SECONDS, modes)
        speeds.append(np.hypot(u, v))
    state["speed_scale"] = float(np.percentile(np.concatenate(speeds), speed_pct))

    highs = []
    for phase in np.arange(3) / 3:
        buf = _render_buffer(phase * LOOP_SECONDS, state)
        lum = buf.max(axis=0)
        highs.append(np.percentile(lum[lum > 0], exposure_pct))
    state["gain"] = 3.0 / max(float(np.max(highs)), 1e-6)
    logger.info(
        f"speed_scale={state['speed_scale']:.3f} gain={state['gain']:.2f} "
        f"n_steps={n_steps} ({2 * n_steps + 1} pts/thread)"
    )

    frames_path = settings.frames_path
    n_workers = min(N_WORKERS, LOOP_FRAMES)
    chunks = np.array_split(np.arange(LOOP_FRAMES), n_workers * 4)
    logger.info(f"rendering {LOOP_FRAMES} frames across {n_workers} workers")
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

    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
