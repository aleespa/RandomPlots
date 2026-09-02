"""Mobius Tide -- loxodromic spirals streaming between the two fixed points of a Mobius map."""

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

FPS = 10
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 2  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150
WIDTH, HEIGHT = 1080, 1920
BG_COLOR = "#000000"

# ECOSPL blue -> ECOSPL rose along the tide (source pole -> sink pole), with a
# near-black toe so the empty plane stays dark and white kept for line cores.
PALETTE = ["#07111f", "#5092B8", "#58c0e7", "#c9d06c", "#ff9b9b", "#ffffff"]

N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


def _log_coordinate(z, p, q):
    """w = log((z - p) / (z - q)) and its derivative dw/dz.

    w is the coordinate in which the Mobius map with fixed points p, q is a
    pure translation: circles |C| = const are the Apollonian circles around
    p and q, lines arg C = const are the circle arcs through both.
    """
    zp, zq = z - p, z - q
    w = np.log(zp / zq)
    dw = 1.0 / zp - 1.0 / zq
    return w, dw


def _layer(w, dw, n, c, k, tau, sigma_px, min_spacing_px, fade_px):
    """Thin bright level lines of phi = Re(g w), g = c - i n, shifted by 2*pi*k*tau.

    phi = c*log|C| + n*arg(C): n = 0 gives Apollonian circles that stream from
    one fixed point to the other (the tide), c = 0 gives arcs through both
    fixed points that rotate about them, anything else is a loxodrome. The
    field is holomorphic so |grad phi| = |g dw/dz| exactly; distance to the
    nearest line is measured in pixels with it, giving a constant on-screen
    line width, and lines fade out where their spacing falls below the
    pixel grid so the poles dissolve instead of aliasing.

    Returns (line mask in [0, 1], soft fill 0.5 + 0.5 cos(phi)).
    """
    g = c - 1j * n
    phi = (g * w).real - 2 * np.pi * k * tau
    grad = np.abs(g * dw) / HEIGHT  # radians per pixel
    spacing = 2 * np.pi / (grad + 1e-12)  # pixels between neighbouring lines
    wrapped = (phi + np.pi) % (2 * np.pi) - np.pi
    dist = np.abs(wrapped) / (grad + 1e-12)
    line = np.exp(-0.5 * (dist / sigma_px) ** 2)
    atten = np.clip((spacing - min_spacing_px) / fade_px, 0.0, 1.0)
    return line * atten, 0.5 + 0.5 * np.cos(phi)


def _render_buffer(tau, st):
    """Additive float RGB buffer for loop phase tau in [0, 1)."""
    w, dw = st["w"], st["dw"]
    u = w.real
    # Position along the tide, source (u << 0) -> sink (u >> 0), through the palette.
    colour_param = 0.5 + 0.5 * np.tanh(u / st["colour_span"])
    lut = st["lut"]
    rgb = lut[(colour_param * (lut.shape[0] - 1)).astype(np.int64)]  # (H, W, 3)

    lines = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    fill = np.zeros_like(lines)
    for n, c, k, weight, fill_weight in st["layers"]:
        line, soft = _layer(
            w, dw, n, c, k, tau, st["sigma_px"], st["min_spacing_px"], st["fade_px"]
        )
        lines += weight * line
        fill += fill_weight * soft
    # Fill only where lines are not already saturating, and never near the poles.
    fill *= np.clip((st["spacing_ref"] - st["min_spacing_px"]) / st["fade_px"], 0, 1)

    buf = rgb * (lines + st["fill_gain"] * fill)[..., None]
    # Where layers cross, push the core towards white.
    core = np.clip(lines - 1.0, 0, None)
    buf += core[..., None] * st["core_gain"]
    return buf.transpose(2, 0, 1)


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
    # Rebuild the (large) grid fields in the worker rather than pickling them.
    _STATE["w"], _STATE["dw"] = _log_coordinate(state["z"], state["p"], state["q"])
    g_ref = state["layers"][0][1] - 1j * state["layers"][0][0]
    _STATE["spacing_ref"] = 2 * np.pi / (np.abs(g_ref * _STATE["dw"]) / HEIGHT + 1e-12)


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
        buf = _render_buffer(f / LOOP_FRAMES, st)
        image.set_data(
            _tone_map(buf, st["gain"], st["bloom_weight"], st["bloom_sigma"])
        )
        fig.savefig(st["frames_path"] / f"frame{f:04d}.png", facecolor=BG_COLOR)
    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Mobius Tide -- level lines of a Mobius map's logarithmic coordinate,
    streaming and spiralling between its two fixed points, as an exactly
    looping 9:16 clip.

    A loxodromic Mobius transformation with fixed points p and q is conjugate
    to multiplication by a complex number: in the coordinate

        w = log C(z),   C(z) = (z - p) / (z - q),

    it is a plain translation. The circles |C| = const are the Apollonian
    circles around p and q, the lines arg C = const are the circle arcs
    through both, and the level sets of

        phi = Re(g w) = c log|C| + n arg C,   g = c - i n,

    are loxodromes: the spirals a point traces under repeated application of
    the map. Translating phi by 2*pi*k per loop (k integer) is the identity
    on the pattern, so frame LOOP_FRAMES == frame 0 with no crossfade, while
    every line visibly streams from one pole into the other (n = 0, the
    tide), rotates about them (c = 0), or spirals (both).

    Each layer is rendered as thin glowing lines with constant on-screen
    width: phi is holomorphic in z, so |grad phi| = |g C'/C| is exact and the
    distance to the nearest level line is known in pixels. Lines fade out
    where their spacing drops below the pixel grid, so the two poles dissolve
    into soft glows rather than aliasing. Colour is position along the tide
    (log|C|) through the palette; where layers cross, the core is pushed to
    white. Frames are independent, so rendering is spread across worker
    processes and one period is copied REPEATS times.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    # (n, c, k, line weight, fill weight) per layer. n: arcs through the poles
    # per turn (0 = pure tide), c: spiral tightness in log|C|, k: signed
    # phase laps per loop (speed and direction).
    layers = [
        (0, 9.0, 2, 1.0, 1.0),  # the tide: Apollonian circles streaming q -> p
        (7, -3.0, -1, 0.75, 0.0),  # counter-rotating loxodromes
        (3, 5.0, 1, 0.45, 0.0),  # a looser spiral riding with the tide
    ]
    pole_gap = 0.34  # distance between fixed points, in frame heights
    pole_tilt_deg = 18.0  # max random tilt of the p-q axis from vertical
    off_screen = False  # True: push q outside the frame for a whirlpool look
    sigma_px = 0.9  # line half-width in pixels
    min_spacing_px = 4.0  # line spacing below which lines are fully faded
    fade_px = 10.0  # extra spacing over which lines fade back in
    colour_span = 2.2  # log|C| range mapped across the palette
    fill_gain = 0.08  # brightness of the soft cos(phi) tide bands
    core_gain = 0.8  # white push where lines overlap
    exposure_pct = 99.7  # raw density percentile driven to full white
    bloom_weight = 0.45
    bloom_sigma = 7.0

    aspect = WIDTH / HEIGHT
    tilt = np.deg2rad(rng.uniform(-pole_tilt_deg, pole_tilt_deg))
    centre = np.array([aspect / 2, 0.5]) + rng.uniform(-0.03, 0.03, 2)
    axis = 0.5 * pole_gap * np.array([np.sin(tilt), np.cos(tilt)])
    p_xy, q_xy = centre + axis, centre - axis
    if off_screen:
        q_xy = centre - 3.5 * axis
    p = complex(p_xy[0], p_xy[1])
    q = complex(q_xy[0], q_xy[1])
    logger.info(f"fixed points p={p:.3f} q={q:.3f}")

    xs = (np.arange(WIDTH) + 0.5) / HEIGHT
    ys = 1.0 - (np.arange(HEIGHT) + 0.5) / HEIGHT
    z = xs[None, :] + 1j * ys[:, None]

    lut = np.array(
        [
            to_rgb(col)
            for col in LinearSegmentedColormap.from_list("tide", PALETTE, N=512)(
                np.linspace(0, 1, 512)
            )
        ],
        dtype=np.float32,
    )

    state = dict(
        z=z,
        p=p,
        q=q,
        layers=layers,
        sigma_px=sigma_px,
        min_spacing_px=min_spacing_px,
        fade_px=fade_px,
        colour_span=colour_span,
        fill_gain=fill_gain,
        core_gain=core_gain,
        lut=lut,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=str(settings.frames_path),
    )

    # --- one exposure for the whole clip, probed at a few phases ---
    _init_worker(state)
    highs = []
    for tau in np.arange(3) / 3:
        buf = _render_buffer(tau, _STATE)
        lum = buf.max(axis=0)
        highs.append(np.percentile(lum[lum > 0], exposure_pct))
    state["gain"] = 1.7 / max(float(np.max(highs)), 1e-6)
    logger.info(f"gain={state['gain']:.2f}")

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
