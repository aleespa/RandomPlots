"""Bifurcation Bloom -- a strange attractor breathing through its parameter space."""

import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from common.image_processing import ImageProcessingSettings

FPS = 60
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 3  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150
WIDTH, HEIGHT = 1080, 1920  # density grid == output pixels, so 1 cell == 1 pixel
BG_COLOR = "#000000"

# SOLAR_FLARE (colors.palettes) with a near-black toe prepended, so empty space
# reads as ground rather than as the darkest ink, and the densest caustics clip
# to pure white -- the deliberate highlight.
PALETTE = [
    "#000000",
    "#12002b",
    "#3b0764",
    "#7209b7",
    "#f72585",
    "#ff3600",
    "#ff9e00",
    "#ffee32",
    "#ffffff",
]

N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


def _params_at(phase, base, amp, harmonic, offset):
    """The four De Jong parameters at loop phase `phase` in [0, 1).

    Each parameter rides a cosine whose frequency is an *integer* number of
    cycles per loop, so the parameter vector traces a closed Lissajous curve in
    R^4 and returns exactly to its start after LOOP_FRAMES frames.
    """
    return base + amp * np.cos(2 * np.pi * harmonic * phase + offset)


def _density(params, x0, y0, n_iter, burn_in, window, shape, blur):
    """Iterate the De Jong map from a fixed point cloud into a 2D histogram.

    De Jong:  x' = sin(a*y) - cos(b*x),  y' = sin(c*x) - cos(d*y)

    Every point is advanced in lockstep as one vectorised array. After a
    burn-in that pulls the cloud onto the attractor, each subsequent iterate is
    binned, so the image is a Monte-Carlo estimate of the attractor's invariant
    measure rather than a set of drawn orbits.

    The histogram is then pooled with a small Gaussian. This is a variance
    reduction, not a cosmetic softening: the sparse haze away from the caustics
    holds only a couple of counts per pixel, and because the map is chaotic,
    each frame's counts there are an independent draw -- left raw, the haze
    boils between frames instead of drifting. Neighbouring pixels are
    independent samples of a locally smooth measure, so pooling them buys down
    the noise (roughly 4*pi*sigma^2 samples per pixel) at the cost of a slight
    widening of the thinnest filaments.
    """
    a, b, c, d = params
    x_min, x_max, y_min, y_max = window
    h, w = shape
    x, y = x0.copy(), y0.copy()
    hist = np.zeros(h * w, dtype=np.int64)
    sx = w / (x_max - x_min)
    sy = h / (y_max - y_min)

    for i in range(burn_in + n_iter):
        x, y = np.sin(a * y) - np.cos(b * x), np.sin(c * x) - np.cos(d * y)
        if i < burn_in:
            continue
        ix = np.floor((x - x_min) * sx).astype(np.int64)
        iy = np.floor((y - y_min) * sy).astype(np.int64)
        keep = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        hist += np.bincount(iy[keep] * w + ix[keep], minlength=h * w)

    density = hist.reshape(h, w).astype(np.float32)
    return gaussian_filter(density, blur) if blur > 0 else density


def _tone_map(hist, shadow, highlight, gamma, bloom_weight, bloom_sigma):
    """Stretch the density between two anchors in log space, then bloom.

    A single peak-referenced curve does not survive across seeds: attractors
    differ enormously in how concentrated their measure is, and one that
    spreads its samples evenly comes out milky under a curve tuned for one
    with sharp caustics. Anchoring on both ends -- `shadow` (a low percentile
    of the density, driven to black) and `highlight` (a top percentile, driven
    to white) -- normalises contrast rather than just brightness. `gamma`
    shapes the midtones between them.

    Both anchors are measured once over the whole loop, not per frame;
    normalising each frame on its own would make the clip pump.
    """
    v = np.log(np.maximum(hist, shadow) / shadow) / np.log(highlight / shadow)
    v = np.clip(v, 0.0, 1.0) ** gamma
    if bloom_weight > 0:
        v = v + bloom_weight * gaussian_filter(v, bloom_sigma)
    return np.clip(v, 0.0, 1.0)


def _lyapunov(params, iterations=2000):
    """Largest Lyapunov exponent of the map, via the tangent (Jacobian) flow.

    Positive => genuinely chaotic (a filamentary attractor worth looking at);
    <= 0 => the orbit has collapsed onto a fixed point or a short cycle, which
    renders as a handful of dots.
    """
    a, b, c, d = params
    x, y = 0.1, 0.1
    vx, vy = 1.0, 0.0
    total = 0.0
    for i in range(iterations):
        j00 = b * np.sin(b * x)
        j01 = a * np.cos(a * y)
        j10 = c * np.cos(c * x)
        j11 = d * np.sin(d * y)
        vx, vy = j00 * vx + j01 * vy, j10 * vx + j11 * vy
        norm = np.hypot(vx, vy)
        if norm == 0 or not np.isfinite(norm):
            return -np.inf
        vx, vy = vx / norm, vy / norm
        if i > 100:
            total += np.log(norm)
        x, y = np.sin(a * y) - np.cos(b * x), np.sin(c * x) - np.cos(d * y)
    return total / (iterations - 100)


def _probe_samples(params, x0, y0, burn_in, keep):
    """A cheap scatter of points on the attractor, for framing and fill tests."""
    a, b, c, d = params
    x, y = x0.copy(), y0.copy()
    xs, ys = [], []
    for i in range(burn_in + keep):
        x, y = np.sin(a * y) - np.cos(b * x), np.sin(c * x) - np.cos(d * y)
        if i >= burn_in:
            xs.append(x.copy())
            ys.append(y.copy())
    return np.concatenate(xs), np.concatenate(ys)


def _cover_window(centre, extent, zoom):
    """The largest 9:16 window that still sits inside a box of size `extent`.

    "Cover", not "contain": the attractor is cropped by the frame edges rather
    than floating inside them with dead space, which is what a vertical feed
    wants. `zoom` > 1 crops in further.
    """
    height = min(extent[1], extent[0] * HEIGHT / WIDTH) / zoom
    width = height * WIDTH / HEIGHT
    return (
        centre[0] - width / 2,
        centre[0] + width / 2,
        centre[1] - height / 2,
        centre[1] + height / 2,
    )


def _frame_fill(x, y, window, nx=54, ny=96):
    """Fraction of a coarse grid over the *rendered window* that gets ink.

    Measured in frame space rather than in the plane, so it answers the
    question that actually matters -- does the composition fill the frame at
    this phase, or is half of it empty?
    """
    x_min, x_max, y_min, y_max = window
    ix = np.floor((x - x_min) / (x_max - x_min) * nx).astype(np.int64)
    iy = np.floor((y - y_min) / (y_max - y_min) * ny).astype(np.int64)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.unique(iy[keep] * nx + ix[keep]).size / (nx * ny)


def _search_parameter_loop(
    rng, phases, probe_x, probe_y, burn_in, amp_scale, zoom, min_fill, attempts=300
):
    """Rejection-sample a parameter loop that is chaotic *and* fills the frame.

    Two gates, cheapest first. A positive Lyapunov exponent at every sampled
    phase keeps the whole loop genuinely chaotic. Then the loop is framed and
    the frame itself is measured: an attractor can be perfectly chaotic and
    still be a thin ribbon that leaves most of a 9:16 frame black, and ink
    coverage measured in frame space is the only test that catches that.

    Returns the best candidate found rather than only an accepted one, so no
    seed can leave the caller with nothing.
    """
    best = None
    for attempt in range(1, attempts + 1):
        candidate = dict(
            base=rng.uniform(-2.6, 2.6, 4),
            amp=amp_scale * rng.uniform(0.5, 1.0, 4),
            harmonic=rng.choice([1, 1, 1, 2, 2, 3], 4),
            offset=rng.uniform(0, 2 * np.pi, 4),
        )
        phase_params = [_params_at(p, **candidate) for p in phases]
        if any(_lyapunov(p) < 0.08 for p in phase_params):
            continue

        # Frame on the median phase, not the union of all of them: the union
        # zooms out far enough to hold the largest phase, which leaves every
        # other phase under-filled.
        clouds = [
            _probe_samples(p, probe_x, probe_y, burn_in, 20) for p in phase_params
        ]
        boxes = np.array(
            [
                [
                    np.percentile(x, 0.2),
                    np.percentile(y, 0.2),
                    np.percentile(x, 99.8),
                    np.percentile(y, 99.8),
                ]
                for x, y in clouds
            ]
        )
        median_box = np.median(boxes, axis=0)
        centre = (
            (median_box[0] + median_box[2]) / 2,
            (median_box[1] + median_box[3]) / 2,
        )
        extent = (median_box[2] - median_box[0], median_box[3] - median_box[1])
        window = _cover_window(centre, extent, zoom)
        fill = min(_frame_fill(x, y, window) for x, y in clouds)

        if best is None or fill > best["fill"]:
            best = dict(**candidate, window=window, fill=fill, attempt=attempt)
        if fill >= min_fill:
            break

    if best is None:
        raise RuntimeError("no chaotic parameter loop found; try another seed")
    return best


def _init_worker(
    base,
    amp,
    harmonic,
    offset,
    x0,
    y0,
    n_iter,
    burn_in,
    window,
    density_blur,
    shadow,
    highlight,
    gamma,
    bloom_weight,
    bloom_sigma,
    frames_path,
):
    """Runs once per worker process: the fixed point cloud and the parameter
    path are large and read-only, so they cross the process boundary once
    rather than being re-pickled with every chunk of frames."""
    _STATE.update(
        base=base,
        amp=amp,
        harmonic=harmonic,
        offset=offset,
        x0=x0,
        y0=y0,
        n_iter=n_iter,
        burn_in=burn_in,
        window=window,
        density_blur=density_blur,
        shadow=shadow,
        highlight=highlight,
        gamma=gamma,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=Path(frames_path),
        cmap=LinearSegmentedColormap.from_list("bifurcation_bloom", PALETTE, N=1024),
    )


def _render_chunk(frame_indices):
    """Render a contiguous run of frames, reusing one Figure/Axes/AxesImage
    across all of them -- only the pixel data changes frame to frame."""
    st = _STATE
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")
    image = ax.imshow(
        np.zeros((HEIGHT, WIDTH), dtype=np.float32),
        cmap=st["cmap"],
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    for f in frame_indices:
        params = _params_at(
            f / LOOP_FRAMES, st["base"], st["amp"], st["harmonic"], st["offset"]
        )
        hist = _density(
            params,
            st["x0"],
            st["y0"],
            st["n_iter"],
            st["burn_in"],
            st["window"],
            (HEIGHT, WIDTH),
            st["density_blur"],
        )
        image.set_data(
            _tone_map(
                hist,
                st["shadow"],
                st["highlight"],
                st["gamma"],
                st["bloom_weight"],
                st["bloom_sigma"],
            )
        )
        fig.savefig(st["frames_path"] / f"frame{f:04d}.png", facecolor=BG_COLOR)

    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Bifurcation Bloom -- a De Jong attractor breathing in and out of chaos,
    as a perfectly looping 9:16 clip.

    The De Jong quadratic map

        x' = sin(a*y) - cos(b*x)
        y' = sin(c*x) - cos(d*y)

    has a strange attractor whose shape depends violently on (a, b, c, d).
    Rather than animating a single attractor, this walks the *parameter* vector
    around a closed Lissajous curve in R^4,

        p_i(t) = base_i + amp_i * cos(2*pi*k_i*t + phi_i),   k_i integer,

    so the family of attractors passes through folds and bifurcations and then
    returns exactly to where it began: frame LOOP_FRAMES is identical to frame
    0, an exact loop with no crossfade.

    Each frame is a density estimate, not a drawing. A fixed cloud of points is
    iterated in lockstep; after a burn-in that lands them on the attractor,
    every iterate is binned into a 1080x1920 histogram (one cell per output
    pixel), which converges to the attractor's invariant measure. The same
    initial cloud is reused for every frame, so the Monte-Carlo noise is
    correlated between frames and evolves rather than sizzling.

    Density is stretched between two anchors in log space -- a low percentile
    driven to black and a top percentile driven to white -- both measured once
    across the whole loop, so contrast (not just brightness) is normalised and
    the clip does not pump. The densest caustics clip to white, and a blurred
    copy is added back as bloom.

    The parameter path is rejection-sampled up front: a candidate is kept only
    if the largest Lyapunov exponent stays positive at every phase of the loop
    (no collapse to a fixed point or a short cycle part-way round) and the
    attractor's ink covers enough of the 9:16 frame at every phase.

    Frames are independent given the parameter path, so rendering is split
    across worker processes, and -- the loop being exact -- only one period is
    rendered and then copied REPEATS times to pad the delivered clip.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    n_points = 420_000  # samples per iteration: grain vs. silk
    n_iter = 100  # binned iterations per frame (n_points * n_iter samples)
    burn_in = 30  # iterations discarded while the cloud lands on the attractor
    amp_scale = 0.30  # radius of the parameter loop: small = one attractor
    # "species" breathing, large = tears through bifurcations
    shadow_pct = 76  # density percentile driven to black: raising it lifts
    # contrast and thins the haze back to bare filaments
    highlight_pct = 99.9  # density percentile driven to white (the bright core)
    gamma = 1.2  # >1 darkens the midtones: smoke -> engraving
    bloom_weight = 0.22
    bloom_sigma = 9.0
    density_blur = 1.1  # counts pooled per pixel: the anti-boil knob (see _density)
    zoom = 1.04  # >1 crops in past the attractor's bounding box
    min_fill = 0.70  # required ink coverage of the frame, at every phase
    n_probe_phases = 8  # phases checked when validating / measuring the loop

    # Fixed, deterministic initial cloud -- identical on every frame.
    x0 = rng.uniform(-2, 2, n_points).astype(np.float32)
    y0 = rng.uniform(-2, 2, n_points).astype(np.float32)
    probe_x, probe_y = x0[:20_000], y0[:20_000]
    phases = np.arange(n_probe_phases) / n_probe_phases

    best = _search_parameter_loop(
        rng, phases, probe_x, probe_y, burn_in, amp_scale, zoom, min_fill
    )
    base, amp = best["base"], best["amp"]
    harmonic, offset = best["harmonic"], best["offset"]
    window = best["window"]
    if best["fill"] < min_fill:
        logger.warning(
            f"no loop reached fill>={min_fill}; using the best found ({best['fill']:.2f})"
        )
    logger.info(f"parameter loop accepted after {best['attempt']} attempt(s)")
    logger.info(f"base={np.round(base, 3)} amp={np.round(amp, 3)} k={harmonic}")
    logger.info(f"window={np.round(window, 3)} min frame fill={best['fill']:.2f}")

    # --- one pair of tone anchors for the whole clip ---
    # Shadow is taken as the median across phases (a stable black point) and
    # highlight as the maximum (so no phase blows out); measuring per frame
    # would make the clip pump.
    shadows, highlights = [], []
    for phase in phases:
        params = _params_at(phase, base, amp, harmonic, offset)
        hist = _density(
            params, x0, y0, n_iter, burn_in, window, (HEIGHT, WIDTH), density_blur
        )
        shadows.append(np.percentile(hist, shadow_pct))
        highlights.append(np.percentile(hist, highlight_pct))
    shadow = max(float(np.median(shadows)), 0.3)
    highlight = max(float(np.max(highlights)), shadow * 10)
    logger.info(
        f"tone anchors: shadow={shadow:.2f} highlight={highlight:.1f} counts/px"
    )

    frames_path = settings.frames_path
    n_workers = min(N_WORKERS, LOOP_FRAMES)
    chunks = np.array_split(np.arange(LOOP_FRAMES), n_workers * 4)

    logger.info(f"rendering {LOOP_FRAMES} frames across {n_workers} workers")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            base,
            amp,
            harmonic,
            offset,
            x0,
            y0,
            n_iter,
            burn_in,
            window,
            density_blur,
            shadow,
            highlight,
            gamma,
            bloom_weight,
            bloom_sigma,
            str(frames_path),
        ),
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    # The loop is exact, so pad the clip by copying the rendered period rather
    # than re-rendering it: frame{0..LOOP_FRAMES-1}, repeated REPEATS times and
    # renumbered into a single contiguous frame%04d.png sequence.
    for repeat in range(1, REPEATS):
        for i in range(LOOP_FRAMES):
            shutil.copyfile(
                frames_path / f"frame{i:04d}.png",
                frames_path / f"frame{repeat * LOOP_FRAMES + i:04d}.png",
            )

    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
