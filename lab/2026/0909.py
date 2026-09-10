"""
Hopf Weave

Every ring here is a circle drawn in four dimensions, and no two of them can
ever be pulled apart.

    h(z₁, z₂) = (2 z₁ z̄₂, |z₁|² − |z₂|²)

The Hopf map sends the sphere in four dimensions onto the ordinary round one,
and everything landing on a single point of it is a whole circle. Those circles
are the fibres. Any two are linked exactly once, like two rings of a chain, no
matter how far the picture is stretched or turned.

Four rings of latitude on the ordinary sphere choose the fibres drawn here, a
colour each, so every colour sweeps out its own doughnut and the four nest
inside one another. Flattening four dimensions down to three takes circles to
circles: each curve on screen is a genuine round circle in space, seen in
perspective, and the crossings you can follow are real links.

The tumble is a rotation of the four-dimensional sphere itself, running round a
closed path, so the last frame turns back into the first.

Made with Python, numpy, scipy and matplotlib. 60 fps, exact loop.

#generativeart #creativecoding #hopffibration #topology #mathart #python
#geometry #4d #linkedrings #fibration
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

from colors.palettes import HOLOGRAPHIC
from common.image_processing import ImageProcessingSettings

# RP_FPS=4 in the environment gives a quick 32-frame test loop with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 2  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150
WIDTH, HEIGHT = 1080, 1920
ASPECT = WIDTH / HEIGHT
BG_COLOR = "#000000"

# HOLOGRAPHIC runs jade -> blue -> violet -> pink -> orange -> amber. The
# fibre's latitude on the base sphere picks its colour, so rings that came
# from the same place match, and the nested tori land a whole hue apart.
PALETTE = HOLOGRAPHIC

N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


def _base_points(rings, fibres_per_ring, offsets):
    """
    Base points on S^2: `fibres_per_ring` points evenly spaced round each latitude.

    `offsets` rotates each latitude's fibres in phi, which is what stops the
    rings from lining up into spokes and is the main per-seed variation.

    Returns (theta, phi, ring_fraction), one entry per fibre. ring_fraction is
    the latitude's position in [0, 1] and drives colour.
    """
    theta, phi, fraction = [], [], []
    for i, (ring_theta, offset) in enumerate(zip(rings, offsets)):
        for j in range(fibres_per_ring):
            theta.append(ring_theta)
            phi.append(2 * np.pi * j / fibres_per_ring + offset)
            fraction.append(i / max(len(rings) - 1, 1))
    return (
        np.array(theta),
        np.array(phi),
        np.array(fraction),
    )


def _fibres(theta, phi, n_samples):
    """
    The Hopf fibre over each base point, as a closed circle in S^3 subset C^2.

    The fibre over the point of S^2 with spherical coordinates (theta, phi) is

        (z1, z2) = (cos(theta/2) e^{i(phi + psi)}, sin(theta/2) e^{i psi}),

    for psi in [0, 2 pi). Feeding that through the Hopf map
    h(z1, z2) = (2 z1 conj(z2), |z1|^2 - |z2|^2) gives (sin theta e^{i phi},
    cos theta) for every psi, which is the base point again -- so the whole
    circle really does sit over one point.

    Returns z1, z2 of shape (fibres, n_samples).
    """
    psi = 2 * np.pi * np.arange(n_samples) / n_samples
    z1 = np.cos(theta / 2)[:, None] * np.exp(1j * (phi[:, None] + psi[None, :]))
    z2 = np.sin(theta / 2)[:, None] * np.exp(1j * psi[None, :])
    return z1, z2


def _view_matrix(pitch_deg, yaw_deg):
    """A fixed camera orientation: yaw about the vertical, then pitch."""
    a, b = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    yaw = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pitch = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return pitch @ yaw


def _loop_pose(phase, st):
    """
    The S^3 rotation angle and the camera for a loop phase in [0, 1).

    The rotation is the one-parameter subgroup of SU(2)

        (z1, z2) -> (z1 cos a - z2 sin a, z1 sin a + z2 cos a),

    which acts on the base sphere as a rotation by 2a. Running a from 0 to pi
    is a full turn of the base sphere and closes the loop, but it also drags
    every ring of fibres across the projection pole, where the picture blows up
    and stops being composed. So the default is instead a *rocking* closed path,
    a = A sin(2 pi phase), whose amplitude is chosen to keep the rings clear of
    the pole -- still exactly periodic, being a closed path in the group rather
    than a full circuit of the subgroup. A whole number of camera turns per loop
    then supplies continuous motion, so the rocking never reads as a rewind.
    """
    if st["rock_amplitude"] > 0:
        alpha = st["rock_amplitude"] * np.sin(2 * np.pi * phase)
    else:
        alpha = np.pi * phase
    yaw = st["yaw_deg"] + 360.0 * st["camera_turns"] * phase
    return alpha, _view_matrix(st["pitch_deg"], yaw)


def _project(z1, z2, alpha, view, st):
    """
    Rotate the fibres through S^3 by alpha, project to R^3, then to the screen.

    Stereographic projection from the pole (0, 0, 0, 1) of S^3 sends each fibre
    -- a circle in S^3 -- to a circle in R^3, except for a fibre through the
    pole itself, which becomes a straight line. The camera then projects that
    to the screen.

    Returns (x, y, depth, valid), each of shape (fibres, samples).
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    w1, w2 = ca * z1 - sa * z2, sa * z1 + ca * z2
    x4 = w2.imag
    denom = 1.0 - x4  # in [0, 2], zero only at the projection pole
    valid = denom > st["pole_epsilon"]
    safe = np.where(valid, denom, 1.0)
    p = np.stack([w1.real / safe, w1.imag / safe, w2.real / safe])

    q = np.einsum("ij,jab->iab", view, p)
    camera_gap = st["camera_distance"] - q[2]
    valid &= camera_gap > st["near_plane"]
    gap = np.where(valid, camera_gap, 1.0)
    return (
        st["focal"] * q[0] / gap,
        st["focal"] * q[1] / gap,
        q[2],
        valid,
    )


def _runs(keep, chunk):
    """
    Cut one fibre's drawable segments into the polylines that will stroke it.

    `keep` is a circular mask over the segments of a single fibre. Each entry
    returned is a half-open range (start, stop) of segment indices, so the
    polyline drawing it has vertices start..stop inclusive, counted modulo the
    sample count. An unbroken fibre gives one span covering the whole circle,
    which strokes as a closed path with no caps at all.

    A span is then cut every `chunk` segments -- `chunk` of 0 meaning never --
    only because opacity and width vary along a fibre and one stroke carries one
    of each. Consecutive chunks share their end vertex and are drawn with butt
    caps, so they abut exactly rather than overlapping.

    That is the whole trick behind a solid line. A stroked path is composited
    once however much it self-overlaps, so a long polyline is evenly
    transparent; separate strokes laid end to end blend twice wherever their
    caps meet, and at one stroke per sample that beading is what turns a fibre
    into a visible string of beads rather than a line.
    """
    n = keep.size
    if keep.all():
        spans = [(0, n)]  # the fibre closes on itself
    elif not keep.any():
        return []
    else:
        # Rotate a gap to the front so that no run straddles the wrap point.
        gap = int(np.argmin(keep))
        k = np.roll(keep, -gap)
        starts = np.flatnonzero(k & ~np.roll(k, 1)) + gap
        stops = np.flatnonzero(k & ~np.roll(k, -1)) + gap + 1
        spans = list(zip(starts.tolist(), stops.tolist()))
    if chunk <= 0:
        return spans
    return [
        (c, min(c + chunk, stop))
        for start, stop in spans
        for c in range(start, stop, chunk)
    ]


def _smooth(a, window):
    """
    Circular moving average along each fibre.

    Shading is constant within a stroke, so wherever it varies faster than the
    stroke length the joins show as steps. Smoothing it over a few strokes'
    worth of samples is what makes consecutive strokes agree at their shared
    vertex; it costs a little contrast in the highlights and nothing else.
    """
    if window <= 1:
        return a
    n = a.shape[1]
    kernel = np.ones(min(window, n)) / min(window, n)
    wide = np.concatenate([a, a, a], axis=1)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = np.convolve(wide[i], kernel, mode="same")[n : 2 * n]
    return out


def _bundle(keep, verts, per_segment, colour, chunk):
    """
    Gather one stroke per chunk of every fibre, sorted back to front.

    `per_segment` is (depth, width, alpha, heat) over the segments; each stroke
    takes the mean of its own chunk, so the chunk length trades how finely those
    follow the fibre against how many joins the fibre is broken at.
    """
    paths, fibre, means = [], [], []
    for i in range(keep.shape[0]):
        for start, stop in _runs(keep[i], chunk):
            paths.append(verts[i, start : stop + 1])
            fibre.append(i)
            means.append([a[i, start:stop].mean() for a in per_segment])

    if not paths:
        return [], np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)

    means = np.asarray(means)
    order = np.argsort(-means[:, 0])  # far first, so near fibres draw over them
    return (
        [paths[k] for k in order],
        colour[np.asarray(fibre)[order], :3],
        means[order, 2],
        means[order, 1],
        means[order, 3],
    )


def _paths(phase, st):
    """
    Build the strokes for a loop phase in [0, 1), one bundle per glow pass.

    Each fibre is stroked as a handful of long polylines rather than as its
    hundreds of individual segments, so that it reads as one solid line -- see
    `_runs` for why that matters. The broad glow passes take a whole fibre per
    stroke: they are wide enough that a step in opacity between two chunks would
    show as a rectangle hanging off the line.

    Brightness follows conservation of ink: a fibre carries a fixed amount per
    unit of its own parameter psi, so a stretch of it spread over many pixels is
    correspondingly fainter. That is what makes a fibre swinging towards the
    projection pole flare out into a giant arc and fade, instead of flashing.
    A second factor is plain aerial perspective -- near strokes read brighter
    and heavier than far ones, which is what separates the nested tori. Both are
    measured per segment and averaged over each stroke, which is the only reason
    the fibres are cut into more than one stroke at all.

    Returns one (paths, rgb, alpha, width, heat) per glow pass, where heat in
    [0, 1] is how strongly a stroke should be pushed towards a white core.
    """
    alpha, view = _loop_pose(phase, st)
    x, y, depth, valid = _project(st["z1"], st["z2"], alpha, view, st)
    # Close each fibre by wrapping the last sample back onto the first.
    x1, y1 = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    pair_valid = valid & np.roll(valid, -1, axis=1)
    length = np.hypot(x1 - x, y1 - y)
    keep = pair_valid & (length < st["max_segment"])

    seg_depth = 0.5 * (depth + np.roll(depth, -1, axis=1))
    stretch = np.maximum(length, 1e-9) / st["reference_length"]
    ink = np.clip(stretch ** -st["ink_exponent"], 0.0, 1.0)

    # Nearer segments read heavier and brighter: both follow the perspective scale.
    scale = st["camera_distance"] / np.maximum(
        st["camera_distance"] - seg_depth, st["near_plane"]
    )
    scale = np.clip(scale / st["reference_scale"], 0.35, 2.2)
    gain = ink * scale ** st["depth_exponent"]
    scale, gain = (_smooth(a, st["smooth_samples"]) for a in (scale, gain))

    # Two copies of every fibre laid end to end, so a chunk that wraps past the
    # last sample is still one contiguous slice. Vertices need one more on top.
    xy = np.stack([x, y], axis=-1)
    verts = np.concatenate([xy, xy, xy[:, :1]], axis=1)
    per_segment = tuple(
        np.concatenate([a, a], axis=1)
        for a in (
            seg_depth,
            st["line_width"] * scale,
            np.clip(st["alpha"] * gain, 0.0, 1.0),
            np.clip(st["core_whiteness"] * gain, 0.0, 1.0),
        )
    )

    # Passes sharing a chunk length share the strokes built for it.
    bundles = {}
    for _, _, _, chunk in st["glow_passes"]:
        if chunk not in bundles:
            bundles[chunk] = _bundle(keep, verts, per_segment, st["colour"], chunk)
    return [bundles[chunk] for _, _, _, chunk in st["glow_passes"]]


def _bloom(rgb, weight, sigma):
    """Add a blurred copy of the frame back over itself, so the strokes glow."""
    if weight <= 0:
        return rgb
    glow = np.stack([gaussian_filter(ch, sigma) for ch in rgb.transpose(2, 0, 1)])
    return np.clip(rgb + weight * glow.transpose(1, 2, 0), 0.0, 1.0)


def _init_worker(state):
    _STATE.update(state)
    _STATE["frames_path"] = Path(state["frames_path"])
    # Rebuild the fibre samples in the worker rather than pickling them.
    _STATE["z1"], _STATE["z2"] = _fibres(
        state["theta"], state["phi"], state["n_samples"]
    )


def _render_chunk(frame_indices):
    """Render a run of frames, reusing one Figure and its three glow layers."""
    st = _STATE
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-st["half_width"], st["half_width"])
    ax.set_ylim(-st["half_height"], st["half_height"])
    ax.axis("off")
    layers = []
    for width_scale, alpha_scale, whiteness, _ in st["glow_passes"]:
        # Butt caps: consecutive chunks of one fibre must abut, never
        # overlap, or every join blends twice and beads the line.
        collection = LineCollection([], capstyle="butt", joinstyle="round")
        ax.add_collection(collection)
        layers.append((collection, width_scale, alpha_scale, whiteness))

    for f in frame_indices:
        bundles = _paths(f / LOOP_FRAMES, st)
        for (collection, width_scale, alpha_scale, whiteness), bundle in zip(
            layers, bundles
        ):
            seg, rgb, alpha, width, heat = bundle
            mix = (whiteness * heat)[:, None]
            rgba = np.empty((len(seg), 4))
            rgba[:, :3] = rgb * (1.0 - mix) + mix  # push the core towards white
            rgba[:, 3] = np.clip(alpha * alpha_scale, 0.0, 1.0)
            collection.set_segments(seg)
            collection.set_color(rgba)
            collection.set_linewidth(width * width_scale)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
        plt.imsave(
            st["frames_path"] / f"frame{f:04d}.png",
            _bloom(frame, st["bloom_weight"], st["bloom_sigma"]),
        )
    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Hopf Weave -- the fibres of the Hopf fibration, drawn as interlocking rings
    and tumbled through a rotation of S^3, as an exactly looping 9:16 clip.

    The Hopf map h: S^3 -> S^2, h(z1, z2) = (2 z1 conj(z2), |z1|^2 - |z2|^2),
    sends the unit sphere of C^2 onto the ordinary sphere, and the preimage of
    every point is a great circle. Those circles are the fibres, and their
    defining property is that any two of them are linked exactly once: no two
    can be pulled apart, however far the picture is deformed. The base points
    here are a handful of latitudes on S^2, each carrying an evenly spaced ring
    of fibres, so the fibres over one latitude sweep out a torus and the tori
    nest inside one another.

    Stereographic projection from the pole (0, 0, 0, 1) drops S^3 into R^3, and
    it takes circles to circles: every fibre drawn here is a genuine round
    circle in space, except for one passing exactly through the pole, which
    becomes a straight line. Colour is the fibre's latitude on the base sphere
    through the palette, so rings sharing an origin share a colour and the
    linking between differently coloured rings is legible.

    The motion is the one-parameter subgroup of SU(2)

        (z1, z2) -> (z1 cos a - z2 sin a, z1 sin a + z2 cos a),

    which acts on the base sphere as a rotation by 2a about a horizontal axis.
    Running a from 0 to pi turns the base sphere once and would close the loop
    on its own, but it also drags every ring across the projection pole, where
    the picture blows up and stops being composed. So a instead follows a
    rocking closed path, a = A sin(2 pi phase), with A small enough to keep the
    rings clear of the pole -- still exactly periodic, being a closed path in
    the group rather than a full circuit of the subgroup -- while a whole number
    of camera turns per loop supplies the continuous motion, so the rocking
    never reads as a rewind. Both parts return exactly at phase 1, and the
    sample count round each fibre is even, so the last frame closes onto the
    first with no crossfade.

    Rings that do swing towards the projection pole swell and unwrap; brightness
    is scaled by the inverse of how far each segment is stretched on screen,
    conserving ink, so those moments flare and fade rather than blowing out.
    Each fibre is stroked as a few long polylines rather than as its hundreds
    of individual samples: a stroked path is composited once however much it
    self-overlaps, so a polyline is evenly transparent, whereas short strokes
    laid end to end blend twice at every cap and bead a translucent line into a
    dotted one. The strokes are depth sorted and drawn in three passes -- broad
    and faint, medium, then a thin bright core -- and the finished frame has a
    blurred copy of itself added back over the top, so the strokes glow.

    Frames are independent, so rendering is spread across worker processes and
    one period is copied REPEATS times.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    n_rings = 4  # latitudes on the base sphere; each is one nested torus
    fibres_per_ring = 16  # fibres drawn per latitude
    ring_span_deg = (46.0, 104.0)  # first and last latitude, kept off the poles
    # Even, so that the alpha = pi half-turn maps the samples onto themselves
    # and the last frame is bit-for-bit the first.
    n_samples = 480  # samples round each fibre circle
    chunk_samples = 24  # samples per core stroke; longer strokes, fewer joins
    smooth_samples = 72  # samples the shading is averaged over, to hide the joins
    rock_amplitude = 0.14 * np.pi  # S^3 rocking amplitude; 0 = full subgroup sweep
    camera_turns = 1  # whole camera turns per loop, for continuous motion
    camera_distance = 4.6  # camera distance from the origin, in R^3 units
    focal = 2.35  # focal length; with half_height below, sets the framing
    pitch_deg = -24.0  # fixed camera tilt
    yaw_deg = 18.0
    half_height = 1.35  # vertical half-extent of the frame in projected units
    line_width = 1.9  # pt, before the perspective scaling
    alpha = 0.95  # base stroke opacity, before the ink and depth corrections
    ink_exponent = 1.0  # 1.0 conserves ink exactly; lower keeps blow-ups brighter
    depth_exponent = 1.6  # how hard near strokes outshine far ones
    core_whiteness = 0.34  # how far the brightest strokes are pushed to white
    max_segment = 0.9  # drop segments longer than this (a blown-up fibre's chord)
    pole_epsilon = 1e-3  # 1 - x4 below this counts as sitting on the pole
    near_plane = 0.35
    # (width x, alpha x, whiteness x, samples per stroke) -- broad haze, body,
    # then a hot thin core. 0 samples means a whole fibre in one stroke, which
    # is what the two wide passes need: they are broad enough that a step in
    # opacity between chunks would read as a rectangle hanging off the line.
    glow_passes = (
        (10.0, 0.075, 0.0, 0),
        (3.4, 0.2, 0.25, 0),
        (1.0, 1.0, 1.0, chunk_samples),
    )
    bloom_weight = 0.55  # how much of the blurred frame is added back as glow
    bloom_sigma = 10.0  # px, the glow's spread
    yaw_jitter_deg = 12.0  # per-seed variation of the camera
    pitch_jitter_deg = 6.0
    colour_span = (0.0, 1.0)  # slice of the palette the latitudes map onto

    rings = np.deg2rad(np.linspace(*ring_span_deg, n_rings))
    offsets = rng.uniform(0.0, 2 * np.pi / fibres_per_ring, n_rings)
    theta, phi, fraction = _base_points(rings, fibres_per_ring, offsets)
    logger.info(f"{theta.size} fibres over {n_rings} latitudes")

    lut = LinearSegmentedColormap.from_list("weave", PALETTE, N=512)
    # Every stop of HOLOGRAPHIC is fully saturated and bright enough to read as a
    # line, so the latitudes take the whole ramp and each torus gets its own hue.
    colour = lut(colour_span[0] + fraction * (colour_span[1] - colour_span[0]))

    state = dict(
        theta=theta,
        phi=phi,
        n_samples=n_samples,
        chunk_samples=chunk_samples,
        smooth_samples=smooth_samples,
        colour=colour,
        rock_amplitude=rock_amplitude,
        camera_turns=camera_turns,
        pitch_deg=pitch_deg + rng.uniform(-1, 1) * pitch_jitter_deg,
        yaw_deg=yaw_deg + rng.uniform(-1, 1) * yaw_jitter_deg,
        camera_distance=camera_distance,
        focal=focal,
        near_plane=near_plane,
        pole_epsilon=pole_epsilon,
        half_height=half_height,
        half_width=half_height * ASPECT,
        line_width=line_width,
        alpha=alpha,
        ink_exponent=ink_exponent,
        depth_exponent=depth_exponent,
        core_whiteness=core_whiteness,
        max_segment=max_segment,
        glow_passes=glow_passes,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=str(settings.frames_path),
        reference_length=1.0,
        reference_scale=1.0,
    )

    # --- ink and depth references for the whole clip, probed around the loop ---
    _init_worker(state)
    lengths, scales = [], []
    for probe in np.linspace(0, 1, 6, endpoint=False):
        alpha, view = _loop_pose(probe, _STATE)
        x, y, depth, valid = _project(_STATE["z1"], _STATE["z2"], alpha, view, _STATE)
        pair = valid & np.roll(valid, -1, axis=1)
        step = np.hypot(np.roll(x, -1, axis=1) - x, np.roll(y, -1, axis=1) - y)
        lengths.append(np.median(step[pair]))
        scales.append(
            np.median(
                camera_distance / np.maximum(camera_distance - depth[valid], near_plane)
            )
        )
    state["reference_length"] = float(np.median(lengths))
    state["reference_scale"] = float(np.median(scales))
    _STATE.update(state)
    logger.info(
        f"reference segment length {state['reference_length']:.5f}, "
        f"perspective scale {state['reference_scale']:.3f}"
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

    gc.collect()
    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
