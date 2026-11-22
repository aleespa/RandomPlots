"""
The Mandelbrot Set Is Hiding Inside Itself

A zoom into the Mandelbrot fractal, three hundred million times closer, down
one of the spirals of Seahorse Valley. At the bottom of the dive waits a
perfect copy of the whole Mandelbrot set, cardioid, bulbs and antenna,
turned to stand exactly as the first frame did.

Every pixel is a number c. Start at z = 0 and repeat z ↦ z² + c: if z never
flies away, c is in the set (black); the colours count how fast it escapes.
That one rule hides infinitely many miniature Mandelbrot sets along its
filaments, each dressed in its own hair of spirals. If the whole set were a
football field wide, this copy would be smaller than a bacterium.

Listen: every time the counter gains a zero, a gamelan bell rings one step
lower, sinking with the dive, until the home chord lands on the copy.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#mandelbrotzoom #fractal #mathart #generativeart #randomplots
"""

import gc

import cv2
import mpmath as mp
import numpy as np
from loguru import logger
from numba import njit, prange

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Zoom 300 million times. Find this again.'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds, a one-shot
FRAME_COUNT = frame_count(DURATION)

# --- The target -------------------------------------------------------------------
# The nucleus of a period-27 minibrot in Seahorse Valley (refined to 50 digits).
NUCLEUS = ("-0.7746880598389581374206918", "-0.137407864636067237829304")
PERIOD = 27
LOCAL_CENTRE = -0.5  # frame centre, in the minibrot's own coordinates (as at start)

# --- The dive ---------------------------------------------------------------------
START_CENTRE = -0.5  # the first frame's centre (the whole set)
START_PX = 2.2e-3  # complex units per pixel in the first frame
ZOOM_END = 13.0  # s: the zoom lands; the rest is a hold on the copy
EASE_IN, EASE_OUT = 0.5, 3.5  # s of acceleration and of landing

# --- The exponential map ------------------------------------------------------------
COLUMNS = 4096  # angular samples; rows are spaced by the same angle in log r
MAX_ITER = 9000
SETTLED = 1e-12  # relative squared distance at which an orbit counts as periodic
ESCAPE = 1e8  # |z|² bailout
CORE = 0.15  # radius (in minibrot units) inside which everything is interior
LEVELS = 9  # mip levels for the oversampled middle of the screen

# --- Look -------------------------------------------------------------------------
BACKGROUND = "#05030b"
PALETTE = [
    "#05030b",
    "#1d1147",
    "#4b2a8a",
    "#b6407f",
    "#f08a5d",
    "#ffd9a0",
    "#fff6e6",
    "#f08a5d",
    "#8e2f6e",
    "#2a165c",
    "#05030b",
]
CYCLES = 2  # palette cycles per 27-fold change in escape time
PHASE = 0.0  # palette phase of the far field: dark
LOG_NU_MAX = np.log(MAX_ITER)  # texture quantisation of log escape time
FLOOR_PERCENTILE = 2  # the "far field" escape time of a ring of the map
FLOOR_SMOOTH = 600  # rows (± 0.9 e-folds) of smoothing for it
FAR_RADIUS = 1000  # px: where on screen the far field is measured
INTERIOR = "#000000"
TEXT, TEXT_Y, TEXT_SIZE = "#fff6e6", 1540, 76
SHADOW = 14  # px: dark halo behind the counter

# --- Sound ------------------------------------------------------------------------
LANDING = 0.985  # fraction of the dive at which the home chord lands
STYLE = "gamelan"  # a key of SOUNDS; an --audio-only tag naming one overrides it
SOUNDS = {  # tag: (lead voice, scale, melody roots, bed voice)
    "gamelan": (audio.bell, "pelog", range(60, 68), "choir"),
    "handpan": (audio.handpan, "minor", range(57, 65), "pad"),
    "handpan_pelog": (audio.handpan, "pelog", range(55, 63), "choir"),
}


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def exp_map(ref, offset, log_r0, step, rows, cols, max_iter, out):
    """
    Smooth escape time of c = centre + r e^{iθ} on a log-polar grid, by perturbation
    around the periodic reference orbit `ref` (with rebasing), -1 for interior.
    """
    period = ref.shape[0] - 1
    for i in prange(rows):
        r = np.exp(log_r0 - i * step)
        for j in range(cols):
            theta = 2 * np.pi * j / cols
            dc = offset + r * complex(np.cos(theta), np.sin(theta))
            dz = 0j
            m = 0
            nu = -1.0
            saved = 0j
            events = 0
            next_save = 1
            for n in range(max_iter):
                dz = (2 * ref[m] + dz) * dz + dc
                m += 1
                if m == period:
                    m = 0
                z = ref[m] + dz
                r2 = z.real * z.real + z.imag * z.imag
                if r2 > ESCAPE:
                    nu = n + 2 - np.log2(0.5 * np.log(r2))
                    break
                if r2 < dz.real * dz.real + dz.imag * dz.imag:
                    dz = z
                    m = 0
                if m == 0:  # z == dz here: check for a settled cycle
                    d = dz - saved
                    if d.real * d.real + d.imag * d.imag < SETTLED * r2:
                        break
                    events += 1
                    if events == next_save:
                        saved = dz
                        next_save *= 2
            out[i, j] = nu


def minibrot():
    """The nucleus (mp), its reference orbit (float64, period + 1 points) and size."""
    mp.mp.dps = 50
    c = mp.mpc(*NUCLEUS)
    for _ in range(8):  # Newton on z_p(c) = 0
        z, dz = mp.mpc(0), mp.mpc(0)
        for _ in range(PERIOD):
            dz = 2 * z * dz + 1
            z = z * z + c
        c -= z / dz
    z, b, l = mp.mpc(0), mp.mpc(1), mp.mpc(1)
    orbit = [0j]
    for _ in range(1, PERIOD):
        z = z * z + c
        l *= 2 * z
        b += 1 / l
        orbit.append(complex(z))
    orbit.append(0j)
    size = 1 / (b * l * l)  # c ≈ nucleus + size · (main-set coordinate)
    return c, np.array(orbit), size


def zoom_progress(seconds):
    """Fraction of the dive (in log scale) done by `seconds`, eased at both ends."""
    s = np.linspace(0, ZOOM_END, 4001)
    speed = smoothstep(np.clip(s / EASE_IN, 0, 1)) * (
        1 - smoothstep(np.clip((s - (ZOOM_END - EASE_OUT)) / EASE_OUT, 0, 1))
    )
    done = np.concatenate([[0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    return np.interp(seconds, s, done / done[-1])


def decade_times(scale_end):
    """
    Seconds at which the zoom crosses ×10, ×100, ... (the counter's ticks), and the
    moment the dive lands (LANDING of the way, in log scale).
    """
    s = np.linspace(0, DURATION, 30001)
    decades = zoom_progress(s) * np.log10(1 / scale_end)
    whole = np.arange(1, int(decades[-1]) + 1)
    land = np.interp(LANDING, zoom_progress(s), s)
    return np.interp(whole, decades, s), float(land)


def soundtrack(scale_end, sound, rng):
    """
    One-shot ambient dive. A low bed (pad or choir) holds the scale's root and fifth,
    its lowpass opening from dark to bright as the zoom deepens; each power of ten
    the counter ticks past strikes the lead one scale step lower (the dive goes
    down), with a soft upper neighbour half-way between ticks, panned left and right
    in turn; when the copy settles into place, the home chord is struck, which the
    choir/pad swells under and lets ring through the hold.
    """
    lead, mode, roots, bed = sound
    scale = Scale(int(rng.choice(roots)), mode)
    ticks, land = decade_times(scale_end)
    n = len(ticks)
    track = audio.silence(DURATION)

    # Bed: root + fifth (+ octave), two dark/bright copies cross-faded by depth.
    layer = audio.silence(DURATION)
    for midi, velocity in (
        (scale.root - 24, 0.14),
        (scale.root - 12, 0.30),
        (scale.root - 5, 0.22),
        (scale.root, 0.12),
    ):
        hz = float(audio.midi_to_hz(midi))
        audio.add(
            layer,
            audio.pad(
                hz, DURATION - 0.6, velocity, attack=1.2, release=1.5, brightness=0.35
            ),
            0.0,
            circular=False,
        )
        if bed == "choir":
            voice = audio.choir(
                2 * hz,
                DURATION - 1.2,
                0.5 * velocity,
                attack=2.5,
                release=1.5,
                vowel="u",
            )
            audio.add(layer, voice, 0.4, circular=False)
    t = np.arange(len(track)) / audio.SAMPLE_RATE
    depth = np.interp(
        t,
        np.linspace(0, DURATION, 30001),
        zoom_progress(np.linspace(0, DURATION, 30001)),
    )[:, None]
    track += audio.lowpass(layer, 450.0, circular=False) * (1 - depth)
    track += audio.lowpass(layer, 2200.0, circular=False) * depth

    # Strikes: the opening note, then one step down per power of ten.
    size = len(scale.steps)
    top = n + 2  # degrees above the root at the start
    audio.add(track, audio.pan(lead(scale.hz(top), 0.55), 0.0), 0.05, circular=False)
    for i, when in enumerate(ticks):
        degree = top - 1 - i
        side = -0.35 if i % 2 else 0.35
        note = lead(scale.hz(degree), 0.5 + 0.15 * i / n)
        audio.add(track, audio.pan(note, side), float(when), circular=False)
        later = 0.5 * (when + (ticks[i + 1] if i + 1 < n else land))
        echo = lead(scale.hz(degree + 2), 0.22)
        audio.add(track, audio.pan(echo, -side), float(later), circular=False)
    # Landing: the home chord, low root to octave.
    for degree, velocity, side in ((0, 0.75, 0.0), (2, 0.45, -0.3), (size, 0.5, 0.3)):
        audio.add(
            track,
            audio.pan(lead(scale.hz(degree), velocity), side),
            land,
            circular=False,
        )
    audio.add(
        track,
        audio.pan(lead(scale.hz(0, octave=-1), 0.55), 0.0),
        land + 0.02,
        circular=False,
    )
    swell = audio.silence(DURATION)
    for midi in (scale.root - 12, scale.root - 5, scale.root):
        voice = audio.choir(
            float(audio.midi_to_hz(midi)),
            DURATION - land - 0.6,
            0.18,
            attack=0.8,
            release=1.2,
            vowel="o",
        )
        audio.add(swell, voice, land - 0.3, circular=False)
    track += audio.lowpass(swell, 2500.0, circular=False)

    track = audio.delay(track, 0.42, feedback=0.3, wet=0.2, circular=False)
    track = audio.reverb(track, rng, 3.2, 0.38, tone=4000.0, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A one-shot zoom into the Mandelbrot set, z ↦ z² + c, from the whole set down to
    the period-27 minibrot whose nucleus is NUCLEUS (size ≈ 3.1e-9), by 3.2e8.

    The size estimate s = 1 / (b l²) (with l = ∏ 2z_k, b = Σ 1/l_k along the nucleus
    orbit) gives the minibrot's scale and orientation: c ≈ nucleus + s c'. The view is
    centred on c' = LOCAL_CENTRE, the same point of the copy as the global centre of
    the first frame (c = -0.5), and turned so the antenna points up: the screen axis
    rotates from i to i s/|s| while the scale shrinks from START_PX to START_PX |s|,
    so the last frame frames the copy exactly as the first frames the set. The first
    frame's centre drifts onto the target as (px/px₀)².

    Render: one exponential map (log-polar image) around the target holds every
    frame. Row k is a circle of radius r = r₀ e^{-kΔ}, column j the angle jΔ, with
    Δ = 2π/COLUMNS so cells are square; each sample is the smooth escape time by
    perturbation, z = Z + δ, δ ↦ (2Z + δ)δ + δc, around the nucleus orbit Z (exactly
    periodic, so it never runs out), rebasing δ ↦ z whenever |z| < |δ|. Interior
    points stop on Brent's cycle check or MAX_ITER. Colour is a palette cycled in
    log(escape time), CYCLES turns per factor of 27 (the copy's escape times are
    ~27× the set's), so the copy wears the first frame's colours. A frame is then
    a cv2.remap of that map (ρ, θ per pixel), blended between mip levels where the
    map is denser than the screen, plus a LaTeX magnification counter.
    """
    settings = settings or ImageProcessingSettings(1)

    nucleus, ref, size = minibrot()
    size_c = complex(size)
    target = nucleus + size * LOCAL_CENTRE
    offset = complex(target - nucleus)  # reference → target, exact enough in float64
    start_centre = complex(mp.mpf(START_CENTRE) - target)  # first frame, relative
    angle0, turn = np.pi / 2, np.angle(size_c)
    scale_end = abs(size_c)
    logger.info(f"Minibrot size {abs(size_c):.3e}, turned {np.degrees(turn):.1f} deg")

    render = None  # an --audio-only run re-uses the frames on disk
    if not settings.audio_only:
        # --- The exponential map, once ---
        rho_max = np.hypot(WIDTH / 2, HEIGHT / 2) + abs(start_centre) / START_PX + 4
        log_r0 = np.log(rho_max * START_PX)
        log_r1 = np.log(CORE * scale_end)
        step = 2 * np.pi / COLUMNS
        rows = int(np.ceil((log_r0 - log_r1) / step)) + 1
        escape = np.empty((rows, COLUMNS), np.float32)
        logger.info(f"Exponential map {rows}×{COLUMNS}")
        exp_map(ref, offset, log_r0, step, rows, COLUMNS, MAX_ITER, escape)
        logger.info(f"Exponential map done, {np.mean(escape < 0):.1%} interior")

        # Texture: log escape time (premultiplied by "outside") and "outside", as uint16,
        # coloured per frame so every scale's far field sits at the same palette phase.
        texture = np.zeros((rows + 2, COLUMNS, 2), np.uint16)
        floor = np.full(rows, np.nan)
        for a in range(0, rows, 512):
            nu = escape[a : a + 512]
            outside = nu > 0
            log_nu = np.log(np.maximum(nu, 1.0))
            texture[a : a + len(nu), :, 0] = np.round(
                np.minimum(log_nu / LOG_NU_MAX, 1) * outside * 65535
            )
            texture[a : a + len(nu), :, 1] = outside * 65535
            masked = np.where(outside, log_nu, np.inf)
            low = np.percentile(masked, FLOOR_PERCENTILE, axis=1)
            floor[a : a + len(nu)] = np.where(np.isfinite(low), low, np.nan)
        del escape
        # the far-field escape time at each radius: filled, monotone and smooth
        known = np.flatnonzero(np.isfinite(floor))
        floor = np.interp(np.arange(rows), known, floor[known])
        floor = np.maximum.accumulate(floor)
        floor = np.convolve(
            np.pad(floor, FLOOR_SMOOTH, mode="edge"),
            np.ones(2 * FLOOR_SMOOTH + 1) / (2 * FLOOR_SMOOTH + 1),
            "valid",
        )
        lut = palette_lut(PALETTE, 1024)[:, ::-1].astype(np.float32) * 255  # BGR
        interior = np.array(hex_to_bgr(INTERIOR), np.float32)
        per_efold = CYCLES / np.log(PERIOD)
        pyramid = [texture]
        for _ in range(LEVELS):
            prev = pyramid[-1]
            pyramid.append(
                cv2.resize(
                    prev, (prev.shape[1] // 2, (prev.shape[0] + 1) // 2), cv2.INTER_AREA
                )
            )
        pad = 2
        pyramid = [np.concatenate([p[:, -pad:], p, p[:, :pad]], 1) for p in pyramid]
        rho_star = 1 / step  # px radius where map cells match screen pixels

        # --- Text ---
        masks = [latex_mask(r"$\times\ 1$", TEXT_SIZE)]
        masks += [latex_mask(rf"$\times\ 10^{{{k}}}$", TEXT_SIZE) for k in range(1, 9)]
        shadows = [
            np.clip(2 * cv2.GaussianBlur(np.pad(m, SHADOW), (0, 0), SHADOW / 2), 0, 1)
            for m in masks
        ]
        text_bgr = np.array(hex_to_bgr(TEXT), np.float32)

        u = np.arange(WIDTH, dtype=np.float32) - (WIDTH - 1) / 2
        v = (HEIGHT - 1) / 2 - np.arange(HEIGHT, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        background = np.array(hex_to_bgr(BACKGROUND), np.float32)

        def view(seconds):
            k = zoom_progress(seconds)
            px = START_PX * scale_end**k
            angle = angle0 + turn * k
            centre = start_centre * (px / START_PX) ** 2  # relative to the target
            return px, angle, centre

        def geometry(d):
            """Per mip level: bounding box, log ρ and θ/Δ of each pixel, blend weight."""
            wx = uu + np.float32(d.real)
            wy = vv + np.float32(d.imag)
            rho = np.maximum(np.hypot(wx, wy), np.float32(1e-3))
            log_rho = np.log(rho) / np.float32(step)
            cols = np.arctan2(wy, wx) / np.float32(step)
            level = np.clip(np.log2(np.float32(rho_star) / rho), 0, LEVELS)
            cx, cy = (WIDTH - 1) / 2 - d.real, (HEIGHT - 1) / 2 + d.imag
            out = []
            for lev in range(LEVELS + 1):
                half = rho_star / 2 ** (lev - 1) + 2 if lev else 1e9
                y0, y1 = int(max(cy - half, 0)), int(min(cy + half + 1, HEIGHT))
                x0, x1 = int(max(cx - half, 0)), int(min(cx + half + 1, WIDTH))
                if y0 >= y1 or x0 >= x1:
                    break
                box = (slice(y0, y1), slice(x0, x1))
                weight = np.maximum(0, 1 - np.abs(level[box] - lev))
                if weight.any():
                    out.append((lev, box, log_rho[box], cols[box], weight[..., None]))
            return out

        still = geometry(0j)  # once the start offset is under 1/100 px

        def render(frame):
            seconds = frame / (FRAME_COUNT - 1) * DURATION
            px, angle, centre = view(seconds)
            d = centre / (px * np.exp(1j * angle))
            parts = still if abs(d) < 0.01 else geometry(d)
            row_shift = np.float32((log_r0 - np.log(px)) / step)
            col_shift = np.float32(angle / step)
            acc = np.zeros((HEIGHT, WIDTH, 2), np.float32)
            for lev, box, log_rho, cols, weight in parts:
                f = np.float32(2.0**lev)
                mx = ((cols + col_shift) % COLUMNS + 0.5) / f + np.float32(pad - 0.5)
                my = (row_shift - log_rho + 0.5) / f - np.float32(0.5)
                sample = cv2.remap(
                    pyramid[lev],
                    mx,
                    my,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                acc[box] += weight * sample
            # colour: log escape time relative to this frame's far field
            far = np.interp(
                row_shift - np.log(FAR_RADIUS) / step, np.arange(rows), floor
            )
            outside = acc[..., 1] / np.float32(65535)
            log_nu = acc[..., 0] * np.float32(LOG_NU_MAX) / np.maximum(acc[..., 1], 1)
            phase = (np.float32(per_efold) * (log_nu - np.float32(far)) + PHASE) % 1
            image = lut[(phase * 1023).astype(np.int32)]
            image *= outside[..., None]
            image += (1 - outside[..., None]) * interior
            # magnification counter, cross-fading between powers of ten
            decades = np.log10(START_PX / px)
            lo = min(int(np.floor(decades + 1e-6)), len(masks) - 1)
            frac = smoothstep(np.clip((decades - lo - 0.85) / 0.15, 0, 1))
            for k, alpha in ((lo, 1 - frac), (lo + 1, frac)):
                if alpha > 0 and k < len(masks):
                    blit_mask(
                        image,
                        shadows[k],
                        WIDTH // 2,
                        TEXT_Y - SHADOW,
                        background,
                        0.8 * alpha,
                        "centre",
                    )
                    blit_mask(
                        image,
                        masks[k],
                        WIDTH // 2,
                        TEXT_Y,
                        text_bgr,
                        0.9 * alpha,
                        "centre",
                    )
            return image

    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    tag = settings.audio_tag
    track = soundtrack(
        scale_end, SOUNDS[tag] if tag in SOUNDS else SOUNDS[STYLE], settings.audio_rng
    )
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
