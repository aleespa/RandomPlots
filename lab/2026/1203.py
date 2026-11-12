"""
Fermat's Christmas Theorem

Which prime numbers are a sum of two squares? 5 = 1² + 2² and 13 = 2² + 3²
are; 3, 7 and 11 never are. Here a circle grows across a grid of whole-number
points. Each time its radius squared reaches a prime p, it lights the points
(a, b) with a² + b² = p, if there are any.

Watch the gold rings catch points and the blue ones catch nothing. The rule
is simple: an odd prime is a sum of two squares exactly when it leaves
remainder 1 on division by 4. Fermat announced it in a letter dated
Christmas Day, 1640. Zoom out and the lit points are the primes of the
Gaussian integers a + bi, with the primes 3, 7, 11… sitting alone on the
axes, a snowflake with eight-fold symmetry that no one designed.

Listen: every prime the circle catches strikes two marimba notes, one for
each square, while strings swell as the view pulls back; the axis primes ring
low.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#gaussianprimes #numbertheory #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from scipy.interpolate import PchipInterpolator

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
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

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The sweep ---------------------------------------------------------------------
# Radius of the circle (lattice units) at key times (s); PCHIP in log r, so it
# eases out into R_END at SWEEP_END and then rests.
KEYS = ((0.0, 1.12), (1.0, 1.9), (2.6, 3.2), (5.5, 5.4), (8.5, 14.0), (11.5, 40.0))
SWEEP_END, R_END = 13.6, 64.0
ZOOM_MARGIN, ZOOM_MIN = 1.1, 2.3  # half-width of the view: hypot(margin r, min)
TICKER_LAST = (
    17  # the ticker names each prime up to here (later ones come < 0.6 s apart)
)
THEOREM_AT = 5.9  # s: the theorem replaces the ticker
FADE = 0.4  # s: fade to the background at the end

# --- Layout (px, 1x) ---------------------------------------------------------------
SS = 2  # supersampling
CX, CY = WIDTH / 2, 1000
TITLE_Y, TICKER_Y = 190, 300
LATTICE_TOP = 440  # no lattice behind the text

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#05060c"
LATTICE = "#1c2033"
HIT_PALETTE = ["#ffd86b", "#ffaa3d", "#ff6f5c", "#e2479b", "#9b4dd6"]
AXIS = "#5fe3ff"  # the inert primes 3, 7, 11, ... on the axes
MISS = "#4a6cff"  # rings that catch no point
SWEEP = "#c9d2ff"
TEXT, TEXT_DIM = "#f6f0ff", "#9aa3c8"
GLOW_SIGMA, GLOW_GAIN = 5.0, 1.1
FLASH = 0.18  # s: a newly lit point flares and settles
RING_LIFE = 0.32  # s: e-folding of a prime's ring flash

# --- Sound -------------------------------------------------------------------------
STYLE = "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one wins
BPM = 96  # notes snap to an eighth-note grid at this tempo (15 s = 6 bars)
CHORDS = (0, 3, 5, 4, 0)  # strings, scale degrees of each chord root


def primes_up_to(n):
    sieve = np.ones(n + 1, bool)
    sieve[:2] = False
    for k in range(2, int(n**0.5) + 1):
        if sieve[k]:
            sieve[k * k :: k] = False
    return sieve


def radius_curve():
    """r(s), the circle's radius at time s, as a vectorised function."""
    times = [k[0] for k in KEYS] + [SWEEP_END, SWEEP_END + 1.0]
    logs = [np.log(k[1]) for k in KEYS] + [np.log(R_END)] * 2
    spline = PchipInterpolator(times, logs)

    def radius(s):
        s = np.clip(np.asarray(s, float), 0.0, SWEEP_END)
        return np.exp(spline(s))

    return radius


def lattice_events(radius):
    """
    Every Gaussian prime a + bi with a² + b² <= R_END², and the time the circle
    reaches it; plus each rational prime p <= R_END² with its ring time √p.
    """
    m = int(R_END)
    is_prime = primes_up_to(m * m)
    a, b = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    a, b = a.ravel(), b.ravel()
    norm = a * a + b * b
    inside = norm <= m * m
    a, b, norm = a[inside], b[inside], norm[inside]
    off_axis = (a != 0) & (b != 0) & is_prime[norm]
    side = np.abs(a + b)  # on an axis: |a| or |b|
    on_axis = ((a == 0) | (b == 0)) & is_prime[side] & (side % 4 == 3)
    gauss = off_axis | on_axis
    # When does the circle reach radius ρ? Invert r(s) on a fine grid.
    grid = np.linspace(0.0, SWEEP_END, 20001)
    r_grid = radius(grid)

    def when(rho):
        return np.interp(rho, r_grid, grid)

    points = {
        "a": a[gauss],
        "b": b[gauss],
        "norm": norm[gauss],
        "axis": on_axis[gauss],
        "time": when(np.sqrt(norm[gauss])),
    }
    primes = np.flatnonzero(is_prime)
    rings = {"p": primes, "time": when(np.sqrt(primes)), "hit": primes % 4 != 3}
    return points, rings, when


def two_squares(p):
    """(a, b) with a <= b and a² + b² = p, or None."""
    for a in range(1, int(np.sqrt(p / 2)) + 1):
        b = int(round(np.sqrt(p - a * a)))
        if a * a + b * b == p:
            return a, b
    return None


def soundtrack(points, rings, style_name, rng):
    """
    Mallet garden over a strings swell, from the same event times as the picture.

    - Every prime p = a² + b² (gold ring) strikes a dyad on `style.voice`: scale
      degree a (low) and b (high), wrapped into two octaves, panned by the angle
      of (a, b). Events are pulled onto an eighth-note grid at BPM and at most one
      dyad sounds per slot (the latest prime in it), louder the more primes share
      the slot, so the dense late sweep becomes a steady arpeggio.
    - Each axis prime q ≡ 3 (mod 4), lit at radius q, plays a low bass note.
    - Blue rings (primes ≡ 3 mod 4) make no sound: the silence is the miss.
    - Strings hold CHORDS, swelling with log r, and the home chord lands at
      SWEEP_END and rings over the rest.
    One-shot: circular=False throughout and master(loop=False).
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    size = len(scale.steps)
    track = audio.silence(DURATION)
    kw = {"circular": False}
    step = 60.0 / BPM / 2

    hits = rings["hit"] & (rings["time"] < SWEEP_END - 0.05)
    slots = {}
    for p, when in zip(rings["p"][hits], rings["time"][hits]):
        slot = int(round(when / step)) if when > 3.0 else when / step
        slots.setdefault(slot, []).append(int(p))
    for slot, ps in sorted(slots.items()):
        when = slot * step
        a, b = two_squares(ps[-1])
        crowd = len(ps)
        velocity = float(np.clip(0.42 + 0.08 * np.log1p(crowd), 0.3, 0.7))
        position = float(np.cos(2 * np.arctan2(b, a)))  # 45° centre, axes wide
        low = scale.hz(a % size)
        high = scale.hz(size + b % (2 * size))
        audio.add(track, audio.pan(style.voice(low, 0.8 * velocity), -0.4), when, **kw)
        audio.add(
            track, audio.pan(style.voice(high, velocity), 0.6 * position), when, **kw
        )

    axis = points["axis"] & (points["a"] > 0)
    for q, when in zip(points["a"][axis], points["time"][axis]):
        bass = scale.hz(int(q) % size, octave=-2)
        audio.add(track, style.bass(bass, 0.55), float(when), **kw)

    # Strings: a chord per stretch of the sweep, louder as the view widens.
    edges = np.linspace(0.0, SWEEP_END, len(CHORDS))
    for i, degree in enumerate(CHORDS[:-1]):
        start, end = edges[i], edges[i + 1]
        velocity = 0.12 + 0.2 * i / (len(CHORDS) - 1)
        for note in scale.chord(degree, 3, octave=-1):
            sound = audio.strings(
                float(audio.midi_to_hz(note)), end - start + 0.6, velocity, 1.0, 1.2
            )
            audio.add(track, sound, start, **kw)
    for note in scale.chord(CHORDS[-1], 4, octave=-1):
        sound = audio.strings(
            float(audio.midi_to_hz(note)), DURATION - SWEEP_END, 0.42, 0.25, 1.4, 0.6
        )
        audio.add(track, sound, SWEEP_END, **kw)
    audio.add(track, style.voice(scale.hz(size), 0.75), SWEEP_END, **kw)
    audio.add(track, style.bass(scale.hz(0, octave=-2), 0.7), SWEEP_END, **kw)
    logger.info(
        f"Soundtrack: {style_name}, root {scale.root}, {len(slots)} dyads, "
        f"{int(axis.sum())} bass notes"
    )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Fermat's two-squares theorem and the Gaussian primes, as a one-shot.

    A circle centred on the origin of the integer lattice Z² grows with radius
    r(s) (PCHIP through KEYS in log r, easing out into R_END = 64 at
    s = SWEEP_END, then resting). When r² passes a rational prime p its ring
    flashes: gold if p = 2 or p ≡ 1 (mod 4), when the circle passes through the
    8 lattice points (±a, ±b), (±b, ±a) with a² + b² = p (4 for p = 2), which
    light up; blue if p ≡ 3 (mod 4), when it passes through none. Points
    (±q, 0), (0, ±q) light when r = q for primes q ≡ 3 (mod 4). The lit set is
    exactly the Gaussian primes of norm <= R_END², in cyan on the axes and
    coloured by norm elsewhere. The camera's half-width is
    hypot(ZOOM_MARGIN r, ZOOM_MIN), so the view pulls back from a close-up of
    the first lattice points to the whole disc; grey lattice points fade out
    as their spacing drops below ~12 px.

    Text (Computer Modern via matplotlib mathtext, typeset once): a title, a
    ticker naming each prime up to TICKER_LAST as a² + b² or ≠ a² + b², then
    the theorem from THEOREM_AT.

    Rendering: every event time comes from inverting r(s) once, so each frame
    is a pure function of s: cv2 circles at 2x (lattice, ring flashes, points
    with a short flare), INTER_AREA downsample, Gaussian glow, text, a fade in
    the last FADE s, on the render_frames thread pool. Nothing is random:
    --seed does not change the picture. The soundtrack (`soundtrack`, Mallet
    garden over strings) uses settings.audio_rng for its root and reverb; the
    posted one is STYLE = "mallet_garden" with --seed 0 (root MIDI 61). An
    --audio-only tag naming a common.music style swaps the style.
    """
    settings = settings or ImageProcessingSettings(1)
    radius = radius_curve()
    points, rings, when = lattice_events(radius)
    logger.info(
        f"{len(points['a'])} Gaussian primes, {len(rings['p'])} rings; "
        f"p = 5 at {when(np.sqrt(5)):.2f} s, p = 41 at {when(np.sqrt(41)):.2f} s"
    )

    W2, H2 = WIDTH * SS, HEIGHT * SS
    shift = 3
    one = 1 << shift
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    lattice_colour = np.array(hex_to_bgr(LATTICE), np.float32)
    lut = palette_lut(HIT_PALETTE, 256)[:, ::-1] * 255  # BGR
    level = np.sqrt(points["norm"]) / R_END
    point_colour = lut[np.clip(level * 255, 0, 255).astype(int)]
    point_colour[points["axis"]] = hex_to_bgr(AXIS)
    hit_colour = np.array(hex_to_bgr(HIT_PALETTE[1]), np.float32)
    miss_colour = np.array(hex_to_bgr(MISS), np.float32)
    sweep_colour = np.array(hex_to_bgr(SWEEP), np.float32)
    order = np.argsort(points["time"])
    for key in points:
        points[key] = points[key][order]
    point_colour = point_colour[order]

    # Text, once.
    title = latex_mask(r"$\mathrm{Which\ primes\ are\ a\ sum\ of\ two\ squares?}$", 50)
    ticker = []  # (time, mask, colour)
    for p, t_p, hit in zip(rings["p"], rings["time"], rings["hit"]):
        if p > TICKER_LAST:
            break
        if hit:
            a, b = two_squares(int(p))
            tex = rf"${p} = {a}^2 + {b}^2$"
        else:
            tex = rf"${p} \neq a^2 + b^2$"
        colour = HIT_PALETTE[1] if hit else MISS
        ticker.append((float(t_p), latex_mask(tex, 100), hex_to_bgr(colour)))
    ticker_times = np.array([t for t, _, _ in ticker])
    theorem = latex_mask(
        r"$p = a^2 + b^2 \ \Leftrightarrow\ p \equiv 1\ (\mathrm{mod}\ 4)$", 74
    )
    credit = latex_mask(
        r"$\mathrm{for\ odd\ primes\ }p\mathrm{{,}\ Fermat{,}\ 25\ December\ 1640}$", 40
    )
    text, text_dim = hex_to_bgr(TEXT), hex_to_bgr(TEXT_DIM)

    def render(frame):
        s = frame / (FRAME_COUNT - 1) * DURATION
        r = float(radius(s))
        half = np.hypot(ZOOM_MARGIN * r, ZOOM_MIN)
        scale = (WIDTH / 2) / half  # px per lattice unit
        canvas = np.zeros((H2, W2, 3), np.float32)

        def sx(x):
            return np.round((CX + scale * x) * SS * one).astype(np.int64)

        def sy(y):
            return np.round((CY - scale * y) * SS * one).astype(np.int64)

        dot = float(np.clip(0.15 * scale, 2.6, 32.0))

        # Grey lattice points, while they are far enough apart to read.
        fade = float(smoothstep((scale - 11.0) / 20.0))
        if fade > 0:
            nx = int(np.ceil(WIDTH / 2 / scale)) + 1
            ny = int(np.ceil(HEIGHT / scale)) + 1
            colour = (lattice_colour * fade).tolist()
            lr = int(round(0.45 * dot * SS * one))
            for j in range(-ny, ny + 1):
                y = int(sy(j))
                if not LATTICE_TOP * SS * one <= y <= H2 * one + lr:
                    continue
                for i in range(-nx, nx + 1):
                    cv2.circle(
                        canvas, (int(sx(i)), y), lr, colour, -1, cv2.LINE_AA, shift
                    )

        # Ring flashes of recent primes.
        recent = (rings["time"] <= s) & (rings["time"] > s - 5 * RING_LIFE)
        width = max(1, int(round(1.6 * SS)))
        crowd = 0.75 * float(np.clip(scale / 60.0, 0.12, 1.0))  # dim when packed
        for p, t_p, hit in zip(
            rings["p"][recent], rings["time"][recent], rings["hit"][recent]
        ):
            strength = np.exp(-(s - t_p) / RING_LIFE)
            colour = (hit_colour if hit else miss_colour) * (crowd * strength)
            rr = int(round(np.sqrt(p) * scale * SS * one))
            cv2.circle(
                canvas,
                (int(sx(0)), int(sy(0))),
                rr,
                colour.tolist(),
                width,
                cv2.LINE_AA,
                shift,
            )
        # The sweeping circle itself.
        if s < SWEEP_END:
            sweep = 0.35 * float(smoothstep((SWEEP_END - s) / 0.8))
            cv2.circle(
                canvas,
                (int(sx(0)), int(sy(0))),
                int(round(r * scale * SS * one)),
                (sweep_colour * sweep).tolist(),
                width,
                cv2.LINE_AA,
                shift,
            )

        # The lit Gaussian primes, flaring as they appear.
        lit = int(np.searchsorted(points["time"], s, side="right"))
        if lit:
            age = s - points["time"][:lit]
            flare = np.exp(-age / FLASH)
            radii = np.round(dot * (1 + 0.9 * flare) * SS * one).astype(int)
            colours = point_colour[:lit] * (1 + 0.45 * flare[:, None])
            xs, ys = sx(points["a"][:lit]), sy(points["b"][:lit])
            for x, y, rad, c in zip(
                xs.tolist(), ys.tolist(), radii.tolist(), colours.tolist()
            ):
                cv2.circle(canvas, (x, y), rad, c, -1, cv2.LINE_AA, shift)

        image = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(image, (0, 0), GLOW_SIGMA)
        image = image + GLOW_GAIN * glow + bg

        # Text.
        intro = float(smoothstep(s / 0.3))
        blit_mask(image, title, WIDTH // 2, TITLE_Y, text, intro, "centre")
        k = int(np.searchsorted(ticker_times, s, side="right")) - 1
        ticker_out = float(smoothstep((THEOREM_AT - s) / 0.4))
        if k >= 0 and ticker_out > 0:
            t_k, mask, colour = ticker[k]
            pop = float(smoothstep((s - t_k) / 0.12))
            blit_mask(
                image, mask, WIDTH // 2, TICKER_Y, colour, pop * ticker_out, "centre"
            )
        show = float(smoothstep((s - THEOREM_AT) / 0.6))
        if show > 0:
            blit_mask(image, theorem, WIDTH // 2, TICKER_Y + 10, text, show, "centre")
            late = float(smoothstep((s - THEOREM_AT - 0.5) / 0.6))
            blit_mask(
                image, credit, WIDTH // 2, TICKER_Y + 120, text_dim, late, "centre"
            )

        out = float(smoothstep((DURATION - s) / FADE))
        if out < 1:
            image = bg + out * (image - bg)
        return image

    render_frames(settings, render, FRAME_COUNT)
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(points, rings, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
