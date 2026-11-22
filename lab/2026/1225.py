"""
The Last Circle to Break

This is chaos in the standard map, the simplest model of a kicked rotor: a
wheel that gets a kick every tick, a little harder as the video goes on (the
number K). Each curve is one rotor's motion, drawn point by point. Blue rings
are order; orange and pink dust is chaos.

Weak kicks leave most motions orderly, and the smooth circles fence the chaos
into separate seas. Turn K up and the circles break, one after another. The
last one standing is the golden circle, whose rhythm is set by the golden
ratio, the most irrational number: no simple ratio of kicks can resonate with
it. At K ≈ 0.9716 it finally breaks into a dotted ghost, and the two seas of
chaos leak into one.

Listen: a harp plays a calm, repeating figure over a soft chord. The more of
the picture turns to chaos, the more its notes scramble, and when the golden
circle breaks, a rolled chord rings out.

Made with Python, NumPy & Numba. Sound synthesized in NumPy.
#standardmap #chaostheory #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from numba import njit, prange
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
    latex_mask,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Turn up the chaos. Which circle survives?'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

# The kick strength K over the video: (seconds, K), joined by a monotone cubic.
K_KEYS = ((0.0, 0.45), (5.5, 0.90), (8.5, 0.985), (13.4, 1.35), (DURATION, 1.35))
K_GOLDEN = 0.971635  # Greene's critical K for the golden-mean circle
GOLDEN = (3 - np.sqrt(5)) / 2  # rotation number 1/phi^2 = 0.381966...

ORBITS = 1200  # starting points, half on theta = 0 and half on theta = pi
STEPS = 2000  # iterations drawn per orbit
GOLDEN_STEPS = 8000
SCALE = WIDTH / (2 * np.pi)  # px per unit of theta and p
STATS_STEPS = 1500
LAMBDA_LO, LAMBDA_HI = 0.012, 0.045  # finite-time Lyapunov: regular .. chaotic

BG = np.array([0.012, 0.012, 0.03])  # RGB
ORDER = ["#58e0c8", "#5f9dff", "#7c6cff"]  # regular circles, by rotation number
SEA_A = "#ff7a2e"  # chaos born around p = 0
SEA_B = "#ff3cc8"  # chaos born around p = pi
GOLD = np.array([1.0, 0.82, 0.42])
TEXT_RGB = (0.93, 0.93, 0.96)
GAIN = 0.22
GOLD_GAIN = 1.2
DUST_BLUR = 1.3  # px, softening of the chaotic layer
CHAOS_WEIGHT = 1.8  # point weight of a chaotic orbit (regular ones weigh 1)
TEXT_BAND = 230  # px, darkened band behind the text at top and bottom
LABEL_TIMES = (0.6, 1.4, 5.6, 6.6)  # s: golden label fade in, full, fade out, gone


# Sound: a pad bed with a plucked accent line that scrambles as chaos spreads.
STYLE = "harp"  # common.music.STYLES; an --audio-only tag naming one overrides it
BPM = 96
ARPEGGIO = (0, 2, 4, 5, 7, 5, 4, 2)  # scale degrees of the orderly figure
BED = ((0.0, (0, 3, 5)), ("break", (1, 3, 6)), (12.6, (0, 3, 5, 8)))  # (s, degrees)
SOUND_SAMPLES = 90  # K values at which the chaotic fraction is measured


def hex_rgb(colour):
    colour = colour.lstrip("#")
    return np.array([int(colour[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def k_of_time(seconds):
    times, values = zip(*K_KEYS)
    return PchipInterpolator(times, values)(seconds)


@njit(nogil=True, fastmath=True, cache=True)
def orbit_stats(k, theta0, p0, steps):
    """Finite-time Lyapunov exponent and rotation number of each orbit."""
    count = theta0.shape[0]
    lyapunov = np.empty(count)
    rotation = np.empty(count)
    for i in range(count):
        theta, p = theta0[i], p0[i]
        dt, dp = 1.0, 0.0
        total, advance = 0.0, 0.0
        for _ in range(steps):
            dp = dp + k * np.cos(theta) * dt
            dt = dt + dp
            p = p + k * np.sin(theta)
            theta = theta + p
            advance += p
            norm = np.sqrt(dt * dt + dp * dp)
            total += np.log(norm)
            dt /= norm
            dp /= norm
        lyapunov[i] = total / steps
        rotation[i] = advance / (2 * np.pi * steps)
    return lyapunov, rotation


@njit(nogil=True, fastmath=True, cache=True)
def splat_orbits(k, theta0, p0, colours, steps, acc, scale):
    """Bilinear splat of every orbit point (and its 2 pi copies in p) into acc."""
    height, width = acc.shape[0], acc.shape[1]
    two_pi = 2 * np.pi
    for i in range(theta0.shape[0]):
        theta, p = theta0[i], p0[i]
        r, g, b, w = colours[i, 0], colours[i, 1], colours[i, 2], colours[i, 3]
        for _ in range(steps):
            p = p + k * np.sin(theta)
            theta = theta + p
            theta -= two_pi * np.floor(theta / two_pi)
            p -= two_pi * np.floor((p + np.pi) / two_pi)
            x = theta * scale - 0.5
            for copy in (-1, 0, 1):
                y = height / 2 - (p + copy * two_pi) * scale - 0.5
                if y < 0 or y >= height - 1:
                    continue
                x0 = int(np.floor(x))
                y0 = int(y)
                fx = x - x0
                fy = y - y0
                xa = x0 % width
                xb = (x0 + 1) % width
                w00 = (1 - fx) * (1 - fy) * w
                w10 = fx * (1 - fy) * w
                w01 = (1 - fx) * fy * w
                w11 = fx * fy * w
                acc[y0, xa, 0] += w00 * r
                acc[y0, xa, 1] += w00 * g
                acc[y0, xa, 2] += w00 * b
                acc[y0, xa, 3] += w00
                acc[y0, xb, 0] += w10 * r
                acc[y0, xb, 1] += w10 * g
                acc[y0, xb, 2] += w10 * b
                acc[y0, xb, 3] += w10
                acc[y0 + 1, xa, 0] += w01 * r
                acc[y0 + 1, xa, 1] += w01 * g
                acc[y0 + 1, xa, 2] += w01 * b
                acc[y0 + 1, xa, 3] += w01
                acc[y0 + 1, xb, 0] += w11 * r
                acc[y0 + 1, xb, 1] += w11 * g
                acc[y0 + 1, xb, 2] += w11 * b
                acc[y0 + 1, xb, 3] += w11


@njit(nogil=True, fastmath=True, cache=True)
def rotation_number(k, theta, p, steps):
    advance = 0.0
    for _ in range(steps):
        p = p + k * np.sin(theta)
        theta = theta + p
        advance += p
    return advance / (2 * np.pi * steps)


@njit(parallel=True, cache=True)
def golden_seeds(ks, steps):
    """
    For each K, the height p on the symmetry line theta = 0 where the orbit's
    rotation number is the golden 1/phi^2 (bisection; the circle is a graph over
    theta, and the rotation number grows with p across the circles).
    """
    seeds = np.empty(ks.shape[0])
    target = (3 - np.sqrt(5)) / 2
    for j in prange(ks.shape[0]):
        lo, hi = 1.6, 3.2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if rotation_number(ks[j], 0.0, mid, steps) < target:
                lo = mid
            else:
                hi = mid
        seeds[j] = 0.5 * (lo + hi)
    return seeds


@njit(nogil=True, cache=True)
def height_at_pi(k, p0, steps):
    """Height of the golden orbit where it crosses theta = pi (nearest visit)."""
    theta, p = 0.0, p0
    best, height = 1e9, p0
    two_pi = 2 * np.pi
    for _ in range(steps):
        p = p + k * np.sin(theta)
        theta = theta + p
        phase = theta - two_pi * np.floor(theta / two_pi)
        if abs(phase - np.pi) < best:
            best = abs(phase - np.pi)
            height = p - two_pi * np.floor((p + np.pi) / two_pi)
    return height


@njit(parallel=True, cache=True)
def chaotic_fraction(ks, theta0, p0, steps, threshold):
    """Fraction of the starting points whose orbit is chaotic, for each K."""
    fraction = np.empty(ks.shape[0])
    for j in prange(ks.shape[0]):
        lyapunov, _ = orbit_stats(ks[j], theta0, p0, steps)
        fraction[j] = np.mean(lyapunov > threshold)
    return fraction


def soundtrack(theta0, p0, t_break, style_name, rng):
    """
    One-shot. The measured chaotic fraction c(t) (orbits with finite-time
    Lyapunov exponent above the colour midpoint, at SOUND_SAMPLES values of K(t))
    drives everything: an eighth-note grid at BPM where a pluck sounds with
    probability 0.3 + 0.45 c, playing the ARPEGGIO in order while c is small and
    replaced, note by note with probability c, by random scale degrees and
    loosened timing. Pan sweeps with the figure. At the golden circle's break a
    rolled chord in the high register; a pad bed (open fifths, a suspended
    chord after the break, home with the top added for the rest) whose lowpass
    opens with c. Ends with a low root ringing over the fade.
    settings.audio_rng picks the key, the scrambles and the room.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)

    sample_t = np.linspace(0, DURATION, SOUND_SAMPLES)
    mid = 0.5 * (LAMBDA_LO + LAMBDA_HI)
    fraction = chaotic_fraction(k_of_time(sample_t), theta0, p0, STATS_STEPS, mid)
    chaos = lambda when: float(np.interp(when, sample_t, fraction))
    c_max = max(fraction.max(), 1e-6)

    step = 30.0 / BPM  # an eighth note
    end = DURATION - 1.6
    for i, when in enumerate(np.arange(0.12, end, step)):
        c = chaos(when) / c_max
        if rng.random() > 0.3 + 0.45 * c or abs(when - t_break) < 0.6:
            continue
        degree = ARPEGGIO[i % len(ARPEGGIO)]
        if rng.random() < c:
            degree = int(rng.integers(-2, 10))
        when = when + c * rng.uniform(-0.05, 0.05)
        velocity = 0.32 + 0.2 * rng.random() + 0.1 * c
        position = 0.6 * np.sin(2 * np.pi * i / len(ARPEGGIO)) * (1 - 0.5 * c)
        position += 0.4 * c * rng.uniform(-1, 1)
        note = style.voice(scale.hz(degree), velocity)
        audio.add(track, audio.pan(note, position), max(when, 0.0), circular=False)

    # the golden circle breaks: a rolled chord, high
    for j, degree in enumerate((0, 2, 4, 5, 7)):
        note = style.voice(scale.hz(degree, 1), 0.5 - 0.04 * j)
        audio.add(
            track, audio.pan(note, -0.5 + 0.25 * j), t_break + 0.07 * j, circular=False
        )
    # the end: a low root rings out
    audio.add(track, style.bass(scale.hz(0, -1), 0.45), end + 0.2, circular=False)
    audio.add(track, style.voice(scale.hz(4, 0), 0.3), end + 0.2, circular=False)

    bed = audio.silence(DURATION)
    starts = [t_break if s == "break" else s for s, _ in BED]
    for (start, (_, degrees)), stop in zip(enumerate(BED), starts[1:] + [DURATION]):
        begin = starts[start]
        for degree in degrees:
            chord = audio.pad(
                scale.hz(degree, -1),
                stop - begin + (0.0 if stop == DURATION else 0.8),
                0.4,
                attack=0.3 if begin == 0 else 1.2,
                release=1.6,
                brightness=0.35,
            )
            audio.add(bed, chord, begin, circular=False)
    # the bed brightens with the chaos: mix of a dark and an open lowpass
    dark = audio.lowpass(bed, 700.0, circular=False)
    bright = audio.lowpass(bed, 2600.0, circular=False)
    mix = np.interp(np.arange(len(bed)) / audio.SAMPLE_RATE, sample_t, fraction / c_max)
    bed = dark + mix[:, None] * (bright - dark)
    track += 0.55 * audio.highpass(bed, 70.0, circular=False)

    track = style.room(track, rng, circular=False)
    track = audio.fade(track, fade_out=FADE)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Chirikov's standard map p' = p + K sin(theta), theta' = theta + p' on the
    torus, as a phase portrait at kick strength K(t) (a monotone cubic through
    K_KEYS: 0.45 -> 1.35, slowest around Greene's K_c = 0.971635 for the
    golden-mean circle; one-shot with a rest and a fade at the end).

    Every frame is a pure function of K: ORBITS starting points (half on the
    symmetry line theta = 0, half on theta = pi, p spread over one period with
    a seeded jitter) are iterated afresh. A first pass (STATS_STEPS) gives each
    orbit's finite-time Lyapunov exponent (tangent map) and rotation number; a
    second pass splats STEPS points bilinearly (numba, nogil, on the thread
    pool) with the periodic copies in p, as the view spans 1.78 periods at
    1:1 aspect. Colour: regular orbits by rotation number (teal -> blue ->
    indigo), chaotic ones (smoothstep between LAMBDA_LO and LAMBDA_HI) by the
    sea they start in, i.e. which side of the golden circle (SEA_A around
    p = 0, SEA_B around p = pi). Tone mapping: mean colour times
    1 - exp(-GAIN n).

    The golden circle (rotation number 1/phi^2 and its mirror) is found per
    frame by bisection on theta = 0 (golden_seeds, numba prange), frozen at
    K_c - 0.004 and drawn from that seed afterwards, so above K_c it shows the
    leaking cantorus. Gold core, white-hot centre and a Gaussian glow. LaTeX
    masks: the map at the top, a K counter at the bottom that flashes gold at
    K_c, a label on the golden circle early on.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    times = np.arange(FRAME_COUNT) / (FRAME_COUNT - 1) * DURATION
    ks = k_of_time(times)

    # starting points: half on the symmetry line theta = 0, half on theta = pi
    half = ORBITS // 2
    grid = (np.arange(half) + rng.uniform(0.2, 0.8, half)) / half
    theta0 = np.concatenate([np.zeros(half), np.full(half, np.pi)])
    p0 = np.concatenate([-np.pi + 2 * np.pi * grid, -np.pi + 2 * np.pi * grid[::-1]])

    # the golden circle: found below K_GOLDEN, frozen just before it breaks
    freeze = K_GOLDEN - 0.004
    seeds = golden_seeds(np.minimum(ks, freeze), 20000)
    heights = np.array(
        [height_at_pi(min(k, freeze), s, 20000) for k, s in zip(ks, seeds)]
    )
    logger.info(f"golden seed {seeds[0]:.4f} .. {seeds[-1]:.4f}")

    order_lut = np.array([hex_rgb(c) for c in ORDER])
    sea_a, sea_b = hex_rgb(SEA_A), hex_rgb(SEA_B)

    formula = latex_mask(
        r"$p_{n+1} = p_n + K\,\sin\theta_n \qquad \theta_{n+1} = \theta_n + p_{n+1}$",
        40,
    )
    counters = {f"{k:.2f}": latex_mask(rf"$K = {k:.2f}$", 64) for k in ks}
    label = latex_mask(r"$\mathrm{the\ golden\ circle}$", 48)
    t_break = float(np.interp(K_GOLDEN, ks, times))
    rows = np.arange(HEIGHT)
    edge = np.minimum(rows, HEIGHT - 1 - rows) / TEXT_BAND
    shade = (0.35 + 0.65 * smoothstep(np.clip(edge, 0, 1)))[:, None, None]

    def render(frame):
        k = float(ks[frame])
        t = times[frame]
        lyapunov, rotation = orbit_stats(k, theta0, p0, STATS_STEPS)
        chaos = smoothstep((lyapunov - LAMBDA_LO) / (LAMBDA_HI - LAMBDA_LO))
        distance = np.abs(rotation - np.round(rotation)) * 2  # 0 .. 1
        order = np.stack(
            [
                np.interp(distance, np.linspace(0, 1, 3), order_lut[:, c])
                for c in range(3)
            ],
            axis=1,
        )
        wall = np.where(theta0 == 0, seeds[frame], heights[frame])
        home_a = (np.abs(p0) < np.abs(wall))[:, None]
        sea = np.where(home_a, sea_a, sea_b)
        colours = np.empty((ORBITS, 4))
        colours[:, :3] = (1 - chaos[:, None]) * order + chaos[:, None] * sea
        colours[:, 3] = 1.0 + (CHAOS_WEIGHT - 1.0) * chaos

        # chaotic dust goes in its own layer, softened so it shimmers, not crackles
        acc = np.zeros((HEIGHT, WIDTH, 4), np.float32)
        dust = np.zeros((HEIGHT, WIDTH, 4), np.float32)
        wild = chaos > 0.5
        splat_orbits(k, theta0[~wild], p0[~wild], colours[~wild], STEPS, acc, SCALE)
        splat_orbits(k, theta0[wild], p0[wild], colours[wild], STEPS, dust, SCALE)
        acc += cv2.GaussianBlur(dust, (0, 0), DUST_BLUR)
        gold = np.zeros((HEIGHT, WIDTH, 4), np.float32)
        g_theta = np.zeros(2)
        g_p = np.array([seeds[frame], -seeds[frame]])
        g_col = np.ones((2, 4))
        splat_orbits(k, g_theta, g_p, g_col, GOLDEN_STEPS, gold, SCALE)

        n = acc[..., 3:]
        image = acc[..., :3] / (n + 1e-6) * (1 - np.exp(-GAIN * n))
        g = gold[..., 3]
        g = 1 - np.exp(-GOLD_GAIN * g)
        g = cv2.GaussianBlur(g, (0, 0), 1.0) * 1.6
        glow = cv2.GaussianBlur(g, (0, 0), 7.0)
        g = np.minimum(g, 1.0)
        image = image * (1 - g[..., None]) + (g + 1.6 * glow)[..., None] * GOLD
        image = BG + image * shade
        image += np.clip(g - 0.7, 0, 1)[..., None] * 0.8  # white-hot core

        blit_mask(image, formula, WIDTH // 2, 70, TEXT_RGB, alpha=0.9, anchor="centre")
        crossing = np.exp(-(((t - t_break) / 0.8) ** 2))
        colour = (1 - crossing) * np.array(TEXT_RGB) + crossing * GOLD
        counter = counters[f"{k:.2f}"]
        blit_mask(image, counter, WIDTH // 2, HEIGHT - 150, colour, anchor="centre")
        a, b, c, e = LABEL_TIMES
        alpha = smoothstep((t - a) / (b - a)) * (1 - smoothstep((t - c) / (e - c)))
        if alpha > 0:
            y = int(HEIGHT / 2 - seeds[frame] * SCALE) - 80
            blit_mask(image, label, 40, y, GOLD, alpha=alpha)
        image *= 1 - smoothstep((t - (DURATION - FADE)) / FADE)
        return np.clip(image * 255, 0, 255)

    render_frames(settings, render, FRAME_COUNT, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(theta0, p0, t_break, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
