"""
The Pendulum Wave

Fifteen pendulums, let go together. The first swings 6 times in 15 seconds,
the next 7, the next 8, and so on up to 20: fifteen harmonics of one slow
beat. A pendulum wave is nothing more than that, and yet it dances.

They start in a line and peel apart into a travelling snake. At a third of
the way, every third pendulum swings in step; at half way, every other one,
two rows swinging against each other; at a quarter, every fourth. After
exactly 15 seconds every pendulum has made a whole number of swings and
they are back in line. The glowing string is the one smooth wave hidden in their
positions. As it gets too fine for fifteen dots to show, it folds back into
a slower one: aliasing, the same effect that makes wagon wheels spin
backwards in films.

Listen: every pendulum strikes a marimba bar at the end of each swing,
longest lowest. Scattered they patter softly; whenever they fall into step,
a chord rings out.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#pendulumwave #harmonics #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

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

# Opening line over the first 3 s, drawn by render_frames.
HOOK = '15 pendulums. Nobody choreographed this.'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # s: one loop, the time T in which every pendulum swings a whole number of times
FRAME_COUNT = frame_count(DURATION)

# --- The pendulums ---------------------------------------------------------------
SWINGS = np.arange(6, 21)  # swings per loop, top to bottom
CENTRE = 13  # the middle pendulum's swings: the common phase of the hidden wave
COUNT = len(SWINGS)

# --- Layout (px, 1x) ---------------------------------------------------------------
SS = 2  # supersampling
CX, AMP = 560, 400  # centre of the swings and their amplitude
ROW_TOP, ROW_BOTTOM = 300, 1580
BOB_R = 25
LABEL_X = 34  # the swing counts on the left
TITLE_Y, SUBTITLE_Y = 70, 158
FRACTION_Y, WORDS_Y = 1648, 1748
TIMELINE_Y, TIMELINE_X = 1850, (150, 930)

# --- Motion details ----------------------------------------------------------------
TRAIL = 0.10  # s of comet behind each bob
TRAIL_POINTS = 10
ECHOES = 7  # faint copies of the hidden wave, ECHO_DT apart, behind the live one
ECHO_DT = 0.035  # s
FRACTION_Q = 6  # moments t = p/q T with q <= FRACTION_Q are labelled
FRACTION_WINDOW = 0.32  # s each side of the moment

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#05060c"
BOBS = ["#ffd36e", "#ffa25e", "#ff6f7d", "#e05ab8", "#9a6bff", "#5aa8ff", "#4fe3d0"]
WAVE = "#bfe3ff"
TRACK = "#1b2033"
TEXT, TEXT_DIM = "#eef1fb", "#7d87a3"
GLOW_SIGMA, GLOW_GAIN = 9.0, 1.1

# --- Sound -------------------------------------------------------------------------
STYLE = "mallet_garden"  # marimba, major pentatonic
STEP = np.cos(0.3)  # a pendulum within 0.3 rad of the right end counts as "in step"


def positions(t):
    """x of every bob (px) at loop time t (any real; period 1): AMP cos(2π n t)."""
    return CX + AMP * np.cos(2 * np.pi * np.outer(np.atleast_1d(t), SWINGS))


def hidden_wave(t, u):
    """
    The slowest wave through the bobs at time t, over a continuous row index u
    (u = 0 .. COUNT-1 are the bobs): AMP cos(2π CENTRE t + 2π w (u − mid)) with
    w = t − round(t), the wavenumber folded into [−½, ½].
    """
    w = t - np.round(t)
    mid = (COUNT - 1) / 2
    return CX + AMP * np.cos(2 * np.pi * CENTRE * t + 2 * np.pi * w * (u - mid))


def fractions():
    """Every p/q in [0, 1) with q <= FRACTION_Q in lowest terms, as (p, q, value)."""
    found = {}
    for q in range(1, FRACTION_Q + 1):
        for p in range(q):
            if np.gcd(p, q) == 1:
                found[(p, q)] = p / q
    return sorted(((p, q, v) for (p, q), v in found.items()), key=lambda f: f[2])


def soundtrack(rng):
    """
    Every pendulum strikes a marimba bar each time it reaches its right end, at
    t = m / n for m = 0 .. n−1 (n its swings per loop): 195 strikes per loop. The
    top (longest, slowest) pendulum plays the lowest note, each next one the next
    degree of a major pentatonic scale, three octaves in all, panned by row.
    Loudness follows how many pendulums are in step at that instant (within
    0.3 rad of the right end, from the same closed form), so the scattered middle
    patters softly and the alignments (all 15 at t = 0, the even half at ½ ...)
    ring out as chords. A quiet pad holds the home chord underneath. Everything
    is loop-periodic and circular.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)) - 12, style.mode)
    track = audio.silence(DURATION)
    strikes = 0
    for i, n in enumerate(SWINGS):
        freq = scale.hz(i)
        side = 0.55 * (2 * i / (COUNT - 1) - 1)
        for m in range(n):
            t = m / n
            together = int(np.count_nonzero(np.cos(2 * np.pi * SWINGS * t) > STEP))
            velocity = 0.16 + 0.4 * ((together - 1) / (COUNT - 1)) ** 0.6
            velocity *= 1.0 - 0.25 * i / (COUNT - 1)
            note = style.voice(freq, float(velocity))
            audio.add(track, audio.pan(note, side), t * DURATION)
            strikes += 1
    for midi in scale.chord(0, 3, octave=0):
        bed = audio.pad(float(audio.midi_to_hz(midi)), DURATION, 0.16, 2.0, 2.0, 0.3)
        audio.add(track, bed, 0.0)
    logger.info(f"Soundtrack: {STYLE}, root {scale.root}, {strikes} strikes per loop")
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    A pendulum wave, as a seamless loop. Pendulum i (top to bottom) has
    x_i(t) = CX + AMP cos(2π n_i t) with n_i = 6, 7, ..., 20 swings per loop
    t ∈ [0, 1) (DURATION = 15 s), all released from the right end at t = 0:
    integer harmonics of the loop, so it closes exactly. Across the rows the
    phase is 2π CENTRE t + 2π (i − 7) t, a cosine wave in i of wavenumber t.
    Sampled at integer i, that is the same as wavenumber w = t − round(t) in
    [−½, ½] (aliasing), and the glowing string draws that folded wave over a
    continuous i (it is continuous at t = ½ because sin(2π CENTRE ·½) = 0). At
    t = p/q the wave repeats every q rows, so the bobs fall into q groups; those
    moments (q <= FRACTION_Q) are labelled and marked on the timeline below.

    Each frame is a pure function of t, drawn with cv2 at 2x on the thread pool:
    dim echoes of the string at earlier times, a comet per bob (the closed form
    at TRAIL_POINTS earlier times), shaded bobs coloured top to bottom, then a
    downsample, a half-resolution Gaussian bloom and LaTeX labels built once
    with latex_mask. Nothing is random in the picture: --seed only affects the
    soundtrack (through settings.audio_rng: root note and reverb); the posted
    one used --seed 0.
    """
    settings = settings or ImageProcessingSettings(1)
    shift = 2  # cv2 fractional bits
    one = 1 << shift

    def px(v):
        return np.round(np.asarray(v) * SS * one).astype(np.int32)

    rows = np.linspace(ROW_TOP, ROW_BOTTOM, COUNT)
    u = np.linspace(-0.45, COUNT - 0.55, 600)
    spacing = rows[1] - rows[0]
    u_y = ROW_TOP + u * spacing  # extrapolates past the end rows too
    lut = palette_lut(BOBS, COUNT)[:, ::-1] * 255  # BGR
    bob_colour = [c.tolist() for c in lut]
    highlight = [(0.45 * c + 0.55 * 255).tolist() for c in lut]
    wave = np.array(hex_to_bgr(WAVE), np.float32)
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    marks = fractions()
    W2, H2 = WIDTH * SS, HEIGHT * SS

    def timeline_x(t):
        return TIMELINE_X[0] + (TIMELINE_X[1] - TIMELINE_X[0]) * t

    # The static layer at 2x: the swing tracks and the timeline.
    static = np.full((H2, W2, 3), hex_to_bgr(BACKGROUND), np.uint8)
    track = hex_to_bgr(TRACK)
    for y in rows:
        cv2.line(
            static,
            (px(CX - AMP), px(y)),
            (px(CX + AMP), px(y)),
            track,
            SS,
            cv2.LINE_AA,
            shift,
        )
        for end in (CX - AMP, CX + AMP):
            cv2.line(
                static,
                (px(end), px(y - 8)),
                (px(end), px(y + 8)),
                track,
                SS,
                cv2.LINE_AA,
                shift,
            )
    dim = hex_to_bgr(TEXT_DIM)
    cv2.line(
        static,
        (px(TIMELINE_X[0]), px(TIMELINE_Y)),
        (px(TIMELINE_X[1]), px(TIMELINE_Y)),
        track,
        SS,
        cv2.LINE_AA,
        shift,
    )
    for p, q, v in marks + [(1, 1, 1.0)]:
        h = 26 / q
        cv2.line(
            static,
            (px(timeline_x(v)), px(TIMELINE_Y - h)),
            (px(timeline_x(v)), px(TIMELINE_Y + h)),
            dim if q <= 2 else track,
            SS,
            cv2.LINE_AA,
            shift,
        )

    # Text, once.
    title = latex_mask(
        r"$x_n(t)\, =\, A\,\cos\!\left(\frac{2\pi\, n\, t}{T}\right)$", 58
    )
    subtitle = latex_mask(
        r"$n\, =\, 6{,}\ 7{,}\ 8{,}\ \ldots{,}\ 20\ \mathrm{swings\ in}\ T\, =\, 15\ \mathrm{s}$",
        36,
    )
    row_labels = [latex_mask(rf"${n}$", 30) for n in SWINGS]
    ordinal = {2: "other", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
    fraction_labels = {}
    for p, q, _ in marks:
        if p == 0:
            top = latex_mask(r"$\mathrm{all\ in\ step}$", 60)
            words = latex_mask(r"$\mathrm{back\ where\ they\ started}$", 40)
        else:
            top = latex_mask(rf"$t\, =\, \frac{{{p}}}{{{q}}}\,T$", 60)
            words = latex_mask(
                rf"$\mathrm{{every\ {ordinal[q]}\ pendulum\ in\ step}}$", 40
            )
        fraction_labels[(p, q)] = (top, words)
    text = hex_to_bgr(TEXT)
    values = np.array([v for _, _, v in marks])

    def render(frame: int) -> np.ndarray:
        t = frame / FRAME_COUNT
        canvas = static.copy()

        # Nearest labelled moment, for the label and to brighten the string.
        gap = (t - values + 0.5) % 1.0 - 0.5
        near = int(np.argmin(np.abs(gap)))
        seconds = abs(gap[near]) * DURATION
        strength = float(smoothstep(1 - seconds / FRACTION_WINDOW))

        # The hidden wave: echoes first, the live string on top.
        ys = px(u_y)
        for e in range(ECHOES, -1, -1):
            te = t - e * ECHO_DT / DURATION
            xs = px(hidden_wave(te, u))
            level = (0.32 + 0.25 * strength) * 0.62**e if e else 0.55 + 0.3 * strength
            colour = (bg + level * (wave - bg)).tolist()
            pts = np.stack([xs, ys], 1)
            cv2.polylines(
                canvas,
                [pts],
                False,
                colour,
                SS * (2 if e == 0 else 1),
                cv2.LINE_AA,
                shift,
            )

        # Comets, then the bobs.
        times = t - np.linspace(TRAIL, 0, TRAIL_POINTS + 1) / DURATION
        trail = positions(times)  # (TRAIL_POINTS + 1, COUNT)
        for i in range(COUNT):
            y = px(rows[i])
            c = np.array(bob_colour[i], np.float32)
            for j in range(TRAIL_POINTS):
                f = (j + 1) / TRAIL_POINTS
                colour = (bg + 0.5 * f**2 * (c - bg)).tolist()
                cv2.line(
                    canvas,
                    (px(trail[j, i]), y),
                    (px(trail[j + 1, i]), y),
                    colour,
                    int(SS * (4 + 10 * f)),
                    cv2.LINE_AA,
                    shift,
                )
        now = trail[-1]
        for i in range(COUNT):
            x, y = px(now[i]), px(rows[i])
            cv2.circle(canvas, (x, y), px(BOB_R), bob_colour[i], -1, cv2.LINE_AA, shift)
            cv2.circle(
                canvas,
                (x - px(BOB_R * 0.3), y - px(BOB_R * 0.3)),
                px(BOB_R * 0.38),
                highlight[i],
                -1,
                cv2.LINE_AA,
                shift,
            )

        # Timeline marker.
        cv2.circle(
            canvas,
            (px(timeline_x(t)), px(TIMELINE_Y)),
            px(7),
            text,
            -1,
            cv2.LINE_AA,
            shift,
        )

        image = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        light = np.clip(image - bg, 0, None)
        small = cv2.resize(
            light, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA
        )
        glow = cv2.GaussianBlur(small, (0, 0), GLOW_SIGMA / 2)
        image += GLOW_GAIN * cv2.resize(glow, (WIDTH, HEIGHT))

        blit_mask(image, title, WIDTH // 2, TITLE_Y, text, anchor="centre")
        blit_mask(image, subtitle, WIDTH // 2, SUBTITLE_Y, dim, anchor="centre")
        for i, mask in enumerate(row_labels):
            blit_mask(image, mask, LABEL_X, int(rows[i]) - mask.shape[0] // 2, dim)
        if strength > 0:
            p, q, _ = marks[near]
            top, words = fraction_labels[(p, q)]
            blit_mask(
                image,
                top,
                WIDTH // 2,
                FRACTION_Y,
                text,
                alpha=strength,
                anchor="centre",
            )
            blit_mask(
                image, words, WIDTH // 2, WORDS_Y, dim, alpha=strength, anchor="centre"
            )
        return image

    logger.info(
        f"{COUNT} pendulums, {SWINGS[0]}..{SWINGS[-1]} swings per {DURATION:.0f} s; "
        f"{len(marks)} labelled moments"
    )
    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    track = soundtrack(settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
