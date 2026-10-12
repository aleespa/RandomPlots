"""
Ptolemy's Pen

Ancient astronomers explained the wandering of the planets with circles
riding on circles: epicycles. They were wrong about the sky, but right
about something deeper. In 1807 Joseph Fourier showed that any closed
curve at all can be drawn this way, if you allow enough circles, each
turning a whole number of times as fast as the first.

Here a chain of thirteen circles spins, large ones slowly and small ones
quickly, and the tip of the last one holds a pen. Only certain speeds
are allowed, which is why the drawing comes out with perfect rotational
symmetry. The sizes of the circles breathe as it draws, so every pass of
the pen traces a slightly different flower over the fading ghost of the
last one.

Every time the pen swings out to the tip of a petal, a marimba bar is
struck. Which note depends on which way the petal points, so the melody
turns with the flower, and the bigger the petal, the louder the note: as
the circles breathe, some petals grow loud enough to be heard and others
fade into silence.

z(t) = Σₖ cₖ e^{ikt}

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#generativeart #fourier #epicycles #fourierseries #mathart
#randomplots #sonification #creativecoding #algorithmicart #mathisbeautiful #animation
"""

import gc

import cv2
import numpy as np
from loguru import logger
from scipy.signal import find_peaks

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

# --- Video -------------------------------------------------------------------
DURATION = 20.0
FRAME_COUNT = frame_count(DURATION)
SUPERSAMPLE = 2  # lines are drawn at this multiple and area-downsampled
SUBPIXEL_BITS = 4  # cv2 fixed-point coordinates

# --- Epicycles ---------------------------------------------------------------
SYMMETRIES = (3, 4, 5, 6, 7)  # m, chosen by the seed: frequencies k = 1 + m j
TERMS = 6  # j = -TERMS .. TERMS, so 2 TERMS + 1 circles
DECAY = 1.3  # amplitude ~ (1 + |j|)^-DECAY
AMPLITUDE_JITTER = 0.25  # log-normal spread of the amplitudes
DRAW_CYCLES = 10  # times the pen goes round per loop (integer -> exact loop)
MORPH = 0.3  # amplitudes breathe by +/- this fraction ...
MORPH_CYCLES = (1,)  # ... a whole number of times per loop
PHASE_DRIFT = (-1, 0, 0, 1)  # whole turns each circle's phase drifts per loop
FIT = 0.9  # largest pen excursion, as a fraction of the half-width

# --- Appearance --------------------------------------------------------------
BACKGROUND_CENTRE = (26, 18, 21)  # BGR
BACKGROUND_EDGE = (8, 6, 7)
CIRCLE_COLOUR = (62, 54, 50)
ARM_COLOUR = (120, 108, 100)
SHOWN_CIRCLES = 13  # the largest circles are drawn; the rest are too small to read
TAIL = 1.0  # trail length, as a fraction of one pass of the pen
TAIL_SAMPLES = 2400
TAIL_BINS = 48  # colour steps along the trail
TAIL_GAMMA = 1.0  # >1 keeps the old trail faint and the fresh ink bright
TRAIL_ANCHORS = [(6, 20, 34), (16, 53, 90), (46, 147, 200), (200, 241, 255)]  # BGR
TRAIL_THICKNESS = 3  # pixels on the supersampled canvas
GUIDE_THICKNESS = 2
HEAD_RADIUS = 4.0  # pixels on the output frame
BLOOM = 0.9
BLOOM_SIGMA = 7.0

# --- Sound -------------------------------------------------------------------
STYLE = "mallet_garden"  # common.music.STYLES: instrument, scale, register, room
BASS_LINE = (0, 4, 3, 2, 1)  # scale degrees two octaves down, one per pass
MAX_NOTE_RATE = 7.0  # notes per second the rhythmic grid may reach
MAX_SUBBEATS = 3  # tips per repeat of the flower shape
GATE = 0.55  # a petal must reach this fraction of the widest to sound
SAMPLES_PER_REPEAT = 2000  # resolution of the tip search


def make_drawing(rng):
    """
    Random epicycle coefficients and their slow, periodic breathing.

    Returns a function pen(t) giving the complex pen position for loop
    times t in [0, 1), and a function terms(t) giving each circle's
    rotating vector, largest circle first.
    """
    m = int(rng.choice(SYMMETRIES))
    j = np.arange(-TERMS, TERMS + 1)
    k = 1 + m * j
    amplitude = (1.0 + np.abs(j)) ** -DECAY
    amplitude *= np.exp(AMPLITUDE_JITTER * rng.standard_normal(len(j)))
    phase = rng.uniform(0.0, 2.0 * np.pi, len(j))
    morph_q = rng.choice(MORPH_CYCLES, len(j))
    morph_phase = rng.uniform(0.0, 2.0 * np.pi, len(j))
    drift = rng.choice(PHASE_DRIFT, len(j))

    order = np.argsort(-amplitude, kind="stable")
    k, amplitude, phase = k[order], amplitude[order], phase[order]
    morph_q, morph_phase, drift = morph_q[order], morph_phase[order], drift[order]

    def terms(t):
        t = np.asarray(t, float)[..., None]
        size = amplitude * (
            1.0 + MORPH * np.sin(2.0 * np.pi * morph_q * t + morph_phase)
        )
        angle = phase + 2.0 * np.pi * (drift + k * DRAW_CYCLES) * t
        return size * np.exp(1j * angle)

    def pen(t):
        return terms(t).sum(axis=-1)

    return m, pen, terms


def background():
    """A dark radial gradient, computed once."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    r = np.hypot((x - WIDTH / 2) / (WIDTH / 2), (y - HEIGHT / 2) / (HEIGHT / 2))
    w = np.clip(r / 1.3, 0.0, 1.0)[..., None]
    centre = np.array(BACKGROUND_CENTRE, np.float32)
    edge = np.array(BACKGROUND_EDGE, np.float32)
    return (centre * (1.0 - w) + edge * w).astype(np.uint8)


def trail_lut():
    """TAIL_BINS BGR colours rising through TRAIL_ANCHORS, oldest to newest."""
    anchors = np.array(TRAIL_ANCHORS, float)
    u = np.linspace(0.0, 1.0, TAIL_BINS) ** TAIL_GAMMA * (len(anchors) - 1)
    lo = np.minimum(u.astype(int), len(anchors) - 2)
    frac = (u - lo)[:, None]
    lut = anchors[lo] * (1.0 - frac) + anchors[lo + 1] * frac
    return [tuple(int(round(v)) for v in c) for c in lut]


def soundtrack(m, pen, rng):
    """
    A note struck whenever the pen reaches a petal tip (a marimba by default).

    The flower shape repeats DRAW_CYCLES * m times per loop (once per
    symmetry sector), so the tips of the average repeat define a fixed
    rhythmic grid of up to MAX_SUBBEATS beats. On each beat of each repeat
    the bar is struck at the moment the pen actually peaks there, if that
    petal currently reaches GATE of the widest; louder for wider petals.
    The pitch is fixed per symmetry sector (a seeded melody in the STYLE's
    scale), the stereo position follows the tip's x, and a bass note marks
    each pass. Notes wrap around the end and the room is circular, so the
    sound loops with the picture.
    """
    style = STYLES[STYLE]
    repeats = DRAW_CYCLES * m
    n = repeats * SAMPLES_PER_REPEAT
    z = pen(np.arange(n) / n)
    r = np.abs(z)
    r_max = r.max()

    # Cut the loop into repeats at the quietest point of the average shape.
    offset = int(np.argmin(r.reshape(repeats, -1).mean(axis=0)))
    shape = np.roll(r, -offset).reshape(repeats, -1)
    profile = shape.mean(axis=0)
    beats = int(np.clip(MAX_NOTE_RATE * DURATION // repeats, 1, MAX_SUBBEATS))
    peaks, _ = find_peaks(profile, prominence=0.01 * np.ptp(profile))
    if len(peaks) == 0:
        peaks = np.array([np.argmax(profile)])
    peaks = np.sort(peaks[np.argsort(-profile[peaks])][:beats])
    gaps = np.diff(np.concatenate([peaks, [peaks[0] + SAMPLES_PER_REPEAT]]))
    half = int(min(gaps.min() // 2, SAMPLES_PER_REPEAT // 4))

    scale = Scale(int(rng.choice(style.roots)), style.mode)
    melody = rng.permutation(8)[:m]  # scale degree for each symmetry sector

    track = audio.silence(DURATION)
    struck = 0
    for repeat in range(repeats):
        for b, p in enumerate(peaks):
            lo, hi = max(p - half, 0), min(p + half + 1, SAMPLES_PER_REPEAT)
            i = lo + int(np.argmax(shape[repeat, lo:hi]))
            j = (repeat * SAMPLES_PER_REPEAT + i + offset) % n
            when = j / n * DURATION
            if b == 0 and repeat % m == 0:
                bass = scale.hz(BASS_LINE[repeat // m % len(BASS_LINE)], octave=-2)
                audio.add(track, style.bass(bass, 0.55, length=4.0), when)
            reach = r[j] / r_max
            if reach < GATE:
                continue
            velocity = (0.25 + 0.5 * (reach - GATE) / (1.0 - GATE)) * 0.8**b
            freq = scale.hz(melody[repeat % m] + 2 * b)
            sound = style.voice(freq, velocity)
            audio.add(track, audio.pan(sound, 0.7 * z[j].real / r_max), when)
            struck += 1

    logger.info(
        f"Soundtrack ({STYLE}): {struck} notes on a grid of {beats} x {repeats}, "
        f"{60 * repeats / DURATION:.0f} repeats/min, root MIDI {scale.root}"
    )
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    A looping Fourier-epicycle drawing with m-fold symmetry.

    The pen follows z(t) = sum_k c_k(t) exp(2 pi i k D t) over loop time
    t in [0, 1), with D = DRAW_CYCLES passes per loop. Only frequencies
    k = 1 + m j are used, so z(s + 2 pi / m) = exp(2 pi i / m) z(s) and the
    figure has m-fold rotational symmetry. Amplitudes fall off as
    (1 + |j|)^-DECAY with log-normal jitter and random phases; each one
    breathes sinusoidally a whole number of times per loop and its phase
    drifts by whole turns, so c_k(t) is periodic and the video loops exactly.

    Every frame is a closed-form function of t, so frames are rendered
    independently on render_frames' thread pool (NumPy and cv2 release the GIL). Each
    frame draws the largest SHOWN_CIRCLES circles centred on the partial
    sums (largest first), the arm through all partial sums, and the pen's
    actual history over the last TAIL of a pass -- ghosts of earlier,
    slightly different figures included -- coloured from dark through gold
    to cream towards the pen. Lines are drawn additively with anti-aliased
    cv2 primitives on a SUPERSAMPLE x canvas, area-downsampled, given a
    gaussian bloom and added to a dark radial background.

    The soundtrack (see `soundtrack`) is synthesized from the same pen(t),
    after the frames, from settings.audio_rng, and muxed into the MP4.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    m, pen, terms = make_drawing(rng)
    extent = np.abs(pen(np.linspace(0.0, 1.0, 20_000, endpoint=False))).max()
    scale = FIT * (WIDTH / 2) / extent  # output pixels per unit
    fixed = SUPERSAMPLE * (1 << SUBPIXEL_BITS)
    base = background()
    lut = trail_lut()
    tail_u = np.linspace(-TAIL / DRAW_CYCLES, 0.0, TAIL_SAMPLES)
    chunk = int(np.ceil(TAIL_SAMPLES / TAIL_BINS))

    def to_fixed(z):
        x = (WIDTH / 2 + scale * z.real) * fixed
        y = (HEIGHT / 2 - scale * z.imag) * fixed
        return np.stack([x, y], axis=-1).round().astype(np.int32)

    def render(frame):
        t = frame / FRAME_COUNT
        layer = np.zeros((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE, 3), np.uint8)

        vectors = terms(t)
        joints = to_fixed(np.concatenate([[0.0], np.cumsum(vectors)]))
        radii = np.round(np.abs(vectors) * scale * fixed).astype(int)
        for i in range(SHOWN_CIRCLES):
            cv2.circle(
                layer,
                tuple(joints[i]),
                int(radii[i]),
                CIRCLE_COLOUR,
                GUIDE_THICKNESS,
                cv2.LINE_AA,
                SUBPIXEL_BITS,
            )
        cv2.polylines(
            layer,
            [joints],
            False,
            ARM_COLOUR,
            GUIDE_THICKNESS,
            cv2.LINE_AA,
            SUBPIXEL_BITS,
        )

        trail = to_fixed(pen(t + tail_u))
        for b in range(TAIL_BINS):
            piece = trail[b * chunk : (b + 1) * chunk + 1]
            if len(piece) > 1:
                cv2.polylines(
                    layer,
                    [piece],
                    False,
                    lut[b],
                    TRAIL_THICKNESS,
                    cv2.LINE_AA,
                    SUBPIXEL_BITS,
                )
        cv2.circle(
            layer,
            tuple(trail[-1]),
            int(HEAD_RADIUS * fixed),
            lut[-1],
            -1,
            cv2.LINE_AA,
            SUBPIXEL_BITS,
        )

        lines = cv2.resize(layer, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(lines, (0, 0), BLOOM_SIGMA)
        return cv2.addWeighted(cv2.add(base, lines), 1.0, glow, BLOOM, 0.0)

    logger.info(f"Ptolemy's Pen, {m}-fold, {2 * TERMS + 1} circles")
    render_frames(settings, render, FRAME_COUNT)
    track = soundtrack(m, pen, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
