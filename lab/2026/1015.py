"""
Prime Spirals

Put every whole number n on the page at distance n from the centre and
angle n radians, and light up only the primes. Up close they scatter like
dust. Pull back and they curl into spiral arms; pull back further and the
arms become dozens of spirals, then hundreds of perfectly straight rays.

None of this is about primes alone. A step of 6 turns almost exactly one
full circle, because 6 ≈ 2π, so numbers 6 apart line up into arms. Further
out 44/7 ≈ 2π takes over, then 710/113 ≈ 2π, the same fraction behind the
famous 355/113 for π. The primes only choose which arms to light: never an
arm whose step shares a factor with the count, a law found by Dirichlet in 1837.

The music climbs with the camera: a bell rings at every step further out,
hurrying through each zoom, and a new chord lands as each pattern appears,
until the last one resolves home.

Made with Python, PyTorch & LaTeX. Sound synthesized in NumPy.
#primenumbers #numbertheory #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from colors.palettes import HOLOGRAPHIC
from common.gpu import gaussian_kernel, blur, select_device, splat, wide_blur
from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

PALETTE = HOLOGRAPHIC + HOLOGRAPHIC[:1]  # cyclic: residue classes go round it
# The camera: half-width of the view (in units of n) at key times, seconds.
# Shallow segments are the holds on each structure.
ZOOM = (
    (0.0, 18.0),
    (1.2, 24.0),
    (3.2, 210.0),
    (4.6, 300.0),
    (6.6, 3800.0),
    (7.9, 4700.0),
    (11.4, 2.0e5),
    (13.0, 2.6e5),
    (15.0, 2.8e5),
)
# Colour primes by residue mod 6, then 44, then 710, crossfading at these times.
MODULI = ((6, 0.0), (44, 5.2), (710, 9.6))
COLOUR_FADE = 1.2  # seconds per crossfade
INTEGERS_FADE = (7.6, 10.0)  # all-integers layer fades out between these times
DOT = 0.3  # dot radius in units of n (until it hits the pixel limits below)
DOT_PX = (1.1, 7.0)
DOT_MIN_EARLY = 2.4  # px: bigger dots while the points are few...
DOT_SHRINK = (8.5, 10.5)  # ...shrinking to DOT_PX[0] over these seconds
INTEGER_LEVEL = 0.35
BLOOM = 0.25
GAIN = 3.0  # tone map 1 - exp(-GAIN I)
CROWDED = 0.05  # pixels per unit of n below which primes dim...
PRIME_FLOOR = 0.4  # ...down to this weight
LABELS = True
# (LaTeX, start, end) in seconds, typeset with the system LaTeX (usetex).
LABEL_TEXT = (
    (r"$n \;\longmapsto\; (r,\theta) = (n,\,n)$", 0.25, 2.3),
    (r"$6 \approx 2\pi$", 2.7, 5.2),
    (r"$\dfrac{44}{7} \approx 2\pi$", 6.3, 9.0),
    (r"$\dfrac{710}{113} \approx 2\pi$", 11.2, 15.0),
)
LABEL_FADE = 0.35
LABEL_SIZE = 40  # points (at 100 dpi)
LABEL_Y = 0.12  # label centre, fraction of the height from the bottom

# Soundtrack: a climb that resolves.
MODE = "major"
ROOTS = range(53, 58)  # melody root, MIDI
CLIMB_NOTES = 30  # bells over the whole zoom, one per equal step of log R
CLIMB_DEGREES = 17  # how far the climb rises, in scale degrees
CHORDS = (1, 5, 3, 0)  # ii under the opening, then vi, IV, I as each structure lands
RISER = 3.0  # seconds of build-up into the last structure


def _primes(limit):
    """Sieve of Eratosthenes: all primes below `limit`."""
    sieve = np.ones(limit, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.flatnonzero(sieve)


def _ramp(t, start, end):
    """0 before `start`, 1 after `end`, eased in between (seconds)."""
    x = np.clip((t - start) / (end - start), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _label_masks():
    """Each label typeset once with LaTeX as an (HEIGHT, WIDTH) alpha mask."""
    masks = []
    style = {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}
    with plt.rc_context(style):
        for text, _, _ in LABEL_TEXT:
            fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
            fig.patch.set_facecolor("k")
            fig.text(
                0.5,
                LABEL_Y,
                text,
                color="w",
                fontsize=LABEL_SIZE,
                ha="center",
                va="center",
            )
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[..., 0]
            masks.append(pixels.astype(np.float32) / 255.0)
            plt.close(fig)
    return masks


def _stage_times():
    """Seconds at which each structure lands: the start of each hold in ZOOM."""
    return [ZOOM[i][0] for i in (2, 4, 6)]


def soundtrack(log_zoom, rng):
    """
    Rise and resolve, driven by the zoom.

    The climb: a bell rings each time log R (the view's half-width) passes
    one of CLIMB_NOTES equal steps between its first and last value, so the
    notes hurry through each zoom and fall silent in each hold. Note k climbs
    CLIMB_DEGREES scale degrees over the video, zig-zagging (up two, back
    one) so it sounds like a melody rather than a ladder, getting louder and
    alternating sides. Under it, a pad holds a chord per stage: ii for the
    opening, then vi, IV and finally I, each landing with a rolled bell chord
    and a marimba bass note as the 6-arm, 44-arm and 710-ray structures
    appear. A riser builds over RISER seconds into the last one, where the
    home chord lands in two octaves and rings out over the ending. One-shot:
    nothing wraps around.
    """
    scale = Scale(int(rng.choice(ROOTS)), MODE)
    track = audio.silence(DURATION)
    stages = _stage_times()

    # The climb.
    seconds = np.linspace(0.0, DURATION, 30001)
    depth = log_zoom(seconds)
    progress = (depth - depth[0]) / (depth[-1] - depth[0])
    step = np.floor(progress * (CLIMB_NOTES - 1e-9)).astype(int)
    onsets = np.concatenate([[0], np.flatnonzero(np.diff(step)) + 1])
    for k, index in enumerate(onsets):
        degree = (k * CLIMB_DEGREES) // CLIMB_NOTES + (k % 2)
        velocity = 0.3 + 0.3 * k / CLIMB_NOTES
        note = audio.pan(audio.bell(scale.hz(degree), velocity), 0.5 * np.sin(2.4 * k))
        audio.add(track, note, seconds[index], circular=False)

    # The harmony: a pad per stage, a rolled chord and a bass note as each lands.
    edges = [0.0] + stages + [DURATION]
    for i, degree in enumerate(CHORDS):
        start, end = edges[i], edges[i + 1]
        for midi in scale.chord(degree, 3, octave=-1):
            tone = audio.pad(
                float(audio.midi_to_hz(midi)),
                end - start + 0.5,
                0.3,
                attack=0.8 if i else 0.3,
                release=1.5,
                brightness=0.2 + 0.2 * i,
            )
            audio.add(track, tone, start, circular=False)
        if i == 0:
            continue
        home = i == len(CHORDS) - 1
        chord = scale.chord(degree, 3)
        if home:
            chord += scale.chord(degree, 3, octave=1)
        for j, midi in enumerate(chord):
            bell = audio.bell(float(audio.midi_to_hz(midi)), 0.5, length=4.0)
            position = -0.4 + 0.8 * j / (len(chord) - 1)
            audio.add(
                track, audio.pan(bell, position), start + 0.05 * j, circular=False
            )
        bass = scale.hz(degree, octave=-2)
        bass = audio.marimba(bass, 0.55 if home else 0.45, length=3.5)
        audio.add(track, bass, start, circular=False)

    riser = audio.riser(RISER, rng, 200.0, 1800.0, 0.35)
    audio.add(track, riser, stages[-1] - RISER, circular=False)

    logger.info(
        f"Soundtrack: root MIDI {scale.root}, {len(onsets)} climb notes, stages at "
        + ", ".join(f"{s:.1f} s" for s in stages)
    )
    track = audio.reverb(track, rng, 2.8, 0.3, tone=6000.0, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Prime spirals (as popularised by 3Blue1Brown): integer n plotted at polar
    (r, θ) = (n, n rad). Integers n and n + q differ in angle by q − 2πk, so
    when q/k is a good rational approximation of 2π (6/1, 44/7, 710/113) the
    residue classes mod q form q slowly turning arms (or, for 710, rays that
    look straight). Primes occupy only the φ(q) classes coprime to q: 2 of 6,
    20 of 44, 280 of 710 (Dirichlet).

    One-shot. The view's half-width follows a monotone PCHIP interpolation of
    log R through ZOOM, with shallow segments that hold on each structure.
    A faint layer of all integers shows the arms the primes skip and fades out
    over INTEGERS_FADE; primes are coloured by their residue mod 6, 44, 710 on
    a cyclic palette, crossfading so each stage's arms get distinct colours.
    Optional labels, typeset with LaTeX, name each approximation.

    Rendering (torch, GPU): every point's (x, y) is computed once in float64
    and stored as float32; a frame scales them to pixels, splats them
    bilinearly, blurs with a Gaussian whose radius is DOT in units of n
    (clamped to DOT_PX, with a larger minimum DOT_MIN_EARLY until DOT_SHRINK)
    and rescaled to keep the peak, tone-maps with
    1 − exp(−I) and adds bloom. Points run up to the farthest visible radius.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()

    log_zoom = PchipInterpolator([k for k, _ in ZOOM], [np.log(r) for _, r in ZOOM])
    # Farthest visible radius: the corner of the widest view.
    limit = int(max(r for _, r in ZOOM) * np.hypot(1.0, HEIGHT / WIDTH) * 1.02) + 2
    integers = np.arange(1, limit, dtype=np.float64)
    primes = _primes(limit).astype(np.float64)
    logger.info(f"{len(integers)} integers, {len(primes)} primes")

    def xy(n):
        return (
            torch.tensor(n * np.cos(n), dtype=torch.float32, device=device),
            torch.tensor(n * np.sin(n), dtype=torch.float32, device=device),
        )

    ix, iy = xy(integers)
    px_, py_ = xy(primes)
    lut = palette_lut(PALETTE, 1024)
    prime_colours = [
        torch.tensor(lut[((primes % m) / m * 1023).astype(int)], device=device)
        for m, _ in MODULI
    ]
    grey = torch.tensor([0.55, 0.6, 0.75], device=device)
    masks = [torch.tensor(m, device=device) for m in _label_masks()]
    canvas = torch.zeros(HEIGHT, WIDTH, 3, device=device)

    def render(frame):
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        half_width = float(np.exp(log_zoom(seconds)))
        scale = 0.5 * WIDTH / half_width  # pixels per unit of n
        low = DOT_MIN_EARLY + (DOT_PX[0] - DOT_MIN_EARLY) * _ramp(seconds, *DOT_SHRINK)
        sigma = float(np.clip(DOT * scale, low, DOT_PX[1]))

        canvas.zero_()
        level = INTEGER_LEVEL * (1.0 - _ramp(seconds, *INTEGERS_FADE))
        if level > 0.0:
            colours = (level * grey).expand(len(integers), 3)
            splat(canvas, 0.5 * WIDTH + ix * scale, 0.5 * HEIGHT - iy * scale, colours)
        colour = prime_colours[0]
        for (_, start), stage in zip(MODULI[1:], prime_colours[1:]):
            w = float(_ramp(seconds, start, start + COLOUR_FADE))
            colour = colour + w * (stage - colour)
        weight = float(
            np.clip(scale / CROWDED, PRIME_FLOOR, 1.0)
        )  # dimmer when dots pile up
        splat(
            canvas,
            0.5 * WIDTH + px_ * scale,
            0.5 * HEIGHT - py_ * scale,
            weight * colour,
        )

        # Dots: blur to the dot radius, keeping a single dot's peak near 1.
        image = blur(canvas, gaussian_kernel(sigma, device)) * (2 * np.pi * sigma**2)
        image = 1.0 - torch.exp(-GAIN * image)
        image = image + BLOOM * wide_blur(image, 18.0, 4)

        if LABELS:
            for mask, (_, start, end) in zip(masks, LABEL_TEXT):
                alpha = _ramp(seconds, start, start + LABEL_FADE) * (
                    1.0 - _ramp(seconds, end - LABEL_FADE, end)
                )
                if end >= DURATION:  # the last label stays
                    alpha = _ramp(seconds, start, start + LABEL_FADE)
                if alpha > 0:
                    image = image + float(0.85 * alpha) * mask[..., None]
        return 255.0 * image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    track = soundtrack(log_zoom, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
