"""
On the Critical Line

A journey up through the Riemann zeta function, the most famous function
in mathematics. Each glowing thread is a direction: along it, ζ(s) points
the same way. Where all twelve threads meet in a starburst, ζ(s) is zero.

Watch where the zeros land. Every single one sits exactly on the same
faint vertical line, the line where the real part of s is one half. That
the zeros always do this, forever, is the Riemann hypothesis. It has been
open since 1859, it decides how the prime numbers are spread out, and
proving it is worth a million dollars. Ten trillion zeros have been
checked by computer. All on the line.

Listen: each zero strikes a piano note as it lands, higher as the zeros
climb, until the last one lands on the home chord. Those heights
are the hidden frequencies of the primes: Riemann counts the prime numbers
with a sum of waves, one wave for every zero.

ζ(s) = 1 + 1/2ˢ + 1/3ˢ + 1/4ˢ + …

Made with Python & PyTorch. Sound synthesized in NumPy.
#riemannhypothesis #zetafunction #complexanalysis #mathart #randomplots
"""

import gc
import math

import numpy as np
import torch
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from common import audio
from common.gpu import select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# Imaginary parts of the first 29 non-trivial zeros, all at real part 1/2.
ZEROS = (
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719,
    43.327073, 48.005151, 49.773832, 52.970321, 56.446248, 59.347044, 60.831779,
    65.112544, 67.079811, 69.546402, 72.067158, 75.704691, 77.144840, 79.337375,
    82.910381, 84.735493, 87.425275, 88.809111, 92.491899, 94.651344, 95.870634,
    98.831194,
)  # fmt: skip

# --- Zeta (Euler-Maclaurin) -------------------------------------------------------
TERMS = 40  # N: terms summed directly; needs N >> |s| / 2 pi
BERNOULLI = (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510)

# --- View ------------------------------------------------------------------------
RE_RANGE = (-4.3, 3.5)  # real parts across the frame; the critical line at 60 %
CLIMB_END = ZEROS[-1]  # the camera comes to rest centred on this zero
CRUISE_IN = 1.0  # s to reach full climbing speed
CRUISE_OUT = (9.5, 13.4)  # s over which the climb slows to a stop

# --- Look ------------------------------------------------------------------------
PHASE_LINES = 12  # threads per full turn of arg zeta
RING_STEP = 1.0  # |zeta| contour every factor 2^RING_STEP
LINE_WIDTH = 1.1  # thread half-width, px
RING_LEVEL = 0.16  # brightness of the |zeta| contours
CRITICAL_LEVEL = 0.28  # brightness of the critical line
PHASE_PALETTE = ["#38e1ff", "#6b7bff", "#c04dff", "#ff4fa3", "#ffc04d", "#38e1ff"]
BACKGROUND = "#05060a"
BLOOM, BLOOM_SIGMA = 0.9, 7.0
FLASH_WIDTH = 0.7  # how long (in units of height) a zero glows around the centre
LABEL = r"$\rho = \dfrac{{1}}{{2}} + {gamma:.2f}\,i$"  # mathtext, Computer Modern
LABEL_SIZE = 25  # pt at 100 dpi
LABEL_COLOUR = (0.93, 0.93, 0.95)
LABEL_GAP = 22  # px between the critical line and the label

# --- Sound -----------------------------------------------------------------------
# "primes": every zero sings a pure tone at its own height, the frequencies of
# the explicit formula for the primes; or a common.music.STYLES name, which
# turns the heights into a rising melody. An --audio-only tag naming one wins.
SOUND = "felt_piano"
TONE_HZ = 10.0  # Hz per unit of height: 14.13 sings at 141 Hz, 98.83 at 988 Hz
TONE_SUSTAIN = 1.5  # s for a zero's tone to fade by a factor e
MELODY_SPAN = 14  # scale degrees from the first zero to the last (STYLES sounds)


def zeta_grid(sigma, t, device):
    """
    zeta(sigma_x + i t_y) on the grid of real parts `sigma` (columns) and
    heights `t` (rows), by Euler-Maclaurin summation in float64:

        zeta(s) = sum_{n<N} n^-s + N^(1-s)/(s-1) + N^-s/2
                  + sum_k B_2k/(2k)! s(s+1)...(s+2k-2) N^(-s-2k+1).

    n^-s = n^-sigma e^(-i t log n) splits into a column and a row factor, so
    the direct sum is one (H x N) @ (N x W) complex matrix product.
    """
    logs = torch.log(torch.arange(1, TERMS + 1, device=device, dtype=torch.float64))
    rows = torch.polar(torch.ones(len(t), TERMS, device=device, dtype=torch.float64),
                       -t[:, None] * logs[None, :])  # fmt: skip
    cols = torch.exp(-logs[:, None] * sigma[None, :]).to(torch.complex128)
    direct = rows[:, :-1] @ cols[:-1]
    s = sigma[None, :] + 1j * t[:, None]
    tail = rows[:, -1:] * cols[-1:]  # N^-s
    total = direct + TERMS * tail / (s - 1) + 0.5 * tail
    factor, fact = s * tail / TERMS, 2.0
    for k, b in enumerate(BERNOULLI, start=1):
        total = total + (b / fact) * factor
        factor = factor * (s + 2 * k - 1) * (s + 2 * k) / TERMS**2
        fact *= (2 * k + 1) * (2 * k + 2)
    return total


def climb(seconds):
    """Height at the centre of the frame after `seconds`: eased in, cruising, eased out."""
    grid = np.linspace(0.0, DURATION, 6001)
    speed = smoothstep(grid / CRUISE_IN) * (
        1 - smoothstep((grid - CRUISE_OUT[0]) / (CRUISE_OUT[1] - CRUISE_OUT[0]))
    )
    height = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2)])
    return CLIMB_END * np.interp(seconds, grid, height / height[-1])


def line_field(q, width):
    """
    Antialiased lines where q crosses an integer, `width` px wide whatever
    q's gradient: the distance to the nearest integer divided by |grad q| (in
    px, by finite differences of the wrapped q). Also returns the line spacing
    in px, used to fade lines that crowd closer than a few pixels.
    """
    gx = torch.zeros_like(q)
    gy = torch.zeros_like(q)
    dx = q[:, 1:] - q[:, :-1]
    dy = q[1:, :] - q[:-1, :]
    gx[:, :-1] = dx - torch.round(dx)
    gy[:-1, :] = dy - torch.round(dy)
    gx[:, -1], gy[-1, :] = gx[:, -2], gy[-2, :]
    gradient = torch.sqrt(gx * gx + gy * gy).clamp(min=1e-6)
    distance = torch.abs(q - torch.round(q)) / gradient
    alpha = (1 - distance / width).clamp(0, 1)
    return alpha, 1 / gradient


def crossing_times():
    """The second at which each zero passes the centre of the frame."""
    grid = np.linspace(0.0, DURATION, 15001)
    heights = climb(grid)
    index = np.searchsorted(heights, np.array(ZEROS) - 1e-9)
    return grid[np.minimum(index, len(grid) - 1)]


def label_sprite(gamma: float) -> np.ndarray:
    """The zero's label, typeset by matplotlib mathtext in Computer Modern, as an alpha mask."""
    fig = Figure(figsize=(5, 1.2), dpi=100)
    fig.patch.set_alpha(0)
    fig.text(
        0.02,
        0.5,
        LABEL.format(gamma=gamma),
        fontsize=LABEL_SIZE,
        color="white",
        va="center",
        math_fontfamily="cm",
    )
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    alpha = np.asarray(canvas.buffer_rgba())[..., 3].astype(np.float32) / 255
    rows, cols = np.flatnonzero(alpha.any(1)), np.flatnonzero(alpha.any(0))
    return alpha[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def tone(freq: float, velocity: float, start: float) -> np.ndarray:
    """A pure sine from `start` to the end, soft 8 ms attack, fading by e every TONE_SUSTAIN s."""
    t = np.arange(int((DURATION - start) * audio.SAMPLE_RATE)) / audio.SAMPLE_RATE
    attack = 1 - np.exp(-t / 0.008)
    return velocity * attack * np.exp(-t / TONE_SUSTAIN) * np.sin(2 * np.pi * freq * t)


def soundtrack(times, sound, rng):
    """
    A one-shot timed to the zeros: each sounds as it passes the centre (the
    flash), at `times`, and the last one, where the camera comes to rest,
    rings out.

    "primes": zero k sings a pure tone at TONE_HZ * gamma_k, panned a little
    left or right in turn, fading over TONE_SUSTAIN, quieter as it climbs
    (gamma^-0.4) so the high ones don't glare; the tones pile up into an
    inharmonic shimmer. Under it, from the first second, a soft hum at the
    first zero's frequency and its octave, swelling in. These heights are the
    frequencies of the oscillations in Riemann's explicit formula for the
    prime-counting function.

    A STYLES name: each zero strikes the style's voice on a scale degree
    rising with log gamma over MELODY_SPAN degrees, over a soft home-chord
    pad; the last zero lands on a rolled home chord and a low root.
    """
    kw = dict(circular=False)
    track = audio.silence(DURATION)
    gammas = np.array(ZEROS)
    if sound == "primes":
        for k, (when, gamma) in enumerate(zip(times, gammas)):
            velocity = 0.5 * (gammas[0] / gamma) ** 0.4
            note = tone(TONE_HZ * gamma, velocity, when)
            audio.add(track, audio.pan(note, 0.35 if k % 2 else -0.35), when, **kw)
        for freq, gain in (
            (TONE_HZ * gammas[0], 0.7),
            (2 * TONE_HZ * gammas[0], 0.3),
        ):
            hum = audio.pad(freq, DURATION - 1.5, gain, 0.5, 1.5, 0.15, 4.0)
            audio.add(track, hum, 0.0, **kw)
            audio.add(track, tone(freq, 0.6 * gain, 0.0), 0.0, **kw)  # its entrance
        room = dict(rt60=3.2, wet=0.4, tone=5000.0)
        summary = f"pure tones {TONE_HZ * gammas[0]:.0f}-{TONE_HZ * gammas[-1]:.0f} Hz"
    else:
        style = STYLES[sound]
        scale = Scale(int(rng.choice(style.roots)), style.mode)
        rise = np.log(gammas / gammas[0]) / np.log(gammas[-1] / gammas[0])
        lift = -1 if scale.root >= 66 else 0  # keep the melody under MIDI ~84
        audio.add(track, style.voice(scale.hz(0, lift), 0.35), 0.0, **kw)
        audio.add(track, style.bass(scale.hz(0, -2), 0.4, length=3.0), 0.0, **kw)
        for k, (when, r) in enumerate(zip(times[:-1], rise[:-1])):
            degree = int(round(MELODY_SPAN * r))
            note = style.voice(scale.hz(degree, lift), 0.3 + 0.2 * r)
            audio.add(track, audio.pan(note, 0.35 if k % 2 else -0.35), when, **kw)
        for note in scale.chord(0, 3, octave=-1):
            pad = audio.pad(
                float(audio.midi_to_hz(note)), DURATION - 2.0, 0.22, 1.5, 2.0
            )
            audio.add(track, pad, 0.0, **kw)
        home = times[-1]
        audio.add(track, style.bass(scale.hz(0, -2), 0.7, length=4.0), home, **kw)
        for i, note in enumerate(scale.chord(0, 3, octave=lift + 1)):
            voice = style.voice(float(audio.midi_to_hz(note)), 0.45)
            audio.add(track, audio.pan(voice, 0.5 * (i - 1)), home + 0.05 * i, **kw)
        room = dict(rt60=style.rt60, wet=style.wet, tone=style.tone)
        summary = f"{sound}, root MIDI {scale.root}"
    logger.info(
        f"Soundtrack: {summary}; zeros from {times[0]:.2f} s to {times[-1]:.2f} s"
    )
    track = audio.reverb(track, rng, circular=False, **room)
    audio.fade(track, 0.0, 0.8)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The Riemann zeta function on a window that climbs the critical strip.

    The window spans RE_RANGE in Re s (Re s = 1/2 sits at 60 % of the width) and
    the same scale vertically; its centre rises from height 0 to the 29th
    zero (98.83) along `climb`: speed eases in over CRUISE_IN, cruises and
    eases out over CRUISE_OUT, then rests. Each frame evaluates zeta at every
    pixel (`zeta_grid`, float64 on the GPU) and draws:

    - phase lines: where arg zeta crosses a multiple of 2 pi / PHASE_LINES,
      coloured by the phase through a cyclic palette; all of them meet at
      every zero (12 threads in a star) and wind around the pole at s = 1;
    - faint |zeta| contours every factor 2^RING_STEP;
    - lines are antialiased to a fixed pixel width from the field's own
      gradient and fade where they crowd under ~3 px apart (the left side,
      where |zeta| grows like t^(1/2 - sigma));
    - a hairline on Re s = 1/2, a flash on each zero as it passes the centre
      (Gaussian in centre height minus zero height), and a label
      rho = 1/2 + gamma i, typeset once per zero with matplotlib mathtext
      (Computer Modern) and alpha-blended in as it does;
    - a quarter-resolution bloom of the lines.

    The soundtrack (`soundtrack`) is timed by the same climb: each zero
    sounds at the second it passes the centre.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()

    re0, re1 = RE_RANGE
    scale = (re1 - re0) / WIDTH  # units per pixel
    half_height = scale * HEIGHT / 2
    sigma = re0 + scale * (
        torch.arange(WIDTH, device=device, dtype=torch.float64) + 0.5
    )
    offsets = half_height - scale * (
        torch.arange(HEIGHT, device=device, dtype=torch.float64) + 0.5
    )
    critical_x = (0.5 - re0) / scale

    stops = torch.tensor(
        [[int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)] for c in PHASE_PALETTE],
        device=device,
    )
    lut_size = 1024
    u = torch.linspace(0, len(PHASE_PALETTE) - 1, lut_size, device=device)
    low = u.floor().long().clamp(max=len(PHASE_PALETTE) - 2)
    lut = stops[low] + (u - low)[:, None] * (stops[low + 1] - stops[low])
    background = torch.tensor(
        [int(BACKGROUND[i : i + 2], 16) / 255 for i in (1, 3, 5)], device=device
    )
    xs = torch.arange(WIDTH, device=device, dtype=torch.float32)
    ys = torch.arange(HEIGHT, device=device, dtype=torch.float32)
    critical = (
        CRITICAL_LEVEL * torch.exp(-((xs - critical_x) ** 2) / 0.8)[None, :, None]
    )
    sprites = [torch.tensor(label_sprite(g), device=device) for g in ZEROS]
    label_colour = torch.tensor(LABEL_COLOUR, device=device)

    def render(frame):
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        centre = float(climb(seconds))
        z = zeta_grid(sigma, centre + offsets, device)
        phase = torch.angle(z).float()
        q = phase * (PHASE_LINES / (2 * math.pi))
        alpha, spacing = line_field(q, LINE_WIDTH)
        alpha = alpha * ((spacing - 2.0) / 4.0).clamp(0, 1)
        rings, ring_spacing = line_field(
            torch.log2(torch.abs(z).float().clamp(min=1e-30)) / RING_STEP, LINE_WIDTH
        )
        rings = RING_LEVEL * rings * ((ring_spacing - 3.0) / 6.0).clamp(0, 1)

        hue = ((phase / (2 * math.pi)) % 1.0) * (lut_size - 1)
        colour = lut[hue.long()]
        lines = alpha[..., None] * colour + rings[..., None] * (0.5 + 0.5 * colour)
        lines = lines + critical

        # zeros near the centre flash
        for gamma in ZEROS:
            y = HEIGHT / 2 - (gamma - centre) / scale
            if -40 < y < HEIGHT + 40:
                flash = math.exp(-(((centre - gamma) / FLASH_WIDTH) ** 2))
                r2 = (xs[None, :] - critical_x) ** 2 + (ys[:, None] - y) ** 2
                glow = (0.25 + 0.75 * flash) * torch.exp(
                    -r2 / 18.0
                ) + flash * 0.35 * torch.exp(-r2 / 900.0)
                lines = lines + glow[..., None] * (0.7 + 0.3 * lut[0])

        image = background + lines + BLOOM * wide_blur(lines, BLOOM_SIGMA, 4)
        image = image.clamp(0, 1)

        # labels fade in as each zero passes the centre and stay while on screen
        for gamma, sprite in zip(ZEROS, sprites):
            seen = 0.85 * float(smoothstep((centre - gamma) / FLASH_WIDTH + 1.0))
            h, w = sprite.shape
            y0 = int(round(HEIGHT / 2 - (gamma - centre) / scale - h / 2))
            x0 = int(critical_x) + LABEL_GAP
            top, bottom = max(y0, 0), min(y0 + h, HEIGHT)
            if seen < 0.01 or top >= bottom:
                continue
            a = seen * sprite[top - y0 : bottom - y0, :, None]
            region = image[top:bottom, x0 : x0 + w]
            region.mul_(1 - a).add_(a * label_colour)
        return 255 * image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    sound = settings.audio_tag if getattr(settings, "audio_tag", "") else SOUND
    track = soundtrack(crossing_times(), sound, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
