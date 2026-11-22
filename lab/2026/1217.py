"""
Spiral Arms Are a Traffic Jam

A spiral galaxy turning, and the surprise is that its spiral arms are not made of the
same stars. Follow any bright star and it sails straight through the arms and out the
other side. The stars move on; the spiral stays.

Every star circles the centre on a slightly flattened orbit, an ellipse. Line all
the ellipses up and the arms vanish. Turn each one a little further than the one
inside it and they crowd together along two curves: the spiral arms. Stars slow
down and bunch up there, like cars at a jam that never clears even though no car
stays in it. Astronomers call this a density wave, and it is how galaxies keep
their spirals for billions of years.

φ(a) = k ln a: the tilt of an orbit grows with its size.

Listen: a bamboo flute plays each time a highlighted star reaches the far end of
its orbit, where it slows down in the jam, higher for the inner, faster stars,
over a soft choir that drifts while the arms vanish and comes home when they
return.

Made with Python & PyTorch. Sound synthesized in NumPy.
#densitywave #spiralgalaxy #patternformation #mathart #randomplots
"""

import gc

import numpy as np
import torch

from common import audio
from common.gpu import gaussian_kernel, blur, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    latex_mask,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = "The stars move. The spiral doesn't."
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
TAU = 2 * np.pi

# Galaxy on screen.
CENTRE = (WIDTH / 2, 860.0)
SCALE_PX = 470.0  # pixels per unit of semi-major axis
STARS = 160_000
BULGE = 24_000
FIELD = 1_500  # faint background stars

# Orbits: semi-major axis a in [A_MIN, A_MAX], tilt phi(a) = k ln(a / A_BAR) outside the
# bar radius (aligned inside it), axis ratio b/a = 1 - ECC(a).
A_MIN, A_MAX = 0.03, 1.08
A_BAR = 0.13
ECC = 0.3
TWIST = 2.6  # k of the spiral
DISC_SCALE = 0.32  # exponential disc scale length
OMEGA = 0.11  # turns per second at a = 1 (flat rotation curve, omega ~ 1/a)
OMEGA_CORE = 0.12  # softening of 1/a near the centre
PATTERN = 0.018  # turns per second of the whole pattern

SCATTER_ECC = 0.035  # per-star spread of the axis ratio
SCATTER_TILT = 0.07  # ... and of the orbit's tilt, rad
# A tracer's head: a small disc of splats.
HEAD = [
    (dx, dy)
    for dx in (-2, -1, 0, 1, 2)
    for dy in (-2, -1, 0, 1, 2)
    if dx * dx + dy * dy < 5
]

ELLIPSES = 26
ELLIPSE_SAMPLES = 2400
TRACERS = 9
TRAIL = 1.6  # seconds of trail behind a tracer
TRAIL_SAMPLES = 240

# Story (seconds): the ellipses appear, untwist into a bar, twist back to a spiral.
ELLIPSE_IN = (3.6, 5.0)
UNTWIST = (7.4, 9.6)
RETWIST = (10.6, 12.9)
ELLIPSE_OUT = (12.9, 13.9)
# (tex, start, end) of the captions on screen
LABELS = (
    (r"$\mathrm{every\ star\ sails\ through\ the\ arms}$", 0.3, 4.0),
    (r"$\mathrm{each\ one\ circles\ on\ an\ ellipse}$", 4.3, 7.4),
    (r"$\mathrm{line\ them\ up\ and\ the\ arms\ vanish}$", 7.7, 10.4),
    (r"$\mathrm{turn\ each\ a\ bit\ more\ and\ they\ return}$", 10.7, 13.0),
    (r"$\mathrm{a\ traffic\ jam\ of\ stars}$", 13.3, 15.5),
)
LABEL_SIZE = 50
LABEL_Y = 1560
LABEL_FADE = 0.35

# Soundtrack.
STYLE = "bamboo_flute"  # common.music.STYLES
FLUTE_LENGTH = 1.0  # s a flute note is held
MIN_GAP = 0.22  # s between flute notes, at least
END_NOTE = 13.3  # s: the closing low note, with the last caption
# (chord degree, start, end) of the choir bed: home, suspended while unwound, home
BED = ((0, 0.0, 7.6), (1, 7.4, 10.8), (0, 10.6, 13.6))

BACKGROUND = "#03040a"
# Disc colour from the core outwards: warm old stars to blue young ones.
DISC = ("#ffd7a1", "#ffe9cf", "#dfe6ff", "#9fbcff", "#6f93ff")
ELLIPSE_COLOUR = "#4fd6ff"
TRACER_COLOUR = "#ffb86b"
LABEL_COLOUR = "#e9ecf5"


def _rgb(colour):
    return [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]


def twist(seconds):
    """k(t): the spiral's twist, unwound to a bar and wound back up."""
    down = smoothstep((seconds - UNTWIST[0]) / (UNTWIST[1] - UNTWIST[0]))
    up = smoothstep((seconds - RETWIST[0]) / (RETWIST[1] - RETWIST[0]))
    return TWIST * (1.0 - down + up)  # a scalar or an array, like `seconds`


def ellipse_alpha(seconds):
    rise = smoothstep((seconds - ELLIPSE_IN[0]) / (ELLIPSE_IN[1] - ELLIPSE_IN[0]))
    fall = smoothstep((seconds - ELLIPSE_OUT[0]) / (ELLIPSE_OUT[1] - ELLIPSE_OUT[0]))
    return float(rise * (1.0 - fall))


def label_alpha(seconds, start, end):
    rise = smoothstep((seconds - start) / LABEL_FADE)
    fall = smoothstep((end - seconds) / LABEL_FADE)
    return float(rise * fall)


def angular_speed(a):
    """Phase speed (rad/s) round an orbit of semi-major axis a: a flat rotation curve."""
    return TAU * OMEGA / (a + OMEGA_CORE) * (1.0 + OMEGA_CORE)


def tilt(a, k):
    """Orientation of the orbit of size a: aligned in the bar, k ln(a / A_BAR) outside."""
    return k * torch.log(torch.clamp(a / A_BAR, min=1.0))


def orbit_xy(a, psi, k, seconds, ecc=ECC, offset=0.0):
    """
    Screen position of a star at phase psi on the orbit of size a, at time `seconds`;
    `ecc` and `offset` (extra tilt) give each star its own scatter about the wave.
    """
    b = a * (1.0 - ecc * torch.clamp(a / A_BAR, max=1.0) ** 0.5)
    x, y = a * torch.cos(psi), b * torch.sin(psi)
    angle = tilt(a, k) + offset + TAU * PATTERN * seconds
    c, s = torch.cos(angle), torch.sin(angle)
    px = CENTRE[0] + SCALE_PX * (c * x - s * y)
    py = CENTRE[1] - SCALE_PX * (s * x + c * y)
    return px, py


def generate(settings: ImageProcessingSettings = None):
    """
    A kinematic density wave (Lindblad, Kalnajs): stars on nested elliptical orbits.

    Every star has a semi-major axis a (an exponential disc plus a compact bulge), an
    axis ratio b/a = 1 - ECC (rounder in the bar), and a phase psi(t) = psi0 +
    omega(a) t with omega ~ 1/a (a flat rotation curve), so inner stars lap outer ones.
    The ellipse of size a is tilted by phi(a) = k ln(a / A_BAR) outside the bar radius,
    and the whole pattern turns slowly at PATTERN turns/s. Because psi advances
    uniformly, stars linger at the ends of their ellipses; where neighbouring ellipses
    crowd, the lingering stars pile up into two arms (k > 0) or a bar (k = 0).

    One-shot, 15 s, every frame a closed form in t: the spiral with nine highlighted
    tracer stars (trails = the same closed form at t - dt); the ellipses fade in;
    k eases to 0 (bar) and back to TWIST (spiral); the ellipses fade out; rest.
    Rendered on the GPU: all stars splatted bilinearly, a tight and a wide Gaussian
    bloom, an exponential tone map, LaTeX captions composited as alpha masks.
    Soundtrack: see `soundtrack`; the posted render is --seed 0 (audio seed 0).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    # --- Stars, drawn once ---------------------------------------------------------
    a_disc = rng.gamma(1.7, DISC_SCALE, STARS)
    a_disc = a_disc[(a_disc > A_MIN) & (a_disc < A_MAX)]
    a_bulge = np.abs(rng.normal(0.0, 0.07, BULGE)) + 0.001
    a_all = np.concatenate([a_disc, a_bulge])
    psi0 = rng.uniform(0.0, TAU, len(a_all))
    lum = rng.lognormal(-0.2, 0.6, len(a_all))
    lut = palette_lut(DISC, 256)
    shade = np.clip(a_all / A_MAX + rng.normal(0.0, 0.06, len(a_all)), 0.0, 1.0)
    colour = lut[(shade * 255).astype(int)] * lum[:, None]
    colour[len(a_disc) :] *= 0.5  # the bulge is a dense, softer glow
    # Scatter about the wave (a velocity dispersion): broadens the arms.
    ecc = ECC + rng.normal(0.0, SCATTER_ECC, len(a_all))
    offset = rng.normal(0.0, SCATTER_TILT, len(a_all))
    offset[len(a_disc) :] = rng.uniform(0.0, TAU, BULGE)  # the bulge has no wave

    a_t = torch.tensor(a_all, device=device, dtype=torch.float32)
    psi_t = torch.tensor(psi0, device=device, dtype=torch.float32)
    omega_t = angular_speed(a_t)
    colour_t = torch.tensor(colour, device=device, dtype=torch.float32)
    ecc_t = torch.tensor(ecc, device=device, dtype=torch.float32)
    offset_t = torch.tensor(offset, device=device, dtype=torch.float32)

    # Tracers: bright stars at chosen radii; their trails are earlier positions.
    tracer_a = np.linspace(0.3, 0.95, TRACERS) + rng.uniform(-0.02, 0.02, TRACERS)
    tracer_psi = rng.uniform(0.0, TAU, TRACERS)
    tr_a = torch.tensor(tracer_a, device=device, dtype=torch.float32)[:, None]
    tr_psi = torch.tensor(tracer_psi, device=device, dtype=torch.float32)[:, None]
    lag = torch.linspace(TRAIL, 0.0, TRAIL_SAMPLES, device=device)[None, :]
    lag_s = np.linspace(TRAIL, 0.0, TRAIL_SAMPLES)
    trail_weight = (1.0 - lag / TRAIL) ** 2
    tracer_rgb = torch.tensor(_rgb(TRACER_COLOUR), device=device)

    # Ellipse skeleton.
    ell_a = torch.linspace(A_BAR * 0.8, 1.0, ELLIPSES, device=device)[:, None]
    ell_psi = torch.linspace(0.0, TAU, ELLIPSE_SAMPLES, device=device)[None, :]
    ell_a, ell_psi = torch.broadcast_tensors(ell_a, ell_psi)
    ellipse_rgb = torch.tensor(_rgb(ELLIPSE_COLOUR), device=device)

    # Background field and the dark sky, static.
    sky = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    fx = torch.tensor(rng.uniform(0, WIDTH, FIELD), device=device, dtype=torch.float32)
    fy = torch.tensor(rng.uniform(0, HEIGHT, FIELD), device=device, dtype=torch.float32)
    fl = rng.lognormal(-2.2, 0.7, FIELD)[:, None] * lut[rng.integers(80, 256, FIELD)]
    splat(sky, fx, fy, torch.tensor(fl, device=device, dtype=torch.float32))
    background = torch.tensor(_rgb(BACKGROUND), device=device) * 255.0

    # Captions.
    labels = []
    for tex, start, end in LABELS:
        mask = torch.tensor(latex_mask(tex, LABEL_SIZE), device=device)
        labels.append((mask, start, end))
    label_rgb = torch.tensor(_rgb(LABEL_COLOUR), device=device) * 255.0

    tight = gaussian_kernel(1.6, device)
    canvas = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    lines = torch.zeros(HEIGHT, WIDTH, 3, device=device)

    def render(frame):
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        k = float(twist(seconds))
        canvas.copy_(sky)
        px, py = orbit_xy(a_t, psi_t + omega_t * seconds, k, seconds, ecc_t, offset_t)
        splat(canvas, px, py, colour_t * 0.16)

        lines.zero_()
        alpha = ellipse_alpha(seconds)
        if alpha > 0:
            ex, ey = orbit_xy(ell_a, ell_psi, k, seconds)
            weight = torch.full((ex.numel(), 1), 0.4 * alpha, device=device)
            splat(lines, ex, ey, weight * ellipse_rgb)

        # Tracer trails: the same closed form at earlier times (k at those times too).
        when = seconds - lag
        tk = torch.tensor(twist(seconds - lag_s), device=device, dtype=torch.float32)
        angle_shift = tk[None, :]
        tx, ty = _tracer_xy(tr_a, tr_psi, when, angle_shift)
        glow = (trail_weight * 0.9).expand_as(tx).reshape(-1, 1) * tracer_rgb
        splat(lines, tx, ty, glow)
        head_x, head_y = tx[:, -1], ty[:, -1]
        for dx, dy in HEAD:
            splat(lines, head_x + dx, head_y + dy, 0.5 * tracer_rgb.expand(TRACERS, 3))

        light = canvas + 0.6 * blur(canvas, tight) + 0.9 * wide_blur(canvas, 26.0, 4)
        light = light + lines + 1.2 * blur(lines, tight) + 0.5 * wide_blur(lines, 14, 2)
        image = background + (255.0 - background) * (1.0 - torch.exp(-1.6 * light))

        for mask, start, end in labels:
            a = label_alpha(seconds, start, end)
            if a <= 0:
                continue
            h, w = mask.shape
            x0, y0 = int(WIDTH / 2 - w / 2), LABEL_Y
            m = (a * mask)[:, :, None]
            region = image[y0 : y0 + h, x0 : x0 + w]
            image[y0 : y0 + h, x0 : x0 + w] = region * (1 - m) + m * label_rgb
        return image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    track = soundtrack(tracer_a, tracer_psi, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def _tracer_xy(a, psi0, when, k):
    """Tracer positions (TRACERS, samples) at times `when` with twists `k`."""
    psi = psi0 + angular_speed(a) * when
    b = a * (1.0 - ECC)
    x, y = a * torch.cos(psi), b * torch.sin(psi)
    angle = k * torch.log(torch.clamp(a / A_BAR, min=1.0)) + TAU * PATTERN * when
    c, s = torch.cos(angle), torch.sin(angle)
    return (
        CENTRE[0] + SCALE_PX * (c * x - s * y),
        CENTRE[1] - SCALE_PX * (s * x + c * y),
    )


def soundtrack(tracer_a, tracer_psi, rng):
    """
    Bamboo flute over a quiet choir, from the tracers' closed forms.

    A tracer reaches the far end of its ellipse (psi = n pi, where it is slowest and
    lingers in the jam) at t = (n pi - psi0) / omega(a): each such moment is a flute
    note, higher for inner (faster) tracers, the end of the major axis alternating
    between two neighbouring degrees, panned by the tracer's x on screen. Notes closer
    than MIN_GAP to the previous one are skipped. Underneath, a soft choir ("u")
    holds a quartal chord on the home degree while the arms are there, moves to a
    suspended one while they are unwound (k -> 0), and comes home as they return;
    a long low flute note lands on the closing caption. One-shot: nothing wraps.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)

    events = []
    for i, (a, psi0) in enumerate(zip(tracer_a, tracer_psi)):
        omega = TAU * OMEGA / (a + OMEGA_CORE) * (1.0 + OMEGA_CORE)
        n_first = int(np.ceil(psi0 / np.pi))
        n_last = int(np.floor((psi0 + omega * DURATION) / np.pi))
        for n in range(n_first, n_last + 1):
            when = (n * np.pi - psi0) / omega
            if not 0.15 <= when <= END_NOTE - 0.4:
                continue
            angle = twist(when) * np.log(max(a / A_BAR, 1.0)) + TAU * PATTERN * when
            x = (1 if n % 2 == 0 else -1) * a * np.cos(angle)
            events.append((when, i, n % 2, x))
    events.sort()
    last = -1.0
    for when, i, side, x in events:
        if when - last < MIN_GAP:
            continue
        last = when
        degree = (TRACERS - 1 - i) // 2 + 2 * side
        velocity = 0.42 + 0.2 * (1.0 - i / (TRACERS - 1))
        note = audio.flute(scale.hz(degree), velocity, length=FLUTE_LENGTH)
        audio.add(track, audio.pan(note, 0.75 * x), when, circular=False)
    final = audio.flute(scale.hz(0, octave=-1), 0.6, length=DURATION - END_NOTE)
    audio.add(track, audio.pan(final, 0.0), END_NOTE, circular=False)

    for degree, start, end in BED:
        for pitch in scale.chord(degree, 3, octave=-1):
            hz = float(audio.midi_to_hz(pitch))
            held = audio.choir(
                hz, end - start, 0.22, attack=1.2, release=1.6, vowel="u"
            )
            audio.add(track, held, start, circular=False)
        root = float(audio.midi_to_hz(scale.note(degree, octave=-2)))
        audio.add(
            track, audio.pad(root, end - start, 0.25, 1.0, 1.6, 0.2), start, False
        )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
