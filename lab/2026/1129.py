"""
The Star Hidden in an Ellipse

Ripples run inwards from the rim of an ellipse, every point moving straight
along the ellipse's normal at the same speed. Near the middle they crumple into
folds and sharp cusps, and every cusp slides along one glowing four-pointed
star. Green is a ripple on its way in, coral is where it has turned inside out.

That star is the ellipse's evolute, the curve of its centres of curvature: a
ripple folds exactly where it has travelled one radius of curvature. Huygens
invented evolutes in 1673 to build a better pendulum clock. By the four-vertex
theorem, every smooth closed oval that isn't a circle has an evolute with at
least four cusps; an ellipse has exactly four.

Listen: every ripple leaves the rim with a soft pluck. The bright ones play a
bamboo flute, held for exactly as long as their folds live: from the moment
the cusps are born at the top and bottom points of the star until they meet
again at its sides.

Made with Python, NumPy & PyTorch. Sound synthesized in NumPy.
#evolute #differentialgeometry #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Every ellipse hides a star'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- Geometry (pixels; x right, y up, origin at the centre of the ellipse) ----------
SEMI_X, SEMI_Y = 430.0, 600.0  # semi-axes: the major axis is vertical
CENTRE = (WIDTH / 2, HEIGHT / 2)
EMISSIONS = 45  # ripples per loop: three a second
SPEED = 165.0  # px/s along the normals
REACH = 1400.0  # px travelled before a ripple has faded away
STRONG = 6  # bright ripples per loop
STRONG_GAP = 6  # at least this many emissions between two bright ones
POINTS = 12288  # points per ripple, uniform in arclength of the rim

# --- Look ---------------------------------------------------------------------------
BACKGROUND = "#040908"
BEFORE, AFTER = "#3fe0a8", "#ff8a7a"  # jade until a point focuses, coral after
FOLD_WIDTH = 0.12  # |J| over which the colour turns at a fold
RIM_LIGHT = 1.0
FOLD_BOOST, FOLD_CAP = 0.35, 3.0  # extra light near folds: min(cap, |J|^-boost)
STAR_LIGHT = 0.12  # the evolute, always faintly there
SPARK_LIGHT, SPARK_TRAIL = (
    2.5,
    45.0,
)  # cusp sparks and their trail along the star (px of d)
EXPOSURE, GAMMA = 1.3, 0.75
SOFT = 0.7
GLOWS = ((6.0, 2, 0.18), (30.0, 6, 0.10))  # (sigma px, shrink, gain)
FADE_IN = 60.0  # px: a ripple brightens as it leaves the rim
FADE_OUT = 0.5  # last fraction of REACH over which it dims away

# --- Sound --------------------------------------------------------------------------
STYLE = (
    "bamboo_flute"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
MELODY = (4, 2, 3, 0, 1, -1, 2, 0)  # scale degrees, walked by the bright ripples


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def rim():
    """
    POINTS points of the ellipse, uniform in arclength: position, outward unit
    normal, signed radius of curvature, and the angle parameter theta.
    """
    theta = np.linspace(0.0, 2 * np.pi, 200_001)
    speed = np.hypot(SEMI_X * np.sin(theta), SEMI_Y * np.cos(theta))
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    arc *= 2 * np.pi / 200_000
    perimeter = arc[-1]
    s = (np.arange(POINTS) + 0.5) * perimeter / POINTS
    th = np.interp(s, arc, theta)
    pos = np.stack([SEMI_X * np.cos(th), SEMI_Y * np.sin(th)], 1)
    normal = np.stack([SEMI_Y * np.cos(th), SEMI_X * np.sin(th)], 1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    rho = (SEMI_X**2 * np.sin(th) ** 2 + SEMI_Y**2 * np.cos(th) ** 2) ** 1.5 / (
        SEMI_X * SEMI_Y
    )
    return pos, normal, rho, th, perimeter


def evolute():
    """
    POINTS points of the evolute P - rho N, uniform in its own arclength, with
    the radius of curvature rho of the rim point each one belongs to.
    """
    theta = np.linspace(0.0, 2 * np.pi, 200_001)
    a2s, b2c = SEMI_X**2 * np.sin(theta) ** 2, SEMI_Y**2 * np.cos(theta) ** 2
    rho = (a2s + b2c) ** 1.5 / (SEMI_X * SEMI_Y)
    c = SEMI_Y**2 - SEMI_X**2
    ex = -c / SEMI_X * np.cos(theta) ** 3  # the ellipse's evolute (an "astroid")
    ey = c / SEMI_Y * np.sin(theta) ** 3
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ex), np.diff(ey)))])
    s = (np.arange(POINTS) + 0.5) * arc[-1] / POINTS
    th = np.interp(s, arc, theta)
    pts = np.stack([np.interp(th, theta, ex), np.interp(th, theta, ey)], 1)
    return pts, np.interp(th, theta, rho), arc[-1]


def ripple_amplitudes(rng):
    """Weak ripples with a little variety, and STRONG bright ones spread round the loop."""
    amp = rng.uniform(0.25, 0.45, EMISSIONS)
    while True:
        strong = np.sort(rng.choice(EMISSIONS, STRONG, replace=False))
        gaps = np.diff(np.r_[strong, strong[0] + EMISSIONS])
        if gaps.min() >= STRONG_GAP:
            break
    amp[strong] = rng.uniform(1.2, 1.4, STRONG)
    return amp, strong


def soundtrack(amp, strong, style_name, rng):
    """
    Bamboo flute over plucked drops; a loop, so every tail wraps.

    Ripple e leaves the rim at e * DURATION / EMISSIONS: a soft high pluck,
    velocity from its amplitude, panned alternately. A ripple travelled d
    folds where d equals the local radius of curvature, so its cusps are born
    at d = rho_min = SEMI_X^2 / SEMI_Y (the star's top and bottom tips) and
    die at d = rho_max = SEMI_Y^2 / SEMI_X (its side tips). Each bright ripple
    holds a flute note over exactly that span, pitched by MELODY. A quiet low
    pad on the root and fifth keeps the room breathing.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    period = DURATION / EMISSIONS
    rho_min, rho_max = SEMI_X**2 / SEMI_Y, SEMI_Y**2 / SEMI_X
    born, held = rho_min / SPEED, (rho_max - rho_min) / SPEED

    for e in range(EMISSIONS):
        degree = 7 + (e * 3) % 5
        drop = audio.pluck(scale.hz(degree), 0.15 + 0.1 * float(amp[e]), 1.5)
        side = 0.45 if e % 2 else -0.45
        audio.add(track, audio.pan(audio.lowpass(drop, 3500.0), side), e * period)

    for k, e in enumerate(strong):
        degree = MELODY[k % len(MELODY)]
        if style.voice is audio.flute:
            note = audio.flute(scale.hz(degree), 0.55, held)
        else:
            note = style.voice(scale.hz(degree), 0.55)
        audio.add(track, audio.pan(note, 0.0), e * period + born)

    for semitones in (0, 7):
        hz = float(audio.midi_to_hz(scale.root - 24 + semitones))
        audio.add(track, audio.pad(hz, DURATION, 0.16, 2.0, 2.0, 0.3), 0.0)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, flute held {held:.2f} s "
        f"from {born:.2f} s after each bright ripple"
    )
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    Parallel curves (wavefronts) of an ellipse, a loop.

    The rim is the ellipse (SEMI_X cos th, SEMI_Y sin th), sampled with POINTS
    points uniform in arclength. A ripple that has travelled d inwards is
    X = P - d N (N the outward unit normal): the parallel curve at distance
    d, i.e. the wavefront of light leaving the rim along its normals. Its
    length element is J = 1 - d / rho times the rim's (rho the radius of
    curvature), so it folds into cusps where d = rho, and those cusps lie on
    the evolute P - rho N, the four-cusped "astroid" of the ellipse.

    A ripple leaves the rim every DURATION / EMISSIONS s and moves at SPEED,
    so ripple e has d = SPEED (tau - e DURATION / EMISSIONS); a ripple lives
    REACH / SPEED < DURATION s, and emission indices are taken mod EMISSIONS,
    so the loop closes exactly. `settings.rng` draws each ripple's amplitude
    and which STRONG ripples are bright (that is all the seed changes).

    Rendering (torch, GPU when available): every live ripple's points are
    splatted bilinearly with equal weight per unit of rim length, so the
    brightness of a front is its ray density, proportional to 1 / |J|: the
    folds blaze on their own and trace the evolute. Colour is jade where
    J > 0 and coral where J < 0 (the point has passed its centre of
    curvature), mixed over |J| < FOLD_WIDTH. Near a fold each point is further
    brightened by min(FOLD_CAP, |J|^-FOLD_BOOST). The evolute itself (closed form
    (-(a^2-b^2)/b cos^3 th, (a^2-b^2)/a sin^3 th)) is drawn faintly, and each
    ripple lights the evolute points with rho <= d with an exp(-(d - rho) /
    SPARK_TRAIL) tail: sparks riding its cusps along the star. A faint rim, two-scale bloom,
    exponential tone mapping. The soundtrack is `soundtrack`; the posted
    one used --seed 0 (audio seed 0).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    amp, strong = ripple_amplitudes(rng)
    logger.info(f"Bright ripples: {strong.tolist()}")

    device = select_device()
    f32 = dict(device=device, dtype=torch.float32)
    pos, normal, rho, _, perimeter = rim()
    weight = perimeter / POINTS  # rim length per point: a J = 1 front sums ~1 per px
    px0 = torch.tensor(CENTRE[0] + pos[:, 0], **f32)
    py0 = torch.tensor(CENTRE[1] - pos[:, 1], **f32)
    nx = torch.tensor(normal[:, 0], **f32)
    ny = torch.tensor(-normal[:, 1], **f32)
    inv_rho = torch.tensor(1.0 / rho, **f32)
    before = torch.tensor(_rgb(BEFORE), **f32)
    after = torch.tensor(_rgb(AFTER), **f32)
    background = torch.tensor(_rgb(BACKGROUND), **f32)

    soft = gaussian_kernel(SOFT, device)
    rim_canvas = torch.zeros(HEIGHT, WIDTH, 3, **f32)
    splat(
        rim_canvas,
        px0,
        py0,
        (RIM_LIGHT * weight) * before.expand(POINTS, 3).contiguous(),
    )
    star, star_rho, star_length = evolute()
    star_weight = star_length / POINTS
    sx = torch.tensor(CENTRE[0] + star[:, 0], **f32)
    sy = torch.tensor(CENTRE[1] - star[:, 1], **f32)
    star_rho = torch.tensor(star_rho, **f32)
    white = torch.ones(3, **f32)
    spark_colour = 0.5 * (before + after) + 0.5 * white
    splat(
        rim_canvas,
        sx,
        sy,
        (STAR_LIGHT * star_weight) * spark_colour.expand(POINTS, 3).contiguous(),
    )
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **f32)
    period = DURATION / EMISSIONS

    def render(frame):
        tau = frame / FRAME_COUNT * DURATION
        canvas.copy_(rim_canvas)
        newest = int(np.floor(tau / period))
        oldest = int(np.ceil((tau - REACH / SPEED) / period))
        for e in range(oldest, newest + 1):
            d = SPEED * (tau - e * period)
            if d <= 0 or d >= REACH:
                continue
            level = float(amp[e % EMISSIONS])
            level *= float(smoothstep(d / FADE_IN))
            level *= float(smoothstep((1 - d / REACH) / FADE_OUT))
            if level <= 0:
                continue
            jac = 1.0 - d * inv_rho
            mix = (0.5 - 0.5 * torch.tanh(jac / FOLD_WIDTH))[:, None]
            colour = (1 - mix) * before + mix * after
            boost = jac.abs().clamp_min(1e-6) ** -FOLD_BOOST
            colour = colour * boost.clamp_max(FOLD_CAP)[:, None]
            splat(canvas, px0 - d * nx, py0 - d * ny, (level * weight) * colour)
            # The cusps of this ripple sit where rho = d; light the star behind them.
            behind = d - star_rho
            trail = torch.exp(-behind / SPARK_TRAIL) * (behind >= 0)
            trail = trail * (behind < 6 * SPARK_TRAIL)
            spark = (SPARK_LIGHT * level * star_weight) * trail
            splat(canvas, sx, sy, spark[:, None] * spark_colour)
        light = blur(canvas, soft)
        glow = sum(g * wide_blur(canvas, s, f) for s, f, g in GLOWS)
        image = 1 - torch.exp(-EXPOSURE * (light + glow).clamp_min(0) ** GAMMA)
        return (background + image).clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    tag = settings.audio_tag
    track = soundtrack(amp, strong, tag if tag in STYLES else STYLE, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
