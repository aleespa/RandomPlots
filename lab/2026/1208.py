"""
Two Smoke Rings Playing Leapfrog

Two smoke rings fly up the screen, one behind the other, cut in half so you see each one as
a pair of spinning curls. The ring behind shrinks, speeds up and shoots straight through the
one in front. Then the new ring behind does the same. They take turns all the way up.

Each ring is a vortex, a loop of spinning air, and every vortex pushes on the other. The front
ring's spin widens the ring behind it and slows it down; the ring behind squeezes the front one
narrow and fast. A narrow ring travels faster, so the roles swap, again and again. Helmholtz
predicted this leapfrogging in 1858. Here each curl is a point vortex, and the faint smoke lines
are the still air they pass through, pulled up and wound into spirals.

Listen: a flute note sounds each time one ring passes through the other, over a soft chord
that follows the rings up the screen.

Made with Python & PyTorch. Sound synthesized in NumPy.
#vortexring #fluiddynamics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger
from scipy.integrate import solve_ivp

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
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
HOOK = 'Two smoke rings take turns flying through each other'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
TAU = 2 * np.pi

# The two vortex pairs (axisymmetric slice): right cores at (+x_i, z_i) with
# circulation -GAMMA, mirrored at (-x_i, z_i) with +GAMMA, so both move up (+z).
GAMMA = 1.0
WIDE = 1.0  # half-width of the wider pair at the moment they are level
NARROW = 0.42  # ... and of the narrower one (Love: leapfrog for NARROW > 0.382)
CORE = 0.075  # Lamb-Oseen core radius that regularises the tracers' velocity

# Screen: world units are the wide pair's half-width.
UNIT_PX = 300.0
Z_FIRST = 1.35  # centre of the vortices above the bottom edge at t = 0 (units)
EXIT_TIME = 14.6  # s: the vortices' centre is EXIT_MARGIN above the top edge
EXIT_MARGIN = 0.5
PHASE = -0.32  # sim time at t = 0, in passes (0 = level): the first pass ~0.7 s in
WARMUP = 0.9  # sim time the cores' dye spins before t = 0
EASE_IN = 0.18  # s: the clock starts gently
FADE_OUT = 0.35  # s: fade to the background at the end
DT_MAX = 0.01  # largest RK4 step for the dye (sim time)

# Dye.
CORE_DYE_RADIUS = 0.30
CORE_DYE_OFFSET = (0.07, 0.09)  # disc centre: inwards and below the core
CORE_DYE_BAND = 0.075  # spacing of its concentric bands
CORE_DYE = 70_000  # particles per core
LINE_SPACING = 0.24  # units between smoke lines
LINES_BELOW = 10  # lines seeded under the bottom edge
LINE_PARTICLES = 16_000  # per half line
LINE_HALF_WIDTH = 3.2  # units: well past the frame's edges (the air is drawn in)

BACKGROUND = ("#04050a", "#0a0d18")  # bottom, top
RING_COLOURS = ("#ff7b3a", "#3fd7c6")  # the pair that starts wide, the narrow one
LINE_COLOUR = "#c9d3e6"
GAINS = (0.10, 0.10, 0.22)  # optical gain per particle density: ring A, ring B, lines
BLOOM = 0.55

# Text
TEXT_COLOUR = "#e8e4da"
TITLE = r"$\mathrm{two\ smoke\ rings{,}\ cut\ in\ half}$"
SUBTITLE = r"$\mathrm{watch\ them\ take\ turns}$"
TEXT_SIZES = (66, 46)
TEXT_Y = (140, 236)
TEXT_TIMES = (0.15, 3.6, 0.5)  # fade in by, start fading out at, fade length

# Sound
STYLE = (
    "bamboo_flute"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
PASS_DEGREES = (0, 2, 4, 3, 5, 4, 7, 6, 9)  # flute melody, one degree per pass
MID_DEGREES = (-1, 1, 2, 1, 3, 2, 4)  # softer answers half-way between passes
CHORDS = (0, 5, 3, 4)  # pad progression, scale degrees


def _rgb(colour):
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


# ------------------------------------------------------------------ the vortices
def _rhs(_, state):
    """Point-vortex dynamics of the symmetric pair of pairs (4 dof)."""
    x = state[0::2]
    z = state[1::2]
    xs = np.concatenate((x, -x))
    zs = np.concatenate((z, z))
    gs = np.array((-GAMMA, -GAMMA, GAMMA, GAMMA))
    out = np.zeros(4)
    for i in range(2):
        dx, dz = x[i] - xs, z[i] - zs
        r2 = dx * dx + dz * dz
        r2[i] = np.inf
        out[2 * i] = np.sum(-gs * dz / r2) / TAU
        out[2 * i + 1] = np.sum(gs * dx / r2) / TAU
    return out


class Orbit:
    """
    The leapfrogging orbit, solved once (it is integrable: the symmetric
    four-vortex problem has two degrees of freedom), dense in sim time. Sim
    time 0 is a moment the pairs are level, the narrow one passing through.
    """

    def __init__(self, s_min, s_max):
        y0 = np.array((WIDE, 0.0, NARROW, 0.0))
        kw = dict(rtol=1e-10, atol=1e-12, dense_output=True)
        self.back = solve_ivp(_rhs, (0, s_min), y0, **kw)
        self.fwd = solve_ivp(_rhs, (0, s_max), y0, **kw)
        # Passes (pairs level) and the mean speed over two of them.
        s = np.linspace(0, s_max, 200_001)
        y = self.fwd.sol(s)
        gap = y[3] - y[1]
        k = np.nonzero(np.sign(gap[1:]) != np.sign(gap[:-1]))[0]
        self.passes = s[k] - gap[k] * (s[k + 1] - s[k]) / (gap[k + 1] - gap[k])
        self.period = float(np.mean(np.diff(self.passes)))  # sim time per pass
        z_mid = 0.5 * (y[1] + y[3])
        self.speed = float(np.interp(2 * self.period, s, z_mid) / (2 * self.period))

    def __call__(self, s):
        """(x1, z1, x2, z2) at sim time(s) s."""
        s = np.asarray(s, float)
        out = np.where(
            s >= 0, self.fwd.sol(np.maximum(s, 0)), self.back.sol(np.minimum(s, 0))
        )
        return out


def clock_shape(seconds):
    """Video seconds -> eased seconds: starts at zero speed, full speed by ~3 EASE_IN."""
    seconds = np.asarray(seconds, float)
    return seconds - EASE_IN * (1 - np.exp(-seconds / EASE_IN))


class Timeline:
    """Video time <-> sim time and the lab-frame offset of the vortices."""

    def __init__(self):
        probe = Orbit(-10, 400)
        self.period = probe.period
        s0 = PHASE * probe.period
        height = HEIGHT / UNIT_PX
        travel = (height + EXIT_MARGIN) - Z_FIRST
        # Mean-z of the vortices at s0 is ~ speed * s0; solve for the rate.
        z_mid = lambda s: 0.5 * (probe(s)[1] + probe(s)[3])
        target = z_mid(s0) + travel
        lo, hi = s0, s0 + 400
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if z_mid(mid) < target else (lo, mid)
        self.s0 = s0
        self.rate = (lo - s0) / float(clock_shape(EXIT_TIME))
        self.s_end = self.sim(DURATION)
        self.orbit = Orbit(s0 - WARMUP - 1, self.s_end + 1)
        self.z_shift = Z_FIRST - z_mid(s0)  # lab z = orbit z + z_shift

    def sim(self, seconds):
        return self.s0 + self.rate * clock_shape(seconds)

    def video(self, s):
        """Inverse of sim (by bisection, vectorised)."""
        s = np.asarray(s, float)
        lo, hi = np.zeros_like(s), np.full_like(s, DURATION * 2)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = self.sim(mid) < s
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def vortices(self, s):
        """(4, 3) array of lab (x, z, circulation) for sim time s."""
        x1, z1, x2, z2 = self.orbit(s)
        z1, z2 = z1 + self.z_shift, z2 + self.z_shift
        return np.array(
            (
                (x1, z1, -GAMMA),
                (x2, z2, -GAMMA),
                (-x1, z1, GAMMA),
                (-x2, z2, GAMMA),
            )
        )


# ----------------------------------------------------------------- the dye
class Dye:
    """Passive tracers advected (RK4, fixed max step) by the regularised field."""

    def __init__(self, timeline, rng, device):
        self.tl = timeline
        self.device = device
        f = dict(device=device, dtype=torch.float32)
        s_seed = timeline.s0 - WARMUP
        vort = timeline.vortices(s_seed)
        # The flow is mirror-symmetric about the axis x = 0, so only the right
        # half (x > 0) is advected; the renderer draws it twice.
        # Core dye: an off-centre disc of concentric bands beside each core,
        # which the core's differential spin winds into a spiral of arms.
        groups, xs, zs = [], [], []
        for j in range(2):  # the right-hand cores: wide pair, narrow pair
            cx = vort[j, 0] - CORE_DYE_OFFSET[0]
            cz = vort[j, 1] - CORE_DYE_OFFSET[1]
            r = CORE_DYE_RADIUS * np.sqrt(rng.random(4 * CORE_DYE))
            keep = rng.random(r.size) < 0.5 + 0.5 * np.cos(TAU * r / CORE_DYE_BAND)
            r = r[keep][:CORE_DYE]
            a = rng.uniform(0, TAU, r.size)
            xs.append(np.abs(cx + r * np.cos(a)))
            zs.append(cz + r * np.sin(a))
            groups.append(np.full(r.size, j))
        # Smoke lines: the still air, from below the frame (it is drawn up
        # into the wake) to above its top.
        top = HEIGHT / UNIT_PX + 0.6
        levels = LINE_SPACING * (0.5 + np.arange(int(top / LINE_SPACING)))
        below = levels[0] - LINE_SPACING * np.arange(LINES_BELOW, 0, -1)
        levels = np.concatenate((below, levels))
        for level in levels:
            x = (np.arange(LINE_PARTICLES) + 0.5) * LINE_HALF_WIDTH / LINE_PARTICLES
            xs.append(x)
            zs.append(np.full(LINE_PARTICLES, level) + rng.normal(0, 0.002, x.size))
            groups.append(np.full(LINE_PARTICLES, 2))
        self.x = torch.tensor(np.concatenate(xs), **f)
        self.z = torch.tensor(np.concatenate(zs), **f)
        self.group = torch.tensor(np.concatenate(groups), device=device)
        self.s = s_seed
        logger.info(
            f"Dye: {self.x.numel():,} tracers, {len(levels)} smoke lines, "
            f"pass every {timeline.period / timeline.rate:.2f} s"
        )

    @staticmethod
    def velocity(x, z, vort):
        """Regularised Biot-Savart velocity of the 4 vortices (vort: (4, 3) tensor)."""
        dx = x[:, None] - vort[:, 0]
        dz = z[:, None] - vort[:, 1]
        r2 = (dx * dx + dz * dz).clamp_min(1e-12)
        f = (-torch.expm1(-r2 / CORE**2)) / (TAU * r2) * vort[:, 2]
        return -(dz * f).sum(1), (dx * f).sum(1)

    def advance(self, s_target):
        """Step the tracers from self.s to s_target."""
        span = s_target - self.s
        if span <= 0:
            return
        n = int(np.ceil(span / DT_MAX))
        h = span / n
        times = self.s + h * np.arange(2 * n + 1) / 2
        vorts = torch.tensor(
            np.stack([self.tl.vortices(s) for s in times]),
            device=self.device,
            dtype=torch.float32,
        )
        x, z = self.x, self.z
        for k in range(n):
            v0, vh, v1 = vorts[2 * k], vorts[2 * k + 1], vorts[2 * k + 2]
            k1 = self.velocity(x, z, v0)
            k2 = self.velocity(x + 0.5 * h * k1[0], z + 0.5 * h * k1[1], vh)
            k3 = self.velocity(x + 0.5 * h * k2[0], z + 0.5 * h * k2[1], vh)
            k4 = self.velocity(x + h * k3[0], z + h * k3[1], v1)
            x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            z = z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.x, self.z, self.s = x, z, s_target


# ----------------------------------------------------------------- the picture
class Renderer:
    def __init__(self, timeline, rng, device):
        self.tl = timeline
        self.device = device
        f = dict(device=device, dtype=torch.float32)
        self.dye = Dye(timeline, rng, device)
        self.canvas = torch.zeros(HEIGHT, WIDTH, 3, **f)
        self.kernel = gaussian_kernel(1.1, device)
        rows = torch.linspace(1, 0, HEIGHT, **f)[:, None, None]
        bottom, top = (torch.tensor(_rgb(c), **f) for c in BACKGROUND)
        self.background = (bottom + (top - bottom) * rows).expand(HEIGHT, WIDTH, 3)
        self.colours = [
            torch.tensor(_rgb(c), **f) for c in RING_COLOURS + (LINE_COLOUR,)
        ]
        self.gains = GAINS
        unit = torch.eye(3, **f)
        self.weights = unit[self.dye.group]  # (N, 3): one density channel per group
        self.text = [
            latex_mask(tex, size) for tex, size in zip((TITLE, SUBTITLE), TEXT_SIZES)
        ]
        self.text_colour = _rgb(TEXT_COLOUR)[::-1].astype(np.float32)  # BGR

    def frame(self, seconds):
        s = float(self.tl.sim(seconds))
        self.dye.advance(s)
        dx = self.dye.x * UNIT_PX
        py = HEIGHT - self.dye.z * UNIT_PX
        self.canvas.zero_()
        splat(self.canvas, WIDTH / 2 + dx, py, self.weights)
        splat(self.canvas, WIDTH / 2 - dx, py, self.weights)  # the mirror half
        density = blur(self.canvas, self.kernel)
        image = self.background.clone()
        for c in range(3):
            ink = -torch.expm1(-self.gains[c] * density[..., c : c + 1])
            image = image + (self.colours[c] - image) * ink * (0.95 if c < 2 else 0.75)
        image = image + BLOOM * wide_blur(image - self.background, 14.0, 4)
        fade = smoothstep((DURATION - seconds) / FADE_OUT)
        image = self.background + (image - self.background) * float(fade)
        image = image.flip(-1).clamp(0, 1) * 255.0  # BGR
        alpha = self.text_alpha(seconds)
        if alpha > 0:
            image = image.cpu().numpy()
            for mask, y in zip(self.text, TEXT_Y):
                blit_mask(
                    image, mask, WIDTH // 2, y, self.text_colour * 255, alpha, "centre"
                )
        return image

    @staticmethod
    def text_alpha(seconds):
        fade_in, out_at, length = TEXT_TIMES
        return float(
            smoothstep(seconds / fade_in)
            * smoothstep((out_at + length - seconds) / length)
        )


# ------------------------------------------------------------------- the sound
def soundtrack(timeline, style_name, rng):
    """
    Bamboo flute over a pad. Every pass seen on screen (the two pairs level,
    found on the orbit) sounds a held flute note, climbing PASS_DEGREES; the
    pass where the orange ring goes through is panned a little left, the teal
    one right. Half a pass earlier (the pairs farthest apart, the one behind
    starting to narrow) a softer, lower MID_DEGREES note answers. A soft pluck
    bass (high-passed at 70 Hz) on root and fifth marks each pass. Under it, a pad
    walks CHORDS (one chord per two passes), its low-pass opening as the rings
    climb the screen and closing as they leave. The last pass rings out into
    the rest. One-shot: nothing wraps.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    passes = timeline.orbit.passes
    passes = passes[passes > timeline.s0]
    # Only the passes seen on screen: the vortices' centre below the top edge.
    z_mid = 0.5 * (timeline.orbit(passes)[1] + timeline.orbit(passes)[3])
    passes = passes[z_mid + timeline.z_shift < HEIGHT / UNIT_PX - 0.1]
    times = timeline.video(passes)
    for i, when in enumerate(times):
        s = timeline.sim(when)
        x1, _, x2, _ = timeline.orbit(s)
        through = 0 if x1 < x2 else 1  # the narrower pair is the one passing through
        degree = PASS_DEGREES[i % len(PASS_DEGREES)]
        last = i == len(times) - 1
        length = 2.6 if last else 1.5
        held = dict(length=length) if style.voice is audio.flute else {}
        note = style.voice(scale.hz(degree), 0.5 + 0.04 * i, **held)
        audio.add(
            track, audio.pan(note, -0.3 if through == 0 else 0.3), when, circular=False
        )
        bass = style.bass(scale.hz(0 if i % 2 == 0 else 4, octave=-2), 0.3)
        bass = audio.highpass(audio.pan(bass, 0.0), 70.0, circular=False)
        audio.add(track, bass, when, circular=False)
    # Half-way between passes the pairs are farthest apart and the one behind
    # starts to narrow: a softer, lower note answers from the other side.
    halves = timeline.video(passes - 0.5 * timeline.period)
    for i, when in enumerate(halves):
        if when < 0.5:  # before the video: open on it instead
            when = 0.05 if when > -1.0 else None
        if when is None:
            continue
        x1, _, x2, _ = timeline.orbit(timeline.sim(when))
        behind = 0 if x1 < x2 else 1  # narrower = about to pass
        held = dict(length=1.1) if style.voice is audio.flute else {}
        note = style.voice(scale.hz(MID_DEGREES[i % len(MID_DEGREES)]), 0.3, **held)
        audio.add(
            track, audio.pan(note, -0.45 if behind == 0 else 0.45), when, circular=False
        )

    # Pad: CHORDS, one per two passes, from the start to the end.
    bounds = np.concatenate(([0.0], times[1::2], [DURATION]))
    pad = audio.silence(DURATION)
    for i in range(len(bounds) - 1):
        start, stop = bounds[i], bounds[i + 1]
        for note in scale.chord(CHORDS[i % len(CHORDS)], 3, -1):
            hz = float(audio.midi_to_hz(note))
            audio.add(
                pad,
                audio.pad(hz, stop - start + 0.8, 0.22, 0.6 if i else 0.2, 1.6, 0.35),
                start - 0.3 if i else 0.0,
                circular=False,
            )
    # Brightness follows the rings up the screen.
    t = np.arange(len(pad)) / audio.SAMPLE_RATE
    height = np.interp(t, (0, EXIT_TIME, DURATION), (0, 1, 0.3))
    dark = audio.lowpass(pad, 800.0, circular=False)
    bright = audio.lowpass(pad, 2500.0, circular=False)
    track += dark * (1 - height[:, None]) + bright * height[:, None]
    logger.info(f"Soundtrack: {len(times)} passes at {np.round(times, 2).tolist()} s")
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 1.0)  # with the picture's fade
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Leapfrogging vortex rings, as a one-shot, in an axisymmetric slice: each
    ring is a pair of point vortices (+x_i, z_i, -Γ) and (-x_i, z_i, +Γ), the
    two rings coaxial, so the four-vortex problem keeps its mirror symmetry and
    reduces to two degrees of freedom, which is integrable: the motion is
    periodic in the frame moving with the pair. From level with half-widths
    WIDE = 1 and NARROW = 0.42 (Love 1894: leapfrogging for a ratio above
    (3 - √5)/2 ≈ 0.382) the narrow pair shoots through, widens and slows, and
    the roles swap. The orbit is solved once with solve_ivp (rtol 1e-10,
    dense output); sim time s(t) = s0 + rate · (t − E(1 − e^(−t/E))) eases in
    and is scaled so the vortices' centre climbs from Z_FIRST above the bottom
    edge to EXIT_MARGIN above the top at EXIT_TIME, about six passes.

    Dye: 764 k passive tracers on the GPU in the right half only (the flow is
    mirror-symmetric; each frame splats them twice, at ±x): banded off-centre
    discs of dye beside each core (wound into spirals during WARMUP sim time
    before t = 0) and horizontal smoke lines LINE_SPACING apart, from
    LINES_BELOW lines under the frame to above its top, out to
    LINE_HALF_WIDTH (the air is drawn in), advected with RK4 (step ≤ DT_MAX in sim time, the
    same for every RP_FPS) by the Biot-Savart field of the four vortices,
    regularised as Lamb-Oseen cores of radius CORE: u = Γ/(2π) ẑ×r (1 −
    e^(−r²/δ²))/r². The vortices themselves move as exact point vortices.
    This is the one sequential part (constant cost per step, ~1 ms); frames
    are rendered in order: bilinear splats into one density channel per dye,
    a small blur, Beer-like 1 − e^(−g·ρ) inks over a dark gradient, a
    quarter-size bloom, a fade over the last FADE_OUT s. The caption lines are
    LaTeX masks (latex_mask) blitted for the first ~4 s.

    Sound (secondary, from the same orbit): bamboo flute notes at every pass
    over a pad; see `soundtrack`. The posted soundtrack is `--seed 0` (audio
    seed 0, STYLE "bamboo_flute"). The seed only jitters the dye and picks the
    musical root.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    timeline = Timeline()
    device = select_device()
    with torch.no_grad():
        renderer = Renderer(timeline, rng, device)
        if not settings.audio_only:
            renderer.dye.advance(timeline.s0)  # spin up the cores' dye

        def render(frame):
            return renderer.frame(frame / (FRAME_COUNT - 1) * DURATION)

        render_frames(settings, render, FRAME_COUNT, parallel=False, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(timeline, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    del renderer
    gc.collect()


if __name__ == "__main__":
    generate()
