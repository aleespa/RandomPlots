"""
Joukowski's Wing

Take a circle, slightly off-centre, and push every point of the plane
through one small formula: ζ = z + c²/z. The circle comes out as the shape
of an aircraft wing, and the air flowing round the circle comes out as the
air flowing round the wing. Nikolai Joukowski found this in 1906, and for
decades it was how wings were designed. Raise the formula to a power n and
the sharp tail opens into a wedge: here the circle slides and the power
drifts, so the wing melts from a slender blade into a fat teardrop, then
into a deep hooked wing like a bird's.

The thin lines are streamlines, the paths the air follows. The gold grains
ride the flow at its true speed: watch them race over the top of the wing
and dawdle underneath. Faster air means lower pressure, so the wing is
sucked upwards. That is lift, L = ρUΓ, and as the wing tilts up and down
the streamlines bend and the lift breathes with it.

Listen too: each wing shape has its own chord, and the chords melt into
one another exactly as the shapes do. Under them runs the wind; the more
lift the wing makes, the brighter and fuller the sound.

Made with Python & PyTorch. Sound synthesized in NumPy.
#aerodynamics #fluiddynamics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat
from common.image_processing import ImageProcessingSettings
from common.music import Scale, beat, beat_times
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

PAPER = "#0a0f24"
INK = "#8fb8d8"
WING = "#02030a"
GOLD = "#ffb347"
SUCTION = "#1f6fff"  # low pressure wash
PRESSURE = "#ff3d7f"  # high pressure wash

C = 1.0  # map constant; the chord is about 4C
# Wing keyframes the loop passes through: circle centre μ (real part sets the
# thickness, imaginary part the camber) and the Kármán–Trefftz exponent n
# (trailing-edge angle (2 − n)π; n = 2 is Joukowski's sharp edge).
SHAPES = (
    (complex(-0.10, 0.12), 1.97),  # the classic cambered wing
    (complex(-0.30, 0.03), 1.86),  # fat teardrop, open tail
    (complex(-0.05, 0.42), 1.99),  # deep hooked wing, a bird's
)
DWELL = 2.0  # how long the loop lingers near each shape (softmax sharpness)
SPEED = 1.5  # free stream, units per second
ALPHA = (0.09, 0.05)  # angle of attack: mean and swing, radians (one cycle per loop)
VIEW = 5.0  # visible width, units
WING_Y = 0.06  # wing centre, as a fraction of the height above the middle
SUPERSAMPLE = 2

STREAM_SPACING = 0.15  # Δψ / SPEED: far-field gap between streamlines, units
LINE_PX = (1.0, 1.9)  # streamline width, thin and every HEAVY_EVERY-th line, px
HEAVY_EVERY = 5
LINE_INK = 0.30
WASH = 0.45  # strength of the pressure wash
WASH_RANGE = 1.5  # |Cp| that saturates the wash

PARTICLE_DENSITY = 30  # particles per square unit, everywhere in the flow
LIVES = 2  # each particle is reborn this many times per loop (lifetime D / LIVES)
FADE = 0.4  # seconds of fade-in after birth and fade-out before rebirth
STEPS_PER_SECOND = 240  # simulation steps (independent of FPS)
TRAIL = 0.18  # seconds of trail behind each particle
TRAIL_SAMPLES = 44  # = TRAIL * STEPS_PER_SECOND: one per step, a solid stroke
TRAIL_FILL = 3  # splats per trail sample gap (fast trails would bead otherwise)
PARTICLE_INK = 1.0

# Soundtrack: one chord per wing shape (scale degrees, same order as SHAPES),
# crossfaded with the picture's own blend weights; lift brightens the sound.
SOUND = "drone"  # "drone", "harp" or "bells"
SOUND_MODE = "lydian"
SOUND_ROOTS = range(50, 56)  # MIDI, D3..G3
SHAPE_CHORDS = (0, 5, 1)  # I (classic), vi (teardrop), II (hooked: lydian's lift)
SOUND_BPM = 96  # 24 beats per loop


def _rgb(colour, device):
    return torch.tensor(
        [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)], device=device
    )


def _attack(t):
    """Angle of attack at loop time t in [0, 1): one smooth swing per loop."""
    return ALPHA[0] + ALPHA[1] * np.sin(2 * np.pi * t)


def _blend(t):
    """Periodic softmax weights of SHAPES at loop time t in [0, 1)."""
    phase = np.cos(2 * np.pi * (t - np.arange(len(SHAPES)) / len(SHAPES)))
    blend = np.exp(DWELL * phase)
    return blend / blend.sum()


def _shape(t):
    """
    Wing at loop time t in [0, 1): circle centre μ, exponent n, radius R and
    β = −arg(C − μ), blending SHAPES with periodic softmax weights.
    """
    blend = _blend(t)
    mu = complex(sum(b * m for b, (m, _) in zip(blend, SHAPES)))
    n = float(sum(b * k for b, (_, k) in zip(blend, SHAPES)))
    return mu, n, abs(C - mu), -np.angle(C - mu)


def _circulation(alpha, shape):
    """Kutta condition: Γ = 4πUR sin(α + β). The lift is ρUΓ."""
    _, _, radius, beta = shape
    return 4 * np.pi * radius * SPEED * np.sin(alpha + beta)


def _flow(zs, alpha, gamma, shape, offset):
    """
    The flow at screen points `zs` (complex, wing frame shifted by `offset`):
    returns (inside, ψ, conjugate velocity in screen coordinates).

    The wing is pitched nose-up by `alpha`, so in its own frame the free
    stream comes in at +alpha; the screen sees a horizontal free stream.
    The Kármán–Trefftz map is inverted through u = (ζ − nC)/(ζ + nC) =
    ((z − C)/(z + C))^n: of the branches v = u^(1/n)·e^(2πik/n) with
    |arg v| < π (the principal branch the forward map uses), keep the one
    whose z lies outside the circle.
    """
    mu, n, radius, _ = shape
    tilt = complex(np.exp(1j * alpha))
    zb = (zs - offset) * tilt  # screen -> wing frame
    u = (zb - n * C) / (zb + n * C)
    log_u = torch.log(u)
    z = torch.zeros_like(zb)
    v = torch.ones_like(zb)
    best = torch.full(zb.shape, -1.0, device=zb.device)
    for k in (-1, 0, 1):
        log_v = (log_u + 2j * np.pi * k) / n
        vk = torch.exp(log_v)
        zk = C * (1 + vk) / (1 - vk)
        dk = (zk - mu).abs()
        better = (log_v.imag.abs() < np.pi) & (dk > radius) & (dk > best)
        z = torch.where(better, zk, z)
        v = torch.where(better, vk, v)
        best = torch.where(better, dk, best)
    inside = best < 0
    z = torch.where(inside, torch.full_like(z, 10.0), z)  # placeholder, masked
    rel = z - mu
    w = SPEED * (rel / tilt + radius * radius * tilt / rel) + 1j * gamma / (
        2 * np.pi
    ) * torch.log(rel)
    dw_dz = SPEED * (1 / tilt - radius * radius * tilt / (rel * rel)) + 1j * gamma / (
        2 * np.pi * rel
    )
    dzeta_dz = 4 * n * n * C * C * u / (v * (z + C) ** 2 * (1 - u) ** 2)
    conj_velocity = dw_dz / dzeta_dz * tilt  # (u - iv) in screen coordinates
    psi = w.imag - gamma * np.log(radius) / (2 * np.pi)  # 0 on the wing
    return inside, psi, conj_velocity


def _push_out(zs, alpha, shape, offset, margin=1.01):
    """
    Move screen points that lie inside the wing to just outside its surface:
    take the inverse-map branch closest to the outside, push it radially to
    `margin`·R in the circle plane and map it forward again,
    ζ = nC(1 + u)/(1 − u) with u = ((z − C)/(z + C))^n. Points already outside
    are returned unchanged.
    """
    mu, n, radius, _ = shape
    tilt = complex(np.exp(1j * alpha))
    zb = (zs - offset) * tilt
    log_u = torch.log((zb - n * C) / (zb + n * C))
    z = torch.zeros_like(zb)
    best = torch.full(zb.shape, -1.0, device=zb.device)
    for k in (-1, 0, 1):
        log_v = (log_u + 2j * np.pi * k) / n
        zk = C * (1 + torch.exp(log_v)) / (1 - torch.exp(log_v))
        dk = (zk - mu).abs()
        better = (log_v.imag.abs() < np.pi) & (dk > best)
        z = torch.where(better, zk, z)
        best = torch.where(better, dk, best)
    inside = best <= radius
    rel = z - mu
    z_out = mu + rel / rel.abs().clamp_min(1e-9) * (margin * radius)
    u = torch.exp(n * torch.log((z_out - C) / (z_out + C)))
    out = n * C * (1 + u) / (1 - u) / tilt + offset
    return torch.where(inside & torch.isfinite(out), out, zs)


def soundtrack(rng):
    """
    A loop in SOUND_MODE built from the same closed forms as the picture.

    Each wing shape owns a chord (SHAPE_CHORDS); a soft pad holds all three,
    each weighted by that shape's blend weight, so the harmony melts exactly
    as the wing does. The lift ρUΓ, normalised over the loop, crossfades the
    pads from a dark to a bright voicing and swells them. On top, by SOUND:
    "drone" adds the air itself (band-passed wind, louder and brighter with
    lift); "harp" plucks the current chord in rising eighths at SOUND_BPM,
    climbing higher the more lift there is; "bells" strikes one high bell per
    beat on the current chord's tones, also lifted by lift. All circular.
    """
    track = audio.silence(DURATION)
    n = len(track)
    t = np.arange(n) / n
    grid = np.linspace(0.0, 1.0, 601)
    weights = np.array([_blend(u) for u in grid])  # (601, shapes)
    lift = np.array([_circulation(_attack(u), _shape(u)) for u in grid])
    lift = (lift - lift.min()) / (lift.max() - lift.min())
    lift_t = np.interp(t, grid, lift)

    scale = Scale(int(rng.choice(SOUND_ROOTS)), SOUND_MODE)
    for j, degree in enumerate(SHAPE_CHORDS):
        weight = np.interp(t, grid, weights[:, j])[:, None]
        dark, bright = audio.silence(DURATION), audio.silence(DURATION)
        for note in scale.chord(degree, 3):
            f = float(audio.midi_to_hz(note))
            audio.add(dark, audio.pad(f, DURATION, 0.45, brightness=0.15), 0.0)
            audio.add(bright, audio.pad(f, DURATION, 0.45, brightness=0.8), 0.0)
        bass = float(audio.midi_to_hz(scale.note(degree, octave=-1)))
        audio.add(dark, audio.pad(bass, DURATION, 0.5, brightness=0.05), 0.0)
        mix = dark + (bright - dark) * lift_t[:, None]
        track += weight * (0.55 + 0.45 * lift_t[:, None]) * mix

    def chord_at(when):
        j = int(np.argmax(_blend((when / DURATION) % 1.0)))
        return SHAPE_CHORDS[j]

    def lift_at(when):
        return float(np.interp((when / DURATION) % 1.0, grid, lift))

    if SOUND == "drone":
        air = rng.standard_normal((n, 2))
        air = audio.highpass(audio.lowpass(air, 1400.0), 250.0)
        air /= np.abs(air).max()
        track += 0.12 * (0.2 + lift_t[:, None] ** 2) * air
    elif SOUND == "harp":
        track *= 0.6
        pattern = (0, 1, 2, 3, 4, 3, 2, 1)
        for i, when in enumerate(beat_times(DURATION, SOUND_BPM, subdivision=2)):
            up = lift_at(when)
            step = pattern[i % len(pattern)] + int(round(2 * up))
            degree = chord_at(when) + 2 * step
            note = audio.pluck(scale.hz(degree, octave=1), 0.25 + 0.3 * up)
            audio.add(track, audio.pan(note, 0.5 if i % 2 else -0.5), when)
        track = audio.delay(track, beat(SOUND_BPM) * 0.75, 0.3, 0.25)
    elif SOUND == "bells":
        track *= 0.7
        for i, when in enumerate(beat_times(DURATION, SOUND_BPM)):
            up = lift_at(when)
            step = (0, 2, 1, 3)[i % 4] + int(round(2 * up))
            degree = chord_at(when) + 2 * step
            note = audio.bell(scale.hz(degree, octave=2), 0.25 + 0.3 * up)
            audio.add(track, audio.pan(note, 0.6 * np.sin(2 * np.pi * i / 6)), when)
        track = audio.delay(track, beat(SOUND_BPM) * 1.5, 0.35, 0.3)
    else:
        raise ValueError(f"unknown SOUND {SOUND!r}")

    logger.info(f"Soundtrack '{SOUND}': root MIDI {scale.root} {SOUND_MODE}")
    track = audio.reverb(track, rng, rt60=3.0, wet=0.35, tone=4500.0)
    return audio.master(track)


def generate(settings: ImageProcessingSettings = None):
    """
    Potential flow past a morphing Kármán–Trefftz airfoil, through a loop.

    In the circle plane (centre μ, radius R = |C − μ| so the circle passes
    through z = C) the flow at angle of attack α with circulation Γ is
    w(z) = U[(z − μ)e^{−iα} + R²e^{iα}/(z − μ)] + iΓ/2π · log(z − μ).
    The Kutta condition (no flow round the trailing edge, z = C) fixes
    Γ = 4πUR sin(α + β) with β = −arg(C − μ). The Kármán–Trefftz map
    ζ = nC[(z + C)^n + (z − C)^n] / [(z + C)^n − (z − C)^n] sends the circle
    to an airfoil with trailing-edge angle (2 − n)π (n = 2 is Joukowski's
    ζ = z + C²/z); see `_flow` for the inversion, and
    dζ/dz = 4n²C²u / (v (z + C)² (1 − u)²) with v = (z − C)/(z + C), u = vⁿ.
    μ and n blend between SHAPES with periodic softmax weights
    exp(DWELL·cos 2π(t − j/K)), so the wing melts from shape to shape and
    the loop is exact. The wing is drawn pitched by α in screen coordinates,
    so the far field stays horizontal while α = ALPHA[0] + ALPHA[1] sin 2πt.

    Streamlines are level sets of ψ = Im w (offset so ψ = 0 on the wing),
    drawn as analytically anti-aliased lines of constant pixel width using
    |∇ψ| = |dw/dζ|; every HEAVY_EVERY-th is heavier. A faint wash shows the
    pressure coefficient Cp = 1 − |V|²/U² (blue suction, magenta pressure).
    Fields are evaluated per pixel on the GPU at SUPERSAMPLE×.

    Gold particles are advected through the same velocity field (RK2 at
    STEPS_PER_SECOND). Each lives D / LIVES and is reborn at a fixed random
    point, uniform over the screen plus the stretch upstream that free-stream
    air crosses in one lifetime, so the density is uniform even in the slow
    pocket under a deep wing (left-edge injection alone never reaches it
    within a loop). Births fade in and deaths fade out over FADE seconds.
    Birth times and points repeat every loop, and one loop is simulated as
    warm-up before frame 0, so the particle field is exactly periodic. Particles the morphing wing
    sweeps over are pushed back onto its surface (`_push_out`). Each particle
    leaves a trail of its last TRAIL seconds, splatted with fading weight and
    TRAIL_FILL interpolated splats per step so fast trails stay solid.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    scale = WIDTH / VIEW  # px per unit at output size
    offset = complex(0.0, WING_Y * HEIGHT / scale)
    half_w, half_h = 0.5 * WIDTH / scale, 0.5 * HEIGHT / scale

    circulation = _circulation

    logger.info(
        f"{len(SHAPES)} wing shapes, α "
        f"{np.degrees(ALPHA[0] - ALPHA[1]):.1f}–{np.degrees(ALPHA[0] + ALPHA[1]):.1f}°"
    )

    # Pixel grid (supersampled), in units, y up.
    ss = SUPERSAMPLE
    xs = (torch.arange(WIDTH * ss, device=device) + 0.5) / (scale * ss) - half_w
    ys = half_h - (torch.arange(HEIGHT * ss, device=device) + 0.5) / (scale * ss)
    grid = torch.complex(
        xs[None, :].expand(HEIGHT * ss, -1), ys[:, None].expand(-1, WIDTH * ss)
    )

    paper, ink, wing, gold = (_rgb(c, device) for c in (PAPER, INK, WING, GOLD))
    suction, pressure = _rgb(SUCTION, device), _rgb(PRESSURE, device)
    trail_kernel = gaussian_kernel(0.7, device)

    # Particles: birth times and points (one per life), repeated every loop.
    life = DURATION / LIVES
    x0, x1 = -half_w - 1.1 * SPEED * life, half_w
    y0, y1 = -half_h * 1.05, half_h * 1.05
    count = int(PARTICLE_DENSITY * (x1 - x0) * (y1 - y0))
    born = torch.tensor(rng.uniform(0, life, count), device=device)
    seeds = torch.complex(
        torch.tensor(rng.uniform(x0, x1, (LIVES, count)), device=device),
        torch.tensor(rng.uniform(y0, y1, (LIVES, count)), device=device),
    ).to(torch.complex64)
    seed = seeds[0]
    position = seed.clone()
    alive = torch.zeros(count, dtype=torch.bool, device=device)
    substeps = max(1, round(STEPS_PER_SECOND / FPS))
    dt = DURATION / (FRAME_COUNT * substeps)
    keep = max(TRAIL_SAMPLES, int(round(TRAIL / dt)))
    history = seed[None].repeat(keep, 1)
    head = 0

    def velocity(z, t):
        loop_t = (t / DURATION) % 1.0
        alpha, shape = _attack(loop_t), _shape(loop_t)
        inside, _, conj = _flow(z, alpha, circulation(alpha, shape), shape, offset)
        v = torch.conj(conj)
        v = torch.where(inside | ~torch.isfinite(v), torch.zeros_like(v), v)
        return v, inside

    def step(t):
        nonlocal position, history, head, alive
        v, _ = velocity(position, t)
        mid = position + 0.5 * dt * v
        v, _ = velocity(mid, t + 0.5 * dt)
        position = position + dt * v
        # The morphing wing sweeps over some particles: put them back on its
        # surface so the flow under a deepening wing doesn't empty out.
        loop_t = ((t + dt) / DURATION) % 1.0
        alpha = _attack(loop_t)
        position = _push_out(position, alpha, _shape(loop_t), offset).to(
            torch.complex64
        )
        # Rebirth: each particle starts a new life at its own times each loop.
        reborn = torch.remainder(t + dt - born, life) < dt
        which = torch.floor(torch.remainder(t + dt - born, DURATION) / life).long()
        seed = seeds.gather(0, which.clamp(0, LIVES - 1)[None])[0]
        position = torch.where(reborn, seed, position)
        alive = (alive | reborn) & (position.real < half_w + 0.3)
        head = (head + 1) % keep
        history[head] = position
        history[:, reborn] = seed[reborn]

    # Warm-up: one full loop before frame 0.
    t = -DURATION
    for _ in range(FRAME_COUNT * substeps):
        step(t)
        t += dt
    logger.info(f"Warm-up done: {int(alive.sum())} of {count} particles in flight")

    trail_index = (
        torch.linspace(0, keep - 1, TRAIL_SAMPLES, device=device).round().long()
    )
    trail_weight = torch.linspace(1.0, 0.15, TRAIL_SAMPLES, device=device)

    def render(frame):
        nonlocal t
        if frame > 0:
            for _ in range(substeps):
                step(t)
                t += dt
        loop_t = frame / FRAME_COUNT
        alpha, shape = _attack(loop_t), _shape(loop_t)
        gamma = circulation(alpha, shape)
        inside, psi, conj = _flow(grid, alpha, gamma, shape, offset)
        speed = conj.abs()

        # Streamlines, anti-aliased to a constant pixel width.
        spacing = STREAM_SPACING * SPEED
        q = psi / spacing
        nearest = torch.round(q)
        distance = (q - nearest).abs() * spacing / speed.clamp_min(1e-4) * (scale * ss)
        heavy = torch.remainder(nearest, HEAVY_EVERY) == 0
        width = torch.where(heavy, LINE_PX[1], LINE_PX[0]) * ss
        lines = (0.5 * width + 0.5 * ss - distance).clamp(0.0, ss) / ss
        lines = torch.where(inside, torch.zeros_like(lines), lines)

        cp = (1.0 - (speed / SPEED) ** 2).clamp(-WASH_RANGE, WASH_RANGE) / WASH_RANGE
        cp = torch.where(inside, torch.zeros_like(cp), cp)

        # Down to output size (box filter = anti-aliasing).
        fields = torch.stack([lines, cp, inside.float()])[None]
        lines, cp, body = torch.nn.functional.avg_pool2d(fields, ss)[0]

        image = paper.expand(HEIGHT, WIDTH, 3).clone()
        image = image + WASH * (
            cp.clamp_min(0)[..., None] * (pressure - paper)
            + (-cp).clamp_min(0)[..., None] * (suction - paper)
        )
        image = image + (LINE_INK * lines)[..., None] * (ink - image)

        # Particles and their trails.
        order = (head - trail_index) % keep
        points = history[order]  # (TRAIL_SAMPLES, count)
        # Fill the gaps between samples so fast trails are strokes, not beads.
        fill = torch.arange(TRAIL_FILL, device=device)[:, None, None] / TRAIL_FILL
        points = (points[:-1] + fill * (points[1:] - points[:-1])).reshape(
            -1, points.shape[1]
        )
        tw = trail_weight[:-1].repeat(TRAIL_FILL) / TRAIL_FILL
        age = torch.remainder(t - born, life)
        fade = ((age / FADE).clamp(0, 1) * ((life - age) / FADE).clamp(0, 1)) ** 2
        on = (alive * fade).float()[None].expand_as(points)
        px = (points.real + half_w) * scale
        py = (half_h - points.imag) * scale
        weights = (tw[:, None] * on).reshape(-1, 1)
        density = torch.zeros(HEIGHT, WIDTH, 1, device=device)
        splat(density, px.reshape(-1), py.reshape(-1), weights)
        density = blur(density, trail_kernel)
        cover = PARTICLE_INK * (1.0 - torch.exp(-4.0 * density))
        image = image + cover * (gold - image)

        image = image + body[..., None] * (wing - image)
        return 255.0 * image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    track = soundtrack(settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
