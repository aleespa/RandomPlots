"""
The Hidden Walls Inside a Flow

Water turning in two whirlpools, one above the other, while the line between them sways.
The glowing curves are hidden walls inside the flow, and the water treats them as real.

A drop of ink that lands on a violet line is torn apart: those walls repel, and neighbouring
specks of water race away from them. Gold lines attract: the water is squeezed onto them, so
every drop is drawn out along a gold curve. Both are found by asking, at each point, how fast
two specks side by side would separate in the flow's future (violet) or its past (gold): the
finite-time Lyapunov exponent. Oceanographers call these Lagrangian coherent structures and use
them to predict where oil spills and lost swimmers will drift.

Listen: each new drop of ink strikes a soft piano note, higher for drops higher up, and a
quieter note answers as the drop begins to tear. Underneath, a hushed choir changes chord
with every swing of the flow.

Made with Python & PyTorch. Sound synthesized in NumPy.
#lagrangiancoherentstructures #fluiddynamics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import PROGRESSIONS, STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'This water has invisible walls'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# The flow: a time-periodic double gyre, stood upright to fill the frame.
AMPLITUDE = 0.1  # stream-function amplitude A
WOBBLE = 0.25  # epsilon: how far the dividing line swings
PERIOD = 10.0  # flow time of one swing
PERIODS_PER_LOOP = 2  # whole swings per video loop
LENGTH = (
    HEIGHT / WIDTH
)  # the box is 1 wide and 16/9 tall (two gyres, one above the other)
WINDOW = 25.0  # flow time of the FTLE integration, forward and backward
DT = 0.1  # RK4 step (flow time)
PHASE0 = 0.0  # flow time at the start of the loop

# Dust: faint particles everywhere, born on a periodic schedule, each living
# DUST_LIFE of the loop. Drops: small discs of bright ink, DROPS per loop.
DUST = 8_000
DUST_LIFE = 0.5
DUST_WEIGHT = 1.2
DROPS = 14
DROP_POINTS = 64_000
DROP_RADIUS = 0.025
DROP_LIFE = 0.7
DROP_WEIGHT = 0.035
WALL_CLEARANCE = 0.06  # drops are placed so their ink keeps off the walls
RIM_PX = 14.0  # lines fade out within this many pixels of the walls
CORE = 0.8  # white-hot cores on the strongest ridges

# Colours (RGB, 0..1)
TOP_WATER = "#0d0a1e"  # water that came from the upper gyre
LOW_WATER = "#050b16"  # ... from the lower gyre
REPEL = "#6d5cff"  # forward FTLE ridges: repelling walls
ATTRACT = "#ffb43c"  # backward FTLE ridges: attracting walls
DUST_COLOUR = "#fff1d6"
DROP_COLOUR = "#d9f4ff"

# Sound
STYLE = "piano_choir"  # a SOUNDS tag; an --audio-only tag naming another wins
SOUNDS = {  # tag -> (lead style, mode override or None, bed voice)
    "felt_piano": ("felt_piano", "dorian", "pad"),
    "piano_choir": ("felt_piano", "dorian", "choir"),
    "bamboo_flute": ("bamboo_flute", None, "pad"),
}
CHORDS = PROGRESSIONS["drift"]  # one chord per half swing of the flow

KERNEL = """
template <typename T> void advect(T x0, T y0, T t0, T duration, T& x1, T& y1) {{
    auto velocity = [](float x, float y, float t, float& u, float& v) {{
        const float pi = 3.14159265358979f;
        float e = {wobble}f * __sinf({omega}f * fmodf(t, {period}f));
        float s = x * {two_over_l}f;
        float f = e * s * s + (1.0f - 2.0f * e) * s;
        float fs = 2.0f * e * s + 1.0f - 2.0f * e;
        float sf, cf, sy, cy;
        __sincosf(pi * f, &sf, &cf);
        __sincosf(pi * y, &sy, &cy);
        u = -pi * {amp}f * sf * cy;
        v = pi * {amp}f * cf * sy * fs * {two_over_l}f;
    }};
    float x = x0, y = y0, t = t0;
    int n = (int)ceilf(fabsf(duration) / {dt}f);
    float h = n > 0 ? duration / n : 0.0f;
    for (int i = 0; i < n; ++i) {{
        float u1, v1, u2, v2, u3, v3, u4, v4;
        velocity(x, y, t, u1, v1);
        velocity(x + 0.5f * h * u1, y + 0.5f * h * v1, t + 0.5f * h, u2, v2);
        velocity(x + 0.5f * h * u2, y + 0.5f * h * v2, t + 0.5f * h, u3, v3);
        velocity(x + h * u3, y + h * v3, t + h, u4, v4);
        x += h / 6.0f * (u1 + 2.0f * u2 + 2.0f * u3 + u4);
        y += h / 6.0f * (v1 + 2.0f * v2 + 2.0f * v3 + v4);
        t += h;
    }}
    x1 = x;
    y1 = y;
}}
"""


def _rgb(colour):
    return [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]


class Flow:
    """The double gyre's flow map on the GPU, as one fused CUDA kernel (jiterator)."""

    def __init__(self, device):
        self.device = device
        code = KERNEL.format(
            wobble=WOBBLE,
            omega=2 * np.pi / PERIOD,
            two_over_l=2.0 / LENGTH,
            amp=AMPLITUDE,
            dt=DT,
            period=PERIOD,
        )
        if device.type == "cuda":
            from torch.cuda.jiterator import _create_multi_output_jit_fn

            self._kernel = _create_multi_output_jit_fn(code, num_outputs=2)
        else:
            self._kernel = None

    def velocity(self, x, y, t):
        e = WOBBLE * torch.sin(2 * np.pi / PERIOD * t)
        s = x * (2.0 / LENGTH)
        f = e * s * s + (1 - 2 * e) * s
        fs = 2 * e * s + 1 - 2 * e
        u = -np.pi * AMPLITUDE * torch.sin(np.pi * f) * torch.cos(np.pi * y)
        v = np.pi * AMPLITUDE * torch.cos(np.pi * f) * torch.sin(np.pi * y) * fs
        return u, v * (2.0 / LENGTH)

    def advect(self, x, y, t0, duration):
        """Positions after `duration` (tensor or float, signed) from flow time t0."""
        t0 = torch.as_tensor(t0, dtype=torch.float32, device=self.device)
        duration = torch.as_tensor(duration, dtype=torch.float32, device=self.device)
        t0, duration = torch.broadcast_tensors(t0, duration, x)[:2]
        if self._kernel is not None:
            return self._kernel(x, y, t0.contiguous(), duration.contiguous())
        # CPU fallback: fixed-step RK4 with the largest step count (slow).
        n = int(np.ceil(float(duration.abs().max()) / DT))
        h = duration / max(n, 1)
        t = t0.clone()
        for _ in range(n):
            u1, v1 = self.velocity(x, y, t)
            u2, v2 = self.velocity(x + 0.5 * h * u1, y + 0.5 * h * v1, t + 0.5 * h)
            u3, v3 = self.velocity(x + 0.5 * h * u2, y + 0.5 * h * v2, t + 0.5 * h)
            u4, v4 = self.velocity(x + h * u3, y + h * v3, t + h)
            x = x + h / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            y = y + h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
            t = t + h
        return x, y


LOOP_TIME = PERIODS_PER_LOOP * PERIOD  # flow time of one video loop


def flow_time(t):
    """Video time t in [0, 1) -> flow time; whole swings per loop."""
    return PHASE0 + PERIODS_PER_LOOP * PERIOD * t


def ftle(flow, gx, gy, t0, window):
    """Finite-time Lyapunov exponent over [t0, t0 + window] on the pixel grid."""
    x1, y1 = flow.advect(gx, gy, t0, window)
    h = 1.0 / WIDTH
    # central differences (one-sided at the edges via torch.gradient)
    dxdi, dxdj = torch.gradient(x1, spacing=h)
    dydi, dydj = torch.gradient(y1, spacing=h)
    # rows are x (vertical), columns are y (horizontal)
    a = dxdi * dxdi + dydi * dydi
    b = dxdi * dxdj + dydi * dydj
    c = dxdj * dxdj + dydj * dydj
    tr = a + c
    det = a * c - b * b
    lam = 0.5 * (tr + torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0)))
    return torch.log(torch.clamp(lam, min=1.0)) / (2 * abs(window)), x1, y1


@torch.no_grad()
def _drop_centre(flow, when, rng):
    """
    Where a drop born at loop time `when` is placed: among the points that its
    life will stretch hardest (forward FTLE over DROP_LIFE in the top 10 %),
    the ones whose ink stays clear of the walls (a ring of 24 marker points
    advected to half, three quarters and all of the life), then one at random.
    """
    step = 8
    rows = torch.arange(0.15, LENGTH - 0.15, step / WIDTH, device=flow.device)
    cols = torch.arange(0.15, 0.85, step / WIDTH, device=flow.device)
    gx, gy = (g.flatten() for g in torch.meshgrid(rows, cols, indexing="ij"))
    t0 = flow_time(when)
    life = DROP_LIFE * LOOP_TIME
    sigma, _, _ = ftle(flow, gx.view(len(rows), -1), gy.view(len(rows), -1), t0, life)
    sigma = sigma.flatten()
    keep = sigma >= torch.quantile(sigma, 0.9)
    cx, cy, sigma = gx[keep], gy[keep], sigma[keep]
    angle = torch.linspace(0, 2 * np.pi, 25, device=flow.device)[:-1]
    mx = (cx[:, None] + DROP_RADIUS * torch.cos(angle)).flatten()
    my = (cy[:, None] + DROP_RADIUS * torch.sin(angle)).flatten()
    near = torch.zeros_like(cx)
    for fraction in (0.5, 0.75, 1.0):
        px, py = flow.advect(mx, my, t0, fraction * life)
        wall = torch.minimum(torch.minimum(px, LENGTH - px), torch.minimum(py, 1 - py))
        near += (wall < WALL_CLEARANCE).float().view(len(cx), -1).mean(1)
    good = torch.nonzero(near <= torch.quantile(near, 0.1)).flatten().cpu()
    i = int(good[rng.integers(len(good))])
    return float(cx[i]), float(cy[i])


def soundtrack(drops, tag, rng):
    """
    Ambient, and a loop like the picture (everything circular). The bed is a
    slow chord that changes every half swing of the flow (CHORDS, four per
    loop), a pad or a soft "u" choir, low-passed darker when the dividing line
    is still and brighter as it swings hardest (|sin ωt|). Each drop of ink
    sounds one lead note when it appears, higher for drops higher on screen and
    panned to its side; a softer note two steps up answers when its ink starts
    to be torn along the walls (40 % into its life). A quiet piano root under
    each chord; the bed is high-passed at 90 Hz.
    """
    lead_style, mode, bed = SOUNDS[tag]
    style = STYLES[lead_style]
    scale = Scale(int(rng.choice(style.roots)), mode or style.mode)
    track = audio.silence(DURATION)
    span = DURATION * DROP_LIFE
    for when, cx, cy in drops:
        degree = int(round(6 * (1 - cx / LENGTH))) - 1
        side = 0.7 * (2 * cy - 1)
        for delay, step, velocity in ((0.0, 0, 0.5), (0.4 * span, 2, 0.28)):
            held = dict(length=2.2) if style.voice is audio.flute else {}
            note = style.voice(scale.hz(degree + step), velocity, **held)
            audio.add(track, audio.pan(note, side), when * DURATION + delay)
    track = audio.delay(track, 60.0 / 64 * 1.5, feedback=0.3, wet=0.18)

    swings = 2 * PERIODS_PER_LOOP  # half swings per loop
    length = DURATION / swings
    bed_track = audio.silence(DURATION)
    for i in range(swings):
        chord = scale.chord(CHORDS[i % len(CHORDS)], 3, -1)
        for note in chord:
            hz = float(audio.midi_to_hz(note))
            if bed == "choir":
                sound = audio.choir(hz, length + 0.6, 0.3, vowel="u")
            else:
                sound = audio.pad(hz, length + 0.6, 0.24, 1.2, 1.8, 0.35)
            audio.add(bed_track, sound, i * length - 0.3)
        root = float(audio.midi_to_hz(chord[0]))
        bass = style.bass(root, 0.3)
        audio.add(bed_track, audio.highpass(audio.pan(bass, 0.0), 70.0), i * length)
    t = np.arange(len(bed_track)) / audio.SAMPLE_RATE
    swing = np.abs(np.sin(2 * np.pi * PERIODS_PER_LOOP * t / DURATION))[:, None]
    bed_track = audio.highpass(bed_track, 90.0)  # keep the low end clear
    dark = audio.lowpass(bed_track, 700.0)
    bright = audio.lowpass(bed_track, 2200.0)
    track += dark * (1 - swing) + bright * swing
    logger.info(f"Soundtrack {tag}: root {scale.root}, {len(drops)} drops")
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    Lagrangian coherent structures of the time-periodic double gyre (Shadden,
    Lekien & Marsden 2005), as an exact loop rendered frame by frame on the GPU.

    Flow: ψ = A sin(π f(s, t)) sin(π y), f = e s² + (1 − 2e) s, e = ε sin(ωt),
    with s = 2x/L so the two gyres fill a 1 × 16/9 box stood upright (x down
    the screen, y across); u = (−∂ψ/∂y, ∂ψ/∂x). A = 0.1, ε = 0.25, ω = 2π/10,
    and one video loop is PERIODS_PER_LOOP = 2 periods of flow time.

    Every frame is a pure function of t. The flow map over [t, t ± WINDOW]
    (WINDOW = 25) of every pixel is integrated with fixed-step RK4 (DT = 0.1)
    in one fused CUDA kernel (torch jiterator, fast-math sincos); its gradient
    by central differences gives the Cauchy–Green tensor and the FTLE
    σ = ln λ_max / (2 |T|). Forward ridges (repelling LCS) are violet, backward
    ridges (attracting LCS) gold, both scaled by fixed quantiles of a few sample
    frames, with white cores on the strongest, faded at the rim (the walls
    shear everything). The background tints each pixel by the gyre its water
    came from WINDOW ago. FTLE is periodic in t, so the loop closes exactly.

    Ink: DROPS discs of DROP_POINTS particles, born at staggered loop times and
    living DROP_LIFE of the loop, each placed (from settings.rng, with a
    forward-FTLE search, see `_drop_centre`) where its life will stretch it
    hard but keep it off the walls; plus faint dust everywhere. Each particle's
    position is the flow map from its birth to now, evaluated in the same
    kernel, so the dust too is a pure function of t; ages wrap modulo the loop.
    Splatted bilinearly, then a light blur, a quarter-size bloom and an
    exponential tone curve.

    Sound (secondary, from the same drops and the flow's swing): see
    `soundtrack`. Rendered and posted with the default --seed 0 (audio seed
    0, STYLE "piano_choir": felt-piano accents in C# dorian over a "u" choir).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    # Dust and drops: seeds, birth phases, lives and weights, drawn once.
    seed_x = [rng.uniform(0, LENGTH, DUST)]
    seed_y = [rng.uniform(0, 1, DUST)]
    birth = [rng.uniform(0, 1, DUST)]
    life = [np.full(DUST, DUST_LIFE)]
    weight = [np.full(DUST, DUST_WEIGHT)]
    is_drop = [np.zeros(DUST)]
    flow = Flow(device)
    drops = []  # (loop time of birth, centre x, centre y), for the soundtrack
    for k in range(DROPS):
        when = (k + 0.3 * rng.uniform()) / DROPS
        cx, cy = _drop_centre(flow, when, rng)
        drops.append((when, cx, cy))
        r = DROP_RADIUS * np.sqrt(rng.uniform(0, 1, DROP_POINTS))
        a = rng.uniform(0, 2 * np.pi, DROP_POINTS)
        seed_x.append(cx + r * np.cos(a))
        seed_y.append(cy + r * np.sin(a))
        birth.append(np.full(DROP_POINTS, when))
        life.append(np.full(DROP_POINTS, DROP_LIFE))
        weight.append(np.full(DROP_POINTS, DROP_WEIGHT))
        is_drop.append(np.ones(DROP_POINTS))
    seed_x, seed_y, birth, life, weight, is_drop = (
        np.concatenate(v) for v in (seed_x, seed_y, birth, life, weight, is_drop)
    )

    with torch.no_grad():
        rows = (torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5) / WIDTH
        cols = (torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5) / WIDTH
        gx, gy = torch.meshgrid(rows, cols, indexing="ij")
        gx, gy = gx.contiguous(), gy.contiguous()
        dust_x = torch.tensor(seed_x, dtype=torch.float32, device=device)
        dust_y = torch.tensor(seed_y, dtype=torch.float32, device=device)
        dust_birth = torch.tensor(birth, dtype=torch.float32, device=device)
        dust_life = torch.tensor(life, dtype=torch.float32, device=device)
        dust_weight = torch.tensor(weight, dtype=torch.float32, device=device)
        dust_is_drop = torch.tensor(is_drop, dtype=torch.float32, device=device)

        # Brightness scale from a few sample times, fixed for the whole loop.
        samples = []
        for t in (0.0, 0.25, 0.5, 0.75):
            for sign in (1, -1):
                sigma, _, _ = ftle(flow, gx, gy, flow_time(t), sign * WINDOW)
                samples.append(sigma[::4, ::4].flatten())
        samples = torch.cat(samples)
        lo = float(torch.quantile(samples, 0.78))
        hi = float(torch.quantile(samples, 0.995))
        logger.info(f"FTLE scale {lo:.3f} .. {hi:.3f}")

        top = torch.tensor(_rgb(TOP_WATER), device=device)
        low = torch.tensor(_rgb(LOW_WATER), device=device)
        repel = torch.tensor(_rgb(REPEL), device=device)
        attract = torch.tensor(_rgb(ATTRACT), device=device)
        dust_colour = torch.tensor(_rgb(DUST_COLOUR), device=device)
        drop_colour = torch.tensor(_rgb(DROP_COLOUR), device=device)
        soft = gaussian_kernel(1.0, device)
        white = torch.ones(3, device=device)
        # the walls of the box shear everything: fade the lines out at the rim
        edge = torch.minimum(torch.minimum(gx, LENGTH - gx), torch.minimum(gy, 1 - gy))
        rim = torch.clamp(edge * WIDTH / RIM_PX, 0, 1)[..., None] ** 2
        canvas = torch.zeros(HEIGHT, WIDTH, 3, device=device)

        def ridge(sigma):
            return torch.clamp((sigma - lo) / (hi - lo), 0, 1) ** 2.0

        def render(frame):
            t = frame / FRAME_COUNT
            t0 = flow_time(t)
            fwd, _, _ = ftle(flow, gx, gy, t0, WINDOW)
            bwd, bx, _ = ftle(flow, gx, gy, t0, -WINDOW)
            # where the water came from: upper or lower gyre, WINDOW ago
            side = torch.sigmoid((bx - 0.5 * LENGTH) * 400.0)[..., None]
            water = top * (1 - side) + low * side
            rf, rb = ridge(fwd)[..., None], ridge(bwd)[..., None]
            lines = repel * rf + attract * rb + CORE * white * (rf**4 + rb**4)
            lines = blur(lines, soft) * rim
            # dust and drops, carried by the flow since they were born
            age = torch.remainder(t - dust_birth, 1.0)
            alive = age < dust_life
            a = age[alive]
            lf = dust_life[alive]
            px, py = flow.advect(
                dust_x[alive],
                dust_y[alive],
                flow_time(dust_birth[alive]),
                a * PERIODS_PER_LOOP * PERIOD,
            )
            u = a / lf
            envelope = (
                torch.clamp(u / 0.04, 0, 1) * torch.clamp((1 - u) / 0.3, 0, 1) ** 2
            )
            d = dust_is_drop[alive][:, None]
            colour = drop_colour * d + dust_colour * (1 - d)
            canvas.zero_()
            splat(
                canvas,
                py * WIDTH - 0.5,
                px * WIDTH - 0.5,
                (envelope * dust_weight[alive])[:, None] * colour,
            )
            light = lines + canvas
            image = water + light + 0.25 * wide_blur(light, 12.0, 4)
            image = 1 - torch.exp(-1.6 * image)
            return image * 255.0

        render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    tag = settings.audio_tag if settings.audio_tag in SOUNDS else STYLE
    track = soundtrack(drops, tag, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
