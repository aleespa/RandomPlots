"""
Break a Wave, Get a Spiral

Spiral waves are born in a chemical dish, like the Belousov–Zhabotinsky
reaction: a pacemaker at the centre beats, and rings of reaction spread
out like ripples. Then a glass rod is dragged through them, twice.

Every broken ring gets two loose ends. Each end keeps advancing into fresh
medium but has nothing to pull it straight, so it curls into a spiral wave
that never stops turning. Spirals turn faster than the pacemaker beats, and
in an excitable medium the fastest source wins: wave by wave, the spirals
take over the dish. The same spiral waves run through heart muscle.

∂u/∂t = ∇²u + u(1 − u)(u − (v + b)/a)/ε,  ∂v/∂t = u − v

Listen: a low note marks each beat of the pacemaker, and seven hidden
listening points strike a marimba bar whenever a wave passes them, higher
near the top. The pulse quickens as the spirals take over.

Made with Python & PyTorch. Sound synthesized in NumPy.
#spiralwaves #patternformation #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

# Barkley's excitable medium, on a 720 x 1280 grid of spacing H_GRID.
GRID_W, GRID_H = 720, 1280
H_GRID = 0.24
A, B, EPS = 0.75, 0.02, 0.02
DT = 0.012  # sim time step

# Sim time against video time (the sim starts at 0; frame 0 steps it to TAU0, so
# the first ring is already out): TAU0 at t = 0, RATE sim units per second, then
# easing to a stop between SLOW_START and SLOW_END and resting there.
TAU0 = 5.0
RATE = 4.4
SLOW_START, SLOW_END = 12.6, 14.0
PACE = 6.0  # pacemaker period, sim units
PACE_R = 2.5  # pacemaker radius, sim units
CUT_TAUS = (23.0, 44.0)  # the rod enters: first downwards, then across the top
CUT_SPEED = 120.0  # sim units of path per sim unit of time
ROD_R = 6.0  # sim units
ROD_WAKE = 0.9  # v left behind the rod: a refractory scar the ends curl around

PALETTE = ["#0a0507", "#2c0a1c", "#7a1035", "#d8344a", "#ff8a5b", "#ffd9a8"]
FRONT = (1.0, 0.95, 0.86)  # RGB of the excited front
ROD = (0.75, 0.9, 1.0)  # the glass rod, cool against the embers

# Listening points (fractions of the screen) and the soundtrack.
PROBES = [
    (0.30, 0.10),
    (0.72, 0.22),
    (0.25, 0.36),
    (0.62, 0.50),
    (0.28, 0.64),
    (0.74, 0.78),
    (0.36, 0.91),
]
STYLE = (
    "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
NOTE_GAP = 0.07  # s: at most one note per gap


def sim_time(t):
    """Sim time at video time t: RATE per second, easing to a stop."""
    span = SLOW_END - SLOW_START
    x = np.clip((t - SLOW_START) / span, 0.0, 1.0)
    eased = np.where(t < SLOW_START, t, SLOW_START + span * (x - x**3 + x**4 / 2))
    return TAU0 + RATE * eased


def rod_paths(rng):
    """
    The rod's two strokes in sim units, each (points, arc length): from just
    below the pacemaker down to the bottom edge, then in from the left edge
    across the top third.
    """
    w, h = GRID_W * H_GRID, GRID_H * H_GRID
    down = (
        np.array([w / 2 + rng.uniform(-6, 6), h / 2 + rng.uniform(16, 22)]),
        np.array([w / 2 + rng.uniform(-40, 40), h + 10]),
        np.array([rng.uniform(-25, 25), 0.0]),
    )
    across = (
        np.array([-10, h * rng.uniform(0.12, 0.17)]),
        np.array([w * rng.uniform(0.7, 0.8), h * rng.uniform(0.26, 0.31)]),
        np.array([0.0, rng.uniform(-20, 20)]),
    )
    return [stroke(*ends) for ends in (down, across)]


def stroke(start, end, bend):
    """A bent stroke from start to end, sagging by the vector bend in the middle."""
    s = np.linspace(0, 1, 400)[:, None]
    points = start + (end - start) * s + np.sin(np.pi * s) * bend
    length = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    return points, length


def soundtrack(events, style_name, rng):
    """
    Marimba (or the style's voice) on every wave arrival at a listening point:
    pitch by the probe's height (degree 8 at the top down to -2 at the bottom),
    pan by its x, softer before the cut. At most one note per NOTE_GAP. Until
    the first cut, each pacemaker pulse strikes the low home note. The rod
    sounds as a low bass note and a brushed noise swish; a pad holds the home
    chord, and a struck chord lands as the dish freezes.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    grid = np.linspace(0, DURATION, 4001)
    t_cuts = np.interp(CUT_TAUS, sim_time(grid), grid)
    paces = np.arange(np.ceil(TAU0 / PACE), CUT_TAUS[0] / PACE) * PACE
    for when in np.interp(paces, sim_time(grid), grid):  # the pacemaker's pulse
        audio.add(track, style.voice(scale.hz(0, -1), 0.45), when, **kw)
    last = -1.0
    for when, probe in sorted(events):
        if when - last < NOTE_GAP or when > SLOW_END:
            continue
        last = when
        x, y = PROBES[probe]
        degree = int(round(8 - 10 * y))
        velocity = (0.45 if when < t_cuts[0] else 0.5) * (0.85 + 0.3 * rng.random())
        note = audio.pan(style.voice(scale.hz(degree), velocity), 1.4 * (x - 0.5))
        audio.add(track, note, when, **kw)

    for t_cut, degree in zip(t_cuts, (0, 3)):
        audio.add(track, style.bass(scale.hz(degree, -1), 0.4), t_cut, **kw)
        swish = audio.lowpass(
            audio.tick(rng, 0.3, 0.12, 2500.0)[:, None].repeat(2, 1),
            4000.0,
            circular=False,
        )
        audio.add(track, swish, t_cut, **kw)
    for degree in (0, 2, 4):
        voice = audio.pad(scale.hz(degree, -1), SLOW_END - 0.5, 0.12, 2.0, 2.5, 0.3)
        audio.add(track, voice, 0.0, **kw)
    for degree, lag in ((0, 0.0), (2, 0.05), (4, 0.1), (7, 0.15)):
        audio.add(track, style.voice(scale.hz(degree), 0.5), SLOW_END + lag, **kw)
    audio.add(track, style.bass(scale.hz(0, -1), 0.45), SLOW_END, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Barkley's model of an excitable medium (a two-variable caricature of the
    Belousov-Zhabotinsky reaction and of cardiac tissue):

        u_t = lap(u) + u (1 - u) (u - (v + b) / a) / eps,   v_t = u - v,

    a = 0.75, b = 0.02, eps = 0.02, on a 720 x 1280 grid (spacing H_GRID, no-flux
    edges), explicit Euler with dt = 0.012 on the GPU. u is the fast excitation
    (the bright front), v the slow recovery (the glowing tail).

    A pacemaker at the centre is re-excited every PACE sim units, sending out
    target rings. At CUT_TAUS[0] a rod (a disc of radius ROD_R that sets u = 0
    and v >= ROD_WAKE, a refractory scar) sweeps along a bent path from just
    below the pacemaker to the bottom edge at CUT_SPEED, breaking the lower
    half of every ring; each pair of free ends curls into a counter-rotating
    pair of spirals. At CUT_TAUS[1] it strokes in from the left edge across
    the top third and breaks the outgoing waves there too. Spirals rotate faster than PACE, so their waves overrun the
    pacemaker's and fill the dish.

    Video time t maps to sim time sim_time(t) (RATE per second, eased to a stop
    over SLOW_START..SLOW_END, then a rest and a FADE). The simulation is
    sequential but cheap: each frame steps from the previous frame's sim time
    to its own with a fixed dt, so RP_FPS previews show the same motion.
    Colour: v through PALETTE plus u as a cream front, upsampled bilinearly to
    1080 x 1920, with bloom; the rod is splatted as a cool trail with glow.
    Sound: upward crossings of u = 0.5 at the seven PROBES, timed by
    interpolating within the frame's steps.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    rods = list(zip(CUT_TAUS, rod_paths(rng)))
    pace_xy = np.array([GRID_W * H_GRID / 2, GRID_H * H_GRID / 2]) + rng.uniform(
        -3, 3, 2
    )

    device = select_device()
    torch.backends.cudnn.benchmark = False
    ys = torch.arange(GRID_H, device=device, dtype=torch.float32) * H_GRID
    xs = torch.arange(GRID_W, device=device, dtype=torch.float32) * H_GRID
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pacer = ((xx - pace_xy[0]) ** 2 + (yy - pace_xy[1]) ** 2) < PACE_R**2
    u = torch.zeros(GRID_H, GRID_W, device=device)
    v = torch.zeros(GRID_H, GRID_W, device=device)
    probe_ij = [(int(py * (GRID_H - 1)), int(px * (GRID_W - 1))) for px, py in PROBES]
    probe_i = torch.tensor([p[0] for p in probe_ij], device=device)
    probe_j = torch.tensor([p[1] for p in probe_ij], device=device)

    lut = torch.tensor(palette_lut(PALETTE, 512), device=device, dtype=torch.float32)
    front = torch.tensor(FRONT, device=device)
    rod_colour = torch.tensor(ROD, device=device)
    kernel = gaussian_kernel(3.0, device)
    scale_px = WIDTH / (GRID_W * H_GRID)  # screen px per sim unit
    rods_t = [
        (
            torch.tensor(path * scale_px, device=device, dtype=torch.float32),
            torch.tensor(path_len, device=device, dtype=torch.float32),
        )
        for _, (path, path_len) in rods
    ]
    vy, vx = torch.meshgrid(
        torch.linspace(-1, 1, HEIGHT, device=device),
        torch.linspace(-1, 1, WIDTH, device=device),
        indexing="ij",
    )
    vignette = 1.0 - 0.35 * (vx**2 * 0.6 + vy**2 * 0.5).clamp(0, 1)

    state = {"step": 0, "t": 0.0, "pace": 0}
    events = []

    def laplacian(f):
        p = F.pad(f[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * f) / (
            H_GRID * H_GRID
        )

    def advance(frame):
        """Step the medium to this frame's sim time; record probe crossings."""
        nonlocal u, v
        t = frame / FPS
        target = int(round(sim_time(t) / DT))
        n = target - state["step"]
        before = u[probe_i, probe_j] > 0.5
        crossings = []
        for k in range(n):
            tau = state["step"] * DT
            if tau >= state["pace"] * PACE:
                u[pacer] = 1.0
                state["pace"] += 1
            for cut_tau, (path, path_len) in rods:
                reach = (tau - cut_tau) * CUT_SPEED
                if 0 <= reach and reach - DT * CUT_SPEED < path_len[-1]:
                    lo = np.searchsorted(path_len, max(reach - DT * CUT_SPEED, 0))
                    hi = np.searchsorted(path_len, reach)
                    for x0, y0 in path[max(lo - 1, 0) : hi + 1]:
                        inside = ((xx - x0) ** 2 + (yy - y0) ** 2) < ROD_R**2
                        u[inside] = 0.0
                        v[inside] = torch.clamp(v[inside], min=ROD_WAKE)
            threshold = (v + B) / A
            u = (u + DT * (laplacian(u) + u * (1 - u) * (u - threshold) / EPS)).clamp_(
                0, 1
            )
            v = v + DT * (u - v)
            state["step"] += 1
            if k % 4 == 3 or k == n - 1:
                now = u[probe_i, probe_j] > 0.5
                crossings.append((k + 1, now & ~before))
                before = now
        if crossings and frame > 0:
            hits = torch.stack([c for _, c in crossings]).cpu().numpy()
            for (k, _), row in zip(crossings, hits):
                for probe in np.nonzero(row)[0]:
                    events.append((t - (1 - k / n) / FPS, int(probe)))
        state["t"] = t
        return t

    def render(frame):
        t = advance(frame)
        tau = float(sim_time(t))
        uv = torch.stack([u, v])[None]
        big = F.interpolate(
            uv, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False
        )[0]
        bu, bv = big[0], big[1]
        level = (bv / 0.9).clamp(0, 1) ** 0.8
        image = lut[(level * (lut.shape[0] - 1)).long()]
        image = image + (bu**2)[..., None] * (front - image)
        glow = wide_blur(bu[..., None] * front + 0.4 * image, 18.0, 4)
        image = image + 0.25 * glow

        # The rod: a cool trail along the swept path, fading after it passes.
        for (cut_tau, (_, path_len)), (path_t, path_len_t) in zip(rods, rods_t):
            reach = (tau - cut_tau) * CUT_SPEED
            if reach <= 0 or reach > path_len[-1] + 2.5 * CUT_SPEED:
                continue
            rod = torch.zeros(HEIGHT, WIDTH, 3, device=device)
            age = (reach - path_len_t) / CUT_SPEED  # sim units since the rod passed
            weight = torch.where(age >= 0, torch.exp(-age / 0.6), torch.zeros_like(age))
            splat(rod, path_t[:, 0], path_t[:, 1], weight[:, None] * rod_colour)
            rod = blur(rod, kernel) * 6.0
            image = image + rod + 0.6 * wide_blur(rod, 20.0, 4)

        image = image * vignette[..., None]
        image = image * float(smoothstep((DURATION - t) / FADE))
        return (image.clamp(0, 1) * 255).round()

    if settings.audio_only:
        for frame in range(FRAME_COUNT):
            advance(frame)
    else:
        render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    logger.info(f"{len(events)} wave arrivals at the probes")
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(events, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
