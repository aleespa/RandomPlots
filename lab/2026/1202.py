"""
Why a Cheetah's Tail Ends in Rings

A cheetah's coat drawn by one equation: the spots on its body and the rings on
its tail are the same Turing pattern. Static settles into spots on the flank,
then the pattern spreads down the tail, and where the tail grows thin the
spots have no room left and turn into rings.

In 1952 Alan Turing showed that chemicals reacting and spreading through skin
can make spots or stripes with a natural size of their own. The biologist James
Murray saw what follows: a broad patch of skin has room for spots, but a tail
narrower than about one spot can only fit bands across it. So a spotted animal
can have a ringed tail, but a striped animal can never have a spotted one.
Cheetahs, jaguars and genets all agree.

∂u/∂t = ru − (1 + ∇²)²u + gu² − u³

Listen: a Rhodes chord as the spots appear, then a note for every ring that
forms down the tail, pitched by its height, over a soft lo-fi beat. The home
chord lands at the tip.

Made with Python & PyTorch. Sound synthesized in NumPy.
#turingpatterns #mathematicalbiology #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.interpolate import CubicSpline

from common import audio
from common.gpu import select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale, beat, quantize
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Why do cheetah tails end in rings?'
HOOK_Y = 230  # placed by hand
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to the background at the end

# --- The animal (screen pixels, y down) ------------------------------------------
BODY = (300.0, -260.0, 760.0, 900.0)  # flank: ellipse centre x, y and semi-axes
TAIL = [  # centreline of the tail, from inside the flank to the curled tip
    (600.0, 260.0),
    (800.0, 560.0),
    (900.0, 860.0),
    (840.0, 1120.0),
    (660.0, 1320.0),
    (480.0, 1500.0),
    (400.0, 1680.0),
    (450.0, 1830.0),
    (570.0, 1860.0),
    (630.0, 1770.0),
]
TAIL_R = (175.0, 27.0, 2.2)  # radius at the root, at the tip, taper exponent
BLEND = 60.0  # px: smooth union of flank and tail
BODY_MARGIN = (
    80.0  # px: the flank's pattern runs on past its edge (cropped, no edge effects)
)
SPACING = 3.0  # px between centreline samples

# --- Swift-Hohenberg: u_t = r u - (1 + lap)^2 u + g u^2 - u^3 ---------------------
CELL = 2  # screen px per grid cell: the field lives on a 540 x 960 grid
WAVELENGTH = 56.0  # screen px per pattern wavelength (2 pi in SH units)
R_IN, R_OUT = 0.18, -0.6  # control parameter inside the skin / outside it
G = 1.6  # quadratic term: favours hexagonal spots
DT = 0.3
NOISE0 = 0.05  # initial static
FORCING = 3e-3  # a fixed whisper of noise everywhere, the seed of the tail's rings
T_END = 330.0  # sim time at the end of the run
SIM_END = 0.88  # fraction of the video by which sim time reaches T_END
EASE = 0.3  # fraction of that run spent slowing down
FRONT_T = (
    10.0,
    290.0,
)  # sim times at which the switch-on front leaves the root / reaches the tip
FRONT_SOFT = 30.0  # px

# --- Look -------------------------------------------------------------------------
BACKGROUND = "#0a0d12"
PELT = ["#2a1608", "#7a4618", "#c9862f", "#e8b25a", "#f7dfa6"]  # shadow .. highlight
INK = "#160d08"
LIGHT = np.array([-0.45, -0.55, 0.7])

# --- Sound ------------------------------------------------------------------------
STYLE = "lofi_keys"  # common.music.STYLES; an --audio-only tag naming one overrides it
BPM = 80.0
BASS_LINE = (0, 3, 0, 4, 0)


def sim_time(t):
    """
    Sim time at video time t (seconds): steady, then easing to a stop at T_END
    at SIM_END * DURATION (the last EASE of the way decelerates linearly).
    """
    x = np.clip(np.asarray(t, float) / (SIM_END * DURATION), 0.0, 1.0)
    y = np.clip(x - (1.0 - EASE), 0.0, None)
    return T_END * (x - y * y / (2.0 * EASE)) / (1.0 - EASE / 2.0)


def front_s(tau, length):
    """Arc length of the switch-on front along the tail at sim time tau."""
    x = np.clip((tau - FRONT_T[0]) / (FRONT_T[1] - FRONT_T[0]), 0.0, 1.0)
    return -2 * FRONT_SOFT + x * (length + 4 * FRONT_SOFT)


def centreline():
    """Tail centreline sampled every SPACING px: points (M, 2), arc length, radius."""
    p = np.array(TAIL)
    chord = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    spline = CubicSpline(chord, p, bc_type="natural")
    fine = spline(np.linspace(0, chord[-1], 20000))
    seg = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))]
    )
    s = np.arange(0, seg[-1], SPACING)
    points = np.stack([np.interp(s, seg, fine[:, k]) for k in range(2)], axis=1)
    r0, r1, power = TAIL_R
    radius = r1 + (r0 - r1) * (1 - s / s[-1]) ** power
    return points, s, radius


def anatomy(device, scale):
    """
    Signed distance to the animal, shading normal and tail arc length on a grid
    `scale` screen px per cell: (sdf, nx, ny, s_tail, w_tail), each (H, W).
    """
    points, s, radius = centreline()
    h, w = HEIGHT // scale, WIDTH // scale
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * scale
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * scale
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    c = torch.tensor(points, device=device, dtype=torch.float32)
    rad = torch.tensor(radius, device=device, dtype=torch.float32)
    arc = torch.tensor(s, device=device, dtype=torch.float32)
    d_tail = torch.empty(h, w, device=device)
    best = torch.empty(h, w, device=device, dtype=torch.long)
    rows = max(1, 2_000_000 // (w * len(points)) * 8)
    for i in range(0, h, rows):
        px = xx[i : i + rows, :, None] - c[:, 0]
        py = yy[i : i + rows, :, None] - c[:, 1]
        d = torch.sqrt(px * px + py * py) - rad
        d_tail[i : i + rows], best[i : i + rows] = d.min(dim=-1)
    tx = (xx - c[best, 0]) / rad[best]
    ty = (yy - c[best, 1]) / rad[best]
    s_tail = arc[best]

    cx, cy, a, b = BODY
    qx, qy = (xx - cx) / a, (yy - cy) / b
    q = torch.sqrt(qx * qx + qy * qy)
    d_body = (q - 1.0) * min(a, b)

    k = BLEND
    h = (0.5 + 0.5 * (d_body - d_tail) / k).clamp(0, 1)
    sdf = d_body + h * (d_tail - d_body) - k * h * (1 - h)
    # Inside the flank the flank wins (shading, pattern); the tail takes over
    # only once it has left it.
    x = ((d_body + 0.5 * k) / (1.5 * k)).clamp(0, 1)
    w_tail = h * x * x * (3 - 2 * x)
    nx = w_tail * tx + (1 - w_tail) * qx
    ny = w_tail * ty + (1 - w_tail) * qy
    return sdf, nx, ny, s_tail, w_tail, float(s[-1])


def soundtrack(rings, body_time, tip_time, style_name, rng):
    """
    Lo-fi keys. A brush of noise for the static, the flank's spots condensing
    as a rolled Rhodes chord; then every
    ring that forms down the tail strikes one Rhodes note, pitch by its height
    on screen (higher up, higher note), pan by its x, softer as the tail thins,
    pulled 70 % onto an eighth-note grid at BPM. A soft kick/tick beat and a
    Rhodes bass walk BASS_LINE underneath once the spots are in, and the home
    chord lands as the last ring forms at the tip.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    end = min(tip_time + 0.3, DURATION - 2.0)

    hiss = audio.lowpass(audio.tick(rng, 0.5, 0.25, 2500.0), 5000.0, circular=False)
    audio.add(track, audio.pan(hiss, -0.2), 0.0, **kw)  # the static
    roll = max(0.12, body_time - 0.35)
    for i, degree in enumerate((0, 2, 4, 6, 9)):  # the spots condense
        note = style.voice(scale.hz(degree), 0.35 + 0.05 * i)
        audio.add(track, audio.pan(note, -0.4 + 0.2 * i), roll + 0.09 * i, **kw)

    if len(rings):
        times = quantize([r[0] for r in rings], DURATION, BPM, 2, amount=0.7)
        last = -1.0
        for when, (_, x, y, width) in sorted(zip(times, rings)):
            if when - last < 0.11 or when > end - 0.05:
                continue
            last = when
            degree = int(round(11 - 13 * y / HEIGHT))
            velocity = 0.3 + 0.35 * np.clip(width / (2 * TAIL_R[0]), 0, 1)
            note = audio.pan(
                style.voice(scale.hz(degree), velocity), 1.3 * (x / WIDTH - 0.5)
            )
            audio.add(track, note, when, **kw)

    step = beat(BPM)
    start = np.ceil(body_time / step) * step
    for i, when in enumerate(np.arange(start, end, step)):
        if i % 2 == 0:
            audio.add(track, audio.kick(0.4, 55.0), when, **kw)
        audio.add(
            track, audio.pan(audio.tick(rng, 0.18, 0.04), 0.3), when + step / 2, **kw
        )
    for i, when in enumerate(np.arange(start, end, 4 * step)):
        bass = style.bass(scale.hz(BASS_LINE[i % len(BASS_LINE)], -1), 0.55, length=4.0)
        audio.add(track, bass, when, **kw)
    for degree in (0, 2, 4):
        audio.add(
            track,
            audio.pad(scale.hz(degree, -1), end - body_time, 0.1, 2.0, 2.5, 0.25),
            body_time,
            **kw,
        )

    for i, degree in enumerate((0, 2, 4, 6, 7)):  # the home chord at the tip
        audio.add(track, style.voice(scale.hz(degree), 0.5), end + 0.04 * i, **kw)
    audio.add(track, style.bass(scale.hz(0, -1), 0.6, length=4.0), end, **kw)
    track = audio.lowpass(track, 6000.0, circular=False)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Murray's tail theorem with a Turing pattern, as a one-shot.

    The pattern is the Swift-Hohenberg equation, the generic model of a Turing
    instability: u_t = r u - (1 + lap)^2 u + g u^2 - u^3, whose unstable band
    sits at wavenumber 1 (wavelength 2 pi = WAVELENGTH screen px). With g = G
    the quadratic term makes hexagonal spots (u maxima, drawn as INK) win over
    stripes. It is solved on a 540 x 960 periodic grid (CELL = 2 px) on the GPU
    by a semi-implicit pseudo-spectral step: the linear operator
    -(1 - k^2)^2 implicitly, r(x) u + g u^2 - u^3 + FORCING * xi(x) explicitly,
    dt = DT (~1100 FFT steps in all, about a second).

    The animal is a signed-distance shape: a flank (an ellipse off the top of
    the frame) smoothly united with a tail, a tube of radius tapering from
    TAIL_R[0] to TAIL_R[1] px along a spline through TAIL. Outside it r = R_OUT
    (stable, so the field dies off: an effective Dirichlet edge); inside,
    r = R_IN. Where the tail is narrower than about one wavelength only modes
    along it fit, so the spots become bands across it: Murray's theorem. The
    flank's r region runs BODY_MARGIN px past its drawn edge, so the flank shows
    no edge effects and its silhouette simply crops the spots.

    Time: the flank starts as static (NOISE0 * white noise) and condenses into
    spots in the first second. The tail is held stable (r = R_OUT) until a
    switch-on front passes down it along arc length, from sim time FRONT_T[0]
    at the root to FRONT_T[1] at the tip, seeded by the fixed whisper xi; the
    rings therefore form one after another down the tail. Video time maps to
    sim time by sim_time (steady, easing to a stop at SIM_END), then a rest and
    a FADE. Every frame steps the field from the previous frame's sim time to
    its own with a fixed dt (constant cost, render_frames(parallel=False)), so
    RP_FPS previews show the same motion.

    Drawing: u is normalised by min(max |u|, 3 std) over the flank (so the
    opening static is visible), upsampled bilinearly to 1080 x 1920 and
    thresholded smoothly into ink over a pelt shaded with a Lambert light on
    the shape's normals (sphere-like flank, cylinder-like tail), over a dark
    vignetted background with a faint rim glow and a fixed dither.

    Sound (`soundtrack`): u is recorded along the tail centreline every other
    step; each local maximum of the final profile is a ring (or a spot on the
    centreline), sounding when it first reaches 60 % of its final height.
    The posted video and soundtrack use the default --seed 0 (audio seed 0).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    gh, gw = HEIGHT // CELL, WIDTH // CELL
    sdf, nx, ny, s_tail, w_tail, length = anatomy(device, CELL)
    inside = torch.sigmoid(-(sdf - BODY_MARGIN * (1 - w_tail)) / 6.0)
    noise0 = torch.tensor(
        rng.standard_normal((gh, gw)), device=device, dtype=torch.float32
    )
    whisper = torch.tensor(
        rng.standard_normal((gh, gw)), device=device, dtype=torch.float32
    )
    u = NOISE0 * noise0 * inside

    h = 2 * np.pi * CELL / WAVELENGTH  # SH units per grid cell
    ky = torch.fft.fftfreq(gh, d=h / (2 * np.pi), device=device)
    kx = torch.fft.rfftfreq(gw, d=h / (2 * np.pi), device=device)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    implicit = 1.0 / (1.0 + DT * (1.0 - k2) ** 2)

    # Listening line: the tail centreline on the grid.
    points, arc, radius = centreline()
    pi_ = torch.tensor(
        np.clip(points[:, 1] / CELL, 0, gh - 1).astype(int), device=device
    )
    pj_ = torch.tensor(
        np.clip(points[:, 0] / CELL, 0, gw - 1).astype(int), device=device
    )
    history = []  # (sim time, u along the centreline)

    state = {"step": 0}

    def advance(frame):
        nonlocal u
        t = frame / FPS
        target = int(round(sim_time(t) / DT))
        while state["step"] < target:
            tau = state["step"] * DT
            act = torch.sigmoid((front_s(tau, length) - s_tail) / FRONT_SOFT)
            act = w_tail * act + (1 - w_tail)
            r = R_OUT + (R_IN - R_OUT) * inside * act
            nonlinear = r * u + G * u * u - u * u * u + FORCING * whisper
            u = torch.fft.irfft2(
                (torch.fft.rfft2(u) + DT * torch.fft.rfft2(nonlinear)) * implicit,
                s=(gh, gw),
            )
            state["step"] += 1
            if state["step"] % 2 == 0:
                history.append((state["step"] * DT, u[pi_, pj_]))
        return t

    # --- picture --------------------------------------------------------------
    up = dict(size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
    geo = F.interpolate(torch.stack([sdf, nx, ny])[None], **up)[0]
    big_sdf, bnx, bny = geo[0], geo[1], geo[2]
    nz = (1 - bnx**2 - bny**2).clamp(0, 1).sqrt()
    light = LIGHT / np.linalg.norm(LIGHT)
    lambert = (bnx * light[0] + bny * light[1] + nz * light[2]).clamp(0, 1)
    shade = (0.18 + 0.82 * lambert) * (0.55 + 0.45 * nz)
    alpha = (0.5 - big_sdf / 1.5).clamp(0, 1)
    pelt = torch.tensor(palette_lut(PELT, 512), device=device)
    pelt_img = pelt[(shade.clamp(0, 1) * 511).long()]
    ink = torch.tensor(palette_lut([INK, INK], 2)[0], device=device)
    bg = torch.tensor(palette_lut([BACKGROUND, BACKGROUND], 2)[0], device=device)
    vy, vx = torch.meshgrid(
        torch.linspace(-1, 1, HEIGHT, device=device),
        torch.linspace(-1, 1, WIDTH, device=device),
        indexing="ij",
    )
    vignette = 1.0 - 0.4 * (0.5 * vx**2 + 0.5 * vy**2).clamp(0, 1)
    dither = torch.tensor(
        rng.uniform(-0.5, 0.5, (HEIGHT, WIDTH, 1)) / 255.0,
        device=device,
        dtype=torch.float32,
    )  # breaks the banding of the dark gradient
    body_mask = (inside * (1 - w_tail)) > 0.5

    def render(frame):
        t = advance(frame)
        body = u[body_mask]
        level = torch.minimum(body.abs().amax(), 3.0 * body.std()).clamp(min=1e-3)
        field = F.interpolate((u / level)[None, None], **up)[0, 0]
        spot = smooth(0.15, 0.5, field)
        skin = (
            pelt_img * (1 - 0.35 * spot[..., None])
            + spot[..., None] * (ink * (0.7 + 0.3 * shade[..., None]) - pelt_img) * 0.95
        )
        rim = wide_blur(alpha[..., None].expand(-1, -1, 3).contiguous(), 30.0, 4)
        image = bg * vignette[..., None] + 0.05 * rim * pelt[300]
        image = image + alpha[..., None] * (skin - image)
        image = image * float(smoothstep((DURATION - t) / FADE)) + dither
        return (image.clamp(0, 1) * 255).round()

    if settings.audio_only:
        for frame in range(FRAME_COUNT):
            advance(frame)
    else:
        render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)

    rings, body_time, tip_time = listen(history, points, arc, radius, length)
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(rings, body_time, tip_time, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def smooth(lo, hi, x):
    x = ((x - lo) / (hi - lo)).clamp(0, 1)
    return x * x * (3 - 2 * x)


def video_time(tau):
    grid = np.linspace(0, DURATION, 6001)
    return np.interp(tau, sim_time(grid), grid)


def listen(history, points, arc, radius, length):
    """Rings along the tail: (time, x, y, width) of each, from the centreline history."""
    taus = np.array([h[0] for h in history])
    line = torch.stack([h[1] for h in history]).cpu().numpy()  # (steps, M)
    final = line[-1]
    peaks = [
        i
        for i in range(2, len(final) - 2)
        if final[i] > 0.2 and final[i] == final[i - 2 : i + 3].max()
    ]
    rings = []
    for i in peaks:
        above = np.nonzero(line[:, i] > 0.6 * final[i])[0]
        if len(above):
            rings.append((float(video_time(taus[above[0]])), *points[i], 2 * radius[i]))
    logger.info(
        f"{len(rings)} rings along the tail, forming from "
        f"{min([r[0] for r in rings], default=0):.1f} to {max([r[0] for r in rings], default=0):.1f} s"
    )
    body_time = float(video_time(18.0))
    tip_time = max([r[0] for r in rings], default=12.0)
    return rings, body_time, tip_time


if __name__ == "__main__":
    generate()
