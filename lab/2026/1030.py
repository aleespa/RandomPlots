"""
Two Mirrors

One flash of light, in two mirrored rooms of the same size. The top room
gathers every ray back into a single blazing point. The bottom room, almost
the same shape, scatters them into fog.

The top room is an ellipse, and the flash goes off at one of its two focal
points. Every ray bounces off the wall and passes through the other focus,
all at the same instant, because the trip is always the same length. Each
pass also tilts the rays a little closer to the long axis, until the whole
rainbow has folded into one bead of light swinging from focus to focus.

The bottom room is a stadium: two half-circles joined by short straight
walls. In 1979 Leonid Bunimovich proved it chaotic. Those short straight
pieces are enough to scramble every ray, and the rainbow of directions mixes
until the whole room glows white.

Listen to the two rooms. A bell rings each time the ellipse's light passes
through a focus, in perfect time. Below, a few rays of the stadium tap out
every wall they hit, together at first, then scattering into patter, while
a dissonant chord swells as the light spreads. Everything slows to a stop
with the light.

Made with Python & PyTorch. Sound synthesized in NumPy.
#billiards #chaostheory #optics #mathart #randomplots
"""

import gc

import cv2
import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The rooms (world units) --------------------------------------------------------
SEMI_A, SEMI_B = 1.0, 0.8  # the ellipse; its foci sit at +-sqrt(a^2 - b^2)
RADIUS = 0.7  # the stadium's half-circles; the straight walls are set for equal area
SOURCE = (-0.6, 0.0)  # the flash, at the ellipse's left focus, same spot in both
PHOTONS = 300_000  # per room
MAX_BOUNCES = 240  # grazing rays creep round the half-circles in many short hops

# --- Time ---------------------------------------------------------------------------
SPEED = 1.5  # world units per second: the ellipse refocuses every 2a / SPEED s
SLOW = (11.5, DURATION)  # s: time slows from full speed to a stop over this span

# --- Look ---------------------------------------------------------------------------
SCALE = 440.0  # px per world unit
CENTRES = ((WIDTH / 2, 0.27 * HEIGHT), (WIDTH / 2, 0.73 * HEIGHT))  # room centres, px
TRAIL, TRAIL_STEPS = 0.05, 6  # world units of motion blur behind each photon
EXPOSURE, GAMMA = 0.45, 0.5  # light = 1 - exp(-EXPOSURE * density^GAMMA)
SOFT = 1.0  # px blur of the light
GLOWS = (
    (12.0, 2, 0.04),
    (45.0, 8, 0.02),
)  # (sigma px, shrink, gain): halos of linear light
BLOOM_SIGMA, BLOOM_FACTOR, BLOOM_GAIN = 12.0, 4, 0.3
WALL_COLOUR, WALL_ALPHA, WALL_WIDTH = "#c8d2e0", 0.55, 4
MARK_ALPHA = 0.5  # the foci and the source, faint dots

# --- Sound -------------------------------------------------------------------------
STYLE = "harp"  # common.music.STYLES; an --audio-only tag naming one overrides it
LISTENED = 3  # stadium photons whose wall hits are played
WALL_DEGREES = 9  # scale degrees once round the stadium wall
MIX_SAMPLE = 20_000  # photons used to measure how evenly the stadium is lit


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def straight() -> float:
    """Half-length of the stadium's straight walls: equal area with the ellipse."""
    return (np.pi * SEMI_A * SEMI_B - np.pi * RADIUS**2) / (4 * RADIUS)


def physical_time(seconds):
    """Light's own clock: full speed, then easing to a stop over SLOW."""
    t0, t1 = SLOW
    s = np.clip(np.asarray(seconds, float), 0, None)
    u = np.clip((s - t0) / (t1 - t0), 0, 1)
    # The integral of 1 - smoothstep(u), times the span: u - u^3 + u^4 / 2.
    eased = (t1 - t0) * (u - u**3 + 0.5 * u**4)
    return np.where(s < t0, s, t0 + eased)


def ellipse_hit(p, d):
    """Distance along d from p (inside) to the ellipse, and the unit normal there."""
    a2, b2 = SEMI_A**2, SEMI_B**2
    qa = d[:, 0] ** 2 / a2 + d[:, 1] ** 2 / b2
    qb = 2 * (p[:, 0] * d[:, 0] / a2 + p[:, 1] * d[:, 1] / b2)
    qc = p[:, 0] ** 2 / a2 + p[:, 1] ** 2 / b2 - 1
    s = (-qb + np.sqrt(np.maximum(qb * qb - 4 * qa * qc, 0))) / (2 * qa)
    q = p + s[:, None] * d
    n = np.stack([q[:, 0] / a2, q[:, 1] / b2], 1)
    return s, n / np.linalg.norm(n, axis=1, keepdims=True)


def stadium_hit(p, d):
    """Distance along d from p (inside) to the stadium wall, and the unit normal."""
    half = straight()
    best = np.full(len(p), np.inf)
    normal = np.zeros_like(p)
    for wall in (RADIUS, -RADIUS):  # the straight walls y = +-r, |x| <= half
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (wall - p[:, 1]) / d[:, 1]
        x = p[:, 0] + s * d[:, 0]
        ok = (s > 1e-9) & (np.abs(x) <= half) & (s < best)
        best = np.where(ok, s, best)
        normal[ok] = (0.0, np.sign(wall))
    for side in (half, -half):  # the half-circles round (+-half, 0)
        rel = p - np.array([side, 0.0])
        qb = (rel * d).sum(1)
        qc = (rel * rel).sum(1) - RADIUS**2
        s = -qb + np.sqrt(np.maximum(qb * qb - qc, 0))
        q = rel + s[:, None] * d
        ok = (s > 1e-9) & (np.sign(side) * q[:, 0] >= 0) & (s < best)
        best = np.where(ok, s, best)
        normal[ok] = q[ok] / RADIUS
    return best, normal


def trace(hit, angles) -> dict:
    """
    Bounce every photon from SOURCE until all of them outlast the clip (at
    most MAX_BOUNCES): the start time, start point and direction of each
    straight segment, (N, bounces + 1) float32 arrays.
    """
    p = np.tile(np.array(SOURCE, float), (len(angles), 1))
    d = np.stack([np.cos(angles), np.sin(angles)], 1)
    t = np.zeros(len(angles))
    end = float(physical_time(DURATION))

    def f32(*arrays):
        return tuple(a.astype(np.float32) for a in arrays)

    keep = [f32(t, p, d)]
    while t.min() <= end and len(keep) <= MAX_BOUNCES:
        s, n = hit(p, d)
        p = p + s[:, None] * d
        d = d - 2 * (d * n).sum(1, keepdims=True) * n
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = t + s / SPEED
        keep.append(f32(t, p, d))
    stuck = int((t <= end).sum())
    if stuck:
        logger.warning(f"{stuck} photons ran out of bounces before the end")
    return {key: np.stack([k[i] for k in keep], 1) for i, key in enumerate("tpd")}


def walls(canvas_shape) -> np.ndarray:
    """The two mirrors and the faint marks, as an alpha mask (H, W), drawn at 4x."""
    big = np.zeros((canvas_shape[0] * 4, canvas_shape[1] * 4), np.uint8)
    u = np.linspace(0, 2 * np.pi, 2000)
    half = straight()
    ellipse = np.stack([SEMI_A * np.cos(u), SEMI_B * np.sin(u)], 1)
    right = np.stack([half + RADIUS * np.sin(u / 2), RADIUS * np.cos(u / 2)], 1)
    stadium = np.concatenate([right, right[::-1] * [-1, 1]])  # closed by the polyline
    for shape, (cx, cy) in zip((ellipse, stadium), CENTRES):
        pts = np.stack([cx + SCALE * shape[:, 0], cy - SCALE * shape[:, 1]], 1)
        cv2.polylines(
            big,
            [np.round(pts * 4 * 4).astype(np.int32)],
            True,
            255,
            WALL_WIDTH * 4,
            cv2.LINE_AA,
            2,
        )
    focus = np.sqrt(SEMI_A**2 - SEMI_B**2)
    marks = [(CENTRES[0], (focus, 0)), (CENTRES[0], (-focus, 0)), (CENTRES[1], SOURCE)]
    for (cx, cy), (x, y) in marks:
        centre = (int(4 * (cx + SCALE * x)), int(4 * (cy - SCALE * y)))
        cv2.circle(big, centre, 10, int(255 * MARK_ALPHA), -1, cv2.LINE_AA)
    mask = cv2.resize(big, canvas_shape[::-1], interpolation=cv2.INTER_AREA)
    return mask.astype(np.float32) / 255.0


def wall_clock(physical):
    """Video seconds at which light's own clock reads `physical` (inverse of physical_time)."""
    grid = np.linspace(0, DURATION, 30001)
    return np.interp(physical, physical_time(grid), grid)


def mixing(room: dict, rng: np.random.Generator) -> tuple:
    """
    How evenly MIX_SAMPLE photons light the stadium over time: the entropy of
    their positions on a 24 x 16 grid, 0 at the flash, 1 when uniform. Returns
    (seconds, mix).
    """
    pick = rng.choice(room["t"].shape[0], MIX_SAMPLE, replace=False)
    t, pos, d = room["t"][pick], room["p"][pick], room["d"][pick]
    seconds = np.linspace(0, DURATION, 151)
    half = straight() + RADIUS
    edges = (np.linspace(-half, half, 25), np.linspace(-RADIUS, RADIUS, 17))
    reachable = None
    out = []
    for now in physical_time(seconds):
        i = np.clip((t <= now).sum(1) - 1, 0, t.shape[1] - 1)
        rows = np.arange(len(pick))
        q = pos[rows, i] + (SPEED * (now - t[rows, i]))[:, None] * d[rows, i]
        counts = np.histogram2d(q[:, 0], q[:, 1], edges)[0].ravel()
        if reachable is None:  # cells inside the stadium, from a dense uniform cloud
            u = np.stack(
                [
                    rng.uniform(-half, half, 200_000),
                    rng.uniform(-RADIUS, RADIUS, 200_000),
                ],
                1,
            )
            inside = np.abs(u[:, 0]) <= straight()
            inside |= np.hypot(np.abs(u[:, 0]) - straight(), u[:, 1]) <= RADIUS
            reachable = np.histogram2d(u[inside, 0], u[inside, 1], edges)[0].ravel() > 0
        prob = counts[reachable] / counts.sum()
        prob = prob[prob > 0]
        out.append(-(prob * np.log(prob)).sum() / np.log(reachable.sum()))
    mix = np.array(out)
    return seconds, (mix - mix[0]) / (mix.max() - mix[0])


def soundtrack(rooms: list, style_name: str, rng: np.random.Generator):
    """
    A one-shot: order in the top room, chaos in the bottom one.

    - The ellipse: a bell each time the pulse passes through a focus (every
      2a / SPEED of light's own time), high for the right focus and a fifth
      lower for the left, panned with it. As the wavefront folds into the
      bead the bells keep exact time; as time slows they slow with it.
    - The stadium: LISTENED photons (chosen with the audio rng) sound each
      wall they hit, the style's voice, pitched by where on the wall
      (WALL_DEGREES once round) and panned by x. They start almost together
      and drift apart into irregular patter.
    - Underneath, the home chord on a soft pad, and a dissonant cluster that
      swells with how evenly the stadium is lit (`mixing`), then melts into
      the home chord as time slows (SLOW).
    All onsets are light's bounce times mapped back to video seconds, so the
    music decelerates with the picture.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    sr = audio.SAMPLE_RATE
    end = float(physical_time(DURATION))
    focus = np.sqrt(SEMI_A**2 - SEMI_B**2)

    # The ellipse: bells at every focal pass.
    period = 2 * SEMI_A / SPEED
    bells = 0
    for k in range(1, int(end / period) + 1):
        when = float(wall_clock(k * period))
        right = k % 2 == 1  # the flash starts at the left focus
        degree = 7 if right else 3
        quiet = 1 - 0.5 * float(smoothstep((when - SLOW[0]) / (DURATION - SLOW[0])))
        note = audio.bell(scale.hz(degree, 1), 0.55 * quiet)
        audio.add(track, audio.pan(note, 0.5 if right else -0.5), when, **kw)
        bells += 1

    # The stadium: a few photons tap out their wall hits.
    stadium = rooms[1]
    taps = 0
    for photon in rng.choice(stadium["t"].shape[0], LISTENED, replace=False):
        hits = stadium["t"][photon, 1:]
        where = stadium["p"][photon, 1:]
        for physical, (x, y) in zip(hits, where):
            if physical > end:
                break
            when = float(wall_clock(physical))
            around = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
            degree = int(round(around * WALL_DEGREES)) - 2
            velocity = rng.uniform(0.22, 0.34)
            pan = float(np.clip(x / (straight() + RADIUS), -1, 1))
            note = style.voice(scale.hz(degree), velocity)
            audio.add(track, audio.pan(note, 0.8 * pan), when, **kw)
            taps += 1

    # The pad: home chord throughout; a cluster swelling with the mixing.
    seconds, mix = mixing(stadium, rng)
    home = [float(audio.midi_to_hz(n)) for n in scale.chord(0, 3, octave=-1)]
    for hz in home:
        audio.add(track, audio.pad(hz, DURATION - 1.0, 0.07, 1.5, 1.0, 0.3), 0.0, **kw)
    cluster = audio.silence(DURATION)
    for n in (scale.note(1, -1), scale.note(1, -1) + 1, scale.note(4, -1) + 1):
        hz = float(audio.midi_to_hz(n))
        audio.add(
            cluster, audio.pad(hz, DURATION - 1.0, 0.1, 1.0, 1.0, 0.45), 0.0, **kw
        )
    t = np.arange(len(track)) / sr
    swell = np.interp(t, seconds, mix) * (1 - smoothstep((t - SLOW[0]) / 2.5))
    track += cluster * swell[:, None]
    chord = scale.chord(0, 4)
    for k, n in enumerate(chord):
        hz = float(audio.midi_to_hz(n))
        audio.add(
            track,
            audio.pan(style.voice(hz, 0.35), 0.3 * (k - 1.5)),
            SLOW[0] + 0.12 * k,
            **kw,
        )
    bass = scale.hz(0, -2)
    bass = 2 * bass if bass < 65 else bass  # phone speakers stop around 60 Hz
    audio.add(track, style.bass(bass, 0.5, length=4.0), SLOW[0], **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {bells} focus bells, "
        f"{taps} stadium taps; focus at +-{focus:.2f}"
    )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Light billiards in an ellipse and a Bunimovich stadium of equal area.

    PHOTONS rays leave SOURCE (the ellipse's left focus) at evenly spaced
    angles, offset by one random phase from settings.rng, and reflect
    specularly (d' = d - 2 (d.n) n) off each wall. Every bounce is traced up
    front in float64, stored as float32 (`trace`, vectorised over photons; the ellipse hit is a
    quadratic, the stadium's the nearest of two lines and two circles). In the
    ellipse every path from one focus to the wall and on to the other has
    length 2a, so the pulse refocuses every 2a / SPEED seconds, and since
    every focal orbit tends to the major axis the wavefront collapses within
    a few passes into one bead sliding between the foci. In the stadium
    (half-circles of RADIUS joined by straight walls of half-length `straight`)
    the flow is ergodic and the pulse spreads over the whole room. Grazing
    rays creep round the half-circles in many short hops, so each room is
    traced until every photon outlasts the clip.

    A frame is a pure function of t. Light's own clock (`physical_time`) runs at
    SPEED until SLOW, then eases to a stop. On the GPU each photon's segment is
    found with a batched searchsorted and its position interpolated at
    TRAIL_STEPS times along a motion trail TRAIL long (shrinking as time slows),
    then splatted, coloured by launch angle with a cosine rainbow whose hues
    sum to white. The density d is tone-mapped as 1 - exp(-EXPOSURE (d^GAMMA
    + glow)), where the glow is the linear density blurred wide (GLOWS), so
    only concentrated light (the bead, the focus flashes, the rings) gets a
    halo while the fog stays soft; then bloomed, with the walls and the foci
    laid on top. The posted render uses --seed 0 (the default), with the harp
    soundtrack.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    phase = settings.rng.uniform(0, 2 * np.pi / PHOTONS)
    angles = phase + 2 * np.pi * np.arange(PHOTONS) / PHOTONS
    rooms = [trace(ellipse_hit, angles), trace(stadium_hit, angles)]
    logger.info(
        f"Traced {PHOTONS:,} photons per room: "
        f"{' and '.join(str(r['t'].shape[1] - 1) for r in rooms)} bounces; "
        f"stadium straight walls {2 * straight():.3f} long; light travels "
        f"{SPEED * float(physical_time(DURATION)):.1f} units"
    )

    to = dict(device=device, dtype=torch.float32)
    hue = angles / (2 * np.pi)
    rainbow = 0.5 + 0.5 * np.cos(
        2 * np.pi * (hue[:, None] + np.array([0.0, 1 / 3, 2 / 3]))
    )
    colours = torch.tensor(rainbow / TRAIL_STEPS, **to)
    gpu_rooms = [
        {key: torch.tensor(value, **to).contiguous() for key, value in room.items()}
        for room in rooms
    ]
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **to)
    soft = gaussian_kernel(SOFT, device)
    wall = torch.tensor(walls((HEIGHT, WIDTH)), **to)[..., None]
    wall_rgb = torch.tensor(_rgb(WALL_COLOUR), **to)

    def render(frame: int) -> torch.Tensor:
        seconds = frame / FPS
        now = float(physical_time(seconds))
        rate = float(physical_time(seconds + 1e-3) - now) / 1e-3  # 1, slowing to 0
        canvas.zero_()
        for room, (cx, cy) in zip(gpu_rooms, CENTRES):
            for k in range(TRAIL_STEPS):
                when = max(0.0, now - rate * TRAIL / SPEED * k / TRAIL_STEPS)
                query = torch.full((PHOTONS, 1), when, **to)
                index = torch.searchsorted(room["t"], query, right=True) - 1
                index = index.clamp(0, room["t"].shape[1] - 1)
                start = room["t"].gather(1, index)[:, 0]
                p = room["p"][torch.arange(PHOTONS, device=device), index[:, 0]]
                d = room["d"][torch.arange(PHOTONS, device=device), index[:, 0]]
                q = p + (SPEED * (when - start))[:, None] * d
                splat(canvas, cx + SCALE * q[:, 0], cy - SCALE * q[:, 1], colours)
        density = canvas.clamp_min(0) ** GAMMA
        glow = sum(g * wide_blur(canvas, sigma, f) for sigma, f, g in GLOWS)
        light = 1 - torch.exp(-EXPOSURE * (blur(density, soft) + glow))
        image = light + BLOOM_GAIN * wide_blur(light, BLOOM_SIGMA, BLOOM_FACTOR)
        image = image * (1 - WALL_ALPHA * wall) + WALL_ALPHA * wall * wall_rgb
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(rooms, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
