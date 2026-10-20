"""
A Million Forecasts

That spark is not one point but a million: a million copies of the same simple weather
model, Edward Lorenz's, each started less than a millionth away from the others. For a
moment they travel as one. Then the tiny differences grow, doubling again and again, and
the spark smears into a thread, folds over itself and scatters across the whole butterfly.

This is the butterfly effect, the heart of chaos: the equations are exact and nothing is
random, yet a difference too small to measure decides where each forecast ends up. What
survives is the shape. No one can say where a point will be, but every one of them stays
on the same two wings, which is why weather is unpredictable beyond a few days while
climate is not.

A string section starts on a single note, one sound for a million forecasts. As the
spark splits, voices drift off by a semitone into a tense cluster, and when the
butterfly is full they settle into one chord. Plucks follow the switches between wings,
together at first, then scattered.

Made with Python & PyTorch. Sound synthesized in NumPy.
#lorenzattractor #chaostheory #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.004  # fixed integration step, the same at every RP_FPS
POINTS = 1 << 20

# Colour of each point by where it started in the ball: violet -> rose -> gold.
PALETTE = ["#3b1c8c", "#7b2fb0", "#d6407a", "#ff8a5c", "#ffc857"]

KNOBS = dict(
    radius=1e-6,  # the starting ball
    t_start=7.0,  # Lorenz time at the first frame (before it, the ball is sub-pixel)
    speed=1.3,  # Lorenz time units per second of video
    trail=(0.2, 0.045),  # s: trail half-life while a spark -> once spread
    weight=0.6,  # light of one point resting in a pixel (trails included)
    head=0.12,  # extra light of each point where it is now (the bright heads)
    yaw=(-75.0, -25.0),  # degrees about the vertical axis; -45 shows both wings apart
    pitch=12.0,  # degrees above the wings
    scale=30.0,  # px per Lorenz unit at the attractor's centre
)


def lorenz(s: np.ndarray) -> np.ndarray:
    x, y, z = s
    return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])


def reference(t_end: float) -> np.ndarray:
    """
    The centre of the ball, (x, y, z) at every half step from 0 to t_end, in float64:
    settled onto the attractor first, then integrated with RK4 at DT / 2.
    """
    s = np.array([1.0, 1.0, 1.0])
    h = DT / 2
    for _ in range(int(30 / h)):  # settle onto the attractor
        s = _rk4(s, h)
    out = [s]
    for _ in range(int(np.ceil(t_end / h)) + 2):
        s = _rk4(s, h)
        out.append(s)
    return np.array(out)


def _rk4(s, h):
    k1 = lorenz(s)
    k2 = lorenz(s + h / 2 * k1)
    k3 = lorenz(s + h / 2 * k2)
    k4 = lorenz(s + h * k3)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def offset_rate(d: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    d/dt of offsets d from the reference point c: f(c + d) - f(c), exactly. Lorenz is
    quadratic, so this is the linear part plus the products of offsets, with no
    cancellation: float32 offsets lose nothing even at 1e-6.
    """
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
    return torch.stack(
        [
            SIGMA * (dy - dx),
            dx * (RHO - cz) - cx * dz - dy - dx * dz,
            cx * dy + cy * dx + dx * dy - BETA * dz,
        ],
        dim=1,
    )


def lorenz_time(t: float, knobs: dict) -> float:
    """Lorenz time at video time t in [0, 1]: steady, slowing over the last second."""
    seconds = t * DURATION
    tail = max(0.0, seconds - (DURATION - 1.0))  # speed falls linearly to 40 %
    return knobs["t_start"] + knobs["speed"] * (seconds - 0.3 * tail**2)


# Soundtrack: strings only, Rise & resolve.
ROOTS = range(50, 56)  # MIDI root of the held note (D3 .. G3)
WATCHED = 6  # forecasts whose own wing switches are plucked once the cloud has split
SPLIT = 0.3  # cloud spread (Lorenz units) beyond which the spark counts as split
CLUSTER = ((1e-3, 1), (1e-2, -1), (1e-1, 2), (0.5, 6), (2.0, -2))  # spread, semitones
RESOLVE = 6.0  # spread at which the cluster resolves into a chord


def soundtrack(diagnostics: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    A one-shot in strings. `diagnostics` holds, per frame, the cloud's spread (median
    distance of the points from their mean) and the x of the reference orbit and of
    WATCHED points (x < 0 is one wing, x > 0 the other).

    - One note is held by the section from the start: a million forecasts in unison.
    - Each time the spread passes a threshold in CLUSTER (each ten times wider), one
      more voice enters a semitone or two away, so the unison thickens into a tense
      cluster as the spark splits.
    - When the spread reaches RESOLVE (the butterfly is filled) the cluster stops and a
      full minor chord swells in, held to the end.
    - Pizzicato marks wing switches: while the cloud is a spark, the reference orbit's
      (every forecast agrees), panned by wing; once it has split, those of WATCHED
      separate forecasts, quieter and out of step.
    """
    scale = Scale(int(rng.choice(ROOTS)), "minor")
    root = scale.root
    kw = dict(circular=False)
    track = audio.silence(DURATION)
    times = np.arange(len(diagnostics)) / (len(diagnostics) - 1) * DURATION
    spread = diagnostics[:, 0]

    def when(level):
        above = np.flatnonzero(spread >= level)
        return float(times[above[0]]) if len(above) else None

    resolve = when(RESOLVE) or DURATION - 3.0
    split = when(SPLIT) or resolve
    hz = lambda midi: float(audio.midi_to_hz(midi))
    for note, gain in ((root, 0.45), (root - 12, 0.35)):
        audio.add(
            track, audio.strings(hz(note), resolve, gain, 0.4, 1.2, 0.35), 0, **kw
        )
    for level, semitones in CLUSTER:
        start = when(level)
        if start is None or start >= resolve:
            continue
        voice = audio.strings(hz(root + semitones), resolve - start, 0.3, 0.8, 1.2, 0.5)
        audio.add(track, audio.pan(voice.mean(1), 0.15 * semitones), start, **kw)
    for i, degree in enumerate((0, 4, 7, 9, 2)):  # i, v, octave, third above, low
        octave = -1 if i == 4 else 0
        chord = audio.strings(
            scale.hz(degree, octave), DURATION - resolve, 0.42, 1.0, 2.0
        )
        audio.add(track, chord, resolve - 0.4, **kw)

    plucks = 0
    for column in range(1, diagnostics.shape[1]):
        x = diagnostics[:, column]
        for i in np.flatnonzero(np.sign(x[1:]) != np.sign(x[:-1])) + 1:
            moment = float(times[i])
            together = column == 1
            if together != (moment < split) or moment > DURATION - 1.0:
                continue
            wing = float(np.sign(x[i]))
            degree = (4 if wing > 0 else 0) + (0 if together else 2 * column)
            velocity = 0.55 if together else 0.3
            note = audio.pluck(scale.hz(degree, 1), velocity, 1.2, 0.15)
            audio.add(track, audio.pan(note, 0.5 * wing), moment, **kw)
            plucks += 1
    logger.info(
        f"Soundtrack: root MIDI {root}, split at {split:.1f} s, "
        f"resolve at {resolve:.1f} s, {plucks} plucks"
    )
    track = audio.reverb(track, rng, rt60=2.8, wet=0.32, tone=4500.0, circular=False)
    audio.fade(track, 0.0, 0.4)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A ball of 2^20 initial conditions, radius 1e-6, under the Lorenz system
    (sigma 10, rho 28, beta 8/3).

    The ball's centre is integrated in float64 on the CPU (RK4 at DT / 2, after settling
    onto the attractor); every point is that centre plus an offset d, integrated on the
    GPU in float32 with RK4 at DT using the exact offset equation d' = f(c + d) - f(c)
    (quadratic, so no cancellation at tiny d). Video time maps to Lorenz time steadily
    (t_start + speed * seconds, easing over the last second), with a fixed step, so an
    RP_FPS preview follows the same trajectories. The ball starts at Lorenz time 0;
    the first frame is at t_start, when it is still far below a pixel.

    Each point keeps a colour from where it started: its rank along the ball's longest
    axis 1.5 s (of video) before the first frame, when it is already a sub-pixel
    needle, through PALETTE, so the filament runs violet to gold and its folds
    interleave the bands. Frames: a perspective camera
    orbiting about the vertical axis, every point splatted bilinearly after every
    integration step at a fixed random fraction along that step (continuous motion
    blur) into a trail buffer that decays per step with a `trail` half-life in video
    seconds (long while the ball is a spark, short once it has spread, so the full
    attractor stays crisp), plus a head layer of the points' current positions, tone-mapped (1 - exp(-light)) with a
    wide bloom of the linear light, so the packed spark burns white. One-shot; the last
    0.4 s fade out. Stateful, so frames render in order on the GPU.

    Sound (see `soundtrack`): each frame also records the cloud's spread and the x of
    the reference orbit and of a few points (diagnostics.npy, reused by --audio-only
    runs). The key comes from settings.audio_rng; the posted soundtrack uses --seed 0
    (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    knobs = KNOBS
    rng = settings.rng
    device = select_device()
    f32 = dict(device=device, dtype=torch.float32)

    t_end = lorenz_time(1.0, knobs)
    centre = reference(t_end)
    logger.info(f"Reference orbit: {len(centre)} half steps to Lorenz time {t_end:.1f}")

    direction = rng.standard_normal((POINTS, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    offsets = direction * rng.random((POINTS, 1)) ** (1 / 3) * knobs["radius"]
    phase = torch.tensor(rng.random(POINTS), **f32)  # where along a step to splat
    lut = torch.tensor(palette_lut(PALETTE, 256), **f32)
    colours = None  # set once the ball has stretched (see `stretch_colours`)
    d = torch.tensor(offsets, **f32)
    centre_t = torch.tensor(centre, **f32)  # positions = centre + d, near 1e-6 precise
    steps_done = 0

    def trail_decay(t):
        """Per-step trail decay at video time t (the same at every RP_FPS)."""
        a, b = knobs["trail"]
        u = float(np.clip((t - 0.3) / 0.4, 0, 1))
        half_life = a + (b - a) * u * u * (3 - 2 * u)
        return 0.5 ** (DT / (knobs["speed"] * half_life))

    canvas = torch.zeros(HEIGHT, WIDTH, 3, **f32)
    head = torch.zeros_like(canvas)
    targets = [
        round(lorenz_time(f / (FRAME_COUNT - 1), knobs) / DT)
        for f in range(FRAME_COUNT)
    ]
    logger.info(f"{targets[-1]} offset steps for {FRAME_COUNT} frames")
    fade_frames = 0.4 * FPS

    def project(t):
        """Camera for video time t: screen x, y of every point."""
        p = centre_t[2 * steps_done] + d
        p = p - torch.tensor([0.0, 0.0, RHO - 1.0], **f32)
        ease = t * t * (3 - 2 * t)
        yaw = np.radians(knobs["yaw"][0] + (knobs["yaw"][1] - knobs["yaw"][0]) * ease)
        pitch = np.radians(knobs["pitch"])
        x = p[:, 0] * np.cos(yaw) - p[:, 1] * np.sin(yaw)
        depth = p[:, 0] * np.sin(yaw) + p[:, 1] * np.cos(yaw)
        z = p[:, 2] * np.cos(pitch) - depth * np.sin(pitch)
        depth = depth * np.cos(pitch) + p[:, 2] * np.sin(pitch)
        distance = 110.0
        s = distance * knobs["scale"] / (distance + depth)
        return WIDTH / 2 + x * s, HEIGHT / 2 - z * s

    def stretch_colours():
        """Colour by rank along the ball's longest axis: the filament runs violet->gold."""
        nonlocal colours
        centred = d - d.mean(0)
        axis = torch.linalg.eigh(centred.T @ centred)[1][:, -1]
        rank = torch.argsort(torch.argsort(centred @ axis)).float() / (POINTS - 1)
        colours = lut[(rank * 255).long()]

    def advance(steps, t, draw=True):
        """Integrate up to `steps`; after every step, decay the trails and splat each
        point at its own fixed fraction along the step (continuous motion blur)."""
        nonlocal d, steps_done
        x0, y0 = project(t)
        decay = trail_decay(t)
        light = colours * (knobs["weight"] * (1 - decay)) if draw else None
        while steps_done < steps:
            i = 2 * steps_done  # half-step index of this step's start
            c0, c1, c2 = centre[i], centre[i + 1], centre[i + 2]
            k1 = offset_rate(d, c0)
            k2 = offset_rate(d + DT / 2 * k1, c1)
            k3 = offset_rate(d + DT / 2 * k2, c1)
            k4 = offset_rate(d + DT * k3, c2)
            d = d + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            steps_done += 1
            if draw:
                x1, y1 = project(t)
                canvas.mul_(decay)
                splat(canvas, x0 + phase * (x1 - x0), y0 + phase * (y1 - y0), light)
                x0, y0 = x1, y1

    # Stop 1.5 s (of video) short of the first frame to read the colour axis; frame 0
    # then draws those steps, so the spark arrives with its tail.
    warm = targets[0] - round(1.5 * knobs["speed"] / DT)
    advance(warm, 0.0, draw=False)
    stretch_colours()

    diagnostics = np.zeros((FRAME_COUNT, 2 + WATCHED))

    def measure(frame):
        """Cloud spread, and x of the reference orbit and of WATCHED points."""
        spread = (d - d.mean(0)).norm(dim=1).median()
        x = d[:WATCHED, 0] + centre_t[2 * steps_done, 0]
        diagnostics[frame, 0] = float(spread)
        diagnostics[frame, 1] = centre[2 * steps_done, 0]
        diagnostics[frame, 2:] = x.cpu().numpy()

    def render(frame):
        t = frame / (FRAME_COUNT - 1)
        advance(targets[frame], t)
        measure(frame)
        head.zero_()
        splat(head, *project(t), colours * knobs["head"])
        total = canvas + head
        image = 1 - torch.exp(-total)
        # The packed spark's light (clamped) blooms into a halo; the spread cloud
        # only gets a faint haze.
        glow = 0.25 * head.clamp(max=2500.0) + 0.02 * total
        image = image + 0.45 * wide_blur(glow, 10.0, 4)
        fade = min(1.0, (FRAME_COUNT - 1 - frame) / fade_frames)
        return image.clamp(0, 1) * (255.0 * fade)

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    saved = settings.output_path / settings.filename / "diagnostics.npy"
    if settings.audio_only and saved.exists():
        diagnostics = np.load(saved)
    elif settings.audio_only:  # frames from before the soundtrack: re-run, no drawing
        for frame in range(FRAME_COUNT):
            advance(targets[frame], 0.0, draw=False)
            measure(frame)
    np.save(saved, diagnostics)
    track = soundtrack(diagnostics, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
