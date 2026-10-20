"""
The Sound Barrier

A point of light sits still, sending out ripples like a stone dropped in a
pond. Then it starts to move. The ripples crowd together in front of it and
spread apart behind. At the speed of sound they can't get out of its way any
more: they pile up into one bright wall. It punches through, and from then on
it drags a sharp V of light behind it.

Every ring is a sound wave, spreading from wherever the source was when it
made it. Faster than sound, the source outruns its own waves, and they can
only meet along the edges of a cone, narrower the faster it flies (at twice
the speed of sound, 30 degrees each side). That cone is the shock wave you
hear as a sonic boom. Colour shows the Doppler shift: blue where the waves
are squeezed, red where they are stretched.

Listen from the middle of the frame. The hum is what you would really
hear there: higher as the source rushes toward you, rising to a scream
as it nears the speed of sound, then the boom as the cone sweeps over,
and right after it the sound it made a moment earlier, arriving
backwards. Underneath, a note for every tenth of the speed of sound.

Made with Python & PyTorch. Sound synthesized in NumPy.
#sonicboom #doppler #physics #mathart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from scipy.signal import butter, sosfilt

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- Physics (pixels and seconds; y measured up from the bottom of the frame) -------
SOUND = 110.0  # px/s
START_Y = 250.0  # where the source rests
RAMP = (
    (0.5, 5.5, 0.0, 1.0),
    (6.5, 11.5, 1.0, 2.0),
)  # (from s, to s, Mach, Mach), eased
RATE = 4.0  # wavefronts per second
HISTORY = 5.0  # s of rings already out at the start, emitted at rest

# --- Look ---------------------------------------------------------------------------
BACKGROUND = "#040509"
RED, NEUTRAL, BLUE = "#ff4a3d", "#ffe9c9", "#5ec8ff"
DOPPLER_SPAN = 2.0  # a factor of this (or its inverse) is full blue (red)
SPACING = 0.7  # px between splatted points along a ring
FADE = 900.0  # px: rings dim as 1 / sqrt(1 + r / FADE)
EXPOSURE, GAMMA = 1.4, 0.6
SOFT = 0.8
GLOWS = ((10.0, 2, 0.1), (40.0, 8, 0.06))  # (sigma px, shrink, gain) on linear light
SOURCE_LIGHT, SOURCE_SIZE = 60.0, 3.0  # the source's own glow

# --- Sound -------------------------------------------------------------------------
STYLE = "lofi_keys"  # common.music.STYLES; an --audio-only tag naming one overrides it
LISTENER = (30.0, 850.0)  # px: just right of the flight path, below mid-frame
HUM_PARTIALS = 6
OVERSAMPLE = 4  # emission-time samples per output sample
MACH_STEP = 0.1  # a note each time the source gains this much Mach


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def mach(seconds):
    """Mach number: at rest, eased up to 1 (pausing there), eased up to 2, held."""
    s = np.asarray(seconds, float)
    m = np.zeros_like(s)
    for t0, t1, m0, m1 in RAMP:
        inside = s >= t0
        m = np.where(inside, m0 + (m1 - m0) * smoothstep((s - t0) / (t1 - t0)), m)
    return m


def height(seconds):
    """The source's height: START_Y plus SOUND times the integral of mach."""
    s = np.asarray(seconds, float)
    grid = np.linspace(0, DURATION + 1, 40001)
    travelled = np.concatenate(
        [[0.0], np.cumsum(0.5 * (mach(grid[1:]) + mach(grid[:-1])) * np.diff(grid))]
    )
    return START_Y + SOUND * np.interp(s, grid, travelled)


def heard(frequency: float) -> tuple:
    """
    The source's hum as it reaches LISTENER, and the boom time.

    The source emits s(t_e), a hum of HUM_PARTIALS harmonics of `frequency`.
    Emission t_e (sampled OVERSAMPLE times per output sample, from HISTORY
    before the clip) arrives at t_a = t_e + |LISTENER - source(t_e)| / SOUND,
    and is added there, quieter as 1 / sqrt(1 + d / 100). Binning by arrival
    time does the physics by itself: compressed arrivals (approach) pile up
    higher and louder, stretched ones (recession) lower and softer, and where
    dt_a/dt_e turns negative the sound arrives in reverse. The first
    supersonic arrival, the earliest t_a on that branch, is the boom.
    """
    sr = audio.SAMPLE_RATE
    te = np.arange(-HISTORY, DURATION, 1 / (sr * OVERSAMPLE))
    distance = np.hypot(LISTENER[0], LISTENER[1] - height(te))
    ta = te + distance / SOUND
    hum = sum(
        np.sin(2 * np.pi * k * frequency * te) / k**1.2
        for k in range(1, HUM_PARTIALS + 1)
    )
    weight = hum / np.sqrt(1 + distance / 100) / OVERSAMPLE
    received = np.zeros(int(DURATION * sr) + 2)
    position = ta * sr
    inside = (position >= 0) & (position < len(received) - 1)
    lower = np.floor(position[inside]).astype(np.int64)
    frac = position[inside] - lower
    np.add.at(received, lower, weight[inside] * (1 - frac))
    np.add.at(received, lower + 1, weight[inside] * frac)
    backwards = np.flatnonzero(np.diff(ta) < 0)
    boom = float(ta[backwards].min()) if len(backwards) else None
    return received[: int(DURATION * sr)], boom


def soundtrack(style_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    A one-shot: the physics, with a little music on top.

    - The hum as heard at LISTENER (`heard`), low-passed against aliasing of
      the near-infinite blueshift and soft-clipped so the pile-up screams
      without drowning everything else.
    - The boom at the first supersonic arrival: an N-wave (a sharp jump up,
      a 30 ms fall, a sharp jump back), a low thump, a long room.
    - A speedometer: the style's voice strikes one scale degree higher each
      time the source gains MACH_STEP (the notes stop while it holds at
      Mach 1), and the home chord rings out after the boom. A soft pad on
      the root fills the first second, before the hum arrives.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    sr = audio.SAMPLE_RATE
    kw = dict(circular=False)
    track = audio.silence(DURATION)

    frequency = scale.hz(0, -2)
    while frequency < 90:  # phone speakers: keep the hum audible
        frequency *= 2
    hum, boom = heard(frequency)
    hum = sosfilt(butter(4, 5000.0, fs=sr, output="sos"), hum)
    level = np.percentile(np.abs(hum[: int(5 * sr)][hum[: int(5 * sr)] != 0]), 99)
    hum = 0.3 * np.tanh(hum / (2 * level))
    track += hum[:, None] * np.array([0.5, 0.5]) ** 0.5

    if boom is not None:
        n = int(0.03 * sr)
        wave = np.concatenate([np.linspace(1, -1, n), np.zeros(int(0.3 * sr))])
        wave = sosfilt(butter(2, 2500.0, fs=sr, output="sos"), wave)
        audio.add(track, 0.9 * wave, boom, **kw)
        audio.add(track, audio.kick(1.0, 45.0), boom, **kw)
        for k, d in enumerate((0, 2, 4, 7)):
            note = style.voice(scale.hz(d), 0.4)
            audio.add(
                track, audio.pan(note, 0.3 * (k - 1.5)), boom + 0.25 + 0.1 * k, **kw
            )
        bass = scale.hz(0, -2)
        bass = 2 * bass if bass < 65 else bass  # phone speakers stop around 60 Hz
        audio.add(track, style.bass(bass, 0.5, length=4.0), boom + 0.25, **kw)

    grid = np.linspace(0, DURATION, 15001)
    speed = mach(grid)
    notes = 0
    for step in range(1, int(round(speed.max() / MACH_STEP)) + 1):
        crossed = np.flatnonzero(speed >= step * MACH_STEP - 1e-9)
        if not len(crossed):
            break
        when = float(grid[crossed[0]])
        note = style.voice(scale.hz(step - 1), 0.28)
        audio.add(track, audio.pan(note, 0.4 * np.sin(step)), when, **kw)
        notes += 1

    for n in scale.chord(0, 3, octave=-1):
        hz = float(audio.midi_to_hz(n))
        audio.add(track, audio.pad(hz, 1.5, 0.12, 0.05, 1.5, 0.3), 0.0, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} speed notes, "
        + (f"boom at {boom:.2f} s" if boom is not None else "no boom")
    )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Wavefronts of a source accelerating through Mach 1, in closed form.

    The source moves straight up the middle of the frame at speed
    SOUND * mach(t) (eased from rest to Mach 1, held there for a second while
    the fronts pile up, then eased to Mach 2).
    It emits a wavefront every 1 / RATE s, the first HISTORY s before the
    clip, at rest. The front emitted at t_e is the circle of radius
    SOUND (t - t_e) about the source's position at t_e. Each point of it is
    coloured by its Doppler factor 1 / (1 - M_e cos phi), phi the angle from
    the source's velocity at emission (factors beyond DOPPLER_SPAN saturate, and
    so do the ahead-of-the-source arcs of supersonic fronts, where the factor
    turns negative). Fronts dim as 1 / sqrt(1 + r / FADE), and brighten as the
    square root of the Doppler factor.

    A frame is a pure function of t: on the GPU every visible front is sampled
    every SPACING px and splatted additively, so where fronts overlap (the
    pile-up at Mach 1, the Mach cone with sin mu = 1 / M after) the light adds
    up into bright lines by itself. Tone-mapped as in 1027 (linear-light glow
    for concentrated light), plus a glowing dot for the source. Nothing is
    random: --seed does not change the picture. The posted soundtrack is
    lofi_keys with --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    to = dict(device=device, dtype=torch.float32)
    emitted = np.arange(-HISTORY * RATE, DURATION * RATE + 1) / RATE
    where = height(emitted)
    speed = mach(emitted)
    logger.info(
        f"{len(emitted)} wavefronts; Mach 1 at {RAMP[0][1]:.1f} s, height there "
        f"{float(height(RAMP[0][1])):.0f} px; source leaves the frame at "
        f"{float(np.interp(HEIGHT + 20, height(np.linspace(0, DURATION, 3001)), np.linspace(0, DURATION, 3001))):.1f} s"
    )

    red, neutral, blue = (torch.tensor(_rgb(c), **to) for c in (RED, NEUTRAL, BLUE))
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **to)
    soft = gaussian_kernel(SOFT, device)
    background = torch.tensor(_rgb(BACKGROUND), **to)

    def render(frame: int) -> torch.Tensor:
        seconds = frame / FPS
        canvas.zero_()
        live = emitted <= seconds
        radius = SOUND * (seconds - emitted[live])
        cy = where[live]
        m = speed[live]
        # Skip fronts that miss the frame: too small to reach it, or enclosing it.
        screen_y = HEIGHT - cy
        nearest = np.maximum(0, np.maximum(-screen_y, screen_y - HEIGHT))
        farthest = np.hypot(
            WIDTH / 2, np.maximum(np.abs(screen_y), np.abs(HEIGHT - screen_y))
        )
        keep = (radius > 0.5) & (radius > nearest - 2) & (radius < farthest + 2)
        radius, cy, m = radius[keep], cy[keep], m[keep]
        if len(radius):
            counts = np.maximum(16, np.ceil(2 * np.pi * radius / SPACING)).astype(
                np.int64
            )
            r = torch.tensor(radius, **to)
            ring = torch.repeat_interleave(
                torch.arange(len(radius), device=device),
                torch.tensor(counts, device=device),
            )
            starts = torch.tensor(
                np.concatenate([[0], np.cumsum(counts)[:-1]]), device=device
            )
            local = torch.arange(int(counts.sum()), device=device) - starts[ring]
            phi = 2 * np.pi * (local.float() + 0.5) / torch.tensor(counts, **to)[ring]
            rr = r[ring]
            x = WIDTH / 2 + rr * torch.sin(phi)
            y = HEIGHT - (torch.tensor(cy, **to)[ring] + rr * torch.cos(phi))
            mm = torch.tensor(m, **to)[ring]
            denominator = 1 - mm * torch.cos(phi)
            factor = torch.where(
                denominator > 1 / DOPPLER_SPAN,
                1 / denominator.clamp_min(1e-6),
                DOPPLER_SPAN,
            )
            u = (torch.log(factor) / np.log(DOPPLER_SPAN)).clamp(-1, 1)[:, None]
            colour = torch.where(
                u > 0, neutral + u * (blue - neutral), neutral - u * (red - neutral)
            )
            amp = SPACING * torch.sqrt(
                factor / (1 + rr / FADE)
            )  # squeezed fronts brighter
            splat(canvas, x, y, colour * amp[:, None])

        # The source.
        sy = HEIGHT - float(height(seconds))
        offsets = torch.linspace(-3 * SOURCE_SIZE, 3 * SOURCE_SIZE, 13, **to)
        gx, gy = torch.meshgrid(offsets, offsets, indexing="xy")
        weight = torch.exp(-(gx**2 + gy**2) / (2 * SOURCE_SIZE**2)).reshape(-1)
        weight = SOURCE_LIGHT * weight / weight.sum()
        splat(
            canvas,
            WIDTH / 2 + gx.reshape(-1),
            sy + gy.reshape(-1),
            weight[:, None] * neutral,
        )

        density = canvas.clamp_min(0) ** GAMMA
        glow = sum(g * wide_blur(canvas, sigma, f) for sigma, f, g in GLOWS)
        light = 1 - torch.exp(-EXPOSURE * (blur(density, soft) + glow))
        return (background + light).clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
