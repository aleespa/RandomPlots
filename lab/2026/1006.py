"""
Tidal Harmonics

Nine harmonic tides circle a dark centre, each one a ribbon of fine threads.
As a ribbon travels it twists: where it turns edge-on its threads pinch into a
bright knot, where it opens they fan out into silk. Every tide turns at its own
pace, so their crests drift into step and out again.

Faint ripples carry the same five-fold rhythm outward and dissolve into the
dark, like tide lines spreading across still water.

Whenever two neighbouring tides fall into step, a marimba bar is struck: the
inner tides ring high, the outer ones low and a little louder. Each ripple
that leaves the weave sounds a deep bass note, five to every loop.

Every rhythm completes a whole number of turns, so the weave and its music
return exactly to where they began. Follow one strand and the motion is
simple; watch them all and new shapes appear at every crossing.

r(θ, t) = r₀ + a·cos(5θ + φ₁) + b·cos(2θ − φ₂) + c·cos(kθ + φ₃)

Made with Python, PyTorch & NumPy. Sound synthesized in NumPy.
#generativeart #harmonics #interference #parametricart #waves #geometry
#mathematics #randomplots #sonification #creativecoding #algorithmicart
#mathisbeautiful
"""

from __future__ import annotations

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds, one loop
FRAME_COUNT = frame_count(DURATION)
CX, CY = WIDTH / 2.0, HEIGHT / 2.0
SCALE = 0.54 * WIDTH  # pixels per unit radius

# --- Weave -------------------------------------------------------------------
STRANDS = 9
SAMPLES = 8192  # per strand: under half a pixel apart
SPEEDS = (1, 2, 3, 2, 1, 3, 2, 1, 2)  # whole turns per loop
FILAMENTS = 9  # threads per ribbon
RIBBON = 0.026  # ribbon half-width, units of radius
TWISTS = (2, 3, 4)  # twists around each strand, cycled across strands
TWIST_SPEEDS = (1, -1, 2, -2)  # whole cycles per loop, cycled across strands
FILAMENT_ENERGY = 0.36
ECHOES = 6  # faded copies at earlier times: a short motion trail
ECHO_STEP = 0.004  # loop fraction between echoes
ECHO_ENERGY = 0.20

# --- Ripples and motes -------------------------------------------------------
RIPPLES = 5
RIPPLE_CYCLES = 1  # each ripple crosses once per loop
RIPPLE_SAMPLES = 16384
RIPPLE_RANGE = (0.70, 1.98)  # birth and death radius
RIPPLE_LINES = ((0.0, 1.0), (0.022, 0.5), (0.044, 0.25))  # (inset, weight)
RIPPLE_ENERGY = 0.55
MOTES = 900
MOTE_ENERGY = 2.4

# --- Light -------------------------------------------------------------------
LINE_PALETTE = ["#7652d5", "#24c7c9", "#fff0bd", "#ff6c9b", "#7652d5"]  # cyclic
BACKGROUND_EDGE = "#05040c"
BACKGROUND_CENTRE = "#120b2e"
LUT_SIZE = 512
BLOOMS = ((2.0, 1, 0.55), (9.0, 2, 0.45), (32.0, 4, 0.35), (110.0, 8, 0.30))
CORE = 0.10  # soft glow in the centre of the rosette
VIGNETTE = 0.30

# --- Sound -------------------------------------------------------------------
PENTATONIC = (0, 2, 4, 7, 9)  # major pentatonic, semitones above the root
ROOTS = range(57, 65)  # root (MIDI), chosen by the seed
BASS_LINE = (0, 9, 7, 4, 2)  # semitones above root - 24, one per ripple
RIPPLE_SOUND_RADIUS = 0.9  # a ripple sounds as it clears the weave
REVERB_RT60 = 2.8
REVERB_WET = 0.35


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def soundtrack(phase, orientation, strand_phases, rng):
    """
    A soft marimba struck whenever neighbouring strands' crests fall into step.

    Ignoring the small 2θ and kθ terms, strand s has its five crests at the
    screen angles α with 5α ≡ ψ_s(t) = 5·orientation − φ_s − phase +
    2π·4·v_s·t, so strands s and s + 1 line up 4·|v_s − v_{s+1}| times per
    loop, at times found in closed form. Each alignment strikes a bar pitched
    by the pair's radius (inner pairs higher, pentatonic over a seeded root),
    louder for the wider outer strands and panned by cos ψ, the way the crests
    point. A bass bar steps through BASS_LINE as each ripple clears the weave.
    Notes wrap around the end and the reverb is circular, so the sound loops.
    """
    root = int(rng.choice(ROOTS))
    track = audio.silence(DURATION)

    def note(degree):
        return root + 12 * (degree // 5) + PENTATONIC[degree % 5]

    pairs = STRANDS - 1
    struck = 0
    for s in range(pairs):
        rate = 4 * (SPEEDS[s] - SPEEDS[s + 1])  # ψ_s − ψ_{s+1} turns per loop
        gap = (strand_phases[s + 1] - strand_phases[s]) / (2.0 * np.pi)
        for m in range(abs(rate)):
            t = ((m - gap) / rate) % 1.0
            crest = (
                5.0 * orientation
                - strand_phases[s]
                - phase
                + 2.0 * np.pi * 4 * SPEEDS[s] * t
            )
            velocity = 0.3 + 0.35 * s / (pairs - 1)
            bar = audio.marimba(audio.midi_to_hz(note(pairs - 1 - s)), velocity)
            audio.add(track, audio.pan(bar, 0.6 * np.cos(crest)), t * DURATION)
            struck += 1

    start, end = RIPPLE_RANGE
    clear = (RIPPLE_SOUND_RADIUS - start) / (end - start)  # ripple progress
    times = sorted(
        ((clear - k / RIPPLES) % 1.0 + j) / RIPPLE_CYCLES
        for k in range(RIPPLES)
        for j in range(RIPPLE_CYCLES)
    )
    for i, t in enumerate(times):
        bass = root - 24 + BASS_LINE[i % len(BASS_LINE)]
        bar = audio.marimba(audio.midi_to_hz(bass), 0.55, length=4.0)
        audio.add(track, audio.pan(bar, 0.0), t * DURATION)

    logger.info(
        f"Soundtrack: {struck} crest notes, {len(times)} ripple bass notes, "
        f"root MIDI {root}"
    )
    track = audio.reverb(track, rng, rt60=REVERB_RT60, wet=REVERB_WET)
    return audio.normalize(track)


def generate(settings: ImageProcessingSettings = None):
    """Nine twisted ribbons of harmonic tides, with ripples, motes and bloom.

    Each strand s is the closed curve r(θ, t) = r₀(s) + a(s)·cos(5θ + φ₁) +
    0.035·cos(2θ − φ₂) + 0.014·cos(k(s)θ + φ₃), rotated by 2π·v(s)·t, where
    the phases advance by 2π·t times the integer speeds v, v − 1 and v + 1.
    Each strand is drawn as a ribbon of FILAMENTS threads offset along the
    curve's normal by u·w(θ, t), u in [−1, 1], with w a cosine of a few
    twists per turn that travels at an integer rate: where w crosses zero the
    threads cross, and the additive splatting makes the pinch a bright knot.
    Six fainter copies of each centre line at slightly earlier times form a
    short motion trail.

    Outside the weave, RIPPLES rings of the same five-fold harmonic expand
    from 0.7 to 1.98 units once per loop (frac(t + k/RIPPLES)), each a crest
    and two smaller trailing crests, fading in and out; motes ride outward on
    the same clock. All periodic terms complete whole cycles, so the loop is
    exact.

    Rendered on the GPU: samples are splatted bilinearly with energy
    proportional to the screen length they carry, colour cycles through a
    closed palette along each strand, then four-scale bloom, a central glow,
    a vignette and an exponential tone map that lets crossings burn white.

    The soundtrack (see `soundtrack`) is computed from the same phases and
    speeds and muxed into the MP4.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    def tensor(values) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

    # Every random choice is made here, once.
    phase = rng.uniform(0.0, 2.0 * np.pi)
    orientation = rng.uniform(0.0, 2.0 * np.pi)
    strand_phases = rng.uniform(0.0, 2.0 * np.pi, STRANDS)
    strand_phase = tensor(strand_phases)[:, None]
    twist_phase = tensor(rng.uniform(0.0, 2.0 * np.pi, STRANDS))[:, None]
    ripple_phase = rng.uniform(0.0, 2.0 * np.pi)
    mote_angle = tensor(rng.uniform(0.0, 2.0 * np.pi, MOTES))
    mote_offset = tensor(rng.uniform(0.0, 1.0, MOTES))
    mote_birth = tensor(rng.uniform(0.35, 0.95, MOTES))
    mote_brightness = tensor(rng.uniform(0.15, 1.0, MOTES) ** 2)
    mote_hue = tensor(rng.uniform(0.0, 1.0, MOTES))
    mote_swirl = tensor(rng.uniform(-0.4, 0.4, MOTES))

    lut = tensor(palette_lut(LINE_PALETTE, LUT_SIZE))

    def colour(position: torch.Tensor) -> torch.Tensor:
        return lut[((position % 1.0) * (LUT_SIZE - 1)).long()]

    # Weave geometry that doesn't depend on t.
    theta = torch.arange(SAMPLES, device=device) * (2.0 * np.pi / SAMPLES)
    strand = torch.arange(STRANDS, device=device, dtype=torch.float32)[:, None]
    base_radius = 0.29 + 0.052 * strand
    amplitude = 0.060 + 0.008 * strand
    high_harmonic = 8.0 + strand % 3
    speeds = tensor(SPEEDS)[:, None]
    twists = tensor([TWISTS[s % len(TWISTS)] for s in range(STRANDS)])[:, None]
    twist_speeds = tensor(
        [TWIST_SPEEDS[s % len(TWIST_SPEEDS)] for s in range(STRANDS)]
    )[:, None]
    u = torch.linspace(-1.0, 1.0, FILAMENTS, device=device)
    filament_weight = (0.45 + 0.55 * (1.0 - u**2))[None, :, None]
    strand_hue = theta / (2.0 * np.pi) + strand * 0.11  # (STRANDS, SAMPLES)

    ripple_theta = torch.arange(RIPPLE_SAMPLES, device=device) * (
        2.0 * np.pi / RIPPLE_SAMPLES
    )
    ripple_offset = torch.arange(RIPPLES, device=device)[:, None, None] / RIPPLES
    ripple_inset = tensor([inset for inset, _ in RIPPLE_LINES])[None, :, None]
    ripple_weight = tensor([weight for _, weight in RIPPLE_LINES])[None, :, None]

    # Background, vignette and central glow, built once.
    yy, xx = torch.meshgrid(
        torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5,
        torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    distance = torch.hypot(xx - CX, yy - CY) / SCALE
    glow = torch.exp(-((distance / 1.05) ** 2))[..., None]
    edge, centre = tensor(_rgb(BACKGROUND_EDGE)), tensor(_rgb(BACKGROUND_CENTRE))
    background = edge + (centre - edge) * glow
    background = -torch.log1p(-background)  # undo the tone map below
    core = torch.exp(-((distance / 0.24) ** 2))[..., None]
    vignette = (1.0 - VIGNETTE * (((xx - CX) / CX) ** 2 + ((yy - CY) / CY) ** 2) / 2.0)[
        ..., None
    ]
    first_bloom = gaussian_kernel(BLOOMS[0][0], device)

    lines = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    haze = torch.zeros(HEIGHT, WIDTH, 3, device=device)

    def weave(t):
        """Centre lines of every strand at time(s) t, in pixels."""
        time_phase = 2.0 * np.pi * t
        phase_a = strand_phase + phase + time_phase * speeds
        phase_b = 0.7 * strand_phase - phase + time_phase * (speeds - 1.0)
        phase_c = 1.3 * strand_phase + time_phase * (speeds + 1.0)
        radius = (
            base_radius
            + amplitude * torch.cos(5.0 * theta + phase_a)
            + 0.035 * torch.cos(2.0 * theta - phase_b)
            + 0.014 * torch.cos(high_harmonic * theta + phase_c)
        )
        angle = theta + orientation + time_phase * speeds
        x = CX + SCALE * radius * torch.cos(angle)
        y = CY - SCALE * radius * torch.sin(angle)
        crest = 0.5 + 0.5 * torch.cos(5.0 * theta + phase_a)
        return x, y, crest

    def tangent(x, y):
        """Central differences along each closed curve and the arc per sample."""
        dx = x.roll(-1, -1) - x.roll(1, -1)
        dy = y.roll(-1, -1) - y.roll(1, -1)
        step = torch.hypot(dx, dy).clamp_min(1e-6)
        return dx / step, dy / step, 0.5 * step

    def deposit(canvas, x, y, rgb, energy):
        splat(canvas, x, y, (rgb * energy[..., None]).reshape(-1, 3))

    def render(frame):
        t = frame / FRAME_COUNT
        lines.zero_()
        haze.zero_()

        # Ribbons: threads offset along the normal by a travelling twist.
        x, y, crest = weave(t)
        tx, ty, ds = tangent(x, y)
        width = (
            RIBBON
            * SCALE
            * torch.cos(twists * theta + 2.0 * np.pi * twist_speeds * t + twist_phase)
        )
        offset = u[None, :, None] * width[:, None, :]  # (STRANDS, FILAMENTS, S)
        fx = x[:, None, :] - offset * ty[:, None, :]
        fy = y[:, None, :] + offset * tx[:, None, :]
        energy = (
            FILAMENT_ENERGY
            * filament_weight
            * (ds * (0.55 + 0.9 * crest**2))[:, None, :]
        )
        rgb = colour(strand_hue - t)[:, None].expand(-1, FILAMENTS, -1, -1)
        deposit(lines, fx, fy, rgb, energy)

        # Motion trail: the centre lines a moment ago, softened.
        echo_t = t - ECHO_STEP * torch.arange(1, ECHOES + 1, device=device)
        ex, ey, _ = weave(echo_t[:, None, None])  # (ECHOES, STRANDS, S)
        _, _, eds = tangent(ex, ey)
        fade = 1.0 - torch.arange(1, ECHOES + 1, device=device) / (ECHOES + 1)
        energy = ECHO_ENERGY * eds * fade[:, None, None]
        rgb = colour(strand_hue[None] - echo_t[:, None, None])
        deposit(haze, ex, ey, rgb, energy)

        # Ripples: expanding five-fold rings, each a crest and two smaller ones.
        progress = (RIPPLE_CYCLES * t + ripple_offset) % 1.0  # (RIPPLES, 1, 1)
        start, end = RIPPLE_RANGE
        lobes = (
            0.05
            * (1.0 - 0.7 * progress)
            * torch.cos(5.0 * ripple_theta + ripple_phase + 2.0 * np.pi * t)
        )
        radius = (start + (end - start) * progress - ripple_inset) * (1.0 + lobes)
        angle = ripple_theta + orientation
        rx = CX + SCALE * radius * torch.cos(angle)
        ry = CY - SCALE * radius * torch.sin(angle)
        _, _, rds = tangent(rx, ry)
        envelope = torch.sin(np.pi * progress) ** 2 * (1.0 - 0.5 * progress)
        energy = RIPPLE_ENERGY * ripple_weight * envelope * rds
        rgb = colour(ripple_theta / (2.0 * np.pi) - t + 0.5 * progress)
        deposit(lines, rx, ry, rgb.expand(RIPPLES, len(RIPPLE_LINES), -1, -1), energy)

        # Motes ride outward on the ripples' clock.
        progress = (RIPPLE_CYCLES * t + mote_offset) % 1.0
        radius = mote_birth + (end - mote_birth) * progress
        angle = mote_angle + mote_swirl * progress
        mx = CX + SCALE * radius * torch.cos(angle)
        my = CY - SCALE * radius * torch.sin(angle)
        energy = MOTE_ENERGY * mote_brightness * torch.sin(np.pi * progress) ** 2
        deposit(lines, mx, my, colour(mote_hue - t), energy)

        source = lines + haze
        light = background + lines + blur(haze, first_bloom)
        for sigma, factor, strength in BLOOMS:
            if factor == 1:
                light = light + strength * blur(source, first_bloom)
            else:
                light = light + strength * wide_blur(source, sigma, factor)
        pulse = 1.0 + 0.25 * np.sin(4.0 * np.pi * t)
        light = light + CORE * pulse * core * colour(tensor(0.25 - t))
        light = light * vignette
        white = 0.35 * (light.mean(-1, keepdim=True) - 0.8).clamp_min(0.0)
        return 255.0 * (1.0 - torch.exp(-(light + white)))

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    track = soundtrack(phase, orientation, strand_phases, rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
