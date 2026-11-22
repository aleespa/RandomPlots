"""
The Lines at the Bottom of a Pool

Why does sunlight draw a net of bright, wobbling lines on the bottom of a
swimming pool? The water barely moves, yet the light gathers into lines many
times brighter than the floor around them.

Every ripple is a weak lens. Where the surface bulges, it bends the rays
together; where it dips, it spreads them apart. Deep enough down, neighbouring
rays cross, and where they cross the light piles up. These folds of light are
called caustics.

The top panel cuts through the pool from the side: watch the rays bend and
fold into bright points on the floor. Below, the floor from above, where every
fold becomes one of the lines.

Each time a fold lands on the floor in the side view, a piano note sounds, low
on the left, high on the right.

Made with Python & PyTorch. Sound synthesized in NumPy.
#caustics #optics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat
from common.image_processing import ImageProcessingSettings
from common.music import PROGRESSIONS, STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    latex_mask,
    palette_lut,
    render_frames,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Why do pool floors have these lines?'
HOOK_Y = 300  # placed by hand
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds; every wave completes a whole number of periods
FRAME_COUNT = frame_count(DURATION)

# --- The waves ------------------------------------------------------------------------
WAVES = 18  # plane-wave components of the surface
CYCLES = (5, 15)  # each wave's whole periods per loop, drawn from this range
SPEED = 125.0  # px/s: shallow water, every wave travels at the same speed
STRENGTH = 1.1  # rms of the focusing (d Delta_x / dx); folds appear where it reaches -1

# --- Layout ---------------------------------------------------------------------------
SIDE_TOP, SIDE_BOTTOM = 150, 662  # px: the side-view panel
SURFACE_Y, FLOOR_Y = 250, 630  # px: still water level and the floor, side view
SURFACE_LIFT = (
    20.0  # px: drawn wave height (exaggerated; the optics use the true slope)
)
FLOOR_TOP = 700  # px: the top-down view of the floor starts here
SLICE_Y = 1310  # px: the floor row the side view cuts through
PAD = 170  # px: photons start this far outside the floor so its edges fill in
PHOTON_STEP = 0.5  # px between photons (before jitter)
RAY_STEP = 9.0  # px between side-view rays
TILE = 135  # px: pool tiles on the floor

# --- Look -----------------------------------------------------------------------------
BACKGROUND = "#03070d"
WATER = "#06182a"  # side-view water tint
SUN = "#ffcf7a"  # light above the surface
LIGHT = ["#03070d", "#0a2a40", "#136f8a", "#4fd0e0", "#d8f8ff", "#ffffff"]
FLOOR_GAIN = 0.20  # tone map: brightness = 1 - exp(-gain * light)
SIDE_GAIN = 0.55  # the drawn rays
GLOW_GAIN = 0.075  # the dense rays: light in the water
LINE_ALPHA = 0.5
STRIP = 8  # px: the floor's light under the side view
BLOOM = 0.55
TEXT = "#cfe9f0"

# --- Sound ----------------------------------------------------------------------------
STYLE = "felt_piano"  # common.music.STYLES: a piano note for every caustic that lands
PROGRESSION = "drift"  # I IV I V under it, one chord per quarter of the loop
SOUND_STEP = 0.004  # s: time grid on which the landings are found
DEGREES = 11  # scale degrees spread across the screen, left (low) to right (high)


def waves(rng):
    """
    Wave vectors k (px⁻¹), angular frequencies ω (rad/s), phases and floor
    displacement amplitudes d (px), drawn once. ω = 2π n / DURATION for whole n,
    so the loop closes; |k| = ω / SPEED (shallow water: one speed for all).
    The amplitudes follow a falling spectrum and are scaled so the rms of the
    focusing term d|k| over all waves is STRENGTH.
    """
    cycles = rng.integers(CYCLES[0], CYCLES[1] + 1, WAVES)
    omega = 2.0 * np.pi * cycles / DURATION
    wavenumber = omega / SPEED
    angle = rng.uniform(0.0, 2.0 * np.pi, WAVES)
    k = wavenumber[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    phase = rng.uniform(0.0, 2.0 * np.pi, WAVES)
    focus = rng.uniform(0.5, 1.0, WAVES) * (cycles / CYCLES[0]) ** -0.6
    focus *= STRENGTH / np.sqrt(0.5 * np.sum(focus**2))
    amplitude = focus / wavenumber  # px of floor displacement
    return k, omega, phase, amplitude


def tiles():
    """Floor albedo: pale tiles with darker grout lines, (H, W) in [0, 1]."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    gx = np.abs(((x - WIDTH / 2) / TILE + 0.5) % 1.0 - 0.5) * TILE
    gy = np.abs(((y - SLICE_Y) / TILE + 0.5) % 1.0 - 0.5) * TILE
    grout = np.exp(-0.5 * (np.minimum(gx, gy) / 1.6) ** 2)
    return 1.0 - 0.45 * grout


def overlay():
    """Static text and marks: (H, W, 3) colour premultiplied by alpha, and alpha."""
    colour = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    alpha = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    text = np.array([int(TEXT[i : i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)

    def put(mask, x, y, strength, anchor="left"):
        blit_mask(colour, mask, x, y, text, strength, anchor)
        blit_mask(alpha, mask, x, y, (1, 1, 1), strength, anchor)

    put(
        latex_mask(r"$n_1 \sin\theta_1 = n_2 \sin\theta_2$", 46),
        WIDTH // 2,
        52,
        0.9,
        "centre",
    )
    put(latex_mask(r"$\mathrm{from\ the\ side}$", 30), 40, SIDE_TOP + 8, 0.6)
    put(
        latex_mask(r"$\mathrm{the\ floor{,}\ from\ above}$", 30),
        40,
        FLOOR_TOP + 22,
        0.6,
    )
    # the slice: two short ticks at the floor's edges and a faint line between them
    alpha[SLICE_Y, :] = 0.10
    colour[SLICE_Y, :] = 0.10 * text
    for x0 in (0, WIDTH - 34):
        alpha[SLICE_Y - 1 : SLICE_Y + 2, x0 : x0 + 34] = 0.8
        colour[SLICE_Y - 1 : SLICE_Y + 2, x0 : x0 + 34] = 0.8 * text
    return colour, alpha


def generate(settings: ImageProcessingSettings = None):
    """
    Pool caustics: sunlight refracted by a rippling water surface, splatted onto
    the floor as photons. A seamless 15 s loop rendered on the GPU.

    Surface: η(p, t) = Σ a_j cos(k_j·p − ω_j t + φ_j), WAVES plane waves with
    ω_j = 2π n_j / DURATION (whole n_j, so the loop closes exactly) and
    |k_j| = ω_j / SPEED (shallow water is non-dispersive). In the paraxial
    limit a vertical ray hitting the surface at p is refracted by Snell's law and
    lands on the floor (depth D) displaced by Δ = D (1 − 1/n) ∇η, so
    Δ(p, t) = −Σ d_j k̂_j sin(k_j·p − ω_j t + φ_j) with d_j = D(1 − 1/n) a_j |k_j|
    in px. The rms focusing Σ d_j|k_j| is set to STRENGTH, enough for the
    Jacobian det(I + ∂Δ/∂p) to vanish along curves: the caustics.

    Floor (bottom panel): a jittered grid of photons every PHOTON_STEP px over
    the floor plus PAD, each moved by Δ and splatted bilinearly; the density
    relative to its mean is the light, tone-mapped 1 − exp(−g·light), lit on
    a tiled albedo, bloomed and coloured through LIGHT.
    Side view (top panel): rays every RAY_STEP px along the floor row SLICE_Y,
    projected onto the vertical plane: from the surface (drawn SURFACE_LIFT px
    high) to x + f·Δ_x at depth fraction f, splatted as samples one row apart,
    so the envelopes (the cusps of the caustic in the water) light up by
    themselves. Everything is a closed form in t, so frames are independent;
    they render in order on the GPU (parallel=False).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()
    f32 = dict(device=device, dtype=torch.float32)

    k, omega, phase, amplitude = waves(rng)
    jitter = rng.uniform(-0.5, 0.5, 2).astype(np.float32)  # sub-grid offset only
    kx, ky = (torch.tensor(k[:, i], **f32) for i in (0, 1))
    om, ph = torch.tensor(omega, **f32), torch.tensor(phase, **f32)
    amp = torch.tensor(amplitude, **f32)
    kn = np.linalg.norm(k, axis=1)
    lift = torch.tensor(amplitude / kn, **f32)  # ∝ surface height
    lift *= SURFACE_LIFT / float(lift.sum()) * 2.2
    ux, uy = kx / torch.tensor(kn, **f32), ky / torch.tensor(kn, **f32)

    # Floor photons: a grid with a fixed, smooth-free random jitter per photon.
    floor_h = HEIGHT - FLOOR_TOP
    gx = torch.arange(-PAD, WIDTH + PAD, PHOTON_STEP, **f32)
    gy = torch.arange(FLOOR_TOP - PAD, HEIGHT + PAD, PHOTON_STEP, **f32)
    px0 = gx[None, :].expand(len(gy), -1).reshape(-1)
    py0 = gy[:, None].expand(-1, len(gx)).reshape(-1)
    gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31)))
    px0 = px0 + PHOTON_STEP * (torch.rand(px0.shape, generator=gen, **f32) - 0.5)
    py0 = py0 + PHOTON_STEP * (torch.rand(py0.shape, generator=gen, **f32) - 0.5)
    ones = torch.ones(len(px0), 1, **f32)
    mean_floor = 1.0 / PHOTON_STEP**2

    # Side-view rays along the slice, and depth fractions one row apart.
    # Dense rays (one per px) light the water; every RAY_STEP-th is also drawn as a line.
    rx = torch.arange(-PAD, WIDTH + PAD, 1.0, **f32) + float(jitter[0])
    drawn = (torch.arange(len(rx), device=device) % int(RAY_STEP)) == 0
    depth = FLOOR_Y - SURFACE_Y
    frac = (torch.arange(0, depth + 40, 1.0, **f32) + 0.5 + float(jitter[1])) / depth
    side_ones = torch.ones(len(rx) * len(frac), 1, **f32)
    line_ones = drawn[None, :].expand(len(frac), -1).reshape(-1, 1).float()
    lines = torch.zeros(HEIGHT, WIDTH, 1, **f32)
    # sunlight above the water: the same rays, vertical, fading in from the top
    beam_x = (np.arange(-PAD, WIDTH + PAD, int(RAY_STEP)) + float(jitter[0]))[:, None]
    beams = np.exp(-0.5 * ((np.arange(WIDTH)[None, :] - beam_x) / 0.6) ** 2).sum(0)
    rise = np.clip((np.arange(HEIGHT) - SIDE_TOP) / (SURFACE_Y - SIDE_TOP), 0, 1)
    sunbeams = torch.tensor(0.32 * rise[:, None] ** 1.5 * beams[None, :], **f32)
    fine = gaussian_kernel(0.8, device)

    lut = torch.tensor(palette_lut(LIGHT, 1024), **f32)
    kernel = gaussian_kernel(7.0, device)
    albedo = torch.tensor(tiles(), **f32)
    rows = torch.arange(HEIGHT, **f32)[:, None]
    cols = torch.arange(WIDTH, **f32)
    fade_in = ((rows - FLOOR_TOP) / 60.0).clamp(0, 1)  # soft top edge of the floor
    floor_rows = rows >= FLOOR_TOP
    side_rows = (rows >= SIDE_TOP) & (rows < SIDE_BOTTOM)
    bg = torch.tensor(palette_lut([BACKGROUND, BACKGROUND], 2)[0], **f32)
    water = torch.tensor(palette_lut([WATER, WATER], 2)[0], **f32)
    sun = torch.tensor(palette_lut([SUN, SUN], 2)[0], **f32)
    side_fade = ((rows - SIDE_TOP) / 50.0).clamp(0, 1) * (
        (SIDE_BOTTOM - rows) / 12.0
    ).clamp(0, 1)
    text_colour, text_alpha = (torch.tensor(a, **f32) for a in overlay())
    canvas = torch.zeros(HEIGHT, WIDTH, 1, **f32)
    slice_y = torch.tensor(float(SLICE_Y), **f32)

    def displacement(x, y, t):
        dx, dy = torch.zeros_like(x), torch.zeros_like(x)
        for j in range(WAVES):  # one wave at a time keeps GPU memory small
            s = torch.sin(kx[j] * x + ky[j] * y + (ph[j] - om[j] * t))
            dx.sub_(amp[j] * ux[j] * s)
            dy.sub_(amp[j] * uy[j] * s)
        return dx, dy

    def surface(x, t):
        theta = (
            kx[:, None] * x[None]
            + ky[:, None] * slice_y
            - om[:, None] * t
            + ph[:, None]
        )
        return SURFACE_Y - (lift[:, None] * torch.cos(theta)).sum(0)

    def tone(light, gain):
        v = 1.0 - torch.exp(-gain * light)
        return v

    def render(frame):
        t = DURATION * frame / FRAME_COUNT
        image = bg.expand(HEIGHT, WIDTH, 3).clone()

        # Floor, from above.
        canvas.zero_()
        dx, dy = displacement(px0, py0, t)
        splat(canvas, px0 + dx, py0 + dy, ones)
        light = blur(canvas, fine)[..., 0] / mean_floor * albedo
        v = tone(light, FLOOR_GAIN)
        v = v + BLOOM * blur(v[..., None], kernel)[..., 0] * 0.6
        floor = lut[(v.clamp(0, 1) * 1023).long()] * fade_in[..., None]
        image = torch.where(floor_rows[..., None], floor, image)

        # The slice, from the side.
        slice_light = v[SLICE_Y]  # the floor's light along the slice, (WIDTH,)
        canvas.zero_()
        lines.zero_()
        sx, _ = displacement(rx, slice_y.expand_as(rx), t)
        top = surface(rx, t)
        x = (rx[None, :] + frac[:, None] * sx[None, :]).reshape(-1)
        y = (top[None, :] + frac[:, None] * (FLOOR_Y - top[None, :])).reshape(-1)
        keep = (y <= FLOOR_Y + 0.5)[:, None].float()
        splat(canvas, x, y, side_ones * keep)
        splat(lines, x, y, line_ones * keep)
        v = tone(canvas[..., 0], GLOW_GAIN) + LINE_ALPHA * tone(
            lines[..., 0], SIDE_GAIN
        )
        v = v + BLOOM * blur(v[..., None], kernel)[..., 0] * 0.6
        rays = lut[(v.clamp(0, 1) * 1023).long()]
        top_px = surface(cols, t)  # surface height per pixel column
        under = (rows > top_px[None, :]) & (rows <= FLOOR_Y)
        above = rows <= top_px[None, :]
        side = bg + 0.0 * rays
        side = torch.where(under[..., None], water + rays, side)
        side = torch.where(above[..., None], bg + sun * sunbeams[..., None], side)
        # the surface line and the floor line
        line = torch.exp(-0.5 * ((rows - top_px[None, :]) / 1.1) ** 2)
        floor_line = torch.exp(-0.5 * ((rows - FLOOR_Y) / 1.0) ** 2)
        side = side + 0.55 * line[..., None] * lut[900] + 0.18 * floor_line[..., None]
        # under the floor line: the floor's own light along the slice
        strip = ((rows > FLOOR_Y) & (rows <= FLOOR_Y + STRIP)).float()
        strip_colour = lut[(slice_light.clamp(0, 1) * 1023).long()]
        side = torch.where(
            strip[..., None] > 0, strip_colour[None, :, :].expand(HEIGHT, -1, -1), side
        )
        side = bg + (side - bg) * side_fade[..., None]
        image = torch.where(side_rows[..., None], side, image)

        image = image * (1 - text_alpha) + text_colour
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    track = soundtrack(landings(k, omega, phase, amplitude), settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def landings(k, omega, phase, amplitude):
    """
    Every moment a caustic lands on the floor in the side view: [(time, x, depth)].

    Along the slice the rays fold where J(x, t) = 1 + ∂Δ_x/∂x reaches zero at the
    floor; a fan of rays touching the floor is a stretch of x where J < 0 being born.
    J is evaluated from the same closed form as the frames on a fine (x, t) grid
    over the loop; a landing is a negative stretch with no negative neighbour at the
    previous time step (circularly), at its x. `depth` is how far J dips below zero
    there over the next 0.3 s: how bright that caustic gets.
    """
    x = np.arange(0.0, WIDTH, 2.0)
    times = np.arange(0.0, DURATION, SOUND_STEP)
    unit = k / np.linalg.norm(k, axis=1)[:, None]
    # ∂Δ_x/∂x = -Σ d_j û_x,j k_x,j cos(k·p - ωt + φ) on the row SLICE_Y
    weight = -(amplitude * unit[:, 0] * k[:, 0])[:, None]
    base = k[:, :1] * x[None, :] + (k[:, 1] * SLICE_Y + phase)[:, None]
    w = omega[:, None]
    jac = np.stack([1.0 + (weight * np.cos(base - w * t)).sum(0) for t in times])
    negative = jac < 0
    previous = np.roll(negative, 1, axis=0)
    near = previous | np.roll(previous, 1, axis=1) | np.roll(previous, -1, axis=1)
    events = []
    ahead = int(round(0.3 / SOUND_STEP))
    for i, j in zip(*np.nonzero(negative & ~near)):
        if j > 0 and negative[i, j - 1]:
            continue  # only the stretch's left end
        span = jac[(i + np.arange(ahead)) % len(times), max(j - 15, 0) : j + 15]
        events.append((float(times[i]), float(x[j]), float(-span.min())))
    events.sort()
    logger.info(f"{len(events)} caustic landings in the side view")
    return events


def soundtrack(events, rng):
    """
    Felt piano over a quiet string bed, a seamless loop (circular adds and room).

    Every landing (see `landings`) strikes one piano note the moment its fan of rays
    touches the floor in the side view: the pitch is the landing's x across the
    screen, DEGREES degrees of the major scale from left (low) to right (high),
    panned the same way; the velocity grows with how bright the caustic gets.
    Landings closer than 0.12 s to the previous note are skipped so the line
    breathes. Under it, strings hold the PROGRESSION, one chord per quarter of the
    loop, and a low piano root starts each chord.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    progression = PROGRESSIONS[PROGRESSION]
    track = audio.silence(DURATION)

    last = -1.0
    played = 0
    for when, x, depth in events:
        if when - last < 0.12:
            continue
        last = when
        degree = int(round((DEGREES - 1) * x / WIDTH)) - 2
        velocity = round(float(np.clip(0.28 + 0.9 * depth, 0.25, 0.75)) * 20) / 20
        note = style.voice(scale.hz(degree), velocity)
        audio.add(track, audio.pan(note, 0.8 * (2 * x / WIDTH - 1)), when)
        played += 1
    logger.info(f"{played} piano notes")

    bar = DURATION / len(progression)
    for i, chord_degree in enumerate(progression):
        for midi in scale.chord(chord_degree, 3, octave=-1):
            bed = audio.strings(float(audio.midi_to_hz(midi)), bar, 0.16, 1.2, 1.6, 0.3)
            audio.add(track, bed, i * bar - 0.3)
        root = style.bass(scale.hz(chord_degree, octave=-2), 0.35)
        audio.add(track, root, i * bar)
    return audio.master(style.room(track, rng))


if __name__ == "__main__":
    generate()
