"""
One Electron at a Time

The double-slit experiment, one electron at a time. Each electron leaves
the gun as a little packet of quantum wave, goes through both slits at
once, and the two halves overlap: crest on crest here, crest on trough
there. Then it touches the screen and the whole wave vanishes into one
green dot.

Each dot lands at random. But dots only land where the two halves of the
wave added up, never where they cancelled, and the stripes of an
interference pattern build up out of single electrons. The chance of a dot
at each point is the brightness of the wave there, |ψ|²: Born's rule.

Each landing strikes a note set by the stripe it falls in, so you hear the
electrons find the pattern too.

Made with Python & PyTorch. Sound synthesized in NumPy.
#doubleslit #quantummechanics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
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

# Geometry, px.
CX = WIDTH / 2
Y_GUN = 150.0
Y_WALL = 600.0
WALL_HALF = 5.0  # wall thickness / 2
Y_SCREEN = 1340.0
PLATE = (1372.0, 1868.0)  # the photographic plate under the screen
SLIT_GAP = 130.0  # centre to centre
SLIT_WIDTH = 34.0
POINTS_PER_SLIT = 6  # Huygens sources across each slit
BEAM_WIDTH = 210.0  # sd of the incoming packet across x

# The wave. de Broglie: phase speed is half the group speed.
WAVELENGTH = 18.0
GROUP_SPEED = 600.0  # px/s, the packets
PHASE_SPEED = GROUP_SPEED / 2
PACKET_SD = 42.0  # px along the path
FRONT = 40.0  # px, softness of the continuous beam's front
R0 = 60.0  # softens 1/sqrt(r) next to the slits

# Story, s.
EMISSIONS = (-0.62, 1.05, 2.5, 3.65, 4.55, 5.25, 5.8, 6.25, 6.62, 6.95)
COLLAPSE = 0.07  # s for a packet to vanish once it touches the screen
BEAM_ON = 6.4  # the continuous beam leaves the gun
BEAM_FULL = 9.0
BEAM_OFF = (12.9, 13.9)
RATE_START = 3.0  # beam hits per second at first ...
RATE_GROWTH = 0.42  # ... growing e-fold every this many seconds ...
RATE_MAX = 5200.0  # ... up to this
HITS_END = 13.6  # s, the last hit

# Look.
WAVE_PALETTE = ["#000000", "#1a0f45", "#4b2bb8", "#9a7bff", "#e6dcff", "#ffffff"]
WAVE_GAIN = 2.2
DOT = np.array([0.32, 1.0, 0.58])  # phosphor green, RGB
FLASH = np.array([0.85, 1.0, 0.9])
DOT_GAIN = 1.0
BEAM_DOT = 4.0  # splat weight of a beam electron's dot
LONE_DOT = 60.0  # and of a lone electron's
WALL_RGB = (58, 52, 80)
SCREEN_RGB = (40, 90, 60)
PLATE_RGB = (7, 13, 10)

STYLE = (
    "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
FRINGE = WAVELENGTH * (Y_SCREEN - Y_WALL) / SLIT_GAP  # px between bright stripes
NOTE_GAP = (0.32, 0.13)  # s between beam notes: sparse at first, then a rain


def slit_sources():
    """x of the Huygens point sources across both slits, and their weights."""
    offsets = (np.arange(POINTS_PER_SLIT) + 0.5) / POINTS_PER_SLIT - 0.5
    xs = np.concatenate(
        [CX + side * SLIT_GAP / 2 + SLIT_WIDTH * offsets for side in (-1, 1)]
    )
    weights = np.exp(-0.5 * ((xs - CX) / BEAM_WIDTH) ** 2) / POINTS_PER_SLIT
    return xs, weights


def screen_amplitude(x):
    """Steady complex amplitude on the screen line, from the same Huygens sum."""
    k = 2 * np.pi / WAVELENGTH
    xs, weights = slit_sources()
    dy = Y_SCREEN - Y_WALL
    r = np.hypot(x[:, None] - xs[None], dy)
    tilt = dy / r
    terms = weights * tilt / np.sqrt((r + R0) / WAVELENGTH) * np.exp(1j * k * r)
    return terms.sum(1)


def beam_rate(t):
    """Hits per second from the continuous beam."""
    first = BEAM_ON + (Y_SCREEN - Y_GUN) / GROUP_SPEED
    rate = RATE_START * np.exp((t - first) / RATE_GROWTH)
    rate = np.minimum(rate, RATE_MAX)
    rate *= np.clip((HITS_END - t) / 0.5, 0, 1)
    return np.where(t >= first, rate, 0.0)


def hits(rng):
    """Every detection: time, x, y on the plate, and whether it was a lone packet."""
    xs = np.arange(0, WIDTH, 0.25)
    density = np.abs(screen_amplitude(xs)) ** 2
    cdf = np.cumsum(density)
    cdf /= cdf[-1]

    travel = (Y_SCREEN - Y_GUN) / GROUP_SPEED
    lone = np.array(EMISSIONS) + travel
    grid = np.linspace(0, DURATION, 30001)
    cum = np.cumsum(beam_rate(grid)) * (grid[1] - grid[0])
    count = int(cum[-1])
    beam = np.interp(np.sort(rng.uniform(0, cum[-1], count)), cum, grid)
    times = np.concatenate([lone, beam])
    x = np.interp(rng.random(len(times)), cdf, xs)
    y = rng.uniform(PLATE[0] + 8, PLATE[1] - 8, len(times))
    is_lone = np.arange(len(times)) < len(lone)
    return times, x, y, is_lone


def soundtrack(style_name, times, x, is_lone, rng):
    """
    Every lone electron strikes a clear note when it lands: the scale degree
    is the stripe it lands in, n = round((x - CX) / FRINGE) (the centre stripe
    is the root, outer stripes higher to the right and lower to the left),
    panned by x. As each lone packet passes the slits a soft fifth an octave
    down sounds, one note per slit. The continuous beam swells a choir chord
    ("u") with its amplitude; its hits ring quietly, at most one per NOTE_GAP
    (shrinking as the rate grows), with the same stripe-to-pitch rule. When
    the hits stop, the home chord is struck and rings over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    def stripe(xi):
        return int(np.clip(round((xi - CX) / FRINGE), -5, 5))

    through = (Y_WALL - Y_GUN) / GROUP_SPEED
    for e in EMISSIONS:
        when = e + through
        if 0 <= when < DURATION:
            for degree, side in ((0, -0.35), (4, 0.35)):
                note = style.bass(scale.hz(degree, -1), 0.2)
                audio.add(track, audio.pan(note, side), when, **kw)

    for when, xi in zip(times[is_lone], x[is_lone]):
        note = style.voice(scale.hz(stripe(xi)), 0.6)
        audio.add(track, audio.pan(note, 0.8 * (xi - CX) / CX), when, **kw)

    order = np.argsort(times)
    beam = order[~is_lone[order]]
    start, end = times[beam].min(), times[beam].max()
    last = -1.0
    for i in beam:
        when = times[i]
        progress = smoothstep((when - start) / (end - start - 1.0))
        if when - last < NOTE_GAP[0] + (NOTE_GAP[1] - NOTE_GAP[0]) * progress:
            continue
        last = when
        note = style.voice(scale.hz(stripe(x[i]), 1), 0.18 + 0.17 * progress)
        audio.add(track, audio.pan(note, 0.8 * (x[i] - CX) / CX), when, **kw)

    held = BEAM_OFF[1] - BEAM_ON
    for degree in (0, 2, 4):
        voice = audio.choir(
            scale.hz(degree, -1), held, 0.5, attack=BEAM_FULL - BEAM_ON, vowel="u"
        )
        audio.add(track, voice, BEAM_ON, **kw)
    ending = HITS_END
    for j, degree in enumerate((0, 2, 4, 7)):
        audio.add(track, style.voice(scale.hz(degree), 0.5), ending + 0.07 * j, **kw)
    audio.add(track, style.bass(scale.hz(0, -2), 0.6), ending, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A double slit in 2D, a closed-form Huygens model evaluated per pixel on the
    GPU. A packet emitted at time e is psi_in = T(x) G(y - Y_GUN - v_g tau)
    exp(i(k (y - Y_GUN) - w tau)) above the wall (tau = t - e, G a Gaussian of
    sd PACKET_SD, T a Gaussian of sd BEAM_WIDTH). Below the wall it is a sum over
    POINTS_PER_SLIT point sources across each slit: a_m cos(theta_m)
    / sqrt(r_m + R0) G(P + r_m - v_g tau) exp(i(k (P + r_m) - w tau)), P the gun-
    to-wall distance, so every source is a delayed copy of the incoming wave
    and the slits' single-slit envelope and two-slit fringes both come out of
    the sum. Phase speed w/k = v_g/2 (de Broglie). Shown as
    |psi|^2 (0.55 + 0.45 cos arg psi): the probability density with its crests.

    Ten lone packets (EMISSIONS) each collapse COLLAPSE s after reaching the
    screen: the wave fades and a dot appears. From BEAM_ON a continuous beam
    (the same sum with G replaced by a soft front) fades in; its hits arrive at
    a rate growing exponentially to RATE_MAX. Every hit's x is drawn from the
    steady |psi|^2 on the screen line (Born's rule) and its y uniformly on the
    plate, all up front from settings.rng; a dot is splatted with a decaying
    white flash, and lone hits also flash on the screen line. Frames render in
    order on the GPU.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()
    f32 = dict(device=device, dtype=torch.float32)

    times, hx, hy, is_lone = hits(rng)
    logger.info(f"{len(times)} electrons ({is_lone.sum()} lone)")

    k = 2 * np.pi / WAVELENGTH
    omega = k * PHASE_SPEED
    path_wall = Y_WALL - Y_GUN

    # Incoming region, rows [0, Y_WALL).
    n_in = int(Y_WALL)
    yy = torch.arange(n_in, **f32)[:, None] + 0.5
    xx = torch.arange(WIDTH, **f32)[None] + 0.5
    profile = torch.exp(-0.5 * ((xx - CX) / BEAM_WIDTH) ** 2)
    gun_mask = torch.clamp((yy - Y_GUN + 25) / 40, 0, 1)
    in_path = (yy - Y_GUN).expand(n_in, WIDTH)
    in_amp = profile * gun_mask * torch.exp(1j * k * in_path)

    # Outgoing region, rows [Y_WALL, Y_SCREEN).
    n_out = int(Y_SCREEN - Y_WALL)
    yo = torch.arange(n_out, **f32)[:, None] + 0.5
    src_x, src_w = slit_sources()
    out_path, out_amp = [], []
    for x0, w in zip(src_x, src_w):
        r = torch.sqrt((xx - x0) ** 2 + yo**2)
        tilt = yo / r
        out_path.append(path_wall + r)
        amp = w * tilt / torch.sqrt((r + R0) / WAVELENGTH)
        out_amp.append(amp * torch.exp(1j * k * (path_wall + r)))
    out_path = torch.stack(out_path)  # (S, n_out, W)
    out_amp = torch.stack(out_amp).to(torch.complex64)

    # Brightness at the screen matches the incoming peak.
    steady = out_amp.sum(0)
    out_amp /= steady[-40:, int(CX) - 60 : int(CX) + 60].abs().mean()

    wave_lut = torch.tensor(palette_lut(WAVE_PALETTE, 1024), **f32)
    dot_rgb = torch.tensor(DOT, **f32)
    flash_rgb = torch.tensor(FLASH, **f32)
    soft = gaussian_kernel(1.1, device)
    lone_kernel = gaussian_kernel(2.6, device)

    # The static set: wall with two gaps, screen line, plate.
    stage = torch.zeros(HEIGHT, WIDTH, 3, **f32)
    wall_rows = slice(int(Y_WALL - WALL_HALF), int(Y_WALL + WALL_HALF))
    stage[wall_rows] = torch.tensor(WALL_RGB, **f32)
    for side in (-1, 1):
        left = int(round(CX + side * SLIT_GAP / 2 - SLIT_WIDTH / 2))
        stage[wall_rows, left : left + int(SLIT_WIDTH)] = 0
    stage[int(PLATE[0]) : int(PLATE[1])] = torch.tensor(PLATE_RGB, **f32)
    stage[int(Y_SCREEN) : int(Y_SCREEN) + 2] = torch.tensor(SCREEN_RGB, **f32)
    open_air = torch.ones(HEIGHT, WIDTH, 1, **f32)
    open_air[wall_rows] = (stage[wall_rows].sum(-1, keepdim=True) == 0).float()
    wall = stage * (1 - open_air)
    gun = torch.zeros(HEIGHT, WIDTH, 3, **f32)
    gun[int(Y_GUN) - 3 : int(Y_GUN) + 3, int(CX) - 3 : int(CX) + 3] = 1.0
    gun = 255 * wide_blur(gun, 14, 2) * torch.tensor([0.6, 0.45, 1.0], **f32) * 40

    t_hit = torch.tensor(times, **f32)
    px = torch.tensor(hx, **f32)
    py = torch.tensor(hy, **f32)
    lone = torch.tensor(is_lone, device=device)
    travel = (Y_SCREEN - Y_GUN) / GROUP_SPEED

    steady = out_amp.sum(0)
    out_path_max = float(out_path.max())
    top = int(Y_SCREEN) - 40  # the dots layer covers rows [top, HEIGHT)

    def packet(tau, front):
        """The field of one packet (or of the beam with `front`), both regions."""
        if front and GROUP_SPEED * tau > out_path_max + 8 * FRONT:
            spin = np.exp(-1j * omega * tau)  # fully on: the steady field
            return in_amp * spin, steady * spin
        if front:
            shape_in = torch.sigmoid((GROUP_SPEED * tau - in_path) / FRONT)
            shape_out = torch.sigmoid((GROUP_SPEED * tau - out_path) / FRONT)
        else:
            shape_in = torch.exp(
                -0.5 * ((in_path - GROUP_SPEED * tau) / PACKET_SD) ** 2
            )
            shape_out = torch.exp(
                -0.5 * ((out_path - GROUP_SPEED * tau) / PACKET_SD) ** 2
            )
        spin = np.exp(-1j * omega * tau)
        psi_in = in_amp * shape_in * spin
        psi_out = (out_amp * shape_out).sum(0) * spin
        return psi_in, psi_out

    def render(frame):
        t = frame / (FRAME_COUNT - 1) * DURATION
        psi_in = torch.zeros(n_in, WIDTH, dtype=torch.complex64, device=device)
        psi_out = torch.zeros(n_out, WIDTH, dtype=torch.complex64, device=device)

        for e in EMISSIONS:
            tau = t - e
            age = tau - travel
            if tau < -0.3 or age > 6 * COLLAPSE:
                continue
            fade = 1.0 if age < 0 else float(np.exp(-age / COLLAPSE))
            a, b = packet(tau, False)
            psi_in += fade * a
            psi_out += fade * b

        beam = np.clip((t - BEAM_ON) / (BEAM_FULL - BEAM_ON), 0, 1) ** 0.5
        beam *= 1 - np.clip((t - BEAM_OFF[0]) / (BEAM_OFF[1] - BEAM_OFF[0]), 0, 1)
        if beam > 0:
            a, b = packet(t - BEAM_ON, True)
            psi_in += 0.75 * beam * a
            psi_out += 0.75 * beam * b

        psi = torch.cat([psi_in, psi_out])
        mag = psi.abs()
        density = mag**2 * 0.55 + 0.45 * mag * psi.real
        light = 1 - torch.exp(-WAVE_GAIN * density.clamp(min=0))
        index = (light * 1023).long().clamp(0, 1023)
        wave = wave_lut[index] * 255
        image = stage.clone()
        image[: int(Y_SCREEN)] += wave
        image[: int(Y_SCREEN)] += 0.5 * wide_blur(wave, 10, 4)
        image += gun
        image = image * open_air + wall

        # Dots: everything that has landed, flashing as it lands.
        seen = t_hit <= t
        if seen.any():
            age = t - t_hit[seen]
            is_l = lone[seen]
            flash = torch.exp(-age / 0.25) * torch.where(is_l, 0.0, 6.0)
            dots = torch.zeros(HEIGHT - top, WIDTH, 3, **f32)
            colour = BEAM_DOT * dot_rgb + flash[:, None] * flash_rgb
            beam_hit = ~is_l
            splat(dots, px[seen][beam_hit], py[seen][beam_hit] - top, colour[beam_hit])
            dots = blur(dots, soft)
            if is_l.any():  # lone electrons: bigger dots, a long flash, a spark
                a = age[is_l]
                big = torch.zeros(HEIGHT - top, WIDTH, 3, **f32)
                flash = 300 * torch.exp(-a / 0.35)
                colour = LONE_DOT * dot_rgb + flash[:, None] * flash_rgb
                splat(big, px[seen][is_l], py[seen][is_l] - top, colour)
                spark = 400 * torch.exp(-a / 0.15)
                splat(
                    big,
                    px[seen][is_l],
                    torch.full_like(spark, Y_SCREEN - 2 - top),
                    spark[:, None] * flash_rgb,
                )
                dots += blur(big, lone_kernel)
            glow = wide_blur(dots, 14, 4)
            light = 1 - torch.exp(-DOT_GAIN * (dots + 1.5 * glow))
            image[top:] += 255 * light

        image *= float(np.clip((DURATION - t) / FADE, 0, 1))
        return image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    style = tag if tag in STYLES else STYLE
    track = soundtrack(style, times, hx, is_lone, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
