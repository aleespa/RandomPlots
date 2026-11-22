"""
Through a Wall It Can't Climb

Quantum tunnelling, watched as it happens. A single quantum particle, an
electron say, is a little packet of wave racing towards a wall. The wall is
higher than the particle's energy: a ball thrown at it would bounce back,
every time.

The wave doesn't. It slams into the wall and folds back on itself in bright
stripes, while a faint tail leaks into the wall. The wall is thin, so the tail
reaches the far side before it dies out, and there it becomes a wave again. The packet splits in two: most of it bounces back, but a large part
simply comes out the other side.

That is how the Sun fuses hydrogen, how radioactive nuclei decay and how a
scanning tunnelling microscope sees single atoms. The colours are the wave's
phase; the brightness is |ψ|², the chance of finding the particle there.

The electric piano keeps the wave's own clock: a note each time its phase
turns once, high and soft for the part that tunnelled, low for the part that
bounced back.

Made with Python, SciPy & PyTorch. Sound synthesized in NumPy.
#quantumtunneling #quantummechanics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger
from scipy.linalg import eigh_tridiagonal

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    latex_mask,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = "Can a particle pass through a wall it can't climb?"
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

# The wave, in pixels and seconds: i dpsi/dt = -D d2psi/dy2 + V(y) psi.
WAVELENGTH = 24.0  # px
GROUP_SPEED = 100.0  # px/s, the packet's speed
PACKET_SD = 72.0  # px, sd of |psi|^2 along the motion
BEAM_SD = 165.0  # px, sd of |psi|^2 across it
SIDE_SPEED = 40.0  # px/s, the packet also drifts sideways, so it strikes at an angle
X_START = -320.0  # px from the centre at physical time 0
Y_START = 1330.0  # packet centre at physical time 0 (it moves up the screen)
Y_WALL = 960.0
WALL_WIDTH = 8.0  # px
WALL_RATIO = 1.18  # wall height V0 over the packet's mean energy E
DOMAIN = (-1300.0, 3200.0)  # rows simulated; hard walls far off screen

# Playback: physical time runs with video time, then slows to a stop.
SLOW_FROM, STOP_AT = 9.2, 12.0  # s
CRAWL = 0.12  # speed after the slowdown, so the phase keeps shimmering
LABELS_IN = (10.2, 11.2)  # s, the percentages fade in
WALL_LABEL = (0.2, 1.0, 2.6, 3.2)  # s: fade in, full, fade out, gone

PALETTE = ["#00e5ff", "#2f6bff", "#ff3d8b", "#ffc15e"]  # cyclic, by phase
WALL_RGB = np.array([1.0, 0.93, 0.8])
GAIN = 2.0
TEXT_RGB = (0.92, 0.92, 0.95)

# Sound: a pad bed in dorian with Rhodes notes struck on the wave's own clock.
MODE = "dorian"
ROOTS = range(50, 56)  # MIDI root of the scale
APPROACH = (0, 2, 4, 5, 7, 9, 11, 9)  # degrees, climbing as the wave nears the wall
THROUGH = (11, 9, 7, 9, 8, 7)  # the tunnelled packet: high and light
BACK = (0, -1, 0, 2, 1, 0)  # the reflected packet: low and fuller
CHORDS = ((0.0, 0), (3.4, 5), (6.6, 3), (10.0, 0))  # (s, chord degree)


def physical_time(t):
    """Seconds of wave evolution shown at video time t: 1:1, then easing to a crawl."""
    x = np.clip((t - SLOW_FROM) / (STOP_AT - SLOW_FROM), 0.0, 1.0)
    # integral of the speed 1 - (1 - CRAWL) smoothstep(x) over the slowdown
    slowed = (STOP_AT - SLOW_FROM) * (x - (1 - CRAWL) * (x**3 - x**4 / 2))
    return np.minimum(t, SLOW_FROM) + slowed + CRAWL * np.maximum(t - STOP_AT, 0.0)


def eigenbasis():
    """Eigenstates of the 1-D Hamiltonian with the wall, and the packet in that basis."""
    k = 2 * np.pi / WAVELENGTH
    d = GROUP_SPEED / (2 * k)  # D = hbar / 2m in px^2/s
    energy = d * k * k
    y = np.arange(*DOMAIN, 1.0)
    wall = np.where(np.abs(y - Y_WALL) < WALL_WIDTH / 2, WALL_RATIO * energy, 0.0)
    values, vectors = eigh_tridiagonal(
        2 * d + wall,
        -d * np.ones(len(y) - 1),
        select="v",
        select_range=(0.0, 3.0 * energy),
    )
    packet = np.exp(-((y - Y_START) ** 2) / (4 * PACKET_SD**2) - 1j * k * y)
    packet /= np.linalg.norm(packet)
    coefficients = vectors.T @ packet
    logger.info(
        f"{len(values)} eigenstates, captured norm {np.sum(abs(coefficients) ** 2):.6f}"
    )
    return y, d, values, vectors, coefficients


def generate(settings: ImageProcessingSettings = None):
    """
    Quantum tunnelling of a 2-D Gaussian wave packet through a thin rectangular
    barrier. The barrier depends on y only, so the problem separates:
    psi(x, y, t) = psi_x(x, t) psi_y(y, t). psi_x is a free Gaussian (closed form,
    spreading with complex width 1 + iDt/sigma^2). psi_y is solved spectrally: the
    tridiagonal finite-difference Hamiltonian H = -D d2/dy2 + V on 4500 rows at 1 px
    is diagonalised once (scipy eigh_tridiagonal, the ~650 states below 3E), the
    packet is projected on it, and every frame is psi_y = Phi (c e^{-i E_n tau}),
    a pure function of time. The wall is WALL_RATIO times the packet's mean energy;
    the transmitted fraction (shown at the end) comes from the same evolution.

    Physical time tau follows video time until SLOW_FROM, then eases down to CRAWL
    speed by STOP_AT and rests there (one-shot), with a fade to black over the last FADE s.
    Rendering (torch, GPU): brightness 1 - exp(-GAIN |psi|^2 / peak), colour from a
    cyclic palette indexed by arg psi, a white-hot core, two bloom layers, and the
    wall as a glowing band. Labels are LaTeX masks built once.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()

    y, d, values, vectors, coefficients = eigenbasis()
    on_screen = (y >= 0) & (y < HEIGHT)
    screen_vectors = vectors[on_screen]

    times = np.arange(FRAME_COUNT) / (FRAME_COUNT - 1) * DURATION
    taus = physical_time(times)
    final = vectors @ (coefficients * np.exp(-1j * values * taus[-1]))
    transmitted = float(np.sum(np.abs(final[y < Y_WALL]) ** 2))
    logger.info(f"transmitted {transmitted:.3f}")
    y_through = float(np.sum(y[y < Y_WALL] * np.abs(final[y < Y_WALL]) ** 2))
    y_through /= transmitted
    y_back = float(np.sum(y[y > Y_WALL] * np.abs(final[y > Y_WALL]) ** 2))
    y_back /= 1 - transmitted

    # psi_y on screen for every frame: (frames, HEIGHT) complex
    phases = np.exp(-1j * np.outer(taus, values))
    psi_y = (phases * coefficients) @ screen_vectors.T
    peak_y = np.max(np.abs(psi_y[0]) ** 2)
    psi_y = torch.tensor(psi_y.astype(np.complex64), device=device)

    x = np.arange(WIDTH) - WIDTH / 2
    kx = SIDE_SPEED / (2 * d)
    x_end = WIDTH / 2 + X_START + SIDE_SPEED * taus[-1]
    lut = torch.tensor(palette_lut(PALETTE + PALETTE[:1], 512), device=device)
    small = gaussian_kernel(3.0, device)

    # the wall: a thin bright band with a soft halo, drawn once
    rows = np.arange(HEIGHT) - Y_WALL
    band = np.exp(-0.5 * (rows / 2.5) ** 2) * (np.abs(rows) < 6)
    halo = 0.25 * np.exp(-0.5 * (rows / 14.0) ** 2)
    wall = (band + halo)[:, None, None] * WALL_RGB[None, None, :]
    wall = torch.tensor(
        np.broadcast_to(wall, (HEIGHT, WIDTH, 3)).copy(), device=device
    ).float()

    percent = int(round(100 * transmitted))
    label_wall = latex_mask(r"$\mathrm{a\ wall\ higher\ than\ its\ energy}$", 42)
    label_through = latex_mask(rf"${percent}\%\ \mathrm{{tunnels\ through}}$", 46)
    label_back = latex_mask(rf"${100 - percent}\%\ \mathrm{{bounces\ back}}$", 46)

    def render(frame):
        t = times[frame]
        tau = taus[frame]
        width = 1 + 1j * d * tau / BEAM_SD**2
        u = x - X_START
        psi_x = np.exp(
            -((u - SIDE_SPEED * tau) ** 2) / (4 * BEAM_SD**2 * width)
            + 1j * kx * u
            - 1j * d * kx * kx * tau
        ) / np.sqrt(width)
        psi_x = torch.tensor(psi_x.astype(np.complex64), device=device)
        field = psi_y[frame][:, None] * psi_x[None, :]
        density = (field.real**2 + field.imag**2) / peak_y
        brightness = 1 - torch.exp(-GAIN * density)
        phase = (torch.angle(field) / (2 * np.pi)) % 1.0
        colour = lut[(phase * (lut.shape[0] - 1)).long()]
        image = brightness[..., None] * colour
        image = image + torch.clamp(density - 0.6, 0.0, 2.0)[..., None] * 0.6
        image = image + 0.6 * blur(image, small) + 0.35 * wide_blur(image, 40.0, 8)
        image = image + wall
        image = image.clamp(0.0, 1.2).cpu().numpy()

        a, b, c, e = WALL_LABEL
        alpha = smoothstep((t - a) / (b - a)) * (1 - smoothstep((t - c) / (e - c)))
        if alpha > 0:
            blit_mask(
                image,
                label_wall,
                WIDTH - label_wall.shape[1] - 40,
                int(Y_WALL) - 30 - label_wall.shape[0],
                TEXT_RGB,
                alpha=0.85 * alpha,
            )
        alpha = smoothstep((t - LABELS_IN[0]) / (LABELS_IN[1] - LABELS_IN[0]))
        if alpha > 0:
            blit_mask(
                image,
                label_through,
                int(x_end),
                int(y_through + 3 * PACKET_SD),
                TEXT_RGB,
                alpha=alpha,
                anchor="centre",
            )
            blit_mask(
                image,
                label_back,
                int(x_end),
                int(y_back - 3 * PACKET_SD) - label_back.shape[0],
                TEXT_RGB,
                alpha=alpha,
                anchor="centre",
            )
        image *= 1 - smoothstep((t - (DURATION - FADE)) / FADE)
        return np.clip(image * 255, 0, 255)

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    track = soundtrack(values, coefficients, transmitted, d, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def soundtrack(values, coefficients, transmitted, d, rng):
    """
    Drone & pad with Rhodes accents, one-shot. The notes keep the wave's own
    clock: the packet's mean energy <E> = sum |c_n|^2 E_n is the angular frequency
    at which its phase turns, and a Rhodes note is struck each time the phase
    completes a turn (in physical time, so the notes slow down with the picture).
    Before the wall a single voice climbs; from the strike on, two voices alternate:
    the tunnelled packet high and soft (velocity ~ sqrt T), the reflected one an
    octave and more below and fuller (~ sqrt(1 - T)). Pan follows the packet's
    sideways drift. A pad (i - bVI - IV - i) holds underneath, a chord on the
    strike, and the home chord with a last Rhodes chord when the percentages
    appear. Same closed forms as the frames, and settings.audio_rng only picks
    the key and the room.
    """
    scale = Scale(int(rng.choice(ROOTS)), MODE)
    track = audio.silence(DURATION)
    sr = audio.SAMPLE_RATE

    energy = float(np.sum(np.abs(coefficients) ** 2 * values))
    t = np.linspace(0.0, DURATION, int(DURATION * 2000))
    turns = np.floor(physical_time(t) * energy / (2 * np.pi))
    onsets = t[1:][np.diff(turns) > 0]
    hit = (Y_START - Y_WALL) / GROUP_SPEED  # physical = video time before the slowdown
    last_note = LABELS_IN[0]

    def x_pan(when):
        x = X_START + SIDE_SPEED * physical_time(when)
        return float(np.clip(x / (WIDTH / 2), -0.8, 0.8))

    approach = [w for w in onsets if w < hit - 0.15]
    for i, when in enumerate(approach):
        degree = APPROACH[min(int(i * len(APPROACH) / len(approach)), 7)]
        velocity = 0.3 + 0.35 * i / len(approach)
        note = audio.rhodes(scale.hz(degree), velocity, length=2.5)
        audio.add(track, audio.pan(note, x_pan(when)), when, circular=False)

    # the strike on the wall: a low and a high note together
    for degree, octave, velocity in ((0, -1, 0.75), (4, 0, 0.55), (9, 0, 0.45)):
        note = audio.rhodes(scale.hz(degree, octave), velocity, length=3.5)
        audio.add(track, audio.pan(note, x_pan(hit)), hit, circular=False)

    after = [w for w in onsets if hit + 0.35 < w < last_note]
    for i, when in enumerate(after):
        through = i % 2 == 0
        pattern = THROUGH if through else BACK
        degree = pattern[(i // 2) % len(pattern)]
        octave = 1 if through else -1
        weight = np.sqrt(transmitted if through else 1 - transmitted)
        note = audio.rhodes(scale.hz(degree, octave), 0.25 + 0.5 * weight, length=3.0)
        side = x_pan(when) + (0.25 if through else -0.25)
        audio.add(track, audio.pan(note, side), when, circular=False)

    # the resolution: home chord on the Rhodes as the percentages appear
    for j, degree in enumerate((0, 2, 4, 7)):
        note = audio.rhodes(scale.hz(degree, -1 if j == 0 else 0), 0.5, length=4.5)
        audio.add(track, audio.pan(note, 0.0), last_note + 0.06 * j, circular=False)

    # the pad bed
    bed = audio.silence(DURATION)
    ends = [c[0] for c in CHORDS[1:]] + [DURATION]
    for (start, degree), end in zip(CHORDS, ends):
        hold = end - start + (0.0 if end == DURATION else 0.6)
        velocity = 0.45 if start < 9.0 else 0.55
        for note in scale.chord(degree, 3):
            chord = audio.pad(
                float(audio.midi_to_hz(note)),
                hold,
                velocity,
                attack=1.2 if start else 0.25,
                release=1.5,
                brightness=0.3,
            )
            audio.add(bed, chord, start, circular=False)
    bed = audio.highpass(
        audio.lowpass(bed, 1800.0, circular=False), 80.0, circular=False
    )
    track += 0.6 * bed

    track = audio.reverb(track, rng, 2.2, 0.3, tone=3500.0, circular=False)
    track = audio.fade(track, fade_out=FADE)
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
