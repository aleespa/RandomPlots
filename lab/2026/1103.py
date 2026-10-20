"""
Clouds That Break Like Waves

Two layers of fluid slide past each other, indigo one way, amber the other. Nothing pushes
the line between them, yet it cannot stay flat: the faintest ripple is lifted by the shear,
curls over like surf and rolls up into a row of spirals. Neighbouring spirals then swallow
each other in pairs, and the layers mix into turbulence.

This is the Kelvin–Helmholtz instability, solved here from the fluid equations
∂ω/∂t + u·∇ω = ν∇²ω, with the dye simply carried along. The only ingredient is a
difference in speed. The same breaking waves appear in billow clouds on a windy evening,
on the bands of Saturn and Jupiter, and at the edge of the Sun.

Each layer hums a note set by its number of rolls: when two rolls merge, the wavelength
doubles and the strings drop an octave, while the wind rises as the fluids mix.

Made with Python, PyTorch & NumPy. Sound synthesized in NumPy.
#kelvinhelmholtz #fluiddynamics #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from common import audio
from common.gpu import select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# Solver grid: 9:16 like the frame, square cells, periodic both ways.
NX, NY = 432, 768
LY = 2 * np.pi
LX = LY * NX / NY
DT = 0.0035  # fixed simulation step, the same at every RP_FPS

# Indigo fluid (dye -1) -> pale mixed zone (0) -> ember fluid (+1).
PALETTE = [
    "#03050f",
    "#0b1440",
    "#1b2a6b",
    "#3f6fd8",
    "#f2dcb4",
    "#ffd27a",
    "#e8823a",
    "#5a1e08",
    "#0f0502",
]

KNOBS = dict(
    billows=4,  # rolls per interface at the start; pairs halve it
    thickness=0.06,  # shear-layer half-width; thin = sharp curls
    nu=6e-5,  # viscosity: lower -> earlier, finer turbulence
    seed_amp=2e-2,  # size of the initial ripple
    pairing=0.12,  # subharmonic share of the ripple: larger -> rolls pair sooner
    t_start=1.0,  # simulation time shown in the first frame (pre-roll)
    t_end=13.0,  # simulation time at which the picture settles
)


class ShearFlow:
    """2D incompressible vorticity + passive dye, pseudo-spectral on a periodic box."""

    def __init__(self, knobs: dict, rng: np.random.Generator, device: torch.device):
        self.device = device
        kx = 2 * np.pi * np.fft.rfftfreq(NX, LX / NX)
        ky = 2 * np.pi * np.fft.fftfreq(NY, LY / NY)
        KX, KY = np.meshgrid(kx, ky)
        k2 = KX**2 + KY**2
        inv_k2 = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
        dealias = (np.abs(KX) < (2 / 3) * np.abs(kx).max()) & (
            np.abs(KY) < (2 / 3) * np.abs(ky).max()
        )
        # Viscosity / dye diffusion plus a touch of hyperviscosity at the grid scale.
        kmax2 = (np.abs(kx).max() ** 2 + np.abs(ky).max() ** 2) / 2
        hyper = 100.0 * (k2 / kmax2) ** 4
        decay = np.stack([knobs["nu"] * k2 + hyper, knobs["nu"] * k2 + hyper])

        def tensor(a, dtype=torch.float32):
            return torch.tensor(a, dtype=dtype, device=device)

        self.ikx = tensor(1j * KX, torch.complex64)
        self.iky = tensor(1j * KY, torch.complex64)
        self.inv_k2 = tensor(inv_k2)
        self.dealias = tensor(dealias)
        self.decay = tensor(decay)

        # Double shear layer: u = -1 outside, +1 in the middle band.
        y = (np.arange(NY) + 0.5) * LY / NY
        x = (np.arange(NX) + 0.5) * LX / NX
        X, Y = np.meshgrid(x, y)
        d = knobs["thickness"]
        y1, y2 = LY / 4, 3 * LY / 4
        sech2 = lambda z: 1.0 / np.cosh(z) ** 2
        omega = -(sech2((Y - y1) / d) - sech2((Y - y2) / d)) / d
        # Ripple: the chosen billow count, its subharmonic (which drives pairing) and
        # a few random neighbours, each interface with its own phases.
        k0 = 2 * np.pi * knobs["billows"] / LX
        ripple = np.zeros_like(X)
        for yc in (y1, y2):
            env = sech2((Y - yc) / d) / d
            wave = np.cos(k0 * X + rng.uniform(0, 2 * np.pi))
            wave += knobs["pairing"] * np.cos(0.5 * k0 * X + rng.uniform(0, 2 * np.pi))
            for n in range(1, 2 * knobs["billows"] + 1):
                kn = 2 * np.pi * n / LX
                wave += (
                    0.08
                    * rng.standard_normal()
                    * np.cos(kn * X + rng.uniform(0, 2 * np.pi))
                )
            ripple += env * wave
        omega += knobs["seed_amp"] * ripple
        dye = np.tanh((Y - y1) / (0.6 * d)) - np.tanh((Y - y2) / (0.6 * d)) - 1.0
        self.state = torch.fft.rfft2(tensor(np.stack([omega, dye])))
        self.steps = 0

    def rhs(self, s):
        """-u·∇(ω, c), dealiased."""
        psi = s[0] * self.inv_k2
        grads = torch.fft.irfft2(
            torch.stack(
                [self.iky * psi, -self.ikx * psi, *(self.ikx * s), *(self.iky * s)]
            ),
            s=(NY, NX),
        )
        u, v = grads[0], grads[1]
        return -torch.fft.rfft2(u * grads[2:4] + v * grads[4:6]) * self.dealias

    def step(self):
        """RK4 with an exact integrating factor for the linear decay."""
        s, dt = self.state, DT
        e1 = torch.exp(-self.decay * dt / 2)
        e2 = e1 * e1
        k1 = self.rhs(s)
        k2 = self.rhs(e1 * (s + 0.5 * dt * k1))
        k3 = self.rhs(e1 * s + 0.5 * dt * k2)
        k4 = self.rhs(e2 * s + dt * e1 * k3)
        self.state = e2 * s + dt / 6 * (e2 * k1 + 2 * e1 * (k2 + k3) + k4)
        self.steps += 1

    def advance_to(self, steps: int):
        while self.steps < steps:
            self.step()

    def fields(self):
        return torch.fft.irfft2(self.state, s=(NY, NX))

    def measure(self):
        """
        (mixed fraction, rolls on the upper layer, rolls on the lower layer). Rolls are
        the strongest Fourier mode along x of each half's vorticity.
        """
        omega, dye = self.fields()
        mixed = float((dye.abs() < 0.5).float().mean())
        power = torch.fft.rfft(omega, dim=1).abs().square()
        rolls = [
            int(power[h, 1:9].sum(0).argmax()) + 1
            for h in (slice(0, NY // 2), slice(NY // 2, NY))
        ]
        return mixed, *rolls


# Soundtracks, by --audio-only tag: the voice holding each layer's drone, the voice
# struck when rolls pair, and the scale. Unknown or empty tags use SOUND.
SOUNDS = {
    "drone": dict(bed="pad", strike=audio.bell, mode="dorian", octave=1),
    "strings": dict(bed="strings", strike=audio.piano, mode="minor", octave=0),
    "flute": dict(bed="pad", strike=audio.flute, mode="minor_pentatonic", octave=1),
}
SOUND = "strings"
DRONE_ROOTS = range(43, 49)  # MIDI of a single roll's drone (G2 .. C3)
MIN_HOLD = 0.6  # s: a roll count must last this long to count as a pairing


def sim_time(t: float, knobs: dict) -> float:
    """
    Simulation time shown at normalised video time t in [0, 1]: steady speed, easing to
    a stop at 0.93 so the last second rests on the mixed layers.
    """
    stop = 0.93
    s = min(t / stop, 1.0)
    progress = (s - 0.25 * s**4) / 0.75  # speed 1 - s^3: constant, then easing out
    return knobs["t_start"] + (knobs["t_end"] - knobs["t_start"]) * progress


def steady(counts: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Roll counts with blips shorter than MIN_HOLD replaced by the count before them."""
    counts = counts.copy()
    start = 0
    for i in range(1, len(counts) + 1):
        if i == len(counts) or counts[i] != counts[start]:
            if start > 0 and times[min(i, len(times) - 1)] - times[start] < MIN_HOLD:
                counts[start:i] = counts[start - 1]
            start = i
    return counts


def soundtrack(sound_name: str, diagnostics: np.ndarray, rng) -> np.ndarray:
    """
    The flow, heard. `diagnostics` holds (mixed fraction, rolls up, rolls down) per frame.

    - Each layer holds a drone at k times one fixed pitch, k its number of rolls: rolls
      are a wavelength, so every pairing (k halves) drops that layer's drone an octave.
      The upper layer is panned left, the lower right; a fifth above fills each chord.
    - Every pairing is struck once (bell, piano or flute by tag), from the scale, lower
      as the rolls grow, and a roll-up chord sounds in the first second.
    - A bed of filtered noise, the wind, swells with the mixed fraction.
    - The last chord rings over the final rest and fades with the picture.
    """
    sound = SOUNDS[sound_name]
    sr = audio.SAMPLE_RATE
    kw = dict(circular=False)
    root = int(rng.choice(DRONE_ROOTS))
    scale = Scale(root, sound["mode"])
    track = audio.silence(DURATION)
    times = np.arange(len(diagnostics)) / (len(diagnostics) - 1) * DURATION
    pairings = 0
    for layer, side in ((1, -0.45), (2, 0.45)):
        counts = steady(diagnostics[:, layer].astype(int), times)
        edges = [0] + [i for i in range(1, len(counts)) if counts[i] != counts[i - 1]]
        for j, i in enumerate(edges):
            end = times[edges[j + 1]] if j + 1 < len(edges) else DURATION
            hz = float(audio.midi_to_hz(root)) * counts[i]
            for ratio, gain in ((1.0, 0.5), (1.5, 0.25)):
                f = hz * ratio
                hold = end - times[i] + (0.0 if j + 1 < len(edges) else 2.0)
                if sound["bed"] == "pad":
                    bed = audio.pad(f, hold, gain, 1.2, 1.5, 0.35)
                else:
                    bed = audio.strings(f, hold, gain, 1.0, 1.5, 0.4)
                bed = bed * np.array([1 - 0.5 * side, 1 + 0.5 * side])
                audio.add(track, bed, max(times[i] - 0.6, 0.0), **kw)
            if i > 0 and counts[i] < counts[i - 1]:
                degree = 2 * int(np.log2(max(counts[i], 1))) + layer - 1
                note = sound["strike"](scale.hz(degree, sound["octave"] + 1), 0.45)
                audio.add(track, audio.pan(note, side), times[i], **kw)
                pairings += 1
    for k, degree in enumerate((0, 2, 4)):
        note = sound["strike"](scale.hz(degree + 4, sound["octave"] + 1), 0.3)
        audio.add(track, audio.pan(note, 0.3 * (k - 1)), 0.15 + 0.18 * k, **kw)

    mixed = np.interp(np.arange(len(track)) / sr, times, diagnostics[:, 0])
    noise = rng.standard_normal((len(track), 2))
    wind = audio.lowpass(
        audio.highpass(noise, 300.0, circular=False), 1800.0, circular=False
    )
    wind *= 0.05 * (mixed / max(mixed.max(), 1e-9))[:, None] / np.abs(wind).max()
    track += wind
    logger.info(
        f"Soundtrack: {sound_name}, drone root MIDI {root}, {pairings} pairings"
    )
    track = audio.reverb(track, rng, rt60=3.2, wet=0.35, tone=4000.0, circular=False)
    audio.fade(track, 0.0, 0.5)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Kelvin-Helmholtz instability of a periodic double shear layer.

    The 2D incompressible Navier-Stokes equations in vorticity form,
        d(omega)/dt + u . grad(omega) = nu laplacian(omega),  laplacian(psi) = -omega,
    with u = dpsi/dy, v = -dpsi/dx, plus a passive dye c carried by the same velocity,
    are solved pseudo-spectrally on a 432 x 768 periodic grid (2/3 dealiasing, RK4 with an
    exact integrating factor for viscosity and a grid-scale hyperviscosity, fixed dt).
    The base flow is u = -1, +1, -1 in three horizontal bands (tanh profiles of half-width
    `thickness`), seeded with a ripple at `billows` rolls per interface plus its
    subharmonic, so the rolls later pair. Random phases and a little broadband noise come
    from settings.rng.

    Frame f shows the state at simulation time sim_time(f / (FRAME_COUNT - 1)): a
    one-shot that starts after a short pre-roll (ripples already visible), runs at a
    constant rate and eases to a stop for the last second, then fades. The step is fixed,
    so an RP_FPS preview follows exactly the same trajectory.

    Rendering: the dye field is bicubically upsampled to 1080 x 1920 and coloured
    through a diverging indigo / cream / ember LUT; the cream mixed zone gets a wide
    GPU bloom. The simulation is stateful, so frames render in order on the GPU.

    Sound (see `soundtrack`): each frame also records the mixed fraction and the roll
    count of each layer (diagnostics.npy, reused by --audio-only runs). Tags in SOUNDS
    ("drone", "strings", "flute") pick the voices; settings.audio_rng the key. The
    posted soundtrack is "strings" with --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    knobs = KNOBS
    device = select_device()
    flow = ShearFlow(knobs, settings.rng, device)
    lut = torch.tensor(palette_lut(PALETTE, 1024), device=device)
    targets = [
        round(sim_time(f / (FRAME_COUNT - 1), knobs) / DT) for f in range(FRAME_COUNT)
    ]
    logger.info(f"{targets[-1]} solver steps for {FRAME_COUNT} frames")
    fade_frames = 0.4 * FPS
    diagnostics = np.zeros((FRAME_COUNT, 3))

    def render(frame):
        flow.advance_to(targets[frame])
        diagnostics[frame] = flow.measure()
        dye = flow.fields()[1]
        big = F.interpolate(
            dye[None, None], size=(HEIGHT, WIDTH), mode="bicubic", align_corners=False
        )[0, 0]
        index = ((big.clamp(-1, 1) + 1) * 0.5 * (len(lut) - 1)).long()
        image = lut[index]
        glow = image * torch.exp(-((big / 0.35) ** 2))[..., None]
        image = image + 0.25 * wide_blur(glow, 18.0, 4)
        fade = min(1.0, (FRAME_COUNT - 1 - frame) / fade_frames)
        return image.clamp(0, 1) * (255.0 * fade)

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    saved = settings.output_path / settings.filename / "diagnostics.npy"
    if settings.audio_only and saved.exists():
        diagnostics = np.load(saved)
    elif settings.audio_only:  # frames from before the soundtrack: re-run the solver
        for frame in range(FRAME_COUNT):
            flow.advance_to(targets[frame])
            diagnostics[frame] = flow.measure()
    np.save(saved, diagnostics)
    tag = settings.audio_tag
    track = soundtrack(tag if tag in SOUNDS else SOUND, diagnostics, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
