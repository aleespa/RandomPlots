"""
Quantum Revival

Two quantum particles start out sitting still in a box. Within a moment
each one smears into a wave that fills the box, bouncing off the walls
and interfering with itself into a shifting web of ripples. Colour is
the phase of the wave; the lines are its wavefronts, and the dark points
where they all meet are tiny whirlpools in the phase.

Yet the chaos is only apparent. The allowed energies in a box grow like
the squares of whole numbers, n², so all the waves fall back into step
at the same moment and the particles reassemble exactly. On the way they
pass through fractions of that time where each one appears as a neat set
of copies: four at a quarter, three at a third, a mirror image at half.
This is the quantum Talbot effect. The clock slows near those moments.

Listen for the same order in the sound. Every note is a harmonic of one
low tone. While the wave is scrambled, an uneasy chord with no home note
swells; whenever the copies appear, one bell rings for each copy. The
tension builds until the particles reassemble and the full chord lands.

iħ ∂ψ/∂t = −(ħ²/2m) ∇²ψ,  Eₙ ∝ n²

Made with Python, PyTorch & NumPy. Sound synthesized in NumPy.
#generativeart #quantummechanics #physics #talboteffect #wavefunction
#randomplots #creativecoding #mathart #mathisbeautiful #sonification #animation
"""

import gc

import numpy as np
import torch
from loguru import logger

from colors.palettes import HOLOGRAPHIC
from common import audio
from common.gpu import blur, gaussian_kernel, select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds: exactly one full revival
FRAME_COUNT = frame_count(DURATION)

# --- Box ---------------------------------------------------------------------
BOX_WIDTH = WIDTH  # pixels; the box is √3 times taller than wide, so E ∝ 3m² + n²
BOX_HEIGHT = int(round(BOX_WIDTH * np.sqrt(3.0)))  # 1871: fits the 1920 frame
X_TURNS = 3  # the short side revives this many times per loop, the long side once
MODES = 64  # sine modes kept per axis

# --- Packets -----------------------------------------------------------------
PACKETS = 2
SIGMA = 0.09  # spread of |ψ|², as a fraction of the box width
START_X = (0.2, 0.34)  # start position ranges, fractions of the box (seed mirrors)
START_Y = (0.12, 0.22)

# --- Clock -------------------------------------------------------------------
# Physical times (fractions of the revival) where the clock slows down: the
# packet itself, the 2 x 2, 1 x 3 and mirrored fractional revivals.
EVENTS = (0.0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4)
DWELL = 0.05  # share of the video spent around each event
DWELL_WIDTH = 0.009  # physical width of each slow-down

# --- Appearance --------------------------------------------------------------
HUE_LUT = palette_lut(HOLOGRAPHIC + HOLOGRAPHIC[:1], 1024)  # cyclic phase wheel
HUE_DRIFT = 2  # whole turns the phase wheel drifts per loop, relative to <E>
BACKGROUND = (6, 6, 10)  # RGB
SUBFRAMES = 6  # shutter samples per frame: fast stretches blur, slow ones stay crisp
SHUTTER = 1.0  # exposure, as a fraction of the frame interval
DENSITY_SCALE = 1.6  # |ψ|² (1 = spread evenly) mapped to half brightness
FILL = 0.4  # brightness of the colour wash under the wavefronts
WAVEFRONTS = 16  # lines of constant phase per turn of the phase wheel
LINE_WIDTH = 2.2  # pixels
LINE_SCALE = 0.5  # |ψ|² at which wavefronts reach half strength
LINE_WHITE = 0.4  # wavefronts are their hue mixed this far toward white
CORE = (8.0, 60.0)  # |ψ|² range over which colour burns to white (log scale)
BLOOM = 0.4
BLOOM_SIGMA = 5.0
WIDE_BLOOM = 0.18
WIDE_BLOOM_SIGMA = 40.0

# --- Sound (rise & resolve, as a loop) ---------------------------------------
# At tau = p/q the bells are harmonics q, 2q, 3q, ... of one base note, one per
# copy of the packet; the revival is the full harmonic chord.
FUNDAMENTALS = range(40, 46)  # base note (MIDI), chosen by the audio seed
EVENT_PARTIALS = {
    0.0: (4, 5, 6, 8),  # the packet itself: a major chord, rolled up
    1 / 4: (4, 8, 12, 16),  # 2 x 2 copies: four multiples of 4
    1 / 3: (6, 9, 12),  # 1 x 3 copies: three multiples of 3
    1 / 2: (8, 6, 5, 4),  # the mirror image: the opening chord, reversed
    2 / 3: (12, 9, 6),
    3 / 4: (16, 12, 8, 4),
}
REVIVAL_BASS = (1, 2)  # partials under the full revival
ROLL = 0.085  # seconds between the bells of one event
ROLL_LEAD = 0.12  # the roll starts this long before the event's exact moment
CHAOS_MODE = "whole_tone"  # the scrambled stretches: no home note
CHAOS_CHORD = (0, 1, 2, 4)  # scale degrees of the uneasy pad, above the base
CHAOS_SPARKS = 4.0  # stray high notes per second at full chaos
RISER = 2.6  # seconds of build-up into the revival at the loop point
IPR_GRID = (96, 166)  # coarse (x, y) samples for measuring how spread psi is
IPR_TIMES = 900  # video times at which it is measured
REVERB_RT60 = 3.2
REVERB_WET = 0.38


def sine_coefficients(centre, sigma, length):
    """
    Coefficients of a Gaussian packet at rest in the sine modes sin(mπu) of
    [0, 1], for m = 1 .. MODES. `centre` and `sigma` are in units of the side,
    whose physical `length` only matters for normalisation.
    """
    u = (np.arange(8192) + 0.5) / 8192
    packet = np.exp(-((u - centre) ** 2) / (4.0 * sigma**2))
    m = np.arange(1, MODES + 1)
    modes = np.sqrt(2.0) * np.sin(np.pi * np.outer(u, m))
    c = modes.T @ packet / len(u)
    return c / np.linalg.norm(c) / np.sqrt(length)


def make_clock():
    """
    phys(t): video time t in [0, 1) -> physical time in [0, 1), in revivals.
    The clock runs slow near EVENTS and fast in between; it's monotone, and
    phys(t + 1) = phys(t) + 1, so the loop is still exact.
    """
    p = np.linspace(0.0, 1.0, 200001)
    events = np.asarray(EVENTS)[:, None]
    bumps = sum(
        np.exp(-(((p - events - shift) / DWELL_WIDTH) ** 2)).sum(axis=0)
        for shift in (-1, 0, 1)
    )
    area = DWELL / (1.0 - len(EVENTS) * DWELL)  # each bump's share of the total
    density = 1.0 + area / (DWELL_WIDTH * np.sqrt(np.pi)) * bumps
    video = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(p))]
    )
    video /= video[-1]

    def phys(t):
        turns = np.floor(t)
        return float(turns + np.interp(t - turns, video, p))

    return phys


def video_time(phys, tau):
    """The video time t in [0, 1) at which the clock reads physical time tau."""
    lo, hi = 0.0, 1.0
    for _ in range(48):  # phys is monotone: bisect
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if phys(mid) < tau else (lo, mid)
    return lo


def chaos_curve(cx, cy, energy_x, energy_y, phys):
    """
    How scrambled the wave is at IPR_TIMES evenly spaced video times: 0 when
    |ψ|² is gathered into the packets (or their copies), 1 when it's spread
    over the box. Measured by the inverse participation ratio Σ|ψ|⁴ / (Σ|ψ|²)²
    on a coarse grid, on a log scale between its extremes over the loop.
    """
    m = np.arange(1, MODES + 1)
    nx, ny = IPR_GRID
    basis_x = np.sin(np.pi * np.outer((np.arange(nx) + 0.5) / nx, m))
    basis_y = np.sin(np.pi * np.outer((np.arange(ny) + 0.5) / ny, m))
    ipr = np.empty(IPR_TIMES)
    for i in range(IPR_TIMES):
        tau = phys(i / IPR_TIMES)
        fx = basis_x @ (cx * np.exp(-2j * np.pi * np.mod(energy_x * tau, 1.0))[:, None])
        gy = basis_y @ (cy * np.exp(-2j * np.pi * np.mod(energy_y * tau, 1.0))[:, None])
        density = np.abs(gy @ fx.T) ** 2
        ipr[i] = (density**2).sum() / density.sum() ** 2
    order = np.log(ipr / ipr.min()) / np.log(ipr.max() / ipr.min())
    # Smooth over ~0.1 s, circularly: the curve loops like everything else.
    width = 0.1 / DURATION * IPR_TIMES
    k = np.fft.rfftfreq(IPR_TIMES) * IPR_TIMES
    kernel = np.exp(-0.5 * (2 * np.pi * k * width / IPR_TIMES) ** 2)
    return np.clip(1.0 - np.fft.irfft(np.fft.rfft(order) * kernel, IPR_TIMES), 0, 1)


def soundtrack(chaos, phys, rng):
    """
    Rise and resolve, closed into a loop: tension while the wave is scrambled,
    harmony whenever it reassembles.

    Every note is a harmonic of one seeded base note f0. At each slow-down
    of the clock (physical time p/q) bells ring at multiples of q, one per copy
    of the packet, rolled across the stereo field: four at 1/4 and 3/4, three
    at 1/3 and 2/3, and at 1/2 the opening chord in reverse, like the mirrored
    packet. In between, a whole-tone pad (no home note) swells and brightens
    with `chaos` (crossfading a dark and a bright voicing) and stray high plucks
    fall at up to CHAOS_SPARKS per second. A riser builds over the last RISER
    seconds and stops dead at the loop point, where the full revival lands:
    the rolled major chord over bass harmonics 1 and 2. Notes, pad, riser and
    reverb are all circular, so the sound loops with the picture.
    """
    track = audio.silence(DURATION)
    n = len(track)
    root = int(rng.choice(FUNDAMENTALS))
    f0 = float(audio.midi_to_hz(root))

    # Harmony: the events.
    moments = {}
    for tau, partials in EVENT_PARTIALS.items():
        when = video_time(phys, tau) * DURATION
        moments[tau] = when
        positions = np.linspace(-0.6, 0.6, len(partials))
        velocity = 0.6 if tau == 0.0 else 0.45
        for i, (k, position) in enumerate(zip(partials, positions)):
            bell = audio.bell(f0 * k, velocity)
            audio.add(track, audio.pan(bell, position), when - ROLL_LEAD + i * ROLL)
    for k in REVIVAL_BASS:
        audio.add(track, audio.marimba(f0 * k, 0.42, length=3.0), 0.0)

    # Tension: the pad and the sparks, following the chaos curve.
    t = np.arange(n) / n
    level = np.interp(
        t, np.arange(IPR_TIMES + 1) / IPR_TIMES, np.append(chaos, chaos[0])
    )
    scale = Scale(root + 12, CHAOS_MODE)
    dark, bright = audio.silence(DURATION), audio.silence(DURATION)
    for degree in CHAOS_CHORD:
        f = scale.hz(degree)
        audio.add(dark, audio.pad(f, DURATION, 0.5, brightness=0.15), 0.0)
        audio.add(bright, audio.pad(f, DURATION, 0.5, brightness=0.85), 0.0)
    track += (level**1.5)[:, None] * (dark + (bright - dark) * level[:, None])

    candidates = rng.uniform(0.0, 1.0, rng.poisson(CHAOS_SPARKS * DURATION))
    keep = rng.uniform(size=len(candidates)) < np.interp(candidates, t, level) ** 2
    for when in candidates[keep]:
        f = scale.hz(int(rng.integers(6, 15)))
        pluck = audio.pluck(f, rng.uniform(0.15, 0.3), length=1.2)
        audio.add(track, audio.pan(pluck, rng.uniform(-0.8, 0.8)), when * DURATION)

    # The build-up, cut off by the revival at the loop point.
    audio.add(track, audio.riser(RISER, rng, 150.0, 1500.0, 0.6), DURATION - RISER)

    logger.info(
        f"Soundtrack: base MIDI {root} ({f0:.0f} Hz), {keep.sum()} sparks, events at "
        + ", ".join(f"{s:.2f} s" for s in sorted(moments.values()))
    )
    track = audio.reverb(track, rng, rt60=REVERB_RT60, wet=REVERB_WET)
    return audio.master(track)


def generate(settings: ImageProcessingSettings = None):
    """
    A Gaussian wave packet in a 2D infinite square well, evolved exactly.

    The box is 1 x √3, so eigenstates sin(mπx) sin(nπy/√3) have energies
    ∝ 3m² + n². With time measured in full revivals, every mode's phase is
    exp(-2πi (3m² + n²) τ): all integers, so ψ(τ + 1) = ψ(τ) exactly (the short
    side revives at τ = 1/3, 2/3 as well). A Gaussian is separable, so each
    packet is F(x, τ) G(y, τ), two 1D sums of MODES sines, and the frame is one
    small complex matrix product on the GPU. At rational τ = p/q the packet
    returns as a few copies (fractional revivals / quantum Talbot effect);
    since a packet spreads in ~σ² of a revival, the video clock τ(t) slows near
    EVENTS and hurries between them, a smooth monotone warp with τ(t+1) = τ(t)+1.

    Brightness is |ψ|² through a soft tone map, burning to white where the
    density concentrates; hue is arg ψ on a cyclic palette, with a whole-number
    reference energy removed so the colours drift slowly. Plus a two-scale
    bloom. Each frame averages SUBFRAMES shutter samples, so the fast stretches of
    the clock blur like a long exposure. Frames render in order on the GPU.

    After the frames, the soundtrack (see `soundtrack`) is built from the same
    coefficients and clock, with settings.audio_rng, and muxed into the MP4.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    # Packets: start positions, isotropic in pixels (σ_y = σ_x / √3 in box units).
    aspect = BOX_HEIGHT / BOX_WIDTH
    cx, cy = [], []
    for _ in range(PACKETS):
        x = rng.uniform(*START_X)
        y = rng.uniform(*START_Y)
        x = x if rng.random() < 0.5 else 1.0 - x
        y = y if rng.random() < 0.5 else 1.0 - y
        cx.append(sine_coefficients(x, SIGMA, BOX_WIDTH))
        cy.append(sine_coefficients(y, SIGMA / aspect, BOX_HEIGHT))
    cx = np.stack(cx, axis=1) / np.sqrt(PACKETS)  # (MODES, PACKETS)
    cy = np.stack(cy, axis=1)

    m = np.arange(1, MODES + 1)
    energy_x = X_TURNS * m**2
    energy_y = m**2
    mean_energy = (np.abs(cx) ** 2 * energy_x[:, None]).sum() / (
        np.abs(cx) ** 2
    ).sum() + (np.abs(cy) ** 2 * energy_y[:, None]).sum() / (np.abs(cy) ** 2).sum()
    reference = int(round(mean_energy)) - HUE_DRIFT
    hue_offset = rng.uniform()

    def basis(size):
        """Sine modes on `size` pixel centres, and their derivative per pixel."""
        u = (np.arange(size) + 0.5) / size
        modes = np.sqrt(2.0) * np.sin(np.pi * np.outer(u, m))
        slopes = np.sqrt(2.0) * np.pi * m * np.cos(np.pi * np.outer(u, m)) / size
        return (
            torch.tensor(modes, dtype=torch.complex64, device=device),
            torch.tensor(slopes, dtype=torch.complex64, device=device),
        )

    basis_x, slope_x = basis(BOX_WIDTH)
    basis_y, slope_y = basis(BOX_HEIGHT)
    cx_t = torch.tensor(cx, dtype=torch.complex64, device=device)
    cy_t = torch.tensor(cy, dtype=torch.complex64, device=device)
    area = BOX_WIDTH * BOX_HEIGHT

    lut = torch.tensor(HUE_LUT, device=device)
    canvas0 = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    canvas0[:] = torch.tensor(BACKGROUND, dtype=torch.float32, device=device)
    x0 = (WIDTH - BOX_WIDTH) // 2
    y0 = (HEIGHT - BOX_HEIGHT) // 2
    kernel = gaussian_kernel(BLOOM_SIGMA, device)
    phys = make_clock()
    shutter = (np.arange(SUBFRAMES) + 0.5) / SUBFRAMES - 0.5

    def phases(energy, tau):
        turns = np.mod(energy * tau, 1.0)  # float64 before the GPU sees it
        return torch.tensor(np.exp(-2j * np.pi * turns), device=device).to(
            torch.complex64
        )

    def exposure(tau):
        """The box at physical time tau: (colour in [0, 1], brightness)."""
        ax = cx_t * phases(energy_x, tau)[:, None]
        ay = cy_t * phases(energy_y, tau)[:, None]
        fx, dfx = basis_x @ ax, slope_x @ ax  # (W, PACKETS)
        gy, dgy = basis_y @ ay, slope_y @ ay  # (H, PACKETS)
        psi = gy @ fx.T  # (H, W)
        amp2 = psi.real**2 + psi.imag**2
        density = amp2 * area  # 1 where the probability is spread evenly

        # Phase in turns, and its gradient in radians per pixel: Im(ψ* ∇ψ) / |ψ|².
        turns = torch.angle(psi) / (2 * np.pi) + reference * tau + hue_offset
        conj = psi.conj()
        safe = amp2.clamp_min(1e-12)
        kx = (conj * (gy @ dfx.T)).imag / safe
        ky = (conj * (dgy @ fx.T)).imag / safe
        wavenumber = torch.sqrt(kx**2 + ky**2).clamp_min(1e-6)

        # Wavefronts: distance in pixels to the nearest line of constant phase.
        level = turns * WAVEFRONTS
        nearest = torch.round(level)
        distance = (level - nearest).abs() * (2 * np.pi / WAVEFRONTS) / wavenumber
        line = (1.0 - distance / LINE_WIDTH).clamp(0.0, 1.0)
        line = line * (1.0 - torch.exp(-density / LINE_SCALE * np.log(2.0)))

        size = len(HUE_LUT) - 1
        wash = lut[(torch.remainder(turns, 1.0) * size).long()]
        ink = lut[(torch.remainder(nearest / WAVEFRONTS, 1.0) * size).long()]
        ink = ink + (1.0 - ink) * LINE_WHITE

        brightness = 1.0 - torch.exp(-density / DENSITY_SCALE * np.log(2.0))
        colour = wash * (FILL * brightness)[..., None]
        colour = colour + (ink - colour) * line[..., None]
        core = torch.log(density.clamp_min(1e-6) / CORE[0]) / np.log(CORE[1] / CORE[0])
        core = core.clamp(0.0, 1.0)
        core = core * core * (3 - 2 * core)
        colour = colour + (1.0 - colour) * core[..., None]
        return colour, torch.maximum(brightness, line)

    def render(frame):
        field = torch.zeros(BOX_HEIGHT, BOX_WIDTH, 3, device=device)
        glow = torch.zeros(HEIGHT, WIDTH, 3, device=device)
        for offset in shutter:
            colour, brightness = exposure(
                phys((frame + SHUTTER * offset) / FRAME_COUNT)
            )
            field += colour
            glow[y0 : y0 + BOX_HEIGHT, x0 : x0 + BOX_WIDTH] += (
                colour * brightness[..., None]
            )
        field *= 255.0 / SUBFRAMES
        glow *= 255.0 / SUBFRAMES

        canvas = canvas0.clone()
        canvas[y0 : y0 + BOX_HEIGHT, x0 : x0 + BOX_WIDTH] += field
        canvas += BLOOM * blur(glow, kernel) + WIDE_BLOOM * wide_blur(
            glow, WIDE_BLOOM_SIGMA, 4
        )
        return canvas

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    chaos = chaos_curve(cx, cy, energy_x, energy_y, phys)
    track = soundtrack(chaos, phys, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
